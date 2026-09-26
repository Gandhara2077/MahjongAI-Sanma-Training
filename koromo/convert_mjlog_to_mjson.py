#!/usr/bin/env python3
"""Convert Tenhou yonma/sanma mjlog XML files to gzip-compressed mjai JSON Lines.

The meld bit-field decoder follows Tenhou's tehai.js algorithm as documented by
the MIT-licensed tenhou-log-utils/mjlog2mjai implementations (Copyright 2017
moto): https://github.com/fstqwq/mjlog2mjai
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable
from urllib.parse import unquote


CONVERT_VARIANTS = {
    "yonma": {
        "players": 4,
        "allow_chi": True,
        "allow_nukidora": False,
        # Yonma suppresses the fourth accepted reach (abortive draw reach4).
        # This flag is intentionally independent of the player count: sanma
        # must emit every reach_accepted (do NOT generalize to count < players).
        "suppress_final_reach_accepted": True,
    },
    "sanma": {
        "players": 3,
        "allow_chi": False,
        "allow_nukidora": True,
        "suppress_final_reach_accepted": False,
    },
}
DRAW_SEATS = {"T": 0, "U": 1, "V": 2, "W": 3}
DISCARD_SEATS = {"D": 0, "E": 1, "F": 2, "G": 3}
WINDS = ["E", "S", "W", "N"]
HONORS = ["E", "S", "W", "N", "P", "F", "C"]
RED_TILES = {16: "5mr", 52: "5pr", 88: "5sr"}
PLAYER_WORDS = {3: "three", 4: "four"}


def tile_to_mjai(tile_id: int, *, aka: bool = True) -> str:
    """Convert a Tenhou 136-tile ID to an mjai tile string."""
    if not 0 <= tile_id < 136:
        raise ValueError(f"invalid Tenhou tile ID: {tile_id}")
    if aka and tile_id in RED_TILES:
        return RED_TILES[tile_id]
    tile_type = tile_id // 4
    if tile_type < 27:
        suit = "mps"[tile_type // 9]
        return f"{tile_type % 9 + 1}{suit}"
    return HONORS[tile_type - 27]


def decode_meld(meld: int) -> dict[str, object]:
    """Decode Tenhou's meld bit field into called and consumed tile IDs."""
    relative_target = meld & 0x3
    if meld & 0x4:  # chi
        value = (meld & 0xFC00) >> 10
        called_index = value % 3
        value //= 3
        base = (9 * (value // 7) + value % 7) * 4
        offsets = [
            (meld & 0x18) >> 3,
            (meld & 0x60) >> 5,
            (meld & 0x180) >> 7,
        ]
        tiles = [base + offsets[0], base + 4 + offsets[1], base + 8 + offsets[2]]
        return {
            "kind": "chi",
            "called": tiles[called_index],
            "consumed": [tile for index, tile in enumerate(tiles) if index != called_index],
            "relative_target": relative_target,
        }
    if meld & 0x8:  # pon
        unused = (meld & 0x60) >> 5
        value = (meld & 0xFE00) >> 9
        called_index = value % 3
        base = (value // 3) * 4
        tiles = [base + copy for copy in range(4) if copy != unused]
        return {
            "kind": "pon",
            "called": tiles[called_index],
            "consumed": [tile for index, tile in enumerate(tiles) if index != called_index],
            "relative_target": relative_target,
        }
    if meld & 0x10:  # added kan
        added_copy = (meld & 0x60) >> 5
        value = (meld & 0xFE00) >> 9
        base = (value // 3) * 4
        return {
            "kind": "kakan",
            "called": base + added_copy,
            "consumed": [base + copy for copy in range(4) if copy != added_copy],
            "relative_target": relative_target,
        }
    if meld & 0x20:  # sanma North extraction
        return {"kind": "nukidora"}

    called = (meld & 0xFF00) >> 8
    base = (called // 4) * 4
    if relative_target == 0:
        return {
            "kind": "ankan",
            "consumed": [base, base + 1, base + 2, base + 3],
            "relative_target": 0,
        }
    return {
        "kind": "daiminkan",
        "called": called,
        "consumed": [tile for tile in range(base, base + 4) if tile != called],
        "relative_target": relative_target,
    }


def _int_list(value: str | None) -> list[int]:
    return [int(item) for item in value.split(",")] if value else []


def _sorted_tiles(tiles: list[int]) -> list[int]:
    return sorted(tiles, key=lambda tile: tile ^ 3)


def _translated_tiles(tiles: list[int], aka: bool) -> list[str]:
    return [tile_to_mjai(tile, aka=aka) for tile in _sorted_tiles(tiles)]


def _score_deltas(attributes: dict[str, str], players: int) -> list[int]:
    scores = _int_list(attributes.get("sc"))
    deltas = [scores[index] * 100 for index in range(1, len(scores), 2)]
    return (deltas + [0] * players)[:players]


def convert_xml(
    xml: str | bytes, variant: dict = CONVERT_VARIANTS["yonma"]
) -> list[dict[str, object]]:
    """Convert one mjlog XML document to native mjai events for the variant."""
    players = variant["players"]
    player_word = PLAYER_WORDS[players]
    if isinstance(xml, bytes):
        xml = xml.decode("utf-8-sig")
    try:
        root = ET.fromstring(xml.lstrip("\ufeff"))
    except ET.ParseError as error:
        raise ValueError(f"invalid mjlog XML: {error}") from error
    if root.tag != "mjloggm":
        raise ValueError(f"expected mjloggm root, got {root.tag!r}")

    go = root.find("GO")
    if go is None or "type" not in go.attrib:
        raise ValueError("mjlog is missing GO type")
    game_type = int(go.attrib["type"])
    actual_players = 3 if game_type & 0x10 else 4
    if actual_players != players:
        raise ValueError(
            f"expected a {players}-player log, got {actual_players}-player"
        )
    aka = not bool(game_type & 0x02)
    hanchan = bool(game_type & 0x08)

    names = [""] * players
    for node in root.findall("UN"):
        for player in range(players):
            encoded = node.attrib.get(f"n{player}")
            if encoded is not None and not names[player]:
                names[player] = unquote(encoded)

    events: list[dict[str, object]] = [
        {
            "type": "start_game",
            "names": names,
            "kyoku_first": 0 if hanchan else 4,
            "aka_flag": aka,
        }
    ]
    round_open = False
    pending_agari: list[dict[str, str]] = []
    pending_ryuukyoku: dict[str, str] | None = None
    last_draw: list[int | None] = [None] * players
    last_discard_actor: int | None = None
    pending_kan_indices: list[int] = []
    reach_accepted_count = 0

    def flush_round() -> None:
        nonlocal pending_agari, pending_ryuukyoku, round_open
        if not round_open:
            return
        if pending_agari:
            ura_ids = max(
                (_int_list(item.get("doraHaiUra")) for item in pending_agari),
                key=len,
                default=[],
            )
            ura_markers = [tile_to_mjai(tile, aka=aka) for tile in ura_ids]
            for item in pending_agari:
                events.append(
                    {
                        "type": "hora",
                        "actor": int(item["who"]),
                        "target": int(item["fromWho"]),
                        "deltas": _score_deltas(item, players),
                        "ura_markers": list(ura_markers),
                    }
                )
        elif pending_ryuukyoku is not None:
            events.append(
                {
                    "type": "ryukyoku",
                    "deltas": _score_deltas(pending_ryuukyoku, players),
                }
            )
        else:
            raise ValueError("mjlog round has no AGARI or RYUUKYOKU result")
        events.append({"type": "end_kyoku"})
        pending_agari = []
        pending_ryuukyoku = None
        round_open = False

    for node in root:
        tag = node.tag
        attributes = node.attrib

        if tag == "INIT":
            flush_round()
            seed = _int_list(attributes.get("seed"))
            scores = _int_list(attributes.get("ten"))
            if len(seed) < 6 or len(scores) < players:
                raise ValueError("INIT is missing seed or scores")
            hands = [
                _int_list(attributes.get(f"hai{player}"))
                for player in range(players)
            ]
            if any(len(hand) != 13 for hand in hands):
                raise ValueError("INIT must contain 13 tiles for every player")
            wind_index = seed[0] // 4
            if not 0 <= wind_index < len(WINDS):
                raise ValueError(f"invalid round index in INIT: {seed[0]}")
            events.append(
                {
                    "type": "start_kyoku",
                    "bakaze": WINDS[wind_index],
                    "dora_marker": tile_to_mjai(seed[5], aka=aka),
                    "kyoku": seed[0] % 4 + 1,
                    "honba": seed[1],
                    "kyotaku": seed[2],
                    "oya": int(attributes["oya"]),
                    "scores": [score * 100 for score in scores[:players]],
                    "tehais": [_translated_tiles(hand, aka) for hand in hands],
                }
            )
            round_open = True
            last_draw = [None] * players
            last_discard_actor = None
            pending_kan_indices = []
            reach_accepted_count = 0
            continue

        if tag in {"SHUFFLE", "GO", "UN", "TAIKYOKU", "BYE"}:
            continue
        if not round_open:
            continue
        if tag == "AGARI":
            pending_agari.append(dict(attributes))
            continue
        if tag == "RYUUKYOKU":
            pending_ryuukyoku = dict(attributes)
            continue
        if tag == "REACH":
            actor = int(attributes["who"])
            if int(attributes["step"]) == 1:
                events.append({"type": "reach", "actor": actor})
            elif variant["suppress_final_reach_accepted"]:
                # Yonma branch: the fourth accepted reach declares the reach4
                # abortive draw, so its reach_accepted is suppressed. Kept as
                # an explicit variant branch; sanma must not inherit this.
                reach_accepted_count += 1
                if reach_accepted_count < players:
                    events.append({"type": "reach_accepted", "actor": actor})
            else:
                # Sanma branch: every accepted reach is emitted unconditionally.
                events.append({"type": "reach_accepted", "actor": actor})
            continue
        if tag == "DORA":
            dora_event = {
                "type": "dora",
                "dora_marker": tile_to_mjai(int(attributes["hai"]), aka=aka),
            }
            if pending_kan_indices:
                kan_index = pending_kan_indices.pop(0)
                offset = 2
                probe = kan_index + 2
                if pending_kan_indices and probe < len(events) and events[probe]["type"] == "kakan":
                    offset = 3
                events.insert(kan_index + offset, dora_event)
                pending_kan_indices = [index + 1 for index in pending_kan_indices]
            else:
                events.append(dora_event)
            continue
        if tag == "N":
            actor = int(attributes["who"])
            decoded = decode_meld(int(attributes["m"]))
            kind = str(decoded["kind"])
            last_draw[actor] = None
            if kind == "chi" and not variant["allow_chi"]:
                raise ValueError(f"chi is invalid in a {player_word}-player log")
            if kind == "nukidora" and not variant["allow_nukidora"]:
                raise ValueError(
                    f"nukidora is invalid in a {player_word}-player log"
                )
            if kind in {"chi", "pon", "daiminkan"}:
                if last_discard_actor is None:
                    raise ValueError(f"{kind} appears without a preceding discard")
                events.append(
                    {
                        "type": kind,
                        "actor": actor,
                        "target": last_discard_actor,
                        "pai": tile_to_mjai(int(decoded["called"]), aka=aka),
                        "consumed": _translated_tiles(list(decoded["consumed"]), aka),
                    }
                )
            elif kind == "kakan":
                events.append(
                    {
                        "type": "kakan",
                        "actor": actor,
                        "pai": tile_to_mjai(int(decoded["called"]), aka=aka),
                        "consumed": _translated_tiles(list(decoded["consumed"]), aka),
                    }
                )
                pending_kan_indices.append(len(events) - 1)
            elif kind == "ankan":
                events.append(
                    {
                        "type": "ankan",
                        "actor": actor,
                        "consumed": _translated_tiles(list(decoded["consumed"]), aka),
                    }
                )
                pending_kan_indices.append(len(events) - 1)
            elif kind == "nukidora":
                events.append({"type": "nukidora", "actor": actor, "pai": "N"})
            else:
                raise ValueError(f"unsupported meld type: {kind}")
            continue

        match = re.fullmatch(r"([TUVW])(\d+)", tag)
        if match:
            actor = DRAW_SEATS[match.group(1)]
            if actor >= players:
                raise ValueError(f"invalid draw actor {actor} in {player_word}-player log")
            tile = int(match.group(2))
            events.append(
                {"type": "tsumo", "actor": actor, "pai": tile_to_mjai(tile, aka=aka)}
            )
            last_draw[actor] = tile
            continue
        match = re.fullmatch(r"([DEFG])(\d+)", tag)
        if match:
            actor = DISCARD_SEATS[match.group(1)]
            if actor >= players:
                raise ValueError(
                    f"invalid discard actor {actor} in {player_word}-player log"
                )
            tile = int(match.group(2))
            events.append(
                {
                    "type": "dahai",
                    "actor": actor,
                    "pai": tile_to_mjai(tile, aka=aka),
                    "tsumogiri": last_draw[actor] == tile,
                }
            )
            last_draw[actor] = None
            last_discard_actor = actor
            continue
        raise ValueError(f"unsupported mjlog tag: {tag}")

    flush_round()
    if not any(event["type"] == "start_kyoku" for event in events):
        raise ValueError("mjlog contains no rounds")
    events.append({"type": "end_game"})
    return events


def is_complete_mjson(path: Path) -> bool:
    """Return whether a gzip mjson is valid JSONL with game boundary events."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            events = [json.loads(line) for line in file if line.strip()]
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        events
        and isinstance(events[0], dict)
        and isinstance(events[-1], dict)
        and events[0].get("type") == "start_game"
        and events[-1].get("type") == "end_game"
    )


def convert_file(
    source: Path,
    destination: Path,
    *,
    variant: dict = CONVERT_VARIANTS["yonma"],
    force: bool = False,
) -> bool:
    """Convert one mjlog; return False when an up-to-date output is skipped."""
    if (
        not force
        and destination.exists()
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
        and is_complete_mjson(destination)
    ):
        return False

    events = convert_xml(source.read_bytes(), variant=variant)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as raw:
            with gzip.GzipFile(
                filename=destination.name,
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                for event in events:
                    line = json.dumps(
                        event, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    compressed.write(line + b"\n")
        if not is_complete_mjson(partial):
            raise OSError(f"converted mjson failed validation: {source}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return True


def _convert_one(task: tuple) -> tuple[str, bool]:
    """Process-pool entry point: convert one file, return (relative, changed)."""
    source, input_root, output_root, variant, force = task
    relative = source.relative_to(input_root).with_suffix(".mjson")
    changed = convert_file(
        source, output_root / relative, variant=variant, force=force
    )
    return relative.as_posix(), changed


def convert_tree(
    input_root: Path,
    output_root: Path,
    *,
    variant: dict = CONVERT_VARIANTS["yonma"],
    force: bool = False,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Recursively convert .mjlog files while preserving relative layout."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"mjlog directory does not exist: {input_root}")
    sources = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mjlog"
    )
    converted = 0
    skipped = 0
    if workers > 1:
        tasks = [
            (source, input_root, output_root, variant, force)
            for source in sources
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, (relative, changed) in enumerate(
                executor.map(_convert_one, tasks, chunksize=32), start=1
            ):
                converted += 1 if changed else 0
                skipped += 0 if changed else 1
                if progress is not None and index % 500 == 0:
                    progress(f"[{index}/{len(sources)}] {relative}")
    else:
        for index, source in enumerate(sources, start=1):
            relative = source.relative_to(input_root).with_suffix(".mjson")
            destination = output_root / relative
            changed = convert_file(source, destination, variant=variant, force=force)
            if changed:
                converted += 1
            else:
                skipped += 1
            if progress is not None:
                action = "converted" if changed else "skipped"
                progress(f"[{index}/{len(sources)}] {action} {relative}")
    return {"found": len(sources), "converted": converted, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recursively convert Tenhou yonma/sanma .mjlog files to gzip .mjson."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(CONVERT_VARIANTS),
        required=True,
        help="game variant to convert (yonma = 四鳳南, sanma = 三鳳南)",
    )
    parser.add_argument("--input", type=Path, default=script_root / "mjlog")
    parser.add_argument("--output", type=Path, default=script_root / "mjson")
    parser.add_argument("--force", action="store_true", help="reconvert existing files")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel conversion processes (default: 1)",
    )
    parser.add_argument("--quiet", action="store_true", help="hide per-file progress")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    summary = convert_tree(
        args.input.resolve(),
        args.output.resolve(),
        variant=CONVERT_VARIANTS[args.variant],
        force=args.force,
        workers=args.workers,
        progress=None if args.quiet else print,
    )
    print(
        "Finished: "
        f"{summary['found']} found, {summary['converted']} converted, "
        f"{summary['skipped']} already current"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
