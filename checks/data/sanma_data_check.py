"""Validate a yonma or sanma corpus and write a reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from koromo.convert_mjlog_to_mjson import is_complete_mjson  # noqa: E402
from koromo.download_tenhou import (  # noqa: E402
    DOWNLOAD_VARIANTS,
    is_complete_variant_mjlog,
)


def tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_raw(path: Path, players: int) -> None:
    assert is_complete_variant_mjlog(path, players), f"invalid mjlog: {path}"
    root = ET.fromstring(path.read_bytes().decode("utf-8-sig"))
    game_type = int(root.find("GO").attrib["type"])
    assert bool(game_type & 0x08), f"not a hanchan: {path}"
    assert bool(game_type & 0x10) == (players == 3), f"wrong player count: {path}"


def validate_mjson(path: Path, players: int) -> None:
    assert is_complete_mjson(path), f"invalid gzip mjson: {path}"
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    assert len(events[0].get("names", [])) == players, f"wrong player count: {path}"
    assert events[0]["type"] == "start_game" and events[-1]["type"] == "end_game"
    for event in events:
        if players == 3:
            assert event.get("type") != "chi", f"chi in sanma log: {path}"
        for field in ("actor", "target"):
            if field in event:
                assert 0 <= event[field] < players, f"invalid {field}: {path}"
        for field in ("scores", "tehais", "deltas"):
            if field in event and event[field] is not None:
                assert len(event[field]) == players, f"invalid {field}: {path}"


def corpus_record(root: Path, paths: list[Path]) -> dict[str, object]:
    return {
        "root": root.relative_to(ROOT).as_posix(),
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "sha256": tree_digest(root, paths),
    }


def select_date_range(
    root: Path,
    paths: list[Path],
    start_date: str,
    end_date: str,
) -> list[Path]:
    """Keep files in the YYYY/MM/DD directory range, inclusive."""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except ValueError as error:
        raise ValueError("dates must use YYYYMMDD") from error
    if end < start:
        raise ValueError("end date must not be earlier than start date")

    selected = []
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 4:
            continue
        date_key = "".join(relative.parts[:3])
        try:
            date = datetime.strptime(date_key, "%Y%m%d")
        except ValueError:
            continue
        if start <= date <= end:
            selected.append(path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(DOWNLOAD_VARIANTS),
        default="sanma",
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mjson-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()

    players = DOWNLOAD_VARIANTS[args.variant]["players"]
    raw_root = args.raw_root.resolve()
    mjson_root = args.mjson_root.resolve()
    raw_paths = select_date_range(
        raw_root,
        sorted(raw_root.rglob("*.mjlog")),
        args.start_date,
        args.end_date,
    )
    mjson_paths = select_date_range(
        mjson_root,
        sorted(mjson_root.rglob("*.mjson")),
        args.start_date,
        args.end_date,
    )
    assert raw_paths, f"no raw logs under {raw_root}"
    assert mjson_paths, f"no converted logs under {mjson_root}"
    assert {path.relative_to(raw_root).with_suffix(".mjson") for path in raw_paths} == {
        path.relative_to(mjson_root) for path in mjson_paths
    }, "raw and converted file sets differ"

    for path in raw_paths:
        validate_raw(path, players)
    for path in mjson_paths:
        validate_mjson(path, players)

    manifest = {
        "variant": args.variant,
        "players": players,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "raw": corpus_record(raw_root, raw_paths),
        "mjson": corpus_record(mjson_root, mjson_paths),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"{args.variant.upper()}_DATA_OK "
        f"games={len(raw_paths)} raw_bytes={manifest['raw']['bytes']} "
        f"mjson_bytes={manifest['mjson']['bytes']}"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ET.ParseError, KeyError, ValueError) as error:
        print(f"DATA_CHECK_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
