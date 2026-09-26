"""Strict paired rank check for a three-way Sanma arena directory.

Usage:
    python checks/stats/three_way_paired_check.py <three-way-log-dir>
    python checks/stats/three_way_paired_check.py --self-check
"""

import argparse
import gzip
import json
import math
import sys
import tempfile
from pathlib import Path


CHECKS_DIR = Path(__file__).resolve().parent
if str(CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(CHECKS_DIR))

from arena_log_paired import FILE_PATTERN  # noqa: E402
from arena_log_results import game_result, rank_of  # noqa: E402


CANDIDATE_NAMES = (
    "sanma-128x8-champion-46k",
    "sanma-lr2-champion-40k",
    "sanma-192x12-best-92k",
)
ROTATIONS = ("a", "b", "c")


class CheckError(ValueError):
    """Raised when the input set cannot support a paired comparison."""


def paired_statistics(values):
    n = len(values)
    if n < 2:
        raise ValueError("at least two paired groups are required")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return {"n": n, "mean": mean, "low": mean - margin, "high": mean + margin}


def _required_int(data, key):
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckError(f"summary.json: {key} must be an integer")
    return value


def _read_summary(directory, required_games, required_seeds):
    path = directory / "summary.json"
    try:
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"cannot read summary.json: {exc}") from exc
    if not isinstance(summary, dict):
        raise CheckError("summary.json: top-level value must be an object")

    if _required_int(summary, "games") != required_games:
        raise CheckError(f"summary.json: games is not {required_games}")
    if _required_int(summary, "expected_log_count") != required_games:
        raise CheckError(f"summary.json: expected_log_count is not {required_games}")
    if "log_count" in summary and _required_int(summary, "log_count") != required_games:
        raise CheckError(f"summary.json: log_count is not {required_games}")

    games_per_iter = _required_int(summary, "games_per_iter")
    iters = _required_int(summary, "iters")
    seeds_per_iter = _required_int(summary, "seeds_per_iter")
    if games_per_iter * iters != required_games:
        raise CheckError("summary.json: games_per_iter * iters does not match games")
    if seeds_per_iter * iters != required_seeds:
        raise CheckError("summary.json: seeds_per_iter * iters does not match seed count")
    if required_games != required_seeds * 3:
        raise CheckError("three-way check requires exactly three games per seed")

    seed_start = _required_int(summary, "seed_start")
    seed_key = _required_int(summary, "seed_key")
    expected_seeds = set(range(seed_start, seed_start + required_seeds))

    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise CheckError("summary.json: candidates must contain exactly three entries")
    names = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
            raise CheckError("summary.json: every candidate must have a string name")
        names.append(candidate["name"])
    if len(set(names)) != 3 or set(names) != set(CANDIDATE_NAMES):
        raise CheckError(f"summary.json: unexpected candidates {names!r}")

    ranking_vectors = summary.get("ranking_vectors")
    if ranking_vectors is not None:
        if (
            not isinstance(ranking_vectors, list)
            or len(ranking_vectors) != 3
            or any(
                not isinstance(vector, list)
                or len(vector) != 3
                or any(isinstance(count, bool) or not isinstance(count, int) for count in vector)
                for vector in ranking_vectors
            )
        ):
            raise CheckError("summary.json: ranking_vectors must be three integer triples")

    return {
        "candidate_names": names,
        "expected_seeds": expected_seeds,
        "ranking_vectors": ranking_vectors,
        "seed_key": seed_key,
    }


def _valid_score_vector(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and (not isinstance(score, float) or math.isfinite(score))
            for score in value
        )
    )


def _read_events(path):
    events = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise CheckError(f"{path.name}: line {line_number} is not a JSON object")
                events.append(event)
    except CheckError:
        raise
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"bad log {path.name}: {exc}") from exc
    if not events:
        raise CheckError(f"bad log {path.name}: empty log")
    return events


def _read_game(path, seed, seed_key, candidate_names):
    events = _read_events(path)
    first = events[0]
    if first.get("type") != "start_game":
        raise CheckError(f"bad log {path.name}: first event is not start_game")

    names = first.get("names")
    if (
        not isinstance(names, list)
        or len(names) != 3
        or any(not isinstance(name, str) for name in names)
        or len(set(names)) != 3
        or set(names) != set(candidate_names)
    ):
        raise CheckError(f"bad log {path.name}: names do not contain the three candidates")
    if first.get("seed") != [seed, seed_key]:
        raise CheckError(f"bad log {path.name}: start_game seed does not match filename/summary")

    if events[-1].get("type") != "end_game":
        raise CheckError(f"bad log {path.name}: missing end_game")
    saw_kyoku = False
    in_kyoku = False
    settled = False
    for event in events:
        if event.get("type") == "start_kyoku":
            if in_kyoku:
                raise CheckError(f"bad log {path.name}: missing end_kyoku")
            if not _valid_score_vector(event.get("scores")):
                raise CheckError(f"bad log {path.name}: start_kyoku has invalid scores")
            saw_kyoku = True
            in_kyoku = True
            settled = False
        elif event.get("type") == "end_kyoku":
            if not in_kyoku or not settled:
                raise CheckError(f"bad log {path.name}: unsettled end_kyoku")
            in_kyoku = False
        elif event.get("type") == "end_game":
            if in_kyoku or event is not events[-1]:
                raise CheckError(f"bad log {path.name}: premature end_game")
        elif "deltas" in event:
            if not in_kyoku or not _valid_score_vector(event["deltas"]):
                raise CheckError(f"bad log {path.name}: invalid score deltas")
            if event.get("type") in ("hora", "ryukyoku", "ryuukyoku"):
                settled = True
    if not saw_kyoku:
        raise CheckError(f"bad log {path.name}: no start_kyoku event")

    try:
        result = game_result(events)
    except (TypeError, ValueError, IndexError) as exc:
        raise CheckError(f"bad log {path.name}: score replay failed: {exc}") from exc
    if result is None:
        raise CheckError(f"bad log {path.name}: score replay is incomplete")
    _, scores = result
    ranks = tuple(rank_of(index, scores) for index in range(3))
    return {"names": names, "ranks": ranks}


def _file_key(path, seed_key, seen):
    match = FILE_PATTERN.fullmatch(path.name)
    if not match:
        raise CheckError(f"bad log filename: {path.name}")
    seed, file_seed_key, rotation = match.groups()
    key = (int(seed), rotation)
    if key in seen:
        raise CheckError(f"duplicate log for seed {key[0]} rotation {key[1]}")
    seen.add(key)
    if int(file_seed_key) != seed_key:
        raise CheckError(f"bad log filename: {path.name} has unexpected seed key")
    return key


def analyze_directory(directory, required_games=3000, required_seeds=1000):
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckError(f"log directory does not exist: {directory}")
    summary = _read_summary(directory, required_games, required_seeds)
    files = sorted(directory.glob("*.json.gz"))
    seen = set()
    parsed = []
    for path in files:
        seed, rotation = _file_key(path, summary["seed_key"], seen)
        parsed.append((seed, rotation, path))
    if len(files) != required_games:
        raise CheckError(f"expected {required_games} compressed logs, found {len(files)}")

    by_seed = {}
    for seed, rotation, path in parsed:
        game = _read_game(path, seed, summary["seed_key"], summary["candidate_names"])
        by_seed.setdefault(seed, {})[rotation] = game

    actual_seeds = set(by_seed)
    missing = sorted(summary["expected_seeds"] - actual_seeds)
    unexpected = sorted(actual_seeds - summary["expected_seeds"])
    if missing or unexpected:
        raise CheckError(f"seed coverage mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    names = summary["candidate_names"]
    baseline_index = names.index(CANDIDATE_NAMES[0])
    candidate_index = names.index(CANDIDATE_NAMES[2])
    seed_means = []
    ranking_vectors = [[0, 0, 0] for _ in names]
    for seed in sorted(summary["expected_seeds"]):
        games = by_seed[seed]
        if set(games) != set(ROTATIONS):
            raise CheckError(
                f"seed {seed}: expected rotations a,b,c, found {sorted(games)}"
            )
        positions = {
            name: sorted(game["names"].index(name) for game in games.values())
            for name in names
        }
        for name, candidate_positions in positions.items():
            if candidate_positions != [0, 1, 2]:
                raise CheckError(
                    f"seed {seed}: candidate {name} does not occupy all three seats"
                )

        diffs = []
        for rotation in ROTATIONS:
            game = games[rotation]
            rank_by_name = {
                name: game["ranks"][seat] for seat, name in enumerate(game["names"])
            }
            diffs.append(
                rank_by_name[names[candidate_index]] - rank_by_name[names[baseline_index]]
            )
            for index, name in enumerate(names):
                ranking_vectors[index][rank_by_name[name] - 1] += 1
        seed_means.append(sum(diffs) / 3)

    if summary["ranking_vectors"] is not None and summary["ranking_vectors"] != ranking_vectors:
        raise CheckError(f"native/log ranking mismatch: {summary['ranking_vectors']} != {ranking_vectors}")
    stats = paired_statistics(seed_means)
    avg_ranks = [
        sum((rank + 1) * count for rank, count in enumerate(vector)) / len(parsed)
        for vector in ranking_vectors
    ]
    return {
        "candidate_names": names,
        "games": len(parsed),
        "seeds": len(seed_means),
        "rotations": len(ROTATIONS),
        "seed_key": summary["seed_key"],
        "seed_start": min(summary["expected_seeds"]),
        "seed_end": max(summary["expected_seeds"]),
        "ranking_vectors": ranking_vectors,
        "summary_ranking_vectors": summary["ranking_vectors"],
        "summary_ranking_vectors_match": (
            summary["ranking_vectors"] is None or summary["ranking_vectors"] == ranking_vectors
        ),
        "avg_ranks": avg_ranks,
        "seed_means": seed_means,
        "better_fraction": sum(value < 0 for value in seed_means) / len(seed_means),
        **stats,
    }


def _write_synthetic_case(directory, *, missing=False, duplicate=False, bad=False, truncated=False):
    candidates = list(CANDIDATE_NAMES)
    summary = {
        "games": 6,
        "expected_log_count": 6,
        "games_per_iter": 6,
        "iters": 1,
        "seed_start": 100,
        "seeds_per_iter": 2,
        "seed_key": 7,
        "candidates": [{"name": name} for name in candidates],
        "ranking_vectors": [[3, 0, 3], [3, 3, 0], [0, 3, 3]],
    }
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rotations = {
        "a": candidates,
        "b": [candidates[1], candidates[2], candidates[0]],
        "c": [candidates[2], candidates[0], candidates[1]],
    }
    scores = {
        100: {candidates[0]: 300, candidates[1]: 200, candidates[2]: 100},
        101: {candidates[0]: 100, candidates[1]: 300, candidates[2]: 200},
    }
    for seed, candidate_scores in scores.items():
        for rotation, names in rotations.items():
            if missing and seed == 101 and rotation == "c":
                continue
            events = [
                {"type": "start_game", "names": names, "seed": [seed, 7]},
                {
                    "type": "start_kyoku",
                    "scores": [candidate_scores[name] for name in names],
                },
            ]
            if not truncated:
                events.extend([{"type": "ryukyoku", "deltas": [0, 0, 0]},
                               {"type": "end_kyoku"}, {"type": "end_game"}])
            path = directory / f"{seed}_7_{rotation}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as file:
                for event in events:
                    file.write(json.dumps(event) + "\n")
    if duplicate:
        (directory / "100_8_a.json.gz").write_bytes(
            (directory / "100_7_a.json.gz").read_bytes()
        )
    if bad:
        (directory / "101_7_c.json.gz").write_bytes(b"not gzip")


def _expect_check_error(directory):
    try:
        analyze_directory(directory, required_games=6, required_seeds=2)
    except CheckError:
        return
    raise AssertionError("expected CheckError")


def _self_check():
    events = [
        {"type": "start_game", "names": list(CANDIDATE_NAMES)},
        {"type": "start_kyoku", "scores": [35000, 35500, 34500], "kyotaku": 0},
        {"type": "reach_accepted", "actor": 1},
        {"type": "hora", "deltas": [1000, 0, 0]},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]
    assert game_result(events)[1] == [36000, 34500, 34500]
    events[3] = {"type": "ryukyoku", "deltas": [0, 0, 0]}
    assert game_result(events)[1] == [36000, 34500, 34500]
    result = paired_statistics([2.0, -1.0])
    assert result["n"] == 2
    assert math.isclose(result["mean"], 0.5)
    assert math.isclose(result["low"], 0.5 - 1.96 * 1.5)
    assert math.isclose(result["high"], 0.5 + 1.96 * 1.5)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        valid = root / "valid"
        _write_synthetic_case(valid)
        result = analyze_directory(valid, required_games=6, required_seeds=2)
        assert result["games"] == 6
        assert result["seeds"] == 2
        assert result["seed_means"] == [2.0, -1.0]
        assert math.isclose(result["mean"], 0.5)
        for kind in ("missing", "duplicate", "bad", "truncated"):
            case = root / kind
            _write_synthetic_case(case, **{kind: True})
            _expect_check_error(case)
    print("THREE_WAY_PAIRED_CHECK_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if not args.directory:
        parser.error("a three-way log directory is required")
    try:
        result = analyze_directory(args.directory)
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    baseline, _, candidate = result["candidate_names"]
    print("summary candidates: " + ", ".join(result["candidate_names"]))
    print(
        f"validated logs: games={result['games']} seeds={result['seeds']} "
        f"rotations_per_seed={result['rotations']} seed_key={result['seed_key']} "
        f"seed_range={result['seed_start']}..{result['seed_end']}"
    )
    if not result["summary_ranking_vectors_match"]:
        print(
            f"  warning: summary ranking_vectors differ from log replay; "
            f"log={result['ranking_vectors']} summary={result['summary_ranking_vectors']}"
        )
    print(
        f"paired {candidate} - {baseline}: mean rank diff={result['mean']:+.6f} "
        f"(CI95 {result['low']:+.6f}~{result['high']:+.6f})"
    )
    print(
        f"  candidate better on {result['better_fraction']:.1%} of seeds; "
        "promotion criterion: CI95 upper bound < 0"
    )
    print(f"  verdict: {'ELIGIBLE' if result['high'] < 0 else 'NOT ELIGIBLE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
