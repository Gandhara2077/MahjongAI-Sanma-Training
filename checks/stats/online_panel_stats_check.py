"""Focused regression check for strict online panel statistics."""

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "checks") not in sys.path:
    sys.path.insert(0, str(ROOT / "checks"))

from online_panel_stats import panel_stats, pooled_stats
from arena_log_results import game_result, rank_of


ROTATIONS = ("a", "b", "c")


def _write_game(
    directory,
    seed,
    seed_key,
    rotation,
    names,
    *,
    end_game=True,
    filename_seed=None,
):
    challenger_seat = names.index("target")
    events = [
        {"type": "start_game", "names": names, "seed": [seed, seed_key]},
        {"type": "start_kyoku", "scores": [35000, 35000, 35000], "kyotaku": 0},
        {"type": "reach_accepted", "actor": challenger_seat},
        {"type": "ryukyoku", "deltas": [0, 0, 0]},
        {"type": "end_kyoku"},
    ]
    if end_game:
        events.append({"type": "end_game"})
    file_seed = seed if filename_seed is None else filename_seed
    path = directory / f"{file_seed}_{seed_key}_{rotation}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")


def _write_complete_seed(directory, seed=100, seed_key=7):
    names_by_rotation = {
        "a": ["other-model", "target", "mortal-baseline"],
        "b": ["target", "other-model", "mortal-baseline"],
        "c": ["mortal-baseline", "other-model", "target"],
    }
    for rotation in ROTATIONS:
        _write_game(
            directory,
            seed,
            seed_key,
            rotation,
            names_by_rotation[rotation],
        )


def _expect_error(callback):
    try:
        callback()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    raise AssertionError("expected strict validation error")


def _stage(seed_key, means, panel_id):
    return {
        "panel_id": panel_id,
        "seed_records": [
            {
                "panel_id": panel_id,
                "seed_key": seed_key,
                "seed": seed,
                "games": [
                    {"rotation": rotation, "challenger_seat": seat, "challenger_rank": mean}
                    for seat, rotation in enumerate(ROTATIONS)
                ],
            }
            for seed, mean in enumerate(means, 1)
        ],
    }


def _check():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        panel = root / "panel"
        panel.mkdir()
        _write_complete_seed(panel)

        result = panel_stats(panel, "target", expected_games=3, panel_id="fixture")
        assert json.dumps(result)
        assert result["panel_id"] == "fixture"
        assert result["games"] == 3
        assert result["seeds"] == 1
        assert result["standard_error"] is None
        assert result["ci95"] is None
        assert result["mean_rank"] == 3.0
        assert result["effect"] == 1.0
        assert result["significant"] is False

        seed_record = result["seed_records"][0]
        assert seed_record["seed_key"] == 7
        assert seed_record["seed"] == 100
        assert [game["challenger_seat"] for game in seed_record["games"]] == [1, 0, 2]
        assert [game["challenger_rank"] for game in seed_record["games"]] == [3, 3, 3]
        assert seed_record["games"][0]["scores"] == [36000, 34000, 35000]

        incomplete = root / "incomplete"
        incomplete.mkdir()
        _write_complete_seed(incomplete)
        (incomplete / "100_7_c.json.gz").unlink()
        _expect_error(lambda: panel_stats(incomplete, "target", 3))

        truncated = root / "truncated"
        truncated.mkdir()
        _write_complete_seed(truncated)
        (truncated / "100_7_b.json.gz").unlink()
        _write_game(
            truncated,
            100,
            7,
            "b",
            ["target", "other-model", "mortal-baseline"],
            end_game=False,
        )
        _expect_error(lambda: panel_stats(truncated, "target", 3))

        duplicate = root / "duplicate"
        duplicate.mkdir()
        _write_complete_seed(duplicate)
        _write_game(
            duplicate,
            100,
            7,
            "a",
            ["other-model", "target", "mortal-baseline"],
            filename_seed="0100",
        )
        _expect_error(lambda: panel_stats(duplicate, "target", 4))

        summary_mismatch = root / "summary-mismatch"
        summary_mismatch.mkdir()
        _write_complete_seed(summary_mismatch)
        (summary_mismatch / "summary.json").write_text(
            json.dumps({"games": 2, "expected_log_count": 2, "log_count": 2}),
            encoding="utf-8",
        )
        _expect_error(lambda: panel_stats(summary_mismatch, "target", 3))

        summary_valid = root / 'summary-valid'
        summary_valid.mkdir()
        _write_complete_seed(summary_valid)
        (summary_valid / 'summary.json').write_text(json.dumps({'games': 3, 'rankings': [0, 0, 3]}), encoding='utf-8')
        assert panel_stats(summary_valid, 'target', 3)['native_summary']['status'] == 'verified'
        native_log = root / 'native.log'
        native_log.write_text('challenger rankings: [0 0 3]', encoding='utf-8')
        assert panel_stats(summary_valid, 'target', 3, native_log=native_log)['native_summary']['status'] == 'verified'
        native_log.write_text('challenger rankings: [1 0 2]', encoding='utf-8')
        _expect_error(lambda: panel_stats(summary_valid, 'target', 3, native_log=native_log))

        malformed = root / 'malformed'
        malformed.mkdir()
        _write_complete_seed(malformed)
        altered_file = malformed / '100_7_a.json.gz'
        with gzip.open(altered_file, 'rt', encoding='utf-8') as source:
            original_events = [json.loads(line) for line in source]
        for field, value in (('seed', [100, 8]), ('names', ['target', 'target', 'other'])):
            events = json.loads(json.dumps(original_events))
            events[0][field] = value
            with gzip.open(altered_file, 'wt', encoding='utf-8') as destination:
                destination.write('\n'.join(json.dumps(event) for event in events))
            _expect_error(lambda: panel_stats(malformed, 'target', 3))
        events = json.loads(json.dumps(original_events))
        events[1]['scores'][0] = float('nan')
        _expect_error(lambda: game_result(events, challenger_name='target'))
        assert rank_of(0, [35000, 35000, 35000]) == 1
        assert rank_of(2, [35000, 35000, 35000]) == 3
        missing_rotation = _stage(7, [1, 2], 'bad')
        missing_rotation['seed_records'][0]['games'][1]['challenger_seat'] = 0
        _expect_error(lambda: pooled_stats([missing_rotation]))

        empty = root / "empty"
        empty.mkdir()
        _expect_error(lambda: panel_stats(empty, "target", 0))

        first = _stage(7, [1, 2, 3], "s1")
        second = _stage(8, [1, 2, 3], "s2")
        pooled = pooled_stats([first, second])
        assert pooled["games"] == 18
        assert pooled["seeds"] == 6
        assert pooled["mean_rank"] == 2.0
        assert math.isclose(pooled["standard_error"], math.sqrt(0.8 / 6))
        assert math.isclose(pooled["ci95"][0], 2.0 - 1.96 * math.sqrt(0.8 / 6))
        assert math.isclose(pooled["ci95"][1], 2.0 + 1.96 * math.sqrt(0.8 / 6))
        assert pooled["effect"] == 0.0
        assert pooled["significant"] is False
        _expect_error(lambda: pooled_stats([first, first]))

    print("ONLINE_PANEL_STATS_CHECK_OK")


if __name__ == "__main__":
    _check()
