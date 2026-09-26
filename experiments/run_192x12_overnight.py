from __future__ import annotations

import gzip
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORTAL = ROOT / "Mortal"
PYTHON = ROOT / ".venv-rocm" / "Scripts" / "python.exe"
LOG = ROOT / "training_192x12_overnight.log"
TRAIN_CFG = MORTAL / "config" / "sanma-training-192x12.toml"
E2_CFG = MORTAL / "config" / "sanma-training-192x12e2.toml"
BASELINE_CFG = MORTAL / "config" / "sanma-eval-192x12-phase1-vs-baseline.toml"
CHAMPION_CFG = MORTAL / "config" / "sanma-eval-192x12-phase1-vs-champion.toml"
STATE = MORTAL / "training" / "sanma_192x12_train_state.pth"
DEPLOYMENT = MORTAL / "training" / "sanma_192x12_mortal3p.pth"


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def run(command: list[str], *, cwd: Path, env: dict[str, str], label: str) -> None:
    log(f"START {label}: {' '.join(command)}")
    with LOG.open("a", encoding="utf-8") as file:
        file.write(f"\n===== {label} =====\n")
        file.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"END {label}: exit={result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def environment(cfg: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MORTAL_CFG"] = str(cfg)
    env["PYO3_PYTHON"] = str(PYTHON)
    env["MORTAL_PYTHON"] = str(PYTHON)
    env["PYTHONUNBUFFERED"] = "1"
    python_dir = str(PYTHON.parent)
    torch_lib = str(ROOT / ".venv-rocm" / "Lib" / "site-packages" / "torch" / "lib")
    env["PATH"] = os.pathsep.join([python_dir, torch_lib, str(Path("C:/Windows/System32")), env.get("PATH", "")])
    return env


def run_training(cfg: Path, label: str) -> None:
    run([str(PYTHON), "mortal/train.py"], cwd=MORTAL, env=environment(cfg), label=label)


def run_evaluation(cfg: Path, label: str, log_dir: str) -> None:
    env = environment(cfg)
    run(
        [str(PYTHON), "mortal/one_vs_two.py", "--log-dir", log_dir],
        cwd=MORTAL,
        env=env,
        label=label,
    )


def rank_of(seat: int, scores: list[int]) -> int:
    return 1 + sum(score > scores[seat] for score in scores) + sum(
        score == scores[seat] and index < seat
        for index, score in enumerate(scores)
    )


def summarize_eval(directory: Path, challenger_prefix: str) -> dict[str, object]:
    ranks = Counter()
    games = 0
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            events = [json.loads(line) for line in file if line.strip()]
        if not events or events[0].get("type") != "start_game":
            continue
        names = events[0].get("names", [])
        challenger_seats = [
            index for index, name in enumerate(names)
            if str(name).startswith(challenger_prefix)
        ]
        if len(challenger_seats) != 1:
            continue
        scores: list[int] | None = None
        for event in events:
            if event.get("type") == "start_kyoku":
                carried = event.get("scores")
                if carried and len(carried) == 3:
                    scores = list(map(int, carried))
            elif scores is not None and len(event.get("deltas", [])) == 3:
                deltas = event["deltas"]
                scores = [scores[index] + int(deltas[index]) for index in range(3)]
        if scores is None:
            continue
        ranks[rank_of(challenger_seats[0], scores)] += 1
        games += 1
    if games == 0:
        raise RuntimeError(f"no parseable games in {directory}")
    average_rank = sum(rank * count for rank, count in ranks.items()) / games
    average_points = sum(
        ranks.get(rank, 0) * [90, 0, -90][rank - 1] for rank in (1, 2, 3)
    ) / games
    variance = sum(
        count * (rank - average_rank) ** 2 for rank, count in ranks.items()
    ) / max(1, games - 1)
    margin = 1.96 * math.sqrt(variance / games)
    return {
        "games": games,
        "ranks": [ranks.get(1, 0), ranks.get(2, 0), ranks.get(3, 0)],
        "avg_rank": round(average_rank, 6),
        "avg_pt": round(average_points, 3),
        "rank_ci95": [round(average_rank - margin, 6), round(average_rank + margin, 6)],
        "log_dir": str(directory),
    }


def latest_validation_loss() -> dict[str, float] | None:
    if not LOG.exists():
        return None
    text = LOG.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"val loss: dqn ([0-9.]+) rank ([0-9.]+)", text)
    if not matches:
        return None
    dqn, rank = matches[-1]
    return {"dqn": float(dqn), "rank": float(rank)}


def write_result(result: dict[str, object]) -> None:
    path = ROOT / "training_192x12_overnight_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if STATE.exists() or DEPLOYMENT.exists():
        raise RuntimeError("192x12 output already exists; refusing to overwrite it")

    log("192x12 overnight pipeline started")
    log("policy: phase1 46.5k, evaluate sequentially, continue to 93k only if competitive")
    run_training(TRAIN_CFG, "192x12 phase1 training 46500 steps")
    if not STATE.exists() or not DEPLOYMENT.exists():
        raise RuntimeError("phase1 training returned successfully but output files are missing")

    baseline_dir = MORTAL / "training" / "sanma_1v2_192x12_phase1_vs_baseline_20260910"
    champion_dir = MORTAL / "training" / "sanma_1v2_192x12_phase1_vs_champion_20260911"
    run_evaluation(BASELINE_CFG, "192x12 phase1 versus original baseline", str(baseline_dir))
    baseline = summarize_eval(baseline_dir, "sanma-192x12")
    log(f"phase1 baseline result: {baseline}")

    run_evaluation(CHAMPION_CFG, "192x12 phase1 versus 128x8 champion", str(champion_dir))
    champion = summarize_eval(champion_dir, "sanma-192x12")
    log(f"phase1 champion result: {champion}")

    validation = latest_validation_loss()
    competitive = (
        float(baseline["avg_rank"]) < 2.0
        and float(baseline["avg_pt"]) > 0.0
        and float(champion["avg_rank"]) <= 2.10
        and (validation is None or math.isfinite(validation["dqn"]))
    )
    decision = {
        "phase1": {"baseline": baseline, "champion": champion, "validation": validation},
        "continue_to_phase2": competitive,
        "criterion": "baseline avg_rank<2 and avg_pt>0, champion avg_rank<=2.10, finite validation loss",
    }
    write_result(decision)
    log(f"phase1 continuation decision: {competitive}")

    if not competitive:
        log("phase1 is not sufficiently competitive; stopping without second epoch")
        return

    run_training(E2_CFG, "192x12 phase2 continuation to 93000 steps")
    if not STATE.exists() or not DEPLOYMENT.exists():
        raise RuntimeError("phase2 training returned successfully but output files are missing")

    e2_baseline_dir = MORTAL / "training" / "sanma_1v2_192x12_e2_vs_baseline_20260913"
    e2_champion_dir = MORTAL / "training" / "sanma_1v2_192x12_e2_vs_champion_20260914"
    run_evaluation(BASELINE_CFG, "192x12 phase2 versus original baseline", str(e2_baseline_dir))
    e2_baseline = summarize_eval(e2_baseline_dir, "sanma-192x12")
    log(f"phase2 baseline result: {e2_baseline}")

    run_evaluation(CHAMPION_CFG, "192x12 phase2 versus 128x8 champion", str(e2_champion_dir))
    e2_champion = summarize_eval(e2_champion_dir, "sanma-192x12")
    log(f"phase2 champion result: {e2_champion}")

    result = {
        **decision,
        "phase2": {"baseline": e2_baseline, "champion": e2_champion, "validation": latest_validation_loss()},
        "completed_steps": 93000,
    }
    write_result(result)
    log(f"192x12 overnight pipeline completed: {result}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("pipeline interrupted")
        raise
    except Exception as error:
        log(f"pipeline failed: {type(error).__name__}: {error}")
        raise
