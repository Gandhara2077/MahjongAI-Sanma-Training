"""Focused contract check for the evaluation worker benchmark entry point."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = ROOT / "Mortal"
EVALUATOR = MORTAL_ROOT / "mortal" / "one_vs_two.py"
WRAPPER = ROOT / "scripts" / "08_benchmark_eval_workers.ps1"
CONFIG = MORTAL_ROOT / "config" / "sanma-eval-lr2-final-multiseed.toml"


def load_evaluator():
    os.environ["MORTAL_CFG"] = str(CONFIG)
    sys.path.insert(0, str(MORTAL_ROOT / "mortal"))
    spec = importlib.util.spec_from_file_location("one_vs_two_for_check", EVALUATOR)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load evaluator: {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_evaluator()
    args = module.parse_args(
        [
            "--max-workers",
            "32",
            "--games-per-iter",
            "600",
            "--iters",
            "1",
            "--log-dir",
            "training/worker-check",
        ]
    )
    assert args.max_workers == 32
    assert args.games_per_iter == 600
    assert args.iters == 1
    assert args.log_dir == "training/worker-check"

    assert WRAPPER.is_file(), WRAPPER
    wrapper = WRAPPER.read_text(encoding="utf-8")
    for marker in (
        "ConfigPath",
        "OutputRoot",
        "WorkerCounts",
        "GamesPerIter",
        "PythonExe",
        "DryRun",
        "Get-TrainingContext",
        "Use-ActiveVariant",
        "Assert-Rocm",
        "MORTAL_CFG",
        "one_vs_two.py",
        "--max-workers",
        "--games-per-iter",
        "--iters",
        "--log-dir",
        "Stopwatch",
        "json.gz",
        "benchmark.csv",
        "worker_count",
        "elapsed_seconds",
        "expected_logs",
        "actual_logs",
        "exit_code",
        "Export-Csv",
    ):
        assert marker in wrapper, marker
    assert "Remove-Item" not in wrapper

    with tempfile.TemporaryDirectory() as temporary:
        output_root = Path(temporary) / "worker-benchmark"
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WRAPPER),
                "-ConfigPath",
                str(CONFIG),
                "-OutputRoot",
                str(output_root),
                "-DryRun",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            completed.stderr.strip() or completed.stdout.strip()
        )
        assert "workers-16" in completed.stdout
        assert "workers-32" in completed.stdout
        assert "workers-64" in completed.stdout
        assert "--games-per-iter 600" in completed.stdout
        assert "--iters 1" in completed.stdout
        assert not output_root.exists()

    print("EVAL_WORKER_BENCHMARK_CONTRACT_OK")


if __name__ == "__main__":
    main()
