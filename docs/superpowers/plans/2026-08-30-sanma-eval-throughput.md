# Sanma Evaluation Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click, reproducible worker-count benchmark to the existing sanma 1v2 evaluator without changing model selection data or hard-coding model paths.

**Architecture:** Keep model files, devices, native library, and base evaluation settings in the selected TOML. Add optional CLI overrides to `one_vs_two.py` only for benchmark controls (`max_workers`, `games_per_iter`, `iters`, and `log_dir`). A PowerShell wrapper runs each worker count serially, gives every run a separate log directory, and writes a compact CSV result after validating the expected compressed-log count.

**Tech Stack:** Python 3.12, existing Mortal evaluator, PyTorch ROCm, PowerShell 5.1, TOML, gzip MJAI logs.

**Spec:** The user requested that the current statistical evaluations finish first, then the evaluation system be changed so worker-count speed can be measured before choosing the next training direction.

## Global Constraints

- Do not modify or reuse the current 40k-final or 28k-best confirmation log directories.
- Run worker variants serially on the single RX 9070 XT.
- Model state files and devices remain TOML-configured; no checkpoint path is added to Python or PowerShell.
- Each benchmark variant must use a unique log directory and must produce exactly the configured number of `.json.gz` logs.
- Keep the current default behavior unchanged when no CLI overrides are supplied.
- Use the ROCm Python environment and the active sanma native extension.

---

### Task 1: Specify the benchmark interface with a failing check

**Files:**
- Create: `checks/eval_worker_benchmark_check.py`
- Verify: `Mortal/mortal/one_vs_two.py`
- Verify: `scripts/08_benchmark_eval_workers.ps1`

**Interfaces:**
- Consumes: a TOML path, worker counts, and a benchmark output root.
- Produces: a check that requires the evaluator to accept `--max-workers`, `--games-per-iter`, `--iters`, and `--log-dir`, and requires the wrapper to expose the same controls.

- [ ] Write a subprocess-based check that runs `one_vs_two.py --help` with `MORTAL_CFG` set to the existing sanma evaluation TOML and asserts exit code 0 plus all four option names.
- [ ] In the same check, parse the wrapper text and assert it invokes `one_vs_two.py`, accepts `WorkerCounts`, and creates a per-worker log directory.
- [ ] Run `\.venv-rocm\Scripts\python.exe checks/eval_worker_benchmark_check.py` and observe the expected failure because the CLI and wrapper do not yet exist.

### Task 2: Add minimal CLI overrides to the evaluator

**Files:**
- Modify: `Mortal/mortal/one_vs_two.py`
- Test: `checks/eval_worker_benchmark_check.py`

**Interfaces:**
- Consumes: existing TOML configuration plus optional CLI arguments.
- Produces: `parse_args(argv)` and `main()` behavior where omitted options use TOML values, while supplied options affect only the current run.

- [ ] Add an `argparse.ArgumentParser` with optional `--max-workers`, `--games-per-iter`, `--iters`, and `--log-dir` arguments.
- [ ] Pass the selected worker count into both challenger and champion `CompatMjaiEngine` instances without changing the TOML default path.
- [ ] Use the selected game count, iteration count, and log directory in `main()`; retain existing positivity/divisibility checks.
- [ ] Run the check and confirm the help assertion passes without loading a model or starting a game.

### Task 3: Add the one-click serial benchmark wrapper

**Files:**
- Create: `scripts/08_benchmark_eval_workers.ps1`
- Modify: `checks/eval_worker_benchmark_check.py`

**Interfaces:**
- Consumes: `-ConfigPath`, `-OutputRoot`, `-WorkerCounts` (default `16,32,64`), and optional `-GamesPerIter` (default `600`).
- Produces: one independent directory per worker count and `benchmark.csv` containing worker count, elapsed seconds, expected logs, actual logs, and exit status.

- [ ] Resolve Python through the existing `common.ps1` context and require Python 3.12 plus ROCm availability.
- [ ] Run worker counts one at a time with `MORTAL_CFG` set to the caller-provided TOML and unique log directories below `OutputRoot`.
- [ ] Invoke `one_vs_two.py --max-workers N --games-per-iter GamesPerIter --iters 1 --log-dir <unique-dir>` and measure wall time around the process.
- [ ] Reject an existing non-empty output directory instead of deleting it; write CSV only after every run has a terminal exit result and log-count check.
- [ ] Run the check again and confirm the wrapper interface is accepted.

### Task 4: Run and verify the worker benchmark

**Files:**
- Outputs: `Mortal/training/sanma_eval_worker_benchmark_20260830/`
- Outputs: `reports/sanma-eval-worker-benchmark-20260830.md`

**Interfaces:**
- Consumes: the completed 40k final evaluation configuration and the active ROCm/native setup.
- Produces: measured 16/32/64 worker timings, complete log counts, and a recommendation without changing model-selection artifacts.

- [ ] Run the wrapper serially for 600 games per worker count with separate output directories.
- [ ] Verify each run has exactly 600 `.json.gz` files, no compatibility failure, and exit code 0.
- [ ] Record cold-start time separately from steady game-batch completion where the evaluator output permits it; otherwise label the result as end-to-end wall time.
- [ ] Choose a higher worker count only if it is at least 10% faster than 16 and has no errors; otherwise retain 16.
- [ ] Run the focused check and a TOML/log-count validation before reporting the result.

