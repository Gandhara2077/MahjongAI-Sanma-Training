# Sanma Multi-Seed Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the strongest sanma checkpoint using many independently generated game seeds instead of trusting one fixed evaluation batch.

**Architecture:** Reuse Mortal's existing `one_vs_two.py` evaluator. Give each candidate its own TOML and log directory, use one common randomly generated seed key across candidates for a paired selection panel, then validate the leading candidates on a fresh seed key. The common panel is not a single fixed game set: each candidate plays 3,000 games across 1,000 seed values, and the fresh panel prevents selection-panel overfitting.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+ROCm 7.2.1, AMD RX 9070 XT, Mortal `OneVsTwo`, PowerShell 5.1, TOML, gzip MJAI logs.

**Spec:** User request to continue checking which trained checkpoint is best and judge strength over a large number of games rather than one fixed seed.

## Global Constraints

- Do not start evaluation while another training or evaluation process is running.
- Evaluate sanma checkpoints only through the configured ROCm Python environment and active sanma native extension.
- Keep the original base model, 32k, 36k, and 40k checkpoints distinguishable and never overwrite their files.
- Each candidate has an independent TOML and log directory; no evaluation run may delete another candidate's logs.
- Use `games_per_iter=600` and `iters=5` for the paired selection panel: 3,000 games per candidate and 1,000 distinct game seed values.
- Use one seed key generated for the selection panel and a different seed key for confirmation; the same panel is shared across candidates for fair paired comparison.
- Promote no candidate solely from a positive point estimate; report uncertainty and retain the base model when the result is statistically inconclusive.

---

### Task 1: Freeze candidates and evaluation inputs

**Files:**
- Create: `Mortal/config/sanma-eval-step32000-multiseed.toml`
- Create: `Mortal/config/sanma-eval-step36000-multiseed.toml`
- Create: `Mortal/config/sanma-eval-step40000-multiseed.toml`
- Verify: `Mortal/training/sanma_snapshots_workers6/mortal_step32000.pth`
- Verify: `Mortal/training/sanma_snapshots_workers6/mortal_step36000.pth`
- Verify: `Mortal/training/sanma_snapshots_workers6/mortal_step40000.pth`

**Interfaces:**
- Consumes: the completed 40k training run and the existing workers6 sanma evaluation configuration.
- Produces: three candidate TOMLs whose challenger state files point to immutable 32k, 36k, and 40k snapshots and whose log directories are unique.

- [ ] Verify that no Python training/evaluation process is active and that all three snapshots load on CPU with `steps` equal to 32000, 36000, and 40000.
- [ ] Copy the existing workers6 evaluation TOML into the three candidate TOMLs and change only the challenger state path, candidate name, and candidate log directory.
- [ ] Set each selection TOML to `games_per_iter=600`, `iters=5`, and the generated selection seed key; keep champion state pointed at the original configured sanma base model.
- [ ] Parse all three TOMLs with `tomllib` and verify each challenger path, champion path, and log directory before running games.

### Task 2: Run the paired 3,000-game selection panel

**Files:**
- Outputs: `Mortal/training/sanma_multiseed_selection_step32000/`
- Outputs: `Mortal/training/sanma_multiseed_selection_step36000/`
- Outputs: `Mortal/training/sanma_multiseed_selection_step40000/`

**Interfaces:**
- Consumes: the three candidate TOMLs from Task 1.
- Produces: 3,000 complete compressed game logs and five per-batch ranking summaries for each candidate.

- [ ] Run candidates serially on the RX 9070 XT with the same selection seed key; do not run them in parallel on the single GPU.
- [ ] Confirm each candidate completes five 600-game batches and produces exactly 3,000 `.json.gz` logs.
- [ ] Sum the five ranking vectors for each candidate and compute average rank and `[90,0,-90]` average rank points.
- [ ] Compute a game-level 95% confidence interval for each average rank and retain the complete per-batch output.

### Task 3: Confirm the leading candidates on a fresh panel

**Files:**
- Create: `Mortal/config/sanma-eval-confirm-step32000.toml` if 32k is selected for confirmation
- Create: `Mortal/config/sanma-eval-confirm-step36000.toml` if 36k is selected for confirmation
- Create: `Mortal/config/sanma-eval-confirm-step40000.toml` if 40k is selected for confirmation
- Outputs: `Mortal/training/sanma_multiseed_confirm_<candidate>/`

**Interfaces:**
- Consumes: selection-panel ranking totals and a new seed key not used by Task 2.
- Produces: an independent 1,998-game confirmation result for each leading candidate, with no reused log directory or seed key.

- [ ] Select at most the top three candidates by the paired selection panel, while retaining the base model as the null reference.
- [ ] Use a new seed key and `games_per_iter=1998`, `iters=1` for each selected candidate.
- [ ] Run each confirmation evaluation serially and verify exactly 1,998 logs per candidate.
- [ ] Compare each candidate against the theoretical base-model line of average rank 2.0 and rank points 0, including 95% intervals.

### Task 4: Produce the model-selection decision

**Files:**
- Create: `reports/sanma-multiseed-evaluation-20260830.md`

**Interfaces:**
- Consumes: selection and confirmation summaries, checkpoint metadata, and log counts.
- Produces: a reproducible report naming the best supported checkpoint or explicitly retaining the base model.

- [ ] Record the seed keys, candidate paths, game counts, ranking vectors, mean ranks, rank points, confidence intervals, and process exit results.
- [ ] Mark a candidate as supported only when the fresh confirmation agrees with the selection panel and the uncertainty interval does not treat the result as a meaningful regression.
- [ ] Keep all checkpoints available even if no candidate clears the evidence threshold; do not overwrite the 40k deployment artifact.
- [ ] Run one final TOML parse and log-count check before reporting the result.

---

### Task 5: Run a three-candidate sanma mixed-table evaluation

**Files:**
- Create: `Mortal/libriichi/src/arena/three_way.rs`
- Modify: `Mortal/libriichi/src/arena/mod.rs`
- Create: `Mortal/mortal/three_way.py`
- Create: `Mortal/config/sanma-three-way-eval.toml`
- Outputs: `Mortal/training/sanma_three_way_eval/`

**Interfaces:**
- Consumes: immutable 32k, 36k, and 40k deployment/state checkpoints and the active sanma native extension.
- Produces: three models on the same three-player table, with each model occupying each seat once per seed, compressed logs, and a reproducible aggregate comparison.

- [ ] Expose a sanma-only three-agent arena by reusing `BatchGame`, `BatchAgent`, `new_py_agent`, and the existing compressed-log format.
- [ ] Rotate `[32k,36k,40k]`, `[36k,40k,32k]`, and `[40k,32k,36k]` across seats for every seed; reject invalid game counts in the Python entry point.
- [ ] Load all three candidates from TOML-configured state files; do not hard-code checkpoint paths in Python or Rust.
- [ ] Use a seed key different from both the common selection panel and confirmation panel, run at least 3,000 games, and keep logs in an independent directory.
- [ ] Add a focused Rust smoke test for the three-agent seat/index layout, rebuild the sanma extension, and run the smoke check before the long evaluation.
- [ ] Include the mixed-table ranking vectors, average ranks, sanma placement points `[90, 0, -90]`, uncertainty intervals, seed key, and log count in the final report.
