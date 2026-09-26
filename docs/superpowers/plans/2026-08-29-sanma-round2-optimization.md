# Sanma Round-2 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed 13,000-step sanma run into a reproducible baseline, validate the CPU/GPU data path, and run a better-configured sanma training round with statistically useful model selection.

**Architecture:** Keep the existing Mortal sanma compat-replay path and TOML/profile selection. The next run uses the clean January 1–July 24 training window, keeps July 25–August 7 held out for GRP validation, and preserves the 13k deployment as a separate baseline. Only measured bottlenecks justify code changes.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+ROCm 7.2.1, AMD RX 9070 XT, Mortal/libriichi, PowerShell 5.1, TOML, gzip MJAI logs.

**Spec:** User request to finish the current 13,000-step run before changing parameters, plus the supplied review recommendations for sanma throughput, GRP training, evaluation, and data-window hygiene.

## Global Constraints

- Do not interrupt or duplicate a running training/evaluation process.
- Preserve unrelated pre-existing worktree changes; never use `git reset --hard` or `git checkout --`.
- Select sanma/yonma, the base model, and all paths through the profile/TOML files; do not hard-code them in scripts.
- Use ROCm `cuda:0` through `.venv-rocm\Scripts\python.exe`.
- Keep `2026/07/25`–`2026/08/07` out of the main-model training globs.
- Treat the 13k checkpoint as a preliminary baseline until 600-game and 1998-game comparisons are complete.

---

### Task 1: Preserve the completed 13k baseline

**Files:**
- Create: `Mortal/training/sanma_mortal3p_preliminary_step13000.pth`
- Verify: `Mortal/training/sanma_train_state.pth`

- [x] Confirm the training process exited with `OFFLINE_TRAIN_OK variant=sanma` and logged `total steps: 13,000`.
- [x] Copy the deployment checkpoint and compare SHA-256 hashes with `sanma_mortal3p.pth`.
- [x] Load the state on CPU and confirm `state['steps'] == 13000`.
- [x] Run the configured 600-game `one_vs_two.py` comparison and record rankings, average rank, and average points.

Acceptance evidence: the preliminary checkpoint is separate, hashes match, the state reports 13,000 steps, and the comparison output is retained in the session record.

### Task 2: Make the next-round profile internally consistent

**Files:**
- Modify: `Mortal/config/sanma-training.toml`
- Modify: `Mortal/automation/sanma.toml`
- Test: `checks/performance_profile_check.py`

- [x] Set the offline path to `batch_size=1024`, `num_workers=2`, `test_every=4000`, `snapshot_every=500`, and `games_per_iter=600`.
- [x] Set both training scheduler and control target to 40,000 steps, subject to the measured epoch estimate.
- [x] Set GRP control to at least 10,000 steps and use the two-week held-out validation window.
- [x] Ensure automation has `deterministic=false`, so configured workers are not silently forced to zero.
- [x] Run `performance_profile_check.py` and the generic profile check.

Acceptance evidence: both TOMLs parse, profile checks pass, and the active profile resolves to sanma with the configured base model and ROCm interpreter.

### Task 3: Measure the compat data path before changing loader code

**Files:**
- Test: `checks/performance_profile_check.py`
- Inspect only unless a measured regression requires it: `Mortal/mortal/dataloader.py`, `Mortal/mortal/train.py`

- [x] Build or load the clean file index for the 63,454 main-window files.
- [x] Measure first-batch latency and sustained samples/sec for `num_workers` 0, 1, and 2 at batch 512; repeat workers 1 and 2 at batch 1024.
- [x] Record CPU utilization, GPU availability, worker errors, and the observed incompatible-replay count.
- [x] Keep the smallest configuration that improves throughput without worker errors or CPU oversubscription.
- [x] Only if the benchmark demonstrates duplicate parsing as the dominant remaining cost, add one failing regression check before changing compat replay code.

Acceptance evidence: a compact benchmark table exists in the session notes, and the selected worker/batch values are justified by measurements.

### Task 4: Run preflight checks and one-click dry-run

**Files:**
- Verify: `scripts/03_validate.ps1`, `scripts/05_train_offline.ps1`, `scripts/06_evaluate.ps1`, `scripts/run_all.ps1`
- Tests: `checks/*.py`

- [ ] Run the config/profile checks.
- [ ] Run `scripts/03_validate.ps1` with the ROCm Python and confirm native sanma, ABI, compat, and replay smoke checks.
- [ ] Run the complete one-click dry-run from `prepare` through `export` and verify every resolved path.
- [ ] Run the focused scheduler-resume, compat-recovery, variant, and offline-selection checks.

Acceptance evidence: static checks, tests, ROCm/native runtime checks, and dry-run output are reported separately.

### Task 5: Train GRP and the second offline round

**Files:**
- Outputs: `Mortal/training/sanma_grp.pth`, `Mortal/training/sanma_train_state.pth`, `Mortal/training/sanma_mortal3p.pth`, `Mortal/training/sanma_snapshots/`

- [ ] Train/resume GRP using the configured 10,000-step target and save its metrics/best state.
- [ ] Confirm the GRP validation set does not overlap the main-model globs.
- [ ] Start one offline sanma run from the configured base/state path with the selected measured data-loader settings.
- [ ] Leave the process uninterrupted until its configured target; monitor steps, loss, samples/sec, GPU availability, and warnings.
- [ ] Preserve snapshots at the configured cadence.

Acceptance evidence: GRP metrics and main-model state report their actual final steps, no second training process is active, and all outputs load successfully.

### Task 6: Select and validate the best model

**Files:**
- Verify: `Mortal/mortal/one_vs_two.py`, `scripts/06_evaluate.ps1`, `Mortal/training/sanma_snapshots/`
- Update if needed: `Mortal/config/sanma-training.toml`

- [ ] Screen the new offline snapshots against the fixed sanma base using 600 games and common seeds.
- [ ] Validate the selected candidate with 1,998 games and report average rank, points, and uncertainty/significance fields where available.
- [ ] Keep the 13k preliminary model, configured base model, and selected new model distinguishable.
- [ ] Promote only a candidate with positive point estimate and the configured significance gate; otherwise retain the stronger prior model.

Acceptance evidence: model choice is based on the configured evaluation gates, not a 30-game sample or an unevaluated `best_*` path.

### Task 7: Update documentation and delivery state

**Files:**
- Modify: `README.md`
- Review: complete worktree diff and untracked file list

- [ ] Document the active profile, ROCm command, data window, checkpoint locations, resume command, and evaluation commands.
- [ ] Document that raw `.mjson` files remain compressed gzip JSON and indexes/checkpoints are PyTorch files.
- [ ] Run one final focused check after documentation changes.
- [ ] Review the complete diff for accidental changes and prepare a focused commit; push only after the final diff scope is confirmed.

Acceptance evidence: a new checkout can follow the README without editing hard-coded paths, and the commit contains only the sanma training workflow changes.
