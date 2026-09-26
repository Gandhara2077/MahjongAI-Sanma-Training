# Sanma Mortal Training Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible three-player Japanese mahjong Mortal training project initialized from the user's MahjongCopilot 3p checkpoint.

**Architecture:** Reuse the checked-in `Rezetyan/MahjongAITraining` fork. Compile `Mortal/libriichi` with the `sanma` feature for native rules and metadata, while replaying each training log through the pinned `libriichi3p` reference extension to preserve the 775-channel deployment ABI. Keep weights, raw logs, native binaries, and training output outside Git; commit manifests, scripts, configs, reports, and stage notes.

**Tech Stack:** Python 3.12, PyTorch 2.9.1 with ROCm 7.2.1, NumPy, TOML, Rust 1.94+ with the MSVC target, Visual Studio C++ build tools, PowerShell, Cargo, Tenhou `.mjlog`, gzip JSONL `.mjson`, and GitHub.

**Spec:** `docs/sanma-training-spec.md`

## Global Constraints

- The active game variant is sanma: `NUM_PLAYERS = 3`, no chi, north extraction enabled, and 108 tiles.
- The deployment ABI is `control.version = 4`, observation shape `(775, 34)`, and action space `44`.
- Native metadata uses version `5`; the historical 3p reference extension supplies deployment observations and masks.
- Python must be 3.12 for the AMD ROCm Windows wheel; the reference extension must be the Python 3.12 x86_64 Windows build.
- Raw logs, converted logs, checkpoints, native binaries, and local caches remain Git-ignored.
- Every implementation task ends with one runnable check and a focused Git commit.

---

### Task 1: Record project identity and local asset contract

**Files:**
- Create: `docs/sanma-training-spec.md`
- Create: `docs/superpowers/plans/2026-08-28-sanma-training-project.md`
- Create: `tools/prepare_sanma_assets.ps1`
- Create: `checks/sanma_setup.py`
- Modify: `.gitignore`

**Interfaces:**
- `tools/prepare_sanma_assets.ps1 -MahjongCopilotRoot <path>` copies the user-provided 3p model and Python 3.12 reference extension into `.cache/` and writes `artifacts/sanma-assets.json` with byte sizes and SHA-256 values.
- `python checks/sanma_setup.py` exits zero only when the manifest, model fingerprint, reference filename, and TOML ABI fields agree.

- [ ] Write `checks/sanma_setup.py` first with assertions for the manifest schema and the fixed `(775, 34, 44)` ABI.
- [ ] Run it before the manifest exists and confirm the expected missing-manifest failure.
- [ ] Write the minimal PowerShell asset copier, manifest writer, and `.gitignore` additions.
- [ ] Run the copier against `C:\Users\Administrator\Downloads\MahjongCopilot - 副本`, then run `python checks/sanma_setup.py` and confirm zero exit status.
- [ ] Commit as `chore: establish sanma asset contract`.

### Task 2: Verify model and reference-extension compatibility

**Files:**
- Create: `checks/sanma_abi.py`
- Modify: `Mortal/mortal/libriichi3p_compat.py` only if the check exposes a real mismatch
- Modify: `Mortal/config/sanma.toml` only for validated path/ABI corrections

**Interfaces:**
- `python checks/sanma_abi.py` imports the cached reference extension, checks `MAX_VERSION = 4`, `ACTION_SPACE = 44`, `obs_shape(4) = (775, 34)`, and loads the checkpoint with `mortal.model.Brain`/`DQN` when PyTorch is installed.

- [ ] Write the ABI assertions and run them without the native extension to capture the expected failure.
- [ ] Install the AMD ROCm 7.2.1 Python 3.12 wheels and rerun until the failure is specifically about a missing or incompatible native module, not an import typo.
- [ ] Copy the cached reference extension into the resolver path and make the ABI check pass.
- [ ] Run the check twice in fresh Python processes to catch module-name/path collisions.
- [ ] Commit as `test: verify sanma deployment ABI`.

### Task 3: Build and validate native sanma rules

**Files:**
- Modify: `Mortal/libriichi/src/*` only when a failing sanma test identifies a rule defect
- Create: `checks/native_sanma_smoke.py`
- Create: `scripts/build_sanma_native.ps1`

**Interfaces:**
- `scripts/build_sanma_native.ps1` builds `Mortal/libriichi` with `--features sanma` for the active Python 3.12 interpreter and places the extension under `Mortal/native/sanma/`.
- `python checks/native_sanma_smoke.py` imports the active native module and asserts `NUM_PLAYERS = 3`, `MAX_VERSION = 5`, `ACTION_SPACE = 44`, and `obs_shape(5) = (780, 34)`.

- [ ] Run `cargo test -p libriichi --features sanma` before adding helper scripts and record the toolchain failure if Rust is absent.
- [ ] Install or expose the existing Rust/MSVC toolchain, then rerun the unmodified sanma test suite.
- [ ] Add only the build wrapper and smoke check needed to make the tested build reproducible.
- [ ] Run the complete sanma test command and the import smoke check.
- [ ] Commit as `build: add reproducible sanma native build`.

### Task 4: Download, convert, and validate public sanma data

**Files:**
- Modify: `koromo/download_tenhou.py` only for a failing downloader test
- Modify: `koromo/convert_mjlog_to_mjson.py` only for a failing converter test
- Create: `scripts/fetch_sanma_data.ps1`
- Create: `data/manifests/README.md`

**Interfaces:**
- `scripts/fetch_sanma_data.ps1 -StartDate <YYYYMMDD> -EndDate <YYYYMMDD` downloads only the sanma variant, converts it, and writes a manifest containing date range, file counts, byte totals, and SHA-256 values.
- The converter must reject yonma files in the sanma command and reject incomplete gzip JSONL output.

- [ ] Run the existing downloader and converter help commands as parser checks.
- [ ] Download a small bounded date window, then verify the downloaded game-type bits and three-player start events.
- [ ] Convert the window and run the native `validate_logs --features sanma` binary.
- [ ] Record the exact data range and validation counts in `data/manifests/` without committing the logs.
- [ ] Commit as `data: add reproducible sanma corpus manifest`.

### Task 5: Run a bounded loader and offline-training smoke test

**Files:**
- Create: `configs/sanma-smoke.toml`
- Create: `scripts/run_sanma_smoke.ps1`
- Create: `checks/training_smoke.py`
- Modify: `Mortal/config/sanma.toml` only for a verified production default

**Interfaces:**
- `scripts/run_sanma_smoke.ps1` selects the sanma native variant, points `MORTAL_CFG` at `configs/sanma-smoke.toml`, runs one small loader pass and a bounded offline train, and leaves output under ignored `Mortal/training/`.
- `python checks/training_smoke.py <deployment-checkpoint>` reloads the compact checkpoint and verifies its fixed shapes and finite tensors.

- [ ] Write the loader assertions and run them against an empty corpus to confirm the expected no-data failure.
- [ ] Configure a small batch/step count and run on one validated log file.
- [ ] Confirm the training process writes both full state and compact deployment checkpoint.
- [ ] Reload the compact checkpoint and assert finite tensors, output shape 45, and legal-mask compatibility.
- [ ] Commit as `train: add bounded sanma smoke pipeline`.

### Task 6: Evaluate the first checkpoint and document the result

**Files:**
- Create: `reports/sanma-smoke-report.md`
- Create: `scripts/run_sanma_eval.ps1`
- Modify: `Mortal/mortal/one_vs_two.py` only if the smoke evaluation exposes a real sanma integration failure

**Interfaces:**
- `scripts/run_sanma_eval.ps1` runs the existing 1v2 evaluator against the baseline and the first trained deployment checkpoint.
- `reports/sanma-smoke-report.md` records commit SHA, model SHA-256, corpus manifest, game count, average rank/points, and known limitations.

- [ ] Run 1v2 on the baseline first to prove the evaluator and rule engine are connected.
- [ ] Run 1v2 on the trained checkpoint with the same seed/configuration.
- [ ] Capture raw JSON/stat output and write the report from those values.
- [ ] Mark the checkpoint as deployable only when the ABI check and evaluation command both pass.
- [ ] Commit as `eval: record first sanma checkpoint result`.

### Task 7: Publish stages to the user's GitHub

**Files:**
- Modify: Git history and remote refs only after repository ownership is verified
- Create: `CHANGELOG.md` if the remote project needs a stage index

**Interfaces:**
- Remote target: `SylastheUnshackled/MahjongAI-Sanma-Training` unless the authenticated GitHub account proves another explicit target.
- Stage branches: `stage/00-bootstrap`, `stage/01-abi`, `stage/02-native`, `stage/03-data`, `stage/04-training`, `stage/05-evaluation`.

- [ ] Verify the target repository exists and the authenticated account has push permission.
- [ ] Push each local stage branch and the final `main` fast-forward without force-pushing.
- [ ] Verify every remote ref and commit SHA through the GitHub API.
- [ ] Put the final model in a release/artifact channel rather than Git history when its size or licensing requires it.
- [ ] Commit as `chore: publish sanma training stages`.
