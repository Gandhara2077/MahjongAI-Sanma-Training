# MahjongAI-Sanma-Training

[简体中文](../README.md) · **English** · [日本語](README.ja.md)

A **reproducible training workflow for three-player Japanese mahjong (sanma) AI**, built on top of
[Mortal](https://github.com/Equim-chan/Mortal). It is not a `pip install`-able Python package.

Upstream Mortal targets four-player mahjong (yonma). This repository adds:

| Capability | Description |
| --- | --- |
| Sanma rules and encoding | Three-player native extension (nukidora, sanma hand flow) with `775x34` observation and 44-action space |
| Variant switching | One `variant` key in `config/training-profile.toml` selects training config, automation config, player count and data roots |
| Data pipeline | Download Tenhou raw `.mjlog`, convert to gzip JSON Lines `.mjson`, emit verified manifests |
| GRP training | Train a placement-point predictor (GRP) first, then the policy; the frozen GRP supplies dense rewards for online self-play |
| Statistical evaluation | Three-way arenas (`one_vs_two` / `one_vs_three`), staged fixed-budget panels, 95% confidence intervals and paired tests |
| Fail-closed | Scripts abort when ROCm is unavailable, assets are missing, or the ABI mismatches — never silently fall back to CPU |

> **Scope boundary:** the native online self-play path is experimental. Its throughput, resume
> behaviour, internal smoke tests or self-play results are *not* equivalent to the compatible
> reference evaluator, and do not establish playing strength, promotion, or deployment safety.
> No "champion" claim is made here.

## Requirements

| Item | Requirement |
| --- | --- |
| OS | 64-bit Windows, PowerShell 5.1 or PowerShell 7 |
| Python | **CPython 3.12** (other minor versions are rejected) |
| GPU | Supported AMD GPU with ROCm runtime; scripts use `cuda:0`, exposed by ROCm PyTorch via `torch.cuda` |
| PyTorch | AMD ROCm wheel matching your GPU, Windows and Python versions |
| Rust | stable/MSVC toolchain and Cargo (plus C++ build tools when no reusable `.pyd` exists) |

Do **not** run a bare `pip install torch` and do **not** use the CUDA PyTorch index — neither proves
AMD ROCm support. The pipeline fails when no GPU is available instead of degrading quietly to CPU.

## Quick start

Full details live in [`../USAGE.md`](../USAGE.md). Shortest path:

```powershell
git clone https://github.com/Gandhara2077/MahjongAI-Sanma-Training.git
cd MahjongAI-Sanma-Training

conda env create -f Mortal/environment.yml
conda activate mortal
python -m pip install -r requirements.txt
# Install the ROCm torch wheel matching your GPU (see ../USAGE.md section 2)

# Preview the pipeline: no download, no training, no external weights required
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -DryRun -SkipData

# Then place the external sanma weights and the libriichi3p reference (see ../USAGE.md section 3)
```

## Pipeline

| Stage | Script | Purpose |
| --- | --- | --- |
| 01 | `01_prepare.ps1` | Build and activate the native extension for the current variant |
| 02 | `02_fetch_data.ps1` | Download, convert and verify Tenhou game logs |
| 03 | `03_validate.ps1` | ROCm, profile, asset, ABI and bounded smoke checks |
| 04 | `04_train_grp.ps1` | Train the GRP state (placement-point predictor) |
| 05 | `05_train_offline.ps1` | Offline training of the main model |
| 06 | `06_evaluate.ps1` | Evaluation: sanma `one_vs_two`, yonma `one_vs_three` |
| 07 | `07_export.ps1` | Export deployment-format weights from the full training state |

Resume from an intermediate stage when native and data already exist:

```powershell
.\scripts\run_all.ps1 -FromStep validate -SkipPrepare -SkipData
```

## Variants

Edit only `config/training-profile.toml`:

```toml
[active]
variant = 'sanma' # or 'yonma'
```

- `sanma`: 3 players, `Mortal/config/sanma-training.toml`, data root `koromo/`, default model
  `baselines/sanma_baseline_mortal3p_original.pth`.
- `yonma`: 4 players, `Mortal/config/yonma.toml`, data root `koromo4p/`, default model
  `baselines/baseline.pth`.

`Mortal/mortal/libriichi.pyd` is a **single active native slot**. Re-run `01_prepare.ps1` after
switching variants; two variants cannot be active in one worktree.

## Data and artifacts

Training data is **not distributed with this repository**. Game logs come from
[Tenhou official logs](https://tenhou.net/sc/raw/) — follow their terms, and never split one game
across the training and validation sets.

Weights, `Mortal/training/` outputs (states, deployment weights, GRP state, indices, TensorBoard
logs, match records), `koromo/{mjlog,mjson}` and the reference extension are local artifacts,
excluded by `.gitignore`. Rebuild them in the order `04 -> 05 -> 06 -> 07`.
`artifacts/sanma-assets.json` records only sizes, SHA-256 and ABI (version 4, observation
`775x34`, action space 44) of external assets — never the binaries themselves.

## Checks and CI

GPU-free checks:

```powershell
python checks/abi/training_profile_check.py --generic
python checks/abi/one_click_dry_run_check.py
git diff --check
```

CI (`../.github/workflows/ci.yml`) runs these two contract checks on every push and pull request,
byte-compiles project-owned Python, and applies error-level ruff rules. The vendored upstream tree
`Mortal/` is excluded from house style checks. CI does **not** verify GPU training or playing strength.

## License

This repository is a modified copy of [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) and
is distributed under **GNU AGPL v3.0** (see [`../LICENSE`](../LICENSE) and
[`../Mortal/LICENSE`](../Mortal/LICENSE)). Attribution and distribution boundaries are in
[`../NOTICE.md`](../NOTICE.md): model weights, Tenhou logs and the MahjongCopilot `libriichi3p`
reference are external inputs and are not redistributed here.
