# Native v4 rollout — 2026-09-06

The offline sanma training observation pipeline is now served by the native
Rust v4 encoder. This report records the acceptance evidence.

## 1. Bit-exact parity gate (PASS)

`checks/native_v4_encoder_check.py --native-extension .cache/native-v4/target/release/riichi.dll --output .cache/native-v4/resume-rerun-full-gate.json`

- Corpus: 40 files from 5 sampled days, 150 games, 17,934 samples (raw + 10
  augmented files).
- Result: `obs_mismatches=0`, `mask_mismatches=0`, `metadata_mismatches=0`,
  `illegal_actions=0`, `compat_skips=0`. `RESULT: PASS`.
- Native build sha256: `6117493924c3a18d4291f9e7423688228ca4cd33f73b78f418eae96fdb86fd3d`
- Reference sha256: `0a0b295e6954c7fe85765b69519c8bcfd31af19c224281b203f9c20672962523`

The final four divergences (rows 652/653, the max-EV rescale rows of the v4
layout) were resolved by the last 2026-09-06 rebuild; this rerun is the first
full gate recorded against that build.

## 2. Training-target parity

Both dataloader paths compute `steps_to_done`, `kyoku_rewards` (GRP) and
`player_ranks` with identical Python code from loader metadata
(`at_kyoku`, `dones`, `apply_gamma`, `grp`). A/B of v4-mode (direct) vs
v5-mode (compat metadata) loading over 9 games shows all target-source fields
bit-identical, so emitted targets are equal by construction.

## 3. Rust tests and smoke

- `cargo test --locked -p libriichi --features sanma --lib`: 34 passed, 0 failed.
- `checks/native_v4_training_smoke.py` (runtime pyd rebuilt from the same
  source): `NATIVE_V4_TRAINING_SMOKE_OK samples=327 batch=8`, no
  `libriichi3p_compat` import.

## 4. Throughput benchmark

`checks/native_v4_throughput_benchmark.py`, evidence
`.cache/native-v4/throughput-benchmark-20260906.json`:

| encoder | samples | seconds | samples/s |
|---|---:|---:|---:|
| compat_v4 (reference replay) | 14,478 | 15.60 | 928 |
| native_v4 (Rust direct) | 14,478 | 1.90 | 7,615 |

**8.21× on the data pipeline** (40 fixed files, single process, raw mode,
frozen GRP targets included). End-to-end training uplift scales with the
share of the pipeline this replaces; the previous ~5,200 samples/s
steady-state observation remains the GPU-side reference until a full
native-v4 training run is measured.

## 5. Configuration

- `Mortal/config/sanma.toml` `[dataset]` now sets
  `observation_encoder = 'native_v4'`.
- Rollback: restore `native_version = 5` (instructions inline in the config).
- Evaluation, self-play and MahjongCopilot deployment paths are unchanged;
  they never consumed the offline dataloader.

## 6. Runtime binary

`Mortal/mortal/libriichi.pyd` rebuilt from this source
(`Mortal/native/sanma/libriichi-py312.pyd`, sha256 identical to the
gate-passing build). The prior runtime pyd (2026-09-01) lacked the
nukidora/EV fixes and must not be restored from backups.
