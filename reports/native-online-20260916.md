# Native online validation, 2026-09-16

## Result

The isolated native validation stage is complete. No model promotion or production
training restart was performed. The production checkpoint remains at 4000 updates.

| Frozen 4000-step policy, 30 games | Seconds | Games/s | Relative throughput |
|---|---:|---:|---:|
| compat, quick off | 20.188 | 1.486 | 1.00 |
| native, quick off | 8.062 | 3.721 | 2.50 |
| native, quick on | 6.906 | 4.344 | 2.92 |

These runs used the same weights, seeds, AMP and agari settings, sequentially.
Every event of all 30 games matched compat in both native runs after removing
metadata only. This is a sampled trajectory check, not exhaustive rule parity or
a playing-strength evaluation. The policy plays all three seats.

| Online resume, 4000 to 4300 | Seconds including startup | Updates/s |
|---|---:|---:|
| compat | 468.656 | 0.640 |
| native (default quick on) | 140.515 | 2.135 |

Observed end-to-end throughput ratio: 3.335. Wall-clock reduction: 70.0%.
Both runs use batch 1024, two data workers, 96-game batches, the same copied
checkpoint, expert mixture, opponent weights, frozen GRP and 10000-step LR
schedule. Each starts with empty buffer/drain directories and runs alone on GPU.
Their random game seeds and opponent draws differ; this is one operational
comparison, not a paired estimate or steady-state confidence interval. Native
quick evaluation is part of the measured switch. Do not extrapolate the earlier
synthetic ~9.5x result to neural training.

## Runtime checks

- Native 20-update smoke: 84.735 seconds, optimizer delta exactly 20.
- Native 300-update smoke: optimizer delta exactly 300; 6 distinct actual Self
  weight digests; 6 observed Self batches; final 96 drained games replayed.
- Compat 300-update control: optimizer delta exactly 300; 8 distinct actual Self
  weight digests; 7 observed Self batches; final 96 drained games replayed.
- In all replay checks, exactly one trainee player is selected per game and all
  extracted action labels are legal in their encoded masks.
- Saved scheduler max_steps remains 10000. Production state, deployment and
  installed extension hashes were unchanged during the isolated runs.
- All bounded runners exited successfully and performed owned-process cleanup.
- CPU checks: native_online_resume_check, online_runner_check,
  online_throughput_check and optimizer_update_budget_check passed.
- git diff --check passed; existing LF/CRLF warnings remain.
- No new Rust source changes/build in this stage. Candidate was built September 14.
- Controller review only; no independent reviewer result is claimed.

## Repaired earlier smoke failure

September 14 client.log shows MIOpen LockFile access denied in the user database
lock directory, during client neural inference. It was not a proven trainer
backward failure. GPU execution with cache access succeeded; no driver reinstall
or cache deletion was necessary.

The previous smoke helper overwrote Mortal/mortal/libriichi.pyd. Its candidate
hash was verified, no Mahjong processes were running, and the installed copy was
restored from Mortal/native/sanma/libriichi-py312.pyd. The repaired helper uses an
independent script directory and candidate extension, including spawn workers,
derives the start step from the checkpoint, preserves the scheduler, starts with
fresh self-play data, verifies optimizer deltas, and writes failure results.

Candidate SHA256: 16f13ca2ede139b2e1ac5592cba7eaa88b23adf5c2b8b9bb5e3c86518c514c5b

Installed SHA256: 6117493924c3a18d4291f9e7423688228ca4cd33f73b78f418eae96fdb86fd3d

Production state SHA256: 94c8a29207b3d58c7d13e07d329c33f8216d786c724ec190faa86ab779779619

## Artifacts and repeatability

Project-relative root: .cache/native-online-20260916/

- resume-smoke/: native 20 updates, result.json and verification.json.
- resume-300/: native 300 updates, result.json and verification.json.
- compat-300/: compat 300 updates, result.json and verification.json.
- neural-native/, neural-native-quick/, neural-compat/: frozen-model arena logs.
- Native arena directories also contain comparison.json with 30/30 identical.

Use a fresh output directory for every run:

```powershell
rtk proxy .venv-rocm/Scripts/python.exe checks/native_online_resume_smoke.py --output .cache/next-native --updates 300
rtk proxy .venv-rocm/Scripts/python.exe checks/native_online_resume_verify.py --output .cache/next-native
rtk proxy .venv-rocm/Scripts/python.exe checks/native_online_resume_smoke.py --output .cache/next-compat --updates 300 --backend compat
```

## Next production stage

The measured gain supports native adoption. Preserve the original 4000-step
state, continue in a separately named production stage with explicit candidate
identity, and keep reward/LR/data/opponents unchanged. Do not promote either
short benchmark checkpoint. Preserve the historical compat evaluation panel and
its seeds so training acceleration is not confused with a change of evaluator.
Observe the first production save boundary before launching an unattended long
run. The current isolated-stage plan explicitly excludes production switching.
