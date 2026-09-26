# Native online reuse: isolated candidate

## Scope and implementation

Reuse existing v4 encoder and MortalBatchAgent. Only production source changed in this stage:
`Mortal/libriichi/src/agent/mortal.rs`.

- Online sanma mask[40] is intersected with native can_nukidora.
- Forced-discard quick evaluation cannot bypass a legal North extraction.
- Offline PlayerState.encode_obs and GameplayLoader are unchanged.
- No production config, installed pyd, model, logs, or promotion policy was replaced.

## Correctness and builds

Two colocated Rust tests exercised actual set_scene and its queued observations/masks.
Both failed before the fix and passed after it. One preserves the complete observation and all mask bits other than the invalid North action; the other checks a legal North remains a model decision with quick evaluation enabled.

- Sanma library: 36 passed, 0 failed, 9 inapplicable yonma fixtures ignored.
- Yonma library: 29 passed, 0 failed.
- Offline release build succeeded (2 build jobs). Existing unused-variable warnings remain; no unrelated cleanup.
- First test launch lacked the Python DLL in PATH. Re-run with Python312 in child PATH and PYO3_PYTHON set to the project interpreter executed the real red/green tests.
- Python smoke script compiled successfully.

Build candidate: `Mortal/target/release/riichi.dll`.
SHA256: `5f10045207bca8824f929311240b0c5679a03635031337782861b54aa6e58d5f`.
Both installed pyd files retain SHA256 `6117493924c3a18d4291f9e7423688228ca4cd33f73b78f418eae96fdb86fd3d`.

## Arena smoke (synthetic policy, NOT trained-model performance)

`checks/native_online_smoke.py` uses the real native arena and a deterministic legal-action policy, without torch/GPU/optimizer updates. Each engine owns its own compat bridge. Explicit candidate path and a fresh output directory are required. Historical reference comes from the existing project cache.

Fixed seed start `(17000, 2026091017)`, 10 seeds x 3 seats:

| Path | Games | Seconds | North actions | Decisions with metadata |
|---|---:|---:|---:|---:|
| native, quick off, initial | 30 | 2.640 | 669 | 10320 |
| native, quick off, repeat | 30 | 2.719 | 669 | 10320 |
| compat, quick off | 30 | 25.735 | 669 | 10320 |
| native, quick on | 30 | 2.657 | 669 | 10035 |

Each run had 30 complete gzip logs, valid metadata mask/value lengths, and rank totals of 30. After removing metadata, every event in all 30 native-repeat logs matched its compat counterpart, and all 30 matched quick-on logs. This is trajectory parity for this policy and sample, not proof of complete observation parity or neural-policy parity.

Artifacts: `.cache/native-online-20260910/{native,native-r1,compat-r1,native-quick}/result.json` and `games/`.
The first compat launch only created its fresh directory, then failed reference discovery; the runner now supplies the existing cache explicitly and the successful rerun is `compat-r1`. No output directory was overwritten.

Repeat from project root (choose a new output each time):

```powershell
rtk proxy .venv-rocm/Scripts/python.exe checks/native_online_smoke.py --extension Mortal/target/release/riichi.dll --output .cache/native-online-next/native --seed-count 10
rtk proxy .venv-rocm/Scripts/python.exe checks/native_online_smoke.py --extension Mortal/target/release/riichi.dll --output .cache/native-online-next/compat --backend compat --seed-count 10
```

## Deployment gate / remaining work

The roughly 9.5x synthetic timing difference is evidence of avoidable bridge overhead, NOT a promised training speedup. Production was still running during this CPU smoke; the neural workload, actual batching, and GPU contention are absent from this comparison.

Before switching production:

1. Freeze a real checkpoint and compare native versus compat in separate sequential processes, same seeds, same quick/agari settings, explicit candidate extension, fresh logs, no competing GPU job.
2. Check real-policy decision/observation discrepancies (including North/riichi/kan/ron), not just aggregate rank.
3. Exercise TrainPlayer, dynamic Self synchronization, trainee-only data selection, replay metadata, and a bounded optimizer smoke with the candidate before changing the production engine selector.
4. Measure end-to-end updates/s and retain compat fallback. Do not change rewards/LR/opponents in the performance experiment. No automatic promotion.

Production runner logged a completed 4000-step segment at 21:06:52 and subsequent Python processes existed; this stage did not interrupt it. GPU A/B has not been run. Controller reviewed the focused diff; no independent reviewer result is claimed. No commit or push.
