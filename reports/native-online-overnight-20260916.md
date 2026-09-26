# Native online continuation, 2026-09-16

## Authorized outcome

Continue the formal 4000 successful-update checkpoint to 10000 updates, then
run the existing fixed-budget legacy-compat final panels. Preserve originals
and baselines; no automatic deployment or promotion. The run was executed
unattended and supervised by a periodic heartbeat check.

## Live launch

- Repository: this repository (all paths below are repository-relative).
- Stage: `Mortal/training/online_native_self_20260916`
- Launch: 2026-09-16 00:38:20 Asia/Shanghai.
- Config: stage `config.toml`; isolated scripts and tested extension: stage `runtime/`.
- Device verified in trainer log: ROCm CUDA API, AMD Radeon RX 9070 XT.
- Port 5025; self-play native, TestPlayer and final evaluator remain compat.
- Output checkpoint `state.pth`, export `deployment.pth`.
- Logs: `runner.log`, `launcher.err`, `runner_logs/{trainer,client,server}.log`.
- Fresh empty buffer and drain; original historical data are not copied.

```powershell
.\.venv-rocm\Scripts\python.exe scripts/run_online_phase1_overnight.py --config Mortal/training/online_native_self_20260916/config.toml --runtime-dir Mortal/training/online_native_self_20260916/runtime --resume-from Mortal/training/online_dynamic_self_20260908/state.pth --prefix native_self_20260916 --seed-base 20261016
```

Already launched. Do not repeat this command. For a diagnosed stopped training
segment, replace `--resume-from ...` with `--resume`, retaining config/runtime,
prefix and seeds. Do not resume while runner or owned children are active.
The runner intentionally refuses completed/partial panel outputs; preserve them
and inspect before recovery rather than deleting them to satisfy preflight.

## Fixed experiment

- Source state SHA256: `94c8a29207b3d58c7d13e07d329c33f8216d786c724ec190faa86ab779779619`.
- Source deployment SHA256: `6047302179e0785e8766525f369af1177ddbfc93381dfed3ae9889afcacba67e`.
- Candidate extension SHA256: `16f13ca2ede139b2e1ac5592cba7eaa88b23adf5c2b8b9bb5e3c86518c514c5b`.
- Old installed extension SHA256: `6117493924c3a18d4291f9e7423688228ca4cd33f73b78f418eae96fdb86fd3d`.
- LR, reward, dataset, opponent weights and optimizer budget unchanged from saved config.
- Dataset preflight: 69213 files; shared existing index exactly matches globs.
- `runner_logs/preflight.json` holds all input, runtime hashes and output paths.
- Native samples still use reference-compatible offline observation encoding.
- save500, submit250, test4000, snapshots2500, target10000; next internal test8000.
- Final panel keys 20261016..20261019, prefix `native_self_20260916`.
- Champion: frozen 128x8 step46500, 3 independent 999-game blocks, total2997.
- Baseline: original mortal3p, 999 games.
- Each 1v2 match is one candidate against two copies of the named opponent.
- Final `result.json` promotion is recommendation only; no baseline files replaced.

## Checks and remaining work

Passed live this turn: online_self_check (native only for TrainPlayer, compat
evaluation/default, invalid mode rejection), online_runner_check (fork preserves
optimization and fresh-output protection), dry-run, git diff --check.
No new Rust build this turn; tested candidate reused. See native-online-20260916.md
for bounded optimizer-delta, replay legality and complete trajectory comparisons.

At00:41:57 the saved checkpoint was4500; all optimizer state counters advanced
exactly500 from the source. Source checkpoint SHA256 remained unchanged.
Client continued to version15 with matching current-opponent/trainee version;
training log advanced beyond4500. CPU regression checks online_throughput_check
and optimizer_update_budget_check also passed. Initial500 took approximately
3 minutes including warmup. Provisional remaining train+panel ETA1.5-2.5h.
The launcher-only RTK process4228 was stopped without its process tree to close
the idle tool terminal; the detached training runner continued normally.

Next: measure steady throughput and watch8000-step internal test.
After completion verify10000 checkpoint and optimizer delta6000 against original,
unchanged protected hashes, complete3996 logs and final significance statistics.
Write final Chinese conclusion distinguishing speed from strength. Clean only
verified task-owned idle processes and pause heartbeat when done.

## Final outcome

Training reached10000 steps. Native self-play was used only in the isolated
training runtime; final evaluation remained compat. The staged final panels
completed3996 games and were verified by the runner.

- Versus the 128x8 champion,2997 games: mean rank1.999666, 95% CI
  [1.979723,2.019610], effect -0.000334, not significant. Rank counts996/1006/995.
- Versus the original baseline,999 games: mean rank1.864865,95% CI
  [1.818438,1.911291], significant versus2.0, but it fails the configured
  baseline guard threshold1.81.
- Automatic promotion: false. No model or baseline was replaced.
- The runner completed at01:48:16. The result is in the stage `result.json`.

The native path materially improved throughput, but this run did not establish
a playing-strength improvement over the current champion.
