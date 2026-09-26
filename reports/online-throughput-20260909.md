# Online throughput optimization — 2026-09-09

## Scope and baseline

User authorized throughput optimization and continuation of the existing 10k-update experiment. The original run was supply-limited: recent 30-game batches took 20–25 seconds to generate, then approximately 12 updates took 6 seconds; buffer size repeatedly returned to zero. Recent aggregate throughput was 0.514 updates/s. This was not a measured CPU-saturation or GPU-compute limit.

Paused only the verified launcher PID 17860 and its descendants. The complete checkpoint contains 1,000 training/optimizer updates; subsequent unsaved updates are replayed after resuming. The checkpoint is preserved at `.cache/online-throughput-20260909/frozen_state.pth`; its hash matched the original before resume. Temporary buffer/drain/train-play directories were copied into the same directory under `paused-*` before the server's normal restart cleanup. No baseline or historical assessment was deleted.

## Changes

- `online_training.reuse_expert_iterator=true`: reuse the expert iterator across online drains, restarting only after expert epoch exhaustion. Empty expert data fails rather than looping. The default remains false for other configurations.
- `online.drain_poll_seconds=0.2`: replace this experiment's 5-second empty-buffer polling delay. Default remains 5 seconds, and nonfinite/nonpositive intervals are rejected.
- Self-play games per batch: 30 -> 96. Keep compat max_workers=16, training batch=1024 and dataset workers=2.
- Add explicit `--resume` to the existing runner. It checks the original preflight inputs, data index, output paths and experiment settings; only the named throughput settings may differ. Evaluation seeds/paths must remain unchanged, and existing partial/completed evaluation output is rejected. Each resume gets a new manifest without overwriting the initial preflight.

No LR, GRP, model size, expert ratio, opponent weights, augmentation setting, update target or evaluation budget changes. Iterator reuse changes sample traversal/order; larger self-play batches also make parameter refresh less frequent. This is not a bit-for-bit equivalent trajectory or an isolated causal test of model strength.

## Isolated real self-play benchmark

Entry: `checks/online_throughput_benchmark.py`. Uses real TrainPlayer, frozen checkpoint and fixed anchor opponent, shared seed prefixes, original trainee exploration settings. No optimizer updates. One 3-game warmup plus two repetitions per case, 447 games total. Logs/results are outside production under `.cache/online-throughput-20260909/`.

| Games/batch | Workers | Seconds, two repetitions | Aggregate games/s |
|---:|---:|---|---:|
| 30 | 16 | 20.234, 22.016 | 1.4201 |
| 96 | 16 | 61.484, 60.172 | 1.5782 |
| 96 | 32 | 59.421, 53.438 | 1.7012 |

Selected 96/16: observed self-play throughput +11.1% over 30/16, with fewer loader restarts per game. 32 workers added only 7.8% over 96/16 and showed more variability, below the 10% threshold for raising concurrency. This is a short throughput probe, not a statistical confidence claim or a measurement of total training speedup. The batch-size cases contain different numbers of seed sets; worker-count cases share the same sets.

## Verification

- `checks/online_throughput_check.py`: expert stream continues across drains; restarts only on exhaustion; empty stream fails; original online-only mixing remains; configured/default poll intervals are exercised with a fake server.
- `checks/online_runner_check.py`: resume permits throughput flags but rejects changed inputs, output paths, dataset identity and learning rate; prior dry-run/error cleanup regressions remain.
- Other passing checks: online_panel_stats, online_self, optimizer_update_budget, online_state_resume, scheduler_resume.
- Changed Python files compile; `git diff --check` passes (line-ending warnings only).
- Real optimized GPU smoke: `.cache/online-self-throughput-smoke-20260909/result.json`, passed, RX 9070 XT, workers=2, 20 actual updates, 33 started games, 62.781 seconds, actual 10->20 checkpoint resume, one AMP skip excluded from the update budget. Self weights and trainee-only replay verification passed. Smoke output is not a production model.
- Production resume dry-run passed: 1000 -> 10000, original seed keys 20260980–20260983 unchanged.

## Active continuation

Config: `Mortal/config/sanma-online-128x8-dynamic-self-perf.toml`.

Launch command from project root (do not run while it is already active):

```powershell
rtk proxy .venv-rocm/Scripts/python.exe scripts/run_online_phase1_overnight.py --config Mortal/config/sanma-online-128x8-dynamic-self-perf.toml --resume --prefix sanma-online-128x8-dynamic-self --seed-base 20260980
```

Background launcher PID 25228. Runner confirmed checkpoint step 1000 at 23:33:54; trainer loaded it on RX 9070 XT at 23:34:00. Same production output directory, appended runtime logs, new resume manifest. Training still ends at 10000, followed by the original 2997-game champion and 999-game baseline panels, without automatic promotion.

Only the intended formal run remains active; the benchmark and bounded smoke completed. No new independent subagent review result is claimed. This iteration used controller code review, focused regressions and real GPU checks.

## Follow-up — 2026-09-10

The optimized run advanced to approximately 3542 before its logs stopped at 00:34:55. At 20:47, process inspection found no surviving project training processes (the only matches during the second query were our new dry-run). No completed result exists. The available logs do not establish why the processes exited; do not attribute this to a GPU driver, machine shutdown, or code deadlock without further evidence.

Across the optimized logged window after 23:36 until 00:34:49, measured aggregate throughput was 0.6988 updates/s, versus the earlier baseline window's 0.5138 updates/s: approximately +36%. These are observational production windows, not controlled paired benchmarks. The logs also show short active bursts around 6–12 updates/s, so self-play supply still limits overall utilization; the GPU is not continuously saturated.

Checkpoint 3500 loaded successfully. Before recovery, it and buffer/drain/train-play data were archived under `.cache/online-perf-resume-20260910/`. Resume dry-run confirmed step 3500 -> 10000 and the same four evaluation seed keys. A new background launcher PID 4612 was started with the same performance configuration; its dedicated stdout/stderr are `.cache/dynamic-self-perf-20260910.*.log`. Up to approximately 42 unsaved updates from the prior process are not present in the checkpoint and must be redone. No new hyperparameter changes were made.

At 20:51:37 the resumed trainer finished 44 additional successful updates (step 3544) and submitted new parameters. The launch stderr was empty at this check. This confirms actual post-resume progress, not merely process creation; the 10k training and final evaluations are still pending.
