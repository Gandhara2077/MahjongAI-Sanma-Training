# Native v4 rollout - 2026-09-05

Approved scope: validate the existing Rust CPU pipeline, retaining compat_v4
as production default until correctness and throughput gates pass. No long
training, model replacement, Git commit/push, or corpus deletion.

## Tasks

- [ ] Freeze model/GRP/native/reference hashes and dirty source baseline.
- [ ] Compare 192x12-best and 128x8 by paired seed blocks from archived logs.
- [ ] Reproduce and classify v4 observation and mask differences.
- [ ] Build separately without replacing the active libriichi.pyd.
- [ ] Fix isolated compatibility defects with strict regression checks.
- [ ] Validate full samples, sanma/yonma regressions, and training smoke.
- [ ] Run independent end-to-end A/B; prepare opt-in native configuration.
- [ ] Independent review and report before considering online pilot work.

## Decisions

- Preserve all existing edits; work on stage/04-training, not main.
- No new Git branch/worktree without consent. Use a separate Cargo target and
  explicitly load its binary; do not activate it in the production directory.
- Separate compatibility fixes from rules/ranking changes. Rule corrections
  require independent evidence and refreshed evaluation provenance.
- Initial narrowed gate: 2141 samples, 928 observation and 14 mask mismatches.
- Paired analysis is independent and owns only its check/report.
- native_v4 must not be combined with native_version=5. No speed claim before parity.
