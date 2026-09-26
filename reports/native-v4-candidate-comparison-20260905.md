# Native v4 candidate comparison — corrected 2026-09-06

## Scope and reproducibility

Read-only replay of the existing archive:

`Mortal/training/sanma_three_way_baselines_128x8_lr2_192x12best_20260918`

The suffix and `seed_key=20260918` identify the experiment, not a verified execution date.
No new games were simulated. Candidates in summary order:

1. sanma-128x8-champion-46k
2. sanma-lr2-champion-40k
3. sanma-192x12-best-92k

Run from the repository root:

`rtk proxy .venv-rocm/Scripts/python.exe checks/three_way_paired_check.py Mortal/training/sanma_three_way_baselines_128x8_lr2_192x12best_20260918`

## Corrected score accounting

The previous report is superseded. The shared Python parser omitted the 1,000-point deduction at `reach_accepted`, whereas native `arena/board.rs` deducts it. It now tracks riichi deposits, applies terminal score deltas, clears the pot on hora, and awards any pot remaining at end_game according to native `arena/game.rs`: highest score, lowest seat on a tie. The previous claim that carry goes to the lowest-score player was wrong (`min_by_key(-score)` selects the highest).

The corrected replay matches all three native summary ranking vectors exactly. Summary disagreement is now an error, not a warning followed by a promotion verdict. The older paired 1v2 script also uses the same corrected parser; historical Python-derived reports should be regenerated before reuse. Existing log files and native summaries were not changed.

## Results

| Candidate | 1st / 2nd / 3rd | Average rank |
|---|---|---:|
| 128x8 champion | 1117 / 1008 / 875 | 1.919333 |
| LR2 40k | 754 / 965 / 1281 | 2.175667 |
| 192x12 best92k | 1129 / 1027 / 844 | 1.905000 |

3,000 games cover 1,000 seeds (10000 through 10999), each with all three seat rotations. The statistical unit is the seed's mean rank difference across its three games, not an independent individual game. The interval uses the sample variance of the 1,000 seed means and 1.96 standard errors.

192x12 minus 128x8 mean rank difference: **-0.014333**.
95% paired interval: **[-0.059654, +0.030987]**.
448 seeds favor 192x12, 407 favor 128x8, 145 tie.

**No promotion:** the interval includes zero. This panel does not establish that 192x12 is stronger than 128x8, nor does it establish equivalence. The old estimate (-0.012000) was based on the incomplete score replay and must not be reused.

## Verification

- `checks/three_way_paired_check.py --self-check`: passed. Includes riichi deduction, final pot award, missing/duplicate rotations, bad gzip and truncated logs.
- Real archive replay: exit 0, 3,000 games, 1,000 seeds, native summary vectors matched.
- Game completeness checks require settled rounds, end_kyoku and final end_game; a start_game/start_kyoku-only fixture is rejected.
- No training, simulated matches, model promotion, native-rule changes, or log deletion performed for this analysis.
