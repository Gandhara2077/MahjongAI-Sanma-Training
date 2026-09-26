"""Paired comparison of two 1v2 arena evaluation directories.

Both directories must come from panels that share the same seed values and
seat rotation (e.g. the multiseed selection TOMLs with a common seed_key).
Games are paired by (seed, seat), the rank difference is averaged within each
seed, and the seed-level mean difference gets a 95% CI. A negative mean
difference means candidate A (first directory) ranks better.

Usage:
    python checks/stats/arena_log_paired.py <dir_a> <dir_b> [<label_a> <label_b>]
"""

import glob
import gzip
import json
import math
import os
import re
import sys

from arena_log_results import game_result, rank_of

FILE_PATTERN = re.compile(r'(\d+)_(\d+)_([abc])\.json\.gz$')


def ranks_by_key(directory):
    out = {}
    for path in glob.glob(os.path.join(directory, '*.json.gz')):
        match = FILE_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        seed, _, seat = match.groups()
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            events = [json.loads(line) for line in file if line.strip()]
        result = game_result(events)
        if result is None:
            continue
        seat_index, scores = result
        out[(int(seed), seat)] = rank_of(seat_index, scores)
    return out


def main():
    if len(sys.argv) < 3:
        print('usage: python checks/stats/arena_log_paired.py <dir_a> <dir_b> [<label_a> <label_b>]')
        return 1
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    label_a = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(dir_a.rstrip('/\\'))
    label_b = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(dir_b.rstrip('/\\'))

    ranks_a = ranks_by_key(dir_a)
    ranks_b = ranks_by_key(dir_b)
    common = sorted(set(ranks_a) & set(ranks_b))
    if not common:
        print('no common (seed, seat) games between the two directories')
        return 1

    diffs_by_seed = {}
    for key in common:
        diffs_by_seed.setdefault(key[0], []).append(ranks_a[key] - ranks_b[key])
    seed_means = [sum(values) / len(values) for values in diffs_by_seed.values()]
    mean = sum(seed_means) / len(seed_means)
    variance = sum((value - mean) ** 2 for value in seed_means) / (len(seed_means) - 1)
    se = math.sqrt(variance / len(seed_means))
    low, high = mean - 1.96 * se, mean + 1.96 * se
    better_fraction = sum(1 for value in seed_means if value < 0) / len(seed_means)

    print(f'paired {label_a} - {label_b}: games={len(common)} seeds={len(seed_means)}')
    print(f'  mean rank diff = {mean:+.4f} (CI95 {low:+.4f}~{high:+.4f}), '
          f'A better on {better_fraction:.1%} of seeds')
    print(f'  verdict: {"A significantly better" if high < 0 else "B significantly better" if low > 0 else "no significant difference"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
