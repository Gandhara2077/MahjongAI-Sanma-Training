"""Compute 1v2 arena results from compressed MJAI game logs.

Each *.json.gz file is one game. The challenger seat is the one whose name is
not 'mortal-baseline'. Final scores are reconstructed by replaying score
carries: each start_kyoku carries absolute scores, and hora/ryuukyoku deltas
within that kyoku are applied cumulatively, so the state after the last kyoku
is the game-end score.

Usage:
    python checks/stats/arena_log_results.py <dir> [<dir> ...]

Validation reference (2026-08-29 logs):
    training/sanma_1v2                -> 213/189/198, avg_rank 1.9750, +2.25
    training/sanma_1v2_eval_step17800 -> avg_rank 1.9533, +4.20
"""

import glob
import gzip
import json
import math
import os
import sys
from collections import Counter

BASELINE_NAME = 'mortal-baseline'
POINTS = [90, 0, -90]


def game_result(events, challenger_name=None):
    """Return the challenger seat and final scores of one game, or None."""
    if not events or events[0].get('type') != 'start_game':
        return None
    names = events[0].get('names', [])
    if len(names) != 3:
        return None
    if challenger_name is not None:
        if events[-1].get('type') != 'end_game' or any(not isinstance(name, str) for name in names):
            raise ValueError('incomplete game or invalid player names')
        if sum(event.get('type') == 'start_game' for event in events) != 1 or sum(
            event.get('type') == 'end_game' for event in events
        ) != 1:
            raise ValueError('duplicate game boundary')
        for event in events:
            if event.get('type') == 'start_kyoku':
                values = event.get('scores')
                if not isinstance(values, list) or len(values) != 3 or any(type(value) is not int for value in values):
                    raise ValueError('invalid scores')
                pot = event.get('kyotaku', 0)
                if type(pot) is not int or pot < 0:
                    raise ValueError('invalid kyotaku')
            if 'deltas' in event or event.get('type') in ('hora', 'ryukyoku'):
                values = event.get('deltas')
                if not isinstance(values, list) or len(values) != 3 or any(type(value) is not int for value in values):
                    raise ValueError('invalid deltas')
        challenger_seats = [i for i, name in enumerate(names) if name == challenger_name]
        if len(challenger_seats) != 1:
            raise ValueError('challenger name must match exactly one seat')
    else:
        challenger_seats = [i for i, name in enumerate(names) if name != BASELINE_NAME]
    seat = challenger_seats[0] if challenger_seats else 0

    scores = None
    kyotaku = 0
    for event in events:
        if event.get('type') == 'start_kyoku':
            carried = event.get('scores')
            if not carried or len(carried) != 3:
                return None
            scores = list(carried)
            kyotaku = event.get('kyotaku', 0)
        elif event.get('type') == 'reach_accepted':
            actor = event.get('actor')
            if scores is None or type(actor) is not int or not 0 <= actor < 3:
                return None
            scores[actor] -= 1000
            kyotaku += 1
        elif scores is not None and 'deltas' in event:
            deltas = event['deltas']
            if len(deltas) == 3:
                scores = [scores[k] + deltas[k] for k in range(3)]
            if event.get('type') == 'hora':
                kyotaku = 0  # The winner's delta already includes the pot.
        elif event.get('type') == 'end_game' and scores is not None and kyotaku:
            # Match arena/game.rs: highest score, lowest seat on a tie.
            scores[max(range(3), key=lambda seat: scores[seat])] += kyotaku * 1000
            kyotaku = 0
    if scores is None:
        return None
    return seat, scores


def rank_of(seat, scores):
    return 1 + sum(1 for j, s in enumerate(scores) if s > scores[seat]) + sum(
        1 for j, s in enumerate(scores) if s == scores[seat] and j < seat
    )


def eval_dir(directory):
    files = sorted(glob.glob(os.path.join(directory, '*.json.gz')))
    ranks = Counter()
    games = 0
    for path in files:
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            events = [json.loads(line) for line in file if line.strip()]
        result = game_result(events)
        if result is None:
            continue
        seat, scores = result
        ranks[rank_of(seat, scores)] += 1
        games += 1
    if games == 0:
        return None
    avg_rank = sum(rank * count for rank, count in ranks.items()) / games
    avg_pt = sum(count * POINTS[rank - 1] for rank, count in ranks.items()) / games
    variance = sum(count * (rank - avg_rank) ** 2 for rank, count in ranks.items()) / (games - 1)
    margin = 1.96 * math.sqrt(variance / games)
    return {
        'games': games,
        'ranks': [ranks.get(1, 0), ranks.get(2, 0), ranks.get(3, 0)],
        'avg_rank': round(avg_rank, 4),
        'avg_pt': round(avg_pt, 2),
        'ci95': round(margin, 4),
        'rank_ci95_low': round(avg_rank - margin, 4),
        'rank_ci95_high': round(avg_rank + margin, 4),
        'pt_ci95': round(1.96 * math.sqrt(
            sum(count * (POINTS[rank - 1] - avg_pt) ** 2 for rank, count in ranks.items()) / (games - 1) / games
        ), 2),
    }


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print('usage: python checks/stats/arena_log_results.py <dir> [<dir> ...]')
        return 1
    for directory in dirs:
        result = eval_dir(directory)
        label = os.path.basename(directory.rstrip('/\\'))
        if result is None:
            print(f'{label}: no parseable games')
            continue
        print(
            f"{label}: n={result['games']} ranks={result['ranks']} "
            f"avg_rank={result['avg_rank']} (CI95 {result['rank_ci95_low']}~{result['rank_ci95_high']}) "
            f"avg_pt={result['avg_pt']} (CI95 ±{result['pt_ci95']})"
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
