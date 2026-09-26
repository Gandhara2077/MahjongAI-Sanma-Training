import json
from pathlib import Path
import secrets

import numpy as np
import prelude
from config import config
from libriichi.arena import ThreeWay
from one_vs_two import load_engine


def require_positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'three_way.{name} must be a positive integer')
    return value


def prepare_log_dir(log_dir):
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f'three_way.log_dir is not empty: {log_dir}')
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def decode_rankings(raw_rankings):
    return np.asarray(
        [list(ranking) if isinstance(ranking, (bytes, bytearray)) else ranking
         for ranking in raw_rankings],
        dtype=np.int64,
    )


def summarize(rankings, pts):
    ranks = np.asarray(rankings, dtype=np.int64) + 1
    if ranks.ndim != 2 or ranks.shape[1] != 3 or ranks.shape[0] == 0:
        raise ValueError(f'expected a non-empty (games, 3) rank array, got {ranks.shape}')
    games = ranks.shape[0]
    ranking_vectors = [
        [int(np.count_nonzero(ranks[:, model] == rank)) for rank in (1, 2, 3)]
        for model in range(3)
    ]
    average_rank = ranks.mean(axis=0)
    average_points = np.asarray(ranking_vectors) @ np.asarray(pts) / games
    if games > 1:
        margin = 1.96 * ranks.std(axis=0, ddof=1) / np.sqrt(games)
    else:
        margin = np.zeros(3)
    return {
        'games': int(games),
        'ranking_vectors': ranking_vectors,
        'average_rank': [float(value) for value in average_rank],
        'average_points': [float(value) for value in average_points],
        'rank_ci95': [
            [float(mean - error), float(mean + error)]
            for mean, error in zip(average_rank, margin)
        ],
    }


def main():
    cfg = config['three_way']
    games_per_iter = require_positive_int('games_per_iter', cfg['games_per_iter'])
    if games_per_iter % 3 != 0:
        raise ValueError('three_way.games_per_iter must be a positive multiple of 3')
    iters = require_positive_int('iters', cfg['iters'])
    expected_games = games_per_iter * iters
    if expected_games < 3000:
        raise ValueError('three_way must schedule at least 3000 games')

    candidates = cfg['candidates']
    if len(candidates) != 3:
        raise ValueError('three_way.candidates must contain exactly three engines')
    pts = [int(value) for value in cfg.get('pts', [90, 0, -90])]
    if pts != [90, 0, -90]:
        raise ValueError(f'three_way.pts must be [90, 0, -90], got {pts}')

    key = cfg.get('seed_key', -1)
    if key == -1:
        key = secrets.randbits(64)
    engines = [load_engine(candidate) for candidate in candidates]
    log_dir = prepare_log_dir(Path(cfg['log_dir']))
    seeds_per_iter = games_per_iter // 3
    batch_rankings = []

    for iteration in range(iters):
        seed = 10000 + iteration * seeds_per_iter
        rankings = decode_rankings(
            ThreeWay(
                disable_progress_bar=False,
                log_dir=str(log_dir),
            ).py_vs_py(
                engines,
                (seed, key),
                seeds_per_iter,
            )
        )
        if rankings.shape != (games_per_iter, 3):
            raise RuntimeError(
                f'three_way returned {rankings.shape}, expected {(games_per_iter, 3)}'
            )
        batch_rankings.append(rankings)
        print(json.dumps({'iteration': iteration, **summarize(rankings, pts)}))

    summary = summarize(np.concatenate(batch_rankings), pts)
    summary.update({
        'seed_key': int(key),
        'games_per_iter': games_per_iter,
        'iters': iters,
        'seed_start': 10000,
        'seeds_per_iter': seeds_per_iter,
        'candidates': [
            {'name': candidate['name'], 'state_file': candidate['state_file']}
            for candidate in candidates
        ],
        'log_dir': str(log_dir),
        'expected_log_count': expected_games,
        'log_count': len(list(log_dir.glob('*.json.gz'))),
    })
    with (log_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps(summary, ensure_ascii=False))
    if summary['log_count'] != expected_games:
        raise RuntimeError(
            f'expected {expected_games} compressed logs, found {summary["log_count"]}'
        )


if __name__ == '__main__':
    main()
