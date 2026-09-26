import argparse
import prelude

import numpy as np
import secrets
import torch
from config import config
from engine import MortalEngine
from libriichi.arena import OneVsTwo
from libriichi3p_compat import CompatMjaiEngine
from model import Brain, DQN


def positive_int(value):
    try:
        value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-workers', type=positive_int)
    parser.add_argument('--games-per-iter', type=positive_int)
    parser.add_argument('--iters', type=positive_int)
    parser.add_argument('--log-dir')
    return parser.parse_args(argv)


def load_engine(cfg, max_workers=None):
    state = torch.load(
        cfg['state_file'],
        weights_only=True,
        map_location=torch.device('cpu'),
    )
    model_cfg = state['config']
    version = model_cfg['control']['version']
    mortal = Brain(version=version, **model_cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    if cfg['enable_compile']:
        mortal.compile()
        dqn.compile()

    delegate = MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(cfg['device']),
        enable_amp=cfg['enable_amp'],
        enable_rule_based_agari_guard=cfg['enable_rule_based_agari_guard'],
        name=cfg['name'],
    )
    compat = config.get('compat', {})
    reference_path = compat.get('libriichi3p') or None
    return CompatMjaiEngine(
        delegate,
        reference_path=reference_path,
        max_workers=(
            max_workers
            if max_workers is not None
            else compat.get('max_workers', 64)
        ),
    )


def main():
    args = parse_args()
    cfg = config['1v2']
    games_per_iter = (
        args.games_per_iter
        if args.games_per_iter is not None
        else cfg['games_per_iter']
    )
    if games_per_iter % 3 != 0:
        raise ValueError('1v2.games_per_iter must be divisible by 3')
    seeds_per_iter = games_per_iter // 3
    iters = args.iters if args.iters is not None else cfg['iters']
    log_dir = args.log_dir if args.log_dir is not None else cfg['log_dir']
    compat = config.get('compat', {})
    max_workers = (
        args.max_workers
        if args.max_workers is not None
        else compat.get('max_workers', 64)
    )
    key = cfg.get('seed_key', -1)
    if key == -1:
        key = secrets.randbits(64)

    champion = load_engine(cfg['champion'], max_workers=max_workers)
    challenger = load_engine(cfg['challenger'], max_workers=max_workers)
    display_pts = np.asarray(cfg.get('pts', [90, 0, -90]))

    seed_start = 10000
    for i, seed in enumerate(
        range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)
    ):
        print('-' * 50)
        print('#', i)
        rankings = np.asarray(
            OneVsTwo(
                disable_progress_bar=False,
                log_dir=log_dir,
            ).py_vs_py(
                challenger=challenger,
                champion=champion,
                seed_start=(seed, key),
                seed_count=seeds_per_iter,
            )
        )
        avg_rank = rankings @ np.arange(1, 4) / rankings.sum()
        avg_pt = rankings @ display_pts / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
