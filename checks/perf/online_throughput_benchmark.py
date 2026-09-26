"""Bounded real TrainPlayer throughput probe; no optimizer updates or promotion."""
from pathlib import Path
import argparse
import json
import os
import time
import sys
import shutil
import toml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.cache') or output == ROOT / '.cache' or output.exists():
        raise ValueError('output must be a fresh directory below project .cache')
    output.mkdir()
    cfg = toml.load(ROOT / 'Mortal/config/sanma-online-128x8-dynamic-self.toml')
    checkpoint = output / 'frozen_state.pth'
    shutil.copyfile(ROOT / 'Mortal' / cfg['control']['state_file'], checkpoint)
    cfg['train_play']['default']['log_dir'] = str(output / 'warmup')
    cfg['train_play']['default']['games'] = 3
    cfg['online_training']['opponents'] = [cfg['online_training']['opponents'][1]]
    config_path = output / 'config.toml'
    config_path.write_text(toml.dumps(cfg), encoding='utf-8')
    os.environ['MORTAL_CFG'] = str(config_path)
    os.chdir(ROOT / 'Mortal')
    sys.path.insert(0, str(ROOT / 'Mortal/mortal'))
    import prelude
    import torch
    from config import config
    from model import Brain, DQN
    from player import TrainPlayer
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm GPU required')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    device = torch.device('cuda:0')
    brain = Brain(version=4, **cfg['resnet']).to(device).eval()
    dqn = DQN(version=4).to(device).eval()
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    player = TrainPlayer()
    player.sync_opponents(state['mortal'], state['current_dqn'], 'throughput:0')
    player.train_key = 2026099101  # Probe seeds never used for promotion.
    player.train_play(brain, dqn, device)
    rows = []
    for games, workers in ((30, 16), (96, 16), (96, 32)):
        for repetition in range(2):
            config['compat']['max_workers'] = workers
            player.opponent_engines[0].engine.max_workers = workers
            player.seed_count = games // 3
            player.train_seed = 10000 + repetition * 100
            player.log_dir = str(output / f'g{games}-w{workers}-r{repetition}')
            torch.cuda.synchronize()
            started = time.monotonic()
            ranks, files = player.train_play(brain, dqn, device)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            assert len(files) == games and int(sum(ranks)) == games
            row = dict(games=games, workers=workers, repetition=repetition,
                       seconds=elapsed, games_per_second=games / elapsed)
            rows.append(row)
            (output / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print('THROUGHPUT ' + json.dumps(row), flush=True)
    print('THROUGHPUT_BENCHMARK_OK', flush=True)


if __name__ == '__main__':
    main()
