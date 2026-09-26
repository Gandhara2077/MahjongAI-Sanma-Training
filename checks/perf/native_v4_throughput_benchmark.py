"""Fixed-file before/after throughput benchmark: compat_v4 vs native_v4.

Measures the real dataloader populate_buffer path (Rust metadata load +
observation encoding + reference replay where applicable + Python target
calculation) over a fixed 40-file corpus, single process, raw mode.

Usage:
    python checks/perf/native_v4_throughput_benchmark.py [--output PATH]

Both encoders consume the same file list and the same frozen GRP, so the
samples/s ratio is the pipeline speedup a worker would see.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_PKG = REPO_ROOT / 'Mortal' / 'mortal'
sys.path.insert(0, str(MORTAL_PKG))
sys.path.insert(0, str(REPO_ROOT / 'checks'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path)
parser.add_argument('--native-extension', type=Path,
                    default=REPO_ROOT / 'Mortal' / 'mortal' / 'libriichi.pyd')
options = parser.parse_args() if __name__ == '__main__' else parser.parse_args([])

if options.native_extension.suffix == '.dll':
    loader = importlib.machinery.ExtensionFileLoader(
        'libriichi', str(options.native_extension))
    spec = importlib.util.spec_from_file_location(
        'libriichi', options.native_extension, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules['libriichi'] = module

from libriichi.consts import obs_shape  # noqa: E402
assert obs_shape(4) == (775, 34), f'unexpected obs shape {obs_shape(4)}'

from dataloader import FileDatasetsIter  # noqa: E402
from model import GRP  # noqa: E402
from reward_calculator import RewardCalculator  # noqa: E402

SAMPLE_MONTH_DAYS = [('01', '04'), ('03', '15'), ('05', '20'), ('07', '11'), ('08', '09')]
FILES_PER_MONTH = 8
MJJSON_ROOT = REPO_ROOT / 'koromo' / 'mjson' / '2026'
GRP_STATE = REPO_ROOT / 'baselines' / 'sanma_grp_10k_last_frozen.pth'
REFERENCE_PATH = REPO_ROOT / '.cache' / 'libriichi3p' / \
    'libriichi3p-3.12-x86_64-pc-windows-msvc.pyd'
PTS = [90.0, 0.0, -90.0]


def sample_files():
    files = []
    for month, day in SAMPLE_MONTH_DAYS:
        day_files = sorted((MJJSON_ROOT / month / day).glob('*.mjson'))
        files.extend(day_files[:FILES_PER_MONTH])
    return [str(p) for p in files]


def build_iter(encoder, file_list):
    it = FileDatasetsIter(
        version=4,
        file_list=list(file_list),
        pts=PTS,
        oracle=False,
        file_batch_size=8,
        reserve_ratio=0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        reference_path=str(REFERENCE_PATH),
        observation_encoder=encoder,
    )
    grp = GRP(hidden_size=64, num_layers=2)
    grp_state = torch.load(GRP_STATE, weights_only=True, map_location='cpu')
    grp.load_state_dict(grp_state['model'])
    it.grp = grp
    it.reward_calc = RewardCalculator(grp, PTS)
    # Mirror dataloader.load_files' loader construction so populate_buffer
    # can be driven directly without the epoch/buffer plumbing.
    from libriichi.dataset import GameplayLoader
    it.loader = GameplayLoader(
        version=it.native_version if it.use_compat_replay else it.version,
        oracle=False,
        player_names=None,
        excludes=None,
        augmented=False,
        encode_observations=not it.use_compat_replay,
    )
    it._metadata_only_loader = it.use_compat_replay
    it.buffer = []
    return it


def run_pass(it, file_list):
    total = 0
    started = time.perf_counter()
    for start in range(0, len(file_list), it.file_batch_size):
        batch = file_list[start:start + it.file_batch_size]
        it.buffer.clear()
        it.populate_buffer(batch, augmented=False)
        total += len(it.buffer)
    elapsed = time.perf_counter() - started
    return total, elapsed


def main():
    files = sample_files()
    assert len(files) == 40, f'expected 40 files, got {len(files)}'
    results = {}

    for encoder in ('compat_v4', 'native_v4'):
        it = build_iter(encoder, files)
        # Warm-up on the first batch (loader/bot/rng init).
        it.populate_buffer(files[:it.file_batch_size], augmented=False)
        samples, elapsed = run_pass(it, files)
        results[encoder] = {
            'samples': samples,
            'seconds': round(elapsed, 3),
            'samples_per_sec': round(samples / elapsed, 1),
        }
        print(f'{encoder}: {samples} samples in {elapsed:.2f}s '
              f'-> {samples / elapsed:.0f} samples/s')

    speedup = results['native_v4']['samples_per_sec'] / results['compat_v4']['samples_per_sec']
    results['speedup_native_vs_compat'] = round(speedup, 3)
    results['file_count'] = len(files)
    results['mode'] = 'raw, single process, frozen GRP targets included'
    results['native_extension'] = str(options.native_extension)
    print(f'speedup native_v4 vs compat_v4: {speedup:.2f}x')

    if options.output is not None:
        options.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        print(f'evidence written: {options.output}')


if __name__ == '__main__':
    main()
