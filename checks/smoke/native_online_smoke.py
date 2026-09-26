"""Isolated native/compat arena check; never installs a binary or trains a model."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import native_sanma_smoke

ROOT = Path(__file__).resolve().parents[2]


class LegalPolicy:
    engine_type = "mortal"
    name = "legal-smoke"
    is_oracle = False
    version = 4
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def react_batch(self, obs, masks, invisible_obs):
        assert invisible_obs is None
        assert len(obs) == len(masks) > 0
        actions, values = [], []
        for feature, mask in zip(obs, masks):
            assert feature.shape == (775, 34) and mask.shape == (44,)
            legal = np.flatnonzero(mask)
            assert len(legal), "empty legal mask"
            action = next((a for a in (41, 40, 37, 39, 43) if mask[a]), int(legal[0]))
            q = np.full(44, -1e9, dtype=np.float32)
            q[mask] = 0
            q[action] = 1
            actions.append(action)
            values.append(q.tolist())
        return actions, values, [m.tolist() for m in masks], [True] * len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['native', 'compat'], default='native')
    parser.add_argument('--seed-count', type=int, default=10)
    parser.add_argument('--quick-eval', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='use frozen neural policy on ROCm instead of synthetic policy')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.cache') or output == ROOT / '.cache':
        raise ValueError('output must be below project .cache')
    if not 1 <= args.seed_count <= 100:
        raise ValueError('smoke seed-count must be 1..100')
    if args.quick_eval and args.backend == 'compat':
        raise ValueError('compat disables quick eval; compare with quick eval off')
    native_sanma_smoke.EXTENSION = args.extension.resolve(strict=True)
    native = native_sanma_smoke.load_extension()
    assert native.consts.NUM_PLAYERS == 3 and native.consts.ACTION_SPACE == 44
    assert tuple(native.consts.obs_shape(4)) == (775, 34)
    output.mkdir(parents=True, exist_ok=False)
    policies = [LegalPolicy(), LegalPolicy()]
    if args.checkpoint:
        import torch
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError('neural smoke requires ROCm')
        torch.set_num_threads(1)
        sys.path.insert(0, str(ROOT / 'Mortal/mortal'))
        from model import Brain, DQN
        from engine import MortalEngine
        state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        cfg = state['config']
        assert cfg['control']['version'] == 4
        policies = []
        for _ in range(2):
            brain = Brain(version=4, **cfg['resnet']).eval()
            dqn = DQN(version=4).eval()
            brain.load_state_dict(state['mortal'])
            dqn.load_state_dict(state['current_dqn'])
            policies.append(MortalEngine(brain, dqn, False, 4, device=torch.device('cuda:0'),
                                         enable_amp=True, name='frozen-neural-smoke'))
    for policy in policies:
        policy.enable_quick_eval = args.quick_eval
    if args.backend == 'compat':
        sys.path.insert(0, str(ROOT / 'Mortal/mortal'))
        from libriichi3p_compat import CompatMjaiEngine
        policies = [CompatMjaiEngine(policy, reference_path=ROOT / '.cache/libriichi3p',
                                    max_workers=16) for policy in policies]
    arena = native.arena.OneVsTwo(disable_progress_bar=True, log_dir=str(output / 'games'))
    started = time.monotonic()
    ranks = arena.py_vs_py(*policies, (17000, 2026091017), args.seed_count)
    elapsed = time.monotonic() - started
    files = sorted((output / 'games').glob('*.json.gz'))
    assert len(files) == sum(ranks) == 3 * args.seed_count
    nuki = decisions = 0
    for path in files:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            events = [json.loads(line) for line in stream]
        assert events[0]['type'] == 'start_game' and events[-1]['type'] == 'end_game'
        assert len(events[0]['names']) == 3
        for event in events:
            nuki += event['type'] == 'nukidora'
            meta = event.get('meta') or {}
            if 'mask_bits' in meta:
                assert len(meta['q_values']) == int(meta['mask_bits']).bit_count()
                decisions += 1
    assert decisions > 0 and nuki > 0, 'smoke must exercise decisions and North'
    result = dict(status='passed', backend=args.backend,
                  policy=str(args.checkpoint.resolve()) if args.checkpoint else 'synthetic-legal-not-a-trained-model',
                  quick_eval=args.quick_eval, games=len(files), ranks=list(ranks),
                  seconds=elapsed, games_per_second=len(files) / elapsed,
                  nukidora=nuki, decisions=decisions,
                  extension=str(native_sanma_smoke.EXTENSION),
                  extension_sha256=hashlib.sha256(native_sanma_smoke.EXTENSION.read_bytes()).hexdigest(),
                  rayon_num_threads=os.environ.get('RAYON_NUM_THREADS', 'default'))
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
