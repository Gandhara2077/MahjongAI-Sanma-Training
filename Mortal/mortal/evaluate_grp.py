from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from automation_pipeline import summarize_grp_validation
from config import config
from libriichi.consts import NUM_PLAYERS
from libriichi.dataset import Grp
from model import GRP


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare two GRP checkpoints on one fixed index.')
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10_000)
    parser.add_argument('--bootstrap-seed', type=int, required=True)
    args = parser.parse_args(argv)

    device = torch.device(config['grp']['control']['device'])
    index = torch.load(args.index, weights_only=True)
    validation_files = list(index.get('val_file_list', index.get('file_list', [])))
    if not validation_files:
        raise ValueError(f'GRP validation index is empty: {args.index}')

    models = {}
    checkpoints = {}
    for name, checkpoint_path in (
        ('incumbent', args.incumbent),
        ('candidate', args.candidate),
    ):
        state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model = GRP(**config['grp']['network']).to(device).eval()
        model.load_state_dict(state['model'])
        models[name] = model
        checkpoints[name] = {
            'path': str(checkpoint_path.resolve()),
            'step': int(state.get('steps', 0)),
            'sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }

    totals = {
        name: {'nll': 0.0, 'correct': 0, 'brier': 0.0, 'samples': 0}
        for name in models
    }
    deltas = {
        'nll_reduction': [],
        'exact_acc_gain': [],
        'brier_reduction': [],
    }
    games = 0
    with torch.inference_mode():
        for file_path in validation_files:
            for game in Grp.load_gz_log_files([file_path]):
                feature = game.take_feature()
                ranks = torch.tensor(
                    list(game.take_rank_by_player()),
                    dtype=torch.int64,
                    device=device,
                )
                sequences = [
                    torch.as_tensor(feature[:index + 1], dtype=torch.float64, device=device)
                    for index in range(len(feature))
                ]
                rank_targets = ranks.unsqueeze(0).expand(len(sequences), -1)
                one_hot_targets = F.one_hot(rank_targets, num_classes=NUM_PLAYERS).to(torch.float64)
                game_metrics = {}
                for name, model in models.items():
                    logits = model(sequences)
                    labels = model.get_label(rank_targets)
                    joint_prob = logits.softmax(-1)
                    marginal_prob = torch.stack([
                        torch.stack([
                            joint_prob[:, model.perms_t[player] == rank].sum(-1)
                            for rank in range(NUM_PLAYERS)
                        ], dim=1)
                        for player in range(NUM_PLAYERS)
                    ], dim=1)
                    nll = F.cross_entropy(logits, labels, reduction='none')
                    exact = (logits.argmax(-1) == labels).to(torch.float64)
                    brier = ((marginal_prob - one_hot_targets) ** 2).sum(dim=(1, 2))
                    game_metrics[name] = {
                        'nll': nll.mean().item(),
                        'exact_acc': exact.mean().item(),
                        'brier': brier.mean().item(),
                    }
                    total = totals[name]
                    total['nll'] += nll.sum().item()
                    total['correct'] += int(exact.sum().item())
                    total['brier'] += brier.sum().item()
                    total['samples'] += len(sequences)

                deltas['nll_reduction'].append(
                    game_metrics['incumbent']['nll'] - game_metrics['candidate']['nll']
                )
                deltas['exact_acc_gain'].append(
                    game_metrics['candidate']['exact_acc']
                    - game_metrics['incumbent']['exact_acc']
                )
                deltas['brier_reduction'].append(
                    game_metrics['incumbent']['brier'] - game_metrics['candidate']['brier']
                )
                games += 1

    summary = summarize_grp_validation(
        deltas,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    if summary['games'] != games:
        raise RuntimeError('GRP validation game count mismatch')
    report = {
        'index': str(args.index.resolve()),
        'validation_files': len(validation_files),
        'games': games,
        'checkpoints': checkpoints,
        'metrics': {
            name: {
                'samples': total['samples'],
                'nll': total['nll'] / total['samples'],
                'exact_acc': total['correct'] / total['samples'],
                'marginal_brier': total['brier'] / total['samples'],
            }
            for name, total in totals.items()
        },
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write('\n')
    temp_path.replace(args.output)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
