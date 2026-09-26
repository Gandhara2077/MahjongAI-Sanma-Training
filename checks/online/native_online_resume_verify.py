"""Verify saved isolated native training, Self sync, and trainee-only replay."""
import argparse
import gzip
import json
import re
from pathlib import Path

import toml
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'smoke'))
import native_sanma_smoke
from native_online_resume_smoke import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'passed' and result['production_unchanged']
    assert all(digest(Path(path)) == value for path, value in result['protected_sha256'].items())
    cfg = toml.load(output / 'config.toml')
    state = torch.load(output / 'state.pth', weights_only=True, map_location='cpu')
    assert state['steps'] == cfg['control']['max_steps']
    assert state['scheduler']['max_steps'] == cfg['optim']['scheduler']['max_steps']
    log = (output / 'runner_logs/client.log').read_text(encoding='utf-8', errors='replace')
    batches = [json.loads(line.split('opponent batch: ', 1)[1])
               for line in log.splitlines() if 'opponent batch: ' in line]
    verified = re.findall(r'self sync verified: (\S+) sha256=([0-9a-f]{64})', log)
    self_batches = [batch for batch in batches if batch['source'] == 'current']
    assert self_batches and verified
    assert all(batch['opponent_version'] == batch['trainee_version'] for batch in self_batches)
    if result['optimizer_updates'] > cfg['control']['submit_every']:
        assert len({value for _, value in verified}) >= 2, 'updated Self weights never observed'
    native_sanma_smoke.EXTENSION = output / 'runtime/libriichi.pyd'
    native = native_sanma_smoke.load_extension()
    assert digest(native_sanma_smoke.EXTENSION) == result['candidate_sha256']
    files = sorted(Path(cfg['online']['server']['drain_dir']).glob('*.json.gz'))
    assert files, 'no completed fresh replay logs'
    loader = native.dataset.GameplayLoader(version=4, oracle=False, player_names=['trainee'],
                                           encode_observations=True)
    for file in files:
        games = loader.load_gz_log_files([str(file)])[0]
        with gzip.open(file, 'rt', encoding='utf-8') as stream:
            names = json.loads(next(stream))['names']
        assert len(games) == 1 and games[0].take_player_id() == names.index('trainee')
        actions, masks = games[0].take_actions(), games[0].take_masks()
        assert actions and len(actions) == len(masks)
        assert all(mask[action] for action, mask in zip(actions, masks))
    verification = dict(status='passed', trainee_only_logs=len(files), self_batches=len(self_batches),
                        distinct_self_weights=len({value for _, value in verified}),
                        end_step=state['steps'], scheduler_max_steps=state['scheduler']['max_steps'])
    (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
