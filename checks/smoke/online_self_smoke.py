from pathlib import Path
import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
import time
import toml

ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / 'Mortal'


def verify_saved_smoke(output, elapsed):
    import torch
    config_path = output / 'smoke.toml'
    cfg = toml.load(config_path)
    client_log = (output / 'client.log').read_text(encoding='utf-8', errors='replace')
    batches = [json.loads(line.split('opponent batch: ', 1)[1])
               for line in client_log.splitlines() if 'opponent batch: ' in line]
    verified = re.findall(r'self sync verified: (\S+) sha256=([0-9a-f]{64})', client_log)
    assert len({digest for _, digest in verified}) >= 2, 'two distinct actual engine weights not observed'
    generated = sum(batch['games'] for batch in batches)
    assert 0 < generated <= 60 and elapsed <= 600
    self_batches = [batch for batch in batches if batch['source'] == 'current']
    assert self_batches, 'no current-self game batch observed'
    assert all(batch['opponent_version'] == batch['trainee_version'] for batch in self_batches)
    for name in {batch['name'] for batch in batches if batch['source'] == 'checkpoint'}:
        assert len({batch['opponent_version'] for batch in batches if batch['name'] == name}) == 1
    state = torch.load(cfg['control']['state_file'], weights_only=True, map_location='cpu')
    assert state['steps'] == 20
    assert state['config']['control']['online']
    assert state['config']['online_training']['phase_start_step'] == 0
    os.environ['MORTAL_CFG'] = str(config_path)
    sys.path.insert(0, str(MORTAL / 'mortal'))
    from libriichi.dataset import GameplayLoader
    from model import Brain, DQN
    restored_brain = Brain(version=4, **cfg['resnet'])
    restored_dqn = DQN(version=4)
    restored_brain.load_state_dict(state['mortal'])
    restored_dqn.load_state_dict(state['current_dqn'])
    assert state['optimizer']['state'], 'optimizer state not saved'
    optimizer_steps = {int(entry['step']) for entry in state['optimizer']['state'].values()}
    assert len(optimizer_steps) == 1 and 10 < max(optimizer_steps) <= 20
    if cfg['control'].get('count_optimizer_updates', False):
        assert optimizer_steps == {20}
    trainer_log = (output / 'trainer.log').read_text(encoding='utf-8', errors='replace')
    assert 'loaded:' in trainer_log and 'total steps: 10' in trainer_log
    files = sorted(Path(cfg['online']['server']['drain_dir']).glob('*.json.gz'))
    assert files, 'last completed self-play logs missing'
    loader = GameplayLoader(version=4, oracle=False, player_names=['trainee'], encode_observations=False)
    samples = loader.load_gz_log_files([str(file) for file in files])
    assert len(samples) == len(files)
    for file, games in zip(files, samples):
        with gzip.open(file, 'rt', encoding='utf-8') as source:
            names = json.loads(next(source))['names']
        assert len(games) == 1 and games[0].take_player_id() == names.index('trainee')
    return dict(status='passed', elapsed_seconds=elapsed, training_steps=state['steps'],
                  optimizer_updates=max(optimizer_steps),
                  amp_skipped_steps=(trainer_log.count('AMP skipped optimizer update;')
                                     if cfg['control'].get('count_optimizer_updates', False)
                                     else state['steps'] - max(optimizer_steps)),
                  generated_games=generated, verified_versions=verified, batches=batches,
                  trainee_only_logs=len(files), checkpoint_reload='Brain/DQN strict load passed',
                  full_optimizer_resume='real trainer resumed 10 to 20; original optimizer state retained')


def main():
    parser = argparse.ArgumentParser(description='Bounded real GPU online/self synchronization smoke')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true', help='verify existing artifacts without playing/training')
    parser.add_argument('--reuse-expert-iterator', action='store_true', help='exercise the optimized expert stream and 0.2s drain polling')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.cache') or output == ROOT / '.cache':
        raise ValueError('smoke output must be a fresh child of project .cache')
    if args.verify_only:
        elapsed = (output / 'result.json').stat().st_mtime - (output / 'preflight.json').stat().st_mtime
        verification = verify_saved_smoke(output, elapsed)
        verification['timing_evidence'] = 'upper bound from preflight/result artifact timestamps'
        with (output / 'verification.json').open('x', encoding='utf-8') as destination:
            json.dump(verification, destination, indent=2)
        print('ONLINE_SELF_SMOKE_VERIFIED ' + json.dumps({key: value for key, value in verification.items()
                                                       if key not in ('batches', 'verified_versions')}))
        return
    if output.exists():
        raise FileExistsError(output)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm GPU unavailable; CPU is not an accepted substitute')
    cfg = toml.load(MORTAL / 'config/sanma-online-128x8-dynamic-self.toml')
    spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_online_phase1_overnight.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for section, fields in runner.OUTPUT_FIELDS.items():
        values = cfg
        for part in section.split('.'):
            values = values[part]
        for field in fields:
            suffix = '.pth' if field.endswith('_file') else ''
            values[field] = str(output / (section + '_' + field + suffix))
    cfg['control'].update(max_steps=20, batch_size=128, save_every=5, submit_every=2, snapshot_every=0,
                          test_every=4000, val_loss_every=0)
    cfg['optim']['scheduler'].update(max_steps=20, warm_up_steps=10)
    cfg['dataset']['num_workers'] = 0
    cfg['compat']['max_workers'] = 4
    cfg['train_play']['default']['games'] = 3
    cfg['online_training'].update(max_generated_games=60, verify_self_sync=True)
    if args.reuse_expert_iterator:
        cfg['online_training']['reuse_expert_iterator'] = True
        cfg['online']['drain_poll_seconds'] = 0.2
        cfg['dataset']['num_workers'] = 2
    cfg['online']['remote']['port'] = 5022
    preflight = runner.describe_run(cfg)
    output.mkdir()
    config_path = output / 'smoke.toml'
    config_path.write_text(toml.dumps(cfg), encoding='utf-8')
    (output / 'preflight.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
    runner.LOGDIR = output
    started = time.monotonic()
    result = {'status': 'running', 'device': torch.cuda.get_device_name(0),
              'limits': {'updates': 20, 'games': 60, 'seconds': 600},
              'smoke_only_changes': 'batch=128, workers=0, file batches unchanged, submit=2, warmup=10, games/batch=3, compat workers=4',
              'configuration': str(config_path)}
    result['reuse_expert_iterator'] = args.reuse_expert_iterator
    if args.reuse_expert_iterator:
        result['smoke_only_changes'] += '; optimized variant: workers=2, expert stream reused, drain poll=0.2s'
    try:
        for target in (10, 20):
            cfg['control']['max_steps'] = target
            previous_log = (output / 'client.log')
            previous_batches = [json.loads(line.split('opponent batch: ', 1)[1])
                                for line in previous_log.read_text(encoding='utf-8', errors='replace').splitlines()
                                if 'opponent batch: ' in line] if previous_log.exists() else []
            remaining_games = 60 - sum(batch['games'] for batch in previous_batches)
            if remaining_games < cfg['train_play']['default']['games']:
                raise RuntimeError('smoke game budget exhausted before resume')
            cfg['online_training']['max_generated_games'] = remaining_games
            config_path.write_text(toml.dumps(cfg), encoding='utf-8')
            remaining_seconds = 600 - (time.monotonic() - started)
            if remaining_seconds <= 0:
                raise TimeoutError('smoke wall-clock budget exhausted')
            runner.run_trainer(config_path, timeout_seconds=remaining_seconds)
            saved = torch.load(cfg['control']['state_file'], weights_only=True, map_location='cpu')
            assert saved['steps'] == target
        elapsed = time.monotonic() - started
        result.update(verify_saved_smoke(output, elapsed))
    except BaseException as error:
        result.update(status='failed', elapsed_seconds=time.monotonic() - started,
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('ONLINE_SELF_SMOKE_OK ' + json.dumps(result))


if __name__ == '__main__':
    main()
