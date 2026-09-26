"""Run a bounded native-engine online resume from a copied checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import hashlib
import importlib.util
from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
PYTHON = ROOT / ".venv-rocm" / "Scripts" / "python.exe"


def configure_resume(cfg, output, start, updates):
    cfg["control"].update(
        state_file=str(output / "state.pth"),
        deployment_file=str(output / "deployment.pth"),
        best_state_file=str(output / "best_state.pth"),
        best_deployment_file=str(output / "best_deployment.pth"),
        snapshot_dir=str(output / "snapshots"),
        tensorboard_dir=str(output / "tb"),
        max_steps=start + updates,
        test_every=1000000,
        snapshot_every=0,
    )
    cfg["online"]["remote"]["port"] = 5024
    cfg["online"]["server"].update(buffer_dir=str(output / "buffer"), drain_dir=str(output / "drain"))
    cfg["test_play"]["log_dir"] = str(output / "test_play")
    cfg["train_play"]["default"]["games"] = 96
    cfg["train_play"]["default"]["log_dir"] = str(output / "train_play")
    cfg["online_training"]["max_generated_games"] = 96 * updates
    cfg["online_training"]["verify_self_sync"] = True
    cfg["dataset"]["num_workers"] = 2
    cfg["compat"] = {}


def prepare_runtime(output, extension):
    runtime = output / 'runtime'
    runtime.mkdir()
    # Separate script directory also propagates to Windows spawn workers.
    for source in (MORTAL / 'mortal').glob('*.py'):
        shutil.copyfile(source, runtime / source.name)
    shutil.copyfile(extension, runtime / 'libriichi.pyd')
    return runtime


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--backend', choices=('native', 'compat'), default='native')
    args = parser.parse_args()
    output = args.output.resolve()
    cache = (ROOT / '.cache').resolve()
    if output.exists() or output == cache or not output.is_relative_to(cache):
        raise ValueError('output must be a fresh child of .cache')
    if not 1 <= args.updates <= 500:
        raise ValueError('updates must be 1..500')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm required; no CPU fallback')
    source = MORTAL / 'training/online_dynamic_self_20260908'
    protected = [source / 'state.pth', source / 'deployment.pth', MORTAL / 'mortal/libriichi.pyd']
    hashes = {str(path): digest(path) for path in protected}
    state = torch.load(source / 'state.pth', weights_only=True, map_location='cpu')
    start = int(state['steps'])
    before = {key: int(value['step']) for key, value in state['optimizer']['state'].items()}
    cfg = toml.load(MORTAL / 'config/sanma-online-128x8-dynamic-self-perf.toml')
    assert state['config']['control']['count_optimizer_updates']
    assert cfg['optim']['scheduler'] == state['config']['optim']['scheduler']
    assert start + args.updates <= cfg['optim']['scheduler']['max_steps']
    compat = cfg['compat'].copy()
    configure_resume(cfg, output, start, args.updates)
    if args.backend == 'compat':
        cfg['compat'] = compat
    output.mkdir(parents=True)
    for name in ('state.pth', 'deployment.pth'):
        shutil.copyfile(source / name, output / name)
    runtime = prepare_runtime(output, ROOT / '.cache/native-online-20260914/libriichi-py312.pyd')
    config_path = output / "config.toml"
    config_path.write_text(toml.dumps(cfg), encoding="utf-8")

    runner_spec = importlib.util.spec_from_file_location(
        "runner", ROOT / "scripts/run_online_phase1_overnight.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    runner.LOGDIR = output / "runner_logs"
    runner.LOGDIR.mkdir()
    runner.LOG = output / "runner.log"
    spawn = runner.spawn
    runner.spawn = lambda script, log_name, cfg: spawn(str(runtime / Path(script).name), log_name, cfg)
    started = time.monotonic()
    result = dict(status='running', engine=args.backend, start_step=start,
                  candidate_sha256=digest(runtime / 'libriichi.pyd'), protected_sha256=hashes,
                  config=str(config_path), runner_logs=str(runner.LOGDIR))
    try:
        runner.run_trainer(config_path, timeout_seconds=1800)
        state = torch.load(output / 'state.pth', weights_only=True, map_location='cpu')
        assert state['steps'] == start + args.updates
        after = {key: int(value['step']) for key, value in state['optimizer']['state'].items()}
        assert after.keys() == before.keys()
        assert all(after[key] - before[key] == args.updates for key in before)
        assert state['config']['optim']['scheduler'] == cfg['optim']['scheduler']
        result.update(status='passed', end_step=int(state['steps']), optimizer_updates=args.updates)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['production_unchanged'] = all(digest(Path(path)) == value for path, value in hashes.items())
        if not result['production_unchanged']:
            result['status'] = 'failed'
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert result['production_unchanged'], 'production artifacts changed during smoke'
    print(json.dumps(result))


if __name__ == "__main__":
    main()
