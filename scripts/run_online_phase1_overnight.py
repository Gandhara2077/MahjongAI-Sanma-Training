"""Online phase runner: train one phase, then staged promotion panel.

1. Runs the online closed loop (server + client + trainer) until the phase
   target (resuming from the on-disk state; the trainer exits by design
   after each test_play and is restarted here).
2. Staged 1v2 panel vs the 128x8 champion: 999 -> 1998 -> 2997 games,
   making a decision only after the fixed 2997-game budget.
3. Lightweight 999-game regression panel vs the original baseline.
4. Writes the result JSON. Never touches baselines/.

Fresh runs require --config and unused outputs. Use --dry-run first.
Choose unused --seed-base values before running the fixed-budget panels.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import glob
import os
import re
import subprocess
import socket
import sys
import time
import copy
import shutil
import toml
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'checks/stats'))
MORTAL = ROOT / "Mortal"
PYTHON = MORTAL.parent / ".venv-rocm" / "Scripts" / "python.exe"
CONFIG = MORTAL / "config" / "sanma-online-128x8.toml"
STATE = MORTAL / "training" / "online_128x8_state.pth"
DEPLOYMENT = MORTAL / "training" / "online_128x8_mortal3p.pth"
CHAMPION = ROOT / "baselines" / "sanma_champion_128x8_46500_20260831.pth"
BASELINE = ROOT / "baselines" / "sanma_baseline_mortal3p_original.pth"
LOG = ROOT / "online_phase1.log"
RESULT = ROOT / "online_phase1_result.json"
LOGDIR = ROOT / "online_phase1_logs"
RUNTIME = None

CHALLENGER_NAME = "sanma-online-128x8"

OUTPUT_FIELDS = {
    'control': ('state_file', 'deployment_file', 'best_state_file', 'best_deployment_file',
                'snapshot_dir', 'tensorboard_dir'),
    'test_play': ('log_dir',),
    'train_play.default': ('log_dir',),
    'online.server': ('buffer_dir', 'drain_dir'),
}


def describe_run(cfg, *, allow_existing=False):
    outputs = {}
    for section, fields in OUTPUT_FIELDS.items():
        values = cfg
        for part in section.split('.'):
            values = values[part]
        for field in fields:
            value = values.get(field)
            if not value:
                raise ValueError(f'isolated run requires {section}.{field}')
            output = (MORTAL / value).resolve()
            if not any(output.is_relative_to(allowed) and output != allowed
                       for allowed in ((MORTAL / 'training').resolve(), (ROOT / '.cache').resolve())):
                raise ValueError(f'output outside isolated training/cache area: {output}')
            if output.exists() and not allow_existing:
                raise FileExistsError(f'fresh run output already exists: {output}')
            outputs[f'{section}.{field}'] = str(output)
    paths = [Path(value) for value in outputs.values()]
    if len(set(paths)) != len(paths) or any(
        left != right and (left in right.parents or right in left.parents)
        for left in paths for right in paths
    ):
        raise ValueError('output fields must be distinct and non-overlapping')
    control = cfg['control']
    target = control['max_steps']
    start = cfg['online_training']['phase_start_step']
    if not control['online'] or start != 0 or target <= 0 or control['opt_step_every'] != 1:
        raise ValueError('fresh experiment requires online step zero and one optimizer update per step')
    if cfg['optim']['scheduler']['max_steps'] != target:
        raise ValueError('scheduler and update budget differ')
    inputs = [control['init_from'], cfg['grp']['state_file'], cfg['baseline']['test']['state_file']]
    inputs.extend(opponent['state_file'] for opponent in cfg['online_training']['opponents'])
    hashes = {}
    for value in sorted(set(inputs)):
        source = (MORTAL / value).resolve()
        with source.open('rb') as file:
            hashes[str(source)] = hashlib.file_digest(file, 'sha256').hexdigest()
    import torch
    index_path = (MORTAL / cfg['dataset']['file_index']).resolve()
    index = torch.load(index_path, weights_only=True, map_location='cpu')['file_list']
    indexed = {(MORTAL / file).resolve() for file in index}
    configured = {Path(file).resolve() for pattern in cfg['dataset']['globs']
                  for file in glob.glob(str(MORTAL / pattern), recursive=True)}
    if not indexed or len(indexed) != len(index) or indexed != configured:
        raise ValueError(f'file index differs from configured data: index={len(indexed)}, '
                         f'globs={len(configured)}, index_only={len(indexed - configured)}, '
                         f'globs_only={len(configured - indexed)}')
    with index_path.open('rb') as source:
        index_digest = hashlib.file_digest(source, 'sha256').hexdigest()
    successful_only = control.get('count_optimizer_updates', False)
    return {'target_training_steps': target, 'target_updates': target if successful_only else None,
            'step_accounting': 'successful_optimizer_updates' if successful_only else 'training_iterations',
            'phase_start_step': start,
            'dataset_index': {'path': str(index_path), 'sha256': index_digest,
                              'files': len(indexed), 'matches_globs': True},
            'input_sha256': hashes, 'outputs': outputs, 'effective_config': cfg}


def validate_resume(previous, current):
    """Allow throughput-only changes; never silently change the experiment."""
    for field in ('outputs', 'input_sha256', 'dataset_index', 'runtime_sha256'):
        if previous.get(field) != current.get(field):
            raise ValueError(f'resume changed {field}')
    configs = [copy.deepcopy(item['effective_config']) for item in (previous, current)]
    for cfg in configs:
        for section, key in (('online_training', 'reuse_expert_iterator'),
                             ('online', 'drain_poll_seconds'), ('compat', 'max_workers')):
            cfg.get(section, {}).pop(key, None)
        cfg['train_play']['default'].pop('games', None)
    if configs[0] != configs[1]:
        raise ValueError('resume may change throughput settings only')


def validate_fork(saved, cfg):
    configs = [copy.deepcopy(item) for item in (saved['config'], cfg)]
    for item in configs:
        for section, fields in OUTPUT_FIELDS.items():
            values = item
            for part in section.split('.'):
                values = values[part]
            for field in fields:
                values.pop(field, None)
        item['online']['remote'].pop('port', None)
        for key in ('selfplay_engine', 'verify_self_sync'):
            item['online_training'].pop(key, None)
    if configs[0] != configs[1]:
        raise ValueError('fork must preserve model, reward, data, opponents and optimization settings')
    if not saved['config']['control'].get('count_optimizer_updates') or not 0 < saved['steps'] < cfg['control']['max_steps']:
        raise ValueError('fork requires a successful-update checkpoint below target')


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def environment(cfg: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MORTAL_CFG"] = str(cfg)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn(script: str, log_name: str, cfg: Path = CONFIG) -> subprocess.Popen:
    if RUNTIME is not None:
        script = str(RUNTIME / Path(script).name)
    with open(LOGDIR / log_name, 'ab') as out:
        process = subprocess.Popen(
            [str(PYTHON), script], cwd=str(MORTAL), env=environment(cfg),
            stdout=out, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    try:
        with (LOGDIR / 'owned_processes.jsonl').open('a', encoding='utf-8') as record:
            record.write(json.dumps({'pid': process.pid, 'script': script, 'config': str(cfg),
                                     'started': datetime.now().isoformat()}) + '\n')
    except OSError:
        kill_tree(process)
        process.wait(timeout=15)
        raise
    return process


def kill_tree(process: subprocess.Popen) -> None:
    """Kill a spawned process AND its children.

    The venv python.exe is a launcher shim whose real interpreter is a child
    process; killing the shim alone would orphan the actual server/client.
    """
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
    )


def run_trainer(cfg: Path = CONFIG, timeout_seconds=None) -> None:
    processes = []
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    try:
        remote = toml.load(cfg)['online']['remote']
        address = (remote['host'], remote['port'])
        with socket.socket() as probe:
            if probe.connect_ex(address) == 0:
                raise RuntimeError(f'online port already occupied: {address}')
        server = spawn('mortal/server.py', 'server.log', cfg)
        processes.append(server)
        ready_deadline = min(deadline or float('inf'), time.monotonic() + 60)
        while True:
            if server.poll() is not None or time.monotonic() >= ready_deadline:
                raise RuntimeError('server failed to become ready')
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(address) == 0:
                    break
            time.sleep(0.2)
        client = spawn('mortal/client.py', 'client.log', cfg)
        processes.append(client)
        trainer = spawn("mortal/train.py", "trainer.log", cfg)
        processes.append(trainer)
        while trainer.poll() is None:
            if server.poll() is not None or client.poll() not in (None, 0):
                raise RuntimeError('online server/client failed; inspect component logs')
            if client.poll() == 0:
                raise RuntimeError('client game budget ended before the trainer target')
            if deadline and time.monotonic() >= deadline:
                raise TimeoutError('bounded online run exceeded its wall-clock limit')
            time.sleep(0.2)
        code = trainer.returncode
    finally:
        cleanup_errors = []
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    kill_tree(process)
                    process.wait(timeout=15)
                except Exception as error:
                    cleanup_errors.append(f'pid {process.pid}: {error}')
        if cleanup_errors:
            raise RuntimeError('owned process cleanup incomplete: ' + '; '.join(cleanup_errors))
    if code != 0:
        raise RuntimeError(f"trainer exited with {code}")


def config_relative(state: Path) -> str:
    """Path as written into the panel TOML (relative to Mortal/)."""
    return Path(os.path.relpath(state, MORTAL)).as_posix()


def run_panel(
    label: str,
    challenger_state: Path,
    opponent_state: Path,
    opponent_name: str,
    games_per_iter: int,
    iters: int,
    seed_key: int,
) -> Path:
    log_dir = MORTAL / "training" / f"sanma_1v2_online_p1_{label}"
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"panel log dir not empty: {log_dir}")
    cfg_text = f"""
[compat]
libriichi3p = '../.cache/libriichi3p/libriichi3p-3.12-x86_64-pc-windows-msvc.pyd'
max_workers = 16

[1v2]
seed_key = {seed_key}
games_per_iter = {games_per_iter}
iters = {iters}
log_dir = 'training/sanma_1v2_online_p1_{label}'
pts = [90, 0, -90]

[1v2.challenger]
device = 'cuda:0'
name = '{CHALLENGER_NAME}'
state_file = '{config_relative(challenger_state)}'
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v2.champion]
device = 'cuda:0'
name = '{opponent_name}'
state_file = '{config_relative(opponent_state)}'
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true
"""
    cfg = MORTAL / "config" / f"sanma-eval-online-p1-{label}.toml"
    cfg.write_text(cfg_text, encoding="utf-8")
    log(f"START panel {label}: {games_per_iter * iters} games")
    result = subprocess.run(
        [
            str(PYTHON), "mortal/one_vs_two.py",
            "--max-workers", "16",
            "--games-per-iter", str(games_per_iter),
            "--iters", str(iters),
            "--log-dir", str(log_dir),
        ],
        cwd=str(MORTAL),
        env=environment(cfg),
        capture_output=True,
        text=True,
    )
    (LOGDIR / f"panel_{label}.log").write_text(result.stdout[-20000:], encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"panel {label} failed: {result.stdout[-2000:]}")
    log(f"END panel {label}")
    return log_dir


def paired_stats(panel_dir: Path, challenger_name: str = CHALLENGER_NAME) -> dict:
    from online_panel_stats import panel_stats
    label = Path(panel_dir).name.removeprefix('sanma_1v2_online_p1_')
    return panel_stats(panel_dir, challenger_name, expected_games=999,
                       native_log=LOGDIR / f'panel_{label}.log')


def write_result(payload: dict) -> None:
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global LOG, RESULT, LOGDIR, STATE, DEPLOYMENT, CHAMPION, BASELINE, RUNTIME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume saved training only; completed/partial panels are never overwritten')
    parser.add_argument('--resume-from', type=Path, help='fork an unchanged experiment into fresh outputs')
    parser.add_argument('--runtime-dir', type=Path, help='isolated training runtime; final panels keep the installed evaluator')
    parser.add_argument('--target', type=int, default=None,
                        help='phase target step (trainer resumes from the on-disk state)')
    parser.add_argument('--prefix', default='online_p1',
                        help='panel label prefix; must differ between phases')
    parser.add_argument('--seed-base', type=int, default=20260922,
                        help='first panel seed key; must differ between phases')
    args = parser.parse_args()
    if args.resume and args.resume_from:
        raise ValueError('resume and resume-from are mutually exclusive')
    configured_path = args.config.resolve()
    phase_cfg = toml.load(configured_path)
    if args.target is not None and args.target != phase_cfg['control']['max_steps']:
        raise ValueError('target must match the explicit config; do not rewrite scheduler silently')
    args.target = phase_cfg['control']['max_steps']
    description = describe_run(phase_cfg, allow_existing=args.resume)
    if args.runtime_dir:
        RUNTIME = args.runtime_dir.resolve()
        runtime_files = list(RUNTIME.glob('*.py')) + [RUNTIME / 'libriichi.pyd']
        for name in ('server.py', 'client.py', 'train.py', 'player.py'):
            if not (RUNTIME / name).is_file():
                raise ValueError(f'missing runtime script: {name}')
        description['runtime_sha256'] = {}
        for file in runtime_files:
            with file.open('rb') as source:
                description['runtime_sha256'][str(file)] = hashlib.file_digest(source, 'sha256').hexdigest()
    if phase_cfg['online_training'].get('selfplay_engine') == 'native' and RUNTIME is None:
        raise ValueError('native self-play requires an explicit isolated runtime')
    if args.resume_from:
        import torch
        source_state = args.resume_from.resolve()
        with source_state.open('rb') as source:
            description['fork_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        saved = torch.load(source_state, weights_only=True, map_location='cpu')
        validate_fork(saved, phase_cfg)
        description.update(fork_from=str(source_state), resume_step=int(saved['steps']))
        del saved
    STATE = Path(description['outputs']['control.state_file'])
    DEPLOYMENT = Path(description['outputs']['control.deployment_file'])
    CHAMPION = (MORTAL / phase_cfg['control']['init_from']).resolve()
    BASELINE = (MORTAL / phase_cfg['baseline']['test']['state_file']).resolve()
    RESULT = STATE.parent / 'result.json'
    LOG = STATE.parent / 'runner.log'
    LOGDIR = STATE.parent / 'runner_logs'
    if args.resume:
        previous = json.loads((LOGDIR / 'preflight.json').read_text(encoding='utf-8'))
        validate_resume(previous, description)
        import torch
        saved = torch.load(STATE, weights_only=True, map_location='cpu')
        if not saved['config']['control'].get('count_optimizer_updates') or not 0 < saved['steps'] <= args.target:
            raise ValueError('resume requires a saved successful-update checkpoint within the budget')
        description['resume_step'] = saved['steps']
        del saved
    for output in (RESULT, LOG, LOGDIR):
        if output.exists() and (not args.resume or output == RESULT):
            raise FileExistsError(f'runner output already exists: {output}')
    if args.prefix == 'online_p1':
        args.prefix = configured_path.stem
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix):
        raise ValueError('panel prefix must be a simple label')
    panel_labels = [f'{args.prefix}_champ_s{stage}_999' for stage in (1, 2, 3)]
    panel_labels.append(f'{args.prefix}_baseline_999')
    panel_outputs = []
    for label in panel_labels:
        panel_outputs.extend((MORTAL / 'training' / f'sanma_1v2_online_p1_{label}',
                              MORTAL / 'config' / f'sanma-eval-online-p1-{label}.toml',
                              LOGDIR / f'panel_{label}.log'))
    for output in panel_outputs:
        if output.exists():
            raise FileExistsError(f'panel output already exists: {output}')
    requested_keys = set(range(args.seed_base, args.seed_base + 4))
    for existing in (MORTAL / 'config').glob('*.toml'):
        used = toml.load(existing).get('1v2', {}).get('seed_key')
        if used in requested_keys:
            raise ValueError(f'evaluation seed key already used by {existing}: {used}')
    description.update(seed_keys=sorted(requested_keys), result=str(RESULT), log=str(LOG),
                       log_dir=str(LOGDIR), panel_outputs=[str(item) for item in panel_outputs])
    if args.resume and (previous['seed_keys'] != description['seed_keys'] or previous['panel_outputs'] != description['panel_outputs']):
        raise ValueError('resume must preserve the original evaluation seed and output plan')
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return
    if description['step_accounting'] != 'successful_optimizer_updates':
        raise ValueError('fresh experiment requires count_optimizer_updates=true; legacy counters are not update budgets')
    LOGDIR.mkdir(parents=True, exist_ok=args.resume)
    manifest_name = 'resume-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json' if args.resume else 'preflight.json'
    with (LOGDIR / manifest_name).open('x', encoding='utf-8') as manifest:
        json.dump(description, manifest, indent=2)
    if args.resume_from:
        with source_state.open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != description['fork_sha256']:
                raise RuntimeError('fork source changed after preflight')
        with source_state.open('rb') as source, STATE.open('xb') as target:
            shutil.copyfileobj(source, target)
    log(f"online pipeline started (target {args.target})")
    log("policy: staged panel 999/1998/2997 vs champion, 999 vs baseline")

    if not CHAMPION.exists() or not BASELINE.exists():
        raise SystemExit("baseline models missing")
    import torch

    start_steps = 0
    if STATE.exists():
        start_steps = int(torch.load(STATE, map_location="cpu", weights_only=False)["steps"])
        log(f"online state at step {start_steps}")
    if start_steps < args.target:
        # Derive a phase config whose max_steps matches this phase's target
        # (the trainer resumes from the on-disk state and runs to max_steps).
        phase_cfg_path = configured_path
        # In online mode the trainer exits by design after each test_play
        # (upstream quirk: the process hangs otherwise), so restart it until
        # the phase target is reached.
        t0 = time.time()
        restarts = 0
        while True:
            run_trainer(phase_cfg_path)
            state = torch.load(STATE, map_location="cpu", weights_only=False)
            start_steps = int(state["steps"])
            log(f"trainer segment finished at {start_steps} steps "
                f"({(time.time() - t0) / 60:.1f} min total)")
            if start_steps >= args.target:
                break
            restarts += 1
            if restarts > 10:
                raise RuntimeError("too many trainer restarts without reaching target")
    else:
        log("training already at target; skipping to panels")
    final_steps = start_steps
    log(f"final steps: {final_steps}")
    if not DEPLOYMENT.exists():
        raise RuntimeError("deployment export missing after training")

    result: dict = {
        "started": LOG.read_text(encoding="utf-8").splitlines()[0],
        "final_steps": final_steps,
        "champion": str(CHAMPION),
    }

    # Staged panel vs the 128x8 champion. Seeds per stage use distinct keys
    # so each stage is an independent block; diffs are pooled across stages.
    stage_specs = [
        (f"{args.prefix}_champ_s1_999", 333, 3, args.seed_base),
        (f"{args.prefix}_champ_s2_999", 333, 3, args.seed_base + 1),
        (f"{args.prefix}_champ_s3_999", 333, 3, args.seed_base + 2),
    ]
    verdict = None
    for label, gpi, iters, key in stage_specs:
        panel_dir = run_panel(label, DEPLOYMENT, CHAMPION, "sanma-128x8-champion-46k", gpi, iters, key)
        stats = paired_stats(panel_dir)
        result.setdefault("vs_champion_stages", []).append({"stage": label, **stats})
        write_result(result)
        log(f"stage {label}: " + str({key: value for key, value in stats.items() if key != 'seed_records'}))
        # significance requires pooling all stages so far
        pooled = paired_stats_pooled(result["vs_champion_stages"])
        result["vs_champion_pooled"] = pooled
        write_result(result)
        log('pooled: ' + str({key: value for key, value in pooled.items() if key != 'seed_records'}))
        verdict = pooled

    # Regression guard vs the original baseline (1 stage only).
    panel_dir = run_panel(f"{args.prefix}_baseline_999", DEPLOYMENT, BASELINE, "mortal-baseline", 333, 3, args.seed_base + 3)
    result["vs_baseline"] = paired_stats(panel_dir, challenger_name=CHALLENGER_NAME)
    write_result(result)
    log('vs baseline: ' + str({key: value for key, value in result['vs_baseline'].items() if key != 'seed_records'}))

    stronger = verdict is not None and verdict["significant"]
    baseline_ok = result["vs_baseline"]["mean_rank"] <= 1.81
    result["promotion"] = {
        "significantly_better_than_champion": bool(stronger),
        "baseline_regression_guard_ok": bool(baseline_ok),
        "baseline_guard_type": "policy_threshold_not_noninferiority",
        "baseline_policy_threshold": 1.81,
        "promote": bool(stronger and baseline_ok),
        "note": "final promotion requires morning human review",
    }
    write_result(result)
    log(f"promotion: {result['promotion']}")
    log("online pipeline completed")


def paired_stats_pooled(stages: list[dict]) -> dict:
    from online_panel_stats import pooled_stats
    return pooled_stats(stages)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"pipeline failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
