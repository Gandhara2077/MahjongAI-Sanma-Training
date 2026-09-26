from pathlib import Path
import importlib.util
import tempfile
import shutil
import subprocess
import sys
import copy
from unittest.mock import patch
import toml

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_online_phase1_overnight.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert hasattr(runner, 'describe_run'), 'runner requires non-mutating isolated preflight'
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
        base = Path(directory)
        # Run the real entrypoint in a disposable checkout so a regression
        # cannot append to the user's historical log while testing it.
        script = base / 'scripts' / 'run_online_phase1_overnight.py'
        script.parent.mkdir()
        shutil.copyfile(ROOT / 'scripts/run_online_phase1_overnight.py', script)
        failed = subprocess.run(
            [sys.executable, str(script), '--dry-run', '--config', str(base / 'missing.toml')],
            capture_output=True, text=True,
        )
        assert failed.returncode != 0 and 'FileNotFoundError' in failed.stderr
        assert not (base / 'online_phase1.log').exists(), 'failed preflight wrote a historical log'
        # A process already started must not escape cleanup if audit logging fails.
        class Child:
            stopped = False
            reaped = False
            pid = 123

            def wait(self, timeout):
                assert self.stopped
                self.reaped = True

        child = Child()
        original_open = Path.open

        def failing_audit(file, *args, **kwargs):
            if file.name == 'owned_processes.jsonl':
                raise OSError('audit write unavailable')
            return original_open(file, *args, **kwargs)

        with patch.object(runner, 'LOGDIR', base), patch.object(Path, 'open', failing_audit), \
                patch.object(runner.subprocess, 'Popen', return_value=child), \
                patch.object(runner, 'kill_tree', side_effect=lambda process: setattr(process, 'stopped', True)):
            try:
                runner.spawn('unused.py', 'child.log')
            except OSError:
                pass
            else:
                raise AssertionError('audit write failure was swallowed')
        assert child.stopped and child.reaped, 'audit failure orphaned a started child'
        cfg = toml.load(ROOT / 'Mortal/config/sanma-online-128x8.toml')
        cfg['control']['count_optimizer_updates'] = True
        for section, keys in runner.OUTPUT_FIELDS.items():
            for key in keys:
                if section == 'train_play.default':
                    target = cfg['train_play']['default']
                else:
                    target = cfg[section] if '.' not in section else cfg['online']['server']
                target[key] = str(base / (section + '_' + key))
        # Keep the check hermetic: the configured file index is a training
        # artifact that need not exist, so describe a tiny synthetic corpus
        # whose index matches its globs exactly.
        import torch
        data_dir = base / 'data' / '2026' / '01' / '01'
        data_dir.mkdir(parents=True)
        for name in ('a.mjson', 'b.mjson'):
            (data_dir / name).write_text('{}\n', encoding='utf-8')
        index_path = base / 'file_index.pth'
        torch.save({'file_list': sorted(str(file) for file in data_dir.glob('*.mjson'))}, index_path)
        cfg['dataset']['globs'] = [str(base / 'data' / '2026' / '**' / '*.mjson')]
        cfg['dataset']['file_index'] = str(index_path)
        description = runner.describe_run(cfg)
        assert description['dataset_index']['files'] == 2
        saved = {'config': copy.deepcopy(cfg), 'steps': 4000}
        fork = copy.deepcopy(cfg)
        fork['control']['state_file'] += '_fork'
        fork['online_training']['selfplay_engine'] = 'native'
        runner.validate_fork(saved, fork)
        fork['optim']['scheduler']['peak'] *= 2
        try:
            runner.validate_fork(saved, fork)
        except ValueError:
            pass
        else:
            raise AssertionError('fork silently changed learning rate')
        assert description['target_updates'] == 10000
        assert description['phase_start_step'] == 0
        resumed = copy.deepcopy(description)
        resumed['effective_config']['online']['drain_poll_seconds'] = 0.2
        resumed['effective_config']['online_training']['reuse_expert_iterator'] = True
        runner.validate_resume(description, resumed)
        for field in ('outputs', 'input_sha256', 'dataset_index', 'effective_config'):
            altered = copy.deepcopy(resumed)
            if field == 'effective_config':
                altered[field]['optim']['scheduler']['peak'] *= 2
            else:
                altered[field]['unexpected_change'] = True
            try:
                runner.validate_resume(description, altered)
            except ValueError:
                pass
            else:
                raise AssertionError(f'resume accepted changed {field}')
        collision = Path(cfg['control']['deployment_file'])
        collision.touch()
        try:
            runner.describe_run(cfg)
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing deployment must reject a fresh run')
    print('ONLINE_RUNNER_CHECK_OK')


if __name__ == '__main__':
    main()
