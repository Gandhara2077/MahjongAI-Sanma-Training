"""Prepare the isolated 2026-09-16 continuation; never start or overwrite a run."""
import sys
from pathlib import Path
import toml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'checks/online'))
from native_online_resume_smoke import prepare_runtime, digest
from run_online_phase1_overnight import OUTPUT_FIELDS, validate_fork, describe_run


def main():
    import torch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm GPU required')
    mortal = ROOT / 'Mortal'
    output = mortal / 'training/online_native_self_20260916'
    if output.exists():
        raise FileExistsError(output)
    cfg = toml.load(mortal / 'config/sanma-online-128x8-dynamic-self-perf.toml')
    source = mortal / cfg['control']['state_file']
    if digest(source) != '94c8a29207b3d58c7d13e07d329c33f8216d786c724ec190faa86ab779779619':
        raise RuntimeError('formal 4000-step source changed')
    saved = torch.load(source, weights_only=True, map_location='cpu')
    assert saved['config'] == cfg and saved['steps'] == 4000
    for section, fields in OUTPUT_FIELDS.items():
        values = cfg
        for part in section.split('.'):
            values = values[part]
        for field in fields:
            values[field] = str(output / Path(values[field]).name)
    cfg['online']['remote']['port'] = 5025
    cfg['online_training'].update(selfplay_engine='native', verify_self_sync=True)
    validate_fork(saved, cfg)
    describe_run(cfg)
    extension = ROOT / '.cache/native-online-20260914/libriichi-py312.pyd'
    if digest(extension) != '16f13ca2ede139b2e1ac5592cba7eaa88b23adf5c2b8b9bb5e3c86518c514c5b':
        raise RuntimeError('tested native extension changed')
    output.mkdir()
    prepare_runtime(output, extension)
    (output / 'config.toml').write_text(toml.dumps(cfg), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
