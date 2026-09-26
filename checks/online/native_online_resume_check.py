"""CPU regression: resume smoke must not overwrite production artifacts."""
import tempfile
from pathlib import Path

import toml
import native_online_resume_smoke as smoke


def main():
    cfg = toml.load(smoke.MORTAL / 'config/sanma-online-128x8-dynamic-self-perf.toml')
    scheduler = dict(cfg['optim']['scheduler'])
    with tempfile.TemporaryDirectory(dir=smoke.ROOT / '.cache') as temporary:
        output = Path(temporary)
        smoke.configure_resume(cfg, output, 4321, 20)
        assert cfg['control']['max_steps'] == 4341
        assert cfg['optim']['scheduler'] == scheduler
        assert cfg['grp']['state_file'] == '../baselines/sanma_grp_10k_last_frozen.pth'
        assert cfg['dataset']['file_index'] == 'training/sanma_192x12_file_index.pth'
        assert Path(cfg['online']['server']['drain_dir']).is_relative_to(output)
        runtime = smoke.prepare_runtime(output, smoke.MORTAL / 'native/sanma/libriichi-py312.pyd')
        assert runtime.is_relative_to(output)
        assert (runtime / 'libriichi.pyd').read_bytes() == (smoke.MORTAL / 'native/sanma/libriichi-py312.pyd').read_bytes()
        assert (runtime / 'train.py').is_file()
        assert not (output / 'drain').exists(), 'must not reuse old self-play'
    print('NATIVE_ONLINE_RESUME_CHECK_OK')


if __name__ == '__main__':
    main()
