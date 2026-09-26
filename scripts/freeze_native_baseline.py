'''Freeze native rollout inputs without overwriting an existing archive.'''

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT).decode('utf-8')


def verify(directory):
    records = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for record in records['files']:
        archived = directory / record['path']
        if archived.stat().st_size != record['bytes'] or digest(archived) != record['sha256']:
            raise ValueError('archive mismatch: ' + record['path'])
    print('BASELINE_VERIFIED files=' + str(len(records['files'])) + ' directory=' + str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.verify:
        verify(directory)
        return
    directory.relative_to((ROOT / '.cache').resolve())
    if directory.exists():
        raise FileExistsError(f'refusing to overwrite {directory}')
    paths = set(git('ls-files', 'Mortal/libriichi', 'Mortal/Cargo.toml',
                    'Mortal/Cargo.lock', 'Mortal/mortal', 'checks', 'scripts').splitlines())
    for pattern in ('Mortal/config/*.toml', 'baselines/*.pth'):
        paths.update(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
    paths.update(['Mortal/mortal/libriichi.pyd',
                  '.cache/libriichi3p/libriichi3p-3.12-x86_64-pc-windows-msvc.pyd'])
    sources = [(relative, ROOT / relative) for relative in sorted(paths)]
    records = [{'path': relative, 'bytes': source.stat().st_size,
                'sha256': digest(source)} for relative, source in sources]
    head = git('rev-parse', 'HEAD').strip()
    status = git('status', '--short')
    patch = git('diff', '--binary')
    directory.mkdir(parents=True)
    for relative, source in sources:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (directory / 'source.patch').write_text(patch, encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps(
        {'head': head, 'status': status, 'files': records}, indent=2), encoding='utf-8')
    verify(directory)


if __name__ == '__main__':
    main()
