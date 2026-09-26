"""Bit-exact gate for native sanma v4 versus the pinned reference encoder.

The frozen MahjongCopilot/libriichi3p deployment ABI is `(775, 34)` with a
44-entry action mask. This check compares native v4 and the pinned historical
binary on the same native decision metadata, for both raw and augmented logs.
It intentionally fails on reference skips, zero compared samples, cardinality
mismatches, illegal labels, shape errors, or any value difference.
"""

import argparse
import gzip
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_DIR = REPO_ROOT / 'Mortal'
MORTAL_PKG_DIR = MORTAL_DIR / 'mortal'
sys.path.insert(0, str(MORTAL_PKG_DIR))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-extension', type=Path)
parser.add_argument('--files-per-day', type=int, default=8)
parser.add_argument('--augmented-files', type=int, default=10)
parser.add_argument('--output', type=Path)
options = parser.parse_args() if __name__ == '__main__' else parser.parse_args([])
if options.files_per_day < 1 or options.augmented_files < 1:
    parser.error('sample counts must be positive')
if options.output is not None and options.output.exists():
    parser.error('output already exists; refusing to overwrite evidence')
if options.native_extension is not None:
    extension_path = options.native_extension.resolve(strict=True)
    extension_loader = importlib.machinery.ExtensionFileLoader('libriichi', str(extension_path))
    extension_spec = importlib.util.spec_from_file_location(
        'libriichi', extension_path, loader=extension_loader,
    )
    extension_module = importlib.util.module_from_spec(extension_spec)
    extension_loader.exec_module(extension_module)
    sys.modules['libriichi'] = extension_module

import libriichi  # noqa: E402  (mortal/libriichi.pyd, active sanma variant)
from libriichi.dataset import GameplayLoader  # noqa: E402
from libriichi3p_compat import (  # noqa: E402
    CompatibilityError,
    augment_log,
    capture_replay,
    resolve_reference_path,
)

MJJSON_ROOT = REPO_ROOT / 'koromo' / 'mjson' / '2026'
EXPECTED_OBS_SHAPE = (775, 34)
EXPECTED_MASK_SHAPE = (44,)

SAMPLE_MONTH_DAYS = [
    ('01', '04'),
    ('03', '15'),
    ('05', '20'),
    ('07', '11'),
    ('08', '09'),
]
FILES_PER_MONTH = options.files_per_day
AUGMENTED_SUBSET = options.augmented_files


def read_log(path):
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        return file.read()


def sample_files():
    files = []
    for month, day in SAMPLE_MONTH_DAYS:
        day_dir = MJJSON_ROOT / month / day
        day_files = sorted(day_dir.glob('*.mjson'))
        if not day_files:
            day_files = sorted(day_dir.glob('**/*.mjson'))
        files.extend(day_files[:FILES_PER_MONTH])
    return files


def load_native(paths, version, augmented, *, encode_observations):
    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=None,
        excludes=None,
        trust_seed=False,
        always_include_kan_select=True,
        augmented=augmented,
        encode_observations=encode_observations,
    )
    started = time.perf_counter()
    data = loader.load_gz_log_files([str(path) for path in paths])
    return data, time.perf_counter() - started


def extract_native(data, *, include_tensors):
    extracted = []
    for games in data:
        file_games = []
        for game in games:
            entry = {
                'player_id': game.take_player_id(),
                'actions': game.take_actions(),
                'event_indices': game.take_event_indices(),
                'at_kan_select': game.take_at_kan_select(),
            }
            if include_tensors:
                entry['obs'] = game.take_obs()
                entry['masks'] = game.take_masks()
            file_games.append(entry)
        extracted.append(file_games)
    return extracted


def metadata_of(game):
    return (
        game['player_id'],
        game['actions'],
        game['event_indices'],
        game['at_kan_select'],
    )


def capture_reference(
    raw_log,
    player_id,
    actions,
    event_indices,
    at_kan_select,
    reference_path,
):
    return capture_replay(
        raw_log,
        player_id=player_id,
        event_indices=event_indices,
        actions=actions,
        at_kan_select=at_kan_select,
        reference_path=str(reference_path),
    )


def main():
    reference_path = resolve_reference_path(REPO_ROOT / '.cache' / 'libriichi3p')
    expected_native = (
        libriichi.consts.NUM_PLAYERS,
        libriichi.consts.MAX_VERSION,
        tuple(libriichi.consts.obs_shape(4)),
        tuple(libriichi.consts.obs_shape(5)),
        libriichi.consts.ACTION_SPACE,
    )
    if expected_native != (3, 5, EXPECTED_OBS_SHAPE, (780, 34), 44):
        raise SystemExit(
            'active native extension is not the expected sanma build: '
            f'{expected_native} from {libriichi.__file__}'
        )

    files = sample_files()
    if not files:
        raise SystemExit('no corpus files sampled; check koromo/mjson layout')
    print(f'native extension: {libriichi.__file__}')
    print(f'reference extension: {reference_path}')
    print(f'sampled {len(files)} files from {len(SAMPLE_MONTH_DAYS)} days')

    stats = {
        'games_selected': 0,
        'games_compared': 0,
        'samples_selected': 0,
        'samples_compared': 0,
        'metadata_mismatches': 0,
        'sample_count_mismatches': 0,
        'shape_errors': 0,
        'obs_mismatches': 0,
        'mask_mismatches': 0,
        'singleton_reference_masks': 0,
        'singleton_reference_mask_mismatches': 0,
        'general_mask_mismatches': 0,
        'illegal_actions': 0,
        'compat_skips': 0,
    }
    mismatch_details = []
    detail_counts = Counter()
    differing_rows = Counter()
    row_examples = {}
    native_seconds = 0.0
    capture_seconds = 0.0

    def record(kind, path, game_index, sample_index, detail):
        stats[kind] += 1
        if detail_counts[kind] < 10:
            detail_counts[kind] += 1
            mismatch_details.append(
                f'{kind} file={Path(path).name} game={game_index} '
                f'sample={sample_index}: {detail}'
            )

    def compare_mode(mode, mode_files, v4_data, v5_data, *, augmented):
        nonlocal capture_seconds
        for path, games_v4, games_v5 in zip(mode_files, v4_data, v5_data):
            if len(games_v4) != len(games_v5):
                record(
                    'metadata_mismatches', path, -1, -1,
                    f'{mode} game count {len(games_v4)} vs {len(games_v5)}',
                )

            raw_log = read_log(path)
            if augmented:
                raw_log = augment_log(raw_log)
            events = [json.loads(line) for line in raw_log.splitlines() if line.strip()]

            for game_index, game in enumerate(games_v4):
                if game_index >= len(games_v5):
                    record(
                        'metadata_mismatches', path, game_index, -1,
                        f'{mode} native v5 game missing',
                    )
                    continue

                player_id, actions, event_indices, at_kan_select = metadata_of(game)
                if metadata_of(games_v5[game_index]) != (
                    player_id, actions, event_indices, at_kan_select,
                ):
                    record(
                        'metadata_mismatches', path, game_index, -1,
                        f'{mode} v4/v5 player/actions/event/kan-select differ',
                    )

                if not actions:
                    continue
                stats['games_selected'] += 1
                stats['samples_selected'] += len(actions)
                native_obs = game['obs']
                native_masks = game['masks']

                try:
                    started = time.perf_counter()
                    ref_obs, ref_masks = capture_reference(
                        raw_log,
                        player_id,
                        actions,
                        event_indices,
                        at_kan_select,
                        reference_path,
                    )
                    capture_seconds += time.perf_counter() - started
                except (CompatibilityError, ValueError) as error:
                    record(
                        'compat_skips', path, game_index, -1,
                        f'{mode}: {error}',
                    )
                    continue

                counts = {
                    'actions': len(actions),
                    'native_obs': len(native_obs),
                    'native_masks': len(native_masks),
                    'ref_obs': len(ref_obs),
                    'ref_masks': len(ref_masks),
                }
                if len(set(counts.values())) != 1:
                    record(
                        'sample_count_mismatches', path, game_index, -1,
                        f'{mode}: {counts}',
                    )
                    continue

                stats['games_compared'] += 1
                for sample_index in range(len(actions)):
                    n_obs = np.asarray(native_obs[sample_index])
                    r_obs = np.asarray(ref_obs[sample_index])
                    n_mask = np.asarray(native_masks[sample_index], dtype=np.bool_)
                    r_mask = np.asarray(ref_masks[sample_index], dtype=np.bool_)
                    stats['samples_compared'] += 1

                    if n_obs.shape != EXPECTED_OBS_SHAPE:
                        record(
                            'shape_errors', path, game_index, sample_index,
                            f'{mode} native obs shape {n_obs.shape}',
                        )
                    if r_obs.shape != EXPECTED_OBS_SHAPE:
                        record(
                            'shape_errors', path, game_index, sample_index,
                            f'{mode} reference obs shape {r_obs.shape}',
                        )
                    if n_mask.shape != EXPECTED_MASK_SHAPE:
                        record(
                            'shape_errors', path, game_index, sample_index,
                            f'{mode} native mask shape {n_mask.shape}',
                        )
                    if r_mask.shape != EXPECTED_MASK_SHAPE:
                        record(
                            'shape_errors', path, game_index, sample_index,
                            f'{mode} reference mask shape {r_mask.shape}',
                        )

                    if n_obs.shape == r_obs.shape and not np.array_equal(n_obs, r_obs):
                        diff = int(np.count_nonzero(n_obs != r_obs))
                        rows, columns = np.nonzero(n_obs != r_obs)
                        differing_rows.update(set(rows.tolist()))
                        for row in set(rows.tolist()):
                            if row not in row_examples:
                                columns_for_row = np.flatnonzero(n_obs[row] != r_obs[row])[:6]
                                row_examples[row] = {
                                    'file': str(path), 'mode': mode, 'player': player_id,
                                    'sample': sample_index, 'event_index': event_indices[sample_index],
                                    'event': events[event_indices[sample_index]],
                                    'cells': [(int(column), float(n_obs[row, column]), float(r_obs[row, column]))
                                              for column in columns_for_row],
                                }
                        cells = [
                            (int(row), int(column), float(n_obs[row, column]), float(r_obs[row, column]))
                            for row, column in zip(rows[:8], columns[:8])
                        ]
                        record(
                            'obs_mismatches', path, game_index, sample_index,
                            f'{mode}: {diff}/{n_obs.size} cells differ; '
                            f'event={event_indices[sample_index]}; cells(row,col,native,reference)={cells}',
                        )
                    if n_mask.shape == r_mask.shape:
                        native_slots = np.flatnonzero(n_mask).tolist()
                        reference_slots = np.flatnonzero(r_mask).tolist()
                        sample_at_kan_select = bool(at_kan_select[sample_index])
                        singleton_reference = (
                            not sample_at_kan_select
                            and 0 <= int(actions[sample_index]) < 37
                            and reference_slots == [int(actions[sample_index])]
                        )
                        if singleton_reference:
                            stats['singleton_reference_masks'] += 1
                        if not np.array_equal(n_mask, r_mask):
                            differing = np.flatnonzero(n_mask != r_mask).tolist()
                            mismatch_kind = (
                                'singleton_reference_mask_mismatches'
                                if singleton_reference
                                else 'general_mask_mismatches'
                            )
                            stats[mismatch_kind] += 1
                            record(
                                'mask_mismatches', path, game_index, sample_index,
                                f'{mode}: differing slots {differing}; '
                                f'native={native_slots}, reference={reference_slots}, '
                                f'at_kan_select={sample_at_kan_select}, '
                                f'event={event_indices[sample_index]}: {events[event_indices[sample_index]]}',
                            )

                    action = int(actions[sample_index])
                    native_legal = 0 <= action < n_mask.size and bool(n_mask[action])
                    reference_legal = 0 <= action < r_mask.size and bool(r_mask[action])
                    if not native_legal or not reference_legal:
                        record(
                            'illegal_actions', path, game_index, sample_index,
                            f'{mode}: action={action}, native={native_legal}, '
                            f'reference={reference_legal}',
                        )

    raw_v4, raw_v4_seconds = load_native(
        files, 4, augmented=False, encode_observations=True,
    )
    raw_v5, raw_v5_seconds = load_native(
        files, 5, augmented=False, encode_observations=False,
    )
    native_seconds += raw_v4_seconds + raw_v5_seconds
    compare_mode(
        'raw',
        files,
        extract_native(raw_v4, include_tensors=True),
        extract_native(raw_v5, include_tensors=False),
        augmented=False,
    )

    aug_files = files[:AUGMENTED_SUBSET]
    aug_v4, aug_v4_seconds = load_native(
        aug_files, 4, augmented=True, encode_observations=True,
    )
    aug_v5, aug_v5_seconds = load_native(
        aug_files, 5, augmented=True, encode_observations=False,
    )
    native_seconds += aug_v4_seconds + aug_v5_seconds
    compare_mode(
        'augmented',
        aug_files,
        extract_native(aug_v4, include_tensors=True),
        extract_native(aug_v5, include_tensors=False),
        augmented=True,
    )

    print()
    print(f'native loading: {native_seconds:.2f}s')
    print(f'reference capture: {capture_seconds:.2f}s')
    for key, value in stats.items():
        print(f'{key}: {value}')
    print(f'differing_rows(samples): {differing_rows.most_common()}')
    if mismatch_details:
        print()
        for detail in mismatch_details:
            print(' ', detail)

    failure_keys = (
        'metadata_mismatches',
        'sample_count_mismatches',
        'shape_errors',
        'obs_mismatches',
        'mask_mismatches',
        'illegal_actions',
        'compat_skips',
    )
    failed = (
        stats['samples_compared'] == 0
        or stats['games_compared'] == 0
        or stats['samples_compared'] != stats['samples_selected']
        or any(stats[key] for key in failure_keys)
    )
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with open(libriichi.__file__, 'rb') as stream:
            native_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        with reference_path.open('rb') as stream:
            reference_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        result = {
            'passed': not failed, 'stats': stats,
            'differing_rows': dict(differing_rows.most_common()),
            'row_examples': row_examples,
            'details': mismatch_details,
            'native_extension': libriichi.__file__, 'native_sha256': native_hash,
            'reference_extension': str(reference_path), 'reference_sha256': reference_hash,
            'files': [str(path) for path in files], 'augmented_files': len(aug_files),
            'native_seconds': native_seconds, 'reference_seconds': capture_seconds,
        }
        with options.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print()
    print('RESULT:', 'FAIL' if failed else 'PASS — native v4 is bit-exact vs reference')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
