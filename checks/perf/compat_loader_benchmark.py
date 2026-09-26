"""Controlled end-to-end benchmark for the sanma compat training path.

Both arms load the same gzip corpus files and run the same historical v4
reference replay. The only variable is whether the native v5 metadata loader
also materializes observations/masks that the compat path immediately drops.
The optimized arm requires the current native extension.
"""

import argparse
import gzip
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / 'Mortal' / 'mortal'
sys.path.insert(0, str(MORTAL))

from libriichi.dataset import GameplayLoader  # noqa: E402
from libriichi3p_compat import capture_replay, prepare_replay_events  # noqa: E402


def run_arm(files, reference_path, *, encode_observations, augmented):
    kwargs = dict(
        version=5,
        oracle=False,
        player_names=None,
        excludes=None,
        trust_seed=False,
        always_include_kan_select=True,
        augmented=augmented,
    )
    try:
        loader = GameplayLoader(**kwargs, encode_observations=encode_observations)
    except TypeError as error:
        if 'encode_observations' not in str(error) or encode_observations:
            # The legacy arm may run against an older pyd; the optimized arm
            # must fail rather than silently benchmark the wrong implementation.
            if not encode_observations:
                raise RuntimeError(
                    'active native extension lacks encode_observations; '
                    'rebuild/activate the optimized sanma extension first'
                ) from error
            loader = GameplayLoader(**kwargs)
        else:
            raise

    started = time.perf_counter()
    data = loader.load_gz_log_files([str(path) for path in files])
    samples = 0
    games = 0
    for path, games_for_file in zip(files, data):
        raw = gzip.open(path, 'rt', encoding='utf-8').read()
        prepared = prepare_replay_events(raw, augmented=augmented)
        for game in games_for_file:
            actions = game.take_actions()
            if not actions:
                continue
            games += 1
            if encode_observations:
                native_obs = game.take_obs()
                native_masks = game.take_masks()
                if len(native_obs) != len(actions) or len(native_masks) != len(actions):
                    raise RuntimeError('native tensor count mismatch')
            capture_replay(
                prepared_events=prepared,
                player_id=game.take_player_id(),
                event_indices=game.take_event_indices(),
                actions=actions,
                at_kan_select=game.take_at_kan_select(),
                reference_path=str(reference_path),
            )
            samples += len(actions)
    return time.perf_counter() - started, games, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--augmented', action='store_true')
    parser.add_argument(
        '--reference',
        type=Path,
        default=ROOT / '.cache' / 'libriichi3p' / 'libriichi3p-3.12-x86_64-pc-windows-msvc.pyd',
    )
    args = parser.parse_args()
    files = sorted((ROOT / 'koromo' / 'mjson' / '2026' / '08' / '09').glob('*.mjson'))[:args.files]
    if not files:
        raise SystemExit('no corpus files found')
    if not args.reference.exists():
        raise SystemExit(f'reference pyd not found: {args.reference}')

    # Warm the reference binary and Rust rayon pools before timing.
    run_arm(files[:1], args.reference, encode_observations=True, augmented=args.augmented)
    run_arm(files[:1], args.reference, encode_observations=False, augmented=args.augmented)

    results = {}
    for name, encode in (('legacy', True), ('metadata_only', False)):
        timings = []
        games = samples = None
        for _ in range(args.repeats):
            elapsed, games, samples = run_arm(
                files,
                args.reference,
                encode_observations=encode,
                augmented=args.augmented,
            )
            timings.append(elapsed)
        median = statistics.median(timings)
        results[name] = (median, samples)
        print(
            f'{name}: files={len(files)} games={games} samples={samples} '
            f'timings_s={[round(x, 3) for x in timings]} '
            f'median_s={median:.3f} samples_per_s={samples / median:.1f}'
        )

    old = results['legacy'][0]
    new = results['metadata_only'][0]
    print(f'speedup_pct={(old / new - 1.) * 100.:.2f}')


if __name__ == '__main__':
    main()
