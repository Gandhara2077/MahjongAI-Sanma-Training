import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import toml
from automation_pipeline import (
    AutomationPipeline,
    Candidate,
    EvalResult,
    _atomic_json,
    _safe_id,
    evaluation_result_from_blocks,
    inspect_checkpoint,
)


def benchmark_candidates(models_dir: Path, reference_fingerprint: str) -> list[Candidate]:
    unique: dict[str, tuple[Path, int | None]] = {}
    for sidecar in sorted(Path(models_dir).glob('*.json')):
        checkpoint = sidecar.with_suffix('.pth')
        if not checkpoint.is_file():
            continue
        with sidecar.open(encoding='utf-8') as file:
            metadata = json.load(file)
        fingerprint = str(metadata.get('fingerprint', '')).strip()
        if not fingerprint or fingerprint == reference_fingerprint:
            continue
        step = metadata.get('step')
        step = int(step) if step is not None else None
        previous = unique.get(fingerprint)
        if previous is None or (len(checkpoint.name), checkpoint.name) < (
            len(previous[0].name), previous[0].name
        ):
            unique[fingerprint] = (checkpoint.resolve(), step)

    candidates = [
        Candidate(
            f'step{step if step is not None else "unknown"}-{fingerprint[:8]}',
            checkpoint,
            step,
            fingerprint,
        )
        for fingerprint, (checkpoint, step) in unique.items()
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.step is None,
            candidate.step if candidate.step is not None else 0,
            candidate.fingerprint,
        ),
    )


def collect_historical_seeds(root: Path) -> set[int]:
    seeds: set[int] = set()
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.json', '.toml'}:
            continue
        try:
            if path.suffix.lower() == '.json':
                with path.open(encoding='utf-8') as file:
                    payload = json.load(file)
            else:
                payload = toml.load(path)
        except (OSError, ValueError, toml.TomlDecodeError):
            continue
        _collect_seed_values(payload, seeds)
    return seeds


def _collect_seed_values(value: Any, seeds: set[int], seed_context: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _collect_seed_values(nested, seeds, seed_context or 'seed' in str(key).lower())
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _collect_seed_values(nested, seeds, seed_context)
        return
    if not seed_context or isinstance(value, bool):
        return
    try:
        seeds.add(int(value))
    except (TypeError, ValueError):
        return


def fresh_common_seeds(
    benchmark_id: str,
    reference_fingerprint: str,
    count: int,
    forbidden: set[int],
) -> list[int]:
    if count < 0:
        raise ValueError('seed count cannot be negative')
    unavailable = {int(seed) for seed in forbidden}
    seeds: list[int] = []
    for index in range(count):
        nonce = 0
        while True:
            text = f'{benchmark_id}:{reference_fingerprint}:common-block:{index}:nonce:{nonce}'
            digest = hashlib.sha256(text.encode('utf-8')).digest()
            seed = int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)
            if seed not in unavailable:
                break
            nonce += 1
        seeds.append(seed)
        unavailable.add(seed)
    return seeds


def pending_evaluations(
    candidates: Sequence[Candidate],
    *,
    block_count: int,
    completed: Mapping[int, set[str]],
) -> list[tuple[int, Candidate]]:
    pending = []
    for block_index in range(block_count):
        completed_fingerprints = completed.get(block_index, set())
        for candidate in candidates:
            if candidate.fingerprint not in completed_fingerprints:
                pending.append((block_index, candidate))
    return pending


def new_benchmark_manifest(
    benchmark_id: str,
    reference: Candidate,
    candidates: Sequence[Candidate],
    *,
    block_games: int,
    total_games: int,
    seeds: Sequence[int],
    historical_seed_count: int,
) -> dict[str, Any]:
    if block_games <= 0 or total_games <= 0:
        raise ValueError('benchmark game counts must be positive')
    if total_games % block_games:
        raise ValueError('total games must be divisible by block games')
    block_count = total_games // block_games
    if len(seeds) != block_count:
        raise ValueError(f'expected {block_count} common seeds, got {len(seeds)}')
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError('common benchmark seeds must be unique')
    fingerprints = [candidate.fingerprint for candidate in candidates]
    if not candidates or len(set(fingerprints)) != len(fingerprints):
        raise ValueError('benchmark candidates must be non-empty and fingerprint-unique')
    if reference.fingerprint in fingerprints:
        raise ValueError('reference model cannot also be a benchmark candidate')
    now = datetime.now().isoformat(timespec='seconds')
    return {
        'version': 1,
        'benchmark_id': benchmark_id,
        'status': 'prepared',
        'created_at': now,
        'updated_at': now,
        'reference': _candidate_payload(reference),
        'settings': {
            'block_games': int(block_games),
            'total_games': int(total_games),
            'block_count': block_count,
        },
        'seed_audit': {
            'historical_seed_count': int(historical_seed_count),
            'all_generated_seeds_were_fresh': True,
        },
        'candidates': [_candidate_payload(candidate) for candidate in candidates],
        'complete_common_blocks': 0,
        'common_games': 0,
        'aggregates': {},
        'common_aggregates': {},
        'ranking': [],
        'blocks': [
            {'index': index, 'seed': int(seed), 'results': {}}
            for index, seed in enumerate(seeds)
        ],
    }


def record_benchmark_result(
    manifest: dict[str, Any],
    block_index: int,
    candidate: Candidate,
    result: Any,
    log_dir: Path,
) -> None:
    if not isinstance(result, EvalResult):
        raise TypeError('benchmark result must be an EvalResult')
    if result.fingerprint != candidate.fingerprint:
        raise ValueError('result fingerprint does not match candidate')
    blocks = {int(block['index']): block for block in manifest['blocks']}
    if block_index not in blocks:
        raise ValueError(f'unknown benchmark block: {block_index}')
    expected_games = int(manifest['settings']['block_games'])
    if result.games != expected_games:
        raise ValueError(f'block produced {result.games} games, expected {expected_games}')
    payload = _result_payload(result)
    payload['log_dir'] = str(log_dir)
    payload['recorded_at'] = datetime.now().isoformat(timespec='seconds')
    existing = blocks[block_index]['results'].get(candidate.fingerprint)
    if existing is not None:
        if existing['rankings'] != payload['rankings']:
            raise ValueError('refusing to overwrite a conflicting benchmark result')
        return
    blocks[block_index]['results'][candidate.fingerprint] = payload
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')


def refresh_benchmark_aggregates(
    manifest: dict[str, Any],
    candidates: Sequence[Candidate],
) -> None:
    candidate_fingerprints = {candidate.fingerprint for candidate in candidates}
    complete_blocks = [
        block for block in manifest['blocks']
        if candidate_fingerprints <= set(block['results'])
    ]
    aggregates = {}
    for candidate in candidates:
        candidate_blocks = [
            block['results'][candidate.fingerprint]
            for block in manifest['blocks']
            if candidate.fingerprint in block['results']
        ]
        if candidate_blocks:
            aggregates[candidate.fingerprint] = _result_payload(
                evaluation_result_from_blocks(candidate, candidate_blocks)
            )
    manifest['complete_common_blocks'] = len(complete_blocks)
    manifest['common_games'] = (
        len(complete_blocks) * int(manifest['settings']['block_games'])
    )
    manifest['aggregates'] = aggregates
    common_aggregates = {}
    for candidate in candidates:
        common_candidate_blocks = [
            block['results'][candidate.fingerprint]
            for block in complete_blocks
        ]
        if common_candidate_blocks:
            common_aggregates[candidate.fingerprint] = _result_payload(
                evaluation_result_from_blocks(candidate, common_candidate_blocks)
            )
    manifest['common_aggregates'] = common_aggregates
    manifest['ranking'] = [
        result['fingerprint']
        for result in sorted(
            common_aggregates.values(),
            key=lambda result: (result['avg_rank'], -result['avg_pt']),
        )
    ]
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')


def completed_evaluations(manifest: Mapping[str, Any]) -> dict[int, set[str]]:
    return {
        int(block['index']): set(block.get('results', {}))
        for block in manifest['blocks']
    }


def validate_benchmark_manifest(
    manifest: Mapping[str, Any],
    benchmark_id: str,
    reference: Candidate,
    candidates: Sequence[Candidate],
    *,
    block_games: int,
    total_games: int,
) -> None:
    if manifest.get('benchmark_id') != benchmark_id:
        raise ValueError('benchmark id does not match the resume manifest')
    if manifest.get('reference', {}).get('fingerprint') != reference.fingerprint:
        raise ValueError('reference fingerprint does not match the resume manifest')
    expected_fingerprints = {candidate.fingerprint for candidate in candidates}
    recorded_fingerprints = {
        str(candidate['fingerprint']) for candidate in manifest.get('candidates', [])
    }
    if recorded_fingerprints != expected_fingerprints:
        raise ValueError('candidate fingerprints do not match the resume manifest')
    expected_settings = {
        'block_games': int(block_games),
        'total_games': int(total_games),
        'block_count': int(total_games) // int(block_games),
    }
    recorded_settings = manifest.get('settings', {})
    if any(recorded_settings.get(key) != value for key, value in expected_settings.items()):
        raise ValueError('benchmark settings do not match the resume manifest')


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        'model_id': candidate.model_id,
        'path': str(candidate.path),
        'step': candidate.step,
        'fingerprint': candidate.fingerprint,
    }


def _result_payload(result: EvalResult) -> dict[str, Any]:
    return {
        **_candidate_payload(Candidate(
            result.model_id,
            result.path,
            result.step,
            result.fingerprint,
        )),
        'rankings': list(result.rankings),
        'games': result.games,
        'avg_rank': result.avg_rank,
        'avg_pt': result.avg_pt,
        'rank_ci_low': result.rank_ci_low,
        'rank_ci_high': result.rank_ci_high,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Resume-safe common-seed benchmark for retained Mortal models.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'automation.toml',
    )
    parser.add_argument('--benchmark-id', required=True)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--expected-models', type=int, default=5)
    parser.add_argument('--block-games', type=int, default=2000)
    parser.add_argument('--total-games', type=int, default=20000)
    parser.add_argument('--timeout-hours', type=float, default=6.0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)

    if args.block_games <= 0 or args.total_games <= 0:
        parser.error('game counts must be positive')
    from libriichi.consts import NUM_PLAYERS
    if args.block_games % NUM_PLAYERS or args.total_games % args.block_games:
        parser.error(
            f'game counts must be multiples of {NUM_PLAYERS} and total must divide into blocks'
        )
    if args.expected_models <= 0:
        parser.error('expected model count must be positive')

    pipeline = AutomationPipeline(
        args.config,
        stages=('evaluate',),
    )
    reference = inspect_checkpoint(pipeline.reference_model, 'baseline').candidate
    models_dir = (
        args.models_dir.resolve() if args.models_dir is not None else pipeline.models_dir
    )
    discovered = benchmark_candidates(models_dir, reference.fingerprint)
    if len(discovered) != args.expected_models:
        raise RuntimeError(
            f'expected {args.expected_models} unique non-reference models, '
            f'found {len(discovered)} in {models_dir}'
        )

    candidates = []
    for candidate in discovered:
        inspected = inspect_checkpoint(candidate.path, candidate.model_id).candidate
        if inspected.fingerprint != candidate.fingerprint:
            raise RuntimeError(
                f'sidecar fingerprint mismatch for {candidate.path}: '
                f'{candidate.fingerprint} != {inspected.fingerprint}'
            )
        candidates.append(Candidate(
            candidate.model_id,
            candidate.path,
            inspected.step,
            inspected.fingerprint,
        ))

    benchmark_dir = pipeline.output_root / 'benchmarks' / _safe_id(args.benchmark_id)
    report_file = benchmark_dir / 'report.json'
    block_count = args.total_games // args.block_games
    manifest: dict[str, Any] | None = None
    try:
        if report_file.exists():
            with report_file.open(encoding='utf-8') as file:
                manifest = json.load(file)
            validate_benchmark_manifest(
                manifest,
                args.benchmark_id,
                reference,
                candidates,
                block_games=args.block_games,
                total_games=args.total_games,
            )
            print(f'Resuming benchmark from {report_file}', flush=True)
        else:
            if benchmark_dir.exists() and any(benchmark_dir.iterdir()):
                raise RuntimeError(
                    f'benchmark directory is non-empty without a report: {benchmark_dir}'
                )
            history_root = pipeline.mortal_root / 'training'
            historical_seeds = collect_historical_seeds(history_root)
            seeds = fresh_common_seeds(
                args.benchmark_id,
                reference.fingerprint,
                block_count,
                historical_seeds,
            )
            if not set(seeds).isdisjoint(historical_seeds):
                raise RuntimeError('generated benchmark seeds overlap historical seeds')
            manifest = new_benchmark_manifest(
                args.benchmark_id,
                reference,
                candidates,
                block_games=args.block_games,
                total_games=args.total_games,
                seeds=seeds,
                historical_seed_count=len(historical_seeds),
            )
            serialized_history = ','.join(str(seed) for seed in sorted(historical_seeds))
            manifest['seed_audit'].update({
                'history_root': str(history_root),
                'historical_seed_digest': hashlib.sha256(
                    serialized_history.encode('ascii')
                ).hexdigest(),
                'historical_seeds': sorted(historical_seeds),
                'generated_seeds': seeds,
            })
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(report_file, manifest)
            print(f'Prepared fresh benchmark at {report_file}', flush=True)

        print(
            f'Reference {reference.fingerprint[:8]}: {reference.path}\n'
            f'Candidates ({len(candidates)}): '
            + ', '.join(candidate.model_id for candidate in candidates)
            + f'\nSchedule: {block_count} blocks x {args.block_games} games x '
              f'{len(candidates)} models',
            flush=True,
        )
        if args.prepare_only:
            return 0

        manifest['status'] = 'running'
        manifest['runtime'] = {
            'pid': os.getpid(),
            'started_or_resumed_at': datetime.now().isoformat(timespec='seconds'),
        }
        manifest.pop('last_error', None)
        _atomic_json(report_file, manifest)

        pending = pending_evaluations(
            candidates,
            block_count=block_count,
            completed=completed_evaluations(manifest),
        )
        total_tasks = block_count * len(candidates)
        completed_tasks = total_tasks - len(pending)
        for block_index, candidate in pending:
            block = manifest['blocks'][block_index]
            seed = int(block['seed'])
            block_root = benchmark_dir / f'block_{block_index:03d}'
            print(
                f'[{completed_tasks + 1}/{total_tasks}] block {block_index + 1}/{block_count} '
                f'{candidate.model_id}, seed={seed}, games={args.block_games}',
                flush=True,
            )
            result = pipeline._evaluate_candidate(
                candidate,
                reference,
                args.block_games,
                seed,
                block_root,
                args.timeout_hours,
            )
            log_dir = block_root / _safe_id(candidate.model_id)
            record_benchmark_result(
                manifest,
                block_index,
                candidate,
                result,
                log_dir,
            )
            refresh_benchmark_aggregates(manifest, candidates)
            completed_tasks += 1
            manifest['status'] = 'running'
            manifest['progress'] = {
                'completed_tasks': completed_tasks,
                'total_tasks': total_tasks,
                'complete_common_blocks': manifest['complete_common_blocks'],
                'common_games_per_model': manifest['common_games'],
            }
            _atomic_json(report_file, manifest)
            print(
                f'Completed {candidate.model_id}: rank={result.avg_rank:.4f}, '
                f'pt={result.avg_pt:+.4f}; common games/model={manifest["common_games"]}',
                flush=True,
            )

        refresh_benchmark_aggregates(manifest, candidates)
        if manifest['complete_common_blocks'] != block_count:
            raise RuntimeError('benchmark ended without all common blocks complete')
        manifest['status'] = 'complete'
        manifest['completed_at'] = datetime.now().isoformat(timespec='seconds')
        manifest['progress'] = {
            'completed_tasks': total_tasks,
            'total_tasks': total_tasks,
            'complete_common_blocks': block_count,
            'common_games_per_model': args.total_games,
        }
        _atomic_json(report_file, manifest)
        print('Benchmark complete. Final ranking:', flush=True)
        for index, fingerprint in enumerate(manifest['ranking'], start=1):
            result = manifest['common_aggregates'][fingerprint]
            print(
                f'{index}. {result["model_id"]}: rank={result["avg_rank"]:.5f}, '
                f'pt={result["avg_pt"]:+.4f}, '
                f'95% CI=[{result["rank_ci_low"]:.5f}, {result["rank_ci_high"]:.5f}]',
                flush=True,
            )
        return 0
    except BaseException as error:
        if manifest is not None:
            manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
            manifest['last_error'] = {
                'type': type(error).__name__,
                'message': str(error),
            }
            _atomic_json(report_file, manifest)
        raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed blocks remain resumable.', file=sys.stderr)
        raise SystemExit(130)
