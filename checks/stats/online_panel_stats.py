"""Strict, seed-clustered 1v2 statistics shared by runners and replay audits."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import re
import statistics

from arena_log_results import game_result, rank_of

FILE_PATTERN = re.compile(r'(\d+)_(\d+)_([abc])\.json\.gz')
RANK_PATTERN = re.compile(r'challenger rankings:\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]')


def pooled_stats(stages):
    records = [record for stage in stages for record in stage['seed_records']]
    identities = [(record['seed_key'], record['seed']) for record in records]
    if not records or len(set(identities)) != len(identities):
        raise ValueError('empty panel or repeated full seed identity')
    means = []
    ranks = Counter()
    for record in records:
        games = record['games']
        if len(games) != 3 or {game['rotation'] for game in games} != set('abc'):
            raise ValueError('seed requires exactly three distinct rotations')
        if {game['challenger_seat'] for game in games} != {0, 1, 2}:
            raise ValueError('challenger must rotate through all seats')
        values = [game['challenger_rank'] for game in games]
        if any(type(value) is not int or value not in (1, 2, 3) for value in values):
            raise ValueError('invalid challenger rank')
        ranks.update(values)
        means.append(statistics.mean(values))
    mean = statistics.mean(means)
    error = statistics.stdev(means) / math.sqrt(len(means)) if len(means) >= 2 else None
    interval = [mean - 1.96 * error, mean + 1.96 * error] if error is not None else None
    return {'games': len(records) * 3, 'seeds': len(records), 'mean_rank': mean,
            'standard_error': error, 'ci95': interval, 'effect': mean - 2.0,
            'significant': interval is not None and interval[1] < 2.0,
            'inference_valid': error is not None,
            'rank_counts': [ranks[value] for value in (1, 2, 3)],
            'method': 'normal_approximation_on_three_rotation_seed_means',
            'seed_records': records}


def panel_stats(panel_dir, challenger_name, expected_games, panel_id=None, native_log=None):
    directory = Path(panel_dir)
    panel_id = panel_id or directory.name
    if type(expected_games) is not int or expected_games <= 0 or expected_games % 3:
        raise ValueError('positive complete rotation budget required')
    files = sorted(directory.glob('*.json.gz'))
    if len(files) != expected_games:
        raise ValueError(f'panel expected {expected_games} logs, found {len(files)}: {directory}')
    grouped = defaultdict(list)
    for file in files:
        match = FILE_PATTERN.fullmatch(file.name)
        if not match:
            raise ValueError(f'invalid log filename: {file}')
        seed, seed_key = (int(value) for value in match.groups()[:2])
        with gzip.open(file, 'rt', encoding='utf-8') as source:
            events = [json.loads(line) for line in source if line.strip()]
        if not events or not all(isinstance(event, dict) for event in events):
            raise ValueError(f'invalid event stream: {file}')
        if events[0].get('seed') != [seed, seed_key]:
            raise ValueError(f'filename/header seed mismatch: {file}')
        parsed = game_result(events, challenger_name=challenger_name)
        if parsed is None:
            raise ValueError(f'invalid or incomplete game: {file}')
        seat, scores = parsed
        with file.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        grouped[(seed_key, seed)].append({
            'rotation': match.group(3), 'challenger_seat': seat,
            'scores': scores, 'challenger_rank': rank_of(seat, scores),
            'source': file.name, 'source_sha256': digest,
        })
    records = [{'panel_id': panel_id, 'seed_key': key, 'seed': seed,
                'games': sorted(games, key=lambda game: game['rotation'])}
               for (key, seed), games in sorted(grouped.items())]
    result = pooled_stats([{'seed_records': records}])
    result.update(panel_id=panel_id, challenger_name=challenger_name, source_directory=str(directory.resolve()),
                  native_summary={'status': 'missing', 'reason': 'no native summary supplied'})
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        counts = summary.get('rankings')
        if isinstance(counts, dict):
            counts = counts.get(challenger_name)
        if not isinstance(counts, list) or counts != result['rank_counts']:
            raise ValueError(f'unsupported or inconsistent native summary rankings: {summary_path}')
        for key in ('games', 'log_count', 'expected_log_count'):
            if key in summary and summary[key] != expected_games:
                raise ValueError(f'native summary count mismatch: {summary_path}')
        result['native_summary'] = {'status': 'verified', 'source': str(summary_path)}
    if native_log is not None:
        native_log = Path(native_log)
        matches = RANK_PATTERN.findall(native_log.read_text(encoding='utf-8', errors='strict'))
        if not matches:
            raise ValueError(f'no native rank counts in {native_log}')
        counts = [sum(int(match[index]) for match in matches) for index in range(3)]
        if counts != result['rank_counts'] or sum(counts) != expected_games:
            raise ValueError(f'native ranks {counts} differ from parsed {result["rank_counts"]}: {native_log}')
        with native_log.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        result['native_summary'] = {'status': 'verified', 'source': str(native_log.resolve()),
                                    'sha256': digest, 'rank_counts': counts}
    return result
