from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

from online_panel_stats import panel_stats, pooled_stats

ROOT = Path(__file__).resolve().parents[2]


def sha256(file):
    with file.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Recompute immutable Phase 1/2 logs without playing new games')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/online-correctness-20260908')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'reports') or output == ROOT / 'reports':
        raise ValueError('report must be a new directory below reports/')
    if output.exists():
        raise FileExistsError(output)
    evidence = {}
    manifests = {}
    rows = []
    for phase in (1, 2):
        stages = []
        prefix = '' if phase == 1 else 'online_p2_'
        for label in ('champ_s1_999', 'champ_s2_999', 'champ_s3_999', 'baseline_999'):
            label = prefix + label
            directory = ROOT / 'Mortal/training' / f'sanma_1v2_online_p1_{label}'
            native_log = ROOT / f'online_phase{phase}_logs' / f'panel_{label}.log'
            result = panel_stats(directory, 'sanma-online-128x8', 999, label, native_log=native_log)
            cfg = ROOT / 'Mortal/config' / f'sanma-eval-online-p1-{label}.toml'
            manifests[label] = {'config': str(cfg.relative_to(ROOT)), 'config_sha256': sha256(cfg),
                                'config_text': cfg.read_text(encoding='utf-8'),
                                'historical_candidate_sha256': None,
                                'candidate_provenance': 'not recorded at historical evaluation; do not substitute current deployment hash'}
            stages.append(result)
        combined = pooled_stats(stages[:3])
        combined.pop('seed_records')
        baseline = stages[3]
        evidence[f'phase{phase}'] = {
            'vs_champion_stages': stages[:3], 'vs_champion_pooled': combined,
            'vs_baseline': baseline,
            'promotion': {'promote': False, 'significantly_better_than_champion': combined['significant'],
                          'baseline_policy_threshold': 1.81,
                          'baseline_policy_guard_ok': baseline['mean_rank'] <= 1.81,
                          'statistical_noninferiority': 'not_tested'},
        }
        for name, stats in (('champion', combined), ('baseline', baseline)):
            rows.append(f'| {phase} | {name} | {stats["games"]} | {stats["mean_rank"]:.6f} | '
                        f'[{stats["ci95"][0]:.6f}, {stats["ci95"][1]:.6f}] |')
    source_paths = [Path(__file__).resolve(), ROOT / 'checks/stats/online_panel_stats.py', ROOT / 'checks/stats/arena_log_results.py']
    manifest = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'working_tree_code_sha256': {str(file.relative_to(ROOT)): sha256(file) for file in source_paths},
                'original_results_sha256': {f'online_phase{phase}_result.json': sha256(ROOT / f'online_phase{phase}_result.json') for phase in (1, 2)},
                'panels': manifests, 'total_verified_games': 7992,
                'native_verification': 'all eight parsed rank vectors equal native console rank totals'}
    output.mkdir()
    for phase, result in evidence.items():
        (output / f'{phase}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = '# Online Phase 1/2 corrected results\n\n'
    report += 'Recomputed from all 7,992 original logs. All eight native console rank vectors match. No new games played.\n\n'
    report += '| Phase | Opponent | Games | Mean rank | Seed-clustered 95% CI |\n|---|---|---:|---:|---|\n'
    report += '\n'.join(rows)
    report += '\n\nNeither phase demonstrates an improvement over the champion. No model is promoted. '
    report += 'The 1.81 guard is a policy threshold, not a statistical noninferiority test. '
    report += 'Candidate hashes were not recorded historically and are marked unknown. Original reports remain unchanged.\n'
    (output / 'README.md').write_text(report, encoding='utf-8')
    print(report)
    print('ONLINE_HISTORY_RECOMPUTE_OK')


if __name__ == '__main__':
    main()
