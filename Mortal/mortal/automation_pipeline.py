from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import toml

from grp_quality import decide_grp_promotion
from libriichi.consts import NUM_PLAYERS
from online_training import resolve_online_resume_numbering


MODEL_SELECTION_VERSION = "bidirectional-adaptive-direct-ladder-v2"

# Average rank at which a player neither gains nor loses rating points:
# 2.5 for yonma, 2.0 for sanma. Every promotion gate compares against it.
BREAKEVEN_RANK = (NUM_PLAYERS + 1) / 2

# Display points per rank for evaluation reports (never used in training).
RANK_POINTS = {
    4: (90, 45, 0, -135),
    3: (90, 0, -90),
}[NUM_PLAYERS]

# Retained-model filename prefix per variant.
MODEL_NAME_PREFIX = {
    4: 'mortal',
    3: 'mortal3p',
}[NUM_PLAYERS]

# Arena evaluation entry points per variant.
ARENA_CONFIG_SECTION = '1v3' if NUM_PLAYERS == 4 else '1v2'
ARENA_SCRIPT = 'one_vs_three.py' if NUM_PLAYERS == 4 else 'one_vs_two.py'
# Sanma arena logs are always gzipped; yonma evaluation counts every file.
ARENA_LOG_GLOB = '*.json.gz' if NUM_PLAYERS == 3 else '*'


@dataclass(frozen=True)
class LogRecord:
    path: Path
    relative_path: str
    size: int
    mtime_ns: int

    @property
    def token(self) -> str:
        return f'{self.size}:{self.mtime_ns}'


@dataclass(frozen=True)
class Candidate:
    model_id: str
    path: Path
    step: int | None
    fingerprint: str


@dataclass(frozen=True)
class EvalResult:
    model_id: str
    path: Path
    rankings: tuple[int, ...]
    step: int | None
    fingerprint: str
    games: int
    avg_rank: float
    avg_pt: float
    rank_ci_low: float
    rank_ci_high: float

    @classmethod
    def from_rankings(
        cls,
        model_id: str,
        path: Path,
        rankings: Sequence[int],
        step: int | None,
        fingerprint: str,
    ) -> 'EvalResult':
        if len(rankings) != NUM_PLAYERS or any(count < 0 for count in rankings):
            raise ValueError(f'rankings must contain {NUM_PLAYERS} non-negative counts')
        counts = tuple(int(count) for count in rankings)
        games = sum(counts)
        if games <= 1:
            raise ValueError('at least two games are required')

        avg_rank = sum(
            count * rank
            for count, rank in zip(counts, range(1, NUM_PLAYERS + 1))
        ) / games
        avg_pt = sum(
            count * point for count, point in zip(counts, RANK_POINTS)
        ) / games
        variance = sum(
            count * (rank - avg_rank) ** 2
            for count, rank in zip(counts, range(1, NUM_PLAYERS + 1))
        ) / (games - 1)
        margin = 1.96 * math.sqrt(variance / games)
        return cls(
            model_id=model_id,
            path=Path(path),
            rankings=counts,
            step=step,
            fingerprint=fingerprint,
            games=games,
            avg_rank=avg_rank,
            avg_pt=avg_pt,
            rank_ci_low=avg_rank - margin,
            rank_ci_high=avg_rank + margin,
        )


@dataclass(frozen=True)
class DirectComparison:
    model_a: Candidate
    model_b: Candidate
    a_vs_b: EvalResult
    b_vs_a: EvalResult
    combined_a: EvalResult

    def result_for(self, candidate: Candidate) -> EvalResult:
        if candidate.fingerprint == self.model_a.fingerprint:
            return self.combined_a
        if candidate.fingerprint != self.model_b.fingerprint:
            raise ValueError(
                f'candidate {candidate.model_id} is not part of this direct comparison'
            )
        # seats rotate between the two directions, so the mirror image of the
        # combined rankings belongs to model_b
        rankings = tuple(reversed(self.combined_a.rankings))
        return EvalResult.from_rankings(
            candidate.model_id,
            candidate.path,
            rankings,
            candidate.step,
            candidate.fingerprint,
        )


def combine_bidirectional_results(
    forward: EvalResult,
    reverse: EvalResult,
) -> EvalResult:
    if forward.fingerprint == reverse.fingerprint:
        raise ValueError('direct comparison requires two different models')
    if forward.games != reverse.games:
        raise ValueError('both direct comparison directions must use the same game count')
    rankings = tuple(
        left + right
        for left, right in zip(forward.rankings, reversed(reverse.rankings))
    )
    return EvalResult.from_rankings(
        forward.model_id,
        forward.path,
        rankings,
        forward.step,
        forward.fingerprint,
    )


def combine_evaluation_batches(
    first: EvalResult,
    second: EvalResult,
) -> EvalResult:
    if first.fingerprint != second.fingerprint:
        raise ValueError('evaluation batches must belong to the same model')
    rankings = tuple(
        left + right
        for left, right in zip(first.rankings, second.rankings)
    )
    return EvalResult.from_rankings(
        first.model_id,
        first.path,
        rankings,
        first.step,
        first.fingerprint,
    )


def is_significant_win(
    result: EvalResult,
    *,
    require_significant: bool = True,
    require_positive_pt: bool = True,
) -> bool:
    if require_significant and result.rank_ci_high >= BREAKEVEN_RANK:
        return False
    if require_positive_pt and result.avg_pt <= 0:
        return False
    return result.avg_rank < BREAKEVEN_RANK


def direct_ladder_update(
    roster: Sequence[Candidate],
    challenger: Candidate,
    compare: Callable[[Candidate, Candidate], EvalResult],
    *,
    top_k: int = 3,
    require_significant: bool = True,
    require_positive_pt: bool = True,
) -> tuple[list[Candidate], list[EvalResult]]:
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    updated = list(roster)
    if len(updated) > top_k:
        raise ValueError('roster is larger than top_k')
    if any(item.fingerprint == challenger.fingerprint for item in updated):
        return updated, []
    if len(updated) < top_k:
        updated.append(challenger)
        return updated, []

    evidence = [compare(challenger, updated[-1])]
    if not is_significant_win(
        evidence[-1],
        require_significant=require_significant,
        require_positive_pt=require_positive_pt,
    ):
        return updated, evidence

    updated[-1] = challenger
    position = len(updated) - 1
    while position > 0:
        evidence.append(compare(challenger, updated[position - 1]))
        if not is_significant_win(
            evidence[-1],
            require_significant=require_significant,
            require_positive_pt=require_positive_pt,
        ):
            break
        updated[position] = updated[position - 1]
        updated[position - 1] = challenger
        position -= 1
    return updated, evidence


def discover_logs(root: Path) -> list[LogRecord]:
    root = Path(root).resolve()
    records = []
    for file_path in root.rglob('*'):
        if not file_path.is_file():
            continue
        if file_path.suffix != '.mjson' and not file_path.name.endswith('.json.gz'):
            continue
        stat = file_path.stat()
        records.append(LogRecord(
            path=file_path.resolve(),
            relative_path=file_path.relative_to(root).as_posix(),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        ))
    return sorted(records, key=lambda record: record.relative_path)


def records_to_manifest(records: Iterable[LogRecord]) -> dict[str, str]:
    return {record.relative_path: record.token for record in records}


def new_logs(
    records: Iterable[LogRecord],
    known_logs: Mapping[str, str],
) -> list[LogRecord]:
    return [
        record
        for record in records
        if known_logs.get(record.relative_path) != record.token
    ]


def dataset_signature(records: Iterable[LogRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode('utf-8'))
        digest.update(b'\0')
        digest.update(record.token.encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sync_native_extension(
    mortal_root: Path,
    *,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    variant: str | None = None,
) -> Path:
    """Place the variant's prebuilt native extension at mortal/libriichi.pyd.

    Lookup order:
    1. <repo>/native/<variant>/libriichi-py<major><minor>.pyd (unified layout)
    2. <mortal_root>/target-py<major><minor>/release/<build artifact> (legacy
       in-repo cargo build layout)
    """
    mortal_root = Path(mortal_root).resolve()
    major, minor = python_version or (sys.version_info.major, sys.version_info.minor)
    system = system or platform.system()
    variant = variant or ('sanma' if NUM_PLAYERS == 3 else 'yonma')
    filenames = {
        'Windows': ('riichi.dll', 'libriichi.pyd'),
        'Linux': ('libriichi.so', 'libriichi.so'),
        'Darwin': ('libriichi.dylib', 'libriichi.so'),
    }
    if system not in filenames:
        raise RuntimeError(f'unsupported native extension platform: {system}')
    legacy_source_name, destination_name = filenames[system]
    candidates = [
        mortal_root.parent / 'native' / variant / f'libriichi-py{major}{minor}.pyd',
        mortal_root / f'target-py{major}{minor}' / 'release' / legacy_source_name,
    ]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        searched = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f'Python {major}.{minor} {variant} native extension build not found; searched: {searched}'
        )
    destination = mortal_root / 'mortal' / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or file_sha256(source) != file_sha256(destination):
        shutil.copy2(source, destination)
    return destination


def split_logs(
    records: Sequence[LogRecord],
    validation_ratio: float,
    seed: str,
) -> tuple[list[LogRecord], list[LogRecord]]:
    if not 0 < validation_ratio < 1:
        raise ValueError('validation_ratio must be between 0 and 1')
    if len(records) < 2:
        raise ValueError('at least two logs are required')

    scored = []
    for record in records:
        digest = hashlib.sha256(
            f'{seed}\0{record.relative_path}'.encode('utf-8')
        ).digest()
        score = int.from_bytes(digest[:8], 'big') / 2**64
        scored.append((score, record))

    train = [record for score, record in scored if score >= validation_ratio]
    validation = [record for score, record in scored if score < validation_ratio]
    if not validation:
        _, record = min(scored, key=lambda item: item[0])
        train.remove(record)
        validation.append(record)
    if not train:
        _, record = max(scored, key=lambda item: item[0])
        validation.remove(record)
        train.append(record)
    return (
        sorted(train, key=lambda record: record.relative_path),
        sorted(validation, key=lambda record: record.relative_path),
    )


def split_grp_logs(
    records: Sequence[LogRecord],
    holdout_ratio: float,
    gate_ratio: float,
    seed: str,
) -> tuple[list[LogRecord], list[LogRecord], list[LogRecord]]:
    if not 0 < gate_ratio < holdout_ratio < 1:
        raise ValueError('GRP ratios must satisfy 0 < gate_ratio < holdout_ratio < 1')
    if len(records) < 3:
        raise ValueError('at least three logs are required for the GRP split')

    scored = []
    for record in records:
        digest = hashlib.sha256(
            f'{seed}\0{record.relative_path}'.encode('utf-8')
        ).digest()
        score = int.from_bytes(digest[:8], 'big') / 2**64
        scored.append((score, record))

    train = [record for score, record in scored if score >= holdout_ratio]
    validation = [
        record
        for score, record in scored
        if gate_ratio <= score < holdout_ratio
    ]
    gate = [record for score, record in scored if score < gate_ratio]
    if len(records) == 3 and (not train or not validation or not gate):
        ordered = [record for _, record in sorted(scored, key=lambda item: item[0])]
        gate, validation, train = [ordered[0]], [ordered[1]], [ordered[2]]
    if not train or not validation or not gate:
        raise ValueError('GRP train, validation, and gate splits must all be non-empty')
    return tuple(
        sorted(group, key=lambda record: record.relative_path)
        for group in (train, validation, gate)
    )


def should_update_grp(
    new_count: int,
    total_count: int,
    state_exists: bool,
    min_new_logs: int,
    min_new_ratio: float,
) -> bool:
    if not state_exists:
        return True
    if new_count >= min_new_logs:
        return True
    return total_count > 0 and new_count / total_count >= min_new_ratio


def should_execute_grp_update(*, enabled: bool, planned_update: bool) -> bool:
    return enabled and planned_update


def aligned_target_step(current_step: int, updates: int, save_every: int) -> int:
    if current_step < 0 or updates < 0 or save_every <= 0:
        raise ValueError('steps must be non-negative and save_every must be positive')
    if updates == 0:
        return current_step
    return math.ceil((current_step + updates) / save_every) * save_every


def estimate_steps_per_epoch(
    log_count: int,
    *,
    batch_size: int,
    samples_per_log: int = 660,
) -> int:
    if log_count < 0 or batch_size <= 0 or samples_per_log <= 0:
        raise ValueError('log_count must be non-negative and sample sizes must be positive')
    return math.ceil(log_count * samples_per_log / batch_size)


def gpu_slot_available(
    memory_used_mb: int,
    utilization: int,
    *,
    max_memory_used_mb: int,
    max_utilization: int,
) -> bool:
    return memory_used_mb <= max_memory_used_mb and utilization <= max_utilization


def validate_grp_gate_settings(settings: Mapping[str, Any]) -> None:
    validation = settings.get('validation', {})
    if int(validation.get('bootstrap_samples', 0)) <= 0:
        raise ValueError('grp.validation.bootstrap_samples must be positive')

    downstream = settings.get('downstream', {})
    if not downstream.get('enabled', True):
        return
    training_seeds = [int(seed) for seed in downstream.get('training_seeds', [])]
    if len(training_seeds) < 2 or len(set(training_seeds)) != len(training_seeds):
        raise ValueError('grp.downstream.training_seeds must contain at least two unique seeds')
    if int(downstream.get('training_updates', 0)) <= 0:
        raise ValueError('grp.downstream.training_updates must be positive')
    games = int(downstream.get('games', 0))
    if games < 1000:
        raise ValueError('grp.downstream.games must be at least 1000')
    if games % NUM_PLAYERS != 0:
        raise ValueError(f'grp.downstream.games must be a multiple of {NUM_PLAYERS}')


def _validate_grp_quality_gate(settings: Mapping[str, Any]) -> None:
    patience = int(settings.get('early_stopping_patience', 0))
    if patience < 0:
        raise ValueError('grp.early_stopping_patience must be non-negative')
    quality = settings.get('quality_gate', {})
    primary_metric = str(quality.get('primary_metric', 'expected_pt_mse'))
    if primary_metric not in {'nll', 'brier', 'expected_pt_mse'}:
        raise ValueError('unsupported grp.quality_gate.primary_metric')
    if int(quality.get('bootstrap_samples', 2000)) <= 0:
        raise ValueError('grp.quality_gate.bootstrap_samples must be positive')
    confidence = float(quality.get('confidence', 0.95))
    if not 0 < confidence < 1:
        raise ValueError('grp.quality_gate.confidence must be between 0 and 1')
    if int(quality.get('min_gate_logs', 100)) <= 0:
        raise ValueError('grp.quality_gate.min_gate_logs must be positive')
    for key in (
        'min_relative_improvement',
        'max_nll_regression',
        'max_brier_regression',
    ):
        if float(quality.get(key, 0.0)) < 0:
            raise ValueError(f'grp.quality_gate.{key} must be non-negative')


def grp_gate_strategy(settings: Mapping[str, Any]) -> str:
    """Select the GRP promotion gate: explicit key wins, otherwise the mere
    presence of a [grp.quality_gate] section selects the quality gate and the
    default stays the validation+downstream gate."""
    strategy = settings.get('gate_strategy')
    if strategy:
        return str(strategy)
    return 'quality_gate' if 'quality_gate' in settings else 'validation'


def direct_game_schedule(settings: Mapping[str, Any]) -> tuple[int, int]:
    maximum = int(settings['validation_games'])
    initial = int(settings.get('direct_initial_games', maximum))
    for key, value in (
        ('direct_initial_games', initial),
        ('validation_games', maximum),
    ):
        if value <= 0 or value % NUM_PLAYERS != 0:
            raise ValueError(
                f'evaluation.{key} must be a positive multiple of {NUM_PLAYERS}'
            )
    if initial > maximum:
        raise ValueError(
            'evaluation.direct_initial_games cannot exceed validation_games'
        )
    return initial, maximum


def frozen_cycle_plan(cycle: dict, proposed_plan: Mapping) -> dict:
    cycle.setdefault('plan', deepcopy(proposed_plan))
    return cycle['plan']


def requested_stages_complete(
    cycle: Mapping,
    stages: Sequence[str],
    config: Mapping,
) -> bool:
    section_by_stage = {'evaluate': 'evaluation'}
    requested = [
        stage
        for stage in stages
        if stage != 'discover'
        and bool(
            config.get(section_by_stage.get(stage, stage), {}).get('enabled', True)
        )
    ]
    return bool(requested) and all(
        cycle.get('stages', {}).get(stage, {}).get('status')
        in ('complete', 'skipped')
        for stage in requested
    )


def _config_path(path: Path) -> str:
    return Path(path).as_posix()


def build_grp_config(
    base: Mapping,
    *,
    state_file: Path,
    index_file: Path,
    tensorboard_dir: Path,
    max_steps: int,
    best_state_file: Path | None = None,
    metrics_file: Path | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    selection_metric: str | None = None,
) -> dict:
    cfg = deepcopy(base)
    cfg['grp']['state_file'] = _config_path(state_file)
    cfg['grp']['control']['tensorboard_dir'] = _config_path(tensorboard_dir)
    cfg['grp']['control']['max_steps'] = max_steps
    cfg['grp']['dataset']['file_index'] = _config_path(index_file)
    if best_state_file is not None:
        cfg['grp']['control']['best_state_file'] = _config_path(best_state_file)
    if metrics_file is not None:
        cfg['grp']['control']['metrics_file'] = _config_path(metrics_file)
    if early_stopping_patience is not None:
        cfg['grp']['control']['early_stopping_patience'] = int(early_stopping_patience)
    if early_stopping_min_delta is not None:
        cfg['grp']['control']['early_stopping_min_delta'] = float(early_stopping_min_delta)
    if selection_metric is not None:
        cfg['grp']['control']['selection_metric'] = str(selection_metric)
    return cfg


def _set_stage_outputs(cfg: dict, run_dir: Path, state_file: Path) -> None:
    cfg['control'].update({
        'state_file': _config_path(state_file),
        'best_state_file': _config_path(run_dir / 'best_train_state.pth'),
        'deployment_file': _config_path(run_dir / 'model_deploy.pth'),
        'best_deployment_file': _config_path(run_dir / 'best_model_deploy.pth'),
        'tensorboard_dir': _config_path(run_dir / 'tensorboard'),
        'snapshot_dir': _config_path(run_dir / 'snapshots'),
    })


def build_offline_config(
    base: Mapping,
    *,
    state_file: Path,
    dataset_index: Path,
    grp_state: Path,
    reference_model: Path,
    run_dir: Path,
    start_step: int,
    save_every: int,
    peak_lr: float,
    final_lr: float,
    warm_up_steps: int,
    updates: int | None = None,
    target_step: int | None = None,
    resume_optimizer: bool = False,
    snapshot_every: int = 0,
    random_seed: int | None = None,
    deterministic: bool = False,
) -> dict:
    if (updates is None) == (target_step is None):
        raise ValueError('exactly one of updates or target_step must be provided')
    cfg = deepcopy(base)
    if target_step is None:
        target_step = aligned_target_step(start_step, updates, save_every)
    run_dir = Path(run_dir)
    _set_stage_outputs(cfg, run_dir, state_file)
    cfg['control'].update({
        'online': False,
        'save_every': save_every,
        'test_every': save_every * 1_000_000,
        'submit_every': save_every,
        'snapshot_every': snapshot_every,
        'max_steps': target_step,
        'resume_optimizer': resume_optimizer,
    })
    cfg['dataset']['file_index'] = _config_path(dataset_index)
    cfg['dataset']['player_names_files'] = []
    cfg['grp']['state_file'] = _config_path(grp_state)
    cfg.setdefault('online_training', {})['phase_start_step'] = start_step
    cfg['freeze_bn']['mortal'] = False
    for mode in ('train', 'test'):
        cfg['baseline'][mode]['state_file'] = _config_path(reference_model)
    cfg['optim']['scheduler'].update({
        'peak': peak_lr,
        'final': final_lr,
        'init': min(1e-8, final_lr),
        'warm_up_steps': warm_up_steps,
        'max_steps': max(0, target_step - start_step),
        'offset': 0,
    })
    if random_seed is not None:
        cfg['control']['random_seed'] = int(random_seed)
    if deterministic:
        cfg['control']['deterministic'] = True
        cfg['control']['enable_cudnn_benchmark'] = False
    return cfg


def build_online_config(
    base: Mapping,
    *,
    state_file: Path,
    dataset_index: Path,
    grp_state: Path,
    reference_model: Path,
    deployed_model: Path,
    historical_model: Path,
    run_dir: Path,
    start_step: int,
    save_every: int,
    snapshot_every: int,
    brain_freeze_updates: int,
    expert_ratio: float,
    peak_lr: float,
    port: int,
    games_per_session: int,
    updates: int | None = None,
    target_step: int | None = None,
    resume_optimizer: bool = False,
    opponents: Sequence[Mapping[str, Any]] | None = None,
    opponent_selection: str | None = None,
    league_models: Sequence[Path] | None = None,
    opponent_weights: Mapping[str, float] | None = None,
) -> dict:
    if (updates is None) == (target_step is None):
        raise ValueError('exactly one of updates or target_step must be provided')
    if games_per_session <= 0 or games_per_session % NUM_PLAYERS != 0:
        raise ValueError(f'games_per_session must be a positive multiple of {NUM_PLAYERS}')
    cfg = deepcopy(base)
    if target_step is None:
        target_step = aligned_target_step(start_step, updates, save_every)
    run_dir = Path(run_dir)
    _set_stage_outputs(cfg, run_dir, state_file)
    cfg['control'].update({
        'online': True,
        'save_every': save_every,
        'test_every': save_every * 1_000_000,
        'submit_every': save_every,
        'snapshot_every': snapshot_every,
        'max_steps': target_step,
        'resume_optimizer': resume_optimizer,
    })
    cfg['dataset']['file_index'] = _config_path(dataset_index)
    cfg['dataset']['player_names_files'] = []
    cfg['grp']['state_file'] = _config_path(grp_state)
    cfg['freeze_bn']['mortal'] = True
    for mode in ('train', 'test'):
        cfg['baseline'][mode]['state_file'] = _config_path(reference_model)

    device = cfg['control'].get('device', 'cuda:0')
    if opponents is not None:
        configured_opponents = deepcopy(list(opponents))
        for opponent in configured_opponents:
            opponent.setdefault('device', device)
            opponent.setdefault('enable_compile', False)
            opponent['state_file'] = _config_path(Path(opponent['state_file']))
    elif league_models is not None or opponent_weights is not None:
        weights = {'anchor': 0.1, 'deployed': 0.4, 'league': 0.5}
        if opponent_weights:
            weights.update({key: float(value) for key, value in opponent_weights.items()})
        if any(value < 0 for value in weights.values()):
            raise ValueError('online opponent weights must be non-negative')
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError('online opponent weights must sum to 1.0')
        selected_league = list(league_models or [historical_model])
        if not selected_league:
            selected_league = [historical_model]
        league_share = weights['league'] / len(selected_league)
        configured_opponents = [
            {
                'name': 'anchor',
                'weight': weights['anchor'],
                'device': device,
                'enable_compile': False,
                'enable_rule_based_agari_guard': True,
                'state_file': _config_path(reference_model),
            },
            {
                'name': 'deployed',
                'weight': weights['deployed'],
                'device': device,
                'enable_compile': False,
                'enable_rule_based_agari_guard': True,
                'state_file': _config_path(deployed_model),
            },
        ]
        configured_opponents.extend(
            {
                'name': f'league-top{index}',
                'weight': league_share,
                'device': device,
                'enable_compile': False,
                'enable_rule_based_agari_guard': True,
                'state_file': _config_path(model),
            }
            for index, model in enumerate(selected_league, start=1)
        )
    else:
        configured_opponents = [
            {
                'name': 'baseline',
                'weight': 0.4,
                'device': device,
                'enable_compile': False,
                'state_file': _config_path(reference_model),
            },
            {
                'name': 'deployed',
                'weight': 0.4,
                'device': device,
                'enable_compile': False,
                'state_file': _config_path(deployed_model),
            },
            {
                'name': 'historical',
                'weight': 0.2,
                'device': device,
                'enable_compile': False,
                'state_file': _config_path(historical_model),
            },
        ]
    cfg['online_training'] = {
        'phase_start_step': start_step,
        'expert_ratio': expert_ratio,
        'brain_freeze_updates': brain_freeze_updates,
        'opponents': configured_opponents,
    }
    if opponent_selection is not None:
        cfg['online_training']['opponent_selection'] = opponent_selection
    cfg['optim']['scheduler'].update({
        'peak': peak_lr,
        'final': peak_lr,
        'init': peak_lr,
        'warm_up_steps': 0,
        'max_steps': max(0, target_step - start_step),
        'offset': 0,
    })
    cfg['train_play']['default']['games'] = games_per_session
    cfg['train_play']['default']['log_dir'] = _config_path(run_dir / 'train_play')
    cfg['test_play']['log_dir'] = _config_path(run_dir / 'test_play')
    cfg['online']['remote']['port'] = port
    cfg['online']['server'].update({
        'buffer_dir': _config_path(run_dir / 'buffer'),
        'drain_dir': _config_path(run_dir / 'drain'),
        'capacity': games_per_session,
        'force_sequential': True,
    })
    return cfg


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.fingerprint in seen:
            continue
        seen.add(candidate.fingerprint)
        unique.append(candidate)
    return unique


def select_champion(
    results: Sequence[EvalResult],
    reference_id: str,
    require_significant: bool = True,
    require_positive_pt: bool = True,
) -> EvalResult:
    reference = next(
        (result for result in results if result.model_id == reference_id),
        None,
    )
    if reference is None:
        raise ValueError(f'reference result not found: {reference_id}')

    eligible = []
    for result in results:
        if result.model_id == reference_id:
            continue
        if require_significant and result.rank_ci_high >= BREAKEVEN_RANK:
            continue
        if require_positive_pt and result.avg_pt <= 0:
            continue
        eligible.append(result)
    if not eligible:
        return reference
    return min(eligible, key=lambda result: (result.avg_rank, -result.avg_pt))


def sequential_evaluation_decision(
    result: EvalResult,
    *,
    min_games: int,
    max_games: int,
    require_positive_pt: bool = True,
) -> str:
    if min_games <= 0 or max_games < min_games:
        raise ValueError('sequential evaluation game limits are invalid')
    if result.games < min_games:
        return 'continue'
    if result.rank_ci_high < BREAKEVEN_RANK and (not require_positive_pt or result.avg_pt > 0):
        return 'promote'
    if result.rank_ci_low >= BREAKEVEN_RANK:
        return 'reject'
    if result.games >= max_games:
        return 'inconclusive'
    return 'continue'


def evaluation_ledger_key(purpose: str, challenger_fingerprint: str, reference_fingerprint: str) -> str:
    payload = f'{purpose}\0{challenger_fingerprint}\0{reference_fingerprint}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def normalize_opponent_pool(opponents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for opponent in opponents:
        fingerprint = str(opponent['fingerprint'])
        weight = float(opponent['weight'])
        if weight < 0:
            raise ValueError('opponent weights must be non-negative')
        if fingerprint in unique:
            unique[fingerprint]['weight'] += weight
            continue
        unique[fingerprint] = deepcopy(dict(opponent))
        unique[fingerprint]['weight'] = weight
    total = sum(float(opponent['weight']) for opponent in unique.values())
    if total <= 0:
        raise ValueError('at least one opponent weight must be positive')
    normalized = list(unique.values())
    for opponent in normalized:
        opponent['weight'] = float(opponent['weight']) / total
    return normalized


def record_evaluation_block(
    entry: dict[str, Any],
    *,
    block_index: int,
    seed: int,
    result: EvalResult,
) -> None:
    blocks = entry.setdefault('blocks', [])
    if any(int(block['index']) == block_index for block in blocks):
        return
    blocks.append({
        'index': int(block_index),
        'seed': int(seed),
        'rankings': list(result.rankings),
        'games': result.games,
        'avg_rank': result.avg_rank,
        'avg_pt': result.avg_pt,
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
    })
    blocks.sort(key=lambda block: int(block['index']))


def evaluation_result_from_blocks(
    candidate: Candidate,
    blocks: Sequence[Mapping[str, Any]],
) -> EvalResult:
    if not blocks:
        raise ValueError('at least one evaluation block is required')
    rankings = [
        sum(int(block['rankings'][index]) for block in blocks)
        for index in range(NUM_PLAYERS)
    ]
    return EvalResult.from_rankings(
        candidate.model_id,
        candidate.path,
        rankings,
        candidate.step,
        candidate.fingerprint,
    )


def evenly_spaced_candidates(candidates: Sequence[Candidate], limit: int) -> list[Candidate]:
    items = list(candidates)
    if limit <= 0:
        raise ValueError('candidate limit must be positive')
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def summarize_grp_validation(
    deltas: Mapping[str, Sequence[float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    if bootstrap_samples <= 0:
        raise ValueError('bootstrap_samples must be positive')
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in deltas.items()}
    if not arrays:
        raise ValueError('at least one metric is required')
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError('metric deltas must have the same non-zero length')

    games = next(iter(lengths))
    rng = np.random.default_rng(seed)
    bootstrapped = {name: np.empty(bootstrap_samples) for name in arrays}
    for index in range(bootstrap_samples):
        sample = rng.integers(0, games, games)
        for name, values in arrays.items():
            bootstrapped[name][index] = values[sample].mean()

    return {
        'games': games,
        'bootstrap_samples': bootstrap_samples,
        'bootstrap_seed': int(seed),
        'deltas': {
            name: {
                'mean': float(values.mean()),
                'ci95_low': float(np.quantile(bootstrapped[name], 0.025)),
                'ci95_high': float(np.quantile(bootstrapped[name], 0.975)),
                'improved_game_fraction': float((values > 0).mean()),
            }
            for name, values in arrays.items()
        },
    }


def grp_validation_passes(
    report: Mapping[str, Any],
    *,
    require_significant: bool = True,
    min_nll_reduction: float = 0.0,
    min_brier_reduction: float = 0.0,
) -> bool:
    deltas = report['deltas']
    value_key = 'ci95_low' if require_significant else 'mean'
    return (
        float(deltas['nll_reduction'][value_key]) > min_nll_reduction
        and float(deltas['brier_reduction'][value_key]) > min_brier_reduction
    )


def aggregate_eval_results(
    results: Sequence[EvalResult],
    model_id: str,
) -> EvalResult:
    if not results:
        raise ValueError('at least one evaluation result is required')
    rankings = [
        sum(result.rankings[index] for result in results)
        for index in range(NUM_PLAYERS)
    ]
    fingerprint = hashlib.sha256(
        '\n'.join(result.fingerprint for result in results).encode('utf-8')
    ).hexdigest()
    steps = [result.step for result in results if result.step is not None]
    return EvalResult.from_rankings(
        model_id,
        results[0].path,
        rankings,
        max(steps) if steps else None,
        fingerprint,
    )


def grp_downstream_passes(
    results: Sequence[EvalResult],
    *,
    require_significant: bool = True,
    require_positive_pt: bool = True,
    require_each_seed_non_regressing: bool = True,
) -> bool:
    if not results:
        return False
    if require_each_seed_non_regressing and any(
        result.avg_rank >= BREAKEVEN_RANK for result in results
    ):
        return False
    combined = aggregate_eval_results(results, 'grp-candidate')
    if require_significant and combined.rank_ci_high >= BREAKEVEN_RANK:
        return False
    if require_positive_pt and combined.avg_pt <= 0:
        return False
    return combined.avg_rank < BREAKEVEN_RANK


def should_promote_grp(
    direct_passed: bool,
    *,
    downstream_enabled: bool,
    downstream_passed: bool,
) -> bool:
    return direct_passed and (not downstream_enabled or downstream_passed)


def atomic_promote_checkpoint(candidate: Path, destination: Path) -> None:
    candidate = Path(candidate)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(candidate, temp_path)
    temp_path.replace(destination)


def ability_filename(result: EvalResult) -> str:
    step = str(result.step) if result.step is not None else 'unknown'
    if NUM_PLAYERS == 3:
        model_id = re.sub(r'[^A-Za-z0-9_.-]+', '-', result.model_id).strip('-')
        return (
            f'{MODEL_NAME_PREFIX}_rank{result.avg_rank:.3f}_pt{result.avg_pt:+.2f}'
            f'_n{result.games}_step{step}_{model_id}_{result.fingerprint[:8]}.pth'
        )
    return (
        f'{MODEL_NAME_PREFIX}_rank{result.avg_rank:.3f}_pt{result.avg_pt:+.2f}'
        f'_n{result.games}_step{step}_{result.fingerprint[:8]}.pth'
    )


def top_model_filename(slot: int, result: EvalResult) -> str:
    if slot not in (1, 2, 3):
        raise ValueError('top model slot must be 1, 2, or 3')
    step = str(result.step) if result.step is not None else 'unknown'
    return (
        f'{MODEL_NAME_PREFIX}_top{slot}_h2hrank{result.avg_rank:.3f}'
        f'_h2hpt{result.avg_pt:+.2f}_n{result.games}'
        f'_step{step}_{result.fingerprint[:8]}.pth'
    )


def compact_checkpoint_state(state: Mapping) -> dict:
    config = state['config']
    resnet = config['resnet']
    return {
        'config': {
            'control': {'version': int(config.get('control', {}).get('version', 4))},
            'resnet': {
                'conv_channels': int(resnet['conv_channels']),
                'num_blocks': int(resnet['num_blocks']),
            },
        },
        'mortal': state['mortal'],
        'current_dqn': state['current_dqn'],
    }


@dataclass(frozen=True)
class CheckpointInfo:
    candidate: Candidate
    full_training_state: bool
    version: int
    resnet: dict[str, Any]


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _safe_id(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-') or 'model'


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write('\n')
    temp_path.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(source, temp_path)
    temp_path.replace(destination)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def _write_toml(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        toml.dump(payload, file)


def _seed_from_text(value: str) -> int:
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)


def inspect_checkpoint(path: Path, model_id: str) -> CheckpointInfo:
    import torch

    path = Path(path).resolve()
    state = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or not {'mortal', 'current_dqn'} <= state.keys():
        raise ValueError(f'not a Mortal gameplay checkpoint: {path}')

    digest = hashlib.sha256()
    for group in ('mortal', 'current_dqn'):
        digest.update(group.encode('ascii'))
        for name, tensor in sorted(state[group].items()):
            digest.update(name.encode('utf-8'))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())

    model_cfg = state.get('config', {})
    version = int(model_cfg.get('control', {}).get('version', 1))
    resnet = model_cfg.get('resnet')
    if not isinstance(resnet, dict):
        raise ValueError(f'checkpoint has no ResNet configuration: {path}')
    full_keys = {'aux_net', 'optimizer', 'scheduler', 'scaler', 'steps', 'timestamp'}
    candidate = Candidate(
        model_id=model_id,
        path=path,
        step=state.get('steps'),
        fingerprint=digest.hexdigest(),
    )
    return CheckpointInfo(
        candidate=candidate,
        full_training_state=full_keys <= state.keys(),
        version=version,
        resnet=deepcopy(resnet),
    )


def _grp_step(path: Path) -> int:
    import torch

    if not path.exists():
        return 0
    state = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or 'model' not in state:
        raise ValueError(f'not a GRP checkpoint: {path}')
    return int(state.get('steps', 0))


def _result_dict(result: EvalResult) -> dict[str, Any]:
    return {
        'model_id': result.model_id,
        'path': str(result.path),
        'rankings': list(result.rankings),
        'step': result.step,
        'fingerprint': result.fingerprint,
        'games': result.games,
        'avg_rank': result.avg_rank,
        'avg_pt': result.avg_pt,
        'rank_ci_low': result.rank_ci_low,
        'rank_ci_high': result.rank_ci_high,
    }


def _direct_comparison_dict(comparison: DirectComparison) -> dict[str, Any]:
    return {
        'model_a': {
            'model_id': comparison.model_a.model_id,
            'fingerprint': comparison.model_a.fingerprint,
        },
        'model_b': {
            'model_id': comparison.model_b.model_id,
            'fingerprint': comparison.model_b.fingerprint,
        },
        'a_vs_b': _result_dict(comparison.a_vs_b),
        'b_vs_a': _result_dict(comparison.b_vs_a),
        'combined_a': _result_dict(comparison.combined_a),
        'combined_b': _result_dict(
            comparison.result_for(comparison.model_b)
        ),
    }


def promote_training_parent(manifest: dict[str, Any], result: EvalResult, retained_path: Path) -> None:
    parent = {
        **_result_dict(result),
        'path': str(Path(retained_path)),
        'promoted_at': datetime.now().isoformat(timespec='seconds'),
    }
    manifest['training_parent'] = parent
    manifest['training_seed_model'] = parent['path']


class AutomationPipeline:
    STAGES = (
        ('parent', 'grp', 'offline', 'online', 'evaluate')
        if NUM_PLAYERS == 4
        else ('grp', 'offline', 'online', 'evaluate')
    )

    def __init__(
        self,
        config_file: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        stages: Sequence[str] | None = None,
    ):
        self.config_file = Path(config_file).resolve()
        self.config = toml.load(self.config_file)
        self.config_dir = self.config_file.parent
        self.dry_run = dry_run
        self.force = force
        self.stages = tuple(stages or self.STAGES)

        paths = self.config['paths']
        self.logs_root = _resolve_path(paths['logs_root'], self.config_dir)
        self.mortal_root = _resolve_path(paths['mortal_root'], self.config_dir)
        self.output_root = _resolve_path(paths['output_root'], self.config_dir)
        self.base_config_file = _resolve_path(paths['base_config'], self.config_dir)
        self.seed_model = _resolve_path(paths['seed_model'], self.config_dir)
        self.reference_model = _resolve_path(paths['reference_model'], self.config_dir)
        self.grp_seed_model = _resolve_path(paths['grp_seed_model'], self.config_dir)
        parent_candidate = self.config.get('parent_evaluation', {}).get('candidate_model')
        self.parent_candidate_model = (
            _resolve_path(parent_candidate, self.config_dir) if parent_candidate else None
        )
        self.manifest_file = self.output_root / 'manifest.json'
        self.evaluation_ledger_file = self.output_root / 'evaluation_ledger.json'
        self.models_dir = self.output_root / 'models'
        self.shared_grp_state = self.output_root / 'state' / 'grp.pth'

        self._validate_config()
        self.manifest = _load_json(self.manifest_file, {
            'version': 2,
            'known_logs': {},
            'cycles': {},
            'champion': None,
            'deployment_champion': None,
            'training_parent': None,
            'training_seed_model': None,
            'model_roster': [],
        })
        self.manifest['version'] = max(int(self.manifest.get('version', 1)), 2)
        self.manifest.setdefault('deployment_champion', self.manifest.get('champion'))
        self.manifest.setdefault('model_roster', [])
        if NUM_PLAYERS == 4 and not self.manifest.get('training_parent'):
            legacy_parent = self.manifest.get('training_seed_model')
            self.manifest['training_parent'] = {
                'path': str(Path(legacy_parent).resolve()) if legacy_parent else str(self.seed_model),
            }
        self.evaluation_ledger = _load_json(
            self.evaluation_ledger_file,
            {'version': 1, 'pairs': {}},
        )

    def _validate_config(self) -> None:
        for path, label in (
            (self.logs_root, 'logs_root'),
            (self.mortal_root, 'mortal_root'),
            (self.base_config_file, 'base_config'),
            (self.seed_model, 'seed_model'),
            (self.reference_model, 'reference_model'),
            (self.grp_seed_model, 'grp_seed_model'),
        ):
            if not path.exists():
                raise FileNotFoundError(f'{label} does not exist: {path}')
        unknown = set(self.stages) - set(self.STAGES) - {'discover'}
        if unknown:
            raise ValueError(f'unknown stages: {sorted(unknown)}')
        if not self.stages:
            raise ValueError('at least one stage is required')
        if NUM_PLAYERS == 4:
            self._validate_config_yonma()
        else:
            self._validate_config_sanma()

    def _validate_config_yonma(self) -> None:
        parent_settings = self.config.get('parent_evaluation', {})
        if parent_settings.get('enabled', False):
            if self.parent_candidate_model is None or not self.parent_candidate_model.exists():
                raise FileNotFoundError(
                    f'parent_evaluation.candidate_model does not exist: {self.parent_candidate_model}'
                )
        for section, key in (
            ('evaluation', 'screen_games'),
            ('evaluation', 'validation_games'),
            ('evaluation', 'validation_block_games'),
            ('evaluation', 'validation_max_games'),
            ('online', 'games_per_session'),
            ('parent_evaluation', 'block_games'),
            ('parent_evaluation', 'max_games'),
        ):
            value = int(self.config[section][key])
            if value <= 0 or value % NUM_PLAYERS != 0:
                raise ValueError(f'{section}.{key} must be a positive multiple of {NUM_PLAYERS}')
        if int(self.config['evaluation']['validation_max_games']) < int(
            self.config['evaluation']['validation_block_games']
        ):
            raise ValueError('evaluation.validation_max_games must cover at least one block')
        if int(parent_settings.get('max_games', 0)) < int(parent_settings.get('block_games', 0)):
            raise ValueError('parent_evaluation.max_games must cover at least one block')
        if int(self.config['offline']['snapshot_every']) % int(self.config['offline']['save_every']):
            raise ValueError('offline.snapshot_every must be a multiple of offline.save_every')
        if int(self.config['online']['brain_freeze_updates']) < int(self.config['online']['updates']):
            raise ValueError('online.brain_freeze_updates must freeze the complete online phase')
        self._validate_grp_strategy()

    def _validate_config_sanma(self) -> None:
        for section, key in (
            ('evaluation', 'screen_games'),
            ('evaluation', 'validation_games'),
            ('online', 'games_per_session'),
        ):
            value = int(self.config[section][key])
            if value <= 0 or value % NUM_PLAYERS != 0:
                raise ValueError(f'{section}.{key} must be a positive multiple of {NUM_PLAYERS}')
        expert_ratio = float(self.config['online']['expert_ratio'])
        if not 0 <= expert_ratio <= 1:
            raise ValueError('online.expert_ratio must be between 0 and 1')
        opponent_weights = self.config['online'].get('opponents', {})
        weights = {
            'anchor': float(opponent_weights.get('anchor_weight', 0.1)),
            'deployed': float(opponent_weights.get('deployed_weight', 0.4)),
            'league': float(opponent_weights.get('league_weight', 0.5)),
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError('online opponent weights must be non-negative')
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError('online opponent weights must sum to 1.0')
        challenger_limit = int(
            self.config['evaluation'].get(
                'max_challengers',
                self.config['evaluation'].get('finalists', 3),
            )
        )
        if challenger_limit < 0:
            raise ValueError('evaluation.max_challengers must be non-negative')
        direct_game_schedule(self.config['evaluation'])
        grp_settings = self.config['grp']
        holdout_ratio = float(grp_settings['validation_ratio'])
        gate_ratio = float(grp_settings.get('gate_ratio', holdout_ratio / 2))
        if not 0 < gate_ratio < holdout_ratio < 1:
            raise ValueError(
                'GRP ratios must satisfy 0 < gate_ratio < validation_ratio < 1'
            )
        self._validate_grp_strategy()

    def _validate_grp_strategy(self) -> None:
        settings = self.config['grp']
        if grp_gate_strategy(settings) == 'quality_gate':
            _validate_grp_quality_gate(settings)
        else:
            validate_grp_gate_settings(settings)

    def _save_manifest(self) -> None:
        if not self.dry_run:
            _atomic_json(self.manifest_file, self.manifest)

    def _save_evaluation_ledger(self) -> None:
        if not self.dry_run:
            _atomic_json(self.evaluation_ledger_file, self.evaluation_ledger)

    def _training_seed(self) -> Path:
        parent = self.manifest.get('training_parent') or {}
        parent_path = parent.get('path')
        if parent_path and Path(parent_path).exists():
            return Path(parent_path).resolve()
        manifest_seed = self.manifest.get('training_seed_model')
        if manifest_seed and Path(manifest_seed).exists():
            return Path(manifest_seed).resolve()
        return self.seed_model

    def _league_model_paths(self) -> list[Path]:
        payload = _load_json(self.models_dir / 'roster.json', {})
        entries = payload.get('models', self.manifest.get('model_roster', []))
        ordered = sorted(entries, key=lambda item: int(item.get('slot', 999)))
        paths = []
        for entry in ordered:
            path = Path(str(entry.get('path', '')))
            if path.is_file() and path.suffix == '.pth':
                paths.append(path.resolve())
        return paths[:3]

    def _unfinished_frozen_cycle(self) -> tuple[str, dict] | None:
        unfinished = [
            (cycle_id, cycle)
            for cycle_id, cycle in self.manifest['cycles'].items()
            if cycle.get('status') != 'complete' and cycle.get('dataset_records')
        ]
        if not unfinished:
            return None
        return min(
            unfinished,
            key=lambda item: (item[1].get('created_at', ''), item[0]),
        )

    def _frozen_records(self, cycle: Mapping) -> list[LogRecord]:
        records = []
        root = self.logs_root.resolve()
        for payload in cycle['dataset_records']:
            relative_path = str(payload['relative_path'])
            file_path = (root / Path(relative_path)).resolve()
            if not file_path.is_relative_to(root):
                raise RuntimeError(f'frozen log path escapes logs_root: {relative_path}')
            if not file_path.is_file():
                raise RuntimeError(f'frozen log is missing: {file_path}')
            stat = file_path.stat()
            record = LogRecord(
                path=file_path,
                relative_path=relative_path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
            expected = f"{int(payload['size'])}:{int(payload['mtime_ns'])}"
            if record.token != expected:
                raise RuntimeError(
                    f'frozen log changed during an unfinished cycle: {file_path}'
                )
            records.append(record)
        records.sort(key=lambda record: record.relative_path)
        signature = dataset_signature(records)
        if signature != cycle.get('dataset_signature'):
            raise RuntimeError('frozen cycle dataset signature does not match its records')
        return records

    def _cycle_for_signature(self, signature: str, new_count: int) -> tuple[str, dict] | None:
        for cycle_id, cycle in self.manifest['cycles'].items():
            if cycle.get('dataset_signature') == signature and cycle.get('status') != 'complete':
                return cycle_id, cycle

        completed = [
            (cycle_id, cycle)
            for cycle_id, cycle in self.manifest['cycles'].items()
            if cycle.get('dataset_signature') == signature and cycle.get('status') == 'complete'
        ]
        if completed and new_count == 0 and not self.force:
            # yonma stops here; sanma still allows an evaluate-only cycle to
            # bootstrap a missing Top-3 roster from retained models
            if NUM_PLAYERS == 4 or self._roster_fingerprints():
                return None

        prefix = datetime.now().strftime('%Y%m%d')
        base_id = f'{prefix}_{signature[:10]}'
        cycle_id = base_id
        counter = 2
        while cycle_id in self.manifest['cycles']:
            cycle_id = f'{base_id}_{counter}'
            counter += 1
        cycle = {
            'dataset_signature': signature,
            'status': 'planned',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'stages': {},
        }
        return cycle_id, cycle

    def plan(self) -> tuple[dict[str, Any], list[LogRecord], list[LogRecord], str, str | None]:
        if NUM_PLAYERS == 4:
            return self._plan_yonma()
        return self._plan_sanma()

    def _plan_yonma(self) -> tuple[dict[str, Any], list[LogRecord], list[LogRecord], str, str | None]:
        records = discover_logs(self.logs_root)
        minimum_logs = int(self.config['pipeline'].get('minimum_logs', 2))
        if len(records) < minimum_logs:
            raise RuntimeError(f'only {len(records)} logs found; need at least {minimum_logs}')
        changed = new_logs(records, self.manifest.get('known_logs', {}))
        signature = dataset_signature(records)
        cycle_pair = self._cycle_for_signature(signature, len(changed))
        cycle_id = cycle_pair[0] if cycle_pair else None
        grp_train, grp_validation = split_logs(
            records,
            float(self.config['grp']['validation_ratio']),
            str(self.config['pipeline']['split_seed']),
        )

        grp_source = self.shared_grp_state if self.shared_grp_state.exists() else self.grp_seed_model
        grp_current = _grp_step(grp_source)
        grp_cfg = self.config['grp']
        grp_needed = bool(grp_cfg.get('enabled', True)) and should_update_grp(
            len(changed),
            len(records),
            grp_source.exists(),
            int(grp_cfg['min_new_logs']),
            float(grp_cfg['min_new_ratio']),
        )
        basis = len(changed) if changed else (len(records) if self.force else 0)
        grp_updates = 0
        if grp_needed:
            grp_updates = math.ceil(basis * float(grp_cfg['updates_per_new_log']))
            grp_updates = max(int(grp_cfg['min_updates']), grp_updates)
            grp_updates = min(int(grp_cfg['max_updates']), grp_updates)
        grp_target = aligned_target_step(
            grp_current,
            grp_updates,
            int(grp_cfg['save_every']),
        )

        seed_info = inspect_checkpoint(self._training_seed(), 'training_seed')
        if not seed_info.full_training_state:
            raise ValueError(
                f'training seed must contain optimizer and auxiliary state: {seed_info.candidate.path}'
            )
        offline_target = aligned_target_step(
            int(seed_info.candidate.step or 0),
            int(self.config['offline']['updates']),
            int(self.config['offline']['save_every']),
        )
        online_target = aligned_target_step(
            offline_target,
            int(self.config['online']['updates']),
            int(self.config['online']['save_every']),
        )
        plan = {
            'parent_evaluation': {
                'enabled': bool(self.config.get('parent_evaluation', {}).get('enabled', False)),
                'candidate_model': str(self.parent_candidate_model) if self.parent_candidate_model else None,
                'current_parent': str(self._training_seed()),
                'block_games': int(self.config.get('parent_evaluation', {}).get('block_games', 0)),
                'max_games': int(self.config.get('parent_evaluation', {}).get('max_games', 0)),
            },
            'dataset': {
                'root': str(self.logs_root),
                'total_logs': len(records),
                'new_or_changed_logs': len(changed),
                'signature': signature,
            },
            'cycle_id': cycle_id,
            'stages': list(self.stages),
            'grp': {
                'update': grp_needed,
                'current_step': grp_current,
                'updates': grp_updates,
                'target_step': grp_target,
                'train_logs': len(grp_train),
                'validation_logs': len(grp_validation),
                'downstream_gate': {
                    'enabled': bool(grp_cfg.get('downstream', {}).get('enabled', True)),
                    'training_seeds': list(grp_cfg.get('downstream', {}).get('training_seeds', [])),
                    'games_per_seed': int(grp_cfg.get('downstream', {}).get('games', 0)),
                },
            },
            'offline': {
                'seed_model': str(seed_info.candidate.path),
                'start_step': seed_info.candidate.step,
                'target_step': offline_target,
                'estimated_steps_per_epoch': estimate_steps_per_epoch(
                    len(records),
                    batch_size=int(self._base_config(seed_info)['control']['batch_size']),
                    samples_per_log=int(self.config['pipeline'].get('estimated_samples_per_log', 660)),
                ),
                'snapshot_every': int(self.config['offline']['snapshot_every']),
            },
            'online': {
                'start_step': offline_target,
                'target_step': online_target,
                'self_play_games_per_session': int(self.config['online']['games_per_session']),
                'expert_ratio': float(self.config['online']['expert_ratio']),
            },
            'evaluation': {
                'screen_games': int(self.config['evaluation']['screen_games']),
                'finalists': int(self.config['evaluation']['finalists']),
                'validation_games': int(self.config['evaluation']['validation_games']),
                'retained_models_dir': str(self.models_dir),
            },
        }
        return plan, records, changed, signature, cycle_id

    def _plan_sanma(self) -> tuple[dict[str, Any], list[LogRecord], list[LogRecord], str, str | None]:
        current_records = discover_logs(self.logs_root)
        frozen_cycle = self._unfinished_frozen_cycle()
        if frozen_cycle is not None:
            cycle_id, cycle = frozen_cycle
            records = self._frozen_records(cycle)
            pending = new_logs(current_records, records_to_manifest(records))
            plan = deepcopy(cycle['plan'])
            plan['cycle_id'] = cycle_id
            plan['stages'] = list(self.stages)
            plan['dataset']['pending_new_logs'] = len(pending)
            return (
                plan,
                records,
                [],
                str(cycle['dataset_signature']),
                cycle_id,
            )

        records = current_records
        minimum_logs = int(self.config['pipeline'].get('minimum_logs', 2))
        if len(records) < minimum_logs:
            raise RuntimeError(f'only {len(records)} logs found; need at least {minimum_logs}')
        changed = new_logs(records, self.manifest.get('known_logs', {}))
        signature = dataset_signature(records)
        cycle_pair = self._cycle_for_signature(signature, len(changed))
        completed_same_dataset = any(
            cycle.get('dataset_signature') == signature
            and cycle.get('status') == 'complete'
            for cycle in self.manifest['cycles'].values()
        )
        if (
            cycle_pair is not None
            and completed_same_dataset
            and not changed
            and not self.force
            and not self._roster_fingerprints()
            and self.stages == self.STAGES
        ):
            self.stages = ('evaluate',)
        cycle_id = cycle_pair[0] if cycle_pair else None
        grp_train, grp_validation, grp_gate = split_grp_logs(
            records,
            holdout_ratio=float(self.config['grp']['validation_ratio']),
            gate_ratio=float(
                self.config['grp'].get(
                    'gate_ratio',
                    float(self.config['grp']['validation_ratio']) / 2,
                )
            ),
            seed=str(self.config['pipeline']['split_seed']),
        )

        grp_source = (
            self.shared_grp_state if self.shared_grp_state.exists() else self.grp_seed_model
        )
        grp_current = _grp_step(grp_source)
        grp_cfg = self.config['grp']
        grp_needed = bool(grp_cfg.get('enabled', True)) and should_update_grp(
            len(changed),
            len(records),
            grp_source.exists(),
            int(grp_cfg['min_new_logs']),
            float(grp_cfg['min_new_ratio']),
        )
        basis = len(changed) if changed else (len(records) if self.force else 0)
        grp_updates = 0
        if grp_needed:
            grp_updates = math.ceil(basis * float(grp_cfg['updates_per_new_log']))
            grp_updates = max(int(grp_cfg['min_updates']), grp_updates)
            grp_updates = min(int(grp_cfg['max_updates']), grp_updates)
        grp_target = aligned_target_step(
            grp_current,
            grp_updates,
            int(grp_cfg['save_every']),
        )

        seed_info = inspect_checkpoint(self._training_seed(), 'training_seed')
        if not seed_info.full_training_state:
            raise ValueError(
                'training seed must contain optimizer and auxiliary state: '
                f'{seed_info.candidate.path}'
            )
        if seed_info.version != 4:
            raise ValueError(
                f'training seed must use MahjongCopilot version 4, got {seed_info.version}'
            )
        reference_info = inspect_checkpoint(self.reference_model, 'baseline')
        if reference_info.version != 4:
            raise ValueError(
                f'reference model must use MahjongCopilot version 4, got {reference_info.version}'
            )

        offline_updates = (
            int(self.config['offline']['updates'])
            if self.config['offline'].get('enabled', True)
            else 0
        )
        offline_target = aligned_target_step(
            int(seed_info.candidate.step or 0),
            offline_updates,
            int(self.config['offline']['save_every']),
        )
        online_updates = (
            int(self.config['online']['updates'])
            if self.config['online'].get('enabled', True)
            else 0
        )
        online_target = aligned_target_step(
            offline_target,
            online_updates,
            int(self.config['online']['save_every']),
        )
        direct_initial_games, direct_max_games = direct_game_schedule(
            self.config['evaluation']
        )
        plan = {
            'dataset': {
                'root': str(self.logs_root),
                'total_logs': len(records),
                'new_or_changed_logs': len(changed),
                'signature': signature,
            },
            'cycle_id': cycle_id,
            'stages': list(self.stages),
            'grp': {
                'update': grp_needed,
                'current_step': grp_current,
                'updates': grp_updates,
                'target_step': grp_target,
                'train_logs': len(grp_train),
                'validation_logs': len(grp_validation),
                'gate_logs': len(grp_gate),
            },
            'offline': {
                'seed_model': str(seed_info.candidate.path),
                'start_step': seed_info.candidate.step,
                'target_step': offline_target,
                'snapshot_every': int(
                    self.config['offline'].get('snapshot_every', 0)
                ),
            },
            'online': {
                'start_step': offline_target,
                'target_step': online_target,
                'self_play_games_per_session': int(
                    self.config['online']['games_per_session']
                ),
                'expert_ratio': float(self.config['online']['expert_ratio']),
                'snapshot_every': int(
                    self.config['online']['snapshot_every']
                ),
                'opponents': {
                    'anchor_weight': float(
                        self.config['online'].get('opponents', {}).get(
                            'anchor_weight', 0.1
                        )
                    ),
                    'deployed_weight': float(
                        self.config['online'].get('opponents', {}).get(
                            'deployed_weight', 0.4
                        )
                    ),
                    'league_weight': float(
                        self.config['online'].get('opponents', {}).get(
                            'league_weight', 0.5
                        )
                    ),
                },
            },
            'evaluation': {
                'screen_games': int(self.config['evaluation']['screen_games']),
                'max_challengers': int(
                    self.config['evaluation'].get(
                        'max_challengers',
                        self.config['evaluation'].get('finalists', 3),
                    )
                ),
                'direct_initial_games_per_direction': direct_initial_games,
                'direct_initial_games': direct_initial_games * 2,
                'direct_max_games_per_direction': direct_max_games,
                'direct_max_games': direct_max_games * 2,
                'retained_models': 3,
                'retained_models_dir': str(self.models_dir),
            },
        }
        return plan, records, changed, signature, cycle_id

    def run(self) -> int:
        if self.dry_run or self.stages == ('discover',):
            plan, _, _, _, _ = self.plan()
            print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
            return 0
        native_extension = sync_native_extension(self.mortal_root)
        print(f'Native extension: {native_extension}', flush=True)
        if NUM_PLAYERS == 4:
            return self._run_yonma()
        return self._run_sanma()

    def _run_yonma(self) -> int:
        if 'parent' in self.stages:
            print('[parent] starting', flush=True)
            parent_result = self._run_parent_evaluation()
            print(json.dumps({'parent': parent_result}, indent=2, ensure_ascii=False), flush=True)
        if self.stages == ('parent',):
            return 0

        plan, records, changed, signature, cycle_id = self.plan()
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        if cycle_id is None:
            print('No new logs and the matching cycle is already complete.', flush=True)
            return 0

        cycle = self.manifest['cycles'].get(cycle_id)
        if cycle is None:
            _, cycle = self._cycle_for_signature(signature, len(changed))
            self.manifest['cycles'][cycle_id] = cycle
        run_dir = self.output_root / 'runs' / cycle_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cycle['run_dir'] = str(run_dir)
        cycle['status'] = 'running'
        cycle['total_logs'] = len(records)
        cycle['new_or_changed_logs'] = len(changed)
        active_plan = frozen_cycle_plan(cycle, plan)
        self.manifest['known_logs'] = records_to_manifest(records)
        self._save_manifest()

        train_records, val_records = split_logs(
            records,
            float(self.config['grp']['validation_ratio']),
            str(self.config['pipeline']['split_seed']),
        )
        dataset_index, grp_index = self._prepare_indices(run_dir, records, train_records, val_records)

        try:
            grp_state = self._ensure_grp_state()
            if 'grp' in self.stages:
                self._run_stage(cycle, 'grp', lambda: self._run_grp(
                    run_dir,
                    grp_state,
                    grp_index,
                    train_records,
                    val_records,
                    dataset_index=dataset_index,
                    records=records,
                    planned_update=bool(active_plan['grp']['update']),
                    target_step=int(active_plan['grp']['target_step']),
                ))
            offline_state = run_dir / 'offline' / 'model.pth'
            if 'offline' in self.stages:
                self._run_stage(cycle, 'offline', lambda: self._run_offline(
                    run_dir, dataset_index, grp_state, records, offline_state
                ))
            online_state = run_dir / 'online' / 'model.pth'
            if 'online' in self.stages:
                self._run_stage(cycle, 'online', lambda: self._run_online(
                    run_dir, dataset_index, grp_state, records, offline_state, online_state
                ))
            if 'evaluate' in self.stages:
                self._run_stage(cycle, 'evaluate', lambda: self._run_evaluation(
                    run_dir, offline_state, online_state
                ))
        except Exception:
            cycle['status'] = 'failed'
            self._save_manifest()
            raise

        section_by_stage = {'evaluate': 'evaluation'}
        enabled_stages = [
            stage for stage in self.STAGES if stage != 'parent'
            if bool(self.config.get(section_by_stage.get(stage, stage), {}).get('enabled', True))
        ]
        if all(cycle['stages'].get(stage, {}).get('status') in ('complete', 'skipped') for stage in enabled_stages):
            cycle['status'] = 'complete'
            cycle['completed_at'] = datetime.now().isoformat(timespec='seconds')
        else:
            cycle['status'] = 'partial'
        self._save_manifest()
        return 0

    def _run_sanma(self) -> int:
        plan, records, changed, signature, cycle_id = self.plan()
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        if cycle_id is None:
            print('No new logs and the matching cycle is already complete.', flush=True)
            return 0

        cycle = self.manifest['cycles'].get(cycle_id)
        if cycle is None:
            _, cycle = self._cycle_for_signature(signature, len(changed))
            self.manifest['cycles'][cycle_id] = cycle
        run_dir = self.output_root / 'runs' / cycle_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cycle['run_dir'] = str(run_dir)
        cycle['status'] = 'running'
        cycle['total_logs'] = len(records)
        cycle['new_or_changed_logs'] = len(changed)
        cycle.setdefault(
            'dataset_records',
            [
                {
                    'relative_path': record.relative_path,
                    'size': record.size,
                    'mtime_ns': record.mtime_ns,
                }
                for record in records
            ],
        )
        active_plan = frozen_cycle_plan(cycle, plan)
        self.manifest['known_logs'] = records_to_manifest(records)
        self._save_manifest()

        train_records, val_records, gate_records = split_grp_logs(
            records,
            holdout_ratio=float(self.config['grp']['validation_ratio']),
            gate_ratio=float(
                self.config['grp'].get(
                    'gate_ratio',
                    float(self.config['grp']['validation_ratio']) / 2,
                )
            ),
            seed=str(self.config['pipeline']['split_seed']),
        )
        dataset_index, grp_index = self._prepare_indices(
            run_dir,
            records,
            train_records,
            val_records,
            gate_records,
        )

        try:
            grp_state = self._ensure_grp_state()
            if 'grp' in self.stages:
                self._run_stage(
                    cycle,
                    'grp',
                    lambda: self._run_grp(
                        run_dir,
                        grp_state,
                        grp_index,
                        train_records,
                        val_records,
                        gate_records,
                        int(active_plan['grp']['target_step']),
                    ),
                )
            offline_state = run_dir / 'offline' / 'model.pth'
            if 'offline' in self.stages:
                self._run_stage(
                    cycle,
                    'offline',
                    lambda: self._run_offline(
                        run_dir,
                        dataset_index,
                        grp_state,
                        records,
                        offline_state,
                        int(active_plan['offline']['target_step']),
                    ),
                )
            online_state = run_dir / 'online' / 'model.pth'
            if 'online' in self.stages:
                self._run_stage(
                    cycle,
                    'online',
                    lambda: self._run_online(
                        run_dir,
                        dataset_index,
                        grp_state,
                        records,
                        offline_state,
                        online_state,
                        int(active_plan['online']['start_step']),
                        int(active_plan['online']['target_step']),
                    ),
                )
            if 'evaluate' in self.stages:
                self._run_stage(
                    cycle,
                    'evaluate',
                    lambda: self._run_evaluation(run_dir, offline_state, online_state),
                )
        except Exception:
            cycle['status'] = 'failed'
            self._save_manifest()
            raise

        if requested_stages_complete(cycle, self.stages, self.config):
            cycle['status'] = 'complete'
            cycle['completed_at'] = datetime.now().isoformat(timespec='seconds')
        else:
            cycle['status'] = 'partial'
        self._save_manifest()
        return 0

    def _prepare_indices(
        self,
        run_dir: Path,
        records: Sequence[LogRecord],
        train_records: Sequence[LogRecord],
        val_records: Sequence[LogRecord],
        gate_records: Sequence[LogRecord] | None = None,
    ) -> tuple[Path, Path]:
        import torch

        indices_dir = run_dir / 'indices'
        indices_dir.mkdir(parents=True, exist_ok=True)
        dataset_index = indices_dir / 'mortal_files.pth'
        grp_index = indices_dir / 'grp_files.pth'
        torch.save({'file_list': [str(record.path) for record in records]}, dataset_index)
        grp_payload = {
            'train_file_list': [str(record.path) for record in train_records],
            'val_file_list': [str(record.path) for record in val_records],
        }
        if gate_records is not None:
            grp_payload['gate_file_list'] = [str(record.path) for record in gate_records]
        torch.save(grp_payload, grp_index)
        metadata = {
            'total': len(records),
            'train': len(train_records),
            'validation': len(val_records),
            'files': records_to_manifest(records),
        }
        if gate_records is not None:
            metadata['gate'] = len(gate_records)
        _atomic_json(indices_dir / 'dataset.json', metadata)
        return dataset_index, grp_index

    def _ensure_grp_state(self) -> Path:
        if not self.shared_grp_state.exists():
            self.shared_grp_state.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.grp_seed_model, self.shared_grp_state)
        return self.shared_grp_state

    def _run_stage(self, cycle: dict, name: str, action: Callable[[], dict]) -> None:
        stage = cycle['stages'].setdefault(name, {})
        if stage.get('status') in ('complete', 'skipped'):
            print(f'[{name}] already {stage["status"]}; reusing it.', flush=True)
            return
        print(f'[{name}] starting', flush=True)
        stage['status'] = 'running'
        stage['started_at'] = datetime.now().isoformat(timespec='seconds')
        stage.pop('error', None)
        self._save_manifest()
        try:
            details = action()
        except Exception as exc:
            stage['status'] = 'failed'
            stage['error'] = f'{type(exc).__name__}: {exc}'
            self._save_manifest()
            raise
        stage.update(details)
        stage['status'] = details.get('status', 'complete')
        stage['completed_at'] = datetime.now().isoformat(timespec='seconds')
        self._save_manifest()
        print(f'[{name}] {stage["status"]}', flush=True)

    def _base_config(self, architecture: CheckpointInfo) -> dict:
        cfg = toml.load(self.base_config_file)
        cfg['control']['version'] = architecture.version
        cfg['resnet'] = deepcopy(architecture.resnet)
        return cfg

    def _run_parent_evaluation(self) -> dict[str, Any]:
        settings = self.config.get('parent_evaluation', {})
        if not settings.get('enabled', False):
            return {'status': 'skipped', 'reason': 'disabled'}
        if self.parent_candidate_model is None:
            return {'status': 'skipped', 'reason': 'no candidate configured'}

        candidate_info = inspect_checkpoint(self.parent_candidate_model, 'parent-candidate')
        reference_info = inspect_checkpoint(self._training_seed(), 'training-parent')
        candidate = candidate_info.candidate
        reference = reference_info.candidate
        if not candidate_info.full_training_state:
            raise ValueError(f'parent candidate is not resumable: {candidate.path}')
        if candidate.fingerprint == reference.fingerprint:
            return {
                'status': 'skipped',
                'reason': 'candidate is already the training parent',
                'training_parent': str(reference.path),
            }
        seed_fingerprint = inspect_checkpoint(self.seed_model, 'initial-seed').candidate.fingerprint
        if settings.get('bootstrap_only', True) and reference.fingerprint != seed_fingerprint:
            return {
                'status': 'skipped',
                'reason': 'the initial training parent has already been replaced',
                'training_parent': str(reference.path),
            }

        result, decision, ledger_key = self._sequential_evaluate(
            candidate,
            reference,
            purpose='training-parent',
            block_games=int(settings['block_games']),
            max_games=int(settings['max_games']),
            require_positive_pt=bool(settings.get('require_positive_pt', True)),
            timeout_hours=float(settings['timeout_hours']),
        )
        retained_path = None
        if decision == 'promote':
            retained_path = self._retain_validated_models([result])[result.model_id]
            promote_training_parent(self.manifest, result, retained_path)
            self._save_manifest()

        report = {
            'candidate': _result_dict(result),
            'reference': {
                'model_id': reference.model_id,
                'path': str(reference.path),
                'step': reference.step,
                'fingerprint': reference.fingerprint,
            },
            'decision': decision,
            'ledger_key': ledger_key,
            'retained_parent': str(retained_path) if retained_path else None,
        }
        report_file = self.output_root / 'parent_evaluation' / f'{ledger_key[:16]}.json'
        _atomic_json(report_file, report)
        return {**report, 'report': str(report_file)}

    def _sequential_evaluate(
        self,
        candidate: Candidate,
        reference: Candidate,
        *,
        purpose: str,
        block_games: int,
        max_games: int,
        require_positive_pt: bool,
        timeout_hours: float,
    ) -> tuple[EvalResult, str, str]:
        key = evaluation_ledger_key(purpose, candidate.fingerprint, reference.fingerprint)
        entry = self.evaluation_ledger['pairs'].setdefault(key, {
            'purpose': purpose,
            'challenger': {
                'model_id': candidate.model_id,
                'path': str(candidate.path),
                'step': candidate.step,
                'fingerprint': candidate.fingerprint,
            },
            'reference': {
                'model_id': reference.model_id,
                'path': str(reference.path),
                'step': reference.step,
                'fingerprint': reference.fingerprint,
            },
            'blocks': [],
        })
        root = self.output_root / 'evaluations' / _safe_id(purpose) / key[:16]

        while True:
            blocks = entry['blocks']
            if blocks:
                aggregate = evaluation_result_from_blocks(candidate, blocks)
                decision = sequential_evaluation_decision(
                    aggregate,
                    min_games=block_games,
                    max_games=max_games,
                    require_positive_pt=require_positive_pt,
                )
                entry['aggregate'] = _result_dict(aggregate)
                entry['decision'] = decision
                self._save_evaluation_ledger()
                if decision != 'continue':
                    return aggregate, decision, key

            completed_games = sum(int(block['games']) for block in blocks)
            remaining_games = max_games - completed_games
            if remaining_games <= 0:
                aggregate = evaluation_result_from_blocks(candidate, blocks)
                return aggregate, 'inconclusive', key
            games = min(block_games, remaining_games)
            block_index = len(blocks)
            seed = _seed_from_text(f'{purpose}:{reference.fingerprint}:block:{block_index}')
            block_root = root / f'block_{block_index:03d}'
            result = self._evaluate_candidate(
                candidate,
                reference,
                games,
                seed,
                block_root,
                timeout_hours,
            )
            record_evaluation_block(
                entry,
                block_index=block_index,
                seed=seed,
                result=result,
            )
            entry['blocks'][-1]['log_dir'] = str(block_root / _safe_id(candidate.model_id))
            self._save_evaluation_ledger()

    def _wait_for_gpu_slot(self) -> None:
        settings = self.config.get('resources', {})
        if not settings.get('wait_for_gpu', False):
            return
        gpu_index = int(settings.get('gpu_index', 0))
        max_memory = int(settings['max_memory_used_mb'])
        max_utilization = int(settings['max_utilization'])
        poll_seconds = max(5, int(settings.get('poll_seconds', 30)))
        deadline = time.monotonic() + float(settings.get('timeout_hours', 72)) * 3600
        last_notice = 0.0

        while True:
            completed = subprocess.run(
                [
                    'nvidia-smi',
                    f'--id={gpu_index}',
                    '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f'nvidia-smi failed: {completed.stderr.strip()}')
            try:
                memory_text, utilization_text = completed.stdout.strip().split(',', maxsplit=1)
                memory_used = int(memory_text.strip())
                utilization = int(utilization_text.strip())
            except ValueError as exc:
                raise RuntimeError(f'unexpected nvidia-smi output: {completed.stdout!r}') from exc
            if gpu_slot_available(
                memory_used,
                utilization,
                max_memory_used_mb=max_memory,
                max_utilization=max_utilization,
            ):
                print(
                    f'[resources] GPU {gpu_index} available: {memory_used} MiB, {utilization}%',
                    flush=True,
                )
                return
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f'GPU {gpu_index} remained busy: {memory_used} MiB, {utilization}%'
                )
            if now - last_notice >= 300 or last_notice == 0:
                print(
                    f'[resources] waiting for GPU {gpu_index}: '
                    f'{memory_used} MiB, {utilization}%',
                    flush=True,
                )
                last_notice = now
            time.sleep(poll_seconds)

    def _run_grp(
        self,
        run_dir: Path,
        grp_state: Path,
        grp_index: Path,
        train_records: Sequence[LogRecord],
        val_records: Sequence[LogRecord],
        gate_records: Sequence[LogRecord] | None = None,
        target_step: int = 0,
        *,
        dataset_index: Path | None = None,
        records: Sequence[LogRecord] | None = None,
        planned_update: bool = True,
    ) -> dict:
        settings = self.config['grp']
        current = _grp_step(grp_state)
        if not settings.get('enabled', True):
            return {'status': 'skipped', 'step': current, 'reason': 'disabled'}
        if not should_execute_grp_update(enabled=True, planned_update=planned_update):
            return {'status': 'skipped', 'step': current, 'reason': 'threshold not reached'}
        if current >= target_step:
            return {
                'status': 'skipped',
                'step': current,
                'reason': 'target already reached',
            }
        if grp_gate_strategy(settings) == 'quality_gate':
            return self._run_grp_quality_gate(
                run_dir,
                grp_state,
                grp_index,
                train_records,
                val_records,
                list(gate_records or []),
                target_step,
            )
        return self._run_grp_validation_gate(
            run_dir,
            grp_state,
            grp_index,
            dataset_index,
            train_records,
            val_records,
            records,
            target_step,
        )

    def _run_grp_quality_gate(
        self,
        run_dir: Path,
        grp_state: Path,
        grp_index: Path,
        train_records: Sequence[LogRecord],
        val_records: Sequence[LogRecord],
        gate_records: Sequence[LogRecord],
        target_step: int,
    ) -> dict:
        settings = self.config['grp']
        current = _grp_step(grp_state)
        grp_dir = run_dir / 'grp'
        grp_dir.mkdir(parents=True, exist_ok=True)
        candidate_state = grp_dir / 'candidate.pth'
        best_state = grp_dir / 'best.pth'
        training_metrics_file = grp_dir / 'training_metrics.json'
        quality_report_file = grp_dir / 'quality_report.json'
        source_file = grp_dir / 'candidate_source.json'
        incumbent_sha256 = file_sha256(grp_state)
        if candidate_state.exists():
            source = _load_json(source_file, {})
            if source.get('incumbent_sha256') != incumbent_sha256:
                raise RuntimeError(
                    'GRP candidate was created from a different incumbent checkpoint'
                )
        else:
            _atomic_copy(grp_state, candidate_state)
            _atomic_json(
                source_file,
                {
                    'incumbent_path': str(grp_state),
                    'incumbent_step': current,
                    'incumbent_sha256': incumbent_sha256,
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                },
            )

        seed_info = inspect_checkpoint(self._training_seed(), 'training_seed')
        quality_settings = settings.get('quality_gate', {})
        cfg = build_grp_config(
            self._base_config(seed_info),
            state_file=candidate_state,
            best_state_file=best_state,
            metrics_file=training_metrics_file,
            index_file=grp_index,
            tensorboard_dir=grp_dir / 'tensorboard',
            max_steps=target_step,
            early_stopping_patience=int(
                settings.get('early_stopping_patience', 0)
            ),
            early_stopping_min_delta=float(
                settings.get('early_stopping_min_delta', 0.0)
            ),
            selection_metric=str(
                quality_settings.get(
                    'primary_metric',
                    'expected_pt_mse',
                )
            ),
        )
        cfg['grp']['control']['save_every'] = int(settings['save_every'])
        cfg['grp']['control']['val_steps'] = int(settings['val_steps'])
        cfg['grp']['dataset']['train_globs'] = [
            str(self.logs_root / '**' / '*.mjson')
        ]
        cfg['grp']['dataset']['val_globs'] = [
            str(self.logs_root / '**' / '*.mjson')
        ]
        cfg['grp']['dataset']['file_batch_size'] = int(
            settings['file_batch_size']
        )
        cfg['grp']['optim']['lr'] = float(settings['lr'])
        cfg['grp']['quality_gate'] = {
            'incumbent_state_file': _config_path(grp_state),
            'candidate_state_file': _config_path(best_state),
            'report_file': _config_path(quality_report_file),
            'confidence': float(quality_settings.get('confidence', 0.95)),
            'bootstrap_samples': int(
                quality_settings.get('bootstrap_samples', 2000)
            ),
            'bootstrap_seed': _seed_from_text(f'{run_dir.name}:grp-quality'),
        }
        config_file = grp_dir / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_logged(
            [sys.executable, str(self.mortal_root / 'mortal' / 'train_grp.py')],
            config_file,
            grp_dir / 'train.log',
            float(settings['timeout_hours']),
        )
        trained_final_step = _grp_step(candidate_state)
        training_metrics = _load_json(training_metrics_file, {})
        early_stopped = bool(training_metrics.get('early_stopped', False))
        if trained_final_step < target_step and not early_stopped:
            raise RuntimeError(
                f'GRP stopped at {trained_final_step}, expected at least {target_step}'
            )
        if not best_state.is_file():
            raise RuntimeError('GRP training did not produce a best checkpoint')

        if bool(quality_settings.get('enabled', True)):
            self._run_logged(
                [sys.executable, str(self.mortal_root / 'mortal' / 'grp_quality.py')],
                config_file,
                grp_dir / 'quality.log',
                float(settings['timeout_hours']),
            )
            quality_report = _load_json(quality_report_file, {})
            promotion = decide_grp_promotion(
                quality_report,
                primary_metric=str(
                    quality_settings.get('primary_metric', 'expected_pt_mse')
                ),
                require_significant=bool(
                    quality_settings.get('require_significant', True)
                ),
                min_relative_improvement=float(
                    quality_settings.get('min_relative_improvement', 0.0)
                ),
                max_nll_regression=float(
                    quality_settings.get('max_nll_regression', 0.0)
                ),
                max_brier_regression=float(
                    quality_settings.get('max_brier_regression', 0.0)
                ),
                min_gate_logs=int(quality_settings.get('min_gate_logs', 100)),
            )
        else:
            quality_report = {}
            promotion = {
                'accepted': True,
                'reasons': [],
                'primary_metric': None,
                'relative_improvement': None,
                'delta': None,
                'confidence_interval': None,
            }

        promoted_sha256 = None
        promoted_step = None
        if promotion['accepted']:
            _atomic_copy(best_state, grp_state)
            promoted_sha256 = file_sha256(grp_state)
            promoted_step = _grp_step(grp_state)
            if promoted_sha256 != file_sha256(best_state):
                raise RuntimeError('promoted GRP checkpoint does not match candidate')
        promotion = {
            **promotion,
            'incumbent_sha256': incumbent_sha256,
            'candidate_sha256': file_sha256(best_state),
            'promoted_sha256': promoted_sha256,
            'promoted_step': promoted_step,
        }
        _atomic_json(
            grp_dir / 'promotion.json',
            {
                'promotion': promotion,
                'training': training_metrics,
                'quality': quality_report,
            },
        )
        final_step = _grp_step(grp_state)
        return {
            'from_step': current,
            'target_step': target_step,
            'final_step': final_step,
            'trained_final_step': trained_final_step,
            'best_step': _grp_step(best_state),
            'early_stopped': early_stopped,
            'train_logs': len(train_records),
            'validation_logs': len(val_records),
            'gate_logs': len(gate_records),
            'config': str(config_file),
            'candidate': str(candidate_state),
            'best_candidate': str(best_state),
            'training_metrics': training_metrics,
            'quality_report': quality_report,
            'promotion': promotion,
        }

    def _run_grp_validation_gate(
        self,
        run_dir: Path,
        grp_state: Path,
        grp_index: Path,
        dataset_index: Path,
        train_records: Sequence[LogRecord],
        val_records: Sequence[LogRecord],
        records: Sequence[LogRecord],
        target_step: int,
    ) -> dict:
        settings = self.config['grp']
        current = _grp_step(grp_state)
        from_step = current
        target = target_step

        grp_dir = run_dir / 'grp'
        incumbent_state = grp_dir / 'incumbent.pth'
        candidate_state = grp_dir / 'candidate.pth'
        grp_dir.mkdir(parents=True, exist_ok=True)
        if not incumbent_state.exists():
            shutil.copy2(grp_state, incumbent_state)
        if not candidate_state.exists():
            shutil.copy2(incumbent_state, candidate_state)

        seed_info = inspect_checkpoint(self._training_seed(), 'training_seed')
        cfg = build_grp_config(
            self._base_config(seed_info),
            state_file=candidate_state,
            index_file=grp_index,
            tensorboard_dir=grp_dir / 'tensorboard',
            max_steps=target,
        )
        cfg['grp']['control']['save_every'] = int(settings['save_every'])
        cfg['grp']['control']['val_steps'] = int(settings['val_steps'])
        cfg['grp']['dataset']['train_globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['grp']['dataset']['val_globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['grp']['dataset']['file_batch_size'] = int(settings['file_batch_size'])
        cfg['grp']['optim']['lr'] = float(settings['lr'])
        config_file = grp_dir / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_logged(
            [sys.executable, str(self.mortal_root / 'mortal' / 'train_grp.py')],
            config_file,
            grp_dir / 'train.log',
            float(settings['timeout_hours']),
        )
        final_step = _grp_step(candidate_state)
        if final_step < target:
            raise RuntimeError(f'GRP stopped at {final_step}, expected at least {target}')

        validation = settings['validation']
        validation_report_file = grp_dir / 'validation.json'
        validation_seed = _seed_from_text(f'{run_dir.name}:grp-validation')
        self._run_logged(
            [
                sys.executable,
                str(self.mortal_root / 'mortal' / 'evaluate_grp.py'),
                '--incumbent', str(incumbent_state),
                '--candidate', str(candidate_state),
                '--index', str(grp_index),
                '--output', str(validation_report_file),
                '--bootstrap-samples', str(int(validation['bootstrap_samples'])),
                '--bootstrap-seed', str(validation_seed),
            ],
            config_file,
            grp_dir / 'validation.log',
            float(validation.get('timeout_hours', settings['timeout_hours'])),
        )
        validation_report = _load_json(validation_report_file, None)
        direct_passed = grp_validation_passes(
            validation_report,
            require_significant=bool(validation.get('require_significant', True)),
            min_nll_reduction=float(validation.get('min_nll_reduction', 0.0)),
            min_brier_reduction=float(validation.get('min_brier_reduction', 0.0)),
        )

        downstream = settings.get('downstream', {})
        downstream_enabled = bool(downstream.get('enabled', True))
        downstream_report = None
        downstream_passed = False
        if direct_passed and downstream_enabled:
            downstream_report = self._run_grp_downstream_gate(
                run_dir,
                dataset_index,
                records,
                incumbent_state,
                candidate_state,
            )
            downstream_passed = bool(downstream_report['passed'])
        promote = should_promote_grp(
            direct_passed,
            downstream_enabled=downstream_enabled,
            downstream_passed=downstream_passed,
        )
        if promote:
            atomic_promote_checkpoint(candidate_state, grp_state)
            reason = 'direct and downstream gates passed' if downstream_enabled else 'direct gate passed'
        elif not direct_passed:
            reason = 'direct GRP validation failed'
        else:
            reason = 'downstream Mortal A/B gate failed'
        return {
            'from_step': from_step,
            'target_step': target,
            'final_step': final_step,
            'selected_step': _grp_step(grp_state),
            'train_logs': len(train_records),
            'validation_logs': len(val_records),
            'config': str(config_file),
            'incumbent_model': str(incumbent_state),
            'candidate_model': str(candidate_state),
            'validation_report': str(validation_report_file),
            'direct_passed': direct_passed,
            'downstream_report': (
                str(grp_dir / 'downstream' / 'report.json') if downstream_report else None
            ),
            'downstream_passed': downstream_passed if downstream_enabled else None,
            'promoted': promote,
            'promotion_reason': reason,
        }

    def _run_grp_downstream_gate(
        self,
        run_dir: Path,
        dataset_index: Path,
        records: Sequence[LogRecord],
        incumbent_grp: Path,
        candidate_grp: Path,
    ) -> dict[str, Any]:
        settings = self.config['grp']['downstream']
        offline_settings = self.config['offline']
        root = run_dir / 'grp' / 'downstream'
        source_state = root / 'source.pth'
        source_state.parent.mkdir(parents=True, exist_ok=True)
        if not source_state.exists():
            shutil.copy2(self._training_seed(), source_state)

        results = []
        seed_reports = []
        for training_seed in [int(seed) for seed in settings['training_seeds']]:
            seed_dir = root / f'seed_{training_seed}'
            incumbent_model = seed_dir / 'incumbent' / 'model.pth'
            candidate_model = seed_dir / 'candidate' / 'model.pth'
            incumbent_training = self._train_offline_checkpoint(
                source=source_state,
                grp_state=incumbent_grp,
                dataset_index=dataset_index,
                records=records,
                state_file=incumbent_model,
                settings=offline_settings,
                updates=int(settings['training_updates']),
                random_seed=training_seed,
                deterministic=True,
                timeout_hours=float(settings.get(
                    'training_timeout_hours',
                    offline_settings['timeout_hours'],
                )),
                model_id=f'grp-incumbent-seed-{training_seed}',
            )
            candidate_training = self._train_offline_checkpoint(
                source=source_state,
                grp_state=candidate_grp,
                dataset_index=dataset_index,
                records=records,
                state_file=candidate_model,
                settings=offline_settings,
                updates=int(settings['training_updates']),
                random_seed=training_seed,
                deterministic=True,
                timeout_hours=float(settings.get(
                    'training_timeout_hours',
                    offline_settings['timeout_hours'],
                )),
                model_id=f'grp-candidate-seed-{training_seed}',
            )
            incumbent = inspect_checkpoint(
                incumbent_model,
                f'grp-incumbent-seed-{training_seed}',
            ).candidate
            candidate = inspect_checkpoint(
                candidate_model,
                f'grp-candidate-seed-{training_seed}',
            ).candidate
            evaluation_seed = _seed_from_text(
                f'{run_dir.name}:grp-downstream:{training_seed}'
            )
            result = self._evaluate_candidate(
                candidate,
                incumbent,
                int(settings['games']),
                evaluation_seed,
                seed_dir / 'evaluation',
                float(settings.get('evaluation_timeout_hours', 6)),
            )
            results.append(result)
            seed_reports.append({
                'training_seed': training_seed,
                'evaluation_seed': evaluation_seed,
                'incumbent_training': incumbent_training,
                'candidate_training': candidate_training,
                'result': _result_dict(result),
            })

        combined = aggregate_eval_results(results, 'grp-candidate-combined')
        passed = grp_downstream_passes(
            results,
            require_significant=bool(settings.get('require_significant', True)),
            require_positive_pt=bool(settings.get('require_positive_pt', True)),
            require_each_seed_non_regressing=bool(
                settings.get('require_each_seed_non_regressing', True)
            ),
        )
        report = {
            'incumbent_grp': str(incumbent_grp),
            'candidate_grp': str(candidate_grp),
            'source_model': str(source_state),
            'games_per_seed': int(settings['games']),
            'seeds': seed_reports,
            'combined': _result_dict(combined),
            'passed': passed,
        }
        _atomic_json(root / 'report.json', report)
        return report

    def _train_offline_checkpoint(
        self,
        *,
        source: Path,
        grp_state: Path,
        dataset_index: Path,
        records: Sequence[LogRecord],
        state_file: Path,
        settings: Mapping[str, Any],
        updates: int,
        random_seed: int | None,
        deterministic: bool,
        timeout_hours: float,
        model_id: str,
    ) -> dict[str, Any]:
        source_info = inspect_checkpoint(source, model_id)
        if not source_info.full_training_state:
            raise ValueError(f'offline seed is not a full training checkpoint: {source}')
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if not state_file.exists():
            shutil.copy2(source, state_file)
        start_step = int(source_info.candidate.step or 0)
        cfg = build_offline_config(
            self._base_config(source_info),
            state_file=state_file,
            dataset_index=dataset_index,
            grp_state=grp_state,
            reference_model=self.reference_model,
            run_dir=state_file.parent,
            start_step=start_step,
            updates=updates,
            save_every=int(settings['save_every']),
            peak_lr=float(settings['peak_lr']),
            final_lr=float(settings['final_lr']),
            warm_up_steps=int(settings['warm_up_steps']),
            snapshot_every=(
                int(settings.get('snapshot_every', 0)) if model_id == 'offline' else 0
            ),
            random_seed=random_seed,
            deterministic=deterministic,
        )
        cfg['dataset']['globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['dataset']['file_batch_size'] = int(settings['file_batch_size'])
        cfg['dataset']['num_workers'] = 0 if deterministic else int(settings['num_workers'])
        cfg['dataset']['num_epochs'] = int(settings['num_epochs'])
        config_file = state_file.parent / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_logged(
            [sys.executable, str(self.mortal_root / 'mortal' / 'train.py')],
            config_file,
            state_file.parent / 'train.log',
            timeout_hours,
        )
        final_info = inspect_checkpoint(state_file, model_id)
        target = int(cfg['control']['max_steps'])
        if int(final_info.candidate.step or 0) < target:
            raise RuntimeError(f'offline training stopped before target {target}')
        return {
            'source_model': str(source),
            'grp_model': str(grp_state),
            'random_seed': random_seed,
            'start_step': start_step,
            'target_step': target,
            'final_step': final_info.candidate.step,
            'logs': len(records),
            'estimated_steps_per_epoch': estimate_steps_per_epoch(
                len(records),
                batch_size=int(cfg['control']['batch_size']),
                samples_per_log=int(self.config['pipeline'].get('estimated_samples_per_log', 660)),
            ),
            'model': str(state_file),
            'snapshots': str(state_file.parent / 'snapshots'),
            'config': str(config_file),
        }

    def _run_offline(
        self,
        run_dir: Path,
        dataset_index: Path,
        grp_state: Path,
        records: Sequence[LogRecord],
        state_file: Path,
        target_step: int | None = None,
    ) -> dict:
        settings = self.config['offline']
        if not settings.get('enabled', True):
            return {'status': 'skipped', 'reason': 'disabled'}
        if target_step is None:
            return self._train_offline_checkpoint(
                source=self._training_seed(),
                grp_state=grp_state,
                dataset_index=dataset_index,
                records=records,
                state_file=state_file,
                settings=settings,
                updates=int(settings['updates']),
                random_seed=(
                    int(settings['random_seed'])
                    if settings.get('random_seed') is not None
                    else None
                ),
                deterministic=bool(settings.get('deterministic', True)),
                timeout_hours=float(settings['timeout_hours']),
                model_id='offline',
            )

        source = self._training_seed()
        source_info = inspect_checkpoint(source, 'training_seed')
        if not source_info.full_training_state:
            raise ValueError(f'offline seed is not a full training checkpoint: {source}')
        state_file.parent.mkdir(parents=True, exist_ok=True)
        resume_optimizer = state_file.exists()
        if not state_file.exists():
            shutil.copy2(source, state_file)
        current_info = inspect_checkpoint(state_file, 'offline')
        current_step = int(current_info.candidate.step or 0)
        if current_step >= target_step:
            return {
                'source_model': str(source),
                'start_step': current_step,
                'target_step': target_step,
                'final_step': current_step,
                'logs': len(records),
                'model': str(state_file),
                'status': 'complete',
            }

        cfg = build_offline_config(
            self._base_config(current_info),
            state_file=state_file,
            dataset_index=dataset_index,
            grp_state=grp_state,
            reference_model=self.reference_model,
            run_dir=state_file.parent,
            start_step=current_step,
            target_step=target_step,
            save_every=int(settings['save_every']),
            peak_lr=float(settings['peak_lr']),
            final_lr=float(settings['final_lr']),
            warm_up_steps=int(settings['warm_up_steps']),
            resume_optimizer=resume_optimizer,
            snapshot_every=int(settings.get('snapshot_every', 0)),
        )
        cfg['dataset']['globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['dataset']['file_batch_size'] = int(settings['file_batch_size'])
        cfg['dataset']['num_workers'] = int(settings['num_workers'])
        cfg['dataset']['num_epochs'] = int(settings['num_epochs'])
        config_file = state_file.parent / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_logged(
            [sys.executable, str(self.mortal_root / 'mortal' / 'train.py')],
            config_file,
            state_file.parent / 'train.log',
            float(settings['timeout_hours']),
        )
        final_info = inspect_checkpoint(state_file, 'offline')
        final_step = int(final_info.candidate.step or 0)
        if final_step < target_step:
            raise RuntimeError(f'offline training stopped before target {target_step}')
        return {
            'source_model': str(source),
            'start_step': current_step,
            'target_step': target_step,
            'final_step': final_step,
            'logs': len(records),
            'model': str(state_file),
            'deployment_model': str(state_file.parent / 'model_deploy.pth'),
            'snapshots': str(state_file.parent / 'snapshots'),
            'config': str(config_file),
        }

    def _select_offline_candidate(
        self,
        run_dir: Path,
        offline_state: Path,
    ) -> tuple[Candidate, dict[str, Any]]:
        parent = inspect_checkpoint(self._training_seed(), 'training-parent').candidate
        if not offline_state.exists():
            return parent, {'status': 'fallback', 'reason': 'offline model is unavailable'}

        snapshot_dir = offline_state.parent / 'snapshots'
        paths = list(snapshot_dir.glob('*.pth'))
        paths.append(offline_state)
        candidates = deduplicate_candidates([
            inspect_checkpoint(path, f'offline-{path.stem}').candidate
            for path in paths
        ])
        candidates.sort(key=lambda candidate: (candidate.step or -1, candidate.model_id))
        candidates = evenly_spaced_candidates(
            candidates,
            int(self.config['offline'].get('selection_max_candidates', 8)),
        )
        seed = _seed_from_text(f'{run_dir.name}:offline-selection')
        selection_root = offline_state.parent / 'selection'
        results = [
            self._evaluate_candidate(
                candidate,
                parent,
                int(self.config['offline'].get('selection_games', 512)),
                seed,
                selection_root / 'logs',
                float(self.config['evaluation']['timeout_hours']),
            )
            for candidate in candidates
        ]
        best = min(results, key=lambda result: (result.avg_rank, -result.avg_pt))
        selected = next(
            candidate for candidate in candidates if candidate.fingerprint == best.fingerprint
        )
        if best.avg_rank >= BREAKEVEN_RANK or best.avg_pt <= 0:
            selected = parent
            reason = 'no offline snapshot had a positive point estimate against the parent'
        else:
            reason = 'best common-seed offline snapshot'
        report = {
            'seed': seed,
            'reference': {
                'path': str(parent.path),
                'fingerprint': parent.fingerprint,
                'step': parent.step,
            },
            'results': [_result_dict(result) for result in sorted(results, key=lambda item: item.avg_rank)],
            'selected': {
                'model_id': selected.model_id,
                'path': str(selected.path),
                'fingerprint': selected.fingerprint,
                'step': selected.step,
            },
            'selection_reason': reason,
        }
        report_file = selection_root / 'report.json'
        _atomic_json(report_file, report)
        return selected, {**report, 'report': str(report_file)}

    def _build_online_opponents(self) -> list[dict[str, Any]]:
        reference = inspect_checkpoint(self.reference_model, 'baseline').candidate
        parent = inspect_checkpoint(self._training_seed(), 'training-parent').candidate
        pool = [
            {
                'name': 'baseline',
                'state_file': str(reference.path),
                'fingerprint': reference.fingerprint,
                'weight': 0.35,
            },
            {
                'name': 'training-parent',
                'state_file': str(parent.path),
                'fingerprint': parent.fingerprint,
                'weight': 0.40,
            },
        ]
        seen = {reference.fingerprint, parent.fingerprint}
        historical = []
        for metadata_file in self.models_dir.glob('*.json'):
            metadata = _load_json(metadata_file, {})
            model_file = metadata_file.with_suffix('.pth')
            if not model_file.exists() or 'avg_rank' not in metadata:
                continue
            try:
                candidate = inspect_checkpoint(model_file, metadata_file.stem).candidate
            except ValueError:
                continue
            if candidate.fingerprint in seen:
                continue
            seen.add(candidate.fingerprint)
            historical.append((float(metadata['avg_rank']), -float(metadata.get('avg_pt', 0)), candidate))
        historical.sort(key=lambda item: (item[0], item[1]))
        max_opponents = int(self.config['online'].get('max_opponents', 3))
        for _, _, candidate in historical[:max(0, max_opponents - len(pool))]:
            pool.append({
                'name': f'historical-{candidate.fingerprint[:8]}',
                'state_file': str(candidate.path),
                'fingerprint': candidate.fingerprint,
                'weight': 0.25,
            })
        return normalize_opponent_pool(pool)

    def _run_online(
        self,
        run_dir: Path,
        dataset_index: Path,
        grp_state: Path,
        records: Sequence[LogRecord],
        offline_state: Path,
        state_file: Path,
        phase_start_step: int | None = None,
        target_step: int | None = None,
    ) -> dict:
        settings = self.config['online']
        if not settings.get('enabled', True):
            return {'status': 'skipped', 'reason': 'disabled'}
        if target_step is None:
            return self._run_online_yonma(
                run_dir,
                dataset_index,
                grp_state,
                records,
                offline_state,
                state_file,
            )
        return self._run_online_sanma(
            run_dir,
            dataset_index,
            grp_state,
            records,
            offline_state,
            state_file,
            phase_start_step,
            target_step,
        )

    def _run_online_yonma(
        self,
        run_dir: Path,
        dataset_index: Path,
        grp_state: Path,
        records: Sequence[LogRecord],
        offline_state: Path,
        state_file: Path,
    ) -> dict:
        settings = self.config['online']
        selected, selection_report = self._select_offline_candidate(run_dir, offline_state)
        source = selected.path
        source_info = inspect_checkpoint(source, 'online_seed')
        if not source_info.full_training_state:
            raise ValueError(f'online seed is not a full training checkpoint: {source}')
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if not state_file.exists():
            shutil.copy2(source, state_file)
        opponents = self._build_online_opponents()
        if state_file.exists():
            # Resume must continue from the on-disk state's actual step;
            # the freshly selected candidate is only the seeding fallback.
            seeded_info = inspect_checkpoint(state_file, 'online_resume')
            start_step = int(seeded_info.candidate.step or 0)
        else:
            start_step = int(source_info.candidate.step or 0)
        cfg = build_online_config(
            self._base_config(source_info),
            state_file=state_file,
            dataset_index=dataset_index,
            grp_state=grp_state,
            reference_model=self.reference_model,
            deployed_model=self._training_seed(),
            historical_model=self._training_seed(),
            run_dir=state_file.parent,
            start_step=start_step,
            updates=int(settings['updates']),
            save_every=int(settings['save_every']),
            snapshot_every=int(settings['snapshot_every']),
            brain_freeze_updates=int(settings['brain_freeze_updates']),
            expert_ratio=float(settings['expert_ratio']),
            peak_lr=float(settings['peak_lr']),
            port=int(settings['port']),
            games_per_session=int(settings['games_per_session']),
            opponents=opponents,
            opponent_selection=settings.get('opponent_selection'),
        )
        cfg['online']['remote']['host'] = str(settings.get('host', '127.0.0.1'))
        cfg['dataset']['globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['dataset']['file_batch_size'] = int(settings['file_batch_size'])
        cfg['dataset']['num_workers'] = int(settings['num_workers'])
        cfg['train_play']['default']['repeats'] = int(settings['repeats'])
        config_file = state_file.parent / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_online_processes(config_file, state_file.parent, settings)
        final_info = inspect_checkpoint(state_file, 'online')
        target = int(cfg['control']['max_steps'])
        if int(final_info.candidate.step or 0) < target:
            raise RuntimeError(f'online training stopped before target {target}')
        return {
            'source_model': str(source),
            'start_step': start_step,
            'target_step': target,
            'final_step': final_info.candidate.step,
            'expert_logs': len(records),
            'offline_selection': selection_report,
            'opponents': opponents,
            'model': str(state_file),
            'snapshots': str(state_file.parent / 'snapshots'),
            'config': str(config_file),
        }

    def _write_online_phase_bookkeeping(
        self,
        state_file: Path,
        source: Path,
        start_step: int,
        target_step: int,
    ) -> None:
        """Record the online phase window next to the state for auditability."""
        bookkeeping = {
            'seeded_from': str(source),
            'start_step': int(start_step),
            'target_step': int(target_step),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        path = state_file.parent / 'online_phase.json'
        path.write_text(json.dumps(bookkeeping, ensure_ascii=False, indent=2) + '\n')

    def _run_online_sanma(
        self,
        run_dir: Path,
        dataset_index: Path,
        grp_state: Path,
        records: Sequence[LogRecord],
        offline_state: Path,
        state_file: Path,
        phase_start_step: int,
        target_step: int,
    ) -> dict:
        settings = self.config['online']
        resuming = state_file.exists()
        selection_report = None
        if resuming:
            source = state_file
        else:
            selected, selection_report = self._select_offline_candidate(
                run_dir,
                offline_state,
            )
            source = selected.path
        source_info = inspect_checkpoint(source, 'online_seed')
        if not source_info.full_training_state:
            raise ValueError(f'online seed is not a full training checkpoint: {source}')
        state_file.parent.mkdir(parents=True, exist_ok=True)
        resume_optimizer = resuming
        if not resuming:
            shutil.copy2(source, state_file)
        current_info = inspect_checkpoint(state_file, 'online')
        current_step = int(current_info.candidate.step or 0)
        if resuming:
            # The on-disk state may predate the phase's step numbering (it
            # was seeded from an earlier offline snapshot); rebasing onto
            # the state's actual step keeps the update budget exact and
            # prevents silent over-training toward a stale target.
            effective_start_step, effective_target_step, rebased = (
                resolve_online_resume_numbering(
                    current_step, phase_start_step, target_step
                )
            )
            if rebased:
                logging.warning(
                    'online state step %d predates phase start %d; '
                    'rebased window to [%d, %d]',
                    current_step,
                    phase_start_step,
                    effective_start_step,
                    effective_target_step,
                )
                self._write_online_phase_bookkeeping(
                    state_file,
                    source,
                    effective_start_step,
                    effective_target_step,
                )
        else:
            effective_start_step = int(source_info.candidate.step or 0)
            online_updates = target_step - phase_start_step
            if online_updates < 0:
                raise ValueError('online target must not precede the phase start')
            effective_target_step = effective_start_step + online_updates
            self._write_online_phase_bookkeeping(
                state_file,
                source,
                effective_start_step,
                effective_target_step,
            )
        if current_step >= effective_target_step:
            return {
                'source_model': str(source),
                'start_step': effective_start_step,
                'target_step': effective_target_step,
                'final_step': current_step,
                'expert_logs': len(records),
                'model': str(state_file),
                'snapshots': str(state_file.parent / 'snapshots'),
                'offline_selection': selection_report,
                'status': 'complete',
            }

        historical = self._training_seed()
        champion = self.manifest.get('champion')
        if champion and Path(champion.get('path', '')).exists():
            historical = Path(champion['path']).resolve()
        league_models = self._league_model_paths()
        if not league_models:
            league_models = [historical]
        opponent_cfg = settings.get('opponents', {})
        cfg = build_online_config(
            self._base_config(current_info),
            state_file=state_file,
            dataset_index=dataset_index,
            grp_state=grp_state,
            reference_model=self.reference_model,
            deployed_model=source,
            historical_model=historical,
            run_dir=state_file.parent,
            start_step=effective_start_step,
            target_step=effective_target_step,
            save_every=int(settings['save_every']),
            snapshot_every=int(settings['snapshot_every']),
            brain_freeze_updates=int(settings['brain_freeze_updates']),
            expert_ratio=float(settings['expert_ratio']),
            peak_lr=float(settings['peak_lr']),
            port=int(settings['port']),
            games_per_session=int(settings['games_per_session']),
            resume_optimizer=resume_optimizer,
            opponent_selection=settings.get('opponent_selection'),
            league_models=league_models,
            opponent_weights={
                'anchor': float(opponent_cfg.get('anchor_weight', 0.1)),
                'deployed': float(opponent_cfg.get('deployed_weight', 0.4)),
                'league': float(opponent_cfg.get('league_weight', 0.5)),
            },
        )
        cfg['online']['remote']['host'] = str(settings.get('host', '127.0.0.1'))
        cfg['dataset']['globs'] = [str(self.logs_root / '**' / '*.mjson')]
        cfg['dataset']['file_batch_size'] = int(settings['file_batch_size'])
        cfg['dataset']['num_workers'] = int(settings['num_workers'])
        cfg['train_play']['default']['repeats'] = int(settings['repeats'])
        config_file = state_file.parent / 'config.toml'
        _write_toml(config_file, cfg)
        self._run_online_processes(config_file, state_file.parent, settings)
        final_info = inspect_checkpoint(state_file, 'online')
        final_step = int(final_info.candidate.step or 0)
        if final_step < effective_target_step:
            raise RuntimeError(
                f'online training stopped before target {effective_target_step}'
            )
        return {
            'source_model': str(source),
            'start_step': effective_start_step,
            'target_step': effective_target_step,
            'final_step': final_step,
            'expert_logs': len(records),
            'offline_selection': selection_report,
            'model': str(state_file),
            'deployment_model': str(state_file.parent / 'model_deploy.pth'),
            'snapshots': str(state_file.parent / 'snapshots'),
            'config': str(config_file),
        }

    def _run_online_processes(self, config_file: Path, run_dir: Path, settings: Mapping) -> None:
        self._wait_for_gpu_slot()
        env = os.environ.copy()
        env['MORTAL_CFG'] = str(config_file)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        processes: list[subprocess.Popen] = []
        process_roles: dict[str, subprocess.Popen] = {}
        handles = []
        try:
            for name in ('server', 'trainer', 'client'):
                handle = (run_dir / f'{name}.log').open('ab')
                handles.append(handle)
            server = subprocess.Popen(
                [sys.executable, str(self.mortal_root / 'mortal' / 'server.py')],
                cwd=self.mortal_root,
                env=env,
                stdout=handles[0],
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            processes.append(server)
            process_roles['server'] = server
            self._wait_for_port(
                str(settings.get('host', '127.0.0.1')),
                int(settings['port']),
                60,
                server,
            )
            trainer = subprocess.Popen(
                [sys.executable, str(self.mortal_root / 'mortal' / 'train.py')],
                cwd=self.mortal_root,
                env=env,
                stdout=handles[1],
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            processes.append(trainer)
            process_roles['trainer'] = trainer
            time.sleep(1)
            for client_index in range(int(settings.get('clients', 1))):
                client = subprocess.Popen(
                    [sys.executable, str(self.mortal_root / 'mortal' / 'client.py')],
                    cwd=self.mortal_root,
                    env=env,
                    stdout=handles[2],
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
                processes.append(client)
                process_roles[f'client[{client_index}]'] = client
            self._wait_for_online_processes(
                process_roles,
                timeout_seconds=float(settings['timeout_hours']) * 3600,
            )
        finally:
            for process in reversed(processes):
                self._terminate_tree(process)
            for handle in handles:
                handle.close()

    @staticmethod
    def _wait_for_online_processes(
        processes: Mapping[str, subprocess.Popen],
        timeout_seconds: float,
        poll_interval: float = 0.2,
    ) -> None:
        trainer = processes.get('trainer')
        if trainer is None:
            raise ValueError('online process monitor requires a trainer')

        deadline = time.monotonic() + timeout_seconds
        while True:
            statuses = {name: process.poll() for name, process in processes.items()}
            for name, code in statuses.items():
                if name == 'trainer' or code is None:
                    continue
                raise RuntimeError(
                    f'online {name} exited unexpectedly with code {code}: '
                    f'{processes[name].args}'
                )

            trainer_code = statuses['trainer']
            if trainer_code is not None:
                if trainer_code != 0:
                    raise subprocess.CalledProcessError(trainer_code, trainer.args)
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(trainer.args, timeout_seconds)
            time.sleep(min(poll_interval, remaining))

    @staticmethod
    def _wait_for_port(host: str, port: int, timeout: float, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'online server exited with code {process.returncode}')
            try:
                with socket.create_connection((host, port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(f'online server did not listen on {host}:{port}')

    @staticmethod
    def _terminate_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == 'nt':
            subprocess.run(
                ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def _candidate_sources(
        self,
        offline_state: Path,
        online_state: Path,
    ) -> list[tuple[str, Path]]:
        if not offline_state.exists() or not online_state.exists():
            latest_offline, latest_online = self._latest_training_states()
            if not offline_state.exists() and latest_offline is not None:
                offline_state = latest_offline
            if not online_state.exists() and latest_online is not None:
                online_state = latest_online
        sources = [
            ('baseline', self.reference_model),
            ('training-seed', self._training_seed()),
        ]
        if offline_state.exists():
            sources.append(('offline', offline_state))
            selection_report = offline_state.parent / 'selection' / 'report.json'
            if selection_report.exists():
                selected_path = _load_json(selection_report, {}).get('selected', {}).get('path')
                if selected_path and Path(selected_path).exists():
                    sources.append(('offline-selected', Path(selected_path)))
            offline_snapshot_dir = offline_state.parent / 'snapshots'
            sources.extend(
                (f'offline-{snapshot.stem}', snapshot)
                for snapshot in sorted(offline_snapshot_dir.glob('*.pth'))
            )
        if online_state.exists():
            sources.append(('online-final', online_state))
            snapshot_dir = online_state.parent / 'snapshots'
            sources.extend(
                (f'online-{snapshot.stem}', snapshot)
                for snapshot in sorted(snapshot_dir.glob('*.pth'))
            )
        sources.extend(
            (f'retained-{fingerprint[:12]}', model)
            for fingerprint, model in self._retained_models_by_fingerprint().items()
        )
        return sources

    def _latest_training_states(self) -> tuple[Path | None, Path | None]:
        runs_dir = self.output_root / 'runs'
        if not runs_dir.exists():
            return None, None
        run_dirs = sorted(
            (path for path in runs_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for run_dir in run_dirs:
            offline = run_dir / 'offline' / 'model.pth'
            online = run_dir / 'online' / 'model.pth'
            if offline.exists() or online.exists():
                return (
                    offline if offline.exists() else None,
                    online if online.exists() else None,
                )
        return None, None

    def _retained_models_by_fingerprint(self) -> dict[str, Path]:
        retained = {}
        for model in sorted(self.models_dir.glob('*.pth')):
            metrics_file = model.with_suffix('.json')
            fingerprint = None
            if metrics_file.exists():
                metrics = _load_json(metrics_file, {})
                fingerprint = metrics.get('fingerprint')
            if not fingerprint:
                fingerprint = inspect_checkpoint(model, model.stem).candidate.fingerprint
            retained.setdefault(str(fingerprint), model)
        return retained

    def _run_evaluation(
        self,
        run_dir: Path,
        offline_state: Path,
        online_state: Path,
    ) -> dict:
        settings = self.config['evaluation']
        if not settings.get('enabled', True):
            return {'status': 'skipped', 'reason': 'disabled'}
        strategy = settings.get('strategy') or (
            'direct_ladder' if NUM_PLAYERS == 3 else 'sequential'
        )
        if strategy == 'direct_ladder':
            return self._run_evaluation_direct_ladder(run_dir, offline_state, online_state)
        return self._run_evaluation_sequential(run_dir, offline_state, online_state)

    def _run_evaluation_sequential(self, run_dir: Path, offline_state: Path, online_state: Path) -> dict:
        settings = self.config['evaluation']
        candidates = deduplicate_candidates([
            inspect_checkpoint(path, model_id).candidate
            for model_id, path in self._candidate_sources(offline_state, online_state)
        ])
        reference = next(candidate for candidate in candidates if candidate.model_id == 'baseline')
        eval_dir = run_dir / 'evaluation'
        screen_seed = _seed_from_text(f'{run_dir.name}:screen')
        screen_results = [
            self._evaluate_candidate(
                candidate,
                reference,
                int(settings['screen_games']),
                screen_seed,
                eval_dir / 'screen',
                float(settings['timeout_hours']),
            )
            for candidate in candidates
        ]
        non_reference = sorted(
            (result for result in screen_results if result.model_id != 'baseline'),
            key=lambda result: (result.avg_rank, -result.avg_pt),
        )
        finalists = [reference] + [
            next(candidate for candidate in candidates if candidate.model_id == result.model_id)
            for result in non_reference[:int(settings['finalists'])]
        ]
        validation_results = []
        validation_decisions = {}
        validation_ledgers = {}
        for candidate in finalists:
            if candidate.fingerprint == reference.fingerprint:
                continue
            result, decision, ledger_key = self._sequential_evaluate(
                candidate,
                reference,
                purpose='deployment',
                block_games=int(settings['validation_block_games']),
                max_games=int(settings['validation_max_games']),
                require_positive_pt=bool(settings.get('require_positive_pt', True)),
                timeout_hours=float(settings['timeout_hours']),
            )
            validation_results.append(result)
            validation_decisions[result.model_id] = decision
            validation_ledgers[result.model_id] = ledger_key

        eligible = [
            result for result in validation_results
            if validation_decisions[result.model_id] == 'promote'
        ]
        winner = min(eligible, key=lambda result: (result.avg_rank, -result.avg_pt)) if eligible else next(
            result for result in screen_results if result.model_id == 'baseline'
        )
        retained = self._retain_validated_models(validation_results) if validation_results else {}
        winner_path = retained[winner.model_id] if winner.model_id in retained else self.reference_model

        parent_report = None
        parent_candidates = sorted(
            validation_results,
            key=lambda result: (result.avg_rank, -result.avg_pt),
        )
        if parent_candidates:
            parent_result = parent_candidates[0]
            parent_info = inspect_checkpoint(parent_result.path, parent_result.model_id)
            current_parent = inspect_checkpoint(self._training_seed(), 'training-parent').candidate
            if (
                parent_info.full_training_state
                and parent_result.fingerprint != current_parent.fingerprint
            ):
                parent_settings = self.config['parent_evaluation']
                parent_eval, parent_decision, parent_ledger = self._sequential_evaluate(
                    parent_info.candidate,
                    current_parent,
                    purpose='training-parent',
                    block_games=int(parent_settings['block_games']),
                    max_games=int(parent_settings['max_games']),
                    require_positive_pt=bool(parent_settings.get('require_positive_pt', True)),
                    timeout_hours=float(parent_settings['timeout_hours']),
                )
                parent_retained = None
                if parent_decision == 'promote':
                    parent_retained = self._retain_validated_models([parent_eval])[parent_eval.model_id]
                    promote_training_parent(self.manifest, parent_eval, parent_retained)
                parent_report = {
                    'result': _result_dict(parent_eval),
                    'decision': parent_decision,
                    'ledger_key': parent_ledger,
                    'retained_parent': str(parent_retained) if parent_retained else None,
                }
        report = {
            'reference_model': str(self.reference_model),
            'screen_seed': screen_seed,
            'screen': [_result_dict(result) for result in sorted(screen_results, key=lambda item: item.avg_rank)],
            'validation': [_result_dict(result) for result in sorted(validation_results, key=lambda item: item.avg_rank)],
            'validation_decisions': validation_decisions,
            'validation_ledgers': validation_ledgers,
            'training_parent_evaluation': parent_report,
            'winner': _result_dict(winner),
            'retained_winner': str(winner_path),
        }
        _atomic_json(eval_dir / 'report.json', report)
        self.manifest['deployment_champion'] = {
            **_result_dict(winner),
            'path': str(winner_path),
        }
        self.manifest['champion'] = deepcopy(self.manifest['deployment_champion'])
        self._save_manifest()
        return {
            'candidate_count': len(candidates),
            'finalist_count': len(finalists),
            'winner': winner.model_id,
            'winner_model': str(winner_path),
            'winner_avg_rank': winner.avg_rank,
            'winner_avg_pt': winner.avg_pt,
            'report': str(eval_dir / 'report.json'),
        }

    def _run_evaluation_direct_ladder(
        self,
        run_dir: Path,
        offline_state: Path,
        online_state: Path,
    ) -> dict:
        settings = self.config['evaluation']
        candidates = deduplicate_candidates(
            [
                inspect_checkpoint(path, model_id).candidate
                for model_id, path in self._candidate_sources(
                    offline_state,
                    online_state,
                )
            ]
        )
        reference = next(
            candidate for candidate in candidates if candidate.model_id == 'baseline'
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate.fingerprint != reference.fingerprint
        ]
        if len(eligible) < 3:
            raise RuntimeError(
                f'evaluation needs at least three non-anchor models, got {len(eligible)}'
            )

        eval_dir = run_dir / 'evaluation'
        screen_seed = _seed_from_text(f'{run_dir.name}:screen')
        screen_games = int(settings['screen_games'])
        anchor_count = screen_games // NUM_PLAYERS
        anchor_result = EvalResult.from_rankings(
            reference.model_id,
            reference.path,
            [anchor_count] * (NUM_PLAYERS - 1) + [screen_games - (NUM_PLAYERS - 1) * anchor_count],
            reference.step,
            reference.fingerprint,
        )
        candidate_screen_results = [
            self._evaluate_candidate(
                candidate,
                reference,
                screen_games,
                screen_seed,
                eval_dir / 'screen',
                float(settings['timeout_hours']),
            )
            for candidate in eligible
        ]
        screen_results = [anchor_result, *candidate_screen_results]
        screen_by_fingerprint = {
            result.fingerprint: result for result in candidate_screen_results
        }
        ordered_candidates = sorted(
            eligible,
            key=lambda candidate: (
                screen_by_fingerprint[candidate.fingerprint].avg_rank,
                -screen_by_fingerprint[candidate.fingerprint].avg_pt,
            ),
        )

        incumbent_fingerprints = self._roster_fingerprints()
        candidate_by_fingerprint = {
            candidate.fingerprint: candidate for candidate in eligible
        }
        roster = [
            candidate_by_fingerprint[fingerprint]
            for fingerprint in incumbent_fingerprints
            if fingerprint in candidate_by_fingerprint
        ]
        bootstrap = len(roster) != 3
        if bootstrap:
            roster = ordered_candidates[:3]

        roster_fingerprints = {candidate.fingerprint for candidate in roster}
        challengers = [
            candidate
            for candidate in ordered_candidates
            if candidate.fingerprint not in roster_fingerprints
        ]
        challenger_limit = int(
            settings.get('max_challengers', settings.get('finalists', 3))
        )
        if challenger_limit > 0:
            challengers = challengers[:challenger_limit]

        direct_root = eval_dir / 'direct'
        direct_initial_games, direct_max_games = direct_game_schedule(settings)
        direct_seed = _seed_from_text(f'{run_dir.name}:direct')
        comparison_cache: dict[tuple[str, str], DirectComparison] = {}

        def compare(challenger: Candidate, incumbent: Candidate) -> EvalResult:
            key = tuple(sorted((challenger.fingerprint, incumbent.fingerprint)))
            comparison = comparison_cache.get(key)
            if comparison is None:
                comparison = self._evaluate_direct_pair(
                    challenger,
                    incumbent,
                    direct_initial_games,
                    direct_seed,
                    direct_root,
                    float(settings['timeout_hours']),
                )
                comparison_cache[key] = comparison
            return comparison.result_for(challenger)

        require_significant = bool(settings.get('require_significant', True))
        require_positive_pt = bool(settings.get('require_positive_pt', True))

        if bootstrap:
            for start in range(1, len(roster)):
                position = start
                while position > 0:
                    result = compare(roster[position], roster[position - 1])
                    if not is_significant_win(
                        result,
                        require_significant=require_significant,
                        require_positive_pt=require_positive_pt,
                    ):
                        break
                    roster[position - 1], roster[position] = (
                        roster[position],
                        roster[position - 1],
                    )
                    position -= 1

        promotions = []
        for challenger in challengers:
            before = list(roster)
            roster, evidence = direct_ladder_update(
                roster,
                challenger,
                compare,
                require_significant=require_significant,
                require_positive_pt=require_positive_pt,
            )
            before_fingerprints = {candidate.fingerprint for candidate in before}
            after_fingerprints = {candidate.fingerprint for candidate in roster}
            promotions.append({
                'challenger': challenger.model_id,
                'challenger_fingerprint': challenger.fingerprint,
                'accepted': challenger.fingerprint in after_fingerprints,
                'replaced_fingerprints': sorted(
                    before_fingerprints - after_fingerprints
                ),
                'evidence': [_result_dict(result) for result in evidence],
            })

        for first_index in range(len(roster)):
            for second_index in range(first_index + 1, len(roster)):
                compare(roster[first_index], roster[second_index])

        league_scores = {}
        for candidate in roster:
            wins = 0
            losses = 0
            for opponent in roster:
                if opponent.fingerprint == candidate.fingerprint:
                    continue
                result = compare(candidate, opponent)
                if is_significant_win(
                    result,
                    require_significant=require_significant,
                    require_positive_pt=require_positive_pt,
                ):
                    wins += 1
                elif is_significant_win(
                    compare(opponent, candidate),
                    require_significant=require_significant,
                    require_positive_pt=require_positive_pt,
                ):
                    losses += 1
            league_scores[candidate.fingerprint] = {
                'wins': wins,
                'losses': losses,
            }
        roster.sort(
            key=lambda candidate: (
                -league_scores[candidate.fingerprint]['wins'],
                league_scores[candidate.fingerprint]['losses'],
                screen_by_fingerprint[candidate.fingerprint].avg_rank,
                -screen_by_fingerprint[candidate.fingerprint].avg_pt,
            )
        )

        direct_results = {
            candidate.fingerprint: self._aggregate_direct_results(
                candidate,
                comparison_cache.values(),
            )
            for candidate in roster
        }
        published = self._publish_top_models(
            roster,
            direct_results,
            screen_by_fingerprint,
            cycle_id=run_dir.name,
        )
        roster_payload = _load_json(self.models_dir / 'roster.json', {})
        roster_entries = roster_payload.get('models', [])
        winner = roster[0]
        winner_result = direct_results[winner.fingerprint]
        winner_path = published[winner.fingerprint]

        report = {
            'selection': MODEL_SELECTION_VERSION,
            'reference_model': str(self.reference_model),
            'reference_fingerprint': reference.fingerprint,
            'screen_seed': screen_seed,
            'direct_seed': direct_seed,
            'direct_initial_games_per_direction': direct_initial_games,
            'direct_initial_games': direct_initial_games * 2,
            'direct_max_games_per_direction': direct_max_games,
            'direct_max_games': direct_max_games * 2,
            'screen': [
                _result_dict(result)
                for result in sorted(screen_results, key=lambda item: item.avg_rank)
            ],
            'bootstrap_roster': bootstrap,
            'challengers': [candidate.model_id for candidate in challengers],
            'promotions': promotions,
            'league_scores': league_scores,
            'direct_comparisons': [
                _direct_comparison_dict(comparison)
                for _, comparison in sorted(comparison_cache.items())
            ],
            'top_models': roster_entries,
            'winner': {
                **_result_dict(winner_result),
                'path': str(winner_path),
            },
        }
        _atomic_json(eval_dir / 'report.json', report)
        self.manifest['model_roster'] = roster_entries
        self.manifest['champion'] = {
            **_result_dict(winner_result),
            'path': str(winner_path),
            'selection': MODEL_SELECTION_VERSION,
            'slot': 1,
        }
        winner_info = inspect_checkpoint(winner.path, winner.model_id)
        if winner_info.full_training_state:
            self.manifest['training_seed_model'] = str(winner.path)
        self._save_manifest()
        return {
            'candidate_count': len(eligible),
            'challenger_count': len(challengers),
            'direct_comparison_count': len(comparison_cache),
            'retained_model_count': len(roster),
            'winner': winner.model_id,
            'winner_model': str(winner_path),
            'winner_avg_rank': winner_result.avg_rank,
            'winner_avg_pt': winner_result.avg_pt,
            'report': str(eval_dir / 'report.json'),
        }

    def _roster_fingerprints(self) -> list[str]:
        payload = _load_json(self.models_dir / 'roster.json', {})
        entries = payload.get('models', self.manifest.get('model_roster', []))
        ordered = sorted(entries, key=lambda item: int(item.get('slot', 999)))
        fingerprints = [
            str(entry['fingerprint'])
            for entry in ordered
            if entry.get('fingerprint')
        ]
        return fingerprints[:3] if len(set(fingerprints[:3])) == 3 else []

    def _evaluate_direct_pair(
        self,
        first: Candidate,
        second: Candidate,
        games: int,
        seed: int,
        root: Path,
        timeout_hours: float,
    ) -> DirectComparison:
        model_a, model_b = sorted(
            (first, second),
            key=lambda candidate: candidate.fingerprint,
        )
        pair_seed = _seed_from_text(
            f'{seed}:{model_a.fingerprint}:{model_b.fingerprint}'
        )
        pair_root = root / (
            f'{model_a.fingerprint[:12]}__{model_b.fingerprint[:12]}'
        )
        a_vs_b = self._evaluate_candidate(
            model_a,
            model_b,
            games,
            pair_seed,
            pair_root / 'a-vs-b',
            timeout_hours,
        )
        b_vs_a = self._evaluate_candidate(
            model_b,
            model_a,
            games,
            pair_seed,
            pair_root / 'b-vs-a',
            timeout_hours,
        )
        comparison = DirectComparison(
            model_a=model_a,
            model_b=model_b,
            a_vs_b=a_vs_b,
            b_vs_a=b_vs_a,
            combined_a=combine_bidirectional_results(a_vs_b, b_vs_a),
        )
        evaluation_settings = self.config.get('evaluation', {})
        max_games = int(evaluation_settings.get('validation_games', games))
        if max_games < games:
            raise ValueError(
                'evaluation.validation_games cannot be below the initial game count'
            )
        require_significant = bool(
            evaluation_settings.get('require_significant', True)
        )
        require_positive_pt = bool(
            evaluation_settings.get('require_positive_pt', True)
        )
        challenger_lost = is_significant_win(
            comparison.result_for(second),
            require_significant=require_significant,
            require_positive_pt=require_positive_pt,
        )
        if max_games > games and not challenger_lost:
            extension_games = max_games - games
            extension_seed = _seed_from_text(
                f'{pair_seed}:extension:{extension_games}'
            )
            extension_root = pair_root / 'extension'
            a_extension = self._evaluate_candidate(
                model_a,
                model_b,
                extension_games,
                extension_seed,
                extension_root / 'a-vs-b',
                timeout_hours,
            )
            b_extension = self._evaluate_candidate(
                model_b,
                model_a,
                extension_games,
                extension_seed,
                extension_root / 'b-vs-a',
                timeout_hours,
            )
            a_vs_b = combine_evaluation_batches(a_vs_b, a_extension)
            b_vs_a = combine_evaluation_batches(b_vs_a, b_extension)
            comparison = DirectComparison(
                model_a=model_a,
                model_b=model_b,
                a_vs_b=a_vs_b,
                b_vs_a=b_vs_a,
                combined_a=combine_bidirectional_results(a_vs_b, b_vs_a),
            )
        return comparison

    @staticmethod
    def _aggregate_direct_results(
        candidate: Candidate,
        comparisons: Iterable[DirectComparison],
    ) -> EvalResult:
        rankings = [0] * NUM_PLAYERS
        for comparison in comparisons:
            if candidate.fingerprint not in {
                comparison.model_a.fingerprint,
                comparison.model_b.fingerprint,
            }:
                continue
            result = comparison.result_for(candidate)
            rankings = [
                total + count for total, count in zip(rankings, result.rankings)
            ]
        if sum(rankings) <= 1:
            raise RuntimeError(
                f'top model has no direct comparison evidence: {candidate.model_id}'
            )
        return EvalResult.from_rankings(
            candidate.model_id,
            candidate.path,
            rankings,
            candidate.step,
            candidate.fingerprint,
        )

    def _evaluate_candidate(
        self,
        candidate: Candidate,
        reference: Candidate,
        games: int,
        seed: int,
        root: Path,
        timeout_hours: float,
    ) -> EvalResult:
        model_id = _safe_id(candidate.model_id)
        log_dir = root / model_id
        actual_games = len(list(log_dir.glob(ARENA_LOG_GLOB))) if log_dir.exists() else 0
        if log_dir.exists() and actual_games != games:
            if not log_dir.resolve().is_relative_to(root.resolve()):
                raise RuntimeError(
                    f'refusing to clean evaluation path outside run: {log_dir}'
                )
            shutil.rmtree(log_dir)
        if not log_dir.exists():
            cfg = toml.load(self.base_config_file)
            arena = cfg[ARENA_CONFIG_SECTION]
            arena['seed_key'] = seed
            arena['games_per_iter'] = games
            arena['iters'] = 1
            arena['log_dir'] = str(log_dir)
            arena['challenger'].update({
                'name': model_id,
                'state_file': str(candidate.path),
            })
            if NUM_PLAYERS == 4:
                arena['champion'].update({
                    'name': 'reference-baseline',
                    'state_file': str(reference.path),
                })
                arena['akochan']['enabled'] = False
            else:
                arena['champion'].update({
                    'name': f'opponent-{_safe_id(reference.model_id)}',
                    'state_file': str(reference.path),
                })
            config_file = root / f'{model_id}.toml'
            _write_toml(config_file, cfg)
            self._run_logged(
                [sys.executable, str(self.mortal_root / 'mortal' / ARENA_SCRIPT)],
                config_file,
                root / f'{model_id}.log',
                timeout_hours,
            )
        return self._read_evaluation(candidate, model_id, log_dir, games)

    @staticmethod
    def _read_evaluation(
        candidate: Candidate,
        engine_name: str,
        log_dir: Path,
        expected_games: int,
    ) -> EvalResult:
        from libriichi.stat import Stat

        actual_games = len(list(log_dir.glob(ARENA_LOG_GLOB)))
        if actual_games != expected_games:
            raise RuntimeError(
                f'evaluation produced {actual_games} logs, '
                f'expected {expected_games}: {log_dir}'
            )
        stat = Stat.from_dir(str(log_dir), engine_name)
        rates = (
            stat.rank_1_rate,
            stat.rank_2_rate,
            stat.rank_3_rate,
            stat.rank_4_rate,
        )[:NUM_PLAYERS]
        counts = [round(rate * actual_games) for rate in rates]
        counts[-1] += actual_games - sum(counts)
        return EvalResult.from_rankings(
            candidate.model_id,
            candidate.path,
            counts,
            candidate.step,
            candidate.fingerprint,
        )

    def _publish_top_models(
        self,
        roster: Sequence[Candidate],
        direct_results: Mapping[str, EvalResult],
        anchor_results: Mapping[str, EvalResult],
        *,
        cycle_id: str,
    ) -> dict[str, Path]:
        import torch

        if len(roster) != 3:
            raise ValueError(f'exactly three top models are required, got {len(roster)}')
        if len({candidate.fingerprint for candidate in roster}) != 3:
            raise ValueError('top model roster contains duplicate fingerprints')

        staging = self.output_root / f'.models-{_safe_id(cycle_id)}.next'
        backup = self.output_root / '.models.previous'
        output_root = self.output_root.resolve()
        for path in (staging, backup):
            if path.resolve().parent != output_root:
                raise RuntimeError(f'unsafe model publication path: {path}')
            if path.exists():
                shutil.rmtree(path)
        staging.mkdir(parents=True)

        published = {}
        entries = []
        for slot, candidate in enumerate(roster, start=1):
            direct = direct_results.get(candidate.fingerprint)
            anchor = anchor_results.get(candidate.fingerprint)
            if direct is None or anchor is None:
                raise RuntimeError(
                    f'missing evaluation metadata for top model {candidate.model_id}'
                )
            filename = top_model_filename(slot, direct)
            staged_model = staging / filename
            final_model = self.models_dir / filename
            state = torch.load(candidate.path, weights_only=True, map_location='cpu')
            temp_model = staged_model.with_suffix(staged_model.suffix + '.tmp')
            torch.save(compact_checkpoint_state(state), temp_model)
            temp_model.replace(staged_model)
            entry = {
                'slot': slot,
                'model_id': candidate.model_id,
                'path': str(final_model),
                'source_path': str(candidate.path),
                'step': candidate.step,
                'fingerprint': candidate.fingerprint,
                'selection': MODEL_SELECTION_VERSION,
                'direct': _result_dict(direct),
                'anchor': _result_dict(anchor),
            }
            entries.append(entry)
            _atomic_json(staged_model.with_suffix('.json'), entry)
            published[candidate.fingerprint] = final_model

        _atomic_json(
            staging / 'roster.json',
            {
                'version': 2,
                'selection': MODEL_SELECTION_VERSION,
                'cycle_id': cycle_id,
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'models': entries,
            },
        )

        had_previous = self.models_dir.exists()
        if had_previous:
            self.models_dir.replace(backup)
        try:
            staging.replace(self.models_dir)
        except Exception:
            if had_previous and backup.exists() and not self.models_dir.exists():
                backup.replace(self.models_dir)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
        return published

    def _retain_validated_models(self, results: Sequence[EvalResult]) -> dict[str, Path]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        retained = {}
        for result in results:
            destination = self.models_dir / ability_filename(result)
            if not destination.exists():
                shutil.copy2(result.path, destination)
            _atomic_json(destination.with_suffix('.json'), _result_dict(result))
            retained[result.model_id] = destination
        return retained

    def _run_logged(
        self,
        command: Sequence[str],
        config_file: Path,
        log_file: Path,
        timeout_hours: float,
    ) -> None:
        self._wait_for_gpu_slot()
        env = os.environ.copy()
        env['MORTAL_CFG'] = str(config_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open('ab') as output:
            completed = subprocess.run(
                list(command),
                cwd=self.mortal_root,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout_hours * 3600,
                check=False,
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _parse_stages(value: str) -> tuple[str, ...]:
    if value.strip().lower() == 'all':
        return AutomationPipeline.STAGES
    return tuple(part.strip().lower() for part in value.split(',') if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Incremental Mortal GRP, offline, online self-play, and evaluation pipeline.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'automation.toml',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--stages',
        default='all',
        help='all or a comma-separated subset of discover,grp,offline,online,evaluate (parent is yonma-only)',
    )
    args = parser.parse_args(argv)
    pipeline = AutomationPipeline(
        args.config,
        dry_run=args.dry_run,
        force=args.force,
        stages=_parse_stages(args.stages),
    )
    return pipeline.run()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        raise SystemExit(130)
