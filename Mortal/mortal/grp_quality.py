from __future__ import annotations

import math
import random
import statistics
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


QUALITY_METRICS = ("nll", "accuracy", "brier", "expected_pt_mse")


def _default_pts() -> list[float]:
    # fallback only; configs are expected to set env.pts explicitly
    from libriichi.consts import NUM_PLAYERS

    return [6.0, 4.0, 2.0, 0.0][:NUM_PLAYERS] if NUM_PLAYERS == 4 else [6.0, 3.0, 0.0]


def calculate_quality_metrics(
    probabilities,
    labels,
    predicted_pts,
    actual_pts,
) -> dict[str, float]:
    import torch
    from torch.nn import functional as F

    if probabilities.ndim != 2 or labels.ndim != 1:
        raise ValueError("GRP probabilities and labels have invalid dimensions")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("GRP probabilities and labels have different batch sizes")
    if predicted_pts.shape != actual_pts.shape:
        raise ValueError("predicted and actual GRP points must have matching shapes")

    one_hot = F.one_hot(
        labels,
        num_classes=probabilities.shape[-1],
    ).to(probabilities.dtype)
    selected = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    return {
        "nll": float(-selected.clamp_min(torch.finfo(selected.dtype).tiny).log().mean()),
        "accuracy": float((probabilities.argmax(-1) == labels).to(torch.float64).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(-1).mean()),
        "expected_pt_mse": float(((predicted_pts - actual_pts) ** 2).mean(-1).mean()),
    }


def paired_bootstrap_interval(
    deltas: list[float],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not deltas:
        raise ValueError("paired bootstrap needs at least one delta")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between 0 and 1")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    values = tuple(float(value) for value in deltas)
    randomizer = random.Random(seed)
    means = sorted(
        statistics.fmean(
            values[randomizer.randrange(len(values))]
            for _ in range(len(values))
        )
        for _ in range(samples)
    )

    def quantile(fraction: float) -> float:
        position = (len(means) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return means[lower]
        weight = position - lower
        return means[lower] * (1 - weight) + means[upper] * weight

    alpha = (1 - confidence) / 2
    return {"low": quantile(alpha), "high": quantile(1 - alpha)}


def build_quality_report(
    incumbent_games: list[Mapping[str, float]],
    candidate_games: list[Mapping[str, float]],
    *,
    sample_counts: list[int],
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not (
        len(incumbent_games) == len(candidate_games) == len(sample_counts)
    ):
        raise ValueError("GRP quality inputs must describe the same games")
    if not incumbent_games:
        raise ValueError("GRP quality evaluation needs at least one game")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("GRP quality sample counts must be positive")

    total_samples = sum(sample_counts)

    def aggregate(games: list[Mapping[str, float]]) -> dict[str, float]:
        return {
            metric: sum(
                float(game[metric]) * count
                for game, count in zip(games, sample_counts)
            ) / total_samples
            for metric in QUALITY_METRICS
        }

    incumbent = aggregate(incumbent_games)
    candidate = aggregate(candidate_games)
    deltas = {
        metric: candidate[metric] - incumbent[metric]
        for metric in QUALITY_METRICS
    }
    intervals = {}
    for index, metric in enumerate(QUALITY_METRICS):
        paired_deltas = [
            float(candidate_game[metric]) - float(incumbent_game[metric])
            for incumbent_game, candidate_game in zip(
                incumbent_games,
                candidate_games,
            )
        ]
        intervals[metric] = paired_bootstrap_interval(
            paired_deltas,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + index,
        )

    return {
        "dataset": {"logs": len(sample_counts), "samples": total_samples},
        "models": {"incumbent": incumbent, "candidate": candidate},
        "comparison": {
            "deltas": deltas,
            "confidence_intervals": intervals,
        },
        "bootstrap": {
            "confidence": confidence,
            "samples": bootstrap_samples,
            "seed": seed,
        },
    }


def decide_grp_promotion(
    report: Mapping[str, Any],
    *,
    primary_metric: str,
    require_significant: bool,
    min_relative_improvement: float,
    max_nll_regression: float,
    max_brier_regression: float,
    min_gate_logs: int,
) -> dict[str, Any]:
    models = report["models"]
    incumbent = models["incumbent"]
    candidate = models["candidate"]
    incumbent_primary = float(incumbent[primary_metric])
    candidate_primary = float(candidate[primary_metric])
    if incumbent_primary <= 0:
        raise ValueError("incumbent primary GRP metric must be positive")

    relative_improvement = (
        incumbent_primary - candidate_primary
    ) / incumbent_primary
    reasons: list[str] = []
    gate_logs = int(report["dataset"]["logs"])
    if gate_logs < min_gate_logs:
        reasons.append(f"gate set has only {gate_logs} logs; need {min_gate_logs}")
    if relative_improvement < min_relative_improvement:
        reasons.append("primary metric did not improve enough")
    if float(candidate["nll"]) > float(incumbent["nll"]) + max_nll_regression:
        reasons.append("NLL regressed")
    if (
        float(candidate["brier"])
        > float(incumbent["brier"]) + max_brier_regression
    ):
        reasons.append("Brier score regressed")
    interval = report["comparison"]["confidence_intervals"][primary_metric]
    if require_significant and float(interval["high"]) >= 0:
        reasons.append("primary metric improvement is not statistically significant")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "primary_metric": primary_metric,
        "relative_improvement": relative_improvement,
        "delta": candidate_primary - incumbent_primary,
        "confidence_interval": {
            "low": float(interval["low"]),
            "high": float(interval["high"]),
        },
    }


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_configured_checkpoints(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

    from libriichi.dataset import Grp
    from model import GRP

    grp_cfg = config["grp"]
    quality_cfg = grp_cfg["quality_gate"]
    index = torch.load(
        grp_cfg["dataset"]["file_index"],
        weights_only=True,
        map_location="cpu",
    )
    gate_files = list(index["gate_file_list"])
    if not gate_files:
        raise ValueError("GRP gate file list is empty")

    device = torch.device(grp_cfg["control"]["device"])
    checkpoints = {
        "incumbent": Path(quality_cfg["incumbent_state_file"]),
        "candidate": Path(quality_cfg["candidate_state_file"]),
    }
    models = {}
    states = {}
    for name, checkpoint in checkpoints.items():
        state = torch.load(checkpoint, weights_only=True, map_location=device)
        model = GRP(**grp_cfg["network"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        models[name] = model
        states[name] = state

    pts = torch.tensor(
        config.get("env", {}).get("pts") or _default_pts(),
        dtype=torch.float64,
        device=device,
    )
    file_batch_size = int(grp_cfg["dataset"].get("file_batch_size", 64))
    per_game = {"incumbent": [], "candidate": []}
    sample_counts: list[int] = []

    with torch.inference_mode():
        for start in range(0, len(gate_files), file_batch_size):
            file_batch = gate_files[start:start + file_batch_size]
            games = Grp.load_gz_log_files(file_batch)
            if len(games) != len(file_batch):
                raise RuntimeError("GRP gate loader returned an unexpected game count")
            for game in games:
                feature = game.take_feature()
                if feature.shape[0] == 0:
                    raise RuntimeError("GRP gate game has no feature sequence")
                sequences = [
                    torch.as_tensor(feature[:index + 1], dtype=torch.float64)
                    for index in range(feature.shape[0])
                ]
                lengths = torch.tensor([len(sequence) for sequence in sequences])
                padded = pad_sequence(sequences, batch_first=True)
                packed = pack_padded_sequence(
                    padded,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                ).to(device)
                final_ranks = torch.tensor(
                    [game.take_rank_by_player()] * len(sequences),
                    dtype=torch.int64,
                    device=device,
                )
                labels = models["incumbent"].get_label(final_ranks)
                actual_pts = F.one_hot(
                    final_ranks,
                    num_classes=len(pts),
                ).to(torch.float64) @ pts

                for name, model in models.items():
                    logits = model.forward_packed(packed)
                    probabilities = logits.softmax(-1)
                    predicted_pts = model.calc_matrix(logits) @ pts
                    per_game[name].append(
                        calculate_quality_metrics(
                            probabilities,
                            labels,
                            predicted_pts,
                            actual_pts,
                        )
                    )
                sample_counts.append(len(sequences))

    report = build_quality_report(
        per_game["incumbent"],
        per_game["candidate"],
        sample_counts=sample_counts,
        confidence=float(quality_cfg["confidence"]),
        bootstrap_samples=int(quality_cfg["bootstrap_samples"]),
        seed=int(quality_cfg["bootstrap_seed"]),
    )
    report["checkpoints"] = {
        name: {
            "path": str(path.resolve()),
            "step": int(states[name].get("steps", 0)),
            "sha256": _checkpoint_sha256(path),
        }
        for name, path in checkpoints.items()
    }
    return report


def main() -> int:
    import toml

    config_file = Path(os.environ.get("MORTAL_CFG", "config.toml"))
    config = toml.load(config_file)
    report = evaluate_configured_checkpoints(config)
    report_file = Path(config["grp"]["quality_gate"]["report_file"])
    report_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = report_file.with_suffix(report_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temp_file.replace(report_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
