"""Run a bounded GPU training step from the configured sanma baseline."""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
_config_value = os.environ.get("MORTAL_CFG")
CONFIG = Path(_config_value) if _config_value else MORTAL / "config" / "sanma-training.toml"
if not CONFIG.is_absolute() and not CONFIG.exists():
    CONFIG = MORTAL / CONFIG
CONFIG = CONFIG.resolve()
sys.path.insert(0, str(MORTAL / "mortal"))


def assert_finite(state_dict) -> None:
    for name, value in state_dict.items():
        assert torch.isfinite(value).all(), f"non-finite tensor: {name}"


def main() -> None:
    os.chdir(MORTAL)
    os.environ["MORTAL_CFG"] = str(CONFIG)

    from checkpoint import initialize_from_baseline
    from config import config
    from libriichi import consts
    from libriichi.dataset import GameplayLoader
    from model import Brain, DQN

    assert consts.NUM_PLAYERS == 3
    version = int(config["control"]["version"])
    assert version == 4
    assert tuple(consts.obs_shape(version)) == (775, 34)
    assert consts.ACTION_SPACE == 44

    device = torch.device(config["control"]["device"])
    assert device.type == "cuda", "sanma training smoke requires a GPU"
    assert torch.cuda.is_available(), "ROCm GPU is unavailable"

    baseline_file = Path(config["control"]["init_from"]).resolve()
    assert baseline_file.is_file(), f"configured baseline is missing: {baseline_file}"
    brain = Brain(version=version, **config["resnet"])
    dqn = DQN(version=version)
    baseline = initialize_from_baseline(baseline_file, brain, dqn)
    assert baseline["config"]["control"]["version"] == version
    assert_finite(baseline["mortal"])
    assert_finite(baseline["current_dqn"])

    replay_files = sorted(
        {
            Path(filename).resolve()
            for pattern in config["dataset"]["globs"]
            for filename in glob(pattern, recursive=True)
        }
    )
    assert replay_files, "configured sanma replay corpus is empty"
    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=None,
        excludes=None,
        augmented=False,
        encode_observations=True,
    )
    games = loader.load_gz_log_files([str(replay_files[0])])
    assert len(games) == 1

    entries = []
    for game in games[0]:
        observations = game.take_obs()
        actions = game.take_actions()
        masks = game.take_masks()
        assert len(observations) == len(actions) == len(masks)
        for observation, mask, action in zip(observations, masks, actions):
            observation = torch.as_tensor(observation, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.bool)
            action = int(action)
            assert tuple(observation.shape) == (775, 34)
            assert tuple(mask.shape) == (44,)
            assert 0 <= action < 44
            assert bool(mask[action])
            assert torch.isfinite(observation).all()
            entries.append((observation, mask, action))
            if len(entries) == 8:
                break
        if len(entries) == 8:
            break
    assert entries, "configured replay has no sanma decision samples"

    observations = torch.stack([entry[0] for entry in entries]).to(device)
    masks = torch.stack([entry[1] for entry in entries]).to(device)
    actions = torch.tensor([entry[2] for entry in entries], dtype=torch.long, device=device)

    brain.to(device).train()
    dqn.to(device).train()
    parameters = list(brain.parameters()) + list(dqn.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["optim"]["scheduler"]["peak"]),
        betas=tuple(config["optim"]["betas"]),
        eps=float(config["optim"]["eps"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    amp_enabled = bool(config["control"].get("enable_amp", False))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, enabled=amp_enabled):
        q_values = dqn(brain(observations), masks)
        loss = torch.nn.functional.cross_entropy(q_values, actions)
    assert tuple(q_values.shape) == (len(entries), 44)
    assert torch.isfinite(q_values[masks]).all()
    assert torch.isneginf(q_values[~masks]).all()
    assert torch.isfinite(loss)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    assert gradients, "training smoke produced no gradients"
    for gradient in gradients:
        assert torch.isfinite(gradient).all(), "non-finite gradient"
    scaler.step(optimizer)
    scaler.update()
    assert_finite(brain.state_dict())
    assert_finite(dqn.state_dict())

    print(
        "SANMA_TRAINING_SMOKE_OK "
        f"device={torch.cuda.get_device_name(device)} "
        f"config={CONFIG.name} baseline={baseline_file.name} "
        f"replay={replay_files[0].name} batch={len(entries)} loss={loss.item():.6g}"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SANMA_TRAINING_SMOKE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
