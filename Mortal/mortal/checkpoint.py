from pathlib import Path

import torch


# Default obs-encoding version stamped into compact deployment checkpoints.
# 4 is the deployment ABI for both variants (yonma native v4, and the
# MahjongCopilot 775-channel ABI for sanma). Override per config with
# control.deployment_version when a deployment target needs another ABI.
DEPLOYMENT_VERSION = 4


def initialize_from_baseline(state_file, mortal, dqn, *, map_location="cpu"):
    state = torch.load(
        state_file,
        weights_only=True,
        map_location=map_location,
    )
    mortal.load_state_dict(state["mortal"], strict=True)
    dqn.load_state_dict(state["current_dqn"], strict=True)
    return state


def _cpu_state_dict(module):
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def deployment_state(mortal, dqn, config):
    resnet = config["resnet"]
    deployment_version = int(
        config.get("control", {}).get("deployment_version", DEPLOYMENT_VERSION)
    )
    compact_config = {
        "control": {"version": deployment_version},
        "resnet": {
            "conv_channels": int(resnet["conv_channels"]),
            "num_blocks": int(resnet["num_blocks"]),
        },
    }
    return {
        "config": compact_config,
        "mortal": _cpu_state_dict(mortal),
        "current_dqn": _cpu_state_dict(dqn),
    }


def export_deployment(output_file, mortal, dqn, config):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deployment_state(mortal, dqn, config), output_file)
