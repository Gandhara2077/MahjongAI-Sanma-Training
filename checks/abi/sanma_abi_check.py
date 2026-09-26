"""Validate the pinned 3p deployment checkpoint and reference ABI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
sys.path.insert(0, str(MORTAL / "mortal"))
os.environ.setdefault(
    "MORTAL_LIBRIICHI3P",
    str(ROOT / ".cache" / "libriichi3p"),
)


def main() -> None:
    from libriichi3p_compat import load_reference

    reference = load_reference()
    assert reference.consts.MAX_VERSION == 4
    assert reference.consts.ACTION_SPACE == 44
    assert tuple(reference.consts.obs_shape(4)) == (775, 34)

    state = torch.load(
        ROOT / "baselines" / "sanma_baseline_mortal3p_original.pth",
        map_location="cpu",
        weights_only=True,
    )
    assert state["config"]["control"]["version"] == 4
    assert state["config"]["resnet"] == {"conv_channels": 32, "num_blocks": 2}
    assert tuple(state["mortal"]["encoder.net.0.weight"].shape) == (32, 775, 3)
    assert tuple(state["current_dqn"]["net.weight"].shape) == (45, 1024)

    native = ROOT / "Mortal" / "native" / "sanma" / "libriichi-py312.pyd"
    assert native.is_file(), f"missing active sanma native extension: {native}"

    print("SANMA_ABI_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, OSError, KeyError) as error:
        print(f"SANMA_ABI_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
