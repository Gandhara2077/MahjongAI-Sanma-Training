"""Check that a configured longer run extends a saved scheduler horizon."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Mortal" / "mortal"))


def main() -> None:
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from online_training import restore_scheduler_state

    old_optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1.0)
    old_scheduler = LinearWarmUpCosineAnnealingLR(
        old_optimizer,
        peak=1e-6,
        final=3e-7,
        warm_up_steps=1,
        max_steps=10,
    )
    old_optimizer.step()
    old_scheduler.step()
    saved = old_scheduler.state_dict()

    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        peak=1e-6,
        final=3e-7,
        warm_up_steps=1,
        max_steps=20,
    )
    restore_scheduler_state(
        scheduler,
        saved,
        {"max_steps": 20, "peak": 1e-6, "final": 3e-7, "warm_up_steps": 1},
    )
    assert scheduler.max_steps == 20
    assert scheduler.last_epoch == saved["last_epoch"]
    print("SCHEDULER_RESUME_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, KeyError, RuntimeError) as error:
        print(f"SCHEDULER_RESUME_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
