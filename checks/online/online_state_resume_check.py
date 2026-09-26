"""Check that online phase resume numbering cannot silently over-train.

Regression guard for the online state-restore step bug: a state file seeded
from an earlier offline snapshot must rebase the phase window onto its actual
step, while a genuine mid-phase resume keeps the phase numbering.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Mortal" / "mortal"))


def expect(name, got, want):
    assert got == want, f"{name}: got {got!r}, want {want!r}"


def main() -> None:
    from online_training import resolve_online_resume_numbering

    # 1. State seeded from an EARLIER offline snapshot (step 13,000) while the
    #    phase window is [46,500, 56,500): must rebase onto the state's step,
    #    keeping the 10,000-update budget. Before the fix this trained from
    #    step 13,000 all the way to 56,500 (43,500 steps of silent
    #    over-training).
    start, target, rebased = resolve_online_resume_numbering(13_000, 46_500, 56_500)
    expect("rebased start", start, 13_000)
    expect("rebased target", target, 23_000)
    expect("rebased flag", rebased, True)

    # 2. Genuine mid-phase resume: state at 50,000 inside [46,500, 56,500)
    #    keeps the phase numbering.
    start, target, rebased = resolve_online_resume_numbering(50_000, 46_500, 56_500)
    expect("midphase start", start, 46_500)
    expect("midphase target", target, 56_500)
    expect("midphase flag", rebased, False)

    # 3. State exactly at the phase start (fresh seed already copied) keeps
    #    the window and is not treated as a rebase.
    start, target, rebased = resolve_online_resume_numbering(46_500, 46_500, 56_500)
    expect("at-start start", start, 46_500)
    expect("at-start target", target, 56_500)
    expect("at-start flag", rebased, False)

    # 4. State at/after the target means the caller sees a complete phase;
    #    the resolver must not shrink the window for such states.
    start, target, rebased = resolve_online_resume_numbering(56_500, 46_500, 56_500)
    expect("done start", start, 46_500)
    expect("done target", target, 56_500)
    expect("done flag", rebased, False)
    assert 56_500 >= target

    # 5. Zero-update budget is rejected rather than producing a degenerate
    #    window.
    try:
        resolve_online_resume_numbering(46_500, 56_500, 46_500)
    except ValueError:
        pass
    else:
        raise AssertionError("negative budget must raise ValueError")

    # 6. Caller-side completion check uses the resolved target, so a rebased
    #    state can never be reported as complete before its own window ends.
    current, phase_start, target_step = 13_000, 46_500, 56_500
    start, target, _ = resolve_online_resume_numbering(current, phase_start, target_step)
    assert not (current >= target), "stale state must not satisfy the new target"

    print("ONLINE_STATE_RESUME_OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ONLINE_STATE_RESUME_FAIL: {error}")
        raise SystemExit(1)
