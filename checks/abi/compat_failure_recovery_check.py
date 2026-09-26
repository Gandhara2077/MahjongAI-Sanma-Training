"""Regression check for isolating one incompatible sanma replay."""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
CONFIG = MORTAL / "config" / "sanma-training.toml"


class FakeGame:
    def take_obs(self):
        return [0]

    def take_actions(self):
        return [43]

    def take_masks(self):
        return [[True] * 44]

    def take_at_kyoku(self):
        return [0]

    def take_dones(self):
        return [True]

    def take_apply_gamma(self):
        return [1]


def main() -> None:
    os.chdir(MORTAL)
    os.environ["MORTAL_CFG"] = str(CONFIG)
    sys.path.insert(0, str(MORTAL / "mortal"))

    import libriichi3p_compat
    from dataloader import FileDatasetsIter

    assert libriichi3p_compat._can_recover_nukidora(
        SimpleNamespace(
            last_cans=SimpleNamespace(can_ron_agari=True),
        ),
        True,
    )
    assert libriichi3p_compat._can_recover_nukidora(
        SimpleNamespace(
            at_furiten=True,
            last_cans=SimpleNamespace(can_ron_agari=True),
        ),
        True,
    )

    assert libriichi3p_compat._should_skip_nukidora(
        {"type": "nukidora", "actor": 1}, 0
    )
    assert not libriichi3p_compat._should_skip_nukidora(
        {"type": "nukidora", "actor": 0}, 0
    )
    legacy_mask = [True] * 44
    native_mask = [True] * 44
    native_mask[40] = False
    adapted_masks = libriichi3p_compat._mask_nukidora_for_native_rules(
        [legacy_mask], native_mask
    )
    assert not adapted_masks[0][40]
    assert legacy_mask[40] is True

    converted = libriichi3p_compat._nukidora_ron_event(
        {"type": "nukidora", "actor": 2, "pai": "N"}
    )
    assert converted == {
        "type": "dahai",
        "actor": 2,
        "pai": "N",
        "tsumogiri": False,
    }
    converted_reference_event = json.loads(
        libriichi3p_compat.CompatMjaiEngine._event_json(
            {"type": "nukidora", "actor": 2, "pai": "N"},
            nukidora_as_dahai=True,
        )
    )
    assert converted_reference_event == {
        "type": "dahai",
        "actor": 2,
        "pai": "N",
        "tsumogiri": False,
    }
    assert json.loads(
        libriichi3p_compat.CompatMjaiEngine._event_json_for_bot(
            SimpleNamespace(state=SimpleNamespace(self_riichi_accepted=True)),
            {"type": "nukidora", "actor": 2, "pai": "N"},
            2,
        )
    ) == converted_reference_event
    assert json.loads(
        libriichi3p_compat.CompatMjaiEngine._event_json_for_bot(
            SimpleNamespace(state=SimpleNamespace(self_riichi_accepted=False)),
            {"type": "nukidora", "actor": 2, "pai": "N"},
            2,
        )
    )["type"] == "nukidora"

    with tempfile.TemporaryDirectory() as temp_dir:
        replay = Path(temp_dir) / "broken.mjson"
        with gzip.open(replay, "wt", encoding="utf-8") as stream:
            stream.write('{"type":"start_game","names":["a","b","c"]}\n')

        dataset = FileDatasetsIter(
            version=4,
            native_version=5,
            reference_path=CONFIG.parent.parent / ".cache" / "libriichi3p" / "libriichi3p-3.12-x86_64-pc-windows-msvc.pyd",
            file_list=[],
            pts=[6.0, 3.0, 0.0],
            player_names=[],
        )
        dataset.buffer = []
        original = libriichi3p_compat.capture_gameplay

        def fail_once(*args, **kwargs):
            raise libriichi3p_compat.CompatibilityError(
                "reference bot did not request decisions [(100, 43)] at event 537"
            )

        libriichi3p_compat.capture_gameplay = fail_once
        try:
            dataset._populate_buffer_compat(
                [str(replay)],
                [[FakeGame()]],
                augmented=False,
            )
        finally:
            libriichi3p_compat.capture_gameplay = original

    assert dataset.buffer == [], "incompatible replay must not add samples"
    print("COMPAT_FAILURE_RECOVERY_OK")


if __name__ == "__main__":
    main()
