"""Verify that sanma online training starts from the selected offline model."""

from __future__ import annotations

import sys
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = ROOT / "Mortal"
sys.path.insert(0, str(MORTAL_ROOT))
sys.path.insert(0, str(MORTAL_ROOT / "mortal"))

import automation_pipeline as pipeline_module  # noqa: E402


def main() -> None:
    with (ROOT / "Mortal" / "automation" / "sanma.toml").open("rb") as file:
        sanma_config = tomllib.load(file)
    selection_games = int(sanma_config["offline"]["selection_games"])
    assert selection_games > 0 and selection_games % 3 == 0
    assert int(sanma_config["offline"]["selection_max_candidates"]) >= 3

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        offline = root / "offline.pth"
        selected = root / "selected.pth"
        online = root / "online.pth"
        reference = root / "reference.pth"
        offline.write_text("offline", encoding="utf-8")
        selected.write_text("selected", encoding="utf-8")
        reference.write_text("reference", encoding="utf-8")

        calls: list[tuple[str, Path]] = []
        online_inspections = 0

        def fake_inspect(path: Path, model_id: str):
            nonlocal online_inspections
            if Path(path) == online:
                online_inspections += 1
            if Path(path) == online and online_inspections > 1:
                step = 13
            elif Path(path) == selected:
                step = 5
            else:
                step = 7
            candidate = pipeline_module.Candidate(
                model_id=model_id,
                path=Path(path),
                step=step,
                fingerprint=model_id,
            )
            return SimpleNamespace(candidate=candidate, full_training_state=True)

        def fake_select(run_dir: Path, offline_state: Path):
            calls.append(("select", offline_state))
            return (
                pipeline_module.Candidate(
                    model_id="selected",
                    path=selected,
                    step=5,
                    fingerprint="selected",
                ),
                {"selected": str(selected)},
            )

        captured: dict[str, Path | int] = {}

        def fake_build(base, **kwargs):
            captured["state_file"] = Path(kwargs["state_file"])
            captured["start_step"] = int(kwargs["start_step"])
            captured["target_step"] = int(kwargs["target_step"])
            return {
                "control": {"max_steps": 13},
                "online": {"remote": {}},
                "dataset": {},
                "train_play": {"default": {}},
            }

        runner = object.__new__(pipeline_module.AutomationPipeline)
        runner.config = {
            "online": {
                "save_every": 1,
                "snapshot_every": 1,
                "brain_freeze_updates": 1,
                "expert_ratio": 0.0,
                "peak_lr": 1e-7,
                "port": 5021,
                "games_per_session": 3,
                "host": "127.0.0.1",
                "file_batch_size": 1,
                "num_workers": 0,
                "repeats": 1,
                "timeout_hours": 1,
            }
        }
        runner.manifest = {}
        runner.reference_model = reference
        runner.logs_root = root
        runner._training_seed = lambda: reference
        runner._league_model_paths = lambda: []
        runner._base_config = lambda info: {"control": {}}
        runner._select_offline_candidate = fake_select
        runner._run_online_processes = lambda config_file, run_dir, settings: None

        with patch.object(pipeline_module, "inspect_checkpoint", side_effect=fake_inspect):
            with patch.object(pipeline_module, "build_online_config", side_effect=fake_build):
                result = runner._run_online_sanma(
                    root,
                    root / "dataset.pth",
                    root / "grp.pth",
                    [],
                    offline,
                    online,
                    7,
                    15,
                )

        assert calls == [("select", offline)]
        assert captured["state_file"] == online
        assert captured["start_step"] == 5
        assert captured["target_step"] == 13
        assert online.read_text(encoding="utf-8") == "selected"
        assert result["source_model"] == str(selected)
        assert result["start_step"] == 5
        assert result["target_step"] == 13
        assert result["offline_selection"] == {"selected": str(selected)}

    print("OFFLINE_SELECTION_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, RuntimeError) as error:
        print(f"OFFLINE_SELECTION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
