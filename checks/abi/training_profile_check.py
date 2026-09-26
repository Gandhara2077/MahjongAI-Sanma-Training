"""Check the config-driven variant and base-model resolution contract."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "training-profile.toml"
RESOLVER = ROOT / "scripts" / "resolve_profile.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--generic", action="store_true")
    args = parser.parse_args()
    completed = subprocess.run(
        [
            sys.executable,
            str(RESOLVER),
            "--profile",
            str(args.profile),
            "--repo-root",
            str(args.repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip() or "profile resolver failed")

    resolved = json.loads(completed.stdout)
    with Path(resolved["automation_config"]).open("rb") as file:
        automation = tomllib.load(file)
    configured_base = (
        Path(resolved["automation_config"]).parent
        / automation["paths"]["base_config"]
    ).resolve()
    assert configured_base == Path(resolved["training_config"]).resolve(), (
        f"automation base config mismatch: {configured_base} != "
        f"{resolved['training_config']}"
    )
    if args.generic:
        assert resolved["variant"] in {"sanma", "yonma"}
        assert resolved["players"] in {3, 4}
        assert resolved["native_variant"] in {"sanma", "yonma"}
        assert Path(resolved["training_config"]).is_file()
        assert Path(resolved["automation_config"]).is_file()
        assert resolved["base_model"]
    else:
        assert resolved["variant"] == "sanma"
        assert resolved["players"] == 3
        assert resolved["training_config"] == str(
            ROOT / "Mortal" / "config" / "sanma-training.toml"
        )
        assert resolved["automation_config"] == str(
            ROOT / "Mortal" / "automation" / "sanma.toml"
        )
        assert resolved["base_model"] == str(ROOT / "baselines" / "sanma_baseline_mortal3p_original.pth")
        assert resolved["native_variant"] == "sanma"
        assert resolved["raw_root"] == str(ROOT / "koromo" / "mjlog")
        assert resolved["mjson_root"] == str(ROOT / "koromo" / "mjson")
        assert resolved["start_date"] == "20260101"
        assert resolved["end_date"] == "20260807"
        assert resolved["download_workers"] == 12

    with tempfile.TemporaryDirectory() as temporary:
        alternate = Path(temporary) / "profile.toml"
        alternate.write_text(
            """[active]
variant = 'yonma'

[variants.yonma]
players = 4
training_config = 'Mortal/config/yonma.toml'
automation_config = 'Mortal/automation/yonma.toml'

[variants.yonma.data]
download_root = 'koromo4p'
raw_root = 'koromo4p/mjlog'
mjson_root = 'koromo4p/mjson'
manifest_root = 'data/manifests'
start_date = '20260101'
end_date = '20260807'
delay = 0.5
download_workers = 4
""",
            encoding="utf-8",
        )
        alternate_run = subprocess.run(
            [
                sys.executable,
                str(RESOLVER),
                "--profile",
                str(alternate),
                "--repo-root",
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert alternate_run.returncode == 0, alternate_run.stderr
        alternate_resolved = json.loads(alternate_run.stdout)
        assert alternate_resolved["variant"] == "yonma"
        assert alternate_resolved["players"] == 4
        assert alternate_resolved["native_variant"] == "yonma"
        assert alternate_resolved["download_workers"] == 4

    print("TRAINING_PROFILE_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, json.JSONDecodeError, OSError, KeyError) as error:
        print(f"TRAINING_PROFILE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
