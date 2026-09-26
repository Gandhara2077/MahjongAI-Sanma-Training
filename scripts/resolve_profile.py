"""Resolve the active Mortal training profile without project dependencies."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def _path(value: str, base: Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _required(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def resolve_profile(
    profile_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    profile = Path(profile_path).resolve()
    root = Path(repo_root).resolve() if repo_root else profile.parent.parent
    values = _load_toml(profile)
    active = values.get("active", {})
    variant = _required(active, "variant", "active")
    variants = values.get("variants", {})
    selected = variants.get(variant)
    if not isinstance(selected, dict):
        raise ValueError(f"active.variant is not configured: {variant}")

    players = selected.get("players")
    if players not in (3, 4):
        raise ValueError(f"variants.{variant}.players must be 3 or 4")

    training_config = _path(
        _required(selected, "training_config", f"variants.{variant}"),
        root,
    )
    automation_config = _path(
        _required(selected, "automation_config", f"variants.{variant}"),
        root,
    )
    for label, path in (
        ("training_config", training_config),
        ("automation_config", automation_config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    training = _load_toml(training_config)
    control = training.get("control", {})
    base_model_value = _required(control, "init_from", "control")
    mortal_root = training_config.parent.parent
    base_model = _path(base_model_value, mortal_root)

    automation = _load_toml(automation_config)
    automation_paths = automation.get("paths", {})
    data = selected.get("data", {})
    if not isinstance(data, dict):
        raise ValueError(f"variants.{variant}.data must be a table")
    download_workers = data.get("download_workers", 4)
    if (
        isinstance(download_workers, bool)
        or not isinstance(download_workers, int)
        or download_workers < 1
    ):
        raise ValueError(
            f"variants.{variant}.data.download_workers must be a positive integer"
        )

    def data_path(key: str) -> str:
        value = _required(data, key, f"variants.{variant}.data")
        return str(_path(value, root))

    result = {
        "profile": str(profile),
        "repo_root": str(root),
        "variant": variant,
        "native_variant": str(selected.get("native_variant", variant)),
        "players": players,
        "training_config": str(training_config),
        "automation_config": str(automation_config),
        "mortal_root": str(mortal_root),
        "base_model": str(base_model),
        "state_file": str(_path(control.get("state_file", ""), mortal_root))
        if control.get("state_file")
        else "",
        "deployment_file": str(
            _path(control.get("deployment_file", ""), mortal_root)
        )
        if control.get("deployment_file")
        else "",
        "grp_state_file": str(
            _path(training.get("grp", {}).get("state_file", ""), mortal_root)
        )
        if training.get("grp", {}).get("state_file")
        else "",
        "logs_root": str(_path(automation_paths.get("logs_root", ""), automation_config.parent))
        if automation_paths.get("logs_root")
        else "",
        "download_root": data_path("download_root"),
        "raw_root": data_path("raw_root"),
        "mjson_root": data_path("mjson_root"),
        "manifest_root": data_path("manifest_root"),
        "start_date": _required(data, "start_date", f"variants.{variant}.data"),
        "end_date": _required(data, "end_date", f"variants.{variant}.data"),
        "delay": float(data.get("delay", 0.5)),
        "download_workers": download_workers,
    }
    if result["delay"] < 0:
        raise ValueError(f"variants.{variant}.data.delay must be non-negative")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "training-profile.toml",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            resolve_profile(args.profile, repo_root=args.repo_root),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"PROFILE_RESOLVE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
