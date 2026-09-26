"""Check the optimized sanma defaults before a long training run."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "Mortal" / "config" / "sanma-training.toml"
AUTOMATION = ROOT / "Mortal" / "automation" / "sanma.toml"


def main() -> None:
    with TRAINING.open("rb") as file:
        training = tomllib.load(file)
    with AUTOMATION.open("rb") as file:
        automation = tomllib.load(file)

    control = training["control"]
    dataset = training["dataset"]
    compat = training["compat"]
    assert control["batch_size"] == 1024
    assert control["test_every"] == 4000
    assert control["max_steps"] == 40000
    assert dataset["num_workers"] == 2
    assert compat["max_workers"] == 16
    assert training["1v2"]["games_per_iter"] == 600
    assert training["grp"]["control"]["max_steps"] >= 10000
    assert all("/08/" not in pattern.replace("\\", "/") for pattern in dataset["globs"])
    assert {"train", "test"} <= set(training["baseline"])
    assert "default" in training["train_play"]
    assert "server" in training["online"]

    offline = automation["offline"]
    assert automation["pipeline"].get("estimated_samples_per_log") == 380
    assert offline["num_workers"] == dataset["num_workers"]
    assert offline["file_batch_size"] == dataset["file_batch_size"]
    assert offline["num_epochs"] == dataset["num_epochs"]
    assert offline["deterministic"] is False
    print("PERFORMANCE_PROFILE_OK")


if __name__ == "__main__":
    main()
