"""Small, dependency-free check for the local sanma asset manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "sanma-assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise AssertionError(f"missing asset manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    assert manifest["variant"] == "sanma"
    assert manifest["reference"]["python"] == "3.12"
    assert manifest["abi"] == {"version": 4, "obs_shape": [775, 34], "action_space": 44}

    for item in (manifest["model"], manifest["reference_extension"]):
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == sha256(path)

    print("SANMA_SETUP_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, json.JSONDecodeError) as error:
        print(f"SANMA_SETUP_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
