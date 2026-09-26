"""Check that the corpus validator accepts the active configured variant."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "checks" / "data" / "sanma_data_check.py"),
                "--variant",
                "sanma",
                "--raw-root",
                str(ROOT / "koromo" / "mjlog"),
                "--mjson-root",
                str(ROOT / "koromo" / "mjson"),
                "--start-date",
                "20260701",
                "--end-date",
                "20260707",
                "--manifest",
                str(Path(temporary) / "manifest.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip() or completed.stdout.strip())
    assert "SANMA_DATA_OK" in completed.stdout
    print("DATA_VARIANT_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError) as error:
        print(f"DATA_VARIANT_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
