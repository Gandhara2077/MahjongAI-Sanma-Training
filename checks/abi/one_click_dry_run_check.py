"""Exercise the user-facing config-driven pipeline preview."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ALL = ROOT / "scripts" / "run_all.ps1"
BUILD_NATIVE = ROOT / "scripts" / "build_native.ps1"
FETCH_DATA = ROOT / "scripts" / "02_fetch_data.ps1"


def main() -> None:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUN_ALL),
            "-DryRun",
            "-SkipData",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip() or completed.stdout.strip())
    output = completed.stdout
    assert "variant=sanma" in output
    assert "sanma-training.toml" in output
    assert "STAGE offline" in output
    assert "sanma_baseline_mortal3p_original.pth" in output

    native = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_NATIVE),
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if native.returncode != 0:
        raise AssertionError(native.stderr.strip() or native.stdout.strip())
    assert "libriichi-py312.pyd" in native.stdout, native.stdout

    with tempfile.TemporaryDirectory() as temporary:
        profile = Path(temporary) / "yonma-profile.toml"
        profile.write_text(
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
        data_only = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(FETCH_DATA),
                "-ProfilePath",
                str(profile),
                "-DryRun",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if data_only.returncode != 0:
            raise AssertionError(data_only.stderr.strip() or data_only.stdout.strip())
        assert "--variant yonma" in data_only.stdout, data_only.stdout
        assert "--workers 4" in data_only.stdout, data_only.stdout
    print("ONE_CLICK_DRY_RUN_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError) as error:
        print(f"ONE_CLICK_DRY_RUN_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
