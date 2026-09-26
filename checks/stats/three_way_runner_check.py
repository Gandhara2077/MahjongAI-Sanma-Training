from pathlib import Path
import os
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
os.environ["MORTAL_CFG"] = str(MORTAL / "config" / "sanma-three-way-eval.toml")
sys.path.insert(0, str(MORTAL / "mortal"))

import three_way  # noqa: E402


def expect(error_type, func, *args):
    try:
        func(*args)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


assert three_way.require_positive_int("games_per_iter", 600) == 600
for invalid in (True, 600.5, 0, -1):
    expect(ValueError, three_way.require_positive_int, "games_per_iter", invalid)

assert three_way.decode_rankings([b"\x02\x01\x00"]).tolist() == [[2, 1, 0]]

with tempfile.TemporaryDirectory() as raw_dir:
    log_dir = Path(raw_dir)
    assert three_way.prepare_log_dir(log_dir) == log_dir
    (log_dir / "old.json.gz").write_bytes(b"stale")
    expect(FileExistsError, three_way.prepare_log_dir, log_dir)

print("THREE_WAY_RUNNER_CHECK_OK")
