"""Verify the orjson swap in libriichi3p_compat is semantically identical.

For a sample of real mjson games, every event line is checked for:
1. loads identity: orjson.loads(line) == json.loads(line)
2. dumps semantic identity: orjson-parsed output == stdlib-parsed output for
   _augment_event and adapt_event transforms
3. byte-level difference count (informational; formatting may differ as long
   as semantics match)
plus a stdlib-vs-orjson timing comparison of the augment path.

Usage: python checks/abi/orjson_compat_check.py [sample_size]
"""

import glob
import gzip
import json
import sys
import time

sys.path.insert(0, "Mortal/mortal")
import libriichi3p_compat as compat


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = sorted(glob.glob("koromo/mjson/2026/*/*/*.mjson"))
    if not files:
        print("no mjson files found")
        return 1
    stride = max(1, len(files) // sample_size)
    sample = files[::stride][:sample_size]

    std_loads = json.loads
    std_dumps = lambda obj: json.dumps(obj, separators=(",", ":"))

    n_events = 0
    loads_fail = 0
    sem_fail = 0
    byte_same = 0
    byte_diff = 0
    t_std = 0.0
    t_orj = 0.0

    for path in sample:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            raw = file.read()
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                e_std = std_loads(line)
                e_orj = compat.json_loads(line)
            except Exception:
                loads_fail += 1
                continue
            if e_std != e_orj:
                loads_fail += 1
                continue
            n_events += 1

            augmented = compat._augment_event(dict(e_std))
            s_std = std_dumps(augmented)
            s_orj = compat.json_dumps_compact(augmented)
            if std_loads(s_std) != compat.json_loads(s_orj):
                sem_fail += 1
                continue
            byte_same += 1 if s_std == s_orj else 0
            byte_diff += 0 if s_std == s_orj else 1

            adapted = compat.adapt_event(dict(e_std))
            r_std = std_dumps(adapted)
            r_orj = compat.json_dumps_compact(adapted)
            if std_loads(r_std) != compat.json_loads(r_orj):
                sem_fail += 1

    # timing on the augment path over the same sample
    for path in sample:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            raw = file.read()
        t0 = time.perf_counter()
        for line in raw.splitlines():
            if line.strip():
                std_dumps(compat._augment_event(std_loads(line)))
        t_std += time.perf_counter() - t0
        t0 = time.perf_counter()
        compat.augment_log(raw)
        t_orj += time.perf_counter() - t0

    speedup = t_std / t_orj if t_orj > 0 else float("inf")
    print(
        f"files={len(sample)} events={n_events} "
        f"loads_fail={loads_fail} sem_fail={sem_fail} "
        f"byte_same={byte_same} byte_diff={byte_diff}"
    )
    print(f"augment timing: stdlib {t_std:.2f}s vs orjson {t_orj:.2f}s -> {speedup:.2f}x")
    if sem_fail or loads_fail:
        print("FAIL: semantic mismatch detected")
        return 1
    print("ORJSON_COMPAT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
