"""Regression check for date-bounded corpus validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sanma_data_check import select_date_range


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = [
            "2026/06/30/outside.mjson",
            "2026/07/01/start.mjson",
            "2026/07/07/end.mjson",
            "2026/07/08/outside.mjson",
        ]
        paths = [root / name for name in names]
        selected = select_date_range(root, paths, "20260701", "20260707")
        assert [path.relative_to(root).as_posix() for path in selected] == names[1:3]
    print("DATA_RANGE_FILTER_OK")


if __name__ == "__main__":
    main()
