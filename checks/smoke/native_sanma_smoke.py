"""Smoke-test the compiled sanma libriichi extension."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "Mortal" / "native" / "sanma" / "libriichi-py312.pyd"


def load_extension():
    if not EXTENSION.is_file():
        raise AssertionError(f"missing sanma extension: {EXTENSION}")
    loader = importlib.machinery.ExtensionFileLoader("libriichi", str(EXTENSION))
    spec = importlib.util.spec_from_loader("libriichi", loader)
    if spec is None:
        raise AssertionError(f"cannot create module spec: {EXTENSION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["libriichi"] = module
    loader.exec_module(module)
    return module


def main() -> None:
    native = load_extension()
    assert native.consts.NUM_PLAYERS == 3
    assert native.consts.MAX_VERSION == 5
    assert native.consts.ACTION_SPACE == 44
    assert tuple(native.consts.obs_shape(4)) == (775, 34)
    assert tuple(native.consts.obs_shape(5)) == (780, 34)
    print("NATIVE_SANMA_OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, OSError, AttributeError) as error:
        print(f"NATIVE_SANMA_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
