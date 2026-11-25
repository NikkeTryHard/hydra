"""Engine selection: the vendored fork, with the installed pin as the oracle.

Hydra3 vendors the pinned MahJax source under ``vendor/mahjax`` so engine work can
be done in-tree. The path is prepended to ``sys.path`` before the engine is first
imported, which means a fork import wins over the installed distribution without
touching the Pixi lock.

Set ``HYDRA3_ENGINE=installed`` to import the installed pin instead. The parity
test runs both engines in separate subprocesses and compares full state digests,
so a fork change either reproduces the pin exactly or the test fails.
"""

import os
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "mahjax"
ENGINE_ENV = "HYDRA3_ENGINE"


def select() -> str:
    """Prepend the vendored engine unless the caller asked for the installed pin."""
    choice = os.environ.get(ENGINE_ENV, "vendor")
    if choice not in ("vendor", "installed"):
        raise ValueError(f"{ENGINE_ENV} must be 'vendor' or 'installed', got {choice!r}")
    if choice == "installed":
        return choice
    root = str(VENDOR_ROOT)
    if not (VENDOR_ROOT / "mahjax" / "__init__.py").is_file():
        raise RuntimeError(f"vendored engine missing at {VENDOR_ROOT}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return choice


def loaded_from() -> str:
    """Return the file the engine was actually imported from."""
    import mahjax

    return str(Path(mahjax.__file__).resolve())
