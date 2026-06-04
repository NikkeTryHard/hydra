from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def shanten_bridge_library_candidates() -> list[Path]:
    env_path = os.environ.get("HYDRA_RAW_MJAI_PYO3_LIB")
    if env_path:
        return [Path(env_path)]
    repo_root = Path(__file__).resolve().parents[4]
    return [
        repo_root / "target" / "release" / "libhydra_raw_mjai_pyo3.so",
        repo_root / "target" / "debug" / "libhydra_raw_mjai_pyo3.so",
    ]


def default_shanten_bridge_library_path() -> Path:
    for path in shanten_bridge_library_candidates():
        if path.exists():
            return path
    return shanten_bridge_library_candidates()[-1]


def _load_shanten_bridge_module(path: Path) -> Any:
    if not path.exists():
        raise ImportError(f"hydra_raw_mjai_pyo3 extension not found at {path}")
    spec = importlib.util.spec_from_file_location("hydra_raw_mjai_pyo3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load hydra_raw_mjai_pyo3 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hydra_raw_mjai_pyo3"] = module
    spec.loader.exec_module(module)
    return module


def _load_shanten_bridge_with_function(library_path: Path | None) -> Any:
    last_error: ImportError | None = None
    candidates = [library_path] if library_path is not None else shanten_bridge_library_candidates()
    for path in candidates:
        try:
            module = _load_shanten_bridge_module(path)
        except ImportError as exc:
            last_error = exc
            continue
        if hasattr(module, "batch_discard_shanten_masks"):
            return module
    if last_error is not None:
        raise last_error
    raise ImportError("hydra_raw_mjai_pyo3 extension lacks batch_discard_shanten_masks; rebuild hydra-raw-mjai-pyo3")


def exact_discard_shanten_masks(
    counts: list[int], library_path: Path | None = None
) -> tuple[int, list[bool], list[bool]]:
    """Return Hydra Rust batch-discard shanten masks for a 34-wide hand count vector."""
    module = _load_shanten_bridge_with_function(library_path)
    base, non_increase, decrease = module.batch_discard_shanten_masks(counts)
    return int(base), list(non_increase), list(decrease)


def exact_shanten_mask_planes(counts: list[int], library_path: Path | None = None) -> tuple[list[float], list[float]]:
    """Return channel-9/10 diagnostic planes from the exact Rust shanten bridge."""
    _, non_increase, decrease = exact_discard_shanten_masks(counts, library_path)
    return ([1.0 if value else 0.0 for value in non_increase], [1.0 if value else 0.0 for value in decrease])


def has_shanten_bridge(library_path: Path | None = None) -> bool:
    try:
        _load_shanten_bridge_with_function(library_path)
    except ImportError:
        return False
    return True
