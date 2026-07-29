"""Shared environment-probe helpers used by config-check, runtime-probe,
and the test suites. Each helper returns evidence strings; failures are
reported, never swallowed.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from hydra2.config import TRAINER_FORBIDDEN_PACKAGES

if TYPE_CHECKING:
    from collections.abc import Sequence


def check_trainer_absence() -> tuple[bool, str]:
    """The dependency tree must prove Trainer packages absent."""
    import importlib.metadata as md

    installed = {
        dist.metadata["Name"].lower()
        for dist in md.distributions()
        if dist.metadata["Name"] is not None and dist.metadata["Name"] != ""
    }
    present = sorted(name for name in TRAINER_FORBIDDEN_PACKAGES if name in installed)
    if len(present) != 0:
        return False, f"forbidden Trainer packages installed: {present}"
    fabric_ok = "lightning-fabric" in installed
    return (
        fabric_ok,
        f"lightning-fabric present={fabric_ok}; "
        f"forbidden absent={list(TRAINER_FORBIDDEN_PACKAGES)}",
    )


def require_fresh_import(module: str) -> tuple[bool, str]:
    """Import ``module`` in a fresh interpreter; returns (ok, detail)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        stderr_text = proc.stderr if proc.stderr is not None else ""
        stripped = stderr_text.strip()
        tail_lines = stripped.splitlines()[-1:]
        tail = tail_lines if len(tail_lines) != 0 else ["<no stderr>"]
        return False, f"import {module}: rc={proc.returncode}: {tail[0]}"
    return True, f"import {module}: rc=0"

def require_module_imports(modules: Sequence[str]) -> tuple[bool, str]:
    """Fresh-process import of every listed module; first failure reported."""
    for module in modules:
        ok, detail = require_fresh_import(module)
        if not ok:
            return False, detail
    return True, f"fresh-process imports OK for {len(tuple(modules))} modules"


def verify_torch_cuda_stack(*, require_sm120: bool = True) -> tuple[bool, str]:
    """torch must be the CUDA build; sm_120 kernels required at import time."""
    import torch

    cuda_build = torch.version.cuda
    available = torch.cuda.is_available()
    arch_list = list(torch.cuda.get_arch_list())
    has_sm120 = "sm_120" in arch_list
    details = (
        f"torch={torch.__version__} cuda={cuda_build} available={available} arch_list={arch_list}"
    )
    if cuda_build is None or not available:
        return False, details
    if require_sm120 and not has_sm120:
        return False, details + " (sm_120/Blackwell kernels missing)"
    return True, details
