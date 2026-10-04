"""Work-package completion verification (BUILD §1, plan C02).

Thin translator over the native ``hydra2._native.rl`` verifier: one native
call validates the completion envelope, canonical hashes, outputs, command
logs, reports, dependencies, and exit disposition, then publishes the index.
No hash/validation loops here — identity math stays native.
"""

from __future__ import annotations

import json

from hydra2._native import rl as _native_rl  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError

__all__ = ["verify"]


def verify(work_package: str, artifact_root: str) -> dict:
    """Verify one work-package completion record; return the native receipt."""
    if not isinstance(work_package, str) or not work_package:
        raise ContractError("package_verify work_package must be a nonempty str")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise ContractError("package_verify artifact_root must be a nonempty str")
    try:
        blob = _native_rl.verify_work_package(
            work_package=work_package, artifact_root=artifact_root
        )
    except Exception as exc:
        raise ContractError(f"package_verify {work_package!r} rejected: {exc}") from exc
    try:
        receipt = json.loads(bytes(blob).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"package_verify {work_package!r} receipt is not JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ContractError(f"package_verify {work_package!r} receipt must be an object")
    return receipt
