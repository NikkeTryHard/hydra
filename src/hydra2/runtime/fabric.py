"""Standalone Lightning-Fabric runtime adapter (SPEC 10) — REMOVED (M3).

End-state: Fabric is KILLED. :mod:`hydra2.runtime.plain` is the sole
runtime adapter; ``fabric_2.6.5`` no longer exists as an adapter id.

Migration (one line per call site): replace
``FabricRuntimeAdapter`` with ``PlainPytorchAdapter`` (``fp32`` →
``fp32``, ``bf16_mixed`` → ``bf16_mixed``; ``fp16_mixed`` was already
rejected on both sides, so it has no mapping).

Retargets recorded for Main (DO NOT edit here — other owners land them):

- ``tests/integration/test_parity_plain_fabric.py`` → retarget to a
  plain-only parity probe (plain-vs-plain seeded repeat + the fp32
  tolerance contract) or delete; it imports this module and must not
  survive the kill.
- ``test-serial`` lane (``pyproject.toml [tool.pixi.tasks]``) → drop the
  ``tests/integration/test_parity_plain_fabric.py`` file entry from the
  ordered file list (the lane lists files, never globs; keep cheapness
  order for the rest).
- ``pyproject.toml`` ``lightning-fabric == 2.6.5`` pin → delete.
- ``src/hydra2/runtime/__init__.py:18,39`` re-exports → delete.
- ``src/hydra2/training/_rc_sections.py:100`` ``_ADAPTER_IDS`` →
  ``("plain_pytorch",)``.
- ``src/hydra2/training/stream_train.py:469-475`` fabric branch →
  delete (plain-only construction).
- ``src/hydra2/runtime/protocol.py:31`` ``SUPPORTED_ADAPTER_IDS`` +
  ``RuntimeSpec.adapter_id`` Literal → ``("plain_pytorch",)`` /
  ``Literal["plain_pytorch"]`` (protocol owner lands with the rest).

Old manifests keep parsing: unknown adapter ids read back as ``MISSING``
(the env-manifest tolerance), never a hard failure.

Future DDP seam (NOT this change): ``RuntimeSpec(adapter_id="c10d_ddp",
device="cuda:N")`` + ``build_runtime`` inits ``dist`` BEFORE
``compile_once`` (DDP wraps compiled-or-eager per current torch
guidance; verify against the 2.14 tree at impl time).
"""

from __future__ import annotations

from hydra2.runtime.plain import PlainPytorchAdapter as PlainPytorchAdapter

__all__ = [
    "PlainPytorchAdapter",
]
