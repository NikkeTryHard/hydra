"""Thin columnar judge over the ``hydra2_replay_rs.columnar`` boundary.

Rust expands (detached) and exports ONE single-consume Arrow-C stream capsule
(attached — B4 capsule-outside-detach); Python imports zero-copy in-proc
(``pa.RecordBatchReader.from_pycapsule``) or persists IPC-file cross-proc
(``pa.ipc.new_file`` / ``open_file``) and compares ``dataset_hash``
fail-close. No expand/parquet/hash math lives here — this module only moves
the capsule across the boundary and asserts equality.

Resolution notes:
- Legacy entry point stays ``import hydra2_replay_rs`` (crate/lib names
  untouched); the ``hydra_bridge._native`` rename is a Phase-6 maturin
  ``module-name`` cutover and MUST NOT be attempted here.
- B4 ordering (compute detached, capsule attached) is enforced Rust-side by
  signature (``export_stream_capsule`` requires the attached token); this
  module only ever receives a finished capsule.
- Batch schema equals ``data/parquet.py`` ``_ACTOR_SCHEMA`` exactly
  (13 ``ACTOR_FIELDS`` order, ``pa.string()``/``pa.int64``, chosen_action_id
  sole nullable) — asserted by comparing logical values, never ``memcmp``.
- DLPack NOT here: packet-bytes path only; device tensors belong to the loop
  (Phase 5).
- Probe-A: ``probe_a_capsule_release`` loops 10k exports with immediate
  consume+drop (operator-observed RSS-stable; no allocator import here).
- Writer-owned, referenced only: 5-exact parquet pins (zstd/level3/dict/
  batch8192/store_schema), ``created_by`` UNSET, ``ARROW:schema`` KV both
  writers, privileged re-freeze gate, Probe-B/C.

"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from hydra2.artifacts.digest import require_digest_match
from hydra2.contracts.common import ContractError, DigestText

__all__ = [
    "ColumnarJudge",
    "batch_size",
    "capsule_stats",
    "dataset_hash_of_ids",
    "next_into_capsule",
    "persist_ipc_file",
    "probe_a_capsule_release",
    "read_ipc_file",
    "table_dataset_hash",
    "table_from_capsule",
]

#: Decision-id column (col 2 of the canonical actor schema; the sorted-ids
#: ``dataset_hash`` input on both writers).
_ID_COLUMN = "decision_id"

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so ``_columnar()`` imports the compiled extension).
_NATIVE_OVERRIDE: Any = None


def _native() -> Any:
    """Import the compiled bridge (fail closed, no fallback expand)."""
    try:
        return importlib.import_module("hydra2_replay_rs")
    except ImportError as exc:
        raise RuntimeError(
            "hydra2_replay_rs extension with columnar not importable; "
            "build the bridge before using the columnar judge"
        ) from exc


def _columnar() -> Any:
    if _NATIVE_OVERRIDE is not None:
        return _NATIVE_OVERRIDE
    ext = _native()
    try:
        return ext.columnar
    except AttributeError as exc:
        raise RuntimeError(
            "hydra2_replay_rs.columnar submodule missing; rebuild the bridge"
        ) from exc


def batch_size() -> int:
    """Single-writer batch width (Rust ``columnar::BATCH`` owns the const)."""
    return int(_columnar().batch_size())


def capsule_stats() -> tuple[int, int]:
    """Judge observability: ``(capsules_exported, batches_exported)``."""
    capsules, batches = _columnar().capsule_stats()
    return (int(capsules), int(batches))


def next_into_capsule(
    ptrs: Sequence[int],
    byte_caps: Sequence[int],
    requested_schema: Any = None,
) -> Any:
    """Expand to batches (Rust, detached) and return ONE stream capsule.

    ``ptrs``/``byte_caps`` are opaque caller descriptors forwarded verbatim
    (caller sizes the input so rows land near ``batch_size()``). The capsule
    is single-consume movable: import it exactly once, immediately, via
    :func:`table_from_capsule`.
    """
    for ptr in ptrs:
        if isinstance(ptr, bool) or not isinstance(ptr, int):
            raise TypeError(f"columnar ptr must be an int, got {type(ptr).__name__}")
        if ptr == 0:
            raise ValueError("columnar null pointer descriptor fails closed")
    for cap in byte_caps:
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise TypeError(f"columnar cap must be an int, got {type(cap).__name__}")
    if len(ptrs) != len(byte_caps):
        raise ValueError(
            f"columnar descriptor length mismatch: {len(ptrs)} ptrs vs {len(byte_caps)} caps"
        )
    return _columnar().next_into_capsule(list(ptrs), list(byte_caps), requested_schema)


def table_from_capsule(capsule: Any) -> pa.Table:
    """Import a stream capsule zero-copy in-proc (single-consume).

    A second import of the same capsule raises (release already ran) — the
    Arrow single-consume rule, enforced producer-side by the destructor.
    """
    reader = pa.RecordBatchReader.from_pycapsule(capsule)
    return reader.read_all()


def persist_ipc_file(table: pa.Table, path: Path | str) -> Path:
    """Persist a table as IPC **file** format (random access, cross-proc).

    The Arrow-C stream is the in-proc handoff; the IPC file is the persist /
    cross-proc edge (mmap reads, random batch access). Defaults carry the
    newest IPC metadata (pyarrow 25.0.1, pixi-pinned).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pa_ipc.new_file(out, table.schema) as writer:
        writer.write_table(table)
    return out


def read_ipc_file(path: Path | str) -> pa.Table:
    """Read back an IPC **file** blob (random access, cross-proc edge)."""
    with pa_ipc.open_file(Path(path)) as reader:
        return reader.read_all()


def dataset_hash_of_ids(decision_ids: list[str]) -> str:
    """Builder-pinned dataset hash: ``sha256:`` + hex over LF-joined sorted ids.

    Canonical owners: ``training/shard_reader.py:dataset_hash_of`` and the
    ``shard_build.py:448`` inline formula (single LF, no trailing newline,
    utf-8). Re-stated here so the judge compares without importing training.
    """
    joined = "\n".join(sorted(decision_ids)).encode("utf-8")
    return "sha256:" + hashlib.sha256(joined).hexdigest()


def table_dataset_hash(table: pa.Table, *, id_column: str = _ID_COLUMN) -> str:
    """Hash a columnar table over its sorted decision ids (fail closed)."""
    if table.schema.get_field_index(id_column) < 0:
        raise ContractError(f"columnar table missing id column {id_column!r}")
    column = table.column(id_column).to_pylist()
    ids: list[str] = []
    for value in column:
        if not isinstance(value, str):
            raise ContractError(
                f"columnar id column {id_column!r} must hold str, got {type(value).__name__}"
            )
        ids.append(value)
    return dataset_hash_of_ids(ids)


@dataclass(frozen=True, slots=True)
class ColumnarJudge:
    """Fail-close comparator: assert recomputed == recorded, else raise."""

    subject: str = "columnar"
    id_column: str = _ID_COLUMN

    def verify_table(self, *, recorded: str, table: pa.Table) -> DigestText:
        """Recompute ``dataset_hash`` over the table ids and compare."""
        recomputed: DigestText = DigestText(table_dataset_hash(table, id_column=self.id_column))
        require_digest_match(recorded=recorded, recomputed=recomputed, subject=self.subject)
        return recomputed

    def verify_ipc_roundtrip(self, *, recorded: str, table: pa.Table, path: Path | str) -> pa.Table:
        """Persist via IPC-file, read back, and compare both hashes.

        Proves the cross-proc edge preserves identity: the in-memory table
        hash and the IPC-file round-trip hash must BOTH equal ``recorded``.
        """
        self.verify_table(recorded=recorded, table=table)
        round_tripped = read_ipc_file(persist_ipc_file(table, path))
        self.verify_table(recorded=recorded, table=round_tripped)
        return round_tripped


def probe_a_capsule_release(
    *,
    ptrs: Sequence[int],
    byte_caps: Sequence[int],
    iters: int = 10_000,
    backend: Any = None,
) -> int:
    """Probe-A: export + immediately consume ``iters`` capsules, then drop.

    Each capsule is imported at once (single-consume) and released at loop
    end; a correct destructor keeps RSS stable across the run (observed by
    the operator — no allocator import here). Returns ``iters``.
    """
    be = backend if backend is not None else _columnar()
    for _ in range(iters):
        capsule = be.next_into_capsule(list(ptrs), list(byte_caps))
        table_from_capsule(capsule)
        del capsule
    return iters
