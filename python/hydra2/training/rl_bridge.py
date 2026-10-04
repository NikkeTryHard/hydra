"""Thin artifact/control FFI boundary for the native rollout pipeline.

Typing, lifetime firewall, and batch calls only: every function forwards
opaque JSON bytes/paths/handles to ``hydra2._native.rl_artifact`` (archive
seal/replay) or ``hydra2._native.rl`` (live drain/register/fill) and returns
descriptors, counts, or digests. No row loops, no hash logic, no
canonicalization here — identity math stays native. Torch materializes
only inside :func:`materialize_slab` (lazy import, zero-copy views).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hydra2._native import rl as _native_rl  # pyrefly: ignore[missing-import]
from hydra2._native import rl_artifact as _native_artifact  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_digest(text: str, *, name: str) -> str:
    if not isinstance(text, str) or _DIGEST_RE.match(text) is None:
        raise ContractError(f"rl_bridge {name} must be sha256:<64 hex>")
    return text


def _encode(document: object) -> bytes:
    try:
        return json.dumps(document).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"rl_bridge document not JSON-encodable: {exc}") from exc


def _decode(blob: bytes, *, name: str) -> dict:
    try:
        value = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"rl_bridge {name} is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"rl_bridge {name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    artifact_digest: str
    row_count: int
    compressed_bytes: int
    uncompressed_bytes: int


def _receipt_of(blob: bytes) -> ArtifactReceipt:
    doc = _decode(blob, name="receipt")
    return ArtifactReceipt(
        artifact_digest=_require_digest(doc["artifact_digest"], name="artifact_digest"),
        row_count=int(doc["row_count"]),
        compressed_bytes=int(doc["compressed_bytes"]),
        uncompressed_bytes=int(doc["uncompressed_bytes"]),
    )


def seal_begin(
    tape_handle: int,
    header: dict,
    source_ledger: list[str],
    eval_ledger: list[str],
    expect_bc: bool,
    artifact_path: str,
    metas_path: str,
) -> int:
    """Admit one async seal job; poll with :func:`seal_poll` (never blocks)."""
    return int(
        _native_artifact.seal_begin(
            int(tape_handle),
            _encode(header),
            list(source_ledger),
            list(eval_ledger),
            bool(expect_bc),
            str(artifact_path),
            str(metas_path),
        )
    )


def seal_poll(job_id: int) -> dict:
    """Nonblocking seal status (`running`, or `done` with a receipt)."""
    doc = _decode(_native_artifact.seal_poll(int(job_id)), name="poll")
    if doc.get("status") == "done":
        _receipt_of(_encode(doc["receipt"]))
    return doc


def seal_wait(job_id: int) -> ArtifactReceipt:
    """Block for one admitted seal (quiesce/drain only, never the hot loop)."""
    return _receipt_of(_native_artifact.seal_wait(int(job_id)))


def seal_sync(
    tape_handle: int,
    header: dict,
    source_ledger: list[str],
    eval_ledger: list[str],
    expect_bc: bool,
    artifact_path: str,
    metas_path: str,
) -> ArtifactReceipt:
    """Synchronous seal for quiesce/drain and single-threaded tooling."""
    return _receipt_of(
        _native_artifact.seal_sync(
            int(tape_handle),
            _encode(header),
            list(source_ledger),
            list(eval_ledger),
            bool(expect_bc),
            str(artifact_path),
            str(metas_path),
        )
    )


def stage_test_tape(games: list[dict]) -> int:
    """Register a bounded gate-fixture tape; returns the native handle.

    Gate fixtures only (never production): rows are canonical documents
    plus per-row behavior values (``{"doc": {...}, "v": 8.0}``); arenas stay
    empty so every row is ``T=32``. The native materializer returns each
    row's embedded observation and the encoder is the real
    ``encode_row_facts`` (zero-history planes) — no stub values.
    """
    return int(_native_artifact.stage_test_tape(_encode(list(games))))


def validate_artifact(
    artifact_path: str,
    metas_path: str,
    header: dict,
    source_ledger: list[str],
    eval_ledger: list[str],
    expect_bc: bool,
) -> ArtifactReceipt:
    """Streaming revalidation of a stored artifact against its metas."""
    return _receipt_of(
        _native_artifact.validate_artifact(
            str(artifact_path),
            str(metas_path),
            _encode(header),
            list(source_ledger),
            list(eval_ledger),
            bool(expect_bc),
        )
    )


def build_order(metas_path: str, shuffle_key: dict) -> tuple[dict, bytes]:
    """Frozen order plan: descriptor (counts/digests) plus opaque index."""
    descriptor_blob, index_bytes = _native_artifact.build_order(
        str(metas_path), _encode(shuffle_key)
    )
    return _decode(descriptor_blob, name="order"), bytes(index_bytes)


def register_replay(index: bytes, metas_path: str) -> int:
    """Freeze one generation's order index plus metas natively (cold, once).

    Parses and validates both documents a single time per generation/resume,
    precomputing per-batch layout and offsets. Returns the process replay
    handle the production :func:`fill_slab` serves; release with
    :func:`drop_replay` at generation teardown.
    """
    return int(_native_artifact.register_replay(bytes(index), str(metas_path)))


def drop_replay(replay_handle: int) -> None:
    """Release one frozen replay handle (unknown handles fail closed)."""
    _native_artifact.drop_replay(int(replay_handle))


def terminal_inputs(tape_handle: int) -> tuple[dict, bytes, bytes, bytes, bytes]:
    """GPU terminal inputs: descriptor plus actor/game-id/value/utility spans.

    Cold, once per complete generation; no rows or documents cross. The Torch
    boundary computes f32 advantages from these spans (mirroring the frozen
    ``f32(G[actor]) - V[actor]`` recipe) and returns them to
    :func:`freeze_terminal_advantages`.
    """
    descriptor_blob, actors, game_index, values, utilities = _native_artifact.terminal_inputs(
        int(tape_handle)
    )
    return (
        _decode(descriptor_blob, name="terminal inputs"),
        bytes(actors),
        bytes(game_index),
        bytes(values),
        bytes(utilities),
    )


def freeze_terminal_advantages(tape_handle: int, rawa_f32: bytes) -> dict:
    """Freeze GPU-computed advantages immutably, exactly once pre-seal."""
    return _decode(
        _native_artifact.freeze_terminal_advantages(int(tape_handle), bytes(rawa_f32)),
        name="freeze receipt",
    )


@dataclass(frozen=True, slots=True)
class SlabSpec:
    batch: int
    bucket_t: int
    count: int
    row_bytes: int
    feature_bytes: int
    fact_bytes: int
    input_layout: tuple
    artifact_digest: str
    order_digest: str
    batch_envelope: dict


def _slab_spec_of(spec: dict) -> SlabSpec:
    return SlabSpec(
        batch=int(spec["batch"]),
        bucket_t=int(spec["t"]),
        count=int(spec["count"]),
        row_bytes=int(spec["row_bytes"]),
        feature_bytes=int(spec["feature_bytes"]),
        fact_bytes=int(spec["fact_bytes"]),
        input_layout=tuple(spec["input_layout"]),
        artifact_digest=_require_digest(spec["artifact_digest"], name="artifact_digest"),
        order_digest=_require_digest(spec["order_digest"], name="order_digest"),
        batch_envelope=dict(spec["batch_envelope"]),
    )


def export_slab(
    tape_handle: int, index: bytes, metas_path: str, batch_idx: int
) -> tuple[SlabSpec, bytes, bytes, bytes, bytes, bytes, bytes, bytes, bytes]:
    """Cold export: one optimizer batch as owned tuples (gate fixtures)."""
    out = _native_artifact.export_slab(
        int(tape_handle), bytes(index), str(metas_path), int(batch_idx)
    )
    if len(out) != 9:
        raise ContractError(f"rl_bridge slab export must carry 9 parts, got {len(out)}")
    (
        spec_blob,
        features,
        old_lp,
        raw_a,
        g4_values,
        selected,
        legal,
        facts,
        envelope,
    ) = out
    return (
        _slab_spec_of(_decode(spec_blob, name="slab spec")),
        bytes(features),
        bytes(old_lp),
        bytes(raw_a),
        bytes(g4_values),
        bytes(selected),
        bytes(legal),
        bytes(facts),
        bytes(envelope),
    )


def slab_geometry(bucket_t: int, capacity: int) -> tuple[int, int]:
    """Exact `(feature_bytes, fact_bytes)` slot spans for one lane."""
    feature_bytes, fact_bytes = _native_artifact.slab_geometry(int(bucket_t), int(capacity))
    return int(feature_bytes), int(fact_bytes)


def verify_row_derivation(header_digest: str, derivation_doc: dict) -> dict:
    """Boundary check: header digest equals the JCS derivation document digest.

    Called by prepare paths before the header is assembled; the seal binds
    rows to the verified header. Returns the canonical binding receipt.
    """
    return _decode(
        _native_artifact.verify_row_derivation(
            _require_digest(header_digest, name="row_derivation_hash"),
            _encode(derivation_doc),
        ),
        name="derivation receipt",
    )


def fill_slab(
    tape_handle: int,
    replay_handle: int,
    batch_idx: int,
    features_buffer: bytearray,
    facts_buffer: bytearray,
) -> SlabSpec:
    """Production fill into caller-owned pinned spans (no expanded copies).

    The order index and metas companion arrive frozen via ``replay_handle``
    (one :func:`register_replay` per generation/resume); no index bytes and
    no metas path cross on this hot path. Buffers are Torch-owned contiguous
    ``uint8`` spans (pinned-tensor numpy views or cached bytearrays sized via
    :func:`slab_geometry`); tails fill the prefix. Returns the canonical spec
    (counts, layout, digests).
    """
    if not isinstance(features_buffer, (bytearray, memoryview)) or not isinstance(
        facts_buffer, (bytearray, memoryview)
    ):
        raise ContractError("rl_bridge fill spans must be writable buffers")
    return _slab_spec_of(
        _decode(
            _native_artifact.fill_slab(
                int(tape_handle),
                int(replay_handle),
                int(batch_idx),
                features_buffer,
                facts_buffer,
            ),
            name="slab spec",
        )
    )


def register_live_replay(
    tape_handle: int, collector_spec: bytes, run_descriptor: bytes
) -> tuple[int, dict]:
    """Typed replay registration from a drained tape (no seal/file round trip).

    Returns the replay handle plus the live descriptor (schema, pipeline,
    generation, behavior policy, row count, batch counts). Full positions
    stay native.
    """
    handle, descriptor_blob = _native_rl.register_live_replay(
        tape_handle=int(tape_handle),
        collector_spec_json=bytes(collector_spec),
        run_descriptor_json=bytes(run_descriptor),
    )
    return int(handle), _decode(descriptor_blob, name="live replay")


def fill_live_slab(
    tape_handle: int,
    replay_handle: int,
    batch_idx: int,
    features_buffer: bytearray,
    facts_buffer: bytearray,
) -> dict:
    """Live production fill into caller-owned pinned spans (no copies).

    The replay must carry this exact tape handle in its live source. Buffers
    match :func:`fill_slab` spans; tails fill the prefix. Returns the live
    slab descriptor (schema, batch, geometry, layout; no checksums).
    """
    if not isinstance(features_buffer, (bytearray, memoryview)) or not isinstance(
        facts_buffer, (bytearray, memoryview)
    ):
        raise ContractError("rl_bridge fill spans must be writable buffers")
    return _decode(
        _native_rl.fill_live_slab(
            tape_handle=int(tape_handle),
            replay_handle=int(replay_handle),
            batch_idx=int(batch_idx),
            features_buffer=features_buffer,
            facts_buffer=facts_buffer,
        ),
        name="live slab",
    )


_PLANE_DTYPES = {"int64": "int64", "float64": "float64", "int32": "int32", "bool": "bool"}


def materialize_slab(spec: SlabSpec, features: bytes, facts48: bytes) -> dict[str, object]:
    """Wrap filled spans as Torch views (no CPU copies on this path).

    ``numpy.frombuffer`` views share the caller's spans (which must outlive
    the tensors); the GPU loop's host-to-device transfer is the single copy.
    Planes follow ``spec.input_layout`` entries (contiguous per plane);
    facts follow the plane-major blocks (selected int64[B] @0, old-LP
    float32[B] @8B, raw-A float32[B] @12B, canonical G4 float64[B,4] @16B).
    """
    import numpy as np
    import torch

    if len(features) < spec.feature_bytes or len(facts48) < spec.fact_bytes:
        raise ContractError("rl_bridge slab spans are smaller than the spec")
    count = spec.count
    planes: dict[str, object] = {}
    for entry in spec.input_layout:
        name, dtype, shape, offset = (
            str(entry["name"]),
            str(entry["dtype"]),
            tuple(int(dim) for dim in entry["shape"]),
            int(entry["offset"]),
        )
        if dtype not in _PLANE_DTYPES:
            raise ContractError(f"rl_bridge unknown plane dtype {dtype!r}")
        elements = 1
        for dim in shape:
            elements *= dim
        planes[name] = torch.from_numpy(
            np.frombuffer(features, dtype=np.dtype(dtype), count=elements, offset=offset)
        ).reshape(shape)
    facts = {
        "selected": torch.from_numpy(np.frombuffer(facts48, dtype=np.int64, count=count, offset=0)),
        "old_lp": torch.from_numpy(
            np.frombuffer(facts48, dtype=np.float32, count=count, offset=8 * count)
        ),
        "raw_a": torch.from_numpy(
            np.frombuffer(facts48, dtype=np.float32, count=count, offset=12 * count)
        ),
        "g4": torch.from_numpy(
            np.frombuffer(facts48, dtype=np.float64, count=4 * count, offset=16 * count)
        ).reshape((count, 4)),
    }
    return {"planes": planes, "facts": facts}


def checkpoint_receipt(root_dir: str, locator: str, manifest: dict) -> dict:
    """Cold receipt for one landed ordinary checkpoint (streams the file)."""
    return _decode(
        _native_artifact.checkpoint_receipt(str(root_dir), str(locator), _encode(manifest)),
        name="checkpoint receipt",
    )


def publish_control(
    root_dir: str,
    control: dict,
    receipts: list[dict],
    expected_parent: str | None,
    blobs: list[dict],
) -> dict:
    """Atomic control publication; returns the committed tip document."""
    return _decode(
        _native_artifact.publish_control(
            str(root_dir),
            _encode(control),
            [_encode(receipt) for receipt in receipts],
            expected_parent,
            _encode(list(blobs)),
        ),
        name="tip",
    )


def read_tip(root_dir: str) -> dict:
    """Authoritative control document behind the last committed tip."""
    return _decode(_native_artifact.read_tip(str(root_dir)), name="control")


def stage_native_snapshot(root_dir: str, game: dict) -> dict:
    """Content-addressed write of one per-game native snapshot envelope."""
    return _decode(
        _native_artifact.stage_native_snapshot(str(root_dir), _encode(game)),
        name="native receipt",
    )


def read_native_snapshot(root_dir: str, game_id: str, digest: str) -> dict:
    """Digest-verified read of one staged native snapshot."""
    return _decode(
        _native_artifact.read_native_snapshot(
            str(root_dir), str(game_id), _require_digest(digest, name="digest")
        ),
        name="native snapshot",
    )


def commit_blob(path: str) -> None:
    """Durable blob step: fsync an immutable file plus its directory."""
    _native_artifact.commit_blob(str(path))
