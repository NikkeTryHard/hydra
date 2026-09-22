"""Actor-visible tensor encoder — padded/bucketed histories with masks.

Converts :class:`ActorObservation` into :class:`ActorTensorBatch` tensors.
All inputs originate from ``ActorObservation`` (actor-visible boundary);
no hidden world, wall, or privileged label is consulted.
Padding values never carry semantics without their mask.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError, DigestText

if TYPE_CHECKING:
    from hydra2.contracts.observation_actor import ActorObservation

from hydra2.models.schema import (
    _BASELINE_FIELDS,
    BASELINE_ACTION_COUNT,
    HISTORY_BUCKET_LENGTHS,
    model_input_schema_digest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bridge-ring glue (Rust-only encoding; no Python oracle).
# ---------------------------------------------------------------------------
#
# The bridge owns encoding end to end: `hydra2._native.encoder`
# `stage_encoder_batch` (bulk LE fill under detach — buffer+ring over the
# retired per-row fill) plus `hydra2._native.ring` (`ring_fill_batch` bulk
# plane copy into torch-owned pinned slots, `validate_encoder_batch`
# frozen-geometry judge, `bucket_for_length` SINGLE ceil fn, `PyRing`
# cursor/slot_used + 512-window h2d/sync accounting). Fused-CE stays
# forced-Python (M7: per-bucket bakeoff wall + 1e-4 + NaN parity gate; D5
# triton_op recipe gated) — gate notes live in ring.rs, which also carries
# the M5/M8/m3/m6 ledger.
#
#: Hardening (fail-closed): the extension is required — an absent surface
#: (``ImportError``/missing attr) raises ``ContractError`` with a build
#: hint, never a silent oracle fallback. Bridge-present errors raise too
#: (geometry ``ValueError`` text mapped 1:1, everything else wrapped).
#: Evidence: encoder 3.2x Rust-faster on real rows + ring padding
#: byte-identity pinned by
#: ``ring_tests::padding_identity_zero_and_neg1_tails`` (0x00 tails on
#: 0/False planes, 0xFF tails on -1 planes). Torch owns the GPU: torch
#: allocates/owns the pinned slots physically; Rust only bulk-copies raw
#: bytes under detach (no GPU math, Burn/Candle out).
_RING_MOD: Any = None
_RING_PROBED: bool = False

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so `_ring_native()` imports the compiled extension). Mirrors the
#: `_NATIVE_OVERRIDE` hook in `_rust_bridge.py` / `_rust_columnar.py` /
#: `_rust_search.py`.
_RING_NATIVE_OVERRIDE: Any = None


def _ring_native() -> Any | None:
    """Import the built `ring` submodule once; None → fail-closed raise by callers.

    ``None`` means the extension (or its ``ring`` surface) is not importable.
    Callers raise ``ContractError`` with a build hint — never a silent fallback.
    """
    global _RING_MOD, _RING_PROBED
    if _RING_NATIVE_OVERRIDE is not None:
        return _RING_NATIVE_OVERRIDE
    if _RING_MOD is not None:
        return _RING_MOD
    # Cheap late-arrival path: someone else (conftest, bench harness) may
    # import the built bridge after our first probe — a sys.modules hit
    # costs one dict lookup, no path rescan.
    late = sys.modules.get("hydra2._native")
    if late is not None:
        mod = getattr(late, "ring", None)
        if mod is not None:
            _RING_MOD = mod
            return _RING_MOD
    if not _RING_PROBED:
        _RING_PROBED = True
        try:
            # Established judge pattern (cf. `_rust_bridge._native` +
            # `_canon_rng`): import the top-level extension, then take the
            # submodule as an attribute (`add_submodule` publishes it).
            _RING_MOD = importlib.import_module("hydra2._native").ring
        except (ImportError, AttributeError):
            # Unbuilt extension / bridge without the ring surface only.
            # Anything else is a broken bridge → raise, never a silent pass.
            _RING_MOD = None
    return _RING_MOD


#: Injectable native encoder backend (same judge pattern as
#: `_RING_NATIVE_OVERRIDE`; tests monkeypatch this, production leaves None).
_ENCODER_NATIVE_OVERRIDE: Any = None
_ENCODER_MOD: Any = None
_ENCODER_PROBED: bool = False


def _encoder_native() -> Any | None:
    """Import the built `encoder` submodule once; None → fail-closed raise by callers.

    ``None`` means the extension (or its ``encoder`` surface) is not
    importable. Callers raise ``ContractError`` with a build hint — never a
    silent fallback.
    """
    global _ENCODER_MOD, _ENCODER_PROBED
    if _ENCODER_NATIVE_OVERRIDE is not None:
        return _ENCODER_NATIVE_OVERRIDE
    if _ENCODER_MOD is not None:
        return _ENCODER_MOD
    late = sys.modules.get("hydra2._native")
    if late is not None:
        mod = getattr(late, "encoder", None)
        if mod is not None:
            _ENCODER_MOD = mod
            return _ENCODER_MOD
    if not _ENCODER_PROBED:
        _ENCODER_PROBED = True
        try:
            _ENCODER_MOD = importlib.import_module("hydra2._native").encoder
        except (ImportError, AttributeError):
            _ENCODER_MOD = None
    return _ENCODER_MOD


def _encode_batch(
    observations: list[ActorObservation],
    buckets: tuple[int, ...],
    *,
    pin_memory: bool,
) -> ActorTensorBatch:
    """Encode one batch through Rust bulk fill (no Python oracle).

    One ``stage_encoder_batch`` FFI under detach fills all 26 planes for
    frozen or custom buckets; geometry, dora width, legal mask, and history
    cap fail closed in Rust. Requires the built extension (fail closed with
    a build hint when absent). Bridge-present errors raise ``ContractError``
    (``ValueError`` text mapped 1:1, everything else wrapped) — mismatch
    raises, never a silent fallback.
    """
    native = _encoder_native()
    if native is None:
        raise ContractError(
            "hydra2._native.encoder missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    try:
        staged: tuple[dict[str, bytearray], int, int, list[str]] = native.stage_encoder_batch(
            list(observations), list(buckets)
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(
            f"encoder stage_encoder_batch failed: {type(exc).__name__}: {exc}"
        ) from exc
    planes, bucket_t, max_len, hashes = staged
    batch_size = len(observations)
    specs = {f.name: f for f in _BASELINE_FIELDS}
    dtype_map = {"bool": torch.bool, "int32": torch.int32, "int64": torch.int64}
    features: dict[str, torch.Tensor] = {}
    for name, spec in specs.items():
        shape = tuple(
            batch_size
            if d == "B"
            else bucket_t
            if d == "T"
            else BASELINE_ACTION_COUNT
            if d == "A"
            else int(d)
            for d in spec.shape
        )
        try:
            buf = planes[name]
        except KeyError as exc:
            raise ContractError(f"encoder stage missing plane {name!r}") from exc
        try:
            dt = dtype_map[spec.dtype]
        except KeyError as exc:
            raise ContractError(f"encoder stage unknown dtype {spec.dtype!r}") from exc
        features[name] = torch.frombuffer(buf, dtype=dt).reshape(shape)
    history_mask = features["history_mask"]
    legal_mask = features["legal_mask"]
    actor_seats = features["actor_seats"]
    if pin_memory and torch.cuda.is_available():
        try:
            if tuple(buckets) == tuple(HISTORY_BUCKET_LENGTHS):
                features = _stage_pinned_batch(
                    features,
                    batch_size=batch_size,
                    max_history_len=max_len,
                    bucket_t=bucket_t,
                )
            else:
                features = {
                    name: tensor.pin_memory()  # type: ignore[attr-defined]  # reason: custom-bucket test probe skips the frozen Ring judge; pageable fallback still owns unpinnable
                    for name, tensor in features.items()
                }
            history_mask = features["history_mask"]
            legal_mask = features["legal_mask"]
            actor_seats = features["actor_seats"]
        except ContractError:
            raise
        except Exception as exc:  # why-broad: any pin failure falls back to pageable
            logger.warning("encoder pin_memory failed, using pageable fallback: %s", exc)
    expected = {f.name for f in _BASELINE_FIELDS}
    produced = set(features.keys())
    if expected != produced:
        missing = sorted(expected - produced)
        extra = sorted(produced - expected)
        raise ContractError(f"feature mismatch missing={missing} extra={extra}")
    return ActorTensorBatch(
        features=features,
        history_mask=history_mask,
        legal_mask=legal_mask,
        observation_hashes=tuple(DigestText(h) for h in hashes),
        actor_seats=actor_seats,
    )


def _bucket_length(actual: int, buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS) -> int:
    """Ceil ``actual`` to the next bucket; over-cap callers fail closed."""
    for bucket in buckets:
        if actual <= bucket:
            return bucket
    return buckets[-1]


@dataclass(frozen=True, slots=True)
class ActorTensorBatch:
    """Batched actor-visible tensors (SPEC 11.1).

    Mask polarity: ``history_mask``/``legal_mask`` use ``True`` = participate,
    while ``key_padding_mask`` in models/model.py uses ``True`` = padding
    (single ``~`` inversion at the encode-to-model boundary).
    """

    features: dict[str, torch.Tensor]
    history_mask: torch.Tensor  # [B,T] bool True=participate
    legal_mask: torch.Tensor  # [B,A] bool
    observation_hashes: tuple[DigestText, ...]
    actor_seats: torch.Tensor  # [B] int64


def encode_observations(
    observations: list[ActorObservation],
    *,
    buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS,
    pin_memory: bool = True,
) -> ActorTensorBatch:
    """Encode a list of observations into a padded, bucketed batch.

    Rust bulk fill only (no Python oracle): one ``stage_encoder_batch`` FFI
    under detach fills all 26 planes for frozen or custom buckets; geometry,
    dora width, legal mask, and history cap fail closed in Rust. Requires
    the built extension (fail closed with a build hint when absent).
    """
    if len(observations) == 0:
        raise ContractError("encode_observations requires at least one observation")
    return _encode_batch(observations, tuple(buckets), pin_memory=pin_memory)


def bucket_for_length(actual: int) -> int:
    """Public helper: bucket length for a given history length.

    SINGLE ceil fn both sides call (scan-side + ring-side): Rust-first via
    ``hydra2._native.ring.bucket_for_length``; the oracle ``_bucket_length``
    runs ONLY when the bridge surface is absent (ImportError/missing attr).
    Bridge-present errors raise ContractError (ValueError text preserved
    1:1) — mismatch=raise, never a silent fallback. Over-cap returns the
    max bucket here; callers fail closed before calling (never truncate).
    """
    ring = _ring_native()
    if ring is not None:
        try:
            bucket: int = ring.bucket_for_length(actual)
            return bucket
        except ContractError:
            raise
        except (ImportError, AttributeError):
            pass  # bridge surface missing → oracle twin decides
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except Exception as exc:
            raise ContractError(
                f"ring bucket_for_length failed: {type(exc).__name__}: {exc}"
            ) from exc
    return _bucket_length(actual)


def validate_encoder_batch(
    *,
    batch_size: int,
    max_history_len: int,
    bucket_t: int,
    dora_width: int = 5,
    num_actions: int = BASELINE_ACTION_COUNT,
) -> None:
    """Frozen-geometry judge pre-export (Rust-first, oracle-identical messages).

    Pins the T-geom gate: non-empty batch + ``bucket_t`` member + ceil
    agreement + dora ``(5,)`` sentinel + ``A == 6792`` + never-truncate.
    No-op on success; raises :class:`ContractError` with identical text on
    both paths (Rust ``ValueError`` text is mapped 1:1). The oracle below
    runs ONLY when the bridge surface is absent; any bridge-present error
    raises — mismatch=raise, never a silent pass.
    """
    ring = _ring_native()
    if ring is not None:
        try:
            ring.validate_encoder_batch(
                batch_size,
                max_history_len,
                dora_width,
                num_actions,
                bucket_t,
            )
            return
        except ContractError:
            raise
        except (ImportError, AttributeError):
            pass  # bridge surface missing → oracle twin decides below
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except Exception as exc:
            # Bridge present but unusable (overflow, descriptor drift, ...):
            # mismatch=raise — never fall through to the oracle silently.
            raise ContractError(
                f"ring validate_encoder_batch failed: {type(exc).__name__}: {exc}"
            ) from exc
    if batch_size == 0:
        raise ContractError("encode_observations requires at least one observation")
    if bucket_t not in HISTORY_BUCKET_LENGTHS:
        raise ContractError(
            f"ring bucket_t {bucket_t} not in buckets {list(HISTORY_BUCKET_LENGTHS)}"
        )
    if max_history_len > HISTORY_BUCKET_LENGTHS[-1]:
        raise ContractError(
            f"visible history {max_history_len} exceeds model bucket cap "
            f"{HISTORY_BUCKET_LENGTHS[-1]}; rows are never truncated"
        )
    expect = _bucket_length(max_history_len)
    if bucket_t != expect:
        raise ContractError(f"ring bucket_t {bucket_t} != ceil({max_history_len}) = {expect}")
    if dora_width != 5:
        raise ContractError(f"ring dora width {dora_width} != 5 sentinel (padding -1, no 4-shim)")
    if num_actions != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"ring num_actions {num_actions} != baseline {BASELINE_ACTION_COUNT} "
            "(narrow slicing is Python allow_narrow test-only)"
        )


def _stage_pinned_batch(
    mapping: dict[str, torch.Tensor],
    *,
    batch_size: int,
    max_history_len: int,
    bucket_t: int,
) -> dict[str, torch.Tensor]:
    """Bulk pinned stage: buffer+ring bulk copy into torch-owned pinned slots.

    Geometry is judged first (fail closed, nothing copied), then ALL planes
    cross in ONE detached ``ring_fill_batch`` bulk copy into pre-pinned
    slots — one release instead of the 29 per-tensor ``pin_memory()`` copies.
    Torch owns the GPU: torch allocates/owns the pinned slots physically
    (pinned tensors, streams, events — no Rust GPU math); Rust moves
    shape-agnostic bytes verbatim under detach (never re-inits, never
    interprets padding). Requires the built extension (fail closed with a
    build hint when absent). Non-contiguous/non-tensor inputs pin via torch
    directly (shape precondition, not a bridge verdict). Copy errors raise
    ContractError (Rust ValueError text preserved 1:1) — mismatch=raise,
    never a silent fallback.
    """
    validate_encoder_batch(
        batch_size=batch_size, max_history_len=max_history_len, bucket_t=bucket_t
    )
    ring = _ring_native()
    if ring is None:
        raise ContractError(
            "hydra2._native.ring missing (stale .so); rebuild the bridge with `pixi run build-ext`"
        )
    names = list(mapping.keys())
    srcs = [mapping[name] for name in names]
    for tensor in srcs:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
            # Shape precondition, not a bridge mismatch: strided views pin
            # via torch directly (byte-identical either way).
            return {
                name: tensor.pin_memory()  # type: ignore[attr-defined]  # reason: CPU Tensor.pin_memory; stubs miss
                for name, tensor in mapping.items()
            }
    # Resource alloc stays outside the Rust try: OOM/RuntimeError propagates
    # raw so the caller's pageable fallback (warn) still owns unpinnable.
    slots = [torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True) for tensor in srcs]
    try:
        ring.ring_fill_batch(
            [slot.data_ptr() for slot in slots],
            [slot.nbytes for slot in slots],
            [tensor.data_ptr() for tensor in srcs],
            [tensor.nbytes for tensor in srcs],
            batch_size,
            max_history_len,
            5,
            BASELINE_ACTION_COUNT,
            bucket_t,
        )
    except ContractError:
        raise
    except (ValueError, BufferError) as exc:
        raise ContractError(str(exc)) from exc
    except Exception as exc:
        raise ContractError(f"ring ring_fill_batch failed: {type(exc).__name__}: {exc}") from exc
    return dict(zip(names, slots, strict=True))


def validate_batch_against_schema(batch: ActorTensorBatch) -> None:
    """Runtime shape/dtype/range validation against the frozen schema.

    Padding values are allowed outside valid ranges only where masked;
    unmasked positions must satisfy valid_min/max.
    """
    field_map = {f.name: f for f in _BASELINE_FIELDS}
    # 3-sync kill: the frozen schema holds exactly one masked field
    # (history_event_kind min+max) plus the legal-row gate = 3 fail-scalars,
    # collected below and crossed in ONE stacked tolist (cap: this path
    # performs exactly 1 host sync; never one .item() per check). Order is
    # preserved (loop order, then legal), so single-fault messages are
    # identical; multi-fault precedence is dtype/shape-first (deterministic).
    # Same single-sync shape as loop_batch._window_means.
    sync_checks: list[tuple[str, torch.Tensor]] = []
    for name, tensor in batch.features.items():
        spec = field_map.get(name)
        if spec is None:
            raise ContractError(f"batch field {name!r} not in schema")
        # Dtype check (bool/int32/int64/float32)
        _dtype_map = {
            torch.bool: "bool",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.float32: "float32",
        }
        dtype_name = _dtype_map.get(tensor.dtype, str(tensor.dtype))
        if dtype_name != spec.dtype:
            raise ContractError(f"field {name}: dtype {dtype_name} != spec {spec.dtype}")
        # Shape check: first dim B must match batch size, rest must match spec tail.
        batch_size = batch.actor_seats.shape[0]
        seq_len = batch.history_mask.shape[1]
        expected_shape = tuple(
            batch_size
            if dim == "B"
            else seq_len
            if dim == "T"
            else BASELINE_ACTION_COUNT
            if dim == "A"
            else dim
            for dim in spec.shape
        )
        if tensor.dim() != len(spec.shape):
            raise ContractError(f"field {name}: rank {tensor.dim()} != spec {spec.shape}")
        if tensor.shape != expected_shape:
            raise ContractError(
                f"field {name}: shape {tuple(tensor.shape)} != spec {expected_shape}"
            )
        # Validate unmasked values when mask_field applies — skip padding positions.
        if spec.mask_field is not None:
            mask = batch.features.get(spec.mask_field, batch.history_mask)
            # mask True => participate; validate only those positions.
            if spec.valid_min is not None or spec.valid_max is not None:
                valid_mask = mask
                # Range check applies only where mask shape == tensor shape.
                if valid_mask.shape == tensor.shape:
                    values = tensor[valid_mask]
                    if values.numel() > 0:
                        if spec.valid_min is not None:
                            sync_checks.append(
                                (
                                    f"field {name} below valid_min",
                                    (values < spec.valid_min).any().reshape(()),
                                )
                            )
                        if spec.valid_max is not None:
                            sync_checks.append(
                                (
                                    f"field {name} above valid_max",
                                    (values > spec.valid_max).any().reshape(()),
                                )
                            )
    # Legal mask at least one true per row (fail-scalar, same single sync).
    sync_checks.append(
        (
            "legal_mask must have at least one True per batch row",
            (~batch.legal_mask.any(dim=1).all()).reshape(()),
        )
    )
    # Single host sync for every range/legal check on this path (cap: 3
    # scalars on the frozen schema — min, max, legal — one tolist total).
    if len(sync_checks) != 0:
        failed: list[bool] = torch.stack([flag for _, flag in sync_checks]).tolist()
        for (message, _), is_failed in zip(sync_checks, failed, strict=True):
            if is_failed:
                raise ContractError(message)
    # History mask shape must match history_event_kind.
    if batch.history_mask.shape != batch.features["history_event_kind"].shape:
        raise ContractError("history_mask shape must match history_event_kind")
    # Observation hashes length matches batch.
    if len(batch.observation_hashes) != batch.actor_seats.shape[0]:
        raise ContractError("observation_hashes length mismatch")


def input_schema_hash() -> DigestText:
    return model_input_schema_digest()
