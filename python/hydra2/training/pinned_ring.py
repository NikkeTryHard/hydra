"""Caller-owned pinned host ring with async H2D over one transfer stream.

Phase-3 feed stage (perf-max-throughput plan appendix C): producer fills
caller-owned ``torch.empty(pin_memory=True)`` slots shaped FROM
:mod:`hydra2.models.schema` field order (never hand-duplicated), the consumer
moves them with ``.to(cuda, non_blocking=True)`` on a single transfer stream,
and slot recycling is guarded by one CUDA event per slot (record on consume,
host ``synchronize`` before refill). No global ``torch.cuda.synchronize`` and
no ``.item``/``.cpu``/tensor-print in the transfer path.

Public surface (ring API contract; telemetry hooks consume read-only):

- :func:`slot_shapes` / :func:`slot_layout` / :func:`slot_nbytes` /
  :func:`ring_nbytes` — schema-derived geometry plus buffer math.
- :class:`PinnedRing` with ``open(shapes)`` / ``next(cpu_batch)`` /
  ``stats()`` / ``close()``.

Fail-closed rules: unpinnable allocation, CUDA requested but unavailable,
shape/dtype/key mismatch, and use-after-close all raise
:class:`~hydra2.contracts.common.ContractError` — never a silent pageable or
synchronous fallback, except the explicit CPU path (``device="cpu"``) used by
tests and CPU-only debug, which keeps identical call order with dummy events.

Evidence: https://pytorch.org/docs/stable/data.html (memory pinning),
https://docs.pytorch.org/docs/stable/notes/cuda.html (streams/events),
https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html

Bridge delegation (Phase 5): the bridge ``ring`` owns slots *logically*
(cursor/slot_used/depth + 512-window timings via ``PyRing``); torch keeps
owning them *physically* (pinned tensors, streams, events — no Rust GPU
math, copies ordered vs the consumer stream only). :func:`bucket_for_length`
is the single Rust fn both sides call (Python pure re-export with local
ceil fallback). Module import probes ``hydra2._native.ring``; when the
bridge is absent (or exposes no ``ring``), every call falls back to the
verbatim Python path below, counted in ``stats()["bridge_fallbacks"]``.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Self

import torch

from hydra2.contracts.common import ContractError
from hydra2.models.schema import _BASELINE_FIELDS, BASELINE_ACTION_COUNT
from hydra2.training._ring_events import _CpuEvent as _CpuEvent
from hydra2.training._ring_events import _percentile as _percentile

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hydra2.models.schema import TensorFieldSpec


__all__ = [
    "PinnedRing",
    "resolve_layout",
    "ring_nbytes",
    "slot_dtypes",
    "slot_layout",
    "slot_nbytes",
    "slot_shapes",
]

# Depth 1 cannot overlap copy with compute (refill would wait on the single
# in-flight slot every step); the plan sizes the ring at depth >= 2.
_MIN_DEPTH = 2
# Bounded timing windows for p50/p99 in stats(); 512 microbatches is enough
# for stable percentiles without growing memory with run length.
_TIMING_WINDOW = 512


_BRIDGE_RING: Any = None
_BRIDGE_RING_PROBED = False


def _bridge_ring() -> Any | None:
    """``hydra2._native.ring`` submodule or ``None`` (never raises).

    Any failure (not built, wrong shape, no ``ring`` attribute) returns
    ``None`` and the caller keeps the verbatim Python path.
    """
    global _BRIDGE_RING, _BRIDGE_RING_PROBED
    if _BRIDGE_RING_PROBED:
        return _BRIDGE_RING
    _BRIDGE_RING_PROBED = True
    try:
        import importlib as _importlib

        sub = getattr(_importlib.import_module("hydra2._native"), "ring", None)
        if sub is not None:
            _BRIDGE_RING = sub
            return _BRIDGE_RING
    except Exception:
        pass
    _BRIDGE_RING = None
    return None


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
}

ShapeSpec = tuple[int, ...] | list[int]
LayoutSpec = tuple[ShapeSpec, torch.dtype | str] | ShapeSpec


def _schema_fields(fields: tuple[TensorFieldSpec, ...] | None) -> tuple[TensorFieldSpec, ...]:
    return _BASELINE_FIELDS if fields is None else fields


def _resolve_dim(
    dim: str | int,
    *,
    batch_size: int,
    seq_len: int,
    action_count: int,
    extra_dims: Mapping[str, int] | None,
) -> int:
    if isinstance(dim, int):
        if dim <= 0:
            raise ContractError(f"schema dim {dim!r} must be positive")
        return dim
    defaults = {"B": batch_size, "T": seq_len, "A": action_count}
    if dim in defaults:
        return defaults[dim]
    if extra_dims is not None and dim in extra_dims:
        value = extra_dims[dim]
        if value <= 0:
            raise ContractError(f"extra dim {dim!r} must be positive")
        return value
    resolved_extra = extra_dims if extra_dims is not None else {}
    raise ContractError(f"shape symbol {dim!r} needs extra_dims={dict(resolved_extra)!r}")


def slot_shapes(
    batch_size: int,
    seq_len: int,
    *,
    action_count: int = BASELINE_ACTION_COUNT,
    fields: tuple[TensorFieldSpec, ...] | None = None,
    extra_dims: Mapping[str, int] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Concrete per-field shapes in canonical schema field order.

    ``B``/``T``/``A`` resolve to ``batch_size``/``seq_len``/``action_count``;
    production passes the batch bucket length as ``seq_len``.
    """
    if batch_size < 1:
        raise ContractError(f"batch_size must be >= 1, got {batch_size}")
    if seq_len < 1:
        raise ContractError(f"seq_len must be >= 1, got {seq_len}")
    if action_count < 1:
        raise ContractError(f"action_count must be >= 1, got {action_count}")
    return {
        spec.name: tuple(
            _resolve_dim(
                dim,
                batch_size=batch_size,
                seq_len=seq_len,
                action_count=action_count,
                extra_dims=extra_dims,
            )
            for dim in spec.shape
        )
        for spec in _schema_fields(fields)
    }


def slot_dtypes(*, fields: tuple[TensorFieldSpec, ...] | None = None) -> dict[str, torch.dtype]:
    """Torch dtype per field, same order as :func:`slot_shapes`."""
    resolved: dict[str, torch.dtype] = {}
    for spec in _schema_fields(fields):
        dtype = _TORCH_DTYPES.get(spec.dtype)
        if dtype is None:
            raise ContractError(f"field {spec.name}: dtype {spec.dtype!r} unknown")
        resolved[spec.name] = dtype
    return resolved


def _checked_shape(name: str, dims: tuple[Any, ...]) -> tuple[int, ...]:
    if len(dims) == 0 or any(not isinstance(d, int) or d <= 0 for d in dims):
        raise ContractError(f"field {name}: shape {dims!r} must be positive ints")
    return tuple(dims)


def slot_layout(
    batch_size: int,
    seq_len: int,
    *,
    action_count: int = BASELINE_ACTION_COUNT,
    fields: tuple[TensorFieldSpec, ...] | None = None,
    extra_dims: Mapping[str, int] | None = None,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``(shape, dtype)`` per field — the canonical argument to :meth:`PinnedRing.open`."""
    shapes = slot_shapes(
        batch_size, seq_len, action_count=action_count, fields=fields, extra_dims=extra_dims
    )
    dtypes = slot_dtypes(fields=fields)
    return {name: (shapes[name], dtypes[name]) for name in shapes}


def resolve_layout(
    shapes: Mapping[str, LayoutSpec],
    *,
    fields: tuple[TensorFieldSpec, ...] | None = None,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Normalize an ``open()`` shapes argument to ``(shape, dtype)`` in schema order.

    Values are either ``(shape, dtype)`` pairs or bare shapes whose dtype is
    looked up from the schema by field name. Names must match the schema
    exactly (missing/extra is :class:`ContractError`, mirroring the encoder
    schema guard) so slots can never silently diverge from the model input.
    """
    specs = _schema_fields(fields)
    expected = [spec.name for spec in specs]
    missing = [name for name in expected if name not in shapes]
    extra = [name for name in shapes if name not in set(expected)]
    if len(missing) > 0 or len(extra) > 0:
        raise ContractError(f"ring layout mismatch missing={missing} extra={extra}")
    by_name = {spec.name: spec for spec in specs}
    layout: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name in expected:
        value = shapes[name]
        shape: tuple[int, ...] | None = None
        dtype: torch.dtype | None = None
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and isinstance(value[0], (tuple, list))
            and isinstance(value[1], (torch.dtype, str))
        ):
            shape = _checked_shape(name, tuple(value[0]))
            raw_dtype: torch.dtype | str = value[1]
            if isinstance(raw_dtype, str):
                mapped = _TORCH_DTYPES.get(raw_dtype)
                if mapped is None:
                    raise ContractError(f"field {name}: dtype {raw_dtype!r} unknown")
                dtype = mapped
            else:
                dtype = raw_dtype
        else:
            bare = value if isinstance(value, (tuple, list)) else (value,)
            shape = _checked_shape(name, tuple(bare))
            resolved = _TORCH_DTYPES.get(by_name[name].dtype)
            if resolved is None:
                raise ContractError(f"field {name}: dtype {by_name[name].dtype!r} unknown")
            dtype = resolved
        layout[name] = (shape, dtype)
    return layout


def slot_nbytes(layout: Mapping[str, tuple[Sequence[int], torch.dtype]]) -> int:
    """Host bytes of one slot (legal_mask bool ``[B,6792]`` dominates)."""
    total = 0
    for shape, dtype in layout.values():
        numel = 1
        for dim in shape:
            numel *= dim
        total += numel * torch.tensor([], dtype=dtype).element_size()
    return total


def ring_nbytes(layout: Mapping[str, tuple[Sequence[int], torch.dtype]], depth: int) -> int:
    """Host bytes of the full ring (``depth`` slots)."""
    if depth < _MIN_DEPTH:
        raise ContractError(f"ring depth must be >= {_MIN_DEPTH}, got {depth}")
    return slot_nbytes(layout) * depth


class PinnedRing:
    """Depth-N pinned host ring with single-stream async H2D.

    Lifecycle: ``ring = PinnedRing.open(shapes, depth=...)`` then per
    microbatch ``cuda_batch = ring.next(cpu_batch)``; ``ring.stats()`` feeds
    the JSONL telemetry hooks; ``ring.close()`` tears down (idempotent, also
    usable as a context manager). Instances are single-lifecycle: after
    ``close()`` the ring never reopens.

    ``next()`` performs synchronize-before-refill on the target slot, copies
    CPU into the pinned slot, issues ``.to(device, non_blocking=True)`` on the
    transfer stream, records the consume event, and orders the calling
    thread's current CUDA stream after the transfer (host never blocks past
    the recycled slot's own event). True device-side transfer milliseconds
    come from per-slot timing events read back on the next reuse of that
    slot — no extra synchronization. NOT thread-safe; serialize ``next()``.
    """

    def __init__(
        self,
        layout: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        depth: int = _MIN_DEPTH,
        device: str | torch.device = "cuda",
        transfer_stream: Any = None,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if depth < _MIN_DEPTH:
            raise ContractError(f"ring depth must be >= {_MIN_DEPTH}, got {depth}")
        if isinstance(device, torch.device):
            resolved_device = device
        else:
            try:
                resolved_device = torch.device(device)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ContractError(f"ring device {device!r} invalid: {exc}") from exc
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ContractError("ring requested CUDA but torch.cuda.is_available() is False")
        if len(layout) == 0:
            raise ContractError("ring layout must list at least one field")
        for name, (shape, _dtype) in layout.items():
            if len(shape) == 0 or any(d <= 0 for d in shape):
                raise ContractError(f"field {name}: shape {shape!r} must be positive ints")

        self._field_names = list(layout.keys())
        self._shapes = {name: layout[name][0] for name in self._field_names}
        self._dtypes = {name: layout[name][1] for name in self._field_names}
        self._depth = depth
        self._device = resolved_device
        self._cuda = resolved_device.type == "cuda"
        self._slot_nbytes = slot_nbytes(layout)
        self._ring_nbytes = self._slot_nbytes * depth

        try:
            self._slots = [
                {
                    name: torch.empty(self._shapes[name], dtype=self._dtypes[name], pin_memory=True)
                    for name in self._field_names
                }
                for _ in range(depth)
            ]
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            raise ContractError(f"pinned ring alloc failed (fail closed): {exc}") from exc
        unpinned = sorted(
            {name for slot in self._slots for name, t in slot.items() if not t.is_pinned()}
        )
        if len(unpinned) > 0:
            self._slots = []
            raise ContractError(f"pinned ring unpinnable for {unpinned} (fail closed)")

        if self._cuda:
            if transfer_stream is None:
                try:
                    self._stream: Any = torch.cuda.Stream(device=self._device)
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise ContractError(f"transfer stream alloc failed: {exc}") from exc
            else:
                self._stream = transfer_stream
        else:
            self._stream = None
        self._real_stream = self._stream is not None and isinstance(self._stream, torch.cuda.Stream)

        def _make_event(index: int, kind: str) -> Any:
            if event_factory is not None:
                try:
                    return event_factory()
                except Exception as exc:
                    raise ContractError(f"ring event factory failed: {exc}") from exc
            if self._cuda:
                return torch.cuda.Event(enable_timing=True)
            return _CpuEvent(f"slot-{index}-{kind}")

        try:
            self._consume = [_make_event(i, "consume") for i in range(depth)]
            self._start = [_make_event(i, "start") for i in range(depth)]
            self._end = [_make_event(i, "end") for i in range(depth)]
        except (RuntimeError, TypeError) as exc:
            raise ContractError(f"ring event alloc failed: {exc}") from exc
        self._real_events = len(self._consume) > 0 and isinstance(
            self._consume[0], torch.cuda.Event
        )

        self._cursor = 0
        self._acquires = 0
        self._transfers = 0
        self._slot_used = [False] * depth
        self._h2d_ms_last: float | None = None
        self._h2d_ms = deque[float](maxlen=_TIMING_WINDOW)
        self._sync_ms_last: float | None = None
        self._sync_ms = deque[float](maxlen=_TIMING_WINDOW)
        self._open = True
        # Bridge-owned logical slots (cursor/slot_used/timings mirror).
        # Physical slots/streams/events stay torch-side above; any bridge
        # failure keeps the verbatim Python path (counted, never raised).
        self._bridge: Any | None = None
        self._bridge_fallbacks = 0
        try:
            bridge_mod = _bridge_ring()
            if bridge_mod is not None:
                self._bridge = bridge_mod.PyRing(depth)
        except Exception:
            self._bridge = None
            self._bridge_fallbacks += 1

    @classmethod
    def open(
        cls,
        shapes: Mapping[str, LayoutSpec],
        *,
        depth: int = _MIN_DEPTH,
        device: str | torch.device = "cuda",
        transfer_stream: Any = None,
        event_factory: Callable[[], Any] | None = None,
        fields: tuple[TensorFieldSpec, ...] | None = None,
    ) -> Self:
        """Allocate a ring from schema-ordered ``shapes`` (see :func:`resolve_layout`)."""
        return cls(
            resolve_layout(shapes, fields=fields),
            depth=depth,
            device=device,
            transfer_stream=transfer_stream,
            event_factory=event_factory,
        )

    @property
    def depth(self) -> int:
        """Ring depth (slots)."""
        return self._depth

    @property
    def device(self) -> torch.device:
        """Transfer target device."""
        return self._device

    @property
    def field_names(self) -> list[str]:
        """Schema-ordered field names held per slot."""
        return list(self._field_names)

    def next(  # reason: ring API contract is open/next/stats/close; builtin next unrelated here
        self, cpu_batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Recycle one slot through refill + async H2D; returns device tensors in schema order."""
        if not self._open:
            raise ContractError("ring is closed")
        missing = [name for name in self._field_names if name not in cpu_batch]
        extra = sorted(set(cpu_batch) - set(self._field_names))
        if len(missing) > 0 or len(extra) > 0:
            raise ContractError(f"ring batch mismatch missing={missing} extra={extra}")
        for name in self._field_names:
            tensor = cpu_batch[name]
            if not isinstance(tensor, torch.Tensor):
                raise ContractError(f"field {name}: expected Tensor, got {type(tensor).__name__}")
            if tuple(tensor.shape) != self._shapes[name]:
                raise ContractError(
                    f"field {name}: shape {tuple(tensor.shape)} != slot {self._shapes[name]}"
                )
            if tensor.dtype != self._dtypes[name]:
                raise ContractError(
                    f"field {name}: dtype {tensor.dtype} != slot {self._dtypes[name]}"
                )

        index = self._cursor
        slot = self._slots[index]
        consume, start, end = self._consume[index], self._start[index], self._end[index]

        wait_began = time.perf_counter()
        consume.synchronize()
        sync_ms = (time.perf_counter() - wait_began) * 1000.0
        self._sync_ms_last = sync_ms
        self._sync_ms.append(sync_ms)

        if self._slot_used[index]:
            try:
                measured: float = start.elapsed_time(end)
            except (RuntimeError, TypeError, AttributeError) as exc:
                raise ContractError(f"ring transfer timer failed: {exc}") from exc
            self._h2d_ms_last = measured
            self._h2d_ms.append(measured)

        copy_began = time.perf_counter()
        for name in self._field_names:
            # intentionally discarded: in-place copy returns self
            _ = slot[name].copy_(cpu_batch[name])
        if self._cuda:
            stream_context = (
                torch.cuda.stream(self._stream) if self._real_stream else contextlib.nullcontext()
            )
            with stream_context:
                start.record()
                out = {
                    name: slot[name].to(self._device, non_blocking=True)
                    for name in self._field_names
                }
                end.record()
                consume.record()
            if self._real_events:
                torch.cuda.current_stream().wait_event(consume)
        else:
            out = {name: slot[name].clone() for name in self._field_names}
            cpu_ms = (time.perf_counter() - copy_began) * 1000.0
            self._h2d_ms_last = cpu_ms
            self._h2d_ms.append(cpu_ms)
            consume.record()

        self._slot_used[index] = True
        self._cursor = (index + 1) % self._depth
        self._acquires += 1
        self._transfers += 1
        if self._bridge is not None:
            # Ring owns the logical slot: advance its cursor and mirror the
            # measured timings. Any bridge failure falls back to the Python
            # counters above (counted, never raised, bytes already moved).
            try:
                advanced: int = self._bridge.advance()
                assert advanced == index, f"bridge cursor drift: {advanced} != {index}"
                if self._h2d_ms_last is not None:
                    self._bridge.record_h2d(self._h2d_ms_last)
                if self._sync_ms_last is not None:
                    self._bridge.record_sync(self._sync_ms_last)
            except Exception:
                self._bridge_fallbacks += 1
        return out

    def stats(self) -> dict[str, Any]:
        """Ring counters plus last/p50/p99 transfer and recycle-wait timings (all ``.get``-able)."""
        h2d = list(self._h2d_ms)
        sync = list(self._sync_ms)
        payload: dict[str, Any] = {
            "open": self._open,
            "device": str(self._device),
            "cuda_available": self._cuda,
            "depth": self._depth,
            "acquires": self._acquires,
            "transfers": self._transfers,
            "slot_nbytes": self._slot_nbytes,
            "ring_nbytes": self._ring_nbytes,
            "h2d_ms_last": self._h2d_ms_last,
            "h2d_ms_p50": _percentile(h2d, 0.50),
            "h2d_ms_p99": _percentile(h2d, 0.99),
            "sync_wait_ms_last": self._sync_ms_last,
            "sync_wait_ms_p50": _percentile(sync, 0.50),
            "sync_wait_ms_p99": _percentile(sync, 0.99),
            "fields": list(self._field_names),
            "ring_owner": "bridge" if self._bridge is not None else "python",
            "bridge_fallbacks": self._bridge_fallbacks,
        }
        if self._bridge is not None:
            # Best-effort bridge mirror (cursor/slot_used/p50/p99); any
            # failure keeps the Python counters above (counted, never raised).
            try:
                bridge_cursor: int = self._bridge.cursor()
                bridge_h2d_p50: float | None = self._bridge.h2d_p50()
                bridge_h2d_p99: float | None = self._bridge.h2d_p99()
                bridge_sync_p50: float | None = self._bridge.sync_p50()
                bridge_sync_p99: float | None = self._bridge.sync_p99()
                payload["bridge_cursor"] = bridge_cursor
                payload["bridge_h2d_p50"] = bridge_h2d_p50
                payload["bridge_h2d_p99"] = bridge_h2d_p99
                payload["bridge_sync_p50"] = bridge_sync_p50
                payload["bridge_sync_p99"] = bridge_sync_p99
            except Exception:
                self._bridge_fallbacks += 1
                payload["bridge_fallbacks"] = self._bridge_fallbacks
        return payload

    def close(self) -> None:
        """Release slots/events/stream refs; idempotent, single lifecycle (never reopens)."""
        if not self._open:
            return
        self._open = False
        self._slots = []
        self._consume = []
        self._start = []
        self._end = []
        self._stream = None
        self._bridge = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
