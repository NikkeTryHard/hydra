"""Pinned ring feed stage: schema-derived slots, event recycling, fail-closed.

Covers :mod:`hydra2.training.pinned_ring` only: slot geometry follows
``_BASELINE_FIELDS`` order (never hand-duplicated), ``next()`` keeps
record-on-consume / synchronize-before-refill order (fake-event log), the CPU
roundtrip preserves values, unpinnable allocation and mismatched batches raise
``ContractError``, and the B=1024/T=256 buffer math matches the plan budget.
Fast CPU lane except the single real-CUDA transfer test (``gpu`` mark).
"""

from __future__ import annotations

import unittest.mock as mock

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.models.schema import _BASELINE_FIELDS, BASELINE_ACTION_COUNT
from hydra2.training.pinned_ring import (
    PinnedRing,
    resolve_layout,
    ring_nbytes,
    slot_dtypes,
    slot_layout,
    slot_nbytes,
    slot_shapes,
)

_SMALL_B = 4
_SMALL_T = 8
_SMALL_A = 16


def _make_batch(layout: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in layout.items():
        numel = 1
        for dim in shape:
            numel *= dim
        base = torch.arange(numel).reshape(shape)
        if dtype is torch.bool:
            batch[name] = base % 2 == 0
        elif dtype is torch.float32:
            batch[name] = (base % 13).to(torch.float32) / 10.0
        else:
            batch[name] = (base % 7).to(dtype)
    return batch


def _assert_batch_equal(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> None:
    assert list(got) == list(want)
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert tuple(got[name].shape) == tuple(want[name].shape), name
        assert bool(torch.equal(got[name].cpu(), want[name].cpu())), name


class _FakeEvent:
    """Duck-typed event logging record/synchronize order for the factory protocol."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log

    def record(self, stream=None) -> None:
        del stream
        self._log.append(f"record:{self._label}")

    def synchronize(self) -> None:
        self._log.append(f"synchronize:{self._label}")

    def elapsed_time(self, other) -> float:
        del other
        return 0.0


def _make_factory(log: list[str], depth: int):
    """Label factory events by creation order: consume[0..d), start, end."""
    calls = {"n": 0}

    def factory():
        n = calls["n"]
        calls["n"] += 1
        if n < depth:
            label = f"slot-{n}-consume"
        elif n < 2 * depth:
            label = f"slot-{n - depth}-start"
        else:
            label = f"slot-{n - 2 * depth}-end"
        return _FakeEvent(label, log)

    return factory


def test_slot_shapes_follow_schema_field_order() -> None:
    shapes = slot_shapes(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    assert list(shapes) == [f.name for f in _BASELINE_FIELDS]
    assert shapes["legal_mask"] == (_SMALL_B, _SMALL_A)
    assert shapes["history_event_kind"] == (_SMALL_B, _SMALL_T)
    assert shapes["history_mask"] == (_SMALL_B, _SMALL_T)
    assert shapes["concealed_hand_counts"] == (_SMALL_B, 34)
    assert shapes["dora_indicators"] == (_SMALL_B, 5)
    assert shapes["actor"] == (_SMALL_B,)

    dtypes = slot_dtypes()
    assert list(dtypes) == [f.name for f in _BASELINE_FIELDS]
    assert dtypes["legal_mask"] is torch.bool
    assert dtypes["history_event_kind"] is torch.int64
    assert dtypes["concealed_hand_counts"] is torch.int32

    full = slot_shapes(2, 4)
    assert full["legal_mask"] == (2, BASELINE_ACTION_COUNT)

    # resolve_layout keeps schema order even when the input mapping is shuffled,
    # and accepts bare shapes (dtype from schema) as well as (shape, dtype) pairs.
    layout = resolve_layout(dict(reversed(list(slot_layout(2, 4).items()))))
    assert list(layout) == [f.name for f in _BASELINE_FIELDS]
    assert layout["legal_mask"] == ((2, BASELINE_ACTION_COUNT), torch.bool)


def test_open_next_stats_close_roundtrip_cpu() -> None:
    layout = slot_layout(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    batch_a = _make_batch(layout)
    batch_b = _make_batch(layout)
    for tensor in batch_b.values():
        if tensor.dtype is torch.bool:
            tensor.logical_not_()
        elif tensor.dtype is torch.float32:
            tensor.add_(1.0)
        else:
            tensor.add_(1)

    with PinnedRing.open(
        slot_shapes(_SMALL_B, _SMALL_T, action_count=_SMALL_A), device="cpu"
    ) as ring:
        assert ring.depth == 2
        assert ring.device == torch.device("cpu")
        assert ring.field_names == [f.name for f in _BASELINE_FIELDS]
        first = ring.next(batch_a)
        _assert_batch_equal(first, batch_a)
        second = ring.next(batch_b)
        _assert_batch_equal(second, batch_b)
        stats = ring.stats()
        assert stats["open"] is True
        assert stats["depth"] == 2
        assert stats["acquires"] == 2
        assert stats["transfers"] == 2
        assert stats["device"] == "cpu"
        assert stats["cuda_available"] is False
        assert stats["slot_nbytes"] == slot_nbytes(layout) > 0
        assert stats["ring_nbytes"] == 2 * stats["slot_nbytes"]
        assert isinstance(stats["h2d_ms_last"], float) and stats["h2d_ms_last"] >= 0.0
        assert isinstance(stats["h2d_ms_p50"], float)
        assert isinstance(stats["h2d_ms_p99"], float)
        assert stats["fields"] == [f.name for f in _BASELINE_FIELDS]
        ring.close()
        ring.close()  # idempotent
        with pytest.raises(ContractError):
            ring.next(batch_a)
    assert ring.stats()["open"] is False


def test_event_recycling_order() -> None:
    depth = 2
    log: list[str] = []
    layout = slot_layout(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    batches = [_make_batch(layout) for _ in range(3)]
    with PinnedRing.open(
        layout, depth=depth, device="cpu", event_factory=_make_factory(log, depth)
    ) as ring:
        outs = [ring.next(batch) for batch in batches]
    for out, want in zip(outs, batches, strict=True):
        _assert_batch_equal(out, want)
    # Depth 2 over 3 transfers: slot 0 is synchronized before its refill.
    assert log == [
        "synchronize:slot-0-consume",
        "record:slot-0-consume",
        "synchronize:slot-1-consume",
        "record:slot-1-consume",
        "synchronize:slot-0-consume",
        "record:slot-0-consume",
    ]


def test_next_rejects_mismatched_batch() -> None:
    layout = slot_layout(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    batch = _make_batch(layout)
    with PinnedRing.open(layout, device="cpu") as ring:
        bad_shape = dict(batch)
        bad_shape["actor"] = torch.zeros((_SMALL_B + 1,), dtype=torch.int64)
        with pytest.raises(ContractError):
            ring.next(bad_shape)
        bad_dtype = dict(batch)
        bad_dtype["actor"] = torch.zeros((_SMALL_B,), dtype=torch.int32)
        with pytest.raises(ContractError):
            ring.next(bad_dtype)
        with pytest.raises(ContractError):
            ring.next({k: v for k, v in batch.items() if k != "actor"})
        with pytest.raises(ContractError):
            ring.next({**batch, "privileged": torch.zeros(1)})


def test_fail_closed() -> None:
    layout = slot_layout(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    with pytest.raises(ContractError):
        PinnedRing.open(layout, depth=1, device="cpu")

    shapes = slot_shapes(_SMALL_B, _SMALL_T, action_count=_SMALL_A)
    dropped = {k: v for k, v in shapes.items() if k != "actor"}
    with pytest.raises(ContractError):
        PinnedRing.open(dropped, device="cpu")
    with pytest.raises(ContractError):
        PinnedRing.open({**shapes, "privileged": ((2,), torch.int64)}, device="cpu")

    def _boom(*args, **kwargs):
        raise RuntimeError("no pinned memory")

    with mock.patch.object(torch, "empty", side_effect=_boom), pytest.raises(ContractError):
        PinnedRing.open(layout, device="cpu")
    with (
        mock.patch.object(torch.Tensor, "is_pinned", lambda self: False),
        pytest.raises(ContractError),
    ):
        PinnedRing.open(layout, device="cpu")


def test_buffer_math_at_production_shape() -> None:
    layout = slot_layout(1024, 256)
    slot = slot_nbytes(layout)
    legal_shape, legal_dtype = layout["legal_mask"]
    assert legal_shape == (1024, BASELINE_ACTION_COUNT)
    assert legal_dtype is torch.bool
    legal_bytes = 1024 * BASELINE_ACTION_COUNT
    assert legal_bytes / slot > 0.70  # legal_mask dominates the slot
    assert 9.5e6 <= slot <= 10.0e6  # plan budget: slot ~9.8MB
    assert ring_nbytes(layout, 2) == 2 * slot  # ring ~20MB @depth2
    assert ring_nbytes(layout, 3) == 3 * slot
    with pytest.raises(ContractError):
        ring_nbytes(layout, 1)


@pytest.mark.gpu
def test_cuda_transfer_path() -> None:
    layout = slot_layout(2, 4, action_count=8)
    batch = _make_batch(layout)
    with PinnedRing.open(layout, depth=2, device="cuda") as ring:
        first = ring.next(batch)
        assert all(t.device.type == "cuda" for t in first.values())
        _assert_batch_equal(first, batch)
        second = ring.next(batch)
        _assert_batch_equal(second, batch)
        # Third transfer recycles slot 0, reading back its device-side H2D time.
        third = ring.next(batch)
        _assert_batch_equal(third, batch)
        stats = ring.stats()
        assert stats["transfers"] == 3
