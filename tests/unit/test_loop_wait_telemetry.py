"""Phase-3 wait-telemetry hooks on SupervisedLoop (JSONL + p50/p99).

Covers the telemetry-only surface of the training loop (``loop_batch``
per-microbatch JSONL rows (queue_wait_ms, fetch_decode_ms, h2d_ms,
compute_ms), the trailing p50/p99 summary row, the in-memory
``telemetry_summary``, opt-out by default, and duck-typed consumption of
the sibling ring contract (``next``/``stats`` — PROVISIONAL, no import of
``hydra2.training.pinned_ring``). CPU lane, fixed seeds; a fake dataset
keeps the smoke run free of parquet IO.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hydra2.training.loop_batch import (
    MicrobatchTelemetry,
    summarize_telemetry,
)
from hydra2.training.loop_state import TrainingLoopConfig
from hydra2.training.loop_train import SupervisedLoop
from tests.unit._manifest_helpers import make_test_manifest_hashes

pytestmark = pytest.mark.contract_package("WP-05B")

FEATURE_DIM = 8
NUM_ACTIONS = 4


class _StubPolicyModel(nn.Module):
    """Minimal policy model over synthetic features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIM, NUM_ACTIONS)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"policy_logits": self.linear(batch["features"])}


class _FakeDataset:
    """Deterministic in-memory batches (no parquet IO)."""

    def __init__(self, *, seed: int = 0) -> None:
        self._gen = torch.Generator().manual_seed(seed)
        self._offset = 0

    def next_batch(self, n: int) -> dict[str, torch.Tensor]:
        batch = {
            "features": torch.randn(n, FEATURE_DIM, generator=self._gen),
            "legal_mask": torch.ones(n, NUM_ACTIONS, dtype=torch.bool),
            "chosen_action_id": torch.randint(0, NUM_ACTIONS, (n,), generator=self._gen),
        }
        self._offset += n
        return batch

    def get_sampler_state(self) -> dict[str, Any]:
        return {"offset": self._offset, "seed": 0, "total": 0, "epoch": 0}

    def set_sampler_state(self, state: Any) -> None:
        if isinstance(state, dict) and isinstance(state.get("offset"), int):
            self._offset = int(state["offset"])

    def __len__(self) -> int:
        return 0


class _FakeRingFeed:
    """Duck-typed stand-in for the sibling PinnedRing contract (no import)."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device)
        self.transfers = 0

    def next(self, cpu_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # The real ring returns device tensors; mirror that (CPU-noCUDA
        # fallback in the sibling contract is a sync copy, same order).
        self.transfers += 1
        return {
            key: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for key, value in cpu_batch.items()
        }

    def stats(self) -> dict[str, Any]:
        return {
            "depth": 2,
            "acquires": self.transfers,
            "transfers": self.transfers,
            "h2d_ms_last": 0.5,
            "slot_nbytes": 64,
            "ring_nbytes": 128,
            "device": str(self._device),
            "cuda_available": self._device.type == "cuda",
            "producer_wait_s": 0.25,
        }

    def close(self) -> None:
        return None


def _build_loop(
    tmp_path: Path,
    *,
    telemetry_path: Path | None = None,
    feed: Any | None = None,
    accumulation_steps: int = 2,
    max_updates: int = 2,
) -> SupervisedLoop:
    torch.manual_seed(0)
    model = _StubPolicyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainingLoopConfig(
        microbatch_size=4,
        accumulation_steps=accumulation_steps,
        gradient_clip_norm=None,
        max_updates=max_updates,
        checkpoint_frequency_updates=100,
        seed=0,
    )
    return SupervisedLoop(
        model=model,
        optimizer=optimizer,
        dataset=_FakeDataset(seed=0),
        config=config,
        checkpoint_dir=tmp_path / "checkpoints",
        manifest_hashes=make_test_manifest_hashes(),
        telemetry_path=telemetry_path,
        feed=feed,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row: dict[str, Any] = json.loads(line)
            rows.append(row)
    return rows


def test_second_train_truncates_telemetry_file(tmp_path: Path) -> None:
    """A fresh train() owns its telemetry file: stale rows never linger.

    Guards loop_train's truncate-on-train (write_text("")): a second train()
    must replace — never append to — the JSONL file, or resume/upload
    tooling would double-count microbatches. A regression that appends
    leaves 10 rows here instead of 5.
    """
    telemetry_path = tmp_path / "telemetry.jsonl"
    loop = _build_loop(tmp_path, telemetry_path=telemetry_path)
    loop.train()
    assert len(_read_jsonl(telemetry_path)) == 5
    loop.train()
    rows = _read_jsonl(telemetry_path)
    assert len(rows) == 5
    assert [row["kind"] for row in rows].count("microbatch") == 4
    assert [row["kind"] for row in rows].count("summary") == 1
    assert len(loop.telemetry_records) == 4


def test_telemetry_jsonl_smoke_run(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    loop = _build_loop(tmp_path, telemetry_path=telemetry_path)
    history = loop.train()
    assert len(history) == 2
    # 2 updates x 2 accumulation steps = 4 microbatch rows + 1 summary row.
    rows = _read_jsonl(telemetry_path)
    assert len(rows) == 5
    micro_rows = [row for row in rows if row["kind"] == "microbatch"]
    summary_rows = [row for row in rows if row["kind"] == "summary"]
    assert len(micro_rows) == 4
    assert len(summary_rows) == 1
    for row in micro_rows:
        for key in ("queue_wait_ms", "fetch_decode_ms", "h2d_ms", "compute_ms"):
            value = row[key]
            assert isinstance(value, (int, float)) and math.isfinite(float(value))
            assert float(value) >= 0.0
        # No queue exists without a ring feed, so the wait is exactly zero.
        assert float(row["queue_wait_ms"]) == 0.0
        assert float(row["producer_wait_s"]) == 0.0
    summary = summary_rows[0]
    assert summary["microbatches"] == 4
    for key in (
        "queue_wait_ms_p50",
        "queue_wait_ms_p99",
        "fetch_decode_ms_p50",
        "fetch_decode_ms_p99",
        "h2d_ms_p50",
        "h2d_ms_p99",
        "compute_ms_p50",
        "compute_ms_p99",
    ):
        assert math.isfinite(float(summary[key]))
        assert float(summary[key]) >= 0.0
    # In-memory summary matches the file summary exactly.
    assert loop.telemetry_summary() == {key: summary[key] for key in loop.telemetry_summary()}
    assert len(loop.telemetry_records) == 4


def test_summarize_telemetry_quantiles() -> None:
    records = [
        MicrobatchTelemetry(
            microstep=i,
            global_update=0,
            queue_wait_ms=float(value),
            fetch_decode_ms=0.0,
            h2d_ms=0.0,
            compute_ms=0.0,
        )
        for i, value in enumerate((3.0, 1.0, 4.0, 2.0))
    ]
    summary = summarize_telemetry(records)
    # Sorted [1,2,3,4]: p50 at pos 1.5 -> 2.5; p99 at pos 2.97 -> 3.97.
    assert summary["queue_wait_ms_p50"] == pytest.approx(2.5)
    assert summary["queue_wait_ms_p99"] == pytest.approx(3.97)
    assert summary["compute_ms_p50"] == pytest.approx(0.0)
    assert summarize_telemetry([]) == {}


def test_telemetry_disabled_by_default(tmp_path: Path) -> None:
    loop = _build_loop(tmp_path, accumulation_steps=1, max_updates=1)
    history = loop.train()
    assert len(history) == 1
    # Records are still collected in memory; no file is created.
    assert len(loop.telemetry_records) == 1
    assert loop.telemetry_path is None
    assert math.isfinite(loop.telemetry_summary()["compute_ms_p50"])


def test_ring_feed_duck_typed_contract(tmp_path: Path) -> None:
    feed = _FakeRingFeed(device="cuda" if torch.cuda.is_available() else "cpu")
    loop = _build_loop(
        tmp_path,
        telemetry_path=tmp_path / "telemetry.jsonl",
        feed=feed,
        accumulation_steps=1,
        max_updates=1,
    )
    loop.train()
    assert feed.transfers == 1
    (record,) = loop.telemetry_records
    # h2d comes from the ring's event-timed stat; the queue wait is the rest.
    assert record.h2d_ms == pytest.approx(0.5)
    assert record.queue_wait_ms >= 0.0
    assert record.producer_wait_s == pytest.approx(0.25)
