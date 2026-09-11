"""Dataset -> train-ready shard feed: correctness + speed bench (<20s budget).

Pipeline under test: builder-envelope tensor shard (``manifest.json`` + Arrow-IPC
planes + ``decision_ids.json`` sidecar) -> :class:`ShardReader` mmap feed ->
GPU-consumable torch batches.

Unit choice: **rows/s is primary, events/s is secondary**.
One row = one decision = one loss sample = one optimizer step-item, so rows/s
is stable across corpus mixes and predicts step time. events/s (= rows/s x
mean valid history length) inflates with longer histories and bucket padding
without adding training samples, so it is reported only as a diagnostic.

events/s here counts VALID history events (``history_mask`` trues), never
padded zeros.
"""

from __future__ import annotations

import hashlib
import json
import time
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
import torch

from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training.shard_reader import ShardReader, dataset_hash_of

if TYPE_CHECKING:
    from pathlib import Path

T_WIDTH = 256
BUCKETS = (32, 64, 128, 256)
BENCH_ROWS = 2048
BENCH_BATCH = 256
BENCH_TIMED_EPOCHS = 3
BUDGET_S = 20.0
SCHEMA_DIGEST = "sha256:" + "22" * 32


def _lengths(rows: int) -> list[int]:
    """Deterministic mix across all four buckets: 10/40/100/200 -> 32/64/128/256."""
    cyc = (10, 40, 100, 200)
    return [cyc[r % 4] for r in range(rows)]


def _write_ipc(path: Path, arr: np.ndarray) -> None:
    with pa.OSFile(str(path), "wb") as handle:
        pa_ipc.write_tensor(pa.Tensor.from_numpy(np.ascontiguousarray(arr)), handle)


def write_feed_shard(shard_dir: Path, *, rows: int) -> dict[str, Any]:
    """Write a builder-envelope shard with full-width legal_mask; no encoder/Rust."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    lengths = _lengths(rows)
    kind = np.zeros((rows, T_WIDTH), dtype=np.int64)
    mask = np.zeros((rows, T_WIDTH), dtype=np.bool_)
    for r, ln in enumerate(lengths):
        kind[r, :ln] = (r + np.arange(ln)) % 20 + 1
        mask[r, :ln] = True
    rng = np.random.default_rng(1234)
    legal = rng.random((rows, BASELINE_ACTION_COUNT)) < 0.05
    legal[np.arange(rows), (np.arange(rows) * 7) % BASELINE_ACTION_COUNT] = True
    arrays: dict[str, np.ndarray] = {
        "history_event_kind": kind,
        "history_mask": mask,
        "history_len": np.asarray(lengths, dtype=np.int64),
        "actor_seats": (np.arange(rows) % 4).astype(np.int64),
        "chosen_action_id": ((np.arange(rows) * 7) % 16).astype(np.int64),
        "scores": (np.arange(rows * 4).reshape(rows, 4) % 1000).astype(np.int32),
        "legal_mask": np.ascontiguousarray(legal),
    }
    ids = [f"feed-{r:08d}" for r in range(rows)]
    manifest_planes: list[dict[str, Any]] = []
    for name, arr in arrays.items():
        storage = np.ascontiguousarray(arr.astype(np.uint8)) if arr.dtype == np.bool_ else arr
        logical = "bool" if arr.dtype == np.bool_ else str(arr.dtype)
        filename = f"{name}.arrow.ipc"
        _write_ipc(shard_dir / filename, storage)
        manifest_planes.append(
            {
                "name": name,
                "dtype": logical,
                "storage_dtype": str(storage.dtype),
                "shape": list(arr.shape),
                "file": filename,
                "sha256": hashlib.sha256((shard_dir / filename).read_bytes()).hexdigest(),
                "chunk": 0,
            }
        )
    (shard_dir / "decision_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (shard_dir / "observation_hashes.json").write_text(
        json.dumps(["sha256:" + f"{r:064x}" for r in range(rows)]), encoding="utf-8"
    )
    manifest = {
        "version": 1,
        "planes": manifest_planes,
        "rows": rows,
        "schema_digest": SCHEMA_DIGEST,
        "dataset_hash": dataset_hash_of(ids),
        "split": "train",
        "order": {"kind": "canonical"},
        "attestation": {"mode": "SYNTHETIC_ATTESTATION", "suite": "shard-feed-bench"},
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest": manifest, "ids": ids, "lengths": lengths}


def _drain_epoch(reader: ShardReader) -> list[dict[str, Any]]:
    return list(reader.iter_epoch())


def test_shard_feed_correctness_gpu_ready(tmp_path: Path) -> None:
    """One epoch delivers every row exactly once as contiguous GPU-ready tensors."""
    rows = 512
    fixed = write_feed_shard(tmp_path / "shard", rows=rows)
    want_len = dict(zip(fixed["ids"], fixed["lengths"], strict=True))
    with ShardReader(tmp_path / "shard", batch_size=64) as reader:
        assert reader.dataset_hash == fixed["manifest"]["dataset_hash"]
        assert len(reader) == rows
        batches = _drain_epoch(reader)
    seen: list[str] = [did for b in batches for did in b["_decision_ids"]]
    assert sorted(seen) == sorted(fixed["ids"])
    assert len(set(seen)) == rows
    for batch in batches:
        bucket = int(batch["_bucket"])
        assert bucket in BUCKETS
        for name, val in batch.items():
            if name.startswith("_"):
                continue
            assert isinstance(val, torch.Tensor)
            assert val.is_contiguous()
        kind = batch["history_event_kind"]
        mask = batch["history_mask"]
        assert kind.shape == mask.shape and mask.dtype == torch.bool
        assert kind.shape[1] == bucket  # bucket-homogeneous, sliced to bucket width
        assert batch["legal_mask"].shape[1] == BASELINE_ACTION_COUNT
        assert bool((batch["legal_mask"].sum(dim=1) >= 1).all())
        for i, did in enumerate(batch["_decision_ids"]):
            true_len = int(mask[i].sum())
            assert true_len == want_len[did]  # mask sums recover per-row history_len
            assert int(kind[i, true_len:].abs().sum()) == 0  # padding is exact zeros
        scores = batch["scores"]
        assert scores.dtype == torch.int32 and scores.shape[1] == 4
        _ = scores.float().mean()  # training-like consumer op runs on the batch


def test_shard_feed_bench_baseline(tmp_path: Path) -> None:
    """Baseline feed throughput: warmup + 3 timed epochs, primary unit rows/s."""
    fixed = write_feed_shard(tmp_path / "shard", rows=BENCH_ROWS)
    events_per_epoch = int(sum(fixed["lengths"]))
    wall0 = time.perf_counter()
    with ShardReader(tmp_path / "shard", batch_size=BENCH_BATCH) as reader:
        _ = _drain_epoch(reader)  # warmup: producer start + page-in, discarded
        per_epoch_s: list[float] = []
        for _ in range(BENCH_TIMED_EPOCHS):
            t0 = time.perf_counter()
            batches = _drain_epoch(reader)
            per_epoch_s.append(time.perf_counter() - t0)
            assert sum(len(b["_decision_ids"]) for b in batches) == BENCH_ROWS
    wall = time.perf_counter() - wall0
    rows_rates = [BENCH_ROWS / s for s in per_epoch_s]
    events_rates = [events_per_epoch / s for s in per_epoch_s]
    med_rows, med_events = median(rows_rates), median(events_rates)
    mean_len = events_per_epoch / BENCH_ROWS
    # Consistency: events/s must equal rows/s x mean valid history length.
    assert med_events == pytest.approx(med_rows * mean_len, rel=1e-9)
    print(
        f"\nFEED_BENCH rows={BENCH_ROWS} batch={BENCH_BATCH} "
        f"epochs={BENCH_TIMED_EPOCHS} mean_len={mean_len:.1f} "
        f"rows/s={med_rows:.1f} events/s={med_events:.1f} "
        f"ms/epoch={median(per_epoch_s) * 1e3:.1f} wall={wall:.2f}s "
        f"(primary unit: rows/s)"
    )
    assert wall < BUDGET_S
