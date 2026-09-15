"""Phase-4 shard reader: builder-format mmap feed, order + resume gates.

Builds synthetic shards in the exact builder envelope (tensor-IPC planes,
uint8-backed bool, chunked entries, decision_ids/observation_hashes sidecars,
pinned dataset_hash join) plus a numpy-plane tolerance variant, and pins:
order-identical reads, deterministic seed perms, exact {epoch,
sample_in_epoch} resume, bucket-homogeneous batches with T-slicing, depth>=2
prefetch, and fail-closed verification. CPU-only, tmp_path fixtures.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.training.shard_reader import ShardReader, dataset_hash_of

T_WIDTH = 256
N_ROWS = 96
BATCH = 16
SEED = 7

SCHEMA_DIGEST = "sha256:" + "11" * 32


def _lengths(rows: int) -> list[int]:
    # Two history buckets: rows mix bucket-32 (len 10) and bucket-128 (len 100).
    return [10 if r % 2 == 0 else 100 for r in range(rows)]


def _source_arrays(rows: int) -> dict[str, np.ndarray]:
    lengths = _lengths(rows)
    kind = np.zeros((rows, T_WIDTH), dtype=np.int64)
    mask = np.zeros((rows, T_WIDTH), dtype=np.bool_)
    for r, ln in enumerate(lengths):
        kind[r, :ln] = (r + np.arange(ln)) % 20 + 1
        mask[r, :ln] = True
    return {
        "history_event_kind": kind,
        "history_mask": mask,
        "actor_seats": (np.arange(rows) % 4).astype(np.int64),
        "chosen_action_id": ((np.arange(rows) * 7) % 16).astype(np.int64),
        "scores": (np.arange(rows * 4).reshape(rows, 4) % 1000).astype(np.int32),
    }


def _storage_of(name: str, arr: np.ndarray) -> tuple[np.ndarray, str, str]:
    if arr.dtype == np.dtype(np.bool_):
        return np.ascontiguousarray(arr.astype(np.uint8)), "bool", "uint8"
    return np.ascontiguousarray(arr), str(arr.dtype), str(arr.dtype)


def _write_tensor_ipc(path: Path, arr: np.ndarray) -> None:
    with pa.OSFile(str(path), "wb") as handle:
        pa_ipc.write_tensor(pa.Tensor.from_numpy(np.ascontiguousarray(arr)), handle)


def write_shard(
    shard_dir: Path,
    *,
    backend: str = "ipc",
    order: str = "canonical",
    seed: int = SEED,
    chunks: int = 1,
    with_sidecars: bool = True,
    rows: int = N_ROWS,
    schema_digest: str = SCHEMA_DIGEST,
) -> dict[str, Any]:
    """Write a builder-envelope shard; returns manifest + source rows."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    arrays = _source_arrays(rows)
    ids = [f"dec-{r:08d}" for r in range(rows)]
    if order == "perm":
        gen = torch.Generator().manual_seed(seed)
        perm = [int(x) for x in torch.randperm(rows, generator=gen).tolist()]
        arrays = {name: arr[perm] for name, arr in arrays.items()}
        ids = [ids[i] for i in perm]
    manifest_planes: list[dict[str, Any]] = []
    for name, arr in arrays.items():
        storage, logical, physical = _storage_of(name, arr)
        bounds = [0, rows] if chunks == 1 else [0, rows // 2, rows]
        for chunk in range(chunks):
            piece = storage[bounds[chunk] : bounds[chunk + 1]]
            filename = f"{name}.arrow.ipc" if chunks == 1 else f"{name}.c{chunk}.arrow.ipc"
            if backend == "npy":
                filename = filename.replace(".arrow.ipc", ".npy")
                piece = arrays[name][bounds[chunk] : bounds[chunk + 1]]
                np.save(shard_dir / filename, np.ascontiguousarray(piece))
                shape = list(piece.shape)
            else:
                _write_tensor_ipc(shard_dir / filename, piece)
                shape = list(piece.shape)
            manifest_planes.append(
                {
                    "name": name,
                    "dtype": logical if backend == "ipc" else str(piece.dtype),
                    "storage_dtype": physical if backend == "ipc" else str(piece.dtype),
                    "shape": shape,
                    "file": filename,
                    "sha256": hashlib.sha256((shard_dir / filename).read_bytes()).hexdigest(),
                    "chunk": chunk,
                }
            )
    dataset_hash = dataset_hash_of(ids)
    if with_sidecars:
        (shard_dir / "decision_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (shard_dir / "observation_hashes.json").write_text(
            json.dumps(["sha256:" + f"{r:064x}" for r in range(rows)]), encoding="utf-8"
        )
    manifest = {
        "version": 1,
        "planes": manifest_planes,
        "rows": rows,
        "schema_digest": schema_digest,
        "dataset_hash": dataset_hash,
        "split": "train",
        "order": {"kind": order, **({"seed": seed} if order == "perm" else {})},
        "attestation": {"mode": "SYNTHETIC_ATTESTATION", "suite": "shard-reader-fixture"},
        "build": {
            "games": rows,
            "games_quarantined": 0,
            "rows": rows,
            "rows_per_s": 1000.0,
            "games_per_s": 1000.0,
            "mib_per_s": 1.0,
            "elapsed_s": 0.1,
            "chunk_rows": rows,
            "uncompressed_bytes": sum(a.nbytes for a in arrays.values()),
        },
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest": manifest, "ids": ids, "arrays": arrays}


def _collect(reader: ShardReader, n_batches: int) -> list[dict[str, Any]]:
    return [reader.next_batch() for _ in range(n_batches)]


def _ids_of(batches: list[dict[str, Any]]) -> list[str]:
    return [did for batch in batches for did in batch["_decision_ids"]]


def test_canonical_order_identical_full_width(tmp_path: Path) -> None:
    fixed = write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, bucket_batches=False) as reader:
        assert len(reader) == N_ROWS
        batches = list(reader.iter_epoch())
    assert _ids_of(batches) == fixed["ids"]
    seats = torch.cat([b["actor_seats"] for b in batches])
    assert torch.equal(seats, torch.from_numpy(fixed["arrays"]["actor_seats"]))
    assert batches[0]["history_mask"].shape == (BATCH, T_WIDTH)
    assert batches[0]["scores"].shape == (BATCH, 4)
    assert all(b["_bucket"] == -1 for b in batches)


def test_bucket_regroup_homogeneous_and_sliced(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH) as reader:
        batches = list(reader.iter_epoch())
    # Stable bucket sort of canonical: all len-10 (bucket 32) rows first.
    expected = [i for i in range(N_ROWS) if i % 2 == 0] + [i for i in range(N_ROWS) if i % 2 == 1]
    assert _ids_of(batches) == [f"dec-{i:08d}" for i in expected]
    for batch in batches:
        bucket = int(batch["_bucket"])
        assert bucket in (32, 128)
        assert batch["history_mask"].dtype == torch.bool
        assert batch["history_mask"].shape[1] == bucket
        assert batch["history_event_kind"].shape[1] == bucket
        assert batch["scores"].shape == (batch["scores"].shape[0], 4)
        assert batch["_sample_end"] - batch["_sample_start"] == batch["history_mask"].shape[0]
    # Mask content survives the T-slice: first batch rows have exactly 10 live.
    assert batches[0]["history_mask"].sum(dim=1).tolist() == [10] * BATCH


def test_seed_perm_deterministic(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as first:
        seq_a = _ids_of(_collect(first, first.batches_per_epoch))
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as second:
        seq_b = _ids_of(_collect(second, second.batches_per_epoch))
    assert seq_a == seq_b
    with ShardReader(tmp_path / "shard", batch_size=BATCH) as third:
        seq_c = _ids_of(_collect(third, third.batches_per_epoch))
    assert seq_a != seq_c


def test_cursor_resume_exact(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as full:
        n_batches = full.batches_per_epoch
        reference = _collect(full, n_batches)
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as head:
        prefix = _collect(head, 2)
        cursor = head.get_state()
    assert cursor == {
        "epoch": 0,
        "sample_in_epoch": prefix[-1]["_sample_end"],
        "seed": SEED,
        "rows": N_ROWS,
    }
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as tail:
        tail.set_state(cursor)
        rest = _collect(tail, n_batches - 2)
    assert _ids_of(prefix + rest) == _ids_of(reference)
    for got, want in zip(prefix + rest, reference, strict=True):
        assert torch.equal(got["chosen_action_id"], want["chosen_action_id"])
        assert torch.equal(got["history_mask"], want["history_mask"])
    # Epoch wrap: consuming past the edge advances the epoch without replay.
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as reader:
        last = _collect(reader, n_batches)[-1]
        assert last["_sample_end"] == N_ROWS
        nxt = reader.next_batch()
        assert (nxt["_epoch"], nxt["_sample_start"]) == (1, 0)
        assert reader.get_state()["epoch"] == 1


def test_cursor_seed_mismatch_rejected(tmp_path: Path) -> None:
    """A cursor from another seed never seeks silently (fail-closed re-permute).

    Guards ShardReader.set_state's seed guard: swapping the recorded seed
    must raise instead of reading the wrong permutation as if it were the
    requested epoch. A regression that ignores the seed returns rows here.
    """
    write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as head:
        _ = _collect(head, 2)
        cursor = head.get_state()
    assert cursor["seed"] == SEED
    tampered = dict(cursor)
    tampered["seed"] = SEED + 1
    with (
        ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as tail,
        pytest.raises(ContractError),
    ):
        tail.set_state(tampered)


def test_cursor_out_of_range_rejected(tmp_path: Path) -> None:
    """A cursor past the row count never seeks (fail-closed bounds check).

    Guards set_state's ``0 <= sample_in_epoch <= rows`` gate: a torn cursor
    (sample past the end) must raise instead of starting the producer past
    the shard edge. A regression that clamps silently accepts here.
    """
    write_shard(tmp_path / "shard", backend="ipc")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as head:
        _ = _collect(head, 2)
        cursor = head.get_state()
    tampered = dict(cursor)
    tampered["sample_in_epoch"] = N_ROWS + 1
    with (
        ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED) as tail,
        pytest.raises(ContractError),
    ):
        tail.set_state(tampered)


def test_perm_manifest_reads_recorded_perm_and_reseeds(tmp_path: Path) -> None:
    fixed = write_shard(tmp_path / "shard", backend="ipc", order="perm", seed=SEED)
    with ShardReader(tmp_path / "shard", batch_size=BATCH, bucket_batches=False) as reader:
        assert _ids_of(list(reader.iter_epoch())) == fixed["ids"]
    with ShardReader(
        tmp_path / "shard", batch_size=BATCH, seed=SEED, bucket_batches=False
    ) as reader:
        assert _ids_of(list(reader.iter_epoch())) == fixed["ids"]
    # A new seed recovers canonical from the sidecar, then permutes.
    other = SEED + 1
    gen = torch.Generator().manual_seed(other)
    perm = [int(x) for x in torch.randperm(N_ROWS, generator=gen).tolist()]
    canon_ids = sorted(fixed["ids"])
    with ShardReader(
        tmp_path / "shard", batch_size=BATCH, seed=other, bucket_batches=False
    ) as reader:
        assert _ids_of(list(reader.iter_epoch())) == [canon_ids[i] for i in perm]


def test_perm_manifest_reseed_without_sidecar_fails(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc", order="perm", seed=SEED, with_sidecars=False)
    with pytest.raises(ContractError):
        ShardReader(tmp_path / "shard", batch_size=BATCH, seed=SEED + 1)


def test_prefetch_depth_floor_and_live_producer(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    with pytest.raises(ContractError):
        ShardReader(tmp_path / "shard", batch_size=BATCH, prefetch_depth=1)
    with ShardReader(tmp_path / "shard", batch_size=BATCH) as reader:
        assert reader.queue_depth >= 2
        reader.next_batch()
        assert reader._producer is not None and reader._producer.is_alive()


def test_corrupt_plane_byte_fails_closed(tmp_path: Path) -> None:
    fixed = write_shard(tmp_path / "shard", backend="ipc")
    target = tmp_path / "shard" / fixed["manifest"]["planes"][0]["file"]
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    target.write_bytes(bytes(raw))
    with pytest.raises(CorruptArtifactError):
        ShardReader(tmp_path / "shard", batch_size=BATCH)


def test_tampered_sidecar_breaks_dataset_hash(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    sidecar = tmp_path / "shard" / "decision_ids.json"
    ids = json.loads(sidecar.read_text(encoding="utf-8"))
    ids[0] = "dec-99999999"
    sidecar.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(CorruptArtifactError):
        ShardReader(tmp_path / "shard", batch_size=BATCH)


def test_schema_digest_gate(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    with pytest.raises(ContractError):
        ShardReader(
            tmp_path / "shard", batch_size=BATCH, expect_schema_digest="sha256:" + "ff" * 32
        )
    with ShardReader(
        tmp_path / "shard", batch_size=BATCH, expect_schema_digest=SCHEMA_DIGEST
    ) as reader:
        assert reader.schema_digest == SCHEMA_DIGEST
        assert reader.dataset_hash.startswith("sha256:")
        assert reader.split == "train"
        assert reader.manifest["order"] == {"kind": "canonical"}


def test_undeclared_narrow_width_rejected(tmp_path: Path) -> None:
    """A manifest action_width below baseline needs allow_narrow to open."""
    fixed = write_shard(tmp_path / "shard", backend="ipc")
    assert "action_width" not in fixed["manifest"]
    manifest = dict(fixed["manifest"])
    manifest["action_width"] = 16
    (tmp_path / "shard" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError):
        ShardReader(tmp_path / "shard", batch_size=BATCH, bucket_batches=False)
    with ShardReader(
        tmp_path / "shard", batch_size=BATCH, bucket_batches=False, allow_narrow=True
    ) as reader:
        assert reader.manifest["action_width"] == 16
        batch = reader.next_batch()
        assert batch["_sample_end"] > 0


def test_npy_backend_reads_identical(tmp_path: Path) -> None:
    fixed = write_shard(tmp_path / "shard", backend="npy")
    with ShardReader(tmp_path / "shard", batch_size=BATCH, bucket_batches=False) as reader:
        batches = list(reader.iter_epoch())
    assert _ids_of(batches) == fixed["ids"]
    seats = torch.cat([b["actor_seats"] for b in batches])
    assert torch.equal(seats, torch.from_numpy(fixed["arrays"]["actor_seats"]))
    assert batches[0]["history_mask"].dtype == torch.bool


def test_chunked_plane_concatenates_across_edge(tmp_path: Path) -> None:
    fixed = write_shard(tmp_path / "shard", backend="ipc", chunks=2)
    with ShardReader(tmp_path / "shard", batch_size=BATCH, bucket_batches=False) as reader:
        batches = list(reader.iter_epoch())
    assert _ids_of(batches) == fixed["ids"]
    for name in ("history_event_kind", "history_mask", "actor_seats", "chosen_action_id", "scores"):
        got = torch.cat([b[name] for b in batches])
        assert torch.equal(got, torch.from_numpy(np.ascontiguousarray(fixed["arrays"][name])))


def test_feed_leaves_files_untouched(tmp_path: Path) -> None:
    write_shard(tmp_path / "shard", backend="ipc")
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((tmp_path / "shard").iterdir())
    }
    with ShardReader(tmp_path / "shard", batch_size=BATCH) as reader:
        list(reader.iter_epoch())
        list(reader.iter_epoch())
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((tmp_path / "shard").iterdir())
    }
    assert before == after
