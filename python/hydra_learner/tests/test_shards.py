from __future__ import annotations

import json
import struct
import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from hydra_learner.raw_mjai_stream import (
    RawMjaiBridgeStats,
    RawMjaiPinnedStream,
    build_raw_mjai_stream_command,
    decode_batch,
)
from hydra_learner.shards import (
    ACTION_SPACE,
    BC_BASE_RECORD_SIZE,
    BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE,
    BC_SHARD_LAYOUT_VERSION,
    BC_SHARD_MAGIC,
    BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION,
    COMPACT_OBS_BASELINE_FACT_BYTES,
    NUM_CHANNELS,
    OBS_SIZE,
    SCORE_BIN_MIN,
    TILE_WIDTH,
    BcShardReader,
    validate_manifest,
)


def _empty_obs_facts() -> bytearray:
    facts = bytearray(COMPACT_OBS_BASELINE_FACT_BYTES)
    for start in (51, 197, 343, 489):
        for tile in range(TILE_WIDTH):
            facts[start + tile * 4 : start + (tile + 1) * 4] = (0xFFFF_FFFF).to_bytes(4, "little")
    facts[757] = 0x01
    return facts


def _legal_mask(action: int) -> bytes:
    mask = bytearray((ACTION_SPACE + 7) // 8)
    mask[action // 8] = 1 << (action % 8)
    return bytes(mask)


def _record(action: int) -> bytes:
    record = bytearray(BC_BASE_RECORD_SIZE)
    record[:COMPACT_OBS_BASELINE_FACT_BYTES] = _empty_obs_facts()
    record[1675] = action
    record[1676:1682] = _legal_mask(action)
    return bytes(record)


def _write_fixture(root: Path) -> Path:
    shard = root / "train-00000.hybc"
    sample_count = 2
    byte_len = BC_SHARD_HEADER_SIZE + sample_count * BC_BASE_RECORD_SIZE
    header = struct.pack(
        "<8sIIIIIQIIIQIIQQ",
        BC_SHARD_MAGIC,
        BC_SHARD_VERSION,
        BC_SHARD_HEADER_SIZE,
        BC_BASE_RECORD_SIZE,
        0,
        0,
        sample_count,
        NUM_CHANNELS,
        TILE_WIDTH,
        ACTION_SPACE,
        0,
        0,
        BC_SHARD_LAYOUT_VERSION,
        0,
        0,
    )
    assert len(header) == BC_SHARD_HEADER_SIZE
    shard.write_bytes(header + _record(0) + _record(1))
    manifest = {
        "manifest_version": BC_SHARD_MANIFEST_VERSION,
        "shard_version": BC_SHARD_VERSION,
        "shard_header_size": BC_SHARD_HEADER_SIZE,
        "base_record_size": BC_BASE_RECORD_SIZE,
        "max_record_size": BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        "obs_size": OBS_SIZE,
        "num_channels": NUM_CHANNELS,
        "action_space": ACTION_SPACE,
        "storage_layout": "compact",
        "split_mode": "train",
        "totals": {"sample_count": sample_count, "shard_count": 1},
        "splits": [
            {
                "split": "train",
                "shard_count": 1,
                "sample_count": sample_count,
                "feature_flags": 0,
                "record_size": BC_BASE_RECORD_SIZE,
                "shards": [
                    {
                        "split": "train",
                        "shard_index": 0,
                        "file_name": shard.name,
                        "sample_count": sample_count,
                        "first_sample_index": 0,
                        "byte_len": byte_len,
                        "feature_flags": 0,
                        "record_size": BC_BASE_RECORD_SIZE,
                    }
                ],
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_low_score_delta_clamps_to_lowest_score_bin(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    shard = tmp_path / "train-00000.hybc"
    payload = bytearray(shard.read_bytes())
    payload[BC_SHARD_HEADER_SIZE + 1682 : BC_SHARD_HEADER_SIZE + 1686] = struct.pack("<i", int(SCORE_BIN_MIN) - 1)
    shard.write_bytes(payload)
    with BcShardReader(manifest_path) as reader:
        batch = reader.batch_range(0, 1)
    assert batch.score_pdf[0, 0] == 1.0
    assert float(batch.score_pdf[0, 1:].sum()) == 0.0


def test_raw_mjai_decode_truncated_metadata_raises_value_error() -> None:
    payload = struct.pack("<QII", 1, 0, 1)
    with pytest.raises(ValueError, match="field header"):
        decode_batch(payload)
    payload += struct.pack("<HBB", 1, 1, 1)
    with pytest.raises(ValueError, match="field shape"):
        decode_batch(payload)
    payload += struct.pack("<Q", 1)
    with pytest.raises(ValueError, match="field byte length"):
        decode_batch(payload)


def test_raw_mjai_stream_command_uses_default_rust_env() -> None:
    cmd = build_raw_mjai_stream_command(
        data_dir=Path("/data/mjai"),
        batch_size=2048,
        max_games=5,
        max_samples=4096,
        queue_bound=8,
        worker_threads=20,
        train_fraction=0.9,
        augment=True,
    )
    assert cmd[:5] == ["pixi", "run", "-e", "default", "cargo"]
    assert _flag_value(cmd, "--input") == "/data/mjai"
    assert _flag_value(cmd, "--batch-size") == "2048"
    assert _flag_value(cmd, "--max-games") == "5"
    assert _flag_value(cmd, "--max-samples") == "4096"
    assert "--augment" in cmd


def test_raw_mjai_pinned_ffi_requires_explicit_library(tmp_path: Path) -> None:
    missing = tmp_path / "libhydra_raw_mjai_ffi.so"
    with pytest.raises(ImportError):
        RawMjaiPinnedStream(
            data_dir=Path("/data/mjai"),
            batch_size=2048,
            queue_bound=8,
            worker_threads=20,
            max_games=5,
            max_samples=4096,
            train_fraction=0.9,
            augment=False,
            split="train",
            library_path=missing,
            ring_size=2,
        )


class _FakeRawMjaiNext:
    def __init__(self, rows: int, batches: int = 1) -> None:
        self.rows = rows
        self.loaded_games = batches
        self.skipped_games = 0
        self.samples = rows * batches
        self.batches = batches
        self.max_games_reached = False
        self.max_samples_reached = False
        self.stats = RawMjaiBridgeStats(open_count=1, last_next_fill_ms=0.1)


class _FakeRawMjaiStream:
    next_calls = 0
    close_calls = 0
    fail_next = False
    block_next = False
    release = threading.Event()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).next_calls = 0
        type(self).close_calls = 0
        type(self).release.clear()

    def next_into(self, *ptrs: object) -> _FakeRawMjaiNext:
        type(self).next_calls += 1
        if type(self).fail_next:
            raise RuntimeError("synthetic producer failure")
        if type(self).block_next:
            type(self).release.wait(timeout=5.0)
        rows = ptrs[-1]
        if not isinstance(rows, int):
            raise TypeError("capacity rows must be int")
        return _FakeRawMjaiNext(rows=rows, batches=type(self).next_calls)

    def stats(self) -> RawMjaiBridgeStats:
        return RawMjaiBridgeStats(open_count=1)

    def close(self) -> None:
        type(self).close_calls += 1
        type(self).release.set()


@pytest.fixture(autouse=True)
def _clear_raw_mjai_stream_override() -> Generator[None]:
    RawMjaiPinnedStream._set_stream_override_for_tests(None)
    yield
    RawMjaiPinnedStream._set_stream_override_for_tests(None)
    _FakeRawMjaiStream.fail_next = False
    _FakeRawMjaiStream.block_next = False
    _FakeRawMjaiStream.release.set()


def _fake_pinned_stream(tmp_path: Path, *, ring_size: int = 2, close_timeout_s: float = 30.0) -> RawMjaiPinnedStream:
    RawMjaiPinnedStream._set_stream_override_for_tests(_FakeRawMjaiStream)
    return RawMjaiPinnedStream(
        data_dir=Path("/data/mjai"),
        batch_size=2,
        queue_bound=1,
        worker_threads=1,
        max_games=1,
        max_samples=2,
        train_fraction=0.9,
        augment=False,
        split="train",
        library_path=tmp_path / "unused.so",
        ring_size=ring_size,
        close_timeout_s=close_timeout_s,
    )


def test_raw_mjai_pinned_close_while_producer_active_does_not_hang(tmp_path: Path) -> None:
    _FakeRawMjaiStream.block_next = True
    stream = _fake_pinned_stream(tmp_path, close_timeout_s=0.1)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="producer did not stop"):
        stream.close()
    assert time.perf_counter() - started < 1.0


def test_raw_mjai_pinned_producer_exception_reaches_consumer(tmp_path: Path) -> None:
    _FakeRawMjaiStream.fail_next = True
    stream = _fake_pinned_stream(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="synthetic producer failure"):
            stream.next_batch()
        assert "synthetic producer failure" in (stream.queue_stats().producer_error or "")
    finally:
        stream.close()


def test_raw_mjai_pinned_slot_not_reused_before_mark_inflight(tmp_path: Path) -> None:
    stream = _fake_pinned_stream(tmp_path, ring_size=2)
    try:
        first, _ = stream.next_batch()
        second, _ = stream.next_batch()
        assert first.obs.data_ptr() != second.obs.data_ptr()
        time.sleep(0.05)
        stats_before = stream.queue_stats()
        assert stats_before.free_queue_size == 0
        stream.mark_inflight(first)
        stream.mark_inflight(second)
    finally:
        stream.close()


def test_raw_mjai_pinned_queues_remain_bounded(tmp_path: Path) -> None:
    stream = _fake_pinned_stream(tmp_path, ring_size=3)
    try:
        time.sleep(0.05)
        stats = stream.queue_stats()
        assert stats.ready_queue_size <= 3
        assert stats.free_queue_size <= 3
        assert stats.produced_batches <= 3
        assert stream.bridge_stats().open_count == 1
    finally:
        stream.close()


def _flag_value(cmd: list[str], flag: str) -> str:
    index = cmd.index(flag)
    return cmd[index + 1]


def test_compact_reader_decodes_policy_batch(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    summary = validate_manifest(manifest_path, check_files=True)
    assert summary.train_samples == 2
    assert summary.record_size == BC_BASE_RECORD_SIZE

    with BcShardReader(manifest_path) as reader:
        batch = reader.batch_range(0, 2)

    assert batch.obs.shape == (2, NUM_CHANNELS, TILE_WIDTH)
    assert batch.obs.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.legal_mask.dtype == np.bool_
    np.testing.assert_array_equal(batch.actions, np.array([0, 1], dtype=np.int64))
    assert bool(batch.legal_mask[0, 0])
    assert not bool(batch.legal_mask[0, 1])
    assert bool(batch.legal_mask[1, 1])
    assert not bool(batch.legal_mask[1, 0])
    assert float(batch.obs[0, 55, 0]) == 1.0


def test_compact_reader_rejects_illegal_action_record(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    shard = tmp_path / "train-00000.hybc"
    data = bytearray(shard.read_bytes())
    data[BC_SHARD_HEADER_SIZE + 1676] = 0
    shard.write_bytes(data)

    with BcShardReader(manifest_path) as reader, pytest.raises(ValueError, match="not legal"):
        reader.batch_range(0, 1)


def test_python_reader_matches_rust_parity_fixture() -> None:
    manifest_path = Path("crates/hydra-bc-shards/target/python-parity-fixture/manifest.json")
    assert manifest_path.exists(), "run Rust compact_reader_exports_python_parity_fixture first"

    with BcShardReader(manifest_path) as reader:
        batch = reader.batch_range(0, 1)

    assert batch.value_target.shape == (1,)
    assert batch.grp_target.shape == (1, 24)
    assert batch.oracle_target.shape == (1, 4)
    assert batch.oracle_target_mask.shape == (1,)
    assert batch.tenpai.shape == (1, 3)
    assert batch.opp_next.shape == (1, 102)
    assert batch.danger.shape == (1, 102)
    assert batch.danger_mask.shape == (1, 102)
    assert batch.score_pdf.shape == (1, 64)
    assert batch.score_cdf.shape == (1, 64)
    assert batch.safety_target is not None
    assert batch.safety_mask is not None
    assert batch.safety_target.shape == (1, ACTION_SPACE)
    assert batch.safety_mask.shape == (1, ACTION_SPACE)

    np.testing.assert_array_equal(batch.actions, np.array([3], dtype=np.int64))
    assert bool(batch.legal_mask[0, 3])
    np.testing.assert_array_equal(batch.value_target, np.array([0.12], dtype=np.float32))
    assert float(batch.grp_target[0, 7]) == 1.0
    assert float(batch.grp_target.sum()) == 1.0
    np.testing.assert_array_equal(batch.oracle_target, np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    np.testing.assert_array_equal(batch.oracle_target_mask, np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(batch.tenpai, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    assert float(batch.opp_next[0, 3]) == 1.0
    assert float(batch.opp_next[0, 34 + 8]) == 1.0
    assert float(batch.opp_next.sum()) == 2.0
    assert float(batch.danger[0, 3]) == 1.0
    assert float(batch.danger[0, 34 + 8]) == 1.0
    assert float(batch.danger_mask[0, 3]) == 1.0
    assert float(batch.danger_mask[0, 34 + 8]) == 1.0
    assert int(batch.score_pdf.sum()) == 1
    assert int(batch.score_cdf.sum()) == 28
    np.testing.assert_array_equal(batch.safety_target[0, 5], np.float32(0.05))
    assert float(batch.safety_mask[0, 4]) == 1.0
    assert float(batch.safety_mask[0, 5]) == 0.0
