from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

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
