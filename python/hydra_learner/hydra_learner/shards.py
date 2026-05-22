"""Compact BC shard reader for the experimental PyTorch learner."""

from __future__ import annotations

import json
import mmap
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
import numpy.typing as npt

BC_SHARD_MAGIC = b"HYBCS3\0\0"
BC_DENSE_SHARD_MAGIC = b"HYBCS2\0\0"
DENSE_REBUILD_MESSAGE = "dense BC shards are obsolete; rebuild from replay"
BC_SHARD_VERSION = 3
BC_SHARD_MANIFEST_VERSION = 3
BC_SHARD_HEADER_SIZE = 80
BC_SHARD_LAYOUT_VERSION = 1
OBS_SIZE = 6528
NUM_CHANNELS = 192
TILE_WIDTH = 34
ACTION_SPACE = 46
OPPONENT_COUNT = 3
PLAYER_COUNT = 4
SPATIAL_TARGET_SIZE = OPPONENT_COUNT * TILE_WIDTH
GRP_CLASS_COUNT = 24
SCORE_BINS = 64
SCORE_BIN_MIN = -50_000.0
SCORE_BIN_MAX = 60_000.0
STORAGE_LAYOUT_COMPACT = "compact"
SPLIT_IDS = {"train": 0, "validation": 1}
PACKED_ACTION_MASK_BYTES = (ACTION_SPACE + 7) // 8
PACKED_LEGAL_MASK_BYTES = PACKED_ACTION_MASK_BYTES
TILE34_COUNT_BYTES = (TILE_WIDTH * 3 + 7) // 8
TILE34_BITSET_BYTES = (TILE_WIDTH + 7) // 8
PACKED_SPATIAL_MASK_BYTES = (SPATIAL_TARGET_SIZE + 7) // 8
ORACLE_FLOAT32_BYTES = PLAYER_COUNT * 4
ORACLE_MASK_BYTES = 1
OPP_NEXT_BYTES = OPPONENT_COUNT
COMPACT_OBS_BASELINE_FACT_BYTES = 1675
COMPACT_OBS_SCALAR_REPEATED_BYTES = 0
COMPACT_OBS_DENSE_BYTES = 0
FLAG_SAFETY_RESIDUAL = 1 << 0
FLAG_EXIT = 1 << 1
FLAG_DELTA_Q = 1 << 2
FLAG_BELIEF_FIELDS = 1 << 3
FLAG_MIXTURE_WEIGHTS = 1 << 4
VALID_FEATURE_FLAGS = FLAG_SAFETY_RESIDUAL | FLAG_EXIT | FLAG_DELTA_Q | FLAG_BELIEF_FIELDS | FLAG_MIXTURE_WEIGHTS
OPTIONAL_ACTION_FLOAT32_BYTES = ACTION_SPACE * 4
OPTIONAL_ACTION_MASK_BYTES = PACKED_ACTION_MASK_BYTES
BELIEF_FIELDS_BYTES = 16 * TILE_WIDTH * 4
MIXTURE_WEIGHTS_BYTES = PLAYER_COUNT * 4
BC_BASE_RECORD_SIZE = (
    COMPACT_OBS_BASELINE_FACT_BYTES
    + 1
    + PACKED_LEGAL_MASK_BYTES
    + 4
    + 1
    + ORACLE_FLOAT32_BYTES
    + ORACLE_MASK_BYTES
    + 1
    + OPP_NEXT_BYTES
    + PACKED_SPATIAL_MASK_BYTES
    + PACKED_SPATIAL_MASK_BYTES
)
BC_RECORD_SIZE_WITH_ALL_OPTIONALS = (
    BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES) * 3
    + BELIEF_FIELDS_BYTES
    + MIXTURE_WEIGHTS_BYTES
)
_DISCARD_EXP_TABLE = np.array(
    [
        1.0,
        0.818_730_8,
        0.670_320_0,
        0.548_811_6,
        0.449_329_0,
        0.367_879_5,
        0.301_194_2,
        0.246_597_0,
        0.201_896_5,
        0.165_298_9,
        0.135_335_3,
        0.110_803_2,
        0.090_717_96,
        0.074_273_58,
        0.060_810_06,
        0.049_787_07,
        0.040_762_20,
        0.033_373_27,
        0.027_323_72,
        0.022_370_77,
        0.018_315_64,
        0.014_995_58,
        0.012_277_34,
        0.010_051_84,
        0.008_229_747,
        0.006_737_947,
        0.005_516_564,
        0.004_516_581,
        0.003_697_864,
        0.003_027_555,
        0.002_478_752,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ShardDescriptor:
    split: str
    shard_index: int
    file_name: str
    sample_count: int
    first_sample_index: int
    byte_len: int
    feature_flags: int
    record_size: int


@dataclass(frozen=True)
class ManifestSummary:
    path: Path
    train_samples: int
    validation_samples: int
    shard_count: int
    record_size: int
    feature_flags: int


@dataclass(frozen=True)
class PolicyBatch:
    obs: npt.NDArray[np.float32]
    actions: npt.NDArray[np.int64]
    legal_mask: npt.NDArray[np.bool_]
    value_target: npt.NDArray[np.float32]
    grp_target: npt.NDArray[np.float32]
    oracle_target: npt.NDArray[np.float32]
    oracle_target_mask: npt.NDArray[np.float32]
    tenpai: npt.NDArray[np.float32]
    opp_next: npt.NDArray[np.float32]
    danger: npt.NDArray[np.float32]
    danger_mask: npt.NDArray[np.float32]
    score_pdf: npt.NDArray[np.float32]
    score_cdf: npt.NDArray[np.float32]
    safety_target: npt.NDArray[np.float32] | None
    safety_mask: npt.NDArray[np.float32] | None


@dataclass(frozen=True)
class _ShardHeader:
    record_size: int
    split_id: int
    shard_index: int
    sample_count: int
    first_sample_index: int
    feature_flags: int


@dataclass(frozen=True)
class _ShardMeta:
    split: str
    shard_index: int
    file_name: str
    sample_count: int
    first_sample_index: int
    byte_len: int
    feature_flags: int
    record_size: int


class _MappedShard:
    def __init__(self, path: Path, meta: _ShardMeta) -> None:
        self.path = path
        self.meta = meta
        self._file: BinaryIO = path.open("rb")
        self.mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        _verify_mapped_shard(self.mmap, meta, path)

    def close(self) -> None:
        self.mmap.close()
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


class BcShardReader:
    """Mmap-backed compact shard reader for policy BC fields."""

    def __init__(self, manifest_path: Path, split: str = "train") -> None:
        summary, metas = _load_split_shards(manifest_path, split, check_files=False)
        if not metas:
            raise ValueError(f"BC shard manifest has no {split!r} shards")
        self.manifest_path = manifest_path
        self.summary = summary
        self.split = split
        base_dir = manifest_path.parent
        self._shards = [_MappedShard(base_dir / meta.file_name, meta) for meta in metas]
        self._starts = np.array([shard.meta.first_sample_index for shard in self._shards], dtype=np.int64)

    @property
    def sample_count(self) -> int:
        return sum(shard.meta.sample_count for shard in self._shards)

    @property
    def feature_flags(self) -> int:
        return self._shards[0].meta.feature_flags

    @property
    def record_size(self) -> int:
        return self._shards[0].meta.record_size

    def close(self) -> None:
        for shard in self._shards:
            shard.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def batch_range(self, start: int, batch_size: int) -> PolicyBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        end = start + batch_size
        if start < 0 or end > self.sample_count:
            raise ValueError(f"BC shard batch range {start}..{end} exceeds sample count {self.sample_count}")
        obs = np.zeros((batch_size, NUM_CHANNELS, TILE_WIDTH), dtype=np.float32)
        actions = np.empty((batch_size,), dtype=np.int64)
        legal = np.empty((batch_size, ACTION_SPACE), dtype=np.bool_)
        value_target = np.empty((batch_size,), dtype=np.float32)
        grp_target = np.zeros((batch_size, GRP_CLASS_COUNT), dtype=np.float32)
        oracle_target = np.empty((batch_size, PLAYER_COUNT), dtype=np.float32)
        oracle_target_mask = np.empty((batch_size,), dtype=np.float32)
        tenpai = np.empty((batch_size, OPPONENT_COUNT), dtype=np.float32)
        opp_next = np.zeros((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        danger = np.empty((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        danger_mask = np.empty((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        score_pdf = np.zeros((batch_size, SCORE_BINS), dtype=np.float32)
        score_cdf = np.zeros((batch_size, SCORE_BINS), dtype=np.float32)
        safety_target = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32)
            if self.feature_flags & FLAG_SAFETY_RESIDUAL
            else None
        )
        safety_mask = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32)
            if self.feature_flags & FLAG_SAFETY_RESIDUAL
            else None
        )
        row = 0
        remaining = batch_size
        sample = start
        while remaining > 0:
            shard_index = int(np.searchsorted(self._starts, sample, side="right") - 1)
            if shard_index < 0:
                raise ValueError(f"BC shard sample index {sample} out of bounds")
            shard = self._shards[shard_index]
            local = sample - shard.meta.first_sample_index
            take = min(remaining, shard.meta.sample_count - local)
            _decode_rows(
                shard.mmap,
                shard.meta.record_size,
                shard.meta.feature_flags,
                local,
                take,
                obs,
                actions,
                legal,
                value_target,
                grp_target,
                oracle_target,
                oracle_target_mask,
                tenpai,
                opp_next,
                danger,
                danger_mask,
                score_pdf,
                score_cdf,
                safety_target,
                safety_mask,
                row,
            )
            row += take
            sample += take
            remaining -= take
        return PolicyBatch(
            obs=obs,
            actions=actions,
            legal_mask=legal,
            value_target=value_target,
            grp_target=grp_target,
            oracle_target=oracle_target,
            oracle_target_mask=oracle_target_mask,
            tenpai=tenpai,
            opp_next=opp_next,
            danger=danger,
            danger_mask=danger_mask,
            score_pdf=score_pdf,
            score_cdf=score_cdf,
            safety_target=safety_target,
            safety_mask=safety_mask,
        )


class BcShardDataset:
    """Sequential real-shard batch source for policy-only BC."""

    def __init__(self, manifest_path: Path, batch_size: int, split: str = "train") -> None:
        self.reader = BcShardReader(manifest_path, split=split)
        self.batch_size = batch_size
        self._cursor = 0
        self.last_fetch_decode_ms = 0.0

    @property
    def sample_count(self) -> int:
        return self.reader.sample_count

    def next_batch(self) -> PolicyBatch:
        if self.sample_count < self.batch_size:
            raise ValueError(f"BC shard split has {self.sample_count} samples, needs batch {self.batch_size}")
        if self._cursor + self.batch_size > self.sample_count:
            self._cursor = 0
        started = time.perf_counter()
        batch = self.reader.batch_range(self._cursor, self.batch_size)
        self.last_fetch_decode_ms = (time.perf_counter() - started) * 1000.0
        self._cursor += self.batch_size
        return batch

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data: object = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("BC shard manifest root must be a JSON object")
    return data


def checked_compact_record_size(flags: int) -> int:
    unknown = flags & ~VALID_FEATURE_FLAGS
    if unknown != 0:
        raise ValueError(f"BC shard feature_flags contain unsupported bits {unknown:#x}")
    size: int = BC_BASE_RECORD_SIZE
    if flags & FLAG_SAFETY_RESIDUAL:
        size += OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES
    if flags & FLAG_EXIT:
        size += OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES
    if flags & FLAG_DELTA_Q:
        size += OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES
    if flags & FLAG_BELIEF_FIELDS:
        size += BELIEF_FIELDS_BYTES
    if flags & FLAG_MIXTURE_WEIGHTS:
        size += MIXTURE_WEIGHTS_BYTES
    return size


def validate_manifest(path: Path, check_files: bool = False) -> ManifestSummary:
    summary, _metas = _load_split_shards(path, split=None, check_files=check_files)
    return summary


def _load_split_shards(path: Path, split: str | None, check_files: bool) -> tuple[ManifestSummary, list[_ShardMeta]]:
    data = load_manifest(path)
    _expect_int(data, "manifest_version", BC_SHARD_MANIFEST_VERSION)
    _expect_int(data, "shard_version", BC_SHARD_VERSION)
    _expect_int(data, "shard_header_size", BC_SHARD_HEADER_SIZE)
    _expect_int(data, "base_record_size", BC_BASE_RECORD_SIZE)
    _expect_int(data, "max_record_size", BC_RECORD_SIZE_WITH_ALL_OPTIONALS)
    _expect_int(data, "obs_size", OBS_SIZE)
    _expect_int(data, "num_channels", NUM_CHANNELS)
    _expect_int(data, "action_space", ACTION_SPACE)
    if data.get("storage_layout") != STORAGE_LAYOUT_COMPACT:
        raise ValueError(f"storage_layout must be {STORAGE_LAYOUT_COMPACT!r}, got {data.get('storage_layout')!r}")
    split_mode = data.get("split_mode", "both")
    if split_mode not in {"both", "train", "validation"}:
        raise ValueError(f"unsupported split_mode {split_mode!r}")

    manifest_dir = path.parent
    split_totals = {"train": 0, "validation": 0}
    shard_count = 0
    record_size = int(data.get("base_record_size", 0))
    feature_flags = 0
    selected: list[_ShardMeta] = []
    seen_splits: set[str] = set()
    splits = data.get("splits", [])
    if not isinstance(splits, list):
        raise TypeError("manifest splits must be a list")
    for split_obj_any in splits:
        if not isinstance(split_obj_any, dict):
            raise TypeError("manifest split entry must be an object")
        split_obj: dict[str, Any] = split_obj_any
        split_name_obj = split_obj.get("split")
        if not isinstance(split_name_obj, str) or split_name_obj not in split_totals:
            raise ValueError(f"unknown split {split_name_obj!r}")
        if split_name_obj in seen_splits:
            raise ValueError(f"duplicate split {split_name_obj!r}")
        seen_splits.add(split_name_obj)
        split_feature_flags = int(split_obj.get("feature_flags", -1))
        split_record_size = int(split_obj.get("record_size", -1))
        if split_record_size != checked_compact_record_size(split_feature_flags):
            raise ValueError(f"{split_name_obj} split record_size mismatch")
        descriptors = split_obj.get("shards", [])
        if not isinstance(descriptors, list):
            raise TypeError("manifest split shards must be a list")
        split_sum = 0
        expected_first = 0
        for expected_shard_index, desc_any in enumerate(descriptors):
            if not isinstance(desc_any, dict):
                raise TypeError("shard descriptor must be an object")
            desc: dict[str, Any] = desc_any
            meta = _parse_descriptor(desc, split_name_obj, expected_shard_index, expected_first)
            if meta.feature_flags != split_feature_flags or meta.record_size != split_record_size:
                raise ValueError(f"{meta.file_name} split layout mismatch")
            expected_first += meta.sample_count
            split_sum += meta.sample_count
            shard_count += 1
            record_size = meta.record_size
            feature_flags |= meta.feature_flags
            if check_files:
                _verify_file_header(manifest_dir / meta.file_name, meta)
            if split is None or split == split_name_obj:
                selected.append(meta)
        if split_sum != int(split_obj.get("sample_count", -1)):
            raise ValueError(f"{split_name_obj} split sample_count mismatch")
        if len(descriptors) != int(split_obj.get("shard_count", -1)):
            raise ValueError(f"{split_name_obj} split shard_count mismatch")
        split_totals[split_name_obj] = split_sum

    if split is not None and split not in split_totals:
        raise ValueError(f"unknown split {split!r}")
    if int(data.get("totals", {}).get("sample_count", -1)) != split_totals["train"] + split_totals["validation"]:
        raise ValueError("manifest totals.sample_count mismatch")
    if int(data.get("totals", {}).get("shard_count", -1)) != shard_count:
        raise ValueError("manifest totals.shard_count mismatch")
    if split_mode == "both" and int(data.get("totals", {}).get("sample_count", 0)) > 0:
        if "train" not in seen_splits or "validation" not in seen_splits:
            raise ValueError("split_mode both requires train and validation split entries")
    elif (
        split_mode in {"train", "validation"}
        and int(data.get("totals", {}).get("sample_count", 0)) > 0
        and split_mode not in seen_splits
    ):
        raise ValueError(f"split_mode {split_mode} requires matching split entry")
    return (
        ManifestSummary(
            path=path,
            train_samples=split_totals["train"],
            validation_samples=split_totals["validation"],
            shard_count=shard_count,
            record_size=record_size,
            feature_flags=feature_flags,
        ),
        selected,
    )


def _parse_descriptor(
    desc: dict[str, Any], split_name: str, expected_shard_index: int, expected_first: int
) -> _ShardMeta:
    if desc.get("split") != split_name:
        raise ValueError(f"shard descriptor split {desc.get('split')!r} does not match {split_name!r}")
    file_name = desc.get("file_name")
    if not isinstance(file_name, str):
        raise ValueError("shard descriptor file_name must be string")
    _safe_file_name(file_name)
    meta = _ShardMeta(
        split=split_name,
        shard_index=int(desc.get("shard_index", -1)),
        file_name=file_name,
        sample_count=int(desc.get("sample_count", -1)),
        first_sample_index=int(desc.get("first_sample_index", -1)),
        byte_len=int(desc.get("byte_len", -1)),
        feature_flags=int(desc.get("feature_flags", -1)),
        record_size=int(desc.get("record_size", -1)),
    )
    if meta.shard_index != expected_shard_index:
        raise ValueError(f"{file_name} shard_index must be {expected_shard_index}, got {meta.shard_index}")
    if meta.sample_count < 0:
        raise ValueError(f"{file_name} sample_count must be non-negative")
    if meta.first_sample_index != expected_first:
        raise ValueError(f"{file_name} first_sample_index must be {expected_first}, got {meta.first_sample_index}")
    if meta.record_size != checked_compact_record_size(meta.feature_flags):
        raise ValueError(f"{file_name} record_size does not match feature_flags")
    if meta.byte_len != BC_SHARD_HEADER_SIZE + meta.sample_count * meta.record_size:
        raise ValueError(f"{file_name} byte_len does not match header + records")
    return meta


def _expect_int(data: dict[str, Any], key: str, expected: int) -> None:
    value = data.get(key)
    if value != expected:
        raise ValueError(f"manifest {key} must be {expected}, got {value!r}")


def _safe_file_name(file_name: str) -> None:
    path = Path(file_name)
    if path.is_absolute() or len(path.parts) != 1 or file_name in {"", ".", ".."}:
        raise ValueError(f"unsafe shard file_name {file_name!r}")


def _read_header(buf: bytes | mmap.mmap, path: Path) -> _ShardHeader:
    if len(buf) < BC_SHARD_HEADER_SIZE:
        raise ValueError(f"{path} file too small for BC shard header")
    if buf[:8] == BC_DENSE_SHARD_MAGIC:
        raise ValueError(DENSE_REBUILD_MESSAGE)
    if buf[:8] != BC_SHARD_MAGIC:
        raise ValueError(f"{path} invalid compact BC shard magic")
    version, header_size, record_size, split_id, shard_index, sample_count = struct.unpack_from("<IIIIIQ", buf, 8)
    num_channels, tile_count, action_space = struct.unpack_from("<III", buf, 36)
    first_sample_index = struct.unpack_from("<Q", buf, 48)[0]
    feature_flags, layout_version = struct.unpack_from("<II", buf, 56)
    if version != BC_SHARD_VERSION:
        raise ValueError(f"{path} shard version must be {BC_SHARD_VERSION}, got {version}")
    if header_size != BC_SHARD_HEADER_SIZE:
        raise ValueError(f"{path} header size must be {BC_SHARD_HEADER_SIZE}, got {header_size}")
    if num_channels != NUM_CHANNELS or tile_count != TILE_WIDTH or action_space != ACTION_SPACE:
        raise ValueError(f"{path} shape mismatch: {num_channels}x{tile_count}, actions {action_space}")
    if layout_version != BC_SHARD_LAYOUT_VERSION:
        raise ValueError(f"{path} layout version must be {BC_SHARD_LAYOUT_VERSION}, got {layout_version}")
    return _ShardHeader(
        record_size=record_size,
        split_id=split_id,
        shard_index=shard_index,
        sample_count=sample_count,
        first_sample_index=first_sample_index,
        feature_flags=feature_flags,
    )


def _verify_file_header(path: Path, meta: _ShardMeta) -> None:
    with path.open("rb") as fh:
        header = fh.read(BC_SHARD_HEADER_SIZE)
    _verify_header(_read_header(header, path), meta, path)
    if path.stat().st_size != meta.byte_len:
        raise ValueError(f"{path} length mismatch")


def _verify_mapped_shard(buf: mmap.mmap, meta: _ShardMeta, path: Path) -> None:
    _verify_header(_read_header(buf, path), meta, path)
    if len(buf) != meta.byte_len:
        raise ValueError(f"{path} length mismatch")


def _verify_header(header: _ShardHeader, meta: _ShardMeta, path: Path) -> None:
    if header.split_id != SPLIT_IDS[meta.split]:
        raise ValueError(f"{path} split mismatch")
    if header.record_size != meta.record_size:
        raise ValueError(f"{path} record size mismatch")
    if header.shard_index != meta.shard_index:
        raise ValueError(f"{path} shard index mismatch")
    if header.sample_count != meta.sample_count:
        raise ValueError(f"{path} sample count mismatch")
    if header.first_sample_index != meta.first_sample_index:
        raise ValueError(f"{path} first sample index mismatch")
    if header.feature_flags != meta.feature_flags:
        raise ValueError(f"{path} feature flags mismatch")


def _decode_rows(
    buf: mmap.mmap,
    record_size: int,
    feature_flags: int,
    start_sample: int,
    sample_count: int,
    obs: npt.NDArray[np.float32],
    actions: npt.NDArray[np.int64],
    legal: npt.NDArray[np.bool_],
    value_target: npt.NDArray[np.float32],
    grp_target: npt.NDArray[np.float32],
    oracle_target: npt.NDArray[np.float32],
    oracle_target_mask: npt.NDArray[np.float32],
    tenpai: npt.NDArray[np.float32],
    opp_next: npt.NDArray[np.float32],
    danger: npt.NDArray[np.float32],
    danger_mask: npt.NDArray[np.float32],
    score_pdf: npt.NDArray[np.float32],
    score_cdf: npt.NDArray[np.float32],
    safety_target: npt.NDArray[np.float32] | None,
    safety_mask: npt.NDArray[np.float32] | None,
    row_start: int,
) -> None:
    for idx in range(sample_count):
        row = row_start + idx
        record_start = BC_SHARD_HEADER_SIZE + (start_sample + idx) * record_size
        record = memoryview(buf)[record_start : record_start + record_size]
        _decode_record(
            record,
            feature_flags,
            obs[row],
            actions,
            legal,
            value_target,
            grp_target,
            oracle_target,
            oracle_target_mask,
            tenpai,
            opp_next,
            danger,
            danger_mask,
            score_pdf,
            score_cdf,
            safety_target,
            safety_mask,
            row,
        )


def _decode_record(
    record: memoryview,
    feature_flags: int,
    obs_row: npt.NDArray[np.float32],
    actions: npt.NDArray[np.int64],
    legal: npt.NDArray[np.bool_],
    value_target: npt.NDArray[np.float32],
    grp_target: npt.NDArray[np.float32],
    oracle_target: npt.NDArray[np.float32],
    oracle_target_mask: npt.NDArray[np.float32],
    tenpai: npt.NDArray[np.float32],
    opp_next: npt.NDArray[np.float32],
    danger: npt.NDArray[np.float32],
    danger_mask: npt.NDArray[np.float32],
    score_pdf: npt.NDArray[np.float32],
    score_cdf: npt.NDArray[np.float32],
    safety_target: npt.NDArray[np.float32] | None,
    safety_mask: npt.NDArray[np.float32] | None,
    row: int,
) -> None:
    cursor: int = 0
    facts = record[cursor : cursor + COMPACT_OBS_BASELINE_FACT_BYTES]
    cursor += COMPACT_OBS_BASELINE_FACT_BYTES
    _decode_compact_obs(facts, obs_row)
    action = record[cursor]
    if action >= ACTION_SPACE:
        raise ValueError(f"BC shard action {action} out of range")
    actions[row] = action
    cursor += 1
    legal_bytes = record[cursor : cursor + PACKED_LEGAL_MASK_BYTES]
    legal[row] = _unpack_bits(legal_bytes, ACTION_SPACE)
    if not legal[row, action]:
        raise ValueError(f"BC shard action {action} is not legal in record")
    cursor += PACKED_LEGAL_MASK_BYTES

    score_delta = struct.unpack_from("<i", record, cursor)[0]
    cursor += 4
    value_target[row] = np.float32(max(-1.0, min(1.0, score_delta / 100_000.0)))
    score_bin = _score_delta_to_bin(score_delta)
    score_pdf[row, score_bin] = 1.0
    score_cdf[row, score_bin:] = 1.0

    grp_label = record[cursor]
    if grp_label < GRP_CLASS_COUNT:
        grp_target[row, grp_label] = 1.0
    cursor += 1

    oracle_target[row] = np.frombuffer(record[cursor : cursor + ORACLE_FLOAT32_BYTES], dtype="<f4", count=PLAYER_COUNT)
    cursor += ORACLE_FLOAT32_BYTES
    oracle_target_mask[row] = 1.0 if record[cursor] != 0 else 0.0
    cursor += ORACLE_MASK_BYTES

    tenpai[row] = _unpack_bits(record[cursor : cursor + 1], OPPONENT_COUNT).astype(np.float32)
    cursor += 1

    opp_next_bytes = record[cursor : cursor + OPP_NEXT_BYTES]
    for opponent, tile in enumerate(opp_next_bytes):
        if tile < TILE_WIDTH:
            opp_next[row, opponent * TILE_WIDTH + tile] = 1.0
    cursor += OPP_NEXT_BYTES

    danger[row] = _unpack_bits(record[cursor : cursor + PACKED_SPATIAL_MASK_BYTES], SPATIAL_TARGET_SIZE).astype(
        np.float32
    )
    cursor += PACKED_SPATIAL_MASK_BYTES
    danger_mask[row] = _unpack_bits(record[cursor : cursor + PACKED_SPATIAL_MASK_BYTES], SPATIAL_TARGET_SIZE).astype(
        np.float32
    )
    cursor += PACKED_SPATIAL_MASK_BYTES

    if feature_flags & FLAG_SAFETY_RESIDUAL:
        if safety_target is None or safety_mask is None:
            raise ValueError("BC shard safety flag set but safety buffers missing")
        safety_target[row] = np.frombuffer(
            record[cursor : cursor + OPTIONAL_ACTION_FLOAT32_BYTES], dtype="<f4", count=ACTION_SPACE
        )
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES
        safety_mask[row] = _unpack_bits(record[cursor : cursor + OPTIONAL_ACTION_MASK_BYTES], ACTION_SPACE).astype(
            np.float32
        )
        cursor += OPTIONAL_ACTION_MASK_BYTES
    if feature_flags & FLAG_EXIT:
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES
    if feature_flags & FLAG_DELTA_Q:
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES
    if feature_flags & FLAG_BELIEF_FIELDS:
        cursor += BELIEF_FIELDS_BYTES
    if feature_flags & FLAG_MIXTURE_WEIGHTS:
        cursor += MIXTURE_WEIGHTS_BYTES
    if cursor != len(record):
        raise ValueError(f"BC shard compact record has {len(record) - cursor} trailing byte(s)")


def _score_delta_to_bin(score_delta: int) -> int:
    normalized = (float(score_delta) - SCORE_BIN_MIN) / (SCORE_BIN_MAX - SCORE_BIN_MIN)
    bin_index = int(normalized * SCORE_BINS)
    return min(bin_index, SCORE_BINS - 1)


def _decode_compact_obs(facts: memoryview, dst: npt.NDArray[np.float32]) -> None:
    if len(facts) != COMPACT_OBS_BASELINE_FACT_BYTES:
        raise ValueError("BC shard compact observation fact section has invalid length")
    dst.fill(0.0)
    cursor: int = 0
    _decode_tile_counts(facts[cursor : cursor + TILE34_COUNT_BYTES], dst, 0)
    cursor += TILE34_COUNT_BYTES
    _decode_tile_counts(facts[cursor : cursor + TILE34_COUNT_BYTES], dst, 4)
    cursor += TILE34_COUNT_BYTES
    _decode_tile_bitset(facts[cursor : cursor + TILE34_BITSET_BYTES], dst, 8)
    cursor += TILE34_BITSET_BYTES
    _decode_tile_bitset(facts[cursor : cursor + TILE34_BITSET_BYTES], dst, 9)
    cursor += TILE34_BITSET_BYTES
    _decode_tile_bitset(facts[cursor : cursor + TILE34_BITSET_BYTES], dst, 10)
    cursor += TILE34_BITSET_BYTES
    cursor = _decode_discard_facts(facts, cursor, dst)
    cursor = _decode_channel_bitsets(facts, cursor, dst, 23, 12)
    _decode_dora_facts(facts[cursor : cursor + TILE_WIDTH], dst)
    cursor += TILE_WIDTH
    _decode_aka_facts(facts[cursor], dst)
    cursor += 1
    cursor = _decode_metadata_facts(facts, cursor, dst)
    cursor = _decode_safety_facts(facts, cursor, dst)
    if cursor != COMPACT_OBS_BASELINE_FACT_BYTES:
        raise ValueError("BC shard compact observation fact cursor mismatch")


def _decode_tile_counts(data: memoryview, dst: npt.NDArray[np.float32], channel_start: int) -> None:
    bits = _unpack_bits(data, TILE_WIDTH * 3).reshape(TILE_WIDTH, 3)
    weights = np.array([1, 2, 4], dtype=np.uint8)
    counts = bits.astype(np.uint8) @ weights
    if bool(np.any(counts > 4)):
        bad = int(np.argmax(counts > 4))
        raise ValueError(f"compact tile count at index {bad} is {int(counts[bad])}, expected 0..=4")
    for threshold in range(4):
        dst[channel_start + threshold, :] = counts > threshold


def _decode_tile_bitset(data: memoryview, dst: npt.NDArray[np.float32], channel: int) -> None:
    dst[channel, :] = _unpack_bits(data, TILE_WIDTH)


def _decode_channel_bitsets(
    data: memoryview, cursor: int, dst: npt.NDArray[np.float32], channel_start: int, channel_count: int
) -> int:
    for channel in range(channel_start, channel_start + channel_count):
        _decode_tile_bitset(data[cursor : cursor + TILE34_BITSET_BYTES], dst, channel)
        cursor += TILE34_BITSET_BYTES
    return cursor


def _decode_discard_facts(data: memoryview, cursor: int, dst: npt.NDArray[np.float32]) -> int:
    for player in range(4):
        base = 11 + player * 3
        _decode_tile_bitset(data[cursor : cursor + TILE34_BITSET_BYTES], dst, base)
        cursor += TILE34_BITSET_BYTES
        _decode_tile_bitset(data[cursor : cursor + TILE34_BITSET_BYTES], dst, base + 1)
        cursor += TILE34_BITSET_BYTES
        values = np.frombuffer(data[cursor : cursor + TILE_WIDTH * 4], dtype="<u4", count=TILE_WIDTH)
        valid = values != np.uint32(0xFFFF_FFFF)
        if bool(np.any(values[valid] >= len(_DISCARD_EXP_TABLE))):
            bad = int(values[valid][np.argmax(values[valid] >= len(_DISCARD_EXP_TABLE))])
            raise ValueError(f"BC shard discard temporal index {bad} out of range")
        dst[base + 2, :] = 0.0
        dst[base + 2, valid] = _DISCARD_EXP_TABLE[values[valid]]
        cursor += TILE_WIDTH * 4
    return cursor


def _decode_dora_facts(data: memoryview, dst: npt.NDArray[np.float32]) -> None:
    counts = np.frombuffer(data, dtype=np.uint8, count=TILE_WIDTH)
    if bool(np.any(counts > 5)):
        raise ValueError("BC shard dora count out of range")
    for threshold in range(5):
        dst[35 + threshold, :] = counts > threshold


def _decode_aka_facts(flags: int, dst: npt.NDArray[np.float32]) -> None:
    for suit in range(3):
        if flags & (1 << suit):
            dst[40 + suit, :].fill(1.0)


def _decode_metadata_facts(data: memoryview, cursor: int, dst: npt.NDArray[np.float32]) -> int:
    _decode_repeated_bool_channels(data[cursor : cursor + TILE34_BITSET_BYTES], dst, 43, 4)
    cursor += TILE34_BITSET_BYTES
    values = np.frombuffer(data[cursor : cursor + 8 * 4], dtype="<f4", count=8)
    for offset, value in enumerate(values):
        dst[47 + offset, :].fill(float(value))
    cursor += 8 * 4
    _decode_repeated_bool_channels(data[cursor : cursor + TILE34_BITSET_BYTES], dst, 55, 4)
    cursor += TILE34_BITSET_BYTES
    values = np.frombuffer(data[cursor : cursor + 3 * 4], dtype="<f4", count=3)
    for offset, value in enumerate(values):
        dst[59 + offset, :].fill(float(value))
    cursor += 3 * 4
    return cursor


def _decode_repeated_bool_channels(
    data: memoryview, dst: npt.NDArray[np.float32], channel_start: int, channel_count: int
) -> None:
    values = _unpack_bits(data, TILE_WIDTH)
    for channel_offset in range(channel_count):
        if values[channel_offset]:
            dst[channel_start + channel_offset, :].fill(1.0)


def _decode_safety_facts(data: memoryview, cursor: int, dst: npt.NDArray[np.float32]) -> int:
    cursor = _decode_channel_bitsets(data, cursor, dst, 62, 9)
    for channel in range(71, 74):
        _decode_dense_channel(data[cursor : cursor + TILE_WIDTH * 4], dst, channel)
        cursor += TILE_WIDTH * 4
    cursor = _decode_channel_bitsets(data, cursor, dst, 74, 3)
    for channel in range(77, 80):
        _decode_dense_channel(data[cursor : cursor + TILE_WIDTH * 4], dst, channel)
        cursor += TILE_WIDTH * 4
    return _decode_channel_bitsets(data, cursor, dst, 80, 5)


def _decode_dense_channel(data: memoryview, dst: npt.NDArray[np.float32], channel: int) -> None:
    dst[channel, :] = np.frombuffer(data, dtype="<f4", count=TILE_WIDTH)


def _unpack_bits(data: memoryview, count: int) -> npt.NDArray[np.bool_]:
    raw = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:count].astype(np.bool_, copy=False)
