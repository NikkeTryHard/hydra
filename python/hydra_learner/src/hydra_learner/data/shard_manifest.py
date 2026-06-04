from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Any

from hydra_learner.data.shard_contracts import (
    ACTION_SPACE,
    BC_BASE_RECORD_SIZE,
    BC_DENSE_SHARD_MAGIC,
    BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE,
    BC_SHARD_LAYOUT_VERSION,
    BC_SHARD_MAGIC,
    BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION,
    BELIEF_FIELDS_BYTES,
    DENSE_REBUILD_MESSAGE,
    FLAG_BELIEF_FIELDS,
    FLAG_DELTA_Q,
    FLAG_EXIT,
    FLAG_MIXTURE_WEIGHTS,
    FLAG_SAFETY_RESIDUAL,
    MIXTURE_WEIGHTS_BYTES,
    NUM_CHANNELS,
    OBS_SIZE,
    OPTIONAL_ACTION_FLOAT32_BYTES,
    OPTIONAL_ACTION_MASK_BYTES,
    SPLIT_IDS,
    STORAGE_LAYOUT_COMPACT,
    TILE_WIDTH,
    VALID_FEATURE_FLAGS,
    ManifestSummary,
    _ShardHeader,
    _ShardMeta,
)


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
        if (split_mode == "train" and split_name_obj == "validation") or (
            split_mode == "validation" and split_name_obj == "train"
        ):
            raise ValueError(f"split_mode {split_mode!r} excludes split {split_name_obj!r}")
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
