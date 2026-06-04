from __future__ import annotations

import mmap
import struct

import numpy as np
import numpy.typing as npt

from hydra_learner.data.shard_contracts import (
    _DISCARD_EXP_TABLE,
    ACTION_SPACE,
    BC_SHARD_HEADER_SIZE,
    BELIEF_FIELDS_BYTES,
    COMPACT_OBS_BASELINE_FACT_BYTES,
    FLAG_BELIEF_FIELDS,
    FLAG_DELTA_Q,
    FLAG_EXIT,
    FLAG_MIXTURE_WEIGHTS,
    FLAG_SAFETY_RESIDUAL,
    GRP_CLASS_COUNT,
    MIXTURE_WEIGHTS_BYTES,
    OPP_NEXT_BYTES,
    OPPONENT_COUNT,
    OPTIONAL_ACTION_FLOAT32_BYTES,
    OPTIONAL_ACTION_MASK_BYTES,
    ORACLE_FLOAT32_BYTES,
    ORACLE_MASK_BYTES,
    PACKED_LEGAL_MASK_BYTES,
    PACKED_SPATIAL_MASK_BYTES,
    PLAYER_COUNT,
    SCORE_BIN_MAX,
    SCORE_BIN_MIN,
    SCORE_BINS,
    SPATIAL_TARGET_SIZE,
    TILE34_BITSET_BYTES,
    TILE34_COUNT_BYTES,
    TILE_WIDTH,
)


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
    exit_target: npt.NDArray[np.float32] | None,
    exit_mask: npt.NDArray[np.float32] | None,
    deltaq_target: npt.NDArray[np.float32] | None,
    deltaq_mask: npt.NDArray[np.float32] | None,
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
            exit_target,
            exit_mask,
            deltaq_target,
            deltaq_mask,
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
    exit_target: npt.NDArray[np.float32] | None,
    exit_mask: npt.NDArray[np.float32] | None,
    deltaq_target: npt.NDArray[np.float32] | None,
    deltaq_mask: npt.NDArray[np.float32] | None,
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
        cursor = _decode_optional_action_pair(record, cursor, row, exit_target, exit_mask, "ExIt")
    if feature_flags & FLAG_DELTA_Q:
        cursor = _decode_optional_action_pair(record, cursor, row, deltaq_target, deltaq_mask, "DeltaQ")
    if feature_flags & FLAG_BELIEF_FIELDS:
        cursor += BELIEF_FIELDS_BYTES
    if feature_flags & FLAG_MIXTURE_WEIGHTS:
        cursor += MIXTURE_WEIGHTS_BYTES
    if cursor != len(record):
        raise ValueError(f"BC shard compact record has {len(record) - cursor} trailing byte(s)")


def _decode_optional_action_pair(
    record: memoryview,
    cursor: int,
    row: int,
    target: npt.NDArray[np.float32] | None,
    mask: npt.NDArray[np.float32] | None,
    name: str,
) -> int:
    if target is None or mask is None:
        raise ValueError(f"BC shard {name} flag set but target buffers missing")
    target[row] = np.frombuffer(
        record[cursor : cursor + OPTIONAL_ACTION_FLOAT32_BYTES], dtype="<f4", count=ACTION_SPACE
    )
    cursor += OPTIONAL_ACTION_FLOAT32_BYTES
    mask[row] = _unpack_bits(record[cursor : cursor + OPTIONAL_ACTION_MASK_BYTES], ACTION_SPACE).astype(np.float32)
    return cursor + OPTIONAL_ACTION_MASK_BYTES


def _score_delta_to_bin(score_delta: int) -> int:
    normalized = (float(score_delta) - SCORE_BIN_MIN) / (SCORE_BIN_MAX - SCORE_BIN_MIN)
    bin_index = int(normalized * SCORE_BINS)
    return max(0, min(bin_index, SCORE_BINS - 1))


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
