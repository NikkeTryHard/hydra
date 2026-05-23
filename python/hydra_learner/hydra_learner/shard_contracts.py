"""Compact BC shard contracts and layout constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """Host batch decoded from compact shard/raw MJAI with obs 192x34 and action width 46."""

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
