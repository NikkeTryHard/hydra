//! Backend-agnostic BC shard host format, manifest, reader, and writer.

pub(crate) mod compact;
pub mod host;
pub mod manifest;
pub mod reader;
pub mod writer;

pub use host::{BcShardHostBatch, BcShardHostScratch, record_size_for_flags};
pub use manifest::{
    BC_BASE_RECORD_SIZE, BC_DENSE_SHARD_MAGIC, BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MAGIC, BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION, BcShardBuildTotals, BcShardDescriptor, BcShardManifest,
    BcShardSidecarManifest, BcShardSplit, BcShardSplitManifest, BcShardSplitMode, DANGER_BYTES,
    DANGER_MASK_BYTES, DENSE_OBS_F32_BYTES, DENSE_REBUILD_MESSAGE, FLAG_BELIEF_FIELDS,
    FLAG_DELTA_Q, FLAG_EXIT, FLAG_MIXTURE_WEIGHTS, FLAG_SAFETY_RESIDUAL, LEGAL_MASK_BYTES,
    OBS_F32_BYTES, OPP_NEXT_BYTES, OPPONENT_COUNT, OPTIONAL_ACTION_FLOAT32_BYTES,
    OPTIONAL_ACTION_MASK_BYTES, ORACLE_FLOAT32_BYTES, ORACLE_MASK_BYTES, PACKED_ACTION_MASK_BYTES,
    PACKED_LEGAL_MASK_BYTES, PACKED_SPATIAL_MASK_BYTES, PLAYER_COUNT, SPATIAL_TARGET_SIZE,
    STORAGE_LAYOUT_COMPACT, TENPAI_BYTES, TILE_COUNT, TILE34_BITSET_BYTES, TILE34_COUNT_BYTES,
    checked_compact_record_size, validate_bc_shard_manifest_contract,
    validate_bc_shard_split_manifest_contract, validate_feature_flags,
};
pub use reader::{BcShardReader, load_bc_shard_reader, read_bc_shard_manifest};
pub use writer::{
    ActiveShardWriter, SplitBuildState, rewrite_shard_header_for_descriptor, write_sample_record,
    write_shard_header,
};
