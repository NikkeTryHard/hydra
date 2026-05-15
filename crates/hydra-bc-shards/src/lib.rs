//! Backend-agnostic BC shard host format, manifest, reader, and writer.

pub mod host;
pub mod manifest;
pub mod reader;
pub mod writer;

pub use host::{BcShardHostBatch, BcShardHostScratch, record_size_for_flags};
pub use manifest::{
    BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS, BC_SHARD_HEADER_SIZE, BC_SHARD_MAGIC,
    BC_SHARD_MANIFEST_VERSION, BC_SHARD_VERSION, BcShardBuildTotals, BcShardDescriptor,
    BcShardManifest, BcShardSidecarManifest, BcShardSplit, BcShardSplitManifest, BcShardSplitMode,
    DANGER_BYTES, DANGER_MASK_BYTES, FLAG_DELTA_Q, FLAG_EXIT, FLAG_SAFETY_RESIDUAL,
    LEGAL_MASK_BYTES, OBS_F32_BYTES, OPP_NEXT_BYTES, OPPONENT_COUNT, OPTIONAL_ACTION_FLOAT32_BYTES,
    OPTIONAL_ACTION_MASK_BYTES, ORACLE_FLOAT32_BYTES, ORACLE_MASK_BYTES, PLAYER_COUNT,
    SPATIAL_TARGET_SIZE, TENPAI_BYTES, TILE_COUNT, validate_bc_shard_manifest_contract,
    validate_bc_shard_split_manifest_contract,
};
pub use reader::{BcShardReader, load_bc_shard_reader, read_bc_shard_manifest};
pub use writer::{
    ActiveShardWriter, SplitBuildState, rewrite_shard_header_for_descriptor, write_sample_record,
    write_shard_header,
};
