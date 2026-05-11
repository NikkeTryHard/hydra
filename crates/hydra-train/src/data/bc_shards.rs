//! Backward-compatible BC shard facade.
//!
//! Canonical backend-free shard format code lives in `hydra-bc-shards`.
//! Exec-owned shard building and Burn materialization live in `hydra-train-exec`.

pub use hydra_bc_shards::{
    BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS, BC_SHARD_HEADER_SIZE, BC_SHARD_MAGIC,
    BC_SHARD_MANIFEST_VERSION, BC_SHARD_VERSION, BcShardBuildTotals, BcShardDescriptor,
    BcShardHostBatch, BcShardHostScratch, BcShardManifest, BcShardReader, BcShardSidecarManifest,
    BcShardSplit, BcShardSplitManifest, BcShardSplitMode, DANGER_BYTES, DANGER_MASK_BYTES,
    FLAG_DELTA_Q, FLAG_EXIT, FLAG_SAFETY_RESIDUAL, LEGAL_MASK_BYTES, OBS_F32_BYTES, OPP_NEXT_BYTES,
    OPPONENT_COUNT, OPTIONAL_ACTION_FLOAT32_BYTES, OPTIONAL_ACTION_MASK_BYTES,
    ORACLE_FLOAT32_BYTES, ORACLE_MASK_BYTES, PLAYER_COUNT, SPATIAL_TARGET_SIZE, TENPAI_BYTES,
    TILE_COUNT, load_bc_shard_reader, read_bc_shard_manifest, validate_bc_shard_manifest_contract,
    validate_bc_shard_split_manifest_contract,
};
pub use hydra_bc_shards::{
    BcShardHostBatch as ExtractedBcShardHostBatch, BcShardManifest as ExtractedBcShardManifest,
    BcShardReader as ExtractedBcShardReader, BcShardSplit as ExtractedBcShardReaderSplit,
    BcShardSplit as ExtractedBcShardSplit, load_bc_shard_reader as load_extracted_bc_shard_reader,
    validate_bc_shard_manifest_contract as validate_extracted_bc_shard_manifest_contract,
    validate_bc_shard_split_manifest_contract as validate_extracted_bc_shard_split_manifest_contract,
};
pub use hydra_train_exec::bc_shard_builder::{
    BcShardBuildOutput, BuildBcShardsConfig, build_bc_shards, policy_target_vec_from_actions,
};
pub use hydra_train_exec::epoch_runner::{
    BcShardDeviceBatch as BcShardBatch, materialize_host_batch_owned,
    target_presence_from_host_batch,
};
