//! BC shard manifest contracts and frozen binary ABI constants.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use serde::{Deserialize, Serialize};

/// Opponent count encoded in spatial auxiliary targets.
pub const OPPONENT_COUNT: usize = 3;
/// Player count encoded in oracle targets.
pub const PLAYER_COUNT: usize = 4;
/// Tile count encoded per observation channel.
pub const TILE_COUNT: usize = 34;
/// Opponent-by-tile spatial target size.
pub const SPATIAL_TARGET_SIZE: usize = OPPONENT_COUNT * TILE_COUNT;

/// Frozen BC shard magic bytes.
pub const BC_SHARD_MAGIC: [u8; 8] = *b"HYBCS2\0\0";
/// Frozen BC shard binary version.
pub const BC_SHARD_VERSION: u32 = 2;
/// Frozen BC shard manifest version.
pub const BC_SHARD_MANIFEST_VERSION: u32 = 2;
/// Frozen BC shard header byte length.
pub const BC_SHARD_HEADER_SIZE: u32 = 80;

/// Feature flag enabling safety residual action targets.
pub const FLAG_SAFETY_RESIDUAL: u32 = 1 << 0;
/// Feature flag enabling ExIt action targets.
pub const FLAG_EXIT: u32 = 1 << 1;
/// Feature flag enabling delta-Q action targets.
pub const FLAG_DELTA_Q: u32 = 1 << 2;

/// Encoded observation byte count per record.
pub const OBS_F32_BYTES: usize = OBS_SIZE * 4;
/// Encoded legal-mask byte count per record.
pub const LEGAL_MASK_BYTES: usize = HYDRA_ACTION_SPACE;
/// Encoded oracle-float byte count per record.
pub const ORACLE_FLOAT32_BYTES: usize = PLAYER_COUNT * 4;
/// Encoded oracle-mask byte count per record.
pub const ORACLE_MASK_BYTES: usize = 1;
/// Encoded tenpai byte count per record.
pub const TENPAI_BYTES: usize = OPPONENT_COUNT;
/// Encoded opponent-next byte count per record.
pub const OPP_NEXT_BYTES: usize = OPPONENT_COUNT;
/// Encoded danger byte count per record.
pub const DANGER_BYTES: usize = SPATIAL_TARGET_SIZE;
/// Encoded danger-mask byte count per record.
pub const DANGER_MASK_BYTES: usize = SPATIAL_TARGET_SIZE;
/// Encoded optional action-float byte count per target.
pub const OPTIONAL_ACTION_FLOAT32_BYTES: usize = HYDRA_ACTION_SPACE * 4;
/// Encoded optional action-mask byte count per target.
pub const OPTIONAL_ACTION_MASK_BYTES: usize = HYDRA_ACTION_SPACE;

/// Frozen base record byte size without optional action targets.
pub const BC_BASE_RECORD_SIZE: u32 = (OBS_F32_BYTES
    + 1
    + LEGAL_MASK_BYTES
    + 4
    + 1
    + ORACLE_FLOAT32_BYTES
    + ORACLE_MASK_BYTES
    + TENPAI_BYTES
    + OPP_NEXT_BYTES
    + DANGER_BYTES
    + DANGER_MASK_BYTES) as u32;
/// Frozen maximum record byte size with every optional action target.
pub const BC_RECORD_SIZE_WITH_ALL_OPTIONALS: u32 = BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32) * 3;

/// BC shard data split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BcShardSplit {
    /// Training split.
    Train,
    /// Validation split.
    Validation,
}

impl BcShardSplit {
    /// File-name prefix for this split.
    pub const fn shard_prefix(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "val",
        }
    }

    /// Binary header split id for this split.
    pub const fn split_id(self) -> u32 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
        }
    }
}

/// Split-selection mode used while building BC shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcShardSplitMode {
    /// Include both train and validation splits.
    Both,
    /// Include only train split.
    Train,
    /// Include only validation split.
    Validation,
}

impl BcShardSplitMode {
    /// Returns whether this mode includes `split`.
    pub const fn includes(self, split: BcShardSplit) -> bool {
        matches!(
            (self, split),
            (Self::Both, _)
                | (Self::Train, BcShardSplit::Train)
                | (Self::Validation, BcShardSplit::Validation)
        )
    }
}

/// Optional sidecar provenance embedded in a BC shard manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSidecarManifest {
    /// Sidecar path recorded at build time.
    pub path: String,
    /// Source network hash recorded in the sidecar metadata.
    pub source_net_hash: u64,
    /// Source version recorded in the sidecar metadata.
    pub source_version: u32,
}

/// One shard file descriptor inside a split manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardDescriptor {
    /// Split containing this shard.
    pub split: BcShardSplit,
    /// Zero-based shard index within the split.
    pub shard_index: usize,
    /// File name relative to manifest directory.
    pub file_name: String,
    /// Number of samples in this shard.
    pub sample_count: u64,
    /// First global sample index in this split.
    pub first_sample_index: u64,
    /// File length in bytes.
    pub byte_len: u64,
    /// Feature flags used by this shard.
    pub feature_flags: u32,
    /// Record size used by this shard.
    pub record_size: u32,
}

/// Per-split BC shard manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSplitManifest {
    /// Split described by this manifest.
    pub split: BcShardSplit,
    /// Number of shard descriptors.
    pub shard_count: usize,
    /// Total samples in the split.
    pub sample_count: u64,
    /// Feature flags shared by the split.
    pub feature_flags: u32,
    /// Record size shared by the split.
    pub record_size: u32,
    /// Shard descriptors in contiguous order.
    pub shards: Vec<BcShardDescriptor>,
}

/// Aggregate BC shard build totals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BcShardBuildTotals {
    /// Total written samples.
    pub sample_count: u64,
    /// Skipped replay games.
    pub skipped_games: u64,
    /// Empty replay games.
    pub empty_games: u64,
    /// Total written shards.
    pub shard_count: usize,
}

/// Top-level BC shard manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardManifest {
    /// Manifest schema version.
    pub manifest_version: u32,
    /// Shard binary version.
    pub shard_version: u32,
    /// Shard header size.
    pub shard_header_size: u32,
    /// Base record size.
    pub base_record_size: u32,
    /// Maximum record size.
    pub max_record_size: u32,
    /// Observation float count.
    pub obs_size: usize,
    /// Observation channel count.
    pub num_channels: usize,
    /// Action-space size.
    pub action_space: usize,
    /// Train split fraction used by the build.
    pub train_fraction: f32,
    /// Target samples per shard used by the build.
    pub shard_samples: usize,
    /// Whether runtime augmentation is expected.
    pub augment_runtime: bool,
    /// Input path recorded by the build.
    pub input: String,
    /// Output directory recorded by the build.
    pub output_dir: String,
    /// Build timestamp.
    pub created_at: String,
    /// Source count hint.
    pub source_count: usize,
    /// Source total games hint.
    pub source_total_games_hint: usize,
    /// Source train count hint.
    pub source_train_count_hint: usize,
    /// Source validation count hint.
    pub source_val_count_hint: usize,
    /// Whether source counts are exact.
    pub source_counts_exact: bool,
    /// Optional ExIt sidecar metadata.
    pub exit_sidecar: Option<BcShardSidecarManifest>,
    /// Optional delta-Q sidecar metadata.
    pub delta_q_sidecar: Option<BcShardSidecarManifest>,
    /// Aggregate totals.
    pub totals: BcShardBuildTotals,
    /// Split manifests.
    pub splits: Vec<BcShardSplitManifest>,
}

/// Validates a BC shard manifest against the current frozen runtime ABI.
pub fn validate_bc_shard_manifest_contract(manifest: &BcShardManifest) -> Result<(), String> {
    if manifest.obs_size != OBS_SIZE {
        return Err(format!(
            "BC shard manifest obs_size {} does not match current OBS_SIZE {} \
             (num_channels: manifest={}, binary={}). \
             Shards must be rebuilt with the current encoder.",
            manifest.obs_size, OBS_SIZE, manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.base_record_size != BC_BASE_RECORD_SIZE {
        return Err(format!(
            "BC shard manifest base_record_size {} does not match current \
             BC_BASE_RECORD_SIZE {}. Shards must be rebuilt with the current encoder.",
            manifest.base_record_size, BC_BASE_RECORD_SIZE,
        ));
    }
    if manifest.action_space != HYDRA_ACTION_SPACE {
        return Err(format!(
            "BC shard manifest action_space {} does not match current HYDRA_ACTION_SPACE {}. \
             Shards must be rebuilt with the current action contract.",
            manifest.action_space, HYDRA_ACTION_SPACE,
        ));
    }
    let mut total_samples = 0u64;
    let mut total_shards = 0usize;
    for split in &manifest.splits {
        validate_bc_shard_split_manifest_contract(split)?;
        total_samples += split.sample_count;
        total_shards += split.shard_count;
    }
    if manifest.totals.sample_count != total_samples {
        return Err(format!(
            "BC shard manifest totals.sample_count {} does not match split total {}",
            manifest.totals.sample_count, total_samples
        ));
    }
    if manifest.totals.shard_count != total_shards {
        return Err(format!(
            "BC shard manifest totals.shard_count {} does not match split shard total {}",
            manifest.totals.shard_count, total_shards
        ));
    }
    Ok(())
}

/// Validates split-level shard descriptor contiguity and consistency.
pub fn validate_bc_shard_split_manifest_contract(
    split: &BcShardSplitManifest,
) -> Result<(), String> {
    if split.shard_count != split.shards.len() {
        return Err(format!(
            "BC shard manifest {:?} shard_count {} does not match descriptor count {}",
            split.split,
            split.shard_count,
            split.shards.len()
        ));
    }
    let mut expected_start = 0u64;
    for (idx, shard) in split.shards.iter().enumerate() {
        if shard.split != split.split {
            return Err(format!(
                "BC shard descriptor {} has split {:?}, expected {:?}",
                idx, shard.split, split.split
            ));
        }
        if shard.shard_index != idx {
            return Err(format!(
                "BC shard descriptor for {:?} has shard_index {}, expected {}",
                split.split, shard.shard_index, idx
            ));
        }
        if shard.first_sample_index != expected_start {
            return Err(format!(
                "BC shard descriptor {} for {:?} starts at {}, expected contiguous start {}",
                idx, split.split, shard.first_sample_index, expected_start
            ));
        }
        if shard.feature_flags != split.feature_flags {
            return Err(format!(
                "BC shard descriptor {} for {:?} feature_flags {} does not match split feature_flags {}",
                idx, split.split, shard.feature_flags, split.feature_flags
            ));
        }
        if shard.record_size != split.record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match split record_size {}",
                idx, split.split, shard.record_size, split.record_size
            ));
        }
        expected_start = expected_start
            .checked_add(shard.sample_count)
            .ok_or_else(|| "BC shard split sample_count overflow".to_string())?;
    }
    if split.sample_count != expected_start {
        return Err(format!(
            "BC shard split {:?} sample_count {} does not match descriptor total {}",
            split.split, split.sample_count, expected_start
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
