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

/// Compact BC shard magic bytes.
pub const BC_SHARD_MAGIC: [u8; 8] = *b"HYBCS3\0\0";
/// Obsolete dense BC shard magic bytes, kept only for hard-error detection.
pub const BC_DENSE_SHARD_MAGIC: [u8; 8] = *b"HYBCS2\0\0";
/// Error emitted when a dense shard is presented to the compact reader.
pub const DENSE_REBUILD_MESSAGE: &str = "dense BC shards are obsolete; rebuild from replay";
/// Compact BC shard binary version.
pub const BC_SHARD_VERSION: u32 = 3;
/// Compact BC shard manifest version.
pub const BC_SHARD_MANIFEST_VERSION: u32 = 3;
/// Compact record layout version inside shard headers.
pub const BC_SHARD_LAYOUT_VERSION: u32 = 1;
/// Compact-only manifest storage layout tag.
pub const STORAGE_LAYOUT_COMPACT: &str = "compact";
/// Compact BC shard header byte length.
pub const BC_SHARD_HEADER_SIZE: u32 = 80;

/// Feature flag enabling safety residual action targets.
pub const FLAG_SAFETY_RESIDUAL: u32 = 1 << 0;
/// Feature flag enabling ExIt action targets.
pub const FLAG_EXIT: u32 = 1 << 1;
/// Feature flag enabling delta-Q action targets.
pub const FLAG_DELTA_Q: u32 = 1 << 2;
/// Feature flag enabling belief-field targets.
pub const FLAG_BELIEF_FIELDS: u32 = 1 << 3;
/// Feature flag enabling mixture-weight targets.
pub const FLAG_MIXTURE_WEIGHTS: u32 = 1 << 4;
/// All compact feature flags supported by this binary.
pub const VALID_FEATURE_FLAGS: u32 =
    FLAG_SAFETY_RESIDUAL | FLAG_EXIT | FLAG_DELTA_Q | FLAG_BELIEF_FIELDS | FLAG_MIXTURE_WEIGHTS;

/// Dense-equivalent observation byte count, for reporting only.
pub const DENSE_OBS_F32_BYTES: usize = OBS_SIZE * 4;
/// Packed legal/action-mask bytes for 46 actions.
pub const PACKED_ACTION_MASK_BYTES: usize = HYDRA_ACTION_SPACE.div_ceil(8);
/// Packed legal-mask byte count per record.
pub const PACKED_LEGAL_MASK_BYTES: usize = PACKED_ACTION_MASK_BYTES;
/// Packed 34 tile-count bytes, using 3 bits per count.
pub const TILE34_COUNT_BYTES: usize = (TILE_COUNT * 3).div_ceil(8);
/// Packed 34-tile bitset byte count.
pub const TILE34_BITSET_BYTES: usize = TILE_COUNT.div_ceil(8);
/// Packed 102-bit spatial mask byte count.
pub const PACKED_SPATIAL_MASK_BYTES: usize = SPATIAL_TARGET_SIZE.div_ceil(8);
/// Encoded oracle-float byte count per record.
pub const ORACLE_FLOAT32_BYTES: usize = PLAYER_COUNT * 4;
/// Encoded oracle-presence byte count per record.
pub const ORACLE_MASK_BYTES: usize = 1;
/// Encoded opponent-next byte count per record.
pub const OPP_NEXT_BYTES: usize = OPPONENT_COUNT;
/// Encoded optional action-float byte count per target.
pub const OPTIONAL_ACTION_FLOAT32_BYTES: usize = HYDRA_ACTION_SPACE * 4;
/// Encoded optional packed action-mask byte count per target.
pub const OPTIONAL_ACTION_MASK_BYTES: usize = PACKED_ACTION_MASK_BYTES;
/// Encoded belief-field byte count per record when enabled.
pub const BELIEF_FIELDS_BYTES: usize = 16 * TILE_COUNT * 4;
/// Encoded mixture-weight byte count per record when enabled.
pub const MIXTURE_WEIGHTS_BYTES: usize = PLAYER_COUNT * 4;
/// Half-open observation channel range in the compact shard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObsChannelRange {
    pub start: usize,
    pub end: usize,
}

const fn obs_channel_count<const N: usize>(ranges: [ObsChannelRange; N]) -> usize {
    let mut idx = 0usize;
    let mut total = 0usize;
    while idx < N {
        total += ranges[idx].end - ranges[idx].start;
        idx += 1;
    }
    total
}
/// Baseline observation fact bytes for exact dense reconstruction of channels 0..85.
///
/// This section stores replay/encoder facts rather than broad dense channel tails:
/// tile counts, tile bitsets, compact metadata scalars, discard temporal indices,
/// and the six exact f32 safety float planes that are not yet integer fact-shaped.
pub const COMPACT_OBS_BASELINE_FACT_BYTES: usize = 1_675;
/// Search/belief scalar channels are absent from compact replay BC shards.
pub const OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_RANGES: [ObsChannelRange; 0] = [];
/// Search/belief and Hand-EV channels are absent from compact replay BC shards.
pub const OBS_ADVANCED_DENSE_CHANNEL_RANGES: [ObsChannelRange; 0] = [];
/// Number of repeated-scalar advanced observation channels.
pub const OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_COUNT: usize =
    obs_channel_count(OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_RANGES);
/// Number of dense advanced observation channels.
pub const OBS_ADVANCED_DENSE_CHANNEL_COUNT: usize =
    obs_channel_count(OBS_ADVANCED_DENSE_CHANNEL_RANGES);
/// Repeated scalar advanced observation bytes stored per record.
pub const COMPACT_OBS_SCALAR_REPEATED_BYTES: usize = 0;
/// Dense advanced observation bytes stored per record.
pub const COMPACT_OBS_DENSE_BYTES: usize = 0;
/// Compact observation bytes rebuilt losslessly by the reader.
pub const COMPACT_OBS_BYTES: usize = COMPACT_OBS_BASELINE_FACT_BYTES;

/// Compact base record byte size without optional action/search targets.
pub const BC_BASE_RECORD_SIZE: u32 = (COMPACT_OBS_BYTES
    + 1
    + PACKED_LEGAL_MASK_BYTES
    + 4
    + 1
    + ORACLE_FLOAT32_BYTES
    + ORACLE_MASK_BYTES
    + 1
    + OPP_NEXT_BYTES
    + PACKED_SPATIAL_MASK_BYTES
    + PACKED_SPATIAL_MASK_BYTES) as u32;
/// Compact maximum record byte size with every optional target.
pub const BC_RECORD_SIZE_WITH_ALL_OPTIONALS: u32 = BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32) * 3
    + BELIEF_FIELDS_BYTES as u32
    + MIXTURE_WEIGHTS_BYTES as u32;

/// Backward-compatible alias for dense-equivalent observation byte reporting.
pub const OBS_F32_BYTES: usize = DENSE_OBS_F32_BYTES;
/// Backward-compatible alias for packed legal-mask bytes.
pub const LEGAL_MASK_BYTES: usize = PACKED_LEGAL_MASK_BYTES;
/// Backward-compatible alias for packed tenpai bytes.
pub const TENPAI_BYTES: usize = 1;
/// Backward-compatible alias for packed danger bytes.
pub const DANGER_BYTES: usize = PACKED_SPATIAL_MASK_BYTES;
/// Backward-compatible alias for packed danger-mask bytes.
pub const DANGER_MASK_BYTES: usize = PACKED_SPATIAL_MASK_BYTES;

/// Returns an error when `flags` contains unsupported compact sections.
pub fn validate_feature_flags(flags: u32) -> Result<(), String> {
    let unknown = flags & !VALID_FEATURE_FLAGS;
    if unknown == 0 {
        Ok(())
    } else {
        Err(format!(
            "BC shard feature_flags contain unsupported bits {unknown:#x}"
        ))
    }
}

/// Returns compact record size for supported feature flags.
pub fn checked_compact_record_size(flags: u32) -> Result<u32, String> {
    validate_feature_flags(flags)?;
    let mut size = BC_BASE_RECORD_SIZE;
    for flag in [FLAG_SAFETY_RESIDUAL, FLAG_EXIT, FLAG_DELTA_Q] {
        if flags & flag != 0 {
            size = size
                .checked_add(OPTIONAL_ACTION_FLOAT32_BYTES as u32)
                .and_then(|value| value.checked_add(OPTIONAL_ACTION_MASK_BYTES as u32))
                .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
        }
    }
    if flags & FLAG_BELIEF_FIELDS != 0 {
        size = size
            .checked_add(BELIEF_FIELDS_BYTES as u32)
            .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        size = size
            .checked_add(MIXTURE_WEIGHTS_BYTES as u32)
            .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
    }
    Ok(size)
}

/// Returns total record payload bytes, rejecting overflow before allocation or I/O.
pub fn checked_record_bytes(sample_count: u64, record_size: u32) -> Result<u64, String> {
    sample_count
        .checked_mul(u64::from(record_size))
        .ok_or_else(|| "BC shard record byte count overflow".to_string())
}

/// Returns encoded record buffer length for an in-memory sample slice.
pub fn checked_encoded_record_len(sample_count: usize, record_size: u32) -> Result<usize, String> {
    let byte_count = checked_record_bytes(sample_count as u64, record_size)?;
    usize::try_from(byte_count).map_err(|_| "BC shard record byte count overflow".to_string())
}

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

    /// Stable manifest string for this mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Train => "train",
            Self::Validation => "validation",
        }
    }
}

fn default_split_mode() -> String {
    BcShardSplitMode::Both.as_str().to_string()
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
    /// Split-selection mode used by the build: both, train, or validation.
    #[serde(default = "default_split_mode")]
    pub split_mode: String,
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
    /// Storage layout tag; compact-only after v3 cutover.
    pub storage_layout: String,
}

/// Validates a BC shard manifest against the current frozen runtime ABI.
pub fn validate_bc_shard_manifest_contract(manifest: &BcShardManifest) -> Result<(), String> {
    if manifest.storage_layout != STORAGE_LAYOUT_COMPACT {
        return Err(format!(
            "BC shard manifest storage_layout {:?} is unsupported; expected compact. Shards must be rebuilt from replay.",
            manifest.storage_layout,
        ));
    }
    if manifest.manifest_version != BC_SHARD_MANIFEST_VERSION {
        return Err(format!(
            "BC shard manifest version {} is unsupported; expected {}. Shards must be rebuilt from replay.",
            manifest.manifest_version, BC_SHARD_MANIFEST_VERSION,
        ));
    }
    if manifest.shard_version != BC_SHARD_VERSION {
        return Err(format!(
            "BC shard version {} is unsupported; expected {}. Shards must be rebuilt from replay.",
            manifest.shard_version, BC_SHARD_VERSION,
        ));
    }
    if manifest.shard_header_size != BC_SHARD_HEADER_SIZE {
        return Err(format!(
            "BC shard header size {} does not match current {}. Shards must be rebuilt from replay.",
            manifest.shard_header_size, BC_SHARD_HEADER_SIZE,
        ));
    }
    if manifest.obs_size != OBS_SIZE {
        return Err(format!(
            "BC shard manifest obs_size {} does not match current OBS_SIZE {} \
             (num_channels: manifest={}, binary={}). \
             Shards must be rebuilt with the current encoder.",
            manifest.obs_size, OBS_SIZE, manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.num_channels != NUM_CHANNELS {
        return Err(format!(
            "BC shard manifest num_channels {} does not match current NUM_CHANNELS {}. \
             Shards must be rebuilt with the current encoder.",
            manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.base_record_size != BC_BASE_RECORD_SIZE {
        return Err(format!(
            "BC shard manifest base_record_size {} does not match current compact BC_BASE_RECORD_SIZE {}. Shards must be rebuilt from replay.",
            manifest.base_record_size, BC_BASE_RECORD_SIZE,
        ));
    }
    if manifest.max_record_size != BC_RECORD_SIZE_WITH_ALL_OPTIONALS {
        return Err(format!(
            "BC shard manifest max_record_size {} does not match current compact max {}. Shards must be rebuilt from replay.",
            manifest.max_record_size, BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        ));
    }
    if manifest.action_space != HYDRA_ACTION_SPACE {
        return Err(format!(
            "BC shard manifest action_space {} does not match current HYDRA_ACTION_SPACE {}. \
             Shards must be rebuilt with the current action contract.",
            manifest.action_space, HYDRA_ACTION_SPACE,
        ));
    }
    let split_mode = match manifest.split_mode.as_str() {
        "both" => BcShardSplitMode::Both,
        "train" => BcShardSplitMode::Train,
        "validation" => BcShardSplitMode::Validation,
        other => {
            return Err(format!(
                "BC shard manifest split_mode {other:?} is unsupported; expected one of both, train, validation"
            ));
        }
    };
    let required_split = match split_mode {
        BcShardSplitMode::Both => None,
        BcShardSplitMode::Train => Some(BcShardSplit::Train),
        BcShardSplitMode::Validation => Some(BcShardSplit::Validation),
    };
    let mut has_train_split = false;
    let mut has_validation_split = false;
    let mut total_samples = 0u64;
    let mut total_shards = 0usize;
    for split in &manifest.splits {
        if !split_mode.includes(split.split) {
            return Err(format!(
                "BC shard manifest split_mode {} excludes {:?} split entries",
                split_mode.as_str(),
                split.split,
            ));
        }
        match split.split {
            BcShardSplit::Train => {
                if has_train_split {
                    return Err(
                        "BC shard manifest contains duplicate train split entries".to_string()
                    );
                }
                has_train_split = true;
            }
            BcShardSplit::Validation => {
                if has_validation_split {
                    return Err(
                        "BC shard manifest contains duplicate validation split entries".to_string(),
                    );
                }
                has_validation_split = true;
            }
        }
        total_samples = total_samples
            .checked_add(split.sample_count)
            .ok_or_else(|| "BC shard manifest split sample_count total overflow".to_string())?;
        total_shards = total_shards
            .checked_add(split.shard_count)
            .ok_or_else(|| "BC shard manifest split shard_count total overflow".to_string())?;
    }
    for split in &manifest.splits {
        validate_bc_shard_split_manifest_contract(split)?;
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
    if manifest.totals.sample_count > 0 {
        match required_split {
            Some(BcShardSplit::Train) if !has_train_split => {
                return Err(
                    "BC shard manifest split_mode train requires a train split entry".to_string(),
                );
            }
            Some(BcShardSplit::Validation) if !has_validation_split => {
                return Err(
                    "BC shard manifest split_mode validation requires a validation split entry"
                        .to_string(),
                );
            }
            None if !has_train_split || !has_validation_split => {
                return Err(
                    "BC shard manifest split_mode both requires train and validation split entries"
                        .to_string(),
                );
            }
            _ => {}
        }
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
        let expected_record_size = checked_compact_record_size(shard.feature_flags)?;
        if shard.record_size != expected_record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match compact record size {} for flags {:#x}",
                idx, split.split, shard.record_size, expected_record_size, shard.feature_flags
            ));
        }
        if shard.record_size != split.record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match split record_size {}",
                idx, split.split, shard.record_size, split.record_size
            ));
        }
        if !is_safe_relative_shard_name(&shard.file_name) {
            return Err(format!(
                "BC shard descriptor {} for {:?} has unsafe file name {:?}",
                idx, split.split, shard.file_name
            ));
        }
        let expected_byte_len = (BC_SHARD_HEADER_SIZE as u64)
            .checked_add(checked_record_bytes(shard.sample_count, shard.record_size)?)
            .ok_or_else(|| "BC shard descriptor byte_len overflow".to_string())?;
        if shard.byte_len != expected_byte_len {
            return Err(format!(
                "BC shard descriptor {} for {:?} byte_len {} does not match header + records {}",
                idx, split.split, shard.byte_len, expected_byte_len
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

fn is_safe_relative_shard_name(name: &str) -> bool {
    let path = std::path::Path::new(name);
    !name.is_empty()
        && path.is_relative()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

#[cfg(test)]
mod tests;
