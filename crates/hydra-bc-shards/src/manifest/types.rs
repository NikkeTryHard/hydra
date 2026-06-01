//! BC shard manifest JSON types.

use serde::{Deserialize, Serialize};

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
