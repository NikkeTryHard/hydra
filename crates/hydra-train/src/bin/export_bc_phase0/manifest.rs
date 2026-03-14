use std::collections::BTreeMap;
use std::path::PathBuf;

pub(crate) const SCHEMA_VERSION: &str = "hydra_bc_phase0_v1";
pub(crate) const EXPORT_SEMANTICS: &str = "hydra_bc_phase0_v1";
pub(crate) const AUGMENT_POLICY: &str = "train_time_suit_6x_validation_none";
pub(crate) const ENCODER_CONTRACT: &str = "192x34";
pub(crate) const ACTION_SPACE: usize = 46;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExportSplit {
    Train,
    Validation,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ShardArtifactHashes {
    pub(crate) files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ShardManifestEntry {
    pub(crate) split: ExportSplit,
    pub(crate) shard_name: String,
    pub(crate) game_count: usize,
    pub(crate) sample_count: usize,
    pub(crate) hashes: ShardArtifactHashes,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SplitCounts {
    pub(crate) train_games: usize,
    pub(crate) train_samples: usize,
    pub(crate) validation_games: usize,
    pub(crate) validation_samples: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CountsExact {
    pub(crate) source_scan_exact: bool,
    pub(crate) export_counts_exact: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ConfigSnapshot {
    pub(crate) buffer_games: usize,
    pub(crate) buffer_samples: usize,
    pub(crate) archive_queue_bound: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TargetPresenceCounts {
    pub(crate) oracle_target: usize,
    pub(crate) safety_residual_target: usize,
    pub(crate) belief_fields_target: usize,
    pub(crate) mixture_weight_target: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportManifest {
    pub(crate) schema_version: String,
    pub(crate) export_semantics: String,
    pub(crate) manifest_fingerprint: String,
    pub(crate) train_fraction: f32,
    pub(crate) seed: u64,
    pub(crate) augment_policy: String,
    pub(crate) source_paths: Vec<PathBuf>,
    pub(crate) split_counts: SplitCounts,
    pub(crate) counts_exact: CountsExact,
    pub(crate) target_presence_counts: TargetPresenceCounts,
    pub(crate) shards: Vec<ShardManifestEntry>,
    pub(crate) config_snapshot: ConfigSnapshot,
    pub(crate) encoder_contract: String,
    pub(crate) action_space: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ShardMetadata {
    pub(crate) schema_version: String,
    pub(crate) split: ExportSplit,
    pub(crate) shard_name: String,
    pub(crate) game_count: usize,
    pub(crate) sample_count: usize,
}
