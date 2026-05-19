//! BC shard manifest adapters for execution bootstrap.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use hydra_bc_shards::{BcShardManifest, BcShardSplit};
use hydra_data_core::{DataManifest, DataSource, SourceFilterConfig};

/// Minimal config contract required to validate BC shard manifests before use.
#[derive(Debug, Clone, Copy)]
pub struct BcShardManifestConfigRef<'a> {
    /// Configured training fraction. Must exactly match the shard build contract.
    pub train_fraction: f32,
    /// Configured source filter contract. Shard manifests currently do not record filters.
    pub source_filters: &'a SourceFilterConfig,
    /// Configured ExIt sidecar path, required when ExIt loss is weighted.
    pub exit_sidecar_path: Option<&'a Path>,
    /// Configured delta-Q sidecar path, required when delta-Q loss is weighted.
    pub delta_q_sidecar_path: Option<&'a Path>,
    /// Configured ExIt loss weight.
    pub exit_loss_weight: Option<f32>,
    /// Configured delta-Q loss weight.
    pub delta_q_loss_weight: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainBcShardManifestCompat {
    manifest_version: u32,
    shard_version: u32,
    shard_header_size: u32,
    base_record_size: u32,
    max_record_size: u32,
    obs_size: usize,
    num_channels: usize,
    action_space: usize,
    train_fraction: f32,
    shard_samples: usize,
    split_mode: String,
    augment_runtime: bool,
    input: String,
    output_dir: String,
    created_at: String,
    source_count: usize,
    source_total_games_hint: usize,
    source_train_count_hint: usize,
    source_val_count_hint: usize,
    source_counts_exact: bool,
    exit_sidecar: Option<hydra_bc_shards::BcShardSidecarManifest>,
    delta_q_sidecar: Option<hydra_bc_shards::BcShardSidecarManifest>,
    totals: hydra_bc_shards::BcShardBuildTotals,
    splits: Vec<hydra_bc_shards::BcShardSplitManifest>,
    storage_layout: String,
}

impl From<TrainBcShardManifestCompat> for BcShardManifest {
    fn from(manifest: TrainBcShardManifestCompat) -> Self {
        Self {
            manifest_version: manifest.manifest_version,
            shard_version: manifest.shard_version,
            shard_header_size: manifest.shard_header_size,
            base_record_size: manifest.base_record_size,
            max_record_size: manifest.max_record_size,
            obs_size: manifest.obs_size,
            num_channels: manifest.num_channels,
            action_space: manifest.action_space,
            train_fraction: manifest.train_fraction,
            shard_samples: manifest.shard_samples,
            split_mode: manifest.split_mode,
            augment_runtime: manifest.augment_runtime,
            input: manifest.input,
            output_dir: manifest.output_dir,
            created_at: manifest.created_at,
            source_count: manifest.source_count,
            source_total_games_hint: manifest.source_total_games_hint,
            source_train_count_hint: manifest.source_train_count_hint,
            source_val_count_hint: manifest.source_val_count_hint,
            source_counts_exact: manifest.source_counts_exact,
            exit_sidecar: manifest.exit_sidecar,
            delta_q_sidecar: manifest.delta_q_sidecar,
            totals: manifest.totals,
            splits: manifest.splits,
            storage_layout: manifest.storage_layout,
        }
    }
}

/// Reads and validates a BC shard manifest from disk.
pub fn read_bc_shard_manifest(manifest_path: &Path) -> Result<BcShardManifest, String> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|err| {
        format!(
            "failed to read BC shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let manifest: TrainBcShardManifestCompat = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse BC shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let manifest = BcShardManifest::from(manifest);
    hydra_bc_shards::validate_bc_shard_manifest_contract(&manifest)?;
    Ok(manifest)
}

/// Converts a BC shard manifest into the legacy data manifest shape used by training bootstrap.
#[must_use]
pub fn data_manifest_from_bc_shard_manifest(manifest: &BcShardManifest) -> DataManifest {
    let train_count = manifest
        .splits
        .iter()
        .find(|split| split.split == BcShardSplit::Train)
        .map(|split| split.sample_count as usize)
        .unwrap_or(0);
    let val_count = manifest
        .splits
        .iter()
        .find(|split| split.split == BcShardSplit::Validation)
        .map(|split| split.sample_count as usize)
        .unwrap_or(0);

    DataManifest {
        sources: vec![DataSource::LooseFile(PathBuf::from(&manifest.input))],
        total_games: manifest.source_total_games_hint,
        train_count,
        val_count,
        counts_exact: true,
    }
}

/// Validates shard manifest settings against the training config contract.
pub fn validate_bc_shard_manifest_for_config(
    manifest: &BcShardManifest,
    config: BcShardManifestConfigRef<'_>,
) -> Result<(), String> {
    if manifest.train_fraction.to_bits() != config.train_fraction.to_bits() {
        return Err(format!(
            "BC shard manifest train_fraction {} does not match config train_fraction {}. Rebuild shards or use matching config.",
            manifest.train_fraction, config.train_fraction
        ));
    }
    if !config.source_filters.is_empty() {
        return Err(
            "BC shard manifest does not record source_filters; shard-backed BC requires empty source_filters or shards rebuilt with an explicit recorded filter contract"
                .to_string(),
        );
    }

    if config.exit_loss_weight.is_some_and(|weight| weight > 0.0) {
        let configured = config
            .exit_sidecar_path
            .ok_or_else(|| "advanced_loss.exit requires exit_sidecar_path".to_string())?;
        let manifest_sidecar = manifest.exit_sidecar.as_ref().ok_or_else(|| {
            "advanced_loss.exit requires BC shards built with matching ExIt sidecar".to_string()
        })?;
        if manifest_sidecar.path != configured.display().to_string() {
            return Err(format!(
                "BC shard ExIt sidecar {} does not match config exit_sidecar_path {}",
                manifest_sidecar.path,
                configured.display()
            ));
        }
    }
    if config
        .delta_q_loss_weight
        .is_some_and(|weight| weight > 0.0)
    {
        let configured = config
            .delta_q_sidecar_path
            .ok_or_else(|| "advanced_loss.delta_q requires delta_q_sidecar_path".to_string())?;
        let manifest_sidecar = manifest.delta_q_sidecar.as_ref().ok_or_else(|| {
            "advanced_loss.delta_q requires BC shards built with matching delta_q sidecar"
                .to_string()
        })?;
        if manifest_sidecar.path != configured.display().to_string() {
            return Err(format!(
                "BC shard delta_q sidecar {} does not match config delta_q_sidecar_path {}",
                manifest_sidecar.path,
                configured.display()
            ));
        }
    }
    Ok(())
}
