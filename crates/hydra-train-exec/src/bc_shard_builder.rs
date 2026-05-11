//! Exec-owned BC shard builder.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hydra_bc_shards::{
    BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS, BC_SHARD_HEADER_SIZE,
    BC_SHARD_MANIFEST_VERSION, BC_SHARD_VERSION, BcShardBuildTotals, BcShardManifest,
    BcShardSidecarManifest, BcShardSplit, BcShardSplitMode, FLAG_DELTA_Q, FLAG_EXIT,
    FLAG_SAFETY_RESIDUAL, validate_bc_shard_manifest_contract,
};

use crate::data_pipeline::{
    compact_error_message, compact_identity, identity_for_archive_entry, identity_for_loose_file,
    is_mjai_archive_entry, is_tar_zst_file, is_train_game, scan_data_sources,
};
use hydra_data_core::{DataManifest, DataSource};
use hydra_replay_loader::mjai_loader::{
    MjaiGame, ReplayLoadPolicy, SidecarProvenance, invalid_data, load_game_from_path_with_policy,
    load_game_from_stream_with_policy,
};
use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};

/// Builds a dense policy-target matrix from sparse action IDs.
pub fn policy_target_vec_from_actions(actions: &[i64], batch_size: usize) -> Vec<f32> {
    let mut policy_target = vec![0.0f32; batch_size * hydra_core::action::HYDRA_ACTION_SPACE];
    for (row, &action) in actions.iter().take(batch_size).enumerate() {
        if action >= 0 {
            let action = action as usize;
            if action < hydra_core::action::HYDRA_ACTION_SPACE {
                policy_target[row * hydra_core::action::HYDRA_ACTION_SPACE + action] = 1.0;
            }
        }
    }
    policy_target
}

/// Configuration for building backend-agnostic BC shard files from MJAI sources.
#[derive(Debug, Clone)]
pub struct BuildBcShardsConfig {
    /// Input loose file, directory, or archive path.
    pub input: PathBuf,
    /// Directory where shard files and manifest are written.
    pub output_dir: PathBuf,
    /// Manifest filename under `output_dir`.
    pub manifest_name: String,
    /// Train/validation split fraction.
    pub train_fraction: f32,
    /// Maximum samples per shard file.
    pub shard_samples: usize,
    /// Which train/validation splits to build.
    pub split_mode: BcShardSplitMode,
    /// Optional pre-scanned source manifest.
    pub source_manifest: Option<DataManifest>,
    /// Optional hydrated ExIt sidecar index.
    pub exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    /// Optional ExIt sidecar path recorded in the manifest.
    pub exit_sidecar_path: Option<PathBuf>,
    /// ExIt sidecar provenance required for hydration.
    pub exit_provenance: SidecarProvenance,
    /// Optional hydrated delta-Q sidecar index.
    pub delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    /// Optional delta-Q sidecar path recorded in the manifest.
    pub delta_q_sidecar_path: Option<PathBuf>,
    /// Delta-Q sidecar provenance required for hydration.
    pub delta_q_provenance: SidecarProvenance,
}

impl Default for BuildBcShardsConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::from("."),
            output_dir: PathBuf::from("bc_shards"),
            manifest_name: "bc_shards_manifest.json".to_string(),
            train_fraction: 0.9,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Both,
            source_manifest: None,
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        }
    }
}

/// Result of a BC shard build.
#[derive(Debug, Clone)]
pub struct BcShardBuildOutput {
    /// Path to the written manifest.
    pub manifest_path: PathBuf,
    /// In-memory manifest that was written.
    pub manifest: BcShardManifest,
}

/// Builds BC shard files and writes a manifest.
pub fn build_bc_shards(config: &BuildBcShardsConfig) -> io::Result<BcShardBuildOutput> {
    if config.shard_samples == 0 {
        return Err(invalid_data("shard_samples must be > 0"));
    }
    std::fs::create_dir_all(&config.output_dir)?;
    let source_manifest = match &config.source_manifest {
        Some(manifest) => manifest.clone(),
        None => scan_data_sources(&config.input)?,
    };
    let feature_flags = feature_flags_from_config(config);

    let mut train_state = config
        .split_mode
        .includes(BcShardSplit::Train)
        .then(|| hydra_bc_shards::SplitBuildState::new(BcShardSplit::Train, feature_flags));
    let mut val_state = config
        .split_mode
        .includes(BcShardSplit::Validation)
        .then(|| hydra_bc_shards::SplitBuildState::new(BcShardSplit::Validation, feature_flags));
    let mut skipped_games = 0u64;
    let mut empty_games = 0u64;

    for source in &source_manifest.sources {
        match source {
            DataSource::LooseFile(path) => process_loose_file(
                path,
                config,
                &mut train_state,
                &mut val_state,
                &mut skipped_games,
                &mut empty_games,
            )?,
            DataSource::Archive(path) => process_archive(
                path,
                config,
                &mut train_state,
                &mut val_state,
                &mut skipped_games,
                &mut empty_games,
            )?,
            DataSource::ParsedSampleCache { path, .. } => {
                return Err(invalid_data(format!(
                    "parsed-sample cache input is not supported by build_bc_shards yet: {}",
                    path.display()
                )));
            }
        }
    }

    let mut split_manifests = Vec::new();
    if let Some(train_state) = train_state {
        split_manifests.push(train_state.finalize()?);
    }
    if let Some(val_state) = val_state {
        split_manifests.push(val_state.finalize()?);
    }

    let mut totals = BcShardBuildTotals {
        skipped_games,
        empty_games,
        ..BcShardBuildTotals::default()
    };
    for split in &split_manifests {
        totals.sample_count += split.sample_count;
        totals.shard_count += split.shard_count;
    }

    let manifest = BcShardManifest {
        manifest_version: BC_SHARD_MANIFEST_VERSION,
        shard_version: BC_SHARD_VERSION,
        shard_header_size: BC_SHARD_HEADER_SIZE,
        base_record_size: BC_BASE_RECORD_SIZE,
        max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        obs_size: hydra_core::encoder::OBS_SIZE,
        num_channels: hydra_core::encoder::NUM_CHANNELS,
        action_space: hydra_core::action::HYDRA_ACTION_SPACE,
        train_fraction: config.train_fraction,
        shard_samples: config.shard_samples,
        augment_runtime: true,
        input: config.input.display().to_string(),
        output_dir: config.output_dir.display().to_string(),
        created_at: time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
        source_count: source_manifest.sources.len(),
        source_total_games_hint: source_manifest.total_games,
        source_train_count_hint: source_manifest.train_count,
        source_val_count_hint: source_manifest.val_count,
        source_counts_exact: source_manifest.counts_exact,
        exit_sidecar: sidecar_manifest(config.exit_sidecar_path.as_deref(), config.exit_provenance),
        delta_q_sidecar: sidecar_manifest(
            config.delta_q_sidecar_path.as_deref(),
            config.delta_q_provenance,
        ),
        totals,
        splits: split_manifests,
    };
    validate_bc_shard_manifest_contract(&manifest).map_err(invalid_data)?;
    let manifest_path = config.output_dir.join(&config.manifest_name);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|err| invalid_data(format!("failed to serialize BC shard manifest: {err}")))?,
    )?;
    Ok(BcShardBuildOutput {
        manifest_path,
        manifest,
    })
}

fn feature_flags_from_config(config: &BuildBcShardsConfig) -> u32 {
    let mut flags = FLAG_SAFETY_RESIDUAL;
    if config.exit_sidecar.is_some() {
        flags |= FLAG_EXIT;
    }
    if config.delta_q_sidecar.is_some() {
        flags |= FLAG_DELTA_Q;
    }
    flags
}

fn sidecar_manifest(
    path: Option<&Path>,
    provenance: SidecarProvenance,
) -> Option<BcShardSidecarManifest> {
    let (source_net_hash, source_version) =
        provenance.source_net_hash.zip(provenance.source_version)?;
    Some(BcShardSidecarManifest {
        path: path?.display().to_string(),
        source_net_hash,
        source_version,
    })
}

fn replay_target_profile_for_bc_shards(
    config: &BuildBcShardsConfig,
) -> hydra_replay_loader::ReplayTargetProfile {
    hydra_replay_loader::ReplayTargetProfile::with_optional_heads(
        false,
        false,
        false,
        false,
        config.exit_sidecar.is_some(),
        config.delta_q_sidecar.is_some(),
    )
}

fn replay_load_policy_for_bc_shards(config: &BuildBcShardsConfig) -> ReplayLoadPolicy<'_> {
    ReplayLoadPolicy::new(
        replay_target_profile_for_bc_shards(config),
        config.exit_provenance,
        config.delta_q_provenance,
        config.exit_sidecar.as_deref(),
        config.delta_q_sidecar.as_deref(),
    )
}

fn load_bc_shard_game_from_path(path: &Path, config: &BuildBcShardsConfig) -> io::Result<MjaiGame> {
    let policy = replay_load_policy_for_bc_shards(config);
    load_game_from_path_with_policy(path, Some(&policy))
}

fn load_bc_shard_game_from_stream<R: io::Read>(
    identity: &str,
    stream: R,
    config: &BuildBcShardsConfig,
) -> io::Result<MjaiGame> {
    let policy = replay_load_policy_for_bc_shards(config);
    load_game_from_stream_with_policy(identity, stream, Some(&policy))
}

struct LoadedGameContext<'a> {
    config: &'a BuildBcShardsConfig,
    train_state: &'a mut Option<hydra_bc_shards::SplitBuildState>,
    val_state: &'a mut Option<hydra_bc_shards::SplitBuildState>,
    skipped_games: &'a mut u64,
    empty_games: &'a mut u64,
}

fn process_loose_file(
    path: &Path,
    config: &BuildBcShardsConfig,
    train_state: &mut Option<hydra_bc_shards::SplitBuildState>,
    val_state: &mut Option<hydra_bc_shards::SplitBuildState>,
    skipped_games: &mut u64,
    empty_games: &mut u64,
) -> io::Result<()> {
    let identity = identity_for_loose_file(path)?;
    let Some(split) = split_for_identity(&identity, config) else {
        return Ok(());
    };
    let result = load_bc_shard_game_from_path(path, config);
    let mut ctx = LoadedGameContext {
        config,
        train_state,
        val_state,
        skipped_games,
        empty_games,
    };
    handle_loaded_game(&identity, split, result, &mut ctx)
}

fn process_archive(
    path: &Path,
    config: &BuildBcShardsConfig,
    train_state: &mut Option<hydra_bc_shards::SplitBuildState>,
    val_state: &mut Option<hydra_bc_shards::SplitBuildState>,
    skipped_games: &mut u64,
    empty_games: &mut u64,
) -> io::Result<()> {
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn io::Read> = if is_tar_zst_file(path) {
        let zstd = zstd::Decoder::new(file).map_err(|err| {
            io::Error::other(format!(
                "failed to open zstd archive {}: {err}",
                path.display()
            ))
        })?;
        Box::new(zstd)
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);

    for entry_result in archive.entries()? {
        let entry = entry_result?;
        let entry_path = entry.path()?.into_owned();
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }
        let identity = identity_for_archive_entry(path, &entry_path)?;
        let Some(split) = split_for_identity(&identity, config) else {
            continue;
        };
        let result = load_bc_shard_game_from_stream(&identity, entry, config);
        let mut ctx = LoadedGameContext {
            config,
            train_state,
            val_state,
            skipped_games,
            empty_games,
        };
        handle_loaded_game(&identity, split, result, &mut ctx)?;
    }
    Ok(())
}

fn handle_loaded_game(
    identity: &str,
    split: BcShardSplit,
    result: io::Result<MjaiGame>,
    ctx: &mut LoadedGameContext<'_>,
) -> io::Result<()> {
    match result {
        Ok(game) => {
            if game.samples.is_empty() {
                *ctx.empty_games += 1;
                return Ok(());
            }
            let state = match split {
                BcShardSplit::Train => ctx.train_state.as_mut(),
                BcShardSplit::Validation => ctx.val_state.as_mut(),
            };
            if let Some(state) = state {
                state.push_samples(
                    &ctx.config.output_dir,
                    ctx.config.shard_samples,
                    &game.samples,
                )?;
            }
        }
        Err(err) => {
            *ctx.skipped_games += 1;
            eprintln!(
                "Skipping {}: {}",
                compact_identity(identity),
                compact_error_message(&err)
            );
        }
    }
    Ok(())
}

fn split_for_identity(identity: &str, config: &BuildBcShardsConfig) -> Option<BcShardSplit> {
    let split = if is_train_game(identity, config.train_fraction) {
        BcShardSplit::Train
    } else {
        BcShardSplit::Validation
    };
    config.split_mode.includes(split).then_some(split)
}

#[cfg(test)]
mod tests;
