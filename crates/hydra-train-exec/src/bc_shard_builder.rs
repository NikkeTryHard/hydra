//! Exec-owned BC shard builder.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use hydra_bc_shards::{
    ActiveShardWriter, BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MANIFEST_VERSION, BC_SHARD_VERSION,
    BcShardBuildTotals, BcShardDescriptor, BcShardManifest, BcShardSidecarManifest, BcShardSplit,
    BcShardSplitManifest, BcShardSplitMode, FLAG_DELTA_Q, FLAG_EXIT, FLAG_SAFETY_RESIDUAL,
    STORAGE_LAYOUT_COMPACT, checked_compact_record_size, encode_sample_records,
    record_size_for_flags, rewrite_shard_header_for_descriptor,
    validate_bc_shard_manifest_contract, validate_bc_shard_split_manifest_contract,
};
use rayon::ThreadPoolBuilder;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::data_pipeline::{
    compact_error_message, compact_identity, identity_for_archive_entry, identity_for_loose_file,
    is_mjai_archive_entry, is_tar_zst_file, is_train_game, scan_data_sources,
};
use hydra_data_core::{DataManifest, DataSource};
use hydra_replay_loader::mjai_loader::{
    MjaiGame, ReplayLoadPolicy, ReplayObservationProfile, SidecarProvenance, invalid_data,
    load_game_from_path_with_policy, load_game_from_stream_with_policy,
};
use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};

const DEFAULT_REPORT_NAME: &str = "bc_shard_build_report.json";
const DEFAULT_RESUME_DIR_NAME: &str = ".hydra-bc-build-state";
const PROGRESS_EVERY_GAMES: u64 = 1_000;
const REPORT_SCHEMA_VERSION: u32 = 1;
const RESUME_SCHEMA_VERSION: u32 = 1;
const SAMPLE_CAP_OUTCOME_NONE: &str = "none";
const SAMPLE_CAP_OUTCOME_NOT_REACHED: &str = "not_reached";
const SAMPLE_CAP_OUTCOME_REACHED_EXACT: &str = "reached_exact";
const SAMPLE_CAP_OUTCOME_REACHED_AFTER_CURRENT_GAME: &str = "reached_after_current_game";

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
    /// Maximum included game attempts after split/filter.
    pub max_games: Option<usize>,
    /// Maximum samples; current game is finished atomically after reaching cap.
    pub max_samples: Option<usize>,
    /// Worker threads for replay materialization. `None` uses Rayon defaults.
    pub num_threads: Option<usize>,
    /// Bounded job/result queue depth.
    pub queue_bound: usize,
    /// Enable committed-fragment resume.
    pub resume: bool,
    /// Resume state dir. Defaults under `output_dir`.
    pub resume_dir: Option<PathBuf>,
    /// Included-game count per committed fragment.
    pub chunk_games: usize,
    /// Optional build report filename under `output_dir`; `None` disables report.
    pub report_name: Option<String>,
    /// Optional progress JSONL filename under `output_dir`; `None` disables progress JSONL.
    pub progress_jsonl_name: Option<String>,
    /// Maximum retained bad-source examples.
    pub max_error_examples: usize,
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
            max_games: None,
            max_samples: None,
            num_threads: None,
            queue_bound: 128,
            resume: false,
            resume_dir: None,
            chunk_games: 10_000,
            report_name: Some(DEFAULT_REPORT_NAME.to_string()),
            progress_jsonl_name: None,
            max_error_examples: 32,
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
    /// Optional path to the operational report.
    pub report_path: Option<PathBuf>,
    /// Optional operational report.
    pub report: Option<BcShardBuildReport>,
}

/// Operational BC shard build report. This is not part of manifest schema v2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardBuildReport {
    /// Report schema version.
    pub schema_version: u32,
    /// UTC RFC3339 start timestamp.
    pub started_at: String,
    /// UTC RFC3339 finish timestamp.
    pub finished_at: String,
    /// Wall-clock elapsed seconds.
    pub elapsed_seconds: f64,
    /// Frozen compact shard ABI summary.
    pub abi: BcShardAbiReport,
    /// Output disk summary.
    pub disk: BcShardDiskReport,
    /// Planned games per split after filtering and caps.
    pub plan_splits: Vec<BcShardSplitPlanReport>,
    /// Command/config summary.
    pub command: BcShardBuildCommandReport,
    /// Source scan summary.
    pub scan: BcShardScanReport,
    /// Materialization summary.
    pub build: BcShardMaterializationReport,
    /// Output summary.
    pub output: BcShardOutputReport,
    /// Derived rates.
    pub rates: BcShardBuildRates,
    /// Manifest path.
    pub manifest_path: String,
    /// Optional progress JSONL path.
    pub progress_jsonl_path: Option<String>,
}

/// Disk summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardDiskReport {
    /// Output directory path.
    pub output_dir: String,
    /// Existing output bytes before this build writes files.
    pub existing_output_bytes: u64,
    /// Available filesystem bytes before the build, when supported.
    pub available_bytes_before: Option<u64>,
    /// Available filesystem bytes after the build, when supported.
    pub available_bytes_after: Option<u64>,
    /// Projected output bytes before build, when trustworthy.
    pub projected_output_bytes: Option<u64>,
    /// Projected sample count before build, when trustworthy.
    pub projected_sample_count: Option<u64>,
    /// Projection source label.
    pub projection_source: String,
}

/// ABI summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardAbiReport {
    /// Storage layout tag.
    pub storage_layout: String,
    /// Manifest schema version.
    pub manifest_version: u32,
    /// Shard binary version.
    pub shard_version: u32,
    /// Shard record layout version.
    pub layout_version: u32,
    /// Shard header byte size.
    pub header_size: u32,
    /// Base compact record byte size.
    pub base_record_size: u32,
    /// Maximum compact record byte size.
    pub max_record_size: u32,
    /// Dense observation f32 bytes per sample.
    pub dense_obs_f32_bytes_per_sample: u64,
    /// Enabled compact feature flags.
    pub feature_flags: u32,
    /// Compact record byte size for enabled features.
    pub record_size: u32,
}

/// Planned game count per split in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSplitPlanReport {
    /// Split name.
    pub split: String,
    /// Planned game count.
    pub planned_games: usize,
}

/// Command/config summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardBuildCommandReport {
    /// Input path.
    pub input: String,
    /// Output directory.
    pub output_dir: String,
    /// Manifest file name.
    pub manifest_name: String,
    /// Target samples per shard.
    pub shard_samples: usize,
    /// Train split fraction.
    pub train_fraction: f32,
    /// Split mode.
    pub split: String,
    /// Worker thread override.
    pub num_threads: Option<usize>,
    /// Queue bound in entries.
    pub queue_bound: usize,
    /// Whether resume was enabled.
    pub resume: bool,
    /// Included games per chunk.
    pub chunk_games: usize,
    /// Optional included-game attempt cap.
    pub limit_max_games: Option<usize>,
    /// Optional sample cap.
    pub limit_max_samples: Option<usize>,
    /// Optional ExIt sidecar path.
    pub exit_sidecar: Option<String>,
    /// Optional delta-Q sidecar path.
    pub delta_q_sidecar: Option<String>,
}

/// Scan summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardScanReport {
    /// Source count.
    pub source_count: usize,
    /// Total game hint.
    pub source_total_games_hint: usize,
    /// Training game hint.
    pub source_train_count_hint: usize,
    /// Validation game hint.
    pub source_val_count_hint: usize,
    /// Whether source counts are exact.
    pub source_counts_exact: bool,
    /// Input compressed bytes when cheap to compute.
    pub input_compressed_bytes: Option<u64>,
}

/// Materialization summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardMaterializationReport {
    /// Loaded non-empty games.
    pub loaded_games: u64,
    /// Skipped included games.
    pub skipped_games: u64,
    /// Included empty games.
    pub empty_games: u64,
    /// Training samples written.
    pub train_samples: u64,
    /// Validation samples written.
    pub validation_samples: u64,
    /// Total samples written.
    pub total_samples: u64,
    /// Whether any configured builder limit was reached.
    pub limit_reached: bool,
    /// Explicit outcome for `--max-samples` collection.
    pub sample_cap_outcome: String,
    /// Samples per loaded non-empty game.
    pub samples_per_loaded_non_empty_game: Option<f64>,
    /// Capped bad-source examples.
    pub bad_source_examples: Vec<String>,
    /// Resume chunks reused.
    pub resume_chunks_reused: u64,
    /// Resume chunks built.
    pub resume_chunks_built: u64,
}

/// Output summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardOutputReport {
    /// Shard file count.
    pub shard_count: usize,
    /// Sum of shard descriptor byte lengths.
    pub output_bytes: u64,
    /// Manifest JSON bytes.
    pub manifest_bytes: u64,
    /// Written shard bytes per sample, excluding manifest JSON.
    pub bytes_per_sample: Option<f64>,
    /// Dense observation-equivalent bytes for the same sample count.
    pub dense_equivalent_observation_bytes: u64,
    /// Dense observation-equivalent bytes per sample.
    pub dense_equivalent_observation_bytes_per_sample: u64,
    /// Ratio of dense observation bytes to compact output bytes.
    pub savings_ratio_vs_dense_observation: Option<f64>,
    /// Per-split output summaries.
    pub splits: Vec<BcShardSplitOutputReport>,
}

/// Per-split output summary in a BC shard build report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSplitOutputReport {
    /// Split name.
    pub split: String,
    /// Split sample count.
    pub sample_count: u64,
    /// Split shard count.
    pub shard_count: usize,
    /// Split shard byte length sum.
    pub byte_len: u64,
    /// Split feature flags.
    pub feature_flags: u32,
    /// Split record size.
    pub record_size: u32,
    /// Split bytes per sample.
    pub bytes_per_sample: Option<f64>,
    /// Smallest shard byte length in this split.
    pub min_shard_bytes: Option<u64>,
    /// Largest shard byte length in this split.
    pub max_shard_bytes: Option<u64>,
}

/// Derived BC shard build rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardBuildRates {
    /// Loaded games per second.
    pub games_per_second: Option<f64>,
    /// Samples per second.
    pub samples_per_second: f64,
    /// Output MiB per second.
    pub output_mib_per_second: f64,
    /// Input MiB per second when input bytes are known.
    pub input_mib_per_second: Option<f64>,
}

#[cfg(test)]
thread_local! {
    static TEST_STOP_AFTER_BUILT_CHUNKS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(all(test, feature = "libtorch"))]
pub(crate) fn set_test_stop_after_built_chunks(chunks: Option<usize>) {
    TEST_STOP_AFTER_BUILT_CHUNKS.with(|slot| slot.set(chunks));
}

#[cfg(test)]
fn should_stop_after_built_chunks(built_chunks: u64) -> bool {
    TEST_STOP_AFTER_BUILT_CHUNKS.with(|slot| {
        slot.get()
            .is_some_and(|limit| built_chunks as usize >= limit)
    })
}

#[cfg(not(test))]
fn should_stop_after_built_chunks(_built_chunks: u64) -> bool {
    false
}

/// Builds BC shard files and writes a manifest.
pub fn build_bc_shards(config: &BuildBcShardsConfig) -> io::Result<BcShardBuildOutput> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;
    if !config.resume {
        reject_stale_non_resume_output(config)?;
    }

    let started = Instant::now();
    let started_at = now_rfc3339();
    let source_manifest = match &config.source_manifest {
        Some(manifest) => manifest.clone(),
        None => scan_data_sources(&config.input)?,
    };
    let feature_flags = feature_flags_from_config(config);
    let record_size = checked_compact_record_size(feature_flags).map_err(invalid_data)?;
    let existing_output_bytes = existing_output_bytes(config)?;
    let available_bytes_before = available_bytes(&config.output_dir);
    let plan = build_plan(&source_manifest, config)?;
    let fingerprint = build_fingerprint(config, &source_manifest, &plan.entries, feature_flags)?;
    let resume = prepare_resume_state(config, &fingerprint, &source_manifest, feature_flags)?;
    let worker_ctx = BuildWorkerContext::from_config(config);
    let mut progress = ProgressSink::new(config, feature_flags, record_size)?;
    let progress_jsonl_path = progress.path.clone();

    let mut fragments = Vec::new();
    let mut reused = 0u64;
    let mut built = 0u64;
    for (chunk_index, chunk) in plan.entries.chunks(config.chunk_games).enumerate() {
        let global_start = chunk.first().map_or(0, |entry| entry.global_sequence);
        let global_end_exclusive = chunk
            .last()
            .map_or(global_start, |entry| entry.global_sequence + 1);
        if let Some(fragment) =
            resume.fragment_for(chunk_index, global_start, global_end_exclusive)?
        {
            progress.fragment_reused(config, &fragment)?;
            fragments.push(fragment);
            reused += 1;
            if fragments_reach_sample_limit(&fragments, config.max_samples) {
                break;
            }
            continue;
        }

        let prior_committed_samples = fragments
            .iter()
            .map(|fragment| fragment.counters.total_samples)
            .sum::<u64>();
        let mut state = WriteState::new_for_chunk(
            config,
            feature_flags,
            chunk_index,
            config.resume,
            prior_committed_samples,
        );
        materialize_chunk(chunk, config, &worker_ctx, &mut state, &mut progress)?;
        let fragment = state.finalize_fragment(
            chunk_index,
            global_start,
            global_end_exclusive,
            &fingerprint,
        )?;
        validate_fragment(&fragment, &fingerprint)?;
        if config.resume {
            atomic_write_json(&resume.fragment_path(chunk_index), &fragment)?;
        }
        progress.fragment_built(config, &fragment)?;
        fragments.push(fragment);
        built += 1;
        if fragments_reach_sample_limit(&fragments, config.max_samples) {
            break;
        }
        if should_stop_after_built_chunks(built) {
            return Err(invalid_data("test stop after committed resume chunk"));
        }
    }

    let split_manifests = merge_fragments_and_rewrite_headers(&config.output_dir, &fragments)?;
    let totals = totals_from_fragments_and_splits(&fragments, &split_manifests);
    let manifest = make_manifest(
        config,
        &source_manifest,
        split_manifests,
        totals,
        started_at.clone(),
    );
    validate_bc_shard_manifest_contract(&manifest).map_err(invalid_data)?;

    let manifest_path = config.output_dir.join(&config.manifest_name);
    atomic_write_json(&manifest_path, &manifest)?;
    let finished_at = now_rfc3339();
    let elapsed = started.elapsed().as_secs_f64();
    let report = make_report(
        config,
        &source_manifest,
        &manifest,
        &manifest_path,
        progress_jsonl_path.as_deref(),
        started_at,
        finished_at,
        elapsed,
        &fragments,
        reused,
        built,
        &plan,
        existing_output_bytes,
        available_bytes_before,
    )?;
    let report_path = match &config.report_name {
        Some(name) => {
            let path = config.output_dir.join(name);
            atomic_write_json(&path, &report)?;
            Some(path)
        }
        None => None,
    };

    Ok(BcShardBuildOutput {
        manifest_path,
        manifest,
        report_path,
        report: Some(report),
    })
}

fn validate_config(config: &BuildBcShardsConfig) -> io::Result<()> {
    if !config.train_fraction.is_finite() {
        return Err(invalid_data("train_fraction must be finite"));
    }
    if !(0.0..=1.0).contains(&config.train_fraction) {
        return Err(invalid_data("train_fraction must be in 0..=1"));
    }
    if config.shard_samples == 0 {
        return Err(invalid_data("shard_samples must be > 0"));
    }
    if config.queue_bound == 0 {
        return Err(invalid_data("queue_bound must be > 0"));
    }
    if config.chunk_games == 0 {
        return Err(invalid_data("chunk_games must be > 0"));
    }
    if matches!(config.num_threads, Some(0)) {
        return Err(invalid_data("num_threads must be > 0"));
    }
    Ok(())
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

#[derive(Debug, Clone)]
struct BuildWorkerContext {
    replay_target_profile: hydra_replay_loader::ReplayTargetProfile,
    exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    exit_provenance: SidecarProvenance,
    delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    delta_q_provenance: SidecarProvenance,
}

impl BuildWorkerContext {
    fn from_config(config: &BuildBcShardsConfig) -> Self {
        Self {
            replay_target_profile: replay_target_profile_for_bc_shards(config),
            exit_sidecar: config.exit_sidecar.clone(),
            exit_provenance: config.exit_provenance,
            delta_q_sidecar: config.delta_q_sidecar.clone(),
            delta_q_provenance: config.delta_q_provenance,
        }
    }

    fn policy(&self) -> ReplayLoadPolicy<'_> {
        ReplayLoadPolicy::new(
            self.replay_target_profile,
            ReplayObservationProfile::BcMinimal,
            self.exit_provenance,
            self.delta_q_provenance,
            self.exit_sidecar.as_deref(),
            self.delta_q_sidecar.as_deref(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildPlanEntry {
    global_sequence: usize,
    source_index: usize,
    identity: String,
    split: BcShardSplit,
    source: BuildPlanSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum BuildPlanSource {
    LooseFile {
        path: PathBuf,
    },
    ArchiveEntry {
        archive_path: PathBuf,
        entry_path: PathBuf,
    },
}

#[derive(Debug)]
struct PathJob {
    sequence: usize,
    path: PathBuf,
    identity: String,
    split: BcShardSplit,
}

#[derive(Debug)]
struct ArchiveJob {
    sequence: usize,
    identity: String,
    split: BcShardSplit,
    data: Vec<u8>,
}

struct MaterializedGame {
    sequence: usize,
    identity: String,
    split: BcShardSplit,
    result: io::Result<MjaiGame>,
}

struct EncodedMaterializedGame {
    sequence: usize,
    identity: String,
    split: BcShardSplit,
    result: io::Result<EncodedGameData>,
}

struct EncodedGameData {
    sample_count: usize,
    records: Vec<u8>,
}

fn encode_materialized_game(
    game: MaterializedGame,
    feature_flags: u32,
    record_size: u32,
) -> EncodedMaterializedGame {
    let MaterializedGame {
        sequence,
        identity,
        split,
        result,
    } = game;
    let result = result.and_then(|game_data| {
        let sample_count = game_data.samples.len();
        encode_sample_records(&game_data.samples, feature_flags, record_size).map(|records| {
            EncodedGameData {
                sample_count,
                records,
            }
        })
    });
    EncodedMaterializedGame {
        sequence,
        identity,
        split,
        result,
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct BuildCounters {
    loaded_games: u64,
    skipped_games: u64,
    empty_games: u64,
    train_samples: u64,
    validation_samples: u64,
    total_samples: u64,
    shard_count: usize,
    completed_sources: u64,
    materialized_games: u64,
}

struct WriteState<'a> {
    config: &'a BuildBcShardsConfig,
    train_state: Option<ChunkSplitBuildState>,
    val_state: Option<ChunkSplitBuildState>,
    counters: BuildCounters,
    prior_committed_samples: u64,
    bad_source_examples: Vec<String>,
}

impl<'a> WriteState<'a> {
    fn new_for_chunk(
        config: &'a BuildBcShardsConfig,
        feature_flags: u32,
        chunk_index: usize,
        chunk_names: bool,
        prior_committed_samples: u64,
    ) -> Self {
        Self {
            config,
            train_state: config.split_mode.includes(BcShardSplit::Train).then(|| {
                ChunkSplitBuildState::new(
                    BcShardSplit::Train,
                    feature_flags,
                    chunk_index,
                    chunk_names,
                )
            }),
            val_state: config
                .split_mode
                .includes(BcShardSplit::Validation)
                .then(|| {
                    ChunkSplitBuildState::new(
                        BcShardSplit::Validation,
                        feature_flags,
                        chunk_index,
                        chunk_names,
                    )
                }),
            counters: BuildCounters::default(),
            prior_committed_samples,
            bad_source_examples: Vec::new(),
        }
    }

    fn sample_limit_reached(&self) -> bool {
        self.config.max_samples.is_some_and(|max| {
            self.prior_committed_samples
                .saturating_add(self.counters.total_samples)
                >= max as u64
        })
    }

    fn handle_encoded_materialized(&mut self, game: EncodedMaterializedGame) -> io::Result<bool> {
        self.counters.materialized_games += 1;
        match game.result {
            Ok(game_data) => {
                if game_data.sample_count == 0 {
                    self.counters.empty_games += 1;
                    return Ok(self.sample_limit_reached());
                }
                let sample_count = game_data.sample_count as u64;
                match game.split {
                    BcShardSplit::Train => {
                        if let Some(state) = self.train_state.as_mut() {
                            state.push_encoded_samples(
                                &self.config.output_dir,
                                self.config.shard_samples,
                                &game_data.records,
                                game_data.sample_count,
                            )?;
                            self.counters.train_samples += sample_count;
                        }
                    }
                    BcShardSplit::Validation => {
                        if let Some(state) = self.val_state.as_mut() {
                            state.push_encoded_samples(
                                &self.config.output_dir,
                                self.config.shard_samples,
                                &game_data.records,
                                game_data.sample_count,
                            )?;
                            self.counters.validation_samples += sample_count;
                        }
                    }
                }
                self.counters.loaded_games += 1;
                self.counters.total_samples += sample_count;
            }
            Err(err) => {
                self.counters.skipped_games += 1;
                if self.bad_source_examples.len() < self.config.max_error_examples {
                    self.bad_source_examples.push(format!(
                        "{}: {}",
                        compact_identity(&game.identity),
                        compact_error_message(&err)
                    ));
                }
            }
        }
        Ok(self.sample_limit_reached())
    }

    fn finalize_fragment(
        mut self,
        chunk_index: usize,
        global_start: usize,
        global_end_exclusive: usize,
        fingerprint: &BuildFingerprint,
    ) -> io::Result<BcShardBuildFragment> {
        let train = self
            .train_state
            .take()
            .map(ChunkSplitBuildState::finalize)
            .transpose()?;
        let validation = self
            .val_state
            .take()
            .map(ChunkSplitBuildState::finalize)
            .transpose()?;
        self.counters.shard_count = train.as_ref().map_or(0, |split| split.shard_count)
            + validation.as_ref().map_or(0, |split| split.shard_count);
        Ok(BcShardBuildFragment {
            schema_version: RESUME_SCHEMA_VERSION,
            config_fingerprint: fingerprint.config_fingerprint.clone(),
            plan_fingerprint: fingerprint.plan_fingerprint.clone(),
            chunk_index,
            global_start,
            global_end_exclusive,
            counters: self.counters,
            train,
            validation,
            bad_source_examples: self.bad_source_examples,
        })
    }
}

struct ChunkSplitBuildState {
    split: BcShardSplit,
    next_shard_index: usize,
    total_samples: u64,
    feature_flags: u32,
    record_size: u32,
    shards: Vec<BcShardDescriptor>,
    active: Option<ActiveShardWriter>,
    active_samples: u64,
    chunk_index: usize,
    chunk_names: bool,
}

impl ChunkSplitBuildState {
    fn new(split: BcShardSplit, feature_flags: u32, chunk_index: usize, chunk_names: bool) -> Self {
        Self {
            split,
            next_shard_index: 0,
            total_samples: 0,
            feature_flags,
            record_size: record_size_for_flags(feature_flags),
            shards: Vec::new(),
            active: None,
            active_samples: 0,
            chunk_index,
            chunk_names,
        }
    }

    fn push_encoded_samples(
        &mut self,
        output_dir: &Path,
        shard_samples: usize,
        records: &[u8],
        sample_count: usize,
    ) -> io::Result<()> {
        if sample_count == 0 {
            return Ok(());
        }
        let game_samples = sample_count as u64;
        if self.active.is_some()
            && self.active_samples > 0
            && self.active_samples + game_samples > shard_samples.max(1) as u64
        {
            self.finish_active()?;
        }
        if self.active.is_none() {
            let file_name = if self.chunk_names {
                format!(
                    "chunk-{chunk:06}-{prefix}-{shard:05}.hydra-bc",
                    chunk = self.chunk_index,
                    prefix = self.split.shard_prefix(),
                    shard = self.next_shard_index
                )
            } else {
                format!(
                    "{}-{shard:05}.hydra-bc",
                    self.split.shard_prefix(),
                    shard = self.next_shard_index
                )
            };
            let shard = ActiveShardWriter::new_named(
                output_dir,
                self.split,
                self.next_shard_index,
                self.total_samples,
                self.feature_flags,
                file_name,
            )?;
            self.next_shard_index += 1;
            self.active = Some(shard);
            self.active_samples = 0;
        }
        let active = self.active.as_mut().expect("active shard should exist");
        active.write_encoded_records(records, sample_count)?;
        self.active_samples += game_samples;
        self.total_samples += game_samples;
        Ok(())
    }

    fn finish_active(&mut self) -> io::Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let descriptor = active.finish()?;
        self.shards.push(descriptor);
        self.active_samples = 0;
        Ok(())
    }

    fn finalize(mut self) -> io::Result<BcShardSplitManifest> {
        self.finish_active()?;
        Ok(BcShardSplitManifest {
            split: self.split,
            shard_count: self.shards.len(),
            sample_count: self.total_samples,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
            shards: self.shards,
        })
    }
}

#[derive(Debug)]
struct BuildPlan {
    entries: Vec<BuildPlanEntry>,
    max_games_reached: bool,
}
fn build_plan(
    source_manifest: &DataManifest,
    config: &BuildBcShardsConfig,
) -> io::Result<BuildPlan> {
    let mut out = Vec::new();
    let mut max_games_reached = false;
    for (source_index, source) in source_manifest.sources.iter().enumerate() {
        match source {
            DataSource::LooseFile(path) => {
                let identity = identity_for_loose_file(path)?;
                if let Some(split) = split_for_identity(&identity, config) {
                    if config.max_games.is_some_and(|max| out.len() >= max) {
                        max_games_reached = true;
                        break;
                    }
                    out.push(BuildPlanEntry {
                        global_sequence: out.len(),
                        source_index,
                        identity,
                        split,
                        source: BuildPlanSource::LooseFile { path: path.clone() },
                    });
                }
            }
            DataSource::Archive(path) => {
                if enumerate_archive_entries(path, config, source_index, &mut out)? {
                    max_games_reached = true;
                    break;
                }
            }
            DataSource::ParsedSampleCache { path, .. } => {
                return Err(invalid_data(format!(
                    "parsed-sample cache input is not supported by build_bc_shards yet: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(BuildPlan {
        entries: out,
        max_games_reached,
    })
}

fn enumerate_archive_entries(
    path: &Path,
    config: &BuildBcShardsConfig,
    source_index: usize,
    out: &mut Vec<BuildPlanEntry>,
) -> io::Result<bool> {
    let file = File::open(path)?;
    let reader = archive_reader(path, file)?;
    let mut archive = tar::Archive::new(reader);
    for entry_result in archive.entries()? {
        let entry = entry_result?;
        let entry_path = entry.path()?.into_owned();
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }
        let identity = identity_for_archive_entry(path, &entry_path)?;
        if config.max_games.is_some_and(|max| out.len() >= max) {
            return Ok(true);
        }
        if let Some(split) = split_for_identity(&identity, config) {
            out.push(BuildPlanEntry {
                global_sequence: out.len(),
                source_index,
                identity,
                split,
                source: BuildPlanSource::ArchiveEntry {
                    archive_path: path.to_path_buf(),
                    entry_path,
                },
            });
        }
    }
    Ok(false)
}

fn archive_reader(path: &Path, file: File) -> io::Result<Box<dyn Read + Send>> {
    if is_tar_zst_file(path) {
        Ok(Box::new(zstd::Decoder::new(file).map_err(|err| {
            io::Error::other(format!(
                "failed to open zstd archive {}: {err}",
                path.display()
            ))
        })?))
    } else {
        Ok(Box::new(file))
    }
}

enum PlanGroup<'a> {
    Loose(Vec<&'a BuildPlanEntry>),
    Archive {
        archive_path: PathBuf,
        entries: Vec<&'a BuildPlanEntry>,
    },
}

fn group_plan_entries_preserving_order(entries: &[BuildPlanEntry]) -> Vec<PlanGroup<'_>> {
    let mut groups = Vec::new();
    let mut i = 0usize;
    while i < entries.len() {
        match &entries[i].source {
            BuildPlanSource::LooseFile { .. } => {
                let mut group = Vec::new();
                while i < entries.len()
                    && matches!(entries[i].source, BuildPlanSource::LooseFile { .. })
                {
                    group.push(&entries[i]);
                    i += 1;
                }
                groups.push(PlanGroup::Loose(group));
            }
            BuildPlanSource::ArchiveEntry { archive_path, .. } => {
                let archive_path = archive_path.clone();
                let mut group = Vec::new();
                while i < entries.len() {
                    match &entries[i].source {
                        BuildPlanSource::ArchiveEntry {
                            archive_path: path, ..
                        } if *path == archive_path => {
                            group.push(&entries[i]);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                groups.push(PlanGroup::Archive {
                    archive_path,
                    entries: group,
                });
            }
        }
    }
    groups
}

fn materialize_chunk(
    entries: &[BuildPlanEntry],
    config: &BuildBcShardsConfig,
    worker_ctx: &BuildWorkerContext,
    write_state: &mut WriteState<'_>,
    progress: &mut ProgressSink,
) -> io::Result<()> {
    for group in group_plan_entries_preserving_order(entries) {
        match group {
            PlanGroup::Loose(entries) => materialize_loose_group_ordered(
                &entries,
                config,
                worker_ctx,
                write_state,
                progress,
            )?,
            PlanGroup::Archive {
                archive_path,
                entries,
            } => materialize_archive_group_ordered(
                &archive_path,
                &entries,
                config,
                worker_ctx,
                write_state,
                progress,
            )?,
        }
        if write_state.sample_limit_reached() {
            break;
        }
    }
    Ok(())
}

fn materialize_loose_group_ordered(
    entries: &[&BuildPlanEntry],
    config: &BuildBcShardsConfig,
    worker_ctx: &BuildWorkerContext,
    write_state: &mut WriteState<'_>,
    progress: &mut ProgressSink,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "loose materialization")?;
    let (job_tx, job_rx) = mpsc::sync_channel::<PathJob>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<EncodedMaterializedGame>>(config.queue_bound);
    let jobs: Vec<PathJob> = entries
        .iter()
        .enumerate()
        .map(|(sequence, entry)| match &entry.source {
            BuildPlanSource::LooseFile { path } => PathJob {
                sequence,
                path: path.clone(),
                identity: entry.identity.clone(),
                split: entry.split,
            },
            BuildPlanSource::ArchiveEntry { .. } => {
                unreachable!("loose group contains archive entry")
            }
        })
        .collect();
    let producer = thread::Builder::new()
        .name("bc-shard-loose-producer".into())
        .spawn(move || -> io::Result<()> {
            for job in jobs {
                if job_tx.send(job).is_err() {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn loose producer: {err}")))?;
    let worker = worker_ctx.clone();
    let feature_flags = feature_flags_from_config(config);
    let record_size = record_size_for_flags(feature_flags);
    let workers = thread::Builder::new()
        .name("bc-shard-loose-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let policy = worker.policy();
                    let result = load_game_from_path_with_policy(&job.path, Some(&policy));
                    let encoded = encode_materialized_game(
                        MaterializedGame {
                            sequence: job.sequence,
                            identity: job.identity,
                            split: job.split,
                            result,
                        },
                        feature_flags,
                        record_size,
                    );
                    let _ = result_tx.send(Ok(encoded));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn loose workers: {err}")))?;
    collect_materialized_in_order(result_rx, entries.len(), config, write_state, progress)?;
    producer
        .join()
        .map_err(|_| io::Error::other("loose producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("loose worker thread panicked"))?;
    Ok(())
}

fn materialize_archive_group_ordered(
    archive_path: &Path,
    entries: &[&BuildPlanEntry],
    config: &BuildBcShardsConfig,
    worker_ctx: &BuildWorkerContext,
    write_state: &mut WriteState<'_>,
    progress: &mut ProgressSink,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "archive materialization")?;
    let wanted: HashMap<PathBuf, (usize, String, BcShardSplit)> = entries
        .iter()
        .enumerate()
        .map(|(sequence, entry)| match &entry.source {
            BuildPlanSource::ArchiveEntry { entry_path, .. } => (
                entry_path.clone(),
                (sequence, entry.identity.clone(), entry.split),
            ),
            BuildPlanSource::LooseFile { .. } => unreachable!("archive group contains loose entry"),
        })
        .collect();
    let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveJob>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<EncodedMaterializedGame>>(config.queue_bound);
    let producer_path = archive_path.to_path_buf();
    let producer = thread::Builder::new()
        .name("bc-shard-archive-producer".into())
        .spawn(move || -> io::Result<()> {
            let file = File::open(&producer_path)?;
            let reader = archive_reader(&producer_path, file)?;
            let mut archive = tar::Archive::new(reader);
            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                let Some((sequence, identity, split)) = wanted.get(&entry_path) else {
                    continue;
                };
                let mut data = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut data)?;
                if job_tx
                    .send(ArchiveJob {
                        sequence: *sequence,
                        identity: identity.clone(),
                        split: *split,
                        data,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive producer: {err}")))?;
    let worker = worker_ctx.clone();
    let feature_flags = feature_flags_from_config(config);
    let record_size = record_size_for_flags(feature_flags);
    let workers = thread::Builder::new()
        .name("bc-shard-archive-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let policy = worker.policy();
                    let result = load_game_from_stream_with_policy(
                        &job.identity,
                        BufReader::new(std::io::Cursor::new(job.data)),
                        Some(&policy),
                    );
                    let encoded = encode_materialized_game(
                        MaterializedGame {
                            sequence: job.sequence,
                            identity: job.identity,
                            split: job.split,
                            result,
                        },
                        feature_flags,
                        record_size,
                    );
                    let _ = result_tx.send(Ok(encoded));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive workers: {err}")))?;
    collect_materialized_in_order(result_rx, entries.len(), config, write_state, progress)?;
    producer
        .join()
        .map_err(|_| io::Error::other("archive producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("archive worker thread panicked"))?;
    Ok(())
}

fn collect_materialized_in_order(
    results: mpsc::Receiver<io::Result<EncodedMaterializedGame>>,
    expected_count: usize,
    config: &BuildBcShardsConfig,
    state: &mut WriteState<'_>,
    progress: &mut ProgressSink,
) -> io::Result<()> {
    let mut next = 0usize;
    let mut pending = BTreeMap::new();
    let mut stopped_on_sample_limit = false;
    for item in results {
        let game = item?;
        pending.insert(game.sequence, game);
        while let Some(game) = pending.remove(&next) {
            let limit_reached = state.handle_encoded_materialized(game)?;
            progress.game_committed(config, &state.counters)?;
            next += 1;
            if limit_reached {
                stopped_on_sample_limit = true;
                break;
            }
        }
        if stopped_on_sample_limit {
            break;
        }
    }
    if !stopped_on_sample_limit && next != expected_count {
        return Err(invalid_data(format!(
            "materialized {next} games, expected {expected_count}"
        )));
    }
    Ok(())
}

fn fragments_reach_sample_limit(
    fragments: &[BcShardBuildFragment],
    max_samples: Option<usize>,
) -> bool {
    max_samples.is_some_and(|max| {
        fragments
            .iter()
            .map(|fragment| fragment.counters.total_samples)
            .sum::<u64>()
            >= max as u64
    })
}

fn sample_cap_reached(config: &BuildBcShardsConfig, total_samples: u64) -> bool {
    config
        .max_samples
        .is_some_and(|max_samples| total_samples >= max_samples as u64)
}

fn sample_cap_outcome(config: &BuildBcShardsConfig, total_samples: u64) -> &'static str {
    let Some(max_samples) = config.max_samples else {
        return SAMPLE_CAP_OUTCOME_NONE;
    };
    let max_samples = max_samples as u64;
    if total_samples < max_samples {
        SAMPLE_CAP_OUTCOME_NOT_REACHED
    } else if total_samples == max_samples {
        SAMPLE_CAP_OUTCOME_REACHED_EXACT
    } else {
        SAMPLE_CAP_OUTCOME_REACHED_AFTER_CURRENT_GAME
    }
}

fn make_optional_pool(num_threads: Option<usize>, label: &str) -> io::Result<rayon::ThreadPool> {
    let mut builder = ThreadPoolBuilder::new();
    if let Some(n) = num_threads {
        builder = builder.num_threads(n);
    }
    builder
        .build()
        .map_err(|err| io::Error::other(format!("failed to build {label} thread pool: {err}")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildFingerprint {
    config_fingerprint: String,
    source_manifest_fingerprint: String,
    plan_fingerprint: String,
}

#[derive(Serialize)]
struct ConfigFingerprintInput<'a> {
    train_fraction_bits: u32,
    shard_samples: usize,
    split_mode: &'a str,
    feature_flags: u32,
    exit_sidecar: Option<BcShardSidecarManifest>,
    delta_q_sidecar: Option<BcShardSidecarManifest>,
    chunk_games: usize,
    max_games: Option<usize>,
    max_samples: Option<usize>,
    source_manifest_fingerprint: &'a str,
    plan_fingerprint: &'a str,
}

fn build_fingerprint(
    config: &BuildBcShardsConfig,
    source_manifest: &DataManifest,
    plan: &[BuildPlanEntry],
    feature_flags: u32,
) -> io::Result<BuildFingerprint> {
    let source_manifest_fingerprint = stable_json_hash(source_manifest)?;
    let plan_fingerprint = stable_json_hash(plan)?;
    let input = ConfigFingerprintInput {
        train_fraction_bits: config.train_fraction.to_bits(),
        shard_samples: config.shard_samples,
        split_mode: split_mode_name(config.split_mode),
        feature_flags,
        exit_sidecar: sidecar_manifest(config.exit_sidecar_path.as_deref(), config.exit_provenance),
        delta_q_sidecar: sidecar_manifest(
            config.delta_q_sidecar_path.as_deref(),
            config.delta_q_provenance,
        ),
        chunk_games: config.chunk_games,
        max_games: config.max_games,
        max_samples: config.max_samples,
        source_manifest_fingerprint: &source_manifest_fingerprint,
        plan_fingerprint: &plan_fingerprint,
    };
    Ok(BuildFingerprint {
        config_fingerprint: stable_json_hash(&input)?,
        source_manifest_fingerprint,
        plan_fingerprint,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BcShardBuildStateHeader {
    schema_version: u32,
    config_fingerprint: String,
    source_manifest_fingerprint: String,
    plan_fingerprint: String,
    input: String,
    output_dir: String,
    train_fraction_bits: u32,
    shard_samples: usize,
    split_mode: String,
    feature_flags: u32,
    exit_sidecar: Option<BcShardSidecarManifest>,
    delta_q_sidecar: Option<BcShardSidecarManifest>,
    chunk_games: usize,
    max_games: Option<usize>,
    max_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BcShardBuildFragment {
    schema_version: u32,
    config_fingerprint: String,
    plan_fingerprint: String,
    chunk_index: usize,
    global_start: usize,
    global_end_exclusive: usize,
    counters: BuildCounters,
    train: Option<BcShardSplitManifest>,
    validation: Option<BcShardSplitManifest>,
    bad_source_examples: Vec<String>,
}

struct ResumeState {
    enabled: bool,
    state_dir: PathBuf,
    fragment_dir: PathBuf,
    fingerprint: BuildFingerprint,
}

impl ResumeState {
    fn fragment_path(&self, chunk_index: usize) -> PathBuf {
        self.fragment_dir
            .join(format!("chunk-{chunk_index:06}.json"))
    }

    fn fragment_for(
        &self,
        chunk_index: usize,
        global_start: usize,
        global_end_exclusive: usize,
    ) -> io::Result<Option<BcShardBuildFragment>> {
        if !self.enabled {
            return Ok(None);
        }
        let path = self.fragment_path(chunk_index);
        if !path.exists() {
            return Ok(None);
        }
        let fragment: BcShardBuildFragment =
            serde_json::from_reader(File::open(&path)?).map_err(|err| {
                invalid_data(format!(
                    "failed to parse resume fragment {}: {err}",
                    path.display()
                ))
            })?;
        validate_fragment(&fragment, &self.fingerprint)?;
        if fragment.chunk_index != chunk_index
            || fragment.global_start != global_start
            || fragment.global_end_exclusive != global_end_exclusive
        {
            return Err(invalid_data(format!(
                "resume fragment {} range mismatch; use a new output dir or delete resume state",
                path.display()
            )));
        }
        if let Some(split) = &fragment.train {
            validate_resume_split_files(split, &self.state_dir, &path)?;
        }
        if let Some(split) = &fragment.validation {
            validate_resume_split_files(split, &self.state_dir, &path)?;
        }
        Ok(Some(fragment))
    }
}

fn prepare_resume_state(
    config: &BuildBcShardsConfig,
    fingerprint: &BuildFingerprint,
    source_manifest: &DataManifest,
    feature_flags: u32,
) -> io::Result<ResumeState> {
    let state_dir = resume_state_dir(config);
    let fragment_dir = state_dir.join("fragments");
    if config.resume {
        fs::create_dir_all(&fragment_dir)?;
        fs::create_dir_all(state_dir.join("tmp"))?;
        let header = resume_header(config, fingerprint, source_manifest, feature_flags);
        let header_path = state_dir.join("build_config.json");
        if header_path.exists() {
            let existing: BcShardBuildStateHeader =
                serde_json::from_reader(File::open(&header_path)?).map_err(|err| {
                    invalid_data(format!(
                        "failed to parse resume state header {}: {err}",
                        header_path.display()
                    ))
                })?;
            if existing.config_fingerprint != header.config_fingerprint
                || existing.source_manifest_fingerprint != header.source_manifest_fingerprint
                || existing.plan_fingerprint != header.plan_fingerprint
            {
                return Err(invalid_data(
                    "resume state mismatch: config/source/plan fingerprint differs; use a new output dir or delete resume state",
                ));
            }
        } else {
            atomic_write_json(&header_path, &header)?;
        }
        atomic_write_json(&state_dir.join("plan.json"), &fingerprint)?;
    }
    Ok(ResumeState {
        enabled: config.resume,
        state_dir,
        fragment_dir,
        fingerprint: fingerprint.clone(),
    })
}

fn resume_header(
    config: &BuildBcShardsConfig,
    fingerprint: &BuildFingerprint,
    _source_manifest: &DataManifest,
    feature_flags: u32,
) -> BcShardBuildStateHeader {
    BcShardBuildStateHeader {
        schema_version: RESUME_SCHEMA_VERSION,
        config_fingerprint: fingerprint.config_fingerprint.clone(),
        source_manifest_fingerprint: fingerprint.source_manifest_fingerprint.clone(),
        plan_fingerprint: fingerprint.plan_fingerprint.clone(),
        input: config.input.display().to_string(),
        output_dir: config.output_dir.display().to_string(),
        train_fraction_bits: config.train_fraction.to_bits(),
        shard_samples: config.shard_samples,
        split_mode: split_mode_name(config.split_mode).to_string(),
        feature_flags,
        exit_sidecar: sidecar_manifest(config.exit_sidecar_path.as_deref(), config.exit_provenance),
        delta_q_sidecar: sidecar_manifest(
            config.delta_q_sidecar_path.as_deref(),
            config.delta_q_provenance,
        ),
        chunk_games: config.chunk_games,
        max_games: config.max_games,
        max_samples: config.max_samples,
    }
}

fn validate_fragment(
    fragment: &BcShardBuildFragment,
    fingerprint: &BuildFingerprint,
) -> io::Result<()> {
    if fragment.schema_version != RESUME_SCHEMA_VERSION {
        return Err(invalid_data("resume fragment schema version mismatch"));
    }
    if fragment.config_fingerprint != fingerprint.config_fingerprint
        || fragment.plan_fingerprint != fingerprint.plan_fingerprint
    {
        return Err(invalid_data("resume fragment fingerprint mismatch"));
    }
    if let Some(split) = &fragment.train {
        validate_bc_shard_split_manifest_contract(split).map_err(invalid_data)?;
    }
    if let Some(split) = &fragment.validation {
        validate_bc_shard_split_manifest_contract(split).map_err(invalid_data)?;
    }
    Ok(())
}

fn validate_resume_split_files(
    split: &BcShardSplitManifest,
    state_dir: &Path,
    fragment_path: &Path,
) -> io::Result<()> {
    let output_dir = state_dir.parent().unwrap_or(state_dir);
    for descriptor in &split.shards {
        let shard_path = output_dir.join(&descriptor.file_name);
        let len = fs::metadata(&shard_path)
            .map_err(|err| {
                invalid_data(format!(
                    "resume fragment {} references unreadable shard {}: {err}",
                    fragment_path.display(),
                    shard_path.display()
                ))
            })?
            .len();
        if len != descriptor.byte_len {
            return Err(invalid_data(format!(
                "resume fragment {} shard {} byte length mismatch",
                fragment_path.display(),
                shard_path.display()
            )));
        }
    }
    Ok(())
}

fn merge_fragments_and_rewrite_headers(
    output_dir: &Path,
    fragments: &[BcShardBuildFragment],
) -> io::Result<Vec<BcShardSplitManifest>> {
    let mut splits = Vec::new();
    if let Some(split) = merge_split(output_dir, fragments, BcShardSplit::Train)? {
        splits.push(split);
    }
    if let Some(split) = merge_split(output_dir, fragments, BcShardSplit::Validation)? {
        splits.push(split);
    }
    Ok(splits)
}

fn merge_split(
    output_dir: &Path,
    fragments: &[BcShardBuildFragment],
    split: BcShardSplit,
) -> io::Result<Option<BcShardSplitManifest>> {
    let mut shards = Vec::new();
    let mut sample_count = 0u64;
    let mut feature_flags = None;
    let mut record_size = None;
    for fragment in fragments {
        let manifest = match split {
            BcShardSplit::Train => fragment.train.as_ref(),
            BcShardSplit::Validation => fragment.validation.as_ref(),
        };
        let Some(manifest) = manifest else {
            continue;
        };
        feature_flags.get_or_insert(manifest.feature_flags);
        record_size.get_or_insert(manifest.record_size);
        for shard in &manifest.shards {
            let mut descriptor = shard.clone();
            descriptor.shard_index = shards.len();
            descriptor.first_sample_index = sample_count;
            rewrite_shard_header_for_descriptor(
                &output_dir.join(&descriptor.file_name),
                &descriptor,
            )?;
            sample_count += descriptor.sample_count;
            shards.push(descriptor);
        }
    }
    if shards.is_empty() {
        return Ok(None);
    }
    let split_manifest = BcShardSplitManifest {
        split,
        shard_count: shards.len(),
        sample_count,
        feature_flags: feature_flags.unwrap_or(FLAG_SAFETY_RESIDUAL),
        record_size: record_size.unwrap_or_else(|| record_size_for_flags(FLAG_SAFETY_RESIDUAL)),
        shards,
    };
    validate_bc_shard_split_manifest_contract(&split_manifest).map_err(invalid_data)?;
    Ok(Some(split_manifest))
}

fn totals_from_fragments_and_splits(
    fragments: &[BcShardBuildFragment],
    splits: &[BcShardSplitManifest],
) -> BcShardBuildTotals {
    let mut totals = BcShardBuildTotals::default();
    for fragment in fragments {
        totals.skipped_games += fragment.counters.skipped_games;
        totals.empty_games += fragment.counters.empty_games;
    }
    for split in splits {
        totals.sample_count += split.sample_count;
        totals.shard_count += split.shard_count;
    }
    totals
}

fn make_manifest(
    config: &BuildBcShardsConfig,
    source_manifest: &DataManifest,
    split_manifests: Vec<BcShardSplitManifest>,
    totals: BcShardBuildTotals,
    created_at: String,
) -> BcShardManifest {
    BcShardManifest {
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
        split_mode: config.split_mode.as_str().to_string(),
        augment_runtime: true,
        input: config.input.display().to_string(),
        output_dir: config.output_dir.display().to_string(),
        created_at,
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
        storage_layout: STORAGE_LAYOUT_COMPACT.to_string(),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "report assembles immutable build artifacts"
)]
fn make_report(
    config: &BuildBcShardsConfig,
    source_manifest: &DataManifest,
    manifest: &BcShardManifest,
    manifest_path: &Path,
    progress_path: Option<&Path>,
    started_at: String,
    finished_at: String,
    elapsed_seconds: f64,
    fragments: &[BcShardBuildFragment],
    reused: u64,
    built: u64,
    plan: &BuildPlan,
    existing_output_bytes: u64,
    available_bytes_before: Option<u64>,
) -> io::Result<BcShardBuildReport> {
    let mut counters = BuildCounters::default();
    let mut bad_source_examples = Vec::new();
    for fragment in fragments {
        counters.loaded_games += fragment.counters.loaded_games;
        counters.skipped_games += fragment.counters.skipped_games;
        counters.empty_games += fragment.counters.empty_games;
        counters.train_samples += fragment.counters.train_samples;
        counters.validation_samples += fragment.counters.validation_samples;
        counters.total_samples += fragment.counters.total_samples;
        counters.shard_count += fragment.counters.shard_count;
        counters.materialized_games += fragment.counters.materialized_games;
        for example in &fragment.bad_source_examples {
            if bad_source_examples.len() < config.max_error_examples {
                bad_source_examples.push(example.clone());
            }
        }
    }
    let output_bytes: u64 = manifest
        .splits
        .iter()
        .flat_map(|split| &split.shards)
        .map(|shard| shard.byte_len)
        .sum();
    let manifest_bytes = fs::metadata(manifest_path).map(|m| m.len()).unwrap_or(0);
    let bytes_per_sample = (manifest.totals.sample_count > 0)
        .then(|| output_bytes as f64 / manifest.totals.sample_count as f64);
    let dense_equivalent_observation_bytes_per_sample = hydra_bc_shards::DENSE_OBS_F32_BYTES as u64;
    let dense_equivalent_observation_bytes = manifest
        .totals
        .sample_count
        .saturating_mul(dense_equivalent_observation_bytes_per_sample);
    let savings_ratio_vs_dense_observation =
        (output_bytes > 0).then(|| dense_equivalent_observation_bytes as f64 / output_bytes as f64);
    let input_bytes = input_compressed_bytes(source_manifest);
    let seconds = elapsed_seconds.max(0.000_001);
    let feature_flags = feature_flags_from_config(config);
    let record_size = checked_compact_record_size(feature_flags).map_err(invalid_data)?;
    let available_bytes_after = available_bytes(&config.output_dir);
    Ok(BcShardBuildReport {
        schema_version: REPORT_SCHEMA_VERSION,
        started_at,
        finished_at,
        elapsed_seconds,
        abi: BcShardAbiReport {
            storage_layout: STORAGE_LAYOUT_COMPACT.to_string(),
            manifest_version: BC_SHARD_MANIFEST_VERSION,
            shard_version: BC_SHARD_VERSION,
            layout_version: BC_SHARD_LAYOUT_VERSION,
            header_size: BC_SHARD_HEADER_SIZE,
            base_record_size: BC_BASE_RECORD_SIZE,
            max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
            dense_obs_f32_bytes_per_sample: hydra_bc_shards::DENSE_OBS_F32_BYTES as u64,
            feature_flags,
            record_size,
        },
        disk: BcShardDiskReport {
            output_dir: config.output_dir.display().to_string(),
            existing_output_bytes,
            available_bytes_before,
            available_bytes_after,
            projected_output_bytes: None,
            projected_sample_count: None,
            projection_source: "unavailable".to_string(),
        },
        plan_splits: plan_split_reports(&plan.entries),
        command: BcShardBuildCommandReport {
            input: config.input.display().to_string(),
            output_dir: config.output_dir.display().to_string(),
            manifest_name: config.manifest_name.clone(),
            shard_samples: config.shard_samples,
            train_fraction: config.train_fraction,
            split: split_mode_name(config.split_mode).to_string(),
            num_threads: config.num_threads,
            queue_bound: config.queue_bound,
            resume: config.resume,
            chunk_games: config.chunk_games,
            limit_max_games: config.max_games,
            limit_max_samples: config.max_samples,
            exit_sidecar: config
                .exit_sidecar_path
                .as_ref()
                .map(|p| p.display().to_string()),
            delta_q_sidecar: config
                .delta_q_sidecar_path
                .as_ref()
                .map(|p| p.display().to_string()),
        },
        scan: BcShardScanReport {
            source_count: source_manifest.sources.len(),
            source_total_games_hint: source_manifest.total_games,
            source_train_count_hint: source_manifest.train_count,
            source_val_count_hint: source_manifest.val_count,
            source_counts_exact: source_manifest.counts_exact,
            input_compressed_bytes: input_bytes,
        },
        build: BcShardMaterializationReport {
            loaded_games: counters.loaded_games,
            skipped_games: counters.skipped_games,
            empty_games: counters.empty_games,
            train_samples: counters.train_samples,
            validation_samples: counters.validation_samples,
            total_samples: counters.total_samples,
            limit_reached: plan.max_games_reached
                || sample_cap_reached(config, counters.total_samples),
            sample_cap_outcome: sample_cap_outcome(config, counters.total_samples).to_string(),
            samples_per_loaded_non_empty_game: (counters.loaded_games > 0)
                .then(|| counters.total_samples as f64 / counters.loaded_games as f64),
            bad_source_examples,
            resume_chunks_reused: reused,
            resume_chunks_built: built,
        },
        output: BcShardOutputReport {
            shard_count: manifest.totals.shard_count,
            output_bytes,
            manifest_bytes,
            bytes_per_sample,
            dense_equivalent_observation_bytes,
            dense_equivalent_observation_bytes_per_sample,
            savings_ratio_vs_dense_observation,
            splits: manifest
                .splits
                .iter()
                .map(|split| {
                    let byte_len = split.shards.iter().map(|shard| shard.byte_len).sum::<u64>();
                    BcShardSplitOutputReport {
                        split: split_name(split.split).to_string(),
                        sample_count: split.sample_count,
                        shard_count: split.shard_count,
                        byte_len,
                        feature_flags: split.feature_flags,
                        record_size: split.record_size,
                        bytes_per_sample: (split.sample_count > 0)
                            .then(|| byte_len as f64 / split.sample_count as f64),
                        min_shard_bytes: split.shards.iter().map(|shard| shard.byte_len).min(),
                        max_shard_bytes: split.shards.iter().map(|shard| shard.byte_len).max(),
                    }
                })
                .collect(),
        },
        rates: BcShardBuildRates {
            games_per_second: Some(counters.loaded_games as f64 / seconds),
            samples_per_second: manifest.totals.sample_count as f64 / seconds,
            output_mib_per_second: (output_bytes as f64 / 1_048_576.0) / seconds,
            input_mib_per_second: input_bytes.map(|bytes| (bytes as f64 / 1_048_576.0) / seconds),
        },
        manifest_path: manifest_path.display().to_string(),
        progress_jsonl_path: progress_path.map(|p| p.display().to_string()),
    })
}

struct ProgressSink {
    path: Option<PathBuf>,
    file: Option<File>,
    committed_games: u64,
    feature_flags: u32,
    record_size: u32,
}

#[derive(Serialize)]
struct ProgressEvent<'a> {
    event: &'a str,
    event_version: u32,
    scope: &'a str,
    output_dir: &'a str,
    manifest_name: &'a str,
    split_mode: &'a str,
    shard_samples: usize,
    train_fraction: f32,
    storage_layout: &'a str,
    feature_flags: u32,
    record_size: u32,
    committed_games: u64,
    loaded_games: u64,
    skipped_games: u64,
    empty_games: u64,
    total_samples: u64,
    shard_count: usize,
    chunk_index: Option<usize>,
}

impl ProgressSink {
    fn new(config: &BuildBcShardsConfig, feature_flags: u32, record_size: u32) -> io::Result<Self> {
        let path = config
            .progress_jsonl_name
            .as_ref()
            .map(|name| config.output_dir.join(name));
        let file = match &path {
            Some(path) => Some(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)?,
            ),
            None => None,
        };
        let mut sink = Self {
            path,
            file,
            committed_games: 0,
            feature_flags,
            record_size,
        };
        sink.write_event(
            "run_started",
            "cumulative",
            config,
            &BuildCounters::default(),
            None,
        )?;
        Ok(sink)
    }

    fn game_committed(
        &mut self,
        config: &BuildBcShardsConfig,
        counters: &BuildCounters,
    ) -> io::Result<()> {
        self.committed_games += 1;
        if self.committed_games == 1 || self.committed_games.is_multiple_of(PROGRESS_EVERY_GAMES) {
            self.write_event("game_committed", "chunk", config, counters, None)?;
        }
        Ok(())
    }

    fn fragment_reused(
        &mut self,
        config: &BuildBcShardsConfig,
        fragment: &BcShardBuildFragment,
    ) -> io::Result<()> {
        self.write_event(
            "fragment_reused",
            "chunk",
            config,
            &fragment.counters,
            Some(fragment.chunk_index),
        )
    }
    fn fragment_built(
        &mut self,
        config: &BuildBcShardsConfig,
        fragment: &BcShardBuildFragment,
    ) -> io::Result<()> {
        self.write_event(
            "fragment_built",
            "chunk",
            config,
            &fragment.counters,
            Some(fragment.chunk_index),
        )
    }

    fn write_event(
        &mut self,
        event: &'static str,
        scope: &'static str,
        config: &BuildBcShardsConfig,
        counters: &BuildCounters,
        chunk_index: Option<usize>,
    ) -> io::Result<()> {
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        let output_dir = config.output_dir.display().to_string();
        let event = ProgressEvent {
            event,
            event_version: REPORT_SCHEMA_VERSION,
            scope,
            output_dir: &output_dir,
            manifest_name: &config.manifest_name,
            split_mode: split_mode_name(config.split_mode),
            shard_samples: config.shard_samples,
            train_fraction: config.train_fraction,
            storage_layout: STORAGE_LAYOUT_COMPACT,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
            committed_games: self.committed_games,
            loaded_games: counters.loaded_games,
            skipped_games: counters.skipped_games,
            empty_games: counters.empty_games,
            total_samples: counters.total_samples,
            shard_count: counters.shard_count,
            chunk_index,
        };
        serde_json::to_writer(&mut *file, &event)
            .map_err(|err| invalid_data(format!("failed to serialize progress event: {err}")))?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("json"),
        std::process::id()
    ));
    let result = (|| -> io::Result<()> {
        let mut file = File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut file, value).map_err(|err| {
            invalid_data(format!(
                "failed to serialize JSON {}: {err}",
                path.display()
            ))
        })?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        fsync_parent_dir(path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let _ = File::open(parent)?.sync_all();
        }
    }
    Ok(())
}

fn stable_json_hash<T: Serialize + ?Sized>(value: &T) -> io::Result<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| invalid_data(format!("failed to serialize fingerprint input: {err}")))?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

fn plan_split_reports(entries: &[BuildPlanEntry]) -> Vec<BcShardSplitPlanReport> {
    let train = entries
        .iter()
        .filter(|entry| entry.split == BcShardSplit::Train)
        .count();
    let validation = entries
        .iter()
        .filter(|entry| entry.split == BcShardSplit::Validation)
        .count();
    let mut splits = Vec::with_capacity(2);
    if train > 0 {
        splits.push(BcShardSplitPlanReport {
            split: split_name(BcShardSplit::Train).to_string(),
            planned_games: train,
        });
    }
    if validation > 0 {
        splits.push(BcShardSplitPlanReport {
            split: split_name(BcShardSplit::Validation).to_string(),
            planned_games: validation,
        });
    }
    splits
}

fn existing_output_bytes(config: &BuildBcShardsConfig) -> io::Result<u64> {
    let mut total = 0u64;
    if !config.output_dir.exists() {
        return Ok(0);
    }
    for entry in fs::read_dir(&config.output_dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let include = name.ends_with(".hydra-bc")
            || name == config.manifest_name
            || config.report_name.as_deref() == Some(name)
            || config.progress_jsonl_name.as_deref() == Some(name);
        if include {
            total = total
                .checked_add(fs::metadata(&path)?.len())
                .ok_or_else(|| {
                    invalid_data(format!(
                        "existing output byte count overflow under {}",
                        config.output_dir.display()
                    ))
                })?;
        }
    }
    Ok(total)
}

fn available_bytes(_path: &Path) -> Option<u64> {
    None
}

fn reject_stale_non_resume_output(config: &BuildBcShardsConfig) -> io::Result<()> {
    let manifest_path = config.output_dir.join(&config.manifest_name);
    if manifest_path.exists() {
        return Err(invalid_data(format!(
            "output manifest already exists: {}; choose a new output dir or enable resume",
            manifest_path.display()
        )));
    }
    for entry in fs::read_dir(&config.output_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".hydra-bc") {
            return Err(invalid_data(format!(
                "output dir contains stale shard file {}; choose a new output dir or enable resume",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn input_compressed_bytes(source_manifest: &DataManifest) -> Option<u64> {
    let mut total = 0u64;
    for source in &source_manifest.sources {
        match source {
            DataSource::LooseFile(path) | DataSource::Archive(path) => {
                total += fs::metadata(path).ok()?.len()
            }
            DataSource::ParsedSampleCache { path, .. } => total += fs::metadata(path).ok()?.len(),
        }
    }
    Some(total)
}

fn resume_state_dir(config: &BuildBcShardsConfig) -> PathBuf {
    config
        .resume_dir
        .clone()
        .unwrap_or_else(|| config.output_dir.join(DEFAULT_RESUME_DIR_NAME))
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

fn split_for_identity(identity: &str, config: &BuildBcShardsConfig) -> Option<BcShardSplit> {
    let split = if is_train_game(identity, config.train_fraction) {
        BcShardSplit::Train
    } else {
        BcShardSplit::Validation
    };
    config.split_mode.includes(split).then_some(split)
}

fn split_mode_name(mode: BcShardSplitMode) -> &'static str {
    match mode {
        BcShardSplitMode::Both => "both",
        BcShardSplitMode::Train => "train",
        BcShardSplitMode::Validation => "validation",
    }
}

fn split_name(split: BcShardSplit) -> &'static str {
    match split {
        BcShardSplit::Train => "train",
        BcShardSplit::Validation => "validation",
    }
}
