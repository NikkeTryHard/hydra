use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::bc_runtime::gated_bc_context;
use crate::data::sample::{MjaiSample, collate_samples, collate_samples_bc_owned_timed};
use crate::data_pipeline::{
    DataManifest, StreamingLoaderConfig, stream_train_epoch, stream_val_microbatches,
};
use crate::losses::HydraLoss;
use crate::model::{HydraModel, HydraModelConfig, HydraModelInit, HydraTrainModelExt};
use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use hydra_bc_shards::{
    BcShardReader as ExtractedBcShardReader, BcShardSplit as ExtractedBcShardSplit,
    load_bc_shard_reader as load_extracted_bc_shard_reader,
};
use hydra_replay_loader::{ReplayMaterializationStats, ReplayTargetProfile};
use hydra_train_runtime::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train_runtime::preflight::{
    BenchmarkMetadata, BenchmarkMode, BenchmarkResult, BenchmarkRuntimeConfig, BenchmarkScore,
    EffectiveRuntimeConfig, ExplicitSettings, LoaderRuntimeConfig, ModelFingerprintInput,
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_COLLATION,
    PROFILING_STAGE_FORWARD, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_LOSS, PROFILING_STAGE_OPTIMIZER_STEP, PROFILING_STAGE_PREFLIGHT_LOSS_INIT,
    PROFILING_STAGE_PREFLIGHT_MODEL_INIT, PROFILING_STAGE_PREFLIGHT_OPTIMIZER_INIT,
    PROFILING_STAGE_STAGE_2_BENCHMARK, PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION,
    PreflightArtifactEvent, PreflightArtifactEventKind, PreflightCacheEntry, PreflightCacheKey,
    PreflightCompletedPhase, PreflightConfig, PreflightReport, PreflightState, PreflightTuningMode,
    ProbeKind, ProbeResult, ProbeStatus, ProfilingEnvelope, candidate_ladder, preflight_cache_key,
    resolve_runtime_config,
};
use tboard::EventWriter;

type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch<f32>>;
use crate::advisory::{RuntimeAdvisory, selected_runtime_probe_advisories};
use crate::artifacts::{
    BcArtifactPaths, LatestCheckpointState, ManifestCacheHitSource, ManifestCacheRequest,
    PreflightBenchmarkPaths, PreflightBenchmarkReport, PreflightPaths, RlArtifactPaths,
    RlPreflightPaths, append_preflight_candidate_to_writer, append_preflight_event_to_writer,
    append_step_log_to_writer, ensure_preflight_metrics_placeholder, load_or_scan_manifest_cache,
    load_or_scan_manifest_cache_with_source, open_preflight_candidate_appender,
    open_preflight_event_appender, open_step_log_appender, read_preflight_cache,
    read_preflight_state, save_latest_checkpoint_and_state, write_preflight_benchmark_report,
    write_preflight_cache, write_preflight_report, write_preflight_state,
};
use crate::bc_fixed_shape::{
    FixedShapeProbeConfig, FixedShapeTrainConfig, benchmark_train_fixed_chunks,
    probe_train_fixed_chunks,
};
use crate::bc_metrics::batch_stats_from_outputs;
use crate::config_runtime::{configure_threads, train_device, trainer_config_from_train_config};
use crate::epoch_runner::{
    TrainLogicalBatchConfig, materialize_host_batch_owned, train_device_batch,
};
use crate::nvtx;
#[cfg(feature = "cuda-graph")]
use crate::pinned_transfer::{AsyncH2DContext, PinnedStagingArea, PreallocatedDeviceTensors};
use crate::presentation::{
    format_preflight_selection_line, format_preflight_summary_line, format_probe_status_line,
    format_timed_phase_message, make_bar, make_spinner, preflight_phase_label,
};
use crate::probe_ladder::{
    adaptive_oom_probe_next_index, dynamic_probe_ladder, probe_only_candidate_ladder,
};
use crate::probe_process::{mem_available_bytes, rl_probe_required_free_bytes};
use crate::probe_search::{
    ProbeGrowthDecision, ProbeGrowthState, ProbeRunSpec, ProbeSearchStopReason,
    finalize_probe_search, maybe_expand_probe_candidates, probe_search_plan,
    refine_probe_winner_locally, refine_top_k_probe_candidates_locally, rerun_probe_finalists,
};
#[cfg(not(test))]
use crate::probe_search::{probe_candidate_ladder, run_candidate_attempts};
use crate::probe_summary::{
    ProbeCandidateSummary, best_probe_summary, candidate_average, format_probe_selection_summary,
    probe_kind_name, summarize_probe_results,
};
use crate::probe_transport::{
    ProbeBatchArtifact, probe_result_path, rl_probe_result_path, write_probe_batch_artifact,
    write_probe_result,
};
use crate::resume::{BestValidation, EpochContinuation, runtime_resume_contract};
use crate::runtime_autotune_shim::{
    LoaderRuntimeScoreSeed, RankedLoaderRuntime, RuntimeTupleStats,
    autotune_ranked_loader_runtime_with_seed,
};
use crate::validation::ValidationSummary;
use crate::validation_runner::{
    ValidationContext, ValidationDataLoader, ValidationRuntime, materialize_validation_samples,
    run_validation,
};
use hydra_data_core::manifest::DataManifest as CoreDataManifest;
#[cfg(test)]
use hydra_train_runtime::config::read_config;
use hydra_train_runtime::config::{ProbeChildRequest, TrainConfig, default_num_threads_for_system};
use hydra_train_runtime::loss_policy::{build_bc_exit_config, build_loss_config};
use hydra_train_runtime::probe_request::{
    ProbeBatchRequest, ProbeRequest, probe_batch_child_request_from_cli,
    probe_child_request_from_cli,
};
use hydra_train_runtime::progress::{ScalarAverages, StepLogEntry};
use hydra_train_runtime::schedule::{TrainerScheduleConfig, effective_lr};
use hydra_train_runtime::validation::ValidationRunLimits;
use indicatif::ProgressBar;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

type BenchmarkOptimizerOf<B> = OptimizerAdaptor<Adam, HydraModel<B>, B>;
fn model_fingerprint_input(model: &HydraModelConfig) -> ModelFingerprintInput {
    ModelFingerprintInput {
        num_blocks: model.num_blocks,
        input_channels: model.input_channels,
        hidden_channels: model.hidden_channels,
        num_groups: model.num_groups,
        action_space: model.action_space,
        score_bins: model.score_bins,
    }
}

type StageTwoCachedValidationSamples = Option<Arc<[Box<[MjaiSample]>]>>;

fn append_step_log(
    path: &Path,
    entry: &StepLogEntry<crate::validation::DeltaQPromotionSnapshot, RuntimeAdvisory>,
) -> Result<(), String> {
    let mut file = open_step_log_appender(path)?;
    append_step_log_to_writer(&mut file, entry)
}

fn log_tensorboard<W: Write>(
    tb: &mut EventWriter<W>,
    epoch: usize,
    train: &ScalarAverages,
    val_summary: Option<&ValidationSummary>,
    lr: f64,
    best_validation: Option<BestValidation>,
) -> Result<(), String> {
    let step = epoch as i64;
    tb.write_scalar(step, "train/total_loss", train.total_loss as f32)
        .map_err(|err| format!("tensorboard write train/total_loss failed: {err}"))?;
    tb.write_scalar(
        step,
        "train/policy_agreement",
        train.policy_agreement as f32,
    )
    .map_err(|err| format!("tensorboard write train/policy_agreement failed: {err}"))?;
    if let Some(val_summary) = val_summary {
        tb.write_scalar(step, "val/policy_agreement", val_summary.agreement as f32)
            .map_err(|err| format!("tensorboard write val/policy_agreement failed: {err}"))?;
        tb.write_scalar(step, "val/policy_loss", val_summary.policy_loss as f32)
            .map_err(|err| format!("tensorboard write val/policy_loss failed: {err}"))?;
        tb.write_scalar(step, "val/total_loss", val_summary.total_loss as f32)
            .map_err(|err| format!("tensorboard write val/total_loss failed: {err}"))?;
    }
    tb.write_scalar(step, "train/lr", lr as f32)
        .map_err(|err| format!("tensorboard write train/lr failed: {err}"))?;
    if let Some(best) = best_validation {
        tb.write_scalar(step, "val/best_policy_loss", best.policy_loss as f32)
            .map_err(|err| format!("tensorboard write val/best_policy_loss failed: {err}"))?;
    }
    Ok(())
}

struct TrainValidationLoader<'a> {
    config: &'a StreamingLoaderConfig,
}

impl ValidationDataLoader for TrainValidationLoader<'_> {
    fn stream_val_microbatches<'b>(
        &'b self,
        manifest: &'b CoreDataManifest,
        microbatch_size: usize,
        progress: Option<&'b ProgressBar>,
    ) -> Box<dyn Iterator<Item = io::Result<Vec<MjaiSample>>> + 'b> {
        Box::new(stream_val_microbatches(
            manifest,
            self.config,
            microbatch_size,
            progress,
        ))
    }
}

fn validation_loader(config: &StreamingLoaderConfig) -> TrainValidationLoader<'_> {
    TrainValidationLoader { config }
}

fn trainer_schedule(config: &hydra_train_types::config::BCTrainerConfig) -> TrainerScheduleConfig {
    TrainerScheduleConfig::new(config.lr, config.min_learning_rate, config.warmup_steps)
}

struct ProbeLoopState {
    completed_steps: usize,
    measure_start: Option<Instant>,
}

impl ProbeLoopState {
    fn new() -> Self {
        Self {
            completed_steps: 0,
            measure_start: None,
        }
    }
}

fn emit_probe_start_progress(request: ProbeRequest, microbatch_size: usize) -> Result<(), String> {
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
        probe_kind_name(request.kind),
        microbatch_size,
        request.warmup_steps,
        request.measure_steps,
    ))
}

fn advance_probe_loop(
    state: &mut ProbeLoopState,
    request: ProbeRequest,
    microbatch_size: usize,
    measured_samples_per_step: usize,
) -> Result<Option<f64>, String> {
    emit_probe_step_progress(
        request.kind,
        microbatch_size,
        state.completed_steps,
        request,
        state.measure_start,
        measured_samples_per_step,
    )?;
    state.completed_steps += 1;
    if state.completed_steps == request.warmup_steps {
        state.measure_start = Some(Instant::now());
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=measure_start total_steps={}",
            probe_kind_name(request.kind),
            microbatch_size,
            request.measure_steps.max(1)
        ))?;
    }
    let target_steps = request.warmup_steps + request.measure_steps;
    if state.completed_steps >= target_steps {
        let elapsed = state
            .measure_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        return Ok(Some(measure_samples_per_second(
            request.measure_steps.max(1) * measured_samples_per_step,
            elapsed,
        )));
    }
    Ok(None)
}

/// Selected BC preflight execution result passed back to mode dispatch.
#[derive(Debug, Clone)]
pub struct PreflightRuntime {
    /// Effective runtime selected by cache/probes/benchmark.
    pub runtime: EffectiveRuntimeConfig,
    /// Probe results for train microbatch search; empty on cache hit.
    pub train_probe_results: Vec<ProbeResult>,
    /// Probe results for validation microbatch search; empty on cache hit.
    pub validation_probe_results: Vec<ProbeResult>,
    /// Stage-2 benchmark result when one was run and persisted.
    pub benchmark: Option<hydra_train_runtime::preflight::BenchmarkResult>,
    /// Runtime advisories derived from observed probe/benchmark data.
    pub advisories: Vec<RuntimeAdvisory>,
    /// Whether train/validation microbatch values were explicit in YAML.
    pub explicit: ExplicitSettings,
}

/// Selected RL preflight execution result passed back to mode dispatch.
#[derive(Debug, Clone)]
pub struct RlPreflightRuntime {
    /// Selected RL games per batch.
    pub selected_games_per_batch: usize,
    /// Selected RL microbatch size.
    pub selected_microbatch_size: usize,
    /// Effective runtime selected by cache/probes.
    pub runtime: EffectiveRuntimeConfig,
    /// RL games probe results; empty on cache hit.
    pub rl_games_probe_results: Vec<ProbeResult>,
    /// RL microbatch probe results; empty on cache hit.
    pub rl_microbatch_probe_results: Vec<ProbeResult>,
}

/// Immutable BC preflight cache lookup context.
pub struct BcPreflightCacheContext {
    /// Cache key computed from selected-runtime inputs.
    pub cache_key: PreflightCacheKey,
    /// Paths used by BC preflight.
    pub paths: PreflightPaths,
    /// Explicit microbatch settings from config.
    pub explicit: ExplicitSettings,
}

/// Computes BC preflight cache context without executing probes.
pub fn bc_preflight_cache_context(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
    artifacts: &BcArtifactPaths,
) -> BcPreflightCacheContext {
    let model_fingerprint = model_fingerprint_input(model_config);
    BcPreflightCacheContext {
        cache_key: preflight_cache_key(
            config,
            preflight,
            &model_fingerprint,
            device_label,
            default_num_threads_for_system(),
        ),
        paths: PreflightPaths::new(artifacts),
        explicit: ExplicitSettings {
            train_microbatch_explicit: config.microbatch_size.is_some(),
            validation_microbatch_explicit: config.validation_microbatch_size.is_some(),
        },
    }
}

/// Returns a matching BC preflight cache entry, preserving cache-hit probe-skip semantics.
pub fn matching_bc_preflight_cache(
    context: &BcPreflightCacheContext,
) -> Result<Option<PreflightCacheEntry>, String> {
    let Some(cached) = read_preflight_cache(&context.paths.cache_path)? else {
        return Ok(None);
    };
    if cached.cache_key == context.cache_key {
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight cache hit:",
                format!(
                    "reusing cached runtime train_mb={} val_mb={} -- benchmark skipped",
                    cached.runtime.selected.train_microbatch_size,
                    cached.runtime.selected.validation_microbatch_size,
                ),
            )
        );
        Ok(Some(cached))
    } else {
        Ok(None)
    }
}

/// Resolves selected BC runtime from a matching preflight cache, preserving probe-skip and explicit-setting semantics.
#[must_use]
pub fn bc_cached_runtime(
    cached: PreflightCacheEntry,
    explicit: ExplicitSettings,
) -> PreflightRuntime {
    PreflightRuntime {
        runtime: cached.runtime,
        train_probe_results: Vec::new(),
        validation_probe_results: Vec::new(),
        benchmark: cached.benchmark,
        advisories: Vec::new(),
        explicit,
    }
}

struct PreflightArtifactLogger {
    events: crate::artifacts::JsonlAppender,
    metrics: crate::artifacts::JsonlAppender,
    candidates: crate::artifacts::JsonlAppender,
    state_path: PathBuf,
    state: PreflightState,
}

impl PreflightArtifactLogger {
    fn new(paths: &PreflightPaths, cache_key: PreflightCacheKey) -> Result<Self, String> {
        paths.create_dirs()?;
        ensure_preflight_metrics_placeholder(&paths.metrics_log_path)?;
        let events = open_preflight_event_appender(&paths.events_log_path)?;
        let metrics = crate::artifacts::open_jsonl_appender(
            &paths.metrics_log_path,
            "preflight metrics log",
        )?;
        let candidates = open_preflight_candidate_appender(&paths.candidates_log_path)?;
        let state = match read_preflight_state(&paths.state_path)? {
            Some(existing) if existing.cache_key == cache_key && !existing.cache_written => {
                existing
            }
            Some(existing) if existing.cache_key == cache_key => PreflightState {
                cache_key,
                completed_phases: existing.completed_phases,
                selected_runtime: existing.selected_runtime,
                cache_written: false,
                candidate_records: existing.candidate_records,
            },
            _ => PreflightState {
                cache_key,
                completed_phases: Vec::new(),
                selected_runtime: None,
                cache_written: false,
                candidate_records: Vec::new(),
            },
        };
        write_preflight_state(&paths.state_path, &state)?;
        Ok(Self {
            events,
            metrics,
            candidates,
            state_path: paths.state_path.clone(),
            state,
        })
    }

    fn phase_started(&mut self, phase: &str, detail: Option<String>) -> Result<(), String> {
        append_preflight_event_to_writer(
            &mut self.events,
            &PreflightArtifactEvent {
                phase: phase.to_string(),
                kind: PreflightArtifactEventKind::Started,
                elapsed_seconds: None,
                detail,
                completed: None,
                planned: None,
            },
        )
    }

    fn phase_skipped(&mut self, phase: &str, detail: String) -> Result<(), String> {
        append_preflight_event_to_writer(
            &mut self.events,
            &PreflightArtifactEvent {
                phase: phase.to_string(),
                kind: PreflightArtifactEventKind::Skipped,
                elapsed_seconds: None,
                detail: Some(detail),
                completed: None,
                planned: None,
            },
        )
    }

    fn phase_interrupted(
        &mut self,
        phase: &str,
        elapsed_seconds: f64,
        detail: String,
    ) -> Result<(), String> {
        append_preflight_event_to_writer(
            &mut self.events,
            &PreflightArtifactEvent {
                phase: phase.to_string(),
                kind: PreflightArtifactEventKind::Interrupted,
                elapsed_seconds: Some(elapsed_seconds),
                detail: Some(detail),
                completed: None,
                planned: None,
            },
        )
    }

    fn phase_completed(
        &mut self,
        phase: &str,
        elapsed_seconds: f64,
        detail: Option<String>,
        selected_runtime: Option<EffectiveRuntimeConfig>,
    ) -> Result<(), String> {
        append_preflight_event_to_writer(
            &mut self.events,
            &PreflightArtifactEvent {
                phase: phase.to_string(),
                kind: PreflightArtifactEventKind::Completed,
                elapsed_seconds: Some(elapsed_seconds),
                detail,
                completed: None,
                planned: None,
            },
        )?;
        if !self
            .state
            .completed_phases
            .iter()
            .any(|completed| completed.phase == phase)
        {
            self.state.completed_phases.push(PreflightCompletedPhase {
                phase: phase.to_string(),
                elapsed_seconds,
            });
        }
        if let Some(runtime) = selected_runtime {
            self.state.selected_runtime = Some(runtime);
        }
        write_preflight_state(&self.state_path, &self.state)
    }

    fn candidate_results(&mut self, phase: &str, results: &[ProbeResult]) -> Result<(), String> {
        for result in results {
            let record = hydra_train_runtime::preflight::PreflightCandidateRecord {
                phase: phase.to_string(),
                result: result.clone(),
            };
            append_preflight_candidate_to_writer(&mut self.candidates, &record)?;
            self.state.candidate_records.push(record);
        }
        write_preflight_state(&self.state_path, &self.state)
    }

    fn metrics_event(
        &mut self,
        event: &hydra_train_runtime::preflight::SystemMetricsEvent,
    ) -> Result<(), String> {
        crate::artifacts::append_jsonl_entry(
            &mut self.metrics,
            event,
            "preflight metrics log",
            "preflight metric event",
        )
    }

    fn cache_written(&mut self, runtime: EffectiveRuntimeConfig) -> Result<(), String> {
        self.state.selected_runtime = Some(runtime);
        self.state.cache_written = true;
        write_preflight_state(&self.state_path, &self.state)
    }
}

fn log_interrupted_phase_if_probe_interrupted(
    logger: &mut PreflightArtifactLogger,
    phase: &str,
    started: Instant,
    err: &str,
) -> Result<(), String> {
    if err.contains("preflight interrupted") {
        logger.phase_interrupted(
            phase,
            started.elapsed().as_secs_f64(),
            "interrupt received; probe child/process group cleanup requested".to_string(),
        )?;
    }
    Ok(())
}

fn completed_phase_seconds(state: &PreflightState, phase: &str) -> Option<f64> {
    state
        .completed_phases
        .iter()
        .find(|completed| completed.phase == phase)
        .map(|completed| completed.elapsed_seconds)
}

fn resumed_candidate_results(state: &PreflightState, phase: &str) -> Vec<ProbeResult> {
    state
        .candidate_records
        .iter()
        .filter(|record| record.phase == phase)
        .map(|record| record.result.clone())
        .collect()
}

fn successful_candidate_selection(results: &[ProbeResult]) -> Option<usize> {
    results
        .iter()
        .filter(|result| result.status == ProbeStatus::Success)
        .max_by(|left, right| {
            let left_sps = left.measured_samples_per_second.unwrap_or(0.0);
            let right_sps = right.measured_samples_per_second.unwrap_or(0.0);
            left_sps.total_cmp(&right_sps)
        })
        .map(|result| result.candidate_microbatch)
}

/// Writes the BC preflight cache and optional benchmark report atomically.
pub fn persist_bc_preflight_runtime(
    cache_key: PreflightCacheKey,
    runtime: EffectiveRuntimeConfig,
    benchmark: Option<hydra_train_runtime::preflight::BenchmarkResult>,
    paths: &PreflightPaths,
    artifacts: &BcArtifactPaths,
) -> Result<(), String> {
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: cache_key.clone(),
            runtime,
            benchmark: benchmark.clone(),
        },
    )?;
    if let Some(benchmark) = benchmark {
        let benchmark_paths = PreflightBenchmarkPaths::new(artifacts);
        benchmark_paths.create_root_dir()?;
        write_preflight_benchmark_report(
            &benchmark_paths.report_path(),
            &PreflightBenchmarkReport {
                cache_key,
                runtime,
                benchmark,
            },
        )?;
    }
    Ok(())
}

/// Returns a matching RL preflight cache entry.
pub fn matching_rl_preflight_cache(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    artifacts: &RlArtifactPaths,
) -> Result<Option<PreflightCacheEntry>, String> {
    let paths = RlPreflightPaths::new(artifacts);
    let model_fingerprint = model_fingerprint_input(model_config);
    let cache_key = preflight_cache_key(
        config,
        preflight,
        &model_fingerprint,
        &config.device,
        default_num_threads_for_system(),
    );
    let Some(cached) = read_preflight_cache(&paths.cache_path)? else {
        return Ok(None);
    };
    Ok((cached.cache_key == cache_key).then_some(cached))
}

/// Persists RL preflight selected runtime in the shared preflight cache schema.
pub fn persist_rl_preflight_runtime(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &RlArtifactPaths,
    runtime: EffectiveRuntimeConfig,
) -> Result<(), String> {
    let paths = RlPreflightPaths::new(artifacts);
    let model_fingerprint = model_fingerprint_input(&HydraModelConfig::learner());
    let cache_key = preflight_cache_key(
        config,
        preflight,
        &model_fingerprint,
        &config.device,
        default_num_threads_for_system(),
    );
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key,
            runtime,
            benchmark: None,
        },
    )
}

#[derive(Debug, Clone)]
struct BenchmarkFinalist {
    runtime: BenchmarkRuntimeConfig,
    train_probe_samples_per_second: f64,
    validation_probe_samples_per_second: f64,
    loader_probe_samples_per_second: f64,
    unsafe_batch_size: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BenchmarkProbeScores {
    train: f64,
    validation: f64,
    loader: f64,
}
struct TrainBenchmarkOutcome<B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    model: HydraModel<B>,
    optimizer: BenchmarkOptimizerOf<B>,
    head_controller: HeadActivationController,
    stats: ScalarAverages,
    elapsed_seconds: f64,
    measured_samples: usize,
    materialization_stats: ReplayMaterializationStats,
    sub_stage_timing: hydra_train_runtime::progress::TrainSubStageTiming,
    model_init_seconds: f64,
    optimizer_init_seconds: f64,
    loss_init_seconds: f64,
}

struct BenchmarkEvaluationOutcome {
    score: BenchmarkScore,
    profiling: ProfilingEnvelope,
}

struct StageTwoFinalistInputs<'a> {
    config: &'a TrainConfig,
    preflight: &'a PreflightConfig,
    selected: &'a EffectiveRuntimeConfig,
    train_candidates: &'a [ProbeCandidateSummary],
    validation_candidates: &'a [ProbeCandidateSummary],
    loader_candidates: &'a [RankedLoaderRuntime],
    train_probe_results: &'a [ProbeResult],
    validation_probe_results: &'a [ProbeResult],
    ranked_loaders: &'a [RankedLoaderRuntime],
    unsafe_batch_candidates: &'a [usize],
}
struct StageTwoBenchmarkContext<'a> {
    config: &'a TrainConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    artifacts: &'a BcArtifactPaths,
    finalists: &'a [BenchmarkFinalist],
    train_candidates: usize,
    validation_candidates: usize,
    loader_candidates: usize,
}

struct StageTwoBenchmarkRunContext<'a> {
    config: &'a TrainConfig,
    benchmark_config: &'a TrainConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    candidate_artifacts: &'a BcArtifactPaths,
    finalist: &'a BenchmarkFinalist,
    train_candidates: usize,
    validation_candidates: usize,
    loader_candidates: usize,
    benchmarked_count: usize,
    cached_validation_samples: StageTwoCachedValidationSamples,
    validation_materialization_seconds: f64,
    validation_materialization_stats: ReplayMaterializationStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StageTwoBenchmarkValidationCacheKey {
    loader_archive_queue_bound: usize,
    loader_buffer_samples: usize,
    loader_buffer_games: usize,
    loader_num_threads: Option<usize>,
    validation_sample_limit: Option<usize>,
}

#[derive(Clone)]
struct StageTwoBenchmarkValidationCacheEntry {
    cached_samples: StageTwoCachedValidationSamples,
    materialization_seconds: f64,
    materialization_stats: ReplayMaterializationStats,
    remaining_uses: usize,
}

#[derive(Default)]
struct StageTwoBenchmarkValidationCache {
    entries: BTreeMap<StageTwoBenchmarkValidationCacheKey, StageTwoBenchmarkValidationCacheEntry>,
}

impl StageTwoBenchmarkValidationCache {
    fn new(
        config: &TrainConfig,
        preflight: &PreflightConfig,
        finalists: &[BenchmarkFinalist],
    ) -> Self {
        let entries = stage_two_benchmark_validation_cache_plan(config, preflight, finalists)
            .into_iter()
            .filter(|(key, uses)| key.validation_sample_limit.is_some() && *uses > 1)
            .map(|(key, remaining_uses)| {
                (
                    key,
                    StageTwoBenchmarkValidationCacheEntry {
                        cached_samples: None,
                        materialization_seconds: 0.0,
                        materialization_stats: ReplayMaterializationStats::default(),
                        remaining_uses,
                    },
                )
            })
            .collect();
        Self { entries }
    }

    fn checkout(
        &mut self,
        key: StageTwoBenchmarkValidationCacheKey,
        benchmark_config: &TrainConfig,
        manifest: &DataManifest,
    ) -> Result<
        (
            StageTwoCachedValidationSamples,
            f64,
            ReplayMaterializationStats,
        ),
        String,
    > {
        let Some(mut entry) = self.entries.remove(&key) else {
            return Ok((None, 0.0, ReplayMaterializationStats::default()));
        };
        if entry.cached_samples.is_none() {
            let loader_config = benchmark_loader_config(
                benchmark_config,
                LoaderRuntimeConfig {
                    num_threads: key.loader_num_threads,
                    buffer_games: key.loader_buffer_games,
                    buffer_samples: key.loader_buffer_samples,
                    archive_queue_bound: key.loader_archive_queue_bound,
                },
            );
            let started = Instant::now();
            let _ = hydra_replay_loader::drain_replay_materialization_stats();
            entry.cached_samples = materialize_validation_samples(
                benchmark_config,
                &validation_loader(&loader_config),
                manifest,
            )?
            .map(Arc::<[Box<[MjaiSample]>]>::from);
            entry.materialization_seconds = started.elapsed().as_secs_f64();
            entry.materialization_stats = hydra_replay_loader::drain_replay_materialization_stats();
        }
        let cached_samples = entry.cached_samples.clone();
        let materialization_seconds = entry.materialization_seconds;
        let materialization_stats = entry.materialization_stats;
        entry.remaining_uses = entry.remaining_uses.saturating_sub(1);
        if entry.remaining_uses > 0 {
            self.entries.insert(key, entry);
        }
        Ok((
            cached_samples,
            materialization_seconds,
            materialization_stats,
        ))
    }
}

fn emit_probe_progress(line: &str) -> Result<(), String> {
    println!("{}", line.trim());
    std::io::stdout()
        .flush()
        .map_err(|err| format!("failed flushing probe progress output: {err}"))
}

fn emit_probe_init_phase(kind_name: &str, candidate_mb: usize, phase: &str) -> Result<(), String> {
    emit_probe_progress(&format!(
        "probe_progress kind={kind_name} candidate_mb={candidate_mb} phase={phase}"
    ))
}

fn emit_probe_init_ready(
    kind: ProbeKind,
    kind_name: &str,
    candidate_mb: usize,
    model_ms: u128,
    optimizer_ms: u128,
    loss_ms: u128,
) -> Result<(), String> {
    crate::system_metrics::emit_system_metrics_event(
        &crate::system_metrics::probe_child_init_event(
            kind,
            candidate_mb,
            model_ms,
            optimizer_ms,
            loss_ms,
        ),
    );
    emit_probe_progress(&format!(
        "probe_progress kind={kind_name} candidate_mb={candidate_mb} phase=init_ready model_ms={model_ms} optimizer_ms={optimizer_ms} loss_ms={loss_ms}"
    ))
}

fn emit_probe_step_progress(
    kind: ProbeKind,
    microbatch_size: usize,
    completed_steps: usize,
    request: ProbeRequest,
    measure_start: Option<Instant>,
    measured_samples_per_step: usize,
) -> Result<(), String> {
    if completed_steps < request.warmup_steps {
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=warmup step={}/{}",
            probe_kind_name(kind),
            microbatch_size,
            completed_steps + 1,
            request.warmup_steps.max(1)
        ))
    } else {
        let measure_step = completed_steps + 1 - request.warmup_steps;
        let throughput = measure_start
            .map(|start| {
                measure_samples_per_second(
                    measure_step * measured_samples_per_step,
                    start.elapsed(),
                )
            })
            .unwrap_or(0.0);
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=measure step={}/{} throughput={:.2} samples/s",
            probe_kind_name(kind),
            microbatch_size,
            measure_step,
            request.measure_steps.max(1),
            throughput,
        ))
    }
}

fn fast_repeated_run_ladder(
    preflight: &PreflightConfig,
    batch_size: usize,
    seed: usize,
) -> Vec<usize> {
    let full = candidate_ladder(preflight, batch_size);
    if full.is_empty() {
        return vec![seed.min(batch_size).max(1)];
    }
    let window = preflight.fast_repeated_run_candidate_window.max(1);
    let nearest_idx = full
        .iter()
        .enumerate()
        .min_by_key(|(_, candidate)| candidate.abs_diff(seed))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let start = nearest_idx.saturating_sub(window);
    let end = (nearest_idx + window + 1).min(full.len());
    let mut candidates = full[start..end].to_vec();
    if !candidates.contains(&seed) && seed >= preflight.min_microbatch_size && seed <= batch_size {
        candidates.push(seed);
    }
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    candidates.dedup();
    candidates
}

fn prioritize_full_batch_train_candidate(
    candidates: &mut Vec<usize>,
    batch_size: usize,
    seed: usize,
) {
    let batch_size = batch_size.max(1);
    if batch_size <= seed.max(1) {
        return;
    }
    candidates.retain(|candidate| *candidate != batch_size);
    let insert_at = usize::from(!candidates.is_empty());
    candidates.insert(insert_at, batch_size);
}
fn exact_train_probe_runtime_seed(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    selected_candidate: usize,
    results: &[ProbeResult],
    standard_attempts_len: usize,
) -> Option<LoaderRuntimeScoreSeed> {
    let standard_attempts = &results[..standard_attempts_len.min(results.len())];
    let matching_attempts = standard_attempts
        .iter()
        .filter(|result| {
            result.kind == ProbeKind::Train && result.candidate_microbatch == selected_candidate
        })
        .collect::<Vec<_>>();
    if matching_attempts.len() != preflight.required_successes.max(1) {
        return None;
    }

    let mut count = 0usize;
    let mut sum = 0.0;
    for attempt in matching_attempts {
        if attempt.status != ProbeStatus::Success {
            return None;
        }
        let throughput = attempt.measured_samples_per_second?;
        count += 1;
        sum += throughput;
    }

    Some(LoaderRuntimeScoreSeed {
        train_microbatch_size: selected_candidate.min(config.batch_size).max(1),
        tuple: (
            config.archive_queue_bound,
            config.buffer_samples,
            config.buffer_games,
        ),
        warmup_steps: preflight.warmup_steps.max(1),
        measure_steps: preflight.measure_steps.max(1),
        stats: RuntimeTupleStats { count, sum },
    })
}

fn train_probe_runtime_seed_from_successes(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    selected_candidate: usize,
    results: &[ProbeResult],
) -> Option<LoaderRuntimeScoreSeed> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for attempt in results.iter().filter(|result| {
        result.kind == ProbeKind::Train
            && result.candidate_microbatch == selected_candidate
            && result.status == ProbeStatus::Success
    }) {
        sum += attempt.measured_samples_per_second?;
        count += 1;
    }

    (count > 0).then_some(LoaderRuntimeScoreSeed {
        train_microbatch_size: selected_candidate.min(config.batch_size).max(1),
        tuple: (
            config.archive_queue_bound,
            config.buffer_samples,
            config.buffer_games,
        ),
        warmup_steps: preflight.warmup_steps.max(1),
        measure_steps: preflight.measure_steps.max(1),
        stats: RuntimeTupleStats { count, sum },
    })
}

fn search_train_microbatch(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    seed: usize,
) -> Result<(usize, Vec<ProbeResult>, Option<LoaderRuntimeScoreSeed>), String> {
    let mut candidates = if preflight.fast_repeated_run_profile {
        fast_repeated_run_ladder(preflight, config.batch_size, seed)
    } else {
        dynamic_probe_ladder(config, preflight, ProbeKind::Train, seed)
    };
    let explicit_candidate = config.microbatch_size;
    let use_explicit_only =
        explicit_candidate.is_some() && !preflight.allow_override_explicit_microbatch;
    if use_explicit_only {
        candidates = vec![explicit_candidate.unwrap_or(1)];
    } else {
        prioritize_full_batch_train_candidate(&mut candidates, config.batch_size, seed);
    }
    if preflight.fast_repeated_run_profile {
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight fast profile:",
                "using narrow train candidate window for repeated shard-backed run",
            )
        );
    }
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind=train candidates={:?} required_successes={}",
                candidates,
                preflight.required_successes.max(1),
            )
        )
    );
    let progress = make_bar(
        (candidates.len() * preflight.required_successes.max(1)) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut results = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut last_successful_candidate: Option<usize> = None;

    let mut index = 0usize;
    while index < candidates.len() {
        let candidate = candidates[index];
        if results
            .iter()
            .any(|result: &ProbeResult| result.candidate_microbatch == candidate)
        {
            index += 1;
            continue;
        }
        if let Some(blocked) = maybe_block_host_ram_growth_probe(
            preflight,
            ProbeKind::Train,
            candidate,
            last_successful_candidate,
        ) {
            println!("{}", format_probe_status_line(&blocked));
            results.push(blocked);
            index += 1;
            continue;
        }
        let mut result_path_for =
            |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt);
        let passed = run_candidate_attempts(
            config_path,
            &mut result_path_for,
            ProbeRunSpec {
                kind: ProbeKind::Train,
                candidate,
                attempts: preflight.required_successes.max(1),
                warmup_steps: preflight.warmup_steps,
                measure_steps: preflight.measure_steps,
            },
            &mut results,
            &progress,
        )?;
        if !passed {
            if let Some(err) = fatal_probe_backend_error(&results) {
                progress.finish_with_message("preflight train ladder failed".red().to_string());
                return Err(err);
            }
            if use_explicit_only {
                progress.finish_with_message("preflight train ladder complete".green().to_string());
                return Err(format!(
                    "explicit train microbatch {} failed preflight",
                    candidate
                ));
            }
            index = adaptive_oom_probe_next_index(&candidates, &results, index + 1);
            continue;
        }
        last_successful_candidate = Some(candidate);
        let throughput = candidate_average(&results, candidate).unwrap_or(0.0);
        if throughput > best_score {
            best_score = throughput;
        }

        if use_explicit_only {
            progress.finish_with_message("preflight train ladder complete".green().to_string());
            let baseline_seed = exact_train_probe_runtime_seed(
                config,
                preflight,
                candidate,
                &results,
                results.len(),
            );
            return Ok((candidate, results, baseline_seed));
        }
        index = adaptive_oom_probe_next_index(&candidates, &results, index + 1);
    }

    if let Some(err) = fatal_probe_backend_error(&results) {
        return Err(err);
    }
    progress.finish_with_message("preflight train ladder complete".green().to_string());
    let standard_attempts_len = results.len();
    refine_probe_winner_locally(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        ProbeKind::Train,
        config,
        preflight,
        &mut results,
        &progress,
    )?;
    refine_top_k_probe_candidates_locally(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        ProbeKind::Train,
        config,
        preflight,
        &mut results,
        &progress,
    )?;
    rerun_probe_finalists(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        ProbeKind::Train,
        config,
        preflight,
        &mut results,
        &progress,
    )?;
    if let Some(err) = fatal_probe_backend_error(&results) {
        return Err(err);
    }
    let selected_summary = best_probe_summary(&results)
        .ok_or_else(|| "no stable train microbatch found in preflight".to_string())?;
    let baseline_seed = exact_train_probe_runtime_seed(
        config,
        preflight,
        selected_summary.candidate_microbatch,
        &results,
        standard_attempts_len,
    )
    .or_else(|| {
        train_probe_runtime_seed_from_successes(
            config,
            preflight,
            selected_summary.candidate_microbatch,
            &results,
        )
    });
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            ProbeKind::Train,
            &selected_summary,
        ))
    );
    Ok((
        selected_summary.candidate_microbatch,
        results,
        baseline_seed,
    ))
}

fn search_validation_microbatch(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    seed: usize,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let spec = ProbeSearchSpec::new(ProbeKind::Validation, seed)
        .with_explicit_candidate(config.validation_microbatch_size)
        .with_fast_repeated_run_ladder(preflight.fast_repeated_run_profile)
        .with_no_stable_error("no stable validation microbatch found in preflight");
    let result = search_probe_candidate_ladder(
        config_path,
        config,
        preflight,
        spec,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
    )?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            ProbeKind::Validation,
            &result.selected_summary,
        ))
    );
    Ok((result.selected_summary.candidate_microbatch, result.results))
}

fn diverse_probe_candidates(
    results: &[ProbeResult],
    selected_microbatch: usize,
    limit: usize,
    margin_ratio: f64,
) -> Vec<ProbeCandidateSummary> {
    let mut summaries = summarize_probe_results(results)
        .into_iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        right
            .average_samples_per_second
            .unwrap_or(0.0)
            .partial_cmp(&left.average_samples_per_second.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.candidate_microbatch.cmp(&left.candidate_microbatch))
    });
    if summaries.is_empty() {
        return summaries;
    }

    let best_score = summaries[0].average_samples_per_second.unwrap_or(0.0);
    let minimum_score = best_score * (1.0 - margin_ratio.max(0.0));
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    let mut selected_index = None;
    for (idx, summary) in summaries.iter().enumerate() {
        if summary.candidate_microbatch == selected_microbatch {
            selected_index = Some(idx);
        }
        if summary.average_samples_per_second.unwrap_or(0.0) >= minimum_score
            && seen.insert(summary.candidate_microbatch)
        {
            selected.push(summary.clone());
        }
    }

    if let Some(idx) = selected_index {
        for neighbor in [
            idx.saturating_sub(1),
            idx,
            (idx + 1).min(summaries.len() - 1),
        ] {
            let summary = &summaries[neighbor];
            if seen.insert(summary.candidate_microbatch) {
                selected.push(summary.clone());
            }
        }
    }

    if let Some(summary) = summaries
        .iter()
        .find(|summary| summary.candidate_microbatch == selected_microbatch)
        && seen.insert(summary.candidate_microbatch)
    {
        selected.push(summary.clone());
    }

    selected.sort_by(|left, right| {
        right
            .average_samples_per_second
            .unwrap_or(0.0)
            .partial_cmp(&left.average_samples_per_second.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.candidate_microbatch.cmp(&left.candidate_microbatch))
    });
    selected.truncate(limit.max(1));
    selected
}

fn select_loader_finalists(
    ranked: &[RankedLoaderRuntime],
    limit: usize,
    margin_ratio: f64,
    selected: LoaderRuntimeConfig,
) -> Vec<RankedLoaderRuntime> {
    if ranked.is_empty() {
        return Vec::new();
    }
    let best_score = ranked[0].train_samples_per_second;
    let minimum_score = best_score * (1.0 - margin_ratio.max(0.0));
    let mut finalists = Vec::new();
    let mut seen = BTreeSet::new();
    for loader in ranked {
        let key = (
            loader.loader.archive_queue_bound,
            loader.loader.buffer_samples,
            loader.loader.buffer_games,
            loader.loader.num_threads.unwrap_or(0),
        );
        if loader.train_samples_per_second >= minimum_score && seen.insert(key) {
            finalists.push(*loader);
        }
    }
    let selected_key = (
        selected.archive_queue_bound,
        selected.buffer_samples,
        selected.buffer_games,
        selected.num_threads.unwrap_or(0),
    );
    if let Some(loader) = ranked.iter().find(|loader| loader.loader == selected)
        && seen.insert(selected_key)
    {
        finalists.push(*loader);
    }
    finalists.truncate(limit.max(1));
    finalists
}

fn benchmark_runtime_matches_selected(
    candidate: &BenchmarkRuntimeConfig,
    selected: &EffectiveRuntimeConfig,
) -> bool {
    candidate.train_microbatch_size == selected.selected.train_microbatch_size
        && candidate.validation_microbatch_size == selected.selected.validation_microbatch_size
        && candidate.accum_steps == selected.selected.accum_steps
        && candidate.loader == selected.loader
}

type BenchmarkFinalistKey = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    u64,
    u64,
    usize,
);

fn benchmark_finalist_key(
    runtime: BenchmarkRuntimeConfig,
    unsafe_batch_size: Option<usize>,
) -> BenchmarkFinalistKey {
    (
        unsafe_batch_size.unwrap_or(0),
        runtime.train_microbatch_size,
        runtime.validation_microbatch_size,
        runtime.accum_steps,
        runtime.loader.archive_queue_bound,
        runtime.loader.buffer_samples,
        runtime.loader.buffer_games,
        runtime.loader.num_threads.unwrap_or(0),
        runtime.learning_rate.map(f64::to_bits).unwrap_or(0),
        runtime.min_learning_rate.map(f64::to_bits).unwrap_or(0),
        runtime.warmup_steps.unwrap_or(0),
    )
}

fn unsafe_lr_scale_candidates(preflight: &PreflightConfig) -> Vec<f64> {
    if preflight.tuning_mode != PreflightTuningMode::Unsafe {
        return Vec::new();
    }
    let mut candidates = preflight
        .unsafe_candidate_lr_scales
        .iter()
        .copied()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|left, right| left.to_bits() == right.to_bits());
    candidates
}

fn unsafe_warmup_step_candidates(preflight: &PreflightConfig) -> Vec<usize> {
    if preflight.tuning_mode != PreflightTuningMode::Unsafe {
        return Vec::new();
    }
    let mut candidates = preflight
        .unsafe_candidate_warmup_steps
        .iter()
        .copied()
        .filter(|steps| *steps > 0)
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn push_stage_two_finalist(
    finalists: &mut Vec<BenchmarkFinalist>,
    seen: &mut BTreeSet<BenchmarkFinalistKey>,
    runtime: BenchmarkRuntimeConfig,
    train_probe_samples_per_second: f64,
    validation_probe_samples_per_second: f64,
    loader_probe_samples_per_second: f64,
    unsafe_batch_size: Option<usize>,
) {
    let key = benchmark_finalist_key(runtime, unsafe_batch_size);
    if seen.insert(key) {
        finalists.push(BenchmarkFinalist {
            runtime,
            train_probe_samples_per_second,
            validation_probe_samples_per_second,
            loader_probe_samples_per_second,
            unsafe_batch_size,
        });
    }
}

fn push_unsafe_hyperparam_finalists(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    finalists: &mut Vec<BenchmarkFinalist>,
    seen: &mut BTreeSet<BenchmarkFinalistKey>,
    base_runtime: BenchmarkRuntimeConfig,
    scores: BenchmarkProbeScores,
    unsafe_batch_size: usize,
) {
    let lr_scales = unsafe_lr_scale_candidates(preflight);
    let warmup_steps = unsafe_warmup_step_candidates(preflight);
    if lr_scales.is_empty() && warmup_steps.is_empty() {
        return;
    }

    if !lr_scales.is_empty() {
        for scale in &lr_scales {
            let mut runtime = base_runtime;
            runtime.learning_rate = Some(config.bc.learning_rate * *scale);
            runtime.min_learning_rate = Some(config.bc.min_learning_rate * *scale);
            push_stage_two_finalist(
                finalists,
                seen,
                runtime,
                scores.train,
                scores.validation,
                scores.loader,
                Some(unsafe_batch_size),
            );
        }
    }

    if !warmup_steps.is_empty() {
        for warmup in &warmup_steps {
            let mut runtime = base_runtime;
            runtime.warmup_steps = Some(*warmup);
            push_stage_two_finalist(
                finalists,
                seen,
                runtime,
                scores.train,
                scores.validation,
                scores.loader,
                Some(unsafe_batch_size),
            );
        }
    }

    for scale in &lr_scales {
        for warmup in &warmup_steps {
            let mut runtime = base_runtime;
            runtime.learning_rate = Some(config.bc.learning_rate * *scale);
            runtime.min_learning_rate = Some(config.bc.min_learning_rate * *scale);
            runtime.warmup_steps = Some(*warmup);
            push_stage_two_finalist(
                finalists,
                seen,
                runtime,
                scores.train,
                scores.validation,
                scores.loader,
                Some(unsafe_batch_size),
            );
        }
    }
}

fn unsafe_stage_two_batch_candidates(preflight: &PreflightConfig) -> Vec<usize> {
    if preflight.tuning_mode != PreflightTuningMode::Unsafe {
        return Vec::new();
    }
    let mut candidates: Vec<usize> = preflight
        .unsafe_candidate_batch_sizes
        .iter()
        .copied()
        .filter(|batch_size| *batch_size > 0)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn build_stage_two_finalists(inputs: StageTwoFinalistInputs<'_>) -> Vec<BenchmarkFinalist> {
    let StageTwoFinalistInputs {
        config,
        preflight,
        selected,
        train_candidates,
        validation_candidates,
        loader_candidates,
        train_probe_results,
        validation_probe_results,
        ranked_loaders,
        unsafe_batch_candidates,
    } = inputs;
    let mut finalists = Vec::new();
    let mut seen = BTreeSet::new();
    for train_summary in train_candidates {
        for validation_summary in validation_candidates {
            for loader in loader_candidates {
                let train_microbatch_size =
                    train_summary.candidate_microbatch.min(config.batch_size);
                let runtime = BenchmarkRuntimeConfig {
                    train_microbatch_size,
                    validation_microbatch_size: validation_summary.candidate_microbatch.max(1),
                    accum_steps: config
                        .batch_size
                        .div_ceil(train_microbatch_size.max(1))
                        .max(1),
                    loader: loader.loader,
                    learning_rate: None,
                    min_learning_rate: None,
                    warmup_steps: None,
                };
                let train_probe_samples_per_second =
                    train_summary.average_samples_per_second.unwrap_or(0.0);
                let validation_probe_samples_per_second =
                    validation_summary.average_samples_per_second.unwrap_or(0.0);
                push_stage_two_finalist(
                    &mut finalists,
                    &mut seen,
                    runtime,
                    train_probe_samples_per_second,
                    validation_probe_samples_per_second,
                    loader.train_samples_per_second,
                    None,
                );
                for unsafe_batch_size in unsafe_batch_candidates {
                    let train_microbatch_size = train_summary
                        .candidate_microbatch
                        .min(*unsafe_batch_size)
                        .max(1);
                    let runtime = BenchmarkRuntimeConfig {
                        train_microbatch_size,
                        validation_microbatch_size: validation_summary.candidate_microbatch.max(1),
                        accum_steps: unsafe_batch_size
                            .div_ceil(train_microbatch_size.max(1))
                            .max(1),
                        loader: loader.loader,
                        learning_rate: None,
                        min_learning_rate: None,
                        warmup_steps: None,
                    };
                    push_stage_two_finalist(
                        &mut finalists,
                        &mut seen,
                        runtime,
                        train_probe_samples_per_second,
                        validation_probe_samples_per_second,
                        loader.train_samples_per_second,
                        Some(*unsafe_batch_size),
                    );
                    push_unsafe_hyperparam_finalists(
                        config,
                        preflight,
                        &mut finalists,
                        &mut seen,
                        runtime,
                        BenchmarkProbeScores {
                            train: train_probe_samples_per_second,
                            validation: validation_probe_samples_per_second,
                            loader: loader.train_samples_per_second,
                        },
                        *unsafe_batch_size,
                    );
                }
            }
        }
    }
    if !finalists
        .iter()
        .any(|candidate| benchmark_runtime_matches_selected(&candidate.runtime, selected))
    {
        finalists.push(BenchmarkFinalist {
            runtime: BenchmarkRuntimeConfig {
                train_microbatch_size: selected.selected.train_microbatch_size,
                validation_microbatch_size: selected.selected.validation_microbatch_size,
                accum_steps: selected.selected.accum_steps,
                loader: selected.loader,
                learning_rate: selected.selected.unsafe_selected_learning_rate,
                min_learning_rate: selected.selected.unsafe_selected_min_learning_rate,
                warmup_steps: selected.selected.unsafe_selected_warmup_steps,
            },
            train_probe_samples_per_second: candidate_average(
                train_probe_results,
                selected.selected.train_microbatch_size,
            )
            .unwrap_or(0.0),
            validation_probe_samples_per_second: candidate_average(
                validation_probe_results,
                selected.selected.validation_microbatch_size,
            )
            .unwrap_or(0.0),
            loader_probe_samples_per_second: ranked_loaders
                .iter()
                .find(|loader| loader.loader == selected.loader)
                .map(|loader| loader.train_samples_per_second)
                .unwrap_or(0.0),
            unsafe_batch_size: selected.selected.unsafe_selected_batch_size,
        });
    }
    finalists.sort_by(|left, right| {
        let left_score = left.train_probe_samples_per_second
            + left.validation_probe_samples_per_second
            + left.loader_probe_samples_per_second;
        let right_score = right.train_probe_samples_per_second
            + right.validation_probe_samples_per_second
            + right.loader_probe_samples_per_second;
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .runtime
                    .train_microbatch_size
                    .cmp(&left.runtime.train_microbatch_size)
            })
    });
    finalists.truncate(preflight.real_benchmark_max_finalists.max(1));
    finalists
}

fn benchmark_loader_config(
    config: &TrainConfig,
    loader: LoaderRuntimeConfig,
) -> StreamingLoaderConfig {
    StreamingLoaderConfig {
        buffer_games: loader.buffer_games,
        buffer_samples: loader.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: loader.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
        source_filters: config.source_filters.clone(),
        replay_target_profile: ReplayTargetProfile::minimal_bc(),
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
        num_threads: config.num_threads,
    }
}

fn benchmark_train_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    runtime: BenchmarkRuntimeConfig,
) -> TrainConfig {
    let mut tuned = config.clone();
    tuned.microbatch_size = Some(runtime.train_microbatch_size);
    if let Some(unsafe_batch_size) = runtime_unsafe_batch_size(preflight, config, runtime) {
        tuned.batch_size = unsafe_batch_size;
    }
    tuned.validation_microbatch_size = Some(runtime.validation_microbatch_size);
    tuned.num_threads = runtime.loader.num_threads;
    tuned.buffer_games = runtime.loader.buffer_games;
    tuned.buffer_samples = runtime.loader.buffer_samples;
    tuned.archive_queue_bound = runtime.loader.archive_queue_bound;
    if let Some(learning_rate) = runtime.learning_rate {
        tuned.bc.learning_rate = learning_rate;
    }
    if let Some(min_learning_rate) = runtime.min_learning_rate {
        tuned.bc.min_learning_rate = min_learning_rate;
    }
    if let Some(warmup_steps) = runtime.warmup_steps {
        tuned.bc.warmup_steps = warmup_steps;
    }
    tuned
}

fn runtime_unsafe_batch_size(
    preflight: &PreflightConfig,
    _config: &TrainConfig,
    runtime: BenchmarkRuntimeConfig,
) -> Option<usize> {
    if preflight.tuning_mode != PreflightTuningMode::Unsafe {
        return None;
    }
    let train_capacity = runtime
        .train_microbatch_size
        .saturating_mul(runtime.accum_steps.max(1));
    unsafe_stage_two_batch_candidates(preflight)
        .into_iter()
        .find(|candidate| *candidate == train_capacity)
}

fn benchmark_validation_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    runtime: BenchmarkRuntimeConfig,
) -> TrainConfig {
    let mut tuned = benchmark_train_config(config, preflight, runtime);
    if tuned.max_validation_batches.is_none() && tuned.max_validation_samples.is_none() {
        tuned.max_validation_samples = Some(
            runtime
                .validation_microbatch_size
                .saturating_mul(8)
                .max(config.batch_size),
        );
    }
    tuned
}

fn benchmark_config_for_finalist(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    finalist: &BenchmarkFinalist,
) -> TrainConfig {
    let mut benchmark_config = benchmark_validation_config(config, preflight, finalist.runtime);
    if let Some(batch_size) = finalist.unsafe_batch_size {
        benchmark_config.batch_size = batch_size.max(1);
    }
    benchmark_config
}

fn stage_two_benchmark_validation_cache_key(
    benchmark_config: &TrainConfig,
    loader: LoaderRuntimeConfig,
) -> StageTwoBenchmarkValidationCacheKey {
    StageTwoBenchmarkValidationCacheKey {
        loader_archive_queue_bound: loader.archive_queue_bound,
        loader_buffer_samples: loader.buffer_samples,
        loader_buffer_games: loader.buffer_games,
        loader_num_threads: loader.num_threads,
        validation_sample_limit: ValidationRunLimits::from_config(benchmark_config).sample_limit,
    }
}

fn stage_two_benchmark_validation_cache_plan(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    finalists: &[BenchmarkFinalist],
) -> BTreeMap<StageTwoBenchmarkValidationCacheKey, usize> {
    let mut counts = BTreeMap::new();
    for finalist in finalists {
        let benchmark_config = benchmark_validation_config(config, preflight, finalist.runtime);
        let key =
            stage_two_benchmark_validation_cache_key(&benchmark_config, finalist.runtime.loader);
        *counts.entry(key).or_default() += 1;
    }
    counts
}

fn benchmark_projected_events(train_steps: usize, interval: usize) -> f64 {
    train_steps as f64 / interval.max(1) as f64
}

fn benchmark_metadata(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    train_candidates: usize,
    validation_candidates: usize,
    loader_candidates: usize,
    finalists_benchmarked: usize,
) -> BenchmarkMetadata {
    let measured_train_steps = preflight.real_benchmark_train_steps.max(1);
    BenchmarkMetadata {
        mode: BenchmarkMode::CadenceAwareProjection,
        selection_metric: "wall_clock_effective_throughput".to_string(),
        train_probe_candidates_considered: train_candidates,
        validation_probe_candidates_considered: validation_candidates,
        loader_candidates_considered: loader_candidates,
        finalists_benchmarked,
        warmup_steps: preflight.real_benchmark_warmup_steps.max(1),
        measured_train_steps,
        projected_validation_events: benchmark_projected_events(
            measured_train_steps,
            config.validate_every_n_steps,
        ),
        projected_checkpoint_events: benchmark_projected_events(
            measured_train_steps,
            config.checkpoint_every_n_steps,
        ),
        projected_logging_events: benchmark_projected_events(
            measured_train_steps,
            config.log_every_n_steps,
        ),
    }
}

fn replay_materialization_profile(stats: ReplayMaterializationStats) -> ProfilingEnvelope {
    ProfilingEnvelope::nested(
        "raw_replay_materialization",
        stats.elapsed().as_secs_f64(),
        vec![
            ProfilingEnvelope::leaf("decompression", nanos_to_seconds(stats.decompress_ns)),
            ProfilingEnvelope::leaf("json_parse", nanos_to_seconds(stats.json_parse_ns)),
            ProfilingEnvelope::leaf("replay_update", nanos_to_seconds(stats.replay_update_ns)),
            ProfilingEnvelope::leaf(
                "observation_encode",
                nanos_to_seconds(stats.observation_encode_ns),
            ),
            ProfilingEnvelope::leaf("mask_build", nanos_to_seconds(stats.mask_build_ns)),
        ],
    )
}

fn train_substage_profile(
    stage: &'static str,
    elapsed_seconds: f64,
    timing: hydra_train_runtime::progress::TrainSubStageTiming,
) -> ProfilingEnvelope {
    ProfilingEnvelope::nested(
        stage,
        elapsed_seconds,
        vec![
            ProfilingEnvelope::leaf(PROFILING_STAGE_COLLATION, timing.collation_seconds),
            ProfilingEnvelope::nested(
                PROFILING_STAGE_H2D_TRANSFER,
                timing.h2d_transfer_seconds,
                timing
                    .to_profiling_children()
                    .into_iter()
                    .find(|child| child.stage == PROFILING_STAGE_H2D_TRANSFER)
                    .map(|child| child.children)
                    .unwrap_or_default(),
            ),
            ProfilingEnvelope::leaf(PROFILING_STAGE_FORWARD, timing.forward_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_LOSS, timing.loss_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_BACKWARD, timing.backward_seconds),
            ProfilingEnvelope::leaf(
                PROFILING_STAGE_OPTIMIZER_STEP,
                timing.optimizer_step_seconds,
            ),
        ],
    )
}

fn nanos_to_seconds(ns: u128) -> f64 {
    ns as f64 / 1_000_000_000.0
}

fn benchmark_score(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    profiling: &ProfilingEnvelope,
    validation_samples: usize,
) -> BenchmarkEvaluationOutcome {
    let train_seconds = profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_TRAIN)
        .map(|child| child.elapsed_seconds)
        .unwrap_or_default();
    let validation_seconds = profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_VALIDATION)
        .map(|child| child.elapsed_seconds)
        .unwrap_or_default();
    let checkpoint_seconds = profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_CHECKPOINT)
        .map(|child| child.elapsed_seconds)
        .unwrap_or_default();
    let logging_seconds = profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_LOGGING)
        .map(|child| child.elapsed_seconds)
        .unwrap_or_default();
    let train_steps = preflight.real_benchmark_train_steps.max(1);
    let train_samples = train_steps * config.batch_size;
    let projected_validation_events =
        benchmark_projected_events(train_steps, config.validate_every_n_steps);
    let projected_checkpoint_events =
        benchmark_projected_events(train_steps, config.checkpoint_every_n_steps);
    let projected_logging_events =
        benchmark_projected_events(train_steps, config.log_every_n_steps);
    let total_elapsed_seconds = train_seconds
        + validation_seconds * projected_validation_events
        + checkpoint_seconds * projected_checkpoint_events
        + logging_seconds * projected_logging_events;
    BenchmarkEvaluationOutcome {
        score: BenchmarkScore {
            wall_clock_samples_per_second: measure_samples_per_second(
                train_samples,
                Duration::from_secs_f64(total_elapsed_seconds.max(f64::EPSILON)),
            ),
            train_only_samples_per_second: measure_samples_per_second(
                train_samples,
                Duration::from_secs_f64(train_seconds.max(f64::EPSILON)),
            ),
            train_seconds,
            validation_seconds,
            checkpoint_seconds,
            logging_seconds,
            total_elapsed_seconds,
            train_steps,
            validation_samples,
        },
        profiling: ProfilingEnvelope::nested(
            PROFILING_STAGE_STAGE_2_BENCHMARK,
            total_elapsed_seconds,
            profiling.children.clone(),
        ),
    }
}

fn run_stage_two_benchmark_scopes<
    TrainOut,
    ValidationOut,
    TrainFn,
    ValidationFn,
    CheckpointFn,
    LoggingFn,
>(
    train: TrainFn,
    validation: ValidationFn,
    checkpoint: CheckpointFn,
    logging: LoggingFn,
) -> Result<(TrainOut, ValidationOut, f64, f64), String>
where
    TrainFn: FnOnce() -> Result<TrainOut, String>,
    ValidationFn: FnOnce(&mut TrainOut) -> Result<ValidationOut, String>,
    CheckpointFn: FnOnce(&TrainOut, &ValidationOut) -> Result<f64, String>,
    LoggingFn: FnOnce(&TrainOut, &ValidationOut) -> Result<f64, String>,
{
    let _benchmark_scope = nvtx::scope(PROFILING_STAGE_STAGE_2_BENCHMARK);
    let mut train_outcome = {
        let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
        train()?
    };
    let validation_summary = {
        let _validation_scope = nvtx::scope(PROFILING_STAGE_VALIDATION);
        validation(&mut train_outcome)?
    };
    let checkpoint_seconds = {
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        checkpoint(&train_outcome, &validation_summary)?
    };
    let logging_seconds = {
        let _logging_scope = nvtx::scope(PROFILING_STAGE_LOGGING);
        logging(&train_outcome, &validation_summary)?
    };
    Ok((
        train_outcome,
        validation_summary,
        checkpoint_seconds,
        logging_seconds,
    ))
}

fn benchmark_train_window_for_backend<B>(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<TrainBenchmarkOutcome<B>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let train_cfg = trainer_config_from_train_config(config);

    let model_init_started = Instant::now();
    let mut model = model_config.init::<B>(train_device);
    let model_init_seconds = model_init_started.elapsed().as_secs_f64();

    let optimizer_init_started = Instant::now();
    let mut optimizer: BenchmarkOptimizerOf<B> = train_cfg.optimizer_config().init();
    let optimizer_init_seconds = optimizer_init_started.elapsed().as_secs_f64();

    let loss_init_started = Instant::now();
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let loss_init_seconds = loss_init_started.elapsed().as_secs_f64();
    let mut head_controller = HeadActivationController::new(
        HeadActivationConfig::default_with_params(model_config.estimated_params()),
    );
    let loader = benchmark_loader_config(
        config,
        LoaderRuntimeConfig {
            num_threads: config.num_threads,
            buffer_games: config.buffer_games,
            buffer_samples: config.buffer_samples,
            archive_queue_bound: config.archive_queue_bound,
        },
    );
    let microbatch_size = config
        .microbatch_size
        .unwrap_or(config.batch_size)
        .min(config.batch_size)
        .max(1);
    let warmup_steps = preflight.real_benchmark_warmup_steps.max(1);
    let measured_train_steps = preflight.real_benchmark_train_steps.max(1);
    let target_steps = warmup_steps + measured_train_steps;
    let mut completed_steps = 0usize;
    let mut pending_samples = std::collections::VecDeque::new();
    let mut measured_stats = ScalarAverages::default();
    let mut measure_start = None;
    let _ = hydra_replay_loader::drain_replay_materialization_stats();
    let mut materialization_stats = ReplayMaterializationStats::default();
    let mut sub_stage_timing = hydra_train_runtime::progress::TrainSubStageTiming::default();

    for buffer_result in stream_train_epoch(manifest, &loader, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("benchmark train stream failed: {err}"))?;
        materialization_stats
            .merge_assign(hydra_replay_loader::drain_replay_materialization_stats());
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let mut step_batches = Vec::new();

            if let Some(fixed_shape) = benchmark_train_fixed_chunks(FixedShapeTrainConfig {
                logical_batch: &logical_batch,
                augment: config.augment,
                microbatch_size,
                train_device,
                loss_fn: &loss_fn,
                bc_exit_cfg: &exit_cfg,
                head_controller: &mut head_controller,
                model: &model,
                use_amp: false,
            })? {
                let lr = effective_lr(
                    trainer_schedule(&train_cfg),
                    completed_steps,
                    target_steps.max(1),
                );
                let optimizer_started = Instant::now();
                model = optimizer.step(lr, model, fixed_shape.grads);
                head_controller.tick_warmup();
                let mut fixed_timing = fixed_shape.sub_stage_timing;
                fixed_timing.optimizer_step_seconds += optimizer_started.elapsed().as_secs_f64();
                sub_stage_timing.accumulate(&fixed_timing);
                step_batches = fixed_shape.batch_stats;
            } else {
                let logical_batch_len = logical_batch.len().max(1) as f32;
                let mut accumulator: GradientsAccumulator<HydraModel<B>> =
                    GradientsAccumulator::new();

                for chunk in logical_batch.chunks(microbatch_size) {
                    let Some(collated) =
                        collate_samples_bc_owned_timed::<B>(chunk, config.augment, train_device)
                            .map_err(|err| format!("benchmark train collation failed: {err}"))?
                    else {
                        continue;
                    };
                    sub_stage_timing.collation_seconds += collated.cpu_prep_seconds;
                    sub_stage_timing.h2d_transfer_seconds += collated.device_materialize_seconds;
                    sub_stage_timing.h2d_tensor_materialize_seconds +=
                        collated.device_materialize_seconds;
                    let obs = collated.obs;
                    let batch = collated.batch;
                    let targets = collated.targets;
                    let (active_loss_fn, warmup_heads) =
                        gated_bc_context(Some(&mut head_controller), &loss_fn, &targets);
                    let t_forward = Instant::now();
                    let output =
                        model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads);
                    sub_stage_timing.forward_seconds += t_forward.elapsed().as_secs_f64();
                    let t_loss = Instant::now();
                    let breakdown = active_loss_fn.total_loss(&output, &targets);
                    let total = crate::bc_runtime::maybe_add_exit_loss(
                        breakdown.total.clone(),
                        output.policy_logits.clone(),
                        batch.exit_target.as_ref(),
                        batch.exit_mask.as_ref(),
                        &exit_cfg,
                    );
                    sub_stage_timing.loss_seconds += t_loss.elapsed().as_secs_f64();
                    step_batches.push(batch_stats_from_outputs(
                        chunk.len(),
                        output.policy_logits.clone(),
                        targets.legal_mask.clone(),
                        batch.actions.clone(),
                        total.clone(),
                        &breakdown,
                    ));
                    let chunk_weight = chunk.len() as f32 / logical_batch_len;
                    let t_backward = Instant::now();
                    let grads = (total * chunk_weight).backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    accumulator.accumulate(&model, grads);
                    sub_stage_timing.backward_seconds += t_backward.elapsed().as_secs_f64();
                }

                if !step_batches.is_empty() {
                    let lr = effective_lr(
                        trainer_schedule(&train_cfg),
                        completed_steps,
                        target_steps.max(1),
                    );
                    let grads = accumulator.grads();
                    let t_optimizer = Instant::now();
                    model = optimizer.step(lr, model, grads);
                    head_controller.tick_warmup();
                    sub_stage_timing.optimizer_step_seconds += t_optimizer.elapsed().as_secs_f64();
                }
            }

            let next_completed_steps = completed_steps + 1;
            if next_completed_steps == warmup_steps {
                measure_start = Some(Instant::now());
            } else if next_completed_steps > warmup_steps {
                for batch_stats in step_batches {
                    measured_stats.record_batch(batch_stats);
                }
            }
            completed_steps = next_completed_steps;

            if completed_steps >= target_steps {
                let elapsed_seconds = measure_start
                    .map(|start| start.elapsed().as_secs_f64())
                    .unwrap_or_default();
                return Ok(TrainBenchmarkOutcome {
                    model,
                    optimizer,
                    head_controller,
                    stats: measured_stats.finalize(),
                    elapsed_seconds,
                    measured_samples: measured_train_steps.saturating_mul(config.batch_size),
                    materialization_stats,
                    sub_stage_timing,
                    model_init_seconds,
                    optimizer_init_seconds,
                    loss_init_seconds,
                });
            }
        }
    }

    Err("not enough train data to finish stage-2 benchmark train window".to_string())
}
fn execute_benchmark_validation_pass<RunValidation>(
    materialization_seconds: f64,
    run_validation_pass: RunValidation,
) -> Result<(ValidationSummary, f64), String>
where
    RunValidation: FnOnce() -> Result<ValidationSummary, String>,
{
    let started = Instant::now();
    let summary = run_validation_pass()?;
    Ok((
        summary,
        started.elapsed().as_secs_f64() + materialization_seconds,
    ))
}

fn benchmark_validation_pass<B>(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    outcome: &mut TrainBenchmarkOutcome<B>,
    cached_samples: Option<&[Box<[MjaiSample]>]>,
    materialization_seconds: f64,
) -> Result<(ValidationSummary, f64), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let valid_loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let loader = benchmark_loader_config(
        config,
        LoaderRuntimeConfig {
            num_threads: config.num_threads,
            buffer_games: config.buffer_games,
            buffer_samples: config.buffer_samples,
            archive_queue_bound: config.archive_queue_bound,
        },
    );
    execute_benchmark_validation_pass(materialization_seconds, || {
        run_validation(
            &outcome.model,
            ValidationContext {
                config,
                loader: &validation_loader(&loader),
                manifest,
                cached_samples,
                device: train_device,
                loss_fn: &valid_loss_fn,
                exit_cfg: &exit_cfg,
            },
            ValidationRuntime {
                head_controller: Some(&mut outcome.head_controller),
                progress: None,
            },
        )
    })
}

fn benchmark_checkpoint_cost<B>(
    artifacts: &BcArtifactPaths,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    outcome: &TrainBenchmarkOutcome<B>,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    let continuation = EpochContinuation {
        next_epoch: 0,
        skip_optimizer_steps_in_epoch: 0,
        epoch_completed: false,
    };
    let started = Instant::now();
    save_latest_checkpoint_and_state(
        artifacts,
        &outcome.model,
        &outcome.optimizer,
        LatestCheckpointState {
            global_step: preflight.real_benchmark_train_steps.max(1),
            train_loss: outcome.stats.total_loss,
            best_validation: None,
            continuation: &continuation,
            runtime: runtime_resume_contract(config),
        },
    )?;
    Ok(started.elapsed().as_secs_f64())
}

fn benchmark_logging_cost(
    artifacts: &BcArtifactPaths,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    train_stats: &ScalarAverages,
    validation_summary: &ValidationSummary,
) -> Result<f64, String> {
    let global_step = preflight.real_benchmark_train_steps.max(1);
    let lr = effective_lr(
        trainer_schedule(&trainer_config_from_train_config(config)),
        global_step,
        (preflight.real_benchmark_warmup_steps + preflight.real_benchmark_train_steps).max(1),
    );
    let best_validation = Some(BestValidation {
        policy_loss: validation_summary.policy_loss,
        agreement: validation_summary.agreement,
    });
    let step_entry = StepLogEntry {
        global_step,
        epoch: 1,
        lr,
        train_total_loss: train_stats.total_loss,
        train_policy_agreement: train_stats.policy_agreement,
        train_loss_policy: train_stats.loss_policy,
        train_loss_value: train_stats.loss_value,
        train_loss_grp: train_stats.loss_grp,
        train_loss_tenpai: train_stats.loss_tenpai,
        train_loss_danger: train_stats.loss_danger,
        train_loss_opp_next: train_stats.loss_opp_next,
        train_loss_score_pdf: train_stats.loss_score_pdf,
        train_loss_score_cdf: train_stats.loss_score_cdf,
        train_rare_actions: train_stats.rare_actions,
        val_rare_actions: Some(validation_summary.rare_actions),
        val_total_loss: Some(validation_summary.total_loss),
        val_policy_loss: Some(validation_summary.policy_loss),
        val_policy_agreement: Some(validation_summary.agreement),
        val_delta_q_promotion: validation_summary.delta_q_promotion_snapshot,
        profiling: None,
        advisories: Vec::new(),
        best_val_policy_loss: best_validation.map(|value| value.policy_loss),
        best_val_agreement: best_validation.map(|value| value.agreement),
    };
    let started = Instant::now();
    append_step_log(&artifacts.step_log_path, &step_entry)?;
    if config.tensorboard {
        artifacts.create_tensorboard_dirs()?;
        let mut tb = EventWriter::create(&artifacts.tb_session_dir)
            .map_err(|err| format!("preflight benchmark tensorboard init: {err}"))?;
        log_tensorboard(
            &mut tb,
            global_step,
            train_stats,
            Some(validation_summary),
            lr,
            best_validation,
        )?;
    }
    Ok(started.elapsed().as_secs_f64())
}

fn run_stage_two_finalist_benchmark(
    preflight: &PreflightConfig,
    context: StageTwoBenchmarkContext<'_>,
) -> Result<BenchmarkResult, String> {
    let StageTwoBenchmarkContext {
        config,
        manifest,
        train_device,
        artifacts,
        finalists,
        train_candidates,
        validation_candidates,
        loader_candidates,
    } = context;
    let benchmark_paths = PreflightBenchmarkPaths::new(artifacts);
    benchmark_paths.create_root_dir()?;
    let initial_count = finalists
        .len()
        .min(preflight.real_benchmark_max_finalists.max(1));
    let mut finalists_to_benchmark = finalists[..initial_count].to_vec();
    let mut benchmarked = 0usize;
    let mut best: Option<BenchmarkResult> = None;
    let mut scored_results = Vec::new();
    let mut tie_expansion_triggered = false;
    let mut candidate_index = 0usize;
    let mut validation_cache = StageTwoBenchmarkValidationCache::new(config, preflight, finalists);
    while candidate_index < finalists_to_benchmark.len() {
        let finalist = &finalists_to_benchmark[candidate_index];
        let benchmark_config = benchmark_config_for_finalist(config, preflight, finalist);
        let validation_cache_key =
            stage_two_benchmark_validation_cache_key(&benchmark_config, finalist.runtime.loader);
        let (
            cached_validation_samples,
            validation_materialization_seconds,
            validation_materialization_stats,
        ) = validation_cache.checkout(validation_cache_key, &benchmark_config, manifest)?;
        let candidate_output_dir = benchmark_paths.create_candidate_dir(candidate_index)?;
        let candidate_artifacts = BcArtifactPaths::new(&candidate_output_dir, 0);
        candidate_artifacts.create_root_dir()?;
        let benchmark_run = StageTwoBenchmarkRunContext {
            config,
            benchmark_config: &benchmark_config,
            manifest,
            train_device,
            candidate_artifacts: &candidate_artifacts,
            finalist,
            train_candidates,
            validation_candidates,
            loader_candidates,
            benchmarked_count: benchmarked + 1,
            cached_validation_samples,
            validation_materialization_seconds,
            validation_materialization_stats,
        };
        let result = match config.precision_mode {
            hydra_train_runtime::config::PrecisionMode::Fp32 => {
                run_stage_two_benchmark_for_backend::<TrainBackend>(preflight, benchmark_run)?
            }
            hydra_train_runtime::config::PrecisionMode::Bf16Autocast => {
                run_stage_two_benchmark_for_backend::<TrainBackend>(preflight, benchmark_run)?
            }
        };
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight benchmark:",
                format!(
                    "candidate={} train_mb={} val_mb={} loader=({}, {}, {}, {:?}) requested_precision={} effective_precision={} effective={:.2} train_only={:.2}",
                    candidate_index + 1,
                    result.runtime.train_microbatch_size,
                    result.runtime.validation_microbatch_size,
                    result.runtime.loader.archive_queue_bound,
                    result.runtime.loader.buffer_samples,
                    result.runtime.loader.buffer_games,
                    result.runtime.loader.num_threads,
                    hydra_train_runtime::preflight::requested_precision_signature(
                        config.precision_mode
                    ),
                    config.effective_precision(),
                    result.score.wall_clock_samples_per_second,
                    result.score.train_only_samples_per_second,
                ),
            )
        );
        let replace_best = best.as_ref().is_none_or(|current| {
            result.score.wall_clock_samples_per_second > current.score.wall_clock_samples_per_second
        });
        if replace_best {
            best = Some(result.clone());
        }
        scored_results.push(result);
        benchmarked += 1;
        candidate_index += 1;
        if candidate_index == finalists_to_benchmark.len() {
            let mut best_score = f64::NEG_INFINITY;
            let mut next_score = f64::NEG_INFINITY;
            for result in &scored_results {
                let score = result.score.wall_clock_samples_per_second;
                if score > best_score {
                    next_score = best_score;
                    best_score = score;
                } else if score > next_score {
                    next_score = score;
                }
            }
            if scored_results.len() >= 2 {
                let tie_margin = preflight.real_benchmark_tie_margin_ratio.max(0.0);
                let threshold = best_score * (1.0 - tie_margin);
                if next_score >= threshold {
                    let current_len = finalists_to_benchmark.len();
                    let target_len = (current_len + preflight.real_benchmark_extra_finalists)
                        .min(finalists.len());
                    if target_len > current_len {
                        finalists_to_benchmark
                            .extend_from_slice(&finalists[current_len..target_len]);
                        tie_expansion_triggered = true;
                    }
                }
            }
        }
    }
    let mut best =
        best.ok_or_else(|| "stage-2 preflight benchmark had no finalists to score".to_string())?;
    best.metadata.finalists_benchmarked = benchmarked;
    if tie_expansion_triggered {
        best.metadata.selection_metric.push_str(" + tie_expansion");
    }
    Ok(best)
}

fn run_stage_two_benchmark_for_backend<B>(
    preflight: &PreflightConfig,
    context: StageTwoBenchmarkRunContext<'_>,
) -> Result<BenchmarkResult, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let StageTwoBenchmarkRunContext {
        config,
        benchmark_config,
        manifest,
        train_device,
        candidate_artifacts,
        finalist,
        train_candidates,
        validation_candidates,
        loader_candidates,
        benchmarked_count,
        cached_validation_samples,
        validation_materialization_seconds,
        validation_materialization_stats,
    } = context;
    let (
        train_outcome,
        (validation_summary, validation_seconds),
        checkpoint_seconds,
        logging_seconds,
    ) = run_stage_two_benchmark_scopes(
        || {
            benchmark_train_window_for_backend::<B>(
                benchmark_config,
                preflight,
                &HydraModelConfig::learner(),
                manifest,
                train_device,
            )
        },
        |train_outcome| {
            benchmark_validation_pass(
                benchmark_config,
                manifest,
                train_device,
                train_outcome,
                cached_validation_samples.as_deref(),
                validation_materialization_seconds,
            )
        },
        |train_outcome, _| {
            benchmark_checkpoint_cost(
                candidate_artifacts,
                benchmark_config,
                preflight,
                train_outcome,
            )
        },
        |train_outcome, (validation_summary, _)| {
            benchmark_logging_cost(
                candidate_artifacts,
                benchmark_config,
                preflight,
                &train_outcome.stats,
                validation_summary,
            )
        },
    )?;

    let evaluation = benchmark_score(
        benchmark_config,
        preflight,
        &ProfilingEnvelope::from_children(
            PROFILING_STAGE_STAGE_2_BENCHMARK,
            vec![
                {
                    let mut train_profile = train_substage_profile(
                        PROFILING_STAGE_TRAIN,
                        train_outcome.elapsed_seconds,
                        train_outcome.sub_stage_timing,
                    );
                    train_profile.children.push(ProfilingEnvelope::leaf(
                        PROFILING_STAGE_PREFLIGHT_MODEL_INIT,
                        train_outcome.model_init_seconds,
                    ));
                    train_profile.children.push(ProfilingEnvelope::leaf(
                        PROFILING_STAGE_PREFLIGHT_OPTIMIZER_INIT,
                        train_outcome.optimizer_init_seconds,
                    ));
                    train_profile.children.push(ProfilingEnvelope::leaf(
                        PROFILING_STAGE_PREFLIGHT_LOSS_INIT,
                        train_outcome.loss_init_seconds,
                    ));
                    train_profile.children.push(replay_materialization_profile(
                        train_outcome.materialization_stats,
                    ));
                    train_profile.children.push(ProfilingEnvelope::leaf(
                        "gpu_train_samples_per_second",
                        crate::system_metrics::rate_per_second(
                            train_outcome.measured_samples as u64,
                            train_outcome.elapsed_seconds,
                        )
                        .unwrap_or_default(),
                    ));
                    train_profile
                },
                validation_summary
                    .profiling
                    .clone()
                    .map(|mut profiling| {
                        profiling.elapsed_seconds = validation_seconds;
                        profiling.children.push(ProfilingEnvelope::leaf(
                            "materialization_samples_per_second",
                            crate::system_metrics::rate_per_second(
                                validation_summary.samples as u64,
                                validation_materialization_seconds,
                            )
                            .unwrap_or_default(),
                        ));
                        profiling.children.push(replay_materialization_profile(
                            validation_materialization_stats,
                        ));
                        profiling
                    })
                    .unwrap_or_else(|| {
                        ProfilingEnvelope::nested(
                            PROFILING_STAGE_VALIDATION,
                            validation_seconds,
                            vec![
                                ProfilingEnvelope::leaf(
                                    "materialization_samples_per_second",
                                    crate::system_metrics::rate_per_second(
                                        validation_summary.samples as u64,
                                        validation_materialization_seconds,
                                    )
                                    .unwrap_or_default(),
                                ),
                                replay_materialization_profile(validation_materialization_stats),
                            ],
                        )
                    }),
                ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, checkpoint_seconds),
                ProfilingEnvelope::leaf(PROFILING_STAGE_LOGGING, logging_seconds),
            ],
        ),
        validation_summary.samples,
    );

    Ok(BenchmarkResult {
        runtime: finalist.runtime,
        score: evaluation.score,
        metadata: benchmark_metadata(
            config,
            preflight,
            train_candidates,
            validation_candidates,
            loader_candidates,
            benchmarked_count,
        ),
        profiling: Some(evaluation.profiling),
    })
}

fn search_rl_runtime_candidate(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &RlArtifactPaths,
    kind: ProbeKind,
    seed: usize,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let explicit_candidate = match kind {
        ProbeKind::RlGames => config.rl.as_ref().map(|rl| rl.games_per_batch),
        ProbeKind::RlMicrobatch => config.rl.as_ref().and_then(|rl| rl.microbatch_size),
        ProbeKind::Train | ProbeKind::Validation => {
            return Err("non-RL probe kind passed to RL runtime search".to_string());
        }
    };
    let result = search_probe_candidate_ladder(
        config_path,
        config,
        preflight,
        ProbeSearchSpec::new(kind, seed.max(1))
            .with_explicit_candidate(explicit_candidate)
            .with_allow_explicit_override(!matches!(kind, ProbeKind::RlGames))
            .with_no_stable_error(format!(
                "no stable {} candidate found in preflight",
                probe_kind_name(kind)
            )),
        |kind, candidate, attempt| rl_probe_result_path(artifacts, kind, candidate, attempt),
    )?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            kind,
            &result.selected_summary,
        ))
    );
    Ok((result.selected_summary.candidate_microbatch, result.results))
}

struct ProbeSearchSpec {
    kind: ProbeKind,
    seed: usize,
    explicit_candidate: Option<usize>,
    allow_explicit_override: bool,
    use_fast_repeated_run_ladder: bool,
    no_stable_error: String,
}

impl ProbeSearchSpec {
    fn new(kind: ProbeKind, seed: usize) -> Self {
        Self {
            kind,
            seed,
            explicit_candidate: None,
            allow_explicit_override: true,
            no_stable_error: String::new(),
            use_fast_repeated_run_ladder: false,
        }
    }

    fn with_explicit_candidate(mut self, explicit_candidate: Option<usize>) -> Self {
        self.explicit_candidate = explicit_candidate;
        self
    }

    fn with_allow_explicit_override(mut self, allow_explicit_override: bool) -> Self {
        self.allow_explicit_override = allow_explicit_override;
        self
    }

    fn with_fast_repeated_run_ladder(mut self, enabled: bool) -> Self {
        self.use_fast_repeated_run_ladder = enabled;
        self
    }

    fn with_no_stable_error(mut self, no_stable_error: impl Into<String>) -> Self {
        self.no_stable_error = no_stable_error.into();
        self
    }
}

struct ProbeSearchOutcome {
    selected_summary: ProbeCandidateSummary,
    results: Vec<ProbeResult>,
}
const UNSUPPORTED_DEVICE_PREFIX: &str = "unsupported HYDRA_TRAIN_DEVICE=";

fn fatal_probe_backend_error(results: &[ProbeResult]) -> Option<String> {
    results
        .iter()
        .find(|result| {
            result.status == ProbeStatus::BackendError
                && result.detail.contains(UNSUPPORTED_DEVICE_PREFIX)
        })
        .and_then(|result| {
            result
                .detail
                .rsplit_once("detail=")
                .map(|(_, detail)| detail)
                .map(str::trim)
                .filter(|detail| detail.starts_with(UNSUPPORTED_DEVICE_PREFIX))
                .map(str::to_string)
        })
}

fn prune_oom_upper_bound(candidates: &mut Vec<usize>, oom_candidate: usize) {
    candidates.retain(|candidate| *candidate <= oom_candidate);
}

fn search_probe_candidate_ladder<F>(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    spec: ProbeSearchSpec,
    mut result_path_for: F,
) -> Result<ProbeSearchOutcome, String>
where
    F: Fn(ProbeKind, usize, usize) -> PathBuf + Copy,
{
    let use_explicit_only = spec.explicit_candidate.is_some()
        && !preflight.allow_override_explicit_microbatch
        && spec.allow_explicit_override;
    let mut candidates = if use_explicit_only {
        vec![spec.explicit_candidate.unwrap_or(spec.seed.max(1)).max(1)]
    } else if spec.use_fast_repeated_run_ladder {
        fast_repeated_run_ladder(preflight, config.batch_size, spec.seed.max(1))
    } else {
        dynamic_probe_ladder(config, preflight, spec.kind, spec.seed.max(1))
    };
    if !use_explicit_only && matches!(spec.kind, ProbeKind::Train) {
        prioritize_full_batch_train_candidate(&mut candidates, config.batch_size, spec.seed);
    }
    let mut seen = BTreeSet::new();
    candidates.retain(|candidate| seen.insert(*candidate));
    let plan = probe_search_plan(candidates.len(), preflight);
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind={} candidates={:?} planned_candidates={} planned_attempts={} max_growth_candidates={} max_planned_attempts={} required_successes={} growth_patience={} growth_max_steps={}",
                probe_kind_name(spec.kind),
                candidates,
                plan.initial_candidates,
                plan.planned_attempts,
                plan.max_growth_candidates,
                plan.max_planned_attempts,
                plan.required_successes,
                preflight.validation_growth_patience.max(1),
                preflight.validation_growth_max_steps.max(1)
            )
        )
    );
    let progress = make_bar(
        (candidates.len() * preflight.required_successes.max(1)) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut results = Vec::new();
    let mut growth_patience = 0usize;
    let mut growth_steps = 0usize;
    let tolerance = preflight.measure_noise_tolerance_ratio;
    let mut prior_best_score: Option<f64> = None;
    let mut last_successful_candidate: Option<usize> = None;
    let mut stop_reason = ProbeSearchStopReason::ExhaustedCandidates;

    let mut index = 0usize;
    while index < candidates.len() {
        let candidate = candidates[index];
        if let Some(blocked) = maybe_block_host_ram_growth_probe(
            preflight,
            spec.kind,
            candidate,
            last_successful_candidate,
        ) {
            println!("{}", format_probe_status_line(&blocked));
            results.push(blocked);
            stop_reason = ProbeSearchStopReason::HostRamGuard;
            break;
        }
        let passed = run_candidate_attempts(
            config_path,
            &mut result_path_for,
            ProbeRunSpec {
                kind: spec.kind,
                candidate,
                attempts: preflight.required_successes.max(1),
                warmup_steps: preflight.warmup_steps,
                measure_steps: preflight.measure_steps,
            },
            &mut results,
            &progress,
        )?;
        if !passed {
            if let Some(err) = fatal_probe_backend_error(&results) {
                progress.finish_with_message(
                    format!("preflight {} ladder failed", probe_kind_name(spec.kind))
                        .red()
                        .to_string(),
                );
                return Err(err);
            }
            let next_index = if results
                .last()
                .is_some_and(|result| result.status == ProbeStatus::Oom)
            {
                prune_oom_upper_bound(&mut candidates, candidate);
                Some(adaptive_oom_probe_next_index(
                    &candidates,
                    &results,
                    index + 1,
                ))
            } else {
                None
            };
            if use_explicit_only {
                progress.finish_with_message(
                    format!("preflight {} ladder complete", probe_kind_name(spec.kind))
                        .green()
                        .to_string(),
                );
                return Err(format!(
                    "explicit {} candidate {} failed preflight",
                    probe_kind_name(spec.kind),
                    candidate
                ));
            }
            stop_reason = ProbeSearchStopReason::FirstFailedCandidate;
            if let Some(next_index) = next_index {
                index = next_index;
                continue;
            }
            break;
        }
        last_successful_candidate = Some(candidate);
        if use_explicit_only {
            progress.finish_with_message(
                format!("preflight {} ladder complete", probe_kind_name(spec.kind))
                    .green()
                    .to_string(),
            );
            let selected_summary =
                best_probe_summary(&results).ok_or_else(|| spec.no_stable_error.clone())?;
            return Ok(ProbeSearchOutcome {
                selected_summary,
                results,
            });
        }

        if !preflight.fast_repeated_run_profile {
            let summary =
                best_probe_summary(&results).ok_or_else(|| spec.no_stable_error.clone())?;
            let candidate_score = candidate_average(&results, candidate).unwrap_or(0.0);
            let mut growth_state = ProbeGrowthState {
                patience: growth_patience,
                steps: growth_steps,
                prior_best_score,
            };
            if let Some(reason) = maybe_expand_probe_candidates(
                &mut candidates,
                ProbeGrowthDecision {
                    index,
                    kind: spec.kind,
                    candidate,
                    summary: &summary,
                    candidate_score,
                    tolerance,
                },
                config,
                preflight,
                &mut growth_state,
            ) {
                stop_reason = reason;
                break;
            }
            growth_patience = growth_state.patience;
            growth_steps = growth_state.steps;
            prior_best_score = growth_state.prior_best_score;
        }
        index = adaptive_oom_probe_next_index(&candidates, &results, index + 1);
    }

    if let Some(err) = fatal_probe_backend_error(&results) {
        return Err(err);
    }
    progress.finish_with_message(
        format!("preflight {} ladder complete", probe_kind_name(spec.kind))
            .green()
            .to_string(),
    );
    let selected_summary = finalize_probe_search(
        config_path,
        result_path_for,
        spec.kind,
        config,
        preflight,
        &mut results,
        &progress,
        spec.no_stable_error,
    )?;
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder stop:",
            format!(
                "kind={} reason={}",
                probe_kind_name(spec.kind),
                stop_reason.as_str()
            ),
        )
    );
    Ok(ProbeSearchOutcome {
        selected_summary,
        results,
    })
}

fn maybe_block_host_ram_growth_probe(
    preflight: &PreflightConfig,
    kind: ProbeKind,
    candidate: usize,
    baseline_candidate: Option<usize>,
) -> Option<ProbeResult> {
    let baseline = baseline_candidate?;
    if candidate <= baseline {
        return None;
    }
    crate::system_metrics::emit_system_metrics_event(
        &crate::system_metrics::probe_host_memory_event(
            kind,
            candidate,
            mem_available_bytes(),
            crate::probe_process::mem_total_bytes(),
        ),
    );
    let available = mem_available_bytes()?;
    let required_free = rl_probe_required_free_bytes(preflight)?;
    let scale = candidate as f64 / baseline as f64;
    let estimated_probe_bytes =
        ((available as f64) * scale * preflight.rl_probe_growth_safety_factor.max(1.0)).ceil()
            as u64;
    let remaining_after_probe = available.saturating_sub(estimated_probe_bytes);
    if remaining_after_probe >= required_free {
        return None;
    }
    Some(ProbeResult {
        kind,
        candidate_microbatch: candidate,
        status: ProbeStatus::BackendError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: format!(
            "probe blocked by host-RAM guard: available={} estimated_probe={} remaining_after_probe={} required_free={} baseline_candidate={} growth_safety_factor={:.2}",
            available,
            estimated_probe_bytes,
            remaining_after_probe,
            required_free,
            baseline,
            preflight.rl_probe_growth_safety_factor.max(1.0),
        ),
    })
}

pub fn measure_samples_per_second(samples: usize, elapsed: Duration) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        0.0
    } else {
        samples as f64 / seconds
    }
}

fn probe_train_candidate_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    run_train_measurement_loop::<B>(TrainMeasurementSpec {
        config,
        model_config,
        candidate_microbatch: request.candidate_microbatch,
        warmup_steps: request.warmup_steps,
        measure_steps: request.measure_steps,
        loader_config,
        manifest,
        train_device,
        on_start: Box::new(|candidate_microbatch, warmup_steps, measure_steps| {
            emit_probe_progress(&format!(
                "probe_progress kind=train candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
                candidate_microbatch, warmup_steps, measure_steps
            ))
        }),
        on_step: Box::new(
            |completed_steps, candidate_microbatch, request, measure_start| {
                emit_probe_step_progress(
                    ProbeKind::Train,
                    candidate_microbatch,
                    completed_steps,
                    ProbeRequest {
                        kind: ProbeKind::Train,
                        candidate_microbatch,
                        warmup_steps: request.warmup_steps,
                        measure_steps: request.measure_steps,
                    },
                    measure_start,
                    config.batch_size,
                )
            },
        ),
        on_measure_start: Box::new(|candidate_microbatch, measure_steps| {
            emit_probe_progress(&format!(
                "probe_progress kind=train candidate_mb={} phase=measure_start total_steps={}",
                candidate_microbatch,
                measure_steps.max(1)
            ))
        }),
        insufficient_data: Box::new(|candidate_microbatch| {
            format!(
                "not enough train data to finish preflight probe at microbatch {}",
                candidate_microbatch
            )
        }),
    })
}

type TrainMeasurementStepCallback<'a> =
    dyn FnMut(usize, usize, ProbeRequest, Option<Instant>) -> Result<(), String> + 'a;

pub struct TrainMeasurementSpec<'a> {
    pub config: &'a TrainConfig,
    pub model_config: &'a HydraModelConfig,
    pub candidate_microbatch: usize,
    pub warmup_steps: usize,
    pub measure_steps: usize,
    pub loader_config: &'a StreamingLoaderConfig,
    pub manifest: &'a DataManifest,
    pub train_device: &'a LibTorchDevice,
    pub on_start: Box<dyn FnMut(usize, usize, usize) -> Result<(), String> + 'a>,
    pub on_step: Box<TrainMeasurementStepCallback<'a>>,
    pub on_measure_start: Box<dyn FnMut(usize, usize) -> Result<(), String> + 'a>,
    pub insufficient_data: Box<dyn FnOnce(usize) -> String + 'a>,
}

pub fn run_train_measurement_loop<B>(spec: TrainMeasurementSpec<'_>) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    let TrainMeasurementSpec {
        config,
        model_config,
        candidate_microbatch,
        warmup_steps,
        measure_steps,
        loader_config,
        manifest,
        train_device,
        mut on_start,
        mut on_step,
        mut on_measure_start,
        insufficient_data,
    } = spec;
    let train_cfg = trainer_config_from_train_config(config);

    emit_probe_init_phase("train", candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let mut model = model_config.init::<B>(train_device);
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", candidate_microbatch, "init_optimizer")?;
    let t0 = Instant::now();
    let mut optimizer: BenchmarkOptimizerOf<B> = train_cfg.optimizer_config().init();
    let optimizer_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Train,
        "train",
        candidate_microbatch,
        model_ms,
        optimizer_ms,
        loss_ms,
    )?;

    let microbatch_size = candidate_microbatch.min(config.batch_size).max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut probe_state = ProbeLoopState::new();
    let mut pending_samples = std::collections::VecDeque::new();
    on_start(microbatch_size, warmup_steps, measure_steps)?;

    for buffer_result in stream_train_epoch(manifest, loader_config, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight train stream failed: {err}"))?;
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let lr = effective_lr(
                trainer_schedule(&train_cfg),
                probe_state.completed_steps,
                target_steps.max(1),
            );
            let grads = if let Some(grads) = probe_train_fixed_chunks(FixedShapeProbeConfig {
                logical_batch: &logical_batch,
                augment: config.augment,
                microbatch_size,
                train_device,
                loss_fn: &loss_fn,
                model: &model,
                use_amp: false,
            })? {
                grads
            } else {
                let logical_batch_len = logical_batch.len().max(1) as f32;
                let mut accumulator: GradientsAccumulator<HydraModel<B>> =
                    GradientsAccumulator::new();
                for chunk in logical_batch.chunks(microbatch_size) {
                    let Some((obs, targets)) =
                        collate_samples::<B>(chunk, config.augment, train_device)
                            .map_err(|err| format!("preflight train collation failed: {err}"))?
                    else {
                        continue;
                    };
                    let output = model.forward(obs);
                    let breakdown = loss_fn.total_loss(&output, &targets);
                    let chunk_weight = chunk.len() as f32 / logical_batch_len;
                    let grads = (breakdown.total * chunk_weight).backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    accumulator.accumulate(&model, grads);
                }
                accumulator.grads()
            };
            model = optimizer.step(lr, model, grads);
            on_step(
                probe_state.completed_steps,
                microbatch_size,
                ProbeRequest {
                    kind: ProbeKind::Train,
                    candidate_microbatch: microbatch_size,
                    warmup_steps,
                    measure_steps,
                },
                probe_state.measure_start,
            )?;
            probe_state.completed_steps += 1;
            if probe_state.completed_steps == warmup_steps {
                probe_state.measure_start = Some(Instant::now());
                on_measure_start(microbatch_size, measure_steps)?;
            }
            if probe_state.completed_steps >= target_steps {
                let elapsed = probe_state
                    .measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    measure_steps.max(1) * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    Err(insufficient_data(microbatch_size))
}

fn probe_train_candidate_from_shards_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    train_device: &LibTorchDevice,
    reader: &ExtractedBcShardReader,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            FloatTensorPrimitive = burn::backend::libtorch::TchTensor,
            IntTensorPrimitive = burn::backend::libtorch::TchTensor,
        >,
{
    let train_cfg = trainer_config_from_train_config(config);

    emit_probe_init_phase("train", request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let mut model = Some(model_config.init::<B>(train_device));
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", request.candidate_microbatch, "init_optimizer")?;
    let t0 = Instant::now();
    let mut optimizer: BenchmarkOptimizerOf<B> = train_cfg.optimizer_config().init();
    let optimizer_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let mut head_controller = HeadActivationController::new(
        HeadActivationConfig::default_with_params(model_config.estimated_params()),
    );
    let loss_ms = t0.elapsed().as_millis();

    let microbatch_size = request.candidate_microbatch.min(config.batch_size).max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut probe_state = ProbeLoopState::new();
    let mut scratch = reader.new_scratch(config.batch_size);

    #[cfg(feature = "cuda-graph")]
    {
        emit_probe_init_phase("train", request.candidate_microbatch, "init_cuda_staging")?;
    }
    #[cfg(feature = "cuda-graph")]
    let mut staging_context = match train_device {
        LibTorchDevice::Cuda(idx) => {
            let device_index = *idx as i64;
            Some((
                PinnedStagingArea::new(config.batch_size),
                AsyncH2DContext::new(device_index),
                PreallocatedDeviceTensors::new(config.batch_size, train_device),
            ))
        }
        _ => None,
    };

    emit_probe_init_ready(
        ProbeKind::Train,
        "train",
        request.candidate_microbatch,
        model_ms,
        optimizer_ms,
        loss_ms,
    )?;

    emit_probe_start_progress(request, microbatch_size)?;

    let total_rows = reader.sample_count();
    let mut idx = 0usize;
    while idx < total_rows {
        let take = config.batch_size.min(total_rows - idx);
        if take < config.batch_size {
            break;
        }
        reader.collate_host_batch_range_into(idx, take, config.augment, &mut scratch)?;
        let host_batch = scratch.take_batch();
        let materialize_started = Instant::now();
        let shard_batch = {
            #[cfg(feature = "cuda-graph")]
            {
                if let Some((pinned_staging, h2d_ctx, gpu_tensors)) = staging_context.as_mut() {
                    crate::pinned_transfer::materialize_staged_reuse::<B>(
                        &host_batch,
                        pinned_staging,
                        h2d_ctx,
                        train_device,
                        gpu_tensors,
                    )
                    .0
                } else {
                    materialize_host_batch_owned::<B>(host_batch, train_device)
                }
            }
            #[cfg(not(feature = "cuda-graph"))]
            {
                materialize_host_batch_owned::<B>(host_batch, train_device)
            }
        };
        let mut timing = hydra_train_runtime::progress::TrainSubStageTiming::default();
        timing.h2d_tensor_materialize_seconds += materialize_started.elapsed().as_secs_f64();
        let lr = effective_lr(
            trainer_schedule(&train_cfg),
            probe_state.completed_steps,
            target_steps.max(1),
        );
        let _ = train_device_batch(
            shard_batch,
            config.batch_size,
            timing,
            TrainLogicalBatchConfig {
                microbatch_size,
                augment: config.augment,
                train_device,
                loss_fn: &loss_fn,
                bc_exit_cfg: &exit_cfg,
                lr,
                use_amp: false,
            },
            &mut head_controller,
            &mut model,
            &mut optimizer,
        )?;
        if let Some(throughput) = advance_probe_loop(
            &mut probe_state,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: microbatch_size,
                warmup_steps: request.warmup_steps,
                measure_steps: request.measure_steps,
            },
            microbatch_size,
            config.batch_size,
        )? {
            return Ok(throughput);
        }
        idx += take;
    }

    Err(format!(
        "not enough train shard data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

fn probe_validation_candidate_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    emit_probe_init_phase("validation", request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let model = model_config.init::<B>(train_device);
    let model_valid = model.valid();
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("validation", request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Validation,
        "validation",
        request.candidate_microbatch,
        model_ms,
        0,
        loss_ms,
    )?;

    let microbatch_size = request.candidate_microbatch.max(1);
    let mut probe_state = ProbeLoopState::new();
    emit_probe_start_progress(request, microbatch_size)?;

    for microbatch_result in stream_val_microbatches(manifest, loader_config, microbatch_size, None)
    {
        let microbatch = microbatch_result
            .map_err(|err| format!("preflight validation stream failed: {err}"))?;
        let Some(collated) = collate_samples_bc_owned_timed::<ValidBackendOf<B>>(
            microbatch.as_slice(),
            false,
            train_device,
        )
        .map_err(|err| format!("preflight validation collation failed: {err}"))?
        else {
            continue;
        };
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let output = model_valid.forward(obs);
        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = crate::bc_runtime::maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            &crate::bc_runtime::BcExitConfig::default(),
        );
        let _ = batch_stats_from_outputs(
            microbatch.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        );
        if let Some(throughput) =
            advance_probe_loop(&mut probe_state, request, microbatch_size, microbatch_size)?
        {
            return Ok(throughput);
        }
    }

    Err(format!(
        "not enough validation data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}
fn execute_shard_validation_probe<RunValidation>(
    config: &TrainConfig,
    _request: ProbeRequest,
    sample_count: usize,
    started_at: Instant,
    run_validation_probe: RunValidation,
) -> Result<f64, String>
where
    RunValidation: FnOnce() -> Result<ValidationSummary, String>,
{
    let _summary = run_validation_probe()?;
    Ok(measure_samples_per_second(
        ValidationRunLimits::from_config(config).bounded_total_rows(sample_count),
        started_at.elapsed(),
    ))
}

fn probe_validation_candidate_from_shards_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    _request: ProbeRequest,
    train_device: &LibTorchDevice,
    reader: &ExtractedBcShardReader,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    emit_probe_init_phase("validation", _request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let model = model_config.init::<B>(train_device);
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("validation", _request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Validation,
        "validation",
        _request.candidate_microbatch,
        model_ms,
        0,
        loss_ms,
    )?;

    execute_shard_validation_probe(
        config,
        _request,
        reader.sample_count(),
        Instant::now(),
        || {
            run_validation(
                &model,
                ValidationContext {
                    config,
                    loader: &validation_loader(&StreamingLoaderConfig {
                        buffer_games: config.buffer_games,
                        buffer_samples: config.buffer_samples,
                        train_fraction: config.train_fraction,
                        seed: config.seed,
                        archive_queue_bound: config.archive_queue_bound,
                        max_skip_logs_per_source: config.max_skip_logs_per_source,
                        aggregate_skip_logs: true,
                        source_filters: config.source_filters.clone(),
                        replay_target_profile: ReplayTargetProfile::minimal_bc(),
                        exit_sidecar: None,
                        exit_sidecar_source_net_hash: None,
                        exit_sidecar_source_version: None,
                        delta_q_sidecar: None,
                        delta_q_sidecar_source_net_hash: None,
                        delta_q_sidecar_source_version: None,
                        num_threads: config.num_threads,
                    }),
                    manifest: &DataManifest {
                        sources: Vec::new(),
                        total_games: 0,
                        train_count: 0,
                        val_count: 0,
                        counts_exact: true,
                    },
                    cached_samples: None,
                    device: train_device,
                    loss_fn: &loss_fn,
                    exit_cfg: &build_bc_exit_config(config.advanced_loss.as_ref()),
                },
                ValidationRuntime {
                    head_controller: None,
                    progress: None,
                },
            )
        },
    )
}

fn run_rl_probe_only(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    let result = run_rl_probe_only_result(config, preflight, request)?;
    write_probe_result(result_path, &result)
}

fn build_probe_success_result(
    request: ProbeRequest,
    measured_samples_per_second: f64,
    elapsed_seconds: f64,
    detail: String,
) -> ProbeResult {
    ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(measured_samples_per_second),
        elapsed_seconds: Some(elapsed_seconds),
        detail,
    }
}

fn configure_probe_threads(config: &TrainConfig) -> Result<(), String> {
    configure_threads(config.num_threads)
        .map_err(|err| format!("failed to configure rayon threads for probe child: {err}"))
}

fn run_probe_attempt_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    if matches!(request.kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        run_rl_probe_only_result(config, preflight, request)
    } else {
        run_probe_only_with_model_config_result(config, model_config, manifest, request)
    }
}

struct ProbeDataContext {
    manifest: Option<DataManifest>,
    train_reader: Option<ExtractedBcShardReader>,
    validation_reader: Option<ExtractedBcShardReader>,
}

impl ProbeDataContext {
    fn manifest_ref(&self) -> Option<&DataManifest> {
        self.manifest.as_ref()
    }

    fn train_reader_ref(&self) -> Option<&ExtractedBcShardReader> {
        self.train_reader.as_ref()
    }

    fn validation_reader_ref(&self) -> Option<&ExtractedBcShardReader> {
        self.validation_reader.as_ref()
    }
}

fn resolve_probe_data_context(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<ProbeDataContext, String> {
    if matches!(kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return Ok(ProbeDataContext {
            manifest: None,
            train_reader: None,
            validation_reader: None,
        });
    }

    if config.bc_shards_manifest_path.is_some()
        && matches!(kind, ProbeKind::Train | ProbeKind::Validation)
    {
        let shard_manifest_path = config
            .bc_shards_manifest_path
            .as_ref()
            .ok_or_else(|| "bc_shards_manifest_path missing for shard probe".to_string())?;
        let train_reader = if matches!(kind, ProbeKind::Train) {
            Some(load_extracted_bc_shard_reader(
                shard_manifest_path,
                ExtractedBcShardSplit::Train,
            )?)
        } else {
            None
        };
        let validation_reader = if matches!(kind, ProbeKind::Validation) {
            Some(load_extracted_bc_shard_reader(
                shard_manifest_path,
                ExtractedBcShardSplit::Validation,
            )?)
        } else {
            None
        };
        return Ok(ProbeDataContext {
            manifest: None,
            train_reader,
            validation_reader,
        });
    }

    Ok(ProbeDataContext {
        manifest: load_probe_batch_manifest(
            config,
            kind,
            manifest_cache_path,
            discovery_summary_path,
            discovery_index_path,
        )?,
        train_reader: None,
        validation_reader: None,
    })
}

fn run_probe_attempt_with_data_context(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    context: &ProbeDataContext,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    if context.train_reader.is_some() || context.validation_reader.is_some() {
        run_probe_attempt_with_shard_readers_result(
            config,
            preflight,
            model_config,
            context.train_reader_ref(),
            context.validation_reader_ref(),
            request,
        )
    } else {
        run_probe_attempt_result(
            config,
            preflight,
            model_config,
            context.manifest_ref(),
            request,
        )
    }
}

fn run_probe_attempt_with_shard_readers_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    train_reader: Option<&ExtractedBcShardReader>,
    validation_reader: Option<&ExtractedBcShardReader>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    let train_device = train_device(&config.device)?;
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => match config.precision_mode {
            hydra_train_runtime::config::PrecisionMode::Fp32
            | hydra_train_runtime::config::PrecisionMode::Bf16Autocast => {
                probe_train_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    train_reader.ok_or_else(|| {
                        "train shard reader missing for shard train probe".to_string()
                    })?,
                )?
            }
        },
        ProbeKind::Validation => match config.precision_mode {
            hydra_train_runtime::config::PrecisionMode::Fp32
            | hydra_train_runtime::config::PrecisionMode::Bf16Autocast => {
                let reader = validation_reader.ok_or_else(|| {
                    "validation shard reader missing for shard validation probe".to_string()
                })?;
                probe_validation_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            }
        },
        ProbeKind::RlGames | ProbeKind::RlMicrobatch => {
            return run_rl_probe_only_result(config, preflight, request);
        }
    };
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        format!(
            "stable {} probe on shard dataset",
            probe_kind_name(request.kind)
        ),
    ))
}

fn load_probe_batch_manifest(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<Option<DataManifest>, String> {
    if config.bc_shards_manifest_path.is_some()
        && matches!(kind, ProbeKind::Train | ProbeKind::Validation)
    {
        return Ok(None);
    }
    if matches!(kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return Ok(None);
    }

    let paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
    let cache_path = manifest_cache_path.unwrap_or(&paths.manifest_cache_path);
    let discovery_summary_path = discovery_summary_path.unwrap_or(&paths.discovery_summary_path);
    let discovery_index_path = discovery_index_path.unwrap_or(&paths.discovery_index_path);
    let loaded = load_or_scan_manifest_cache_with_source(ManifestCacheRequest {
        cache_path,
        discovery_summary_path,
        discovery_index_path,
        data_dir: &config.data_dir,
        train_fraction: config.train_fraction,
        source_filters: &config.source_filters,
        progress: None,
        scan_error_context: "preflight data",
    })?;
    let source = loaded
        .hit_source
        .map(ManifestCacheHitSource::as_str)
        .unwrap_or("scan");
    emit_probe_progress(&format!(
        "probe_progress kind={} phase=manifest_load source={} discovery_summary_path={} discovery_index_path={} legacy_manifest_path={} sources={} total_games={} train_count={} val_count={} counts_exact={}",
        probe_kind_name(kind),
        source,
        discovery_summary_path.display(),
        discovery_index_path.display(),
        cache_path.display(),
        loaded.manifest.sources.len(),
        loaded.manifest.total_games,
        loaded.manifest.train_count,
        loaded.manifest.val_count,
        loaded.manifest.counts_exact,
    ))?;
    Ok(Some(loaded.manifest))
}

fn load_probe_child_manifest(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<Option<DataManifest>, String> {
    load_probe_batch_manifest(
        config,
        kind,
        manifest_cache_path,
        discovery_summary_path,
        discovery_index_path,
    )
}

fn run_probe_child_batch_request_with_model_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    batch: ProbeBatchRequest,
    results_path: &Path,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
    model_config: &HydraModelConfig,
) -> Result<ProbeBatchArtifact, String> {
    configure_probe_threads(config)?;
    std::fs::remove_file(results_path).ok();
    let context = resolve_probe_data_context(
        config,
        batch.request.kind,
        manifest_cache_path,
        discovery_summary_path,
        discovery_index_path,
    )?;
    let mut artifact = ProbeBatchArtifact::pending();

    for _attempt in 0..batch.attempts {
        let result = run_probe_attempt_with_data_context(
            config,
            preflight,
            model_config,
            &context,
            batch.request,
        )?;
        let passed = result.status == ProbeStatus::Success;
        artifact.push_result(result);
        write_probe_batch_artifact(results_path, &artifact)?;
        if !passed {
            return Ok(artifact);
        }
    }

    artifact.mark_finished();
    write_probe_batch_artifact(results_path, &artifact)?;
    Ok(artifact)
}

fn run_rl_probe_only_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    let train_device = train_device(&config.device)?;
    let rl = config
        .rl
        .as_ref()
        .ok_or_else(|| "RL probe requested without rl config block".to_string())?;
    let mut tuned = rl.clone();
    match request.kind {
        ProbeKind::RlGames => {
            tuned.games_per_batch = request.candidate_microbatch;
            if tuned.microbatch_size.is_none() {
                tuned.microbatch_size = Some(hydra_train_types::config::DEFAULT_RL_MICROBATCH_SIZE);
            }
        }
        ProbeKind::RlMicrobatch => {
            tuned.microbatch_size = Some(request.candidate_microbatch.max(1));
        }
        ProbeKind::Train | ProbeKind::Validation => {
            return Err("non-RL probe routed to RL probe handler".to_string());
        }
    }
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=rl_selfplay",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
    ))?;
    let started_at = Instant::now();
    let measured_samples_per_second = crate::runtime_autotune_shim::measure_rl_runtime_throughput(
        config,
        preflight,
        &tuned,
        &train_device,
    )?;
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=done throughput={:.2} samples/s elapsed={:.2}s",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        measured_samples_per_second,
        elapsed_seconds,
    ))?;
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        String::new(),
    ))
}

fn run_probe_only_with_model_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    configure_probe_threads(config)?;
    if matches!(request.kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return run_rl_probe_only(config, preflight, request, result_path);
    }
    let result = run_probe_only_with_model_config_result(config, model_config, manifest, request)?;
    write_probe_result(result_path, &result)
}

fn run_probe_only_with_model_config_result(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    debug_assert!(matches!(
        request.kind,
        ProbeKind::Train | ProbeKind::Validation
    ));

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
        source_filters: config.source_filters.clone(),
        replay_target_profile: ReplayTargetProfile::minimal_bc(),
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
        num_threads: config.num_threads,
    };
    let probe_context = if config.bc_shards_manifest_path.is_some() {
        resolve_probe_data_context(config, request.kind, None, None, None)?
    } else if let Some(manifest) = manifest.cloned() {
        ProbeDataContext {
            manifest: Some(manifest),
            train_reader: None,
            validation_reader: None,
        }
    } else {
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=scan_start data_dir={}",
            probe_kind_name(request.kind),
            request.candidate_microbatch,
            config.data_dir.display(),
        ))?;
        let paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
        let loaded = load_or_scan_manifest_cache_with_source(ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &config.data_dir,
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            progress: None,
            scan_error_context: "preflight data",
        })?;
        let source = loaded
            .hit_source
            .map(ManifestCacheHitSource::as_str)
            .unwrap_or("scan");
        let manifest = loaded.manifest;
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=scan_complete source={} discovery_summary_path={} discovery_index_path={} sources={} total_games={} train_count={} val_count={} counts_exact={}",
            probe_kind_name(request.kind),
            request.candidate_microbatch,
            source,
            paths.discovery_summary_path.display(),
            paths.discovery_index_path.display(),
            manifest.sources.len(),
            manifest.total_games,
            manifest.train_count,
            manifest.val_count,
            manifest.counts_exact,
        ))?;
        ProbeDataContext {
            manifest: Some(manifest),
            train_reader: None,
            validation_reader: None,
        }
    };
    let train_device = train_device(&config.device)?;
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => {
            if let Some(reader) = probe_context.train_reader_ref() {
                probe_train_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            } else {
                probe_train_candidate_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &loader_config,
                    probe_context
                        .manifest_ref()
                        .ok_or_else(|| "manifest missing for non-shard train probe".to_string())?,
                    &train_device,
                )?
            }
        }
        ProbeKind::Validation => {
            if let Some(reader) = probe_context.validation_reader_ref() {
                probe_validation_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            } else {
                probe_validation_candidate_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &loader_config,
                    probe_context.manifest_ref().ok_or_else(|| {
                        "manifest missing for non-shard validation probe".to_string()
                    })?,
                    &train_device,
                )?
            }
        }
        ProbeKind::RlGames | ProbeKind::RlMicrobatch => {
            unreachable!("RL probes handled by run_rl_probe_only")
        }
    };
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=done throughput={:.2} samples/s elapsed={:.2}s",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        measured_samples_per_second,
        elapsed_seconds,
    ))?;
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        format!(
            "stable {} probe on real dataset",
            probe_kind_name(request.kind)
        ),
    ))
}

#[cfg(test)]
pub fn run_probe_only(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    run_probe_only_with_model_config(
        config,
        preflight,
        &HydraModelConfig::learner(),
        None,
        request,
        result_path,
    )
}

pub fn run_probe_child_mode(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
) -> Result<bool, String> {
    run_probe_child_mode_with_model_config(config, child, &HydraModelConfig::learner())
}

pub fn run_probe_child_mode_with_model_config(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
    model_config: &HydraModelConfig,
) -> Result<bool, String> {
    if let Some((batch, results_path, manifest_paths)) =
        probe_batch_child_request_from_cli(child.clone())?
    {
        let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
        run_probe_child_batch_request_with_model_config(
            config,
            &PreflightConfig::default(),
            batch,
            &results_path,
            manifest_cache_path.as_deref(),
            discovery_summary_path.as_deref(),
            discovery_index_path.as_deref(),
            model_config,
        )?;
        return Ok(true);
    }

    let Some((request, result_path, manifest_paths)) = probe_child_request_from_cli(child)? else {
        return Ok(false);
    };
    let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
    let manifest = load_probe_child_manifest(
        config,
        request.kind,
        manifest_cache_path.as_deref(),
        discovery_summary_path.as_deref(),
        discovery_index_path.as_deref(),
    )?;
    run_probe_only_with_model_config(
        config,
        &PreflightConfig::default(),
        model_config,
        manifest.as_ref(),
        request,
        &result_path,
    )?;
    Ok(true)
}

#[cfg(test)]
fn run_candidate_attempts<F>(
    config_path: &Path,
    result_path_for: &mut F,
    spec: ProbeRunSpec,
    results: &mut Vec<ProbeResult>,
    progress: &ProgressBar,
) -> Result<bool, String>
where
    F: FnMut(ProbeKind, usize, usize) -> PathBuf,
{
    let config = read_config(config_path)?;
    let preflight = PreflightConfig::default();
    let attempts = spec.attempts.max(1);
    let request = ProbeRequest {
        kind: spec.kind,
        candidate_microbatch: spec.candidate,
        warmup_steps: spec.warmup_steps,
        measure_steps: spec.measure_steps,
    };
    let tiny = HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1);

    for attempt in 0..attempts {
        progress.set_message(format_probe_attempt_message(
            spec.kind,
            spec.candidate,
            attempt + 1,
            attempts,
        ));
        let result_path = result_path_for(spec.kind, spec.candidate, attempt);
        std::fs::remove_file(&result_path).ok();
        let result = match request.kind {
            ProbeKind::Train | ProbeKind::Validation => {
                run_probe_only_with_model_config_result(&config, &tiny, None, request)?
            }
            ProbeKind::RlGames | ProbeKind::RlMicrobatch => {
                run_rl_probe_only_result(&config, &preflight, request)?
            }
        };
        write_probe_result(&result_path, &result)?;
        let passed = result.status == ProbeStatus::Success;
        progress.inc(1);
        println!("{}", format_probe_status_line(&result));
        results.push(result);
        if !passed {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
pub fn run_probe_child_batch_mode_result(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
) -> Result<Option<ProbeBatchArtifact>, String> {
    let Some((batch, results_path, manifest_paths)) = probe_batch_child_request_from_cli(child)?
    else {
        return Ok(None);
    };
    let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
    let tiny = HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1);
    Ok(Some(run_probe_child_batch_request_with_model_config(
        config,
        &PreflightConfig::default(),
        batch,
        &results_path,
        manifest_cache_path.as_deref(),
        discovery_summary_path.as_deref(),
        discovery_index_path.as_deref(),
        &tiny,
    )?))
}

#[cfg(test)]
pub fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<ProbeResult, String> {
    let config = read_config(config_path)?;
    run_probe_only_with_model_config(
        &config,
        &PreflightConfig::default(),
        &HydraModelConfig::learner(),
        None,
        request,
        result_path,
    )?;
    crate::probe_transport::read_probe_result(result_path)
}

#[cfg(test)]
pub fn format_probe_attempt_message(
    kind: ProbeKind,
    candidate: usize,
    attempt: usize,
    total_attempts: usize,
) -> String {
    format!(
        "[preflight:{}] candidate_mb={} attempt {}/{}",
        probe_kind_name(kind),
        candidate,
        attempt,
        total_attempts.max(1)
    )
}

#[cfg(test)]
pub fn format_probe_result_summary(result: &ProbeResult) -> String {
    format_probe_status_line(result)
}

#[cfg(test)]
fn probe_candidate_ladder_with_local_executor(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    let explicit_candidate = match kind {
        ProbeKind::Train => config.microbatch_size,
        ProbeKind::Validation => config.validation_microbatch_size,
        ProbeKind::RlGames => config.rl.as_ref().map(|rl| rl.games_per_batch),
        ProbeKind::RlMicrobatch => config.rl.as_ref().and_then(|rl| rl.microbatch_size),
    };
    let spec = ProbeSearchSpec::new(kind, candidates.first().copied().unwrap_or(1))
        .with_explicit_candidate(explicit_candidate)
        .with_no_stable_error(format!(
            "no stable {} microbatch found in preflight",
            probe_kind_name(kind)
        ));
    let outcome = search_probe_candidate_ladder(
        config_path,
        config,
        preflight,
        spec,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
    )?;
    Ok((
        outcome.selected_summary.candidate_microbatch,
        outcome.results,
    ))
}

#[cfg(not(test))]
fn probe_candidate_ladder_with_local_executor(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    probe_candidate_ladder(config_path, config, preflight, artifacts, kind, candidates)
}

pub fn run_probe_ladder_only(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    preflight: &PreflightConfig,
    request: ProbeRequest,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let scan_pb = make_spinner("{spinner:.cyan} {msg}")?;
    scan_pb.set_message(format!(
        "scanning data for {} probe",
        probe_kind_name(request.kind)
    ));
    let paths = PreflightPaths::new(artifacts);
    let _ = load_or_scan_manifest_cache(
        ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &config.data_dir,
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            progress: Some(&scan_pb),
            scan_error_context: "preflight data",
        },
        |_| {},
    )?;
    scan_pb.finish_with_message(
        format!("scan complete for {} probe", probe_kind_name(request.kind))
            .green()
            .to_string(),
    );

    let candidates = probe_only_candidate_ladder(config, preflight, request);
    let selected = probe_candidate_ladder_with_local_executor(
        config_path,
        config,
        preflight,
        artifacts,
        request.kind,
        &candidates,
    )?;
    Ok(selected)
}

pub fn classify_probe_detail(detail: &str) -> ProbeStatus {
    let lowered = detail.to_ascii_lowercase();
    if lowered.contains("out of memory") || lowered.contains("oom") {
        ProbeStatus::Oom
    } else if lowered.contains("cuda") || lowered.contains("cudnn") || lowered.contains("libtorch")
    {
        ProbeStatus::BackendError
    } else if lowered.contains("data") || lowered.contains("collate") || lowered.contains("replay")
    {
        ProbeStatus::DataError
    } else {
        ProbeStatus::BackendError
    }
}

pub fn run_preflight(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
    artifacts: &BcArtifactPaths,
) -> Result<PreflightRuntime, String> {
    let preflight_started = Instant::now();
    let cache_context =
        bc_preflight_cache_context(config, preflight, model_config, device_label, artifacts);
    let cache_key = cache_context.cache_key.clone();
    let paths = cache_context.paths;
    let explicit = cache_context.explicit;
    let mut artifact_logger = PreflightArtifactLogger::new(&paths, cache_key.clone())?;
    let mut resource_sampler = crate::system_metrics::ResourceSampler::new();
    artifact_logger.phase_started("cache_lookup", None)?;
    let t_cache_lookup = Instant::now();

    // Phase 1-3: Probe search (or cache hit -- return cached runtime truthfully)
    if let Some(cached) = matching_bc_preflight_cache(&BcPreflightCacheContext {
        cache_key: cache_key.clone(),
        paths: PreflightPaths {
            cache_path: paths.cache_path.clone(),
            manifest_cache_path: paths.manifest_cache_path.clone(),
            discovery_summary_path: paths.discovery_summary_path.clone(),
            discovery_index_path: paths.discovery_index_path.clone(),
            events_log_path: paths.events_log_path.clone(),
            metrics_log_path: paths.metrics_log_path.clone(),
            candidates_log_path: paths.candidates_log_path.clone(),
            state_path: paths.state_path.clone(),
            report_path: paths.report_path.clone(),
        },
        explicit,
    })? {
        let runtime = cached.runtime;
        let benchmark = cached.benchmark;
        artifact_logger.phase_completed(
            "cache_lookup",
            t_cache_lookup.elapsed().as_secs_f64(),
            Some("matching cache entry found; reused cached runtime and benchmark".to_string()),
            Some(runtime),
        )?;
        artifact_logger.phase_skipped(
            "train_probe",
            "cache hit reused cached train probe selection".to_string(),
        )?;
        artifact_logger.phase_skipped(
            "validation_probe",
            "cache hit reused cached validation probe selection".to_string(),
        )?;
        artifact_logger.phase_skipped(
            "runtime_resolve",
            "cache hit reused cached runtime resolution".to_string(),
        )?;
        artifact_logger.phase_skipped(
            "manifest_scan",
            "cache hit reused cached manifest-dependent runtime".to_string(),
        )?;
        artifact_logger.phase_skipped(
            "loader_tuning",
            "cache hit reused cached loader runtime".to_string(),
        )?;
        artifact_logger.phase_skipped(
            "stage_2_benchmark",
            "cache hit reused cached benchmark result without fake finalists".to_string(),
        )?;
        artifact_logger.cache_written(runtime)?;
        let total_secs = preflight_started.elapsed().as_secs_f64();
        write_preflight_report(
            &paths.report_path,
            &PreflightReport {
                cache_key,
                runtime,
                cache_hit: true,
                train_probe_results: 0,
                validation_probe_results: 0,
                benchmark: benchmark.clone(),
                total_elapsed_seconds: total_secs,
            },
        )?;
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight cache hit:",
                "reused cached runtime and benchmark; skipped probes, loader tuning, and stage-2 benchmark",
            )
        );
        println!(
            "{}",
            format_timed_phase_message("preflight_total", "cache hit reused", total_secs)
        );
        return Ok(PreflightRuntime {
            runtime,
            train_probe_results: Vec::new(),
            validation_probe_results: Vec::new(),
            benchmark,
            advisories: Vec::new(),
            explicit,
        });
    }

    let (
        _train_microbatch,
        train_probe_results,
        train_runtime_seed,
        _validation_microbatch,
        validation_probe_results,
        selected,
        probe_train_secs,
        probe_val_secs,
        resolve_secs,
    ) = {
        artifact_logger.phase_completed(
            "cache_lookup",
            t_cache_lookup.elapsed().as_secs_f64(),
            Some("no matching cache entry".to_string()),
            None,
        )?;
        artifact_logger.metrics_event(&resource_sampler.snapshot("cache_lookup"))?;
        let phase_pb = make_bar(6, "[{bar:30.magenta/black}] {pos}/{len} {msg}")?;
        phase_pb.set_message(preflight_phase_label("train microbatch probe"));

        let train_seed = config.microbatch_size.unwrap_or(config.batch_size);

        let resumed_train_results =
            resumed_candidate_results(&artifact_logger.state, "train_probe");
        let resumed_train = completed_phase_seconds(&artifact_logger.state, "train_probe")
            .and_then(|secs| {
                successful_candidate_selection(&resumed_train_results)
                    .map(|selected| (selected, secs))
            });
        let (train_microbatch, train_probe_results, train_runtime_seed, probe_train_secs) =
            if let Some((selected, secs)) = resumed_train {
                artifact_logger.phase_skipped(
                    "train_probe",
                    format!("reused interrupted state selected_candidate_mb={selected}"),
                )?;
                let seed = exact_train_probe_runtime_seed(
                    config,
                    preflight,
                    selected,
                    &resumed_train_results,
                    resumed_train_results.len(),
                );
                let should_rerun = seed.is_none() && config.bc_shards_manifest_path.is_none();
                if should_rerun {
                    artifact_logger.phase_started(
                        "train_probe",
                        Some(format!(
                            "resume_state_missing_loader_seed selected_candidate_mb={selected}; rerunning"
                        )),
                    )?;
                    let t_train_probe = Instant::now();
                    let (selected, results, seed) = match search_train_microbatch(
                        config_path,
                        config,
                        preflight,
                        artifacts,
                        train_seed,
                    ) {
                        Ok(result) => result,
                        Err(err) => {
                            log_interrupted_phase_if_probe_interrupted(
                                &mut artifact_logger,
                                "train_probe",
                                t_train_probe,
                                &err,
                            )?;
                            return Err(err);
                        }
                    };
                    let secs = t_train_probe.elapsed().as_secs_f64();
                    artifact_logger.candidate_results("train_probe", &results)?;
                    artifact_logger.phase_completed(
                        "train_probe",
                        secs,
                        Some(format!("selected_candidate_mb={selected}")),
                        None,
                    )?;
                    (selected, results, seed, secs)
                } else {
                    (selected, resumed_train_results, seed, secs)
                }
            } else {
                artifact_logger.phase_started(
                    "train_probe",
                    Some(format!("seed_candidate_mb={train_seed}")),
                )?;
                let t_train_probe = Instant::now();
                let (selected, results, seed) = match search_train_microbatch(
                    config_path,
                    config,
                    preflight,
                    artifacts,
                    train_seed,
                ) {
                    Ok(result) => result,
                    Err(err) => {
                        log_interrupted_phase_if_probe_interrupted(
                            &mut artifact_logger,
                            "train_probe",
                            t_train_probe,
                            &err,
                        )?;
                        return Err(err);
                    }
                };
                let secs = t_train_probe.elapsed().as_secs_f64();
                artifact_logger.candidate_results("train_probe", &results)?;
                artifact_logger.phase_completed(
                    "train_probe",
                    secs,
                    Some(format!("selected_candidate_mb={selected}")),
                    None,
                )?;
                (selected, results, seed, secs)
            };
        artifact_logger.metrics_event(&resource_sampler.snapshot("train_probe"))?;
        phase_pb.inc(1);

        phase_pb.set_message(preflight_phase_label("validation microbatch probe"));
        let validation_seed = config.validation_microbatch_size.unwrap_or(train_seed);
        let resumed_validation_results =
            resumed_candidate_results(&artifact_logger.state, "validation_probe");
        let resumed_validation =
            completed_phase_seconds(&artifact_logger.state, "validation_probe").and_then(|secs| {
                successful_candidate_selection(&resumed_validation_results)
                    .map(|selected| (selected, secs))
            });
        let (validation_microbatch, validation_probe_results, probe_val_secs) =
            if let Some((selected, secs)) = resumed_validation {
                artifact_logger.phase_skipped(
                    "validation_probe",
                    format!("reused interrupted state selected_candidate_mb={selected}"),
                )?;
                (selected, resumed_validation_results, secs)
            } else {
                artifact_logger.phase_started(
                    "validation_probe",
                    Some(format!("seed_candidate_mb={validation_seed}")),
                )?;
                let t_val_probe = Instant::now();
                let (selected, results) = match search_validation_microbatch(
                    config_path,
                    config,
                    preflight,
                    artifacts,
                    validation_seed,
                ) {
                    Ok(result) => result,
                    Err(err) => {
                        log_interrupted_phase_if_probe_interrupted(
                            &mut artifact_logger,
                            "validation_probe",
                            t_val_probe,
                            &err,
                        )?;
                        return Err(err);
                    }
                };
                let secs = t_val_probe.elapsed().as_secs_f64();
                artifact_logger.candidate_results("validation_probe", &results)?;
                artifact_logger.phase_completed(
                    "validation_probe",
                    secs,
                    Some(format!("selected_candidate_mb={selected}")),
                    None,
                )?;
                (selected, results, secs)
            };
        artifact_logger.metrics_event(&resource_sampler.snapshot("validation_probe"))?;
        phase_pb.inc(1);

        phase_pb.set_message(preflight_phase_label("resolve runtime"));
        artifact_logger.phase_started("runtime_resolve", None)?;
        let t_resolve = Instant::now();
        let selected = resolve_runtime_config(
            config.batch_size,
            explicit,
            train_microbatch,
            validation_microbatch,
        );
        let resolve_secs = t_resolve.elapsed().as_secs_f64();
        artifact_logger.phase_completed(
            "runtime_resolve",
            resolve_secs,
            Some(format!(
                "train_mb={} validation_mb={} accum_steps={}",
                selected.train_microbatch_size,
                selected.validation_microbatch_size,
                selected.accum_steps
            )),
            Some(EffectiveRuntimeConfig::from_config(
                selected,
                hydra_train_runtime::config::loader_runtime_config(config),
                config,
            )),
        )?;
        println!(
            "{}",
            format_timed_phase_message(
                "post_validation",
                "selected validation candidate; preparing runtime tuning",
                0.0,
            )
        );
        phase_pb.inc(1);
        phase_pb.finish_and_clear();

        (
            train_microbatch,
            train_probe_results,
            train_runtime_seed,
            validation_microbatch,
            validation_probe_results,
            selected,
            probe_train_secs,
            probe_val_secs,
            resolve_secs,
        )
    };

    // Phase 4: Manifest scan (always runs)
    let t_scan = Instant::now();
    artifact_logger.phase_started("manifest_scan", None)?;
    let mut tuned_config = config.clone();
    tuned_config.microbatch_size = Some(selected.train_microbatch_size);
    tuned_config.validation_microbatch_size = Some(selected.validation_microbatch_size);
    let manifest = load_or_scan_manifest_cache(
        ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &config.data_dir,
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            progress: None,
            scan_error_context: "preflight data",
        },
        |_| {},
    )
    .map_err(|err| {
        err.replacen(
            "failed to scan preflight data",
            "failed to scan preflight runtime data",
            1,
        )
    })?;
    let scan_secs = t_scan.elapsed().as_secs_f64();
    artifact_logger.phase_completed(
        "manifest_scan",
        scan_secs,
        Some(format!("sources={}", manifest.sources.len())),
        None,
    )?;
    artifact_logger.metrics_event(&crate::system_metrics::progress_event(
        "manifest_scan",
        manifest.sources.len() as u64,
        manifest.sources.len() as u64,
        scan_secs,
        crate::system_metrics::rate_per_second(manifest.sources.len() as u64, scan_secs),
        None,
    ))?;
    artifact_logger.metrics_event(&resource_sampler.snapshot("manifest_scan"))?;

    // Phase 5: Loader runtime tuning (always runs)
    let t_loader = Instant::now();
    artifact_logger.phase_started("loader_tuning", None)?;
    let train_device = train_device(&config.device)?;
    let ranked_loaders = if config.bc_shards_manifest_path.is_some() {
        vec![crate::runtime_autotune_shim::RankedLoaderRuntime {
            loader: hydra_train_runtime::config::loader_runtime_config(&tuned_config),
            tuple: (
                tuned_config.archive_queue_bound,
                tuned_config.buffer_samples,
                tuned_config.buffer_games,
            ),
            train_samples_per_second: 0.0,
        }]
    } else {
        autotune_ranked_loader_runtime_with_seed(
            &tuned_config,
            preflight,
            &manifest,
            &train_device,
            preflight.real_benchmark_loader_candidates.max(1),
            train_runtime_seed,
        )?
    };
    let loader = ranked_loaders
        .first()
        .map(|ranked| ranked.loader)
        .ok_or_else(|| "loader runtime autotune returned no ranked candidates".to_string())?;
    let mut runtime = EffectiveRuntimeConfig::from_config(selected, loader, config);
    let loader_secs = t_loader.elapsed().as_secs_f64();
    artifact_logger.phase_completed(
        "loader_tuning",
        loader_secs,
        Some(format!("ranked_candidates={}", ranked_loaders.len())),
        Some(runtime),
    )?;

    // Phase 6: Stage-2 finalist benchmark (always runs when enabled --
    // the benchmark IS the ground truth for tuning values)
    let t_benchmark = Instant::now();
    artifact_logger.phase_started("stage_2_benchmark", None)?;
    let benchmark = if preflight.real_benchmark_enabled && config.bc_shards_manifest_path.is_none()
    {
        let train_candidates = diverse_probe_candidates(
            &train_probe_results,
            selected.train_microbatch_size,
            preflight.real_benchmark_train_candidates,
            preflight.finalist_margin_ratio,
        );
        let validation_candidates = diverse_probe_candidates(
            &validation_probe_results,
            selected.validation_microbatch_size,
            preflight.real_benchmark_validation_candidates,
            preflight.finalist_margin_ratio,
        );
        let loader_candidates = select_loader_finalists(
            &ranked_loaders,
            preflight.real_benchmark_loader_candidates,
            preflight.finalist_margin_ratio,
            runtime.loader,
        );
        let finalists = build_stage_two_finalists(StageTwoFinalistInputs {
            config,
            preflight,
            selected: &runtime,
            train_candidates: &train_candidates,
            validation_candidates: &validation_candidates,
            loader_candidates: &loader_candidates,
            train_probe_results: &train_probe_results,
            validation_probe_results: &validation_probe_results,
            ranked_loaders: &ranked_loaders,
            unsafe_batch_candidates: &unsafe_stage_two_batch_candidates(preflight),
        });
        let best = run_stage_two_finalist_benchmark(
            preflight,
            StageTwoBenchmarkContext {
                config,
                manifest: &manifest,
                train_device: &train_device,
                artifacts,
                finalists: &finalists,
                train_candidates: train_candidates.len(),
                validation_candidates: validation_candidates.len(),
                loader_candidates: loader_candidates.len(),
            },
        )?;
        runtime = EffectiveRuntimeConfig::from_config(
            hydra_train_runtime::preflight::SelectedRuntimeConfig {
                train_microbatch_size: best.runtime.train_microbatch_size,
                validation_microbatch_size: best.runtime.validation_microbatch_size.max(1),
                accum_steps: best.runtime.accum_steps.max(1),
                unsafe_selected_batch_size: finalists
                    .iter()
                    .find(|finalist| finalist.runtime == best.runtime)
                    .and_then(|finalist| finalist.unsafe_batch_size),
                unsafe_selected_learning_rate: best.runtime.learning_rate,
                unsafe_selected_min_learning_rate: best.runtime.min_learning_rate,
                unsafe_selected_warmup_steps: best.runtime.warmup_steps,
            },
            best.runtime.loader,
            config,
        );
        println!(
            "{}",
            format_preflight_selection_line(format!(
                "stage-2 winner train_mb={} val_mb={} accum_steps={} requested_precision={} effective_precision={} wall_clock_effective={:.2} samples/s mode={:?}",
                best.runtime.train_microbatch_size,
                best.runtime.validation_microbatch_size,
                runtime.selected.accum_steps,
                hydra_train_runtime::preflight::requested_precision_signature(
                    runtime.requested_precision
                ),
                runtime.effective_precision,
                best.score.wall_clock_samples_per_second,
                best.metadata.mode,
            ))
        );
        if let Some(selected_batch) = runtime.selected.unsafe_selected_batch_size {
            let selected_lr = runtime
                .selected
                .unsafe_selected_learning_rate
                .unwrap_or(config.bc.learning_rate);
            let selected_min_lr = runtime
                .selected
                .unsafe_selected_min_learning_rate
                .unwrap_or(config.bc.min_learning_rate);
            let selected_warmup_steps = runtime
                .selected
                .unsafe_selected_warmup_steps
                .unwrap_or(config.bc.warmup_steps);
            println!(
                "{}",
                format_preflight_selection_line(format!(
                    "unsafe math winner selected_batch={} selected_lr={:.2e} selected_min_lr={:.2e} selected_warmup_steps={} lr_auto_scaled=false apply=fresh_start_only resume=ignored_or_refused_if_checkpoint_contract_mismatch",
                    selected_batch, selected_lr, selected_min_lr, selected_warmup_steps,
                ))
            );
        }
        Some(best)
    } else {
        None
    };
    let benchmark_secs = t_benchmark.elapsed().as_secs_f64();
    artifact_logger.phase_completed(
        "stage_2_benchmark",
        benchmark_secs,
        Some(format!(
            "finalists_benchmarked={}",
            benchmark
                .as_ref()
                .map(|b| b.metadata.finalists_benchmarked)
                .unwrap_or(0)
        )),
        Some(runtime),
    )?;
    let mut advisories = Vec::new();
    if benchmark.is_none() {
        advisories.extend(selected_runtime_probe_advisories(
            ProbeKind::Train,
            runtime.selected.train_microbatch_size,
            &train_probe_results,
        ));
        advisories.extend(selected_runtime_probe_advisories(
            ProbeKind::Validation,
            runtime.selected.validation_microbatch_size,
            &validation_probe_results,
        ));
    }

    // Atomic cache write: only update cache after ALL work completes successfully.
    persist_bc_preflight_runtime(
        cache_key.clone(),
        runtime,
        benchmark.clone(),
        &paths,
        artifacts,
    )?;
    artifact_logger.cache_written(runtime)?;

    // Timing summary
    let total_secs = preflight_started.elapsed().as_secs_f64();
    write_preflight_report(
        &paths.report_path,
        &PreflightReport {
            cache_key,
            runtime,
            cache_hit: false,
            train_probe_results: train_probe_results.len(),
            validation_probe_results: validation_probe_results.len(),
            benchmark: benchmark.clone(),
            total_elapsed_seconds: total_secs,
        },
    )?;
    println!(
        "{}",
        format_timed_phase_message("train_probe", "microbatch search", probe_train_secs)
    );
    println!(
        "{}",
        format_timed_phase_message("validation_probe", "microbatch search", probe_val_secs)
    );
    println!(
        "{}",
        format_timed_phase_message("runtime_resolve", "config resolution", resolve_secs)
    );
    println!(
        "{}",
        format_timed_phase_message("manifest_scan", "data source scan", scan_secs)
    );
    println!(
        "{}",
        format_timed_phase_message("loader_tuning", "runtime autotune", loader_secs)
    );
    println!(
        "{}",
        format_timed_phase_message(
            "stage_2_benchmark",
            &format!(
                "{} finalists",
                benchmark
                    .as_ref()
                    .map(|b| b.metadata.finalists_benchmarked)
                    .unwrap_or(0)
            ),
            benchmark_secs
        )
    );
    if benchmark.is_some() {
        let benchmark_paths = PreflightBenchmarkPaths::new(artifacts);
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight benchmark report:",
                benchmark_paths.report_path().display(),
            )
        );
    }
    println!(
        "{}",
        format_timed_phase_message("preflight_total", "all phases complete", total_secs)
    );

    Ok(PreflightRuntime {
        runtime,
        train_probe_results,
        validation_probe_results,
        benchmark,
        advisories,
        explicit,
    })
}

pub fn run_rl_preflight(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    _train_device: &LibTorchDevice,
) -> Result<RlPreflightRuntime, String> {
    let rl = config
        .rl
        .as_ref()
        .ok_or_else(|| "RL preflight requested without rl config block".to_string())?;
    let started = Instant::now();
    let artifacts = RlArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;

    if let Some(cached) =
        matching_rl_preflight_cache(config, preflight, &HydraModelConfig::learner(), &artifacts)?
    {
        let runtime = cached.runtime;
        let tuned_games = runtime.loader.buffer_games;
        let tuned_microbatch = runtime.selected.train_microbatch_size;
        println!(
            "{}",
            format_preflight_summary_line(
                "RL preflight cache hit:",
                format!(
                    "reusing cached runtime games_per_batch={} microbatch_size={} requested_precision={} effective_precision={} (identical fingerprint)",
                    tuned_games,
                    tuned_microbatch,
                    hydra_train_runtime::preflight::requested_precision_signature(
                        runtime.requested_precision
                    ),
                    runtime.effective_precision,
                ),
            )
        );
        println!(
            "{}",
            format_timed_phase_message(
                "rl_runtime_tuning",
                "complete (cached)",
                started.elapsed().as_secs_f64(),
            )
        );
        return Ok(RlPreflightRuntime {
            selected_games_per_batch: tuned_games,
            selected_microbatch_size: tuned_microbatch,
            runtime,
            rl_games_probe_results: Vec::new(),
            rl_microbatch_probe_results: Vec::new(),
        });
    }

    println!(
        "{}",
        format_timed_phase_message("rl_runtime_tuning", "starting", 0.0)
    );
    let (selected_games_per_batch, game_results) = search_rl_runtime_candidate(
        config_path,
        config,
        preflight,
        &artifacts,
        ProbeKind::RlGames,
        rl.games_per_batch,
    )?;
    let microbatch_seed = rl
        .microbatch_size
        .unwrap_or(hydra_train_types::config::DEFAULT_RL_MICROBATCH_SIZE)
        .min(config.batch_size.max(1))
        .max(1);
    let (selected_microbatch_size, microbatch_results) = search_rl_runtime_candidate(
        config_path,
        config,
        preflight,
        &artifacts,
        ProbeKind::RlMicrobatch,
        microbatch_seed,
    )?;
    let runtime = EffectiveRuntimeConfig::from_config(
        hydra_train_runtime::preflight::SelectedRuntimeConfig {
            train_microbatch_size: selected_microbatch_size,
            validation_microbatch_size: config
                .validation_microbatch_size
                .unwrap_or(selected_microbatch_size),
            accum_steps: config
                .batch_size
                .div_ceil(selected_microbatch_size.max(1))
                .max(1),
            unsafe_selected_batch_size: None,
            unsafe_selected_learning_rate: None,
            unsafe_selected_min_learning_rate: None,
            unsafe_selected_warmup_steps: None,
        },
        hydra_train_runtime::preflight::LoaderRuntimeConfig {
            num_threads: config.num_threads,
            buffer_games: selected_games_per_batch,
            buffer_samples: config.buffer_samples,
            archive_queue_bound: config.archive_queue_bound,
        },
        config,
    );
    persist_rl_preflight_runtime(config, preflight, &artifacts, runtime)?;
    println!(
        "{}",
        format_preflight_summary_line(
            "RL Preflight:",
            format!(
                "selected games_per_batch={} rl.microbatch_size={} requested_precision={} effective_precision={} (stored in preflight cache for RL runtime reuse)",
                selected_games_per_batch,
                selected_microbatch_size,
                hydra_train_runtime::preflight::requested_precision_signature(
                    runtime.requested_precision
                ),
                runtime.effective_precision,
            )
        )
    );
    println!(
        "{}",
        format_timed_phase_message(
            "rl_runtime_tuning",
            "complete",
            started.elapsed().as_secs_f64(),
        )
    );
    Ok(RlPreflightRuntime {
        selected_games_per_batch,
        selected_microbatch_size,
        runtime,
        rl_games_probe_results: game_results,
        rl_microbatch_probe_results: microbatch_results,
    })
}

#[cfg(test)]
mod tests;
