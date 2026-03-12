#[path = "train/config.rs"]
mod config;
#[path = "train/artifacts.rs"]
mod artifacts;
#[path = "train/presentation.rs"]
mod presentation;
#[path = "train/resume.rs"]
mod resume;
#[path = "train/status.rs"]
mod status;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use burn::backend::{Autodiff, LibTorch, libtorch::LibTorchDevice};
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use colored::Colorize;
use hydra_train::data::pipeline::{
    StreamingLoaderConfig, scan_data_sources_with_progress, stream_train_epoch, stream_val_pass,
};
use hydra_train::data::sample::collate_samples;
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::preflight::{
    ExplicitSettings, HardwareFingerprint, PreflightCacheEntry, PreflightCacheKey,
    PreflightReport, ProbeKind, ProbeResult, ProbeStatus, SelectedRuntimeConfig,
    WorkloadFingerprint, candidate_ladder, default_notes, resolve_runtime_config,
};
use hydra_train::training::bc::{
    BCTrainerConfig, policy_agreement, target_actions_from_policy_target, warmup_then_cosine_lr,
};
use hydra_train::training::losses::{HydraLoss, HydraLossConfig, HydraTargets, LossBreakdown};
use indicatif::{MultiProgress, ProgressBar};
use tboard::EventWriter;

use self::artifacts::{
    BcArtifactPaths, PreflightPaths, append_step_log, append_training_log, log_tensorboard,
    save_checkpoint, save_latest_checkpoint_and_state, write_preflight_cache,
    write_preflight_report,
};
use self::config::{
    AdvancedLossConfig, TrainConfig, configure_threads, device_label, parse_args, read_config,
    train_device, train_microbatch_size, validate_config,
    validation_microbatch_size, validation_sample_limit,
};
use self::presentation::{format_progress_message, make_bar, make_spinner, phase_label};
use self::resume::{
    BestValidation, EpochContinuation, ResumeContext, paused_training_message,
    resumed_progress_message, runtime_resume_contract,
    validate_resume_runtime_compatibility,
};
use self::status::{
    display_step_label, display_validation_scope_label, epoch_progress_message_with_rate,
    estimate_epoch_progress, reached_session_step_budget, session_steps_completed,
};

#[cfg(test)]
use self::status::{EpochProgressEstimate, format_rough_duration};

#[cfg(test)]
use self::config::default_seed;
#[cfg(test)]
use hydra_train::preflight::PreflightConfig;
#[cfg(test)]
use self::resume::{
    BcResumeState, ResumeSemantics, build_resume_state, checkpoint_base_from_path,
    latest_state_path_for_checkpoint_base, resume_banner_message, test_runtime_resume_contract,
};

type TrainBackend = Autodiff<LibTorch<f32>>;
type ValidBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

struct PreflightRuntime {
    report: PreflightReport,
    selected: SelectedRuntimeConfig,
}

#[derive(Default, serde::Serialize)]
struct ScalarAverages {
    total_loss: f64,
    policy_agreement: f64,
    loss_policy: f64,
    loss_value: f64,
    loss_grp: f64,
    loss_tenpai: f64,
    loss_danger: f64,
    loss_opp_next: f64,
    loss_score_pdf: f64,
    loss_score_cdf: f64,
    num_batches: usize,
}

#[derive(Clone, Copy, Default)]
struct BatchStats {
    total_loss: f64,
    policy_agreement: f64,
    loss_policy: f64,
    loss_value: f64,
    loss_grp: f64,
    loss_tenpai: f64,
    loss_danger: f64,
    loss_opp_next: f64,
    loss_score_pdf: f64,
    loss_score_cdf: f64,
}

#[derive(serde::Serialize)]
struct EpochLogEntry {
    epoch: usize,
    global_step: usize,
    lr: f64,
    train_total_loss: f64,
    train_policy_agreement: f64,
    train_loss_policy: f64,
    train_loss_value: f64,
    train_loss_grp: f64,
    train_loss_tenpai: f64,
    train_loss_danger: f64,
    train_loss_opp_next: f64,
    train_loss_score_pdf: f64,
    train_loss_score_cdf: f64,
    val_total_loss: Option<f64>,
    val_policy_loss: Option<f64>,
    val_policy_agreement: Option<f64>,
    best_val_policy_loss: Option<f64>,
    best_val_agreement: Option<f64>,
    num_batches: usize,
}

#[derive(serde::Serialize)]
struct StepLogEntry {
    global_step: usize,
    epoch: usize,
    lr: f64,
    train_total_loss: f64,
    train_policy_agreement: f64,
    train_loss_policy: f64,
    train_loss_value: f64,
    train_loss_grp: f64,
    train_loss_tenpai: f64,
    train_loss_danger: f64,
    train_loss_opp_next: f64,
    train_loss_score_pdf: f64,
    train_loss_score_cdf: f64,
    val_total_loss: Option<f64>,
    val_policy_loss: Option<f64>,
    val_policy_agreement: Option<f64>,
    best_val_policy_loss: Option<f64>,
    best_val_agreement: Option<f64>,
}

struct BannerStats {
    total_sources: usize,
    total_games: usize,
    train_count: usize,
    val_count: usize,
    accum_steps: usize,
    counts_exact: bool,
}

#[derive(Clone, Copy)]
struct ValidationSummary {
    total_loss: f64,
    policy_loss: f64,
    agreement: f64,
    samples: usize,
}

fn schedule_total_steps(config: &TrainConfig, session_start_global_step: usize) -> usize {
    config
        .max_train_steps
        .map(|budget| session_start_global_step + budget)
        .unwrap_or(config.num_epochs.max(1))
        .max(1)
}

fn scalar1<B: Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor.clone().into_scalar().elem::<f64>()
}

impl ScalarAverages {
    fn record_batch(&mut self, batch: BatchStats) {
        self.total_loss += batch.total_loss;
        self.policy_agreement += batch.policy_agreement;
        self.loss_policy += batch.loss_policy;
        self.loss_value += batch.loss_value;
        self.loss_grp += batch.loss_grp;
        self.loss_tenpai += batch.loss_tenpai;
        self.loss_danger += batch.loss_danger;
        self.loss_opp_next += batch.loss_opp_next;
        self.loss_score_pdf += batch.loss_score_pdf;
        self.loss_score_cdf += batch.loss_score_cdf;
        self.num_batches += 1;
    }

    fn finalize(mut self) -> Self {
        if self.num_batches == 0 {
            return self;
        }
        let denom = self.num_batches as f64;
        self.total_loss /= denom;
        self.policy_agreement /= denom;
        self.loss_policy /= denom;
        self.loss_value /= denom;
        self.loss_grp /= denom;
        self.loss_tenpai /= denom;
        self.loss_danger /= denom;
        self.loss_opp_next /= denom;
        self.loss_score_pdf /= denom;
        self.loss_score_cdf /= denom;
        self
    }
}

fn batch_stats_from_breakdown<B: Backend>(
    agreement: f64,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    BatchStats {
        total_loss: scalar1(&breakdown.total),
        policy_agreement: agreement,
        loss_policy: scalar1(&breakdown.policy),
        loss_value: scalar1(&breakdown.value),
        loss_grp: scalar1(&breakdown.grp),
        loss_tenpai: scalar1(&breakdown.tenpai),
        loss_danger: scalar1(&breakdown.danger),
        loss_opp_next: scalar1(&breakdown.opp_next),
        loss_score_pdf: scalar1(&breakdown.score_pdf),
        loss_score_cdf: scalar1(&breakdown.score_cdf),
    }
}

fn model_kind(config: &HydraModelConfig) -> &'static str {
    if config.is_learner() {
        "learner"
    } else {
        "actor"
    }
}

fn read_preflight_cache(path: &Path) -> Result<PreflightCacheEntry, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read preflight cache {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse preflight cache {}: {err}", path.display()))
}

fn advanced_loss_signature(config: Option<&AdvancedLossConfig>) -> String {
    match config {
        Some(config) => serde_json::to_string(config)
            .unwrap_or_else(|_| "advanced_loss:unserializable".to_string()),
        None => "advanced_loss:none".to_string(),
    }
}

fn workload_fingerprint(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
) -> WorkloadFingerprint {
    WorkloadFingerprint {
        batch_size: config.batch_size,
        augment: config.augment,
        model_signature: format!(
            "blocks:{} input:{} hidden:{} groups:{} action:{} score_bins:{}",
            model_config.num_blocks,
            model_config.input_channels,
            model_config.hidden_channels,
            model_config.num_groups,
            model_config.action_space,
            model_config.score_bins,
        ),
        code_signature: format!(
            "hydra-train:{}:{}:preflight-v2",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_NAME")
        ),
        advanced_loss_signature: advanced_loss_signature(config.advanced_loss.as_ref()),
    }
}

fn hardware_fingerprint(device_label: &str) -> HardwareFingerprint {
    HardwareFingerprint {
        device_label: device_label.to_string(),
        backend: "burn-libtorch".to_string(),
    }
}

fn preflight_cache_key(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
) -> PreflightCacheKey {
    PreflightCacheKey {
        hardware: hardware_fingerprint(device_label),
        workload: workload_fingerprint(config, model_config),
    }
}

#[derive(Clone, Copy)]
struct ProbeRequest {
    kind: ProbeKind,
    candidate_microbatch: usize,
    warmup_steps: usize,
    measure_steps: usize,
}

fn probe_kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
    }
}

fn parse_probe_kind(value: &str) -> Result<ProbeKind, String> {
    match value {
        "train" => Ok(ProbeKind::Train),
        "validation" => Ok(ProbeKind::Validation),
        _ => Err(format!("unsupported preflight probe kind: {value}")),
    }
}

fn preflight_request_from_env() -> Result<Option<(ProbeRequest, PathBuf)>, String> {
    let Some(kind) = env::var_os("HYDRA_PREFLIGHT_PROBE_KIND") else {
        return Ok(None);
    };
    let kind = parse_probe_kind(&kind.to_string_lossy())?;
    let candidate_microbatch = env::var("HYDRA_PREFLIGHT_CANDIDATE_MB")
        .map_err(|_| "missing HYDRA_PREFLIGHT_CANDIDATE_MB".to_string())?
        .parse::<usize>()
        .map_err(|err| format!("invalid HYDRA_PREFLIGHT_CANDIDATE_MB: {err}"))?;
    let warmup_steps = env::var("HYDRA_PREFLIGHT_WARMUP_STEPS")
        .map_err(|_| "missing HYDRA_PREFLIGHT_WARMUP_STEPS".to_string())?
        .parse::<usize>()
        .map_err(|err| format!("invalid HYDRA_PREFLIGHT_WARMUP_STEPS: {err}"))?;
    let measure_steps = env::var("HYDRA_PREFLIGHT_MEASURE_STEPS")
        .map_err(|_| "missing HYDRA_PREFLIGHT_MEASURE_STEPS".to_string())?
        .parse::<usize>()
        .map_err(|err| format!("invalid HYDRA_PREFLIGHT_MEASURE_STEPS: {err}"))?;
    let result_path = PathBuf::from(
        env::var("HYDRA_PREFLIGHT_RESULT_PATH")
            .map_err(|_| "missing HYDRA_PREFLIGHT_RESULT_PATH".to_string())?,
    );
    Ok(Some((
        ProbeRequest {
            kind,
            candidate_microbatch,
            warmup_steps,
            measure_steps,
        },
        result_path,
    )))
}

fn write_probe_result(path: &Path, result: &ProbeResult) -> Result<(), String> {
    let json = serde_json::to_string(result)
        .map_err(|err| format!("failed to serialize probe result {}: {err}", path.display()))?;
    fs::write(path, json)
        .map_err(|err| format!("failed to write probe result {}: {err}", path.display()))
}

fn read_probe_result(path: &Path) -> Result<ProbeResult, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read probe result {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse probe result {}: {err}", path.display()))
}

fn probe_result_path(
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidate_microbatch: usize,
    attempt: usize,
) -> PathBuf {
    artifacts.root.join(format!(
        "preflight_probe_{}_{}_{}.json",
        probe_kind_name(kind),
        candidate_microbatch,
        attempt
    ))
}

fn measure_samples_per_second(samples: usize, elapsed: Duration) -> f64 {
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

fn probe_train_candidate(
    config: &TrainConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &hydra_train::data::pipeline::DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let scheduler_warmup_steps = config.max_train_steps.map_or(
        BCTrainerConfig::default_learner().warmup_steps,
        |max_steps| max_steps.clamp(1, BCTrainerConfig::default_learner().warmup_steps.min(100)),
    );
    let train_cfg = BCTrainerConfig::default_learner()
        .with_batch_size(config.batch_size)
        .with_lr(BCTrainerConfig::default_learner().lr)
        .with_warmup_steps(scheduler_warmup_steps);
    let mut model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let microbatch_size = request.candidate_microbatch.min(config.batch_size).max(1);
    let accum_steps = config.batch_size.div_ceil(microbatch_size).max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut completed_steps = 0usize;
    let mut accum_current = 0usize;
    let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
        GradientsAccumulator::new();
    let mut measure_start = None;

    for buffer_result in stream_train_epoch(manifest, loader_config, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight train stream failed: {err}"))?;
        for chunk in buffer.chunks(microbatch_size) {
            let Some((obs, targets)) =
                collate_samples::<TrainBackend>(chunk, config.augment, train_device)
            else {
                continue;
            };
            let output = model.forward(obs);
            let breakdown = loss_fn.total_loss(&output, &targets);
            let grads = breakdown.total.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            accumulator.accumulate(&model, grads);
            accum_current += 1;
            if accum_current < accum_steps {
                continue;
            }
            let lr = effective_lr(&train_cfg, completed_steps, target_steps.max(1));
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            accum_current = 0;
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    request.measure_steps.max(1) * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    Err(format!(
        "not enough train data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

fn probe_validation_candidate(
    config: &TrainConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &hydra_train::data::pipeline::DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let model_valid = model.valid();
    let loss_fn = HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let microbatch_size = request.candidate_microbatch.max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut completed_steps = 0usize;
    let mut measure_start = None;

    for buffer_result in stream_val_pass(manifest, loader_config, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight validation stream failed: {err}"))?;
        for chunk in buffer.chunks(microbatch_size) {
            let Some((obs, targets)) = collate_samples::<ValidBackend>(chunk, false, train_device)
            else {
                continue;
            };
            let output = model_valid.forward(obs);
            let _ = validation_batch_stats(&output, &targets, &loss_fn);
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    request.measure_steps.max(1) * microbatch_size,
                    elapsed,
                ));
            }
        }
    }

    Err(format!(
        "not enough validation data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

fn run_probe_only(
    config: &TrainConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
    };
    let manifest = scan_data_sources_with_progress(&config.data_dir, config.train_fraction, None)
        .map_err(|err| {
        format!(
            "failed to scan preflight data from {}: {err}",
            config.data_dir.display()
        )
    })?;
    let train_device = train_device(&config.device);
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => {
            probe_train_candidate(config, request, &loader_config, &manifest, &train_device)?
        }
        ProbeKind::Validation => {
            probe_validation_candidate(config, request, &loader_config, &manifest, &train_device)?
        }
    };
    write_probe_result(
        result_path,
        &ProbeResult {
            kind: request.kind,
            candidate_microbatch: request.candidate_microbatch,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(measured_samples_per_second),
            detail: format!(
                "stable {} probe on real dataset",
                probe_kind_name(request.kind)
            ),
        },
    )
}

fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<ProbeResult, String> {
    fs::remove_file(result_path).ok();
    let output =
        Command::new(env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?)
            .arg(config_path)
            .env("HYDRA_PREFLIGHT_PROBE_KIND", probe_kind_name(request.kind))
            .env(
                "HYDRA_PREFLIGHT_CANDIDATE_MB",
                request.candidate_microbatch.to_string(),
            )
            .env(
                "HYDRA_PREFLIGHT_WARMUP_STEPS",
                request.warmup_steps.to_string(),
            )
            .env(
                "HYDRA_PREFLIGHT_MEASURE_STEPS",
                request.measure_steps.to_string(),
            )
            .env("HYDRA_PREFLIGHT_RESULT_PATH", result_path)
            .output()
            .map_err(|err| format!("failed to spawn preflight probe child: {err}"))?;

    if result_path.exists() {
        let result = read_probe_result(result_path)?;
        fs::remove_file(result_path).ok();
        return Ok(result);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = format!(
        "probe process status={:?} stdout={} stderr={}",
        output.status.code(),
        stdout.trim(),
        stderr.trim()
    );
    Ok(ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status: classify_probe_detail(&detail),
        measured_samples_per_second: None,
        detail,
    })
}

fn best_probe_result(results: &[ProbeResult]) -> Option<&ProbeResult> {
    results
        .iter()
        .filter(|result| result.status == ProbeStatus::Success)
        .max_by(|left, right| {
            left.measured_samples_per_second
                .unwrap_or(0.0)
                .partial_cmp(&right.measured_samples_per_second.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn probe_candidate_ladder(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    let explicit_candidate = match kind {
        ProbeKind::Train => config.microbatch_size,
        ProbeKind::Validation => config.validation_microbatch_size,
    };
    let use_explicit_only =
        explicit_candidate.is_some() && !config.preflight.allow_override_explicit_microbatch;
    let candidate_list: Vec<usize> = if use_explicit_only {
        vec![explicit_candidate.unwrap_or(1)]
    } else {
        candidates.to_vec()
    };
    let mut results = Vec::new();
    let mut stable_results = Vec::new();

    for candidate in candidate_list {
        let mut stable = true;
        let stable_start = results.len();
        let attempts = config.preflight.required_successes.max(1);
        for attempt in 0..attempts {
            let request = ProbeRequest {
                kind,
                candidate_microbatch: candidate,
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            };
            let result_path = probe_result_path(artifacts, kind, candidate, attempt);
            let result = execute_probe_request(config_path, request, &result_path)?;
            let passed = result.status == ProbeStatus::Success;
            results.push(result);
            if !passed {
                stable = false;
                break;
            }
        }
        if stable {
            stable_results.extend(results[stable_start..].iter().cloned());
            if use_explicit_only {
                return Ok((candidate, results));
            }
        }
    }

    if use_explicit_only {
        return Err(format!(
            "explicit {} microbatch {} failed preflight",
            probe_kind_name(kind),
            explicit_candidate.unwrap_or(1)
        ));
    }

    let selected = best_probe_result(&stable_results)
        .map(|result| result.candidate_microbatch)
        .ok_or_else(|| {
            format!(
                "no stable {} microbatch found in preflight",
                probe_kind_name(kind)
            )
        })?;
    Ok((selected, results))
}

fn apply_preflight_selection(config: &TrainConfig, selected: SelectedRuntimeConfig) -> TrainConfig {
    if config.preflight.advisory_only {
        return config.clone();
    }
    let mut resolved = config.clone();
    resolved.microbatch_size = Some(selected.train_microbatch_size);
    resolved.validation_microbatch_size = Some(selected.validation_microbatch_size);
    resolved
}

fn classify_probe_detail(detail: &str) -> ProbeStatus {
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

fn run_preflight(
    config_path: &Path,
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
    artifacts: &BcArtifactPaths,
) -> Result<PreflightRuntime, String> {
    let cache_key = preflight_cache_key(config, model_config, device_label);
    let paths = PreflightPaths::new(artifacts);
    let explicit = ExplicitSettings {
        train_microbatch_explicit: config.microbatch_size.is_some(),
        validation_microbatch_explicit: config.validation_microbatch_size.is_some(),
    };

    if config.preflight.enabled && config.preflight.reuse_cache && paths.cache_path.exists() {
        let entry = read_preflight_cache(&paths.cache_path)?;
        if entry.cache_key == cache_key {
            let report = PreflightReport {
                schema_version: 1,
                cache_key,
                selected: entry.selected,
                explicit,
                advisory_only: config.preflight.advisory_only,
                cache_hit: true,
                train_probe_results: Vec::new(),
                validation_probe_results: Vec::new(),
                notes: default_notes(),
            };
            write_preflight_report(&paths.report_path, &report)?;
            return Ok(PreflightRuntime {
                report,
                selected: entry.selected,
            });
        }
    }

    let candidates = candidate_ladder(&config.preflight, config.batch_size);
    let (train_microbatch, train_probe_results) = probe_candidate_ladder(
        config_path,
        config,
        artifacts,
        ProbeKind::Train,
        &candidates,
    )?;
    let (validation_microbatch, validation_probe_results) = probe_candidate_ladder(
        config_path,
        config,
        artifacts,
        ProbeKind::Validation,
        &candidates,
    )?;
    let selected = resolve_runtime_config(
        config.batch_size,
        explicit,
        train_microbatch,
        validation_microbatch,
    );
    let report = PreflightReport {
        schema_version: 1,
        cache_key: cache_key.clone(),
        selected,
        explicit,
        advisory_only: config.preflight.advisory_only,
        cache_hit: false,
        train_probe_results,
        validation_probe_results,
        notes: default_notes(),
    };
    write_preflight_report(&paths.report_path, &report)?;
    if config.preflight.enabled && config.preflight.reuse_cache {
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key,
                selected,
            },
        )?;
    }
    Ok(PreflightRuntime { report, selected })
}

fn reject_blocked_advanced_loss_presence(field: &str, weight: Option<f32>) -> Result<(), String> {
    match weight {
        Some(_) => Err(format!(
            "advanced_loss.{field} is not supported in train.rs because this BC data path does not safely support it yet"
        )),
        None => Ok(()),
    }
}

fn build_loss_config(
    advanced_loss: Option<&AdvancedLossConfig>,
) -> Result<HydraLossConfig, String> {
    if let Some(cfg) = advanced_loss {
        reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
        reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
        reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
        reject_blocked_advanced_loss_presence("delta_q", cfg.delta_q)?;
    }

    let safety_residual = advanced_loss
        .and_then(|cfg| cfg.safety_residual)
        .unwrap_or(0.0);

    let loss_config = HydraLossConfig::new().with_w_safety_residual(safety_residual);
    loss_config
        .validate()
        .map_err(|err| format!("invalid loss config: {err}"))?;
    Ok(loss_config)
}

fn lr_status_message(step: usize, warmup_steps: usize, lr: f64) -> String {
    if warmup_steps > 0 && step < warmup_steps {
        format!("lr={lr:.2e} warmup {}/{}", step, warmup_steps)
    } else {
        format!("lr={lr:.2e} cosine")
    }
}

fn effective_lr(train_cfg: &BCTrainerConfig, step: usize, total_steps: usize) -> f64 {
    warmup_then_cosine_lr(
        step,
        train_cfg.warmup_steps.min(total_steps),
        total_steps,
        train_cfg.lr,
        1e-6,
    )
}

fn steps_per_second(window_steps: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if window_steps == 0 || secs <= f64::EPSILON {
        0.0
    } else {
        window_steps as f64 / secs
    }
}

fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &BannerStats,
    scheduler_warmup_steps: usize,
) {
    println!();
    println!();
    println!("{}", "Hydra BC trainer".bold().cyan());
    println!(
        "  {} {}",
        "Model:".white(),
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green()
    );
    println!("  {} {}", "Device:".white(), device_name.green());
    println!(
        "  {} {}",
        "Dataset:".white(),
        if stats.counts_exact {
            format!(
                "{} ({} sources, {} games)",
                config.data_dir.display(),
                stats.total_sources,
                stats.total_games
            )
        } else {
            format!(
                "{} ({} sources, archive counts deferred)",
                config.data_dir.display(),
                stats.total_sources,
            )
        }
        .green()
    );
    println!(
        "  {} {}",
        "Train:".white(),
        if stats.counts_exact {
            format!("{} games | Val: {} games", stats.train_count, stats.val_count)
        } else {
            "streaming split, counts estimated while loading".to_string()
        }
        .green()
    );
    println!(
        "  {} {}",
        "Buffer:".white(),
        format!("{} samples (max {} games)", config.buffer_samples, config.buffer_games).yellow()
    );
    println!(
        "  {} {}",
        "Optimizer batch:".white(),
        format!(
            "{} ({} x {} accum)",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Epochs:".white(),
        config.num_epochs.to_string().yellow()
    );
    println!(
        "  {} {}",
        "Schedule:".white(),
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            scheduler_warmup_steps,
            config
                .max_train_steps
                .map(|steps| steps.to_string())
                .unwrap_or_else(|| "epoch-derived".to_string())
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Output:".white(),
        artifacts.root.display().to_string().green()
    );
    println!(
        "  {} {}",
        "TBoard:".white(),
        if config.tensorboard {
            artifacts.tb_session_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        }
    );
    println!();
}

fn validation_batch_stats<B: Backend>(
    output: &hydra_train::model::HydraOutput<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
) -> BatchStats {
    let target_actions = target_actions_from_policy_target(targets.policy_target.clone());
    let agreement = policy_agreement(
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        target_actions,
    );
    let breakdown = loss_fn.total_loss(output, targets);
    batch_stats_from_breakdown(agreement, &breakdown)
}

fn is_better_validation(summary: ValidationSummary, best: Option<BestValidation>) -> bool {
    match best {
        None => true,
        Some(best) => {
            summary.policy_loss < best.policy_loss
                || ((summary.policy_loss - best.policy_loss).abs() <= f64::EPSILON
                    && summary.agreement > best.agreement)
        }
    }
}

fn run_validation(
    model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &hydra_train::data::pipeline::DataManifest,
    device: &<ValidBackend as Backend>::Device,
    loss_fn: &HydraLoss<ValidBackend>,
    progress: Option<&ProgressBar>,
) -> Result<ValidationSummary, String> {
    let model_valid = model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut stats = ScalarAverages::default();
    let mut total_samples = 0usize;

    for buffer_result in stream_val_pass(manifest, loader_config, progress) {
        let buffer = buffer_result.map_err(|err| format!("validation stream failed: {err}"))?;
        for chunk in buffer.chunks(validation_batch_size) {
            if let Some(limit) = validation_sample_limit
                && total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk
            };
            if capped_chunk.is_empty() {
                break;
            }
            let Some((obs, targets)) = collate_samples::<ValidBackend>(capped_chunk, false, device)
            else {
                continue;
            };
            let output = model_valid.forward(obs);
            let batch_stats = validation_batch_stats(&output, &targets, loss_fn);
            stats.record_batch(batch_stats);
            total_samples += capped_chunk.len();
        }
        if let Some(limit) = validation_sample_limit
            && total_samples >= limit
        {
            break;
        }
    }

    if total_samples == 0 {
        Ok(ValidationSummary {
            total_loss: 0.0,
            policy_loss: 0.0,
            agreement: 0.0,
            samples: 0,
        })
    } else {
        let stats = stats.finalize();
        Ok(ValidationSummary {
            total_loss: stats.total_loss,
            policy_loss: stats.loss_policy,
            agreement: stats.policy_agreement,
            samples: total_samples,
        })
    }
}

fn run() -> Result<(), String> {
    let config_path = parse_args(env::args())?;
    let mut config = read_config(&config_path)?;
    if let Some((request, result_path)) = preflight_request_from_env()? {
        return run_probe_only(&config, request, &result_path);
    }
    validate_config(&config)?;
    configure_threads(config.num_threads)?;

    let resume = ResumeContext::load(&config)?;
    let session_start_global_step = resume.session_start_global_step;
    let artifacts = BcArtifactPaths::new(&config.output_dir, session_start_global_step);
    artifacts.create_dirs()?;

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
    };

    let scan_sources_len = if config.data_dir.is_file() {
        1
    } else {
        fs::read_dir(&config.data_dir)
            .map_err(|err| {
                format!(
                    "failed to read data dir {}: {err}",
                    config.data_dir.display()
                )
            })?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_type().ok().filter(|ft| ft.is_file()))
            .count()
    };
    let scan_pb = make_bar(
        scan_sources_len as u64,
        "[scan] [{bar:40.cyan/blue}] {pos}/{len} sources {msg}",
    )?;
    scan_pb.set_message("Scanning archives...".to_string());
    let manifest =
        scan_data_sources_with_progress(&config.data_dir, config.train_fraction, Some(&scan_pb))
            .map_err(|err| {
                format!(
                    "failed to scan MJAI data from {}: {err}",
                    config.data_dir.display()
                )
            })?;
    if manifest.counts_exact {
        scan_pb.finish_with_message(format!(
            "found {} train / {} val games",
            manifest.train_count, manifest.val_count
        ));
    } else {
        scan_pb.finish_with_message(format!(
            "found {} sources; exact game counts deferred to streaming load",
            manifest.sources.len()
        ));
    }

    let scheduler_warmup_steps = config.max_train_steps.map_or(
        BCTrainerConfig::default_learner().warmup_steps,
        |max_steps| max_steps.clamp(1, BCTrainerConfig::default_learner().warmup_steps.min(100)),
    );
    let train_cfg = BCTrainerConfig::default_learner()
        .with_batch_size(config.batch_size)
        .with_lr(BCTrainerConfig::default_learner().lr)
        .with_warmup_steps(scheduler_warmup_steps);
    train_cfg
        .validate()
        .map_err(|err| format!("invalid trainer config: {err}"))?;

    let device_name = device_label(&config.device);
    let model_config = HydraModelConfig::learner();
    let preflight = run_preflight(
        &config_path,
        &config,
        &model_config,
        &device_name,
        &artifacts,
    )?;
    config = apply_preflight_selection(&config, preflight.selected);
    let current_runtime = runtime_resume_contract(&config);
    if let Some(state) = resume.state.as_ref() {
        validate_resume_runtime_compatibility(state, current_runtime)?;
    }
    let train_device = train_device(&config.device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut model = model_config.init::<TrainBackend>(&train_device);
    if let Some(checkpoint_base) = resume.checkpoint_base.as_ref() {
        model = model
            .load_file(checkpoint_base, &recorder, &train_device)
            .map_err(|err| {
                format!(
                    "failed to load checkpoint {}: {err}",
                    checkpoint_base.display()
                )
            })?;
    }

    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let valid_loss_fn =
        HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let total_steps = schedule_total_steps(&config, session_start_global_step);
    let microbatch_size = train_microbatch_size(&config);
    let accum_steps = config.batch_size.div_ceil(microbatch_size).max(1);
    let mut best_validation = resume.best_validation();
    let mut global_step = session_start_global_step;
    let run_start = Instant::now();
    let mut last_log_step = global_step;
    let mut last_log_time = run_start;
    let mut tb = if config.tensorboard {
        Some(
            EventWriter::create(&artifacts.tb_session_dir)
                .map_err(|err| format!("tensorboard init: {err}"))?,
        )
    } else {
        None
    };

    let banner_stats = BannerStats {
        total_sources: manifest.sources.len(),
        total_games: manifest.total_games,
        train_count: manifest.train_count,
        val_count: manifest.val_count,
        accum_steps,
        counts_exact: manifest.counts_exact,
    };
    print_banner(
        &model_config,
        &config,
        &artifacts,
        &device_name,
        &banner_stats,
        scheduler_warmup_steps,
    );
    println!(
        "{} {} {} {}",
        "Preflight:".bold().cyan(),
        format!("train_mb={}", preflight.selected.train_microbatch_size).yellow(),
        format!("val_mb={}", preflight.selected.validation_microbatch_size).yellow(),
        if preflight.report.cache_hit {
            "cache=hit".green()
        } else {
            "cache=miss".yellow()
        }
    );

    resume.print_banner();

    for epoch in resume.start_epoch..config.num_epochs {
        let multi = MultiProgress::new();
        let load_label = phase_label("load", epoch, config.num_epochs);
        let train_label = phase_label("train", epoch, config.num_epochs);
        let steps_to_skip = resume.steps_to_skip_for_epoch(epoch);
        let load_pb = if manifest.counts_exact {
            multi.add(make_bar(
                manifest.train_count as u64,
                &format!("[{load_label}] [{{bar:30.cyan/blue}}] {{pos}}/{{len}} games {{msg}}"),
            )?)
        } else {
            multi.add(make_spinner(&format!(
                "[{load_label}] {{spinner:.cyan}} games={{pos}} {{msg}}"
            ))?)
        };
        let train_pb = if let Some(max_train_steps) = config.max_train_steps {
            multi.add(make_bar(
                max_train_steps as u64,
                &format!("[{train_label}] [{{bar:30.green/black}}] {{pos}}/{{len}} steps {{msg}}"),
            )?)
        } else {
            multi.add(make_spinner(&format!(
                "[{train_label}] {{spinner:.green}} steps={{pos}} {{msg}}"
            ))?)
        };

        let mut stats = ScalarAverages::default();
        let mut step_window = ScalarAverages::default();
        let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_current = 0usize;
        let mut pending_breakdowns: Vec<BatchStats> = Vec::new();
        let mut seen_samples = 0usize;
        let mut epoch_completed = true;
        let mut assumed_games_seen = 0usize;
        let mut remaining_games = manifest.train_count;
        let mut epoch_optimizer_steps = 0usize;

        for buffer_result in stream_train_epoch(&manifest, &loader_config, epoch, Some(&load_pb)) {
            let buffer = buffer_result.map_err(|err| format!("training stream failed: {err}"))?;
            if manifest.counts_exact {
                let assumed_games = remaining_games.min(config.buffer_games);
                remaining_games = remaining_games.saturating_sub(assumed_games);
                assumed_games_seen += assumed_games;
            }
            seen_samples += buffer.len();
            if manifest.counts_exact && assumed_games_seen > 0 {
                let estimated_steps = estimate_epoch_progress(
                    &manifest,
                    seen_samples,
                    assumed_games_seen,
                    epoch_optimizer_steps,
                    microbatch_size,
                    accum_steps,
                )
                .map(|progress| progress.estimated_total_optimizer_steps)
                .unwrap_or(1);
                if config.max_train_steps.is_none() {
                    train_pb.set_length(estimated_steps as u64);
                }
            } else if !manifest.counts_exact {
                load_pb.set_message(format!(
                    "samples={} steps={}",
                    seen_samples, stats.num_batches
                ));
            }

            for chunk in buffer.chunks(microbatch_size) {
                let lr = effective_lr(&train_cfg, global_step, total_steps);

                let mut completed_step = false;

                {
                    let Some((obs, targets)) =
                        collate_samples::<TrainBackend>(chunk, config.augment, &train_device)
                    else {
                        continue;
                    };
                    let output = model.forward(obs.clone());
                    let agreement = policy_agreement(
                        output.policy_logits.clone(),
                        targets.legal_mask.clone(),
                        target_actions_from_policy_target(targets.policy_target.clone()),
                    );
                    let breakdown = loss_fn.total_loss(&output, &targets);
                    let batch_stats = batch_stats_from_breakdown(agreement, &breakdown);
                    let grads = breakdown.total.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    accumulator.accumulate(&model, grads);
                    pending_breakdowns.push(batch_stats);
                    accum_current += 1;

                    if accum_current >= accum_steps {
                        let grads = accumulator.grads();
                        let drained = std::mem::take(&mut pending_breakdowns);
                        accum_current = 0;
                        if epoch_optimizer_steps < steps_to_skip {
                            epoch_optimizer_steps += 1;
                            train_pb.set_message(resumed_progress_message(
                                epoch_optimizer_steps,
                                steps_to_skip,
                            ));
                        } else {
                            model = optimizer.step(lr, model, grads);
                            for batch_stats in drained {
                                stats.record_batch(batch_stats);
                                step_window.record_batch(batch_stats);
                            }
                            epoch_optimizer_steps += 1;
                            global_step += 1;
                            train_pb.inc(1);
                            let lr_message =
                                lr_status_message(global_step, train_cfg.warmup_steps, lr);
                            train_pb.set_message(format_progress_message(
                                stats.total_loss / stats.num_batches.max(1) as f64,
                                stats.policy_agreement / stats.num_batches.max(1) as f64,
                                &lr_message,
                                steps_per_second(
                                    session_steps_completed(global_step, session_start_global_step),
                                    run_start.elapsed(),
                                ),
                            ));
                        }
                        completed_step = true;
                    }
                }

                if completed_step {
                    let session_step =
                        session_steps_completed(global_step, session_start_global_step);
                    let val_summary = if session_step > 0
                        && session_step.is_multiple_of(config.validate_every_n_steps)
                    {
                        multi
                            .println(format!(
                                "{} {}",
                                display_validation_scope_label(
                                    global_step,
                                    session_start_global_step,
                                    config.max_train_steps,
                                )
                                .bold()
                                .magenta(),
                                match validation_sample_limit(&config) {
                                    Some(limit) => format!("target_samples={limit}").yellow(),
                                    None => "target_samples=all".yellow(),
                                }
                            ))
                            .map_err(|err| {
                                format!("failed to print validation start summary: {err}")
                            })?;
                        let summary = run_validation(
                            &model,
                            &config,
                            &loader_config,
                            &manifest,
                            &train_device,
                            &valid_loss_fn,
                            None,
                        )?;
                        if is_better_validation(summary, best_validation) {
                            best_validation = Some(BestValidation {
                                policy_loss: summary.policy_loss,
                                agreement: summary.agreement,
                            });
                            save_checkpoint(
                                &model,
                                &artifacts.best_model_base,
                                global_step,
                                step_window.total_loss / step_window.num_batches.max(1) as f64,
                                Some(summary),
                            )?;
                        }
                        multi
                            .println(format!(
                                "{} {} {} {} {}",
                                display_validation_scope_label(
                                    global_step,
                                    session_start_global_step,
                                    config.max_train_steps,
                                )
                                .bold()
                                .magenta(),
                                format!("val_samples={}", summary.samples).yellow(),
                                format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
                                format!("val_total={:.4}", summary.total_loss).yellow(),
                                format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
                            ))
                            .map_err(|err| format!("failed to print validation summary: {err}"))?;
                        Some(summary)
                    } else {
                        None
                    };

                    if session_step > 0 && session_step.is_multiple_of(config.log_every_n_steps) {
                        let window_stats = std::mem::take(&mut step_window).finalize();
                        let lr_message = lr_status_message(global_step, train_cfg.warmup_steps, lr);
                        let window_steps = global_step.saturating_sub(last_log_step);
                        let step_rate = steps_per_second(window_steps, last_log_time.elapsed());
                        last_log_step = global_step;
                        last_log_time = Instant::now();

                        multi
                            .println(format!(
                                "{} {} {} {} {} {} {} {}",
                                display_step_label(
                                    global_step,
                                    session_start_global_step,
                                    config.max_train_steps,
                                )
                                .bold()
                                .cyan(),
                                format!("train_loss={:.4}", window_stats.total_loss).green(),
                                format!(
                                    "train_agree={:.2}%",
                                    window_stats.policy_agreement * 100.0
                                )
                                .green(),
                                if let Some(val_summary) = val_summary.as_ref() {
                                    format!(
                                        "val_ce={:.4} val_agree={:.2}%",
                                        val_summary.policy_loss,
                                        val_summary.agreement * 100.0
                                    )
                                } else {
                                    "val=skipped".to_string()
                                }
                                .bold()
                                .yellow(),
                                if let Some(best_validation) = best_validation {
                                    format!(
                                        "best_ce={:.4} best_agree={:.2}%",
                                        best_validation.policy_loss,
                                        best_validation.agreement * 100.0
                                    )
                                } else {
                                    "best=n/a".to_string()
                                }
                                .bold()
                                .magenta(),
                                epoch_progress_message_with_rate(
                                    estimate_epoch_progress(
                                        &manifest,
                                        seen_samples,
                                        assumed_games_seen,
                                        epoch_optimizer_steps,
                                        microbatch_size,
                                        accum_steps,
                                    ),
                                    Some(step_rate),
                                )
                                .white(),
                                format!("steps/s={step_rate:.2}").white(),
                                lr_message.white(),
                            ))
                            .map_err(|err| format!("failed to print train summary: {err}"))?;

                        if let Some(ref mut tb) = tb {
                            log_tensorboard(
                                tb,
                                global_step,
                                &window_stats,
                                val_summary,
                                lr,
                                best_validation,
                            )?;
                        }

                        let step_entry = StepLogEntry {
                            global_step,
                            epoch: epoch + 1,
                            lr,
                            train_total_loss: window_stats.total_loss,
                            train_policy_agreement: window_stats.policy_agreement,
                            train_loss_policy: window_stats.loss_policy,
                            train_loss_value: window_stats.loss_value,
                            train_loss_grp: window_stats.loss_grp,
                            train_loss_tenpai: window_stats.loss_tenpai,
                            train_loss_danger: window_stats.loss_danger,
                            train_loss_opp_next: window_stats.loss_opp_next,
                            train_loss_score_pdf: window_stats.loss_score_pdf,
                            train_loss_score_cdf: window_stats.loss_score_cdf,
                            val_total_loss: val_summary.map(|summary| summary.total_loss),
                            val_policy_loss: val_summary.map(|summary| summary.policy_loss),
                            val_policy_agreement: val_summary.map(|summary| summary.agreement),
                            best_val_policy_loss: best_validation.map(|best| best.policy_loss),
                            best_val_agreement: best_validation.map(|best| best.agreement),
                        };
                        append_step_log(&artifacts.step_log_path, &step_entry)?;
                    }

                    if session_step > 0
                        && session_step.is_multiple_of(config.checkpoint_every_n_steps)
                    {
                        let continuation = EpochContinuation {
                            next_epoch: epoch,
                            skip_optimizer_steps_in_epoch: epoch_optimizer_steps,
                            epoch_completed: false,
                        };
                        save_latest_checkpoint_and_state(
                            &artifacts,
                            &model,
                            global_step,
                            stats.total_loss / stats.num_batches.max(1) as f64,
                            best_validation,
                            &continuation,
                            current_runtime,
                        )?;
                    }

                    if reached_session_step_budget(
                        global_step,
                        session_start_global_step,
                        config.max_train_steps,
                    ) {
                        epoch_completed = false;
                        break;
                    }
                }
            }

            if reached_session_step_budget(
                global_step,
                session_start_global_step,
                config.max_train_steps,
            ) {
                epoch_completed = false;
                break;
            }
        }

        if accum_current > 0 && epoch_completed {
            let lr = effective_lr(&train_cfg, global_step, total_steps);
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            for batch_stats in pending_breakdowns.drain(..) {
                stats.record_batch(batch_stats);
                step_window.record_batch(batch_stats);
            }
            global_step += 1;
            train_pb.inc(1);
        }

        load_pb.finish_with_message("training data stream complete".to_string());
        let train_stats = stats.finalize();
        let final_steps = config.max_train_steps.unwrap_or(global_step).max(1) as u64;
        let final_lr = effective_lr(&train_cfg, global_step, total_steps);
        train_pb.set_length(final_steps);
        train_pb.finish_with_message(format_progress_message(
            train_stats.total_loss,
            train_stats.policy_agreement,
            &lr_status_message(global_step, train_cfg.warmup_steps, final_lr),
            steps_per_second(
                session_steps_completed(global_step, session_start_global_step),
                run_start.elapsed(),
            ),
        ));

        let continuation = EpochContinuation {
            next_epoch: if epoch_completed { epoch + 1 } else { epoch },
            skip_optimizer_steps_in_epoch: if epoch_completed {
                0
            } else {
                epoch_optimizer_steps
            },
            epoch_completed,
        };
        save_latest_checkpoint_and_state(
            &artifacts,
            &model,
            global_step,
            train_stats.total_loss,
            best_validation,
            &continuation,
            current_runtime,
        )?;

        if !continuation.epoch_completed {
            println!(
                "{} {}",
                "Paused BC training".bold().cyan(),
                paused_training_message(&continuation).yellow(),
            );
            break;
        }

        let should_validate =
            (epoch + 1) % config.validation_every_n_epochs == 0 || epoch + 1 == config.num_epochs;
        let val_summary = if should_validate {
            println!(
                "{} {}",
                "validation @ epoch end".bold().magenta(),
                match validation_sample_limit(&config) {
                    Some(limit) => format!("target_samples={limit}").yellow(),
                    None => "target_samples=all".yellow(),
                }
            );
            let summary = run_validation(
                &model,
                &config,
                &loader_config,
                &manifest,
                &train_device,
                &valid_loss_fn,
                None,
            )?;
            if is_better_validation(summary, best_validation) {
                best_validation = Some(BestValidation {
                    policy_loss: summary.policy_loss,
                    agreement: summary.agreement,
                });
                save_checkpoint(
                    &model,
                    &artifacts.best_model_base,
                    epoch + 1,
                    train_stats.total_loss,
                    Some(summary),
                )?;
            }
            println!(
                "{} {} {} {} {}",
                "validation @ epoch end".bold().magenta(),
                format!("val_samples={}", summary.samples).yellow(),
                format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
                format!("val_total={:.4}", summary.total_loss).yellow(),
                format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
            );
            Some(summary)
        } else {
            None
        };

        if let Some(ref mut tb) = tb {
            log_tensorboard(
                tb,
                epoch + 1,
                &train_stats,
                val_summary,
                final_lr,
                best_validation,
            )?;
        }

        let entry = EpochLogEntry {
            epoch: epoch + 1,
            global_step,
            lr: final_lr,
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
            val_total_loss: val_summary.as_ref().map(|summary| summary.total_loss),
            val_policy_loss: val_summary.as_ref().map(|summary| summary.policy_loss),
            val_policy_agreement: val_summary.as_ref().map(|summary| summary.agreement),
            best_val_policy_loss: best_validation.map(|best| best.policy_loss),
            best_val_agreement: best_validation.map(|best| best.agreement),
            num_batches: train_stats.num_batches,
        };
        append_training_log(&artifacts.training_log_path, &entry)?;

        let lr_message = lr_status_message(global_step, train_cfg.warmup_steps, final_lr);

        println!(
            "{} {} {} {} {} {}",
            phase_label("epoch", epoch, config.num_epochs).bold().cyan(),
            format!("train_loss={:.4}", train_stats.total_loss).green(),
            format!("train_agree={:.2}%", train_stats.policy_agreement * 100.0).green(),
            if let Some(val_summary) = val_summary.as_ref() {
                format!(
                    "val_ce={:.4} val_agree={:.2}% val_samples={}",
                    val_summary.policy_loss,
                    val_summary.agreement * 100.0,
                    val_summary.samples
                )
            } else {
                "val=skipped".to_string()
            }
            .bold()
            .yellow(),
            if let Some(best_validation) = best_validation {
                format!(
                    "best_ce={:.4} best_agree={:.2}%",
                    best_validation.policy_loss,
                    best_validation.agreement * 100.0
                )
            } else {
                "best=n/a".to_string()
            }
            .bold()
            .magenta(),
            lr_message.white(),
        );

        if reached_session_step_budget(
            global_step,
            session_start_global_step,
            config.max_train_steps,
        ) {
            break;
        }
    }

    println!(
        "{} {}",
        "Finished BC training. Best validation policy CE:"
            .bold()
            .cyan(),
        if let Some(best_validation) = best_validation {
            format!(
                "{:.4} (agree {:.2}%)",
                best_validation.policy_loss,
                best_validation.agreement * 100.0
            )
        } else {
            "n/a".to_string()
        }
        .bold()
        .green()
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train::training::bc::policy_agreement_counts;

    #[test]
    fn parse_args_accepts_single_config_path() {
        let args = vec!["train".to_string(), "config.yaml".to_string()];
        let parsed = parse_args(args).expect("single config arg should parse");
        assert_eq!(parsed, PathBuf::from("config.yaml"));
    }

    #[test]
    fn parse_args_rejects_missing_config() {
        let args = vec!["train".to_string()];
        let err = parse_args(args).expect_err("missing config should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn parse_args_rejects_extra_args() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "extra".to_string(),
        ];
        let err = parse_args(args).expect_err("extra args should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn read_config_applies_defaults() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_config_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 3
"#;
        fs::write(&base, yaml).expect("write config");
        let cfg = read_config(&base).expect("read config");
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/data"));
        assert_eq!(cfg.output_dir, PathBuf::from("/tmp/out"));
        assert_eq!(cfg.num_epochs, 3);
        assert_eq!(cfg.batch_size, 2048);
        assert!(cfg.microbatch_size.is_none());
        assert!(cfg.validation_microbatch_size.is_none());
        assert!((cfg.train_fraction - 0.9).abs() < f32::EPSILON);
        assert!(cfg.augment);
        assert_eq!(cfg.seed, 0);
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.buffer_games, 50_000);
        assert_eq!(cfg.buffer_samples, 32_768);
        assert!(cfg.num_threads.is_none());
        assert!(cfg.tensorboard);
        assert_eq!(cfg.archive_queue_bound, 128);
        assert_eq!(cfg.validation_every_n_epochs, 1);
        assert_eq!(cfg.max_skip_logs_per_source, 32);
        assert!(cfg.max_validation_batches.is_none());
        assert_eq!(cfg.max_validation_samples, Some(8_192));
        assert!(cfg.advanced_loss.is_none());
        fs::remove_file(base).ok();
    }

    #[test]
    fn bc_artifact_paths_use_bc_subdir_and_unique_tb_session() {
        let paths = BcArtifactPaths::new(Path::new("/tmp/out"), 42);
        assert_eq!(paths.root, PathBuf::from("/tmp/out/bc"));
        assert_eq!(
            paths.latest_model_base,
            PathBuf::from("/tmp/out/bc/latest_model")
        );
        assert_eq!(
            paths.best_model_base,
            PathBuf::from("/tmp/out/bc/best_model")
        );
        assert_eq!(
            paths.latest_state_path,
            PathBuf::from("/tmp/out/bc/latest_state.yaml")
        );
        assert_eq!(
            paths.training_log_path,
            PathBuf::from("/tmp/out/bc/training_log.jsonl")
        );
        assert_eq!(
            paths.step_log_path,
            PathBuf::from("/tmp/out/bc/step_log.jsonl")
        );
        assert!(
            paths
                .tb_session_dir
                .starts_with(Path::new("/tmp/out/bc/tb"))
        );
        assert_ne!(paths.tb_session_dir, paths.tb_root);
    }

    #[test]
    fn checkpoint_base_from_path_strips_mpk_only() {
        assert_eq!(
            checkpoint_base_from_path(Path::new("/tmp/out/bc/latest_model.mpk")),
            PathBuf::from("/tmp/out/bc/latest_model")
        );
        assert_eq!(
            checkpoint_base_from_path(Path::new("/tmp/out/bc/latest_model")),
            PathBuf::from("/tmp/out/bc/latest_model")
        );
    }

    #[test]
    fn latest_state_path_is_only_available_for_latest_model() {
        assert_eq!(
            latest_state_path_for_checkpoint_base(Path::new("/tmp/out/bc/latest_model")),
            Some(PathBuf::from("/tmp/out/bc/latest_state.yaml"))
        );
        assert_eq!(
            latest_state_path_for_checkpoint_base(Path::new("/tmp/out/bc/best_model")),
            None
        );
    }

    #[test]
    fn session_step_budget_is_relative_to_resume_point() {
        assert_eq!(session_steps_completed(1250, 1000), 250);
        assert!(reached_session_step_budget(1200, 1000, Some(200)));
        assert!(!reached_session_step_budget(1199, 1000, Some(200)));
        assert_eq!(
            display_step_label(1200, 1000, Some(200)),
            "step 200/200 global=1200"
        );
        assert_eq!(
            display_validation_scope_label(1100, 1000, Some(200)),
            "validation @ step 100/200 global=1100"
        );
    }

    #[test]
    fn schedule_total_steps_extends_from_resume_global_step() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(16),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 25,
            validate_every_n_steps: 200,
            checkpoint_every_n_steps: 200,
            max_train_steps: Some(1000),
            max_validation_batches: None,
            max_validation_samples: Some(8192),
            preflight: PreflightConfig::default(),
        };
        assert_eq!(schedule_total_steps(&cfg, 0), 1000);
        assert_eq!(schedule_total_steps(&cfg, 400), 1400);
    }

    #[test]
    fn resume_state_yaml_roundtrip_preserves_fields() {
        let state = build_resume_state(
            0,
            37,
            137,
            Some(BestValidation {
                policy_loss: 1.23,
                agreement: 0.45,
            }),
            test_runtime_resume_contract(2048, 256, 256),
        );
        let yaml = serde_yaml::to_string(&state).expect("serialize state");
        let parsed: BcResumeState = serde_yaml::from_str(&yaml).expect("parse state");
        assert_eq!(parsed.schema_version, 2);
        assert_eq!(
            parsed.resume_semantics,
            ResumeSemantics::ReplaySkippedStepsFreshOptimizer
        );
        assert_eq!(parsed.next_epoch, 0);
        assert_eq!(parsed.skip_optimizer_steps_in_epoch, 37);
        assert_eq!(parsed.global_step, 137);
        assert_eq!(parsed.best_validation, state.best_validation);
        assert_eq!(parsed.runtime, state.runtime);
    }

    #[test]
    fn resume_banner_message_mentions_replay_when_needed() {
        let state = build_resume_state(
            2,
            137,
            2048,
            Some(BestValidation {
                policy_loss: 1.5,
                agreement: 0.41,
            }),
            test_runtime_resume_contract(2048, 256, 128),
        );
        assert_eq!(
            resume_banner_message(&state),
            "global_step=2048 semantics=ReplaySkippedStepsFreshOptimizer replaying 137 completed optimizer steps from epoch 3 before new updates runtime=train_mb:256 val_mb:128 accum_steps:8"
        );
    }

    #[test]
    fn resume_banner_message_mentions_immediate_updates_when_no_replay() {
        let state = build_resume_state(
            1,
            0,
            500,
            None,
            test_runtime_resume_contract(2048, 512, 256),
        );
        assert_eq!(
            resume_banner_message(&state),
            "global_step=500 semantics=ReplaySkippedStepsFreshOptimizer resuming at epoch 2 with new updates immediately runtime=train_mb:512 val_mb:256 accum_steps:4"
        );
    }

    #[test]
    fn paused_training_message_spells_out_resume_contract() {
        let continuation = EpochContinuation {
            next_epoch: 0,
            skip_optimizer_steps_in_epoch: 88,
            epoch_completed: false,
        };
        assert_eq!(
            paused_training_message(&continuation),
            "resume_epoch=1 replay_steps_in_epoch=88 exact_sample_cursor=not_restored partial_epoch_requires_matching_runtime"
        );
    }

    #[test]
    fn partial_epoch_resume_rejects_runtime_mismatch() {
        let state = build_resume_state(
            0,
            12,
            400,
            None,
            test_runtime_resume_contract(2048, 256, 128),
        );
        let err = validate_resume_runtime_compatibility(
            &state,
            test_runtime_resume_contract(2048, 512, 128),
        )
        .expect_err("partial epoch resume should fail when runtime differs");
        assert!(err.contains("partial-epoch resume requires identical runtime contract"));
    }

    #[test]
    fn epoch_boundary_resume_allows_runtime_change_with_same_batch_size() {
        let state = build_resume_state(
            1,
            0,
            400,
            None,
            test_runtime_resume_contract(2048, 256, 128),
        );
        validate_resume_runtime_compatibility(&state, test_runtime_resume_contract(2048, 512, 256))
            .expect("epoch-boundary resume should allow new runtime contract");
    }

    #[test]
    fn read_config_supports_yaml_and_json_during_transition() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_yaml_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let json_path = dir.join(format!(
            "hydra_train_json_{}_{}.json",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
"#;
        let json = r#"{
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "num_epochs": 1
        }"#;
        fs::write(&yaml_path, yaml).expect("write yaml config");
        fs::write(&json_path, json).expect("write json config");
        assert_eq!(read_config(&yaml_path).expect("yaml config").num_epochs, 1);
        assert_eq!(read_config(&json_path).expect("json config").num_epochs, 1);
        fs::remove_file(yaml_path).ok();
        fs::remove_file(json_path).ok();
    }

    #[test]
    fn estimate_epoch_progress_returns_none_without_exact_counts() {
        let manifest = hydra_train::data::pipeline::DataManifest {
            sources: vec![],
            total_games: 0,
            train_count: 100,
            val_count: 0,
            counts_exact: false,
        };
        assert_eq!(
            estimate_epoch_progress(&manifest, 10_000, 10, 25, 64, 4),
            None
        );
    }

    #[test]
    fn estimate_epoch_progress_computes_remaining_steps() {
        let manifest = hydra_train::data::pipeline::DataManifest {
            sources: vec![],
            total_games: 100,
            train_count: 100,
            val_count: 0,
            counts_exact: true,
        };
        let progress = estimate_epoch_progress(&manifest, 12_800, 10, 40, 64, 4)
            .expect("exact counts should yield estimate");
        assert_eq!(progress.completed_optimizer_steps, 40);
        assert_eq!(progress.estimated_total_optimizer_steps, 500);
        assert_eq!(progress.estimated_remaining_optimizer_steps, 460);
        assert!((progress.completion_fraction - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn epoch_progress_message_formats_estimate_and_pending() {
        assert_eq!(
            epoch_progress_message_with_rate(None, None),
            "epoch=pending"
        );
        assert_eq!(
            epoch_progress_message_with_rate(
                Some(EpochProgressEstimate {
                    completed_optimizer_steps: 200,
                    estimated_total_optimizer_steps: 500,
                    estimated_remaining_optimizer_steps: 300,
                    completion_fraction: 0.4,
                }),
                None,
            ),
            "epoch=40.0% epoch_left≈300 steps"
        );
    }

    #[test]
    fn format_rough_duration_prefers_human_sized_units() {
        assert_eq!(format_rough_duration(12.2), "~12s");
        assert_eq!(format_rough_duration(125.0), "~2m5s");
        assert_eq!(format_rough_duration(3720.0), "~1h2m");
    }

    #[test]
    fn epoch_progress_message_with_rate_appends_rough_eta() {
        let message = epoch_progress_message_with_rate(
            Some(EpochProgressEstimate {
                completed_optimizer_steps: 200,
                estimated_total_optimizer_steps: 500,
                estimated_remaining_optimizer_steps: 300,
                completion_fraction: 0.4,
            }),
            Some(2.0),
        );
        assert_eq!(message, "epoch=40.0% epoch_left≈300 steps rough_eta=~2m30s");
    }

    #[test]
    fn better_validation_prefers_lower_policy_loss_then_higher_agreement() {
        let summary = ValidationSummary {
            total_loss: 2.0,
            policy_loss: 1.0,
            agreement: 0.35,
            samples: 8192,
        };
        assert!(is_better_validation(summary, None));

        let best = BestValidation {
            policy_loss: 1.1,
            agreement: 0.60,
        };
        assert!(is_better_validation(summary, Some(best)));

        let tied = ValidationSummary {
            total_loss: 2.1,
            policy_loss: 1.0,
            agreement: 0.40,
            samples: 8192,
        };
        assert!(is_better_validation(
            tied,
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.39
            })
        ));
        assert!(!is_better_validation(
            tied,
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.41
            })
        ));
    }

    #[test]
    fn validation_microbatch_and_sample_limit_fallbacks_work() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: Some(32),
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
        };

        assert_eq!(train_microbatch_size(&cfg), 64);
        assert_eq!(validation_microbatch_size(&cfg), 64);
        assert_eq!(validation_sample_limit(&cfg), Some(2048));

        let cfg = TrainConfig {
            validation_microbatch_size: Some(32),
            max_validation_batches: Some(32),
            max_validation_samples: Some(1500),
            ..cfg
        };

        assert_eq!(validation_microbatch_size(&cfg), 32);
        assert_eq!(validation_sample_limit(&cfg), Some(1500));
    }

    #[test]
    fn validate_config_rejects_zero_validation_microbatch_and_samples() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(0),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: Some(0),
            preflight: PreflightConfig::default(),
        };

        let err = validate_config(&cfg).expect_err("zero validation controls should fail");
        assert!(
            err.contains("max_validation_samples") || err.contains("validation_microbatch_size")
        );
    }

    #[test]
    fn build_loss_config_defaults_match_baseline() {
        let loss = build_loss_config(None).expect("default loss config should build");
        assert!((loss.w_safety_residual - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_allows_safety_residual_only() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(0.1),
            ..Default::default()
        };
        let loss = build_loss_config(Some(&advanced)).expect("safety residual should be allowed");
        assert!((loss.w_safety_residual - 0.1).abs() < 1e-6);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_rejects_negative_safety_residual() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(-0.1),
            ..Default::default()
        };
        let err =
            build_loss_config(Some(&advanced)).expect_err("negative safety residual should fail");
        assert!(err.contains("invalid loss config"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_activation() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("belief fields should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_even_at_zero() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.0),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("blocked belief fields key should be rejected even at zero");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_mixture_weight_activation() {
        let advanced = AdvancedLossConfig {
            mixture_weight: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("mixture weight should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.mixture_weight"));
    }

    #[test]
    fn build_loss_config_rejects_opponent_hand_type_activation() {
        let advanced = AdvancedLossConfig {
            opponent_hand_type: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("opponent hand type should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.opponent_hand_type"));
    }

    #[test]
    fn build_loss_config_rejects_delta_q_activation() {
        let advanced = AdvancedLossConfig {
            delta_q: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("delta_q should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.delta_q"));
    }

    #[test]
    fn read_config_rejects_unknown_advanced_loss_field() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_bad_advanced_loss_{}_{}.json",
            std::process::id(),
            default_seed()
        ));
        let json = r#"{
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "num_epochs": 3,
            "advanced_loss": {
                "not_a_real_field": 0.1
            }
        }"#;
        fs::write(&base, json).expect("write config");
        let err = read_config(&base).expect_err("unknown advanced loss field should fail");
        assert!(err.contains("not_a_real_field"));
        fs::remove_file(base).ok();
    }

    #[test]
    fn train_device_prefers_env_override_then_config() {
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cpu);
        assert_eq!(train_device("cuda:2"), LibTorchDevice::Cuda(2));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cuda:0");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cuda(0));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cpu");
        }
        assert_eq!(train_device("cuda:3"), LibTorchDevice::Cpu);

        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }

    #[test]
    #[should_panic(expected = "unsupported HYDRA_TRAIN_DEVICE")]
    fn train_device_rejects_invalid_env_value() {
        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "vulkan");
        }
        let _ = train_device("cpu");
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }

    #[test]
    fn validation_agreement_is_sample_weighted_across_chunks() {
        let device: <burn::backend::ndarray::NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::from_floats(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
            &device,
        );
        let mask = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::ones([5, 2], &device);
        let targets = Tensor::<burn::backend::ndarray::NdArray<f32>, 1, Int>::from_ints(
            [0, 1, 1, 1, 1],
            &device,
        );

        let (chunk1_correct, chunk1_total) = policy_agreement_counts(
            logits.clone().slice([0..4, 0..2]),
            mask.clone().slice([0..4, 0..2]),
            targets.clone().slice(0..4),
        );
        let (chunk2_correct, chunk2_total) = policy_agreement_counts(
            logits.slice([4..5, 0..2]),
            mask.slice([4..5, 0..2]),
            targets.slice(4..5),
        );

        let weighted =
            (chunk1_correct + chunk2_correct) as f64 / (chunk1_total + chunk2_total) as f64;
        let naive_chunk_average = ((chunk1_correct as f64 / chunk1_total as f64)
            + (chunk2_correct as f64 / chunk2_total as f64))
            / 2.0;

        assert!((weighted - 0.4).abs() < 1e-12);
        assert!((naive_chunk_average - 0.625).abs() < 1e-12);
    }

    #[test]
    fn phase_label_hides_redundant_single_epoch_denominator() {
        assert_eq!(phase_label("train", 0, 1), "train");
        assert_eq!(phase_label("train", 1, 3), "train 2/3");
    }

    #[test]
    fn lr_status_message_marks_warmup_and_cosine() {
        assert_eq!(
            lr_status_message(25, 100, 1.25e-4),
            "lr=1.25e-4 warmup 25/100"
        );
        assert_eq!(lr_status_message(100, 100, 2.50e-4), "lr=2.50e-4 cosine");
    }

    #[test]
    fn steps_per_second_and_progress_message_are_stable() {
        assert_eq!(steps_per_second(0, Duration::from_secs(1)), 0.0);
        assert_eq!(steps_per_second(10, Duration::from_secs(0)), 0.0);
        assert!((steps_per_second(10, Duration::from_secs(2)) - 5.0).abs() < 1e-12);
        assert_eq!(
            format_progress_message(3.0, 0.25, "lr=1.00e-4 cosine", 5.5),
            "loss=3.0000 agree=25.00% steps/s=5.50 lr=1.00e-4 cosine"
        );
    }
}
