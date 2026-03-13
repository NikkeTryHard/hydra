use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use burn::backend::libtorch::LibTorchDevice;
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;

use hydra_train::data::pipeline::{
    scan_data_sources_with_progress, stream_train_epoch, stream_val_pass, DataManifest,
    StreamingLoaderConfig,
};
use hydra_train::data::sample::{collate_samples, MjaiSample};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::preflight::{
    candidate_ladder, resolve_runtime_config, ExplicitSettings, HardwareFingerprint,
    PreflightCacheEntry, PreflightCacheKey, ProbeKind, ProbeResult, ProbeStatus,
    SelectedRuntimeConfig, WorkloadFingerprint,
};
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{write_preflight_cache, BcArtifactPaths, PreflightPaths};
use super::config::{
    train_device, trainer_config_from_train_config, AdvancedLossConfig, ProbeChildRequest,
    ProbeCliRequest, TrainConfig,
};
use super::loss_policy::build_loss_config;
use super::presentation::{format_probe_status_line, make_bar, make_spinner};
use super::schedule::effective_lr;
use super::validation::validation_batch_stats;
use super::{TrainBackend, ValidBackend};

pub(super) struct PreflightRuntime {
    pub(super) selected: SelectedRuntimeConfig,
    pub(super) train_probe_results: Vec<ProbeResult>,
    pub(super) validation_probe_results: Vec<ProbeResult>,
    pub(super) explicit: ExplicitSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TrainingRuntimeWarning {
    pub(super) configured: SelectedRuntimeConfig,
    pub(super) saved: SelectedRuntimeConfig,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ProbeRequest {
    pub(super) kind: ProbeKind,
    pub(super) candidate_microbatch: usize,
    pub(super) warmup_steps: usize,
    pub(super) measure_steps: usize,
}

pub(super) fn read_preflight_cache(path: &Path) -> Result<PreflightCacheEntry, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read preflight cache {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse preflight cache {}: {err}", path.display()))
}

pub(super) fn load_saved_preflight_entry(
    artifacts: &BcArtifactPaths,
) -> Result<Option<PreflightCacheEntry>, String> {
    let path = PreflightPaths::new(artifacts).cache_path;
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(read_preflight_cache(&path)?))
}

pub(super) fn configured_runtime_selection(config: &TrainConfig) -> SelectedRuntimeConfig {
    resolve_runtime_config(
        config.batch_size,
        ExplicitSettings {
            train_microbatch_explicit: config.microbatch_size.is_some(),
            validation_microbatch_explicit: config.validation_microbatch_size.is_some(),
        },
        config.microbatch_size.unwrap_or(config.batch_size),
        config
            .validation_microbatch_size
            .unwrap_or_else(|| config.microbatch_size.unwrap_or(config.batch_size)),
    )
}

pub(super) fn training_runtime_warning(
    config: &TrainConfig,
    saved: Option<&PreflightCacheEntry>,
    expected_cache_key: &PreflightCacheKey,
) -> Option<TrainingRuntimeWarning> {
    let saved = saved?;
    if &saved.cache_key != expected_cache_key {
        return None;
    }
    if config.microbatch_size.is_none() && config.validation_microbatch_size.is_none() {
        return None;
    }
    let configured = configured_runtime_selection(config);
    if configured == saved.selected {
        None
    } else {
        Some(TrainingRuntimeWarning {
            configured,
            saved: saved.selected,
        })
    }
}

pub(super) fn advanced_loss_signature(config: Option<&AdvancedLossConfig>) -> String {
    match config {
        Some(config) => serde_json::to_string(config)
            .unwrap_or_else(|_| "advanced_loss:unserializable".to_string()),
        None => "advanced_loss:none".to_string(),
    }
}

pub(super) fn workload_fingerprint(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
) -> WorkloadFingerprint {
    WorkloadFingerprint {
        batch_size: config.batch_size,
        augment: config.augment,
        train_fraction_bits: config.train_fraction.to_bits(),
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        num_threads: config.num_threads,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        max_validation_batches: config.max_validation_batches,
        max_validation_samples: config.max_validation_samples,
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

pub(super) fn hardware_fingerprint(device_label: &str) -> HardwareFingerprint {
    HardwareFingerprint {
        device_label: device_label.to_string(),
        backend: "burn-libtorch".to_string(),
    }
}

pub(super) fn preflight_cache_key(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
) -> PreflightCacheKey {
    PreflightCacheKey {
        hardware: hardware_fingerprint(device_label),
        workload: workload_fingerprint(config, model_config),
    }
}

pub(super) fn probe_kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
    }
}

pub(super) fn probe_request_from_cli(
    config: &TrainConfig,
    probe: Option<ProbeCliRequest>,
) -> Result<Option<ProbeRequest>, String> {
    let Some(probe) = probe else {
        return Ok(None);
    };
    let warmup_steps = probe.warmup_steps.unwrap_or(config.preflight.warmup_steps);
    let measure_steps = probe
        .measure_steps
        .unwrap_or(config.preflight.measure_steps);
    if probe.candidate_microbatch == 0 {
        return Err("--probe-candidate-microbatch must be greater than 0".to_string());
    }
    if warmup_steps == 0 {
        return Err("--probe-warmup-steps must be greater than 0".to_string());
    }
    if measure_steps == 0 {
        return Err("--probe-measure-steps must be greater than 0".to_string());
    }
    Ok(Some(ProbeRequest {
        kind: probe.kind,
        candidate_microbatch: probe.candidate_microbatch,
        warmup_steps,
        measure_steps,
    }))
}

pub(super) fn probe_child_request_from_cli(
    child: Option<ProbeChildRequest>,
) -> Result<Option<(ProbeRequest, PathBuf)>, String> {
    let Some(child) = child else {
        return Ok(None);
    };
    let request = ProbeRequest {
        kind: child.request.kind,
        candidate_microbatch: child.request.candidate_microbatch,
        warmup_steps: child
            .request
            .warmup_steps
            .ok_or_else(|| "internal probe child missing resolved warmup steps".to_string())?,
        measure_steps: child
            .request
            .measure_steps
            .ok_or_else(|| "internal probe child missing resolved measure steps".to_string())?,
    };
    Ok(Some((request, child.result_path)))
}

pub(super) fn probe_candidate_ceiling(request: ProbeRequest) -> usize {
    request.candidate_microbatch.max(1)
}

pub(super) fn probe_only_candidate_ladder(
    config: &TrainConfig,
    request: ProbeRequest,
) -> Vec<usize> {
    let ceiling = probe_candidate_ceiling(request);
    let mut candidates: Vec<usize> = candidate_ladder(&config.preflight, config.batch_size)
        .into_iter()
        .filter(|candidate| *candidate <= ceiling)
        .collect();
    if candidates.is_empty() {
        candidates.push(ceiling);
    }
    candidates
}

pub(super) fn write_probe_result(path: &Path, result: &ProbeResult) -> Result<(), String> {
    let json = serde_json::to_string(result)
        .map_err(|err| format!("failed to serialize probe result {}: {err}", path.display()))?;
    fs::write(path, json)
        .map_err(|err| format!("failed to write probe result {}: {err}", path.display()))
}

pub(super) fn read_probe_result(path: &Path) -> Result<ProbeResult, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read probe result {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse probe result {}: {err}", path.display()))
}

pub(super) fn probe_result_path(
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

pub(super) fn measure_samples_per_second(samples: usize, elapsed: Duration) -> f64 {
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

fn child_scan_progress_template(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "[probe-train-scan] {spinner:.cyan} {msg}",
        ProbeKind::Validation => "[probe-val-scan] {spinner:.cyan} {msg}",
    }
}

fn child_step_progress_template(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "[probe-train] {spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
        ProbeKind::Validation => "[probe-val] {spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    }
}

fn probe_phase_message(
    kind: ProbeKind,
    microbatch_size: usize,
    completed_steps: usize,
    request: ProbeRequest,
) -> String {
    if completed_steps < request.warmup_steps {
        format!(
            "{} candidate_mb={} phase=warmup step {}/{}",
            probe_kind_name(kind),
            microbatch_size,
            completed_steps + 1,
            request.warmup_steps.max(1)
        )
    } else {
        format!(
            "{} candidate_mb={} phase=measure step {}/{}",
            probe_kind_name(kind),
            microbatch_size,
            completed_steps + 1 - request.warmup_steps,
            request.measure_steps.max(1)
        )
    }
}

pub(super) fn probe_train_candidate(
    config: &TrainConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let train_cfg = trainer_config_from_train_config(config);
    let mut model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let microbatch_size = request.candidate_microbatch.min(config.batch_size).max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut completed_steps = 0usize;
    let mut pending_samples = std::collections::VecDeque::new();
    let mut measure_start = None;
    let step_pb = make_bar(
        target_steps as u64,
        child_step_progress_template(ProbeKind::Train),
    )?;

    for buffer_result in stream_train_epoch(manifest, loader_config, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight train stream failed: {err}"))?;
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let logical_batch_len = logical_batch.len().max(1) as f32;
            let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
                GradientsAccumulator::new();
            for chunk in logical_batch.chunks(microbatch_size) {
                let Some((obs, targets)) =
                    collate_samples::<TrainBackend>(chunk, config.augment, train_device)
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
            let lr = effective_lr(&train_cfg, completed_steps, target_steps.max(1));
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            step_pb.set_message(probe_phase_message(
                ProbeKind::Train,
                microbatch_size,
                completed_steps,
                request,
            ));
            step_pb.inc(1);
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                step_pb.finish_and_clear();
                return Ok(measure_samples_per_second(
                    request.measure_steps.max(1) * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    step_pb.finish_and_clear();

    Err(format!(
        "not enough train data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

pub(super) fn probe_validation_candidate(
    config: &TrainConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let model_valid = model.valid();
    let loss_fn = HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let microbatch_size = request.candidate_microbatch.max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut completed_steps = 0usize;
    let mut measure_start = None;
    let step_pb = make_bar(
        target_steps as u64,
        child_step_progress_template(ProbeKind::Validation),
    )?;

    for buffer_result in stream_val_pass(manifest, loader_config, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight validation stream failed: {err}"))?;
        for chunk in buffer.chunks(microbatch_size) {
            let Some((obs, targets)) = collate_samples::<ValidBackend>(chunk, false, train_device)
            else {
                continue;
            };
            let output = model_valid.forward(obs);
            let _ = validation_batch_stats(chunk.len(), &output, &targets, &loss_fn);
            step_pb.set_message(probe_phase_message(
                ProbeKind::Validation,
                microbatch_size,
                completed_steps,
                request,
            ));
            step_pb.inc(1);
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                step_pb.finish_and_clear();
                return Ok(measure_samples_per_second(
                    request.measure_steps.max(1) * microbatch_size,
                    elapsed,
                ));
            }
        }
    }

    step_pb.finish_and_clear();

    Err(format!(
        "not enough validation data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

pub(super) fn run_probe_only(
    config: &TrainConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    if let Some(num_threads) = config.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|err| format!("failed to configure rayon threads for probe child: {err}"))?;
    }
    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
    };
    let scan_pb = make_spinner(child_scan_progress_template(request.kind))?;
    scan_pb.set_message(format!(
        "{} candidate_mb={} scanning dataset",
        probe_kind_name(request.kind),
        request.candidate_microbatch
    ));
    let manifest =
        scan_data_sources_with_progress(&config.data_dir, config.train_fraction, Some(&scan_pb))
            .map_err(|err| {
                format!(
                    "failed to scan preflight data from {}: {err}",
                    config.data_dir.display()
                )
            })?;
    scan_pb.finish_with_message(
        format!(
            "{} candidate_mb={} scan complete",
            probe_kind_name(request.kind),
            request.candidate_microbatch
        )
        .green()
        .to_string(),
    );
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

pub(super) fn run_probe_child_mode(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
) -> Result<bool, String> {
    let Some((request, result_path)) = probe_child_request_from_cli(child)? else {
        return Ok(false);
    };
    run_probe_only(config, request, &result_path)?;
    Ok(true)
}

pub(super) fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<ProbeResult, String> {
    fs::remove_file(result_path).ok();
    let output =
        Command::new(env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?)
            .arg(config_path)
            .arg("--probe-kind")
            .arg(probe_kind_name(request.kind))
            .arg("--probe-candidate-microbatch")
            .arg(request.candidate_microbatch.to_string())
            .arg("--probe-warmup-steps")
            .arg(request.warmup_steps.to_string())
            .arg("--probe-measure-steps")
            .arg(request.measure_steps.to_string())
            .arg("--probe-result-path")
            .arg(result_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
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

pub(super) fn format_probe_attempt_message(
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

pub(super) fn format_probe_result_summary(result: &ProbeResult) -> String {
    format_probe_status_line(result)
}

pub(super) fn format_probe_selection_summary(kind: ProbeKind, result: &ProbeResult) -> String {
    format!(
        "selected {} microbatch={} throughput={:.2} samples/s",
        probe_kind_name(kind),
        result.candidate_microbatch,
        result.measured_samples_per_second.unwrap_or(0.0)
    )
}

pub(super) fn best_probe_result(results: &[ProbeResult]) -> Option<&ProbeResult> {
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

pub(super) fn probe_candidate_ladder(
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
    println!(
        "{} {}",
        "Preflight ladder:".bold().cyan(),
        format!(
            "kind={} candidates={:?} required_successes={}",
            probe_kind_name(kind),
            candidate_list,
            config.preflight.required_successes.max(1)
        )
        .yellow()
    );
    let progress = make_bar(
        (candidate_list.len() * config.preflight.required_successes.max(1)) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;

    for candidate in candidate_list {
        let mut stable = true;
        let stable_start = results.len();
        let attempts = config.preflight.required_successes.max(1);
        for attempt in 0..attempts {
            progress.set_message(format_probe_attempt_message(
                kind,
                candidate,
                attempt + 1,
                attempts,
            ));
            let request = ProbeRequest {
                kind,
                candidate_microbatch: candidate,
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            };
            let result_path = probe_result_path(artifacts, kind, candidate, attempt);
            let result = execute_probe_request(config_path, request, &result_path)?;
            let passed = result.status == ProbeStatus::Success;
            progress.inc(1);
            progress.println(format!("{}", format_probe_result_summary(&result).yellow()));
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
    progress.finish_and_clear();

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
    if let Some(result) = best_probe_result(&stable_results) {
        println!(
            "{} {}",
            "Preflight selected:".bold().cyan(),
            format_probe_selection_summary(kind, result).green()
        );
    }
    Ok((selected, results))
}

pub(super) fn run_probe_ladder_only(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    request: ProbeRequest,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let scan_pb = make_spinner("{spinner:.cyan} {msg}")?;
    scan_pb.set_message(format!(
        "scanning data for {} probe",
        probe_kind_name(request.kind)
    ));
    let _ =
        scan_data_sources_with_progress(&config.data_dir, config.train_fraction, Some(&scan_pb))
            .map_err(|err| {
                format!(
                    "failed to scan preflight data from {}: {err}",
                    config.data_dir.display()
                )
            })?;
    scan_pb.finish_with_message(
        format!("scan complete for {} probe", probe_kind_name(request.kind))
            .green()
            .to_string(),
    );

    let candidates = probe_only_candidate_ladder(config, request);
    let selected =
        probe_candidate_ladder(config_path, config, artifacts, request.kind, &candidates)?;
    Ok(selected)
}

pub(super) fn classify_probe_detail(detail: &str) -> ProbeStatus {
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

pub(super) fn run_preflight(
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
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key,
            selected,
        },
    )?;
    Ok(PreflightRuntime {
        selected,
        train_probe_results,
        validation_probe_results,
        explicit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train::preflight::{PreflightConfig, ProbeStatus};

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: Default::default(),
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
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
        }
    }

    #[test]
    fn measure_samples_per_second_handles_zero_samples_and_zero_time() {
        assert_eq!(measure_samples_per_second(0, Duration::from_secs(2)), 0.0);
        assert_eq!(measure_samples_per_second(10, Duration::from_secs(0)), 0.0);
        assert!((measure_samples_per_second(24, Duration::from_secs(3)) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn classify_probe_detail_maps_oom_backend_and_data_cases() {
        assert_eq!(
            classify_probe_detail("CUDA out of memory"),
            ProbeStatus::Oom
        );
        assert_eq!(
            classify_probe_detail("libtorch backend failed"),
            ProbeStatus::BackendError
        );
        assert_eq!(
            classify_probe_detail("replay data collate mismatch"),
            ProbeStatus::DataError
        );
    }

    #[test]
    fn training_runtime_warning_only_fires_for_manual_mismatch() {
        let mut config = dummy_config();
        let cache_key = preflight_cache_key(&config, &HydraModelConfig::learner(), "cpu");
        let saved = PreflightCacheEntry {
            cache_key: cache_key.clone(),
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 128,
                validation_microbatch_size: 96,
                accum_steps: 2,
            },
        };
        let warning =
            training_runtime_warning(&config, Some(&saved), &cache_key).expect("warning expected");
        assert_eq!(warning.saved, saved.selected);
        assert_eq!(warning.configured.train_microbatch_size, 64);

        config.microbatch_size = None;
        config.validation_microbatch_size = None;
        assert!(training_runtime_warning(&config, Some(&saved), &cache_key).is_none());
    }

    #[test]
    fn training_runtime_warning_ignores_stale_cache_key() {
        let config = dummy_config();
        let expected = preflight_cache_key(&config, &HydraModelConfig::learner(), "cpu");
        let stale = PreflightCacheEntry {
            cache_key: preflight_cache_key(&config, &HydraModelConfig::learner(), "cuda:0"),
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 128,
                validation_microbatch_size: 96,
                accum_steps: 2,
            },
        };
        assert!(training_runtime_warning(&config, Some(&stale), &expected).is_none());
    }

    #[test]
    fn preflight_cache_key_changes_when_runtime_relevant_knobs_change() {
        let config = dummy_config();
        let model = HydraModelConfig::learner();
        let baseline = preflight_cache_key(&config, &model, "cpu");

        let mut threaded = config.clone();
        threaded.num_threads = Some(8);
        assert_ne!(baseline, preflight_cache_key(&threaded, &model, "cpu"));

        let mut buffered = config.clone();
        buffered.buffer_samples += 1;
        assert_ne!(baseline, preflight_cache_key(&buffered, &model, "cpu"));

        let mut validation_limited = config.clone();
        validation_limited.max_validation_batches = Some(4);
        assert_ne!(
            baseline,
            preflight_cache_key(&validation_limited, &model, "cpu")
        );
    }

    #[test]
    fn training_runtime_warning_ignores_matching_manual_values() {
        let config = dummy_config();
        let cache_key = preflight_cache_key(&config, &HydraModelConfig::learner(), "cpu");
        let saved = PreflightCacheEntry {
            cache_key: cache_key.clone(),
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 64,
                validation_microbatch_size: 32,
                accum_steps: 4,
            },
        };
        assert!(training_runtime_warning(&config, Some(&saved), &cache_key).is_none());
    }

    #[test]
    fn configured_runtime_selection_matches_train_defaults() {
        let mut config = dummy_config();
        config.microbatch_size = None;
        config.validation_microbatch_size = None;
        let configured = configured_runtime_selection(&config);
        assert_eq!(configured.train_microbatch_size, config.batch_size);
        assert_eq!(configured.validation_microbatch_size, config.batch_size);
        assert_eq!(configured.accum_steps, 1);
    }

    #[test]
    fn probe_request_from_cli_uses_probe_overrides() {
        let config = dummy_config();

        let request = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 192,
                warmup_steps: Some(7),
                measure_steps: Some(9),
            }),
        )
        .expect("probe request should parse")
        .expect("probe request should be present");
        assert_eq!(request.kind, ProbeKind::Validation);
        assert_eq!(request.candidate_microbatch, 192);
        assert_eq!(request.warmup_steps, 7);
        assert_eq!(request.measure_steps, 9);
    }

    #[test]
    fn probe_request_from_cli_falls_back_to_preflight_defaults() {
        let mut config = dummy_config();
        config.preflight.warmup_steps = 11;
        config.preflight.measure_steps = 13;
        let request = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 256,
                warmup_steps: None,
                measure_steps: None,
            }),
        )
        .expect("probe request should parse")
        .expect("probe request should be present");
        assert_eq!(request.warmup_steps, 11);
        assert_eq!(request.measure_steps, 13);
    }

    #[test]
    fn probe_request_from_cli_rejects_zero_values() {
        let config = dummy_config();

        let err = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 0,
                warmup_steps: Some(0),
                measure_steps: Some(0),
            }),
        )
        .expect_err("zero candidate should fail");
        assert!(err.contains("--probe-candidate-microbatch"));
    }

    #[test]
    fn probe_only_candidate_ladder_respects_ceiling_and_descending_order() {
        let mut config = dummy_config();
        config.batch_size = 512;
        config.preflight.candidate_microbatches = vec![64, 512, 192, 128, 256, 192, 32];
        let ladder = probe_only_candidate_ladder(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 192,
                warmup_steps: 4,
                measure_steps: 12,
            },
        );
        assert_eq!(ladder, vec![192, 128, 64, 32]);
    }

    #[test]
    fn probe_only_candidate_ladder_falls_back_to_requested_candidate_when_filtered_empty() {
        let mut config = dummy_config();
        config.batch_size = 512;
        config.preflight.candidate_microbatches = vec![512, 256];
        let ladder = probe_only_candidate_ladder(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 48,
                warmup_steps: 4,
                measure_steps: 12,
            },
        );
        assert_eq!(ladder, vec![48]);
    }

    #[test]
    fn probe_child_request_from_cli_parses_child_probe_inputs() {
        let (request, path) = probe_child_request_from_cli(Some(ProbeChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 192,
                warmup_steps: Some(4),
                measure_steps: Some(12),
            },
            result_path: PathBuf::from("/tmp/probe.json"),
        }))
        .expect("child request should parse")
        .expect("child request should be present");
        assert_eq!(request.kind, ProbeKind::Train);
        assert_eq!(request.candidate_microbatch, 192);
        assert_eq!(request.warmup_steps, 4);
        assert_eq!(request.measure_steps, 12);
        assert_eq!(path, PathBuf::from("/tmp/probe.json"));
    }

    #[test]
    fn format_probe_result_summary_reports_success_and_oom() {
        let success = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 192,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(1234.5),
            detail: String::new(),
        });
        assert!(success.contains("candidate_mb=192"));
        assert!(success.contains("1234.50 samples/s"));

        let oom = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 256,
            status: ProbeStatus::Oom,
            measured_samples_per_second: None,
            detail: String::new(),
        });
        assert_eq!(oom, "[train] candidate_mb=256 status=oom");
    }
}
