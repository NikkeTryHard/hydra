use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use burn::backend::libtorch::LibTorchDevice;
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};

use hydra_train::data::pipeline::{
    scan_data_sources_with_progress, stream_train_epoch, stream_val_pass, DataManifest,
    StreamingLoaderConfig,
};
use hydra_train::data::sample::{collate_samples, MjaiSample};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::preflight::{
    candidate_ladder, default_notes, resolve_runtime_config, ExplicitSettings, HardwareFingerprint,
    PreflightCacheEntry, PreflightCacheKey, PreflightReport, ProbeKind, ProbeResult, ProbeStatus,
    SelectedRuntimeConfig, WorkloadFingerprint,
};
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{
    write_preflight_cache, write_preflight_report, BcArtifactPaths, PreflightPaths,
};
use super::config::{
    train_device, trainer_config_from_train_config, AdvancedLossConfig, TrainConfig,
};
use super::loss_policy::build_loss_config;
use super::schedule::effective_lr;
use super::validation::validation_batch_stats;
use super::{TrainBackend, ValidBackend};

pub(super) struct PreflightRuntime {
    pub(super) report: PreflightReport,
    pub(super) selected: SelectedRuntimeConfig,
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

pub(super) fn probe_request_from_config(
    config: &TrainConfig,
) -> Result<Option<ProbeRequest>, String> {
    let Some(probe) = config.preflight.probe_only.as_ref() else {
        return Ok(None);
    };
    let warmup_steps = probe.warmup_steps.unwrap_or(config.preflight.warmup_steps);
    let measure_steps = probe
        .measure_steps
        .unwrap_or(config.preflight.measure_steps);
    if probe.candidate_microbatch == 0 {
        return Err("preflight.probe_only.candidate_microbatch must be greater than 0".to_string());
    }
    if warmup_steps == 0 {
        return Err("preflight.probe_only warmup_steps must be greater than 0".to_string());
    }
    if measure_steps == 0 {
        return Err("preflight.probe_only measure_steps must be greater than 0".to_string());
    }
    Ok(Some(ProbeRequest {
        kind: probe.kind,
        candidate_microbatch: probe.candidate_microbatch,
        warmup_steps,
        measure_steps,
    }))
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

pub(super) fn run_probe_only(
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

pub(super) fn execute_probe_request(
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

pub(super) fn apply_preflight_selection(
    config: &TrainConfig,
    selected: SelectedRuntimeConfig,
) -> TrainConfig {
    if config.preflight.advisory_only {
        return config.clone();
    }
    let mut resolved = config.clone();
    resolved.microbatch_size = Some(selected.train_microbatch_size);
    resolved.validation_microbatch_size = Some(selected.validation_microbatch_size);
    resolved
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

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train::preflight::{PreflightConfig, ProbeOnlyConfig, ProbeStatus};

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
    fn apply_preflight_selection_respects_advisory_only() {
        let mut advisory = dummy_config();
        advisory.preflight.advisory_only = true;
        let advisory_selected = apply_preflight_selection(
            &advisory,
            SelectedRuntimeConfig {
                train_microbatch_size: 128,
                validation_microbatch_size: 96,
                accum_steps: 2,
            },
        );
        assert_eq!(advisory_selected.microbatch_size, advisory.microbatch_size);
        assert_eq!(
            advisory_selected.validation_microbatch_size,
            advisory.validation_microbatch_size
        );

        let applied = apply_preflight_selection(
            &dummy_config(),
            SelectedRuntimeConfig {
                train_microbatch_size: 128,
                validation_microbatch_size: 96,
                accum_steps: 2,
            },
        );
        assert_eq!(applied.microbatch_size, Some(128));
        assert_eq!(applied.validation_microbatch_size, Some(96));
    }

    #[test]
    fn probe_request_from_config_uses_probe_only_overrides() {
        let mut config = dummy_config();
        config.preflight.probe_only = Some(ProbeOnlyConfig {
            kind: ProbeKind::Validation,
            candidate_microbatch: 192,
            warmup_steps: Some(7),
            measure_steps: Some(9),
        });

        let request = probe_request_from_config(&config)
            .expect("probe request should parse")
            .expect("probe_only should be present");
        assert_eq!(request.kind, ProbeKind::Validation);
        assert_eq!(request.candidate_microbatch, 192);
        assert_eq!(request.warmup_steps, 7);
        assert_eq!(request.measure_steps, 9);
    }

    #[test]
    fn probe_request_from_config_falls_back_to_preflight_defaults() {
        let mut config = dummy_config();
        config.preflight.warmup_steps = 11;
        config.preflight.measure_steps = 13;
        config.preflight.probe_only = Some(ProbeOnlyConfig {
            kind: ProbeKind::Train,
            candidate_microbatch: 256,
            warmup_steps: None,
            measure_steps: None,
        });

        let request = probe_request_from_config(&config)
            .expect("probe request should parse")
            .expect("probe_only should be present");
        assert_eq!(request.warmup_steps, 11);
        assert_eq!(request.measure_steps, 13);
    }

    #[test]
    fn probe_request_from_config_rejects_zero_values() {
        let mut config = dummy_config();
        config.preflight.probe_only = Some(ProbeOnlyConfig {
            kind: ProbeKind::Train,
            candidate_microbatch: 0,
            warmup_steps: Some(0),
            measure_steps: Some(0),
        });

        let err = probe_request_from_config(&config).expect_err("zero candidate should fail");
        assert!(err.contains("candidate_microbatch"));
    }
}
