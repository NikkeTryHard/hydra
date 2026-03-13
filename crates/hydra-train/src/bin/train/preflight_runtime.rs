use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, collections::BTreeSet};

use burn::backend::libtorch::LibTorchDevice;
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;
use hydra_train::data::pipeline::{
    DataManifest, StreamingLoaderConfig, scan_data_sources_with_progress, stream_train_epoch,
    stream_val_pass,
};
use hydra_train::data::sample::{MjaiSample, collate_samples};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, HardwareFingerprint, LoaderRuntimeConfig,
    PreflightCacheEntry, PreflightCacheKey, ProbeKind, ProbeResult, ProbeStatus,
    WorkloadFingerprint, candidate_ladder, resolve_runtime_config,
};
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{BcArtifactPaths, PreflightPaths, write_preflight_cache};
use super::config::{
    AdvancedLossConfig, ProbeChildRequest, ProbeCliRequest, TrainConfig,
    default_num_threads_for_system, loader_runtime_config, train_device,
    trainer_config_from_train_config,
};
use super::loss_policy::build_loss_config;
use super::presentation::{
    format_preflight_selection_line, format_preflight_summary_line, format_probe_progress_line,
    format_probe_status_line, format_runtime_tuning_message, format_runtime_tuning_result,
    format_status_line, format_timed_phase_message, make_bar, make_spinner, preflight_phase_label,
};
use super::schedule::effective_lr;
use super::validation::validation_batch_stats;
use super::{TrainBackend, ValidBackend};

pub(super) struct PreflightRuntime {
    pub(super) runtime: EffectiveRuntimeConfig,
    pub(super) train_probe_results: Vec<ProbeResult>,
    pub(super) validation_probe_results: Vec<ProbeResult>,
    pub(super) explicit: ExplicitSettings,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ProbeRequest {
    pub(super) kind: ProbeKind,
    pub(super) candidate_microbatch: usize,
    pub(super) warmup_steps: usize,
    pub(super) measure_steps: usize,
}

fn interrupt_flag() -> Result<Arc<AtomicBool>, String> {
    static INTERRUPTED: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    static HANDLER_INSTALLED: OnceLock<()> = OnceLock::new();
    let flag = INTERRUPTED
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();
    if HANDLER_INSTALLED.get().is_none() {
        ctrlc::set_handler({
            let flag = flag.clone();
            move || {
                flag.store(true, Ordering::SeqCst);
            }
        })
        .map_err(|err| format!("failed to install preflight interrupt handler: {err}"))?;
        let _ = HANDLER_INSTALLED.set(());
    }
    Ok(flag)
}

fn should_suppress_probe_output_line(line: &str) -> bool {
    let lowered = line.to_ascii_lowercase();
    lowered.contains("thread 'main'")
        || lowered.contains("called `result::unwrap()`")
        || lowered.contains("called `result::unwrap()")
        || lowered.contains("note: run with `rust_backtrace=1`")
        || lowered.contains("stack backtrace")
        || lowered.contains("frame #")
        || lowered.contains("exception raised from malloc")
        || lowered.contains("/pytorch/")
        || lowered.contains("/opt/conda/lib/python")
        || lowered.contains("cudacachingallocator")
        || lowered.contains("skipping ")
}

fn normalized_probe_output_line(line: &str) -> Option<String> {
    if let Some(formatted) = format_probe_progress_line(line) {
        return Some(formatted);
    }
    if line.trim_start().starts_with("probe_progress ") {
        return None;
    }
    if should_suppress_probe_output_line(line) {
        return None;
    }
    Some(line.trim().to_string())
}

fn spawn_output_forwarder<R>(reader: R, stderr: bool) -> thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut collected = Vec::new();
        let mut buffered = BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = buffered
                .read_until(b'\n', &mut line)
                .map_err(|err| format!("failed reading preflight probe output: {err}"))?;
            if read == 0 {
                break;
            }
            collected.extend_from_slice(&line);
            let text = String::from_utf8_lossy(&line);
            if let Some(formatted) = normalized_probe_output_line(&text) {
                if stderr {
                    writeln!(std::io::stderr(), "{formatted}").map_err(|err| {
                        format!("failed forwarding preflight probe stderr: {err}")
                    })?;
                    std::io::stderr()
                        .flush()
                        .map_err(|err| format!("failed flushing preflight probe stderr: {err}"))?;
                } else {
                    writeln!(std::io::stdout(), "{formatted}").map_err(|err| {
                        format!("failed forwarding preflight probe stdout: {err}")
                    })?;
                    std::io::stdout()
                        .flush()
                        .map_err(|err| format!("failed flushing preflight probe stdout: {err}"))?;
                }
            }
        }
        Ok(collected)
    })
}

fn summarize_probe_failure_output(output: &str) -> String {
    let mut lines = Vec::new();
    for line in output.lines() {
        if should_suppress_probe_output_line(line) {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("probe_progress ") {
            continue;
        }
        lines.push(trimmed.to_string());
        if lines.len() >= 3 {
            break;
        }
    }
    lines.join(" | ")
}

fn probe_failure_detail(
    status: ProbeStatus,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
) -> String {
    match status {
        ProbeStatus::Oom => format!(
            "probe process status={exit_code:?} detail=libtorch/cuda oom during preflight probe; raw panic output suppressed"
        ),
        _ => {
            let summary = summarize_probe_failure_output(stderr);
            let fallback = if summary.is_empty() {
                summarize_probe_failure_output(stdout)
            } else {
                summary
            };
            if fallback.is_empty() {
                format!(
                    "probe process status={exit_code:?} detail=probe child failed without structured result"
                )
            } else {
                format!("probe process status={exit_code:?} detail={fallback}")
            }
        }
    }
}

fn join_output_forwarder(
    handle: thread::JoinHandle<Result<Vec<u8>, String>>,
    stream_name: &str,
) -> Result<Vec<u8>, String> {
    handle
        .join()
        .map_err(|_| format!("preflight probe {stream_name} forwarder panicked"))?
}

fn emit_probe_progress(line: &str) -> Result<(), String> {
    if let Some(formatted) = format_probe_progress_line(line) {
        println!("{formatted}");
    }
    std::io::stdout()
        .flush()
        .map_err(|err| format!("failed flushing probe progress output: {err}"))
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

fn child_output(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> std::process::Output {
    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

fn wait_for_probe_child(
    child: &mut Child,
    interrupted: &AtomicBool,
) -> Result<Option<ExitStatus>, String> {
    loop {
        if interrupted.load(Ordering::SeqCst) {
            child.kill().ok();
            child.wait().ok();
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(err) => {
                child.kill().ok();
                child.wait().ok();
                return Err(format!(
                    "failed while waiting for preflight probe child: {err}"
                ));
            }
        }
    }
}

fn total_memory_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
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
            "hydra-train:{}:{}:preflight-v3",
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
        cpu_logical_cores: default_num_threads_for_system(),
        total_memory_bytes: total_memory_bytes(),
    }
}

fn autotune_buffer_samples_candidates(config: &TrainConfig) -> Vec<usize> {
    let mut candidates = vec![
        config.buffer_samples.max(1),
        config.buffer_samples.saturating_mul(2).max(1),
        config.buffer_samples.saturating_mul(4).max(1),
    ];
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn autotune_buffer_games_candidates(config: &TrainConfig) -> Vec<usize> {
    let mut candidates = vec![
        config.buffer_games.max(1),
        config.buffer_games.saturating_mul(2).max(1),
    ];
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn autotune_archive_queue_candidates(config: &TrainConfig) -> Vec<usize> {
    let mut candidates = vec![config.archive_queue_bound.max(1), 64, 128, 256];
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn runtime_probe_loader_config(config: &TrainConfig) -> StreamingLoaderConfig {
    StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
    }
}

fn measure_train_runtime_throughput(
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let train_cfg = trainer_config_from_train_config(config);
    let mut model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let microbatch_size = config
        .microbatch_size
        .unwrap_or(config.batch_size)
        .min(config.batch_size)
        .max(1);
    let warmup_steps = config.preflight.warmup_steps.max(1);
    let measure_steps = config.preflight.measure_steps.max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut completed_steps = 0usize;
    let mut pending_samples = std::collections::VecDeque::new();
    let mut measure_start = None;

    for buffer_result in stream_train_epoch(manifest, loader_config, 0, None) {
        let buffer = buffer_result.map_err(|err| format!("runtime train stream failed: {err}"))?;
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
            if completed_steps == warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    measure_steps * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    Err("not enough train data to finish runtime probe".to_string())
}

fn tune_runtime_knob<T, F>(
    base: &TrainConfig,
    knob_name: &str,
    candidates: &[T],
    display: impl Fn(T) -> String,
    apply: impl Fn(&mut TrainConfig, T),
    score: &mut F,
) -> Result<T, String>
where
    T: Copy,
    F: FnMut(&TrainConfig) -> Result<f64, String>,
{
    let mut best = *candidates
        .first()
        .ok_or_else(|| "runtime autotune candidate list cannot be empty".to_string())?;
    let mut best_score = f64::NEG_INFINITY;
    let progress = make_bar(
        candidates.len() as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    for (index, candidate) in candidates.iter().enumerate() {
        let candidate_label = display(*candidate);
        progress.set_message(format_runtime_tuning_message(
            knob_name,
            candidate_label.clone(),
            index,
            candidates.len(),
        ));
        let mut candidate_config = base.clone();
        apply(&mut candidate_config, *candidate);
        let throughput = score(&candidate_config)?;
        if throughput > best_score {
            best_score = throughput;
            best = *candidate;
        }
        progress.inc(1);
        println!(
            "{}",
            format_runtime_tuning_result(
                knob_name,
                candidate_label,
                throughput,
                display(best),
                best_score,
            )
        );
    }
    progress.finish_with_message(
        format!("runtime tuning {knob_name} complete")
            .green()
            .to_string(),
    );
    Ok(best)
}

fn candidate_average(results: &[ProbeResult], candidate: usize) -> Option<f64> {
    summarize_probe_results(results)
        .into_iter()
        .find(|summary| {
            summary.candidate_microbatch == candidate && summary.status == ProbeStatus::Success
        })
        .and_then(|summary| summary.average_samples_per_second)
}

fn close_probe_finalists(
    results: &[ProbeResult],
    margin_ratio: f64,
    max_candidates: usize,
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
    });
    let Some(best) = summaries
        .first()
        .and_then(|summary| summary.average_samples_per_second)
    else {
        return Vec::new();
    };
    summaries
        .into_iter()
        .filter(|summary| {
            summary
                .average_samples_per_second
                .map(|score| score >= best * (1.0 - margin_ratio.max(0.0)))
                .unwrap_or(false)
        })
        .take(max_candidates.max(1))
        .collect()
}

fn rerun_probe_finalists(
    config_path: &Path,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String> {
    let finalists = close_probe_finalists(
        results,
        config.preflight.finalist_margin_ratio,
        config.preflight.finalist_max_candidates,
    );
    if finalists.len() < 2 {
        return Ok(());
    }
    let extra_attempts = config.preflight.finalist_extra_successes.max(1);
    let extra_measure_steps = config.preflight.finalist_extra_measure_steps.max(1);
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight refine:",
            format!(
                "kind={} finalists={:?} extra_attempts={} extra_measure_steps={}",
                probe_kind_name(kind),
                finalists
                    .iter()
                    .map(|summary| summary.candidate_microbatch)
                    .collect::<Vec<_>>(),
                extra_attempts,
                extra_measure_steps,
            )
        )
    );
    for summary in finalists {
        let seconds_per_step = summary.average_elapsed_seconds.unwrap_or(0.0)
            / (config.preflight.warmup_steps + config.preflight.measure_steps).max(1) as f64;
        let (warmup_steps, measure_steps) = adaptive_probe_steps(config, seconds_per_step);
        rerun_candidate_attempts(
            config_path,
            artifacts,
            kind,
            summary.candidate_microbatch,
            extra_attempts,
            warmup_steps,
            measure_steps + extra_measure_steps,
            results,
            progress,
        )?;
    }
    Ok(())
}

fn local_refinement_candidates(
    summaries: &[ProbeCandidateSummary],
    min_gap: usize,
    max_candidates: usize,
    ceiling: usize,
) -> Vec<usize> {
    let mut successful = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .filter_map(|summary| {
            summary
                .average_samples_per_second
                .map(|score| (summary.candidate_microbatch, score))
        })
        .collect::<Vec<_>>();
    successful.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some((winner, _)) = successful.first().copied() else {
        return Vec::new();
    };

    let mut all_candidates = successful
        .iter()
        .map(|(candidate, _)| *candidate)
        .collect::<Vec<_>>();
    all_candidates.sort_unstable();
    all_candidates.dedup();

    let winner_index = all_candidates
        .iter()
        .position(|candidate| *candidate == winner);
    let Some(winner_index) = winner_index else {
        return Vec::new();
    };

    let mut refined = BTreeSet::new();
    let lower = winner_index
        .checked_sub(1)
        .and_then(|index| all_candidates.get(index).copied());
    let upper = all_candidates.get(winner_index + 1).copied();
    let failed_above = summaries
        .iter()
        .filter(|summary| {
            summary.candidate_microbatch > winner && summary.status == ProbeStatus::Oom
        })
        .map(|summary| summary.candidate_microbatch)
        .min();

    for neighbor in [lower, upper, failed_above].into_iter().flatten() {
        let lo = neighbor.min(winner);
        let hi = neighbor.max(winner);
        if hi.saturating_sub(lo) < min_gap.max(1) {
            continue;
        }
        let midpoint = lo + (hi - lo) / 2;
        if midpoint != lo && midpoint != hi && midpoint <= ceiling {
            refined.insert(midpoint);
        }
    }

    refined.into_iter().take(max_candidates.max(1)).collect()
}

fn refine_probe_winner_locally(
    config_path: &Path,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String> {
    if !config.preflight.local_refinement_enabled {
        return Ok(());
    }
    let summaries = summarize_probe_results(results);
    let ceiling = dynamic_probe_ceiling(
        config,
        kind,
        best_probe_summary(results)
            .map(|summary| summary.candidate_microbatch)
            .unwrap_or(config.batch_size),
    );
    let candidates = local_refinement_candidates(
        &summaries,
        config.preflight.local_refinement_min_gap,
        config.preflight.local_refinement_max_candidates,
        ceiling,
    );
    if candidates.is_empty() {
        return Ok(());
    }
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight local refine:",
            format!(
                "kind={} candidates={:?} extra_measure_steps={}",
                probe_kind_name(kind),
                candidates,
                config.preflight.local_refinement_extra_measure_steps.max(1),
            )
        )
    );
    for candidate in candidates {
        let base = candidate_average(results, candidate).unwrap_or(0.0);
        let seconds_per_step = if base > 0.0 {
            summarize_probe_results(results)
                .into_iter()
                .find(|summary| summary.candidate_microbatch == candidate)
                .and_then(|summary| summary.average_elapsed_seconds)
                .unwrap_or(0.0)
                / (config.preflight.warmup_steps + config.preflight.measure_steps).max(1) as f64
        } else {
            0.0
        };
        let (warmup_steps, measure_steps) = adaptive_probe_steps(config, seconds_per_step);
        rerun_candidate_attempts(
            config_path,
            artifacts,
            kind,
            candidate,
            1,
            warmup_steps,
            measure_steps + config.preflight.local_refinement_extra_measure_steps.max(1),
            results,
            progress,
        )?;
    }
    Ok(())
}

fn score_tuple_samples_mean(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

fn adaptive_probe_steps(config: &TrainConfig, seconds_per_step: f64) -> (usize, usize) {
    let bounded_seconds = seconds_per_step.max(0.001);
    let warmup_steps = ((config.preflight.target_warmup_seconds / bounded_seconds).ceil() as usize)
        .clamp(
            config.preflight.warmup_steps.max(1),
            config
                .preflight
                .max_adaptive_warmup_steps
                .max(config.preflight.warmup_steps.max(1)),
        );
    let measure_steps =
        ((config.preflight.target_measure_seconds / bounded_seconds).ceil() as usize).clamp(
            config.preflight.measure_steps.max(1),
            config
                .preflight
                .max_adaptive_measure_steps
                .max(config.preflight.measure_steps.max(1)),
        );
    (warmup_steps, measure_steps)
}

fn rerun_candidate_attempts(
    config_path: &Path,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidate: usize,
    attempts: usize,
    warmup_steps: usize,
    measure_steps: usize,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String> {
    run_candidate_attempts(
        config_path,
        artifacts,
        kind,
        candidate,
        attempts,
        warmup_steps,
        measure_steps,
        results,
        progress,
    )?;
    Ok(())
}

fn should_continue_validation_growth(best: f64, challenger: f64, tolerance_ratio: f64) -> bool {
    challenger >= best * (1.0 - tolerance_ratio.max(0.0))
}

fn run_candidate_attempts(
    config_path: &Path,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidate: usize,
    attempts: usize,
    warmup_steps: usize,
    measure_steps: usize,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<bool, String> {
    for attempt in 0..attempts.max(1) {
        let attempt_number = attempt + 1;
        progress.set_message(format_probe_attempt_message(
            kind,
            candidate,
            attempt_number,
            attempts,
        ));
        let request = ProbeRequest {
            kind,
            candidate_microbatch: candidate,
            warmup_steps,
            measure_steps,
        };
        let result_path = probe_result_path(artifacts, kind, candidate, attempt);
        println!(
            "{}",
            format_status_line(
                &format!("[preflight:{}]", probe_kind_name(kind)),
                format!(
                    "candidate_mb={} attempt={}/{} phase=probe",
                    candidate, attempt_number, attempts,
                )
            )
        );
        let result = execute_probe_request(config_path, request, &result_path)?;
        let passed = result.status == ProbeStatus::Success;
        progress.inc(1);
        println!("{}", format_probe_result_summary(&result));
        results.push(result);
        if !passed {
            return Ok(false);
        }
    }
    Ok(true)
}

fn search_train_microbatch(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    seed: usize,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let mut candidates = dynamic_probe_ladder(config, ProbeKind::Train, seed);
    let explicit_candidate = config.microbatch_size;
    let use_explicit_only =
        explicit_candidate.is_some() && !config.preflight.allow_override_explicit_microbatch;
    if use_explicit_only {
        candidates = vec![explicit_candidate.unwrap_or(1)];
    }
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind=train candidates={:?} required_successes={}",
                candidates,
                config.preflight.required_successes.max(1),
            )
        )
    );
    let progress = make_bar(
        (candidates.len() * config.preflight.required_successes.max(1)) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut results = Vec::new();
    let mut stable_results = Vec::new();
    let mut best_score = f64::NEG_INFINITY;

    for candidate in candidates {
        let stable_start = results.len();
        let passed = run_candidate_attempts(
            config_path,
            artifacts,
            ProbeKind::Train,
            candidate,
            config.preflight.required_successes.max(1),
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            &mut results,
            &progress,
        )?;
        if !passed {
            if use_explicit_only {
                progress.finish_with_message("preflight train ladder complete".green().to_string());
                return Err(format!(
                    "explicit train microbatch {} failed preflight",
                    candidate
                ));
            }
            continue;
        }
        stable_results.extend(results[stable_start..].iter().cloned());
        let throughput = candidate_average(&results, candidate).unwrap_or(0.0);
        if throughput > best_score {
            best_score = throughput;
        }

        if use_explicit_only {
            progress.finish_with_message("preflight train ladder complete".green().to_string());
            return Ok((candidate, results));
        }
    }

    progress.finish_with_message("preflight train ladder complete".green().to_string());
    refine_probe_winner_locally(
        config_path,
        artifacts,
        ProbeKind::Train,
        config,
        &mut results,
        &progress,
    )?;
    stable_results = results
        .iter()
        .filter(|result| result.status == ProbeStatus::Success)
        .cloned()
        .collect();
    rerun_probe_finalists(
        config_path,
        artifacts,
        ProbeKind::Train,
        config,
        &mut stable_results,
        &progress,
    )?;
    let selected_summary = best_probe_summary(&stable_results)
        .ok_or_else(|| "no stable train microbatch found in preflight".to_string())?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            ProbeKind::Train,
            &selected_summary,
        ))
    );
    Ok((selected_summary.candidate_microbatch, results))
}

fn search_validation_microbatch(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    seed: usize,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let explicit_candidate = config.validation_microbatch_size;
    let use_explicit_only =
        explicit_candidate.is_some() && !config.preflight.allow_override_explicit_microbatch;
    let mut candidates = if use_explicit_only {
        vec![explicit_candidate.unwrap_or(1)]
    } else {
        dynamic_probe_ladder(config, ProbeKind::Validation, seed)
    };
    let mut seen = BTreeSet::new();
    candidates.retain(|candidate| seen.insert(*candidate));
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind=validation candidates={:?} required_successes={} growth_patience={} growth_max_steps={}",
                candidates,
                config.preflight.required_successes.max(1),
                config.preflight.validation_growth_patience.max(1),
                config.preflight.validation_growth_max_steps.max(1)
            )
        )
    );
    let progress = make_bar(
        (candidates.len() * config.preflight.required_successes.max(1)) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut results = Vec::new();
    let mut stable_results = Vec::new();
    let mut growth_patience = 0usize;
    let mut growth_steps = 0usize;
    let tolerance = config.preflight.measure_noise_tolerance_ratio;
    let mut prior_best_score: Option<f64> = None;

    let mut index = 0usize;
    while index < candidates.len() {
        let candidate = candidates[index];
        let stable_start = results.len();
        let passed = run_candidate_attempts(
            config_path,
            artifacts,
            ProbeKind::Validation,
            candidate,
            config.preflight.required_successes.max(1),
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            &mut results,
            &progress,
        )?;
        if !passed {
            if use_explicit_only {
                progress.finish_with_message(
                    "preflight validation ladder complete".green().to_string(),
                );
                return Err(format!(
                    "explicit validation microbatch {} failed preflight",
                    candidate
                ));
            }
            break;
        }
        stable_results.extend(results[stable_start..].iter().cloned());
        if use_explicit_only {
            progress
                .finish_with_message("preflight validation ladder complete".green().to_string());
            return Ok((candidate, results));
        }

        let summary = best_probe_summary(&stable_results)
            .ok_or_else(|| "no stable validation microbatch found in preflight".to_string())?;
        let candidate_score = candidate_average(&results, candidate).unwrap_or(0.0);
        let is_top = index + 1 == candidates.len();
        if is_top && summary.candidate_microbatch == candidate {
            let ceiling = dynamic_probe_ceiling(config, ProbeKind::Validation, candidate);
            let next_candidate = candidate.saturating_mul(2);
            if next_candidate > candidate && next_candidate <= ceiling {
                if growth_steps >= config.preflight.validation_growth_max_steps.max(1) {
                    break;
                }
                let reference_score = prior_best_score.unwrap_or_else(|| {
                    summary
                        .average_samples_per_second
                        .unwrap_or(candidate_score)
                });
                if should_continue_validation_growth(reference_score, candidate_score, tolerance) {
                    growth_patience = 0;
                    growth_steps += 1;
                    candidates.push(next_candidate);
                    prior_best_score = Some(reference_score.max(candidate_score));
                } else {
                    growth_patience += 1;
                    prior_best_score = Some(reference_score.max(candidate_score));
                    if growth_patience >= config.preflight.validation_growth_patience.max(1) {
                        break;
                    }
                }
            }
        }
        prior_best_score = Some(
            prior_best_score.unwrap_or(0.0).max(
                summary
                    .average_samples_per_second
                    .unwrap_or(candidate_score),
            ),
        );
        index += 1;
    }

    progress.finish_with_message("preflight validation ladder complete".green().to_string());
    refine_probe_winner_locally(
        config_path,
        artifacts,
        ProbeKind::Validation,
        config,
        &mut results,
        &progress,
    )?;
    stable_results = results
        .iter()
        .filter(|result| result.status == ProbeStatus::Success)
        .cloned()
        .collect();
    rerun_probe_finalists(
        config_path,
        artifacts,
        ProbeKind::Validation,
        config,
        &mut stable_results,
        &progress,
    )?;
    let selected_summary = best_probe_summary(&stable_results)
        .ok_or_else(|| "no stable validation microbatch found in preflight".to_string())?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            ProbeKind::Validation,
            &selected_summary,
        ))
    );
    Ok((selected_summary.candidate_microbatch, results))
}

fn score_runtime_tuple(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), Vec<f64>>,
) -> Result<f64, String> {
    let key = (
        config.archive_queue_bound,
        config.buffer_samples,
        config.buffer_games,
    );
    if let Some(scores) = cache.get(&key)
        && !scores.is_empty()
    {
        return Ok(score_tuple_samples_mean(scores));
    }
    let loader = runtime_probe_loader_config(config);
    let score = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    cache.insert(key, vec![score]);
    Ok(score)
}

fn push_runtime_tuple_sample(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), Vec<f64>>,
) -> Result<f64, String> {
    let key = (
        config.archive_queue_bound,
        config.buffer_samples,
        config.buffer_games,
    );
    let loader = runtime_probe_loader_config(config);
    let sample = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    let samples = cache.entry(key).or_default();
    samples.push(sample);
    Ok(score_tuple_samples_mean(samples))
}

fn autotune_loader_runtime(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<LoaderRuntimeConfig, String> {
    let runtime_tuning_started = Instant::now();
    let mut tuned = config.clone();
    tuned.num_threads = loader_runtime_config(&tuned).num_threads;

    if let Some(num_threads) = tuned.num_threads
        && num_threads == 0
    {
        return Err("runtime autotune produced invalid num_threads=0".to_string());
    }

    let mut score_cache: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();

    let queue_candidates = autotune_archive_queue_candidates(&tuned);
    let sample_candidates = autotune_buffer_samples_candidates(&tuned);
    let game_candidates = autotune_buffer_games_candidates(&tuned);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_tuple = (
        tuned.archive_queue_bound,
        tuned.buffer_samples,
        tuned.buffer_games,
    );
    let mut coarse_scores = Vec::new();
    let coarse_started = Instant::now();
    println!(
        "{}",
        format_timed_phase_message(
            "runtime_coarse_search",
            &format!(
                "starting tuples={}",
                queue_candidates.len() * sample_candidates.len() * game_candidates.len()
            ),
            0.0,
        )
    );

    let coarse_progress = make_bar(
        (queue_candidates.len() * sample_candidates.len() * game_candidates.len()) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    for queue in &queue_candidates {
        for samples in &sample_candidates {
            for games in &game_candidates {
                coarse_progress.set_message(format_runtime_tuning_message(
                    "coarse_search",
                    format!("q={queue}, samples={samples}, games={games}"),
                    coarse_progress.position() as usize,
                    coarse_progress.length().unwrap_or(1) as usize,
                ));
                let mut candidate = tuned.clone();
                candidate.archive_queue_bound = *queue;
                candidate.buffer_samples = *samples;
                candidate.buffer_games = *games;
                let score =
                    score_runtime_tuple(&candidate, manifest, train_device, &mut score_cache)?;
                coarse_progress.inc(1);
                coarse_scores.push(((*queue, *samples, *games), score));
                if score > best_score {
                    best_score = score;
                    best_tuple = (*queue, *samples, *games);
                }
            }
        }
    }
    coarse_progress.finish_with_message("runtime coarse search complete".green().to_string());
    println!(
        "{}",
        format_timed_phase_message(
            "runtime_coarse_search",
            "complete",
            coarse_started.elapsed().as_secs_f64(),
        )
    );

    coarse_scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let close_tuples = coarse_scores
        .iter()
        .filter(|(_, score)| {
            *score >= best_score * (1.0 - config.preflight.loader_tuple_margin_ratio)
        })
        .take(2)
        .map(|(tuple, _)| *tuple)
        .collect::<Vec<_>>();
    if close_tuples.len() >= 2 {
        let refine_started = Instant::now();
        println!(
            "{}",
            format_preflight_summary_line(
                "Runtime refine:",
                format!(
                    "close_tuples={:?} extra_samples={}",
                    close_tuples,
                    config.preflight.loader_tuple_extra_samples.max(1)
                )
            )
        );
        for tuple in &close_tuples {
            let mut candidate = tuned.clone();
            candidate.archive_queue_bound = tuple.0;
            candidate.buffer_samples = tuple.1;
            candidate.buffer_games = tuple.2;
            for _ in 0..config.preflight.loader_tuple_extra_samples.max(1) {
                let averaged = push_runtime_tuple_sample(
                    &candidate,
                    manifest,
                    train_device,
                    &mut score_cache,
                )?;
                if averaged > best_score {
                    best_score = averaged;
                    best_tuple = *tuple;
                }
            }
        }
        println!(
            "{}",
            format_timed_phase_message(
                "runtime_refine",
                "complete",
                refine_started.elapsed().as_secs_f64(),
            )
        );
    }

    tuned.archive_queue_bound = best_tuple.0;
    tuned.buffer_samples = best_tuple.1;
    tuned.buffer_games = best_tuple.2;

    for _round in 0..config.preflight.loader_runtime_rounds.max(1) {
        let mut score = |candidate: &TrainConfig| {
            score_runtime_tuple(candidate, manifest, train_device, &mut score_cache)
        };

        let queue_candidates = autotune_archive_queue_candidates(&tuned);
        tuned.archive_queue_bound = tune_runtime_knob(
            &tuned,
            "archive_queue_bound",
            &queue_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.archive_queue_bound = value,
            &mut score,
        )?;

        let sample_candidates = autotune_buffer_samples_candidates(&tuned);
        tuned.buffer_samples = tune_runtime_knob(
            &tuned,
            "buffer_samples",
            &sample_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.buffer_samples = value,
            &mut score,
        )?;

        let game_candidates = autotune_buffer_games_candidates(&tuned);
        tuned.buffer_games = tune_runtime_knob(
            &tuned,
            "buffer_games",
            &game_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.buffer_games = value,
            &mut score,
        )?;
    }

    println!(
        "{}",
        format_timed_phase_message(
            "runtime_tuning_total",
            "complete",
            runtime_tuning_started.elapsed().as_secs_f64(),
        )
    );

    Ok(loader_runtime_config(&tuned))
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

const MAX_DYNAMIC_PROBE_CANDIDATE: usize = 8192;

fn dynamic_probe_ceiling(config: &TrainConfig, kind: ProbeKind, seed: usize) -> usize {
    match kind {
        ProbeKind::Train => config.batch_size.max(seed),
        ProbeKind::Validation => config
            .max_validation_samples
            .unwrap_or(MAX_DYNAMIC_PROBE_CANDIDATE.saturating_mul(8))
            .max(config.batch_size.max(seed).saturating_mul(8)),
    }
}

fn dynamic_probe_growth_candidates(
    config: &TrainConfig,
    kind: ProbeKind,
    seed: usize,
) -> Vec<usize> {
    let ceiling = dynamic_probe_ceiling(config, kind, seed);
    let mut candidates = Vec::new();
    let mut current = seed.max(1);
    loop {
        let next = current.saturating_mul(2);
        if next <= current || next > ceiling {
            break;
        }
        candidates.push(next);
        current = next;
    }
    candidates
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

pub(super) fn dynamic_probe_ladder(
    config: &TrainConfig,
    kind: ProbeKind,
    seed: usize,
) -> Vec<usize> {
    let mut lower = candidate_ladder(&config.preflight, config.batch_size)
        .into_iter()
        .filter(|candidate| *candidate < seed)
        .collect::<Vec<_>>();
    let mut ladder = vec![seed.max(1)];
    ladder.extend(dynamic_probe_growth_candidates(config, kind, seed));
    lower.sort_unstable_by(|a, b| b.cmp(a));
    ladder.extend(lower);
    ladder.dedup();
    ladder
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

#[derive(Debug, Clone)]
pub(super) struct ProbeCandidateSummary {
    pub(super) candidate_microbatch: usize,
    pub(super) status: ProbeStatus,
    pub(super) attempts: usize,
    pub(super) average_samples_per_second: Option<f64>,
    pub(super) average_elapsed_seconds: Option<f64>,
}

fn average(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / count as f64)
}

pub(super) fn summarize_probe_results(results: &[ProbeResult]) -> Vec<ProbeCandidateSummary> {
    let mut grouped = std::collections::BTreeMap::<usize, Vec<&ProbeResult>>::new();
    for result in results {
        grouped
            .entry(result.candidate_microbatch)
            .or_default()
            .push(result);
    }

    grouped
        .into_iter()
        .rev()
        .map(|(candidate_microbatch, group)| {
            let status = group
                .iter()
                .find(|result| result.status != ProbeStatus::Success)
                .map(|result| result.status.clone())
                .unwrap_or(ProbeStatus::Success);
            ProbeCandidateSummary {
                candidate_microbatch,
                status,
                attempts: group.len(),
                average_samples_per_second: average(
                    group
                        .iter()
                        .filter_map(|result| result.measured_samples_per_second),
                ),
                average_elapsed_seconds: average(
                    group.iter().filter_map(|result| result.elapsed_seconds),
                ),
            }
        })
        .collect()
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
    emit_probe_progress(&format!(
        "probe_progress kind=train candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
        microbatch_size, request.warmup_steps, request.measure_steps
    ))?;

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
            emit_probe_step_progress(
                ProbeKind::Train,
                microbatch_size,
                completed_steps,
                request,
                measure_start,
                config.batch_size,
            )?;
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
                emit_probe_progress(&format!(
                    "probe_progress kind=train candidate_mb={} phase=measure_start total_steps={}",
                    microbatch_size,
                    request.measure_steps.max(1)
                ))?;
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
    emit_probe_progress(&format!(
        "probe_progress kind=validation candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
        microbatch_size, request.warmup_steps, request.measure_steps
    ))?;

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
            emit_probe_step_progress(
                ProbeKind::Validation,
                microbatch_size,
                completed_steps,
                request,
                measure_start,
                microbatch_size,
            )?;
            completed_steps += 1;
            if completed_steps == request.warmup_steps {
                measure_start = Some(Instant::now());
                emit_probe_progress(&format!(
                    "probe_progress kind=validation candidate_mb={} phase=measure_start total_steps={}",
                    microbatch_size,
                    request.measure_steps.max(1)
                ))?;
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
        aggregate_skip_logs: true,
    };
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=scan_start data_dir={}",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        config.data_dir.display(),
    ))?;
    let manifest = scan_data_sources_with_progress(&config.data_dir, config.train_fraction, None)
        .map_err(|err| {
        format!(
            "failed to scan preflight data from {}: {err}",
            config.data_dir.display()
        )
    })?;
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=scan_complete sources={} total_games={} train_count={} val_count={} counts_exact={}",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        manifest.sources.len(),
        manifest.total_games,
        manifest.train_count,
        manifest.val_count,
        manifest.counts_exact,
    ))?;
    let train_device = train_device(&config.device);
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => {
            probe_train_candidate(config, request, &loader_config, &manifest, &train_device)?
        }
        ProbeKind::Validation => {
            probe_validation_candidate(config, request, &loader_config, &manifest, &train_device)?
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
    write_probe_result(
        result_path,
        &ProbeResult {
            kind: request.kind,
            candidate_microbatch: request.candidate_microbatch,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(measured_samples_per_second),
            elapsed_seconds: Some(elapsed_seconds),
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
    let interrupted = interrupt_flag()?;
    interrupted.store(false, Ordering::SeqCst);
    let mut child =
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
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| format!("failed to spawn preflight probe child: {err}"))?;
    let stdout_handle = child
        .stdout
        .take()
        .map(|stdout| spawn_output_forwarder(stdout, false));
    let stderr_handle = child
        .stderr
        .take()
        .map(|stderr| spawn_output_forwarder(stderr, true));
    if wait_for_probe_child(&mut child, interrupted.as_ref())?.is_none() {
        fs::remove_file(result_path).ok();
        if let Some(handle) = stdout_handle {
            let _ = join_output_forwarder(handle, "stdout");
        }
        if let Some(handle) = stderr_handle {
            let _ = join_output_forwarder(handle, "stderr");
        }
        return Err("preflight interrupted; probe child terminated".to_string());
    }
    let stdout = match stdout_handle {
        Some(handle) => join_output_forwarder(handle, "stdout")?,
        None => Vec::new(),
    };
    let stderr = match stderr_handle {
        Some(handle) => join_output_forwarder(handle, "stderr")?,
        None => Vec::new(),
    };
    let status = child
        .try_wait()
        .map_err(|err| format!("failed to query preflight probe child status: {err}"))?
        .ok_or_else(|| "preflight probe child exited without final status".to_string())?;
    let output = child_output(status, stdout, stderr);

    if result_path.exists() {
        let result = read_probe_result(result_path)?;
        fs::remove_file(result_path).ok();
        return Ok(result);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    let stderr = stderr.trim();
    let combined = format!("stdout={stdout} stderr={stderr}");
    let status = classify_probe_detail(&combined);
    let detail = probe_failure_detail(status.clone(), stdout, stderr, output.status.code());
    Ok(ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status,
        measured_samples_per_second: None,
        elapsed_seconds: None,
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

pub(super) fn format_probe_selection_summary(
    kind: ProbeKind,
    summary: &ProbeCandidateSummary,
) -> String {
    format!(
        "selected {} microbatch={} avg_throughput={:.2} samples/s avg_elapsed={:.2}s attempts={}",
        probe_kind_name(kind),
        summary.candidate_microbatch,
        summary.average_samples_per_second.unwrap_or(0.0),
        summary.average_elapsed_seconds.unwrap_or(0.0),
        summary.attempts,
    )
}

pub(super) fn best_probe_summary(results: &[ProbeResult]) -> Option<ProbeCandidateSummary> {
    summarize_probe_results(results)
        .into_iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .max_by(|left, right| {
            left.average_samples_per_second
                .unwrap_or(0.0)
                .partial_cmp(&right.average_samples_per_second.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.candidate_microbatch.cmp(&right.candidate_microbatch))
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
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind={} candidates={:?} required_successes={}",
                probe_kind_name(kind),
                candidate_list,
                config.preflight.required_successes.max(1)
            )
        )
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
            let attempt_number = attempt + 1;
            println!(
                "{}",
                format_status_line(
                    &format!("[preflight:{}]", probe_kind_name(kind)),
                    format!(
                        "candidate_mb={} attempt={}/{} stage=starting",
                        candidate, attempt_number, attempts,
                    )
                )
            );
            progress.set_message(format_probe_attempt_message(
                kind,
                candidate,
                attempt_number,
                attempts,
            ));
            let request = ProbeRequest {
                kind,
                candidate_microbatch: candidate,
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            };
            let result_path = probe_result_path(artifacts, kind, candidate, attempt);
            println!(
                "{}",
                format_status_line(
                    &format!("[preflight:{}]", probe_kind_name(kind)),
                    format!(
                        "candidate_mb={} attempt={}/{} stage=running probe",
                        candidate, attempt_number, attempts,
                    )
                )
            );
            let result = execute_probe_request(config_path, request, &result_path)?;
            let passed = result.status == ProbeStatus::Success;
            progress.inc(1);
            println!("{}", format_probe_result_summary(&result));
            results.push(result);
            if !passed {
                println!(
                    "{}",
                    format_status_line(
                        &format!("[preflight:{}]", probe_kind_name(kind)),
                        format!(
                            "candidate_mb={} attempt={}/{} stage=backing off",
                            candidate, attempt_number, attempts,
                        )
                    )
                );
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
    progress.finish_with_message(
        format!("preflight {} ladder complete", probe_kind_name(kind))
            .green()
            .to_string(),
    );

    if use_explicit_only {
        return Err(format!(
            "explicit {} microbatch {} failed preflight",
            probe_kind_name(kind),
            explicit_candidate.unwrap_or(1)
        ));
    }

    let selected_summary = best_probe_summary(&stable_results).ok_or_else(|| {
        format!(
            "no stable {} microbatch found in preflight",
            probe_kind_name(kind)
        )
    })?;
    if !stable_results.is_empty() {
        println!(
            "{}",
            format_preflight_selection_line(format_probe_selection_summary(
                kind,
                &selected_summary
            ))
        );
    }
    Ok((selected_summary.candidate_microbatch, results))
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
    let phase_pb = make_bar(5, "[{bar:30.magenta/black}] {pos}/{len} {msg}")?;
    phase_pb.set_message(preflight_phase_label("train microbatch probe"));

    let train_seed = config
        .microbatch_size
        .unwrap_or_else(|| candidate_ladder(&config.preflight, config.batch_size)[0]);
    let (train_microbatch, train_probe_results) =
        search_train_microbatch(config_path, config, artifacts, train_seed)?;
    phase_pb.inc(1);
    phase_pb.set_message(preflight_phase_label("validation microbatch probe"));
    let validation_seed = config.validation_microbatch_size.unwrap_or(train_seed);
    let (validation_microbatch, validation_probe_results) =
        search_validation_microbatch(config_path, config, artifacts, validation_seed)?;
    phase_pb.inc(1);
    phase_pb.set_message(preflight_phase_label("resolve runtime"));
    let selected = resolve_runtime_config(
        config.batch_size,
        explicit,
        train_microbatch,
        validation_microbatch,
    );
    println!(
        "{}",
        format_timed_phase_message(
            "post_validation",
            "selected validation candidate; preparing runtime tuning",
            0.0,
        )
    );
    phase_pb.inc(1);
    phase_pb.set_message(preflight_phase_label("scan runtime data"));
    let mut tuned_config = config.clone();
    tuned_config.microbatch_size = Some(selected.train_microbatch_size);
    tuned_config.validation_microbatch_size = Some(selected.validation_microbatch_size);
    let manifest = scan_data_sources_with_progress(&config.data_dir, config.train_fraction, None)
        .map_err(|err| {
        format!(
            "failed to scan preflight runtime data from {}: {err}",
            config.data_dir.display()
        )
    })?;
    phase_pb.inc(1);
    phase_pb.set_message(preflight_phase_label("loader runtime tuning"));
    let train_device = train_device(&config.device);
    let loader = autotune_loader_runtime(&tuned_config, &manifest, &train_device)?;
    let runtime = EffectiveRuntimeConfig { selected, loader };
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry { cache_key, runtime },
    )?;
    phase_pb.inc(1);
    phase_pb.finish_with_message("preflight complete".green().to_string());
    Ok(PreflightRuntime {
        runtime,
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
    fn preflight_cache_key_changes_only_for_workload_relevant_inputs() {
        let config = dummy_config();
        let model = HydraModelConfig::learner();
        let baseline = preflight_cache_key(&config, &model, "cpu");

        let mut threaded = config.clone();
        threaded.num_threads = Some(8);
        assert_eq!(baseline, preflight_cache_key(&threaded, &model, "cpu"));

        let mut buffered = config.clone();
        buffered.buffer_samples += 1;
        assert_eq!(baseline, preflight_cache_key(&buffered, &model, "cpu"));

        let mut validation_limited = config.clone();
        validation_limited.max_validation_batches = Some(4);
        assert_ne!(
            baseline,
            preflight_cache_key(&validation_limited, &model, "cpu")
        );
    }

    #[test]
    fn loader_runtime_config_uses_deterministic_auto_threads_when_unset() {
        let config = dummy_config();
        let loader = autotune_loader_runtime(
            &config,
            &DataManifest {
                sources: Vec::new(),
                total_games: 0,
                train_count: 0,
                val_count: 0,
                counts_exact: false,
            },
            &LibTorchDevice::Cpu,
        );
        assert!(loader.is_err());
        let effective = loader_runtime_config(&config);
        assert!(effective.num_threads.is_some());
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
    fn dynamic_probe_ladder_grows_up_then_sweeps_down() {
        let mut config = dummy_config();
        config.batch_size = 512;
        config.preflight.candidate_microbatches = vec![512, 256, 128, 64, 32, 16];
        let ladder = dynamic_probe_ladder(&config, ProbeKind::Train, 64);
        assert_eq!(ladder, vec![64, 128, 256, 512, 32, 16]);
    }

    #[test]
    fn dynamic_validation_probe_ladder_can_grow_past_batch_size() {
        let mut config = dummy_config();
        config.batch_size = 512;
        config.preflight.candidate_microbatches = vec![512, 256, 128, 64, 32, 16];
        let ladder = dynamic_probe_ladder(&config, ProbeKind::Validation, 512);
        assert!(ladder.starts_with(&[512, 1024, 2048, 4096]));
        assert!(ladder.contains(&256));
        assert!(ladder.contains(&16));
    }

    #[test]
    fn local_refinement_candidates_include_midpoints_around_winner() {
        let summaries = vec![
            ProbeCandidateSummary {
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(570.0),
                average_elapsed_seconds: Some(16.0),
            },
            ProbeCandidateSummary {
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(520.0),
                average_elapsed_seconds: Some(18.0),
            },
            ProbeCandidateSummary {
                candidate_microbatch: 32,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(500.0),
                average_elapsed_seconds: Some(18.0),
            },
        ];
        let candidates = local_refinement_candidates(&summaries, 8, 3, 256);
        assert_eq!(candidates, vec![40, 56]);
    }

    #[test]
    fn local_refinement_candidates_skip_small_gaps() {
        let summaries = vec![
            ProbeCandidateSummary {
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(570.0),
                average_elapsed_seconds: Some(16.0),
            },
            ProbeCandidateSummary {
                candidate_microbatch: 72,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(560.0),
                average_elapsed_seconds: Some(17.0),
            },
        ];
        let candidates = local_refinement_candidates(&summaries, 16, 3, 256);
        assert!(candidates.is_empty());
    }

    #[test]
    fn local_refinement_candidates_include_success_failure_boundary_midpoint() {
        let summaries = vec![
            ProbeCandidateSummary {
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(570.0),
                average_elapsed_seconds: Some(16.0),
            },
            ProbeCandidateSummary {
                candidate_microbatch: 64,
                status: ProbeStatus::Oom,
                attempts: 1,
                average_samples_per_second: None,
                average_elapsed_seconds: None,
            },
            ProbeCandidateSummary {
                candidate_microbatch: 32,
                status: ProbeStatus::Success,
                attempts: 2,
                average_samples_per_second: Some(520.0),
                average_elapsed_seconds: Some(18.0),
            },
        ];
        let candidates = local_refinement_candidates(&summaries, 8, 3, 256);
        assert_eq!(candidates, vec![40, 56]);
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
            elapsed_seconds: Some(1.5),
            detail: String::new(),
        });
        assert!(success.contains("candidate_mb=192"));
        assert!(success.contains("1234.50 samples/s"));
        assert!(success.contains("elapsed=1.50s"));

        let oom = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 256,
            status: ProbeStatus::Oom,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: String::new(),
        });
        assert!(oom.contains("[train] candidate_mb=256 outcome=oom next=smaller_microbatch"));
    }

    #[test]
    fn summarize_probe_results_averages_all_successful_attempts_for_candidate() {
        let summaries = summarize_probe_results(&[
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(400.0),
                elapsed_seconds: Some(2.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(500.0),
                elapsed_seconds: Some(3.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(450.0),
                elapsed_seconds: Some(2.5),
                detail: String::new(),
            },
        ]);
        assert_eq!(summaries[0].candidate_microbatch, 64);
        assert_eq!(summaries[0].attempts, 2);
        assert_eq!(summaries[0].average_samples_per_second, Some(450.0));
        assert_eq!(summaries[0].average_elapsed_seconds, Some(2.5));
    }

    #[test]
    fn best_probe_summary_prefers_higher_average_not_single_spike() {
        let summary = best_probe_summary(&[
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(400.0),
                elapsed_seconds: Some(2.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(500.0),
                elapsed_seconds: Some(2.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(470.0),
                elapsed_seconds: Some(2.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(480.0),
                elapsed_seconds: Some(2.0),
                detail: String::new(),
            },
        ])
        .expect("best summary should exist");
        assert_eq!(summary.candidate_microbatch, 48);
        assert_eq!(summary.average_samples_per_second, Some(475.0));
    }
}
