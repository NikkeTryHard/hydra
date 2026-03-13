use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
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
    candidate_ladder, resolve_runtime_config, EffectiveRuntimeConfig, ExplicitSettings,
    PreflightCacheEntry, ProbeKind, ProbeResult, ProbeStatus,
};
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{write_preflight_cache, BcArtifactPaths, PreflightPaths};
use super::config::{
    configure_threads, default_num_threads_for_system, train_device,
    trainer_config_from_train_config, ProbeChildRequest, TrainConfig,
};
use super::loss_policy::build_loss_config;
use super::preflight_fingerprint::preflight_cache_key;
use super::presentation::{
    format_preflight_selection_line, format_preflight_summary_line, format_probe_progress_line,
    format_probe_status_line, format_status_line, format_timed_phase_message, make_bar,
    make_spinner, preflight_phase_label,
};
use super::probe_ladder::{
    candidate_average, close_probe_finalists, dynamic_probe_ceiling, dynamic_probe_ladder,
    local_refinement_candidates, probe_only_candidate_ladder,
};
use super::probe_process::{probe_result_path, write_probe_result};
use super::probe_request::{probe_child_request_from_cli, ProbeRequest};
use super::probe_summary::{
    best_probe_summary, format_probe_selection_summary, probe_kind_name, summarize_probe_results,
};
use super::runtime_autotune::autotune_loader_runtime;
use super::schedule::effective_lr;
use super::validation::validation_batch_stats;
use super::{TrainBackend, ValidBackend};

pub(super) struct PreflightRuntime {
    pub(super) runtime: EffectiveRuntimeConfig,
    pub(super) train_probe_results: Vec<ProbeResult>,
    pub(super) validation_probe_results: Vec<ProbeResult>,
    pub(super) explicit: ExplicitSettings,
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
    let successful_summaries = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .cloned()
        .collect::<Vec<_>>();
    for candidate in candidates {
        let seconds_per_step = successful_summaries
            .iter()
            .min_by_key(|summary| summary.candidate_microbatch.abs_diff(candidate))
            .and_then(|summary| summary.average_elapsed_seconds)
            .map(|elapsed| {
                elapsed
                    / (config.preflight.warmup_steps + config.preflight.measure_steps).max(1) as f64
            })
            .unwrap_or_else(|| {
                config.preflight.target_measure_seconds
                    / config.preflight.measure_steps.max(1) as f64
            });
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
    configure_threads(config.num_threads)
        .map_err(|err| format!("failed to configure rayon threads for probe child: {err}"))?;
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
    super::probe_process::execute_probe_request(
        config_path,
        request,
        result_path,
        classify_probe_detail,
    )
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
    let cache_key = preflight_cache_key(
        config,
        model_config,
        device_label,
        default_num_threads_for_system(),
    );
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
    use std::path::PathBuf;

    use super::*;
    use crate::config::loader_runtime_config;
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
        let baseline = preflight_cache_key(&config, &model, "cpu", 8);

        let mut threaded = config.clone();
        threaded.num_threads = Some(8);
        assert_eq!(baseline, preflight_cache_key(&threaded, &model, "cpu", 8));

        let mut buffered = config.clone();
        buffered.buffer_samples += 1;
        assert_eq!(baseline, preflight_cache_key(&buffered, &model, "cpu", 8));

        let mut validation_limited = config.clone();
        validation_limited.max_validation_batches = Some(4);
        assert_ne!(
            baseline,
            preflight_cache_key(&validation_limited, &model, "cpu", 8)
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
}
