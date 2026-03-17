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

use super::artifacts::{
    write_preflight_cache, BcArtifactPaths, PreflightPaths, RlArtifactPaths, RlPreflightPaths,
};
use super::config::{
    configure_threads, default_num_threads_for_system, train_device,
    trainer_config_from_train_config, ProbeChildRequest, TrainConfig,
};
use super::loss_policy::build_loss_config;
use super::preflight_fingerprint::preflight_cache_key;
use super::presentation::{
    format_preflight_selection_line, format_preflight_summary_line, format_probe_progress_line,
    format_probe_status_line, format_timed_phase_message, make_bar, make_spinner,
    preflight_phase_label,
};
use super::probe_ladder::{candidate_average, dynamic_probe_ladder, probe_only_candidate_ladder};
use super::probe_process::{
    mem_available_bytes, probe_result_path, rl_probe_required_free_bytes, rl_probe_result_path,
    write_probe_result,
};
use super::probe_request::{probe_child_request_from_cli, ProbeRequest};
use super::probe_search::{
    finalize_probe_search, maybe_expand_probe_candidates, probe_candidate_ladder,
    refine_probe_winner_locally, refine_top_k_probe_candidates_locally, rerun_probe_finalists,
    run_candidate_attempts, ProbeGrowthDecision, ProbeGrowthState, ProbeRunSpec,
};
use super::probe_summary::{best_probe_summary, format_probe_selection_summary, probe_kind_name};
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

pub(super) struct RlPreflightRuntime {
    pub(super) selected_games_per_batch: usize,
    pub(super) selected_microbatch_size: usize,
    pub(super) rl_games_probe_results: Vec<ProbeResult>,
    pub(super) rl_microbatch_probe_results: Vec<ProbeResult>,
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
    let mut last_successful_candidate: Option<usize> = None;

    for candidate in candidates {
        if let Some(blocked) = maybe_block_host_ram_growth_probe(
            config,
            ProbeKind::Train,
            candidate,
            last_successful_candidate,
        ) {
            println!("{}", format_probe_status_line(&blocked));
            results.push(blocked);
            continue;
        }
        let stable_start = results.len();
        let mut result_path_for =
            |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt);
        let passed = run_candidate_attempts(
            config_path,
            &mut result_path_for,
            ProbeRunSpec {
                kind: ProbeKind::Train,
                candidate,
                attempts: config.preflight.required_successes.max(1),
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            },
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
        last_successful_candidate = Some(candidate);
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
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        ProbeKind::Train,
        config,
        &mut results,
        &progress,
    )?;
    refine_top_k_probe_candidates_locally(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
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
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
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
    let mut last_successful_candidate: Option<usize> = None;

    let mut index = 0usize;
    while index < candidates.len() {
        let candidate = candidates[index];
        if let Some(blocked) = maybe_block_host_ram_growth_probe(
            config,
            ProbeKind::Validation,
            candidate,
            last_successful_candidate,
        ) {
            println!("{}", format_probe_status_line(&blocked));
            results.push(blocked);
            break;
        }
        let stable_start = results.len();
        let mut result_path_for =
            |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt);
        let passed = run_candidate_attempts(
            config_path,
            &mut result_path_for,
            ProbeRunSpec {
                kind: ProbeKind::Validation,
                candidate,
                attempts: config.preflight.required_successes.max(1),
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            },
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
        last_successful_candidate = Some(candidate);
        if use_explicit_only {
            progress
                .finish_with_message("preflight validation ladder complete".green().to_string());
            return Ok((candidate, results));
        }

        let summary = best_probe_summary(&stable_results)
            .ok_or_else(|| "no stable validation microbatch found in preflight".to_string())?;
        let candidate_score = candidate_average(&results, candidate).unwrap_or(0.0);
        let mut growth_state = ProbeGrowthState {
            patience: growth_patience,
            steps: growth_steps,
            prior_best_score,
        };
        if maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index,
                kind: ProbeKind::Validation,
                candidate,
                summary: &summary,
                candidate_score,
                tolerance,
            },
            config,
            &mut growth_state,
        ) {
            break;
        }
        growth_patience = growth_state.patience;
        growth_steps = growth_state.steps;
        prior_best_score = growth_state.prior_best_score;
        index += 1;
    }

    progress.finish_with_message("preflight validation ladder complete".green().to_string());
    let selected_summary = finalize_probe_search(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        ProbeKind::Validation,
        config,
        &mut results,
        &progress,
        "no stable validation microbatch found in preflight".to_string(),
    )?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            ProbeKind::Validation,
            &selected_summary,
        ))
    );
    Ok((selected_summary.candidate_microbatch, results))
}

fn search_rl_runtime_candidate(
    config_path: &Path,
    config: &TrainConfig,
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
    let use_explicit_only = explicit_candidate.is_some()
        && !config.preflight.allow_override_explicit_microbatch
        && !matches!(kind, ProbeKind::RlGames);
    let mut candidates = if use_explicit_only {
        vec![explicit_candidate.unwrap_or(seed.max(1)).max(1)]
    } else {
        dynamic_probe_ladder(config, kind, seed.max(1))
    };
    let mut seen = BTreeSet::new();
    candidates.retain(|candidate| seen.insert(*candidate));
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight ladder:",
            format!(
                "kind={} candidates={:?} required_successes={} growth_patience={} growth_max_steps={}",
                probe_kind_name(kind),
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
    let mut last_successful_candidate: Option<usize> = None;

    let mut index = 0usize;
    while index < candidates.len() {
        let candidate = candidates[index];
        if let Some(blocked) =
            maybe_block_host_ram_growth_probe(config, kind, candidate, last_successful_candidate)
        {
            println!("{}", format_probe_status_line(&blocked));
            results.push(blocked);
            break;
        }
        let stable_start = results.len();
        let mut result_path_for =
            |kind, candidate, attempt| rl_probe_result_path(artifacts, kind, candidate, attempt);
        let passed = run_candidate_attempts(
            config_path,
            &mut result_path_for,
            ProbeRunSpec {
                kind,
                candidate,
                attempts: config.preflight.required_successes.max(1),
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps,
            },
            &mut results,
            &progress,
        )?;
        if !passed {
            if use_explicit_only {
                progress.finish_with_message(
                    format!("preflight {} ladder complete", probe_kind_name(kind))
                        .green()
                        .to_string(),
                );
                return Err(format!(
                    "explicit {} candidate {} failed preflight",
                    probe_kind_name(kind),
                    candidate
                ));
            }
            break;
        }
        stable_results.extend(results[stable_start..].iter().cloned());
        last_successful_candidate = Some(candidate);
        if use_explicit_only {
            progress.finish_with_message(
                format!("preflight {} ladder complete", probe_kind_name(kind))
                    .green()
                    .to_string(),
            );
            return Ok((candidate, results));
        }

        let summary = best_probe_summary(&stable_results).ok_or_else(|| {
            format!(
                "no stable {} candidate found in preflight",
                probe_kind_name(kind)
            )
        })?;
        let candidate_score = candidate_average(&results, candidate).unwrap_or(0.0);
        let mut growth_state = ProbeGrowthState {
            patience: growth_patience,
            steps: growth_steps,
            prior_best_score,
        };
        if maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index,
                kind,
                candidate,
                summary: &summary,
                candidate_score,
                tolerance,
            },
            config,
            &mut growth_state,
        ) {
            break;
        }
        growth_patience = growth_state.patience;
        growth_steps = growth_state.steps;
        prior_best_score = growth_state.prior_best_score;
        index += 1;
    }

    progress.finish_with_message(
        format!("preflight {} ladder complete", probe_kind_name(kind))
            .green()
            .to_string(),
    );
    let selected_summary = finalize_probe_search(
        config_path,
        |kind, candidate, attempt| rl_probe_result_path(artifacts, kind, candidate, attempt),
        kind,
        config,
        &mut results,
        &progress,
        format!(
            "no stable {} candidate found in preflight",
            probe_kind_name(kind)
        ),
    )?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(kind, &selected_summary,))
    );
    Ok((selected_summary.candidate_microbatch, results))
}

fn maybe_block_host_ram_growth_probe(
    config: &TrainConfig,
    kind: ProbeKind,
    candidate: usize,
    baseline_candidate: Option<usize>,
) -> Option<ProbeResult> {
    let baseline = baseline_candidate?;
    if candidate <= baseline {
        return None;
    }
    let available = mem_available_bytes()?;
    let required_free = rl_probe_required_free_bytes(config)?;
    let scale = candidate as f64 / baseline as f64;
    let estimated_probe_bytes =
        ((available as f64) * scale * config.preflight.rl_probe_growth_safety_factor.max(1.0))
            .ceil() as u64;
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
            config.preflight.rl_probe_growth_safety_factor.max(1.0),
        ),
    })
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
            let Some((obs, batch)) =
                hydra_train::data::sample::collate_batch_samples::<ValidBackend>(
                    chunk,
                    false,
                    train_device,
                )
                .map_err(|err| format!("preflight validation collation failed: {err}"))?
            else {
                continue;
            };
            let targets = batch.to_hydra_targets();
            let output = model_valid.forward(obs);
            let _ = validation_batch_stats(
                chunk.len(),
                &output,
                &batch,
                &targets,
                &loss_fn,
                &hydra_train::training::bc::BcExitConfig::default(),
            );
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

fn run_rl_probe_only(
    config: &TrainConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
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
                tuned.microbatch_size = Some(hydra_train::training::rl::DEFAULT_RL_MICROBATCH_SIZE);
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
    let measured_samples_per_second =
        super::runtime_autotune::measure_rl_runtime_throughput(config, &tuned, &train_device)?;
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
            detail: String::new(),
        },
    )
}

pub(super) fn run_probe_only(
    config: &TrainConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    configure_threads(config.num_threads)
        .map_err(|err| format!("failed to configure rayon threads for probe child: {err}"))?;

    if matches!(request.kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return run_rl_probe_only(config, request, result_path);
    }

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
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
    let train_device = train_device(&config.device)?;
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => {
            probe_train_candidate(config, request, &loader_config, &manifest, &train_device)?
        }
        ProbeKind::Validation => {
            probe_validation_candidate(config, request, &loader_config, &manifest, &train_device)?
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
    let train_device = train_device(&config.device)?;
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

pub(super) fn run_rl_preflight(
    config_path: &Path,
    config: &TrainConfig,
    _train_device: &LibTorchDevice,
) -> Result<RlPreflightRuntime, String> {
    let rl = config
        .rl
        .as_ref()
        .ok_or_else(|| "RL preflight requested without rl config block".to_string())?;
    let started = Instant::now();
    let artifacts = RlArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    let paths = RlPreflightPaths::new(&artifacts);
    let cache_key = preflight_cache_key(
        config,
        &HydraModelConfig::learner(),
        &config.device,
        default_num_threads_for_system(),
    );
    println!(
        "{}",
        format_timed_phase_message("rl_runtime_tuning", "starting", 0.0)
    );
    let (selected_games_per_batch, game_results) = search_rl_runtime_candidate(
        config_path,
        config,
        &artifacts,
        ProbeKind::RlGames,
        rl.games_per_batch,
    )?;
    let microbatch_seed = rl
        .microbatch_size
        .unwrap_or(hydra_train::training::rl::DEFAULT_RL_MICROBATCH_SIZE)
        .min(config.batch_size.max(1))
        .max(1);
    let (selected_microbatch_size, microbatch_results) = search_rl_runtime_candidate(
        config_path,
        config,
        &artifacts,
        ProbeKind::RlMicrobatch,
        microbatch_seed,
    )?;
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key,
            runtime: EffectiveRuntimeConfig {
                selected: hydra_train::preflight::SelectedRuntimeConfig {
                    train_microbatch_size: selected_microbatch_size,
                    validation_microbatch_size: config
                        .validation_microbatch_size
                        .unwrap_or(selected_microbatch_size),
                    accum_steps: config
                        .batch_size
                        .div_ceil(selected_microbatch_size.max(1))
                        .max(1),
                },
                loader: hydra_train::preflight::LoaderRuntimeConfig {
                    num_threads: config.num_threads,
                    buffer_games: selected_games_per_batch,
                    buffer_samples: config.buffer_samples,
                    archive_queue_bound: config.archive_queue_bound,
                },
            },
        },
    )?;
    println!(
        "{}",
        format_preflight_summary_line(
            "RL Preflight:",
            format!(
                "selected games_per_batch={} rl.microbatch_size={} (stored in preflight cache for RL runtime reuse)",
                selected_games_per_batch,
                selected_microbatch_size,
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
        rl_games_probe_results: game_results,
        rl_microbatch_probe_results: microbatch_results,
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
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            rl: None,
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
        assert!(oom.contains(
            "[train] candidate_mb=256 outcome=oom(generic) next=smaller_microbatch detail=n/a"
        ));
    }
}
