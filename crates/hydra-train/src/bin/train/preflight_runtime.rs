use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use burn::backend::libtorch::LibTorchDevice;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;
use hydra_train::data::pipeline::{
    scan_data_sources_with_progress, stream_train_epoch, stream_val_pass, DataManifest,
    StreamingLoaderConfig,
};
use hydra_train::data::sample::{collate_batch_samples, collate_samples, MjaiSample};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::preflight::{
    candidate_ladder, resolve_runtime_config, BenchmarkMetadata, BenchmarkMode, BenchmarkResult,
    BenchmarkRuntimeConfig, BenchmarkScore, EffectiveRuntimeConfig, ExplicitSettings,
    LoaderRuntimeConfig, PreflightCacheEntry, ProbeKind, ProbeResult, ProbeStatus,
};
use hydra_train::training::bc::{bc_total_with_exit, gated_bc_context};
use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train::training::losses::HydraLoss;
use tboard::EventWriter;

use super::artifacts::{
    append_step_log, log_tensorboard, save_latest_checkpoint_and_state, write_preflight_cache,
    BcArtifactPaths, LatestCheckpointState, PreflightBenchmarkPaths, PreflightPaths,
    RlArtifactPaths, RlPreflightPaths,
};
use super::config::{
    configure_threads, default_num_threads_for_system, train_device,
    trainer_config_from_train_config, ProbeChildRequest, TrainConfig,
};
use super::loss_policy::{build_bc_exit_config, build_loss_config};
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
use super::probe_summary::{
    best_probe_summary, format_probe_selection_summary, probe_kind_name, summarize_probe_results,
    ProbeCandidateSummary,
};
use super::progress::{ScalarAverages, StepLogEntry};
use super::resume::{runtime_resume_contract, BestValidation, EpochContinuation};
use super::runtime_autotune::{autotune_ranked_loader_runtime, RankedLoaderRuntime};
use super::schedule::effective_lr;
use super::validation::{
    run_validation, validation_batch_stats, ValidationContext, ValidationRuntime, ValidationSummary,
};
use super::{TrainBackend, ValidBackend};

type BenchmarkOptimizer = OptimizerAdaptor<Adam, HydraModel<TrainBackend>, TrainBackend>;

pub(super) struct PreflightRuntime {
    pub(super) runtime: EffectiveRuntimeConfig,
    pub(super) train_probe_results: Vec<ProbeResult>,
    pub(super) validation_probe_results: Vec<ProbeResult>,
    pub(super) benchmark: Option<BenchmarkResult>,
    pub(super) explicit: ExplicitSettings,
}

pub(super) struct RlPreflightRuntime {
    pub(super) selected_games_per_batch: usize,
    pub(super) selected_microbatch_size: usize,
    pub(super) rl_games_probe_results: Vec<ProbeResult>,
    pub(super) rl_microbatch_probe_results: Vec<ProbeResult>,
}

#[derive(Debug, Clone)]
struct BenchmarkFinalist {
    runtime: BenchmarkRuntimeConfig,
    train_probe_samples_per_second: f64,
    validation_probe_samples_per_second: f64,
    loader_probe_samples_per_second: f64,
}

struct TrainBenchmarkOutcome {
    model: HydraModel<TrainBackend>,
    optimizer: BenchmarkOptimizer,
    head_controller: HeadActivationController,
    stats: ScalarAverages,
    elapsed_seconds: f64,
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
            .then_with(|| left.candidate_microbatch.cmp(&right.candidate_microbatch))
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
        for neighbor in [idx.saturating_sub(1), idx, (idx + 1).min(summaries.len() - 1)] {
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
            .then_with(|| left.candidate_microbatch.cmp(&right.candidate_microbatch))
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

fn build_stage_two_finalists(
    config: &TrainConfig,
    selected: &EffectiveRuntimeConfig,
    train_candidates: &[ProbeCandidateSummary],
    validation_candidates: &[ProbeCandidateSummary],
    loader_candidates: &[RankedLoaderRuntime],
    train_probe_results: &[ProbeResult],
    validation_probe_results: &[ProbeResult],
    ranked_loaders: &[RankedLoaderRuntime],
) -> Vec<BenchmarkFinalist> {
    let mut finalists = Vec::new();
    let mut seen = BTreeSet::new();
    for train_summary in train_candidates {
        for validation_summary in validation_candidates {
            for loader in loader_candidates {
                let runtime = BenchmarkRuntimeConfig {
                    train_microbatch_size: train_summary
                        .candidate_microbatch
                        .min(config.batch_size),
                    validation_microbatch_size: validation_summary.candidate_microbatch.max(1),
                    accum_steps: config
                        .batch_size
                        .div_ceil(train_summary.candidate_microbatch.max(1))
                        .max(1),
                    loader: loader.loader,
                };
                let key = (
                    runtime.train_microbatch_size,
                    runtime.validation_microbatch_size,
                    runtime.accum_steps,
                    runtime.loader.archive_queue_bound,
                    runtime.loader.buffer_samples,
                    runtime.loader.buffer_games,
                    runtime.loader.num_threads.unwrap_or(0),
                );
                if seen.insert(key) {
                    finalists.push(BenchmarkFinalist {
                        runtime,
                        train_probe_samples_per_second: train_summary
                            .average_samples_per_second
                            .unwrap_or(0.0),
                        validation_probe_samples_per_second: validation_summary
                            .average_samples_per_second
                            .unwrap_or(0.0),
                        loader_probe_samples_per_second: loader.train_samples_per_second,
                    });
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
                left.runtime
                    .train_microbatch_size
                    .cmp(&right.runtime.train_microbatch_size)
            })
    });
    finalists.truncate(config.preflight.real_benchmark_max_finalists.max(1));
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
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
    }
}

fn benchmark_train_config(config: &TrainConfig, runtime: BenchmarkRuntimeConfig) -> TrainConfig {
    let mut tuned = config.clone();
    tuned.microbatch_size = Some(runtime.train_microbatch_size);
    tuned.validation_microbatch_size = Some(runtime.validation_microbatch_size);
    tuned.num_threads = runtime.loader.num_threads;
    tuned.buffer_games = runtime.loader.buffer_games;
    tuned.buffer_samples = runtime.loader.buffer_samples;
    tuned.archive_queue_bound = runtime.loader.archive_queue_bound;
    tuned
}

fn benchmark_validation_config(
    config: &TrainConfig,
    runtime: BenchmarkRuntimeConfig,
) -> TrainConfig {
    let mut tuned = benchmark_train_config(config, runtime);
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

fn benchmark_projected_events(train_steps: usize, interval: usize) -> f64 {
    train_steps as f64 / interval.max(1) as f64
}

fn benchmark_metadata(
    config: &TrainConfig,
    train_candidates: usize,
    validation_candidates: usize,
    loader_candidates: usize,
    finalists_benchmarked: usize,
) -> BenchmarkMetadata {
    let measured_train_steps = config.preflight.real_benchmark_train_steps.max(1);
    BenchmarkMetadata {
        mode: BenchmarkMode::CadenceAwareProjection,
        selection_metric: "wall_clock_effective_throughput".to_string(),
        train_probe_candidates_considered: train_candidates,
        validation_probe_candidates_considered: validation_candidates,
        loader_candidates_considered: loader_candidates,
        finalists_benchmarked,
        warmup_steps: config.preflight.real_benchmark_warmup_steps.max(1),
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

fn benchmark_score(
    config: &TrainConfig,
    train_seconds: f64,
    validation_seconds: f64,
    checkpoint_seconds: f64,
    logging_seconds: f64,
    validation_samples: usize,
) -> BenchmarkScore {
    let train_steps = config.preflight.real_benchmark_train_steps.max(1);
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
    BenchmarkScore {
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
    }
}

fn benchmark_train_window(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<TrainBenchmarkOutcome, String> {
    let train_cfg = trainer_config_from_train_config(config);
    let mut model = HydraModelConfig::learner().init::<TrainBackend>(train_device);
    let mut optimizer: BenchmarkOptimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let mut head_controller = HeadActivationController::new(
        HeadActivationConfig::default_with_params(HydraModelConfig::learner().estimated_params()),
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
    let warmup_steps = config.preflight.real_benchmark_warmup_steps.max(1);
    let measured_train_steps = config.preflight.real_benchmark_train_steps.max(1);
    let target_steps = warmup_steps + measured_train_steps;
    let mut completed_steps = 0usize;
    let mut pending_samples = std::collections::VecDeque::new();
    let mut measured_stats = ScalarAverages::default();
    let mut measure_start = None;

    for buffer_result in stream_train_epoch(manifest, &loader, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("benchmark train stream failed: {err}"))?;
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let logical_batch_len = logical_batch.len().max(1) as f32;
            let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
                GradientsAccumulator::new();
            let mut step_batches = Vec::new();

            for chunk in logical_batch.chunks(microbatch_size) {
                let Some((obs, batch)) =
                    collate_batch_samples::<TrainBackend>(chunk, config.augment, train_device)
                        .map_err(|err| format!("benchmark train collation failed: {err}"))?
                else {
                    continue;
                };
                let targets = batch.to_hydra_targets();
                let (active_loss_fn, warmup_heads) =
                    gated_bc_context(Some(&mut head_controller), &loss_fn, &targets);
                let output =
                    model.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
                let breakdown = active_loss_fn.total_loss(&output, &targets);
                let total = bc_total_with_exit(&output, &batch, &targets, &active_loss_fn, &exit_cfg);
                step_batches.push(validation_batch_stats(
                    chunk.len(),
                    &output,
                    &batch,
                    &targets,
                    &breakdown,
                    &total,
                    &exit_cfg,
                ));
                let chunk_weight = chunk.len() as f32 / logical_batch_len;
                let grads = (total * chunk_weight).backward();
                let grads = GradientsParams::from_grads(grads, &model);
                accumulator.accumulate(&model, grads);
            }

            if !step_batches.is_empty() {
                let lr = effective_lr(&train_cfg, completed_steps, target_steps.max(1));
                let grads = accumulator.grads();
                model = optimizer.step(lr, model, grads);
                head_controller.tick_warmup();
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
                });
            }
        }
    }

    Err("not enough train data to finish stage-2 benchmark train window".to_string())
}

fn benchmark_validation_pass(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    outcome: &mut TrainBenchmarkOutcome,
) -> Result<(ValidationSummary, f64), String> {
    let valid_loss_fn =
        HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
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
    let started = Instant::now();
    let summary = run_validation(
        &outcome.model,
        ValidationContext {
            config,
            loader_config: &loader,
            manifest,
            cached_samples: None,
            device: train_device,
            loss_fn: &valid_loss_fn,
            exit_cfg: &exit_cfg,
        },
        ValidationRuntime {
            head_controller: Some(&mut outcome.head_controller),
            progress: None,
        },
    )?;
    Ok((summary, started.elapsed().as_secs_f64()))
}

fn benchmark_checkpoint_cost(
    artifacts: &BcArtifactPaths,
    config: &TrainConfig,
    outcome: &TrainBenchmarkOutcome,
) -> Result<f64, String> {
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
            global_step: config.preflight.real_benchmark_train_steps.max(1),
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
    train_stats: &ScalarAverages,
    validation_summary: &ValidationSummary,
) -> Result<f64, String> {
    let global_step = config.preflight.real_benchmark_train_steps.max(1);
    let lr = effective_lr(
        &trainer_config_from_train_config(config),
        global_step,
        (config.preflight.real_benchmark_warmup_steps
            + config.preflight.real_benchmark_train_steps)
            .max(1),
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
        val_total_loss: Some(validation_summary.total_loss),
        val_policy_loss: Some(validation_summary.policy_loss),
        val_policy_agreement: Some(validation_summary.agreement),
        val_delta_q_promotion: validation_summary.delta_q_promotion_snapshot,
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
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    artifacts: &BcArtifactPaths,
    finalists: &[BenchmarkFinalist],
    train_candidates: usize,
    validation_candidates: usize,
    loader_candidates: usize,
) -> Result<BenchmarkResult, String> {
    let benchmark_paths = PreflightBenchmarkPaths::new(artifacts);
    benchmark_paths.create_root_dir()?;
    let initial_count = finalists.len().min(config.preflight.real_benchmark_max_finalists.max(1));
    let mut finalists_to_benchmark = finalists[..initial_count].to_vec();
    let mut benchmarked = 0usize;
    let mut best: Option<BenchmarkResult> = None;
    let mut scored_results = Vec::new();
    let mut tie_expansion_triggered = false;
    let mut candidate_index = 0usize;
    while candidate_index < finalists_to_benchmark.len() {
        let finalist = &finalists_to_benchmark[candidate_index];
        let benchmark_config = benchmark_validation_config(config, finalist.runtime);
        let candidate_output_dir = benchmark_paths.create_candidate_dir(candidate_index)?;
        let candidate_artifacts = BcArtifactPaths::new(&candidate_output_dir, 0);
        candidate_artifacts.create_root_dir()?;
        let mut train_outcome = benchmark_train_window(&benchmark_config, manifest, train_device)?;
        let (validation_summary, validation_seconds) = benchmark_validation_pass(
            &benchmark_config,
            manifest,
            train_device,
            &mut train_outcome,
        )?;
        let checkpoint_seconds =
            benchmark_checkpoint_cost(&candidate_artifacts, &benchmark_config, &train_outcome)?;
        let logging_seconds = benchmark_logging_cost(
            &candidate_artifacts,
            &benchmark_config,
            &train_outcome.stats,
            &validation_summary,
        )?;
        let score = benchmark_score(
            &benchmark_config,
            train_outcome.elapsed_seconds,
            validation_seconds,
            checkpoint_seconds,
            logging_seconds,
            validation_summary.samples,
        );
        let result = BenchmarkResult {
            runtime: finalist.runtime,
            score,
            metadata: benchmark_metadata(
                config,
                train_candidates,
                validation_candidates,
                loader_candidates,
                benchmarked + 1,
            ),
        };
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight benchmark:",
                format!(
                    "candidate={} train_mb={} val_mb={} loader=({}, {}, {}, {:?}) effective={:.2} train_only={:.2}",
                    candidate_index + 1,
                    result.runtime.train_microbatch_size,
                    result.runtime.validation_microbatch_size,
                    result.runtime.loader.archive_queue_bound,
                    result.runtime.loader.buffer_samples,
                    result.runtime.loader.buffer_games,
                    result.runtime.loader.num_threads,
                    result.score.wall_clock_samples_per_second,
                    result.score.train_only_samples_per_second,
                ),
            )
        );
        if best.as_ref().is_none_or(|current| {
            result.score.wall_clock_samples_per_second > current.score.wall_clock_samples_per_second
        }) {
            best = Some(result.clone());
        }
        scored_results.push(result.clone());
        benchmarked += 1;
        candidate_index += 1;
        if candidate_index == finalists_to_benchmark.len() {
            let mut ranked = scored_results.clone();
            ranked.sort_by(|left, right| {
                right
                    .score
                    .wall_clock_samples_per_second
                    .partial_cmp(&left.score.wall_clock_samples_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if ranked.len() >= 2 {
                let best_score = ranked[0].score.wall_clock_samples_per_second;
                let next_score = ranked[1].score.wall_clock_samples_per_second;
                let tie_margin = config.preflight.real_benchmark_tie_margin_ratio.max(0.0);
                let threshold = best_score * (1.0 - tie_margin);
                if next_score >= threshold {
                    let current_len = finalists_to_benchmark.len();
                    let target_len = (current_len + config.preflight.real_benchmark_extra_finalists)
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
    let mut best = best
        .ok_or_else(|| "stage-2 preflight benchmark had no finalists to score".to_string())?;
    best.metadata.finalists_benchmarked = benchmarked;
    if tie_expansion_triggered {
        best.metadata.selection_metric.push_str(" + tie_expansion");
    }
    Ok(best)
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
            let breakdown = loss_fn.total_loss(&output, &targets);
            let total = hydra_train::training::bc::bc_total_with_exit(
                &output,
                &batch,
                &targets,
                &loss_fn,
                &hydra_train::training::bc::BcExitConfig::default(),
            );
            let _ = validation_batch_stats(
                chunk.len(),
                &output,
                &batch,
                &targets,
                &breakdown,
                &total,
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
    let phase_pb = make_bar(6, "[{bar:30.magenta/black}] {pos}/{len} {msg}")?;
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
    let ranked_loaders = autotune_ranked_loader_runtime(
        &tuned_config,
        &manifest,
        &train_device,
        config.preflight.real_benchmark_loader_candidates.max(1),
    )?;
    let loader = ranked_loaders
        .first()
        .map(|ranked| ranked.loader)
        .ok_or_else(|| "loader runtime autotune returned no ranked candidates".to_string())?;
    let mut runtime = EffectiveRuntimeConfig { selected, loader };
    phase_pb.inc(1);
    phase_pb.set_message(preflight_phase_label("stage-2 finalist benchmark"));
    let benchmark = if config.preflight.real_benchmark_enabled {
        let train_candidates = diverse_probe_candidates(
            &train_probe_results,
            selected.train_microbatch_size,
            config.preflight.real_benchmark_train_candidates,
            config.preflight.finalist_margin_ratio,
        );
        let validation_candidates = diverse_probe_candidates(
            &validation_probe_results,
            selected.validation_microbatch_size,
            config.preflight.real_benchmark_validation_candidates,
            config.preflight.finalist_margin_ratio,
        );
        let loader_candidates = select_loader_finalists(
            &ranked_loaders,
            config.preflight.real_benchmark_loader_candidates,
            config.preflight.finalist_margin_ratio,
            runtime.loader,
        );
        let finalists = build_stage_two_finalists(
            config,
            &runtime,
            &train_candidates,
            &validation_candidates,
            &loader_candidates,
            &train_probe_results,
            &validation_probe_results,
            &ranked_loaders,
        );
        let best = run_stage_two_finalist_benchmark(
            config,
            &manifest,
            &train_device,
            artifacts,
            &finalists,
            train_candidates.len(),
            validation_candidates.len(),
            loader_candidates.len(),
        )?;
        runtime = EffectiveRuntimeConfig {
            selected: resolve_runtime_config(
                config.batch_size,
                explicit,
                best.runtime.train_microbatch_size,
                best.runtime.validation_microbatch_size,
            ),
            loader: best.runtime.loader,
        };
        println!(
            "{}",
            format_preflight_selection_line(format!(
                "stage-2 winner train_mb={} val_mb={} accum_steps={} wall_clock_effective={:.2} samples/s mode={:?}",
                best.runtime.train_microbatch_size,
                best.runtime.validation_microbatch_size,
                runtime.selected.accum_steps,
                best.score.wall_clock_samples_per_second,
                best.metadata.mode,
            ))
        );
        Some(best)
    } else {
        None
    };
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key,
            runtime,
            benchmark: benchmark.clone(),
        },
    )?;
    phase_pb.inc(1);
    phase_pb.finish_with_message("preflight complete".green().to_string());
    Ok(PreflightRuntime {
        runtime,
        train_probe_results,
        validation_probe_results,
        benchmark,
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
            benchmark: None,
        },
    )?;
    println!(
        "{}",
        format_preflight_summary_line(
            "RL Preflight:",
            format!(
                "selected games_per_batch={} rl.microbatch_size={} (stored in preflight cache for RL runtime reuse)",
                selected_games_per_batch, selected_microbatch_size,
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
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::config::{loader_runtime_config, ProbeChildRequest, ProbeCliRequest, RlTrainConfig};
    use hydra_train::preflight::{PreflightConfig, ProbeStatus};

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/home/nikketryhard/tmp/hydra-test-data"),
            output_dir: PathBuf::from("/home/nikketryhard/tmp/hydra-test-out"),
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

    fn unique_test_path(label: &str) -> PathBuf {
        let base = PathBuf::from("/home/nikketryhard/tmp");
        fs::create_dir_all(&base).expect("test temp root should be creatable");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        base.join(format!("hydra-preflight-runtime-{label}-{unique}"))
    }

    fn write_temp_file(label: &str, extension: &str, contents: &str) -> PathBuf {
        let path = unique_test_path(label).with_extension(extension);
        fs::write(&path, contents).expect("temporary test file should be writable");
        path
    }

    fn missing_test_path(label: &str) -> PathBuf {
        let path = unique_test_path(label);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&path);
        path
    }

    fn dummy_rl_train_config() -> RlTrainConfig {
        RlTrainConfig {
            games_per_batch: 8,
            microbatch_size: Some(16),
            ..RlTrainConfig::default()
        }
    }

    fn tiny_real_mjai_replay() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn write_real_probe_fixture(label: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = unique_test_path(label);
        fs::create_dir_all(&root).expect("create real probe fixture dir");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("write real probe replay");
        let result_path = root.join("probe-result.json");
        (root, replay_path, result_path)
    }

    #[test]
    fn measure_samples_per_second_handles_zero_samples_and_zero_time() {
        assert_eq!(measure_samples_per_second(0, Duration::from_secs(2)), 0.0);
        assert_eq!(measure_samples_per_second(10, Duration::from_secs(0)), 0.0);
        assert!((measure_samples_per_second(24, Duration::from_secs(3)) - 8.0).abs() < 1e-12);
        assert_eq!(
            measure_samples_per_second(10, Duration::from_secs_f64(f64::EPSILON / 2.0)),
            0.0
        );
    }

    #[test]
    fn run_probe_only_train_writes_success_result_for_real_loose_replay() {
        let (root, replay_path, result_path) = write_real_probe_fixture("train-success");
        let mut config = dummy_config();
        config.data_dir = replay_path;
        config.batch_size = 1;
        config.train_fraction = 1.0;
        config.device = "cpu".to_string();

        run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 1,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect("train probe should succeed on a real loose replay");

        assert!(result_path.exists());
        let raw = fs::read_to_string(&result_path).expect("read written train probe result json");
        let result: ProbeResult =
            serde_json::from_str(&raw).expect("deserialize written train probe result json");
        assert_eq!(result.kind, ProbeKind::Train);
        assert_eq!(result.status, ProbeStatus::Success);
        assert_eq!(result.candidate_microbatch, 1);
        assert!(result.measured_samples_per_second.is_some());
        assert!(result.elapsed_seconds.is_some());
        assert_eq!(result.detail, "stable train probe on real dataset");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_probe_only_validation_writes_success_result_for_real_loose_replay() {
        let (root, replay_path, result_path) = write_real_probe_fixture("validation-success");
        let mut config = dummy_config();
        config.data_dir = replay_path;
        config.batch_size = 1;
        config.train_fraction = 0.0;
        config.device = "cpu".to_string();

        run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 1,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect("validation probe should succeed on a real loose replay");

        assert!(result_path.exists());
        let raw =
            fs::read_to_string(&result_path).expect("read written validation probe result json");
        let result: ProbeResult =
            serde_json::from_str(&raw).expect("deserialize written validation probe result json");
        assert_eq!(result.kind, ProbeKind::Validation);
        assert_eq!(result.status, ProbeStatus::Success);
        assert_eq!(result.candidate_microbatch, 1);
        assert!(result.measured_samples_per_second.is_some());
        assert!(result.elapsed_seconds.is_some());
        assert_eq!(result.detail, "stable validation probe on real dataset");

        let _ = fs::remove_dir_all(root);
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
        assert_eq!(
            classify_probe_detail("unexpected worker panic"),
            ProbeStatus::BackendError
        );
    }

    #[test]
    fn format_probe_attempt_message_uses_probe_kind_label_and_min_attempt_denominator() {
        assert_eq!(
            format_probe_attempt_message(ProbeKind::Validation, 64, 2, 0),
            "[preflight:validation] candidate_mb=64 attempt 2/1"
        );
        assert_eq!(
            format_probe_attempt_message(ProbeKind::RlMicrobatch, 128, 1, 3),
            "[preflight:rl_microbatch] candidate_mb=128 attempt 1/3"
        );
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_returns_none_for_non_growth_cases() {
        let config = dummy_config();

        assert!(maybe_block_host_ram_growth_probe(&config, ProbeKind::Train, 64, None).is_none());
        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::Validation, 64, Some(64))
                .is_none()
        );
        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::Validation, 32, Some(64))
                .is_none()
        );
        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 64, Some(64)).is_none()
        );
    }

    #[test]
    fn run_probe_child_mode_without_child_request_is_a_no_op() {
        let config = dummy_config();

        assert_eq!(run_probe_child_mode(&config, None), Ok(false));
    }

    #[test]
    fn search_rl_runtime_candidate_rejects_validation_kind_as_non_rl() {
        let config = dummy_config();
        let artifacts = RlArtifactPaths::new(&config.output_dir, 0);

        let err = search_rl_runtime_candidate(
            std::path::Path::new("dummy-config.yaml"),
            &config,
            &artifacts,
            ProbeKind::Validation,
            8,
        )
        .expect_err("validation should be rejected by RL runtime search");

        assert_eq!(err, "non-RL probe kind passed to RL runtime search");
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
        let loader = crate::runtime_autotune::autotune_loader_runtime(
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

        let backend = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::RlGames,
            candidate_microbatch: 512,
            status: ProbeStatus::BackendError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: "probe blocked by host-RAM guard".to_string(),
        });
        assert!(backend.contains(
            "[rl_games] candidate_mb=512 outcome=backend_error(host_ram_guard) detail=probe blocked by host-RAM guard"
        ));
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_returns_backend_error_with_host_ram_details() {
        let Some(available) = mem_available_bytes() else {
            return;
        };
        let Some(required_free) = rl_probe_required_free_bytes(&{
            let mut config = dummy_config();
            config.preflight.rl_probe_min_free_memory_bytes = available;
            config.preflight.rl_probe_memory_headroom_ratio = 0.0;
            config
        }) else {
            return;
        };

        let mut config = dummy_config();
        config.preflight.rl_probe_min_free_memory_bytes = available.max(required_free);
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        config.preflight.rl_probe_growth_safety_factor = 1.0;

        let blocked = maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 128, Some(64))
            .expect(
                "growth probe should be blocked when required free memory matches available memory",
            );

        assert_eq!(blocked.kind, ProbeKind::RlGames);
        assert_eq!(blocked.candidate_microbatch, 128);
        assert_eq!(blocked.status, ProbeStatus::BackendError);
        assert!(blocked.measured_samples_per_second.is_none());
        assert!(blocked.elapsed_seconds.is_none());
        assert!(blocked.detail.contains("probe blocked by host-RAM guard"));
        assert!(blocked.detail.contains("available="));
        assert!(blocked.detail.contains("required_free="));
        assert!(blocked.detail.contains("estimated_probe="));
        assert!(blocked.detail.contains("remaining_after_probe="));
        assert!(blocked.detail.contains("baseline_candidate=64"));
        assert!(blocked.detail.contains("growth_safety_factor=1.00"));
    }

    #[test]
    fn search_rl_runtime_candidate_rejects_non_rl_probe_kinds() {
        let config = dummy_config();
        let artifacts = RlArtifactPaths::new(&config.output_dir, 0);

        let err = search_rl_runtime_candidate(
            std::path::Path::new("dummy-config.yaml"),
            &config,
            &artifacts,
            ProbeKind::Train,
            64,
        )
        .expect_err("train probe kind should be rejected for RL runtime search");

        assert_eq!(err, "non-RL probe kind passed to RL runtime search");
    }

    #[test]
    fn run_probe_only_rl_games_fails_fast_without_rl_config() {
        let config = dummy_config();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let result_path = std::env::temp_dir().join(format!(
            "hydra-preflight-runtime-test-rl-games-missing-config-{unique}.json"
        ));

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::RlGames,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("RL games probe should fail before runtime work when RL config is missing");

        assert_eq!(err, "RL probe requested without rl config block");
        assert!(!result_path.exists());
    }

    #[test]
    fn emit_probe_progress_and_step_progress_cover_warmup_and_measure_paths() {
        assert!(emit_probe_progress("plain text that should only flush").is_ok());
        assert!(emit_probe_progress(
            "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=2 measure_steps=3"
        )
        .is_ok());

        let request = ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 64,
            warmup_steps: 2,
            measure_steps: 3,
        };
        assert!(
            emit_probe_step_progress(ProbeKind::Train, 64, 0, request, None, 256).is_ok(),
            "warmup branch should format and flush"
        );
        assert!(
            emit_probe_step_progress(ProbeKind::Train, 64, 2, request, Some(Instant::now()), 256,)
                .is_ok(),
            "measure branch should format and flush"
        );
        assert!(
            emit_probe_step_progress(ProbeKind::Validation, 64, 2, request, None, 64).is_ok(),
            "measure branch should still flush without a start timestamp"
        );
    }

    #[test]
    fn run_probe_child_mode_rejects_unresolved_child_probe_steps() {
        let config = dummy_config();
        let result_path = unique_test_path("probe-child.json");

        let warmup_err = run_probe_child_mode(
            &config,
            Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind: ProbeKind::Train,
                    candidate_microbatch: 32,
                    warmup_steps: None,
                    measure_steps: Some(2),
                },
                result_path: result_path.clone(),
            }),
        )
        .expect_err("missing warmup steps should be rejected before running child mode");
        assert_eq!(
            warmup_err,
            "internal probe child missing resolved warmup steps"
        );

        let measure_err = run_probe_child_mode(
            &config,
            Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind: ProbeKind::Validation,
                    candidate_microbatch: 32,
                    warmup_steps: Some(1),
                    measure_steps: None,
                },
                result_path,
            }),
        )
        .expect_err("missing measure steps should be rejected before running child mode");
        assert_eq!(
            measure_err,
            "internal probe child missing resolved measure steps"
        );
    }

    #[test]
    fn run_probe_child_mode_bubbles_probe_runtime_errors_after_cli_resolution() {
        let mut config = dummy_config();
        config.data_dir = missing_test_path("probe-child-missing-data");
        let result_path = unique_test_path("probe-child-runtime-error.json");

        let err = run_probe_child_mode(
            &config,
            Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind: ProbeKind::Validation,
                    candidate_microbatch: 32,
                    warmup_steps: Some(1),
                    measure_steps: Some(1),
                },
                result_path: result_path.clone(),
            }),
        )
        .expect_err("resolved child requests should bubble probe runtime errors");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_child_mode_routes_rl_requests_into_rl_probe_wrapper_errors() {
        let config = dummy_config();
        let result_path = unique_test_path("probe-child-rl-runtime-error.json");

        let err = run_probe_child_mode(
            &config,
            Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind: ProbeKind::RlGames,
                    candidate_microbatch: 8,
                    warmup_steps: Some(1),
                    measure_steps: Some(1),
                },
                result_path: result_path.clone(),
            }),
        )
        .expect_err("resolved RL child requests should route into the RL probe wrapper");

        assert_eq!(err, "RL probe requested without rl config block");
        assert!(!result_path.exists());
    }

    #[test]
    fn execute_probe_request_rejects_unsupported_config_extension_before_spawning() {
        let config_path = write_temp_file("unsupported-config", "txt", "not yaml");
        let result_path = unique_test_path("probe-result.json");

        let err = execute_probe_request(
            &config_path,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("unsupported config extension should fail before spawning child process");

        assert_eq!(
            err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
        assert!(!result_path.exists());
    }

    #[test]
    fn run_rl_probe_only_rejects_non_rl_probe_kinds() {
        let config = dummy_config();
        let result_path = unique_test_path("non-rl-probe-result.json");

        let err = run_rl_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 16,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("non-RL kinds should be rejected by the RL-only handler");

        assert_eq!(err, "RL probe requested without rl config block");
        assert!(!result_path.exists());

        let mut config = dummy_config();
        config.rl = Some(dummy_rl_train_config());
        let err = run_rl_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 16,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("non-RL kinds should be rejected even when rl config exists");

        assert_eq!(err, "non-RL probe routed to RL probe handler");
    }

    #[test]
    fn run_rl_probe_only_rl_games_bubbles_invalid_device_before_runtime_work() {
        let mut config = dummy_config();
        config.device = "definitely-not-a-device".to_string();
        config.rl = Some(dummy_rl_train_config());
        let result_path = unique_test_path("rl-games-invalid-device.json");

        let err = run_rl_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::RlGames,
                candidate_microbatch: 8,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("invalid RL device should fail before self-play runtime work");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_rl_probe_only_rl_microbatch_bubbles_invalid_device_before_runtime_work() {
        let mut config = dummy_config();
        config.device = "definitely-not-a-device".to_string();
        config.rl = Some(dummy_rl_train_config());
        let result_path = unique_test_path("rl-micro-invalid-device.json");

        let err = run_rl_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::RlMicrobatch,
                candidate_microbatch: 16,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("invalid RL device should fail before RL microbatch runtime work");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_only_rl_microbatch_fails_fast_without_rl_config() {
        let config = dummy_config();
        let result_path = unique_test_path("rl-microbatch-result.json");

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::RlMicrobatch,
                candidate_microbatch: 24,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err(
            "RL microbatch probe should fail before runtime work when RL config is missing",
        );

        assert_eq!(err, "RL probe requested without rl config block");
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_only_rejects_invalid_thread_configuration_before_any_probe_work() {
        let mut config = dummy_config();
        config.num_threads = Some(0);
        let result_path = unique_test_path("invalid-thread-probe-result.json");

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("invalid rayon thread config should fail before scan or device setup");

        assert!(err.starts_with("failed to configure rayon threads for probe child: "));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_child_mode_bubbles_invalid_thread_configuration_before_probe_execution() {
        let mut config = dummy_config();
        config.num_threads = Some(0);
        let result_path = unique_test_path("invalid-thread-child-result.json");

        let err = run_probe_child_mode(
            &config,
            Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind: ProbeKind::RlMicrobatch,
                    candidate_microbatch: 16,
                    warmup_steps: Some(1),
                    measure_steps: Some(1),
                },
                result_path: result_path.clone(),
            }),
        )
        .expect_err("invalid rayon thread config should bubble before child probe execution");

        assert!(err.starts_with("failed to configure rayon threads for probe child: "));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_only_train_fails_fast_when_dataset_scan_cannot_start() {
        let mut config = dummy_config();
        config.data_dir = missing_test_path("missing-train-data");
        let result_path = unique_test_path("train-probe-result.json");

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("missing dataset path should fail before any heavy train probing");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
        assert!(!result_path.exists());
    }

    #[test]
    fn run_probe_only_train_bubbles_invalid_device_after_successful_scan() {
        let root = unique_test_path("train-invalid-device-scan");
        fs::create_dir_all(&root).expect("create empty data dir");
        let mut config = dummy_config();
        config.data_dir = root.clone();
        config.device = "definitely-not-a-device".to_string();
        let result_path = unique_test_path("train-invalid-device-result.json");

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("invalid device should fail after scan but before train probing");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        assert!(!result_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_probe_only_validation_bubbles_invalid_device_after_successful_scan() {
        let root = unique_test_path("validation-invalid-device-scan");
        fs::create_dir_all(&root).expect("create empty data dir");
        let mut config = dummy_config();
        config.data_dir = root.clone();
        config.device = "definitely-not-a-device".to_string();
        let result_path = unique_test_path("validation-invalid-device-result.json");

        let err = run_probe_only(
            &config,
            ProbeRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("invalid device should fail after scan but before validation probing");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        assert!(!result_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_probe_ladder_only_fails_before_probe_attempts_when_data_scan_fails() {
        let mut config = dummy_config();
        config.data_dir = missing_test_path("missing-ladder-data");
        let artifacts = BcArtifactPaths::new(&unique_test_path("ladder-artifacts"), 0);

        let err = run_probe_ladder_only(
            Path::new("ignored-config.yaml"),
            &config,
            &artifacts,
            ProbeRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("missing dataset path should stop probe ladder before child probes");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn search_train_and_validation_microbatch_fail_fast_on_invalid_probe_config_path() {
        let config_path = write_temp_file("invalid-search-config", "txt", "not yaml");
        let artifacts = BcArtifactPaths::new(&unique_test_path("search-bc-artifacts"), 0);
        let config = dummy_config();

        let train_err = search_train_microbatch(&config_path, &config, &artifacts, 64)
            .expect_err("invalid config path should stop train search before launching probes");
        assert_eq!(
            train_err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );

        let validation_err = search_validation_microbatch(&config_path, &config, &artifacts, 32)
            .expect_err(
                "invalid config path should stop validation search before launching probes",
            );
        assert_eq!(
            validation_err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
    }

    #[test]
    fn search_rl_runtime_candidate_fails_fast_on_invalid_probe_config_path() {
        let config_path = write_temp_file("invalid-rl-search-config", "txt", "not yaml");
        let mut config = dummy_config();
        config.rl = Some(dummy_rl_train_config());
        config.preflight.allow_override_explicit_microbatch = false;
        let artifacts = RlArtifactPaths::new(&unique_test_path("search-rl-artifacts"), 0);

        let err = search_rl_runtime_candidate(
            &config_path,
            &config,
            &artifacts,
            ProbeKind::RlMicrobatch,
            16,
        )
        .expect_err("invalid config path should stop RL candidate search before launching probes");

        assert_eq!(
            err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
    }

    #[test]
    fn run_preflight_stops_at_train_probe_when_probe_config_path_is_invalid() {
        let config_path = write_temp_file("invalid-preflight-config", "txt", "not yaml");
        let config = dummy_config();
        let artifacts = BcArtifactPaths::new(&unique_test_path("preflight-artifacts"), 0);

        let err = match run_preflight(
            &config_path,
            &config,
            &HydraModelConfig::learner(),
            "cpu",
            &artifacts,
        ) {
            Err(err) => err,
            Ok(_) => panic!("invalid config path should stop preflight before runtime autotuning"),
        };

        assert_eq!(
            err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
    }

    #[test]
    fn run_rl_preflight_handles_missing_rl_config_and_invalid_probe_config_path() {
        let train_device = LibTorchDevice::Cpu;
        let config_path = write_temp_file("invalid-rl-preflight-config", "txt", "not yaml");

        let missing_rl_err = match run_rl_preflight(&config_path, &dummy_config(), &train_device) {
            Err(err) => err,
            Ok(_) => {
                panic!("RL preflight should reject missing rl config before any filesystem work")
            }
        };
        assert_eq!(
            missing_rl_err,
            "RL preflight requested without rl config block"
        );

        let mut config = dummy_config();
        config.output_dir = unique_test_path("rl-preflight-output");
        config.rl = Some(dummy_rl_train_config());
        let err = match run_rl_preflight(&config_path, &config, &train_device) {
            Err(err) => err,
            Ok(_) => {
                panic!("invalid config path should stop RL preflight before heavy runtime probes")
            }
        };
        assert_eq!(
            err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
    }

    #[test]
    fn search_rl_runtime_candidate_explicit_microbatch_failure_uses_explicit_error_path() {
        let data_dir = unique_test_path("rl-explicit-microbatch-data");
        fs::create_dir_all(&data_dir).expect("create empty RL data dir");
        let output_dir = unique_test_path("rl-explicit-microbatch-out");
        let artifacts = RlArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.rl = Some(RlTrainConfig {
            games_per_batch: 8,
            microbatch_size: Some(24),
            ..RlTrainConfig::default()
        });
        let config_path = unique_test_path("rl-explicit-microbatch-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid RL config");
        fs::write(&config_path, config_yaml).expect("write valid RL config yaml");

        let err = search_rl_runtime_candidate(
            &config_path,
            &config,
            &artifacts,
            ProbeKind::RlMicrobatch,
            16,
        )
        .expect_err("explicit RL microbatch failure should use explicit-only error path");

        assert_eq!(err, "explicit rl_microbatch candidate 24 failed preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_train_microbatch_explicit_failure_uses_explicit_error_path() {
        let data_dir = unique_test_path("train-explicit-microbatch-data");
        fs::create_dir_all(&data_dir).expect("create empty train data dir");
        let output_dir = unique_test_path("train-explicit-microbatch-out");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.microbatch_size = Some(96);
        let config_path =
            unique_test_path("train-explicit-microbatch-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid train config");
        fs::write(&config_path, config_yaml).expect("write valid train config yaml");

        let err = search_train_microbatch(&config_path, &config, &artifacts, 64)
            .expect_err("explicit train microbatch failure should use explicit-only error path");

        assert_eq!(err, "explicit train microbatch 96 failed preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_train_microbatch_non_explicit_failure_reports_no_stable_result() {
        let data_dir = unique_test_path("train-no-stable-data");
        fs::create_dir_all(&data_dir).expect("create empty train data dir");
        let output_dir = unique_test_path("train-no-stable-out");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("train-no-stable-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid train config");
        fs::write(&config_path, config_yaml).expect("write valid train config yaml");

        let err = search_train_microbatch(&config_path, &config, &artifacts, 64)
            .expect_err("all-failing train search should report no stable result");

        assert_eq!(err, "no stable train microbatch found in preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_validation_microbatch_explicit_failure_uses_explicit_error_path() {
        let data_dir = unique_test_path("validation-explicit-microbatch-data");
        fs::create_dir_all(&data_dir).expect("create empty validation data dir");
        let output_dir = unique_test_path("validation-explicit-microbatch-out");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.validation_microbatch_size = Some(48);
        let config_path =
            unique_test_path("validation-explicit-microbatch-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid validation config");
        fs::write(&config_path, config_yaml).expect("write valid validation config yaml");

        let err = search_validation_microbatch(&config_path, &config, &artifacts, 32).expect_err(
            "explicit validation microbatch failure should use explicit-only error path",
        );

        assert_eq!(err, "explicit validation microbatch 48 failed preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_validation_microbatch_non_explicit_failure_reports_no_stable_result() {
        let data_dir = unique_test_path("validation-no-stable-data");
        fs::create_dir_all(&data_dir).expect("create empty validation data dir");
        let output_dir = unique_test_path("validation-no-stable-out");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("validation-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid validation config");
        fs::write(&config_path, config_yaml).expect("write valid validation config yaml");

        let err = search_validation_microbatch(&config_path, &config, &artifacts, 32)
            .expect_err("non-explicit validation failure should report no stable result");

        assert_eq!(err, "no stable validation microbatch found in preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_rl_games_non_explicit_failure_reports_no_stable_result() {
        let data_dir = unique_test_path("rl-games-no-stable-data");
        fs::create_dir_all(&data_dir).expect("create empty RL data dir");
        let output_dir = unique_test_path("rl-games-no-stable-out");
        let artifacts = RlArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.rl = Some(dummy_rl_train_config());
        let config_path = unique_test_path("rl-games-no-stable-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid RL games config");
        fs::write(&config_path, config_yaml).expect("write valid RL games config yaml");

        let err =
            search_rl_runtime_candidate(&config_path, &config, &artifacts, ProbeKind::RlGames, 16)
                .expect_err("all-failing RL games search should report no stable result");

        assert_eq!(err, "no stable rl_games candidate found in preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn search_rl_microbatch_non_explicit_failure_reports_no_stable_result() {
        let data_dir = unique_test_path("rl-micro-no-stable-data");
        fs::create_dir_all(&data_dir).expect("create empty RL data dir");
        let output_dir = unique_test_path("rl-micro-no-stable-out");
        let artifacts = RlArtifactPaths::new(&output_dir, 0);
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir;
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        config.rl = Some(dummy_rl_train_config());
        let config_path = unique_test_path("rl-micro-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid RL microbatch config");
        fs::write(&config_path, config_yaml).expect("write valid RL microbatch config yaml");

        let err = search_rl_runtime_candidate(
            &config_path,
            &config,
            &artifacts,
            ProbeKind::RlMicrobatch,
            16,
        )
        .expect_err("all-failing RL microbatch search should report no stable result");

        assert_eq!(err, "no stable rl_microbatch candidate found in preflight");
        let _ = fs::remove_dir_all(data_dir);
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn format_probe_result_summary_reports_data_error_and_plain_backend_error() {
        let data = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Validation,
            candidate_microbatch: 48,
            status: ProbeStatus::DataError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: "replay parse mismatch".to_string(),
        });
        assert!(data.contains(
            "[validation] candidate_mb=48 outcome=data_error detail=replay parse mismatch"
        ));

        let backend = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 96,
            status: ProbeStatus::BackendError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: "unexpected worker panic".to_string(),
        });
        assert!(backend.contains("[train] candidate_mb=96 outcome=backend_error("));
        assert!(backend.contains("detail=unexpected worker panic"));
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_uses_baseline_guard_for_rl_microbatch_too() {
        let Some(available) = mem_available_bytes() else {
            return;
        };
        let mut config = dummy_config();
        config.preflight.rl_probe_min_free_memory_bytes = available;
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        config.preflight.rl_probe_growth_safety_factor = 1.0;

        let blocked =
            maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 64, Some(32))
                .expect(
                "growth probe should be blocked when required free memory matches available memory",
            );

        assert_eq!(blocked.kind, ProbeKind::RlMicrobatch);
        assert_eq!(blocked.candidate_microbatch, 64);
        assert_eq!(blocked.status, ProbeStatus::BackendError);
        assert!(blocked.detail.contains("baseline_candidate=32"));
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_clamps_subunit_safety_factor_to_one() {
        let Some(available) = mem_available_bytes() else {
            return;
        };
        let mut config = dummy_config();
        config.preflight.rl_probe_min_free_memory_bytes = available;
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        config.preflight.rl_probe_growth_safety_factor = 0.25;

        let blocked = maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 128, Some(64))
            .expect("sub-unit safety factors should still clamp to the host-RAM guard path");

        assert_eq!(blocked.kind, ProbeKind::RlGames);
        assert_eq!(blocked.status, ProbeStatus::BackendError);
        assert!(blocked.detail.contains("growth_safety_factor=1.00"));
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_allows_growth_when_required_free_is_zero() {
        let Some(_available) = mem_available_bytes() else {
            return;
        };
        let mut config = dummy_config();
        config.preflight.rl_probe_min_free_memory_bytes = 0;
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        config.preflight.rl_probe_growth_safety_factor = 1.0;

        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 33, Some(32))
                .is_none()
        );
    }

    #[test]
    fn execute_probe_request_rejects_missing_config_before_spawning() {
        let config_path = missing_test_path("missing-probe-config.yaml").with_extension("yaml");
        let result_path = unique_test_path("missing-probe-result.json");

        let err = execute_probe_request(
            &config_path,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &result_path,
        )
        .expect_err("missing config path should fail before spawning child process");

        assert!(err.contains(config_path.to_string_lossy().as_ref()));
        assert!(!result_path.exists());
    }

    #[test]
    fn format_probe_result_summary_handles_success_without_elapsed_samples() {
        let summary = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Validation,
            candidate_microbatch: 24,
            status: ProbeStatus::Success,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: String::new(),
        });
        assert!(summary.contains("[validation] candidate_mb=24 outcome=success"));
        assert!(summary.contains("0.00 samples/s"));
        assert!(summary.contains("elapsed=0.00s"));
    }

    #[test]
    fn classify_probe_detail_treats_cudnn_and_oom_strings_as_expected() {
        assert_eq!(
            classify_probe_detail("cuDNN kernel launch failed"),
            ProbeStatus::BackendError
        );
        assert_eq!(
            classify_probe_detail("OOM killer terminated child process"),
            ProbeStatus::Oom
        );
    }

    #[test]
    fn run_rl_preflight_fails_fast_on_invalid_microbatch_config_path() {
        let train_device = LibTorchDevice::Cpu;
        let config_path = write_temp_file("invalid-rl-micro-config", "txt", "not yaml");
        let mut config = dummy_config();
        config.output_dir = unique_test_path("rl-preflight-fastfail");
        config.rl = Some(dummy_rl_train_config());

        let err = match run_rl_preflight(&config_path, &config, &train_device) {
            Err(err) => err,
            Ok(_) => {
                panic!("invalid config path should stop RL preflight before runtime probing")
            }
        };

        assert_eq!(
            err,
            format!(
                "unsupported config extension for {}; use .yaml",
                config_path.display()
            )
        );
    }

    #[test]
    fn run_probe_ladder_only_accepts_rl_request_wrapper_and_fails_on_missing_data_first() {
        let mut config = dummy_config();
        config.data_dir = missing_test_path("missing-rl-ladder-data");
        config.rl = Some(dummy_rl_train_config());
        let artifacts = BcArtifactPaths::new(&unique_test_path("rl-ladder-artifacts"), 0);

        let err = run_probe_ladder_only(
            Path::new("ignored-config.yaml"),
            &config,
            &artifacts,
            ProbeRequest {
                kind: ProbeKind::RlMicrobatch,
                candidate_microbatch: 16,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err(
            "missing dataset path should stop RL-flavored probe ladder before child probes",
        );

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn classify_probe_detail_prefers_oom_and_data_keywords_over_backend_defaults() {
        assert_eq!(
            classify_probe_detail("OOM from replay loader while CUDA kernel was active"),
            ProbeStatus::Oom
        );
        assert_eq!(
            classify_probe_detail("collate data replay failure in worker thread"),
            ProbeStatus::DataError
        );
    }

    #[test]
    fn classify_probe_detail_prefers_backend_keywords_over_data_without_oom() {
        assert_eq!(
            classify_probe_detail("cuda replay mismatch in collate worker"),
            ProbeStatus::BackendError
        );
        assert_eq!(
            classify_probe_detail("libtorch data loader replay error"),
            ProbeStatus::BackendError
        );
    }

    #[test]
    fn format_probe_result_summary_reports_plain_success_detail_for_rl_microbatch() {
        let summary = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::RlMicrobatch,
            candidate_microbatch: 12,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(42.25),
            elapsed_seconds: Some(0.5),
            detail: "stable rl_microbatch probe on real dataset".to_string(),
        });

        assert!(summary.contains("[rl_microbatch] candidate_mb=12 outcome=success"));
        assert!(summary.contains("42.25 samples/s"));
        assert!(summary.contains("elapsed=0.50s"));
    }

    #[test]
    fn format_probe_attempt_message_uses_exact_denominator_when_positive() {
        assert_eq!(
            format_probe_attempt_message(ProbeKind::Train, 32, 3, 4),
            "[preflight:train] candidate_mb=32 attempt 3/4"
        );
    }

    #[test]
    fn measure_samples_per_second_handles_fractional_elapsed_time() {
        assert!((measure_samples_per_second(9, Duration::from_millis(450)) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn format_probe_result_summary_keeps_empty_rl_backend_detail_field_stable() {
        let summary = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::RlGames,
            candidate_microbatch: 40,
            status: ProbeStatus::BackendError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: String::new(),
        });

        assert!(summary.contains("[rl_games] candidate_mb=40 outcome=backend_error(generic)"));
        assert!(summary.contains("detail="));
        assert!(!summary.contains("detail=n/a"));
    }

    #[test]
    fn classify_probe_detail_treats_plain_data_keywords_as_data_errors() {
        assert_eq!(
            classify_probe_detail("data pipeline mismatch in worker"),
            ProbeStatus::DataError
        );
        assert_eq!(
            classify_probe_detail("collate failure without backend keywords"),
            ProbeStatus::DataError
        );
    }

    #[test]
    fn format_probe_attempt_message_clamps_zero_total_attempts_for_rl_games() {
        assert_eq!(
            format_probe_attempt_message(ProbeKind::RlGames, 12, 1, 0),
            "[preflight:rl_games] candidate_mb=12 attempt 1/1"
        );
    }

    #[test]
    fn maybe_block_host_ram_growth_probe_returns_none_when_candidate_does_not_grow() {
        let config = dummy_config();

        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 31, Some(32))
                .is_none()
        );
        assert!(
            maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 32, Some(32)).is_none()
        );
    }

    #[test]
    fn classify_probe_detail_prefers_oom_over_backend_and_data_keywords() {
        assert_eq!(
            classify_probe_detail("cuda oom while replay data collate failed"),
            ProbeStatus::Oom
        );
    }

    #[test]
    fn format_probe_result_summary_keeps_empty_backend_detail_field_stable() {
        let summary = format_probe_result_summary(&ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 40,
            status: ProbeStatus::BackendError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: String::new(),
        });

        assert!(summary.contains("[train] candidate_mb=40 outcome=backend_error(generic)"));
        assert!(summary.ends_with("detail="));
    }
}
