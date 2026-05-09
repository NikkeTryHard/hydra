use std::collections::BTreeSet;
use std::path::Path;

use colored::Colorize;
use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use super::config::TrainConfig;
use super::preflight_runtime::{
    execute_probe_request, format_probe_attempt_message, format_probe_result_summary,
};
use super::presentation::{format_status_line, make_bar};
use super::probe_ladder::{dynamic_probe_ceiling, top_k_refinement_candidates};
use super::probe_process::execute_probe_request_batch;
use super::probe_request::{ProbeBatchRequest, ProbeRequest};
use super::probe_summary::{
    ProbeCandidateSummary, best_probe_summary, probe_kind_name, summarize_probe_results,
};
use super::probe_transport::{probe_batch_results_path, probe_result_path};

pub(super) struct ProbeRunSpec {
    pub(super) kind: ProbeKind,
    pub(super) candidate: usize,
    pub(super) attempts: usize,
    pub(super) warmup_steps: usize,
    pub(super) measure_steps: usize,
}

#[derive(Default)]
pub(super) struct ProbeGrowthState {
    pub(super) patience: usize,
    pub(super) steps: usize,
    pub(super) prior_best_score: Option<f64>,
}

pub(super) struct ProbeGrowthDecision<'a> {
    pub(super) index: usize,
    pub(super) kind: ProbeKind,
    pub(super) candidate: usize,
    pub(super) summary: &'a ProbeCandidateSummary,
    pub(super) candidate_score: f64,
    pub(super) tolerance: f64,
}

pub(super) fn adaptive_probe_steps(config: &TrainConfig, seconds_per_step: f64) -> (usize, usize) {
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

pub(super) fn should_continue_validation_growth(
    best: f64,
    challenger: f64,
    tolerance_ratio: f64,
) -> bool {
    challenger >= best * (1.0 - tolerance_ratio.max(0.0))
}

pub(super) fn maybe_expand_probe_candidates(
    candidates: &mut Vec<usize>,
    decision: ProbeGrowthDecision<'_>,
    config: &TrainConfig,
    growth_state: &mut ProbeGrowthState,
) -> bool {
    let ProbeGrowthDecision {
        index,
        kind,
        candidate,
        summary,
        candidate_score,
        tolerance,
    } = decision;
    let is_top = index + 1 == candidates.len();
    if is_top && summary.candidate_microbatch == candidate {
        let ceiling = dynamic_probe_ceiling(config, kind, candidate);
        let next_candidate = candidate.saturating_mul(2);
        if next_candidate > candidate && next_candidate <= ceiling {
            if growth_state.steps >= config.preflight.validation_growth_max_steps.max(1) {
                return true;
            }
            let reference_score = growth_state.prior_best_score.unwrap_or_else(|| {
                summary
                    .average_samples_per_second
                    .unwrap_or(candidate_score)
            });
            if should_continue_validation_growth(reference_score, candidate_score, tolerance) {
                growth_state.patience = 0;
                growth_state.steps += 1;
                candidates.push(next_candidate);
                growth_state.prior_best_score = Some(reference_score.max(candidate_score));
            } else {
                growth_state.patience += 1;
                growth_state.prior_best_score = Some(reference_score.max(candidate_score));
                if growth_state.patience >= config.preflight.validation_growth_patience.max(1) {
                    return true;
                }
            }
        }
    }

    growth_state.prior_best_score = Some(
        growth_state.prior_best_score.unwrap_or(0.0).max(
            summary
                .average_samples_per_second
                .unwrap_or(candidate_score),
        ),
    );
    false
}

pub(super) fn run_candidate_attempts<F>(
    config_path: &Path,
    result_path_for: &mut F,
    spec: ProbeRunSpec,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<bool, String>
where
    F: FnMut(ProbeKind, usize, usize) -> std::path::PathBuf,
{
    run_candidate_attempts_with_batch_executor(
        config_path,
        result_path_for,
        spec,
        results,
        progress,
        |config_path, batch, results_path| {
            execute_probe_request_batch(
                config_path,
                batch,
                results_path,
                super::preflight_runtime::classify_probe_detail,
            )
        },
    )
}

fn run_candidate_attempts_with_batch_executor<F, E>(
    config_path: &Path,
    result_path_for: &mut F,
    spec: ProbeRunSpec,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
    mut execute_batch: E,
) -> Result<bool, String>
where
    F: FnMut(ProbeKind, usize, usize) -> std::path::PathBuf,
    E: FnMut(&Path, ProbeBatchRequest, &Path) -> Result<Vec<ProbeResult>, String>,
{
    let ProbeRunSpec {
        kind,
        candidate,
        attempts,
        warmup_steps,
        measure_steps,
    } = spec;
    let attempts = attempts.max(1);
    let request = ProbeRequest {
        kind,
        candidate_microbatch: candidate,
        warmup_steps,
        measure_steps,
    };
    progress.set_message(format_probe_attempt_message(kind, candidate, 1, attempts));
    let batch_results_path = probe_batch_results_path(&result_path_for(kind, candidate, 0));
    println!(
        "{}",
        format_status_line(
            &format!("[preflight:{}]", probe_kind_name(kind)),
            format!(
                "candidate_mb={} attempt=1/{} phase=probe",
                candidate, attempts,
            )
        )
    );
    let batch_results = execute_batch(
        config_path,
        ProbeBatchRequest { request, attempts },
        &batch_results_path,
    )?;
    Ok(replay_candidate_attempt_results(
        kind,
        candidate,
        attempts,
        batch_results,
        results,
        progress,
        1,
    ))
}

fn replay_candidate_attempt_results(
    kind: ProbeKind,
    candidate: usize,
    attempts: usize,
    batch_results: Vec<ProbeResult>,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
    prelogged_attempts: usize,
) -> bool {
    for (attempt, result) in batch_results.into_iter().enumerate() {
        let attempt_number = attempt + 1;
        progress.set_message(format_probe_attempt_message(
            kind,
            candidate,
            attempt_number,
            attempts,
        ));
        if attempt >= prelogged_attempts {
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
        }
        let passed = result.status == ProbeStatus::Success;
        progress.inc(1);
        println!("{}", format_probe_result_summary(&result));
        results.push(result);
        if !passed {
            return false;
        }
    }
    true
}

pub(super) fn rerun_candidate_attempts<F>(
    config_path: &Path,
    result_path_for: &mut F,
    spec: ProbeRunSpec,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String>
where
    F: FnMut(ProbeKind, usize, usize) -> std::path::PathBuf,
{
    run_candidate_attempts(config_path, result_path_for, spec, results, progress)?;
    Ok(())
}

pub(super) fn rerun_probe_finalists<F>(
    config_path: &Path,
    mut result_path_for: F,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String>
where
    F: FnMut(ProbeKind, usize, usize) -> std::path::PathBuf,
{
    let finalists = super::probe_ladder::close_probe_finalists(
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
        super::presentation::format_preflight_summary_line(
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
            &mut result_path_for,
            ProbeRunSpec {
                kind,
                candidate: summary.candidate_microbatch,
                attempts: extra_attempts,
                warmup_steps,
                measure_steps: measure_steps + extra_measure_steps,
            },
            results,
            progress,
        )?;
    }
    Ok(())
}

pub(super) fn refine_probe_winner_locally<F>(
    config_path: &Path,
    mut result_path_for: F,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String>
where
    F: FnMut(ProbeKind, usize, usize) -> std::path::PathBuf,
{
    if !config.preflight.local_refinement_enabled {
        return Ok(());
    }
    let summaries = summarize_probe_results(results);
    let ceiling = dynamic_probe_ceiling(
        config,
        kind,
        super::probe_summary::best_probe_summary(results)
            .map(|summary| summary.candidate_microbatch)
            .unwrap_or(config.batch_size),
    );
    let candidates = super::probe_ladder::local_refinement_candidates(
        &summaries,
        config.preflight.local_refinement_min_gap,
        config.preflight.local_refinement_max_candidates,
        ceiling,
    );
    if candidates.is_empty() {
        return Ok(());
    }
    let successful_elapsed = summaries
        .iter()
        .filter_map(|summary| {
            (summary.status == ProbeStatus::Success)
                .then_some(
                    summary
                        .average_elapsed_seconds
                        .map(|elapsed| (summary.candidate_microbatch, elapsed)),
                )
                .flatten()
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        super::presentation::format_preflight_summary_line(
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
        let seconds_per_step = successful_elapsed
            .iter()
            .min_by_key(|(summary_candidate, _)| summary_candidate.abs_diff(candidate))
            .map(|(_, elapsed)| *elapsed)
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
            &mut result_path_for,
            ProbeRunSpec {
                kind,
                candidate,
                attempts: 1,
                warmup_steps,
                measure_steps: measure_steps
                    + config.preflight.local_refinement_extra_measure_steps.max(1),
            },
            results,
            progress,
        )?;
    }
    Ok(())
}

pub(super) fn refine_top_k_probe_candidates_locally<F>(
    config_path: &Path,
    probe_result_path: F,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
) -> Result<(), String>
where
    F: Fn(ProbeKind, usize, usize) -> std::path::PathBuf + Copy,
{
    let summaries = summarize_probe_results(results);
    let ceiling = dynamic_probe_ceiling(
        config,
        kind,
        summaries
            .iter()
            .map(|summary| summary.candidate_microbatch)
            .max()
            .unwrap_or(1),
    );
    let candidates = top_k_refinement_candidates(
        &summaries,
        config.preflight.finalist_margin_ratio,
        config.preflight.search_top_k,
        config.preflight.local_refinement_min_gap,
        config.preflight.local_refinement_max_candidates,
        ceiling,
    );
    if candidates.is_empty() {
        return Ok(());
    }
    let successful_candidates = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .map(|summary| summary.candidate_microbatch)
        .collect::<BTreeSet<_>>();
    for _round in 0..config.preflight.search_coordinate_rounds.max(1) {
        for candidate in &candidates {
            if successful_candidates.contains(candidate) {
                continue;
            }
            let request = ProbeRequest {
                kind,
                candidate_microbatch: *candidate,
                warmup_steps: config.preflight.warmup_steps,
                measure_steps: config.preflight.measure_steps.max(1)
                    + config.preflight.local_refinement_extra_measure_steps.max(1),
            };
            progress.set_message(super::presentation::format_probe_progress_line(&format!(
                "probe_progress kind={} candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
                probe_kind_name(kind),
                candidate,
                request.warmup_steps,
                request.measure_steps,
            ))
            .unwrap_or_else(|| format!("refining {} candidate {}", probe_kind_name(kind), candidate)));
            let result = execute_probe_request(
                config_path,
                request,
                &probe_result_path(kind, *candidate, 0),
            )?;
            progress.inc(1);
            println!("{}", super::presentation::format_probe_status_line(&result));
            results.push(result);
        }
    }
    Ok(())
}

pub(super) fn finalize_probe_search<F>(
    config_path: &Path,
    result_path_for: F,
    kind: ProbeKind,
    config: &TrainConfig,
    results: &mut Vec<ProbeResult>,
    progress: &indicatif::ProgressBar,
    missing_error: String,
) -> Result<ProbeCandidateSummary, String>
where
    F: Fn(ProbeKind, usize, usize) -> std::path::PathBuf + Copy,
{
    if config.bc_shards_manifest_path.is_some() {
        return best_probe_summary(results).ok_or(missing_error);
    }
    refine_probe_winner_locally(
        config_path,
        result_path_for,
        kind,
        config,
        results,
        progress,
    )?;
    refine_top_k_probe_candidates_locally(
        config_path,
        result_path_for,
        kind,
        config,
        results,
        progress,
    )?;
    rerun_probe_finalists(
        config_path,
        result_path_for,
        kind,
        config,
        results,
        progress,
    )?;
    best_probe_summary(results).ok_or(missing_error)
}

pub(super) fn probe_candidate_ladder(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &super::artifacts::BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    let explicit_candidate = match kind {
        ProbeKind::Train => config.microbatch_size,
        ProbeKind::Validation => config.validation_microbatch_size,
        ProbeKind::RlGames => config.rl.as_ref().map(|rl| rl.games_per_batch),
        ProbeKind::RlMicrobatch => config.rl.as_ref().and_then(|rl| rl.microbatch_size),
    };
    let use_explicit_only =
        explicit_candidate.is_some() && !config.preflight.allow_override_explicit_microbatch;
    let candidate_list: Vec<usize> = if use_explicit_only {
        vec![explicit_candidate.unwrap_or(1)]
    } else {
        candidates.to_vec()
    };
    let mut results = Vec::new();
    println!(
        "{}",
        super::presentation::format_preflight_summary_line(
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
        let attempts = config.preflight.required_successes.max(1);
        let result_path_for =
            |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt);
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
            let result_path = result_path_for(kind, candidate, attempt);
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
        if stable && use_explicit_only {
            return Ok((candidate, results));
        }
    }
    progress.finish_with_message(
        format!("preflight {} ladder complete", probe_kind_name(kind))
            .green()
            .to_string(),
    );

    refine_probe_winner_locally(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        kind,
        config,
        &mut results,
        &progress,
    )?;
    rerun_probe_finalists(
        config_path,
        |kind, candidate, attempt| probe_result_path(artifacts, kind, candidate, attempt),
        kind,
        config,
        &mut results,
        &progress,
    )?;

    if use_explicit_only {
        return Err(format!(
            "explicit {} microbatch {} failed preflight",
            probe_kind_name(kind),
            explicit_candidate.unwrap_or(1)
        ));
    }

    let selected_summary = best_probe_summary(&results).ok_or_else(|| {
        format!(
            "no stable {} microbatch found in preflight",
            probe_kind_name(kind)
        )
    })?;
    if selected_summary.attempts > 0 {
        println!(
            "{}",
            super::presentation::format_preflight_selection_line(
                super::probe_summary::format_probe_selection_summary(kind, &selected_summary)
            )
        );
    }
    Ok((selected_summary.candidate_microbatch, results))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hydra_train::preflight::{PreflightConfig, ProbeKind, ProbeStatus};

    use super::*;
    use crate::config::TrainConfig;
    use crate::probe_transport::{ProbeBatchArtifact, probe_batch_results_path};

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
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            validation_gates: crate::config::ValidationGateConfig::default(),
            rl: None,
            bc: Default::default(),
            nsight_trace: None,
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
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    fn summary(
        candidate_microbatch: usize,
        average_samples_per_second: Option<f64>,
    ) -> ProbeCandidateSummary {
        ProbeCandidateSummary {
            candidate_microbatch,
            status: ProbeStatus::Success,
            attempts: 1,
            average_samples_per_second,
            average_elapsed_seconds: Some(1.0),
        }
    }

    fn probe_result(
        kind: ProbeKind,
        candidate_microbatch: usize,
        status: ProbeStatus,
        measured_samples_per_second: Option<f64>,
        elapsed_seconds: Option<f64>,
    ) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status,
            measured_samples_per_second,
            elapsed_seconds,
            detail: String::new(),
        }
    }

    fn probe_result_with_detail(
        kind: ProbeKind,
        candidate_microbatch: usize,
        status: ProbeStatus,
        detail: &str,
    ) -> ProbeResult {
        ProbeResult {
            detail: detail.to_string(),
            ..probe_result(kind, candidate_microbatch, status, Some(123.0), Some(1.0))
        }
    }

    fn hidden_progress() -> indicatif::ProgressBar {
        indicatif::ProgressBar::hidden()
    }

    #[test]
    fn adaptive_probe_steps_clamp_to_minimums_for_slow_steps() {
        let mut config = dummy_config();
        config.preflight.warmup_steps = 2;
        config.preflight.measure_steps = 3;
        config.preflight.target_warmup_seconds = 1.0;
        config.preflight.target_measure_seconds = 2.0;
        config.preflight.max_adaptive_warmup_steps = 8;
        config.preflight.max_adaptive_measure_steps = 9;

        let (warmup_steps, measure_steps) = adaptive_probe_steps(&config, 10.0);

        assert_eq!(warmup_steps, 2);
        assert_eq!(measure_steps, 3);
    }

    #[test]
    fn adaptive_probe_steps_clamp_to_maximums_for_tiny_or_zero_step_times() {
        let mut config = dummy_config();
        config.preflight.warmup_steps = 2;
        config.preflight.measure_steps = 3;
        config.preflight.target_warmup_seconds = 6.0;
        config.preflight.target_measure_seconds = 12.0;
        config.preflight.max_adaptive_warmup_steps = 4;
        config.preflight.max_adaptive_measure_steps = 5;

        let (warmup_steps, measure_steps) = adaptive_probe_steps(&config, 0.0);

        assert_eq!(warmup_steps, 4);
        assert_eq!(measure_steps, 5);
    }

    #[test]
    fn should_continue_validation_growth_treats_negative_tolerance_as_zero() {
        assert!(should_continue_validation_growth(100.0, 100.0, -0.5));
        assert!(!should_continue_validation_growth(100.0, 99.0, -0.5));
        assert!(should_continue_validation_growth(100.0, 95.0, 0.05));
    }

    #[test]
    fn maybe_expand_probe_candidates_pushes_next_candidate_when_score_is_within_tolerance() {
        let config = dummy_config();
        let mut candidates = vec![32, 64];
        let summary = summary(64, None);
        let mut growth_state = ProbeGrowthState {
            patience: 1,
            steps: 0,
            prior_best_score: None,
        };

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 1,
                kind: ProbeKind::Train,
                candidate: 64,
                summary: &summary,
                candidate_score: 97.0,
                tolerance: 0.03,
            },
            &config,
            &mut growth_state,
        );

        assert!(!should_stop);
        assert_eq!(candidates, vec![32, 64, 128]);
        assert_eq!(growth_state.patience, 0);
        assert_eq!(growth_state.steps, 1);
        assert_eq!(growth_state.prior_best_score, Some(97.0));
    }

    #[test]
    fn maybe_expand_probe_candidates_stops_when_growth_steps_are_exhausted() {
        let mut config = dummy_config();
        config.preflight.validation_growth_max_steps = 0;
        let mut candidates = vec![64];
        let summary = summary(64, Some(110.0));
        let mut growth_state = ProbeGrowthState {
            patience: 0,
            steps: 1,
            prior_best_score: None,
        };

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 0,
                kind: ProbeKind::Train,
                candidate: 64,
                summary: &summary,
                candidate_score: 109.0,
                tolerance: 0.02,
            },
            &config,
            &mut growth_state,
        );

        assert!(should_stop);
        assert_eq!(candidates, vec![64]);
        assert_eq!(growth_state.prior_best_score, None);
    }

    #[test]
    fn maybe_expand_probe_candidates_stops_after_patience_limit_on_dropoff() {
        let mut config = dummy_config();
        config.preflight.validation_growth_patience = 2;
        let mut candidates = vec![64];
        let summary = summary(64, Some(118.0));
        let mut growth_state = ProbeGrowthState {
            patience: 1,
            steps: 0,
            prior_best_score: Some(120.0),
        };

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 0,
                kind: ProbeKind::Train,
                candidate: 64,
                summary: &summary,
                candidate_score: 100.0,
                tolerance: 0.05,
            },
            &config,
            &mut growth_state,
        );

        assert!(should_stop);
        assert_eq!(candidates, vec![64]);
        assert_eq!(growth_state.patience, 2);
        assert_eq!(growth_state.steps, 0);
        assert_eq!(growth_state.prior_best_score, Some(120.0));
    }

    #[test]
    fn maybe_expand_probe_candidates_updates_best_score_even_without_expansion() {
        let config = dummy_config();
        let mut candidates = vec![32, 64, 128];
        let summary = summary(64, Some(110.0));
        let mut growth_state = ProbeGrowthState {
            patience: 0,
            steps: 0,
            prior_best_score: Some(105.0),
        };

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 1,
                kind: ProbeKind::Train,
                candidate: 64,
                summary: &summary,
                candidate_score: 90.0,
                tolerance: 0.01,
            },
            &config,
            &mut growth_state,
        );

        assert!(!should_stop);
        assert_eq!(candidates, vec![32, 64, 128]);
        assert_eq!(growth_state.patience, 0);
        assert_eq!(growth_state.steps, 0);
        assert_eq!(growth_state.prior_best_score, Some(110.0));
    }

    #[test]
    fn maybe_expand_probe_candidates_ignores_non_top_candidate() {
        let config = dummy_config();
        let mut candidates = vec![32, 64];
        let summary = summary(32, Some(111.0));
        let mut growth_state = ProbeGrowthState::default();

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 0,
                kind: ProbeKind::Train,
                candidate: 32,
                summary: &summary,
                candidate_score: 100.0,
                tolerance: 0.05,
            },
            &config,
            &mut growth_state,
        );

        assert!(!should_stop);
        assert_eq!(candidates, vec![32, 64]);
        assert_eq!(growth_state.patience, 0);
        assert_eq!(growth_state.steps, 0);
        assert_eq!(growth_state.prior_best_score, Some(111.0));
    }

    #[test]
    fn maybe_expand_probe_candidates_ignores_mismatched_summary_candidate() {
        let config = dummy_config();
        let mut candidates = vec![32, 64];
        let summary = summary(32, Some(103.0));
        let mut growth_state = ProbeGrowthState {
            patience: 1,
            steps: 2,
            prior_best_score: Some(101.0),
        };

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 1,
                kind: ProbeKind::Train,
                candidate: 64,
                summary: &summary,
                candidate_score: 99.0,
                tolerance: 0.02,
            },
            &config,
            &mut growth_state,
        );

        assert!(!should_stop);
        assert_eq!(candidates, vec![32, 64]);
        assert_eq!(growth_state.patience, 1);
        assert_eq!(growth_state.steps, 2);
        assert_eq!(growth_state.prior_best_score, Some(103.0));
    }

    #[test]
    fn maybe_expand_probe_candidates_stops_growing_when_ceiling_is_reached() {
        let config = dummy_config();
        let mut candidates = vec![128, 256];
        let summary = summary(256, Some(140.0));
        let mut growth_state = ProbeGrowthState::default();

        let should_stop = maybe_expand_probe_candidates(
            &mut candidates,
            ProbeGrowthDecision {
                index: 1,
                kind: ProbeKind::Train,
                candidate: 256,
                summary: &summary,
                candidate_score: 135.0,
                tolerance: 0.01,
            },
            &config,
            &mut growth_state,
        );

        assert!(!should_stop);
        assert_eq!(candidates, vec![128, 256]);
        assert_eq!(growth_state.steps, 0);
        assert_eq!(growth_state.prior_best_score, Some(140.0));
    }

    #[test]
    fn rerun_probe_finalists_returns_early_when_fewer_than_two_finalists_exist() {
        let config = dummy_config();
        let mut results = vec![probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(500.0),
            Some(1.0),
        )];
        let progress = hidden_progress();

        rerun_probe_finalists(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected finalist rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
        )
        .expect("single finalist should skip reruns");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].candidate_microbatch, 64);
    }

    #[test]
    fn refine_probe_winner_locally_returns_early_when_local_refinement_is_disabled() {
        let mut config = dummy_config();
        config.preflight.local_refinement_enabled = false;
        let mut results = vec![probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(480.0),
            Some(1.0),
        )];
        let progress = hidden_progress();

        refine_probe_winner_locally(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected local refinement rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
        )
        .expect("disabled local refinement should be a no-op");

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn refine_probe_winner_locally_returns_early_when_no_local_candidates_exist() {
        let config = dummy_config();
        let mut results = vec![probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(480.0),
            Some(1.0),
        )];
        let progress = hidden_progress();

        refine_probe_winner_locally(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected local refinement rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
        )
        .expect("missing local candidates should skip reruns");

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn refine_top_k_probe_candidates_locally_returns_early_when_no_candidates_exist() {
        let config = dummy_config();
        let mut results = vec![probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(480.0),
            Some(1.0),
        )];
        let progress = hidden_progress();

        refine_top_k_probe_candidates_locally(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected top-k refinement rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
        )
        .expect("missing top-k candidates should skip reruns");

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn finalize_probe_search_returns_missing_error_when_no_stable_results_exist() {
        let mut config = dummy_config();
        config.preflight.local_refinement_enabled = false;
        let mut results = Vec::new();
        let progress = hidden_progress();

        let error = finalize_probe_search(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected finalize rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
            "missing winner".to_string(),
        )
        .expect_err("empty successful results should fail");

        assert_eq!(error, "missing winner");
        assert!(results.is_empty());
    }

    #[test]
    fn finalize_probe_search_returns_best_summary_when_refinements_have_no_work() {
        let mut config = dummy_config();
        config.preflight.local_refinement_enabled = false;
        let mut results = vec![probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(512.0),
            Some(1.5),
        )];
        let progress = hidden_progress();

        let best = finalize_probe_search(
            Path::new("/dev/null"),
            |_, _, _| panic!("unexpected finalize rerun"),
            ProbeKind::Train,
            &config,
            &mut results,
            &progress,
            "missing winner".to_string(),
        )
        .expect("single stable result should finalize cleanly");

        assert_eq!(best.candidate_microbatch, 64);
        assert_eq!(best.average_samples_per_second, Some(512.0));
        assert_eq!(best.average_elapsed_seconds, Some(1.5));
        assert_eq!(best.attempts, 1);
    }

    #[test]
    fn probe_candidate_ladder_errors_when_candidate_list_is_empty() {
        let mut config = dummy_config();
        config.microbatch_size = None;
        let artifacts = crate::artifacts::BcArtifactPaths::new(Path::new("/tmp/out"), 0);

        let error = probe_candidate_ladder(
            Path::new("/dev/null"),
            &config,
            &artifacts,
            ProbeKind::Train,
            &[],
        )
        .expect_err("empty candidate ladder should not find a stable winner");

        assert_eq!(error, "no stable train microbatch found in preflight");
    }

    #[test]
    fn probe_candidate_ladder_prefers_explicit_only_train_candidate_before_results_exist() {
        let mut config = dummy_config();
        config.preflight.allow_override_explicit_microbatch = false;
        config.microbatch_size = Some(96);
        let artifacts = crate::artifacts::BcArtifactPaths::new(Path::new("/tmp/out"), 0);

        let error = probe_candidate_ladder(
            Path::new("/dev/null"),
            &config,
            &artifacts,
            ProbeKind::Train,
            &[32, 64],
        )
        .expect_err(
            "explicit-only train ladder should fail against invalid config path before probing",
        );

        assert!(error.contains("/dev/null"));
        assert!(error.contains("config") || error.contains("extension"));
    }

    #[test]
    fn probe_candidate_ladder_prefers_explicit_only_rl_microbatch_candidate() {
        let mut config = dummy_config();
        config.preflight.allow_override_explicit_microbatch = false;
        config.rl = Some(crate::config::RlTrainConfig {
            microbatch_size: Some(24),
            ..crate::config::RlTrainConfig::default()
        });
        let artifacts = crate::artifacts::BcArtifactPaths::new(Path::new("/tmp/out"), 0);

        let error = probe_candidate_ladder(
            Path::new("/dev/null"),
            &config,
            &artifacts,
            ProbeKind::RlMicrobatch,
            &[16, 32],
        )
        .expect_err("explicit-only RL microbatch ladder should fail against invalid config path before probing");

        assert!(error.contains("/dev/null"));
        assert!(error.contains("config") || error.contains("extension"));
    }

    #[test]
    fn probe_candidate_ladder_rl_games_does_not_use_explicit_only_override_path() {
        let mut config = dummy_config();
        config.preflight.allow_override_explicit_microbatch = false;
        config.rl = Some(crate::config::RlTrainConfig {
            games_per_batch: 24,
            ..crate::config::RlTrainConfig::default()
        });
        let artifacts = crate::artifacts::BcArtifactPaths::new(Path::new("/tmp/out"), 0);

        let error = probe_candidate_ladder(
            Path::new("missing-config.yaml"),
            &config,
            &artifacts,
            ProbeKind::RlGames,
            &[16, 32],
        )
        .expect_err("rl games ladder should still use the normal candidate ladder path");

        assert!(error.contains("failed to read config missing-config.yaml"));
    }

    #[test]
    fn probe_candidate_ladder_returns_explicit_failure_message_when_probe_attempt_fails() {
        let mut config = dummy_config();
        config.preflight.allow_override_explicit_microbatch = false;
        config.microbatch_size = Some(64);
        let artifacts = crate::artifacts::BcArtifactPaths::new(Path::new("/tmp/out"), 0);

        let error = probe_candidate_ladder(
            Path::new("missing-config.yaml"),
            &config,
            &artifacts,
            ProbeKind::Train,
            &[64],
        )
        .expect_err("explicit-only train candidate should bubble probe-request failure");

        assert!(error.contains("failed to read config missing-config.yaml"));
    }

    #[test]
    fn run_candidate_attempts_batches_attempts_and_preserves_result_order() {
        let mut called_paths = Vec::new();
        let mut results = Vec::new();
        let progress = hidden_progress();
        let returned_results = vec![
            probe_result_with_detail(ProbeKind::Train, 1, ProbeStatus::Success, "attempt 1"),
            probe_result_with_detail(ProbeKind::Train, 1, ProbeStatus::Success, "attempt 2"),
            probe_result_with_detail(ProbeKind::Train, 1, ProbeStatus::Success, "attempt 3"),
        ];

        let passed = run_candidate_attempts_with_batch_executor(
            Path::new("/dev/null.yaml"),
            &mut |kind, candidate, attempt| {
                let path = PathBuf::from("/home/nikketryhard/tmp").join(format!(
                    "legacy-{}-{candidate}-{attempt}.json",
                    probe_kind_name(kind)
                ));
                called_paths.push(path.clone());
                path
            },
            ProbeRunSpec {
                kind: ProbeKind::Train,
                candidate: 1,
                attempts: 3,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &mut results,
            &progress,
            |config_path, batch, results_path| {
                assert_eq!(config_path, Path::new("/dev/null.yaml"));
                assert_eq!(batch.request.kind, ProbeKind::Train);
                assert_eq!(batch.request.candidate_microbatch, 1);
                assert_eq!(batch.attempts, 3);
                assert!(results_path.ends_with("legacy-train-1-0.batch.json"));
                Ok(returned_results.clone())
            },
        )
        .expect("batched candidate attempts should succeed");

        assert!(passed);
        assert_eq!(called_paths.len(), 1);
        assert_eq!(results.len(), 3);
        assert!(
            results
                .iter()
                .all(|result| result.status == ProbeStatus::Success)
        );
        assert_eq!(
            results
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["attempt 1", "attempt 2", "attempt 3"]
        );
        assert_eq!(progress.position(), 3);
    }

    #[test]
    fn run_candidate_attempts_explicit_success_path_replays_all_logical_attempts() {
        let mut results = Vec::new();
        let progress = hidden_progress();
        let returned_results = vec![
            probe_result_with_detail(ProbeKind::Train, 1, ProbeStatus::Success, "first"),
            probe_result_with_detail(ProbeKind::Train, 1, ProbeStatus::Success, "second"),
        ];

        let passed = run_candidate_attempts_with_batch_executor(
            Path::new("/dev/null.yaml"),
            &mut |kind, candidate, attempt| {
                PathBuf::from("/home/nikketryhard/tmp").join(format!(
                    "explicit-{}-{candidate}-{attempt}.json",
                    probe_kind_name(kind)
                ))
            },
            ProbeRunSpec {
                kind: ProbeKind::Train,
                candidate: 1,
                attempts: 2,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &mut results,
            &progress,
            |_, batch, results_path| {
                assert_eq!(batch.attempts, 2);
                assert!(results_path.ends_with("explicit-train-1-0.batch.json"));
                Ok(returned_results.clone())
            },
        )
        .expect("explicit success path should replay every logical attempt");

        assert!(passed);
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|result| result.status == ProbeStatus::Success)
        );
        assert_eq!(
            results
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(progress.position(), 2);
    }

    #[test]
    fn run_candidate_attempts_stops_on_first_failure_with_same_prefix_results() {
        let mut results = Vec::new();
        let progress = hidden_progress();

        let passed = run_candidate_attempts_with_batch_executor(
            Path::new("/dev/null.yaml"),
            &mut |kind, candidate, attempt| {
                PathBuf::from("/home/nikketryhard/tmp").join(format!(
                    "failure-{}-{candidate}-{attempt}.json",
                    probe_kind_name(kind)
                ))
            },
            ProbeRunSpec {
                kind: ProbeKind::Train,
                candidate: 64,
                attempts: 4,
                warmup_steps: 1,
                measure_steps: 1,
            },
            &mut results,
            &progress,
            |_, batch, results_path| {
                assert_eq!(batch.attempts, 4);
                assert!(results_path.ends_with("failure-train-64-0.batch.json"));
                Ok(vec![
                    probe_result_with_detail(
                        ProbeKind::Train,
                        64,
                        ProbeStatus::Success,
                        "attempt 1",
                    ),
                    probe_result_with_detail(
                        ProbeKind::Train,
                        64,
                        ProbeStatus::BackendError,
                        "attempt 2 failed",
                    ),
                    probe_result_with_detail(
                        ProbeKind::Train,
                        64,
                        ProbeStatus::Success,
                        "attempt 3 should not replay",
                    ),
                ])
            },
        )
        .expect("first failing replay should bubble as Ok(false)");

        assert!(!passed);
        assert_eq!(progress.position(), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["attempt 1", "attempt 2 failed"]
        );
        assert_eq!(results[1].status, ProbeStatus::BackendError);
    }

    #[test]
    fn run_candidate_attempts_uses_batch_artifact_path_without_legacy_collision() {
        let legacy = PathBuf::from("/home/nikketryhard/tmp/legacy-train-128-0.json");
        let batch = probe_batch_results_path(&legacy);
        let artifact = ProbeBatchArtifact::from_results(
            vec![probe_result_with_detail(
                ProbeKind::Train,
                128,
                ProbeStatus::Success,
                "attempt 1",
            )],
            true,
        );

        assert_eq!(
            batch,
            PathBuf::from("/home/nikketryhard/tmp/legacy-train-128-0.batch.json")
        );
        assert_ne!(batch, legacy);
        assert_ne!(
            batch,
            PathBuf::from("/home/nikketryhard/tmp/legacy-train-128-1.json")
        );
        assert_eq!(artifact.replay_ordered_results().count(), 1);
    }

    #[test]
    fn replay_candidate_attempt_results_preserves_order() {
        let progress = hidden_progress();
        let mut results = Vec::new();

        let passed = replay_candidate_attempt_results(
            ProbeKind::Validation,
            32,
            3,
            vec![
                probe_result_with_detail(
                    ProbeKind::Validation,
                    32,
                    ProbeStatus::Success,
                    "attempt 1",
                ),
                probe_result_with_detail(
                    ProbeKind::Validation,
                    32,
                    ProbeStatus::Success,
                    "attempt 2",
                ),
                probe_result_with_detail(
                    ProbeKind::Validation,
                    32,
                    ProbeStatus::Success,
                    "attempt 3",
                ),
            ],
            &mut results,
            &progress,
            0,
        );

        assert!(passed);
        assert_eq!(
            results
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["attempt 1", "attempt 2", "attempt 3"]
        );
        assert_eq!(progress.position(), 3);
    }

    #[test]
    fn replay_candidate_attempt_results_stops_on_first_failure() {
        let progress = hidden_progress();
        let mut results = Vec::new();

        let passed = replay_candidate_attempt_results(
            ProbeKind::Train,
            64,
            3,
            vec![
                probe_result_with_detail(ProbeKind::Train, 64, ProbeStatus::Success, "attempt 1"),
                probe_result_with_detail(
                    ProbeKind::Train,
                    64,
                    ProbeStatus::BackendError,
                    "attempt 2 failed",
                ),
                probe_result_with_detail(ProbeKind::Train, 64, ProbeStatus::Success, "attempt 3"),
            ],
            &mut results,
            &progress,
            0,
        );

        assert!(!passed);
        assert_eq!(progress.position(), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].detail, "attempt 1");
        assert_eq!(results[1].detail, "attempt 2 failed");
        assert_eq!(results[1].status, ProbeStatus::BackendError);
    }

    #[test]
    fn replay_candidate_attempt_results_explicit_success_path_replays_all_attempts() {
        let progress = hidden_progress();
        let mut results = Vec::new();

        let passed = replay_candidate_attempt_results(
            ProbeKind::Train,
            96,
            2,
            vec![
                probe_result_with_detail(ProbeKind::Train, 96, ProbeStatus::Success, "first"),
                probe_result_with_detail(ProbeKind::Train, 96, ProbeStatus::Success, "second"),
            ],
            &mut results,
            &progress,
            0,
        );

        assert!(passed);
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|result| result.status == ProbeStatus::Success)
        );
        assert_eq!(
            results
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(progress.position(), 2);
    }
}
