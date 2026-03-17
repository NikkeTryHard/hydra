use std::path::Path;

use colored::Colorize;
use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use super::config::TrainConfig;
use super::preflight_runtime::{
    execute_probe_request, format_probe_attempt_message, format_probe_result_summary,
};
use super::presentation::{format_status_line, make_bar};
use super::probe_ladder::{dynamic_probe_ceiling, top_k_refinement_candidates};
use super::probe_request::ProbeRequest;
use super::probe_summary::{
    best_probe_summary, probe_kind_name, summarize_probe_results, ProbeCandidateSummary,
};

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
    let ProbeRunSpec {
        kind,
        candidate,
        attempts,
        warmup_steps,
        measure_steps,
    } = spec;
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
        let result_path = result_path_for(kind, candidate, attempt);
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
    for _round in 0..config.preflight.search_coordinate_rounds.max(1) {
        for candidate in &candidates {
            if results.iter().any(|result| {
                result.kind == kind
                    && result.candidate_microbatch == *candidate
                    && result.status == ProbeStatus::Success
            }) {
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
    let mut stable_results = results
        .iter()
        .filter(|result| result.status == ProbeStatus::Success)
        .cloned()
        .collect::<Vec<_>>();
    rerun_probe_finalists(
        config_path,
        result_path_for,
        kind,
        config,
        &mut stable_results,
        progress,
    )?;
    best_probe_summary(&stable_results).ok_or(missing_error)
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
    let mut stable_results = Vec::new();
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
        let stable_start = results.len();
        let attempts = config.preflight.required_successes.max(1);
        let result_path_for = |kind, candidate, attempt| {
            super::probe_process::probe_result_path(artifacts, kind, candidate, attempt)
        };
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

    refine_probe_winner_locally(
        config_path,
        |kind, candidate, attempt| {
            super::probe_process::probe_result_path(artifacts, kind, candidate, attempt)
        },
        kind,
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
        |kind, candidate, attempt| {
            super::probe_process::probe_result_path(artifacts, kind, candidate, attempt)
        },
        kind,
        config,
        &mut stable_results,
        &progress,
    )?;

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
            super::presentation::format_preflight_selection_line(
                super::probe_summary::format_probe_selection_summary(kind, &selected_summary)
            )
        );
    }
    Ok((selected_summary.candidate_microbatch, results))
}
