#![allow(
    missing_docs,
    reason = "migrated train binary helpers preserve existing internal surface"
)]

use std::collections::BTreeSet;

use hydra_train_runtime::preflight::{ProbeKind, ProbeResult, ProbeStatus, candidate_ladder};

use super::probe_summary::{ProbeCandidateSummary, probe_summary_iter};
use hydra_train_runtime::config::TrainConfig;
use hydra_train_runtime::probe_request::{ProbeRequest, probe_candidate_ceiling};

const MAX_DYNAMIC_PROBE_CANDIDATE: usize = 8192;
const OOM_GEOMETRIC_BACKOFF_DIVISOR: usize = 2;

pub fn close_probe_finalists(
    results: &[ProbeResult],
    margin_ratio: f64,
    max_candidates: usize,
) -> Vec<ProbeCandidateSummary> {
    let mut summaries = probe_summary_iter(results)
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

pub fn local_refinement_candidates(
    summaries: &[ProbeCandidateSummary],
    min_gap: usize,
    max_candidates: usize,
    ceiling: usize,
) -> Vec<usize> {
    let Some(winner) = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .filter_map(|summary| {
            summary
                .average_samples_per_second
                .map(|score| (summary.candidate_microbatch, score))
        })
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(candidate, _)| candidate)
    else {
        return Vec::new();
    };

    let all_candidates = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .filter_map(|summary| {
            summary
                .average_samples_per_second
                .map(|_| summary.candidate_microbatch)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

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

pub fn top_k_refinement_candidates(
    summaries: &[ProbeCandidateSummary],
    margin_ratio: f64,
    top_k: usize,
    min_gap: usize,
    max_candidates: usize,
    ceiling: usize,
) -> Vec<usize> {
    let finalists = close_probe_finalists_from_summaries(summaries, margin_ratio, top_k);
    let all_candidates = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .filter_map(|summary| {
            summary
                .average_samples_per_second
                .map(|_| summary.candidate_microbatch)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut refined = BTreeSet::new();
    for finalist in finalists {
        let Some(index) = all_candidates
            .iter()
            .position(|candidate| *candidate == finalist)
        else {
            continue;
        };
        let lower = index
            .checked_sub(1)
            .and_then(|idx| all_candidates.get(idx).copied());
        let upper = all_candidates.get(index + 1).copied();
        let failed_above = summaries
            .iter()
            .filter(|summary| {
                summary.candidate_microbatch > finalist && summary.status != ProbeStatus::Success
            })
            .map(|summary| summary.candidate_microbatch)
            .min();
        for neighbor in [lower, upper, failed_above].into_iter().flatten() {
            let lo = neighbor.min(finalist);
            let hi = neighbor.max(finalist);
            if hi.saturating_sub(lo) < min_gap.max(1) {
                continue;
            }
            let midpoint = lo + (hi - lo) / 2;
            if midpoint != lo && midpoint != hi && midpoint <= ceiling {
                refined.insert(midpoint);
            }
        }
    }

    refined.into_iter().take(max_candidates.max(1)).collect()
}

fn close_probe_finalists_from_summaries(
    summaries: &[ProbeCandidateSummary],
    margin_ratio: f64,
    max_candidates: usize,
) -> Vec<usize> {
    let mut successful = summaries
        .iter()
        .filter(|summary| summary.status == ProbeStatus::Success)
        .collect::<Vec<_>>();
    successful.sort_by(|left, right| {
        right
            .average_samples_per_second
            .unwrap_or(0.0)
            .partial_cmp(&left.average_samples_per_second.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(best) = successful
        .first()
        .and_then(|summary| summary.average_samples_per_second)
    else {
        return Vec::new();
    };
    successful
        .into_iter()
        .filter(|summary| {
            summary
                .average_samples_per_second
                .map(|score| score >= best * (1.0 - margin_ratio.max(0.0)))
                .unwrap_or(false)
        })
        .map(|summary| summary.candidate_microbatch)
        .take(max_candidates.max(1))
        .collect()
}
fn first_unprobed_from(
    candidates: &[usize],
    probed: &BTreeSet<usize>,
    start_index: usize,
) -> usize {
    candidates
        .iter()
        .enumerate()
        .skip(start_index)
        .find(|(_, candidate)| !probed.contains(candidate))
        .map(|(index, _)| index)
        .unwrap_or(candidates.len())
}

fn closest_unprobed_candidate_in_open_range(
    candidates: &[usize],
    probed: &BTreeSet<usize>,
    lower: usize,
    upper: usize,
) -> Option<usize> {
    if lower >= upper.saturating_sub(1) {
        return None;
    }
    let midpoint = lower + (upper - lower) / 2;
    candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            **candidate > lower && **candidate < upper && !probed.contains(candidate)
        })
        .min_by_key(|(_, candidate)| candidate.abs_diff(midpoint))
        .map(|(index, _)| index)
}

fn geometric_lower_unprobed_candidate(
    candidates: &[usize],
    probed: &BTreeSet<usize>,
    oom_candidate: usize,
) -> Option<usize> {
    let target = (oom_candidate / OOM_GEOMETRIC_BACKOFF_DIVISOR).max(1);
    candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            **candidate < oom_candidate && **candidate <= target && !probed.contains(candidate)
        })
        .min_by_key(|(_, candidate)| candidate.abs_diff(target))
        .map(|(index, _)| index)
        .or_else(|| {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| **candidate < oom_candidate && !probed.contains(candidate))
                .min_by_key(|(_, candidate)| oom_candidate.saturating_sub(**candidate))
                .map(|(index, _)| index)
        })
}

pub fn adaptive_oom_probe_next_index(
    candidates: &[usize],
    results: &[ProbeResult],
    default_next_index: usize,
) -> usize {
    let probed = results
        .iter()
        .map(|result| result.candidate_microbatch)
        .collect::<BTreeSet<_>>();
    let Some(lowest_oom) = results
        .iter()
        .filter(|result| result.status == ProbeStatus::Oom)
        .map(|result| result.candidate_microbatch)
        .min()
    else {
        return first_unprobed_from(candidates, &probed, default_next_index);
    };

    if let Some(highest_success_below_oom) = results
        .iter()
        .filter(|result| {
            result.status == ProbeStatus::Success && result.candidate_microbatch < lowest_oom
        })
        .map(|result| result.candidate_microbatch)
        .max()
    {
        if let Some(index) = closest_unprobed_candidate_in_open_range(
            candidates,
            &probed,
            highest_success_below_oom,
            lowest_oom,
        ) {
            return index;
        }
    } else if results
        .last()
        .is_some_and(|result| result.status == ProbeStatus::Oom)
        && let Some(index) = geometric_lower_unprobed_candidate(candidates, &probed, lowest_oom)
    {
        return index;
    }

    first_unprobed_from(candidates, &probed, default_next_index)
}

pub fn dynamic_probe_ceiling(config: &TrainConfig, kind: ProbeKind, seed: usize) -> usize {
    match kind {
        ProbeKind::Train => config.batch_size.max(seed),
        ProbeKind::Validation => config
            .max_validation_samples
            .unwrap_or(MAX_DYNAMIC_PROBE_CANDIDATE.saturating_mul(8))
            .max(config.batch_size.max(seed).saturating_mul(8)),
        ProbeKind::RlGames => MAX_DYNAMIC_PROBE_CANDIDATE.max(seed),
        ProbeKind::RlMicrobatch => config.batch_size.max(seed),
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

pub fn probe_only_candidate_ladder(config: &TrainConfig, request: ProbeRequest) -> Vec<usize> {
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

pub fn dynamic_probe_ladder(config: &TrainConfig, kind: ProbeKind, seed: usize) -> Vec<usize> {
    let candidate_limit = dynamic_probe_ceiling(config, kind, seed.max(1));
    let mut lower = candidate_ladder(&config.preflight, candidate_limit)
        .into_iter()
        .filter(|candidate| *candidate < seed)
        .collect::<Vec<_>>();
    let mut ladder = vec![seed.max(1)];
    if matches!(kind, ProbeKind::Train) && config.batch_size > seed.max(1) {
        ladder.push(config.batch_size);
    }
    ladder.extend(dynamic_probe_growth_candidates(config, kind, seed));
    lower.sort_unstable_by(|a, b| b.cmp(a));
    ladder.extend(lower);
    let mut seen = BTreeSet::new();
    ladder.retain(|candidate| seen.insert(*candidate));
    ladder
}
