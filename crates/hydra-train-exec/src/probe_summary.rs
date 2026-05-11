#![allow(
    missing_docs,
    reason = "moved train execution support preserves existing public surface"
)]

use hydra_train_runtime::preflight::{ProbeKind, ProbeResult, ProbeStatus};

#[derive(Debug, Clone)]
pub struct ProbeCandidateSummary {
    pub candidate_microbatch: usize,
    pub status: ProbeStatus,
    pub attempts: usize,
    pub average_samples_per_second: Option<f64>,
    pub average_elapsed_seconds: Option<f64>,
}

#[derive(Default)]
struct ProbeCandidateAggregate {
    attempts: usize,
    throughput_sum: f64,
    throughput_count: usize,
    elapsed_sum: f64,
    elapsed_count: usize,
    first_failure: Option<ProbeStatus>,
}

pub fn probe_kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
        ProbeKind::RlGames => "rl_games",
        ProbeKind::RlMicrobatch => "rl_microbatch",
    }
}

pub fn summarize_probe_results(results: &[ProbeResult]) -> Vec<ProbeCandidateSummary> {
    let grouped = aggregate_probe_results(results);

    grouped
        .into_iter()
        .rev()
        .map(|(candidate_microbatch, aggregate)| ProbeCandidateSummary {
            candidate_microbatch,
            status: aggregate.first_failure.unwrap_or(ProbeStatus::Success),
            attempts: aggregate.attempts,
            average_samples_per_second: (aggregate.throughput_count > 0)
                .then_some(aggregate.throughput_sum / aggregate.throughput_count as f64),
            average_elapsed_seconds: (aggregate.elapsed_count > 0)
                .then_some(aggregate.elapsed_sum / aggregate.elapsed_count as f64),
        })
        .collect()
}

pub fn probe_summary_iter(
    results: &[ProbeResult],
) -> impl DoubleEndedIterator<Item = ProbeCandidateSummary> {
    aggregate_probe_results(results)
        .into_iter()
        .rev()
        .map(|(candidate_microbatch, aggregate)| ProbeCandidateSummary {
            candidate_microbatch,
            status: aggregate.first_failure.unwrap_or(ProbeStatus::Success),
            attempts: aggregate.attempts,
            average_samples_per_second: (aggregate.throughput_count > 0)
                .then_some(aggregate.throughput_sum / aggregate.throughput_count as f64),
            average_elapsed_seconds: (aggregate.elapsed_count > 0)
                .then_some(aggregate.elapsed_sum / aggregate.elapsed_count as f64),
        })
}

fn aggregate_probe_results(
    results: &[ProbeResult],
) -> std::collections::BTreeMap<usize, ProbeCandidateAggregate> {
    let mut grouped = std::collections::BTreeMap::<usize, ProbeCandidateAggregate>::new();
    for result in results {
        let entry = grouped.entry(result.candidate_microbatch).or_default();
        entry.attempts += 1;
        if let Some(value) = result.measured_samples_per_second {
            entry.throughput_sum += value;
            entry.throughput_count += 1;
        }
        if let Some(value) = result.elapsed_seconds {
            entry.elapsed_sum += value;
            entry.elapsed_count += 1;
        }
        if entry.first_failure.is_none() && result.status != ProbeStatus::Success {
            entry.first_failure = Some(result.status.clone());
        }
    }
    grouped
}

pub fn format_probe_selection_summary(kind: ProbeKind, summary: &ProbeCandidateSummary) -> String {
    format!(
        "selected {} microbatch={} avg_throughput={:.2} samples/s avg_elapsed={:.2}s attempts={}",
        probe_kind_name(kind),
        summary.candidate_microbatch,
        summary.average_samples_per_second.unwrap_or(0.0),
        summary.average_elapsed_seconds.unwrap_or(0.0),
        summary.attempts,
    )
}

pub fn best_probe_summary(results: &[ProbeResult]) -> Option<ProbeCandidateSummary> {
    aggregate_probe_results(results)
        .into_iter()
        .map(|(candidate_microbatch, aggregate)| ProbeCandidateSummary {
            candidate_microbatch,
            status: aggregate.first_failure.unwrap_or(ProbeStatus::Success),
            attempts: aggregate.attempts,
            average_samples_per_second: (aggregate.throughput_count > 0)
                .then_some(aggregate.throughput_sum / aggregate.throughput_count as f64),
            average_elapsed_seconds: (aggregate.elapsed_count > 0)
                .then_some(aggregate.elapsed_sum / aggregate.elapsed_count as f64),
        })
        .filter(|summary| summary.status == ProbeStatus::Success)
        .max_by(|left, right| {
            left.average_samples_per_second
                .unwrap_or(0.0)
                .partial_cmp(&right.average_samples_per_second.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.candidate_microbatch.cmp(&right.candidate_microbatch))
        })
}

pub fn candidate_average(results: &[ProbeResult], candidate: usize) -> Option<f64> {
    summarize_probe_results(results)
        .into_iter()
        .find(|summary| summary.candidate_microbatch == candidate)
        .filter(|summary| summary.status == ProbeStatus::Success)
        .and_then(|summary| summary.average_samples_per_second)
}

#[cfg(test)]
mod tests;
