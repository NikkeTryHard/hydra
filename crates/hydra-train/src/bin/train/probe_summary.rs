use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

#[derive(Debug, Clone)]
pub(super) struct ProbeCandidateSummary {
    pub(super) candidate_microbatch: usize,
    pub(super) status: ProbeStatus,
    pub(super) attempts: usize,
    pub(super) average_samples_per_second: Option<f64>,
    pub(super) average_elapsed_seconds: Option<f64>,
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

pub(super) fn probe_kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
        ProbeKind::RlGames => "rl_games",
        ProbeKind::RlMicrobatch => "rl_microbatch",
    }
}

pub(super) fn summarize_probe_results(results: &[ProbeResult]) -> Vec<ProbeCandidateSummary> {
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

pub(super) fn probe_summary_iter(
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

#[cfg(test)]
mod tests {
    use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

    use super::*;

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

    #[test]
    fn probe_kind_name_supports_rl_microbatch() {
        assert_eq!(probe_kind_name(ProbeKind::RlMicrobatch), "rl_microbatch");
    }

    #[test]
    fn summarize_probe_results_keeps_first_non_success_status_for_candidate() {
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
                status: ProbeStatus::BackendError,
                measured_samples_per_second: None,
                elapsed_seconds: None,
                detail: "backend".to_string(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Oom,
                measured_samples_per_second: None,
                elapsed_seconds: None,
                detail: "oom".to_string(),
            },
        ]);

        assert_eq!(summaries[0].status, ProbeStatus::BackendError);
    }
}
