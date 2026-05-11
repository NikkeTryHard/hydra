use hydra_train_runtime::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use crate::probe_summary::{best_probe_summary, probe_kind_name, summarize_probe_results};

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
fn best_probe_summary_prefers_larger_microbatch_when_average_ties() {
    let summary = best_probe_summary(&[
        ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(500.0),
            elapsed_seconds: Some(2.0),
            detail: String::new(),
        },
        ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 128,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(500.0),
            elapsed_seconds: Some(2.0),
            detail: String::new(),
        },
    ])
    .expect("best summary should exist");

    assert_eq!(summary.candidate_microbatch, 128);
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
