use std::collections::BTreeSet;

use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus, candidate_ladder};

use super::config::TrainConfig;
use super::probe_request::{ProbeRequest, probe_candidate_ceiling};
use super::probe_summary::{ProbeCandidateSummary, summarize_probe_results};

const MAX_DYNAMIC_PROBE_CANDIDATE: usize = 8192;

pub(super) fn candidate_average(results: &[ProbeResult], candidate: usize) -> Option<f64> {
    summarize_probe_results(results)
        .into_iter()
        .find(|summary| {
            summary.candidate_microbatch == candidate && summary.status == ProbeStatus::Success
        })
        .and_then(|summary| summary.average_samples_per_second)
}

pub(super) fn close_probe_finalists(
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

pub(super) fn local_refinement_candidates(
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

pub(super) fn dynamic_probe_ceiling(config: &TrainConfig, kind: ProbeKind, seed: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hydra_train::preflight::{PreflightConfig, ProbeKind, ProbeResult, ProbeStatus};

    use super::*;
    use crate::config::TrainConfig;

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
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
    fn close_probe_finalists_keeps_candidates_within_margin() {
        let results = vec![
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(100.0),
                elapsed_seconds: Some(1.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 48,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(95.0),
                elapsed_seconds: Some(1.0),
                detail: String::new(),
            },
            ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(70.0),
                elapsed_seconds: Some(1.0),
                detail: String::new(),
            },
        ];
        let finalists = close_probe_finalists(&results, 0.1, 4);
        assert_eq!(finalists.len(), 2);
        assert_eq!(finalists[0].candidate_microbatch, 64);
        assert_eq!(finalists[1].candidate_microbatch, 48);
    }
}
