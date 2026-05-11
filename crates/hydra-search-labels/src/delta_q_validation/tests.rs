use super::*;
use burn::backend::NdArray;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::TrajectoryDeltaQLabel;
use hydra_model::model::{HydraModelConfig, HydraModelInit};

type B = NdArray<f32>;

fn step_with_discards(discard_actions: &[usize]) -> TrajectoryStep {
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    for &action in discard_actions {
        legal_mask[action] = true;
    }
    TrajectoryStep {
        obs: [0.0; hydra_core::encoder::OBS_SIZE],
        action: discard_actions.first().copied().unwrap_or_default() as u8,
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask,
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: false,
        player_id: 0,
        game_id: 0,
        turn: 0,
        temperature: 1.0,
    }
}

fn tiny_model() -> HydraModel<B> {
    let device = Default::default();
    HydraModelConfig::new(1)
        .with_hidden_channels(16)
        .with_se_bottleneck(4)
        .with_num_groups(4)
        .init::<B>(&device)
}

fn passing_report() -> DeltaQValidationReport {
    DeltaQValidationReport {
        total_states: 2_000,
        compatible_discard_states: 1_500,
        hard_states: 300,
        labels_emitted: 100,
        labels_rejected: 1_900,
        rejected_incompatible_state: 500,
        rejected_too_few_discards: 200,
        rejected_not_hard_state: 900,
        rejected_other: 300,
        coverage_sum: 75.0,
        supported_actions_sum: 350,
        root_visits_sum: 6_400,
        masked_abs_sum: 42.0,
        masked_entry_count: 350,
        masked_zero_count: 20,
        masked_positive_count: 170,
        masked_negative_count: 160,
    }
}

fn criterion<'a>(result: &'a DeltaQValidationResult, name: &str) -> &'a DeltaQCriterionResult {
    result
        .criteria
        .iter()
        .find(|criterion| criterion.name == name)
        .unwrap_or_else(|| panic!("missing criterion: {name}"))
}

#[test]
fn test_empty_report_defaults() {
    let report = DeltaQValidationReport::new();
    assert_eq!(report.total_states, 0);
    assert_eq!(report.labels_emitted, 0);
    assert_eq!(report.emission_rate(), 0.0);
    assert_eq!(report.mean_coverage(), 0.0);
    assert_eq!(report.mean_supported_actions(), 0.0);
    assert_eq!(report.mean_abs(), 0.0);
    assert_eq!(report.positive_fraction(), 0.0);
    assert_eq!(report.negative_fraction(), 0.0);
    assert_eq!(report.zero_fraction(), 0.0);
}

#[test]
fn test_report_merge() {
    let mut lhs = DeltaQValidationReport {
        total_states: 10,
        compatible_discard_states: 7,
        hard_states: 4,
        labels_emitted: 2,
        labels_rejected: 8,
        rejected_incompatible_state: 1,
        rejected_too_few_discards: 2,
        rejected_not_hard_state: 3,
        rejected_other: 4,
        coverage_sum: 1.2,
        supported_actions_sum: 8,
        root_visits_sum: 9,
        masked_abs_sum: 1.5,
        masked_entry_count: 8,
        masked_zero_count: 1,
        masked_positive_count: 4,
        masked_negative_count: 3,
    };
    let rhs = DeltaQValidationReport {
        total_states: 5,
        compatible_discard_states: 3,
        hard_states: 2,
        labels_emitted: 1,
        labels_rejected: 4,
        rejected_incompatible_state: 2,
        rejected_too_few_discards: 3,
        rejected_not_hard_state: 4,
        rejected_other: 5,
        coverage_sum: 0.8,
        supported_actions_sum: 4,
        root_visits_sum: 11,
        masked_abs_sum: 0.7,
        masked_entry_count: 4,
        masked_zero_count: 0,
        masked_positive_count: 2,
        masked_negative_count: 2,
    };

    lhs.merge(&rhs);

    assert_eq!(lhs.total_states, 15);
    assert_eq!(lhs.compatible_discard_states, 10);
    assert_eq!(lhs.hard_states, 6);
    assert_eq!(lhs.labels_emitted, 3);
    assert_eq!(lhs.labels_rejected, 12);
    assert_eq!(lhs.rejected_incompatible_state, 3);
    assert_eq!(lhs.rejected_too_few_discards, 5);
    assert_eq!(lhs.rejected_not_hard_state, 7);
    assert_eq!(lhs.rejected_other, 9);
    assert!((lhs.coverage_sum - 2.0).abs() < 1e-9);
    assert_eq!(lhs.supported_actions_sum, 12);
    assert_eq!(lhs.root_visits_sum, 20);
    assert!((lhs.masked_abs_sum - 2.2).abs() < 1e-9);
    assert_eq!(lhs.masked_entry_count, 12);
    assert_eq!(lhs.masked_zero_count, 1);
    assert_eq!(lhs.masked_positive_count, 6);
    assert_eq!(lhs.masked_negative_count, 5);
}

#[test]
fn test_evaluate_report_all_pass() {
    let report = passing_report();
    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    assert!(result.passed);
    assert!(result.criteria.iter().all(|criterion| criterion.passed));
}

#[test]
fn test_evaluate_report_low_emission_fails() {
    let mut report = passing_report();
    report.labels_emitted = 10;
    report.coverage_sum = 8.0;
    report.supported_actions_sum = 40;

    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    assert!(!result.passed);
    assert!(!criterion(&result, "emission_rate").passed);
}

#[test]
fn test_evaluate_report_insufficient_samples_fails() {
    let mut report = passing_report();
    report.total_states = 999;

    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    assert!(!result.passed);
    assert!(!criterion(&result, "sample_size").passed);
}

#[test]
fn test_evaluate_report_no_labels_fails() {
    let report = DeltaQValidationReport {
        total_states: 2_000,
        compatible_discard_states: 1_000,
        hard_states: 100,
        labels_emitted: 0,
        labels_rejected: 2_000,
        ..DeltaQValidationReport::default()
    };

    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    assert!(!result.passed);
    assert!(!criterion(&result, "emission_rate").passed);
    assert!(!criterion(&result, "mean_coverage").passed);
    assert!(!criterion(&result, "mean_supported_actions").passed);
}

#[test]
fn test_display_formatting() {
    let report = passing_report();
    let report_text = format!("{report}");
    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    let result_text = format!("{result}");

    assert!(report_text.contains("DeltaQ Validation Report"));
    assert!(report_text.contains("Mean |delta_q|"));
    assert!(result_text.contains("DeltaQ Validation Result: PASS"));
    assert!(
        result
            .criteria
            .iter()
            .any(|criterion| criterion.name == "sample_size")
    );
}

#[test]
fn test_report_derived_metrics_match_aggregates() {
    let report = passing_report();

    assert!((report.hard_state_rate() - 0.15).abs() < 1e-9);
    assert!((report.mean_root_visits() - 64.0).abs() < 1e-9);
    assert!((report.mean_abs() - 0.12).abs() < 1e-9);
    assert!((report.positive_fraction() - (170.0 / 350.0)).abs() < 1e-9);
    assert!((report.negative_fraction() - (160.0 / 350.0)).abs() < 1e-9);
    assert!((report.zero_fraction() - (20.0 / 350.0)).abs() < 1e-9);
}

#[test]
fn test_no_label_report_keeps_sample_size_but_fails_label_quality_criteria() {
    let report = DeltaQValidationReport {
        total_states: 2_000,
        labels_rejected: 2_000,
        ..DeltaQValidationReport::default()
    };

    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());

    assert!(criterion(&result, "sample_size").passed);
    assert!(!criterion(&result, "emission_rate").passed);
    assert!(!criterion(&result, "mean_coverage").passed);
    assert!(!criterion(&result, "mean_supported_actions").passed);
    assert!(matches!(
        criterion(&result, "sample_size").direction,
        ThresholdDirection::Min
    ));
}

#[test]
fn test_display_formatting_for_failures_shows_threshold_direction() {
    let report = DeltaQValidationReport {
        total_states: 500,
        ..DeltaQValidationReport::default()
    };
    let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
    let result_text = format!("{result}");

    assert!(result_text.contains("DeltaQ Validation Result: FAIL"));
    assert!(result_text.contains("sample_size"));
    assert!(result_text.contains(">="));
}

#[test]
fn test_mean_supported_actions_and_root_visits_return_zero_without_labels() {
    let report = DeltaQValidationReport::default();

    assert_eq!(report.mean_supported_actions(), 0.0);
    assert_eq!(report.mean_root_visits(), 0.0);
}

#[test]
fn test_evaluate_report_exact_thresholds_pass() {
    let thresholds = DeltaQValidationThresholds::default();
    let report = DeltaQValidationReport {
        total_states: thresholds.min_sample_size,
        compatible_discard_states: 100,
        hard_states: 50,
        labels_emitted: 20,
        labels_rejected: 980,
        coverage_sum: thresholds.min_mean_coverage * 20.0,
        supported_actions_sum: (thresholds.min_mean_supported_actions * 20.0) as u64,
        root_visits_sum: 640,
        masked_abs_sum: 4.0,
        masked_entry_count: 20,
        masked_zero_count: 5,
        masked_positive_count: 10,
        masked_negative_count: 5,
        ..DeltaQValidationReport::default()
    };

    let result = evaluate_report(&report, &thresholds);

    assert!(result.passed);
    assert!(result.criteria.iter().all(|criterion| criterion.passed));
}

#[test]
fn test_push_min_criterion_records_direction_and_threshold_checks() {
    let mut criteria = Vec::new();
    push_min_criterion(&mut criteria, "sample_size", 5.0, 5.0);
    push_min_criterion(&mut criteria, "coverage", 0.4, 0.5);

    assert_eq!(criteria.len(), 2);
    assert!(matches!(criteria[0].direction, ThresholdDirection::Min));
    assert!(criteria[0].passed);
    assert!(!criteria[1].passed);
}

#[test]
fn test_delta_q_label_sign_fractions_cover_positive_negative_and_zero_cases() {
    let report = DeltaQValidationReport {
        labels_emitted: 1,
        masked_abs_sum: 1.5,
        masked_entry_count: 3,
        masked_zero_count: 1,
        masked_positive_count: 1,
        masked_negative_count: 1,
        ..DeltaQValidationReport::default()
    };

    assert!((report.mean_abs() - 0.5).abs() < 1e-12);
    assert!((report.positive_fraction() - (1.0 / 3.0)).abs() < 1e-12);
    assert!((report.negative_fraction() - (1.0 / 3.0)).abs() < 1e-12);
    assert!((report.zero_fraction() - (1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn test_delta_q_supported_action_counting_ignores_unmasked_entries() {
    let mut step = TrajectoryStep {
        obs: [0.0; hydra_core::encoder::OBS_SIZE],
        action: 1,
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask: [false; HYDRA_ACTION_SPACE],
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: false,
        player_id: 0,
        game_id: 0,
        turn: 0,
        temperature: 1.0,
    };
    for action in [1, 2, 3] {
        step.legal_mask[action] = true;
    }
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[1] = 0.2;
    mask[1] = 1.0;
    target[2] = -0.4;
    mask[2] = 1.0;
    target[3] = 0.0;
    mask[3] = 1.0;
    step.delta_q_label = Some(TrajectoryDeltaQLabel { target, mask });

    let label = step.delta_q_label.expect("delta-q label should be present");
    let mut report = DeltaQValidationReport::default();
    for action in 0..=DISCARD_END as usize {
        if label.mask[action] <= 0.0 {
            continue;
        }
        report.supported_actions_sum += 1;
        let value = label.target[action] as f64;
        report.masked_abs_sum += value.abs();
        report.masked_entry_count += 1;
        if value > 0.0 {
            report.masked_positive_count += 1;
        } else if value < 0.0 {
            report.masked_negative_count += 1;
        } else {
            report.masked_zero_count += 1;
        }
    }
    report.coverage_sum += report.supported_actions_sum as f64 / 3.0;

    assert_eq!(report.supported_actions_sum, 3);
    assert_eq!(report.masked_entry_count, 3);
    assert_eq!(report.masked_positive_count, 1);
    assert_eq!(report.masked_negative_count, 1);
    assert_eq!(report.masked_zero_count, 1);
    assert!((report.coverage_sum - 1.0).abs() < 1e-12);
}

#[test]
fn test_collect_validation_metrics_maps_gate_outcomes_to_delta_q_counters() {
    let device = Default::default();
    let model = tiny_model();
    let mut report = DeltaQValidationReport::default();

    let mut incompatible = step_with_discards(&[0, 1]);
    incompatible.legal_mask[DISCARD_END as usize + 1] = true;
    collect_validation_metrics_for_step(
        &incompatible,
        &model,
        &device,
        &ExitConfig::default_phase3(),
        &mut report,
    );
    assert_eq!(report.total_states, 1);
    assert_eq!(report.compatible_discard_states, 0);
    assert_eq!(report.hard_states, 0);
    assert_eq!(report.labels_rejected, 1);
    assert_eq!(report.rejected_incompatible_state, 1);

    let too_few = step_with_discards(&[2]);
    collect_validation_metrics_for_step(
        &too_few,
        &model,
        &device,
        &ExitConfig::default_phase3(),
        &mut report,
    );
    assert_eq!(report.total_states, 2);
    assert_eq!(report.compatible_discard_states, 1);
    assert_eq!(report.hard_states, 0);
    assert_eq!(report.labels_rejected, 2);
    assert_eq!(report.rejected_too_few_discards, 1);

    let not_hard = step_with_discards(&[1, 3]);
    let strict_cfg = ExitConfig {
        hard_state_threshold: f32::INFINITY,
        ..ExitConfig::default_phase3()
    };
    collect_validation_metrics_for_step(&not_hard, &model, &device, &strict_cfg, &mut report);
    assert_eq!(report.total_states, 3);
    assert_eq!(report.compatible_discard_states, 2);
    assert_eq!(report.hard_states, 0);
    assert_eq!(report.labels_rejected, 3);
    assert_eq!(report.rejected_not_hard_state, 1);

    let mut passing = step_with_discards(&[1, 3]);
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[1] = 0.25;
    target[3] = -0.25;
    mask[1] = 1.0;
    mask[3] = 1.0;
    passing.delta_q_label = Some(TrajectoryDeltaQLabel { target, mask });
    let permissive_cfg = ExitConfig {
        hard_state_threshold: -1.0,
        ..ExitConfig::default_phase3()
    };
    collect_validation_metrics_for_step(&passing, &model, &device, &permissive_cfg, &mut report);
    assert_eq!(report.total_states, 4);
    assert_eq!(report.compatible_discard_states, 3);
    assert_eq!(report.hard_states, 1);
    assert_eq!(report.labels_emitted, 1);
    assert_eq!(report.labels_rejected, 3);
}

#[test]
fn test_display_includes_zeroed_structure_metrics_when_no_masked_entries_exist() {
    let report = DeltaQValidationReport {
        total_states: 10,
        labels_rejected: 10,
        ..DeltaQValidationReport::default()
    };

    let text = format!("{report}");

    assert!(text.contains("Mean |delta_q|:      0.0000"));
    assert!(text.contains("Positive frac:       0.000"));
    assert!(text.contains("Negative frac:       0.000"));
    assert!(text.contains("Zero frac:           0.000"));
}

#[test]
fn test_report_merge_accumulates_zero_and_sign_counts_with_empty_rhs() {
    let mut lhs = DeltaQValidationReport {
        masked_abs_sum: 2.0,
        masked_entry_count: 4,
        masked_zero_count: 1,
        masked_positive_count: 2,
        masked_negative_count: 1,
        ..DeltaQValidationReport::default()
    };
    let rhs = DeltaQValidationReport::default();

    lhs.merge(&rhs);

    assert_eq!(lhs.masked_abs_sum, 2.0);
    assert_eq!(lhs.masked_entry_count, 4);
    assert_eq!(lhs.masked_zero_count, 1);
    assert_eq!(lhs.masked_positive_count, 2);
    assert_eq!(lhs.masked_negative_count, 1);
}
