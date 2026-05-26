use super::*;
use burn::backend::NdArray;
use hydra_core::arena::TrajectoryExitLabel;
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
fn passing_report() -> ExitValidationReport {
    ExitValidationReport {
        total_states: 2_000,
        compatible_discard_states: 1_500,
        hard_states: 300,
        labels_emitted: 100,
        labels_rejected: 1_900,
        rejected_incompatible_state: 500,
        rejected_too_few_discards: 200,
        rejected_not_hard_state: 900,
        rejected_child_obs_failure: 0,
        rejected_low_coverage: 0,
        rejected_kl_safety: 0,
        rejected_other: 300,
        coverage_sum: 75.0,
        supported_actions_sum: 350,
        root_visits_sum: 6_400,
        top1_agreement_count: 97,
        kl_sum: 3.0,
    }
}

fn criterion<'a>(result: &'a ExitValidationResult, name: &str) -> &'a ExitCriterionResult {
    result
        .criteria
        .iter()
        .find(|criterion| criterion.name == name)
        .unwrap_or_else(|| panic!("missing criterion: {name}"))
}

#[test]
fn test_empty_report_defaults() {
    let report = ExitValidationReport::new();

    assert_eq!(report.total_states, 0);
    assert_eq!(report.labels_emitted, 0);
    assert_eq!(report.labels_rejected, 0);
    assert_eq!(report.emission_rate(), 0.0);
    assert_eq!(report.hard_state_rate(), 0.0);
    assert_eq!(report.mean_coverage(), 0.0);
    assert_eq!(report.mean_supported_actions(), 0.0);
    assert_eq!(report.mean_root_visits(), 0.0);
    assert_eq!(report.top1_agreement_rate(), 0.0);
    assert_eq!(report.mean_kl(), 0.0);
}

#[test]
fn test_report_merge() {
    let mut lhs = ExitValidationReport {
        total_states: 10,
        compatible_discard_states: 7,
        hard_states: 4,
        labels_emitted: 2,
        labels_rejected: 8,
        rejected_incompatible_state: 1,
        rejected_too_few_discards: 2,
        rejected_not_hard_state: 3,
        rejected_child_obs_failure: 4,
        rejected_low_coverage: 5,
        rejected_kl_safety: 6,
        rejected_other: 7,
        coverage_sum: 1.2,
        supported_actions_sum: 8,
        root_visits_sum: 9,
        top1_agreement_count: 1,
        kl_sum: 0.3,
    };
    let rhs = ExitValidationReport {
        total_states: 5,
        compatible_discard_states: 3,
        hard_states: 2,
        labels_emitted: 1,
        labels_rejected: 4,
        rejected_incompatible_state: 2,
        rejected_too_few_discards: 3,
        rejected_not_hard_state: 4,
        rejected_child_obs_failure: 5,
        rejected_low_coverage: 6,
        rejected_kl_safety: 7,
        rejected_other: 8,
        coverage_sum: 0.8,
        supported_actions_sum: 4,
        root_visits_sum: 11,
        top1_agreement_count: 1,
        kl_sum: 0.2,
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
    assert_eq!(lhs.rejected_child_obs_failure, 9);
    assert_eq!(lhs.rejected_low_coverage, 11);
    assert_eq!(lhs.rejected_kl_safety, 13);
    assert_eq!(lhs.rejected_other, 15);
    assert!((lhs.coverage_sum - 2.0).abs() < 1e-9);
    assert_eq!(lhs.supported_actions_sum, 12);
    assert_eq!(lhs.root_visits_sum, 20);
    assert_eq!(lhs.top1_agreement_count, 2);
    assert!((lhs.kl_sum - 0.5).abs() < 1e-9);
}

#[test]
fn test_emission_rate_calculation() {
    let report = ExitValidationReport {
        total_states: 80,
        labels_emitted: 20,
        ..ExitValidationReport::default()
    };

    assert!((report.emission_rate() - 0.25).abs() < 1e-12);
}

#[test]
fn test_mean_coverage_calculation() {
    let report = ExitValidationReport {
        labels_emitted: 4,
        coverage_sum: 3.0,
        ..ExitValidationReport::default()
    };

    assert!((report.mean_coverage() - 0.75).abs() < 1e-12);
}

#[test]
fn test_evaluate_report_all_pass() {
    let report = passing_report();
    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(result.passed);
    assert!(result.criteria.iter().all(|criterion| criterion.passed));
}

#[test]
fn test_evaluate_report_low_emission_fails() {
    let mut report = passing_report();
    report.labels_emitted = 10;
    report.coverage_sum = 8.0;
    report.supported_actions_sum = 40;
    report.root_visits_sum = 640;
    report.top1_agreement_count = 10;
    report.kl_sum = 0.1;

    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(!result.passed);
    assert!(!criterion(&result, "emission_rate").passed);
}

#[test]
fn test_evaluate_report_high_kl_fails() {
    let mut report = passing_report();
    report.kl_sum = 6.0;

    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(!result.passed);
    assert!(!criterion(&result, "mean_kl").passed);
}

#[test]
fn test_evaluate_report_low_agreement_fails() {
    let mut report = passing_report();
    report.top1_agreement_count = 80;

    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(!result.passed);
    assert!(!criterion(&result, "top1_agreement").passed);
}

#[test]
fn test_evaluate_report_insufficient_samples_fails() {
    let mut report = passing_report();
    report.total_states = 999;

    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(!result.passed);
    assert!(!criterion(&result, "sample_size").passed);
}

#[test]
fn test_evaluate_report_no_labels_fails() {
    let report = ExitValidationReport {
        total_states: 2_000,
        compatible_discard_states: 1_000,
        hard_states: 100,
        labels_emitted: 0,
        labels_rejected: 2_000,
        ..ExitValidationReport::default()
    };

    let result = evaluate_report(&report, &ExitValidationThresholds::default());

    assert!(!result.passed);
    assert!(!criterion(&result, "emission_rate").passed);
    assert!(!criterion(&result, "mean_coverage").passed);
    assert!(!criterion(&result, "mean_supported_actions").passed);
    assert!(!criterion(&result, "mean_kl").passed);
    assert!(!criterion(&result, "top1_agreement").passed);
}

#[test]
fn test_display_formatting() {
    let report = passing_report();
    let report_text = format!("{report}");
    let result_text = format!(
        "{}",
        evaluate_report(&report, &ExitValidationThresholds::default())
    );

    assert!(report_text.contains("ExIt Validation Report"));
    assert!(report_text.contains("Mean KL"));
    assert!(result_text.contains("ExIt Validation Result: PASS"));
    assert!(result_text.contains("sample_size"));
}

#[test]
fn test_mean_supported_actions_and_root_visits_return_zero_without_labels() {
    let report = ExitValidationReport::default();

    assert_eq!(report.mean_supported_actions(), 0.0);
    assert_eq!(report.mean_root_visits(), 0.0);
}

#[test]
fn test_evaluate_report_exact_thresholds_pass() {
    let thresholds = ExitValidationThresholds::default();
    let report = ExitValidationReport {
        total_states: thresholds.min_sample_size,
        compatible_discard_states: 100,
        hard_states: 50,
        labels_emitted: 20,
        labels_rejected: 980,
        coverage_sum: thresholds.min_mean_coverage * 20.0,
        supported_actions_sum: (thresholds.min_mean_supported_actions * 20.0) as u64,
        root_visits_sum: 640,
        top1_agreement_count: (thresholds.min_top1_agreement * 20.0) as u64,
        kl_sum: thresholds.max_mean_kl * 20.0,
        ..ExitValidationReport::default()
    };

    let result = evaluate_report(&report, &thresholds);

    assert!(result.passed);
    assert!(result.criteria.iter().all(|criterion| criterion.passed));
}

#[test]
fn test_push_criterion_helpers_record_direction_and_threshold_checks() {
    let mut criteria = Vec::new();
    push_min_criterion(&mut criteria, "min", 0.5, 0.5);
    push_max_criterion(&mut criteria, "max", 0.6, 0.5);

    assert_eq!(criteria.len(), 2);
    assert!(matches!(criteria[0].direction, ThresholdDirection::Min));
    assert!(criteria[0].passed);
    assert!(matches!(criteria[1].direction, ThresholdDirection::Max));
    assert!(!criteria[1].passed);
}

#[test]
fn test_top1_index_returns_best_action_within_subset() {
    let mut values = [0.0f32; HYDRA_ACTION_SPACE];
    values[2] = 0.25;
    values[7] = 0.9;
    values[9] = 0.5;

    assert_eq!(top1_index(&values, &[2, 7, 9]), 7);
}

#[test]
fn test_kl_divergence_ignores_unmasked_and_nonpositive_terms() {
    let mut base_pi = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    base_pi[1] = 0.6;
    exit_target[1] = 0.3;
    exit_mask[1] = 1.0;
    base_pi[2] = 0.4;
    exit_target[2] = 0.8;

    let kl = kl_divergence(&base_pi, &exit_target, &exit_mask);

    assert!(kl > 0.0);
    assert!(kl.is_finite());
}

#[test]
fn test_exit_label_top1_agreement_math_matches_expected_paths() {
    let legal_discards = vec![2usize, 4usize, 6usize];
    let mut base_pi = [0.0f32; HYDRA_ACTION_SPACE];
    base_pi[2] = 0.1;
    base_pi[4] = 0.7;
    base_pi[6] = 0.2;

    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[4] = 1.0;
    mask[4] = 1.0;
    let label = TrajectoryExitLabel { target, mask };

    assert_eq!(top1_index(&base_pi, &legal_discards), 4);
    assert_eq!(top1_index(&label.target, &legal_discards), 4);

    target[2] = 1.0;
    target[4] = 0.0;
    let disagreeing = TrajectoryExitLabel { target, mask };
    assert_ne!(
        top1_index(&base_pi, &legal_discards),
        top1_index(&disagreeing.target, &legal_discards)
    );
}

#[test]
fn test_collect_validation_metrics_maps_gate_outcomes_to_exit_counters() {
    let device = Default::default();
    let model = tiny_model();
    let mut report = ExitValidationReport::default();

    let mut incompatible = step_with_discards(&[0, 1]);
    incompatible.legal_mask[DISCARD_END as usize + 1] = true;
    collect_validation_metrics_for_step(
        &incompatible,
        &model,
        &device,
        &ExitConfig::default_live_exit(),
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
        &ExitConfig::default_live_exit(),
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
        ..ExitConfig::default_live_exit()
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
    target[1] = 1.0;
    mask[1] = 1.0;
    passing.exit_label = Some(TrajectoryExitLabel { target, mask });
    let permissive_cfg = ExitConfig {
        hard_state_threshold: -1.0,
        ..ExitConfig::default_live_exit()
    };
    collect_validation_metrics_for_step(&passing, &model, &device, &permissive_cfg, &mut report);
    assert_eq!(report.total_states, 4);
    assert_eq!(report.compatible_discard_states, 3);
    assert_eq!(report.hard_states, 1);
    assert_eq!(report.labels_emitted, 1);
    assert_eq!(report.labels_rejected, 3);
}
