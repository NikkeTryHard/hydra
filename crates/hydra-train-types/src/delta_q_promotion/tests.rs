use super::*;

#[test]
fn compare_delta_q_state_requires_two_supported_actions() {
    let policy = [0.0f32; 46];
    let candidate = [0.0f32; 46];
    let mut target = [0.0f32; 46];
    target[0] = 1.0;
    let mut mask = [0.0f32; 46];
    mask[0] = 1.0;
    let legal = [1.0f32; 46];
    assert!(compare_delta_q_state(&policy, &candidate, &target, &mask, &legal).is_none());
}

#[test]
fn compare_delta_q_state_computes_regret_and_lift() {
    let mut policy = [0.0f32; 46];
    let mut candidate = [0.0f32; 46];
    let mut target = [0.0f32; 46];
    let mut mask = [0.0f32; 46];
    let legal = [1.0f32; 46];
    for entry in mask.iter_mut().take(3) {
        *entry = 1.0;
    }
    target[0] = 0.40;
    target[1] = 0.10;
    target[2] = -0.30;
    policy[1] = 3.0;
    candidate[0] = 4.0;
    let comparison =
        compare_delta_q_state(&policy, &candidate, &target, &mask, &legal).expect("state");
    assert_eq!(comparison.teacher_best_action, 0);
    assert_eq!(comparison.baseline_action, 1);
    assert_eq!(comparison.candidate_action, 0);
    assert!((comparison.baseline_regret - 0.30).abs() < 1e-6);
    assert!(comparison.candidate_regret.abs() < 1e-6);
    assert!((comparison.decision_lift - 0.30).abs() < 1e-6);
    assert!((comparison.top_gap - 0.30).abs() < 1e-6);
}

#[test]
fn collect_promotion_metrics_reports_candidate_advantage() {
    let mut delta_q_target = vec![0.0f32; 2 * 46];
    let mut delta_q_mask = vec![0.0f32; 2 * 46];
    let mut policy_logits = vec![0.0f32; 2 * 46];
    let mut candidate_delta_q = vec![0.0f32; 2 * 46];
    let legal = vec![1.0f32; 2 * 46];
    delta_q_target[0] = 0.5;
    delta_q_target[1] = 0.2;
    delta_q_target[2] = -0.1;
    delta_q_mask[0] = 1.0;
    delta_q_mask[1] = 1.0;
    delta_q_mask[2] = 1.0;
    policy_logits[1] = 5.0;
    candidate_delta_q[0] = 2.0;
    let row = 46;
    delta_q_target[row] = 0.10;
    delta_q_target[row + 1] = 0.40;
    delta_q_target[row + 2] = 0.35;
    delta_q_mask[row] = 1.0;
    delta_q_mask[row + 1] = 1.0;
    delta_q_mask[row + 2] = 1.0;
    policy_logits[row] = 3.0;
    candidate_delta_q[row + 1] = 4.0;
    let report = collect_promotion_metrics_from_slices(
        DeltaQPromotionSliceInputs {
            policy_logits: &policy_logits,
            policy_rows: 2,
            policy_width: 46,
            candidate_delta_q: &candidate_delta_q,
            candidate_delta_q_rows: 2,
            candidate_delta_q_width: 46,
            teacher_target: &delta_q_target,
            teacher_target_rows: 2,
            teacher_target_width: 46,
            teacher_mask: &delta_q_mask,
            teacher_mask_rows: 2,
            teacher_mask_width: 46,
            legal_mask: &legal,
            legal_mask_rows: 2,
            legal_mask_width: 46,
        },
        0.5,
    );
    assert_eq!(report.eligible_states, 2);
    assert_eq!(report.compared_states, 2);
    assert_eq!(report.masked_entries, 6);
    assert_eq!(report.candidate_top1_agreement_count, 2);
    assert_eq!(report.baseline_top1_agreement_count, 0);
    assert_eq!(report.candidate_regret_beats_baseline_count, 2);
    assert_eq!(report.candidate_top1_beats_baseline_count, 2);
    assert!(report.mean_decision_lift() > 0.0);
    assert_eq!(report.negative_lift_count, 0);
}

#[test]
fn checked_promotion_metrics_reject_mismatched_and_non_finite_slices() {
    let policy = vec![0.0f32; 2 * 46];
    let candidate = vec![0.0f32; 2 * 46];
    let target = vec![0.0f32; 2 * 46];
    let mask = vec![1.0f32; 2 * 46];
    let legal = vec![1.0f32; 2 * 46];
    let mut inputs = DeltaQPromotionSliceInputs {
        policy_logits: &policy,
        policy_rows: 2,
        policy_width: 46,
        candidate_delta_q: &candidate,
        candidate_delta_q_rows: 2,
        candidate_delta_q_width: 46,
        teacher_target: &target,
        teacher_target_rows: 2,
        teacher_target_width: 46,
        teacher_mask: &mask,
        teacher_mask_rows: 2,
        teacher_mask_width: 46,
        legal_mask: &legal,
        legal_mask_rows: 2,
        legal_mask_width: 46,
    };

    inputs.legal_mask_rows = 1;
    assert_eq!(
        collect_promotion_metrics_from_slices_checked(inputs, 0.5).unwrap_err(),
        "legal_mask"
    );

    let mut bad_policy = policy.clone();
    bad_policy[0] = f32::NAN;
    inputs.legal_mask_rows = 2;
    inputs.policy_logits = &bad_policy;
    assert_eq!(
        collect_promotion_metrics_from_slices_checked(inputs, 0.5).unwrap_err(),
        "delta-q slice values must be finite"
    );
}

#[test]
fn collect_policy_transfer_metrics_reports_policy_advantage() {
    let mut delta_q_target = vec![0.0f32; 2 * 46];
    let mut delta_q_mask = vec![0.0f32; 2 * 46];
    let mut candidate_policy = vec![0.0f32; 2 * 46];
    let mut baseline_policy = vec![0.0f32; 2 * 46];
    let legal = vec![1.0f32; 2 * 46];
    delta_q_target[0] = 0.5;
    delta_q_target[1] = 0.2;
    delta_q_target[2] = -0.1;
    delta_q_mask[0] = 1.0;
    delta_q_mask[1] = 1.0;
    delta_q_mask[2] = 1.0;
    candidate_policy[0] = 4.0;
    baseline_policy[1] = 3.0;
    let row = 46;
    delta_q_target[row] = 0.10;
    delta_q_target[row + 1] = 0.40;
    delta_q_target[row + 2] = 0.35;
    delta_q_mask[row] = 1.0;
    delta_q_mask[row + 1] = 1.0;
    delta_q_mask[row + 2] = 1.0;
    candidate_policy[row + 1] = 2.5;
    baseline_policy[row] = 3.2;
    let report = collect_policy_transfer_metrics_from_slices(DeltaQPolicyTransferSliceInputs {
        candidate_policy_logits: &candidate_policy,
        candidate_policy_rows: 2,
        candidate_policy_width: 46,
        baseline_policy_logits: &baseline_policy,
        baseline_policy_rows: 2,
        baseline_policy_width: 46,
        teacher_target: &delta_q_target,
        teacher_target_rows: 2,
        teacher_target_width: 46,
        teacher_mask: &delta_q_mask,
        teacher_mask_rows: 2,
        teacher_mask_width: 46,
        legal_mask: &legal,
        legal_mask_rows: 2,
        legal_mask_width: 46,
    });
    assert_eq!(report.compared_states, 2);
    assert_eq!(report.candidate_policy_top1_to_teacher_count, 2);
    assert_eq!(report.baseline_policy_top1_to_teacher_count, 0);
    assert_eq!(report.candidate_beats_baseline_count, 2);
    assert_eq!(report.negative_transfer_count, 0);
    assert!(report.mean_regret_improvement() > 0.0);
}

#[test]
fn checked_policy_transfer_metrics_reject_shape_mismatch() {
    let candidate = vec![0.0f32; 2 * 46];
    let baseline = vec![0.0f32; 2 * 46];
    let target = vec![0.0f32; 2 * 46];
    let mask = vec![1.0f32; 2 * 46];
    let legal = vec![1.0f32; 2 * 46];
    let inputs = DeltaQPolicyTransferSliceInputs {
        candidate_policy_logits: &candidate,
        candidate_policy_rows: 2,
        candidate_policy_width: 46,
        baseline_policy_logits: &baseline,
        baseline_policy_rows: 2,
        baseline_policy_width: 45,
        teacher_target: &target,
        teacher_target_rows: 2,
        teacher_target_width: 46,
        teacher_mask: &mask,
        teacher_mask_rows: 2,
        teacher_mask_width: 46,
        legal_mask: &legal,
        legal_mask_rows: 2,
        legal_mask_width: 46,
    };
    assert_eq!(
        collect_policy_transfer_metrics_from_slices_checked(inputs).unwrap_err(),
        "baseline_policy_logits"
    );
}

#[test]
fn evaluate_policy_transfer_report_requires_candidate_advantage() {
    let mut report = DeltaQPolicyTransferReport::new();
    report.compared_states = 1_500;
    report.candidate_policy_top1_to_teacher_count = 900;
    report.baseline_policy_top1_to_teacher_count = 700;
    report.candidate_policy_regret_sum = 90.0;
    report.baseline_policy_regret_sum = 150.0;
    report.candidate_beats_baseline_count = 950;
    report.negative_transfer_count = 250;
    let result = evaluate_policy_transfer_report(&report, &Default::default());
    assert!(result.passed);
    assert_eq!(
        result.recommendation(),
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    );
    report.candidate_policy_regret_sum = 170.0;
    let fail = evaluate_policy_transfer_report(&report, &Default::default());
    assert!(!fail.passed);
    assert_eq!(
        fail.recommendation(),
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    );
}

#[test]
fn evaluate_promotion_report_fails_when_candidate_regresses() {
    let report = DeltaQPromotionReport {
        eligible_states: 1_200,
        compared_states: 1_100,
        masked_entries: 4_400,
        supported_actions_sum: 4_400,
        candidate_top1_agreement_count: 500,
        baseline_top1_agreement_count: 700,
        candidate_high_gap_top1_count: 100,
        baseline_high_gap_top1_count: 150,
        high_gap_states: 200,
        candidate_regret_sum: 220.0,
        baseline_regret_sum: 110.0,
        decision_lift_sum: -110.0,
        negative_lift_count: 700,
        candidate_regret_beats_baseline_count: 100,
        candidate_top1_beats_baseline_count: 50,
    };
    let result = evaluate_promotion_report(&report, &DeltaQPromotionThresholds::default());
    assert!(!result.passed);
    assert!(result.criteria.iter().any(|criterion| !criterion.passed));
    assert!(
        result
            .criteria
            .iter()
            .any(|criterion| criterion.name == "regret_beats_baseline_rate" && !criterion.passed)
    );
    assert_eq!(
        result.recommendation(),
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    );
}

#[test]
fn promotion_result_recommends_arena_after_gate_pass() {
    let mut report = DeltaQPromotionReport::new();
    report.compared_states = 1_500;
    report.supported_actions_sum = 6_000;
    report.candidate_regret_sum = 45.0;
    report.baseline_regret_sum = 120.0;
    report.candidate_top1_agreement_count = 1_020;
    report.negative_lift_count = 150;
    report.candidate_regret_beats_baseline_count = 1_050;
    report.candidate_top1_beats_baseline_count = 700;
    let result = evaluate_promotion_report(&report, &DeltaQPromotionThresholds::default());
    assert!(result.passed);
    assert_eq!(
        result.recommendation(),
        DeltaQPromotionRecommendation::RequiresPolicyTransferGate
    );
}

#[test]
fn arena_confirmation_request_default_summary_is_stable() {
    let request = DeltaQArenaConfirmationRequest::default();
    let summary = request.summary();
    assert!(summary.contains("same_seeds=true"));
    assert!(summary.contains("same_rotation=true"));
    assert!(summary.contains("same_budget=true"));
    assert!(summary.contains("same_temp=true"));
    assert!(summary.contains("frozen_pool=true"));
    assert!(summary.contains("min_games=10000"));
}

#[test]
fn compare_delta_q_state_respects_legal_mask_overlap() {
    let policy = [0.0f32; 46];
    let candidate = [0.0f32; 46];
    let mut target = [0.0f32; 46];
    let mut mask = [0.0f32; 46];
    let mut legal = [0.0f32; 46];
    target[0] = 0.4;
    target[1] = 0.2;
    mask[0] = 1.0;
    mask[1] = 1.0;
    legal[0] = 1.0;
    assert!(compare_delta_q_state(&policy, &candidate, &target, &mask, &legal).is_none());
}

#[test]
fn argmax_over_actions_returns_none_for_missing_value() {
    assert_eq!(argmax_over_actions(&[1.0, 2.0], &[0, 2]), None);
    assert_eq!(argmax_over_actions(&[1.0, 2.0], &[]), None);
}

#[test]
fn evaluate_policy_transfer_report_handles_zero_baseline_regret_edges() {
    let mut report = DeltaQPolicyTransferReport::new();
    report.compared_states = 1_500;
    report.candidate_beats_baseline_count = 1_000;
    let zero_ratio_result = evaluate_policy_transfer_report(&report, &Default::default());
    assert!(zero_ratio_result.passed);
    assert!(zero_ratio_result.criteria.iter().any(|criterion| {
        criterion.name == "candidate_policy_mean_teacher_regret_ratio"
            && criterion.measured == 0.0
            && matches!(criterion.direction, PromotionThresholdDirection::Max)
    }));
    report.candidate_policy_regret_sum = 1.0;
    let infinite_ratio_result = evaluate_policy_transfer_report(&report, &Default::default());
    assert!(!infinite_ratio_result.passed);
    assert!(infinite_ratio_result.criteria.iter().any(|criterion| {
        criterion.name == "candidate_policy_mean_teacher_regret_ratio"
            && criterion.measured.is_infinite()
            && !criterion.passed
    }));
}

#[test]
fn promotion_recommendations_and_displays_are_stable() {
    assert_eq!(
        DeltaQPromotionRecommendation::RejectAtOfflineGate.to_string(),
        "reject_at_offline_gate"
    );
    assert_eq!(
        DeltaQPromotionRecommendation::RequiresPolicyTransferGate.to_string(),
        "requires_policy_transfer_gate"
    );
    assert_eq!(
        DeltaQPromotionRecommendation::RequiresArenaConfirmation.to_string(),
        "requires_arena_confirmation"
    );
    let promotion_result = DeltaQPromotionResult {
        passed: true,
        criteria: vec![DeltaQPromotionCriterionResult {
            name: "compared_states".to_string(),
            measured: 1_500.0,
            threshold: 1_000.0,
            passed: true,
            direction: PromotionThresholdDirection::Min,
        }],
    };
    let policy_result = DeltaQPolicyTransferResult {
        passed: false,
        criteria: vec![DeltaQPromotionCriterionResult {
            name: "candidate_policy_mean_teacher_regret_ratio".to_string(),
            measured: 1.2,
            threshold: 0.95,
            passed: false,
            direction: PromotionThresholdDirection::Max,
        }],
    };
    assert!(format!("{promotion_result}").contains("DeltaQ Promotion Result: PASS"));
    let policy_text = format!("{policy_result}");
    assert!(policy_text.contains("DeltaQ Policy Transfer Result: FAIL"));
    assert!(policy_text.contains("<="));
}
