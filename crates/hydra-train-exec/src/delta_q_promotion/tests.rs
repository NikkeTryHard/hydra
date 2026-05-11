use super::*;
use burn::backend::NdArray;
use burn::tensor::TensorData;
use hydra_model::model::HydraModelConfig;
use hydra_train_types::delta_q_promotion::ArenaPromotionDecision;

type B = NdArray<f32>;

fn tensor2(values: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values, [rows, cols]), &Default::default())
}

fn dummy_targets() -> HydraTargets<B> {
    let device = Default::default();
    HydraTargets {
        policy_target: Tensor::zeros([2, 46], &device),
        legal_mask: Tensor::ones([2, 46], &device),
        value_target: Tensor::zeros([2], &device),
        grp_target: Tensor::zeros([2, 24], &device),
        tenpai_target: Tensor::zeros([2, 3], &device),
        danger_target: Tensor::zeros([2, 3, 34], &device),
        danger_mask: Tensor::zeros([2, 3, 34], &device),
        opp_next_target: Tensor::zeros([2, 3, 34], &device),
        score_pdf_target: Tensor::zeros([2, 64], &device),
        score_cdf_target: Tensor::zeros([2, 64], &device),
        oracle_target: None,
        belief_fields_target: None,
        belief_fields_mask: None,
        mixture_weight_target: None,
        mixture_weight_mask: None,
        opponent_hand_type_target: None,
        delta_q_target: None,
        delta_q_mask: None,
        safety_residual_target: None,
        safety_residual_mask: None,
        oracle_guidance_mask: None,
        target_presence: None,
    }
}

fn dummy_output(policy_logits: Tensor<B, 2>, delta_q: Tensor<B, 2>) -> HydraOutput<B> {
    let device = Default::default();
    HydraOutput {
        policy_logits,
        value: Tensor::zeros([2, 1], &device),
        score_pdf: Tensor::zeros([2, 64], &device),
        score_cdf: Tensor::zeros([2, 64], &device),
        opp_tenpai: Tensor::zeros([2, 3], &device),
        grp: Tensor::zeros([2, 24], &device),
        opp_next_discard: Tensor::zeros([2, 3, 34], &device),
        danger: Tensor::zeros([2, 3, 34], &device),
        oracle_critic: Tensor::zeros([2, 4], &device),
        belief_fields: Tensor::zeros([2, 16, 34], &device),
        mixture_weight_logits: Tensor::zeros([2, 4], &device),
        opponent_hand_type: Tensor::zeros([2, 24], &device),
        delta_q,
        safety_residual: Tensor::zeros([2, 46], &device),
    }
}

#[test]
fn collect_promotion_metrics_reports_candidate_advantage() {
    let mut targets = dummy_targets();
    let mut delta_q_target = vec![0.0f32; 2 * 46];
    let mut delta_q_mask = vec![0.0f32; 2 * 46];
    let mut policy_logits = vec![0.0f32; 2 * 46];
    let mut candidate_delta_q = vec![0.0f32; 2 * 46];

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

    targets.delta_q_target = Some(tensor2(delta_q_target, 2, 46));
    targets.delta_q_mask = Some(tensor2(delta_q_mask, 2, 46));

    let output = dummy_output(
        tensor2(policy_logits, 2, 46),
        tensor2(candidate_delta_q, 2, 46),
    );
    let report = collect_promotion_metrics_from_outputs(&output, &targets, 0.5);

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
fn collect_policy_transfer_metrics_reports_policy_advantage() {
    let mut targets = dummy_targets();
    let mut delta_q_target = vec![0.0f32; 2 * 46];
    let mut delta_q_mask = vec![0.0f32; 2 * 46];
    let mut candidate_policy = vec![0.0f32; 2 * 46];
    let mut baseline_policy = vec![0.0f32; 2 * 46];

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

    targets.delta_q_target = Some(tensor2(delta_q_target, 2, 46));
    targets.delta_q_mask = Some(tensor2(delta_q_mask, 2, 46));

    let report = collect_policy_transfer_metrics_from_policy_outputs(
        tensor2(candidate_policy, 2, 46),
        tensor2(baseline_policy, 2, 46),
        &targets,
    );

    assert_eq!(report.compared_states, 2);
    assert_eq!(report.candidate_policy_top1_to_teacher_count, 2);
    assert_eq!(report.baseline_policy_top1_to_teacher_count, 0);
    assert_eq!(report.candidate_beats_baseline_count, 2);
    assert_eq!(report.negative_transfer_count, 0);
    assert!(report.mean_regret_improvement() > 0.0);
}

#[test]
fn collect_metrics_without_delta_q_targets_returns_empty_reports() {
    let targets = dummy_targets();
    let output = dummy_output(
        tensor2(vec![0.0; 2 * 46], 2, 46),
        tensor2(vec![0.0; 2 * 46], 2, 46),
    );

    let promotion = collect_promotion_metrics_from_outputs(&output, &targets, 0.5);
    let transfer = collect_policy_transfer_metrics_from_policy_outputs(
        tensor2(vec![0.0; 2 * 46], 2, 46),
        tensor2(vec![0.0; 2 * 46], 2, 46),
        &targets,
    );

    assert_eq!(promotion.eligible_states, 0);
    assert_eq!(promotion.compared_states, 0);
    assert_eq!(transfer.compared_states, 0);
}

#[test]
fn paired_arena_result_recommends_non_regression_then_strong_promotion() {
    let cfg = PairedArenaEvalConfig::new()
        .with_max_mean_placement_regression(0.025)
        .with_strong_promotion_mean_placement_target(0.0);
    let candidate = vec![0, 1, 1, 2, 2, 2, 0, 1];
    let baseline = vec![1, 2, 2, 3, 2, 3, 1, 2];

    let result = paired_arena_result_from_placements(&candidate, &baseline, 0.02);
    assert_eq!(
        result.recommendation(&cfg),
        ArenaPromotionDecision::NonRegressionOnly
    );

    let strong = paired_arena_result_from_placements(&candidate, &baseline, -0.01);
    assert_eq!(
        strong.recommendation(&cfg),
        ArenaPromotionDecision::StrongPromotion
    );
}

#[test]
fn paired_arena_result_rejects_regression() {
    let cfg = PairedArenaEvalConfig::new().with_max_mean_placement_regression(0.025);
    let candidate = vec![2, 2, 3, 3, 2, 3, 3, 2];
    let baseline = vec![0, 1, 1, 2, 1, 2, 1, 2];
    let result = paired_arena_result_from_placements(&candidate, &baseline, 0.05);
    assert_eq!(result.recommendation(&cfg), ArenaPromotionDecision::Reject);
}

#[test]
fn paired_delta_q_arena_confirmation_is_zero_delta_for_identical_models() {
    let device = Default::default();
    let model = HydraModelConfig::new(1)
        .with_hidden_channels(2)
        .with_se_bottleneck(1)
        .with_num_groups(1)
        .init::<B>(&device);
    let cfg = PairedArenaEvalConfig::new()
        .with_min_games(2)
        .with_seed(123);

    let outcome = run_paired_delta_q_arena_confirmation(&model, &model, &device, &cfg, 1.0);

    assert_eq!(outcome.paired_result.compared_games, 2);
    assert!(outcome.paired_result.delta_mean_placement.abs() < 1e-6);
    assert!(
        outcome
            .paired_result
            .upper_confidence_bound_mean_placement
            .abs()
            < 1e-6
    );
    assert!(outcome.lower_confidence_bound_mean_placement.abs() < 1e-6);
}

#[test]
fn paired_bootstrap_ci_tracks_candidate_regression_direction() {
    let candidate = [2, 2, 3, 3, 2, 3, 3, 2];
    let baseline = [0, 1, 1, 2, 1, 2, 1, 2];

    let (lower, upper) = paired_bootstrap_mean_placement_ci(&candidate, &baseline, 99, 128);

    assert!(lower > 0.0);
    assert!(upper > 0.0);
    let result = paired_arena_result_from_placements(&candidate, &baseline, upper);
    assert!(result.delta_mean_placement > 0.0);
}

#[test]
fn paired_result_and_bootstrap_helpers_cover_empty_and_singleton_edges() {
    let cfg = PairedArenaEvalConfig::new();
    let single = paired_arena_result_from_placements(&[0], &[1], 0.1);
    assert_eq!(single.compared_games, 1);
    assert!(single.summary(&cfg).contains("decision="));

    let (lower, upper) = paired_bootstrap_mean_placement_ci(&[], &[], 7, 0);
    assert_eq!((lower, upper), (0.0, 0.0));

    let (lower, upper) = paired_bootstrap_mean_placement_ci(&[0], &[1], 7, 0);
    assert_eq!((lower, upper), (-1.0, -1.0));
}

#[test]
fn pre_arena_recommendation_requires_both_offline_and_transfer_gate() {
    assert_eq!(
        pre_arena_recommendation(true, Some(true)),
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    );
    assert_eq!(
        pre_arena_recommendation(true, None),
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    );
    assert_eq!(
        pre_arena_recommendation(true, Some(false)),
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    );
    assert_eq!(
        pre_arena_recommendation(false, Some(true)),
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    );
}

#[test]
fn default_arena_confirmation_request_tracks_recommendation() {
    let request = default_arena_confirmation_request(
        DeltaQPromotionRecommendation::RequiresArenaConfirmation,
    )
    .expect("arena confirmation request should exist");
    assert!(request.same_seeds);
    assert_eq!(request.min_games, 10_000);
    assert!(
        default_arena_confirmation_request(DeltaQPromotionRecommendation::RejectAtOfflineGate,)
            .is_none()
    );
}

#[test]
fn delta_q_stage_and_requirement_summary_follow_arena_presence() {
    assert_eq!(
        delta_q_promotion_stage(true),
        "offline_transfer_and_arena_gate"
    );
    assert_eq!(
        delta_q_promotion_stage(false),
        "offline_and_policy_transfer_gate"
    );

    let request = DeltaQArenaConfirmationRequest::default();
    let summary = delta_q_arena_requirement_summary(Some(&request));
    assert!(summary.contains("same_seeds=true"));
    assert!(summary.contains("min_games=10000"));
    assert_eq!(delta_q_arena_requirement_summary(None), "n/a");
}

#[test]
fn delta_q_arena_requirement_summary_reports_custom_request_fields() {
    let request = DeltaQArenaConfirmationRequest {
        min_games: 256,
        same_seeds: false,
        same_seat_rotation_schedule: false,
        same_search_budget: false,
        same_temperature: false,
        same_frozen_opponent_pool: false,
    };
    let summary = delta_q_arena_requirement_summary(Some(&request));
    assert!(summary.contains("same_seeds=false"));
    assert!(summary.contains("min_games=256"));
}

#[test]
fn delta_q_promotion_formatters_cover_offline_holdout_and_gate_messages() {
    let offline = format_delta_q_offline_gate_message(
        64,
        DeltaQPromotionSnapshot {
            compared_states: 12,
            candidate_top1_agreement: 0.75,
            candidate_mean_regret: 0.2,
            baseline_mean_regret: 0.3,
            mean_decision_lift: 0.1,
            negative_lift_fraction: 0.25,
            regret_beats_baseline_rate: 0.8,
            top1_beats_baseline_rate: 0.7,
            passed: true,
        },
        DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        "same_seeds=true min_games=10000",
        Path::new("/tmp/delta_q.json"),
    );
    assert!(offline.contains("DeltaQ offline gate"));
    assert!(offline.contains("samples=64"));
    assert!(offline.contains("compared=12"));
    assert!(offline.contains("next=requires_arena_confirmation"));
    assert!(offline.contains("artifact=/tmp/delta_q.json"));

    let holdout = format_delta_q_policy_holdout_message(DeltaQPolicyTransferSnapshot {
        compared_states: 20,
        candidate_policy_top1_to_teacher: 0.6,
        baseline_policy_top1_to_teacher: 0.5,
        candidate_policy_mean_teacher_regret: 0.2,
        baseline_policy_mean_teacher_regret: 0.25,
        candidate_beats_baseline_rate: 0.7,
        negative_transfer_fraction: 0.1,
    });
    assert!(holdout.contains("DeltaQ policy-vs-teacher holdout"));
    assert!(holdout.contains("compared=20"));
    assert!(holdout.contains("policy_top1=60.00%/50.00%"));

    let gate = format_delta_q_policy_transfer_gate_message(
        true,
        DeltaQPromotionRecommendation::RequiresArenaConfirmation,
    );
    assert!(gate.contains("DeltaQ policy transfer gate"));
    assert!(gate.contains("pass=true"));
    assert!(gate.contains("next=requires_arena_confirmation"));
}

#[test]
fn delta_q_policy_transfer_gate_and_offline_messages_cover_reject_paths() {
    let gate = format_delta_q_policy_transfer_gate_message(
        false,
        DeltaQPromotionRecommendation::RejectAtOfflineGate,
    );
    assert!(gate.contains("pass=false"));
    assert!(gate.contains("next=reject_at_offline_gate"));

    let offline = format_delta_q_offline_gate_message(
        8,
        DeltaQPromotionSnapshot {
            compared_states: 4,
            candidate_top1_agreement: 0.25,
            candidate_mean_regret: 0.5,
            baseline_mean_regret: 0.4,
            mean_decision_lift: -0.1,
            negative_lift_fraction: 0.75,
            regret_beats_baseline_rate: 0.25,
            top1_beats_baseline_rate: 0.1,
            passed: false,
        },
        DeltaQPromotionRecommendation::RejectAtOfflineGate,
        "n/a",
        Path::new("/tmp/reject.json"),
    );
    assert!(offline.contains("dq_offline_gate=false"));
    assert!(offline.contains("next=reject_at_offline_gate"));
    assert!(offline.contains("artifact=/tmp/reject.json"));
}
