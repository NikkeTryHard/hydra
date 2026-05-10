pub use hydra_train_runtime::delta_q_promotion::*;

pub use hydra_train_exec::delta_q_promotion::delta_q_arena_report_from_paired_eval;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use burn::tensor::TensorData;
    use hydra_train_exec::model::HydraOutput;
    use hydra_train_types::losses::HydraTargets;

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
    fn delta_q_arena_report_maps_from_paired_eval() {
        let paired =
            crate::eval::paired_arena_result_from_placements(&[0, 1, 1, 2], &[1, 2, 2, 3], 0.02);
        let report = DeltaQArenaReport::from_paired_eval(&paired, -0.01);
        assert_eq!(report.compared_games, 4);
        assert!((report.lower_confidence_bound_mean_placement + 0.01).abs() < 1e-9);
        assert!((report.upper_confidence_bound_mean_placement - 0.02).abs() < 1e-9);
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
}
