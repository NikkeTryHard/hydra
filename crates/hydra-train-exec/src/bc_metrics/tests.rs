use burn::backend::NdArray;
use burn::prelude::Tensor;
use burn::tensor::Int;
use burn::tensor::TensorData;
use hydra_train_types::losses::LossBreakdown;

use super::{
    batch_metric_sums_from_outputs, batch_stats_from_breakdown, batch_stats_from_metric_sums,
    batch_stats_from_outputs, scalar1,
};

#[test]
fn batch_stats_from_breakdown_maps_all_scalar_fields() {
    type B = NdArray<f32>;

    let breakdown = LossBreakdown {
        policy: scalar_tensor::<B>(1.0),
        value: scalar_tensor::<B>(2.0),
        grp: scalar_tensor::<B>(5.0),
        tenpai: scalar_tensor::<B>(0.0),
        danger: scalar_tensor::<B>(7.0),
        opp_next: scalar_tensor::<B>(6.0),
        score_pdf: scalar_tensor::<B>(3.0),
        score_cdf: scalar_tensor::<B>(4.0),
        oracle_critic: scalar_tensor::<B>(0.0),
        belief_fields: scalar_tensor::<B>(0.0),
        mixture_weight: scalar_tensor::<B>(0.0),
        opponent_hand_type: scalar_tensor::<B>(0.0),
        delta_q: scalar_tensor::<B>(0.0),
        safety_residual: scalar_tensor::<B>(0.0),
        total: scalar_tensor::<B>(10.0),
    };

    let stats = batch_stats_from_breakdown::<B>(4, 0.75, &breakdown);
    assert_eq!(stats.sample_count, 4);
    assert_eq!(stats.total_loss, 10.0);
    assert_eq!(stats.policy_agreement, 0.75);
    assert_eq!(stats.loss_policy, 1.0);
    assert_eq!(stats.loss_value, 2.0);
    assert_eq!(stats.loss_score_pdf, 3.0);
    assert_eq!(stats.loss_score_cdf, 4.0);
    assert_eq!(stats.loss_grp, 5.0);
    assert_eq!(stats.loss_opp_next, 6.0);
    assert_eq!(stats.loss_danger, 7.0);
}

#[test]
fn scalar1_reads_single_element_tensor() {
    type B = NdArray<f32>;
    let tensor = scalar_tensor::<B>(3.5);
    assert_eq!(scalar1(&tensor), 3.5);
}

#[test]
fn batch_stats_from_outputs_matches_breakdown_scalars_and_agreement() {
    type B = NdArray<f32>;

    let device = Default::default();
    let policy_logits = Tensor::<B, 2>::from_floats([[5.0, 1.0], [0.0, 4.0]], &device);
    let legal_mask = Tensor::<B, 2>::ones([2, 2], &device);
    let actions = Tensor::<B, 1, Int>::from_ints([0, 1], &device);
    let breakdown = LossBreakdown {
        policy: scalar_tensor::<B>(1.0),
        value: scalar_tensor::<B>(2.0),
        grp: scalar_tensor::<B>(5.0),
        tenpai: scalar_tensor::<B>(0.0),
        danger: scalar_tensor::<B>(7.0),
        opp_next: scalar_tensor::<B>(6.0),
        score_pdf: scalar_tensor::<B>(3.0),
        score_cdf: scalar_tensor::<B>(4.0),
        oracle_critic: scalar_tensor::<B>(0.0),
        belief_fields: scalar_tensor::<B>(0.0),
        mixture_weight: scalar_tensor::<B>(0.0),
        opponent_hand_type: scalar_tensor::<B>(0.0),
        delta_q: scalar_tensor::<B>(0.0),
        safety_residual: scalar_tensor::<B>(0.0),
        total: scalar_tensor::<B>(10.0),
    };

    let stats = batch_stats_from_outputs(
        2,
        policy_logits,
        legal_mask,
        actions,
        breakdown.total.clone(),
        &breakdown,
    );
    assert_eq!(stats.sample_count, 2);
    assert_eq!(stats.total_loss, 10.0);
    assert_eq!(stats.policy_agreement, 1.0);
    assert_eq!(stats.loss_policy, 1.0);
    assert_eq!(stats.loss_value, 2.0);
    assert_eq!(stats.loss_score_pdf, 3.0);
    assert_eq!(stats.loss_score_cdf, 4.0);
    assert_eq!(stats.loss_grp, 5.0);
    assert_eq!(stats.loss_opp_next, 6.0);
    assert_eq!(stats.loss_danger, 7.0);
}

#[test]
fn batch_stats_from_outputs_keeps_breakdown_total_even_when_policy_is_wrong() {
    type B = NdArray<f32>;

    let device = Default::default();
    let policy_logits = Tensor::<B, 2>::from_floats([[1.0, 5.0], [4.0, 0.0]], &device);
    let legal_mask = Tensor::<B, 2>::ones([2, 2], &device);
    let actions = Tensor::<B, 1, Int>::from_ints([0, 1], &device);
    let breakdown = LossBreakdown {
        policy: scalar_tensor::<B>(1.5),
        value: scalar_tensor::<B>(2.5),
        grp: scalar_tensor::<B>(3.5),
        tenpai: scalar_tensor::<B>(4.5),
        danger: scalar_tensor::<B>(5.5),
        opp_next: scalar_tensor::<B>(6.5),
        score_pdf: scalar_tensor::<B>(7.5),
        score_cdf: scalar_tensor::<B>(8.5),
        oracle_critic: scalar_tensor::<B>(0.0),
        belief_fields: scalar_tensor::<B>(0.0),
        mixture_weight: scalar_tensor::<B>(0.0),
        opponent_hand_type: scalar_tensor::<B>(0.0),
        delta_q: scalar_tensor::<B>(0.0),
        safety_residual: scalar_tensor::<B>(0.0),
        total: scalar_tensor::<B>(12.25),
    };

    let stats = batch_stats_from_outputs(
        2,
        policy_logits,
        legal_mask,
        actions,
        breakdown.total.clone(),
        &breakdown,
    );
    assert_eq!(stats.sample_count, 2);
    assert_eq!(stats.total_loss, 12.25);
    assert_eq!(stats.policy_agreement, 0.0);
    assert_eq!(stats.loss_policy, 1.5);
    assert_eq!(stats.loss_value, 2.5);
    assert_eq!(stats.loss_grp, 3.5);
    assert_eq!(stats.loss_tenpai, 4.5);
    assert_eq!(stats.loss_danger, 5.5);
    assert_eq!(stats.loss_opp_next, 6.5);
    assert_eq!(stats.loss_score_pdf, 7.5);
    assert_eq!(stats.loss_score_cdf, 8.5);
}

#[test]
fn rare_action_metrics_reuse_masked_prediction_for_all_buckets() {
    type B = NdArray<f32>;

    let device = Default::default();
    let actions = Tensor::<B, 1, Int>::from_ints([0, 34, 37, 38, 41, 42, 43, 44, 45], &device);
    let policy_logits = Tensor::<B, 2>::from_floats(
        [
            row_with_best(0),
            row_with_best(34),
            row_with_best(0),
            row_with_best(38),
            row_with_best(0),
            row_with_best(42),
            row_with_best(0),
            row_with_best(44),
            row_with_best(45),
        ],
        &device,
    );
    let legal_mask = Tensor::<B, 2>::ones([9, hydra_core::action::HYDRA_ACTION_SPACE], &device);
    let breakdown = zero_breakdown::<B>(1.0);

    let stats = batch_stats_from_outputs(
        9,
        policy_logits,
        legal_mask,
        actions,
        breakdown.total.clone(),
        &breakdown,
    );

    assert_eq!(stats.rare_actions.discard.count, 1);
    assert_eq!(stats.rare_actions.discard.accuracy, 1.0);
    assert_eq!(stats.rare_actions.aka_discard.count, 1);
    assert_eq!(stats.rare_actions.aka_discard.accuracy, 1.0);
    assert_eq!(stats.rare_actions.riichi.count, 1);
    assert_eq!(stats.rare_actions.riichi.accuracy, 0.0);
    assert_eq!(stats.rare_actions.chi.count, 1);
    assert_eq!(stats.rare_actions.chi.accuracy, 1.0);
    assert_eq!(stats.rare_actions.pon.count, 1);
    assert_eq!(stats.rare_actions.pon.accuracy, 0.0);
    assert_eq!(stats.rare_actions.kan.count, 1);
    assert_eq!(stats.rare_actions.kan.accuracy, 1.0);
    assert_eq!(stats.rare_actions.agari.count, 1);
    assert_eq!(stats.rare_actions.agari.accuracy, 0.0);
    assert_eq!(stats.rare_actions.ryuukyoku.count, 1);
    assert_eq!(stats.rare_actions.ryuukyoku.accuracy, 1.0);
    assert_eq!(stats.rare_actions.pass.count, 1);
    assert_eq!(stats.rare_actions.pass.accuracy, 1.0);
}

#[test]
fn metric_prediction_respects_legal_mask() {
    type B = NdArray<f32>;

    let device = Default::default();
    let policy_logits = Tensor::<B, 2>::from_floats([[1.0, 100.0, 3.0]], &device);
    let legal_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
    let actions = Tensor::<B, 1, Int>::from_ints([2], &device);
    let breakdown = zero_breakdown::<B>(1.0);

    let stats = batch_stats_from_outputs(
        1,
        policy_logits,
        legal_mask,
        actions,
        breakdown.total.clone(),
        &breakdown,
    );

    assert_eq!(stats.policy_agreement, 1.0);
    assert_eq!(stats.rare_actions.discard.count, 1);
    assert_eq!(stats.rare_actions.discard.accuracy, 1.0);
}

#[test]
fn rare_action_metrics_accumulate_across_microbatches_on_device() {
    type B = NdArray<f32>;

    let device = Default::default();
    let actions_all = Tensor::<B, 1, Int>::from_ints([37, 37, 45, 45], &device);
    let logits_all = Tensor::<B, 2>::from_floats(
        [
            row_with_best(37),
            row_with_best(0),
            row_with_best(45),
            row_with_best(0),
        ],
        &device,
    );
    let mask_all = Tensor::<B, 2>::ones([4, hydra_core::action::HYDRA_ACTION_SPACE], &device);
    let breakdown = zero_breakdown::<B>(1.0);
    let whole = batch_stats_from_outputs(
        4,
        logits_all,
        mask_all,
        actions_all,
        breakdown.total.clone(),
        &breakdown,
    );

    let first = batch_metric_sums_from_outputs(
        2,
        Tensor::<B, 2>::from_floats([row_with_best(37), row_with_best(0)], &device),
        Tensor::<B, 2>::ones([2, hydra_core::action::HYDRA_ACTION_SPACE], &device),
        Tensor::<B, 1, Int>::from_ints([37, 37], &device),
        breakdown.total.clone(),
        &breakdown,
    );
    let second = batch_metric_sums_from_outputs(
        2,
        Tensor::<B, 2>::from_floats([row_with_best(45), row_with_best(0)], &device),
        Tensor::<B, 2>::ones([2, hydra_core::action::HYDRA_ACTION_SPACE], &device),
        Tensor::<B, 1, Int>::from_ints([45, 45], &device),
        breakdown.total.clone(),
        &breakdown,
    );
    let accumulated = batch_stats_from_metric_sums(4, 2, first.accumulate(second));

    assert_eq!(accumulated.policy_agreement, whole.policy_agreement);
    assert_eq!(accumulated.rare_actions.riichi.count, 2);
    assert_eq!(accumulated.rare_actions.riichi.accuracy, 0.5);
    assert_eq!(accumulated.rare_actions.pass.count, 2);
    assert_eq!(accumulated.rare_actions.pass.accuracy, 0.5);
    assert_eq!(accumulated.rare_actions, whole.rare_actions);
}

fn scalar_tensor<B: burn::tensor::backend::Backend>(value: f32) -> burn::tensor::Tensor<B, 1> {
    burn::tensor::Tensor::<B, 1>::from_data(TensorData::from([value]), &Default::default())
}

fn row_with_best(best: usize) -> [f32; hydra_core::action::HYDRA_ACTION_SPACE] {
    let mut row = [0.0; hydra_core::action::HYDRA_ACTION_SPACE];
    row[best] = 10.0;
    row
}

fn zero_breakdown<B: burn::tensor::backend::Backend>(total: f32) -> LossBreakdown<B> {
    LossBreakdown {
        policy: scalar_tensor::<B>(0.0),
        value: scalar_tensor::<B>(0.0),
        grp: scalar_tensor::<B>(0.0),
        tenpai: scalar_tensor::<B>(0.0),
        danger: scalar_tensor::<B>(0.0),
        opp_next: scalar_tensor::<B>(0.0),
        score_pdf: scalar_tensor::<B>(0.0),
        score_cdf: scalar_tensor::<B>(0.0),
        oracle_critic: scalar_tensor::<B>(0.0),
        belief_fields: scalar_tensor::<B>(0.0),
        mixture_weight: scalar_tensor::<B>(0.0),
        opponent_hand_type: scalar_tensor::<B>(0.0),
        delta_q: scalar_tensor::<B>(0.0),
        safety_residual: scalar_tensor::<B>(0.0),
        total: scalar_tensor::<B>(total),
    }
}
