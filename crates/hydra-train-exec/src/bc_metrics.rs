//! Behavioral-cloning metric readback helpers.

use crate::progress::{BatchStats, RareActionBucketMetrics, RareActionMetrics};
use burn::prelude::*;
use hydra_core::action::{
    AGARI, AKA_5M, AKA_5S, CHI_LEFT, CHI_RIGHT, DISCARD_START, KAN, PASS, PON, RIICHI, RYUUKYOKU,
};
use hydra_train_types::losses::LossBreakdown;

/// Accumulated GPU and rare-action metric sums for one or more batches.
pub struct BatchMetricSums<B: Backend> {
    gpu_sums: Tensor<B, 1>,
    rare_values: [f32; 19],
}

impl<B: Backend> BatchMetricSums<B> {
    /// Adds another metric-sum accumulator into this one.
    pub fn accumulate(self, other: Self) -> Self {
        let mut rare_values = self.rare_values;
        for (lhs, rhs) in rare_values.iter_mut().zip(other.rare_values) {
            *lhs += rhs;
        }
        Self {
            gpu_sums: self.gpu_sums + other.gpu_sums,
            rare_values,
        }
    }
}

const DISCARD_BUCKET: usize = 0;
const AKA_DISCARD_BUCKET: usize = 1;
const RIICHI_BUCKET: usize = 2;
const CHI_BUCKET: usize = 3;
const PON_BUCKET: usize = 4;
const KAN_BUCKET: usize = 5;
const AGARI_BUCKET: usize = 6;
const RYUUKYOKU_BUCKET: usize = 7;
const PASS_BUCKET: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RareActionSums {
    counts: [usize; 9],
    correct: [usize; 9],
}

impl RareActionSums {
    fn record(&mut self, action: u8, predicted: u8) {
        let bucket = action_bucket(action);
        self.counts[bucket] += 1;
        if predicted == action {
            self.correct[bucket] += 1;
        }
    }
}

fn action_bucket(action: u8) -> usize {
    match action {
        DISCARD_START..=33 => DISCARD_BUCKET,
        AKA_5M..=AKA_5S => AKA_DISCARD_BUCKET,
        RIICHI => RIICHI_BUCKET,
        CHI_LEFT..=CHI_RIGHT => CHI_BUCKET,
        PON => PON_BUCKET,
        KAN => KAN_BUCKET,
        AGARI => AGARI_BUCKET,
        RYUUKYOKU => RYUUKYOKU_BUCKET,
        PASS => PASS_BUCKET,
        other => panic!("invalid Hydra action id for rare-action metrics: {other}"),
    }
}

/// Reads a rank-1 single-element tensor into an f64 scalar.
pub fn scalar1<B: Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor.clone().into_scalar().elem::<f64>()
}

/// Builds batch metric sums from model outputs and loss components.
pub fn batch_metric_sums_from_outputs<B: Backend>(
    sample_count: usize,
    policy_logits: Tensor<B, 2>,
    legal_mask: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    total_loss: Tensor<B, 1>,
    breakdown: &LossBreakdown<B>,
) -> BatchMetricSums<B> {
    let masked = policy_logits + (legal_mask.ones_like() - legal_mask) * (-1e9f32);
    let predicted_actions = masked.argmax(1).squeeze_dim::<1>(1);
    let rare_values = rare_action_metric_values_from_predictions(predicted_actions, actions);
    let sample_weight = sample_count as f32;

    let gpu_sums = Tensor::cat(
        vec![
            total_loss * sample_weight,
            breakdown.policy.clone() * sample_weight,
            breakdown.value.clone() * sample_weight,
            breakdown.grp.clone() * sample_weight,
            breakdown.tenpai.clone() * sample_weight,
            breakdown.danger.clone() * sample_weight,
            breakdown.opp_next.clone() * sample_weight,
            breakdown.score_pdf.clone() * sample_weight,
            breakdown.score_cdf.clone() * sample_weight,
        ],
        0,
    );

    BatchMetricSums {
        gpu_sums,
        rare_values,
    }
}

/// Converts accumulated metric sums into scalar batch stats.
pub fn batch_stats_from_metric_sums<B: Backend>(
    sample_count: usize,
    batch_count: usize,
    metric_sums: BatchMetricSums<B>,
) -> BatchStats {
    let metrics = metric_sums.gpu_sums.into_data().convert::<f32>();
    let values = metrics
        .as_slice::<f32>()
        .expect("profiling metrics should be readable as f32");
    let agreement = average_metric(metric_sums.rare_values[0], sample_count);

    BatchStats {
        sample_count,
        batch_count,
        total_loss: average_metric(values[0], sample_count),
        policy_agreement: agreement,
        loss_policy: average_metric(values[1], sample_count),
        loss_value: average_metric(values[2], sample_count),
        loss_grp: average_metric(values[3], sample_count),
        loss_tenpai: average_metric(values[4], sample_count),
        loss_danger: average_metric(values[5], sample_count),
        loss_opp_next: average_metric(values[6], sample_count),
        loss_score_pdf: average_metric(values[7], sample_count),
        loss_score_cdf: average_metric(values[8], sample_count),
        rare_actions: rare_metrics_from_values(&metric_sums.rare_values),
    }
}

/// Builds scalar stats directly from one batch of outputs.
pub fn batch_stats_from_outputs<B: Backend>(
    sample_count: usize,
    policy_logits: Tensor<B, 2>,
    legal_mask: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    total_loss: Tensor<B, 1>,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    batch_stats_from_metric_sums(
        sample_count,
        1,
        batch_metric_sums_from_outputs(
            sample_count,
            policy_logits,
            legal_mask,
            actions,
            total_loss,
            breakdown,
        ),
    )
}

fn average_metric(value_sum: f32, sample_count: usize) -> f64 {
    if sample_count == 0 {
        0.0
    } else {
        value_sum as f64 / sample_count as f64
    }
}

fn rare_action_metric_values_from_predictions<B: Backend>(
    predicted: Tensor<B, 1, Int>,
    actions: Tensor<B, 1, Int>,
) -> [f32; 19] {
    let predicted = predicted.into_data().convert::<i64>();
    let actions = actions.into_data().convert::<i64>();
    let predicted = predicted
        .as_slice::<i64>()
        .expect("predicted actions should be readable as i64");
    let actions = actions
        .as_slice::<i64>()
        .expect("target actions should be readable as i64");
    let mut sums = RareActionSums::default();
    let mut values = [0.0f32; 19];
    let mut overall_correct = 0usize;
    for (&action, &predicted) in actions.iter().zip(predicted.iter()) {
        if action == predicted {
            overall_correct += 1;
        }
        sums.record(action as u8, predicted as u8);
    }
    values[0] = overall_correct as f32;
    for idx in 0..9 {
        values[1 + idx * 2] = sums.counts[idx] as f32;
        values[1 + idx * 2 + 1] = sums.correct[idx] as f32;
    }
    values
}

fn rare_metrics_from_values(values: &[f32]) -> RareActionMetrics {
    let bucket = |idx: usize| {
        let count = values[1 + idx * 2] as usize;
        let correct = values[1 + idx * 2 + 1] as f64;
        RareActionBucketMetrics {
            count,
            accuracy: if count == 0 {
                0.0
            } else {
                correct / count as f64
            },
        }
    };
    RareActionMetrics {
        discard: bucket(DISCARD_BUCKET),
        aka_discard: bucket(AKA_DISCARD_BUCKET),
        riichi: bucket(RIICHI_BUCKET),
        chi: bucket(CHI_BUCKET),
        pon: bucket(PON_BUCKET),
        kan: bucket(KAN_BUCKET),
        agari: bucket(AGARI_BUCKET),
        ryuukyoku: bucket(RYUUKYOKU_BUCKET),
        pass: bucket(PASS_BUCKET),
    }
}

/// Builds scalar stats directly from a loss breakdown.
pub fn batch_stats_from_breakdown<B: Backend>(
    sample_count: usize,
    agreement: f64,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    let metrics = Tensor::cat(
        vec![
            breakdown.total.clone(),
            breakdown.policy.clone(),
            breakdown.value.clone(),
            breakdown.grp.clone(),
            breakdown.tenpai.clone(),
            breakdown.danger.clone(),
            breakdown.opp_next.clone(),
            breakdown.score_pdf.clone(),
            breakdown.score_cdf.clone(),
        ],
        0,
    )
    .into_data()
    .convert::<f32>();
    let values = metrics
        .as_slice::<f32>()
        .expect("breakdown scalars should be readable as f32");
    BatchStats {
        sample_count,
        batch_count: 1,
        total_loss: values[0] as f64,
        policy_agreement: agreement,
        loss_policy: values[1] as f64,
        loss_value: values[2] as f64,
        loss_grp: values[3] as f64,
        loss_tenpai: values[4] as f64,
        loss_danger: values[5] as f64,
        loss_opp_next: values[6] as f64,
        loss_score_pdf: values[7] as f64,
        loss_score_cdf: values[8] as f64,
        rare_actions: RareActionMetrics::default(),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::Int;
    use burn::tensor::TensorData;
    use hydra_train_types::losses::LossBreakdown;

    use super::{batch_stats_from_breakdown, batch_stats_from_outputs, scalar1};

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
}
