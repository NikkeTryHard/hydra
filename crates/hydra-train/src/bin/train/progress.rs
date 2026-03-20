use super::validation::DeltaQPromotionSnapshot;
use burn::prelude::*;

use hydra_train::training::losses::LossBreakdown;

#[derive(Default, serde::Serialize, Clone, Copy)]
pub(super) struct ScalarAverages {
    pub(super) total_loss: f64,
    pub(super) policy_agreement: f64,
    pub(super) loss_policy: f64,
    pub(super) loss_value: f64,
    pub(super) loss_grp: f64,
    pub(super) loss_tenpai: f64,
    pub(super) loss_danger: f64,
    pub(super) loss_opp_next: f64,
    pub(super) loss_score_pdf: f64,
    pub(super) loss_score_cdf: f64,
    pub(super) num_samples: usize,
    pub(super) num_batches: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct BatchStats {
    pub(super) sample_count: usize,
    pub(super) total_loss: f64,
    pub(super) policy_agreement: f64,
    pub(super) loss_policy: f64,
    pub(super) loss_value: f64,
    pub(super) loss_grp: f64,
    pub(super) loss_tenpai: f64,
    pub(super) loss_danger: f64,
    pub(super) loss_opp_next: f64,
    pub(super) loss_score_pdf: f64,
    pub(super) loss_score_cdf: f64,
}

#[derive(serde::Serialize)]
pub(super) struct EpochLogEntry {
    pub(super) epoch: usize,
    pub(super) global_step: usize,
    pub(super) lr: f64,
    pub(super) train_total_loss: f64,
    pub(super) train_policy_agreement: f64,
    pub(super) train_loss_policy: f64,
    pub(super) train_loss_value: f64,
    pub(super) train_loss_grp: f64,
    pub(super) train_loss_tenpai: f64,
    pub(super) train_loss_danger: f64,
    pub(super) train_loss_opp_next: f64,
    pub(super) train_loss_score_pdf: f64,
    pub(super) train_loss_score_cdf: f64,
    pub(super) val_total_loss: Option<f64>,
    pub(super) val_policy_loss: Option<f64>,
    pub(super) val_policy_agreement: Option<f64>,
    pub(super) val_delta_q_promotion: Option<DeltaQPromotionSnapshot>,
    pub(super) best_val_policy_loss: Option<f64>,
    pub(super) best_val_agreement: Option<f64>,
    pub(super) num_batches: usize,
}

#[derive(serde::Serialize)]
pub(super) struct StepLogEntry {
    pub(super) global_step: usize,
    pub(super) epoch: usize,
    pub(super) lr: f64,
    pub(super) train_total_loss: f64,
    pub(super) train_policy_agreement: f64,
    pub(super) train_loss_policy: f64,
    pub(super) train_loss_value: f64,
    pub(super) train_loss_grp: f64,
    pub(super) train_loss_tenpai: f64,
    pub(super) train_loss_danger: f64,
    pub(super) train_loss_opp_next: f64,
    pub(super) train_loss_score_pdf: f64,
    pub(super) train_loss_score_cdf: f64,
    pub(super) val_total_loss: Option<f64>,
    pub(super) val_policy_loss: Option<f64>,
    pub(super) val_policy_agreement: Option<f64>,
    pub(super) val_delta_q_promotion: Option<DeltaQPromotionSnapshot>,
    pub(super) best_val_policy_loss: Option<f64>,
    pub(super) best_val_agreement: Option<f64>,
}

#[derive(serde::Serialize)]
pub(super) struct RlStepLogEntry {
    pub(super) global_step: usize,
    pub(super) phase: String,
    pub(super) loss: f64,
    pub(super) effective_lr: f64,
    pub(super) exit_weight: f32,
    pub(super) games_per_batch: usize,
    pub(super) samples_in_batch: usize,
    pub(super) total_games: u64,
    pub(super) total_samples: u64,
    pub(super) delta_q_state: String,
}

pub(super) struct BannerStats {
    pub(super) total_sources: usize,
    pub(super) total_games: usize,
    pub(super) train_count: usize,
    pub(super) val_count: usize,
    pub(super) accum_steps: usize,
    pub(super) counts_exact: bool,
}

pub(super) fn scalar1<B: Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor.clone().into_scalar().elem::<f64>()
}

impl ScalarAverages {
    pub(super) fn record_batch(&mut self, batch: BatchStats) {
        let weight = batch.sample_count as f64;
        if weight <= f64::EPSILON {
            return;
        }
        self.total_loss += batch.total_loss * weight;
        self.policy_agreement += batch.policy_agreement * weight;
        self.loss_policy += batch.loss_policy * weight;
        self.loss_value += batch.loss_value * weight;
        self.loss_grp += batch.loss_grp * weight;
        self.loss_tenpai += batch.loss_tenpai * weight;
        self.loss_danger += batch.loss_danger * weight;
        self.loss_opp_next += batch.loss_opp_next * weight;
        self.loss_score_pdf += batch.loss_score_pdf * weight;
        self.loss_score_cdf += batch.loss_score_cdf * weight;
        self.num_samples += batch.sample_count;
        self.num_batches += 1;
    }

    pub(super) fn finalize(mut self) -> Self {
        if self.num_samples == 0 {
            return self;
        }
        let denom = self.num_samples as f64;
        self.total_loss /= denom;
        self.policy_agreement /= denom;
        self.loss_policy /= denom;
        self.loss_value /= denom;
        self.loss_grp /= denom;
        self.loss_tenpai /= denom;
        self.loss_danger /= denom;
        self.loss_opp_next /= denom;
        self.loss_score_pdf /= denom;
        self.loss_score_cdf /= denom;
        self
    }
}

pub(super) fn batch_stats_from_breakdown<B: Backend>(
    sample_count: usize,
    agreement: f64,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    BatchStats {
        sample_count,
        total_loss: scalar1(&breakdown.total),
        policy_agreement: agreement,
        loss_policy: scalar1(&breakdown.policy),
        loss_value: scalar1(&breakdown.value),
        loss_grp: scalar1(&breakdown.grp),
        loss_tenpai: scalar1(&breakdown.tenpai),
        loss_danger: scalar1(&breakdown.danger),
        loss_opp_next: scalar1(&breakdown.opp_next),
        loss_score_pdf: scalar1(&breakdown.score_pdf),
        loss_score_cdf: scalar1(&breakdown.score_cdf),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use hydra_train::training::losses::LossBreakdown;
    use serde_json::Value;

    use super::{batch_stats_from_breakdown, scalar1, BatchStats, ScalarAverages};

    fn batch(sample_count: usize, total_loss: f64, agreement: f64) -> BatchStats {
        BatchStats {
            sample_count,
            total_loss,
            policy_agreement: agreement,
            ..Default::default()
        }
    }

    #[test]
    fn scalar_averages_are_sample_weighted() {
        let mut stats = ScalarAverages::default();
        stats.record_batch(batch(4, 1.0, 0.25));
        stats.record_batch(batch(1, 4.0, 1.0));

        let stats = stats.finalize();
        assert_eq!(stats.num_batches, 2);
        assert_eq!(stats.num_samples, 5);
        assert!((stats.total_loss - 1.6).abs() < 1e-12);
        assert!((stats.policy_agreement - 0.4).abs() < 1e-12);
    }

    #[test]
    fn zero_weight_batches_do_not_change_averages() {
        let mut stats = ScalarAverages::default();
        stats.record_batch(batch(0, 7.0, 0.9));
        stats.record_batch(batch(2, 3.0, 0.5));

        let stats = stats.finalize();
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.num_samples, 2);
        assert!((stats.total_loss - 3.0).abs() < 1e-12);
        assert!((stats.policy_agreement - 0.5).abs() < 1e-12);
    }

    #[test]
    fn finalize_leaves_empty_accumulator_unchanged() {
        let stats = ScalarAverages::default().finalize();
        assert_eq!(stats.num_batches, 0);
        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(stats.loss_score_cdf, 0.0);
    }

    #[test]
    fn record_batch_weights_all_loss_fields() {
        let mut stats = ScalarAverages::default();
        stats.record_batch(BatchStats {
            sample_count: 2,
            total_loss: 5.0,
            policy_agreement: 0.25,
            loss_policy: 1.0,
            loss_value: 2.0,
            loss_grp: 3.0,
            loss_tenpai: 4.0,
            loss_danger: 5.0,
            loss_opp_next: 6.0,
            loss_score_pdf: 7.0,
            loss_score_cdf: 8.0,
        });

        let stats = stats.finalize();
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.num_samples, 2);
        assert_eq!(stats.loss_policy, 1.0);
        assert_eq!(stats.loss_value, 2.0);
        assert_eq!(stats.loss_grp, 3.0);
        assert_eq!(stats.loss_tenpai, 4.0);
        assert_eq!(stats.loss_danger, 5.0);
        assert_eq!(stats.loss_opp_next, 6.0);
        assert_eq!(stats.loss_score_pdf, 7.0);
        assert_eq!(stats.loss_score_cdf, 8.0);
    }

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
    fn log_entries_and_banner_stats_cover_data_fields() {
        let epoch = super::EpochLogEntry {
            epoch: 2,
            global_step: 42,
            lr: 1e-3,
            train_total_loss: 1.0,
            train_policy_agreement: 0.5,
            train_loss_policy: 0.1,
            train_loss_value: 0.2,
            train_loss_grp: 0.3,
            train_loss_tenpai: 0.4,
            train_loss_danger: 0.5,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            val_total_loss: Some(1.5),
            val_policy_loss: Some(1.25),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            best_val_policy_loss: Some(1.0),
            best_val_agreement: Some(0.8),
            num_batches: 3,
        };
        let step = super::StepLogEntry {
            global_step: 42,
            epoch: 2,
            lr: 1e-3,
            train_total_loss: 1.0,
            train_policy_agreement: 0.5,
            train_loss_policy: 0.1,
            train_loss_value: 0.2,
            train_loss_grp: 0.3,
            train_loss_tenpai: 0.4,
            train_loss_danger: 0.5,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            val_total_loss: Some(1.5),
            val_policy_loss: Some(1.25),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            best_val_policy_loss: Some(1.0),
            best_val_agreement: Some(0.8),
        };
        let rl = super::RlStepLogEntry {
            global_step: 5,
            phase: "ExitPondering".to_string(),
            loss: 0.25,
            effective_lr: 5e-4,
            exit_weight: 0.1,
            games_per_batch: 8,
            samples_in_batch: 64,
            total_games: 128,
            total_samples: 1024,
            delta_q_state: "Warmup".to_string(),
        };
        let banner = super::BannerStats {
            total_sources: 2,
            total_games: 30,
            train_count: 24,
            val_count: 6,
            accum_steps: 4,
            counts_exact: true,
        };

        let epoch_json = serde_json::to_value(epoch).expect("epoch log should serialize");
        let step_json = serde_json::to_value(step).expect("step log should serialize");
        let rl_json = serde_json::to_value(rl).expect("rl log should serialize");

        assert_eq!(epoch_json["global_step"], Value::from(42));
        assert_eq!(epoch_json["num_batches"], Value::from(3));
        assert_eq!(step_json["epoch"], Value::from(2));
        assert_eq!(rl_json["phase"], Value::from("ExitPondering"));
        assert_eq!(rl_json["total_samples"], Value::from(1024));

        assert_eq!(banner.total_sources, 2);
        assert_eq!(banner.total_games, 30);
        assert_eq!(banner.train_count, 24);
        assert_eq!(banner.val_count, 6);
        assert_eq!(banner.accum_steps, 4);
        assert!(banner.counts_exact);
    }

    fn scalar_tensor<B: burn::tensor::backend::Backend>(value: f32) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::<B, 1>::from_data(TensorData::from([value]), &Default::default())
    }
}
