pub(super) use hydra_train_exec::progress::BannerStats;
#[cfg(test)]
pub(super) use hydra_train_exec::progress::{BatchStats, ScalarAverages};
#[cfg(test)]
pub(super) type EpochLogEntry = hydra_train_exec::progress::EpochLogEntry<
    hydra_train_exec::validation::DeltaQPromotionSnapshot,
    hydra_train_exec::advisory::RuntimeAdvisory,
>;
#[cfg(test)]
pub(super) type StepLogEntry = hydra_train_exec::progress::StepLogEntry<
    hydra_train_exec::validation::DeltaQPromotionSnapshot,
    hydra_train_exec::advisory::RuntimeAdvisory,
>;
#[cfg(test)]
pub(super) type RlStepLogEntry =
    hydra_train_exec::progress::RlStepLogEntry<hydra_train_exec::advisory::RuntimeAdvisory>;
#[cfg(test)]
pub(super) use hydra_train_exec::bc_metrics::scalar1;
#[cfg(test)]
pub(super) use hydra_train_exec::bc_metrics::{
    batch_stats_from_breakdown, batch_stats_from_outputs,
};
#[cfg(test)]
pub(super) use hydra_train_exec::progress::RareActionMetrics;

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::Int;
    use burn::tensor::TensorData;
    use hydra_train::preflight::ProfilingEnvelope;
    use hydra_train::training::losses::LossBreakdown;
    use serde_json::Value;

    use super::{
        BatchStats, RareActionMetrics, ScalarAverages, batch_stats_from_breakdown,
        batch_stats_from_outputs, scalar1,
    };

    fn batch(sample_count: usize, total_loss: f64, agreement: f64) -> BatchStats {
        BatchStats {
            sample_count,
            batch_count: 1,
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
            batch_count: 1,
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
            rare_actions: RareActionMetrics::default(),
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
            train_rare_actions: RareActionMetrics::default(),
            val_total_loss: Some(1.5),
            val_policy_loss: Some(1.25),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            val_rare_actions: None,
            profiling: Some(ProfilingEnvelope::leaf("bc_epoch", 1.25)),
            advisories: Vec::new(),
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
            train_rare_actions: RareActionMetrics::default(),
            val_total_loss: Some(1.5),
            val_policy_loss: Some(1.25),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            val_rare_actions: None,
            profiling: Some(ProfilingEnvelope::leaf("bc_interval", 0.5)),
            advisories: Vec::new(),
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
            profiling: Some(ProfilingEnvelope::leaf("rl_step", 0.75)),
            advisories: Vec::new(),
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
        assert_eq!(epoch_json["profiling"]["stage"], Value::from("bc_epoch"));
        assert_eq!(step_json["epoch"], Value::from(2));
        assert_eq!(step_json["profiling"]["stage"], Value::from("bc_interval"));
        assert_eq!(rl_json["phase"], Value::from("ExitPondering"));
        assert_eq!(rl_json["total_samples"], Value::from(1024));
        assert_eq!(rl_json["profiling"]["stage"], Value::from("rl_step"));

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
