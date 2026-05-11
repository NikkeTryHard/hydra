//! Training progress DTOs shared by runtime executors.

use serde::Serialize;

use crate::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED, PROFILING_STAGE_H2D_STREAM_SYNC,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOSS,
    PROFILING_STAGE_METRIC_READBACK, PROFILING_STAGE_OPTIMIZER_STEP, PROFILING_STAGE_PRODUCER_WAIT,
    ProfilingEnvelope,
};

/// Scalar train sub-stage timings used to build profiling envelopes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainSubStageTiming {
    pub producer_wait_seconds: f64,
    pub collation_seconds: f64,
    pub h2d_transfer_seconds: f64,
    pub h2d_pageable_to_pinned_seconds: f64,
    pub h2d_tensor_materialize_seconds: f64,
    pub h2d_stream_sync_seconds: f64,
    pub forward_seconds: f64,
    pub loss_seconds: f64,
    pub backward_seconds: f64,
    pub metric_readback_seconds: f64,
    pub optimizer_step_seconds: f64,
}

impl TrainSubStageTiming {
    /// Adds another timing sample into this accumulator.
    pub fn accumulate(&mut self, other: &TrainSubStageTiming) {
        self.producer_wait_seconds += other.producer_wait_seconds;
        self.h2d_transfer_seconds += other.h2d_transfer_seconds;
        self.h2d_pageable_to_pinned_seconds += other.h2d_pageable_to_pinned_seconds;
        self.h2d_tensor_materialize_seconds += other.h2d_tensor_materialize_seconds;
        self.h2d_stream_sync_seconds += other.h2d_stream_sync_seconds;
        self.collation_seconds += other.collation_seconds;
        self.forward_seconds += other.forward_seconds;
        self.loss_seconds += other.loss_seconds;
        self.backward_seconds += other.backward_seconds;
        self.metric_readback_seconds += other.metric_readback_seconds;
        self.optimizer_step_seconds += other.optimizer_step_seconds;
    }

    /// Converts this scalar timing snapshot into profiling envelope children.
    pub fn to_profiling_children(&self) -> Vec<ProfilingEnvelope> {
        vec![
            ProfilingEnvelope::leaf(PROFILING_STAGE_PRODUCER_WAIT, self.producer_wait_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_COLLATION, self.collation_seconds),
            ProfilingEnvelope::nested(
                PROFILING_STAGE_H2D_TRANSFER,
                self.h2d_transfer_seconds,
                vec![
                    ProfilingEnvelope::leaf(
                        PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED,
                        self.h2d_pageable_to_pinned_seconds,
                    ),
                    ProfilingEnvelope::leaf(
                        PROFILING_STAGE_H2D_TENSOR_MATERIALIZE,
                        self.h2d_tensor_materialize_seconds,
                    ),
                    ProfilingEnvelope::leaf(
                        PROFILING_STAGE_H2D_STREAM_SYNC,
                        self.h2d_stream_sync_seconds,
                    ),
                ],
            ),
            ProfilingEnvelope::leaf(PROFILING_STAGE_FORWARD, self.forward_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_LOSS, self.loss_seconds),
            ProfilingEnvelope::leaf(
                PROFILING_STAGE_METRIC_READBACK,
                self.metric_readback_seconds,
            ),
            ProfilingEnvelope::leaf(PROFILING_STAGE_BACKWARD, self.backward_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_OPTIMIZER_STEP, self.optimizer_step_seconds),
        ]
    }
}

/// Sample-weighted scalar averages for training metrics.
#[derive(Default, Serialize, Clone, Copy)]
pub struct ScalarAverages {
    pub total_loss: f64,
    pub policy_agreement: f64,
    pub loss_policy: f64,
    pub loss_value: f64,
    pub loss_grp: f64,
    pub loss_tenpai: f64,
    pub loss_danger: f64,
    pub loss_opp_next: f64,
    pub loss_score_pdf: f64,
    pub loss_score_cdf: f64,
    pub num_samples: usize,
    pub num_batches: usize,
    pub rare_actions: RareActionMetrics,
}

/// Scalar stats for one logical batch.
#[derive(Clone, Copy, Default)]
pub struct BatchStats {
    pub sample_count: usize,
    pub batch_count: usize,
    pub total_loss: f64,
    pub policy_agreement: f64,
    pub loss_policy: f64,
    pub loss_value: f64,
    pub loss_grp: f64,
    pub loss_tenpai: f64,
    pub loss_danger: f64,
    pub loss_opp_next: f64,
    pub loss_score_pdf: f64,
    pub loss_score_cdf: f64,
    pub rare_actions: RareActionMetrics,
}

/// Epoch JSONL progress entry.
#[derive(Serialize)]
pub struct EpochLogEntry<DeltaQPromotionSnapshot, Advisory = String> {
    pub epoch: usize,
    pub global_step: usize,
    pub lr: f64,
    pub train_total_loss: f64,
    pub train_policy_agreement: f64,
    pub train_loss_policy: f64,
    pub train_loss_value: f64,
    pub train_loss_grp: f64,
    pub train_loss_tenpai: f64,
    pub train_loss_danger: f64,
    pub train_loss_opp_next: f64,
    pub train_loss_score_pdf: f64,
    pub train_loss_score_cdf: f64,
    pub train_rare_actions: RareActionMetrics,
    pub val_rare_actions: Option<RareActionMetrics>,
    pub val_total_loss: Option<f64>,
    pub val_policy_loss: Option<f64>,
    pub val_policy_agreement: Option<f64>,
    pub val_delta_q_promotion: Option<DeltaQPromotionSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingEnvelope>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<Advisory>,
    pub best_val_policy_loss: Option<f64>,
    pub best_val_agreement: Option<f64>,
    pub num_batches: usize,
}

/// Step JSONL progress entry.
#[derive(Serialize)]
pub struct StepLogEntry<DeltaQPromotionSnapshot, Advisory = String> {
    pub global_step: usize,
    pub epoch: usize,
    pub lr: f64,
    pub train_total_loss: f64,
    pub train_policy_agreement: f64,
    pub train_loss_policy: f64,
    pub train_loss_value: f64,
    pub train_loss_grp: f64,
    pub train_loss_tenpai: f64,
    pub train_loss_danger: f64,
    pub train_loss_opp_next: f64,
    pub train_loss_score_pdf: f64,
    pub train_loss_score_cdf: f64,
    pub train_rare_actions: RareActionMetrics,
    pub val_rare_actions: Option<RareActionMetrics>,
    pub val_total_loss: Option<f64>,
    pub val_policy_loss: Option<f64>,
    pub val_policy_agreement: Option<f64>,
    pub val_delta_q_promotion: Option<DeltaQPromotionSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingEnvelope>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<Advisory>,
    pub best_val_policy_loss: Option<f64>,
    pub best_val_agreement: Option<f64>,
}

/// RL JSONL progress entry.
#[derive(Serialize)]
pub struct RlStepLogEntry<Advisory = String> {
    pub global_step: usize,
    pub phase: String,
    pub loss: f64,
    pub effective_lr: f64,
    pub exit_weight: f32,
    pub games_per_batch: usize,
    pub samples_in_batch: usize,
    pub total_games: u64,
    pub total_samples: u64,
    pub delta_q_state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingEnvelope>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisories: Vec<Advisory>,
}

/// Startup banner counts.
pub struct BannerStats {
    pub total_sources: usize,
    pub total_games: usize,
    pub train_count: usize,
    pub val_count: usize,
    pub accum_steps: usize,
    pub counts_exact: bool,
}

/// Accuracy for one rare-action bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RareActionBucketMetrics {
    pub count: usize,
    pub accuracy: f64,
}

/// Accuracy by rare-action class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RareActionMetrics {
    pub discard: RareActionBucketMetrics,
    pub aka_discard: RareActionBucketMetrics,
    pub riichi: RareActionBucketMetrics,
    pub chi: RareActionBucketMetrics,
    pub pon: RareActionBucketMetrics,
    pub kan: RareActionBucketMetrics,
    pub agari: RareActionBucketMetrics,
    pub ryuukyoku: RareActionBucketMetrics,
    pub pass: RareActionBucketMetrics,
}

impl ScalarAverages {
    /// Records one batch into this sample-weighted accumulator.
    pub fn record_batch(&mut self, batch: BatchStats) {
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
        self.num_batches += batch.batch_count.max(1);
        self.rare_actions = merge_rare_metrics(self.rare_actions, batch.rare_actions);
    }

    /// Finalizes weighted sums into averages.
    #[must_use]
    pub fn finalize(mut self) -> Self {
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
        self.rare_actions = finalize_rare_metrics(self.rare_actions);
        self
    }
}

fn merge_rare_metrics(mut lhs: RareActionMetrics, rhs: RareActionMetrics) -> RareActionMetrics {
    fn merge_bucket(lhs: &mut RareActionBucketMetrics, rhs: RareActionBucketMetrics) {
        lhs.count += rhs.count;
        lhs.accuracy += rhs.accuracy * rhs.count as f64;
    }
    merge_bucket(&mut lhs.discard, rhs.discard);
    merge_bucket(&mut lhs.aka_discard, rhs.aka_discard);
    merge_bucket(&mut lhs.riichi, rhs.riichi);
    merge_bucket(&mut lhs.chi, rhs.chi);
    merge_bucket(&mut lhs.pon, rhs.pon);
    merge_bucket(&mut lhs.kan, rhs.kan);
    merge_bucket(&mut lhs.agari, rhs.agari);
    merge_bucket(&mut lhs.ryuukyoku, rhs.ryuukyoku);
    merge_bucket(&mut lhs.pass, rhs.pass);
    lhs
}

fn finalize_rare_metrics(mut metrics: RareActionMetrics) -> RareActionMetrics {
    fn finalize_bucket(bucket: &mut RareActionBucketMetrics) {
        if bucket.count == 0 {
            bucket.accuracy = 0.0;
        } else {
            bucket.accuracy /= bucket.count as f64;
        }
    }
    finalize_bucket(&mut metrics.discard);
    finalize_bucket(&mut metrics.aka_discard);
    finalize_bucket(&mut metrics.riichi);
    finalize_bucket(&mut metrics.chi);
    finalize_bucket(&mut metrics.pon);
    finalize_bucket(&mut metrics.kan);
    finalize_bucket(&mut metrics.agari);
    finalize_bucket(&mut metrics.ryuukyoku);
    finalize_bucket(&mut metrics.pass);
    metrics
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        BannerStats, BatchStats, EpochLogEntry, RareActionMetrics, RlStepLogEntry, ScalarAverages,
        StepLogEntry,
    };
    use crate::preflight::ProfilingEnvelope;

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
    fn log_entries_and_banner_stats_cover_data_fields() {
        let epoch = EpochLogEntry::<(), String> {
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
        let step = StepLogEntry::<(), String> {
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
        let rl = RlStepLogEntry::<String> {
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
        let banner = BannerStats {
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
}
