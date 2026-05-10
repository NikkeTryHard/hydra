#![allow(
    missing_docs,
    reason = "moved train progress DTOs preserve existing public surface"
)]

use hydra_train_runtime::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED, PROFILING_STAGE_H2D_STREAM_SYNC,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOSS,
    PROFILING_STAGE_METRIC_READBACK, PROFILING_STAGE_OPTIMIZER_STEP, PROFILING_STAGE_PRODUCER_WAIT,
    ProfilingEnvelope,
};
use serde::Serialize;

use crate::advisory::RuntimeAdvisory;

/// Scalar train sub-stage timings used to build profiling envelopes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainSubStageTiming {
    /// Time spent waiting for producer batches, in seconds.
    pub producer_wait_seconds: f64,
    /// Time spent collating CPU samples, in seconds.
    pub collation_seconds: f64,
    /// Total host-to-device transfer wall time, in seconds.
    pub h2d_transfer_seconds: f64,
    /// Time spent copying pageable host memory to pinned staging, in seconds.
    pub h2d_pageable_to_pinned_seconds: f64,
    /// Time spent materializing device tensors, in seconds.
    pub h2d_tensor_materialize_seconds: f64,
    /// Time spent synchronizing H2D streams, in seconds.
    pub h2d_stream_sync_seconds: f64,
    /// Forward-pass time, in seconds.
    pub forward_seconds: f64,
    /// Loss computation time, in seconds.
    pub loss_seconds: f64,
    /// Backward-pass time, in seconds.
    pub backward_seconds: f64,
    /// Metric readback time, in seconds.
    pub metric_readback_seconds: f64,
    /// Optimizer step time, in seconds.
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

#[derive(Serialize)]
pub struct EpochLogEntry<DeltaQPromotionSnapshot> {
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
    pub advisories: Vec<RuntimeAdvisory>,
    pub best_val_policy_loss: Option<f64>,
    pub best_val_agreement: Option<f64>,
    pub num_batches: usize,
}

#[derive(Serialize)]
pub struct StepLogEntry<DeltaQPromotionSnapshot> {
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
    pub advisories: Vec<RuntimeAdvisory>,
    pub best_val_policy_loss: Option<f64>,
    pub best_val_agreement: Option<f64>,
}

#[derive(Serialize)]
pub struct RlStepLogEntry {
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
    pub advisories: Vec<RuntimeAdvisory>,
}

pub struct BannerStats {
    pub total_sources: usize,
    pub total_games: usize,
    pub train_count: usize,
    pub val_count: usize,
    pub accum_steps: usize,
    pub counts_exact: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RareActionBucketMetrics {
    pub count: usize,
    pub accuracy: f64,
}

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
