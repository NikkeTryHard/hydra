use burn::prelude::*;

use hydra_train::training::losses::LossBreakdown;

#[derive(Default, serde::Serialize)]
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
    pub(super) num_batches: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct BatchStats {
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
    pub(super) best_val_policy_loss: Option<f64>,
    pub(super) best_val_agreement: Option<f64>,
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
        self.total_loss += batch.total_loss;
        self.policy_agreement += batch.policy_agreement;
        self.loss_policy += batch.loss_policy;
        self.loss_value += batch.loss_value;
        self.loss_grp += batch.loss_grp;
        self.loss_tenpai += batch.loss_tenpai;
        self.loss_danger += batch.loss_danger;
        self.loss_opp_next += batch.loss_opp_next;
        self.loss_score_pdf += batch.loss_score_pdf;
        self.loss_score_cdf += batch.loss_score_cdf;
        self.num_batches += 1;
    }

    pub(super) fn finalize(mut self) -> Self {
        if self.num_batches == 0 {
            return self;
        }
        let denom = self.num_batches as f64;
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
    agreement: f64,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    BatchStats {
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
