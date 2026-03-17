use burn::module::AutodiffModule;
use burn::prelude::*;
use indicatif::ProgressBar;

use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_val_pass};
use hydra_train::data::sample::collate_batch_samples;
use hydra_train::model::{HydraModel, HydraOutput};
use hydra_train::training::bc::{
    gated_bc_context, BcExitConfig, bc_total_with_exit, policy_agreement,
    target_actions_from_policy_target,
};
use hydra_train::training::delta_q_promotion::{
    DeltaQPolicyTransferReport, DeltaQPolicyTransferResult, DeltaQPolicyTransferThresholds,
    DeltaQPromotionReport, DeltaQPromotionResult, DeltaQPromotionThresholds,
    collect_policy_transfer_metrics_from_policy_outputs, collect_promotion_metrics_from_outputs,
    evaluate_policy_transfer_report, evaluate_promotion_report,
};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::{HydraLoss, HydraTargets};

use super::config::{TrainConfig, validation_microbatch_size, validation_sample_limit};
use super::progress::{BatchStats, ScalarAverages, batch_stats_from_breakdown};
use super::resume::BestValidation;
use super::{TrainBackend, ValidBackend};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct DeltaQPromotionSnapshot {
    pub(super) compared_states: u64,
    pub(super) candidate_top1_agreement: f64,
    pub(super) candidate_mean_regret: f64,
    pub(super) baseline_mean_regret: f64,
    pub(super) mean_decision_lift: f64,
    pub(super) negative_lift_fraction: f64,
    pub(super) regret_beats_baseline_rate: f64,
    pub(super) top1_beats_baseline_rate: f64,
    pub(super) passed: bool,
}

impl DeltaQPromotionSnapshot {
    fn from_report(
        report: &DeltaQPromotionReport,
        result: &DeltaQPromotionResult,
    ) -> Self {
        Self {
            compared_states: report.compared_states,
            candidate_top1_agreement: report.candidate_top1_agreement(),
            candidate_mean_regret: report.candidate_mean_regret(),
            baseline_mean_regret: report.baseline_mean_regret(),
            mean_decision_lift: report.mean_decision_lift(),
            negative_lift_fraction: report.negative_lift_fraction(),
            regret_beats_baseline_rate: report.candidate_regret_beats_baseline_rate(),
            top1_beats_baseline_rate: report.candidate_top1_beats_baseline_rate(),
            passed: result.passed,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct DeltaQPolicyTransferSnapshot {
    pub(super) compared_states: u64,
    pub(super) candidate_policy_top1_to_teacher: f64,
    pub(super) baseline_policy_top1_to_teacher: f64,
    pub(super) candidate_policy_mean_teacher_regret: f64,
    pub(super) baseline_policy_mean_teacher_regret: f64,
    pub(super) candidate_beats_baseline_rate: f64,
    pub(super) negative_transfer_fraction: f64,
}

impl DeltaQPolicyTransferSnapshot {
    fn from_report(report: &DeltaQPolicyTransferReport) -> Self {
        Self {
            compared_states: report.compared_states,
            candidate_policy_top1_to_teacher: report.candidate_policy_top1_to_teacher(),
            baseline_policy_top1_to_teacher: report.baseline_policy_top1_to_teacher(),
            candidate_policy_mean_teacher_regret: report.candidate_policy_mean_teacher_regret(),
            baseline_policy_mean_teacher_regret: report.baseline_policy_mean_teacher_regret(),
            candidate_beats_baseline_rate: report.candidate_beats_baseline_rate(),
            negative_transfer_fraction: report.negative_transfer_fraction(),
        }
    }
}

#[derive(Clone)]
pub(super) struct ValidationSummary {
    pub(super) total_loss: f64,
    pub(super) policy_loss: f64,
    pub(super) agreement: f64,
    pub(super) samples: usize,
    pub(super) delta_q_promotion: Option<DeltaQPromotionReport>,
    pub(super) delta_q_promotion_result: Option<DeltaQPromotionResult>,
    pub(super) delta_q_promotion_snapshot: Option<DeltaQPromotionSnapshot>,
    pub(super) delta_q_policy_transfer: Option<DeltaQPolicyTransferReport>,
    pub(super) delta_q_policy_transfer_result: Option<DeltaQPolicyTransferResult>,
    pub(super) delta_q_policy_transfer_snapshot: Option<DeltaQPolicyTransferSnapshot>,
}

pub(super) fn validation_batch_stats<B: Backend>(
    sample_count: usize,
    output: &HydraOutput<B>,
    batch: &hydra_train::data::sample::MjaiBatch<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    exit_cfg: &BcExitConfig,
) -> BatchStats {
    let target_actions = target_actions_from_policy_target(targets.policy_target.clone());
    let agreement = policy_agreement(
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        target_actions,
    );
    let breakdown = loss_fn.total_loss(output, targets);
    let mut stats = batch_stats_from_breakdown(sample_count, agreement, &breakdown);
    stats.total_loss = bc_total_with_exit(output, batch, targets, loss_fn, exit_cfg)
        .into_scalar()
        .elem::<f64>();
    stats
}

pub(super) fn is_better_validation(
    summary: &ValidationSummary,
    best: Option<BestValidation>,
) -> bool {
    match best {
        None => true,
        Some(best) => {
            summary.policy_loss < best.policy_loss
                || ((summary.policy_loss - best.policy_loss).abs() <= f64::EPSILON
                    && summary.agreement > best.agreement)
        }
    }
}

pub(super) fn run_validation(
    model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    device: &<ValidBackend as Backend>::Device,
    loss_fn: &HydraLoss<ValidBackend>,
    exit_cfg: &BcExitConfig,
    head_controller: Option<&mut HeadActivationController>,
    progress: Option<&ProgressBar>,
) -> Result<ValidationSummary, String> {
    run_validation_with_policy_baseline(
        model,
        model,
        config,
        loader_config,
        manifest,
        device,
        loss_fn,
        exit_cfg,
        head_controller,
        progress,
    )
}

pub(super) fn run_validation_with_policy_baseline(
    model: &HydraModel<TrainBackend>,
    baseline_model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    device: &<ValidBackend as Backend>::Device,
    loss_fn: &HydraLoss<ValidBackend>,
    exit_cfg: &BcExitConfig,
    head_controller: Option<&mut HeadActivationController>,
    progress: Option<&ProgressBar>,
) -> Result<ValidationSummary, String> {
    let model_valid = model.valid();
    let baseline_valid = baseline_model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut stats = ScalarAverages::default();
    let mut total_samples = 0usize;
    let mut head_controller = head_controller;
    let mut delta_q_promotion = DeltaQPromotionReport::new();
    let mut delta_q_policy_transfer = DeltaQPolicyTransferReport::new();
    let mut saw_delta_q_targets = false;

    for buffer_result in stream_val_pass(manifest, loader_config, progress) {
        let buffer = buffer_result.map_err(|err| format!("validation stream failed: {err}"))?;
        for chunk in buffer.chunks(validation_batch_size) {
            if let Some(limit) = validation_sample_limit
                && total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk
            };
            if capped_chunk.is_empty() {
                break;
            }
            let Some((obs, batch)) =
                collate_batch_samples::<ValidBackend>(capped_chunk, false, device)
            else {
                continue;
            };
            let targets = batch.to_hydra_targets();
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(head_controller.as_deref_mut(), loss_fn, &targets);
            let output =
                model_valid.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
            let baseline_output = baseline_valid.forward(obs);
            let batch_stats = validation_batch_stats(
                capped_chunk.len(),
                &output,
                &batch,
                &targets,
                &active_loss_fn,
                exit_cfg,
            );
            if targets.delta_q_target.is_some() && targets.delta_q_mask.is_some() {
                delta_q_promotion.merge(&collect_promotion_metrics_from_outputs(
                    &output, &targets, 0.75,
                ));
                delta_q_policy_transfer.merge(&collect_policy_transfer_metrics_from_policy_outputs(
                    output.policy_logits.clone(),
                    baseline_output.policy_logits.clone(),
                    &targets,
                ));
                saw_delta_q_targets = true;
            }
            stats.record_batch(batch_stats);
            total_samples += capped_chunk.len();
        }
        if let Some(limit) = validation_sample_limit
            && total_samples >= limit
        {
            break;
        }
    }

    if total_samples == 0 {
        Ok(ValidationSummary {
            total_loss: 0.0,
            policy_loss: 0.0,
            agreement: 0.0,
            samples: 0,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        })
    } else {
        let stats = stats.finalize();
        let (
            delta_q_promotion,
            delta_q_promotion_result,
            delta_q_promotion_snapshot,
            delta_q_policy_transfer,
            delta_q_policy_transfer_result,
            delta_q_policy_transfer_snapshot,
        ) =
            if saw_delta_q_targets {
                let result = evaluate_promotion_report(
                    &delta_q_promotion,
                    &DeltaQPromotionThresholds::default(),
                );
                let policy_transfer_result = evaluate_policy_transfer_report(
                    &delta_q_policy_transfer,
                    &DeltaQPolicyTransferThresholds::default(),
                );
                let snapshot = DeltaQPromotionSnapshot::from_report(&delta_q_promotion, &result);
                let policy_transfer_snapshot =
                    DeltaQPolicyTransferSnapshot::from_report(&delta_q_policy_transfer);
                (
                    Some(delta_q_promotion),
                    Some(result),
                    Some(snapshot),
                    Some(delta_q_policy_transfer),
                    Some(policy_transfer_result),
                    Some(policy_transfer_snapshot),
                )
            } else {
                (None, None, None, None, None, None)
            };
        Ok(ValidationSummary {
            total_loss: stats.total_loss,
            policy_loss: stats.loss_policy,
            agreement: stats.policy_agreement,
            samples: total_samples,
            delta_q_promotion,
            delta_q_promotion_result,
            delta_q_promotion_snapshot,
            delta_q_policy_transfer,
            delta_q_policy_transfer_result,
            delta_q_policy_transfer_snapshot,
        })
    }
}
