//! Full RL training step: DRDA-wrapped ACH with auxiliary losses.
#![allow(
    missing_docs,
    reason = "compatibility wrapper moved from hydra-train during RL seam cutover"
)]

use crate::losses::HydraLoss;
use crate::model::HydraTrainModelExt;
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use hydra_model::model::HydraModel;
use hydra_search_labels::exit::exit_loss;
use hydra_train_algo::ach::ach_policy_loss;
use hydra_train_algo::drda;

use hydra_train_types::config::RlConfig;
use hydra_train_types::head_gates::{
    AdvancedHead, HeadActivationController, borrow_or_extract_target_presence,
};
use hydra_train_types::losses::{HydraLossConfig, HydraTargets};
use hydra_train_types::orchestrator::PhaseTrainReport;
use hydra_train_types::phase::{PipelineState, TrainingPhase};
pub use hydra_train_types::rl::RlBatch;

pub const MAX_RL_BATCH_SIZE: usize = 512;
pub const ONE_EPOCH_ONLY: bool = true;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeltaQBatchStats {
    pub examples_present: usize,
    pub actions_present: usize,
    pub examples_absent: usize,
}

impl DeltaQBatchStats {
    pub fn from_targets<B: Backend>(targets: &HydraTargets<B>) -> Result<Self, &'static str> {
        match (&targets.delta_q_target, &targets.delta_q_mask) {
            (None, None) => Ok(Self {
                examples_present: 0,
                actions_present: 0,
                examples_absent: targets.policy_target.dims()[0],
            }),
            (Some(_), Some(_)) if targets.target_presence.is_some() => {
                let presence = targets
                    .target_presence
                    .as_ref()
                    .expect("checked target presence");
                let examples_present = presence.count(AdvancedHead::DeltaQ);
                Ok(Self {
                    examples_present,
                    actions_present: presence.delta_q_actions_present,
                    examples_absent: presence.batch_size.saturating_sub(examples_present),
                })
            }
            (Some(_), Some(mask)) => {
                let [batch, cols] = mask.dims();
                let mask_data = mask.to_data().convert::<f32>();
                let data = mask_data
                    .as_slice::<f32>()
                    .map_err(|_| "delta_q mask unreadable")?;
                let mut examples_present = 0usize;
                let mut actions_present = 0usize;
                for row in data.chunks(cols).take(batch) {
                    let row_actions = row.iter().filter(|&&v| v > 0.0).count();
                    if row_actions > 0 {
                        examples_present += 1;
                        actions_present += row_actions;
                    }
                }
                Ok(Self {
                    examples_present,
                    actions_present,
                    examples_absent: batch - examples_present,
                })
            }
            _ => Err("delta_q target/mask mismatch"),
        }
    }
}

pub fn validate_optional_target_pairs<B: Backend>(
    targets: &HydraTargets<B>,
) -> Result<(), &'static str> {
    match (&targets.delta_q_target, &targets.delta_q_mask) {
        (None, None) | (Some(_), Some(_)) => Ok(()),
        _ => Err("delta_q target/mask mismatch"),
    }
}

pub fn apply_head_gating_to_batch<B: Backend>(
    controller: &mut HeadActivationController,
    base_loss: &HydraLossConfig,
    targets: &HydraTargets<B>,
) -> Result<(HydraLossConfig, DeltaQBatchStats), &'static str> {
    validate_optional_target_pairs(targets)?;
    let presence = borrow_or_extract_target_presence(targets);
    controller.record_batch(&presence);
    let delta_q_stats = DeltaQBatchStats::from_targets(targets)?;
    let effective_loss = controller.approved_loss_config(base_loss);
    Ok((effective_loss, delta_q_stats))
}
pub struct RlPhaseTrainRequest<'a, B: Backend> {
    pub state: &'a PipelineState,
    pub batch: &'a RlBatch<B>,
    pub cfg: &'a RlConfig,
    pub loss_fn: &'a HydraLoss<B>,
    pub controller: Option<&'a mut HeadActivationController>,
}

pub struct RlStepRequest<'a, B: Backend> {
    pub batch: &'a RlBatch<B>,
    pub cfg: &'a RlConfig,
    pub phase: u8,
    pub progress: f32,
    pub loss_fn: &'a HydraLoss<B>,
    pub controller: Option<&'a mut HeadActivationController>,
}

struct RlFastPathRequest<'a, B: Backend> {
    batch: &'a RlBatch<B>,
    advantages_normed: &'a Tensor<B, 1>,
    cfg: &'a RlConfig,
    phase: u8,
    progress: f32,
    loss_fn: &'a HydraLoss<B>,
}

pub fn rl_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    rl_step_with_stage_progress_and_controller(
        model,
        RlStepRequest {
            batch,
            cfg,
            phase: 3,
            progress: 1.0,
            loss_fn,
            controller: None,
        },
        optimizer,
    )
}

pub fn rl_step_with_stage_progress<B: AutodiffBackend>(
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    phase: u8,
    progress: f32,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    rl_step_with_stage_progress_and_controller(
        model,
        RlStepRequest {
            batch,
            cfg,
            phase,
            progress,
            loss_fn,
            controller: None,
        },
        optimizer,
    )
}
pub fn rl_stage_train_step<B: AutodiffBackend>(
    state: &PipelineState,
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    rl_stage_train_step_with_controller(
        model,
        RlPhaseTrainRequest {
            state,
            batch,
            cfg,
            loss_fn,
            controller: None,
        },
        optimizer,
    )
}

pub fn rl_stage_train_step_with_controller<B: AutodiffBackend>(
    model: HydraModel<B>,
    request: RlPhaseTrainRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    match request.state.phase {
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
            let exit_phase = request.state.phase.exit_schedule_phase();
            let progress = request.state.stage_progress();
            let exit_weight = request.cfg.effective_exit_weight(exit_phase, progress);
            let (model, loss) = rl_step_with_stage_progress_and_controller(
                model,
                RlStepRequest {
                    batch: request.batch,
                    cfg: request.cfg,
                    phase: exit_phase,
                    progress,
                    loss_fn: request.loss_fn,
                    controller: request.controller,
                },
                optimizer,
            );
            Ok((
                model,
                PhaseTrainReport {
                    phase: request.state.phase,
                    skipped: false,
                    loss: Some(loss),
                    effective_lr: request.cfg.lr,
                    oracle_keep_prob: None,
                    kept_oracle_fraction: None,
                    exit_weight: Some(exit_weight),
                },
            ))
        }
        _ => Err("rl_stage_train_step only supports self-play phases"),
    }
}

/// Single RL training step with gradient accumulation across microbatches.
///
/// Advantages are normalized over the full batch before splitting so the
/// statistics match the non-microbatched path exactly. Each microbatch
/// runs forward+backward independently; gradients are accumulated and
/// applied in one optimizer step at the end.
pub fn rl_step_with_stage_progress_and_controller<B: AutodiffBackend>(
    model: HydraModel<B>,
    mut request: RlStepRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    let effective_loss_fn;
    let active_loss_fn = if let Some(ctrl) = request.controller.as_mut() {
        let (effective_cfg, _) =
            apply_head_gating_to_batch(ctrl, &request.loss_fn.config, &request.batch.targets)
                .expect("validated optional targets before RL step");
        effective_loss_fn = HydraLoss::<B>::new(effective_cfg);
        &effective_loss_fn
    } else {
        request.loss_fn
    };

    let adv = request.batch.advantages.clone();
    let adv_mean = adv.clone().mean();
    let adv_var = (adv.clone() - adv_mean.clone()).powf_scalar(2.0).mean();
    let adv_std = (adv_var + 1e-8).sqrt();
    let advantages_normed = (adv - adv_mean) / adv_std;

    let batch_size = request.batch.batch_size();
    let microbatch = request.cfg.microbatch_size.unwrap_or(batch_size).max(1);
    if batch_size <= microbatch {
        return rl_microbatch_forward(
            model,
            RlFastPathRequest {
                batch: request.batch,
                advantages_normed: &advantages_normed,
                cfg: request.cfg,
                phase: request.phase,
                progress: request.progress,
                loss_fn: active_loss_fn,
            },
            optimizer,
        );
    }

    let mut accum = GradientsAccumulator::new();
    let m = model;
    let mut total_loss = 0.0f64;
    let mut total_weight = 0usize;

    let mut start = 0usize;
    while start < batch_size {
        let end = (start + microbatch).min(batch_size);
        let mb_batch = request.batch.slice(start, end);
        #[expect(
            clippy::single_range_in_vec_init,
            reason = "Burn Tensor::slice uses a one-range array for 1D tensors"
        )]
        let adv_mb = advantages_normed.clone().slice([start..end]);
        let mb_size = end - start;

        let output = m.forward_active_train(mb_batch.obs.clone(), &active_loss_fn.config);
        let combined = drda::combined_logits(
            mb_batch.base_logits.clone(),
            output.policy_logits.clone(),
            request.cfg.tau_drda,
        );
        let ach_loss = ach_policy_loss(
            combined,
            mb_batch.targets.legal_mask.clone(),
            mb_batch.actions.clone(),
            mb_batch.pi_old.clone(),
            adv_mb,
            &request.cfg.ach_cfg,
        );
        let mut loss = active_loss_fn.total_loss(&output, &mb_batch.targets).total
            * request.cfg.aux_weight
            + ach_loss;
        if let (Some(exit_target), Some(exit_mask)) = (&mb_batch.exit_target, &mb_batch.exit_mask) {
            let exit_weight = request
                .cfg
                .effective_exit_weight(request.phase, request.progress);
            let exit_loss = exit_loss(
                output.policy_logits,
                exit_target.clone(),
                exit_mask.clone(),
                exit_weight,
            );
            loss = loss + exit_loss;
        }
        let loss_data = loss.clone().into_data().convert::<f32>();
        let loss_value = loss_data.as_slice::<f32>().expect("loss scalar readable")[0] as f64;
        total_loss += loss_value * mb_size as f64;
        total_weight += mb_size;

        let grads = loss.backward();
        accum.accumulate(&m, GradientsParams::from_grads(grads, &m));
        start = end;
    }

    let grads = accum.grads();
    let m = optimizer.step(request.cfg.lr, m, grads);
    (m, total_loss / total_weight.max(1) as f64)
}

/// Run a single (micro)batch forward+backward+step. Used for the fast
/// path when the entire batch fits in VRAM without splitting.
fn rl_microbatch_forward<B: AutodiffBackend>(
    model: HydraModel<B>,
    request: RlFastPathRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    let output = model.forward_active_train(request.batch.obs.clone(), &request.loss_fn.config);
    let combined = drda::combined_logits(
        request.batch.base_logits.clone(),
        output.policy_logits.clone(),
        request.cfg.tau_drda,
    );
    let ach_loss = ach_policy_loss(
        combined,
        request.batch.targets.legal_mask.clone(),
        request.batch.actions.clone(),
        request.batch.pi_old.clone(),
        request.advantages_normed.clone(),
        &request.cfg.ach_cfg,
    );
    let aux = request
        .loss_fn
        .total_loss(&output, &request.batch.targets)
        .total
        * request.cfg.aux_weight;
    let mut loss = ach_loss + aux;
    if let (Some(exit_target), Some(exit_mask)) =
        (&request.batch.exit_target, &request.batch.exit_mask)
    {
        let exit_weight = request
            .cfg
            .effective_exit_weight(request.phase, request.progress);
        let exit_loss = exit_loss(
            output.policy_logits,
            exit_target.clone(),
            exit_mask.clone(),
            exit_weight,
        );
        loss = loss + exit_loss;
    }
    let loss_data = loss.clone().into_data().convert::<f32>();
    let loss_value = loss_data.as_slice::<f32>().expect("loss scalar readable")[0] as f64;
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optimizer.step(request.cfg.lr, model, grads);
    (model, loss_value)
}
