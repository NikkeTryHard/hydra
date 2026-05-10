//! Behavioral-cloning execution adapters.

use burn::prelude::*;
use hydra_model::model::HydraOutput;
use hydra_train_runtime::data::sample::MjaiBatch;
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_types::head_gates::{AdvancedHead, borrow_or_extract_target_presence};
use hydra_train_types::losses::{HydraTargets, LossBreakdown};

use crate::losses::HydraLoss;

pub use hydra_train_algo::bc::{
    BcExitConfig, cosine_annealing_lr, maybe_add_exit_loss, oracle_guidance_mask_tensor,
    oracle_guidance_mask_values, policy_agreement, policy_agreement_counts,
    target_actions_from_policy_target, warmup_then_cosine_lr,
};

/// Adds optional ExIt loss to a BC breakdown total.
pub fn bc_total_with_exit_from_breakdown<B: Backend>(
    output: &HydraOutput<B>,
    batch: &MjaiBatch<B>,
    breakdown: &LossBreakdown<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    maybe_add_exit_loss(
        breakdown.total.clone(),
        output.policy_logits.clone(),
        batch.exit_target.as_ref(),
        batch.exit_mask.as_ref(),
        exit_cfg,
    )
}

/// Builds the effective BC loss and warmup head set for one batch.
pub fn gated_bc_context<B: Backend>(
    controller: Option<&mut HeadActivationController>,
    base_loss_fn: &HydraLoss<B>,
    targets: &HydraTargets<B>,
) -> (HydraLoss<B>, Vec<AdvancedHead>) {
    if let Some(ctrl) = controller {
        if base_loss_fn.config.w_delta_q > 0.0 {
            let presence = borrow_or_extract_target_presence(targets);
            ctrl.record_batch(&presence);
            ctrl.try_activate(AdvancedHead::DeltaQ);
        }
        let effective_cfg = ctrl.approved_loss_config(&base_loss_fn.config);
        let warmup_heads = ctrl.warmup_heads();
        (HydraLoss::<B>::new(effective_cfg), warmup_heads)
    } else {
        (HydraLoss::<B>::new(base_loss_fn.config.clone()), Vec::new())
    }
}
