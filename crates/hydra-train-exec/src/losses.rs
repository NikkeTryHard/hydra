use burn::prelude::*;
use std::marker::PhantomData;

use hydra_model::model::HydraOutput;
use hydra_train_algo::bc::{BcExitConfig, maybe_add_exit_loss};
use hydra_train_algo::losses::{
    belief_fields_bce_per_sample, combine_sample_masks, danger_focal_bce, grp_ce,
    masked_action_mse, masked_mean, mixture_weight_ce_per_sample, opp_next_ce,
    opponent_hand_type_ce_per_sample, oracle_critic_loss_per_sample, policy_ce, score_cdf_bce,
    score_pdf_ce, tenpai_bce, value_mse,
};
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_LOSS_ADVANCED_HEADS, PROFILING_STAGE_LOSS_BASE_HEADS,
    PROFILING_STAGE_LOSS_EXIT, PROFILING_STAGE_LOSS_POLICY_CE, PROFILING_STAGE_LOSS_TOTAL_COMBINE,
    PROFILING_STAGE_LOSS_VALUE_MSE,
};
use hydra_train_types::losses::{HydraLossConfig, HydraTargets, LossBreakdown};

/// Multi-head Hydra training loss adapter.
pub struct HydraLoss<B: Backend> {
    /// Loss weights and optional-head gates used when computing totals.
    pub config: HydraLossConfig,
    _backend: PhantomData<B>,
}

#[allow(
    missing_docs,
    reason = "BC loss fusion seam keeps borrowed field names explicit"
)]
/// Borrowed inputs for the BC loss seam.
pub struct BcLossInputs<'a, B: Backend> {
    pub outputs: &'a HydraOutput<B>,
    pub targets: &'a HydraTargets<B>,
    pub exit_target: Option<&'a Tensor<B, 2>>,
    pub exit_mask: Option<&'a Tensor<B, 2>>,
    pub exit_cfg: &'a BcExitConfig,
}

#[allow(
    missing_docs,
    reason = "BC loss fusion seam keeps result field names explicit"
)]
/// BC loss result: component breakdown plus ExIt-adjusted total for backward.
pub struct BcLossResult<B: Backend> {
    pub breakdown: LossBreakdown<B>,
    pub total: Tensor<B, 1>,
}

impl<B: Backend> HydraLoss<B> {
    /// Creates a loss adapter from a shared Hydra loss configuration.
    pub fn new(config: HydraLossConfig) -> Self {
        Self {
            config,
            _backend: PhantomData,
        }
    }

    /// Computes BC loss through the oracle seam.
    pub fn bc_loss(&self, inputs: BcLossInputs<'_, B>) -> BcLossResult<B> {
        let breakdown = self.total_loss(inputs.outputs, inputs.targets);
        let total = {
            let _exit_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_EXIT);
            maybe_add_exit_loss(
                breakdown.total.clone(),
                inputs.outputs.policy_logits.clone(),
                inputs.exit_target,
                inputs.exit_mask,
                inputs.exit_cfg,
            )
        };
        BcLossResult { breakdown, total }
    }

    /// Computes all configured loss components and their weighted total.
    pub fn total_loss(
        &self,
        outputs: &HydraOutput<B>,
        targets: &HydraTargets<B>,
    ) -> LossBreakdown<B> {
        let oracle_mask = targets.oracle_guidance_mask.clone();
        let l_pi = {
            let _policy_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_POLICY_CE);
            policy_ce(
                outputs.policy_logits.clone(),
                targets.policy_target.clone(),
                targets.legal_mask.clone(),
            )
            .mean()
        };
        let l_v = {
            let _value_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_VALUE_MSE);
            value_mse(
                outputs.value.clone().squeeze_dim::<1>(1),
                targets.value_target.clone(),
            )
            .mean()
        };
        let (l_grp, l_tenpai, l_danger, l_opp, l_pdf, l_cdf) = {
            let _base_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_BASE_HEADS);
            let l_grp = grp_ce(outputs.grp.clone(), targets.grp_target.clone()).mean();
            let l_tenpai =
                tenpai_bce(outputs.opp_tenpai.clone(), targets.tenpai_target.clone()).mean();
            let l_danger = danger_focal_bce(
                outputs.danger.clone(),
                targets.danger_target.clone(),
                targets.danger_mask.clone(),
            )
            .mean();
            let l_opp = opp_next_ce(
                outputs.opp_next_discard.clone(),
                targets.opp_next_target.clone(),
            )
            .mean();
            let l_pdf =
                score_pdf_ce(outputs.score_pdf.clone(), targets.score_pdf_target.clone()).mean();
            let l_cdf =
                score_cdf_bce(outputs.score_cdf.clone(), targets.score_cdf_target.clone()).mean();
            (l_grp, l_tenpai, l_danger, l_opp, l_pdf, l_cdf)
        };
        let zero = outputs.value.clone().sum() * 0.0;
        let c = &self.config;
        let _advanced_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_ADVANCED_HEADS);
        let l_oracle = if c.w_oracle_critic > 0.0 {
            match &targets.oracle_target {
                Some(target) => masked_mean(
                    oracle_critic_loss_per_sample(outputs.oracle_critic.clone(), target.clone()),
                    oracle_mask.clone(),
                ),
                None => zero.clone(),
            }
        } else {
            zero.clone()
        };
        let l_belief = if c.w_belief_fields > 0.0 {
            match (&targets.belief_fields_target, &targets.belief_fields_mask) {
                (Some(target), Some(mask)) => masked_mean(
                    belief_fields_bce_per_sample(outputs.belief_fields.clone(), target.clone()),
                    combine_sample_masks(Some(mask.clone()), oracle_mask.clone()),
                ),
                _ => zero.clone(),
            }
        } else {
            zero.clone()
        };
        let l_mix = if c.w_mixture_weight > 0.0 {
            match (&targets.mixture_weight_target, &targets.mixture_weight_mask) {
                (Some(target), Some(mask)) => masked_mean(
                    mixture_weight_ce_per_sample(
                        outputs.mixture_weight_logits.clone(),
                        target.clone(),
                    ),
                    combine_sample_masks(Some(mask.clone()), oracle_mask.clone()),
                ),
                _ => zero.clone(),
            }
        } else {
            zero.clone()
        };
        let l_hand_type = if c.w_opponent_hand_type > 0.0 {
            match &targets.opponent_hand_type_target {
                Some(target) => masked_mean(
                    opponent_hand_type_ce_per_sample(
                        outputs.opponent_hand_type.clone(),
                        target.clone(),
                    ),
                    oracle_mask.clone(),
                ),
                None => zero.clone(),
            }
        } else {
            zero.clone()
        };
        let l_delta_q = if c.w_delta_q > 0.0 {
            match (&targets.delta_q_target, &targets.delta_q_mask) {
                (Some(target), Some(mask)) => {
                    masked_action_mse(outputs.delta_q.clone(), target.clone(), mask.clone())
                }
                _ => zero.clone(),
            }
        } else {
            zero.clone()
        };
        let l_safety_residual = if c.w_safety_residual > 0.0 {
            match (
                &targets.safety_residual_target,
                &targets.safety_residual_mask,
            ) {
                (Some(target), Some(mask)) => masked_action_mse(
                    outputs.safety_residual.clone(),
                    target.clone(),
                    mask.clone(),
                ),
                _ => zero.clone(),
            }
        } else {
            zero.clone()
        };
        drop(_advanced_scope);
        let total = {
            let _total_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS_TOTAL_COMBINE);
            l_pi.clone() * c.w_pi
                + l_v.clone() * c.w_v
                + l_grp.clone() * c.w_grp
                + l_tenpai.clone() * c.w_tenpai
                + l_danger.clone() * c.w_danger
                + l_opp.clone() * c.w_opp
                + l_pdf.clone() * c.w_score
                + l_cdf.clone() * c.w_score
                + l_oracle.clone() * c.w_oracle_critic
                + l_belief.clone() * c.w_belief_fields
                + l_mix.clone() * c.w_mixture_weight
                + l_hand_type.clone() * c.w_opponent_hand_type
                + l_delta_q.clone() * c.w_delta_q
                + l_safety_residual.clone() * c.w_safety_residual
        };
        LossBreakdown {
            policy: l_pi,
            value: l_v,
            grp: l_grp,
            tenpai: l_tenpai,
            danger: l_danger,
            opp_next: l_opp,
            score_pdf: l_pdf,
            score_cdf: l_cdf,
            oracle_critic: l_oracle,
            belief_fields: l_belief,
            mixture_weight: l_mix,
            opponent_hand_type: l_hand_type,
            delta_q: l_delta_q,
            safety_residual: l_safety_residual,
            total,
        }
    }
}
