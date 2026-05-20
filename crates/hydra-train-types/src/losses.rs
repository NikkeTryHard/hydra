//! Tensor target and scalar loss configuration types shared by training crates.
//!
//! This module owns the target batch container and loss-weight configuration
//! without depending on `hydra-train` models, loss functions, or training
//! orchestration. Keeping these types below `hydra-train` lets callers share
//! target/config contracts without creating dependency cycles.

use burn::prelude::*;

use crate::head_gates::TargetPresence;

/// Scalar loss components produced by Hydra's loss function.
///
/// Fields are one-element tensors so callers can preserve backend placement,
/// combine losses before synchronization, and report metrics without changing
/// historical names or ordering.
pub struct LossBreakdown<B: Backend> {
    pub policy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    pub grp: Tensor<B, 1>,
    pub tenpai: Tensor<B, 1>,
    pub danger: Tensor<B, 1>,
    pub opp_next: Tensor<B, 1>,
    pub score_pdf: Tensor<B, 1>,
    pub score_cdf: Tensor<B, 1>,
    pub oracle_critic: Tensor<B, 1>,
    pub belief_fields: Tensor<B, 1>,
    pub mixture_weight: Tensor<B, 1>,
    pub opponent_hand_type: Tensor<B, 1>,
    pub delta_q: Tensor<B, 1>,
    pub safety_residual: Tensor<B, 1>,
    pub total: Tensor<B, 1>,
}

impl<B: Backend> LossBreakdown<B> {
    pub fn all_finite(&self) -> bool {
        tensor_scalar_finite(&self.policy)
            && tensor_scalar_finite(&self.value)
            && tensor_scalar_finite(&self.grp)
            && tensor_scalar_finite(&self.tenpai)
            && tensor_scalar_finite(&self.danger)
            && tensor_scalar_finite(&self.opp_next)
            && tensor_scalar_finite(&self.score_pdf)
            && tensor_scalar_finite(&self.score_cdf)
            && tensor_scalar_finite(&self.oracle_critic)
            && tensor_scalar_finite(&self.belief_fields)
            && tensor_scalar_finite(&self.mixture_weight)
            && tensor_scalar_finite(&self.opponent_hand_type)
            && tensor_scalar_finite(&self.delta_q)
            && tensor_scalar_finite(&self.safety_residual)
            && tensor_scalar_finite(&self.total)
    }
}

fn tensor_scalar_finite<B: Backend>(tensor: &Tensor<B, 1>) -> bool {
    let data = tensor.clone().into_data().convert::<f32>();
    data.as_slice::<f32>()
        .is_ok_and(|values| values.len() == 1 && values[0].is_finite())
}

/// Batched supervised targets consumed by Hydra loss functions.
///
/// All tensor fields are batched on dimension 0. Optional advanced-head targets
/// are present only when the batch carries labels for that head. When present,
/// masks use positive values to mark valid per-sample or per-action labels.
#[derive(Clone)]
pub struct HydraTargets<B: Backend> {
    /// Policy target distribution over legal actions, shape `[batch, 46]`.
    pub policy_target: Tensor<B, 2>,
    /// Legal action mask, shape `[batch, 46]`.
    pub legal_mask: Tensor<B, 2>,
    /// Scalar value target, shape `[batch]`.
    pub value_target: Tensor<B, 1>,
    /// GRP class target, shape `[batch, classes]`.
    pub grp_target: Tensor<B, 2>,
    /// Tenpai target, shape `[batch, classes]`.
    pub tenpai_target: Tensor<B, 2>,
    /// Danger target, shape `[batch, players, tiles]`.
    pub danger_target: Tensor<B, 3>,
    /// Danger target mask, shape `[batch, players, tiles]`.
    pub danger_mask: Tensor<B, 3>,
    /// Opponent next-discard target, shape `[batch, opponents, tiles]`.
    pub opp_next_target: Tensor<B, 3>,
    /// Score PDF target, shape `[batch, buckets]`.
    pub score_pdf_target: Tensor<B, 2>,
    /// Score CDF target, shape `[batch, buckets]`.
    pub score_cdf_target: Tensor<B, 2>,
    /// Optional oracle critic target, shape `[batch, players]`.
    pub oracle_target: Option<Tensor<B, 2>>,
    /// Optional belief-field target, shape `[batch, fields, values]`.
    pub belief_fields_target: Option<Tensor<B, 3>>,
    /// Optional per-sample belief-field mask, shape `[batch]`.
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    /// Optional mixture-weight target, shape `[batch, classes]`.
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    /// Optional per-sample mixture-weight mask, shape `[batch]`.
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    /// Optional opponent hand-type target, shape `[batch, classes]`.
    pub opponent_hand_type_target: Option<Tensor<B, 2>>,
    /// Optional delta-Q target, shape `[batch, actions]`.
    pub delta_q_target: Option<Tensor<B, 2>>,
    /// Optional delta-Q action mask, shape `[batch, actions]`.
    pub delta_q_mask: Option<Tensor<B, 2>>,
    /// Optional safety residual target, shape `[batch, actions]`.
    pub safety_residual_target: Option<Tensor<B, 2>>,
    /// Optional safety residual action mask, shape `[batch, actions]`.
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    /// Optional per-sample oracle-guidance mask, shape `[batch]`.
    pub oracle_guidance_mask: Option<Tensor<B, 1>>,
    /// Optional cached target-presence metadata for head activation gates.
    pub target_presence: Option<TargetPresence>,
}

impl<B: Backend> HydraTargets<B> {
    /// Slice all target tensors along the batch dimension (dim 0).
    ///
    /// Produces a sub-batch covering `[start..end)`. Used by microbatch
    /// accumulation to split a full RL batch into VRAM-friendly chunks.
    #[allow(
        clippy::single_range_in_vec_init,
        reason = "Burn slice API expects a one-element range slice"
    )]
    pub fn slice_batch(&self, start: usize, end: usize) -> Self {
        let r1 = [start..end];
        let r2 = [start..end];
        let r3 = [start..end];
        Self {
            policy_target: self.policy_target.clone().slice(r1.clone()),
            legal_mask: self.legal_mask.clone().slice(r1.clone()),
            value_target: self.value_target.clone().slice(r2.clone()),
            grp_target: self.grp_target.clone().slice(r1.clone()),
            tenpai_target: self.tenpai_target.clone().slice(r1.clone()),
            danger_target: self.danger_target.clone().slice(r3.clone()),
            danger_mask: self.danger_mask.clone().slice(r3.clone()),
            opp_next_target: self.opp_next_target.clone().slice(r3.clone()),
            score_pdf_target: self.score_pdf_target.clone().slice(r1.clone()),
            score_cdf_target: self.score_cdf_target.clone().slice(r1.clone()),
            oracle_target: self
                .oracle_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            belief_fields_target: self
                .belief_fields_target
                .as_ref()
                .map(|t| t.clone().slice(r3.clone())),
            belief_fields_mask: self
                .belief_fields_mask
                .as_ref()
                .map(|t| t.clone().slice(r2.clone())),
            mixture_weight_target: self
                .mixture_weight_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            mixture_weight_mask: self
                .mixture_weight_mask
                .as_ref()
                .map(|t| t.clone().slice(r2.clone())),
            opponent_hand_type_target: self
                .opponent_hand_type_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            delta_q_target: self
                .delta_q_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            delta_q_mask: self
                .delta_q_mask
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            safety_residual_target: self
                .safety_residual_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            safety_residual_mask: self
                .safety_residual_mask
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            oracle_guidance_mask: self
                .oracle_guidance_mask
                .as_ref()
                .map(|t| t.clone().slice(r2)),
            target_presence: None,
        }
    }
}

/// Loss-weight configuration for Hydra's baseline and advanced heads.
///
/// Advanced heads default to zero and are intended to be enabled by explicit
/// configuration or by the head-gate controller once density/interference gates
/// pass. The generated `Config` builder API preserves the historical
/// `HydraLossConfig::new().with_w_*` call pattern.
#[derive(Config, Debug)]
pub struct HydraLossConfig {
    /// Policy cross-entropy weight.
    #[config(default = "1.0")]
    pub w_pi: f32,
    /// Value MSE weight.
    #[config(default = "0.5")]
    pub w_v: f32,
    /// GRP classification weight.
    #[config(default = "0.2")]
    pub w_grp: f32,
    /// Tenpai prediction weight.
    #[config(default = "0.1")]
    pub w_tenpai: f32,
    /// Tile danger prediction weight.
    #[config(default = "0.1")]
    pub w_danger: f32,
    /// Opponent next-discard prediction weight.
    #[config(default = "0.1")]
    pub w_opp: f32,
    /// Score distribution weight, applied to both PDF and CDF losses.
    #[config(default = "0.025")]
    pub w_score: f32,
    /// Oracle critic advanced-head weight.
    #[config(default = "0.0")]
    pub w_oracle_critic: f32,
    /// Belief-fields advanced-head weight.
    #[config(default = "0.0")]
    pub w_belief_fields: f32,
    /// Mixture-weight advanced-head weight.
    #[config(default = "0.0")]
    pub w_mixture_weight: f32,
    /// Opponent hand-type advanced-head weight.
    #[config(default = "0.0")]
    pub w_opponent_hand_type: f32,
    /// Delta-Q advanced-head weight.
    #[config(default = "0.0")]
    pub w_delta_q: f32,
    /// Safety-residual advanced-head weight.
    #[config(default = "0.0")]
    pub w_safety_residual: f32,
}

impl HydraLossConfig {
    /// Returns the total configured scalar weight across all loss components.
    pub fn total_weight(&self) -> f32 {
        self.w_pi
            + self.w_v
            + self.w_grp
            + self.w_tenpai
            + self.w_danger
            + self.w_opp
            + self.w_score * 2.0
            + self.w_oracle_critic
            + self.w_belief_fields
            + self.w_mixture_weight
            + self.w_opponent_hand_type
            + self.w_delta_q
            + self.w_safety_residual
    }

    /// Returns a copy with every loss weight multiplied by `factor`.
    pub fn scale_all(&self, factor: f32) -> Self {
        Self::new()
            .with_w_pi(self.w_pi * factor)
            .with_w_v(self.w_v * factor)
            .with_w_grp(self.w_grp * factor)
            .with_w_tenpai(self.w_tenpai * factor)
            .with_w_danger(self.w_danger * factor)
            .with_w_opp(self.w_opp * factor)
            .with_w_score(self.w_score * factor)
            .with_w_oracle_critic(self.w_oracle_critic * factor)
            .with_w_belief_fields(self.w_belief_fields * factor)
            .with_w_mixture_weight(self.w_mixture_weight * factor)
            .with_w_opponent_hand_type(self.w_opponent_hand_type * factor)
            .with_w_delta_q(self.w_delta_q * factor)
            .with_w_safety_residual(self.w_safety_residual * factor)
    }

    /// Returns a compact human-readable summary of core loss weights.
    pub fn summary(&self) -> String {
        format!(
            "loss(pi={:.1}, v={:.1}, grp={:.1})",
            self.w_pi, self.w_v, self.w_grp
        )
    }

    /// Validates that every configured loss weight is finite and non-negative.
    pub fn validate(&self) -> Result<(), &'static str> {
        let weights = [
            self.w_pi,
            self.w_v,
            self.w_grp,
            self.w_tenpai,
            self.w_danger,
            self.w_opp,
            self.w_score,
            self.w_oracle_critic,
            self.w_belief_fields,
            self.w_mixture_weight,
            self.w_opponent_hand_type,
            self.w_delta_q,
            self.w_safety_residual,
        ];
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err("loss weights must be finite");
        }
        if weights.iter().any(|&weight| weight < 0.0) {
            return Err("loss weights must be non-negative");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
