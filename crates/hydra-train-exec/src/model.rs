//! Training execution model adapters.
//!
//! The canonical Burn model implementation lives in `hydra_model::model`; this
//! module owns adapters from shared training configuration types to model
//! forward policies.

pub use hydra_model::model::{
    ActorNet, HydraForwardPolicy, HydraModel, HydraModelConfig, HydraModelInit, HydraOutput,
    LearnerNet, ModelAdvancedHead,
};

use hydra_train_types::{head_gates::AdvancedHead, losses::HydraLossConfig};

/// Converts a training loss configuration into the model crate's forward-head policy.
pub fn forward_policy_from_loss_config(value: &HydraLossConfig) -> HydraForwardPolicy {
    HydraForwardPolicy {
        w_score: value.w_score,
        w_tenpai: value.w_tenpai,
        w_grp: value.w_grp,
        w_opp: value.w_opp,
        w_danger: value.w_danger,
        w_oracle_critic: value.w_oracle_critic,
        w_belief_fields: value.w_belief_fields,
        w_mixture_weight: value.w_mixture_weight,
        w_opponent_hand_type: value.w_opponent_hand_type,
        w_delta_q: value.w_delta_q,
        w_safety_residual: value.w_safety_residual,
    }
}

/// Converts a training head-gate enum value into the model crate's warmup-head enum.
pub fn model_head_from_train_head(value: AdvancedHead) -> ModelAdvancedHead {
    match value {
        AdvancedHead::OracleCritic => ModelAdvancedHead::OracleCritic,
        AdvancedHead::BeliefFields => ModelAdvancedHead::BeliefFields,
        AdvancedHead::MixtureWeight => ModelAdvancedHead::MixtureWeight,
        AdvancedHead::OpponentHandType => ModelAdvancedHead::OpponentHandType,
        AdvancedHead::DeltaQ => ModelAdvancedHead::DeltaQ,
        AdvancedHead::SafetyResidual => ModelAdvancedHead::SafetyResidual,
    }
}

/// Training extension methods for invoking hydra-model forwards with shared train config types.
pub trait HydraTrainModelExt<B: burn::prelude::Backend> {
    /// Runs an active-head forward pass using weights from `HydraLossConfig`.
    fn forward_active_train(
        &self,
        x: burn::prelude::Tensor<B, 3>,
        loss_cfg: &HydraLossConfig,
    ) -> HydraOutput<B>;

    /// Runs a warmup-aware forward pass using train-local advanced head gates.
    fn forward_with_warmup_train(
        &self,
        x: burn::prelude::Tensor<B, 3>,
        loss_cfg: &HydraLossConfig,
        warmup_heads: &[AdvancedHead],
    ) -> HydraOutput<B>;
}

impl<B: burn::prelude::Backend> HydraTrainModelExt<B> for HydraModel<B> {
    fn forward_active_train(
        &self,
        x: burn::prelude::Tensor<B, 3>,
        loss_cfg: &HydraLossConfig,
    ) -> HydraOutput<B> {
        self.forward_active(x, &forward_policy_from_loss_config(loss_cfg))
    }

    fn forward_with_warmup_train(
        &self,
        x: burn::prelude::Tensor<B, 3>,
        loss_cfg: &HydraLossConfig,
        warmup_heads: &[AdvancedHead],
    ) -> HydraOutput<B> {
        let model_warmup_heads: Vec<ModelAdvancedHead> = warmup_heads
            .iter()
            .copied()
            .map(model_head_from_train_head)
            .collect();
        self.forward_with_warmup(
            x,
            &forward_policy_from_loss_config(loss_cfg),
            &model_warmup_heads,
        )
    }
}
