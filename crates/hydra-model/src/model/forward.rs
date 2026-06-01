use burn::prelude::*;
use burn::tensor::module;
use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::profiling;

use super::outputs::*;
use super::{HydraForwardPolicy, HydraModel, ModelAdvancedHead};
use super::{
    MODEL_SCOPE_BACKBONE, MODEL_SCOPE_HEADS_ADVANCED, MODEL_SCOPE_HEADS_LINEAR_BASE,
    MODEL_SCOPE_HEADS_POLICY, MODEL_SCOPE_HEADS_SPATIAL_BASE, MODEL_SCOPE_HEADS_VALUE,
};

impl<B: Backend> HydraModel<B> {
    pub fn policy_logits_for(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward_policy(x)
    }
    /// Runs only backbone + policy + value heads.
    ///
    /// Self-play inference only needs logits and value. Skipping the
    /// other 12 heads avoids ~12 unnecessary matmuls and their VRAM
    /// allocations per forward pass.
    pub fn forward_value(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (_, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let _value_scope = profiling::scope(MODEL_SCOPE_HEADS_VALUE);
        self.value.forward(pooled)
    }

    pub fn forward_policy(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let pooled = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward_pooled(x)
        };
        let _policy_scope = profiling::scope(MODEL_SCOPE_HEADS_POLICY);
        self.policy.forward(pooled)
    }

    pub fn forward_policy_value(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (_, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let policy_logits = {
            let _policy_scope = profiling::scope(MODEL_SCOPE_HEADS_POLICY);
            self.policy.forward(pooled.clone())
        };
        let value = {
            let _value_scope = profiling::scope(MODEL_SCOPE_HEADS_VALUE);
            self.value.forward(pooled)
        };
        (policy_logits, value)
    }

    /// Forward pass that detaches outputs of zero-weight heads.
    ///
    /// All heads still run their forward pass (shapes must match), but
    /// heads with zero loss weight have their outputs detached from the
    /// autograd graph. This prevents gradient computation and reduces
    /// VRAM usage for activations that won't contribute to the loss.
    pub fn forward_active(&self, x: Tensor<B, 3>, policy: &HydraForwardPolicy) -> HydraOutput<B> {
        self.forward_with_warmup(x, policy, &[])
    }

    fn forward_base_linear_heads(&self, pooled: Tensor<B, 2>) -> BaseLinearHeadOutput<B> {
        let packed_weight = Tensor::cat(
            vec![
                self.policy.linear().weight.val(),
                self.value.linear().weight.val(),
                self.score_pdf.linear().weight.val(),
                self.score_cdf.linear().weight.val(),
                self.opp_tenpai.linear().weight.val(),
                self.grp.linear().weight.val(),
            ],
            1,
        );
        let packed_bias = Tensor::cat(
            vec![
                self.policy
                    .linear()
                    .bias
                    .as_ref()
                    .expect("policy head bias should exist")
                    .val(),
                self.value
                    .linear()
                    .bias
                    .as_ref()
                    .expect("value head bias should exist")
                    .val(),
                self.score_pdf
                    .linear()
                    .bias
                    .as_ref()
                    .expect("score_pdf head bias should exist")
                    .val(),
                self.score_cdf
                    .linear()
                    .bias
                    .as_ref()
                    .expect("score_cdf head bias should exist")
                    .val(),
                self.opp_tenpai
                    .linear()
                    .bias
                    .as_ref()
                    .expect("opp_tenpai head bias should exist")
                    .val(),
                self.grp
                    .linear()
                    .bias
                    .as_ref()
                    .expect("grp head bias should exist")
                    .val(),
            ],
            0,
        );
        let packed = module::linear(pooled, packed_weight, Some(packed_bias));
        let batch = packed.dims()[0];
        debug_assert_eq!(packed.dims()[1], BASE_LINEAR_HEAD_WIDTH);
        let policy_logits = packed
            .clone()
            .slice([0..batch, POLICY_HEAD_START..VALUE_HEAD_START]);
        let value = packed
            .clone()
            .slice([0..batch, VALUE_HEAD_START..SCORE_PDF_HEAD_START])
            .tanh();
        let score_pdf = packed
            .clone()
            .slice([0..batch, SCORE_PDF_HEAD_START..SCORE_CDF_HEAD_START]);
        let score_cdf = packed
            .clone()
            .slice([0..batch, SCORE_CDF_HEAD_START..OPP_TENPAI_HEAD_START]);
        let opp_tenpai = packed
            .clone()
            .slice([0..batch, OPP_TENPAI_HEAD_START..GRP_HEAD_START]);
        let grp = packed.slice([0..batch, GRP_HEAD_START..BASE_LINEAR_HEAD_WIDTH]);
        BaseLinearHeadOutput {
            policy_logits,
            value,
            score_pdf,
            score_cdf,
            opp_tenpai,
            grp,
        }
    }

    pub fn forward_with_warmup(
        &self,
        x: Tensor<B, 3>,
        policy: &HydraForwardPolicy,
        warmup_heads: &[ModelAdvancedHead],
    ) -> HydraOutput<B> {
        self.forward_with_warmup_by(x, policy, |head| warmup_heads.contains(&head))
    }

    pub fn forward_with_warmup_by(
        &self,
        x: Tensor<B, 3>,
        policy: &HydraForwardPolicy,
        is_warmup: impl Fn(ModelAdvancedHead) -> bool,
    ) -> HydraOutput<B> {
        let (spatial, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let oracle_input = pooled.clone().detach();
        let batch = pooled.dims()[0];
        let device = pooled.device();

        let base = {
            let _linear_base_scope = profiling::scope(MODEL_SCOPE_HEADS_LINEAR_BASE);
            self.forward_base_linear_heads(pooled.clone())
        };
        let (opp_next_discard, danger) = {
            let _spatial_base_scope = profiling::scope(MODEL_SCOPE_HEADS_SPATIAL_BASE);
            let opp_next_discard = self.opp_next_discard.forward(spatial.clone());
            let danger = self.danger.forward(spatial.clone());
            (opp_next_discard, danger)
        };
        let _advanced_scope = profiling::scope(MODEL_SCOPE_HEADS_ADVANCED);
        let oracle_critic =
            if policy.w_oracle_critic > 0.0 && !is_warmup(ModelAdvancedHead::OracleCritic) {
                self.oracle_critic.forward(oracle_input)
            } else {
                zero_linear_head(batch, 4, &device)
            };
        let belief_fields =
            if policy.w_belief_fields > 0.0 && !is_warmup(ModelAdvancedHead::BeliefFields) {
                self.belief_field.forward(spatial.clone())
            } else {
                zero_spatial_head(batch, 16, 34, &device)
            };
        let mixture_weight_logits =
            if policy.w_mixture_weight > 0.0 && !is_warmup(ModelAdvancedHead::MixtureWeight) {
                self.mixture_weight.forward(pooled.clone())
            } else {
                zero_linear_head(batch, 4, &device)
            };
        let opponent_hand_type = if policy.w_opponent_hand_type > 0.0
            && !is_warmup(ModelAdvancedHead::OpponentHandType)
        {
            self.opponent_hand_type.forward(pooled.clone())
        } else {
            zero_linear_head(batch, 24, &device)
        };
        let delta_q = if policy.w_delta_q > 0.0 && !is_warmup(ModelAdvancedHead::DeltaQ) {
            self.delta_q.forward(pooled.clone())
        } else {
            zero_linear_head(batch, HYDRA_ACTION_SPACE, &device)
        };
        let safety_residual =
            if policy.w_safety_residual > 0.0 && !is_warmup(ModelAdvancedHead::SafetyResidual) {
                self.safety_residual.forward(pooled)
            } else {
                zero_linear_head(batch, HYDRA_ACTION_SPACE, &device)
            };
        let _ = _advanced_scope;

        HydraOutput {
            policy_logits: base.policy_logits,
            value: base.value,
            score_pdf: if policy.w_score > 0.0 {
                base.score_pdf
            } else {
                base.score_pdf.detach()
            },
            score_cdf: if policy.w_score > 0.0 {
                base.score_cdf
            } else {
                base.score_cdf.detach()
            },
            opp_tenpai: if policy.w_tenpai > 0.0 {
                base.opp_tenpai
            } else {
                base.opp_tenpai.detach()
            },
            grp: if policy.w_grp > 0.0 {
                base.grp
            } else {
                base.grp.detach()
            },
            opp_next_discard: if policy.w_opp > 0.0 {
                opp_next_discard
            } else {
                opp_next_discard.detach()
            },
            danger: if policy.w_danger > 0.0 {
                danger
            } else {
                danger.detach()
            },
            oracle_critic,
            belief_fields,
            mixture_weight_logits,
            opponent_hand_type,
            delta_q,
            safety_residual,
        }
    }

    pub fn forward_train_with_warmup_by(
        &self,
        x: Tensor<B, 3>,
        policy: &HydraForwardPolicy,
        is_warmup: impl Fn(ModelAdvancedHead) -> bool,
    ) -> HydraTrainOutput<B> {
        let (spatial, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let oracle_input = pooled.clone().detach();

        let base = {
            let _linear_base_scope = profiling::scope(MODEL_SCOPE_HEADS_LINEAR_BASE);
            self.forward_base_linear_heads(pooled.clone())
        };
        let (opp_next_discard, danger) = {
            let _spatial_base_scope = profiling::scope(MODEL_SCOPE_HEADS_SPATIAL_BASE);
            let opp_next_discard = self.opp_next_discard.forward(spatial.clone());
            let danger = self.danger.forward(spatial.clone());
            (opp_next_discard, danger)
        };
        let _advanced_scope = profiling::scope(MODEL_SCOPE_HEADS_ADVANCED);
        let oracle_critic =
            if policy.w_oracle_critic > 0.0 && !is_warmup(ModelAdvancedHead::OracleCritic) {
                Some(self.oracle_critic.forward(oracle_input))
            } else {
                None
            };
        let belief_fields =
            if policy.w_belief_fields > 0.0 && !is_warmup(ModelAdvancedHead::BeliefFields) {
                Some(self.belief_field.forward(spatial.clone()))
            } else {
                None
            };
        let mixture_weight_logits =
            if policy.w_mixture_weight > 0.0 && !is_warmup(ModelAdvancedHead::MixtureWeight) {
                Some(self.mixture_weight.forward(pooled.clone()))
            } else {
                None
            };
        let opponent_hand_type = if policy.w_opponent_hand_type > 0.0
            && !is_warmup(ModelAdvancedHead::OpponentHandType)
        {
            Some(self.opponent_hand_type.forward(pooled.clone()))
        } else {
            None
        };
        let delta_q = if policy.w_delta_q > 0.0 && !is_warmup(ModelAdvancedHead::DeltaQ) {
            Some(self.delta_q.forward(pooled.clone()))
        } else {
            None
        };
        let safety_residual =
            if policy.w_safety_residual > 0.0 && !is_warmup(ModelAdvancedHead::SafetyResidual) {
                Some(self.safety_residual.forward(pooled))
            } else {
                None
            };
        let _ = _advanced_scope;

        HydraTrainOutput {
            policy_logits: base.policy_logits,
            value: base.value,
            score_pdf: if policy.w_score > 0.0 {
                base.score_pdf
            } else {
                base.score_pdf.detach()
            },
            score_cdf: if policy.w_score > 0.0 {
                base.score_cdf
            } else {
                base.score_cdf.detach()
            },
            opp_tenpai: if policy.w_tenpai > 0.0 {
                base.opp_tenpai
            } else {
                base.opp_tenpai.detach()
            },
            grp: if policy.w_grp > 0.0 {
                base.grp
            } else {
                base.grp.detach()
            },
            opp_next_discard: if policy.w_opp > 0.0 {
                opp_next_discard
            } else {
                opp_next_discard.detach()
            },
            danger: if policy.w_danger > 0.0 {
                danger
            } else {
                danger.detach()
            },
            oracle_critic,
            belief_fields,
            mixture_weight_logits,
            opponent_hand_type,
            delta_q,
            safety_residual,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
        let (spatial, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let oracle_input = pooled.clone().detach();
        let base = {
            let _linear_base_scope = profiling::scope(MODEL_SCOPE_HEADS_LINEAR_BASE);
            self.forward_base_linear_heads(pooled.clone())
        };
        let (opp_next_discard, danger) = {
            let _spatial_base_scope = profiling::scope(MODEL_SCOPE_HEADS_SPATIAL_BASE);
            let opp_next_discard = self.opp_next_discard.forward(spatial.clone());
            let danger = self.danger.forward(spatial.clone());
            (opp_next_discard, danger)
        };
        let (
            oracle_critic,
            belief_fields,
            mixture_weight_logits,
            opponent_hand_type,
            delta_q,
            safety_residual,
        ) = {
            let _advanced_scope = profiling::scope(MODEL_SCOPE_HEADS_ADVANCED);
            let oracle_critic = self.oracle_critic.forward(oracle_input);
            let belief_fields = self.belief_field.forward(spatial);
            let mixture_weight_logits = self.mixture_weight.forward(pooled.clone());
            let opponent_hand_type = self.opponent_hand_type.forward(pooled.clone());
            let delta_q = self.delta_q.forward(pooled.clone());
            let safety_residual = self.safety_residual.forward(pooled);
            (
                oracle_critic,
                belief_fields,
                mixture_weight_logits,
                opponent_hand_type,
                delta_q,
                safety_residual,
            )
        };
        HydraOutput {
            policy_logits: base.policy_logits,
            value: base.value,
            score_pdf: base.score_pdf,
            score_cdf: base.score_cdf,
            opp_tenpai: base.opp_tenpai,
            grp: base.grp,
            opp_next_discard,
            danger,
            oracle_critic,
            belief_fields,
            mixture_weight_logits,
            opponent_hand_type,
            delta_q,
            safety_residual,
        }
    }
}
