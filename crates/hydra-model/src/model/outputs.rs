use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;

pub(super) const POLICY_HEAD_START: usize = 0;
pub(super) const VALUE_HEAD_START: usize = POLICY_HEAD_START + HYDRA_ACTION_SPACE;
pub(super) const VALUE_HEAD_WIDTH: usize = 1;
pub(super) const SCORE_PDF_HEAD_START: usize = VALUE_HEAD_START + VALUE_HEAD_WIDTH;
pub(super) const SCORE_HEAD_WIDTH: usize = 64;
pub(super) const SCORE_CDF_HEAD_START: usize = SCORE_PDF_HEAD_START + SCORE_HEAD_WIDTH;
pub(super) const OPP_TENPAI_HEAD_START: usize = SCORE_CDF_HEAD_START + SCORE_HEAD_WIDTH;
pub(super) const OPP_TENPAI_HEAD_WIDTH: usize = 3;
pub(super) const GRP_HEAD_START: usize = OPP_TENPAI_HEAD_START + OPP_TENPAI_HEAD_WIDTH;
pub(super) const GRP_HEAD_WIDTH: usize = 24;
pub(super) const BASE_LINEAR_HEAD_WIDTH: usize = GRP_HEAD_START + GRP_HEAD_WIDTH;

/// Full-shape Burn forward output used by runtime/reference paths.
///
/// Inactive advanced heads are represented by zero tensors so callers retain a
/// stable tensor shape contract.
pub struct HydraOutput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub score_pdf: Tensor<B, 2>,
    pub score_cdf: Tensor<B, 2>,
    pub opp_tenpai: Tensor<B, 2>,
    pub grp: Tensor<B, 2>,
    pub opp_next_discard: Tensor<B, 3>,
    pub danger: Tensor<B, 3>,
    pub oracle_critic: Tensor<B, 2>,
    pub belief_fields: Tensor<B, 3>,
    pub mixture_weight_logits: Tensor<B, 2>,
    pub opponent_hand_type: Tensor<B, 2>,
    pub delta_q: Tensor<B, 2>,
    pub safety_residual: Tensor<B, 2>,
}

/// Train-only Burn forward output.
///
/// Inactive advanced heads are omitted with `None` to avoid materializing
/// unused tensors during warmup/partial-head training.
pub struct HydraTrainOutput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub score_pdf: Tensor<B, 2>,
    pub score_cdf: Tensor<B, 2>,
    pub opp_tenpai: Tensor<B, 2>,
    pub grp: Tensor<B, 2>,
    pub opp_next_discard: Tensor<B, 3>,
    pub danger: Tensor<B, 3>,
    pub oracle_critic: Option<Tensor<B, 2>>,
    pub belief_fields: Option<Tensor<B, 3>>,
    pub mixture_weight_logits: Option<Tensor<B, 2>>,
    pub opponent_hand_type: Option<Tensor<B, 2>>,
    pub delta_q: Option<Tensor<B, 2>>,
    pub safety_residual: Option<Tensor<B, 2>>,
}

pub(super) struct BaseLinearHeadOutput<B: Backend> {
    pub(super) policy_logits: Tensor<B, 2>,
    pub(super) value: Tensor<B, 2>,
    pub(super) score_pdf: Tensor<B, 2>,
    pub(super) score_cdf: Tensor<B, 2>,
    pub(super) opp_tenpai: Tensor<B, 2>,
    pub(super) grp: Tensor<B, 2>,
}

/// Advanced auxiliary heads whose warmup mode detaches them from the shared backbone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelAdvancedHead {
    /// Oracle critic head.
    OracleCritic,
    /// Belief-field spatial head.
    BeliefFields,
    /// Belief mixture-weight head.
    MixtureWeight,
    /// Opponent hand-type head.
    OpponentHandType,
    /// Delta-Q auxiliary head.
    DeltaQ,
    /// Safety residual auxiliary head.
    SafetyResidual,
}

/// Model-local forward activation policy derived by callers from their loss settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydraForwardPolicy {
    /// Score PDF/CDF heads are active when positive.
    pub w_score: f32,
    /// Opponent tenpai head weight.
    pub w_tenpai: f32,
    /// Global reward prediction head weight.
    pub w_grp: f32,
    /// Opponent spatial heads weight.
    pub w_opp: f32,
    /// Opponent danger spatial head weight.
    pub w_danger: f32,
    /// Oracle critic head weight.
    pub w_oracle_critic: f32,
    /// Belief-field head weight.
    pub w_belief_fields: f32,
    /// Mixture-weight head weight.
    pub w_mixture_weight: f32,
    /// Opponent hand-type head weight.
    pub w_opponent_hand_type: f32,
    /// Delta-Q head weight.
    pub w_delta_q: f32,
    /// Safety residual head weight.
    pub w_safety_residual: f32,
}

impl Default for HydraForwardPolicy {
    fn default() -> Self {
        Self {
            w_score: 1.0,
            w_tenpai: 1.0,
            w_grp: 1.0,
            w_opp: 1.0,
            w_danger: 1.0,
            w_oracle_critic: 0.0,
            w_belief_fields: 0.0,
            w_mixture_weight: 0.0,
            w_opponent_hand_type: 0.0,
            w_delta_q: 0.0,
            w_safety_residual: 0.0,
        }
    }
}

impl<B: Backend> HydraOutput<B> {
    pub fn masked_policy(&self, legal_mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
        self.policy_logits.clone() + neg_inf
    }

    pub fn policy_logits_cpu(&self) -> Option<Vec<f32>> {
        self.policy_logits
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .ok()
            .map(|s| s.to_vec())
    }

    pub fn value_scalar(&self) -> Option<f32> {
        self.value
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .ok()
            .and_then(|s| s.first().copied())
    }

    pub fn is_finite(&self) -> bool {
        let check2 = |t: &Tensor<B, 2>| -> bool {
            if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        let check3 = |t: &Tensor<B, 3>| -> bool {
            if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        check2(&self.policy_logits)
            && check2(&self.value)
            && check2(&self.score_pdf)
            && check2(&self.score_cdf)
            && check2(&self.opp_tenpai)
            && check2(&self.grp)
            && check2(&self.oracle_critic)
            && check3(&self.opp_next_discard)
            && check3(&self.danger)
            && check3(&self.belief_fields)
            && check2(&self.mixture_weight_logits)
            && check2(&self.opponent_hand_type)
            && check2(&self.delta_q)
            && check2(&self.safety_residual)
    }
}

pub(super) fn zero_linear_head<B: Backend>(
    batch: usize,
    width: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros([batch, width], device)
}

pub(super) fn zero_spatial_head<B: Backend>(
    batch: usize,
    channels: usize,
    width: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::zeros([batch, channels, width], device)
}
