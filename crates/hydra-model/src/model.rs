//! Full HydraModel combining backbone and all output heads.

mod cpu;
mod forward;
mod init;
mod outputs;

pub use outputs::{HydraForwardPolicy, HydraOutput, HydraTrainOutput, ModelAdvancedHead};

use burn::prelude::*;
use hydra_train_types::config::ModelShapeConfig;

use crate::backbone::SEResNet;
use crate::heads::*;
#[cfg(test)]
use hydra_core::action::HYDRA_ACTION_SPACE;
#[cfg(test)]
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};

const MODEL_SCOPE_BACKBONE: &str = "model_backbone";
const MODEL_SCOPE_HEADS_POLICY: &str = "model_heads_policy";
const MODEL_SCOPE_HEADS_VALUE: &str = "model_heads_value";
const MODEL_SCOPE_HEADS_LINEAR_BASE: &str = "model_heads_linear_base";
const MODEL_SCOPE_HEADS_SPATIAL_BASE: &str = "model_heads_spatial_base";
const MODEL_SCOPE_HEADS_ADVANCED: &str = "model_heads_advanced";
#[derive(Module, Debug)]
pub struct HydraModel<B: Backend> {
    backbone: SEResNet<B>,
    policy: PolicyHead<B>,
    value: ValueHead<B>,
    score_pdf: ScorePdfHead<B>,
    score_cdf: ScoreCdfHead<B>,
    opp_tenpai: OppTenpaiHead<B>,
    grp: GrpHead<B>,
    opp_next_discard: OppNextDiscardHead<B>,
    danger: DangerHead<B>,
    oracle_critic: OracleCriticHead<B>,
    belief_field: BeliefFieldHead<B>,
    mixture_weight: MixtureWeightHead<B>,
    opponent_hand_type: OpponentHandTypeHead<B>,
    delta_q: DeltaQHead<B>,
    safety_residual: SafetyResidualHead<B>,
}

pub type HydraModelConfig = ModelShapeConfig;

pub trait HydraModelInit {
    fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> HydraModel<B>;
}

#[cfg(test)]
mod tests;
