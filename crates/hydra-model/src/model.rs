//! Full HydraModel combining backbone and all output heads.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};
use hydra_train_types::config::ModelShapeConfig;

use crate::backbone::{SEResNet, SEResNetConfig};
use crate::heads::*;
use crate::profiling;

const MODEL_SCOPE_BACKBONE: &str = "model_backbone";
const MODEL_SCOPE_HEADS_POLICY: &str = "model_heads_policy";
const MODEL_SCOPE_HEADS_VALUE: &str = "model_heads_value";
const MODEL_SCOPE_HEADS_LINEAR_BASE: &str = "model_heads_linear_base";
const MODEL_SCOPE_HEADS_SPATIAL_BASE: &str = "model_heads_spatial_base";
const MODEL_SCOPE_HEADS_ADVANCED: &str = "model_heads_advanced";

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

pub type ActorNet<B> = HydraModel<B>;
pub type LearnerNet<B> = HydraModel<B>;

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

fn zero_linear_head<B: Backend>(
    batch: usize,
    width: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::zeros([batch, width], device)
}

fn zero_spatial_head<B: Backend>(
    batch: usize,
    channels: usize,
    width: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::zeros([batch, channels, width], device)
}

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

impl HydraModelInit for ModelShapeConfig {
    fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> HydraModel<B> {
        self.validate().expect("invalid Hydra model shape config");
        let backbone_cfg = SEResNetConfig::new(
            self.num_blocks,
            self.input_channels,
            self.hidden_channels,
            self.num_groups,
            self.se_bottleneck,
        );
        let heads_cfg = HeadsConfig::new()
            .with_hidden_channels(self.hidden_channels)
            .with_action_space(self.action_space)
            .with_score_bins(self.score_bins)
            .with_num_opponents(self.num_opponents)
            .with_grp_classes(self.grp_classes)
            .with_num_belief_components(self.num_belief_components)
            .with_opponent_hand_type_classes(self.opponent_hand_type_classes);
        HydraModel {
            backbone: backbone_cfg.init(device),
            policy: heads_cfg.init_policy(device),
            value: heads_cfg.init_value(device),
            score_pdf: heads_cfg.init_score_pdf(device),
            score_cdf: heads_cfg.init_score_cdf(device),
            opp_tenpai: heads_cfg.init_opp_tenpai(device),
            grp: heads_cfg.init_grp(device),
            opp_next_discard: heads_cfg.init_opp_next_discard(device),
            danger: heads_cfg.init_danger(device),
            oracle_critic: heads_cfg.init_oracle_critic(device),
            belief_field: heads_cfg.init_belief_field(device),
            mixture_weight: heads_cfg.init_mixture_weight(device),
            opponent_hand_type: heads_cfg.init_opponent_hand_type(device),
            delta_q: heads_cfg.init_delta_q(device),
            safety_residual: heads_cfg.init_safety_residual(device),
        }
    }
}

impl<B: Backend> HydraModel<B> {
    pub fn policy_logits_for(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (_, pooled) = self.backbone.forward(x);
        self.policy.forward(pooled)
    }

    /// Runs a single observation through the full model and returns policy
    /// logits and value scalar on the CPU.
    ///
    /// This is the adapter used by the live ExIt producer during self-play.
    /// It performs a single-sample forward pass, extracts the policy logits
    /// as a fixed-size array and the value head output as a scalar.
    ///
    /// # Panics
    ///
    /// Panics if the forward pass produces non-extractable tensor data.
    pub fn policy_value_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("policy logits extraction failed");
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(&logits_slice[..HYDRA_ACTION_SPACE]);
        let value_scalar = value
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("value extraction failed")[0];
        (logits, value_scalar)
    }

    pub fn policy_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> [f32; HYDRA_ACTION_SPACE] {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let policy_logits = self.forward_policy(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("policy logits extraction failed");
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(&logits_slice[..HYDRA_ACTION_SPACE]);
        logits
    }

    pub fn value_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> f32 {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let value = self.forward_value(input);
        value
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("value extraction failed")[0]
    }

    pub fn policy_and_value_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
        self.policy_value_cpu(obs, device)
    }

    /// Batch inference using a caller-provided flat buffer to avoid
    /// per-call allocation. The buffer is cleared and reused each call.
    pub fn fill_batch_policy_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        outputs: &mut Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
    ) {
        if observations.is_empty() {
            outputs.clear();
            return;
        }
        let n = observations.len();
        flat_buf.clear();
        flat_buf.reserve(n * OBS_SIZE);
        for obs in observations {
            flat_buf.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_flat = logits_data
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed");
        let values_data = value.to_data().convert::<f32>();
        let values_flat = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");

        outputs.clear();
        outputs.reserve(n);
        for (i, &value) in values_flat.iter().enumerate().take(n) {
            let logits_start = i * HYDRA_ACTION_SPACE;
            let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
                [logits_start..logits_start + HYDRA_ACTION_SPACE]
                .try_into()
                .expect("logits slice length mismatch");
            outputs.push((logits, value));
        }
    }

    pub fn batch_policy_value_cpu_reuse(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        outputs: &mut Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
    ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
        self.fill_batch_policy_value_cpu(observations, device, flat_buf, outputs);
        std::mem::take(outputs)
    }

    pub fn fill_batch_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        values_out: &mut Vec<f32>,
    ) {
        if observations.is_empty() {
            values_out.clear();
            return;
        }
        let n = observations.len();
        flat_buf.clear();
        flat_buf.reserve(n * OBS_SIZE);
        for obs in observations {
            flat_buf.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let value = self.forward_value(input);
        let values_data = value.to_data().convert::<f32>();
        let values = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");
        values_out.clear();
        values_out.extend_from_slice(values);
    }

    pub fn batch_value_cpu_reuse(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        values_out: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.fill_batch_value_cpu(observations, device, flat_buf, values_out);
        std::mem::take(values_out)
    }

    /// Runs a batch of observations through the full model and returns
    /// per-sample policy logits and value scalars on the CPU.
    ///
    /// This amortizes GPU kernel launch overhead across N samples. The
    /// input observations are concatenated into a single `[N, C, T]` tensor
    /// for one forward pass, then results are sliced per sample.
    pub fn batch_policy_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
        if observations.is_empty() {
            return Vec::new();
        }
        let n = observations.len();
        let mut flat = Vec::with_capacity(n * OBS_SIZE);
        for obs in observations {
            flat.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_flat = logits_data
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed");
        let values_data = value.to_data().convert::<f32>();
        let values_flat = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");

        (0..n)
            .map(|i| {
                let logits_start = i * HYDRA_ACTION_SPACE;
                let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
                    [logits_start..logits_start + HYDRA_ACTION_SPACE]
                    .try_into()
                    .expect("logits slice length mismatch");
                let value = values_flat[i];
                (logits, value)
            })
            .collect()
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
        let (_, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
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

    pub fn forward_with_warmup(
        &self,
        x: Tensor<B, 3>,
        policy: &HydraForwardPolicy,
        warmup_heads: &[ModelAdvancedHead],
    ) -> HydraOutput<B> {
        let (spatial, pooled) = {
            let _backbone_scope = profiling::scope(MODEL_SCOPE_BACKBONE);
            self.backbone.forward(x)
        };
        let oracle_input = pooled.clone().detach();
        let is_warmup = |head: ModelAdvancedHead| warmup_heads.contains(&head);
        let batch = pooled.dims()[0];
        let device = pooled.device();

        let (policy_logits, value, score_pdf, score_cdf, opp_tenpai, grp) = {
            let _linear_base_scope = profiling::scope(MODEL_SCOPE_HEADS_LINEAR_BASE);
            let policy_logits = self.policy.forward(pooled.clone());
            let value = self.value.forward(pooled.clone());
            let score_pdf = self.score_pdf.forward(pooled.clone());
            let score_cdf = self.score_cdf.forward(pooled.clone());
            let opp_tenpai = self.opp_tenpai.forward(pooled.clone());
            let grp = self.grp.forward(pooled.clone());
            (policy_logits, value, score_pdf, score_cdf, opp_tenpai, grp)
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
        drop(_advanced_scope);

        HydraOutput {
            policy_logits,
            value,
            score_pdf: if policy.w_score > 0.0 {
                score_pdf
            } else {
                score_pdf.detach()
            },
            score_cdf: if policy.w_score > 0.0 {
                score_cdf
            } else {
                score_cdf.detach()
            },
            opp_tenpai: if policy.w_tenpai > 0.0 {
                opp_tenpai
            } else {
                opp_tenpai.detach()
            },
            grp: if policy.w_grp > 0.0 {
                grp
            } else {
                grp.detach()
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
        let (policy_logits, value, score_pdf, score_cdf, opp_tenpai, grp) = {
            let _linear_base_scope = profiling::scope(MODEL_SCOPE_HEADS_LINEAR_BASE);
            let policy_logits = self.policy.forward(pooled.clone());
            let value = self.value.forward(pooled.clone());
            let score_pdf = self.score_pdf.forward(pooled.clone());
            let score_cdf = self.score_cdf.forward(pooled.clone());
            let opp_tenpai = self.opp_tenpai.forward(pooled.clone());
            let grp = self.grp.forward(pooled.clone());
            (policy_logits, value, score_pdf, score_cdf, opp_tenpai, grp)
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
            policy_logits,
            value,
            score_pdf,
            score_cdf,
            opp_tenpai,
            grp,
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

#[cfg(test)]
mod tests;
