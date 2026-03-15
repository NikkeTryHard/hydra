//! Full HydraModel combining backbone and all output heads.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};

use crate::backbone::{SEResNet, SEResNetConfig};
use crate::config::INPUT_CHANNELS;
use crate::heads::*;

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

impl<B: Backend> HydraOutput<B> {
    pub fn masked_policy(&self, legal_mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
        self.policy_logits.clone() + neg_inf
    }

    pub fn policy_logits_cpu(&self) -> Option<Vec<f32>> {
        self.policy_logits
            .to_data()
            .as_slice::<f32>()
            .ok()
            .map(|s| s.to_vec())
    }

    pub fn value_scalar(&self) -> Option<f32> {
        self.value
            .to_data()
            .as_slice::<f32>()
            .ok()
            .and_then(|s| s.first().copied())
    }

    pub fn is_finite(&self) -> bool {
        let check2 = |t: &Tensor<B, 2>| -> bool {
            if let Ok(s) = t.to_data().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        let check3 = |t: &Tensor<B, 3>| -> bool {
            if let Ok(s) = t.to_data().as_slice::<f32>() {
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

#[derive(Config, Debug)]
pub struct HydraModelConfig {
    pub num_blocks: usize,
    #[config(default = "192")]
    pub input_channels: usize,
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
    #[config(default = "46")]
    pub action_space: usize,
    #[config(default = "64")]
    pub score_bins: usize,
    #[config(default = "3")]
    pub num_opponents: usize,
    #[config(default = "24")]
    pub grp_classes: usize,
    #[config(default = "4")]
    pub num_belief_components: usize,
    #[config(default = "8")]
    pub opponent_hand_type_classes: usize,
}

impl HydraModelConfig {
    pub fn summary(&self) -> String {
        let kind = if self.num_blocks <= 12 {
            "actor"
        } else {
            "learner"
        };
        format!(
            "{}(blocks={}, ch={})",
            kind, self.num_blocks, self.hidden_channels
        )
    }

    pub fn is_actor(&self) -> bool {
        self.num_blocks == 12
    }
    pub fn is_learner(&self) -> bool {
        self.num_blocks == 24
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
            return Err("hidden_channels must be divisible by num_groups");
        }
        if self.num_blocks == 0 {
            return Err("num_blocks must be > 0");
        }
        if self.se_bottleneck == 0 {
            return Err("se_bottleneck must be > 0");
        }
        if self.num_belief_components == 0 {
            return Err("num_belief_components must be > 0");
        }
        if self.opponent_hand_type_classes == 0 {
            return Err("opponent_hand_type_classes must be > 0");
        }
        Ok(())
    }

    pub fn actor() -> Self {
        Self::new(12).with_input_channels(INPUT_CHANNELS)
    }

    pub fn estimated_params(&self) -> usize {
        let h = self.hidden_channels;
        let se_b = self.se_bottleneck;
        let input_conv = self.input_channels * h * 3 + h;
        let gn = h * 2;
        let block = (h * h * 3 + h) * 2 + gn * 2 + (h * se_b + se_b) + (se_b * h + h);
        let backbone = input_conv + gn + block * self.num_blocks + gn;
        let policy = h * self.action_space + self.action_space;
        let value = h + 1;
        let score = (h * self.score_bins + self.score_bins) * 2;
        let tenpai = h * self.num_opponents + self.num_opponents;
        let grp = h * self.grp_classes + self.grp_classes;
        let opp_next = h * self.num_opponents + self.num_opponents;
        let danger = h * self.num_opponents + self.num_opponents;
        let oracle = h * 4 + 4;
        let belief_field = h * (self.num_belief_components * 4) + (self.num_belief_components * 4);
        let mixture_weight = h * self.num_belief_components + self.num_belief_components;
        let opponent_hand_type = h * (self.num_opponents * self.opponent_hand_type_classes)
            + (self.num_opponents * self.opponent_hand_type_classes);
        let delta_q = h * self.action_space + self.action_space;
        let safety_residual = h * self.action_space + self.action_space;
        backbone
            + policy
            + value
            + score
            + tenpai
            + grp
            + opp_next
            + danger
            + oracle
            + belief_field
            + mixture_weight
            + opponent_hand_type
            + delta_q
            + safety_residual
    }

    pub fn learner() -> Self {
        Self::new(24).with_input_channels(INPUT_CHANNELS)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> HydraModel<B> {
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
        device: &B::Device,
    ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_vec = policy_logits
            .to_data()
            .as_slice::<f32>()
            .expect("policy logits extraction failed")
            .to_vec();
        let logits: [f32; HYDRA_ACTION_SPACE] = logits_vec
            .try_into()
            .expect("policy logits length mismatch");
        let value_scalar = value
            .to_data()
            .as_slice::<f32>()
            .expect("value extraction failed")[0];
        (logits, value_scalar)
    }

    /// Batch inference using a caller-provided flat buffer to avoid
    /// per-call allocation. The buffer is cleared and reused each call.
    pub fn batch_policy_value_cpu_reuse(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &B::Device,
        flat_buf: &mut Vec<f32>,
    ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
        if observations.is_empty() {
            return Vec::new();
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
        let logits_flat = policy_logits
            .to_data()
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed")
            .to_vec();
        let values_flat = value
            .to_data()
            .as_slice::<f32>()
            .expect("batch value extraction failed")
            .to_vec();

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

    /// Runs a batch of observations through the full model and returns
    /// per-sample policy logits and value scalars on the CPU.
    ///
    /// This amortizes GPU kernel launch overhead across N samples. The
    /// input observations are concatenated into a single `[N, C, T]` tensor
    /// for one forward pass, then results are sliced per sample.
    pub fn batch_policy_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &B::Device,
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
        let logits_flat = policy_logits
            .to_data()
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed")
            .to_vec();
        let values_flat = value
            .to_data()
            .as_slice::<f32>()
            .expect("batch value extraction failed")
            .to_vec();

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
    pub fn forward_policy_value(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (_, pooled) = self.backbone.forward(x);
        let policy_logits = self.policy.forward(pooled.clone());
        let value = self.value.forward(pooled);
        (policy_logits, value)
    }

    /// Forward pass that detaches outputs of zero-weight heads.
    ///
    /// All heads still run their forward pass (shapes must match), but
    /// heads with zero loss weight have their outputs detached from the
    /// autograd graph. This prevents gradient computation and reduces
    /// VRAM usage for activations that won't contribute to the loss.
    pub fn forward_active(
        &self,
        x: Tensor<B, 3>,
        loss_cfg: &crate::training::losses::HydraLossConfig,
    ) -> HydraOutput<B> {
        let (spatial, pooled) = self.backbone.forward(x);
        let oracle_input = pooled.clone().detach();

        let policy_logits = self.policy.forward(pooled.clone());
        let value = self.value.forward(pooled.clone());
        let score_pdf = self.score_pdf.forward(pooled.clone());
        let score_cdf = self.score_cdf.forward(pooled.clone());
        let opp_tenpai = self.opp_tenpai.forward(pooled.clone());
        let grp = self.grp.forward(pooled.clone());
        let opp_next_discard = self.opp_next_discard.forward(spatial.clone());
        let danger = self.danger.forward(spatial.clone());
        let oracle_critic = self.oracle_critic.forward(oracle_input);
        let belief_fields = self.belief_field.forward(spatial);
        let mixture_weight_logits = self.mixture_weight.forward(pooled.clone());
        let opponent_hand_type = self.opponent_hand_type.forward(pooled.clone());
        let delta_q = self.delta_q.forward(pooled.clone());
        let safety_residual = self.safety_residual.forward(pooled);

        HydraOutput {
            policy_logits,
            value,
            score_pdf: if loss_cfg.w_score > 0.0 {
                score_pdf
            } else {
                score_pdf.detach()
            },
            score_cdf: if loss_cfg.w_score > 0.0 {
                score_cdf
            } else {
                score_cdf.detach()
            },
            opp_tenpai: if loss_cfg.w_tenpai > 0.0 {
                opp_tenpai
            } else {
                opp_tenpai.detach()
            },
            grp: if loss_cfg.w_grp > 0.0 {
                grp
            } else {
                grp.detach()
            },
            opp_next_discard: if loss_cfg.w_opp > 0.0 {
                opp_next_discard
            } else {
                opp_next_discard.detach()
            },
            danger: if loss_cfg.w_danger > 0.0 {
                danger
            } else {
                danger.detach()
            },
            oracle_critic: if loss_cfg.w_oracle_critic > 0.0 {
                oracle_critic
            } else {
                oracle_critic.detach()
            },
            belief_fields: if loss_cfg.w_belief_fields > 0.0 {
                belief_fields
            } else {
                belief_fields.detach()
            },
            mixture_weight_logits: if loss_cfg.w_mixture_weight > 0.0 {
                mixture_weight_logits
            } else {
                mixture_weight_logits.detach()
            },
            opponent_hand_type: if loss_cfg.w_opponent_hand_type > 0.0 {
                opponent_hand_type
            } else {
                opponent_hand_type.detach()
            },
            delta_q: if loss_cfg.w_delta_q > 0.0 {
                delta_q
            } else {
                delta_q.detach()
            },
            safety_residual: if loss_cfg.w_safety_residual > 0.0 {
                safety_residual
            } else {
                safety_residual.detach()
            },
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
        let (spatial, pooled) = self.backbone.forward(x);
        let oracle_input = pooled.clone().detach();
        HydraOutput {
            policy_logits: self.policy.forward(pooled.clone()),
            value: self.value.forward(pooled.clone()),
            score_pdf: self.score_pdf.forward(pooled.clone()),
            score_cdf: self.score_cdf.forward(pooled.clone()),
            opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
            grp: self.grp.forward(pooled.clone()),
            opp_next_discard: self.opp_next_discard.forward(spatial.clone()),
            danger: self.danger.forward(spatial.clone()),
            oracle_critic: self.oracle_critic.forward(oracle_input),
            belief_fields: self.belief_field.forward(spatial),
            mixture_weight_logits: self.mixture_weight.forward(pooled.clone()),
            opponent_hand_type: self.opponent_hand_type.forward(pooled.clone()),
            delta_q: self.delta_q.forward(pooled.clone()),
            safety_residual: self.safety_residual.forward(pooled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;

    type B = NdArray<f32>;
    type AB = Autodiff<NdArray<f32>>;

    fn assert_output_shapes(out: &HydraOutput<B>, batch: usize) {
        assert_eq!(out.policy_logits.dims(), [batch, 46]);
        assert_eq!(out.value.dims(), [batch, 1]);
        assert_eq!(out.score_pdf.dims(), [batch, 64]);
        assert_eq!(out.score_cdf.dims(), [batch, 64]);
        assert_eq!(out.opp_tenpai.dims(), [batch, 3]);
        assert_eq!(out.grp.dims(), [batch, 24]);
        assert_eq!(out.opp_next_discard.dims(), [batch, 3, 34]);
        assert_eq!(out.danger.dims(), [batch, 3, 34]);
        assert_eq!(out.oracle_critic.dims(), [batch, 4]);
        assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
        assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
        assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
        assert_eq!(out.delta_q.dims(), [batch, 46]);
        assert_eq!(out.safety_residual.dims(), [batch, 46]);
    }

    #[test]
    fn actor_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, INPUT_CHANNELS, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 4);
    }

    #[test]
    fn learner_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::learner().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, INPUT_CHANNELS, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 2);
    }

    #[test]
    fn value_head_bounded() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = model.forward(x);
        let data = out.value.to_data();
        for &v in data.as_slice::<f32>().expect("f32") {
            assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
        }
    }

    #[test]
    fn policy_value_cpu_returns_correct_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let obs = [0.0f32; OBS_SIZE];
        let (logits, value) = model.policy_value_cpu(&obs, &device);
        assert_eq!(logits.len(), HYDRA_ACTION_SPACE);
        assert!(value.is_finite());
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn actor_and_learner_param_counts_differ() {
        let device = Default::default();
        let actor = HydraModelConfig::actor().init::<B>(&device);
        let learner = HydraModelConfig::learner().init::<B>(&device);
        let a_params = actor.num_params();
        let l_params = learner.num_params();
        assert!(
            l_params > a_params,
            "learner ({l_params}) should have more params than actor ({a_params})"
        );
        assert!(
            a_params > 1_000_000,
            "actor should have >1M params, got {a_params}"
        );
        assert!(
            l_params > 5_000_000,
            "learner should have >5M params, got {l_params}"
        );
    }

    #[test]
    fn all_outputs_finite_for_random_input() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [8, INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = model.forward(x);
        let check = |t: &Tensor<B, 2>, name: &str| {
            let d = t.to_data();
            for &v in d.as_slice::<f32>().expect("f32") {
                assert!(v.is_finite(), "{name} has non-finite: {v}");
            }
        };
        let check_spatial = |t: &Tensor<B, 3>, name: &str| {
            let d = t.to_data();
            for &v in d.as_slice::<f32>().expect("f32") {
                assert!(v.is_finite(), "{name} has non-finite: {v}");
            }
        };
        check(&out.policy_logits, "policy");
        check(&out.value, "value");
        check(&out.score_pdf, "score_pdf");
        check(&out.score_cdf, "score_cdf");
        check(&out.opp_tenpai, "opp_tenpai");
        check(&out.grp, "grp");
        check(&out.oracle_critic, "oracle_critic");
        check_spatial(&out.opp_next_discard, "opp_next_discard");
        check_spatial(&out.danger, "danger");
        check_spatial(&out.belief_fields, "belief_fields");
        check(&out.mixture_weight_logits, "mixture_weight_logits");
        check(&out.opponent_hand_type, "opponent_hand_type");
        check(&out.delta_q, "delta_q");
        check(&out.safety_residual, "safety_residual");
    }

    #[test]
    fn oracle_head_does_not_backprop_to_backbone_input() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<AB>(&device);
        let x = Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device).require_grad();
        let out = model.forward(x.clone());
        let target = Tensor::<AB, 2>::ones([2, 4], &device);
        let diff = out.oracle_critic - target;
        let loss = (diff.clone() * diff).mean();
        let grads = loss.backward();

        assert!(
            x.grad(&grads).is_none(),
            "oracle-only loss must not backpropagate through the shared backbone"
        );
    }

    #[test]
    fn model_config_actor_learner_defaults() {
        let actor = HydraModelConfig::actor();
        assert_eq!(actor.num_blocks, 12);
        assert_eq!(actor.hidden_channels, 256);
        assert_eq!(actor.num_groups, 32);
        let learner = HydraModelConfig::learner();
        assert_eq!(learner.num_blocks, 24);
        assert_eq!(learner.hidden_channels, 256);
    }

    #[test]
    fn validate_passes_for_standard_configs() {
        assert!(HydraModelConfig::actor().validate().is_ok());
        assert!(HydraModelConfig::learner().validate().is_ok());
    }
}
