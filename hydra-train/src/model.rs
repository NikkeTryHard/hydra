//! Full HydraModel combining backbone and all output heads.

use burn::prelude::*;

use crate::backbone::{SEResNet, SEResNetConfig};
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
}

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
        let check = |t: &Tensor<B, 2>| -> bool {
            if let Ok(s) = t.to_data().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        check(&self.policy_logits)
            && check(&self.value)
            && check(&self.score_pdf)
            && check(&self.score_cdf)
            && check(&self.opp_tenpai)
            && check(&self.grp)
            && check(&self.oracle_critic)
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
}

#[derive(Config, Debug)]
pub struct HydraModelConfig {
    pub num_blocks: usize,
    #[config(default = "85")]
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
}

impl HydraModelConfig {
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
        Ok(())
    }

    pub fn actor() -> Self {
        Self::new(12)
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
        backbone + policy + value + score + tenpai + grp + opp_next + danger + oracle
    }

    pub fn learner() -> Self {
        Self::new(24)
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
            .with_grp_classes(self.grp_classes);
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
        }
    }
}

impl<B: Backend> HydraModel<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
        let (spatial, pooled) = self.backbone.forward(x);
        HydraOutput {
            policy_logits: self.policy.forward(pooled.clone()),
            value: self.value.forward(pooled.clone()),
            score_pdf: self.score_pdf.forward(pooled.clone()),
            score_cdf: self.score_cdf.forward(pooled.clone()),
            opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
            grp: self.grp.forward(pooled.clone()),
            opp_next_discard: self.opp_next_discard.forward(spatial.clone()),
            danger: self.danger.forward(spatial),
            oracle_critic: self.oracle_critic.forward(pooled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

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
    }

    #[test]
    fn actor_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 85, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 4);
    }

    #[test]
    fn learner_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::learner().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, 85, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 2);
    }

    #[test]
    fn value_head_bounded() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, 85, 34],
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
            [8, 85, 34],
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
        check(&out.policy_logits, "policy");
        check(&out.value, "value");
        check(&out.score_pdf, "score_pdf");
        check(&out.score_cdf, "score_cdf");
        check(&out.opp_tenpai, "opp_tenpai");
        check(&out.grp, "grp");
        check(&out.oracle_critic, "oracle_critic");
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
