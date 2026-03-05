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
    pub fn actor() -> Self {
        Self::new(12)
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
}
