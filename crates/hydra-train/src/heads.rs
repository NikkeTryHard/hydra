//! Output heads: 8 inference heads + 1 oracle critic.

use burn::nn::{
    Linear, LinearConfig,
    conv::{Conv1d, Conv1dConfig},
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled).tanh()
    }
}

#[derive(Module, Debug)]
pub struct ScorePdfHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> ScorePdfHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct ScoreCdfHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> ScoreCdfHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct OppTenpaiHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> OppTenpaiHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct GrpHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> GrpHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct OppNextDiscardHead<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> OppNextDiscardHead<B> {
    pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(spatial)
    }
}

#[derive(Module, Debug)]
pub struct DangerHead<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> DangerHead<B> {
    pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(spatial)
    }
}

#[derive(Module, Debug)]
pub struct OracleCriticHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> OracleCriticHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct BeliefFieldHead<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> BeliefFieldHead<B> {
    pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(spatial)
    }
}

#[derive(Module, Debug)]
pub struct MixtureWeightHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> MixtureWeightHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct OpponentHandTypeHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> OpponentHandTypeHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct DeltaQHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> DeltaQHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct SafetyResidualHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> SafetyResidualHead<B> {
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

impl HeadsConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.hidden_channels == 0 {
            return Err("hidden_channels must be > 0");
        }
        if self.action_space == 0 {
            return Err("action_space must be > 0");
        }
        if self.num_belief_components == 0 {
            return Err("num_belief_components must be > 0");
        }
        if self.opponent_hand_type_classes == 0 {
            return Err("opponent_hand_type_classes must be > 0");
        }
        Ok(())
    }
}

#[derive(Config, Debug)]
pub struct HeadsConfig {
    #[config(default = "256")]
    pub hidden_channels: usize,
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

impl HeadsConfig {
    pub fn init_policy<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
        }
    }

    pub fn init_value<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            linear: LinearConfig::new(self.hidden_channels, 1).init(device),
        }
    }

    pub fn init_score_pdf<B: Backend>(&self, device: &B::Device) -> ScorePdfHead<B> {
        ScorePdfHead {
            linear: LinearConfig::new(self.hidden_channels, self.score_bins).init(device),
        }
    }

    pub fn init_score_cdf<B: Backend>(&self, device: &B::Device) -> ScoreCdfHead<B> {
        ScoreCdfHead {
            linear: LinearConfig::new(self.hidden_channels, self.score_bins).init(device),
        }
    }

    pub fn init_opp_tenpai<B: Backend>(&self, device: &B::Device) -> OppTenpaiHead<B> {
        OppTenpaiHead {
            linear: LinearConfig::new(self.hidden_channels, self.num_opponents).init(device),
        }
    }

    pub fn init_grp<B: Backend>(&self, device: &B::Device) -> GrpHead<B> {
        GrpHead {
            linear: LinearConfig::new(self.hidden_channels, self.grp_classes).init(device),
        }
    }

    pub fn init_opp_next_discard<B: Backend>(&self, device: &B::Device) -> OppNextDiscardHead<B> {
        OppNextDiscardHead {
            conv: Conv1dConfig::new(self.hidden_channels, self.num_opponents, 1).init(device),
        }
    }

    pub fn init_danger<B: Backend>(&self, device: &B::Device) -> DangerHead<B> {
        DangerHead {
            conv: Conv1dConfig::new(self.hidden_channels, self.num_opponents, 1).init(device),
        }
    }

    pub fn init_oracle_critic<B: Backend>(&self, device: &B::Device) -> OracleCriticHead<B> {
        OracleCriticHead {
            linear: LinearConfig::new(self.hidden_channels, 4).init(device),
        }
    }

    pub fn init_belief_field<B: Backend>(&self, device: &B::Device) -> BeliefFieldHead<B> {
        BeliefFieldHead {
            conv: Conv1dConfig::new(self.hidden_channels, self.num_belief_components * 4, 1)
                .init(device),
        }
    }

    pub fn init_mixture_weight<B: Backend>(&self, device: &B::Device) -> MixtureWeightHead<B> {
        MixtureWeightHead {
            linear: LinearConfig::new(self.hidden_channels, self.num_belief_components)
                .init(device),
        }
    }

    pub fn init_opponent_hand_type<B: Backend>(
        &self,
        device: &B::Device,
    ) -> OpponentHandTypeHead<B> {
        OpponentHandTypeHead {
            linear: LinearConfig::new(
                self.hidden_channels,
                self.num_opponents * self.opponent_hand_type_classes,
            )
            .init(device),
        }
    }

    pub fn init_delta_q<B: Backend>(&self, device: &B::Device) -> DeltaQHead<B> {
        DeltaQHead {
            linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
        }
    }

    pub fn init_safety_residual<B: Backend>(&self, device: &B::Device) -> SafetyResidualHead<B> {
        SafetyResidualHead {
            linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn cfg() -> HeadsConfig {
        HeadsConfig::new()
    }

    #[test]
    fn policy_head_shape() {
        let device = Default::default();
        let head = cfg().init_policy::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 46]);
    }

    #[test]
    fn value_head_shape_and_range() {
        let device = Default::default();
        let head = cfg().init_value::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [4, 256],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = head.forward(x);
        assert_eq!(out.dims(), [4, 1]);
        let data = out.to_data();
        for &v in data.as_slice::<f32>().expect("f32 slice") {
            assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
        }
    }

    #[test]
    fn score_pdf_head_shape() {
        let device = Default::default();
        let head = cfg().init_score_pdf::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 64]);
    }

    #[test]
    fn score_cdf_head_shape() {
        let device = Default::default();
        let head = cfg().init_score_cdf::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 64]);
    }

    #[test]
    fn opp_tenpai_head_shape() {
        let device = Default::default();
        let head = cfg().init_opp_tenpai::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 3]);
    }

    #[test]
    fn grp_head_shape() {
        let device = Default::default();
        let head = cfg().init_grp::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 24]);
    }

    #[test]
    fn opp_next_discard_head_shape() {
        let device = Default::default();
        let head = cfg().init_opp_next_discard::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        assert_eq!(head.forward(x).dims(), [4, 3, 34]);
    }

    #[test]
    fn danger_head_shape() {
        let device = Default::default();
        let head = cfg().init_danger::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        assert_eq!(head.forward(x).dims(), [4, 3, 34]);
    }

    #[test]
    fn oracle_critic_head_shape() {
        let device = Default::default();
        let head = cfg().init_oracle_critic::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 4]);
    }

    #[test]
    fn belief_field_head_shape() {
        let device = Default::default();
        let head = cfg().init_belief_field::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        assert_eq!(head.forward(x).dims(), [4, 16, 34]);
    }

    #[test]
    fn mixture_weight_head_shape() {
        let device = Default::default();
        let head = cfg().init_mixture_weight::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 4]);
    }

    #[test]
    fn opponent_hand_type_head_shape() {
        let device = Default::default();
        let head = cfg().init_opponent_hand_type::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 24]);
    }

    #[test]
    fn delta_q_head_shape() {
        let device = Default::default();
        let head = cfg().init_delta_q::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 46]);
    }

    #[test]
    fn safety_residual_head_shape() {
        let device = Default::default();
        let head = cfg().init_safety_residual::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, 256], &device);
        assert_eq!(head.forward(x).dims(), [4, 46]);
    }
}
