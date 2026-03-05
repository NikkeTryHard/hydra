//! SE-ResNet backbone: SEBlock, SEResBlock, and SEResNet.

use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::prelude::*;
use burn::tensor::activation;

#[derive(Config, Debug)]
pub struct SEBlockConfig {
    pub channels: usize,
    pub bottleneck: usize,
}

#[derive(Module, Debug)]
pub struct SEBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl SEBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEBlock<B> {
        SEBlock {
            fc1: LinearConfig::new(self.channels, self.bottleneck).init(device),
            fc2: LinearConfig::new(self.bottleneck, self.channels).init(device),
        }
    }
}

impl<B: Backend> SEBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = x.clone().mean_dim(2).squeeze_dim::<2>(2);
        let scale = activation::mish(self.fc1.forward(scale));
        let scale = activation::sigmoid(self.fc2.forward(scale));
        let scale = scale.unsqueeze_dim::<3>(2);
        x.mul(scale)
    }
}

#[derive(Config, Debug)]
pub struct SEResBlockConfig {
    pub channels: usize,
    pub num_groups: usize,
    pub se_bottleneck: usize,
}

#[derive(Module, Debug)]
pub struct SEResBlock<B: Backend> {
    gn1: GroupNorm<B>,
    conv1: Conv1d<B>,
    gn2: GroupNorm<B>,
    conv2: Conv1d<B>,
    se: SEBlock<B>,
}

impl SEResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEResBlock<B> {
        let conv_cfg =
            Conv1dConfig::new(self.channels, self.channels, 3).with_padding(PaddingConfig1d::Same);
        let gn_cfg = GroupNormConfig::new(self.num_groups, self.channels);
        let se_cfg = SEBlockConfig::new(self.channels, self.se_bottleneck);
        SEResBlock {
            gn1: gn_cfg.init(device),
            conv1: conv_cfg.init(device),
            gn2: GroupNormConfig::new(self.num_groups, self.channels).init(device),
            conv2: Conv1dConfig::new(self.channels, self.channels, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            se: se_cfg.init(device),
        }
    }
}

impl<B: Backend> SEResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let out = activation::mish(self.gn1.forward(x));
        let out = self.conv1.forward(out);
        let out = activation::mish(self.gn2.forward(out));
        let out = self.conv2.forward(out);
        let out = self.se.forward(out);
        out + residual
    }
}

#[derive(Config, Debug)]
pub struct SEResNetConfig {
    pub num_blocks: usize,
    pub input_channels: usize,
    pub hidden_channels: usize,
    pub num_groups: usize,
    pub se_bottleneck: usize,
}

impl SEResNetConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_blocks == 0 {
            return Err("num_blocks > 0");
        }
        if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
            return Err("hidden_channels % num_groups != 0");
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct SEResNet<B: Backend> {
    input_conv: Conv1d<B>,
    input_gn: GroupNorm<B>,
    blocks: Vec<SEResBlock<B>>,
    final_gn: GroupNorm<B>,
}

impl SEResNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEResNet<B> {
        let input_conv = Conv1dConfig::new(self.input_channels, self.hidden_channels, 3)
            .with_padding(PaddingConfig1d::Same)
            .init(device);
        let input_gn = GroupNormConfig::new(self.num_groups, self.hidden_channels).init(device);
        let block_cfg =
            SEResBlockConfig::new(self.hidden_channels, self.num_groups, self.se_bottleneck);
        let blocks = (0..self.num_blocks)
            .map(|_| block_cfg.init(device))
            .collect();
        let final_gn = GroupNormConfig::new(self.num_groups, self.hidden_channels).init(device);
        SEResNet {
            input_conv,
            input_gn,
            blocks,
            final_gn,
        }
    }
}

impl<B: Backend> SEResNet<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let x = self.input_conv.forward(x);
        let x = activation::mish(self.input_gn.forward(x));
        let x = self.blocks.iter().fold(x, |acc, block| block.forward(acc));
        let spatial = activation::mish(self.final_gn.forward(x));
        let pooled = spatial.clone().mean_dim(2).squeeze_dim::<2>(2);
        (spatial, pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn se_block_preserves_shape() {
        let device = Default::default();
        let se = SEBlockConfig::new(256, 64).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        let out = se.forward(x);
        assert_eq!(out.dims(), [4, 256, 34]);
    }

    #[test]
    fn se_res_block_preserves_shape() {
        let device = Default::default();
        let block = SEResBlockConfig::new(256, 32, 64).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        let out = block.forward(x);
        assert_eq!(out.dims(), [4, 256, 34]);
    }

    #[test]
    fn backbone_output_shapes_12_blocks() {
        let device = Default::default();
        let cfg = SEResNetConfig::new(12, 85, 256, 32, 64);
        let net = cfg.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 85, 34], &device);
        let (spatial, pooled) = net.forward(x);
        assert_eq!(spatial.dims(), [4, 256, 34]);
        assert_eq!(pooled.dims(), [4, 256]);
    }

    #[test]
    fn backbone_output_shapes_24_blocks() {
        let device = Default::default();
        let cfg = SEResNetConfig::new(24, 85, 256, 32, 64);
        let net = cfg.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, 85, 34], &device);
        let (spatial, pooled) = net.forward(x);
        assert_eq!(spatial.dims(), [2, 256, 34]);
        assert_eq!(pooled.dims(), [2, 256]);
    }

    #[test]
    fn residual_block_output_differs_from_input() {
        let device = Default::default();
        let block = SEResBlockConfig::new(256, 32, 64).init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [1, 256, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out = block.forward(x.clone());
        let diff = (out - x).abs().mean();
        let d = diff.into_scalar().elem::<f32>();
        assert!(
            d > 1e-6,
            "residual output should differ from input: diff={d}"
        );
    }

    #[test]
    fn se_block_channel_attention() {
        let device = Default::default();
        let se = SEBlockConfig::new(4, 2).init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [1, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = se.forward(x.clone());
        assert_eq!(out.dims(), [1, 4, 8]);
        let x_data = x.to_data();
        let o_data = out.to_data();
        let xv = x_data.as_slice::<f32>().expect("f32");
        let ov = o_data.as_slice::<f32>().expect("f32");
        let mut any_diff = false;
        for i in 0..32 {
            if (xv[i] - ov[i]).abs() > 1e-6 {
                any_diff = true;
            }
        }
        assert!(any_diff, "SE should modulate channels");
    }

    #[test]
    fn se_block_output_bounded_by_input() {
        let device = Default::default();
        let se = SEBlockConfig::new(4, 2).init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [2, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 2.0),
            &device,
        );
        let out = se.forward(x.clone());
        let x_abs = x.abs().max();
        let o_abs = out.abs().max();
        let x_max: f32 = x_abs.into_scalar().elem();
        let o_max: f32 = o_abs.into_scalar().elem();
        assert!(
            o_max <= x_max + 1e-4,
            "SE output max ({o_max}) should be <= input max ({x_max}) due to sigmoid gate"
        );
    }
}
