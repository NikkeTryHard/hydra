//! SE-ResNet backbone: SEBlock, SEResBlock, and SEResNet.

use burn::nn::{
    GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::prelude::*;
use burn::tensor::activation;

/// Configuration for a squeeze-excitation block.
#[derive(Config, Debug)]
pub struct SEBlockConfig {
    /// Number of input and output channels.
    pub channels: usize,
    /// Hidden bottleneck width for channel attention.
    pub bottleneck: usize,
}

/// Squeeze-excitation channel attention block.
#[derive(Module, Debug)]
pub struct SEBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl SEBlockConfig {
    /// Initialize the block on `device`.
    pub fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> SEBlock<B> {
        SEBlock {
            fc1: LinearConfig::new(self.channels, self.bottleneck).init(device),
            fc2: LinearConfig::new(self.bottleneck, self.channels).init(device),
        }
    }
}

impl<B: Backend> SEBlock<B> {
    /// Apply channel attention to a `[batch, channels, tiles]` tensor.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale = x.clone().mean_dim(2).squeeze_dim::<2>(2);
        let scale = activation::mish(self.fc1.forward(scale));
        let scale = activation::sigmoid(self.fc2.forward(scale));
        let scale = scale.unsqueeze_dim::<3>(2);
        x.mul(scale)
    }
}

/// Configuration for a residual SE block.
#[derive(Config, Debug)]
pub struct SEResBlockConfig {
    /// Number of residual channels.
    pub channels: usize,
    /// Number of groups used by group normalization.
    pub num_groups: usize,
    /// Hidden bottleneck width for the nested SE block.
    pub se_bottleneck: usize,
}

/// Residual 1D convolution block with group norm and SE gating.
#[derive(Module, Debug)]
pub struct SEResBlock<B: Backend> {
    gn1: GroupNorm<B>,
    conv1: Conv1d<B>,
    gn2: GroupNorm<B>,
    conv2: Conv1d<B>,
    se: SEBlock<B>,
}

impl SEResBlockConfig {
    /// Initialize the residual block on `device`.
    pub fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> SEResBlock<B> {
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
    /// Apply the residual block to a `[batch, channels, tiles]` tensor.
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

/// Configuration for the SE-ResNet backbone.
#[derive(Config, Debug)]
pub struct SEResNetConfig {
    /// Number of residual blocks.
    pub num_blocks: usize,
    /// Number of input observation channels.
    pub input_channels: usize,
    /// Number of hidden backbone channels.
    pub hidden_channels: usize,
    /// Number of groups used by group normalization.
    pub num_groups: usize,
    /// Hidden bottleneck width for each SE block.
    pub se_bottleneck: usize,
}

impl SEResNetConfig {
    /// Validate structural backbone dimensions.
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

/// SE-ResNet backbone producing spatial and pooled features.
#[derive(Module, Debug)]
pub struct SEResNet<B: Backend> {
    input_conv: Conv1d<B>,
    input_gn: GroupNorm<B>,
    blocks: Vec<SEResBlock<B>>,
    final_gn: GroupNorm<B>,
}

impl SEResNetConfig {
    /// Initialize the backbone on `device`.
    pub fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> SEResNet<B> {
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
    /// Run the backbone and return `(spatial, pooled)` features.
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
mod tests;
