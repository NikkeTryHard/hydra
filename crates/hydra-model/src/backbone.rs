//! SE-ResNet backbone: SEBlock, SEResBlock, and SEResNet.

use hydra_train_types::config::{BackboneActivationConfig, BackboneNormConfig};

use crate::native_group_norm_mish;
use crate::profiling;
use burn::nn::{
    GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::prelude::*;
use burn::tensor::activation;

const BACKBONE_SCOPE_STEM: &str = "backbone_stem";
const BACKBONE_SCOPE_BLOCKS: &str = "backbone_blocks";
const BACKBONE_SCOPE_TAIL: &str = "backbone_tail";
const BACKBONE_SCOPE_BLOCK_CONV: &str = "backbone_block_conv";
const BACKBONE_SCOPE_BLOCK_SE: &str = "backbone_block_se";
const BACKBONE_SCOPE_BLOCK_ADD: &str = "backbone_block_add";

fn apply_activation<B: Backend, const D: usize>(
    activation_kind: BackboneActivationConfig,
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    match activation_kind {
        BackboneActivationConfig::Mish => activation::mish(x),
        BackboneActivationConfig::Silu => x.clone().mul(activation::sigmoid(x)),
        BackboneActivationConfig::Relu => activation::relu(x),
    }
}

fn apply_norm_activation<B: Backend>(
    norm: &GroupNorm<B>,
    activation_kind: BackboneActivationConfig,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    match activation_kind {
        BackboneActivationConfig::Mish => native_group_norm_mish::group_norm_mish(norm, x),
        BackboneActivationConfig::Silu | BackboneActivationConfig::Relu => {
            apply_activation(activation_kind, norm.forward(x))
        }
    }
}
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
        let scale = apply_activation(BackboneActivationConfig::Mish, self.fc1.forward(scale));
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
    /// Backbone activation profile.
    #[config(default = "BackboneActivationConfig::default()")]
    pub activation: BackboneActivationConfig,
    /// Per-block normalization layout.
    #[config(default = "BackboneNormConfig::default()")]
    pub norm: BackboneNormConfig,
    /// Whether this residual block applies SE after its second convolution.
    #[config(default = "true")]
    pub use_se: bool,
}

/// Residual 1D convolution block with group norm and SE gating.
#[derive(Module, Debug)]
pub struct SEResBlock<B: Backend> {
    gn1: GroupNorm<B>,
    conv1: Conv1d<B>,
    gn2: GroupNorm<B>,
    conv2: Conv1d<B>,
    se: SEBlock<B>,
    activation: BackboneActivationConfig,
    norm: BackboneNormConfig,
    use_se: bool,
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
        let use_se = self.use_se;
        let se_cfg = SEBlockConfig::new(self.channels, self.se_bottleneck);
        SEResBlock {
            gn1: gn_cfg.init(device),
            conv1: conv_cfg.init(device),
            gn2: GroupNormConfig::new(self.num_groups, self.channels).init(device),
            conv2: Conv1dConfig::new(self.channels, self.channels, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            se: se_cfg.init(device),
            activation: self.activation,
            norm: self.norm,
            use_se,
        }
    }
}

impl<B: Backend> SEResBlock<B> {
    /// Apply the residual block to a `[batch, channels, tiles]` tensor.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let out = {
            let _conv_scope = profiling::scope(BACKBONE_SCOPE_BLOCK_CONV);
            let out = apply_norm_activation(&self.gn1, self.activation, x);
            let out = self.conv1.forward(out);
            let out = match self.norm {
                BackboneNormConfig::Both => apply_norm_activation(&self.gn2, self.activation, out),
                BackboneNormConfig::FirstOnly => apply_activation(self.activation, out),
            };
            self.conv2.forward(out)
        };
        let out = if self.use_se {
            let _se_scope = profiling::scope(BACKBONE_SCOPE_BLOCK_SE);
            self.se.forward(out)
        } else {
            out
        };
        let _add_scope = profiling::scope(BACKBONE_SCOPE_BLOCK_ADD);
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
    /// Backbone activation profile.
    #[config(default = "BackboneActivationConfig::default()")]
    pub activation: BackboneActivationConfig,
    /// Apply SE on every Nth residual block.
    #[config(default = "1")]
    pub se_every_n: usize,
    /// Per-block normalization layout.
    #[config(default = "BackboneNormConfig::default()")]
    pub norm: BackboneNormConfig,
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
        if self.se_every_n == 0 {
            return Err("se_every_n > 0");
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
    activation: BackboneActivationConfig,
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
            SEResBlockConfig::new(self.hidden_channels, self.num_groups, self.se_bottleneck)
                .with_activation(self.activation)
                .with_norm(self.norm);
        let blocks = (0..self.num_blocks)
            .map(|idx| {
                block_cfg
                    .clone()
                    .with_use_se((idx + 1) % self.se_every_n == 0)
                    .init(device)
            })
            .collect();
        let final_gn = GroupNormConfig::new(self.num_groups, self.hidden_channels).init(device);
        SEResNet {
            input_conv,
            input_gn,
            blocks,
            final_gn,
            activation: self.activation,
        }
    }
}

impl<B: Backend> SEResNet<B> {
    /// Run the backbone and return `(spatial, pooled)` features.
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let spatial = self.forward_spatial(x);
        let pooled = spatial.clone().mean_dim(2).squeeze_dim::<2>(2);
        (spatial, pooled)
    }

    /// Run the backbone and return pooled features only.
    pub fn forward_pooled(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward_spatial(x).mean_dim(2).squeeze_dim::<2>(2)
    }

    fn forward_spatial(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = {
            let _stem_scope = profiling::scope(BACKBONE_SCOPE_STEM);
            let x = self.input_conv.forward(x);
            apply_norm_activation(&self.input_gn, self.activation, x)
        };
        let x = {
            let _blocks_scope = profiling::scope(BACKBONE_SCOPE_BLOCKS);
            self.blocks.iter().fold(x, |acc, block| block.forward(acc))
        };
        let _tail_scope = profiling::scope(BACKBONE_SCOPE_TAIL);
        apply_norm_activation(&self.final_gn, self.activation, x)
    }
}

#[cfg(test)]
mod tests;
