use hydra_train_types::config::{BackboneActivationConfig, BackboneNormConfig, ModelShapeConfig};

use super::cli_types::BcBackend;
use super::schema::default_backbone_se_every_n;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BcHeadProfile {
    #[default]
    Full,
    PolicyOnly,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonModelProfileConfig {
    Default,
    Balanced,
    #[default]
    Large,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonBackboneProfileConfig {
    #[default]
    Conv2dLocal3,
    TileformerBias,
    ConvnextTileK7,
    GlobalPoolBias,
}

impl PythonBackboneProfileConfig {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conv2dLocal3 => "conv2d_local3",
            Self::TileformerBias => "tileformer_bias",
            Self::ConvnextTileK7 => "convnext_tile_k7",
            Self::GlobalPoolBias => "global_pool_bias",
        }
    }
}

impl PythonModelProfileConfig {
    pub const fn hidden(self) -> usize {
        match self {
            Self::Default | Self::Balanced => 256,
            Self::Large => 384,
        }
    }

    pub const fn blocks(self) -> usize {
        match self {
            Self::Default => 10,
            Self::Balanced => 12,
            Self::Large => 16,
        }
    }

    pub const fn bottleneck(self) -> usize {
        match self {
            Self::Default | Self::Balanced => 64,
            Self::Large => 96,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonResidualProfileConfig {
    #[default]
    MishSe,
    SiluSe,
    ReluSe,
    MishNoSe,
    MishEca,
    ReluNoSe,
    ReluNoNormNoSe,
}

impl PythonResidualProfileConfig {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MishSe => "mish_se",
            Self::SiluSe => "silu_se",
            Self::ReluSe => "relu_se",
            Self::MishNoSe => "mish_no_se",
            Self::MishEca => "mish_eca",
            Self::ReluNoSe => "relu_no_se",
            Self::ReluNoNormNoSe => "relu_no_norm_no_se",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalBackboneProfileConfig {
    #[serde(default)]
    pub activation: BackboneActivationConfig,
    #[serde(default = "default_backbone_se_every_n")]
    pub se_every_n: usize,
    #[serde(default)]
    pub norm: BackboneNormConfig,
    #[serde(default)]
    pub num_blocks: Option<usize>,
    #[serde(default)]
    pub hidden_channels: Option<usize>,
}

impl ExperimentalBackboneProfileConfig {
    pub fn apply_to_model_shape(&self, mut model: ModelShapeConfig) -> ModelShapeConfig {
        model.backbone_activation = self.activation;
        model.backbone_se_every_n = self.se_every_n;
        model.backbone_norm = self.norm;
        if let Some(num_blocks) = self.num_blocks {
            model.num_blocks = num_blocks;
        }
        if let Some(hidden_channels) = self.hidden_channels {
            model.hidden_channels = hidden_channels;
        }
        model
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonLearnerVariant {
    EagerFp32,
    EagerBf16,
    CompileDefault,
    CompileReduceOverhead,
    #[default]
    CompileMaxAutotune,
}

impl PythonLearnerVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EagerFp32 => "eager_fp32",
            Self::EagerBf16 => "eager_bf16",
            Self::CompileDefault => "compile_default",
            Self::CompileReduceOverhead => "compile_reduce_overhead",
            Self::CompileMaxAutotune => "compile_max_autotune",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonRawMjaiTransportConfig {
    #[default]
    PinnedPyo3,
    Stdout,
}

impl PythonRawMjaiTransportConfig {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PinnedPyo3 => "pinned_pyo3",
            Self::Stdout => "stdout",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonConvMemoryFormatConfig {
    #[default]
    Contiguous,
    ChannelsLast,
}

impl PythonConvMemoryFormatConfig {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::ChannelsLast => "channels_last",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonAdamwFlagConfig {
    #[default]
    Auto,
    On,
    Off,
}

impl PythonAdamwFlagConfig {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BcBackendConfig {
    #[default]
    Python,
    RustBurn,
}

impl BcBackendConfig {
    pub const fn as_cli_backend(self) -> BcBackend {
        match self {
            Self::Python => BcBackend::Python,
            Self::RustBurn => BcBackend::RustBurn,
        }
    }
}
