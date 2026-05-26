use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::preflight::{PreflightConfig, ProbeKind};
pub use hydra_data_core::SourceFilterConfig;
use hydra_train_types::config::ModelShapeConfig;
pub use hydra_train_types::config::{BackboneActivationConfig, BackboneNormConfig};
use hydra_train_types::phase::TrainingPhase as PipelineTrainingPhase;

pub use super::config_runtime::{
    default_num_threads_for_system, display_num_threads, loader_runtime_config,
    shard_prefetch_depth, train_microbatch_size, validate_config, validate_preflight_config,
    validation_microbatch_size, validation_sample_limit,
};

mod cli;
pub mod python;
pub use cli::{parse_args, usage, version};
pub use python::{
    PYTHON_TIMING_WARMUP_STEPS, python_options_from_config, python_resume_checkpoint,
    raw_mjai_cursor_resume_supported,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    #[serde(default)]
    pub raw_mjai_data_dirs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub num_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub microbatch_size: Option<usize>,
    #[serde(default)]
    pub validation_microbatch_size: Option<usize>,
    #[serde(default)]
    pub exit_sidecar_path: Option<PathBuf>,
    #[serde(default)]
    pub delta_q_sidecar_path: Option<PathBuf>,
    #[serde(default)]
    pub bc_shards_manifest_path: Option<PathBuf>,
    #[serde(default)]
    pub bc_backend: BcBackendConfig,
    #[serde(default)]
    pub shard_prefetch_depth: Option<usize>,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f32,
    #[serde(default)]
    pub source_filters: SourceFilterConfig,
    #[serde(default = "default_augment")]
    pub augment: bool,
    pub resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_resume_latest")]
    pub resume_latest: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub advanced_loss: Option<AdvancedLossConfig>,
    #[serde(default)]
    pub bc_head_profile: BcHeadProfile,
    #[serde(default)]
    pub python_residual_profile: PythonResidualProfileConfig,
    #[serde(default)]
    pub python_variant: PythonLearnerVariant,
    #[serde(default)]
    pub python_model_profile: PythonModelProfileConfig,
    #[serde(default)]
    pub python_backbone_profile: PythonBackboneProfileConfig,
    #[serde(default)]
    pub python_conv_memory_format: PythonConvMemoryFormatConfig,
    #[serde(default)]
    pub experimental_backbone_profile: Option<ExperimentalBackboneProfileConfig>,
    #[serde(default)]
    pub python_raw_mjai_transport: PythonRawMjaiTransportConfig,
    #[serde(default)]
    pub python_raw_mjai_target_games: Option<usize>,
    #[serde(default)]
    pub python_raw_mjai_estimated_samples_per_game: Option<usize>,
    #[serde(default)]
    pub validation_gates: ValidationGateConfig,
    #[serde(default)]
    pub ema: EmaConfig,
    pub rl: Option<RlTrainConfig>,
    #[serde(default)]
    pub bc: BcHyperparamConfig,
    #[serde(default)]
    pub nsight_trace: Option<NsightTraceConfig>,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub precision_mode: PrecisionMode,
    #[serde(default = "default_buffer_games")]
    pub buffer_games: usize,
    #[serde(default = "default_buffer_samples")]
    pub buffer_samples: usize,
    #[serde(default)]
    pub num_threads: Option<usize>,
    #[serde(default = "default_tensorboard")]
    pub tensorboard: bool,
    #[serde(default = "default_archive_queue_bound")]
    pub archive_queue_bound: usize,
    #[serde(default = "default_validation_every_n_epochs")]
    pub validation_every_n_epochs: usize,
    #[serde(default = "default_max_skip_logs_per_source")]
    pub max_skip_logs_per_source: usize,
    #[serde(default = "default_log_every_n_steps")]
    pub log_every_n_steps: usize,
    #[serde(default = "default_validate_every_n_steps")]
    pub validate_every_n_steps: usize,
    #[serde(default = "default_checkpoint_every_n_steps")]
    pub checkpoint_every_n_steps: usize,
    #[serde(default)]
    pub keep_step_checkpoints: bool,
    #[serde(default)]
    pub launch_tensorboard: bool,
    #[serde(default = "default_tensorboard_host")]
    pub tensorboard_host: String,
    #[serde(default = "default_tensorboard_port")]
    pub tensorboard_port: u16,
    #[serde(default)]
    pub background: bool,
    #[serde(default)]
    pub max_train_steps: Option<usize>,
    #[serde(default)]
    pub full_epoch: bool,
    #[serde(default)]
    pub max_validation_batches: Option<usize>,
    #[serde(default = "default_max_validation_samples")]
    pub max_validation_samples: Option<usize>,
}

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
pub enum PrecisionMode {
    #[default]
    Fp32,
    Bf16Autocast,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectivePrecision {
    #[default]
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "fp32_noop")]
    Fp32NoopForBf16Request,
    #[serde(rename = "bf16_amp")]
    Bf16Amp,
}

impl EffectivePrecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp32NoopForBf16Request => "fp32_noop",
            Self::Bf16Amp => "bf16_amp",
        }
    }
}

impl std::fmt::Display for EffectivePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TrainConfig {
    pub fn use_amp(&self) -> bool {
        matches!(self.precision_mode, PrecisionMode::Bf16Autocast)
    }

    pub fn effective_precision(&self) -> EffectivePrecision {
        match self.precision_mode {
            PrecisionMode::Fp32 => EffectivePrecision::Fp32,
            PrecisionMode::Bf16Autocast => EffectivePrecision::Bf16Amp,
        }
    }

    pub fn default_preflight_bench() -> Self {
        Self {
            data_dir: PathBuf::new(),
            raw_mjai_data_dirs: Vec::new(),
            output_dir: PathBuf::from("preflight_bench"),
            num_epochs: 0,
            batch_size: default_batch_size(),
            microbatch_size: None,
            validation_microbatch_size: None,
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            bc_backend: BcBackendConfig::default(),
            shard_prefetch_depth: Some(default_shard_prefetch_depth()),
            train_fraction: default_train_fraction(),
            source_filters: SourceFilterConfig::default(),
            augment: default_augment(),
            resume_checkpoint: None,
            resume_latest: default_resume_latest(),
            seed: default_seed(),
            advanced_loss: None,
            bc_head_profile: BcHeadProfile::default(),
            python_residual_profile: PythonResidualProfileConfig::default(),
            python_variant: PythonLearnerVariant::default(),
            python_model_profile: PythonModelProfileConfig::default(),
            python_backbone_profile: PythonBackboneProfileConfig::default(),
            python_conv_memory_format: PythonConvMemoryFormatConfig::default(),
            experimental_backbone_profile: None,
            python_raw_mjai_transport: PythonRawMjaiTransportConfig::default(),
            python_raw_mjai_target_games: None,
            python_raw_mjai_estimated_samples_per_game: None,
            validation_gates: ValidationGateConfig::default(),
            ema: EmaConfig::default(),
            rl: None,
            bc: BcHyperparamConfig::default(),
            nsight_trace: None,
            device: default_device(),
            precision_mode: PrecisionMode::default(),
            buffer_games: default_buffer_games(),
            buffer_samples: default_buffer_samples(),
            num_threads: None,
            tensorboard: default_tensorboard(),
            archive_queue_bound: default_archive_queue_bound(),
            validation_every_n_epochs: default_validation_every_n_epochs(),
            max_skip_logs_per_source: default_max_skip_logs_per_source(),
            log_every_n_steps: default_log_every_n_steps(),
            validate_every_n_steps: default_validate_every_n_steps(),
            checkpoint_every_n_steps: default_checkpoint_every_n_steps(),
            keep_step_checkpoints: false,
            launch_tensorboard: false,
            tensorboard_host: default_tensorboard_host(),
            tensorboard_port: default_tensorboard_port(),
            background: false,
            max_train_steps: None,
            full_epoch: false,
            max_validation_batches: None,
            max_validation_samples: default_max_validation_samples(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RlPhaseConfig {
    DrdaAchSelfPlay,
    ExitPondering,
}

impl RlPhaseConfig {
    pub fn to_training_phase(self) -> PipelineTrainingPhase {
        match self {
            Self::DrdaAchSelfPlay => PipelineTrainingPhase::DrdaAchSelfPlay,
            Self::ExitPondering => PipelineTrainingPhase::ExitPondering,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RlTrainConfig {
    #[serde(default = "default_rl_games_per_batch")]
    pub games_per_batch: usize,
    #[serde(default = "default_rl_temperature")]
    pub temperature: f32,
    #[serde(default = "default_rl_phase")]
    pub phase: RlPhaseConfig,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub exit_weight: Option<f32>,
    #[serde(default)]
    pub aux_weight: Option<f32>,
    #[serde(default)]
    pub microbatch_size: Option<usize>,
}

impl Default for RlTrainConfig {
    fn default() -> Self {
        Self {
            games_per_batch: default_rl_games_per_batch(),
            temperature: default_rl_temperature(),
            phase: default_rl_phase(),
            learning_rate: None,
            exit_weight: None,
            aux_weight: None,
            microbatch_size: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeCliRequest {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub warmup_steps: Option<usize>,
    pub measure_steps: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeSingleChildRequest {
    pub request: ProbeCliRequest,
    pub result_path: PathBuf,
    pub manifest_cache_path: Option<PathBuf>,
    pub discovery_summary_path: Option<PathBuf>,
    pub discovery_index_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeBatchChildRequest {
    pub request: ProbeCliRequest,
    pub attempts: usize,
    pub results_path: PathBuf,
    pub manifest_cache_path: Option<PathBuf>,
    pub discovery_summary_path: Option<PathBuf>,
    pub discovery_index_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeChildRequest {
    Single(ProbeSingleChildRequest),
    Batch(ProbeBatchChildRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkBaselineSource {
    Mjai,
    BcShards,
    Both,
}

/// Benchmark backend selector; defaults to LibTorch and Burn-CUDA is parked probe-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentalTrainBackend {
    LibTorch,
    BurnCuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcBackend {
    Python,
    RustBurn,
}

impl BcBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::RustBurn => "rust_burn",
        }
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

#[derive(Debug, Clone, PartialEq)]
pub struct PythonLearnerCliOptions {
    pub bc_shards_manifest: PathBuf,
    pub input: PythonLearnerInput,
    pub output_dir: PathBuf,
    pub device: String,
    pub batch_size: usize,
    pub microbatch_size: usize,
    pub variant: PythonLearnerVariant,
    pub residual_profile: PythonResidualProfileConfig,
    pub conv_memory_format: PythonConvMemoryFormatConfig,
    pub backbone_profile: PythonBackboneProfileConfig,
    pub hidden: usize,
    pub blocks: usize,
    pub bottleneck: usize,
    pub warmup_steps: usize,
    pub steps: Option<usize>,
    pub full_epoch: bool,
    pub validation_steps: usize,
    pub validation_max_samples: Option<usize>,
    pub validation_every: usize,
    pub raw_mjai_validation_augment: bool,
    pub validation_source_mode: String,
    pub checkpoint_out: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    pub checkpoint_every_steps: usize,
    pub log_every_steps: usize,
    pub keep_step_checkpoints: bool,
    pub tensorboard: bool,
    pub launch_tensorboard: bool,
    pub tensorboard_host: String,
    pub tensorboard_port: u16,
    pub background: bool,
    pub learning_rate: f64,
    pub min_learning_rate: f64,
    pub lr_warmup_steps: usize,
    pub lr_schedule: String,
    pub schedule_total_steps: Option<usize>,
    pub schedule_target_games: Option<usize>,
    pub grad_clip_norm: f64,
    pub weight_decay: f64,
    pub ema_enabled: bool,
    pub ema_decay: f64,
    pub ema_start_step: usize,
    pub ema_update_every_steps: usize,
    pub ema_device: EmaDeviceConfig,
    pub adamw_fused: PythonAdamwFlagConfig,
    pub adamw_foreach: PythonAdamwFlagConfig,
    pub compile_fullgraph_check: bool,
    pub oracle_critic_weight: f64,
    pub safety_residual_weight: f64,
    pub exit_weight: f64,
    pub deltaq_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PythonLearnerInput {
    BcShards {
        manifest: PathBuf,
    },
    RawMjai {
        data_dirs: Vec<PathBuf>,
        max_games: Option<usize>,
        max_samples: Option<usize>,
        skip_games: usize,
        train_fraction: f32,
        augment: bool,
        transport: PythonRawMjaiTransportConfig,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkBaselineCliOptions {
    pub data_dir: Option<PathBuf>,
    pub bc_shards_manifest_path: Option<PathBuf>,
    pub source: BenchmarkBaselineSource,
    pub output_dir: PathBuf,
    pub device: String,
    pub max_games: usize,
    pub max_train_steps: usize,
    pub batch_size: usize,
    pub microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub num_threads: usize,
    pub train_threads: usize,
    pub queue_bound: usize,
    pub shard_samples: usize,
    pub train_fraction: f32,
    pub experimental_backend: ExperimentalTrainBackend,
    pub experimental_backbone_profile: Option<ExperimentalBackboneProfileConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainCli {
    pub config_path: Option<PathBuf>,
    pub list_devices: bool,
    pub preflight: Option<PreflightCliOptions>,
    pub benchmark_baseline: Option<BenchmarkBaselineCliOptions>,
    pub delta_q_promotion: bool,
    pub delta_q_baseline_checkpoint: Option<PathBuf>,
    pub probe_only: Option<ProbeCliRequest>,
    pub probe_child: Option<ProbeChildRequest>,
    pub experimental_backend: ExperimentalTrainBackend,
    pub python_learner: Option<PythonLearnerCliOptions>,
    pub bc_backend: BcBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightProfile {
    Default,
    FastRepeatedRun,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreflightCliOptions {
    pub preflight_config: PreflightConfig,
    pub profile: PreflightProfile,
    pub output_dir: PathBuf,
    pub device: String,
    pub bc_shards_manifest_path: Option<PathBuf>,
    pub bc_backend: BcBackend,
    pub python_variant: PythonLearnerVariant,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct BcHyperparamConfig {
    #[serde(default = "default_bc_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_bc_min_learning_rate")]
    pub min_learning_rate: f64,
    #[serde(default = "default_bc_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_bc_grad_clip_norm")]
    pub grad_clip_norm: f32,
    #[serde(default = "default_bc_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default)]
    pub adamw_fused: PythonAdamwFlagConfig,
    #[serde(default)]
    pub adamw_foreach: PythonAdamwFlagConfig,
}

impl Default for BcHyperparamConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_bc_learning_rate(),
            min_learning_rate: default_bc_min_learning_rate(),
            weight_decay: default_bc_weight_decay(),
            grad_clip_norm: default_bc_grad_clip_norm(),
            warmup_steps: default_bc_warmup_steps(),
            adamw_fused: PythonAdamwFlagConfig::default(),
            adamw_foreach: PythonAdamwFlagConfig::default(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct NsightTraceConfig {
    pub kernel_launch_count: Option<u64>,
    pub tiny_kernel_fraction: Option<f64>,
    pub cuda_runtime_launch_seconds: Option<f64>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct AdvancedLossConfig {
    pub exit: Option<f32>,
    pub safety_residual: Option<f32>,
    pub belief_fields: Option<f32>,
    pub mixture_weight: Option<f32>,
    pub opponent_hand_type: Option<f32>,
    pub delta_q: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ValidationGateConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_validation_gate_min_samples")]
    pub min_validation_samples: Option<usize>,
    #[serde(default = "default_validation_gate_max_policy_loss_regression")]
    pub max_policy_loss_regression: Option<f64>,
    #[serde(default)]
    pub min_policy_agreement_delta: Option<f64>,
    #[serde(default)]
    pub fail_training_on_gate_failure: bool,
    #[serde(default = "default_true")]
    pub require_sidecar_coverage_when_weighted: bool,
}

impl Default for ValidationGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_validation_samples: default_validation_gate_min_samples(),
            max_policy_loss_regression: default_validation_gate_max_policy_loss_regression(),
            min_policy_agreement_delta: None,
            fail_training_on_gate_failure: false,
            require_sidecar_coverage_when_weighted: true,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmaDeviceConfig {
    #[default]
    Auto,
    Cuda,
    Cpu,
}

impl EmaDeviceConfig {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmaConfig {
    #[serde(default = "default_ema_enabled")]
    pub enabled: bool,
    #[serde(default = "default_ema_decay")]
    pub decay: f64,
    #[serde(default)]
    pub start_step: usize,
    #[serde(default = "default_ema_update_every_steps")]
    pub update_every_steps: usize,
    #[serde(default)]
    pub device: EmaDeviceConfig,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            enabled: default_ema_enabled(),
            decay: default_ema_decay(),
            start_step: 0,
            update_every_steps: default_ema_update_every_steps(),
            device: EmaDeviceConfig::Auto,
        }
    }
}

pub const fn default_ema_enabled() -> bool {
    true
}

pub fn default_ema_decay() -> f64 {
    0.999
}

pub fn default_ema_update_every_steps() -> usize {
    1
}

fn default_validation_gate_min_samples() -> Option<usize> {
    Some(1024)
}

fn default_validation_gate_max_policy_loss_regression() -> Option<f64> {
    Some(0.0)
}

fn default_true() -> bool {
    true
}

pub fn default_backbone_se_every_n() -> usize {
    1
}
pub fn default_batch_size() -> usize {
    2048
}

pub fn default_rl_games_per_batch() -> usize {
    4
}

pub fn default_rl_temperature() -> f32 {
    1.0
}

pub fn default_rl_phase() -> RlPhaseConfig {
    RlPhaseConfig::DrdaAchSelfPlay
}

pub fn default_bc_learning_rate() -> f64 {
    2.5e-4
}

pub fn default_bc_min_learning_rate() -> f64 {
    1e-6
}

pub fn default_bc_weight_decay() -> f32 {
    1e-5
}

pub fn default_bc_grad_clip_norm() -> f32 {
    1.0
}

pub fn default_bc_warmup_steps() -> usize {
    1000
}

pub fn default_train_fraction() -> f32 {
    0.9
}

pub fn default_augment() -> bool {
    true
}

pub const fn default_resume_latest() -> bool {
    true
}

pub fn default_seed() -> u64 {
    0
}

pub fn default_device() -> String {
    "cpu".to_string()
}

pub fn default_buffer_games() -> usize {
    50_000
}

pub fn default_buffer_samples() -> usize {
    32_768
}

pub fn default_tensorboard() -> bool {
    true
}

pub fn default_tensorboard_host() -> String {
    "127.0.0.1".to_string()
}

pub fn default_tensorboard_port() -> u16 {
    6006
}

pub fn default_archive_queue_bound() -> usize {
    128
}

pub fn default_shard_prefetch_depth() -> usize {
    2
}

pub fn default_validation_every_n_epochs() -> usize {
    1
}

pub fn default_max_skip_logs_per_source() -> usize {
    32
}

pub fn default_log_every_n_steps() -> usize {
    50
}

pub fn default_validate_every_n_steps() -> usize {
    200
}

pub fn default_checkpoint_every_n_steps() -> usize {
    200
}

pub fn default_max_validation_samples() -> Option<usize> {
    Some(8_192)
}

pub fn default_preflight_config_for_profile(profile: PreflightProfile) -> PreflightConfig {
    let mut config = PreflightConfig::default();
    match profile {
        PreflightProfile::Default => {}
        PreflightProfile::FastRepeatedRun => {
            config.fast_repeated_run_profile = true;
            config.fast_repeated_run_candidate_window = 1;
            config.required_successes = 1;
            config.warmup_steps = 1;
            config.measure_steps = 1;
            config.loader_runtime_rounds = 0;
            config.loader_tuple_extra_samples = 0;
            config.real_benchmark_enabled = false;
        }
    }
    config
}

pub fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => {
            let precision_omitted = precision_mode_is_omitted(&raw)
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))?;
            serde_yaml::from_str::<TrainConfig>(&raw)
                .map(|config| resolve_omitted_precision_mode(config, precision_omitted))
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))
        }
        _ => Err(format!(
            "unsupported config extension for {}; use .yaml",
            path.display()
        )),
    }
}

fn resolve_omitted_precision_mode(mut config: TrainConfig, precision_omitted: bool) -> TrainConfig {
    if config.precision_mode == PrecisionMode::Fp32 && precision_omitted {
        let delta_q_active = config
            .advanced_loss
            .as_ref()
            .and_then(|loss| loss.delta_q)
            .is_some_and(|weight| weight > 0.0);
        if config.device.starts_with("cuda") && config.rl.is_none() && !delta_q_active {
            config.precision_mode = PrecisionMode::Bf16Autocast;
        }
    }
    config
}

fn precision_mode_is_omitted(raw: &str) -> Result<bool, serde_yaml::Error> {
    match serde_yaml::from_str::<serde_yaml::Value>(raw)? {
        serde_yaml::Value::Mapping(mapping) => {
            Ok(!mapping.contains_key(serde_yaml::Value::String("precision_mode".to_string())))
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
