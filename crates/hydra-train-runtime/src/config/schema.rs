use std::path::PathBuf;

use hydra_data_core::SourceFilterConfig;

use super::cli_types::PreflightProfile;
use super::profiles::{
    BcBackendConfig, BcHeadProfile, ExperimentalBackboneProfileConfig, PythonAdamwFlagConfig,
    PythonBackboneProfileConfig, PythonConvMemoryFormatConfig, PythonLearnerVariant,
    PythonModelProfileConfig, PythonRawMjaiTransportConfig, PythonResidualProfileConfig,
};
use super::stages::rl_stage_for_phase;
use crate::preflight::PreflightConfig;
use hydra_train_types::phase::TrainingPhase as PipelineTrainingPhase;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    #[serde(default)]
    pub raw_mjai_data_dirs: Vec<PathBuf>,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub run_name: Option<String>,
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
            stage: None,
            run_name: None,
        }
    }
}

/// YAML selector for RL training lane.
///
/// `ppo_control` dispatches the Python T1 masked PPO-GAE operator path.
/// `drda_ach_self_play` remains the legacy Rust/Burn RL lane when that
/// compatibility feature is available.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RlPhaseConfig {
    PpoControl,
    DrdaAchSelfPlay,
    ExitPondering,
}

impl RlPhaseConfig {
    pub const fn default_stage(self) -> &'static str {
        rl_stage_for_phase(self)
    }

    pub fn to_training_phase(self) -> PipelineTrainingPhase {
        match self {
            Self::DrdaAchSelfPlay => PipelineTrainingPhase::DrdaAchSelfPlay,
            Self::PpoControl => PipelineTrainingPhase::DrdaAchSelfPlay,
            Self::ExitPondering => PipelineTrainingPhase::ExitPondering,
        }
    }
}

/// YAML-owned RL training configuration.
///
/// Accepted fields are exactly `games_per_batch`, `temperature`, `phase`,
/// `learning_rate`, `exit_weight`, `aux_weight`, and `microbatch_size`.
/// Unknown fields fail deserialization.
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
    #[serde(default)]
    pub epochs: Option<usize>,
    #[serde(default)]
    pub target_kl: Option<f64>,
    #[serde(default)]
    pub run_forever: bool,
    #[serde(default)]
    pub rollout_inference: Option<String>,
    #[serde(default)]
    pub ppo_rollout_device: Option<String>,
    #[serde(default)]
    pub bc_kl_reverse_coef: Option<f64>,
    #[serde(default)]
    pub lr_warmup_samples: Option<usize>,
    #[serde(default)]
    pub lr_decay_samples: Option<usize>,
    #[serde(default)]
    pub arena_batch_decisions: Option<usize>,
    #[serde(default)]
    pub ppo_pipeline_depth: Option<usize>,
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
            epochs: None,
            target_kl: None,
            run_forever: false,
            lr_warmup_samples: None,
            lr_decay_samples: None,
            rollout_inference: None,
            ppo_rollout_device: None,
            bc_kl_reverse_coef: None,
            arena_batch_decisions: None,
            ppo_pipeline_depth: None,
        }
    }
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
    1024
}

pub fn default_rl_temperature() -> f32 {
    1.0
}

pub fn default_rl_phase() -> RlPhaseConfig {
    RlPhaseConfig::PpoControl
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
