use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::preflight::{PreflightConfig, ProbeKind};
pub use hydra_data_core::SourceFilterConfig;
use hydra_train_types::phase::TrainingPhase as PipelineTrainingPhase;

pub use super::config_runtime::{
    default_num_threads_for_system, display_num_threads, loader_runtime_config,
    shard_prefetch_depth, train_microbatch_size, validate_config, validate_preflight_config,
    validation_microbatch_size, validation_sample_limit,
};

mod cli;
pub use cli::{parse_args, usage, version};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
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
    pub shard_prefetch_depth: Option<usize>,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f32,
    #[serde(default)]
    pub source_filters: SourceFilterConfig,
    #[serde(default = "default_augment")]
    pub augment: bool,
    pub resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub advanced_loss: Option<AdvancedLossConfig>,
    #[serde(default)]
    pub validation_gates: ValidationGateConfig,
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
    pub max_train_steps: Option<usize>,
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
            output_dir: PathBuf::from("preflight_bench"),
            num_epochs: 0,
            batch_size: default_batch_size(),
            microbatch_size: None,
            validation_microbatch_size: None,
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: Some(default_shard_prefetch_depth()),
            train_fraction: default_train_fraction(),
            source_filters: SourceFilterConfig::default(),
            augment: default_augment(),
            resume_checkpoint: None,
            seed: default_seed(),
            advanced_loss: None,
            validation_gates: ValidationGateConfig::default(),
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
            max_train_steps: None,
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
}

impl Default for BcHyperparamConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_bc_learning_rate(),
            min_learning_rate: default_bc_min_learning_rate(),
            weight_decay: default_bc_weight_decay(),
            grad_clip_norm: default_bc_grad_clip_norm(),
            warmup_steps: default_bc_warmup_steps(),
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

fn default_validation_gate_min_samples() -> Option<usize> {
    Some(1024)
}

fn default_validation_gate_max_policy_loss_regression() -> Option<f64> {
    Some(0.0)
}

fn default_true() -> bool {
    true
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
