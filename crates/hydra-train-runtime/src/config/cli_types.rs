use std::path::PathBuf;

use crate::preflight::{PreflightConfig, ProbeKind};

use super::profiles::{
    ExperimentalBackboneProfileConfig, PythonAdamwFlagConfig, PythonBackboneProfileConfig,
    PythonConvMemoryFormatConfig, PythonLearnerVariant, PythonRawMjaiTransportConfig,
    PythonResidualProfileConfig,
};
use super::schema::EmaDeviceConfig;

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

#[derive(Debug, Clone, PartialEq)]
pub struct PythonPpoControlCliOptions {
    pub init_checkpoint: PathBuf,
    pub output_dir: PathBuf,
    pub stage: Option<String>,
    pub run_name: Option<String>,
    pub device: String,
    pub rollout_device: Option<String>,
    pub steps: Option<usize>,
    pub games_per_update: usize,
    pub seed: u64,
    pub temperature: f32,
    pub arena_batch_decisions: usize,
    pub microbatch_size: usize,
    pub epochs: usize,
    pub target_kl: Option<f64>,
    pub arena_threads: usize,
    pub hidden: usize,
    pub blocks: usize,
    pub bottleneck: usize,
    pub residual_profile: PythonResidualProfileConfig,
    pub conv_memory_format: PythonConvMemoryFormatConfig,
    pub backbone_profile: PythonBackboneProfileConfig,
    pub learning_rate: f64,
    pub min_learning_rate: f64,
    pub lr_warmup_samples: usize,
    pub lr_decay_samples: Option<usize>,
    pub grad_clip_norm: f64,
    pub weight_decay: f64,
    pub adamw_fused: PythonAdamwFlagConfig,
    pub adamw_foreach: PythonAdamwFlagConfig,
    pub bc_kl_reverse_coef: f64,
    pub resume: Option<PathBuf>,
    pub checkpoint_every_steps: usize,
    pub log_every_steps: usize,
    pub keep_step_checkpoints: bool,
    pub tensorboard: bool,
    pub launch_tensorboard: bool,
    pub tensorboard_host: String,
    pub tensorboard_port: u16,
    pub rollout_inference: String,
    pub ppo_pipeline_depth: usize,
    pub background: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PythonLearnerCliOptions {
    pub bc_shards_manifest: PathBuf,
    pub input: PythonLearnerInput,
    pub output_dir: PathBuf,
    pub stage: Option<String>,
    pub run_name: Option<String>,
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
