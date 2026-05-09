use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::preflight::{PreflightConfig, ProbeKind};
use hydra_train_types::phase::TrainingPhase as PipelineTrainingPhase;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub struct SourceFilterConfig {
    #[serde(default)]
    pub include_source_patterns: Vec<String>,
    #[serde(default)]
    pub exclude_source_patterns: Vec<String>,
}

impl SourceFilterConfig {
    pub fn is_empty(&self) -> bool {
        self.include_source_patterns.is_empty() && self.exclude_source_patterns.is_empty()
    }
}

pub use super::config_runtime::{
    configure_threads, default_num_threads_for_system, device_label, display_num_threads,
    loader_runtime_config, shard_prefetch_depth, train_device, train_microbatch_size,
    validate_config, validation_microbatch_size, validation_sample_limit,
};

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
    #[serde(default)]
    pub preflight: PreflightConfig,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    #[default]
    Fp32,
    Bf16Autocast,
}

impl TrainConfig {
    pub fn use_amp(&self) -> bool {
        matches!(self.precision_mode, PrecisionMode::Bf16Autocast)
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeBatchChildRequest {
    pub request: ProbeCliRequest,
    pub attempts: usize,
    pub results_path: PathBuf,
    pub manifest_cache_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeChildRequest {
    Single(ProbeSingleChildRequest),
    Batch(ProbeBatchChildRequest),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainCli {
    pub config_path: PathBuf,
    pub preflight: bool,
    pub delta_q_promotion: bool,
    pub delta_q_baseline_checkpoint: Option<PathBuf>,
    pub probe_only: Option<ProbeCliRequest>,
    pub probe_child: Option<ProbeChildRequest>,
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

pub fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <config.yaml> [--preflight] [--delta-q-promotion [--delta-q-baseline-checkpoint <path>]] [--probe-kind <train|validation> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]]"
    )
}

pub fn version(program: &str) -> String {
    format!("{program} {}", env!("CARGO_PKG_VERSION"))
}

fn parse_probe_kind(value: &str) -> Result<ProbeKind, String> {
    match value {
        "train" => Ok(ProbeKind::Train),
        "validation" => Ok(ProbeKind::Validation),
        "rl_games" => Ok(ProbeKind::RlGames),
        "rl_microbatch" => Ok(ProbeKind::RlMicrobatch),
        _ => Err(format!(
            "unsupported --probe-kind value '{value}'; expected train, validation, rl_games, or rl_microbatch"
        )),
    }
}

fn parse_usize_flag(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))
}

pub fn parse_args<I>(args: I) -> Result<TrainCli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "train".to_string());
    let first = args.next().ok_or_else(|| usage(&program))?;
    match first.as_str() {
        "--help" | "-h" => return Err(usage(&program)),
        "--version" | "-V" => return Err(version(&program)),
        _ => {}
    }
    let config = first;
    let mut probe_kind = None;
    let mut candidate_microbatch = None;
    let mut warmup_steps = None;
    let mut measure_steps = None;
    let mut probe_attempts = None;
    let mut probe_result_path = None;
    let mut probe_results_path = None;
    let mut probe_manifest_cache_path = None;
    let mut preflight = false;
    let mut delta_q_promotion = false;
    let mut delta_q_baseline_checkpoint = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Err(usage(&program)),
            "--version" | "-V" => return Err(version(&program)),
            "--preflight" => {
                preflight = true;
            }
            "--delta-q-promotion" => {
                delta_q_promotion = true;
            }
            "--delta-q-baseline-checkpoint" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --delta-q-baseline-checkpoint".to_string())?;
                delta_q_baseline_checkpoint = Some(PathBuf::from(value));
            }
            "--probe-kind" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-kind".to_string())?;
                probe_kind = Some(parse_probe_kind(&value)?);
            }
            "--probe-candidate-microbatch" => {
                candidate_microbatch = Some(parse_usize_flag(
                    "--probe-candidate-microbatch",
                    args.next(),
                )?);
            }
            "--probe-warmup-steps" => {
                warmup_steps = Some(parse_usize_flag("--probe-warmup-steps", args.next())?);
            }
            "--probe-measure-steps" => {
                measure_steps = Some(parse_usize_flag("--probe-measure-steps", args.next())?);
            }
            "--probe-attempts" => {
                probe_attempts = Some(parse_usize_flag("--probe-attempts", args.next())?);
            }
            "--probe-result-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-result-path".to_string())?;
                probe_result_path = Some(PathBuf::from(value));
            }
            "--probe-results-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-results-path".to_string())?;
                probe_results_path = Some(PathBuf::from(value));
            }
            "--probe-manifest-cache-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-manifest-cache-path".to_string())?;
                probe_manifest_cache_path = Some(PathBuf::from(value));
            }
            _ => return Err(usage(&program)),
        }
    }

    let config_path = PathBuf::from(config);
    if preflight
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some()
            || delta_q_promotion
            || delta_q_baseline_checkpoint.is_some())
    {
        return Err(format!(
            "{}\n--preflight cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if delta_q_promotion
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some())
    {
        return Err(format!(
            "{}\n--delta-q-promotion cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if delta_q_baseline_checkpoint.is_some() && !delta_q_promotion {
        return Err(format!(
            "{}\n--delta-q-baseline-checkpoint requires --delta-q-promotion",
            usage(&program)
        ));
    }
    if probe_result_path.is_some() && (probe_results_path.is_some() || probe_attempts.is_some()) {
        return Err(format!(
            "{}\ninternal probe child mode cannot combine --probe-result-path with --probe-attempts/--probe-results-path",
            usage(&program)
        ));
    }
    if probe_results_path.is_some() ^ probe_attempts.is_some() {
        return Err(format!(
            "{}\ninternal probe batch child mode requires both --probe-attempts and --probe-results-path",
            usage(&program)
        ));
    }
    match (
        probe_kind,
        candidate_microbatch,
        probe_result_path,
        probe_results_path,
        probe_attempts,
    ) {
        (None, None, None, None, None) => Ok(TrainCli {
            config_path,
            preflight,
            delta_q_promotion,
            delta_q_baseline_checkpoint,
            probe_only: None,
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), None, None, None) => Ok(TrainCli {
            config_path,
            preflight: false,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: Some(ProbeCliRequest {
                kind,
                candidate_microbatch,
                warmup_steps,
                measure_steps,
            }),
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), Some(result_path), None, None) => Ok(TrainCli {
            config_path,
            preflight: false,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
                request: ProbeCliRequest {
                    kind,
                    candidate_microbatch,
                    warmup_steps,
                    measure_steps,
                },
                result_path,
                manifest_cache_path: probe_manifest_cache_path,
            })),
        }),
        (Some(kind), Some(candidate_microbatch), None, Some(results_path), Some(attempts)) => {
            Ok(TrainCli {
                config_path,
                preflight: false,
                delta_q_promotion: false,
                delta_q_baseline_checkpoint: None,
                probe_only: None,
                probe_child: Some(ProbeChildRequest::Batch(ProbeBatchChildRequest {
                    request: ProbeCliRequest {
                        kind,
                        candidate_microbatch,
                        warmup_steps,
                        measure_steps,
                    },
                    attempts,
                    results_path,
                    manifest_cache_path: probe_manifest_cache_path,
                })),
            })
        }
        _ => Err(format!(
            "{}\nprobe mode requires both --probe-kind and --probe-candidate-microbatch",
            usage(&program)
        )),
    }
}

pub fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => serde_yaml::from_str(&raw)
            .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display())),
        _ => Err(format!(
            "unsupported config extension for {}; use .yaml",
            path.display()
        )),
    }
}
