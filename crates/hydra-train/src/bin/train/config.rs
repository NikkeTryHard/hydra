use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use hydra_train::config::TrainingPhase as PipelineTrainingPhase;
use hydra_train::preflight::{PreflightConfig, ProbeKind};

pub(crate) use super::config_runtime::{
    configure_threads, default_num_threads_for_system, device_label, display_num_threads,
    loader_runtime_config, train_device, train_microbatch_size, trainer_config_from_train_config,
    validate_config, validation_microbatch_size, validation_sample_limit,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainConfig {
    pub(crate) data_dir: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) num_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub(crate) batch_size: usize,
    #[serde(default)]
    pub(crate) microbatch_size: Option<usize>,
    #[serde(default)]
    pub(crate) validation_microbatch_size: Option<usize>,
    #[serde(default)]
    pub(crate) exit_sidecar_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) delta_q_sidecar_path: Option<PathBuf>,
    #[serde(default = "default_train_fraction")]
    pub(crate) train_fraction: f32,
    #[serde(default = "default_augment")]
    pub(crate) augment: bool,
    pub(crate) resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    pub(crate) seed: u64,
    #[serde(default)]
    pub(crate) advanced_loss: Option<AdvancedLossConfig>,
    #[serde(default)]
    pub(crate) rl: Option<RlTrainConfig>,
    #[serde(default)]
    pub(crate) bc: BcHyperparamConfig,
    #[serde(default = "default_device")]
    pub(crate) device: String,
    #[serde(default)]
    pub(crate) precision_mode: PrecisionMode,
    #[serde(default = "default_buffer_games")]
    pub(crate) buffer_games: usize,
    #[serde(default = "default_buffer_samples")]
    pub(crate) buffer_samples: usize,
    #[serde(default)]
    pub(crate) num_threads: Option<usize>,
    #[serde(default = "default_tensorboard")]
    pub(crate) tensorboard: bool,
    #[serde(default = "default_archive_queue_bound")]
    pub(crate) archive_queue_bound: usize,
    #[serde(default = "default_validation_every_n_epochs")]
    pub(crate) validation_every_n_epochs: usize,
    #[serde(default = "default_max_skip_logs_per_source")]
    pub(crate) max_skip_logs_per_source: usize,
    #[serde(default = "default_log_every_n_steps")]
    pub(crate) log_every_n_steps: usize,
    #[serde(default = "default_validate_every_n_steps")]
    pub(crate) validate_every_n_steps: usize,
    #[serde(default = "default_checkpoint_every_n_steps")]
    pub(crate) checkpoint_every_n_steps: usize,
    #[serde(default)]
    pub(crate) max_train_steps: Option<usize>,
    #[serde(default)]
    pub(crate) max_validation_batches: Option<usize>,
    #[serde(default = "default_max_validation_samples")]
    pub(crate) max_validation_samples: Option<usize>,
    #[serde(default)]
    pub(crate) preflight: PreflightConfig,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PrecisionMode {
    #[default]
    Fp32,
    Bf16Autocast,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RlPhaseConfig {
    DrdaAchSelfPlay,
    ExitPondering,
}

impl RlPhaseConfig {
    pub(crate) fn to_training_phase(self) -> PipelineTrainingPhase {
        match self {
            Self::DrdaAchSelfPlay => PipelineTrainingPhase::DrdaAchSelfPlay,
            Self::ExitPondering => PipelineTrainingPhase::ExitPondering,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct RlTrainConfig {
    #[serde(default = "default_rl_games_per_batch")]
    pub(crate) games_per_batch: usize,
    #[serde(default = "default_rl_temperature")]
    pub(crate) temperature: f32,
    #[serde(default = "default_rl_phase")]
    pub(crate) phase: RlPhaseConfig,
    #[serde(default)]
    pub(crate) learning_rate: Option<f64>,
    #[serde(default)]
    pub(crate) exit_weight: Option<f32>,
    #[serde(default)]
    pub(crate) aux_weight: Option<f32>,
    #[serde(default)]
    pub(crate) microbatch_size: Option<usize>,
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
pub(crate) struct ProbeCliRequest {
    pub(crate) kind: ProbeKind,
    pub(crate) candidate_microbatch: usize,
    pub(crate) warmup_steps: Option<usize>,
    pub(crate) measure_steps: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeSingleChildRequest {
    pub(crate) request: ProbeCliRequest,
    pub(crate) result_path: PathBuf,
    pub(crate) manifest_cache_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeBatchChildRequest {
    pub(crate) request: ProbeCliRequest,
    pub(crate) attempts: usize,
    pub(crate) results_path: PathBuf,
    pub(crate) manifest_cache_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProbeChildRequest {
    Single(ProbeSingleChildRequest),
    Batch(ProbeBatchChildRequest),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrainCli {
    pub(crate) config_path: PathBuf,
    pub(crate) preflight: bool,
    pub(crate) delta_q_promotion: bool,
    pub(crate) delta_q_baseline_checkpoint: Option<PathBuf>,
    pub(crate) probe_only: Option<ProbeCliRequest>,
    pub(crate) probe_child: Option<ProbeChildRequest>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct BcHyperparamConfig {
    #[serde(default = "default_bc_learning_rate")]
    pub(crate) learning_rate: f64,
    #[serde(default = "default_bc_min_learning_rate")]
    pub(crate) min_learning_rate: f64,
    #[serde(default = "default_bc_weight_decay")]
    pub(crate) weight_decay: f32,
    #[serde(default = "default_bc_grad_clip_norm")]
    pub(crate) grad_clip_norm: f32,
    #[serde(default = "default_bc_warmup_steps")]
    pub(crate) warmup_steps: usize,
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
pub(crate) struct AdvancedLossConfig {
    pub(crate) exit: Option<f32>,
    pub(crate) safety_residual: Option<f32>,
    pub(crate) belief_fields: Option<f32>,
    pub(crate) mixture_weight: Option<f32>,
    pub(crate) opponent_hand_type: Option<f32>,
    pub(crate) delta_q: Option<f32>,
}

pub(crate) fn default_batch_size() -> usize {
    2048
}

pub(crate) fn default_rl_games_per_batch() -> usize {
    4
}

pub(crate) fn default_rl_temperature() -> f32 {
    1.0
}

pub(crate) fn default_rl_phase() -> RlPhaseConfig {
    RlPhaseConfig::DrdaAchSelfPlay
}

pub(crate) fn default_bc_learning_rate() -> f64 {
    2.5e-4
}

pub(crate) fn default_bc_min_learning_rate() -> f64 {
    1e-6
}

pub(crate) fn default_bc_weight_decay() -> f32 {
    1e-5
}

pub(crate) fn default_bc_grad_clip_norm() -> f32 {
    1.0
}

pub(crate) fn default_bc_warmup_steps() -> usize {
    1000
}

pub(crate) fn default_train_fraction() -> f32 {
    0.9
}

pub(crate) fn default_augment() -> bool {
    true
}

pub(crate) fn default_seed() -> u64 {
    0
}

pub(crate) fn default_device() -> String {
    "cpu".to_string()
}

pub(crate) fn default_buffer_games() -> usize {
    50_000
}

pub(crate) fn default_buffer_samples() -> usize {
    32_768
}

pub(crate) fn default_tensorboard() -> bool {
    true
}

pub(crate) fn default_archive_queue_bound() -> usize {
    128
}

pub(crate) fn default_validation_every_n_epochs() -> usize {
    1
}

pub(crate) fn default_max_skip_logs_per_source() -> usize {
    32
}

pub(crate) fn default_log_every_n_steps() -> usize {
    50
}

pub(crate) fn default_validate_every_n_steps() -> usize {
    200
}

pub(crate) fn default_checkpoint_every_n_steps() -> usize {
    200
}

pub(crate) fn default_max_validation_samples() -> Option<usize> {
    Some(8_192)
}

pub(crate) fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <config.yaml> [--preflight] [--delta-q-promotion [--delta-q-baseline-checkpoint <path>]] [--probe-kind <train|validation> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]]"
    )
}

pub(crate) fn version(program: &str) -> String {
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

pub(crate) fn parse_args<I>(args: I) -> Result<TrainCli, String>
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

pub(crate) fn read_config(path: &Path) -> Result<TrainConfig, String> {
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
