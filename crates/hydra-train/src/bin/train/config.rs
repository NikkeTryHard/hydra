use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::libtorch::LibTorchDevice;
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{LoaderRuntimeConfig, PreflightConfig, ProbeKind};
use hydra_train::training::bc::BCTrainerConfig;
use rayon::ThreadPoolBuilder;
use std::thread::available_parallelism;

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
    pub(crate) bc: BcHyperparamConfig,
    #[serde(default = "default_device")]
    pub(crate) device: String,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeCliRequest {
    pub(crate) kind: ProbeKind,
    pub(crate) candidate_microbatch: usize,
    pub(crate) warmup_steps: Option<usize>,
    pub(crate) measure_steps: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeChildRequest {
    pub(crate) request: ProbeCliRequest,
    pub(crate) result_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrainCli {
    pub(crate) config_path: PathBuf,
    pub(crate) preflight: bool,
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
    pub(crate) safety_residual: Option<f32>,
    pub(crate) belief_fields: Option<f32>,
    pub(crate) mixture_weight: Option<f32>,
    pub(crate) opponent_hand_type: Option<f32>,
    pub(crate) delta_q: Option<f32>,
}

pub(crate) fn default_batch_size() -> usize {
    2048
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
        "Usage: {program} <config.yaml> [--preflight] [--probe-kind <train|validation> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]]"
    )
}

fn parse_probe_kind(value: &str) -> Result<ProbeKind, String> {
    match value {
        "train" => Ok(ProbeKind::Train),
        "validation" => Ok(ProbeKind::Validation),
        _ => Err(format!(
            "unsupported --probe-kind value '{value}'; expected train or validation"
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
    let config = args.next().ok_or_else(|| usage(&program))?;
    let mut probe_kind = None;
    let mut candidate_microbatch = None;
    let mut warmup_steps = None;
    let mut measure_steps = None;
    let mut probe_result_path = None;
    let mut preflight = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--preflight" => {
                preflight = true;
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
            "--probe-result-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-result-path".to_string())?;
                probe_result_path = Some(PathBuf::from(value));
            }
            _ => return Err(usage(&program)),
        }
    }

    let config_path = PathBuf::from(config);
    if preflight && (probe_kind.is_some() || probe_result_path.is_some()) {
        return Err(format!(
            "{}\n--preflight cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    match (probe_kind, candidate_microbatch, probe_result_path) {
        (None, None, None) => Ok(TrainCli {
            config_path,
            preflight,
            probe_only: None,
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), None) => Ok(TrainCli {
            config_path,
            preflight: false,
            probe_only: Some(ProbeCliRequest {
                kind,
                candidate_microbatch,
                warmup_steps,
                measure_steps,
            }),
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), Some(result_path)) => Ok(TrainCli {
            config_path,
            preflight: false,
            probe_only: None,
            probe_child: Some(ProbeChildRequest {
                request: ProbeCliRequest {
                    kind,
                    candidate_microbatch,
                    warmup_steps,
                    measure_steps,
                },
                result_path,
            }),
        }),
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

pub(crate) fn parse_train_device(value: &str) -> LibTorchDevice {
    let value = value.trim().to_ascii_lowercase();
    if value == "cpu" {
        return LibTorchDevice::Cpu;
    }
    if value == "cuda" {
        return LibTorchDevice::Cuda(0);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        let index = index.parse::<usize>().unwrap_or_else(|_| {
            panic!("unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>")
        });
        return LibTorchDevice::Cuda(index);
    }
    panic!("unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>");
}

pub(crate) fn train_device(config_device: &str) -> LibTorchDevice {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => parse_train_device(&value),
        Err(_) => parse_train_device(config_device),
    }
}

pub(crate) fn device_label(config_device: &str) -> String {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => value,
        Err(_) => config_device.to_string(),
    }
}

pub(crate) fn configure_threads(num_threads: Option<usize>) -> Result<(), String> {
    let num_threads = resolved_num_threads(num_threads)?;
    if num_threads <= 1 {
        return Ok(());
    }
    match ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(()) => Ok(()),
        Err(err) if err.to_string().contains("initialized") => Ok(()),
        Err(err) => Err(format!("failed to configure rayon thread pool: {err}")),
    }
}

pub(crate) fn default_num_threads_for_system() -> usize {
    let logical = available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    match logical {
        0 | 1 => 1,
        2..=4 => logical.saturating_sub(1).max(1),
        5..=8 => logical.saturating_sub(2).max(1),
        _ => 8,
    }
}

pub(crate) fn resolved_num_threads(num_threads: Option<usize>) -> Result<usize, String> {
    if let Some(num_threads) = num_threads {
        if num_threads == 0 {
            return Err("num_threads must be greater than 0".to_string());
        }
        return Ok(num_threads);
    }
    Ok(default_num_threads_for_system())
}

pub(crate) fn display_num_threads(num_threads: Option<usize>) -> String {
    match num_threads {
        Some(value) => value.to_string(),
        None => format!("{} (auto)", default_num_threads_for_system()),
    }
}

pub(crate) fn validate_config(config: &TrainConfig) -> Result<(), String> {
    if config.num_epochs == 0 {
        return Err("num_epochs must be greater than 0".to_string());
    }
    if config.batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }
    if config.buffer_games == 0 {
        return Err("buffer_games must be greater than 0".to_string());
    }
    if config.buffer_samples == 0 {
        return Err("buffer_samples must be greater than 0".to_string());
    }
    if config.archive_queue_bound == 0 {
        return Err("archive_queue_bound must be greater than 0".to_string());
    }
    if config.validation_every_n_epochs == 0 {
        return Err("validation_every_n_epochs must be greater than 0".to_string());
    }
    if config.log_every_n_steps == 0 {
        return Err("log_every_n_steps must be greater than 0".to_string());
    }
    if config.validate_every_n_steps == 0 {
        return Err("validate_every_n_steps must be greater than 0".to_string());
    }
    if config.checkpoint_every_n_steps == 0 {
        return Err("checkpoint_every_n_steps must be greater than 0".to_string());
    }
    if let Some(max_train_steps) = config.max_train_steps
        && max_train_steps == 0
    {
        return Err("max_train_steps must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_batches) = config.max_validation_batches
        && max_validation_batches == 0
    {
        return Err("max_validation_batches must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_samples) = config.max_validation_samples
        && max_validation_samples == 0
    {
        return Err("max_validation_samples must be greater than 0 when set".to_string());
    }
    if let Some(microbatch_size) = config.microbatch_size
        && microbatch_size == 0
    {
        return Err("microbatch_size must be greater than 0".to_string());
    }
    if let Some(validation_microbatch_size) = config.validation_microbatch_size
        && validation_microbatch_size == 0
    {
        return Err("validation_microbatch_size must be greater than 0".to_string());
    }
    if config.bc.learning_rate <= 0.0 {
        return Err("bc.learning_rate must be greater than 0".to_string());
    }
    if config.bc.min_learning_rate <= 0.0 {
        return Err("bc.min_learning_rate must be greater than 0".to_string());
    }
    if config.bc.min_learning_rate > config.bc.learning_rate {
        return Err(
            "bc.min_learning_rate must be less than or equal to bc.learning_rate".to_string(),
        );
    }
    if config.bc.weight_decay < 0.0 {
        return Err("bc.weight_decay must be non-negative".to_string());
    }
    if config.bc.grad_clip_norm <= 0.0 {
        return Err("bc.grad_clip_norm must be greater than 0".to_string());
    }
    if config.bc.warmup_steps == 0 {
        return Err("bc.warmup_steps must be greater than 0".to_string());
    }
    Ok(())
}

pub(crate) fn trainer_config_from_train_config(config: &TrainConfig) -> BCTrainerConfig {
    BCTrainerConfig::new(HydraModelConfig::learner())
        .with_batch_size(config.batch_size)
        .with_lr(config.bc.learning_rate)
        .with_min_learning_rate(config.bc.min_learning_rate)
        .with_weight_decay(config.bc.weight_decay)
        .with_grad_clip_norm(config.bc.grad_clip_norm)
        .with_warmup_steps(config.bc.warmup_steps)
}

pub(crate) fn loader_runtime_config(config: &TrainConfig) -> LoaderRuntimeConfig {
    LoaderRuntimeConfig {
        num_threads: Some(default_num_threads_for_system())
            .filter(|_| config.num_threads.is_none())
            .or(config.num_threads),
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        archive_queue_bound: config.archive_queue_bound,
    }
}

pub(crate) fn train_microbatch_size(config: &TrainConfig) -> usize {
    config.microbatch_size.unwrap_or(config.batch_size)
}

pub(crate) fn validation_microbatch_size(config: &TrainConfig) -> usize {
    config
        .validation_microbatch_size
        .unwrap_or_else(|| train_microbatch_size(config))
}

pub(crate) fn validation_sample_limit(config: &TrainConfig) -> Option<usize> {
    config.max_validation_samples.or_else(|| {
        config
            .max_validation_batches
            .map(|limit| limit.saturating_mul(validation_microbatch_size(config)))
    })
}
