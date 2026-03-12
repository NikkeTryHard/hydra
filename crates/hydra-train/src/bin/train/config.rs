use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::libtorch::LibTorchDevice;
use hydra_train::preflight::PreflightConfig;
use rayon::ThreadPoolBuilder;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    format!("Usage: {program} <config.json>")
}

pub(crate) fn parse_args<I>(args: I) -> Result<PathBuf, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "train".to_string());
    match (args.next(), args.next()) {
        (Some(config), None) => Ok(PathBuf::from(config)),
        _ => Err(usage(&program)),
    }
}

pub(crate) fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => serde_yaml::from_str(&raw)
            .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display())),
        Some("json") => serde_json::from_str(&raw)
            .map_err(|err| format!("failed to parse json config {}: {err}", path.display())),
        _ => Err(format!(
            "unsupported config extension for {}; use .yaml or .yml",
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
    let Some(num_threads) = num_threads else {
        return Ok(());
    };
    if num_threads == 0 {
        return Err("num_threads must be greater than 0".to_string());
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
    Ok(())
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
