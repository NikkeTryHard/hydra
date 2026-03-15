use std::env;
use std::thread::available_parallelism;

use burn::backend::libtorch::LibTorchDevice;
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::LoaderRuntimeConfig;
use hydra_train::training::bc::BCTrainerConfig;
use hydra_train::training::rl::RlConfig;
use rayon::ThreadPoolBuilder;

use super::config::{RlTrainConfig, TrainConfig};

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
    available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .max(1)
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
    if let Some(rl) = config.rl.as_ref() {
        validate_rl_config(rl)?;
    }
    if config
        .advanced_loss
        .as_ref()
        .and_then(|loss| loss.exit)
        .is_some_and(|weight| weight > 0.0)
        && config.exit_sidecar_path.is_none()
    {
        return Err(
            "advanced_loss.exit requires exit_sidecar_path so replay ExIt labels are present"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_rl_config(rl: &RlTrainConfig) -> Result<(), String> {
    if rl.games_per_batch == 0 {
        return Err("rl.games_per_batch must be greater than 0".to_string());
    }
    if rl.temperature <= 0.0 {
        return Err("rl.temperature must be greater than 0".to_string());
    }
    if let Some(lr) = rl.learning_rate
        && lr <= 0.0
    {
        return Err("rl.learning_rate must be greater than 0 when set".to_string());
    }
    if let Some(exit_weight) = rl.exit_weight
        && exit_weight < 0.0
    {
        return Err("rl.exit_weight must be non-negative".to_string());
    }
    if let Some(aux_weight) = rl.aux_weight
        && aux_weight < 0.0
    {
        return Err("rl.aux_weight must be non-negative".to_string());
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

pub(crate) fn rl_config_from_train_config(rl: &RlTrainConfig) -> RlConfig {
    let mut cfg = match rl.phase {
        super::config::RlPhaseConfig::DrdaAchSelfPlay => RlConfig::default_phase2(),
        super::config::RlPhaseConfig::ExitPondering => RlConfig::default_phase3(),
    };
    if let Some(lr) = rl.learning_rate {
        cfg = cfg.with_lr(lr);
    }
    if let Some(exit_weight) = rl.exit_weight {
        cfg = cfg.with_exit_weight(exit_weight);
    }
    if let Some(aux_weight) = rl.aux_weight {
        cfg = cfg.with_aux_weight(aux_weight);
    }
    cfg.microbatch_size = Some(
        rl.microbatch_size
            .unwrap_or(hydra_train::training::rl::DEFAULT_RL_MICROBATCH_SIZE),
    );
    cfg
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
