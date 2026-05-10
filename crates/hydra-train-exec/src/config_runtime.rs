//! Execution-owned runtime configuration adapters.

use std::env;

use burn::backend::libtorch::LibTorchDevice;
use hydra_model::model::HydraModelConfig;
use hydra_train_runtime::config::{
    RlPhaseConfig, RlTrainConfig, TrainConfig, validate_config as validate_runtime_config,
};
use hydra_train_runtime::config_runtime::resolved_num_threads;
use hydra_train_types::config::{BCTrainerConfig, DEFAULT_RL_MICROBATCH_SIZE, RlConfig};
use rayon::ThreadPoolBuilder;

/// Parses the train device string into a LibTorch device.
pub fn parse_train_device(value: &str) -> Result<LibTorchDevice, String> {
    let value = value.trim().to_ascii_lowercase();
    if value == "cpu" {
        return Ok(LibTorchDevice::Cpu);
    }
    if value == "cuda" {
        return Ok(LibTorchDevice::Cuda(0));
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        let index = index.parse::<usize>().map_err(|_| {
            format!("unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>")
        })?;
        return Ok(LibTorchDevice::Cuda(index));
    }
    Err(format!(
        "unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>"
    ))
}

/// Resolves the effective train device, honoring `HYDRA_TRAIN_DEVICE`.
pub fn train_device(config_device: &str) -> Result<LibTorchDevice, String> {
    let value = match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => value,
        Err(_) => config_device.to_string(),
    };
    parse_train_device(&value)
}

/// Resolves the device label shown in runtime output and fingerprints.
pub fn device_label(config_device: &str) -> String {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => value,
        Err(_) => config_device.to_string(),
    }
}

/// Configures the global Rayon thread pool for train execution.
pub fn configure_threads(num_threads: Option<usize>) -> Result<(), String> {
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

/// Validates train runtime config before heavy execution setup.
pub fn validate_config(config: &TrainConfig) -> Result<(), String> {
    validate_runtime_config(config)
}

/// Builds the Burn BC trainer config from YAML/runtime scalar config.
pub fn trainer_config_from_train_config(config: &TrainConfig) -> BCTrainerConfig {
    BCTrainerConfig::new(HydraModelConfig::learner())
        .with_batch_size(config.batch_size)
        .with_lr(config.bc.learning_rate)
        .with_min_learning_rate(config.bc.min_learning_rate)
        .with_weight_decay(config.bc.weight_decay)
        .with_grad_clip_norm(config.bc.grad_clip_norm)
        .with_warmup_steps(config.bc.warmup_steps)
}

/// Builds the RL train-step config from YAML/runtime scalar config.
pub fn rl_config_from_train_config(rl: &RlTrainConfig) -> RlConfig {
    let mut cfg = match rl.phase {
        RlPhaseConfig::DrdaAchSelfPlay => RlConfig::default_phase2(),
        RlPhaseConfig::ExitPondering => RlConfig::default_phase3(),
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
    cfg.microbatch_size = Some(rl.microbatch_size.unwrap_or(DEFAULT_RL_MICROBATCH_SIZE));
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_train_device_accepts_cpu_cuda_and_indices() {
        assert_eq!(parse_train_device("cpu"), Ok(LibTorchDevice::Cpu));
        assert_eq!(parse_train_device(" CUDA "), Ok(LibTorchDevice::Cuda(0)));
        assert_eq!(parse_train_device("cuda:3"), Ok(LibTorchDevice::Cuda(3)));
    }

    #[test]
    fn parse_train_device_rejects_invalid_values() {
        let err = parse_train_device("cuda:abc").expect_err("invalid cuda index should fail");
        assert!(err.contains("expected cpu, cuda, or cuda:<index>"));

        let err = parse_train_device("metal").expect_err("unsupported backend should fail");
        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=metal"));
    }
}
