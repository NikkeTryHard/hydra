use std::fs;

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{HardwareFingerprint, PreflightCacheKey, WorkloadFingerprint};

use super::config::{AdvancedLossConfig, TrainConfig};

fn total_memory_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub(super) fn advanced_loss_signature(config: Option<&AdvancedLossConfig>) -> String {
    match config {
        Some(config) => serde_json::to_string(config)
            .unwrap_or_else(|_| "advanced_loss:unserializable".to_string()),
        None => "advanced_loss:none".to_string(),
    }
}

pub(super) fn workload_fingerprint(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
) -> WorkloadFingerprint {
    WorkloadFingerprint {
        batch_size: config.batch_size,
        augment: config.augment,
        train_fraction_bits: config.train_fraction.to_bits(),
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        max_validation_batches: config.max_validation_batches,
        max_validation_samples: config.max_validation_samples,
        model_signature: format!(
            "blocks:{} input:{} hidden:{} groups:{} action:{} score_bins:{}",
            model_config.num_blocks,
            model_config.input_channels,
            model_config.hidden_channels,
            model_config.num_groups,
            model_config.action_space,
            model_config.score_bins,
        ),
        code_signature: format!(
            "hydra-train:{}:{}:preflight-v3",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_NAME")
        ),
        advanced_loss_signature: advanced_loss_signature(config.advanced_loss.as_ref()),
    }
}

pub(super) fn hardware_fingerprint(
    device_label: &str,
    cpu_logical_cores: usize,
) -> HardwareFingerprint {
    HardwareFingerprint {
        device_label: device_label.to_string(),
        backend: "burn-libtorch".to_string(),
        cpu_logical_cores,
        total_memory_bytes: total_memory_bytes(),
    }
}

pub(super) fn preflight_cache_key(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
    cpu_logical_cores: usize,
) -> PreflightCacheKey {
    PreflightCacheKey {
        hardware: hardware_fingerprint(device_label, cpu_logical_cores),
        workload: workload_fingerprint(config, model_config),
    }
}
