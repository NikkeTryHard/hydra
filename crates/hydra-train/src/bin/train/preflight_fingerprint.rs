use std::fs;

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{HardwareFingerprint, PreflightCacheKey, WorkloadFingerprint};

use super::config::{AdvancedLossConfig, PrecisionMode, TrainConfig};

fn total_memory_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub(super) fn precision_mode_signature(mode: PrecisionMode) -> String {
    match mode {
        PrecisionMode::Fp32 => "fp32".to_string(),
        PrecisionMode::Bf16Autocast => "bf16_autocast".to_string(),
    }
}

pub(super) fn advanced_loss_signature(config: Option<&AdvancedLossConfig>) -> String {
    match config {
        Some(config) => serde_json::to_string(config)
            .unwrap_or_else(|_| "advanced_loss:unserializable".to_string()),
        None => "advanced_loss:none".to_string(),
    }
}

pub(super) fn preflight_config_signature(config: &TrainConfig) -> String {
    serde_json::to_string(&config.preflight)
        .unwrap_or_else(|_| "preflight_config:unserializable".to_string())
}

pub(super) fn workload_fingerprint(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
) -> WorkloadFingerprint {
    WorkloadFingerprint {
        batch_size: config.batch_size,
        augment: config.augment,
        precision_mode: precision_mode_signature(config.precision_mode),
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
            "hydra-train:{}:{}:preflight-v4",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_NAME")
        ),
        advanced_loss_signature: advanced_loss_signature(config.advanced_loss.as_ref()),
        preflight_config_signature: preflight_config_signature(config),
        explicit_train_microbatch: config.microbatch_size,
        explicit_validation_microbatch: config.validation_microbatch_size,
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use hydra_train::preflight::PreflightConfig;

    use crate::config::{AdvancedLossConfig, BcHyperparamConfig, PrecisionMode};

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/data"),
            output_dir: PathBuf::from("/output"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
            train_fraction: 0.875,
            augment: true,
            resume_checkpoint: None,
            seed: 7,
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
            device: "cpu".to_string(),
            precision_mode: PrecisionMode::Fp32,
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(2),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 5,
            validate_every_n_steps: 6,
            checkpoint_every_n_steps: 7,
            max_train_steps: Some(9),
            max_validation_batches: Some(3),
            max_validation_samples: Some(99),
            preflight: PreflightConfig::default(),
        }
    }

    #[test]
    fn advanced_loss_signature_distinguishes_none_and_some() {
        assert_eq!(advanced_loss_signature(None), "advanced_loss:none");

        let signature = advanced_loss_signature(Some(&AdvancedLossConfig {
            exit: Some(0.5),
            safety_residual: None,
            belief_fields: Some(0.25),
            mixture_weight: None,
            opponent_hand_type: None,
            delta_q: Some(1.0),
        }));
        assert!(signature.contains("\"exit\":0.5"));
        assert!(signature.contains("\"belief_fields\":0.25"));
        assert!(signature.contains("\"delta_q\":1.0"));
    }

    #[test]
    fn workload_fingerprint_captures_model_and_config_shape() {
        let mut config = dummy_config();
        config.precision_mode = PrecisionMode::Bf16Autocast;
        config.advanced_loss = Some(AdvancedLossConfig {
            exit: Some(0.25),
            safety_residual: Some(0.75),
            belief_fields: None,
            mixture_weight: None,
            opponent_hand_type: None,
            delta_q: None,
        });
        let model_config = HydraModelConfig::learner();

        let fingerprint = workload_fingerprint(&config, &model_config);
        assert_eq!(fingerprint.batch_size, 256);
        assert!(fingerprint.augment);
        assert_eq!(fingerprint.precision_mode, "bf16_autocast");
        assert_eq!(fingerprint.train_fraction_bits, 0.875f32.to_bits());
        assert_eq!(fingerprint.max_skip_logs_per_source, 4);
        assert_eq!(fingerprint.max_validation_batches, Some(3));
        assert_eq!(fingerprint.max_validation_samples, Some(99));
        assert!(fingerprint.model_signature.contains("blocks:24"));
        assert!(fingerprint.model_signature.contains("action:46"));
        assert!(fingerprint.code_signature.contains("hydra-train:"));
        assert!(fingerprint
            .advanced_loss_signature
            .contains("\"exit\":0.25"));
        assert!(fingerprint
            .advanced_loss_signature
            .contains("\"safety_residual\":0.75"));
        assert!(
            !fingerprint.preflight_config_signature.is_empty(),
            "preflight config signature should be populated"
        );
        assert_eq!(fingerprint.explicit_train_microbatch, Some(64));
        assert_eq!(fingerprint.explicit_validation_microbatch, Some(32));
    }

    #[test]
    fn hardware_and_cache_key_include_runtime_identity() {
        let config = dummy_config();
        let model_config = HydraModelConfig::actor();

        let hardware = hardware_fingerprint("cuda:0", 32);
        assert_eq!(hardware.device_label, "cuda:0");
        assert_eq!(hardware.backend, "burn-libtorch");
        assert_eq!(hardware.cpu_logical_cores, 32);

        let cache_key = preflight_cache_key(&config, &model_config, "cuda:0", 32);
        assert_eq!(cache_key.hardware.device_label, "cuda:0");
        assert_eq!(cache_key.hardware.cpu_logical_cores, 32);
        assert_eq!(cache_key.workload.batch_size, config.batch_size);
        assert_eq!(cache_key.workload.precision_mode, "fp32");
        assert!(cache_key.workload.model_signature.contains("blocks:12"));
    }

    #[test]
    fn precision_mode_changes_preflight_cache_key() {
        let fp32 = dummy_config();
        let mut bf16 = dummy_config();
        bf16.precision_mode = PrecisionMode::Bf16Autocast;
        let model_config = HydraModelConfig::learner();

        let fp32_key = preflight_cache_key(&fp32, &model_config, "cuda:0", 32);
        let bf16_key = preflight_cache_key(&bf16, &model_config, "cuda:0", 32);

        assert_eq!(fp32_key.workload.precision_mode, "fp32");
        assert_eq!(bf16_key.workload.precision_mode, "bf16_autocast");
        assert_ne!(fp32_key, bf16_key);
    }

    #[test]
    fn preflight_config_knob_change_invalidates_cache_key() {
        let baseline = dummy_config();
        let mut changed = dummy_config();
        changed.preflight.warmup_steps = baseline.preflight.warmup_steps + 1;
        let model_config = HydraModelConfig::learner();

        let baseline_key = preflight_cache_key(&baseline, &model_config, "cuda:0", 32);
        let changed_key = preflight_cache_key(&changed, &model_config, "cuda:0", 32);

        assert_ne!(
            baseline_key.workload.preflight_config_signature,
            changed_key.workload.preflight_config_signature,
        );
        assert_ne!(baseline_key, changed_key);
    }

    #[test]
    fn preflight_config_candidate_microbatches_change_invalidates_cache_key() {
        let baseline = dummy_config();
        let mut changed = dummy_config();
        changed.preflight.candidate_microbatches = vec![128, 64, 32];
        let model_config = HydraModelConfig::learner();

        let baseline_key = preflight_cache_key(&baseline, &model_config, "cuda:0", 32);
        let changed_key = preflight_cache_key(&changed, &model_config, "cuda:0", 32);

        assert_ne!(baseline_key, changed_key);
    }

    #[test]
    fn explicit_train_microbatch_change_invalidates_cache_key() {
        let mut no_override = dummy_config();
        no_override.microbatch_size = None;
        let mut with_override = dummy_config();
        with_override.microbatch_size = Some(128);
        let model_config = HydraModelConfig::learner();

        let no_key = preflight_cache_key(&no_override, &model_config, "cuda:0", 32);
        let with_key = preflight_cache_key(&with_override, &model_config, "cuda:0", 32);

        assert_eq!(no_key.workload.explicit_train_microbatch, None);
        assert_eq!(with_key.workload.explicit_train_microbatch, Some(128));
        assert_ne!(no_key, with_key);
    }

    #[test]
    fn explicit_validation_microbatch_change_invalidates_cache_key() {
        let mut no_override = dummy_config();
        no_override.validation_microbatch_size = None;
        let mut with_override = dummy_config();
        with_override.validation_microbatch_size = Some(256);
        let model_config = HydraModelConfig::learner();

        let no_key = preflight_cache_key(&no_override, &model_config, "cuda:0", 32);
        let with_key = preflight_cache_key(&with_override, &model_config, "cuda:0", 32);

        assert_eq!(no_key.workload.explicit_validation_microbatch, None);
        assert_eq!(with_key.workload.explicit_validation_microbatch, Some(256));
        assert_ne!(no_key, with_key);
    }

    #[test]
    fn non_selection_fields_do_not_change_cache_key() {
        let baseline = dummy_config();
        let mut changed = dummy_config();
        changed.num_threads = Some(99);
        changed.buffer_samples = 9999;
        changed.buffer_games = 9999;
        changed.seed = 42;
        changed.data_dir = PathBuf::from("/different/data");
        changed.output_dir = PathBuf::from("/different/output");
        changed.tensorboard = !baseline.tensorboard;
        changed.log_every_n_steps = baseline.log_every_n_steps + 100;
        changed.checkpoint_every_n_steps = baseline.checkpoint_every_n_steps + 100;
        let model_config = HydraModelConfig::learner();

        let baseline_key = preflight_cache_key(&baseline, &model_config, "cuda:0", 32);
        let changed_key = preflight_cache_key(&changed, &model_config, "cuda:0", 32);

        assert_eq!(baseline_key, changed_key);
    }

    #[test]
    fn code_signature_uses_v4_version() {
        let config = dummy_config();
        let model_config = HydraModelConfig::learner();
        let fingerprint = workload_fingerprint(&config, &model_config);
        assert!(
            fingerprint.code_signature.contains("preflight-v4"),
            "code_signature should contain preflight-v4, got: {}",
            fingerprint.code_signature
        );
    }

    #[test]
    fn total_memory_bytes_is_optional_but_never_negative() {
        assert!(total_memory_bytes().map(|bytes| bytes > 0).unwrap_or(true));
    }
}
