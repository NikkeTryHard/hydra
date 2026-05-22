#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use crate::config::{BcHyperparamConfig, TrainConfig, ValidationGateConfig};

#[cfg(test)]
pub(crate) fn dummy_train_config() -> TrainConfig {
    TrainConfig {
        data_dir: std::env::temp_dir().join("hydra-test-data"),
        output_dir: std::env::temp_dir().join("hydra-test-out"),
        num_epochs: 1,
        batch_size: 256,
        microbatch_size: Some(64),
        validation_microbatch_size: Some(32),
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: crate::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        bc_head_profile: crate::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: ValidationGateConfig::default(),
        rl: None,
        bc: BcHyperparamConfig::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        buffer_games: 16,
        buffer_samples: 128,
        num_threads: None,
        tensorboard: false,
        archive_queue_bound: 8,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 4,
        log_every_n_steps: 10,
        validate_every_n_steps: 10,
        checkpoint_every_n_steps: 10,
        keep_step_checkpoints: false,
        launch_tensorboard: false,
        tensorboard_host: "127.0.0.1".to_string(),
        tensorboard_port: 6006,
        background: false,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: None,
        precision_mode: crate::config::PrecisionMode::Fp32,
    }
}

#[allow(
    dead_code,
    reason = "test helpers are shared across feature-specific test targets"
)]
#[cfg(test)]
pub(crate) fn unique_test_path(prefix: &str, label: &str) -> PathBuf {
    let base = std::env::temp_dir();
    fs::create_dir_all(&base).expect("test temp root should be creatable");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    base.join(format!("{prefix}-{label}-{}-{nanos}", std::process::id()))
}

#[allow(
    dead_code,
    reason = "extension helper is used only by tests that need file suffixes"
)]
#[cfg(test)]
pub(crate) fn unique_test_path_with_extension(
    prefix: &str,
    label: &str,
    extension: &str,
) -> PathBuf {
    unique_test_path(prefix, label).with_extension(extension)
}
