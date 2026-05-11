#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use crate::config::{BcHyperparamConfig, TrainConfig, ValidationGateConfig};
#[cfg(test)]
use hydra_train_runtime::preflight::PreflightConfig;

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
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
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
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: None,
        preflight: PreflightConfig::default(),
        precision_mode: crate::config::PrecisionMode::Fp32,
    }
}

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
