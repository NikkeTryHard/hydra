use super::super::config::{
    AdvancedLossConfig, EffectivePrecision, PrecisionMode, RlPhaseConfig, RlTrainConfig,
    read_config,
};
use super::*;
use crate::preflight::{PreflightConfig, PreflightTuningMode};
use crate::test_support::dummy_train_config;
use std::path::PathBuf;

fn dummy_config() -> TrainConfig {
    let mut config = dummy_train_config();
    config.data_dir = PathBuf::from("/data");
    config.output_dir = PathBuf::from("/output");
    config.precision_mode = PrecisionMode::Fp32;
    config.num_threads = Some(6);
    config
}

fn write_config_yaml(label: &str, yaml: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "hydra-config-runtime-{label}-{}.yaml",
        std::process::id()
    ));
    std::fs::write(&path, yaml).expect("config fixture should write");
    path
}

#[test]
fn read_config_defaults_cuda_bc_to_bf16_amp_when_precision_omitted() {
    let path = write_config_yaml(
        "cuda-bf16-default",
        r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
device: cuda:0
"#,
    );
    let config = read_config(&path).expect("config should parse");
    assert_eq!(config.precision_mode, PrecisionMode::Bf16Autocast);
    assert_eq!(config.effective_precision(), EffectivePrecision::Bf16Amp);
    std::fs::remove_file(path).ok();
}

#[test]
fn read_config_keeps_explicit_cuda_fp32_override() {
    let path = write_config_yaml(
        "cuda-fp32-explicit",
        r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
device: cuda:0
precision_mode: fp32
"#,
    );
    let config = read_config(&path).expect("config should parse");
    assert_eq!(config.precision_mode, PrecisionMode::Fp32);
    assert_eq!(config.effective_precision(), EffectivePrecision::Fp32);
    std::fs::remove_file(path).ok();
}

#[test]
fn read_config_keeps_cpu_omitted_precision_fp32() {
    let path = write_config_yaml(
        "cpu-fp32-default",
        r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
device: cpu
"#,
    );
    let config = read_config(&path).expect("config should parse");
    assert_eq!(config.precision_mode, PrecisionMode::Fp32);
    assert_eq!(config.effective_precision(), EffectivePrecision::Fp32);
    std::fs::remove_file(path).ok();
}

#[test]
fn resolved_num_threads_rejects_zero_and_uses_explicit_value() {
    assert_eq!(resolved_num_threads(Some(4)), Ok(4));
    assert_eq!(
        resolved_num_threads(Some(0)),
        Err("num_threads must be greater than 0".to_string())
    );
}

#[test]
fn validate_config_requires_positive_sidecar_backed_advanced_losses() {
    let mut config = dummy_config();
    config.advanced_loss = Some(AdvancedLossConfig {
        exit: Some(0.25),
        ..AdvancedLossConfig::default()
    });
    assert_eq!(
        validate_config(&config),
        Err(
            "advanced_loss.exit requires exit_sidecar_path so replay ExIt labels are present"
                .to_string()
        )
    );

    config.exit_sidecar_path = Some(PathBuf::from("/tmp/exit.sidecar"));
    assert!(validate_config(&config).is_ok());

    config.advanced_loss = Some(AdvancedLossConfig {
        delta_q: Some(0.1),
        ..AdvancedLossConfig::default()
    });
    assert_eq!(
        validate_config(&config),
        Err(
            "advanced_loss.delta_q requires delta_q_sidecar_path so replay delta_q labels are present"
                .to_string()
        )
    );
}

#[test]
fn validate_config_rejects_bf16_for_rl_and_delta_q() {
    let mut rl_config = dummy_config();
    rl_config.precision_mode = PrecisionMode::Bf16Autocast;
    rl_config.rl = Some(RlTrainConfig::default());
    assert_eq!(
        validate_config(&rl_config),
        Err("precision_mode=bf16_autocast is not supported for RL training yet".to_string())
    );

    let mut delta_q_config = dummy_config();
    delta_q_config.precision_mode = PrecisionMode::Bf16Autocast;
    delta_q_config.delta_q_sidecar_path = Some(PathBuf::from("/data/delta-q.sidecar"));
    delta_q_config.advanced_loss = Some(AdvancedLossConfig {
        delta_q: Some(0.1),
        ..AdvancedLossConfig::default()
    });
    assert_eq!(
        validate_config(&delta_q_config),
        Err("precision_mode=bf16_autocast is not supported for DeltaQ training yet".to_string())
    );
}

#[test]
fn validate_config_allows_bc_only_bf16_on_cuda() {
    let mut config = dummy_config();
    config.device = "cuda:0".to_string();
    config.precision_mode = PrecisionMode::Bf16Autocast;
    assert!(validate_config(&config).is_ok());
}

#[test]
fn validate_config_rejects_bf16_for_non_cuda_bc() {
    let mut config = dummy_config();
    config.device = "cpu".to_string();
    config.precision_mode = PrecisionMode::Bf16Autocast;
    assert_eq!(
        validate_config(&config),
        Err("precision_mode=bf16_autocast requires a CUDA device for BC training".to_string())
    );
}

#[test]
fn validate_config_rejects_default_bf16_for_delta_q_on_cuda() {
    let mut config = dummy_config();
    config.device = "cuda:0".to_string();
    config.precision_mode = PrecisionMode::Bf16Autocast;
    config.delta_q_sidecar_path = Some(PathBuf::from("/data/delta-q.sidecar"));
    config.advanced_loss = Some(AdvancedLossConfig {
        delta_q: Some(0.1),
        ..AdvancedLossConfig::default()
    });
    assert_eq!(
        validate_config(&config),
        Err("precision_mode=bf16_autocast is not supported for DeltaQ training yet".to_string())
    );
}

#[test]
fn validate_preflight_config_rejects_safe_mode_with_unsafe_candidates() {
    let mut preflight = PreflightConfig::default();
    assert!(preflight.unsafe_candidate_batch_sizes.is_empty());
    assert!(validate_preflight_config(&preflight).is_ok());

    preflight.unsafe_candidate_batch_sizes = vec![512];
    assert_eq!(
        validate_preflight_config(&preflight),
        Err(
            "preflight.unsafe_candidate_batch_sizes requires preflight.tuning_mode = unsafe"
                .to_string()
        )
    );

    preflight.unsafe_candidate_batch_sizes.clear();
    preflight.unsafe_candidate_lr_scales = vec![1.5];
    assert_eq!(
        validate_preflight_config(&preflight),
        Err(
            "preflight.unsafe_candidate_lr_scales requires preflight.tuning_mode = unsafe"
                .to_string()
        )
    );

    preflight.unsafe_candidate_lr_scales.clear();
    preflight.unsafe_candidate_warmup_steps = vec![500];
    assert_eq!(
        validate_preflight_config(&preflight),
        Err(
            "preflight.unsafe_candidate_warmup_steps requires preflight.tuning_mode = unsafe"
                .to_string()
        )
    );
}

#[test]
fn validate_preflight_config_rejects_unsafe_mode_without_batch_candidates() {
    let mut preflight = PreflightConfig {
        tuning_mode: PreflightTuningMode::Unsafe,
        ..PreflightConfig::default()
    };
    assert_eq!(
        validate_preflight_config(&preflight),
        Err(
            "preflight.unsafe_candidate_batch_sizes must be non-empty when preflight.tuning_mode = unsafe"
                .to_string()
        )
    );

    preflight.unsafe_candidate_batch_sizes = vec![256, 0];
    assert_eq!(
        validate_preflight_config(&preflight),
        Err("preflight.unsafe_candidate_batch_sizes entries must be greater than 0".to_string())
    );
}

#[test]
fn validate_preflight_config_rejects_non_finite_or_non_positive_unsafe_lr_scales() {
    let mut preflight = PreflightConfig {
        tuning_mode: PreflightTuningMode::Unsafe,
        unsafe_candidate_batch_sizes: vec![512],
        unsafe_candidate_lr_scales: vec![1.0, 0.0],
        ..PreflightConfig::default()
    };
    assert_eq!(
        validate_preflight_config(&preflight),
        Err(
            "preflight.unsafe_candidate_lr_scales entries must be finite and greater than 0"
                .to_string()
        )
    );

    preflight.unsafe_candidate_lr_scales = vec![1.0];
    preflight.unsafe_candidate_warmup_steps = vec![100, 0];
    assert_eq!(
        validate_preflight_config(&preflight),
        Err("preflight.unsafe_candidate_warmup_steps entries must be greater than 0".to_string())
    );

    preflight.unsafe_candidate_warmup_steps = vec![100];
    assert!(validate_preflight_config(&preflight).is_ok());
}

#[test]
fn validate_config_no_longer_validates_preflight_tuning() {
    let config = dummy_config();
    assert!(validate_config(&config).is_ok());
}

#[test]
fn rl_and_loader_runtime_configs_map_expected_fields() {
    let mut config = dummy_config();
    let _rl = RlTrainConfig {
        games_per_batch: 4,
        temperature: 1.25,
        phase: RlPhaseConfig::ExitPondering,
        learning_rate: Some(3e-4),
        exit_weight: Some(0.7),
        aux_weight: Some(0.2),
        microbatch_size: None,
        run_forever: false,
        rollout_inference: None,
        bc_kl_reverse_coef: None,
        lr_warmup_samples: None,
        lr_decay_samples: None,
        arena_batch_decisions: None,
    };

    config.num_threads = None;
    let loader_cfg = loader_runtime_config(&config);
    assert_eq!(loader_cfg.buffer_games, config.buffer_games);
    assert_eq!(loader_cfg.buffer_samples, config.buffer_samples);
    assert_eq!(loader_cfg.archive_queue_bound, config.archive_queue_bound);
    assert!(loader_cfg.num_threads.is_some());
}

#[test]
fn validation_helpers_prefer_explicit_sample_limit_then_batch_limit() {
    let mut config = dummy_config();
    config.max_validation_batches = Some(5);
    assert_eq!(validation_sample_limit(&config), Some(160));

    config.max_validation_samples = Some(77);
    assert_eq!(validation_sample_limit(&config), Some(77));

    assert_eq!(train_microbatch_size(&config), 64);
    assert_eq!(validation_microbatch_size(&config), 32);
}

#[test]
fn shard_prefetch_depth_defaults_and_validates_bounds() {
    let mut config = dummy_config();
    config.shard_prefetch_depth = None;
    assert_eq!(shard_prefetch_depth(&config), 2);
    assert!(validate_config(&config).is_ok());

    config.shard_prefetch_depth = Some(4);
    assert_eq!(shard_prefetch_depth(&config), 4);
    assert!(validate_config(&config).is_ok());

    config.shard_prefetch_depth = Some(0);
    assert_eq!(
        validate_config(&config),
        Err("shard_prefetch_depth must be greater than 0".to_string())
    );

    config.shard_prefetch_depth = Some(65);
    assert_eq!(
        validate_config(&config),
        Err("shard_prefetch_depth must be at most 64".to_string())
    );
}

#[test]
fn validation_sample_limit_uses_validation_microbatch_size_when_only_batch_limit_is_set() {
    let mut config = dummy_config();
    config.max_validation_batches = Some(3);
    config.max_validation_samples = None;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = None;

    assert_eq!(validation_microbatch_size(&config), 64);
    assert_eq!(validation_sample_limit(&config), Some(192));

    config.validation_microbatch_size = Some(20);
    assert_eq!(validation_microbatch_size(&config), 20);
    assert_eq!(validation_sample_limit(&config), Some(60));
}
