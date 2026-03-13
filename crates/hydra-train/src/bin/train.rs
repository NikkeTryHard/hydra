#[path = "train/artifacts.rs"]
mod artifacts;
#[path = "train/bootstrap.rs"]
mod bootstrap;
#[path = "train/config.rs"]
mod config;
#[path = "train/config_runtime.rs"]
mod config_runtime;
#[path = "train/epoch_runner.rs"]
mod epoch_runner;
#[path = "train/loss_policy.rs"]
mod loss_policy;
#[path = "train/modes.rs"]
mod modes;
#[path = "train/preflight_runtime.rs"]
mod preflight_runtime;
#[path = "train/preflight_fingerprint.rs"]
mod preflight_fingerprint;
#[path = "train/probe_request.rs"]
mod probe_request;
#[path = "train/probe_process.rs"]
mod probe_process;
#[path = "train/probe_ladder.rs"]
mod probe_ladder;
#[path = "train/probe_summary.rs"]
mod probe_summary;
#[path = "train/presentation.rs"]
mod presentation;
#[path = "train/progress.rs"]
mod progress;
#[path = "train/runtime_autotune.rs"]
mod runtime_autotune;
#[path = "train/resume.rs"]
mod resume;
#[path = "train/schedule.rs"]
mod schedule;
#[path = "train/status.rs"]
mod status;
#[path = "train/validation.rs"]
mod validation;

use std::env;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

use burn::backend::{Autodiff, LibTorch};
use colored::control as color_control;

use self::config::{parse_args, read_config};
use self::modes::{handle_preflight_mode, handle_probe_mode, handle_training_mode};
use self::preflight_runtime::run_probe_child_mode;
use self::probe_request::probe_request_from_cli;

#[cfg(test)]
use self::config::{
    AdvancedLossConfig, BcHyperparamConfig, TrainConfig, default_seed, validation_microbatch_size,
    validation_sample_limit,
};
#[cfg(test)]
use self::resume::{
    BcResumeState, ResumeSemantics, build_resume_state, checkpoint_base_from_path,
    latest_optimizer_base_for_checkpoint_base, latest_state_path_for_checkpoint_base,
    read_resume_state, resume_banner_message, test_runtime_resume_contract,
};
#[cfg(test)]
use hydra_train::preflight::PreflightConfig;

type TrainBackend = Autodiff<LibTorch<f32>>;
type ValidBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

fn run() -> Result<(), String> {
    color_control::set_override(true);
    let cli = parse_args(env::args())?;
    let config = read_config(&cli.config_path)?;
    if run_probe_child_mode(&config, cli.probe_child.clone())? {
        return Ok(());
    }
    if cli.preflight {
        return handle_preflight_mode(&cli.config_path, &config);
    }
    if let Some(request) = probe_request_from_cli(&config, cli.probe_only.clone())? {
        return handle_probe_mode(&cli.config_path, &config, request);
    }
    handle_training_mode(&cli.config_path, config)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::libtorch::LibTorchDevice;
    use burn::prelude::*;
    use hydra_train::training::bc::policy_agreement_counts;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::artifacts::BcArtifactPaths;
    use crate::config::{train_device, train_microbatch_size, validate_config};
    use crate::loss_policy::build_loss_config;
    use crate::presentation::{format_progress_message, phase_label};
    use crate::resume::{
        BestValidation, EpochContinuation, paused_training_message,
        validate_resume_runtime_compatibility,
    };
    use crate::schedule::{lr_status_message, schedule_total_steps, steps_per_second};
    use crate::status::{
        EpochProgressEstimate, display_step_label, display_validation_scope_label,
        epoch_progress_message_with_rate, estimate_epoch_progress, format_rough_duration,
        reached_session_step_budget, session_steps_completed,
    };
    use crate::validation::{ValidationSummary, is_better_validation};

    #[test]
    fn parse_args_accepts_single_config_path() {
        let args = vec!["train".to_string(), "config.yaml".to_string()];
        let parsed = parse_args(args).expect("single config arg should parse");
        assert_eq!(parsed.config_path, PathBuf::from("config.yaml"));
        assert!(!parsed.preflight);
        assert!(parsed.probe_only.is_none());
        assert!(parsed.probe_child.is_none());
    }

    #[test]
    fn parse_args_rejects_missing_config() {
        let args = vec!["train".to_string()];
        let err = parse_args(args).expect_err("missing config should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn parse_args_rejects_extra_args() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "extra".to_string(),
        ];
        let err = parse_args(args).expect_err("extra args should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn parse_args_accepts_probe_only_flags() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--probe-kind".to_string(),
            "train".to_string(),
            "--probe-candidate-microbatch".to_string(),
            "192".to_string(),
            "--probe-warmup-steps".to_string(),
            "4".to_string(),
            "--probe-measure-steps".to_string(),
            "8".to_string(),
        ];
        let parsed = parse_args(args).expect("probe args should parse");
        let probe = parsed.probe_only.expect("probe_only should be present");
        assert_eq!(probe.kind, hydra_train::preflight::ProbeKind::Train);
        assert_eq!(probe.candidate_microbatch, 192);
        assert_eq!(probe.warmup_steps, Some(4));
        assert_eq!(probe.measure_steps, Some(8));
        assert!(!parsed.preflight);
    }

    #[test]
    fn parse_args_accepts_preflight_flag() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--preflight".to_string(),
        ];
        let parsed = parse_args(args).expect("preflight arg should parse");
        assert!(parsed.preflight);
        assert!(parsed.probe_only.is_none());
        assert!(parsed.probe_child.is_none());
    }

    #[test]
    fn parse_args_rejects_preflight_with_probe_flags() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--preflight".to_string(),
            "--probe-kind".to_string(),
            "train".to_string(),
            "--probe-candidate-microbatch".to_string(),
            "192".to_string(),
        ];
        let err = parse_args(args).expect_err("mixed preflight/probe flags should fail");
        assert!(err.contains("--preflight cannot be combined"));
    }

    #[test]
    fn parse_args_rejects_partial_probe_flags() {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--probe-kind".to_string(),
            "train".to_string(),
        ];
        let err = parse_args(args).expect_err("partial probe args should fail");
        assert!(
            err.contains("probe mode requires both --probe-kind and --probe-candidate-microbatch")
        );
    }

    #[test]
    fn read_config_applies_defaults() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_config_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 3
"#;
        fs::write(&base, yaml).expect("write config");
        let cfg = read_config(&base).expect("read config");
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/data"));
        assert_eq!(cfg.output_dir, PathBuf::from("/tmp/out"));
        assert_eq!(cfg.num_epochs, 3);
        assert_eq!(cfg.batch_size, 2048);
        assert!(cfg.microbatch_size.is_none());
        assert!(cfg.validation_microbatch_size.is_none());
        assert!((cfg.train_fraction - 0.9).abs() < f32::EPSILON);
        assert!(cfg.augment);
        assert_eq!(cfg.seed, 0);
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.buffer_games, 50_000);
        assert_eq!(cfg.buffer_samples, 32_768);
        assert!(cfg.num_threads.is_none());
        assert!(cfg.tensorboard);
        assert_eq!(cfg.archive_queue_bound, 128);
        assert_eq!(cfg.validation_every_n_epochs, 1);
        assert_eq!(cfg.max_skip_logs_per_source, 32);
        assert!(cfg.max_validation_batches.is_none());
        assert_eq!(cfg.max_validation_samples, Some(8_192));
        assert!(cfg.advanced_loss.is_none());
        fs::remove_file(base).ok();
    }

    #[test]
    fn bc_artifact_paths_use_bc_subdir_and_unique_tb_session() {
        let paths = BcArtifactPaths::new(Path::new("/tmp/out"), 42);
        assert_eq!(paths.root, PathBuf::from("/tmp/out/bc"));
        assert_eq!(
            paths.latest_model_base,
            PathBuf::from("/tmp/out/bc/latest_model")
        );
        assert_eq!(
            paths.best_model_base,
            PathBuf::from("/tmp/out/bc/best_model")
        );
        assert_eq!(
            paths.latest_state_path,
            PathBuf::from("/tmp/out/bc/latest_state.yaml")
        );
        assert_eq!(
            paths.training_log_path,
            PathBuf::from("/tmp/out/bc/training_log.jsonl")
        );
        assert_eq!(
            paths.step_log_path,
            PathBuf::from("/tmp/out/bc/step_log.jsonl")
        );
        assert!(
            paths
                .tb_session_dir
                .starts_with(Path::new("/tmp/out/bc/tb"))
        );
        assert_ne!(paths.tb_session_dir, paths.tb_root);
    }

    #[test]
    fn checkpoint_base_from_path_strips_mpk_only() {
        assert_eq!(
            checkpoint_base_from_path(Path::new("/tmp/out/bc/latest_model.mpk")),
            PathBuf::from("/tmp/out/bc/latest_model")
        );
        assert_eq!(
            checkpoint_base_from_path(Path::new("/tmp/out/bc/latest_model")),
            PathBuf::from("/tmp/out/bc/latest_model")
        );
    }

    #[test]
    fn latest_state_path_is_only_available_for_latest_model() {
        assert_eq!(
            latest_state_path_for_checkpoint_base(Path::new("/tmp/out/bc/latest_model")),
            Some(PathBuf::from("/tmp/out/bc/latest_state.yaml"))
        );
        assert_eq!(
            latest_state_path_for_checkpoint_base(Path::new("/tmp/out/bc/best_model")),
            None
        );
        assert_eq!(
            latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/out/bc/latest_model")),
            Some(PathBuf::from("/tmp/out/bc/latest_optimizer"))
        );
        assert_eq!(
            latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/out/bc/best_model")),
            None
        );
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hydra_train_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn read_resume_state_rejects_legacy_resume_semantics() {
        let dir = unique_temp_dir("legacy_resume_state");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let state_path = dir.join("latest_state.yaml");
        let legacy_yaml = r#"schema_version: 2
resume_semantics: ReplaySkippedStepsFreshOptimizer
next_epoch: 1
skip_optimizer_steps_in_epoch: 12
global_step: 400
best_validation:
  policy_loss: 1.5
  agreement: 0.4
runtime:
  batch_size: 2048
  train_microbatch_size: 256
  validation_microbatch_size: 128
  accum_steps: 8
saved_at_unix_s: 123
"#;
        std::fs::write(&state_path, legacy_yaml).expect("write legacy state");

        let err = read_resume_state(&state_path).expect_err("legacy resume state should fail");
        assert!(err.contains("failed to parse resume state"));
        std::fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn read_resume_state_rejects_unknown_fields() {
        let dir = unique_temp_dir("resume_unknown_field");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let state_path = dir.join("latest_state.yaml");
        let yaml = r#"schema_version: 3
resume_semantics: RestoreOptimizerSkipSeenSamples
next_epoch: 1
skip_optimizer_steps_in_epoch: 12
global_step: 400
best_validation:
  policy_loss: 1.5
  agreement: 0.4
runtime:
  batch_size: 2048
  train_microbatch_size: 256
  validation_microbatch_size: 128
  accum_steps: 8
saved_at_unix_s: 123
unexpected_field: true
"#;
        std::fs::write(&state_path, yaml).expect("write invalid state");
        let err = read_resume_state(&state_path).expect_err("unknown field should fail");
        assert!(err.contains("failed to parse resume state"));
        std::fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn session_step_budget_is_relative_to_resume_point() {
        assert_eq!(session_steps_completed(1250, 1000), 250);
        assert!(reached_session_step_budget(1200, 1000, Some(200)));
        assert!(!reached_session_step_budget(1199, 1000, Some(200)));
        assert_eq!(
            display_step_label(1200, 1000, Some(200)),
            "step 200/200 global=1200"
        );
        assert_eq!(
            display_validation_scope_label(1100, 1000, Some(200)),
            "validation @ step 100/200 global=1100"
        );
    }

    #[test]
    fn schedule_total_steps_extends_from_resume_global_step() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(16),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: BcHyperparamConfig::default(),
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 25,
            validate_every_n_steps: 200,
            checkpoint_every_n_steps: 200,
            max_train_steps: Some(1000),
            max_validation_batches: None,
            max_validation_samples: Some(8192),
            preflight: PreflightConfig::default(),
        };
        assert_eq!(schedule_total_steps(&cfg, 0), 1000);
        assert_eq!(schedule_total_steps(&cfg, 400), 1400);
    }

    #[test]
    fn resume_state_yaml_roundtrip_preserves_fields() {
        let state = build_resume_state(
            0,
            37,
            137,
            Some(BestValidation {
                policy_loss: 1.23,
                agreement: 0.45,
            }),
            test_runtime_resume_contract(2048, 256, 256),
        );
        let yaml = serde_yaml::to_string(&state).expect("serialize state");
        let parsed: BcResumeState = serde_yaml::from_str(&yaml).expect("parse state");
        assert_eq!(parsed.schema_version, 3);
        assert_eq!(
            parsed.resume_semantics,
            ResumeSemantics::RestoreOptimizerSkipSeenSamples
        );
        assert_eq!(parsed.next_epoch, 0);
        assert_eq!(parsed.skip_optimizer_steps_in_epoch, 37);
        assert_eq!(parsed.global_step, 137);
        assert_eq!(parsed.best_validation, state.best_validation);
        assert_eq!(parsed.runtime, state.runtime);
    }

    #[test]
    fn resume_banner_message_mentions_replay_when_needed() {
        let state = build_resume_state(
            2,
            137,
            2048,
            Some(BestValidation {
                policy_loss: 1.5,
                agreement: 0.41,
            }),
            test_runtime_resume_contract(2048, 256, 128),
        );
        assert_eq!(
            resume_banner_message(&state),
            "global_step=2048 semantics=RestoreOptimizerSkipSeenSamples skipping 137 completed optimizer steps worth of samples in epoch 3 before new updates runtime=train_mb:256 val_mb:128 accum_steps:8"
        );
    }

    #[test]
    fn resume_banner_message_mentions_immediate_updates_when_no_replay() {
        let state = build_resume_state(
            1,
            0,
            500,
            None,
            test_runtime_resume_contract(2048, 512, 256),
        );
        assert_eq!(
            resume_banner_message(&state),
            "global_step=500 semantics=RestoreOptimizerSkipSeenSamples resuming at epoch 2 with new updates immediately runtime=train_mb:512 val_mb:256 accum_steps:4"
        );
    }

    #[test]
    fn paused_training_message_spells_out_resume_contract() {
        let continuation = EpochContinuation {
            next_epoch: 0,
            skip_optimizer_steps_in_epoch: 88,
            epoch_completed: false,
        };
        assert_eq!(
            paused_training_message(&continuation),
            "resume_epoch=1 skipped_optimizer_steps_in_epoch=88 optimizer_state=restored sample_cursor=reconstructed_from_logical_batch_count partial_epoch_requires_matching_runtime"
        );
    }

    #[test]
    fn partial_epoch_resume_rejects_runtime_mismatch() {
        let state = build_resume_state(
            0,
            12,
            400,
            None,
            test_runtime_resume_contract(2048, 256, 128),
        );
        let err = validate_resume_runtime_compatibility(
            &state,
            test_runtime_resume_contract(2048, 512, 128),
        )
        .expect_err("partial epoch resume should fail when runtime differs");
        assert!(err.contains("partial-epoch resume requires identical runtime contract"));
    }

    #[test]
    fn epoch_boundary_resume_allows_runtime_change_with_same_batch_size() {
        let state = build_resume_state(
            1,
            0,
            400,
            None,
            test_runtime_resume_contract(2048, 256, 128),
        );
        validate_resume_runtime_compatibility(&state, test_runtime_resume_contract(2048, 512, 256))
            .expect("epoch-boundary resume should allow new runtime contract");
    }

    #[test]
    fn read_config_supports_yaml_only() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_yaml_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        assert_eq!(read_config(&yaml_path).expect("yaml config").num_epochs, 1);
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_json_config() {
        let dir = std::env::temp_dir();
        let json_path = dir.join(format!(
            "hydra_train_json_{}_{}.json",
            std::process::id(),
            default_seed()
        ));
        let json = r#"{
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "num_epochs": 1
        }"#;
        std::fs::write(&json_path, json).expect("write json config");
        let err = read_config(&json_path).expect_err("json config should be rejected");
        assert!(err.contains("unsupported config extension"));
        assert!(err.contains("use .yaml"));
        std::fs::remove_file(json_path).ok();
    }

    #[test]
    fn read_config_rejects_unknown_top_level_fields() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_unknown_field_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
old_field: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("unknown field should fail");
        assert!(err.contains("failed to parse yaml config"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_legacy_preflight_probe_only_block() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_probe_only_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
        preflight:
  probe_only:
    kind: train
    candidate_microbatch: 256
    warmup_steps: 5
    measure_steps: 7
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("legacy probe-only config should fail");
        assert!(err.contains("probe_only"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_accepts_bc_hyperparameter_block() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_bc_block_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
bc:
  learning_rate: 1.0e-4
  min_learning_rate: 1.0e-5
  weight_decay: 2.0e-5
  grad_clip_norm: 0.5
  warmup_steps: 321
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let config = read_config(&yaml_path).expect("bc block should parse");
        assert!((config.bc.learning_rate - 1.0e-4).abs() < 1e-12);
        assert!((config.bc.min_learning_rate - 1.0e-5).abs() < 1e-12);
        assert!((config.bc.weight_decay - 2.0e-5).abs() < 1e-12);
        assert!((config.bc.grad_clip_norm - 0.5).abs() < 1e-6);
        assert_eq!(config.bc.warmup_steps, 321);
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_unknown_bc_fields() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_bc_unknown_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
bc:
  learning_rate: 1.0e-4
  old_knob: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("unknown bc field should fail");
        assert!(err.contains("failed to parse yaml config"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_probe_only_block_entirely() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_probe_only_unknown_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  probe_only:
    kind: train
    candidate_microbatch: 256
    mystery_field: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("probe_only block should fail");
        assert!(err.contains("failed to parse yaml config"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_removed_preflight_enabled_field() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_preflight_enabled_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  enabled: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("removed enabled field should fail");
        assert!(err.contains("enabled"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_removed_preflight_reuse_cache_field() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_preflight_reuse_cache_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  reuse_cache: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("removed reuse_cache field should fail");
        assert!(err.contains("reuse_cache"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_removed_preflight_advisory_only_field() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_preflight_advisory_only_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  advisory_only: true
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("removed advisory_only field should fail");
        assert!(err.contains("advisory_only"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn read_config_rejects_removed_preflight_safety_backoff_rungs_field() {
        let dir = std::env::temp_dir();
        let yaml_path = dir.join(format!(
            "hydra_train_preflight_safety_backoff_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  safety_backoff_rungs: 1
"#;
        std::fs::write(&yaml_path, yaml).expect("write yaml config");
        let err = read_config(&yaml_path).expect_err("removed safety_backoff_rungs should fail");
        assert!(err.contains("safety_backoff_rungs"));
        std::fs::remove_file(yaml_path).ok();
    }

    #[test]
    fn estimate_epoch_progress_returns_none_without_exact_counts() {
        let manifest = hydra_train::data::pipeline::DataManifest {
            sources: vec![],
            total_games: 0,
            train_count: 100,
            val_count: 0,
            counts_exact: false,
        };
        assert_eq!(
            estimate_epoch_progress(&manifest, 10_000, 10, 25, 256),
            None
        );
    }

    #[test]
    fn estimate_epoch_progress_computes_remaining_steps() {
        let manifest = hydra_train::data::pipeline::DataManifest {
            sources: vec![],
            total_games: 100,
            train_count: 100,
            val_count: 0,
            counts_exact: true,
        };
        let progress = estimate_epoch_progress(&manifest, 12_800, 10, 40, 256)
            .expect("exact counts should yield estimate");
        assert_eq!(progress.completed_optimizer_steps, 40);
        assert_eq!(progress.estimated_total_optimizer_steps, 500);
        assert_eq!(progress.estimated_remaining_optimizer_steps, 460);
        assert!((progress.completion_fraction - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn epoch_progress_message_formats_estimate_and_pending() {
        assert_eq!(
            epoch_progress_message_with_rate(None, None),
            "epoch=pending"
        );
        assert_eq!(
            epoch_progress_message_with_rate(
                Some(EpochProgressEstimate {
                    completed_optimizer_steps: 200,
                    estimated_total_optimizer_steps: 500,
                    estimated_remaining_optimizer_steps: 300,
                    completion_fraction: 0.4,
                }),
                None,
            ),
            "epoch=40.0% epoch_left≈300 steps"
        );
    }

    #[test]
    fn format_rough_duration_prefers_human_sized_units() {
        assert_eq!(format_rough_duration(12.2), "~12s");
        assert_eq!(format_rough_duration(125.0), "~2m5s");
        assert_eq!(format_rough_duration(3720.0), "~1h2m");
    }

    #[test]
    fn epoch_progress_message_with_rate_appends_rough_eta() {
        let message = epoch_progress_message_with_rate(
            Some(EpochProgressEstimate {
                completed_optimizer_steps: 200,
                estimated_total_optimizer_steps: 500,
                estimated_remaining_optimizer_steps: 300,
                completion_fraction: 0.4,
            }),
            Some(2.0),
        );
        assert_eq!(message, "epoch=40.0% epoch_left≈300 steps rough_eta=~2m30s");
    }

    #[test]
    fn better_validation_prefers_lower_policy_loss_then_higher_agreement() {
        let summary = ValidationSummary {
            total_loss: 2.0,
            policy_loss: 1.0,
            agreement: 0.35,
            samples: 8192,
        };
        assert!(is_better_validation(summary, None));

        let best = BestValidation {
            policy_loss: 1.1,
            agreement: 0.60,
        };
        assert!(is_better_validation(summary, Some(best)));

        let tied = ValidationSummary {
            total_loss: 2.1,
            policy_loss: 1.0,
            agreement: 0.40,
            samples: 8192,
        };
        assert!(is_better_validation(
            tied,
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.39
            })
        ));
        assert!(!is_better_validation(
            tied,
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.41
            })
        ));
    }

    #[test]
    fn validation_microbatch_and_sample_limit_fallbacks_work() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: BcHyperparamConfig::default(),
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
            max_validation_batches: Some(32),
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
        };

        assert_eq!(train_microbatch_size(&cfg), 64);
        assert_eq!(validation_microbatch_size(&cfg), 64);
        assert_eq!(validation_sample_limit(&cfg), Some(2048));

        let cfg = TrainConfig {
            validation_microbatch_size: Some(32),
            max_validation_batches: Some(32),
            max_validation_samples: Some(1500),
            ..cfg
        };

        assert_eq!(validation_microbatch_size(&cfg), 32);
        assert_eq!(validation_sample_limit(&cfg), Some(1500));
    }

    #[test]
    fn validate_config_rejects_zero_validation_microbatch_and_samples() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(0),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: BcHyperparamConfig::default(),
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
            max_validation_samples: Some(0),
            preflight: PreflightConfig::default(),
        };

        let err = validate_config(&cfg).expect_err("zero validation controls should fail");
        assert!(
            err.contains("max_validation_samples") || err.contains("validation_microbatch_size")
        );
    }

    #[test]
    fn build_loss_config_defaults_match_baseline() {
        let loss = build_loss_config(None).expect("default loss config should build");
        assert!((loss.w_safety_residual - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_allows_safety_residual_only() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(0.1),
            ..Default::default()
        };
        let loss = build_loss_config(Some(&advanced)).expect("safety residual should be allowed");
        assert!((loss.w_safety_residual - 0.1).abs() < 1e-6);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_rejects_negative_safety_residual() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(-0.1),
            ..Default::default()
        };
        let err =
            build_loss_config(Some(&advanced)).expect_err("negative safety residual should fail");
        assert!(err.contains("invalid loss config"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_activation() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("belief fields should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_even_at_zero() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.0),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("blocked belief fields key should be rejected even at zero");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_mixture_weight_activation() {
        let advanced = AdvancedLossConfig {
            mixture_weight: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("mixture weight should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.mixture_weight"));
    }

    #[test]
    fn build_loss_config_rejects_opponent_hand_type_activation() {
        let advanced = AdvancedLossConfig {
            opponent_hand_type: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("opponent hand type should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.opponent_hand_type"));
    }

    #[test]
    fn build_loss_config_rejects_delta_q_activation() {
        let advanced = AdvancedLossConfig {
            delta_q: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("delta_q should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.delta_q"));
    }

    #[test]
    fn read_config_rejects_unknown_advanced_loss_field() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_bad_advanced_loss_{}_{}.yaml",
            std::process::id(),
            default_seed()
        ));
        let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 3
advanced_loss:
  not_a_real_field: 0.1
"#;
        fs::write(&base, yaml).expect("write config");
        let err = read_config(&base).expect_err("unknown advanced loss field should fail");
        assert!(err.contains("not_a_real_field"));
        fs::remove_file(base).ok();
    }

    #[test]
    fn validate_config_rejects_invalid_bc_hyperparameter_ranges() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: BcHyperparamConfig {
                learning_rate: 1e-4,
                min_learning_rate: 2e-4,
                weight_decay: 1e-5,
                grad_clip_norm: 1.0,
                warmup_steps: 100,
            },
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
        };
        let err = validate_config(&cfg).expect_err("invalid bc ranges should fail");
        assert!(err.contains("bc.min_learning_rate"));
    }

    #[test]
    fn train_device_prefers_env_override_then_config() {
        struct EnvGuard;
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                unsafe {
                    env::remove_var("HYDRA_TRAIN_DEVICE");
                }
            }
        }
        let _guard = EnvGuard;
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cpu);
        assert_eq!(train_device("cuda:2"), LibTorchDevice::Cuda(2));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cuda:0");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cuda(0));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cpu");
        }
        assert_eq!(train_device("cuda:3"), LibTorchDevice::Cpu);
    }

    #[test]
    #[should_panic(expected = "unsupported HYDRA_TRAIN_DEVICE")]
    fn train_device_rejects_invalid_env_value() {
        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "vulkan");
        }
        struct EnvGuard;
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                unsafe {
                    env::remove_var("HYDRA_TRAIN_DEVICE");
                }
            }
        }
        let _guard = EnvGuard;
        let _ = train_device("cpu");
    }

    #[test]
    fn validation_agreement_is_sample_weighted_across_chunks() {
        let device: <burn::backend::ndarray::NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::from_floats(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
            &device,
        );
        let mask = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::ones([5, 2], &device);
        let targets = Tensor::<burn::backend::ndarray::NdArray<f32>, 1, Int>::from_ints(
            [0, 1, 1, 1, 1],
            &device,
        );

        let (chunk1_correct, chunk1_total) = policy_agreement_counts(
            logits.clone().slice([0..4, 0..2]),
            mask.clone().slice([0..4, 0..2]),
            targets.clone().slice(0..4),
        );
        let (chunk2_correct, chunk2_total) = policy_agreement_counts(
            logits.slice([4..5, 0..2]),
            mask.slice([4..5, 0..2]),
            targets.slice(4..5),
        );

        let weighted =
            (chunk1_correct + chunk2_correct) as f64 / (chunk1_total + chunk2_total) as f64;
        let naive_chunk_average = ((chunk1_correct as f64 / chunk1_total as f64)
            + (chunk2_correct as f64 / chunk2_total as f64))
            / 2.0;

        assert!((weighted - 0.4).abs() < 1e-12);
        assert!((naive_chunk_average - 0.625).abs() < 1e-12);
    }

    #[test]
    fn phase_label_hides_redundant_single_epoch_denominator() {
        assert_eq!(phase_label("train", 0, 1), "train");
        assert_eq!(phase_label("train", 1, 3), "train 2/3");
    }

    #[test]
    fn lr_status_message_marks_warmup_and_cosine() {
        assert_eq!(
            lr_status_message(25, 100, 1.25e-4),
            "lr=1.25e-4 warmup 25/100"
        );
        assert_eq!(lr_status_message(100, 100, 2.50e-4), "lr=2.50e-4 cosine");
    }

    #[test]
    fn steps_per_second_and_progress_message_are_stable() {
        assert_eq!(steps_per_second(0, Duration::from_secs(1)), 0.0);
        assert_eq!(steps_per_second(10, Duration::from_secs(0)), 0.0);
        assert!((steps_per_second(10, Duration::from_secs(2)) - 5.0).abs() < 1e-12);
        assert_eq!(
            format_progress_message(3.0, 0.25, "lr=1.00e-4 cosine", 5.5),
            "loss=3.0000 agree=25.00% steps/s=5.50 lr=1.00e-4 cosine"
        );
    }
}
