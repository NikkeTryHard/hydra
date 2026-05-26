use super::*;
#[cfg(feature = "burn-libtorch-obsolete")]
use burn::backend::libtorch::LibTorchDevice;
#[cfg(feature = "burn-libtorch-obsolete")]
use burn::prelude::*;
#[cfg(feature = "burn-libtorch-obsolete")]
use hydra_train_algo::bc::policy_agreement_counts;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "burn-libtorch-obsolete")]
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use hydra_train_exec::artifacts::BcArtifactPaths;
#[cfg(feature = "burn-libtorch-obsolete")]
use hydra_train_exec::config_runtime::train_device;
use hydra_train_exec::presentation::{format_progress_message, progress_label};
use hydra_train_exec::resume::{
    BestValidation, EpochContinuation, paused_training_message,
    validate_resume_runtime_compatibility,
};
use hydra_train_exec::validation::{ValidationSummary, is_better_validation};
use hydra_train_runtime::config::{train_microbatch_size, validate_config};
use hydra_train_runtime::loss_policy::build_loss_config;
use hydra_train_runtime::schedule::{lr_status_message, schedule_total_steps, steps_per_second};
use hydra_train_runtime::status::{
    EpochProgressEstimate, display_step_label, display_validation_scope_label,
    epoch_progress_message_with_rate, estimate_epoch_progress, format_rough_duration,
    reached_session_step_budget, session_steps_completed,
};

#[test]
fn parse_args_accepts_single_config_path() {
    let args = vec!["train".to_string(), "config.yaml".to_string()];
    let parsed = parse_args(args).expect("single config arg should parse");
    assert_eq!(parsed.config_path, Some(PathBuf::from("config.yaml")));
    assert!(parsed.preflight.is_none());
    assert!(!parsed.delta_q_promotion);
    assert!(parsed.probe_only.is_none());
    assert!(parsed.benchmark_baseline.is_none());
    assert!(parsed.probe_child.is_none());
}

#[test]
fn parse_args_rejects_missing_config() {
    let args = vec!["train".to_string()];
    let err = parse_args(args).expect_err("missing config should fail");
    assert!(err.contains("Usage:"));
}

#[test]
fn parse_args_reports_help_flag() {
    for flag in ["--help", "-h"] {
        let args = vec!["train".to_string(), flag.to_string()];
        let err = parse_args(args).expect_err("help flag should short-circuit");
        assert!(err.contains("Usage:"));
    }
}

#[test]
fn parse_args_reports_version_flag() {
    for flag in ["--version", "-V"] {
        let args = vec!["train".to_string(), flag.to_string()];
        let err = parse_args(args).expect_err("version flag should short-circuit");
        assert!(err.contains(env!("CARGO_PKG_VERSION")));
    }
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
fn parse_args_help_flag_after_config_short_circuits() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--help".to_string(),
    ];
    let err = parse_args(args).expect_err("help after config should short-circuit");
    assert!(err.contains("Usage:"));
}

#[test]
fn parse_args_version_flag_after_config_short_circuits() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--version".to_string(),
    ];
    let err = parse_args(args).expect_err("version after config should short-circuit");
    assert!(err.contains(env!("CARGO_PKG_VERSION")));
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
    assert_eq!(probe.kind, hydra_train_runtime::preflight::ProbeKind::Train);
    assert_eq!(probe.candidate_microbatch, 192);
    assert_eq!(probe.warmup_steps, Some(4));
    assert_eq!(probe.measure_steps, Some(8));
    assert!(parsed.preflight.is_none());
}

#[test]
fn parse_args_accepts_probe_child_flags_after_libtest_separator() {
    let args = vec![
        "train".to_string(),
        "--".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "validation".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
        "--probe-warmup-steps".to_string(),
        "4".to_string(),
        "--probe-measure-steps".to_string(),
        "8".to_string(),
        "--probe-result-path".to_string(),
        "/tmp/probe.json".to_string(),
        "--probe-manifest-cache-path".to_string(),
        "/tmp/manifest.json".to_string(),
        "--probe-discovery-summary-path".to_string(),
        "/tmp/summary.json".to_string(),
        "--probe-discovery-index-path".to_string(),
        "/tmp/index.bin".to_string(),
    ];

    let parsed = parse_args(args).expect("libtest-separated probe child args should parse");
    assert_eq!(parsed.config_path, Some(PathBuf::from("config.yaml")));
    assert!(parsed.probe_only.is_none());
    match parsed.probe_child.expect("probe child should be present") {
        hydra_train_runtime::config::ProbeChildRequest::Single(child) => {
            assert_eq!(
                child.request.kind,
                hydra_train_runtime::preflight::ProbeKind::Validation
            );
            assert_eq!(child.request.candidate_microbatch, 192);
            assert_eq!(child.result_path, PathBuf::from("/tmp/probe.json"));
            assert_eq!(
                child.discovery_summary_path,
                Some(PathBuf::from("/tmp/summary.json"))
            );
            assert_eq!(
                child.discovery_index_path,
                Some(PathBuf::from("/tmp/index.bin"))
            );
        }
        hydra_train_runtime::config::ProbeChildRequest::Batch(_) => {
            panic!("single probe child flags should stay on the single-request path")
        }
    }
}

#[test]
fn parse_args_accepts_single_probe_child_flags_unchanged() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "validation".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
        "--probe-warmup-steps".to_string(),
        "4".to_string(),
        "--probe-measure-steps".to_string(),
        "8".to_string(),
        "--probe-result-path".to_string(),
        "/tmp/probe.json".to_string(),
        "--probe-manifest-cache-path".to_string(),
        "/tmp/manifest.json".to_string(),
        "--probe-discovery-summary-path".to_string(),
        "/tmp/summary.json".to_string(),
        "--probe-discovery-index-path".to_string(),
        "/tmp/index.bin".to_string(),
    ];
    let parsed = parse_args(args).expect("single probe child args should parse");

    assert!(parsed.probe_only.is_none());
    match parsed.probe_child.expect("probe child should be present") {
        hydra_train_runtime::config::ProbeChildRequest::Single(child) => {
            assert_eq!(
                child.request.kind,
                hydra_train_runtime::preflight::ProbeKind::Validation
            );
            assert_eq!(child.request.candidate_microbatch, 192);
            assert_eq!(child.request.warmup_steps, Some(4));
            assert_eq!(child.request.measure_steps, Some(8));
            assert_eq!(child.result_path, PathBuf::from("/tmp/probe.json"));
            assert_eq!(
                child.manifest_cache_path,
                Some(PathBuf::from("/tmp/manifest.json"))
            );
            assert_eq!(
                child.discovery_summary_path,
                Some(PathBuf::from("/tmp/summary.json"))
            );
            assert_eq!(
                child.discovery_index_path,
                Some(PathBuf::from("/tmp/index.bin"))
            );
        }
        hydra_train_runtime::config::ProbeChildRequest::Batch(_) => {
            panic!("single probe child flags should stay on the single-request path")
        }
    }
}

#[test]
fn parse_args_accepts_internal_probe_batch_child_flags() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "256".to_string(),
        "--probe-warmup-steps".to_string(),
        "5".to_string(),
        "--probe-measure-steps".to_string(),
        "9".to_string(),
        "--probe-attempts".to_string(),
        "3".to_string(),
        "--probe-results-path".to_string(),
        "/tmp/probe-results.json".to_string(),
        "--probe-manifest-cache-path".to_string(),
        "/tmp/manifest.json".to_string(),
        "--probe-discovery-summary-path".to_string(),
        "/tmp/summary.json".to_string(),
        "--probe-discovery-index-path".to_string(),
        "/tmp/index.bin".to_string(),
    ];
    let parsed = parse_args(args).expect("internal probe batch child args should parse");

    assert!(parsed.probe_only.is_none());
    match parsed
        .probe_child
        .expect("probe batch child should be present")
    {
        hydra_train_runtime::config::ProbeChildRequest::Batch(child) => {
            assert_eq!(
                child.request.kind,
                hydra_train_runtime::preflight::ProbeKind::Train
            );
            assert_eq!(child.request.candidate_microbatch, 256);
            assert_eq!(child.request.warmup_steps, Some(5));
            assert_eq!(child.request.measure_steps, Some(9));
            assert_eq!(child.attempts, 3);
            assert_eq!(child.results_path, PathBuf::from("/tmp/probe-results.json"));
            assert_eq!(
                child.manifest_cache_path,
                Some(PathBuf::from("/tmp/manifest.json"))
            );
            assert_eq!(
                child.discovery_summary_path,
                Some(PathBuf::from("/tmp/summary.json"))
            );
            assert_eq!(
                child.discovery_index_path,
                Some(PathBuf::from("/tmp/index.bin"))
            );
        }
        hydra_train_runtime::config::ProbeChildRequest::Single(_) => {
            panic!("batch probe child flags should stay on the batch-request path")
        }
    }
}

#[test]
fn parse_args_rejects_preflight_config_path() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--preflight".to_string(),
    ];
    let err = parse_args(args).expect_err("preflight with config path should fail");
    assert!(err.contains("--preflight does not accept a config path"));
}

#[test]
fn parse_args_accepts_configless_preflight_safe_defaults() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
        "--output-dir".to_string(),
        "bench-out".to_string(),
    ];
    let parsed = parse_args(args).expect("configless preflight arg should parse");
    let preflight = parsed.preflight.expect("preflight options should exist");
    assert!(parsed.config_path.is_none());
    assert_eq!(preflight.output_dir, PathBuf::from("bench-out"));
    assert_eq!(preflight.device, "cpu");
    assert_eq!(
        preflight.profile,
        hydra_train_runtime::config::PreflightProfile::Default
    );
    assert_eq!(
        preflight.preflight_config.tuning_mode,
        hydra_train_runtime::preflight::PreflightTuningMode::Safe
    );
    assert!(preflight.preflight_config.real_benchmark_enabled);
    assert!(!parsed.delta_q_promotion);
    assert!(parsed.probe_only.is_none());
    assert!(parsed.probe_child.is_none());
}

#[test]
fn parse_args_accepts_fast_repeated_run_profile() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--pf-profile".to_string(),
        "fast-repeated-run".to_string(),
    ];
    let parsed = parse_args(args).expect("fast repeated profile should parse");
    let preflight = parsed.preflight.expect("preflight options");
    assert_eq!(
        preflight.profile,
        hydra_train_runtime::config::PreflightProfile::FastRepeatedRun
    );
    assert!(preflight.preflight_config.fast_repeated_run_profile);
    assert_eq!(preflight.preflight_config.required_successes, 1);
    assert_eq!(preflight.preflight_config.warmup_steps, 1);
    assert_eq!(preflight.preflight_config.measure_steps, 1);
    assert_eq!(preflight.preflight_config.loader_runtime_rounds, 0);
    assert_eq!(preflight.preflight_config.loader_tuple_extra_samples, 0);
    assert!(!preflight.preflight_config.real_benchmark_enabled);
}

#[test]
fn parse_args_allows_fast_repeated_profile_with_unsafe_mode() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--preflight-mode".to_string(),
        "unsafe".to_string(),
        "--pf-profile".to_string(),
        "fast-repeated-run".to_string(),
        "--pf-unsafe-batch-size".to_string(),
        "1024,2048".to_string(),
    ];
    let parsed = parse_args(args).expect("unsafe fast profile should parse");
    let preflight = parsed.preflight.expect("preflight options");
    assert_eq!(
        preflight.preflight_config.tuning_mode,
        hydra_train_runtime::preflight::PreflightTuningMode::Unsafe
    );
    assert_eq!(
        preflight.preflight_config.unsafe_candidate_batch_sizes,
        vec![1024, 2048]
    );
    assert_eq!(preflight.preflight_config.required_successes, 1);
}

#[test]
fn parse_args_applies_profile_then_flag_overrides() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--pf-profile".to_string(),
        "fast-repeated-run".to_string(),
        "--preflight-mode".to_string(),
        "safe".to_string(),
        "--pf-loader-rounds".to_string(),
        "3".to_string(),
        "--pf-real-benchmark".to_string(),
        "1".to_string(),
    ];
    let parsed = parse_args(args).expect("profile with overrides should parse");
    let preflight = parsed.preflight.expect("preflight options");
    assert_eq!(preflight.preflight_config.loader_runtime_rounds, 3);
    assert!(preflight.preflight_config.real_benchmark_enabled);
    assert_eq!(preflight.preflight_config.required_successes, 1);
}

#[test]
fn parse_args_rejects_deprecated_candidate_microbatch() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--pf-candidate-microbatch".to_string(),
        "64-256*2".to_string(),
    ];
    let err = parse_args(args).expect_err("deprecated candidate microbatch should fail");
    assert!(err.contains("--pf-candidate-tuples"));
}

#[test]
fn parse_args_rejects_preflight_flags_without_preflight() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--pf-warmup-steps".to_string(),
        "1".to_string(),
    ];
    let err = parse_args(args).expect_err("pf flag without preflight should fail");
    assert_eq!(err, "--pf-* flags require --preflight");
}

#[test]
fn parse_args_rejects_unsafe_flags_in_safe_mode() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--preflight-mode".to_string(),
        "safe".to_string(),
        "--pf-unsafe-batch-size".to_string(),
        "1024".to_string(),
    ];
    let err = parse_args(args).expect_err("unsafe flag in safe mode should fail");
    assert!(err.contains("--preflight-mode unsafe"));
}

#[test]
fn parse_args_accepts_unsafe_flags_in_unsafe_mode() {
    let args = vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--preflight-mode".to_string(),
        "unsafe".to_string(),
        "--pf-unsafe-batch-size".to_string(),
        "1024,2048".to_string(),
        "--pf-unsafe-lr-scale".to_string(),
        "0.5,1.0".to_string(),
        "--pf-unsafe-warmup-steps".to_string(),
        "500-1500+500".to_string(),
    ];
    let parsed = parse_args(args).expect("unsafe flags should parse");
    let preflight = parsed.preflight.expect("preflight options");
    assert_eq!(
        preflight.preflight_config.unsafe_candidate_batch_sizes,
        vec![1024, 2048]
    );
    assert_eq!(
        preflight.preflight_config.unsafe_candidate_lr_scales,
        vec![0.5, 1.0]
    );
    assert_eq!(
        preflight.preflight_config.unsafe_candidate_warmup_steps,
        vec![500, 1000, 1500]
    );
}

#[test]
fn parse_args_accepts_list_devices_without_config() {
    let args = vec!["train".to_string(), "--list-devices".to_string()];
    let parsed = parse_args(args).expect("list devices should parse without config");
    assert!(parsed.list_devices);
    assert!(parsed.config_path.is_none());
}

#[test]
fn parse_args_rejects_list_devices_with_other_args() {
    let args = vec![
        "train".to_string(),
        "--list-devices".to_string(),
        "config.yaml".to_string(),
    ];
    let err = parse_args(args).expect_err("list devices should not combine");
    assert!(err.contains("--list-devices cannot be combined with config path or train mode flags"));
}

#[test]
fn parse_args_accepts_delta_q_promotion_flag() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--delta-q-promotion".to_string(),
    ];
    let parsed = parse_args(args).expect("delta_q_promotion arg should parse");
    assert!(parsed.preflight.is_none());
    assert!(parsed.delta_q_promotion);
    assert!(parsed.delta_q_baseline_checkpoint.is_none());
    assert!(parsed.probe_only.is_none());
    assert!(parsed.probe_child.is_none());
}

#[test]
fn parse_args_accepts_delta_q_baseline_checkpoint() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--delta-q-promotion".to_string(),
        "--delta-q-baseline-checkpoint".to_string(),
        "baseline_model.mpk".to_string(),
    ];
    let parsed = parse_args(args).expect("delta_q baseline arg should parse");
    assert!(parsed.delta_q_promotion);
    assert_eq!(
        parsed.delta_q_baseline_checkpoint,
        Some(PathBuf::from("baseline_model.mpk"))
    );
}

#[test]
fn parse_args_rejects_delta_q_baseline_checkpoint_without_promotion() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--delta-q-baseline-checkpoint".to_string(),
        "baseline_model.mpk".to_string(),
    ];
    let err =
        parse_args(args).expect_err("delta_q baseline checkpoint without promotion should fail");
    assert!(err.contains("--delta-q-baseline-checkpoint requires --delta-q-promotion"));
}

#[test]
fn parse_args_rejects_delta_q_promotion_with_probe_flags() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--delta-q-promotion".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
    ];
    let err = parse_args(args).expect_err("mixed delta_q_promotion/probe flags should fail");
    assert!(err.contains("--delta-q-promotion cannot be combined"));
}

#[test]
fn parse_args_rejects_preflight_with_probe_flags() {
    let args = vec![
        "train".to_string(),
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
        err.contains("probe-only mode requires both --probe-kind and --probe-candidate-microbatch")
    );
}

#[test]
fn parse_args_rejects_probe_batch_child_flags_without_both_batch_fields() {
    let missing_attempts = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
        "--probe-results-path".to_string(),
        "/tmp/probe-results.json".to_string(),
    ];
    let err = parse_args(missing_attempts)
        .expect_err("batch child mode should require attempts with results path");
    assert!(err.contains(
        "internal probe batch child mode requires both --probe-attempts and --probe-results-path"
    ));

    let missing_results_path = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
        "--probe-attempts".to_string(),
        "2".to_string(),
    ];
    let err = parse_args(missing_results_path)
        .expect_err("batch child mode should require results path with attempts");
    assert!(err.contains(
        "internal probe batch child mode requires both --probe-attempts and --probe-results-path"
    ));
}

#[test]
fn parse_args_rejects_mixing_single_and_batch_probe_child_flags() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
        "--probe-candidate-microbatch".to_string(),
        "192".to_string(),
        "--probe-result-path".to_string(),
        "/tmp/probe.json".to_string(),
        "--probe-attempts".to_string(),
        "2".to_string(),
        "--probe-results-path".to_string(),
        "/tmp/probe-results.json".to_string(),
    ];
    let err = parse_args(args)
        .expect_err("single and batch child probe flags should be mutually exclusive");
    assert!(err.contains(
        "internal probe child mode cannot combine --probe-result-path with --probe-attempts/--probe-results-path"
    ));
}

#[test]
fn read_config_applies_defaults() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 3
"#;
    let base = write_temp_file("config", "yaml", yaml);
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
    assert_eq!(
        cfg.python_variant,
        hydra_train_runtime::config::PythonLearnerVariant::CompileMaxAutotune
    );
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

fn unique_temp_file(label: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hydra_train_{label}_{}_{}.{}",
        std::process::id(),
        default_seed(),
        extension
    ))
}

fn write_temp_file(label: &str, extension: &str, contents: &str) -> PathBuf {
    let path = unique_temp_file(label, extension);
    fs::write(&path, contents).expect("write temp file");
    path
}

#[cfg(feature = "burn-libtorch-obsolete")]
fn train_device_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(feature = "burn-libtorch-obsolete")]
struct TrainDeviceEnvGuard {
    _lock: MutexGuard<'static, ()>,
}

#[cfg(feature = "burn-libtorch-obsolete")]
impl TrainDeviceEnvGuard {
    fn reset() -> Self {
        let lock = train_device_env_lock()
            .lock()
            .expect("train device env lock should not be poisoned");
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
        Self { _lock: lock }
    }
}

#[cfg(feature = "burn-libtorch-obsolete")]
impl Drop for TrainDeviceEnvGuard {
    fn drop(&mut self) {
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }
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
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
        num_epochs: 1,
        batch_size: 256,
        microbatch_size: Some(64),
        validation_microbatch_size: Some(16),
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
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
        log_every_n_steps: 25,
        validate_every_n_steps: 200,
        checkpoint_every_n_steps: 200,
        keep_step_checkpoints: false,
        launch_tensorboard: false,
        tensorboard_host: "127.0.0.1".to_string(),
        tensorboard_port: 6006,
        background: false,
        max_train_steps: Some(1000),
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: Some(8192),
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
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
        resume_banner_message(&state, None),
        "global_step=2048 semantics=RestoreOptimizerSkipSeenSamples skipping 137 completed optimizer steps worth of samples in epoch 3 before new updates runtime=train_mb:256 val_mb:128 accum_steps:8 requested_precision=fp32 effective_precision=fp32"
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
        resume_banner_message(&state, None),
        "global_step=500 semantics=RestoreOptimizerSkipSeenSamples resuming at epoch 2 with new updates immediately runtime=train_mb:512 val_mb:256 accum_steps:4 requested_precision=fp32 effective_precision=fp32"
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
    let err =
        validate_resume_runtime_compatibility(&state, test_runtime_resume_contract(2048, 512, 128))
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
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
"#;
    let yaml_path = write_temp_file("yaml", "yaml", yaml);
    assert_eq!(read_config(&yaml_path).expect("yaml config").num_epochs, 1);
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_yaml_preflight_block() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  warmup_steps: 1
"#;
    let yaml_path = write_temp_file("yaml_preflight_rejected", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("preflight yaml block should be rejected");
    assert!(err.contains("preflight"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_accepts_yaml_without_preflight_block() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
"#;
    let yaml_path = write_temp_file("yaml_without_preflight", "yaml", yaml);
    let config = read_config(&yaml_path).expect("yaml without preflight should parse");
    assert_eq!(config.num_epochs, 1);
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_json_config() {
    let json = r#"{
        "data_dir": "/tmp/data",
        "output_dir": "/tmp/out",
        "num_epochs": 1
    }"#;
    let json_path = write_temp_file("json", "json", json);
    let err = read_config(&json_path).expect_err("json config should be rejected");
    assert!(err.contains("unsupported config extension"));
    assert!(err.contains("use .yaml"));
    std::fs::remove_file(json_path).ok();
}

#[test]
fn read_config_rejects_unknown_top_level_fields() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
old_field: true
"#;
    let yaml_path = write_temp_file("unknown_field", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("unknown field should fail");
    assert!(err.contains("failed to parse yaml config"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_legacy_preflight_probe_only_block() {
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
    let yaml_path = write_temp_file("probe_only", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("legacy probe-only config should fail");
    assert!(err.contains("probe_only"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_accepts_bc_hyperparameter_block() {
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
    let yaml_path = write_temp_file("bc_block", "yaml", yaml);
    let config = read_config(&yaml_path).expect("bc block should parse");
    assert!((config.bc.learning_rate - 1.0e-4).abs() < 1e-12);
    assert!((config.bc.min_learning_rate - 1.0e-5).abs() < 1e-12);
    assert!((config.bc.weight_decay - 2.0e-5).abs() < 1e-12);
    assert!((config.bc.grad_clip_norm - 0.5).abs() < 1e-6);
    assert_eq!(config.bc.warmup_steps, 321);
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_accepts_nsight_trace_metrics() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
nsight_trace:
  kernel_launch_count: 314697
  tiny_kernel_fraction: 0.981
  cuda_runtime_launch_seconds: 0.797
"#;
    let yaml_path = write_temp_file("nsight_trace", "yaml", yaml);
    let config = read_config(&yaml_path).expect("nsight trace block should parse");
    let trace = config.nsight_trace.expect("trace metrics should exist");
    assert_eq!(trace.kernel_launch_count, Some(314697));
    assert_eq!(trace.tiny_kernel_fraction, Some(0.981));
    assert_eq!(trace.cuda_runtime_launch_seconds, Some(0.797));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_unknown_nsight_trace_fields() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
nsight_trace:
  made_up_metric: 1
"#;
    let yaml_path = write_temp_file("nsight_trace_unknown", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("unknown nsight field should fail");
    assert!(err.contains("failed to parse yaml config"));
    assert!(err.contains("made_up_metric"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_unknown_bc_fields() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
bc:
  learning_rate: 1.0e-4
  old_knob: true
"#;
    let yaml_path = write_temp_file("bc_unknown", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("unknown bc field should fail");
    assert!(err.contains("failed to parse yaml config"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_probe_only_block_entirely() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  probe_only:
kind: train
candidate_microbatch: 256
mystery_field: true
"#;
    let yaml_path = write_temp_file("probe_only_unknown", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("probe_only block should fail");
    assert!(err.contains("failed to parse yaml config"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_removed_preflight_enabled_field() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  enabled: true
"#;
    let yaml_path = write_temp_file("preflight_enabled", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("removed enabled field should fail");
    assert!(err.contains("enabled"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_removed_preflight_reuse_cache_field() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  reuse_cache: true
"#;
    let yaml_path = write_temp_file("preflight_reuse_cache", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("removed reuse_cache field should fail");
    assert!(err.contains("reuse_cache"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_removed_preflight_advisory_only_field() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  advisory_only: true
"#;
    let yaml_path = write_temp_file("preflight_advisory_only", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("removed advisory_only field should fail");
    assert!(err.contains("advisory_only"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn read_config_rejects_removed_preflight_safety_backoff_rungs_field() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 1
preflight:
  safety_backoff_rungs: 1
"#;
    let yaml_path = write_temp_file("preflight_safety_backoff", "yaml", yaml);
    let err = read_config(&yaml_path).expect_err("removed safety_backoff_rungs should fail");
    assert!(err.contains("preflight"));
    std::fs::remove_file(yaml_path).ok();
}

#[test]
fn estimate_epoch_progress_returns_none_without_exact_counts() {
    let manifest = hydra_data_core::DataManifest {
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
    let manifest = hydra_data_core::DataManifest {
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
        profiling: None,
        delta_q_promotion: None,
        delta_q_promotion_result: None,
        delta_q_promotion_snapshot: None,
        delta_q_policy_transfer: None,
        delta_q_policy_transfer_result: None,
        delta_q_policy_transfer_snapshot: None,
        rare_actions: hydra_train_runtime::progress::RareActionMetrics::default(),
        saw_exit_targets: false,
        saw_delta_q_targets: false,
    };
    assert!(is_better_validation(&summary.scalar_summary(), None));

    let best = BestValidation {
        policy_loss: 1.1,
        agreement: 0.60,
    };
    assert!(is_better_validation(&summary.scalar_summary(), Some(best)));

    let tied = ValidationSummary {
        total_loss: 2.1,
        policy_loss: 1.0,
        agreement: 0.40,
        samples: 8192,
        profiling: None,
        delta_q_promotion: None,
        delta_q_promotion_result: None,
        delta_q_promotion_snapshot: None,
        delta_q_policy_transfer: None,
        delta_q_policy_transfer_result: None,
        delta_q_policy_transfer_snapshot: None,
        rare_actions: hydra_train_runtime::progress::RareActionMetrics::default(),
        saw_exit_targets: false,
        saw_delta_q_targets: false,
    };
    assert!(is_better_validation(
        &tied.scalar_summary(),
        Some(BestValidation {
            policy_loss: 1.0,
            agreement: 0.39
        })
    ));
    assert!(!is_better_validation(
        &tied.scalar_summary(),
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
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
        num_epochs: 1,
        batch_size: 256,
        microbatch_size: Some(64),
        validation_microbatch_size: None,
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
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
        full_epoch: false,
        max_validation_batches: Some(32),
        max_validation_samples: None,
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
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

    let mut python_config = cfg.clone();
    python_config.batch_size = 256;
    python_config.max_validation_samples = Some(1500);
    let options =
        python_options_from_config(&python_config).expect("plain BC should route to Python");
    assert_eq!(options.validation_steps, 6);
    assert_eq!(options.validation_max_samples, Some(1500));

    python_config.max_validation_samples = None;
    python_config.max_validation_batches = Some(3);
    let options =
        python_options_from_config(&python_config).expect("plain BC should route to Python");
    assert_eq!(options.validation_steps, 1);
    assert_eq!(options.validation_max_samples, None);
}

#[test]
fn validate_config_rejects_zero_validation_microbatch_and_samples() {
    let cfg = TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
        num_epochs: 1,
        batch_size: 256,
        microbatch_size: Some(64),
        validation_microbatch_size: Some(0),
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
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
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: Some(0),
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    };

    let err = validate_config(&cfg).expect_err("zero validation controls should fail");
    assert!(err.contains("max_validation_samples") || err.contains("validation_microbatch_size"));
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
fn validate_config_accepts_basic_rl_block() {
    let cfg = TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
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
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
        rl: Some(hydra_train_runtime::config::RlTrainConfig::default()),
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
        max_train_steps: Some(4),
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: Some(64),
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    };

    validate_config(&cfg).expect("basic rl block should validate");
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
    let err = build_loss_config(Some(&advanced)).expect_err("negative safety residual should fail");
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
fn build_loss_config_allows_delta_q_activation() {
    let advanced = AdvancedLossConfig {
        delta_q: Some(0.1),
        ..Default::default()
    };
    let cfg = build_loss_config(Some(&advanced)).expect("delta_q should be allowed in train.rs");
    assert_eq!(cfg.w_delta_q, 0.1);
}

#[test]
fn build_loss_config_allows_delta_q_even_at_zero_weight() {
    let advanced = AdvancedLossConfig {
        delta_q: Some(0.0),
        ..Default::default()
    };
    let cfg = build_loss_config(Some(&advanced))
        .expect("delta_q key should be accepted even at zero weight");
    assert_eq!(cfg.w_delta_q, 0.0);
}

#[test]
fn read_config_rejects_unknown_advanced_loss_field() {
    let yaml = r#"data_dir: /tmp/data
output_dir: /tmp/out
num_epochs: 3
advanced_loss:
  not_a_real_field: 0.1
"#;
    let base = write_temp_file("bad_advanced_loss", "yaml", yaml);
    let err = read_config(&base).expect_err("unknown advanced loss field should fail");
    assert!(err.contains("not_a_real_field"));
    fs::remove_file(base).ok();
}

#[test]
fn validate_config_rejects_invalid_bc_hyperparameter_ranges() {
    let cfg = TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
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
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
        rl: None,
        bc: BcHyperparamConfig {
            learning_rate: 1e-4,
            min_learning_rate: 2e-4,
            weight_decay: 1e-5,
            grad_clip_norm: 1.0,
            warmup_steps: 100,
            adamw_fused: Default::default(),
            adamw_foreach: Default::default(),
        },
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
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: None,
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    };
    let err = validate_config(&cfg).expect_err("invalid bc ranges should fail");
    assert!(err.contains("bc.min_learning_rate"));
}

#[test]
fn validate_config_requires_sidecar_when_exit_loss_is_enabled() {
    let cfg = TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
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
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: Some(AdvancedLossConfig {
            exit: Some(0.1),
            ..Default::default()
        }),
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
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
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: None,
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    };
    let err = validate_config(&cfg).expect_err("exit loss without sidecar should fail");
    assert!(err.contains("exit_sidecar_path"));
}

#[test]
fn validate_config_requires_sidecar_when_delta_q_loss_is_enabled() {
    let cfg = TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
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
        source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: Some(AdvancedLossConfig {
            delta_q: Some(0.1),
            ..Default::default()
        }),
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        ema: hydra_train_runtime::config::EmaConfig::default(),
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
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: None,
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    };
    let err = validate_config(&cfg).expect_err("delta_q loss without sidecar should fail");
    assert!(err.contains("delta_q_sidecar_path"));
}

#[cfg(feature = "burn-libtorch-obsolete")]
#[test]
fn train_device_prefers_env_override_then_config() {
    let _guard = TrainDeviceEnvGuard::reset();
    assert_eq!(
        train_device("cpu").expect("cpu device"),
        LibTorchDevice::Cpu
    );
    assert_eq!(
        train_device("cuda:2").expect("cuda device"),
        LibTorchDevice::Cuda(2)
    );

    unsafe {
        env::set_var("HYDRA_TRAIN_DEVICE", "cuda:0");
    }
    assert_eq!(
        train_device("cpu").expect("env cuda device"),
        LibTorchDevice::Cuda(0)
    );

    unsafe {
        env::set_var("HYDRA_TRAIN_DEVICE", "cpu");
    }
    assert_eq!(
        train_device("cuda:3").expect("env cpu device"),
        LibTorchDevice::Cpu
    );
}

#[cfg(feature = "burn-libtorch-obsolete")]
#[test]
fn train_device_rejects_invalid_env_value() {
    let _guard = TrainDeviceEnvGuard::reset();
    unsafe {
        env::set_var("HYDRA_TRAIN_DEVICE", "vulkan");
    }
    let err = train_device("cpu").expect_err("invalid env value should fail");
    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE"));
}

#[cfg(feature = "burn-libtorch-obsolete")]
#[test]
fn validation_agreement_is_sample_weighted_across_chunks() {
    let device: <burn::backend::ndarray::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let logits = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::from_floats(
        [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
        &device,
    );
    let mask = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::ones([5, 2], &device);
    let targets =
        Tensor::<burn::backend::ndarray::NdArray<f32>, 1, Int>::from_ints([0, 1, 1, 1, 1], &device);

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

    let weighted = (chunk1_correct + chunk2_correct) as f64 / (chunk1_total + chunk2_total) as f64;
    let naive_chunk_average = ((chunk1_correct as f64 / chunk1_total as f64)
        + (chunk2_correct as f64 / chunk2_total as f64))
        / 2.0;

    assert!((weighted - 0.4).abs() < 1e-12);
    assert!((naive_chunk_average - 0.625).abs() < 1e-12);
}

#[test]
fn progress_label_hides_redundant_single_epoch_denominator() {
    assert_eq!(progress_label("train", 0, 1), "train");
    assert_eq!(progress_label("train", 1, 3), "train 2/3");
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

#[test]
fn preflight_bench_matrix_expands_in_stable_order() {
    let cli = parse_args(
        [
            "train",
            "--preflight",
            "--pf-candidate-tuples",
            "1024:2:1:1,2048:4:2:2,4096:8:4:2",
            "--pf-output",
            "md",
        ]
        .into_iter()
        .map(str::to_string),
    )
    .expect("preflight tuple args parse");
    let preflight = cli.preflight.expect("preflight enabled");
    let tuples = preflight.preflight_config.bench_candidate_tuples;
    assert_eq!(tuples.len(), 3);
    assert_eq!(
        (
            tuples[0].batch_size,
            tuples[0].ring_batches,
            tuples[0].loader_threads,
            tuples[0].prefetch_batches
        ),
        (1024, 2, 1, 1)
    );
    assert_eq!(
        (
            tuples[1].batch_size,
            tuples[1].ring_batches,
            tuples[1].loader_threads,
            tuples[1].prefetch_batches
        ),
        (2048, 4, 2, 2)
    );
    assert_eq!(
        (
            tuples[2].batch_size,
            tuples[2].ring_batches,
            tuples[2].loader_threads,
            tuples[2].prefetch_batches
        ),
        (4096, 8, 4, 2)
    );
}

#[test]
fn preflight_rejects_cartesian_candidate_microbatch_flag() {
    let err = parse_args(
        ["train", "--preflight", "--pf-candidate-microbatch", "1024"]
            .into_iter()
            .map(str::to_string),
    )
    .expect_err("cartesian preflight flag is rejected");
    assert!(err.contains("--pf-candidate-tuples"));
}
