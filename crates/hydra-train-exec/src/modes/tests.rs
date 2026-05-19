use super::*;
use crate::test_loose_replay_fixtures::tiny_real_mjai_replay;
use crate::test_support::unique_test_path as shared_unique_test_path;
use hydra_train_runtime::config::{
    BcHyperparamConfig, ProbeCliRequest, RlTrainConfig, TrainConfig, ValidationGateConfig,
};
use hydra_train_runtime::preflight::{PreflightConfig, ProbeKind, ProbeResult, ProbeStatus};

fn cli() -> TrainCli {
    TrainCli {
        config_path: Some(PathBuf::from("config.yaml")),
        list_devices: false,
        preflight: None,
        delta_q_promotion: false,
        delta_q_baseline_checkpoint: None,
        probe_only: None,
        probe_child: None,
    }
}

fn preflight_cli_options(config: PreflightConfig) -> PreflightCliOptions {
    PreflightCliOptions {
        preflight_config: config,
        profile: hydra_train_runtime::config::PreflightProfile::Default,
        output_dir: PathBuf::from("preflight_bench"),
        device: "cpu".to_string(),
    }
}

fn config() -> TrainConfig {
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
        source_filters: hydra_data_core::SourceFilterConfig::default(),
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
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
    }
}

fn probe_result(kind: ProbeKind, candidate_microbatch: usize, selected: bool) -> ProbeResult {
    ProbeResult {
        kind,
        candidate_microbatch,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(if selected { 512.0 } else { 384.0 }),
        elapsed_seconds: Some(if selected { 1.5 } else { 2.0 }),
        detail: String::new(),
    }
}

fn probe_request(kind: ProbeKind) -> ProbeRequest {
    ProbeRequest {
        kind,
        candidate_microbatch: 192,
        warmup_steps: 4,
        measure_steps: 8,
    }
}

fn dummy_best_validation(policy_loss: f64, agreement: f64) -> BestValidation {
    BestValidation {
        policy_loss,
        agreement,
    }
}

fn unique_test_path(label: &str) -> PathBuf {
    shared_unique_test_path("hydra-modes-test", label)
}

#[test]
fn format_list_devices_stdout_is_exact() {
    assert_eq!(
        format_list_devices_stdout(),
        "Hydra train device labels:\n  cpu        supported; always available\n  cuda       supported syntax; equivalent to cuda:0; requires CUDA-capable LibTorch at runtime\n  cuda:<N>   supported syntax; N is zero-based CUDA device index; availability checked when training opens device\n\nHYDRA_TRAIN_DEVICE overrides YAML device with one of: cpu, cuda, cuda:<N>\n"
    );
}

fn write_tiny_replay_data_dir(label: &str) -> PathBuf {
    let data_dir = unique_test_path(label);
    std::fs::create_dir_all(&data_dir).expect("create tiny replay data dir");
    std::fs::write(data_dir.join("game.mjai.json"), tiny_real_mjai_replay())
        .expect("write tiny replay");
    data_dir
}

#[test]
fn formats_probe_table_message_with_selected_candidate() {
    let message = format_probe_table_message(
        "Probe final table",
        ProbeKind::Train,
        &[
            probe_result(ProbeKind::Train, 64, true),
            probe_result(ProbeKind::Train, 48, false),
        ],
        64,
    );

    assert!(message.contains("Probe final table"));
    assert!(message.contains("candidate_mb"));
    assert!(message.contains("train        yes       64"));
    assert!(message.contains("train        no        48"));
}

#[test]
fn formats_probe_only_status_message_for_rl_games() {
    let message = format_probe_only_status_message(probe_request(ProbeKind::RlGames));

    assert!(message.contains("Probe-only:"));
    assert!(message.contains("kind=rl_games candidate_mb=192 warmup_steps=4 measure_steps=8"));
}

#[test]
fn formats_probe_best_candidate_detail_for_all_probe_kinds() {
    assert_eq!(
        format_probe_only_status_detail(probe_request(ProbeKind::RlMicrobatch)),
        "kind=rl_microbatch candidate_mb=192 warmup_steps=4 measure_steps=8"
    );
    assert_eq!(
        format_probe_best_candidate_detail(ProbeKind::Train, 48),
        "train=48"
    );
    assert_eq!(
        format_probe_best_candidate_detail(ProbeKind::Validation, 96),
        "validation=96"
    );
    assert_eq!(
        format_probe_best_candidate_detail(ProbeKind::RlGames, 8),
        "rl_games=8"
    );
}

#[test]
fn format_best_validation_summary_formats_metrics_and_none_case() {
    let summary = dummy_best_validation(0.125, 0.875);
    assert_eq!(
        format_best_validation_summary(Some(&summary)),
        "0.1250 (agree 87.50%)"
    );
    assert_eq!(format_best_validation_summary(None), "n/a");
}

#[test]
fn format_best_validation_summary_rounds_and_handles_zero_agreement() {
    let summary = dummy_best_validation(1.0 / 3.0, 0.0);
    assert_eq!(
        format_best_validation_summary(Some(&summary)),
        "0.3333 (agree 0.00%)"
    );
}

#[test]
fn model_kind_distinguishes_actor_and_learner_configs() {
    assert_eq!(
        model_kind(&hydra_model::model::HydraModelConfig::actor()),
        "actor"
    );
    assert_eq!(
        model_kind(&hydra_model::model::HydraModelConfig::learner()),
        "learner"
    );
}

#[test]
fn optimized_path_summary_reports_raw_replay_defaults() {
    let mut config = config();
    config.bc_shards_manifest_path = None;

    assert_eq!(
        optimized_path_summary(&config),
        "input=raw_replay pinned_h2d=off prealloc_gpu_tensors=off cuda_graph_replay=experimental_probe_only copy_compute_overlap=off"
    );
}

#[test]
fn optimized_path_summary_reports_shard_path() {
    let mut config = config();
    config.bc_shards_manifest_path = Some(PathBuf::from("/shards/manifest.json"));

    let summary = optimized_path_summary(&config);
    assert!(summary.contains("input=bc_shards"));
    assert!(summary.contains("cuda_graph_replay=experimental_probe_only"));
}

#[test]
fn handle_probe_mode_validates_config_before_probe_runtime() {
    let mut config = config();
    config.batch_size = 0;

    let err = handle_probe_mode(
        Path::new("config.yaml"),
        &config,
        probe_request(ProbeKind::Train),
    )
    .expect_err("invalid config should fail before probe runtime");

    assert_eq!(err, "batch_size must be greater than 0");
}

#[test]
fn dispatches_preflight_before_other_modes() {
    let output_path = unique_test_path("dispatch-preflight-artifact-file");
    std::fs::write(&output_path, "not a directory").expect("write output blocker");
    let mut cli = cli();
    cli.config_path = None;
    let mut preflight = preflight_cli_options(PreflightConfig::default());
    preflight.output_dir = output_path.clone();
    cli.preflight = Some(preflight);
    cli.delta_q_promotion = true;

    let err = run_train_modes(cli, config()).expect_err("preflight should dispatch first");

    assert!(err.contains("failed to create BC artifact dir"));
    let _ = std::fs::remove_file(output_path);
}

#[test]
fn dispatches_delta_q_before_probe_only() {
    let mut cli = cli();
    cli.delta_q_promotion = true;
    cli.delta_q_baseline_checkpoint = Some(PathBuf::from("baseline.mpk"));
    cli.probe_only = Some(ProbeCliRequest {
        kind: ProbeKind::Train,
        candidate_microbatch: 128,
        warmup_steps: Some(2),
        measure_steps: Some(3),
    });
    let mut config = config();
    config.buffer_samples = 0;

    let err = run_train_modes(cli, config).expect_err("delta-q should validate first");

    assert_eq!(err, "buffer_samples must be greater than 0");
}

#[test]
fn resolves_probe_defaults_before_dispatching_probe_mode() {
    let mut cli = cli();
    cli.probe_only = Some(ProbeCliRequest {
        kind: ProbeKind::Validation,
        candidate_microbatch: 64,
        warmup_steps: None,
        measure_steps: Some(5),
    });
    let mut config = config();
    config.batch_size = 0;

    let err = run_train_modes(cli, config).expect_err("probe mode should validate first");

    assert_eq!(err, "batch_size must be greater than 0");
}

#[test]
fn dispatches_default_training_mode() {
    let mut config = config();
    config.archive_queue_bound = 0;

    let err = run_train_modes(cli(), config).expect_err("training should validate first");

    assert_eq!(err, "archive_queue_bound must be greater than 0");
}

#[test]
fn returns_probe_resolution_errors_before_mode_call() {
    let mut cli = cli();
    cli.probe_only = Some(ProbeCliRequest {
        kind: ProbeKind::Train,
        candidate_microbatch: 0,
        warmup_steps: Some(1),
        measure_steps: Some(1),
    });

    let err = run_train_modes(cli, config()).expect_err("invalid probe request should fail");

    assert_eq!(err, "--probe-candidate-microbatch must be greater than 0");
}

#[test]
fn handle_training_mode_returns_validation_errors_from_bootstrap() {
    let mut config = config();
    config.archive_queue_bound = 0;

    let err = handle_training_mode(Path::new("config.yaml"), config)
        .expect_err("invalid config should fail before training bootstrap work");
    assert_eq!(err, "archive_queue_bound must be greater than 0");
}

#[test]
fn handle_training_mode_rl_branch_rejects_invalid_device_before_runtime_work() {
    let mut config = config();
    config.rl = Some(RlTrainConfig::default());
    config.device = "definitely-not-a-device".to_string();

    let err = handle_training_mode(Path::new("config.yaml"), config)
        .expect_err("invalid RL device should fail before bootstrap runtime work");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
}

#[test]
fn handle_training_mode_bc_branch_bubbles_bootstrap_errors_before_device_runtime_work() {
    let mut config = config();
    config.data_dir = unique_test_path("missing-bc-train-data");
    config.output_dir = unique_test_path("bc-train-out");

    let err = handle_training_mode(Path::new("config.yaml"), config)
        .expect_err("missing BC data should fail while bootstrap initializes training mode");

    assert!(
        err.contains("failed to scan data_dir") || err.contains("No such file"),
        "unexpected error: {err}"
    );
}

#[test]
fn handle_delta_q_promotion_mode_returns_validation_errors_from_bootstrap() {
    let mut config = config();
    config.buffer_samples = 0;

    let err = handle_delta_q_promotion_mode(Path::new("config.yaml"), config, None)
        .expect_err("invalid config should fail before promotion runtime");
    assert_eq!(err, "buffer_samples must be greater than 0");
}

#[test]
fn normal_training_does_not_require_explicit_preflight_tuning_mode() {
    let config_path =
        unique_test_path("training-missing-preflight-tuning-mode-config").with_extension("yaml");
    std::fs::write(
        &config_path,
        "data_dir: /tmp/hydra-test-data\noutput_dir: /tmp/hydra-test-out\nnum_epochs: 1\npreflight:\n  warmup_steps: 1\n",
    )
    .expect("write normal training config without explicit preflight tuning_mode");
    let mut config = config();
    config.archive_queue_bound = 0;

    let err = run_train_modes(cli(), config).expect_err(
        "normal training should reach ordinary validation without preflight presence gate",
    );

    assert_eq!(err, "archive_queue_bound must be greater than 0");
    let _ = std::fs::remove_file(config_path);
}

#[test]
fn handle_probe_mode_validates_rl_probe_requests_before_runtime() {
    let mut config = config();
    config.batch_size = 0;
    config.rl = Some(RlTrainConfig::default());

    let microbatch_err = handle_probe_mode(
        Path::new("config.yaml"),
        &config,
        probe_request(ProbeKind::RlMicrobatch),
    )
    .expect_err("invalid config should fail before RL microbatch probe runtime");
    assert_eq!(microbatch_err, "batch_size must be greater than 0");

    let games_err = handle_probe_mode(
        Path::new("config.yaml"),
        &config,
        probe_request(ProbeKind::RlGames),
    )
    .expect_err("invalid config should fail before RL games probe runtime");
    assert_eq!(games_err, "batch_size must be greater than 0");
}

#[test]
fn handle_probe_mode_bc_branch_allows_bf16_past_top_level_gate() {
    let data_dir = write_tiny_replay_data_dir("bf16-bc-probe-no-stable-data");
    let output_dir = unique_test_path("bf16-bc-probe-no-stable-out");
    let mut config = config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir.clone();
    config.device = "definitely-not-a-device".to_string();
    config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
    let config_path = unique_test_path("bf16-bc-probe-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid BF16 BC probe config");
    std::fs::write(&config_path, config_yaml).expect("write valid BF16 BC probe config");

    let err = handle_probe_mode(&config_path, &config, probe_request(ProbeKind::Train))
        .expect_err("BF16 BC probe should fall through the mode gate");

    assert_eq!(
        err,
        "precision_mode=bf16_autocast requires a CUDA device for BC training"
    );
    let _ = std::fs::remove_dir_all(data_dir);
    let _ = std::fs::remove_dir_all(output_dir);
    let _ = std::fs::remove_file(config_path);
}

#[test]
fn handle_probe_mode_rl_branch_allows_bf16_past_top_level_gate() {
    let mut config = config();
    config.rl = Some(RlTrainConfig::default());
    config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
    config.device = "definitely-not-a-device".to_string();
    config.data_dir = unique_test_path("missing-rl-bf16-probe-data");
    config.output_dir = unique_test_path("missing-rl-bf16-probe-out");

    let err = handle_probe_mode(
        Path::new("config.yaml"),
        &config,
        probe_request(ProbeKind::RlMicrobatch),
    )
    .expect_err("RL probe mode should fall through the top-level gate");

    assert_eq!(
        err,
        "precision_mode=bf16_autocast is not supported for RL training yet"
    );
}

#[test]
fn handle_delta_q_promotion_mode_requires_baseline_checkpoint_after_bootstrap() {
    let data_dir = write_tiny_replay_data_dir("promotion-data");
    let output_dir = unique_test_path("promotion-out");
    std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
    let mut config = config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir.clone();

    let err = handle_delta_q_promotion_mode(Path::new("config.yaml"), config, None)
        .expect_err("promotion mode should require a baseline checkpoint after bootstrap");

    assert_eq!(
        err,
        "delta_q promotion mode requires --delta-q-baseline-checkpoint for arena confirmation"
    );
    let _ = std::fs::remove_dir_all(data_dir);
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn handle_delta_q_promotion_mode_bubbles_baseline_checkpoint_load_errors() {
    let data_dir = write_tiny_replay_data_dir("promotion-load-error-data");
    let output_dir = unique_test_path("promotion-load-error-out");
    let baseline_checkpoint = unique_test_path("missing-baseline-checkpoint");
    std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
    let mut config = config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir.clone();

    let err = handle_delta_q_promotion_mode(
        Path::new("config.yaml"),
        config,
        Some(baseline_checkpoint.clone()),
    )
    .expect_err("missing baseline checkpoint should fail during load");

    assert!(err.contains("failed to load delta_q baseline checkpoint"));
    assert!(err.contains(baseline_checkpoint.to_string_lossy().as_ref()));
    let _ = std::fs::remove_dir_all(data_dir);
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn format_probe_table_message_preserves_selected_candidate_even_without_rows() {
    let message = format_probe_table_message("Empty probe table", ProbeKind::RlGames, &[], 256);
    assert!(message.contains("Empty probe table"));
    assert!(message.contains("candidate_mb"));
    assert!(message.contains("selected"));
}

#[test]
fn format_probe_table_message_supports_rl_microbatch_rows() {
    let message = format_probe_table_message(
        "RL microbatch probe table",
        ProbeKind::RlMicrobatch,
        &[probe_result(ProbeKind::RlMicrobatch, 16, true)],
        16,
    );

    assert!(message.contains("RL microbatch probe table"));
    assert!(message.contains("rl_microbatch"));
    assert!(message.contains("candidate_mb"));
    assert!(message.contains("yes       16"));
}

#[test]
fn handle_probe_mode_bubbles_ladder_scan_and_artifact_errors() {
    let mut bc_config = config();
    bc_config.data_dir = unique_test_path("missing-probe-data");
    bc_config.output_dir = unique_test_path("probe-out");
    let bc_err = handle_probe_mode(
        Path::new("config.yaml"),
        &bc_config,
        probe_request(ProbeKind::Validation),
    )
    .expect_err("missing dataset should fail during probe ladder setup");
    assert!(bc_err.starts_with("failed to scan preflight data from "));
    assert!(bc_err.contains(bc_config.data_dir.to_string_lossy().as_ref()));

    let mut rl_config = config();
    rl_config.data_dir = unique_test_path("missing-rl-probe-data");
    rl_config.output_dir = unique_test_path("rl-probe-out");
    rl_config.rl = Some(RlTrainConfig::default());
    let rl_err = handle_probe_mode(
        Path::new("config.yaml"),
        &rl_config,
        probe_request(ProbeKind::RlMicrobatch),
    )
    .expect_err("missing dataset should fail during RL probe ladder setup");
    assert!(rl_err.starts_with("failed to scan preflight data from "));
    assert!(rl_err.contains(rl_config.data_dir.to_string_lossy().as_ref()));

    let output_path = unique_test_path("probe-artifact-file");
    std::fs::write(&output_path, "not a directory").expect("write probe artifact blocker file");
    let mut artifact_config = config();
    artifact_config.output_dir = output_path.clone();
    let artifact_err = handle_probe_mode(
        Path::new("config.yaml"),
        &artifact_config,
        probe_request(ProbeKind::Train),
    )
    .expect_err("file-backed output path should fail probe artifact dir creation");
    assert!(artifact_err.contains("failed to create BC artifact dir"));
    let _ = std::fs::remove_file(output_path);
}

#[test]
fn handle_probe_mode_bubbles_no_stable_results_from_ladder_failures() {
    let data_dir = write_tiny_replay_data_dir("probe-no-stable-data");
    let output_dir = unique_test_path("probe-no-stable-out");
    let mut config = config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir.clone();
    config.device = "definitely-not-a-device".to_string();
    let config_path = unique_test_path("probe-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid probe config");
    std::fs::write(&config_path, config_yaml).expect("write valid probe config");

    let validation_err = handle_probe_mode(
        &config_path,
        &config,
        ProbeRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("all-failing validation probe ladder should bubble the no-stable-result error");
    assert!(validation_err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));

    let train_err = handle_probe_mode(
        &config_path,
        &config,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 64,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("all-failing train probe ladder should bubble the no-stable-result error");
    assert!(train_err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));

    let _ = std::fs::remove_dir_all(data_dir);
    let _ = std::fs::remove_dir_all(output_dir);
    let _ = std::fs::remove_file(config_path);
}
