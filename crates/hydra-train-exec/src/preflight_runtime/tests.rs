use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::artifacts::write_manifest_cache;
use crate::data_pipeline::DataSource;
use crate::probe_ladder::adaptive_oom_probe_next_index;
use crate::probe_search::{
    ProbeGrowthDecision, ProbeGrowthState, ProbeSearchStopReason, maybe_expand_probe_candidates,
    probe_candidate_ladder, probe_search_plan,
};
use crate::probe_summary::ProbeCandidateSummary;
use crate::test_loose_replay_fixtures::write_real_probe_fixture;
use crate::test_support::{dummy_train_config, unique_test_path as shared_unique_test_path};
use hydra_train_runtime::config::{
    ProbeBatchChildRequest, ProbeChildRequest, ProbeCliRequest, ProbeSingleChildRequest,
    RlTrainConfig, loader_runtime_config,
};
use hydra_train_runtime::preflight::{ManifestCacheEntry, PreflightBenchStatus, ProbeStatus};

fn dummy_config() -> TrainConfig {
    dummy_train_config()
}

fn unique_test_path(label: &str) -> PathBuf {
    shared_unique_test_path("hydra-preflight-runtime", label)
}

fn write_temp_file(label: &str, extension: &str, contents: &str) -> PathBuf {
    let path = unique_test_path(label).with_extension(extension);
    fs::write(&path, contents).expect("temporary test file should be writable");
    path
}
fn write_tiny_replay_data_dir(label: &str) -> PathBuf {
    let data_dir = unique_test_path(label);
    fs::create_dir_all(&data_dir).expect("create tiny replay data dir");
    fs::write(
        data_dir.join("game.mjai.json"),
        crate::test_loose_replay_fixtures::tiny_real_mjai_replay(),
    )
    .expect("write tiny replay");
    data_dir
}

fn missing_test_path(label: &str) -> PathBuf {
    let path = unique_test_path(label);
    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(&path);
    path
}

fn dummy_rl_train_config() -> RlTrainConfig {
    RlTrainConfig {
        games_per_batch: 8,
        microbatch_size: Some(16),
        ..RlTrainConfig::default()
    }
}

fn tiny_test_probe_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1)
}

fn assert_probe_result_matches_with_tolerance(left: &ProbeResult, right: &ProbeResult) {
    assert_eq!(left.kind, right.kind);
    assert_eq!(left.candidate_microbatch, right.candidate_microbatch);
    assert_eq!(left.status, right.status);
    assert_eq!(left.elapsed_seconds, right.elapsed_seconds);
    assert_eq!(left.detail, right.detail);
    match (
        left.measured_samples_per_second,
        right.measured_samples_per_second,
    ) {
        (Some(left), Some(right)) => {
            assert!((left - right).abs() < 1e-12);
        }
        (None, None) => {}
        (left, right) => {
            panic!("mismatched measured throughput presence: left={left:?} right={right:?}")
        }
    }
}

#[test]
fn measure_samples_per_second_handles_zero_samples_and_zero_time() {
    assert_eq!(measure_samples_per_second(0, Duration::from_secs(2)), 0.0);
    assert_eq!(measure_samples_per_second(10, Duration::from_secs(0)), 0.0);
    assert!((measure_samples_per_second(24, Duration::from_secs(3)) - 8.0).abs() < 1e-12);
    assert_eq!(
        measure_samples_per_second(10, Duration::from_secs_f64(f64::EPSILON / 2.0)),
        0.0
    );
}

#[test]
fn emit_probe_start_progress_supports_train_and_validation_requests() {
    let train_request = ProbeRequest {
        kind: ProbeKind::Train,
        candidate_microbatch: 64,
        warmup_steps: 2,
        measure_steps: 3,
    };
    let validation_request = ProbeRequest {
        kind: ProbeKind::Validation,
        candidate_microbatch: 32,
        warmup_steps: 1,
        measure_steps: 4,
    };

    assert!(emit_probe_start_progress(train_request, 64).is_ok());
    assert!(emit_probe_start_progress(validation_request, 32).is_ok());
}

#[test]
fn advance_probe_loop_transitions_from_warmup_to_measurement() {
    let request = ProbeRequest {
        kind: ProbeKind::Train,
        candidate_microbatch: 64,
        warmup_steps: 2,
        measure_steps: 3,
    };
    let mut state = ProbeLoopState::new();

    let first =
        advance_probe_loop(&mut state, request, 64, 256).expect("first warmup step should succeed");
    assert!(first.is_none());
    assert_eq!(state.completed_steps, 1);
    assert!(state.measure_start.is_none());

    let second = advance_probe_loop(&mut state, request, 64, 256)
        .expect("second warmup step should start measurement");
    assert!(second.is_none());
    assert_eq!(state.completed_steps, 2);
    assert!(state.measure_start.is_some());
}

#[test]
fn advance_probe_loop_returns_throughput_once_target_steps_complete() {
    let request = ProbeRequest {
        kind: ProbeKind::Validation,
        candidate_microbatch: 32,
        warmup_steps: 1,
        measure_steps: 1,
    };
    let mut state = ProbeLoopState {
        completed_steps: 1,
        measure_start: Some(Instant::now()),
    };

    let throughput = advance_probe_loop(&mut state, request, 32, 32)
        .expect("final measurement step should succeed")
        .expect("target steps should produce throughput");
    assert!(throughput >= 0.0);
    assert_eq!(state.completed_steps, 2);
    assert!(state.measure_start.is_some());
}

fn probe_summary(
    candidate_microbatch: usize,
    average_samples_per_second: f64,
) -> ProbeCandidateSummary {
    ProbeCandidateSummary {
        candidate_microbatch,
        status: ProbeStatus::Success,
        attempts: 1,
        average_samples_per_second: Some(average_samples_per_second),
        average_elapsed_seconds: Some(1.0),
    }
}

#[test]
fn probe_search_plan_counts_initial_and_growth_attempts() {
    let preflight = PreflightConfig {
        required_successes: 3,
        validation_growth_max_steps: 2,
        ..PreflightConfig::default()
    };

    let plan = probe_search_plan(4, &preflight);

    assert_eq!(plan.initial_candidates, 4);
    assert_eq!(plan.required_successes, 3);
    assert_eq!(plan.planned_attempts, 12);
    assert_eq!(plan.max_growth_candidates, 2);
    assert_eq!(plan.max_planned_attempts, 18);
}

#[test]
fn validation_growth_budget_returns_stop_reason_without_adding_candidate() {
    let mut preflight = PreflightConfig::default();
    let config = dummy_config();
    preflight.validation_growth_max_steps = 1;
    let mut candidates = vec![64];
    let summary = probe_summary(64, 100.0);
    let mut growth_state = ProbeGrowthState {
        patience: 0,
        steps: 1,
        prior_best_score: Some(100.0),
    };

    let reason = maybe_expand_probe_candidates(
        &mut candidates,
        ProbeGrowthDecision {
            index: 0,
            kind: ProbeKind::Validation,
            candidate: 64,
            summary: &summary,
            candidate_score: 100.0,
            tolerance: 0.0,
        },
        &config,
        &preflight,
        &mut growth_state,
    );

    assert_eq!(reason, Some(ProbeSearchStopReason::ValidationGrowthBudget));
    assert_eq!(candidates, vec![64]);
}

fn probe_result(kind: ProbeKind, candidate_microbatch: usize, status: ProbeStatus) -> ProbeResult {
    let measured_samples_per_second =
        (status == ProbeStatus::Success).then_some(candidate_microbatch as f64);
    ProbeResult {
        kind,
        candidate_microbatch,
        status,
        measured_samples_per_second,
        elapsed_seconds: Some(0.01),
        detail: String::new(),
    }
}

#[test]
fn oom_adaptive_search_jumps_geometrically_after_first_high_oom() {
    let candidates = vec![2048, 512, 384, 320, 288, 256, 224, 192, 160, 128, 64];
    let results = vec![probe_result(ProbeKind::Train, 2048, ProbeStatus::Oom)];

    let next = adaptive_oom_probe_next_index(&candidates, &results, 1);

    assert_eq!(candidates[next], 512);
}

#[test]
fn oom_adaptive_search_binary_refines_after_safe_lower_bound() {
    let candidates = vec![2048, 512, 384, 320, 288, 256, 224, 192, 160, 128, 64];
    let mut results = vec![probe_result(ProbeKind::Train, 2048, ProbeStatus::Oom)];
    let mut probed = Vec::new();
    let mut index = adaptive_oom_probe_next_index(&candidates, &results, 1);
    while index < candidates.len() {
        let candidate = candidates[index];
        probed.push(candidate);
        let status = if candidate > 160 {
            ProbeStatus::Oom
        } else {
            ProbeStatus::Success
        };
        results.push(probe_result(ProbeKind::Train, candidate, status));
        index = adaptive_oom_probe_next_index(&candidates, &results, index + 1);
        if results.iter().any(|result| {
            result.status == ProbeStatus::Success && result.candidate_microbatch == 160
        }) {
            break;
        }
    }

    assert_eq!(probed, vec![512, 256, 128, 192, 160]);
    assert!(
        [384, 320, 288, 224]
            .into_iter()
            .all(|candidate| !probed.contains(&candidate)),
        "high OOM-only candidates must be skipped once bounded by geometric/binary search"
    );
}

#[test]
fn hybrid_child_reuse_recovers_window_results_and_resets_after_oom() {
    let mut preflight = PreflightConfig::default();
    let config_path = unique_test_path("hybrid-child-reuse-config").with_extension("yaml");
    let config = dummy_config();
    preflight.required_successes = 2;
    preflight.fast_repeated_run_candidate_window = 4;
    preflight.allow_override_explicit_microbatch = true;
    preflight.local_refinement_enabled = false;
    let config_yaml = serde_yaml::to_string(&config).expect("serialize train config");
    fs::write(&config_path, config_yaml).expect("write train config");
    let output_dir = unique_test_path("hybrid-child-reuse-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);

    let (_selected, results) = probe_candidate_ladder(
        &config_path,
        &config,
        &preflight,
        &artifacts,
        ProbeKind::Train,
        &[64, 128, 512, 1024],
    )
    .expect("parent should recover reused-window results and continue after OOM");

    assert!(
        results.iter().any(
            |result| result.candidate_microbatch == 64 && result.status == ProbeStatus::Success
        )
    );
    assert!(
        results
            .iter()
            .any(|result| result.candidate_microbatch == 128
                && result.status == ProbeStatus::Success)
    );
    assert!(
        results
            .iter()
            .any(|result| result.candidate_microbatch == 512 && result.status == ProbeStatus::Oom)
    );
    assert!(
        !results
            .iter()
            .any(|result| result.candidate_microbatch == 1024),
        "OOM at 512 must prune larger candidates before probing"
    );
    let _ = fs::remove_file(config_path);
}

#[test]
fn run_probe_only_train_writes_success_result_for_real_loose_replay_variants() {
    let (root, replay_path, _result_path) = write_real_probe_fixture("train-success");
    let manifest =
        crate::test_loose_replay_fixtures::loose_file_manifest(replay_path.clone(), 1, 0);

    assert_probe_only_train_success_real_loose_replay_case(
        &root,
        &replay_path,
        &manifest,
        "fp32",
        hydra_train_runtime::config::PrecisionMode::Fp32,
    );

    assert_probe_only_train_success_real_loose_replay_case(
        &root,
        &replay_path,
        &manifest,
        "bf16",
        hydra_train_runtime::config::PrecisionMode::Bf16Autocast,
    );

    let _ = fs::remove_dir_all(root);
}

fn assert_probe_only_train_success_real_loose_replay_case(
    root: &Path,
    replay_path: &Path,
    manifest: &DataManifest,
    label: &str,
    precision_mode: hydra_train_runtime::config::PrecisionMode,
) {
    let preflight = PreflightConfig::default();
    let result_path = root.join(format!("probe-result-{label}.json"));
    let mut config = dummy_config();
    config.data_dir = replay_path.to_path_buf();
    config.output_dir = root.join(format!("out-{label}"));
    config.batch_size = 1;
    config.train_fraction = 1.0;
    config.device = "cpu".to_string();
    config.precision_mode = precision_mode;

    run_probe_only_with_model_config(
        &config,
        &preflight,
        &tiny_test_probe_model_config(),
        Some(manifest),
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 1,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect("train probe should succeed on a real loose replay");

    assert!(result_path.exists());
    let raw = fs::read_to_string(&result_path).expect("read written train probe result json");
    let result: ProbeResult =
        serde_json::from_str(&raw).expect("deserialize written train probe result json");
    assert_eq!(result.kind, ProbeKind::Train);
    assert_eq!(result.status, ProbeStatus::Success);
    assert_eq!(result.candidate_microbatch, 1);
    assert!(result.measured_samples_per_second.is_some());
    assert!(result.elapsed_seconds.is_some());
    assert_eq!(result.detail, "stable train probe on real dataset");
}

#[test]
fn run_probe_only_validation_writes_success_result_for_real_loose_replay() {
    let preflight = PreflightConfig::default();
    let (root, replay_path, result_path) = write_real_probe_fixture("validation-success");
    let manifest =
        crate::test_loose_replay_fixtures::loose_file_manifest(replay_path.clone(), 0, 1);
    let mut config = dummy_config();
    config.data_dir = replay_path;
    config.output_dir = root.join("out");
    config.batch_size = 1;
    config.train_fraction = 0.0;
    config.device = "cpu".to_string();

    run_probe_only_with_model_config(
        &config,
        &preflight,
        &tiny_test_probe_model_config(),
        Some(&manifest),
        ProbeRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 1,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect("validation probe should succeed on a real loose replay");

    assert!(result_path.exists());
    let raw = fs::read_to_string(&result_path).expect("read written validation probe result json");
    let result: ProbeResult =
        serde_json::from_str(&raw).expect("deserialize written validation probe result json");
    assert_eq!(result.kind, ProbeKind::Validation);
    assert_eq!(result.status, ProbeStatus::Success);
    assert_eq!(result.candidate_microbatch, 1);
    assert!(result.measured_samples_per_second.is_some());
    assert!(result.elapsed_seconds.is_some());
    assert_eq!(result.detail, "stable validation probe on real dataset");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn classify_probe_detail_maps_oom_backend_and_data_cases() {
    assert_eq!(
        classify_probe_detail("CUDA out of memory"),
        ProbeStatus::Oom
    );
    assert_eq!(
        classify_probe_detail("libtorch backend failed"),
        ProbeStatus::BackendError
    );
    assert_eq!(
        classify_probe_detail("replay data collate mismatch"),
        ProbeStatus::DataError
    );
    assert_eq!(
        classify_probe_detail("unexpected worker panic"),
        ProbeStatus::BackendError
    );
}

#[test]
fn format_probe_attempt_message_uses_probe_kind_label_and_min_attempt_denominator() {
    assert_eq!(
        format_probe_attempt_message(ProbeKind::Validation, 64, 2, 0),
        "[preflight:validation] candidate_mb=64 attempt 2/1"
    );
    assert_eq!(
        format_probe_attempt_message(ProbeKind::RlMicrobatch, 128, 1, 3),
        "[preflight:rl_microbatch] candidate_mb=128 attempt 1/3"
    );
}

#[test]
fn run_probe_child_mode_without_child_request_is_a_no_op() {
    let config = dummy_config();

    assert_eq!(run_probe_child_mode(&config, None), Ok(false));
}

#[test]
fn loader_runtime_config_uses_deterministic_auto_threads_when_unset() {
    let preflight = PreflightConfig::default();
    let config = dummy_config();
    let loader = crate::runtime_autotune_shim::autotune_loader_runtime(
        &config,
        &preflight,
        &DataManifest {
            sources: Vec::new(),
            total_games: 0,
            train_count: 0,
            val_count: 0,
            counts_exact: false,
        },
        &LibTorchDevice::Cpu,
    );
    assert!(loader.is_err());
    let effective = loader_runtime_config(&config);
    assert!(effective.num_threads.is_some());
}

#[test]
fn format_probe_result_summary_reports_success_and_oom() {
    let success = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Train,
        candidate_microbatch: 192,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(1234.5),
        elapsed_seconds: Some(1.5),
        detail: String::new(),
    });
    assert!(success.contains("candidate_mb=192"));
    assert!(success.contains("1234.50 samples/s"));
    assert!(success.contains("elapsed=1.50s"));

    let oom = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Train,
        candidate_microbatch: 256,
        status: ProbeStatus::Oom,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: String::new(),
    });
    assert!(oom.contains(
        "[train] candidate_mb=256 outcome=oom(generic) next=smaller_microbatch detail=n/a"
    ));

    let backend = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::RlGames,
        candidate_microbatch: 512,
        status: ProbeStatus::BackendError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: "probe blocked by host-RAM guard".to_string(),
    });
    assert!(backend.contains(
            "[rl_games] candidate_mb=512 outcome=backend_error(host_ram_guard) detail=probe blocked by host-RAM guard"
        ));
}

#[test]
fn run_probe_only_rl_games_fails_fast_without_rl_config() {
    let preflight = PreflightConfig::default();
    let config = dummy_config();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let result_path = std::env::temp_dir().join(format!(
        "hydra-preflight-runtime-test-rl-games-missing-config-{unique}.json"
    ));

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::RlGames,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("RL games probe should fail before runtime work when RL config is missing");

    assert_eq!(err, "RL probe requested without rl config block");
    assert!(!result_path.exists());
}

#[test]
fn emit_probe_progress_and_step_progress_cover_warmup_and_measure_paths() {
    assert!(emit_probe_progress("plain text that should only flush").is_ok());
    assert!(emit_probe_progress(
        "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=2 measure_steps=3"
    )
    .is_ok());

    let request = ProbeRequest {
        kind: ProbeKind::Train,
        candidate_microbatch: 64,
        warmup_steps: 2,
        measure_steps: 3,
    };
    assert!(
        emit_probe_step_progress(ProbeKind::Train, 64, 0, request, None, 256).is_ok(),
        "warmup branch should format and flush"
    );
    assert!(
        emit_probe_step_progress(ProbeKind::Train, 64, 2, request, Some(Instant::now()), 256,)
            .is_ok(),
        "measure branch should format and flush"
    );
    assert!(
        emit_probe_step_progress(ProbeKind::Validation, 64, 2, request, None, 64).is_ok(),
        "measure branch should still flush without a start timestamp"
    );
}

#[test]
fn run_probe_child_mode_rejects_unresolved_child_probe_steps() {
    let config = dummy_config();
    let result_path = unique_test_path("probe-child.json");

    let warmup_err = run_probe_child_mode(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                warmup_steps: None,
                measure_steps: Some(2),
            },
            result_path: result_path.clone(),
            manifest_cache_path: None,
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect_err("missing warmup steps should be rejected before running child mode");
    assert_eq!(
        warmup_err,
        "internal probe child missing resolved warmup steps"
    );

    let measure_err = run_probe_child_mode(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: Some(1),
                measure_steps: None,
            },
            result_path,
            manifest_cache_path: None,
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect_err("missing measure steps should be rejected before running child mode");
    assert_eq!(
        measure_err,
        "internal probe child missing resolved measure steps"
    );
}

#[test]
fn run_probe_child_mode_bubbles_probe_runtime_errors_after_cli_resolution() {
    let mut config = dummy_config();
    config.data_dir = missing_test_path("probe-child-missing-data");
    config.output_dir = unique_test_path("probe-child-runtime-error-out");
    let result_path = unique_test_path("probe-child-runtime-error.json");

    let err = run_probe_child_mode(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: Some(1),
                measure_steps: Some(1),
            },
            result_path: result_path.clone(),
            manifest_cache_path: None,
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect_err("resolved child requests should bubble probe runtime errors");

    assert!(err.starts_with("failed to scan preflight data from "));
    assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    assert!(!result_path.exists());
}

#[test]
fn run_probe_child_mode_reuses_manifest_cache_for_child_probe_scan_bypass() {
    let (root, replay_path, result_path) = write_real_probe_fixture("probe-child-manifest-reuse");
    let mut config = dummy_config();
    config.data_dir = missing_test_path("probe-child-missing-data-but-cached-manifest");
    config.batch_size = 1;
    config.train_fraction = 0.0;
    config.device = "cpu".to_string();

    let manifest_cache_path = root.join("preflight_manifest.json");
    write_manifest_cache(
        &manifest_cache_path,
        &ManifestCacheEntry {
            data_dir: replay_path.clone(),
            train_fraction_bits: 0.0f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: DataManifest {
                sources: vec![DataSource::LooseFile(replay_path)],
                total_games: 1,
                train_count: 0,
                val_count: 1,
                counts_exact: true,
            },
        },
    )
    .expect("write manifest cache for child probe");

    run_probe_child_mode_with_model_config(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 1,
                warmup_steps: Some(1),
                measure_steps: Some(1),
            },
            result_path: result_path.clone(),
            manifest_cache_path: Some(manifest_cache_path),
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
        &tiny_test_probe_model_config(),
    )
    .expect("child probe should reuse manifest cache and succeed without rescanning data_dir");

    assert!(result_path.exists());
    let raw = fs::read_to_string(&result_path).expect("read child probe result");
    let result: ProbeResult = serde_json::from_str(&raw).expect("deserialize child probe result");
    assert_eq!(result.kind, ProbeKind::Validation);
    assert_eq!(result.status, ProbeStatus::Success);
    assert!(result.measured_samples_per_second.is_some());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_probe_child_batch_mode_reuses_manifest_cache_across_attempts() {
    let (root, replay_path, _result_path) =
        write_real_probe_fixture("probe-child-batch-manifest-reuse");
    let mut config = dummy_config();
    config.data_dir = missing_test_path("probe-child-batch-missing-data-but-cached-manifest");
    config.batch_size = 1;
    config.train_fraction = 0.0;
    config.device = "cpu".to_string();

    let manifest_cache_path = root.join("preflight_manifest.json");
    let results_path = root.join("probe-batch-results.json");
    write_manifest_cache(
        &manifest_cache_path,
        &ManifestCacheEntry {
            data_dir: replay_path.clone(),
            train_fraction_bits: 0.0f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: DataManifest {
                sources: vec![DataSource::LooseFile(replay_path)],
                total_games: 1,
                train_count: 0,
                val_count: 1,
                counts_exact: true,
            },
        },
    )
    .expect("write manifest cache for child batch probe");

    let artifact = run_probe_child_batch_mode_result(
        &config,
        Some(ProbeChildRequest::Batch(ProbeBatchChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 1,
                warmup_steps: Some(1),
                measure_steps: Some(1),
            },
            attempts: 2,
            results_path: results_path.clone(),
            manifest_cache_path: Some(manifest_cache_path),
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect("child batch probe should reuse manifest cache across attempts")
    .expect("child batch artifact should be present");

    assert!(artifact.is_finished());
    assert_eq!(artifact.results.len(), 2);
    assert!(
        artifact
            .results
            .iter()
            .all(|result| result.kind == ProbeKind::Validation)
    );
    assert!(
        artifact
            .results
            .iter()
            .all(|result| result.status == ProbeStatus::Success)
    );

    let persisted = crate::probe_transport::read_probe_batch_artifact(&results_path)
        .expect("persisted child batch artifact should parse");
    assert_eq!(persisted.is_finished(), artifact.is_finished());
    assert_eq!(persisted.results.len(), artifact.results.len());
    for (persisted_result, artifact_result) in persisted.results.iter().zip(&artifact.results) {
        assert_probe_result_matches_with_tolerance(persisted_result, artifact_result);
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_probe_child_mode_routes_rl_requests_into_rl_probe_wrapper_errors() {
    let config = dummy_config();
    let result_path = unique_test_path("probe-child-rl-runtime-error.json");

    let err = run_probe_child_mode(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::RlGames,
                candidate_microbatch: 8,
                warmup_steps: Some(1),
                measure_steps: Some(1),
            },
            result_path: result_path.clone(),
            manifest_cache_path: None,
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect_err("resolved RL child requests should route into the RL probe wrapper");

    assert_eq!(err, "RL probe requested without rl config block");
    assert!(!result_path.exists());
}

#[test]
fn execute_probe_request_rejects_unsupported_config_extension_before_spawning() {
    let config_path = write_temp_file("unsupported-config", "txt", "not yaml");
    let result_path = unique_test_path("probe-result.json");

    let err = execute_probe_request(
        &config_path,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 64,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("unsupported config extension should fail before spawning child process");

    assert_eq!(
        err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
    assert!(!result_path.exists());
}

#[test]
fn run_rl_probe_only_rejects_non_rl_probe_kinds() {
    let preflight = PreflightConfig::default();
    let config = dummy_config();
    let result_path = unique_test_path("non-rl-probe-result.json");

    let err = run_rl_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 16,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("non-RL kinds should be rejected by the RL-only handler");

    assert_eq!(err, "RL probe requested without rl config block");
    assert!(!result_path.exists());

    let mut config = dummy_config();
    config.rl = Some(dummy_rl_train_config());
    let err = run_rl_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 16,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("non-RL kinds should be rejected even when rl config exists");

    assert_eq!(err, "non-RL probe routed to RL probe handler");
}

#[test]
fn run_rl_probe_only_rl_games_bubbles_invalid_device_before_runtime_work() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.device = "definitely-not-a-device".to_string();
    config.rl = Some(dummy_rl_train_config());
    let result_path = unique_test_path("rl-games-invalid-device.json");

    let err = run_rl_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::RlGames,
            candidate_microbatch: 8,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("invalid RL device should fail before self-play runtime work");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    assert!(!result_path.exists());
}

#[test]
fn run_rl_probe_only_rl_microbatch_bubbles_invalid_device_before_runtime_work() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.device = "definitely-not-a-device".to_string();
    config.rl = Some(dummy_rl_train_config());
    let result_path = unique_test_path("rl-micro-invalid-device.json");

    let err = run_rl_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::RlMicrobatch,
            candidate_microbatch: 16,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("invalid RL device should fail before RL microbatch runtime work");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    assert!(!result_path.exists());
}

#[test]
fn run_probe_only_rl_microbatch_fails_fast_without_rl_config() {
    let preflight = PreflightConfig::default();
    let config = dummy_config();
    let result_path = unique_test_path("rl-microbatch-result.json");

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::RlMicrobatch,
            candidate_microbatch: 24,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("RL microbatch probe should fail before runtime work when RL config is missing");

    assert_eq!(err, "RL probe requested without rl config block");
    assert!(!result_path.exists());
}

#[test]
fn run_probe_only_rejects_invalid_thread_configuration_before_any_probe_work() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.num_threads = Some(0);
    let result_path = unique_test_path("invalid-thread-probe-result.json");

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("invalid rayon thread config should fail before scan or device setup");

    assert!(err.starts_with("failed to configure rayon threads for probe child: "));
    assert!(!result_path.exists());
}

#[test]
fn run_probe_child_mode_bubbles_invalid_thread_configuration_before_probe_execution() {
    let mut config = dummy_config();
    config.num_threads = Some(0);
    let result_path = unique_test_path("invalid-thread-child-result.json");

    let err = run_probe_child_mode(
        &config,
        Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::RlMicrobatch,
                candidate_microbatch: 16,
                warmup_steps: Some(1),
                measure_steps: Some(1),
            },
            result_path: result_path.clone(),
            manifest_cache_path: None,
            discovery_summary_path: None,
            discovery_index_path: None,
        })),
    )
    .expect_err("invalid rayon thread config should bubble before child probe execution");

    assert!(err.starts_with("failed to configure rayon threads for probe child: "));
    assert!(!result_path.exists());
}

#[test]
fn run_probe_only_train_fails_fast_when_dataset_scan_cannot_start() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-train-data");
    config.output_dir = unique_test_path("missing-train-data-out");
    let result_path = unique_test_path("train-probe-result.json");

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("missing dataset path should fail before any heavy train probing");

    assert!(err.starts_with("failed to scan preflight data from "));
    assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    assert!(!result_path.exists());
}

#[test]
fn run_probe_only_train_bubbles_invalid_device_after_successful_scan() {
    let preflight = PreflightConfig::default();
    let root = write_tiny_replay_data_dir("train-invalid-device-scan");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("train-invalid-device-out");
    config.device = "definitely-not-a-device".to_string();
    let result_path = unique_test_path("train-invalid-device-result.json");

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("invalid device should fail after scan but before train probing");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    assert!(!result_path.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_probe_only_validation_bubbles_invalid_device_after_successful_scan() {
    let preflight = PreflightConfig::default();
    let root = write_tiny_replay_data_dir("validation-invalid-device-scan");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("validation-invalid-device-out");
    config.device = "definitely-not-a-device".to_string();
    let result_path = unique_test_path("validation-invalid-device-result.json");

    let err = run_probe_only(
        &config,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("invalid device should fail after scan but before validation probing");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    assert!(!result_path.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_probe_ladder_only_fails_before_probe_attempts_when_data_scan_fails() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-ladder-data");
    config.output_dir = unique_test_path("missing-ladder-data-out");
    let artifacts = BcArtifactPaths::new(&unique_test_path("ladder-artifacts"), 0);

    let err = run_probe_ladder_only(
        Path::new("ignored-config.yaml"),
        &config,
        &artifacts,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("missing dataset path should stop probe ladder before child probes");

    assert!(err.starts_with("failed to scan preflight data from "));
    assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
}

#[test]
fn run_probe_ladder_only_rescans_when_manifest_cache_data_dir_mismatches() {
    let preflight = PreflightConfig::default();
    let (root, replay_path, _) = write_real_probe_fixture("ladder-manifest-mismatch");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("ladder-manifest-mismatch-out");
    config.device = "definitely-not-a-device".to_string();
    let config_path = unique_test_path("ladder-manifest-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize ladder manifest config");
    fs::write(&config_path, config_yaml).expect("write ladder manifest config");
    let artifacts = BcArtifactPaths::new(&unique_test_path("ladder-manifest-artifacts"), 0);
    artifacts
        .create_root_dir()
        .expect("create ladder artifact root");
    let manifest_cache_path = PreflightPaths::new(&artifacts).manifest_cache_path;
    write_manifest_cache(
        &manifest_cache_path,
        &ManifestCacheEntry {
            data_dir: missing_test_path("stale-ladder-data-dir"),
            train_fraction_bits: config.train_fraction.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: DataManifest {
                sources: vec![DataSource::LooseFile(replay_path)],
                total_games: 1,
                train_count: 1,
                val_count: 0,
                counts_exact: true,
            },
        },
    )
    .expect("write stale manifest cache");

    let err = run_probe_ladder_only(
        &config_path,
        &config,
        &artifacts,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("mismatched manifest cache should fall back to rescanning real data");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(artifacts.root);
    let _ = fs::remove_file(config_path);
}

#[test]
fn run_probe_ladder_only_rescans_when_manifest_cache_train_fraction_mismatches() {
    let preflight = PreflightConfig::default();
    let (root, replay_path, _) = write_real_probe_fixture("ladder-manifest-fraction-mismatch");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("ladder-manifest-fraction-mismatch-out");
    config.device = "definitely-not-a-device".to_string();
    let config_path = unique_test_path("ladder-manifest-fraction-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize ladder fraction config");
    fs::write(&config_path, config_yaml).expect("write ladder fraction config");
    let artifacts = BcArtifactPaths::new(&unique_test_path("ladder-fraction-artifacts"), 0);
    artifacts
        .create_root_dir()
        .expect("create ladder fraction artifact root");
    let manifest_cache_path = PreflightPaths::new(&artifacts).manifest_cache_path;
    write_manifest_cache(
        &manifest_cache_path,
        &ManifestCacheEntry {
            data_dir: root.clone(),
            train_fraction_bits: 0.0f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: DataManifest {
                sources: vec![DataSource::LooseFile(replay_path)],
                total_games: 1,
                train_count: 0,
                val_count: 1,
                counts_exact: true,
            },
        },
    )
    .expect("write stale train-fraction manifest cache");

    let err = run_probe_ladder_only(
        &config_path,
        &config,
        &artifacts,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("mismatched train_fraction cache should fall back to rescanning real data");

    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(artifacts.root);
    let _ = fs::remove_file(config_path);
}

#[test]
fn format_probe_result_summary_reports_data_error_and_plain_backend_error() {
    let data = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Validation,
        candidate_microbatch: 48,
        status: ProbeStatus::DataError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: "replay parse mismatch".to_string(),
    });
    assert!(
        data.contains(
            "[validation] candidate_mb=48 outcome=data_error detail=replay parse mismatch"
        )
    );

    let backend = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Train,
        candidate_microbatch: 96,
        status: ProbeStatus::BackendError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: "unexpected worker panic".to_string(),
    });
    assert!(backend.contains("[train] candidate_mb=96 outcome=backend_error("));
    assert!(backend.contains("detail=unexpected worker panic"));
}

#[test]
fn execute_probe_request_rejects_missing_config_before_spawning() {
    let config_path = missing_test_path("missing-probe-config.yaml").with_extension("yaml");
    let result_path = unique_test_path("missing-probe-result.json");

    let err = execute_probe_request(
        &config_path,
        ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        },
        &result_path,
    )
    .expect_err("missing config path should fail before spawning child process");

    assert!(err.contains(config_path.to_string_lossy().as_ref()));
    assert!(!result_path.exists());
}

#[test]
fn format_probe_result_summary_handles_success_without_elapsed_samples() {
    let summary = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Validation,
        candidate_microbatch: 24,
        status: ProbeStatus::Success,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: String::new(),
    });
    assert!(summary.contains("[validation] candidate_mb=24 outcome=success"));
    assert!(summary.contains("0.00 samples/s"));
    assert!(summary.contains("elapsed=0.00s"));
}

#[test]
fn classify_probe_detail_treats_cudnn_and_oom_strings_as_expected() {
    assert_eq!(
        classify_probe_detail("cuDNN kernel launch failed"),
        ProbeStatus::BackendError
    );
    assert_eq!(
        classify_probe_detail("OOM killer terminated child process"),
        ProbeStatus::Oom
    );
}

#[test]
fn run_probe_ladder_only_accepts_rl_request_wrapper_and_fails_on_missing_data_first() {
    let preflight = PreflightConfig::default();
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-rl-ladder-data");
    config.output_dir = unique_test_path("missing-rl-ladder-data-out");
    config.rl = Some(dummy_rl_train_config());
    let artifacts = BcArtifactPaths::new(&unique_test_path("rl-ladder-artifacts"), 0);

    let err = run_probe_ladder_only(
        Path::new("ignored-config.yaml"),
        &config,
        &artifacts,
        &preflight,
        ProbeRequest {
            kind: ProbeKind::RlMicrobatch,
            candidate_microbatch: 16,
            warmup_steps: 1,
            measure_steps: 1,
        },
    )
    .expect_err("missing dataset path should stop RL-flavored probe ladder before child probes");

    assert!(err.starts_with("failed to scan preflight data from "));
    assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
}

#[test]
fn classify_probe_detail_prefers_oom_and_data_keywords_over_backend_defaults() {
    assert_eq!(
        classify_probe_detail("OOM from replay loader while CUDA kernel was active"),
        ProbeStatus::Oom
    );
    assert_eq!(
        classify_probe_detail("collate data replay failure in worker thread"),
        ProbeStatus::DataError
    );
}

#[test]
fn classify_probe_detail_prefers_backend_keywords_over_data_without_oom() {
    assert_eq!(
        classify_probe_detail("cuda replay mismatch in collate worker"),
        ProbeStatus::BackendError
    );
    assert_eq!(
        classify_probe_detail("libtorch data loader replay error"),
        ProbeStatus::BackendError
    );
}

#[test]
fn format_probe_result_summary_reports_plain_success_detail_for_rl_microbatch() {
    let summary = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::RlMicrobatch,
        candidate_microbatch: 12,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(42.25),
        elapsed_seconds: Some(0.5),
        detail: "stable rl_microbatch probe on real dataset".to_string(),
    });

    assert!(summary.contains("[rl_microbatch] candidate_mb=12 outcome=success"));
    assert!(summary.contains("42.25 samples/s"));
    assert!(summary.contains("elapsed=0.50s"));
}

#[test]
fn format_probe_attempt_message_uses_exact_denominator_when_positive() {
    assert_eq!(
        format_probe_attempt_message(ProbeKind::Train, 32, 3, 4),
        "[preflight:train] candidate_mb=32 attempt 3/4"
    );
}

#[test]
fn measure_samples_per_second_handles_fractional_elapsed_time() {
    assert!((measure_samples_per_second(9, Duration::from_millis(450)) - 20.0).abs() < 1e-12);
}

#[test]
fn format_probe_result_summary_keeps_empty_rl_backend_detail_field_stable() {
    let summary = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::RlGames,
        candidate_microbatch: 40,
        status: ProbeStatus::BackendError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: String::new(),
    });

    assert!(summary.contains("[rl_games] candidate_mb=40 outcome=backend_error(generic)"));
    assert!(summary.contains("detail="));
    assert!(!summary.contains("detail=n/a"));
}

#[test]
fn classify_probe_detail_treats_plain_data_keywords_as_data_errors() {
    assert_eq!(
        classify_probe_detail("data pipeline mismatch in worker"),
        ProbeStatus::DataError
    );
    assert_eq!(
        classify_probe_detail("collate failure without backend keywords"),
        ProbeStatus::DataError
    );
}

#[test]
fn replay_load_error_classification_keeps_required_classes_distinct() {
    use crate::data_pipeline::{ReplayLoadErrorClass, classify_replay_load_error};

    let cases = [
        (
            "notes.txt",
            "ignored extension",
            ReplayLoadErrorClass::IgnoredUnsupportedFile,
        ),
        (
            "bad.mjai.json.zst",
            "failed to open zstd MJAI stream: unknown frame descriptor",
            ReplayLoadErrorClass::CorruptCompressedFile,
        ),
        (
            "bad.mjai.json",
            "failed to parse MJAI events: expected value at line 1 column 1",
            ReplayLoadErrorClass::InvalidJson,
        ),
        (
            "bad.mjai.json",
            "unsupported mjai event: reach_accepted",
            ReplayLoadErrorClass::UnsupportedEvent,
        ),
        (
            "bad.mjai.json",
            "replay observation failed: legal action not found",
            ReplayLoadErrorClass::ReplayDesync,
        ),
        (
            "bad.mjai.json",
            "score placement invariant failed",
            ReplayLoadErrorClass::EngineInvariantFailure,
        ),
    ];

    for (path, err, expected) in cases {
        assert_eq!(classify_replay_load_error(path, &err), expected);
    }
}

#[test]
fn format_probe_attempt_message_clamps_zero_total_attempts_for_rl_games() {
    assert_eq!(
        format_probe_attempt_message(ProbeKind::RlGames, 12, 1, 0),
        "[preflight:rl_games] candidate_mb=12 attempt 1/1"
    );
}

#[test]
fn classify_probe_detail_prefers_oom_over_backend_and_data_keywords() {
    assert_eq!(
        classify_probe_detail("cuda oom while replay data collate failed"),
        ProbeStatus::Oom
    );
}

#[test]
fn format_probe_result_summary_keeps_empty_backend_detail_field_stable() {
    let summary = format_probe_result_summary(&ProbeResult {
        kind: ProbeKind::Train,
        candidate_microbatch: 40,
        status: ProbeStatus::BackendError,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail: String::new(),
    });

    assert!(summary.contains("[train] candidate_mb=40 outcome=backend_error(generic)"));
    assert!(summary.ends_with("detail="));
}

#[test]
fn preflight_ignores_existing_legacy_preflight_cache_file() {
    let root = unique_test_path("preflight-bench-ignores-cache");
    let legacy_cache_path = root
        .join("preflight")
        .join("cache")
        .join("preflight_runtime.json");
    std::fs::create_dir_all(legacy_cache_path.parent().expect("cache path has parent"))
        .expect("cache dir created");
    std::fs::write(&legacy_cache_path, "not valid cache").expect("cache fixture written");

    let config = dummy_config();
    let preflight = PreflightConfig::default();
    let result = run_preflight_bench(&config, &preflight, "cpu")
        .expect("benchmark preflight ignores cache contents");

    assert_eq!(
        result.report.rows.len(),
        preflight.bench_candidate_tuples.len()
    );
    assert_eq!(
        std::fs::read_to_string(&legacy_cache_path).expect("cache fixture still exists"),
        "not valid cache"
    );
}

#[test]
fn preflight_does_not_create_legacy_preflight_cache_file() {
    let root = unique_test_path("preflight-bench-no-cache-create");
    let legacy_cache_path = root
        .join("preflight")
        .join("cache")
        .join("preflight_runtime.json");
    let config = dummy_config();
    let preflight = PreflightConfig::default();

    run_preflight_bench(&config, &preflight, "cpu")
        .expect("benchmark preflight runs without cache writes");

    assert!(!legacy_cache_path.exists());
}

#[test]
fn preflight_error_candidate_emits_error_row() {
    let config = dummy_config();
    let mut preflight = PreflightConfig::default();
    preflight.bench_candidate_tuples = vec![hydra_train_runtime::preflight::PreflightBenchTuple {
        batch_size: 0,
        ring_batches: 2,
        loader_threads: 1,
        prefetch_batches: 1,
    }];

    let result = run_preflight_bench(&config, &preflight, "cpu")
        .expect("invalid tuple is reported as an error row");

    assert_eq!(result.report.rows.len(), 1);
    assert_eq!(result.report.rows[0].status, PreflightBenchStatus::Error);
    assert_eq!(
        result.report.rows[0].error.as_deref(),
        Some("batch must be greater than 0")
    );
}

#[test]
fn preflight_bench_pass_row_emits_numeric_metrics() {
    let mut config = dummy_config();
    config.augment = false;
    let mut preflight = PreflightConfig::default();
    preflight.warmup_steps = 1;
    preflight.measure_steps = 2;
    preflight.bench_candidate_tuples = vec![hydra_train_runtime::preflight::PreflightBenchTuple {
        batch_size: 2,
        ring_batches: 2,
        loader_threads: 1,
        prefetch_batches: 1,
    }];

    let result =
        run_preflight_bench(&config, &preflight, "cpu").expect("benchmark preflight should run");

    let row = &result.report.rows[0];
    assert_eq!(row.status, PreflightBenchStatus::Pass);
    assert!(row.samples_per_second.is_some_and(|value| value > 0.0));
    assert!(row.mib_per_second.is_some_and(|value| value > 0.0));
    assert!(row.p50_batch_ms.is_some());
    assert!(row.p95_batch_ms.is_some());
    assert!(row.producer_wait_ratio.is_some());
    assert!(row.consumer_wait_ratio.is_some());
    assert_eq!(row.disk_wait_ratio, Some(0.0));
    assert_eq!(row.gpu_input_wait_ratio, Some(0.0));
    let markdown = crate::presentation::format_preflight_bench_markdown_table(&result.report);
    let row_text = markdown.lines().nth(2).expect("markdown data row");
    let cells = row_text.split('|').map(str::trim).collect::<Vec<_>>();
    for (column, cell) in cells.iter().enumerate().take(19).skip(11) {
        assert!(
            !cell.is_empty(),
            "required metric column {column} should be numeric: {row_text}"
        );
        assert!(
            cell.parse::<f64>().is_ok(),
            "required metric column {column} should parse as f64: {row_text}"
        );
    }
}
