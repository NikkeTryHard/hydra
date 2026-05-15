use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::probe_search::probe_candidate_ladder;
use crate::test_loose_replay_fixtures::{write_real_preflight_fixture, write_real_probe_fixture};
use crate::test_support::{dummy_train_config, unique_test_path as shared_unique_test_path};
use hydra_train_runtime::config::{
    ProbeBatchChildRequest, ProbeChildRequest, ProbeCliRequest, ProbeSingleChildRequest,
    RlTrainConfig, loader_runtime_config,
};
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_STAGE_2_BENCHMARK, PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION,
    PreflightCompletedPhase, PreflightState, ProbeStatus, SelectedRuntimeConfig,
    preflight_cache_key,
};

fn benchmark_runtime_config(
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
    accum_steps: usize,
    loader: LoaderRuntimeConfig,
) -> BenchmarkRuntimeConfig {
    BenchmarkRuntimeConfig {
        train_microbatch_size,
        validation_microbatch_size,
        accum_steps,
        loader,
        learning_rate: None,
        min_learning_rate: None,
        warmup_steps: None,
    }
}

fn selected_runtime_config(
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
    accum_steps: usize,
) -> SelectedRuntimeConfig {
    SelectedRuntimeConfig {
        train_microbatch_size,
        validation_microbatch_size,
        accum_steps,
        unsafe_selected_batch_size: None,
        unsafe_selected_learning_rate: None,
        unsafe_selected_min_learning_rate: None,
        unsafe_selected_warmup_steps: None,
    }
}

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

#[test]
fn fast_repeated_run_ladder_uses_measurable_candidate_when_batch_below_minimum() {
    let mut config = dummy_config();
    config.batch_size = 10;
    config.microbatch_size = None;
    config.preflight.fast_repeated_run_profile = true;
    config.preflight.min_microbatch_size = 16;
    config.preflight.candidate_microbatches = vec![32, 16];

    let seed = config.microbatch_size.unwrap_or(config.batch_size);
    let candidates = fast_repeated_run_ladder(&config.preflight, config.batch_size, seed);

    assert_eq!(seed, 10);
    assert_eq!(candidates, vec![16]);
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

fn empty_manifest() -> DataManifest {
    DataManifest {
        sources: Vec::new(),
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    }
}

fn validation_summary(samples: usize) -> ValidationSummary {
    ValidationSummary {
        total_loss: 0.0,
        policy_loss: 0.0,
        agreement: 0.0,
        samples,
        rare_actions: Default::default(),
        saw_exit_targets: false,
        saw_delta_q_targets: false,
        profiling: None,
        delta_q_promotion: None,
        delta_q_promotion_result: None,
        delta_q_promotion_snapshot: None,
        delta_q_policy_transfer: None,
        delta_q_policy_transfer_result: None,
        delta_q_policy_transfer_snapshot: None,
    }
}

fn benchmark_finalist(runtime: BenchmarkRuntimeConfig) -> BenchmarkFinalist {
    BenchmarkFinalist {
        runtime,
        train_probe_samples_per_second: 0.0,
        validation_probe_samples_per_second: 0.0,
        loader_probe_samples_per_second: 0.0,
        unsafe_batch_size: None,
    }
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

fn sample_stage_two_benchmark_profiling() -> ProfilingEnvelope {
    ProfilingEnvelope::from_children(
        PROFILING_STAGE_STAGE_2_BENCHMARK,
        vec![
            ProfilingEnvelope::leaf(PROFILING_STAGE_TRAIN, 10.0),
            ProfilingEnvelope::leaf(PROFILING_STAGE_VALIDATION, 2.0),
            ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, 0.5),
            ProfilingEnvelope::leaf(PROFILING_STAGE_LOGGING, 0.25),
        ],
    )
}

fn child<'a>(profile: &'a ProfilingEnvelope, stage: &str) -> &'a ProfilingEnvelope {
    profile
        .children
        .iter()
        .find(|child| child.stage == stage)
        .unwrap_or_else(|| panic!("missing profiling stage {stage}"))
}

#[test]
fn train_substage_profile_separates_collation_from_h2d_materialization() {
    let timing = hydra_train_runtime::progress::TrainSubStageTiming {
        collation_seconds: 1.25,
        h2d_transfer_seconds: 0.5,
        h2d_tensor_materialize_seconds: 0.5,
        forward_seconds: 2.0,
        ..Default::default()
    };

    let profile = train_substage_profile(PROFILING_STAGE_TRAIN, 3.75, timing);

    assert_eq!(
        child(&profile, PROFILING_STAGE_COLLATION).elapsed_seconds,
        1.25
    );
    let h2d = child(&profile, PROFILING_STAGE_H2D_TRANSFER);
    assert_eq!(h2d.elapsed_seconds, 0.5);
    assert_eq!(
        child(h2d, PROFILING_STAGE_H2D_TENSOR_MATERIALIZE).elapsed_seconds,
        0.5
    );
    assert_eq!(
        child(&profile, PROFILING_STAGE_FORWARD).elapsed_seconds,
        2.0
    );
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

fn probe_result_with_runtime(
    kind: ProbeKind,
    candidate_microbatch: usize,
    status: ProbeStatus,
    measured_samples_per_second: Option<f64>,
) -> ProbeResult {
    ProbeResult {
        kind,
        candidate_microbatch,
        status,
        measured_samples_per_second,
        elapsed_seconds: Some(1.0),
        detail: String::new(),
    }
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
fn exact_train_probe_runtime_seed_uses_only_exact_standard_attempts() {
    let mut config = dummy_config();
    config.preflight.required_successes = 2;
    config.preflight.warmup_steps = 2;
    config.preflight.measure_steps = 3;
    config.batch_size = 256;
    config.microbatch_size = Some(64);
    config.archive_queue_bound = 8;
    config.buffer_samples = 128;
    config.buffer_games = 16;
    let results = vec![
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(100.0)),
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(110.0)),
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(400.0)),
        probe_result_with_runtime(ProbeKind::Train, 72, ProbeStatus::Success, Some(999.0)),
    ];

    let seed = exact_train_probe_runtime_seed(&config, 64, &results, 2)
        .expect("selected train candidate should seed from exact standard attempts only");

    assert_eq!(seed.train_microbatch_size, 64);
    assert_eq!(seed.tuple, (8, 128, 16));
    assert_eq!(seed.warmup_steps, 2);
    assert_eq!(seed.measure_steps, 3);
    assert_eq!(seed.stats.count, 2);
    assert!((seed.stats.sum - 210.0).abs() < 1e-12);
}

#[test]
fn exact_train_probe_runtime_seed_ignores_non_standard_or_mismatched_attempts() {
    let mut config = dummy_config();
    config.preflight.required_successes = 2;
    config.preflight.warmup_steps = 2;
    config.preflight.measure_steps = 3;
    let selected = 64;

    let wrong_candidate = vec![
        probe_result_with_runtime(ProbeKind::Train, 32, ProbeStatus::Success, Some(100.0)),
        probe_result_with_runtime(ProbeKind::Train, 32, ProbeStatus::Success, Some(110.0)),
    ];
    assert!(exact_train_probe_runtime_seed(&config, selected, &wrong_candidate, 2).is_none());

    let non_standard_only = vec![
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(100.0),
        ),
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(110.0),
        ),
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(120.0),
        ),
    ];
    assert!(exact_train_probe_runtime_seed(&config, selected, &non_standard_only, 1).is_none());

    let missing_throughput = vec![
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(100.0),
        ),
        probe_result_with_runtime(ProbeKind::Train, selected, ProbeStatus::Success, None),
    ];
    assert!(exact_train_probe_runtime_seed(&config, selected, &missing_throughput, 2).is_none());

    let failed_attempt = vec![
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(100.0),
        ),
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::BackendError,
            Some(110.0),
        ),
    ];
    assert!(exact_train_probe_runtime_seed(&config, selected, &failed_attempt, 2).is_none());

    let mixed_kind = vec![
        probe_result_with_runtime(
            ProbeKind::Train,
            selected,
            ProbeStatus::Success,
            Some(100.0),
        ),
        probe_result_with_runtime(
            ProbeKind::Validation,
            selected,
            ProbeStatus::Success,
            Some(110.0),
        ),
    ];
    assert!(exact_train_probe_runtime_seed(&config, selected, &mixed_kind, 2).is_none());
}

#[test]
fn train_probe_runtime_seed_from_successes_uses_refined_winner_attempts() {
    let mut config = dummy_config();
    config.batch_size = 256;
    config.microbatch_size = Some(44);
    config.preflight.warmup_steps = 2;
    config.preflight.measure_steps = 4;
    config.archive_queue_bound = 8;
    config.buffer_samples = 128;
    config.buffer_games = 16;
    let results = vec![
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(100.0)),
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(90.0)),
        probe_result_with_runtime(ProbeKind::Train, 44, ProbeStatus::Success, Some(720.0)),
        probe_result_with_runtime(ProbeKind::Train, 44, ProbeStatus::Success, Some(722.0)),
        probe_result_with_runtime(ProbeKind::Validation, 44, ProbeStatus::Success, Some(999.0)),
    ];

    let seed = train_probe_runtime_seed_from_successes(&config, 44, &results)
        .expect("refined train winner should seed loader tuning");

    assert_eq!(seed.train_microbatch_size, 44);
    assert_eq!(seed.tuple, (8, 128, 16));
    assert_eq!(seed.warmup_steps, 2);
    assert_eq!(seed.measure_steps, 4);
    assert_eq!(seed.stats.count, 2);
    assert!((seed.stats.sum - 1442.0).abs() < 1e-12);
}

#[test]
fn benchmark_score_builds_stage_two_profiling_projection() {
    let config = dummy_config();
    let evaluation = benchmark_score(&config, &sample_stage_two_benchmark_profiling(), 512);

    assert_eq!(
        evaluation.profiling.stage,
        PROFILING_STAGE_STAGE_2_BENCHMARK
    );
    assert_eq!(evaluation.profiling.children.len(), 4);
    assert_eq!(evaluation.score.train_seconds, 10.0);
    assert_eq!(evaluation.score.validation_seconds, 2.0);
    assert_eq!(evaluation.score.checkpoint_seconds, 0.5);
    assert_eq!(evaluation.score.logging_seconds, 0.25);
    assert_eq!(evaluation.score.validation_samples, 512);
    assert!(evaluation.score.wall_clock_samples_per_second.is_finite());
}

#[test]
fn prioritize_full_batch_train_candidate_preserves_seed_first() {
    let mut candidates = vec![32, 64, 16];

    prioritize_full_batch_train_candidate(&mut candidates, 128, 32);

    assert_eq!(candidates, vec![32, 128, 64, 16]);
}

#[test]
fn prioritize_full_batch_train_candidate_noops_when_seed_is_full_batch() {
    let mut candidates = vec![128, 64, 32];

    prioritize_full_batch_train_candidate(&mut candidates, 128, 128);

    assert_eq!(candidates, vec![128, 64, 32]);
}

#[test]
fn probe_search_plan_counts_initial_and_growth_attempts() {
    let mut config = dummy_config();
    config.preflight.required_successes = 3;
    config.preflight.validation_growth_max_steps = 2;

    let plan = probe_search_plan(4, &config);

    assert_eq!(plan.initial_candidates, 4);
    assert_eq!(plan.required_successes, 3);
    assert_eq!(plan.planned_attempts, 12);
    assert_eq!(plan.max_growth_candidates, 2);
    assert_eq!(plan.max_planned_attempts, 18);
}

#[test]
fn validation_growth_budget_returns_stop_reason_without_adding_candidate() {
    let mut config = dummy_config();
    config.preflight.validation_growth_max_steps = 1;
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
        &mut growth_state,
    );

    assert_eq!(reason, Some(ProbeSearchStopReason::ValidationGrowthBudget));
    assert_eq!(candidates, vec![64]);
}

#[test]
fn oom_candidate_prunes_higher_candidates_from_ladder() {
    let mut candidates = vec![256, 128, 64, 32];

    prune_oom_upper_bound(&mut candidates, 128);

    assert_eq!(candidates, vec![128, 64, 32]);
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
    let config_path = unique_test_path("hybrid-child-reuse-config").with_extension("yaml");
    let mut config = dummy_config();
    config.preflight.required_successes = 2;
    config.preflight.fast_repeated_run_candidate_window = 4;
    config.preflight.allow_override_explicit_microbatch = true;
    config.preflight.local_refinement_enabled = false;
    let config_yaml = serde_yaml::to_string(&config).expect("serialize train config");
    fs::write(&config_path, config_yaml).expect("write train config");
    let output_dir = unique_test_path("hybrid-child-reuse-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);

    let (_selected, results) = probe_candidate_ladder(
        &config_path,
        &config,
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
fn diverse_probe_candidates_prefer_larger_microbatch_when_scores_tie() {
    let results = vec![
        probe_result_with_runtime(ProbeKind::Train, 32, ProbeStatus::Success, Some(100.0)),
        probe_result_with_runtime(ProbeKind::Train, 128, ProbeStatus::Success, Some(100.0)),
        probe_result_with_runtime(ProbeKind::Train, 64, ProbeStatus::Success, Some(100.0)),
    ];

    let candidates = diverse_probe_candidates(&results, 32, 3, 0.0);

    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.candidate_microbatch)
            .collect::<Vec<_>>(),
        vec![128, 64, 32]
    );
}

#[test]
fn stage_two_finalists_accept_loader_ranked_first_by_runtime_autotune() {
    let mut config = dummy_config();
    config.preflight.real_benchmark_loader_candidates = 1;
    config.preflight.real_benchmark_max_finalists = 2;
    let selected_loader = LoaderRuntimeConfig {
        num_threads: Some(6),
        buffer_games: 32,
        buffer_samples: 256,
        archive_queue_bound: 16,
    };
    let ranked_loaders = vec![
        RankedLoaderRuntime {
            loader: selected_loader,
            tuple: (16, 256, 32),
            train_samples_per_second: 105.0,
        },
        RankedLoaderRuntime {
            loader: loader_runtime_config(&dummy_config()),
            tuple: (8, 128, 16),
            train_samples_per_second: 100.0,
        },
    ];
    let selected = EffectiveRuntimeConfig {
        selected: selected_runtime_config(64, 32, 4),
        loader: selected_loader,
    };
    let train_candidates = vec![ProbeCandidateSummary {
        candidate_microbatch: 64,
        status: ProbeStatus::Success,
        attempts: 1,
        average_samples_per_second: Some(400.0),
        average_elapsed_seconds: Some(1.0),
    }];
    let validation_candidates = vec![ProbeCandidateSummary {
        candidate_microbatch: 32,
        status: ProbeStatus::Success,
        attempts: 1,
        average_samples_per_second: Some(300.0),
        average_elapsed_seconds: Some(1.0),
    }];

    let loader_candidates = select_loader_finalists(
        &ranked_loaders,
        config.preflight.real_benchmark_loader_candidates,
        config.preflight.finalist_margin_ratio,
        selected.loader,
    );
    let finalists = build_stage_two_finalists(StageTwoFinalistInputs {
        config: &config,
        selected: &selected,
        train_candidates: &train_candidates,
        validation_candidates: &validation_candidates,
        loader_candidates: &loader_candidates,
        train_probe_results: &[],
        validation_probe_results: &[],
        ranked_loaders: &ranked_loaders,
        unsafe_batch_candidates: &[],
    });

    assert_eq!(loader_candidates.len(), 1);
    assert_eq!(loader_candidates[0].loader, selected_loader);
    assert!(
        finalists
            .iter()
            .any(|finalist| finalist.runtime.loader == selected_loader)
    );
    assert_eq!(finalists[0].runtime.loader, selected_loader);
    assert_eq!(finalists[0].loader_probe_samples_per_second, 105.0);
}

#[test]
fn stage_two_finalists_prefer_larger_train_microbatch_when_scores_tie() {
    let mut config = dummy_config();
    config.batch_size = 128;
    config.preflight.real_benchmark_max_finalists = 2;
    let loader = LoaderRuntimeConfig {
        num_threads: Some(6),
        buffer_games: 32,
        buffer_samples: 256,
        archive_queue_bound: 16,
    };
    let selected = EffectiveRuntimeConfig {
        selected: selected_runtime_config(32, 64, 4),
        loader,
    };
    let train_candidates = vec![probe_summary(32, 400.0), probe_summary(128, 400.0)];
    let validation_candidates = vec![probe_summary(64, 300.0)];
    let loader_candidates = vec![RankedLoaderRuntime {
        loader,
        tuple: (16, 256, 32),
        train_samples_per_second: 100.0,
    }];

    let finalists = build_stage_two_finalists(StageTwoFinalistInputs {
        config: &config,
        selected: &selected,
        train_candidates: &train_candidates,
        validation_candidates: &validation_candidates,
        loader_candidates: &loader_candidates,
        train_probe_results: &[],
        validation_probe_results: &[],
        ranked_loaders: &loader_candidates,
        unsafe_batch_candidates: &[],
    });

    assert_eq!(finalists[0].runtime.train_microbatch_size, 128);
    assert_eq!(finalists[0].runtime.accum_steps, 1);
}

#[test]
fn stage_two_validation_cache_plan_groups_only_identical_validation_workloads() {
    let mut config = dummy_config();
    config.batch_size = 32;
    let shared_loader = LoaderRuntimeConfig {
        num_threads: Some(2),
        buffer_games: 8,
        buffer_samples: 64,
        archive_queue_bound: 4,
    };
    let other_loader = LoaderRuntimeConfig {
        num_threads: Some(4),
        ..shared_loader
    };
    let shared_runtime_a = benchmark_runtime_config(16, 8, 2, shared_loader);
    let shared_runtime_b = benchmark_runtime_config(32, 8, 1, shared_loader);
    let other_runtime = benchmark_runtime_config(32, 8, 1, other_loader);
    let shared_key = stage_two_benchmark_validation_cache_key(
        &benchmark_validation_config(&config, shared_runtime_a),
        shared_loader,
    );
    let other_key = stage_two_benchmark_validation_cache_key(
        &benchmark_validation_config(&config, other_runtime),
        other_loader,
    );

    let plan = stage_two_benchmark_validation_cache_plan(
        &config,
        &[
            benchmark_finalist(shared_runtime_a),
            benchmark_finalist(shared_runtime_b),
            benchmark_finalist(other_runtime),
        ],
    );

    assert_eq!(shared_key.validation_sample_limit, Some(64));
    assert_eq!(plan.get(&shared_key), Some(&2));
    assert_eq!(plan.get(&other_key), Some(&1));
    assert_eq!(plan.len(), 2);
}

#[test]
fn stage_two_validation_cache_key_separates_resolved_sample_limits() {
    let mut config = dummy_config();
    config.batch_size = 32;
    config.max_validation_batches = Some(3);
    let loader = LoaderRuntimeConfig {
        num_threads: Some(2),
        buffer_games: 8,
        buffer_samples: 64,
        archive_queue_bound: 4,
    };
    let smaller_runtime = benchmark_runtime_config(16, 8, 2, loader);
    let larger_runtime = benchmark_runtime_config(16, 16, 2, loader);
    let smaller_key = stage_two_benchmark_validation_cache_key(
        &benchmark_validation_config(&config, smaller_runtime),
        loader,
    );
    let larger_key = stage_two_benchmark_validation_cache_key(
        &benchmark_validation_config(&config, larger_runtime),
        loader,
    );

    let plan = stage_two_benchmark_validation_cache_plan(
        &config,
        &[
            benchmark_finalist(smaller_runtime),
            benchmark_finalist(larger_runtime),
        ],
    );

    assert_ne!(smaller_key, larger_key);
    assert_eq!(smaller_key.validation_sample_limit, Some(24));
    assert_eq!(larger_key.validation_sample_limit, Some(48));
    assert_eq!(plan.get(&smaller_key), Some(&1));
    assert_eq!(plan.get(&larger_key), Some(&1));
}

#[test]
fn stage_two_validation_cache_drops_entries_after_planned_reuses() {
    let mut config = dummy_config();
    config.batch_size = 32;
    let loader = LoaderRuntimeConfig {
        num_threads: Some(2),
        buffer_games: 8,
        buffer_samples: 64,
        archive_queue_bound: 4,
    };
    let runtime_a = benchmark_runtime_config(16, 8, 2, loader);
    let runtime_b = benchmark_runtime_config(32, 8, 1, loader);
    let benchmark_config = benchmark_validation_config(&config, runtime_a);
    let key = stage_two_benchmark_validation_cache_key(&benchmark_config, loader);
    let mut cache = StageTwoBenchmarkValidationCache::new(
        &config,
        &[benchmark_finalist(runtime_a), benchmark_finalist(runtime_b)],
    );

    assert_eq!(cache.entries.len(), 1);

    let (first_samples, first_materialization_seconds, first_stats) = cache
        .checkout(key, &benchmark_config, &empty_manifest())
        .expect("first cache checkout should materialize cached validation samples");
    assert!(first_samples.is_some());
    assert!(first_materialization_seconds >= 0.0);
    assert_eq!(first_stats.event_count, 0);
    assert_eq!(
        cache.entries.get(&key).map(|entry| entry.remaining_uses),
        Some(1)
    );

    let (second_samples, second_materialization_seconds, second_stats) = cache
        .checkout(key, &benchmark_config, &empty_manifest())
        .expect("second cache checkout should reuse cached validation samples");
    assert!(second_samples.is_some());
    assert!((second_materialization_seconds - first_materialization_seconds).abs() < 1e-12);
    assert_eq!(second_stats.event_count, first_stats.event_count);
    assert!(cache.entries.is_empty());
}

#[test]
fn benchmark_validation_pass_charges_materialization_seconds_into_validation_time() {
    let mut config = dummy_config();
    config.batch_size = 32;
    let device = LibTorchDevice::Cpu;
    let train_cfg = trainer_config_from_train_config(&config);
    let optimizer: BenchmarkOptimizerOf<TrainBackend> = train_cfg.optimizer_config().init();
    let mut outcome = TrainBenchmarkOutcome {
        model: tiny_test_probe_model_config().init::<TrainBackend>(&device),
        optimizer,
        head_controller: HeadActivationController::new(HeadActivationConfig::default_with_params(
            1,
        )),
        stats: ScalarAverages::default(),
        elapsed_seconds: 0.0,
        measured_samples: 0,
        materialization_stats: Default::default(),
        sub_stage_timing: Default::default(),
        model_init_seconds: 0.0,
        optimizer_init_seconds: 0.0,
        loss_init_seconds: 0.0,
    };

    let (summary, validation_seconds) = benchmark_validation_pass(
        &config,
        &empty_manifest(),
        &device,
        &mut outcome,
        Some(&[]),
        0.75,
    )
    .expect("benchmark validation pass should succeed on empty cached validation samples");

    assert_eq!(summary.samples, 0);
    assert!(validation_seconds >= 0.75);
}

#[test]
fn benchmark_validation_executor_runs_callback_and_charges_materialization() {
    let (summary, validation_seconds) =
        execute_benchmark_validation_pass(0.5, || Ok(validation_summary(7)))
            .expect("validation executor should return callback summary");

    assert_eq!(summary.samples, 7);
    assert!(validation_seconds >= 0.5);
}

#[test]
fn shard_validation_executor_runs_callback_and_uses_bounded_sample_count() {
    let mut config = dummy_config();
    config.max_validation_samples = Some(3);
    let request = ProbeRequest {
        kind: ProbeKind::Validation,
        candidate_microbatch: 4,
        warmup_steps: 1,
        measure_steps: 1,
    };

    let throughput = execute_shard_validation_probe(
        &config,
        request,
        10,
        Instant::now() - Duration::from_secs(1),
        || Ok(validation_summary(10)),
    )
    .expect("shard validation executor should run callback");

    assert!(throughput > 0.0);
    assert!(throughput <= 3.0);
}

#[test]
fn stage_two_benchmark_scopes_record_expected_nested_order() {
    let (result, events) = nvtx::with_test_recorder(|| {
        run_stage_two_benchmark_scopes(
            || Ok(10usize),
            |train_outcome| {
                *train_outcome += 1;
                Ok((20usize, 2.5f64))
            },
            |_, _| Ok(0.5),
            |_, _| Ok(0.25),
        )
        .expect("stage two benchmark scopes should succeed")
    });

    assert_eq!(result, (11, (20, 2.5), 0.5, 0.25));
    assert_eq!(
        events,
        vec![
            "push:stage_2_benchmark".to_string(),
            "push:train".to_string(),
            "pop:train".to_string(),
            "push:validation".to_string(),
            "pop:validation".to_string(),
            "push:checkpoint".to_string(),
            "pop:checkpoint".to_string(),
            "push:logging".to_string(),
            "pop:logging".to_string(),
            "pop:stage_2_benchmark".to_string(),
        ]
    );
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
fn benchmark_train_window_bf16_fails_fast_without_train_data() {
    let config = dummy_config();
    let err = benchmark_train_window_for_backend::<TrainBackend>(
        &config,
        &tiny_test_probe_model_config(),
        &empty_manifest(),
        &LibTorchDevice::Cpu,
    )
    .err()
    .expect("empty manifests should fail before BF16 stage-2 train benchmarking");

    assert_eq!(
        err,
        "not enough train data to finish stage-2 benchmark train window"
    );
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
fn maybe_block_host_ram_growth_probe_returns_none_for_non_growth_cases() {
    let config = dummy_config();

    assert!(maybe_block_host_ram_growth_probe(&config, ProbeKind::Train, 64, None).is_none());
    assert!(
        maybe_block_host_ram_growth_probe(&config, ProbeKind::Validation, 64, Some(64)).is_none()
    );
    assert!(
        maybe_block_host_ram_growth_probe(&config, ProbeKind::Validation, 32, Some(64)).is_none()
    );
    assert!(maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 64, Some(64)).is_none());
}

#[test]
fn run_probe_child_mode_without_child_request_is_a_no_op() {
    let config = dummy_config();

    assert_eq!(run_probe_child_mode(&config, None), Ok(false));
}

#[test]
fn search_rl_runtime_candidate_rejects_validation_kind_as_non_rl() {
    let config = dummy_config();
    let artifacts = RlArtifactPaths::new(&config.output_dir, 0);

    let err = search_rl_runtime_candidate(
        std::path::Path::new("dummy-config.yaml"),
        &config,
        &artifacts,
        ProbeKind::Validation,
        8,
    )
    .expect_err("validation should be rejected by RL runtime search");

    assert_eq!(err, "non-RL probe kind passed to RL runtime search");
}

#[test]
fn preflight_cache_key_changes_only_for_workload_relevant_inputs() {
    let config = dummy_config();
    let model = model_fingerprint_input(&HydraModelConfig::learner());
    let baseline = preflight_cache_key(&config, &model, "cpu", 8);

    let mut threaded = config.clone();
    threaded.num_threads = Some(8);
    assert_eq!(baseline, preflight_cache_key(&threaded, &model, "cpu", 8));

    let mut buffered = config.clone();
    buffered.buffer_samples += 1;
    assert_eq!(baseline, preflight_cache_key(&buffered, &model, "cpu", 8));

    let mut validation_limited = config.clone();
    validation_limited.max_validation_batches = Some(4);
    assert_ne!(
        baseline,
        preflight_cache_key(&validation_limited, &model, "cpu", 8)
    );
}

#[test]
fn loader_runtime_config_uses_deterministic_auto_threads_when_unset() {
    let config = dummy_config();
    let loader = crate::runtime_autotune_shim::autotune_loader_runtime(
        &config,
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
fn maybe_block_host_ram_growth_probe_returns_backend_error_with_host_ram_details() {
    let Some(available) = mem_available_bytes() else {
        return;
    };
    let Some(required_free) = rl_probe_required_free_bytes(&{
        let mut config = dummy_config();
        config.preflight.rl_probe_min_free_memory_bytes = available;
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        config
    }) else {
        return;
    };

    let mut config = dummy_config();
    config.preflight.rl_probe_min_free_memory_bytes = available.max(required_free);
    config.preflight.rl_probe_memory_headroom_ratio = 0.0;
    config.preflight.rl_probe_growth_safety_factor = 1.0;

    let blocked = maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 128, Some(64))
        .expect(
            "growth probe should be blocked when required free memory matches available memory",
        );

    assert_eq!(blocked.kind, ProbeKind::RlGames);
    assert_eq!(blocked.candidate_microbatch, 128);
    assert_eq!(blocked.status, ProbeStatus::BackendError);
    assert!(blocked.measured_samples_per_second.is_none());
    assert!(blocked.elapsed_seconds.is_none());
    assert!(blocked.detail.contains("probe blocked by host-RAM guard"));
    assert!(blocked.detail.contains("available="));
    assert!(blocked.detail.contains("required_free="));
    assert!(blocked.detail.contains("estimated_probe="));
    assert!(blocked.detail.contains("remaining_after_probe="));
    assert!(blocked.detail.contains("baseline_candidate=64"));
    assert!(blocked.detail.contains("growth_safety_factor=1.00"));
}

#[test]
fn search_rl_runtime_candidate_rejects_non_rl_probe_kinds() {
    let config = dummy_config();
    let artifacts = RlArtifactPaths::new(&config.output_dir, 0);

    let err = search_rl_runtime_candidate(
        std::path::Path::new("dummy-config.yaml"),
        &config,
        &artifacts,
        ProbeKind::Train,
        64,
    )
    .expect_err("train probe kind should be rejected for RL runtime search");

    assert_eq!(err, "non-RL probe kind passed to RL runtime search");
}

#[test]
fn run_probe_only_rl_games_fails_fast_without_rl_config() {
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
    let config = dummy_config();
    let result_path = unique_test_path("non-rl-probe-result.json");

    let err = run_rl_probe_only(
        &config,
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
    let mut config = dummy_config();
    config.device = "definitely-not-a-device".to_string();
    config.rl = Some(dummy_rl_train_config());
    let result_path = unique_test_path("rl-games-invalid-device.json");

    let err = run_rl_probe_only(
        &config,
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
    let mut config = dummy_config();
    config.device = "definitely-not-a-device".to_string();
    config.rl = Some(dummy_rl_train_config());
    let result_path = unique_test_path("rl-micro-invalid-device.json");

    let err = run_rl_probe_only(
        &config,
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
    let config = dummy_config();
    let result_path = unique_test_path("rl-microbatch-result.json");

    let err = run_probe_only(
        &config,
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
    let mut config = dummy_config();
    config.num_threads = Some(0);
    let result_path = unique_test_path("invalid-thread-probe-result.json");

    let err = run_probe_only(
        &config,
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
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-train-data");
    config.output_dir = unique_test_path("missing-train-data-out");
    let result_path = unique_test_path("train-probe-result.json");

    let err = run_probe_only(
        &config,
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
    let root = write_tiny_replay_data_dir("train-invalid-device-scan");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("train-invalid-device-out");
    config.device = "definitely-not-a-device".to_string();
    let result_path = unique_test_path("train-invalid-device-result.json");

    let err = run_probe_only(
        &config,
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
    let root = write_tiny_replay_data_dir("validation-invalid-device-scan");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = unique_test_path("validation-invalid-device-out");
    config.device = "definitely-not-a-device".to_string();
    let result_path = unique_test_path("validation-invalid-device-result.json");

    let err = run_probe_only(
        &config,
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
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-ladder-data");
    config.output_dir = unique_test_path("missing-ladder-data-out");
    let artifacts = BcArtifactPaths::new(&unique_test_path("ladder-artifacts"), 0);

    let err = run_probe_ladder_only(
        Path::new("ignored-config.yaml"),
        &config,
        &artifacts,
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
fn search_train_and_validation_microbatch_fail_fast_on_invalid_probe_config_path() {
    let config_path = write_temp_file("invalid-search-config", "txt", "not yaml");
    let artifacts = BcArtifactPaths::new(&unique_test_path("search-bc-artifacts"), 0);
    let config = dummy_config();

    let train_err = search_train_microbatch(&config_path, &config, &artifacts, 64)
        .expect_err("invalid config path should stop train search before launching probes");
    assert_eq!(
        train_err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );

    let validation_err = search_validation_microbatch(&config_path, &config, &artifacts, 32)
        .expect_err("invalid config path should stop validation search before launching probes");
    assert_eq!(
        validation_err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
}

#[test]
fn search_rl_runtime_candidate_fails_fast_on_invalid_probe_config_path() {
    let config_path = write_temp_file("invalid-rl-search-config", "txt", "not yaml");
    let mut config = dummy_config();
    config.rl = Some(dummy_rl_train_config());
    config.preflight.allow_override_explicit_microbatch = false;
    let artifacts = RlArtifactPaths::new(&unique_test_path("search-rl-artifacts"), 0);

    let err = search_rl_runtime_candidate(
        &config_path,
        &config,
        &artifacts,
        ProbeKind::RlMicrobatch,
        16,
    )
    .expect_err("invalid config path should stop RL candidate search before launching probes");

    assert_eq!(
        err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
}

#[test]
fn run_preflight_stops_at_train_probe_when_probe_config_path_is_invalid() {
    let config_path = write_temp_file("invalid-preflight-config", "txt", "not yaml");
    let config = dummy_config();
    let artifacts = BcArtifactPaths::new(&unique_test_path("preflight-artifacts"), 0);

    let err = match run_preflight(
        &config_path,
        &config,
        &HydraModelConfig::learner(),
        "cpu",
        &artifacts,
    ) {
        Err(err) => err,
        Ok(_) => panic!("invalid config path should stop preflight before runtime autotuning"),
    };

    assert_eq!(
        err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
}

#[test]
fn run_preflight_succeeds_on_real_loose_replay_in_bf16_mode() {
    let root = write_real_preflight_fixture("preflight-success-bf16");
    let output_dir = unique_test_path("preflight-success-bf16-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts
        .create_root_dir()
        .expect("create BF16 preflight artifact root");
    let mut config = dummy_config();
    config.data_dir = root.clone();
    config.output_dir = output_dir.clone();
    config.batch_size = 1;
    config.microbatch_size = Some(1);
    config.validation_microbatch_size = Some(1);
    config.train_fraction = 0.5;
    config.augment = false;
    config.buffer_games = 1;
    config.buffer_samples = 1;
    config.archive_queue_bound = 1;
    config.device = "cpu".to_string();
    config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
    config.preflight.allow_override_explicit_microbatch = false;
    config.preflight.required_successes = 1;
    config.preflight.warmup_steps = 1;
    config.preflight.measure_steps = 1;
    config.preflight.real_benchmark_enabled = false;
    config.preflight.loader_runtime_rounds = 0;
    config.preflight.loader_tuple_extra_samples = 0;
    config.preflight.real_benchmark_loader_candidates = 1;
    config.preflight.real_benchmark_train_candidates = 1;
    config.preflight.real_benchmark_validation_candidates = 1;
    config.preflight.finalist_max_candidates = 1;
    config.preflight.candidate_microbatches = vec![1];
    config.preflight.local_refinement_enabled = false;
    config.preflight.search_coordinate_rounds = 0;
    let config_path = unique_test_path("preflight-success-bf16-config").with_extension("yaml");
    let config_yaml =
        serde_yaml::to_string(&config).expect("serialize valid BF16 preflight config");
    fs::write(&config_path, config_yaml).expect("write valid BF16 preflight config yaml");

    let runtime = run_preflight(
        &config_path,
        &config,
        &tiny_test_probe_model_config(),
        "cpu",
        &artifacts,
    )
    .expect("BF16 preflight should succeed on a real loose replay");

    assert_eq!(runtime.runtime.selected.train_microbatch_size, 1);
    assert_eq!(runtime.runtime.selected.validation_microbatch_size, 1);
    assert_eq!(runtime.runtime.selected.accum_steps, 1);
    assert!(runtime.benchmark.is_none());
    assert!(!runtime.train_probe_results.is_empty());
    assert!(!runtime.validation_probe_results.is_empty());
    assert!(
        runtime
            .train_probe_results
            .iter()
            .any(|result| result.status == ProbeStatus::Success)
    );
    assert!(
        runtime
            .validation_probe_results
            .iter()
            .any(|result| result.status == ProbeStatus::Success)
    );

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(output_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn run_rl_preflight_handles_missing_rl_config_and_invalid_probe_config_path() {
    let train_device = LibTorchDevice::Cpu;
    let config_path = write_temp_file("invalid-rl-preflight-config", "txt", "not yaml");

    let missing_rl_err = match run_rl_preflight(&config_path, &dummy_config(), &train_device) {
        Err(err) => err,
        Ok(_) => {
            panic!("RL preflight should reject missing rl config before any filesystem work")
        }
    };
    assert_eq!(
        missing_rl_err,
        "RL preflight requested without rl config block"
    );

    let mut config = dummy_config();
    config.output_dir = unique_test_path("rl-preflight-output");
    config.rl = Some(dummy_rl_train_config());
    let err = match run_rl_preflight(&config_path, &config, &train_device) {
        Err(err) => err,
        Ok(_) => {
            panic!("invalid config path should stop RL preflight before heavy runtime probes")
        }
    };
    assert_eq!(
        err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
}

#[test]
fn search_rl_runtime_candidate_explicit_microbatch_failure_uses_explicit_error_path() {
    let data_dir = unique_test_path("rl-explicit-microbatch-data");
    fs::create_dir_all(&data_dir).expect("create empty RL data dir");
    let output_dir = unique_test_path("rl-explicit-microbatch-out");
    let artifacts = RlArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = false;
    config.rl = Some(RlTrainConfig {
        games_per_batch: 8,
        microbatch_size: Some(24),
        ..RlTrainConfig::default()
    });
    let config_path = unique_test_path("rl-explicit-microbatch-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid RL config");
    fs::write(&config_path, config_yaml).expect("write valid RL config yaml");

    let err = search_rl_runtime_candidate(
        &config_path,
        &config,
        &artifacts,
        ProbeKind::RlMicrobatch,
        16,
    )
    .expect_err("explicit RL microbatch failure should use explicit-only error path");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_train_microbatch_explicit_failure_uses_explicit_error_path() {
    let data_dir = write_tiny_replay_data_dir("train-explicit-microbatch-data");
    let output_dir = unique_test_path("train-explicit-microbatch-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = false;
    config.preflight.required_successes = 1;
    config.microbatch_size = Some(96);
    let config_path = unique_test_path("train-explicit-microbatch-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid train config");
    fs::write(&config_path, config_yaml).expect("write valid train config yaml");

    let err = search_train_microbatch(&config_path, &config, &artifacts, 64)
        .expect_err("explicit train microbatch failure should use explicit-only error path");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_train_microbatch_non_explicit_failure_reports_no_stable_result() {
    let data_dir = write_tiny_replay_data_dir("train-no-stable-data");
    let output_dir = unique_test_path("train-no-stable-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = true;
    config.preflight.required_successes = 1;
    let config_path = unique_test_path("train-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid train config");
    fs::write(&config_path, config_yaml).expect("write valid train config yaml");

    let err = search_train_microbatch(&config_path, &config, &artifacts, 64)
        .expect_err("all-failing train search should report no stable result");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_train_microbatch_propagates_fatal_backend_error_without_extra_refinement() {
    let data_dir = write_tiny_replay_data_dir("train-fatal-backend-data");
    let output_dir = unique_test_path("train-fatal-backend-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = true;
    config.preflight.required_successes = 1;
    config.preflight.candidate_microbatches = vec![64];
    config.preflight.validation_growth_max_steps = 1;
    let config_path = unique_test_path("train-fatal-backend-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid train config");
    fs::write(&config_path, config_yaml).expect("write valid train config yaml");

    let err = search_train_microbatch(&config_path, &config, &artifacts, 64)
        .expect_err("fatal backend errors should be propagated");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_validation_microbatch_explicit_failure_uses_explicit_error_path() {
    let data_dir = write_tiny_replay_data_dir("validation-explicit-microbatch-data");
    let output_dir = unique_test_path("validation-explicit-microbatch-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = false;
    config.preflight.required_successes = 1;
    config.validation_microbatch_size = Some(48);
    let config_path =
        unique_test_path("validation-explicit-microbatch-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid validation config");
    fs::write(&config_path, config_yaml).expect("write valid validation config yaml");

    let err = search_validation_microbatch(&config_path, &config, &artifacts, 32)
        .expect_err("explicit validation microbatch failure should use explicit-only error path");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_validation_microbatch_non_explicit_failure_reports_no_stable_result() {
    let data_dir = write_tiny_replay_data_dir("validation-no-stable-data");
    let output_dir = unique_test_path("validation-no-stable-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = true;
    config.preflight.required_successes = 1;
    let config_path = unique_test_path("validation-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid validation config");
    fs::write(&config_path, config_yaml).expect("write valid validation config yaml");

    let err = search_validation_microbatch(&config_path, &config, &artifacts, 32)
        .expect_err("non-explicit validation failure should report no stable result");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_rl_games_non_explicit_failure_reports_no_stable_result() {
    let data_dir = unique_test_path("rl-games-no-stable-data");
    fs::create_dir_all(&data_dir).expect("create empty RL data dir");
    let output_dir = unique_test_path("rl-games-no-stable-out");
    let artifacts = RlArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = false;
    config.preflight.required_successes = 1;
    config.rl = Some(dummy_rl_train_config());
    let config_path = unique_test_path("rl-games-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid RL games config");
    fs::write(&config_path, config_yaml).expect("write valid RL games config yaml");

    let err =
        search_rl_runtime_candidate(&config_path, &config, &artifacts, ProbeKind::RlGames, 16)
            .expect_err("all-failing RL games search should report no stable result");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
    let _ = fs::remove_file(config_path);
}

#[test]
fn search_rl_microbatch_non_explicit_failure_reports_no_stable_result() {
    let data_dir = unique_test_path("rl-micro-no-stable-data");
    fs::create_dir_all(&data_dir).expect("create empty RL data dir");
    let output_dir = unique_test_path("rl-micro-no-stable-out");
    let artifacts = RlArtifactPaths::new(&output_dir, 0);
    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.output_dir = output_dir;
    config.device = "definitely-not-a-device".to_string();
    config.preflight.allow_override_explicit_microbatch = true;
    config.preflight.required_successes = 1;
    config.rl = Some(dummy_rl_train_config());
    let config_path = unique_test_path("rl-micro-no-stable-config").with_extension("yaml");
    let config_yaml = serde_yaml::to_string(&config).expect("serialize valid RL microbatch config");
    fs::write(&config_path, config_yaml).expect("write valid RL microbatch config yaml");

    let err = search_rl_runtime_candidate(
        &config_path,
        &config,
        &artifacts,
        ProbeKind::RlMicrobatch,
        16,
    )
    .expect_err("all-failing RL microbatch search should report no stable result");

    assert_eq!(
        err,
        "unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device; expected cpu, cuda, or cuda:<index>"
    );
    let _ = fs::remove_dir_all(data_dir);
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
fn maybe_block_host_ram_growth_probe_uses_baseline_guard_for_rl_microbatch_too() {
    let Some(available) = mem_available_bytes() else {
        return;
    };
    let mut config = dummy_config();
    config.preflight.rl_probe_min_free_memory_bytes = available;
    config.preflight.rl_probe_memory_headroom_ratio = 0.0;
    config.preflight.rl_probe_growth_safety_factor = 1.0;

    let blocked = maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 64, Some(32))
        .expect(
            "growth probe should be blocked when required free memory matches available memory",
        );

    assert_eq!(blocked.kind, ProbeKind::RlMicrobatch);
    assert_eq!(blocked.candidate_microbatch, 64);
    assert_eq!(blocked.status, ProbeStatus::BackendError);
    assert!(blocked.detail.contains("baseline_candidate=32"));
}

#[test]
fn maybe_block_host_ram_growth_probe_clamps_subunit_safety_factor_to_one() {
    let Some(available) = mem_available_bytes() else {
        return;
    };
    let mut config = dummy_config();
    config.preflight.rl_probe_min_free_memory_bytes = available;
    config.preflight.rl_probe_memory_headroom_ratio = 0.0;
    config.preflight.rl_probe_growth_safety_factor = 0.25;

    let blocked = maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 128, Some(64))
        .expect("sub-unit safety factors should still clamp to the host-RAM guard path");

    assert_eq!(blocked.kind, ProbeKind::RlGames);
    assert_eq!(blocked.status, ProbeStatus::BackendError);
    assert!(blocked.detail.contains("growth_safety_factor=1.00"));
}

#[test]
fn maybe_block_host_ram_growth_probe_allows_growth_when_required_free_is_zero() {
    let Some(_available) = mem_available_bytes() else {
        return;
    };
    let mut config = dummy_config();
    config.preflight.rl_probe_min_free_memory_bytes = 0;
    config.preflight.rl_probe_memory_headroom_ratio = 0.0;
    config.preflight.rl_probe_growth_safety_factor = 1.0;

    assert!(
        maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 33, Some(32)).is_none()
    );
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
fn run_rl_preflight_fails_fast_on_invalid_microbatch_config_path() {
    let train_device = LibTorchDevice::Cpu;
    let config_path = write_temp_file("invalid-rl-micro-config", "txt", "not yaml");
    let mut config = dummy_config();
    config.output_dir = unique_test_path("rl-preflight-fastfail");
    config.rl = Some(dummy_rl_train_config());

    let err = match run_rl_preflight(&config_path, &config, &train_device) {
        Err(err) => err,
        Ok(_) => {
            panic!("invalid config path should stop RL preflight before runtime probing")
        }
    };

    assert_eq!(
        err,
        format!(
            "unsupported config extension for {}; use .yaml",
            config_path.display()
        )
    );
}

#[test]
fn run_probe_ladder_only_accepts_rl_request_wrapper_and_fails_on_missing_data_first() {
    let mut config = dummy_config();
    config.data_dir = missing_test_path("missing-rl-ladder-data");
    config.output_dir = unique_test_path("missing-rl-ladder-data-out");
    config.rl = Some(dummy_rl_train_config());
    let artifacts = BcArtifactPaths::new(&unique_test_path("rl-ladder-artifacts"), 0);

    let err = run_probe_ladder_only(
        Path::new("ignored-config.yaml"),
        &config,
        &artifacts,
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
fn preflight_resume_state_reuses_matching_completed_candidate() {
    let key = preflight_cache_key(
        &dummy_config(),
        &model_fingerprint_input(&HydraModelConfig::learner()),
        "cpu",
        hydra_train_runtime::config::default_num_threads_for_system(),
    );
    let state = PreflightState {
        cache_key: key,
        completed_phases: vec![PreflightCompletedPhase {
            phase: "train_probe".to_string(),
            elapsed_seconds: 1.25,
        }],
        selected_runtime: None,
        cache_written: false,
        candidate_records: vec![hydra_train_runtime::preflight::PreflightCandidateRecord {
            phase: "train_probe".to_string(),
            result: ProbeResult {
                kind: ProbeKind::Train,
                candidate_microbatch: 32,
                status: ProbeStatus::Success,
                measured_samples_per_second: Some(100.0),
                elapsed_seconds: Some(1.0),
                detail: "ok".to_string(),
            },
        }],
    };

    assert_eq!(completed_phase_seconds(&state, "train_probe"), Some(1.25));
    let results = resumed_candidate_results(&state, "train_probe");
    assert_eq!(successful_candidate_selection(&results), Some(32));
}

#[test]
fn resumed_train_seed_requires_exact_successful_attempts() {
    let mut config = dummy_config();
    config.preflight.required_successes = 2;
    let incomplete = vec![ProbeResult {
        kind: ProbeKind::Train,
        candidate_microbatch: 32,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(100.0),
        elapsed_seconds: Some(1.0),
        detail: "ok".to_string(),
    }];
    assert!(exact_train_probe_runtime_seed(&config, 32, &incomplete, incomplete.len()).is_none());

    let complete = vec![
        ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(100.0),
            elapsed_seconds: Some(1.0),
            detail: "ok".to_string(),
        },
        ProbeResult {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(120.0),
            elapsed_seconds: Some(1.0),
            detail: "ok".to_string(),
        },
    ];
    let seed = exact_train_probe_runtime_seed(&config, 32, &complete, complete.len())
        .expect("exact complete attempts should yield loader seed");
    assert_eq!(seed.stats.count, 2);
    assert_eq!(seed.stats.sum, 220.0);
}

#[test]
fn preflight_resume_state_ignores_completed_phase_without_successful_candidate() {
    let key = preflight_cache_key(
        &dummy_config(),
        &model_fingerprint_input(&HydraModelConfig::learner()),
        "cpu",
        hydra_train_runtime::config::default_num_threads_for_system(),
    );
    let state = PreflightState {
        cache_key: key,
        completed_phases: vec![PreflightCompletedPhase {
            phase: "validation_probe".to_string(),
            elapsed_seconds: 2.0,
        }],
        selected_runtime: None,
        cache_written: false,
        candidate_records: vec![hydra_train_runtime::preflight::PreflightCandidateRecord {
            phase: "validation_probe".to_string(),
            result: ProbeResult {
                kind: ProbeKind::Validation,
                candidate_microbatch: 64,
                status: ProbeStatus::Oom,
                measured_samples_per_second: None,
                elapsed_seconds: Some(0.5),
                detail: "oom".to_string(),
            },
        }],
    };

    let results = resumed_candidate_results(&state, "validation_probe");
    assert_eq!(successful_candidate_selection(&results), None);
}

#[test]
fn format_probe_attempt_message_clamps_zero_total_attempts_for_rl_games() {
    assert_eq!(
        format_probe_attempt_message(ProbeKind::RlGames, 12, 1, 0),
        "[preflight:rl_games] candidate_mb=12 attempt 1/1"
    );
}

#[test]
fn maybe_block_host_ram_growth_probe_returns_none_when_candidate_does_not_grow() {
    let config = dummy_config();

    assert!(
        maybe_block_host_ram_growth_probe(&config, ProbeKind::RlMicrobatch, 31, Some(32)).is_none()
    );
    assert!(maybe_block_host_ram_growth_probe(&config, ProbeKind::RlGames, 32, Some(32)).is_none());
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
fn run_preflight_returns_cached_runtime_on_identical_fingerprint() {
    use crate::artifacts::{PreflightPaths, write_preflight_cache};
    use hydra_train_runtime::preflight::preflight_cache_key;
    use hydra_train_runtime::preflight::{
        EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry,
    };

    let output_dir = unique_test_path("preflight-cache-hit-out");
    let data_dir = write_tiny_replay_data_dir("preflight-cache-hit-data");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts
        .create_root_dir()
        .expect("create artifact root for cache hit test");

    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.preflight.real_benchmark_enabled = false;
    let model_config = HydraModelConfig::learner();
    let model_fingerprint = model_fingerprint_input(&model_config);
    let key = preflight_cache_key(
        &config,
        &model_fingerprint,
        "cpu",
        hydra_train_runtime::config::default_num_threads_for_system(),
    );

    let cached_runtime = EffectiveRuntimeConfig {
        selected: selected_runtime_config(42, 21, 7),
        loader: LoaderRuntimeConfig {
            num_threads: Some(4),
            buffer_games: 256,
            buffer_samples: 1024,
            archive_queue_bound: 16,
        },
    };
    let paths = PreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: key,
            runtime: cached_runtime,
            benchmark: None,
        },
    )
    .expect("write matching cache entry");

    let config_path = write_temp_file(
        "preflight-cache-hit-config",
        "yaml",
        &serde_yaml::to_string(&config).expect("serialize config"),
    );
    let result = run_preflight(&config_path, &config, &model_config, "cpu", &artifacts)
        .expect("cache hit should succeed through common path");

    assert_eq!(result.runtime.selected.train_microbatch_size, 42);
    assert_eq!(result.runtime.selected.validation_microbatch_size, 21);
    assert_eq!(result.runtime.selected.accum_steps, 7);
    assert!(
        result.train_probe_results.is_empty(),
        "cache hit should skip probing"
    );
    assert!(
        result.validation_probe_results.is_empty(),
        "cache hit should skip validation probing"
    );
    assert!(
        result.advisories.is_empty(),
        "cache hit has no probe results and must not invent selected-vs-best advisories"
    );

    let paths = PreflightPaths::new(&artifacts);
    assert!(
        paths.events_log_path.exists(),
        "events log should be created"
    );
    assert!(
        paths.metrics_log_path.exists(),
        "metrics placeholder should be created"
    );
    assert!(paths.state_path.exists(), "state file should be created");
    assert!(paths.report_path.exists(), "report file should be created");

    let events = fs::read_to_string(&paths.events_log_path).expect("read preflight events");
    let parsed_events: Vec<hydra_train_runtime::preflight::PreflightArtifactEvent> = events
        .lines()
        .map(|line| serde_json::from_str(line).expect("event JSONL line should parse"))
        .collect();
    assert!(
        parsed_events
            .iter()
            .any(|event| event.phase == "cache_lookup"
                && event.kind
                    == hydra_train_runtime::preflight::PreflightArtifactEventKind::Completed),
        "cache-hit preflight should record cache_lookup completion"
    );

    let state_raw = fs::read_to_string(&paths.state_path).expect("read preflight state");
    let state: hydra_train_runtime::preflight::PreflightState =
        serde_json::from_str(&state_raw).expect("state should parse");
    assert!(
        state.cache_written,
        "state should be durable after cache write"
    );
    assert!(
        state
            .completed_phases
            .iter()
            .any(|phase| phase.phase == "cache_lookup"),
        "state should persist completed phases"
    );
    assert_eq!(
        state
            .selected_runtime
            .expect("selected runtime should persist")
            .selected
            .train_microbatch_size,
        42
    );

    let report_raw = fs::read_to_string(&paths.report_path).expect("read preflight report");
    let report: hydra_train_runtime::preflight::PreflightReport =
        serde_json::from_str(&report_raw).expect("report should parse");
    assert!(report.cache_hit, "report should preserve cache-hit summary");

    let _ = fs::remove_dir_all(&output_dir);
    let _ = fs::remove_dir_all(&data_dir);
}

#[test]
fn run_preflight_cache_hit_preserves_benchmark_result() {
    use crate::artifacts::{PreflightPaths, write_preflight_cache};
    use hydra_train_runtime::preflight::preflight_cache_key;
    use hydra_train_runtime::preflight::{
        BenchmarkMetadata, BenchmarkMode, BenchmarkResult, BenchmarkScore, EffectiveRuntimeConfig,
        LoaderRuntimeConfig, PreflightCacheEntry, ProfilingEnvelope,
    };

    let output_dir = unique_test_path("preflight-cache-hit-benchmark-out");
    let data_dir = write_tiny_replay_data_dir("preflight-cache-hit-benchmark-data");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts
        .create_root_dir()
        .expect("create artifact root for benchmark cache hit test");

    let mut config = dummy_config();
    config.data_dir = data_dir.clone();
    config.preflight.real_benchmark_enabled = true;
    let model_config = HydraModelConfig::learner();
    let model_fingerprint = model_fingerprint_input(&model_config);
    let key = preflight_cache_key(
        &config,
        &model_fingerprint,
        "cpu",
        hydra_train_runtime::config::default_num_threads_for_system(),
    );

    let benchmark = BenchmarkResult {
        runtime: benchmark_runtime_config(
            8,
            4,
            2,
            LoaderRuntimeConfig {
                num_threads: Some(2),
                buffer_games: 32,
                buffer_samples: 128,
                archive_queue_bound: 4,
            },
        ),
        score: BenchmarkScore {
            wall_clock_samples_per_second: 123.456,
            train_only_samples_per_second: 200.0,
            train_seconds: 1.0,
            validation_seconds: 0.5,
            checkpoint_seconds: 0.1,
            logging_seconds: 0.05,
            total_elapsed_seconds: 1.65,
            train_steps: 10,
            validation_samples: 50,
        },
        metadata: BenchmarkMetadata {
            mode: BenchmarkMode::CadenceAwareProjection,
            ..Default::default()
        },
        profiling: Some(ProfilingEnvelope::leaf("stage_2_benchmark", 1.5)),
    };

    let paths = PreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: key,
            runtime: EffectiveRuntimeConfig {
                selected: selected_runtime_config(8, 4, 2),
                loader: LoaderRuntimeConfig {
                    num_threads: Some(2),
                    buffer_games: 32,
                    buffer_samples: 128,
                    archive_queue_bound: 4,
                },
            },
            benchmark: Some(benchmark.clone()),
        },
    )
    .expect("write cache entry with benchmark");

    let config_path = write_temp_file(
        "preflight-cache-hit-benchmark-config",
        "yaml",
        &serde_yaml::to_string(&config).expect("serialize config"),
    );
    let result = run_preflight(&config_path, &config, &model_config, "cpu", &artifacts)
        .expect("cache hit should succeed through common path");

    let returned_benchmark = result
        .benchmark
        .as_ref()
        .expect("cache hit should return cached benchmark without re-benchmarking");
    assert_eq!(returned_benchmark.runtime, benchmark.runtime);
    assert_eq!(
        returned_benchmark.metadata.finalists_benchmarked, benchmark.metadata.finalists_benchmarked,
        "cache-hit real benchmark must not synthesize finalists from empty probe vectors"
    );
    assert_eq!(
        returned_benchmark.score.wall_clock_samples_per_second,
        benchmark.score.wall_clock_samples_per_second
    );
    assert_eq!(result.runtime.selected.train_microbatch_size, 8);
    assert_eq!(result.runtime.selected.validation_microbatch_size, 4);

    let events = fs::read_to_string(&paths.events_log_path).expect("read preflight events");
    let parsed_events: Vec<hydra_train_runtime::preflight::PreflightArtifactEvent> = events
        .lines()
        .map(|line| serde_json::from_str(line).expect("event JSONL line should parse"))
        .collect();
    assert!(
        parsed_events.iter().any(|event| {
            event.phase == "stage_2_benchmark"
                && event.kind == hydra_train_runtime::preflight::PreflightArtifactEventKind::Skipped
                && event
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.contains("reused cached benchmark result"))
        }),
        "cache-hit real benchmark should report reuse, not an empty-finalist re-benchmark"
    );
    assert!(
        result.advisories.is_empty(),
        "cache hit has no probe results and must not invent selected-vs-best advisories"
    );

    let _ = fs::remove_dir_all(&output_dir);
    let _ = fs::remove_dir_all(&data_dir);
}

#[test]
fn run_preflight_misses_cache_on_different_fingerprint() {
    use crate::artifacts::{PreflightPaths, write_preflight_cache};
    use hydra_train_runtime::preflight::{
        EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, PreflightCacheEntry,
        PreflightCacheKey, WorkloadFingerprint,
    };

    let output_dir = unique_test_path("preflight-cache-miss-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts
        .create_root_dir()
        .expect("create artifact root for cache miss test");

    let stale_key = PreflightCacheKey {
        hardware: HardwareFingerprint {
            device_label: "stale-gpu".to_string(),
            backend: "burn-libtorch".to_string(),
            cpu_logical_cores: 999,
            total_memory_bytes: None,
        },
        workload: WorkloadFingerprint {
            batch_size: 9999,
            augment: false,
            precision_mode: "fp32".to_string(),
            train_fraction_bits: 0,
            max_skip_logs_per_source: 0,
            max_validation_batches: None,
            max_validation_samples: None,
            model_signature: "stale".to_string(),
            code_signature: "stale".to_string(),
            advanced_loss_signature: "stale".to_string(),
            preflight_config_signature: "stale".to_string(),
            explicit_train_microbatch: None,
            explicit_validation_microbatch: None,
        },
    };
    let paths = PreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: stale_key,
            runtime: EffectiveRuntimeConfig {
                selected: selected_runtime_config(99, 99, 1),
                loader: LoaderRuntimeConfig {
                    num_threads: Some(1),
                    buffer_games: 999,
                    buffer_samples: 999,
                    archive_queue_bound: 1,
                },
            },
            benchmark: None,
        },
    )
    .expect("write stale cache entry");

    let config_path = write_temp_file("preflight-cache-miss-config", "txt", "not yaml");
    let config = dummy_config();
    let result = run_preflight(
        &config_path,
        &config,
        &HydraModelConfig::learner(),
        "cpu",
        &artifacts,
    );

    assert!(
        result.is_err(),
        "stale cache should miss and proceed to probing which fails on invalid config"
    );

    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn run_rl_preflight_returns_cached_runtime_on_identical_fingerprint() {
    use crate::artifacts::{RlArtifactPaths, RlPreflightPaths, write_preflight_cache};
    use hydra_train_runtime::preflight::preflight_cache_key;
    use hydra_train_runtime::preflight::{
        EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry,
    };

    let output_dir = unique_test_path("rl-preflight-cache-hit-out");
    let mut config = dummy_config();
    config.rl = Some(dummy_rl_train_config());
    config.output_dir = output_dir.clone();
    config.device = "cpu".to_string();

    let artifacts = RlArtifactPaths::new(&output_dir, 0);
    artifacts
        .create_root_dir()
        .expect("create RL artifact root for cache hit test");

    let model_config = HydraModelConfig::learner();
    let model_fingerprint = model_fingerprint_input(&model_config);
    let key = preflight_cache_key(
        &config,
        &model_fingerprint,
        &config.device,
        hydra_train_runtime::config::default_num_threads_for_system(),
    );

    let cached_runtime = EffectiveRuntimeConfig {
        selected: selected_runtime_config(77, 33, 3),
        loader: LoaderRuntimeConfig {
            num_threads: Some(4),
            buffer_games: 256,
            buffer_samples: 1024,
            archive_queue_bound: 16,
        },
    };
    let paths = RlPreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: key,
            runtime: cached_runtime,
            benchmark: None,
        },
    )
    .expect("write matching RL cache entry");

    let config_path = write_temp_file("rl-preflight-cache-hit-config", "yaml", "batch_size: 256\n");
    let device = burn::backend::libtorch::LibTorchDevice::Cpu;
    let result = run_rl_preflight(&config_path, &config, &device)
        .expect("RL cache hit should return Ok without probing");

    assert_eq!(
        result.selected_games_per_batch, 256,
        "games_per_batch should come from cached loader.buffer_games"
    );
    assert_eq!(
        result.selected_microbatch_size, 77,
        "microbatch_size should come from cached selected.train_microbatch_size"
    );
    assert!(
        result.rl_games_probe_results.is_empty(),
        "cache hit should skip RL games probing"
    );
    assert!(
        result.rl_microbatch_probe_results.is_empty(),
        "cache hit should skip RL microbatch probing"
    );

    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn apply_cached_runtime_applies_loader_tuple_for_fresh_start() {
    let output_dir = unique_test_path("loader-cache-apply-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts.create_root_dir().expect("create artifact root");
    let mut config = dummy_config();
    config.output_dir = output_dir.clone();
    config.batch_size = 128;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = Some(32);
    config.num_threads = Some(1);
    config.buffer_games = 16;
    config.buffer_samples = 128;
    config.archive_queue_bound = 8;
    let model_config = HydraModelConfig::learner();
    let paths = PreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: preflight_cache_key(
                &config,
                &model_fingerprint_input(&model_config),
                &config.device,
                hydra_train_runtime::config::default_num_threads_for_system(),
            ),
            runtime: EffectiveRuntimeConfig {
                selected: selected_runtime_config(32, 16, 4),
                loader: LoaderRuntimeConfig {
                    num_threads: Some(2),
                    buffer_games: 7,
                    buffer_samples: 64,
                    archive_queue_bound: 3,
                },
            },
            benchmark: None,
        },
    )
    .expect("write safe preflight cache");
    let resume = ResumeContext {
        checkpoint_base: None,
        state: None,
        optimizer_base: None,
        session_start_global_step: 0,
        start_epoch: 0,
    };

    apply_cached_bc_runtime_if_matching(&mut config, &resume, &artifacts, &model_config)
        .expect("fresh safe cache should apply");

    assert_eq!(config.batch_size, 128);
    assert_eq!(config.microbatch_size, Some(32));
    assert_eq!(config.validation_microbatch_size, Some(16));
    assert_eq!(config.num_threads, Some(2));
    assert_eq!(config.buffer_games, 7);
    assert_eq!(config.buffer_samples, 64);
    assert_eq!(config.archive_queue_bound, 3);
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn apply_cached_runtime_keeps_loader_tuple_on_partial_resume() {
    let output_dir = unique_test_path("loader-cache-partial-resume-out");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts.create_root_dir().expect("create artifact root");
    let mut config = dummy_config();
    config.output_dir = output_dir.clone();
    config.batch_size = 128;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = Some(32);
    config.num_threads = Some(1);
    config.buffer_games = 16;
    config.buffer_samples = 128;
    config.archive_queue_bound = 8;
    let model_config = HydraModelConfig::learner();
    let paths = PreflightPaths::new(&artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: preflight_cache_key(
                &config,
                &model_fingerprint_input(&model_config),
                &config.device,
                hydra_train_runtime::config::default_num_threads_for_system(),
            ),
            runtime: EffectiveRuntimeConfig {
                selected: selected_runtime_config(32, 16, 4),
                loader: LoaderRuntimeConfig {
                    num_threads: Some(2),
                    buffer_games: 7,
                    buffer_samples: 64,
                    archive_queue_bound: 3,
                },
            },
            benchmark: None,
        },
    )
    .expect("write safe preflight cache");
    let resume = ResumeContext {
        checkpoint_base: None,
        state: Some(crate::resume::build_resume_state(
            1,
            3,
            7,
            None,
            crate::resume::test_runtime_resume_contract(128, 64, 32),
        )),
        optimizer_base: None,
        session_start_global_step: 7,
        start_epoch: 1,
    };

    apply_cached_bc_runtime_if_matching(&mut config, &resume, &artifacts, &model_config)
        .expect("partial resume should ignore safe cache");

    assert_eq!(config.batch_size, 128);
    assert_eq!(config.microbatch_size, Some(64));
    assert_eq!(config.validation_microbatch_size, Some(32));
    assert_eq!(config.num_threads, Some(1));
    assert_eq!(config.buffer_games, 16);
    assert_eq!(config.buffer_samples, 128);
    assert_eq!(config.archive_queue_bound, 8);
    let _ = fs::remove_dir_all(&output_dir);
}
