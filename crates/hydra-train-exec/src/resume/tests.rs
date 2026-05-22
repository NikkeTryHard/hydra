use super::*;
use hydra_train_types::phase::TrainingPhase;
use std::fs;

use hydra_train_runtime::config::{
    EffectivePrecision, PrecisionMode, RlPhaseConfig, RlTrainConfig, TrainConfig,
};
use hydra_train_types::config::DEFAULT_RL_MICROBATCH_SIZE;

fn dummy_train_config() -> TrainConfig {
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
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: Default::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
        bc_head_profile: hydra_train_runtime::config::BcHeadProfile::Full,
        experimental_backbone_profile: None,
        validation_gates: Default::default(),
        rl: None,
        bc: Default::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        precision_mode: PrecisionMode::Fp32,
        buffer_games: 1,
        buffer_samples: 1,
        num_threads: Some(1),
        tensorboard: false,
        archive_queue_bound: 1,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 0,
        log_every_n_steps: 1,
        validate_every_n_steps: 1,
        checkpoint_every_n_steps: 1,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: Some(1),
    }
}

fn unique_test_path(prefix: &str, label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{label}-{unique}"))
}

fn dummy_config() -> TrainConfig {
    let mut config = dummy_train_config();
    config.num_epochs = 4;
    config.seed = 7;
    config.precision_mode = PrecisionMode::Fp32;
    config.num_threads = Some(1);
    config
}

fn dummy_best_validation() -> BestValidation {
    BestValidation {
        policy_loss: 0.25,
        agreement: 0.8,
    }
}

fn write_yaml_file(label: &str, contents: &str) -> PathBuf {
    let path = unique_test_path("hydra-resume-test", label).with_extension("yaml");
    fs::write(&path, contents).expect("yaml fixture should be writable");
    path
}

#[test]
fn checkpoint_base_and_latest_path_helpers_cover_latest_and_non_latest_names() {
    let latest = Path::new("/tmp/latest_model.mpk");
    let other = Path::new("/tmp/epoch_1_model.mpk");

    assert_eq!(
        checkpoint_base_from_path(latest),
        PathBuf::from("/tmp/latest_model")
    );
    assert_eq!(
        checkpoint_base_from_path(other),
        PathBuf::from("/tmp/epoch_1_model")
    );
    assert_eq!(
        checkpoint_base_from_path(Path::new("/tmp/latest_model")),
        PathBuf::from("/tmp/latest_model")
    );

    assert_eq!(
        latest_state_path_for_checkpoint_base(Path::new("/tmp/latest_model")),
        Some(PathBuf::from("/tmp/latest_state.yaml"))
    );
    assert_eq!(
        latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/latest_model")),
        Some(PathBuf::from("/tmp/latest_optimizer"))
    );
    assert!(latest_state_path_for_checkpoint_base(Path::new("/tmp/epoch_1_model")).is_none());
    assert!(latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/epoch_1_model")).is_none());
}

#[test]
fn runtime_contract_helpers_compute_expected_values() {
    let config = dummy_config();
    let bc = runtime_resume_contract(&config);
    assert_eq!(bc.batch_size, 256);
    assert_eq!(bc.train_microbatch_size, 64);
    assert_eq!(bc.validation_microbatch_size, 32);
    assert_eq!(bc.accum_steps, 4);
    assert_eq!(bc.precision_mode, PrecisionMode::Fp32);
    assert_eq!(bc.requested_precision, PrecisionMode::Fp32);
    assert_eq!(bc.effective_precision, EffectivePrecision::Fp32);

    let rl = rl_runtime_resume_contract(&RlTrainConfig::default());
    assert_eq!(rl.games_per_batch, RlTrainConfig::default().games_per_batch);
    assert_eq!(rl.microbatch_size, DEFAULT_RL_MICROBATCH_SIZE);
    assert_eq!(rl.phase, RlTrainConfig::default().phase);
    assert_eq!(rl.precision_mode, PrecisionMode::Fp32);
    assert_eq!(rl.requested_precision, PrecisionMode::Fp32);
    assert_eq!(rl.effective_precision, EffectivePrecision::Fp32);
}

#[test]
fn validate_resume_runtime_compatibility_checks_batch_and_partial_epoch_contracts() {
    let current = test_runtime_resume_contract(256, 64, 32);
    let mut state = build_resume_state(2, 0, 12, Some(dummy_best_validation()), current);
    assert_eq!(
        validate_resume_runtime_compatibility(&state, current),
        Ok(())
    );

    let mismatched_batch = test_runtime_resume_contract(128, 64, 32);
    let err = validate_resume_runtime_compatibility(&state, mismatched_batch)
        .expect_err("batch size mismatch should be rejected");
    assert!(err.contains("resume batch_size mismatch"));

    state.skip_optimizer_steps_in_epoch = 3;
    let mismatched_partial = test_runtime_resume_contract(256, 32, 32);
    let err = validate_resume_runtime_compatibility(&state, mismatched_partial)
        .expect_err("partial epoch resume should require identical runtime contract");
    assert!(err.contains("partial-epoch resume requires identical runtime contract"));

    state.skip_optimizer_steps_in_epoch = 1;
    let mut mismatched_precision = current;
    mismatched_precision.precision_mode = PrecisionMode::Bf16Autocast;
    let err = validate_resume_runtime_compatibility(&state, mismatched_precision)
        .expect_err("partial epoch resume should reject precision mode mismatch");
    assert!(err.contains("partial-epoch resume requires identical runtime contract"));
}

#[test]
fn validate_resume_runtime_compatibility_allows_epoch_boundary_runtime_changes() {
    let checkpoint_runtime = test_runtime_resume_contract(256, 64, 32);
    let state = build_resume_state(2, 0, 12, Some(dummy_best_validation()), checkpoint_runtime);
    let current = test_runtime_resume_contract(256, 32, 16);

    assert_eq!(
        validate_resume_runtime_compatibility(&state, current),
        Ok(())
    );
}

#[test]
fn validate_rl_resume_runtime_compatibility_rejects_mismatched_runtime() {
    let state = build_rl_resume_state(
        10,
        PipelineState {
            phase: TrainingPhase::ExitPondering,
            ..PipelineState::default()
        },
        RlRuntimeResumeContract {
            games_per_batch: 8,
            microbatch_size: 16,
            phase: RlPhaseConfig::ExitPondering,
            precision_mode: PrecisionMode::Fp32,
            requested_precision: PrecisionMode::Fp32,
            effective_precision: EffectivePrecision::Fp32,
        },
    );

    let err = validate_rl_resume_runtime_compatibility(
        &state,
        RlRuntimeResumeContract {
            games_per_batch: 16,
            microbatch_size: 16,
            phase: RlPhaseConfig::ExitPondering,
            precision_mode: PrecisionMode::Fp32,
            requested_precision: PrecisionMode::Fp32,
            effective_precision: EffectivePrecision::Fp32,
        },
    )
    .expect_err("RL runtime mismatch should be rejected");

    assert!(err.contains("RL resume runtime mismatch"));
}

#[test]
fn read_resume_state_rejects_schema_and_semantics_mismatches() {
    let schema_path = write_yaml_file(
        "bad-bc-schema",
        r#"schema_version: 2
resume_semantics: RestoreOptimizerSkipSeenSamples
next_epoch: 1
skip_optimizer_steps_in_epoch: 0
global_step: 4
best_validation: null
runtime:
  batch_size: 256
  train_microbatch_size: 64
  validation_microbatch_size: 32
  accum_steps: 4
  precision_mode: fp32
saved_at_unix_s: 1
"#,
    );
    let schema_err = read_resume_state(&schema_path).expect_err("schema mismatch should fail");
    assert!(schema_err.contains("unsupported resume schema_version 2"));

    let semantics_path = write_yaml_file(
        "bad-bc-semantics",
        r#"schema_version: 3
resume_semantics: RestoreOptimizerFreshSelfPlay
next_epoch: 1
skip_optimizer_steps_in_epoch: 0
global_step: 4
best_validation: null
runtime:
  batch_size: 256
  train_microbatch_size: 64
  validation_microbatch_size: 32
  accum_steps: 4
  precision_mode: fp32
saved_at_unix_s: 1
"#,
    );
    let semantics_err =
        read_resume_state(&semantics_path).expect_err("semantics mismatch should fail");
    assert!(semantics_err.contains("failed to parse resume state"));
}

#[test]
fn read_resume_state_backfills_legacy_bf16_precision_contract() {
    let path = write_yaml_file(
        "legacy-bf16-bc-runtime",
        r#"schema_version: 3
resume_semantics: RestoreOptimizerSkipSeenSamples
next_epoch: 2
skip_optimizer_steps_in_epoch: 3
global_step: 11
best_validation: null
runtime:
  batch_size: 256
  train_microbatch_size: 64
  validation_microbatch_size: 32
  accum_steps: 4
  precision_mode: bf16_autocast
saved_at_unix_s: 1
"#,
    );

    let state = read_resume_state(&path).expect("legacy BF16 state should deserialize");

    assert_eq!(state.runtime.precision_mode, PrecisionMode::Bf16Autocast);
    assert_eq!(
        state.runtime.requested_precision,
        PrecisionMode::Bf16Autocast
    );
    assert_eq!(
        state.runtime.effective_precision,
        EffectivePrecision::Fp32NoopForBf16Request
    );

    let mut current = state.runtime;
    current.effective_precision = EffectivePrecision::Fp32;
    let err = validate_resume_runtime_compatibility(&state, current)
        .expect_err("partial-epoch resume should compare backfilled effective precision");
    assert!(err.contains("partial-epoch resume requires identical runtime contract"));
}

#[test]
fn read_rl_resume_state_rejects_schema_mismatch() {
    let path = write_yaml_file(
        "bad-rl-schema",
        r#"schema_version: 2
resume_semantics: RestoreOptimizerFreshSelfPlay
global_step: 7
pipeline_state:
  phase: ExitPondering
  total_games: 0
  total_samples: 0
  gpu_hours_used: 0.0
  learner_version: 0
runtime:
  games_per_batch: 8
  microbatch_size: 16
  phase: ExitPondering
  precision_mode: fp32
saved_at_unix_s: 1
"#,
    );
    let err = read_rl_resume_state(&path).expect_err("schema mismatch should fail");
    assert!(err.contains("failed to parse RL resume state"));
}

#[test]
fn resume_context_helpers_cover_state_access_and_restore_flags() {
    let runtime = test_runtime_resume_contract(256, 64, 32);
    let state = build_resume_state(3, 2, 11, Some(dummy_best_validation()), runtime);
    let ctx = ResumeContext {
        checkpoint_base: None,
        state: Some(state.clone()),
        optimizer_base: None,
        session_start_global_step: state.global_step,
        start_epoch: state.next_epoch,
    };

    assert_eq!(ctx.best_validation(), Some(dummy_best_validation()));
    assert_eq!(ctx.steps_to_skip_for_epoch(3), 2);
    assert_eq!(ctx.steps_to_skip_for_epoch(1), 0);
    assert!(ctx.restores_optimizer_state());

    let empty = ResumeContext {
        checkpoint_base: None,
        state: None,
        optimizer_base: None,
        session_start_global_step: 0,
        start_epoch: 0,
    };
    assert_eq!(empty.best_validation(), None);
    assert_eq!(empty.steps_to_skip_for_epoch(0), 0);
    assert!(!empty.restores_optimizer_state());
}

#[test]
fn rl_resume_context_restore_flag_tracks_presence_of_state() {
    let ctx = RlResumeContext {
        checkpoint_base: None,
        state: None,
        optimizer_base: None,
        session_start_global_step: 0,
    };
    assert!(!ctx.restores_optimizer_state());

    let ctx = RlResumeContext {
        checkpoint_base: None,
        state: Some(build_rl_resume_state(
            5,
            PipelineState::default(),
            RlRuntimeResumeContract {
                games_per_batch: 8,
                microbatch_size: 16,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: PrecisionMode::Fp32,
                requested_precision: PrecisionMode::Fp32,
                effective_precision: EffectivePrecision::Fp32,
            },
        )),
        optimizer_base: None,
        session_start_global_step: 5,
    };
    assert!(ctx.restores_optimizer_state());
}

#[test]
fn banner_and_pause_messages_include_runtime_details() {
    let state = build_resume_state(
        1,
        3,
        9,
        Some(dummy_best_validation()),
        test_runtime_resume_contract(256, 64, 32),
    );
    let resume_banner = resume_banner_message(&state, None);
    assert!(resume_banner.contains("global_step=9"));
    assert!(resume_banner.contains("skipping 3 completed optimizer steps"));
    assert!(resume_banner.contains("runtime=train_mb:64 val_mb:32 accum_steps:4"));
    assert!(resume_banner.contains("requested_precision=fp32 effective_precision=fp32"));

    let immediate_state =
        build_resume_state(0, 0, 1, None, test_runtime_resume_contract(256, 64, 32));
    let immediate_banner = resume_banner_message(&immediate_state, None);
    assert!(immediate_banner.contains("resuming at epoch 1 with new updates immediately"));

    let effective_banner = resume_banner_message(
        &immediate_state,
        Some(test_runtime_resume_contract(256, 32, 16)),
    );
    assert!(effective_banner.contains("runtime=train_mb:64 val_mb:32 accum_steps:4"));
    assert!(effective_banner.contains("effective_runtime=train_mb:32 val_mb:16 accum_steps:8 requested_precision=fp32 effective_precision=fp32"));

    let unchanged_runtime_banner = resume_banner_message(
        &immediate_state,
        Some(test_runtime_resume_contract(256, 64, 32)),
    );
    assert!(!unchanged_runtime_banner.contains("effective_runtime="));

    let rl_banner = rl_resume_banner_message(&build_rl_resume_state(
        10,
        PipelineState {
            phase: TrainingPhase::ExitPondering,
            total_games: 12,
            total_samples: 128,
            ..PipelineState::default()
        },
        RlRuntimeResumeContract {
            games_per_batch: 8,
            microbatch_size: 16,
            phase: RlPhaseConfig::ExitPondering,
            precision_mode: PrecisionMode::Fp32,
            requested_precision: PrecisionMode::Fp32,
            effective_precision: EffectivePrecision::Fp32,
        },
    ));
    assert!(rl_banner.contains("phase=ExitPondering"));
    assert!(rl_banner.contains("games=12 samples=128"));
    assert!(rl_banner.contains("runtime=games_per_batch:8 microbatch_size:16"));
    assert!(rl_banner.contains("requested_precision=fp32 effective_precision=fp32"));

    let paused = paused_training_message(&EpochContinuation {
        next_epoch: 2,
        skip_optimizer_steps_in_epoch: 4,
        epoch_completed: false,
    });
    assert!(paused.contains("resume_epoch=3"));
    assert!(paused.contains("skipped_optimizer_steps_in_epoch=4"));
}

#[test]
fn write_resume_state_round_trips_and_load_helpers_detect_latest_files() {
    let root = unique_test_path("hydra-resume-test", "resume-roundtrip");
    fs::create_dir_all(&root).expect("temp root should be creatable");
    let checkpoint = root.join("latest_model.mpk");
    fs::write(&checkpoint, b"model").expect("checkpoint marker should be writable");
    let latest_state = root.join("latest_state.yaml");
    let latest_optimizer = root.join("latest_optimizer.bin");
    fs::write(&latest_optimizer, b"optimizer").expect("optimizer marker should be writable");

    let state = build_resume_state(
        2,
        1,
        7,
        Some(dummy_best_validation()),
        test_runtime_resume_contract(256, 64, 32),
    );
    write_resume_state(&latest_state, &state).expect("resume state should write");

    let loaded = read_resume_state(&latest_state).expect("written resume state should parse");
    assert_eq!(loaded, state);

    let mut config = dummy_config();
    config.resume_checkpoint = Some(checkpoint);
    let ctx = ResumeContext::load(&config).expect("resume context should load latest files");
    assert_eq!(ctx.session_start_global_step, 7);
    assert_eq!(ctx.start_epoch, 2);
    assert_eq!(ctx.optimizer_base, Some(root.join("latest_optimizer")));
    assert_eq!(ctx.state, Some(state));
}

#[test]
fn rl_resume_context_load_detects_latest_state_and_optimizer() {
    let root = unique_test_path("hydra-resume-test", "rl-resume-load");
    fs::create_dir_all(&root).expect("temp root should be creatable");
    let checkpoint = root.join("latest_model.mpk");
    fs::write(&checkpoint, b"model").expect("checkpoint marker should be writable");
    let latest_state = root.join("latest_state.yaml");
    let latest_optimizer = root.join("latest_optimizer.bin");
    fs::write(&latest_optimizer, b"optimizer").expect("optimizer marker should be writable");

    let state = build_rl_resume_state(
        5,
        PipelineState {
            phase: TrainingPhase::ExitPondering,
            ..PipelineState::default()
        },
        RlRuntimeResumeContract {
            games_per_batch: 8,
            microbatch_size: 16,
            phase: RlPhaseConfig::ExitPondering,
            precision_mode: PrecisionMode::Fp32,
            requested_precision: PrecisionMode::Fp32,
            effective_precision: EffectivePrecision::Fp32,
        },
    );
    let yaml = serde_yaml::to_string(&state).expect("RL resume state should serialize");
    fs::write(&latest_state, yaml).expect("RL resume state should write");

    let mut config = dummy_config();
    config.resume_checkpoint = Some(checkpoint);
    let ctx = RlResumeContext::load(&config).expect("RL resume context should load latest files");
    assert_eq!(ctx.session_start_global_step, 5);
    assert_eq!(ctx.optimizer_base, Some(root.join("latest_optimizer")));
    assert_eq!(ctx.state, Some(state));
}
