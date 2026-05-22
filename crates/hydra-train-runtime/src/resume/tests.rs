use super::*;
use hydra_train_types::phase::TrainingPhase;

fn dummy_config() -> TrainConfig {
    TrainConfig {
        data_dir: "/data".into(),
        output_dir: "/output".into(),
        num_epochs: 4,
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
        seed: 7,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        bc_head_profile: crate::config::BcHeadProfile::Full,
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
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
        keep_step_checkpoints: false,
        launch_tensorboard: false,
        tensorboard_host: "127.0.0.1".to_string(),
        tensorboard_port: 6006,
        background: false,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: Some(1),
    }
}

fn dummy_best_validation() -> BestValidation {
    BestValidation {
        policy_loss: 0.25,
        agreement: 0.8,
    }
}

fn resume_state(
    skip_optimizer_steps_in_epoch: usize,
    runtime: RuntimeResumeContract,
) -> BcResumeState {
    BcResumeState {
        schema_version: 3,
        resume_semantics: ResumeSemantics::RestoreOptimizerSkipSeenSamples,
        next_epoch: 2,
        skip_optimizer_steps_in_epoch,
        global_step: 12,
        best_validation: Some(dummy_best_validation()),
        runtime,
        saved_at_unix_s: 1,
    }
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
    let mut state = resume_state(0, current);
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
    let state = resume_state(0, checkpoint_runtime);
    let current = test_runtime_resume_contract(256, 32, 16);

    assert_eq!(
        validate_resume_runtime_compatibility(&state, current),
        Ok(())
    );
}

#[test]
fn validate_rl_resume_runtime_compatibility_rejects_mismatched_runtime() {
    let state = RlResumeState {
        schema_version: 1,
        resume_semantics: RlResumeSemantics::RestoreOptimizerFreshSelfPlay,
        global_step: 10,
        pipeline_state: PipelineState {
            phase: TrainingPhase::ExitPondering,
            ..PipelineState::default()
        },
        runtime: RlRuntimeResumeContract {
            games_per_batch: 8,
            microbatch_size: 16,
            phase: RlPhaseConfig::ExitPondering,
            precision_mode: PrecisionMode::Fp32,
            requested_precision: PrecisionMode::Fp32,
            effective_precision: EffectivePrecision::Fp32,
        },
        saved_at_unix_s: 1,
    };

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
fn deserialize_bc_runtime_backfills_legacy_bf16_precision_contract() {
    let state: BcResumeState = serde_yaml::from_str(
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
    )
    .expect("legacy BC resume state should parse");

    assert_eq!(state.runtime.precision_mode, PrecisionMode::Bf16Autocast);
    assert_eq!(
        state.runtime.requested_precision,
        PrecisionMode::Bf16Autocast
    );
    assert_eq!(
        state.runtime.effective_precision,
        EffectivePrecision::Fp32NoopForBf16Request
    );
}

#[test]
fn deserialize_rl_runtime_backfills_legacy_precision_contract() {
    let state: RlResumeState = serde_yaml::from_str(
        r#"schema_version: 1
resume_semantics: RestoreOptimizerFreshSelfPlay
global_step: 9
pipeline_state:
  phase: DrdaAchSelfPlay
  gpu_hours_used: 0.0
  total_games: 4
  total_samples: 16
  learner_version: 0
  actor_version: 0
runtime:
  games_per_batch: 4
  microbatch_size: 16
  phase: drda_ach_self_play
  precision_mode: bf16_autocast
saved_at_unix_s: 1
"#,
    )
    .expect("legacy RL resume state should parse");

    assert_eq!(state.runtime.precision_mode, PrecisionMode::Bf16Autocast);
    assert_eq!(
        state.runtime.requested_precision,
        PrecisionMode::Bf16Autocast
    );
    assert_eq!(
        state.runtime.effective_precision,
        EffectivePrecision::Fp32NoopForBf16Request
    );
}
