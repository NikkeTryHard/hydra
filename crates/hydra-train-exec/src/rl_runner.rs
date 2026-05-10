use std::time::Instant;

use colored::Colorize;

use crate::nvtx;
use crate::presentation::format_status_line;
use hydra_selfplay::{
    CooperativeSelfPlayCoordinator, CooperativeSelfPlayRequest, generate_self_play_rl_batch_reuse,
};
use hydra_train_algo::distill::{DistillConfig, DistillState};
use hydra_train_algo::drda::RebaseTracker;
use hydra_train_runtime::config::{RlTrainConfig, TrainConfig};
use hydra_train_runtime::head_gates::{AdvancedHead, HeadState};
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_LOGGING, PROFILING_STAGE_RL_STEP,
    PROFILING_STAGE_SELF_PLAY, PROFILING_STAGE_TRAIN, ProfilingEnvelope,
};
use hydra_train_runtime::progress::RlStepLogEntry;
use hydra_train_runtime::status::{reached_session_step_budget, session_steps_completed};
use hydra_train_types::orchestrator::{
    MaintenancePlan, OrchestratorPlanInputs, PhaseTrainReport, maintenance_plan_from_inputs,
};
use hydra_train_types::phase::{PipelineState, TrainingPhase};

use crate::artifacts::{
    RlArtifactPaths, append_rl_step_log_to_writer, save_latest_rl_checkpoint_and_state,
};
use crate::bootstrap::{RlTrainingBootstrap, RlTrainingRuntime};
use crate::presentation::timestamped;
use crate::resume::{RlRuntimeResumeContract, build_rl_resume_state};
use crate::rl_step::{RlPhaseTrainRequest, rl_phase_train_step_with_controller};

/// Converts the scalar maintenance plan into the live ExIt producer config.
#[must_use]
pub fn live_exit_config_from_plan(
    plan: &MaintenancePlan,
) -> hydra_search_labels::live_exit::LiveExitConfig {
    hydra_search_labels::live_exit::LiveExitConfig {
        enabled: plan.shallow_exit_enabled || plan.deep_exit_enabled,
        exit_config: hydra_search_labels::exit::ExitConfig::default_phase3(),
    }
}

/// Formats the RL runtime mode summary shown at training start.
pub fn rl_mode_summary(rl_config: &RlTrainConfig, total_steps: usize) -> String {
    format!(
        "phase={:?} games_per_batch={} temperature={:.2} total_steps={}",
        rl_config.phase, rl_config.games_per_batch, rl_config.temperature, total_steps
    )
}

/// Computes the deterministic per-step base seed.
pub fn base_seed_for_step(seed: u64, global_step: usize) -> u64 {
    seed.wrapping_add(global_step as u64 * 1_000_003)
}

/// Expands the per-step seed into deterministic per-game seeds.
pub fn game_seeds_for_batch(base_seed: u64, games_per_batch: usize) -> Vec<u64> {
    (0..games_per_batch)
        .map(|idx| base_seed.wrapping_add(idx as u64))
        .collect()
}

/// TensorBoard scalar encoding for advanced-head state.
pub fn head_state_scalar(head_state: HeadState) -> f32 {
    match head_state {
        HeadState::Off => 0.0,
        HeadState::Warmup => 1.0,
        HeadState::Active => 2.0,
    }
}

/// Returns whether the current step should emit progress for a session-relative cadence.
pub fn should_log_progress(
    global_step: usize,
    session_start_global_step: usize,
    log_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(log_every_n_steps)
}

/// Returns whether the current step should persist a checkpoint for a session-relative cadence.
pub fn should_save_checkpoint(
    global_step: usize,
    session_start_global_step: usize,
    checkpoint_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(checkpoint_every_n_steps)
}

/// Formats the final RL completion summary.
pub fn final_rl_summary(total_games: u64, total_samples: u64) -> String {
    format!(
        "{} {}",
        format!("games={total_games}").green(),
        format!("samples={total_samples}").green()
    )
}

/// Advances scalar RL pipeline counters after a successful RL train step.
pub fn advance_rl_pipeline_state(
    state: &mut PipelineState,
    games_per_batch: usize,
    batch_samples: usize,
    total_steps: usize,
) {
    state.total_games += games_per_batch as u64;
    state.total_samples += batch_samples as u64;
    state.increment_learner_version();
    state.tick_gpu_hours(state.phase.gpu_hours_budget() as f32 / total_steps.max(1) as f32);
}

/// Inputs needed to build an RL JSONL step-log entry.
pub struct RlStepEntryInputs<'a> {
    /// Global optimizer step after incrementing.
    pub global_step: usize,
    /// Current pipeline training phase.
    pub phase: &'a TrainingPhase,
    /// Scalar train loss.
    pub loss: f64,
    /// Effective learning rate used for the step.
    pub effective_lr: f64,
    /// Effective ExIt loss weight used for the step.
    pub exit_weight: f32,
    /// Self-play games generated for this step.
    pub games_per_batch: usize,
    /// Samples present in the RL batch.
    pub samples_in_batch: usize,
    /// Cumulative games observed by the pipeline.
    pub total_games: u64,
    /// Cumulative samples observed by the pipeline.
    pub total_samples: u64,
    /// Delta-Q head state after the step.
    pub delta_q_state: HeadState,
    /// Optional profiling envelope for the step.
    pub profiling: Option<ProfilingEnvelope>,
}

/// Builds the persisted RL JSONL step-log entry.
pub fn make_rl_step_entry(inputs: RlStepEntryInputs<'_>) -> RlStepLogEntry {
    RlStepLogEntry {
        global_step: inputs.global_step,
        phase: format!("{:?}", inputs.phase),
        loss: inputs.loss,
        effective_lr: inputs.effective_lr,
        exit_weight: inputs.exit_weight,
        games_per_batch: inputs.games_per_batch,
        samples_in_batch: inputs.samples_in_batch,
        total_games: inputs.total_games,
        total_samples: inputs.total_samples,
        delta_q_state: format!("{:?}", inputs.delta_q_state),
        profiling: inputs.profiling,
        advisories: Vec::new(),
    }
}

/// Formats the human-readable RL progress line.
pub fn format_rl_progress_message(
    global_step: usize,
    loss_value: f64,
    delta_q_state: HeadState,
) -> String {
    timestamped(format!(
        "{} {} {} {}",
        "RL step".bold().cyan(),
        format!("global_step={}", global_step).yellow(),
        format!("loss={loss_value:.4}").green(),
        format!("delta_q={delta_q_state:?}").magenta(),
    ))
}

/// Returns whether the current RL step should emit progress.
pub fn should_emit_rl_progress(
    global_step: usize,
    session_start_global_step: usize,
    log_every_n_steps: usize,
) -> bool {
    should_log_progress(global_step, session_start_global_step, log_every_n_steps)
}

/// Returns whether the current RL step should persist a checkpoint.
pub fn should_persist_rl_checkpoint(
    global_step: usize,
    session_start_global_step: usize,
    checkpoint_every_n_steps: usize,
) -> bool {
    should_save_checkpoint(
        global_step,
        session_start_global_step,
        checkpoint_every_n_steps,
    )
}

/// Returns whether the bounded RL session has consumed its step budget.
pub fn should_stop_rl_session(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> bool {
    reached_session_step_budget(global_step, session_start_global_step, max_train_steps)
}

/// Side-effect inputs for persisting one finalized RL step.
pub struct RlStepFinalizeContext<'a> {
    /// RL artifact paths for this run.
    pub artifacts: &'a RlArtifactPaths,
    /// Effective root training configuration.
    pub config: &'a TrainConfig,
    /// Effective RL config after preflight overrides.
    pub rl_config: &'a RlTrainConfig,
    /// Current strict RL resume runtime contract.
    pub current_runtime: RlRuntimeResumeContract,
    /// Global step at session start.
    pub session_start_global_step: usize,
    /// Scheduled total global steps.
    pub total_steps: usize,
    /// Samples in this RL batch.
    pub batch_size: usize,
    /// Scalar train-step report.
    pub report: PhaseTrainReport,
    /// Optional profiling envelope.
    pub profiling: Option<ProfilingEnvelope>,
}

/// Finalizes side effects for one RL train step, including TensorBoard/log/checkpoint writes.
pub fn finalize_rl_step_side_effects(
    runtime: &mut RlTrainingRuntime,
    rebase_tracker: &mut RebaseTracker,
    context: RlStepFinalizeContext<'_>,
) -> Result<bool, String> {
    runtime.global_step += 1;
    advance_rl_pipeline_state(
        &mut runtime.pipeline_state,
        context.rl_config.games_per_batch,
        context.batch_size,
        context.total_steps,
    );
    rebase_tracker.tick(runtime.run_start.elapsed().as_secs_f32());

    let delta_q_state = runtime.head_controller.head_state(AdvancedHead::DeltaQ);
    let loss_value = context.report.loss.unwrap_or(0.0);
    if let Some(tb) = runtime.tb.as_mut() {
        let step = runtime.global_step as i64;
        tb.write_scalar(step, "rl/loss", loss_value as f32)
            .map_err(|err| format!("tensorboard write rl/loss failed: {err}"))?;
        tb.write_scalar(
            step,
            "rl/exit_weight",
            context.report.exit_weight.unwrap_or(0.0),
        )
        .map_err(|err| format!("tensorboard write rl/exit_weight failed: {err}"))?;
        tb.write_scalar(step, "rl/delta_q_state", head_state_scalar(delta_q_state))
            .map_err(|err| format!("tensorboard write rl/delta_q_state failed: {err}"))?;
        tb.write_scalar(
            step,
            "rl/total_games",
            runtime.pipeline_state.total_games as f32,
        )
        .map_err(|err| format!("tensorboard write rl/total_games failed: {err}"))?;
        tb.write_scalar(
            step,
            "rl/total_samples",
            runtime.pipeline_state.total_samples as f32,
        )
        .map_err(|err| format!("tensorboard write rl/total_samples failed: {err}"))?;
    }

    if should_emit_rl_progress(
        runtime.global_step,
        context.session_start_global_step,
        context.config.log_every_n_steps,
    ) {
        println!(
            "{}",
            format_rl_progress_message(runtime.global_step, loss_value, delta_q_state)
        );
        runtime.last_log_step = runtime.global_step;
        runtime.last_log_time = Instant::now();
    }

    let checkpoint_seconds = if should_persist_rl_checkpoint(
        runtime.global_step,
        context.session_start_global_step,
        context.config.checkpoint_every_n_steps,
    ) {
        let state = build_rl_resume_state(
            runtime.global_step,
            runtime.pipeline_state,
            context.current_runtime,
        );
        let checkpoint_started = Instant::now();
        {
            let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
            save_latest_rl_checkpoint_and_state(
                context.artifacts,
                &runtime.model,
                &runtime.optimizer,
                runtime.global_step,
                loss_value,
                &state,
            )?;
        }
        checkpoint_started.elapsed().as_secs_f64()
    } else {
        0.0
    };

    let logging_seconds = {
        let logging_started = Instant::now();
        let _logging_scope = nvtx::scope(PROFILING_STAGE_LOGGING);
        logging_started.elapsed().as_secs_f64()
    };
    let profiling = context.profiling.as_ref().map(|profiling| {
        let mut profiling = profiling.clone();
        let mut children = vec![ProfilingEnvelope::leaf(
            PROFILING_STAGE_LOGGING,
            logging_seconds,
        )];
        if checkpoint_seconds > 0.0 {
            children.push(ProfilingEnvelope::leaf(
                PROFILING_STAGE_CHECKPOINT,
                checkpoint_seconds,
            ));
        }
        profiling.merge_assign(&ProfilingEnvelope::from_children(
            profiling.stage.clone(),
            children,
        ));
        profiling
    });
    let step_entry = make_rl_step_entry(RlStepEntryInputs {
        global_step: runtime.global_step,
        phase: &runtime.pipeline_state.phase,
        loss: loss_value,
        effective_lr: context.report.effective_lr,
        exit_weight: context.report.exit_weight.unwrap_or(0.0),
        games_per_batch: context.rl_config.games_per_batch,
        samples_in_batch: context.batch_size,
        total_games: runtime.pipeline_state.total_games,
        total_samples: runtime.pipeline_state.total_samples,
        delta_q_state,
        profiling,
    });
    append_rl_step_log_to_writer(&mut runtime.step_log, &step_entry)?;

    Ok(should_stop_rl_session(
        runtime.global_step,
        context.session_start_global_step,
        context.config.max_train_steps,
    ))
}

/// Runs one profiled RL step split into self-play and train closures.
pub fn run_profiled_rl_step<B, T>(
    self_play: impl FnOnce() -> Result<B, String>,
    train: impl FnOnce(B) -> Result<(usize, T, PhaseTrainReport), String>,
) -> Result<(usize, T, PhaseTrainReport, ProfilingEnvelope), String> {
    let _rl_step_scope = nvtx::scope(PROFILING_STAGE_RL_STEP);

    let self_play_started = Instant::now();
    let batch = {
        let _self_play_scope = nvtx::scope(PROFILING_STAGE_SELF_PLAY);
        self_play()?
    };
    let self_play_seconds = self_play_started.elapsed().as_secs_f64();

    let train_started = Instant::now();
    let (batch_size, train_output, report) = {
        let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
        train(batch)?
    };
    let train_seconds = train_started.elapsed().as_secs_f64();

    let profiling = ProfilingEnvelope::from_children(
        PROFILING_STAGE_RL_STEP,
        vec![
            ProfilingEnvelope::leaf(PROFILING_STAGE_SELF_PLAY, self_play_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_TRAIN, train_seconds),
        ],
    );

    Ok((batch_size, train_output, report, profiling))
}

/// Runs the full RL training loop, including self-play, train-step, logging, and final state persistence.
pub fn run_rl_training_loop(
    bootstrap: RlTrainingBootstrap,
    mut runtime: RlTrainingRuntime,
) -> Result<(), String> {
    let RlTrainingBootstrap {
        config,
        rl_config,
        resume,
        artifacts,
        model_config: _,
        device_name: _,
        train_device,
        current_runtime,
        session_start_global_step,
        total_steps,
        loss_fn,
        rl_step_cfg,
        gae_config,
    } = bootstrap;

    resume.print_banner();
    println!(
        "{}",
        format_status_line("RL mode:", rl_mode_summary(&rl_config, total_steps))
    );

    let mut rebase_tracker = RebaseTracker::default_phase2();
    let distill_state = DistillState::default();
    let distill_cfg = DistillConfig::fast_distill();
    let mut self_play_coordinator = CooperativeSelfPlayCoordinator::new();

    while runtime.global_step < total_steps {
        let _rl_step_scope = nvtx::scope(PROFILING_STAGE_RL_STEP);
        let step_started = Instant::now();
        let elapsed_secs = runtime.run_start.elapsed().as_secs();
        let batch = {
            let _self_play_scope = nvtx::scope(PROFILING_STAGE_SELF_PLAY);
            let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
                phase: runtime.pipeline_state.phase,
                phase_progress: runtime.pipeline_state.phase_progress(),
                should_advance_phase: runtime.pipeline_state.should_advance_phase(),
                rebase_due: rebase_tracker.should_rebase(),
                distill_due: distill_state.should_distill(&distill_cfg, elapsed_secs),
                distill_should_warn: distill_state.should_warn(0.05),
            });
            let live_exit_cfg = live_exit_config_from_plan(&plan);
            let base_seed = base_seed_for_step(config.seed, runtime.global_step);
            let game_seeds = game_seeds_for_batch(base_seed, rl_config.games_per_batch);
            generate_self_play_rl_batch_reuse(
                &mut self_play_coordinator,
                CooperativeSelfPlayRequest {
                    game_seeds: &game_seeds,
                    temperature: rl_config.temperature,
                    rng_seed: base_seed,
                    live_exit_cfg,
                },
                &runtime.model,
                &train_device,
                &gae_config,
            )
        };
        let self_play_seconds = step_started.elapsed().as_secs_f64();

        runtime.head_controller.try_activate(AdvancedHead::DeltaQ);
        let train_started = Instant::now();
        let batch_size = batch.batch_size();
        let (model, report) = {
            let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
            rl_phase_train_step_with_controller(
                runtime.model,
                RlPhaseTrainRequest {
                    state: &runtime.pipeline_state,
                    batch: &batch,
                    cfg: &rl_step_cfg,
                    loss_fn: &loss_fn,
                    controller: Some(&mut runtime.head_controller),
                },
                &mut runtime.optimizer,
            )
            .map_err(|err| format!("rl phase train step failed: {err}"))?
        };
        let train_seconds = train_started.elapsed().as_secs_f64();
        runtime.model = model;
        runtime.head_controller.tick_warmup();
        let profiling = ProfilingEnvelope::from_children(
            PROFILING_STAGE_RL_STEP,
            vec![
                ProfilingEnvelope::leaf(PROFILING_STAGE_SELF_PLAY, self_play_seconds),
                ProfilingEnvelope::leaf(PROFILING_STAGE_TRAIN, train_seconds),
            ],
        );

        if finalize_rl_step_side_effects(
            &mut runtime,
            &mut rebase_tracker,
            RlStepFinalizeContext {
                artifacts: &artifacts,
                config: &config,
                rl_config: &rl_config,
                current_runtime,
                session_start_global_step,
                total_steps,
                batch_size,
                report,
                profiling: Some(profiling),
            },
        )? {
            break;
        }
    }

    let final_state =
        build_rl_resume_state(runtime.global_step, runtime.pipeline_state, current_runtime);
    {
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        save_latest_rl_checkpoint_and_state(
            &artifacts,
            &runtime.model,
            &runtime.optimizer,
            runtime.global_step,
            0.0,
            &final_state,
        )?;
    }

    println!(
        "{}",
        timestamped(format!(
            "{} {}",
            "Finished RL training.".bold().cyan(),
            final_rl_summary(
                runtime.pipeline_state.total_games,
                runtime.pipeline_state.total_samples,
            ),
        ))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::initialize_rl_training_bootstrap;
    use crate::resume::read_rl_resume_state;
    use burn::Tensor;
    use burn::tensor::Int;
    use hydra_train_runtime::config::{
        PrecisionMode, RlPhaseConfig, SourceFilterConfig, TrainConfig, ValidationGateConfig,
    };
    use hydra_train_runtime::preflight::PreflightConfig;
    use hydra_train_types::losses::HydraTargets;
    use hydra_train_types::rl::RlBatch;
    use serde_json::Value;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tboard::SummaryReader;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let base_dir = std::env::temp_dir();
        fs::create_dir_all(&base_dir).expect("create test temp root");
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        base_dir.join(format!("hydra-rl-runner-{label}-{nanos}"))
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    fn read_jsonl_entry(path: &Path) -> Value {
        let raw = fs::read_to_string(path).expect("read jsonl file");
        let line = raw.lines().next().expect("jsonl entry line");
        serde_json::from_str(line).expect("parse jsonl entry")
    }

    fn modified_time(path: &Path) -> SystemTime {
        fs::metadata(path)
            .expect("read file metadata")
            .modified()
            .expect("read file modified time")
    }

    fn tensorboard_tags_from_dir(path: &Path) -> Vec<String> {
        let event_path = fs::read_dir(path)
            .expect("read tensorboard dir")
            .map(|entry| entry.expect("tensorboard dir entry").path())
            .find(|entry| entry.is_file())
            .expect("tensorboard event file");
        let file = fs::File::open(event_path).expect("open tensorboard event file");
        let mut tags = Vec::new();
        for event in SummaryReader::new(file).skip(1) {
            let event = event.expect("decode tensorboard event");
            let summary = match event.what.expect("event payload") {
                tboard::tensorboard::event::What::Summary(summary) => summary,
                other => panic!("expected summary event, got {other:?}"),
            };
            for value in summary.value {
                tags.push(value.tag);
            }
        }
        tags
    }

    fn helper_test_rl_config(output_dir: PathBuf, tensorboard: bool) -> TrainConfig {
        let mut config = dummy_rl_config(output_dir);
        config.tensorboard = tensorboard;
        let rl = config.rl.as_mut().expect("rl config");
        rl.games_per_batch = 1;
        rl.microbatch_size = Some(1);
        config.batch_size = 1;
        config.validation_microbatch_size = Some(1);
        config.buffer_games = 1;
        config.buffer_samples = 1;
        config.archive_queue_bound = 1;
        config
    }

    fn synthetic_phase_report(loss: Option<f64>, exit_weight: Option<f32>) -> PhaseTrainReport {
        PhaseTrainReport {
            phase: TrainingPhase::DrdaAchSelfPlay,
            skipped: false,
            loss,
            effective_lr: 2.5e-4,
            oracle_keep_prob: None,
            kept_oracle_fraction: None,
            exit_weight,
        }
    }

    fn dummy_rl_config(output_dir: PathBuf) -> TrainConfig {
        TrainConfig {
            data_dir: output_dir.join("data"),
            output_dir,
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: None,
            validation_microbatch_size: Some(64),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: SourceFilterConfig::default(),
            augment: false,
            resume_checkpoint: None,
            seed: 7,
            advanced_loss: None,
            validation_gates: ValidationGateConfig::default(),
            rl: Some(RlTrainConfig::default()),
            bc: Default::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(1),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: Some(2),
            max_validation_batches: None,
            max_validation_samples: Some(64),
            preflight: PreflightConfig::default(),
            precision_mode: PrecisionMode::Fp32,
        }
    }

    #[test]
    fn rl_mode_summary_formats_phase_runtime_and_steps() {
        let rl_config = RlTrainConfig {
            games_per_batch: 16,
            temperature: 0.375,
            phase: RlPhaseConfig::ExitPondering,
            learning_rate: None,
            exit_weight: None,
            aux_weight: None,
            microbatch_size: None,
        };

        let summary = rl_mode_summary(&rl_config, 2048);

        assert_eq!(
            summary,
            "phase=ExitPondering games_per_batch=16 temperature=0.38 total_steps=2048"
        );
    }

    #[test]
    fn base_seed_for_step_uses_wrapping_step_stride() {
        let seed = u64::MAX - 2;

        let base_seed = base_seed_for_step(seed, 3);

        assert_eq!(base_seed, seed.wrapping_add(3 * 1_000_003));
    }

    #[test]
    fn game_seeds_for_batch_counts_up_from_base_seed() {
        let seeds = game_seeds_for_batch(41, 4);

        assert_eq!(seeds, vec![41, 42, 43, 44]);
    }

    #[test]
    fn game_seeds_for_batch_wraps_near_u64_max() {
        let seeds = game_seeds_for_batch(u64::MAX - 1, 4);

        assert_eq!(seeds, vec![u64::MAX - 1, u64::MAX, 0, 1]);
    }

    #[test]
    fn base_seed_for_step_leaves_seed_unchanged_at_step_zero() {
        assert_eq!(base_seed_for_step(123, 0), 123);
    }

    #[test]
    fn head_state_scalar_matches_tensorboard_encoding() {
        assert_eq!(head_state_scalar(HeadState::Off), 0.0);
        assert_eq!(head_state_scalar(HeadState::Warmup), 1.0);
        assert_eq!(head_state_scalar(HeadState::Active), 2.0);
    }

    #[test]
    fn session_progress_helpers_respect_session_offset() {
        assert!(!should_log_progress(11, 10, 2));
        assert!(should_log_progress(12, 10, 2));
        assert!(!should_save_checkpoint(14, 10, 5));
        assert!(should_save_checkpoint(15, 10, 5));
    }

    #[test]
    fn session_progress_helpers_include_first_session_step() {
        assert!(should_log_progress(10, 10, 3));
        assert!(should_save_checkpoint(10, 10, 4));
    }

    #[test]
    fn session_progress_helpers_fire_every_step_when_interval_is_one() {
        assert!(should_log_progress(17, 10, 1));
        assert!(should_save_checkpoint(17, 10, 1));
    }

    #[test]
    fn session_progress_helpers_handle_steps_before_session_start() {
        assert!(should_log_progress(8, 10, 2));
        assert!(should_save_checkpoint(8, 10, 5));
        assert_eq!(session_steps_completed(8, 10), 0);
    }

    #[test]
    fn should_emit_and_persist_rl_helpers_delegate_to_session_rules() {
        assert!(should_emit_rl_progress(10, 10, 4));
        assert!(!should_emit_rl_progress(11, 10, 4));
        assert!(should_emit_rl_progress(8, 10, 4));

        assert!(should_persist_rl_checkpoint(10, 10, 6));
        assert!(!should_persist_rl_checkpoint(11, 10, 6));
        assert!(should_persist_rl_checkpoint(8, 10, 6));
    }

    #[test]
    fn reached_session_step_budget_is_false_without_budget() {
        assert!(!reached_session_step_budget(25, 10, None));
    }

    #[test]
    fn reached_session_step_budget_trips_on_exact_session_budget() {
        assert!(reached_session_step_budget(15, 10, Some(5)));
        assert!(!reached_session_step_budget(14, 10, Some(5)));
    }

    #[test]
    fn maintenance_plan_keeps_live_exit_disabled_before_phase2_midpoint() {
        let state = PipelineState {
            phase: TrainingPhase::DrdaAchSelfPlay,
            gpu_hours_used: TrainingPhase::DrdaAchSelfPlay.cumulative_budget_before() as f32
                + (TrainingPhase::DrdaAchSelfPlay.gpu_hours_budget() as f32 * 0.5),
            ..PipelineState::default()
        };
        let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
            phase: state.phase,
            phase_progress: state.phase_progress(),
            should_advance_phase: state.should_advance_phase(),
            rebase_due: RebaseTracker::default_phase2().should_rebase(),
            distill_due: DistillState::default().should_distill(&DistillConfig::fast_distill(), 29),
            distill_should_warn: DistillState::default().should_warn(0.05),
        });

        assert!(!plan.shallow_exit_enabled);
        assert!(!plan.deep_exit_enabled);
        assert!(!live_exit_config_from_plan(&plan).enabled);
    }

    #[test]
    fn maintenance_plan_enables_deep_live_exit_in_exit_phase() {
        let mut rebase = RebaseTracker::default_phase2();
        rebase.tick(40.0);
        let distill = DistillState {
            last_kl_drift: 0.08,
            ..DistillState::default()
        };
        let state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            gpu_hours_used: TrainingPhase::ExitPondering.cumulative_budget_before() as f32,
            ..PipelineState::default()
        };
        let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
            phase: state.phase,
            phase_progress: state.phase_progress(),
            should_advance_phase: state.should_advance_phase(),
            rebase_due: rebase.should_rebase(),
            distill_due: distill.should_distill(&DistillConfig::fast_distill(), 30),
            distill_should_warn: distill.should_warn(0.05),
        });
        let live_exit_cfg = live_exit_config_from_plan(&plan);

        assert!(plan.should_rebase);
        assert!(plan.should_distill);
        assert!(plan.distill_warning);
        assert!(plan.shallow_exit_enabled);
        assert!(plan.deep_exit_enabled);
        assert!(live_exit_cfg.enabled);
    }

    #[test]
    fn maintenance_plan_stays_idle_in_oracle_guiding_phase() {
        let mut rebase = RebaseTracker::default_phase2();
        rebase.tick(100.0);
        let state = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: TrainingPhase::OracleGuiding.cumulative_budget_before() as f32 + 150.0,
            ..PipelineState::default()
        };
        let distill = DistillState {
            last_kl_drift: 0.2,
            ..DistillState::default()
        };
        let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
            phase: state.phase,
            phase_progress: state.phase_progress(),
            should_advance_phase: state.should_advance_phase(),
            rebase_due: rebase.should_rebase(),
            distill_due: distill.should_distill(&DistillConfig::fast_distill(), 999),
            distill_should_warn: distill.should_warn(0.05),
        });

        assert!(!plan.should_rebase);
        assert!(!plan.should_distill);
        assert!(plan.distill_warning);
        assert!(!plan.shallow_exit_enabled);
        assert!(!plan.deep_exit_enabled);
        assert!(!live_exit_config_from_plan(&plan).enabled);
    }

    #[test]
    fn final_rl_summary_reports_games_and_samples() {
        let summary = final_rl_summary(128, 4096);

        assert!(summary.contains("games=128"));
        assert!(summary.contains("samples=4096"));
    }

    #[test]
    fn final_rl_summary_formats_zero_counts_too() {
        let summary = final_rl_summary(0, 0);

        assert!(summary.contains("games=0"));
        assert!(summary.contains("samples=0"));
    }

    #[test]
    fn game_seeds_for_batch_returns_empty_for_zero_games() {
        let seeds = game_seeds_for_batch(123, 0);

        assert!(seeds.is_empty());
    }

    #[test]
    fn advance_rl_pipeline_state_updates_counts_version_and_gpu_budget() {
        let mut state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            total_games: 10,
            total_samples: 20,
            learner_version: 3,
            gpu_hours_used: 1.0,
            ..PipelineState::default()
        };

        advance_rl_pipeline_state(&mut state, 4, 32, 8);

        assert_eq!(state.total_games, 14);
        assert_eq!(state.total_samples, 52);
        assert_eq!(state.learner_version, 4);
        assert!(state.gpu_hours_used > 1.0);
    }

    #[test]
    fn advance_rl_pipeline_state_handles_zero_total_steps_without_nan_budget() {
        let mut state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            gpu_hours_used: 2.0,
            ..PipelineState::default()
        };

        advance_rl_pipeline_state(&mut state, 1, 8, 0);

        assert_eq!(state.total_games, 1);
        assert_eq!(state.total_samples, 8);
        assert_eq!(state.learner_version, 1);
        assert!(state.gpu_hours_used.is_finite());
    }

    #[test]
    fn rl_step_entry_and_progress_message_format_runtime_values() {
        let entry = make_rl_step_entry(RlStepEntryInputs {
            global_step: 12,
            phase: &TrainingPhase::ExitPondering,
            loss: 0.25,
            effective_lr: 5e-4,
            exit_weight: 0.1,
            games_per_batch: 8,
            samples_in_batch: 64,
            total_games: 128,
            total_samples: 1024,
            delta_q_state: HeadState::Warmup,
            profiling: Some(ProfilingEnvelope::leaf(PROFILING_STAGE_RL_STEP, 0.75)),
        });

        assert_eq!(entry.global_step, 12);
        assert_eq!(entry.phase, "ExitPondering");
        assert_eq!(entry.loss, 0.25);
        assert_eq!(entry.effective_lr, 5e-4);
        assert_eq!(entry.exit_weight, 0.1);
        assert_eq!(entry.games_per_batch, 8);
        assert_eq!(entry.samples_in_batch, 64);
        assert_eq!(entry.total_games, 128);
        assert_eq!(entry.total_samples, 1024);
        assert_eq!(entry.delta_q_state, "Warmup");
        assert_eq!(
            entry.profiling.as_ref().map(|p| p.stage.as_str()),
            Some("rl_step")
        );

        let message = format_rl_progress_message(12, 0.25, HeadState::Warmup);
        assert!(message.contains("RL step"));
        assert!(message.contains("global_step=12"));
        assert!(message.contains("loss=0.2500"));
        assert!(message.contains("delta_q=Warmup"));
    }

    #[test]
    fn rl_step_entry_and_progress_message_cover_active_delta_q_state() {
        let entry = make_rl_step_entry(RlStepEntryInputs {
            global_step: 3,
            phase: &TrainingPhase::DrdaAchSelfPlay,
            loss: 1.25,
            effective_lr: 1e-3,
            exit_weight: 0.0,
            games_per_batch: 4,
            samples_in_batch: 40,
            total_games: 12,
            total_samples: 120,
            delta_q_state: HeadState::Active,
            profiling: None,
        });

        assert_eq!(entry.phase, "DrdaAchSelfPlay");
        assert_eq!(entry.delta_q_state, "Active");

        let message = format_rl_progress_message(3, 1.25, HeadState::Active);
        assert!(message.contains("global_step=3"));
        assert!(message.contains("loss=1.2500"));
        assert!(message.contains("delta_q=Active"));
    }

    #[test]
    fn rl_step_entry_and_progress_message_cover_off_delta_q_state() {
        let entry = make_rl_step_entry(RlStepEntryInputs {
            global_step: 1,
            phase: &TrainingPhase::BcWarmStart,
            loss: 0.0,
            effective_lr: 2.5e-4,
            exit_weight: 0.0,
            games_per_batch: 2,
            samples_in_batch: 8,
            total_games: 2,
            total_samples: 8,
            delta_q_state: HeadState::Off,
            profiling: None,
        });

        assert_eq!(entry.phase, "BcWarmStart");
        assert_eq!(entry.delta_q_state, "Off");

        let message = format_rl_progress_message(1, 0.0, HeadState::Off);
        assert!(message.contains("global_step=1"));
        assert!(message.contains("loss=0.0000"));
        assert!(message.contains("delta_q=Off"));
    }

    #[test]
    fn rl_loop_decision_helpers_match_underlying_session_rules() {
        assert!(should_emit_rl_progress(10, 10, 3));
        assert!(!should_emit_rl_progress(11, 10, 3));

        assert!(should_persist_rl_checkpoint(10, 10, 5));
        assert!(!should_persist_rl_checkpoint(11, 10, 5));

        assert!(should_stop_rl_session(15, 10, Some(5)));
        assert!(!should_stop_rl_session(14, 10, Some(5)));
        assert!(!should_stop_rl_session(25, 10, None));
    }

    #[test]
    fn should_stop_rl_session_respects_zero_budget_and_pre_session_clamping() {
        assert!(should_stop_rl_session(10, 10, Some(0)));
        assert!(!should_stop_rl_session(9, 10, Some(1)));
    }

    #[test]
    fn rl_loop_decision_helpers_treat_pre_session_steps_as_immediate_emit_and_persist() {
        assert!(should_emit_rl_progress(8, 10, 4));
        assert!(should_persist_rl_checkpoint(8, 10, 6));
        assert_eq!(session_steps_completed(8, 10), 0);
    }

    #[test]
    fn advance_rl_pipeline_state_uses_phase_budget_fraction_per_step() {
        let mut state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            gpu_hours_used: 3.0,
            ..PipelineState::default()
        };
        let expected_increment = TrainingPhase::ExitPondering.gpu_hours_budget() as f32 / 4.0;

        advance_rl_pipeline_state(&mut state, 2, 16, 4);

        assert_eq!(state.total_games, 2);
        assert_eq!(state.total_samples, 16);
        assert_eq!(state.learner_version, 1);
        assert!((state.gpu_hours_used - (3.0 + expected_increment)).abs() < 1e-6);
    }

    #[test]
    fn run_rl_training_loop_persists_final_state_when_session_is_already_complete() {
        let output_dir = unique_temp_dir("final-save");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let config = dummy_rl_config(output_dir.clone());
        fs::create_dir_all(config.data_dir.clone()).expect("create RL data dir");
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, mut runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");
        let latest_state_path = bootstrap.artifacts.latest_state_path.clone();
        let latest_step_log_path = bootstrap.artifacts.step_log_path.clone();
        let expected_global_step = bootstrap.total_steps;
        runtime.global_step = bootstrap.total_steps;

        run_rl_training_loop(bootstrap, runtime)
            .expect("zero-step RL run should still finalize cleanly");

        assert!(latest_state_path.exists());
        assert!(latest_step_log_path.exists());
        assert!(
            fs::read_to_string(&latest_step_log_path)
                .expect("read empty RL step log")
                .is_empty()
        );

        let state_yaml =
            fs::read_to_string(&latest_state_path).expect("latest RL state should exist");
        assert!(state_yaml.contains("schema_version: 1"));
        assert!(state_yaml.contains(&format!("global_step: {expected_global_step}")));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn run_rl_training_loop_does_not_rewrite_latest_checkpoint_after_boundary_save() {
        let output_dir = unique_temp_dir("final-save-dedupe");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let mut config = dummy_rl_config(output_dir.clone());
        config.log_every_n_steps = 1;
        config.checkpoint_every_n_steps = 1;
        config.max_train_steps = Some(1);
        fs::create_dir_all(config.data_dir.clone()).expect("create RL data dir");
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, mut runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");
        let latest_state_path = bootstrap.artifacts.latest_state_path.clone();
        let latest_model_base = bootstrap.artifacts.latest_model_base.clone();
        let latest_optimizer_base = bootstrap.artifacts.latest_optimizer_base.clone();

        let latest_model_path = latest_model_base.with_extension("mpk");
        let latest_meta_path = latest_model_base.with_extension("meta.json");
        let latest_optimizer_path = latest_optimizer_base.with_extension("bin");
        let mut rebase_tracker = RebaseTracker::default_phase2();

        runtime.head_controller.try_activate(AdvancedHead::DeltaQ);
        runtime.head_controller.tick_warmup();

        finalize_rl_step_side_effects(
            &mut runtime,
            &mut rebase_tracker,
            RlStepFinalizeContext {
                artifacts: &bootstrap.artifacts,
                config: &bootstrap.config,
                rl_config: &bootstrap.rl_config,
                current_runtime: bootstrap.current_runtime,
                session_start_global_step: bootstrap.session_start_global_step,
                total_steps: bootstrap.total_steps,
                batch_size: 6,
                report: synthetic_phase_report(Some(0.75), Some(0.25)),
                profiling: Some(ProfilingEnvelope::leaf(PROFILING_STAGE_RL_STEP, 1.5)),
            },
        )
        .expect("step side effects should succeed");

        let model_before = modified_time(&latest_model_path);
        let meta_before = modified_time(&latest_meta_path);
        let optimizer_before = modified_time(&latest_optimizer_path);
        let state_before = modified_time(&latest_state_path);
        thread::sleep(std::time::Duration::from_millis(1100));

        run_rl_training_loop(bootstrap, runtime).expect("single-step RL run should succeed");

        let state = read_rl_resume_state(&latest_state_path).expect("read latest RL state");
        assert_eq!(state.global_step, 1);
        assert!(latest_model_path.exists());
        assert!(latest_meta_path.exists());
        assert!(latest_optimizer_path.exists());
        assert_eq!(modified_time(&latest_model_path), model_before);
        assert!(modified_time(&latest_meta_path) > meta_before);
        assert_eq!(modified_time(&latest_optimizer_path), optimizer_before);
        assert!(modified_time(&latest_state_path) > state_before);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn finalize_rl_step_side_effects_records_logging_scope_order() {
        let output_dir = unique_temp_dir("nvtx_rl_logging_scope");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let mut config = helper_test_rl_config(output_dir.clone(), false);
        config.log_every_n_steps = 99;
        config.checkpoint_every_n_steps = 99;
        config.max_train_steps = Some(2);
        fs::create_dir_all(config.data_dir.clone()).expect("create RL data dir");
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, mut runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");
        let mut rebase_tracker = RebaseTracker::default_phase2();

        let (_, events) = nvtx::with_test_recorder(|| {
            finalize_rl_step_side_effects(
                &mut runtime,
                &mut rebase_tracker,
                RlStepFinalizeContext {
                    artifacts: &bootstrap.artifacts,
                    config: &bootstrap.config,
                    rl_config: &bootstrap.rl_config,
                    current_runtime: bootstrap.current_runtime,
                    session_start_global_step: bootstrap.session_start_global_step,
                    total_steps: bootstrap.total_steps,
                    batch_size: 6,
                    report: synthetic_phase_report(Some(0.75), Some(0.25)),
                    profiling: Some(ProfilingEnvelope::leaf(PROFILING_STAGE_RL_STEP, 1.5)),
                },
            )
            .expect("finalize RL step side effects should succeed");
        });

        assert_eq!(events, vec!["push:logging", "pop:logging"]);
        cleanup_dir(&output_dir);
    }

    #[test]
    fn finalize_rl_step_side_effects_persists_progress_checkpoint_and_stop_boundary() {
        let output_dir = unique_temp_dir("step-side-effects-boundary");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let mut config = helper_test_rl_config(output_dir.clone(), false);
        config.log_every_n_steps = 1;
        config.checkpoint_every_n_steps = 1;
        config.max_train_steps = Some(1);
        fs::create_dir_all(config.data_dir.clone()).expect("create RL data dir");
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, mut runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");
        let step_log_path = bootstrap.artifacts.step_log_path.clone();
        let latest_state_path = bootstrap.artifacts.latest_state_path.clone();
        let latest_model_base = bootstrap.artifacts.latest_model_base.clone();
        let latest_optimizer_base = bootstrap.artifacts.latest_optimizer_base.clone();
        let mut rebase_tracker = RebaseTracker::default_phase2();

        runtime.head_controller.try_activate(AdvancedHead::DeltaQ);
        runtime.head_controller.tick_warmup();

        let should_stop = finalize_rl_step_side_effects(
            &mut runtime,
            &mut rebase_tracker,
            RlStepFinalizeContext {
                artifacts: &bootstrap.artifacts,
                config: &bootstrap.config,
                rl_config: &bootstrap.rl_config,
                current_runtime: bootstrap.current_runtime,
                session_start_global_step: bootstrap.session_start_global_step,
                total_steps: bootstrap.total_steps,
                batch_size: 6,
                report: synthetic_phase_report(Some(0.75), Some(0.25)),
                profiling: Some(ProfilingEnvelope::leaf(PROFILING_STAGE_RL_STEP, 1.5)),
            },
        )
        .expect("step side effects should succeed");

        assert!(should_stop);
        assert_eq!(runtime.global_step, 1);
        assert_eq!(runtime.last_log_step, 1);
        assert_eq!(runtime.pipeline_state.total_games, 1);
        assert_eq!(runtime.pipeline_state.total_samples, 6);
        assert_eq!(runtime.pipeline_state.learner_version, 1);

        let entry = read_jsonl_entry(&step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(1));
        assert_eq!(entry["phase"].as_str(), Some("DrdaAchSelfPlay"));
        assert_eq!(entry["games_per_batch"].as_u64(), Some(1));
        assert_eq!(entry["total_games"].as_u64(), Some(1));
        assert_eq!(entry["loss"].as_f64(), Some(0.75));
        assert_eq!(entry["effective_lr"].as_f64(), Some(2.5e-4));
        assert_eq!(entry["exit_weight"].as_f64(), Some(0.25));
        assert_eq!(entry["samples_in_batch"].as_u64(), Some(6));
        assert_eq!(entry["total_samples"].as_u64(), Some(6));
        assert!(entry["delta_q_state"].as_str().is_some());
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("rl_step"));

        let state = read_rl_resume_state(&latest_state_path).expect("read latest RL state");
        assert_eq!(state.global_step, 1);
        assert_eq!(state.pipeline_state.phase, TrainingPhase::DrdaAchSelfPlay);
        assert_eq!(state.pipeline_state.total_games, 1);
        assert_eq!(state.pipeline_state.total_samples, 6);

        assert!(latest_model_base.with_extension("mpk").exists());
        assert!(latest_model_base.with_extension("meta.json").exists());
        assert!(latest_optimizer_base.with_extension("bin").exists());

        cleanup_dir(&output_dir);
    }

    #[test]
    fn run_rl_training_loop_records_rl_step_scope_order() {
        let (_, events) = nvtx::with_test_recorder(|| {
            let device = burn::backend::libtorch::LibTorchDevice::Cpu;
            let (_batch_size, _unit, _report, _profiling) = run_profiled_rl_step(
                || {
                    let obs = Tensor::<crate::bootstrap::TrainBackend, 3>::zeros(
                        [1, hydra_core::encoder::NUM_CHANNELS, 34],
                        &device,
                    );
                    let actions =
                        Tensor::<crate::bootstrap::TrainBackend, 1, Int>::zeros([1], &device);
                    let pi_old = Tensor::<crate::bootstrap::TrainBackend, 1>::zeros([1], &device);
                    let advantages =
                        Tensor::<crate::bootstrap::TrainBackend, 1>::zeros([1], &device);
                    let base_logits =
                        Tensor::<crate::bootstrap::TrainBackend, 2>::zeros([1, 46], &device);
                    let targets = HydraTargets {
                        policy_target: Tensor::<crate::bootstrap::TrainBackend, 2>::zeros(
                            [1, 46],
                            &device,
                        ),
                        legal_mask: Tensor::<crate::bootstrap::TrainBackend, 2>::ones(
                            [1, 46],
                            &device,
                        ),
                        value_target: Tensor::<crate::bootstrap::TrainBackend, 1>::zeros(
                            [1],
                            &device,
                        ),
                        grp_target: Tensor::<crate::bootstrap::TrainBackend, 2>::zeros(
                            [1, 24],
                            &device,
                        ),
                        tenpai_target: Tensor::<crate::bootstrap::TrainBackend, 2>::zeros(
                            [1, 4],
                            &device,
                        ),
                        danger_target: Tensor::<crate::bootstrap::TrainBackend, 3>::zeros(
                            [1, 4, 34],
                            &device,
                        ),
                        danger_mask: Tensor::<crate::bootstrap::TrainBackend, 3>::zeros(
                            [1, 4, 34],
                            &device,
                        ),
                        opp_next_target: Tensor::<crate::bootstrap::TrainBackend, 3>::zeros(
                            [1, 4, 34],
                            &device,
                        ),
                        score_pdf_target: Tensor::<crate::bootstrap::TrainBackend, 2>::zeros(
                            [1, 121],
                            &device,
                        ),
                        score_cdf_target: Tensor::<crate::bootstrap::TrainBackend, 2>::zeros(
                            [1, 121],
                            &device,
                        ),
                        oracle_target: None,
                        belief_fields_target: None,
                        belief_fields_mask: None,
                        mixture_weight_target: None,
                        mixture_weight_mask: None,
                        opponent_hand_type_target: None,
                        delta_q_target: None,
                        delta_q_mask: None,
                        safety_residual_target: None,
                        safety_residual_mask: None,
                        oracle_guidance_mask: None,
                        target_presence: None,
                    };
                    Ok(RlBatch {
                        obs,
                        actions,
                        pi_old,
                        advantages,
                        base_logits,
                        targets,
                        exit_target: None,
                        exit_mask: None,
                    })
                },
                |batch| {
                    Ok((
                        batch.batch_size(),
                        (),
                        synthetic_phase_report(Some(0.75), Some(0.25)),
                    ))
                },
            )
            .expect("profiled RL step should succeed");
        });

        assert_eq!(
            events,
            vec![
                "push:rl_step",
                "push:self_play",
                "pop:self_play",
                "push:train",
                "pop:train",
                "pop:rl_step",
            ]
        );
    }

    #[test]
    fn finalize_rl_step_side_effects_writes_tensorboard_and_fallback_values_without_checkpoint() {
        let output_dir = unique_temp_dir("step-side-effects-tensorboard");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let mut config = helper_test_rl_config(output_dir.clone(), true);
        config.log_every_n_steps = 99;
        config.checkpoint_every_n_steps = 99;
        config.max_train_steps = Some(2);
        fs::create_dir_all(config.data_dir.clone()).expect("create RL data dir");
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, mut runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");
        let tb_session_dir = bootstrap.artifacts.tb_session_dir.clone();
        let step_log_path = bootstrap.artifacts.step_log_path.clone();
        let latest_state_path = bootstrap.artifacts.latest_state_path.clone();
        let latest_model_base = bootstrap.artifacts.latest_model_base.clone();
        let latest_optimizer_base = bootstrap.artifacts.latest_optimizer_base.clone();
        let mut rebase_tracker = RebaseTracker::default_phase2();

        runtime.head_controller.try_activate(AdvancedHead::DeltaQ);

        let should_stop = finalize_rl_step_side_effects(
            &mut runtime,
            &mut rebase_tracker,
            RlStepFinalizeContext {
                artifacts: &bootstrap.artifacts,
                config: &bootstrap.config,
                rl_config: &bootstrap.rl_config,
                current_runtime: bootstrap.current_runtime,
                session_start_global_step: bootstrap.session_start_global_step,
                total_steps: bootstrap.total_steps,
                batch_size: 7,
                report: synthetic_phase_report(None, None),
                profiling: None,
            },
        )
        .expect("tensorboard step side effects should succeed");

        assert!(!should_stop);
        assert_eq!(runtime.global_step, 1);
        assert_eq!(runtime.last_log_step, 0);
        assert_eq!(runtime.pipeline_state.total_games, 1);
        assert_eq!(runtime.pipeline_state.total_samples, 7);

        let entry = read_jsonl_entry(&step_log_path);
        assert_eq!(entry["loss"].as_f64(), Some(0.0));
        assert_eq!(entry["exit_weight"].as_f64(), Some(0.0));
        assert_eq!(entry["samples_in_batch"].as_u64(), Some(7));
        assert_eq!(entry["total_samples"].as_u64(), Some(7));

        assert!(!latest_state_path.exists());
        assert!(!latest_model_base.with_extension("mpk").exists());
        assert!(!latest_optimizer_base.with_extension("bin").exists());

        drop(runtime.tb.take());

        let tags = tensorboard_tags_from_dir(&tb_session_dir);
        assert!(tags.iter().any(|tag| tag == "rl/loss"));
        assert!(tags.iter().any(|tag| tag == "rl/exit_weight"));
        assert!(tags.iter().any(|tag| tag == "rl/delta_q_state"));
        assert!(tags.iter().any(|tag| tag == "rl/total_games"));
        assert!(tags.iter().any(|tag| tag == "rl/total_samples"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn rl_mode_summary_formats_zero_temperature_and_zero_steps() {
        let rl_config = RlTrainConfig {
            games_per_batch: 4,
            temperature: 0.0,
            phase: RlPhaseConfig::DrdaAchSelfPlay,
            learning_rate: None,
            exit_weight: None,
            aux_weight: None,
            microbatch_size: None,
        };

        let summary = rl_mode_summary(&rl_config, 0);

        assert_eq!(
            summary,
            "phase=DrdaAchSelfPlay games_per_batch=4 temperature=0.00 total_steps=0"
        );
    }

    #[test]
    fn wrapper_decision_helpers_match_zero_interval_session_boundaries() {
        assert!(should_emit_rl_progress(10, 10, 1));
        assert!(should_persist_rl_checkpoint(10, 10, 1));
        assert!(should_stop_rl_session(10, 10, Some(0)));
    }

    #[test]
    fn rl_wrapper_helpers_match_direct_session_decisions_for_non_boundary_steps() {
        let global_step = 23;
        let session_start = 20;

        assert_eq!(
            should_emit_rl_progress(global_step, session_start, 4),
            should_log_progress(global_step, session_start, 4)
        );
        assert_eq!(
            should_persist_rl_checkpoint(global_step, session_start, 6),
            should_save_checkpoint(global_step, session_start, 6)
        );
        assert_eq!(
            should_stop_rl_session(global_step, session_start, Some(4)),
            reached_session_step_budget(global_step, session_start, Some(4))
        );
    }

    #[test]
    fn run_profiled_rl_step_produces_correct_profiling_envelope_with_nvtx() {
        let (result, events) = nvtx::with_test_recorder(|| {
            super::run_profiled_rl_step(
                || Ok(42u64),
                |batch_val| {
                    assert_eq!(batch_val, 42);
                    Ok((
                        8usize,
                        "train_output",
                        PhaseTrainReport {
                            phase: TrainingPhase::DrdaAchSelfPlay,
                            skipped: false,
                            loss: Some(0.5),
                            effective_lr: 1e-4,
                            oracle_keep_prob: None,
                            kept_oracle_fraction: None,
                            exit_weight: Some(0.1),
                        },
                    ))
                },
            )
        });

        let (batch_size, output, report, profiling) =
            result.expect("profiled rl step should succeed");
        assert_eq!(batch_size, 8);
        assert_eq!(output, "train_output");
        assert_eq!(report.phase, TrainingPhase::DrdaAchSelfPlay);
        assert!(report.loss.is_some());

        assert_eq!(profiling.stage, PROFILING_STAGE_RL_STEP);
        assert!(profiling.elapsed_seconds > 0.0);
        assert_eq!(profiling.children.len(), 2);
        assert_eq!(profiling.children[0].stage, PROFILING_STAGE_SELF_PLAY);
        assert_eq!(profiling.children[1].stage, PROFILING_STAGE_TRAIN);
        assert!(profiling.children[0].elapsed_seconds >= 0.0);
        assert!(profiling.children[1].elapsed_seconds >= 0.0);

        assert!(events.contains(&"push:rl_step".to_string()));
        assert!(events.contains(&"pop:rl_step".to_string()));
        assert!(events.contains(&"push:self_play".to_string()));
        assert!(events.contains(&"pop:self_play".to_string()));
        assert!(events.contains(&"push:train".to_string()));
        assert!(events.contains(&"pop:train".to_string()));
    }
}
