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
mod tests;
