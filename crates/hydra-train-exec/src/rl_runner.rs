//! RL training-loop execution helpers owned by the execution crate.
//!
//! The full RL loop still depends on lower APIs that have not all moved out of
//! the `hydra-train` facade: backend self-play batch production and the tensor
//! RL train-step wrapper. This module owns the backend-independent RL execution
//! seam now so the final loop move is a mechanical call retarget once those APIs
//! exist below the facade.

use std::time::Instant;

use colored::Colorize;
use hydra_search_labels::exit::ExitConfig;
use hydra_search_labels::live_exit::LiveExitConfig;
use hydra_train_algo::distill::{DistillConfig, DistillState};
use hydra_train_algo::drda::RebaseTracker;
use hydra_train_runtime::config::RlTrainConfig;
use hydra_train_runtime::head_gates::HeadState;
use hydra_train_runtime::nvtx;
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

use crate::artifacts::{RlArtifactPaths, append_rl_step_log_to_writer};
use crate::presentation::timestamped;
use crate::resume::RlRuntimeResumeContract;

/// Formats the RL runtime mode summary shown at training start.
#[must_use]
pub fn rl_mode_summary(rl_config: &RlTrainConfig, total_steps: usize) -> String {
    format!(
        "phase={:?} games_per_batch={} temperature={:.2} total_steps={}",
        rl_config.phase, rl_config.games_per_batch, rl_config.temperature, total_steps
    )
}

/// Computes the deterministic per-step base seed.
#[must_use]
pub fn base_seed_for_step(seed: u64, global_step: usize) -> u64 {
    seed.wrapping_add(global_step as u64 * 1_000_003)
}

/// Expands the per-step seed into deterministic per-game seeds.
#[must_use]
pub fn game_seeds_for_batch(base_seed: u64, games_per_batch: usize) -> Vec<u64> {
    (0..games_per_batch)
        .map(|idx| base_seed.wrapping_add(idx as u64))
        .collect()
}

/// TensorBoard scalar encoding for advanced-head state.
#[must_use]
pub fn head_state_scalar(head_state: HeadState) -> f32 {
    match head_state {
        HeadState::Off => 0.0,
        HeadState::Warmup => 1.0,
        HeadState::Active => 2.0,
    }
}

/// Computes the maintenance plan used by RL self-play before live ExIt labels.
#[must_use]
pub fn rl_maintenance_plan(
    state: &PipelineState,
    rebase_tracker: &RebaseTracker,
    distill_state: &DistillState,
    distill_cfg: &DistillConfig,
    elapsed_secs: u64,
    max_distill_kl_drift: f32,
) -> MaintenancePlan {
    maintenance_plan_from_inputs(OrchestratorPlanInputs {
        phase: state.phase,
        phase_progress: state.phase_progress(),
        should_advance_phase: state.should_advance_phase(),
        rebase_due: rebase_tracker.should_rebase(),
        distill_due: distill_state.should_distill(distill_cfg, elapsed_secs),
        distill_should_warn: distill_state.should_warn(max_distill_kl_drift),
    })
}

/// Converts the scalar maintenance plan into the live ExIt producer config.
#[must_use]
pub fn live_exit_config_from_plan(plan: &MaintenancePlan) -> LiveExitConfig {
    LiveExitConfig {
        enabled: plan.shallow_exit_enabled || plan.deep_exit_enabled,
        exit_config: ExitConfig::default_phase3(),
    }
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
#[must_use]
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
#[must_use]
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
#[must_use]
pub fn should_emit_rl_progress(
    global_step: usize,
    session_start_global_step: usize,
    log_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(log_every_n_steps)
}

/// Returns whether the current RL step should persist a checkpoint.
#[must_use]
pub fn should_persist_rl_checkpoint(
    global_step: usize,
    session_start_global_step: usize,
    checkpoint_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(checkpoint_every_n_steps)
}

/// Returns whether the bounded RL session has consumed its step budget.
#[must_use]
pub fn should_stop_rl_session(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> bool {
    reached_session_step_budget(global_step, session_start_global_step, max_train_steps)
}

/// Side-effect inputs for persisting one finalized RL step.
pub struct RlStepSideEffects<'a> {
    /// RL artifact paths for this run.
    pub artifacts: &'a RlArtifactPaths,
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
    /// Configured progress log cadence.
    pub log_every_n_steps: usize,
    /// Configured checkpoint cadence.
    pub checkpoint_every_n_steps: usize,
    /// Optional session-relative step budget.
    pub max_train_steps: Option<usize>,
}

/// Mutable scalar/logging state needed to finalize an RL step.
pub struct RlStepSideEffectState<'a, W> {
    /// Global optimizer step, incremented in-place.
    pub global_step: &'a mut usize,
    /// Pipeline scalar phase state, advanced in-place.
    pub pipeline_state: &'a mut PipelineState,
    /// RL step-log writer.
    pub step_log: &'a mut W,
    /// Last progress-log step, updated when emitted.
    pub last_log_step: &'a mut usize,
    /// Last progress-log wall clock, updated when emitted.
    pub last_log_time: &'a mut Instant,
}

/// Finalizes scalar side effects for one RL step without owning model/optimizer IO.
///
/// The full loop still performs checkpoint payload writes in the legacy runner
/// until the lower self-play and RL train-step APIs are available.
pub fn finalize_rl_step_scalar_side_effects<W>(
    state: RlStepSideEffectState<'_, W>,
    rebase_tracker: &mut RebaseTracker,
    context: RlStepSideEffects<'_>,
) -> Result<bool, String>
where
    W: std::io::Write,
{
    *state.global_step += 1;
    advance_rl_pipeline_state(
        state.pipeline_state,
        context.rl_config.games_per_batch,
        context.batch_size,
        context.total_steps,
    );
    rebase_tracker.tick(0.0);

    let delta_q_state = HeadState::Off;
    let loss_value = context.report.loss.unwrap_or(0.0);
    if should_emit_rl_progress(
        *state.global_step,
        context.session_start_global_step,
        context.log_every_n_steps,
    ) {
        println!(
            "{}",
            format_rl_progress_message(*state.global_step, loss_value, delta_q_state)
        );
        *state.last_log_step = *state.global_step;
        *state.last_log_time = Instant::now();
    }

    let step_entry = make_rl_step_entry(RlStepEntryInputs {
        global_step: *state.global_step,
        phase: &state.pipeline_state.phase,
        loss: loss_value,
        effective_lr: context.report.effective_lr,
        exit_weight: context.report.exit_weight.unwrap_or(0.0),
        games_per_batch: context.rl_config.games_per_batch,
        samples_in_batch: context.batch_size,
        total_games: state.pipeline_state.total_games,
        total_samples: state.pipeline_state.total_samples,
        delta_q_state,
        profiling: context.profiling,
    });
    append_rl_step_log_to_writer(state.step_log, &step_entry)?;

    Ok(should_stop_rl_session(
        *state.global_step,
        context.session_start_global_step,
        context.max_train_steps,
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

/// Builds a profiling envelope that appends logging/checkpoint child timings.
#[must_use]
pub fn merge_rl_logging_profile(
    mut profiling: ProfilingEnvelope,
    logging_seconds: f64,
    checkpoint_seconds: f64,
) -> ProfilingEnvelope {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_types::phase::TrainingPhase;

    #[test]
    fn rl_mode_summary_formats_phase_runtime_and_steps() {
        let rl_config = RlTrainConfig {
            games_per_batch: 16,
            temperature: 0.375,
            phase: hydra_train_runtime::config::RlPhaseConfig::ExitPondering,
            learning_rate: None,
            exit_weight: None,
            aux_weight: None,
            microbatch_size: None,
        };

        assert_eq!(
            rl_mode_summary(&rl_config, 2048),
            "phase=ExitPondering games_per_batch=16 temperature=0.38 total_steps=2048"
        );
    }

    #[test]
    fn deterministic_seed_helpers_wrap_and_count_up() {
        let seed = u64::MAX - 2;
        let base_seed = base_seed_for_step(seed, 3);

        assert_eq!(base_seed, seed.wrapping_add(3 * 1_000_003));
        assert_eq!(
            game_seeds_for_batch(u64::MAX - 1, 4),
            [u64::MAX - 1, u64::MAX, 0, 1]
        );
    }

    #[test]
    fn maintenance_plan_enables_live_exit_in_exit_phase() {
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

        let plan = rl_maintenance_plan(
            &state,
            &rebase,
            &distill,
            &DistillConfig::fast_distill(),
            30,
            0.05,
        );
        let live_exit_cfg = live_exit_config_from_plan(&plan);

        assert!(plan.should_rebase);
        assert!(plan.should_distill);
        assert!(plan.distill_warning);
        assert!(plan.shallow_exit_enabled);
        assert!(plan.deep_exit_enabled);
        assert!(live_exit_cfg.enabled);
    }

    #[test]
    fn pipeline_state_advance_updates_counts_version_and_budget() {
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
    fn profiled_step_records_self_play_and_train_children() {
        let report = PhaseTrainReport {
            phase: TrainingPhase::DrdaAchSelfPlay,
            skipped: false,
            loss: Some(0.25),
            effective_lr: 2.5e-4,
            oracle_keep_prob: None,
            kept_oracle_fraction: None,
            exit_weight: Some(0.5),
        };

        let (batch_size, value, observed, profiling) =
            run_profiled_rl_step(|| Ok(42u64), |batch| Ok((7, batch + 1, report)))
                .expect("profiled step should succeed");

        assert_eq!(batch_size, 7);
        assert_eq!(value, 43);
        assert_eq!(observed.loss, Some(0.25));
        assert_eq!(profiling.stage, PROFILING_STAGE_RL_STEP);
        assert_eq!(profiling.children.len(), 2);
        assert_eq!(profiling.children[0].stage, PROFILING_STAGE_SELF_PLAY);
        assert_eq!(profiling.children[1].stage, PROFILING_STAGE_TRAIN);
    }
}
