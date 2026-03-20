use std::time::Instant;

use colored::Colorize;

use hydra_train::selfplay::generate_self_play_rl_batch;
use hydra_train::training::distill::{DistillConfig, DistillState};
use hydra_train::training::drda::RebaseTracker;
use hydra_train::training::head_gates::{AdvancedHead, HeadState};
use hydra_train::training::orchestrator::{
    live_exit_config_from_plan, maintenance_plan, rl_phase_train_step_with_controller,
};

use super::artifacts::{append_rl_step_log, save_latest_rl_checkpoint_and_state};
use super::bootstrap::{RlTrainingBootstrap, RlTrainingRuntime};
use super::config::RlTrainConfig;
use super::presentation::{format_status_line, timestamped};
use super::progress::RlStepLogEntry;
use super::resume::build_rl_resume_state;
use super::status::{reached_session_step_budget, session_steps_completed};

fn rl_mode_summary(rl_config: &RlTrainConfig, total_steps: usize) -> String {
    format!(
        "phase={:?} games_per_batch={} temperature={:.2} total_steps={}",
        rl_config.phase, rl_config.games_per_batch, rl_config.temperature, total_steps
    )
}

fn base_seed_for_step(seed: u64, global_step: usize) -> u64 {
    seed.wrapping_add(global_step as u64 * 1_000_003)
}

fn game_seeds_for_batch(base_seed: u64, games_per_batch: usize) -> Vec<u64> {
    (0..games_per_batch)
        .map(|idx| base_seed.wrapping_add(idx as u64))
        .collect()
}

fn head_state_scalar(head_state: HeadState) -> f32 {
    match head_state {
        HeadState::Off => 0.0,
        HeadState::Warmup => 1.0,
        HeadState::Active => 2.0,
    }
}

fn should_log_progress(
    global_step: usize,
    session_start_global_step: usize,
    log_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(log_every_n_steps)
}

fn should_save_checkpoint(
    global_step: usize,
    session_start_global_step: usize,
    checkpoint_every_n_steps: usize,
) -> bool {
    session_steps_completed(global_step, session_start_global_step)
        .is_multiple_of(checkpoint_every_n_steps)
}

fn final_rl_summary(total_games: u64, total_samples: u64) -> String {
    format!(
        "{} {}",
        format!("games={total_games}").green(),
        format!("samples={total_samples}").green()
    )
}

fn advance_rl_pipeline_state(
    state: &mut hydra_train::config::PipelineState,
    games_per_batch: usize,
    batch_samples: usize,
    total_steps: usize,
) {
    state.total_games += games_per_batch as u64;
    state.total_samples += batch_samples as u64;
    state.increment_learner_version();
    state.tick_gpu_hours(state.phase.gpu_hours_budget() as f32 / total_steps as f32);
}

fn make_rl_step_entry(
    global_step: usize,
    phase: &hydra_train::config::TrainingPhase,
    loss: f64,
    effective_lr: f64,
    exit_weight: f32,
    games_per_batch: usize,
    samples_in_batch: usize,
    total_games: u64,
    total_samples: u64,
    delta_q_state: HeadState,
) -> RlStepLogEntry {
    RlStepLogEntry {
        global_step,
        phase: format!("{:?}", phase),
        loss,
        effective_lr,
        exit_weight,
        games_per_batch,
        samples_in_batch,
        total_games,
        total_samples,
        delta_q_state: format!("{:?}", delta_q_state),
    }
}

fn format_rl_progress_message(
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

fn should_emit_rl_progress(
    global_step: usize,
    session_start_global_step: usize,
    log_every_n_steps: usize,
) -> bool {
    should_log_progress(global_step, session_start_global_step, log_every_n_steps)
}

fn should_persist_rl_checkpoint(
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

fn should_stop_rl_session(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> bool {
    reached_session_step_budget(global_step, session_start_global_step, max_train_steps)
}

pub(super) fn run_rl_training_loop(
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

    while runtime.global_step < total_steps {
        let elapsed_secs = runtime.run_start.elapsed().as_secs();
        let plan = maintenance_plan(
            &runtime.pipeline_state,
            &rebase_tracker,
            &distill_state,
            &distill_cfg,
            elapsed_secs,
            0.05,
        );
        let live_exit_cfg = live_exit_config_from_plan(&plan);
        let base_seed = base_seed_for_step(config.seed, runtime.global_step);
        let game_seeds = game_seeds_for_batch(base_seed, rl_config.games_per_batch);
        let batch = generate_self_play_rl_batch(
            &game_seeds,
            rl_config.temperature,
            base_seed,
            &runtime.model,
            &train_device,
            &gae_config,
            live_exit_cfg,
        );

        runtime.head_controller.try_activate(AdvancedHead::DeltaQ);
        let (model, report) = rl_phase_train_step_with_controller(
            &runtime.pipeline_state,
            runtime.model,
            &batch,
            &rl_step_cfg,
            &loss_fn,
            &mut runtime.optimizer,
            Some(&mut runtime.head_controller),
        )
        .map_err(|err| format!("rl phase train step failed: {err}"))?;
        runtime.model = model;
        runtime.head_controller.tick_warmup();

        runtime.global_step += 1;
        advance_rl_pipeline_state(
            &mut runtime.pipeline_state,
            rl_config.games_per_batch,
            batch.batch_size(),
            total_steps,
        );
        rebase_tracker.tick(runtime.run_start.elapsed().as_secs_f32());

        let delta_q_state = runtime.head_controller.head_state(AdvancedHead::DeltaQ);
        let loss_value = report.loss.unwrap_or(0.0);
        let step_entry = make_rl_step_entry(
            runtime.global_step,
            &runtime.pipeline_state.phase,
            loss_value,
            report.effective_lr,
            report.exit_weight.unwrap_or(0.0),
            rl_config.games_per_batch,
            batch.batch_size(),
            runtime.pipeline_state.total_games,
            runtime.pipeline_state.total_samples,
            delta_q_state,
        );
        append_rl_step_log(&artifacts.step_log_path, &step_entry)?;

        if let Some(ref mut tb) = runtime.tb.as_mut() {
            let step = runtime.global_step as i64;
            tb.write_scalar(step, "rl/loss", loss_value as f32)
                .map_err(|err| format!("tensorboard write rl/loss failed: {err}"))?;
            tb.write_scalar(step, "rl/exit_weight", report.exit_weight.unwrap_or(0.0))
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
            session_start_global_step,
            config.log_every_n_steps,
        ) {
            println!(
                "{}",
                format_rl_progress_message(runtime.global_step, loss_value, delta_q_state)
            );
            runtime.last_log_step = runtime.global_step;
            runtime.last_log_time = Instant::now();
        }

        if should_persist_rl_checkpoint(
            runtime.global_step,
            session_start_global_step,
            config.checkpoint_every_n_steps,
        ) {
            let state =
                build_rl_resume_state(runtime.global_step, runtime.pipeline_state, current_runtime);
            save_latest_rl_checkpoint_and_state(
                &artifacts,
                &runtime.model,
                &runtime.optimizer,
                runtime.global_step,
                loss_value,
                &state,
            )?;
        }

        if should_stop_rl_session(
            runtime.global_step,
            session_start_global_step,
            config.max_train_steps,
        ) {
            break;
        }
    }

    let final_state =
        build_rl_resume_state(runtime.global_step, runtime.pipeline_state, current_runtime);
    save_latest_rl_checkpoint_and_state(
        &artifacts,
        &runtime.model,
        &runtime.optimizer,
        runtime.global_step,
        0.0,
        &final_state,
    )?;

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
    use crate::config::RlPhaseConfig;
    use hydra_train::config::{PipelineState, TrainingPhase};

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
    fn session_progress_helpers_handle_steps_before_session_start() {
        assert!(should_log_progress(8, 10, 2));
        assert!(should_save_checkpoint(8, 10, 5));
        assert_eq!(session_steps_completed(8, 10), 0);
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
        let plan = maintenance_plan(
            &state,
            &RebaseTracker::default_phase2(),
            &DistillState::default(),
            &DistillConfig::fast_distill(),
            29,
            0.05,
        );

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
        let plan = maintenance_plan(
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
    fn maintenance_plan_stays_idle_in_oracle_guiding_phase() {
        let mut rebase = RebaseTracker::default_phase2();
        rebase.tick(100.0);
        let state = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: TrainingPhase::OracleGuiding.cumulative_budget_before() as f32 + 150.0,
            ..PipelineState::default()
        };
        let plan = maintenance_plan(
            &state,
            &rebase,
            &DistillState {
                last_kl_drift: 0.2,
                ..DistillState::default()
            },
            &DistillConfig::fast_distill(),
            999,
            0.05,
        );

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
    fn rl_step_entry_and_progress_message_format_runtime_values() {
        let entry = make_rl_step_entry(
            12,
            &TrainingPhase::ExitPondering,
            0.25,
            5e-4,
            0.1,
            8,
            64,
            128,
            1024,
            HeadState::Warmup,
        );

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

        let message = format_rl_progress_message(12, 0.25, HeadState::Warmup);
        assert!(message.contains("RL step"));
        assert!(message.contains("global_step=12"));
        assert!(message.contains("loss=0.2500"));
        assert!(message.contains("delta_q=Warmup"));
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
}
