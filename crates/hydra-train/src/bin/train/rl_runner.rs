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
use super::presentation::{format_status_line, timestamped};
use super::progress::RlStepLogEntry;
use super::resume::build_rl_resume_state;
use super::status::{reached_session_step_budget, session_steps_completed};
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
        format_status_line(
            "RL mode:",
            format!(
                "phase={:?} games_per_batch={} temperature={:.2} total_steps={}",
                rl_config.phase, rl_config.games_per_batch, rl_config.temperature, total_steps
            )
        )
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
        let base_seed = config
            .seed
            .wrapping_add(runtime.global_step as u64 * 1_000_003);
        let game_seeds: Vec<u64> = (0..rl_config.games_per_batch)
            .map(|idx| base_seed.wrapping_add(idx as u64))
            .collect();
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
        runtime.pipeline_state.total_games += rl_config.games_per_batch as u64;
        runtime.pipeline_state.total_samples += batch.batch_size() as u64;
        runtime.pipeline_state.increment_learner_version();
        runtime.pipeline_state.tick_gpu_hours(
            runtime.pipeline_state.phase.gpu_hours_budget() as f32 / total_steps as f32,
        );
        rebase_tracker.tick(runtime.run_start.elapsed().as_secs_f32());

        let delta_q_state = runtime.head_controller.head_state(AdvancedHead::DeltaQ);
        let loss_value = report.loss.unwrap_or(0.0);
        let step_entry = RlStepLogEntry {
            global_step: runtime.global_step,
            phase: format!("{:?}", runtime.pipeline_state.phase),
            loss: loss_value,
            effective_lr: report.effective_lr,
            exit_weight: report.exit_weight.unwrap_or(0.0),
            games_per_batch: rl_config.games_per_batch,
            samples_in_batch: batch.batch_size(),
            total_games: runtime.pipeline_state.total_games,
            total_samples: runtime.pipeline_state.total_samples,
            delta_q_state: format!("{:?}", delta_q_state),
        };
        append_rl_step_log(&artifacts.step_log_path, &step_entry)?;

        if let Some(ref mut tb) = runtime.tb.as_mut() {
            let step = runtime.global_step as i64;
            tb.write_scalar(step, "rl/loss", loss_value as f32)
                .map_err(|err| format!("tensorboard write rl/loss failed: {err}"))?;
            tb.write_scalar(step, "rl/exit_weight", report.exit_weight.unwrap_or(0.0))
                .map_err(|err| format!("tensorboard write rl/exit_weight failed: {err}"))?;
            tb.write_scalar(
                step,
                "rl/delta_q_state",
                match delta_q_state {
                    HeadState::Off => 0.0,
                    HeadState::Warmup => 1.0,
                    HeadState::Active => 2.0,
                },
            )
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

        if session_steps_completed(runtime.global_step, session_start_global_step)
            .is_multiple_of(config.log_every_n_steps)
        {
            println!(
                "{}",
                timestamped(format!(
                    "{} {} {} {}",
                    "RL step".bold().cyan(),
                    format!("global_step={}", runtime.global_step).yellow(),
                    format!("loss={loss_value:.4}").green(),
                    format!("delta_q={delta_q_state:?}").magenta(),
                ))
            );
            runtime.last_log_step = runtime.global_step;
            runtime.last_log_time = Instant::now();
        }

        if session_steps_completed(runtime.global_step, session_start_global_step)
            .is_multiple_of(config.checkpoint_every_n_steps)
        {
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

        if reached_session_step_budget(
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
            "{} {} {}",
            "Finished RL training.".bold().cyan(),
            format!("games={}", runtime.pipeline_state.total_games).green(),
            format!("samples={}", runtime.pipeline_state.total_samples).green(),
        ))
    );
    Ok(())
}
