use std::time::Duration;

use hydra_train::training::bc::{warmup_then_cosine_lr, BCTrainerConfig};

use super::config::TrainConfig;

pub(super) fn schedule_total_steps(
    config: &TrainConfig,
    session_start_global_step: usize,
) -> usize {
    config
        .max_train_steps
        .map(|budget| session_start_global_step + budget)
        .unwrap_or(config.num_epochs.max(1))
        .max(1)
}

pub(super) fn lr_status_message(step: usize, warmup_steps: usize, lr: f64) -> String {
    if warmup_steps > 0 && step < warmup_steps {
        format!("lr={lr:.2e} warmup {}/{}", step, warmup_steps)
    } else {
        format!("lr={lr:.2e} cosine")
    }
}

pub(super) fn effective_lr(train_cfg: &BCTrainerConfig, step: usize, total_steps: usize) -> f64 {
    warmup_then_cosine_lr(
        step,
        train_cfg.warmup_steps.min(total_steps),
        total_steps,
        train_cfg.lr,
        1e-6,
    )
}

pub(super) fn steps_per_second(window_steps: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if window_steps == 0 || secs <= f64::EPSILON {
        0.0
    } else {
        window_steps as f64 / secs
    }
}
