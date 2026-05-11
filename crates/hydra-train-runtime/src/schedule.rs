//! Runtime training schedule helpers.

use std::time::Duration;

use crate::config::TrainConfig;

/// Minimal trainer schedule inputs needed to compute BC learning rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainerScheduleConfig {
    /// Maximum learning rate after warmup.
    pub lr: f64,
    /// Cosine schedule floor.
    pub min_learning_rate: f64,
    /// Linear warmup steps.
    pub warmup_steps: usize,
}

impl TrainerScheduleConfig {
    /// Builds schedule inputs from scalar fields.
    pub fn new(lr: f64, min_learning_rate: f64, warmup_steps: usize) -> Self {
        Self {
            lr,
            min_learning_rate,
            warmup_steps,
        }
    }
}

impl<T> From<&T> for TrainerScheduleConfig
where
    T: AsRef<TrainerScheduleConfig>,
{
    fn from(config: &T) -> Self {
        *config.as_ref()
    }
}

/// Returns total global steps addressed by this training session.
pub fn schedule_total_steps(config: &TrainConfig, session_start_global_step: usize) -> usize {
    config
        .max_train_steps
        .map(|budget| session_start_global_step + budget)
        .unwrap_or(config.num_epochs.max(1))
        .max(1)
}

/// Formats the current learning-rate schedule state.
pub fn lr_status_message(step: usize, warmup_steps: usize, lr: f64) -> String {
    if warmup_steps > 0 && step < warmup_steps {
        format!("lr={lr:.2e} warmup {}/{}", step, warmup_steps)
    } else {
        format!("lr={lr:.2e} cosine")
    }
}

/// Computes the effective BC learning rate for a global step.
pub fn effective_lr(
    train_cfg: impl Into<TrainerScheduleConfig>,
    step: usize,
    total_steps: usize,
) -> f64 {
    let train_cfg = train_cfg.into();
    let warmup_steps = train_cfg.warmup_steps.min(total_steps);
    if step < warmup_steps {
        train_cfg.lr * (step as f64 / warmup_steps as f64)
    } else {
        cosine_annealing_lr(
            step - warmup_steps,
            total_steps - warmup_steps,
            train_cfg.lr,
            train_cfg.min_learning_rate,
        )
    }
}

fn cosine_annealing_lr(step: usize, total_steps: usize, lr_max: f64, lr_min: f64) -> f64 {
    if total_steps == 0 {
        return lr_max;
    }
    let t = (step as f64 / total_steps as f64).min(1.0);
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Computes a step rate from count and wall-clock duration.
pub fn steps_per_second(window_steps: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if window_steps == 0 || secs <= f64::EPSILON {
        0.0
    } else {
        window_steps as f64 / secs
    }
}

#[cfg(test)]
mod tests;
