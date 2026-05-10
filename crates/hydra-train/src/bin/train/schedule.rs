use hydra_train_types::config::BCTrainerConfig;

#[cfg(test)]
pub(crate) use hydra_train_runtime::schedule::schedule_total_steps;
pub(crate) use hydra_train_runtime::schedule::{
    TrainerScheduleConfig, lr_status_message, steps_per_second,
};

pub(crate) fn effective_lr(train_cfg: &BCTrainerConfig, step: usize, total_steps: usize) -> f64 {
    hydra_train_runtime::schedule::effective_lr(
        TrainerScheduleConfig::new(
            train_cfg.lr,
            train_cfg.min_learning_rate,
            train_cfg.warmup_steps,
        ),
        step,
        total_steps,
    )
}
