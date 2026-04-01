use std::time::Duration;

use hydra_train::training::bc::{BCTrainerConfig, warmup_then_cosine_lr};

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
        train_cfg.min_learning_rate,
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

#[cfg(test)]
mod tests {
    use super::super::config::{BcHyperparamConfig, TrainConfig};
    use super::*;
    use std::path::PathBuf;

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/data"),
            output_dir: PathBuf::from("/output"),
            num_epochs: 3,
            batch_size: 256,
            microbatch_size: None,
            validation_microbatch_size: None,
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: hydra_train::preflight::PreflightConfig::default(),
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    #[test]
    fn schedule_total_steps_prefers_budget_over_epoch_count() {
        let mut config = dummy_config();
        assert_eq!(schedule_total_steps(&config, 5), 3);

        config.max_train_steps = Some(12);
        assert_eq!(schedule_total_steps(&config, 5), 17);
    }

    #[test]
    fn lr_status_message_reports_warmup_and_cosine_modes() {
        assert_eq!(lr_status_message(2, 10, 1e-4), "lr=1.00e-4 warmup 2/10");
        assert_eq!(lr_status_message(10, 10, 1e-4), "lr=1.00e-4 cosine");
    }

    #[test]
    fn effective_lr_and_steps_per_second_handle_boundaries() {
        let cfg = BCTrainerConfig::new(hydra_train::model::HydraModelConfig::learner())
            .with_lr(2.5e-4)
            .with_min_learning_rate(1e-6)
            .with_warmup_steps(4);

        let start_lr = effective_lr(&cfg, 0, 20);
        let mid_lr = effective_lr(&cfg, 2, 20);
        let end_lr = effective_lr(&cfg, 20, 20);
        assert!(start_lr <= mid_lr);
        assert!(end_lr >= 1e-6);

        assert_eq!(steps_per_second(0, Duration::from_secs(2)), 0.0);
        assert_eq!(steps_per_second(10, Duration::from_secs(0)), 0.0);
        assert!((steps_per_second(25, Duration::from_secs(5)) - 5.0).abs() < 1e-12);
    }
}
