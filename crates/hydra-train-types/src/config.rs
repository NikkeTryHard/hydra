//! Backend-independent training configuration contracts.
//!
//! These DTOs live below the train facade so bin/runtime code can share epoch/RL
//! configuration without depending on `hydra-train`.

use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use hydra_model::model::HydraModelConfig;

/// Default RL physical microbatch size used when the train YAML omits one.
pub const DEFAULT_RL_MICROBATCH_SIZE: usize = 128;
/// Default ExIt loss weight for RL phases.
pub const DEFAULT_EXIT_WEIGHT: f32 = 0.5;
/// Default auxiliary-loss weight for RL phases.
pub const DEFAULT_AUX_WEIGHT: f32 = 0.1;
/// Generalized Advantage Estimation gamma used by RL self-play collation.
pub const GAE_GAMMA: f32 = 0.995;
/// Generalized Advantage Estimation lambda used by RL self-play collation.
pub const GAE_LAMBDA: f32 = 0.95;

#[derive(Config, Debug)]
pub struct AchConfig {
    #[config(default = "1.0")]
    pub eta: f32,
    #[config(default = "0.5")]
    pub eps: f32,
    #[config(default = "8.0")]
    pub l_th: f32,
    #[config(default = "5e-4")]
    pub beta_ent: f32,
}

impl AchConfig {
    pub fn summary(&self) -> String {
        format!(
            "ach(eta={:.1}, eps={:.1}, l_th={:.0}, ent={:.1e})",
            self.eta, self.eps, self.l_th, self.beta_ent
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.eta <= 0.0 {
            return Err("eta must be positive");
        }
        if self.eps <= 0.0 || self.eps >= 1.0 {
            return Err("eps must be in (0,1)");
        }
        if self.l_th <= 0.0 {
            return Err("l_th must be positive");
        }
        Ok(())
    }
}

pub const MIN_TAU_DRDA: f32 = 2.0;

/// Computes a cosine-annealed learning rate clamped to the configured floor.
pub fn cosine_annealing_lr(step: usize, total_steps: usize, lr_max: f64, lr_min: f64) -> f64 {
    if total_steps == 0 {
        return lr_max;
    }
    let t = (step as f64 / total_steps as f64).min(1.0);
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Computes a linear warmup followed by cosine annealing.
pub fn warmup_then_cosine_lr(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    lr_max: f64,
    lr_min: f64,
) -> f64 {
    if step < warmup_steps {
        lr_max * (step as f64 / warmup_steps as f64)
    } else {
        cosine_annealing_lr(
            step - warmup_steps,
            total_steps - warmup_steps,
            lr_max,
            lr_min,
        )
    }
}

/// Behavioral-cloning trainer hyperparameters and model shape.
#[derive(Config, Debug)]
pub struct BCTrainerConfig {
    pub model_config: HydraModelConfig,
    #[config(default = "2.5e-4")]
    pub lr: f64,
    #[config(default = "1e-6")]
    pub min_learning_rate: f64,
    #[config(default = "2048")]
    pub batch_size: usize,
    #[config(default = "1.0")]
    pub grad_clip_norm: f32,
    #[config(default = "1e-5")]
    pub weight_decay: f32,
    #[config(default = "1000")]
    pub warmup_steps: usize,
}

impl BCTrainerConfig {
    pub fn summary(&self) -> String {
        format!(
            "lr={:.1e} min_lr={:.1e} batch={} clip={:.1} wd={:.1e}",
            self.lr,
            self.min_learning_rate,
            self.batch_size,
            self.grad_clip_norm,
            self.weight_decay
        )
    }

    pub fn default_actor() -> Self {
        Self::new(HydraModelConfig::actor())
    }

    pub fn default_learner() -> Self {
        Self::new(HydraModelConfig::learner())
    }

    pub fn estimated_training_time_hours(&self, num_samples: usize, samples_per_sec: f32) -> f32 {
        if samples_per_sec <= 0.0 {
            return 0.0;
        }
        num_samples as f32 / samples_per_sec / 3600.0
    }

    pub fn num_epochs_for(&self, num_samples: usize) -> usize {
        let batches = self.total_batches(num_samples);
        if batches == 0 {
            return 0;
        }
        batches
    }

    pub fn is_warmup(&self, step: usize) -> bool {
        step < self.warmup_steps
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        if self.min_learning_rate <= 0.0 {
            return Err("min_learning_rate must be positive");
        }
        if self.min_learning_rate > self.lr {
            return Err("min_learning_rate must be <= lr");
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0");
        }
        if self.grad_clip_norm <= 0.0 {
            return Err("grad_clip_norm must be positive");
        }
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative");
        }
        if self.warmup_steps == 0 {
            return Err("warmup_steps must be > 0");
        }
        Ok(())
    }

    pub fn total_batches(&self, num_samples: usize) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        num_samples / self.batch_size
    }

    pub fn effective_lr(&self, step: usize, total_steps: usize) -> f64 {
        warmup_then_cosine_lr(
            step,
            self.warmup_steps,
            total_steps,
            self.lr,
            self.min_learning_rate,
        )
    }

    pub fn optimizer_config(&self) -> AdamConfig {
        AdamConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                self.weight_decay,
            )))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.grad_clip_norm)))
    }
}

/// RL step hyperparameters shared by train orchestration and runtime seams.
pub struct RlConfig {
    pub tau_drda: f32,
    pub ach_cfg: AchConfig,
    pub lr: f64,
    pub exit_weight: f32,
    pub aux_weight: f32,
    pub microbatch_size: Option<usize>,
}

impl RlConfig {
    pub fn default_phase2() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 2.5e-4,
            exit_weight: DEFAULT_EXIT_WEIGHT,
            aux_weight: DEFAULT_AUX_WEIGHT,
            microbatch_size: None,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_exit_weight(mut self, w: f32) -> Self {
        self.exit_weight = w;
        self
    }

    pub fn with_aux_weight(mut self, w: f32) -> Self {
        self.aux_weight = w;
        self
    }

    pub fn default_phase3() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 1e-4,
            exit_weight: DEFAULT_EXIT_WEIGHT,
            aux_weight: DEFAULT_AUX_WEIGHT,
            microbatch_size: None,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "rl(tau={:.1}, lr={:.1e}, exit_w={:.2}, aux_w={:.2})",
            self.tau_drda, self.lr, self.exit_weight, self.aux_weight
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tau_drda < MIN_TAU_DRDA {
            return Err("tau_drda below minimum");
        }
        self.ach_cfg.validate()?;
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        Ok(())
    }

    pub fn effective_exit_weight(&self, phase: u8, progress: f32) -> f32 {
        match phase {
            0 | 1 => 0.0,
            2 => {
                let progress = progress.clamp(0.0, 1.0);
                if progress <= 0.5 {
                    0.0
                } else {
                    self.exit_weight * ((progress - 0.5) / 0.5)
                }
            }
            _ => self.exit_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc_config_defaults_match_legacy_contract() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::actor());
        assert!((cfg.lr - 2.5e-4).abs() < 1e-10);
        assert!((cfg.min_learning_rate - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.batch_size, 2048);
        assert_eq!(cfg.warmup_steps, 1000);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rl_config_defaults_match_legacy_contract() {
        let phase2 = RlConfig::default_phase2();
        assert_eq!(phase2.tau_drda, 4.0);
        assert!((phase2.lr - 2.5e-4).abs() < f64::EPSILON);
        assert!((phase2.exit_weight - DEFAULT_EXIT_WEIGHT).abs() < f32::EPSILON);
        assert!((phase2.aux_weight - DEFAULT_AUX_WEIGHT).abs() < f32::EPSILON);
        assert!(phase2.validate().is_ok());

        let phase3 = RlConfig::default_phase3();
        assert!((phase3.lr - 1e-4).abs() < f64::EPSILON);
        assert!((phase3.exit_weight - DEFAULT_EXIT_WEIGHT).abs() < f32::EPSILON);
        assert!((phase3.aux_weight - DEFAULT_AUX_WEIGHT).abs() < f32::EPSILON);
        assert!(phase3.validate().is_ok());
    }

    #[test]
    fn ach_config_defaults_and_validation_match_algo_contract() {
        let cfg = AchConfig::new();
        assert!((cfg.eta - 1.0).abs() < 1e-6);
        assert!((cfg.eps - 0.5).abs() < 1e-6);
        assert!((cfg.l_th - 8.0).abs() < 1e-6);
        assert!((cfg.beta_ent - 5e-4).abs() < 1e-8);
        assert!(cfg.validate().is_ok());

        assert_eq!(
            AchConfig::new().with_eta(0.0).validate(),
            Err("eta must be positive")
        );
        assert_eq!(
            AchConfig::new().with_eps(1.0).validate(),
            Err("eps must be in (0,1)")
        );
        assert_eq!(
            AchConfig::new().with_l_th(0.0).validate(),
            Err("l_th must be positive")
        );
    }

    #[test]
    fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
        assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

        let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
        assert!((warmup_lr - 2.5e-4).abs() < 1e-12);

        let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
        let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
        assert!((post_warmup_lr - expected).abs() < 1e-12);
    }
}
