//! DRDA wrapper: Dilated Regularized Dual Averaging (Farina et al., ICLR 2025).

use burn::prelude::*;
use burn::tensor::activation;
pub use hydra_train_types::config::MIN_TAU_DRDA;

#[derive(Config, Debug)]
pub struct DrdaConfig {
    #[config(default = "4.0")]
    pub tau_drda: f32,
}

pub const MIN_REBASE_INTERVAL_HOURS: f32 = 25.0;

impl DrdaConfig {
    pub fn summary(&self) -> String {
        format!("drda(tau={:.1})", self.tau_drda)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tau_drda < MIN_TAU_DRDA {
            return Err("tau_drda below minimum 2.0");
        }
        Ok(())
    }
}

pub struct RebaseTracker {
    pub gpu_hours_since_rebase: f32,
    pub rebase_interval_hours: f32,
    pub total_rebases: u32,
}

impl RebaseTracker {
    pub fn default_phase2() -> Self {
        Self::new(37.5)
    }

    pub fn new(interval_hours: f32) -> Self {
        Self {
            gpu_hours_since_rebase: 0.0,
            rebase_interval_hours: interval_hours.max(MIN_REBASE_INTERVAL_HOURS),
            total_rebases: 0,
        }
    }

    pub fn progress(&self) -> f32 {
        if self.rebase_interval_hours <= 0.0 {
            return 0.0;
        }
        (self.gpu_hours_since_rebase / self.rebase_interval_hours).min(1.0)
    }

    pub fn hours_until_next(&self) -> f32 {
        (self.rebase_interval_hours - self.gpu_hours_since_rebase).max(0.0)
    }

    pub fn is_overdue(&self, factor: f32) -> bool {
        self.gpu_hours_since_rebase >= self.rebase_interval_hours * factor
    }

    pub fn should_rebase(&self) -> bool {
        self.gpu_hours_since_rebase >= self.rebase_interval_hours
    }

    pub fn record_rebase(&mut self) {
        self.total_rebases += 1;
        self.gpu_hours_since_rebase = 0.0;
    }

    pub fn tick(&mut self, hours: f32) {
        self.gpu_hours_since_rebase += hours;
    }
}

pub fn combined_logits<B: Backend>(
    base_logits: Tensor<B, 2>,
    residual_logits: Tensor<B, 2>,
    tau_drda: f32,
) -> Tensor<B, 2> {
    base_logits + residual_logits / tau_drda
}

pub fn verify_rebase_preserves_pi<B: Backend>(
    pi_before: Tensor<B, 2>,
    pi_after: Tensor<B, 2>,
) -> f32 {
    let eps = 1e-8f32;
    let p = pi_before.clamp(eps, 1.0);
    let q = pi_after.clamp(eps, 1.0);
    let log_ratio = (p.clone() / q).log();
    let kl = (p * log_ratio).sum_dim(1).mean();
    kl.into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("kl scalar should be readable as f32")[0]
}

pub fn compute_rebase_kl<B: Backend>(
    base_logits: Tensor<B, 2>,
    residual_logits: Tensor<B, 2>,
    tau_drda: f32,
    legal_mask: Tensor<B, 2>,
) -> f32 {
    let combined = combined_logits(base_logits.clone(), residual_logits, tau_drda);
    let neg_inf = (legal_mask.clone().ones_like() - legal_mask) * (-1e9f32);
    let pi_before = activation::softmax(combined + neg_inf.clone(), 1);
    let pi_after = activation::softmax(base_logits + neg_inf, 1);
    verify_rebase_preserves_pi(pi_before, pi_after)
}

#[cfg(test)]
mod tests;
