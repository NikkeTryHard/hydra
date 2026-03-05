//! Continuous distillation: LearnerNet -> ActorNet (IMPALA-style).

use burn::prelude::*;
use burn::tensor::activation;

#[derive(Config, Debug)]
pub struct DistillConfig {
    #[config(default = "1.0")]
    pub kd_kl_weight: f32,
    #[config(default = "0.5")]
    pub kd_mse_weight: f32,
    #[config(default = "1e-4")]
    pub distill_lr: f64,
    #[config(default = "256")]
    pub distill_batch_size: usize,
    #[config(default = "60")]
    pub update_interval_secs: u64,
    #[config(default = "0.999")]
    pub ema_decay: f32,
}

impl DistillConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.distill_lr <= 0.0 {
            return Err("distill_lr must be positive");
        }
        if self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err("ema_decay must be in (0,1)");
        }
        Ok(())
    }
}

pub fn distill_loss<B: Backend>(
    learner_logits: Tensor<B, 2>,
    actor_logits: Tensor<B, 2>,
    learner_value: Tensor<B, 2>,
    actor_value: Tensor<B, 2>,
    kd_kl_weight: f32,
    kd_mse_weight: f32,
) -> Tensor<B, 1> {
    let teacher_pi = activation::softmax(learner_logits, 1);
    let student_log_pi = activation::log_softmax(actor_logits, 1);
    let kl = (teacher_pi.clone() * (teacher_pi.log() - student_log_pi))
        .sum_dim(1)
        .mean();

    let diff = learner_value - actor_value;
    let mse = (diff.clone() * diff).mean();

    kl * kd_kl_weight + mse * kd_mse_weight
}

pub struct DistillState {
    pub steps_since_update: u64,
    pub total_distill_steps: u64,
    pub last_kl_drift: f32,
}

impl DistillState {
    pub fn new() -> Self {
        Self {
            steps_since_update: 0,
            total_distill_steps: 0,
            last_kl_drift: 0.0,
        }
    }

    pub fn elapsed_steps(&self) -> u64 {
        self.total_distill_steps
    }

    pub fn should_distill(&self, config: &DistillConfig, elapsed_secs: u64) -> bool {
        elapsed_secs >= config.update_interval_secs
    }

    pub fn record_step(&mut self, kl_drift: f32) {
        self.total_distill_steps += 1;
        self.steps_since_update = 0;
        self.last_kl_drift = kl_drift;
    }
}

impl DistillState {
    pub fn is_healthy(&self, max_kl_drift: f32) -> bool {
        self.last_kl_drift <= max_kl_drift
    }
}

impl DistillConfig {
    pub fn summary(&self) -> String {
        format!(
            "distill(lr={:.1e}, ema={:.3}, interval={}s)",
            self.distill_lr, self.ema_decay, self.update_interval_secs
        )
    }
}

impl DistillState {
    pub fn summary(&self) -> String {
        format!(
            "distill_steps={} kl={:.4}",
            self.total_distill_steps, self.last_kl_drift
        )
    }
}

impl Default for DistillState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn distill_loss_zero_when_identical() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let value = Tensor::<B, 2>::from_floats([[0.5]], &device);
        let loss = distill_loss(logits.clone(), logits, value.clone(), value, 1.0, 0.5);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.abs() < 1e-5, "identical should give ~0 loss, got {val}");
    }

    #[test]
    fn distill_loss_positive_when_different() {
        let device = Default::default();
        let teacher = Tensor::<B, 2>::from_floats([[5.0, 1.0, 0.0]], &device);
        let student = Tensor::<B, 2>::from_floats([[0.0, 0.0, 5.0]], &device);
        let t_val = Tensor::<B, 2>::from_floats([[0.8]], &device);
        let s_val = Tensor::<B, 2>::from_floats([[-0.3]], &device);
        let loss = distill_loss(teacher, student, t_val, s_val, 1.0, 0.5);
        let val = loss.into_scalar().elem::<f32>();
        assert!(
            val > 0.1,
            "different outputs should give positive loss: {val}"
        );
    }
}
