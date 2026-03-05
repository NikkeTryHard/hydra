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
}
