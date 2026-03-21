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
    pub fn fast_distill() -> Self {
        Self::new()
            .with_update_interval_secs(30)
            .with_ema_decay(0.995)
    }

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
    legal_mask: Tensor<B, 2>,
    kd_kl_weight: f32,
    kd_mse_weight: f32,
) -> Tensor<B, 1> {
    let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
    let teacher_pi = activation::softmax(learner_logits + neg_inf.clone(), 1);
    let student_log_pi = activation::log_softmax(actor_logits + neg_inf, 1);
    let teacher_log_pi = teacher_pi.clone().clamp(1e-8, 1.0).log();
    let kl = (teacher_pi * (teacher_log_pi - student_log_pi))
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

impl DistillState {
    pub fn should_warn(&self, max_kl: f32) -> bool {
        self.last_kl_drift > max_kl
    }

    pub fn tick(&mut self) {
        self.steps_since_update += 1;
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
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let loss = distill_loss(logits.clone(), logits, value.clone(), value, mask, 1.0, 0.5);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.abs() < 1e-5, "identical should give ~0 loss, got {val}");
    }

    #[test]
    fn distill_loss_with_partial_mask_no_nan() {
        let device = Default::default();
        let teacher = Tensor::<B, 2>::from_floats([[2.0, -1.0, 3.0]], &device);
        let student = Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let t_val = Tensor::<B, 2>::from_floats([[0.5]], &device);
        let s_val = Tensor::<B, 2>::from_floats([[0.3]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0]], &device);
        let loss = distill_loss(teacher, student, t_val, s_val, mask, 1.0, 0.5);
        let val = loss.into_scalar().elem::<f32>();
        assert!(
            val.is_finite(),
            "partial mask (1 of 3 legal) should not produce NaN/Inf: {val}"
        );
    }

    #[test]
    fn distill_loss_positive_when_different() {
        let device = Default::default();
        let teacher = Tensor::<B, 2>::from_floats([[5.0, 1.0, 0.0]], &device);
        let student = Tensor::<B, 2>::from_floats([[0.0, 0.0, 5.0]], &device);
        let t_val = Tensor::<B, 2>::from_floats([[0.8]], &device);
        let s_val = Tensor::<B, 2>::from_floats([[-0.3]], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let loss = distill_loss(teacher, student, t_val, s_val, mask, 1.0, 0.5);
        let val = loss.into_scalar().elem::<f32>();
        assert!(
            val > 0.1,
            "different outputs should give positive loss: {val}"
        );
    }

    #[test]
    fn fast_distill_uses_faster_update_schedule() {
        let config = DistillConfig::fast_distill();

        assert_eq!(config.update_interval_secs, 30);
        assert!((config.ema_decay - 0.995).abs() < 1e-6);
        assert!((config.kd_kl_weight - 1.0).abs() < 1e-6);
        assert!((config.kd_mse_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn validate_rejects_bad_learning_rate_and_ema_decay() {
        let bad_lr = DistillConfig::new().with_distill_lr(0.0);
        assert_eq!(bad_lr.validate(), Err("distill_lr must be positive"));

        let bad_zero_decay = DistillConfig::new().with_ema_decay(0.0);
        assert_eq!(bad_zero_decay.validate(), Err("ema_decay must be in (0,1)"));

        let bad_one_decay = DistillConfig::new().with_ema_decay(1.0);
        assert_eq!(bad_one_decay.validate(), Err("ema_decay must be in (0,1)"));
    }

    #[test]
    fn distill_state_tracks_ticks_recording_and_health() {
        let config = DistillConfig::new().with_update_interval_secs(5);
        let mut state = DistillState::new();

        assert_eq!(state.elapsed_steps(), 0);
        assert!(!state.should_distill(&config, 4));
        assert!(state.should_distill(&config, 5));

        state.tick();
        state.tick();
        assert_eq!(state.steps_since_update, 2);

        state.record_step(0.25);
        assert_eq!(state.elapsed_steps(), 1);
        assert_eq!(state.steps_since_update, 0);
        assert!((state.last_kl_drift - 0.25).abs() < 1e-6);
        assert!(state.is_healthy(0.3));
        assert!(!state.is_healthy(0.2));
        assert!(state.should_warn(0.2));
        assert!(!state.should_warn(0.3));
    }

    #[test]
    fn summaries_include_key_runtime_fields() {
        let config = DistillConfig::new();
        let mut state = DistillState::default();
        state.record_step(0.125);

        let config_summary = config.summary();
        let state_summary = state.summary();

        assert!(config_summary.contains("distill(lr="));
        assert!(config_summary.contains("interval=60s"));
        assert!(state_summary.contains("distill_steps=1"));
        assert!(state_summary.contains("kl=0.1250"));
    }
}
