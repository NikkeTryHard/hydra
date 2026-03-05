//! DRDA wrapper: Dilated Regularized Dual Averaging (Farina et al., ICLR 2025).

use burn::prelude::*;
use burn::tensor::activation;

#[derive(Config, Debug)]
pub struct DrdaConfig {
    #[config(default = "4.0")]
    pub tau_drda: f32,
}

pub const MIN_TAU_DRDA: f32 = 2.0;

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
            rebase_interval_hours: interval_hours,
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

    pub fn summary(&self) -> String {
        format!(
            "rebases={} hours_since={:.1}",
            self.total_rebases, self.gpu_hours_since_rebase
        )
    }

    pub fn tick(&mut self, hours: f32) {
        self.gpu_hours_since_rebase += hours;
    }
}

type BaseLogitsFn<B> = Box<dyn Fn(Tensor<B, 3>) -> Tensor<B, 2>>;

pub struct DrdaWrapper<B: Backend> {
    pub base_logits_fn: Option<BaseLogitsFn<B>>,
    pub tau_drda: f32,
}

impl<B: Backend> DrdaWrapper<B> {
    pub fn new(tau_drda: f32) -> Self {
        Self {
            base_logits_fn: None,
            tau_drda: tau_drda.max(MIN_TAU_DRDA),
        }
    }

    pub fn combined_logits(
        &self,
        base_logits: Tensor<B, 2>,
        residual_logits: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        base_logits + residual_logits / self.tau_drda
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
    kl.into_scalar().elem::<f32>()
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

pub fn policy_head_is_zeroed<B: Backend>(logits: Tensor<B, 2>) -> bool {
    let max_abs: f32 = logits.abs().max().into_scalar().elem();
    max_abs < 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_drda_defaults_match_roadmap() {
        let cfg = DrdaConfig::new();
        assert!((cfg.tau_drda - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_drda_combined_logits() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let residual = Tensor::<B, 2>::from_floats([[4.0, 8.0, 12.0]], &device);
        let out = combined_logits(base, residual, 4.0);
        let data = out.to_data();
        let vals = data.as_slice::<f32>().expect("f32");
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
        assert!((vals[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_drda_rebase_preserves_pi() {
        let device = Default::default();
        let pi = Tensor::<B, 2>::from_floats([[0.2, 0.3, 0.5]], &device);
        let kl = verify_rebase_preserves_pi(pi.clone(), pi);
        assert!(kl.abs() < 1e-6, "KL should be ~0, got {kl}");
    }

    #[test]
    fn test_drda_zero_residual_equals_base() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let zero = Tensor::<B, 2>::zeros([1, 3], &device);
        let out = combined_logits(base.clone(), zero, 4.0);
        let b_data = base.to_data();
        let o_data = out.to_data();
        let b = b_data.as_slice::<f32>().expect("f32");
        let o = o_data.as_slice::<f32>().expect("f32");
        for i in 0..3 {
            assert!(
                (b[i] - o[i]).abs() < 1e-6,
                "zero residual should equal base at {i}"
            );
        }
    }

    #[test]
    fn test_drda_wrapper_method() {
        let device = Default::default();
        let wrapper = DrdaWrapper::<B>::new(4.0);
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &device);
        let res = Tensor::<B, 2>::from_floats([[8.0, 4.0]], &device);
        let out = wrapper.combined_logits(base, res);
        let data = out.to_data();
        let vals = data.as_slice::<f32>().expect("f32");
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_rebase_kl_zero_residual() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let zero_res = Tensor::<B, 2>::zeros([1, 3], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let kl = compute_rebase_kl(base, zero_res, 4.0, mask);
        assert!(kl.abs() < 1e-5, "zero residual should give KL~0: {kl}");
    }

    #[test]
    fn test_compute_rebase_kl_nonzero_residual() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let res = Tensor::<B, 2>::from_floats([[5.0, -5.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let kl = compute_rebase_kl(base, res, 4.0, mask);
        assert!(kl > 0.0, "non-zero residual should give positive KL: {kl}");
    }

    #[test]
    fn test_drda_tau_below_minimum() {
        let cfg = DrdaConfig { tau_drda: 1.5 };
        let result = cfg.validate();
        assert!(result.is_err(), "tau_drda=1.5 should fail validation");
        assert_eq!(result.unwrap_err(), "tau_drda below minimum 2.0");
    }

    #[test]
    fn test_drda_rebase_tracker_timing() {
        let mut tracker = RebaseTracker::new(37.5);
        assert!(!tracker.should_rebase(), "fresh tracker should not rebase");

        tracker.tick(38.0);
        assert!(
            tracker.should_rebase(),
            "after 38h with 37.5h interval, should_rebase must be true"
        );

        tracker.record_rebase();
        assert!(
            !tracker.should_rebase(),
            "after record_rebase, should_rebase must be false"
        );
        assert_eq!(tracker.total_rebases, 1);
    }
}
