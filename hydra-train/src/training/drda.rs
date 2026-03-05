//! DRDA wrapper: Dilated Regularized Dual Averaging (Farina et al., ICLR 2025).

use burn::prelude::*;

#[derive(Config, Debug)]
pub struct DrdaConfig {
    #[config(default = "4.0")]
    pub tau_drda: f32,
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
    let log_ratio = (pi_before.clone() / (pi_after + eps)).log();
    let kl = (pi_before * log_ratio).sum_dim(1).mean();
    kl.into_scalar().elem::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

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
}
