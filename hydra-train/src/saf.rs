//! Search-as-Feature (SaF) MLP adaptor.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

const SAF_INPUT_DIM: usize = 8;

#[derive(Config, Debug)]
pub struct SafConfig {
    #[config(default = "1.0")]
    pub alpha: f32,
    #[config(default = "0.3")]
    pub dropout: f32,
    #[config(default = "32")]
    pub hidden_dim: usize,
}

#[derive(Module, Debug)]
pub struct SafMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl SafConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SafMlp<B> {
        SafMlp {
            fc1: LinearConfig::new(SAF_INPUT_DIM, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> SafMlp<B> {
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::mish(self.fc1.forward(features));
        self.fc2.forward(h)
    }
}

pub fn apply_saf_logit<B: Backend>(
    base_logits: Tensor<B, 2>,
    saf_output: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 2> {
    base_logits + saf_output * mask * alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn saf_mlp_shape() {
        let device = Default::default();
        let mlp = SafConfig::new().init::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, SAF_INPUT_DIM], &device);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [4, 1]);
    }

    #[test]
    fn saf_logit_addition_correct() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let saf_out = Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 0.0]], &device);
        let result = apply_saf_logit(base, saf_out, mask, 2.0);
        let data = result.to_data();
        let vals = data.as_slice::<f32>().expect("f32");
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn saf_zero_mask_is_noop() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &device);
        let saf_out = Tensor::<B, 2>::from_floats([[10.0, 10.0]], &device);
        let mask = Tensor::<B, 2>::zeros([1, 2], &device);
        let result = apply_saf_logit(base.clone(), saf_out, mask, 1.0);
        let b = base.to_data().as_slice::<f32>().expect("f32").to_vec();
        let r = result.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert_eq!(b, r, "zero mask should produce identical logits");
    }
}
