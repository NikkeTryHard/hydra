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
}
