use burn::prelude::*;
use burn::tensor::activation;

pub fn exit_loss<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    exit_mask: Tensor<B, 2>,
    weight: f32,
) -> Tensor<B, 1> {
    let neg_inf = (exit_mask.ones_like() - exit_mask) * (-1e9f32);
    let log_pi = activation::log_softmax(model_logits + neg_inf, 1);
    let ce = (exit_target * log_pi).sum_dim(1).neg().mean();
    ce * weight
}
