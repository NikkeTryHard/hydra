//! Native GroupNorm+Mish fusion seam.
//!
//! This module is deliberately parity-first. The backend default keeps the
//! exact Burn GroupNorm and Mish formulas; accelerated backends can override the
//! narrow op after forward and gradient parity are proven.

use burn::nn::GroupNorm;
use burn::prelude::*;
use burn::tensor::TensorPrimitive;

/// GroupNorm followed by Mish for the residual pre-conv activation pair.
#[inline]
pub(crate) fn group_norm_mish<B: Backend, const D: usize>(
    group_norm: &GroupNorm<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    assert_eq!(
        input.shape()[1],
        group_norm.num_channels,
        "The number of channels in the input tensor should be equal to the number of channels in the GroupNorm module. Expected {}, got {}",
        group_norm.num_channels,
        input.shape()[1]
    );

    let gamma = group_norm.gamma.as_ref().map(|param| param.val());
    let beta = group_norm.beta.as_ref().map(|param| param.val());

    Tensor::from_primitive(TensorPrimitive::Float(B::group_norm_mish::<D>(
        input.into_primitive().tensor(),
        gamma.map(|tensor| tensor.into_primitive().tensor()),
        beta.map(|tensor| tensor.into_primitive().tensor()),
        group_norm.num_groups,
        group_norm.epsilon,
        group_norm.affine,
    )))
}
