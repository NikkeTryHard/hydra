//! Native GroupNorm+Mish fusion seam.
//! This module keeps the GroupNorm+Mish call site narrow. It uses upstream Burn
//! ops directly so the model crate does not depend on a vendored Burn fork.

use burn::nn::GroupNorm;
use burn::prelude::*;

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

    let output = group_norm.forward(input);
    burn::tensor::activation::mish(output)
}
