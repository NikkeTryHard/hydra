use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;
type AD = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

fn assert_close<const D: usize>(actual: Tensor<B, D>, expected: Tensor<B, D>, label: &str) {
    let diff = (actual - expected).abs().max().into_scalar().elem::<f32>();
    assert!(diff <= 1e-4, "{label} max abs diff {diff}");
}

#[cfg(feature = "libtorch-tests")]
fn assert_close_libtorch<const D: usize>(
    actual: Tensor<burn::backend::LibTorch<f32>, D>,
    expected: Tensor<burn::backend::LibTorch<f32>, D>,
    label: &str,
) {
    let diff = (actual - expected).abs().max().into_scalar().elem::<f32>();
    assert!(diff <= 1e-4, "{label} max abs diff {diff}");
}

#[cfg(feature = "libtorch-tests")]
fn libtorch_device() -> burn::backend::libtorch::LibTorchDevice {
    if tch::Cuda::is_available() {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    } else {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }
}

#[test]
fn se_block_preserves_shape() {
    let device = Default::default();
    let se = SEBlockConfig::new(256, 64).init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
    let out = se.forward(x);
    assert_eq!(out.dims(), [4, 256, 34]);
}

#[test]
fn se_res_block_preserves_shape() {
    let device = Default::default();
    let block = SEResBlockConfig::new(256, 32, 64).init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
    let out = block.forward(x);
    assert_eq!(out.dims(), [4, 256, 34]);
}

#[test]
fn backbone_output_shapes_12_blocks() {
    let device = Default::default();
    let cfg = SEResNetConfig::new(12, hydra_core::encoder::NUM_CHANNELS, 256, 32, 64);
    let net = cfg.init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, hydra_core::encoder::NUM_CHANNELS, 34], &device);
    let (spatial, pooled) = net.forward(x);
    assert_eq!(spatial.dims(), [4, 256, 34]);
    assert_eq!(pooled.dims(), [4, 256]);
}

#[test]
fn backbone_output_shapes_24_blocks() {
    let device = Default::default();
    let cfg = SEResNetConfig::new(24, hydra_core::encoder::NUM_CHANNELS, 256, 32, 64);
    let net = cfg.init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([2, hydra_core::encoder::NUM_CHANNELS, 34], &device);
    let (spatial, pooled) = net.forward(x);
    assert_eq!(spatial.dims(), [2, 256, 34]);
    assert_eq!(pooled.dims(), [2, 256]);
}

#[test]
fn residual_block_output_differs_from_input() {
    let device = Default::default();
    let block = SEResBlockConfig::new(256, 32, 64).init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [1, 256, 34],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    let out = block.forward(x.clone());
    let diff = (out - x).abs().mean();
    let d = diff.into_scalar().elem::<f32>();
    assert!(
        d > 1e-6,
        "residual output should differ from input: diff={d}"
    );
}

#[test]
fn se_block_channel_attention() {
    let device = Default::default();
    let se = SEBlockConfig::new(4, 2).init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [1, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let out = se.forward(x.clone());
    assert_eq!(out.dims(), [1, 4, 8]);
    let x_data = x.to_data();
    let o_data = out.to_data();
    let xv = x_data.as_slice::<f32>().expect("f32");
    let ov = o_data.as_slice::<f32>().expect("f32");
    let mut any_diff = false;
    for i in 0..32 {
        if (xv[i] - ov[i]).abs() > 1e-6 {
            any_diff = true;
        }
    }
    assert!(any_diff, "SE should modulate channels");
}

#[test]
fn se_block_output_bounded_by_input() {
    let device = Default::default();
    let se = SEBlockConfig::new(4, 2).init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [2, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 2.0),
        &device,
    );
    let out = se.forward(x.clone());
    let x_abs = x.abs().max();
    let o_abs = out.abs().max();
    let x_max: f32 = x_abs.into_scalar().elem();
    let o_max: f32 = o_abs.into_scalar().elem();
    assert!(
        o_max <= x_max + 1e-4,
        "SE output max ({o_max}) should be <= input max ({x_max}) due to sigmoid gate"
    );
}

#[test]
fn group_norm_mish_seam_matches_oracle() {
    let device = Default::default();
    let gn = GroupNormConfig::new(2, 4).init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [3, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let expected = burn::tensor::activation::mish(gn.forward(x.clone()));
    let actual = crate::native_group_norm_mish::group_norm_mish(&gn, x);

    assert!(actual.all_close(expected, Some(1e-5), Some(1e-5)));
}

#[test]
fn residual_block_native_seam_preserves_shape() {
    let device = Default::default();
    let block = SEResBlockConfig::new(4, 2, 2).init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [2, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );

    let out = block.forward(x);

    assert_eq!(out.dims(), [2, 4, 8]);
}

#[test]
fn group_norm_mish_autodiff_backward_matches_oracle_affine() {
    let device = Default::default();
    let gn = GroupNormConfig::new(2, 4).init::<AD>(&device);
    let x_data = [
        0.10, -0.20, 0.30, 0.40, -0.50, 0.60, -0.70, 0.80, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65,
        0.75, -0.85, -0.11, 0.21, -0.31, 0.41, -0.51, 0.61, -0.71, 0.81,
    ];
    let w_data = [
        1.0, 0.7, -0.3, 0.2, -0.5, 0.9, 1.1, -0.8, 0.6, -0.4, 0.3, -0.2, 0.5, -0.7, 0.8, -0.9,
        -1.0, 0.4, -0.6, 0.2, 0.7, -0.1, 0.5, -0.3,
    ];
    let x_explicit =
        Tensor::<AD, 3>::from_data(TensorData::new(x_data.to_vec(), [3, 4, 2]), &device)
            .require_grad();
    let x_oracle = Tensor::<AD, 3>::from_data(TensorData::new(x_data.to_vec(), [3, 4, 2]), &device)
        .require_grad();
    let weights = Tensor::<AD, 3>::from_data(TensorData::new(w_data.to_vec(), [3, 4, 2]), &device);

    let explicit = crate::native_group_norm_mish::group_norm_mish(&gn, x_explicit.clone());
    let oracle = burn::tensor::activation::mish(gn.forward(x_oracle.clone()));
    let explicit_loss = (explicit * weights.clone()).sum();
    let oracle_loss = (oracle * weights).sum();
    let explicit_grads = explicit_loss.backward();
    let oracle_grads = oracle_loss.backward();

    assert_close(
        x_explicit.grad(&explicit_grads).expect("explicit x grad"),
        x_oracle.grad(&oracle_grads).expect("oracle x grad"),
        "x grad",
    );
    assert_close(
        gn.gamma
            .as_ref()
            .unwrap()
            .val()
            .grad(&explicit_grads)
            .expect("explicit gamma grad"),
        gn.gamma
            .as_ref()
            .unwrap()
            .val()
            .grad(&oracle_grads)
            .expect("oracle gamma grad"),
        "gamma grad",
    );
    assert_close(
        gn.beta
            .as_ref()
            .unwrap()
            .val()
            .grad(&explicit_grads)
            .expect("explicit beta grad"),
        gn.beta
            .as_ref()
            .unwrap()
            .val()
            .grad(&oracle_grads)
            .expect("oracle beta grad"),
        "beta grad",
    );
}

#[test]
fn group_norm_mish_autodiff_backward_matches_oracle_no_affine() {
    let device = Default::default();
    let gn = GroupNormConfig::new(2, 4)
        .with_affine(false)
        .init::<AD>(&device);
    let x_data = [
        0.20, -0.10, 0.50, -0.30, 0.70, -0.40, 0.90, -0.60, -0.25, 0.45, -0.65, 0.85, -1.05, 1.25,
        -1.45, 1.65,
    ];
    let w_data = [
        0.3, -0.8, 1.2, -0.4, 0.9, -1.1, 0.6, -0.2, -0.5, 0.7, -0.9, 1.1, -1.3, 1.5, -1.7, 1.9,
    ];
    let x_explicit =
        Tensor::<AD, 3>::from_data(TensorData::new(x_data.to_vec(), [2, 4, 2]), &device)
            .require_grad();
    let x_oracle = Tensor::<AD, 3>::from_data(TensorData::new(x_data.to_vec(), [2, 4, 2]), &device)
        .require_grad();
    let weights = Tensor::<AD, 3>::from_data(TensorData::new(w_data.to_vec(), [2, 4, 2]), &device);

    let explicit = crate::native_group_norm_mish::group_norm_mish(&gn, x_explicit.clone());
    let oracle = burn::tensor::activation::mish(gn.forward(x_oracle.clone()));
    let explicit_loss = (explicit * weights.clone()).sum();
    let oracle_loss = (oracle * weights).sum();
    let explicit_grads = explicit_loss.backward();
    let oracle_grads = oracle_loss.backward();

    assert_close(
        x_explicit.grad(&explicit_grads).expect("explicit x grad"),
        x_oracle.grad(&oracle_grads).expect("oracle x grad"),
        "x grad no affine",
    );
}

#[cfg(feature = "libtorch-tests")]
#[test]
fn group_norm_mish_libtorch_forward_backward_matches_oracle_affine() {
    type TchB = burn::backend::LibTorch<f32>;

    let device = libtorch_device();
    let gn = GroupNormConfig::new(2, 4).init::<burn::backend::Autodiff<TchB>>(&device);
    let x_data = [
        0.10, -0.20, 0.30, 0.40, -0.50, 0.60, -0.70, 0.80, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65,
        0.75, -0.85, -0.11, 0.21, -0.31, 0.41, -0.51, 0.61, -0.71, 0.81,
    ];
    let w_data = [
        1.0, 0.7, -0.3, 0.2, -0.5, 0.9, 1.1, -0.8, 0.6, -0.4, 0.3, -0.2, 0.5, -0.7, 0.8, -0.9,
        -1.0, 0.4, -0.6, 0.2, 0.7, -0.1, 0.5, -0.3,
    ];
    let x_explicit = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(x_data.to_vec(), [3, 4, 2]),
        &device,
    )
    .require_grad();
    let x_oracle = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(x_data.to_vec(), [3, 4, 2]),
        &device,
    )
    .require_grad();
    let weights = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(w_data.to_vec(), [3, 4, 2]),
        &device,
    );

    let explicit = crate::native_group_norm_mish::group_norm_mish(&gn, x_explicit.clone());
    let oracle = burn::tensor::activation::mish(gn.forward(x_oracle.clone()));
    let explicit_loss = (explicit.clone() * weights.clone()).sum();
    let oracle_loss = (oracle.clone() * weights).sum();
    let explicit_grads = explicit_loss.backward();
    let oracle_grads = oracle_loss.backward();

    assert_close_libtorch(explicit.inner(), oracle.inner(), "forward");
    assert_close_libtorch(
        x_explicit.grad(&explicit_grads).expect("explicit x grad"),
        x_oracle.grad(&oracle_grads).expect("oracle x grad"),
        "x grad",
    );
    assert_close_libtorch(
        gn.gamma
            .as_ref()
            .unwrap()
            .val()
            .grad(&explicit_grads)
            .expect("explicit gamma grad"),
        gn.gamma
            .as_ref()
            .unwrap()
            .val()
            .grad(&oracle_grads)
            .expect("oracle gamma grad"),
        "gamma grad",
    );
    assert_close_libtorch(
        gn.beta
            .as_ref()
            .unwrap()
            .val()
            .grad(&explicit_grads)
            .expect("explicit beta grad"),
        gn.beta
            .as_ref()
            .unwrap()
            .val()
            .grad(&oracle_grads)
            .expect("oracle beta grad"),
        "beta grad",
    );
}

#[cfg(feature = "libtorch-tests")]
#[test]
fn group_norm_mish_libtorch_forward_backward_matches_oracle_no_affine() {
    type TchB = burn::backend::LibTorch<f32>;

    let device = libtorch_device();
    let gn = GroupNormConfig::new(2, 4)
        .with_affine(false)
        .init::<burn::backend::Autodiff<TchB>>(&device);
    let x_data = [
        0.20, -0.10, 0.50, -0.30, 0.70, -0.40, 0.90, -0.60, -0.25, 0.45, -0.65, 0.85, -1.05, 1.25,
        -1.45, 1.65,
    ];
    let w_data = [
        0.3, -0.8, 1.2, -0.4, 0.9, -1.1, 0.6, -0.2, -0.5, 0.7, -0.9, 1.1, -1.3, 1.5, -1.7, 1.9,
    ];
    let x_explicit = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(x_data.to_vec(), [2, 4, 2]),
        &device,
    )
    .require_grad();
    let x_oracle = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(x_data.to_vec(), [2, 4, 2]),
        &device,
    )
    .require_grad();
    let weights = Tensor::<burn::backend::Autodiff<TchB>, 3>::from_data(
        TensorData::new(w_data.to_vec(), [2, 4, 2]),
        &device,
    );

    let explicit = crate::native_group_norm_mish::group_norm_mish(&gn, x_explicit.clone());
    let oracle = burn::tensor::activation::mish(gn.forward(x_oracle.clone()));
    let explicit_loss = (explicit.clone() * weights.clone()).sum();
    let oracle_loss = (oracle.clone() * weights).sum();
    let explicit_grads = explicit_loss.backward();
    let oracle_grads = oracle_loss.backward();

    assert_close_libtorch(explicit.inner(), oracle.inner(), "forward no affine");
    assert_close_libtorch(
        x_explicit.grad(&explicit_grads).expect("explicit x grad"),
        x_oracle.grad(&oracle_grads).expect("oracle x grad"),
        "x grad no affine",
    );
}
