use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

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
