use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

fn cfg() -> HeadsConfig {
    HeadsConfig::new()
}

#[test]
fn policy_head_shape() {
    let device = Default::default();
    let head = cfg().init_policy::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 46]);
}

#[test]
fn value_head_shape_and_range() {
    let device = Default::default();
    let head = cfg().init_value::<B>(&device);
    let x = Tensor::<B, 2>::random(
        [4, 256],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let out = head.forward(x);
    assert_eq!(out.dims(), [4, 1]);
    let data = out.to_data();
    for &v in data.as_slice::<f32>().expect("f32 slice") {
        assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
    }
}

#[test]
fn score_pdf_head_shape() {
    let device = Default::default();
    let head = cfg().init_score_pdf::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 64]);
}

#[test]
fn score_cdf_head_shape() {
    let device = Default::default();
    let head = cfg().init_score_cdf::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 64]);
}

#[test]
fn opp_tenpai_head_shape() {
    let device = Default::default();
    let head = cfg().init_opp_tenpai::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 3]);
}

#[test]
fn grp_head_shape() {
    let device = Default::default();
    let head = cfg().init_grp::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 24]);
}

#[test]
fn opp_next_discard_head_shape() {
    let device = Default::default();
    let head = cfg().init_opp_next_discard::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
    assert_eq!(head.forward(x).dims(), [4, 3, 34]);
}

#[test]
fn danger_head_shape() {
    let device = Default::default();
    let head = cfg().init_danger::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
    assert_eq!(head.forward(x).dims(), [4, 3, 34]);
}

#[test]
fn oracle_critic_head_shape() {
    let device = Default::default();
    let head = cfg().init_oracle_critic::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 4]);
}

#[test]
fn belief_field_head_shape() {
    let device = Default::default();
    let head = cfg().init_belief_field::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
    assert_eq!(head.forward(x).dims(), [4, 16, 34]);
}

#[test]
fn mixture_weight_head_shape() {
    let device = Default::default();
    let head = cfg().init_mixture_weight::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 4]);
}

#[test]
fn opponent_hand_type_head_shape() {
    let device = Default::default();
    let head = cfg().init_opponent_hand_type::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 24]);
}

#[test]
fn delta_q_head_shape() {
    let device = Default::default();
    let head = cfg().init_delta_q::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 46]);
}

#[test]
fn safety_residual_head_shape() {
    let device = Default::default();
    let head = cfg().init_safety_residual::<B>(&device);
    let x = Tensor::<B, 2>::zeros([4, 256], &device);
    assert_eq!(head.forward(x).dims(), [4, 46]);
}
