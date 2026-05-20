use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;
type AchInputs = (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 1, Int>,
    Tensor<B, 1>,
    Tensor<B, 1>,
);

fn make_ach_inputs(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> AchInputs {
    let logits = Tensor::<B, 2>::from_floats([[0.0, 1.0, -1.0]], device);
    let mask = Tensor::<B, 2>::ones([1, 3], device);
    let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], device);
    let pi_old = Tensor::<B, 1>::from_floats([0.5], device);
    let advantages = Tensor::<B, 1>::from_floats([1.0], device);
    (logits, mask, actions, pi_old, advantages)
}

#[test]
fn test_ach_defaults_match_roadmap() {
    let cfg = AchConfig::new();
    assert!((cfg.eta - 1.0).abs() < 1e-6);
    assert!((cfg.eps - 0.5).abs() < 1e-6);
    assert!((cfg.l_th - 8.0).abs() < 1e-6);
    assert!((cfg.beta_ent - 5e-4).abs() < 1e-8);
}

#[test]
fn test_ach_gate_positive_adv() {
    let device = Default::default();
    let (logits, mask, actions, pi_old, advantages) = make_ach_inputs(&device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.is_finite(), "ACH loss should be finite: {val}");
}

#[test]
fn test_ach_gate_clips_ratio() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[0.0, 5.0, -5.0]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
    let pi_old = Tensor::<B, 1>::from_floats([0.01], &device);
    let adv = Tensor::<B, 1>::from_floats([1.0], &device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.is_finite());
}

#[test]
fn test_ach_negative_adv() {
    let device = Default::default();
    let (logits, mask, actions, pi_old, _) = make_ach_inputs(&device);
    let neg_adv = Tensor::<B, 1>::from_floats([-1.0], &device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, neg_adv, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.is_finite());
}

#[test]
fn test_ach_gate_clips_logit() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[0.0, 20.0, -20.0]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
    let pi_old = Tensor::<B, 1>::from_floats([0.5], &device);
    let adv = Tensor::<B, 1>::from_floats([1.0], &device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.is_finite(), "clipped logit should produce finite loss");
}

#[test]
fn ach_loss_zero_pi_old_no_nan() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[0.0, 1.0, -1.0]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
    let pi_old = Tensor::<B, 1>::from_floats([0.0], &device);
    let advantages = Tensor::<B, 1>::from_floats([1.0], &device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(
        val.is_finite(),
        "pi_old=0 should not produce NaN/Inf: {val}"
    );
}

#[test]
fn test_ach_batch_of_8() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::random(
        [8, 46],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let mask = Tensor::<B, 2>::ones([8, 46], &device);
    let actions = Tensor::<B, 1, Int>::from_ints(&[0i32, 1, 2, 3, 4, 5, 6, 7][..], &device);
    let pi_old = Tensor::<B, 1>::from_floats([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2], &device);
    let adv = Tensor::<B, 1>::from_floats([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.1], &device);
    let cfg = AchConfig::new();
    let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.is_finite(), "batch ACH should be finite: {val}");
}

#[test]
#[should_panic(expected = "l_th must be positive")]
fn ach_loss_rejects_nonpositive_logit_threshold() {
    let device = Default::default();
    let (logits, mask, actions, pi_old, advantages) = make_ach_inputs(&device);
    let cfg = AchConfig::new().with_l_th(0.0);
    let _ = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
}

#[test]
#[should_panic(expected = "eps must be finite")]
fn ach_loss_rejects_nonfinite_eps() {
    let device = Default::default();
    let (logits, mask, actions, pi_old, advantages) = make_ach_inputs(&device);
    let cfg = AchConfig::new().with_eps(f32::NAN);
    let _ = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
}
