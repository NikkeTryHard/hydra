use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

#[test]
fn test_policy_ce_with_mask() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0, -1.0]], &device);
    let mut mask_data = [1.0f32; 4];
    mask_data[3] = 0.0;
    let mask = Tensor::<B, 2>::from_floats([mask_data], &device);
    let target = Tensor::<B, 2>::from_floats([[0.0, 0.0, 1.0, 0.0]], &device);
    let loss = policy_ce(logits, target, mask);
    let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
    assert!(val > 0.0, "policy CE should be positive, got {val}");
    assert!(val < 5.0, "policy CE too large: {val}");
}

#[test]
fn test_policy_ce_illegal_action_zero_gradient() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[10.0, -10.0, 0.0]], &device);
    let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
    let target = Tensor::<B, 2>::from_floats([[0.5, 0.0, 0.5]], &device);
    let loss = policy_ce(logits.clone(), target, mask);
    let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
    assert!(val.is_finite(), "masked loss should be finite: {val}");
}

#[test]
fn test_soft_target_differs_from_hard() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 0.5]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let hard = Tensor::<B, 2>::from_floats([[0.0, 1.0, 0.0]], &device);
    let soft = Tensor::<B, 2>::from_floats([[0.3, 0.7, 0.0]], &device);
    let l_hard = policy_ce(logits.clone(), hard, mask.clone());
    let l_soft = policy_ce(logits, soft, mask);
    let h = l_hard.to_data().as_slice::<f32>().expect("f32")[0];
    let s = l_soft.to_data().as_slice::<f32>().expect("f32")[0];
    assert!(
        (h - s).abs() > 0.01,
        "soft vs hard should differ: {h} vs {s}"
    );
}

#[test]
fn test_oracle_critic_zero_sum() {
    let device = Default::default();
    let v = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
    let target = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
    let loss = oracle_critic_loss(v, target);
    let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
    assert!(
        val.abs() < 1e-4,
        "zero-sum input should give near-zero loss, got {val}"
    );
}

#[test]
fn test_oracle_target_zero_sum() {
    let target = oracle_target_from_scores([30000, 25000, 25000, 20000]);
    let sum: f32 = target.iter().sum();
    assert!(sum.abs() < 1e-5, "oracle target should be zero-sum: {sum}");
    assert!(target[0] > 0.0, "1st place should be positive");
    assert!(target[3] < 0.0, "4th place should be negative");
}

#[test]
fn test_focal_bce_vs_standard_bce() {
    let device = Default::default();
    let logits = Tensor::<B, 3>::from_floats([[[3.0; 34]; 3]], &device);
    let target = Tensor::<B, 3>::ones([1, 3, 34], &device);
    let mask = Tensor::<B, 3>::ones([1, 3, 34], &device);
    let focal = danger_focal_bce(logits.clone(), target.clone(), mask.clone());
    let standard = bce_with_logits_3d(logits, target);
    let standard_sum = (standard * mask)
        .sum_dim(2)
        .sum_dim(1)
        .squeeze_dim::<2>(2)
        .squeeze_dim::<1>(1);
    let f = focal.into_scalar().elem::<f32>();
    let s = standard_sum.into_scalar().elem::<f32>();
    assert!(
        f < s,
        "focal ({f}) should be < standard ({s}) for high-confidence correct"
    );
}

#[test]
fn test_compute_cvar() {
    let pdf = [0.1f32, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1];
    let cvar = compute_cvar(&pdf, 0.3);
    assert!(cvar >= 0.0 && cvar.is_finite(), "CVaR: {cvar}");
    let cvar_full = compute_cvar(&pdf, 1.0);
    assert!(cvar <= cvar_full, "CVaR(0.3) <= CVaR(1.0)");
}

#[test]
fn test_bce_extreme_logits() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[100.0, -100.0]], &device);
    let target = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
    let loss = bce_with_logits(logits, target);
    let data = loss.to_data();
    for &v in data.as_slice::<f32>().expect("f32") {
        assert!(v.is_finite(), "extreme logits should give finite BCE: {v}");
    }
}

#[test]
fn test_policy_ce_single_legal_action() {
    let device = Default::default();
    let mut mask_data = [0.0f32; 46];
    mask_data[5] = 1.0;
    let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device).reshape([1, 46]);
    let target = mask.clone();
    let logits = Tensor::<B, 2>::zeros([1, 46], &device);
    let loss = policy_ce(logits, target, mask);
    let v: f32 = loss.into_scalar().elem();
    assert!(v < 0.01, "single legal action loss should be ~0, got {v}");
}

#[test]
fn test_value_mse_extreme_values() {
    let device = Default::default();
    let pred = Tensor::<B, 1>::from_floats([0.99, -0.99], &device);
    let target = Tensor::<B, 1>::from_floats([1.0, -1.0], &device);
    let loss = value_mse(pred, target);
    let data = loss.to_data();
    for &v in data.as_slice::<f32>().expect("f32") {
        assert!(v.is_finite(), "extreme value MSE should be finite, got {v}");
        assert!(v < 0.01, "near-boundary MSE should be small, got {v}");
    }
}

#[test]
fn test_oracle_target_from_scores_zero_sum() {
    let target = oracle_target_from_scores([25000, 25000, 25000, 25000]);
    for (i, &v) in target.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "equal scores should give zero delta, player {i} got {v}"
        );
    }
}

#[test]
fn test_kl_divergence_identical_distributions() {
    let device = Default::default();
    let p = Tensor::<B, 2>::from_floats([[0.3, 0.5, 0.2]], &device);
    let kl = kl_divergence(p.clone(), p);
    let v: f32 = kl.into_scalar().elem();
    assert!(v.abs() < 1e-6, "KL(p, p) should be ~0, got {v}");
}
