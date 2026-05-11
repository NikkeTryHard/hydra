use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

#[test]
fn test_drda_defaults_match_roadmap() {
    let cfg = DrdaConfig::new();
    assert!((cfg.tau_drda - 4.0).abs() < 1e-6);
}

#[test]
fn test_drda_combined_logits() {
    let device = Default::default();
    let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let residual = Tensor::<B, 2>::from_floats([[4.0, 8.0, 12.0]], &device);
    let out = combined_logits(base, residual, 4.0);
    let data = out.to_data();
    let vals = data.as_slice::<f32>().expect("f32");
    assert!((vals[0] - 2.0).abs() < 1e-5);
    assert!((vals[1] - 4.0).abs() < 1e-5);
    assert!((vals[2] - 6.0).abs() < 1e-5);
}

#[test]
fn test_drda_rebase_preserves_pi() {
    let device = Default::default();
    let pi = Tensor::<B, 2>::from_floats([[0.2, 0.3, 0.5]], &device);
    let kl = verify_rebase_preserves_pi(pi.clone(), pi);
    assert!(kl.abs() < 1e-6, "KL should be ~0, got {kl}");
}

#[test]
fn test_drda_zero_residual_equals_base() {
    let device = Default::default();
    let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let zero = Tensor::<B, 2>::zeros([1, 3], &device);
    let out = combined_logits(base.clone(), zero, 4.0);
    let b_data = base.to_data();
    let o_data = out.to_data();
    let b = b_data.as_slice::<f32>().expect("f32");
    let o = o_data.as_slice::<f32>().expect("f32");
    for i in 0..3 {
        assert!(
            (b[i] - o[i]).abs() < 1e-6,
            "zero residual should equal base at {i}"
        );
    }
}

#[test]
fn test_compute_rebase_kl_zero_residual() {
    let device = Default::default();
    let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let zero_res = Tensor::<B, 2>::zeros([1, 3], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let kl = compute_rebase_kl(base, zero_res, 4.0, mask);
    assert!(kl.abs() < 1e-5, "zero residual should give KL~0: {kl}");
}

#[test]
fn test_compute_rebase_kl_nonzero_residual() {
    let device = Default::default();
    let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let res = Tensor::<B, 2>::from_floats([[5.0, -5.0, 0.0]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let kl = compute_rebase_kl(base, res, 4.0, mask);
    assert!(kl > 0.0, "non-zero residual should give positive KL: {kl}");
}

#[test]
fn test_drda_tau_below_minimum() {
    let cfg = DrdaConfig { tau_drda: 1.5 };
    let result = cfg.validate();
    assert!(result.is_err(), "tau_drda=1.5 should fail validation");
    assert_eq!(result.unwrap_err(), "tau_drda below minimum 2.0");
}

#[test]
fn test_drda_rebase_tracker_timing() {
    let mut tracker = RebaseTracker::new(37.5);
    assert!(!tracker.should_rebase(), "fresh tracker should not rebase");

    tracker.tick(38.0);
    assert!(
        tracker.should_rebase(),
        "after 38h with 37.5h interval, should_rebase must be true"
    );

    tracker.record_rebase();
    assert!(
        !tracker.should_rebase(),
        "after record_rebase, should_rebase must be false"
    );
    assert_eq!(tracker.total_rebases, 1);
}
