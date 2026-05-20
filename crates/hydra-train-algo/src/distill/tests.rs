use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

#[test]
fn distill_loss_zero_when_identical() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let value = Tensor::<B, 2>::from_floats([[0.5]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let loss = distill_loss(logits.clone(), logits, value.clone(), value, mask, 1.0, 0.5);
    let val = loss.into_scalar().elem::<f32>();
    assert!(val.abs() < 1e-5, "identical should give ~0 loss, got {val}");
}

#[test]
fn distill_loss_with_partial_mask_no_nan() {
    let device = Default::default();
    let teacher = Tensor::<B, 2>::from_floats([[2.0, -1.0, 3.0]], &device);
    let student = Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
    let t_val = Tensor::<B, 2>::from_floats([[0.5]], &device);
    let s_val = Tensor::<B, 2>::from_floats([[0.3]], &device);
    let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0]], &device);
    let loss = distill_loss(teacher, student, t_val, s_val, mask, 1.0, 0.5);
    let val = loss.into_scalar().elem::<f32>();
    assert!(
        val.is_finite(),
        "partial mask (1 of 3 legal) should not produce NaN/Inf: {val}"
    );
}

#[test]
fn distill_loss_positive_when_different() {
    let device = Default::default();
    let teacher = Tensor::<B, 2>::from_floats([[5.0, 1.0, 0.0]], &device);
    let student = Tensor::<B, 2>::from_floats([[0.0, 0.0, 5.0]], &device);
    let t_val = Tensor::<B, 2>::from_floats([[0.8]], &device);
    let s_val = Tensor::<B, 2>::from_floats([[-0.3]], &device);
    let mask = Tensor::<B, 2>::ones([1, 3], &device);
    let loss = distill_loss(teacher, student, t_val, s_val, mask, 1.0, 0.5);
    let val = loss.into_scalar().elem::<f32>();
    assert!(
        val > 0.1,
        "different outputs should give positive loss: {val}"
    );
}

#[test]
fn fast_distill_uses_faster_update_schedule() {
    let config = DistillConfig::fast_distill();

    assert_eq!(config.update_interval_secs, 30);
    assert!((config.ema_decay - 0.995).abs() < 1e-6);
    assert!((config.kd_kl_weight - 1.0).abs() < 1e-6);
    assert!((config.kd_mse_weight - 0.5).abs() < 1e-6);
}

#[test]
fn validate_rejects_bad_learning_rate_and_ema_decay() {
    let bad_lr = DistillConfig::new().with_distill_lr(0.0);
    assert_eq!(bad_lr.validate(), Err("distill_lr must be positive"));

    let bad_zero_decay = DistillConfig::new().with_ema_decay(0.0);
    assert_eq!(bad_zero_decay.validate(), Err("ema_decay must be in (0,1)"));

    let bad_one_decay = DistillConfig::new().with_ema_decay(1.0);
    assert_eq!(bad_one_decay.validate(), Err("ema_decay must be in (0,1)"));
}

#[test]
#[should_panic(expected = "kd_kl_weight must be non-negative")]
fn distill_loss_rejects_negative_kl_weight() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::zeros([1, 2], &device);
    let value = Tensor::<B, 2>::zeros([1, 1], &device);
    let mask = Tensor::<B, 2>::ones([1, 2], &device);
    let _ = distill_loss(
        logits.clone(),
        logits,
        value.clone(),
        value,
        mask,
        -1.0,
        0.5,
    );
}

#[test]
#[should_panic(expected = "kd_mse_weight must be finite")]
fn distill_loss_rejects_nan_mse_weight() {
    let device = Default::default();
    let logits = Tensor::<B, 2>::zeros([1, 2], &device);
    let value = Tensor::<B, 2>::zeros([1, 1], &device);
    let mask = Tensor::<B, 2>::ones([1, 2], &device);
    let _ = distill_loss(
        logits.clone(),
        logits,
        value.clone(),
        value,
        mask,
        1.0,
        f32::NAN,
    );
}

#[test]
fn distill_state_tracks_ticks_recording_and_health() {
    let config = DistillConfig::new().with_update_interval_secs(5);
    let mut state = DistillState::new();

    assert_eq!(state.elapsed_steps(), 0);
    assert!(!state.should_distill(&config, 4));
    assert!(state.should_distill(&config, 5));

    state.tick();
    state.tick();
    assert_eq!(state.steps_since_update, 2);

    state.record_step(0.25);
    assert_eq!(state.elapsed_steps(), 1);
    assert_eq!(state.steps_since_update, 0);
    assert!((state.last_kl_drift - 0.25).abs() < 1e-6);
    assert!(state.is_healthy(0.3));
    assert!(!state.is_healthy(0.2));
    assert!(state.should_warn(0.2));
    assert!(!state.should_warn(0.3));
}

#[test]
fn summaries_include_key_runtime_fields() {
    let config = DistillConfig::new();
    let mut state = DistillState::default();
    state.record_step(0.125);

    let config_summary = config.summary();
    let state_summary = state.summary();

    assert!(config_summary.contains("distill(lr="));
    assert!(config_summary.contains("interval=60s"));
    assert!(state_summary.contains("distill_steps=1"));
    assert!(state_summary.contains("kl=0.1250"));
}
