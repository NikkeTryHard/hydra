use super::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

fn onehot2d<B: Backend>(
    device: &B::Device,
    batch: usize,
    classes: usize,
    idx: usize,
) -> Tensor<B, 2> {
    let mut d = vec![0.0f32; batch * classes];
    for i in 0..batch {
        d[i * classes + idx] = 1.0;
    }
    Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, classes])
}

fn onehot3d<B: Backend>(device: &B::Device, batch: usize, c1: usize, c2: usize) -> Tensor<B, 3> {
    let mut d = vec![0.0f32; batch * c1 * c2];
    for i in 0..(batch * c1) {
        d[i * c2] = 1.0;
    }
    Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, c1, c2])
}

fn make_dummy_targets<B: Backend>(device: &B::Device, batch: usize) -> HydraTargets<B> {
    HydraTargets {
        policy_target: onehot2d(device, batch, 46, 0),
        legal_mask: Tensor::ones([batch, 46], device),
        value_target: Tensor::zeros([batch], device),
        grp_target: onehot2d(device, batch, 24, 0),
        tenpai_target: Tensor::zeros([batch, 3], device),
        danger_target: Tensor::zeros([batch, 3, 34], device),
        danger_mask: Tensor::ones([batch, 3, 34], device),
        opp_next_target: onehot3d(device, batch, 3, 34),
        score_pdf_target: onehot2d(device, batch, 64, 32),
        score_cdf_target: Tensor::zeros([batch, 64], device),
        oracle_target: None,
        belief_fields_target: None,
        belief_fields_mask: None,
        mixture_weight_target: None,
        mixture_weight_mask: None,
        opponent_hand_type_target: None,
        delta_q_target: None,
        delta_q_mask: None,
        safety_residual_target: None,
        safety_residual_mask: None,
        oracle_guidance_mask: None,
        target_presence: None,
    }
}

#[test]
fn default_weights_match_roadmap() {
    let cfg = HydraLossConfig::new();
    assert!((cfg.w_pi - 1.0).abs() < 1e-6);
    assert!((cfg.w_v - 0.5).abs() < 1e-6);
    assert!((cfg.w_grp - 0.2).abs() < 1e-6);
    assert!((cfg.w_tenpai - 0.1).abs() < 1e-6);
    assert!((cfg.w_danger - 0.1).abs() < 1e-6);
    assert!((cfg.w_opp - 0.1).abs() < 1e-6);
    assert!((cfg.w_score - 0.025).abs() < 1e-6);
    assert!((cfg.w_oracle_critic - 0.0).abs() < 1e-6);
    assert!((cfg.w_belief_fields - 0.0).abs() < 1e-6);
    assert!((cfg.w_mixture_weight - 0.0).abs() < 1e-6);
    assert!((cfg.w_opponent_hand_type - 0.0).abs() < 1e-6);
    assert!((cfg.w_delta_q - 0.0).abs() < 1e-6);
    assert!((cfg.w_safety_residual - 0.0).abs() < 1e-6);
    assert!((cfg.total_weight() - 2.05).abs() < 1e-4);
}

#[test]
fn validate_rejects_negative_primary_weights() {
    assert!(
        HydraLossConfig::new()
            .with_w_tenpai(-0.1)
            .validate()
            .is_err()
    );
    assert!(
        HydraLossConfig::new()
            .with_w_danger(-0.1)
            .validate()
            .is_err()
    );
    assert!(HydraLossConfig::new().with_w_opp(-0.1).validate().is_err());
    assert!(
        HydraLossConfig::new()
            .with_w_score(-0.1)
            .validate()
            .is_err()
    );
}

#[test]
fn slice_batch_clears_cached_target_presence() {
    let device = Default::default();
    let mut targets = make_dummy_targets::<B>(&device, 4);
    targets.target_presence = Some(TargetPresence {
        counts: [1, 2, 3, 4, 5, 6],
        delta_q_actions_present: 7,
        batch_size: 4,
    });

    let sliced = targets.slice_batch(1, 3);
    assert!(
        sliced.target_presence.is_none(),
        "sliced targets must drop cached full-batch presence metadata"
    );
}
