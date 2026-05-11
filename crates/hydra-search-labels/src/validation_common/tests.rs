use super::*;
use burn::backend::NdArray;
use hydra_core::encoder::OBS_SIZE;
use hydra_model::model::{HydraModelConfig, HydraModelInit};

type B = NdArray<f32>;

fn step_with_discards(discard_actions: &[usize]) -> TrajectoryStep {
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    for &action in discard_actions {
        legal_mask[action] = true;
    }
    TrajectoryStep {
        obs: [0.0; OBS_SIZE],
        action: discard_actions.first().copied().unwrap_or_default() as u8,
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask,
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: false,
        player_id: 0,
        game_id: 0,
        turn: 0,
        temperature: 1.0,
    }
}

fn tiny_model() -> HydraModel<B> {
    let device = Default::default();
    HydraModelConfig::new(1)
        .with_hidden_channels(16)
        .with_se_bottleneck(4)
        .with_num_groups(4)
        .init::<B>(&device)
}

#[test]
fn legal_discard_actions_filters_to_discard_range_only() {
    let mut step = step_with_discards(&[0, 5, DISCARD_END as usize]);
    step.legal_mask[DISCARD_END as usize + 1] = true;
    step.legal_mask[HYDRA_ACTION_SPACE - 1] = true;

    let actions = legal_discard_actions(&step);

    assert_eq!(actions, vec![0, 5, DISCARD_END as usize]);
}

#[test]
fn ratio_helpers_handle_zero_and_nonzero_denominators() {
    assert_eq!(ratio_u64(3, 0), 0.0);
    assert_eq!(ratio_f64(3.0, 0), 0.0);
    assert!((ratio_u64(3, 4) - 0.75).abs() < 1e-12);
    assert!((ratio_f64(3.0, 4) - 0.75).abs() < 1e-12);
}

#[test]
fn common_gate_rejects_incompatible_state() {
    let mut step = step_with_discards(&[0, 1]);
    step.legal_mask[DISCARD_END as usize + 1] = true;
    let device = Default::default();
    let model = tiny_model();

    assert!(matches!(
        evaluate_common_validation_gate(&step, &model, &device, -1.0),
        CommonGateOutcome::IncompatibleState
    ));
}

#[test]
fn common_gate_rejects_too_few_discards() {
    let step = step_with_discards(&[2]);
    let device = Default::default();
    let model = tiny_model();

    assert!(matches!(
        evaluate_common_validation_gate(&step, &model, &device, -1.0),
        CommonGateOutcome::TooFewDiscards
    ));
}

#[test]
fn common_gate_rejects_not_hard_state() {
    let step = step_with_discards(&[1, 3]);
    let device = Default::default();
    let model = tiny_model();

    assert!(matches!(
        evaluate_common_validation_gate(&step, &model, &device, f32::INFINITY),
        CommonGateOutcome::NotHardState
    ));
}

#[test]
fn common_gate_passes_with_actions_and_policy() {
    let step = step_with_discards(&[1, 3]);
    let device = Default::default();
    let model = tiny_model();

    let outcome = evaluate_common_validation_gate(&step, &model, &device, -1.0);

    match outcome {
        CommonGateOutcome::Pass(pass) => {
            assert_eq!(pass.legal_discards, vec![1, 3]);
            assert_eq!(pass.legal_discard_count, 2);
            assert!(pass.base_pi[1].is_finite());
            assert!(pass.base_pi[3].is_finite());
        }
        other => panic!("expected pass, got {other:?}"),
    }
}
