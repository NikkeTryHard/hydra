use burn::prelude::Backend;
use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::arena::{TrajectoryStep, softmax_temperature};

use hydra_model::model::HydraModel;

use crate::exit::compatible_discard_state;

#[derive(Debug, Clone)]
pub struct CommonGatePass {
    pub base_pi: [f32; HYDRA_ACTION_SPACE],
    pub legal_discards: Vec<usize>,
    pub legal_discard_count: usize,
}

#[derive(Debug, Clone)]
pub enum CommonGateOutcome {
    IncompatibleState,
    TooFewDiscards,
    NotHardState,
    Pass(Box<CommonGatePass>),
}

pub fn evaluate_common_validation_gate<B: Backend>(
    step: &TrajectoryStep,
    model: &HydraModel<B>,
    device: &B::Device,
    hard_state_threshold: f32,
) -> CommonGateOutcome {
    let legal_f32 = step
        .legal_mask
        .map(|is_legal| if is_legal { 1.0 } else { 0.0 });
    if !compatible_discard_state(&legal_f32) {
        return CommonGateOutcome::IncompatibleState;
    }

    let legal_discards = legal_discard_actions(step);
    let legal_discard_count = legal_discards.len();
    if legal_discard_count < 2 {
        return CommonGateOutcome::TooFewDiscards;
    }

    let policy_logits = model.policy_cpu(&step.obs, device);
    let base_pi = softmax_temperature(&policy_logits, &step.legal_mask, 1.0);
    if !is_hard_state_for_legal_discards(&base_pi, &legal_discards, hard_state_threshold) {
        return CommonGateOutcome::NotHardState;
    }

    CommonGateOutcome::Pass(Box::new(CommonGatePass {
        base_pi,
        legal_discards,
        legal_discard_count,
    }))
}

pub fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub fn ratio_f64(numerator: f64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}

pub fn legal_discard_actions(step: &TrajectoryStep) -> Vec<usize> {
    (0..=DISCARD_END as usize)
        .filter(|&action| step.legal_mask[action])
        .collect()
}

fn is_hard_state_for_legal_discards(
    policy: &[f32; HYDRA_ACTION_SPACE],
    legal_discards: &[usize],
    threshold: f32,
) -> bool {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;

    for &action in legal_discards {
        let value = policy[action];
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }

    legal_discards.len() >= 2 && (best - second) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::encoder::OBS_SIZE;
    use hydra_model::model::HydraModelConfig;

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
}
