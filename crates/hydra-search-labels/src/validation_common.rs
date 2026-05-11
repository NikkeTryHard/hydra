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
mod tests;
