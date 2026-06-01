use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;

use super::normalize_policy_cpu;

/// Converts a boolean legal mask to a [1, 46] float tensor.
pub fn legal_mask_to_tensor<B: Backend>(
    mask: &[bool; HYDRA_ACTION_SPACE],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let mut f32_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in mask.iter().enumerate() {
        f32_mask[i] = if m { 1.0 } else { 0.0 };
    }
    Tensor::<B, 1>::from_floats(&f32_mask[..], device).unsqueeze_dim::<2>(0)
}
pub fn batch_legal_masks_to_tensor<B: Backend>(
    masks: &[[bool; HYDRA_ACTION_SPACE]],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let batch = masks.len();
    let mut flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (i, mask) in masks.iter().enumerate() {
        for (j, &m) in mask.iter().enumerate() {
            if m {
                flat[i * HYDRA_ACTION_SPACE + j] = 1.0;
            }
        }
    }
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
}

/// Runs inference with wall-clock time measurement against a budget.
pub fn infer_action_timed<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    budget_ms: u64,
) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
    let start = std::time::Instant::now();
    let (action, policy) = infer_action(policy_logits, legal_mask);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let within_budget = elapsed_ms <= budget_ms;
    (action, policy, within_budget)
}

/// Returns the fraction of batch elements where argmax picks an illegal action.
pub fn illegal_action_rate<B: Backend>(logits: Tensor<B, 2>, legal_mask: Tensor<B, 2>) -> f32 {
    let neg_inf = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
    let raw_predicted = logits.clone().argmax(1);
    let masked = logits + neg_inf;
    let predicted = masked.argmax(1);
    let same = predicted.equal(raw_predicted).int().sum();
    let batch = legal_mask.dims()[0] as f32;
    1.0 - same
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("illegal action rate scalar should be readable as f32")[0]
        / batch
}

/// Runs masked softmax inference, returns (best_action, policy_probs).
pub fn infer_action<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let logits_data = policy_logits.to_data().convert::<f32>();
    let logits = logits_data
        .as_slice::<f32>()
        .expect("policy logits extraction failed");
    let mut logits_arr = [0.0f32; HYDRA_ACTION_SPACE];
    logits_arr.copy_from_slice(&logits[..HYDRA_ACTION_SPACE]);
    let policy = normalize_policy_cpu(&logits_arr, legal_mask);

    let mut best_action = 0u8;
    let mut best_prob = f32::NEG_INFINITY;
    for (i, &p) in policy.iter().enumerate() {
        if legal_mask[i] && p > best_prob {
            best_prob = p;
            best_action = i as u8;
        }
    }
    (best_action, policy)
}
