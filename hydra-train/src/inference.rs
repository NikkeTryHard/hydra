//! Inference server: fast path (network + SaF) and slow path (pondered AFBS).

use burn::prelude::*;
use burn::tensor::activation;
use hydra_core::action::HYDRA_ACTION_SPACE;

pub fn infer_action<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let mut mask_f32 = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in legal_mask.iter().enumerate() {
        mask_f32[i] = if m { 1.0 } else { 0.0 };
    }
    let device = policy_logits.device();
    let mask_tensor = Tensor::<B, 1>::from_floats(&mask_f32[..], &device).unsqueeze_dim::<2>(0);
    let neg_inf = (mask_tensor.ones_like() - mask_tensor) * (-1e9f32);
    let masked = policy_logits + neg_inf;
    let probs = activation::softmax(masked, 1);
    let probs_data = probs.to_data();
    let probs_slice = probs_data.as_slice::<f32>().expect("f32");

    let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
    policy.copy_from_slice(&probs_slice[..HYDRA_ACTION_SPACE]);

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn inference_picks_legal_action() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats(
            [[
                10.0, -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]],
            &device,
        );
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[1] = true;
        mask[2] = true;
        let (action, policy) = infer_action(logits, &mask);
        assert!(mask[action as usize], "picked illegal action {action}");
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "policy should sum to 1, got {sum}"
        );
    }
}
