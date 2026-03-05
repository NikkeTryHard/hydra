//! Inference server: fast path (network + SaF) and slow path (pondered AFBS).

use burn::prelude::*;
use burn::tensor::activation;
use hydra_core::action::HYDRA_ACTION_SPACE;

pub struct InferenceConfig {
    pub on_turn_budget_ms: u64,
    pub call_reaction_budget_ms: u64,
    pub agari_guard: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            on_turn_budget_ms: 150,
            call_reaction_budget_ms: 50,
            agari_guard: true,
        }
    }
}

impl InferenceConfig {
    pub fn summary(&self) -> String {
        format!(
            "infer(turn={}ms, call={}ms, guard={})",
            self.on_turn_budget_ms, self.call_reaction_budget_ms, self.agari_guard
        )
    }
}

pub fn legal_mask_to_tensor<B: Backend>(
    mask: &[bool; HYDRA_ACTION_SPACE],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut f32_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in mask.iter().enumerate() {
        f32_mask[i] = if m { 1.0 } else { 0.0 };
    }
    Tensor::<B, 1>::from_floats(&f32_mask[..], device).unsqueeze_dim::<2>(0)
}

pub fn normalize_policy_cpu(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            adjusted[i] = logits[i];
            if logits[i] > max_val {
                max_val = logits[i];
            }
        }
    }
    let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            probs[i] = (adjusted[i] - max_val).exp();
            total += probs[i];
        }
    }
    if total > 0.0 {
        for p in &mut probs {
            *p /= total;
        }
    }
    probs
}

pub fn validate_legal_mask(mask: &[bool; HYDRA_ACTION_SPACE]) -> bool {
    mask.iter().any(|&m| m)
}

pub fn policy_entropy(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-8 {
            h -= p * p.ln();
        }
    }
    h
}

pub fn needs_search(probs: &[f32; HYDRA_ACTION_SPACE], gap_threshold: f32) -> bool {
    policy_top2_gap(probs) < gap_threshold
}

pub fn is_confident(probs: &[f32; HYDRA_ACTION_SPACE], threshold: f32) -> bool {
    policy_top1_confidence(probs) >= threshold
}

pub fn sample_from_policy(probs: &[f32; HYDRA_ACTION_SPACE], rng_val: f32) -> u8 {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_val <= cumsum {
            return i as u8;
        }
    }
    (HYDRA_ACTION_SPACE - 1) as u8
}

pub fn num_legal_actions(mask: &[bool; HYDRA_ACTION_SPACE]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

pub fn argmax_legal(probs: &[f32; HYDRA_ACTION_SPACE], mask: &[bool; HYDRA_ACTION_SPACE]) -> u8 {
    let mut best = 0u8;
    let mut best_p = f32::NEG_INFINITY;
    for (i, (&p, &m)) in probs.iter().zip(mask.iter()).enumerate() {
        if m && p > best_p {
            best_p = p;
            best = i as u8;
        }
    }
    best
}

pub fn compute_entropy_from_logits(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> f32 {
    let probs = normalize_policy_cpu(logits, legal_mask);
    policy_entropy(&probs)
}

pub fn policy_top2_gap(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut first = 0.0f32;
    let mut second = 0.0f32;
    for &p in probs {
        if p > first {
            second = first;
            first = p;
        } else if p > second {
            second = p;
        }
    }
    first - second
}

pub fn policy_top1_confidence(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    probs.iter().cloned().fold(0.0f32, f32::max)
}

pub fn batch_legal_masks_to_tensor<B: Backend>(
    masks: &[[bool; HYDRA_ACTION_SPACE]],
    device: &B::Device,
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

pub fn infer_action<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let device = policy_logits.device();
    let mask_tensor = legal_mask_to_tensor(legal_mask, &device);
    let neg_inf = (mask_tensor.ones_like() - mask_tensor) * (-1e9f32);
    let masked = policy_logits + neg_inf;
    let probs = activation::softmax(masked, 1);
    let probs_data = probs.to_data();
    let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
    if let Ok(probs_slice) = probs_data.as_slice::<f32>() {
        policy.copy_from_slice(&probs_slice[..HYDRA_ACTION_SPACE]);
    }

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

    #[test]
    fn agari_guard_prevents_illegal() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[43] = 100.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[45] = true;
        let (action, _) = infer_action(logits, &mask);
        assert_ne!(action, 43, "agari (43) is illegal but has highest logit");
        assert!(mask[action as usize], "must pick legal: got {action}");
    }

    #[test]
    fn inference_config_defaults() {
        let cfg = InferenceConfig::default();
        assert_eq!(cfg.on_turn_budget_ms, 150);
        assert_eq!(cfg.call_reaction_budget_ms, 50);
        assert!(cfg.agari_guard);
    }

    #[test]
    fn illegal_actions_get_zero_probability() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[0] = 5.0;
        logits_data[1] = 3.0;
        logits_data[2] = 1.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[2] = true;
        let (_, policy) = infer_action(logits, &mask);
        assert!(
            policy[1] < 1e-6,
            "illegal action 1 should have ~0 prob: {}",
            policy[1]
        );
        assert!(
            policy[0] > 0.1,
            "legal action 0 should have significant prob"
        );
        assert!(policy[2] > 0.01, "legal action 2 should have some prob");
    }

    #[test]
    fn normalize_policy_cpu_sums_to_one() {
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[5] = 3.0;
        logits[10] = 1.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[5] = true;
        mask[10] = true;
        let probs = normalize_policy_cpu(&logits, &mask);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
        assert!(probs[0] > probs[5]);
        assert!(probs[5] > probs[10]);
    }

    #[test]
    fn sample_from_policy_respects_distribution() {
        let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
        probs[0] = 0.7;
        probs[1] = 0.3;
        let a0 = sample_from_policy(&probs, 0.0);
        assert_eq!(a0, 0);
        let a1 = sample_from_policy(&probs, 0.8);
        assert_eq!(a1, 1);
    }
}
