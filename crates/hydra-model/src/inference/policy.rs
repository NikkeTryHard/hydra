use hydra_core::action::HYDRA_ACTION_SPACE;

/// Computes softmax policy on CPU with legal masking and max subtraction.
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

pub fn mask_policy_cpu(
    policy: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut masked = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            masked[i] = policy[i].max(0.0);
            total += masked[i];
        }
    }
    if total > 0.0 {
        for value in &mut masked {
            *value /= total;
        }
        return masked;
    }

    let legal_count = legal_mask.iter().filter(|&&m| m).count();
    if legal_count > 0 {
        let uniform = 1.0 / legal_count as f32;
        for (i, value) in masked.iter_mut().enumerate() {
            if legal_mask[i] {
                *value = uniform;
            }
        }
    }
    masked
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

pub fn action_rank(probs: &[f32; HYDRA_ACTION_SPACE], action: u8) -> usize {
    let p = probs[action as usize];
    probs.iter().filter(|&&q| q > p).count()
}

pub fn needs_search(probs: &[f32; HYDRA_ACTION_SPACE], gap_threshold: f32) -> bool {
    policy_top2_gap(probs) < gap_threshold
}

pub fn is_confident(probs: &[f32; HYDRA_ACTION_SPACE], threshold: f32) -> bool {
    policy_top1_confidence(probs) >= threshold
}

pub fn sample_from_policy(probs: &[f32; HYDRA_ACTION_SPACE], rng_val: f32) -> u8 {
    let mut cumsum = 0.0f32;
    let mut last_positive = 0u8;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_positive = i as u8;
        }
        cumsum += p;
        if rng_val <= cumsum {
            return i as u8;
        }
    }
    last_positive
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
