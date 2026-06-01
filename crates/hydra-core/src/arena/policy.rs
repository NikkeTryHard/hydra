use crate::action::HYDRA_ACTION_SPACE;

fn masked_softmax_probs(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            adjusted[i] = logits[i] / temperature;
            if adjusted[i] > max_val {
                max_val = adjusted[i];
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

pub fn softmax_temperature(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    masked_softmax_probs(logits, legal_mask, temperature)
}

pub fn greedy_action(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> u8 {
    let mut best = 0u8;
    let mut best_val = f32::NEG_INFINITY;
    for (i, (&l, &m)) in logits.iter().zip(legal_mask.iter()).enumerate() {
        if m && l > best_val {
            best_val = l;
            best = i as u8;
        }
    }
    best
}

pub fn sample_action_with_temperature(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
    rng_val: f32,
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let probs = masked_softmax_probs(logits, legal_mask, temperature);
    let mut cumsum = 0.0f32;
    let mut chosen = 0u8;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_val <= cumsum {
            chosen = i as u8;
            break;
        }
    }
    if !legal_mask[chosen as usize] {
        for (i, &m) in legal_mask.iter().enumerate() {
            if m {
                chosen = i as u8;
                break;
            }
        }
    }
    (chosen, probs)
}
