pub fn anneal_exit_weight(base_weight: f32, phase: u8, progress: f32) -> f32 {
    match phase {
        0 | 1 => 0.0,
        2 => {
            let progress = progress.clamp(0.0, 1.0);
            if progress <= 0.5 {
                0.0
            } else {
                base_weight * ((progress - 0.5) / 0.5)
            }
        }
        _ => base_weight,
    }
}

pub fn is_hard_state(policy: &[f32], threshold: f32) -> bool {
    if policy.len() < 2 {
        return false;
    }
    let mut sorted: Vec<f32> = policy.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[0] - sorted[1] < threshold
}

pub fn exit_policy_from_q(q_values: &[f32], tau: f32, legal_mask: Option<&[bool]>) -> Vec<f32> {
    let n = q_values.len();
    let max_q = q_values
        .iter()
        .enumerate()
        .filter(|(i, _)| legal_mask.is_none_or(|m| *i < m.len() && m[*i]))
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; n];
    let mut total = 0.0f32;
    for i in 0..n {
        let is_legal = legal_mask.is_none_or(|m| i < m.len() && m[i]);
        if is_legal {
            probs[i] = ((q_values[i] - max_q) / tau).exp();
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
