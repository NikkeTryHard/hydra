use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};

pub fn compatible_discard_state(legal_mask: &[f32]) -> bool {
    if legal_mask.len() != HYDRA_ACTION_SPACE {
        return false;
    }
    let non_discard_legal = legal_mask[(DISCARD_END as usize + 1)..]
        .iter()
        .any(|&x| x > 0.0);
    if non_discard_legal {
        return false;
    }
    let aka_legal = legal_mask[AKA_5M as usize] > 0.0
        || legal_mask[AKA_5P as usize] > 0.0
        || legal_mask[AKA_5S as usize] > 0.0;
    !aka_legal
}

pub fn safety_valve_check(
    base_pi: &[f32],
    exit_pi: &[f32],
    max_kl: f32,
    min_visits: u32,
    visit_count: u32,
) -> bool {
    if visit_count < min_visits {
        return false;
    }
    let mut kl = 0.0f32;
    for i in 0..base_pi.len() {
        if exit_pi[i] > 1e-10 && base_pi[i] > 1e-10 {
            kl += exit_pi[i] * (exit_pi[i] / base_pi[i]).ln();
        }
    }
    kl <= max_kl
}
