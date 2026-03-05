//! ExIt pipeline: search target generation and safety valve.

use burn::prelude::*;
use burn::tensor::activation;

#[derive(Config, Debug)]
pub struct ExitConfig {
    #[config(default = "1.0")]
    pub tau_exit: f32,
    #[config(default = "0.5")]
    pub exit_weight: f32,
    #[config(default = "64")]
    pub min_visits: u32,
    #[config(default = "0.1")]
    pub hard_state_threshold: f32,
    #[config(default = "2.0")]
    pub safety_valve_max_kl: f32,
}

pub fn is_hard_state(policy: &[f32], threshold: f32) -> bool {
    if policy.len() < 2 {
        return false;
    }
    let mut sorted: Vec<f32> = policy.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[0] - sorted[1] < threshold
}

pub fn exit_policy_from_q(q_values: &[f32], tau: f32) -> Vec<f32> {
    let n = q_values.len();
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; n];
    let mut total = 0.0f32;
    for i in 0..n {
        probs[i] = ((q_values[i] - max_q) / tau).exp();
        total += probs[i];
    }
    if total > 0.0 {
        for p in &mut probs {
            *p /= total;
        }
    }
    probs
}

pub fn exit_loss<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    weight: f32,
) -> Tensor<B, 1> {
    let log_pi = activation::log_softmax(model_logits, 1);
    let ce = (exit_target * log_pi).sum_dim(1).neg().mean();
    ce * weight
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_safety_valve_skips_low_visits() {
        let base = vec![0.5, 0.3, 0.2];
        let exit = vec![0.4, 0.4, 0.2];
        assert!(!safety_valve_check(&base, &exit, 2.0, 64, 10));
    }

    #[test]
    fn exit_safety_valve_passes_good_target() {
        let p = vec![0.5, 0.3, 0.2];
        assert!(safety_valve_check(&p, &p, 2.0, 64, 100));
    }

    #[test]
    fn exit_policy_sums_to_one() {
        let q = vec![1.0, 2.0, 0.5, 3.0];
        let pi = exit_policy_from_q(&q, 1.0);
        let sum: f32 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "should sum to 1, got {sum}");
    }

    #[test]
    fn exit_safety_valve_rejects_high_kl() {
        let base = vec![0.9, 0.05, 0.05];
        let exit = vec![0.05, 0.05, 0.9];
        assert!(!safety_valve_check(&base, &exit, 0.5, 64, 100));
    }

    #[test]
    fn exit_policy_concentrates_on_best() {
        let q = vec![10.0, 0.0, 0.0, 0.0];
        let pi = exit_policy_from_q(&q, 0.1);
        assert!(
            pi[0] > 0.99,
            "low tau should concentrate on best action: {}",
            pi[0]
        );
    }
}
