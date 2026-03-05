//! Robust opponent modeling: KL-ball uncertainty + archetype soft-min.

pub fn log_sum_exp_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }
    let max_v = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_v == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = values.iter().map(|v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

pub struct ArchetypeWeights {
    pub weights: Vec<f32>,
}

impl ArchetypeWeights {
    pub fn uniform(n: usize) -> Self {
        Self {
            weights: vec![1.0 / n as f32; n],
        }
    }

    pub fn update_posterior(&mut self, log_likelihoods: &[f32]) {
        let max_ll = log_likelihoods
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        for (w, &ll) in self.weights.iter_mut().zip(log_likelihoods) {
            *w *= (ll - max_ll).exp();
        }
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

pub struct RobustOpponentConfig {
    pub epsilon: f32,
    pub tau_search_iters: u8,
    pub num_archetypes: usize,
    pub tau_arch: f32,
}

impl Default for RobustOpponentConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.2,
            tau_search_iters: 20,
            num_archetypes: 4,
            tau_arch: 1.0,
        }
    }
}

pub fn find_robust_tau(p: &[f32], q_values: &[f32], epsilon: f32, iters: u8) -> (f32, Vec<f32>) {
    assert_eq!(p.len(), q_values.len());
    let n = p.len();
    let mut tau_lo = 0.01f32;
    let mut tau_hi = 100.0f32;
    let mut best_tau = 1.0f32;
    let mut best_q = vec![0.0f32; n];

    for _ in 0..iters {
        let tau = (tau_lo + tau_hi) / 2.0;
        let mut q_tau = vec![0.0f32; n];
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            let v = p[i].ln() - q_values[i] / tau;
            if v > max_val {
                max_val = v;
            }
        }
        let mut z = 0.0f32;
        for i in 0..n {
            q_tau[i] = (p[i].ln() - q_values[i] / tau - max_val).exp();
            z += q_tau[i];
        }
        for v in &mut q_tau {
            *v /= z;
        }

        let mut kl = 0.0f32;
        for i in 0..n {
            if q_tau[i] > 1e-10 && p[i] > 1e-10 {
                kl += q_tau[i] * (q_tau[i] / p[i]).ln();
            }
        }

        if kl > epsilon {
            tau_lo = tau;
        } else {
            tau_hi = tau;
        }
        best_tau = tau;
        best_q = q_tau;
    }
    (best_tau, best_q)
}

pub fn archetype_softmin(q_per_arch: &[Vec<f32>], weights: &[f32], tau_arch: f32) -> Vec<f32> {
    if q_per_arch.is_empty() {
        return Vec::new();
    }
    let n_actions = q_per_arch[0].len();
    let mut result = vec![0.0f32; n_actions];
    for a in 0..n_actions {
        let mut max_v = f32::NEG_INFINITY;
        for (i, qs) in q_per_arch.iter().enumerate() {
            let v = weights[i].ln() - qs[a] / tau_arch;
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f32;
        for (i, qs) in q_per_arch.iter().enumerate() {
            sum += (weights[i].ln() - qs[a] / tau_arch - max_v).exp();
        }
        result[a] = -tau_arch * (max_v + sum.ln());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_tau_converges() {
        let p = vec![0.3, 0.5, 0.2];
        let q = vec![1.0, 0.5, 2.0];
        let eps = 0.1;
        let (tau, q_tau) = find_robust_tau(&p, &q, eps, 20);
        assert!(tau > 0.0);
        let sum: f32 = q_tau.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "q_tau should sum to 1");
        let mut kl = 0.0f32;
        for i in 0..3 {
            if q_tau[i] > 1e-10 && p[i] > 1e-10 {
                kl += q_tau[i] * (q_tau[i] / p[i]).ln();
            }
        }
        let kl_error = (kl - eps).abs() / eps;
        assert!(kl_error < 0.05, "KL={kl} should be within 5% of eps={eps}");
    }

    #[test]
    fn archetype_softmin_equal_q() {
        let qs = vec![vec![1.0, 2.0, 3.0]; 4];
        let w = vec![0.25; 4];
        let result = archetype_softmin(&qs, &w, 1.0);
        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert!(
                (result[i] - (i as f32 + 1.0)).abs() < 0.1,
                "expected ~{}, got {}",
                i as f32 + 1.0,
                result[i]
            );
        }
    }

    #[test]
    fn archetype_softmin_different_q_shifts() {
        let qs = vec![
            vec![1.0, 5.0, 3.0],
            vec![5.0, 1.0, 3.0],
            vec![3.0, 3.0, 1.0],
            vec![3.0, 3.0, 5.0],
        ];
        let w = vec![0.25; 4];
        let result = archetype_softmin(&qs, &w, 1.0);
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn robust_tau_uniform_policy() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let (tau, q_tau) = find_robust_tau(&p, &q, 0.05, 20);
        assert!(tau > 0.0);
        let sum: f32 = q_tau.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn robust_tau_identical_q_stays_close_to_prior() {
        let p = vec![0.3, 0.5, 0.2];
        let q = vec![1.0, 1.0, 1.0];
        let (_, q_tau) = find_robust_tau(&p, &q, 0.1, 20);
        let sum: f32 = q_tau.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        for i in 0..3 {
            assert!(
                (q_tau[i] - p[i]).abs() < 0.2,
                "identical Q -> q_tau should be near prior: {} vs {}",
                q_tau[i],
                p[i]
            );
        }
    }
}
