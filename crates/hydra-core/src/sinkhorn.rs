//! Sinkhorn-Knopp projection for Mixture-SIB belief inference.

const NUM_TILE_TYPES: usize = 34;
const NUM_ZONES: usize = 4;
const BELIEF_SIZE: usize = NUM_TILE_TYPES * NUM_ZONES;

fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_v.is_finite() {
        return max_v;
    }
    let sum: f64 = values.iter().map(|v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

pub struct SinkhornConfig {
    pub max_iters: u16,
    pub tol: f64,
    pub num_components: u8,
}

impl SinkhornConfig {
    pub fn new(max_iters: u16, tol: f64, num_components: u8) -> Self {
        Self {
            max_iters,
            tol,
            num_components,
        }
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn summary(&self) -> String {
        format!(
            "sinkhorn(iters={}, tol={:.1e}, L={})",
            self.max_iters, self.tol, self.num_components
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_iters == 0 {
            return Err("max_iters must be > 0");
        }
        if self.tol <= 0.0 {
            return Err("tol must be positive");
        }
        if self.num_components == 0 {
            return Err("num_components must be > 0");
        }
        Ok(())
    }
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-8,
            num_components: 4,
        }
    }
}

pub fn sinkhorn_project(
    kernel: &[f64; BELIEF_SIZE],
    row_sums: &[f64; NUM_TILE_TYPES],
    col_sums: &[f64; NUM_ZONES],
    max_iters: u16,
    tol: f64,
) -> [f64; BELIEF_SIZE] {
    sinkhorn_project_log_domain(kernel, row_sums, col_sums, max_iters, tol)
}

pub fn sinkhorn_project_log_domain(
    kernel: &[f64; BELIEF_SIZE],
    row_sums: &[f64; NUM_TILE_TYPES],
    col_sums: &[f64; NUM_ZONES],
    max_iters: u16,
    tol: f64,
) -> [f64; BELIEF_SIZE] {
    let mut log_u = [0.0f64; NUM_TILE_TYPES];
    let mut log_v = [0.0f64; NUM_ZONES];
    let mut log_kernel = [f64::NEG_INFINITY; BELIEF_SIZE];

    for i in 0..BELIEF_SIZE {
        if kernel[i] > 0.0 {
            log_kernel[i] = kernel[i].ln();
        }
    }

    for _ in 0..max_iters {
        for i in 0..NUM_TILE_TYPES {
            if row_sums[i] <= 0.0 {
                log_u[i] = f64::NEG_INFINITY;
                continue;
            }
            let mut row_terms = [f64::NEG_INFINITY; NUM_ZONES];
            for j in 0..NUM_ZONES {
                row_terms[j] = log_kernel[i * NUM_ZONES + j] + log_v[j];
            }
            let log_row = log_sum_exp(&row_terms);
            log_u[i] = if log_row.is_finite() {
                row_sums[i].ln() - log_row
            } else {
                f64::NEG_INFINITY
            };
        }

        for j in 0..NUM_ZONES {
            if col_sums[j] <= 0.0 {
                log_v[j] = f64::NEG_INFINITY;
                continue;
            }
            let mut col_terms = [f64::NEG_INFINITY; NUM_TILE_TYPES];
            for i in 0..NUM_TILE_TYPES {
                col_terms[i] = log_kernel[i * NUM_ZONES + j] + log_u[i];
            }
            let log_col = log_sum_exp(&col_terms);
            log_v[j] = if log_col.is_finite() {
                col_sums[j].ln() - log_col
            } else {
                f64::NEG_INFINITY
            };
        }

        let mut row_err = 0.0f64;
        for i in 0..NUM_TILE_TYPES {
            let mut s = 0.0;
            for j in 0..NUM_ZONES {
                let log_b = log_u[i] + log_kernel[i * NUM_ZONES + j] + log_v[j];
                if log_b.is_finite() {
                    s += log_b.exp();
                }
            }
            row_err += (s - row_sums[i]).abs();
        }
        if row_err < tol {
            break;
        }
    }
    let mut b = [0.0f64; BELIEF_SIZE];
    for i in 0..NUM_TILE_TYPES {
        for j in 0..NUM_ZONES {
            let log_b = log_u[i] + log_kernel[i * NUM_ZONES + j] + log_v[j];
            b[i * NUM_ZONES + j] = if log_b.is_finite() { log_b.exp() } else { 0.0 };
        }
    }
    b
}

pub struct SibComponent {
    pub belief: [f64; 136],
    pub log_weight: f64,
}

pub struct MixtureSib {
    pub components: Vec<SibComponent>,
}

impl MixtureSib {
    fn renormalize_log_weights(&mut self) {
        let log_weights: Vec<f64> = self.components.iter().map(|c| c.log_weight).collect();
        let log_z = log_sum_exp(&log_weights);
        if log_z.is_finite() {
            for component in &mut self.components {
                component.log_weight -= log_z;
            }
        } else if !self.components.is_empty() {
            self.reset_weights();
        }
    }

    fn set_weights_from_probs(&mut self, weights: &[f64]) {
        let sum: f64 = weights.iter().copied().sum();
        if sum <= 0.0 || !sum.is_finite() || self.components.is_empty() {
            self.reset_weights();
            return;
        }
        for (component, &weight) in self.components.iter_mut().zip(weights.iter()) {
            let normalized = (weight / sum).max(1e-300);
            component.log_weight = normalized.ln();
        }
        self.renormalize_log_weights();
    }

    fn belief_l1_distance(a: &[f64; BELIEF_SIZE], b: &[f64; BELIEF_SIZE]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    pub fn new(
        num_components: u8,
        kernel: &[f64; 136],
        row_sums: &[f64; 34],
        col_sums: &[f64; 4],
    ) -> Self {
        let belief = sinkhorn_project(kernel, row_sums, col_sums, 50, 1e-8);
        let components = (0..num_components)
            .map(|_| SibComponent {
                belief,
                log_weight: -(num_components as f64).ln(),
            })
            .collect();
        Self { components }
    }

    pub fn bayesian_update(&mut self, event_log_likelihoods: &[f64]) {
        assert_eq!(event_log_likelihoods.len(), self.components.len());
        for (comp, &ll) in self.components.iter_mut().zip(event_log_likelihoods) {
            comp.log_weight += ll;
        }
        self.renormalize_log_weights();
    }

    pub fn apply_entropy_regularizer(&mut self, mix: f64) {
        if self.components.is_empty() {
            return;
        }
        let gamma = mix.clamp(0.0, 1.0);
        if gamma <= 0.0 {
            return;
        }

        let uniform = 1.0 / self.components.len() as f64;
        let blended: Vec<f64> = self
            .weights()
            .into_iter()
            .map(|w| (1.0 - gamma) * w + gamma * uniform)
            .collect();
        self.set_weights_from_probs(&blended);
    }

    pub fn apply_diversity_penalty(&mut self, penalty: f64) {
        if self.components.len() < 2 || penalty <= 0.0 {
            return;
        }

        let mut adjusted = self.weights();
        for (i, weight) in adjusted.iter_mut().enumerate() {
            let mut overlap_penalty = 0.0;
            for j in 0..self.components.len() {
                if i == j {
                    continue;
                }
                let dist = Self::belief_l1_distance(
                    &self.components[i].belief,
                    &self.components[j].belief,
                );
                overlap_penalty += 1.0 / (1.0 + dist);
            }
            *weight *= (-penalty * overlap_penalty).exp();
        }
        self.set_weights_from_probs(&adjusted);
    }

    pub fn split_dominant_component_if_low_ess(&mut self, min_ess_ratio: f64, jitter: f64) -> bool {
        if self.components.is_empty() {
            return false;
        }
        let threshold = min_ess_ratio * self.components.len() as f64;
        if self.ess() >= threshold {
            return false;
        }

        let dominant = self.dominant_component();
        let mut clone = SibComponent {
            belief: self.components[dominant].belief,
            log_weight: self.components[dominant].log_weight,
        };
        let total_mass: f64 = clone.belief.iter().sum();
        let delta = jitter.abs();
        for (idx, value) in clone.belief.iter_mut().enumerate() {
            let factor = if idx % 2 == 0 {
                1.0 + delta
            } else {
                1.0 - delta
            };
            *value = (*value * factor).max(0.0);
        }
        let new_mass: f64 = clone.belief.iter().sum();
        if total_mass > 0.0 && new_mass > 0.0 {
            let scale = total_mass / new_mass;
            for value in &mut clone.belief {
                *value *= scale;
            }
        }

        let split_weight = self.components[dominant].log_weight - (2.0f64).ln();
        self.components[dominant].log_weight = split_weight;
        clone.log_weight = split_weight;
        self.components.push(clone);
        self.renormalize_log_weights();
        true
    }

    pub fn merge_closest_components(&mut self, distance_threshold: f64) -> bool {
        if self.components.len() < 2 {
            return false;
        }

        let mut best_pair = None;
        let mut best_distance = f64::INFINITY;
        for i in 0..self.components.len() {
            for j in (i + 1)..self.components.len() {
                let distance = Self::belief_l1_distance(
                    &self.components[i].belief,
                    &self.components[j].belief,
                );
                if distance < best_distance {
                    best_distance = distance;
                    best_pair = Some((i, j));
                }
            }
        }

        let Some((left, right)) = best_pair else {
            return false;
        };
        if best_distance > distance_threshold {
            return false;
        }

        let log_pair = log_sum_exp(&[
            self.components[left].log_weight,
            self.components[right].log_weight,
        ]);
        let left_w = (self.components[left].log_weight - log_pair).exp();
        let right_w = (self.components[right].log_weight - log_pair).exp();
        let mut merged = [0.0f64; BELIEF_SIZE];
        for (idx, value) in merged.iter_mut().enumerate() {
            *value = left_w * self.components[left].belief[idx]
                + right_w * self.components[right].belief[idx];
        }

        self.components[left].belief = merged;
        self.components[left].log_weight = log_pair;
        self.components.remove(right);
        self.renormalize_log_weights();
        true
    }

    pub fn posterior_step(
        &mut self,
        event_log_likelihoods: &[f64],
        entropy_mix: f64,
        min_ess_ratio: f64,
        split_jitter: f64,
        merge_distance_threshold: f64,
        diversity_penalty: f64,
    ) {
        self.bayesian_update(event_log_likelihoods);
        self.apply_diversity_penalty(diversity_penalty);
        self.apply_entropy_regularizer(entropy_mix);
        self.split_dominant_component_if_low_ess(min_ess_ratio, split_jitter);
        self.merge_closest_components(merge_distance_threshold);
    }

    pub fn summary(&self) -> String {
        format!("sib(L={}, ess={:.1})", self.num_components(), self.ess())
    }

    pub fn weight_entropy(&self) -> f64 {
        let w = self.weights();
        let mut h = 0.0f64;
        for &wi in &w {
            if wi > 1e-15 {
                h -= wi * wi.ln();
            }
        }
        h
    }

    pub fn min_weight(&self) -> f64 {
        self.weights().into_iter().fold(f64::INFINITY, f64::min)
    }

    pub fn max_weight(&self) -> f64 {
        self.weights().into_iter().fold(0.0f64, f64::max)
    }

    pub fn dominant_component(&self) -> usize {
        let w = self.weights();
        w.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn reset_weights(&mut self) {
        let n = self.components.len();
        let w = -(n as f64).ln();
        for comp in &mut self.components {
            comp.log_weight = w;
        }
    }

    pub fn is_converged(&self, tol: f64) -> bool {
        let w = self.weights();
        w.iter().all(|&w| w.is_finite() && w >= 0.0) && (w.iter().sum::<f64>() - 1.0).abs() < tol
    }

    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    pub fn weights(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.log_weight.exp()).collect()
    }

    pub fn marginal_belief(&self) -> [f64; 136] {
        let w = self.weights();
        let mut result = [0.0f64; 136];
        for (comp, &weight) in self.components.iter().zip(w.iter()) {
            for (r, &b) in result.iter_mut().zip(comp.belief.iter()) {
                *r += weight * b;
            }
        }
        result
    }

    pub fn reproject(&mut self, kernels: &[[f64; 136]], row_sums: &[f64; 34], col_sums: &[f64; 4]) {
        for (comp, kernel) in self.components.iter_mut().zip(kernels.iter()) {
            comp.belief = sinkhorn_project(kernel, row_sums, col_sums, 50, 1e-8);
        }
    }

    pub fn ess(&self) -> f64 {
        let w = self.weights();
        let sum: f64 = w.iter().sum();
        let sum_sq: f64 = w.iter().map(|x| x * x).sum();
        if sum_sq == 0.0 {
            return 0.0;
        }
        (sum * sum) / sum_sq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_converges_to_margins() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let b = sinkhorn_project(&kernel, &row_sums, &col_sums, 100, 1e-6);
        for i in 0..34 {
            let s: f64 = (0..4).map(|j| b[i * 4 + j]).sum();
            assert!((s - 4.0).abs() < 0.01, "row {i} sum = {s}");
        }
        for j in 0..4 {
            let s: f64 = (0..34).map(|i| b[i * 4 + j]).sum();
            assert!((s - 34.0).abs() < 0.01, "col {j} sum = {s}");
        }
    }

    #[test]
    fn mixture_weight_update_is_bayesian() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
        let w_before = mix.weights();
        let sum_before: f64 = w_before.iter().sum();
        assert!((sum_before - 1.0).abs() < 0.01);
        mix.bayesian_update(&[0.0, -1.0, -2.0]);
        let w_after = mix.weights();
        let sum_after: f64 = w_after.iter().sum();
        assert!(
            (sum_after - 1.0).abs() < 0.01,
            "weights should sum to 1 after update"
        );
        assert!(
            w_after[0] > w_after[1],
            "component 0 should have higher weight"
        );
        assert!(
            w_after[1] > w_after[2],
            "component 1 should have higher weight than 2"
        );
    }

    #[test]
    fn sinkhorn_nonuniform_kernel() {
        let mut kernel = [1.0f64; 136];
        for i in 0..10 {
            kernel[i * 4] = 2.0;
        }
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let b = sinkhorn_project(&kernel, &row_sums, &col_sums, 500, 1e-6);
        for i in 0..34 {
            let s: f64 = (0..4).map(|j| b[i * 4 + j]).sum();
            assert!((s - 4.0).abs() < 0.5, "nonuniform row {i} sum = {s}");
        }
    }

    #[test]
    fn mixture_marginal_sums_to_total() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        let marginal = mix.marginal_belief();
        let total: f64 = marginal.iter().sum();
        assert!((total - 136.0).abs() < 1.0, "marginal sum = {total}");
    }

    #[test]
    fn mixture_ess_equals_num_components_uniform() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        let ess = mix.ess();
        assert!(
            (ess - 4.0).abs() < 0.01,
            "uniform weights -> ESS=N, got {ess}"
        );
    }

    #[test]
    fn bayesian_update_collapsed_no_nan() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        mix.bayesian_update(&[-1000.0, -1000.0, -1000.0, 0.0]);
        let w = mix.weights();
        for (i, &wi) in w.iter().enumerate() {
            assert!(
                wi.is_finite(),
                "weight[{i}] should be finite after collapsed update, got {wi}"
            );
        }
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn mixture_ess_decreases_after_biased_update() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        let ess_before = mix.ess();
        mix.bayesian_update(&[0.0, -5.0, -10.0, -15.0]);
        let ess_after = mix.ess();
        assert!(ess_after < ess_before, "biased update should reduce ESS");
    }

    #[test]
    fn entropy_regularizer_increases_weight_entropy() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        mix.bayesian_update(&[0.0, -10.0, -10.0, -10.0]);
        let before = mix.weight_entropy();
        mix.apply_entropy_regularizer(0.2);
        let after = mix.weight_entropy();
        assert!(
            after > before,
            "entropy regularizer should increase entropy"
        );
    }

    #[test]
    fn split_low_ess_adds_component() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
        mix.bayesian_update(&[0.0, -20.0, -20.0]);
        let before = mix.num_components();
        assert!(mix.split_dominant_component_if_low_ess(0.9, 0.1));
        assert_eq!(mix.num_components(), before + 1);
    }

    #[test]
    fn merge_identical_components_reduces_component_count() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
        let before = mix.num_components();
        assert!(mix.merge_closest_components(0.0));
        assert_eq!(mix.num_components(), before - 1);
    }

    #[test]
    fn diversity_penalty_keeps_weights_normalized() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        mix.apply_diversity_penalty(0.5);
        let sum: f64 = mix.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "weights should remain normalized");
    }

    #[test]
    fn sinkhorn_log_domain_handles_extreme_kernel_scales() {
        let mut kernel = [1.0f64; 136];
        kernel[0] = 1e-200;
        kernel[1] = 1e200;
        kernel[2] = 1e-120;
        kernel[3] = 1e120;
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0; 4];
        let belief = sinkhorn_project(&kernel, &row_sums, &col_sums, 200, 1e-6);

        for (idx, value) in belief.iter().enumerate() {
            assert!(
                value.is_finite(),
                "belief[{idx}] should be finite, got {value}"
            );
            assert!(
                *value >= 0.0,
                "belief[{idx}] should be non-negative, got {value}"
            );
        }

        for i in 0..34 {
            let row_sum: f64 = (0..4).map(|j| belief[i * 4 + j]).sum();
            assert!((row_sum - 4.0).abs() < 0.05, "row {i} sum = {row_sum}");
        }
    }
}
