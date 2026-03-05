//! Sinkhorn-Knopp projection for Mixture-SIB belief inference.

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
    kernel: &[f64; 136],
    row_sums: &[f64; 34],
    col_sums: &[f64; 4],
    max_iters: u16,
    tol: f64,
) -> [f64; 136] {
    let mut u = [1.0f64; 34];
    let mut v = [1.0f64; 4];

    for _ in 0..max_iters {
        for i in 0..34 {
            let mut s = 0.0;
            for j in 0..4 {
                s += kernel[i * 4 + j] * v[j];
            }
            u[i] = if s > 1e-15 { row_sums[i] / s } else { 0.0 };
        }
        for j in 0..4 {
            let mut s = 0.0;
            for i in 0..34 {
                s += kernel[i * 4 + j] * u[i];
            }
            v[j] = if s > 1e-15 { col_sums[j] / s } else { 0.0 };
        }

        let mut row_err = 0.0f64;
        for i in 0..34 {
            let mut s = 0.0;
            for j in 0..4 {
                s += kernel[i * 4 + j] * u[i] * v[j];
            }
            row_err += (s - row_sums[i]).abs();
        }
        if row_err < tol {
            break;
        }
    }
    let mut b = [0.0f64; 136];
    for i in 0..34 {
        for j in 0..4 {
            b[i * 4 + j] = kernel[i * 4 + j] * u[i] * v[j];
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
        let log_sum: f64 =
            self.components
                .iter()
                .map(|c| c.log_weight)
                .fold(f64::NEG_INFINITY, |a, b| {
                    let mx = a.max(b);
                    mx + ((a - mx).exp() + (b - mx).exp()).ln()
                });
        for comp in &mut self.components {
            comp.log_weight -= log_sum;
        }
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
}
