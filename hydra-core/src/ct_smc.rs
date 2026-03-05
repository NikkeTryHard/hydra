//! CT-SMC: Exact contingency-table sampler via log-space DP.
//!
//! Samples hidden tile allocations X[34][4] respecting row/column sum
//! constraints. Uses a 3D DP (c1,c2,c3) with c_W derived. Sub-millisecond
//! for 128 backward samples in optimized Rust.

use rand::Rng;
use std::collections::HashMap;

pub struct CtSmcConfig {
    pub num_particles: usize,
    pub ess_threshold: f32,
}

impl Default for CtSmcConfig {
    fn default() -> Self {
        Self {
            num_particles: 128,
            ess_threshold: 0.4,
        }
    }
}

pub struct Particle {
    pub allocation: [[u8; 4]; 34],
    pub log_weight: f64,
}

fn compositions(r: u8) -> Vec<[u8; 4]> {
    let mut result = Vec::new();
    for x0 in 0..=r {
        for x1 in 0..=(r - x0) {
            for x2 in 0..=(r - x0 - x1) {
                let x3 = r - x0 - x1 - x2;
                result.push([x0, x1, x2, x3]);
            }
        }
    }
    result
}

fn log_phi(comp: &[u8; 4], log_omega_k: &[f64; 4]) -> f64 {
    let mut val = 0.0;
    for j in 0..4 {
        val += comp[j] as f64 * log_omega_k[j];
    }
    val
}

fn logsumexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

type DpTable = Vec<HashMap<(u8, u8, u8), f64>>;

pub fn forward_dp(
    row_sums: &[u8; 34],
    col_sums: &[usize; 4],
    log_omega: &[[f64; 4]; 34],
) -> DpTable {
    let mut dp: DpTable = vec![HashMap::new(); 35];
    dp[34].insert((0, 0, 0), 0.0);

    for k in (0..34).rev() {
        let comps = compositions(row_sums[k]);
        let max_c0 = col_sums[0] as u8;
        let max_c1 = col_sums[1] as u8;
        let max_c2 = col_sums[2] as u8;
        for c0 in 0..=max_c0 {
            for c1 in 0..=max_c1 {
                for c2 in 0..=max_c2 {
                    let mut val = f64::NEG_INFINITY;
                    for comp in &comps {
                        if comp[0] > c0 || comp[1] > c1 || comp[2] > c2 {
                            continue;
                        }
                        let rem = (c0 - comp[0], c1 - comp[1], c2 - comp[2]);
                        let next = dp[k + 1].get(&rem).copied().unwrap_or(f64::NEG_INFINITY);
                        if next == f64::NEG_INFINITY {
                            continue;
                        }
                        let lp = log_phi(comp, &log_omega[k]);
                        val = logsumexp(val, lp + next);
                    }
                    if val != f64::NEG_INFINITY {
                        dp[k].insert((c0, c1, c2), val);
                    }
                }
            }
        }
    }
    dp
}

pub fn backward_sample<R: Rng>(
    dp: &DpTable,
    row_sums: &[u8; 34],
    col_sums: &[usize; 4],
    log_omega: &[[f64; 4]; 34],
    rng: &mut R,
) -> [[u8; 4]; 34] {
    let mut allocation = [[0u8; 4]; 34];
    let mut c = (col_sums[0] as u8, col_sums[1] as u8, col_sums[2] as u8);

    for k in 0..34 {
        let comps = compositions(row_sums[k]);
        let z_k = dp[k].get(&c).copied().unwrap_or(f64::NEG_INFINITY);
        if z_k == f64::NEG_INFINITY {
            return allocation;
        }
        let mut log_probs = Vec::with_capacity(comps.len());
        let mut valid_comps = Vec::with_capacity(comps.len());
        for comp in &comps {
            if comp[0] > c.0 || comp[1] > c.1 || comp[2] > c.2 {
                continue;
            }
            let next_c = (c.0 - comp[0], c.1 - comp[1], c.2 - comp[2]);
            let next_val = dp[k + 1].get(&next_c).copied().unwrap_or(f64::NEG_INFINITY);
            if next_val == f64::NEG_INFINITY {
                continue;
            }
            let lp = log_phi(comp, &log_omega[k]) + next_val - z_k;
            log_probs.push(lp);
            valid_comps.push(*comp);
        }
        if valid_comps.is_empty() {
            return allocation;
        }
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs.iter().map(|lp| (lp - max_lp).exp()).collect();
        let total: f64 = probs.iter().sum();
        let u: f64 = rng.random::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen = valid_comps.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                chosen = i;
                break;
            }
        }
        allocation[k] = valid_comps[chosen];
        c = (
            c.0 - valid_comps[chosen][0],
            c.1 - valid_comps[chosen][1],
            c.2 - valid_comps[chosen][2],
        );
    }
    allocation
}

pub struct CtSmc {
    pub config: CtSmcConfig,
    pub particles: Vec<Particle>,
}

impl CtSmc {
    pub fn new(config: CtSmcConfig) -> Self {
        Self {
            config,
            particles: Vec::new(),
        }
    }

    pub fn sample_particles<R: Rng>(
        &mut self,
        row_sums: &[u8; 34],
        col_sums: &[usize; 4],
        log_omega: &[[f64; 4]; 34],
        rng: &mut R,
    ) {
        let dp = forward_dp(row_sums, col_sums, log_omega);
        self.particles = (0..self.config.num_particles)
            .map(|_| Particle {
                allocation: backward_sample(&dp, row_sums, col_sums, log_omega, rng),
                log_weight: 0.0,
            })
            .collect();
    }

    pub fn ess(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }
        let max_w = self
            .particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = self
            .particles
            .iter()
            .map(|p| (p.log_weight - max_w).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq == 0.0 {
            return 0.0;
        }
        ((sum * sum) / sum_sq) as f32
    }

    pub fn systematic_resample<R: Rng>(&mut self, rng: &mut R) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let max_w = self
            .particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = self
            .particles
            .iter()
            .map(|p| (p.log_weight - max_w).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let step = total / n as f64;
        let mut u: f64 = rng.random::<f64>() * step;
        let mut cumsum = 0.0;
        let mut indices = Vec::with_capacity(n);
        let mut j = 0;
        for _ in 0..n {
            while cumsum + weights[j] < u && j + 1 < n {
                cumsum += weights[j];
                j += 1;
            }
            indices.push(j);
            u += step;
        }
        let old = std::mem::take(&mut self.particles);
        self.particles = indices
            .into_iter()
            .map(|i| Particle {
                allocation: old[i].allocation,
                log_weight: 0.0,
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn particles_satisfy_constraints() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut row_sums = [0u8; 34];
        row_sums[0] = 2;
        row_sums[1] = 1;
        row_sums[2] = 1;
        let col_sums = [1, 1, 1, 1];
        let log_omega = [[0.0f64; 4]; 34];
        let cfg = CtSmcConfig {
            num_particles: 32,
            ess_threshold: 0.4,
        };
        let mut smc = CtSmc::new(cfg);
        smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
        assert_eq!(smc.particles.len(), 32);
        for p in &smc.particles {
            for (k, &expected) in row_sums.iter().enumerate() {
                let rs: u8 = p.allocation[k].iter().sum();
                assert_eq!(rs, expected, "row {k}");
            }
            for (z, &expected) in col_sums.iter().enumerate() {
                let cs: usize = (0..34).map(|k| p.allocation[k][z] as usize).sum();
                assert_eq!(cs, expected, "col {z}");
            }
        }
    }

    #[test]
    fn uniform_likelihood_high_ess() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut row_sums = [0u8; 34];
        row_sums[0] = 1;
        row_sums[1] = 1;
        let col_sums = [1, 1, 0, 0];
        let log_omega = [[0.0f64; 4]; 34];
        let cfg = CtSmcConfig {
            num_particles: 64,
            ess_threshold: 0.4,
        };
        let mut smc = CtSmc::new(cfg);
        smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
        let ess = smc.ess();
        assert!(ess > 60.0, "ESS near P for uniform, got {ess}");
    }

    #[test]
    fn compositions_counts() {
        assert_eq!(compositions(0).len(), 1);
        assert_eq!(compositions(1).len(), 4);
        assert_eq!(compositions(2).len(), 10);
        assert_eq!(compositions(3).len(), 20);
        assert_eq!(compositions(4).len(), 35);
    }
}
