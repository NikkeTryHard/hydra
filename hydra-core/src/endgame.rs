//! PIMC endgame solver for wall <= 10.

use crate::action::HYDRA_ACTION_SPACE;
use crate::ct_smc::Particle;

pub struct EndgameSolver {
    pub max_wall: u8,
    pub mass_threshold: f32,
}

impl Default for EndgameSolver {
    fn default() -> Self {
        Self {
            max_wall: 10,
            mass_threshold: 0.95,
        }
    }
}

pub fn top_mass_particles(particles: &[Particle], threshold: f32) -> Vec<usize> {
    if particles.is_empty() {
        return Vec::new();
    }
    let max_w = particles
        .iter()
        .map(|p| p.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut indexed: Vec<(usize, f64)> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| (i, (p.log_weight - max_w).exp()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = indexed.iter().map(|(_, w)| w).sum();
    let mut cumsum = 0.0;
    let mut result = Vec::new();
    for (i, w) in &indexed {
        cumsum += w;
        result.push(*i);
        if (cumsum / total) as f32 >= threshold {
            break;
        }
    }
    result
}

pub fn pimc_endgame_q(
    particles: &[Particle],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    eval_fn: &dyn Fn(&Particle, u8) -> f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut q = [0.0f32; HYDRA_ACTION_SPACE];
    let n = particles.len() as f32;
    if n == 0.0 {
        return q;
    }
    for a in 0..HYDRA_ACTION_SPACE {
        if !legal_mask[a] {
            continue;
        }
        let total: f32 = particles.iter().map(|p| eval_fn(p, a as u8)).sum();
        q[a] = total / n;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endgame_returns_finite_q() {
        let particles = vec![
            Particle {
                allocation: [[0; 4]; 34],
                log_weight: 0.0,
            },
            Particle {
                allocation: [[0; 4]; 34],
                log_weight: 0.0,
            },
        ];
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[1] = true;
        let eval = |_: &Particle, a: u8| a as f32 * 0.1;
        let q = pimc_endgame_q(&particles, &mask, &eval);
        assert!(q[0].is_finite());
        assert!(q[1].is_finite());
        assert!((q[0] - 0.0).abs() < 1e-5);
        assert!((q[1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn top_mass_selects_heavy_particles() {
        let particles = vec![
            Particle {
                allocation: [[0; 4]; 34],
                log_weight: 0.0,
            },
            Particle {
                allocation: [[0; 4]; 34],
                log_weight: -10.0,
            },
            Particle {
                allocation: [[0; 4]; 34],
                log_weight: -0.1,
            },
        ];
        let selected = top_mass_particles(&particles, 0.95);
        assert!(selected.len() <= 3);
        assert!(
            selected.contains(&0),
            "highest weight particle should be selected"
        );
    }
}
