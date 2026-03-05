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
}
