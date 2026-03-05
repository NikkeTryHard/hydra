//! PIMC endgame solver for wall <= 10.

use crate::action::HYDRA_ACTION_SPACE;
use crate::ct_smc::Particle;

pub struct EndgameSolver {
    pub max_wall: u8,
    pub mass_threshold: f32,
}

impl EndgameSolver {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_wall == 0 {
            return Err("max_wall must be > 0");
        }
        if self.mass_threshold <= 0.0 || self.mass_threshold > 1.0 {
            return Err("mass_threshold must be in (0, 1]");
        }
        Ok(())
    }
}

impl Default for EndgameSolver {
    fn default() -> Self {
        Self {
            max_wall: 10,
            mass_threshold: 0.95,
        }
    }
}

impl EndgameSolver {
    pub fn new(max_wall: u8, mass_threshold: f32) -> Self {
        Self {
            max_wall,
            mass_threshold,
        }
    }
    pub fn wall_threshold(&self) -> u8 {
        self.max_wall
    }

    pub fn should_activate(&self, wall_remaining: u8, has_threat: bool) -> bool {
        wall_remaining <= self.max_wall && has_threat
    }

    pub fn solve_with_particles(
        &self,
        particles: &[Particle],
        legal_mask: &[bool; HYDRA_ACTION_SPACE],
        eval_fn: &dyn Fn(&Particle, u8) -> f32,
    ) -> [f32; HYDRA_ACTION_SPACE] {
        pimc_endgame_q_topk(particles, legal_mask, eval_fn, self.mass_threshold)
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

pub fn pimc_endgame_q_topk(
    particles: &[Particle],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    eval_fn: &dyn Fn(&Particle, u8) -> f32,
    mass_threshold: f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    let indices = top_mass_particles(particles, mass_threshold);
    if indices.is_empty() {
        return [0.0f32; HYDRA_ACTION_SPACE];
    }
    let selected: Vec<&Particle> = indices.iter().map(|&i| &particles[i]).collect();
    let n = selected.len() as f32;
    let mut q = [0.0f32; HYDRA_ACTION_SPACE];
    for a in 0..HYDRA_ACTION_SPACE {
        if !legal_mask[a] {
            continue;
        }
        let total: f32 = selected.iter().map(|p| eval_fn(p, a as u8)).sum();
        q[a] = total / n;
    }
    q
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

    #[test]
    fn endgame_empty_particles() {
        let mask = [true; HYDRA_ACTION_SPACE];
        let eval = |_: &Particle, _: u8| 1.0f32;
        let q = pimc_endgame_q(&[], &mask, &eval);
        assert!(q.iter().all(|&v| v == 0.0), "empty particles -> zero Q");
    }

    #[test]
    fn top_mass_empty_returns_empty() {
        assert!(top_mass_particles(&[], 0.95).is_empty());
    }

    #[test]
    fn endgame_respects_legal_mask() {
        let particles = vec![Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        }];
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[5] = true;
        let eval = |_: &Particle, a: u8| a as f32;
        let q = pimc_endgame_q(&particles, &mask, &eval);
        assert!((q[5] - 5.0).abs() < 1e-5);
        assert!(q[0] == 0.0, "illegal action should have Q=0");
        assert!(q[4] == 0.0, "illegal action should have Q=0");
    }

    #[test]
    fn endgame_with_weighted_particles() {
        let particles = vec![
            Particle {
                allocation: [[1; 4]; 34],
                log_weight: 0.0,
            },
            Particle {
                allocation: [[2; 4]; 34],
                log_weight: -5.0,
            },
        ];
        let selected = top_mass_particles(&particles, 0.95);
        assert!(selected.contains(&0));
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;
        let eval = |p: &Particle, a: u8| p.allocation[a as usize][0] as f32;
        let q = pimc_endgame_q(&particles, &mask, &eval);
        for i in 0..3 {
            assert!(q[i].is_finite(), "q[{i}] should be finite");
            assert!(q[i] > 0.0, "q[{i}] should be positive: {}", q[i]);
        }
    }
}
