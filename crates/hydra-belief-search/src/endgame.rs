//! PIMC endgame solver for wall <= 10.

use crate::ct_smc::Particle;
use hydra_runtime_types::action::HYDRA_ACTION_SPACE;

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
    pub fn with_mass_threshold(mut self, t: f32) -> Self {
        self.mass_threshold = t;
        self
    }
    pub fn with_max_wall(mut self, w: u8) -> Self {
        self.max_wall = w;
        self
    }

    pub fn urgency(&self, wall: u8, danger: f32) -> f32 {
        if wall > self.max_wall {
            return 0.0;
        }
        let proximity = 1.0 - (wall as f32 / self.max_wall as f32);
        proximity * danger
    }

    pub fn tiles_remaining(&self, wall: u8) -> u8 {
        wall.min(self.max_wall)
    }

    pub fn is_active_wall(&self, wall: u8) -> bool {
        wall <= self.max_wall
    }

    pub fn summary(&self) -> String {
        format!(
            "endgame(wall<={}, mass>{:.0}%)",
            self.max_wall,
            self.mass_threshold * 100.0
        )
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
        if total <= 0.0 || (cumsum / total) as f32 >= threshold {
            break;
        }
    }
    result
}

fn normalized_particle_weights(particles: &[&Particle]) -> Vec<f32> {
    if particles.is_empty() {
        return Vec::new();
    }
    let max_w = particles
        .iter()
        .map(|p| p.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_w.is_finite() {
        let uniform = 1.0 / particles.len() as f32;
        return vec![uniform; particles.len()];
    }
    let weights: Vec<f64> = particles
        .iter()
        .map(|p| (p.log_weight - max_w).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        let uniform = 1.0 / particles.len() as f32;
        return vec![uniform; particles.len()];
    }
    weights.into_iter().map(|w| (w / total) as f32).collect()
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
    let weights = normalized_particle_weights(&selected);
    let mut q = [0.0f32; HYDRA_ACTION_SPACE];
    for a in 0..HYDRA_ACTION_SPACE {
        if !legal_mask[a] {
            continue;
        }
        q[a] = selected
            .iter()
            .zip(weights.iter())
            .map(|(p, &w)| eval_fn(p, a as u8) * w)
            .sum();
    }
    q
}

pub fn pimc_endgame_q(
    particles: &[Particle],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    eval_fn: &dyn Fn(&Particle, u8) -> f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut q = [0.0f32; HYDRA_ACTION_SPACE];
    if particles.is_empty() {
        return q;
    }
    let selected: Vec<&Particle> = particles.iter().collect();
    let weights = normalized_particle_weights(&selected);
    for a in 0..HYDRA_ACTION_SPACE {
        if !legal_mask[a] {
            continue;
        }
        q[a] = selected
            .iter()
            .zip(weights.iter())
            .map(|(p, &w)| eval_fn(p, a as u8) * w)
            .sum();
    }
    q
}

#[cfg(test)]
mod tests;
