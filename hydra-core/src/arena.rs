//! Self-play arena: batch game simulation with trajectory collection.

use crate::action::HYDRA_ACTION_SPACE;
use crate::encoder::OBS_SIZE;

pub struct ArenaConfig {
    pub num_parallel_games: usize,
    pub temperature_range: (f32, f32),
    pub max_trajectory_buffer: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            num_parallel_games: 500,
            temperature_range: (0.5, 1.5),
            max_trajectory_buffer: 100_000,
        }
    }
}

pub struct TrajectoryStep {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub pi_old: [f32; HYDRA_ACTION_SPACE],
    pub reward: f32,
    pub done: bool,
    pub player_id: u8,
    pub game_id: u32,
    pub turn: u16,
    pub temperature: f32,
}

pub struct Trajectory {
    pub steps: Vec<TrajectoryStep>,
    pub final_scores: [i32; 4],
    pub game_id: u32,
    pub seed: u64,
}

impl Trajectory {
    pub fn new(game_id: u32, seed: u64) -> Self {
        Self {
            steps: Vec::new(),
            final_scores: [0; 4],
            game_id,
            seed,
        }
    }
}

pub fn sample_action_with_temperature(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
    rng_val: f32,
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            adjusted[i] = logits[i] / temperature;
            if adjusted[i] > max_val {
                max_val = adjusted[i];
            }
        }
    }
    let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            probs[i] = (adjusted[i] - max_val).exp();
            total += probs[i];
        }
    }
    if total > 0.0 {
        for p in &mut probs {
            *p /= total;
        }
    }
    let mut cumsum = 0.0f32;
    let mut chosen = 0u8;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_val <= cumsum {
            chosen = i as u8;
            break;
        }
    }
    if !legal_mask[chosen as usize] {
        for (i, &m) in legal_mask.iter().enumerate() {
            if m {
                chosen = i as u8;
                break;
            }
        }
    }
    (chosen, probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_sampling_legal_only() {
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 10.0;
        logits[1] = -10.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[1] = true;
        mask[2] = true;
        for rng in [0.0, 0.5, 0.99] {
            let (action, _) = sample_action_with_temperature(&logits, &mask, 1.0, rng);
            assert!(mask[action as usize], "selected illegal action {action}");
        }
    }

    #[test]
    fn trajectory_non_empty() {
        let mut traj = Trajectory::new(0, 42);
        traj.steps.push(TrajectoryStep {
            obs: [0.0; OBS_SIZE],
            action: 0,
            pi_old: [0.0; HYDRA_ACTION_SPACE],
            reward: 0.0,
            done: false,
            player_id: 0,
            game_id: 0,
            turn: 0,
            temperature: 1.0,
        });
        assert!(!traj.steps.is_empty());
    }
}
