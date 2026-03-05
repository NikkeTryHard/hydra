//! Self-play arena: batch game simulation with trajectory collection.

use crate::action::HYDRA_ACTION_SPACE;
use crate::encoder::OBS_SIZE;

pub struct ArenaConfig {
    pub num_parallel_games: usize,
    pub game_mode: u8,
    pub temperature_range: (f32, f32),
    pub exit_fraction: f32,
    pub max_trajectory_buffer: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            num_parallel_games: 500,
            game_mode: 0,
            temperature_range: (0.5, 1.5),
            exit_fraction: 0.2,
            max_trajectory_buffer: 100_000,
        }
    }
}

pub struct SelfPlayConfig {
    pub arena: ArenaConfig,
    pub gae_gamma: f32,
    pub gae_lambda: f32,
    pub rebase_interval_hours: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            arena: ArenaConfig::default(),
            gae_gamma: 0.995,
            gae_lambda: 0.95,
            rebase_interval_hours: 37.5,
        }
    }
}

#[repr(C)]
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

pub struct Arena {
    pub config: ArenaConfig,
    pub trajectory_buffer: Vec<Trajectory>,
    pub games_completed: u64,
}

impl Arena {
    pub fn new(config: ArenaConfig) -> Self {
        Self {
            config,
            trajectory_buffer: Vec::new(),
            games_completed: 0,
        }
    }

    pub fn add_trajectory(&mut self, traj: Trajectory) {
        if self.trajectory_buffer.len() < self.config.max_trajectory_buffer {
            self.trajectory_buffer.push(traj);
        }
        self.games_completed += 1;
    }

    pub fn is_full(&self) -> bool {
        self.trajectory_buffer.len() >= self.config.max_trajectory_buffer
    }

    pub fn completed_trajectories(&self) -> usize {
        self.trajectory_buffer
            .iter()
            .filter(|t| t.is_complete())
            .count()
    }

    pub fn total_steps(&self) -> usize {
        self.trajectory_buffer.iter().map(|t| t.steps.len()).sum()
    }

    pub fn num_buffered(&self) -> usize {
        self.trajectory_buffer.len()
    }

    pub fn drain_trajectories(&mut self) -> Vec<Trajectory> {
        std::mem::take(&mut self.trajectory_buffer)
    }

    pub fn mean_scores(&self) -> [f32; 4] {
        if self.trajectory_buffer.is_empty() {
            return [0.0; 4];
        }
        let n = self.trajectory_buffer.len() as f32;
        let mut sums = [0.0f32; 4];
        for t in &self.trajectory_buffer {
            for (i, &s) in t.final_scores.iter().enumerate() {
                sums[i] += s as f32;
            }
        }
        for s in &mut sums {
            *s /= n;
        }
        sums
    }

    pub fn placement_distribution(&self, player_id: u8) -> [f32; 4] {
        if self.trajectory_buffer.is_empty() {
            return [0.25; 4];
        }
        let mut counts = [0u32; 4];
        let n = self.trajectory_buffer.len();
        for t in &self.trajectory_buffer {
            let mut scores_indexed: Vec<(i32, u8)> = t
                .final_scores
                .iter()
                .enumerate()
                .map(|(i, &s)| (s, i as u8))
                .collect();
            scores_indexed.sort_by(|a, b| b.0.cmp(&a.0));
            for (rank, (_, idx)) in scores_indexed.iter().enumerate() {
                if *idx == player_id && rank < 4 {
                    counts[rank] += 1;
                }
            }
        }
        let mut dist = [0.0f32; 4];
        for (i, &c) in counts.iter().enumerate() {
            dist[i] = c as f32 / n as f32;
        }
        dist
    }

    pub fn compute_rewards(&self, player_id: u8) -> Vec<Vec<f32>> {
        self.trajectory_buffer
            .iter()
            .map(|t| {
                t.steps
                    .iter()
                    .filter(|s| s.player_id == player_id)
                    .map(|s| s.reward)
                    .collect()
            })
            .collect()
    }

    pub fn stats_summary(&self) -> String {
        format!(
            "games={} steps={} buffered={} complete={}",
            self.games_completed,
            self.total_steps(),
            self.num_buffered(),
            self.completed_trajectories()
        )
    }

    pub fn collect_player_steps(&self, player_id: u8) -> Vec<&TrajectoryStep> {
        self.trajectory_buffer
            .iter()
            .flat_map(|t| t.steps.iter())
            .filter(|s| s.player_id == player_id)
            .collect()
    }
}

impl Trajectory {
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn is_complete(&self) -> bool {
        self.steps.last().is_some_and(|s| s.done)
    }

    pub fn steps_for_player(&self, player_id: u8) -> Vec<&TrajectoryStep> {
        self.steps
            .iter()
            .filter(|s| s.player_id == player_id)
            .collect()
    }

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

    #[test]
    fn trajectory_roundtrip() {
        let mut traj = Trajectory::new(42, 12345);
        traj.final_scores = [25000, 30000, 20000, 25000];
        traj.steps.push(TrajectoryStep {
            obs: [0.5; OBS_SIZE],
            action: 7,
            pi_old: {
                let mut p = [0.0; HYDRA_ACTION_SPACE];
                p[7] = 0.8;
                p[45] = 0.2;
                p
            },
            reward: 1.5,
            done: false,
            player_id: 2,
            game_id: 42,
            turn: 10,
            temperature: 0.8,
        });
        let step = &traj.steps[0];
        assert_eq!(step.action, 7);
        assert_eq!(step.player_id, 2);
        assert_eq!(step.turn, 10);
        assert!((step.reward - 1.5).abs() < 1e-5);
        assert!((step.temperature - 0.8).abs() < 1e-5);
        assert_eq!(traj.game_id, 42);
        assert_eq!(traj.seed, 12345);
        assert_eq!(traj.final_scores, [25000, 30000, 20000, 25000]);
        assert!((step.obs[0] - 0.5).abs() < 1e-5);
        assert!((step.pi_old[7] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn arena_trajectory_management() {
        let config = ArenaConfig {
            max_trajectory_buffer: 3,
            ..Default::default()
        };
        let mut arena = Arena::new(config);
        assert_eq!(arena.total_steps(), 0);
        for i in 0..5u32 {
            let mut t = Trajectory::new(i, i as u64);
            t.steps.push(TrajectoryStep {
                obs: [0.0; OBS_SIZE],
                action: 0,
                pi_old: [0.0; HYDRA_ACTION_SPACE],
                reward: 0.0,
                done: true,
                player_id: 0,
                game_id: i,
                turn: 0,
                temperature: 1.0,
            });
            arena.add_trajectory(t);
        }
        assert_eq!(arena.games_completed, 5);
        assert_eq!(arena.trajectory_buffer.len(), 3);
        let drained = arena.drain_trajectories();
        assert_eq!(drained.len(), 3);
        assert!(arena.trajectory_buffer.is_empty());
    }

    #[test]
    fn temperature_affects_distribution() {
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 3.0;
        logits[1] = 1.0;
        logits[2] = 0.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;
        let (_, probs_low) = sample_action_with_temperature(&logits, &mask, 0.1, 0.0);
        let (_, probs_high) = sample_action_with_temperature(&logits, &mask, 10.0, 0.0);
        assert!(
            probs_low[0] > probs_high[0],
            "low temp should concentrate: {:.3} vs {:.3}",
            probs_low[0],
            probs_high[0]
        );
    }

    #[test]
    fn single_legal_action_always_selected() {
        let logits = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[33] = true;
        for rng in [0.0, 0.5, 0.99] {
            let (action, probs) = sample_action_with_temperature(&logits, &mask, 1.0, rng);
            assert_eq!(action, 33);
            assert!((probs[33] - 1.0).abs() < 1e-5);
        }
    }
}
