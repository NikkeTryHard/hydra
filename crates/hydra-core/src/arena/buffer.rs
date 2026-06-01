use super::config::ArenaConfig;
use super::trajectory::{Trajectory, TrajectoryStep};

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

    pub fn max_capacity(&self) -> usize {
        self.config.max_trajectory_buffer
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

    pub fn validate_all(&self) -> Result<(), String> {
        for (i, traj) in self.trajectory_buffer.iter().enumerate() {
            traj.validate()
                .map_err(|e| format!("trajectory {i}: {e}"))?;
        }
        Ok(())
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
            scores_indexed.sort_by_key(|&(score, _)| std::cmp::Reverse(score));
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

    pub fn reset(&mut self) {
        self.trajectory_buffer.clear();
        self.games_completed = 0;
    }

    pub fn mean_score_for(&self, player: u8) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .trajectory_buffer
            .iter()
            .map(|t| t.score_for(player) as f32)
            .sum();
        sum / self.trajectory_buffer.len() as f32
    }

    pub fn score_variance(&self) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 0.0;
        }
        let means = self.mean_scores();
        let n = self.trajectory_buffer.len() as f32;
        let mut var = 0.0f32;
        for t in &self.trajectory_buffer {
            for (i, &s) in t.final_scores.iter().enumerate() {
                var += (s as f32 - means[i]).powi(2);
            }
        }
        var / (n * 4.0)
    }

    pub fn mean_game_length(&self) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 0.0;
        }
        let total_turns: u32 = self
            .trajectory_buffer
            .iter()
            .map(|t| t.max_turn() as u32)
            .sum();
        total_turns as f32 / self.trajectory_buffer.len() as f32
    }

    pub fn latest_game_id(&self) -> Option<u32> {
        self.trajectory_buffer.last().map(|t| t.game_id)
    }

    pub fn mean_placement_for(&self, player_id: u8) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 2.5;
        }
        let sum: f32 = self
            .trajectory_buffer
            .iter()
            .map(|t| t.placement_for(player_id) as f32 + 1.0)
            .sum();
        sum / self.trajectory_buffer.len() as f32
    }

    pub fn fourth_place_count(&self, player_id: u8) -> usize {
        self.trajectory_buffer
            .iter()
            .filter(|t| t.placement_for(player_id) == 3)
            .count()
    }

    pub fn win_rate_for(&self, player_id: u8) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 0.0;
        }
        self.win_count(player_id) as f32 / self.trajectory_buffer.len() as f32
    }

    pub fn win_count(&self, player_id: u8) -> usize {
        self.trajectory_buffer
            .iter()
            .filter(|t| t.winner() == player_id)
            .count()
    }

    pub fn oldest_game_id(&self) -> Option<u32> {
        self.trajectory_buffer.first().map(|t| t.game_id)
    }

    pub fn utilization(&self) -> String {
        format!(
            "{}/{} ({:.0}%)",
            self.num_buffered(),
            self.max_capacity(),
            self.fill_ratio() * 100.0
        )
    }

    pub fn fill_ratio(&self) -> f32 {
        if self.config.max_trajectory_buffer == 0 {
            return 0.0;
        }
        self.trajectory_buffer.len() as f32 / self.config.max_trajectory_buffer as f32
    }

    pub fn avg_trajectory_length(&self) -> f32 {
        if self.trajectory_buffer.is_empty() {
            return 0.0;
        }
        self.total_steps() as f32 / self.trajectory_buffer.len() as f32
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
