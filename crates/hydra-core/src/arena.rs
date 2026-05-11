//! Self-play arena: batch game simulation with trajectory collection.

use crate::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use crate::encoder::OBS_SIZE;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryExitLabel {
    pub target: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryDeltaQLabel {
    pub target: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],
}

fn label_from_slices(
    target: &[f32],
    mask: &[f32],
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])> {
    if target.len() != HYDRA_ACTION_SPACE || mask.len() != HYDRA_ACTION_SPACE {
        return None;
    }
    let mut target_arr = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask_arr = [0.0f32; HYDRA_ACTION_SPACE];
    target_arr.copy_from_slice(target);
    mask_arr.copy_from_slice(mask);
    Some((target_arr, mask_arr))
}

fn label_to_vec_pair(
    target: [f32; HYDRA_ACTION_SPACE],
    mask: [f32; HYDRA_ACTION_SPACE],
) -> (Vec<f32>, Vec<f32>) {
    (target.to_vec(), mask.to_vec())
}

fn masked_softmax_probs(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
) -> [f32; HYDRA_ACTION_SPACE] {
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
    probs
}

impl TrajectoryDeltaQLabel {
    pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
        let (target, mask) = label_from_slices(target, mask)?;
        Some(Self { target, mask })
    }

    pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
        (self.target, self.mask)
    }

    pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
        label_to_vec_pair(self.target, self.mask)
    }
}

impl TrajectoryExitLabel {
    pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
        let (target, mask) = label_from_slices(target, mask)?;
        Some(Self { target, mask })
    }

    pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
        (self.target, self.mask)
    }

    pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
        label_to_vec_pair(self.target, self.mask)
    }
}

pub struct ArenaConfig {
    pub num_parallel_games: usize,
    pub game_mode: u8,
    pub temperature_range: (f32, f32),
    pub exit_fraction: f32,
    pub max_trajectory_buffer: usize,
}

impl ArenaConfig {
    pub fn summary(&self) -> String {
        format!(
            "arena(games={}, temp={:.1}-{:.1}, buf={})",
            self.num_parallel_games,
            self.temperature_range.0,
            self.temperature_range.1,
            self.max_trajectory_buffer
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_parallel_games == 0 {
            return Err("num_parallel_games > 0");
        }
        if self.max_trajectory_buffer == 0 {
            return Err("max_trajectory_buffer > 0");
        }
        if self.temperature_range.0 <= 0.0 {
            return Err("temperature range start > 0");
        }
        if self.temperature_range.1 < self.temperature_range.0 {
            return Err("temperature range end >= start");
        }
        Ok(())
    }
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

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.arena.validate()?;
        if self.gae_gamma <= 0.0 || self.gae_gamma >= 1.0 {
            return Err("gae_gamma in (0,1)");
        }
        if self.gae_lambda <= 0.0 || self.gae_lambda >= 1.0 {
            return Err("gae_lambda in (0,1)");
        }
        Ok(())
    }
}

impl SelfPlayConfig {
    pub fn with_games(mut self, n: usize) -> Self {
        self.arena.num_parallel_games = n;
        self
    }

    pub fn summary(&self) -> String {
        format!(
            "selfplay(games={}, gamma={:.3}, rebase={:.0}h)",
            self.arena.num_parallel_games, self.gae_gamma, self.rebase_interval_hours
        )
    }
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
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub exit_label: Option<TrajectoryExitLabel>,
    pub delta_q_label: Option<TrajectoryDeltaQLabel>,
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

impl Trajectory {
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn active_players(&self) -> Vec<u8> {
        let mut seen = [false; 4];
        for s in &self.steps {
            if (s.player_id as usize) < 4 {
                seen[s.player_id as usize] = true;
            }
        }
        (0..4).filter(|&i| seen[i as usize]).collect()
    }

    pub fn score_delta(&self, player: u8) -> i32 {
        let mean = self.final_scores.iter().sum::<i32>() / 4;
        self.score_for(player) - mean
    }

    pub fn score_for(&self, player: u8) -> i32 {
        self.final_scores.get(player as usize).copied().unwrap_or(0)
    }

    pub fn placement_for(&self, player: u8) -> u8 {
        compute_placements(self.final_scores)[player as usize]
    }

    pub fn winner(&self) -> u8 {
        compute_placements(self.final_scores)
            .iter()
            .position(|&p| p == 0)
            .unwrap_or(0) as u8
    }

    pub fn max_turn(&self) -> u16 {
        self.steps.last().map_or(0, |s| s.turn)
    }

    pub fn player_reward_sum(&self, player_id: u8) -> f32 {
        self.steps
            .iter()
            .filter(|s| s.player_id == player_id)
            .map(|s| s.reward)
            .sum()
    }

    pub fn total_reward(&self) -> f32 {
        self.steps.iter().map(|s| s.reward).sum()
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

    pub fn validate(&self) -> Result<(), String> {
        if self.steps.is_empty() {
            return Err("trajectory has no steps".into());
        }
        for (i, step) in self.steps.iter().enumerate() {
            if step.player_id >= 4 {
                return Err(format!("step {i}: invalid player_id {}", step.player_id));
            }
            if step.action as usize >= HYDRA_ACTION_SPACE {
                return Err(format!("step {i}: invalid action {}", step.action));
            }
            if !step.legal_mask.iter().any(|&is_legal| is_legal) {
                return Err(format!("step {i}: legal_mask has no legal actions"));
            }
            if !step.legal_mask[step.action as usize] {
                return Err(format!(
                    "step {i}: selected action {} is not marked legal",
                    step.action
                ));
            }
            let pi_sum: f32 = step.pi_old.iter().sum();
            if pi_sum > 0.0 && (pi_sum - 1.0).abs() > 0.05 {
                return Err(format!("step {i}: pi_old sums to {pi_sum}"));
            }
            if let Some(exit_label) = step.exit_label {
                let mut masked_mass = 0.0f32;
                let mut saw_masked_action = false;
                for action_idx in 0..HYDRA_ACTION_SPACE {
                    let mask_value = exit_label.mask[action_idx];
                    if mask_value < -1e-6 || (mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6 {
                        return Err(format!(
                            "step {i}: exit mask at action {action_idx} is not approximately binary ({mask_value})"
                        ));
                    }
                    let target_value = exit_label.target[action_idx];
                    if target_value < -1e-6 {
                        return Err(format!(
                            "step {i}: exit target at action {action_idx} is negative ({target_value})"
                        ));
                    }
                    if mask_value > 0.5 {
                        saw_masked_action = true;
                        if !step.legal_mask[action_idx] {
                            return Err(format!(
                                "step {i}: exit label masks illegal action {action_idx}"
                            ));
                        }
                        if action_idx > DISCARD_END as usize {
                            return Err(format!(
                                "step {i}: exit label masks non-discard action {action_idx}"
                            ));
                        }
                        if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
                            return Err(format!(
                                "step {i}: exit label includes aka discard action {action_idx}"
                            ));
                        }
                        masked_mass += target_value;
                    } else if target_value.abs() > 1e-5 {
                        return Err(format!(
                            "step {i}: exit target has non-zero mass outside mask at action {action_idx}"
                        ));
                    }
                }
                if !saw_masked_action {
                    return Err(format!("step {i}: exit label mask is empty"));
                }
                if (masked_mass - 1.0).abs() > 1e-3 {
                    return Err(format!(
                        "step {i}: exit target mass over masked actions is {masked_mass}"
                    ));
                }
            }
            if let Some(delta_q_label) = step.delta_q_label {
                let mut saw_masked_action = false;
                for action_idx in 0..HYDRA_ACTION_SPACE {
                    let mask_value = delta_q_label.mask[action_idx];
                    if mask_value < -1e-6 || (mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6 {
                        return Err(format!(
                            "step {i}: delta_q mask at action {action_idx} is not approximately binary ({mask_value})"
                        ));
                    }
                    let target_value = delta_q_label.target[action_idx];
                    if !target_value.is_finite() {
                        return Err(format!(
                            "step {i}: delta_q target at action {action_idx} is not finite ({target_value})"
                        ));
                    }
                    if mask_value > 0.5 {
                        saw_masked_action = true;
                        if !step.legal_mask[action_idx] {
                            return Err(format!(
                                "step {i}: delta_q label masks illegal action {action_idx}"
                            ));
                        }
                        if action_idx > DISCARD_END as usize {
                            return Err(format!(
                                "step {i}: delta_q label masks non-discard action {action_idx}"
                            ));
                        }
                        if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
                            return Err(format!(
                                "step {i}: delta_q label includes aka discard action {action_idx}"
                            ));
                        }
                    } else if target_value.abs() > 1e-5 {
                        return Err(format!(
                            "step {i}: delta_q target has non-zero value outside mask at action {action_idx}"
                        ));
                    }
                }
                if !saw_masked_action {
                    return Err(format!("step {i}: delta_q label mask is empty"));
                }
            }
        }
        Ok(())
    }
}

pub fn softmax_temperature(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
) -> [f32; HYDRA_ACTION_SPACE] {
    masked_softmax_probs(logits, legal_mask, temperature)
}

pub fn games_played(scores: &[[i32; 4]]) -> usize {
    scores.len()
}

pub fn total_score_sum(scores: &[[i32; 4]]) -> i64 {
    scores
        .iter()
        .flat_map(|s| s.iter())
        .map(|&s| s as i64)
        .sum()
}

pub fn score_std(scores: &[[i32; 4]], player: u8) -> f32 {
    let mean = avg_score(scores, player);
    if scores.is_empty() {
        return 0.0;
    }
    let var: f32 = scores
        .iter()
        .map(|s| (s[player as usize] as f32 - mean).powi(2))
        .sum::<f32>()
        / scores.len() as f32;
    var.sqrt()
}

pub fn avg_score(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    scores
        .iter()
        .map(|s| s[player as usize] as f32)
        .sum::<f32>()
        / scores.len() as f32
}

pub fn top_two_rate(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let top2 = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] <= 1)
        .count();
    top2 as f32 / scores.len() as f32
}

pub fn fourth_place_rate(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let fourths = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] == 3)
        .count();
    fourths as f32 / scores.len() as f32
}

pub fn win_rate_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let wins = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] == 0)
        .count();
    wins as f32 / scores.len() as f32
}

pub fn mean_placement_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 2.5;
    }
    let sum: f32 = scores
        .iter()
        .map(|s| compute_placements(*s)[player as usize] as f32 + 1.0)
        .sum();
    sum / scores.len() as f32
}

pub fn compute_placements(scores: [i32; 4]) -> [u8; 4] {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut placements = [0u8; 4];
    for (rank, &(_, player)) in indexed.iter().enumerate() {
        placements[player as usize] = rank as u8;
    }
    placements
}

pub fn greedy_action(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> u8 {
    let mut best = 0u8;
    let mut best_val = f32::NEG_INFINITY;
    for (i, (&l, &m)) in logits.iter().zip(legal_mask.iter()).enumerate() {
        if m && l > best_val {
            best_val = l;
            best = i as u8;
        }
    }
    best
}

pub fn sample_action_with_temperature(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
    rng_val: f32,
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let probs = masked_softmax_probs(logits, legal_mask, temperature);
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
mod tests;
