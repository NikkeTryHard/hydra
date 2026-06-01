use crate::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use crate::encoder::OBS_SIZE;

use super::labels::{TrajectoryDeltaQLabel, TrajectoryExitLabel};
use super::scores::compute_placements;

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
