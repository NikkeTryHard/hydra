//! RL self-play batch collation helpers.

use burn::prelude::*;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::{Trajectory, TrajectoryDeltaQLabel, TrajectoryExitLabel};
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_search_labels::exit::{collate_delta_q_targets, collate_exit_targets};
use hydra_train_algo::gae::{GaeConfig, normalize_advantages};
use hydra_train_types::config::{GAE_GAMMA, GAE_LAMBDA};
use hydra_train_types::head_gates::{AdvancedHead, TargetPresence};
use hydra_train_types::losses::HydraTargets;
use hydra_train_types::rl::RlBatch;

const SCORE_BINS: usize = 64;
const GRP_CLASSES: usize = 24;
const NUM_OPPONENTS: usize = 3;
const NUM_TILES: usize = 34;

#[derive(Default)]
pub struct RlBatchScratch {
    obs_flat: Vec<f32>,
    actions: Vec<i32>,
    pi_old: Vec<f32>,
    advantages: Vec<f32>,
    trajectory_advantages: Vec<f32>,
    legal_mask: Vec<f32>,
    policy_target: Vec<f32>,
    value_target: Vec<f32>,
    grp_target: Vec<f32>,
    tenpai_target: Vec<f32>,
    danger_target: Vec<f32>,
    danger_mask: Vec<f32>,
    opp_next_target: Vec<f32>,
    score_pdf_target: Vec<f32>,
    score_cdf_target: Vec<f32>,
    base_logits: Vec<f32>,
    exit_samples: Vec<Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>>,
    delta_q_samples: Vec<Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>>,
    player_indices: Vec<usize>,
    player_rewards: Vec<f32>,
    player_values: Vec<f32>,
    dones: Vec<bool>,
    player_advantages: Vec<f32>,
}

pub fn finalize_rewards(trajectory: &mut Trajectory) {
    let mut steps_per_player = [0usize; 4];
    for step in &trajectory.steps {
        steps_per_player[step.player_id as usize] += 1;
    }

    for step in &mut trajectory.steps {
        let player = step.player_id as usize;
        let count = steps_per_player[player].max(1) as f32;
        step.reward = trajectory.final_scores[player] as f32 / 100_000.0 / count;
    }
}

fn compute_trajectory_advantages_reuse<'a>(
    trajectory: &Trajectory,
    values: &[f32],
    gae_config: &GaeConfig,
    scratch: &'a mut RlBatchScratch,
) -> &'a [f32] {
    scratch.trajectory_advantages.clear();
    scratch
        .trajectory_advantages
        .resize(trajectory.steps.len(), 0.0);

    for player in 0..4u8 {
        scratch.player_indices.clear();
        for (idx, step) in trajectory.steps.iter().enumerate() {
            if step.player_id == player {
                scratch.player_indices.push(idx);
            }
        }
        if scratch.player_indices.is_empty() {
            continue;
        }

        scratch.player_rewards.clear();
        scratch.player_rewards.reserve(scratch.player_indices.len());
        scratch.player_values.clear();
        scratch
            .player_values
            .reserve(scratch.player_indices.len() + 1);
        scratch.dones.clear();
        if scratch.player_indices.len() > 1 {
            scratch
                .dones
                .extend(std::iter::repeat_n(false, scratch.player_indices.len() - 1));
        }
        scratch.dones.push(true);

        for &idx in &scratch.player_indices {
            scratch.player_rewards.push(trajectory.steps[idx].reward);
            scratch
                .player_values
                .push(values.get(idx).copied().unwrap_or(0.0));
        }
        scratch.player_values.push(0.0);

        scratch.player_advantages.clear();
        scratch
            .player_advantages
            .resize(scratch.player_rewards.len(), 0.0);
        let mut gae = 0.0f32;
        for idx in (0..scratch.player_rewards.len()).rev() {
            let mask = if idx + 1 == scratch.player_rewards.len() {
                0.0
            } else {
                1.0
            };
            let delta = scratch.player_rewards[idx]
                + gae_config.gamma * scratch.player_values[idx + 1] * mask
                - scratch.player_values[idx];
            gae = delta + gae_config.gamma * gae_config.lambda * mask * gae;
            scratch.player_advantages[idx] = gae;
        }

        for (local_idx, &global_idx) in scratch.player_indices.iter().enumerate() {
            scratch.trajectory_advantages[global_idx] = scratch.player_advantages[local_idx];
        }
    }

    &scratch.trajectory_advantages
}

fn score_to_bin(score: i32) -> usize {
    let normalized = ((score as f32 / 1000.0) + 32.0).floor();
    normalized.clamp(0.0, (SCORE_BINS - 1) as f32) as usize
}

pub fn default_gae_config() -> GaeConfig {
    GaeConfig {
        gamma: GAE_GAMMA,
        lambda: GAE_LAMBDA,
    }
}

pub fn trajectories_to_rl_batch<B: Backend>(
    trajectories: &[Trajectory],
    values: &[Vec<f32>],
    gae_config: &GaeConfig,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> RlBatch<B> {
    let mut scratch = RlBatchScratch::default();
    trajectories_to_rl_batch_reuse(trajectories, values, gae_config, device, &mut scratch)
}

pub fn trajectories_to_rl_batch_reuse<B: Backend>(
    trajectories: &[Trajectory],
    values: &[Vec<f32>],
    gae_config: &GaeConfig,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    scratch: &mut RlBatchScratch,
) -> RlBatch<B> {
    let total_steps: usize = trajectories
        .iter()
        .map(|trajectory| trajectory.steps.len())
        .sum();

    scratch.obs_flat.clear();
    scratch.obs_flat.reserve(total_steps * OBS_SIZE);
    scratch.actions.clear();
    scratch.actions.reserve(total_steps);
    scratch.pi_old.clear();
    scratch.pi_old.reserve(total_steps);
    scratch.advantages.clear();
    scratch.advantages.reserve(total_steps);
    scratch.legal_mask.clear();
    scratch.legal_mask.reserve(total_steps * HYDRA_ACTION_SPACE);
    scratch.policy_target.clear();
    scratch
        .policy_target
        .resize(total_steps * HYDRA_ACTION_SPACE, 0.0);
    scratch.value_target.clear();
    scratch.value_target.reserve(total_steps);
    scratch.grp_target.clear();
    scratch.grp_target.resize(total_steps * GRP_CLASSES, 0.0);
    scratch.tenpai_target.clear();
    scratch
        .tenpai_target
        .resize(total_steps * NUM_OPPONENTS, 0.0);
    scratch.danger_target.clear();
    scratch
        .danger_target
        .resize(total_steps * NUM_OPPONENTS * NUM_TILES, 0.0);
    scratch.danger_mask.clear();
    scratch
        .danger_mask
        .resize(total_steps * NUM_OPPONENTS * NUM_TILES, 1.0);
    scratch.opp_next_target.clear();
    scratch
        .opp_next_target
        .resize(total_steps * NUM_OPPONENTS * NUM_TILES, 0.0);
    scratch.score_pdf_target.clear();
    scratch
        .score_pdf_target
        .resize(total_steps * SCORE_BINS, 0.0);
    scratch.score_cdf_target.clear();
    scratch
        .score_cdf_target
        .resize(total_steps * SCORE_BINS, 0.0);
    scratch.base_logits.clear();
    scratch
        .base_logits
        .resize(total_steps * HYDRA_ACTION_SPACE, 0.0);
    scratch.exit_samples.clear();
    scratch.exit_samples.reserve(total_steps);
    scratch.delta_q_samples.clear();
    scratch.delta_q_samples.reserve(total_steps);
    let mut target_presence = TargetPresence::with_batch_size(total_steps);

    let mut global_step = 0usize;
    for (trajectory_idx, trajectory) in trajectories.iter().enumerate() {
        let trajectory_values = values.get(trajectory_idx).map_or(&[][..], Vec::as_slice);
        compute_trajectory_advantages_reuse(trajectory, trajectory_values, gae_config, scratch);

        for (step_idx, step) in trajectory.steps.iter().enumerate() {
            scratch.obs_flat.extend_from_slice(&step.obs);
            scratch.actions.push(step.action as i32);
            scratch.pi_old.push(step.pi_old[step.action as usize]);
            scratch
                .advantages
                .push(scratch.trajectory_advantages[step_idx]);

            for action_idx in 0..HYDRA_ACTION_SPACE {
                scratch.legal_mask.push(if step.legal_mask[action_idx] {
                    1.0
                } else {
                    0.0
                });
            }
            scratch
                .exit_samples
                .push(step.exit_label.map(TrajectoryExitLabel::to_array_pair));
            let delta_q_sample = step.delta_q_label.map(TrajectoryDeltaQLabel::to_array_pair);
            if let Some((_, mask)) = delta_q_sample.as_ref() {
                let action_count = mask.iter().filter(|&&value| value > 0.0).count();
                if action_count > 0 {
                    target_presence.counts[AdvancedHead::DeltaQ.index()] += 1;
                    target_presence.delta_q_actions_present += action_count;
                }
            }
            scratch.delta_q_samples.push(delta_q_sample);

            scratch.policy_target[global_step * HYDRA_ACTION_SPACE + step.action as usize] = 1.0;
            scratch.value_target.push(step.reward);

            let placement_class = trajectory.placement_for(step.player_id) as usize;
            if placement_class < GRP_CLASSES {
                scratch.grp_target[global_step * GRP_CLASSES + placement_class] = 1.0;
            }

            for opponent in 0..NUM_OPPONENTS {
                scratch.opp_next_target
                    [global_step * NUM_OPPONENTS * NUM_TILES + opponent * NUM_TILES] = 1.0;
            }

            let score_bin = score_to_bin(trajectory.final_scores[step.player_id as usize]);
            scratch.score_pdf_target[global_step * SCORE_BINS + score_bin] = 1.0;
            for bin in score_bin..SCORE_BINS {
                scratch.score_cdf_target[global_step * SCORE_BINS + bin] = 1.0;
            }

            global_step += 1;
        }
    }

    normalize_advantages(&mut scratch.advantages);
    let (exit_target, exit_mask) = collate_exit_targets::<B>(&scratch.exit_samples, device);
    let (delta_q_target, delta_q_mask) =
        collate_delta_q_targets::<B>(&scratch.delta_q_samples, device);

    RlBatch {
        obs: Tensor::<B, 1>::from_floats(scratch.obs_flat.as_slice(), device).reshape([
            total_steps,
            NUM_CHANNELS,
            NUM_TILES,
        ]),
        actions: Tensor::<B, 1, Int>::from_ints(scratch.actions.as_slice(), device),
        pi_old: Tensor::<B, 1>::from_floats(scratch.pi_old.as_slice(), device),
        advantages: Tensor::<B, 1>::from_floats(scratch.advantages.as_slice(), device),
        base_logits: Tensor::<B, 1>::from_floats(scratch.base_logits.as_slice(), device)
            .reshape([total_steps, HYDRA_ACTION_SPACE]),
        targets: HydraTargets {
            policy_target: Tensor::<B, 1>::from_floats(scratch.policy_target.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            legal_mask: Tensor::<B, 1>::from_floats(scratch.legal_mask.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            value_target: Tensor::<B, 1>::from_floats(scratch.value_target.as_slice(), device),
            grp_target: Tensor::<B, 1>::from_floats(scratch.grp_target.as_slice(), device)
                .reshape([total_steps, GRP_CLASSES]),
            tenpai_target: Tensor::<B, 1>::from_floats(scratch.tenpai_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS]),
            danger_target: Tensor::<B, 1>::from_floats(scratch.danger_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS, NUM_TILES]),
            danger_mask: Tensor::<B, 1>::from_floats(scratch.danger_mask.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS, NUM_TILES]),
            opp_next_target: Tensor::<B, 1>::from_floats(
                scratch.opp_next_target.as_slice(),
                device,
            )
            .reshape([total_steps, NUM_OPPONENTS, NUM_TILES]),
            score_pdf_target: Tensor::<B, 1>::from_floats(
                scratch.score_pdf_target.as_slice(),
                device,
            )
            .reshape([total_steps, SCORE_BINS]),
            score_cdf_target: Tensor::<B, 1>::from_floats(
                scratch.score_cdf_target.as_slice(),
                device,
            )
            .reshape([total_steps, SCORE_BINS]),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target,
            delta_q_mask,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
            target_presence: Some(target_presence),
        },
        exit_target,
        exit_mask,
    }
}

#[cfg(test)]
mod tests;
