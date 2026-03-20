use burn::prelude::*;

use crate::config::{GAE_GAMMA, GAE_LAMBDA};
use crate::training::exit::{collate_delta_q_targets, collate_exit_targets};
use crate::training::gae::{GaeConfig, compute_per_player_gae, normalize_advantages};
use crate::training::losses::HydraTargets;
use crate::training::rl::RlBatch;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::{Trajectory, TrajectoryDeltaQLabel, TrajectoryExitLabel};
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};

const SCORE_BINS: usize = 64;
const GRP_CLASSES: usize = 24;
const NUM_OPPONENTS: usize = 3;
const NUM_TILES: usize = 34;

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

fn compute_trajectory_advantages(
    trajectory: &Trajectory,
    values: &[f32],
    gae_config: &GaeConfig,
) -> Vec<f32> {
    let mut advantages = vec![0.0f32; trajectory.steps.len()];

    for player in 0..4u8 {
        let player_indices: Vec<usize> = trajectory
            .steps
            .iter()
            .enumerate()
            .filter_map(|(idx, step)| (step.player_id == player).then_some(idx))
            .collect();
        if player_indices.is_empty() {
            continue;
        }

        let mut player_rewards = Vec::with_capacity(player_indices.len());
        let mut player_values = Vec::with_capacity(player_indices.len() + 1);
        let dones = vec![false; player_indices.len() - 1]
            .into_iter()
            .chain(std::iter::once(true))
            .collect::<Vec<_>>();

        for &idx in &player_indices {
            let mut reward_row = [0.0f32; 4];
            reward_row[player as usize] = trajectory.steps[idx].reward;
            player_rewards.push(reward_row);

            let mut value_row = [0.0f32; 4];
            value_row[player as usize] = values.get(idx).copied().unwrap_or(0.0);
            player_values.push(value_row);
        }
        player_values.push([0.0; 4]);

        let player_advantages = compute_per_player_gae(
            &player_rewards,
            &player_values,
            &dones,
            gae_config.gamma,
            gae_config.lambda,
        );

        for (local_idx, &global_idx) in player_indices.iter().enumerate() {
            advantages[global_idx] = player_advantages[local_idx][player as usize];
        }
    }

    advantages
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
    device: &B::Device,
) -> RlBatch<B> {
    let total_steps: usize = trajectories
        .iter()
        .map(|trajectory| trajectory.steps.len())
        .sum();

    let mut obs_flat = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut actions = Vec::with_capacity(total_steps);
    let mut pi_old = Vec::with_capacity(total_steps);
    let mut advantages = Vec::with_capacity(total_steps);
    let mut legal_mask = Vec::with_capacity(total_steps * HYDRA_ACTION_SPACE);
    let mut policy_target = vec![0.0f32; total_steps * HYDRA_ACTION_SPACE];
    let mut value_target = Vec::with_capacity(total_steps);
    let mut grp_target = vec![0.0f32; total_steps * GRP_CLASSES];
    let tenpai_target = vec![0.0f32; total_steps * NUM_OPPONENTS];
    let danger_target = vec![0.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let danger_mask = vec![1.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let mut opp_next_target = vec![0.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let mut score_pdf_target = vec![0.0f32; total_steps * SCORE_BINS];
    let mut score_cdf_target = vec![0.0f32; total_steps * SCORE_BINS];
    let base_logits = vec![0.0f32; total_steps * HYDRA_ACTION_SPACE];
    let mut exit_samples = Vec::with_capacity(total_steps);
    let mut delta_q_samples = Vec::with_capacity(total_steps);

    let mut global_step = 0usize;
    for (trajectory_idx, trajectory) in trajectories.iter().enumerate() {
        let trajectory_values = values.get(trajectory_idx).map_or(&[][..], Vec::as_slice);
        let trajectory_advantages =
            compute_trajectory_advantages(trajectory, trajectory_values, gae_config);

        for (step_idx, step) in trajectory.steps.iter().enumerate() {
            obs_flat.extend_from_slice(&step.obs);
            actions.push(step.action as i32);
            pi_old.push(step.pi_old[step.action as usize]);
            advantages.push(trajectory_advantages[step_idx]);

            for action_idx in 0..HYDRA_ACTION_SPACE {
                legal_mask.push(if step.legal_mask[action_idx] {
                    1.0
                } else {
                    0.0
                });
            }
            exit_samples.push(step.exit_label.map(TrajectoryExitLabel::to_vec_pair));
            delta_q_samples.push(step.delta_q_label.map(TrajectoryDeltaQLabel::to_vec_pair));

            policy_target[global_step * HYDRA_ACTION_SPACE + step.action as usize] = 1.0;
            value_target.push(step.reward);

            let placement_class = trajectory.placement_for(step.player_id) as usize;
            if placement_class < GRP_CLASSES {
                grp_target[global_step * GRP_CLASSES + placement_class] = 1.0;
            }

            for opponent in 0..NUM_OPPONENTS {
                opp_next_target[global_step * NUM_OPPONENTS * NUM_TILES + opponent * NUM_TILES] =
                    1.0;
            }

            let score_bin = score_to_bin(trajectory.final_scores[step.player_id as usize]);
            score_pdf_target[global_step * SCORE_BINS + score_bin] = 1.0;
            for bin in score_bin..SCORE_BINS {
                score_cdf_target[global_step * SCORE_BINS + bin] = 1.0;
            }

            global_step += 1;
        }
    }

    normalize_advantages(&mut advantages);
    let (exit_target, exit_mask) = collate_exit_targets::<B>(&exit_samples, device);
    let (delta_q_target, delta_q_mask) = collate_delta_q_targets::<B>(&delta_q_samples, device);

    RlBatch {
        obs: Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device).reshape([
            total_steps,
            NUM_CHANNELS,
            NUM_TILES,
        ]),
        actions: Tensor::<B, 1, Int>::from_ints(actions.as_slice(), device),
        pi_old: Tensor::<B, 1>::from_floats(pi_old.as_slice(), device),
        advantages: Tensor::<B, 1>::from_floats(advantages.as_slice(), device),
        base_logits: Tensor::<B, 1>::from_floats(base_logits.as_slice(), device)
            .reshape([total_steps, HYDRA_ACTION_SPACE]),
        targets: HydraTargets {
            policy_target: Tensor::<B, 1>::from_floats(policy_target.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            legal_mask: Tensor::<B, 1>::from_floats(legal_mask.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            value_target: Tensor::<B, 1>::from_floats(value_target.as_slice(), device),
            grp_target: Tensor::<B, 1>::from_floats(grp_target.as_slice(), device)
                .reshape([total_steps, GRP_CLASSES]),
            tenpai_target: Tensor::<B, 1>::from_floats(tenpai_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS]),
            danger_target: Tensor::<B, 1>::from_floats(danger_target.as_slice(), device).reshape([
                total_steps,
                NUM_OPPONENTS,
                NUM_TILES,
            ]),
            danger_mask: Tensor::<B, 1>::from_floats(danger_mask.as_slice(), device).reshape([
                total_steps,
                NUM_OPPONENTS,
                NUM_TILES,
            ]),
            opp_next_target: Tensor::<B, 1>::from_floats(opp_next_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS, NUM_TILES]),
            score_pdf_target: Tensor::<B, 1>::from_floats(score_pdf_target.as_slice(), device)
                .reshape([total_steps, SCORE_BINS]),
            score_cdf_target: Tensor::<B, 1>::from_floats(score_cdf_target.as_slice(), device)
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
        },
        exit_target,
        exit_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::arena::{TrajectoryDeltaQLabel, TrajectoryExitLabel, TrajectoryStep};

    type TestBackend = NdArray<f32>;

    fn test_step(player_id: u8, action: u8, reward: f32, done: bool, turn: u16) -> TrajectoryStep {
        let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
        pi_old[action as usize] = 1.0;
        let mut legal_mask = [false; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = true;
        legal_mask[(action as usize + 1) % HYDRA_ACTION_SPACE] = true;
        TrajectoryStep {
            obs: [action as f32; OBS_SIZE],
            action,
            pi_old,
            legal_mask,
            exit_label: Some(
                TrajectoryExitLabel::from_slices(&pi_old, &pi_old).expect("valid exit label"),
            ),
            delta_q_label: Some(
                TrajectoryDeltaQLabel::from_slices(&pi_old, &pi_old).expect("valid delta q label"),
            ),
            reward,
            done,
            player_id,
            game_id: 0,
            turn,
            temperature: 1.0,
        }
    }

    #[test]
    fn finalize_rewards_splits_final_scores_by_player_step_count() {
        let mut trajectory = Trajectory::new(7, 99);
        trajectory.final_scores = [30000, 15000, 0, 0];
        trajectory.steps.push(test_step(0, 2, 0.0, false, 0));
        trajectory.steps.push(test_step(0, 3, 0.0, true, 1));
        trajectory.steps.push(test_step(1, 4, 0.0, true, 2));

        finalize_rewards(&mut trajectory);

        assert!((trajectory.steps[0].reward - 0.15).abs() < 1e-6);
        assert!((trajectory.steps[1].reward - 0.15).abs() < 1e-6);
        assert!((trajectory.steps[2].reward - 0.15).abs() < 1e-6);
    }

    #[test]
    fn score_to_bin_clamps_extreme_scores() {
        assert_eq!(score_to_bin(-100_000), 0);
        assert_eq!(score_to_bin(0), 32);
        assert_eq!(score_to_bin(999_999), SCORE_BINS - 1);
    }

    #[test]
    fn trajectories_to_rl_batch_preserves_shapes_masks_and_score_targets() {
        let device = Default::default();
        let mut trajectory = Trajectory::new(1, 42);
        trajectory.final_scores = [32000, 24000, 22000, 22000];
        trajectory.steps.push(test_step(0, 5, 0.2, false, 0));
        trajectory.steps.push(test_step(1, 9, -0.1, true, 1));

        let batch = trajectories_to_rl_batch::<TestBackend>(
            &[trajectory],
            &[vec![0.3, -0.2]],
            &default_gae_config(),
            &device,
        );

        assert_eq!(batch.batch_size(), 2);
        assert!(batch.shapes_consistent());
        assert_eq!(batch.obs.dims(), [2, NUM_CHANNELS, NUM_TILES]);
        assert_eq!(batch.targets.legal_mask.dims(), [2, HYDRA_ACTION_SPACE]);
        assert_eq!(batch.targets.score_pdf_target.dims(), [2, SCORE_BINS]);
        assert_eq!(batch.targets.score_cdf_target.dims(), [2, SCORE_BINS]);

        let actions = batch
            .actions
            .to_data()
            .as_slice::<i64>()
            .expect("int actions readable")
            .to_vec();
        assert_eq!(actions, vec![5, 9]);

        let policy_data = batch.targets.policy_target.to_data();
        let policy = policy_data
            .as_slice::<f32>()
            .expect("policy target readable");
        assert_eq!(policy[5], 1.0);
        assert_eq!(policy[HYDRA_ACTION_SPACE + 9], 1.0);

        let score_pdf_data = batch.targets.score_pdf_target.to_data();
        let score_pdf = score_pdf_data
            .as_slice::<f32>()
            .expect("score pdf readable");
        let score_cdf_data = batch.targets.score_cdf_target.to_data();
        let score_cdf = score_cdf_data
            .as_slice::<f32>()
            .expect("score cdf readable");
        let expected_bin = score_to_bin(32_000);
        assert_eq!(score_pdf[expected_bin], 1.0);
        assert_eq!(score_cdf[expected_bin], 1.0);
        assert_eq!(score_cdf[SCORE_BINS - 1], 1.0);

        let opp_next_data = batch.targets.opp_next_target.to_data();
        let opp_next = opp_next_data.as_slice::<f32>().expect("opp_next readable");
        assert_eq!(opp_next[0], 1.0);
        assert_eq!(opp_next[NUM_TILES], 1.0);
        assert_eq!(opp_next[NUM_TILES * 2], 1.0);

        assert!(batch.exit_target.is_some());
        assert!(batch.exit_mask.is_some());
        assert!(batch.targets.delta_q_target.is_some());
        assert!(batch.targets.delta_q_mask.is_some());
    }
}
