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

    let target_presence = batch
        .targets
        .target_presence
        .as_ref()
        .expect("RL self-play batches should cache target presence metadata");
    assert_eq!(target_presence.batch_size, 2);
    assert_eq!(target_presence.count(AdvancedHead::DeltaQ), 2);
    assert_eq!(target_presence.delta_q_actions_present, 2);
}

#[test]
fn trajectories_to_rl_batch_reuse_matches_fresh_and_does_not_leak_state() {
    let device = Default::default();
    let mut trajectory_a = Trajectory::new(1, 42);
    trajectory_a.final_scores = [32000, 24000, 22000, 22000];
    trajectory_a.steps.push(test_step(0, 5, 0.2, false, 0));
    trajectory_a.steps.push(test_step(1, 9, -0.1, true, 1));
    let mut trajectory_a_reuse = Trajectory::new(1, 42);
    trajectory_a_reuse.final_scores = [32000, 24000, 22000, 22000];
    trajectory_a_reuse
        .steps
        .push(test_step(0, 5, 0.2, false, 0));
    trajectory_a_reuse
        .steps
        .push(test_step(1, 9, -0.1, true, 1));

    let mut trajectory_b = Trajectory::new(2, 77);
    trajectory_b.final_scores = [18000, 26000, 28000, 28000];
    trajectory_b.steps.push(test_step(2, 11, 0.3, false, 0));
    trajectory_b.steps.push(test_step(2, 12, -0.2, true, 1));
    let mut trajectory_b_reuse = Trajectory::new(2, 77);
    trajectory_b_reuse.final_scores = [18000, 26000, 28000, 28000];
    trajectory_b_reuse
        .steps
        .push(test_step(2, 11, 0.3, false, 0));
    trajectory_b_reuse
        .steps
        .push(test_step(2, 12, -0.2, true, 1));

    let mut scratch = RlBatchScratch::default();

    let batch_a_reuse = trajectories_to_rl_batch_reuse::<TestBackend>(
        &[trajectory_a_reuse],
        &[vec![0.3, -0.2]],
        &default_gae_config(),
        &device,
        &mut scratch,
    );
    let batch_a_fresh = trajectories_to_rl_batch::<TestBackend>(
        &[trajectory_a],
        &[vec![0.3, -0.2]],
        &default_gae_config(),
        &device,
    );
    assert_eq!(batch_a_reuse.obs.to_data(), batch_a_fresh.obs.to_data());
    assert_eq!(
        batch_a_reuse.actions.to_data(),
        batch_a_fresh.actions.to_data()
    );
    assert_eq!(
        batch_a_reuse.pi_old.to_data(),
        batch_a_fresh.pi_old.to_data()
    );
    assert_eq!(
        batch_a_reuse.advantages.to_data(),
        batch_a_fresh.advantages.to_data()
    );
    assert_eq!(
        batch_a_reuse.targets.policy_target.to_data(),
        batch_a_fresh.targets.policy_target.to_data()
    );

    let batch_b_reuse = trajectories_to_rl_batch_reuse::<TestBackend>(
        &[trajectory_b_reuse],
        &[vec![0.1, 0.05]],
        &default_gae_config(),
        &device,
        &mut scratch,
    );
    let batch_b_fresh = trajectories_to_rl_batch::<TestBackend>(
        &[trajectory_b],
        &[vec![0.1, 0.05]],
        &default_gae_config(),
        &device,
    );
    assert_eq!(batch_b_reuse.obs.to_data(), batch_b_fresh.obs.to_data());
    assert_eq!(
        batch_b_reuse.actions.to_data(),
        batch_b_fresh.actions.to_data()
    );
    assert_eq!(
        batch_b_reuse.pi_old.to_data(),
        batch_b_fresh.pi_old.to_data()
    );
    assert_eq!(
        batch_b_reuse.advantages.to_data(),
        batch_b_fresh.advantages.to_data()
    );
    assert_eq!(
        batch_b_reuse.targets.policy_target.to_data(),
        batch_b_fresh.targets.policy_target.to_data()
    );
}
