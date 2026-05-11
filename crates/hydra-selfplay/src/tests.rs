use super::*;
use crate::batch::default_gae_config;
use burn::backend::NdArray;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};
use hydra_model::model::{HydraModelConfig, HydraModelInit};
use hydra_train_types::config::{GAE_GAMMA, GAE_LAMBDA};

type B = NdArray<f32>;

fn discard_actions() -> [Action; 2] {
    [
        Action::new(ActionType::Discard, Some(0), &[], None),
        Action::new(ActionType::Discard, Some(4), &[], None),
    ]
}

fn make_test_trajectory() -> Trajectory {
    let mut trajectory = Trajectory::new(7, 999);
    trajectory.final_scores = [32000, 26000, 22000, 20000];
    for idx in 0..4u8 {
        let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
        pi_old[idx as usize] = 0.7;
        pi_old[(idx as usize + 1) % HYDRA_ACTION_SPACE] = 0.3;
        trajectory.steps.push(TrajectoryStep {
            obs: [idx as f32; OBS_SIZE],
            action: idx,
            pi_old,
            legal_mask: {
                let mut legal_mask = [false; HYDRA_ACTION_SPACE];
                legal_mask[idx as usize] = true;
                legal_mask[(idx as usize + 1) % HYDRA_ACTION_SPACE] = true;
                legal_mask
            },
            exit_label: None,
            delta_q_label: None,
            reward: if idx % 2 == 0 { 1.0 } else { -1.0 },
            done: idx == 3,
            player_id: idx % 4,
            game_id: 7,
            turn: idx as u16,
            temperature: 1.0,
        });
    }
    trajectory
}

fn small_test_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_hidden_channels(2)
        .with_se_bottleneck(1)
        .with_num_groups(1)
}

fn make_test_step_record(player_id: u8, action: u8) -> StepRecord {
    let mut policy_logits = [0.0f32; HYDRA_ACTION_SPACE];
    policy_logits[0] = 1.0;
    policy_logits[1] = 0.5;
    let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
    pi_old[0] = 0.6;
    pi_old[1] = 0.4;
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    legal_mask[0] = true;
    legal_mask[1] = true;
    StepRecord {
        obs: [player_id as f32; OBS_SIZE],
        action,
        policy_logits,
        pi_old,
        legal_mask,
        player_id,
    }
}

fn make_test_trajectory_step(player_id: u8, action: u8, turn: u16) -> TrajectoryStep {
    let record = make_test_step_record(player_id, action);
    TrajectoryStep {
        obs: record.obs,
        action: record.action,
        pi_old: record.pi_old,
        legal_mask: record.legal_mask,
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: false,
        player_id,
        game_id: 99,
        turn,
        temperature: 1.0,
    }
}

#[test]
fn test_nn_action_selector_selects_legal() {
    let mut selector = NnActionSelector::new(1.0, 42);
    selector.pending_obs = Some([0.0; OBS_SIZE]);
    selector.pending_context = Some(PendingContext {
        phase: ActionPhase::Normal,
        last_discard: None,
        hand: [0; 14],
        hand_len: 0,
    });
    let mut logits = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    logits[0] = 5.0;
    logits[1] = 1.0;
    selector.set_logits(logits);

    let legal_actions = discard_actions();
    let chosen = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        &mut selector,
        0,
        &legal_actions,
    );

    assert!(legal_actions.contains(&chosen));
}

#[test]
fn test_nn_action_selector_temperature() {
    let legal_actions = discard_actions();

    let mut low_temp = NnActionSelector::new(0.1, 7);
    low_temp.pending_obs = Some([0.0; OBS_SIZE]);
    low_temp.pending_context = Some(PendingContext {
        phase: ActionPhase::Normal,
        last_discard: None,
        hand: [0; 14],
        hand_len: 0,
    });
    let mut logits = [0.0; HYDRA_ACTION_SPACE];
    logits[0] = 3.0;
    logits[1] = 1.0;
    low_temp.set_logits(logits);
    let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        &mut low_temp,
        0,
        &legal_actions,
    );
    let low_probs = low_temp
        .take_last_step()
        .map(|step| step.pi_old)
        .unwrap_or([0.0; HYDRA_ACTION_SPACE]);

    let mut high_temp = NnActionSelector::new(10.0, 7);
    high_temp.pending_obs = Some([0.0; OBS_SIZE]);
    high_temp.pending_context = Some(PendingContext {
        phase: ActionPhase::Normal,
        last_discard: None,
        hand: [0; 14],
        hand_len: 0,
    });
    high_temp.set_logits(logits);
    let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        &mut high_temp,
        0,
        &legal_actions,
    );
    let high_probs = high_temp
        .take_last_step()
        .map(|step| step.pi_old)
        .unwrap_or([0.0; HYDRA_ACTION_SPACE]);

    assert!(low_probs[0] > high_probs[0]);
}

#[test]
fn test_step_record_captured() {
    let mut selector = NnActionSelector::new(1.0, 11);
    selector.pending_obs = Some([1.0; OBS_SIZE]);
    selector.pending_context = Some(PendingContext {
        phase: ActionPhase::Normal,
        last_discard: None,
        hand: [0; 14],
        hand_len: 0,
    });
    let mut logits = [0.0; HYDRA_ACTION_SPACE];
    logits[0] = 2.0;
    selector.set_logits(logits);

    let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        &mut selector,
        2,
        &discard_actions(),
    );
    let record = selector.take_last_step();

    assert!(record.is_some());
    let record = record.unwrap_or_else(|| unreachable!());
    assert_eq!(record.player_id, 2);
    assert_eq!(record.action, 0);
    assert_eq!(record.obs[0], 1.0);
    assert_eq!(record.policy_logits[0], 2.0);
    assert!(record.legal_mask[0]);
}

#[test]
fn test_run_self_play_game_basic() {
    let trajectory = run_self_play_game(42, 1.0, 123, |_| [0.0; HYDRA_ACTION_SPACE]);
    assert!(!trajectory.steps.is_empty());
    assert!(trajectory.is_complete());
    assert!(trajectory.validate().is_ok());
}

#[test]
fn test_generate_self_play_batch_source_without_exit() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);
    let seeds = [42u64];
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };

    let source = generate_self_play_batch_source(&seeds, 1.0, 40, &model, &device, cfg);

    assert_eq!(source.trajectories.len(), 1);
    assert_eq!(source.values.len(), 1);
    for (traj, vals) in source.trajectories.iter().zip(source.values.iter()) {
        assert_eq!(traj.steps.len(), vals.len());
        assert!(traj.steps.iter().all(|s| s.exit_label.is_none()));
    }
}

#[test]
fn test_batched_source_matches_serial() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);
    let seeds = [42u64];
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };

    let serial = generate_self_play_batch_source(&seeds, 1.0, 40, &model, &device, cfg.clone());
    let batched =
        generate_self_play_batch_source_batched(&seeds, 1.0, 40, &model, &device, &device, cfg);

    assert_eq!(serial.trajectories.len(), batched.trajectories.len());
    assert_eq!(serial.values.len(), batched.values.len());

    for (idx, (s_traj, b_traj)) in serial
        .trajectories
        .iter()
        .zip(batched.trajectories.iter())
        .enumerate()
    {
        assert_eq!(
            s_traj.steps.len(),
            b_traj.steps.len(),
            "trajectory {idx} step count mismatch"
        );
        assert_eq!(s_traj.seed, b_traj.seed, "trajectory {idx} seed mismatch");
        assert_eq!(
            s_traj.final_scores, b_traj.final_scores,
            "trajectory {idx} final_scores mismatch"
        );
        for (step_idx, (s_step, b_step)) in s_traj.steps.iter().zip(b_traj.steps.iter()).enumerate()
        {
            assert_eq!(
                s_step.action, b_step.action,
                "trajectory {idx} step {step_idx} action mismatch"
            );
            assert_eq!(
                s_step.player_id, b_step.player_id,
                "trajectory {idx} step {step_idx} player_id mismatch"
            );
            assert_eq!(
                s_step.legal_mask, b_step.legal_mask,
                "trajectory {idx} step {step_idx} legal_mask mismatch"
            );
            assert_eq!(
                s_step.exit_label.is_some(),
                b_step.exit_label.is_some(),
                "trajectory {idx} step {step_idx} exit_label presence mismatch"
            );
            assert_eq!(
                s_step.delta_q_label.is_some(),
                b_step.delta_q_label.is_some(),
                "trajectory {idx} step {step_idx} delta_q_label presence mismatch"
            );
        }
    }

    for (idx, (s_vals, b_vals)) in serial.values.iter().zip(batched.values.iter()).enumerate() {
        assert_eq!(
            s_vals.len(),
            b_vals.len(),
            "trajectory {idx} value count mismatch"
        );
        for (step_idx, (s_val, b_val)) in s_vals.iter().zip(b_vals.iter()).enumerate() {
            assert!(
                (s_val - b_val).abs() < 1e-5,
                "trajectory {idx} step {step_idx} value mismatch: serial={s_val} batched={b_val}"
            );
        }
    }
}

fn assert_batch_sources_match(expected: &SelfPlayBatchSource, actual: &SelfPlayBatchSource) {
    assert_eq!(expected.trajectories.len(), actual.trajectories.len());
    assert_eq!(expected.values.len(), actual.values.len());

    for (idx, (lhs, rhs)) in expected
        .trajectories
        .iter()
        .zip(actual.trajectories.iter())
        .enumerate()
    {
        assert_eq!(lhs.seed, rhs.seed, "trajectory {idx} seed mismatch");
        assert_eq!(
            lhs.final_scores, rhs.final_scores,
            "trajectory {idx} final score mismatch"
        );
        assert_eq!(
            lhs.steps.len(),
            rhs.steps.len(),
            "trajectory {idx} step count mismatch"
        );

        for (step_idx, (lhs_step, rhs_step)) in lhs.steps.iter().zip(rhs.steps.iter()).enumerate() {
            assert_eq!(
                lhs_step.action, rhs_step.action,
                "trajectory {idx} step {step_idx} action mismatch"
            );
            assert_eq!(
                lhs_step.player_id, rhs_step.player_id,
                "trajectory {idx} step {step_idx} player mismatch"
            );
            assert_eq!(
                lhs_step.turn, rhs_step.turn,
                "trajectory {idx} step {step_idx} turn mismatch"
            );
            assert_eq!(
                lhs_step.done, rhs_step.done,
                "trajectory {idx} step {step_idx} done mismatch"
            );
            assert_eq!(
                lhs_step.legal_mask, rhs_step.legal_mask,
                "trajectory {idx} step {step_idx} legal mask mismatch"
            );
            assert_eq!(
                lhs_step.exit_label, rhs_step.exit_label,
                "trajectory {idx} step {step_idx} exit label mismatch"
            );
            assert_eq!(
                lhs_step.delta_q_label, rhs_step.delta_q_label,
                "trajectory {idx} step {step_idx} delta q mismatch"
            );
            assert_eq!(
                lhs_step.obs, rhs_step.obs,
                "trajectory {idx} step {step_idx} obs mismatch"
            );
            assert_eq!(
                lhs_step.pi_old, rhs_step.pi_old,
                "trajectory {idx} step {step_idx} pi_old mismatch"
            );
        }
    }

    for (idx, (lhs, rhs)) in expected.values.iter().zip(actual.values.iter()).enumerate() {
        assert_eq!(lhs.len(), rhs.len(), "values {idx} len mismatch");
        for (step_idx, (&lhs_value, &rhs_value)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs_value - rhs_value).abs() < 1e-6,
                "values {idx} step {step_idx} mismatch: {lhs_value} vs {rhs_value}"
            );
        }
    }
}

#[test]
fn cooperative_reuse_matches_fresh_batches() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);
    let seeds = [42u64];
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };

    let fresh =
        generate_self_play_batch_source_cooperative(&seeds, 1.0, 4, &model, &device, cfg.clone());
    let mut coordinator = CooperativeSelfPlayCoordinator::new();
    let reused = generate_self_play_batch_source_cooperative_reuse(
        &mut coordinator,
        &seeds,
        1.0,
        4,
        &model,
        &device,
        cfg,
    );

    assert_batch_sources_match(&fresh, &reused);
}

#[test]
fn cooperative_reuse_does_not_leak_state_across_batches() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);
    let first_seeds = [100u64];
    let second_seeds = [200u64];
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };

    let mut coordinator = CooperativeSelfPlayCoordinator::new();
    let first_reused = generate_self_play_batch_source_cooperative_reuse(
        &mut coordinator,
        &first_seeds,
        0.85,
        7,
        &model,
        &device,
        cfg.clone(),
    );
    let second_reused = generate_self_play_batch_source_cooperative_reuse(
        &mut coordinator,
        &second_seeds,
        0.85,
        9,
        &model,
        &device,
        cfg.clone(),
    );

    let first_fresh = generate_self_play_batch_source_cooperative(
        &first_seeds,
        0.85,
        7,
        &model,
        &device,
        cfg.clone(),
    );
    let second_fresh =
        generate_self_play_batch_source_cooperative(&second_seeds, 0.85, 9, &model, &device, cfg);

    assert_batch_sources_match(&first_fresh, &first_reused);
    assert_batch_sources_match(&second_fresh, &second_reused);
    assert_eq!(
        second_reused
            .trajectories
            .iter()
            .map(|trajectory| trajectory.seed)
            .collect::<Vec<_>>(),
        second_seeds
    );
}

#[test]
fn mixed_policy_runner_matches_single_model_selfplay_when_all_seats_share_model() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);

    let trajectory = run_self_play_game(77, 1.0, 1234, |obs| model.policy_cpu(obs, &device));
    let scores =
        run_mixed_policy_game_scores(77, 1.0, 1234, [&model, &model, &model, &model], &device);

    assert_eq!(trajectory.final_scores, scores);
}

#[test]
fn test_generate_self_play_rl_batch_produces_valid_batch() {
    type AB = burn::backend::Autodiff<B>;
    let device = Default::default();
    let model = small_test_model_config().init::<AB>(&device);
    let seeds = [42u64];
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };
    let gae = GaeConfig {
        gamma: GAE_GAMMA,
        lambda: GAE_LAMBDA,
    };

    let batch = generate_self_play_rl_batch(&seeds, 1.0, 40, &model, &device, &gae, cfg);
    let [steps, action_dim] = batch.targets.policy_target.dims();
    assert!(steps > 0);
    assert_eq!(action_dim, HYDRA_ACTION_SPACE);
}

#[test]
fn cooperative_runner_waits_for_pending_exit_search_before_flushing_turn() {
    let mut runner = CooperativeGameRunner::new(42, 1.0, 123, LiveExitConfig::default());
    let mut tree = AfbsTree::new();
    let root = tree.add_node(7, 1.0, false);
    let mut turn_state = PendingTurnState::new(vec![0], 7);
    turn_state.next_index = turn_state.players.len();
    turn_state.pending_steps.push(None);
    turn_state.pending_values.push(0.25);
    runner.turn_state = Some(turn_state);
    runner.pending_exit_search = Some(ExitSearchState {
        steps: vec![PendingExitStep {
            step_record: make_test_step_record(0, 0),
            turn: 7,
            tree,
            root,
            base_pi: [0.0; HYDRA_ACTION_SPACE],
            legal_f32: [0.0; HYDRA_ACTION_SPACE],
            budget: 0,
            child_offset: 0,
            child_count: 1,
            output_index: 0,
        }],
        child_requests: vec![ExitChildRequest {
            child_idx: root,
            obs: [0.0; OBS_SIZE],
        }],
    });

    let advance = runner.advance_until_inference_needed();

    assert!(!advance.needs_policy);
    assert_eq!(runner.trajectory.steps.len(), 0);
    assert_eq!(runner.step_values.len(), 0);
    assert_eq!(runner.total_steps, 0);
    assert!(runner.turn_state.is_some());
    assert!(runner.has_pending_exit_search());
}

#[test]
fn cooperative_runner_finalizes_pending_exit_step_into_original_slot() {
    let mut runner = CooperativeGameRunner::new(42, 1.0, 123, LiveExitConfig::default());
    let preserved_a_action = 0u8;
    let preserved_b_action = 1u8;
    let mut turn_state = PendingTurnState::new(vec![0, 1, 2], 3);
    turn_state.pending_steps = vec![
        Some(make_test_trajectory_step(0, preserved_a_action, 3)),
        None,
        Some(make_test_trajectory_step(2, preserved_b_action, 3)),
    ];
    turn_state.pending_values = vec![0.1, 0.2, 0.3];
    runner.turn_state = Some(turn_state);

    let mut tree = AfbsTree::new();
    let root = tree.add_node(11, 1.0, false);
    let delayed_record = make_test_step_record(1, 1);
    runner.pending_exit_search = Some(ExitSearchState {
        steps: vec![PendingExitStep {
            step_record: delayed_record,
            turn: 3,
            tree,
            root,
            base_pi: [0.0; HYDRA_ACTION_SPACE],
            legal_f32: [0.0; HYDRA_ACTION_SPACE],
            budget: 0,
            child_offset: 0,
            child_count: 1,
            output_index: 1,
        }],
        child_requests: vec![ExitChildRequest {
            child_idx: root,
            obs: [0.0; OBS_SIZE],
        }],
    });

    runner.finalize_pending_exit_search(&[]);

    assert!(runner.pending_exit_search.is_none());
    let turn_state = runner.turn_state.as_ref().expect("turn state");
    assert_eq!(turn_state.pending_values, vec![0.1, 0.2, 0.3]);
    assert_eq!(
        turn_state.pending_steps[0].as_ref().map(|s| s.action),
        Some(preserved_a_action)
    );
    let inserted = turn_state.pending_steps[1]
        .as_ref()
        .expect("delayed step inserted");
    assert_eq!(inserted.action, delayed_record.action);
    assert_eq!(inserted.player_id, delayed_record.player_id);
    assert_eq!(inserted.turn, 3);
    assert!(inserted.exit_label.is_none());
    assert!(inserted.delta_q_label.is_none());
    assert_eq!(
        turn_state.pending_steps[2].as_ref().map(|s| s.action),
        Some(preserved_b_action)
    );
}

#[test]
fn test_trajectories_to_rl_batch_shapes() {
    let device = Default::default();
    let trajectory = make_test_trajectory();
    let values = vec![vec![0.1, 0.2, 0.3, 0.4]];
    let batch =
        trajectories_to_rl_batch::<B>(&[trajectory], &values, &default_gae_config(), &device);

    assert_eq!(batch.obs.dims(), [4, NUM_CHANNELS, NUM_TILES]);
    assert_eq!(batch.actions.dims(), [4]);
    assert_eq!(batch.pi_old.dims(), [4]);
    assert_eq!(batch.advantages.dims(), [4]);
    assert_eq!(batch.base_logits.dims(), [4, HYDRA_ACTION_SPACE]);
    assert_eq!(batch.targets.legal_mask.dims(), [4, HYDRA_ACTION_SPACE]);
    assert!(batch.exit_target.is_none());
    assert!(batch.exit_mask.is_none());
}

#[test]
fn test_trajectories_to_rl_batch_advantages_normalized() {
    let device = Default::default();
    let trajectory_a = make_test_trajectory();
    let mut trajectory_b = make_test_trajectory();
    trajectory_b.game_id = 8;
    trajectory_b.seed = 1000;
    trajectory_b.final_scores = [18000, 24000, 26000, 32000];

    let batch = trajectories_to_rl_batch::<B>(
        &[trajectory_a, trajectory_b],
        &[vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.3, 0.2, 0.1]],
        &default_gae_config(),
        &device,
    );

    let advantage_data = batch
        .advantages
        .to_data()
        .as_slice::<f32>()
        .expect("advantages")
        .to_vec();
    let mean = advantage_data.iter().sum::<f32>() / advantage_data.len() as f32;
    let variance = advantage_data
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / advantage_data.len() as f32;

    assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
    assert!(
        (variance - 1.0).abs() < 0.1,
        "variance should be ~1, got {variance}"
    );
}

#[test]
fn test_trajectories_to_rl_batch_collates_exit_labels() {
    let device = Default::default();
    let mut trajectory = make_test_trajectory();

    let exit_label = TrajectoryExitLabel::from_slices(
        &{
            let mut target = [0.0f32; HYDRA_ACTION_SPACE];
            target[0] = 0.625;
            target[1] = 0.375;
            target
        },
        &{
            let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
            mask[0] = 1.0;
            mask[1] = 1.0;
            mask
        },
    )
    .expect("valid exit label");

    trajectory.steps[0].exit_label = Some(exit_label);

    let batch = trajectories_to_rl_batch::<B>(
        &[trajectory],
        &[vec![0.1, 0.2, 0.3, 0.4]],
        &default_gae_config(),
        &device,
    );

    let exit_target = batch.exit_target.expect("exit target");
    let exit_mask = batch.exit_mask.expect("exit mask");
    assert_eq!(exit_target.dims(), [4, HYDRA_ACTION_SPACE]);
    assert_eq!(exit_mask.dims(), [4, HYDRA_ACTION_SPACE]);

    let target_data = exit_target
        .to_data()
        .as_slice::<f32>()
        .expect("exit target slice")
        .to_vec();
    let mask_data = exit_mask
        .to_data()
        .as_slice::<f32>()
        .expect("exit mask slice")
        .to_vec();

    assert!((target_data[0] - 0.625).abs() < 1e-6);
    assert!((target_data[1] - 0.375).abs() < 1e-6);
    assert_eq!(mask_data[0], 1.0);
    assert_eq!(mask_data[1], 1.0);

    let second_row_offset = HYDRA_ACTION_SPACE;
    assert!(
        target_data[second_row_offset..second_row_offset + HYDRA_ACTION_SPACE]
            .iter()
            .all(|value| value.abs() < 1e-6)
    );
    assert!(
        mask_data[second_row_offset..second_row_offset + HYDRA_ACTION_SPACE]
            .iter()
            .all(|value| value.abs() < 1e-6)
    );
}

#[test]
fn test_run_self_play_game_with_exit_labels_persists_hook_output() {
    let trajectory = run_self_play_game_with_exit_labels(
        42,
        1.0,
        123,
        |_| [0.0; HYDRA_ACTION_SPACE],
        |_, _, step, _, _| {
            let legal_f32 = step
                .legal_mask
                .map(|is_legal| if is_legal { 1.0 } else { 0.0 });
            if !hydra_search_labels::exit::compatible_discard_state(&legal_f32) {
                return None;
            }
            let legal_actions = step
                .legal_mask
                .iter()
                .enumerate()
                .filter_map(|(idx, &is_legal)| {
                    (is_legal && idx <= hydra_core::action::DISCARD_END as usize).then_some(idx)
                })
                .collect::<Vec<_>>();
            if legal_actions.len() < 2 {
                return None;
            }
            let mut target = [0.0f32; HYDRA_ACTION_SPACE];
            let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
            target[legal_actions[0]] = 0.5;
            target[legal_actions[1]] = 0.5;
            mask[legal_actions[0]] = 1.0;
            mask[legal_actions[1]] = 1.0;
            TrajectoryExitLabel::from_slices(&target, &mask).map(|exit| {
                hydra_search_labels::live_exit::TrajectorySearchLabels {
                    exit: Some(exit),
                    delta_q: None,
                }
            })
        },
    );

    assert!(
        trajectory
            .steps
            .iter()
            .any(|step| step.exit_label.is_some())
    );
    assert!(trajectory.validate().is_ok());
}
