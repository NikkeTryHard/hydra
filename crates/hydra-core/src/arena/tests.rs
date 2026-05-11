use super::*;

fn legal_step(action: u8, player_id: u8, reward: f32, done: bool, turn: u16) -> TrajectoryStep {
    let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
    pi_old[action as usize] = 1.0;
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    legal_mask[action as usize] = true;
    TrajectoryStep {
        obs: [0.0; OBS_SIZE],
        action,
        pi_old,
        legal_mask,
        exit_label: None,
        delta_q_label: None,
        reward,
        done,
        player_id,
        game_id: 0,
        turn,
        temperature: 1.0,
    }
}

#[test]
fn labels_roundtrip_and_reject_wrong_lengths() {
    let target = vec![0.25f32; HYDRA_ACTION_SPACE];
    let mask = vec![1.0f32; HYDRA_ACTION_SPACE];

    let exit = TrajectoryExitLabel::from_slices(&target, &mask).expect("valid exit label");
    let delta = TrajectoryDeltaQLabel::from_slices(&target, &mask).expect("valid delta q label");

    let (exit_target, exit_mask) = exit.to_vec_pair();
    let (delta_target, delta_mask) = delta.to_vec_pair();
    assert_eq!(exit_target, target);
    assert_eq!(exit_mask, mask);
    assert_eq!(delta_target, target);
    assert_eq!(delta_mask, mask);

    assert!(TrajectoryExitLabel::from_slices(&target[..10], &mask).is_none());
    assert!(TrajectoryDeltaQLabel::from_slices(&target, &mask[..10]).is_none());
}

#[test]
fn arena_and_selfplay_configs_validate_expected_bounds() {
    let mut arena_cfg = ArenaConfig::default();
    assert!(arena_cfg.validate().is_ok());
    assert!(arena_cfg.summary().contains("arena(games=500"));

    arena_cfg.num_parallel_games = 0;
    assert_eq!(arena_cfg.validate(), Err("num_parallel_games > 0"));

    let arena_cfg = ArenaConfig {
        max_trajectory_buffer: 0,
        ..ArenaConfig::default()
    };
    assert_eq!(arena_cfg.validate(), Err("max_trajectory_buffer > 0"));

    let mut selfplay = SelfPlayConfig::default().with_games(128);
    assert_eq!(selfplay.arena.num_parallel_games, 128);
    assert!(selfplay.summary().contains("selfplay(games=128"));
    assert!(selfplay.validate().is_ok());

    selfplay.gae_gamma = 1.0;
    assert_eq!(selfplay.validate(), Err("gae_gamma in (0,1)"));
}

#[test]
fn score_summary_helpers_handle_empty_and_ranked_games() {
    let scores = [
        [30_000, 25_000, 20_000, 15_000],
        [15_000, 30_000, 25_000, 20_000],
    ];
    assert_eq!(games_played(&scores), 2);
    assert_eq!(total_score_sum(&scores), 180_000);
    assert_eq!(avg_score(&scores, 0), 22_500.0);
    assert!(score_std(&scores, 0) > 0.0);
    assert_eq!(top_two_rate(&scores, 1), 1.0);
    assert_eq!(fourth_place_rate(&scores, 0), 0.5);
    assert_eq!(win_rate_from_scores(&scores, 0), 0.5);
    assert_eq!(mean_placement_from_scores(&scores, 0), 2.5);

    assert_eq!(avg_score(&[], 0), 0.0);
    assert_eq!(score_std(&[], 0), 0.0);
    assert_eq!(top_two_rate(&[], 0), 0.0);
    assert_eq!(fourth_place_rate(&[], 0), 0.0);
    assert_eq!(win_rate_from_scores(&[], 0), 0.0);
    assert_eq!(mean_placement_from_scores(&[], 0), 2.5);
}

#[test]
fn trajectory_and_arena_summary_helpers_compute_expected_values() {
    let mut t1 = Trajectory::new(7, 111);
    t1.final_scores = [30_000, 20_000, 25_000, 15_000];
    t1.steps.push(legal_step(0, 0, 1.5, false, 0));
    t1.steps.push(legal_step(1, 1, -0.5, true, 3));

    let mut t2 = Trajectory::new(8, 222);
    t2.final_scores = [15_000, 35_000, 25_000, 25_000];
    t2.steps.push(legal_step(2, 1, 2.0, true, 5));

    assert_eq!(t1.num_steps(), 2);
    assert_eq!(t1.active_players(), vec![0, 1]);
    assert_eq!(t1.score_for(2), 25_000);
    assert_eq!(t1.score_delta(0), 7_500);
    assert_eq!(t1.placement_for(0), 0);
    assert_eq!(t1.winner(), 0);
    assert_eq!(t1.max_turn(), 3);
    assert_eq!(t1.player_reward_sum(0), 1.5);
    assert_eq!(t1.total_reward(), 1.0);
    assert!(t1.is_complete());
    assert_eq!(t1.steps_for_player(1).len(), 1);

    let mut arena = Arena::new(ArenaConfig {
        max_trajectory_buffer: 4,
        ..Default::default()
    });
    arena.add_trajectory(t1);
    arena.add_trajectory(t2);

    assert_eq!(arena.max_capacity(), 4);
    assert!(!arena.is_full());
    assert_eq!(arena.completed_trajectories(), 2);
    assert_eq!(arena.total_steps(), 3);
    assert_eq!(arena.num_buffered(), 2);
    assert_eq!(arena.oldest_game_id(), Some(7));
    assert_eq!(arena.latest_game_id(), Some(8));
    assert_eq!(
        arena.mean_scores(),
        [22_500.0, 27_500.0, 25_000.0, 20_000.0]
    );
    assert_eq!(arena.mean_score_for(1), 27_500.0);
    assert!(arena.score_variance() > 0.0);
    assert_eq!(arena.mean_game_length(), 4.0);
    assert_eq!(arena.mean_placement_for(1), 2.0);
    assert_eq!(arena.fourth_place_count(0), 1);
    assert_eq!(arena.win_count(1), 1);
    assert_eq!(arena.win_rate_for(1), 0.5);
    assert_eq!(arena.fill_ratio(), 0.5);
    assert_eq!(arena.utilization(), "2/4 (50%)");
    assert_eq!(arena.avg_trajectory_length(), 1.5);
    assert!(
        arena
            .stats_summary()
            .contains("games=2 steps=3 buffered=2 complete=2")
    );
    assert_eq!(arena.collect_player_steps(1).len(), 2);
    assert_eq!(arena.compute_rewards(1), vec![vec![-0.5], vec![2.0]]);
    assert_eq!(arena.placement_distribution(1), [0.5, 0.0, 0.5, 0.0]);
    assert!(arena.validate_all().is_ok());

    arena.reset();
    assert_eq!(arena.games_completed, 0);
    assert!(arena.trajectory_buffer.is_empty());
}

#[test]
fn masked_softmax_and_sampling_fallback_handle_degenerate_inputs() {
    let logits = [0.0f32; HYDRA_ACTION_SPACE];
    let legal_mask = [false; HYDRA_ACTION_SPACE];
    let probs = softmax_temperature(&logits, &legal_mask, 1.0);
    assert!(probs.iter().all(|&p| p == 0.0));

    let mut single_legal = [false; HYDRA_ACTION_SPACE];
    single_legal[9] = true;
    let (action, probs) = sample_action_with_temperature(&logits, &single_legal, 1.0, 1.5);
    assert_eq!(action, 9);
    assert_eq!(probs[9], 1.0);
}

#[test]
fn trajectory_validate_rejects_bad_policy_and_label_shapes() {
    let mut traj = Trajectory::new(1, 2);
    let mut bad_step = legal_step(0, 0, 0.0, true, 0);
    bad_step.pi_old[0] = 0.7;
    bad_step.pi_old[1] = 0.7;
    traj.steps.push(bad_step);
    assert!(traj.validate().unwrap_err().contains("pi_old sums to"));

    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_mask[0] = 1.0;
    exit_mask[(DISCARD_END as usize) + 1] = 1.0;
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[0] = 0.5;
    exit_target[(DISCARD_END as usize) + 1] = 0.5;
    step.exit_label = Some(TrajectoryExitLabel {
        target: exit_target,
        mask: exit_mask,
    });
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("exit label masks illegal action")
    );

    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    let mut delta_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_mask[0] = 1.0;
    let mut delta_target = [0.0f32; HYDRA_ACTION_SPACE];
    delta_target[0] = f32::NAN;
    step.delta_q_label = Some(TrajectoryDeltaQLabel {
        target: delta_target,
        mask: delta_mask,
    });
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("delta_q target at action 0 is not finite")
    );
}

#[test]
fn config_validation_catches_temperature_and_lambda_bounds() {
    let arena_cfg = ArenaConfig {
        temperature_range: (0.0, 1.0),
        ..ArenaConfig::default()
    };
    assert_eq!(arena_cfg.validate(), Err("temperature range start > 0"));

    let arena_cfg = ArenaConfig {
        temperature_range: (1.2, 1.1),
        ..ArenaConfig::default()
    };
    assert_eq!(arena_cfg.validate(), Err("temperature range end >= start"));

    let selfplay = SelfPlayConfig {
        gae_lambda: 1.0,
        ..SelfPlayConfig::default()
    };
    assert_eq!(selfplay.validate(), Err("gae_lambda in (0,1)"));
}

#[test]
fn arena_helpers_cover_empty_defaults_and_drain_behavior() {
    let mut arena = Arena::new(ArenaConfig {
        max_trajectory_buffer: 2,
        ..Default::default()
    });

    assert_eq!(arena.mean_scores(), [0.0; 4]);
    assert_eq!(arena.placement_distribution(0), [0.25; 4]);
    assert!(arena.compute_rewards(0).is_empty());
    assert_eq!(arena.mean_score_for(0), 0.0);
    assert_eq!(arena.score_variance(), 0.0);
    assert_eq!(arena.mean_game_length(), 0.0);
    assert_eq!(arena.mean_placement_for(0), 2.5);
    assert_eq!(arena.win_rate_for(0), 0.0);
    assert_eq!(arena.win_count(0), 0);
    assert_eq!(arena.oldest_game_id(), None);
    assert_eq!(arena.latest_game_id(), None);
    assert_eq!(arena.fill_ratio(), 0.0);
    assert_eq!(arena.avg_trajectory_length(), 0.0);
    assert!(arena.collect_player_steps(0).is_empty());

    let mut t = Trajectory::new(1, 9);
    t.steps.push(legal_step(0, 0, 0.5, true, 0));
    arena.add_trajectory(t);
    let drained = arena.drain_trajectories();
    assert_eq!(drained.len(), 1);
    assert!(arena.trajectory_buffer.is_empty());
}

#[test]
fn trajectory_validate_rejects_no_legal_actions_illegal_choice_and_bad_exit_mass() {
    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    step.legal_mask = [false; HYDRA_ACTION_SPACE];
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("legal_mask has no legal actions")
    );

    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    step.legal_mask[0] = false;
    step.legal_mask[1] = true;
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("selected action 0 is not marked legal")
    );

    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_mask[0] = 1.0;
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[0] = 0.7;
    step.exit_label = Some(TrajectoryExitLabel {
        target: exit_target,
        mask: exit_mask,
    });
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("exit target mass over masked actions is 0.7")
    );
}

#[test]
fn trajectory_validate_rejects_bad_delta_masks_and_invalid_action_index() {
    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    let mut delta_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_mask[0] = 0.3;
    step.delta_q_label = Some(TrajectoryDeltaQLabel {
        target: [0.0; HYDRA_ACTION_SPACE],
        mask: delta_mask,
    });
    traj.steps.push(step);
    assert!(
        traj.validate()
            .unwrap_err()
            .contains("delta_q mask at action 0 is not approximately binary")
    );

    let mut traj = Trajectory::new(1, 2);
    let mut step = legal_step(0, 0, 0.0, true, 0);
    step.action = HYDRA_ACTION_SPACE as u8;
    traj.steps.push(step);
    assert!(traj.validate().unwrap_err().contains("invalid action"));
}

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
        legal_mask: {
            let mut mask = [false; HYDRA_ACTION_SPACE];
            mask[0] = true;
            mask
        },
        exit_label: None,
        delta_q_label: None,
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
        legal_mask: {
            let mut mask = [false; HYDRA_ACTION_SPACE];
            mask[7] = true;
            mask[45] = true;
            mask
        },
        exit_label: None,
        delta_q_label: None,
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
            legal_mask: {
                let mut mask = [false; HYDRA_ACTION_SPACE];
                mask[0] = true;
                mask
            },
            exit_label: None,
            delta_q_label: None,
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

#[test]
fn compute_placements_correct() {
    let p = compute_placements([40000, 30000, 20000, 10000]);
    assert_eq!(p, [0, 1, 2, 3]);
    let p2 = compute_placements([10000, 40000, 20000, 30000]);
    assert_eq!(p2, [3, 0, 2, 1]);
}

#[test]
fn greedy_picks_highest_legal() {
    let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
    logits[10] = 100.0;
    logits[20] = 50.0;
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[20] = true;
    mask[30] = true;
    let action = greedy_action(&logits, &mask);
    assert_eq!(action, 20, "should pick highest LEGAL action");
}

#[test]
fn trajectory_validate_catches_bad_player() {
    let mut traj = Trajectory::new(0, 0);
    traj.steps.push(TrajectoryStep {
        obs: [0.0; OBS_SIZE],
        action: 0,
        pi_old: {
            let mut p = [0.0; HYDRA_ACTION_SPACE];
            p[0] = 1.0;
            p
        },
        legal_mask: {
            let mut mask = [false; HYDRA_ACTION_SPACE];
            mask[0] = true;
            mask
        },
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: true,
        player_id: 5,
        game_id: 0,
        turn: 0,
        temperature: 1.0,
    });
    assert!(traj.validate().is_err());
}

#[test]
fn trajectory_validate_passes_good_data() {
    let mut traj = Trajectory::new(0, 0);
    traj.steps.push(TrajectoryStep {
        obs: [0.0; OBS_SIZE],
        action: 3,
        pi_old: {
            let mut p = [0.0; HYDRA_ACTION_SPACE];
            p[3] = 1.0;
            p
        },
        legal_mask: {
            let mut mask = [false; HYDRA_ACTION_SPACE];
            mask[3] = true;
            mask
        },
        exit_label: None,
        delta_q_label: None,
        reward: 0.0,
        done: true,
        player_id: 0,
        game_id: 0,
        turn: 0,
        temperature: 1.0,
    });
    assert!(traj.validate().is_ok());
}

#[test]
fn arena_500_games_completes() {
    let config = ArenaConfig {
        num_parallel_games: 500,
        max_trajectory_buffer: 600,
        ..Default::default()
    };
    let mut arena = Arena::new(config);
    for g in 0..500u32 {
        let mut traj = Trajectory::new(g, g as u64);
        for turn in 0..10u16 {
            traj.steps.push(TrajectoryStep {
                obs: [0.0; OBS_SIZE],
                action: (turn % 34) as u8,
                pi_old: {
                    let mut p = [0.0; HYDRA_ACTION_SPACE];
                    p[(turn % 34) as usize] = 1.0;
                    p
                },
                legal_mask: {
                    let mut mask = [false; HYDRA_ACTION_SPACE];
                    mask[(turn % 34) as usize] = true;
                    mask
                },
                exit_label: None,
                delta_q_label: None,
                reward: 0.0,
                done: turn == 9,
                player_id: (turn % 4) as u8,
                game_id: g,
                turn,
                temperature: 1.0,
            });
        }
        traj.final_scores = [25000; 4];
        arena.add_trajectory(traj);
    }
    assert_eq!(arena.games_completed, 500);
    assert!(arena.total_steps() >= 5000);
    assert!(arena.validate_all().is_ok());
}

#[test]
fn softmax_temperature_sums_to_one() {
    let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
    logits[0] = 3.0;
    logits[5] = 1.0;
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[5] = true;
    mask[10] = true;
    let probs = softmax_temperature(&logits, &mask, 1.0);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
}

#[test]
fn test_trajectory_empty_fails_validation() {
    let traj = Trajectory::new(0, 42);
    assert!(traj.steps.is_empty());
    let result = traj.validate();
    assert!(result.is_err(), "empty trajectory should fail validation");
}
