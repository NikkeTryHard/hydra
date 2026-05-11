use super::*;

#[test]
fn test_gae_simple() {
    let rewards = [1.0, 0.0, 1.0, 0.0, 0.0];
    let values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.0];
    let dones = [false, false, false, false, true];
    let (adv, ret) = compute_gae(&rewards, &values, &dones, 0.99, 0.95);
    assert_eq!(adv.len(), 5);
    assert_eq!(ret.len(), 5);
    for (a, v) in adv.iter().zip(values.iter().take(5)) {
        let r = a + v;
        assert!((r - ret[adv.iter().position(|x| (x - a).abs() < 1e-10).unwrap()]).abs() < 1e-4);
    }
}

#[test]
fn test_gae_done_resets() {
    let rewards = [1.0, 2.0, 3.0];
    let values = [0.0, 0.0, 0.0, 0.0];
    let dones = [false, true, false];
    let (adv, _) = compute_gae(&rewards, &values, &dones, 0.99, 0.95);
    let (adv_nodone, _) = compute_gae(&rewards, &values, &[false; 3], 0.99, 0.95);
    assert!(
        (adv[0] - adv_nodone[0]).abs() > 0.01,
        "done should affect earlier steps"
    );
}

#[test]
fn test_normalize_advantages() {
    let mut adv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    normalize_advantages(&mut adv);
    let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
    assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    let var: f32 = adv.iter().map(|a| a.powi(2)).sum::<f32>() / adv.len() as f32;
    assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {var}");
}

#[test]
fn test_gae_hand_computed() {
    let gamma = 0.99f32;
    let lambda = 0.95f32;
    let rewards = [1.0, 2.0];
    let values = [0.5, 0.8, 0.3];
    let dones = [false, false];
    let (adv, _) = compute_gae(&rewards, &values, &dones, gamma, lambda);
    let delta1 = rewards[1] + gamma * values[2] - values[1];
    let gae1 = delta1;
    let delta0 = rewards[0] + gamma * values[1] - values[0];
    let gae0 = delta0 + gamma * lambda * gae1;
    assert!(
        (adv[0] - gae0).abs() < 1e-4,
        "adv[0]: {} vs {}",
        adv[0],
        gae0
    );
    assert!(
        (adv[1] - gae1).abs() < 1e-4,
        "adv[1]: {} vs {}",
        adv[1],
        gae1
    );
}

#[test]
fn test_gae_single_step_terminal() {
    let (adv, ret) = compute_gae(&[5.0], &[1.0, 0.0], &[true], 0.99, 0.95);
    assert!((adv[0] - 4.0).abs() < 1e-4, "terminal: adv={}", adv[0]);
    assert!((ret[0] - 5.0).abs() < 1e-4, "terminal: ret={}", ret[0]);
}

#[test]
fn test_rewards_from_scores() {
    let scores = [50000, -10000, 30000, -70000];
    let steps = [10, 10, 10, 10];
    let rewards = rewards_from_final_scores(scores, &steps);
    assert_eq!(rewards.len(), 10);
    assert!(
        (rewards[0][0] - 0.05).abs() < 1e-4,
        "player 0 reward per step"
    );
    assert!(
        (rewards[0][3] - (-0.07)).abs() < 1e-4,
        "player 3 reward per step"
    );
}

#[test]
fn test_per_player_gae_shape() {
    let rewards = vec![[1.0, -1.0, 0.5, -0.5]; 5];
    let values = vec![[0.0; 4]; 6];
    let dones = vec![false, false, false, false, true];
    let adv = compute_per_player_gae(&rewards, &values, &dones, 0.99, 0.95);
    assert_eq!(adv.len(), 5);
    for a in &adv {
        for &v in a {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn test_gae_config_defaults() {
    let cfg = GaeConfig::default();
    assert!((cfg.gamma - 0.995).abs() < 1e-6);
    assert!((cfg.lambda - 0.95).abs() < 1e-6);
}

#[test]
fn test_explained_variance_perfect() {
    let returns = [1.0f32, 2.0, 3.0];
    let predictions = [1.0f32, 2.0, 3.0];
    let ev = explained_variance(&returns, &predictions);
    assert!(
        (ev - 1.0).abs() < 1e-5,
        "perfect predictions should give EV ~1.0, got {ev}"
    );
}

#[test]
fn test_reward_std_constant() {
    let rewards = [5.0f32, 5.0, 5.0];
    let std = reward_std(&rewards);
    // Constant rewards: variance is 0, but epsilon 1e-8 is added
    assert!(
        std < 1e-3,
        "constant rewards should give std ~0 (epsilon only), got {std}"
    );
}

#[test]
fn test_gae_config_builder_summary_and_validate_bounds() {
    let cfg = GaeConfig::mahjong_defaults()
        .with_gamma(0.9)
        .with_lambda(0.8);

    assert_eq!(cfg.summary(), "gae(gamma=0.900, lambda=0.80)");
    assert!(cfg.validate().is_ok());

    assert_eq!(
        GaeConfig::default().with_gamma(0.0).validate(),
        Err("gamma in (0,1)")
    );
    assert_eq!(
        GaeConfig::default().with_gamma(1.0).validate(),
        Err("gamma in (0,1)")
    );
    assert_eq!(
        GaeConfig::default().with_lambda(0.0).validate(),
        Err("lambda in (0,1)")
    );
    assert_eq!(
        GaeConfig::default().with_lambda(1.0).validate(),
        Err("lambda in (0,1)")
    );
}

#[test]
fn test_advantage_helpers_discount_returns_and_zero_step_rewards() {
    let advantages = [-2.0f32, 0.5, 3.0];
    assert_eq!(advantage_range(&advantages), (-2.0, 3.0));
    assert!((mean_advantage(&advantages) - 0.5).abs() < 1e-6);

    let clipped = clipped_advantages(&advantages, 1.5);
    assert_eq!(clipped, vec![-1.5, 0.5, 1.5]);

    let discounted = discount_returns(&[1.0, 2.0, 3.0], 0.5);
    assert_eq!(discounted.len(), 3);
    assert!((discounted[0] - 2.75).abs() < 1e-6);
    assert!((discounted[1] - 3.5).abs() < 1e-6);
    assert!((discounted[2] - 3.0).abs() < 1e-6);

    let rewards = rewards_from_final_scores([25_000, -10_000, 5_000, 0], &[2, 0, 1, 0]);
    assert_eq!(rewards.len(), 2);
    assert!((rewards[0][0] - 0.125).abs() < 1e-6);
    assert_eq!(rewards[0][1], 0.0);
    assert!((rewards[0][2] - 0.05).abs() < 1e-6);
    assert_eq!(rewards[1][2], 0.0);
}

#[test]
fn test_explained_variance_handles_empty_and_constant_returns() {
    assert_eq!(explained_variance(&[], &[]), 0.0);

    let constant_returns = [4.0f32, 4.0, 4.0];
    let noisy_predictions = [1.0f32, 2.0, 3.0];
    assert_eq!(
        explained_variance(&constant_returns, &noisy_predictions),
        1.0
    );
}
