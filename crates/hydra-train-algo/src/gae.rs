//! Generalized Advantage Estimation (gamma=0.995, lambda=0.95).

/// Configuration for generalized advantage estimation.
pub struct GaeConfig {
    /// Discount factor applied to future rewards.
    pub gamma: f32,
    /// GAE trace decay factor.
    pub lambda: f32,
}

impl GaeConfig {
    /// Return this config with `gamma` replaced.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Return Hydra Mahjong defaults.
    pub fn mahjong_defaults() -> Self {
        Self::default()
    }

    /// Return a compact human-readable summary.
    pub fn summary(&self) -> String {
        format!("gae(gamma={:.3}, lambda={:.2})", self.gamma, self.lambda)
    }

    /// Validate discount and trace factors are in `(0, 1)`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.gamma <= 0.0 || self.gamma >= 1.0 {
            return Err("gamma in (0,1)");
        }
        if self.lambda <= 0.0 || self.lambda >= 1.0 {
            return Err("lambda in (0,1)");
        }
        Ok(())
    }

    /// Return this config with `lambda` replaced.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }
}

impl Default for GaeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.995,
            lambda: 0.95,
        }
    }
}

/// Compute single-trajectory advantages and returns.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    assert_eq!(values.len(), t + 1, "values must have T+1 entries");
    assert_eq!(dones.len(), t);
    let mut advantages = vec![0.0f32; t];
    let mut gae = 0.0f32;
    for i in (0..t).rev() {
        let mask = if dones[i] { 0.0 } else { 1.0 };
        let delta = rewards[i] + gamma * values[i + 1] * mask - values[i];
        gae = delta + gamma * lambda * mask * gae;
        advantages[i] = gae;
    }
    let returns: Vec<f32> = advantages
        .iter()
        .zip(values.iter().take(t))
        .map(|(a, v)| a + v)
        .collect();
    (advantages, returns)
}

/// Compute per-player advantages for four-player trajectories.
pub fn compute_per_player_gae(
    player_rewards: &[[f32; 4]],
    player_values: &[[f32; 4]],
    dones: &[bool],
    gamma: f32,
    lambda: f32,
) -> Vec<[f32; 4]> {
    let t = player_rewards.len();
    assert_eq!(player_values.len(), t + 1);
    let mut advantages = vec![[0.0f32; 4]; t];
    for p in 0..4 {
        let r: Vec<f32> = player_rewards.iter().map(|r| r[p]).collect();
        let v: Vec<f32> = player_values.iter().map(|v| v[p]).collect();
        let (adv, _) = compute_gae(&r, &v, dones, gamma, lambda);
        for (i, a) in adv.into_iter().enumerate() {
            advantages[i][p] = a;
        }
    }
    advantages
}

/// Compute single-player advantages, discarding returns.
pub fn compute_single_player_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    gamma: f32,
    lambda: f32,
) -> Vec<f32> {
    compute_gae(rewards, values, dones, gamma, lambda).0
}

/// Return the maximum advantage value.
pub fn max_advantage(advantages: &[f32]) -> f32 {
    advantages.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Return the minimum advantage value.
pub fn min_advantage(advantages: &[f32]) -> f32 {
    advantages.iter().cloned().fold(f32::INFINITY, f32::min)
}

/// Return population standard deviation of advantages with epsilon smoothing.
pub fn advantage_std(advantages: &[f32]) -> f32 {
    let n = advantages.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean = advantages.iter().sum::<f32>() / n;
    let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
    (var + 1e-8).sqrt()
}

/// Return mean advantage or zero for an empty slice.
pub fn mean_advantage(advantages: &[f32]) -> f32 {
    if advantages.is_empty() {
        return 0.0;
    }
    advantages.iter().sum::<f32>() / advantages.len() as f32
}

/// Return population standard deviation of rewards with epsilon smoothing.
pub fn reward_std(rewards: &[f32]) -> f32 {
    let mean = reward_mean(rewards);
    let n = rewards.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
    (var + 1e-8).sqrt()
}

/// Return mean reward or zero for an empty slice.
pub fn reward_mean(rewards: &[f32]) -> f32 {
    if rewards.is_empty() {
        return 0.0;
    }
    rewards.iter().sum::<f32>() / rewards.len() as f32
}

/// Return the undiscounted sum of rewards.
pub fn total_return(rewards: &[f32]) -> f32 {
    rewards.iter().sum()
}

/// Return explained variance for returns and predictions.
pub fn explained_variance(returns: &[f32], predictions: &[f32]) -> f32 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f32;
    let mean_r = returns.iter().sum::<f32>() / n;
    let var_r = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / n;
    if var_r < 1e-8 {
        return 1.0;
    }
    let var_diff: f32 = returns
        .iter()
        .zip(predictions)
        .map(|(r, p)| (r - p).powi(2))
        .sum::<f32>()
        / n;
    1.0 - var_diff / var_r
}

/// Clamp advantages to `[-max_abs, max_abs]`.
pub fn clipped_advantages(advantages: &[f32], max_abs: f32) -> Vec<f32> {
    advantages
        .iter()
        .map(|&a| a.clamp(-max_abs, max_abs))
        .collect()
}

/// Return `(min, max)` advantage values.
pub fn advantage_range(advantages: &[f32]) -> (f32, f32) {
    (min_advantage(advantages), max_advantage(advantages))
}

/// Compute discounted returns for a reward sequence.
pub fn discount_returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut returns = vec![0.0f32; rewards.len()];
    let mut g = 0.0f32;
    for i in (0..rewards.len()).rev() {
        g = rewards[i] + gamma * g;
        returns[i] = g;
    }
    returns
}

/// Spread final player scores over each player's available steps.
pub fn rewards_from_final_scores(final_scores: [i32; 4], num_steps: &[usize; 4]) -> Vec<[f32; 4]> {
    let total_steps: usize = num_steps.iter().sum();
    if total_steps == 0 {
        return Vec::new();
    }
    let max_steps = *num_steps.iter().max().unwrap_or(&0);
    let mut rewards = vec![[0.0f32; 4]; max_steps];
    for p in 0..4 {
        if num_steps[p] > 0 {
            let per_step = final_scores[p] as f32 / num_steps[p] as f32 / 100_000.0;
            for r in rewards.iter_mut().take(num_steps[p]) {
                r[p] = per_step;
            }
        }
    }
    rewards
}

/// Normalize advantages in place to zero mean and unit variance.
pub fn normalize_advantages(advantages: &mut [f32]) {
    if advantages.is_empty() {
        return;
    }
    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
    let std = (var + 1e-8).sqrt();
    for a in advantages.iter_mut() {
        *a = (*a - mean) / std;
    }
}

#[cfg(test)]
mod tests {
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
            assert!(
                (r - ret[adv.iter().position(|x| (x - a).abs() < 1e-10).unwrap()]).abs() < 1e-4
            );
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
}
