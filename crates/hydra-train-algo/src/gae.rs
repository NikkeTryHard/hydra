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
mod tests;
