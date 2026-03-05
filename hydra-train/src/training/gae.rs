//! Generalized Advantage Estimation (gamma=0.995, lambda=0.95).

pub struct GaeConfig {
    pub gamma: f32,
    pub lambda: f32,
}

impl Default for GaeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.995,
            lambda: 0.95,
        }
    }
}

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
}
