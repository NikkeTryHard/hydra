//! Generalized Advantage Estimation (gamma=0.995, lambda=0.95).

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
}
