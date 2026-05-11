//! Behavioral cloning helper algorithms.

use burn::prelude::*;
pub use hydra_train_types::config::{cosine_annealing_lr, warmup_then_cosine_lr};

/// Behavioral-cloning ExIt loss weighting.
#[derive(Debug, Clone, Copy)]
pub struct BcExitConfig {
    pub exit_weight: f32,
}

impl Default for BcExitConfig {
    fn default() -> Self {
        Self { exit_weight: 0.0 }
    }
}

/// Adds ExIt imitation loss when targets and masks are present.
pub fn maybe_add_exit_loss<B: Backend>(
    total: Tensor<B, 1>,
    policy_logits: Tensor<B, 2>,
    exit_target: Option<&Tensor<B, 2>>,
    exit_mask: Option<&Tensor<B, 2>>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    if let (Some(exit_target), Some(exit_mask)) = (exit_target, exit_mask) {
        total
            + exit_loss(
                policy_logits,
                exit_target.clone(),
                exit_mask.clone(),
                exit_cfg.exit_weight,
            )
    } else {
        total
    }
}

/// Computes policy agreement as a fraction in `[0, 1]`.
pub fn policy_agreement<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> f64 {
    let (correct, total) = policy_agreement_counts(logits, mask, targets);
    correct as f64 / total as f64
}

/// Computes masked top-1 agreement counts.
pub fn policy_agreement_counts<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> (usize, usize) {
    let masked = logits + (mask.ones_like() - mask) * (-1e9f32);
    let predicted = masked.argmax(1).squeeze_dim::<1>(1);
    let correct = predicted.equal(targets);
    let dims = correct.dims();
    let total = dims[0];
    let correct = correct
        .into_data()
        .convert::<i64>()
        .as_slice::<i64>()
        .expect("policy agreement correctness should be readable as i64")
        .iter()
        .map(|&value| value as usize)
        .sum();
    (correct, total)
}

/// Recovers hard target action IDs from one-hot or soft policy targets.
pub fn target_actions_from_policy_target<B: Backend>(
    policy_target: Tensor<B, 2>,
) -> Tensor<B, 1, Int> {
    policy_target.argmax(1).squeeze_dim::<1>(1)
}

/// Builds an oracle-guidance keep mask from caller-supplied RNG samples.
pub fn oracle_guidance_mask_values(
    batch_size: usize,
    keep_prob: f32,
    rng_values: &[f32],
) -> Vec<f32> {
    (0..batch_size)
        .map(|idx| {
            let sample = rng_values.get(idx).copied().unwrap_or(0.0);
            if sample < keep_prob { 1.0 } else { 0.0 }
        })
        .collect()
}

/// Builds an oracle-guidance tensor mask from caller-supplied RNG samples.
pub fn oracle_guidance_mask_tensor<B: Backend>(
    batch_size: usize,
    keep_prob: f32,
    rng_values: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let mask = oracle_guidance_mask_values(batch_size, keep_prob, rng_values);
    Tensor::<B, 1>::from_floats(mask.as_slice(), device)
}

fn exit_loss<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    exit_mask: Tensor<B, 2>,
    weight: f32,
) -> Tensor<B, 1> {
    let neg_inf = (exit_mask.ones_like() - exit_mask) * (-1e9f32);
    let log_pi = burn::tensor::activation::log_softmax(model_logits + neg_inf, 1);
    let ce = (exit_target * log_pi).sum_dim(1).neg().mean();
    ce * weight
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn oracle_guidance_mask_values_follow_keep_probability() {
        let mask = oracle_guidance_mask_values(4, 0.5, &[0.1, 0.7, 0.49, 0.9]);
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn policy_agreement_counts_match_fraction() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<NdArray<f32>, 2>::from_floats(
            [
                [10.0, 1.0, 0.0],
                [0.0, 9.0, 1.0],
                [0.0, 1.0, 8.0],
                [3.0, 2.0, 1.0],
            ],
            &device,
        );
        let mask = Tensor::<NdArray<f32>, 2>::ones([4, 3], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints([0, 1, 1, 2], &device);

        let acc = policy_agreement(logits.clone(), mask.clone(), targets.clone());
        let (correct, total) = policy_agreement_counts(logits, mask, targets);

        assert_eq!((correct, total), (2, 4));
        assert!((acc - correct as f64 / total as f64).abs() < 1e-12);
    }

    #[test]
    fn policy_target_argmax_matches_batch_actions() {
        let device = Default::default();
        let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[0i32, 7, 45][..], &device);
        let mut policy_target = vec![0.0f32; 3 * 46];
        policy_target[0] = 1.0;
        policy_target[46 + 7] = 1.0;
        policy_target[2 * 46 + 45] = 1.0;
        let recovered = target_actions_from_policy_target(
            Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device)
                .reshape([3, 46]),
        );
        let same = recovered.equal(actions).into_data().convert::<i64>();
        assert_eq!(
            same.as_slice::<i64>().expect("policy action parity"),
            &[1, 1, 1]
        );
    }

    #[test]
    fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
        assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

        let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
        assert!((warmup_lr - 2.5e-4).abs() < 1e-12);

        let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
        let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
        assert!((post_warmup_lr - expected).abs() < 1e-12);
    }

    #[test]
    fn oracle_guidance_mask_tensor_uses_rng_fallback_for_missing_samples() {
        let device = Default::default();
        let mask = oracle_guidance_mask_tensor::<TestBackend>(3, 0.5, &[0.25], &device)
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        assert_eq!(mask, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn maybe_add_exit_loss_is_noop_without_targets() {
        let device = Default::default();
        let total = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
        let logits = Tensor::<TestBackend, 2>::zeros([1, 3], &device);
        let output = maybe_add_exit_loss(total, logits, None, None, &BcExitConfig::default())
            .into_scalar()
            .elem::<f32>();
        assert!((output - 2.0).abs() < 1e-6);
    }
}
