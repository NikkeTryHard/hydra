//! Behavioral cloning helper algorithms.

use crate::losses::masked_logits;
use burn::prelude::*;
pub use hydra_train_types::config::{BcExitConfig, cosine_annealing_lr, warmup_then_cosine_lr};

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
    let predicted = masked_logits(logits, mask).argmax(1).squeeze_dim::<1>(1);
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
    assert!(keep_prob.is_finite(), "keep_prob must be finite");
    assert!(
        (0.0..=1.0).contains(&keep_prob),
        "keep_prob must be in [0,1]"
    );
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
    let log_pi = burn::tensor::activation::log_softmax(masked_logits(model_logits, exit_mask), 1);
    let ce = (exit_target * log_pi).sum_dim(1).neg().mean();
    ce * weight
}

#[cfg(test)]
mod tests;
