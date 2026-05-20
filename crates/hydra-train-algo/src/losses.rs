use burn::prelude::*;
use burn::tensor::activation;

pub(crate) const MASKED_LOGIT_SENTINEL: f32 = -1e9;

pub(crate) fn masked_logits<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    logits + (mask.ones_like() - mask) * MASKED_LOGIT_SENTINEL
}

/// Computes masked soft-label policy cross entropy per sample.
pub fn policy_ce<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(masked_logits(logits, mask), 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

/// Computes half mean-squared value error per sample.
pub fn value_mse<B: Backend>(pred: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.clone() * diff * 0.5
}

/// Computes group-classification soft-label cross entropy per sample.
pub fn grp_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(logits, 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

/// Computes tenpai binary cross entropy per sample.
pub fn tenpai_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let loss = bce_with_logits(logits, target);
    loss.mean_dim(1).squeeze_dim::<1>(1)
}

/// Computes focal binary cross entropy for danger predictions per sample.
pub fn danger_focal_bce<B: Backend>(
    logits: Tensor<B, 3>,
    target: Tensor<B, 3>,
    mask: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let alpha = 0.25f32;
    let gamma = 2.0f32;
    let p = activation::sigmoid(logits.clone());
    let bce = bce_with_logits_3d(logits, target.clone());
    let p_t = target.clone() * p.clone() + (target.ones_like() - target) * (p.ones_like() - p);
    let focal_weight = (p_t.ones_like() - p_t).powf_scalar(gamma) * alpha;
    let focal = focal_weight * bce * mask;
    let sum_per_sample = focal.sum_dim(2).sum_dim(1);
    sum_per_sample.squeeze_dim::<2>(2).squeeze_dim::<1>(1)
}

/// Computes opponent next-discard cross entropy averaged over opponents per sample.
pub fn opp_next_ce<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let [batch, opps, tiles] = logits.dims();
    let logits_flat = logits.reshape([batch * opps, tiles]);
    let target_flat = target.reshape([batch * opps, tiles]);
    let log_probs = activation::log_softmax(logits_flat, 1);
    let per_sample = (target_flat * log_probs)
        .sum_dim(1)
        .neg()
        .squeeze_dim::<1>(1);
    per_sample
        .reshape([batch, opps])
        .mean_dim(1)
        .squeeze_dim::<1>(1)
}

/// Computes score-PDF soft-label cross entropy per sample.
pub fn score_pdf_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(logits, 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

/// Computes score-CDF binary cross entropy per sample.
pub fn score_cdf_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let loss = bce_with_logits(logits, target);
    loss.mean_dim(1).squeeze_dim::<1>(1)
}

/// Computes mean binary cross entropy for belief-field logits.
pub fn belief_fields_bce<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    belief_fields_bce_per_sample(logits, target).mean()
}

/// Computes belief-field binary cross entropy per sample.
pub fn belief_fields_bce_per_sample<B: Backend>(
    logits: Tensor<B, 3>,
    target: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, channels, tiles] = logits.dims();
    bce_with_logits_3d(logits, target)
        .reshape([batch, channels * tiles])
        .mean_dim(1)
        .squeeze_dim::<1>(1)
}

/// Computes mean mixture-weight cross entropy.
pub fn mixture_weight_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    mixture_weight_ce_per_sample(logits, target).mean()
}

/// Computes mixture-weight cross entropy per sample.
pub fn mixture_weight_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
}

/// Computes mean opponent-hand-type cross entropy.
pub fn opponent_hand_type_ce<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    opponent_hand_type_ce_per_sample(logits, target).mean()
}

/// Computes opponent-hand-type cross entropy per sample.
pub fn opponent_hand_type_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
}

/// Computes dense-regression half mean-squared error.
pub fn dense_regression_mse<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    (diff.clone() * diff).mean() * 0.5
}

/// Computes half mean-squared error over masked action entries.
pub fn masked_action_mse<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = pred - target;
    let sq = diff.clone() * diff * 0.5;
    let masked = sq * mask.clone();
    let denom = mask.sum().clamp_min(1.0);
    masked.sum() / denom
}

/// Computes lower-tail conditional value-at-risk from a discrete PDF.
pub fn compute_cvar(pdf: &[f32], alpha: f32) -> f32 {
    if pdf.is_empty() || !alpha.is_finite() || alpha <= 0.0 {
        return 0.0;
    }
    let alpha = alpha.min(1.0);
    let n = pdf.len();
    let mut cumsum = 0.0f32;
    let mut weighted_sum = 0.0f32;
    let bin_width = 1.0 / n as f32;
    for (i, &p) in pdf.iter().enumerate() {
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let next_cum = cumsum + p;
        if cumsum < alpha {
            let contrib = p.min(alpha - cumsum);
            let bin_center = (i as f32 + 0.5) * bin_width;
            weighted_sum += contrib * bin_center;
        }
        cumsum = next_cum;
    }
    weighted_sum / alpha
}

/// Mixes GAE return with a baseline value and clamps the target to [-1, 1].
pub fn value_target_from_gae(gae_return: f32, value_baseline: f32, lambda_weight: f32) -> f32 {
    assert!(lambda_weight.is_finite(), "lambda_weight must be finite");
    assert!(
        (0.0..=1.0).contains(&lambda_weight),
        "lambda_weight must be in [0,1]"
    );
    (lambda_weight * gae_return + (1.0 - lambda_weight) * value_baseline).clamp(-1.0, 1.0)
}

/// Builds a soft policy target by mixing model probabilities with an ExIt target.
pub fn soft_target_from_exit<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    mix: f32,
) -> Tensor<B, 2> {
    let model_probs = burn::tensor::activation::softmax(masked_logits(model_logits, mask), 1);
    model_probs * (1.0 - mix) + exit_target * mix
}

/// Applies uniform label smoothing to a 2-D target distribution.
pub fn label_smoothing<B: Backend>(target: Tensor<B, 2>, alpha: f32) -> Tensor<B, 2> {
    assert!(alpha.is_finite(), "label smoothing alpha must be finite");
    assert!(
        (0.0..=1.0).contains(&alpha),
        "label smoothing alpha must be in [0,1]"
    );
    let n = target.dims()[1] as f32;
    target * (1.0 - alpha) + (alpha / n)
}

/// Computes policy cross entropy after dividing logits by a temperature.
pub fn policy_ce_with_temperature<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    temperature: f32,
) -> Tensor<B, 1> {
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be finite and positive"
    );
    policy_ce(logits / temperature, target, mask)
}

/// Reads the absolute scalar value from a one-element loss tensor.
pub fn loss_abs<B: Backend>(loss: &Tensor<B, 1>) -> f32 {
    loss.clone()
        .abs()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("loss scalar should be readable as f32")[0]
}

/// Returns whether a one-element loss tensor contains a finite scalar.
pub fn loss_is_finite<B: Backend>(loss: &Tensor<B, 1>) -> bool {
    let v = loss
        .clone()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("loss scalar should be readable as f32")[0];
    v.is_finite()
}

/// Computes KL divergence from masked logits to a target distribution.
pub fn batch_kl_from_target<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask);
    let probs = log_probs.clone().exp();
    kl_divergence(probs, target)
}

/// Reads a scalar proxy for gradient norm from a one-element loss tensor.
pub fn grad_norm_approx<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    loss.abs()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("grad norm scalar should be readable as f32")[0]
}

/// Computes per-dimension value variance across a batch.
pub fn batch_value_variance<B: Backend>(values: Tensor<B, 2>) -> Tensor<B, 1> {
    let mean = values.clone().mean_dim(0);
    let diff = values - mean;
    (diff.clone() * diff).mean_dim(0).squeeze_dim::<1>(0)
}

/// Computes mean masked policy entropy for a batch of logits.
pub fn batch_policy_entropy<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask.clone());
    let probs = log_probs.clone().exp();
    (probs * log_probs * mask).sum_dim(1).neg().mean()
}

/// Computes mean entropy from probability rows.
pub fn mean_entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    entropy(probs).mean()
}

/// Computes masked log-softmax over action logits.
pub fn masked_log_softmax<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    burn::tensor::activation::log_softmax(masked_logits(logits, mask), 1)
}

/// Computes soft-label cross entropy from log probabilities per sample.
pub fn cross_entropy_soft<B: Backend>(
    log_probs: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

/// Computes Shannon entropy for probability rows.
pub fn entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-8f32;
    let safe = probs.clone().clamp(eps, 1.0);
    (probs * safe.log()).sum_dim(1).neg().squeeze_dim::<1>(1)
}

/// Computes row-wise KL divergence KL(p || q).
pub fn kl_divergence<B: Backend>(p: Tensor<B, 2>, q: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-8f32;
    let p_safe = p.clone().clamp(eps, 1.0);
    let q_safe = q.clamp(eps, 1.0);
    (p * (p_safe.log() - q_safe.log()))
        .sum_dim(1)
        .squeeze_dim::<1>(1)
}

/// Converts final scores into zero-sum oracle critic targets.
pub fn oracle_target_from_scores(final_scores: [i32; 4]) -> [f32; 4] {
    let mean = final_scores.iter().sum::<i32>() as f32 / 4.0;
    let mut target = [0.0f32; 4];
    for (i, &s) in final_scores.iter().enumerate() {
        target[i] = (s as f32 - mean) / 100_000.0;
    }
    target
}

/// Computes mean oracle-critic loss.
pub fn oracle_critic_loss<B: Backend>(
    v_oracle: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    oracle_critic_loss_per_sample(v_oracle, target).mean()
}

/// Computes oracle-critic loss per sample, including a zero-sum penalty.
pub fn oracle_critic_loss_per_sample<B: Backend>(
    v_oracle: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let v_norm = v_oracle.clone() - v_oracle.clone().mean_dim(1);
    let diff = v_norm - target;
    let mse = (diff.clone() * diff).mean_dim(1).squeeze_dim::<1>(1) * 0.5;
    let zero_sum_penalty = v_oracle.sum_dim(1).squeeze_dim::<1>(1);
    let zero_sum_penalty = zero_sum_penalty.clone() * zero_sum_penalty * 10.0;
    mse + zero_sum_penalty
}

/// Reduces per-sample losses with an optional sample mask.
pub fn masked_mean<B: Backend>(
    per_sample: Tensor<B, 1>,
    mask: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    match mask {
        Some(mask) => {
            let denom = mask.clone().sum().clamp_min(1.0);
            (per_sample * mask).sum() / denom
        }
        None => per_sample.mean(),
    }
}

/// Multiplies two optional sample masks, preserving whichever side is present.
pub fn combine_sample_masks<B: Backend>(
    primary: Option<Tensor<B, 1>>,
    secondary: Option<Tensor<B, 1>>,
) -> Option<Tensor<B, 1>> {
    match (primary, secondary) {
        (Some(primary), Some(secondary)) => Some(primary * secondary),
        (Some(primary), None) => Some(primary),
        (None, Some(secondary)) => Some(secondary),
        (None, None) => None,
    }
}

fn bce_with_logits<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 2> {
    let max_val = logits.clone().clamp_min(0.0);
    let neg_abs = logits.clone().abs().neg();
    max_val - logits * target + neg_abs.exp().add_scalar(1.0).log()
}

fn bce_with_logits_3d<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 3> {
    let max_val = logits.clone().clamp_min(0.0);
    let neg_abs = logits.clone().abs().neg();
    max_val - logits * target + neg_abs.exp().add_scalar(1.0).log()
}

#[cfg(test)]
mod tests;
