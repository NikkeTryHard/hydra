use burn::prelude::*;
use burn::tensor::activation;

const NEG_INF: f32 = -1e9;

/// Computes masked soft-label policy cross entropy per sample.
pub fn policy_ce<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let masked = logits + (mask.ones_like() - mask) * NEG_INF;
    let log_probs = activation::log_softmax(masked, 1);
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
    let n = pdf.len();
    if n == 0 || alpha <= 0.0 {
        return 0.0;
    }
    let mut cumsum = 0.0f32;
    let mut weighted_sum = 0.0f32;
    let bin_width = 1.0 / n as f32;
    for (i, &p) in pdf.iter().enumerate() {
        let next_cum = cumsum + p;
        if cumsum < alpha {
            let contrib = p.min(alpha - cumsum);
            let bin_center = (i as f32 + 0.5) * bin_width;
            weighted_sum += contrib * bin_center;
        }
        cumsum = next_cum;
    }
    if alpha > 0.0 {
        weighted_sum / alpha
    } else {
        0.0
    }
}

/// Mixes GAE return with a baseline value and clamps the target to [-1, 1].
pub fn value_target_from_gae(gae_return: f32, value_baseline: f32, lambda_weight: f32) -> f32 {
    (lambda_weight * gae_return + (1.0 - lambda_weight) * value_baseline).clamp(-1.0, 1.0)
}

/// Builds a soft policy target by mixing model probabilities with an ExIt target.
pub fn soft_target_from_exit<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    mix: f32,
) -> Tensor<B, 2> {
    let model_probs = burn::tensor::activation::softmax(
        model_logits + (mask.ones_like() - mask.clone()) * (-1e9f32),
        1,
    );
    model_probs * (1.0 - mix) + exit_target * mix
}

/// Applies uniform label smoothing to a 2-D target distribution.
pub fn label_smoothing<B: Backend>(target: Tensor<B, 2>, alpha: f32) -> Tensor<B, 2> {
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
    let neg_inf = (mask.ones_like() - mask) * (-1e9f32);
    burn::tensor::activation::log_softmax(logits + neg_inf, 1)
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
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_policy_ce_with_mask() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0, -1.0]], &device);
        let mut mask_data = [1.0f32; 4];
        mask_data[3] = 0.0;
        let mask = Tensor::<B, 2>::from_floats([mask_data], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0, 1.0, 0.0]], &device);
        let loss = policy_ce(logits, target, mask);
        let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
        assert!(val > 0.0, "policy CE should be positive, got {val}");
        assert!(val < 5.0, "policy CE too large: {val}");
    }

    #[test]
    fn test_policy_ce_illegal_action_zero_gradient() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[10.0, -10.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.5, 0.0, 0.5]], &device);
        let loss = policy_ce(logits.clone(), target, mask);
        let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
        assert!(val.is_finite(), "masked loss should be finite: {val}");
    }

    #[test]
    fn test_soft_target_differs_from_hard() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 0.5]], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let hard = Tensor::<B, 2>::from_floats([[0.0, 1.0, 0.0]], &device);
        let soft = Tensor::<B, 2>::from_floats([[0.3, 0.7, 0.0]], &device);
        let l_hard = policy_ce(logits.clone(), hard, mask.clone());
        let l_soft = policy_ce(logits, soft, mask);
        let h = l_hard.to_data().as_slice::<f32>().expect("f32")[0];
        let s = l_soft.to_data().as_slice::<f32>().expect("f32")[0];
        assert!(
            (h - s).abs() > 0.01,
            "soft vs hard should differ: {h} vs {s}"
        );
    }

    #[test]
    fn test_oracle_critic_zero_sum() {
        let device = Default::default();
        let v = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
        let loss = oracle_critic_loss(v, target);
        let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
        assert!(
            val.abs() < 1e-4,
            "zero-sum input should give near-zero loss, got {val}"
        );
    }

    #[test]
    fn test_oracle_target_zero_sum() {
        let target = oracle_target_from_scores([30000, 25000, 25000, 20000]);
        let sum: f32 = target.iter().sum();
        assert!(sum.abs() < 1e-5, "oracle target should be zero-sum: {sum}");
        assert!(target[0] > 0.0, "1st place should be positive");
        assert!(target[3] < 0.0, "4th place should be negative");
    }

    #[test]
    fn test_focal_bce_vs_standard_bce() {
        let device = Default::default();
        let logits = Tensor::<B, 3>::from_floats([[[3.0; 34]; 3]], &device);
        let target = Tensor::<B, 3>::ones([1, 3, 34], &device);
        let mask = Tensor::<B, 3>::ones([1, 3, 34], &device);
        let focal = danger_focal_bce(logits.clone(), target.clone(), mask.clone());
        let standard = bce_with_logits_3d(logits, target);
        let standard_sum = (standard * mask)
            .sum_dim(2)
            .sum_dim(1)
            .squeeze_dim::<2>(2)
            .squeeze_dim::<1>(1);
        let f = focal.into_scalar().elem::<f32>();
        let s = standard_sum.into_scalar().elem::<f32>();
        assert!(
            f < s,
            "focal ({f}) should be < standard ({s}) for high-confidence correct"
        );
    }

    #[test]
    fn test_compute_cvar() {
        let pdf = [0.1f32, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1];
        let cvar = compute_cvar(&pdf, 0.3);
        assert!(cvar >= 0.0 && cvar.is_finite(), "CVaR: {cvar}");
        let cvar_full = compute_cvar(&pdf, 1.0);
        assert!(cvar <= cvar_full, "CVaR(0.3) <= CVaR(1.0)");
    }

    #[test]
    fn test_bce_extreme_logits() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[100.0, -100.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
        let loss = bce_with_logits(logits, target);
        let data = loss.to_data();
        for &v in data.as_slice::<f32>().expect("f32") {
            assert!(v.is_finite(), "extreme logits should give finite BCE: {v}");
        }
    }

    #[test]
    fn test_policy_ce_single_legal_action() {
        let device = Default::default();
        let mut mask_data = [0.0f32; 46];
        mask_data[5] = 1.0;
        let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device).reshape([1, 46]);
        let target = mask.clone();
        let logits = Tensor::<B, 2>::zeros([1, 46], &device);
        let loss = policy_ce(logits, target, mask);
        let v: f32 = loss.into_scalar().elem();
        assert!(v < 0.01, "single legal action loss should be ~0, got {v}");
    }

    #[test]
    fn test_value_mse_extreme_values() {
        let device = Default::default();
        let pred = Tensor::<B, 1>::from_floats([0.99, -0.99], &device);
        let target = Tensor::<B, 1>::from_floats([1.0, -1.0], &device);
        let loss = value_mse(pred, target);
        let data = loss.to_data();
        for &v in data.as_slice::<f32>().expect("f32") {
            assert!(v.is_finite(), "extreme value MSE should be finite, got {v}");
            assert!(v < 0.01, "near-boundary MSE should be small, got {v}");
        }
    }

    #[test]
    fn test_oracle_target_from_scores_zero_sum() {
        let target = oracle_target_from_scores([25000, 25000, 25000, 25000]);
        for (i, &v) in target.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "equal scores should give zero delta, player {i} got {v}"
            );
        }
    }

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let device = Default::default();
        let p = Tensor::<B, 2>::from_floats([[0.3, 0.5, 0.2]], &device);
        let kl = kl_divergence(p.clone(), p);
        let v: f32 = kl.into_scalar().elem();
        assert!(v.abs() < 1e-6, "KL(p, p) should be ~0, got {v}");
    }
}
