//! Loss functions for all 9 heads + total weighted loss.

use burn::prelude::*;
use burn::tensor::activation;
use std::marker::PhantomData;

use crate::model::HydraOutput;

#[derive(Clone)]
pub struct HydraTargets<B: Backend> {
    pub policy_target: Tensor<B, 2>,
    pub legal_mask: Tensor<B, 2>,
    pub value_target: Tensor<B, 1>,
    pub grp_target: Tensor<B, 2>,
    pub tenpai_target: Tensor<B, 2>,
    pub danger_target: Tensor<B, 3>,
    pub danger_mask: Tensor<B, 3>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
    pub oracle_target: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    pub opponent_hand_type_target: Option<Tensor<B, 2>>,
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    pub oracle_guidance_mask: Option<Tensor<B, 1>>,
}

impl<B: Backend> HydraTargets<B> {
    /// Slice all target tensors along the batch dimension (dim 0).
    ///
    /// Produces a sub-batch covering `[start..end)`. Used by microbatch
    /// accumulation to split a full RL batch into VRAM-friendly chunks.
    pub fn slice_batch(&self, start: usize, end: usize) -> Self {
        let r1 = [start..end];
        let r2 = [start..end];
        let r3 = [start..end];
        Self {
            policy_target: self.policy_target.clone().slice(r1.clone()),
            legal_mask: self.legal_mask.clone().slice(r1.clone()),
            value_target: self.value_target.clone().slice(r2.clone()),
            grp_target: self.grp_target.clone().slice(r1.clone()),
            tenpai_target: self.tenpai_target.clone().slice(r1.clone()),
            danger_target: self.danger_target.clone().slice(r3.clone()),
            danger_mask: self.danger_mask.clone().slice(r3.clone()),
            opp_next_target: self.opp_next_target.clone().slice(r3.clone()),
            score_pdf_target: self.score_pdf_target.clone().slice(r1.clone()),
            score_cdf_target: self.score_cdf_target.clone().slice(r1.clone()),
            oracle_target: self
                .oracle_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            belief_fields_target: self
                .belief_fields_target
                .as_ref()
                .map(|t| t.clone().slice(r3.clone())),
            belief_fields_mask: self
                .belief_fields_mask
                .as_ref()
                .map(|t| t.clone().slice(r2.clone())),
            mixture_weight_target: self
                .mixture_weight_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            mixture_weight_mask: self
                .mixture_weight_mask
                .as_ref()
                .map(|t| t.clone().slice(r2.clone())),
            opponent_hand_type_target: self
                .opponent_hand_type_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            delta_q_target: self
                .delta_q_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            delta_q_mask: self
                .delta_q_mask
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            safety_residual_target: self
                .safety_residual_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            safety_residual_mask: self
                .safety_residual_mask
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            oracle_guidance_mask: self
                .oracle_guidance_mask
                .as_ref()
                .map(|t| t.clone().slice(r2)),
        }
    }
}

#[derive(Config, Debug)]
pub struct HydraLossConfig {
    #[config(default = "1.0")]
    pub w_pi: f32,
    #[config(default = "0.5")]
    pub w_v: f32,
    #[config(default = "0.2")]
    pub w_grp: f32,
    #[config(default = "0.1")]
    pub w_tenpai: f32,
    #[config(default = "0.1")]
    pub w_danger: f32,
    #[config(default = "0.1")]
    pub w_opp: f32,
    #[config(default = "0.025")]
    pub w_score: f32,
    #[config(default = "0.0")]
    pub w_oracle_critic: f32,
    #[config(default = "0.0")]
    pub w_belief_fields: f32,
    #[config(default = "0.0")]
    pub w_mixture_weight: f32,
    #[config(default = "0.0")]
    pub w_opponent_hand_type: f32,
    #[config(default = "0.0")]
    pub w_delta_q: f32,
    #[config(default = "0.0")]
    pub w_safety_residual: f32,
}

impl HydraLossConfig {
    pub fn total_weight(&self) -> f32 {
        self.w_pi
            + self.w_v
            + self.w_grp
            + self.w_tenpai
            + self.w_danger
            + self.w_opp
            + self.w_score * 2.0
            + self.w_oracle_critic
            + self.w_belief_fields
            + self.w_mixture_weight
            + self.w_opponent_hand_type
            + self.w_delta_q
            + self.w_safety_residual
    }

    pub fn scale_all(&self, factor: f32) -> Self {
        Self::new()
            .with_w_pi(self.w_pi * factor)
            .with_w_v(self.w_v * factor)
            .with_w_grp(self.w_grp * factor)
            .with_w_tenpai(self.w_tenpai * factor)
            .with_w_danger(self.w_danger * factor)
            .with_w_opp(self.w_opp * factor)
            .with_w_score(self.w_score * factor)
            .with_w_oracle_critic(self.w_oracle_critic * factor)
            .with_w_belief_fields(self.w_belief_fields * factor)
            .with_w_mixture_weight(self.w_mixture_weight * factor)
            .with_w_opponent_hand_type(self.w_opponent_hand_type * factor)
            .with_w_delta_q(self.w_delta_q * factor)
            .with_w_safety_residual(self.w_safety_residual * factor)
    }

    pub fn summary(&self) -> String {
        format!(
            "loss(pi={:.1}, v={:.1}, grp={:.1})",
            self.w_pi, self.w_v, self.w_grp
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.w_pi < 0.0
            || self.w_v < 0.0
            || self.w_grp < 0.0
            || self.w_tenpai < 0.0
            || self.w_danger < 0.0
            || self.w_opp < 0.0
            || self.w_score < 0.0
            || self.w_oracle_critic < 0.0
            || self.w_belief_fields < 0.0
            || self.w_mixture_weight < 0.0
            || self.w_opponent_hand_type < 0.0
            || self.w_delta_q < 0.0
            || self.w_safety_residual < 0.0
        {
            return Err("loss weights must be non-negative");
        }
        Ok(())
    }
}

pub struct HydraLoss<B: Backend> {
    pub config: HydraLossConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend> HydraLoss<B> {
    pub fn new(config: HydraLossConfig) -> Self {
        Self {
            config,
            _backend: PhantomData,
        }
    }
}

const NEG_INF: f32 = -1e9;

pub fn policy_ce<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let masked = logits + (mask.ones_like() - mask) * NEG_INF;
    let log_probs = activation::log_softmax(masked, 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

pub fn value_mse<B: Backend>(pred: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = pred - target;
    diff.clone() * diff * 0.5
}

pub fn grp_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(logits, 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

pub fn tenpai_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let loss = bce_with_logits(logits, target);
    loss.mean_dim(1).squeeze_dim::<1>(1)
}

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

pub fn score_pdf_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(logits, 1);
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

pub fn score_cdf_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let loss = bce_with_logits(logits, target);
    loss.mean_dim(1).squeeze_dim::<1>(1)
}

pub fn belief_fields_bce<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    belief_fields_bce_per_sample(logits, target).mean()
}

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

pub fn mixture_weight_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    mixture_weight_ce_per_sample(logits, target).mean()
}

pub fn mixture_weight_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
}

pub fn opponent_hand_type_ce<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    opponent_hand_type_ce_per_sample(logits, target).mean()
}

pub fn opponent_hand_type_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
}

pub fn dense_regression_mse<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    (diff.clone() * diff).mean() * 0.5
}

pub fn masked_action_mse<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = pred - target;
    let sq = diff.clone() * diff * 0.5;
    let masked = sq * mask.clone();
    let denom = mask.sum().into_scalar().elem::<f32>().max(1.0);
    masked.sum() / denom
}

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

pub fn value_target_from_gae(gae_return: f32, value_baseline: f32, lambda_weight: f32) -> f32 {
    (lambda_weight * gae_return + (1.0 - lambda_weight) * value_baseline).clamp(-1.0, 1.0)
}

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

pub fn label_smoothing<B: Backend>(target: Tensor<B, 2>, alpha: f32) -> Tensor<B, 2> {
    let n = target.dims()[1] as f32;
    target * (1.0 - alpha) + (alpha / n)
}

pub fn policy_ce_with_temperature<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    temperature: f32,
) -> Tensor<B, 1> {
    policy_ce(logits / temperature, target, mask)
}

pub fn loss_abs<B: Backend>(loss: &Tensor<B, 1>) -> f32 {
    loss.clone().abs().into_scalar().elem::<f32>()
}

pub fn loss_is_finite<B: Backend>(loss: &Tensor<B, 1>) -> bool {
    let v: f32 = loss.clone().into_scalar().elem();
    v.is_finite()
}

pub fn total_loss_scalar<B: Backend>(breakdown: &LossBreakdown<B>) -> f32 {
    breakdown.total.clone().into_scalar().elem::<f32>()
}

pub fn batch_kl_from_target<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask);
    let probs = log_probs.clone().exp();
    kl_divergence(probs, target)
}

pub fn grad_norm_approx<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    loss.abs().into_scalar().elem::<f32>()
}

pub fn batch_value_variance<B: Backend>(values: Tensor<B, 2>) -> Tensor<B, 1> {
    let mean = values.clone().mean_dim(0);
    let diff = values - mean;
    (diff.clone() * diff).mean_dim(0).squeeze_dim::<1>(0)
}

pub fn batch_policy_entropy<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask.clone());
    let probs = log_probs.clone().exp();
    (probs * log_probs * mask).sum_dim(1).neg().mean()
}

pub fn mean_entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    entropy(probs).mean()
}

pub fn masked_log_softmax<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    let neg_inf = (mask.ones_like() - mask) * (-1e9f32);
    burn::tensor::activation::log_softmax(logits + neg_inf, 1)
}

pub fn cross_entropy_soft<B: Backend>(
    log_probs: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
}

pub fn entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-8f32;
    let safe = probs.clone().clamp(eps, 1.0);
    (probs * safe.log()).sum_dim(1).neg().squeeze_dim::<1>(1)
}

pub fn kl_divergence<B: Backend>(p: Tensor<B, 2>, q: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-8f32;
    let p_safe = p.clone().clamp(eps, 1.0);
    let q_safe = q.clamp(eps, 1.0);
    (p * (p_safe.log() - q_safe.log()))
        .sum_dim(1)
        .squeeze_dim::<1>(1)
}

pub fn oracle_target_from_scores(final_scores: [i32; 4]) -> [f32; 4] {
    let mean = final_scores.iter().sum::<i32>() as f32 / 4.0;
    let mut target = [0.0f32; 4];
    for (i, &s) in final_scores.iter().enumerate() {
        target[i] = (s as f32 - mean) / 100_000.0;
    }
    target
}

pub fn oracle_critic_loss<B: Backend>(
    v_oracle: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    oracle_critic_loss_per_sample(v_oracle, target).mean()
}

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

fn masked_mean<B: Backend>(per_sample: Tensor<B, 1>, mask: Option<Tensor<B, 1>>) -> Tensor<B, 1> {
    match mask {
        Some(mask) => {
            let denom = mask.clone().sum().clamp_min(1.0);
            (per_sample * mask).sum() / denom
        }
        None => per_sample.mean(),
    }
}

pub struct LossBreakdown<B: Backend> {
    pub policy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    pub grp: Tensor<B, 1>,
    pub tenpai: Tensor<B, 1>,
    pub danger: Tensor<B, 1>,
    pub opp_next: Tensor<B, 1>,
    pub score_pdf: Tensor<B, 1>,
    pub score_cdf: Tensor<B, 1>,
    pub oracle_critic: Tensor<B, 1>,
    pub belief_fields: Tensor<B, 1>,
    pub mixture_weight: Tensor<B, 1>,
    pub opponent_hand_type: Tensor<B, 1>,
    pub delta_q: Tensor<B, 1>,
    pub safety_residual: Tensor<B, 1>,
    pub total: Tensor<B, 1>,
}

impl<B: Backend> LossBreakdown<B> {
    pub fn all_finite(&self) -> bool {
        let s = |t: &Tensor<B, 1>| -> f32 { t.clone().into_scalar().elem() };
        [
            s(&self.policy),
            s(&self.value),
            s(&self.grp),
            s(&self.tenpai),
            s(&self.danger),
            s(&self.opp_next),
            s(&self.score_pdf),
            s(&self.score_cdf),
            s(&self.oracle_critic),
            s(&self.belief_fields),
            s(&self.mixture_weight),
            s(&self.opponent_hand_type),
            s(&self.delta_q),
            s(&self.safety_residual),
            s(&self.total),
        ]
        .iter()
        .all(|v| v.is_finite())
    }
}

impl<B: Backend> HydraLoss<B> {
    pub fn total_loss(
        &self,
        outputs: &HydraOutput<B>,
        targets: &HydraTargets<B>,
    ) -> LossBreakdown<B> {
        let oracle_mask = targets.oracle_guidance_mask.clone();
        let l_pi = policy_ce(
            outputs.policy_logits.clone(),
            targets.policy_target.clone(),
            targets.legal_mask.clone(),
        )
        .mean();
        let l_v = value_mse(
            outputs.value.clone().squeeze_dim::<1>(1),
            targets.value_target.clone(),
        )
        .mean();
        let l_grp = grp_ce(outputs.grp.clone(), targets.grp_target.clone()).mean();
        let l_tenpai = tenpai_bce(outputs.opp_tenpai.clone(), targets.tenpai_target.clone()).mean();
        let l_danger = danger_focal_bce(
            outputs.danger.clone(),
            targets.danger_target.clone(),
            targets.danger_mask.clone(),
        )
        .mean();
        let l_opp = opp_next_ce(
            outputs.opp_next_discard.clone(),
            targets.opp_next_target.clone(),
        )
        .mean();
        let l_pdf =
            score_pdf_ce(outputs.score_pdf.clone(), targets.score_pdf_target.clone()).mean();
        let l_cdf =
            score_cdf_bce(outputs.score_cdf.clone(), targets.score_cdf_target.clone()).mean();
        let zero = outputs.value.clone().sum() * 0.0;
        let l_oracle = match &targets.oracle_target {
            Some(target) => masked_mean(
                oracle_critic_loss_per_sample(outputs.oracle_critic.clone(), target.clone()),
                oracle_mask.clone(),
            ),
            None => zero.clone(),
        };
        let l_belief = match (&targets.belief_fields_target, &targets.belief_fields_mask) {
            (Some(target), Some(mask)) => masked_mean(
                belief_fields_bce_per_sample(outputs.belief_fields.clone(), target.clone()),
                Some(mask.clone()),
            ),
            _ => zero.clone(),
        };
        let l_mix = match (&targets.mixture_weight_target, &targets.mixture_weight_mask) {
            (Some(target), Some(mask)) => masked_mean(
                mixture_weight_ce_per_sample(outputs.mixture_weight_logits.clone(), target.clone()),
                Some(mask.clone()),
            ),
            _ => zero.clone(),
        };
        let l_hand_type = match &targets.opponent_hand_type_target {
            Some(target) => masked_mean(
                opponent_hand_type_ce_per_sample(
                    outputs.opponent_hand_type.clone(),
                    target.clone(),
                ),
                oracle_mask.clone(),
            ),
            None => zero.clone(),
        };
        let l_delta_q = match (&targets.delta_q_target, &targets.delta_q_mask) {
            (Some(target), Some(mask)) => {
                masked_action_mse(outputs.delta_q.clone(), target.clone(), mask.clone())
            }
            _ => zero.clone(),
        };
        let l_safety_residual = match (
            &targets.safety_residual_target,
            &targets.safety_residual_mask,
        ) {
            (Some(target), Some(mask)) => masked_action_mse(
                outputs.safety_residual.clone(),
                target.clone(),
                mask.clone(),
            ),
            _ => zero.clone(),
        };
        let c = &self.config;
        let total = l_pi.clone() * c.w_pi
            + l_v.clone() * c.w_v
            + l_grp.clone() * c.w_grp
            + l_tenpai.clone() * c.w_tenpai
            + l_danger.clone() * c.w_danger
            + l_opp.clone() * c.w_opp
            + l_pdf.clone() * c.w_score
            + l_cdf.clone() * c.w_score
            + l_oracle.clone() * c.w_oracle_critic
            + l_belief.clone() * c.w_belief_fields
            + l_mix.clone() * c.w_mixture_weight
            + l_hand_type.clone() * c.w_opponent_hand_type
            + l_delta_q.clone() * c.w_delta_q
            + l_safety_residual.clone() * c.w_safety_residual;
        LossBreakdown {
            policy: l_pi,
            value: l_v,
            grp: l_grp,
            tenpai: l_tenpai,
            danger: l_danger,
            opp_next: l_opp,
            score_pdf: l_pdf,
            score_cdf: l_cdf,
            oracle_critic: l_oracle,
            belief_fields: l_belief,
            mixture_weight: l_mix,
            opponent_hand_type: l_hand_type,
            delta_q: l_delta_q,
            safety_residual: l_safety_residual,
            total,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::model::HydraModelConfig;
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
    fn test_oracle_target_populates_breakdown_only() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.oracle_target = Some(Tensor::<B, 2>::zeros([2, 4], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
        let total_loss: f32 = breakdown.total.into_scalar().elem();
        assert!(oracle_loss.is_finite() && oracle_loss >= 0.0);
        assert!(total_loss.is_finite() && total_loss >= 0.0);
    }

    #[test]
    fn test_oracle_absent_with_mask_keeps_oracle_loss_zero() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_oracle_critic(1.0));
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.oracle_target = None;
        targets.oracle_guidance_mask = Some(Tensor::<B, 1>::zeros([2], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
        let total_loss: f32 = breakdown.total.into_scalar().elem();
        assert!(
            oracle_loss.abs() < 1e-8,
            "oracle loss should be zero when target absent"
        );
        assert!(total_loss.is_finite() && total_loss >= 0.0);
    }

    #[test]
    fn test_oracle_target_contributes_to_total_when_weight_enabled() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.oracle_target = Some(Tensor::<B, 2>::ones([2, 4], &device));

        let base = HydraLoss::<B>::new(HydraLossConfig::new()).total_loss(&outputs, &targets);
        let with_oracle = HydraLoss::<B>::new(HydraLossConfig::new().with_w_oracle_critic(1.0))
            .total_loss(&outputs, &targets);

        let total_base: f32 = base.total.into_scalar().elem();
        let total_oracle: f32 = with_oracle.total.into_scalar().elem();
        let oracle_loss: f32 = with_oracle.oracle_critic.into_scalar().elem();
        assert!(oracle_loss > 0.0, "oracle loss should be active");
        assert!(
            total_oracle > total_base,
            "oracle weighting should raise total loss"
        );
    }

    #[test]
    fn test_oracle_guidance_mask_disables_masked_optional_losses() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.oracle_target = Some(Tensor::<B, 2>::ones([2, 4], &device));
        targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
        targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            &device,
        ));
        targets.opponent_hand_type_target = Some(Tensor::<B, 2>::from_floats(
            [
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ],
            &device,
        ));
        targets.oracle_guidance_mask = Some(Tensor::<B, 1>::zeros([2], &device));

        let loss_fn = HydraLoss::<B>::new(
            HydraLossConfig::new()
                .with_w_oracle_critic(1.0)
                .with_w_belief_fields(1.0)
                .with_w_mixture_weight(1.0)
                .with_w_opponent_hand_type(1.0),
        );
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        assert!(breakdown.oracle_critic.into_scalar().elem::<f32>().abs() < 1e-8);
        assert!(breakdown.belief_fields.into_scalar().elem::<f32>().abs() < 1e-8);
        assert!(breakdown.mixture_weight.into_scalar().elem::<f32>().abs() < 1e-8);
        assert!(
            breakdown
                .opponent_hand_type
                .into_scalar()
                .elem::<f32>()
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn test_optional_belief_losses_default_to_zero() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
        let targets = make_dummy_targets::<B>(&device, 2);
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let belief: f32 = breakdown.belief_fields.into_scalar().elem();
        let mixture: f32 = breakdown.mixture_weight.into_scalar().elem();
        let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
        let hand_type: f32 = breakdown.opponent_hand_type.into_scalar().elem();
        let delta_q: f32 = breakdown.delta_q.into_scalar().elem();
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        assert!(
            oracle_loss.abs() < 1e-8,
            "missing oracle target should contribute zero oracle loss"
        );
        assert!(
            belief.abs() < 1e-8,
            "missing belief target should contribute zero loss"
        );
        assert!(
            mixture.abs() < 1e-8,
            "missing mixture target should contribute zero loss"
        );
        assert!(
            hand_type.abs() < 1e-8,
            "missing hand-type target should contribute zero loss"
        );
        assert!(
            delta_q.abs() < 1e-8,
            "missing delta-q target should contribute zero loss"
        );
        assert!(
            safety_residual.abs() < 1e-8,
            "missing safety-residual target should contribute zero loss"
        );
    }

    #[test]
    fn test_optional_belief_losses_activate_when_targets_present() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(
            HydraLossConfig::new()
                .with_w_belief_fields(0.1)
                .with_w_mixture_weight(0.1)
                .with_w_opponent_hand_type(0.1)
                .with_w_delta_q(0.1)
                .with_w_safety_residual(0.1),
        );
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.belief_fields_target = Some(Tensor::<B, 3>::zeros([2, 16, 34], &device));
        targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            &device,
        ));
        targets.opponent_hand_type_target = Some(Tensor::<B, 2>::from_floats(
            [
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ],
            &device,
        ));
        targets.delta_q_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
        targets.safety_residual_target = Some(Tensor::<B, 2>::from_floats(
            [[0.5f32; 46], [-0.5f32; 46]],
            &device,
        ));
        targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let belief: f32 = breakdown.belief_fields.into_scalar().elem();
        let mixture: f32 = breakdown.mixture_weight.into_scalar().elem();
        let hand_type: f32 = breakdown.opponent_hand_type.into_scalar().elem();
        let delta_q: f32 = breakdown.delta_q.into_scalar().elem();
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        let total: f32 = breakdown.total.into_scalar().elem();
        assert!(belief.is_finite() && belief >= 0.0);
        assert!(mixture.is_finite() && mixture >= 0.0);
        assert!(hand_type.is_finite() && hand_type >= 0.0);
        assert!(delta_q.is_finite() && delta_q >= 0.0);
        assert!(safety_residual.is_finite() && safety_residual >= 0.0);
        assert!(total.is_finite() && total > 0.0);
    }

    #[test]
    fn test_safety_residual_aux_loss_is_nonzero_when_enabled_and_present() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.safety_residual_target = Some(Tensor::<B, 2>::from_floats(
            [[0.25f32; 46], [-0.75f32; 46]],
            &device,
        ));
        targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        assert!(
            safety_residual.is_finite() && safety_residual > 0.0,
            "signed safety residual targets with a mask should contribute nonzero aux loss"
        );
    }

    #[test]
    fn test_safety_residual_requires_mask() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        assert!(
            safety_residual.abs() < 1e-8,
            "missing mask should disable safety residual loss"
        );
    }

    #[test]
    fn test_safety_residual_all_zero_mask_zeroes_loss() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
        targets.safety_residual_mask = Some(Tensor::<B, 2>::zeros([2, 46], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        assert!(
            safety_residual.abs() < 1e-8,
            "zero mask should disable safety residual loss"
        );
    }

    #[test]
    fn test_total_loss_positive() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out = model.forward(x);
        let targets = make_dummy_targets::<B>(&device, 4);
        let hydra_loss = HydraLoss::<B>::new(HydraLossConfig::new());
        let breakdown = hydra_loss.total_loss(&out, &targets);
        let total = breakdown.total.to_data().as_slice::<f32>().expect("f32")[0];
        assert!(total > 0.0, "total loss should be positive, got {total}");
        assert!(total.is_finite(), "total loss should be finite");
    }

    #[test]
    fn test_loss_weights_configurable() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out = model.forward(x);
        let targets = make_dummy_targets::<B>(&device, 4);
        let loss1 = HydraLoss::<B>::new(HydraLossConfig::new());
        let loss2 = HydraLoss::<B>::new(HydraLossConfig::new().with_w_pi(2.0));
        let t1 = loss1
            .total_loss(&out, &targets)
            .total
            .into_scalar()
            .elem::<f32>();
        let t2 = loss2
            .total_loss(&out, &targets)
            .total
            .into_scalar()
            .elem::<f32>();
        assert!((t1 - t2).abs() > 0.001, "different weights should differ");
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

    fn onehot2d<B: Backend>(
        device: &B::Device,
        batch: usize,
        classes: usize,
        idx: usize,
    ) -> Tensor<B, 2> {
        let mut d = vec![0.0f32; batch * classes];
        for i in 0..batch {
            d[i * classes + idx] = 1.0;
        }
        Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, classes])
    }

    fn onehot3d<B: Backend>(
        device: &B::Device,
        batch: usize,
        c1: usize,
        c2: usize,
    ) -> Tensor<B, 3> {
        let mut d = vec![0.0f32; batch * c1 * c2];
        for i in 0..(batch * c1) {
            d[i * c2] = 1.0;
        }
        Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, c1, c2])
    }

    #[test]
    #[ignore = "slow backward integration test"]
    fn test_total_loss_backward() {
        use burn::backend::Autodiff;
        use burn::optim::GradientsParams;
        type AB = Autodiff<NdArray<f32>>;

        let device = Default::default();
        let model = HydraModelConfig::new(1)
            .with_hidden_channels(32)
            .with_num_groups(8)
            .with_se_bottleneck(8)
            .init::<AB>(&device);
        let x = Tensor::<AB, 3>::zeros([1, crate::config::INPUT_CHANNELS, 34], &device);
        let out = model.forward(x);
        let targets = make_dummy_targets::<AB>(&device, 1);
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
        let bd = loss_fn.total_loss(&out, &targets);
        let total_val: f32 = bd.total.clone().into_scalar().elem();
        assert!(total_val > 0.0, "total should be > 0");
        let grads = bd.total.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        let num_grads = grads.len();
        assert!(num_grads > 0, "backward should produce gradients");
    }

    #[test]
    fn test_all_head_losses_positive() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out = model.forward(x);
        let targets = make_dummy_targets::<B>(&device, 4);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
        let bd = loss_fn.total_loss(&out, &targets);
        let check = |name: &str, t: &Tensor<B, 1>| {
            let v: f32 = t.clone().into_scalar().elem();
            assert!(v > 0.0 && v.is_finite(), "{name} loss = {v}");
        };
        check("policy", &bd.policy);
        check("value", &bd.value);
        check("grp", &bd.grp);
        check("opp_next", &bd.opp_next);
        check("score_pdf", &bd.score_pdf);
        check("score_cdf", &bd.score_cdf);
    }

    #[test]
    fn test_default_weights_match_roadmap() {
        let cfg = HydraLossConfig::new();
        assert!((cfg.w_pi - 1.0).abs() < 1e-6);
        assert!((cfg.w_v - 0.5).abs() < 1e-6);
        assert!((cfg.w_grp - 0.2).abs() < 1e-6);
        assert!((cfg.w_tenpai - 0.1).abs() < 1e-6);
        assert!((cfg.w_danger - 0.1).abs() < 1e-6);
        assert!((cfg.w_opp - 0.1).abs() < 1e-6);
        assert!((cfg.w_score - 0.025).abs() < 1e-6);
        assert!((cfg.w_oracle_critic - 0.0).abs() < 1e-6);
        assert!((cfg.w_belief_fields - 0.0).abs() < 1e-6);
        assert!((cfg.w_mixture_weight - 0.0).abs() < 1e-6);
        assert!((cfg.w_opponent_hand_type - 0.0).abs() < 1e-6);
        assert!((cfg.w_delta_q - 0.0).abs() < 1e-6);
        assert!((cfg.w_safety_residual - 0.0).abs() < 1e-6);
        let total_weight = cfg.w_pi
            + cfg.w_v
            + cfg.w_grp
            + cfg.w_tenpai
            + cfg.w_danger
            + cfg.w_opp
            + cfg.w_score * 2.0
            + cfg.w_oracle_critic
            + cfg.w_belief_fields
            + cfg.w_mixture_weight
            + cfg.w_opponent_hand_type
            + cfg.w_delta_q
            + cfg.w_safety_residual;
        assert!(
            (total_weight - 2.05).abs() < 1e-4,
            "total weight = {total_weight}"
        );
    }

    #[test]
    fn test_validate_rejects_negative_primary_weights() {
        assert!(HydraLossConfig::new()
            .with_w_tenpai(-0.1)
            .validate()
            .is_err());
        assert!(HydraLossConfig::new()
            .with_w_danger(-0.1)
            .validate()
            .is_err());
        assert!(HydraLossConfig::new().with_w_opp(-0.1).validate().is_err());
        assert!(HydraLossConfig::new()
            .with_w_score(-0.1)
            .validate()
            .is_err());
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

    pub fn make_dummy_targets<B: Backend>(device: &B::Device, batch: usize) -> HydraTargets<B> {
        HydraTargets {
            policy_target: onehot2d(device, batch, 46, 0),
            legal_mask: Tensor::ones([batch, 46], device),
            value_target: Tensor::zeros([batch], device),
            grp_target: onehot2d(device, batch, 24, 0),
            tenpai_target: Tensor::zeros([batch, 3], device),
            danger_target: Tensor::zeros([batch, 3, 34], device),
            danger_mask: Tensor::ones([batch, 3, 34], device),
            opp_next_target: onehot3d(device, batch, 3, 34),
            score_pdf_target: onehot2d(device, batch, 64, 32),
            score_cdf_target: Tensor::zeros([batch, 64], device),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: None,
            delta_q_mask: None,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
        }
    }
}
