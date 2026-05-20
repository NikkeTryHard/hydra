//! Actor-Critic Hedge loss (LuckyJ's algorithm, ICLR 2022).

use crate::losses::{MASKED_LOGIT_SENTINEL, masked_logits};
use burn::prelude::*;
use burn::tensor::activation;
pub use hydra_train_types::config::AchConfig;

pub fn ach_policy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    legal_mask: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    pi_old: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    cfg: &AchConfig,
) -> Tensor<B, 1> {
    assert!(cfg.eta.is_finite(), "eta must be finite");
    assert!(cfg.eps.is_finite(), "eps must be finite");
    assert!(cfg.l_th.is_finite(), "l_th must be finite");
    assert!(cfg.beta_ent.is_finite(), "beta_ent must be finite");
    assert!(cfg.l_th > 0.0, "l_th must be positive");
    let masked_logits = masked_logits(logits, legal_mask.clone());

    let legal_sum = legal_mask.clone().sum_dim(1).clamp_min(1.0);
    let legal_mean = (masked_logits.clone() * legal_mask.clone()).sum_dim(1) / legal_sum;
    let centered = masked_logits - legal_mean;
    let clamped = centered.clamp(-cfg.l_th, cfg.l_th);

    let for_softmax = clamped.clone()
        + (legal_mask.clone().ones_like() - legal_mask.clone()) * MASKED_LOGIT_SENTINEL;
    let pi = activation::softmax(for_softmax, 1);

    let actions_2d = actions.unsqueeze_dim::<2>(1);
    let y_a = clamped.gather(1, actions_2d.clone()).squeeze_dim::<1>(1);
    let pi_a = pi.clone().gather(1, actions_2d).squeeze_dim::<1>(1);

    let pi_old_safe = pi_old.clone().clamp_min(1e-8);
    let ratio = pi_a.clone() / pi_old_safe.clone();

    let adv_pos = advantages.clone().clamp_min(0.0);
    let adv_neg = advantages.clone().clamp_max(0.0);
    let has_pos = adv_pos.clone().sign();
    let has_neg = adv_neg.clone().sign().neg();

    let gate_pos_ratio = ratio.clone().lower_elem(1.0 + cfg.eps).float();
    let gate_pos_logit = y_a.clone().lower_elem(cfg.l_th).float();
    let gate_pos = has_pos * gate_pos_ratio * gate_pos_logit;

    let gate_neg_ratio = ratio.clone().greater_elem(1.0 - cfg.eps).float();
    let gate_neg_logit = y_a.clone().greater_elem(-cfg.l_th).float();
    let gate_neg = has_neg * gate_neg_ratio * gate_neg_logit;

    let gate = gate_pos + gate_neg;

    let policy_loss = (gate * y_a / pi_old_safe * advantages).neg().mean();

    let log_pi = pi.clone().clamp(1e-8, 1.0).log();
    let entropy = (pi * log_pi * legal_mask).sum_dim(1).neg().mean();
    let ent_bonus = entropy * cfg.beta_ent;

    policy_loss * cfg.eta - ent_bonus
}

#[cfg(test)]
mod tests;
