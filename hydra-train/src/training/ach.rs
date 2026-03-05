//! Actor-Critic Hedge loss (LuckyJ's algorithm, ICLR 2022).

use burn::prelude::*;
use burn::tensor::activation;

#[derive(Config, Debug)]
pub struct AchConfig {
    #[config(default = "1.0")]
    pub eta: f32,
    #[config(default = "0.5")]
    pub eps: f32,
    #[config(default = "8.0")]
    pub l_th: f32,
    #[config(default = "5e-4")]
    pub beta_ent: f32,
}

pub fn ach_policy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    legal_mask: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    pi_old: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    cfg: &AchConfig,
) -> Tensor<B, 1> {
    let neg_inf_mask = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
    let masked_logits = logits + neg_inf_mask;

    let legal_sum = legal_mask.clone().sum_dim(1);
    let legal_mean = (masked_logits.clone() * legal_mask.clone()).sum_dim(1) / legal_sum;
    let centered = masked_logits - legal_mean;
    let clamped = centered.clamp(-cfg.l_th, cfg.l_th);

    let neg_inf_mask2 = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
    let for_softmax = clamped.clone() + neg_inf_mask2;
    let pi = activation::softmax(for_softmax, 1);

    let actions_2d = actions.unsqueeze_dim::<2>(1);
    let y_a = clamped.gather(1, actions_2d.clone()).squeeze_dim::<1>(1);
    let pi_a = pi.clone().gather(1, actions_2d).squeeze_dim::<1>(1);

    let ratio = pi_a.clone() / pi_old.clone();

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

    let policy_loss = (gate * y_a / pi_old * advantages).neg().mean();

    let log_pi = pi.clone().clamp(1e-8, 1.0).log();
    let entropy = (pi * log_pi * legal_mask).sum_dim(1).neg().mean();
    let ent_bonus = entropy * cfg.beta_ent;

    policy_loss * cfg.eta - ent_bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;
    type AchInputs = (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 1, Int>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    );

    fn make_ach_inputs(device: &<B as Backend>::Device) -> AchInputs {
        let logits = Tensor::<B, 2>::from_floats([[0.0, 1.0, -1.0]], device);
        let mask = Tensor::<B, 2>::ones([1, 3], device);
        let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], device);
        let pi_old = Tensor::<B, 1>::from_floats([0.5], device);
        let advantages = Tensor::<B, 1>::from_floats([1.0], device);
        (logits, mask, actions, pi_old, advantages)
    }

    #[test]
    fn test_ach_gate_positive_adv() {
        let device = Default::default();
        let (logits, mask, actions, pi_old, advantages) = make_ach_inputs(&device);
        let cfg = AchConfig::new();
        let loss = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.is_finite(), "ACH loss should be finite: {val}");
    }

    #[test]
    fn test_ach_gate_clips_ratio() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[0.0, 5.0, -5.0]], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
        let pi_old = Tensor::<B, 1>::from_floats([0.01], &device);
        let adv = Tensor::<B, 1>::from_floats([1.0], &device);
        let cfg = AchConfig::new();
        let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.is_finite());
    }

    #[test]
    fn test_ach_negative_adv() {
        let device = Default::default();
        let (logits, mask, actions, pi_old, _) = make_ach_inputs(&device);
        let neg_adv = Tensor::<B, 1>::from_floats([-1.0], &device);
        let cfg = AchConfig::new();
        let loss = ach_policy_loss(logits, mask, actions, pi_old, neg_adv, &cfg);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.is_finite());
    }

    #[test]
    fn test_ach_gate_clips_logit() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[0.0, 20.0, -20.0]], &device);
        let mask = Tensor::<B, 2>::ones([1, 3], &device);
        let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
        let pi_old = Tensor::<B, 1>::from_floats([0.5], &device);
        let adv = Tensor::<B, 1>::from_floats([1.0], &device);
        let cfg = AchConfig::new();
        let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.is_finite(), "clipped logit should produce finite loss");
    }

    #[test]
    fn test_ach_batch_of_8() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::random(
            [8, 46],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 2>::ones([8, 46], &device);
        let actions = Tensor::<B, 1, Int>::from_ints(&[0i32, 1, 2, 3, 4, 5, 6, 7][..], &device);
        let pi_old = Tensor::<B, 1>::from_floats([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2], &device);
        let adv = Tensor::<B, 1>::from_floats([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.1], &device);
        let cfg = AchConfig::new();
        let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
        let val = loss.into_scalar().elem::<f32>();
        assert!(val.is_finite(), "batch ACH should be finite: {val}");
    }
}
