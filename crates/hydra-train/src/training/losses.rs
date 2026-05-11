pub use hydra_train_algo::losses::{
    batch_kl_from_target, batch_policy_entropy, batch_value_variance, belief_fields_bce,
    belief_fields_bce_per_sample, compute_cvar, cross_entropy_soft, danger_focal_bce,
    dense_regression_mse, entropy, grad_norm_approx, grp_ce, kl_divergence, label_smoothing,
    loss_abs, loss_is_finite, masked_log_softmax, mean_entropy, mixture_weight_ce,
    mixture_weight_ce_per_sample, opp_next_ce, opponent_hand_type_ce,
    opponent_hand_type_ce_per_sample, oracle_critic_loss, oracle_critic_loss_per_sample,
    oracle_target_from_scores, policy_ce, policy_ce_with_temperature, score_cdf_bce, score_pdf_ce,
    soft_target_from_exit, tenpai_bce, value_mse, value_target_from_gae,
};
pub use hydra_train_algo::losses::{combine_sample_masks, masked_action_mse, masked_mean};
pub use hydra_train_exec::losses::HydraLoss;
pub use hydra_train_types::losses::{HydraLossConfig, HydraTargets, LossBreakdown};

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::model::{HydraModelConfig, HydraModelInit, HydraOutput};
    use burn::backend::NdArray;
    use burn::prelude::*;

    type B = NdArray<f32>;

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
    fn test_oracle_guidance_mask_intersects_belief_and_mixture_masks() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let mut first_only = make_dummy_targets::<B>(&device, 1);
        first_only.belief_fields_target = Some(Tensor::<B, 3>::ones([1, 16, 34], &device));
        first_only.belief_fields_mask = Some(Tensor::<B, 1>::ones([1], &device));
        first_only.mixture_weight_target =
            Some(Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 0.0]], &device));
        first_only.mixture_weight_mask = Some(Tensor::<B, 1>::ones([1], &device));

        let mut masked_targets = make_dummy_targets::<B>(&device, 2);
        masked_targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
        masked_targets.belief_fields_mask = Some(Tensor::<B, 1>::ones([2], &device));
        masked_targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            &device,
        ));
        masked_targets.mixture_weight_mask = Some(Tensor::<B, 1>::ones([2], &device));
        masked_targets.oracle_guidance_mask =
            Some(Tensor::<B, 1>::from_floats([1.0, 0.0], &device));

        let loss_fn = HydraLoss::<B>::new(
            HydraLossConfig::new()
                .with_w_belief_fields(1.0)
                .with_w_mixture_weight(1.0),
        );

        #[allow(
            clippy::single_range_in_vec_init,
            reason = "Burn slice API expects a one-element range slice"
        )]
        let first_outputs = HydraOutput {
            policy_logits: outputs.policy_logits.clone().slice([0..1]),
            value: outputs.value.clone().slice([0..1]),
            grp: outputs.grp.clone().slice([0..1]),
            opp_tenpai: outputs.opp_tenpai.clone().slice([0..1]),
            danger: outputs.danger.clone().slice([0..1]),
            opp_next_discard: outputs.opp_next_discard.clone().slice([0..1]),
            score_pdf: outputs.score_pdf.clone().slice([0..1]),
            score_cdf: outputs.score_cdf.clone().slice([0..1]),
            oracle_critic: outputs.oracle_critic.clone().slice([0..1]),
            belief_fields: outputs.belief_fields.clone().slice([0..1]),
            mixture_weight_logits: outputs.mixture_weight_logits.clone().slice([0..1]),
            opponent_hand_type: outputs.opponent_hand_type.clone().slice([0..1]),
            delta_q: outputs.delta_q.clone().slice([0..1]),
            safety_residual: outputs.safety_residual.clone().slice([0..1]),
        };

        let first_breakdown = loss_fn.total_loss(&first_outputs, &first_only);
        let masked_breakdown = loss_fn.total_loss(&outputs, &masked_targets);
        let belief_first: f32 = first_breakdown.belief_fields.into_scalar().elem();
        let mixture_first: f32 = first_breakdown.mixture_weight.into_scalar().elem();
        let belief_with: f32 = masked_breakdown.belief_fields.into_scalar().elem();
        let mixture_with: f32 = masked_breakdown.mixture_weight.into_scalar().elem();

        assert!(belief_first.is_finite() && belief_first > 0.0);
        assert!(mixture_first.is_finite() && mixture_first > 0.0);
        assert!(belief_with.is_finite() && belief_with > 0.0);
        assert!(mixture_with.is_finite() && mixture_with > 0.0);
        assert!((belief_with - belief_first).abs() < 1e-6);
        assert!((mixture_with - mixture_first).abs() < 1e-6);
    }

    #[test]
    fn test_optional_belief_losses_require_presence_masks() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x);
        let loss_fn = HydraLoss::<B>::new(
            HydraLossConfig::new()
                .with_w_belief_fields(1.0)
                .with_w_mixture_weight(1.0),
        );
        let mut targets = make_dummy_targets::<B>(&device, 2);
        targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
        targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            &device,
        ));

        let breakdown = loss_fn.total_loss(&outputs, &targets);
        assert!(breakdown.belief_fields.into_scalar().elem::<f32>().abs() < 1e-8);
        assert!(breakdown.mixture_weight.into_scalar().elem::<f32>().abs() < 1e-8);
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
    fn test_zero_weight_advanced_heads_keep_baseline_losses_unchanged() {
        let device = Default::default();
        use crate::model::HydraTrainModelExt;
        let model = HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let outputs = model.forward(x.clone());
        let optimized_outputs = model.forward_active_train(x, &HydraLossConfig::new());
        let targets = make_dummy_targets::<B>(&device, 2);
        let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
        let baseline = loss_fn.total_loss(&outputs, &targets);
        let optimized = loss_fn.total_loss(&optimized_outputs, &targets);

        let scalar = |t: Tensor<B, 1>| t.into_scalar().elem::<f32>();
        assert!((scalar(baseline.total) - scalar(optimized.total)).abs() < 1e-6);
        assert!((scalar(baseline.policy) - scalar(optimized.policy)).abs() < 1e-6);
        assert!((scalar(baseline.value) - scalar(optimized.value)).abs() < 1e-6);
        assert!((scalar(baseline.grp) - scalar(optimized.grp)).abs() < 1e-6);
        assert!((scalar(baseline.tenpai) - scalar(optimized.tenpai)).abs() < 1e-6);
        assert!((scalar(baseline.danger) - scalar(optimized.danger)).abs() < 1e-6);
        assert!((scalar(baseline.opp_next) - scalar(optimized.opp_next)).abs() < 1e-6);
        assert!((scalar(baseline.score_pdf) - scalar(optimized.score_pdf)).abs() < 1e-6);
        assert!((scalar(baseline.score_cdf) - scalar(optimized.score_cdf)).abs() < 1e-6);
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
            target_presence: None,
        }
    }
}
