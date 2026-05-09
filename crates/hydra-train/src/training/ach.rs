//! Actor-Critic Hedge loss compatibility exports.

pub use hydra_train_algo::ach::*;

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};
    use burn::prelude::*;

    type AB = Autodiff<NdArray<f32>>;

    #[test]
    fn test_ach_one_epoch_changes_weights() {
        use crate::model::HydraModelConfig;
        use crate::training::losses::{HydraLoss, HydraLossConfig, tests::make_dummy_targets};
        use crate::training::rl::{RlBatch, RlConfig, rl_step};
        use burn::optim::AdamConfig;

        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);

        let obs = Tensor::<AB, 3>::random(
            [2, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let out_before = model.forward(obs.clone());
        let val_before: f32 = out_before.value.clone().mean().into_scalar().elem();

        let batch = RlBatch {
            obs: obs.clone(),
            actions: Tensor::<AB, 1, Int>::from_ints(&[0i32, 1][..], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.3], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -0.5], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: make_dummy_targets::<AB>(&device, 2),
            exit_target: None,
            exit_mask: None,
        };
        let cfg = RlConfig::default_phase2().with_lr(1e-3);
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
        let mut opt = AdamConfig::new().init();

        let (model_after, _) = rl_step(model, &batch, &cfg, &loss_fn, &mut opt);

        let out_after = model_after.forward(obs);
        let val_after: f32 = out_after.value.clone().mean().into_scalar().elem();

        assert!(
            (val_before - val_after).abs() > 1e-8,
            "one ACH epoch must change weights: before={val_before}, after={val_after}"
        );
    }
}
