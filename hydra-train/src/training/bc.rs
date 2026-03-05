//! Behavioral cloning training loop (Phase 0).

use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::model::{HydraModel, HydraModelConfig};
use crate::training::losses::{HydraLoss, HydraTargets};

#[derive(Config, Debug)]
pub struct BCTrainerConfig {
    pub model_config: HydraModelConfig,
    #[config(default = "2.5e-4")]
    pub lr: f64,
    #[config(default = "2048")]
    pub batch_size: usize,
    #[config(default = "1.0")]
    pub grad_clip_norm: f32,
}

pub struct EpochStats {
    pub avg_loss: f64,
    pub policy_agreement: f64,
    pub num_batches: usize,
}

pub fn policy_agreement<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> f64 {
    let masked = logits + (mask.ones_like() - mask) * (-1e9f32);
    let predicted = masked.argmax(1).squeeze_dim::<1>(1);
    let correct = predicted.equal(targets);
    let n = correct.dims()[0] as f64;
    correct.int().sum().into_scalar().elem::<i64>() as f64 / n
}

pub fn bc_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    obs: Tensor<B, 3>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    lr: f64,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    let output = model.forward(obs);
    let breakdown = loss_fn.total_loss(&output, targets);
    let loss_val = breakdown.total.clone().into_scalar().elem::<f64>();
    let grads = breakdown.total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(lr, model, grads);
    (model, loss_val)
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_agreement: f64,
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::losses::{tests::make_dummy_targets, HydraLossConfig};
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::optim::AdamConfig;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_bc_one_step() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([4, 85, 34], &device);
        let targets = make_dummy_targets::<TestBackend>(&device, 4);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = AdamConfig::new().init();
        let (_, loss1) = bc_train_step(model, obs, &targets, &loss_fn, 1e-3, &mut optimizer);
        assert!(loss1.is_finite(), "loss should be finite: {loss1}");
        assert!(loss1 > 0.0, "loss should be positive: {loss1}");
    }

    #[test]
    fn test_bc_overfit_10_samples() {
        let device = Default::default();
        let mut model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [10, 85, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let targets = make_dummy_targets::<TestBackend>(&device, 10);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = AdamConfig::new().init();
        let mut last_loss = f64::MAX;
        for _ in 0..50 {
            let (m, loss) =
                bc_train_step(model, obs.clone(), &targets, &loss_fn, 1e-3, &mut optimizer);
            model = m;
            last_loss = loss;
        }
        assert!(last_loss < 5.0, "should overfit: loss={last_loss}");
    }

    #[test]
    fn test_policy_agreement_range() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let model = HydraModelConfig::actor().init::<NdArray<f32>>(&device);
        let x = Tensor::<NdArray<f32>, 3>::random(
            [32, 85, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = model.forward(x);
        let mask = Tensor::<NdArray<f32>, 2>::ones([32, 46], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints(&[0i32; 32][..], &device);
        let acc = policy_agreement(output.policy_logits, mask, targets);
        assert!((0.0..=1.0).contains(&acc), "agreement {acc} out of [0,1]");
    }
}
