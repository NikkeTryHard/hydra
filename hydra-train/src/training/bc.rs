//! Behavioral cloning training loop (Phase 0).

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::model::{HydraModel, HydraModelConfig};
use crate::training::losses::{HydraLoss, HydraTargets};

pub fn cosine_annealing_lr(step: usize, total_steps: usize, lr_max: f64, lr_min: f64) -> f64 {
    if total_steps == 0 {
        return lr_max;
    }
    let t = (step as f64 / total_steps as f64).min(1.0);
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

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

impl BCTrainerConfig {
    pub fn optimizer_config(&self) -> AdamConfig {
        AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.grad_clip_norm)))
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_agreement: f64,
    pub timestamp: u64,
    pub num_blocks: usize,
    pub hidden_channels: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::losses::{tests::make_dummy_targets, HydraLossConfig};
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::grad_clipping::GradientClippingConfig;
    use burn::optim::AdamConfig;

    type TestBackend = Autodiff<NdArray<f32>>;

    fn bc_optimizer() -> impl burn::optim::Optimizer<HydraModel<TestBackend>, TestBackend> {
        AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init()
    }

    #[test]
    fn test_bc_one_step() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([4, 85, 34], &device);
        let targets = make_dummy_targets::<TestBackend>(&device, 4);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = bc_optimizer();
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
        let mut optimizer = bc_optimizer();
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

    #[test]
    fn test_checkpoint_save_load() {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<NdArray<f32>>(&device);
        let x = Tensor::<NdArray<f32>, 3>::random(
            [2, 85, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out1 = model.forward(x.clone());
        let path = "/tmp/hydra_test_ckpt";
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.save_file(path, &recorder).expect("save failed");
        let loaded = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<NdArray<f32>>(&device)
            .load_file(path, &recorder, &device)
            .expect("load failed");
        let out2 = loaded.forward(x);
        let d1 = out1.policy_logits.to_data();
        let d2 = out2.policy_logits.to_data();
        let s1 = d1.as_slice::<f32>().expect("f32");
        let s2 = d2.as_slice::<f32>().expect("f32");
        for (i, (&a, &b)) in s1.iter().zip(s2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
        std::fs::remove_file(format!("{path}.mpk")).ok();
    }
}
