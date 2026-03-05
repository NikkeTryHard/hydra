//! Full RL training step: DRDA-wrapped ACH with auxiliary losses.

use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::model::HydraModel;
use crate::training::ach::{ach_policy_loss, AchConfig};
use crate::training::drda;
use crate::training::losses::{HydraLoss, HydraTargets};

pub const MAX_RL_BATCH_SIZE: usize = 512;
pub const ONE_EPOCH_ONLY: bool = true;
pub const DEFAULT_EXIT_WEIGHT: f32 = 0.5;
pub const DEFAULT_AUX_WEIGHT: f32 = 0.1;

pub struct RlBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub actions: Tensor<B, 1, Int>,
    pub pi_old: Tensor<B, 1>,
    pub advantages: Tensor<B, 1>,
    pub base_logits: Tensor<B, 2>,
    pub targets: HydraTargets<B>,
    pub exit_target: Option<Tensor<B, 2>>,
}

pub struct RlConfig {
    pub tau_drda: f32,
    pub ach_cfg: AchConfig,
    pub lr: f64,
    pub exit_weight: f32,
    pub aux_weight: f32,
}

impl RlConfig {
    pub fn default_phase2() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 2.5e-4,
            exit_weight: 0.0,
            aux_weight: 0.1,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    pub fn with_exit_weight(mut self, w: f32) -> Self {
        self.exit_weight = w;
        self
    }
    pub fn with_aux_weight(mut self, w: f32) -> Self {
        self.aux_weight = w;
        self
    }

    pub fn default_phase3() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 1e-4,
            exit_weight: 0.5,
            aux_weight: 0.1,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "rl(tau={:.1}, lr={:.1e}, exit_w={:.2}, aux_w={:.2})",
            self.tau_drda, self.lr, self.exit_weight, self.aux_weight
        )
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tau_drda < crate::training::drda::MIN_TAU_DRDA {
            return Err("tau_drda below minimum");
        }
        self.ach_cfg.validate()?;
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        Ok(())
    }
}

pub fn rl_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    let output = model.forward(batch.obs.clone());
    let combined = drda::combined_logits(
        batch.base_logits.clone(),
        output.policy_logits.clone(),
        cfg.tau_drda,
    );
    let adv = batch.advantages.clone();
    let adv_mean = adv.clone().mean();
    let adv_var = (adv.clone() - adv_mean.clone()).powf_scalar(2.0).mean();
    let adv_std = (adv_var + 1e-8).sqrt();
    let advantages_normed = (adv - adv_mean) / adv_std;
    let ach_loss = ach_policy_loss(
        combined,
        batch.targets.legal_mask.clone(),
        batch.actions.clone(),
        batch.pi_old.clone(),
        advantages_normed,
        &cfg.ach_cfg,
    );
    let aux = loss_fn.total_loss(&output, &batch.targets);
    let mut total = ach_loss + aux.total * cfg.aux_weight;
    if let Some(ref exit_target) = batch.exit_target {
        let exit_loss = crate::training::exit::exit_loss(
            output.policy_logits,
            exit_target.clone(),
            batch.targets.legal_mask.clone(),
            cfg.exit_weight,
        );
        total = total + exit_loss;
    }
    let loss_val = total.clone().into_scalar().elem::<f64>();
    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(cfg.lr, model, grads);
    (model, loss_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HydraModelConfig;
    use crate::training::losses::{tests::make_dummy_targets, HydraLossConfig};
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::AdamConfig;

    type AB = Autodiff<NdArray<f32>>;

    #[test]
    fn test_rl_step_finite() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let batch = RlBatch {
            obs: Tensor::<AB, 3>::zeros([2, 85, 34], &device),
            actions: Tensor::<AB, 1, Int>::from_ints(&[0i32, 1][..], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.3], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -0.5], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: make_dummy_targets::<AB>(&device, 2),
            exit_target: None,
        };
        let cfg = RlConfig {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 1e-4,
            exit_weight: 0.5,
            aux_weight: 0.1,
        };
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
        let mut optimizer = AdamConfig::new().init();
        let (_, loss) = rl_step(model, &batch, &cfg, &loss_fn, &mut optimizer);
        assert!(loss.is_finite(), "RL step loss should be finite: {loss}");
    }

    #[test]
    fn test_rl_two_steps_change_loss() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let batch = RlBatch {
            obs: Tensor::<AB, 3>::random(
                [2, 85, 34],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            ),
            actions: Tensor::<AB, 1, Int>::from_ints(&[0i32, 1][..], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.3], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -0.5], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: make_dummy_targets::<AB>(&device, 2),
            exit_target: None,
        };
        let cfg = RlConfig {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 1e-3,
            exit_weight: 0.5,
            aux_weight: 0.1,
        };
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
        let mut opt = AdamConfig::new().init();
        let (m1, l1) = rl_step(model, &batch, &cfg, &loss_fn, &mut opt);
        let (_, l2) = rl_step(m1, &batch, &cfg, &loss_fn, &mut opt);
        assert!(l1.is_finite() && l2.is_finite());
        assert!((l1 - l2).abs() > 1e-8, "two steps should change loss");
    }
}
