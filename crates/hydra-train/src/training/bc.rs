//! Behavioral cloning training loop (Phase 0).

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsAccumulator, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::config::OracleGuidingConfig;
use crate::data::pipeline::collate_sample_chunk;
use crate::data::sample::MjaiSample;
use crate::model::{HydraModel, HydraModelConfig};
use crate::training::losses::{HydraLoss, HydraTargets};

pub fn phase_learning_rate(
    phase: crate::config::TrainingPhase,
    step: usize,
    total_steps: usize,
) -> f64 {
    use crate::config::TrainingPhase;
    let (lr_max, lr_min) = match phase {
        TrainingPhase::BcWarmStart => (2.5e-4, 1e-6),
        TrainingPhase::OracleGuiding => (1e-4, 1e-6),
        TrainingPhase::DrdaAchSelfPlay => (2.5e-4, 2.5e-5),
        TrainingPhase::ExitPondering => (1e-4, 1e-5),
        TrainingPhase::BenchmarkGates => (2.5e-4, 2.5e-4),
    };
    cosine_annealing_lr(step, total_steps, lr_max, lr_min)
}

pub fn warmup_then_cosine_lr(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    lr_max: f64,
    lr_min: f64,
) -> f64 {
    if step < warmup_steps {
        lr_max * (step as f64 / warmup_steps as f64)
    } else {
        cosine_annealing_lr(
            step - warmup_steps,
            total_steps - warmup_steps,
            lr_max,
            lr_min,
        )
    }
}

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
    #[config(default = "1e-6")]
    pub min_learning_rate: f64,
    #[config(default = "2048")]
    pub batch_size: usize,
    #[config(default = "1.0")]
    pub grad_clip_norm: f32,
    #[config(default = "1e-5")]
    pub weight_decay: f32,
    #[config(default = "1000")]
    pub warmup_steps: usize,
}

pub struct EpochStats {
    pub avg_loss: f64,
    pub policy_agreement: f64,
    pub num_batches: usize,
}

impl EpochStats {
    pub fn summary(&self) -> String {
        format!(
            "loss={:.4} agree={:.2}% batches={}",
            self.avg_loss,
            self.policy_agreement * 100.0,
            self.num_batches
        )
    }

    pub fn is_improving(&self, previous: &EpochStats) -> bool {
        self.avg_loss < previous.avg_loss
    }
}

pub struct OracleGuidingStepStats {
    pub skipped: bool,
    pub effective_lr: f64,
    pub oracle_keep_prob: f32,
    pub kept_oracle_fraction: f32,
    pub loss: Option<f64>,
}

pub fn policy_agreement<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> f64 {
    let (correct, total) = policy_agreement_counts(logits, mask, targets);
    correct as f64 / total as f64
}

pub fn policy_agreement_counts<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> (usize, usize) {
    let masked = logits + (mask.ones_like() - mask) * (-1e9f32);
    let predicted = masked.argmax(1).squeeze_dim::<1>(1);
    let correct = predicted.equal(targets);
    let total = correct.dims()[0];
    let correct = correct.int().sum().into_scalar().elem::<i64>() as usize;
    (correct, total)
}

pub fn target_actions_from_policy_target<B: Backend>(
    policy_target: Tensor<B, 2>,
) -> Tensor<B, 1, Int> {
    policy_target.argmax(1).squeeze_dim::<1>(1)
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

pub fn oracle_guidance_mask_values(
    batch_size: usize,
    keep_prob: f32,
    rng_values: &[f32],
) -> Vec<f32> {
    (0..batch_size)
        .map(|idx| {
            let sample = rng_values.get(idx).copied().unwrap_or(0.0);
            if sample < keep_prob { 1.0 } else { 0.0 }
        })
        .collect()
}

pub fn oracle_guidance_mask_tensor<B: Backend>(
    batch_size: usize,
    keep_prob: f32,
    rng_values: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let mask = oracle_guidance_mask_values(batch_size, keep_prob, rng_values);
    Tensor::<B, 1>::from_floats(mask.as_slice(), device)
}

#[allow(clippy::too_many_arguments)]
pub fn oracle_guiding_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    obs: Tensor<B, 3>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    base_lr: f64,
    oracle_cfg: &OracleGuidingConfig,
    step: usize,
    total_steps: usize,
    importance_weight: f32,
    max_importance_weight: f32,
    rng_values: &[f32],
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, OracleGuidingStepStats) {
    let oracle_keep_prob = oracle_cfg.dropout_at_step(step, total_steps);
    let effective_lr = oracle_cfg.effective_learning_rate(base_lr, step, total_steps);

    if oracle_cfg.should_reject_importance_weight(
        importance_weight,
        max_importance_weight,
        step,
        total_steps,
    ) {
        return (
            model,
            OracleGuidingStepStats {
                skipped: true,
                effective_lr,
                oracle_keep_prob,
                kept_oracle_fraction: 0.0,
                loss: None,
            },
        );
    }

    let batch_size = obs.dims()[0];
    let device = obs.device();
    let oracle_mask =
        oracle_guidance_mask_tensor::<B>(batch_size, oracle_keep_prob, rng_values, &device);
    let kept_oracle_fraction =
        oracle_mask.clone().sum().into_scalar().elem::<f32>() / batch_size as f32;
    let mut masked_targets = targets.clone();
    masked_targets.oracle_guidance_mask = Some(oracle_mask);

    let (model, loss) = bc_train_step(
        model,
        obs,
        &masked_targets,
        loss_fn,
        effective_lr,
        optimizer,
    );
    (
        model,
        OracleGuidingStepStats {
            skipped: false,
            effective_lr,
            oracle_keep_prob,
            kept_oracle_fraction,
            loss: Some(loss),
        },
    )
}

impl BCTrainerConfig {
    pub fn summary(&self) -> String {
        format!(
            "lr={:.1e} min_lr={:.1e} batch={} clip={:.1} wd={:.1e}",
            self.lr,
            self.min_learning_rate,
            self.batch_size,
            self.grad_clip_norm,
            self.weight_decay
        )
    }

    pub fn default_actor() -> Self {
        Self::new(crate::model::HydraModelConfig::actor())
    }

    pub fn estimated_training_time_hours(&self, num_samples: usize, samples_per_sec: f32) -> f32 {
        if samples_per_sec <= 0.0 {
            return 0.0;
        }
        num_samples as f32 / samples_per_sec / 3600.0
    }

    pub fn num_epochs_for(&self, num_samples: usize) -> usize {
        let batches = self.total_batches(num_samples);
        if batches == 0 {
            return 0;
        }
        batches
    }

    pub fn default_learner() -> Self {
        Self::new(crate::model::HydraModelConfig::learner())
    }

    pub fn is_warmup(&self, step: usize) -> bool {
        step < self.warmup_steps
    }

    pub fn effective_lr(&self, step: usize, total_steps: usize) -> f64 {
        warmup_then_cosine_lr(
            step,
            self.warmup_steps,
            total_steps,
            self.lr,
            self.min_learning_rate,
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        if self.min_learning_rate <= 0.0 {
            return Err("min_learning_rate must be positive");
        }
        if self.min_learning_rate > self.lr {
            return Err("min_learning_rate must be <= lr");
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0");
        }
        if self.grad_clip_norm <= 0.0 {
            return Err("grad_clip_norm must be positive");
        }
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative");
        }
        if self.warmup_steps == 0 {
            return Err("warmup_steps must be > 0");
        }
        Ok(())
    }

    pub fn total_batches(&self, num_samples: usize) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        num_samples / self.batch_size
    }

    pub fn optimizer_config(&self) -> AdamConfig {
        AdamConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                self.weight_decay,
            )))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.grad_clip_norm)))
    }
}

/// Run one epoch of behavioral cloning with optional gradient accumulation.
///
/// `microbatch_size` is the physical batch size that goes through forward/backward
/// at once (controls peak VRAM). `accum_steps` is how many microbatches to
/// accumulate before one optimizer step. The effective logical batch size is
/// `microbatch_size * accum_steps`.
///
/// For backwards compatibility: set `accum_steps = 1` and `microbatch_size =
/// batch_size` to get the original behavior.
#[allow(clippy::too_many_arguments)]
pub fn train_epoch<B: AutodiffBackend>(
    model: HydraModel<B>,
    samples: &[&MjaiSample],
    microbatch_size: usize,
    accum_steps: usize,
    augment: bool,
    device: &B::Device,
    loss_fn: &HydraLoss<B>,
    lr: f64,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, EpochStats)
where
    HydraModel<B>: AutodiffModule<B>,
{
    let accum_steps = accum_steps.max(1);
    let mut m = model;
    let mut total_loss = 0.0;
    let mut total_agreement = 0.0;
    let mut num_batches = 0usize;
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut accum_current = 0usize;
    let mut accum_loss = 0.0;
    let mut accum_agreement = 0.0;

    for chunk in samples.chunks(microbatch_size) {
        let Some((obs, targets)) = collate_sample_chunk::<B>(chunk, augment, device) else {
            continue;
        };
        let output = m.forward(obs.clone());
        accum_agreement += policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            target_actions_from_policy_target(targets.policy_target.clone()),
        );
        let breakdown = loss_fn.total_loss(&output, &targets);
        let loss = breakdown.total.clone().into_scalar().elem::<f64>();
        let grads = breakdown.total.backward();
        let grads = GradientsParams::from_grads(grads, &m);
        accumulator.accumulate(&m, grads);
        accum_loss += loss;
        accum_current += 1;

        if accum_current >= accum_steps {
            let grads = accumulator.grads();
            m = optimizer.step(lr, m, grads);
            total_loss += accum_loss / accum_current as f64;
            total_agreement += accum_agreement / accum_current as f64;
            num_batches += 1;
            accum_current = 0;
            accum_loss = 0.0;
            accum_agreement = 0.0;
        }
    }

    // Flush any remaining accumulated microbatches
    if accum_current > 0 {
        let grads = accumulator.grads();
        m = optimizer.step(lr, m, grads);
        total_loss += accum_loss / accum_current as f64;
        total_agreement += accum_agreement / accum_current as f64;
        num_batches += 1;
    }

    let stats = EpochStats {
        avg_loss: if num_batches == 0 {
            0.0
        } else {
            total_loss / num_batches as f64
        },
        policy_agreement: if num_batches == 0 {
            0.0
        } else {
            total_agreement / num_batches as f64
        },
        num_batches,
    };
    (m, stats)
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_agreement: Option<f64>,
    pub eval_policy_loss: Option<f64>,
    pub eval_total_loss: Option<f64>,
    pub timestamp: u64,
    pub num_blocks: usize,
    pub hidden_channels: usize,
}

impl CheckpointMeta {
    pub fn new(
        epoch: u32,
        train_loss: f64,
        eval_agreement: Option<f64>,
        eval_policy_loss: Option<f64>,
        eval_total_loss: Option<f64>,
    ) -> Self {
        Self {
            epoch,
            train_loss,
            eval_agreement,
            eval_policy_loss,
            eval_total_loss,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            num_blocks: 24,
            hidden_channels: 256,
        }
    }

    pub fn summary(&self) -> String {
        let eval_summary = match (
            self.eval_policy_loss,
            self.eval_total_loss,
            self.eval_agreement,
        ) {
            (Some(policy_loss), Some(total_loss), Some(agreement)) => format!(
                "policy_ce={policy_loss:.4} total={total_loss:.4} agree={:.2}%",
                agreement * 100.0
            ),
            _ => "eval=n/a".to_string(),
        };
        format!(
            "epoch={} loss={:.4} {}",
            self.epoch, self.train_loss, eval_summary
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::losses::{HydraLossConfig, tests::make_dummy_targets};
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
    fn test_bc_config_defaults() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::actor());
        assert!((cfg.lr - 2.5e-4).abs() < 1e-10);
        assert!((cfg.min_learning_rate - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.batch_size, 2048);
        assert!((cfg.grad_clip_norm - 1.0).abs() < 1e-6);
        assert!((cfg.weight_decay - 1e-5).abs() < 1e-8);
        assert_eq!(cfg.warmup_steps, 1000);
    }

    #[test]
    fn effective_lr_respects_configured_min_learning_rate() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::actor())
            .with_lr(1e-3)
            .with_min_learning_rate(1e-4)
            .with_warmup_steps(10);
        let lr = cfg.effective_lr(1_000, 1_000);
        assert!(
            lr >= 1e-4 - 1e-10,
            "lr floor should respect configured min: {lr}"
        );
    }

    #[test]
    fn test_bc_one_step() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([4, crate::config::INPUT_CHANNELS, 34], &device);
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
            [10, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let targets = make_dummy_targets::<TestBackend>(&device, 10);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = bc_optimizer();
        let mut last_loss = f64::MAX;
        for _ in 0..100 {
            let (m, loss) =
                bc_train_step(model, obs.clone(), &targets, &loss_fn, 1e-3, &mut optimizer);
            model = m;
            last_loss = loss;
        }
        assert!(last_loss < 5.0, "should overfit: loss={last_loss}");
    }

    #[test]
    fn test_oracle_guidance_mask_values_follow_keep_probability() {
        let mask = oracle_guidance_mask_values(4, 0.5, &[0.1, 0.7, 0.49, 0.9]);
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_oracle_guiding_train_step_skips_large_importance_post_dropout() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let mut targets = make_dummy_targets::<TestBackend>(&device, 2);
        targets.oracle_target = Some(Tensor::<TestBackend, 2>::ones([2, 4], &device));
        let loss_fn =
            HydraLoss::<TestBackend>::new(HydraLossConfig::new().with_w_oracle_critic(1.0));
        let mut optimizer = bc_optimizer();
        let oracle_cfg = crate::config::OracleGuidingConfig::default();

        let (_, stats) = oracle_guiding_train_step(
            model,
            obs,
            &targets,
            &loss_fn,
            1e-4,
            &oracle_cfg,
            100,
            100,
            3.0,
            2.0,
            &[0.1, 0.9],
            &mut optimizer,
        );

        assert!(stats.skipped);
        assert!(stats.loss.is_none());
        assert!((stats.effective_lr - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn test_oracle_guiding_train_step_applies_dropout_mask_and_trains() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [2, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let mut targets = make_dummy_targets::<TestBackend>(&device, 2);
        targets.oracle_target = Some(Tensor::<TestBackend, 2>::ones([2, 4], &device));
        targets.belief_fields_target = Some(Tensor::<TestBackend, 3>::ones([2, 16, 34], &device));
        let loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_oracle_critic(1.0)
                .with_w_belief_fields(1.0),
        );
        let mut optimizer = bc_optimizer();
        let oracle_cfg = crate::config::OracleGuidingConfig::default();

        let (_, stats) = oracle_guiding_train_step(
            model,
            obs,
            &targets,
            &loss_fn,
            1e-4,
            &oracle_cfg,
            50,
            100,
            1.0,
            2.0,
            &[0.0, 0.9],
            &mut optimizer,
        );

        assert!(!stats.skipped);
        assert!(stats.loss.expect("loss") > 0.0);
        assert!((stats.oracle_keep_prob - 0.5).abs() < 1e-6);
        assert!((stats.kept_oracle_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_policy_agreement_range() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let model = HydraModelConfig::actor().init::<NdArray<f32>>(&device);
        let x = Tensor::<NdArray<f32>, 3>::random(
            [32, crate::config::INPUT_CHANNELS, 34],
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
    fn test_policy_agreement_counts_match_fraction() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<NdArray<f32>, 2>::from_floats(
            [
                [10.0, 1.0, 0.0],
                [0.0, 9.0, 1.0],
                [0.0, 1.0, 8.0],
                [3.0, 2.0, 1.0],
            ],
            &device,
        );
        let mask = Tensor::<NdArray<f32>, 2>::ones([4, 3], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints([0, 1, 1, 2], &device);

        let acc = policy_agreement(logits.clone(), mask.clone(), targets.clone());
        let (correct, total) = policy_agreement_counts(logits, mask, targets);

        assert_eq!((correct, total), (2, 4));
        assert!((acc - correct as f64 / total as f64).abs() < 1e-12);
    }

    #[test]
    fn test_train_epoch_reports_policy_agreement() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = model.forward(obs.clone());
        let predicted = output
            .policy_logits
            .clone()
            .argmax(1)
            .squeeze_dim::<1>(1)
            .to_data();
        let predicted = predicted.as_slice::<i64>().expect("i64");
        let mut policy_target = vec![0.0f32; predicted.len() * 46];
        for (row, &action) in predicted.iter().enumerate() {
            policy_target[row * 46 + action as usize] = 1.0;
        }

        let mut targets = make_dummy_targets::<TestBackend>(&device, predicted.len());
        targets.policy_target =
            Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device)
                .reshape([predicted.len(), 46]);

        let acc = policy_agreement(
            output.policy_logits,
            targets.legal_mask,
            target_actions_from_policy_target(targets.policy_target),
        );
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "agreement should reflect matching targets, got {acc}"
        );
    }

    #[test]
    fn test_bc_step_advanced_aux_targets_change_loss() {
        let device = Default::default();
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let model1 = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let baseline_targets = make_dummy_targets::<TestBackend>(&device, 4);
        let baseline_loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut opt1 = bc_optimizer();
        let (_, loss_baseline) = bc_train_step(
            model1,
            obs.clone(),
            &baseline_targets,
            &baseline_loss_fn,
            1e-3,
            &mut opt1,
        );

        let model2 = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let mut advanced_targets = make_dummy_targets::<TestBackend>(&device, 4);
        advanced_targets.belief_fields_target =
            Some(Tensor::<TestBackend, 3>::ones([4, 16, 34], &device));
        advanced_targets.belief_fields_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        advanced_targets.safety_residual_target = Some(Tensor::<TestBackend, 2>::from_floats(
            [[0.5f32; 46], [-0.5f32; 46], [0.3f32; 46], [-0.3f32; 46]],
            &device,
        ));
        advanced_targets.safety_residual_mask =
            Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        advanced_targets.delta_q_target = Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        advanced_targets.oracle_target = Some(Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, -0.1, 0.05, -0.05],
                [0.2, -0.2, 0.1, -0.1],
                [0.05, -0.05, 0.0, 0.0],
                [-0.1, 0.1, -0.05, 0.05],
            ],
            &device,
        ));
        advanced_targets.oracle_guidance_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        let advanced_loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_oracle_critic(0.1)
                .with_w_belief_fields(0.1)
                .with_w_safety_residual(0.1)
                .with_w_delta_q(0.05),
        );
        let mut opt2 = bc_optimizer();
        let (_, loss_advanced) = bc_train_step(
            model2,
            obs,
            &advanced_targets,
            &advanced_loss_fn,
            1e-3,
            &mut opt2,
        );

        assert!(
            loss_baseline.is_finite(),
            "baseline BC loss should be finite"
        );
        assert!(
            loss_advanced.is_finite(),
            "advanced BC loss should be finite"
        );
        assert!(
            (loss_baseline - loss_advanced).abs() > 1e-6,
            "advanced aux targets should change BC loss: baseline={loss_baseline}, advanced={loss_advanced}"
        );
    }

    #[test]
    fn test_bc_train_epoch_with_advanced_targets() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let mut targets = make_dummy_targets::<TestBackend>(&device, 4);
        targets.belief_fields_target = Some(Tensor::<TestBackend, 3>::ones([4, 16, 34], &device));
        targets.belief_fields_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        targets.safety_residual_target =
            Some(Tensor::<TestBackend, 2>::ones([4, 46], &device) * 0.25);
        targets.safety_residual_mask = Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        targets.delta_q_target = Some(Tensor::<TestBackend, 2>::zeros([4, 46], &device));

        let loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_belief_fields(0.1)
                .with_w_safety_residual(0.1)
                .with_w_delta_q(0.05),
        );
        let mut optimizer = bc_optimizer();
        let (_, stats) = bc_train_step(model, obs, &targets, &loss_fn, 1e-3, &mut optimizer);

        assert!(stats.is_finite(), "loss should be finite");
        assert!(stats > 0.0, "loss should be positive with advanced targets");
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
            [2, crate::config::INPUT_CHANNELS, 34],
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

    #[test]
    fn checkpoint_meta_summary_handles_missing_eval_metrics() {
        let meta = CheckpointMeta::new(10, 2.5, None, None, None);
        assert_eq!(meta.summary(), "epoch=10 loss=2.5000 eval=n/a");
    }

    #[test]
    fn checkpoint_meta_summary_reports_eval_metrics() {
        let meta = CheckpointMeta::new(10, 2.5, Some(0.375), Some(1.75), Some(2.25));
        assert_eq!(
            meta.summary(),
            "epoch=10 loss=2.5000 policy_ce=1.7500 total=2.2500 agree=37.50%"
        );
    }
}
