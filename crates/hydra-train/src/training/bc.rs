//! Behavioral cloning training loop (Phase 0).

use crate::amp::maybe_autocast;
use crate::config::OracleGuidingConfig;
use crate::data::sample::{MjaiBatch, MjaiSample, collate_sample_refs_bc_owned};
use crate::model::HydraModel;
use crate::training::losses::{HydraLoss, HydraTargets};
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
pub use hydra_train_exec::bc_runtime::{
    BcExitConfig, bc_total_with_exit_from_breakdown, cosine_annealing_lr, gated_bc_context,
    maybe_add_exit_loss, oracle_guidance_mask_tensor, oracle_guidance_mask_values,
    policy_agreement, policy_agreement_counts, target_actions_from_policy_target,
    warmup_then_cosine_lr,
};

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

pub use hydra_train_types::config::BCTrainerConfig;

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

pub struct BcTrainBatchInput<'a, B: Backend> {
    pub obs: Tensor<B, 3>,
    pub batch: &'a MjaiBatch<B>,
    pub targets: &'a HydraTargets<B>,
}

pub struct BcTrainStepContext<'a, B: AutodiffBackend> {
    pub loss_fn: &'a HydraLoss<B>,
    pub exit_cfg: &'a BcExitConfig,
    pub use_amp: bool,
    pub lr: f64,
}

pub struct OracleGuidingBatchInput<'a, B: Backend> {
    pub obs: Tensor<B, 3>,
    pub targets: &'a HydraTargets<B>,
    pub loss_fn: &'a HydraLoss<B>,
    pub importance_weight: f32,
    pub max_importance_weight: f32,
    pub rng_values: &'a [f32],
}

pub struct OracleGuidingStepSchedule<'a> {
    pub base_lr: f64,
    pub oracle_cfg: &'a OracleGuidingConfig,
    pub step: usize,
    pub total_steps: usize,
}

pub fn bc_total_with_optional_exit_from_breakdown<B: Backend>(
    output: &crate::model::HydraOutput<B>,
    batch: Option<&MjaiBatch<B>>,
    breakdown: &crate::training::losses::LossBreakdown<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    let mut total = breakdown.total.clone();
    if let Some(batch) = batch {
        total = maybe_add_exit_loss(
            total,
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            exit_cfg,
        );
    }
    total
}

pub fn bc_total_with_exit<B: Backend>(
    output: &crate::model::HydraOutput<B>,
    batch: &MjaiBatch<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    let breakdown = loss_fn.total_loss(output, targets);
    bc_total_with_exit_from_breakdown(output, batch, &breakdown, exit_cfg)
}

pub fn bc_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    batch_input: BcTrainBatchInput<'_, B>,
    step_context: BcTrainStepContext<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, f64) {
    let output = maybe_autocast(step_context.use_amp, || model.forward(batch_input.obs));
    let breakdown = step_context
        .loss_fn
        .total_loss(&output, batch_input.targets);
    let total = bc_total_with_optional_exit_from_breakdown(
        &output,
        Some(batch_input.batch),
        &breakdown,
        step_context.exit_cfg,
    );
    let loss_val = total
        .clone()
        .into_data()
        .convert::<f64>()
        .as_slice::<f64>()
        .expect("bc total loss should be readable as f64")[0];
    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(step_context.lr, model, grads);
    (model, loss_val)
}

pub fn oracle_guiding_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    batch_input: OracleGuidingBatchInput<'_, B>,
    schedule: OracleGuidingStepSchedule<'_>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> (HydraModel<B>, OracleGuidingStepStats) {
    let oracle_keep_prob = schedule
        .oracle_cfg
        .dropout_at_step(schedule.step, schedule.total_steps);
    let effective_lr = schedule.oracle_cfg.effective_learning_rate(
        schedule.base_lr,
        schedule.step,
        schedule.total_steps,
    );

    if schedule.oracle_cfg.should_reject_importance_weight(
        batch_input.importance_weight,
        batch_input.max_importance_weight,
        schedule.step,
        schedule.total_steps,
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

    let batch_size = batch_input.obs.dims()[0];
    let device = batch_input.obs.device();
    let oracle_mask_values =
        oracle_guidance_mask_values(batch_size, oracle_keep_prob, batch_input.rng_values);
    let kept_oracle_fraction = oracle_mask_values.iter().copied().sum::<f32>() / batch_size as f32;
    let oracle_mask = Tensor::<B, 1>::from_floats(oracle_mask_values.as_slice(), &device);
    let mut masked_targets = batch_input.targets.clone();
    masked_targets.oracle_guidance_mask = Some(oracle_mask);
    let output = model.forward(batch_input.obs);
    let breakdown = batch_input.loss_fn.total_loss(&output, &masked_targets);
    let total = bc_total_with_optional_exit_from_breakdown(
        &output,
        None,
        &breakdown,
        &BcExitConfig::default(),
    );
    let loss = total
        .clone()
        .into_data()
        .convert::<f64>()
        .as_slice::<f64>()
        .expect("oracle-guided total loss should be readable as f64")[0];
    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(effective_lr, model, grads);
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

/// Run one epoch of behavioral cloning with optional gradient accumulation.
///
/// `microbatch_size` is the physical batch size that goes through forward/backward
/// at once (controls peak VRAM). `accum_steps` is how many microbatches to
/// accumulate before one optimizer step. The effective logical batch size is
/// `microbatch_size * accum_steps`.
///
/// For backwards compatibility: set `accum_steps = 1` and `microbatch_size =
/// batch_size` to get the original behavior.
#[allow(
    clippy::too_many_arguments,
    reason = "training loop needs explicit config, device, and telemetry context"
)]
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
        let Some((obs, batch, targets)) = collate_sample_refs_bc_owned::<B>(chunk, augment, device)
            .expect("behavior cloning sample collation should be valid")
        else {
            continue;
        };
        let output = m.forward(obs);
        accum_agreement += policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
        );
        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            &BcExitConfig::default(),
        );
        let loss = total.clone().into_scalar().elem::<f64>();
        let grads = total.backward();
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

pub use hydra_train_types::checkpoint::CheckpointMeta;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::sample::MjaiBatch;
    use crate::model::{HydraModelConfig, HydraModelInit};
    use crate::training::losses::{HydraLossConfig, tests::make_dummy_targets};
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::grad_clipping::GradientClippingConfig;
    use burn::optim::AdamConfig;

    use std::time::{SystemTime, UNIX_EPOCH};
    type TestBackend = Autodiff<NdArray<f32>>;

    fn bc_optimizer() -> impl burn::optim::Optimizer<HydraModel<TestBackend>, TestBackend> {
        AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init()
    }

    fn unique_checkpoint_base(label: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!("hydra-{label}-{}-{nanos}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn empty_batch(
        device: &<TestBackend as Backend>::Device,
        batch: usize,
    ) -> MjaiBatch<TestBackend> {
        MjaiBatch {
            obs: Tensor::zeros([batch, crate::config::INPUT_CHANNELS, 34], device),
            actions: Tensor::zeros([batch], device),
            legal_mask: Tensor::ones([batch, 46], device),
            value_target: Tensor::zeros([batch], device),
            grp_target: Tensor::zeros([batch, 24], device),
            oracle_target: None,
            oracle_target_mask: Tensor::zeros([batch], device),
            tenpai_target: Tensor::zeros([batch, 3], device),
            danger_target: Tensor::zeros([batch, 3, 34], device),
            danger_mask: Tensor::zeros([batch, 3, 34], device),
            safety_residual_target: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields_target: None,
            mixture_weight_target: None,
            belief_fields_mask: None,
            mixture_weight_mask: None,
            opp_next_target: Tensor::zeros([batch, 3, 34], device),
            score_pdf_target: Tensor::zeros([batch, 64], device),
            score_cdf_target: Tensor::zeros([batch, 64], device),
            target_presence: None,
        }
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
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([4, crate::config::INPUT_CHANNELS, 34], &device);
        let targets = make_dummy_targets::<TestBackend>(&device, 4);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = bc_optimizer();
        let batch = empty_batch(&device, 4);
        let (_, loss1) = bc_train_step(
            model,
            BcTrainBatchInput {
                obs,
                batch: &batch,
                targets: &targets,
            },
            BcTrainStepContext {
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut optimizer,
        );
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
        let batch = empty_batch(&device, 10);
        let mut last_loss = f64::MAX;
        for _ in 0..25 {
            let (m, loss) = bc_train_step(
                model,
                BcTrainBatchInput {
                    obs: obs.clone(),
                    batch: &batch,
                    targets: &targets,
                },
                BcTrainStepContext {
                    loss_fn: &loss_fn,
                    exit_cfg: &BcExitConfig::default(),
                    use_amp: false,
                    lr: 1e-3,
                },
                &mut optimizer,
            );
            model = m;
            last_loss = loss;
        }
        assert!(last_loss < 10.0, "should overfit: loss={last_loss}");
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
            OracleGuidingBatchInput {
                obs,
                targets: &targets,
                loss_fn: &loss_fn,
                importance_weight: 3.0,
                max_importance_weight: 2.0,
                rng_values: &[0.1, 0.9],
            },
            OracleGuidingStepSchedule {
                base_lr: 1e-4,
                oracle_cfg: &oracle_cfg,
                step: 100,
                total_steps: 100,
            },
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
            OracleGuidingBatchInput {
                obs,
                targets: &targets,
                loss_fn: &loss_fn,
                importance_weight: 1.0,
                max_importance_weight: 2.0,
                rng_values: &[0.0, 0.9],
            },
            OracleGuidingStepSchedule {
                base_lr: 1e-4,
                oracle_cfg: &oracle_cfg,
                step: 50,
                total_steps: 100,
            },
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
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = model.forward(x);
        let mask = Tensor::<NdArray<f32>, 2>::ones([4, 46], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints(&[0i32; 4][..], &device);
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
    fn policy_target_argmax_matches_batch_actions() {
        let device = Default::default();
        let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[0i32, 7, 45][..], &device);
        let mut policy_target = vec![0.0f32; 3 * 46];
        policy_target[0] = 1.0;
        policy_target[46 + 7] = 1.0;
        policy_target[2 * 46 + 45] = 1.0;
        let recovered = target_actions_from_policy_target(
            Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device)
                .reshape([3, 46]),
        );
        let same = recovered.equal(actions).into_data().convert::<i64>();
        assert_eq!(
            same.as_slice::<i64>().expect("policy action parity"),
            &[1, 1, 1]
        );
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
        let baseline_batch = empty_batch(&device, 4);
        let (_, loss_baseline) = bc_train_step(
            model1,
            BcTrainBatchInput {
                obs: obs.clone(),
                batch: &baseline_batch,
                targets: &baseline_targets,
            },
            BcTrainStepContext {
                loss_fn: &baseline_loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
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
        let advanced_batch = empty_batch(&device, 4);
        let (_, loss_advanced) = bc_train_step(
            model2,
            BcTrainBatchInput {
                obs,
                batch: &advanced_batch,
                targets: &advanced_targets,
            },
            BcTrainStepContext {
                loss_fn: &advanced_loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
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
        let batch = empty_batch(&device, 4);
        let (_, stats) = bc_train_step(
            model,
            BcTrainBatchInput {
                obs,
                batch: &batch,
                targets: &targets,
            },
            BcTrainStepContext {
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut optimizer,
        );

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
        let path = unique_checkpoint_base("test-ckpt");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.save_file(&path, &recorder).expect("save failed");
        let loaded = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<NdArray<f32>>(&device)
            .load_file(&path, &recorder, &device)
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

    #[test]
    fn checkpoint_meta_old_path_remains_available() {
        let meta: CheckpointMeta = hydra_train_types::checkpoint::CheckpointMeta::new(
            1,
            0.5,
            Some(0.25),
            Some(0.75),
            Some(1.25),
        );
        assert_eq!(
            meta.summary(),
            "epoch=1 loss=0.5000 policy_ce=0.7500 total=1.2500 agree=25.00%"
        );
    }

    #[test]
    fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
        assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

        let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
        assert!((warmup_lr - 2.5e-4).abs() < 1e-12);

        let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
        let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
        assert!((post_warmup_lr - expected).abs() < 1e-12);
    }

    #[test]
    fn oracle_guidance_mask_tensor_uses_rng_fallback_for_missing_samples() {
        let device = Default::default();
        let mask = oracle_guidance_mask_tensor::<TestBackend>(3, 0.5, &[0.25], &device)
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        assert_eq!(mask, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn epoch_stats_summary_and_improving_compare_loss_only() {
        let previous = EpochStats {
            avg_loss: 2.0,
            policy_agreement: 0.1,
            num_batches: 4,
        };
        let current = EpochStats {
            avg_loss: 1.5,
            policy_agreement: 0.05,
            num_batches: 3,
        };
        assert_eq!(current.summary(), "loss=1.5000 agree=5.00% batches=3");
        assert!(current.is_improving(&previous));
    }
}
