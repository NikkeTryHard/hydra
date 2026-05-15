//! Behavioral-cloning execution adapters.

use crate::data::sample::{MjaiBcBatch, MjaiSample, collate_sample_refs_bc_owned};
use crate::losses::HydraLoss;
use crate::model::HydraModel;
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use hydra_model::amp::maybe_autocast;
use hydra_model::model::HydraOutput;
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_types::config::OracleGuidingConfig;
use hydra_train_types::head_gates::{AdvancedHead, borrow_or_extract_target_presence};
use hydra_train_types::losses::{HydraTargets, LossBreakdown};

pub use hydra_train_algo::bc::{
    BcExitConfig, cosine_annealing_lr, maybe_add_exit_loss, oracle_guidance_mask_tensor,
    oracle_guidance_mask_values, policy_agreement, policy_agreement_counts,
    target_actions_from_policy_target, warmup_then_cosine_lr,
};

/// Batch view that can expose optional ExIt sidecar targets.
pub trait ExitTargetBatch<B: Backend> {
    /// Optional ExIt policy target.
    fn exit_target(&self) -> Option<&Tensor<B, 2>>;
    /// Optional ExIt mask.
    fn exit_mask(&self) -> Option<&Tensor<B, 2>>;
}

impl<B: Backend> ExitTargetBatch<B> for MjaiBcBatch<B> {
    fn exit_target(&self) -> Option<&Tensor<B, 2>> {
        self.exit_target.as_ref()
    }

    fn exit_mask(&self) -> Option<&Tensor<B, 2>> {
        self.exit_mask.as_ref()
    }
}

impl<B: Backend> ExitTargetBatch<B> for crate::data::sample::MjaiBatch<B> {
    fn exit_target(&self) -> Option<&Tensor<B, 2>> {
        self.exit_target.as_ref()
    }

    fn exit_mask(&self) -> Option<&Tensor<B, 2>> {
        self.exit_mask.as_ref()
    }
}

/// Adds optional ExIt loss to a BC breakdown total.
pub fn bc_total_with_exit_from_breakdown<B: Backend>(
    output: &HydraOutput<B>,
    batch: &impl ExitTargetBatch<B>,
    breakdown: &LossBreakdown<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    maybe_add_exit_loss(
        breakdown.total.clone(),
        output.policy_logits.clone(),
        batch.exit_target(),
        batch.exit_mask(),
        exit_cfg,
    )
}

/// Returns the default phase learning rate schedule.
pub fn phase_learning_rate(
    phase: hydra_train_types::phase::TrainingPhase,
    step: usize,
    total_steps: usize,
) -> f64 {
    use hydra_train_types::phase::TrainingPhase;
    let (lr_max, lr_min) = match phase {
        TrainingPhase::BcWarmStart => (2.5e-4, 1e-6),
        TrainingPhase::OracleGuiding => (1e-4, 1e-6),
        TrainingPhase::DrdaAchSelfPlay => (2.5e-4, 2.5e-5),
        TrainingPhase::ExitPondering => (1e-4, 1e-5),
        TrainingPhase::BenchmarkGates => (2.5e-4, 2.5e-4),
    };
    cosine_annealing_lr(step, total_steps, lr_max, lr_min)
}

/// Behavioral-cloning epoch aggregate metrics.
pub struct EpochStats {
    /// Mean optimized loss across optimizer steps.
    pub avg_loss: f64,
    /// Mean policy agreement across optimizer steps.
    pub policy_agreement: f64,
    /// Number of optimizer steps in the epoch.
    pub num_batches: usize,
}

impl EpochStats {
    /// Human-readable compact metric summary.
    pub fn summary(&self) -> String {
        format!(
            "loss={:.4} agree={:.2}% batches={}",
            self.avg_loss,
            self.policy_agreement * 100.0,
            self.num_batches
        )
    }

    /// Returns true when loss improved relative to `previous`.
    pub fn is_improving(&self, previous: &EpochStats) -> bool {
        self.avg_loss < previous.avg_loss
    }
}

/// Per-step oracle-guiding outcome and effective schedule values.
pub struct OracleGuidingStepStats {
    /// Whether the batch was skipped due to importance-weight rejection.
    pub skipped: bool,
    /// Learning rate used for the step, after oracle schedule decay.
    pub effective_lr: f64,
    /// Probability of retaining oracle-guided targets for this step.
    pub oracle_keep_prob: f32,
    /// Fraction retained in this batch's sampled oracle mask.
    pub kept_oracle_fraction: f32,
    /// Read-back optimized loss, absent when skipped.
    pub loss: Option<f64>,
}

/// Inputs for one BC train step.
pub struct BcTrainBatchInput<'a, B: Backend> {
    /// Observation tensor for the step.
    pub obs: Tensor<B, 3>,
    /// Action/exit BC batch surfaces.
    pub batch: &'a MjaiBcBatch<B>,
    /// Multi-head loss targets.
    pub targets: &'a HydraTargets<B>,
}

/// Immutable train-step execution context.
pub struct BcTrainStepContext<'a, B: AutodiffBackend> {
    /// Loss adapter.
    pub loss_fn: &'a HydraLoss<B>,
    /// Optional ExIt loss config.
    pub exit_cfg: &'a BcExitConfig,
    /// Whether to enable backend autocast for the forward pass.
    pub use_amp: bool,
    /// Optimizer learning rate.
    pub lr: f64,
}

/// Inputs for one oracle-guiding train step.
pub struct OracleGuidingBatchInput<'a, B: Backend> {
    /// Observation tensor for the step.
    pub obs: Tensor<B, 3>,
    /// Multi-head loss targets.
    pub targets: &'a HydraTargets<B>,
    /// Loss adapter.
    pub loss_fn: &'a HydraLoss<B>,
    /// Batch importance weight.
    pub importance_weight: f32,
    /// Rejection threshold for large importance weights.
    pub max_importance_weight: f32,
    /// Deterministic random values used to sample oracle dropout mask.
    pub rng_values: &'a [f32],
}

/// Oracle-guiding step schedule inputs.
pub struct OracleGuidingStepSchedule<'a> {
    /// Base learning rate before oracle schedule decay.
    pub base_lr: f64,
    /// Oracle-guiding scalar schedule config.
    pub oracle_cfg: &'a OracleGuidingConfig,
    /// Current global step.
    pub step: usize,
    /// Total scheduled steps.
    pub total_steps: usize,
}

/// Adds optional ExIt loss to a BC breakdown total when a batch is available.
pub fn bc_total_with_optional_exit_from_breakdown<B: Backend>(
    output: &HydraOutput<B>,
    batch: Option<&impl ExitTargetBatch<B>>,
    breakdown: &LossBreakdown<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    let mut total = breakdown.total.clone();
    if let Some(batch) = batch {
        total = maybe_add_exit_loss(
            total,
            output.policy_logits.clone(),
            batch.exit_target(),
            batch.exit_mask(),
            exit_cfg,
        );
    }
    total
}

/// Computes BC total loss with optional ExIt contribution.
pub fn bc_total_with_exit<B: Backend>(
    output: &HydraOutput<B>,
    batch: &MjaiBcBatch<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    exit_cfg: &BcExitConfig,
) -> Tensor<B, 1> {
    let breakdown = loss_fn.total_loss(output, targets);
    bc_total_with_optional_exit_from_breakdown(output, Some(batch), &breakdown, exit_cfg)
}

/// Executes one behavioral-cloning optimizer step.
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

/// Executes one oracle-guiding optimizer step.
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
        Option::<&MjaiBcBatch<B>>::None,
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

/// Runs one BC epoch with optional gradient accumulation.
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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

/// Builds the effective BC loss and warmup head set for one batch.
pub fn gated_bc_context<B: Backend>(
    controller: Option<&mut HeadActivationController>,
    base_loss_fn: &HydraLoss<B>,
    targets: &HydraTargets<B>,
) -> (HydraLoss<B>, Vec<AdvancedHead>) {
    if let Some(ctrl) = controller {
        if base_loss_fn.config.w_delta_q > 0.0 {
            let presence = borrow_or_extract_target_presence(targets);
            ctrl.record_batch(&presence);
            ctrl.try_activate(AdvancedHead::DeltaQ);
        }
        let effective_cfg = ctrl.approved_loss_config(&base_loss_fn.config);
        let warmup_heads = ctrl.warmup_heads();
        (HydraLoss::<B>::new(effective_cfg), warmup_heads)
    } else {
        (HydraLoss::<B>::new(base_loss_fn.config.clone()), Vec::new())
    }
}
