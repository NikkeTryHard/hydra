//! Epoch-runner execution helpers shared by the train binary.
//!
//! This module owns the hot BC logical-batch execution step without depending on
//! the `hydra-train` binary crate. Direct inspection shows the outer epoch loop still has
//! train-bin-owned seams that block a safe full move in this slice.

use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use hydra_bc_shards::BcShardHostBatch;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::NUM_CHANNELS;
use hydra_model::amp::maybe_autocast;
use hydra_train_runtime::bc_fixed_shape::{
    FixedShapeTrainConfig, run_train_logical_batch_fixed_chunks,
};
use hydra_train_runtime::bc_metrics::{
    BatchMetricSums, batch_metric_sums_from_outputs, batch_stats_from_metric_sums,
};
use hydra_train_runtime::bc_runtime::{BcExitConfig, gated_bc_context, maybe_add_exit_loss};
use hydra_train_runtime::data::sample::{MjaiBcBatch, MjaiSample, collate_samples_bc_owned};
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_runtime::losses::HydraLoss;
use hydra_train_runtime::model::{HydraModel, HydraTrainModelExt};
use hydra_train_runtime::nvtx;
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_BC_EPOCH, PROFILING_STAGE_BC_INTERVAL,
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED, PROFILING_STAGE_H2D_STREAM_SYNC,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_LOSS, PROFILING_STAGE_METRIC_READBACK, PROFILING_STAGE_OPTIMIZER_STEP,
    PROFILING_STAGE_PRODUCER_WAIT, PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION,
    ProfilingEnvelope,
};
use hydra_train_runtime::progress::{BatchStats, ScalarAverages, TrainSubStageTiming};
use hydra_train_types::head_gates::{AdvancedHead, TargetPresence};
use hydra_train_types::losses::{HydraTargets, LossBreakdown};

use crate::advisory::IntervalTimingInput;
use crate::resume::EpochContinuation;

/// Minimal config fields needed for session-relative epoch cadence decisions.
pub trait EpochCadenceConfig {
    /// Log interval in optimizer steps.
    fn log_every_n_steps(&self) -> usize;
    /// Validation interval in optimizer steps.
    fn validate_every_n_steps(&self) -> usize;
    /// Checkpoint interval in optimizer steps.
    fn checkpoint_every_n_steps(&self) -> usize;
    /// Optional session-relative optimizer-step budget.
    fn max_train_steps(&self) -> Option<usize>;
}

/// Value object for epoch cadence decisions.
pub struct EpochCadenceInput {
    /// Log interval in optimizer steps.
    pub log_every_n_steps: usize,
    /// Validation interval in optimizer steps.
    pub validate_every_n_steps: usize,
    /// Checkpoint interval in optimizer steps.
    pub checkpoint_every_n_steps: usize,
    /// Optional session-relative optimizer-step budget.
    pub max_train_steps: Option<usize>,
}

impl EpochCadenceConfig for EpochCadenceInput {
    fn log_every_n_steps(&self) -> usize {
        self.log_every_n_steps
    }

    fn validate_every_n_steps(&self) -> usize {
        self.validate_every_n_steps
    }

    fn checkpoint_every_n_steps(&self) -> usize {
        self.checkpoint_every_n_steps
    }

    fn max_train_steps(&self) -> Option<usize> {
        self.max_train_steps
    }
}

impl From<&hydra_train_runtime::config::TrainConfig> for EpochCadenceInput {
    fn from(config: &hydra_train_runtime::config::TrainConfig) -> Self {
        Self {
            log_every_n_steps: config.log_every_n_steps,
            validate_every_n_steps: config.validate_every_n_steps,
            checkpoint_every_n_steps: config.checkpoint_every_n_steps,
            max_train_steps: config.max_train_steps,
        }
    }
}

/// Per-logical-batch BC execution config.
pub struct TrainLogicalBatchConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    /// Number of samples per microbatch.
    pub microbatch_size: usize,
    /// Whether data augmentation is enabled.
    pub augment: bool,
    /// Device receiving collated tensors.
    pub train_device: &'a LibTorchDevice,
    /// Active BC loss function.
    pub loss_fn: &'a HydraLoss<B>,
    /// Optional ExIt loss config.
    pub bc_exit_cfg: &'a BcExitConfig,
    /// Effective optimizer learning rate for this step.
    pub lr: f64,
    /// Whether bf16 autocast is enabled.
    pub use_amp: bool,
}

/// Returns true when a periodic checkpoint should be saved at `global_step`.
#[must_use]
pub fn should_save_periodic_checkpoint(
    config: &impl EpochCadenceConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    let interval = config.checkpoint_every_n_steps();
    interval > 0
        && hydra_train_runtime::status::session_steps_completed(
            global_step,
            session_start_global_step,
        ) > 0
        && hydra_train_runtime::status::session_steps_completed(
            global_step,
            session_start_global_step,
        )
        .is_multiple_of(interval)
}

/// Returns true when the train progress message should be refreshed.
#[must_use]
pub fn should_refresh_train_progress_message(
    config: &impl EpochCadenceConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    let session_step = hydra_train_runtime::status::session_steps_completed(
        global_step,
        session_start_global_step,
    );
    if session_step == 0 {
        return false;
    }
    session_step == 1
        || config.log_every_n_steps() > 0 && session_step.is_multiple_of(config.log_every_n_steps())
        || config.validate_every_n_steps() > 0
            && session_step.is_multiple_of(config.validate_every_n_steps())
        || should_save_periodic_checkpoint(config, global_step, session_start_global_step)
        || hydra_train_runtime::status::reached_session_step_budget(
            global_step,
            session_start_global_step,
            config.max_train_steps(),
        )
}

/// Returns true when epoch-end validation is due.
#[must_use]
pub fn should_run_epoch_end_validation(
    epoch: usize,
    num_epochs: usize,
    every_n_epochs: usize,
) -> bool {
    (epoch + 1).is_multiple_of(every_n_epochs) || epoch + 1 == num_epochs
}

/// Builds the persisted resume continuation for a completed or paused epoch.
#[must_use]
pub fn build_epoch_continuation(
    epoch: usize,
    epoch_completed: bool,
    epoch_optimizer_steps: usize,
) -> EpochContinuation {
    EpochContinuation {
        next_epoch: if epoch_completed { epoch + 1 } else { epoch },
        skip_optimizer_steps_in_epoch: if epoch_completed {
            0
        } else {
            epoch_optimizer_steps
        },
        epoch_completed,
    }
}

/// Merges a profiling tree into an optional accumulator.
pub fn merge_optional_profiling(
    target: &mut Option<ProfilingEnvelope>,
    source: Option<&ProfilingEnvelope>,
) {
    let Some(source) = source.cloned() else {
        return;
    };
    if let Some(target) = target.as_mut() {
        target.merge_assign(&source);
    } else {
        *target = Some(source);
    }
}

/// Builds an interval-level profiling tree.
#[must_use]
pub fn bc_interval_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
) -> ProfilingEnvelope {
    ProfilingEnvelope::from_children(
        PROFILING_STAGE_BC_INTERVAL,
        vec![
            ProfilingEnvelope::nested(
                PROFILING_STAGE_TRAIN,
                train_seconds,
                sub_timing.to_profiling_children(),
            ),
            validation.unwrap_or_else(|| ProfilingEnvelope::leaf(PROFILING_STAGE_VALIDATION, 0.0)),
            ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, checkpoint_seconds),
        ],
    )
}

/// Builds an epoch-level profiling tree.
#[must_use]
pub fn bc_epoch_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
    logging_seconds: f64,
) -> ProfilingEnvelope {
    ProfilingEnvelope::from_children(
        PROFILING_STAGE_BC_EPOCH,
        vec![
            ProfilingEnvelope::nested(
                PROFILING_STAGE_TRAIN,
                train_seconds,
                sub_timing.to_profiling_children(),
            ),
            validation.unwrap_or_else(|| ProfilingEnvelope::leaf(PROFILING_STAGE_VALIDATION, 0.0)),
            ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, checkpoint_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_LOGGING, logging_seconds),
        ],
    )
}

/// Returns elapsed seconds for a direct profiling child stage.
#[must_use]
pub fn child_elapsed_seconds(profiling: &ProfilingEnvelope, stage: &str) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == stage)
        .map(|child| child.elapsed_seconds)
        .unwrap_or(0.0)
}

/// Returns elapsed seconds for a child under the train profiling node.
#[must_use]
pub fn train_child_elapsed_seconds(profiling: &ProfilingEnvelope, stage: &str) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_TRAIN)
        .map(|train| child_elapsed_seconds(train, stage))
        .unwrap_or(0.0)
}

/// Returns elapsed seconds for a nested child below a train sub-stage.
#[must_use]
pub fn train_nested_child_elapsed_seconds(
    profiling: &ProfilingEnvelope,
    parent_stage: &str,
    child_stage: &str,
) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_TRAIN)
        .and_then(|train| {
            train
                .children
                .iter()
                .find(|child| child.stage == parent_stage)
        })
        .map(|parent| child_elapsed_seconds(parent, child_stage))
        .unwrap_or(0.0)
}

/// Converts interval profiling into advisory timing input.
#[must_use]
pub fn interval_timing_input(
    device: &str,
    kernel_launch_count: Option<u64>,
    tiny_kernel_fraction: Option<f64>,
    cuda_runtime_launch_seconds: Option<f64>,
    profiling: &ProfilingEnvelope,
    window_steps: usize,
) -> IntervalTimingInput {
    let device = device.trim().to_ascii_lowercase();
    IntervalTimingInput {
        producer_wait_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_PRODUCER_WAIT,
        ),
        collation_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_COLLATION),
        h2d_transfer_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_H2D_TRANSFER),
        h2d_pageable_to_pinned_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED,
        ),
        h2d_tensor_materialize_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_TENSOR_MATERIALIZE,
        ),
        h2d_stream_sync_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_STREAM_SYNC,
        ),
        metric_readback_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_METRIC_READBACK,
        ),
        forward_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_FORWARD),
        backward_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_BACKWARD),
        optimizer_step_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_OPTIMIZER_STEP,
        ),
        kernel_launch_count,
        tiny_kernel_fraction,
        cuda_runtime_launch_seconds,
        validation_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_VALIDATION),
        checkpoint_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_CHECKPOINT),
        logging_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_LOGGING),
        total_seconds: profiling.elapsed_seconds,
        steps: window_steps,
        is_cuda: device == "cuda" || device.starts_with("cuda:"),
    }
}

/// Device-resident BC shard batch materialized from a backend-agnostic host batch.
pub struct BcShardDeviceBatch<B: Backend> {
    /// Observation tensor, shape `[batch, 192, 34]`.
    pub obs: Tensor<B, 3>,
    /// Minimal BC batch targets consumed by ExIt-aware loss code.
    pub batch: MjaiBcBatch<B>,
    /// Full Hydra loss targets.
    pub targets: HydraTargets<B>,
}

/// Materializes a backend-agnostic BC shard host batch onto a Burn device.
#[must_use]
pub fn materialize_host_batch_owned<B: Backend>(
    host: BcShardHostBatch,
    device: &B::Device,
) -> BcShardDeviceBatch<B> {
    let target_presence = target_presence_from_host_batch(&host, host.batch_size);
    materialize_host_parts_owned::<B>(
        host.batch_size,
        host.obs_flat,
        host.actions,
        host.legal_mask_flat,
        host.value_target,
        host.grp_target_flat,
        host.oracle_target_flat,
        host.oracle_target_mask,
        host.tenpai_flat,
        host.danger_flat,
        host.danger_mask_flat,
        host.opp_next_flat,
        host.score_pdf_flat,
        host.score_cdf_flat,
        host.safety_target_flat,
        host.safety_mask_flat,
        host.exit_target_flat,
        host.exit_mask_flat,
        host.delta_q_target_flat,
        host.delta_q_mask_flat,
        target_presence,
        device,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "materializes a flat shard record without regrouping hot-path owned buffers"
)]
fn materialize_host_parts_owned<B: Backend>(
    batch_size: usize,
    obs_flat: Vec<f32>,
    actions: Vec<i64>,
    legal_mask_flat: Vec<f32>,
    value_target: Vec<f32>,
    grp_target_flat: Vec<f32>,
    oracle_target_flat: Vec<f32>,
    oracle_target_mask: Vec<f32>,
    tenpai_flat: Vec<f32>,
    danger_flat: Vec<f32>,
    danger_mask_flat: Vec<f32>,
    opp_next_flat: Vec<f32>,
    score_pdf_flat: Vec<f32>,
    score_cdf_flat: Vec<f32>,
    safety_target_flat: Option<Vec<f32>>,
    safety_mask_flat: Option<Vec<f32>>,
    exit_target_flat: Option<Vec<f32>>,
    exit_mask_flat: Option<Vec<f32>>,
    delta_q_target_flat: Option<Vec<f32>>,
    delta_q_mask_flat: Option<Vec<f32>>,
    target_presence: TargetPresence,
    device: &B::Device,
) -> BcShardDeviceBatch<B> {
    let b = batch_size;

    let obs = Tensor::<B, 3>::from_data(TensorData::new(obs_flat, [b, NUM_CHANNELS, 34]), device);
    let policy_target = policy_target_from_action_slice::<B>(actions.as_slice(), b, device);
    let actions_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(actions, [b]), device);
    let legal_mask = Tensor::<B, 2>::from_data(
        TensorData::new(legal_mask_flat, [b, HYDRA_ACTION_SPACE]),
        device,
    );
    let value_target = Tensor::<B, 1>::from_data(TensorData::new(value_target, [b]), device);
    let grp_target = Tensor::<B, 2>::from_data(TensorData::new(grp_target_flat, [b, 24]), device);
    let oracle_target =
        Tensor::<B, 2>::from_data(TensorData::new(oracle_target_flat, [b, 4]), device);
    let oracle_target_mask =
        Tensor::<B, 1>::from_data(TensorData::new(oracle_target_mask, [b]), device);
    let tenpai_target = Tensor::<B, 2>::from_data(TensorData::new(tenpai_flat, [b, 3]), device);
    let danger_target = Tensor::<B, 3>::from_data(TensorData::new(danger_flat, [b, 3, 34]), device);
    let danger_mask =
        Tensor::<B, 3>::from_data(TensorData::new(danger_mask_flat, [b, 3, 34]), device);
    let opp_next_target =
        Tensor::<B, 3>::from_data(TensorData::new(opp_next_flat, [b, 3, 34]), device);
    let score_pdf_target =
        Tensor::<B, 2>::from_data(TensorData::new(score_pdf_flat, [b, 64]), device);
    let score_cdf_target =
        Tensor::<B, 2>::from_data(TensorData::new(score_cdf_flat, [b, 64]), device);

    let exit_target_tensor = exit_target_flat.map(|buf| {
        Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
    });
    let exit_mask_tensor = exit_mask_flat.map(|buf| {
        Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
    });

    let batch = MjaiBcBatch {
        actions: actions_tensor,
        exit_target: exit_target_tensor,
        exit_mask: exit_mask_tensor,
    };

    let targets = HydraTargets {
        policy_target,
        legal_mask,
        value_target,
        grp_target,
        tenpai_target,
        danger_target,
        danger_mask,
        opp_next_target,
        score_pdf_target,
        score_cdf_target,
        oracle_target: Some(oracle_target),
        belief_fields_target: None,
        belief_fields_mask: None,
        mixture_weight_target: None,
        mixture_weight_mask: None,
        opponent_hand_type_target: None,
        delta_q_target: delta_q_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        delta_q_mask: delta_q_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        safety_residual_target: safety_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        safety_residual_mask: safety_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        oracle_guidance_mask: Some(oracle_target_mask),
        target_presence: Some(target_presence),
    };

    BcShardDeviceBatch {
        obs,
        batch,
        targets,
    }
}

fn policy_target_from_action_slice<B: Backend>(
    actions: &[i64],
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut flat = vec![0.0f32; batch_size * HYDRA_ACTION_SPACE];
    for (row, &action) in actions.iter().take(batch_size).enumerate() {
        if action >= 0 {
            let idx = action as usize;
            if idx < HYDRA_ACTION_SPACE {
                flat[row * HYDRA_ACTION_SPACE + idx] = 1.0;
            }
        }
    }
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [batch_size, HYDRA_ACTION_SPACE]),
        device,
    )
}

fn target_presence_from_host_batch(host: &BcShardHostBatch, batch_size: usize) -> TargetPresence {
    let mut presence = TargetPresence::with_batch_size(batch_size);
    presence.counts[AdvancedHead::OracleCritic.index()] = host
        .oracle_target_mask
        .iter()
        .take(batch_size)
        .filter(|&&value| value > 0.0)
        .count();
    if let (Some(_target), Some(mask)) = (&host.safety_target_flat, &host.safety_mask_flat) {
        presence.counts[AdvancedHead::SafetyResidual.index()] =
            count_nonzero_action_rows(mask, batch_size, HYDRA_ACTION_SPACE);
    }
    if let (Some(_target), Some(mask)) = (&host.delta_q_target_flat, &host.delta_q_mask_flat) {
        let (rows, actions) =
            count_nonzero_action_rows_and_entries(mask, batch_size, HYDRA_ACTION_SPACE);
        presence.counts[AdvancedHead::DeltaQ.index()] = rows;
        presence.delta_q_actions_present = actions;
    }
    presence
}

fn count_nonzero_action_rows(mask: &[f32], batch_size: usize, action_space: usize) -> usize {
    mask.chunks_exact(action_space)
        .take(batch_size)
        .filter(|row| row.iter().any(|&value| value > 0.0))
        .count()
}

fn count_nonzero_action_rows_and_entries(
    mask: &[f32],
    batch_size: usize,
    action_space: usize,
) -> (usize, usize) {
    let mut rows = 0usize;
    let mut entries = 0usize;
    for row in mask.chunks_exact(action_space).take(batch_size) {
        let row_entries = row.iter().filter(|&&value| value > 0.0).count();
        if row_entries > 0 {
            rows += 1;
            entries += row_entries;
        }
    }
    (rows, entries)
}

/// Runs forward/backward/optimizer for one raw replay logical batch.
pub fn train_logical_batch<B, O>(
    logical_batch: &[MjaiSample],
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    <B as AutodiffBackend>::InnerBackend: Backend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
{
    let TrainLogicalBatchConfig {
        microbatch_size,
        use_amp,
        augment,
        train_device,
        loss_fn,
        bc_exit_cfg,
        lr,
    } = config;
    if logical_batch.is_empty() {
        return Ok((Vec::new(), TrainSubStageTiming::default()));
    }

    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let logical_batch_len = logical_batch.len().max(1) as f32;
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut metric_sums: Option<BatchMetricSums<B>> = None;

    if let Some(fixed_shape) = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size: microbatch_size.max(1),
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model: epoch_model(model_slot)?,
        use_amp,
    })? {
        let optimizer_started = Instant::now();
        let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
        let model = model_slot
            .take()
            .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
        *model_slot = Some(optimizer.step(lr, model, fixed_shape.grads));
        head_controller.tick_warmup();
        let mut sub_timing = fixed_shape.sub_stage_timing;
        sub_timing.optimizer_step_seconds += optimizer_started.elapsed().as_secs_f64();
        return Ok((vec![fixed_shape.batch_stats], sub_timing));
    }

    let mut sub_timing_fallback = TrainSubStageTiming::default();

    for chunk in logical_batch.chunks(microbatch_size.max(1)) {
        let t = Instant::now();
        let collated = {
            let _collation_scope = nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned::<B>(chunk, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        sub_timing_fallback.collation_seconds += t.elapsed().as_secs_f64();
        let Some((obs, batch, targets)) = collated else {
            continue;
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let model = epoch_model(model_slot)?;
        let t = Instant::now();
        let output = {
            let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing_fallback.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing_fallback.loss_seconds += t.elapsed().as_secs_f64();

        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let weighted_chunk_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            chunk.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total,
            &breakdown,
        );

        metric_sums = Some(match metric_sums.take() {
            Some(existing) => existing.accumulate(chunk_metric_sums),
            None => chunk_metric_sums,
        });
        total_samples += chunk.len();
        microbatch_count += 1;

        {
            let t = Instant::now();
            let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_chunk_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing_fallback.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    let batch_stats = if let Some(metric_sums) = metric_sums {
        let metric_started = Instant::now();
        let stats = vec![batch_stats_from_metric_sums(
            total_samples,
            microbatch_count,
            metric_sums,
        )];
        sub_timing_fallback.metric_readback_seconds += metric_started.elapsed().as_secs_f64();
        stats
    } else {
        Vec::new()
    };

    if !batch_stats.is_empty() {
        let t = Instant::now();
        let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
        let grads = accumulator.grads();
        let model = model_slot
            .take()
            .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
        *model_slot = Some(optimizer.step(lr, model, grads));
        head_controller.tick_warmup();
        sub_timing_fallback.optimizer_step_seconds += t.elapsed().as_secs_f64();
    }

    Ok((batch_stats, sub_timing_fallback))
}

fn epoch_model<B>(model_slot: &Option<HydraModel<B>>) -> Result<&HydraModel<B>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    model_slot
        .as_ref()
        .ok_or_else(|| "epoch runner model slot should stay populated".to_string())
}

/// Records drained batch stats into epoch and step-window accumulators.
pub fn record_drained_batch_stats(
    drained: Vec<BatchStats>,
    stats: &mut ScalarAverages,
    step_window: &mut ScalarAverages,
) {
    for batch_stats in drained {
        stats.record_batch(batch_stats);
        step_window.record_batch(batch_stats);
    }
}
