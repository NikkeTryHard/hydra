use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::data::sample::{MjaiSample, collate_samples, collate_samples_bc_owned_timed};
use crate::losses::HydraLoss;
use crate::model::{HydraModel, HydraTrainModelExt};
use hydra_model::amp::maybe_autocast;
use hydra_train_algo::bc::{BcExitConfig, maybe_add_exit_loss};
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_LOSS,
};
use hydra_train_types::losses::LossBreakdown;

use crate::bc_metrics::{
    BatchMetricSums, batch_metric_sums_from_outputs, batch_stats_from_metric_sums,
};
use crate::bc_runtime::gated_bc_context;
use hydra_train_runtime::progress::{BatchStats, TrainSubStageTiming};

use std::time::Instant;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

pub struct FixedShapeTrainStepOutput {
    pub grads: GradientsParams,
    pub batch_stats: BatchStats,
    pub sub_stage_timing: TrainSubStageTiming,
}

pub struct FixedShapeBenchmarkStepOutput {
    pub grads: GradientsParams,
    pub batch_stats: Vec<BatchStats>,
    pub sub_stage_timing: TrainSubStageTiming,
}

pub struct FixedShapeTrainConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    pub logical_batch: &'a [MjaiSample],
    pub augment: bool,
    pub microbatch_size: usize,
    pub train_device: &'a LibTorchDevice,
    pub loss_fn: &'a HydraLoss<B>,
    pub bc_exit_cfg: &'a BcExitConfig,
    pub head_controller: &'a mut HeadActivationController,
    pub model: &'a HydraModel<B>,
    pub use_amp: bool,
}

pub struct FixedShapeProbeConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    pub logical_batch: &'a [MjaiSample],
    pub augment: bool,
    pub microbatch_size: usize,
    pub train_device: &'a LibTorchDevice,
    pub loss_fn: &'a HydraLoss<B>,
    pub model: &'a HydraModel<B>,
    pub use_amp: bool,
}

fn accumulate_metric_sums<B: AutodiffBackend<Device = LibTorchDevice>>(
    total_samples: usize,
    microbatch_count: usize,
    metric_sums: BatchMetricSums<B>,
) -> BatchStats
where
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    batch_stats_from_metric_sums(total_samples, microbatch_count, metric_sums)
}

fn split_divisible_prefix(
    logical_batch: &[MjaiSample],
    microbatch_size: usize,
) -> (&[MjaiSample], &[MjaiSample]) {
    let fixed_shape_prefix_len = logical_batch.len() / microbatch_size * microbatch_size;
    logical_batch.split_at(fixed_shape_prefix_len)
}

fn amp_enabled_for_device(use_amp: bool, train_device: &LibTorchDevice) -> bool {
    use_amp && matches!(train_device, LibTorchDevice::Cuda(_))
}

fn merge_metric_sums<B: Backend>(
    metric_sums: &mut Option<BatchMetricSums<B>>,
    chunk_metric_sums: BatchMetricSums<B>,
) {
    *metric_sums = Some(match metric_sums.take() {
        Some(existing) => existing.accumulate(chunk_metric_sums),
        None => chunk_metric_sums,
    });
}

pub fn run_train_logical_batch_fixed_chunks<B>(
    config: FixedShapeTrainConfig<'_, B>,
) -> Result<Option<FixedShapeTrainStepOutput>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut metric_sums: Option<BatchMetricSums<B>> = None;
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut sub_timing = TrainSubStageTiming::default();

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let collated = {
            let _collation_scope = crate::nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned_timed::<B>(chunk, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        let Some(collated) = collated else {
            continue;
        };
        sub_timing.collation_seconds += collated.cpu_prep_seconds;
        sub_timing.h2d_transfer_seconds += collated.device_materialize_seconds;
        sub_timing.h2d_tensor_materialize_seconds += collated.device_materialize_seconds;
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t = Instant::now();
        let output = {
            let _forward_scope = crate::nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
                model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS);
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
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let weighted_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            chunk.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total,
            &breakdown,
        );
        merge_metric_sums(&mut metric_sums, chunk_metric_sums);
        total_samples += chunk.len();
        microbatch_count += 1;
        {
            let t = Instant::now();
            let _backward_scope = crate::nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    if !tail_remainder.is_empty() {
        let collated = {
            let _collation_scope = crate::nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned_timed::<B>(tail_remainder, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        let Some(collated) = collated else {
            return Ok(None);
        };
        sub_timing.collation_seconds += collated.cpu_prep_seconds;
        sub_timing.h2d_transfer_seconds += collated.device_materialize_seconds;
        sub_timing.h2d_tensor_materialize_seconds += collated.device_materialize_seconds;
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t = Instant::now();
        let output = {
            let _forward_scope = crate::nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
                model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = crate::nvtx::scope(PROFILING_STAGE_LOSS);
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
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let weighted_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            tail_remainder.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        );
        merge_metric_sums(&mut metric_sums, chunk_metric_sums);
        total_samples += tail_remainder.len();
        microbatch_count += 1;
        {
            let t = Instant::now();
            let _backward_scope = crate::nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    let Some(metric_sums) = metric_sums else {
        return Ok(None);
    };

    let stats_started = Instant::now();
    let batch_stats = accumulate_metric_sums(total_samples, microbatch_count, metric_sums);
    sub_timing.metric_readback_seconds += stats_started.elapsed().as_secs_f64();

    Ok(Some(FixedShapeTrainStepOutput {
        grads: accumulator.grads(),
        batch_stats,
        sub_stage_timing: sub_timing,
    }))
}

pub fn probe_train_fixed_chunks<B>(
    config: FixedShapeProbeConfig<'_, B>,
) -> Result<Option<GradientsParams>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeProbeConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape probe executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let Some((obs, targets)) = collate_samples::<B>(chunk, augment, train_device)
            .map_err(|err| format!("fixed-shape probe collation failed: {err}"))?
        else {
            continue;
        };
        let output = maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
            model.forward(obs)
        });
        let breakdown = loss_fn.total_loss(&output, &targets);
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let grads = (breakdown.total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    if !tail_remainder.is_empty() {
        let Some((obs, targets)) = collate_samples::<B>(tail_remainder, augment, train_device)
            .map_err(|err| format!("fixed-shape probe collation failed: {err}"))?
        else {
            return Ok(Some(accumulator.grads()));
        };
        let output = maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
            model.forward(obs)
        });
        let breakdown = loss_fn.total_loss(&output, &targets);
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let grads = (breakdown.total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    Ok(Some(accumulator.grads()))
}

pub fn benchmark_train_fixed_chunks<B>(
    config: FixedShapeTrainConfig<'_, B>,
) -> Result<Option<FixedShapeBenchmarkStepOutput>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape benchmark executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut step_batches = Vec::with_capacity(
        fixed_shape_prefix.len() / microbatch_size + usize::from(!tail_remainder.is_empty()),
    );
    let mut sub_timing = TrainSubStageTiming::default();

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let Some(collated) = collate_samples_bc_owned_timed::<B>(chunk, augment, train_device)
            .map_err(|err| format!("benchmark train collation failed: {err}"))?
        else {
            continue;
        };
        sub_timing.collation_seconds += collated.cpu_prep_seconds;
        sub_timing.h2d_transfer_seconds += collated.device_materialize_seconds;
        sub_timing.h2d_tensor_materialize_seconds += collated.device_materialize_seconds;
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t_forward = Instant::now();
        let output = maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
            model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
        });
        sub_timing.forward_seconds += t_forward.elapsed().as_secs_f64();
        let t_loss = Instant::now();
        let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            bc_exit_cfg,
        );
        sub_timing.loss_seconds += t_loss.elapsed().as_secs_f64();
        step_batches.push(BatchStats {
            sample_count: chunk.len(),
            batch_count: 1,
            ..accumulate_metric_sums(
                chunk.len(),
                1,
                batch_metric_sums_from_outputs(
                    chunk.len(),
                    output.policy_logits.clone(),
                    targets.legal_mask.clone(),
                    batch.actions.clone(),
                    total.clone(),
                    &breakdown,
                ),
            )
        });
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let t_backward = Instant::now();
        let grads = (total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
        sub_timing.backward_seconds += t_backward.elapsed().as_secs_f64();
    }

    if !tail_remainder.is_empty() {
        let Some(collated) =
            collate_samples_bc_owned_timed::<B>(tail_remainder, augment, train_device)
                .map_err(|err| format!("benchmark train collation failed: {err}"))?
        else {
            return Ok(Some(FixedShapeBenchmarkStepOutput {
                grads: accumulator.grads(),
                batch_stats: step_batches,
                sub_stage_timing: sub_timing,
            }));
        };
        sub_timing.collation_seconds += collated.cpu_prep_seconds;
        sub_timing.h2d_transfer_seconds += collated.device_materialize_seconds;
        sub_timing.h2d_tensor_materialize_seconds += collated.device_materialize_seconds;
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t_forward = Instant::now();
        let output = maybe_autocast(amp_enabled_for_device(use_amp, train_device), || {
            model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
        });
        sub_timing.forward_seconds += t_forward.elapsed().as_secs_f64();
        let t_loss = Instant::now();
        let breakdown = active_loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            bc_exit_cfg,
        );
        sub_timing.loss_seconds += t_loss.elapsed().as_secs_f64();
        step_batches.push(BatchStats {
            sample_count: tail_remainder.len(),
            batch_count: 1,
            ..accumulate_metric_sums(
                tail_remainder.len(),
                1,
                batch_metric_sums_from_outputs(
                    tail_remainder.len(),
                    output.policy_logits.clone(),
                    targets.legal_mask.clone(),
                    batch.actions.clone(),
                    total.clone(),
                    &breakdown,
                ),
            )
        });
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let t_backward = Instant::now();
        let grads = (total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
        sub_timing.backward_seconds += t_backward.elapsed().as_secs_f64();
    }

    Ok(Some(FixedShapeBenchmarkStepOutput {
        grads: accumulator.grads(),
        batch_stats: step_batches,
        sub_stage_timing: sub_timing,
    }))
}

#[cfg(test)]
mod tests;
