use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use indicatif::ProgressBar;
use std::sync::mpsc;
use std::time::Instant;

use hydra_train::data::bc_shards::{
    BcShardHostBatch, BcShardReader, BcShardSplit, load_bc_shard_reader,
};
use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_val_microbatches};
use hydra_train::data::sample::{MjaiBcBatch, MjaiSample, collate_samples_bc_owned};
#[cfg(test)]
use hydra_train::model::HydraOutput;
use hydra_train::model::{HydraModel, HydraTrainModelExt};
use hydra_train::preflight::{
    PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS, PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD,
    PROFILING_STAGE_VALIDATION, ProfilingEnvelope,
};
use hydra_train::training::bc::{BcExitConfig, gated_bc_context, maybe_add_exit_loss};
use hydra_train::training::delta_q_promotion::{
    DeltaQPolicyTransferReport, DeltaQPolicyTransferResult, DeltaQPolicyTransferThresholds,
    DeltaQPromotionReport, DeltaQPromotionResult, DeltaQPromotionThresholds,
    collect_policy_transfer_metrics_from_policy_outputs, collect_promotion_metrics_from_outputs,
    evaluate_policy_transfer_report, evaluate_promotion_report,
};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::{HydraLoss, HydraTargets};

use super::config::{
    TrainConfig, shard_prefetch_depth, validation_microbatch_size, validation_sample_limit,
};
use super::nvtx;
#[cfg(feature = "cuda-graph")]
use super::pinned_transfer::{AsyncH2DContext, PinnedStagingArea, PreallocatedDeviceTensors};
use super::progress::{BatchMetricSums, RareActionMetrics};
#[cfg(test)]
use super::progress::{BatchStats, batch_stats_from_outputs};
use super::progress::{batch_metric_sums_from_outputs, batch_stats_from_metric_sums};
use super::resume::BestValidation;
pub(super) use hydra_train_exec::validation::{
    DeltaQPolicyTransferSnapshot, DeltaQPromotionSnapshot, ValidationGateDecision,
    ValidationScalarSummary,
};
type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

pub(super) struct ValidationContext<'a, B: Backend> {
    pub(super) config: &'a TrainConfig,
    pub(super) loader_config: &'a StreamingLoaderConfig,
    pub(super) manifest: &'a DataManifest,
    pub(super) cached_samples: Option<&'a [Box<[MjaiSample]>]>,
    pub(super) shard_reader: Option<&'a BcShardReader>,
    pub(super) device: &'a B::Device,
    pub(super) loss_fn: &'a HydraLoss<B>,
    pub(super) exit_cfg: &'a BcExitConfig,
}

pub(super) type ValidationCachedMicrobatches = Box<[Box<[MjaiSample]>]>;

pub(super) struct ValidationRuntime<'a> {
    pub(super) head_controller: Option<&'a mut HeadActivationController>,
    pub(super) progress: Option<&'a ProgressBar>,
}
#[derive(Clone)]
pub(super) struct ValidationSummary {
    pub(super) total_loss: f64,
    pub(super) policy_loss: f64,
    pub(super) agreement: f64,
    pub(super) samples: usize,
    pub(super) rare_actions: RareActionMetrics,
    pub(super) profiling: Option<ProfilingEnvelope>,
    pub(super) delta_q_promotion: Option<DeltaQPromotionReport>,
    pub(super) delta_q_promotion_result: Option<DeltaQPromotionResult>,
    pub(super) delta_q_promotion_snapshot: Option<DeltaQPromotionSnapshot>,
    pub(super) delta_q_policy_transfer: Option<DeltaQPolicyTransferReport>,
    pub(super) delta_q_policy_transfer_result: Option<DeltaQPolicyTransferResult>,
    pub(super) delta_q_policy_transfer_snapshot: Option<DeltaQPolicyTransferSnapshot>,
    pub(super) saw_exit_targets: bool,
    pub(super) saw_delta_q_targets: bool,
}

impl ValidationSummary {
    fn scalar_summary(&self) -> ValidationScalarSummary {
        ValidationScalarSummary {
            total_loss: self.total_loss,
            policy_loss: self.policy_loss,
            agreement: self.agreement,
            samples: self.samples,
            saw_exit_targets: self.saw_exit_targets,
            saw_delta_q_targets: self.saw_delta_q_targets,
        }
    }
}

fn delta_q_promotion_snapshot_from_report(
    report: &DeltaQPromotionReport,
    result: &DeltaQPromotionResult,
) -> DeltaQPromotionSnapshot {
    DeltaQPromotionSnapshot {
        compared_states: report.compared_states,
        candidate_top1_agreement: report.candidate_top1_agreement(),
        candidate_mean_regret: report.candidate_mean_regret(),
        baseline_mean_regret: report.baseline_mean_regret(),
        mean_decision_lift: report.mean_decision_lift(),
        negative_lift_fraction: report.negative_lift_fraction(),
        regret_beats_baseline_rate: report.candidate_regret_beats_baseline_rate(),
        top1_beats_baseline_rate: report.candidate_top1_beats_baseline_rate(),
        passed: result.passed,
    }
}

fn delta_q_policy_transfer_snapshot_from_report(
    report: &DeltaQPolicyTransferReport,
) -> DeltaQPolicyTransferSnapshot {
    DeltaQPolicyTransferSnapshot {
        compared_states: report.compared_states,
        candidate_policy_top1_to_teacher: report.candidate_policy_top1_to_teacher(),
        baseline_policy_top1_to_teacher: report.baseline_policy_top1_to_teacher(),
        candidate_policy_mean_teacher_regret: report.candidate_policy_mean_teacher_regret(),
        baseline_policy_mean_teacher_regret: report.baseline_policy_mean_teacher_regret(),
        candidate_beats_baseline_rate: report.candidate_beats_baseline_rate(),
        negative_transfer_fraction: report.negative_transfer_fraction(),
    }
}

pub(super) fn evaluate_validation_gates(
    config: &TrainConfig,
    summary: &ValidationSummary,
    best: Option<BestValidation>,
) -> ValidationGateDecision {
    hydra_train_exec::validation::evaluate_validation_gates(
        &config.validation_gates,
        config.advanced_loss.as_ref(),
        &summary.scalar_summary(),
        best,
    )
}

struct ValidationAccumulator<B: Backend> {
    metric_sums: Option<BatchMetricSums<B>>,
    microbatch_count: usize,
    total_samples: usize,
    delta_q_promotion: DeltaQPromotionReport,
    delta_q_policy_transfer: DeltaQPolicyTransferReport,
    saw_delta_q_targets: bool,
    saw_exit_targets: bool,
    profiling: ProfilingEnvelope,
}

impl<B: Backend> ValidationAccumulator<B> {
    fn new() -> Self {
        Self {
            metric_sums: None,
            microbatch_count: 0,
            total_samples: 0,
            delta_q_promotion: DeltaQPromotionReport::new(),
            delta_q_policy_transfer: DeltaQPolicyTransferReport::new(),
            saw_delta_q_targets: false,
            saw_exit_targets: false,
            profiling: ProfilingEnvelope::nested(
                PROFILING_STAGE_VALIDATION,
                0.0,
                vec![
                    ProfilingEnvelope::leaf(PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS, 0.0),
                    ProfilingEnvelope::leaf(PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD, 0.0),
                ],
            ),
        }
    }
}

struct ValidationModels<'a, TB>
where
    TB: AutodiffBackend,
{
    model: &'a HydraModel<TB>,
    baseline_model: &'a HydraModel<TB>,
    model_valid: &'a HydraModel<ValidBackendOf<TB>>,
    baseline_valid: &'a HydraModel<ValidBackendOf<TB>>,
}

struct ValidationBatchInput<B: Backend> {
    obs: Tensor<B, 3>,
    batch: MjaiBcBatch<B>,
    targets: HydraTargets<B>,
    sample_count: usize,
}

fn process_validation_batch<TB>(
    models: ValidationModels<'_, TB>,
    loss_fn: &HydraLoss<ValidBackendOf<TB>>,
    exit_cfg: &BcExitConfig,
    batch_input: ValidationBatchInput<ValidBackendOf<TB>>,
    head_controller: &mut Option<&mut HeadActivationController>,
    accumulator: &mut ValidationAccumulator<ValidBackendOf<TB>>,
) where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let ValidationModels {
        model,
        baseline_model,
        model_valid,
        baseline_valid,
    } = models;
    let ValidationBatchInput {
        obs,
        batch,
        targets,
        sample_count,
    } = batch_input;
    let (active_loss_fn, warmup_heads) =
        gated_bc_context(head_controller.as_deref_mut(), loss_fn, &targets);
    let candidate_started = Instant::now();
    let (output, breakdown, total) = {
        let _candidate_scope = nvtx::scope(PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS);
        let output = model_valid.forward_with_warmup_train(
            obs.clone(),
            &active_loss_fn.config,
            &warmup_heads,
        );
        let breakdown = active_loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            exit_cfg,
        );
        (output, breakdown, total)
    };
    let candidate_elapsed_seconds = candidate_started.elapsed().as_secs_f64();
    let chunk_metric_sums = batch_metric_sums_from_outputs(
        sample_count,
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        batch.actions.clone(),
        total.clone(),
        &breakdown,
    );
    let mut baseline_elapsed_seconds = 0.0;
    if batch.exit_target.is_some() && batch.exit_mask.is_some() {
        accumulator.saw_exit_targets = true;
    }
    if targets.delta_q_target.is_some() && targets.delta_q_mask.is_some() {
        let baseline_policy_logits = if std::ptr::eq(model, baseline_model) {
            output.policy_logits.clone()
        } else {
            let baseline_started = Instant::now();
            let baseline_policy_logits = {
                let _baseline_scope = nvtx::scope(PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD);
                baseline_valid.forward_policy(obs)
            };
            baseline_elapsed_seconds = baseline_started.elapsed().as_secs_f64();
            baseline_policy_logits
        };
        accumulator
            .delta_q_promotion
            .merge(&collect_promotion_metrics_from_outputs(
                &output, &targets, 0.75,
            ));
        accumulator.delta_q_policy_transfer.merge(
            &collect_policy_transfer_metrics_from_policy_outputs(
                output.policy_logits.clone(),
                baseline_policy_logits,
                &targets,
            ),
        );
        accumulator.saw_delta_q_targets = true;
    }
    accumulator
        .profiling
        .merge_assign(&ProfilingEnvelope::nested(
            PROFILING_STAGE_VALIDATION,
            candidate_elapsed_seconds + baseline_elapsed_seconds,
            vec![
                ProfilingEnvelope::leaf(
                    PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS,
                    candidate_elapsed_seconds,
                ),
                ProfilingEnvelope::leaf(
                    PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD,
                    baseline_elapsed_seconds,
                ),
            ],
        ));
    accumulator.metric_sums = Some(match accumulator.metric_sums.take() {
        Some(existing) => existing.accumulate(chunk_metric_sums),
        None => chunk_metric_sums,
    });
    accumulator.microbatch_count += 1;
    accumulator.total_samples += sample_count;
}

fn finalize_validation_summary<B: Backend>(
    accumulator: ValidationAccumulator<B>,
) -> ValidationSummary {
    let ValidationAccumulator {
        metric_sums,
        microbatch_count,
        total_samples,
        delta_q_promotion,
        delta_q_policy_transfer,
        saw_delta_q_targets,
        saw_exit_targets,
        profiling,
    } = accumulator;

    if total_samples == 0 {
        return ValidationSummary {
            total_loss: 0.0,
            policy_loss: 0.0,
            agreement: 0.0,
            samples: 0,
            profiling: Some(profiling),
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
            rare_actions: RareActionMetrics::default(),
            saw_exit_targets: false,
            saw_delta_q_targets: false,
        };
    }

    let stats = batch_stats_from_metric_sums(
        total_samples,
        microbatch_count,
        metric_sums.expect("validation metric sums should exist when samples > 0"),
    );
    let (
        delta_q_promotion,
        delta_q_promotion_result,
        delta_q_promotion_snapshot,
        delta_q_policy_transfer,
        delta_q_policy_transfer_result,
        delta_q_policy_transfer_snapshot,
    ) = if saw_delta_q_targets {
        let result =
            evaluate_promotion_report(&delta_q_promotion, &DeltaQPromotionThresholds::default());
        let policy_transfer_result = evaluate_policy_transfer_report(
            &delta_q_policy_transfer,
            &DeltaQPolicyTransferThresholds::default(),
        );
        let snapshot = delta_q_promotion_snapshot_from_report(&delta_q_promotion, &result);
        let policy_transfer_snapshot =
            delta_q_policy_transfer_snapshot_from_report(&delta_q_policy_transfer);
        (
            Some(delta_q_promotion),
            Some(result),
            Some(snapshot),
            Some(delta_q_policy_transfer),
            Some(policy_transfer_result),
            Some(policy_transfer_snapshot),
        )
    } else {
        (None, None, None, None, None, None)
    };

    ValidationSummary {
        total_loss: stats.total_loss,
        policy_loss: stats.loss_policy,
        agreement: stats.policy_agreement,
        samples: total_samples,
        rare_actions: stats.rare_actions,
        saw_exit_targets,
        saw_delta_q_targets,
        profiling: Some(profiling),
        delta_q_promotion,
        delta_q_promotion_result,
        delta_q_promotion_snapshot,
        delta_q_policy_transfer,
        delta_q_policy_transfer_result,
        delta_q_policy_transfer_snapshot,
    }
}

#[cfg(test)]
pub(super) fn validation_batch_stats<B: Backend>(
    sample_count: usize,
    output: &HydraOutput<B>,
    batch: &hydra_train::data::sample::MjaiBatch<B>,
    targets: &HydraTargets<B>,
    breakdown: &hydra_train::training::losses::LossBreakdown<B>,
    total_loss: &Tensor<B, 1>,
) -> BatchStats {
    batch_stats_from_outputs(
        sample_count,
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        batch.actions.clone(),
        total_loss.clone(),
        breakdown,
    )
}

pub(super) fn is_better_validation(
    summary: &ValidationSummary,
    best: Option<BestValidation>,
) -> bool {
    hydra_train_exec::validation::is_better_validation(&summary.scalar_summary(), best)
}

pub(super) fn run_validation<TB>(
    model: &HydraModel<TB>,
    context: ValidationContext<'_, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    run_validation_with_policy_baseline(model, model, context, runtime)
}

pub(super) fn run_validation_with_policy_baseline<TB>(
    model: &HydraModel<TB>,
    baseline_model: &HydraModel<TB>,
    context: ValidationContext<'_, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    if let Some(reader) = context.shard_reader {
        return run_validation_from_shards(model, baseline_model, context, runtime, reader);
    }
    if let Some(manifest_path) = context.config.bc_shards_manifest_path.as_ref() {
        let reader = load_bc_shard_reader(manifest_path, BcShardSplit::Validation)?;
        return run_validation_from_shards(model, baseline_model, context, runtime, &reader);
    }
    let ValidationContext {
        config,
        loader_config,
        manifest,
        cached_samples,
        shard_reader: _,
        device,
        loss_fn,
        exit_cfg,
    } = context;
    let ValidationRuntime {
        head_controller,
        progress,
    } = runtime;
    let model_valid = model.valid();
    let baseline_valid = baseline_model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut accumulator = ValidationAccumulator::new();
    let mut head_controller = head_controller;

    let run_chunk = |capped_chunk: &[MjaiSample],
                     head_controller: &mut Option<&mut HeadActivationController>,
                     accumulator: &mut ValidationAccumulator<ValidBackendOf<TB>>|
     -> Result<(), String> {
        let Some((obs, batch, targets)) =
            collate_samples_bc_owned::<ValidBackendOf<TB>>(capped_chunk, false, device)
                .map_err(|err| format!("validation collation failed: {err}"))?
        else {
            return Ok(());
        };
        process_validation_batch(
            ValidationModels {
                model,
                baseline_model,
                model_valid: &model_valid,
                baseline_valid: &baseline_valid,
            },
            loss_fn,
            exit_cfg,
            ValidationBatchInput {
                obs,
                batch,
                targets,
                sample_count: capped_chunk.len(),
            },
            head_controller,
            accumulator,
        );
        Ok(())
    };

    if let Some(cached_samples) = cached_samples {
        for chunk in cached_samples {
            if let Some(limit) = validation_sample_limit
                && accumulator.total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(accumulator.total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk.as_ref()
            };
            if capped_chunk.is_empty() {
                break;
            }
            run_chunk(capped_chunk, &mut head_controller, &mut accumulator)?;
        }
    } else {
        for microbatch_result in
            stream_val_microbatches(manifest, loader_config, validation_batch_size, progress)
        {
            let microbatch =
                microbatch_result.map_err(|err| format!("validation stream failed: {err}"))?;
            if let Some(limit) = validation_sample_limit
                && accumulator.total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(accumulator.total_samples);
                &microbatch[..microbatch.len().min(remaining)]
            } else {
                microbatch.as_slice()
            };
            if capped_chunk.is_empty() {
                break;
            }
            run_chunk(capped_chunk, &mut head_controller, &mut accumulator)?;
            if let Some(limit) = validation_sample_limit
                && accumulator.total_samples >= limit
            {
                break;
            }
        }
    }

    Ok(finalize_validation_summary(accumulator))
}

pub(super) fn materialize_validation_samples(
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
) -> Result<Option<ValidationCachedMicrobatches>, String> {
    if config.bc_shards_manifest_path.is_some() {
        return Ok(None);
    }
    let Some(limit) = validation_sample_limit(config) else {
        return Ok(None);
    };
    let microbatch_size = validation_microbatch_size(config);
    let mut microbatches = Vec::new();
    let mut total_samples = 0usize;
    for microbatch_result in stream_val_microbatches(manifest, loader_config, microbatch_size, None)
    {
        let microbatch =
            microbatch_result.map_err(|err| format!("validation stream failed: {err}"))?;
        if total_samples >= limit {
            break;
        }
        let remaining = limit.saturating_sub(total_samples);
        if remaining == 0 {
            break;
        }
        let take = microbatch.len().min(remaining);
        microbatches.push(
            microbatch
                .into_iter()
                .take(take)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        total_samples += take;
    }
    Ok(Some(microbatches.into_boxed_slice()))
}

/// Default prefetch queue depth for the validation CPU producer thread.
///
/// Depth 2 keeps at most two host batches resident while the GPU processes
/// the current one. User config may raise this conservatively.
pub(super) fn run_validation_from_shards<TB>(
    model: &HydraModel<TB>,
    baseline_model: &HydraModel<TB>,
    context: ValidationContext<'_, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
    reader: &BcShardReader,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let ValidationContext {
        config,
        loader_config: _,
        manifest: _,
        cached_samples: _,
        shard_reader: _,
        device,
        loss_fn,
        exit_cfg,
    } = context;
    let ValidationRuntime {
        head_controller,
        progress,
    } = runtime;
    let model_valid = model.valid();
    let baseline_valid = baseline_model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut accumulator = ValidationAccumulator::new();
    let mut head_controller = head_controller;

    let total_rows = reader.sample_count();
    let limit_rows = validation_sample_limit
        .unwrap_or(total_rows)
        .min(total_rows);

    #[cfg(feature = "cuda-graph")]
    let mut staging_context = match device {
        burn::backend::libtorch::LibTorchDevice::Cuda(idx) => {
            let device_index = *idx as i64;
            Some((
                PinnedStagingArea::new(validation_batch_size),
                AsyncH2DContext::new(device_index),
                PreallocatedDeviceTensors::new(validation_batch_size, device),
            ))
        }
        _ => None,
    };

    // -- producer/consumer pipeline: CPU collation on a scoped background thread --
    let batch_size = validation_batch_size;
    let prefetch_depth = shard_prefetch_depth(config);
    let (tx, rx) = mpsc::sync_channel::<Result<(BcShardHostBatch, usize), String>>(prefetch_depth);
    let (recycle_tx, recycle_rx) = mpsc::sync_channel::<BcShardHostBatch>(prefetch_depth + 1);

    let consumer_result: Result<(), String> = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut scratch = reader.new_scratch(batch_size);
            let mut idx = 0usize;
            while idx < limit_rows {
                let take = batch_size.min(limit_rows - idx);
                let result = reader
                    .collate_host_batch_range_into(idx, take, false, &mut scratch)
                    .map(|()| {
                        let batch = if let Ok(mut recycled) = recycle_rx.try_recv() {
                            scratch.swap_batch(&mut recycled)
                        } else {
                            scratch.take_batch()
                        };
                        (batch, take)
                    });
                if tx.send(result).is_err() {
                    break;
                }
                idx += take;
            }
            drop(tx);
        });

        for recv_result in rx {
            let (host_batch, take) = recv_result?;
            #[cfg(feature = "cuda-graph")]
            let (shard_batch, recycled_host_batch) =
                {
                    if let Some((ref mut pinned_staging, ref h2d_ctx, ref mut gpu_tensors)) =
                        staging_context
                    {
                        (
                            super::pinned_transfer::materialize_staged_reuse_inner::<
                                ValidBackendOf<TB>,
                            >(
                                &host_batch, pinned_staging, h2d_ctx, device, gpu_tensors
                            ),
                            Some(host_batch),
                        )
                    } else {
                        (
                            host_batch.materialize_owned::<ValidBackendOf<TB>>(device),
                            None,
                        )
                    }
                };
            #[cfg(not(feature = "cuda-graph"))]
            let (shard_batch, recycled_host_batch) = (
                host_batch.materialize_owned::<ValidBackendOf<TB>>(device),
                None,
            );
            let obs = shard_batch.obs;
            let batch = shard_batch.batch;
            let targets = shard_batch.targets;
            process_validation_batch(
                ValidationModels {
                    model,
                    baseline_model,
                    model_valid: &model_valid,
                    baseline_valid: &baseline_valid,
                },
                loss_fn,
                exit_cfg,
                ValidationBatchInput {
                    obs,
                    batch,
                    targets,
                    sample_count: take,
                },
                &mut head_controller,
                &mut accumulator,
            );
            if let Some(host_batch) = recycled_host_batch {
                let _ = recycle_tx.try_send(host_batch);
            }
            if let Some(progress) = progress {
                progress.inc(take as u64);
            }
        }
        Ok(())
    });
    consumer_result?;

    Ok(finalize_validation_summary(accumulator))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::PathBuf;

    use burn::backend::libtorch::LibTorchDevice;
    use burn::tensor::Tensor;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use hydra_train::data::pipeline::{DataSource, stream_val_pass};
    use hydra_train::data::sample::MjaiBatch;
    use hydra_train::data::sample::MjaiSample;
    use hydra_train::model::HydraModelConfig;
    use hydra_train::training::bc::{BcExitConfig, bc_total_with_exit};
    use hydra_train::training::losses::HydraLossConfig;

    use super::super::TrainBackend;
    use crate::config::{BcHyperparamConfig, TrainConfig};
    use crate::resume::BestValidation;
    use crate::test_loose_replay_fixtures::write_real_preflight_fixture;
    use crate::test_support::dummy_train_config;

    type TestValidBackend = ValidBackendOf<TrainBackend>;

    struct FixtureRootGuard(PathBuf);

    impl FixtureRootGuard {
        fn new(path: PathBuf) -> Self {
            Self(path)
        }
    }

    impl Drop for FixtureRootGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn dummy_config() -> TrainConfig {
        let mut config = dummy_train_config();
        config.bc = BcHyperparamConfig::default();
        config.num_threads = Some(2);
        config
    }

    fn empty_manifest() -> DataManifest {
        DataManifest {
            sources: Vec::new(),
            total_games: 0,
            train_count: 0,
            val_count: 0,
            counts_exact: true,
        }
    }

    fn empty_summary(policy_loss: f64, agreement: f64) -> ValidationSummary {
        ValidationSummary {
            total_loss: policy_loss + 1.0,
            policy_loss,
            agreement,
            samples: 64,
            rare_actions: RareActionMetrics::default(),
            saw_exit_targets: false,
            saw_delta_q_targets: false,
            profiling: None,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        }
    }

    fn tiny_validation_model_config() -> HydraModelConfig {
        HydraModelConfig::new(1)
            .with_input_channels(hydra_train::config::INPUT_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
    }

    fn empty_batch(device: &LibTorchDevice, batch: usize) -> MjaiBatch<TestValidBackend> {
        MjaiBatch {
            obs: Tensor::zeros([batch, hydra_train::config::INPUT_CHANNELS, 34], device),
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

    fn delta_q_sample() -> MjaiSample {
        let mut sample = MjaiSample {
            obs: [0.0; OBS_SIZE],
            action: 0,
            legal_mask: [1.0; HYDRA_ACTION_SPACE],
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [255; 3],
            danger: [0.0; 102],
            danger_mask: [0.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        };
        sample.delta_q_target = Some([0.0; HYDRA_ACTION_SPACE]);
        sample.delta_q_mask = Some([1.0; HYDRA_ACTION_SPACE]);
        sample
    }

    fn tensor_rows_f32<const D: usize>(tensor: Tensor<TestValidBackend, D>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("tensor data should be readable as f32")
            .to_vec()
    }

    #[test]
    fn delta_q_promotion_snapshot_reflects_report_metrics_and_result() {
        let report = DeltaQPromotionReport {
            eligible_states: 16,
            compared_states: 8,
            masked_entries: 2,
            supported_actions_sum: 24,
            candidate_top1_agreement_count: 6,
            baseline_top1_agreement_count: 4,
            candidate_high_gap_top1_count: 3,
            baseline_high_gap_top1_count: 2,
            high_gap_states: 5,
            candidate_regret_sum: 2.0,
            baseline_regret_sum: 4.0,
            decision_lift_sum: 1.5,
            negative_lift_count: 1,
            candidate_regret_beats_baseline_count: 7,
            candidate_top1_beats_baseline_count: 5,
        };
        let result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };

        let snapshot = delta_q_promotion_snapshot_from_report(&report, &result);

        assert_eq!(snapshot.compared_states, 8);
        assert!((snapshot.candidate_top1_agreement - 0.75).abs() < 1e-12);
        assert!((snapshot.candidate_mean_regret - 0.25).abs() < 1e-12);
        assert!((snapshot.baseline_mean_regret - 0.5).abs() < 1e-12);
        assert!((snapshot.mean_decision_lift - 0.1875).abs() < 1e-12);
        assert!((snapshot.negative_lift_fraction - 0.125).abs() < 1e-12);
        assert!((snapshot.regret_beats_baseline_rate - 0.875).abs() < 1e-12);
        assert!((snapshot.top1_beats_baseline_rate - 0.625).abs() < 1e-12);
        assert!(snapshot.passed);
    }

    #[test]
    fn delta_q_policy_transfer_snapshot_reflects_report_metrics() {
        let report = DeltaQPolicyTransferReport {
            compared_states: 8,
            candidate_policy_top1_to_teacher_count: 5,
            baseline_policy_top1_to_teacher_count: 3,
            candidate_policy_regret_sum: 1.6,
            baseline_policy_regret_sum: 2.4,
            candidate_beats_baseline_count: 6,
            negative_transfer_count: 1,
        };

        let snapshot = delta_q_policy_transfer_snapshot_from_report(&report);

        assert_eq!(snapshot.compared_states, 8);
        assert!((snapshot.candidate_policy_top1_to_teacher - 0.625).abs() < 1e-12);
        assert!((snapshot.baseline_policy_top1_to_teacher - 0.375).abs() < 1e-12);
        assert!((snapshot.candidate_policy_mean_teacher_regret - 0.2).abs() < 1e-12);
        assert!((snapshot.baseline_policy_mean_teacher_regret - 0.3).abs() < 1e-12);
        assert!((snapshot.candidate_beats_baseline_rate - 0.75).abs() < 1e-12);
        assert!((snapshot.negative_transfer_fraction - 0.125).abs() < 1e-12);
    }

    #[test]
    fn delta_q_snapshots_handle_zero_compared_states() {
        let promotion_snapshot = delta_q_promotion_snapshot_from_report(
            &DeltaQPromotionReport::new(),
            &DeltaQPromotionResult {
                passed: false,
                criteria: Vec::new(),
            },
        );
        assert_eq!(promotion_snapshot.compared_states, 0);
        assert_eq!(promotion_snapshot.candidate_top1_agreement, 0.0);
        assert_eq!(promotion_snapshot.candidate_mean_regret, 0.0);
        assert_eq!(promotion_snapshot.baseline_mean_regret, 0.0);
        assert_eq!(promotion_snapshot.mean_decision_lift, 0.0);
        assert_eq!(promotion_snapshot.negative_lift_fraction, 0.0);
        assert_eq!(promotion_snapshot.regret_beats_baseline_rate, 0.0);
        assert_eq!(promotion_snapshot.top1_beats_baseline_rate, 0.0);
        assert!(!promotion_snapshot.passed);

        let transfer_snapshot =
            delta_q_policy_transfer_snapshot_from_report(&DeltaQPolicyTransferReport::new());
        assert_eq!(transfer_snapshot.compared_states, 0);
        assert_eq!(transfer_snapshot.candidate_policy_top1_to_teacher, 0.0);
        assert_eq!(transfer_snapshot.baseline_policy_top1_to_teacher, 0.0);
        assert_eq!(transfer_snapshot.candidate_policy_mean_teacher_regret, 0.0);
        assert_eq!(transfer_snapshot.baseline_policy_mean_teacher_regret, 0.0);
        assert_eq!(transfer_snapshot.candidate_beats_baseline_rate, 0.0);
        assert_eq!(transfer_snapshot.negative_transfer_fraction, 0.0);
    }

    #[test]
    fn empty_summary_preserves_optional_delta_q_fields_as_none() {
        let summary = empty_summary(0.8, 0.6);

        assert_eq!(summary.total_loss, 1.8);
        assert_eq!(summary.policy_loss, 0.8);
        assert_eq!(summary.agreement, 0.6);
        assert_eq!(summary.samples, 64);
        assert!(summary.profiling.is_none());
        assert!(summary.delta_q_promotion.is_none());
        assert!(summary.delta_q_promotion_result.is_none());
        assert!(summary.delta_q_promotion_snapshot.is_none());
        assert!(summary.delta_q_policy_transfer.is_none());
        assert!(summary.delta_q_policy_transfer_result.is_none());
        assert!(summary.delta_q_policy_transfer_snapshot.is_none());
    }

    #[test]
    fn empty_batch_initializes_optional_targets_and_shapes_consistently() {
        let device = LibTorchDevice::Cpu;
        let batch = empty_batch(&device, 3);

        assert_eq!(
            batch.obs.dims(),
            [3, hydra_train::config::INPUT_CHANNELS, 34]
        );
        assert_eq!(batch.actions.dims(), [3]);
        assert_eq!(batch.legal_mask.dims(), [3, 46]);
        assert_eq!(batch.grp_target.dims(), [3, 24]);
        assert_eq!(batch.tenpai_target.dims(), [3, 3]);
        assert_eq!(batch.danger_target.dims(), [3, 3, 34]);
        assert_eq!(batch.opp_next_target.dims(), [3, 3, 34]);
        assert_eq!(batch.score_pdf_target.dims(), [3, 64]);
        assert_eq!(batch.score_cdf_target.dims(), [3, 64]);
        assert!(batch.oracle_target.is_none());
        assert!(batch.exit_target.is_none());
        assert!(batch.delta_q_target.is_none());
        assert!(batch.belief_fields_target.is_none());
        assert!(batch.mixture_weight_target.is_none());
    }

    #[test]
    fn better_validation_rejects_higher_loss_and_lower_agreement_ties() {
        let summary = empty_summary(1.0, 0.4);

        assert!(!is_better_validation(
            &empty_summary(1.1, 0.9),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(!is_better_validation(
            &empty_summary(summary.policy_loss, summary.agreement),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(is_better_validation(
            &empty_summary(
                summary.policy_loss + f64::EPSILON / 2.0,
                summary.agreement + 0.05,
            ),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(!is_better_validation(
            &empty_summary(
                summary.policy_loss + f64::EPSILON / 2.0,
                summary.agreement - 0.05,
            ),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));
    }

    #[test]
    fn better_validation_accepts_first_result_without_prior_best() {
        assert!(is_better_validation(&empty_summary(1.2, 0.3), None));
    }

    #[test]
    fn validation_batch_stats_projects_breakdown_and_exit_adjusted_total() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TestValidBackend>(&device);
        let batch = empty_batch(&device, 2);
        let targets = batch.to_hydra_targets();
        let output = model.forward(batch.obs.clone());
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let exit_cfg = BcExitConfig::default();

        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = bc_total_with_exit(&output, &batch, &targets, &loss_fn, &exit_cfg);
        let stats = validation_batch_stats(2, &output, &batch, &targets, &breakdown, &total);
        let expected_total: f64 = total.clone().into_scalar().elem();

        assert_eq!(stats.sample_count, 2);
        assert!(stats.policy_agreement.is_finite());
        assert!(stats.loss_policy.is_finite());
        assert!(stats.loss_value.is_finite());
        assert!(stats.loss_grp.is_finite());
        assert!(stats.loss_tenpai.is_finite());
        assert!(stats.loss_danger.is_finite());
        assert!(stats.loss_opp_next.is_finite());
        assert!(stats.loss_score_pdf.is_finite());
        assert!(stats.loss_score_cdf.is_finite());
        assert!((stats.total_loss - expected_total).abs() < 1e-12);
    }

    #[test]
    fn run_validation_returns_zero_summary_for_empty_manifest() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: None,
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("empty manifest validation should succeed");

        assert_eq!(summary.total_loss, 0.0);
        assert_eq!(summary.policy_loss, 0.0);
        assert_eq!(summary.agreement, 0.0);
        assert_eq!(summary.samples, 0);
        assert_eq!(
            summary.profiling.as_ref().map(|p| p.stage.as_str()),
            Some("validation")
        );
        assert!(summary.delta_q_promotion.is_none());
        assert!(summary.delta_q_promotion_result.is_none());
        assert!(summary.delta_q_promotion_snapshot.is_none());
        assert!(summary.delta_q_policy_transfer.is_none());
        assert!(summary.delta_q_policy_transfer_result.is_none());
        assert!(summary.delta_q_policy_transfer_snapshot.is_none());
    }

    #[test]
    fn run_validation_wrapper_matches_policy_baseline_variant_on_empty_manifest() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());

        let wrapped = run_validation(
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: None,
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("wrapper validation should succeed");

        let direct = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: None,
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("direct validation should succeed");

        assert_eq!(wrapped.total_loss, direct.total_loss);
        assert_eq!(wrapped.policy_loss, direct.policy_loss);
        assert_eq!(wrapped.agreement, direct.agreement);
        assert_eq!(wrapped.samples, direct.samples);
        assert_eq!(wrapped.profiling, direct.profiling);
        assert!(wrapped.delta_q_promotion.is_none());
        assert!(wrapped.delta_q_policy_transfer.is_none());
    }

    #[test]
    fn run_validation_cached_samples_match_streamed_summary_on_real_loose_replay_buffers() {
        let root =
            FixtureRootGuard::new(write_real_preflight_fixture("validation-cache-equivalence"));
        let mut sources: Vec<DataSource> = std::fs::read_dir(&root.0)
            .expect("fixture dir should be readable")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .filter(|ext| *ext == "json")?;
                Some(DataSource::LooseFile(path))
            })
            .collect();
        sources.sort_by(|a, b| match (a, b) {
            (DataSource::LooseFile(lhs), DataSource::LooseFile(rhs)) => lhs.cmp(rhs),
            (
                DataSource::ParsedSampleCache { path: lhs, .. },
                DataSource::ParsedSampleCache { path: rhs, .. },
            ) => lhs.cmp(rhs),
            _ => std::cmp::Ordering::Equal,
        });
        let manifest = DataManifest {
            sources,
            total_games: 2,
            train_count: 0,
            val_count: 2,
            counts_exact: true,
        };
        let loader_config = StreamingLoaderConfig {
            buffer_games: 1,
            buffer_samples: 1,
            train_fraction: 0.0,
            seed: 0,
            archive_queue_bound: 1,
            max_skip_logs_per_source: 1,
            replay_target_profile: hydra_train::data::mjai_loader::ReplayTargetProfile::minimal_bc(
            ),
            ..StreamingLoaderConfig::default()
        };
        let streamed_buffers = stream_val_pass(&manifest, &loader_config, None)
            .collect::<std::io::Result<Vec<_>>>()
            .expect("real loose-replay validation fixture should stream successfully");
        let max_buffer_len = streamed_buffers
            .iter()
            .map(Vec::len)
            .max()
            .expect("real loose-replay fixture should yield at least one validation buffer");
        let total_streamed_samples: usize = streamed_buffers.iter().map(Vec::len).sum();
        assert!(
            streamed_buffers.len() >= 2,
            "small loader buffers should force multiple streamed validation buffers"
        );
        assert!(
            total_streamed_samples > max_buffer_len,
            "validation microbatch should be able to cross a streamed buffer boundary"
        );

        let mut config = dummy_config();
        config.validation_microbatch_size = Some(max_buffer_len + 1);
        config.max_validation_samples = Some(total_streamed_samples);
        config.train_fraction = 0.0;
        config.buffer_games = loader_config.buffer_games;
        config.buffer_samples = loader_config.buffer_samples;
        config.archive_queue_bound = loader_config.archive_queue_bound;
        config.device = "cpu".to_string();

        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let cached_samples = materialize_validation_samples(&config, &loader_config, &manifest)
            .expect("validation samples should materialize from the real loose-replay fixture")
            .expect("validation cache should materialize when a sample limit is configured");

        assert_eq!(
            cached_samples
                .iter()
                .map(|batch| batch.len())
                .sum::<usize>(),
            total_streamed_samples
        );

        let streamed = run_validation(
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: None,
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("streamed validation should succeed on the real loose-replay fixture");

        let cached = run_validation(
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: Some(&cached_samples),
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("cached validation should succeed on the real loose-replay fixture");

        let tolerance = 1e-6;
        assert_eq!(streamed.samples, cached.samples);
        assert!(
            (streamed.policy_loss - cached.policy_loss).abs() <= tolerance,
            "expected policy_loss equivalence within {tolerance}, got streamed={} cached={}",
            streamed.policy_loss,
            cached.policy_loss
        );
        assert!(
            (streamed.agreement - cached.agreement).abs() <= tolerance,
            "expected agreement equivalence within {tolerance}, got streamed={} cached={}",
            streamed.agreement,
            cached.agreement
        );
        assert!(
            (streamed.total_loss - cached.total_loss).abs() <= tolerance,
            "expected total_loss equivalence within {tolerance}, got streamed={} cached={}",
            streamed.total_loss,
            cached.total_loss
        );
    }

    #[test]
    fn run_validation_same_model_short_circuits_baseline_policy_forward() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let valid = model.valid();
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &empty_manifest(),
                cached_samples: Some(&cached_samples),
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("same-model validation should succeed");

        let profiling = summary.profiling.expect("profiling should exist");
        let baseline_stage = profiling
            .children
            .iter()
            .find(|child| child.stage == PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD)
            .expect("baseline child stage should exist");
        assert_eq!(baseline_stage.elapsed_seconds, 0.0);

        let obs = Tensor::zeros([1, hydra_train::config::INPUT_CHANNELS, 34], &device);
        let (policy_only_logits, _) = valid.forward_policy_value(obs.clone());
        let full_logits = valid.forward(obs).policy_logits;
        let policy_only_rows = tensor_rows_f32(policy_only_logits);
        let full_rows = tensor_rows_f32(full_logits);
        assert_eq!(policy_only_rows, full_rows);
    }

    #[test]
    fn run_validation_distinct_baseline_uses_policy_only_forward_without_drift() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let baseline = tiny_validation_model_config().init::<TrainBackend>(&device);
        let baseline_valid = baseline.valid();
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

        let summary = run_validation_with_policy_baseline(
            &model,
            &baseline,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &empty_manifest(),
                cached_samples: Some(&cached_samples),
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("distinct-baseline validation should succeed");

        assert!(summary.delta_q_promotion.is_some());
        assert!(summary.delta_q_policy_transfer.is_some());

        let obs = Tensor::zeros([1, hydra_train::config::INPUT_CHANNELS, 34], &device);
        let (policy_only_logits, _) = baseline_valid.forward_policy_value(obs.clone());
        let full_logits = baseline_valid.forward(obs).policy_logits;
        let policy_only_rows = tensor_rows_f32(policy_only_logits);
        let full_rows = tensor_rows_f32(full_logits);
        assert_eq!(policy_only_rows, full_rows);
    }

    #[test]
    fn validation_zero_summary_exposes_coarse_child_stages() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                cached_samples: None,
                shard_reader: None,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("empty manifest validation should succeed");

        let profiling = summary
            .profiling
            .expect("validation profiling should be attached");
        assert_eq!(profiling.stage, PROFILING_STAGE_VALIDATION);
        assert_eq!(profiling.children.len(), 2);
        assert_eq!(
            profiling.children[0].stage,
            PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS
        );
        assert_eq!(
            profiling.children[1].stage,
            PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD
        );
    }
}
