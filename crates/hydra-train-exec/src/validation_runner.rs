use crate::bc_metrics::{
    BatchMetricSums, batch_metric_sums_from_outputs, batch_stats_from_metric_sums,
};
use crate::bc_runtime::{BcExitConfig, gated_bc_context, maybe_add_exit_loss};
use crate::data::sample::{MjaiBcBatch, MjaiSample, collate_samples_bc_owned};
use crate::delta_q_promotion::{
    collect_policy_transfer_metrics_from_policy_outputs, collect_promotion_metrics_from_outputs,
};
use crate::losses::HydraLoss;
use crate::model::{HydraModel, HydraTrainModelExt};
use crate::nvtx;
use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use hydra_bc_shards::{
    BcShardHostBatch as ExtractedBcShardHostBatch, BcShardReader as ExtractedBcShardReader,
    BcShardSplit as ExtractedBcShardSplit, load_bc_shard_reader as load_extracted_bc_shard_reader,
};
use hydra_data_core::manifest::DataManifest;
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS, PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD,
    PROFILING_STAGE_VALIDATION, ProfilingEnvelope,
};
use hydra_train_runtime::validation::ValidationRunConfig;
use hydra_train_types::delta_q_promotion::{
    DeltaQPolicyTransferReport, DeltaQPolicyTransferThresholds, DeltaQPromotionReport,
    DeltaQPromotionResult, DeltaQPromotionThresholds, evaluate_policy_transfer_report,
    evaluate_promotion_report,
};
use hydra_train_types::head_gates::{AdvancedHead, TargetPresence};
use hydra_train_types::losses::HydraTargets;
use indicatif::ProgressBar;
use std::io;
use std::sync::mpsc;
use std::time::Instant;

use crate::validation::{DeltaQPolicyTransferSnapshot, DeltaQPromotionSnapshot, ValidationSummary};

/// Validation-split microbatches cached for repeated validation passes.
pub type ValidationCachedMicrobatches = Box<[Box<[MjaiSample]>]>;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

/// Minimal train configuration contract required by validation execution.
pub trait ValidationTrainConfig {
    /// Optional extracted BC shard manifest path.
    fn bc_shards_manifest_path(&self) -> Option<&std::path::Path>;
    /// Runtime validation run configuration.
    fn validation_run_config(&self) -> ValidationRunConfig;
}

impl ValidationTrainConfig for hydra_train_runtime::config::TrainConfig {
    fn bc_shards_manifest_path(&self) -> Option<&std::path::Path> {
        self.bc_shards_manifest_path.as_deref()
    }

    fn validation_run_config(&self) -> ValidationRunConfig {
        ValidationRunConfig::from_config(self)
    }
}

/// Streaming loader contract required by validation execution.
pub trait ValidationDataLoader {
    /// Returns validation microbatches for the given manifest and optional progress bar.
    fn stream_val_microbatches<'a>(
        &'a self,
        manifest: &'a DataManifest,
        microbatch_size: usize,
        progress: Option<&'a ProgressBar>,
    ) -> Box<dyn Iterator<Item = io::Result<Vec<MjaiSample>>> + 'a>;
}

/// Runtime context for one validation pass.
pub struct ValidationContext<'a, C, L, B: Backend> {
    /// Training config contract.
    pub config: &'a C,
    /// Streaming validation loader.
    pub loader: &'a L,
    /// Data manifest.
    pub manifest: &'a DataManifest,
    /// Optional cached validation microbatches.
    pub cached_samples: Option<&'a [Box<[MjaiSample]>]>,
    /// Device used for validation tensors.
    pub device: &'a B::Device,
    /// Validation loss function.
    pub loss_fn: &'a HydraLoss<B>,
    /// ExIt auxiliary-loss configuration.
    pub exit_cfg: &'a BcExitConfig,
}

/// Runtime-only mutable validation handles.
pub struct ValidationRuntime<'a> {
    /// Optional advanced-head activation controller.
    pub head_controller: Option<&'a mut HeadActivationController>,
    /// Optional validation progress bar.
    pub progress: Option<&'a ProgressBar>,
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
            rare_actions: hydra_train_runtime::progress::RareActionMetrics::default(),
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

/// Runs validation against `model`, using `model` as the DeltaQ baseline.
pub fn run_validation<TB, C, L>(
    model: &HydraModel<TB>,
    context: ValidationContext<'_, C, L, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    C: ValidationTrainConfig,
    L: ValidationDataLoader,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    run_validation_with_policy_baseline(model, model, context, runtime)
}

/// Runs validation against `model` and compares DeltaQ policy transfer against `baseline_model`.
pub fn run_validation_with_policy_baseline<TB, C, L>(
    model: &HydraModel<TB>,
    baseline_model: &HydraModel<TB>,
    context: ValidationContext<'_, C, L, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    C: ValidationTrainConfig,
    L: ValidationDataLoader,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    if let Some(manifest_path) = context.config.bc_shards_manifest_path() {
        let reader =
            load_extracted_bc_shard_reader(manifest_path, ExtractedBcShardSplit::Validation)?;
        return run_validation_from_shard_reader(model, baseline_model, context, runtime, &reader);
    }
    let ValidationContext {
        config,
        loader,
        manifest,
        cached_samples,
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
    let run_config = config.validation_run_config();
    let validation_limits = run_config.limits;
    let validation_batch_size = validation_limits.microbatch_size;
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
            if validation_limits.reached_sample_limit(accumulator.total_samples) {
                break;
            }
            let take = validation_limits.capped_len(accumulator.total_samples, chunk.len());
            let capped_chunk = &chunk[..take];
            if capped_chunk.is_empty() {
                break;
            }
            run_chunk(capped_chunk, &mut head_controller, &mut accumulator)?;
        }
    } else {
        for microbatch_result in
            loader.stream_val_microbatches(manifest, validation_batch_size, progress)
        {
            let microbatch =
                microbatch_result.map_err(|err| format!("validation stream failed: {err}"))?;
            if validation_limits.reached_sample_limit(accumulator.total_samples) {
                break;
            }
            let take = validation_limits.capped_len(accumulator.total_samples, microbatch.len());
            let capped_chunk = &microbatch[..take];
            if capped_chunk.is_empty() {
                break;
            }
            run_chunk(capped_chunk, &mut head_controller, &mut accumulator)?;
            if validation_limits.reached_sample_limit(accumulator.total_samples) {
                break;
            }
        }
    }

    Ok(finalize_validation_summary(accumulator))
}

/// Materializes validation microbatches when a sample limit makes caching useful.
pub fn materialize_validation_samples<C, L>(
    config: &C,
    loader: &L,
    manifest: &DataManifest,
) -> Result<Option<ValidationCachedMicrobatches>, String>
where
    C: ValidationTrainConfig,
    L: ValidationDataLoader,
{
    if config.bc_shards_manifest_path().is_some() {
        return Ok(None);
    }
    let limits = config.validation_run_config().limits;
    let Some(limit) = limits.sample_limit else {
        return Ok(None);
    };
    let microbatch_size = limits.microbatch_size;
    let mut microbatches = Vec::new();
    let mut total_samples = 0usize;
    for microbatch_result in loader.stream_val_microbatches(manifest, microbatch_size, None) {
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

/// Runs validation from a caller-supplied extracted BC shard reader.
pub fn run_validation_from_shards<TB, C, L>(
    model: &HydraModel<TB>,
    baseline_model: &HydraModel<TB>,
    context: ValidationContext<'_, C, L, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
    reader: &ExtractedBcShardReader,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    C: ValidationTrainConfig,
    L: ValidationDataLoader,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    run_validation_from_shard_reader(model, baseline_model, context, runtime, reader)
}

fn run_validation_from_shard_reader<TB, C, L>(
    model: &HydraModel<TB>,
    baseline_model: &HydraModel<TB>,
    context: ValidationContext<'_, C, L, ValidBackendOf<TB>>,
    runtime: ValidationRuntime<'_>,
    reader: &ExtractedBcShardReader,
) -> Result<ValidationSummary, String>
where
    TB: AutodiffBackend<Device = LibTorchDevice>,
    C: ValidationTrainConfig,
    L: ValidationDataLoader,
    ValidBackendOf<TB>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let ValidationContext {
        config,
        loader: _,
        manifest: _,
        cached_samples: _,
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
    let run_config = config.validation_run_config();
    let validation_limits = run_config.limits;
    let validation_batch_size = validation_limits.microbatch_size;
    let mut accumulator = ValidationAccumulator::new();
    let mut head_controller = head_controller;

    let total_rows = reader.sample_count();
    let limit_rows = validation_limits.bounded_total_rows(total_rows);
    let batch_size = validation_batch_size;
    let prefetch_depth = run_config.shard_prefetch_depth;
    let (tx, rx) =
        mpsc::sync_channel::<Result<(ExtractedBcShardHostBatch, usize), String>>(prefetch_depth);
    let (recycle_tx, recycle_rx) =
        mpsc::sync_channel::<ExtractedBcShardHostBatch>(prefetch_depth + 1);

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
            let shard_batch =
                materialize_extracted_host_batch::<ValidBackendOf<TB>>(host_batch, device);
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
            if let Some(progress) = progress {
                progress.inc(take as u64);
            }
        }
        let _ = recycle_tx;
        Ok(())
    });
    consumer_result?;

    Ok(finalize_validation_summary(accumulator))
}

struct BcShardBatch<B: Backend> {
    obs: Tensor<B, 3>,
    batch: MjaiBcBatch<B>,
    targets: HydraTargets<B>,
}

fn materialize_extracted_host_batch<B: Backend>(
    host: ExtractedBcShardHostBatch,
    device: &B::Device,
) -> BcShardBatch<B> {
    let target_presence = target_presence_from_extracted_host_batch(&host, host.batch_size);
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
) -> BcShardBatch<B> {
    let b = batch_size;

    let obs = Tensor::<B, 3>::from_data(
        TensorData::new(obs_flat, [b, hydra_core::encoder::NUM_CHANNELS, 34]),
        device,
    );
    let policy_target = policy_target_from_action_slice::<B>(actions.as_slice(), b, device);
    let actions_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(actions, [b]), device);
    let legal_mask = Tensor::<B, 2>::from_data(
        TensorData::new(legal_mask_flat, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
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
        Tensor::<B, 2>::from_data(
            TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
            device,
        )
    });
    let exit_mask_tensor = exit_mask_flat.map(|buf| {
        Tensor::<B, 2>::from_data(
            TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
            device,
        )
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
            Tensor::<B, 2>::from_data(
                TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
                device,
            )
        }),
        delta_q_mask: delta_q_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(
                TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
                device,
            )
        }),
        safety_residual_target: safety_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(
                TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
                device,
            )
        }),
        safety_residual_mask: safety_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(
                TensorData::new(buf, [b, hydra_core::action::HYDRA_ACTION_SPACE]),
                device,
            )
        }),
        oracle_guidance_mask: Some(oracle_target_mask),
        target_presence: Some(target_presence),
    };

    BcShardBatch {
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
    let mut flat = vec![0.0f32; batch_size * hydra_core::action::HYDRA_ACTION_SPACE];
    for (row, &action) in actions.iter().take(batch_size).enumerate() {
        if action >= 0 {
            let idx = action as usize;
            if idx < hydra_core::action::HYDRA_ACTION_SPACE {
                flat[row * hydra_core::action::HYDRA_ACTION_SPACE + idx] = 1.0;
            }
        }
    }
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [batch_size, hydra_core::action::HYDRA_ACTION_SPACE]),
        device,
    )
}

fn target_presence_from_extracted_host_batch(
    host: &ExtractedBcShardHostBatch,
    batch_size: usize,
) -> TargetPresence {
    let mut presence = TargetPresence::with_batch_size(batch_size);
    presence.counts[AdvancedHead::OracleCritic.index()] = host
        .oracle_target_mask
        .iter()
        .take(batch_size)
        .filter(|&&value| value > 0.0)
        .count();
    if let (Some(_target), Some(mask)) = (&host.safety_target_flat, &host.safety_mask_flat) {
        presence.counts[AdvancedHead::SafetyResidual.index()] =
            count_nonzero_action_rows(mask, batch_size, hydra_core::action::HYDRA_ACTION_SPACE);
    }
    if let (Some(_target), Some(mask)) = (&host.delta_q_target_flat, &host.delta_q_mask_flat) {
        let (rows, actions) = count_nonzero_action_rows_and_entries(
            mask,
            batch_size,
            hydra_core::action::HYDRA_ACTION_SPACE,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bc_runtime::bc_total_with_exit_from_breakdown;
    use crate::data::sample::MjaiBatch;
    use crate::model::HydraModelConfig;
    use burn::backend::libtorch::LibTorchDevice;
    use burn::tensor::Tensor;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
    use hydra_train_types::losses::HydraLossConfig;

    type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch>;
    type TestValidBackend = ValidBackendOf<TrainBackend>;

    struct EmptyLoader;

    impl ValidationDataLoader for EmptyLoader {
        fn stream_val_microbatches(
            &self,
            _manifest: &DataManifest,
            _microbatch_size: usize,
            _progress: Option<&ProgressBar>,
        ) -> Box<dyn Iterator<Item = io::Result<Vec<MjaiSample>>> + '_> {
            Box::new(std::iter::empty())
        }
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

    fn dummy_config() -> hydra_train_runtime::config::TrainConfig {
        hydra_train_runtime::config::TrainConfig {
            data_dir: std::path::PathBuf::new(),
            output_dir: std::path::PathBuf::new(),
            num_epochs: 1,
            batch_size: 1,
            microbatch_size: None,
            validation_microbatch_size: Some(1),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_data_core::SourceFilterConfig::default(),
            augment: false,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
            rl: None,
            bc: hydra_train_runtime::config::BcHyperparamConfig::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
            buffer_games: 1,
            buffer_samples: 1,
            num_threads: Some(1),
            tensorboard: false,
            archive_queue_bound: 1,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 1,
            log_every_n_steps: 1,
            validate_every_n_steps: 0,
            checkpoint_every_n_steps: 0,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: Some(1),
            preflight: hydra_train_runtime::preflight::PreflightConfig::default(),
        }
    }

    fn tiny_validation_model_config() -> HydraModelConfig {
        HydraModelConfig::new(1)
            .with_input_channels(NUM_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
    }

    fn empty_batch(device: &LibTorchDevice, batch: usize) -> MjaiBatch<TestValidBackend> {
        MjaiBatch {
            obs: Tensor::zeros([batch, NUM_CHANNELS, 34], device),
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
    fn validation_batch_stats_projects_breakdown_and_exit_adjusted_total() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TestValidBackend>(&device);
        let batch = empty_batch(&device, 2);
        let targets = batch.to_hydra_targets();
        let output = model.forward(batch.obs.clone());
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let exit_cfg = BcExitConfig::default();

        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = bc_total_with_exit_from_breakdown(&output, &batch, &breakdown, &exit_cfg);
        let stats = crate::bc_metrics::batch_stats_from_outputs(
            2,
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        );
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
        let loader = EmptyLoader;
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader: &loader,
                manifest: &manifest,
                cached_samples: None,
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
    fn run_validation_same_model_short_circuits_baseline_policy_forward() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_validation_model_config().init::<TrainBackend>(&device);
        let valid = model.valid();
        let config = dummy_config();
        let loader = EmptyLoader;
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader: &loader,
                manifest: &empty_manifest(),
                cached_samples: Some(&cached_samples),
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

        let obs = Tensor::zeros([1, NUM_CHANNELS, 34], &device);
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
        let loader = EmptyLoader;
        let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
        let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

        let summary = run_validation_with_policy_baseline(
            &model,
            &baseline,
            ValidationContext {
                config: &config,
                loader: &loader,
                manifest: &empty_manifest(),
                cached_samples: Some(&cached_samples),
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

        let obs = Tensor::zeros([1, NUM_CHANNELS, 34], &device);
        let (policy_only_logits, _) = baseline_valid.forward_policy_value(obs.clone());
        let full_logits = baseline_valid.forward(obs).policy_logits;
        let policy_only_rows = tensor_rows_f32(policy_only_logits);
        let full_rows = tensor_rows_f32(full_logits);
        assert_eq!(policy_only_rows, full_rows);
    }
}
