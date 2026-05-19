use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::data::sample::{MjaiSample, collate_samples, collate_samples_bc_owned_timed};
use crate::data_pipeline::{
    DataManifest, StreamingLoaderConfig, stream_train_epoch, stream_val_microbatches,
};
use crate::losses::HydraLoss;
use crate::model::{HydraModel, HydraModelConfig, HydraModelInit};
use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use hydra_bc_shards::{
    BC_SHARD_HEADER_SIZE, BcShardReader as ExtractedBcShardReader,
    BcShardSplit as ExtractedBcShardSplit, load_bc_shard_reader as load_extracted_bc_shard_reader,
};
use hydra_replay_loader::ReplayTargetProfile;
use hydra_train_runtime::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train_runtime::preflight::{
    PreflightBenchMode, PreflightBenchReport, PreflightBenchRow, PreflightBenchStatus,
    PreflightBenchTuple, PreflightCodec, PreflightConfig, PreflightShuffleMode, ProbeKind,
    ProbeResult, ProbeStatus,
};

type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch<f32>>;
use crate::artifacts::{
    BcArtifactPaths, ManifestCacheHitSource, ManifestCacheRequest, PreflightPaths,
    load_or_scan_manifest_cache, load_or_scan_manifest_cache_with_source,
};
use crate::bc_fixed_shape::{FixedShapeProbeConfig, probe_train_fixed_chunks};
use crate::bc_metrics::batch_stats_from_outputs;
use crate::config_runtime::{configure_threads, train_device, trainer_config_from_train_config};
use crate::epoch_runner::{
    TrainLogicalBatchConfig, materialize_host_batch_owned, train_device_batch,
};
#[cfg(feature = "cuda-graph")]
use crate::pinned_transfer::PinnedTransferStaging;
#[cfg(test)]
use crate::presentation::format_probe_status_line;
use crate::presentation::make_spinner;
use crate::probe_ladder::probe_only_candidate_ladder;
use crate::probe_search::probe_candidate_ladder;
use crate::probe_summary::probe_kind_name;
use crate::probe_transport::{ProbeBatchArtifact, write_probe_batch_artifact, write_probe_result};
use crate::validation::ValidationSummary;
use crate::validation_runner::{
    ValidationContext, ValidationDataLoader, ValidationRuntime, run_validation,
};
use hydra_data_core::manifest::DataManifest as CoreDataManifest;
use hydra_train_runtime::config::{ProbeChildRequest, TrainConfig};
use hydra_train_runtime::loss_policy::{build_bc_exit_config, build_loss_config};
use hydra_train_runtime::probe_request::{
    ProbeBatchRequest, ProbeRequest, probe_batch_child_request_from_cli,
    probe_child_request_from_cli,
};
use hydra_train_runtime::schedule::{TrainerScheduleConfig, effective_lr};
use hydra_train_runtime::validation::ValidationRunLimits;
use indicatif::ProgressBar;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

type BenchmarkOptimizerOf<B> = OptimizerAdaptor<Adam, HydraModel<B>, B>;

struct TrainValidationLoader<'a> {
    config: &'a StreamingLoaderConfig,
}

impl ValidationDataLoader for TrainValidationLoader<'_> {
    fn stream_val_microbatches<'b>(
        &'b self,
        manifest: &'b CoreDataManifest,
        microbatch_size: usize,
        progress: Option<&'b ProgressBar>,
    ) -> Box<dyn Iterator<Item = io::Result<Vec<MjaiSample>>> + 'b> {
        Box::new(stream_val_microbatches(
            manifest,
            self.config,
            microbatch_size,
            progress,
        ))
    }
}

fn validation_loader(config: &StreamingLoaderConfig) -> TrainValidationLoader<'_> {
    TrainValidationLoader { config }
}

fn trainer_schedule(config: &hydra_train_types::config::BCTrainerConfig) -> TrainerScheduleConfig {
    TrainerScheduleConfig::new(config.lr, config.min_learning_rate, config.warmup_steps)
}

struct ProbeLoopState {
    completed_steps: usize,
    measure_start: Option<Instant>,
}

impl ProbeLoopState {
    fn new() -> Self {
        Self {
            completed_steps: 0,
            measure_start: None,
        }
    }
}

fn emit_probe_start_progress(request: ProbeRequest, microbatch_size: usize) -> Result<(), String> {
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
        probe_kind_name(request.kind),
        microbatch_size,
        request.warmup_steps,
        request.measure_steps,
    ))
}

fn advance_probe_loop(
    state: &mut ProbeLoopState,
    request: ProbeRequest,
    microbatch_size: usize,
    measured_samples_per_step: usize,
) -> Result<Option<f64>, String> {
    emit_probe_step_progress(
        request.kind,
        microbatch_size,
        state.completed_steps,
        request,
        state.measure_start,
        measured_samples_per_step,
    )?;
    state.completed_steps += 1;
    if state.completed_steps == request.warmup_steps {
        state.measure_start = Some(Instant::now());
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=measure_start total_steps={}",
            probe_kind_name(request.kind),
            microbatch_size,
            request.measure_steps.max(1)
        ))?;
    }
    let target_steps = request.warmup_steps + request.measure_steps;
    if state.completed_steps >= target_steps {
        let elapsed = state
            .measure_start
            .map(|start| start.elapsed())
            .unwrap_or_default();
        return Ok(Some(measure_samples_per_second(
            request.measure_steps.max(1) * measured_samples_per_step,
            elapsed,
        )));
    }
    Ok(None)
}

fn emit_probe_progress(line: &str) -> Result<(), String> {
    println!("{}", line.trim());
    std::io::stdout()
        .flush()
        .map_err(|err| format!("failed flushing probe progress output: {err}"))
}

fn emit_probe_init_phase(kind_name: &str, candidate_mb: usize, phase: &str) -> Result<(), String> {
    emit_probe_progress(&format!(
        "probe_progress kind={kind_name} candidate_mb={candidate_mb} phase={phase}"
    ))
}

fn emit_probe_init_ready(
    kind: ProbeKind,
    kind_name: &str,
    candidate_mb: usize,
    model_ms: u128,
    optimizer_ms: u128,
    loss_ms: u128,
) -> Result<(), String> {
    crate::system_metrics::emit_system_metrics_event(
        &crate::system_metrics::probe_child_init_event(
            kind,
            candidate_mb,
            model_ms,
            optimizer_ms,
            loss_ms,
        ),
    );
    emit_probe_progress(&format!(
        "probe_progress kind={kind_name} candidate_mb={candidate_mb} phase=init_ready model_ms={model_ms} optimizer_ms={optimizer_ms} loss_ms={loss_ms}"
    ))
}

fn emit_probe_step_progress(
    kind: ProbeKind,
    microbatch_size: usize,
    completed_steps: usize,
    request: ProbeRequest,
    measure_start: Option<Instant>,
    measured_samples_per_step: usize,
) -> Result<(), String> {
    if completed_steps < request.warmup_steps {
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=warmup step={}/{}",
            probe_kind_name(kind),
            microbatch_size,
            completed_steps + 1,
            request.warmup_steps.max(1)
        ))
    } else {
        let measure_step = completed_steps + 1 - request.warmup_steps;
        let throughput = measure_start
            .map(|start| {
                measure_samples_per_second(
                    measure_step * measured_samples_per_step,
                    start.elapsed(),
                )
            })
            .unwrap_or(0.0);
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=measure step={}/{} throughput={:.2} samples/s",
            probe_kind_name(kind),
            microbatch_size,
            measure_step,
            request.measure_steps.max(1),
            throughput,
        ))
    }
}

pub fn measure_samples_per_second(samples: usize, elapsed: Duration) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        0.0
    } else {
        samples as f64 / seconds
    }
}

fn probe_train_candidate_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    run_train_measurement_loop::<B>(TrainMeasurementSpec {
        config,
        model_config,
        candidate_microbatch: request.candidate_microbatch,
        warmup_steps: request.warmup_steps,
        measure_steps: request.measure_steps,
        loader_config,
        manifest,
        train_device,
        on_start: Box::new(|candidate_microbatch, warmup_steps, measure_steps| {
            emit_probe_progress(&format!(
                "probe_progress kind=train candidate_mb={} phase=starting warmup_steps={} measure_steps={}",
                candidate_microbatch, warmup_steps, measure_steps
            ))
        }),
        on_step: Box::new(
            |completed_steps, candidate_microbatch, request, measure_start| {
                emit_probe_step_progress(
                    ProbeKind::Train,
                    candidate_microbatch,
                    completed_steps,
                    ProbeRequest {
                        kind: ProbeKind::Train,
                        candidate_microbatch,
                        warmup_steps: request.warmup_steps,
                        measure_steps: request.measure_steps,
                    },
                    measure_start,
                    config.batch_size,
                )
            },
        ),
        on_measure_start: Box::new(|candidate_microbatch, measure_steps| {
            emit_probe_progress(&format!(
                "probe_progress kind=train candidate_mb={} phase=measure_start total_steps={}",
                candidate_microbatch,
                measure_steps.max(1)
            ))
        }),
        insufficient_data: Box::new(|candidate_microbatch| {
            format!(
                "not enough train data to finish preflight probe at microbatch {}",
                candidate_microbatch
            )
        }),
    })
}

type TrainMeasurementStepCallback<'a> =
    dyn FnMut(usize, usize, ProbeRequest, Option<Instant>) -> Result<(), String> + 'a;

pub struct TrainMeasurementSpec<'a> {
    pub config: &'a TrainConfig,
    pub model_config: &'a HydraModelConfig,
    pub candidate_microbatch: usize,
    pub warmup_steps: usize,
    pub measure_steps: usize,
    pub loader_config: &'a StreamingLoaderConfig,
    pub manifest: &'a DataManifest,
    pub train_device: &'a LibTorchDevice,
    pub on_start: Box<dyn FnMut(usize, usize, usize) -> Result<(), String> + 'a>,
    pub on_step: Box<TrainMeasurementStepCallback<'a>>,
    pub on_measure_start: Box<dyn FnMut(usize, usize) -> Result<(), String> + 'a>,
    pub insufficient_data: Box<dyn FnOnce(usize) -> String + 'a>,
}

pub fn run_train_measurement_loop<B>(spec: TrainMeasurementSpec<'_>) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    let TrainMeasurementSpec {
        config,
        model_config,
        candidate_microbatch,
        warmup_steps,
        measure_steps,
        loader_config,
        manifest,
        train_device,
        mut on_start,
        mut on_step,
        mut on_measure_start,
        insufficient_data,
    } = spec;
    let train_cfg = trainer_config_from_train_config(config);

    emit_probe_init_phase("train", candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let mut model = model_config.init::<B>(train_device);
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", candidate_microbatch, "init_optimizer")?;
    let t0 = Instant::now();
    let mut optimizer: BenchmarkOptimizerOf<B> = train_cfg.optimizer_config().init();
    let optimizer_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Train,
        "train",
        candidate_microbatch,
        model_ms,
        optimizer_ms,
        loss_ms,
    )?;

    let microbatch_size = candidate_microbatch.min(config.batch_size).max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut probe_state = ProbeLoopState::new();
    let mut pending_samples = std::collections::VecDeque::new();
    on_start(microbatch_size, warmup_steps, measure_steps)?;

    for buffer_result in stream_train_epoch(manifest, loader_config, 0, None) {
        let buffer =
            buffer_result.map_err(|err| format!("preflight train stream failed: {err}"))?;
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let lr = effective_lr(
                trainer_schedule(&train_cfg),
                probe_state.completed_steps,
                target_steps.max(1),
            );
            let grads = if let Some(grads) = probe_train_fixed_chunks(FixedShapeProbeConfig {
                logical_batch: &logical_batch,
                augment: config.augment,
                microbatch_size,
                train_device,
                loss_fn: &loss_fn,
                model: &model,
                use_amp: false,
            })? {
                grads
            } else {
                let logical_batch_len = logical_batch.len().max(1) as f32;
                let mut accumulator: GradientsAccumulator<HydraModel<B>> =
                    GradientsAccumulator::new();
                for chunk in logical_batch.chunks(microbatch_size) {
                    let Some((obs, targets)) =
                        collate_samples::<B>(chunk, config.augment, train_device)
                            .map_err(|err| format!("preflight train collation failed: {err}"))?
                    else {
                        continue;
                    };
                    let output = model.forward(obs);
                    let breakdown = loss_fn.total_loss(&output, &targets);
                    let chunk_weight = chunk.len() as f32 / logical_batch_len;
                    let grads = (breakdown.total * chunk_weight).backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    accumulator.accumulate(&model, grads);
                }
                accumulator.grads()
            };
            model = optimizer.step(lr, model, grads);
            on_step(
                probe_state.completed_steps,
                microbatch_size,
                ProbeRequest {
                    kind: ProbeKind::Train,
                    candidate_microbatch: microbatch_size,
                    warmup_steps,
                    measure_steps,
                },
                probe_state.measure_start,
            )?;
            probe_state.completed_steps += 1;
            if probe_state.completed_steps == warmup_steps {
                probe_state.measure_start = Some(Instant::now());
                on_measure_start(microbatch_size, measure_steps)?;
            }
            if probe_state.completed_steps >= target_steps {
                let elapsed = probe_state
                    .measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    measure_steps.max(1) * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    Err(insufficient_data(microbatch_size))
}

fn probe_train_candidate_from_shards_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    train_device: &LibTorchDevice,
    reader: &ExtractedBcShardReader,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            FloatTensorPrimitive = burn::backend::libtorch::TchTensor,
            IntTensorPrimitive = burn::backend::libtorch::TchTensor,
        >,
{
    let train_cfg = trainer_config_from_train_config(config);

    emit_probe_init_phase("train", request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let mut model = Some(model_config.init::<B>(train_device));
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", request.candidate_microbatch, "init_optimizer")?;
    let t0 = Instant::now();
    let mut optimizer: BenchmarkOptimizerOf<B> = train_cfg.optimizer_config().init();
    let optimizer_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("train", request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let mut head_controller = HeadActivationController::new(
        HeadActivationConfig::default_with_params(model_config.estimated_params()),
    );
    let loss_ms = t0.elapsed().as_millis();

    let microbatch_size = request.candidate_microbatch.min(config.batch_size).max(1);
    let target_steps = request.warmup_steps + request.measure_steps;
    let mut probe_state = ProbeLoopState::new();
    let mut scratch = reader.new_scratch(config.batch_size);

    #[cfg(feature = "cuda-graph")]
    {
        emit_probe_init_phase("train", request.candidate_microbatch, "init_cuda_staging")?;
    }
    #[cfg(feature = "cuda-graph")]
    let mut staging_context = PinnedTransferStaging::from_device(config.batch_size, train_device);

    emit_probe_init_ready(
        ProbeKind::Train,
        "train",
        request.candidate_microbatch,
        model_ms,
        optimizer_ms,
        loss_ms,
    )?;

    emit_probe_start_progress(request, microbatch_size)?;

    let total_rows = reader.sample_count();
    let mut idx = 0usize;
    while idx < total_rows {
        let take = config.batch_size.min(total_rows - idx);
        if take < config.batch_size {
            break;
        }
        reader.collate_host_batch_range_into(idx, take, config.augment, &mut scratch)?;
        let host_batch = scratch.take_batch();
        let materialize_started = Instant::now();
        let shard_batch = {
            #[cfg(feature = "cuda-graph")]
            {
                if let Some(staging) = staging_context.as_mut() {
                    let (pinned_staging, h2d_ctx, gpu_tensors) = staging.as_parts();
                    crate::pinned_transfer::materialize_staged_reuse::<B>(
                        &host_batch,
                        pinned_staging,
                        h2d_ctx,
                        train_device,
                        gpu_tensors,
                    )
                    .0
                } else {
                    materialize_host_batch_owned::<B>(host_batch, train_device)
                }
            }
            #[cfg(not(feature = "cuda-graph"))]
            {
                materialize_host_batch_owned::<B>(host_batch, train_device)
            }
        };
        let mut timing = hydra_train_runtime::progress::TrainSubStageTiming::default();
        timing.h2d_tensor_materialize_seconds += materialize_started.elapsed().as_secs_f64();
        let lr = effective_lr(
            trainer_schedule(&train_cfg),
            probe_state.completed_steps,
            target_steps.max(1),
        );
        let _ = train_device_batch(
            shard_batch,
            config.batch_size,
            timing,
            TrainLogicalBatchConfig {
                microbatch_size,
                augment: config.augment,
                train_device,
                loss_fn: &loss_fn,
                bc_exit_cfg: &exit_cfg,
                lr,
                use_amp: false,
            },
            &mut head_controller,
            &mut model,
            &mut optimizer,
        )?;
        if let Some(throughput) = advance_probe_loop(
            &mut probe_state,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: microbatch_size,
                warmup_steps: request.warmup_steps,
                measure_steps: request.measure_steps,
            },
            microbatch_size,
            config.batch_size,
        )? {
            return Ok(throughput);
        }
        idx += take;
    }

    Err(format!(
        "not enough train shard data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}

fn probe_validation_candidate_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    request: ProbeRequest,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    emit_probe_init_phase("validation", request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let model = model_config.init::<B>(train_device);
    let model_valid = model.valid();
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("validation", request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Validation,
        "validation",
        request.candidate_microbatch,
        model_ms,
        0,
        loss_ms,
    )?;

    let microbatch_size = request.candidate_microbatch.max(1);
    let mut probe_state = ProbeLoopState::new();
    emit_probe_start_progress(request, microbatch_size)?;

    for microbatch_result in stream_val_microbatches(manifest, loader_config, microbatch_size, None)
    {
        let microbatch = microbatch_result
            .map_err(|err| format!("preflight validation stream failed: {err}"))?;
        let Some(collated) = collate_samples_bc_owned_timed::<ValidBackendOf<B>>(
            microbatch.as_slice(),
            false,
            train_device,
        )
        .map_err(|err| format!("preflight validation collation failed: {err}"))?
        else {
            continue;
        };
        let obs = collated.obs;
        let batch = collated.batch;
        let targets = collated.targets;
        let output = model_valid.forward(obs);
        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = crate::bc_runtime::maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            &crate::bc_runtime::BcExitConfig::default(),
        );
        let _ = batch_stats_from_outputs(
            microbatch.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        );
        if let Some(throughput) =
            advance_probe_loop(&mut probe_state, request, microbatch_size, microbatch_size)?
        {
            return Ok(throughput);
        }
    }

    Err(format!(
        "not enough validation data to finish preflight probe at microbatch {}",
        microbatch_size
    ))
}
fn execute_shard_validation_probe<RunValidation>(
    config: &TrainConfig,
    _request: ProbeRequest,
    sample_count: usize,
    started_at: Instant,
    run_validation_probe: RunValidation,
) -> Result<f64, String>
where
    RunValidation: FnOnce() -> Result<ValidationSummary, String>,
{
    let _summary = run_validation_probe()?;
    Ok(measure_samples_per_second(
        ValidationRunLimits::from_config(config).bounded_total_rows(sample_count),
        started_at.elapsed(),
    ))
}

fn probe_validation_candidate_from_shards_for_backend<B>(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    _request: ProbeRequest,
    train_device: &LibTorchDevice,
    reader: &ExtractedBcShardReader,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    emit_probe_init_phase("validation", _request.candidate_microbatch, "init_model")?;
    let t0 = Instant::now();
    let model = model_config.init::<B>(train_device);
    let model_ms = t0.elapsed().as_millis();

    emit_probe_init_phase("validation", _request.candidate_microbatch, "init_loss")?;
    let t0 = Instant::now();
    let loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let loss_ms = t0.elapsed().as_millis();

    emit_probe_init_ready(
        ProbeKind::Validation,
        "validation",
        _request.candidate_microbatch,
        model_ms,
        0,
        loss_ms,
    )?;

    execute_shard_validation_probe(
        config,
        _request,
        reader.sample_count(),
        Instant::now(),
        || {
            run_validation(
                &model,
                ValidationContext {
                    config,
                    loader: &validation_loader(&StreamingLoaderConfig {
                        buffer_games: config.buffer_games,
                        buffer_samples: config.buffer_samples,
                        train_fraction: config.train_fraction,
                        seed: config.seed,
                        archive_queue_bound: config.archive_queue_bound,
                        max_skip_logs_per_source: config.max_skip_logs_per_source,
                        aggregate_skip_logs: true,
                        source_filters: config.source_filters.clone(),
                        replay_target_profile: ReplayTargetProfile::minimal_bc(),
                        exit_sidecar: None,
                        exit_sidecar_source_net_hash: None,
                        exit_sidecar_source_version: None,
                        delta_q_sidecar: None,
                        delta_q_sidecar_source_net_hash: None,
                        delta_q_sidecar_source_version: None,
                        num_threads: config.num_threads,
                    }),
                    manifest: &DataManifest {
                        sources: Vec::new(),
                        total_games: 0,
                        train_count: 0,
                        val_count: 0,
                        counts_exact: true,
                    },
                    cached_samples: None,
                    device: train_device,
                    loss_fn: &loss_fn,
                    exit_cfg: &build_bc_exit_config(config.advanced_loss.as_ref()),
                },
                ValidationRuntime {
                    head_controller: None,
                    progress: None,
                },
            )
        },
    )
}

fn run_rl_probe_only(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    let result = run_rl_probe_only_result(config, preflight, request)?;
    write_probe_result(result_path, &result)
}

fn build_probe_success_result(
    request: ProbeRequest,
    measured_samples_per_second: f64,
    elapsed_seconds: f64,
    detail: String,
) -> ProbeResult {
    ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status: ProbeStatus::Success,
        measured_samples_per_second: Some(measured_samples_per_second),
        elapsed_seconds: Some(elapsed_seconds),
        detail,
    }
}

fn configure_probe_threads(config: &TrainConfig) -> Result<(), String> {
    configure_threads(config.num_threads)
        .map_err(|err| format!("failed to configure rayon threads for probe child: {err}"))
}

fn run_probe_attempt_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    if matches!(request.kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        run_rl_probe_only_result(config, preflight, request)
    } else {
        run_probe_only_with_model_config_result(config, model_config, manifest, request)
    }
}

struct ProbeDataContext {
    manifest: Option<DataManifest>,
    train_reader: Option<ExtractedBcShardReader>,
    validation_reader: Option<ExtractedBcShardReader>,
}

impl ProbeDataContext {
    fn manifest_ref(&self) -> Option<&DataManifest> {
        self.manifest.as_ref()
    }

    fn train_reader_ref(&self) -> Option<&ExtractedBcShardReader> {
        self.train_reader.as_ref()
    }

    fn validation_reader_ref(&self) -> Option<&ExtractedBcShardReader> {
        self.validation_reader.as_ref()
    }
}

fn resolve_probe_data_context(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<ProbeDataContext, String> {
    if matches!(kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return Ok(ProbeDataContext {
            manifest: None,
            train_reader: None,
            validation_reader: None,
        });
    }

    if config.bc_shards_manifest_path.is_some()
        && matches!(kind, ProbeKind::Train | ProbeKind::Validation)
    {
        let shard_manifest_path = config
            .bc_shards_manifest_path
            .as_ref()
            .ok_or_else(|| "bc_shards_manifest_path missing for shard probe".to_string())?;
        let train_reader = if matches!(kind, ProbeKind::Train) {
            Some(load_extracted_bc_shard_reader(
                shard_manifest_path,
                ExtractedBcShardSplit::Train,
            )?)
        } else {
            None
        };
        let validation_reader = if matches!(kind, ProbeKind::Validation) {
            Some(load_extracted_bc_shard_reader(
                shard_manifest_path,
                ExtractedBcShardSplit::Validation,
            )?)
        } else {
            None
        };
        return Ok(ProbeDataContext {
            manifest: None,
            train_reader,
            validation_reader,
        });
    }

    Ok(ProbeDataContext {
        manifest: load_probe_batch_manifest(
            config,
            kind,
            manifest_cache_path,
            discovery_summary_path,
            discovery_index_path,
        )?,
        train_reader: None,
        validation_reader: None,
    })
}

fn run_probe_attempt_with_data_context(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    context: &ProbeDataContext,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    if context.train_reader.is_some() || context.validation_reader.is_some() {
        run_probe_attempt_with_shard_readers_result(
            config,
            preflight,
            model_config,
            context.train_reader_ref(),
            context.validation_reader_ref(),
            request,
        )
    } else {
        run_probe_attempt_result(
            config,
            preflight,
            model_config,
            context.manifest_ref(),
            request,
        )
    }
}

fn run_probe_attempt_with_shard_readers_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    train_reader: Option<&ExtractedBcShardReader>,
    validation_reader: Option<&ExtractedBcShardReader>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    let train_device = train_device(&config.device)?;
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => match config.precision_mode {
            hydra_train_runtime::config::PrecisionMode::Fp32
            | hydra_train_runtime::config::PrecisionMode::Bf16Autocast => {
                probe_train_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    train_reader.ok_or_else(|| {
                        "train shard reader missing for shard train probe".to_string()
                    })?,
                )?
            }
        },
        ProbeKind::Validation => match config.precision_mode {
            hydra_train_runtime::config::PrecisionMode::Fp32
            | hydra_train_runtime::config::PrecisionMode::Bf16Autocast => {
                let reader = validation_reader.ok_or_else(|| {
                    "validation shard reader missing for shard validation probe".to_string()
                })?;
                probe_validation_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            }
        },
        ProbeKind::RlGames | ProbeKind::RlMicrobatch => {
            return run_rl_probe_only_result(config, preflight, request);
        }
    };
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        format!(
            "stable {} probe on shard dataset",
            probe_kind_name(request.kind)
        ),
    ))
}

fn load_probe_batch_manifest(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<Option<DataManifest>, String> {
    if config.bc_shards_manifest_path.is_some()
        && matches!(kind, ProbeKind::Train | ProbeKind::Validation)
    {
        return Ok(None);
    }
    if matches!(kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return Ok(None);
    }

    let paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
    let cache_path = manifest_cache_path.unwrap_or(&paths.manifest_cache_path);
    let discovery_summary_path = discovery_summary_path.unwrap_or(&paths.discovery_summary_path);
    let discovery_index_path = discovery_index_path.unwrap_or(&paths.discovery_index_path);
    let loaded = load_or_scan_manifest_cache_with_source(ManifestCacheRequest {
        cache_path,
        discovery_summary_path,
        discovery_index_path,
        data_dir: &config.data_dir,
        train_fraction: config.train_fraction,
        source_filters: &config.source_filters,
        progress: None,
        scan_error_context: "preflight data",
    })?;
    let source = loaded
        .hit_source
        .map(ManifestCacheHitSource::as_str)
        .unwrap_or("scan");
    emit_probe_progress(&format!(
        "probe_progress kind={} phase=manifest_load source={} discovery_summary_path={} discovery_index_path={} legacy_manifest_path={} sources={} total_games={} train_count={} val_count={} counts_exact={}",
        probe_kind_name(kind),
        source,
        discovery_summary_path.display(),
        discovery_index_path.display(),
        cache_path.display(),
        loaded.manifest.sources.len(),
        loaded.manifest.total_games,
        loaded.manifest.train_count,
        loaded.manifest.val_count,
        loaded.manifest.counts_exact,
    ))?;
    Ok(Some(loaded.manifest))
}

fn load_probe_child_manifest(
    config: &TrainConfig,
    kind: ProbeKind,
    manifest_cache_path: Option<&Path>,
    discovery_summary_path: Option<&Path>,
    discovery_index_path: Option<&Path>,
) -> Result<Option<DataManifest>, String> {
    load_probe_batch_manifest(
        config,
        kind,
        manifest_cache_path,
        discovery_summary_path,
        discovery_index_path,
    )
}

struct ProbeChildBatchRuntimePaths<'a> {
    results_path: &'a Path,
    manifest_cache_path: Option<&'a Path>,
    discovery_summary_path: Option<&'a Path>,
    discovery_index_path: Option<&'a Path>,
}

fn run_probe_child_batch_request_with_model_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    batch: ProbeBatchRequest,
    paths: ProbeChildBatchRuntimePaths<'_>,
    model_config: &HydraModelConfig,
) -> Result<ProbeBatchArtifact, String> {
    configure_probe_threads(config)?;
    std::fs::remove_file(paths.results_path).ok();
    let context = resolve_probe_data_context(
        config,
        batch.request.kind,
        paths.manifest_cache_path,
        paths.discovery_summary_path,
        paths.discovery_index_path,
    )?;
    let mut artifact = ProbeBatchArtifact::pending();

    for _attempt in 0..batch.attempts {
        let result = run_probe_attempt_with_data_context(
            config,
            preflight,
            model_config,
            &context,
            batch.request,
        )?;
        let passed = result.status == ProbeStatus::Success;
        artifact.push_result(result);
        write_probe_batch_artifact(paths.results_path, &artifact)?;
        if !passed {
            return Ok(artifact);
        }
    }

    artifact.mark_finished();
    write_probe_batch_artifact(paths.results_path, &artifact)?;
    Ok(artifact)
}

fn run_rl_probe_only_result(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    let train_device = train_device(&config.device)?;
    let rl = config
        .rl
        .as_ref()
        .ok_or_else(|| "RL probe requested without rl config block".to_string())?;
    let mut tuned = rl.clone();
    match request.kind {
        ProbeKind::RlGames => {
            tuned.games_per_batch = request.candidate_microbatch;
            if tuned.microbatch_size.is_none() {
                tuned.microbatch_size = Some(hydra_train_types::config::DEFAULT_RL_MICROBATCH_SIZE);
            }
        }
        ProbeKind::RlMicrobatch => {
            tuned.microbatch_size = Some(request.candidate_microbatch.max(1));
        }
        ProbeKind::Train | ProbeKind::Validation => {
            return Err("non-RL probe routed to RL probe handler".to_string());
        }
    }
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=rl_selfplay",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
    ))?;
    let started_at = Instant::now();
    let measured_samples_per_second = crate::runtime_autotune_shim::measure_rl_runtime_throughput(
        config,
        preflight,
        &tuned,
        &train_device,
    )?;
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=done throughput={:.2} samples/s elapsed={:.2}s",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        measured_samples_per_second,
        elapsed_seconds,
    ))?;
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        String::new(),
    ))
}

fn run_probe_only_with_model_config(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    configure_probe_threads(config)?;
    if matches!(request.kind, ProbeKind::RlGames | ProbeKind::RlMicrobatch) {
        return run_rl_probe_only(config, preflight, request, result_path);
    }
    let result = run_probe_only_with_model_config_result(config, model_config, manifest, request)?;
    write_probe_result(result_path, &result)
}

fn run_probe_only_with_model_config_result(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    manifest: Option<&DataManifest>,
    request: ProbeRequest,
) -> Result<ProbeResult, String> {
    debug_assert!(matches!(
        request.kind,
        ProbeKind::Train | ProbeKind::Validation
    ));

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
        source_filters: config.source_filters.clone(),
        replay_target_profile: ReplayTargetProfile::minimal_bc(),
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
        num_threads: config.num_threads,
    };
    let probe_context = if config.bc_shards_manifest_path.is_some() {
        resolve_probe_data_context(config, request.kind, None, None, None)?
    } else if let Some(manifest) = manifest.cloned() {
        ProbeDataContext {
            manifest: Some(manifest),
            train_reader: None,
            validation_reader: None,
        }
    } else {
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=scan_start data_dir={}",
            probe_kind_name(request.kind),
            request.candidate_microbatch,
            config.data_dir.display(),
        ))?;
        let paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
        let loaded = load_or_scan_manifest_cache_with_source(ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &config.data_dir,
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            progress: None,
            scan_error_context: "preflight data",
        })?;
        let source = loaded
            .hit_source
            .map(ManifestCacheHitSource::as_str)
            .unwrap_or("scan");
        let manifest = loaded.manifest;
        emit_probe_progress(&format!(
            "probe_progress kind={} candidate_mb={} phase=scan_complete source={} discovery_summary_path={} discovery_index_path={} sources={} total_games={} train_count={} val_count={} counts_exact={}",
            probe_kind_name(request.kind),
            request.candidate_microbatch,
            source,
            paths.discovery_summary_path.display(),
            paths.discovery_index_path.display(),
            manifest.sources.len(),
            manifest.total_games,
            manifest.train_count,
            manifest.val_count,
            manifest.counts_exact,
        ))?;
        ProbeDataContext {
            manifest: Some(manifest),
            train_reader: None,
            validation_reader: None,
        }
    };
    let train_device = train_device(&config.device)?;
    let started_at = Instant::now();
    let measured_samples_per_second = match request.kind {
        ProbeKind::Train => {
            if let Some(reader) = probe_context.train_reader_ref() {
                probe_train_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            } else {
                probe_train_candidate_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &loader_config,
                    probe_context
                        .manifest_ref()
                        .ok_or_else(|| "manifest missing for non-shard train probe".to_string())?,
                    &train_device,
                )?
            }
        }
        ProbeKind::Validation => {
            if let Some(reader) = probe_context.validation_reader_ref() {
                probe_validation_candidate_from_shards_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &train_device,
                    reader,
                )?
            } else {
                probe_validation_candidate_for_backend::<TrainBackend>(
                    config,
                    model_config,
                    request,
                    &loader_config,
                    probe_context.manifest_ref().ok_or_else(|| {
                        "manifest missing for non-shard validation probe".to_string()
                    })?,
                    &train_device,
                )?
            }
        }
        ProbeKind::RlGames | ProbeKind::RlMicrobatch => {
            unreachable!("RL probes handled by run_rl_probe_only")
        }
    };
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    emit_probe_progress(&format!(
        "probe_progress kind={} candidate_mb={} phase=done throughput={:.2} samples/s elapsed={:.2}s",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        measured_samples_per_second,
        elapsed_seconds,
    ))?;
    Ok(build_probe_success_result(
        request,
        measured_samples_per_second,
        elapsed_seconds,
        format!(
            "stable {} probe on real dataset",
            probe_kind_name(request.kind)
        ),
    ))
}

#[cfg(test)]
pub fn run_probe_only(
    config: &TrainConfig,
    preflight: &PreflightConfig,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<(), String> {
    run_probe_only_with_model_config(
        config,
        preflight,
        &HydraModelConfig::learner(),
        None,
        request,
        result_path,
    )
}

pub fn run_probe_child_mode(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
) -> Result<bool, String> {
    run_probe_child_mode_with_model_config(config, child, &HydraModelConfig::learner())
}

pub fn run_probe_child_mode_with_model_config(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
    model_config: &HydraModelConfig,
) -> Result<bool, String> {
    if let Some((batch, results_path, manifest_paths)) =
        probe_batch_child_request_from_cli(child.clone())?
    {
        let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
        run_probe_child_batch_request_with_model_config(
            config,
            &PreflightConfig::default(),
            batch,
            ProbeChildBatchRuntimePaths {
                results_path: &results_path,
                manifest_cache_path: manifest_cache_path.as_deref(),
                discovery_summary_path: discovery_summary_path.as_deref(),
                discovery_index_path: discovery_index_path.as_deref(),
            },
            model_config,
        )?;
        return Ok(true);
    }

    let Some((request, result_path, manifest_paths)) = probe_child_request_from_cli(child)? else {
        return Ok(false);
    };
    let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
    let manifest = load_probe_child_manifest(
        config,
        request.kind,
        manifest_cache_path.as_deref(),
        discovery_summary_path.as_deref(),
        discovery_index_path.as_deref(),
    )?;
    run_probe_only_with_model_config(
        config,
        &PreflightConfig::default(),
        model_config,
        manifest.as_ref(),
        request,
        &result_path,
    )?;
    Ok(true)
}

#[cfg(test)]
pub fn run_probe_child_batch_mode_result(
    config: &TrainConfig,
    child: Option<ProbeChildRequest>,
) -> Result<Option<ProbeBatchArtifact>, String> {
    let Some((batch, results_path, manifest_paths)) = probe_batch_child_request_from_cli(child)?
    else {
        return Ok(None);
    };
    let (manifest_cache_path, discovery_summary_path, discovery_index_path) = manifest_paths;
    let tiny = HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1);
    Ok(Some(run_probe_child_batch_request_with_model_config(
        config,
        &PreflightConfig::default(),
        batch,
        ProbeChildBatchRuntimePaths {
            results_path: &results_path,
            manifest_cache_path: manifest_cache_path.as_deref(),
            discovery_summary_path: discovery_summary_path.as_deref(),
            discovery_index_path: discovery_index_path.as_deref(),
        },
        &tiny,
    )?))
}

#[cfg(test)]
pub fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
) -> Result<ProbeResult, String> {
    let config = hydra_train_runtime::config::read_config(config_path)?;
    run_probe_only_with_model_config(
        &config,
        &PreflightConfig::default(),
        &HydraModelConfig::learner(),
        None,
        request,
        result_path,
    )?;
    crate::probe_transport::read_probe_result(result_path)
}

#[cfg(test)]
pub fn format_probe_attempt_message(
    kind: ProbeKind,
    candidate: usize,
    attempt: usize,
    total_attempts: usize,
) -> String {
    format!(
        "[preflight:{}] candidate_mb={} attempt {}/{}",
        probe_kind_name(kind),
        candidate,
        attempt,
        total_attempts.max(1)
    )
}

#[cfg(test)]
pub fn format_probe_result_summary(result: &ProbeResult) -> String {
    format_probe_status_line(result)
}

#[cfg(test)]
fn probe_candidate_ladder_with_local_executor(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    probe_candidate_ladder(config_path, config, preflight, artifacts, kind, candidates)
}

#[cfg(not(test))]
fn probe_candidate_ladder_with_local_executor(
    config_path: &Path,
    config: &TrainConfig,
    preflight: &PreflightConfig,
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidates: &[usize],
) -> Result<(usize, Vec<ProbeResult>), String> {
    probe_candidate_ladder(config_path, config, preflight, artifacts, kind, candidates)
}

pub fn run_probe_ladder_only(
    config_path: &Path,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    preflight: &PreflightConfig,
    request: ProbeRequest,
) -> Result<(usize, Vec<ProbeResult>), String> {
    let scan_pb = make_spinner("{spinner:.cyan} {msg}")?;
    scan_pb.set_message(format!(
        "scanning data for {} probe",
        probe_kind_name(request.kind)
    ));
    let paths = PreflightPaths::new(artifacts);
    let _ = load_or_scan_manifest_cache(
        ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &config.data_dir,
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            progress: Some(&scan_pb),
            scan_error_context: "preflight data",
        },
        |_| {},
    )?;
    scan_pb.finish_with_message(
        format!("scan complete for {} probe", probe_kind_name(request.kind))
            .green()
            .to_string(),
    );

    let candidates = probe_only_candidate_ladder(config, preflight, request);
    let selected = probe_candidate_ladder_with_local_executor(
        config_path,
        config,
        preflight,
        artifacts,
        request.kind,
        &candidates,
    )?;
    Ok(selected)
}

pub fn classify_probe_detail(detail: &str) -> ProbeStatus {
    let lowered = detail.to_ascii_lowercase();
    if lowered.contains("out of memory") || lowered.contains("oom") {
        ProbeStatus::Oom
    } else if lowered.contains("cuda") || lowered.contains("cudnn") || lowered.contains("libtorch")
    {
        ProbeStatus::BackendError
    } else if lowered.contains("data") || lowered.contains("collate") || lowered.contains("replay")
    {
        ProbeStatus::DataError
    } else {
        ProbeStatus::BackendError
    }
}

/// Benchmark-only BC preflight result.
struct LoaderOnlyBenchMetrics {
    samples_per_second: f64,
    mib_per_second: f64,
    p50_batch_ms: f64,
    p95_batch_ms: f64,
    producer_wait_ratio: f64,
    consumer_wait_ratio: f64,
    disk_wait_ratio: f64,
    gpu_input_wait_ratio: f64,
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f64 - 1.0) * percentile).ceil() as usize;
    values[idx.min(values.len() - 1)]
}

fn run_synthetic_preflight_bench_row(
    tuple: PreflightBenchTuple,
    preflight: &PreflightConfig,
) -> LoaderOnlyBenchMetrics {
    let warmup_batches = preflight.warmup_steps.max(1);
    let measure_batches = preflight.measure_steps.max(1) * preflight.required_successes.max(1);
    let total_batches = warmup_batches.saturating_add(measure_batches);
    let bytes_per_sample = BC_SHARD_HEADER_SIZE as f64;
    let mut batch_ms = Vec::with_capacity(measure_batches);
    let mut measured_samples = 0usize;
    let mut measured_bytes = 0.0f64;
    let mut state = tuple.batch_size as u64
        ^ ((tuple.ring_batches as u64) << 17)
        ^ ((tuple.loader_threads as u64) << 33)
        ^ ((tuple.prefetch_batches as u64) << 49);
    let mut measured_window_started = None;

    for batch_index in 0..total_batches {
        let should_measure = batch_index >= warmup_batches;
        if should_measure && measured_window_started.is_none() {
            measured_window_started = Some(Instant::now());
        }
        let batch_started = Instant::now();
        let synthetic_iters = (tuple.batch_size / tuple.loader_threads.max(1)).max(1);
        for lane in 0..synthetic_iters {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add((batch_index as u64) ^ (lane as u64));
        }
        std::hint::black_box(state);
        if should_measure {
            batch_ms.push(batch_started.elapsed().as_secs_f64() * 1000.0);
            measured_samples = measured_samples.saturating_add(tuple.batch_size);
            measured_bytes += tuple.batch_size as f64 * bytes_per_sample;
        }
    }

    let elapsed_seconds = measured_window_started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0)
        .max(f64::EPSILON);
    batch_ms.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    LoaderOnlyBenchMetrics {
        samples_per_second: measured_samples as f64 / elapsed_seconds,
        mib_per_second: measured_bytes / (1024.0 * 1024.0) / elapsed_seconds,
        p50_batch_ms: percentile_sorted(&batch_ms, 0.50),
        p95_batch_ms: percentile_sorted(&batch_ms, 0.95),
        producer_wait_ratio: 0.0,
        consumer_wait_ratio: 0.0,
        disk_wait_ratio: 0.0,
        gpu_input_wait_ratio: 0.0,
    }
}

#[derive(Debug, Clone)]
pub struct PreflightBenchRuntime {
    pub report: PreflightBenchReport,
}

/// Runs benchmark-only preflight rows without cache lookup, cache write, or runtime selection.
pub fn run_preflight_bench(
    _config: &TrainConfig,
    preflight: &PreflightConfig,
    device_label: &str,
) -> Result<PreflightBenchRuntime, String> {
    let started = Instant::now();
    let mut rows = Vec::with_capacity(preflight.bench_candidate_tuples.len());
    for (index, tuple) in preflight.bench_candidate_tuples.iter().copied().enumerate() {
        let error = if tuple.batch_size == 0 {
            Some("batch must be greater than 0".to_string())
        } else if tuple.ring_batches == 0 {
            Some("ring must be greater than 0".to_string())
        } else if tuple.loader_threads == 0 {
            Some("threads must be greater than 0".to_string())
        } else if tuple.prefetch_batches == 0 {
            Some("prefetch must be greater than 0".to_string())
        } else {
            None
        };
        let metrics = if error.is_none() {
            Some(run_synthetic_preflight_bench_row(tuple, preflight))
        } else {
            None
        };
        let status = if error.is_some() {
            PreflightBenchStatus::Error
        } else {
            PreflightBenchStatus::Pass
        };
        rows.push(PreflightBenchRow {
            index,
            status,
            device: device_label.to_string(),
            mode: PreflightBenchMode::LoaderOnly,
            batch_size: tuple.batch_size,
            ring_batches: tuple.ring_batches,
            loader_threads: tuple.loader_threads,
            prefetch_batches: tuple.prefetch_batches,
            shuffle: PreflightShuffleMode::None,
            codec: PreflightCodec::None,
            samples_per_second: metrics.as_ref().map(|metrics| metrics.samples_per_second),
            mib_per_second: metrics.as_ref().map(|metrics| metrics.mib_per_second),
            p50_batch_ms: metrics.as_ref().map(|metrics| metrics.p50_batch_ms),
            p95_batch_ms: metrics.as_ref().map(|metrics| metrics.p95_batch_ms),
            producer_wait_ratio: metrics.as_ref().map(|metrics| metrics.producer_wait_ratio),
            consumer_wait_ratio: metrics.as_ref().map(|metrics| metrics.consumer_wait_ratio),
            disk_wait_ratio: metrics.as_ref().map(|metrics| metrics.disk_wait_ratio),
            gpu_input_wait_ratio: metrics.as_ref().map(|metrics| metrics.gpu_input_wait_ratio),
            cpu_user_seconds: None,
            cpu_system_seconds: None,
            error,
        });
    }
    Ok(PreflightBenchRuntime {
        report: PreflightBenchReport {
            schema_version: 1,
            rows,
            total_elapsed_seconds: started.elapsed().as_secs_f64(),
        },
    })
}

#[cfg(test)]
mod tests;
