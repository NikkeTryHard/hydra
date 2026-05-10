use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[cfg(feature = "cuda-graph")]
use crate::bootstrap::{TrainBackend, initialize_training_bootstrap};
#[cfg(feature = "cuda-graph")]
use crate::data_pipeline::stream_train_epoch;
#[cfg(feature = "cuda-graph")]
use crate::epoch_runner::{self, TrainLogicalBatchConfig, train_logical_batch};
#[cfg(feature = "cuda-graph")]
use hydra_bc_shards::{BcShardSplit, load_bc_shard_reader};
#[cfg(feature = "cuda-graph")]
use hydra_model::model::HydraModel;
#[cfg(feature = "cuda-graph")]
use hydra_train_runtime::schedule::{TrainerScheduleConfig, effective_lr};

const GRAPH_PROBE_SCHEMA_VERSION: u32 = 2;
const GRAPH_PROBE_MODE: &str = "compute_capture_only";
const GRAPH_PROBE_PRODUCTION_REPLAY: &str = "blocked";
const GRAPH_PROBE_TRAINING_SEMANTICS: &str = "unchanged_probe_child_only";
const GRAPH_PROBE_BLOCKER_BURN_GRADS: &str = "Burn GradientsParams prevents production replay";
#[cfg(feature = "cuda-graph")]
const GRAPH_PROBE_REPLAY_SCOPE: &str = "prematerialized_compute_only";

fn graph_probe_blockers() -> Vec<&'static str> {
    vec![GRAPH_PROBE_BLOCKER_BURN_GRADS]
}

/// Serialized CUDA graph probe report emitted by parent and child modes.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CudaGraphProbeReport {
    /// Event discriminator used by the parent to locate the child report.
    pub event: &'static str,
    /// Success/failure status.
    pub status: String,
    /// Probe stage that produced the report.
    pub stage: String,
    /// Human-readable result detail.
    pub message: String,
    /// Wall-clock elapsed seconds.
    pub elapsed_seconds: f64,
    /// Report schema version.
    pub schema_version: u32,
    /// Probe execution mode.
    pub probe_mode: &'static str,
    /// Production replay status.
    pub production_replay: &'static str,
    /// Training-semantics status.
    pub training_semantics: &'static str,
    /// Known blockers preventing production graph replay.
    pub blockers: Vec<&'static str>,
    /// Warmup timing summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<CudaGraphWarmupSummary>,
    /// Static parity summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<CudaGraphParitySummary>,
    /// Capture/replay attempt summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<CudaGraphCaptureSummary>,
}

/// Timing summary for the warmup train step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaGraphWarmupSummary {
    /// Input source used by the probe.
    pub input: &'static str,
    /// Logical batch size.
    pub batch_size: usize,
    /// Microbatch size.
    pub microbatch_size: usize,
    /// Number of stats records produced.
    pub stats_count: usize,
    /// Host-to-device seconds.
    pub h2d_seconds: f64,
    /// Forward seconds.
    pub forward_seconds: f64,
    /// Loss seconds.
    pub loss_seconds: f64,
    /// Backward seconds.
    pub backward_seconds: f64,
    /// Optimizer-step seconds.
    pub optimizer_step_seconds: f64,
}

/// Static parity summary for repeated non-mutating probe steps.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaGraphParitySummary {
    /// Number of parity repeats.
    pub repeats: usize,
    /// Maximum total-loss absolute difference.
    pub max_total_loss_abs_diff: f64,
    /// Maximum policy-agreement absolute difference.
    pub max_policy_agreement_abs_diff: f64,
    /// Reference total loss.
    pub reference_total_loss: f64,
    /// Reference policy agreement.
    pub reference_policy_agreement: f64,
}

/// Capture/replay attempt summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaGraphCaptureSummary {
    /// Whether capture was attempted.
    pub attempted: bool,
    /// Success/failure/panic status.
    pub status: String,
    /// Probe stage.
    pub stage: String,
    /// Human-readable detail.
    pub message: String,
    /// Replay scope guarded by this probe.
    pub replay_scope: &'static str,
    /// Production replay status.
    pub production_replay: &'static str,
    /// Known production replay blocker.
    pub blocker: &'static str,
    /// Capture seconds when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_seconds: Option<f64>,
    /// Replay repeat count when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_repeats: Option<usize>,
    /// Total replay seconds when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_total_seconds: Option<f64>,
    /// Replay seconds per step when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_seconds_per_step: Option<f64>,
    /// Optional post-replay total-loss absolute difference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_replay_total_loss_abs_diff: Option<f64>,
    /// Optional post-replay policy-agreement absolute difference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_replay_policy_agreement_abs_diff: Option<f64>,
    /// Whether post-replay parity was enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_replay_parity_enabled: Option<bool>,
}

impl CudaGraphProbeReport {
    fn failure(stage: impl Into<String>, message: impl Into<String>, elapsed_seconds: f64) -> Self {
        Self {
            event: "cuda_graph_probe",
            status: "failure".to_string(),
            stage: stage.into(),
            message: message.into(),
            elapsed_seconds,
            schema_version: GRAPH_PROBE_SCHEMA_VERSION,
            probe_mode: GRAPH_PROBE_MODE,
            production_replay: GRAPH_PROBE_PRODUCTION_REPLAY,
            training_semantics: GRAPH_PROBE_TRAINING_SEMANTICS,
            blockers: graph_probe_blockers(),
            warmup: None,
            parity: None,
            capture: None,
        }
    }

    #[cfg(feature = "cuda-graph")]
    fn success(
        stage: impl Into<String>,
        message: impl Into<String>,
        elapsed_seconds: f64,
        warmup: CudaGraphWarmupSummary,
        parity: CudaGraphParitySummary,
        capture: CudaGraphCaptureSummary,
    ) -> Self {
        Self {
            event: "cuda_graph_probe",
            status: "success".to_string(),
            stage: stage.into(),
            message: message.into(),
            elapsed_seconds,
            schema_version: GRAPH_PROBE_SCHEMA_VERSION,
            probe_mode: GRAPH_PROBE_MODE,
            production_replay: GRAPH_PROBE_PRODUCTION_REPLAY,
            training_semantics: GRAPH_PROBE_TRAINING_SEMANTICS,
            blockers: graph_probe_blockers(),
            warmup: Some(warmup),
            parity: Some(parity),
            capture: Some(capture),
        }
    }
}

/// Runs the CUDA graph probe parent process, spawning a quiet child.
pub fn handle_graph_probe_parent(config_path: &Path) -> Result<(), String> {
    let started = Instant::now();
    let exe = current_executable()?;
    let output = Command::new(exe)
        .arg(config_path)
        .env("HYDRA_CUDA_GRAPH_PROBE_CHILD", "1")
        .env("HYDRA_BENCHMARK_QUIET", "1")
        .output()
        .map_err(|err| format!("failed to spawn CUDA graph probe child: {err}"))?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = stdout.lines().find(|line| {
            line.trim_start()
                .starts_with("{\"event\":\"cuda_graph_probe\"")
        }) {
            println!("{line}");
            return Ok(());
        }
        let report = CudaGraphProbeReport::failure(
            "child_output",
            "CUDA graph probe child exited successfully without report",
            started.elapsed().as_secs_f64(),
        );
        print_report(&report)?;
        return Err(report.message);
    }
    let report = CudaGraphProbeReport::failure(
        "child_exit",
        summarize_child_failure(&output),
        started.elapsed().as_secs_f64(),
    );
    print_report(&report)?;
    Err(format!("CUDA graph probe child failed: {}", report.message))
}

/// Runs the CUDA graph probe child process.
#[cfg(feature = "cuda-graph")]
pub fn handle_graph_probe_child(config_path: &Path) -> Result<(), String> {
    let started = Instant::now();
    match run_probe_step(config_path) {
        Ok((warmup, parity, capture)) => {
            let report = CudaGraphProbeReport::success(
                "capture_probe",
                "child-process CUDA graph probe completed warmup, static parity, and guarded compute capture attempt",
                started.elapsed().as_secs_f64(),
                warmup,
                parity,
                capture,
            );
            print_report(&report)?;
            Ok(())
        }
        Err(err) => {
            let report =
                CudaGraphProbeReport::failure("warmup", err, started.elapsed().as_secs_f64());
            print_report(&report)?;
            Err(report.message)
        }
    }
}

/// Reports that CUDA graph probe child mode requires the cuda-graph feature.
#[cfg(not(feature = "cuda-graph"))]
pub fn handle_graph_probe_child(_config_path: &Path) -> Result<(), String> {
    let report = CudaGraphProbeReport::failure(
        "build",
        "CUDA graph probe child requires --features cuda-graph",
        0.0,
    );
    print_report(&report)?;
    Err(report.message)
}

#[cfg(feature = "cuda-graph")]
fn run_probe_step(
    config_path: &Path,
) -> Result<
    (
        CudaGraphWarmupSummary,
        CudaGraphParitySummary,
        CudaGraphCaptureSummary,
    ),
    String,
> {
    let config = hydra_train_runtime::config::read_config(config_path)?;
    let (bootstrap, mut runtime, _readers) = initialize_training_bootstrap(config_path, config)?;
    let batch_size = bootstrap.config.batch_size;
    let microbatch_size = bootstrap.microbatch_size;
    let lr = effective_lr(
        TrainerScheduleConfig::new(
            bootstrap.train_cfg.lr,
            bootstrap.train_cfg.min_learning_rate,
            bootstrap.train_cfg.warmup_steps,
        ),
        runtime.global_step,
        bootstrap.total_steps,
    );
    let mut model_slot = Some(runtime.model);

    if let Some(manifest_path) = bootstrap.config.bc_shards_manifest_path.as_ref() {
        let reader = load_bc_shard_reader(manifest_path, BcShardSplit::Train)?;
        let take = batch_size.min(reader.sample_count());
        if take == 0 {
            return Err("CUDA graph warmup requires at least one training sample".to_string());
        }
        let warmup_host_batch =
            reader.collate_host_batch_range(0, take, bootstrap.config.augment)?;
        let parity_host_batch_a =
            reader.collate_host_batch_range(0, take, bootstrap.config.augment)?;
        let parity_host_batch_b =
            reader.collate_host_batch_range(0, take, bootstrap.config.augment)?;
        let capture_host_batch =
            reader.collate_host_batch_range(0, take, bootstrap.config.augment)?;
        let post_replay_host_batch =
            reader.collate_host_batch_range(0, take, bootstrap.config.augment)?;
        let t_materialize = Instant::now();
        let warmup_device_batch = epoch_runner::materialize_host_batch_owned::<TrainBackend>(
            warmup_host_batch,
            &bootstrap.train_device,
        );
        let mut timing = hydra_train_runtime::progress::TrainSubStageTiming::default();
        timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
        timing.h2d_transfer_seconds += timing.h2d_tensor_materialize_seconds;
        let (stats, timing) = epoch_runner::train_device_batch(
            warmup_device_batch,
            take,
            timing,
            TrainLogicalBatchConfig {
                microbatch_size,
                use_amp: bootstrap.use_amp,
                augment: bootstrap.config.augment,
                train_device: &bootstrap.train_device,
                loss_fn: &bootstrap.loss_fn,
                bc_exit_cfg: &bootstrap.bc_exit_cfg,
                lr,
            },
            &mut runtime.head_controller,
            &mut model_slot,
            &mut runtime.optimizer,
        )?;
        runtime.model = take_model(model_slot)?;
        let mut parity_controller = runtime.head_controller.clone();
        let (parity_stats_a, _timing_a) = epoch_runner::probe_logical_batch_from_host_batch(
            parity_host_batch_a,
            TrainLogicalBatchConfig {
                microbatch_size,
                use_amp: bootstrap.use_amp,
                augment: bootstrap.config.augment,
                train_device: &bootstrap.train_device,
                loss_fn: &bootstrap.loss_fn,
                bc_exit_cfg: &bootstrap.bc_exit_cfg,
                lr,
            },
            &mut parity_controller,
            &runtime.model,
            None,
        )?;
        let mut parity_controller = runtime.head_controller.clone();
        let (parity_stats_b, _timing_b) = epoch_runner::probe_logical_batch_from_host_batch(
            parity_host_batch_b,
            TrainLogicalBatchConfig {
                microbatch_size,
                use_amp: bootstrap.use_amp,
                augment: bootstrap.config.augment,
                train_device: &bootstrap.train_device,
                loss_fn: &bootstrap.loss_fn,
                bc_exit_cfg: &bootstrap.bc_exit_cfg,
                lr,
            },
            &mut parity_controller,
            &runtime.model,
            None,
        )?;
        let parity = parity_summary(&parity_stats_a, &parity_stats_b)?;
        let warmup = CudaGraphWarmupSummary::from_timing(
            "bc_shards",
            take,
            microbatch_size,
            stats.len(),
            timing,
        );
        let capture = guarded_capture_attempt(
            capture_host_batch,
            Some(post_replay_host_batch),
            microbatch_size,
            bootstrap.use_amp,
            bootstrap.config.augment,
            &bootstrap.train_device,
            &bootstrap.loss_fn,
            &bootstrap.bc_exit_cfg,
            lr,
            &mut runtime.head_controller.clone(),
            &runtime.model,
            &parity_stats_a,
        );
        return Ok((warmup, parity, capture));
    }

    let mut pending = Vec::new();
    for chunk in stream_train_epoch(&bootstrap.manifest, &bootstrap.loader_config, 0, None) {
        pending.extend(chunk.map_err(|err| format!("CUDA graph warmup data load failed: {err}"))?);
        if pending.len() >= batch_size {
            pending.truncate(batch_size);
            break;
        }
    }
    if pending.is_empty() {
        return Err("CUDA graph warmup requires at least one training sample".to_string());
    }
    let (stats, timing) = train_logical_batch(
        &pending,
        TrainLogicalBatchConfig {
            microbatch_size,
            use_amp: bootstrap.use_amp,
            augment: bootstrap.config.augment,
            train_device: &bootstrap.train_device,
            loss_fn: &bootstrap.loss_fn,
            bc_exit_cfg: &bootstrap.bc_exit_cfg,
            lr,
        },
        &mut runtime.head_controller,
        &mut model_slot,
        &mut runtime.optimizer,
    )?;
    runtime.model = take_model(model_slot)?;
    let warmup = CudaGraphWarmupSummary::from_timing(
        "raw_replay",
        pending.len(),
        microbatch_size,
        stats.len(),
        timing,
    );
    Err(format!(
        "raw replay warmup succeeded (stats_count={}), but static parity currently requires bc_shards so the exact host batch can be replayed without reloading/reordering samples",
        warmup.stats_count
    ))
}

#[cfg(feature = "cuda-graph")]
fn take_model(
    model_slot: Option<HydraModel<TrainBackend>>,
) -> Result<HydraModel<TrainBackend>, String> {
    model_slot.ok_or_else(|| "CUDA graph warmup model slot was unexpectedly empty".to_string())
}

#[cfg(feature = "cuda-graph")]
#[allow(
    clippy::too_many_arguments,
    reason = "probe mirrors train step contract"
)]
fn guarded_capture_attempt(
    host_batch: hydra_bc_shards::BcShardHostBatch,
    post_replay_host_batch: Option<hydra_bc_shards::BcShardHostBatch>,
    microbatch_size: usize,
    use_amp: bool,
    augment: bool,
    train_device: &burn::backend::libtorch::LibTorchDevice,
    loss_fn: &hydra_train_runtime::losses::HydraLoss<TrainBackend>,
    bc_exit_cfg: &hydra_train_runtime::bc_runtime::BcExitConfig,
    lr: f64,
    head_controller: &mut hydra_train_runtime::head_gates::HeadActivationController,
    model: &HydraModel<TrainBackend>,
    reference_stats: &[hydra_train_runtime::progress::BatchStats],
) -> CudaGraphCaptureSummary {
    let shard_batch =
        epoch_runner::materialize_host_batch_owned::<TrainBackend>(host_batch, train_device);
    let graph = crate::cuda_graph::CudaGraph::new(false);
    let previous_stream = crate::cuda_graph::CudaStream::current(0);
    let capture_stream = crate::cuda_graph::CudaStream::from_pool(0);
    capture_stream.set_current();
    let current_stream = crate::cuda_graph::CudaStream::current(0);
    current_stream.synchronize();
    if let Err(err) = crate::cuda_graph::synchronize_device() {
        previous_stream.set_current();
        return capture_summary_failure("pre_capture_sync", err);
    }
    if let Err(err) = graph.try_capture_begin((0, 0)) {
        previous_stream.set_current();
        return capture_summary_failure("capture_begin", err);
    }
    let capture_started = Instant::now();
    let step_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        epoch_runner::probe_device_batch_compute_no_stats(
            shard_batch,
            microbatch_size,
            use_amp,
            loss_fn,
            bc_exit_cfg,
            head_controller,
            model,
        )
    }));
    let capture = match step_result {
        Ok(Ok(_timing)) => match graph.try_capture_end() {
            Ok(()) => {
                let capture_seconds = capture_started.elapsed().as_secs_f64();
                let replay_repeats = graph_probe_replay_repeats();
                let post_replay_parity_enabled = graph_probe_post_replay_parity_enabled();
                if let Err(err) = crate::cuda_graph::synchronize_device() {
                    return capture_summary_failure("pre_replay_sync", err);
                }
                let replay_started = Instant::now();
                let mut replay_error = None;
                for _ in 0..replay_repeats {
                    if let Err(err) = graph.try_replay() {
                        replay_error = Some(err);
                        break;
                    }
                }
                if replay_error.is_none()
                    && let Err(err) = crate::cuda_graph::synchronize_device()
                {
                    replay_error = Some(err);
                }
                let replay_total_seconds = replay_started.elapsed().as_secs_f64();
                match replay_error {
                    None => {
                        let (post_loss_diff, post_agreement_diff) = if post_replay_parity_enabled {
                            if let Some(host_batch) = post_replay_host_batch {
                                let mut replay_controller = head_controller.clone();
                                match epoch_runner::probe_logical_batch_from_host_batch(
                                    host_batch,
                                    TrainLogicalBatchConfig {
                                        microbatch_size,
                                        use_amp,
                                        augment,
                                        train_device,
                                        loss_fn,
                                        bc_exit_cfg,
                                        lr,
                                    },
                                    &mut replay_controller,
                                    model,
                                    None,
                                ) {
                                    Ok((stats, _)) => post_replay_diff(reference_stats, &stats),
                                    Err(_) => (None, None),
                                }
                            } else {
                                (None, None)
                            }
                        } else {
                            (None, None)
                        };
                        CudaGraphCaptureSummary {
                            attempted: true,
                            status: "success".to_string(),
                            stage: "replay".to_string(),
                            message:
                                "prematerialized compute-only capture and repeated replay completed"
                                    .to_string(),
                            replay_scope: GRAPH_PROBE_REPLAY_SCOPE,
                            production_replay: GRAPH_PROBE_PRODUCTION_REPLAY,
                            blocker: GRAPH_PROBE_BLOCKER_BURN_GRADS,
                            capture_seconds: Some(capture_seconds),
                            replay_repeats: Some(replay_repeats),
                            replay_total_seconds: Some(replay_total_seconds),
                            replay_seconds_per_step: Some(
                                replay_total_seconds / replay_repeats as f64,
                            ),
                            post_replay_total_loss_abs_diff: post_loss_diff,
                            post_replay_policy_agreement_abs_diff: post_agreement_diff,
                            post_replay_parity_enabled: Some(post_replay_parity_enabled),
                        }
                    }
                    Some(err) => capture_summary_failure("replay", err),
                }
            }
            Err(err) => capture_summary_failure("capture_end", err),
        },
        Ok(Err(err)) => capture_summary_failure("compute", err),
        Err(payload) => CudaGraphCaptureSummary {
            attempted: true,
            status: "panic".to_string(),
            stage: "compute".to_string(),
            message: panic_message(payload),
            replay_scope: GRAPH_PROBE_REPLAY_SCOPE,
            production_replay: GRAPH_PROBE_PRODUCTION_REPLAY,
            blocker: GRAPH_PROBE_BLOCKER_BURN_GRADS,
            capture_seconds: None,
            replay_repeats: None,
            replay_total_seconds: None,
            replay_seconds_per_step: None,
            post_replay_total_loss_abs_diff: None,
            post_replay_policy_agreement_abs_diff: None,
            post_replay_parity_enabled: None,
        },
    };
    previous_stream.set_current();
    capture
}

#[cfg(feature = "cuda-graph")]
fn capture_summary_failure(stage: &str, message: String) -> CudaGraphCaptureSummary {
    CudaGraphCaptureSummary {
        attempted: true,
        status: "failure".to_string(),
        stage: stage.to_string(),
        message,
        replay_scope: GRAPH_PROBE_REPLAY_SCOPE,
        production_replay: GRAPH_PROBE_PRODUCTION_REPLAY,
        blocker: GRAPH_PROBE_BLOCKER_BURN_GRADS,
        capture_seconds: None,
        replay_repeats: None,
        replay_total_seconds: None,
        replay_seconds_per_step: None,
        post_replay_total_loss_abs_diff: None,
        post_replay_policy_agreement_abs_diff: None,
        post_replay_parity_enabled: None,
    }
}

#[cfg(feature = "cuda-graph")]
fn graph_probe_replay_repeats() -> usize {
    const DEFAULT_REPLAYS: usize = 16;
    const MAX_REPLAYS: usize = 1024;
    match env::var("HYDRA_CUDA_GRAPH_PROBE_REPLAYS") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .map(|value| value.min(MAX_REPLAYS))
            .unwrap_or(DEFAULT_REPLAYS),
        Err(_) => DEFAULT_REPLAYS,
    }
}

#[cfg(feature = "cuda-graph")]
fn graph_probe_post_replay_parity_enabled() -> bool {
    match env::var("HYDRA_CUDA_GRAPH_PROBE_POST_REPLAY_PARITY") {
        Ok(value) => !matches!(value.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"),
        Err(_) => true,
    }
}

#[cfg(feature = "cuda-graph")]
fn post_replay_diff(
    reference: &[hydra_train_runtime::progress::BatchStats],
    observed: &[hydra_train_runtime::progress::BatchStats],
) -> (Option<f64>, Option<f64>) {
    match (reference.first(), observed.first()) {
        (Some(a), Some(b)) => (
            Some((a.total_loss - b.total_loss).abs()),
            Some((a.policy_agreement - b.policy_agreement).abs()),
        ),
        _ => (None, None),
    }
}

#[cfg(feature = "cuda-graph")]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "panic during CUDA graph capture".to_string()
    }
}

#[cfg(feature = "cuda-graph")]
fn parity_summary(
    first: &[hydra_train_runtime::progress::BatchStats],
    second: &[hydra_train_runtime::progress::BatchStats],
) -> Result<CudaGraphParitySummary, String> {
    let a = first
        .first()
        .ok_or_else(|| "CUDA graph parity probe produced no first stats".to_string())?;
    let b = second
        .first()
        .ok_or_else(|| "CUDA graph parity probe produced no second stats".to_string())?;
    Ok(CudaGraphParitySummary {
        repeats: 2,
        max_total_loss_abs_diff: (a.total_loss - b.total_loss).abs(),
        max_policy_agreement_abs_diff: (a.policy_agreement - b.policy_agreement).abs(),
        reference_total_loss: a.total_loss,
        reference_policy_agreement: a.policy_agreement,
    })
}

impl CudaGraphWarmupSummary {
    #[cfg(feature = "cuda-graph")]
    fn from_timing(
        input: &'static str,
        batch_size: usize,
        microbatch_size: usize,
        stats_count: usize,
        timing: hydra_train_runtime::progress::TrainSubStageTiming,
    ) -> Self {
        Self {
            input,
            batch_size,
            microbatch_size,
            stats_count,
            h2d_seconds: timing.h2d_transfer_seconds,
            forward_seconds: timing.forward_seconds,
            loss_seconds: timing.loss_seconds,
            backward_seconds: timing.backward_seconds,
            optimizer_step_seconds: timing.optimizer_step_seconds,
        }
    }
}

fn current_executable() -> Result<PathBuf, String> {
    #[cfg(target_os = "linux")]
    {
        let proc_self = PathBuf::from("/proc/self/exe");
        if proc_self.exists() {
            return Ok(proc_self);
        }
    }
    env::current_exe().map_err(|err| format!("current_exe failed: {err}"))
}

fn print_report(report: &CudaGraphProbeReport) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string(report)
            .map_err(|err| format!("failed to serialize CUDA graph probe report: {err}"))?
    );
    Ok(())
}

fn summarize_child_failure(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let useful = stderr
        .lines()
        .chain(stdout.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with("warning:"))
        .filter(|line| !line.starts_with("Compiling "))
        .filter(|line| !line.starts_with("Finished `"))
        .take(4)
        .collect::<Vec<_>>()
        .join(" | ");
    if useful.is_empty() {
        format!("child exited with status {}", output.status)
    } else {
        format!("child exited with status {}: {useful}", output.status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn report_serializes_probe_event() {
        let report = CudaGraphProbeReport::failure("stage", "message", 0.25);
        let json = serde_json::to_value(&report).expect("report should serialize");
        assert_eq!(json["event"], "cuda_graph_probe");
        assert_eq!(json["status"], "failure");
        assert_eq!(json["stage"], "stage");
    }

    #[cfg(unix)]
    #[test]
    fn summarize_child_failure_keeps_useful_lines() {
        let output = std::process::Output {
            status: std::process::ExitStatus::from_raw(1 << 8),
            stdout: b"warning: noisy\nuseful stdout\n".to_vec(),
            stderr: b"thread panicked\nFinished `release`\nsecond\n".to_vec(),
        };
        let summary = summarize_child_failure(&output);
        assert!(summary.contains("thread panicked"));
        assert!(summary.contains("second"));
        assert!(summary.contains("useful stdout"));
        assert!(!summary.contains("warning: noisy"));
    }
}
