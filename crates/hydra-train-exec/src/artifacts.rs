//! Artifact path and log-only helpers shared across training execution seams.

use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::optim::Optimizer;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use hydra_data_core::{DataManifest, DataSource};
use hydra_sample_cache::{
    ParsedSampleCacheMetadata, is_parsed_sample_cache_file, read_parsed_sample_cache_metadata,
};
use hydra_train_runtime::model::HydraModel;
use hydra_train_runtime::preflight::{
    BenchmarkResult, EffectiveRuntimeConfig, ManifestCacheEntry, PreflightCacheEntry,
    PreflightCacheKey, default_cache_name, default_manifest_cache_name,
};
use hydra_train_types::checkpoint::CheckpointMeta;
use hydra_train_types::delta_q_promotion::{
    ArenaPromotionDecision, DeltaQArenaConfirmationRequest, DeltaQArenaReport,
    DeltaQPolicyTransferReport, DeltaQPolicyTransferResult, DeltaQPromotionRecommendation,
    DeltaQPromotionReport, DeltaQPromotionResult,
};
use tboard::EventWriter;

use crate::advisory::AdvisoryEvent;
use crate::progress::{
    EpochLogEntry, RareActionMetrics, RlStepLogEntry, ScalarAverages, StepLogEntry,
};
use crate::resume::current_timestamp_s;
use crate::resume::{
    BestValidation, EpochContinuation, RuntimeResumeContract, build_resume_state,
    read_resume_state, read_rl_resume_state, write_resume_state, write_rl_resume_state,
};
use crate::validation::{ValidationGateDecision, ValidationSummary};

/// BC artifact paths rooted below the configured output directory.
pub struct BcArtifactPaths {
    /// BC artifact root directory.
    pub root: PathBuf,
    /// TensorBoard root directory.
    pub tb_root: PathBuf,
    /// TensorBoard session directory.
    pub tb_session_dir: PathBuf,
    /// Latest model checkpoint base path.
    pub latest_model_base: PathBuf,
    /// Latest optimizer checkpoint base path.
    pub latest_optimizer_base: PathBuf,
    /// Best model checkpoint base path.
    pub best_model_base: PathBuf,
    /// Latest BC resume state path.
    pub latest_state_path: PathBuf,
    /// Epoch training log path.
    pub training_log_path: PathBuf,
    /// Step training log path.
    pub step_log_path: PathBuf,
    /// DeltaQ promotion artifact path.
    pub delta_q_promotion_path: PathBuf,
    /// Validation gate artifact path.
    pub validation_gate_path: PathBuf,
}

/// RL artifact paths rooted below the configured output directory.
pub struct RlArtifactPaths {
    /// RL artifact root directory.
    pub root: PathBuf,
    /// TensorBoard root directory.
    pub tb_root: PathBuf,
    /// TensorBoard session directory.
    pub tb_session_dir: PathBuf,
    /// Latest model checkpoint base path.
    pub latest_model_base: PathBuf,
    /// Latest optimizer checkpoint base path.
    pub latest_optimizer_base: PathBuf,
    /// Latest RL resume state path.
    pub latest_state_path: PathBuf,
    /// RL step log path.
    pub step_log_path: PathBuf,
}

/// BC preflight cache paths.
pub struct PreflightPaths {
    /// Runtime preflight cache path.
    pub cache_path: PathBuf,
    /// Manifest cache path.
    pub manifest_cache_path: PathBuf,
}

/// BC preflight benchmark artifact paths.
pub struct PreflightBenchmarkPaths {
    /// Preflight benchmark root directory.
    pub root: PathBuf,
}

/// Persisted preflight benchmark report.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PreflightBenchmarkReport {
    /// Cache key used for the benchmark.
    pub cache_key: PreflightCacheKey,
    /// Effective runtime config benchmarked.
    pub runtime: EffectiveRuntimeConfig,
    /// Benchmark result payload.
    pub benchmark: BenchmarkResult,
}
/// Persisted validation gate decision artifact.
#[derive(serde::Serialize)]
pub struct PersistedValidationGateArtifact<'a> {
    /// Validation or promotion scope associated with the gate.
    pub scope: &'a str,
    /// Step or epoch associated with the gate.
    pub step_or_epoch: usize,
    /// Gate decision payload.
    pub decision: &'a ValidationGateDecision,
    /// Number of validation samples.
    pub samples: usize,
    /// Validation policy loss.
    pub policy_loss: f64,
    /// Validation policy agreement.
    pub policy_agreement: f64,
    /// Best prior policy loss, if present.
    pub best_policy_loss: Option<f64>,
    /// Best prior policy agreement, if present.
    pub best_policy_agreement: Option<f64>,
}

/// Writes a validation gate artifact as pretty JSON using atomic sibling replacement.
pub fn write_validation_gate_artifact(
    path: &Path,
    artifact: &PersistedValidationGateArtifact<'_>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(artifact)
        .map_err(|err| format!("failed to serialize validation gate artifact: {err}"))?;
    atomic_write_text(path, &json, "validation gate artifact")
}
/// Persisted DeltaQ promotion artifact.
#[derive(serde::Serialize)]
pub struct PersistedDeltaQPromotionArtifact<'a> {
    /// Validation or promotion scope associated with the artifact.
    pub scope: &'a str,
    /// Step or epoch associated with the artifact.
    pub step_or_epoch: usize,
    /// Promotion recommendation before or after arena confirmation.
    pub recommendation: DeltaQPromotionRecommendation,
    /// Promotion pipeline stage that produced the artifact.
    pub stage: &'a str,
    /// Arena confirmation request, if arena gating was requested.
    pub arena_confirmation: Option<DeltaQArenaConfirmationRequest>,
    /// Arena confirmation decision, if arena gating ran.
    pub arena_decision: Option<ArenaPromotionDecision>,
    /// Arena confirmation report, if arena gating ran.
    pub arena_report: Option<&'a DeltaQArenaReport>,
    /// Offline DeltaQ promotion report.
    pub report: &'a DeltaQPromotionReport,
    /// Offline DeltaQ promotion result.
    pub result: &'a DeltaQPromotionResult,
    /// Policy-transfer report, if policy-transfer gating ran.
    pub policy_transfer: Option<&'a DeltaQPolicyTransferReport>,
    /// Policy-transfer result, if policy-transfer gating ran.
    pub policy_transfer_result: Option<&'a DeltaQPolicyTransferResult>,
}

/// Writes a DeltaQ promotion artifact as pretty JSON using atomic sibling replacement.
pub fn write_delta_q_promotion_artifact(
    path: &Path,
    artifact: &PersistedDeltaQPromotionArtifact<'_>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(artifact)
        .map_err(|err| format!("failed to serialize delta_q promotion artifact: {err}"))?;
    atomic_write_text(path, &json, "delta_q promotion artifact")
}

/// RL preflight cache paths.
pub struct RlPreflightPaths {
    /// Runtime preflight cache path.
    pub cache_path: PathBuf,
}

/// JSONL append writer type.
pub type JsonlAppender = fs::File;

/// Atomically writes text by writing a timestamped temporary sibling then renaming it.
pub fn atomic_write_text(path: &Path, contents: &str, label: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {label} dir {}: {err}", parent.display()))?;
    }
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("tmp");
    let tmp_path = path.with_extension(format!(
        "{extension}.tmp-{}-{}",
        std::process::id(),
        current_timestamp_s()
    ));
    fs::write(&tmp_path, contents).map_err(|err| {
        format!(
            "failed to write temporary {label} {}: {err}",
            tmp_path.display()
        )
    })?;
    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "failed to finalize {label} {} from {}: {err}",
            path.display(),
            tmp_path.display()
        )
    })
}

impl PreflightPaths {
    /// Builds BC preflight cache paths from BC artifact paths.
    #[must_use]
    pub fn new(artifacts: &BcArtifactPaths) -> Self {
        Self {
            cache_path: artifacts.root.join(default_cache_name()),
            manifest_cache_path: artifacts.root.join(default_manifest_cache_name()),
        }
    }
}

impl PreflightBenchmarkPaths {
    /// Builds BC preflight benchmark paths from BC artifact paths.
    #[must_use]
    pub fn new(artifacts: &BcArtifactPaths) -> Self {
        Self {
            root: artifacts.root.join("preflight_benchmark"),
        }
    }

    /// Creates the benchmark root directory.
    pub fn create_root_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|err| {
            format!(
                "failed to create preflight benchmark dir {}: {err}",
                self.root.display()
            )
        })
    }

    /// Returns a candidate benchmark directory path.
    #[must_use]
    pub fn candidate_dir(&self, candidate_index: usize) -> PathBuf {
        self.root.join(format!("candidate_{candidate_index:02}"))
    }

    /// Creates and returns a candidate benchmark directory path.
    pub fn create_candidate_dir(&self, candidate_index: usize) -> Result<PathBuf, String> {
        let path = self.candidate_dir(candidate_index);
        fs::create_dir_all(&path).map_err(|err| {
            format!(
                "failed to create preflight benchmark candidate dir {}: {err}",
                path.display()
            )
        })?;
        Ok(path)
    }

    /// Returns the benchmark report path.
    #[must_use]
    pub fn report_path(&self) -> PathBuf {
        self.root.join("report.json")
    }
}

impl RlPreflightPaths {
    /// Builds RL preflight cache paths from RL artifact paths.
    #[must_use]
    pub fn new(artifacts: &RlArtifactPaths) -> Self {
        Self {
            cache_path: artifacts.root.join(default_cache_name()),
        }
    }
}

impl BcArtifactPaths {
    /// Builds BC artifact paths from the output directory and resume step.
    #[must_use]
    pub fn new(output_dir: &Path, resume_global_step: usize) -> Self {
        let root = output_dir.join("bc");
        let tb_root = root.join("tb");
        let tb_session_dir = tb_root.join(format!(
            "run_g{:08}_{}",
            resume_global_step,
            current_timestamp_s()
        ));
        Self {
            latest_model_base: root.join("latest_model"),
            latest_optimizer_base: root.join("latest_optimizer"),
            best_model_base: root.join("best_model"),
            latest_state_path: root.join("latest_state.yaml"),
            training_log_path: root.join("training_log.jsonl"),
            step_log_path: root.join("step_log.jsonl"),
            delta_q_promotion_path: root.join("delta_q_promotion.json"),
            validation_gate_path: root.join("validation_gate.json"),
            root,
            tb_root,
            tb_session_dir,
        }
    }

    /// Creates the BC artifact root directory.
    pub fn create_root_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|err| {
            format!(
                "failed to create BC artifact dir {}: {err}",
                self.root.display()
            )
        })?;
        Ok(())
    }

    /// Creates BC TensorBoard directories.
    pub fn create_tensorboard_dirs(&self) -> Result<(), String> {
        for dir in [&self.tb_root, &self.tb_session_dir] {
            fs::create_dir_all(dir).map_err(|err| {
                format!("failed to create BC artifact dir {}: {err}", dir.display())
            })?;
        }
        Ok(())
    }
}

impl RlArtifactPaths {
    /// Builds RL artifact paths from the output directory and resume step.
    #[must_use]
    pub fn new(output_dir: &Path, resume_global_step: usize) -> Self {
        let root = output_dir.join("rl");
        let tb_root = root.join("tb");
        let tb_session_dir = tb_root.join(format!(
            "run_g{:08}_{}",
            resume_global_step,
            current_timestamp_s()
        ));
        Self {
            latest_model_base: root.join("latest_model"),
            latest_optimizer_base: root.join("latest_optimizer"),
            latest_state_path: root.join("latest_state.yaml"),
            step_log_path: root.join("step_log.jsonl"),
            root,
            tb_root,
            tb_session_dir,
        }
    }

    /// Creates the RL artifact root directory.
    pub fn create_root_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|err| {
            format!(
                "failed to create RL artifact dir {}: {err}",
                self.root.display()
            )
        })?;
        Ok(())
    }

    /// Creates RL TensorBoard directories.
    pub fn create_tensorboard_dirs(&self) -> Result<(), String> {
        for dir in [&self.tb_root, &self.tb_session_dir] {
            fs::create_dir_all(dir).map_err(|err| {
                format!("failed to create RL artifact dir {}: {err}", dir.display())
            })?;
        }
        Ok(())
    }
}

/// Opens a JSONL appender for a named log.
pub fn open_jsonl_appender(path: &Path, log_name: &str) -> Result<JsonlAppender, String> {
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {log_name} {}: {err}", path.display()))
}

/// Appends a serializable entry as one JSONL line and flushes the writer.
pub fn append_jsonl_entry<W, T>(
    writer: &mut W,
    entry: &T,
    target: &str,
    entry_name: &str,
) -> Result<(), String>
where
    W: Write,
    T: serde::Serialize,
{
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize {entry_name}: {err}"))?;
    writeln!(writer, "{line}").map_err(|err| format!("failed to append {target}: {err}"))?;
    writer
        .flush()
        .map_err(|err| format!("failed to flush {target}: {err}"))
}

/// Opens the BC epoch training log JSONL appender.
pub fn open_training_log_appender(path: &Path) -> Result<JsonlAppender, String> {
    open_jsonl_appender(path, "training log")
}

/// Opens the BC step training log JSONL appender.
pub fn open_step_log_appender(path: &Path) -> Result<JsonlAppender, String> {
    open_jsonl_appender(path, "step log")
}

/// Opens the RL step training log JSONL appender.
pub fn open_rl_step_log_appender(path: &Path) -> Result<JsonlAppender, String> {
    open_jsonl_appender(path, "RL step log")
}

/// Appends a BC epoch training log entry as one JSONL line.
pub fn append_training_log_to_writer<W, DeltaQPromotionSnapshot, Advisory>(
    writer: &mut W,
    entry: &EpochLogEntry<DeltaQPromotionSnapshot, Advisory>,
) -> Result<(), String>
where
    W: Write,
    DeltaQPromotionSnapshot: serde::Serialize,
    Advisory: serde::Serialize,
{
    append_jsonl_entry(writer, entry, "training log", "training log entry")
}

/// Appends a BC step training log entry as one JSONL line.
pub fn append_step_log_to_writer<W, DeltaQPromotionSnapshot, Advisory>(
    writer: &mut W,
    entry: &StepLogEntry<DeltaQPromotionSnapshot, Advisory>,
) -> Result<(), String>
where
    W: Write,
    DeltaQPromotionSnapshot: serde::Serialize,
    Advisory: serde::Serialize,
{
    append_jsonl_entry(writer, entry, "step log", "step log entry")
}

/// Appends a runtime advisory event as one JSONL line in the step log schema.
pub fn append_advisory_event_to_writer<W>(
    writer: &mut W,
    entry: &AdvisoryEvent<'_>,
) -> Result<(), String>
where
    W: Write,
{
    append_jsonl_entry(writer, entry, "step log", "runtime advisory event")
}

/// Writes BC train, validation, LR, best-validation, and DeltaQ promotion scalars to TensorBoard.
pub fn log_tensorboard<W: Write>(
    tb: &mut EventWriter<W>,
    epoch: usize,
    train: &ScalarAverages,
    val_summary: Option<&ValidationSummary>,
    lr: f64,
    best_validation: Option<BestValidation>,
) -> Result<(), String> {
    let step = epoch as i64;
    tb.write_scalar(step, "train/total_loss", train.total_loss as f32)
        .map_err(|err| format!("tensorboard write train/total_loss failed: {err}"))?;
    tb.write_scalar(
        step,
        "train/policy_agreement",
        train.policy_agreement as f32,
    )
    .map_err(|err| format!("tensorboard write train/policy_agreement failed: {err}"))?;
    write_rare_action_tensorboard(tb, step, "train", &train.rare_actions)?;
    if let Some(val_summary) = val_summary {
        tb.write_scalar(step, "val/policy_agreement", val_summary.agreement as f32)
            .map_err(|err| format!("tensorboard write val/policy_agreement failed: {err}"))?;
        tb.write_scalar(step, "val/policy_loss", val_summary.policy_loss as f32)
            .map_err(|err| format!("tensorboard write val/policy_loss failed: {err}"))?;
        tb.write_scalar(step, "val/total_loss", val_summary.total_loss as f32)
            .map_err(|err| format!("tensorboard write val/total_loss failed: {err}"))?;
        write_rare_action_tensorboard(tb, step, "val", &val_summary.rare_actions)?;
        if let Some(delta_q) = val_summary.delta_q_promotion_snapshot {
            tb.write_scalar(
                step,
                "val/delta_q_candidate_top1_agreement",
                delta_q.candidate_top1_agreement as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_candidate_top1_agreement failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_candidate_mean_regret",
                delta_q.candidate_mean_regret as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_candidate_mean_regret failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_baseline_mean_regret",
                delta_q.baseline_mean_regret as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_baseline_mean_regret failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_mean_decision_lift",
                delta_q.mean_decision_lift as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_mean_decision_lift failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_negative_lift_fraction",
                delta_q.negative_lift_fraction as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_negative_lift_fraction failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_regret_beats_baseline_rate",
                delta_q.regret_beats_baseline_rate as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_regret_beats_baseline_rate failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_top1_beats_baseline_rate",
                delta_q.top1_beats_baseline_rate as f32,
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_top1_beats_baseline_rate failed: {err}")
            })?;
            tb.write_scalar(
                step,
                "val/delta_q_offline_gate_passed",
                if delta_q.passed { 1.0 } else { 0.0 },
            )
            .map_err(|err| {
                format!("tensorboard write val/delta_q_offline_gate_passed failed: {err}")
            })?;
        }
    }

    tb.write_scalar(step, "lr", lr as f32)
        .map_err(|err| format!("tensorboard write lr failed: {err}"))?;
    if let Some(best_validation) = best_validation {
        tb.write_scalar(
            step,
            "val/best_policy_loss",
            best_validation.policy_loss as f32,
        )
        .map_err(|err| format!("tensorboard write val/best_policy_loss failed: {err}"))?;
        tb.write_scalar(
            step,
            "val/best_policy_agreement",
            best_validation.agreement as f32,
        )
        .map_err(|err| format!("tensorboard write val/best_policy_agreement failed: {err}"))?;
    }
    Ok(())
}

fn write_rare_action_tensorboard<W: Write>(
    tb: &mut EventWriter<W>,
    step: i64,
    split: &str,
    metrics: &RareActionMetrics,
) -> Result<(), String> {
    let buckets = [
        ("discard", metrics.discard),
        ("aka_discard", metrics.aka_discard),
        ("riichi", metrics.riichi),
        ("chi", metrics.chi),
        ("pon", metrics.pon),
        ("kan", metrics.kan),
        ("agari", metrics.agari),
        ("ryuukyoku", metrics.ryuukyoku),
        ("pass", metrics.pass),
    ];
    for (name, bucket) in buckets {
        tb.write_scalar(
            step,
            &format!("{split}/rare_action/{name}_count"),
            bucket.count as f32,
        )
        .map_err(|err| {
            format!("tensorboard write {split}/rare_action/{name}_count failed: {err}")
        })?;
        tb.write_scalar(
            step,
            &format!("{split}/rare_action/{name}_accuracy"),
            bucket.accuracy as f32,
        )
        .map_err(|err| {
            format!("tensorboard write {split}/rare_action/{name}_accuracy failed: {err}")
        })?;
    }
    Ok(())
}

/// Appends an RL step training log entry as one JSONL line.
pub fn append_rl_step_log_to_writer<W, Advisory>(
    writer: &mut W,
    entry: &RlStepLogEntry<Advisory>,
) -> Result<(), String>
where
    W: Write,
    Advisory: serde::Serialize,
{
    append_jsonl_entry(writer, entry, "RL step log", "RL step log entry")
}

/// Returns true when persisted checkpoint metadata is semantically identical to a candidate.
///
/// `timestamp` is intentionally ignored: it records write time, not checkpoint identity.
#[must_use]
pub fn checkpoint_meta_semantically_matches(
    existing: &CheckpointMeta,
    candidate: &CheckpointMeta,
) -> bool {
    existing.epoch == candidate.epoch
        && existing.train_loss == candidate.train_loss
        && existing.eval_agreement == candidate.eval_agreement
        && existing.eval_policy_loss == candidate.eval_policy_loss
        && existing.eval_total_loss == candidate.eval_total_loss
        && existing.num_blocks == candidate.num_blocks
        && existing.hidden_channels == candidate.hidden_channels
}

/// Writes checkpoint metadata next to a checkpoint base path, preserving existing semantic matches.
pub fn write_checkpoint_meta(base: &Path, meta: &CheckpointMeta) -> Result<(), String> {
    let meta_path = base.with_extension("meta.json");
    if let Ok(raw) = fs::read_to_string(&meta_path)
        && let Ok(existing) = serde_json::from_str::<CheckpointMeta>(&raw)
        && checkpoint_meta_semantically_matches(&existing, meta)
    {
        return Ok(());
    }
    let meta_json = serde_json::to_string_pretty(meta).map_err(|err| {
        format!(
            "failed to serialize checkpoint metadata for epoch {}: {err}",
            meta.epoch
        )
    })?;
    fs::write(&meta_path, meta_json).map_err(|err| {
        format!(
            "failed to write checkpoint metadata {}: {err}",
            meta_path.display()
        )
    })
}

/// Saves a model checkpoint payload at the provided base path.
pub fn save_model_payload<B: Backend>(model: &HydraModel<B>, base: &Path) -> Result<(), String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(base, &recorder)
        .map_err(|err| format!("failed to save checkpoint {}: {err}", base.display()))
}

/// Saves an optimizer checkpoint payload at the provided base path.
pub fn save_optimizer_payload<B, O>(optimizer: &O, base: &Path) -> Result<(), String>
where
    B: AutodiffBackend,
    O: Optimizer<HydraModel<B>, B>,
{
    let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    optimizer_recorder
        .record(optimizer.to_record(), base.to_path_buf())
        .map_err(|err| format!("failed to save optimizer state {}: {err}", base.display()))
}

/// Builds checkpoint metadata from train/eval metrics.
#[must_use]
pub fn checkpoint_meta(
    epoch: usize,
    loss: f64,
    eval_agreement: Option<f64>,
    eval_policy_loss: Option<f64>,
    eval_total_loss: Option<f64>,
) -> CheckpointMeta {
    CheckpointMeta::new(
        epoch as u32,
        loss,
        eval_agreement,
        eval_policy_loss,
        eval_total_loss,
    )
}

/// Checkpoint/resume state needed when saving the latest BC checkpoint.
pub struct LatestCheckpointState<'a> {
    /// Resume-state global step.
    pub global_step: usize,
    /// Latest train loss persisted into checkpoint metadata.
    pub train_loss: f64,
    /// Best validation snapshot to preserve in resume state.
    pub best_validation: Option<BestValidation>,
    /// Epoch continuation cursor to persist.
    pub continuation: &'a EpochContinuation,
    /// Runtime contract to persist.
    pub runtime: RuntimeResumeContract,
}

/// Writes the latest BC checkpoint payloads, metadata, and resume state.
pub fn save_latest_checkpoint_and_state<B, O>(
    artifacts: &BcArtifactPaths,
    model: &HydraModel<B>,
    optimizer: &O,
    state: LatestCheckpointState<'_>,
) -> Result<(), String>
where
    B: AutodiffBackend,
    O: Optimizer<HydraModel<B>, B>,
{
    let LatestCheckpointState {
        global_step,
        train_loss,
        best_validation,
        continuation,
        runtime,
    } = state;
    let skip_payload_write = latest_bc_payload_is_current(artifacts, global_step);
    if !skip_payload_write {
        save_model_payload(model, &artifacts.latest_model_base)?;
        save_optimizer_payload(optimizer, &artifacts.latest_optimizer_base)?;
    }
    let meta = checkpoint_meta(global_step, train_loss, None, None, None);
    write_checkpoint_meta(&artifacts.latest_model_base, &meta)?;
    let state = build_resume_state(
        continuation.next_epoch,
        continuation.skip_optimizer_steps_in_epoch,
        global_step,
        best_validation,
        runtime,
    );
    write_resume_state(&artifacts.latest_state_path, &state)
}

/// Saves a named BC checkpoint payload and metadata from validation metrics.
pub fn save_checkpoint<B: Backend>(
    model: &HydraModel<B>,
    base: &Path,
    epoch: usize,
    loss: f64,
    val_summary: Option<&ValidationSummary>,
) -> Result<(), String> {
    save_model_payload(model, base)?;
    let meta = checkpoint_meta(
        epoch,
        loss,
        val_summary.map(|summary| summary.agreement),
        val_summary.map(|summary| summary.policy_loss),
        val_summary.map(|summary| summary.total_loss),
    );
    write_checkpoint_meta(base, &meta)
}

/// Writes the latest RL checkpoint payloads, metadata, and resume state.
pub fn save_latest_rl_checkpoint_and_state<B, O>(
    artifacts: &RlArtifactPaths,
    model: &HydraModel<B>,
    optimizer: &O,
    global_step: usize,
    train_loss: f64,
    state: &crate::resume::RlResumeState,
) -> Result<(), String>
where
    B: AutodiffBackend,
    O: Optimizer<HydraModel<B>, B>,
{
    let skip_payload_write = latest_rl_payload_is_current(artifacts, global_step);
    if !skip_payload_write {
        save_model_payload(model, &artifacts.latest_model_base)?;
        save_optimizer_payload(optimizer, &artifacts.latest_optimizer_base)?;
    }
    let meta = checkpoint_meta(global_step, train_loss, None, None, None);
    write_checkpoint_meta(&artifacts.latest_model_base, &meta)?;
    write_rl_resume_state(&artifacts.latest_state_path, state)
}

fn latest_checkpoint_payload_exists(model_base: &Path, optimizer_base: &Path) -> bool {
    model_base.with_extension("mpk").exists()
        && model_base.with_extension("meta.json").exists()
        && optimizer_base.with_extension("bin").exists()
}

/// Returns true when latest BC checkpoint payload files match the resume state's global step.
#[must_use]
pub fn latest_bc_payload_is_current(artifacts: &BcArtifactPaths, global_step: usize) -> bool {
    latest_checkpoint_payload_exists(
        &artifacts.latest_model_base,
        &artifacts.latest_optimizer_base,
    ) && read_resume_state(&artifacts.latest_state_path)
        .map(|state| state.global_step == global_step)
        .unwrap_or(false)
}

/// Returns true when latest RL checkpoint payload files match the resume state's global step.
#[must_use]
pub fn latest_rl_payload_is_current(artifacts: &RlArtifactPaths, global_step: usize) -> bool {
    latest_checkpoint_payload_exists(
        &artifacts.latest_model_base,
        &artifacts.latest_optimizer_base,
    ) && read_rl_resume_state(&artifacts.latest_state_path)
        .map(|state| state.global_step == global_step)
        .unwrap_or(false)
}

/// Writes a preflight cache entry.
pub fn write_preflight_cache(path: &Path, entry: &PreflightCacheEntry) -> Result<(), String> {
    let json = serde_json::to_string_pretty(entry).map_err(|err| {
        format!(
            "failed to serialize preflight cache {}: {err}",
            path.display()
        )
    })?;
    atomic_write_text(path, &json, "preflight cache")
}

/// Writes a preflight benchmark report.
pub fn write_preflight_benchmark_report(
    path: &Path,
    report: &PreflightBenchmarkReport,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(report).map_err(|err| {
        format!(
            "failed to serialize preflight benchmark report {}: {err}",
            path.display()
        )
    })?;
    atomic_write_text(path, &json, "preflight benchmark report")
}

/// Reads a preflight cache entry if the path exists.
pub fn read_preflight_cache(path: &Path) -> Result<Option<PreflightCacheEntry>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read preflight cache {}: {err}", path.display()))?;
    let entry: PreflightCacheEntry = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse preflight cache {}: {err}", path.display()))?;
    Ok(Some(entry))
}

/// Writes a manifest cache entry.
pub fn write_manifest_cache(path: &Path, entry: &ManifestCacheEntry) -> Result<(), String> {
    let json = serde_json::to_string_pretty(entry).map_err(|err| {
        format!(
            "failed to serialize manifest cache {}: {err}",
            path.display()
        )
    })?;
    atomic_write_text(path, &json, "manifest cache")
}

/// Reads a manifest cache entry if the path exists.
pub fn read_manifest_cache(path: &Path) -> Result<Option<ManifestCacheEntry>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read manifest cache {}: {err}", path.display()))?;
    let entry: ManifestCacheEntry = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse manifest cache {}: {err}", path.display()))?;
    Ok(Some(entry))
}

/// Returns true when a manifest cache entry matches the current scan inputs.
#[must_use]
pub fn manifest_cache_matches(
    cached: &ManifestCacheEntry,
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &hydra_train_runtime::config::SourceFilterConfig,
) -> bool {
    cached.data_dir == data_dir
        && cached.train_fraction_bits == train_fraction.to_bits()
        && cached.include_source_patterns == source_filters.include_source_patterns
        && cached.exclude_source_patterns == source_filters.exclude_source_patterns
}

/// Scans data sources and persists the resulting manifest cache entry.
pub fn scan_and_write_manifest_cache(
    cache_path: &Path,
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &hydra_train_runtime::config::SourceFilterConfig,
    progress: Option<&indicatif::ProgressBar>,
    scan_error_context: &str,
) -> Result<DataManifest, String> {
    let manifest =
        scan_data_sources_with_progress(data_dir, train_fraction, source_filters, progress)
            .map_err(|err| {
                format!(
                    "failed to scan {scan_error_context} from {}: {err}",
                    data_dir.display()
                )
            })?;
    write_manifest_cache(
        cache_path,
        &ManifestCacheEntry {
            data_dir: data_dir.to_path_buf(),
            train_fraction_bits: train_fraction.to_bits(),
            include_source_patterns: source_filters.include_source_patterns.clone(),
            exclude_source_patterns: source_filters.exclude_source_patterns.clone(),
            manifest: manifest.clone(),
        },
    )?;
    Ok(manifest)
}

/// Loads a matching manifest cache entry, or scans and rewrites it.
pub fn load_or_scan_manifest_cache<F>(
    cache_path: &Path,
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &hydra_train_runtime::config::SourceFilterConfig,
    progress: Option<&indicatif::ProgressBar>,
    scan_error_context: &str,
    on_cache_hit: F,
) -> Result<DataManifest, String>
where
    F: FnOnce(&ManifestCacheEntry),
{
    if let Some(cached) = read_manifest_cache(cache_path)?
        && manifest_cache_matches(&cached, data_dir, train_fraction, source_filters)
    {
        on_cache_hit(&cached);
        return Ok(cached.manifest);
    }
    scan_and_write_manifest_cache(
        cache_path,
        data_dir,
        train_fraction,
        source_filters,
        progress,
        scan_error_context,
    )
}

/// Scan data_dir and return all source locators without loading replay payloads.
pub fn scan_data_sources_with_progress(
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &hydra_train_runtime::config::SourceFilterConfig,
    progress: Option<&indicatif::ProgressBar>,
) -> io::Result<DataManifest> {
    let sources = if data_dir.is_file() {
        if is_tar_zst_file(data_dir) || is_tar_file(data_dir) {
            vec![DataSource::Archive(data_dir.to_path_buf())]
        } else if is_mjai_file(data_dir) {
            vec![DataSource::LooseFile(data_dir.to_path_buf())]
        } else if is_parsed_sample_cache_file(data_dir) {
            vec![data_source_for_cache_path(data_dir)?]
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "expected directory, MJAI file, parsed-sample cache file, or .tar/.tar.zst archive, got {}",
                    data_dir.display()
                ),
            ));
        }
    } else {
        scan_directory_sources(data_dir)?
    };
    let sources: Vec<DataSource> = sources
        .into_iter()
        .filter(|source| source_matches_filters(source, source_filters))
        .collect();

    let mut total_games = 0usize;
    let mut train_count = 0usize;
    let mut counts_exact = true;
    for source in &sources {
        match source {
            DataSource::LooseFile(path) => {
                total_games += 1;
                let identity = identity_for_loose_file(path)?;
                if is_train_game(&identity, train_fraction) {
                    train_count += 1;
                }
            }
            DataSource::ParsedSampleCache {
                original_identity, ..
            } => {
                total_games += 1;
                if is_train_game(original_identity, train_fraction) {
                    train_count += 1;
                }
            }
            DataSource::Archive(_) => {
                counts_exact = false;
            }
        }
        if let Some(pb) = progress {
            pb.inc(1);
        }
    }

    Ok(DataManifest {
        sources,
        total_games,
        train_count,
        val_count: total_games.saturating_sub(train_count),
        counts_exact,
    })
}

fn scan_directory_sources(dir: &Path) -> io::Result<Vec<DataSource>> {
    let mut sources = Vec::new();
    scan_directory_sources_recursive(dir, &mut sources)?;
    sources.sort_by(|a, b| a.path().cmp(b.path()));
    Ok(sources)
}

fn scan_directory_sources_recursive(dir: &Path, sources: &mut Vec<DataSource>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            scan_directory_sources_recursive(&path, sources)?;
        } else if file_type.is_file() {
            if is_mjai_file(&path) {
                sources.push(DataSource::LooseFile(path));
            } else if is_parsed_sample_cache_file(&path) {
                sources.push(data_source_for_cache_path(&path)?);
            } else if is_tar_zst_file(&path) || is_tar_file(&path) {
                sources.push(DataSource::Archive(path));
            }
        }
    }
    Ok(())
}

fn data_source_for_cache_path(path: &Path) -> io::Result<DataSource> {
    let ParsedSampleCacheMetadata {
        original_identity,
        original_source_path,
        ..
    } = read_parsed_sample_cache_metadata(path)?;
    Ok(DataSource::ParsedSampleCache {
        path: path.to_path_buf(),
        original_identity,
        original_source_path,
    })
}

fn source_matches_filters(
    source: &DataSource,
    filters: &hydra_train_runtime::config::SourceFilterConfig,
) -> bool {
    if filters.is_empty() {
        return true;
    }
    let path = match source {
        DataSource::ParsedSampleCache {
            original_source_path,
            ..
        } => original_source_path.to_string_lossy(),
        _ => source.path().to_string_lossy(),
    };
    let included = filters.include_source_patterns.is_empty()
        || filters
            .include_source_patterns
            .iter()
            .any(|pattern| path.contains(pattern));
    included
        && !filters
            .exclude_source_patterns
            .iter()
            .any(|pattern| path.contains(pattern))
}

fn identity_for_loose_file(path: &Path) -> io::Result<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "loose file path does not have a recognizable filename: {}",
                    path.display()
                ),
            )
        })?;
    if let Some(parent) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
    {
        Ok(format!("{parent}/{file_name}"))
    } else {
        Ok(file_name.to_owned())
    }
}

/// Deterministic train/val assignment by hashing game identity.
#[must_use]
pub fn is_train_game(identity: &str, train_fraction: f32) -> bool {
    let threshold = (normalized_train_fraction(train_fraction) * 1000.0).round() as u64;
    fnv1a_hash(identity.as_bytes()) % 1000 < threshold
}

fn normalized_train_fraction(train_fraction: f32) -> f32 {
    train_fraction.clamp(0.0, 1.0)
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".json")
                || name.ends_with(".json.gz")
                || name.ends_with(".json.zst")
    )
}

fn is_tar_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar")
    )
}

fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_checkpoint_base(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra_exec_{label}_{unique}"))
    }

    #[test]
    fn checkpoint_meta_semantic_match_ignores_timestamp_only() {
        let mut existing = CheckpointMeta::new(3, 1.25, Some(0.5), Some(1.0), Some(2.0));
        let mut candidate = existing.clone();
        candidate.timestamp = existing.timestamp.saturating_add(10);

        assert!(checkpoint_meta_semantically_matches(&existing, &candidate));

        existing.hidden_channels += 1;
        assert!(!checkpoint_meta_semantically_matches(&existing, &candidate));
    }

    #[test]
    fn write_checkpoint_meta_preserves_existing_semantic_match() {
        let base = temp_checkpoint_base("checkpoint_meta_preserve");
        let meta_path = base.with_extension("meta.json");
        let meta = CheckpointMeta::new(4, 2.5, None, None, None);

        write_checkpoint_meta(&base, &meta).expect("write checkpoint metadata");
        let first_raw = fs::read_to_string(&meta_path).expect("read checkpoint metadata");
        let mut same_semantics = meta.clone();
        same_semantics.timestamp = meta.timestamp.saturating_add(60);
        write_checkpoint_meta(&base, &same_semantics)
            .expect("rewrite matching checkpoint metadata");

        assert_eq!(
            fs::read_to_string(&meta_path).expect("read preserved checkpoint metadata"),
            first_raw
        );
        let _ = fs::remove_file(meta_path);
    }
}
