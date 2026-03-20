use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::optim::Optimizer;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use tboard::EventWriter;

use hydra_train::eval::ArenaPromotionDecision;
use hydra_train::model::HydraModel;
use hydra_train::preflight::{PreflightCacheEntry, default_cache_name};
use hydra_train::training::bc::CheckpointMeta;
use hydra_train::training::delta_q_promotion::{
    DeltaQArenaConfirmationRequest, DeltaQArenaReport, DeltaQPolicyTransferReport,
    DeltaQPolicyTransferResult, DeltaQPromotionRecommendation, DeltaQPromotionReport,
    DeltaQPromotionResult,
};

use super::TrainBackend;
use super::progress::{EpochLogEntry, RlStepLogEntry, ScalarAverages, StepLogEntry};
use super::resume::{
    BestValidation, EpochContinuation, RlResumeState, RuntimeResumeContract, build_resume_state,
    current_timestamp_s, write_resume_state,
};
use super::validation::ValidationSummary;

pub(crate) struct BcArtifactPaths {
    pub(crate) root: PathBuf,
    pub(crate) tb_root: PathBuf,
    pub(crate) tb_session_dir: PathBuf,
    pub(crate) latest_model_base: PathBuf,
    pub(crate) latest_optimizer_base: PathBuf,
    pub(crate) best_model_base: PathBuf,
    pub(crate) latest_state_path: PathBuf,
    pub(crate) training_log_path: PathBuf,
    pub(crate) step_log_path: PathBuf,
    pub(crate) delta_q_promotion_path: PathBuf,
}

pub(crate) struct RlArtifactPaths {
    pub(crate) root: PathBuf,
    pub(crate) tb_root: PathBuf,
    pub(crate) tb_session_dir: PathBuf,
    pub(crate) latest_model_base: PathBuf,
    pub(crate) latest_optimizer_base: PathBuf,
    pub(crate) latest_state_path: PathBuf,
    pub(crate) step_log_path: PathBuf,
}

pub(crate) struct PreflightPaths {
    pub(crate) cache_path: PathBuf,
}

pub(crate) struct RlPreflightPaths {
    pub(crate) cache_path: PathBuf,
}

pub(crate) struct LatestCheckpointState<'a> {
    pub(crate) global_step: usize,
    pub(crate) train_loss: f64,
    pub(crate) best_validation: Option<BestValidation>,
    pub(crate) continuation: &'a EpochContinuation,
    pub(crate) runtime: RuntimeResumeContract,
}

impl PreflightPaths {
    pub(crate) fn new(artifacts: &BcArtifactPaths) -> Self {
        Self {
            cache_path: artifacts.root.join(default_cache_name()),
        }
    }
}

impl RlPreflightPaths {
    pub(crate) fn new(artifacts: &RlArtifactPaths) -> Self {
        Self {
            cache_path: artifacts.root.join(default_cache_name()),
        }
    }
}

impl BcArtifactPaths {
    pub(crate) fn new(output_dir: &Path, resume_global_step: usize) -> Self {
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
            root,
            tb_root,
            tb_session_dir,
        }
    }

    pub(crate) fn create_root_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|err| {
            format!(
                "failed to create BC artifact dir {}: {err}",
                self.root.display()
            )
        })?;
        Ok(())
    }

    pub(crate) fn create_tensorboard_dirs(&self) -> Result<(), String> {
        for dir in [&self.tb_root, &self.tb_session_dir] {
            fs::create_dir_all(dir).map_err(|err| {
                format!("failed to create BC artifact dir {}: {err}", dir.display())
            })?;
        }
        Ok(())
    }
}

impl RlArtifactPaths {
    pub(crate) fn new(output_dir: &Path, resume_global_step: usize) -> Self {
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

    pub(crate) fn create_root_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|err| {
            format!(
                "failed to create RL artifact dir {}: {err}",
                self.root.display()
            )
        })?;
        Ok(())
    }

    pub(crate) fn create_tensorboard_dirs(&self) -> Result<(), String> {
        for dir in [&self.tb_root, &self.tb_session_dir] {
            fs::create_dir_all(dir).map_err(|err| {
                format!("failed to create RL artifact dir {}: {err}", dir.display())
            })?;
        }
        Ok(())
    }
}

pub(crate) fn write_preflight_cache(
    path: &Path,
    entry: &PreflightCacheEntry,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(entry).map_err(|err| {
        format!(
            "failed to serialize preflight cache {}: {err}",
            path.display()
        )
    })?;
    fs::write(path, json)
        .map_err(|err| format!("failed to write preflight cache {}: {err}", path.display()))
}

pub(crate) fn read_preflight_cache(path: &Path) -> Result<Option<PreflightCacheEntry>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read preflight cache {}: {err}", path.display()))?;
    let entry: PreflightCacheEntry = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse preflight cache {}: {err}", path.display()))?;
    Ok(Some(entry))
}

pub(crate) fn save_latest_checkpoint_and_state<O>(
    artifacts: &BcArtifactPaths,
    model: &HydraModel<TrainBackend>,
    optimizer: &O,
    state: LatestCheckpointState<'_>,
) -> Result<(), String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
    let LatestCheckpointState {
        global_step,
        train_loss,
        best_validation,
        continuation,
        runtime,
    } = state;
    save_checkpoint(
        model,
        &artifacts.latest_model_base,
        global_step,
        train_loss,
        None,
    )?;
    let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    optimizer_recorder
        .record(
            optimizer.to_record(),
            artifacts.latest_optimizer_base.clone(),
        )
        .map_err(|err| {
            format!(
                "failed to save optimizer state {}: {err}",
                artifacts.latest_optimizer_base.display()
            )
        })?;
    let state = build_resume_state(
        continuation.next_epoch,
        continuation.skip_optimizer_steps_in_epoch,
        global_step,
        best_validation,
        runtime,
    );
    write_resume_state(&artifacts.latest_state_path, &state)
}

pub(crate) fn save_checkpoint(
    model: &HydraModel<TrainBackend>,
    base: &Path,
    epoch: usize,
    loss: f64,
    val_summary: Option<&ValidationSummary>,
) -> Result<(), String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(base, &recorder)
        .map_err(|err| format!("failed to save checkpoint {}: {err}", base.display()))?;
    let meta = CheckpointMeta::new(
        epoch as u32,
        loss,
        val_summary.map(|summary| summary.agreement),
        val_summary.map(|summary| summary.policy_loss),
        val_summary.map(|summary| summary.total_loss),
    );
    let meta_json = serde_json::to_string_pretty(&meta).map_err(|err| {
        format!(
            "failed to serialize checkpoint metadata {}: {err}",
            base.display()
        )
    })?;
    let meta_path = base.with_extension("meta.json");
    fs::write(&meta_path, meta_json).map_err(|err| {
        format!(
            "failed to write checkpoint metadata {}: {err}",
            meta_path.display()
        )
    })
}

pub(crate) fn append_training_log(path: &Path, entry: &EpochLogEntry) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open training log {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize training log entry: {err}"))?;
    writeln!(file, "{line}")
        .map_err(|err| format!("failed to append training log {}: {err}", path.display()))
}

pub(crate) fn append_step_log(path: &Path, entry: &StepLogEntry) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open step log {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize step log entry: {err}"))?;
    writeln!(file, "{line}")
        .map_err(|err| format!("failed to append step log {}: {err}", path.display()))
}

pub(crate) fn append_rl_step_log(path: &Path, entry: &RlStepLogEntry) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open RL step log {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize RL step log entry: {err}"))?;
    writeln!(file, "{line}")
        .map_err(|err| format!("failed to append RL step log {}: {err}", path.display()))
}

#[derive(serde::Serialize)]
pub(crate) struct PersistedDeltaQPromotionArtifact<'a> {
    pub(crate) scope: &'a str,
    pub(crate) step_or_epoch: usize,
    pub(crate) recommendation: DeltaQPromotionRecommendation,
    pub(crate) stage: &'a str,
    pub(crate) arena_confirmation: Option<DeltaQArenaConfirmationRequest>,
    pub(crate) arena_decision: Option<ArenaPromotionDecision>,
    pub(crate) arena_report: Option<&'a DeltaQArenaReport>,
    pub(crate) report: &'a DeltaQPromotionReport,
    pub(crate) result: &'a DeltaQPromotionResult,
    pub(crate) policy_transfer: Option<&'a DeltaQPolicyTransferReport>,
    pub(crate) policy_transfer_result: Option<&'a DeltaQPolicyTransferResult>,
}

pub(crate) fn write_delta_q_promotion_artifact(
    path: &Path,
    artifact: &PersistedDeltaQPromotionArtifact<'_>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(artifact)
        .map_err(|err| format!("failed to serialize delta_q promotion artifact: {err}"))?;
    fs::write(path, json).map_err(|err| {
        format!(
            "failed to write delta_q promotion artifact {}: {err}",
            path.display()
        )
    })
}

pub(crate) fn save_latest_rl_checkpoint_and_state<O>(
    artifacts: &RlArtifactPaths,
    model: &HydraModel<TrainBackend>,
    optimizer: &O,
    global_step: usize,
    train_loss: f64,
    state: &RlResumeState,
) -> Result<(), String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
    save_checkpoint(
        model,
        &artifacts.latest_model_base,
        global_step,
        train_loss,
        None,
    )?;
    let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    optimizer_recorder
        .record(
            optimizer.to_record(),
            artifacts.latest_optimizer_base.clone(),
        )
        .map_err(|err| {
            format!(
                "failed to save RL optimizer state {}: {err}",
                artifacts.latest_optimizer_base.display()
            )
        })?;
    write_rl_resume_state(&artifacts.latest_state_path, state)
}

fn write_rl_resume_state(path: &Path, state: &RlResumeState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(state).map_err(|err| {
        format!(
            "failed to serialize RL resume state {}: {err}",
            path.display()
        )
    })?;
    fs::write(path, yaml)
        .map_err(|err| format!("failed to write RL resume state {}: {err}", path.display()))
}

pub(crate) fn log_tensorboard<W: Write>(
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
    if let Some(val_summary) = val_summary {
        tb.write_scalar(step, "val/policy_agreement", val_summary.agreement as f32)
            .map_err(|err| format!("tensorboard write val/policy_agreement failed: {err}"))?;
        tb.write_scalar(step, "val/policy_loss", val_summary.policy_loss as f32)
            .map_err(|err| format!("tensorboard write val/policy_loss failed: {err}"))?;
        tb.write_scalar(step, "val/total_loss", val_summary.total_loss as f32)
            .map_err(|err| format!("tensorboard write val/total_loss failed: {err}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RlPhaseConfig;
    use crate::resume::{RlResumeSemantics, RlRuntimeResumeContract};
    use crate::validation::DeltaQPromotionSnapshot;
    use hydra_train::config::{PipelineState, TrainingPhase};
    use hydra_train::preflight::{
        EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, PreflightCacheKey,
        SelectedRuntimeConfig, WorkloadFingerprint,
    };
    use std::time::{SystemTime, UNIX_EPOCH};
    use tboard::SummaryReader;

    fn temp_dir_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra_{label}_{unique}"))
    }

    fn sample_preflight_cache_entry() -> PreflightCacheEntry {
        PreflightCacheEntry {
            cache_key: PreflightCacheKey {
                hardware: HardwareFingerprint {
                    device_label: "test-gpu".to_string(),
                    backend: "wgpu".to_string(),
                    cpu_logical_cores: 16,
                    total_memory_bytes: Some(64 * 1024),
                },
                workload: WorkloadFingerprint {
                    batch_size: 128,
                    augment: true,
                    train_fraction_bits: 1234,
                    max_skip_logs_per_source: 5,
                    max_validation_batches: Some(7),
                    max_validation_samples: Some(256),
                    model_signature: "model-sig".to_string(),
                    code_signature: "code-sig".to_string(),
                    advanced_loss_signature: "loss-sig".to_string(),
                },
            },
            runtime: EffectiveRuntimeConfig {
                selected: SelectedRuntimeConfig {
                    train_microbatch_size: 32,
                    validation_microbatch_size: 64,
                    accum_steps: 4,
                },
                loader: LoaderRuntimeConfig {
                    num_threads: Some(8),
                    buffer_games: 512,
                    buffer_samples: 2048,
                    archive_queue_bound: 32,
                },
            },
        }
    }

    fn sample_epoch_log_entry() -> EpochLogEntry {
        EpochLogEntry {
            epoch: 3,
            global_step: 17,
            lr: 0.01,
            train_total_loss: 1.5,
            train_policy_agreement: 0.25,
            train_loss_policy: 0.5,
            train_loss_value: 0.1,
            train_loss_grp: 0.2,
            train_loss_tenpai: 0.3,
            train_loss_danger: 0.4,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            val_total_loss: Some(1.2),
            val_policy_loss: Some(0.9),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            best_val_policy_loss: Some(0.8),
            best_val_agreement: Some(0.77),
            num_batches: 4,
        }
    }

    fn sample_step_log_entry() -> StepLogEntry {
        StepLogEntry {
            global_step: 17,
            epoch: 3,
            lr: 0.01,
            train_total_loss: 1.5,
            train_policy_agreement: 0.25,
            train_loss_policy: 0.5,
            train_loss_value: 0.1,
            train_loss_grp: 0.2,
            train_loss_tenpai: 0.3,
            train_loss_danger: 0.4,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            val_total_loss: Some(1.2),
            val_policy_loss: Some(0.9),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            best_val_policy_loss: Some(0.8),
            best_val_agreement: Some(0.77),
        }
    }

    fn sample_rl_step_log_entry() -> RlStepLogEntry {
        RlStepLogEntry {
            global_step: 12,
            phase: "exit_pondering".to_string(),
            loss: 0.55,
            effective_lr: 0.005,
            exit_weight: 0.25,
            games_per_batch: 8,
            samples_in_batch: 64,
            total_games: 1024,
            total_samples: 8192,
            delta_q_state: "Active".to_string(),
        }
    }

    fn sample_validation_summary() -> ValidationSummary {
        let promotion_report = DeltaQPromotionReport::new();
        let promotion_result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };
        ValidationSummary {
            total_loss: 1.2,
            policy_loss: 0.8,
            agreement: 0.7,
            samples: 64,
            delta_q_promotion: Some(promotion_report.clone()),
            delta_q_promotion_result: Some(promotion_result.clone()),
            delta_q_promotion_snapshot: Some(DeltaQPromotionSnapshot {
                compared_states: promotion_report.compared_states,
                candidate_top1_agreement: promotion_report.candidate_top1_agreement(),
                candidate_mean_regret: promotion_report.candidate_mean_regret(),
                baseline_mean_regret: promotion_report.baseline_mean_regret(),
                mean_decision_lift: promotion_report.mean_decision_lift(),
                negative_lift_fraction: promotion_report.negative_lift_fraction(),
                regret_beats_baseline_rate: promotion_report.candidate_regret_beats_baseline_rate(),
                top1_beats_baseline_rate: promotion_report.candidate_top1_beats_baseline_rate(),
                passed: promotion_result.passed,
            }),
            delta_q_policy_transfer: Some(DeltaQPolicyTransferReport::new()),
            delta_q_policy_transfer_result: Some(DeltaQPolicyTransferResult {
                passed: true,
                criteria: Vec::new(),
            }),
            delta_q_policy_transfer_snapshot: None,
        }
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    fn tensorboard_tags_from_dir(path: &Path) -> Vec<String> {
        let event_path = fs::read_dir(path)
            .expect("read tensorboard dir")
            .map(|entry| entry.expect("tensorboard dir entry").path())
            .find(|entry| entry.is_file())
            .expect("tensorboard event file");
        let file = fs::File::open(event_path).expect("open tensorboard event file");
        let mut tags = Vec::new();
        for event in SummaryReader::new(file).skip(1) {
            let event = event.expect("decode tensorboard event");
            let summary = match event.what.expect("event payload") {
                tboard::tensorboard::event::What::Summary(summary) => summary,
                other => panic!("expected summary event, got {other:?}"),
            };
            for value in summary.value {
                tags.push(value.tag);
            }
        }
        tags
    }

    #[test]
    fn bc_artifact_paths_build_expected_names() {
        let output_dir = temp_dir_path("bc_artifact_paths");
        let artifacts = BcArtifactPaths::new(&output_dir, 42);

        assert_eq!(artifacts.root, output_dir.join("bc"));
        assert_eq!(artifacts.tb_root, artifacts.root.join("tb"));
        assert_eq!(
            artifacts.latest_model_base,
            artifacts.root.join("latest_model")
        );
        assert_eq!(
            artifacts.latest_optimizer_base,
            artifacts.root.join("latest_optimizer")
        );
        assert_eq!(artifacts.best_model_base, artifacts.root.join("best_model"));
        assert_eq!(
            artifacts.latest_state_path,
            artifacts.root.join("latest_state.yaml")
        );
        assert_eq!(
            artifacts.training_log_path,
            artifacts.root.join("training_log.jsonl")
        );
        assert_eq!(
            artifacts.step_log_path,
            artifacts.root.join("step_log.jsonl")
        );
        assert_eq!(
            artifacts.delta_q_promotion_path,
            artifacts.root.join("delta_q_promotion.json")
        );
        let tb_session = artifacts
            .tb_session_dir
            .file_name()
            .expect("tensorboard session dir name")
            .to_string_lossy();
        assert!(tb_session.starts_with("run_g00000042_"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn rl_artifact_paths_build_expected_names() {
        let output_dir = temp_dir_path("rl_artifact_paths");
        let artifacts = RlArtifactPaths::new(&output_dir, 7);

        assert_eq!(artifacts.root, output_dir.join("rl"));
        assert_eq!(artifacts.tb_root, artifacts.root.join("tb"));
        assert_eq!(
            artifacts.latest_model_base,
            artifacts.root.join("latest_model")
        );
        assert_eq!(
            artifacts.latest_optimizer_base,
            artifacts.root.join("latest_optimizer")
        );
        assert_eq!(
            artifacts.latest_state_path,
            artifacts.root.join("latest_state.yaml")
        );
        assert_eq!(
            artifacts.step_log_path,
            artifacts.root.join("step_log.jsonl")
        );
        let tb_session = artifacts
            .tb_session_dir
            .file_name()
            .expect("tensorboard session dir name")
            .to_string_lossy();
        assert!(tb_session.starts_with("run_g00000007_"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn preflight_paths_use_default_cache_name_under_artifact_roots() {
        let output_dir = temp_dir_path("preflight_paths");
        let bc_artifacts = BcArtifactPaths::new(&output_dir, 0);
        let rl_artifacts = RlArtifactPaths::new(&output_dir, 0);

        let bc_preflight = PreflightPaths::new(&bc_artifacts);
        let rl_preflight = RlPreflightPaths::new(&rl_artifacts);

        assert_eq!(
            bc_preflight.cache_path,
            bc_artifacts.root.join(default_cache_name())
        );
        assert_eq!(
            rl_preflight.cache_path,
            rl_artifacts.root.join(default_cache_name())
        );

        cleanup_dir(&output_dir);
    }

    #[test]
    fn artifact_directory_creators_make_expected_directories() {
        let output_dir = temp_dir_path("artifact_dir_create");
        let bc_artifacts = BcArtifactPaths::new(&output_dir, 3);
        let rl_artifacts = RlArtifactPaths::new(&output_dir, 9);

        bc_artifacts.create_root_dir().expect("create bc root");
        bc_artifacts
            .create_tensorboard_dirs()
            .expect("create bc tensorboard dirs");
        rl_artifacts.create_root_dir().expect("create rl root");
        rl_artifacts
            .create_tensorboard_dirs()
            .expect("create rl tensorboard dirs");

        assert!(bc_artifacts.root.is_dir());
        assert!(bc_artifacts.tb_root.is_dir());
        assert!(bc_artifacts.tb_session_dir.is_dir());
        assert!(rl_artifacts.root.is_dir());
        assert!(rl_artifacts.tb_root.is_dir());
        assert!(rl_artifacts.tb_session_dir.is_dir());

        cleanup_dir(&output_dir);
    }

    #[test]
    fn read_preflight_cache_returns_none_for_missing_file() {
        let output_dir = temp_dir_path("missing_preflight_cache");
        let path = output_dir.join("missing.json");

        let entry = read_preflight_cache(&path).expect("read missing cache path");
        assert_eq!(entry, None);
    }

    #[test]
    fn preflight_cache_roundtrips_through_json() {
        let output_dir = temp_dir_path("preflight_cache_roundtrip");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("preflight_cache.json");
        let entry = sample_preflight_cache_entry();

        write_preflight_cache(&path, &entry).expect("write preflight cache");
        let restored = read_preflight_cache(&path)
            .expect("read preflight cache")
            .expect("cache entry present");

        assert_eq!(restored, entry);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn append_training_log_appends_jsonl_lines() {
        let output_dir = temp_dir_path("append_training_log");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("training_log.jsonl");
        let first = sample_epoch_log_entry();
        let second = EpochLogEntry {
            epoch: first.epoch + 1,
            global_step: first.global_step + 10,
            ..sample_epoch_log_entry()
        };

        append_training_log(&path, &first).expect("append first training log line");
        append_training_log(&path, &second).expect("append second training log line");

        let raw = fs::read_to_string(&path).expect("read training log");
        let lines: Vec<_> = raw.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"epoch\":3"));
        assert!(lines[1].contains("\"epoch\":4"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn append_step_logs_append_jsonl_lines() {
        let output_dir = temp_dir_path("append_step_logs");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let step_path = output_dir.join("step_log.jsonl");
        let rl_path = output_dir.join("rl_step_log.jsonl");

        append_step_log(&step_path, &sample_step_log_entry()).expect("append step log");
        append_rl_step_log(&rl_path, &sample_rl_step_log_entry()).expect("append rl step log");

        let step_raw = fs::read_to_string(&step_path).expect("read step log");
        let rl_raw = fs::read_to_string(&rl_path).expect("read rl step log");
        assert!(step_raw.contains("\"global_step\":17"));
        assert!(step_raw.contains("\"best_val_policy_loss\":0.8"));
        assert!(rl_raw.contains("\"phase\":\"exit_pondering\""));
        assert!(rl_raw.contains("\"delta_q_state\":\"Active\""));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn write_rl_resume_state_serializes_expected_yaml_fields() {
        let output_dir = temp_dir_path("write_rl_resume_state");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("latest_state.yaml");
        let state = RlResumeState {
            schema_version: 1,
            resume_semantics: RlResumeSemantics::RestoreOptimizerFreshSelfPlay,
            global_step: 19,
            pipeline_state: PipelineState {
                phase: TrainingPhase::DrdaAchSelfPlay,
                gpu_hours_used: 12.5,
                total_games: 500,
                total_samples: 4000,
                learner_version: 3,
                actor_version: 4,
            },
            runtime: RlRuntimeResumeContract {
                games_per_batch: 16,
                microbatch_size: 32,
                phase: RlPhaseConfig::ExitPondering,
            },
            saved_at_unix_s: 123,
        };

        write_rl_resume_state(&path, &state).expect("write rl resume state");

        let raw = fs::read_to_string(&path).expect("read rl resume state");
        assert!(raw.contains("schema_version: 1"));
        assert!(raw.contains("global_step: 19"));
        assert!(raw.contains("phase: DrdaAchSelfPlay"));
        assert!(raw.contains("games_per_batch: 16"));
        assert!(raw.contains("microbatch_size: 32"));
        assert!(raw.contains("phase: exit_pondering"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn log_tensorboard_writes_core_and_validation_scalars() {
        let output_dir = temp_dir_path("tensorboard_full");
        fs::create_dir_all(&output_dir).expect("create tensorboard dir");
        let train = ScalarAverages {
            total_loss: 2.0,
            policy_agreement: 0.25,
            ..Default::default()
        };
        let val_summary = sample_validation_summary();
        let best_validation = BestValidation {
            policy_loss: 0.5,
            agreement: 0.9,
        };
        let mut tb = EventWriter::create(&output_dir).expect("create tb writer");

        log_tensorboard(
            &mut tb,
            11,
            &train,
            Some(&val_summary),
            0.001,
            Some(best_validation),
        )
        .expect("write tensorboard scalars");

        drop(tb);
        let tags = tensorboard_tags_from_dir(&output_dir);

        assert!(tags.iter().any(|tag| tag == "train/total_loss"));
        assert!(tags.iter().any(|tag| tag == "train/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "val/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "val/policy_loss"));
        assert!(tags.iter().any(|tag| tag == "val/total_loss"));
        assert!(
            tags.iter()
                .any(|tag| tag == "val/delta_q_candidate_top1_agreement")
        );
        assert!(
            tags.iter()
                .any(|tag| tag == "val/delta_q_offline_gate_passed")
        );
        assert!(tags.iter().any(|tag| tag == "lr"));
        assert!(tags.iter().any(|tag| tag == "val/best_policy_loss"));
        assert!(tags.iter().any(|tag| tag == "val/best_policy_agreement"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn log_tensorboard_skips_optional_validation_scalars_when_absent() {
        let output_dir = temp_dir_path("tensorboard_train_only");
        fs::create_dir_all(&output_dir).expect("create tensorboard dir");
        let train = ScalarAverages {
            total_loss: 3.0,
            policy_agreement: 0.4,
            ..Default::default()
        };
        let mut tb = EventWriter::create(&output_dir).expect("create tb writer");

        log_tensorboard(&mut tb, 5, &train, None, 0.05, None).expect("write tensorboard scalars");

        drop(tb);
        let tags = tensorboard_tags_from_dir(&output_dir);

        assert!(tags.iter().any(|tag| tag == "train/total_loss"));
        assert!(tags.iter().any(|tag| tag == "train/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "lr"));
        assert!(!tags.iter().any(|tag| tag.starts_with("val/")));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn delta_q_promotion_artifact_serializes_arena_fields() {
        let dir = temp_dir_path("delta_q_promotion_artifact");
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("delta_q_promotion.json");

        let report = DeltaQPromotionReport::new();
        let result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };
        let arena_request = DeltaQArenaConfirmationRequest::default();
        let paired = hydra_train::eval::paired_arena_result_from_placements(
            &[0, 1, 1, 2],
            &[1, 2, 2, 3],
            0.02,
        );
        let arena_report = DeltaQArenaReport::from_paired_eval(&paired, -0.01);

        write_delta_q_promotion_artifact(
            &path,
            &PersistedDeltaQPromotionArtifact {
                scope: "promotion_mode",
                step_or_epoch: 0,
                recommendation: DeltaQPromotionRecommendation::RequiresArenaConfirmation,
                stage: "offline_transfer_and_arena_gate",
                arena_confirmation: Some(arena_request),
                arena_decision: Some(ArenaPromotionDecision::NonRegressionOnly),
                arena_report: Some(&arena_report),
                report: &report,
                result: &result,
                policy_transfer: None,
                policy_transfer_result: None,
            },
        )
        .expect("write artifact");

        let raw = fs::read_to_string(&path).expect("read artifact");
        assert!(raw.contains("\"arena_confirmation\""));
        assert!(raw.contains("\"arena_decision\""));
        assert!(raw.contains("\"arena_report\""));
        assert!(raw.contains("\"lower_confidence_bound_mean_placement\""));
        assert!(raw.contains("\"upper_confidence_bound_mean_placement\""));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }
}
