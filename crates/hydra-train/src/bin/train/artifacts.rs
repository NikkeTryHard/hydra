use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::optim::Optimizer;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use tboard::EventWriter;

use hydra_train::model::HydraModel;
use hydra_train::preflight::{default_cache_name, PreflightCacheEntry};
use hydra_train::training::bc::CheckpointMeta;

use super::progress::{EpochLogEntry, RlStepLogEntry, ScalarAverages, StepLogEntry};
use super::resume::{
    build_resume_state, current_timestamp_s, write_resume_state, BestValidation, EpochContinuation,
    RlResumeState, RuntimeResumeContract,
};
use super::validation::ValidationSummary;
use super::TrainBackend;

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
    global_step: usize,
    train_loss: f64,
    best_validation: Option<BestValidation>,
    continuation: &EpochContinuation,
    runtime: RuntimeResumeContract,
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
    val_summary: Option<ValidationSummary>,
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
    val_summary: Option<ValidationSummary>,
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
