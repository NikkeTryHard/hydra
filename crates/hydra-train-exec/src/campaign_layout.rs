//! Shared Hydra campaign/run artifact layout helpers.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

const CAMPAIGN_FILE: &str = "campaign.json";
const REGISTRY_DIR: &str = "registry";
const STAGES_DIR: &str = "stages";
const RUNS_DIR: &str = "runs";

/// Resolved campaign root, stage, run id, and concrete run artifact directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CampaignRunLayout {
    /// Campaign root containing registry and stage directories.
    pub(crate) campaign_root: PathBuf,
    /// Campaign stage name.
    pub(crate) stage: String,
    /// Run id/name within the stage.
    pub(crate) run_id: String,
    /// Concrete run directory used by launchers and artifact writers.
    pub(crate) run_dir: PathBuf,
}

impl CampaignRunLayout {
    /// Resolves a campaign layout from either a campaign root or an already-resolved run dir.
    pub(crate) fn new(
        root_or_run_dir: &Path,
        stage: Option<&str>,
        run_name: Option<&str>,
        default_stage: &str,
    ) -> Self {
        if let Some((campaign_root, stage, run_id)) = split_run_dir(root_or_run_dir) {
            return Self {
                campaign_root,
                stage,
                run_id,
                run_dir: root_or_run_dir.to_path_buf(),
            };
        }

        let stage = stage.unwrap_or(default_stage).to_string();
        let run_id = run_name.map_or_else(timestamp_run_id, ToString::to_string);
        let run_dir = root_or_run_dir
            .join(STAGES_DIR)
            .join(&stage)
            .join(RUNS_DIR)
            .join(&run_id);
        Self {
            campaign_root: root_or_run_dir.to_path_buf(),
            stage,
            run_id,
            run_dir,
        }
    }

    /// Creates campaign registry files, run artifact dirs, latest marker, and launch metadata.
    pub(crate) fn ensure(&self) -> Result<(), String> {
        fs::create_dir_all(&self.campaign_root).map_err(|err| {
            format!(
                "failed to create campaign root {}: {err}",
                self.campaign_root.display()
            )
        })?;
        let registry = self.campaign_root.join(REGISTRY_DIR);
        fs::create_dir_all(registry.join("opponent_pools")).map_err(|err| {
            format!(
                "failed to create campaign registry {}: {err}",
                registry.display()
            )
        })?;
        ensure_file(
            &self.campaign_root.join(CAMPAIGN_FILE),
            b"{\n  \"schema_version\": 1,\n  \"kind\": \"hydra_campaign\"\n}\n",
            "campaign.json",
        )?;
        ensure_file(&registry.join("baselines.jsonl"), b"", "baseline registry")?;
        ensure_file(
            &registry.join("promotions.jsonl"),
            b"",
            "promotion registry",
        )?;
        ensure_file(
            &registry.join("seed_bank.json"),
            b"{\n  \"seeds\": []\n}\n",
            "seed bank",
        )?;

        for dir in ["logs", "checkpoints", "exports", "rollouts", "eval"] {
            fs::create_dir_all(self.run_dir.join(dir)).map_err(|err| {
                format!(
                    "failed to create run artifact dir {}: {err}",
                    self.run_dir.join(dir).display()
                )
            })?;
        }
        let latest_run = self
            .campaign_root
            .join(STAGES_DIR)
            .join(&self.stage)
            .join("latest_run");
        fs::write(&latest_run, format!("{}\n", self.run_id)).map_err(|err| {
            format!(
                "failed to write stage latest_run marker {}: {err}",
                latest_run.display()
            )
        })?;
        fs::write(
            self.run_dir.join("launch_metadata.json"),
            serde_json::to_vec_pretty(&json!({
                "schema_version": 1,
                "campaign_root": self.campaign_root,
                "stage": self.stage,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
            }))
            .map_err(|err| format!("failed to encode launch metadata: {err}"))?,
        )
        .map_err(|err| format!("failed to write launch metadata: {err}"))?;
        Ok(())
    }

    /// Creates the TensorBoard artifact directory for launchers that need it.
    pub(crate) fn ensure_tensorboard_dir(&self) -> Result<(), String> {
        fs::create_dir_all(self.run_dir.join("tensorboard")).map_err(|err| {
            format!(
                "failed to create run tensorboard dir {}: {err}",
                self.run_dir.join("tensorboard").display()
            )
        })
    }
}

fn ensure_file(path: &Path, contents: &[u8], label: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    fs::write(path, contents)
        .map_err(|err| format!("failed to create {label} {}: {err}", path.display()))
}

fn split_run_dir(path: &Path) -> Option<(PathBuf, String, String)> {
    let run_id = path.file_name()?.to_str()?.to_string();
    let runs = path.parent()?;
    if runs.file_name()?.to_str()? != RUNS_DIR {
        return None;
    }
    let stage_dir = runs.parent()?;
    let stage = stage_dir.file_name()?.to_str()?.to_string();
    let stages = stage_dir.parent()?;
    if stages.file_name()?.to_str()? != STAGES_DIR {
        return None;
    }
    let campaign_root = stages.parent()?.to_path_buf();
    Some((campaign_root, stage, run_id))
}

fn timestamp_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("run-{millis}")
}

#[cfg(test)]
mod tests;
