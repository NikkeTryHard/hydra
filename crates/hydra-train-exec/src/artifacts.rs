//! Artifact path and log-only helpers shared across training execution seams.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use hydra_train_runtime::preflight::{
    BenchmarkResult, EffectiveRuntimeConfig, ManifestCacheEntry, PreflightCacheEntry,
    PreflightCacheKey, default_cache_name, default_manifest_cache_name,
};

use crate::resume::current_timestamp_s;

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
