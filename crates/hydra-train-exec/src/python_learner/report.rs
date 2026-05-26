//! Python learner result report DTOs and JSON decode shapes.

use std::path::PathBuf;

use serde::Deserialize;

/// Minimal success report parsed from Python learner JSON output.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonLearnerReport {
    /// JSON result path consumed by this report.
    pub result_path: PathBuf,
    /// Measured samples per second.
    pub samples_per_second: f64,
    /// Final global step after this run.
    pub global_step: u64,
    /// Optional checkpoint path emitted by Python.
    pub checkpoint_path: Option<PathBuf>,
    /// Training log directory.
    pub log_dir: PathBuf,
    /// TensorBoard-compatible scalar directory, when enabled.
    pub tensorboard_dir: Option<PathBuf>,
    /// TensorBoard URL to open, when auto-launch requested.
    pub tensorboard_url: Option<String>,
    /// Background learner process id, when detached.
    pub pid: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PythonLearnerJson {
    pub(crate) summary: PythonLearnerSummaryJson,
    pub(crate) global_step: u64,
    pub(crate) checkpoint_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PythonLearnerSummaryJson {
    pub(crate) samples_per_s: f64,
}
