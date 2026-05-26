//! Python learner process-launch facade.
//!
//! Rust owns CLI/config and process boundary validation. Python owns training.

mod bc_runner;
mod command;
mod command_builder;
mod layout;
mod ppo_runner;
mod report;
mod runner;
mod tensorboard;

use std::fs;
use std::path::Path;

pub use bc_runner::{
    run_python_learner, run_python_learner_benchmark_row, run_python_learner_with_runner,
};
pub use command::PythonLearnerCommand;
pub use command_builder::{build_python_learner_command, build_python_ppo_control_command};
pub use ppo_runner::{run_python_ppo_control, run_python_ppo_control_with_runner};
use report::PythonLearnerJson;
pub use report::PythonLearnerReport;
pub use runner::{OsPythonLearnerRunner, PythonLearnerRunner};

/// Converts config to Python learner options and runs the launcher through the default runner.
pub fn run_python_learner_from_config(
    config: &hydra_train_runtime::config::TrainConfig,
) -> Result<PythonLearnerReport, String> {
    let options = hydra_train_runtime::config::python_options_from_config(config)?;
    run_python_learner(&options)
}

/// Parses the minimal JSON fields reported to Rust users.
pub fn parse_python_learner_report(path: &Path) -> Result<PythonLearnerReport, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read Python learner result {}: {err}",
            path.display()
        )
    })?;
    let parsed: PythonLearnerJson = serde_json::from_str(&text).map_err(|err| {
        format!(
            "failed to parse Python learner result {}: {err}",
            path.display()
        )
    })?;
    if !parsed.summary.samples_per_s.is_finite() {
        return Err(format!(
            "Python BC learner result {} has non-finite samples_per_s",
            path.display()
        ));
    }
    Ok(PythonLearnerReport {
        result_path: path.to_path_buf(),
        samples_per_second: parsed.summary.samples_per_s,
        global_step: parsed.global_step,
        checkpoint_path: parsed.checkpoint_path,
        log_dir: path.parent().unwrap_or_else(|| Path::new("")).join("logs"),
        tensorboard_dir: path.parent().map(|parent| parent.join("tensorboard")),
        tensorboard_url: None,
        pid: None,
    })
}

#[cfg(test)]
mod tests;
