//! Built Python learner command DTO.

use std::path::PathBuf;
use std::process::Command;

/// Built command for launching the Python BC learner through Pixi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonLearnerCommand {
    /// Executable name.
    pub program: String,
    /// Argument vector, excluding executable.
    pub args: Vec<String>,
    /// JSON result path passed to Python.
    pub result_path: PathBuf,
}

impl PythonLearnerCommand {
    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        command
    }
}
