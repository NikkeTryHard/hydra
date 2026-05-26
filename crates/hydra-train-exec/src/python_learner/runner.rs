//! Process runner seam for Python learner launch.

use std::fs::File;
use std::process::{ExitStatus, Stdio};

use super::PythonLearnerCommand;

/// Process runner seam for unit tests and the real OS process boundary.
pub trait PythonLearnerRunner {
    /// Runs a fully built command and returns its exit status.
    fn run(&self, command: &PythonLearnerCommand) -> Result<ExitStatus, String>;
    /// Spawns a command detached from the terminal and returns its process id.
    fn spawn_background(
        &self,
        command: &PythonLearnerCommand,
        stdout: File,
        stderr: File,
    ) -> Result<u32, String>;
}

/// OS-backed Python learner process runner.
#[derive(Debug, Clone, Copy, Default)]
pub struct OsPythonLearnerRunner;

impl PythonLearnerRunner for OsPythonLearnerRunner {
    fn run(&self, command: &PythonLearnerCommand) -> Result<ExitStatus, String> {
        command
            .command()
            .status()
            .map_err(|err| format!("failed to spawn Python learner through pixi: {err}"))
    }

    fn spawn_background(
        &self,
        command: &PythonLearnerCommand,
        stdout: File,
        stderr: File,
    ) -> Result<u32, String> {
        let child = command
            .command()
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|err| {
                format!("failed to spawn background Python learner through pixi: {err}")
            })?;
        Ok(child.id())
    }
}
