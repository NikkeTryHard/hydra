//! Python T1 PPO-control learner run orchestration.

use std::fs::{self, File};

use hydra_train_runtime::config::PythonPpoControlCliOptions;

use super::command_builder::build_python_ppo_control_command_for_run_dir;
use super::layout;
use super::runner::{OsPythonLearnerRunner, PythonLearnerRunner};
use super::{PythonLearnerReport, parse_python_learner_report};

/// Runs the Python T1 PPO-control learner.
pub fn run_python_ppo_control(
    options: &PythonPpoControlCliOptions,
) -> Result<PythonLearnerReport, String> {
    run_python_ppo_control_with_runner(options, &OsPythonLearnerRunner)
}

/// Runs the Python T1 PPO-control learner with an injectable process runner.
pub fn run_python_ppo_control_with_runner(
    options: &PythonPpoControlCliOptions,
    runner: &impl PythonLearnerRunner,
) -> Result<PythonLearnerReport, String> {
    if !options.init_checkpoint.is_file() {
        return Err(format!(
            "Python PPO control init checkpoint does not exist or is not a file: {}",
            options.init_checkpoint.display()
        ));
    }
    let layout = layout::for_ppo(options);
    layout.ensure()?;
    if options.tensorboard {
        layout.ensure_tensorboard_dir()?;
    }
    let command = build_python_ppo_control_command_for_run_dir(options, &layout.run_dir);
    if options.background {
        let stdout_path = layout.run_dir.join("logs/stdout.log");
        let stderr_path = layout.run_dir.join("logs/stderr.log");
        let stdout = File::create(&stdout_path).map_err(|err| {
            format!(
                "failed to create background stdout log {}: {err}",
                stdout_path.display()
            )
        })?;
        let stderr = File::create(&stderr_path).map_err(|err| {
            format!(
                "failed to create background stderr log {}: {err}",
                stderr_path.display()
            )
        })?;
        let pid = runner.spawn_background(&command, stdout, stderr)?;
        fs::write(layout.run_dir.join("train.pid"), format!("{pid}\n"))
            .map_err(|err| format!("failed to write train.pid: {err}"))?;
        return Ok(PythonLearnerReport {
            result_path: command.result_path,
            samples_per_second: 0.0,
            global_step: 0,
            checkpoint_path: Some(layout.run_dir.join("checkpoints/latest.pt")),
            log_dir: layout.run_dir.join("logs"),
            tensorboard_dir: options
                .tensorboard
                .then(|| layout.run_dir.join("tensorboard")),
            tensorboard_url: options.tensorboard.then(|| {
                format!(
                    "http://{}:{}/",
                    options.tensorboard_host, options.tensorboard_port
                )
            }),
            pid: Some(pid),
        });
    }
    let status = runner.run(&command)?;
    if !status.success() {
        return Err(format!(
            "Python PPO control failed with status {status}; JSON result path: {}",
            command.result_path.display()
        ));
    }
    parse_python_learner_report(&command.result_path)
}
