//! TensorBoard port, URL, and supervised background helpers.

use std::fs;
use std::fs::File;
use std::io::Write;
use std::net::TcpListener;
use std::path::Path;

use super::command_builder::build_python_learner_command_for_run_dir;
use hydra_train_runtime::config::{PythonLearnerCliOptions, PythonPpoControlCliOptions};

use super::PythonLearnerCommand;

pub(crate) const TENSORBOARD_PID_FILE: &str = "tensorboard.pid";

pub(crate) fn tensorboard_url(options: &PythonLearnerCliOptions, port: u16) -> String {
    format!("http://{}:{port}/", options.tensorboard_host)
}

pub(crate) fn tensorboard_url_for_ppo_options(
    options: &PythonPpoControlCliOptions,
    port: u16,
) -> String {
    format!("http://{}:{port}/", options.tensorboard_host)
}

pub(crate) fn first_free_port(host: &str, preferred_port: u16) -> Result<u16, String> {
    for port in preferred_port..=u16::MAX {
        if TcpListener::bind((host, port)).is_ok() {
            return Ok(port);
        }
    }
    Err(format!(
        "no available TensorBoard port on {host} at or above {preferred_port}"
    ))
}

pub(crate) fn build_python_learner_command_for_run_dir_with_tensorboard_port(
    options: &PythonLearnerCliOptions,
    run_dir: &Path,
    selected_tensorboard_port: u16,
) -> PythonLearnerCommand {
    let mut command = build_python_learner_command_for_run_dir(options, run_dir);
    if options.tensorboard {
        for window in command.args.windows(2) {
            if window[0] == "--tensorboard-url" {
                let old = window[1].clone();
                let new = tensorboard_url(options, selected_tensorboard_port);
                for arg in &mut command.args {
                    if *arg == old {
                        *arg = new;
                        break;
                    }
                }
                break;
            }
        }
    }
    command
}

pub(crate) fn write_tensorboard_pid_file_for_run_dir(
    options: &PythonLearnerCliOptions,
    run_dir: &Path,
    selected_port: u16,
) -> Result<(), String> {
    if !options.tensorboard || !options.launch_tensorboard {
        return Ok(());
    }
    let pid_path = run_dir.join(TENSORBOARD_PID_FILE);
    let mut pid_file = File::create(&pid_path).map_err(|err| {
        format!(
            "failed to create TensorBoard pid file {}: {err}",
            pid_path.display()
        )
    })?;
    writeln!(pid_file, "supervised:{selected_port}").map_err(|err| {
        format!(
            "failed to write TensorBoard pid file {}: {err}",
            pid_path.display()
        )
    })?;
    Ok(())
}

pub(crate) fn write_tensorboard_pid_file_for_ppo_run_dir(
    options: &PythonPpoControlCliOptions,
    run_dir: &Path,
    selected_port: u16,
) -> Result<(), String> {
    if !options.tensorboard || !options.launch_tensorboard {
        return Ok(());
    }
    let pid_path = run_dir.join(TENSORBOARD_PID_FILE);
    let mut pid_file = File::create(&pid_path).map_err(|err| {
        format!(
            "failed to create TensorBoard pid file {}: {err}",
            pid_path.display()
        )
    })?;
    writeln!(pid_file, "supervised:{selected_port}").map_err(|err| {
        format!(
            "failed to write TensorBoard pid file {}: {err}",
            pid_path.display()
        )
    })?;
    Ok(())
}

pub(crate) fn supervised_background_command_for_ppo_run_dir(
    command: &PythonLearnerCommand,
    options: &PythonPpoControlCliOptions,
    run_dir: &Path,
    selected_port: u16,
) -> PythonLearnerCommand {
    let mut args = vec![
        "scripts/python_train_supervisor.py".to_string(),
        "--tensorboard-pid-file".to_string(),
        run_dir.join(TENSORBOARD_PID_FILE).display().to_string(),
        "--tensorboard-logdir".to_string(),
        run_dir.join("tensorboard").display().to_string(),
        "--tensorboard-host".to_string(),
        options.tensorboard_host.clone(),
        "--tensorboard-port".to_string(),
        selected_port.to_string(),
        "--tensorboard-log".to_string(),
        run_dir.join("logs/tensorboard.log").display().to_string(),
        "--".to_string(),
        command.program.clone(),
    ];
    args.extend(command.args.clone());
    PythonLearnerCommand {
        program: "python".to_string(),
        args,
        result_path: command.result_path.clone(),
    }
}

pub(crate) fn supervised_background_command_for_run_dir(
    command: &PythonLearnerCommand,
    options: &PythonLearnerCliOptions,
    run_dir: &Path,
    selected_port: u16,
) -> PythonLearnerCommand {
    let mut args = vec![
        "scripts/python_train_supervisor.py".to_string(),
        "--tensorboard-pid-file".to_string(),
        run_dir.join(TENSORBOARD_PID_FILE).display().to_string(),
        "--tensorboard-logdir".to_string(),
        run_dir.join("tensorboard").display().to_string(),
        "--tensorboard-host".to_string(),
        options.tensorboard_host.clone(),
        "--tensorboard-port".to_string(),
        selected_port.to_string(),
        "--tensorboard-log".to_string(),
        run_dir.join("logs/tensorboard.log").display().to_string(),
        "--".to_string(),
        command.program.clone(),
    ];
    args.extend(command.args.clone());
    PythonLearnerCommand {
        program: "python".to_string(),
        args,
        result_path: command.result_path.clone(),
    }
}

pub(crate) fn tensorboard_port_for_run_dir(
    options: &PythonLearnerCliOptions,
    run_dir: &Path,
) -> Result<u16, String> {
    if options.tensorboard && options.launch_tensorboard {
        let pid_path = run_dir.join(TENSORBOARD_PID_FILE);
        if pid_path.is_file()
            && let Ok(contents) = fs::read_to_string(&pid_path)
            && let Ok(pid) = contents.trim().parse::<u32>()
            && process_is_running(pid)
        {
            return Ok(options.tensorboard_port);
        }
    }
    first_free_port(&options.tensorboard_host, options.tensorboard_port)
}

pub(crate) fn tensorboard_port_for_ppo_options(
    options: &PythonPpoControlCliOptions,
    run_dir: &Path,
) -> Result<u16, String> {
    if options.tensorboard && options.launch_tensorboard {
        let pid_path = run_dir.join(TENSORBOARD_PID_FILE);
        if pid_path.is_file()
            && let Ok(contents) = fs::read_to_string(&pid_path)
            && let Ok(pid) = contents.trim().parse::<u32>()
            && process_is_running(pid)
        {
            return Ok(options.tensorboard_port);
        }
    }
    first_free_port(&options.tensorboard_host, options.tensorboard_port)
}

pub(crate) fn process_is_running(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}
