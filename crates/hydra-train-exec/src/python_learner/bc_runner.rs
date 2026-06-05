//! Python BC learner run orchestration.

use std::env;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use hydra_train_runtime::config::{
    PythonLearnerCliOptions, PythonLearnerInput, PythonRawMjaiTransportConfig,
};

use super::layout;
use super::runner::{OsPythonLearnerRunner, PythonLearnerRunner};
use super::tensorboard::{
    build_python_learner_command_for_run_dir_with_tensorboard_port,
    supervised_background_command_for_run_dir, tensorboard_port_for_run_dir, tensorboard_url,
    write_tensorboard_pid_file_for_run_dir,
};
use super::{PythonLearnerReport, parse_python_learner_report};

/// Runs the Python learner after validating Rust-owned launch contracts.
pub fn run_python_learner(
    options: &PythonLearnerCliOptions,
) -> Result<PythonLearnerReport, String> {
    run_python_learner_with_runner(options, &OsPythonLearnerRunner)
}

/// Runs the Python learner with an injectable process runner.
pub fn run_python_learner_with_runner(
    options: &PythonLearnerCliOptions,
    runner: &impl PythonLearnerRunner,
) -> Result<PythonLearnerReport, String> {
    match &options.input {
        PythonLearnerInput::BcShards { manifest } if !manifest.is_file() => {
            return Err(format!(
                "Python BC learner manifest does not exist or is not a file: {}",
                manifest.display()
            ));
        }
        PythonLearnerInput::RawMjai { data_dirs, .. } => {
            if data_dirs.is_empty() {
                return Err(
                    "Python BC learner raw MJAI input requires at least one data dir".to_string(),
                );
            }
            for data_dir in data_dirs {
                if !data_dir.exists() {
                    return Err(format!(
                        "Python BC learner raw MJAI data dir does not exist: {}",
                        data_dir.display()
                    ));
                }
            }
        }
        _ => {}
    }
    ensure_raw_mjai_pyo3_extension(options)?;
    let layout = layout::for_bc(options);
    layout.ensure()?;
    if options.tensorboard {
        layout.ensure_tensorboard_dir()?;
    }
    let selected_tensorboard_port = tensorboard_port_for_run_dir(options, &layout.run_dir)?;
    if !options.background {
        write_tensorboard_pid_file_for_run_dir(
            options,
            &layout.run_dir,
            selected_tensorboard_port,
        )?;
    }
    let command = build_python_learner_command_for_run_dir_with_tensorboard_port(
        options,
        &layout.run_dir,
        selected_tensorboard_port,
    );
    if options.background {
        let command = if options.tensorboard && options.launch_tensorboard {
            supervised_background_command_for_run_dir(
                &command,
                options,
                &layout.run_dir,
                selected_tensorboard_port,
            )
        } else {
            command
        };
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
            tensorboard_url: options
                .tensorboard
                .then(|| tensorboard_url(options, selected_tensorboard_port)),
            pid: Some(pid),
        });
    }
    if options.tensorboard && options.launch_tensorboard {
        let _ = Command::new("pixi")
            .args([
                "run",
                "-e",
                "py-train",
                "tensorboard",
                "--logdir",
                &layout.run_dir.join("tensorboard").display().to_string(),
                "--host",
                &options.tensorboard_host,
                "--port",
                &selected_tensorboard_port.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    }
    let status = runner.run(&command)?;
    if !status.success() {
        return Err(format!(
            "Python BC learner failed with status {status}; JSON result path: {}",
            command.result_path.display()
        ));
    }
    parse_python_learner_report(&command.result_path)
}

fn ensure_raw_mjai_pyo3_extension(options: &PythonLearnerCliOptions) -> Result<(), String> {
    if !matches!(
        &options.input,
        PythonLearnerInput::RawMjai {
            transport: PythonRawMjaiTransportConfig::PinnedPyo3,
            ..
        }
    ) {
        return Ok(());
    }
    if let Some(path) = env::var_os("HYDRA_RAW_MJAI_PYO3_LIB") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(());
        }
        return Err(format!(
            "HYDRA_RAW_MJAI_PYO3_LIB points to missing raw MJAI PyO3 extension: {}",
            path.display()
        ));
    }
    let extension_path = raw_mjai_pyo3_release_path()?;
    if extension_path.is_file() {
        return Ok(());
    }
    let status = Command::new("cargo")
        .args(["build", "-p", "hydra-raw-mjai-pyo3", "--release", "--quiet"])
        .status()
        .map_err(|err| format!("failed to spawn raw MJAI PyO3 extension build: {err}"))?;
    if !status.success() {
        return Err(format!(
            "raw MJAI PyO3 extension build failed with status {status}; run `cargo build -p hydra-raw-mjai-pyo3 --release` for details"
        ));
    }
    if !extension_path.is_file() {
        return Err(format!(
            "raw MJAI PyO3 extension build completed but did not create {}",
            extension_path.display()
        ));
    }
    Ok(())
}

fn raw_mjai_pyo3_release_path() -> Result<PathBuf, String> {
    let cwd = env::current_dir().map_err(|err| format!("failed to resolve current dir: {err}"))?;
    Ok(cwd
        .join("target")
        .join("release")
        .join(raw_mjai_pyo3_library_name()))
}

fn raw_mjai_pyo3_library_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libhydra_raw_mjai_pyo3.dylib"
    } else if cfg!(target_os = "windows") {
        "hydra_raw_mjai_pyo3.dll"
    } else {
        "libhydra_raw_mjai_pyo3.so"
    }
}

/// Runs a Python BC learner benchmark for one batch/microbatch candidate.
pub fn run_python_learner_benchmark_row(
    base: &PythonLearnerCliOptions,
    batch_size: usize,
    microbatch_size: usize,
    warmup_steps: usize,
    measure_steps: usize,
    runner: &impl PythonLearnerRunner,
) -> Result<PythonLearnerReport, String> {
    let mut options = base.clone();
    options.batch_size = batch_size;
    options.microbatch_size = microbatch_size;
    options.warmup_steps = warmup_steps;
    options.steps = Some(measure_steps.max(1));
    options.schedule_total_steps = Some(measure_steps.max(1));
    options.checkpoint_out = None;
    options.resume = None;
    options.checkpoint_every_steps = 0;
    run_python_learner_with_runner(&options, runner)
}
