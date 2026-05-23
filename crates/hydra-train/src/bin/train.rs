use colored::control as color_control;
use std::env;

#[cfg(test)]
use hydra_train_runtime::config::validation_sample_limit;
use hydra_train_runtime::config::{BcBackend, parse_args, python_options_from_config, read_config};

#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use hydra_train_exec::resume::{
    BcResumeState, ResumeSemantics, build_resume_state, checkpoint_base_from_path,
    latest_optimizer_base_for_checkpoint_base, latest_state_path_for_checkpoint_base,
    read_resume_state, resume_banner_message, test_runtime_resume_contract,
};
#[cfg(test)]
use hydra_train_runtime::config::{
    AdvancedLossConfig, BcHyperparamConfig, TrainConfig, default_seed, validation_microbatch_size,
};

fn run() -> Result<(), String> {
    color_control::set_override(true);
    let cli = parse_args(env::args())?;

    if cli.list_devices {
        return Err("--list-devices is not supported by the Python BC launcher".to_string());
    }
    if cli.preflight.is_some() || cli.benchmark_baseline.is_some() {
        return Err(
            "--preflight and --benchmark-baseline are not supported by the Python BC launcher"
                .to_string(),
        );
    }
    if let Some(python_learner) = cli.python_learner.as_ref() {
        hydra_train_exec::gpu_config::apply_gpu_performance_flags(&python_learner.device);
        let report = hydra_train_exec::python_learner::run_python_learner(python_learner)?;
        if let Some(pid) = report.pid {
            println!(
                "Python learner running in background: pid={} output={} logs={} checkpoint={} tensorboard={}",
                pid,
                python_learner.output_dir.display(),
                report.log_dir.display(),
                report
                    .checkpoint_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                report.tensorboard_url.as_deref().unwrap_or("disabled")
            );
            println!(
                "watch logs: tail -f {}",
                report.log_dir.join("train_steps.jsonl").display()
            );
        } else {
            println!(
                "Python learner complete: samples/s={:.2} global_step={} result={} checkpoint={} logs={} tensorboard={}",
                report.samples_per_second,
                report.global_step,
                report.result_path.display(),
                report
                    .checkpoint_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                report.log_dir.display(),
                report.tensorboard_url.as_deref().unwrap_or("disabled")
            );
        }
        return Ok(());
    }
    let config_path = cli.config_path.as_deref().ok_or_else(|| {
        "config path is required unless --list-devices or --preflight is used".to_string()
    })?;
    let config = read_config(config_path)?;
    if config.rl.is_none()
        && !cli.delta_q_promotion
        && config.bc_backend.as_cli_backend() == BcBackend::Python
    {
        let python_learner = python_options_from_config(&config)?;
        hydra_train_exec::gpu_config::apply_gpu_performance_flags(&python_learner.device);
        let report = hydra_train_exec::python_learner::run_python_learner(&python_learner)?;
        if let Some(pid) = report.pid {
            println!(
                "Python BC learner running in background: pid={} output={} logs={} checkpoint={} tensorboard={}",
                pid,
                python_learner.output_dir.display(),
                report.log_dir.display(),
                report
                    .checkpoint_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                report.tensorboard_url.as_deref().unwrap_or("disabled")
            );
            println!(
                "watch logs: tail -f {}",
                report.log_dir.join("train_steps.jsonl").display()
            );
        } else {
            println!(
                "Python BC learner complete: samples/s={:.2} global_step={} result={} checkpoint={} logs={} tensorboard={}",
                report.samples_per_second,
                report.global_step,
                report.result_path.display(),
                report
                    .checkpoint_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                report.log_dir.display(),
                report.tensorboard_url.as_deref().unwrap_or("disabled")
            );
        }
        return Ok(());
    }
    if cli.delta_q_promotion {
        return Err("DeltaQ promotion is not supported by the Python BC launcher".to_string());
    }
    if cli.probe_child.is_some() || cli.probe_only.is_some() {
        return Err("probe modes are not supported by the Python BC launcher".to_string());
    }
    Err("Rust Burn BC training has been removed from hydra-train; set bc_backend: python or use Python BC launcher flags".to_string())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "train/tests/mod.rs"]
mod tests;
