#[cfg(test)]
#[path = "train/epoch_runner.rs"]
mod epoch_runner;

use std::env;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use burn::backend::{Autodiff, LibTorch};
use colored::control as color_control;

use hydra_train_exec::graph_probe::{handle_graph_probe_child, handle_graph_probe_parent};
use hydra_train_exec::modes::run_train_modes;
use hydra_train_exec::preflight_runtime::run_probe_child_mode;
use hydra_train_runtime::config::{parse_args, read_config};

#[cfg(test)]
use hydra_train_exec::resume::{
    BcResumeState, ResumeSemantics, build_resume_state, checkpoint_base_from_path,
    latest_optimizer_base_for_checkpoint_base, latest_state_path_for_checkpoint_base,
    read_resume_state, resume_banner_message, test_runtime_resume_contract,
};
#[cfg(test)]
use hydra_train_runtime::config::{
    AdvancedLossConfig, BcHyperparamConfig, TrainConfig, default_seed, validation_microbatch_size,
    validation_sample_limit,
};
#[cfg(test)]
use hydra_train_runtime::preflight::PreflightConfig;

#[cfg(test)]
type TrainBackend = Autodiff<LibTorch<f32>>;

fn run() -> Result<(), String> {
    color_control::set_override(true);
    let cli = parse_args(env::args())?;
    let config = read_config(&cli.config_path)?;
    hydra_train_exec::gpu_config::apply_gpu_performance_flags(&config.device);
    if std::env::var_os("HYDRA_CUDA_GRAPH_PROBE_CHILD").is_some() {
        return handle_graph_probe_child(&cli.config_path);
    }
    if std::env::var_os("HYDRA_CUDA_GRAPH_PROBE").is_some() {
        return handle_graph_probe_parent(&cli.config_path);
    }
    if run_probe_child_mode(&config, cli.probe_child.clone())? {
        return Ok(());
    }
    run_train_modes(cli, config)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "train/tests.rs"]
mod tests;
