use colored::control as color_control;
use std::env;

use hydra_train_exec::graph_probe::{handle_graph_probe_child, handle_graph_probe_parent};
use hydra_train_exec::modes::{handle_list_devices_mode, run_train_modes};
use hydra_train_exec::preflight_runtime::run_probe_child_mode;
use hydra_train_runtime::config::{
    BcBackend, PythonLearnerCliOptions, PythonLearnerInput, parse_args, read_config,
};
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
    validation_sample_limit,
};

fn run() -> Result<(), String> {
    color_control::set_override(true);
    let cli = parse_args(env::args())?;
    hydra_train_exec::gpu_config::configure_libtorch_cpu_threads(
        hydra_train_runtime::config::default_num_threads_for_system(),
    );
    if cli.list_devices {
        return handle_list_devices_mode();
    }

    if cli.preflight.is_some() || cli.benchmark_baseline.is_some() {
        let device = cli
            .preflight
            .as_ref()
            .map(|preflight| preflight.device.clone())
            .or_else(|| {
                cli.benchmark_baseline
                    .as_ref()
                    .map(|benchmark| benchmark.device.clone())
            })
            .unwrap_or_else(hydra_train_runtime::config::default_device);
        let _benchmark_quiet = if cli.benchmark_baseline.is_some() {
            unsafe {
                std::env::set_var("HYDRA_BENCHMARK_QUIET", "1");
            }
            true
        } else {
            false
        };
        hydra_train_exec::gpu_config::apply_gpu_performance_flags(&device);
        return run_train_modes(
            cli,
            hydra_train_runtime::config::TrainConfig::default_preflight_bench(),
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
    hydra_train_exec::gpu_config::apply_gpu_performance_flags(&config.device);
    if std::env::var_os("HYDRA_CUDA_GRAPH_PROBE_CHILD").is_some() {
        return handle_graph_probe_child(config_path);
    }
    if std::env::var_os("HYDRA_CUDA_GRAPH_PROBE").is_some() {
        return handle_graph_probe_parent(config_path);
    }
    if run_probe_child_mode(&config, cli.probe_child.clone())? {
        return Ok(());
    }
    run_train_modes(cli, config)
}

fn python_options_from_config(
    config: &hydra_train_runtime::config::TrainConfig,
) -> Result<PythonLearnerCliOptions, String> {
    if config.exit_sidecar_path.is_some() {
        return Err(
            "Python BC learner does not support ExIt sidecars yet; set bc_backend: rust_burn for legacy Rust BC"
                .to_string(),
        );
    }
    if config.delta_q_sidecar_path.is_some() {
        return Err(
            "Python BC learner does not support DeltaQ sidecars yet; set bc_backend: rust_burn for legacy Rust BC"
                .to_string(),
        );
    }
    if let Some(loss) = config.advanced_loss.as_ref() {
        if loss.exit.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.exit yet; set bc_backend: rust_burn for legacy Rust BC"
                    .to_string(),
            );
        }
        if loss.delta_q.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.delta_q yet; set bc_backend: rust_burn for legacy Rust BC"
                    .to_string(),
            );
        }
        if loss.belief_fields.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.belief_fields yet; set bc_backend: rust_burn for legacy Rust BC"
                    .to_string(),
            );
        }
        if loss.mixture_weight.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.mixture_weight yet; set bc_backend: rust_burn for legacy Rust BC"
                    .to_string(),
            );
        }
        if loss.opponent_hand_type.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.opponent_hand_type yet; set bc_backend: rust_burn for legacy Rust BC"
                    .to_string(),
            );
        }
    }
    let advanced = config.advanced_loss.as_ref();
    Ok(PythonLearnerCliOptions {
        bc_shards_manifest: config
            .bc_shards_manifest_path
            .clone()
            .unwrap_or_else(|| config.data_dir.clone()),
        input: if let Some(manifest) = config.bc_shards_manifest_path.clone() {
            PythonLearnerInput::BcShards { manifest }
        } else {
            PythonLearnerInput::RawMjai {
                data_dir: config.data_dir.clone(),
                max_games: None,
                max_samples: None,
                train_fraction: config.train_fraction,
                augment: config.augment,
                transport: config.python_raw_mjai_transport,
            }
        },
        output_dir: config.output_dir.clone(),
        device: config.device.clone(),
        batch_size: config.batch_size,
        microbatch_size: config.microbatch_size.unwrap_or(1024),
        variant: config.python_variant,
        residual_profile: config.python_residual_profile,
        warmup_steps: config.bc.warmup_steps,
        steps: config.max_train_steps.unwrap_or(30),
        checkpoint_out: None,
        resume: config.resume_checkpoint.clone(),
        checkpoint_every_steps: config.checkpoint_every_n_steps,
        log_every_steps: config.log_every_n_steps,
        keep_step_checkpoints: config.keep_step_checkpoints,
        tensorboard: config.tensorboard,
        launch_tensorboard: config.launch_tensorboard,
        tensorboard_host: config.tensorboard_host.clone(),
        tensorboard_port: config.tensorboard_port,
        background: config.background,
        learning_rate: config.bc.learning_rate,
        weight_decay: f64::from(config.bc.weight_decay),
        compile_fullgraph_check: false,
        oracle_critic_weight: 0.0,
        safety_residual_weight: advanced
            .and_then(|loss| loss.safety_residual)
            .unwrap_or(0.0)
            .into(),
    })
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
