//! Command construction for Python learner entrypoints.

use hydra_train_runtime::config::{
    PythonLearnerCliOptions, PythonLearnerInput, PythonPpoControlCliOptions,
};

use super::PythonLearnerCommand;
use super::tensorboard::tensorboard_url;
use std::path::Path;

pub(crate) const PYTHON_LEARNER_SCRIPT: &str = "scripts/hydra_pytorch_oracle.py";
const PYTHON_LEARNER_RESULT: &str = "python_learner_result.json";
pub(crate) const PYTHON_PPO_CONTROL_RESULT: &str = "ppo_control_result.json";

/// Builds the strict Pixi/Python command for the Python BC learner.
pub fn build_python_learner_command(options: &PythonLearnerCliOptions) -> PythonLearnerCommand {
    build_python_learner_command_for_run_dir(options, &options.output_dir)
}

pub(crate) fn build_python_learner_command_for_run_dir(
    options: &PythonLearnerCliOptions,
    run_dir: &Path,
) -> PythonLearnerCommand {
    let result_path = run_dir.join(PYTHON_LEARNER_RESULT);
    let mut args = vec![
        "run".to_string(),
        "-e".to_string(),
        "py-train".to_string(),
        "python".to_string(),
        PYTHON_LEARNER_SCRIPT.to_string(),
        "--variant".to_string(),
        options.variant.as_str().to_string(),
        "--residual-profile".to_string(),
        options.residual_profile.as_str().to_string(),
        "--conv-memory-format".to_string(),
        options.conv_memory_format.as_str().to_string(),
        "--backbone-profile".to_string(),
        options.backbone_profile.as_str().to_string(),
        "--hidden".to_string(),
        options.hidden.to_string(),
        "--blocks".to_string(),
        options.blocks.to_string(),
        "--bottleneck".to_string(),
        options.bottleneck.to_string(),
        "--batch".to_string(),
        options.batch_size.to_string(),
        "--microbatch".to_string(),
        options.microbatch_size.to_string(),
        "--warmup".to_string(),
        options.warmup_steps.to_string(),
    ];
    if let Some(steps) = options.steps {
        args.push("--steps".to_string());
        args.push(steps.to_string());
    }
    args.extend([
        "--out".to_string(),
        result_path.display().to_string(),
        "--quiet".to_string(),
        "--w-oracle-critic".to_string(),
        options.oracle_critic_weight.to_string(),
        "--w-safety-residual".to_string(),
        options.safety_residual_weight.to_string(),
        "--w-exit".to_string(),
        options.exit_weight.to_string(),
        "--w-deltaq".to_string(),
        options.deltaq_weight.to_string(),
        "--lr".to_string(),
        options.learning_rate.to_string(),
        "--min-lr".to_string(),
        options.min_learning_rate.to_string(),
        "--lr-warmup-steps".to_string(),
        options.lr_warmup_steps.to_string(),
        "--lr-schedule".to_string(),
        options.lr_schedule.clone(),
        "--grad-clip-norm".to_string(),
        options.grad_clip_norm.to_string(),
        "--weight-decay".to_string(),
        options.weight_decay.to_string(),
        "--adamw-fused".to_string(),
        options.adamw_fused.as_str().to_string(),
        "--adamw-foreach".to_string(),
        options.adamw_foreach.as_str().to_string(),
    ]);
    if options.ema_enabled {
        args.extend([
            "--ema-enabled".to_string(),
            "--ema-decay".to_string(),
            options.ema_decay.to_string(),
            "--ema-start-step".to_string(),
            options.ema_start_step.to_string(),
            "--ema-update-every-steps".to_string(),
            options.ema_update_every_steps.to_string(),
            "--ema-device".to_string(),
            options.ema_device.as_str().to_string(),
        ]);
    }
    if let Some(total_steps) = options.schedule_total_steps {
        args.push("--schedule-total-steps".to_string());
        args.push(total_steps.to_string());
    }
    if let Some(target_games) = options.schedule_target_games {
        args.push("--schedule-target-games".to_string());
        args.push(target_games.to_string());
    }
    if options.full_epoch {
        args.push("--full-epoch".to_string());
    }
    match &options.input {
        PythonLearnerInput::BcShards { manifest } => {
            args.push("--manifest".to_string());
            args.push(manifest.display().to_string());
        }
        PythonLearnerInput::RawMjai {
            data_dirs,
            max_games,
            max_samples,
            skip_games,
            train_fraction,
            augment,
            transport,
        } => {
            for data_dir in data_dirs {
                args.push("--raw-mjai-data-dir".to_string());
                args.push(data_dir.display().to_string());
            }
            args.push("--raw-mjai-worker-threads".to_string());
            args.push("20".to_string());
            args.push("--raw-mjai-train-fraction".to_string());
            args.push(train_fraction.to_string());
            args.push("--raw-mjai-transport".to_string());
            args.push(transport.as_str().to_string());
            if let Some(max_games) = max_games {
                args.push("--raw-mjai-max-games".to_string());
                args.push(max_games.to_string());
            }
            if let Some(max_samples) = max_samples {
                args.push("--raw-mjai-max-samples".to_string());
                args.push(max_samples.to_string());
            }
            if *skip_games != 0 {
                args.push("--raw-mjai-skip-games".to_string());
                args.push(skip_games.to_string());
            }
            if *augment {
                args.push("--raw-mjai-augment".to_string());
            }
        }
    }
    if options.compile_fullgraph_check {
        args.push("--compile-fullgraph-check".to_string());
    }
    let checkpoint_out = options
        .checkpoint_out
        .as_ref()
        .cloned()
        .unwrap_or_else(|| run_dir.join("checkpoints/latest.pt"));
    args.push("--checkpoint-out".to_string());
    args.push(checkpoint_out.display().to_string());
    if let Some(path) = options.resume.as_ref() {
        args.push("--resume".to_string());
        args.push(path.display().to_string());
    }
    if options.checkpoint_every_steps != 0 {
        args.push("--checkpoint-every-steps".to_string());
        args.push(options.checkpoint_every_steps.to_string());
    }
    args.push("--log-dir".to_string());
    args.push(run_dir.join("logs").display().to_string());
    args.push("--log-every-steps".to_string());
    args.push(options.log_every_steps.to_string());
    if options.validation_steps != 0 && options.validation_every != 0 {
        args.push("--validation-steps".to_string());
        args.push(options.validation_steps.to_string());
        if let Some(max_samples) = options.validation_max_samples {
            args.push("--validation-max-samples".to_string());
            args.push(max_samples.to_string());
        }
        args.push("--validation-every".to_string());
        args.push(options.validation_every.to_string());
        args.push("--validation-source-mode".to_string());
        args.push(options.validation_source_mode.clone());
        if options.raw_mjai_validation_augment {
            args.push("--raw-mjai-validation-augment".to_string());
        }
    }
    if options.tensorboard {
        args.push("--tensorboard-dir".to_string());
        args.push(run_dir.join("tensorboard").display().to_string());
        args.push("--tensorboard-url".to_string());
        args.push(tensorboard_url(options, options.tensorboard_port));
    }
    PythonLearnerCommand {
        program: "pixi".to_string(),
        args,
        result_path,
    }
}

/// Builds the strict Pixi/Python command for the Python T1 PPO-control learner.
pub fn build_python_ppo_control_command(
    options: &PythonPpoControlCliOptions,
) -> PythonLearnerCommand {
    build_python_ppo_control_command_for_run_dir(options, &options.output_dir)
}

pub(crate) fn build_python_ppo_control_command_for_run_dir(
    options: &PythonPpoControlCliOptions,
    run_dir: &Path,
) -> PythonLearnerCommand {
    let result_path = run_dir.join(PYTHON_PPO_CONTROL_RESULT);
    let mut args = vec![
        "run".to_string(),
        "-e".to_string(),
        "py-train".to_string(),
        "python".to_string(),
        PYTHON_LEARNER_SCRIPT.to_string(),
        "ppo-control".to_string(),
        "--init-checkpoint".to_string(),
        options.init_checkpoint.display().to_string(),
        "--out".to_string(),
        run_dir.display().to_string(),
        "--steps".to_string(),
        options.steps.to_string(),
        "--games-per-update".to_string(),
        options.games_per_update.to_string(),
        "--seed".to_string(),
        options.seed.to_string(),
        "--device".to_string(),
        options.device.clone(),
        "--temperature".to_string(),
        options.temperature.to_string(),
        "--arena-batch-decisions".to_string(),
        options.arena_batch_decisions.to_string(),
        "--arena-threads".to_string(),
        options.arena_threads.to_string(),
        "--hidden".to_string(),
        options.hidden.to_string(),
        "--blocks".to_string(),
        options.blocks.to_string(),
        "--bottleneck".to_string(),
        options.bottleneck.to_string(),
        "--residual-profile".to_string(),
        options.residual_profile.as_str().to_string(),
        "--conv-memory-format".to_string(),
        options.conv_memory_format.as_str().to_string(),
        "--backbone-profile".to_string(),
        options.backbone_profile.as_str().to_string(),
        "--lr".to_string(),
        options.learning_rate.to_string(),
        "--min-lr".to_string(),
        options.min_learning_rate.to_string(),
        "--lr-warmup-steps".to_string(),
        options.lr_warmup_steps.to_string(),
        "--grad-clip-norm".to_string(),
        options.grad_clip_norm.to_string(),
        "--weight-decay".to_string(),
        options.weight_decay.to_string(),
        "--adamw-fused".to_string(),
        options.adamw_fused.as_str().to_string(),
        "--adamw-foreach".to_string(),
        options.adamw_foreach.as_str().to_string(),
        "--bc-kl-reverse-coef".to_string(),
        options.bc_kl_reverse_coef.to_string(),
        "--checkpoint-every-steps".to_string(),
        options.checkpoint_every_steps.to_string(),
        "--log-every-steps".to_string(),
        options.log_every_steps.to_string(),
        "--quiet".to_string(),
    ];
    if let Some(resume) = options.resume.as_ref() {
        args.push("--resume".to_string());
        args.push(resume.display().to_string());
    }
    if options.keep_step_checkpoints {
        args.push("--keep-step-checkpoints".to_string());
    }
    if options.tensorboard {
        args.push("--tensorboard-dir".to_string());
        args.push(run_dir.join("tensorboard").display().to_string());
    }
    PythonLearnerCommand {
        program: "pixi".to_string(),
        args,
        result_path,
    }
}
