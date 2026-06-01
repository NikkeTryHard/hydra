use std::path::{Path, PathBuf};

use super::{
    DEFAULT_BC_STAGE, PythonLearnerCliOptions, PythonLearnerInput, PythonPpoControlCliOptions,
    RlPhaseConfig, TrainConfig, rl_stage_for_config, validation_sample_limit,
};

/// Warmup steps used by train-config-driven Python timing launches.
pub const PYTHON_TIMING_WARMUP_STEPS: usize = 10;

/// Raw-MJAI launches restore by skipping deterministic completed games from checkpoint progress.
pub const fn raw_mjai_cursor_resume_supported() -> bool {
    true
}

/// Resolves the Python run directory for a config-owned launch.
pub fn python_run_dir(config: &TrainConfig, default_stage: &str) -> PathBuf {
    let stage = config.stage.as_deref().unwrap_or(default_stage);
    let run_name = config.run_name.as_deref().unwrap_or("latest_run");
    config
        .output_dir
        .join("stages")
        .join(stage)
        .join("runs")
        .join(run_name)
}
pub fn python_resume_checkpoint(config: &TrainConfig) -> Result<Option<PathBuf>, String> {
    python_resume_checkpoint_for_stage(config, DEFAULT_BC_STAGE)
}

fn python_resume_checkpoint_for_stage(
    config: &TrainConfig,
    default_stage: &str,
) -> Result<Option<PathBuf>, String> {
    let latest = python_run_dir(config, default_stage).join("checkpoints/latest.pt");

    if let Some(path) = &config.resume_checkpoint {
        return Ok(Some(path.clone()));
    }
    if config.resume_latest && latest.is_file() {
        return Ok(Some(latest));
    }
    Ok(None)
}

/// Converts YAML-owned train config into Python learner CLI options.
pub fn python_options_from_config(config: &TrainConfig) -> Result<PythonLearnerCliOptions, String> {
    validate_python_advanced_loss_guards(config)?;
    let advanced = config.advanced_loss.as_ref();
    let schedule_total_steps = python_schedule_total_steps(config)?;
    Ok(PythonLearnerCliOptions {
        bc_shards_manifest: config
            .bc_shards_manifest_path
            .clone()
            .unwrap_or_else(|| config.data_dir.clone()),
        input: if let Some(manifest) = config.bc_shards_manifest_path.clone() {
            PythonLearnerInput::BcShards { manifest }
        } else {
            PythonLearnerInput::RawMjai {
                data_dirs: if config.raw_mjai_data_dirs.is_empty() {
                    vec![config.data_dir.clone()]
                } else {
                    config.raw_mjai_data_dirs.clone()
                },
                max_games: None,
                max_samples: None,
                skip_games: 0,
                train_fraction: config.train_fraction,
                augment: config.augment,
                transport: config.python_raw_mjai_transport,
            }
        },
        output_dir: python_run_dir(config, DEFAULT_BC_STAGE),
        stage: config.stage.clone(),
        run_name: config.run_name.clone(),
        device: config.device.clone(),
        batch_size: config.batch_size,
        microbatch_size: config.microbatch_size.unwrap_or(1024),
        variant: config.python_variant,
        residual_profile: config.python_residual_profile,
        conv_memory_format: config.python_conv_memory_format,
        backbone_profile: config.python_backbone_profile,
        hidden: config.python_model_profile.hidden(),
        blocks: config.python_model_profile.blocks(),
        bottleneck: config.python_model_profile.bottleneck(),
        warmup_steps: PYTHON_TIMING_WARMUP_STEPS,
        steps: config.max_train_steps,
        full_epoch: config.full_epoch,
        validation_steps: validation_sample_limit(config)
            .unwrap_or(0)
            .div_ceil(config.batch_size),
        validation_max_samples: config.max_validation_samples,
        validation_every: config.validate_every_n_steps,
        raw_mjai_validation_augment: false,
        validation_source_mode: "fixed".to_string(),
        checkpoint_out: None,
        resume: python_resume_checkpoint(config)?,
        checkpoint_every_steps: config.checkpoint_every_n_steps,
        log_every_steps: config.log_every_n_steps,
        keep_step_checkpoints: config.keep_step_checkpoints,
        tensorboard: config.tensorboard,
        launch_tensorboard: config.launch_tensorboard,
        tensorboard_host: config.tensorboard_host.clone(),
        tensorboard_port: config.tensorboard_port,
        background: config.background,
        learning_rate: config.bc.learning_rate,
        min_learning_rate: config.bc.min_learning_rate,
        lr_warmup_steps: config.bc.warmup_steps,
        lr_schedule: if config.full_epoch
            && config.max_train_steps.is_none()
            && schedule_total_steps.is_none()
            && config.python_raw_mjai_target_games.is_none()
        {
            "constant".to_string()
        } else {
            "cosine".to_string()
        },
        schedule_total_steps,
        schedule_target_games: if config.max_train_steps.is_none() {
            config.python_raw_mjai_target_games
        } else {
            None
        },
        grad_clip_norm: f64::from(config.bc.grad_clip_norm),
        weight_decay: f64::from(config.bc.weight_decay),
        ema_enabled: config.ema.enabled,
        ema_decay: config.ema.decay,
        ema_start_step: config.ema.start_step,
        ema_update_every_steps: config.ema.update_every_steps,
        ema_device: config.ema.device,
        adamw_fused: config.bc.adamw_fused,
        adamw_foreach: config.bc.adamw_foreach,
        compile_fullgraph_check: false,
        oracle_critic_weight: 0.0,
        safety_residual_weight: advanced
            .and_then(|loss| loss.safety_residual)
            .unwrap_or(0.0)
            .into(),
        exit_weight: advanced.and_then(|loss| loss.exit).unwrap_or(0.0).into(),
        deltaq_weight: advanced.and_then(|loss| loss.delta_q).unwrap_or(0.0).into(),
    })
}

/// Converts YAML-owned train config into Python T1 PPO-control CLI options.
pub fn python_ppo_control_options_from_config(
    config: &TrainConfig,
) -> Result<PythonPpoControlCliOptions, String> {
    let rl = config
        .rl
        .as_ref()
        .ok_or_else(|| "rl config is required for Python PPO control".to_string())?;
    if rl.phase != RlPhaseConfig::PpoControl {
        return Err("rl.phase must be ppo_control for Python PPO control".to_string());
    }
    if config.python_backbone_profile != super::PythonBackboneProfileConfig::Conv2dLocal3 {
        return Err(
            "Python PPO control native rollout requires python_backbone_profile=conv2d_local3"
                .to_string(),
        );
    }
    if let Some(depth) = rl.ppo_pipeline_depth
        && depth > 1
    {
        return Err("rl.ppo_pipeline_depth must be 0 or 1".to_string());
    }
    let steps = if rl.run_forever {
        None
    } else {
        Some(
            config
                .max_train_steps
                .or(Some(config.num_epochs))
                .filter(|steps| *steps > 0)
                .ok_or_else(|| {
                    "max_train_steps or num_epochs must be greater than 0".to_string()
                })?,
        )
    };
    Ok(PythonPpoControlCliOptions {
        init_checkpoint: python_ppo_control_init_checkpoint(config)?,
        output_dir: python_run_dir(config, rl_stage_for_config(config)),
        stage: config.stage.clone(),
        run_name: config.run_name.clone(),
        device: config.device.clone(),
        rollout_device: rl.ppo_rollout_device.clone(),
        steps,
        games_per_update: rl.games_per_batch,
        seed: config.seed,
        temperature: rl.temperature,
        arena_batch_decisions: rl.arena_batch_decisions.unwrap_or(config.batch_size),
        microbatch_size: rl.microbatch_size.unwrap_or(1024),
        epochs: rl.epochs.unwrap_or(1),
        target_kl: rl.target_kl,
        arena_threads: config.num_threads.unwrap_or(0),
        hidden: config.python_model_profile.hidden(),
        blocks: config.python_model_profile.blocks(),
        bottleneck: config.python_model_profile.bottleneck(),
        residual_profile: config.python_residual_profile,
        conv_memory_format: config.python_conv_memory_format,
        backbone_profile: config.python_backbone_profile,
        learning_rate: rl.learning_rate.unwrap_or(config.bc.learning_rate),
        min_learning_rate: config.bc.min_learning_rate,
        lr_warmup_samples: rl.lr_warmup_samples.unwrap_or(1_000_000),
        lr_decay_samples: rl.lr_decay_samples,
        grad_clip_norm: f64::from(config.bc.grad_clip_norm),
        weight_decay: f64::from(config.bc.weight_decay),
        adamw_fused: config.bc.adamw_fused,
        adamw_foreach: config.bc.adamw_foreach,
        bc_kl_reverse_coef: rl.bc_kl_reverse_coef.unwrap_or(0.01),
        resume: None,
        checkpoint_every_steps: config.checkpoint_every_n_steps,
        log_every_steps: config.log_every_n_steps,
        keep_step_checkpoints: config.keep_step_checkpoints,
        tensorboard: config.tensorboard,
        launch_tensorboard: config.launch_tensorboard,
        tensorboard_host: config.tensorboard_host.clone(),
        tensorboard_port: config.tensorboard_port,
        rollout_inference: rl
            .rollout_inference
            .clone()
            .unwrap_or_else(|| "torch-callback".to_owned()),
        ppo_pipeline_depth: rl.ppo_pipeline_depth.unwrap_or(0),
        background: config.background,
    })
}

fn python_ppo_control_init_checkpoint(config: &TrainConfig) -> Result<PathBuf, String> {
    config.resume_checkpoint.clone().ok_or_else(|| {
        "rl.phase=ppo_control requires resume_checkpoint to name the BC/init .pt checkpoint"
            .to_string()
    })
}

fn python_schedule_total_steps(config: &TrainConfig) -> Result<Option<usize>, String> {
    if config.max_train_steps.is_some() || config.bc_shards_manifest_path.is_some() {
        return Ok(config.max_train_steps);
    }
    if config.python_raw_mjai_target_games.is_some() {
        return Ok(None);
    }
    Ok(None)
}

fn validate_python_advanced_loss_guards(config: &TrainConfig) -> Result<(), String> {
    if config.exit_sidecar_path.is_some() && config.bc_shards_manifest_path.is_none() {
        return Err(
            "Python BC learner supports ExIt sidecars only through compact BC shards".to_string(),
        );
    }
    if config.delta_q_sidecar_path.is_some() && config.bc_shards_manifest_path.is_none() {
        return Err(
            "Python BC learner supports DeltaQ sidecars only through compact BC shards".to_string(),
        );
    }
    if let Some(loss) = config.advanced_loss.as_ref() {
        if loss.exit.is_some_and(|weight| weight > 0.0) {
            validate_python_exit_target_contract(config)?;
        }
        if loss.delta_q.is_some_and(|weight| weight > 0.0) {
            if config.delta_q_sidecar_path.is_none() {
                return Err(
                    "advanced_loss.delta_q requires DeltaQ sidecar-backed compact shard labels"
                        .to_string(),
                );
            }
            return Err("delta_q_output_contract_missing".to_string());
        }
        if loss.belief_fields.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.belief_fields".to_string(),
            );
        }
        if loss.mixture_weight.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.mixture_weight".to_string(),
            );
        }
        if loss.opponent_hand_type.is_some_and(|weight| weight > 0.0) {
            return Err(
                "Python BC learner does not support advanced_loss.opponent_hand_type".to_string(),
            );
        }
    }
    Ok(())
}

fn validate_python_exit_target_contract(config: &TrainConfig) -> Result<(), String> {
    let manifest_path = config.bc_shards_manifest_path.as_ref().ok_or_else(|| {
        "advanced_loss.exit requires ExIt sidecar-backed compact shard labels".to_string()
    })?;
    let sidecar_path = config.exit_sidecar_path.as_ref().ok_or_else(|| {
        "advanced_loss.exit requires ExIt sidecar-backed compact shard labels".to_string()
    })?;
    let text = std::fs::read_to_string(manifest_path).map_err(|err| {
        format!("advanced_loss.exit requires readable BC shard manifest provenance: {err}")
    })?;
    let manifest: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        format!("advanced_loss.exit requires valid BC shard manifest provenance: {err}")
    })?;
    let exit_sidecar = manifest
        .get("exit_sidecar")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            "advanced_loss.exit requires manifest exit_sidecar provenance".to_string()
        })?;
    let manifest_sidecar = exit_sidecar
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            "advanced_loss.exit requires manifest exit_sidecar.path provenance".to_string()
        })?;
    if Path::new(manifest_sidecar) != sidecar_path.as_path() {
        return Err(
            "advanced_loss.exit sidecar path must match BC shard manifest exit_sidecar.path"
                .to_string(),
        );
    }
    if exit_sidecar
        .get("source_net_hash")
        .and_then(serde_json::Value::as_u64)
        .is_none()
    {
        return Err(
            "advanced_loss.exit requires manifest exit_sidecar.source_net_hash provenance"
                .to_string(),
        );
    }
    if exit_sidecar
        .get("source_version")
        .and_then(serde_json::Value::as_u64)
        .is_none()
    {
        return Err(
            "advanced_loss.exit requires manifest exit_sidecar.source_version provenance"
                .to_string(),
        );
    }
    Ok(())
}
