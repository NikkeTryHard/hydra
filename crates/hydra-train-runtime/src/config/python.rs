use std::path::PathBuf;

use super::{PythonLearnerCliOptions, PythonLearnerInput, TrainConfig, validation_sample_limit};

/// Warmup steps used by train-config-driven Python timing launches.
pub const PYTHON_TIMING_WARMUP_STEPS: usize = 10;

/// Raw-MJAI launches cannot restore checkpoints until durable stream cursor
/// resume exists. Restoring weights/RNG while replaying corpus prefix is unsafe.
pub const fn raw_mjai_cursor_resume_supported() -> bool {
    false
}

/// Resolves the Python learner checkpoint for a config-owned launch.
pub fn python_resume_checkpoint(config: &TrainConfig) -> Result<Option<PathBuf>, String> {
    let raw_mjai_input = config.bc_shards_manifest_path.is_none();
    let latest = config.output_dir.join("checkpoints/latest.pt");

    if raw_mjai_input && !raw_mjai_cursor_resume_supported() {
        if config.resume_checkpoint.is_some() {
            return Err(
                "Python Raw-MJAI input does not support resume_checkpoint until raw stream cursor resume exists; use a fresh output_dir or BC shards"
                    .to_string(),
            );
        }
        if config.resume_latest {
            return Err(
                "Python Raw-MJAI input does not support resume_latest until raw stream cursor resume exists; use a fresh output_dir or BC shards"
                    .to_string(),
            );
        }
        if latest.is_file() {
            return Err(
                "Python Raw-MJAI input found occupied checkpoints/latest.pt; raw stream cursor resume is unsupported, so choose a fresh output_dir or BC shards"
                    .to_string(),
            );
        }
        return Ok(None);
    }

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
    if config.exit_sidecar_path.is_some() {
        return Err("Python BC learner does not support ExIt sidecars".to_string());
    }
    if config.delta_q_sidecar_path.is_some() {
        return Err("Python BC learner does not support DeltaQ sidecars".to_string());
    }
    if let Some(loss) = config.advanced_loss.as_ref() {
        if loss.exit.is_some_and(|weight| weight > 0.0) {
            return Err("Python BC learner does not support advanced_loss.exit".to_string());
        }
        if loss.delta_q.is_some_and(|weight| weight > 0.0) {
            return Err("Python BC learner does not support advanced_loss.delta_q".to_string());
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
