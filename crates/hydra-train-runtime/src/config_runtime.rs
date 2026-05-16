use std::thread::available_parallelism;

use crate::preflight::{LoaderRuntimeConfig, PreflightTuningMode};

use super::config::{RlTrainConfig, TrainConfig};

pub fn default_num_threads_for_system() -> usize {
    available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .max(1)
}

pub fn resolved_num_threads(num_threads: Option<usize>) -> Result<usize, String> {
    if let Some(num_threads) = num_threads {
        if num_threads == 0 {
            return Err("num_threads must be greater than 0".to_string());
        }
        return Ok(num_threads);
    }
    Ok(default_num_threads_for_system())
}

pub fn display_num_threads(num_threads: Option<usize>) -> String {
    match num_threads {
        Some(value) => value.to_string(),
        None => format!("{} (auto)", default_num_threads_for_system()),
    }
}

pub fn validate_config(config: &TrainConfig) -> Result<(), String> {
    if config.num_epochs == 0 {
        return Err("num_epochs must be greater than 0".to_string());
    }
    if config.batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }
    if config.buffer_games == 0 {
        return Err("buffer_games must be greater than 0".to_string());
    }
    if config.buffer_samples == 0 {
        return Err("buffer_samples must be greater than 0".to_string());
    }
    if config.archive_queue_bound == 0 {
        return Err("archive_queue_bound must be greater than 0".to_string());
    }
    if let Some(shard_prefetch_depth) = config.shard_prefetch_depth {
        if shard_prefetch_depth == 0 {
            return Err("shard_prefetch_depth must be greater than 0".to_string());
        }
        if shard_prefetch_depth > 64 {
            return Err("shard_prefetch_depth must be at most 64".to_string());
        }
    }
    if config.validation_every_n_epochs == 0 {
        return Err("validation_every_n_epochs must be greater than 0".to_string());
    }
    if config.log_every_n_steps == 0 {
        return Err("log_every_n_steps must be greater than 0".to_string());
    }
    if config.validate_every_n_steps == 0 {
        return Err("validate_every_n_steps must be greater than 0".to_string());
    }
    if config.checkpoint_every_n_steps == 0 {
        return Err("checkpoint_every_n_steps must be greater than 0".to_string());
    }
    if let Some(max_train_steps) = config.max_train_steps
        && max_train_steps == 0
    {
        return Err("max_train_steps must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_batches) = config.max_validation_batches
        && max_validation_batches == 0
    {
        return Err("max_validation_batches must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_samples) = config.max_validation_samples
        && max_validation_samples == 0
    {
        return Err("max_validation_samples must be greater than 0 when set".to_string());
    }
    if config.validation_gates.enabled {
        if config.validation_gates.min_validation_samples == Some(0) {
            return Err(
                "validation_gates.min_validation_samples must be greater than 0 when set"
                    .to_string(),
            );
        }
        if config
            .validation_gates
            .max_policy_loss_regression
            .is_some_and(|value| value < 0.0)
        {
            return Err(
                "validation_gates.max_policy_loss_regression must be non-negative when set"
                    .to_string(),
            );
        }
        if config
            .validation_gates
            .min_policy_agreement_delta
            .is_some_and(|value| value < 0.0)
        {
            return Err(
                "validation_gates.min_policy_agreement_delta must be non-negative when set"
                    .to_string(),
            );
        }
    }
    if let Some(microbatch_size) = config.microbatch_size
        && microbatch_size == 0
    {
        return Err("microbatch_size must be greater than 0".to_string());
    }
    if let Some(validation_microbatch_size) = config.validation_microbatch_size
        && validation_microbatch_size == 0
    {
        return Err("validation_microbatch_size must be greater than 0".to_string());
    }
    match config.preflight.tuning_mode {
        PreflightTuningMode::Safe => {
            if !config.preflight.unsafe_candidate_batch_sizes.is_empty() {
                return Err(
                    "preflight.unsafe_candidate_batch_sizes requires preflight.tuning_mode = unsafe"
                        .to_string(),
                );
            }
            if !config.preflight.unsafe_candidate_lr_scales.is_empty() {
                return Err(
                    "preflight.unsafe_candidate_lr_scales requires preflight.tuning_mode = unsafe"
                        .to_string(),
                );
            }
            if !config.preflight.unsafe_candidate_warmup_steps.is_empty() {
                return Err(
                    "preflight.unsafe_candidate_warmup_steps requires preflight.tuning_mode = unsafe"
                        .to_string(),
                );
            }
        }
        PreflightTuningMode::Unsafe => {
            if config.preflight.unsafe_candidate_batch_sizes.is_empty() {
                return Err(
                    "preflight.unsafe_candidate_batch_sizes must be non-empty when preflight.tuning_mode = unsafe"
                        .to_string(),
                );
            }
            if config.preflight.unsafe_candidate_batch_sizes.contains(&0) {
                return Err(
                    "preflight.unsafe_candidate_batch_sizes entries must be greater than 0"
                        .to_string(),
                );
            }
            if config
                .preflight
                .unsafe_candidate_lr_scales
                .iter()
                .any(|candidate| !candidate.is_finite() || *candidate <= 0.0)
            {
                return Err(
                    "preflight.unsafe_candidate_lr_scales entries must be finite and greater than 0"
                        .to_string(),
                );
            }
            if config.preflight.unsafe_candidate_warmup_steps.contains(&0) {
                return Err(
                    "preflight.unsafe_candidate_warmup_steps entries must be greater than 0"
                        .to_string(),
                );
            }
        }
    }
    if config.bc.learning_rate <= 0.0 {
        return Err("bc.learning_rate must be greater than 0".to_string());
    }
    if config.bc.min_learning_rate <= 0.0 {
        return Err("bc.min_learning_rate must be greater than 0".to_string());
    }
    if config.bc.min_learning_rate > config.bc.learning_rate {
        return Err(
            "bc.min_learning_rate must be less than or equal to bc.learning_rate".to_string(),
        );
    }
    if config.bc.weight_decay < 0.0 {
        return Err("bc.weight_decay must be non-negative".to_string());
    }
    if config.bc.grad_clip_norm <= 0.0 {
        return Err("bc.grad_clip_norm must be greater than 0".to_string());
    }
    if config.bc.warmup_steps == 0 {
        return Err("bc.warmup_steps must be greater than 0".to_string());
    }
    if matches!(
        config.precision_mode,
        crate::config::PrecisionMode::Bf16Autocast
    ) {
        if config.rl.is_some() {
            return Err(
                "precision_mode=bf16_autocast is not supported for RL training yet".to_string(),
            );
        }
        if config
            .advanced_loss
            .as_ref()
            .and_then(|loss| loss.delta_q)
            .is_some_and(|weight| weight > 0.0)
        {
            return Err(
                "precision_mode=bf16_autocast is not supported for DeltaQ training yet".to_string(),
            );
        }
    }
    if let Some(rl) = config.rl.as_ref() {
        validate_rl_config(rl)?;
    }
    if config
        .advanced_loss
        .as_ref()
        .and_then(|loss| loss.exit)
        .is_some_and(|weight| weight > 0.0)
        && config.exit_sidecar_path.is_none()
    {
        return Err(
            "advanced_loss.exit requires exit_sidecar_path so replay ExIt labels are present"
                .to_string(),
        );
    }
    if config
        .advanced_loss
        .as_ref()
        .and_then(|loss| loss.delta_q)
        .is_some_and(|weight| weight > 0.0)
        && config.delta_q_sidecar_path.is_none()
    {
        return Err(
            "advanced_loss.delta_q requires delta_q_sidecar_path so replay delta_q labels are present"
                .to_string(),
        );
    }
    Ok(())
}

pub fn validate_rl_config(rl: &RlTrainConfig) -> Result<(), String> {
    if rl.games_per_batch == 0 {
        return Err("rl.games_per_batch must be greater than 0".to_string());
    }
    if rl.temperature <= 0.0 {
        return Err("rl.temperature must be greater than 0".to_string());
    }
    if let Some(lr) = rl.learning_rate
        && lr <= 0.0
    {
        return Err("rl.learning_rate must be greater than 0 when set".to_string());
    }
    if let Some(exit_weight) = rl.exit_weight
        && exit_weight < 0.0
    {
        return Err("rl.exit_weight must be non-negative".to_string());
    }
    if let Some(aux_weight) = rl.aux_weight
        && aux_weight < 0.0
    {
        return Err("rl.aux_weight must be non-negative".to_string());
    }
    Ok(())
}

pub fn loader_runtime_config(config: &TrainConfig) -> LoaderRuntimeConfig {
    LoaderRuntimeConfig {
        num_threads: Some(default_num_threads_for_system())
            .filter(|_| config.num_threads.is_none())
            .or(config.num_threads),
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        archive_queue_bound: config.archive_queue_bound,
    }
}

pub fn train_microbatch_size(config: &TrainConfig) -> usize {
    config.microbatch_size.unwrap_or(config.batch_size)
}

pub fn validation_microbatch_size(config: &TrainConfig) -> usize {
    config
        .validation_microbatch_size
        .unwrap_or_else(|| train_microbatch_size(config))
}

pub fn validation_sample_limit(config: &TrainConfig) -> Option<usize> {
    config.max_validation_samples.or_else(|| {
        config
            .max_validation_batches
            .map(|limit| limit.saturating_mul(validation_microbatch_size(config)))
    })
}

pub fn shard_prefetch_depth(config: &TrainConfig) -> usize {
    config
        .shard_prefetch_depth
        .unwrap_or_else(super::config::default_shard_prefetch_depth)
}

#[cfg(test)]
mod tests;
