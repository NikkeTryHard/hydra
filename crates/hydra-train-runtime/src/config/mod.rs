use std::ffi::OsStr;
use std::fs;
use std::path::Path;

pub use hydra_data_core::SourceFilterConfig;
pub use hydra_train_types::config::{BackboneActivationConfig, BackboneNormConfig};

pub use super::config_runtime::{
    default_num_threads_for_system, display_num_threads, loader_runtime_config,
    shard_prefetch_depth, train_microbatch_size, validate_config, validate_preflight_config,
    validation_microbatch_size, validation_sample_limit,
};

mod cli;
mod cli_types;
pub mod profiles;
pub mod python;
pub mod schema;
pub mod stages;

pub use cli::{parse_args, usage, version};
pub use cli_types::{
    BcBackend, BenchmarkBaselineCliOptions, BenchmarkBaselineSource, ExperimentalTrainBackend,
    PreflightCliOptions, PreflightProfile, ProbeBatchChildRequest, ProbeChildRequest,
    ProbeCliRequest, ProbeSingleChildRequest, PythonLearnerCliOptions, PythonLearnerInput,
    PythonPpoControlCliOptions, TrainCli,
};
pub use profiles::{
    BcBackendConfig, BcHeadProfile, ExperimentalBackboneProfileConfig, PythonAdamwFlagConfig,
    PythonBackboneProfileConfig, PythonConvMemoryFormatConfig, PythonLearnerVariant,
    PythonModelProfileConfig, PythonRawMjaiTransportConfig, PythonResidualProfileConfig,
};
pub use python::{
    PYTHON_TIMING_WARMUP_STEPS, python_options_from_config, python_ppo_control_options_from_config,
    python_resume_checkpoint, python_run_dir, raw_mjai_cursor_resume_supported,
};
pub use schema::{
    AdvancedLossConfig, BcHyperparamConfig, EffectivePrecision, EmaConfig, EmaDeviceConfig,
    NsightTraceConfig, PrecisionMode, RlPhaseConfig, RlTrainConfig, TrainConfig,
    ValidationGateConfig, default_archive_queue_bound, default_augment,
    default_backbone_se_every_n, default_batch_size, default_bc_grad_clip_norm,
    default_bc_learning_rate, default_bc_min_learning_rate, default_bc_warmup_steps,
    default_bc_weight_decay, default_buffer_games, default_buffer_samples,
    default_checkpoint_every_n_steps, default_device, default_ema_decay, default_ema_enabled,
    default_ema_update_every_steps, default_log_every_n_steps, default_max_skip_logs_per_source,
    default_max_validation_samples, default_preflight_config_for_profile, default_resume_latest,
    default_rl_games_per_batch, default_rl_phase, default_rl_temperature, default_seed,
    default_shard_prefetch_depth, default_tensorboard, default_tensorboard_host,
    default_tensorboard_port, default_train_fraction, default_validate_every_n_steps,
    default_validation_every_n_epochs,
};
pub use stages::{
    DEFAULT_BC_STAGE, DEFAULT_PPO_STAGE, T0_BC_STAGE, T1_PPO_CONTROL_STAGE,
    T2_DIRECT_SAMPLED_ACH_STAGE, T3_DRDA_RESIDUAL_ACH_STAGE, T4_PBRS_BETA_SWEEP_STAGE,
    T5_EXIT_AUXILIARY_STAGE, T6_DELTAQ_EXPERIMENT_STAGE, T7_POPULATION_WINDOW_STAGE,
    rl_stage_for_config, rl_stage_for_phase,
};

pub fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => {
            let precision_omitted = precision_mode_is_omitted(&raw)
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))?;
            serde_yaml::from_str::<TrainConfig>(&raw)
                .map(|config| resolve_omitted_precision_mode(config, precision_omitted))
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))
        }
        _ => Err(format!(
            "unsupported config extension for {}; use .yaml",
            path.display()
        )),
    }
}

fn resolve_omitted_precision_mode(mut config: TrainConfig, precision_omitted: bool) -> TrainConfig {
    if config.precision_mode == PrecisionMode::Fp32 && precision_omitted {
        let delta_q_active = config
            .advanced_loss
            .as_ref()
            .and_then(|loss| loss.delta_q)
            .is_some_and(|weight| weight > 0.0);
        if config.device.starts_with("cuda") && config.rl.is_none() && !delta_q_active {
            config.precision_mode = PrecisionMode::Bf16Autocast;
        }
    }
    config
}

fn precision_mode_is_omitted(raw: &str) -> Result<bool, serde_yaml::Error> {
    match serde_yaml::from_str::<serde_yaml::Value>(raw)? {
        serde_yaml::Value::Mapping(mapping) => {
            Ok(!mapping.contains_key(serde_yaml::Value::String("precision_mode".to_string())))
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
