pub(crate) use hydra_train_runtime::config_runtime::*;

use hydra_train::model::HydraModelConfig;
use hydra_train::training::bc::BCTrainerConfig;
use hydra_train::training::rl::{DEFAULT_RL_MICROBATCH_SIZE, RlConfig};

use super::config::{RlTrainConfig, TrainConfig};

pub(crate) fn trainer_config_from_train_config(config: &TrainConfig) -> BCTrainerConfig {
    BCTrainerConfig::new(HydraModelConfig::learner())
        .with_batch_size(config.batch_size)
        .with_lr(config.bc.learning_rate)
        .with_min_learning_rate(config.bc.min_learning_rate)
        .with_weight_decay(config.bc.weight_decay)
        .with_grad_clip_norm(config.bc.grad_clip_norm)
        .with_warmup_steps(config.bc.warmup_steps)
}

pub(crate) fn rl_config_from_train_config(rl: &RlTrainConfig) -> RlConfig {
    let mut cfg = match rl.phase {
        super::config::RlPhaseConfig::DrdaAchSelfPlay => RlConfig::default_phase2(),
        super::config::RlPhaseConfig::ExitPondering => RlConfig::default_phase3(),
    };
    if let Some(lr) = rl.learning_rate {
        cfg = cfg.with_lr(lr);
    }
    if let Some(exit_weight) = rl.exit_weight {
        cfg = cfg.with_exit_weight(exit_weight);
    }
    if let Some(aux_weight) = rl.aux_weight {
        cfg = cfg.with_aux_weight(aux_weight);
    }
    cfg.microbatch_size = Some(rl.microbatch_size.unwrap_or(DEFAULT_RL_MICROBATCH_SIZE));
    cfg
}
