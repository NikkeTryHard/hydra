//! Backward-compatible RL training-step exports.

pub use hydra_train_exec::rl_step::*;
pub use hydra_train_types::config::RlConfig;
pub use hydra_train_types::config::{
    DEFAULT_AUX_WEIGHT, DEFAULT_EXIT_WEIGHT, DEFAULT_RL_MICROBATCH_SIZE, GAE_GAMMA, GAE_LAMBDA,
};
