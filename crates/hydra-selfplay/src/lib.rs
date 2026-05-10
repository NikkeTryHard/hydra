//! Self-play support surfaces split from `hydra-train`.
//!
//! This crate currently owns backend-independent self-play coordination state
//! that is shared by training adapters without depending on `hydra-train`.

pub mod batch;
pub mod cooperative_state;

pub use hydra_train_types::selfplay::{RootDecisionContext, StepRecord};
