//! Shared training data-transfer types for Hydra training crates.
//!
//! This crate owns backend-independent scalar gate/configuration types and the
//! Burn tensor target/config types that define loss inputs across training
//! crates. It remains independent of `hydra-train` runtime, model, loss-builder,
//! and orchestration code so it can sit below `hydra-train` in the dependency
//! graph without creating cycles.

pub mod checkpoint;
pub mod config;
pub mod delta_q_promotion;
pub mod eval;
pub mod head_gates;
pub mod losses;
pub mod orchestrator;
pub mod phase;
pub mod rl;
pub mod selfplay;
