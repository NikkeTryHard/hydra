//! Belief-state search components for Hydra.
//!
//! This crate owns search-side support modules that must remain independent of
//! encoder implementation details.

pub mod afbs;
pub mod ct_smc;
pub mod endgame;
pub mod hand_ev;
pub mod robust_opponent;
pub mod shanten_batch;
pub mod sinkhorn;
