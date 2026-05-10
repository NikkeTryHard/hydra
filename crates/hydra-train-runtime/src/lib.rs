//! Runtime configuration, CLI parsing, probe request, and preflight contracts for Hydra training.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

pub mod bc_fixed_shape;
pub mod bc_metrics;
pub mod bc_runtime;
pub mod config;
pub mod config_runtime;
pub mod data;
pub mod delta_q_promotion;
pub mod exit;
pub mod gpu_config;
pub mod head_gates;
pub mod loss_policy;
pub mod losses;
pub mod model;
pub mod nvtx;
pub mod preflight;
pub mod probe_request;
pub mod progress;
pub mod schedule;
pub mod status;
pub mod validation;

#[cfg(test)]
mod test_support;
