//! Runtime configuration, CLI parsing, probe request, and preflight contracts for Hydra training.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

pub mod config;
pub mod config_runtime;
pub mod delta_q_promotion;
pub mod head_gates;
pub mod loss_policy;
pub mod preflight;
pub mod probe_request;
pub mod progress;
pub mod schedule;
pub mod status;
pub mod timing_metrics;
pub mod validation;

#[cfg(test)]
mod test_support;
