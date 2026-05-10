//! Runtime configuration, CLI parsing, probe request, and preflight contracts for Hydra training.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

pub mod config;
pub mod config_runtime;
pub mod data;
pub mod exit;
pub mod gpu_config;
pub mod loss_policy;
pub mod losses;
pub mod model;
pub mod nvtx;
pub mod preflight;
pub mod probe_request;
pub mod schedule;
pub mod status;

#[cfg(test)]
mod test_support;
