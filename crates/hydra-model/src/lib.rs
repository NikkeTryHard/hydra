//! Neural model components for Hydra.
//!
//! This crate owns the pure Burn model building blocks used by `hydra-train`.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

/// Automatic mixed precision compatibility helpers.
pub mod amp;
/// SE-ResNet backbone modules.
pub mod backbone;
/// Model output head modules.
pub mod heads;
/// Inference server and policy-selection utilities.
pub mod inference;
/// Full Hydra model and forward DTOs.
pub mod model;
/// Search-as-Feature adaptor modules.
pub mod saf;
