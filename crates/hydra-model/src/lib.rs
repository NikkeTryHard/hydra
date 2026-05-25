//! Burn neural model components for Hydra's legacy/reference Rust path.
//!
//! Default plain BC model/loss/checkpoint ownership lives in Python/PyTorch.
//! This crate remains the Burn implementation used for Rust fallback/debug and
//! advanced lanes such as ExIt, DeltaQ, belief, mixture, opponent hand-type,
//! safety, oracle, and search-as-feature experiments.
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
mod native_group_norm_mish;
/// ONNX Runtime policy inference path.
pub mod onnx_policy;
/// ONNX Runtime initialization helpers.
pub mod ort_init;
/// Model-local profiling helpers.
mod profiling;
/// Search-as-Feature adaptor modules.
pub mod saf;
