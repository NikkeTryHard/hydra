//! Search label generation and validation support for Hydra training.
//!
//! This crate owns ExIt and delta-q search-label algorithms that are shared
//! by self-play and replay sidecar producers without depending on
//! `hydra-train` internals.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

#[cfg(feature = "model-eval")]
/// Delta-q validation reports and step-level collectors.
pub mod delta_q_validation;
/// ExIt target construction, losses, and AFBS tree adapters.
pub mod exit;
#[cfg(feature = "model-eval")]
/// ExIt validation reports and step-level collectors.
pub mod exit_validation;
/// Live/root-decision search-label producers and adapter traits.
pub mod live_exit;
#[cfg(feature = "model-eval")]
/// Replay-indexed offline delta-q sidecar helpers.
pub mod replay_delta_q;
#[cfg(feature = "model-eval")]
/// Replay-indexed offline ExIt sidecar helpers.
pub mod replay_exit;
#[cfg(feature = "model-eval")]
/// Shared validation metric helpers for search-label harnesses.
pub mod validation_common;
