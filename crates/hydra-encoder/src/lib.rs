//! Observation encoder components for Hydra.
//!
//! This crate owns encoding modules while preserving Hydra's public action and
//! observation ABI through compatibility re-exports in `hydra-core`.
pub mod batch_encoder;
pub mod encoder;
