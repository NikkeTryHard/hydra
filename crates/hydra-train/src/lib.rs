//! Hydra Training Pipeline
//!
//! SE-ResNet backbone with Hydra's current train/data/runtime-support surfaces.
//! This crate contains both the active shipped baseline and some staged or reserve modules.
//! Treat [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) as the authority for
//! shipped-vs-staged status rather than assuming every exported module is part of the
//! current active baseline.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

pub mod config;
pub mod data;
pub mod eval;
pub mod teacher;
pub mod training;
