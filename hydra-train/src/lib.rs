//! Hydra Training Pipeline
//!
//! SE-ResNet backbone with ACH training, ExIt search targets, and DRDA wrapping.
//! Implements the full training pipeline from HYDRA_FINAL.md.

pub mod backbone;
pub mod config;
pub mod data;
pub mod eval;
pub mod heads;
pub mod inference;
pub mod league;
pub mod model;
pub mod saf;
pub mod selfplay;
pub mod teacher;
pub mod training;
