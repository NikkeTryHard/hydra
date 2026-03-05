//! Hydra Training Pipeline
//!
//! SE-ResNet backbone with ACH training, ExIt search targets, and DRDA wrapping.
//! Implements the full training pipeline from HYDRA_FINAL.md.

pub mod backbone;
pub mod config;
pub mod heads;
pub mod model;
