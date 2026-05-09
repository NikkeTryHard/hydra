//! Scalar training coordination types shared by Hydra training crates.
//!
//! This crate intentionally stays free of tensor backends, loss builders, and
//! runtime training orchestration so it can sit below `hydra-train` in the
//! dependency graph without creating cycles.

pub mod head_gates;
