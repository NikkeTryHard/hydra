//! Backward-compatible training-crate model exports.
//!
//! The canonical Burn model implementation lives in `hydra_model::model`; the
//! training forward adapters live in `hydra_train_runtime::model`.

pub use hydra_train_runtime::model::*;
