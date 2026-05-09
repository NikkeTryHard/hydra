//! Compatibility re-exports for MJAI replay loading.
//!
//! Replay loading now lives in `hydra-replay-loader`; this module preserves
//! the historical `hydra_train::data::mjai_loader` path.

pub use hydra_replay_loader::mjai_loader::*;
pub use hydra_replay_loader::replay_targets::bool_mask_to_f32;
