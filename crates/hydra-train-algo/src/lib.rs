//! Pure Rust/Burn algorithm and loss math shared by Hydra training crates.
//!
//! Python/PyTorch owns the default plain BC losses and optimizer path. This
//! crate remains the Rust/Burn reference/fallback owner for optional ExIt BC
//! helpers, RL/self-play algorithms, distillation, DRDA, GAE, and advanced-head
//! tensor losses.

pub mod ach;
pub mod bc;
pub mod distill;
pub mod drda;
pub mod gae;
pub mod losses;
