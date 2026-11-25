//! Keep the canonical action vocabulary at the pinned Hydra2 revision.
//! The stable action IDs are part of Hydra3's feature and corpus format.

extern crate self as hydra_feed;
#[path = "../../vendor/hydra2/crates/feed/src/census.rs"]
pub mod census;
#[path = "../../vendor/hydra2/crates/bridge/src/rl"]
pub mod rl {
    pub mod actions;
}
