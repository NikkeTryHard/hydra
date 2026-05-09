//! Hydra Mahjong Game Engine
//!
//! High-performance Riichi Mahjong simulator built on riichienv-core.
//! Provides observation encoding, safety calculations, and batch simulation
//! for training the Hydra AI.

#![deny(clippy::dbg_macro, clippy::manual_assert)]

pub mod action;
pub mod afbs {
    //! Compatibility re-export for AFBS APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::afbs::*;
}
pub mod arena;
pub mod batch_encoder {
    //! Compatibility re-export for batch encoder APIs now owned by `hydra-encoder`.
    pub use hydra_encoder::batch_encoder::*;
}
pub mod bridge;
pub mod ct_smc {
    //! Compatibility re-export for CT-SMC APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::ct_smc::*;
}
pub mod encoder {
    //! Compatibility re-export for encoder APIs now owned by `hydra-encoder`.
    pub use hydra_encoder::encoder::*;
}
pub mod endgame {
    //! Compatibility re-export for endgame APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::endgame::*;
}
pub mod game_loop;
pub mod hand_ev {
    //! Compatibility re-export for Hand-EV APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::hand_ev::*;
}
pub mod robust_opponent {
    //! Compatibility re-export for robust opponent APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::robust_opponent::*;
}
pub mod safety {
    //! Compatibility re-export for safety APIs now owned by `hydra-safety`.
    pub use hydra_safety::*;
}
pub mod seeding;
pub mod shanten_batch {
    //! Compatibility re-export for batch shanten APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::shanten_batch::*;
}
pub mod simulator;
pub mod sinkhorn {
    //! Compatibility re-export for Sinkhorn/SIB APIs now owned by `hydra-belief-search`.
    pub use hydra_belief_search::sinkhorn::*;
}
pub mod tile {
    //! Compatibility re-export for tile APIs now owned by `hydra-runtime-types`.
    pub use hydra_runtime_types::tile::*;
}
