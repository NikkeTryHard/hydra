//! Hydra Mahjong Game Engine
//!
//! High-performance Riichi Mahjong simulator built on riichienv-core.
//! Provides observation encoding, safety calculations, and batch simulation
//! for training the Hydra AI.

pub mod action;
pub mod afbs;
pub mod arena;
pub mod batch_encoder;
pub mod bridge;
pub mod ct_smc;
pub mod encoder;
pub mod endgame;
pub mod game_loop;
pub mod hand_ev;
pub mod robust_opponent;
pub mod safety;
pub mod seeding;
pub mod shanten_batch;
pub mod simulator;
pub mod sinkhorn;
pub mod tile;
