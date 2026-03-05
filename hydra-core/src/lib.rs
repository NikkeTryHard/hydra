//! Hydra Mahjong Game Engine
//!
//! High-performance Riichi Mahjong simulator built on riichienv-core.
//! Provides observation encoding, safety calculations, and batch simulation
//! for training the Hydra AI.

pub mod tile;
pub mod action;
pub mod safety;
pub mod encoder;
pub mod simulator;
pub mod seeding;
pub mod bridge;
pub mod game_loop;
pub mod shanten_batch;
pub mod afbs;
pub mod arena;
pub mod batch_encoder;
pub mod ct_smc;
pub mod robust_opponent;
