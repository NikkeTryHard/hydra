//! Batch game simulation with rayon parallelism.
//!
//! Wraps riichienv-core's game engine to run N complete games
//! in parallel using a dedicated rayon ThreadPool.

mod batch;
mod config;
mod result;
mod runner;

pub use batch::{BatchSimulator, run_batch_simple};
pub use config::BatchConfig;
pub use result::GameResult;

#[cfg(test)]
use runner::simulate_single_game;

#[cfg(test)]
mod tests;
