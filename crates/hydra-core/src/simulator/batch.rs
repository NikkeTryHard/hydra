use rayon::prelude::*;

use super::config::BatchConfig;
use super::result::GameResult;
use super::runner::simulate_single_game_with_runner;

/// Parallel batch simulator using a dedicated rayon ThreadPool.
pub struct BatchSimulator {
    pool: rayon::ThreadPool,
}

impl BatchSimulator {
    /// Create a new batch simulator with the given thread count.
    pub fn new(num_threads: Option<usize>) -> anyhow::Result<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = num_threads {
            builder = builder.num_threads(n);
        }
        let pool = builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build thread pool: {}", e))?;
        Ok(Self { pool })
    }

    /// Run a batch of games in parallel. Returns results for all games.
    pub fn run_batch(&self, config: &BatchConfig) -> Vec<GameResult> {
        let num_games = config.num_games;
        let base_seed = config.base_seed;
        let game_mode = config.game_mode;

        self.pool.install(|| {
            (0..num_games)
                .into_par_iter()
                .map_init(
                    || crate::game_loop::GameRunner::new(None, game_mode),
                    |runner, i| {
                        let seed = base_seed.map(|s| s.wrapping_add(i as u64));
                        simulate_single_game_with_runner(runner, seed)
                    },
                )
                .collect()
        })
    }
}

/// Convenience: run a batch without constructing a BatchSimulator.
/// Uses rayon's global thread pool.
pub fn run_batch_simple(config: &BatchConfig) -> Vec<GameResult> {
    let num_games = config.num_games;
    let base_seed = config.base_seed;
    let game_mode = config.game_mode;

    (0..num_games)
        .into_par_iter()
        .map_init(
            || crate::game_loop::GameRunner::new(None, game_mode),
            |runner, i| {
                let seed = base_seed.map(|s| s.wrapping_add(i as u64));
                simulate_single_game_with_runner(runner, seed)
            },
        )
        .collect()
}
