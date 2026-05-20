//! Batch game simulation with rayon parallelism.
//!
//! Wraps riichienv-core's game engine to run N complete games
//! in parallel using a dedicated rayon ThreadPool.

use rayon::prelude::*;
/// Configuration for a batch simulation run.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BatchConfig {
    /// Number of games to simulate.
    pub num_games: usize,
    /// Base seed for deterministic simulation. Each game gets seed + game_index.
    pub base_seed: Option<u64>,
    /// Number of threads in the rayon pool. None = use rayon default (num CPUs).
    pub num_threads: Option<usize>,
    /// Game mode: 0 = hanchan (east+south), 1 = east only, 2 = single round.
    pub game_mode: u8,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_games: 100,
            base_seed: None,
            num_threads: None,
            game_mode: 0,
        }
    }
}

/// Result from a single completed game.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct GameResult {
    /// Final scores for each player (4 players).
    pub scores: [i32; 4],
    /// Number of rounds (kyoku) played.
    pub rounds_played: u32,
    /// Total number of actions taken across all rounds.
    pub total_actions: u32,
    /// The seed used for this game.
    pub seed: Option<u64>,
}

/// Simulate a single complete game with first-legal-action selection.
/// Used for benchmarking throughput -- real training uses NN policy.
#[cfg(test)]
fn simulate_single_game(seed: Option<u64>, game_mode: u8) -> GameResult {
    let mut runner = crate::game_loop::GameRunner::new(None, game_mode);
    simulate_single_game_with_runner(&mut runner, seed)
}

fn simulate_single_game_with_runner(
    runner: &mut crate::game_loop::GameRunner,
    seed: Option<u64>,
) -> GameResult {
    let mut selector = crate::game_loop::FirstActionSelector;
    runner.reset_for_new_game(seed);
    let outcome = runner.run_to_completion(&mut selector);
    assert_eq!(
        outcome,
        crate::game_loop::StepOutcome::Complete,
        "batch simulator stopped before game completion: {outcome:?}"
    );
    GameResult {
        scores: runner.scores(),
        rounds_played: runner.rounds_played(),
        total_actions: runner.total_actions(),
        seed,
    }
}

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

#[cfg(test)]
mod tests;
