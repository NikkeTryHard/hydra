//! Batch game simulation with rayon parallelism.
//!
//! Wraps riichienv-core's game engine to run N complete games
//! in parallel using a dedicated rayon ThreadPool.

use rayon::prelude::*;
use riichienv_core::action::Action;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use std::collections::HashMap;

/// Configuration for a batch simulation run.
#[derive(Debug, Clone)]
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
fn simulate_single_game(seed: Option<u64>, game_mode: u8) -> GameResult {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(game_mode, true, seed, 0, rule);
    let mut total_actions: u32 = 0;
    let mut rounds: u32 = 1; // Game starts in round 1.

    // Safety limit to prevent infinite loops from engine bugs.
    const MAX_STEPS: u32 = 10_000;

    while !state.is_done && total_actions < MAX_STEPS {
        // When a round ends, step() auto-initializes the next round.
        if state.needs_initialize_next_round {
            let empty: HashMap<u8, Action> = HashMap::new();
            state.step(&empty);
            rounds += 1;
            continue;
        }

        // Get legal actions for the current player.
        let obs = state.get_observation(state.current_player);
        let legal = obs.legal_actions_method();

        if legal.is_empty() {
            break;
        }

        // Pick first legal action (deterministic baseline).
        let mut actions = HashMap::new();
        actions.insert(state.current_player, legal[0].clone());
        state.step(&actions);
        total_actions += 1;
    }

    let scores = [
        state.players[0].score,
        state.players[1].score,
        state.players[2].score,
        state.players[3].score,
    ];

    GameResult {
        scores,
        rounds_played: rounds,
        total_actions,
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
                .map(|i| {
                    let seed = base_seed.map(|s| s.wrapping_add(i as u64));
                    simulate_single_game(seed, game_mode)
                })
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
        .map(|i| {
            let seed = base_seed.map(|s| s.wrapping_add(i as u64));
            simulate_single_game(seed, game_mode)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_game_completes() {
        let result = simulate_single_game(Some(42), 0);
        assert!(result.total_actions > 0, "game should have actions");
        assert!(result.rounds_played > 0, "game should have rounds");
    }

    #[test]
    fn batch_returns_correct_count() {
        let config = BatchConfig {
            num_games: 4,
            base_seed: Some(100),
            game_mode: 0,
            ..Default::default()
        };
        let results = run_batch_simple(&config);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn seeded_games_are_deterministic() {
        let r1 = simulate_single_game(Some(999), 0);
        let r2 = simulate_single_game(Some(999), 0);
        assert_eq!(r1.scores, r2.scores);
        assert_eq!(r1.total_actions, r2.total_actions);
        assert_eq!(r1.rounds_played, r2.rounds_played);
    }

    #[test]
    fn scores_sum_is_plausible() {
        // Standard mahjong: 4 players x 25000 = 100000 total.
        // Riichi sticks on table can cause deviations, but sum
        // should be close to 100000.
        let result = simulate_single_game(Some(123), 0);
        let sum: i32 = result.scores.iter().sum();
        // Allow deviation for riichi deposits still on table.
        assert!(
            (90_000..=110_000).contains(&sum),
            "score sum {} outside plausible range",
            sum
        );
    }

    #[test]
    fn batch_simulator_with_threads() {
        let sim = BatchSimulator::new(Some(2)).unwrap();
        let config = BatchConfig {
            num_games: 4,
            base_seed: Some(500),
            game_mode: 0,
            ..Default::default()
        };
        let results = sim.run_batch(&config);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.total_actions > 0);
        }
    }

}
