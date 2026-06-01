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
