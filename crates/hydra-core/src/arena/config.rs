pub struct ArenaConfig {
    pub num_parallel_games: usize,
    pub game_mode: u8,
    pub temperature_range: (f32, f32),
    pub exit_fraction: f32,
    pub max_trajectory_buffer: usize,
}

impl ArenaConfig {
    pub fn summary(&self) -> String {
        format!(
            "arena(games={}, temp={:.1}-{:.1}, buf={})",
            self.num_parallel_games,
            self.temperature_range.0,
            self.temperature_range.1,
            self.max_trajectory_buffer
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_parallel_games == 0 {
            return Err("num_parallel_games > 0");
        }
        if self.max_trajectory_buffer == 0 {
            return Err("max_trajectory_buffer > 0");
        }
        if self.temperature_range.0 <= 0.0 {
            return Err("temperature range start > 0");
        }
        if self.temperature_range.1 < self.temperature_range.0 {
            return Err("temperature range end >= start");
        }
        Ok(())
    }
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            num_parallel_games: 500,
            game_mode: 0,
            temperature_range: (0.5, 1.5),
            exit_fraction: 0.2,
            max_trajectory_buffer: 100_000,
        }
    }
}

pub struct SelfPlayConfig {
    pub arena: ArenaConfig,
    pub gae_gamma: f32,
    pub gae_lambda: f32,
    pub rebase_interval_hours: f32,
}

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.arena.validate()?;
        if self.gae_gamma <= 0.0 || self.gae_gamma >= 1.0 {
            return Err("gae_gamma in (0,1)");
        }
        if self.gae_lambda <= 0.0 || self.gae_lambda >= 1.0 {
            return Err("gae_lambda in (0,1)");
        }
        Ok(())
    }
}

impl SelfPlayConfig {
    pub fn with_games(mut self, n: usize) -> Self {
        self.arena.num_parallel_games = n;
        self
    }

    pub fn summary(&self) -> String {
        format!(
            "selfplay(games={}, gamma={:.3}, rebase={:.0}h)",
            self.arena.num_parallel_games, self.gae_gamma, self.rebase_interval_hours
        )
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            arena: ArenaConfig::default(),
            gae_gamma: 0.995,
            gae_lambda: 0.95,
            rebase_interval_hours: 37.5,
        }
    }
}
