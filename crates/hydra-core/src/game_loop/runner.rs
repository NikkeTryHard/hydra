use crate::encoder::ObservationEncoder;
use crate::safety::SafetyInfo;
use crate::seeding::SessionRng;
use riichienv_core::action::Action;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

/// Runs a complete game with proper phase handling and safety tracking.
pub struct GameRunner {
    pub(super) state: GameState,
    pub(super) safety: [SafetyInfo; 4],
    pub(super) total_actions: u32,
    pub(super) rounds_played: u32,
    pub(super) actions: [Option<Action>; 4],
    pub(super) legal_buf: Vec<Action>,
    pub(super) encoder: ObservationEncoder,
}
impl GameRunner {
    /// Create a new game runner.
    pub fn new(seed: Option<u64>, game_mode: u8) -> Self {
        let rule = GameRule::default_tenhou();
        let state = GameState::new(game_mode, true, seed, 0, rule);
        Self {
            state,
            safety: std::array::from_fn(|_| SafetyInfo::new()),
            total_actions: 0,
            rounds_played: 1,
            actions: [None; 4],
            legal_buf: Vec::with_capacity(46),
            encoder: ObservationEncoder::new(),
        }
    }

    /// Create a new game runner using Hydra's deterministic seeding.
    ///
    /// Derives a game seed from the session RNG via SHA-256 KDF,
    /// then passes it to riichienv-core's GameState.
    pub fn new_with_session(session: &mut SessionRng, game_mode: u8) -> Self {
        let game_seed = session.next_game_seed();
        // Convert first 8 bytes of the 32-byte seed to u64 for riichienv
        let seed_u64 = u64::from_le_bytes({
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&game_seed[..8]);
            buf
        });
        let rule = GameRule::default_tenhou();
        let state = GameState::new(game_mode, true, Some(seed_u64), 0, rule);
        Self {
            state,
            safety: std::array::from_fn(|_| SafetyInfo::new()),
            total_actions: 0,
            rounds_played: 1,
            actions: [None; 4],
            legal_buf: Vec::with_capacity(46),
            encoder: ObservationEncoder::new(),
        }
    }

    pub fn reset_for_new_game(&mut self, seed: Option<u64>) {
        self.state.reset_for_new_game(seed);
        for safety in &mut self.safety {
            safety.reset();
        }
        self.total_actions = 0;
        self.rounds_played = 1;
        self.actions = [None; 4];
        self.legal_buf.clear();
        self.encoder = ObservationEncoder::new();
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.state.is_done
    }

    #[inline]
    pub fn total_actions(&self) -> u32 {
        self.total_actions
    }

    #[inline]
    pub fn rounds_played(&self) -> u32 {
        self.rounds_played
    }

    #[inline]
    pub fn scores(&self) -> [i32; 4] {
        std::array::from_fn(|i| self.state.players[i].score)
    }

    /// Get safety info from a specific player's perspective.
    #[inline]
    pub fn safety(&self, player: u8) -> &SafetyInfo {
        &self.safety[player as usize]
    }
}
