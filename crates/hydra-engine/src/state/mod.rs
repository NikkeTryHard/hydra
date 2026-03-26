use std::collections::HashMap;

use serde_json::Value;

use crate::action::{Action, ActionType, Phase};
use crate::errors::{RiichiError, RiichiResult};
use crate::observation::Observation;
use crate::observation_ref::ObservationRef;
use crate::parser::tid_to_mjai;
use crate::replay::Action as LogAction;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::types::{Conditions, Meld, MeldType, WinResult, Wind};

pub mod event_handler;
pub mod game_mode;
pub mod legal_actions;
pub mod player;
pub mod wall;
use event_handler::GameStateEventHandler;
use game_mode::GameModeConfig;
use legal_actions::GameStateLegalActions;
use player::PlayerState;
use wall::WallState;

const NP: usize = 4;

/// Insert `tile` into a sorted fixed-size hand array, maintaining sort order.
#[inline]
fn sorted_insert_arr(arr: &mut [u8; 14], len: &mut u8, val: u8) {
    let l = *len as usize;
    debug_assert!(
        l < 14,
        "sorted_insert_arr: hand overflow (len={l}, val={val})"
    );
    let pos = arr[..l].partition_point(|&x| x < val);
    for i in (pos..l).rev() {
        arr[i + 1] = arr[i];
    }
    arr[pos] = val;
    *len += 1;
}

/// Copy a tile slice into a `[u8; 5]` buffer, sorted-insert one extra tile, return (buf, len).
#[inline]
fn copy_and_sorted_insert(src: &[u8], extra: u8) -> ([u8; 5], usize) {
    let mut buf = [0u8; 5];
    let n = src.len().min(4);
    buf[..n].copy_from_slice(&src[..n]);
    let pos = buf[..n].partition_point(|&x| x < extra);
    for i in (pos..n).rev() {
        buf[i + 1] = buf[i];
    }
    buf[pos] = extra;
    (buf, n + 1)
}

/// Full game state for a 4-player Riichi Mahjong game.
#[derive(Debug, Clone)]
pub struct GameState {
    /// Wall state containing tiles, dora indicators, and draw cursors.
    pub wall: WallState,
    /// Per-player state for all four players.
    pub players: [PlayerState; 4],

    /// Index of the player whose turn it is (0-3).
    pub current_player: u8,
    /// Number of full turns elapsed in the current round.
    pub turn_count: u32,
    /// Whether the game has ended.
    pub is_done: bool,
    /// Whether the current player needs to draw a tile.
    pub needs_tsumo: bool,
    /// Whether the next step should initialize a new round.
    pub needs_initialize_next_round: bool,
    /// Whether the oya won the previous round (for renchan).
    pub pending_oya_won: bool,
    /// Whether the previous round ended in a draw.
    pub pending_is_draw: bool,

    /// Number of riichi deposit sticks on the table.
    pub riichi_sticks: u32,
    /// Current game phase (WaitAct or WaitResponse).
    pub phase: Phase,
    /// Player indices that must act in the current phase.
    pub active_players: [u8; 4],
    /// Number of active players in the current phase.
    pub active_player_count: u8,
    /// Last discarded tile as (player_id, tile), if any.
    pub last_discard: Option<(u8, u8)>,
    /// Pending claim actions per player for the current discard.
    pub current_claims: [[Action; 54]; NP],
    /// Number of pending claims per player.
    pub current_claim_counts: [u8; NP],
    /// Pending kan action awaiting chankan resolution.
    pub pending_kan: Option<(u8, Action)>,

    /// Dealer seat index (0-3).
    pub oya: u8,
    /// Repeat counter for the current round.
    pub honba: u8,
    /// Kyoku index (same as oya for display).
    pub kyoku_idx: u8,
    /// Prevailing wind (0=East, 1=South, 2=West, 3=North).
    pub round_wind: u8,

    /// Whether the current draw is a rinshan (replacement after kan).
    pub is_rinshan_flag: bool,
    /// Whether it is still the first go-around of discards.
    pub is_first_turn: bool,
    /// Player whose riichi stick payment is pending, if any.
    pub riichi_pending_acceptance: Option<u8>,
    /// Tile most recently drawn by the current player, if any.
    pub drawn_tile: Option<u8>,

    /// Win results for the current round, one per player.
    pub win_results: [Option<WinResult>; NP],
    /// Win results from the previous round.
    pub last_win_results: [Option<WinResult>; NP],

    /// Global MJAI event log as JSON strings.
    pub mjai_log: Vec<String>,
    /// Typed MJAI events for zero-cost structured logging.
    pub mjai_events: Vec<crate::mjai_event::MjaiEvent>,
    /// Number of events each player has consumed from their log.
    pub player_event_counts: [usize; NP],
    /// Per-player MJAI event logs with masked private info.
    pub mjai_log_per_player: [Vec<String>; NP],

    /// Game mode configuration derived from `game_mode` and `rule`.
    pub mode: GameModeConfig,
    /// Numeric game mode identifier (e.g. 0=one-round, 1=tonpuu, 2=hanchan).
    pub game_mode: u8,
    /// Whether to skip MJAI logging for throughput.
    pub skip_mjai_logging: bool,
    /// Optional deterministic seed for reproducible games.
    pub seed: Option<u64>,
    /// Rule configuration (Tenhou, MjSoul, or custom).
    pub rule: GameRule,
    /// Last error message from an illegal action, if any.
    pub last_error: Option<String>,
    /// Whether the last action was a kan (for kan-related logic).
    pub is_after_kan: bool,

    /// Tile discarded when declaring riichi, per player.
    pub riichi_sutehais: [Option<u8>; NP],
    /// Last hand discard (not tsumogiri) per player.
    pub last_tedashis: [Option<u8>; NP],
}

impl GameState {
    /// Return the number of players (always 4).
    pub fn np(&self) -> usize {
        NP
    }

    /// Returns the currently active players as a slice.
    #[inline]
    pub fn active_player_slice(&self) -> &[u8] {
        &self.active_players[..self.active_player_count as usize]
    }

    /// Clears the active players list.
    #[inline]
    fn clear_active_players(&mut self) {
        self.active_player_count = 0;
    }

    /// Returns the claims slice for a given player.
    #[inline]
    fn claims_slice(&self, pid: usize) -> &[Action] {
        &self.current_claims[pid][..self.current_claim_counts[pid] as usize]
    }

    /// Pushes a claim action for a player.
    #[inline]
    fn push_claim(&mut self, pid: usize, action: Action) {
        let idx = self.current_claim_counts[pid] as usize;
        self.current_claims[pid][idx] = action;
        self.current_claim_counts[pid] += 1;
    }

    /// Clears all current claims.
    #[inline]
    fn clear_claims(&mut self) {
        self.current_claim_counts = [0; NP];
    }

    /// Sets active players to a single player.
    #[inline]
    fn set_single_active_player(&mut self, pid: u8) {
        self.active_players[0] = pid;
        self.active_player_count = 1;
    }

    /// Sets active players from a slice.
    #[inline]
    fn set_active_players_from_slice(&mut self, pids: &[u8]) {
        self.active_players[..pids.len()].copy_from_slice(pids);
        self.active_player_count = pids.len() as u8;
    }

    /// Create a new game state with the given configuration and deal the first hand.
    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        let mode = GameModeConfig::from_game_mode(game_mode, rule);
        let players = [(); 4].map(|_| PlayerState::new(mode.starting_score()));

        let wall = WallState::new(seed);

        let mut state = Self {
            wall,
            players,
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
            needs_initialize_next_round: false,
            pending_oya_won: false,
            pending_is_draw: false,
            riichi_sticks: 0,
            phase: Phase::WaitAct,
            active_players: [0; 4],
            active_player_count: 0,
            last_discard: None,
            current_claims: [[Action::default(); 54]; NP],
            current_claim_counts: [0; NP],
            pending_kan: None,
            oya: 0,
            honba: 0,
            kyoku_idx: 0,
            round_wind,
            is_rinshan_flag: false,
            is_first_turn: true,
            riichi_pending_acceptance: None,
            drawn_tile: None,
            win_results: Default::default(),
            last_win_results: Default::default(),
            mjai_log: if skip_mjai_logging {
                Vec::new()
            } else {
                Vec::with_capacity(300)
            },
            mjai_events: if skip_mjai_logging {
                Vec::new()
            } else {
                Vec::with_capacity(300)
            },
            player_event_counts: [0; NP],
            mjai_log_per_player: Default::default(),
            mode,
            game_mode,
            skip_mjai_logging,
            seed,
            rule,
            last_error: None,
            is_after_kan: false,
            riichi_sutehais: [None; NP],
            last_tedashis: [None; NP],
        };

        if !state.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            state._push_mjai_event(Value::Object(ev));
        }
        mjai_event!(state, crate::mjai_event::MjaiEvent::StartGame);

        // Initial setup
        state._initialize_round(0, round_wind, 0, 0, None, None);
        state
    }

    /// Reset MJAI logs and event counters without changing game state.
    pub fn reset(&mut self) {
        self.mjai_log = Vec::new();
        self.mjai_log_per_player = Default::default();
        self.player_event_counts = [0; NP];
        self.mjai_events = Vec::with_capacity(300);

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
        mjai_event!(self, crate::mjai_event::MjaiEvent::StartGame);
    }

    /// Resets the game state for a new game, reusing configuration.
    ///
    /// Uses `*self = Self::new(...)` for correctness -- can't miss a field.
    /// The allocation cost (~2.5us) is negligible compared to a full game.
    pub fn reset_for_new_game(&mut self, new_seed: Option<u64>) {
        let rule = self.rule;
        let game_mode = self.game_mode;
        let skip_logging = self.skip_mjai_logging;
        *self = Self::new(game_mode, skip_logging, new_seed, 0, rule);
    }

    /// Build a player-facing observation with legal actions and event diff.
    pub fn get_observation(&mut self, player_id: u8) -> Observation {
        let pid = player_id as usize;

        let masked_hands: [Vec<u8>; 4] = std::array::from_fn(|i| {
            if i == pid {
                self.players[i].hand_slice().to_vec()
            } else {
                Vec::new()
            }
        });

        let legal_actions = if self.is_done {
            Vec::new()
        } else if (self.phase == Phase::WaitAct && self.current_player == player_id)
            || (self.phase == Phase::WaitResponse
                && self.active_player_slice().contains(&player_id))
        {
            self._get_legal_actions_internal(player_id)
        } else {
            Vec::new()
        };

        let old_count = self.player_event_counts[pid];
        let full_log_len = self.mjai_log_per_player[pid].len();
        let new_events = if old_count < full_log_len {
            self.mjai_log_per_player[pid][old_count..].to_vec()
        } else {
            Vec::new()
        };
        self.player_event_counts[pid] = full_log_len;

        let calc = crate::hand_evaluator::HandEvaluator::new(
            self.players[pid].hand_slice(),
            self.players[pid].melds_slice(),
        );
        let waits = calc.get_waits_u8();
        let is_tenpai = !waits.is_empty();

        let melds: [Vec<Meld>; 4] = std::array::from_fn(|i| self.players[i].melds_slice().to_vec());
        let discards: [Vec<u8>; 4] =
            std::array::from_fn(|i| self.players[i].discards_slice().to_vec());
        let scores: [i32; 4] = std::array::from_fn(|i| self.players[i].score);
        let riichi_declared: [bool; 4] = std::array::from_fn(|i| self.players[i].riichi_declared);

        Observation::new(
            player_id,
            masked_hands,
            melds,
            discards,
            self.wall.dora_indicator_slice().to_vec(),
            scores,
            riichi_declared,
            legal_actions,
            new_events,
            self.honba,
            self.riichi_sticks,
            self.round_wind,
            self.oya,
            self.kyoku_idx,
            waits,
            is_tenpai,
            self.riichi_sutehais,
            self.last_tedashis,
            self.last_discard.map(|(tile, _pid)| tile as u32),
        )
    }

    /// Zero-copy observation view. Borrows from self, zero heap allocations.
    pub fn observe(&self, player_id: u8) -> ObservationRef<'_> {
        let pid = player_id as usize;
        ObservationRef {
            player_id,
            observer_hand: self.players[pid].hand_slice(),
            melds: std::array::from_fn(|i| self.players[i].melds_slice()),
            discards: std::array::from_fn(|i| self.players[i].discards_slice()),
            tsumogiri_flags: std::array::from_fn(|i| {
                &self.players[i].discard_from_hand[..self.players[i].discard_len as usize]
            }),
            dora_indicators: self.wall.dora_indicator_slice(),
            scores: std::array::from_fn(|i| self.players[i].score),
            riichi_declared: std::array::from_fn(|i| self.players[i].riichi_declared),
            honba: self.honba,
            riichi_sticks: self.riichi_sticks,
            round_wind: self.round_wind,
            oya: self.oya,
            kyoku_index: self.kyoku_idx,
            current_player: self.current_player,
            drawn_tile: self.drawn_tile,
            is_done: self.is_done,
        }
    }

    /// Get legal actions without constructing a full Observation.
    #[inline]
    pub fn get_legal_actions(&self, player_id: u8) -> Vec<Action> {
        self._get_legal_actions_internal(player_id)
    }

    /// Get legal actions without allocating a new Vec.
    /// Clears `buf` and pushes legal actions into it.
    #[inline]
    pub fn get_legal_actions_into(&self, player_id: u8, buf: &mut Vec<Action>) {
        buf.clear();
        self._get_legal_actions_into(player_id, buf);
    }

    #[inline]
    fn replay_action_matches_legal(legal: &Action, replay: &Action) -> bool {
        if legal.action_type != replay.action_type {
            return false;
        }

        let tiles_match = match (legal.tile, replay.tile) {
            (Some(legal_tile), Some(replay_tile)) => {
                Self::replay_tile_matches_mjai_semantics(legal_tile, replay_tile)
            }
            (None, None) => true,
            _ => false,
        };
        let consumes_match = legal.consume_slice() == replay.consume_slice();

        if tiles_match {
            if consumes_match {
                return true;
            }

            if replay.consume_count == 0 && legal.action_type == ActionType::Kakan {
                return true;
            }

            if replay.consume_count == 0
                && matches!(
                    legal.action_type,
                    ActionType::Discard
                        | ActionType::Riichi
                        | ActionType::Tsumo
                        | ActionType::Ron
                        | ActionType::Pass
                )
            {
                return true;
            }
        }

        if consumes_match && matches!(legal.action_type, ActionType::Ankan | ActionType::Kakan) {
            return true;
        }

        if matches!(legal.action_type, ActionType::Ankan | ActionType::Kakan) {
            if let (Some(legal_tile), Some(replay_tile)) = (legal.tile, replay.tile) {
                return legal_tile / 4 == replay_tile / 4;
            }
        }

        if replay.tile.is_none() {
            return matches!(
                legal.action_type,
                ActionType::Tsumo
                    | ActionType::Ron
                    | ActionType::Riichi
                    | ActionType::KyushuKyuhai
                    | ActionType::Kita
            );
        }

        false
    }

    #[inline]
    fn replay_tile_matches_mjai_semantics(legal_tile: u8, replay_tile: u8) -> bool {
        if legal_tile == replay_tile {
            return true;
        }

        let legal_type = legal_tile / 4;
        let replay_type = replay_tile / 4;
        if legal_type != replay_type {
            return false;
        }

        if matches!(legal_type, 4 | 13 | 22) {
            let red_copy = legal_type * 4;
            let legal_is_red = legal_tile == red_copy;
            let replay_is_red = replay_tile == red_copy;
            return legal_is_red == replay_is_red;
        }

        true
    }

    /// Build an observation for replay validation, temporarily adjusting phase if needed.
    pub fn get_observation_for_replay(
        &mut self,
        pid: u8,
        env_action: &Action,
        log_action_str: &str,
    ) -> RiichiResult<Observation> {
        let original_phase = self.phase;
        let original_active_players = self.active_players;
        let original_active_player_count = self.active_player_count;
        let original_claims = self.current_claims;
        let original_claim_counts = self.current_claim_counts;
        let original_riichi = self.players[pid as usize].riichi_declared;
        let original_current_player = self.current_player;

        match env_action.action_type {
            ActionType::Ron | ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                self.phase = Phase::WaitResponse;
                self.set_single_active_player(pid);
                self.push_claim(pid as usize, *env_action);
            }
            _ => {}
        }

        let mut obs = self.get_observation(pid);

        let mut exists = obs
            ._legal_actions
            .iter()
            .any(|a| Self::replay_action_matches_legal(a, env_action));

        if !exists
            && env_action.action_type == ActionType::Discard
            && self.players[pid as usize].riichi_declared
        {
            self.players[pid as usize].riichi_declared = false;
            let new_obs = self.get_observation(pid);
            let is_legal_retry = new_obs._legal_actions.iter().any(|a| {
                a.action_type == ActionType::Discard
                    && match (a.tile, env_action.tile) {
                        (Some(legal_tile), Some(replay_tile)) => {
                            Self::replay_tile_matches_mjai_semantics(legal_tile, replay_tile)
                        }
                        _ => false,
                    }
            });

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            } else {
                self.players[pid as usize].riichi_declared = original_riichi;
            }
        }

        if !exists && env_action.action_type == ActionType::Kakan && self.drawn_tile.is_some() {
            self.set_single_active_player(pid);
            let new_obs = self.get_observation(pid);
            let is_legal_retry = new_obs
                ._legal_actions
                .iter()
                .any(|a| Self::replay_action_matches_legal(a, env_action));

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            }
        }

        self.phase = original_phase;
        self.current_player = original_current_player;
        self.active_players = original_active_players;
        self.active_player_count = original_active_player_count;
        self.current_claims = original_claims;
        self.current_claim_counts = original_claim_counts;

        if !exists {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "Replay desync:\n  Env action: {:?}\n  Log action: {}\n  Self state:\n    phase: {:?}\n    drawn: {:?}",
                    env_action,
                    log_action_str,
                    self.phase,
                    self.drawn_tile
                ),
            });
        }

        Ok(obs)
    }

    pub fn replay_observation_contains_action(obs: &Observation, env_action: &Action) -> bool {
        obs._legal_actions
            .iter()
            .any(|action| Self::replay_action_matches_legal(action, env_action))
    }

    pub fn resolve_replay_all_passes(&mut self) {
        self.clear_claims();
        self.clear_active_players();
        self._accept_riichi();
        self.phase = Phase::WaitAct;
    }

    /// Advance the game by one step, validating all player actions.
    pub fn step(&mut self, actions: &HashMap<u8, Action>) {
        if self.is_done {
            return;
        }

        if self.needs_initialize_next_round {
            self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
            return;
        }
        // Validation
        let np = NP;
        for pid in 0..np {
            if let Some(act) = actions.get(&(pid as u8)) {
                let legals = self._get_legal_actions_internal(pid as u8);
                let is_valid = legals.iter().any(|l| {
                    if l.action_type != act.action_type {
                        return false;
                    }

                    let tiles_match = l.tile == act.tile;
                    let consumes_match = l.consume_slice() == act.consume_slice();

                    if tiles_match {
                        if consumes_match {
                            return true;
                        }
                        // Allow empty consume for Kakan
                        if act.consume_count == 0 && l.action_type == ActionType::Kakan {
                            return true;
                        }
                        // Allow empty consume for Discard, Riichi, Tsumo, Ron, Pass
                        if act.consume_count == 0
                            && matches!(
                                l.action_type,
                                ActionType::Discard
                                    | ActionType::Riichi
                                    | ActionType::Tsumo
                                    | ActionType::Ron
                                    | ActionType::Pass
                            )
                        {
                            return true;
                        }
                    }

                    if consumes_match
                        && matches!(l.action_type, ActionType::Ankan | ActionType::Kakan)
                    {
                        return true;
                    }

                    // Allow tile-less replay actions for context-implied actions.
                    if act.tile.is_none() {
                        return matches!(
                            l.action_type,
                            ActionType::Tsumo
                                | ActionType::Ron
                                | ActionType::Riichi
                                | ActionType::KyushuKyuhai
                                | ActionType::Kita
                        );
                    }
                    false
                });

                if !is_valid {
                    let reason = format!("Error: Illegal Action by Player {}", pid);
                    self.last_error = Some(reason.clone());
                    self._trigger_ryukyoku(&reason);
                    return;
                }
            }
        }
        // Convert HashMap to array and delegate to the single implementation
        let mut action_arr: [Option<Action>; 4] = [None; 4];
        for (&pid, &act) in actions {
            action_arr[pid as usize] = Some(act);
        }
        self._execute_step_array(&action_arr);
    }

    /// Execute game logic for one step, without validating actions.
    ///
    /// This is the shared implementation called by both `step()` (after validation)
    /// and `step_unchecked()` (without validation).
    #[inline]
    fn _execute_step_array(&mut self, actions: &[Option<Action>; 4]) {
        // --- Phase: WaitAct (Discards, Riichi, Tsumo, Kan) ---
        if self.phase == Phase::WaitAct {
            let pid = self.current_player;
            if let Some(act) = actions[pid as usize] {
                match act.action_type {
                    ActionType::Discard => {
                        self._handle_discard(pid, act);
                    }
                    ActionType::KyushuKyuhai => {
                        self._trigger_ryukyoku("kyushu_kyuhai");
                    }
                    ActionType::Riichi => {
                        self._handle_riichi(pid, act);
                    }
                    ActionType::Ankan => {
                        self._handle_ankan(pid, act);
                    }
                    ActionType::Kakan => {
                        self._handle_kakan(pid, act);
                    }
                    ActionType::Tsumo => {
                        self._handle_tsumo(pid);
                    }
                    ActionType::Kita => {
                        // Kita is only valid in 3P; handled by GameState3P
                    }
                    _ => {}
                }
            }
        } else if self.phase == Phase::WaitResponse {
            self._handle_wait_response(actions);
        }
    }

    /// Handles a discard action during WaitAct phase.
    fn _handle_discard(&mut self, pid: u8, act: Action) {
        if let Some(tile) = act.tile {
            let mut tsumogiri = false;
            let mut valid = false;
            if let Some(dt) = self.drawn_tile {
                if dt == tile {
                    tsumogiri = true;
                    valid = true;
                }
            }
            if let Some(idx) = self.players[pid as usize]
                .hand
                .iter()
                .position(|&t| t == tile)
            {
                self.players[pid as usize].remove_hand(idx);
                valid = true;
                if let Some(dt) = self.drawn_tile {
                    if dt == tile {
                        tsumogiri = true;
                    }
                }
            }
            if valid {
                self._resolve_discard(pid, tile, tsumogiri);
            }
        }
    }

    /// Handles a riichi declaration during WaitAct phase.
    fn _handle_riichi(&mut self, pid: u8, act: Action) {
        // Declare Riichi
        if self.players[pid as usize].score >= 1000
            && self.wall.remaining() >= 18
            && !self.players[pid as usize].riichi_declared
        {
            self.players[pid as usize].riichi_stage = true;
            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("reach".to_string()));
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                self._push_mjai_event(Value::Object(ev));
            }
            if let Some(t) = act.tile {
                let mut tsumogiri = false;
                if let Some(dt) = self.drawn_tile {
                    if dt == t {
                        tsumogiri = true;
                    }
                }
                // Record riichi sutehai (riichi discard tile)
                self.riichi_sutehais[pid as usize] = Some(t);
                // Record last tedashi if not tsumogiri
                if !tsumogiri {
                    self.last_tedashis[pid as usize] = Some(t);
                }
                {
                    let p = &mut self.players[pid as usize];
                    let pos = p.hand_slice().partition_point(|&x| x < t);
                    if pos < p.hand_len as usize && p.hand[pos] == t {
                        p.remove_hand(pos);
                    }
                }
                self._resolve_discard(pid, t, tsumogiri);
            }
        }
    }

    /// Handles an ankan (concealed kan) action during WaitAct phase.
    fn _handle_ankan(&mut self, pid: u8, act: Action) {
        let np = NP;
        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let mut chankan_count: usize = 0;
        let mut chankan_ronners = [0u8; 3];
        if self.rule.allows_ron_on_ankan_for_kokushi_musou {
            for i in 0..np as u8 {
                if i == pid {
                    continue;
                }

                // Check Kokushi Only
                let hand = self.players[i as usize].hand_slice();
                let melds = self.players[i as usize].melds_slice();

                // Furiten check
                let tile_class = tile / 4;
                let in_discards = self.players[i as usize]
                    .discards_slice()
                    .iter()
                    .any(|&d| d / 4 == tile_class);
                if in_discards {
                    continue;
                }

                let p_wind = (i + np as u8 - self.oya) % np as u8;
                let cond = Conditions {
                    tsumo: false,
                    riichi: self.players[i as usize].riichi_declared,
                    chankan: true,
                    player_wind: Wind::from(p_wind),
                    round_wind: Wind::from(self.round_wind),
                    ..Default::default()
                };
                let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);
                let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));

                // 42=Kokushi, 49=Kokushi13
                if res.is_win && (res.yaku_slice().contains(&42) || res.yaku_slice().contains(&49))
                {
                    chankan_ronners[chankan_count] = i;
                    chankan_count += 1;
                    self.push_claim(
                        i as usize,
                        Action::new(ActionType::Ron, Some(tile), &[], Some(i)),
                    );
                }
            }
        }

        if chankan_count > 0 {
            self.pending_kan = Some((pid, act));
            self.phase = Phase::WaitResponse;
            self.set_active_players_from_slice(&chankan_ronners[..chankan_count]);
            self.last_discard = Some((pid, tile));
        } else {
            self._resolve_kan(pid, act);
        }
    }

    /// Handles a kakan (added kan / upgrade pon to kan) action during WaitAct phase.
    fn _handle_kakan(&mut self, pid: u8, act: Action) {
        let np = NP;
        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let p_idx = pid as usize;

        // Update state BEFORE logging/waiting to keep observations in sync
        {
            let p = &mut self.players[p_idx];
            let pos = p.hand_slice().partition_point(|&x| x < tile);
            if pos < p.hand_len as usize && p.hand[pos] == tile {
                p.remove_hand(pos);
            }
        }
        for m in self.players[p_idx].melds_slice_mut().iter_mut() {
            if m.meld_type == crate::types::MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                m.meld_type = crate::types::MeldType::Kakan;
                m.push_tile(tile);
                m.tiles_slice_mut().sort();
                break;
            }
        }

        // Log Kakan immediately (before Chankan check)
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("kakan".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            let cons: Vec<String> = act
                .consume_slice()
                .iter()
                .map(|&t| tid_to_mjai(t))
                .collect();
            // SAFETY: serialization of Vec<String> never fails
            ev.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
            self._push_mjai_event(Value::Object(ev));
        }

        // Reveal any pending kan doras from previous kans
        while self.wall.pending_kan_dora_count > 0 {
            self.wall.pending_kan_dora_count -= 1;
            self._reveal_kan_dora();
        }

        // Kakan Logic
        // Check Chankan
        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let mut chankan_ronners = [0u8; 3];
        let mut chankan_count: usize = 0;
        for i in 0..np as u8 {
            if i == pid {
                continue;
            }
            // Check WinResult
            let hand = self.players[i as usize].hand_slice();
            let melds = self.players[i as usize].melds_slice();
            let p_wind = (i + np as u8 - self.oya) % np as u8;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i as usize].riichi_declared,
                double_riichi: self.players[i as usize].double_riichi_declared,
                ippatsu: self.players[i as usize].ippatsu_cycle,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: true,
                haitei: false,
                houtei: false,
                rinshan: false,
                tsumo_first_turn: false,
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                ..Default::default()
            };
            let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);

            // Check Furiten
            let mut is_furiten = false;
            let waits = calc.get_waits_u8();
            for &w in &waits {
                if self.players[i as usize]
                    .discards_slice()
                    .iter()
                    .any(|&d| d / 4 == w)
                {
                    is_furiten = true;
                    break;
                }
            }
            if self.players[i as usize].missed_agari_riichi
                || self.players[i as usize].missed_agari_doujun
            {
                is_furiten = true;
            }

            // If valid:
            let res = if !is_furiten {
                calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond))
            } else {
                crate::types::WinResult::new(
                    false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
                )
            };

            if res.is_win && (res.yakuman || res.han >= 1) {
                // Add Ron action offer
                chankan_ronners[chankan_count] = i;
                chankan_count += 1;
                self.push_claim(
                    i as usize,
                    Action::new(ActionType::Ron, Some(tile), &[], Some(i)),
                );
            }
        }

        if chankan_count > 0 {
            self.pending_kan = Some((pid, act));
            self.phase = Phase::WaitResponse;
            self.set_active_players_from_slice(&chankan_ronners[..chankan_count]);
            self.last_discard = Some((pid, tile)); // Treat Kakan tile as discard for Ron targeting
        } else {
            self._resolve_kan(pid, act);
        }
    }

    /// Handles a tsumo (self-draw win) action during WaitAct phase.
    fn _handle_tsumo(&mut self, pid: u8) {
        let np = NP;
        let hand = self.players[pid as usize].hand_slice();
        let melds = self.players[pid as usize].melds_slice();
        let p_wind = (pid + np as u8 - self.oya) % np as u8;
        let cond = Conditions {
            tsumo: true,
            riichi: self.players[pid as usize].riichi_declared,
            double_riichi: self.players[pid as usize].double_riichi_declared,
            ippatsu: self.players[pid as usize].ippatsu_cycle,
            haitei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
            rinshan: self.is_rinshan_flag,
            tsumo_first_turn: self.is_first_turn && self.players.iter().all(|p| p.meld_count == 0),
            player_wind: Wind::from(p_wind),
            round_wind: Wind::from(self.round_wind),
            riichi_sticks: self.riichi_sticks,
            honba: self.honba as u32,
            ..Default::default()
        };
        let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);
        let win_tile = self.drawn_tile.unwrap_or(0);
        let ura_indicators = if self.players[pid as usize].riichi_declared {
            self._get_ura_indicators()
        } else {
            vec![]
        };
        let mut res = calc.calc(
            win_tile,
            self.wall.dora_indicator_slice(),
            &ura_indicators,
            Some(cond.clone()),
        );

        // Cap double yakuman patterns when not enabled per rule flags
        if res.yakuman && res.han > 13 {
            let mut cap = 0u32;
            for &y in res.yaku_slice() {
                match y {
                    47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                    48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                    49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                    50 if !self.rule.is_daisuushii_double => cap += 13,
                    _ => {}
                }
            }
            if cap > 0 {
                res.han = res.han.saturating_sub(cap).max(13);
                let capped = crate::score::calculate_score(
                    res.han as u8,
                    0,
                    pid == self.oya,
                    cond.tsumo,
                    cond.honba,
                    np as u8,
                );
                res.ron_agari = capped.pay_ron;
                res.tsumo_agari_oya = capped.pay_tsumo_oya;
                res.tsumo_agari_ko = capped.pay_tsumo_ko;
            }
        }

        if res.is_win {
            let mut deltas = [0i32; NP];
            let mut total_win = 0;

            // Check Pao
            let mut pao_payer = None;
            let mut pao_yakuman_val = 0;
            let mut total_yakuman_val = 0;

            if res.yakuman {
                for &yid in res.yaku_slice() {
                    let val = match yid {
                        47 if self.rule.is_junsei_chuurenpoutou_double => 2,
                        48 if self.rule.is_suuankou_tanki_double => 2,
                        49 if self.rule.is_kokushi_musou_13machi_double => 2,
                        50 if self.rule.is_daisuushii_double => 2,
                        _ => 1,
                    };
                    total_yakuman_val += val;
                    if let Some(liable) = self.players[pid as usize].pao_get(yid as u8) {
                        pao_yakuman_val += val;
                        pao_payer = Some(liable);
                    }
                }
            }

            if pao_yakuman_val > 0 {
                let unit = if pid == self.oya { 48000 } else { 32000 };
                let honba_total = self.honba as i32 * (np as i32 - 1) * 100;

                if let Some(pp) = pao_payer {
                    if self.rule.yakuman_pao_is_liability_only {
                        // Majsoul: PAO pays PAO portion only, non-PAO split normally
                        let pao_amt = pao_yakuman_val * unit + honba_total;
                        let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;

                        deltas[pp as usize] -= pao_amt;
                        total_win += pao_amt;

                        if non_pao_yakuman_val > 0 {
                            if pid == self.oya {
                                let share = non_pao_yakuman_val * 16000;
                                for i in 0..np as u8 {
                                    if i != pid {
                                        deltas[i as usize] -= share;
                                        total_win += share;
                                    }
                                }
                            } else {
                                let oya_pay = non_pao_yakuman_val * 16000;
                                let ko_pay = non_pao_yakuman_val * 8000;
                                for i in 0..np as u8 {
                                    if i != pid {
                                        if i == self.oya {
                                            deltas[i as usize] -= oya_pay;
                                            total_win += oya_pay;
                                        } else {
                                            deltas[i as usize] -= ko_pay;
                                            total_win += ko_pay;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Tenhou: PAO pays ALL yakuman (full amount)
                        let full_amt = total_yakuman_val * unit + honba_total;
                        deltas[pp as usize] -= full_amt;
                        total_win += full_amt;
                    }
                }
            } else {
                // Standard Scoring
                if pid == self.oya {
                    for i in 0..np as u8 {
                        if i != pid {
                            deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                            total_win += res.tsumo_agari_ko as i32;
                        }
                    }
                } else {
                    for i in 0..np as u8 {
                        if i != pid {
                            if i == self.oya {
                                deltas[i as usize] = -(res.tsumo_agari_oya as i32);
                                total_win += res.tsumo_agari_oya as i32;
                            } else {
                                deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                                total_win += res.tsumo_agari_ko as i32;
                            }
                        }
                    }
                }
            }

            total_win += (self.riichi_sticks * 1000) as i32;
            self.riichi_sticks = 0;

            deltas[pid as usize] += total_win;

            self.players[pid as usize].score_delta = deltas[pid as usize]; // Actually we need to set for all
            for (i, p) in self.players.iter_mut().enumerate() {
                p.score += deltas[i];
                p.score_delta = deltas[i];
            }

            let mut val = res;
            for i in 0..self.players[pid as usize].pao_count as usize {
                let (yid, liable) = self.players[pid as usize].pao[i];
                if val.yaku_slice().contains(&(yid as u32)) {
                    val.pao_payer = Some(liable);
                    break;
                }
            }
            self.win_results[pid as usize] = Some(val);

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("hora".to_string()));
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                ev.insert("target".to_string(), Value::Number(pid.into()));
                ev.insert(
                    "deltas".to_string(),
                    // SAFETY: serialization of Vec<i32> never fails
                    serde_json::to_value(deltas).unwrap(),
                );
                ev.insert("tsumo".to_string(), Value::Bool(true));

                let mut ura_markers = Vec::new();
                if self.players[pid as usize].riichi_declared {
                    ura_markers = self._get_ura_markers();
                }
                ev.insert(
                    "ura_markers".to_string(),
                    // SAFETY: serialization of Vec<String> never fails
                    serde_json::to_value(&ura_markers).unwrap(),
                );

                self._push_mjai_event(Value::Object(ev));
            }

            self._initialize_next_round(pid == self.oya, false);
        } else {
            self.current_player = (self.current_player + 1) % np as u8;
            self._deal_next();
        }
    }

    /// Handles the WaitResponse phase (Ron, Pon, Chi, Daiminkan claims).
    fn _handle_wait_response(&mut self, actions: &[Option<Action>; 4]) {
        let np = NP;
        // Check Missed WinResult for all who could Ron but didn't
        for pid in 0..NP {
            if self.current_claim_counts[pid] == 0 {
                continue;
            }
            let legals = self.claims_slice(pid);
            let pid = pid as u8;
            if legals.iter().any(|a| a.action_type == ActionType::Ron) {
                let mut roned = false;
                if let Some(act) = actions[pid as usize] {
                    if act.action_type == ActionType::Ron {
                        roned = true;
                    }
                }
                if !roned {
                    self.players[pid as usize].missed_agari_doujun = true;
                    if self.players[pid as usize].riichi_declared {
                        self.players[pid as usize].missed_agari_riichi = true;
                    }
                }
            }
        }

        let mut ron_claims = [0u8; 3];
        let mut ron_count: usize = 0;
        let mut call_claim: Option<(u8, Action)> = None;

        for &pid in self.active_player_slice() {
            if let Some(act) = actions[pid as usize] {
                if act.action_type == ActionType::Ron {
                    ron_claims[ron_count] = pid;
                    ron_count += 1;
                } else if act.action_type == ActionType::Pon
                    || act.action_type == ActionType::Daiminkan
                    || act.action_type == ActionType::Chi
                {
                    if let Some((_old_pid, old_act)) = &call_claim {
                        let old_is_pon = old_act.action_type == ActionType::Pon
                            || old_act.action_type == ActionType::Daiminkan;
                        let new_is_pon = act.action_type == ActionType::Pon
                            || act.action_type == ActionType::Daiminkan;
                        if !old_is_pon && new_is_pon {
                            call_claim = Some((pid, act));
                        }
                    } else {
                        call_claim = Some((pid, act));
                    }
                }
            }
        }

        if ron_count > 0 {
            // Sanchaho: all non-discarders ron -> abortive draw
            if ron_count >= NP - 1 && self.rule.sanchaho_is_draw {
                self._trigger_ryukyoku("sanchaho");
                return;
            }

            let (target_pid, win_tile) = self.last_discard.unwrap_or((self.current_player, 0));

            ron_claims[..ron_count].sort_by_key(|&pid| (pid + np as u8 - target_pid) % np as u8);

            let winners = &ron_claims[..ron_count];

            let mut total_deltas = [0i32; NP];
            let mut oya_won = false;
            let mut deposit_taken = false;
            let mut honba_taken = false;

            for &w_pid in winners {
                let hand = self.players[w_pid as usize].hand_slice();
                let melds = self.players[w_pid as usize].melds_slice();
                let p_wind = (w_pid + np as u8 - self.oya) % np as u8;
                let is_chankan = self.pending_kan.is_some();

                // Only the first winner (closest to discarder) gets honba
                let ron_honba = if !honba_taken {
                    honba_taken = true;
                    self.honba as u32
                } else {
                    0
                };

                let cond = Conditions {
                    tsumo: false,
                    riichi: self.players[w_pid as usize].riichi_declared,
                    double_riichi: self.players[w_pid as usize].double_riichi_declared,
                    ippatsu: self.players[w_pid as usize].ippatsu_cycle,
                    haitei: false,
                    houtei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
                    rinshan: false,
                    chankan: is_chankan,
                    tsumo_first_turn: false,
                    player_wind: Wind::from(p_wind),
                    round_wind: Wind::from(self.round_wind),
                    riichi_sticks: self.riichi_sticks,
                    honba: ron_honba,
                    ..Default::default()
                };

                let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);
                let ura_indicators = if self.players[w_pid as usize].riichi_declared {
                    self._get_ura_indicators()
                } else {
                    vec![]
                };
                let mut res = calc.calc(
                    win_tile,
                    self.wall.dora_indicator_slice(),
                    &ura_indicators,
                    Some(cond),
                );

                // Cap double yakuman patterns when not enabled per rule flags
                if res.yakuman && res.han > 13 {
                    let mut cap = 0u32;
                    for &y in res.yaku_slice() {
                        match y {
                            47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                            48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                            49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                            50 if !self.rule.is_daisuushii_double => cap += 13,
                            _ => {}
                        }
                    }
                    if cap > 0 {
                        res.han = res.han.saturating_sub(cap).max(13);
                        let capped = crate::score::calculate_score(
                            res.han as u8,
                            0,
                            w_pid == self.oya,
                            false,
                            ron_honba,
                            np as u8,
                        );
                        res.ron_agari = capped.pay_ron;
                        res.tsumo_agari_oya = capped.pay_tsumo_oya;
                        res.tsumo_agari_ko = capped.pay_tsumo_ko;
                    }
                }

                if res.is_win {
                    let score = res.ron_agari as i32;

                    let mut pao_payer = target_pid;
                    let mut pao_amt = 0i32;

                    if res.yakuman {
                        let mut has_pao = false;
                        let mut total_yakuman_val = 0i32;
                        let mut pao_yakuman_val = 0i32;

                        for &yid in res.yaku_slice() {
                            let val: i32 = match yid {
                                47 if self.rule.is_junsei_chuurenpoutou_double => 2,
                                48 if self.rule.is_suuankou_tanki_double => 2,
                                49 if self.rule.is_kokushi_musou_13machi_double => 2,
                                50 if self.rule.is_daisuushii_double => 2,
                                _ => 1,
                            };
                            total_yakuman_val += val;
                            if let Some(liable) = self.players[w_pid as usize].pao_get(yid as u8) {
                                has_pao = true;
                                pao_payer = liable;
                                pao_yakuman_val += val;
                            }
                        }

                        if has_pao {
                            let is_oya = w_pid == self.oya;
                            let unit: i32 = if is_oya { 48000 } else { 32000 };
                            let honba_ron = ron_honba as i32 * (np as i32 - 1) * 100;

                            // Ron with PAO: split between PAO player and deal-in player.
                            // yakuman_pao_is_liability_only controls the split base:
                            //   true  (MjSoul): only PAO-triggering yakuman portion split 50/50
                            //   false (Tenhou): total yakuman split 50/50
                            let split_base = if self.rule.yakuman_pao_is_liability_only {
                                pao_yakuman_val * unit
                            } else {
                                total_yakuman_val * unit
                            };
                            pao_amt = split_base / 2 + honba_ron;
                        }
                    }

                    let mut this_deltas = [0i32; NP];
                    this_deltas[w_pid as usize] += score;
                    this_deltas[pao_payer as usize] -= pao_amt;
                    this_deltas[target_pid as usize] -= score - pao_amt;

                    total_deltas[w_pid as usize] += score;
                    total_deltas[pao_payer as usize] -= pao_amt;
                    total_deltas[target_pid as usize] -= score - pao_amt;

                    if !deposit_taken {
                        let stick_pts = (self.riichi_sticks * 1000) as i32;
                        total_deltas[w_pid as usize] += stick_pts;
                        this_deltas[w_pid as usize] += stick_pts;
                        self.riichi_sticks = 0;
                        deposit_taken = true;
                    }

                    let mut val = res;
                    for i in 0..self.players[w_pid as usize].pao_count as usize {
                        let (yid, liable) = self.players[w_pid as usize].pao[i];
                        if val.yaku_slice().contains(&(yid as u32)) {
                            val.pao_payer = Some(liable);
                            break;
                        }
                    }
                    self.win_results[w_pid as usize] = Some(val);

                    if w_pid == self.oya {
                        oya_won = true;
                    }

                    if !self.skip_mjai_logging {
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(w_pid.into()));
                        ev.insert("target".to_string(), Value::Number(target_pid.into()));
                        ev.insert(
                            "deltas".to_string(),
                            // SAFETY: serialization of Vec<i32> never fails
                            serde_json::to_value(this_deltas).unwrap(),
                        );

                        let mut ura_markers = Vec::new();
                        if self.players[w_pid as usize].riichi_declared {
                            ura_markers = self._get_ura_markers();
                        }
                        ev.insert(
                            "ura_markers".to_string(),
                            // SAFETY: serialization of Vec<String> never fails
                            serde_json::to_value(&ura_markers).unwrap(),
                        );

                        self._push_mjai_event(Value::Object(ev));
                    }
                }
            }

            for (i, p) in self.players.iter_mut().enumerate() {
                p.score += total_deltas[i];
                p.score_delta = total_deltas[i];
            }

            self._initialize_next_round(oya_won, false);
        } else if let Some((claimer, action)) = call_claim {
            self._accept_riichi();
            self.is_rinshan_flag = false;
            self.is_first_turn = false;
            self.players[claimer as usize].missed_agari_doujun = false;

            // Discard was called -> discarder loses nagashi eligibility
            if let Some((discarder_pid, _)) = self.last_discard {
                self.players[discarder_pid as usize].nagashi_eligible = false;
            }

            for p in 0..np {
                self.players[p].ippatsu_cycle = false;
            }

            if action.action_type == ActionType::Daiminkan {
                self.current_player = claimer;
                self.set_single_active_player(claimer);
                self.players[claimer as usize].clear_forbidden();
                // Handled exclusively by _resolve_kan
                self._resolve_kan(claimer, action);
                return; // Skip the rest of claim handling (Pon/Chi)
            }

            for &t in action.consume_slice() {
                if let Some(idx) = self.players[claimer as usize]
                    .hand
                    .iter()
                    .position(|&x| x == t)
                {
                    self.players[claimer as usize].remove_hand(idx);
                }
            }
            // SAFETY: last_discard is always Some when processing claim actions (pon/chi/kan)
            let (discarder, tile) = self.last_discard.unwrap();
            let (tiles_buf, tiles_len) = copy_and_sorted_insert(action.consume_slice(), tile);
            let meld_type = match action.action_type {
                ActionType::Pon => MeldType::Pon,
                ActionType::Chi => MeldType::Chi,
                _ => MeldType::Chi, // Should not happen for this block anymore
            };
            self.players[claimer as usize].push_meld(Meld::new(
                meld_type,
                &tiles_buf[..tiles_len],
                true,
                discarder as i8,
                Some(tile),
            ));

            if !self.skip_mjai_logging {
                let type_str = match action.action_type {
                    ActionType::Pon => Some("pon"),
                    ActionType::Chi => Some("chi"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    _ => None,
                };
                if let Some(s) = type_str {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), serde_json::Value::String(s.to_string()));
                    ev.insert(
                        "actor".to_string(),
                        serde_json::Value::Number(claimer.into()),
                    );
                    ev.insert(
                        "target".to_string(),
                        serde_json::Value::Number(discarder.into()),
                    );
                    ev.insert(
                        "pai".to_string(),
                        serde_json::Value::String(tid_to_mjai(tile)),
                    );
                    let cons_strs: Vec<String> = action
                        .consume_slice()
                        .iter()
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        // SAFETY: serialization of Vec<String> never fails
                        serde_json::to_value(cons_strs).unwrap(),
                    );
                    self._push_mjai_event(serde_json::Value::Object(ev));
                }
            }

            // PAO implementation
            if meld_type == MeldType::Pon
                || meld_type == MeldType::Daiminkan
                || meld_type == MeldType::Kakan
            {
                let tile_val = tile / 4;
                if (31..=33).contains(&tile_val) {
                    let dragon_melds = self.players[claimer as usize]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if dragon_melds == 3 {
                        self.players[claimer as usize].pao_insert(37, discarder);
                    }
                } else if (27..=30).contains(&tile_val) {
                    let wind_melds = self.players[claimer as usize]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if wind_melds == 4 {
                        self.players[claimer as usize].pao_insert(50, discarder);
                    }
                }
            }

            self.current_player = claimer;
            self.phase = Phase::WaitAct;
            self.set_single_active_player(claimer);
            self.players[claimer as usize].clear_forbidden();

            if action.action_type == ActionType::Pon {
                self.players[claimer as usize].push_forbidden(tile);
            } else if action.action_type == ActionType::Chi {
                self.players[claimer as usize].push_forbidden(tile);
                let t34 = tile / 4;
                let cs = action.consume_slice();
                let mut consumed_34 = [cs[0] / 4, cs[1] / 4];
                consumed_34.sort();
                if consumed_34[0] == t34 + 1 && consumed_34[1] == t34 + 2 {
                    if t34 % 9 <= 5 {
                        self.players[claimer as usize].push_forbidden((t34 + 3) * 4);
                    }
                } else if t34 >= 2
                    && consumed_34[1] == t34 - 1
                    && consumed_34[0] == t34 - 2
                    && t34 % 9 >= 3
                {
                    self.players[claimer as usize].push_forbidden((t34 - 3) * 4);
                }
            }

            if action.action_type == ActionType::Daiminkan {
                self._resolve_kan(claimer, action);
            } else {
                self.needs_tsumo = false;
                self.drawn_tile = None;
            }
        } else {
            // All Pass
            self.clear_claims();
            self.clear_active_players();

            if let Some((pk_pid, pk_act)) = self.pending_kan.take() {
                self._resolve_kan(pk_pid, pk_act);
            } else {
                self._accept_riichi();
                self.turn_count += 1;
                self.current_player = (self.current_player + 1) % np as u8;
                self._deal_next();
                if self.turn_count >= np as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }
    /// Step with array-indexed actions instead of HashMap.
    ///
    /// `actions[pid]` = `Some(action)` if player pid has an action, `None` otherwise.
    /// Thin wrapper that converts to HashMap and delegates to `step()`.
    pub fn step_array(&mut self, actions: &[Option<Action>; 4]) {
        let mut map = std::collections::HashMap::with_capacity(4);
        for (pid, act) in actions.iter().enumerate() {
            if let Some(a) = act {
                map.insert(pid as u8, *a);
            }
        }
        self.step(&map);
    }

    #[inline]
    /// Step without validating actions against legal moves.
    ///
    /// For trusted self-play only -- caller guarantees actions are legal.
    /// Using this with illegal actions will corrupt game state.
    pub fn step_unchecked(&mut self, actions: &[Option<Action>; 4]) {
        if self.is_done {
            return;
        }

        if self.needs_initialize_next_round {
            self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
            return;
        }

        self._execute_step_array(actions);
    }

    /// Unchecked step with array-indexed actions instead of HashMap.
    ///
    /// `actions[pid]` = `Some(action)` if player pid has an action, `None` otherwise.
    /// For trusted self-play only -- caller guarantees actions are legal.
    pub fn step_array_unchecked(&mut self, actions: &[Option<Action>; 4]) {
        self.step_unchecked(actions);
    }

    fn _resolve_discard(&mut self, pid: u8, tile: u8, tsumogiri: bool) {
        // After a discard the rinshan context is over. Clearing here ensures
        // that houtei (last-discard win) is correctly detected even when the
        // discard comes after a kan draw.
        self.is_rinshan_flag = false;

        // Clear ippatsu for the discarding player. When a riichi player discards
        // without tsumo winning, their ippatsu window is over. Note: the riichi
        // declaration discard won't wrongly clear it because _accept_riichi() runs
        // AFTER this and sets ippatsu_cycle = true.
        self.players[pid as usize].ippatsu_cycle = false;
        let riichi_stage = self.players[pid as usize].riichi_stage;
        self.players[pid as usize].push_discard(tile, !tsumogiri, riichi_stage);
        self.last_discard = Some((pid, tile));
        self.drawn_tile = None;

        // Track last tedashi (hand discard, not tsumogiri)
        if !tsumogiri {
            self.last_tedashis[pid as usize] = Some(tile);
        }

        self.needs_tsumo = true;

        if self.players[pid as usize].riichi_stage {
            self.players[pid as usize].riichi_declared = true;
            if self.is_first_turn {
                self.players[pid as usize].double_riichi_declared = true;
            }
            self.players[pid as usize].riichi_declaration_index =
                Some(self.players[pid as usize].discard_len as usize - 1);
            self.players[pid as usize].riichi_stage = false;
            self.riichi_pending_acceptance = Some(pid);
        }

        // Tenhou: reveal pending kan doras before dahai event
        if !self.rule.open_kan_dora_after_discard {
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("dahai".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
            self._push_mjai_event(Value::Object(ev));
        }

        // MjSoul: reveal pending kan doras after dahai event
        if self.rule.open_kan_dora_after_discard {
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }
        }

        self.players[pid as usize].missed_agari_doujun = false;
        self.players[pid as usize].nagashi_eligible &= crate::types::is_terminal_tile(tile);

        self.clear_claims();
        self.clear_active_players();
        let mut has_claims = false;
        let mut claim_active = [0u8; 3];
        let mut claim_count: usize = 0;

        // Loop players for claim actions
        let np = NP;
        for i in 0..np as u8 {
            if i == pid {
                continue;
            }
            let (count, missed_agari) = self._get_claim_actions_into_claims(i, pid, tile);
            if missed_agari {
                self.players[i as usize].missed_agari_doujun = true;
            }
            if count > 0 {
                has_claims = true;
                claim_active[claim_count] = i;
                claim_count += 1;
                // claims already set directly by _get_claim_actions_into_claims
            }
        }

        if has_claims {
            self.phase = Phase::WaitResponse;
            self.set_active_players_from_slice(&claim_active[..claim_count]);
        } else {
            if let Some(_rp) = self.riichi_pending_acceptance {
                self._accept_riichi();
            }
            if !self.check_abortive_draw() {
                self.turn_count += 1;
                self.current_player = (pid + 1) % np as u8;
                self._deal_next();
                if self.turn_count >= np as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }

    /// Resolve a kan action (ankan, daiminkan, or kakan) and draw a rinshan tile.
    pub fn _resolve_kan(&mut self, pid: u8, action: Action) {
        let p_idx = pid as usize;
        if action.action_type == ActionType::Kakan {
            // Hand and melds were already updated in step() to keep observations in sync
        } else {
            // Ankan / Daiminkan
            for &t in action.consume_slice() {
                let pos = self.players[p_idx].hand_slice().partition_point(|&x| x < t);
                if pos < self.players[p_idx].hand_len as usize && self.players[p_idx].hand[pos] == t
                {
                    self.players[p_idx].remove_hand(pos);
                }
            }
            let (m_type, tiles_buf, tiles_len, from_who, ct) =
                if action.action_type == ActionType::Ankan {
                    let src = action.consume_slice();
                    let mut buf = [0u8; 5];
                    let n = src.len().min(5);
                    buf[..n].copy_from_slice(&src[..n]);
                    (MeldType::Ankan, buf, n, -1i8, None)
                } else {
                    // SAFETY: last_discard is always Some when processing daiminkan claims
                    let (discarder, tile) = self.last_discard.unwrap();
                    let (buf, n) = copy_and_sorted_insert(action.consume_slice(), tile);
                    (MeldType::Daiminkan, buf, n, discarder as i8, Some(tile))
                };
            self.players[p_idx].push_meld(Meld::new(
                m_type,
                &tiles_buf[..tiles_len],
                m_type == MeldType::Daiminkan,
                from_who,
                ct,
            ));

            // PAO check for Daiminkan
            if action.action_type == ActionType::Daiminkan {
                // SAFETY: last_discard is always Some when processing daiminkan claims
                let (discarder, tile) = self.last_discard.unwrap();
                let tile_val = tile / 4;
                if (31..=33).contains(&tile_val) {
                    let dragon_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if dragon_melds == 3 {
                        self.players[p_idx].pao_insert(37, discarder);
                    }
                } else if (27..=30).contains(&tile_val) {
                    let wind_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if wind_melds == 4 {
                        self.players[p_idx].pao_insert(50, discarder);
                    }
                }
            }
        }

        self.is_first_turn = false;
        for p in &mut self.players {
            p.ippatsu_cycle = false;
        }

        if self.wall.remaining() > 14 {
            // Rinshan tiles drawn via cursor (no memmove)
            let t = self.wall.draw_rinshan();
            sorted_insert_arr(
                &mut self.players[p_idx].hand,
                &mut self.players[p_idx].hand_len,
                t,
            );
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            if !self.skip_mjai_logging {
                let m_type = match action.action_type {
                    ActionType::Ankan => Some("ankan"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    ActionType::Kakan => None, // Logged in step()
                    _ => None,
                };
                if let Some(s) = m_type {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String(s.to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    if action.action_type == ActionType::Ankan {
                        let tile = action.tile.unwrap_or_else(|| action.consume_tiles[0]);
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                    } else if action.action_type == ActionType::Daiminkan {
                        if let Some((target, tile)) = self.last_discard {
                            ev.insert("target".to_string(), Value::Number(target.into()));
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                        }
                    }
                    let cons_strs: Vec<String> = action
                        .consume_slice()
                        .iter()
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        // SAFETY: serialization of Vec<String> never fails
                        serde_json::to_value(cons_strs).unwrap(),
                    );
                    self._push_mjai_event(Value::Object(ev));
                }
            }

            // Reveal any pending doras from previous kans
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }

            // Ankan: always reveal dora immediately (before rinshan tsumo)
            // Daiminkan/Kakan: defer dora reveal to after discard
            if action.action_type == ActionType::Ankan {
                self._reveal_kan_dora();
            } else {
                self.wall.pending_kan_dora_count += 1;
            }

            if !self.skip_mjai_logging {
                // Rinshan tsumo logging should apply to Kakan as well
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }
            self.phase = Phase::WaitAct;
            self.set_single_active_player(pid);
        }
    }

    fn _accept_riichi(&mut self) {
        if let Some(p) = self.riichi_pending_acceptance {
            self.players[p as usize].score -= 1000;
            self.players[p as usize].score_delta -= 1000;
            self.riichi_sticks += 1;
            self.players[p as usize].riichi_declared = true;
            self.players[p as usize].ippatsu_cycle = true;
            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(p.into()));
                self._push_mjai_event(Value::Object(ev));
            }
            self.riichi_pending_acceptance = None;
        }
    }

    /// Deal the next tile to the current player, or trigger exhaustive draw.
    pub fn _deal_next(&mut self) {
        self.is_rinshan_flag = false;
        if self.wall.remaining() <= 14 {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }
        if let Some(t) = self.wall.draw_back() {
            let pid = self.current_player;
            sorted_insert_arr(
                &mut self.players[pid as usize].hand,
                &mut self.players[pid as usize].hand_len,
                t,
            );
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;
            self.phase = Phase::WaitAct;
            self.set_single_active_player(pid);

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
            self.players[pid as usize].clear_forbidden();
        }
    }

    /// Advance to the next round or end the game based on scores and wind rotation.
    pub fn _initialize_next_round(&mut self, oya_won: bool, is_draw: bool) {
        if self.is_done {
            return;
        }

        let np: u8 = NP as u8;

        // Tobi (bankruptcy) check: game ends if any player has negative score
        if self.players.iter().any(|p| p.score < 0) {
            self._process_end_game();
            return;
        }

        let mut next_honba = self.honba;
        let mut next_oya = self.oya;
        let mut next_round_wind = self.round_wind;

        if oya_won {
            next_honba = next_honba.saturating_add(1);
        } else if is_draw {
            next_honba = next_honba.saturating_add(1);
            next_oya = (next_oya + 1) % np;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        } else {
            next_honba = 0;
            next_oya = (next_oya + 1) % np;
            if next_oya == 0 {
                next_round_wind += 1;
            }
        }

        match self.game_mode {
            1 | 4 => {
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 1 && (max_score >= 30000 || next_round_wind > 1) {
                    self._process_end_game();
                    return;
                }
            }
            2 | 5 => {
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 2 && (max_score >= 30000 || next_round_wind > 2) {
                    self._process_end_game();
                    return;
                }
            }
            0 | 3 => {
                self._process_end_game();
                return;
            }
            _ => {
                if next_round_wind >= 1 {
                    self._process_end_game();
                    return;
                }
            }
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }

        let next_scores: Vec<i32> = self.players.iter().map(|p| p.score).collect();
        let next_sticks = self.riichi_sticks;
        self._initialize_round(
            next_oya,
            next_round_wind,
            next_honba,
            next_sticks,
            None,
            Some(next_scores),
        );
    }

    /// Initialize a round with the given parameters, shuffle the wall, and deal hands.
    pub fn _initialize_round(
        &mut self,
        oya: u8,
        round_wind: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<Vec<i32>>,
    ) {
        let np = NP;
        self.oya = oya;
        self.kyoku_idx = oya;
        self.current_player = oya;
        self.honba = honba;
        self.riichi_sticks = kyotaku;
        self.round_wind = round_wind;

        for p in &mut self.players {
            p.reset_round();
        }
        self.is_done = false;
        self.clear_claims();
        self.pending_kan = None;
        self.is_rinshan_flag = false;
        self.wall.rinshan_draw_count = 0;
        self.wall.pending_kan_dora_count = 0;
        self.is_first_turn = true;
        self.riichi_pending_acceptance = None;
        self.turn_count = 0;
        self.needs_tsumo = true;
        self.needs_initialize_next_round = false;
        self.pending_oya_won = false;
        self.pending_is_draw = false;
        self.last_discard = None;
        self.win_results = Default::default();
        self.last_win_results = Default::default();
        self.riichi_sutehais = [None; NP];
        self.last_tedashis = [None; NP];

        if let Some(s) = scores {
            for (i, &sc) in s.iter().enumerate() {
                if i < self.players.len() {
                    self.players[i].score = sc;
                }
            }
        }

        if let Some(w) = wall {
            self.wall.load_wall(w);
        } else {
            self.wall.shuffle(self.skip_mjai_logging);
        }

        // Deal logic
        for _ in 0..3 {
            for idx in 0..np {
                let p = (idx + oya as usize) % np;
                for _ in 0..4 {
                    if let Some(t) = self.wall.draw_back() {
                        self.players[p].push_hand(t);
                    }
                }
            }
        }
        for idx in 0..np {
            let p = (idx + oya as usize) % np;
            if let Some(t) = self.wall.draw_back() {
                self.players[p].push_hand(t);
            }
        }
        for p in &mut self.players {
            p.hand_slice_mut().sort();
        }

        if !self.skip_mjai_logging {
            let wind_str = match round_wind % 4 {
                0 => "E",
                1 => "S",
                2 => "W",
                _ => "N",
            };
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_kyoku".to_string()));
            ev.insert("bakaze".to_string(), Value::String(wind_str.to_string()));
            ev.insert("kyoku".to_string(), Value::Number((oya + 1).into()));
            ev.insert("honba".to_string(), Value::Number(honba.into()));
            ev.insert("kyotaku".to_string(), Value::Number(kyotaku.into()));
            ev.insert("oya".to_string(), Value::Number(oya.into()));
            let scores_vec: Vec<i32> = self.players.iter().map(|p| p.score).collect();
            ev.insert(
                "scores".to_string(),
                // SAFETY: serialization of Vec<i32> never fails
                serde_json::to_value(scores_vec).unwrap(),
            );
            ev.insert(
                "dora_marker".to_string(),
                Value::String(tid_to_mjai(self.wall.dora_indicators[0])),
            );

            let mut tehais = Vec::new();
            for p in &self.players {
                let hand_strs: Vec<String> =
                    p.hand_slice().iter().map(|&t| tid_to_mjai(t)).collect();
                tehais.push(hand_strs);
            }
            // SAFETY: serialization of Vec<Vec<String>> never fails
            ev.insert("tehais".to_string(), serde_json::to_value(tehais).unwrap());

            self._push_mjai_event(Value::Object(ev));
        }

        self.current_player = self.oya;
        self.phase = Phase::WaitAct;
        self.set_single_active_player(self.oya);

        // Draw 14th tile for Oya
        if let Some(t) = self.wall.draw_back() {
            sorted_insert_arr(
                &mut self.players[self.oya as usize].hand,
                &mut self.players[self.oya as usize].hand_len,
                t,
            );
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(self.oya.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
        } else {
            self.needs_tsumo = true;
            self.drawn_tile = None;
        }
    }

    /// Trigger a draw (ryuukyoku) for the given reason and settle scores.
    pub fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi();

        let np = NP;
        let mut tenpai = vec![false; np];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

        if reason == "exhaustive_draw" {
            for (i, p) in self.players.iter().enumerate() {
                let calc = crate::hand_evaluator::HandEvaluator::new(&p.hand, &p.melds);
                if calc.is_tenpai() {
                    tenpai[i] = true;
                }
            }
            for (i, p) in self.players.iter().enumerate() {
                if p.nagashi_eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                final_reason = "nagashimangan".to_string();
                // Apply mangan tsumo payment for each nagashi winner (no honba)
                for &w in &nagashi_winners {
                    let is_oya = w == self.oya;
                    let score_res = crate::score::calculate_score(5, 30, is_oya, true, 0, np as u8);
                    if is_oya {
                        for i in 0..np {
                            if i as u8 != w {
                                self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                self.players[i].score_delta -= score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score_delta +=
                                    score_res.pay_tsumo_ko as i32;
                            }
                        }
                    } else {
                        for i in 0..np {
                            if i as u8 != w {
                                let pay = if i as u8 == self.oya {
                                    score_res.pay_tsumo_oya as i32
                                } else {
                                    score_res.pay_tsumo_ko as i32
                                };
                                self.players[i].score -= pay;
                                self.players[i].score_delta -= pay;
                                self.players[w as usize].score += pay;
                                self.players[w as usize].score_delta += pay;
                            }
                        }
                    }
                }
            } else {
                let tenpai_pool = 3000;
                let num_tp = tenpai.iter().filter(|&&t| t).count();
                if num_tp > 0 && num_tp < np {
                    let pk = tenpai_pool / num_tp as i32;
                    let pn = tenpai_pool / (np - num_tp) as i32;
                    for (i, tp) in tenpai.iter().enumerate() {
                        let delta = if *tp { pk } else { -pn };
                        self.players[i].score += delta;
                        self.players[i].score_delta = delta;
                    }
                }
            }
        } else if let Some(stripped) = reason.strip_prefix("Error: Illegal Action by Player ") {
            if let Ok(pid) = stripped.parse::<usize>() {
                if pid < np {
                    let is_offender_oya = (pid as u8) == self.oya;
                    if is_offender_oya {
                        let penalty = 4000 * (np as i32 - 1);
                        let each_get = penalty / (np as i32 - 1);
                        for i in 0..np {
                            if i == pid {
                                self.players[i].score -= penalty;
                                self.players[i].score_delta = -penalty;
                            } else {
                                self.players[i].score += each_get;
                                self.players[i].score_delta = each_get;
                            }
                        }
                    } else {
                        let total_penalty = 4000 + 2000 * (np as i32 - 2);
                        for i in 0..np {
                            if i == pid {
                                self.players[i].score -= total_penalty;
                                self.players[i].score_delta = -total_penalty;
                            } else if (i as u8) == self.oya {
                                self.players[i].score += 4000;
                                self.players[i].score_delta = 4000;
                            } else {
                                self.players[i].score += 2000;
                                self.players[i].score_delta = 2000;
                            }
                        }
                    }
                }
            }
        }

        let is_renchan = if final_reason == "exhaustive_draw" {
            tenpai[self.oya as usize]
        } else if final_reason == "nagashimangan" {
            nagashi_winners.contains(&self.oya)
        } else {
            true
        };

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
            ev.insert("reason".to_string(), Value::String(final_reason.clone()));
            let deltas: Vec<i32> = self.players.iter().map(|p| p.score_delta).collect();
            // SAFETY: serialization of Vec<i32> never fails
            ev.insert("deltas".to_string(), serde_json::to_value(deltas).unwrap());
            self._push_mjai_event(Value::Object(ev));
        }

        self._initialize_next_round(is_renchan, true);
    }

    fn check_abortive_draw(&mut self) -> bool {
        // 1. Sufuurenta (Four Winds)
        let turns_ok = self.players.iter().all(|p| p.discard_len == 1);
        let melds_empty = self.players.iter().all(|p| p.meld_count == 0);

        if turns_ok && melds_empty {
            if let Some(&first_tile) = self.players[0].discards_slice().first() {
                let first = first_tile / 4;
                if (27..=30).contains(&first)
                    && self
                        .players
                        .iter()
                        .all(|p| p.discards_slice().first().map(|&t| t / 4) == Some(first))
                {
                    self._trigger_ryukyoku("sufuurenta");
                    return true;
                }
            }
        }

        // 2. Suukansansen (4 Kans)
        let mut kan_owners = Vec::new();
        for (pid, p) in self.players.iter().enumerate() {
            for m in &p.melds {
                if m.meld_type == crate::types::MeldType::Daiminkan
                    || m.meld_type == crate::types::MeldType::Ankan
                    || m.meld_type == crate::types::MeldType::Kakan
                {
                    kan_owners.push(pid);
                }
            }
        }

        if kan_owners.len() == 4 {
            let first_owner = kan_owners[0];
            if !kan_owners.iter().all(|&o| o == first_owner) {
                self._trigger_ryukyoku("suukansansen");
                return true;
            }
        }

        // 3. Suucha Riichi (Four Riichis)
        if self.players.iter().all(|p| p.riichi_declared) {
            self._trigger_ryukyoku("suucha_riichi");
            return true;
        }

        false
    }

    /// Reveal the next kan dora indicator from the dead wall.
    pub fn _reveal_kan_dora(&mut self) {
        let count = self.wall.dora_indicator_count as usize;
        if count < 5 {
            // Base indices for Omote Dora are 4, 6, 8, 10, 12 in the wall.
            // With draw_cursor, tiles stay in place so indices are stable.
            let base_idx = 4 + 2 * count;
            if base_idx < self.wall.tile_count as usize {
                let new_dora = self.wall.tiles[base_idx];
                self.wall.push_dora_indicator(new_dora);
                if !self.skip_mjai_logging {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("dora".to_string()));
                    ev.insert(
                        "dora_marker".to_string(),
                        Value::String(tid_to_mjai(new_dora)),
                    );
                    self._push_mjai_event(Value::Object(ev));
                }
            }
        }
    }

    fn _get_ura_indicators(&self) -> Vec<u8> {
        let mut indicators = Vec::new();
        for i in 0..self.wall.dora_indicator_count as usize {
            let idx = 5 + 2 * i;
            if idx < self.wall.tile_count as usize {
                indicators.push(self.wall.tiles[idx]);
            }
        }
        indicators
    }

    /// Return ura-dora indicator tiles as MJAI notation strings.
    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for i in 0..self.wall.dora_indicator_count as usize {
            let idx = 5 + 2 * i;
            if idx < self.wall.tile_count as usize {
                markers.push(tid_to_mjai(self.wall.tiles[idx]));
            }
        }
        markers
    }

    /// Mark the game as done and log the end-game event.
    pub(crate) fn _process_end_game(&mut self) {
        self.is_done = true;
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    /// Apply a typed MJAI event to advance game state.
    pub fn apply_mjai_event(&mut self, event: MjaiEvent) {
        <Self as GameStateEventHandler>::apply_mjai_event(self, event)
    }

    /// Apply a replay log action to advance game state.
    pub fn apply_log_action(&mut self, action: &LogAction) {
        <Self as GameStateEventHandler>::apply_log_action(self, action)
    }
}

impl GameState {
    /// Append a JSON MJAI event to global and per-player logs.
    pub fn _push_mjai_event(&mut self, event: Value) {
        if self.skip_mjai_logging {
            return;
        }
        // SAFETY: serialization of serde_json::Value always succeeds
        let json_str = serde_json::to_string(&event).unwrap();
        self.mjai_log.push(json_str.clone());

        let type_str = event["type"].as_str().unwrap_or("");
        let actor = event["actor"].as_u64().map(|a| a as usize);

        let np = NP;
        for pid in 0..np {
            let should_push = true;
            let mut final_json = json_str.clone();

            if type_str == "start_kyoku" {
                if let Some(tehais_val) = event.get("tehais").and_then(|v| v.as_array()) {
                    let mut masked_tehais = Vec::new();
                    for (i, hand_val) in tehais_val.iter().enumerate() {
                        if i == pid {
                            masked_tehais.push(hand_val.clone());
                        } else {
                            let len = hand_val.as_array().map(|a| a.len()).unwrap_or(13);
                            let masked = vec!["?".to_string(); len];
                            // SAFETY: serialization of Vec<String> never fails
                            masked_tehais.push(serde_json::to_value(masked).unwrap());
                        }
                    }
                    // SAFETY: event was constructed as Value::Object, so as_object() always succeeds
                    let mut masked_event = event.as_object().unwrap().clone();
                    masked_event.insert("tehais".to_string(), Value::Array(masked_tehais));
                    // SAFETY: serialization of serde_json::Value always succeeds
                    final_json = serde_json::to_string(&Value::Object(masked_event)).unwrap();
                }
            } else if type_str == "tsumo" {
                if let Some(act_id) = actor {
                    if act_id != pid {
                        // SAFETY: event was constructed as Value::Object, so as_object() always succeeds
                        let mut masked_event = event.as_object().unwrap().clone();
                        masked_event.insert("pai".to_string(), Value::String("?".to_string()));
                        // SAFETY: serialization of serde_json::Value always succeeds
                        final_json = serde_json::to_string(&Value::Object(masked_event)).unwrap();
                    }
                }
            }

            if should_push {
                self.mjai_log_per_player[pid].push(final_json);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_state() -> GameState {
        GameState::new(1, false, Some(7), 0, GameRule::default_tenhou())
    }

    fn parsed_tile(text: &str) -> u8 {
        crate::parser::parse_tile(text).expect("test tile should parse")
    }

    fn tiles_to_u32(tiles: &[u8]) -> Vec<u32> {
        tiles.iter().map(|&tile| tile as u32).collect()
    }

    #[test]
    fn sorted_insert_helpers_keep_tiles_ordered_across_edge_positions() {
        let mut hand = [0u8; 14];
        hand[..4].copy_from_slice(&[4, 12, 20, 28]);
        let mut hand_len = 4;

        sorted_insert_arr(&mut hand, &mut hand_len, 2);
        sorted_insert_arr(&mut hand, &mut hand_len, 16);
        sorted_insert_arr(&mut hand, &mut hand_len, 40);

        assert_eq!(&hand[..hand_len as usize], &[2, 4, 12, 16, 20, 28, 40]);

        let (insert_front, front_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 4);
        let (insert_middle, middle_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 14);
        let (insert_back, back_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 24);

        assert_eq!(&insert_front[..front_len], &[4, 8, 12, 16, 20]);
        assert_eq!(&insert_middle[..middle_len], &[8, 12, 14, 16, 20]);
        assert_eq!(&insert_back[..back_len], &[8, 12, 16, 20, 24]);
    }

    #[test]
    fn helper_methods_manage_active_players_and_claims() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());

        state.set_single_active_player(3);
        assert_eq!(state.active_player_slice(), &[3]);

        state.set_active_players_from_slice(&[1, 2, 3]);
        assert_eq!(state.active_player_slice(), &[1, 2, 3]);

        state.clear_active_players();
        assert!(state.active_player_slice().is_empty());

        let ron = Action::new(ActionType::Ron, Some(88), &[], Some(1));
        let pon = Action::new(ActionType::Pon, Some(88), &[84, 85], Some(1));
        state.push_claim(1, ron);
        state.push_claim(1, pon);
        assert_eq!(state.claims_slice(1), &[ron, pon]);

        state.clear_claims();
        assert_eq!(state.current_claim_counts, [0; 4]);
        assert!(state.claims_slice(1).is_empty());
    }

    #[test]
    fn replay_ankan_matcher_accepts_same_tile_class_with_different_copy_ids() {
        let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay = Action::new(ActionType::Ankan, Some(17), &[17, 17, 17, 17], Some(0));

        assert!(GameState::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_kakan_matcher_accepts_same_tile_class_with_different_copy_ids() {
        let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
        let replay = Action::new(ActionType::Kakan, Some(17), &[], Some(0));

        assert!(GameState::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_discard_matcher_accepts_same_tile_class_with_different_copy_ids() {
        let legal = Action::new(ActionType::Discard, Some(44), &[], Some(0));
        let replay = Action::new(ActionType::Discard, Some(46), &[], Some(0));

        assert!(GameState::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_discard_matcher_distinguishes_plain_and_red_fives() {
        let legal_red = Action::new(ActionType::Discard, Some(52), &[], Some(0));
        let replay_plain = Action::new(ActionType::Discard, Some(53), &[], Some(0));
        let legal_plain = Action::new(ActionType::Discard, Some(54), &[], Some(0));
        let replay_red = Action::new(ActionType::Discard, Some(52), &[], Some(0));

        assert!(!GameState::replay_action_matches_legal(
            &legal_red,
            &replay_plain
        ));
        assert!(GameState::replay_action_matches_legal(
            &legal_plain,
            &replay_plain
        ));
        assert!(!GameState::replay_action_matches_legal(
            &legal_plain,
            &replay_red
        ));
    }

    #[test]
    fn replay_kan_matcher_rejects_different_tile_classes() {
        let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay = Action::new(ActionType::Ankan, Some(20), &[20, 20, 20, 20], Some(0));

        assert!(!GameState::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_matcher_accepts_context_implied_actions_and_kans_with_matching_consumes() {
        let legal_riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
        let replay_riichi = Action::new(ActionType::Riichi, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_riichi,
            &replay_riichi
        ));

        let legal_kita = Action::new(ActionType::Kita, Some(120), &[], Some(0));
        let replay_kita = Action::new(ActionType::Kita, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_kita,
            &replay_kita
        ));

        let legal_ankan = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay_ankan = Action::new(ActionType::Ankan, Some(99), &[16, 17, 18, 19], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_ankan,
            &replay_ankan
        ));
    }

    #[test]
    fn reset_and_reset_for_new_game_restore_logging_and_seeded_state() {
        let mut state = fresh_state();
        state.mjai_log.push("junk".to_string());
        state.mjai_log_per_player[0].push("junk".to_string());
        state.player_event_counts = [3, 2, 1, 0];
        state.mjai_events.clear();
        state.reset();

        assert_eq!(state.player_event_counts, [0; 4]);
        assert_eq!(state.mjai_log.len(), 1);
        assert_eq!(state.mjai_log_per_player[0].len(), 1);
        assert_eq!(state.mjai_events.len(), 1);

        state.players[0].score = 1234;
        state.turn_count = 99;
        state.reset_for_new_game(Some(99));
        assert_eq!(state.seed, Some(99));
        assert_eq!(state.turn_count, 0);
        assert_eq!(state.players[0].score, state.mode.starting_score());
        assert!(!state.mjai_log.is_empty());
    }

    #[test]
    fn get_observation_masks_other_hands_and_drains_only_new_events() {
        let mut state = fresh_state();
        state.mjai_log_per_player[0] = vec![
            "already-seen".to_string(),
            "fresh-event-a".to_string(),
            "fresh-event-b".to_string(),
        ];
        state.player_event_counts[0] = 1;
        state.riichi_sutehais[0] = Some(12);
        state.last_tedashis[0] = Some(16);
        state.last_discard = Some((40, 2));

        let expected_hand = tiles_to_u32(state.players[0].hand_slice());

        let obs = state.get_observation(0);

        assert_eq!(obs.hands[0], expected_hand);
        assert!(obs.hands[1].is_empty());
        assert!(obs.hands[2].is_empty());
        assert!(obs.hands[3].is_empty());
        assert_eq!(obs.new_events(), vec!["fresh-event-a", "fresh-event-b"]);
        assert_eq!(obs.riichi_sutehais[0], Some(12));
        assert_eq!(obs.last_tedashis[0], Some(16));
        assert_eq!(obs.last_discard, Some(40));
        assert_eq!(state.player_event_counts[0], 3);

        let obs_again = state.get_observation(0);
        assert!(obs_again.new_events().is_empty());
    }

    #[test]
    fn get_observation_limits_legal_actions_to_visible_turn_owners() {
        let mut state = fresh_state();

        let current_obs = state.get_observation(0);
        assert!(!current_obs.legal_actions_method().is_empty());

        let hidden_obs = state.get_observation(1);
        assert!(hidden_obs.legal_actions_method().is_empty());

        state.phase = Phase::WaitResponse;
        state.active_players = [1, 0, 0, 0];
        state.active_player_count = 1;
        state.current_claim_counts[1] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Ron, Some(12), &[], Some(1));

        let response_obs = state.get_observation(1);
        let response_legals = response_obs.legal_actions_method();
        assert!(response_legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(response_legals
            .iter()
            .any(|action| action.action_type == ActionType::Pass));

        state.is_done = true;
        let done_obs = state.get_observation(1);
        assert!(done_obs.legal_actions_method().is_empty());
    }

    #[test]
    fn observe_and_public_legal_action_wrappers_match_internal_state() {
        let mut state = fresh_state();
        state.players[0].score = 31_500;
        state.players[1].score = 22_100;
        state.players[2].score = 24_200;
        state.players[3].score = 22_200;
        state.players[0].riichi_declared = true;
        state.players[1].riichi_declared = true;
        state.players[0].push_discard(parsed_tile("1m"), true, false);
        state.players[1].push_discard(parsed_tile("2p"), false, true);
        state.honba = 2;
        state.riichi_sticks = 1;
        state.round_wind = 1;
        state.oya = 3;
        state.kyoku_idx = 2;
        state.current_player = 0;

        let expected_legals = state._get_legal_actions_internal(0);
        assert_eq!(state.get_legal_actions(0), expected_legals);

        let mut buf = vec![Action::new(ActionType::Pass, None, &[], Some(3))];
        state.get_legal_actions_into(0, &mut buf);
        assert_eq!(buf, expected_legals);

        let obs = state.observe(0);
        assert_eq!(obs.player_id, 0);
        assert_eq!(obs.observer_hand, state.players[0].hand_slice());
        assert_eq!(obs.melds[0].len(), state.players[0].melds_slice().len());
        assert_eq!(obs.melds[1].len(), state.players[1].melds_slice().len());
        assert_eq!(obs.discards[0], state.players[0].discards_slice());
        assert_eq!(obs.discards[1], state.players[1].discards_slice());
        assert_eq!(obs.dora_indicators, state.wall.dora_indicator_slice());
        assert_eq!(obs.scores, [31_500, 22_100, 24_200, 22_200]);
        assert_eq!(obs.riichi_declared, [true, true, false, false]);
        assert_eq!(obs.honba, 2);
        assert_eq!(obs.riichi_sticks, 1);
        assert_eq!(obs.round_wind, 1);
        assert_eq!(obs.oya, 3);
        assert_eq!(obs.kyoku_index, 2);
        assert_eq!(obs.current_player, 0);
        assert_eq!(obs.drawn_tile, state.drawn_tile);
        assert!(!obs.is_done);
    }

    #[test]
    fn ura_helpers_and_kan_dora_reveal_use_dead_wall_layout() {
        let mut state = fresh_state();
        state.wall.tile_count = 20;
        state.wall.tiles[4] = 16;
        state.wall.tiles[5] = 52;
        state.wall.tiles[6] = 88;
        state.wall.tiles[7] = 108;
        state.wall.dora_indicator_count = 1;

        assert_eq!(state._get_ura_indicators(), vec![52]);
        assert_eq!(state._get_ura_markers(), vec!["5pr".to_string()]);

        state._reveal_kan_dora();
        assert_eq!(state.wall.dora_indicator_slice(), &[4, 88]);
    }

    #[test]
    fn initialize_next_round_ends_single_round_games_and_rotates_east_games() {
        let mut single = GameState::new(0, true, Some(1), 0, GameRule::default_tenhou());
        single.is_done = false;
        single._initialize_next_round(false, false);
        assert!(single.is_done);

        let mut east = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
        east.is_done = false;
        east.oya = 3;
        east.honba = 2;
        east.players.iter_mut().for_each(|p| p.score = 25_000);
        east._initialize_next_round(false, true);
        assert!(!east.is_done);
        assert_eq!(east.oya, 0);
        assert_eq!(east.round_wind, 1);
        assert_eq!(east.honba, 3);
    }

    #[test]
    fn initialize_next_round_east_mode_ends_when_south_starts_with_30000_leader() {
        let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[0].score = 30_000;
        state.players[1].score = 25_000;
        state.players[2].score = 24_000;
        state.players[3].score = 21_000;
        state.oya = 3;
        state.honba = 1;
        state.round_wind = 0;

        state._initialize_next_round(false, true);

        assert!(state.is_done);
        assert_eq!(state.oya, 3);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 0);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_game\"")));
        assert!(!state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    }

    #[test]
    fn initialize_next_round_half_mode_stays_alive_in_west_before_limit_even_with_30000_leader() {
        let mut state = GameState::new(2, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[0].score = 31_000;
        state.players[1].score = 24_000;
        state.players[2].score = 23_000;
        state.players[3].score = 22_000;
        state.oya = 2;
        state.honba = 2;
        state.round_wind = 1;

        state._initialize_next_round(false, false);

        assert!(!state.is_done);
        assert_eq!(state.oya, 3);
        assert_eq!(state.honba, 0);
        assert_eq!(state.round_wind, 1);
        assert_eq!(state.current_player, 3);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[3]);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_kyoku\"")));
        assert!(!state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_game\"")));
    }

    #[test]
    fn initialize_next_round_half_mode_ends_after_west_wrap_without_30000_leader() {
        let mut state = GameState::new(2, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[0].score = 29_000;
        state.players[1].score = 28_000;
        state.players[2].score = 27_000;
        state.players[3].score = 26_000;
        state.oya = 3;
        state.honba = 0;
        state.round_wind = 2;

        state._initialize_next_round(false, false);

        assert!(state.is_done);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_game\"")));
        assert!(!state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    }

    #[test]
    fn trigger_ryukyoku_handles_illegal_action_penalties_and_nagashi() {
        let mut state = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
        state.players.iter_mut().for_each(|p| {
            p.score = 25_000;
            p.score_delta = 0;
        });
        state.oya = 0;
        state._trigger_ryukyoku("Error: Illegal Action by Player 1");
        assert_eq!(state.players[0].score_delta, 4000);
        assert_eq!(state.players[1].score_delta, -8000);
        assert_eq!(state.players[2].score_delta, 2000);
        assert_eq!(state.players[3].score_delta, 2000);

        let mut nagashi = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
        nagashi.players.iter_mut().for_each(|p| {
            p.score = 25_000;
            p.score_delta = 0;
            p.nagashi_eligible = false;
        });
        nagashi.players[0].nagashi_eligible = true;
        nagashi.oya = 0;
        nagashi._trigger_ryukyoku("exhaustive_draw");
        assert!(nagashi.players[0].score > 25_000);
        assert!(nagashi.players[1].score < 25_000);
        assert_eq!(nagashi.honba, 1);
    }

    #[test]
    fn trigger_ryukyoku_illegal_action_penalizes_dealer_offender_and_keeps_renchan() {
        let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players.iter_mut().for_each(|player| {
            player.score = 25_000;
            player.score_delta = 0;
        });
        state.oya = 0;

        state._trigger_ryukyoku("Error: Illegal Action by Player 0");

        assert_eq!(state.players[0].score, 13_000);
        assert_eq!(state.players[0].score_delta, -12_000);
        assert_eq!(state.players[1].score, 29_000);
        assert_eq!(state.players[1].score_delta, 4_000);
        assert_eq!(state.players[2].score, 29_000);
        assert_eq!(state.players[2].score_delta, 4_000);
        assert_eq!(state.players[3].score, 29_000);
        assert_eq!(state.players[3].score_delta, 4_000);
        assert_eq!(state.oya, 0);
        assert_eq!(state.honba, 1);

        let ryukyoku_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"ryukyoku\""))
            .expect("ryukyoku event should be logged");
        let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
        assert_eq!(
            ryukyoku_event["deltas"],
            serde_json::json!([-12000, 4000, 4000, 4000])
        );
        assert_eq!(
            ryukyoku_event["reason"],
            serde_json::json!("Error: Illegal Action by Player 0")
        );
    }

    #[test]
    fn push_mjai_event_masks_start_kyoku_hands_and_other_players_draws() {
        let mut state = fresh_state();
        let mut start = serde_json::Map::new();
        start.insert("type".to_string(), Value::String("start_kyoku".to_string()));
        start.insert(
            "tehais".to_string(),
            serde_json::json!([["1m", "2m"], ["3m", "4m"], ["5m", "6m"], ["7m", "8m"]]),
        );
        state._push_mjai_event(Value::Object(start));
        let masked_start = state.mjai_log_per_player[1]
            .last()
            .expect("masked start_kyoku event should exist");
        assert!(masked_start.contains("?"));
        assert!(!masked_start.contains("1m"));
        assert!(!masked_start.contains("2m"));

        let mut tsumo = serde_json::Map::new();
        tsumo.insert("type".to_string(), Value::String("tsumo".to_string()));
        tsumo.insert("actor".to_string(), Value::Number(0.into()));
        tsumo.insert("pai".to_string(), Value::String("5pr".to_string()));
        state._push_mjai_event(Value::Object(tsumo));
        let masked_tsumo = state.mjai_log_per_player[1]
            .last()
            .expect("masked tsumo event should exist");
        let actor_tsumo = state.mjai_log_per_player[0]
            .last()
            .expect("actor tsumo event should exist");
        assert!(masked_tsumo.contains("\"pai\":\"?\""));
        assert!(actor_tsumo.contains("5pr"));
    }

    #[test]
    fn push_mjai_event_start_kyoku_masks_missing_tehai_lengths_to_default_13_in_4p() {
        let mut state = fresh_state();
        state._push_mjai_event(serde_json::json!({
            "type": "start_kyoku",
            "tehais": [
                ["1m", "2m"],
                serde_json::Value::Null,
                ["5m"],
                ["7m", "8m", "9m", "1p"]
            ]
        }));

        let p0: Value = serde_json::from_str(state.mjai_log_per_player[0].last().unwrap()).unwrap();
        let p1: Value = serde_json::from_str(state.mjai_log_per_player[1].last().unwrap()).unwrap();
        let p2: Value = serde_json::from_str(state.mjai_log_per_player[2].last().unwrap()).unwrap();
        let p3: Value = serde_json::from_str(state.mjai_log_per_player[3].last().unwrap()).unwrap();

        assert_eq!(p0["tehais"][0], serde_json::json!(["1m", "2m"]));
        assert_eq!(p0["tehais"][1].as_array().unwrap().len(), 13);
        assert_eq!(p0["tehais"][2], serde_json::json!(["?"]));
        assert_eq!(p0["tehais"][3], serde_json::json!(["?", "?", "?", "?"]));

        assert_eq!(p1["tehais"][0], serde_json::json!(["?", "?"]));
        assert_eq!(p1["tehais"][1], serde_json::Value::Null);

        assert_eq!(p2["tehais"][2], serde_json::json!(["5m"]));
        assert_eq!(p3["tehais"][3], serde_json::json!(["7m", "8m", "9m", "1p"]));
    }

    #[test]
    fn push_mjai_event_keeps_tsumo_tile_visible_when_actor_is_unknown_in_4p() {
        let mut state = fresh_state();
        state._push_mjai_event(serde_json::json!({
            "type": "tsumo",
            "pai": "5pr"
        }));

        for pid in 0..NP {
            let event = state.mjai_log_per_player[pid]
                .last()
                .expect("tsumo event should be logged for every player");
            assert!(event.contains("5pr"));
            assert!(!event.contains("\"pai\":\"?\""));
        }
    }

    #[test]
    fn push_mjai_event_start_kyoku_keeps_known_tehai_lengths_for_actor_with_empty_others() {
        let mut state = fresh_state();
        state._push_mjai_event(serde_json::json!({
            "type": "start_kyoku",
            "tehais": [
                ["1m"],
                [],
                ["5m", "6m", "7m"],
                []
            ]
        }));

        let p2: Value = serde_json::from_str(state.mjai_log_per_player[2].last().unwrap()).unwrap();

        assert_eq!(p2["tehais"][0], serde_json::json!(["?"]));
        assert_eq!(p2["tehais"][1], serde_json::json!([]));
        assert_eq!(p2["tehais"][2], serde_json::json!(["5m", "6m", "7m"]));
        assert_eq!(p2["tehais"][3], serde_json::json!([]));
    }

    #[test]
    fn process_end_game_idempotent_when_logging_disabled() {
        let mut state = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());

        state._process_end_game();
        state._process_end_game();

        assert!(state.is_done);
        assert!(state.mjai_log.is_empty());
        assert!(state
            .mjai_log_per_player
            .iter()
            .all(|events| events.is_empty()));
    }

    #[test]
    fn initialize_next_round_returns_immediately_when_game_is_already_done() {
        let mut state = fresh_state();
        state.is_done = true;
        state.oya = 3;
        state.honba = 2;
        state.round_wind = 1;

        state._initialize_next_round(false, false);

        assert!(state.is_done);
        assert_eq!(state.oya, 3);
        assert_eq!(state.honba, 2);
        assert_eq!(state.round_wind, 1);
    }

    #[test]
    fn deal_next_exhaustive_draw_without_nagashi_keeps_scores_even_and_renchan_depends_on_oya_tenpai(
    ) {
        let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 2;
        state.current_player = 1;
        state.honba = 0;
        state.round_wind = 0;
        state.wall.tile_count = 14;
        state.wall.draw_cursor = 0;
        state.players.iter_mut().for_each(|player| {
            player.nagashi_eligible = false;
            player.score = 25_000;
            player.score_delta = 0;
            player.hand = [0; 14];
            player.hand_len = 0;
            player.melds = [Meld::default(); 4];
            player.meld_count = 0;
        });

        state.players[0].hand[..13]
            .copy_from_slice(&[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112]);
        state.players[0].hand_len = 13;
        state.players[1].hand[..13]
            .copy_from_slice(&[1, 5, 9, 37, 41, 45, 73, 77, 81, 113, 117, 121, 125]);
        state.players[1].hand_len = 13;
        state.players[2].hand[..13]
            .copy_from_slice(&[0, 1, 2, 36, 37, 38, 72, 73, 74, 108, 109, 110, 112]);
        state.players[2].hand_len = 13;
        state.players[3].hand[..13]
            .copy_from_slice(&[2, 6, 10, 38, 42, 46, 74, 78, 82, 114, 118, 122, 126]);
        state.players[3].hand_len = 13;

        state._deal_next();

        assert_eq!(state.players[0].score, 25_000);
        assert_eq!(state.players[1].score, 25_000);
        assert_eq!(state.players[2].score, 25_000);
        assert_eq!(state.players[3].score, 25_000);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"ryukyoku\"")));
        assert_eq!(state.oya, 3);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 0);
    }

    #[test]
    fn process_end_game_is_idempotent_for_done_flag_and_logging_shape() {
        let mut logged = fresh_state();
        logged.mjai_log.clear();
        logged.mjai_log_per_player = Default::default();

        logged._process_end_game();
        logged._process_end_game();

        assert!(logged.is_done);
        assert_eq!(logged.mjai_log.len(), 2);
        assert!(logged
            .mjai_log
            .iter()
            .all(|event| event.contains("\"type\":\"end_game\"")));

        let mut silent = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
        silent._process_end_game();
        silent._process_end_game();
        assert!(silent.is_done);
        assert!(silent.mjai_log.is_empty());
    }

    #[test]
    fn check_abortive_draw_triggers_suufon_renda_and_rejects_mixed_winds() {
        let mut triggered = fresh_state();
        for (idx, player) in triggered.players.iter_mut().enumerate() {
            player.discards[0] = [108, 109, 110, 111][idx];
            player.discard_len = 1;
            player.meld_count = 0;
        }

        assert!(triggered.check_abortive_draw());
        assert!(triggered
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"ryukyoku\"")));

        let mut mixed = fresh_state();
        mixed.players[0].discards[0] = 108;
        mixed.players[1].discards[0] = 112;
        mixed.players[2].discards[0] = 109;
        mixed.players[3].discards[0] = 110;
        for player in &mut mixed.players {
            player.discard_len = 1;
            player.meld_count = 0;
        }
        assert!(!mixed.check_abortive_draw());
    }

    #[test]
    fn check_abortive_draw_triggers_suucha_riichi_when_all_players_declared() {
        let mut state = fresh_state();
        state.players.iter_mut().for_each(|player| {
            player.riichi_declared = true;
            player.score_delta = 0;
        });

        assert!(state.check_abortive_draw());
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"reason\":\"suucha_riichi\"")));
    }

    #[test]
    fn check_abortive_draw_triggers_suukansansen_only_with_four_kans_by_multiple_players() {
        let mut triggered = fresh_state();
        triggered.players[0].push_meld(Meld::new(MeldType::Ankan, &[0, 1, 2, 3], false, -1, None));
        triggered.players[0].push_meld(Meld::new(MeldType::Kakan, &[4, 5, 6, 7], true, -1, None));
        triggered.players[1].push_meld(Meld::new(
            MeldType::Daiminkan,
            &[8, 9, 10, 11],
            true,
            0,
            Some(8),
        ));
        triggered.players[2].push_meld(Meld::new(
            MeldType::Ankan,
            &[12, 13, 14, 15],
            false,
            -1,
            None,
        ));

        assert!(triggered.check_abortive_draw());
        assert!(triggered
            .mjai_log
            .iter()
            .any(|event| event.contains("\"reason\":\"suukansansen\"")));

        let mut same_owner = fresh_state();
        for meld_tiles in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] {
            same_owner.players[0].push_meld(Meld::new(
                MeldType::Ankan,
                &meld_tiles,
                false,
                -1,
                None,
            ));
        }

        assert!(!same_owner.check_abortive_draw());

        let mut not_enough = fresh_state();
        not_enough.players[0].push_meld(Meld::new(MeldType::Ankan, &[0, 1, 2, 3], false, -1, None));
        not_enough.players[1].push_meld(Meld::new(MeldType::Kakan, &[4, 5, 6, 7], true, -1, None));
        not_enough.players[2].push_meld(Meld::new(
            MeldType::Daiminkan,
            &[8, 9, 10, 11],
            true,
            0,
            Some(8),
        ));

        assert!(!not_enough.check_abortive_draw());
    }

    #[test]
    fn handle_wait_response_triple_ron_uses_sanchaho_abortive_draw_when_enabled() {
        let mut rule = GameRule::default_tenhou();
        rule.sanchaho_is_draw = true;
        let mut state = GameState::new(1, false, Some(7), 0, rule);
        let win_tile = 48;
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.phase = Phase::WaitResponse;
        state.current_player = 0;
        state.last_discard = Some((0, win_tile));
        state.active_players = [1, 2, 3, 0];
        state.active_player_count = 3;
        for pid in 1..4usize {
            state.current_claim_counts[pid] = 1;
            state.current_claims[pid][0] =
                Action::new(ActionType::Ron, Some(win_tile), &[], Some(pid as u8));
        }

        state._handle_wait_response(&[
            None,
            Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
            Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(2))),
            Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(3))),
        ]);

        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"reason\":\"sanchaho\"")));
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_claim_counts, [0, 0, 0, 0]);
        assert!(state.is_first_turn);
    }

    #[test]
    fn initialize_next_round_ends_game_immediately_when_any_player_is_bankrupt() {
        let mut state = fresh_state();
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        let oya_before = state.oya;
        let honba_before = state.honba;
        let round_wind_before = state.round_wind;
        state.players[2].score = -1;

        state._initialize_next_round(false, false);

        assert!(state.is_done);
        assert_eq!(state.oya, oya_before);
        assert_eq!(state.honba, honba_before);
        assert_eq!(state.round_wind, round_wind_before);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_game\"")));
    }

    #[test]
    fn accept_riichi_is_noop_without_pending_player_and_logs_when_present() {
        let mut silent = fresh_state();
        let score_before = silent.players[0].score;
        silent._accept_riichi();
        assert_eq!(silent.players[0].score, score_before);
        assert_eq!(silent.riichi_sticks, 0);

        let mut logged = fresh_state();
        logged.mjai_log.clear();
        logged.mjai_log_per_player = Default::default();
        logged.riichi_pending_acceptance = Some(2);
        logged._accept_riichi();
        assert_eq!(logged.players[2].score, 24_000);
        assert_eq!(logged.players[2].score_delta, -1000);
        assert_eq!(logged.riichi_sticks, 1);
        assert!(logged.players[2].riichi_declared);
        assert!(logged.players[2].ippatsu_cycle);
        assert!(logged.riichi_pending_acceptance.is_none());
        assert!(logged
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"reach_accepted\"")));
    }

    #[test]
    fn deal_next_draws_tile_from_back_and_clears_needs_tsumo() {
        let mut state = fresh_state();
        state.drawn_tile = None;
        state.needs_tsumo = true;
        state.current_player = 1;
        state.wall.tile_count = 40;
        let hand_len_before = state.players[1].hand_len;

        state._deal_next();

        assert!(state.drawn_tile.is_some());
        assert!(!state.needs_tsumo);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[1]);
        assert_eq!(state.players[1].hand_len, hand_len_before + 1);
        assert!(state.players[1]
            .hand_slice()
            .contains(&state.drawn_tile.expect("drawn tile should be recorded")));
    }

    #[test]
    fn resolve_kan_at_dead_wall_threshold_skips_rinshan_draw_and_keeps_pending_dora_count() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = [0, 0, 0, 0];
        state.active_player_count = 1;
        state.drawn_tile = None;
        state.is_rinshan_flag = false;
        state.wall.tile_count = 34;
        state.wall.draw_cursor = 20;
        state.wall.pending_kan_dora_count = 2;
        state.players[0].ippatsu_cycle = true;
        state.players[0].hand = [0; 14];
        state.players[0].hand_len = 0;
        for tile in [0u8, 1, 2, 3] {
            state.players[0].push_hand(tile);
        }

        state._resolve_kan(
            0,
            Action::new(ActionType::Ankan, Some(0), &[0, 1, 2, 3], Some(0)),
        );

        assert_eq!(state.players[0].meld_count, 1);
        assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
        assert_eq!(state.players[0].hand_len, 0);
        assert_eq!(state.wall.remaining(), 14);
        assert_eq!(state.wall.rinshan_draw_count, 0);
        assert_eq!(state.wall.pending_kan_dora_count, 2);
        assert_eq!(state.drawn_tile, None);
        assert!(!state.is_rinshan_flag);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    }

    #[test]
    fn resolve_discard_reveals_pending_kan_dora_before_dahai_for_mortal_rules() {
        let mut state = fresh_state();
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.rule = GameRule::default_mortal();
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = [0, 0, 0, 0];
        state.active_player_count = 1;
        state.wall.pending_kan_dora_count = 1;
        state.wall.dora_indicator_count = 1;
        state.wall.tiles[6] = parsed_tile("2p");
        for pid in 1..4 {
            state.players[pid].hand = [0; 14];
            state.players[pid].hand_len = 0;
            state.players[pid].melds = [Meld::default(); 4];
            state.players[pid].meld_count = 0;
        }
        let discard_tile = state
            .drawn_tile
            .expect("fresh state should start with a drawn tile");

        state._resolve_discard(0, discard_tile, true);

        assert_eq!(state.wall.pending_kan_dora_count, 0);
        assert_eq!(
            state.wall.dora_indicator_slice(),
            &[state.wall.tiles[4], parsed_tile("2p")]
        );
        let dora_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"dora\""))
            .expect("dora reveal should be logged");
        let dahai_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"dahai\""))
            .expect("discard should be logged");
        assert!(dora_idx < dahai_idx);
    }

    #[test]
    fn resolve_discard_sets_riichi_side_effects_and_logs_dahai() {
        let mut state = fresh_state();
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.current_player = 0;
        let drawn = state
            .drawn_tile
            .expect("fresh state should start with a drawn tile");
        state.players[0].riichi_stage = true;
        state.players[0].nagashi_eligible = true;

        state._resolve_discard(0, drawn, false);

        assert_eq!(state.last_discard, Some((0, drawn)));
        assert!(state.players[0].riichi_declared);
        assert_eq!(state.last_tedashis[0], Some(drawn));
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"dahai\"")));
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"reach_accepted\"")));
    }

    #[test]
    fn initialize_round_applies_scores_and_logs_start_and_initial_tsumo() {
        let mut state = fresh_state();
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._initialize_round(2, 1, 3, 4, None, Some(vec![21_000, 22_000, 23_000, 24_000]));

        assert_eq!(state.oya, 2);
        assert_eq!(state.round_wind, 1);
        assert_eq!(state.honba, 3);
        assert_eq!(state.riichi_sticks, 4);
        assert_eq!(state.players[0].score, 21_000);
        assert_eq!(state.players[3].score, 24_000);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[2]);
        assert!(state.drawn_tile.is_some());
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"start_kyoku\"")));
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"tsumo\"")));
    }

    #[test]
    fn initialize_round_without_oya_draw_leaves_needs_tsumo_true() {
        let mut state = fresh_state();
        let wall = vec![0; 52];
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._initialize_round(0, 0, 0, 0, Some(wall), Some(vec![25_000; 4]));

        assert!(state.drawn_tile.is_none());
        assert!(state.needs_tsumo);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"start_kyoku\"")));
        assert!(!state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"tsumo\"")));
    }

    #[test]
    fn apply_mjai_event_and_log_action_wrappers_delegate_to_handlers() {
        let mut state = fresh_state();
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.apply_mjai_event(MjaiEvent::Reach { actor: 0 });
        assert!(state.players[0].riichi_stage);

        let mut replay = fresh_state();
        replay.mjai_log.clear();
        replay.mjai_log_per_player = Default::default();
        replay.last_tedashis = [None; NP];
        let action = LogAction::DiscardTile {
            seat: 0,
            tile: 16,
            is_liqi: false,
            is_wliqi: false,
            doras: None,
        };
        replay.apply_log_action(&action);
        assert_eq!(replay.last_discard.map(|(_, t)| t / 4), Some(4));
    }

    #[test]
    fn replay_matcher_rejects_mismatched_action_kinds_and_tiles() {
        let discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
        let riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
        assert!(!GameState::replay_action_matches_legal(&discard, &riichi));

        let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
        let replay_with_wrong_tile = Action::new(ActionType::Ron, Some(20), &[], Some(0));
        assert!(!GameState::replay_action_matches_legal(
            &legal_ron,
            &replay_with_wrong_tile
        ));
    }

    #[test]
    fn replay_matcher_accepts_kakan_context_actions_and_red_five_rules() {
        let legal_kakan = Action::new(ActionType::Kakan, Some(16), &[17, 18, 19], Some(0));
        let replay_kakan = Action::new(ActionType::Kakan, Some(16), &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_kakan,
            &replay_kakan
        ));

        let legal_ankan = Action::new(ActionType::Ankan, Some(20), &[20, 21, 22, 23], Some(0));
        let replay_ankan = Action::new(ActionType::Ankan, Some(23), &[20, 21, 22, 23], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_ankan,
            &replay_ankan
        ));

        let legal_riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
        let replay_riichi = Action::new(ActionType::Riichi, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_riichi,
            &replay_riichi
        ));

        assert!(GameState::replay_tile_matches_mjai_semantics(16, 16));
        assert!(!GameState::replay_tile_matches_mjai_semantics(16, 17));
        assert!(GameState::replay_tile_matches_mjai_semantics(0, 3));
    }

    #[test]
    fn replay_observation_accepts_sparse_kakan_action_when_drawn_tile_matches() {
        let rule = GameRule::default_mjsoul();
        let mut state = GameState::new(0, false, Some(1), 0, rule);
        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            dora_marker: "1p".to_string(),
            scores: vec![25000, 25000, 25000, 25000],
            tehais: vec![
                vec![
                    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
                    "5p",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                vec![
                    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                vec![
                    "1p", "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                vec![
                    "6m", "6m", "6m", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "P",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ],
        });
        state.apply_mjai_event(MjaiEvent::Pon {
            actor: 3,
            target: 0,
            pai: "6m".to_string(),
            consumed: vec!["6m".to_string(), "6m".to_string()],
        });
        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 3,
            pai: "6m".to_string(),
        });

        let replay_kakan = Action::new(ActionType::Kakan, Some(20), &[], Some(3));
        let obs = state
            .get_observation_for_replay(
                3,
                &replay_kakan,
                r#"{"actor":3,"pai":"6m","type":"kakan"}"#,
            )
            .expect("sparse kakan replay action should be accepted");

        assert!(obs
            .legal_actions_ref()
            .iter()
            .any(|action| action.action_type == ActionType::Kakan));
    }

    #[test]
    fn replay_matcher_accepts_context_implied_tile_less_actions_and_consume_matched_kan_upgrades() {
        let legal_tsumo = Action::new(ActionType::Tsumo, Some(48), &[], Some(0));
        let replay_tsumo = Action::new(ActionType::Tsumo, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_tsumo,
            &replay_tsumo
        ));

        let legal_kyushu = Action::new(ActionType::KyushuKyuhai, Some(0), &[], Some(0));
        let replay_kyushu = Action::new(ActionType::KyushuKyuhai, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_kyushu,
            &replay_kyushu
        ));

        let legal_kakan = Action::new(ActionType::Kakan, Some(16), &[17, 18, 19], Some(0));
        let replay_kakan = Action::new(ActionType::Kakan, Some(64), &[17, 18, 19], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_kakan,
            &replay_kakan
        ));

        let legal_discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
        let replay_discard = Action::new(ActionType::Discard, None, &[], Some(0));
        assert!(!GameState::replay_action_matches_legal(
            &legal_discard,
            &replay_discard
        ));
    }

    #[test]
    fn sorted_hand_helpers_cover_front_middle_end_and_copy_clamp_edges() {
        let mut hand = [0u8; 14];
        hand[..3].copy_from_slice(&[8, 16, 24]);
        let mut len = 3;

        sorted_insert_arr(&mut hand, &mut len, 0);
        sorted_insert_arr(&mut hand, &mut len, 20);
        sorted_insert_arr(&mut hand, &mut len, 32);

        assert_eq!(len, 6);
        assert_eq!(&hand[..len as usize], &[0, 8, 16, 20, 24, 32]);

        let (buf, copied_len) = copy_and_sorted_insert(&[12, 20, 28, 36, 44], 24);
        assert_eq!(copied_len, 5);
        assert_eq!(&buf[..copied_len], &[12, 20, 24, 28, 36]);
    }

    #[test]
    fn claim_and_active_player_helpers_round_trip_state() {
        let mut state = fresh_state();
        let ron = Action::new(ActionType::Ron, Some(48), &[], Some(1));
        let pass = Action::new(ActionType::Pass, None, &[], Some(1));

        state.clear_active_players();
        assert!(state.active_player_slice().is_empty());

        state.set_single_active_player(2);
        assert_eq!(state.active_player_slice(), &[2]);

        state.set_active_players_from_slice(&[1, 3]);
        assert_eq!(state.active_player_slice(), &[1, 3]);

        state.push_claim(1, ron);
        state.push_claim(1, pass);
        assert_eq!(state.claims_slice(1), &[ron, pass]);

        state.push_claim(2, ron);
        assert_eq!(state.claims_slice(2), &[ron]);

        state.clear_claims();
        assert!(state.claims_slice(1).is_empty());
        assert!(state.claims_slice(2).is_empty());
    }

    #[test]
    fn replay_matcher_rejects_tileless_non_contextual_actions_but_allows_ron() {
        let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
        let replay_ron = Action::new(ActionType::Ron, None, &[], Some(0));
        assert!(GameState::replay_action_matches_legal(
            &legal_ron,
            &replay_ron
        ));

        let legal_chi = Action::new(ActionType::Chi, Some(16), &[12, 20], Some(0));
        let replay_chi = Action::new(ActionType::Chi, None, &[12, 20], Some(0));
        assert!(!GameState::replay_action_matches_legal(
            &legal_chi,
            &replay_chi
        ));
    }

    #[test]
    fn replay_observation_allows_pass_for_active_response_player_and_restores_state() {
        let mut state = fresh_state();
        state.phase = Phase::WaitResponse;
        state.active_players = [1, 0, 0, 0];
        state.active_player_count = 1;
        state.current_claim_counts[1] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Ron, Some(48), &[], Some(1));

        let obs = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Pass, None, &[], Some(1)),
                "{\"type\":\"none\"}",
            )
            .expect("pass should be exposed as legal during response replay");

        assert!(obs
            ._legal_actions
            .iter()
            .any(|action| action.action_type == ActionType::Pass));
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_players, [1, 0, 0, 0]);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.current_claim_counts[1], 1);
    }

    #[test]
    fn replay_observation_exposes_call_action_for_response_player_and_restores_claims() {
        let mut state = fresh_state();
        state.phase = Phase::WaitResponse;
        state.active_players = [2, 0, 0, 0];
        state.active_player_count = 1;
        state.current_claim_counts[2] = 1;
        state.current_claims[2][0] = Action::new(ActionType::Pon, Some(48), &[49, 50], Some(2));

        let obs = state
            .get_observation_for_replay(
                2,
                &Action::new(ActionType::Pon, Some(48), &[49, 50], Some(2)),
                "{\"type\":\"pon\"}",
            )
            .expect("pon should be exposed as legal during response replay");

        assert!(obs
            ._legal_actions
            .iter()
            .any(|action| action.action_type == ActionType::Pon));
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_players, [2, 0, 0, 0]);
        assert_eq!(state.current_claim_counts[2], 1);
    }

    #[test]
    fn replay_observation_restores_riichi_after_failed_discard_retry() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        let drawn_tile = state
            .drawn_tile
            .expect("fresh state should start with a drawn tile");
        let invalid_tile = (0..136u8)
            .find(|&tile| tile != drawn_tile && !state.players[0].hand_slice().contains(&tile))
            .expect("there should be a tile outside player 0's hand");
        state.players[0].riichi_declared = true;

        let err = state
            .get_observation_for_replay(
                0,
                &Action::new(ActionType::Discard, Some(invalid_tile), &[], Some(0)),
                "{\"type\":\"dahai\"}",
            )
            .expect_err("illegal replay discard should stay illegal after riichi retry");

        assert!(matches!(err, RiichiError::InvalidState { .. }));
        assert!(state.players[0].riichi_declared);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
    }

    #[test]
    fn replay_observation_retries_discard_after_temporarily_clearing_riichi() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        let drawn_tile = state
            .drawn_tile
            .expect("fresh state should start with a drawn tile");
        let retry_tile = state.players[0]
            .hand_slice()
            .iter()
            .copied()
            .find(|&tile| tile != drawn_tile)
            .expect("hand should contain a non-drawn tile to discard");
        state.players[0].riichi_declared = true;

        let obs = state
            .get_observation_for_replay(
                0,
                &Action::new(ActionType::Discard, Some(retry_tile), &[], Some(0)),
                "{\"type\":\"dahai\"}",
            )
            .expect("replay discard should succeed after riichi retry path");

        assert!(obs
            .legal_actions_method()
            .iter()
            .any(|action| action.action_type == ActionType::Discard
                && action.tile == Some(retry_tile)));
        assert!(!state.players[0].riichi_declared);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
    }

    #[test]
    fn replay_observation_rejects_wait_act_discard_that_is_only_tile_semantic_match_in_hand() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        let replay_tile = state.players[0]
            .hand_slice()
            .iter()
            .copied()
            .find(|&tile| {
                !state.players[0]
                    .forbidden_slice()
                    .iter()
                    .any(|&forbidden| forbidden / 4 == tile / 4)
            })
            .expect("fresh state should have at least one discardable hand tile");
        state.players[0].push_forbidden(replay_tile);

        let baseline_obs = state.get_observation(0);
        assert!(baseline_obs
            ._legal_actions
            .iter()
            .filter(|action| action.action_type == ActionType::Discard)
            .all(|action| action.tile.is_some_and(|tile| tile / 4 != replay_tile / 4)));

        let err = state
            .get_observation_for_replay(
                0,
                &Action::new(ActionType::Discard, Some(replay_tile), &[], Some(0)),
                "{\"type\":\"dahai\"}",
            )
            .expect_err(
                "replay discard should stay illegal when only a hand tile semantically matches",
            );

        assert!(matches!(err, RiichiError::InvalidState { .. }));
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.players[0]
            .forbidden_slice()
            .iter()
            .any(|&forbidden| forbidden / 4 == replay_tile / 4));
    }

    #[test]
    fn replay_observation_rejects_wait_act_discard_for_non_current_player_even_with_drawn_tile() {
        let mut state = fresh_state();
        let replay_tile = state.players[1]
            .hand_slice()
            .iter()
            .copied()
            .find(|&tile| {
                state.players[1]
                    .forbidden_slice()
                    .iter()
                    .all(|&forbidden| forbidden / 4 != tile / 4)
            })
            .expect("player 1 should have at least one non-forbidden tile in hand");

        let err = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Discard, Some(replay_tile), &[], Some(1)),
                "{\"type\":\"dahai\"}",
            )
            .expect_err("non-current player discard should stay illegal during wait-act replay");

        assert!(matches!(err, RiichiError::InvalidState { .. }));
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.active_player_slice(), &[0]);
    }

    #[test]
    fn wait_response_marks_missed_ron_and_riichi_when_player_passes_on_win() {
        let mut state = fresh_state();
        state.phase = Phase::WaitResponse;
        state.active_players = [1, 0, 0, 0];
        state.active_player_count = 1;
        state.current_claim_counts[1] = 2;
        state.current_claims[1][0] = Action::new(ActionType::Ron, Some(48), &[], Some(1));
        state.current_claims[1][1] = Action::new(ActionType::Pass, None, &[], Some(1));
        state.players[1].riichi_declared = true;

        state._handle_wait_response(&[
            None,
            Some(Action::new(ActionType::Pass, None, &[], Some(1))),
            None,
            None,
        ]);

        assert!(state.players[1].missed_agari_doujun);
        assert!(state.players[1].missed_agari_riichi);
    }

    #[test]
    fn wait_response_resolves_pending_kakan_after_all_players_pass() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        state.phase = Phase::WaitResponse;
        state.current_player = 0;
        state.last_discard = Some((0, 48));
        state.active_players = [1, 2, 3, 0];
        state.active_player_count = 3;
        state.current_claim_counts[1] = 1;
        state.current_claim_counts[2] = 1;
        state.current_claim_counts[3] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
        state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
        state.current_claims[3][0] = Action::new(ActionType::Pass, None, &[], Some(3));
        state.pending_kan = Some((
            0,
            Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0)),
        ));
        if state.players[0].hand_len > 0 {
            state.players[0].hand_len -= 1;
        }
        state.drawn_tile = None;
        let rinshan_before = state.wall.rinshan_draw_count;
        let pending_before = state.wall.pending_kan_dora_count;

        state._handle_wait_response(&[
            None,
            Some(Action::new(ActionType::Pass, None, &[], Some(1))),
            Some(Action::new(ActionType::Pass, None, &[], Some(2))),
            Some(Action::new(ActionType::Pass, None, &[], Some(3))),
        ]);

        assert!(state.pending_kan.is_none());
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.drawn_tile.is_some());
        assert!(state.is_rinshan_flag);
        assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
        assert_eq!(state.wall.pending_kan_dora_count, pending_before + 1);
        assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    }

    #[test]
    fn wait_response_all_pass_accepts_riichi_and_advances_turn_out_of_first_cycle() {
        let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
        state.phase = Phase::WaitResponse;
        state.current_player = 3;
        state.turn_count = NP as u32 - 1;
        state.is_first_turn = true;
        state.riichi_pending_acceptance = Some(2);
        if state.players[0].hand_len > 0 {
            state.players[0].hand_len -= 1;
        }
        state.drawn_tile = None;

        state._handle_wait_response(&[None, None, None, None]);

        assert!(state.riichi_pending_acceptance.is_none());
        assert_eq!(state.players[2].score, 24_000);
        assert!(state.players[2].riichi_declared);
        assert!(state.players[2].ippatsu_cycle);
        assert_eq!(state.riichi_sticks, 1);
        assert_eq!(state.turn_count, NP as u32);
        assert!(!state.is_first_turn);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.drawn_tile.is_some());
    }

    #[test]
    fn step_array_initializes_pending_round_before_processing_actions() {
        let mut state = fresh_state();
        state.oya = 1;
        state.honba = 2;
        state.round_wind = 0;
        state.phase = Phase::WaitResponse;
        state.needs_initialize_next_round = true;
        state.pending_oya_won = false;
        state.pending_is_draw = false;
        state.last_discard = Some((0, 44));
        state.last_error = None;

        let actions = [
            None,
            None,
            Some(Action::new(ActionType::Discard, Some(255), &[], Some(2))),
            None,
        ];

        state.step_array(&actions);

        assert!(!state.needs_initialize_next_round);
        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 0);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.active_player_slice(), &[2]);
        assert_eq!(state.last_discard, None);
        assert!(state.last_error.is_none());
    }

    #[test]
    fn step_rejects_tileless_discard_and_records_illegal_action_error() {
        let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players.iter_mut().for_each(|player| {
            player.score = 25_000;
            player.score_delta = 0;
        });

        let mut actions = std::collections::HashMap::new();
        actions.insert(
            state.current_player,
            Action::new(ActionType::Discard, None, &[], Some(state.current_player)),
        );

        state.step(&actions);

        assert_eq!(
            state.last_error.as_deref(),
            Some("Error: Illegal Action by Player 0")
        );
        assert_eq!(state.players[0].score, 13_000);
        assert_eq!(state.players[0].score_delta, -12_000);
        assert_eq!(state.players[1].score, 29_000);
        assert_eq!(state.players[1].score_delta, 4_000);
        assert_eq!(state.players[2].score, 29_000);
        assert_eq!(state.players[2].score_delta, 4_000);
        assert_eq!(state.players[3].score, 29_000);
        assert_eq!(state.players[3].score_delta, 4_000);
        assert_eq!(state.oya, 0);
        assert_eq!(state.honba, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.drawn_tile.is_some());
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"ryukyoku\"")));
    }

    #[test]
    fn step_array_unchecked_initializes_pending_round_before_processing_actions() {
        let mut state = fresh_state();
        state.oya = 1;
        state.honba = 2;
        state.round_wind = 0;
        state.phase = Phase::WaitResponse;
        state.needs_initialize_next_round = true;
        state.pending_oya_won = false;
        state.pending_is_draw = false;
        state.last_discard = Some((0, 44));
        state.last_error = None;

        let actions = [
            None,
            None,
            Some(Action::new(ActionType::Riichi, None, &[], Some(2))),
            None,
        ];

        state.step_array_unchecked(&actions);

        assert!(!state.needs_initialize_next_round);
        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 0);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.active_player_slice(), &[2]);
        assert_eq!(state.last_discard, None);
        assert!(state.last_error.is_none());
        assert!(!state.players[2].riichi_stage);
    }

    #[test]
    fn wait_response_prioritizes_pon_over_existing_chi_claim() {
        let mut state = fresh_state();
        state.phase = Phase::WaitResponse;
        state.last_discard = Some((0, 12));
        state.active_players = [1, 2, 0, 0];
        state.active_player_count = 2;
        state.current_claim_counts[1] = 1;
        state.current_claim_counts[2] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Chi, Some(12), &[8, 16], Some(1));
        state.current_claims[2][0] = Action::new(ActionType::Pon, Some(12), &[13, 14], Some(2));
        state.players[1].hand[..2].copy_from_slice(&[8, 16]);
        state.players[1].hand_len = 2;
        state.players[2].hand[..2].copy_from_slice(&[13, 14]);
        state.players[2].hand_len = 2;

        state._handle_wait_response(&[
            None,
            Some(Action::new(ActionType::Chi, Some(12), &[8, 16], Some(1))),
            Some(Action::new(ActionType::Pon, Some(12), &[13, 14], Some(2))),
            None,
        ]);

        assert_eq!(state.current_player, 2);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[2]);
        assert_eq!(state.players[2].meld_count, 1);
        assert_eq!(state.players[2].melds[0].meld_type, MeldType::Pon);
        assert_eq!(state.players[1].meld_count, 0);
    }

    #[test]
    fn initialize_round_clears_pending_round_and_kan_riichi_transients() {
        let mut state = fresh_state();
        state.pending_kan = Some((
            0,
            Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0)),
        ));
        state.is_rinshan_flag = true;
        state.wall.rinshan_draw_count = 2;
        state.wall.pending_kan_dora_count = 1;
        state.riichi_pending_acceptance = Some(1);
        state.needs_initialize_next_round = true;
        state.pending_oya_won = true;
        state.pending_is_draw = true;
        state.last_discard = Some((0, 16));
        state.win_results[0] = Some(WinResult::new(
            false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
        ));
        state.riichi_sutehais[0] = Some(16);
        state.last_tedashis[0] = Some(16);
        state.players[0].ippatsu_cycle = true;

        state._initialize_round(0, 0, 0, 0, Some(vec![0; 52]), Some(vec![25_000; 4]));

        assert!(state.pending_kan.is_none());
        assert!(!state.is_rinshan_flag);
        assert_eq!(state.wall.rinshan_draw_count, 0);
        assert_eq!(state.wall.pending_kan_dora_count, 0);
        assert!(state.riichi_pending_acceptance.is_none());
        assert!(!state.needs_initialize_next_round);
        assert!(!state.pending_oya_won);
        assert!(!state.pending_is_draw);
        assert!(state.last_discard.is_none());
        assert!(state.win_results.iter().all(Option::is_none));
        assert_eq!(state.riichi_sutehais, [None; NP]);
        assert_eq!(state.last_tedashis, [None; NP]);
        assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    }

    #[test]
    fn replay_tile_semantics_reject_different_tile_classes_and_plain_red_mismatch() {
        assert!(!GameState::replay_tile_matches_mjai_semantics(0, 4));
        assert!(!GameState::replay_tile_matches_mjai_semantics(52, 53));
        assert!(GameState::replay_tile_matches_mjai_semantics(53, 54));
    }

    #[test]
    fn replay_tile_semantics_accepts_same_non_red_tile_copies() {
        assert!(GameState::replay_tile_matches_mjai_semantics(17, 18));
        assert!(GameState::replay_tile_matches_mjai_semantics(89, 91));
    }

    #[test]
    fn replay_matcher_rejects_consume_mismatches_for_non_kan_calls() {
        let legal_pon = Action::new(ActionType::Pon, Some(16), &[17, 18], Some(1));
        let replay_wrong_consume = Action::new(ActionType::Pon, Some(16), &[17, 19], Some(1));
        assert!(!GameState::replay_action_matches_legal(
            &legal_pon,
            &replay_wrong_consume
        ));

        let legal_chi = Action::new(ActionType::Chi, Some(16), &[12, 20], Some(1));
        let replay_empty_consume = Action::new(ActionType::Chi, Some(16), &[], Some(1));
        assert!(!GameState::replay_action_matches_legal(
            &legal_chi,
            &replay_empty_consume
        ));
    }

    #[test]
    fn replay_observation_error_restores_original_response_state_on_illegal_action() {
        let mut state = fresh_state();
        state.phase = Phase::WaitResponse;
        state.active_players = [0, 2, 3, 1];
        state.active_player_count = 2;
        state.current_claim_counts[2] = 1;
        state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

        let err = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Discard, Some(parsed_tile("1m")), &[], Some(1)),
                "{\"type\":\"dahai\"}",
            )
            .expect_err("inactive response player should reject unrelated replay discard");

        assert!(matches!(err, RiichiError::InvalidState { .. }));
        let message = match err {
            RiichiError::InvalidState { message } => message,
            other => panic!("expected InvalidState replay error, got {other:?}"),
        };
        assert!(message.contains("Replay desync"));
        assert!(message.contains("Log action: {\"type\":\"dahai\"}"));
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_players, [0, 2, 3, 1]);
        assert_eq!(state.active_player_count, 2);
        assert_eq!(state.current_claim_counts[2], 1);
        assert!(state.claims_slice(1).is_empty());
    }
}
