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
#[cfg_attr(feature = "python", pyo3::pyclass)]
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

        let tiles_match = legal.tile == replay.tile;
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
            let is_legal_retry = new_obs
                ._legal_actions
                .iter()
                .any(|a| a.action_type == ActionType::Discard && a.tile == env_action.tile);

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            } else {
                self.players[pid as usize].riichi_declared = original_riichi;
            }
        }

        self.phase = original_phase;
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

                    // Allow None from python for context-implied actions
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn replay_kan_matcher_rejects_different_tile_classes() {
        let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay = Action::new(ActionType::Ankan, Some(20), &[20, 20, 20, 20], Some(0));

        assert!(!GameState::replay_action_matches_legal(&legal, &replay));
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
