use std::collections::HashMap;

use serde_json::Value;

use crate::action::{Action, ActionType, Phase};
use crate::errors::{RiichiError, RiichiResult};
use crate::observation_3p::Observation3P;
use crate::parser::tid_to_mjai;
use crate::replay::Action as LogAction;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::types::{Conditions, Meld, MeldType, WinResult, Wind};

pub mod event_handler;
pub mod game_mode;
pub mod legal_actions;
pub mod player;
pub mod sanma;
pub mod wall;
use event_handler::GameState3PEventHandler;
use game_mode::GameSubMode3P;
use legal_actions::GameState3PLegalActions;
use player::PlayerState3P;
use wall::WallState3P;

const NP: usize = 3;

#[derive(Debug, Clone)]
pub struct GameState3P {
    pub wall: WallState3P,
    pub players: [PlayerState3P; NP],

    pub current_player: u8,
    pub turn_count: u32,
    pub is_done: bool,
    pub needs_tsumo: bool,
    pub needs_initialize_next_round: bool,
    pub pending_oya_won: bool,
    pub pending_is_draw: bool,

    pub riichi_sticks: u32,
    pub phase: Phase,
    pub active_players: [u8; 4],
    pub active_player_count: u8,
    pub last_discard: Option<(u8, u8)>,
    pub current_claims: [[Action; 54]; NP],
    pub current_claim_counts: [u8; NP],
    pub pending_kan: Option<(u8, Action)>,

    pub oya: u8,
    pub honba: u8,
    pub kyoku_idx: u8,
    pub round_wind: u8,

    pub is_rinshan_flag: bool,
    pub is_first_turn: bool,
    pub riichi_pending_acceptance: Option<u8>,
    pub drawn_tile: Option<u8>,

    pub win_results: [Option<WinResult>; NP],
    pub last_win_results: [Option<WinResult>; NP],

    pub mjai_log: Vec<String>,
    pub player_event_counts: [usize; NP],
    pub mjai_log_per_player: [Vec<String>; NP],

    pub sub_mode: GameSubMode3P,
    pub game_mode: u8,
    pub skip_mjai_logging: bool,
    pub seed: Option<u64>,
    pub rule: GameRule,
    pub last_error: Option<String>,
    pub is_after_kan: bool,

    pub riichi_sutehais: [Option<u8>; NP],
    pub last_tedashis: [Option<u8>; NP],
}

impl GameState3P {
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

    /// Sets claims for a player from a Vec.
    #[inline]
    fn set_claims_from_vec(&mut self, pid: usize, legals: &[Action]) {
        let count = legals.len().min(54);
        self.current_claims[pid][..count].copy_from_slice(&legals[..count]);
        self.current_claim_counts[pid] = count as u8;
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

    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        let sub_mode = GameSubMode3P::from_game_mode(game_mode);
        let players = [(); NP].map(|_| PlayerState3P::new(game_mode::starting_score()));

        let wall = WallState3P::new(seed);

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
            win_results: [None, None, None],
            last_win_results: [None, None, None],
            mjai_log: Vec::new(),
            player_event_counts: [0; NP],
            mjai_log_per_player: Default::default(),
            sub_mode,
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

        state._initialize_round(0, round_wind, 0, 0, None, None);
        state
    }

    pub fn reset(&mut self) {
        self.mjai_log = Vec::new();
        self.mjai_log_per_player = Default::default();
        self.player_event_counts = [0; NP];

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
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

    pub fn get_observation(&mut self, player_id: u8) -> Observation3P {
        let pid = player_id as usize;

        let masked_hands: [Vec<u8>; 3] = std::array::from_fn(|i| {
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

        let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
            self.players[pid].hand_slice(),
            self.players[pid].melds_slice(),
        );
        let waits = calc.get_waits_u8();
        let is_tenpai = !waits.is_empty();

        let melds: [Vec<Meld>; 3] = std::array::from_fn(|i| self.players[i].melds_slice().to_vec());
        let discards: [Vec<u8>; 3] =
            std::array::from_fn(|i| self.players[i].discards_slice().to_vec());
        let scores: [i32; 3] = std::array::from_fn(|i| self.players[i].score);
        let riichi_declared: [bool; 3] = std::array::from_fn(|i| self.players[i].riichi_declared);

        Observation3P::new(
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

    pub fn get_observation_for_replay(
        &mut self,
        pid: u8,
        env_action: &Action,
        log_action_str: &str,
    ) -> RiichiResult<Observation3P> {
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

    pub fn step(&mut self, actions: &HashMap<u8, Action>) {
        if self.is_done {
            return;
        }

        if self.needs_initialize_next_round {
            self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
            return;
        }

        // Validation
        for pid in 0..NP {
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
                        if act.consume_count == 0 && l.action_type == ActionType::Kakan {
                            return true;
                        }
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
        let mut action_arr: [Option<Action>; 3] = [None; 3];
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
    fn _execute_step_array(&mut self, actions: &[Option<Action>; 3]) {
        if self.phase == Phase::WaitAct {
            let pid = self.current_player;
            if let Some(act) = actions[pid as usize] {
                match act.action_type {
                    ActionType::Discard => self._handle_discard(pid, act),
                    ActionType::KyushuKyuhai => self._trigger_ryukyoku("kyushu_kyuhai"),
                    ActionType::Riichi => self._handle_riichi(pid, act),
                    ActionType::Ankan => self._handle_ankan(pid, act),
                    ActionType::Kakan => self._handle_kakan(pid, act),
                    ActionType::Tsumo => self._handle_tsumo(pid),
                    ActionType::Kita => self.handle_kita(pid, &act),
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
                self.players[pid as usize].hand_slice_mut().sort();
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
        if self.players[pid as usize].score >= 1000
            && self.wall.remaining() > 14
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
                self.riichi_sutehais[pid as usize] = Some(t);
                if !tsumogiri {
                    self.last_tedashis[pid as usize] = Some(t);
                }
                if let Some(idx) = self.players[pid as usize]
                    .hand_slice()
                    .iter()
                    .position(|&x| x == t)
                {
                    self.players[pid as usize].remove_hand(idx);
                    self.players[pid as usize].hand_slice_mut().sort();
                }
                self._resolve_discard(pid, t, tsumogiri);
            }
        }
    }

    /// Handles an ankan (closed kan) action during WaitAct phase.
    fn _handle_ankan(&mut self, pid: u8, act: Action) {
        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let mut chankan_ronners = [0u8; 2];
        let mut chankan_count = 0usize;
        if self.rule.allows_ron_on_ankan_for_kokushi_musou {
            for i in 0..NP as u8 {
                if i == pid {
                    continue;
                }
                let hand = self.players[i as usize].hand_slice();
                let melds = self.players[i as usize].melds_slice();
                let tile_class = tile / 4;
                let in_discards = self.players[i as usize]
                    .discards_slice()
                    .iter()
                    .any(|&d| d / 4 == tile_class);
                if in_discards {
                    continue;
                }
                let p_wind = (i + NP as u8 - self.oya) % NP as u8;
                let cond = Conditions {
                    tsumo: false,
                    riichi: self.players[i as usize].riichi_declared,
                    chankan: true,
                    player_wind: Wind::from(p_wind),
                    round_wind: Wind::from(self.round_wind),
                    is_sanma: true,
                    num_players: NP as u8,
                    ..Default::default()
                };
                let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand, melds);
                let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
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

    /// Handles a kakan (added kan) action during WaitAct phase.
    fn _handle_kakan(&mut self, pid: u8, act: Action) {
        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let p_idx = pid as usize;

        if let Some(idx) = self.players[p_idx]
            .hand_slice()
            .iter()
            .position(|&x| x == tile)
        {
            self.players[p_idx].remove_hand(idx);
        }
        for m in self.players[p_idx].melds_slice_mut().iter_mut() {
            if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                m.meld_type = MeldType::Kakan;
                m.push_tile(tile);
                m.tiles_slice_mut().sort();
                break;
            }
        }

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

        let tile = act
            .tile
            .or(act.consume_slice().first().copied())
            .unwrap_or(0);
        let mut chankan_ronners = [0u8; 2];
        let mut chankan_count = 0usize;
        for i in 0..NP as u8 {
            if i == pid {
                continue;
            }
            let hand = self.players[i as usize].hand_slice();
            let melds = self.players[i as usize].melds_slice();
            let p_wind = (i + NP as u8 - self.oya) % NP as u8;
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
                is_sanma: true,
                num_players: NP as u8,
                ..Default::default()
            };
            let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand, melds);

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

            let res = if !is_furiten {
                calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond))
            } else {
                WinResult::new(false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false)
            };

            if res.is_win && (res.yakuman || res.han >= 1) {
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
            self.last_discard = Some((pid, tile));
        } else {
            self._resolve_kan(pid, act);
        }
    }

    /// Handles a tsumo (self-draw win) action during WaitAct phase.
    fn _handle_tsumo(&mut self, pid: u8) {
        let hand = self.players[pid as usize].hand_slice();
        let melds = self.players[pid as usize].melds_slice();
        let p_wind = (pid + NP as u8 - self.oya) % NP as u8;
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
            kita_count: self.players[pid as usize].kita_count,
            is_sanma: true,
            num_players: NP as u8,
            ..Default::default()
        };
        let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand, melds);
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
                    NP as u8,
                );
                res.ron_agari = capped.pay_ron;
                res.tsumo_agari_oya = capped.pay_tsumo_oya;
                res.tsumo_agari_ko = capped.pay_tsumo_ko;
            }
        }

        if res.is_win {
            let mut deltas = [0i32; NP];
            let mut total_win = 0;

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
                // Per-yakuman tsumo total depends on player count.
                // Yakuman base = 8000; pay_oya = 16000, pay_ko = 8000.
                let np = NP as i32;
                let unit = if pid == self.oya {
                    (np - 1) * 16000 // oya tsumo: each ko pays 16000
                } else {
                    16000 + (np - 2) * 8000 // ko tsumo: oya pays 16000 + (np-2) ko pay 8000
                };
                let honba_total = self.honba as i32 * (np - 1) * 100;

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
                                for i in 0..NP as u8 {
                                    if i != pid {
                                        deltas[i as usize] -= share;
                                        total_win += share;
                                    }
                                }
                            } else {
                                let oya_pay = non_pao_yakuman_val * 16000;
                                let ko_pay = non_pao_yakuman_val * 8000;
                                for i in 0..NP as u8 {
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
            } else if pid == self.oya {
                for i in 0..NP as u8 {
                    if i != pid {
                        deltas[i as usize] = -(res.tsumo_agari_ko as i32);
                        total_win += res.tsumo_agari_ko as i32;
                    }
                }
            } else {
                for i in 0..NP as u8 {
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

            total_win += (self.riichi_sticks * 1000) as i32;
            self.riichi_sticks = 0;
            deltas[pid as usize] += total_win;

            self.players[pid as usize].score_delta = deltas[pid as usize];
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
            self.current_player = (self.current_player + 1) % NP as u8;
            self._deal_next();
        }
    }

    /// Handles the WaitResponse phase (claims, ron, pon, etc.).
    fn _handle_wait_response(&mut self, actions: &[Option<Action>; NP]) {
        // Check Missed WinResult
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

        let mut ron_claims = [0u8; 2];
        let mut ron_count = 0usize;
        let mut call_claim: Option<(u8, Action)> = None;

        for &pid in self.active_player_slice() {
            if let Some(act) = actions[pid as usize] {
                if act.action_type == ActionType::Ron {
                    ron_claims[ron_count] = pid;
                    ron_count += 1;
                } else if act.action_type == ActionType::Pon
                    || act.action_type == ActionType::Daiminkan
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
            let (target_pid, win_tile) = self.last_discard.unwrap_or((self.current_player, 0));
            ron_claims[..ron_count].sort_by_key(|&pid| (pid + NP as u8 - target_pid) % NP as u8);

            let winners = &ron_claims[..ron_count];

            let mut total_deltas = [0i32; NP];
            let mut oya_won = false;
            let mut deposit_taken = false;
            let mut honba_taken = false;

            for &w_pid in winners {
                let hand = self.players[w_pid as usize].hand_slice();
                let melds = self.players[w_pid as usize].melds_slice();
                let p_wind = (w_pid + NP as u8 - self.oya) % NP as u8;
                // Chankan yaku applies to kakan/ankan, but NOT to kita (BaBei).
                // MjSoul allows ron on kita tiles but does not award chankan yaku.
                let is_chankan = self
                    .pending_kan
                    .as_ref()
                    .is_some_and(|(_, act)| act.action_type != ActionType::Kita);

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
                    kita_count: self.players[w_pid as usize].kita_count,
                    is_sanma: true,
                    num_players: NP as u8,
                };

                let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand, melds);
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
                            NP as u8,
                        );
                        res.ron_agari = capped.pay_ron;
                        res.tsumo_agari_oya = capped.pay_tsumo_oya;
                        res.tsumo_agari_ko = capped.pay_tsumo_ko;
                    }
                }

                if res.is_win {
                    let score = res.ron_agari as i32;
                    let mut pao_payer = target_pid;
                    let mut pao_amt = 0;

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
                            // Ron with PAO: split between PAO player and deal-in player.
                            // yakuman_pao_is_liability_only controls the split base:
                            //   true  (MjSoul): only PAO-triggering yakuman portion split 50/50
                            //   false (Tenhou): total yakuman split 50/50
                            let is_oya = w_pid == self.oya;
                            let unit: i32 = if is_oya { 48000 } else { 32000 };
                            let honba_ron = ron_honba as i32 * (NP as i32 - 1) * 100;
                            let split_base = if self.rule.yakuman_pao_is_liability_only {
                                pao_yakuman_val * unit
                            } else {
                                total_yakuman_val * unit
                            };
                            pao_amt = (split_base / 2 + honba_ron) as usize;
                        }
                    }

                    let mut this_deltas = [0i32; NP];
                    this_deltas[w_pid as usize] += score;
                    this_deltas[pao_payer as usize] -= pao_amt as i32;
                    this_deltas[target_pid as usize] -= score - pao_amt as i32;

                    total_deltas[w_pid as usize] += score;
                    total_deltas[pao_payer as usize] -= pao_amt as i32;
                    total_deltas[target_pid as usize] -= score - pao_amt as i32;

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

            // Discard was called → discarder loses nagashi eligibility
            if let Some((discarder_pid, _)) = self.last_discard {
                self.players[discarder_pid as usize].nagashi_eligible = false;
            }

            for p in 0..NP {
                self.players[p].ippatsu_cycle = false;
            }

            if action.action_type == ActionType::Daiminkan {
                self.current_player = claimer;
                self.set_single_active_player(claimer);
                self.players[claimer as usize].clear_forbidden();
                self._resolve_kan(claimer, action);
                return;
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
            let mut tiles = action.consume_slice().to_vec();
            tiles.push(tile);
            tiles.sort();
            let meld_type = match action.action_type {
                ActionType::Pon => MeldType::Pon,
                _ => MeldType::Pon,
            };
            self.players[claimer as usize].push_meld(Meld::new(
                meld_type,
                &tiles,
                true,
                discarder as i8,
                Some(tile),
            ));

            if !self.skip_mjai_logging {
                let type_str = match action.action_type {
                    ActionType::Pon => Some("pon"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    _ => None,
                };
                if let Some(s) = type_str {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String(s.to_string()));
                    ev.insert("actor".to_string(), Value::Number(claimer.into()));
                    ev.insert("target".to_string(), Value::Number(discarder.into()));
                    ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
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
                if pk_act.action_type == ActionType::Kita {
                    // All players passed on kita ron — break ippatsu now
                    for p in &mut self.players {
                        p.ippatsu_cycle = false;
                    }
                    self.resolve_kita_rinshan(pk_pid);
                } else {
                    self._resolve_kan(pk_pid, pk_act);
                }
            } else {
                self._accept_riichi();
                self.turn_count += 1;
                self.current_player = (self.current_player + 1) % NP as u8;
                self._deal_next();
                if self.turn_count >= NP as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }

    /// Step with array-indexed actions instead of HashMap.
    ///
    /// `actions[pid]` = `Some(action)` if player pid has an action, `None` otherwise.
    /// Thin wrapper that converts to HashMap and delegates to `step()`.
    pub fn step_array(&mut self, actions: &[Option<Action>; 3]) {
        let mut map = std::collections::HashMap::with_capacity(3);
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
    pub fn step_unchecked(&mut self, actions: &[Option<Action>; 3]) {
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
    pub fn step_array_unchecked(&mut self, actions: &[Option<Action>; 3]) {
        self.step_unchecked(actions);
    }

    fn _resolve_discard(&mut self, pid: u8, tile: u8, tsumogiri: bool) {
        // A normal discard is never chankan, so clear any stale pending_kan
        // to prevent false chankan detection on subsequent ron claims.
        self.pending_kan = None;

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
        let mut claim_active = Vec::new();

        for i in 0..NP as u8 {
            if i == pid {
                continue;
            }
            let (legals, missed_agari) = self._get_claim_actions_for_player(i, pid, tile);
            if missed_agari {
                self.players[i as usize].missed_agari_doujun = true;
            }
            if !legals.is_empty() {
                has_claims = true;
                claim_active.push(i);
                self.set_claims_from_vec(i as usize, &legals);
            }
        }

        if has_claims {
            self.phase = Phase::WaitResponse;
            self.set_active_players_from_slice(&claim_active);
        } else {
            if let Some(_rp) = self.riichi_pending_acceptance {
                self._accept_riichi();
            }
            if !self.check_abortive_draw() {
                self.turn_count += 1;
                self.current_player = (pid + 1) % NP as u8;
                self._deal_next();
                if self.turn_count >= NP as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }

    pub fn _resolve_kan(&mut self, pid: u8, action: Action) {
        let p_idx = pid as usize;
        if action.action_type == ActionType::Kakan {
            // Already updated in step()
        } else {
            for &t in action.consume_slice() {
                if let Some(idx) = self.players[p_idx]
                    .hand_slice()
                    .iter()
                    .position(|&x| x == t)
                {
                    self.players[p_idx].remove_hand(idx);
                }
            }
            let (m_type, tiles, from_who, ct) = if action.action_type == ActionType::Ankan {
                (MeldType::Ankan, action.consume_slice().to_vec(), -1i8, None)
            } else {
                // SAFETY: last_discard is always Some when processing daiminkan claims
                let (discarder, tile) = self.last_discard.unwrap();
                let mut t_vec = action.consume_slice().to_vec();
                t_vec.push(tile);
                t_vec.sort();
                (MeldType::Daiminkan, t_vec, discarder as i8, Some(tile))
            };
            self.players[p_idx].push_meld(Meld::new(
                m_type,
                &tiles,
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
            let t = self.wall.draw_rinshan();
            self.players[p_idx].push_hand(t);
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            if !self.skip_mjai_logging {
                let m_type = match action.action_type {
                    ActionType::Ankan => Some("ankan"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    ActionType::Kakan => None,
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

    pub fn _deal_next(&mut self) {
        self.is_rinshan_flag = false;
        if self.wall.remaining() <= 14 {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }
        if let Some(t) = self.wall.draw_back() {
            let pid = self.current_player;
            self.players[pid as usize].push_hand(t);
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

    pub fn _initialize_next_round(&mut self, oya_won: bool, is_draw: bool) {
        if self.is_done {
            return;
        }

        let np: u8 = NP as u8;

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
            4 => {
                // 3p-red-east
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 1 && (max_score >= 30000 || next_round_wind > 1) {
                    self._process_end_game();
                    return;
                }
            }
            5 => {
                // 3p-red-half
                let max_score = self.players.iter().map(|p| p.score).max().unwrap_or(0);
                if next_round_wind >= 2 && (max_score >= 30000 || next_round_wind > 2) {
                    self._process_end_game();
                    return;
                }
            }
            3 => {
                // 3p-red-single
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

    pub fn _initialize_round(
        &mut self,
        oya: u8,
        round_wind: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<Vec<i32>>,
    ) {
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
        self.win_results = [None; NP];
        self.last_win_results = [None; NP];
        self.riichi_sutehais = [None; NP];
        self.last_tedashis = [None; NP];

        if let Some(s) = scores {
            for (i, &sc) in s.iter().enumerate() {
                if i < NP {
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
            for idx in 0..NP {
                let p = (idx + oya as usize) % NP;
                for _ in 0..4 {
                    if let Some(t) = self.wall.draw_back() {
                        self.players[p].push_hand(t);
                    }
                }
            }
        }
        for idx in 0..NP {
            let p = (idx + oya as usize) % NP;
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

        if let Some(t) = self.wall.draw_back() {
            self.players[self.oya as usize].push_hand(t);
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

    pub fn _trigger_ryukyoku(&mut self, reason: &str) {
        self._accept_riichi();

        let mut tenpai = [false; NP];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

        if reason == "exhaustive_draw" {
            for (i, p) in self.players.iter().enumerate() {
                let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(&p.hand, &p.melds);
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
                    let score_res = crate::score::calculate_score(5, 30, is_oya, true, 0, NP as u8);
                    if is_oya {
                        for i in 0..NP {
                            if i as u8 != w {
                                self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                self.players[i].score_delta -= score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                self.players[w as usize].score_delta +=
                                    score_res.pay_tsumo_ko as i32;
                            }
                        }
                    } else {
                        for i in 0..NP {
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
                let tenpai_pool = game_mode::tenpai_pool();
                let num_tp = tenpai.iter().filter(|&&t| t).count();
                if num_tp > 0 && num_tp < NP {
                    let pk = tenpai_pool / num_tp as i32;
                    let pn = tenpai_pool / (NP - num_tp) as i32;
                    for (i, tp) in tenpai.iter().enumerate() {
                        let delta = if *tp { pk } else { -pn };
                        self.players[i].score += delta;
                        self.players[i].score_delta = delta;
                    }
                }
            }
        } else if let Some(stripped) = reason.strip_prefix("Error: Illegal Action by Player ") {
            if let Ok(pid) = stripped.parse::<usize>() {
                if pid < NP {
                    let is_offender_oya = (pid as u8) == self.oya;
                    if is_offender_oya {
                        let penalty = 4000 * (NP as i32 - 1);
                        let each_get = penalty / (NP as i32 - 1);
                        for i in 0..NP {
                            if i == pid {
                                self.players[i].score -= penalty;
                                self.players[i].score_delta = -penalty;
                            } else {
                                self.players[i].score += each_get;
                                self.players[i].score_delta = each_get;
                            }
                        }
                    } else {
                        let total_penalty = 4000 + 2000 * (NP as i32 - 2);
                        for i in 0..NP {
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
        // 1. Sufuurenta (Four Winds) - disabled in 3P
        // Sufuurenta requires all 4 players to discard the same wind tile.
        // With only 3 players this rule does not apply (MjSoul 3P confirmed).

        // 2. Suukansansen (4 Kans)
        let mut kan_owners = Vec::new();
        for (pid, p) in self.players.iter().enumerate() {
            for m in &p.melds {
                if m.meld_type == MeldType::Daiminkan
                    || m.meld_type == MeldType::Ankan
                    || m.meld_type == MeldType::Kakan
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

        // 3. Suucha Riichi (All Riichis) - disabled in 3P
        // Suucha riichi requires all 4 players to declare riichi.
        // With only 3 players this rule does not apply (MjSoul 3P confirmed).

        false
    }

    pub fn _reveal_kan_dora(&mut self) {
        let count = self.wall.dora_indicator_count as usize;
        if count < 5 {
            let new_dora = self.wall.dora_indicator_tiles[count];
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

    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for i in 0..self.wall.dora_indicator_count as usize {
            markers.push(tid_to_mjai(self.wall.ura_indicator_tiles[i]));
        }
        markers
    }

    fn _get_ura_indicators(&self) -> Vec<u8> {
        let mut indicators = Vec::new();
        for i in 0..self.wall.dora_indicator_count as usize {
            indicators.push(self.wall.ura_indicator_tiles[i]);
        }
        indicators
    }

    pub(crate) fn _process_end_game(&mut self) {
        self.is_done = true;
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub fn apply_mjai_event(&mut self, event: MjaiEvent) {
        <Self as GameState3PEventHandler>::apply_mjai_event(self, event)
    }

    pub fn apply_log_action(&mut self, action: &LogAction) {
        <Self as GameState3PEventHandler>::apply_log_action(self, action)
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
}

impl GameState3P {
    pub fn _push_mjai_event(&mut self, event: Value) {
        if self.skip_mjai_logging {
            return;
        }
        // SAFETY: serialization of serde_json::Value always succeeds
        let json_str = serde_json::to_string(&event).unwrap();
        self.mjai_log.push(json_str.clone());

        let type_str = event["type"].as_str().unwrap_or("");
        let actor = event["actor"].as_u64().map(|a| a as usize);

        for pid in 0..NP {
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

    fn test_state_with_mode(game_mode: u8, skip_mjai_logging: bool) -> GameState3P {
        GameState3P::new(
            game_mode,
            skip_mjai_logging,
            Some(7),
            0,
            GameRule::default_tenhou(),
        )
    }

    fn test_state(skip_mjai_logging: bool) -> GameState3P {
        test_state_with_mode(5, skip_mjai_logging)
    }

    fn tiles_to_u32(tiles: &[u8]) -> Vec<u32> {
        tiles.iter().copied().map(u32::from).collect()
    }

    fn test_start_kyoku_event() -> Value {
        serde_json::json!({
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [35000, 35000, 35000],
            "dora_marker": "1p",
            "tehais": [
                ["1p", "2p", "3p"],
                ["4p", "5p", "6p"],
                ["7p", "8p", "9p"]
            ]
        })
    }

    #[test]
    fn helper_methods_manage_active_players_and_claims() {
        let mut state = test_state(true);

        state.set_single_active_player(2);
        assert_eq!(state.active_player_slice(), &[2]);

        state.set_active_players_from_slice(&[1, 2]);
        assert_eq!(state.active_player_slice(), &[1, 2]);

        state.clear_active_players();
        assert!(state.active_player_slice().is_empty());

        let ron = Action::new(ActionType::Ron, Some(88), &[], Some(1));
        let pon = Action::new(ActionType::Pon, Some(88), &[84, 85], Some(1));
        state.push_claim(1, ron);
        state.push_claim(1, pon);
        assert_eq!(state.claims_slice(1), &[ron, pon]);

        let many_claims = vec![Action::new(ActionType::Pass, None, &[], Some(1)); 60];
        state.set_claims_from_vec(1, &many_claims);
        assert_eq!(state.current_claim_counts[1], 54);
        assert_eq!(state.claims_slice(1).len(), 54);
        assert!(state
            .claims_slice(1)
            .iter()
            .all(|claim| claim.action_type == ActionType::Pass));

        state.clear_claims();
        assert_eq!(state.current_claim_counts, [0; 3]);
        assert!(state.claims_slice(1).is_empty());
    }

    #[test]
    fn reset_rebuilds_logging_buffers_when_logging_enabled() {
        let mut state = test_state(false);
        state.reset();

        state.mjai_log.push("stale".to_string());
        state.mjai_log_per_player[0].push("p0".to_string());
        state.mjai_log_per_player[1].push("p1".to_string());
        state.player_event_counts = [3, 2, 1];

        state.reset();

        assert_eq!(state.player_event_counts, [0; 3]);
        assert_eq!(state.mjai_log.len(), 1);
        assert!(state.mjai_log[0].contains("start_game"));
        assert_eq!(state.mjai_log_per_player[0].len(), 1);
        assert_eq!(state.mjai_log_per_player[1].len(), 1);
        assert_eq!(state.mjai_log_per_player[2].len(), 1);
    }

    #[test]
    fn reset_drops_logs_without_readding_events_when_logging_is_disabled() {
        let mut state = test_state(true);
        state.mjai_log.push("stale".to_string());
        state.mjai_log_per_player[0].push("p0".to_string());
        state.player_event_counts = [4, 5, 6];

        state.reset();

        assert!(state.mjai_log.is_empty());
        assert!(state
            .mjai_log_per_player
            .iter()
            .all(|events| events.is_empty()));
        assert_eq!(state.player_event_counts, [0; 3]);
    }

    #[test]
    fn reset_for_new_game_recreates_state_with_new_seed() {
        let mut state = test_state(true);
        state.current_player = 2;
        state.turn_count = 9;
        state.pending_oya_won = true;
        state.pending_is_draw = true;
        state.needs_initialize_next_round = true;
        state.active_players = [2, 1, 0, 0];
        state.active_player_count = 3;
        state.last_discard = Some((1, 40));
        state.riichi_sticks = 4;

        state.reset_for_new_game(Some(99));

        assert_eq!(state.seed, Some(99));
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.turn_count, 0);
        assert!(!state.pending_oya_won);
        assert!(!state.pending_is_draw);
        assert!(!state.needs_initialize_next_round);
        assert_eq!(state.active_player_slice(), &[0]);
        assert_eq!(state.last_discard, None);
        assert_eq!(state.riichi_sticks, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.game_mode, 5);
        assert!(state.skip_mjai_logging);
    }

    #[test]
    fn initialize_next_round_keeps_oya_and_increments_honba_after_oya_win() {
        let mut state = test_state(true);
        state.oya = 1;
        state.honba = 2;
        state.round_wind = 0;
        state.current_player = 2;
        state.phase = Phase::WaitResponse;
        state.active_player_count = 0;

        state._initialize_next_round(true, false);

        assert_eq!(state.oya, 1);
        assert_eq!(state.honba, 3);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[1]);
        assert!(state.drawn_tile.is_some());
    }

    #[test]
    fn initialize_next_round_draw_wraps_oya_and_advances_round_wind() {
        let mut state = test_state(true);
        state.oya = 2;
        state.honba = 1;
        state.round_wind = 0;

        state._initialize_next_round(false, true);

        assert_eq!(state.oya, 0);
        assert_eq!(state.honba, 2);
        assert_eq!(state.round_wind, 1);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.active_player_slice(), &[0]);
    }

    #[test]
    fn initialize_next_round_loss_resets_honba_and_rotates_oya() {
        let mut state = test_state(true);
        state.oya = 1;
        state.honba = 3;
        state.round_wind = 0;

        state._initialize_next_round(false, false);

        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 0);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.active_player_slice(), &[2]);
    }

    #[test]
    fn reveal_kan_dora_stops_after_five_indicators() {
        let mut state = test_state(true);

        for _ in 0..10 {
            state._reveal_kan_dora();
        }

        assert_eq!(state.wall.dora_indicator_count, 5);
        assert_eq!(state.wall.dora_indicator_slice().len(), 5);
    }

    #[test]
    fn reveal_kan_dora_logs_marker_and_ura_helpers_follow_indicator_count() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        let expected_marker = tid_to_mjai(state.wall.dora_indicator_tiles[1]);
        let expected_ura = vec![
            tid_to_mjai(state.wall.ura_indicator_tiles[0]),
            tid_to_mjai(state.wall.ura_indicator_tiles[1]),
        ];
        let expected_ura_ids = vec![
            state.wall.ura_indicator_tiles[0],
            state.wall.ura_indicator_tiles[1],
        ];

        state._reveal_kan_dora();

        assert_eq!(state.wall.dora_indicator_count, 2);
        assert_eq!(state._get_ura_markers(), expected_ura);
        assert_eq!(state._get_ura_indicators(), expected_ura_ids);

        let event: Value = serde_json::from_str(state.mjai_log.last().unwrap()).unwrap();
        assert_eq!(event["type"], Value::String("dora".to_string()));
        assert_eq!(event["dora_marker"], Value::String(expected_marker));
    }

    #[test]
    fn push_mjai_event_masks_start_kyoku_hands_for_other_players() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._push_mjai_event(test_start_kyoku_event());

        let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
        let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();

        assert_eq!(p0["tehais"][0], serde_json::json!(["1p", "2p", "3p"]));
        assert_eq!(p0["tehais"][1], serde_json::json!(["?", "?", "?"]));
        assert_eq!(p1["tehais"][0], serde_json::json!(["?", "?", "?"]));
        assert_eq!(p1["tehais"][1], serde_json::json!(["4p", "5p", "6p"]));
    }

    #[test]
    fn push_mjai_event_masks_tsumo_tile_for_non_actor_players() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._push_mjai_event(serde_json::json!({
            "type": "tsumo",
            "actor": 1,
            "pai": "5p"
        }));

        let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
        let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();

        assert_eq!(p0["pai"], Value::String("?".to_string()));
        assert_eq!(p1["pai"], Value::String("5p".to_string()));
    }

    #[test]
    fn push_mjai_event_is_noop_when_logging_disabled() {
        let mut state = test_state(true);

        state._push_mjai_event(serde_json::json!({ "type": "tsumo", "actor": 0, "pai": "1p" }));

        assert!(state.mjai_log.is_empty());
        assert!(state
            .mjai_log_per_player
            .iter()
            .all(|events| events.is_empty()));
    }

    #[test]
    fn get_observation_masks_other_hands_and_drains_only_new_events() {
        let mut state = test_state(true);
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
        let mut state = test_state(true);

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
    fn process_end_game_marks_done_and_emits_end_game_only_when_logging_enabled() {
        let mut logged = test_state(false);
        logged.mjai_log.clear();
        logged.mjai_log_per_player = Default::default();

        logged._process_end_game();

        assert!(logged.is_done);
        let logged_event: Value = serde_json::from_str(logged.mjai_log.last().unwrap()).unwrap();
        assert_eq!(logged_event["type"], Value::String("end_game".to_string()));
        assert_eq!(logged.mjai_log_per_player[0].len(), 1);
        assert_eq!(logged.mjai_log_per_player[1].len(), 1);
        assert_eq!(logged.mjai_log_per_player[2].len(), 1);

        let mut silent = test_state(true);
        silent._process_end_game();

        assert!(silent.is_done);
        assert!(silent.mjai_log.is_empty());
        assert!(silent
            .mjai_log_per_player
            .iter()
            .all(|events| events.is_empty()));
    }

    #[test]
    fn check_abortive_draw_triggers_suukansansen_only_with_four_kans_by_multiple_players() {
        let mut triggered = test_state(true);
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
        let scores_before: Vec<i32> = triggered
            .players
            .iter()
            .map(|player| player.score)
            .collect();

        assert!(triggered.check_abortive_draw());
        assert!(!triggered.needs_tsumo);
        assert!(triggered.drawn_tile.is_some());
        assert_eq!(triggered.phase, Phase::WaitAct);
        assert_eq!(triggered.current_player, triggered.oya);
        assert_eq!(triggered.active_player_slice(), &[triggered.oya]);
        let scores_after: Vec<i32> = triggered
            .players
            .iter()
            .map(|player| player.score)
            .collect();
        assert_eq!(scores_after, scores_before);
        assert!(triggered
            .players
            .iter()
            .all(|player| player.score_delta == 0));

        let mut same_owner = test_state(true);
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

        let mut not_enough = test_state(true);
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
    fn trigger_ryukyoku_illegal_action_penalizes_non_oya_offender_and_keeps_renchan() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 0;
        state.current_player = 1;
        state.phase = Phase::WaitResponse;

        state._trigger_ryukyoku("Error: Illegal Action by Player 1");

        assert_eq!(state.players[0].score, 39000);
        assert_eq!(state.players[0].score_delta, 4000);
        assert_eq!(state.players[1].score, 29000);
        assert_eq!(state.players[1].score_delta, -6000);
        assert_eq!(state.players[2].score, 37000);
        assert_eq!(state.players[2].score_delta, 2000);
        assert_eq!(state.oya, 0);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);

        let ryukyoku_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"ryukyoku\""))
            .expect("ryukyoku event should be logged");
        let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
        assert_eq!(
            ryukyoku_event["reason"],
            Value::String("Error: Illegal Action by Player 1".to_string())
        );
        assert_eq!(
            ryukyoku_event["deltas"],
            serde_json::json!([4000, -6000, 2000])
        );
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    }

    #[test]
    fn initialize_next_round_ends_game_on_negative_score_before_restarting() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[2].score = -100;
        state.oya = 1;
        state.honba = 2;
        state.round_wind = 0;

        state._initialize_next_round(false, false);

        assert!(state.is_done);
        assert_eq!(state.oya, 1);
        assert_eq!(state.honba, 2);
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
    fn initialize_next_round_single_mode_ends_game_immediately() {
        let mut state = test_state_with_mode(3, false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 2;
        state.honba = 1;
        state.round_wind = 0;

        state._initialize_next_round(false, true);

        assert!(state.is_done);
        assert_eq!(state.oya, 2);
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
    fn initialize_next_round_east_mode_continues_into_south_if_nobody_has_30000() {
        let mut state = test_state_with_mode(4, false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[0].score = 29000;
        state.players[1].score = 28000;
        state.players[2].score = 27000;
        state.oya = 2;
        state.honba = 1;
        state.round_wind = 0;

        state._initialize_next_round(false, true);

        assert!(!state.is_done);
        assert_eq!(state.oya, 0);
        assert_eq!(state.honba, 2);
        assert_eq!(state.round_wind, 1);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.active_player_slice(), &[0]);
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
    fn initialize_next_round_east_mode_ends_when_south_starts_with_30000_leader() {
        let mut state = test_state_with_mode(4, false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.players[0].score = 31000;
        state.players[1].score = 25000;
        state.players[2].score = 24000;
        state.oya = 2;
        state.honba = 1;
        state.round_wind = 0;

        state._initialize_next_round(false, true);

        assert!(state.is_done);
        assert_eq!(state.oya, 2);
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
    fn initialize_round_uses_provided_wall_scores_and_resets_round_state() {
        let mut state = test_state(true);
        state.phase = Phase::WaitResponse;
        state.is_done = true;
        state.pending_kan = Some((1, Action::new(ActionType::Kakan, Some(16), &[], Some(1))));
        state.is_rinshan_flag = true;
        state.wall.rinshan_draw_count = 3;
        state.wall.pending_kan_dora_count = 2;
        state.is_first_turn = false;
        state.riichi_pending_acceptance = Some(2);
        state.turn_count = 8;
        state.needs_tsumo = false;
        state.needs_initialize_next_round = true;
        state.pending_oya_won = true;
        state.pending_is_draw = true;
        state.last_discard = Some((2, 44));
        state.win_results[0] = Some(WinResult::new(
            false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
        ));
        state.last_win_results[1] = Some(WinResult::new(
            false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
        ));
        state.riichi_sutehais = [Some(12), Some(16), Some(20)];
        state.last_tedashis = [Some(24), Some(28), Some(32)];
        state.active_players = [2, 1, 0, 0];
        state.active_player_count = 3;
        state.players[0].riichi_declared = true;
        state.players[1].double_riichi_declared = true;
        state.players[2].missed_agari_doujun = true;
        state.players[0].nagashi_eligible = false;
        state.players[1].ippatsu_cycle = true;
        state.players[2].push_forbidden(60);
        state.players[0].pao_insert(37, 2);

        state._initialize_round(
            1,
            1,
            2,
            3,
            Some((0..108).collect()),
            Some(vec![11000, 22000, 33000]),
        );

        assert_eq!(state.oya, 1);
        assert_eq!(state.kyoku_idx, 1);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.honba, 2);
        assert_eq!(state.riichi_sticks, 3);
        assert_eq!(state.round_wind, 1);
        assert!(!state.is_done);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[1]);
        assert!(state.pending_kan.is_none());
        assert!(!state.is_rinshan_flag);
        assert_eq!(state.wall.rinshan_draw_count, 0);
        assert_eq!(state.wall.pending_kan_dora_count, 0);
        assert!(state.is_first_turn);
        assert!(state.riichi_pending_acceptance.is_none());
        assert_eq!(state.turn_count, 0);
        assert!(!state.needs_tsumo);
        assert!(!state.needs_initialize_next_round);
        assert!(!state.pending_oya_won);
        assert!(!state.pending_is_draw);
        assert_eq!(state.last_discard, None);
        assert!(state.win_results.iter().all(Option::is_none));
        assert!(state.last_win_results.iter().all(Option::is_none));
        assert_eq!(state.riichi_sutehais, [None; 3]);
        assert_eq!(state.last_tedashis, [None; 3]);
        assert_eq!(state.players[0].score, 11000);
        assert_eq!(state.players[1].score, 22000);
        assert_eq!(state.players[2].score, 33000);
        assert!(!state.players[0].riichi_declared);
        assert!(!state.players[1].double_riichi_declared);
        assert!(!state.players[2].missed_agari_doujun);
        assert!(state.players[0].nagashi_eligible);
        assert!(!state.players[1].ippatsu_cycle);
        assert!(state.players[2].forbidden_slice().is_empty());
        assert_eq!(state.players[0].pao_count, 0);
        assert_eq!(state.players[0].hand_slice().len(), 13);
        assert_eq!(state.players[1].hand_slice().len(), 14);
        assert_eq!(state.players[2].hand_slice().len(), 13);
        assert_eq!(state.wall.tile_count, 68);
        assert_eq!(state.wall.draw_cursor, 0);
        assert_eq!(state.wall.dora_indicator_slice(), &[99]);
        assert_eq!(state.drawn_tile, Some(39));
    }

    #[test]
    fn deal_next_draws_from_wall_and_clears_forbidden_discards() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.current_player = 1;
        state.phase = Phase::WaitResponse;
        state.needs_tsumo = true;
        state.is_rinshan_flag = true;
        state.drawn_tile = None;
        state.players[1].push_forbidden(44);

        let expected_tile = state.wall.tiles[state.wall.tile_count as usize - 1];
        let hand_len_before = state.players[1].hand_slice().len();

        state._deal_next();

        assert!(!state.is_rinshan_flag);
        assert_eq!(state.drawn_tile, Some(expected_tile));
        assert_eq!(state.players[1].hand_slice().len(), hand_len_before + 1);
        assert!(!state.needs_tsumo);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[1]);
        assert!(state.players[1].forbidden_slice().is_empty());

        let event: Value = serde_json::from_str(state.mjai_log.last().unwrap()).unwrap();
        assert_eq!(event["type"], Value::String("tsumo".to_string()));
        assert_eq!(event["actor"], Value::Number(1.into()));
        assert_eq!(event["pai"], Value::String(tid_to_mjai(expected_tile)));
    }

    #[test]
    fn deal_next_exhaustive_draw_with_oya_nagashi_keeps_renchan_and_scores() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 2;
        state.current_player = 1;
        state.honba = 0;
        state.round_wind = 0;
        state.wall.tile_count = 14;
        state.wall.draw_cursor = 0;
        state.players[0].nagashi_eligible = false;
        state.players[1].nagashi_eligible = false;
        state.players[2].nagashi_eligible = true;

        let nagashi_score = crate::score::calculate_score(5, 30, true, true, 0, 3);

        state._deal_next();

        assert_eq!(
            state.players[0].score,
            35000 - nagashi_score.pay_tsumo_ko as i32
        );
        assert_eq!(
            state.players[1].score,
            35000 - nagashi_score.pay_tsumo_ko as i32
        );
        assert_eq!(
            state.players[2].score,
            35000 + 2 * nagashi_score.pay_tsumo_ko as i32
        );
        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 0);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[2]);
        assert!(state.drawn_tile.is_some());

        let ryukyoku_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"ryukyoku\""))
            .expect("ryukyoku event should be logged");
        let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
        assert_eq!(
            ryukyoku_event["reason"],
            Value::String("nagashimangan".to_string())
        );
        assert_eq!(
            ryukyoku_event["deltas"],
            serde_json::json!([
                -(nagashi_score.pay_tsumo_ko as i32),
                -(nagashi_score.pay_tsumo_ko as i32),
                2 * nagashi_score.pay_tsumo_ko as i32,
            ])
        );
    }

    #[test]
    fn trigger_ryukyoku_accepts_pending_riichi_before_scoring_and_carries_stick_forward() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 0;
        state.riichi_pending_acceptance = Some(2);

        state._trigger_ryukyoku("Error: Illegal Action by Player 1");

        assert_eq!(state.players[0].score, 39000);
        assert_eq!(state.players[1].score, 29000);
        assert_eq!(state.players[2].score, 36000);
        assert_eq!(state.riichi_sticks, 1);
        assert!(state.riichi_pending_acceptance.is_none());
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"reach_accepted\"")));
        let reach_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"reach_accepted\""))
            .expect("reach_accepted should be logged");
        let ryukyoku_idx = state
            .mjai_log
            .iter()
            .position(|event| event.contains("\"type\":\"ryukyoku\""))
            .expect("ryukyoku should be logged");
        assert!(reach_idx < ryukyoku_idx);
    }

    #[test]
    fn trigger_ryukyoku_illegal_action_penalizes_oya_offender() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 0;

        state._trigger_ryukyoku("Error: Illegal Action by Player 0");

        assert_eq!(state.players[0].score, 27000);
        assert_eq!(state.players[0].score_delta, -8000);
        assert_eq!(state.players[1].score, 39000);
        assert_eq!(state.players[1].score_delta, 4000);
        assert_eq!(state.players[2].score, 39000);
        assert_eq!(state.players[2].score_delta, 4000);
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
            serde_json::json!([-8000, 4000, 4000])
        );
    }

    #[test]
    fn replay_observation_retries_discard_after_temporarily_clearing_riichi() {
        let mut state = test_state(true);
        let drawn_tile = state
            .drawn_tile
            .expect("test state should start with a drawn tile");
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
    fn push_mjai_event_keeps_tsumo_tile_visible_when_actor_is_missing() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._push_mjai_event(serde_json::json!({
            "type": "tsumo",
            "pai": "5p"
        }));

        for pid in 0..3 {
            let event: Value = serde_json::from_str(&state.mjai_log_per_player[pid][0]).unwrap();
            assert_eq!(event["pai"], Value::String("5p".to_string()));
        }
    }

    #[test]
    fn replay_observation_temporarily_injects_claims_and_restores_state_on_success() {
        let mut state = test_state(true);
        state.active_players = [2, 1, 0, 0];
        state.active_player_count = 2;
        state.current_claim_counts[2] = 1;
        state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

        let obs = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Ron, Some(48), &[], Some(1)),
                "{\"type\":\"hora\"}",
            )
            .expect("ron replay action should be exposed as legal");

        let legal_actions = obs.legal_actions_method();
        assert!(legal_actions
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(legal_actions
            .iter()
            .any(|action| action.action_type == ActionType::Pass));

        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_players, [2, 1, 0, 0]);
        assert_eq!(state.active_player_count, 2);
        assert_eq!(state.current_claim_counts, [0, 0, 1]);
        assert_eq!(state.current_claims[2][0].action_type, ActionType::Pass);
    }

    #[test]
    fn replay_observation_restores_state_after_invalid_action_error() {
        let mut state = test_state(true);
        state.active_players = [0, 2, 0, 0];
        state.active_player_count = 2;
        state.current_claim_counts[2] = 1;
        state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

        let err = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Discard, Some(0), &[], Some(1)),
                "{\"type\":\"dahai\"}",
            )
            .expect_err("non-active player discard should stay illegal in replay observation");

        assert!(matches!(err, RiichiError::InvalidState { .. }));
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_players, [0, 2, 0, 0]);
        assert_eq!(state.active_player_count, 2);
        assert_eq!(state.current_claim_counts, [0, 0, 1]);
        assert_eq!(state.current_claims[2][0].action_type, ActionType::Pass);
    }

    #[test]
    fn step_array_unchecked_initializes_pending_round_before_processing_actions() {
        let mut state = test_state(true);
        state.oya = 1;
        state.honba = 2;
        state.round_wind = 0;
        state.phase = Phase::WaitResponse;
        state.needs_initialize_next_round = true;
        state.pending_oya_won = false;
        state.pending_is_draw = false;
        state.last_discard = Some((0, 44));

        let actions = [
            Some(Action::new(
                ActionType::Discard,
                state.drawn_tile,
                &[],
                Some(state.current_player),
            )),
            None,
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
    }

    #[test]
    fn replay_ankan_matcher_accepts_same_tile_class_with_different_copy_ids() {
        let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay = Action::new(ActionType::Ankan, Some(17), &[17, 17, 17, 17], Some(0));

        assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_kakan_matcher_accepts_same_tile_class_with_different_copy_ids() {
        let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
        let replay = Action::new(ActionType::Kakan, Some(17), &[], Some(0));

        assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_kan_matcher_rejects_different_tile_classes() {
        let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
        let replay = Action::new(ActionType::Ankan, Some(20), &[20, 20, 20, 20], Some(0));

        assert!(!GameState3P::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_matcher_accepts_tileless_special_actions() {
        let legal = Action::new(ActionType::Kita, Some(120), &[], Some(0));
        let replay = Action::new(ActionType::Kita, None, &[], Some(0));

        assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn replay_matcher_accepts_matching_kan_consumes_even_without_matching_tiles() {
        let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
        let replay = Action::new(ActionType::Kakan, Some(52), &[16, 17, 18], Some(0));

        assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
    }

    #[test]
    fn initialize_next_round_returns_immediately_when_game_already_done() {
        let mut state = test_state(true);
        state.is_done = true;
        state.oya = 2;
        state.honba = 1;
        state.round_wind = 1;

        state._initialize_next_round(false, false);

        assert!(state.is_done);
        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 1);
    }

    #[test]
    fn deal_next_exhaustive_draw_without_nagashi_keeps_scores_even_and_renchan_depends_on_oya_tenpai(
    ) {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.oya = 1;
        state.current_player = 2;
        state.honba = 0;
        state.round_wind = 0;
        state.wall.tile_count = 14;
        state.wall.draw_cursor = 0;
        state.players.iter_mut().for_each(|player| {
            player.nagashi_eligible = false;
            player.score = 35_000;
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
            .copy_from_slice(&[0, 5, 9, 36, 41, 45, 72, 77, 81, 108, 112, 116, 120]);
        state.players[1].hand_len = 13;
        state.players[2].hand[..13]
            .copy_from_slice(&[1, 6, 10, 37, 42, 46, 73, 78, 82, 109, 113, 117, 121]);
        state.players[2].hand_len = 13;

        state._deal_next();

        assert_eq!(state.players[0].score, 35_000);
        assert_eq!(state.players[1].score, 35_000);
        assert_eq!(state.players[2].score, 35_000);
        assert!(state
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"ryukyoku\"")));
        assert_eq!(state.oya, 2);
        assert_eq!(state.honba, 1);
        assert_eq!(state.round_wind, 0);
    }

    #[test]
    fn process_end_game_is_idempotent_for_done_flag_and_logging_shape() {
        let mut logged = test_state(false);
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

        let mut silent = test_state(true);
        silent._process_end_game();
        silent._process_end_game();
        assert!(silent.is_done);
        assert!(silent.mjai_log.is_empty());
    }

    #[test]
    fn replay_matcher_rejects_mismatched_action_kinds_and_tiles() {
        let discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
        let riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
        assert!(!GameState3P::replay_action_matches_legal(&discard, &riichi));

        let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
        let replay_with_wrong_tile = Action::new(ActionType::Ron, Some(52), &[], Some(0));
        assert!(!GameState3P::replay_action_matches_legal(
            &legal_ron,
            &replay_with_wrong_tile
        ));
    }

    #[test]
    fn accept_riichi_is_noop_without_pending_player_and_logs_when_present() {
        let mut silent = test_state(true);
        let score_before = silent.players[0].score;
        silent._accept_riichi();
        assert_eq!(silent.players[0].score, score_before);
        assert_eq!(silent.riichi_sticks, 0);

        let mut logged = test_state(false);
        logged.mjai_log.clear();
        logged.mjai_log_per_player = Default::default();
        logged.riichi_pending_acceptance = Some(1);
        logged._accept_riichi();
        assert_eq!(logged.players[1].score, 34_000);
        assert_eq!(logged.players[1].score_delta, -1000);
        assert_eq!(logged.riichi_sticks, 1);
        assert!(logged.players[1].riichi_declared);
        assert!(logged.players[1].ippatsu_cycle);
        assert!(logged.riichi_pending_acceptance.is_none());
        assert!(logged
            .mjai_log
            .iter()
            .any(|event| event.contains("\"type\":\"reach_accepted\"")));
    }

    #[test]
    fn replay_observation_allows_pass_for_active_response_player_and_restores_state() {
        let mut state = test_state(true);
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
            .legal_actions_method()
            .iter()
            .any(|action| action.action_type == ActionType::Pass));
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_players, [1, 0, 0, 0]);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.current_claim_counts[1], 1);
    }

    #[test]
    fn wait_response_marks_missed_ron_and_riichi_when_player_passes_on_win() {
        let mut state = test_state(true);
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
        ]);

        assert!(state.players[1].missed_agari_doujun);
        assert!(state.players[1].missed_agari_riichi);
    }

    #[test]
    fn wait_response_resolves_pending_kakan_after_all_players_pass() {
        let mut state = test_state(true);
        state.phase = Phase::WaitResponse;
        state.current_player = 0;
        state.last_discard = Some((0, 48));
        state.active_players = [1, 2, 0, 0];
        state.active_player_count = 2;
        state.current_claim_counts[1] = 1;
        state.current_claim_counts[2] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
        state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
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
        ]);

        assert!(state.pending_kan.is_none());
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
        assert!(state.drawn_tile.is_some());
        assert!(state.is_rinshan_flag);
        assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
        assert_eq!(state.wall.pending_kan_dora_count, pending_before + 1);
    }

    #[test]
    fn resolve_discard_sets_riichi_pending_and_logs_after_tedashi() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.current_player = 0;
        let drawn = state
            .drawn_tile
            .expect("test state should start with a drawn tile");
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
    fn replay_observation_temporarily_injects_daiminkan_and_restores_state_on_success() {
        let mut state = test_state(true);
        state.phase = Phase::WaitResponse;
        state.active_players = [1, 0, 0, 0];
        state.active_player_count = 1;
        state.current_claim_counts[1] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));

        let obs = state
            .get_observation_for_replay(
                1,
                &Action::new(ActionType::Daiminkan, Some(48), &[48, 49, 50], Some(1)),
                "{\"type\":\"daiminkan\"}",
            )
            .expect("daiminkan replay action should be exposed as legal");

        assert!(obs
            .legal_actions_method()
            .iter()
            .any(|action| action.action_type == ActionType::Daiminkan));
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_players, [1, 0, 0, 0]);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.current_claim_counts[1], 1);
        assert_eq!(state.current_claims[1][0].action_type, ActionType::Pass);
    }

    #[test]
    fn abortive_draw_disabled_rules_stay_false_in_sanma() {
        let mut four_winds_like = test_state(true);
        for player in &mut four_winds_like.players {
            player.discards[0] = 108;
            player.discard_len = 1;
            player.meld_count = 0;
        }
        assert!(!four_winds_like.check_abortive_draw());

        let mut all_riichi = test_state(true);
        all_riichi
            .players
            .iter_mut()
            .for_each(|player| player.riichi_declared = true);
        assert!(!all_riichi.check_abortive_draw());
    }

    #[test]
    fn initialize_round_without_oya_draw_leaves_needs_tsumo_true() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();

        state._initialize_round(0, 0, 0, 0, Some((0..39).collect()), Some(vec![35_000; 3]));

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
    fn wrapper_methods_delegate_into_event_and_log_handlers() {
        let mut state = test_state(false);
        state.mjai_log.clear();
        state.mjai_log_per_player = Default::default();
        state.apply_mjai_event(MjaiEvent::Reach { actor: 0 });
        assert!(state.players[0].riichi_stage);

        let mut replay = test_state(false);
        replay.mjai_log.clear();
        replay.mjai_log_per_player = Default::default();
        let action = LogAction::DiscardTile {
            seat: 0,
            tile: replay
                .drawn_tile
                .expect("test state should start with a drawn tile"),
            is_liqi: false,
            is_wliqi: false,
            doras: None,
        };
        replay.apply_log_action(&action);
        assert!(replay.last_discard.is_some());
    }
}
