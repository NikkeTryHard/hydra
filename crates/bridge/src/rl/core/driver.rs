//! rl/core/driver: private hand lifecycle with legal views and applies.
//!
//! A hand opens with explicit ordinal ownership: the caller assigns the index
//! once, the engine seed derives from the schedule and the ordinal, and a
//! fresh engine state starts the round with the given wall, dealer, scores,
//! and wind. Hand zero uses the schedule tiles; later hands use continuation
//! walls. Draining parses newly appended engine log records once, translates
//! them to typed envelopes, and handles carries and terminals. Detection
//! reports either the single draw decision or the sorted window responders.
//! Legal views expand engine legals to canonical actions with a full mask;
//! observations bind the filtered view and re-check it independently. Applies
//! buffer window responders until all answer, then steps once with the
//! combined map; draw applies step once, with a two-step declaration plus
//! discard for riichi. Replay re-applies a canonical prefix against identical
//! walls. Any engine error fails closed. No Python.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use hydra_feed::census::ActionKind;
use riichienv_core::action::{Action as CoreAction, ActionType, Phase};
use riichienv_core::check_riichi_candidates;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::types::MeldType;

use crate::rl::actions::{
    CanonicalAction, canonical_id, engine_matches_canonical, expand_engine_legals, legal_view,
};
use crate::rl::events::{StateDelta, TypedEvent, Visibility};
use crate::rl::observation::{NativeMeld, NativeObservation, independent_validate};

use super::{
    BatchOutcome, BufferedDecision, CoreError, GAME_MODE_YON_HANCHAN, GameMode, HandCarry,
    PendingDecision, Stamp, TypedPayload, canonical_scores, seat_winds_for,
};
// Process-global legal-view cache counters, measurement only. Relaxed
// increments never feed control flow, so torn reads only perturb the
// diagnostic print; the cache behaves identically with counters present.
// Reported env-gated at pool close, so default runs emit nothing.
static VIEW_HITS: AtomicU64 = AtomicU64::new(0);
static VIEW_MISSES: AtomicU64 = AtomicU64::new(0);

/// Diagnostic view-cache totals for the env-gated close report.
pub(crate) fn view_cache_counts() -> (u64, u64) {
    (VIEW_HITS.load(Ordering::Relaxed), VIEW_MISSES.load(Ordering::Relaxed))
}

impl super::NativeGame {
    // -- hand lifecycle ----------------------------------------------------

    pub(super) fn open_hand(
        &mut self,
        hand_index: u64,
        tiles: &[u8; 136],
        oya_engine: u8,
        honba: u8,
        kyotaku: u32,
        scores_engine: [i32; 4],
        round_wind: u8,
    ) -> Result<(), CoreError> {
        self.hand_index = hand_index;
        let seed = crate::rl::wall::hand_rng_seed(
            &self.ctx.schedule_id,
            &self.ctx.digest_noprefix(),
            hand_index,
        );
        let mut state = GameState::new(
            GAME_MODE_YON_HANCHAN,
            false,
            Some(seed),
            round_wind,
            GameRule::default_tenhou(),
        );
        state.reset();
        state._initialize_round(
            oya_engine,
            round_wind,
            honba,
            kyotaku,
            Some(tiles.to_vec()),
            Some(scores_engine.to_vec()),
        );
        self.state = state;
        self.check_error()?;
        self.cursor = 0;
        self.mode = None;
        self.decision_seat = None;
        self.pending.clear();
        self.views.clear();
        self.buffered.clear();
        self.last_discard = (None, None);
        self.dahai_cursor = [0; 4];
        self.drawn_cache = [None; 4];
        self.riichi_states = [
            "none".to_string(),
            "none".to_string(),
            "none".to_string(),
            "none".to_string(),
        ];
        self.kan_count = 0;
        self.pre_step_hands.clear();
        self.kakan_added.clear();
        self.window_offered_ids.clear();
        self.draw_queue.clear();
        let initial_draws = self
            .state
            .mjai_log
            .iter()
            .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .filter(|v| v.get("type").and_then(|t| t.as_str()) == Some("tsumo"))
            .count();
        if initial_draws > 0
            && let Some(drawn) = self.state.drawn_tile
        {
            self.draw_queue.push_back(drawn);
        }
        Ok(())
    }

    fn reopen(&mut self, carry: HandCarry) -> Result<(), CoreError> {
        let nxt = self.hand_index + 1;
        let tiles = if nxt == 0 {
            self.ctx.schedule_tiles
        } else {
            crate::rl::wall::derive_hand_wall(&self.ctx.schedule_id, &self.ctx.digest_noprefix(), nxt)
                .map_err(|e| CoreError(format!("continuation wall rejected: {e:?}")))?
        };
        let wind = match carry.bakaze.as_str() {
            "E" => 0,
            "S" => 1,
            "W" => 2,
            "N" => 3,
            other => return Err(CoreError(format!("bad bakaze {other:?}"))),
        };
        self.open_hand(nxt, &tiles, carry.oya, carry.honba, carry.kyotaku, carry.scores, wind)
    }

    // -- drain / detect -----------------------------------------------------

    /// Drain new mjai records to typed envelopes, handle hand boundaries,
    /// then detect the pending decision.
    pub fn drain(&mut self) -> Result<Option<PendingDecision>, CoreError> {
        loop {
            if self.terminal {
                return Ok(None);
            }
            // A record-free re-drain cannot change this prepared draw. Every
            // engine step clears views first; skipping detection without both
            // guards would reuse stale legal actions after a kan or new draw.
            if self.cursor == self.state.mjai_log.len()
                && self.state.phase == Phase::WaitAct
                && !self.state.is_done
                && self.mode == Some(GameMode::Draw)
            {
                let seat = self.inv[self.state.current_player as usize] as u8;
                if self.decision_seat == Some(seat) && self.views.contains_key(&seat) {
                    return Ok(Some(PendingDecision::Draw { seat }));
                }
            }
            if self.cursor < self.state.mjai_log.len() {
                let records: Vec<serde_json::Value> = self.state.mjai_log[self.cursor..]
                    .iter()
                    .map(|s| {
                        serde_json::from_str(s)
                            .map_err(|e| CoreError(format!("core mjai record unparsable: {e}")))
                    })
                    .collect::<Result<_, _>>()?;
                let mut consumed = 0usize;
                let outcome = self.translate_batch(&records, &mut consumed)?;
                self.cursor += consumed;
                match outcome {
                    BatchOutcome::Continue => {
                        if consumed == 0 {
                            // Trailing split (e.g. lone `reach` whose declaration
                            // lands in the next batch): detect once on current
                            // state instead of spinning.
                            return self.detect();
                        }
                    }
                    BatchOutcome::Carry(carry) => {
                        self.reopen(carry)?;
                        continue;
                    }
                    BatchOutcome::Terminal => return Ok(None),
                }
            }
            if let Some(decision) = self.detect()? {
                return Ok(Some(decision));
            }
            if self.cursor >= self.state.mjai_log.len() {
                return Ok(None);
            }
        }
    }

    fn detect(&mut self) -> Result<Option<PendingDecision>, CoreError> {
        if self.terminal {
            return Ok(None);
        }
        if self.state.phase == Phase::WaitResponse {
            let mut responders = vec![];
            for pid in 0..4u8 {
                if !self.state.active_players.contains(&pid) {
                    continue;
                }
                if !self.trait_legals(pid)?.is_empty() {
                    responders.push(self.inv[pid as usize] as u8);
                }
            }
            if !responders.is_empty() {
                responders.sort_unstable();
                // Re-detects of the SAME window must not wipe buffered
                // answers: responders split across history buckets commit in
                // different lanes/iterations, with drains between the answers.
                // Identity = same responder set plus zero new arena events
                // since the fresh detect (answers buffer silently; only the
                // combined step appends). Unchanged windows keep buffered
                // answers/views and stage the still-pending seats only, so no
                // seat is ever asked twice and `call_window` emits once.
                if self.mode == Some(GameMode::Window) && self.arena.len() == self.window_arena_len
                {
                    let mut have: Vec<u8> =
                        self.pending.iter().chain(self.buffered.keys()).copied().collect();
                    have.sort_unstable();
                    if have == responders {
                        return Ok(Some(PendingDecision::Window { pending: self.pending.clone() }));
                    }
                }
                self.pending = responders.clone();
                self.mode = Some(GameMode::Window);
                self.decision_seat = None;
                self.views.clear();
                self.buffered.clear();
                // Rebuild legal plans for every responder (observe/describe/
                // commit re-derive from these; a missing plan fails closed).
                let mut offered = vec![];
                for seat in &responders {
                    let (actions, _) = self.view_for(*seat)?;
                    offered.extend(
                        actions
                            .iter()
                            .filter(|(_, a)| a.kind != ActionKind::Pass)
                            .map(|(id, _)| *id),
                    );
                }
                offered.sort_unstable();
                offered.dedup();
                self.window_offered_ids = offered;
                if self.window_opened_by_discard && self.arena_last_kind().as_deref() == Some("discard") {
                    let window = self.make_envelope(
                        "call_window",
                        None,
                        Visibility::Public,
                        vec![0, 1, 2, 3],
                        TypedPayload::empty(),
                        vec![],
                    )?;
                    self.emit(window)?;
                }
                // Record AFTER any fresh-detect emission: re-detects compare
                // against the post-emit length, so the first re-drain matches.
                self.window_arena_len = self.arena.len();
                return Ok(Some(PendingDecision::Window { pending: responders }));
            }
        }
        if self.state.phase == Phase::WaitAct && !self.state.is_done {
            let current = self.state.current_player;
            if !self.trait_legals(current)?.is_empty() {
                let seat = self.inv[current as usize] as u8;
                self.mode = Some(GameMode::Draw);
                self.decision_seat = Some(seat);
                self.views.clear();
                let _ = self.view_for(seat)?;
                return Ok(Some(PendingDecision::Draw { seat }));
            }
        }
        Ok(None)
    }

    // -- legal plans ----------------------------------------------------------

    fn trait_legals(&self, engine_pid: u8) -> Result<Vec<CoreAction>, CoreError> {
        Ok(self.state._get_legal_actions_internal(engine_pid))
    }

    fn phase_for(&self, seat: u8) -> String {
        let _ = seat;
        match self.mode {
            Some(GameMode::Window) => {
                if self.window_opened_by_discard {
                    "discard_response".to_string()
                } else {
                    "kan_response".to_string()
                }
            }
            _ => "draw_decision".to_string(),
        }
    }

    /// Sorted canonical actions plus the full legal mask over the frozen action table.
    /// View-prep note: every cache hit clones the full plan (ordered actions
    /// plus the 6792-bool mask), and a staged draw row re-derives it at
    /// observe, apply-verify, and commit — one build plus up to three clones
    /// per row. A borrowed or reference-counted view would need exactly the
    /// pre-step invalidation already enforced in this file (clears before
    /// every engine step); skipping any of them re-serves stale legality
    /// after a kan or new draw. Measured on the frozen 8x128 workload
    /// (HYDRA2_RL_VIEW_STATS=1): 2,742,855 hits versus 914,285 misses, exactly
    /// 3:1, i.e. one build plus three clones per staged row at roughly 8KB per
    /// clone (~22GB moved plus ~5.5M small allocations per run). A temporary
    /// Arc-shared build measured 336.0s versus 326.9-337.2s same-code range
    /// with 3776/3776 bitwise parity: the clones hide inside overlapped
    /// windows, so fusion saves nothing measurable. Reverted; do not re-fuse
    /// without new evidence, and fusing shares one object between observe,
    /// apply-verify, and commit, which weakens their independence anyway.
    pub fn view_for(
        &mut self,
        seat: u8,
    ) -> Result<(Vec<(u32, CanonicalAction)>, Vec<bool>), CoreError> {
        if let Some(cached) = self.views.get(&seat) {
            VIEW_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(cached.clone());
        }
        VIEW_MISSES.fetch_add(1, Ordering::Relaxed);
        let pid = self.ctx.seat_perm[seat as usize];
        let hand = self.state.players[pid as usize].hand.clone();
        let drawn = self.state.drawn_tile;
        let engine_legals = self.trait_legals(pid)?;
        let melds = self.actor_meld_pairs(seat);
        let offered = self.last_discard_offer(seat);
        let candidates = check_riichi_candidates(hand.clone());
        let phase = self.phase_for(seat);
        let expanded = expand_engine_legals(
            &engine_legals,
            seat,
            &phase,
            drawn,
            &hand,
            offered,
            &melds,
            &candidates,
        )?;
        let view = legal_view(expanded)?;
        self.views.insert(seat, view.clone());
        Ok(view)
    }

    fn actor_meld_pairs(&self, seat: u8) -> Vec<(String, Vec<u8>)> {
        self.engine_melds(seat).iter().map(|m| (m.kind.clone(), m.tiles.clone())).collect()
    }

    fn last_discard_offer(&self, seat: u8) -> Option<u8> {
        // Prefer the live window record; fall back to the engine's own
        // `last_discard` so claim sources stay attributed to the true seat.
        if let Some(by) = self.last_discard.0 {
            return Some(by);
        }
        if let Some((engine_seat, _)) = self.state.last_discard {
            let canonical = self.inv[engine_seat as usize] as u8;
            if canonical != seat {
                return Some(canonical);
            }
        }
        None
    }

    // -- observation capture ----------------------------------------------------

    /// Actor observation for a deciding seat. Runs two checks: bind the
    /// filtered observation with its hash, then independently re-derive and
    /// re-check visibility before returning.
    pub fn observe(&mut self, seat: u8) -> Result<NativeObservation, CoreError> {
        if !self.views.contains_key(&seat) {
            return Err(CoreError(format!("seat {seat} has no recorded decision to observe")));
        }
        let pid = self.ctx.seat_perm[seat as usize];
        let mut hand = self.state.players[pid as usize].hand.clone();
        // Retained per-seat draw (see `drawn_cache`): the live engine blanks
        // `drawn_tile` on discard, but responders still observe their tile.
        let drawn = self.drawn_cache[seat as usize];
        if self.mode == Some(GameMode::Draw)
            && self.decision_seat == Some(seat)
            && let Some(d) = drawn
        {
            hand.retain(|t| *t != d);
        }
        hand.sort_unstable();
        let (ordered, mask) = self.view_for(seat)?;
        let kinds: Vec<ActionKind> = ordered.iter().map(|(_, a)| a.kind).collect();
        let furiten = if self.state.players[pid as usize].riichi_declared {
            "riichi"
        } else if self.state.players[pid as usize].missed_agari_doujun {
            "temporary"
        } else {
            "none"
        };
        // Sequence law: the last visible history event's sequence (0 when history
        // is empty), not the global envelope counter (server-private envelopes
        // advance the counter but never enter any actor's history).
        let history_indices = self.arena.visible_indices_for_hand(seat, self.arena.len());
        let sequence = history_indices
            .last()
            .and_then(|index| self.arena.get(*index))
            .map_or(0, |event| event.sequence);
        let obs = NativeObservation {
            game_id: self.game_id.clone(),
            decision_id: format!("{}:{}", self.game_id, self.arena.len()),
            sequence,
            actor: seat,
            rules_id: self.ctx.rules_id.clone(),
            rules_hash: self.ctx.rules_hash.clone(),
            action_table_hash: self.ctx.action_table_hash.clone(),
            event_schema_hash: self.ctx.event_schema_hash.clone(),
            observation_schema_hash: self.ctx.observation_schema_hash.clone(),
            packet_boundary_hash: self.ctx.packet_boundary_hash.clone(),
            round_index: self.hand_index,
            round_wind: 27 + i64::from(self.state.round_wind),
            hand_number: u32::from(self.state.kyoku_idx) + 1,
            seat_winds: seat_winds_for(self.inv[self.state.oya as usize] as u8),
            honba: u32::from(self.state.honba),
            riichi_sticks: self.state.riichi_sticks,
            dealer: self.inv[self.state.oya as usize] as u8,
            scores: canonical_scores(&self.state, self.inv),
            turn_actor: self.inv[self.state.current_player as usize] as u8,
            phase: self.phase_for(seat),
            live_wall_remaining: self.state.wall.tiles.len().saturating_sub(14) as u32,
            kan_count: self.kan_count,
            ippatsu_active: self.ippatsu,
            actor_furiten: furiten.to_string(),
            actor_can_tsumo: kinds.contains(&ActionKind::Tsumo),
            actor_can_riichi: kinds.contains(&ActionKind::RiichiDiscard),
            pending_declaration_discard: None,
            concealed_hand: hand,
            own_drawn_tile: drawn,
            visible_discards: self.canonical_rivers(),
            visible_melds: self.canonical_melds(),
            riichi_states: self.riichi_states.clone(),
            // Shared event replay (same function the seal uses): dora comes
            // from `dora_revealed` envelopes, never from the wall (which
            // already holds future, unrevealed indicators).
            dora_indicators: crate::rl::observation::revealed_dora(&self.arena, self.arena.len()),
            history_indices,
            legal_mask: mask,
            observation_hash: None,
        };
        obs.validate(&self.arena)?;
        let mut bound = obs;
        bound.bind_hash(&self.arena)?;
        independent_validate(&bound, &self.arena)?;
        Ok(bound)
    }

    // -- apply ---------------------------------------------------------------------

    /// Apply one canonical action at the pending decision.
    pub fn apply(&mut self, action: &CanonicalAction) -> Result<(), CoreError> {
        if self.terminal {
            return Err(CoreError("simulation is terminal; no further actions exist".to_string()));
        }
        // Any pending responder may answer; resolution waits for the complete
        // map. Record each committed answer immediately: omitting a partial
        // window loses that action on restore, and HashMap order is not replay.
        if self.mode == Some(GameMode::Window) {
            if !self.pending.contains(&action.actor) {
                return Err(CoreError(format!(
                    "action actor {} not in pending window {:?}",
                    action.actor, self.pending
                )));
            }
        } else {
            let expected = self.expected_actor()?;
            if action.actor != expected {
                return Err(CoreError(format!(
                    "action actor {} != expected decision actor {expected}",
                    action.actor
                )));
            }
        }
        let seat = action.actor;
        let (ordered, _) = self.view_for(seat)?;
        let stamp = canonical_id(action)
            .map_err(|e| CoreError(format!("action rejected by codec context: {e}")))?;
        if !ordered.iter().any(|(_, a)| a == action) {
            let tile_mark = action.tile.unwrap_or(255);
            return Err(CoreError(format!(
                "action {:?}:{tile_mark} is not in the legal mask slot set (id {stamp})",
                action.kind,
            )));
        }
        let pid = self.ctx.seat_perm[seat as usize];
        let engine_legals = self.trait_legals(pid)?;
        let candidates = check_riichi_candidates(self.state.players[pid as usize].hand.clone());
        let matched = engine_legals
            .iter()
            .find(|eng| engine_matches_canonical(eng, action, &candidates))
            .cloned()
            .ok_or_else(|| {
                CoreError(format!("canonical action {:?} has no engine counterpart", action.kind))
            })?;
        self.stamp = Some(Stamp { id: stamp, actor: seat, kind: action.kind });
        if action.kind == ActionKind::Kakan {
            let added = action
                .tile
                .ok_or_else(|| CoreError("kakan without an added tile".to_string()))?;
            self.kakan_added.insert(pid, added);
            self.window_opened_by_discard = false;
        }
        if self.mode == Some(GameMode::Window) {
            self.buffered.insert(
                seat,
                BufferedDecision { engine: matched, canonical_id: stamp },
            );
            self.applied.push(action.clone());
            self.pending.retain(|s| *s != seat);
            self.views.remove(&seat);
            if !self.pending.is_empty() {
                return Ok(());
            }
            let responder_count = self.buffered.len();
            let mut combined: HashMap<u8, CoreAction> = HashMap::with_capacity(responder_count);
            for seat in 0..4u8 {
                if let Some(buffered) = self.buffered.get(&seat) {
                    combined.insert(self.ctx.seat_perm[seat as usize], buffered.engine.clone());
                }
            }
            if combined.len() != responder_count || combined.is_empty() {
                return Err(CoreError("window resolution without buffered responders".to_string()));
            }
            self.capture_hands(&combined);
            let pre = self.state.mjai_log.len();
            // The engine mutation invalidates all prepared legal plans;
            // leaving them live would make a later record-free drain stale.
            self.views.clear();
            self.state.step(&combined);
            self.check_error()?;
            self.note_step_draws(pre);
            // Resolution translation needs every responder's canonical id;
            // clearing early loses a winning claim when the last seat passed.
            self.flush_window_resolution()?;
            self.buffered.clear();
        } else {
            self.applied.push(action.clone());
            // Riichi may step twice and kakan may retain the deciding seat;
            // neither may leave a pre-step legal plan available for reuse.
            self.views.clear();
            if action.kind == ActionKind::RiichiDiscard {
                let declaration = matched;
                let pre = self.state.mjai_log.len();
                self.capture_hands_single(pid);
                self.state.step(&HashMap::from([(pid, declaration)]));
                self.check_error()?;
                self.note_step_draws(pre);
                let _ = self.consume_new()?;
                let inner = self.trait_legals(pid)?;
                let declared_tile = match action.tile {
                    None => self.state.drawn_tile,
                    Some(t) => Some(t),
                };
                let discard_match = inner
                    .iter()
                    .find(|eng| {
                        eng.action_type == ActionType::Discard && eng.tile == declared_tile
                    })
                    .cloned()
                    .ok_or_else(|| {
                        CoreError("engine lost the declared discard candidate".to_string())
                    })?;
                let pre_inner = self.state.mjai_log.len();
                self.capture_hands_single(pid);
                self.state.step(&HashMap::from([(pid, discard_match)]));
                self.check_error()?;
                self.note_step_draws(pre_inner);
            } else {
                let pre = self.state.mjai_log.len();
                self.capture_hands_single(pid);
                self.state.step(&HashMap::from([(pid, matched)]));
                self.check_error()?;
                self.note_step_draws(pre);
            }
            if action.kind == ActionKind::Kakan {
                self.last_discard = (Some(seat), action.tile);
            } else {
                self.mode = None;
                self.decision_seat = None;
                self.views.remove(&seat);
                self.last_discard = (None, None);
            }
        }
        let _ = self.drain()?;
        // One fully resolved engine decision: draw answers and completed
        // windows converge here after successful drain. Buffered partial
        // answers return early above and failures return Err above, so
        // neither advances the scheduling frontier. The counter counts
        // resolutions only — never staged answers, captures, or rows.
        self.resolved_frontier_count += 1;
        Ok(())
    }

    // -- replay --------------------------------------------------------------------------

    /// Replay the canonical action prefix against identical walls (resume path;
    /// no inference, no resampling, no version change).
    pub fn replay_prefix(&mut self, prefix: &[CanonicalAction]) -> Result<(), CoreError> {
        self.applied.clear();
        for action in prefix {
            let _ = self.drain()?;
            self.apply(action)?;
        }
        Ok(())
    }
}

impl super::NativeGame {
    pub(super) fn check_error(&self) -> Result<(), CoreError> {
        if let Some(err) = &self.state.last_error {
            return Err(CoreError(format!("core rejected step: {err}")));
        }
        Ok(())
    }

    pub(super) fn emit(&mut self, event: TypedEvent) -> Result<(), CoreError> {
        if let Some(staging) = self.staging.as_mut() {
            staging.push(event);
        } else {
            self.arena.admit(event)?;
        }
        Ok(())
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        seq
    }

    pub(super) fn make_envelope(
        &mut self,
        kind: &str,
        actor: Option<u8>,
        visibility: Visibility,
        visible_to: Vec<u8>,
        payload: TypedPayload,
        public_delta: Vec<StateDelta>,
    ) -> Result<TypedEvent, CoreError> {
        let seq = self.next_seq();
        self.make_envelope_with_seq(kind, actor, visibility, visible_to, payload, public_delta, seq)
    }

    fn make_envelope_with_seq(
        &mut self,
        kind: &str,
        actor: Option<u8>,
        visibility: Visibility,
        visible_to: Vec<u8>,
        payload: TypedPayload,
        public_delta: Vec<StateDelta>,
        seq: u64,
    ) -> Result<TypedEvent, CoreError> {
        let event = TypedEvent {
            game_id: self.game_id.clone(),
            sequence: seq,
            kind: kind.to_string(),
            actor,
            visibility,
            visible_to,
            tile: payload.tile,
            action_id: payload.action_id,
            source_seat: payload.source_seat,
            consumed_tiles: payload.consumed_tiles,
            offered_action_ids: payload.offered_action_ids,
            accepted_action_ids: payload.accepted_action_ids,
            round_index: payload.round_index,
            scores: payload.scores,
            reason: payload.reason,
            public_delta,
            rules_hash: self.ctx.rules_hash.clone(),
            schema_hash: self.ctx.event_schema_hash.clone(),
        };
        event.check_visibility()?;
        event.check_kind_shape()?;
        Ok(event)
    }

    fn arena_last_kind(&self) -> Option<String> {
        if self.arena.is_empty() {
            return None;
        }
        self.arena.get(self.arena.len() - 1).map(|e| e.kind.clone())
    }

    fn expected_actor(&self) -> Result<u8, CoreError> {
        match &self.mode {
            Some(GameMode::Draw) => {
                self.decision_seat.ok_or_else(|| CoreError("no decision is pending".to_string()))
            }
            Some(GameMode::Window) => self
                .pending
                .iter()
                .min()
                .copied()
                .ok_or_else(|| CoreError("no decision is pending".to_string())),
            None => Err(CoreError("no decision is pending".to_string())),
        }
    }

    fn capture_hands(&mut self, actions: &HashMap<u8, CoreAction>) {
        for pid in actions.keys() {
            self.pre_step_hands.insert(*pid, self.state.players[*pid as usize].hand.clone());
        }
    }

    fn capture_hands_single(&mut self, pid: u8) {
        self.pre_step_hands.insert(pid, self.state.players[pid as usize].hand.clone());
    }

    pub(super) fn note_step_draws(&mut self, previous_log_length: usize) {
        let has_tsumo = self.state.mjai_log[previous_log_length..].iter().any(|s| {
            serde_json::from_str::<serde_json::Value>(s)
                .ok()
                .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(|t| t == "tsumo"))
                .unwrap_or(false)
        });
        if has_tsumo
            && let Some(drawn) = self.state.drawn_tile
        {
            self.draw_queue.push_back(drawn);
        }
    }

    pub(super) fn exact_draw(&mut self) -> Result<u8, CoreError> {
        self.draw_queue
            .pop_front()
            .ok_or_else(|| CoreError("tsumo event without a captured draw id".to_string()))
    }

    fn consume_new(&mut self) -> Result<BatchOutcome, CoreError> {
        if self.cursor >= self.state.mjai_log.len() {
            return Ok(BatchOutcome::Continue);
        }
        let records: Vec<serde_json::Value> = self.state.mjai_log[self.cursor..]
            .iter()
            .map(|s| serde_json::from_str(s).map_err(|e| CoreError(format!("mjai parse: {e}"))))
            .collect::<Result<_, _>>()?;
        let mut consumed = 0usize;
        let outcome = self.translate_batch(&records, &mut consumed)?;
        self.cursor += consumed;
        Ok(outcome)
    }

    fn flush_window_resolution(&mut self) -> Result<(), CoreError> {
        // Oracle sequencing: staged envelopes are built with creation
        // sequences, then EVERYTHING (call_resolved plus staged, in emit
        // order) is refitted with fresh sequences. The creation sequences
        // burn counter values that never appear in the arena; skipping the
        // refit would lag every later history sequence by the staged count.
        self.staging = Some(vec![]);
        let outcome = self.consume_new()?;
        let staged = self.staging.take().unwrap_or_default();
        let accepted = staged
            .iter()
            .find(|e| matches!(e.kind.as_str(), "chi" | "pon" | "daiminkan" | "ron") && e.action_id.is_some())
            .and_then(|e| e.action_id);
        let offered = if accepted.is_some() { self.window_offered_ids.clone() } else { vec![] };
        if self.window_opened_by_discard {
            // Creation sequence burns (discarded by the refit below).
            let burned = self.next_seq();
            let mut resolution = self.make_envelope_with_seq(
                "call_resolved",
                None,
                Visibility::ServerPrivate,
                vec![],
                TypedPayload {
                    offered_action_ids: offered,
                    accepted_action_ids: accepted.into_iter().collect(),
                    ..TypedPayload::empty()
                },
                vec![],
                burned,
            )?;
            // Refit in emit order: resolution heads the group.
            resolution.sequence = self.next_seq();
            self.arena.admit(resolution)?;
        }
        for mut event in staged {
            event.sequence = self.next_seq();
            self.arena.admit(event)?;
        }
        self.window_offered_ids.clear();
        match outcome {
            BatchOutcome::Carry(carry) => {
                self.reopen(carry)?;
            }
            BatchOutcome::Terminal | BatchOutcome::Continue => {}
        }
        Ok(())
    }

    fn canonical_rivers(&self) -> [Vec<u8>; 4] {
        // Engine rivers can advance ahead of translated envelopes; only
        // published current-hand discards belong in actor-visible observations.
        let mut rivers = std::array::from_fn(|_| Vec::new());
        let prefix = self.arena.len();
        for index in self.arena.hand_window_start(prefix)..prefix {
            if let Some(event) = self.arena.get(index)
                && event.kind == "discard"
                && let (Some(actor), Some(tile)) = (event.actor, event.tile)
            {
                rivers[actor as usize].push(tile);
            }
        }
        rivers
    }

    fn canonical_melds(&self) -> [Vec<NativeMeld>; 4] {
        std::array::from_fn(|seat| self.engine_melds(seat as u8))
    }

    fn engine_melds(&self, seat: u8) -> Vec<NativeMeld> {
        let pid = self.ctx.seat_perm[seat as usize] as usize;
        self.state.players[pid]
            .melds
            .iter()
            .map(|m| {
                let kind = match m.meld_type {
                    MeldType::Chi => "chi",
                    MeldType::Pon => "pon",
                    MeldType::Daiminkan => "daiminkan",
                    MeldType::Ankan => "ankan",
                    MeldType::Kakan => "kakan",
                };
                let source = if matches!(m.meld_type, MeldType::Ankan | MeldType::Kakan) {
                    None
                } else if m.from_who >= 0 {
                    Some(self.inv[m.from_who as usize] as u8)
                } else {
                    self.last_discard.0
                };
                NativeMeld {
                    kind: kind.to_string(),
                    owner: seat,
                    source,
                    called_tile: m.called_tile,
                    tiles: m.tiles.clone(),
                }
            })
            .collect()
    }
}
