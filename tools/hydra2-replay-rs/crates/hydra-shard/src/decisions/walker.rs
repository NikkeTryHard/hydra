use super::table::ActionTable;
use crate::engine::{EngineDesync, ResponderView, SeatView, window_open};
use crate::mjai_event::{SKIP_TYPES, TRANSPARENT_KINDS, is_claim_kind};
use crate::parity::{ChosenAction, ReplayRow};
use crate::stream::{KyokuTrack, ParsedGame};
use crate::tile::{copies_of_string, mjai_string_of, norm_pai};
use std::collections::HashMap;

/// Relative seat offset: (source - actor) mod 4 with 3 folded to -1.
pub fn source_offset(source: u8, actor: u8) -> i8 {
    match (source + 4 - actor) % 4 {
        3 => -1,
        delta => delta as i8,
    }
}

// ---------------------------------------------------------------------------
// Walker.
// ---------------------------------------------------------------------------

/// Whole-game quarantine with a stable reason code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkReject {
    pub game_id: String,
    pub code: String,
    pub detail: String,
}

impl WalkReject {
    /// Classified rejection: `code` is the closed taxonomy
    /// (`bare-dora | double-ron | unknown-event | tile-conservation |
    /// turn-order | claim-no-offer | draw-past-wall | kyushu-ambiguous |
    /// unmapped-ryukyoku-reason | engine-desync`; `framing` /
    /// `wall-bearing` live on `GameReject`, `action-id-unresolved` is
    /// row-level). `step` keeps the oracle's event-step vocabulary in
    /// `detail` so downstream `_quarantine_class` normalization keeps
    /// working.
    pub(crate) fn new(game_id: &str, kyoku: i64, code: &str, step: &str, why: &str) -> Self {
        WalkReject {
            game_id: game_id.to_string(),
            code: code.to_string(),
            detail: format!("sim replay desync game {game_id:?} kyoku {kyoku} {step}: {why}"),
        }
    }
}

pub(crate) fn is_end(kind: &str) -> bool {
    matches!(kind, "end_game" | "endGame" | "game_end" | "end")
}

pub(crate) fn is_start(kind: &str) -> bool {
    matches!(kind, "start_game" | "startGame" | "game_start" | "start")
}

pub(crate) fn in_skip(kind: &str) -> bool {
    SKIP_TYPES.contains(&kind)
}

pub(crate) fn is_transparent(kind: &str) -> bool {
    TRANSPARENT_KINDS.contains(&kind)
}

pub(crate) struct Walker<'a> {
    pub(crate) game: &'a ParsedGame,
    pub(crate) table: &'a ActionTable,
    pub(crate) seq: u32,
    pub(crate) round_idx: i64,
    pub(crate) hand_index: i64,
    pub(crate) kyoku_ordinal: i64,
    pub(crate) track: Option<KyokuTrack>,
    pub(crate) dealer: u8,
    pub(crate) histories: [Vec<String>; 4],
    pub(crate) last_kind: Option<String>,
    pub(crate) last_discard: Option<(u8, u8)>,
    pub(crate) opened_by_discard: bool,
    pub(crate) decided: bool,
    pub(crate) terminal: bool,
    pub(crate) rows: Vec<ReplayRow>,
    /// Tracked scores per seat (kyoku headers, -1000 per riichi
    /// declaration, plus hora/ryukyoku deltas). Feeds the engine's riichi
    /// availability gate (score >= 1000), mirroring `tracked_scores`.
    pub(crate) scores: [i32; 4],
    /// Selected backend: V1 (pinned drained-oracle, bug-compatible) or V2
    /// (single-pass live-engine semantics). V1 is the frozen default.
    pub(crate) version: crate::engine::EngineVersion,
    /// V2 one-shot immediate filter: (claimer, just-claimed called take,
    /// just-claimed meld takes, chi takes for the neighborhood gate, empty
    /// unless chi). Set on chi/pon claims, cleared on the next discard or
    /// kyoku start.
    pub(crate) kuikae_gate: Option<(u8, u8, Vec<u8>, Vec<u8>)>,
    /// Whether each seat has made a decision this kyoku (any emitted row).
    /// Feeds V2 first-draw detection for the kyushu offer.
    pub(crate) acted: [bool; 4],
    /// Claim/kan count this kyoku (any seat). Feeds V2 kyushu (no calls).
    pub(crate) claims: u32,
    /// S7 walled binding: `Some` real wall digest when the framed game
    /// carries a 136-tile wall (`replay-{game_id}` schedule), `None` on the
    /// wall-less path (SIM mark). Minted once at walk start via
    /// `wall_schedule_digest`; every emitted row clones it so the
    /// derivation retires the SIM mark for walled rows only.
    pub(crate) wall_digest: Option<String>,
}

/// S7 port of the `ReplayExpander` mode machine
/// (`replay_expand.ReplayExpander._step_decision`: `expected-actor` /
/// `window-pending` / `draw-mode` / `terminal` against engine answers).
///
/// The engine owns legality; the log owns order. `mode` is derived from
/// engine answers on ledger state (window responders via `window_open` /
/// per-seat offer predicates, draw expectation via the tracked drawer),
/// never from ledger guesses alone:
/// - `Terminal`: the kyoku was decided (hora/ryukyoku) — no decision is
///   pending, so any further row event fails closed (post-terminal rows);
/// - `Window { pending }`: a discard offer is live with a non-empty sorted
///   responder set — the next significant event must be a matching claim on
///   that discard (earlier pending seats pass mechanically, in order) or the
///   window passes with no row;
/// - `Draw { expected }`: the tracked drawer owns the next draw decision —
///   claims outside any window fail closed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkerMode {
    /// No live kyoku yet (before the first `start_kyoku`).
    Idle,
    /// Kyoku decided: no decision pending until the next `start_kyoku`.
    Terminal,
    /// Claim window live on `discarder`'s `tile` with sorted `pending`.
    Window {
        discarder: u8,
        tile: u8,
        pending: Vec<u8>,
    },
    /// Draw decision owned by `expected`.
    Draw { expected: u8 },
}

impl<'a> Walker<'a> {
    pub(crate) fn fail(&self, code: &str, step: &str, why: &str) -> WalkReject {
        WalkReject::new(&self.game.game_id, self.kyoku_ordinal, code, step, why)
    }

    pub(crate) fn track(&self, step: &str) -> Result<&KyokuTrack, WalkReject> {
        self.track
            .as_ref()
            .ok_or_else(|| self.fail("turn-order", step, "no live kyoku"))
    }

    pub(crate) fn track_mut(&mut self, step: &str) -> Result<&mut KyokuTrack, WalkReject> {
        let ordinal = self.kyoku_ordinal;
        self.track.as_mut().ok_or_else(|| {
            WalkReject::new(
                &self.game.game_id,
                ordinal,
                "turn-order",
                step,
                "no live kyoku",
            )
        })
    }

    pub(crate) fn push_public(&mut self, kind: &str) {
        for seat in 0..4 {
            self.histories[seat].push(kind.to_string());
        }
        self.last_kind = Some(kind.to_string());
    }

    pub(crate) fn live_wall(&self) -> u32 {
        let draws = self.track.as_ref().map(|t| t.draws).unwrap_or(0);
        crate::stream::LIVE_WALL_BASE.saturating_sub(draws)
    }

    /// Reported (string-canonical, pool-first) concealed multiset for `seat`.
    pub(crate) fn reported_hand(&self, seat: u8) -> Result<Vec<u8>, WalkReject> {
        let track = self.track("row")?;
        // Collapse every true copy to its string-canonical id, then expand
        // per-string pools in hand order (mirrors `_distinct_copies`).
        let mut per_string: HashMap<String, usize> = HashMap::new();
        let mut out = Vec::with_capacity(track.hands[seat as usize].len());
        // First pass: count occurrences per rendered string in hand order.
        let rendered: Vec<String> = track.hands[seat as usize]
            .iter()
            .map(|t| {
                mjai_string_of(*t)
                    .map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))
            })
            .collect::<Result<_, _>>()?;
        // Assign pool copies per occurrence (order-independent multiset).
        let mut pools: HashMap<String, Vec<u8>> = HashMap::new();
        for pai in &rendered {
            let entry = pools
                .entry(pai.clone())
                .or_insert_with(|| copies_of_string(pai).unwrap_or_default());
            let used = per_string.get(pai).copied().unwrap_or(0);
            if used < entry.len() {
                out.push(entry[used]);
            } else {
                // Overused strings keep the verbatim collapsed id.
                out.push(
                    copies_of_string(pai)
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0),
                );
            }
            per_string.insert(pai.clone(), used + 1);
        }
        out.sort_unstable();
        Ok(out)
    }

    pub(crate) fn reported_drawn(&self, seat: u8) -> Result<Option<u8>, WalkReject> {
        let track = self.track("row")?;
        match track.drawn[seat as usize] {
            None => Ok(None),
            Some(d) => {
                let pai = mjai_string_of(d)
                    .map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))?;
                Ok(copies_of_string(&pai)
                    .map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))?
                    .first()
                    .copied())
            }
        }
    }

    pub(crate) fn concealed_for_row(&self, seat: u8) -> Result<Vec<u8>, WalkReject> {
        let mut hand = self.reported_hand(seat)?;
        // Mirror `_concealed_for_build` on collapsed takes: the drained
        // step hand is pool-prefix, so the collapsed drawn copy is always
        // present exactly when the seat drew this turn. Stale draws persist
        // in the display but never subtract from the concealed hand.
        let live = self.track("row")?.drawn_live[seat as usize].is_some();
        if live {
            if let Some(drawn) = self.reported_drawn(seat)? {
                if let Some(pos) = hand.iter().position(|t| *t == drawn) {
                    hand.remove(pos);
                }
            }
        }
        Ok(hand)
    }
    /// keeping the oracle's `game + kyoku + step` detail vocabulary.
    pub(crate) fn desync(&self, err: EngineDesync) -> WalkReject {
        WalkReject::new(
            &err.game_id,
            err.kyoku,
            "engine-desync",
            &err.step,
            &err.why,
        )
    }

    /// Seat-filtered engine view for a draw decision: the collapsed hand
    /// takes (pool-first per string occurrence, exactly like the drained
    /// `_distinct_copies` expansion) plus the collapsed live draw and the
    /// true drawn take for riichi-forced answers.
    pub(crate) fn seat_view(
        &self,
        seat: u8,
        step: &'static str,
    ) -> Result<SeatView<'_>, WalkReject> {
        let hand = self.reported_hand(seat)?;
        let track = self.track(step)?;
        let live_take = track.drawn_live[seat as usize];
        let live = live_take.is_some();
        // Collapsed live draw (pool-first of the drawn string); the true
        // take rides alongside for riichi-forced answers.
        let drawn = if live {
            self.reported_drawn(seat)?
        } else {
            None
        };
        SeatView::new(
            &self.game.game_id,
            self.kyoku_ordinal,
            step,
            seat,
            hand,
            drawn,
            live_take,
            &track.melds[seat as usize],
            track.riichi_declared[seat as usize],
            self.scores[seat as usize],
            self.live_wall(),
        )
        .map_err(|err| self.desync(err))
    }
    pub(crate) fn draw_mask_ids(
        &self,
        seat: u8,
        offer: &crate::engine::DrawOffer,
        chosen: &ChosenAction,
        log_win: bool,
    ) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        for tile in &offer.discards {
            if let Some(id) =
                self.table
                    .lookup("discard", Some(*tile), None, &[], None, false, false)
            {
                mask.insert(id);
            }
        }
        if let Some(tile) = offer.tsumogiri {
            if let Some(id) =
                self.table
                    .lookup("tsumogiri", Some(tile), None, &[], None, false, false)
            {
                mask.insert(id);
            }
        }
        for tile in &offer.riichi {
            if let Some(id) =
                self.table
                    .lookup("riichi_discard", Some(*tile), None, &[], None, true, false)
            {
                mask.insert(id);
            }
        }
        for quad in &offer.ankan {
            if let Some(id) = self
                .table
                .lookup("ankan", None, None, quad, None, false, false)
            {
                mask.insert(id);
            }
        }
        for tile in &offer.kakan {
            if let Some(id) = self
                .table
                .lookup("kakan", Some(*tile), None, &[], None, false, true)
            {
                mask.insert(id);
            }
        }
        if log_win {
            if let ChosenAction::Tsumo { tile } = chosen {
                if let Some(id) =
                    self.table
                        .lookup("tsumo", Some(*tile), None, &[], None, false, false)
                {
                    mask.insert(id);
                }
            }
        }
        if let Some(id) = chosen.lookup(self.table, seat) {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Claim-response mask: exactly `{pass, chosen}`. The drained engine
    /// offers no chi enumeration beyond the stashed (logged) claim, and
    /// ron never rides a non-ron claim row (both corpus-proven).
    pub(crate) fn claim_mask_ids(&self, seat: u8, chosen: &ChosenAction) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        if let Some(id) = self
            .table
            .lookup("pass", None, None, &[], None, false, false)
        {
            mask.insert(id);
        }
        if let Some(id) = chosen.lookup(self.table, seat) {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Ron-response mask: exactly `{pass, ron}` (pinned on every ron row).
    pub(crate) fn ron_mask_ids(&self, seat: u8, tile: u8, discarder: u8) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        if let Some(id) = self
            .table
            .lookup("pass", None, None, &[], None, false, false)
        {
            mask.insert(id);
        }
        let delta = (discarder + 4 - seat) % 4;
        let offset = if delta == 3 { -1 } else { delta as i8 };
        if let Some(id) =
            self.table
                .lookup("ron", Some(tile), None, &[], Some(offset), false, false)
        {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "row capture takes explicit context args; grouping churns cold walker"
    )]
    pub(crate) fn capture_row(
        &mut self,
        seat: u8,
        phase: &str,
        turn_actor: u8,
        chosen: ChosenAction,
        mask: &[u32],
    ) -> Result<(), WalkReject> {
        let seq = self.seq;
        self.seq += 1;
        self.acted[seat as usize] = true;
        let decision_id = format!("{}:d{seq:04}", self.game.game_id);
        let round_id = format!("{}:h{:02}", self.game.game_id, self.round_idx.max(0));
        let concealed = self.concealed_for_row(seat)?;
        let concealed_hand: Vec<String> = concealed
            .iter()
            .map(|t| {
                mjai_string_of(*t)
                    .map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))
            })
            .collect::<Result<_, _>>()?;
        let drawn_id = self.reported_drawn(seat)?;
        let drawn_tile = drawn_id
            .map(|d| {
                mjai_string_of(d).map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))
            })
            .transpose()?;
        let track = self.track("row")?;
        let dora_indicators: Vec<Option<String>> =
            (0..5).map(|i| track.dora.get(i).cloned()).collect();
        let (tile, called, consumed) = match &chosen {
            ChosenAction::Discard { tile }
            | ChosenAction::Tsumogiri { tile }
            | ChosenAction::RiichiDiscard { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Chi { called, consumed } => (None, Some(*called), consumed.clone()),
            ChosenAction::Pon {
                called, consumed, ..
            } => (None, Some(*called), consumed.clone()),
            ChosenAction::Daiminkan {
                called, consumed, ..
            } => (None, Some(*called), consumed.clone()),
            ChosenAction::Ankan { consumed } => (None, None, consumed.clone()),
            ChosenAction::Kakan { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Ron { tile, .. } => (Some(*tile), None, vec![]),
            ChosenAction::Tsumo { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Pass => (None, None, vec![]),
        };
        let kind = chosen.kind_str().to_string();
        let declares_riichi = matches!(chosen, ChosenAction::RiichiDiscard { .. });
        let offset = match &chosen {
            ChosenAction::Chi { .. } => Some(-1),
            ChosenAction::Pon { source, .. }
            | ChosenAction::Daiminkan { source, .. }
            | ChosenAction::Ron { source, .. } => Some(source_offset(*source, seat)),
            _ => None,
        };
        let meldref = matches!(chosen, ChosenAction::Kakan { .. });
        let chosen_action_id = self.table.lookup(
            &kind,
            tile,
            called,
            &consumed,
            offset,
            declares_riichi,
            meldref,
        );
        let chosen_unresolved = if chosen_action_id.is_none() {
            Some(format!("action-id-unresolved:{kind}"))
        } else {
            None
        };
        let mut legal_mask: Vec<u32> = mask.to_vec();
        legal_mask.sort_unstable();
        legal_mask.dedup();
        self.rows.push(ReplayRow {
            game_id: self.game.game_id.clone(),
            round_id,
            decision_id,
            seat,
            phase: phase.to_string(),
            turn_actor,
            chosen,
            chosen_action_id,
            chosen_unresolved,
            legal_mask,
            history_kinds: self.histories[seat as usize].clone(),
            concealed_hand,
            drawn_tile,
            dora_indicators,
            wall_remaining: self.live_wall(),
            wall_digest: self.wall_digest.clone(),
        });
        Ok(())
    }

    // -- engine window evaluation ------------------------------------------

    /// Responder snapshots for one window query (ledger-derived inputs to
    /// the engine's window rule; only booleans leave the engine, so the
    /// seat filter holds by construction).
    pub(crate) fn responders(&self, step: &str) -> Result<Vec<ResponderView<'_>>, WalkReject> {
        let track = self.track(step)?;
        let mut out = Vec::with_capacity(4);
        for seat in 0..4u8 {
            out.push(ResponderView {
                seat,
                riichi: track.riichi_declared[seat as usize],
                hand: &track.hands[seat as usize],
                river: &track.rivers[seat as usize],
                open: track.melds[seat as usize].len(),
            });
        }
        Ok(out)
    }

    /// Next window claim on `discarder`'s tile, if the log shows one
    /// (mirrors `_peek_window_claim`: scans past transparent events; a
    /// draw decision, another offer, a terminal, or any boundary ends the
    /// window with no claim).
    pub(crate) fn peek_window_claim(
        &self,
        from_idx: usize,
        discarder: u8,
    ) -> Option<&crate::mjai_event::MjaiEvent> {
        let mut idx = from_idx;
        while idx < self.game.events.len() {
            let event = &self.game.events[idx];
            let kind = event.type_.as_str();
            if is_transparent(kind) {
                idx += 1;
                continue;
            }
            if is_claim_kind(kind) {
                if event.target_seat().ok() == Some(discarder) {
                    return Some(event);
                }
                return None;
            }
            if kind == "hora" && !event.tsumo {
                if event.actor == event.target {
                    return None;
                }
                if event.target_seat().ok() == Some(discarder) {
                    return Some(event);
                }
                return None;
            }
            return None;
        }
        None
    }

    /// Whether the log takes a ron on `discarder`'s tile at the window
    /// opening at `from_idx`. A logged ron proves the engine's window was
    /// open (the engine must have offered it): sound window-open fact for
    /// the responder-ron cases the responder rules cannot see.
    pub(crate) fn peek_window_ron(&self, from_idx: usize, discarder: u8) -> bool {
        match self.peek_window_claim(from_idx, discarder) {
            Some(event) if event.type_.as_str() == "hora" => true,
            _ => false,
        }
    }

    /// Emit the public `call_window` envelope when the engine opens a
    /// window after a discard (responder rules) or the log proves one (a
    /// ron taken on this discard). Kan windows never emit envelopes: the
    /// adapter grammar routes kakan straight to ron, and
    /// `opened_by_discard` is false there on both sides.
    pub(crate) fn open_window(&mut self, discarder: u8, tile: u8, chankan: bool, from_idx: usize) {
        let tile_norm = mjai_string_of(tile)
            .map(|p| norm_pai(&p).to_string())
            .unwrap_or_default();
        let engine_open = self
            .responders("window")
            .map(|views| window_open(discarder, &tile_norm, &views, chankan))
            .unwrap_or(false);
        let log_ron = self.peek_window_ron(from_idx, discarder);
        if (engine_open || log_ron)
            && self.opened_by_discard
            && self.last_kind.as_deref() == Some("discard")
        {
            self.push_public("call_window");
        }
    }

    // -- S7 mode machine (ReplayExpander port) ------------------------------
    //
    // Reference: `replay_expand.ReplayExpander._step_decision` /
    // `_step_window` / `_step_draw` plus the adapter's
    // `_expected_actor_or_none` / `_detect_decision` (draw vs window vs
    // terminal). The engine owns legality (responder predicates, drawer
    // expectation); the log owns order (claim matching, reach-collapse
    // indices, terminal checks). Both walled and wall-less rows flow through
    // this gate so the fork stays unified.

    /// Sorted pending responders for a live discard offer.
    ///
    /// Mirrors the adapter's `WaitResponse` responder set (seats whose engine
    /// observation would list a legal response): pon counts, kamicha chi
    /// availability, or a furiten-clean riichi shape — plus the log-ron fact
    /// (a logged ron proves the engine offered it, sound since the engine
    /// must have). Kan (`chankan`) windows suppress pon/chi exactly like the
    /// oracle, which additionally never emits post-kakan envelopes.
    pub(crate) fn window_pending(&self, discarder: u8, tile: u8, chankan: bool) -> Vec<u8> {
        let Ok(track) = self.track("window") else {
            return Vec::new();
        };
        let tile_norm = mjai_string_of(tile)
            .map(|p| norm_pai(&p).to_string())
            .unwrap_or_default();
        let mut pending = Vec::new();
        for seat in 0..4u8 {
            if seat == discarder {
                continue;
            }
            let hand = &track.hands[seat as usize];
            let river = &track.rivers[seat as usize];
            let open = track.melds[seat as usize].len();
            let riichi = track.riichi_declared[seat as usize];
            // Pon: two held copies of the called type (aka folds).
            let mut pon = false;
            let mut chi = false;
            if !chankan {
                let mut held = 0u8;
                for t in hand.iter() {
                    if let Ok(p) = mjai_string_of(*t) {
                        if norm_pai(&p) == tile_norm {
                            held += 1;
                        }
                    }
                }
                if held >= 2 {
                    pon = true;
                }
                // Chi: kamicha only, via the engine's chi predicate.
                if seat == (discarder + 1) % 4 && !tile_norm.is_empty() {
                    // Honors never chi; the predicate folds aka.
                    chi = crate::engine::chi_offered(hand, &tile_norm);
                }
            }
            // Ron shape: the discard completes a winning shape for a
            // closed-or-open hand. Responder-ron by non-riichi seats is
            // invisible to the responder rules, so the mode keeps the
            // responder in pending only when the shape holds AND (riichi
            // declared OR the log takes the ron here). The log fact is
            // resolved by the caller via `peek_window_ron`; here we keep the
            // shape gate so the set stays engine-derived.
            let mut completed = hand.clone();
            completed.push(tile);
            let shape = crate::engine::win_shape_14(&completed, open);
            // Furiten-clean riichi shape is the visible ron offer; the log
            // fact covers non-riichi ron separately at the window gate.
            let ron_offered = shape
                && riichi
                && !river.iter().any(|r| {
                    mjai_string_of(*r)
                        .map(|p| norm_pai(&p).to_string())
                        .unwrap_or_default()
                        == tile_norm
                });
            if pon || chi || ron_offered {
                pending.push(seat);
            }
        }
        // Log-ron soundness: a logged ron on this discard proves the window
        // was open for its winner even when the responder rules cannot see
        // non-riichi ron. The caller ORs this at the gate (`open_window`
        // already does); `pending` itself stays engine-derived so the mode
        // never guesses from the log alone.
        pending.sort_unstable();
        pending
    }

    /// Current mode against engine answers (never ledger guesses alone).
    ///
    /// - `Terminal` once the kyoku is decided (post-terminal rows fail
    ///   closed) or before the first kyoku (`Idle` maps to a turn-order
    ///   failure at use sites);
    /// - `Window` while a discard offer is live with non-empty pending;
    /// - `Draw` otherwise, owned by the tracked drawer.
    pub(crate) fn walker_mode(&self) -> WalkerMode {
        if self.track.is_none() || self.kyoku_ordinal < 0 {
            return WalkerMode::Idle;
        }
        if self.decided {
            return WalkerMode::Terminal;
        }
        if let Some((discarder, tile)) = self.last_discard {
            if self.opened_by_discard {
                let pending = self.window_pending(discarder, tile, false);
                // Kan windows (opened_by_discard false) never surface here;
                // chankan ron still resolves via the ron path's shape gate.
                if !pending.is_empty() {
                    return WalkerMode::Window {
                        discarder,
                        tile,
                        pending,
                    };
                }
            }
        }
        let expected = self
            .track
            .as_ref()
            .and_then(|t| t.drawer.or(t.exp_drawer).or(Some(self.dealer)))
            .unwrap_or(self.dealer);
        WalkerMode::Draw { expected }
    }

    /// Expected decision actor, if any (`None` on terminal/idle — mirrors
    /// `sim._expected_actor_or_none`: terminal => None, window => min
    /// pending, draw => drawer).
    #[allow(
        dead_code,
        reason = "mirrors sim expected-actor for walled gate; wall-less keeps frozen checks"
    )]
    pub(crate) fn expected_actor_or_none(&self) -> Option<u8> {
        match self.walker_mode() {
            WalkerMode::Idle | WalkerMode::Terminal => None,
            WalkerMode::Window { pending, .. } => pending.into_iter().min(),
            WalkerMode::Draw { expected } => Some(expected),
        }
    }

    /// Assert the log event at `idx` may act in the current mode (fail
    /// closed like `_step_decision`): post-terminal rows, claims outside any
    /// window, and draw-actor mismatches are named desyncs, never drops.
    /// Reach-collapse callers pass the declaration index for error detail.
    /// Walled-only gate (wall-less keeps its frozen turn-order checks so
    /// PG-S6 parity holds bit-for-bit); both paths share the vocabulary and
    /// the `walker_mode` construction.
    pub(crate) fn check_mode_for(
        &self,
        kind: &str,
        actor: u8,
        idx: usize,
    ) -> Result<(), WalkReject> {
        // Wall-less keeps frozen behavior: no extra gate.
        if self.wall_digest.is_none() {
            return Ok(());
        }
        let is_ron = kind == "hora" && self.game.events.get(idx).map(|e| !e.tsumo).unwrap_or(false);
        match self.walker_mode() {
            WalkerMode::Terminal => Err(self.fail(
                "turn-order",
                kind,
                &format!("log carries row-events past kyoku terminal at index {idx}"),
            )),
            WalkerMode::Window {
                discarder,
                pending,
                tile,
                ..
            } => {
                if crate::mjai_event::is_claim_kind(kind) {
                    // Pass placement: earlier pending seats pass mechanically
                    // in sorted order ahead of the claim (no rows, never
                    // synthesized) — the claim must sit inside the pending
                    // set, exactly like `_step_window`.
                    if !pending.contains(&actor) {
                        return Err(self.fail(
                            "claim-no-offer",
                            kind,
                            &format!("logged {kind} by seat {actor} outside the claim window [{discarder:?}->{pending:?}]"),
                        ));
                    }
                } else if is_ron {
                    // Log-ron soundness: the responder rules cannot see
                    // non-riichi ron, so a logged ron with a winning shape on
                    // the offered tile proves the window (sound: the engine
                    // must have offered it). Riichi ron must sit in pending;
                    // non-riichi ron passes on shape alone.
                    if pending.contains(&actor) {
                    } else {
                        let track = self.track("hora")?;
                        let mut completed = track.hands[actor as usize].clone();
                        completed.push(tile);
                        if !crate::engine::win_shape_14(
                            &completed,
                            track.melds[actor as usize].len(),
                        ) {
                            return Err(self.fail(
                                "claim-no-offer",
                                kind,
                                &format!("logged {kind} by seat {actor} outside the claim window [{discarder:?}->{pending:?}]"),
                            ));
                        }
                    }
                }
                // Draws inside a window are the mechanical-pass path: the
                // window must have no logged claim (checked by the caller via
                // `peek_window_claim`); the mode itself never blocks the pass.
                Ok(())
            }
            WalkerMode::Draw { expected: _ } => {
                if crate::mjai_event::is_claim_kind(kind) {
                    return Err(self.fail(
                        "claim-no-offer",
                        kind,
                        &format!("logged {kind} outside any claim window at index {idx}"),
                    ));
                }
                if is_ron {
                    // Chankan: kakan sets `last_discard` with
                    // `opened_by_discard == false` (no envelope, grammar
                    // routes kakan straight to ron). A ron on that tile with
                    // a winning shape is the kan-response path, not a
                    // claim-outside-window failure.
                    if let Some((discarder, tile)) = self.last_discard {
                        let event = self.game.events.get(idx);
                        let target_ok = event
                            .and_then(|e| e.target_seat().ok())
                            .map(|t| t == discarder)
                            .unwrap_or(false);
                        if target_ok {
                            if let Ok(track) = self.track("hora") {
                                let mut completed = track.hands[actor as usize].clone();
                                completed.push(tile);
                                if crate::engine::win_shape_14(
                                    &completed,
                                    track.melds[actor as usize].len(),
                                ) {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    return Err(self.fail(
                        "claim-no-offer",
                        kind,
                        &format!("logged ron outside any claim window at index {idx}"),
                    ));
                }
                Ok(())
            }
            WalkerMode::Idle => {
                Err(self.fail("turn-order", kind, "decision before the first start_kyoku"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Pre-scan: bare kan-dora markers and double ron quarantine whole games,
/// exactly like `replay_game` (fail closed before any row is built).
pub(crate) fn prescan(game: &ParsedGame) -> Result<(), WalkReject> {
    let mut kyoku_idx: i64 = -1;
    let mut pending_ron: Option<(u8, u8)> = None;
    for event in &game.events {
        let kind = event.type_.as_str();
        if kind == "start_kyoku" {
            kyoku_idx += 1;
            pending_ron = None;
            continue;
        }
        if kind == "dora" {
            match &event.dora_marker {
                Some(m) if !m.is_empty() => {}
                _ => {
                    return Err(WalkReject::new(
                        &game.game_id,
                        kyoku_idx,
                        "bare-dora",
                        "dora",
                        "kan-dora indicator unrecoverable: dora event without a dora_marker",
                    ));
                }
            }
            continue;
        }
        if kind == "hora" {
            let actor = seat_value(&event.actor);
            let target = seat_value(&event.target);
            let tsumo_flag = event.tsumo;
            if !tsumo_flag {
                if let (Some(a), Some(t)) = (actor, target) {
                    if a != t {
                        if let Some((prev_a, prev_t)) = pending_ron {
                            if prev_t == t && prev_a != a {
                                return Err(WalkReject::new(
                                    &game.game_id,
                                    kyoku_idx,
                                    "double-ron",
                                    "hora",
                                    "double ron on one discard is quarantined (single-winner pipeline)",
                                ));
                            }
                        }
                        pending_ron = Some((a, t));
                        continue;
                    }
                }
            }
            pending_ron = None;
            continue;
        }
        if is_transparent(kind) {
            continue;
        }
        pending_ron = None;
    }
    Ok(())
}

fn seat_value(value: &Option<serde_json::Value>) -> Option<u8> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            let seat = n.as_u64()?;
            if seat <= 3 { Some(seat as u8) } else { None }
        }
        _ => None,
    }
}

pub(crate) fn int_value(value: &Option<serde_json::Value>) -> Option<i64> {
    match value {
        Some(serde_json::Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}
