//! Pool round-plan runtime: restore check + frontier freeze for `super::round_plan::RoundPlan`.
//!
//! Invariant: `validate_restored_plan`/`ensure_round_plan` run against the frozen grid in
//! `crates/bridge/src/rl/pool/round_plan.rs` with locks never nested (brief state read, then
//! game/completed locks, then install); failure mode is fail-closed `super::PoolError`, never
//! a partial plan install or a dropped unissued ref.
use super::super::core::PendingDecision;
use super::round_plan::{PlanRow, RoundPlan};
use super::state::NativePool;
use super::{HISTORY_BUCKETS, PoolError, SlotPhase, bucket_ceil};

impl NativePool {
    /// Semantic restore check for an adopted round plan (restore path only,
    /// AFTER replay + READY restage, state lock DROPPED: game/completed
    /// locks only, never nested in state, so no state/game deadlock).
    ///
    /// Invariant: every UNISSUED ref must still be pending on its replayed
    /// game at the frozen frontier/original-bucket/decision-sequence
    /// (nonterminal, same frontier, drained decision holds the seat, actual
    /// history ceil matches `original_bucket`, sequence law matches the frozen
    /// decision); transport is `HISTORY_BUCKETS[pos]` (T256 for live mixed).
    /// A terminal game holding an unissued ref fails closed, never drops.
    /// Every ISSUED ref must be retained in a READY lane or already captured
    /// with the same (seat, sequence, original_bucket) triple (capture copies
    /// seat/sequence verbatim and the original bucket re-derives from the
    /// capture prefix, so a previous round's same-seat row can never satisfy
    /// a current offset). Every retained READY lane must hold a FULL issued
    /// chunk in exact triple order under the frozen round, and every pending
    /// replayed decision must be covered by the plan (no phantom refs, no
    /// missing refs; games already past the round are legitimately uncovered).
    pub(crate) fn validate_restored_plan(&self, plan: &RoundPlan) -> Result<(), PoolError> {
        let cohort = self.games.len();
        plan.validate(cohort)?;
        // Retained READY rows per lane (brief state lock, then dropped
        // before any game lock). Original buckets derive from
        // `obs.history_indices.len()` via `bucket_ceil`, the same law as
        // freeze; transport stays the lane bucket.
        let retained: Vec<(u64, usize, usize, Vec<(usize, u8, u64, usize)>, u64, u64)> = {
            let state = self.lock_state()?;
            let mut out = Vec::new();
            for slot in state.slots.values() {
                if slot.phase != SlotPhase::Ready || slot.rows.is_empty() {
                    continue;
                }
                let mut rows = Vec::with_capacity(slot.rows.len());
                for row in &slot.rows {
                    let original_bucket = super::bucket_ceil(row.obs.history_indices.len()).map_err(|e| {
                        PoolError(format!("restored staged history over-cap: {e}"))
                    })?;
                    rows.push((row.game_idx, row.actor, row.obs.sequence, original_bucket));
                }
                out.push((
                    slot.id,
                    slot.bucket_t,
                    slot.capacity,
                    rows,
                    slot.chunk,
                    slot.logical_round,
                ));
            }
            out
        };
        // Retained rows must be FULL issued chunks in exact order: each lane
        // holds one frozen chunk (start = chunk * cap) with the whole chunk
        // body present (short only as the refs tail), every row issued, and
        // (game, seat, decision-sequence, original_bucket) quadruples in plan
        // order under the frozen round. Transport is the lane bucket (T256
        // for live mixed); the original bucket must match the frozen ref.
        // A partial or unissued retention fails closed.
        for (slot_id, bucket_t, _capacity, rows, chunk, logical_round) in &retained {
            if *logical_round != plan.round {
                return Err(PoolError(format!(
                    "restored slot {slot_id} round {logical_round} mismatches plan round {}",
                    plan.round
                )));
            }
            let pos = RoundPlan::bucket_pos(*bucket_t)?;
            let cap = plan.min_cap[pos].ok_or_else(|| {
                PoolError(format!("restored slot {slot_id} retains rows with no frozen capacity"))
            })?;
            let start = (*chunk as usize).checked_mul(cap).ok_or_else(|| {
                PoolError(format!("restored slot {slot_id} chunk {chunk} overflows"))
            })?;
            let refs_len = plan.refs[pos].len();
            if start >= refs_len {
                return Err(PoolError(format!(
                    "restored slot {slot_id} retains chunk {chunk} past frozen refs"
                )));
            }
            if rows.len() != cap.min(refs_len - start) {
                return Err(PoolError(format!(
                    "restored slot {slot_id} retains a partial chunk {chunk}"
                )));
            }
            if start + rows.len() > plan.issued[pos] {
                return Err(PoolError(format!(
                    "restored slot {slot_id} retains rows past plan issuance"
                )));
            }
            for (offset, (game_idx, actor, seq, original_bucket)) in rows.iter().enumerate() {
                let expect = &plan.refs[pos][start + offset];
                if expect.game_idx != *game_idx || expect.actor != *actor || expect.seq != *seq || expect.original_bucket != *original_bucket {
                    return Err(PoolError(format!(
                        "restored slot {slot_id} row {offset} outside frozen chunk {chunk}"
                    )));
                }
            }
        }
        let retained_contains = |game_idx: usize, actor: u8, seq: u64, original_bucket: usize| {
            retained.iter().any(|(_, _, _, rows, _, _)| {
                rows.iter().any(|(g, a, s, o)| *g == game_idx && *a == actor && *s == seq && *o == original_bucket)
            })
        };
        // Issued refs: retained in a READY lane, or already captured. The
        // capture copies the staged observation's (seat, sequence) verbatim
        // and the original bucket re-derives from the capture prefix, so the
        // quadruple binds the exact frozen decision: a previous round's
        // same-seat row carries an older sequence and can never satisfy a
        // current issued offset. Unissued refs are checked pending below.
        for (pos, rows) in plan.refs.iter().enumerate() {
            for row in rows.iter().take(plan.issued[pos]) {
                if retained_contains(row.game_idx, row.actor, row.seq, row.original_bucket) {
                    continue;
                }
                let captured_rows = self.lock_completed(row.game_idx)?.clone();
                let visible_ok = {
                    let game = self.lock_game(row.game_idx)?;
                    let arena = game.arena();
                    captured_rows.iter().any(|captured| {
                        if captured.actor != row.actor || captured.sequence != row.seq {
                            return false;
                        }
                        let prefix = captured.event_prefix_len as usize;
                        if prefix > arena.len() {
                            return false;
                        }
                        let start = arena.hand_window_start(prefix);
                        let visible = (start..prefix)
                            .filter(|i| arena.get(*i).is_some_and(|e| crate::rl::events::visible_to_actor(e, row.actor)))
                            .count();
                        super::bucket_ceil(visible).is_ok_and(|bucket| bucket == row.original_bucket)
                    })
                };
                if !visible_ok {
                    return Err(PoolError(format!(
                        "restored plan issued game {} seat {} neither retained nor captured",
                        row.game_idx, row.actor
                    )));
                }
            }
        }
        // Unissued refs: still pending on their replayed games now.
        for (pos, rows) in plan.refs.iter().enumerate() {
            for row in rows.iter().skip(plan.issued[pos]) {
                let mut game = self.lock_game(row.game_idx)?;
                if game.is_terminal() {
                    return Err(PoolError(format!(
                        "restored plan holds unissued game {} seat {} on a terminal game",
                        row.game_idx, row.actor
                    )));
                }
                if game.frontier_count() != plan.round {
                    return Err(PoolError(format!(
                        "restored plan game {} frontier drift from round {}",
                        row.game_idx, plan.round
                    )));
                }
                let decision = game.drain().map_err(|e| {
                    PoolError(format!(
                        "restored plan game {} re-drain rejected: {e}",
                        self.wall_ids[row.game_idx]
                    ))
                })?;
                let seats: Vec<u8> = match decision {
                    None => vec![],
                    Some(PendingDecision::Draw { seat }) => vec![seat],
                    Some(PendingDecision::Window { pending }) => pending,
                };
                if !seats.contains(&row.actor) {
                    return Err(PoolError(format!(
                        "restored plan game {} seat {} no longer pending",
                        row.game_idx, row.actor
                    )));
                }
                let bucket = bucket_ceil(
                    game.arena()
                        .visible_count_for_hand(row.actor, game.arena().len()),
                )
                .map_err(|e| {
                    PoolError(format!(
                        "restored plan game {} over-cap history: {e}",
                        self.wall_ids[row.game_idx]
                    ))
                })?;
                if bucket != row.original_bucket {
                    return Err(PoolError(format!(
                        "restored plan game {} seat {} original bucket drift",
                        row.game_idx, row.actor
                    )));
                }
                if plan.is_mixed() {
                    if pos != 3 {
                        return Err(PoolError(format!(
                            "restored plan game {} seat {} outside transport T256",
                            row.game_idx, row.actor
                        )));
                    }
                } else if bucket != HISTORY_BUCKETS[pos] {
                    return Err(PoolError(format!(
                        "restored plan game {} seat {} bucket drift",
                        row.game_idx, row.actor
                    )));
                }
                if RoundPlan::decision_seq(&game, row.actor) != row.seq {
                    return Err(PoolError(format!(
                        "restored plan game {} seat {} decision turned over since freeze",
                        row.game_idx, row.actor
                    )));
                }
            }
        }
        // Reverse cover: every pending replayed decision at the frozen round
        // must be frozen in the plan (no missing refs; restage already
        // re-drained retained games, and re-drains are idempotent windows).
        for game_idx in 0..cohort {
            let mut game = self.lock_game(game_idx)?;
            if game.is_terminal() || game.frontier_count() != plan.round {
                continue;
            }
            let decision = game.drain().map_err(|e| {
                PoolError(format!(
                    "restored cover game {} re-drain rejected: {e}",
                    self.wall_ids[game_idx]
                ))
            })?;
            let seats: Vec<u8> = match decision {
                None => vec![],
                Some(PendingDecision::Draw { seat }) => vec![seat],
                Some(PendingDecision::Window { pending }) => pending,
            };
            for seat in seats {
                if seat > 3 {
                    return Err(PoolError(format!(
                        "restored cover game {game_idx} seat {seat} outside 0..=3"
                    )));
                }
                let bucket = bucket_ceil(
                    game.arena()
                        .visible_count_for_hand(seat, game.arena().len()),
                )
                .map_err(|e| {
                    PoolError(format!(
                        "restored cover game {game_idx} (wall {}) over-cap history: {e}",
                        self.wall_ids[game_idx]
                    ))
                })?;
                let seq = RoundPlan::decision_seq(&game, seat);
                let covered = if plan.is_mixed() {
                    plan.refs[3].iter().any(|row| {
                        row.game_idx == game_idx && row.actor == seat && row.seq == seq && row.original_bucket == bucket
                    })
                } else {
                    let pos = RoundPlan::bucket_pos(bucket)?;
                    plan.refs[pos].iter().any(|row| {
                        row.game_idx == game_idx && row.actor == seat && row.seq == seq && row.original_bucket == bucket
                    })
                };
                if !covered {
                    return Err(PoolError(format!(
                        "restored cover game {game_idx} seat {seat} missing from plan"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Freeze the next frontier's refs once (O(cohort)); every later fill
    /// reuses the frozen grid and never rebuilds it. Returns true while a
    /// servable plan is active, false when blocked (outstanding applies at
    /// the frontier, an all-terminal cohort, or nothing pending). Archive
    /// pools freeze per-bucket refs in cohort order; live pools (`live_binding`
    /// `Some`) freeze every real bucket into transport T256 in
    /// (`original_bucket`, game, actor) order using registered T256 lanes
    /// only. Locks never nest: one brief state read, then game locks, then
    /// one install.
    pub(super) fn ensure_round_plan(&self) -> Result<bool, PoolError> {
        let is_live = self.live_binding.is_some();
        let (caps, pinned) = {
            let state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            if state.round_plan.is_some() {
                return Ok(true);
            }
            // Minimum registered lane capacity per bucket (None for buckets
            // with no lane yet; the first registration freezes those). Live
            // pools use registered T256 lanes only.
            let mut caps: [Option<usize>; 4] = [None, None, None, None];
            for slot in state.slots.values() {
                if is_live && slot.bucket_t != 256 {
                    continue;
                }
                if let Some(pos) = HISTORY_BUCKETS.iter().position(|b| *b == slot.bucket_t) {
                    caps[pos] = Some(caps[pos].map_or(slot.capacity, |min: usize| min.min(slot.capacity)));
                }
            }
            let pinned = RoundPlan::pins(&state);
            (caps, pinned)
        };
        // Frontier scan with no state lock held.
        let mut frontiers: Vec<(u64, bool)> = Vec::with_capacity(self.games.len());
        for idx in 0..self.games.len() {
            let game = self.lock_game(idx)?;
            frontiers.push((game.frontier_count(), game.is_terminal()));
        }
        let mut round = u64::MAX;
        for (_, (frontier, terminal)) in frontiers.iter().enumerate() {
            if !terminal {
                round = round.min(*frontier);
            }
        }
        if round == u64::MAX {
            return Ok(false);
        }
        // Outstanding applies at or below the frontier block the next round:
        // their games have not landed the decisions this freeze would read.
        for (idx, (frontier, _)) in frontiers.iter().enumerate() {
            if pinned[idx] & 2 != 0 && *frontier <= round {
                return Ok(false);
            }
        }
        // Drain pending seats per game in cohort order (drains are
        // idempotent peeks: staging re-drains the same windows verbatim).
        // `original_bucket` derives from the visible history length via
        // `bucket_ceil`, the same law as `obs.history_indices.len()`.
        let mut refs: [Vec<PlanRow>; 4] = Default::default();
        if is_live {
            let mut mixed: Vec<PlanRow> = Vec::new();
            for idx in 0..self.games.len() {
                if pinned[idx] != 0 {
                    continue;
                }
                let (frontier, terminal) = frontiers[idx];
                if terminal || frontier != round {
                    continue;
                }
                let mut game = self.lock_game(idx)?;
                let decision = game.drain().map_err(|e| {
                    PoolError(format!("game {} drain rejected: {e}", self.wall_ids[idx]))
                })?;
                let seats: Vec<u8> = match decision {
                    None => vec![],
                    Some(PendingDecision::Draw { seat }) => vec![seat],
                    Some(PendingDecision::Window { pending }) => pending,
                };
                for seat in seats {
                    if seat > 3 {
                        return Err(PoolError(format!(
                            "game {} decision seat {seat} outside 0..=3",
                            self.wall_ids[idx]
                        )));
                    }
                    let original_bucket = bucket_ceil(
                        game.arena()
                            .visible_count_for_hand(seat, game.arena().len()),
                    )
                    .map_err(|e| {
                        PoolError(format!("game {} over-cap history: {e}", self.wall_ids[idx]))
                    })?;
                    let seq = RoundPlan::decision_seq(&game, seat);
                    mixed.push(PlanRow { game_idx: idx, actor: seat, seq, original_bucket });
                }
            }
            if mixed.is_empty() {
                return Ok(false);
            }
            mixed.sort_by_key(|row| {
                (
                    HISTORY_BUCKETS.iter().position(|b| *b == row.original_bucket).unwrap_or(usize::MAX),
                    row.game_idx,
                    row.actor,
                )
            });
            refs[3] = mixed;
        } else {
            for idx in 0..self.games.len() {
                if pinned[idx] != 0 {
                    continue;
                }
                let (frontier, terminal) = frontiers[idx];
                if terminal || frontier != round {
                    continue;
                }
                let mut game = self.lock_game(idx)?;
                let decision = game.drain().map_err(|e| {
                    PoolError(format!("game {} drain rejected: {e}", self.wall_ids[idx]))
                })?;
                let seats: Vec<u8> = match decision {
                    None => vec![],
                    Some(PendingDecision::Draw { seat }) => vec![seat],
                    Some(PendingDecision::Window { pending }) => pending,
                };
                for seat in seats {
                    if seat > 3 {
                        return Err(PoolError(format!(
                            "game {} decision seat {seat} outside 0..=3",
                            self.wall_ids[idx]
                        )));
                    }
                    let bucket = bucket_ceil(
                        game.arena()
                            .visible_count_for_hand(seat, game.arena().len()),
                    )
                    .map_err(|e| {
                        PoolError(format!("game {} over-cap history: {e}", self.wall_ids[idx]))
                    })?;
                    let pos = RoundPlan::bucket_pos(bucket)?;
                    let seq = RoundPlan::decision_seq(&game, seat);
                    refs[pos].push(PlanRow { game_idx: idx, actor: seat, seq, original_bucket: bucket });
                }
            }
            if refs.iter().all(|rows| rows.is_empty()) {
                return Ok(false);
            }
        }
        let plan = RoundPlan::freeze(round, refs, caps)?;
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        if state.round_plan.is_some() {
            return Ok(true);
        }
        state.round_plan = Some(plan);
        self.notify(&mut state);
        Ok(true)
    }
}
