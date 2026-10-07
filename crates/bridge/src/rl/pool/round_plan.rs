//! Immutable per-frontier round plan: frozen row references, min-cap chunks,
//! head progression.
//!
//! A plan freezes ONCE per resolved frontier: every nonterminal game's
//! pending decision, in cohort order, grouped by history bucket ascending.
//! Each bucket's rows split into fixed-size pieces of that bucket's frozen
//! minimum registered lane capacity (final piece may be short).
//! Scheduling serves pieces in bucket order; a piece stages only when none
//! of its rows' games are pinned and its lane fits the whole piece.
//! Progress is per-bucket issued row offsets; chunk lists derive
//! deterministically and are never stored. Membership is immutable once
//! frozen: a pinned head waits, never reassigns rows to other fills.

use super::super::core::NativeGame;
use super::state::PoolState;
use super::{HISTORY_BUCKETS, PoolError, SlotPhase};

/// One frozen row reference: a seat whose pending decision was drained at
/// plan freeze. `seq` is the frozen observation-sequence law value for that
/// exact decision (last visible history event sequence, 0 when empty):
/// it binds the ref to one decision, so a previous round's same-seat row
/// can never satisfy this round's issuance. `original_bucket` is the actual
/// history bucket derived from `obs.history_indices.len()` via `bucket_ceil`
/// (32/64/128/256): archive refs carry `original_bucket` equal to the
/// transport bucket holding the ref; live refs all ride transport T256 while
/// `original_bucket` preserves the real bucket. Observations, uniforms, and
/// sampler words still bind later at staging; a bare ref never authorizes
/// inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlanRow {
    pub(super) game_idx: usize,
    pub(super) actor: u8,
    pub(super) seq: u64,
    pub(super) original_bucket: usize,
}

/// Immutable per-frontier collection plan. Frozen once per round; only the
/// per-bucket issued offsets advance (exactly on READY install), and only
/// forward to exact chunk boundaries. Buckets with pending rows but no
/// registered lane keep refs with `min_cap` unset: their rows wait, never
/// dropped, never guessed into chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundPlan {
    pub(super) round: u64,
    pub(super) refs: [Vec<PlanRow>; 4],
    pub(super) min_cap: [Option<usize>; 4],
    pub(super) issued: [usize; 4],
}

impl RoundPlan {
    /// Bucket position 0..4 for a lane bucket, or fail loud on unknown T
    /// (a lane outside the frozen vocabulary can never serve plan rows).
    pub(super) fn bucket_pos(bucket_t: usize) -> Result<usize, PoolError> {
        HISTORY_BUCKETS
            .iter()
            .position(|b| *b == bucket_t)
            .ok_or_else(|| {
                PoolError(format!("plan bucket T={bucket_t} outside {HISTORY_BUCKETS:?}"))
            })
    }

    /// Frozen decision identity for one seat: the observation-sequence law
    /// value (`observe` lines 399-406: last visible history event sequence,
    /// 0 when empty), replicated without building an observation. Staged
    /// rows carry it as `obs.sequence` and captured rows as `GameRow.sequence`
    /// (capture copies it verbatim), so this one u64 binds a ref to its exact
    /// decision on every path. Call only with the seat's pending decision
    /// drained (same game borrow as the bucket check, never nested).
    pub(super) fn decision_seq(game: &NativeGame, seat: u8) -> u64 {
        let arena = game.arena();
        arena
            .visible_indices_for_hand(seat, arena.len())
            .into_iter()
            .last()
            .and_then(|index| arena.get(index))
            .map_or(0, |event| event.sequence)
    }

    /// Freeze a plan from per-bucket row refs with the frozen minimum lane
    /// capacity per bucket (`None` for buckets with no registered lane).
    /// Archive refs arrive per-bucket in ascending (game_idx, actor) order
    /// with `original_bucket` equal to the holding transport bucket. Live
    /// refs collapse every real bucket into transport T256 in ascending
    /// (original_bucket, game_idx, actor) order; mixed is determined by T256
    /// refs containing an earlier bucket. Rejects unsorted refs, bad actors,
    /// bad original buckets, and zero caps: a silently reordered plan would
    /// fork batch membership with no diagnostic trail.
    pub(super) fn freeze(
        round: u64,
        refs: [Vec<PlanRow>; 4],
        min_cap: [Option<usize>; 4],
    ) -> Result<Self, PoolError> {
        let mixed = refs[3].iter().any(|row| row.original_bucket != 256);
        if mixed {
            if !refs[0].is_empty() || !refs[1].is_empty() || !refs[2].is_empty() {
                return Err(PoolError(
                    "live mixed refs must collapse every bucket into transport T256".to_string(),
                ));
            }
            if min_cap[0].is_some() || min_cap[1].is_some() || min_cap[2].is_some() {
                return Err(PoolError(
                    "live mixed plan caps outside transport T256".to_string(),
                ));
            }
            let order = |bucket: usize| {
                HISTORY_BUCKETS.iter().position(|b| *b == bucket).unwrap_or(usize::MAX)
            };
            let mut prev: Option<(usize, usize, u8)> = None;
            for row in &refs[3] {
                if row.actor > 3 {
                    return Err(PoolError(format!(
                        "plan refs seat {} outside 0..=3",
                        row.actor
                    )));
                }
                if !HISTORY_BUCKETS.contains(&row.original_bucket) {
                    return Err(PoolError(format!(
                        "plan refs original bucket {} outside {HISTORY_BUCKETS:?}",
                        row.original_bucket
                    )));
                }
                let key = (order(row.original_bucket), row.game_idx, row.actor);
                if prev.is_some_and(|p| p >= key) {
                    return Err(PoolError(
                        "plan refs for transport T256 unordered by (original_bucket, game, actor)".to_string(),
                    ));
                }
                prev = Some(key);
            }
        } else {
            for (pos, rows) in refs.iter().enumerate() {
                let mut prev: Option<(usize, u8)> = None;
                for row in rows {
                    if row.actor > 3 {
                        return Err(PoolError(format!(
                            "plan refs seat {} outside 0..=3",
                            row.actor
                        )));
                    }
                    if row.original_bucket != HISTORY_BUCKETS[pos] {
                        return Err(PoolError(format!(
                            "plan refs original bucket {} != transport T={} for archive",
                            row.original_bucket, HISTORY_BUCKETS[pos]
                        )));
                    }
                    let key = (row.game_idx, row.actor);
                    if prev.is_some_and(|p| p >= key) {
                        return Err(PoolError(format!(
                            "plan refs for T={} unordered",
                            HISTORY_BUCKETS[pos]
                        )));
                    }
                    prev = Some(key);
                }
            }
        }
        let mut seen: Vec<(usize, u8)> = Vec::new();
        for rows in refs.iter() {
            for row in rows {
                let key = (row.game_idx, row.actor);
                if seen.contains(&key) {
                    return Err(PoolError(format!(
                        "plan refs game {} seat {} twice across buckets",
                        row.game_idx, row.actor
                    )));
                }
                seen.push(key);
            }
        }
        for cap in min_cap.iter().flatten() {
            if *cap == 0 || *cap > super::TICKET_CAP {
                return Err(PoolError(format!(
                    "plan min capacity {cap} outside 1..={}",
                    super::TICKET_CAP
                )));
            }
        }
        Ok(Self {
            round,
            refs,
            min_cap,
            issued: [0; 4],
        })
    }

    /// True when the T256 transport bucket carries mixed original buckets
    /// (live collapsed transport). Archive plans, including all-T256 archive
    /// plans whose original buckets are all 256, report false with identical
    /// semantics on either path.
    pub(super) fn is_mixed(&self) -> bool {
        self.refs[3].iter().any(|row| row.original_bucket != 256)
    }

    /// True when every frozen row has been installed (the round may still
    /// need leases/applies to settle before the plan drops; exhaustion is
    /// about issuance, never about GPU progress).
    pub(super) fn exhausted(&self) -> bool {
        (0..4).all(|b| self.issued[b] >= self.refs[b].len())
    }

    /// Global head: smallest bucket position with unissued refs. Callers
    /// serve this bucket and MUST NOT overtake with later buckets when it
    /// cannot proceed (overtaking reintroduces timing-dependent batch splits
    /// with no diagnostic trail).
    pub(super) fn head_bucket(&self) -> Option<usize> {
        (0..4).find(|b| self.issued[*b] < self.refs[*b].len())
    }

    /// Next unissued chunk for one bucket: (plan chunk ordinal, row range).
    /// `None` while exhausted OR cap unknown (caller waits in both cases;
    /// waiting on an unknown cap is not an error). Errors only on an
    /// off-grid issued offset, which means the progress state does not
    /// describe this law and must fail, never install.
    pub(super) fn next_chunk(
        &self,
        bucket_t: usize,
    ) -> Result<Option<(u64, std::ops::Range<usize>)>, PoolError> {
        let b = Self::bucket_pos(bucket_t)?;
        let cap = match self.min_cap[b] {
            Some(cap) => cap,
            None => return Ok(None),
        };
        let start = self.issued[b];
        let len = self.refs[b].len();
        if start >= len {
            return Ok(None);
        }
        if start % cap != 0 {
            return Err(PoolError(format!(
                "plan issued offset {start} off chunk grid {cap} for T={bucket_t}"
            )));
        }
        let end = (start + cap).min(len);
        Ok(Some(((start / cap) as u64, start..end)))
    }

    /// Advance past exactly the rows just installed READY, returning the
    /// installed chunk ordinal. Rejects overrun and non-boundary advances:
    /// progress that is not an exact chunk boundary would fork lane chunk
    /// ordinals silently.
    pub(super) fn advance(&mut self, bucket_t: usize, n: usize) -> Result<u64, PoolError> {
        let b = Self::bucket_pos(bucket_t)?;
        let cap = self.min_cap[b].ok_or_else(|| {
            PoolError(format!("plan advances T={bucket_t} with no frozen capacity"))
        })?;
        let start = self.issued[b];
        let len = self.refs[b].len();
        if start >= len {
            return Err(PoolError(format!(
                "plan advances exhausted bucket T={bucket_t}"
            )));
        }
        let expect = cap.min(len - start);
        if n != expect {
            return Err(PoolError(format!(
                "plan advance {n} rows != head chunk {expect} rows for T={bucket_t}"
            )));
        }
        self.issued[b] = start + n;
        Ok((start / cap) as u64)
    }

    /// Borrowed rows of the next chunk (the exact slice staging must serve).
    pub(super) fn head_rows(
        &self,
        bucket_t: usize,
    ) -> Result<Option<(u64, &[PlanRow])>, PoolError> {
        Ok(match self.next_chunk(bucket_t)? {
            Some((ordinal, range)) => {
                let b = Self::bucket_pos(bucket_t)?;
                Some((ordinal, &self.refs[b][range]))
            }
            None => None,
        })
    }

    /// Canonical JSON for snapshots: round, per-bucket refs in bucket order,
    /// nullable per-bucket min caps, per-bucket issued offsets. Chunks never
    /// serialize (they derive deterministically from refs + caps). Each row
    /// carries the exact `original_bucket` key (actual 32/64/128/256).
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "round": self.round,
            "refs": self.refs.iter().map(|rows| rows.iter().map(|row| {
                serde_json::json!({"game": row.game_idx, "actor": row.actor, "seq": row.seq, "original_bucket": row.original_bucket})
            }).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "min_cap": self.min_cap.iter().map(|cap| match cap {
                Some(cap) => serde_json::Value::from(*cap as u64),
                None => serde_json::Value::Null,
            }).collect::<Vec<_>>(),
            "issued": self.issued.iter().map(|n| serde_json::Value::from(*n as u64)).collect::<Vec<_>>(),
        })
    }

    /// Closed-shape parse (structure only; call `validate` for semantics).
    /// Unknown shapes fail here so old or foreign plans never load silently.
    pub(crate) fn from_json(value: &serde_json::Value) -> Result<Self, PoolError> {
        let fail = |what: &str| PoolError(format!("round plan {what}"));
        super::check_closed(value, &["round", "refs", "min_cap", "issued"], "round plan")?;
        let doc = value
            .as_object()
            .ok_or_else(|| fail("must be an object"))?;
        let round = doc
            .get("round")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| fail("lacks integer round"))?;
        let refs_json = doc
            .get("refs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| fail("lacks refs array"))?;
        if refs_json.len() != 4 {
            return Err(fail("refs must hold four buckets"));
        }
        let caps_json = doc
            .get("min_cap")
            .and_then(|v| v.as_array())
            .ok_or_else(|| fail("lacks min_cap array"))?;
        if caps_json.len() != 4 {
            return Err(fail("min_cap must hold four buckets"));
        }
        let issued_json = doc
            .get("issued")
            .and_then(|v| v.as_array())
            .ok_or_else(|| fail("lacks issued array"))?;
        if issued_json.len() != 4 {
            return Err(fail("issued must hold four buckets"));
        }
        let mut refs: [Vec<PlanRow>; 4] = Default::default();
        for (pos, entries) in refs_json.iter().enumerate() {
            let entries = entries
                .as_array()
                .ok_or_else(|| fail("refs bucket must be an array"))?;
            for entry in entries {
                super::check_closed(entry, &["game", "actor", "seq", "original_bucket"], "round plan row")?;
                let obj = entry
                    .as_object()
                    .ok_or_else(|| fail("plan row must be an object"))?;
                let game_idx = obj
                    .get("game")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| fail("plan row lacks integer game"))?
                    as usize;
                let actor = obj
                    .get("actor")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| fail("plan row lacks integer actor"))?;
                if actor > 3 {
                    return Err(fail("plan row actor outside 0..=3"));
                }
                let seq = obj
                    .get("seq")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| fail("plan row lacks integer seq"))?;
                let original_bucket = obj
                    .get("original_bucket")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| fail("plan row lacks integer original_bucket"))?
                    as usize;
                if !HISTORY_BUCKETS.contains(&original_bucket) {
                    return Err(fail("plan row original_bucket outside 32/64/128/256"));
                }
                refs[pos].push(PlanRow {
                    game_idx,
                    actor: actor as u8,
                    seq,
                    original_bucket,
                });
            }
        }
        let mut min_cap: [Option<usize>; 4] = [None, None, None, None];
        for (pos, entry) in caps_json.iter().enumerate() {
            match entry {
                serde_json::Value::Null => {}
                serde_json::Value::Number(_) => {
                    let cap = entry
                        .as_u64()
                        .ok_or_else(|| fail("plan cap must be an integer or null"))?
                        as usize;
                    min_cap[pos] = Some(cap);
                }
                _ => return Err(fail("plan cap must be an integer or null")),
            }
        }
        let mut issued = [0usize; 4];
        for (pos, entry) in issued_json.iter().enumerate() {
            issued[pos] = entry
                .as_u64()
                .ok_or_else(|| fail("plan issued must hold integers"))?
                as usize;
        }
        let plan = Self::freeze(round, refs, min_cap)?;
        for (pos, _) in HISTORY_BUCKETS.iter().enumerate() {
            plan.check_offset(pos, issued[pos])?;
        }
        let mut plan = plan;
        plan.issued = issued;
        Ok(plan)
    }

    /// Boundary check for one bucket's issued offset without advancing: it
    /// must sit on an exact chunk gridline, or exactly at the tail end.
    fn check_offset(&self, pos: usize, issued: usize) -> Result<(), PoolError> {
        let len = self.refs[pos].len();
        if issued > len {
            return Err(PoolError(format!(
                "plan issued offset {issued} past frozen refs {len} for T={}",
                HISTORY_BUCKETS[pos]
            )));
        }
        match self.min_cap[pos] {
            Some(cap) => {
                if issued % cap != 0 && issued != len {
                    return Err(PoolError(format!(
                        "plan issued offset {issued} off chunk grid {cap} for T={}",
                        HISTORY_BUCKETS[pos]
                    )));
                }
                Ok(())
            }
            None => {
                if issued != 0 {
                    return Err(PoolError(format!(
                        "plan issued rows for T={} with no frozen capacity",
                        HISTORY_BUCKETS[pos]
                    )));
                }
                Ok(())
            }
        }
    }

    /// Semantic validation against the cohort size: game indices in range,
    /// original buckets in vocabulary with archive equality
    /// (`original_bucket` equals holding transport) or live mixed order
    /// (T256 refs sorted by `original_bucket`, game, actor), plus issued
    /// offsets on grid (caps are enforced once in `freeze`, which every
    /// constructor path runs through). Frontier/pending cross-checks happen
    /// at restore against replayed games, not here.
    pub(crate) fn validate(&self, cohort_len: usize) -> Result<(), PoolError> {
        for rows in self.refs.iter() {
            for row in rows {
                if row.game_idx >= cohort_len {
                    return Err(PoolError(format!(
                        "plan row game {} outside cohort {cohort_len}",
                        row.game_idx
                    )));
                }
                if !HISTORY_BUCKETS.contains(&row.original_bucket) {
                    return Err(PoolError(format!(
                        "plan row original bucket {} outside {HISTORY_BUCKETS:?}",
                        row.original_bucket
                    )));
                }
            }
        }
        if self.is_mixed() {
            if !self.refs[0].is_empty() || !self.refs[1].is_empty() || !self.refs[2].is_empty() {
                return Err(PoolError(
                    "live mixed refs must collapse every bucket into transport T256".to_string(),
                ));
            }
            let order = |bucket: usize| {
                HISTORY_BUCKETS.iter().position(|b| *b == bucket).unwrap_or(usize::MAX)
            };
            let mut prev: Option<(usize, usize, u8)> = None;
            for row in &self.refs[3] {
                let key = (order(row.original_bucket), row.game_idx, row.actor);
                if prev.is_some_and(|p| p >= key) {
                    return Err(PoolError(
                        "plan refs for transport T256 unordered by (original_bucket, game, actor)".to_string(),
                    ));
                }
                prev = Some(key);
            }
        } else {
            for (pos, rows) in self.refs.iter().enumerate() {
                let mut prev: Option<(usize, u8)> = None;
                for row in rows {
                    if row.original_bucket != HISTORY_BUCKETS[pos] {
                        return Err(PoolError(format!(
                            "plan refs original bucket {} != transport T={} for archive",
                            row.original_bucket, HISTORY_BUCKETS[pos]
                        )));
                    }
                    let key = (row.game_idx, row.actor);
                    if prev.is_some_and(|p| p >= key) {
                        return Err(PoolError(format!(
                            "plan refs for T={} unordered",
                            HISTORY_BUCKETS[pos]
                        )));
                    }
                    prev = Some(key);
                }
            }
        }
        for (pos, _) in HISTORY_BUCKETS.iter().enumerate() {
            self.check_offset(pos, self.issued[pos])?;
        }
        Ok(())
    }

    /// Frozen frontier this plan collects (every staged row installs under
    /// this round; leases order on it).
    pub(super) fn round(&self) -> u64 {
        self.round
    }

    /// Freeze one bucket's chunk size on its FIRST lane registration while
    /// the plan is active. Later registrations never repartition: the frozen
    /// grid is already installed in some lane's chunk ordinals.
    pub(super) fn note_lane(&mut self, bucket_t: usize, capacity: usize) -> Result<(), PoolError> {
        let pos = Self::bucket_pos(bucket_t)?;
        if self.min_cap[pos].is_none() {
            if capacity == 0 || capacity > super::TICKET_CAP {
                return Err(PoolError(format!(
                    "plan lane capacity {capacity} outside 1..={}",
                    super::TICKET_CAP
                )));
            }
            self.min_cap[pos] = Some(capacity);
        }
        Ok(())
    }

    /// Drop an exhausted plan only once every READY lane, lease, apply, and
    /// dispatched fill has settled. Anything outstanding means the round's
    /// rows are still in flight, so the plan stays (and blocks the next
    /// freeze) instead of letting a new frontier cannibalize its games.
    pub(super) fn maybe_retire(state: &mut PoolState) {
        let exhausted = state.round_plan.as_ref().is_some_and(|plan| plan.exhausted());
        if !exhausted {
            return;
        }
        let lanes_settled = state.slots.values().all(|slot| slot.phase == SlotPhase::Free);
        let applies_settled = state.pending_applies.is_empty()
            && state.apply_outstanding.iter().all(|count| *count == 0);
        if lanes_settled
            && state.leases.is_empty()
            && applies_settled
            && state.fill_inflight.is_none()
        {
            state.round_plan = None;
        }
    }

    /// Per-game pin bits for scheduling: bit0 set while a non-Free lane
    /// holds the game's rows, bit1 while applies are outstanding. A chunk
    /// naming a pinned game waits unchanged instead of splitting around it.
    pub(super) fn pins(state: &PoolState) -> Vec<u8> {
        let mut pinned = vec![0u8; state.apply_outstanding.len()];
        for slot in state.slots.values() {
            if slot.phase != SlotPhase::Free {
                for row in &slot.rows {
                    if let Some(bit) = pinned.get_mut(row.game_idx) {
                        *bit |= 1;
                    }
                }
            }
        }
        for (idx, count) in state.apply_outstanding.iter().enumerate() {
            if *count > 0 {
                pinned[idx] |= 2;
            }
        }
        pinned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> RoundPlan {
        RoundPlan::freeze(
            3,
            [
                vec![
                    PlanRow { game_idx: 0, actor: 0, seq: 11, original_bucket: 32 },
                    PlanRow { game_idx: 0, actor: 2, seq: 12, original_bucket: 32 },
                    PlanRow { game_idx: 1, actor: 1, seq: 13, original_bucket: 32 },
                ],
                vec![PlanRow { game_idx: 2, actor: 0, seq: 14, original_bucket: 64 }],
                vec![],
                vec![PlanRow { game_idx: 3, actor: 3, seq: 15, original_bucket: 256 }],
            ],
            [Some(2), Some(8), None, None],
        )
        .expect("fixture plan")
    }

    #[test]
    fn freeze_rejects_unordered_refs_and_bad_actors() {
        let refs: [Vec<PlanRow>; 4] = [
            vec![
                PlanRow { game_idx: 1, actor: 0, seq: 1, original_bucket: 32 },
                PlanRow { game_idx: 0, actor: 0, seq: 2, original_bucket: 32 },
            ],
            vec![],
            vec![],
            vec![],
        ];
        assert!(RoundPlan::freeze(0, refs, [Some(2), None, None, None]).is_err());
        let refs: [Vec<PlanRow>; 4] = [
            vec![PlanRow { game_idx: 0, actor: 4, seq: 3, original_bucket: 32 }],
            vec![],
            vec![],
            vec![],
        ];
        assert!(RoundPlan::freeze(0, refs, [Some(2), None, None, None]).is_err());
        let refs: [Vec<PlanRow>; 4] = [vec![], vec![], vec![], vec![]];
        assert!(RoundPlan::freeze(0, refs, [Some(0), None, None, None]).is_err());
        // One seat frozen twice across buckets would capture twice.
        let refs: [Vec<PlanRow>; 4] = [
            vec![PlanRow { game_idx: 0, actor: 1, seq: 4, original_bucket: 32 }],
            vec![PlanRow { game_idx: 0, actor: 1, seq: 4, original_bucket: 64 }],
            vec![],
            vec![],
        ];
        assert!(RoundPlan::freeze(0, refs, [Some(2), Some(2), None, None]).is_err());
    }
    #[test]
    fn head_follows_bucket_order_and_waits_on_missing_cap() {
        let plan = plan();
        // Smallest bucket with unissued refs wins, even though T64+ hold rows.
        assert_eq!(plan.head_bucket(), Some(0));
        let (ordinal, rows) = plan.head_rows(32).expect("head").expect("chunk");
        assert_eq!(ordinal, 0);
        assert_eq!(rows.len(), 2);
        // Unknown bucket fails loud; exhausted bucket yields nothing.
        assert!(plan.next_chunk(33).is_err());
        assert!(plan.next_chunk(128).expect("empty").is_none());
        // Bucket 256 refs exist but no cap is frozen: wait, never guess.
        assert!(plan.next_chunk(256).expect("uncapped").is_none());
    }
    #[test]
    fn advance_moves_by_exact_chunks_only() {
        let mut plan = plan();
        assert_eq!(plan.advance(32, 2).expect("advance"), 0);
        assert_eq!(plan.advance(32, 1).expect("tail"), 1);
        assert!(plan.advance(32, 1).is_err());
        assert!(plan.advance(64, 2).is_err());
        assert_eq!(plan.advance(64, 1).expect("solo"), 0);
        // Bucket 256 holds a ref but never registered a lane: no cap is
        // frozen, so it waits instead of issuing; the plan is not exhausted.
        assert!(plan.advance(256, 1).is_err());
        assert!(!plan.exhausted());
        assert_eq!(plan.head_bucket(), Some(3));
        // The first lane registration freezes the missing cap; the waiting
        // ref then issues exactly, and only then is the plan exhausted.
        plan.note_lane(256, 4).expect("freeze");
        assert_eq!(plan.advance(256, 1).expect("late"), 0);
        assert!(plan.exhausted());
        assert_eq!(plan.head_bucket(), None);
    }
    #[test]
    fn serde_round_trip_and_closed_shape() {
        let mut plan = plan();
        plan.advance(32, 2).expect("advance");
        let bytes = serde_json::to_string(&plan.to_json()).expect("encode");
        let back =
            RoundPlan::from_json(&serde_json::from_str(&bytes).expect("parse")).expect("decode");
        assert_eq!(back, plan);
        assert!(back.validate(4).is_ok());
        let mut bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad.as_object_mut()
            .expect("object")
            .remove("issued");
        assert!(RoundPlan::from_json(&bad).is_err());
        bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad["refs"][0][0]["actor"] = serde_json::Value::from(9u64);
        assert!(RoundPlan::from_json(&bad).is_err());
        bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad["issued"][0] = serde_json::Value::from(1u64);
        assert!(RoundPlan::from_json(&bad).is_err());
        bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad["refs"][0]
            .as_array_mut()
            .expect("refs array")
            .push(serde_json::json!({"game": 9u64, "actor": 0u8, "seq": 1u64, "original_bucket": 32u64}));
        assert!(back.validate(4).is_ok());
        assert!(RoundPlan::from_json(&bad).expect("shape ok").validate(4).is_err());
        // Dropping the decision identity breaks the closed shape (a 2-field
        // legacy row can no longer bind its decision).
        bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad["refs"][0][0]
            .as_object_mut()
            .expect("row object")
            .remove("seq");
        assert!(RoundPlan::from_json(&bad).is_err());
        // Dropping the original bucket breaks the closed shape (transport
        // alone cannot bind the real bucket for live mixed refs).
        bad = serde_json::from_str::<serde_json::Value>(&bytes).expect("parse");
        bad["refs"][0][0]
            .as_object_mut()
            .expect("row object")
            .remove("original_bucket");
        assert!(RoundPlan::from_json(&bad).is_err());
    }
}
