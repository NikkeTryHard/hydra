//! rl/pool/fill: lane registration, dispatched fills, takes, refills.
//!
//! Invariant: at most one fill is ever in flight and a fill stages each
//! eligible decision at most once (cohort order, live-lane pins plus
//! outstanding-apply pins), with exactly one sampler reservation per real
//! decision that is never rewound even when the install fails. Takes install
//! under the state lock with chunk assignment on the taker, never the worker.
//! Failure mode: a busy lane, an undersized slab, or a worker-side encode
//! failure errors (or poisons loudly) instead of staging partial rows; empty
//! fills leave the lane FREE for work-conserving stalls (no dummy rows).

use std::sync::Arc;

use super::super::core::PendingDecision;
use super::super::observation::{self, NativeObservation};
use super::round_plan::{PlanRow, RoundPlan};
use super::state::{FillInflight, FillOutput, FilledRow, WorkerJob, slot_mut, slot_ref};
use super::{
    HISTORY_BUCKETS, NativePool, PoolError, SlotPhase, TicketDescriptor, bucket_ceil,
    buffer_geometry,
};

impl NativePool {
    pub fn register_slot(
        &self,
        slot_id: u64,
        bucket_t: usize,
        capacity: usize,
    ) -> Result<(), PoolError> {
        buffer_geometry(bucket_t, capacity)?;
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        if state.slots.contains_key(&slot_id) {
            return Err(PoolError(format!("slot {slot_id} already registered")));
        }
        state.slots.insert(
            slot_id,
            super::state::Slot {
                id: slot_id,
                epoch: 0,
                bucket_t,
                capacity,
                phase: SlotPhase::Free,
                leased: false,
                rows: vec![],
                chunk: 0,
                logical_round: 0,
            },
        );
        // A bucket with frozen refs but no registered lane waits with
        // `min_cap` unset; the FIRST lane registration freezes its chunk
        // size. Later registrations never repartition the active grid.
        if let Some(plan) = state.round_plan.as_mut() {
            plan.note_lane(bucket_t, capacity)?;
        }
        self.notify(&mut state);
        Ok(())
    }

    /// Attach buffers to a lane: fresh lanes register; restore-adopted lanes
    /// revalidate `(T, capacity)` against the snapshot adoption and keep
    /// everything else (epochs continue; retained READY rows, leases, and
    /// phases are adopted state, not live work, so re-registration is a
    /// buffer binding, never a reset).
    pub fn attach_slot(
        &self,
        slot_id: u64,
        bucket_t: usize,
        capacity: usize,
    ) -> Result<(), PoolError> {
        buffer_geometry(bucket_t, capacity)?;
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        if let Some(slot) = state.slots.get(&slot_id) {
            if slot.bucket_t != bucket_t || slot.capacity != capacity {
                return Err(PoolError(format!(
                    "slot {slot_id} re-registration (T={bucket_t},cap={capacity}) mismatches adopted (T={},cap={})",
                    slot.bucket_t, slot.capacity
                )));
            }
        } else {
            state.slots.insert(
                slot_id,
                super::state::Slot {
                    id: slot_id,
                    epoch: 0,
                    bucket_t,
                    capacity,
                    phase: SlotPhase::Free,
                    leased: false,
                    rows: vec![],
                    chunk: 0,
                    logical_round: 0,
                },
            );
        }
        if let Some(plan) = state.round_plan.as_mut() {
            plan.note_lane(bucket_t, capacity)?;
        }
        self.notify(&mut state);
        Ok(())
    }

    /// Admit a dispatched fill for one FREE lane: validates geometry,
    /// reserves the single in-flight fill slot, and dispatches to the persistent
    /// fill worker that drains/stages/encodes off the CUDA owner thread. The owner keeps
    /// the GPU supplied and collects the result with `take_fill` (or the
    /// blocking `request_ready_fill` wrapper). At most one fill is ever in
    /// flight (concurrent fills could stage the same decision twice); a
    /// second admit errors as backpressure. Commits are rejected while a
    /// fill is in flight (the fill's staged games stay thread-local until
    /// install, so the commit pin check cannot see them); the reverse needs
    /// no flag (a committing lane stays non-Free with its rows, so admitted
    /// fills skip its games via the live-lane pins).
    pub fn admit_fill(
        self: &Arc<Self>,
        slot_id: u64,
        output_capacity: usize,
    ) -> Result<(), PoolError> {
        self.harvest_applies();
        let (bucket_t, capacity, epoch) = {
            let mut state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            if state.drained {
                return Err(PoolError("pool trajectories already drained".to_string()));
            }
            if state.quiesced {
                return Err(PoolError("pool quiesced; no new fills".to_string()));
            }
            if state.fill_inflight.is_some() {
                return Err(PoolError(
                    "fill already in flight; take before admitting".to_string(),
                ));
            }
            // Retire a settled plan before this admission creates a Filling pin;
            // retiring inside the worker can never see a fully settled state.
            RoundPlan::maybe_retire(&mut state);
            // Copy geometry under a short slot borrow, then release it before
            // touching `fill_inflight` (a live `&mut Slot` forbids it).
            let (slot_bucket, slot_capacity, slot_epoch) = {
                let slot = slot_mut(&mut state, slot_id)?;
                if slot.phase != SlotPhase::Free {
                    return Err(PoolError(format!(
                        "slot {slot_id} busy; admission backpressure"
                    )));
                }
                let (_, output_need) = buffer_geometry(slot.bucket_t, slot.capacity)?;
                if output_capacity < output_need {
                    return Err(PoolError(format!(
                        "output slab {output_capacity} < capacity {output_need}"
                    )));
                }
                slot.phase = SlotPhase::Filling;
                (slot.bucket_t, slot.capacity, slot.epoch)
            };
            state.fill_inflight = Some(FillInflight {
                slot_id,
                epoch: slot_epoch,
                result: None,
            });
            state.admit_ok = state.admit_ok.saturating_add(1);
            self.notify(&mut state);
            (slot_bucket, slot_capacity, slot_epoch)
        };
        let sender = self.lock_workers()?.fill_sender.clone();
        if sender.is_none_or(|sender| sender.send((slot_id, epoch, bucket_t, capacity)).is_err()) {
            self.poison("fill worker queue closed".to_string())?;
            return Err(PoolError("fill worker queue closed".to_string()));
        }
        Ok(())
    }

    /// Worker deposit (fill thread only): publish the payload under the
    /// state lock and wake any blocking taker. A mismatched deposit (no
    /// matching admission) poisons loudly instead of landing nowhere.
    pub(super) fn deposit_fill(&self, output: FillOutput) {
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(_) => return,
        };
        match state.fill_inflight.as_mut() {
            Some(inflight)
                if inflight.slot_id == output.slot_id && inflight.epoch == output.epoch =>
            {
                inflight.result = Some(output);
                self.notify(&mut state);
            }
            _ => self.poison_locked(
                &mut state,
                format!("slot {} fill deposit without admission", output.slot_id),
            ),
        }
    }

    /// Readiness gate for the FFI (one lock, zero allocation): `Err` when
    /// no fill was admitted at all (owner bug, loud), `Ok(false)` while
    /// the worker runs or another slot's fill is in flight, `Ok(true)`
    /// when this slot's payload waits to be taken.
    pub fn fill_ready(&self, slot_id: u64) -> Result<bool, PoolError> {
        let state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        match state.fill_inflight.as_ref() {
            None => Err(PoolError(format!("slot {slot_id} has no admitted fill"))),
            Some(inflight) if inflight.slot_id != slot_id => Ok(false),
            Some(inflight) => Ok(inflight.result.is_some()),
        }
    }
    /// Best-effort take-path telemetry for the FFI (never fails the take:
    /// a poisoned lock means the pool is already broken elsewhere).
    pub fn bump_take_hit(&self) {
        if let Ok(mut state) = self.lock_state() {
            state.take_hit = state.take_hit.saturating_add(1);
        }
    }

    pub fn bump_take_miss(&self) {
        if let Ok(mut state) = self.lock_state() {
            state.take_miss = state.take_miss.saturating_add(1);
        }
    }

    pub fn bump_take_blocking(&self) {
        if let Ok(mut state) = self.lock_state() {
            state.take_blocking = state.take_blocking.saturating_add(1);
        }
    }

    /// Route the frozen head chunk to one fitting free lane. The head is the
    /// lowest bucket with unissued refs; a missing cap or a pinned head
    /// WAITS (returns None), never bypasses to a later bucket and never
    /// shrinks or skips rows. Among free lanes that fit the whole head
    /// chunk, the lowest slot id wins (deterministic: every fitting lane
    /// serves identical rows). A disjoint next chunk may fill while a
    /// previous READY chunk is still outstanding; run_fill rechecks head,
    /// pins, and frontier before staging, so advice never authorizes a
    /// stale row.
    pub fn advise_fill(&self, free_slot_ids: &[u64]) -> Result<Option<u64>, PoolError> {
        self.harvest_applies();
        {
            let mut state = self.lock_state()?;
            RoundPlan::maybe_retire(&mut state);
        }
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        if state.drained {
            return Err(PoolError("pool trajectories already drained".to_string()));
        }
        if state.quiesced {
            state.advise_none_quiesced = state.advise_none_quiesced.saturating_add(1);
            return Ok(None);
        }
        if free_slot_ids.is_empty() {
            state.advise_none_owner_empty = state.advise_none_owner_empty.saturating_add(1);
            return Ok(None);
        }
        let free: Vec<u64> = free_slot_ids
            .iter()
            .copied()
            .filter(|id| {
                state.slots.get(id).is_some_and(|s| s.phase == SlotPhase::Free)
            })
            .collect();
        if free.is_empty() {
            state.advise_none_no_truly_free = state.advise_none_no_truly_free.saturating_add(1);
            return Ok(None);
        }
        // Never nest state/game locks: the freeze scans games without the
        // state lock held, then re-locks to serve the frozen head.
        drop(state);
        if !self.ensure_round_plan()? {
            return Ok(None);
        }
        let state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        let plan = match state.round_plan.as_ref() {
            Some(plan) => plan,
            None => return Ok(None),
        };
        let head = match plan.head_bucket() {
            Some(head) => head,
            // Exhausted but unsettled (leases/applies/READY outstanding):
            // the round is not complete, so no new fill can help yet.
            None => return Ok(None),
        };
        let head_t = HISTORY_BUCKETS[head];
        let head_rows = match plan.head_rows(head_t)? {
            Some((_, rows)) => rows,
            // Head refs exist but no lane ever registered this bucket:
            // wait for the first registration to freeze the chunk size.
            None => return Ok(None),
        };
        if head_rows.is_empty() {
            return Ok(None);
        }
        let pinned = RoundPlan::pins(&state);
        // A chunk naming a pinned game waits unchanged: rows stay frozen
        // for the next fill instead of splitting around the pin.
        if head_rows.iter().any(|row| pinned[row.game_idx] != 0) {
            return Ok(None);
        }
        // Larger fitting lanes may serve the frozen chunk; smaller lanes
        // wait for a chunk that fits them (never a shrunk repack).
        let mut fitting: Vec<u64> = free
            .iter()
            .copied()
            .filter(|id| {
                state.slots.get(id).is_some_and(|slot| {
                    slot.phase == SlotPhase::Free
                        && slot.bucket_t == head_t
                        && slot.capacity >= head_rows.len()
                })
            })
            .collect();
        fitting.sort_unstable();
        Ok(fitting.into_iter().next())
    }

    /// Stage exactly the frozen head chunk for this lane's bucket. Lanes on
    /// other buckets, lanes smaller than the chunk, and chunks naming a
    /// pinned game all wait empty (never bypass, shrink, or skip). Sampler
    /// reservations still bind once per real staged decision with the same
    /// legal/observation/bucket checks; a terminal game holding an unissued
    /// head ref fails the fill loud (poison at install, never a silent
    /// drop). The head advances only at READY install, never here.
    pub(super) fn run_fill(&self, slot_id: u64, epoch: u64, bucket_t: usize, capacity: usize) -> FillOutput {
        let failed = |detail: String| FillOutput {
            slot_id,
            epoch,
            bucket_t,
            staged: Vec::new(),
            bytes: Vec::new(),
            error: Some(detail),
        };
        let empty = || FillOutput {
            slot_id,
            epoch,
            bucket_t,
            staged: Vec::new(),
            bytes: Vec::new(),
            error: None,
        };
        self.harvest_applies();
        {
            let mut state = match self.lock_state() {
                Ok(state) => state,
                Err(e) => return failed(e.0),
            };
            RoundPlan::maybe_retire(&mut state);
        }
        match self.ensure_round_plan() {
            Ok(true) => {}
            Ok(false) => return empty(),
            Err(e) => return failed(e.0),
        }
        // Load the head chunk plus pins under one lock; the chunk rows are
        // the only copy across the lock boundary (cap-bounded, never the
        // full plan). Locks never nest.
        struct HeadChunk {
            round: u64,
            rows: Vec<PlanRow>,
        }
        let head: Option<HeadChunk> = match self.lock_state() {
            Ok(state) => {
                if let Err(e) = self.fail_if_poisoned(&state) {
                    return failed(e.0);
                }
                let plan = match state.round_plan.as_ref() {
                    Some(plan) => plan,
                    None => return empty(),
                };
                let pos = match RoundPlan::bucket_pos(bucket_t) {
                    Ok(pos) => pos,
                    Err(e) => return failed(e.0),
                };
                if plan.head_bucket() != Some(pos) {
                    return empty();
                }
                let (round, rows) = match plan.head_rows(bucket_t) {
                    Ok(Some((_, rows))) => (plan.round(), rows.to_vec()),
                    Ok(None) => return empty(),
                    Err(e) => return failed(e.0),
                };
                if rows.len() > capacity {
                    return empty();
                }
                let pinned = RoundPlan::pins(&state);
                if rows.iter().any(|row| pinned[row.game_idx] != 0) {
                    return empty();
                }
                Some(HeadChunk { round, rows })
            }
            Err(e) => return failed(e.0),
        };
        let Some(chunk) = head else {
            return empty();
        };
        let logical_round = chunk.round;
        // Group only this frozen round, in fixed cohort order.
        let mut observations = Vec::with_capacity(capacity);
        let sender = match self.lock_workers().and_then(|workers| {
            workers
                .sender
                .clone()
                .ok_or_else(|| PoolError("workers shut down".to_string()))
        }) {
            Ok(sender) => sender,
            Err(e) => return failed(e.0),
        };
        let mut staged: Vec<FilledRow> = vec![];
        // Stage exactly the frozen chunk refs (already canonical
        // (game, seat) order, so same-game seats form contiguous runs).
        // Reservations stay actual-stage-only: only seats staged here draw
        // sampler words, with the same per-seat checks as before.
        let mut cursor = 0usize;
        while cursor < chunk.rows.len() {
            let game_idx = chunk.rows[cursor].game_idx;
            let mut run_end = cursor;
            while run_end < chunk.rows.len() && chunk.rows[run_end].game_idx == game_idx {
                run_end += 1;
            }
            // Game guard is scoped: it drops before the sampler-counter lock
            // below (locks are never nested game -> counter).
            let seats: Vec<u8> = match self.lock_game(game_idx) {
                Ok(mut game) => {
                    if game.is_terminal() {
                        return failed(format!(
                            "game {} terminal with unissued frozen ref",
                            self.wall_ids[game_idx]
                        ));
                    }
                    if game.frontier_count() != logical_round {
                        return failed(format!(
                            "game {} frontier drift from frozen round {logical_round}",
                            self.wall_ids[game_idx]
                        ));
                    }
                    let decision = match game.drain() {
                        Ok(decision) => decision,
                        Err(e) => {
                            return failed(format!(
                                "game {} drain rejected: {e}",
                                self.wall_ids[game_idx]
                            ));
                        }
                    };
                    let pending: Vec<u8> = match decision {
                        None => vec![],
                        Some(PendingDecision::Draw { seat }) => vec![seat],
                        Some(PendingDecision::Window { pending }) => pending,
                    };
                    let mut rows = vec![];
                    for row in &chunk.rows[cursor..run_end] {
                        let seat = row.actor;
                        if !pending.contains(&seat) {
                            return failed(format!(
                                "game {} seat {seat} left the frozen chunk",
                                self.wall_ids[game_idx]
                            ));
                        }
                        let bucket = match bucket_ceil(
                            game.arena()
                                .visible_count_for_hand(seat, game.arena().len()),
                        ) {
                            Ok(bucket) => bucket,
                            Err(e) => {
                                return failed(format!(
                                    "game {} over-cap history: {e}",
                                    self.wall_ids[game_idx]
                                ));
                            }
                        };
                        if self.live_binding.is_some() {
                            if bucket != row.original_bucket {
                                return failed(format!(
                                    "game {} seat {seat} original bucket drift from {}",
                                    self.wall_ids[game_idx], row.original_bucket
                                ));
                            }
                            if bucket_t != 256 {
                                return failed(format!(
                                    "game {} seat {seat} live transport must be T256",
                                    self.wall_ids[game_idx]
                                ));
                            }
                        } else if bucket != bucket_t || bucket != row.original_bucket {
                            return failed(format!(
                                "game {} seat {seat} bucket drift from T={bucket_t}",
                                self.wall_ids[game_idx]
                            ));
                        }
                        if RoundPlan::decision_seq(&game, seat) != row.seq {
                            return failed(format!(
                                "game {} seat {seat} decision turned over since freeze",
                                self.wall_ids[game_idx]
                            ));
                        }
                        rows.push(seat);
                    }
                    rows
                }
                Err(e) => return failed(e.0),
            };
            for seat in seats {
                let (respond, receiver) = std::sync::mpsc::sync_channel(1);
                if sender
                    .send(WorkerJob::Observe {
                        game_idx,
                        seat,
                        respond,
                    })
                    .is_err()
                {
                    return failed("observation worker queue closed".to_string());
                }
                observations.push((game_idx, seat, receiver));
            }
            cursor = run_end;
        }
        // Eligible jobs are selected in cohort order before sampler reservation;
        // ordered receives preserve rows, and a lost worker rejects the fill.
        for (game_idx, seat, receiver) in observations {
            let obs = match receiver.recv() {
                Ok(Ok(obs)) => obs,
                Ok(Err(detail)) => return failed(detail),
                Err(_) => return failed("observation worker lost".to_string()),
            };
            // Original observation bucket stays true: the staged observation's
            // real history length must ceil to the frozen `original_bucket`
            // (derived from `history_indices.len()`); transport T256 merely
            // permits `<=256` and never authorizes skipping gates.
            let actual_bucket = match super::bucket_ceil(obs.history_indices.len()) {
                Ok(bucket) => bucket,
                Err(e) => return failed(format!("game {game_idx} over-cap history: {e}")),
            };
            let expect = chunk.rows.iter().find(|row| row.game_idx == game_idx && row.actor == seat);
            match expect {
                Some(row) => {
                    if actual_bucket != row.original_bucket {
                        return failed(format!(
                            "game {} seat {seat} original bucket drift from {}",
                            self.wall_ids[game_idx], row.original_bucket
                        ));
                    }
                    if obs.sequence != row.seq {
                        return failed(format!(
                            "game {} seat {seat} decision turned over since freeze",
                            self.wall_ids[game_idx]
                        ));
                    }
                    if self.live_binding.is_some() && bucket_t != 256 {
                        return failed(format!(
                            "game {} seat {seat} live transport must be T256",
                            self.wall_ids[game_idx]
                        ));
                    }
                }
                None => {
                    return failed(format!(
                        "game {} seat {seat} outside frozen chunk",
                        self.wall_ids[game_idx]
                    ));
                }
            }
            // One reservation per real decision on the seat's own stream
            // (counter advances exactly once here, never rewound, even
            // if the row is later dropped by quiesce: skipped words are
            // never reused, so there is no duplication).
            let counter = match self.lock_counter(game_idx) {
                Ok(mut guards) => {
                    let current = guards[seat as usize];
                    guards[seat as usize] = current.saturating_add(1);
                    current
                }
                Err(e) => return failed(e.0),
            };
            let seed = match self
                .sample_seeds
                .get(game_idx)
                .and_then(|seats| seats.get(seat as usize))
            {
                Some(seed) => seed,
                None => return failed(format!("game {game_idx} sampler seed missing")),
            };
            let (cursor_before, uniform) = match super::sampler_draw(seed, counter) {
                Ok(draw) => draw,
                Err(e) => return failed(e.0),
            };
            staged.push(FilledRow {
                game_idx,
                actor: seat,
                obs,
                uniform,
                cursor_before,
            });
        }
        if staged.is_empty() {
            return FillOutput {
                slot_id,
                epoch,
                bucket_t,
                staged,
                bytes: Vec::new(),
                error: None,
            };
        }
        let n_valid = staged.len();
        let mut bytes = vec![0u8; observation::active_input_bytes(bucket_t, n_valid)];
        if let Err(e) = self.encode_staged(&staged, bucket_t, &mut bytes) {
            return failed(e.0);
        }
        FillOutput {
            slot_id,
            epoch,
            bucket_t,
            staged,
            bytes,
            error: None,
        }
    }

    /// Shared typed encode: plane split (aligned, gaps zeroed) + oracle
    /// sentinels + one `fill_row` per staged row. Used by fill workers and
    /// by `refill_staged` (byte-identical re-encode after restore).
    fn encode_staged(
        &self,
        staged: &[FilledRow],
        bucket_t: usize,
        out: &mut [u8],
    ) -> Result<(), PoolError> {
        let n_valid = staged.len();
        let mut planes = observation::split_planes(out, bucket_t, n_valid, true)
            .map_err(|e| PoolError(format!("plane split rejected: {e}")))?;
        // Pre-fill sentinels: 0x00 except dora/own_drawn 0xFF (mirrors the
        // oracle `np.empty + fill` contract; Rust fills valid prefixes).
        for byte in planes.dora.iter_mut() {
            *byte = 0xFF;
        }
        for byte in planes.own_drawn.iter_mut() {
            *byte = 0xFF;
        }
        for (row, filled) in staged.iter().enumerate() {
            let game = self.lock_game(filled.game_idx)?;
            observation::fill_row(
                &mut planes,
                row,
                &filled.obs,
                game.arena(),
                bucket_t,
                filled.uniform,
                true,
            )
            .map_err(|e| PoolError(format!("row {row} fill rejected: {e}")))?;
        }
        Ok(())
    }

    /// Deliver one worker fill payload (taking thread; installs or fails
    /// loud — a failed install frees the lane but never rewinds sampler
    /// counters, so reserved words are skipped, never duplicated).
    fn install_fill(
        &self,
        output: FillOutput,
        input: &mut [u8],
    ) -> Result<TicketDescriptor, PoolError> {
        if let Some(detail) = output.error {
            self.release_to_free(output.slot_id)?;
            self.clear_inflight(output.slot_id)?;
            self.poison(format!("fill for slot {} failed: {detail}", output.slot_id))?;
            return Err(PoolError(format!(
                "fill for slot {} failed: {detail}",
                output.slot_id
            )));
        }
        let n_valid = output.staged.len();
        if n_valid == 0 {
            self.release_to_free(output.slot_id)?;
            self.clear_inflight(output.slot_id)?;
            if let Ok(mut state) = self.lock_state() {
                self.notify(&mut state);
            }
            return Ok(TicketDescriptor {
                slot_id: output.slot_id,
                slot_epoch: output.epoch,
                generation: self.profile.generation,
                logical_round: 0,
                policy_id: self.profile.policy_id.clone(),
                bucket_t: output.bucket_t,
                chunk: 0,
                n_valid: 0,
                game_ids: vec![],
                cursors_before: vec![],
            });
        }
        let active = observation::active_input_bytes(output.bucket_t, n_valid);
        if input.len() < active {
            self.release_to_free(output.slot_id)?;
            self.clear_inflight(output.slot_id)?;
            return Err(PoolError(format!(
                "input slab {} < active {active} for B={n_valid} T={}",
                input.len(),
                output.bucket_t
            )));
        }
        if output.bytes.len() < active {
            self.release_to_free(output.slot_id)?;
            self.clear_inflight(output.slot_id)?;
            return Err(PoolError(
                "fill payload shorter than its active prefix".to_string(),
            ));
        }
        let game_ids: Vec<String> = output
            .staged
            .iter()
            .map(|r| self.wall_ids[r.game_idx].clone())
            .collect();
        let cursors_before: Vec<u64> = output.staged.iter().map(|r| r.cursor_before).collect();
        let (chunk, round) = {
            let mut state = self.lock_state()?;
            // Clear the matching inflight entry when present (nonblocking
            // take leaves it in place; the blocking take removed it before
            // recv — both paths converge here exactly once per admission).
            // A DIFFERENT inflight entry here means a newer admission
            // superseded this payload: fail loud, never double-install.
            match state.fill_inflight.as_ref() {
                Some(inflight)
                    if inflight.slot_id == output.slot_id && inflight.epoch == output.epoch => {}
                Some(inflight) => {
                    return Err(PoolError(format!(
                        "slot {} fill superseded by slot {}",
                        output.slot_id, inflight.slot_id
                    )));
                }
                None => {}
            }
            // Phase check under a short borrow so the chunk table can be
            // mutated without a live `&mut Slot`.
            {
                let slot = slot_mut(&mut state, output.slot_id)?;
                if slot.phase != SlotPhase::Filling || slot.epoch != output.epoch {
                    return Err(PoolError(format!(
                        "slot {} left FILLING before take",
                        output.slot_id
                    )));
                }
            }
            // The head advances EXACTLY once here, at READY installation
            // (never at advice or staging): the payload must be the frozen
            // head chunk for its bucket, in plan order.
            let plan = state.round_plan.as_mut().ok_or_else(|| {
                PoolError(format!(
                    "slot {} fill installed with no active round plan",
                    output.slot_id
                ))
            })?;
            let pos = RoundPlan::bucket_pos(output.bucket_t).map_err(|e| {
                PoolError(format!(
                    "slot {} fill installed outside plan buckets: {e}",
                    output.slot_id
                ))
            })?;
            if plan.head_bucket() != Some(pos) {
                return Err(PoolError(format!(
                    "slot {} fill installed for non-head bucket T={}",
                    output.slot_id, output.bucket_t
                )));
            }
            let head_rows = plan.head_rows(output.bucket_t)?.ok_or_else(|| {
                PoolError(format!(
                    "slot {} fill installed past plan issuance",
                    output.slot_id
                ))
            })?;
            if head_rows.1.len() != n_valid
                || head_rows
                    .1
                    .iter()
                    .zip(output.staged.iter())
                    .any(|(expect, got)| {
                        expect.game_idx != got.game_idx
                            || expect.actor != got.actor
                            || expect.seq != got.obs.sequence
                            || super::bucket_ceil(got.obs.history_indices.len()).unwrap_or(0) != expect.original_bucket
                    })
            {
                return Err(PoolError(format!(
                    "slot {} fill rows drift from frozen head chunk",
                    output.slot_id
                )));
            }
            let chunk = plan.advance(output.bucket_t, n_valid)?;
            let round = plan.round();
            // Install rows under a fresh short borrow (no other `state`
            // borrow is live across this move).
            {
                let slot = slot_mut(&mut state, output.slot_id)?;
                slot.rows = output.staged;
                slot.phase = SlotPhase::Ready;
                slot.chunk = chunk;
                slot.logical_round = round;
            }
            state.fill_inflight = None;
            RoundPlan::maybe_retire(&mut state);
            self.notify(&mut state);
            (chunk, round)
        };
        input[..active].copy_from_slice(&output.bytes[..active]);
        Ok(TicketDescriptor {
            slot_id: output.slot_id,
            slot_epoch: output.epoch,
            generation: self.profile.generation,
            logical_round: round,
            policy_id: self.profile.policy_id.clone(),
            bucket_t: output.bucket_t,
            chunk,
            n_valid,
            game_ids,
            cursors_before,
        })
    }

    /// Nonblocking take of the admitted fill (`None` = worker still running;
    /// the owner does GPU work and retries — never a sleep-spin, never an
    /// allocation on the miss path: gate with `fill_ready` first).
    pub fn take_fill(
        self: &Arc<Self>,
        slot_id: u64,
        input: &mut [u8],
    ) -> Result<Option<TicketDescriptor>, PoolError> {
        self.harvest_applies();
        let output = {
            let mut state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            match state.fill_inflight.take() {
                Some(inflight) if inflight.slot_id == slot_id => match inflight.result {
                    Some(output) => output,
                    None => {
                        state.fill_inflight = Some(inflight);
                        return Ok(None);
                    }
                },
                Some(inflight) => {
                    let other = inflight.slot_id;
                    state.fill_inflight = Some(inflight);
                    return Err(PoolError(format!(
                        "slot {slot_id} fill not in flight (slot {other} is)"
                    )));
                }
                None => {
                    return Err(PoolError(format!("slot {slot_id} has no admitted fill")));
                }
            }
        };
        self.install_fill(output, input).map(Some)
    }

    /// Blocking take (compat wrapper and quiesce/drain tooling only; never
    /// the hot loop). Waits on the progress condvar with the lock released
    /// by the waiter and no slab allocated until the payload lands.
    pub fn take_fill_blocking(
        self: &Arc<Self>,
        slot_id: u64,
        input: &mut [u8],
    ) -> Result<TicketDescriptor, PoolError> {
        let output = {
            let mut state = self.lock_state()?;
            loop {
                self.fail_if_poisoned(&state)?;
                match state.fill_inflight.take() {
                    Some(inflight) if inflight.slot_id == slot_id => match inflight.result {
                        Some(output) => break output,
                        None => {
                            state.fill_inflight = Some(inflight);
                            state = self
                                .progress_cv
                                .wait(state)
                                .map_err(|_| PoolError("progress condvar poisoned".to_string()))?;
                        }
                    },
                    Some(inflight) => {
                        let other = inflight.slot_id;
                        state.fill_inflight = Some(inflight);
                        return Err(PoolError(format!(
                            "slot {slot_id} fill not in flight (slot {other} is)"
                        )));
                    }
                    None => {
                        return Err(PoolError(format!("slot {slot_id} has no admitted fill")));
                    }
                }
            }
        };
        self.install_fill(output, input)
    }

    /// Re-encode retained staged rows into the caller slab (restore path:
    /// re-registers buffers, then refills byte-identical inputs for adopted
    /// READY lanes without reserving new sampler words). FREE lanes (no
    /// retained rows, e.g. post-restore sweeps over every lane) return 0
    /// instead of erroring, so owners loop uniformly.
    pub fn refill_staged(
        self: &Arc<Self>,
        slot_id: u64,
        input: &mut [u8],
    ) -> Result<usize, PoolError> {
        self.harvest_applies();
        let (bucket_t, staged) = {
            let state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            let slot = slot_ref(&state, slot_id)?;
            if slot.phase == SlotPhase::Free {
                return Ok(0);
            }
            if slot.phase != SlotPhase::Ready {
                return Err(PoolError(format!("slot {slot_id} not READY for refill")));
            }
            (slot.bucket_t, slot.rows.len())
        };
        if staged == 0 {
            return Ok(0);
        }
        let active = observation::active_input_bytes(bucket_t, staged);
        if input.len() < active {
            return Err(PoolError(format!(
                "input slab {} < active {active} for B={staged} T={bucket_t}",
                input.len()
            )));
        }
        // Clone the staged rows' facts under one short lock, then encode
        // without holding state (game locks only, never nested in state).
        // `cursor_before: 0` is a safe placeholder: `encode_staged` reads
        // obs/uniform only, so refill restores bytes, never reservations.
        let rows: Vec<(usize, NativeObservation, f64)> = {
            let state = self.lock_state()?;
            slot_ref(&state, slot_id)?
                .rows
                .iter()
                .map(|r| (r.game_idx, r.obs.clone(), r.uniform))
                .collect()
        };
        let scratch: Vec<FilledRow> = rows
            .into_iter()
            .map(|(game_idx, obs, uniform)| FilledRow {
                game_idx,
                actor: obs.actor,
                obs,
                uniform,
                cursor_before: 0,
            })
            .collect();
        self.encode_staged(&scratch, bucket_t, &mut input[..active])?;
        Ok(staged)
    }

    /// Admission-scheduled typed fill into the caller slab (compat wrapper:
    /// admit + blocking take). Prefer `admit_fill` + `take_fill` on the hot
    /// loop so the owner overlaps GPU work with native encoding. Empty fills
    /// (`n_valid == 0`) leave the slot FREE for work-conserving stalls (no
    /// blocking, no dummy rows).
    pub fn request_ready_fill(
        self: &Arc<Self>,
        slot_id: u64,
        input: &mut [u8],
        output_capacity: usize,
    ) -> Result<TicketDescriptor, PoolError> {
        self.admit_fill(slot_id, output_capacity)?;
        self.take_fill_blocking(slot_id, input)
    }

    fn release_to_free(&self, slot_id: u64) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        slot_mut(&mut state, slot_id)?.rows.clear();
        slot_mut(&mut state, slot_id)?.phase = SlotPhase::Free;
        self.notify(&mut state);
        Ok(())
    }

    /// Drop the in-flight fill record for one slot (after install, abort,
    /// or worker loss; the slot phase itself is handled by the caller).
    fn clear_inflight(&self, slot_id: u64) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        if state
            .fill_inflight
            .as_ref()
            .is_some_and(|inflight| inflight.slot_id == slot_id)
        {
            state.fill_inflight = None;
            self.notify(&mut state);
        }
        Ok(())
    }
}
