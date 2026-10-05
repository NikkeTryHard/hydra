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
use super::state::{FillInflight, FillOutput, FilledRow, WorkerJob, slot_mut, slot_ref};
use super::{NativePool, PoolError, SlotPhase, TicketDescriptor, bucket_ceil, buffer_geometry};

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

    /// Advisory fill routing for the collector: among the owner's FREE
    /// lanes (intersected with truly-Free lanes here), prefer lanes with
    /// recent row yield and rotate within the preferred set; when every
    /// free lane recently yielded empty, rotate through all of them
    /// (conditions change as commits land, so empty lanes are retried,
    /// never parked). Returns `None` when no lane is free or the pool is
    /// quiesced (expected stall, never poison).
    pub fn advise_fill(&self, free_slot_ids: &[u64]) -> Result<Option<u64>, PoolError> {
        self.harvest_applies();
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
        let mut free: Vec<u64> = free_slot_ids
            .iter()
            .copied()
            .filter(|id| {
                state
                    .slots
                    .get(id)
                    .is_some_and(|s| s.phase == SlotPhase::Free)
            })
            .collect();
        if free.is_empty() {
            state.advise_none_no_truly_free = state.advise_none_no_truly_free.saturating_add(1);
            return Ok(None);
        }
        free.sort_unstable();
        let productive: Vec<u64> = free
            .iter()
            .copied()
            .filter(|id| state.fill_yield.get(id).is_none_or(|yielded| *yielded > 0))
            .collect();
        let set = if productive.is_empty() {
            free
        } else {
            productive
        };
        let cursor = state.advise_cursor as usize % set.len();
        state.advise_cursor = state.advise_cursor.wrapping_add(1);
        Ok(Some(set[cursor]))
    }

    /// Freeze one next canonical decision per nonterminal cohort game at the
    /// minimum committed-decision count. Pins prevent duplicate staging;
    /// outstanding applies at that frontier prevent the next round opening.
    /// This keeps real batch membership independent of worker completion.
    /// Staged rows are capped at the lane `capacity` admit_fill copied from
    /// the slot; excess eligible games stay for the next fill.
    pub(super) fn run_fill(&self, slot_id: u64, epoch: u64, bucket_t: usize, capacity: usize) -> FillOutput {
        let failed = |detail: String| FillOutput {
            slot_id,
            epoch,
            bucket_t,
            staged: Vec::new(),
            bytes: Vec::new(),
            logical_round: 0,
            error: Some(detail),
        };
        self.harvest_applies();
        // Locks never nest: snapshot staged/apply pins before reading games.
        // Separate bits retain the apply barrier without a second allocation.
        let pinned: Vec<u8> = match self.lock_state() {
            Ok(state) => {
                if let Err(e) = self.fail_if_poisoned(&state) {
                    return failed(e.0);
                }
                let mut pinned = vec![0u8; self.games.len()];
                for slot in state.slots.values() {
                    if slot.phase == SlotPhase::Free {
                        continue;
                    }
                    for row in &slot.rows {
                        pinned[row.game_idx] |= 1;
                    }
                }
                for (idx, count) in state.apply_outstanding.iter().enumerate() {
                    if *count > 0 {
                        pinned[idx] |= 2;
                    }
                }
                pinned
            }
            Err(e) => return failed(e.0),
        };
        let mut logical_round = u64::MAX;
        let mut applying_frontier = u64::MAX;
        for (game_idx, pins) in pinned.iter().enumerate() {
            let game = match self.lock_game(game_idx) {
                Ok(game) => game,
                Err(e) => return failed(e.0),
            };
            let committed = game.captured_count();
            if !game.is_terminal() {
                logical_round = logical_round.min(committed);
            }
            if pins & 2 != 0 {
                applying_frontier = applying_frontier.min(committed);
            }
        }
        if logical_round == u64::MAX || applying_frontier <= logical_round {
            return FillOutput {
                slot_id,
                epoch,
                bucket_t,
                staged: Vec::new(),
                bytes: Vec::new(),
                logical_round: 0,
                error: None,
            };
        }
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
        for game_idx in 0..self.games.len() {
            if observations.len() >= capacity {
                break;
            }
            if pinned[game_idx] != 0 {
                continue;
            }
            // Game guard is scoped: it drops before the sampler-counter lock
            // below (locks are never nested game -> counter).
            let seats: Vec<u8> = match self.lock_game(game_idx) {
                Ok(mut game) => {
                    if game.is_terminal() || game.captured_count() != logical_round {
                        continue;
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
                    let seats: Vec<u8> = match decision {
                        None => vec![],
                        Some(PendingDecision::Draw { seat }) => vec![seat],
                        Some(PendingDecision::Window { pending }) => pending,
                    };
                    let mut rows = vec![];
                    for seat in seats.into_iter().take(1) {
                        if seat > 3 {
                            return failed(format!(
                                "game {} decision seat {seat} outside 0..=3",
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
                        if bucket != bucket_t {
                            continue;
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
        }
        // Eligible jobs are selected in cohort order before sampler reservation;
        // ordered receives preserve rows, and a lost worker rejects the fill.
        for (game_idx, seat, receiver) in observations {
            let obs = match receiver.recv() {
                Ok(Ok(obs)) => obs,
                Ok(Err(detail)) => return failed(detail),
                Err(_) => return failed("observation worker lost".to_string()),
            };
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
                logical_round: 0,
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
            logical_round,
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
                state.fill_yield.insert(output.slot_id, 0);
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
        let chunk = {
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
            let entry = state.next_chunk.entry(output.bucket_t).or_insert(0);
            let chunk = *entry;
            *entry += 1;
            // Install rows under a fresh short borrow (no other `state`
            // borrow is live across this move).
            {
                let slot = slot_mut(&mut state, output.slot_id)?;
                slot.rows = output.staged;
                slot.phase = SlotPhase::Ready;
                slot.chunk = chunk;
                slot.logical_round = output.logical_round;
            }
            state.fill_inflight = None;
            state.fill_yield.insert(output.slot_id, n_valid);
            self.notify(&mut state);
            chunk
        };
        input[..active].copy_from_slice(&output.bytes[..active]);
        Ok(TicketDescriptor {
            slot_id: output.slot_id,
            slot_epoch: output.epoch,
            generation: self.profile.generation,
            logical_round: output.logical_round,
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
