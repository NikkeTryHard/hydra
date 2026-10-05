//! rl/pool/apply: leases, phases, commits, worker harvest/join, lifecycle.
//!
//! Invariant: `commit_actions` validates the ENTIRE batch (length, ownership,
//! legal membership, finite logp/value4, `0 < exp(logp) <= 1`,
//! epoch/generation/policy identity under the lease binding) BEFORE any
//! capture; the sampler cursor binds verbatim, never matched. The slot flips
//! to `OwnedCapture` only after validation; captured rows are retained per
//! game before any worker apply is admitted, and every apply channel is registered
//! with its outstanding count BEFORE the slot releases, so an admitted fill
//! can never stage a decision these applies have not landed. Completions are
//! harvested nonblocking; only quiesce/snapshot/close join. Failure mode: any
//! validation miss, worker error, lost worker, or post-apply `last_error`
//! poisons the generation sticky-loud (surfaces on poll/drain/close/commit).

use std::sync::{Arc, mpsc};

use super::super::actions::{BASELINE_ACTIONS, CanonicalAction};
use super::super::core::GameRow;
use super::super::observation::NativeObservation;
use super::state::{ApplyJob, PendingApply, WorkerJob, slot_mut, slot_ref};
use super::{DrainStatus, Lease, NativePool, PoolError, PoolStatus, SlotPhase, output_plane_bases};

impl NativePool {
    /// Nonblocking harvest of admitted worker applies: completed jobs retire
    /// their game's outstanding count; worker errors, lost workers, and
    /// post-apply `last_error` findings poison the generation loudly. Never
    /// blocks, never takes game locks (workers own those). Call at the top
    /// of every owner entry point so completions surface on poll/drain
    /// without the owner ever waiting on work it must release.
    pub(super) fn harvest_applies(&self) {
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(_) => return,
        };
        let mut index = 0usize;
        while index < state.pending_applies.len() {
            let outcome = state.pending_applies[index].receiver.try_recv();
            match outcome {
                Ok(Ok(())) => {
                    let done = state.pending_applies.remove(index);
                    if let Some(count) = state.apply_outstanding.get_mut(done.game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    state.progress += 1;
                }
                Ok(Err(reason)) => {
                    let game_idx = state.pending_applies[index].game_idx;
                    state.pending_applies.remove(index);
                    if let Some(count) = state.apply_outstanding.get_mut(game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    self.poison_locked(
                        &mut state,
                        format!("game {game_idx} worker failed: {reason}"),
                    );
                }
                Err(mpsc::TryRecvError::Empty) => index += 1,
                Err(mpsc::TryRecvError::Disconnected) => {
                    let game_idx = state.pending_applies[index].game_idx;
                    state.pending_applies.remove(index);
                    if let Some(count) = state.apply_outstanding.get_mut(game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    self.poison_locked(&mut state, format!("game {game_idx} worker lost"));
                }
            }
        }
        self.progress_cv.notify_all();
    }

    /// Blocking join of admitted applies (quiesce/snapshot/close paths only;
    /// never the hot loop). Retains unissued READY rows: joining only lands
    /// already-captured decisions, it never stages, drops, or resamples.
    /// First worker failure poisons and aborts the join loudly.
    pub(super) fn join_applies(&self) -> Result<(), PoolError> {
        loop {
            let pending = {
                let mut state = self.lock_state()?;
                match state.pending_applies.pop() {
                    Some(pending) => pending,
                    None => return Ok(()),
                }
            };
            match pending.receiver.recv() {
                Ok(Ok(())) => {
                    let mut state = self.lock_state()?;
                    if let Some(count) = state.apply_outstanding.get_mut(pending.game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    self.notify(&mut state);
                }
                Ok(Err(reason)) => {
                    let mut state = self.lock_state()?;
                    if let Some(count) = state.apply_outstanding.get_mut(pending.game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    let detail = format!("game {} worker failed: {reason}", pending.game_idx);
                    self.poison_locked(&mut state, detail.clone());
                    return Err(PoolError(detail));
                }
                Err(_) => {
                    let mut state = self.lock_state()?;
                    if let Some(count) = state.apply_outstanding.get_mut(pending.game_idx) {
                        *count = count.saturating_sub(1);
                    }
                    let detail = format!("game {} worker lost", pending.game_idx);
                    self.poison_locked(&mut state, detail.clone());
                    return Err(PoolError(detail));
                }
            }
        }
    }

    /// Nonblocking scan of ALL READY unleased tickets; min eligible
    /// `(generation, logical_round, policy_id, bucket_t, chunk; slot id
    /// breaks ties)`. Never waits for a FILLING lower-T lane
    pub fn peek_ready(self: &Arc<Self>) -> Result<Option<Lease>, PoolError> {
        self.harvest_applies();
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        let mut best: Option<(u64, u64, String, usize, u64, u64)> = None;
        for slot in state.slots.values() {
            if slot.phase != SlotPhase::Ready || slot.leased {
                continue;
            }
            let key = (
                self.profile.generation,
                slot.logical_round,
                self.profile.policy_id.clone(),
                slot.bucket_t,
                slot.chunk,
                slot.id,
            );
            if best.as_ref().is_none_or(|current| key < *current) {
                best = Some(key);
            }
        }
        let slot_id = match best {
            None => return Ok(None),
            Some((_, _, _, _, _, id)) => id,
        };
        // Copy the lease facts under a short slot borrow, then release it
        // before mutating the lease table (a live `&mut Slot` would forbid
        // any other `state` borrow).
        let (epoch, bucket_t, chunk, n_valid, logical_round) = {
            let slot = slot_mut(&mut state, slot_id)?;
            slot.leased = true;
            (
                slot.epoch,
                slot.bucket_t,
                slot.chunk,
                slot.rows.len(),
                slot.logical_round,
            )
        };
        let lease_id = state.next_lease;
        state.next_lease += 1;
        let lease = Lease {
            lease_id,
            slot_id,
            slot_epoch: epoch,
            generation: self.profile.generation,
            policy_id: self.profile.policy_id.clone(),
            bucket_t,
            chunk,
            n_valid,
            logical_round,
        };
        state.leases.insert(lease_id, lease.clone());
        self.notify(&mut state);
        Ok(Some(lease))
    }

    pub(super) fn lease_checked(
        &self,
        state: &super::state::PoolState,
        lease: &Lease,
    ) -> Result<(), PoolError> {
        self.fail_if_poisoned(state)?;
        let live = state.leases.get(&lease.lease_id).ok_or_else(|| {
            PoolError(format!(
                "lease {} unknown (stale or double commit)",
                lease.lease_id
            ))
        })?;
        if live != lease {
            return Err(PoolError(format!(
                "lease {} identity drift",
                lease.lease_id
            )));
        }
        let slot = slot_ref(state, lease.slot_id)?;
        if slot.epoch != lease.slot_epoch {
            return Err(PoolError(format!(
                "lease {} slot epoch drift",
                lease.lease_id
            )));
        }
        if lease.generation != self.profile.generation || lease.policy_id != self.profile.policy_id
        {
            return Err(PoolError(format!(
                "lease {} generation/policy drift",
                lease.lease_id
            )));
        }
        Ok(())
    }

    /// Return an unissued reservation without dropping its READY rows or
    /// changing their uniforms/epoch; issued work cannot be parked for resume.
    pub fn return_ready_lease(&self, lease: &Lease) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        self.lease_checked(&state, lease)?;
        let slot = slot_mut(&mut state, lease.slot_id)?;
        if slot.phase != SlotPhase::Ready {
            return Err(PoolError(
                "only an unissued READY lease can be returned".to_string(),
            ));
        }
        slot.leased = false;
        state.leases.remove(&lease.lease_id);
        self.notify(&mut state);
        Ok(())
    }

    /// Record owner-observed transfer/compute phases (order-enforced).
    pub fn mark_phase(&self, lease: &Lease, phase: SlotPhase) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        self.lease_checked(&state, lease)?;
        let slot = slot_mut(&mut state, lease.slot_id)?;
        let allowed = matches!(
            (slot.phase, phase),
            (SlotPhase::Ready, SlotPhase::H2dIssued)
                | (SlotPhase::H2dIssued, SlotPhase::InferQueued)
                | (SlotPhase::InferQueued, SlotPhase::D2hIssued)
                | (SlotPhase::D2hIssued, SlotPhase::HostReady)
        );
        if !allowed {
            return Err(PoolError(format!(
                "lease {} phase {:?} -> {:?} rejected",
                lease.lease_id, slot.phase, phase
            )));
        }
        slot.phase = phase;
        self.notify(&mut state);
        Ok(())
    }

    /// Commit GPU outputs: validate the ENTIRE batch first (length, finite,
    /// `0 < exp(logp) <= 1`, legal membership, epoch/generation/policy identity),
    /// capture owned rows against the live plans, retain them per game for
    /// the generation-end tape handoff, release the slot immediately,
    /// enqueue deterministic worker applies in game order, and return the
    /// COUNT without waiting for engine applies. The CUDA owner never blocks
    /// here: completions are harvested nonblocking by `poll`/`peek_ready`/
    /// fills (worker errors/panics poison loudly there), and `quiesce`/
    /// snapshot join admitted applies while retaining unissued READY rows.
    /// Rejected while a fill is in flight (take first): the fill's staged
    /// games stay thread-local until install, so the pin check below could
    /// otherwise miss them.
    pub fn commit_actions(
        self: &Arc<Self>,
        lease: &Lease,
        output: &[u8],
    ) -> Result<usize, PoolError> {
        self.harvest_applies();
        // Staged snapshot under one lock; the lock is released before worker
        // applies (workers take per-game locks; never pool state).
        let staged_meta: Vec<(usize, u8, u64, Vec<u32>)> = {
            let state = self.lock_state()?;
            self.lease_checked(&state, lease)?;
            if state.fill_inflight.is_some() {
                return Err(PoolError(format!(
                    "lease {} commit while a fill is in flight; take first",
                    lease.lease_id
                )));
            }
            let slot = slot_ref(&state, lease.slot_id)?;
            if slot.phase != SlotPhase::HostReady {
                return Err(PoolError(format!(
                    "lease {} commit before HOST_READY (phase {:?})",
                    lease.lease_id, slot.phase
                )));
            }
            if slot.rows.len() != lease.n_valid {
                return Err(PoolError(format!(
                    "lease {} row count drift",
                    lease.lease_id
                )));
            }
            slot.rows
                .iter()
                .map(|r| {
                    (
                        r.game_idx,
                        r.actor,
                        r.cursor_before,
                        r.obs
                            .legal_mask
                            .iter()
                            .enumerate()
                            .filter(|(_, f)| **f)
                            .map(|(i, _)| i as u32)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        };
        let b = lease.n_valid;
        let need = b * super::OUTPUT_ROW_STRIDE;
        if output.len() < need {
            return Err(PoolError(format!(
                "output slab {} < active {need} for B={b}",
                output.len(),
            )));
        }
        // Entire-batch validation BEFORE any capture or apply, decoded from
        // the owner planes (selected int64[B] @ 0, logp float32[B] @ 8B,
        // value4 float32[B,4] @ 12B; byte-stride reads, no alignment casts).
        let (_, logp_base, value_base) = output_plane_bases(b);
        let mut selected: Vec<u32> = Vec::with_capacity(b);
        let mut logps: Vec<f32> = Vec::with_capacity(b);
        let mut values: Vec<[f32; 4]> = Vec::with_capacity(b);
        for row in 0..b {
            let mut id_bytes = [0u8; 8];
            id_bytes.copy_from_slice(&output[row * 8..row * 8 + 8]);
            let id = u64::from_le_bytes(id_bytes);
            if id > u32::MAX as u64 {
                return Err(PoolError(format!("row {row} selected id out of range")));
            }
            let mut lp_bytes = [0u8; 4];
            lp_bytes.copy_from_slice(&output[logp_base + row * 4..logp_base + row * 4 + 4]);
            let logp = f32::from_le_bytes(lp_bytes);
            let mut value4 = [0f32; 4];
            for (k, slot) in value4.iter_mut().enumerate() {
                let at = value_base + (row * 4 + k) * 4;
                let mut value_bytes = [0u8; 4];
                value_bytes.copy_from_slice(&output[at..at + 4]);
                *slot = f32::from_le_bytes(value_bytes);
            }
            if !logp.is_finite() || !value4.iter().all(|v| v.is_finite()) {
                self.poison(format!("row {row} nonfinite commit values"))?;
                return Err(PoolError(format!("row {row} nonfinite commit values")));
            }
            let pi_old = logp.exp();
            if !(pi_old > 0.0 && pi_old <= 1.0) {
                self.poison(format!("row {row} exp(logp)={pi_old} outside (0,1]"))?;
                return Err(PoolError(format!("row {row} exp(logp) outside (0,1]")));
            }
            let (_, _, _, legal_ids) = &staged_meta[row];
            if (id as usize) >= BASELINE_ACTIONS || !legal_ids.contains(&(id as u32)) {
                self.poison(format!("row {row} selected-illegal action {id}"))?;
                return Err(PoolError(format!(
                    "row {row} selected action outside served legal set"
                )));
            }
            selected.push(id as u32);
            logps.push(logp);
            values.push(value4);
        }
        // Pin validation: no two live slots stage rows from the same game.
        self.check_game_pins(lease)?;
        // Validation complete: the slot owns its capture now (buffers stay
        // held until the release below).
        {
            let mut state = self.lock_state()?;
            slot_mut(&mut state, lease.slot_id)?.phase = SlotPhase::OwnedCapture;
            self.notify(&mut state);
        }
        // Capture owned rows against live plans (state pinned since fill, so
        // plans re-derive identically); resolve actions from the same plans.
        // Staged observations are cloned first so no game guard ever nests a
        // state lock (global order: state, then game, never nested). The
        // sampler cursor binds verbatim (pool reservation for this exact
        // decision); game capture order rides `captured`, independent of
        // sampler counters by construction (reservation-once is structural:
        // counters advance only at fill and are never rewound).
        let staged_obses: Vec<NativeObservation> = {
            let state = self.lock_state()?;
            slot_ref(&state, lease.slot_id)?
                .rows
                .iter()
                .map(|r| r.obs.clone())
                .collect()
        };
        let mut captured: Vec<(usize, usize, GameRow, CanonicalAction)> = Vec::with_capacity(b);
        for (row_index, (game_idx, actor, cursor_before, _)) in staged_meta.iter().enumerate() {
            let mut game = self.lock_game(*game_idx)?;
            let staged_obs = &staged_obses[row_index];
            let (ordered, _) = game
                .view_for(*actor)
                .map_err(|e| PoolError(format!("commit plan re-derivation rejected: {e}")))?;
            let action = ordered
                .iter()
                .find(|(id, _)| *id == selected[row_index])
                .map(|(_, a)| a.clone())
                .ok_or_else(|| {
                    PoolError(format!(
                        "row {row_index} selected action vanished from plan"
                    ))
                })?;
            let wall_id = self.wall_ids[*game_idx].clone();
            let row = game
                .capture_row(
                    staged_obs,
                    selected[row_index],
                    logps[row_index],
                    values[row_index],
                    &wall_id,
                    self.action_table_hash()?,
                    self.derivation,
                    *cursor_before,
                )
                .map_err(|e| PoolError(format!("row {row_index} capture rejected: {e}")))?;
            captured.push((row_index, *game_idx, row, action));
        }
        captured.sort_by_key(|(i, _, _, _)| *i);
        // Retain captured rows per game (generation-end tape handoff) before
        // any worker can advance the games further.
        for (_, game_idx, row, _) in &captured {
            self.lock_completed(*game_idx)?.push(row.clone());
        }
        // Dispatch per-game applies to fixed workers in deterministic game
        // order WITHOUT waiting: register every completion channel and mark
        // every game outstanding BEFORE releasing the slot, so an admitted
        // fill can never stage a decision these applies have not landed.
        let mut game_order: Vec<usize> = staged_meta.iter().map(|(g, _, _, _)| *g).collect();
        game_order.sort_unstable();
        game_order.dedup();
        let sender = {
            let workers = self.lock_workers()?;
            workers
                .sender
                .clone()
                .ok_or_else(|| PoolError("workers shut down".to_string()))?
        };
        let count = captured.len();
        {
            let mut state = self.lock_state()?;
            for game_idx in &game_order {
                let actions: Vec<CanonicalAction> = captured
                    .iter()
                    .filter(|(_, g, _, _)| g == game_idx)
                    .map(|(_, _, _, action)| action.clone())
                    .collect();
                let (tx, rx) = mpsc::sync_channel(1);
                sender
                    .send(WorkerJob::Apply(ApplyJob {
                        game_idx: *game_idx,
                        actions,
                        respond: tx,
                    }))
                    .map_err(|_| PoolError("worker queue closed".to_string()))?;
                state.pending_applies.push(PendingApply {
                    game_idx: *game_idx,
                    receiver: rx,
                });
                if let Some(count) = state.apply_outstanding.get_mut(*game_idx) {
                    *count = count.saturating_add(1);
                }
            }
            // Capture done: slot + lease release (buffers reusable now; the
            // games advance under the admitted applies above).
            let slot = slot_mut(&mut state, lease.slot_id)?;
            slot.rows.clear();
            slot.phase = SlotPhase::Free;
            slot.leased = false;
            slot.epoch += 1;
            state.leases.remove(&lease.lease_id);
            self.notify(&mut state);
        }
        Ok(count)
    }

    fn check_game_pins(&self, lease: &Lease) -> Result<(), PoolError> {
        let state = self.lock_state()?;
        let slot = slot_ref(&state, lease.slot_id)?;
        let mut mine: Vec<usize> = slot.rows.iter().map(|r| r.game_idx).collect();
        mine.sort_unstable();
        mine.dedup();
        for (id, other) in &state.slots {
            if *id == lease.slot_id || other.phase == SlotPhase::Free {
                continue;
            }
            for row in &other.rows {
                if mine.contains(&row.game_idx) {
                    return Err(PoolError(format!(
                        "game {} pinned by live slot {id}; interleaved applies rejected",
                        self.wall_ids[row.game_idx]
                    )));
                }
            }
        }
        Ok(())
    }

    fn action_table_hash(&self) -> Result<[u8; 32], PoolError> {
        super::hex32(&self.profile.shared.action_table_hash, "action_table_hash")
    }

    /// Poison the generation (sticky; surfaces on poll/drain/close/commit).
    pub fn poison(&self, reason: String) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        if state.poison.is_none() {
            state.poison = Some(reason);
        }
        self.notify(&mut state);
        Ok(())
    }

    /// Owner poll: poison status + per-slot phases + collection progress
    /// (never blocks; harvests admitted applies first so completions and
    /// worker failures surface here without the owner waiting on them).
    /// Counts are collected before the state lock (locks are never nested:
    /// state, then game, never both held).
    pub fn poll(&self) -> Result<PoolStatus, PoolError> {
        self.harvest_applies();
        let (live_rows, live_terminal) = self.collection_counts()?;
        let state = self.lock_state()?;
        // A drained pool moved its storage into the tape: report the
        // preserved generation-end counts, not the emptied buffers.
        let (captured_rows, terminal_games) = state.drained_counts.unwrap_or((live_rows, live_terminal));
        Ok(Self::status_from(&state, captured_rows, terminal_games))
    }

    /// Collection counts for the poll descriptor (captured rows retained
    /// per game + terminal games; game/completed locks only, never nested
    /// in the state lock).
    fn collection_counts(&self) -> Result<(usize, usize), PoolError> {
        let mut captured_rows = 0usize;
        let mut terminal_games = 0usize;
        for idx in 0..self.games.len() {
            captured_rows += self.lock_completed(idx)?.len();
            if self.lock_game(idx)?.is_terminal() {
                terminal_games += 1;
            }
        }
        Ok((captured_rows, terminal_games))
    }

    /// Pure status constructor from one state snapshot plus pre-collected
    /// counts (no locks inside).
    fn status_from(
        state: &super::state::PoolState,
        captured_rows: usize,
        terminal_games: usize,
    ) -> PoolStatus {
        PoolStatus {
            poison: state.poison.clone(),
            slots: state
                .slots
                .values()
                .map(|s| (s.id, s.phase, s.rows.len()))
                .collect(),
            live_leases: state.leases.len(),
            pending_applies: state.pending_applies.len(),
            retained_ready_rows: state
                .slots
                .values()
                .filter(|s| s.phase == SlotPhase::Ready)
                .map(|s| s.rows.len())
                .sum(),
            captured_rows,
            terminal_games,
            last_tape_handle: state.last_tape_handle,
            admit_ok: state.admit_ok,
            advise_none_quiesced: state.advise_none_quiesced,
            advise_none_owner_empty: state.advise_none_owner_empty,
            advise_none_no_truly_free: state.advise_none_no_truly_free,
            take_miss: state.take_miss,
            take_hit: state.take_hit,
            take_blocking: state.take_blocking,
            fill_inflight: u64::from(state.fill_inflight.is_some()),
            fill_result_ready: u64::from(
                state
                    .fill_inflight
                    .as_ref()
                    .is_some_and(|inflight| inflight.result.is_some()),
            ),
            quiesced: u64::from(state.quiesced),
        }
    }

    /// Block only while admitted work remains; completed applies are harvested
    /// before sleeping so a completion racing the call cannot strand the owner.
    pub fn wait_for_progress(&self) -> Result<PoolStatus, PoolError> {
        let current = self.lock_state()?.progress;
        self.harvest_applies();
        let mut state = self.lock_state()?;
        while state.progress == current
            && state.poison.is_none()
            && (!state.pending_applies.is_empty() || state.fill_inflight.is_some())
        {
            state = self
                .progress_cv
                .wait(state)
                .map_err(|_| PoolError("progress condvar poisoned".to_string()))?;
        }
        // Reap anything that landed while waking (still nonblocking).
        drop(state);
        self.harvest_applies();
        let (live_rows, live_terminal) = self.collection_counts()?;
        let state = self.lock_state()?;
        let (captured_rows, terminal_games) = state.drained_counts.unwrap_or((live_rows, live_terminal));
        Ok(Self::status_from(&state, captured_rows, terminal_games))
    }

    /// Owner-driven quiesce: no new fills; admitted applies join here
    /// (blocking, owner-level) while unissued READY rows and live leases are
    /// RETAINED with identical uniforms, reservations, and descriptors
    /// (nothing is dropped, rewound, or redrawn: the fork/snapshot path
    /// below preserves them verbatim).
    pub fn quiesce(&self) -> Result<(), PoolError> {
        self.join_applies()?;
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        state.quiesced = true;
        self.notify(&mut state);
        Ok(())
    }

    /// Resume admission after quiesce + snapshot/control durable: clears
    /// the quiesce flag so checkpointing continues without reset or
    /// resample (counters, epochs, retained READY rows, and completed
    /// rows all continue verbatim). Allowed only in a zero-issued state
    /// (no live leases, no admitted applies, no in-flight fill; retained
    /// unissued READY rows are fine and keep flowing).
    pub fn resume_admission(&self) -> Result<(), PoolError> {
        let mut state = self.lock_state()?;
        self.fail_if_poisoned(&state)?;
        if state.drained {
            return Err(PoolError("pool trajectories already drained".to_string()));
        }
        if !state.leases.is_empty() {
            return Err(PoolError(format!(
                "{} live leases at resume; owner must drain first",
                state.leases.len()
            )));
        }
        if !state.pending_applies.is_empty() {
            return Err(PoolError(format!(
                "{} admitted applies at resume; owner must quiesce first",
                state.pending_applies.len()
            )));
        }
        if state.fill_inflight.is_some() {
            return Err(PoolError(
                "fill in flight at resume; owner must take first".to_string(),
            ));
        }
        state.quiesced = false;
        self.notify(&mut state);
        Ok(())
    }

    /// Drain: report unissued READY rows (retained as canonical facts),
    /// live leases, and admitted-but-unlanded applies; never captures,
    /// drops, or resamples.
    pub fn drain_status(&self) -> Result<DrainStatus, PoolError> {
        self.harvest_applies();
        let state = self.lock_state()?;
        let mut retained = vec![];
        for slot in state.slots.values() {
            if slot.phase == SlotPhase::Ready {
                retained.push((slot.id, slot.rows.len()));
            }
        }
        Ok(DrainStatus {
            poison: state.poison.clone(),
            retained_ready: retained,
            live_leases: state.leases.keys().copied().collect(),
            pending_applies: state.pending_applies.len(),
        })
    }

    /// Close: rejects issued work (live leases, leased lanes, Filling
    /// lanes, admitted applies, in-flight fills) and joins workers — but
    /// ACCEPTS retained unissued READY rows (quiesce+snapshot first; the
    /// snapshot persists everything, so close drops only memory, never
    /// semantic state: no erasing, no dummy capture, no stream rewind).
    /// Never close while an issued lease is live.
    /// The FFI holds registered host owners alive through this call.
    pub fn close(self: &Arc<Self>) -> Result<(), PoolError> {
        {
            let state = self.lock_state()?;
            if !state.leases.is_empty() {
                return Err(PoolError(format!(
                    "{} live leases at close; owner must drain first",
                    state.leases.len()
                )));
            }
            if !state.pending_applies.is_empty() {
                return Err(PoolError(format!(
                    "{} admitted applies at close; owner must quiesce first",
                    state.pending_applies.len()
                )));
            }
            if state.fill_inflight.is_some() {
                return Err(PoolError(
                    "fill in flight at close; owner must take first".to_string(),
                ));
            }
            for slot in state.slots.values() {
                if slot.leased || slot.phase == SlotPhase::Filling {
                    return Err(PoolError(format!(
                        "slot {} issued ({:?}, leased={}) at close; owner must drain first",
                        slot.id, slot.phase, slot.leased
                    )));
                }
            }
        }
        let mut workers = self.lock_workers()?;
        workers.fill_sender = None;
        if let Some(handle) = workers.fill_handle.take() {
            handle.join().map_err(|_| PoolError("fill worker join failed".to_string()))?;
        }
        workers.sender = None;
        while let Some(handle) = workers.handles.pop() {
            handle
                .join()
                .map_err(|_| PoolError("worker join failed".to_string()))?;
        }
        Ok(())
    }
}
