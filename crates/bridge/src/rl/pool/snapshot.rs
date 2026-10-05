//! rl/pool/snapshot: MAIN descriptors, tape handoff, snapshot/restore.
//!
//! Invariant: snapshots carry applied action prefixes + digests + per-seat
//! sampler counters + FULL captured rows (LP/value/selected/cursors/hashes)
//! + slot epochs/phases/chunks/rounds with retained READY staged rows
//! (uniforms, cursors, obs hashes) + live leases — and no opaque RNG state
//! anywhere (streams are counters + re-derived seeds). Restore replays every
//! prefix against identical walls, verifies digests/ordinals/terminals, and
//! re-stages retained rows with identical reservations (re-derived
//! observations verified by hash, never redrawn: no dropped staged decision,
//! no additional draw). Lanes in flight or mid-transfer fail closed: settle
//! them to Free/Ready before snapshotting. Failure mode: any digest, ordinal,
//! terminal, epoch, row-count, or observation-hash drift fails restore loudly
//! instead of forking a diverged generation.

use std::sync::Arc;

use super::super::actions::CanonicalAction;
use super::super::core::{GameRow, PendingDecision, TerminalOutcome};
use super::super::{events, observation, trajectory};
use super::state::{FilledRow, Slot, slot_mut, slot_ref};
use super::{Lease, NativePool, PoolError, PoolProfile, SlotPhase, buffer_geometry, json_str};

/// Owner-visible pool status (the MAIN seal loop detects collection
/// completion here: `captured_rows`/`terminal_games`/`retained_ready_rows`
/// plus `last_tape_handle` after `export_tape`).
#[derive(Debug, Clone)]
pub struct PoolStatus {
    pub poison: Option<String>,
    pub slots: Vec<(u64, SlotPhase, usize)>,
    pub live_leases: usize,
    pub pending_applies: usize,
    pub retained_ready_rows: usize,
    pub captured_rows: usize,
    pub terminal_games: usize,
    pub last_tape_handle: Option<u64>,
    /// Admission telemetry (monotonic counters + live gauges; additive
    /// status keys, never persisted in snapshots).
    pub admit_ok: u64,
    pub advise_none_quiesced: u64,
    pub advise_none_owner_empty: u64,
    pub advise_none_no_truly_free: u64,
    pub take_miss: u64,
    pub take_hit: u64,
    pub take_blocking: u64,
    pub fill_inflight: u64,
    pub fill_result_ready: u64,
    pub quiesced: u64,
}

/// Owner-visible drain status.
#[derive(Debug, Clone)]
pub struct DrainStatus {
    pub poison: Option<String>,
    pub retained_ready: Vec<(u64, usize)>,
    pub live_leases: Vec<u64>,
    pub pending_applies: usize,
}

/// Per-game completion descriptor for Main (terminal facts + counts; row
/// payloads never cross Python — they ride `export_tape` Rust-to-Rust).
#[derive(Debug, Clone)]
pub struct GameCompletion {
    pub wall_id: String,
    pub game_id: String,
    pub terminal: bool,
    pub captured: u64,
    pub outcome: Option<TerminalOutcome>,
}

#[derive(Debug, Clone)]
pub struct StoredAction {
    pub kind: String,
    pub actor: u8,
    pub tile: Option<u8>,
    pub called_tile: Option<u8>,
    pub consumed: Vec<u8>,
    pub source: Option<u8>,
    pub declares_riichi: bool,
    pub prior_pon_id: Option<String>,
}

impl StoredAction {
    pub fn from_canonical(action: &CanonicalAction) -> Self {
        Self {
            kind: action.kind.name().to_string(),
            actor: action.actor,
            tile: action.tile,
            called_tile: action.called_tile,
            consumed: action.consumed.clone(),
            source: action.source,
            declares_riichi: action.declares_riichi,
            prior_pon_id: action.prior_pon_id.clone(),
        }
    }

    pub fn to_canonical(&self) -> Result<CanonicalAction, PoolError> {
        use hydra_feed::census::ActionKind;
        let kind = ActionKind::from_name(&self.kind)
            .ok_or_else(|| PoolError(format!("snapshot unknown kind {:?}", self.kind)))?;
        Ok(CanonicalAction {
            kind,
            actor: self.actor,
            tile: self.tile,
            called_tile: self.called_tile,
            consumed: self.consumed.clone(),
            source: self.source,
            declares_riichi: self.declares_riichi,
            prior_pon_id: self.prior_pon_id.clone(),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind,
            "actor": self.actor,
            "tile": self.tile,
            "called_tile": self.called_tile,
            "consumed": self.consumed,
            "source": self.source,
            "declares_riichi": self.declares_riichi,
            "prior_pon_id": self.prior_pon_id,
        })
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self, PoolError> {
        Ok(Self {
            kind: json_str(value, "kind")?,
            actor: value.get("actor").and_then(|v| v.as_u64()).unwrap_or(0) as u8,
            tile: value.get("tile").and_then(|v| v.as_u64()).map(|t| t as u8),
            called_tile: value.get("called_tile").and_then(|v| v.as_u64()).map(|t| t as u8),
            consumed: value
                .get("consumed")
                .and_then(|v| v.as_array())
                .map_or(vec![], |items| {
                    items.iter().filter_map(|t| t.as_u64().map(|x| x as u8)).collect()
                }),
            source: value.get("source").and_then(|v| v.as_u64()).map(|t| t as u8),
            declares_riichi: value
                .get("declares_riichi")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            prior_pon_id: value.get("prior_pon_id").and_then(|v| v.as_str()).map(str::to_string),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GameSnapshot {
    pub wall_id: String,
    pub ordinal: u64,
    pub applied: Vec<StoredAction>,
    pub digest: [u8; 32],
    pub captured: u64,
    /// Independent sampler counters per seat (prescribed streams).
    pub sample_counters: [u64; 4],
    pub terminal: bool,
    /// Full captured compact rows (LP/value/selected/cursors/hashes):
    /// replay cannot invent LP/V, so these persist verbatim.
    pub completed: Vec<GameRow>,
}

/// One retained unissued READY row (uniform + reservation + identity; the
/// observation itself re-derives deterministically and is verified by hash,
/// never redrawn).
#[derive(Debug, Clone)]
pub struct StagedRowSnapshot {
    pub wall_id: String,
    pub actor: u8,
    pub uniform_bits: u64,
    pub cursor_before: u64,
    pub obs_hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct SlotSnapshot {
    pub id: u64,
    pub epoch: u64,
    pub phase: SlotPhase,
    pub bucket_t: usize,
    pub capacity: usize,
    pub chunk: u64,
    pub logical_round: u64,
    pub staged: Vec<StagedRowSnapshot>,
}

#[derive(Debug, Clone)]
pub struct PoolSnapshot {
    pub pool_id: String,
    pub generation: u64,
    pub games: Vec<GameSnapshot>,
    pub slots: Vec<SlotSnapshot>,
    pub leases: Vec<Lease>,
    pub poison: Option<String>,
}

fn hex_of(raw: &[u8; 32]) -> String {
    raw.iter().map(|b| format!("{b:02x}")).collect()
}

impl PoolSnapshot {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pool_id": self.pool_id,
            "generation": self.generation,
            "games": self.games.iter().map(|g| serde_json::json!({
                "wall_id": g.wall_id,
                "ordinal": g.ordinal,
                "applied": g.applied.iter().map(|a| a.to_json()).collect::<Vec<_>>(),
                "digest": hex_of(&g.digest),
                "captured": g.captured,
                "sample_counters": g.sample_counters,
                "terminal": g.terminal,
                "completed": g.completed.iter().map(|r| r.to_json()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "slots": self.slots.iter().map(|s| serde_json::json!({
                "id": s.id, "epoch": s.epoch, "phase": format!("{:?}", s.phase),
                "bucket_t": s.bucket_t, "capacity": s.capacity,
                "chunk": s.chunk, "logical_round": s.logical_round,
                "staged": s.staged.iter().map(|r| serde_json::json!({
                    "wall_id": r.wall_id, "actor": r.actor,
                    "uniform_bits": r.uniform_bits, "cursor_before": r.cursor_before,
                    "obs_hash": hex_of(&r.obs_hash),
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "leases": self.leases.iter().map(|l| serde_json::json!({
                "lease_id": l.lease_id, "slot_id": l.slot_id, "slot_epoch": l.slot_epoch,
                "generation": l.generation, "policy_id": l.policy_id,
                "bucket_t": l.bucket_t, "chunk": l.chunk,
                "n_valid": l.n_valid, "logical_round": l.logical_round,
            })).collect::<Vec<_>>(),
            "poison": self.poison,
        })
    }
}

impl NativePool {
    /// Per-game completion descriptors for Main (terminal flags, captured
    /// counts, terminal outcome facts; row payloads never cross Python).
    /// Game/completed locks only, never nested in the state lock.
    pub fn game_completions(&self) -> Result<Vec<GameCompletion>, PoolError> {
        let mut out = Vec::with_capacity(self.games.len());
        for idx in 0..self.games.len() {
            let game = self.lock_game(idx)?;
            out.push(GameCompletion {
                wall_id: self.wall_ids[idx].clone(),
                game_id: game.game_id().to_string(),
                terminal: game.is_terminal(),
                captured: game.captured_count(),
                outcome: game.outcome().cloned(),
            });
        }
        Ok(out)
    }

    /// Staged-row descriptor for one live lease (owner scheduling,
    /// verification tooling, and the reference-parity harness; never the
    /// hot loop): per-row wall/game/actor/decision/cursor/uniform plus the
    /// full legal signature set derived from the live plan. The derived
    /// mask must equal the staged observation mask exactly (fail loud on
    /// any drift: the commit path re-derives the same plan).
    pub fn lease_descriptor(&self, lease: &Lease) -> Result<serde_json::Value, PoolError> {
        let staged: Vec<(usize, u8, u64, f64, observation::NativeObservation)> = {
            let state = self.lock_state()?;
            self.lease_checked(&state, lease)?;
            slot_ref(&state, lease.slot_id)?
                .rows
                .iter()
                .map(|r| (r.game_idx, r.actor, r.cursor_before, r.uniform, r.obs.clone()))
                .collect()
        };
        let mut rows = Vec::with_capacity(staged.len());
        for (game_idx, actor, cursor_before, uniform, obs) in &staged {
            let mut game = self.lock_game(*game_idx)?;
            let (ordered, mask) = game.view_for(*actor).map_err(|e| {
                PoolError(format!("descriptor plan re-derivation rejected: {e}"))
            })?;
            if *mask != obs.legal_mask {
                return Err(PoolError(format!(
                    "game {} seat {actor} plan drifted since staging",
                    self.wall_ids[*game_idx]
                )));
            }
            let legal: Vec<serde_json::Value> = ordered
                .iter()
                .map(|(id, action)| {
                    serde_json::json!({
                        "id": id,
                        "kind": action.kind.name(),
                        "actor": action.actor,
                        "tile": action.tile,
                        "called_tile": action.called_tile,
                        "consumed": action.consumed,
                        "source": action.source,
                        "declares_riichi": action.declares_riichi,
                        "prior_pon_id": action.prior_pon_id,
                    })
                })
                .collect();
            rows.push(serde_json::json!({
                "wall_id": self.wall_ids[*game_idx],
                "game_id": game.game_id(),
                "actor": actor,
                "decision_id": obs.decision_id,
                "sequence": obs.sequence,
                "obs_hash": obs.observation_hash.map(|h| hex_of(&h)),
                "cursor_before": cursor_before,
                "uniform": uniform,
                "legal": legal,
            }));
        }
        Ok(serde_json::json!({
            "lease_id": lease.lease_id,
            "slot_id": lease.slot_id,
            "slot_epoch": lease.slot_epoch,
            "generation": lease.generation,
            "logical_round": lease.logical_round,
            "policy_id": lease.policy_id,
            "bucket_t": lease.bucket_t,
            "chunk": lease.chunk,
            "n_valid": lease.n_valid,
            "rows": rows,
        }))
    }

    /// Generation-end tape handoff (ArtifactStorage contract): clone every
    /// game's retained rows + event arena into `trajectory::register_tape`
    /// with the pool's canonical materializer/encoder, and return the
    /// process handle Main submits to the seal. Production rows stay
    /// UNBOUND: `TapeGame.utilities` carries the terminal G4 and Main's GPU
    /// `gather_terminal_inputs` -> `freeze_terminal_advantages` freezes
    /// exactly once before seal (a CPU bind here would make the GPU freeze
    /// a loud second-freeze error; `bind_terminal` remains an oracle
    /// helper only). Unterminated games register with `utilities: None`
    /// (seal-side decides; loud on unbound). Requires: no live leases, no
    /// admitted applies, no in-flight fill. Rows never cross Python.
    /// Idempotent: clones rows out, never mutates the retained originals.
    pub fn export_tape(self: &Arc<Self>) -> Result<trajectory::TapeHandle, PoolError> {
        self.harvest_applies();
        {
            let state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            if state.drained {
                return Err(PoolError("pool trajectories already drained".to_string()));
            }
            if !state.leases.is_empty() {
                return Err(PoolError(format!(
                    "{} live leases at export; owner must drain first",
                    state.leases.len()
                )));
            }
            if !state.pending_applies.is_empty() {
                return Err(PoolError(format!(
                    "{} admitted applies at export; owner must quiesce first",
                    state.pending_applies.len()
                )));
            }
            if state.fill_inflight.is_some() {
                return Err(PoolError("fill in flight at export; owner must take first".to_string()));
            }
        }
        // Only row-bearing games ride the tape (rowless cohort members
        // carry nothing to seal; the utilities gate therefore applies to
        // row-bearing games exactly, never vacuously).
        let mut games = Vec::with_capacity(self.games.len());
        for idx in 0..self.games.len() {
            let game = self.lock_game(idx)?;
            let rows = self.lock_completed(idx)?.clone();
            if rows.is_empty() {
                continue;
            }
            games.push(trajectory::TapeGame {
                wall_id: self.wall_ids[idx].clone(),
                game_id: game.game_id().to_string(),
                arena: game.arena().clone(),
                rows,
                utilities: game.outcome().map(|o| o.utility_g4),
            });
        }
        if games.is_empty() {
            return Err(PoolError("export_tape with no captured rows".to_string()));
        }
        let materialize = |row: &GameRow, arena: &events::EventArena| {
            observation::materialize_canonical(row, arena)
                .map_err(|e| trajectory::TapeError(e.0))
        };
        let encode = |row: &GameRow, arena: &events::EventArena, t: u32| {
            observation::encode_row_facts(row, arena, t as usize)
                .map_err(|e| trajectory::TapeError(e.0))
        };
        let handle =
            trajectory::register_tape(games, Box::new(materialize), Box::new(encode))
                .map_err(|e| PoolError(format!("tape registration rejected: {e}")))?;
        {
            let mut state = self.lock_state()?;
            state.last_tape_handle = Some(handle);
            self.notify(&mut state);
        }
        Ok(handle)
    }

    /// Destructive live tape handoff: moves (never clones) every completed
    /// game arena plus retained rows into `trajectory::register_live_tape`
    /// with the verified source binding installed atomically, and returns
    /// the process handle. Production rows stay UNBOUND (utilities carried;
    /// the GPU freezes terminal facts exactly once downstream).
    ///
    /// Full-cohort preflight runs before the first move: owner quiesced, no
    /// leases/admits/in-flight fill/issued slots, every cohort game terminal
    /// with an outcome, finite utilities on every row-bearing game, at least
    /// one captured row, and an exact generation/behavior/pipeline match
    /// against the profile plus the immutable live binding from `open_live`.
    /// Preflight failure leaves the pool unchanged. Registration failure
    /// after movement poisons the consumed pool and fails the run (no
    /// half-drained cohort is ever advertised usable). After success the
    /// pool is drained: snapshot/clone export/second drain/admission
    /// reject, close stays legal, and read-only terminal status keeps the
    /// preserved aggregate counts.
    pub fn drain_tape(
        self: &Arc<Self>,
        binding: trajectory::LiveTapeBinding,
    ) -> Result<trajectory::TapeHandle, PoolError> {
        self.harvest_applies();
        let expected = {
            let state = self.lock_state()?;
            self.fail_if_poisoned(&state)?;
            if state.drained {
                return Err(PoolError("pool trajectories already drained".to_string()));
            }
            if !state.quiesced {
                return Err(PoolError(
                    "drain_tape before quiesce; owner must quiesce first".to_string(),
                ));
            }
            if !state.leases.is_empty() {
                return Err(PoolError(format!(
                    "{} live leases at drain; owner must drain first",
                    state.leases.len()
                )));
            }
            if !state.pending_applies.is_empty() {
                return Err(PoolError(format!(
                    "{} admitted applies at drain; owner must quiesce first",
                    state.pending_applies.len()
                )));
            }
            if state.fill_inflight.is_some() {
                return Err(PoolError(
                    "fill in flight at drain; owner must take first".to_string(),
                ));
            }
            for slot in state.slots.values() {
                if slot.phase != SlotPhase::Free || slot.leased {
                    return Err(PoolError(format!(
                        "slot {} issued ({:?}, leased={}) at drain; owner must drain first",
                        slot.id, slot.phase, slot.leased
                    )));
                }
            }
            self.live_binding.clone()
        };
        let expected = expected.ok_or_else(|| {
            PoolError("pool has no live binding; use open_live for the live path".to_string())
        })?;
        if binding.generation != self.profile.generation
            || binding.generation != expected.generation
        {
            return Err(PoolError(format!(
                "live binding generation {} mismatches pool generation {}",
                binding.generation, self.profile.generation
            )));
        }
        if binding.behavior_policy_id != self.profile.policy_id
            || binding.behavior_policy_id != expected.behavior_policy_id
        {
            return Err(PoolError(
                "live binding behavior policy mismatches pool policy".to_string(),
            ));
        }
        if binding.pipeline_spec_id != expected.pipeline_spec_id {
            return Err(PoolError(
                "live binding pipeline mismatches pool live binding".to_string(),
            ));
        }
        // Read-only full-cohort preflight in stable game-index order: every
        // game terminal with an outcome, finite utilities, at least one
        // captured row overall. No moves yet, so failure leaves the pool
        // unchanged. Rowless terminal members gate completion but ride no
        // tape, mirroring the clone export's row-bearing-only shape.
        let mut total_rows = 0usize;
        for idx in 0..self.games.len() {
            let game = self.lock_game(idx)?;
            if !game.is_terminal() {
                return Err(PoolError(format!(
                    "game {} not terminal at drain; cohort incomplete",
                    self.wall_ids[idx]
                )));
            }
            let outcome = game.outcome().cloned().ok_or_else(|| {
                PoolError(format!(
                    "game {} terminal without outcome at drain",
                    self.wall_ids[idx]
                ))
            })?;
            if outcome.utility_g4.iter().any(|v| !v.is_finite()) {
                return Err(PoolError(format!(
                    "game {} utilities must be finite at drain",
                    self.wall_ids[idx]
                )));
            }
            total_rows += self.lock_completed(idx)?.len();
        }
        if total_rows == 0 {
            return Err(PoolError("drain_tape with no captured rows".to_string()));
        }
        // Destructive moves (infallible): each row-bearing game's arena and
        // rows travel together with its utilities and ids into one TapeGame.
        let mut games = Vec::with_capacity(self.games.len());
        for idx in 0..self.games.len() {
            let mut game = self.lock_game(idx)?;
            let game_id = game.game_id().to_string();
            let utilities = game
                .outcome()
                .map(|o| o.utility_g4)
                .ok_or_else(|| {
                    PoolError(format!(
                        "game {} terminal without outcome at drain",
                        self.wall_ids[idx]
                    ))
                })?;
            let mut rows = self.lock_completed(idx)?;
            if rows.is_empty() {
                continue;
            }
            let arena = game.take_arena();
            let taken: Vec<GameRow> = std::mem::take(&mut *rows);
            games.push(trajectory::TapeGame {
                wall_id: self.wall_ids[idx].clone(),
                game_id,
                arena,
                rows: taken,
                utilities: Some(utilities),
            });
        }
        let materialize = |row: &GameRow, arena: &events::EventArena| {
            observation::materialize_canonical(row, arena)
                .map_err(|e| trajectory::TapeError(e.0))
        };
        let encode = |row: &GameRow, arena: &events::EventArena, t: u32| {
            observation::encode_row_facts(row, arena, t as usize)
                .map_err(|e| trajectory::TapeError(e.0))
        };
        let cohort = self.games.len();
        let handle = match trajectory::register_live_tape(
            games,
            Box::new(materialize),
            Box::new(encode),
            binding,
        ) {
            Ok(handle) => handle,
            Err(e) => {
                let mut state = self.lock_state().map_err(|_| {
                    PoolError("pool state mutex poisoned after tape failure".to_string())
                })?;
                self.poison_locked(&mut state, format!("live tape registration failed: {e}"));
                return Err(PoolError(format!("live tape registration rejected: {e}")));
            }
        };
        {
            let mut state = self.lock_state()?;
            state.drained = true;
            state.drained_counts = Some((total_rows, cohort));
            state.last_tape_handle = Some(handle);
            self.notify(&mut state);
        }
        Ok(handle)
    }


    /// Snapshot: per-game applied prefixes + digests + per-(game, seat)
    /// sampler counters + FULL captured compact rows (LP/value/selected/
    /// cursors/hashes/lineage: replay cannot invent LP/V, so restore must
    /// never re-infer or resample) + slot epochs/phases/chunks/rounds with
    /// retained READY staged rows (uniforms, cursors, obs hashes) + live
    /// leases. No opaque RNG state anywhere (streams are counters + seeds
    /// re-derived deterministically at open). Admitted applies join first
    /// (blocking, owner-level); lanes in flight (Filling) or mid-transfer
    /// (H2D/Infer/D2H/HostReady/OwnedCapture) fail closed: settle them
    /// before snapshotting.
    pub fn snapshot(&self) -> Result<PoolSnapshot, PoolError> {
        self.join_applies()?;
        // Slot/lease summary first (state guard drops before game guards;
        // locks are never nested).
        let (slots, leases, poison) = {
            let state = self.lock_state()?;
            if state.drained {
                return Err(PoolError("pool trajectories already drained".to_string()));
            }
            if state.fill_inflight.is_some() {
                return Err(PoolError("snapshot with a fill in flight; take first".to_string()));
            }
            let mut slots = vec![];
            for s in state.slots.values() {
                match s.phase {
                    SlotPhase::Free | SlotPhase::Ready => {}
                    _ => {
                        return Err(PoolError(format!(
                            "snapshot with slot {} in phase {:?}; settle to Free/Ready first",
                            s.id, s.phase
                        )));
                    }
                }
                slots.push(SlotSnapshot {
                    id: s.id,
                    epoch: s.epoch,
                    phase: s.phase,
                    bucket_t: s.bucket_t,
                    capacity: s.capacity,
                    chunk: s.chunk,
                    logical_round: s.logical_round,
                    staged: s
                        .rows
                        .iter()
                        .map(|r| {
                            Ok(StagedRowSnapshot {
                                wall_id: self.wall_ids[r.game_idx].clone(),
                                actor: r.actor,
                                uniform_bits: r.uniform.to_bits(),
                                cursor_before: r.cursor_before,
                                obs_hash: r.obs.observation_hash.ok_or_else(|| {
                                    PoolError("staged row lacks a bound observation hash".to_string())
                                })?,
                            })
                        })
                        .collect::<Result<Vec<_>, PoolError>>()?,
                });
            }
            slots.sort_by_key(|s| s.id);
            let mut leases: Vec<Lease> = state.leases.values().cloned().collect();
            leases.sort_by_key(|l| l.lease_id);
            (slots, leases, state.poison.clone())
        };
        let mut games = vec![];
        for idx in 0..self.games.len() {
            let sample_counters = *self.lock_counter(idx)?;
            let game = self.lock_game(idx)?;
            let completed = self.lock_completed(idx)?.clone();
            games.push(GameSnapshot {
                wall_id: self.wall_ids[idx].clone(),
                ordinal: game.hand_index(),
                applied: game.applied_prefix().iter().map(StoredAction::from_canonical).collect(),
                digest: game.state_digest(),
                captured: game.captured_count(),
                sample_counters,
                terminal: game.is_terminal(),
                completed,
            });
        }
        Ok(PoolSnapshot {
            pool_id: self.profile.pool_id.clone(),
            generation: self.profile.generation,
            games,
            slots,
            leases,
            poison,
        })
    }

    /// Restore into a FRESH pool: replay every prefix against identical walls,
    /// verify digests/ordinals/terminals, reinstate per-seat counters AND the
    /// full captured compact rows (no re-inference, no resampling), adopt
    /// slot lanes with epochs, and re-stage retained READY rows with
    /// identical uniforms/cursors/descriptors (re-derived observations are
    /// verified by hash, never redrawn: no dropped staged decision, no
    /// additional draw).
    pub fn restore(
        profile: PoolProfile,
        snap: &PoolSnapshot,
        derivation: [u8; 32],
    ) -> Result<Arc<Self>, PoolError> {
        if snap.pool_id != profile.pool_id || snap.generation != profile.generation {
            return Err(PoolError("snapshot pool/generation mismatch".to_string()));
        }
        if snap.games.len() != profile.games.len() {
            return Err(PoolError("snapshot cohort size mismatch".to_string()));
        }
        let pool = NativePool::open(profile, derivation)?;
        for (idx, game_snap) in snap.games.iter().enumerate() {
            if game_snap.wall_id != pool.wall_ids[idx] {
                return Err(PoolError(format!("game {idx} wall mismatch on restore")));
            }
            let prefix: Vec<CanonicalAction> = game_snap
                .applied
                .iter()
                .map(StoredAction::to_canonical)
                .collect::<Result<_, _>>()?;
            {
                let mut game = pool.lock_game(idx)?;
                game.replay_prefix(&prefix).map_err(|e| {
                    PoolError(format!("game {} replay rejected: {e}", game_snap.wall_id))
                })?;
                game.set_captured(game_snap.captured);
                if game.state_digest() != game_snap.digest {
                    return Err(PoolError(format!(
                        "game {} digest mismatch after replay",
                        game_snap.wall_id
                    )));
                }
                if game.hand_index() != game_snap.ordinal {
                    return Err(PoolError(format!(
                        "game {} ordinal mismatch after replay",
                        game_snap.wall_id
                    )));
                }
                if game.is_terminal() != game_snap.terminal {
                    return Err(PoolError(format!(
                        "game {} terminal mismatch after replay",
                        game_snap.wall_id
                    )));
                }
            }
            *pool.lock_counter(idx)? = game_snap.sample_counters;
            *pool.lock_completed(idx)? = game_snap.completed.clone();
        }
        // Adopt slot lanes with epochs/chunks/rounds (buffers re-registered
        // by the owner; refill_staged re-encodes byte-identical inputs).
        // Re-stage retained READY rows by re-derivation: replay leaves each
        // game exactly where staging happened (no applies have landed since
        // snapshot joined them), so drain + observe reproduce the identical
        // observations (verified by hash) with the snapshot reservations.
        {
            let mut state = pool.lock_state()?;
            for slot_snap in &snap.slots {
                buffer_geometry(slot_snap.bucket_t, slot_snap.capacity).map_err(|e| {
                    PoolError(format!("snapshot slot {} rejected: {e}", slot_snap.id))
                })?;
                match slot_snap.phase {
                    SlotPhase::Free | SlotPhase::Ready => {}
                    _ => {
                        return Err(PoolError(format!(
                            "snapshot slot {} phase {:?} unrestorable",
                            slot_snap.id, slot_snap.phase
                        )));
                    }
                }
                state.slots.insert(slot_snap.id, Slot {
                    id: slot_snap.id,
                    epoch: slot_snap.epoch,
                    bucket_t: slot_snap.bucket_t,
                    capacity: slot_snap.capacity,
                    phase: slot_snap.phase,
                    leased: false,
                    rows: vec![],
                    chunk: slot_snap.chunk,
                    logical_round: slot_snap.logical_round,
                });
            }
        }
        for slot_snap in &snap.slots {
            if slot_snap.phase != SlotPhase::Ready || slot_snap.staged.is_empty() {
                continue;
            }
            pool.restage_slot(slot_snap)?;
        }
        // Reinsert live leases over the re-staged rows (owner re-registers
        // buffers, then commits with the same output slab).
        {
            let mut state = pool.lock_state()?;
            for lease in &snap.leases {
                let slot = slot_ref(&state, lease.slot_id).map_err(|_| {
                    PoolError(format!("lease {} slot {} unrestorable", lease.lease_id, lease.slot_id))
                })?;
                if slot.epoch != lease.slot_epoch {
                    return Err(PoolError(format!("lease {} epoch drift on restore", lease.lease_id)));
                }
                if slot.rows.len() != lease.n_valid {
                    return Err(PoolError(format!(
                        "lease {} row count drift on restore",
                        lease.lease_id
                    )));
                }
                if lease.generation != pool.profile.generation
                    || lease.policy_id != pool.profile.policy_id
                {
                    return Err(PoolError(format!(
                        "lease {} generation/policy drift on restore",
                        lease.lease_id
                    )));
                }
                slot_mut(&mut state, lease.slot_id)?.leased = true;
                state.leases.insert(lease.lease_id, lease.clone());
                if lease.lease_id >= state.next_lease {
                    state.next_lease = lease.lease_id + 1;
                }
            }
            pool.notify(&mut state);
        }
        Ok(pool)
    }

    /// Re-stage one adopted READY lane from its snapshot rows (restore path;
    /// game locks only, never nested in the state lock).
    fn restage_slot(self: &Arc<Self>, slot_snap: &SlotSnapshot) -> Result<(), PoolError> {
        // Resolve wall ids once (fail loud on unknown members).
        let mut staged: Vec<(usize, StagedRowSnapshot)> = Vec::with_capacity(slot_snap.staged.len());
        for row_snap in &slot_snap.staged {
            let game_idx = self.wall_ids.iter().position(|w| w == &row_snap.wall_id).ok_or_else(|| {
                PoolError(format!("staged wall {} unrestorable", row_snap.wall_id))
            })?;
            staged.push((game_idx, row_snap.clone()));
        }
        // Drain once per consecutive same-game run (staging order is cohort
        // order, so each game's rows are contiguous) and verify every
        // re-derived observation by hash (identical uniforms/cursors, zero
        // new sampler draws).
        let mut rows: Vec<FilledRow> = Vec::with_capacity(staged.len());
        let mut run_start = 0usize;
        while run_start < staged.len() {
            let game_idx = staged[run_start].0;
            let mut run_end = run_start;
            while run_end < staged.len() && staged[run_end].0 == game_idx {
                run_end += 1;
            }
            let mut game = self.lock_game(game_idx)?;
            if game.is_terminal() {
                return Err(PoolError(format!(
                    "game {} terminal with retained staged rows",
                    self.wall_ids[game_idx]
                )));
            }
            let decision = game.drain().map_err(|e| {
                PoolError(format!("game {} re-drain rejected: {e}", self.wall_ids[game_idx]))
            })?;
            let seats: Vec<u8> = match decision {
                None => {
                    return Err(PoolError(format!(
                        "game {} re-drain found no decision for retained rows",
                        self.wall_ids[game_idx]
                    )));
                }
                Some(PendingDecision::Draw { seat }) => vec![seat],
                Some(PendingDecision::Window { pending }) => pending,
            };
            for (_, row_snap) in &staged[run_start..run_end] {
                if !seats.contains(&row_snap.actor) {
                    return Err(PoolError(format!(
                        "game {} seat {} missing from re-derived decision",
                        self.wall_ids[game_idx], row_snap.actor
                    )));
                }
                let obs = game.observe(row_snap.actor).map_err(|e| {
                    PoolError(format!("game {} re-observe rejected: {e}", self.wall_ids[game_idx]))
                })?;
                if obs.observation_hash != Some(row_snap.obs_hash) {
                    return Err(PoolError(format!(
                        "game {} seat {} observation drift on restore",
                        self.wall_ids[game_idx], row_snap.actor
                    )));
                }
                rows.push(FilledRow {
                    game_idx,
                    actor: row_snap.actor,
                    obs,
                    uniform: f64::from_bits(row_snap.uniform_bits),
                    cursor_before: row_snap.cursor_before,
                });
            }
            run_start = run_end;
        }
        let mut state = self.lock_state()?;
        let slot = slot_mut(&mut state, slot_snap.id)?;
        if slot.phase != SlotPhase::Ready || slot.epoch != slot_snap.epoch {
            return Err(PoolError(format!("slot {} moved during restage", slot_snap.id)));
        }
        slot.rows = rows;
        self.notify(&mut state);
        Ok(())
    }
}
