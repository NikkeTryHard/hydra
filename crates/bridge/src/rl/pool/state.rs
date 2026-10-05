//! rl/pool/state: slot/lease tables, pool storage, lock/poison primitives.
//!
//! Invariant: `NativePool` is the sole owner of games, slots, leases, and
//! workers; locks never nest (state, then game, never both held), and every
//! mutex/condvar failure degrades to `PoolError`, never a panic. `open` is
//! the sole constructor: every cohort game opens to its first decision and
//! every (game, seat) sampler seed derives deterministically from the
//! profile, so a fresh pool and a restore start from identical material.
//! Failure mode: poison is sticky (first reason wins) and every entry point
//! checks it first, so one worker loss fails the generation loudly instead
//! of serving partial rounds.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc};
use std::thread::JoinHandle;

use super::super::actions::CanonicalAction;
use super::super::core::NativeGame;
use super::super::trajectory;
use super::{PoolError, PoolProfile, sampler_seed, sampler_split_id};

/// Slot lifecycle phase (lease state machine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotPhase {
    Free,
    Filling,
    Ready,
    H2dIssued,
    InferQueued,
    D2hIssued,
    HostReady,
    OwnedCapture,
}

pub(super) struct Slot {
    pub(super) id: u64,
    pub(super) epoch: u64,
    pub(super) bucket_t: usize,
    pub(super) capacity: usize,
    pub(super) phase: SlotPhase,
    pub(super) leased: bool,
    pub(super) rows: Vec<FilledRow>,
    pub(super) chunk: u64,
    pub(super) logical_round: u64,
}

/// One staged decision: game + actor + validated observation + stream cursor.
pub(super) struct FilledRow {
    pub(super) game_idx: usize,
    pub(super) actor: u8,
    pub(super) obs: super::super::observation::NativeObservation,
    pub(super) uniform: f64,
    pub(super) cursor_before: u64,
}

/// Opaque lease binding (slot + epoch + generation + policy + pool).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub lease_id: u64,
    pub slot_id: u64,
    pub slot_epoch: u64,
    pub generation: u64,
    pub policy_id: String,
    pub bucket_t: usize,
    pub chunk: u64,
    pub n_valid: usize,
    pub logical_round: u64,
}

/// Stable ticket descriptor (returned with fills for owner scheduling).
#[derive(Debug, Clone)]
pub struct TicketDescriptor {
    pub slot_id: u64,
    pub slot_epoch: u64,
    pub generation: u64,
    pub logical_round: u64,
    pub policy_id: String,
    pub bucket_t: usize,
    pub chunk: u64,
    pub n_valid: usize,
    pub game_ids: Vec<String>,
    pub cursors_before: Vec<u64>,
}

pub(super) struct ApplyJob {
    pub(super) game_idx: usize,
    pub(super) actions: Vec<CanonicalAction>,
    pub(super) respond: mpsc::SyncSender<Result<(), String>>,
}

// Apply and observation jobs share fixed workers; per-game locks prohibit
// concurrent mutation, and response loss fails the owning batch closed.
pub(super) enum WorkerJob {
    Apply(ApplyJob),
    Observe {
        game_idx: usize,
        seat: u8,
        respond: mpsc::SyncSender<Result<super::super::observation::NativeObservation, String>>,
    },
}

/// One admitted engine apply awaiting worker completion (commit returns
/// before these land; harvest is nonblocking, join is owner-level).
pub(super) struct PendingApply {
    pub(super) game_idx: usize,
    pub(super) receiver: mpsc::Receiver<Result<(), String>>,
}

/// One admitted fill running on a dispatched worker thread (the CUDA owner
/// never encodes synchronously: admit, keep the GPU supplied, then take).
/// The worker deposits its payload under the state lock and notifies;
/// takes check readiness BEFORE allocating any slab (a None poll costs one
/// lock, never a megabyte memset).
pub(super) struct FillInflight {
    pub(super) slot_id: u64,
    pub(super) epoch: u64,
    pub(super) result: Option<FillOutput>,
}

/// Worker-computed fill payload (staging + encoded bytes); installed into
/// the slot by the taking thread under the state lock (chunk assignment and
/// phase flip stay on the taker, never the worker).
pub(super) struct FillOutput {
    pub(super) slot_id: u64,
    pub(super) epoch: u64,
    pub(super) bucket_t: usize,
    pub(super) staged: Vec<FilledRow>,
    pub(super) bytes: Vec<u8>,
    pub(super) logical_round: u64,
    /// Worker-side failure (install frees the lane, poisons loudly, and
    /// never rewinds sampler counters).
    pub(super) error: Option<String>,
}

pub(super) struct WorkerPool {
    pub(super) sender: Option<mpsc::Sender<WorkerJob>>,
    pub(super) handles: Vec<JoinHandle<()>>,
    pub(super) fill_sender: Option<mpsc::Sender<(u64, u64, usize, usize)>>,
    pub(super) fill_handle: Option<JoinHandle<()>>,
}

pub(super) struct PoolState {
    pub(super) slots: HashMap<u64, Slot>,
    pub(super) leases: HashMap<u64, Lease>,
    pub(super) next_lease: u64,
    pub(super) next_chunk: HashMap<usize, u64>,
    pub(super) poison: Option<String>,
    pub(super) quiesced: bool,
    pub(super) progress: u64,
    /// Admitted applies not yet harvested (game order preserved per job).
    pub(super) pending_applies: Vec<PendingApply>,
    /// Per-game outstanding apply count (fills skip nonzero entries so a
    /// new round never cannibalizes a game whose applies have not landed).
    pub(super) apply_outstanding: Vec<usize>,
    /// At most one dispatched fill at a time (fills drain games
    /// sequentially; concurrent fills could stage the same decision twice).
    pub(super) fill_inflight: Option<FillInflight>,
    /// Last installed row count per lane (advisory fill routing: lanes
    /// that recently yielded rows are preferred; empty lanes are retried
    /// only when nothing productive is free).
    pub(super) fill_yield: HashMap<u64, usize>,
    /// Rotation cursor for advisory fill routing (no lane starves).
    pub(super) advise_cursor: u64,
    /// Generation-end tape handle from the last `export_tape` (None until
    /// the first export; reported in the poll descriptor for the seal loop).
    pub(super) last_tape_handle: Option<u64>,
    /// Destructive drain latch (`drain_tape`): once set, snapshot/clone
    /// export/second drain/admission all reject while close stays legal.
    /// Not poison: a drained pool is consumed, not broken.
    pub(super) drained: bool,
    /// Aggregate counts preserved at drain (captured rows + terminal
    /// games) so poll status keeps reporting the consumed cohort after
    /// its storage moved into the tape.
    pub(super) drained_counts: Option<(usize, usize)>,
    /// Admission telemetry (monotonic, process lifetime; excluded from
    /// snapshots). Distinguishes worker-busy stalls from no-free-lane
    /// stalls so a replenishment change is only built on measured need.
    /// All increments happen under the already-held state lock.
    pub(super) admit_ok: u64,
    pub(super) advise_none_quiesced: u64,
    pub(super) advise_none_owner_empty: u64,
    pub(super) advise_none_no_truly_free: u64,
    pub(super) take_miss: u64,
    pub(super) take_hit: u64,
    pub(super) take_blocking: u64,
}

/// Fixed-worker native pool (sole owner of games, slots, leases, workers).
pub struct NativePool {
    pub(super) profile: PoolProfile,
    pub(super) games: Arc<Vec<Mutex<NativeGame>>>,
    pub(super) wall_ids: Vec<String>,
    /// Independent sampler counters per (game, seat) (prescribed
    /// `actor_policy_sample` streams; seats never share a counter, so
    /// interleave order cannot fork uniforms).
    pub(super) sample_counters: Vec<Mutex<[u64; 4]>>,
    /// Derived semantic seeds per (game, seat) (cold, deterministic from
    /// the profile + derived game ids; identical after restore, never
    /// re-drawn).
    pub(super) sample_seeds: Vec<[[u8; 32]; 4]>,
    /// Captured rows retained per game for the generation-end tape handoff
    /// (`export_tape`; rows never cross Python).
    pub(super) completed: Vec<Mutex<Vec<super::super::core::GameRow>>>,
    /// Immutable live source binding installed by `open_live` (None for
    /// archive/test pools opened via `open` or `restore`).
    pub(super) live_binding: Option<trajectory::LiveTapeBinding>,
    pub(super) state: Mutex<PoolState>,
    pub(super) progress_cv: Condvar,
    pub(super) workers: Mutex<WorkerPool>,
    pub(super) derivation: [u8; 32],
}

pub(super) fn slot_mut(state: &mut PoolState, slot_id: u64) -> Result<&mut Slot, PoolError> {
    state
        .slots
        .get_mut(&slot_id)
        .ok_or_else(|| PoolError(format!("slot {slot_id} not registered")))
}

pub(super) fn slot_ref(state: &PoolState, slot_id: u64) -> Result<&Slot, PoolError> {
    state
        .slots
        .get(&slot_id)
        .ok_or_else(|| PoolError(format!("slot {slot_id} not registered")))
}

impl NativePool {
    pub(super) fn lock_state(&self) -> Result<MutexGuard<'_, PoolState>, PoolError> {
        self.state
            .lock()
            .map_err(|_| PoolError("pool state mutex poisoned".to_string()))
    }

    pub(super) fn lock_game(&self, idx: usize) -> Result<MutexGuard<'_, NativeGame>, PoolError> {
        self.games
            .get(idx)
            .ok_or_else(|| PoolError(format!("game index {idx} outside cohort")))?
            .lock()
            .map_err(|_| PoolError(format!("game {idx} mutex poisoned")))
    }

    pub(super) fn lock_counter(&self, idx: usize) -> Result<MutexGuard<'_, [u64; 4]>, PoolError> {
        self.sample_counters
            .get(idx)
            .ok_or_else(|| PoolError(format!("sample counter {idx} outside cohort")))?
            .lock()
            .map_err(|_| PoolError(format!("sample counter {idx} mutex poisoned")))
    }

    pub(super) fn lock_completed(
        &self,
        idx: usize,
    ) -> Result<MutexGuard<'_, Vec<super::super::core::GameRow>>, PoolError> {
        self.completed
            .get(idx)
            .ok_or_else(|| PoolError(format!("completed buffer {idx} outside cohort")))?
            .lock()
            .map_err(|_| PoolError(format!("completed buffer {idx} mutex poisoned")))
    }

    pub(super) fn lock_workers(&self) -> Result<MutexGuard<'_, WorkerPool>, PoolError> {
        self.workers
            .lock()
            .map_err(|_| PoolError("worker pool mutex poisoned".to_string()))
    }

    /// Open + cold-check: every cohort game opens to its first decision and
    /// every (game, seat) sampler seed derives deterministically from the
    /// profile + derived game ids (cold; identical after restore).
    pub fn open(profile: PoolProfile, derivation: [u8; 32]) -> Result<Arc<Self>, PoolError> {
        Self::open_impl(profile, derivation, None)
    }

    /// Shared open implementation with an optional immutable live source
    /// binding (installed at construction because worker `Weak` handles
    /// forbid post-hoc `Arc::get_mut`).
    fn open_impl(
        profile: PoolProfile,
        derivation: [u8; 32],
        live_binding: Option<trajectory::LiveTapeBinding>,
    ) -> Result<Arc<Self>, PoolError> {
        let split_id = sampler_split_id(&profile.split_scope, profile.generation);
        let mut games = Vec::with_capacity(profile.games.len());
        let mut wall_ids = Vec::with_capacity(profile.games.len());
        let mut counters = Vec::with_capacity(profile.games.len());
        let mut seeds = Vec::with_capacity(profile.games.len());
        let mut completed = Vec::with_capacity(profile.games.len());
        for game in &profile.games {
            let ctx = profile.game_context(game);
            let native = NativeGame::open(ctx, profile.starting_points).map_err(|e| {
                PoolError(format!(
                    "cohort game {} rejected at open: {e}",
                    game.wall_id
                ))
            })?;
            let game_id = native.game_id().to_string();
            let mut game_seeds = [[0u8; 32]; 4];
            for (seat, slot) in game_seeds.iter_mut().enumerate() {
                *slot = sampler_seed(
                    &profile.master_seed_hex,
                    &profile.experiment_id,
                    &split_id,
                    &profile.policy_id,
                    &game_id,
                    seat as u8,
                    profile.replicate_id,
                    profile.attempt_id,
                )
                .map_err(|e| {
                    PoolError(format!(
                        "cohort game {} seat {seat} seed rejected: {e}",
                        game.wall_id
                    ))
                })?;
            }
            games.push(Mutex::new(native));
            wall_ids.push(game.wall_id.clone());
            counters.push(Mutex::new([0u64; 4]));
            seeds.push(game_seeds);
            completed.push(Mutex::new(Vec::new()));
        }
        let games: Arc<Vec<Mutex<NativeGame>>> = Arc::new(games);
        let worker_count = profile.worker_count;
        let cohort = profile.games.len();
        let pool = Arc::new(Self {
            profile,
            games: Arc::clone(&games),
            wall_ids,
            sample_counters: counters,
            sample_seeds: seeds,
            completed,
            live_binding,
            state: Mutex::new(PoolState {
                slots: HashMap::new(),
                leases: HashMap::new(),
                next_lease: 1,
                next_chunk: HashMap::new(),
                poison: None,
                quiesced: false,
                progress: 0,
                pending_applies: Vec::new(),
                apply_outstanding: vec![0usize; cohort],
                fill_inflight: None,
                fill_yield: HashMap::new(),
                advise_cursor: 0,
                admit_ok: 0,
                advise_none_quiesced: 0,
                advise_none_owner_empty: 0,
                advise_none_no_truly_free: 0,
                take_miss: 0,
                take_hit: 0,
                take_blocking: 0,
                last_tape_handle: None,
                drained: false,
                drained_counts: None,
            }),
            progress_cv: Condvar::new(),
            workers: Mutex::new(WorkerPool {
                sender: None,
                handles: vec![],
                fill_sender: None,
                fill_handle: None,
            }),
            derivation,
        });
        // Fixed workers execute game-disjoint applies and observation builds;
        // per-game locks prevent overlap, and lost responses fail batches closed.
        let (sender, receiver) = mpsc::channel::<WorkerJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = vec![];
        for _ in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            let games = Arc::clone(&games);
            // Weak ownership avoids a worker/pool cycle; publishing under the
            // state mutex makes completion visible before a waiter can sleep.
            let pool_weak = Arc::downgrade(&pool);
            let handle = std::thread::spawn(move || {
                loop {
                    let job = match receiver.lock() {
                        Ok(guard) => match guard.recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        },
                        Err(_) => break,
                    };
                    match job {
                        WorkerJob::Apply(job) => {
                            let result = (|| {
                                let game_mutex = games
                                    .get(job.game_idx)
                                    .ok_or_else(|| "game index outside cohort".to_string())?;
                                let mut game = game_mutex
                                    .lock()
                                    .map_err(|_| "game mutex poisoned".to_string())?;
                                for action in &job.actions {
                                    game.apply(action)
                                        .map_err(|e| format!("worker apply rejected: {e}"))?;
                                }
                                if game.state_last_error() {
                                    return Err("core last_error after apply".to_string());
                                }
                                Ok(())
                            })();
                            let _ = job.respond.send(result);
                        }
                        WorkerJob::Observe {
                            game_idx,
                            seat,
                            respond,
                        } => {
                            let result = (|| {
                                let mut game = games
                                    .get(game_idx)
                                    .ok_or_else(|| "game index outside cohort".to_string())?
                                    .lock()
                                    .map_err(|_| "game mutex poisoned".to_string())?;
                                game.observe(seat)
                                    .map_err(|e| format!("worker observe rejected: {e}"))
                            })();
                            let _ = respond.send(result);
                        }
                    }
                    if let Some(pool) = pool_weak.upgrade() {
                        if let Ok(mut state) = pool.lock_state() {
                            pool.notify(&mut state);
                        }
                    }
                }
            });
            handles.push(handle);
        }
        pool.lock_workers()?.sender = Some(sender);
        pool.lock_workers()?.handles = handles;
        // A dedicated fill worker may wait for observation workers; placing it
        // on their queue would deadlock a one-worker pool and lose decisions.
        let (fill_sender, fill_receiver) = mpsc::channel();
        let weak = Arc::downgrade(&pool);
        let fill_handle = std::thread::spawn(move || {
            while let Ok((slot, epoch, bucket, capacity)) = fill_receiver.recv() {
                let Some(pool) = weak.upgrade() else { break };
                let result = pool.run_fill(slot, epoch, bucket, capacity);
                pool.deposit_fill(result);
            }
        });
        let mut workers = pool.lock_workers()?;
        workers.fill_sender = Some(fill_sender);
        workers.fill_handle = Some(fill_handle);
        drop(workers);
        Ok(pool)
    }

    /// Live open: shares the `open` implementation, then installs the
    /// immutable live collector binding. Generation/behavior are compared
    /// against the parsed profile here; the pipeline id is stored and
    /// re-compared at `drain_tape` (the profile carries no pipeline
    /// field). Pools from `open`/`restore` carry no binding and reject
    /// `drain_tape`.
    pub fn open_live(
        profile: PoolProfile,
        derivation: [u8; 32],
        binding: trajectory::LiveTapeBinding,
    ) -> Result<Arc<Self>, PoolError> {
        if binding.generation != profile.generation {
            return Err(PoolError(format!(
                "live binding generation {} mismatches profile generation {}",
                binding.generation, profile.generation
            )));
        }
        if binding.behavior_policy_id != profile.policy_id {
            return Err(PoolError(
                "live binding behavior policy mismatches profile policy".to_string(),
            ));
        }
        Self::open_impl(profile, derivation, Some(binding))
    }

    pub(super) fn fail_if_poisoned(&self, state: &PoolState) -> Result<(), PoolError> {
        if let Some(reason) = &state.poison {
            return Err(PoolError(format!("generation poisoned: {reason}")));
        }
        Ok(())
    }

    pub(super) fn notify(&self, state: &mut PoolState) {
        state.progress += 1;
        self.progress_cv.notify_all();
    }

    /// Sticky poison under the state lock (first reason wins; every later
    /// poll/drain surfaces it).
    pub(super) fn poison_locked(&self, state: &mut PoolState, reason: String) {
        if state.poison.is_none() {
            state.poison = Some(reason);
        }
        self.notify(state);
    }
}
