//! rl/pool: fixed-worker native pool with typed staging + lease state machine.
//!
//! DAG: `super::{actions, core, events, observation}` + std sync only (no
//! extra crates; `Cargo.toml` is registrar-owned). PyO3 appears ONLY in
//! `super::pool_ffi`, never here; GPU/CUDA event queries stay owner-side.
//! FORBIDS: Python callbacks in any worker path, per-row metadata allocation,
//! blocking worker future backpressure, silent pageable fallback, opaque RNG
//! serialization (replay is canonical action prefixes + stream counters),
//! bare `expect`/`unwrap` outside tests (poisoned mutexes fail closed via
//! `PoolError`, mirroring `stream.rs`/`ring.rs`).
//!
//! Contract (plan 4 + Main/ArtifactStorage bindings):
//! - `PoolProfile` (this module's domain; `LoopIntegrate` maps `RlConfig`
//!   into it later): closed JSON, unknown fields rejected. Cohort
//!   wall/game/seat ids fixed before collection; one pending decision per
//!   live game; stable `(policy, T)` grouping before cap splits.
//! - slots: `(T, slot_id)` lanes with fixed capacity derived from registered
//!   buffer sizes. Fills pack up to `cap` eligible decisions (history ceil
//!   == T) or report none; a busy slot errors (admission-only
//!   backpressure, never blocking). Fills run as dispatched worker jobs
//!   (`admit_fill` + nonblocking `take_fill`; the blocking
//!   `request_ready_fill` wraps both), never synchronously on the CUDA
//!   owner thread: the owner admits, keeps the GPU supplied, then takes.
//! - tickets group rows by `(policy_id, T)`; `peek_ready` scans ALL READY
//!   unleased tickets and returns min `(generation, logical_round,
//!   policy_id, T, chunk)`.
//! - lease phases `FREE -> FILLING -> READY -> H2D_ISSUED -> INFER_QUEUED ->
//!   D2H_ISSUED -> HOST_READY -> OWNED_CAPTURE -> FREE`, driven by explicit
//!   owner calls with last-read + D2H attestations. Per-lease ordering only:
//!   the next lease may issue H2D while the previous D2H is still deferred
//!   (deferred-D2H overlap needs no pool change).
//! - `commit_actions` validates the ENTIRE batch (length, ownership, legal
//!   membership, finite logp/value4, `0 < exp(logp) <= 1`, epoch/cursor/policy
//!   match) BEFORE any capture, captures owned rows, retains them per game
//!   for the generation-end tape handoff, releases the slot immediately,
//!   enqueues deterministic worker applies, and returns the COUNT without
//!   waiting for engine applies. Completions are harvested nonblocking by
//!   `poll`/`peek_ready`/fills; worker errors/panics poison loudly there.
//!   Fills skip games with outstanding applies, so complete-round membership
//!   only advances after applies land. `quiesce`/snapshot join admitted
//!   applies while retaining unissued READY rows.
//! - sampling stream: prescribed per-(game, seat) `actor_policy_sample`
//!   semantic key (case = derived game id, candidate = behavior/policy id,
//!   root_seat = seat, generation split scope, replicate/attempt explicit;
//!   seed via `hydra_search::eval::semantic_seed`, words via the feed CTR
//!   window, `u = (w >> 11) * 2^-53`); 8 bytes reserved once per real
//!   decision at fill (forced singletons included), counters checkpointed
//!   per (game, seat), restored exactly (no resampling, no rewinds).
//! - snapshot/restore: applied prefixes + digests + counters + epochs +
//!   retained READY staged rows (uniforms, cursors, obs hashes, leases);
//!   restore replays prefixes against identical walls, verifies digests,
//!   and re-stages retained rows with identical reservations (no dropped
//!   staged decision, no additional draw).
//! - Main descriptors: `game_completions` (per-game terminal/captured/
//!   utility facts) and `export_tape` (generation-end `TapeHandle` with
//!   terminal-frozen rows for the ArtifactStorage seal; rows never cross
//!   Python).
//! - progress notification via condvar (no busy spin, no sleep guesses).
//!
//! Split map (this facade keeps the public surface, the capacity geometry,
//! and the shared closed-JSON helpers; behavior lives in the children, which
//! see this module's privates via `super::` and share state tables via
//! narrowly scoped `pub(super)` — no wider re-exports, no shims):
//! - `profile`: closed `PoolProfile`/sampler streams (parse + seeds + draws).
//! - `state`: slot/lease tables, `NativePool` storage, lock/poison
//!   primitives, `open` (sole constructor; workers spawn here).
//! - `fill`: lane registration, dispatched fills, takes, refills (`Free` ->
//!   `Filling` -> `Ready`; sampler reservations happen only in `run_fill`).
//! - `round_plan`: frozen per-frontier refs, min-cap chunks, head progress,
//!   plus the restore-time semantic check (`validate_restored_plan`).
//! - `apply`: leases, phases, `commit_actions`, worker harvest/join,
//!   poll/wait, quiesce/resume/drain/close (`Ready` -> ... -> `Free`).
//! - `snapshot`: Main descriptors, tape handoff, snapshot/restore/restage,
//!   and the returned status/snapshot types.

mod apply;
mod fill;
mod profile;
pub(crate) mod round_plan;
mod round_plan_runtime;
mod snapshot;
mod state;

pub use profile::{
    CohortGame, PoolProfile, SharedIdentities, sampler_draw, sampler_seed, sampler_split_id,
};
pub use snapshot::{
    DrainStatus, GameCompletion, GameSnapshot, PoolSnapshot, PoolStatus, SlotSnapshot,
    StagedRowSnapshot, StoredAction,
};
pub use state::{Lease, NativePool, SlotPhase, TicketDescriptor};

use super::observation;

/// Output payload is three contiguous planes (CUDA-owner contract, exact):
/// selected `int64[B]` at byte 0, logp `float32[B]` at byte `8*B`, value4
/// `float32[B,4]` row-major at byte `12*B`. Total `28*B` bytes. Decode uses
/// byte-stride-aware `from_le_bytes` reads only (never alignment-assuming
/// casts, never packed-int64 views, never row-interleaved stride-28 reads:
/// for B > 1 the interleaved layout disagrees with the owner planes).
pub const OUTPUT_ROW_STRIDE: usize = 28;
/// Plane bases for B valid rows: ids at `i * 8`, logp at `8 * B + i * 4`,
/// value4 element `k` of row `i` at `12 * B + (i * 4 + k) * 4`.
fn output_plane_bases(n_valid: usize) -> (usize, usize, usize) {
    (0, 8 * n_valid, 12 * n_valid)
}
/// Max rows per ticket (cap128 rings).
pub const TICKET_CAP: usize = 128;
/// Frozen history buckets served by slots.
pub const HISTORY_BUCKETS: [usize; 4] = [32, 64, 128, 256];

/// Native failure (fail-closed text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolError(pub String);

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl pool: {}", self.0)
    }
}

impl std::error::Error for PoolError {}

/// Exact capacity geometry (cold helper for registration).
/// Input = aligned `active_input_bytes(bucket, capacity)` (26 planes +
/// `uniform` f64 with bounded alignment padding); output = capacity * 28.
pub fn buffer_geometry(bucket_t: usize, capacity: usize) -> Result<(usize, usize), PoolError> {
    if !HISTORY_BUCKETS.contains(&bucket_t) {
        return Err(PoolError(format!("bucket T={bucket_t} outside {HISTORY_BUCKETS:?}")));
    }
    if capacity == 0 || capacity > TICKET_CAP {
        return Err(PoolError(format!("capacity {capacity} outside 1..={TICKET_CAP}")));
    }
    Ok((
        observation::active_input_bytes(bucket_t, capacity),
        capacity * OUTPUT_ROW_STRIDE,
    ))
}

/// Ceil history length to the frozen bucket (over-cap fails closed).
pub fn bucket_ceil(actual: usize) -> Result<usize, PoolError> {
    for bucket in HISTORY_BUCKETS {
        if actual <= bucket {
            return Ok(bucket);
        }
    }
    Err(PoolError(format!(
        "visible history {actual} exceeds model bucket cap {}; rows are never truncated",
        HISTORY_BUCKETS[3]
    )))
}

/// Closed-JSON string field (missing/wrong-typed fields fail closed; shared
/// by the profile parser and the snapshot action parser, single owner).
fn json_str(value: &serde_json::Value, key: &str) -> Result<String, PoolError> {
    value.get(key).and_then(|v| v.as_str()).map(str::to_string).ok_or_else(|| {
        PoolError(format!("profile/context missing str field {key:?}"))
    })
}

/// Closed-JSON integer field (same fail-closed ownership as `json_str`).
fn json_u64(value: &serde_json::Value, key: &str) -> Result<u64, PoolError> {
    value.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        PoolError(format!("profile/context missing int field {key:?}"))
    })
}

/// 32-byte hex digest with optional `sha256:` prefix (digests cross the
/// FFI as text; fixed width keeps the sampler/ckpt keys byte-exact).
fn hex32(text: &str, key: &str) -> Result<[u8; 32], PoolError> {
    let noprefix = text.trim_start_matches("sha256:");
    if noprefix.len() != 64 {
        return Err(PoolError(format!("{key} must be 32 bytes hex")));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&noprefix[2 * i..2 * i + 2], 16)
            .map_err(|_| PoolError(format!("{key} must be 32 bytes hex")))?;
    }
    Ok(out)
}

/// Closed-object gate: unknown fields are rejected loudly (profile, context,
/// and cohort games all pin their schema; silent extension would fork
/// sampler keys and snapshot digests).
fn check_closed(value: &serde_json::Value, allowed: &[&str], what: &str) -> Result<(), PoolError> {
    let object =
        value.as_object().ok_or_else(|| PoolError(format!("{what} must be an object")))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(PoolError(format!("{what} rejects unknown field {key:?}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::actions::BASELINE_ACTIONS;
    use super::super::observation;
    use super::*;
    use std::sync::Arc;

    fn profile_json() -> (String, String) {
        let tiles: Vec<u8> = (0u8..136).collect();
        let profile = serde_json::json!({
            "pool_id": "test-pool",
            "generation": 3,
            "policy_id": "ckpt-test",
            "master_seed_hex": "ab".repeat(32),
            "experiment_id": "exp-test",
            "split_scope": "selection_training",
            "replicate_id": 0,
            "attempt_id": 0,
            "starting_points": 25000,
            "worker_count": 2,
            "games": [
                {"wall_id": "w0", "schedule_id": "sched", "schedule_digest": "sha256:ab",
                 "schedule_tiles": tiles, "seat_perm": [0, 1, 2, 3]},
                {"wall_id": "w1", "schedule_id": "sched", "schedule_digest": "sha256:ab",
                 "schedule_tiles": tiles, "seat_perm": [1, 2, 3, 0]},
            ],
        });
        // Valid-shaped distinct sentinels (commit validates 32-byte hex).
        let context = serde_json::json!({
            "rules_id": "test-rules",
            "rules_hash": format!("sha256:{}", "00".repeat(32)),
            "action_table_hash": format!("sha256:{}", "01".repeat(32)),
            "event_schema_hash": format!("sha256:{}", "02".repeat(32)),
            "observation_schema_hash": format!("sha256:{}", "03".repeat(32)),
            "packet_boundary_hash": format!("sha256:{}", "04".repeat(32)),
            "adapter_version": "2.0.0",
            "rank_values": [1.0, 0.0, -0.5, -1.0],
            "behavior_ckpt": "07".repeat(32),
        });
        (profile.to_string(), context.to_string())
    }

    fn open_pool() -> Arc<NativePool> {
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        NativePool::open(profile, [9u8; 32]).unwrap()
    }

    /// Plane-major owner output for `n` rows: first legal id per row from
    /// the descriptor, logp 0 (pi_old = 1, in (0, 1]), value4 zeros.
    fn first_legal_output(pool: &NativePool, lease: &Lease) -> Vec<u8> {
        let descriptor = pool.lease_descriptor(lease).unwrap();
        let rows = descriptor.get("rows").and_then(|r| r.as_array()).unwrap().clone();
        let b = rows.len();
        assert_eq!(b, lease.n_valid);
        let mut out = vec![0u8; b * OUTPUT_ROW_STRIDE];
        for (i, row) in rows.iter().enumerate() {
            let legal = row.get("legal").and_then(|l| l.as_array()).unwrap();
            assert!(!legal.is_empty());
            let id = legal[0].get("id").and_then(|v| v.as_u64()).unwrap();
            out[i * 8..i * 8 + 8].copy_from_slice(&(id as i64).to_le_bytes());
            out[8 * b + i * 4..8 * b + i * 4 + 4].copy_from_slice(&0f32.to_le_bytes());
        }
        out
    }

    fn drive_to_host_ready(pool: &Arc<NativePool>, lease: &Lease) {
        pool.mark_phase(lease, SlotPhase::H2dIssued).unwrap();
        pool.mark_phase(lease, SlotPhase::InferQueued).unwrap();
        pool.mark_phase(lease, SlotPhase::D2hIssued).unwrap();
        pool.mark_phase(lease, SlotPhase::HostReady).unwrap();
    }

    #[test]
    fn profile_rejects_unknown_fields() {
        let (profile, context) = profile_json();
        let mut value: serde_json::Value = serde_json::from_str(&profile).unwrap();
        value["bogus"] = serde_json::Value::from(1);
        assert!(PoolProfile::parse(&value.to_string(), &context).is_err());
    }

    #[test]
    fn profile_requires_sampler_scope() {
        let (profile, context) = profile_json();
        let mut value: serde_json::Value = serde_json::from_str(&profile).unwrap();
        value.as_object_mut().unwrap().remove("experiment_id");
        assert!(PoolProfile::parse(&value.to_string(), &context).is_err());
        let (profile, context) = profile_json();
        let mut value: serde_json::Value = serde_json::from_str(&profile).unwrap();
        value["split_scope"] = serde_json::Value::from("final_evaluation");
        assert!(PoolProfile::parse(&value.to_string(), &context).is_err());
    }

    #[test]
    fn geometry_matches_aligned_capacity() {
        // Cap128 carries zero alignment padding (every region a multiple
        // of 8); odd capacities carry bounded deterministic padding.
        assert_eq!(buffer_geometry(32, 128).unwrap(), (128 * (7254 + 288) + 1024, 128 * 28));
        assert!(buffer_geometry(33, 128).is_err());
        assert!(buffer_geometry(32, 129).is_err());
        for t in [32usize, 64, 128, 256] {
            for cap in [1usize, 3, 7, 64, 128] {
                let (input, output) = buffer_geometry(t, cap).unwrap();
                assert_eq!(input, observation::active_input_bytes(t, cap));
                assert_eq!(output, cap * OUTPUT_ROW_STRIDE);
            }
        }
        assert_eq!(bucket_ceil(0).unwrap(), 32);
        assert_eq!(bucket_ceil(193).unwrap(), 256);
        assert!(bucket_ceil(257).is_err());
    }

    #[test]
    fn sampler_key_is_prescribed_and_per_seat() {
        assert_eq!(sampler_split_id("selection_training", 3), "selection_training:g=3");
        let seed = |seat| {
            sampler_seed(
                &"ab".repeat(32),
                "exp-test",
                "selection_training:g=3",
                "ckpt-test",
                "game-x",
                seat,
                0,
                0,
            )
            .unwrap()
        };
        // Deterministic per (game, seat); seats never share a stream.
        assert_eq!(seed(0), seed(0));
        assert_ne!(seed(0), seed(1));
        assert_ne!(seed(0), seed(2));
        // Counters advance independently without redrawing.
        let (c0, u0) = sampler_draw(&seed(0), 0).unwrap();
        let (c1, u1) = sampler_draw(&seed(0), 1).unwrap();
        assert_eq!((c0, c1), (0, 8));
        assert!((0.0..1.0).contains(&u0));
        assert!((0.0..1.0).contains(&u1));
        // u53 upper endpoint over the full word space.
        let top = ((u64::MAX >> 11) as f64) / 9007199254740992.0;
        assert_eq!(top, 0.9999999999999999);
        // Empty identities fail closed (never a default stream).
        assert!(sampler_seed(&"ab".repeat(32), "", "s", "c", "g", 0, 0, 0).is_err());
        assert!(sampler_seed(&"ab".repeat(32), "e", "s", "c", "g", 4, 0, 0).is_err());
    }

    #[test]
    fn lease_rejects_stale_double_and_bad_phases() {
        let pool = open_pool();
        pool.register_slot(0, 32, 4).unwrap();
        assert!(pool.register_slot(0, 32, 4).is_err());
        let lease = Lease {
            lease_id: 999,
            slot_id: 0,
            slot_epoch: 0,
            generation: 3,
            policy_id: "ckpt-test".to_string(),
            bucket_t: 32,
            chunk: 0,
            n_valid: 0,
            logical_round: 0,
        };
        assert!(pool.mark_phase(&lease, SlotPhase::H2dIssued).is_err());
        assert!(pool.commit_actions(&lease, &[]).is_err());
    }

    #[test]
    fn plane_major_commit_returns_count_without_blocking() {
        let pool = open_pool();
        // Fresh games stage only T=32 rows (history starts empty); larger
        // lanes stall work-conserving, so failure gates use the live lane.
        pool.register_slot(7, 32, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(7, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let lease = pool.peek_ready().unwrap().expect("ready lease");
        assert_eq!(lease.n_valid, ticket.n_valid);
        // Second peek must not re-issue the leased slot.
        assert!(pool.peek_ready().unwrap().is_none());
        drive_to_host_ready(&pool, &lease);
        let output = first_legal_output(&pool, &lease);
        // Commit returns the count immediately (applies land async).
        let count = pool.commit_actions(&lease, &output).unwrap();
        assert_eq!(count, lease.n_valid);
        // Slot is reusable at once (no blocked caller waits on applies).
        let status = pool.poll().unwrap();
        assert!(status.poison.is_none());
        // Double commit of the same lease fails (stale lease).
        assert!(pool.commit_actions(&lease, &output).is_err());
        // Quiesce joins the admitted applies; rows are retained per game.
        pool.quiesce().unwrap();
        let status = pool.poll().unwrap();
        assert_eq!(status.pending_applies, 0);
        assert_eq!(status.captured_rows, count);
        let completions = pool.game_completions().unwrap();
        assert_eq!(completions.iter().map(|c| c.captured).sum::<u64>() as usize, count);
    }

    #[test]
    fn nonfinite_commit_poisons_deterministically() {
        let pool = open_pool();
        pool.register_slot(7, 32, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(7, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let lease = pool.peek_ready().unwrap().expect("ready lease");
        drive_to_host_ready(&pool, &lease);
        // NaN logp in row 0 of the OWNER logp plane: entire batch rejected
        // + poisoned (plane-major offsets, never interleaved rows).
        let b = lease.n_valid;
        let mut output = vec![0u8; b * OUTPUT_ROW_STRIDE];
        output[8 * b..8 * b + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(pool.commit_actions(&lease, &output).is_err());
        assert!(pool.poll().unwrap().poison.is_some());
    }

    #[test]
    fn illegal_selection_fails_the_whole_batch() {
        let pool = open_pool();
        pool.register_slot(7, 32, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(7, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let lease = pool.peek_ready().unwrap().expect("ready lease");
        drive_to_host_ready(&pool, &lease);
        let mut output = first_legal_output(&pool, &lease);
        // Select an action outside the served legal set (id past the end).
        let illegal = (BASELINE_ACTIONS as i64).to_le_bytes();
        output[0..8].copy_from_slice(&illegal);
        assert!(pool.commit_actions(&lease, &output).is_err());
        assert!(pool.poll().unwrap().poison.is_some());
    }

    #[test]
    fn commit_rejected_while_fill_in_flight() {
        let pool = open_pool();
        pool.register_slot(7, 32, 8).unwrap();
        pool.register_slot(8, 32, 1).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let (input_need8, output_need8) = buffer_geometry(32, 1).unwrap();
        // Settle lane 8 to a committable lease first. Cap 1 stages a single
        // game, leaving the other game free so lane 7's in-flight fill has
        // rows to take (a cap-8 lane would pack both games and starve it).
        let mut input8 = vec![0u8; input_need8];
        let ticket8 = pool.request_ready_fill(8, &mut input8, output_need8).unwrap();
        let lease8 = pool.peek_ready().unwrap().expect("ready lease");
        drive_to_host_ready(&pool, &lease8);
        // Admit lane 7 without taking: commits are rejected until taken.
        let mut input = vec![0u8; input_need];
        pool.admit_fill(7, output_need).unwrap();
        assert!(pool.admit_fill(7, output_need).is_err());
        let output8 = first_legal_output(&pool, &lease8);
        assert!(pool.commit_actions(&lease8, &output8).is_err());
        let ticket = pool.take_fill_blocking(7, &mut input).unwrap();
        assert!(ticket.n_valid > 0);
        // After the take the earlier lease commits cleanly.
        let count = pool.commit_actions(&lease8, &output8).unwrap();
        assert_eq!(count, lease8.n_valid);
        pool.quiesce().unwrap();
    }
    #[test]
    fn snapshot_restore_round_trip() {
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        let pool = NativePool::open(profile.clone(), [2u8; 32]).unwrap();
        pool.register_slot(1, 256, 8).unwrap();
        let snap = pool.snapshot().unwrap();
        assert_eq!(snap.games.len(), 2);
        let restored = NativePool::restore(profile, &snap, [2u8; 32]).unwrap();
        let snap2 = restored.snapshot().unwrap();
        for (a, b) in snap.games.iter().zip(snap2.games.iter()) {
            assert_eq!(a.digest, b.digest);
            assert_eq!(a.sample_counters, b.sample_counters);
            assert_eq!(a.completed.len(), b.completed.len());
        }
    }

    #[test]
    fn restore_rejects_retained_rows_without_round_plan() {
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        let pool = NativePool::open(profile.clone(), [3u8; 32]).unwrap();
        pool.register_slot(1, 32, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(1, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let mut snap = pool.snapshot().unwrap();
        snap.round_plan = None;
        // Retained reservations require their frozen plan; dropping it must
        // reject instead of bypassing frontier and bucket verification.
        match NativePool::restore(profile, &snap, [3u8; 32]) {
            Ok(restored) => {
                restored.close().unwrap();
                panic!("restore accepted retained rows without a frozen plan");
            }
            Err(_) => {}
        }
    }

    #[test]
    fn forked_restore_preserves_ready_reservations() {
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        let pool = NativePool::open(profile.clone(), [3u8; 32]).unwrap();
        pool.register_slot(1, 32, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(32, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(1, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        // Fork BEFORE peek/commit: unissued READY rows must survive with
        // identical uniforms, cursors, and descriptors (no redraw).
        let snap = pool.snapshot().unwrap();
        let fork = NativePool::restore(profile, &snap, [3u8; 32]).unwrap();
        fork.attach_slot(1, 32, 8).unwrap();
        let mut fork_input = vec![0u8; input_need];
        let refilled = fork.refill_staged(1, &mut fork_input).unwrap();
        assert_eq!(refilled, ticket.n_valid);
        let active = observation::active_input_bytes(32, ticket.n_valid);
        assert_eq!(fork_input[..active], input[..active]);
        let lease = fork.peek_ready().unwrap().expect("ready lease");
        assert_eq!(lease.n_valid, ticket.n_valid);
        drive_to_host_ready(&fork, &lease);
        let output = first_legal_output(&fork, &lease);
        let count = fork.commit_actions(&lease, &output).unwrap();
        assert_eq!(count, ticket.n_valid);
        fork.quiesce().unwrap();
        // Cursors advanced exactly once per staged decision on both sides.
        let snap2 = fork.snapshot().unwrap();
        for (a, b) in snap.games.iter().zip(snap2.games.iter()) {
            assert_eq!(a.sample_counters, b.sample_counters);
        }
    }

    #[test]
    fn snapshot_carries_writing_derivation() {
        // The snapshot root binds the writing run's row-derivation digest so
        // restores can reject foreign scheduling/provenance laws outright.
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        let pool = NativePool::open(profile, [9u8; 32]).unwrap();
        let snap = pool.snapshot().unwrap();
        assert_eq!(snap.derivation, [9u8; 32]);
    }

    #[test]
    fn restore_rejects_foreign_derivation() {
        // Same bytes, foreign derivation: restore must fail BEFORE opening
        // or replaying anything — old-law rows must never mix into this run.
        let (profile, context) = profile_json();
        let profile = PoolProfile::parse(&profile, &context).unwrap();
        let pool = NativePool::open(profile.clone(), [9u8; 32]).unwrap();
        let snap = pool.snapshot().unwrap();
        assert!(NativePool::restore(profile, &snap, [1u8; 32]).is_err());
    }

    fn live_binding_for(profile: &PoolProfile) -> super::super::trajectory::LiveTapeBinding {
        super::super::trajectory::LiveTapeBinding {
            pipeline_spec_id: format!("sha256:{}", "aa".repeat(32)),
            generation: profile.generation,
            behavior_policy_id: profile.policy_id.clone(),
        }
    }

    fn open_live_pool() -> Arc<NativePool> {
        let (profile_json, context) = profile_json();
        let profile = PoolProfile::parse(&profile_json, &context).unwrap();
        let binding = live_binding_for(&profile);
        NativePool::open_live(profile, [9u8; 32], binding).unwrap()
    }

    #[test]
    fn live_mixed_transport_freezes_T256_with_original_order() {
        let pool = open_live_pool();
        pool.register_slot(7, 256, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(256, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(7, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        assert_eq!(ticket.bucket_t, 256);
        let lease = pool.peek_ready().unwrap().expect("ready lease");
        assert_eq!(lease.bucket_t, 256);
        // Live references before exported changes: wall/game identity rides
        // the descriptor, never a tape handle.
        let descriptor = pool.lease_descriptor(&lease).unwrap();
        let rows = descriptor.get("rows").and_then(|r| r.as_array()).unwrap();
        assert_eq!(rows.len(), lease.n_valid);
        for row in rows {
            let wall = row.get("wall_id").and_then(|w| w.as_str()).unwrap();
            assert!(wall == "w0" || wall == "w1");
            assert!(row.get("game_id").and_then(|g| g.as_str()).is_some());
        }
        // No missing refs: every staged row appears in the frozen T256 plan
        // with its true original bucket derived from history length.
        let plan = pool.snapshot().unwrap().round_plan.expect("live plan");
        assert!(plan.refs[0].is_empty() && plan.refs[1].is_empty() && plan.refs[2].is_empty());
        assert_eq!(plan.refs[3].len(), lease.n_valid);
        let mut prev: Option<(usize, usize, u8)> = None;
        let order = |b: usize| [32, 64, 128, 256].iter().position(|x| *x == b).unwrap();
        for row in &plan.refs[3] {
            let key = (order(row.original_bucket), row.game_idx, row.actor);
            if let Some(p) = prev {
                assert!(p < key, "live T256 refs must hold (original_bucket, game, actor) order");
            }
            prev = Some(key);
        }
        // Dense T256 preserves every real mask/kind prefix; tails stay zero.
        let layout = observation::active_layout(256, lease.n_valid, true);
        let (mut kind_off, mut mask_off) = (0usize, 0usize);
        for (name, _, _, off) in &layout {
            if name == "history_event_kind" {
                kind_off = *off;
            } else if name == "history_mask" {
                mask_off = *off;
            }
        }
        let staged_lengths: Vec<usize> = {
            let state = pool.lock_state().unwrap();
            let slot = state.slots.get(&lease.slot_id).unwrap();
            slot.rows.iter().map(|r| r.obs.history_indices.len()).collect()
        };
        for (r, len) in staged_lengths.iter().enumerate() {
            for pos in 0..*len {
                assert_eq!(input[mask_off + r * 256 + pos], 1);
                let at = kind_off + (r * 256 + pos) * 8;
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&input[at..at + 8]);
                assert_ne!(i64::from_le_bytes(bytes), 0xffff_ffff_ffff_ffffu64 as i64);
            }
            for pos in *len..256 {
                assert_eq!(input[mask_off + r * 256 + pos], 0);
                let at = kind_off + (r * 256 + pos) * 8;
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&input[at..at + 8]);
                assert_eq!(i64::from_le_bytes(bytes), 0);
            }
        }
        drive_to_host_ready(&pool, &lease);
        let output = first_legal_output(&pool, &lease);
        let count = pool.commit_actions(&lease, &output).unwrap();
        assert_eq!(count, lease.n_valid);
        pool.quiesce().unwrap();
    }

    #[test]
    fn archive_vs_live_first_legal_parity_over_closed_frontiers() {
        use std::collections::{BTreeMap, BTreeSet};
        // Independent deterministic first-legal drives to completion:
        // archive per-bucket lanes and live T256 transport must reach equal
        // per-game terminal digests/outcomes with equal legal sets at every
        // decision. Uniforms are ignored by first-legal choice, so the live
        // binding hash never forks the comparison. The bound guarantees a
        // genuine mixed history (32 with 128/256 across varied games).
        let (profile_json, context) = profile_json();
        let profile = PoolProfile::parse(&profile_json, &context).unwrap();
        let archive = NativePool::open(profile.clone(), [9u8; 32]).unwrap();
        for (id, bucket) in [(10u64, 32usize), (11, 64), (12, 128), (13, 256)] {
            archive.register_slot(id, bucket, 8).unwrap();
        }
        let live_binding = live_binding_for(&profile);
        let live = NativePool::open_live(profile.clone(), [9u8; 32], live_binding).unwrap();
        live.register_slot(20, 256, 8).unwrap();
        let drain_one = |pool: &Arc<NativePool>, slots: &[(u64, usize)], decisions: &mut std::collections::BTreeMap<(String, u8, u64), Vec<u64>>| -> bool {
            for (slot_id, bucket) in slots {
                let (input_need, output_need) = buffer_geometry(*bucket, 8).unwrap();
                let mut input = vec![0u8; input_need];
                let ticket = match pool.request_ready_fill(*slot_id, &mut input, output_need) {
                    Ok(ticket) => ticket,
                    Err(_) => continue,
                };
                if ticket.n_valid == 0 {
                    continue;
                }
                let lease = pool.peek_ready().unwrap().expect("ready lease");
                let descriptor = pool.lease_descriptor(&lease).unwrap();
                let rows = descriptor.get("rows").and_then(|r| r.as_array()).unwrap().clone();
                assert_eq!(rows.len(), lease.n_valid);
                for row in &rows {
                    let key = (
                        row.get("wall_id").and_then(|w| w.as_str()).unwrap().to_string(),
                        row.get("actor").and_then(|a| a.as_u64()).unwrap() as u8,
                        row.get("sequence").and_then(|s| s.as_u64()).unwrap(),
                    );
                    let legal: Vec<u64> = row
                        .get("legal")
                        .and_then(|l| l.as_array())
                        .unwrap()
                        .iter()
                        .map(|e| e.get("id").and_then(|v| v.as_u64()).unwrap())
                        .collect();
                    assert!(!legal.is_empty());
                    if let Some(prior) = decisions.insert(key.clone(), legal.clone()) {
                        assert_eq!(prior, legal, "legal drift at {key:?}");
                    }
                }
                drive_to_host_ready(pool, &lease);
                let output = first_legal_output(pool, &lease);
                assert_eq!(pool.commit_actions(&lease, &output).unwrap(), lease.n_valid);
                pool.quiesce().unwrap();
                pool.resume_admission().unwrap();
                return true;
            }
            false
        };
        let all_terminal = |pool: &Arc<NativePool>| -> bool {
            pool.snapshot().unwrap().games.iter().all(|g| g.terminal)
        };
        let archive_slots = [(10u64, 32usize), (11, 64), (12, 128), (13, 256)];
        let live_slots = [(20u64, 256usize)];
        let mut archive_decisions: BTreeMap<(String, u8, u64), Vec<u64>> = BTreeMap::new();
        let mut live_decisions: BTreeMap<(String, u8, u64), Vec<u64>> = BTreeMap::new();
        let mut live_orig_seen: BTreeSet<usize> = BTreeSet::new();
        let mut live_max_len: usize = 0;
        for _ in 0..2000 {
            if all_terminal(&archive) {
                break;
            }
            assert!(drain_one(&archive, &archive_slots, &mut archive_decisions), "archive stalled before terminal");
        }
        assert!(all_terminal(&archive), "archive never reached hanchan end");
        for _ in 0..2000 {
            if all_terminal(&live) {
                break;
            }
            let (input_need, output_need) = buffer_geometry(256, 8).unwrap();
            let mut input = vec![0u8; input_need];
            let ticket = live.request_ready_fill(20, &mut input, output_need).unwrap();
            if ticket.n_valid == 0 {
                assert!(all_terminal(&live), "live stalled with empty fill before terminal");
                break;
            }
            let lease = live.peek_ready().unwrap().expect("ready lease");
            // READY-retained measurement, before commit clears the slot:
            // frozen plan originals plus staged true lengths.
            {
                let state = live.lock_state().unwrap();
                if let Some(plan) = state.round_plan.as_ref() {
                    for row in &plan.refs[3] {
                        live_orig_seen.insert(row.original_bucket);
                    }
                }
                let slot = state.slots.get(&lease.slot_id).unwrap();
                for row in &slot.rows {
                    if let Ok(bucket) = super::bucket_ceil(row.obs.history_indices.len()) {
                        live_orig_seen.insert(bucket);
                    }
                    live_max_len = live_max_len.max(row.obs.history_indices.len());
                }
            }
            let descriptor = live.lease_descriptor(&lease).unwrap();
            let rows = descriptor.get("rows").and_then(|r| r.as_array()).unwrap().clone();
            assert_eq!(rows.len(), lease.n_valid);
            for row in &rows {
                let key = (
                    row.get("wall_id").and_then(|w| w.as_str()).unwrap().to_string(),
                    row.get("actor").and_then(|a| a.as_u64()).unwrap() as u8,
                    row.get("sequence").and_then(|s| s.as_u64()).unwrap(),
                );
                let legal: Vec<u64> = row
                    .get("legal")
                    .and_then(|l| l.as_array())
                    .unwrap()
                    .iter()
                    .map(|e| e.get("id").and_then(|v| v.as_u64()).unwrap())
                    .collect();
                assert!(!legal.is_empty());
                if let Some(prior) = live_decisions.insert(key.clone(), legal.clone()) {
                    assert_eq!(prior, legal, "legal drift at {key:?}");
                }
            }
            drive_to_host_ready(&live, &lease);
            let output = first_legal_output(&live, &lease);
            assert_eq!(live.commit_actions(&lease, &output).unwrap(), lease.n_valid);
            live.quiesce().unwrap();
            live.resume_admission().unwrap();
        }
        assert!(all_terminal(&live), "live never reached hanchan end");
        // Genuine mixed history: initial T32 plus a later 128/256 bucket
        // across varied games, with >128 real events observed.
        assert!(live_orig_seen.contains(&32), "live never saw original T32, got {live_orig_seen:?}");
        assert!(
            live_orig_seen.contains(&128) || live_orig_seen.contains(&256),
            "live never reached mixed 128/256, got {live_orig_seen:?}"
        );
        assert!(live_max_len > 64, "live max history {live_max_len} never exceeded T64");
        // Per-decision legal parity across the full hanchan.
        assert_eq!(archive_decisions, live_decisions);
        let archive_snap = archive.snapshot().unwrap();
        let live_snap = live.snapshot().unwrap();
        assert_eq!(archive_snap.games.len(), live_snap.games.len());
        for (a, b) in archive_snap.games.iter().zip(live_snap.games.iter()) {
            assert_eq!(a.digest, b.digest);
            assert_eq!(a.captured, b.captured);
            assert_eq!(a.completed.len(), b.completed.len());
            assert_eq!(a.terminal, b.terminal);
        }
    }

    #[test]
    fn packed_metadata_prefix_empty_bounds() {
        let pool = open_live_pool();
        pool.register_slot(30, 256, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(256, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(30, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let lease = pool.peek_ready().unwrap().expect("ready lease");
        // Getter safe at READY, before any H2D issue.
        let total = pool.packed_tokens(&lease).unwrap();
        let staged_total: usize = {
            let state = pool.lock_state().unwrap();
            let slot = state.slots.get(&lease.slot_id).unwrap();
            slot.rows.iter().map(|r| r.obs.history_indices.len()).sum()
        };
        assert_eq!(total, staged_total);
        // Exact layout: kind at 0, cu at 8*C, align8, row/pos/valid.
        let capacity = 2048usize;
        let layout = pool.packed_layout_for_lease(&lease, capacity).unwrap();
        let mut offsets = std::collections::HashMap::new();
        for (name, _, _, off) in &layout {
            offsets.insert(name.clone(), *off);
        }
        assert_eq!(offsets["packed_kind"], 0);
        assert_eq!(offsets["cu_seqlens"], 8 * capacity);
        let aligned = ((8 * capacity + (lease.n_valid + 1) * 4 + 7) / 8) * 8;
        assert_eq!(offsets["token_row"], aligned);
        assert_eq!(offsets["token_pos"], aligned + 8 * capacity);
        assert_eq!(offsets["valid"], aligned + 16 * capacity);
        let need = pool.packed_bytes_for_lease(&lease, capacity).unwrap();
        assert_eq!(need, aligned + 16 * capacity + capacity);
        // Bounds: under-capacity and over-max fail closed.
        if total > 0 {
            assert!(pool.packed_bytes_for_lease(&lease, total.saturating_sub(1)).is_ok());
            let mut tiny = vec![0u8; pool.packed_bytes_for_lease(&lease, total).unwrap().max(1)];
            assert!(pool.fill_packed(&lease, &input, &mut tiny, total.saturating_sub(1)).is_err());
        }
        assert!(pool.packed_layout_for_lease(&lease, 32769).is_err());
        // Fill and verify prefix boundaries, positions, tails.
        let mut meta = vec![0u8; need];
        pool.fill_packed(&lease, &input, &mut meta, capacity).unwrap();
        let read_i64 = |at: usize| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&meta[at..at + 8]);
            i64::from_le_bytes(bytes)
        };
        let read_i32 = |at: usize| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&meta[at..at + 4]);
            i32::from_le_bytes(bytes)
        };
        assert_eq!(read_i32(offsets["cu_seqlens"]), 0);
        assert_eq!(read_i32(offsets["cu_seqlens"] + lease.n_valid * 4) as usize, total);
        for i in 0..lease.n_valid {
            let cu = read_i32(offsets["cu_seqlens"] + i * 4) as usize;
            let next = read_i32(offsets["cu_seqlens"] + (i + 1) * 4) as usize;
            assert!(next >= cu);
        }
        for t in 0..total {
            assert_eq!(meta[offsets["valid"] + t], 1);
            assert!(read_i64(offsets["token_row"] + t * 8) >= 0);
            assert!(read_i64(offsets["token_pos"] + t * 8) >= 0);
        }
        for t in total..capacity {
            assert_eq!(meta[offsets["valid"] + t], 0);
            assert_eq!(read_i64(offsets["packed_kind"] + t * 8), 0);
            assert_eq!(read_i64(offsets["token_row"] + t * 8), 0);
            assert_eq!(read_i64(offsets["token_pos"] + t * 8), 0);
        }
        drive_to_host_ready(&pool, &lease);
        let output = first_legal_output(&pool, &lease);
        pool.commit_actions(&lease, &output).unwrap();
        pool.quiesce().unwrap();
    }

    #[test]
    fn live_ready_restore_preserves_mixed_order_and_binding() {
        let (profile_json, context) = profile_json();
        let profile = PoolProfile::parse(&profile_json, &context).unwrap();
        let binding = live_binding_for(&profile);
        let pool = NativePool::open_live(profile.clone(), [9u8; 32], binding.clone()).unwrap();
        pool.register_slot(40, 256, 8).unwrap();
        let (input_need, output_need) = buffer_geometry(256, 8).unwrap();
        let mut input = vec![0u8; input_need];
        let ticket = pool.request_ready_fill(40, &mut input, output_need).unwrap();
        assert!(ticket.n_valid > 0);
        let snap = pool.snapshot().unwrap();
        let plan = snap.round_plan.clone().expect("live plan");
        // Required live_binding preserves the mixed stage ordering.
        let saved = snap.live_binding.clone().expect("live binding");
        assert_eq!(saved.generation, profile.generation);
        assert_eq!(saved.behavior_policy_id, profile.policy_id);
        assert_eq!(saved.pipeline_spec_id, binding.pipeline_spec_id);
        let json = snap.to_json();
        assert!(json.get("live_binding").is_some());
        assert_ne!(json.get("live_binding").unwrap(), &serde_json::Value::Null);
        // Restore keeps the exact T256 original-bucket order (32→256 law:
        // every retained row's derived bucket equals its frozen original).
        let restored = NativePool::restore(profile.clone(), &snap, [9u8; 32]).unwrap();
        let restored_snap = restored.snapshot().unwrap();
        assert_eq!(restored_snap.live_binding.unwrap().pipeline_spec_id, binding.pipeline_spec_id);
        let restored_plan = restored_snap.round_plan.expect("restored plan");
        assert_eq!(restored_plan.refs[3], plan.refs[3]);
        restored.attach_slot(40, 256, 8).unwrap();
        let mut restored_input = vec![0u8; input_need];
        let refilled = restored.refill_staged(40, &mut restored_input).unwrap();
        assert_eq!(refilled, ticket.n_valid);
        let active = observation::active_input_bytes(256, ticket.n_valid);
        assert_eq!(restored_input[..active], input[..active]);
        // Pending original refs still bind after restore.
        for row in &restored_plan.refs[3] {
            assert!([32, 64, 128, 256].contains(&row.original_bucket));
        }
    }

    #[test]
    fn mixed_freeze_splits_same_game_across_chunks_without_skipping() {
        use super::round_plan::RoundPlan;
        // Same game seats reordered by original bucket: cohort order alone
        // would interleave them, but live transport sorts by original first,
        // so one game's seats can land in different chunks. Pins must wait,
        // never skip the later chunk.
        let plan = RoundPlan::freeze(
            0,
            [
                vec![],
                vec![],
                vec![],
                vec![
                    super::round_plan::PlanRow { game_idx: 0, actor: 0, seq: 1, original_bucket: 32 },
                    super::round_plan::PlanRow { game_idx: 1, actor: 0, seq: 2, original_bucket: 64 },
                    super::round_plan::PlanRow { game_idx: 0, actor: 1, seq: 3, original_bucket: 256 },
                ],
            ],
            [None, None, None, Some(2)],
        )
        .unwrap();
        assert!(plan.is_mixed());
        let (ordinal, rows) = plan.head_rows(256).unwrap().unwrap();
        assert_eq!(ordinal, 0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].original_bucket, 32);
        assert_eq!(rows[1].original_bucket, 64);
    }
}
