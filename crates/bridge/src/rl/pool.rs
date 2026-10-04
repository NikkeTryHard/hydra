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
//! - `apply`: leases, phases, `commit_actions`, worker harvest/join,
//!   poll/wait, quiesce/resume/drain/close (`Ready` -> ... -> `Free`).
//! - `snapshot`: Main descriptors, tape handoff, snapshot/restore/restage,
//!   and the returned status/snapshot types.

mod apply;
mod fill;
mod profile;
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
}
