//! Frozen replay handles: process-once registration for one generation.
//!
//! The hot fill re-parsed the full order index and metas JSON and rebuilt
//! every layout on every batch. This registry parses and validates both
//! documents exactly once per generation/resume (`register_replay_from_bytes`),
//! freezes typed meta rows, and precomputes per-batch geometry through the
//! single shared [`super::fill::plane_table`] home the cold export also uses,
//! so both laws agree by construction. The production fill then serves only
//! from the handle plus the live tape: no index bytes, no metas path, no
//! re-parse, no slab-sized intermediate on the hot path.
//!
//! Lock discipline (replay -> tape, never the reverse): the frozen entries
//! are immutable after registration, so the fill holds the replay lock across
//! one batch while taking short per-row tape holds. That hold blocks only
//! register/drop (once per generation), never tape progress or the sealer.
//! A poisoned registry fails every step closed; an unknown handle fails
//! closed (stale use-after-drop is loud, never a silent reuse).

use serde_json::Value;

use crate::rl::artifact::parse_metas;
use crate::rl::trajectory::{TapeError, TapeHandle, parse_digest_text};

use super::fill::plane_table;
use super::{BatchPlan, FACTS48_STRIDE, OPTIMIZER_CAP};

/// Process handle for one registered frozen replay (one generation's order
/// index plus its metas companion, parsed and validated once). Handles start
/// at 1; 0 is never issued (0 = absent, mirroring the tape registry).
pub type ReplayHandle = u64;

/// One canonical row's frozen learner facts, parsed once at registration from
/// the metas companion. The `as f32`/`as_f64` widening matches the cold
/// export's per-fill reads exactly; the finite gates reject a corrupt
/// companion at registration instead of mid-fill (honest seal output always
/// passes, so cold/frozen parity is unaffected).
#[derive(Debug, Clone)]
pub struct FrozenMetaRow {
    /// Tape address back into the registered tape (bounds checked per fill,
    /// since registration is tape-independent).
    pub gi: usize,
    pub ri: usize,
    pub old_lp: f32,
    pub raw_a: f32,
    pub g4: [f64; 4],
}

/// One optimizer batch with geometry precomputed at registration. The spans
/// and layout come from the shared plane table, never a second byte law;
/// the envelope and checksums still compute once per real fill because they
/// hash the filled bytes.
#[derive(Debug, Clone)]
pub struct FrozenBatch {
    pub t: u32,
    pub count: usize,
    /// Tight encoder bytes per row `S(T)` (encoder-drift gate per fill).
    pub row_bytes: usize,
    /// Aligned slab bytes for `count` rows (span gate per fill).
    pub feature_bytes: usize,
    /// Always `count * 48` (span gate per fill).
    pub fact_bytes: usize,
    /// Canonical metas positions, in serving order.
    pub positions: Vec<u32>,
    /// Plane transpose spans `(row_off, base, plane_len)` in plane order.
    pub spans: Vec<(usize, usize, usize)>,
    /// Spec layout `(name, dtype, shape, offset)` for the descriptor.
    pub layout: Vec<(String, String, Vec<usize>, usize)>,
    /// Rows frozen before this batch in the same order (envelope scope).
    pub offset: u64,
}

/// Frozen generation replay: validated plan plus typed meta rows plus
/// precomputed batch geometry. Immutable after registration.
#[derive(Debug)]
pub struct FrozenReplay {
    pub source: ReplaySource,
    pub row_count: usize,
    pub metas: Vec<FrozenMetaRow>,
    pub batches: Vec<FrozenBatch>,
}

/// Replay provenance: sealed archive replays carry their artifact/order
/// digests (verified against the tape seal stamp per fill); live replays
/// carry the exact drained tape handle plus its verified source binding
/// (matched per fill, never a fabricated canonical digest).
#[derive(Debug, Clone)]
pub enum ReplaySource {
    Sealed {
        artifact_digest: String,
        order_digest: String,
    },
    Live {
        tape_handle: TapeHandle,
        pipeline_spec_id: String,
        generation: u64,
        behavior_policy_id: String,
    },
}

struct RegistryInner {
    replays: std::collections::HashMap<ReplayHandle, FrozenReplay>,
    next: ReplayHandle,
}

static REPLAY_REGISTRY: std::sync::LazyLock<std::sync::Mutex<RegistryInner>> =
    std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(RegistryInner {
            replays: std::collections::HashMap::new(),
            next: 1,
        })
    });

fn registry() -> &'static std::sync::Mutex<RegistryInner> {
    &REPLAY_REGISTRY
}

fn replay_err(detail: String) -> TapeError {
    TapeError(detail)
}

/// Freeze one generation's order index plus metas companion (cold, exactly
/// once per generation/resume): validates the artifact binding, the row
/// counts, and exact-once position coverage, freezes typed meta rows, and
/// precomputes every batch's geometry. Returns the process replay handle.
/// Structural corruption fails here; tape binding is checked per fill.
pub fn register_replay_from_bytes(
    index_bytes: &[u8],
    metas_bytes: &[u8],
) -> Result<ReplayHandle, TapeError> {
    let plan = super::parse_index(index_bytes)?;
    let metas = parse_metas(metas_bytes)?;
    let artifact_digest = metas
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| replay_err("metas lacks artifact_digest".to_string()))?
        .to_string();
    parse_digest_text(&artifact_digest)?;
    if artifact_digest != plan.artifact_digest {
        return Err(replay_err(
            "index artifact digest mismatches the metas companion".to_string(),
        ));
    }
    let meta_rows = metas
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| replay_err("metas lacks rows".to_string()))?;
    if plan.row_count != meta_rows.len() {
        return Err(replay_err(format!(
            "order covers {} rows, metas carries {}",
            plan.row_count,
            meta_rows.len()
        )));
    }
    // Freeze typed meta rows once (the cold export re-reads these JSON
    // fields on every fill; identical `as f32` law, finite-gated).
    let mut frozen = Vec::with_capacity(meta_rows.len());
    for meta in meta_rows {
        let gi = meta
            .get("gi")
            .and_then(Value::as_u64)
            .ok_or_else(|| replay_err("metas row lacks gi".to_string()))?
            as usize;
        let ri = meta
            .get("ri")
            .and_then(Value::as_u64)
            .ok_or_else(|| replay_err("metas row lacks ri".to_string()))?
            as usize;
        let old_lp = meta
            .get("old_lp")
            .and_then(Value::as_f64)
            .ok_or_else(|| replay_err("metas row lacks old_lp".to_string()))?
            as f32;
        if !old_lp.is_finite() {
            return Err(replay_err("metas old_lp must be finite".to_string()));
        }
        let raw_a = meta
            .get("raw_a")
            .and_then(Value::as_f64)
            .ok_or_else(|| replay_err("metas row lacks raw_a".to_string()))?
            as f32;
        if !raw_a.is_finite() {
            return Err(replay_err("metas raw_a must be finite".to_string()));
        }
        let g4_value = meta
            .get("g4")
            .and_then(Value::as_array)
            .ok_or_else(|| replay_err("metas row lacks g4".to_string()))?;
        if g4_value.len() != 4 {
            return Err(replay_err("metas g4 must hold four seats".to_string()));
        }
        let mut g4 = [0.0f64; 4];
        for (seat, slot) in g4_value.iter().zip(g4.iter_mut()) {
            *slot = seat
                .as_f64()
                .ok_or_else(|| replay_err("metas g4 entries must be numbers".to_string()))?;
        }
        if g4.iter().any(|v| !v.is_finite()) {
            return Err(replay_err("metas g4 must be finite".to_string()));
        }
        frozen.push(FrozenMetaRow {
            gi,
            ri,
            old_lp,
            raw_a,
            g4,
        });
    }
    // Shared insertion: exact-once audit plus precomputed batch geometry
    // plus handle allocation (one home for archive and live registration).
    let source = ReplaySource::Sealed {
        artifact_digest,
        order_digest: plan.order_digest,
    };
    insert_frozen_replay(source, frozen, &plan.batches)
}

/// Insert a validated replay: one exact-once position audit over the
/// planned batches, per-batch geometry through the single shared table,
/// then handle allocation and insertion. Either one valid handle exists
/// afterwards or an error returns with nothing inserted. Allocation uses
/// checked addition so exhaustion fails closed instead of reusing a live
/// handle.
pub(crate) fn insert_frozen_replay(
    source: ReplaySource,
    metas: Vec<FrozenMetaRow>,
    plans: &[BatchPlan],
) -> Result<ReplayHandle, TapeError> {
    let row_count = metas.len();
    let mut seen = vec![false; row_count];
    let mut covered = 0usize;
    for batch in plans {
        if batch.positions.is_empty() || batch.positions.len() > OPTIMIZER_CAP {
            return Err(replay_err(format!(
                "batch violates the {OPTIMIZER_CAP} cap"
            )));
        }
        for position in &batch.positions {
            if *position as usize >= row_count || seen[*position as usize] {
                return Err(replay_err("order drops or duplicates a row".to_string()));
            }
            seen[*position as usize] = true;
            covered += 1;
        }
    }
    if covered != row_count {
        return Err(replay_err("order drops or duplicates a row".to_string()));
    }
    let mut batches = Vec::with_capacity(plans.len());
    let mut offset: u64 = 0;
    for batch in plans {
        let count = batch.positions.len();
        let table = plane_table(batch.t, count)?;
        batches.push(FrozenBatch {
            t: batch.t,
            count,
            row_bytes: table.row_bytes,
            feature_bytes: table.slab_total,
            fact_bytes: count * FACTS48_STRIDE,
            positions: batch.positions.clone(),
            spans: table.spans,
            layout: table.layout,
            offset,
        });
        offset = offset.saturating_add(count as u64);
    }
    let replay = FrozenReplay {
        source,
        row_count,
        metas,
        batches,
    };
    let mut guard = registry()
        .lock()
        .map_err(|e| replay_err(format!("replay registry poisoned: {e}")))?;
    let handle = guard.next;
    guard.next = guard
        .next
        .checked_add(1)
        .ok_or_else(|| replay_err("replay handle space exhausted".to_string()))?;
    if handle == 0 {
        return Err(replay_err("replay handle space exhausted".to_string()));
    }
    guard.replays.insert(handle, replay);
    Ok(handle)
}

/// Run `f` against the frozen replay behind `handle` (unknown handle fails
/// closed). The hold may span one batch fill (entries are immutable, so it
/// blocks only register/drop); every tape access inside stays a short
/// per-row hold under the replay -> tape lock order.
pub fn with_replay<R>(
    handle: ReplayHandle,
    f: impl FnOnce(&FrozenReplay) -> Result<R, TapeError>,
) -> Result<R, TapeError> {
    let guard = registry()
        .lock()
        .map_err(|e| replay_err(format!("replay registry poisoned: {e}")))?;
    let replay = guard
        .replays
        .get(&handle)
        .ok_or_else(|| replay_err(format!("unknown replay handle {handle}")))?;
    f(replay)
}

/// Release one frozen replay handle (generation teardown). Unknown handles
/// fail closed so a double-drop or use-after-drop is loud, never silent.
pub fn drop_replay(handle: ReplayHandle) -> Result<(), TapeError> {
    match registry().lock() {
        Ok(mut guard) => guard
            .replays
            .remove(&handle)
            .map(|_| ())
            .ok_or_else(|| replay_err(format!("unknown replay handle {handle}"))),
        Err(e) => Err(replay_err(format!("replay registry poisoned: {e}"))),
    }
}
