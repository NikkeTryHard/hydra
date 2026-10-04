//! Learner-slab staging: one shared geometry/transpose/facts law for the
//! cold oracle and the hot frozen fill.
//!
//! [`plane_table`] is the single home for per-batch geometry (plane spans,
//! slab total, spec layout, legal/history slices), built over the shared
//! `observation::active_layout` table — never `count * S(T)`, never a second
//! byte law. The cold [`stage_batch`]/[`export_learner_slab`] oracle (owned
//! buffers, tooling only) and the hot [`fill_slab_buffers`] (caller-owned
//! pinned spans, production) transpose rows and pack the 48B facts through
//! the same two helpers, so cold/frozen parity holds by construction rather
//! than by mirrored code. The learner consumes stored `A`; nothing is
//! recomputed on admission.

use serde_json::{Map, Value};

use crate::rl::artifact::parse_metas;
use crate::rl::config::{RL_REPLAY_INDEX_ARTIFACT, RL_SCHEMA_VERSION, check_envelope_keys};
use crate::rl::trajectory::{
    ACTION_WIDTH, DIGEST_PREFIX, TapeError, TapeHandle, check_tape_stamp, hex_of, sha256_raw,
    with_tape,
};

use super::registry::{FrozenBatch, FrozenMetaRow, ReplayHandle, ReplaySource, with_replay};
use super::{FACTS48_STRIDE, OPTIMIZER_CAP};

/// Typed learner slab for one selected batch (owned contiguous buffers the
/// Torch boundary wraps with zero per-row work).
pub struct LearnerSlab {
    pub batch: u32,
    pub t: u32,
    pub count: usize,
    pub row_bytes: usize,
    /// Actual feature-slab bytes (plane-major, alignment-aware; may exceed
    /// `count * S(T)` once the live layout inserts bounded padding).
    pub feature_bytes: usize,
    /// Plane table `(name, dtype, shape, byte offset)` sourced from the
    /// shared `active_layout` (uniform excluded): the Torch side slices
    /// typed views with zero per-row work and caches them per batch shape.
    pub planes: Vec<(String, String, Vec<usize>, usize)>,
    pub artifact_digest: String,
    pub order_digest: String,
    pub features: Vec<u8>,
    pub old_lp_le: Vec<u8>,
    pub raw_a_le: Vec<u8>,
    pub g4_f64_le: Vec<u8>,
    pub selected_i64_le: Vec<u8>,
    pub legal_bytes: Vec<u8>,
    pub history_bytes: Vec<u8>,
    /// Plane-major facts blocks, B*48 bytes total: selected[B] @0,
    /// old-LP[B] @8B, raw-A[B] @12B, canonical G4[B,4] @16B.
    pub facts48: Vec<u8>,
    /// Canonical bytes of the registered per-batch replay-index envelope
    /// (`RL_REPLAY_INDEX_FIELDS`, digest attached): binds rlo/order,
    /// layout, bucket, order offset, count, feature checksum and facts
    /// digest for the learner's admission check.
    pub batch_envelope: Vec<u8>,
}

/// Per-batch plane geometry from the shared `active_layout` table (uniform
/// excluded): transpose spans plus the spec layout plus the legal/history
/// row slices the cold oracle serves. One home for the geometry the cold
/// export builds per fill, the frozen registration precomputes once, and
/// the slot allocator sizes from.
pub struct PlaneTable {
    /// Tight encoder bytes per row `S(T)` (bucket-gated).
    pub row_bytes: usize,
    /// Aligned slab bytes for `count` rows (last region end, gaps included).
    pub slab_total: usize,
    /// Transpose spans `(row_off, base, plane_len)` in plane-layout order:
    /// each plane's `count` rows are contiguous at `base`, one row's plane
    /// slice sits at `row_off` inside the tight encoder row.
    pub spans: Vec<(usize, usize, usize)>,
    /// Spec layout `(name, dtype, shape, offset)` for the descriptor.
    pub layout: Vec<(String, String, Vec<usize>, usize)>,
    /// `(row_off, plane_len)` slices for the sealed-mask convenience lanes.
    pub legal: (usize, usize),
    pub history: (usize, usize),
}

/// Build the plane table for `(T, count)`: bucket-gated `S(T)`, aligned
/// slab total, transpose spans, spec layout, and mask slices. Rejects a
/// zero count fail-closed (shape math would divide by it); capacity caps
/// stay with the callers so each keeps its own error law.
pub fn plane_table(t: u32, count: usize) -> Result<PlaneTable, TapeError> {
    // Bucket gate first (same misuse law as the encoder wrapper).
    let row_bytes = super::feature_row_bytes(t)?;
    if count == 0 {
        return Err(super::order_err(
            "plane table needs a nonzero row count".to_string(),
        ));
    }
    // Plane-major packing (the pool/observation layout): each plane's B rows
    // are contiguous, in `plane_layout` order with `uniform` excluded. The
    // encoder emits tight row-major S(T) rows; the transpose below moves each
    // plane slice into its region so the Torch side consumes exact
    // `active_layout` offsets with zero per-row work.
    // Reliance (pool-owned invariant): `encode_row_facts` writes planes in
    // `plane_layout` order; lengths are enforced per row by the callers, and
    // the plane table plus total are cross-checked here — any drift fails loud.
    let layout = crate::rl::observation::active_layout(t as usize, count, false);
    // Per plane: (row offset inside one encoded row, batch region base,
    // one-row plane length). Regions may carry bounded alignment padding
    // between them (the live `active_layout` update); rows stay contiguous
    // inside each region, and the tight row sum still equals S(T) (per-row
    // S exact is preserved).
    let mut spans: Vec<(String, usize, usize, usize)> = Vec::with_capacity(layout.len());
    let mut row_cursor = 0usize;
    let mut slab_total = 0usize;
    for (name, dtype, shape, base) in &layout {
        let width = match dtype.as_str() {
            "int64" | "float64" => 8,
            "int32" => 4,
            "bool" => 1,
            other => {
                return Err(super::order_err(format!("unknown plane dtype {other}")));
            }
        };
        let per_row: usize = shape.iter().product::<usize>() / count;
        let plane_len = per_row * width;
        if plane_len == 0 {
            return Err(super::order_err(format!("plane {name} collapses to zero")));
        }
        slab_total = slab_total.max(base + count * plane_len);
        spans.push((name.clone(), row_cursor, *base, plane_len));
        row_cursor += plane_len;
    }
    if row_cursor != row_bytes {
        return Err(super::order_err(format!(
            "plane table spans {row_cursor}B, not the encoder S(T)={row_bytes}B"
        )));
    }
    let legal = spans
        .iter()
        .find(|(name, _, _, _)| name == "legal_mask")
        .ok_or_else(|| super::order_err("plane table lacks legal_mask".to_string()))?;
    let history = spans
        .iter()
        .find(|(name, _, _, _)| name == "history_mask")
        .ok_or_else(|| super::order_err("plane table lacks history_mask".to_string()))?;
    Ok(PlaneTable {
        row_bytes,
        slab_total,
        spans: spans
            .iter()
            .map(|(_, row_off, base, plane_len)| (*row_off, *base, *plane_len))
            .collect(),
        layout,
        legal: (legal.1, legal.3),
        history: (history.1, history.3),
    })
}

/// Transpose one tight encoder row into its plane regions (shared cold/hot
/// law): slice each plane from the encoder's own bytes and scatter it into
/// the batch region. Spans are consistent by construction (built with the
/// slab they fill), so every index below is in range on honest tables.
fn transpose_row_into(
    features: &mut [u8],
    spans: &[(usize, usize, usize)],
    encoded: &[u8],
    r: usize,
) {
    for (row_off, base, plane_len) in spans.iter() {
        let src = &encoded[*row_off..*row_off + plane_len];
        let dst = &mut features[*base + r * plane_len..*base + (r + 1) * plane_len];
        dst.copy_from_slice(src);
    }
}

/// Pack one row's facts into the plane-major 48B blocks (shared cold/hot
/// law): selected[B] @0, old-LP[B] @8B, raw-A[B] @12B, canonical G4[B,4]
/// @16B (total B*48). The Torch side takes contiguous views — no strides,
/// no repack.
fn write_facts_row(
    facts: &mut [u8],
    count: usize,
    r: usize,
    selected_le: &[u8; 8],
    old_lp: f32,
    raw_a: f32,
    g4: &[f64; 4],
) {
    let base_selected = r * 8;
    let base_old_lp = count * 8 + r * 4;
    let base_raw_a = count * 12 + r * 4;
    let base_g4 = count * 16 + r * 32;
    facts[base_selected..base_selected + 8].copy_from_slice(selected_le);
    facts[base_old_lp..base_old_lp + 4].copy_from_slice(&old_lp.to_le_bytes());
    facts[base_raw_a..base_raw_a + 4].copy_from_slice(&raw_a.to_le_bytes());
    for (seat, v) in g4.iter().enumerate() {
        let at = base_g4 + seat * 8;
        facts[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
}

/// Export one optimizer batch as owned typed buffers: feature slab from the
/// registered pure encoder over borrowed typed facts (never the artifact
/// file, never a row re-parse), frozen facts from the metas companion.
/// Cold path for tests/tooling (owned buffers for every fact lane);
/// production uses [`fill_slab_buffers`] (caller-owned pinned buffers, no
/// expanded copies). Fails closed on stale handles, unknown batches, and
/// encoder drift.
pub fn export_learner_slab(
    handle: TapeHandle,
    index_bytes: &[u8],
    metas_bytes: &[u8],
    batch_idx: usize,
) -> Result<LearnerSlab, TapeError> {
    stage_batch(handle, index_bytes, metas_bytes, batch_idx, true)
}

/// Shared staging core: `cold` gates the separate convenience buffers
/// (the production fill serves features + facts48 alone).
pub fn stage_batch(
    handle: TapeHandle,
    index_bytes: &[u8],
    metas_bytes: &[u8],
    batch_idx: usize,
    cold: bool,
) -> Result<LearnerSlab, TapeError> {
    let plan = super::parse_index(index_bytes)?;
    let metas = parse_metas(metas_bytes)?;
    let metas_digest = metas
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| super::order_err("metas lacks artifact_digest".to_string()))?;
    if metas_digest != plan.artifact_digest {
        return Err(super::order_err(
            "index artifact digest mismatches the metas companion".to_string(),
        ));
    }
    check_tape_stamp(handle, &plan.artifact_digest)?;
    let batch = plan
        .batches
        .get(batch_idx)
        .ok_or_else(|| super::order_err(format!("unknown learner batch {batch_idx}")))?;
    let meta_rows = metas
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| super::order_err("metas lacks rows".to_string()))?;
    let count = batch.positions.len();
    if count == 0 {
        return Err(super::order_err("cannot export an empty learner batch".to_string()));
    }
    // Single geometry home (the frozen registration precomputes this same
    // table once, so cold and hot agree by construction).
    let table = plane_table(batch.t, count)?;
    let row_bytes = table.row_bytes;
    let mut features = vec![0u8; table.slab_total];
    // Cold-only convenience buffers (tests/tooling): the production fill
    // path leaves these empty and serves features + facts48 alone.
    let mut old_lp_le = Vec::with_capacity(cold as usize * count * 4);
    let mut raw_a_le = Vec::with_capacity(cold as usize * count * 4);
    let mut g4_f64_le = Vec::with_capacity(cold as usize * count * 32);
    let mut selected_i64_le = Vec::with_capacity(cold as usize * count * 8);
    let mut legal_bytes = Vec::with_capacity(cold as usize * count * ACTION_WIDTH);
    let mut history_bytes = Vec::with_capacity(cold as usize * count * batch.t as usize);
    let mut facts48 = vec![0u8; count * FACTS48_STRIDE];
    with_tape(handle, |tape| {
        for (r, position) in batch.positions.iter().enumerate() {
            let meta = meta_rows.get(*position as usize).ok_or_else(|| {
                super::order_err(format!("index position {position} outside metas rows"))
            })?;
            let game_idx = meta
                .get("gi")
                .and_then(Value::as_u64)
                .ok_or_else(|| super::order_err("metas row lacks gi".to_string()))? as usize;
            let row_idx = meta
                .get("ri")
                .and_then(Value::as_u64)
                .ok_or_else(|| super::order_err("metas row lacks ri".to_string()))? as usize;
            let game = tape.games.get(game_idx).ok_or_else(|| {
                super::order_err(format!("tape address game {game_idx} vanished at export"))
            })?;
            let row = game.rows.get(row_idx).ok_or_else(|| {
                super::order_err(format!("tape address row {row_idx} vanished at export"))
            })?;
            let prefix_len = row.event_prefix_len as usize;
            if prefix_len > game.arena.len() {
                return Err(super::order_err(format!(
                    "row {} prefix exceeds arena at export",
                    row.decision_id
                )));
            }
            let encoded = (tape.encode)(row, &game.arena, batch.t)?;
            if encoded.len() != row_bytes {
                return Err(super::order_err(format!(
                    "encoder drift: row {} emitted {} bytes (want S(T)={row_bytes})",
                    row.decision_id,
                    encoded.len()
                )));
            }
            // Shared transpose law (slice the legal/history masks from the
            // encoder's own bytes: consistent with the sealed mask by
            // construction, never re-derived).
            transpose_row_into(&mut features, &table.spans, &encoded, r);
            // Stored selected id (a GPU-return fact, never recomputed).
            let selected_le = (row.selected as i64).to_le_bytes();
            if cold {
                legal_bytes.extend_from_slice(&encoded[table.legal.0..table.legal.0 + table.legal.1]);
                history_bytes.extend_from_slice(
                    &encoded[table.history.0..table.history.0 + table.history.1],
                );
                selected_i64_le.extend_from_slice(&selected_le);
            }
            let old_lp = meta
                .get("old_lp")
                .and_then(Value::as_f64)
                .ok_or_else(|| super::order_err("metas row lacks old_lp".to_string()))?
                as f32;
            let raw_a = meta
                .get("raw_a")
                .and_then(Value::as_f64)
                .ok_or_else(|| super::order_err("metas row lacks raw_a".to_string()))?
                as f32;
            let mut g4w = [0.0f64; 4];
            let g4 = meta
                .get("g4")
                .and_then(Value::as_array)
                .ok_or_else(|| super::order_err("metas row lacks g4".to_string()))?;
            if g4.len() != 4 {
                return Err(super::order_err("metas g4 must hold four seats".to_string()));
            }
            for (seat, slot) in g4.iter().zip(g4w.iter_mut()) {
                *slot = seat
                    .as_f64()
                    .ok_or_else(|| super::order_err("metas g4 entries must be numbers".to_string()))?;
            }
            // Shared facts law (identical 48B blocks the Torch side views).
            write_facts_row(&mut facts48, count, r, &selected_le, old_lp, raw_a, &g4w);
            if cold {
                old_lp_le.extend_from_slice(&old_lp.to_le_bytes());
                raw_a_le.extend_from_slice(&raw_a.to_le_bytes());
                for v in g4w {
                    g4_f64_le.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        Ok(())
    })?;
    // Order offset: rows frozen before this batch in the same order.
    let mut offset: u64 = 0;
    for prior in plan.batches.iter().take(batch_idx) {
        offset = offset.saturating_add(prior.positions.len() as u64);
    }
    let batch_envelope = batch_envelope_bytes(
        &plan.artifact_digest,
        features.len(),
        batch.t,
        offset,
        count as u64,
        &features,
        &facts48,
    )?;
    Ok(LearnerSlab {
        batch: batch_idx as u32,
        t: batch.t,
        count,
        row_bytes,
        feature_bytes: features.len(),
        planes: table.layout,
        artifact_digest: plan.artifact_digest,
        order_digest: plan.order_digest,
        features,
        old_lp_le,
        raw_a_le,
        g4_f64_le,
        selected_i64_le,
        legal_bytes,
        history_bytes,
        facts48,
        batch_envelope,
    })
}

/// Shared row encode/transpose/fact-fill kernel: zeroes the used feature
/// prefix, then serves every batch position from the tape through the
/// registered pure encoder (never a Python callback) into caller memory.
/// One home for the archive `fill_slab_buffers` and live
/// `fill_live_slab_buffers`; callers own source/span gating.
pub(crate) fn fill_rows_into(
    features: &mut [u8],
    facts: &mut [u8],
    batch: &FrozenBatch,
    metas: &[FrozenMetaRow],
    tape_handle: TapeHandle,
) -> Result<(), TapeError> {
    // Deterministic padding: the caller span may hold stale bytes and the
    // layout carries bounded alignment gaps, so the used feature prefix is
    // zeroed once (every facts byte is overwritten per row below).
    features[..batch.feature_bytes].fill(0);
    let features = &mut features[..batch.feature_bytes];
    let facts = &mut facts[..batch.fact_bytes];
    for (r, position) in batch.positions.iter().enumerate() {
        let meta = metas
            .get(*position as usize)
            .ok_or_else(|| super::order_err(format!("index position {position} outside metas rows")))?;
        // Short tape hold per row: address lookup plus the pure-encoder
        // call only; the transpose below is caller-memory only.
        let (selected_le, encoded) = with_tape(tape_handle, |tape| {
            let game = tape.games.get(meta.gi).ok_or_else(|| {
                super::order_err(format!("tape address game {} vanished at export", meta.gi))
            })?;
            let row = game.rows.get(meta.ri).ok_or_else(|| {
                super::order_err(format!("tape address row {} vanished at export", meta.ri))
            })?;
            if row.event_prefix_len as usize > game.arena.len() {
                return Err(super::order_err(format!(
                    "row {} prefix exceeds arena at export",
                    row.decision_id
                )));
            }
            let encoded = (tape.encode)(row, &game.arena, batch.t)?;
            if encoded.len() != batch.row_bytes {
                return Err(super::order_err(format!(
                    "encoder drift: row {} emitted {} bytes (want S(T)={})",
                    row.decision_id,
                    encoded.len(),
                    batch.row_bytes
                )));
            }
            Ok(((row.selected as i64).to_le_bytes(), encoded))
        })?;
        transpose_row_into(features, &batch.spans, &encoded, r);
        write_facts_row(facts, batch.count, r, &selected_le, meta.old_lp, meta.raw_a, &meta.g4);
    }
    Ok(())
}

/// Fill one optimizer batch directly into caller-owned spans (production):
/// features plane-major plus interleaved 48B facts, no slab-sized
/// intermediate, no per-row callback beyond the registered pure encoder
/// (whose bounded per-row output is inherent to its signature). The order
/// index and metas companion arrive frozen via the replay handle (parsed
/// once at registration); this path takes no index bytes and no metas path.
/// Span gates, the frozen artifact identity (tape seal stamp), the counts,
/// and the checksum/envelope check run once per real batch, not per row.
/// Per-row tape holds stay short (lookup plus the pure-encoder call); the
/// transpose below touches caller memory alone.
pub fn fill_slab_buffers(
    tape_handle: TapeHandle,
    replay_handle: ReplayHandle,
    batch_idx: usize,
    features: &mut [u8],
    facts: &mut [u8],
) -> Result<SlabSpec, TapeError> {
    // Lock order is replay -> tape everywhere on this path (registration
    // takes the replay lock alone; tape paths take the tape lock alone), so
    // the replay hold below cannot deadlock. The hold spans the batch
    // because the frozen plan is borrowed per row; it blocks only
    // register/drop (once per generation), never tape progress or sealing.
    with_replay(replay_handle, |replay| {
        // Source gate: the archive fill serves sealed replays only and
        // verifies the tape seal stamp; live replays require the live fill.
        let (artifact_digest, order_digest) = match &replay.source {
            ReplaySource::Sealed {
                artifact_digest,
                order_digest,
            } => (artifact_digest.clone(), order_digest.clone()),
            ReplaySource::Live { .. } => {
                return Err(super::order_err(
                    "live replay requires fill_live_slab".to_string(),
                ))
            }
        };
        let batch = replay
            .batches
            .get(batch_idx)
            .ok_or_else(|| super::order_err(format!("unknown learner batch {batch_idx}")))?;
        // Span gate before any byte is touched (caller-sized spans; tails
        // fill the prefix, never dummy rows).
        if features.len() < batch.feature_bytes || facts.len() < batch.fact_bytes {
            return Err(super::order_err(format!(
                "slab spans too small: have ({},{}), need ({},{})",
                features.len(),
                facts.len(),
                batch.feature_bytes,
                batch.fact_bytes
            )));
        }
        // Frozen identity gate: the tape's seal stamp must equal the
        // replay's artifact digest, or the fill would serve a stale or
        // wrong-source generation.
        check_tape_stamp(tape_handle, &artifact_digest)?;
        let count = batch.count;
        fill_rows_into(features, facts, batch, &replay.metas, tape_handle)?;
        // Envelope and checksums compute once per real batch over the filled
        // prefixes (the same bytes the cold export hashes, so digests agree).
        let batch_envelope = batch_envelope_bytes(
            &artifact_digest,
            batch.feature_bytes,
            batch.t,
            batch.offset,
            count as u64,
            &features[..batch.feature_bytes],
            &facts[..batch.fact_bytes],
        )?;
        Ok(SlabSpec {
            batch: batch_idx as u32,
            t: batch.t,
            count,
            row_bytes: batch.row_bytes,
            feature_bytes: batch.feature_bytes,
            fact_bytes: batch.fact_bytes,
            planes: batch.layout.clone(),
            artifact_digest,
            order_digest,
            batch_envelope,
        })
    })
}

/// Live slab descriptor schema (typed fill, no checksums or envelopes).
pub const LIVE_SLAB_SCHEMA: &str = "hydra2.rl_live_slab/v1";

/// Fill one optimizer batch from a live replay directly into caller-owned
/// spans (production): features plane-major plus interleaved 48B facts
/// through the shared row kernel, no slab-sized intermediate, no per-row
/// callback. The replay must carry the exact drained tape handle in its
/// live source; sealed replays require the archive fill. Unknown, dropped,
/// or wrong-source handles and geometry overflows fail before mutation.
/// Returns only the live slab descriptor
/// `{schema, batch, t, count, feature_bytes, fact_bytes, input_layout}`:
/// no per-batch checksums, archive envelope, or full JSON-row hashing.
pub fn fill_live_slab_buffers(
    tape_handle: TapeHandle,
    replay_handle: ReplayHandle,
    batch_idx: usize,
    features: &mut [u8],
    facts: &mut [u8],
) -> Result<Vec<u8>, TapeError> {
    // Lock order is replay -> tape everywhere on this path (registration
    // takes the replay lock alone; tape paths take the tape lock alone), so
    // the replay hold below cannot deadlock.
    with_replay(replay_handle, |replay| {
        match &replay.source {
            ReplaySource::Live { tape_handle: bound, .. } if *bound == tape_handle => {}
            ReplaySource::Live { .. } => {
                return Err(super::order_err(
                    "live fill tape handle mismatches registration".to_string(),
                ))
            }
            ReplaySource::Sealed { .. } => {
                return Err(super::order_err("sealed replay requires fill_slab".to_string()))
            }
        }
        let batch = replay
            .batches
            .get(batch_idx)
            .ok_or_else(|| super::order_err(format!("unknown learner batch {batch_idx}")))?;
        // Span gate before any byte is touched (caller-sized spans; tails
        // fill the prefix, never dummy rows).
        if features.len() < batch.feature_bytes || facts.len() < batch.fact_bytes {
            return Err(super::order_err(format!(
                "slab spans too small: have ({},{}), need ({},{})",
                features.len(),
                facts.len(),
                batch.feature_bytes,
                batch.fact_bytes
            )));
        }
        // Existence probe before mutation: a dropped tape fails here, not
        // after the feature prefix was already zeroed.
        crate::rl::trajectory::with_tape(tape_handle, |_| Ok(()))?;
        fill_rows_into(features, facts, batch, &replay.metas, tape_handle)?;
        let layout: Vec<Value> = batch
            .layout
            .iter()
            .map(|(name, dtype, shape, off)| {
                serde_json::json!({
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "offset": off,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "schema": LIVE_SLAB_SCHEMA,
            "batch": batch_idx,
            "t": batch.t,
            "count": batch.count,
            "feature_bytes": batch.feature_bytes,
            "fact_bytes": batch.fact_bytes,
            "input_layout": layout,
        });
        hydra_feed::canon::canonical_bytes_value(&doc, "rl:live-slab")
            .map_err(|e| super::order_err(format!("live slab JCS emit failed: {e:?}")))
    })
}

/// Registered per-batch replay-index envelope (`RL_REPLAY_INDEX_FIELDS`):
/// binds the rollout digest, feature layout, bucket, order offset, count,
/// feature checksum and facts digest under one digest the learner admits.
/// `feature_layout` names the plane-major active convention plus the actual
/// slab bytes; it makes no dtype claim beyond the encoder's frozen planes.
/// The order digest rides the slab spec beside this envelope (the registered
/// 9 fields carry the rollout binding; order scope is the index's).
/// `facts_digest` covers the interleaved 48B facts (selected/LP/A/G4).
fn batch_envelope_bytes(
    rlo_digest: &str,
    feature_bytes: usize,
    t: u32,
    offset: u64,
    count: u64,
    features: &[u8],
    facts48: &[u8],
) -> Result<Vec<u8>, TapeError> {
    let mut doc = Map::with_capacity(9);
    doc.insert(
        "schema_version".to_string(),
        Value::String(RL_SCHEMA_VERSION.to_string()),
    );
    doc.insert("rlo_digest".to_string(), Value::String(rlo_digest.to_string()));
    doc.insert(
        "feature_layout".to_string(),
        Value::String(format!("plane_major_active:{feature_bytes}B")),
    );
    doc.insert("history_bucket_T".to_string(), Value::from(u64::from(t)));
    doc.insert("offset".to_string(), Value::from(offset));
    doc.insert("count".to_string(), Value::from(count));
    doc.insert(
        "checksum".to_string(),
        Value::String(format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(features)))),
    );
    doc.insert(
        "facts_digest".to_string(),
        Value::String(format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(facts48)))),
    );
    // Closed-loop conformance: the emitted keys must satisfy the registered
    // envelope (without `digest`, which is computed next).
    let keys: Vec<String> = doc.keys().cloned().collect();
    check_envelope_keys(RL_REPLAY_INDEX_ARTIFACT, &keys, &[]).map_err(super::order_err)?;
    let bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(doc), "rl:batch-envelope")
        .map_err(|e| super::order_err(format!("batch envelope JCS emit failed: {e:?}")))?;
    let digest = format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(&bytes)));
    let mut full: Map<String, Value> = serde_json::from_slice(&bytes)
        .map_err(|e| super::order_err(format!("batch envelope reparse failed: {e}")))?;
    full.insert("digest".to_string(), Value::String(digest));
    hydra_feed::canon::canonical_bytes_value(&Value::Object(full), "rl:batch-envelope-full")
        .map_err(|e| super::order_err(format!("batch envelope full JCS emit failed: {e:?}")))
}

/// Production slab descriptor: counts, layout, sizes, and digests — never
/// row positions or row bytes.
pub struct SlabSpec {
    pub batch: u32,
    pub t: u32,
    pub count: usize,
    pub row_bytes: usize,
    pub feature_bytes: usize,
    pub fact_bytes: usize,
    pub planes: Vec<(String, String, Vec<usize>, usize)>,
    pub artifact_digest: String,
    pub order_digest: String,
    pub batch_envelope: Vec<u8>,
}

impl SlabSpec {
    /// Build the descriptor from a staged slab (cold and fill paths share
    /// this shape: counts, `input_layout`, actual sizes, digests, envelope).
    pub fn from_slab(slab: &LearnerSlab) -> Self {
        Self {
            batch: slab.batch,
            t: slab.t,
            count: slab.count,
            row_bytes: slab.row_bytes,
            feature_bytes: slab.feature_bytes,
            fact_bytes: slab.facts48.len(),
            planes: slab.planes.clone(),
            artifact_digest: slab.artifact_digest.clone(),
            order_digest: slab.order_digest.clone(),
            batch_envelope: slab.batch_envelope.clone(),
        }
    }
}

/// Canonical spec document shared by the cold and fill paths.
fn slab_spec_doc(spec: &SlabSpec) -> Value {
    let planes: Vec<Value> = spec
        .planes
        .iter()
        .map(|(name, dtype, shape, off)| {
            serde_json::json!({
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset": off,
            })
        })
        .collect();
    let envelope: Value =
        serde_json::from_slice(&spec.batch_envelope).map_or(Value::Null, |v| v);
    serde_json::json!({
        "batch": spec.batch,
        "t": spec.t,
        "count": spec.count,
        "row_bytes": spec.row_bytes,
        "feature_bytes": spec.feature_bytes,
        "fact_bytes": spec.fact_bytes,
        "input_layout": planes,
        "artifact_digest": spec.artifact_digest,
        "order_digest": spec.order_digest,
        "batch_envelope": envelope,
    })
}

/// Canonical spec bytes for one staged slab (cold and fill FFI share it).
pub fn slab_spec_bytes(slab: &LearnerSlab) -> Result<Vec<u8>, TapeError> {
    slab_spec_bytes_of(&SlabSpec::from_slab(slab))
}

/// Canonical spec bytes for one filled slab descriptor (hot fill FFI).
pub fn slab_spec_bytes_of(spec: &SlabSpec) -> Result<Vec<u8>, TapeError> {
    hydra_feed::canon::canonical_bytes_value(&slab_spec_doc(spec), "rl:slab-spec")
        .map_err(|e| super::order_err(format!("slab spec JCS emit failed: {e:?}")))
}

/// Slot geometry for one `(T, capacity)` learner lane: exact
/// `(feature_bytes, fact_bytes)` the fill spans validate against. Feature
/// bytes come from the shared `active_layout` table max-end (uniform
/// excluded; alignment-aware, never `count * S(T)`); facts are always
/// `capacity * 48`. Single home for the geometry both the FFI span check
/// and the Python slot allocator use.
pub fn slab_geometry(t: u32, capacity: usize) -> Result<(usize, usize), TapeError> {
    super::feature_row_bytes(t)?;
    if capacity == 0 || capacity > OPTIMIZER_CAP {
        return Err(super::order_err(format!(
            "slab capacity {capacity} outside 1..={OPTIMIZER_CAP}"
        )));
    }
    let table = plane_table(t, capacity)?;
    Ok((table.slab_total, capacity * FACTS48_STRIDE))
}
