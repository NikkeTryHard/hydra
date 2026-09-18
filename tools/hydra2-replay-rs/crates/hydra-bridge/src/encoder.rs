//! encoder: thin per-row staging boundary over torch-owned CPU bytes.
//!
//! DAG: this module depends on pyo3 ONLY (no feed/shard/search edge; no new
//! dependency — pure extraction + LE byte fill + dict return). FORBIDS:
//! parse/hash/pool/selection/GPU-math logic — arg-check + detach + struct
//! return only. All torch math, pin/H2D, and model forward stay Python-side
//! (`models/encoder.py` owns `torch.frombuffer` + `_stage_pinned_batch`);
//! Rust only stages raw LE bytes the caller wraps zero-copy.
//!
//! Bytes path (precedent: `training/rust_batch.py::_slice_game_planes`):
//! Rust returns `{plane: bytearray}` (writable, buffer-protocol) + `bucket_t`
//! + `max_history_len` + `hashes`; Python wraps each via `torch.frombuffer`
//! (borrowed, never copied) and reshapes to `[B] / [B,34] / [B,5] / [B,4] /
//! [B,T] / [B,6792]`. `torch.frombuffer` keeps the exporter alive (verified:
//! tensor survives `del` of the source bytearray), so the dict may drop after
//! wrapping. `bytes` (immutable) would warn non-writable — hence bytearray,
//! mirroring `replay::staged_to_py` (`PyByteArray` per plane).
//!
//! Padding contract (byte-identical to the numpy oracle it replaces):
//! caller-visible pre-fill is `0` (int kinds), `False` (bool masks), `-1`
//! (dora `(B,5)` + `own_drawn` int32 sentinel, `0xFF` per byte); Rust fills
//! valid-prefix bytes and never re-inits per row. `-1` int32 lanes are
//! `0xFF` per byte. Unwritten tails stay valid without per-row init, exactly
//! like the deleted `np.empty + .fill(0)/.fill(False)/.fill(-1)` oracle.
//!
//! Concurrency pattern (matches `ring.rs` + `stream.rs`): extract attached
//! (only owned `i64`/`bool`/`String`/`Vec` cross — never a borrow of Python
//! memory), fill detached (`py.detach` around the whole bulk fill, never
//! per-row attach), dict build attached. Frozen surface only (free function;
//! no pyclass, no shared state).
//!
//! Geometry under test (mirrors `schema.py` frozen consts — single source is
//! Python; Rust FAILS CLOSED on drift, never redefines):
//! - `HISTORY_BUCKET_LENGTHS == (32, 64, 128, 256)` (frozen ceil when the
//!   caller passes `None` or the frozen list; custom single-bucket probes
//!   `(32,)`/`(64,)`/`(128,)` compute their own ceil and skip the frozen
//!   agreement check, exactly like the deleted Python branch).
//! - dora sentinel width `(5,)` exact, legal `[B, 6792]` exact, over-cap
//!   history (> cap) fails closed — rows are NEVER truncated.
//! - `tile // 4` counts: concealed (no clamp) + visible (discards + public
//!   meld tiles, dora excluded, `min(c, 4)` per type), mirroring the deleted
//!   `_concealed_counts` / `_visible_discards_counts` exactly.
//! - history kinds: `EVENT_KINDS` order 0..20, unknown -> 0 (mirrors
//!   `_event_to_id.get(kind, 0)`); phase/furiten/riichi/wind are closed maps
//!   (validated upstream by `ActorObservation`, unreachable here — fail
//!   closed on drift).
//!
//! Single-cdylib tree: registers as the `encoder` submodule
//! (`hydra2_replay_rs.encoder`, `hydra_bridge._native.encoder` after the
//! Phase-6 cutover) via `register`, mirroring `ring::register`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyModule};

/// Canonical history buckets, mirrors `schema.HISTORY_BUCKET_LENGTHS`.
pub const HISTORY_BUCKETS: [usize; 4] = [32, 64, 128, 256];
/// Frozen dora-indicator width.
pub const DORA_WIDTH: usize = 5;
/// Frozen baseline action width.
pub const BASELINE_ACTIONS: usize = 6792;

/// Ceil `actual` to the next bucket in `buckets` (mirrors deleted
/// `encoder._bucket_length`; over-cap returns the last bucket — callers fail
/// closed before trusting it, never truncate).
fn bucket_ceil_impl(actual: usize, buckets: &[usize]) -> usize {
    for bucket in buckets {
        if actual <= *bucket {
            return *bucket;
        }
    }
    match buckets.last() {
        Some(last) => *last,
        None => HISTORY_BUCKETS[HISTORY_BUCKETS.len() - 1],
    }
}

/// Frozen ceil (hot path).
fn bucket_frozen(actual: usize) -> usize {
    bucket_ceil_impl(actual, &HISTORY_BUCKETS)
}

/// `EVENT_KINDS` order 0..20; unknown -> 0 (mirrors `.get(kind, 0)`).
fn event_kind_id(kind: &str) -> i64 {
    match kind {
        "game_start" => 0,
        "round_start" => 1,
        "turn_advance" => 2,
        "draw_tile" => 3,
        "discard" => 4,
        "riichi_declared" => 5,
        "riichi_accepted" => 6,
        "call_window" => 7,
        "call_resolved" => 8,
        "chi" => 9,
        "pon" => 10,
        "daiminkan" => 11,
        "ankan" => 12,
        "kakan" => 13,
        "dora_revealed" => 14,
        "ron" => 15,
        "tsumo" => 16,
        "draw_end" => 17,
        "abortive_draw" => 18,
        "round_end" => 19,
        "game_end" => 20,
        _ => 0,
    }
}

/// `PHASES` order 0..5.
fn phase_id(phase: &str) -> Result<i64, String> {
    match phase {
        "round_start" => Ok(0),
        "draw_decision" => Ok(1),
        "discard_response" => Ok(2),
        "kan_response" => Ok(3),
        "round_end" => Ok(4),
        "game_end" => Ok(5),
        _ => Err(format!("encoder stage: unknown phase {phase:?}")),
    }
}

fn furiten_id(state: &str) -> Result<i64, String> {
    match state {
        "none" => Ok(0),
        "temporary" => Ok(1),
        "riichi" => Ok(2),
        "discard" => Ok(3),
        _ => Err(format!("encoder stage: unknown furiten {state:?}")),
    }
}

fn riichi_id(state: &str) -> Result<i64, String> {
    match state {
        "none" => Ok(0),
        "declared" => Ok(1),
        "accepted" => Ok(2),
        _ => Err(format!("encoder stage: unknown riichi state {state:?}")),
    }
}

fn wind_id(tile: i64) -> Result<i64, String> {
    match tile {
        27 => Ok(0),
        28 => Ok(1),
        29 => Ok(2),
        30 => Ok(3),
        _ => Err(format!("encoder stage: wind tile {tile} not in 27..30")),
    }
}

// ---------------------------------------------------------------------------
// Attached extraction helpers (owned values only cross detach).
// ---------------------------------------------------------------------------

fn getattr_i64(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<i64, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<i64>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be int"))
        })
}

fn getattr_bool(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<bool, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<bool>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be bool"))
        })
}

fn getattr_string(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<String, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<String>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be str"))
        })
}

fn getattr_vec_i64(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<Vec<i64>, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<Vec<i64>>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be sequence[int]"))
        })
}

fn getattr_vec_bool(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<Vec<bool>, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<Vec<bool>>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be sequence[bool]"))
        })
}

fn getattr_vec_string(obs: &Bound<'_, PyAny>, field: &str, row: usize) -> Result<Vec<String>, PyErr> {
    obs.getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} unreadable: {e}"))
        })?
        .extract::<Vec<String>>()
        .map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .{field} must be sequence[str]"))
        })
}

/// Owned per-row staging input (no Python borrows cross detach).
struct StagedRow {
    actor: i64,
    dealer: i64,
    turn_actor: i64,
    can_riichi: bool,
    can_tsumo: bool,
    furiten: i64,
    hand_number: i32,
    round_index: i32,
    round_wind: i64,
    seat_winds: [i64; 4],
    honba: i32,
    riichi_sticks: i32,
    scores: [i32; 4],
    phase: i64,
    live_wall: i32,
    kan_count: i32,
    ippatsu: [bool; 4],
    riichi: [i64; 4],
    concealed: Vec<i64>,
    own_drawn: i32,
    dora: [i32; 5],
    discards: Vec<i64>,
    meld_tiles: Vec<i64>,
    history: Vec<i64>,
    legal: Vec<bool>,
    hash: String,
}

/// Extract one observation (attached; GIL held). Fails closed with
/// `ValueError` (Python maps text 1:1 to `ContractError`).
fn extract_row(obs: &Bound<'_, PyAny>, row: usize) -> Result<StagedRow, PyErr> {
    let actor = getattr_i64(obs, "actor", row)?;
    let dealer = getattr_i64(obs, "dealer", row)?;
    let turn_actor = getattr_i64(obs, "turn_actor", row)?;
    let can_riichi = getattr_bool(obs, "actor_can_riichi", row)?;
    let can_tsumo = getattr_bool(obs, "actor_can_tsumo", row)?;
    let furiten_raw = getattr_string(obs, "actor_furiten", row)?;
    let furiten = furiten_id(furiten_raw.as_str())
        .map_err(PyValueError::new_err)?;
    let hand_number_raw = getattr_i64(obs, "hand_number", row)?;
    let round_index_raw = getattr_i64(obs, "round_index", row)?;
    let round_wind_raw = getattr_i64(obs, "round_wind", row)?;
    let round_wind = wind_id(round_wind_raw).map_err(PyValueError::new_err)?;
    let seat_raw = getattr_vec_i64(obs, "seat_winds", row)?;
    if seat_raw.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .seat_winds must hold 4 entries"
        )));
    }
    let mut seat_winds = [0i64; 4];
    for (i, tile) in seat_raw.iter().enumerate() {
        match seat_winds.get_mut(i) {
            Some(slot) => {
                *slot = wind_id(*tile).map_err(PyValueError::new_err)?;
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: .seat_winds index {i} out of range"
                )));
            }
        }
    }
    let honba_raw = getattr_i64(obs, "honba", row)?;
    let riichi_sticks_raw = getattr_i64(obs, "riichi_sticks", row)?;
    let scores_raw = getattr_vec_i64(obs, "scores", row)?;
    if scores_raw.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .scores must hold 4 entries"
        )));
    }
    let mut scores = [0i32; 4];
    for (i, value) in scores_raw.iter().enumerate() {
        match scores.get_mut(i) {
            Some(slot) => {
                *slot = *value as i32;
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: .scores index {i} out of range"
                )));
            }
        }
    }
    let phase_raw = getattr_string(obs, "phase", row)?;
    let phase = phase_id(phase_raw.as_str()).map_err(PyValueError::new_err)?;
    let live_wall_raw = getattr_i64(obs, "live_wall_tiles_remaining", row)?;
    let kan_raw = getattr_i64(obs, "kan_count", row)?;
    let ippatsu_raw = getattr_vec_bool(obs, "ippatsu_active", row)?;
    if ippatsu_raw.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .ippatsu_active must hold 4 entries"
        )));
    }
    let mut ippatsu = [false; 4];
    for (i, flag) in ippatsu_raw.iter().enumerate() {
        match ippatsu.get_mut(i) {
            Some(slot) => {
                *slot = *flag;
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: .ippatsu_active index {i} out of range"
                )));
            }
        }
    }
    let riichi_raw = getattr_vec_string(obs, "riichi_states", row)?;
    if riichi_raw.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .riichi_states must hold 4 entries"
        )));
    }
    let mut riichi = [0i64; 4];
    for (i, state) in riichi_raw.iter().enumerate() {
        match riichi.get_mut(i) {
            Some(slot) => {
                *slot = riichi_id(state.as_str()).map_err(PyValueError::new_err)?;
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: .riichi_states index {i} out of range"
                )));
            }
        }
    }
    let concealed = getattr_vec_i64(obs, "concealed_hand", row)?;
    for tile in concealed.iter() {
        if *tile < 0 || *tile >= 136 {
            return Err(PyValueError::new_err(format!(
                "encoder stage row {row}: concealed tile {tile} out of range 0..135"
            )));
        }
    }
    let drawn_value = obs
        .getattr("own_drawn_tile")
        .map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: .own_drawn_tile unreadable: {e}"))
        })?;
    let own_drawn: i32 = if drawn_value.is_none() {
        -1
    } else {
        let tile: i64 = drawn_value.extract().map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: .own_drawn_tile must be int | None"))
        })?;
        if tile < 0 || tile >= 136 {
            return Err(PyValueError::new_err(format!(
                "encoder stage row {row}: drawn tile {tile} out of range 0..135"
            )));
        }
        tile as i32
    };
    let dora_raw = getattr_vec_i64(obs, "dora_indicators", row)?;
    if dora_raw.len() != DORA_WIDTH {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .dora_indicators must hold {DORA_WIDTH} entries"
        )));
    }
    let mut dora = [-1i32; 5];
    for (i, value) in dora_raw.iter().enumerate() {
        match dora.get_mut(i) {
            Some(slot) => {
                if *value != -1 && (*value < 0 || *value >= 136) {
                    return Err(PyValueError::new_err(format!(
                        "encoder stage row {row}: dora[{i}] tile {value} out of range"
                    )));
                }
                *slot = *value as i32;
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: .dora_indicators index {i} out of range"
                )));
            }
        }
    }
    // Visible discards: 4 rivers of tile ids (flat for counts).
    let rivers_value = obs.getattr("visible_discards").map_err(|e| {
        PyValueError::new_err(format!("encoder stage row {row}: .visible_discards unreadable: {e}"))
    })?;
    let rivers: Vec<Vec<i64>> = rivers_value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "encoder stage row {row}: .visible_discards must be 4 sequences of tile ids"
        ))
    })?;
    if rivers.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .visible_discards must hold exactly four seat rivers"
        )));
    }
    let mut discards: Vec<i64> = Vec::new();
    for river in rivers.iter() {
        for tile in river.iter() {
            if *tile < 0 || *tile >= 136 {
                return Err(PyValueError::new_err(format!(
                    "encoder stage row {row}: discard tile {tile} out of range 0..135"
                )));
            }
            discards.push(*tile);
        }
    }
    // Visible melds: 4 seat rows of VisibleMeld objects; only `.tiles` counts
    // (dora excluded upstream, clamped below alongside discards).
    let melds_value = obs.getattr("visible_melds").map_err(|e| {
        PyValueError::new_err(format!("encoder stage row {row}: .visible_melds unreadable: {e}"))
    })?;
    let meld_rows: Vec<Bound<'_, PyAny>> = melds_value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "encoder stage row {row}: .visible_melds must hold exactly four seat meld rows"
        ))
    })?;
    if meld_rows.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "encoder stage row {row}: .visible_melds must hold exactly four seat meld rows"
        )));
    }
    let mut meld_tiles: Vec<i64> = Vec::new();
    for (seat, meld_row) in meld_rows.iter().enumerate() {
        let melds: Vec<Bound<'_, PyAny>> = meld_row.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "encoder stage row {row}: .visible_melds[{seat}] must be a sequence of VisibleMeld"
            ))
        })?;
        for meld in melds.iter() {
            let tiles_value = meld.getattr("tiles").map_err(|e| {
                PyValueError::new_err(format!(
                    "encoder stage row {row}: meld.tiles unreadable: {e}"
                ))
            })?;
            let tiles: Vec<i64> = tiles_value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "encoder stage row {row}: meld.tiles must be sequence[int]"
                ))
            })?;
            for tile in tiles.iter() {
                if *tile < 0 || *tile >= 136 {
                    return Err(PyValueError::new_err(format!(
                        "encoder stage row {row}: meld tile {tile} out of range 0..135"
                    )));
                }
                meld_tiles.push(*tile);
            }
        }
    }
    // History: EventEnvelope objects; only `.kind` maps (unknown -> 0).
    let history_value = obs.getattr("visible_history").map_err(|e| {
        PyValueError::new_err(format!("encoder stage row {row}: .visible_history unreadable: {e}"))
    })?;
    let events: Vec<Bound<'_, PyAny>> = history_value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "encoder stage row {row}: .visible_history must be a sequence of EventEnvelope"
        ))
    })?;
    let mut history: Vec<i64> = Vec::with_capacity(events.len());
    for event in events.iter() {
        let kind_value = event.getattr("kind").map_err(|e| {
            PyValueError::new_err(format!("encoder stage row {row}: event.kind unreadable: {e}"))
        })?;
        let kind: String = kind_value.extract().map_err(|_| {
            PyValueError::new_err(format!("encoder stage row {row}: event.kind must be str"))
        })?;
        history.push(event_kind_id(kind.as_str()));
    }
    // Legal mask: frozen width + at least one True (oracle-identical text).
    let legal = getattr_vec_bool(obs, "legal_mask", row)?;
    if legal.len() != BASELINE_ACTIONS {
        return Err(PyValueError::new_err(format!(
            "legal_mask length {} != baseline {BASELINE_ACTIONS}",
            legal.len()
        )));
    }
    if !legal.iter().any(|flag| *flag) {
        return Err(PyValueError::new_err(
            "legal_mask must contain at least one True at a decision".to_string(),
        ));
    }
    let hash_value = obs.getattr("observation_hash").map_err(|e| {
        PyValueError::new_err(format!(
            "encoder stage row {row}: .observation_hash unreadable: {e}"
        ))
    })?;
    if hash_value.is_none() {
        return Err(PyValueError::new_err(
            "observation_hash must be bound".to_string(),
        ));
    }
    let hash: String = hash_value.extract().map_err(|_| {
        PyValueError::new_err(format!("encoder stage row {row}: .observation_hash must be str"))
    })?;
    Ok(StagedRow {
        actor,
        dealer,
        turn_actor,
        can_riichi,
        can_tsumo,
        furiten,
        hand_number: hand_number_raw as i32,
        round_index: round_index_raw as i32,
        round_wind,
        seat_winds,
        honba: honba_raw as i32,
        riichi_sticks: riichi_sticks_raw as i32,
        scores,
        phase,
        live_wall: live_wall_raw as i32,
        kan_count: kan_raw as i32,
        ippatsu,
        riichi,
        concealed,
        own_drawn,
        dora,
        discards,
        meld_tiles,
        history,
        legal,
        hash,
    })
}

// ---------------------------------------------------------------------------
// Detached fill (no Python interaction; caller keeps buffers alive).
// ---------------------------------------------------------------------------

fn write_i64_le(buf: &mut [u8], offset: usize, value: i64) -> Result<(), String> {
    let end = offset.saturating_add(8);
    if end > buf.len() {
        return Err(format!("encoder fill: i64 write out of range at {offset}"));
    }
    match buf.get_mut(offset..end) {
        Some(slot) => {
            slot.copy_from_slice(&value.to_le_bytes());
            Ok(())
        }
        None => Err(format!("encoder fill: i64 slice missing at {offset}")),
    }
}

fn write_i32_le(buf: &mut [u8], offset: usize, value: i32) -> Result<(), String> {
    let end = offset.saturating_add(4);
    if end > buf.len() {
        return Err(format!("encoder fill: i32 write out of range at {offset}"));
    }
    match buf.get_mut(offset..end) {
        Some(slot) => {
            slot.copy_from_slice(&value.to_le_bytes());
            Ok(())
        }
        None => Err(format!("encoder fill: i32 slice missing at {offset}")),
    }
}

fn write_bool_byte(buf: &mut [u8], offset: usize, value: bool) -> Result<(), String> {
    match buf.get_mut(offset) {
        Some(slot) => {
            *slot = u8::from(value);
            Ok(())
        }
        None => Err(format!("encoder fill: bool write out of range at {offset}")),
    }
}

/// Fill all 26 plane buffers from staged rows (pure over inputs).
#[allow(clippy::too_many_arguments)]
fn fill_buffers(
    rows: &[StagedRow],
    bucket_t: usize,
    actor: &mut [u8],
    can_riichi: &mut [u8],
    can_tsumo: &mut [u8],
    furiten: &mut [u8],
    seats: &mut [u8],
    concealed: &mut [u8],
    dealer: &mut [u8],
    dora: &mut [u8],
    hand_number: &mut [u8],
    history_kind: &mut [u8],
    history_mask: &mut [u8],
    honba: &mut [u8],
    ippatsu: &mut [u8],
    kan_count: &mut [u8],
    legal: &mut [u8],
    live_wall: &mut [u8],
    own_drawn: &mut [u8],
    phase: &mut [u8],
    riichi_states: &mut [u8],
    riichi_sticks: &mut [u8],
    round_index: &mut [u8],
    round_wind: &mut [u8],
    scores: &mut [u8],
    seat_winds: &mut [u8],
    turn_actor: &mut [u8],
    visible: &mut [u8],
) -> Result<(), String> {
    for (idx, row) in rows.iter().enumerate() {
        write_i64_le(actor, idx.saturating_mul(8), row.actor)?;
        write_bool_byte(can_riichi, idx, row.can_riichi)?;
        write_bool_byte(can_tsumo, idx, row.can_tsumo)?;
        write_i64_le(furiten, idx.saturating_mul(8), row.furiten)?;
        write_i64_le(seats, idx.saturating_mul(8), row.actor)?;
        write_i64_le(dealer, idx.saturating_mul(8), row.dealer)?;
        write_i32_le(hand_number, idx.saturating_mul(4), row.hand_number)?;
        write_i32_le(honba, idx.saturating_mul(4), row.honba)?;
        write_i32_le(kan_count, idx.saturating_mul(4), row.kan_count)?;
        write_i32_le(live_wall, idx.saturating_mul(4), row.live_wall)?;
        write_i32_le(own_drawn, idx.saturating_mul(4), row.own_drawn)?;
        write_i64_le(phase, idx.saturating_mul(8), row.phase)?;
        write_i32_le(riichi_sticks, idx.saturating_mul(4), row.riichi_sticks)?;
        write_i32_le(round_index, idx.saturating_mul(4), row.round_index)?;
        write_i64_le(round_wind, idx.saturating_mul(8), row.round_wind)?;
        write_i64_le(turn_actor, idx.saturating_mul(8), row.turn_actor)?;
        for j in 0..4 {
            let base = idx
                .saturating_mul(4)
                .saturating_add(j)
                .saturating_mul(8);
            let value = match row.seat_winds.get(j) {
                Some(v) => *v,
                None => return Err(format!("encoder fill row {idx}: seat_winds[{j}] missing")),
            };
            write_i64_le(seat_winds, base, value)?;
            let rvalue = match row.riichi.get(j) {
                Some(v) => *v,
                None => return Err(format!("encoder fill row {idx}: riichi[{j}] missing")),
            };
            write_i64_le(
                riichi_states,
                idx.saturating_mul(4).saturating_add(j).saturating_mul(8),
                rvalue,
            )?;
            let svalue = match row.scores.get(j) {
                Some(v) => *v,
                None => return Err(format!("encoder fill row {idx}: scores[{j}] missing")),
            };
            write_i32_le(
                scores,
                idx.saturating_mul(4).saturating_add(j).saturating_mul(4),
                svalue,
            )?;
            let ivalue = match row.ippatsu.get(j) {
                Some(v) => *v,
                None => return Err(format!("encoder fill row {idx}: ippatsu[{j}] missing")),
            };
            write_bool_byte(ippatsu, idx.saturating_mul(4).saturating_add(j), ivalue)?;
        }
        for j in 0..DORA_WIDTH {
            let dvalue = match row.dora.get(j) {
                Some(v) => *v,
                None => return Err(format!("encoder fill row {idx}: dora[{j}] missing")),
            };
            write_i32_le(dora, idx.saturating_mul(5).saturating_add(j).saturating_mul(4), dvalue)?;
        }
        // Concealed counts: tile // 4, no clamp (mirrors deleted oracle).
        let mut concealed_counts = [0i32; 34];
        for tile in row.concealed.iter() {
            let kind = (*tile as usize) / 4;
            match concealed_counts.get_mut(kind) {
                Some(slot) => {
                    *slot = slot.saturating_add(1);
                }
                None => {
                    return Err(format!("encoder fill row {idx}: concealed kind {kind} out of range"));
                }
            }
        }
        for (j, count) in concealed_counts.iter().enumerate() {
            write_i32_le(
                concealed,
                idx.saturating_mul(34).saturating_add(j).saturating_mul(4),
                *count,
            )?;
        }
        // Visible counts: discards + public meld tiles, dora excluded
        // upstream, clamp to 4 per type (mirrors deleted oracle).
        let mut visible_counts = [0i32; 34];
        for tile in row.discards.iter().chain(row.meld_tiles.iter()) {
            let kind = (*tile as usize) / 4;
            match visible_counts.get_mut(kind) {
                Some(slot) => {
                    *slot = slot.saturating_add(1);
                }
                None => {
                    return Err(format!("encoder fill row {idx}: visible kind {kind} out of range"));
                }
            }
        }
        for slot in visible_counts.iter_mut() {
            if *slot > 4 {
                *slot = 4;
            }
        }
        for (j, count) in visible_counts.iter().enumerate() {
            write_i32_le(
                visible,
                idx.saturating_mul(34).saturating_add(j).saturating_mul(4),
                *count,
            )?;
        }
        // History prefix; tails stay 0/False pre-fill.
        for (pos, kind) in row.history.iter().enumerate() {
            if pos >= bucket_t {
                return Err(format!(
                    "encoder fill row {idx}: history {pos} exceeds bucket {bucket_t}"
                ));
            }
            write_i64_le(
                history_kind,
                idx.saturating_mul(bucket_t).saturating_add(pos).saturating_mul(8),
                *kind,
            )?;
            write_bool_byte(history_mask, idx.saturating_mul(bucket_t).saturating_add(pos), true)?;
        }
        // Legal mask full-row copy.
        for (j, flag) in row.legal.iter().enumerate() {
            write_bool_byte(legal, idx.saturating_mul(BASELINE_ACTIONS).saturating_add(j), *flag)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 surface.
// ---------------------------------------------------------------------------

/// Stage one encoder batch in Rust; serve raw LE bytes for zero-copy
/// `torch.frombuffer` handoff.
///
/// `observations` is the `ActorObservation` list the deleted numpy oracle
/// looped over; `buckets` is `None` (frozen `(32,64,128,256)`) or an explicit
/// custom probe (e.g. `(32,)`). Returns `(planes, bucket_t, max_history_len,
/// hashes)` where `planes` maps the 26 frozen feature names to writable
/// bytearrays (LE bytes, padding pre-filled `0`/`False`/`-1`), `bucket_t` is
/// the ceil for the batch max, `max_history_len` is that max, and `hashes`
/// are `str(observation_hash)` in row order for the caller to bind.
#[pyfunction]
#[pyo3(signature = (observations, buckets = None))]
fn stage_encoder_batch(
    py: Python<'_>,
    observations: Vec<Bound<'_, PyAny>>,
    buckets: Option<Vec<usize>>,
) -> PyResult<(Py<PyDict>, usize, usize, Vec<String>)> {
    if observations.is_empty() {
        return Err(PyValueError::new_err(
            "encode_observations requires at least one observation".to_string(),
        ));
    }
    let use_buckets: Vec<usize> = match buckets {
        Some(list) => {
            if list.is_empty() {
                return Err(PyValueError::new_err(
                    "encoder stage buckets must be non-empty".to_string(),
                ));
            }
            list
        }
        None => HISTORY_BUCKETS.to_vec(),
    };
    let frozen = use_buckets.as_slice() == HISTORY_BUCKETS.as_slice();
    // Attached: extract owned rows (only owned data crosses detach).
    let mut rows: Vec<StagedRow> = Vec::with_capacity(observations.len());
    let mut hashes: Vec<String> = Vec::with_capacity(observations.len());
    let mut max_history_len: usize = 0;
    for (idx, obs) in observations.iter().enumerate() {
        let staged = extract_row(obs, idx)?;
        if staged.history.len() > max_history_len {
            max_history_len = staged.history.len();
        }
        hashes.push(staged.hash.clone());
        rows.push(staged);
    }
    let cap = match use_buckets.last() {
        Some(last) => *last,
        None => HISTORY_BUCKETS[HISTORY_BUCKETS.len() - 1],
    };
    if max_history_len > cap {
        return Err(PyValueError::new_err(format!(
            "visible history {max_history_len} exceeds model bucket cap {cap}; rows are never truncated"
        )));
    }
    let bucket_t = bucket_ceil_impl(max_history_len, &use_buckets);
    if frozen {
        // Frozen-geometry judge (oracle-identical messages via shared text).
        if bucket_t != bucket_frozen(max_history_len) {
            return Err(PyValueError::new_err(format!(
                "ring bucket_t {bucket_t} != ceil({max_history_len}) = {}",
                bucket_frozen(max_history_len)
            )));
        }
        if !HISTORY_BUCKETS.contains(&bucket_t) {
            return Err(PyValueError::new_err(format!(
                "ring bucket_t {bucket_t} not in buckets {:?}",
                HISTORY_BUCKETS
            )));
        }
    } else if !use_buckets.contains(&bucket_t) {
        return Err(PyValueError::new_err(format!(
            "ring bucket_t {bucket_t} not in buckets {use_buckets:?}"
        )));
    }
    let batch_rows = rows.len();
    let t_len = bucket_t;
    // Attached: allocate plane buffers (0x00 except -1 sentinels 0xFF).
    let mut actor = vec![0u8; batch_rows.saturating_mul(8)];
    let mut can_riichi = vec![0u8; batch_rows];
    let mut can_tsumo = vec![0u8; batch_rows];
    let mut furiten = vec![0u8; batch_rows.saturating_mul(8)];
    let mut seats = vec![0u8; batch_rows.saturating_mul(8)];
    let mut concealed = vec![0u8; batch_rows.saturating_mul(34).saturating_mul(4)];
    let mut dealer = vec![0u8; batch_rows.saturating_mul(8)];
    let mut dora = vec![0xFFu8; batch_rows.saturating_mul(5).saturating_mul(4)];
    let mut hand_number = vec![0u8; batch_rows.saturating_mul(4)];
    let mut history_kind = vec![0u8; batch_rows.saturating_mul(t_len).saturating_mul(8)];
    let mut history_mask = vec![0u8; batch_rows.saturating_mul(t_len)];
    let mut honba = vec![0u8; batch_rows.saturating_mul(4)];
    let mut ippatsu = vec![0u8; batch_rows.saturating_mul(4)];
    let mut kan_count = vec![0u8; batch_rows.saturating_mul(4)];
    let mut legal = vec![0u8; batch_rows.saturating_mul(BASELINE_ACTIONS)];
    let mut live_wall = vec![0u8; batch_rows.saturating_mul(4)];
    let mut own_drawn = vec![0xFFu8; batch_rows.saturating_mul(4)];
    let mut phase = vec![0u8; batch_rows.saturating_mul(8)];
    let mut riichi_states = vec![0u8; batch_rows.saturating_mul(4).saturating_mul(8)];
    let mut riichi_sticks = vec![0u8; batch_rows.saturating_mul(4)];
    let mut round_index = vec![0u8; batch_rows.saturating_mul(4)];
    let mut round_wind = vec![0u8; batch_rows.saturating_mul(8)];
    let mut scores = vec![0u8; batch_rows.saturating_mul(4).saturating_mul(4)];
    let mut seat_winds = vec![0u8; batch_rows.saturating_mul(4).saturating_mul(8)];
    let mut turn_actor = vec![0u8; batch_rows.saturating_mul(8)];
    let mut visible = vec![0u8; batch_rows.saturating_mul(34).saturating_mul(4)];
    // Detached: bulk fill (pure CPU, no Python).
    py.detach(|| {
        fill_buffers(
            &rows,
            t_len,
            &mut actor,
            &mut can_riichi,
            &mut can_tsumo,
            &mut furiten,
            &mut seats,
            &mut concealed,
            &mut dealer,
            &mut dora,
            &mut hand_number,
            &mut history_kind,
            &mut history_mask,
            &mut honba,
            &mut ippatsu,
            &mut kan_count,
            &mut legal,
            &mut live_wall,
            &mut own_drawn,
            &mut phase,
            &mut riichi_states,
            &mut riichi_sticks,
            &mut round_index,
            &mut round_wind,
            &mut scores,
            &mut seat_winds,
            &mut turn_actor,
            &mut visible,
        )
    })
    .map_err(PyValueError::new_err)?;
    // Attached: serve writable bytearrays (torch.frombuffer borrows, never copies).
    let dict = PyDict::new(py);
    dict.set_item("actor", PyByteArray::new(py, &actor))?;
    dict.set_item("actor_can_riichi", PyByteArray::new(py, &can_riichi))?;
    dict.set_item("actor_can_tsumo", PyByteArray::new(py, &can_tsumo))?;
    dict.set_item("actor_furiten", PyByteArray::new(py, &furiten))?;
    dict.set_item("actor_seats", PyByteArray::new(py, &seats))?;
    dict.set_item("concealed_hand_counts", PyByteArray::new(py, &concealed))?;
    dict.set_item("dealer", PyByteArray::new(py, &dealer))?;
    dict.set_item("dora_indicators", PyByteArray::new(py, &dora))?;
    dict.set_item("hand_number", PyByteArray::new(py, &hand_number))?;
    dict.set_item("history_event_kind", PyByteArray::new(py, &history_kind))?;
    dict.set_item("history_mask", PyByteArray::new(py, &history_mask))?;
    dict.set_item("honba", PyByteArray::new(py, &honba))?;
    dict.set_item("ippatsu_active", PyByteArray::new(py, &ippatsu))?;
    dict.set_item("kan_count", PyByteArray::new(py, &kan_count))?;
    dict.set_item("legal_mask", PyByteArray::new(py, &legal))?;
    dict.set_item("live_wall_tiles_remaining", PyByteArray::new(py, &live_wall))?;
    dict.set_item("own_drawn_tile", PyByteArray::new(py, &own_drawn))?;
    dict.set_item("phase", PyByteArray::new(py, &phase))?;
    dict.set_item("riichi_states", PyByteArray::new(py, &riichi_states))?;
    dict.set_item("riichi_sticks", PyByteArray::new(py, &riichi_sticks))?;
    dict.set_item("round_index", PyByteArray::new(py, &round_index))?;
    dict.set_item("round_wind", PyByteArray::new(py, &round_wind))?;
    dict.set_item("scores", PyByteArray::new(py, &scores))?;
    dict.set_item("seat_winds", PyByteArray::new(py, &seat_winds))?;
    dict.set_item("turn_actor", PyByteArray::new(py, &turn_actor))?;
    dict.set_item("visible_discards_counts", PyByteArray::new(py, &visible))?;
    Ok((dict.unbind(), bucket_t, max_history_len, hashes))
}

/// Register the `encoder` submodule (mirrors `ring::register`): extract
/// attached, fill detached, serve attached; single cdylib, no new entry.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "encoder")?;
    sub.add_function(wrap_pyfunction!(stage_encoder_batch, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod encoder_tests {
    use super::*;

    #[test]
    fn kind_ids_match_oracle_order() {
        assert_eq!(event_kind_id("game_start"), 0);
        assert_eq!(event_kind_id("turn_advance"), 2);
        assert_eq!(event_kind_id("discard"), 4);
        assert_eq!(event_kind_id("game_end"), 20);
        assert_eq!(event_kind_id("bogus-kind"), 0);
    }

    #[test]
    fn phase_furiten_riichi_wind_closed() {
        assert_eq!(phase_id("draw_decision"), Ok(1));
        assert_eq!(furiten_id("discard"), Ok(3));
        assert_eq!(riichi_id("accepted"), Ok(2));
        assert_eq!(wind_id(27), Ok(0));
        assert_eq!(wind_id(30), Ok(3));
        assert!(phase_id("bogus").is_err());
        assert!(furiten_id("bogus").is_err());
        assert!(riichi_id("bogus").is_err());
        assert!(wind_id(0).is_err());
    }

    #[test]
    fn bucket_ceil_frozen_and_custom() {
        assert_eq!(bucket_ceil_impl(0, &HISTORY_BUCKETS), 32);
        assert_eq!(bucket_ceil_impl(33, &HISTORY_BUCKETS), 64);
        assert_eq!(bucket_ceil_impl(5, &[32]), 32);
        assert_eq!(bucket_ceil_impl(5, &[64]), 64);
        assert_eq!(bucket_ceil_impl(999, &HISTORY_BUCKETS), 256);
    }

    #[test]
    fn le_writes_shape() {
        let mut buf = vec![0u8; 8];
        write_i64_le(&mut buf, 0, 3).unwrap();
        assert_eq!(i64::from_le_bytes(buf[0..8].try_into().unwrap()), 3);
        let mut buf32 = vec![0u8; 4];
        write_i32_le(&mut buf32, 0, -1).unwrap();
        assert_eq!(buf32, vec![0xFF, 0xFF, 0xFF, 0xFF]);
        let mut buf_bool = vec![0u8; 1];
        write_bool_byte(&mut buf_bool, 0, true).unwrap();
        assert_eq!(buf_bool, vec![1]);
    }

    #[test]
    fn detach_smoke_bucket() {
        Python::attach(|py| {
            let bucket = py.detach(|| bucket_frozen(33));
            assert_eq!(bucket, 64);
        });
    }
}
