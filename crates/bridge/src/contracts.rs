//! contracts: thin SPEC-contracts boundary over `hydra-feed` (Wave 4).
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as the
//! rest of the bridge; no new dependency). FORBIDS: canon re-printing (the
//! feed `canon` owner is the single printer), wall-clock/urandom seeding
//! (semantic counter-based seeds only), and live-object orchestration
//! (ledgers, registries, builders holding callbacks stay Python-side).
//! Arg-check runs attached, every compute section runs under ONE
//! `py.detach(|| ...)` with zero Python API inside, and results are wrapped
//! attached — EXCEPT the frozen-census leaves below (rank 4 size/counts and
//! rank 5 `census_index_of`): a sub-microsecond slice `binary_search` over the
//! process-once `frozen_census()` costs less than a GIL round-trip, so those
//! run attached end-to-end. Fail-closed: every shape violation is
//! `PyValueError`, never a default or a fallback.
//!
//! Rank map (Wave 4 leaves-before-roots; this module fills the deterministic
//! leaves, the roots stay Python until the caller-migration round):
//! - rank 1 atomic: digest gates already bridged (`canon_rng.sha256_digest` /
//!   `sha256_file`); temp/link/fsync discipline plus `os.urandom` temp names
//!   stay Python (OS authority + nondeterministic names, never ported).
//! - rank 2 comparators: `is_seat` / `is_tile_id` / `is_digest_text` /
//!   `make_seat` / `make_tile_id` / `make_action_id` / `make_digest_text`
//!   (mirror `contracts/common.py`; bool is never an int).
//! - rank 3 validators: `is_action_kind` / `action_kind_ordinal` /
//!   `tile_type_of` / `is_event_kind` / `is_visibility` (frozen vocabularies;
//!   record `__post_init__` bodies stay Python as the validating roots).
//! - rank 4 census: `action_kind_ordinals` / `action_census_count` /
//!   `action_census_counts` / `action_census` / `census_template_at`
//!   (6792 frozen records in generation order, `PacketSuccessor` precedent).
//! - rank 5 codec leaves: `census_index_of` (table-id lookup = bisect on sort
//!   keys, `None` when absent); context-gated encode and live-object builders
//!   stay Python-side.
//! - rank 6 digest identities: every canon+hash identity is already covered by
//!   `canon_rng.of_canonical_json` / `sha256_hex`; this module adds only the
//!   multi-step fold `public_chain_hash` (SPEC 7.2) that has no single-doc
//!   form.
//! - rank 7 canon/RNG: the canon-bytes authority stays Python by design (no
//!   bytes-returning pyfn); Philox NEW streams already live in `canon_rng`;
//!   this module adds the legacy sha256-CTR leaves `ctr_block` /
//!   `ctr_stream_bytes` (SPEC 13, pure functions of seed + counter) while
//!   `semantic_seed` judging, `StreamLedger`, and checkpoints stay Python.
//! - rank 8 scoring: the fixed-point judges (`validate_ranks`,
//!   `utility_fixed`, `utility_for_ranks_fixed`, `exact_total_is_zero`) are
//!   already bridged; this module adds `resolve_final_ranks` (SPEC 5.1 east1
//!   tie-break) while `utility()` Fraction scoring stays the Python oracle.
//!
//! Single-cdylib tree: registers as the `hydra2._native.contracts` submodule,
//! mirroring `canon_rng::register`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyFrozenSet, PyModule, PyTuple};
use std::collections::BTreeMap;

/// Score domain for rank resolution (`rules_manifest.py:405-407`).
const SCORE_MIN: i64 = -(1_000_000_000_000);
const SCORE_MAX: i64 = 1_000_000_000_000;

/// CTR block width in bytes (`RandomStream._BLOCK`, `randomness.py:406`).
const CTR_BLOCK: u64 = 32;

// ---------------------------------------------------------------------------
// Rank 2 - comparators (mirror `contracts/common.py`).
// ---------------------------------------------------------------------------

/// Plain-int view of a Python value: `bool` is excluded first (it subclasses
/// `int`), non-ints and out-of-`i64` values map to `None` (predicates report
/// `False`; makers raise instead).
fn plain_int_value(obj: &Bound<'_, PyAny>) -> Option<i64> {
    if obj.is_instance_of::<PyBool>() {
        return None;
    }
    obj.extract::<i64>().ok()
}

/// Fail-closed maker half of [`plain_int_value`].
fn plain_int_or_err(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    match plain_int_value(obj) {
        Some(value) => Ok(value),
        None => Err(PyValueError::new_err(format!(
            "contracts {name} must be a plain int (bool excluded)"
        ))),
    }
}

/// Narrowing predicate: true iff the value is a valid Seat (0..=3).
#[pyfunction]
fn is_seat(value: Bound<'_, PyAny>) -> bool {
    matches!(plain_int_value(&value), Some(v) if (0..=3).contains(&v))
}

/// Narrowing predicate: true iff the value is a valid TileId (0..=135).
#[pyfunction]
fn is_tile_id(value: Bound<'_, PyAny>) -> bool {
    matches!(plain_int_value(&value), Some(v) if (0..=135).contains(&v))
}

/// True iff the text matches `sha256:<64 lowercase hex>` exactly.
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

/// Narrowing predicate: true iff the value is a well-formed digest text.
#[pyfunction]
fn is_digest_text(value: &str) -> bool {
    is_digest_shape(value)
}

/// Validated Seat (0..=3, bool excluded).
#[pyfunction]
fn make_seat(value: Bound<'_, PyAny>) -> PyResult<u8> {
    let v = plain_int_or_err(&value, "seat")?;
    if !(0..=3).contains(&v) {
        return Err(PyValueError::new_err(format!(
            "contracts seat={v} outside [0, 3]"
        )));
    }
    // proof: `v` in 0..=3 (range-checked above), fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = v as u8;
    Ok(b)
}

/// Validated TileId (0..=135, bool excluded).
#[pyfunction]
fn make_tile_id(value: Bound<'_, PyAny>) -> PyResult<u8> {
    let v = plain_int_or_err(&value, "tile_id")?;
    if !(0..=135).contains(&v) {
        return Err(PyValueError::new_err(format!(
            "contracts tile_id={v} outside [0, 135]"
        )));
    }
    // proof: `v` in 0..=135 (range-checked above), fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = v as u8;
    Ok(b)
}

/// Validated ActionId (nonnegative int, bool excluded).
#[pyfunction]
fn make_action_id(value: Bound<'_, PyAny>) -> PyResult<u32> {
    let v = plain_int_or_err(&value, "action_id")?;
    if v < 0 {
        return Err(PyValueError::new_err(format!(
            "contracts action_id={v} must be a nonnegative int"
        )));
    }
    u32::try_from(v)
        .map_err(|_| PyValueError::new_err(format!("contracts action_id={v} outside u32")))
}

/// Validated digest text (returned verbatim; the shape is the value).
#[pyfunction]
fn make_digest_text(value: &str) -> PyResult<String> {
    if !is_digest_shape(value) {
        return Err(PyValueError::new_err(
            "contracts digest_text must match sha256:<64 lowercase hex>",
        ));
    }
    Ok(value.to_string())
}

/// Logical tile type 0..=33 of a validated physical id (`tile // 4`).
#[pyfunction]
fn tile_type_of(tile: Bound<'_, PyAny>) -> PyResult<u8> {
    let v = plain_int_or_err(&tile, "tile_id")?;
    if !(0..=135).contains(&v) {
        return Err(PyValueError::new_err(format!(
            "contracts tile_id={v} outside [0, 135]"
        )));
    }
    // proof: `v` in 0..=135, so `v/4` in 0..=33, fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = (v / 4) as u8;
    Ok(b)
}

/// Frozen SPEC 6.1 kind -> ordinal table (`action_kinds.py:71-85`).
const ACTION_KIND_TABLE: [(&str, u8); 13] = [
    ("pass", 0),
    ("discard", 1),
    ("tsumogiri", 2),
    ("riichi_discard", 3),
    ("chi", 4),
    ("pon", 5),
    ("daiminkan", 6),
    ("ankan", 7),
    ("kakan", 8),
    ("ron", 9),
    ("tsumo", 10),
    ("abort_nine_terminals", 11),
    ("accept_abortive_draw", 12),
];

/// True iff the literal is a frozen SPEC 6.1 action kind.
#[pyfunction]
fn is_action_kind(kind: &str) -> bool {
    ACTION_KIND_TABLE.iter().any(|(name, _)| *name == kind)
}

/// Frozen kind ordinal (never reordered); unknown kinds fail closed.
#[pyfunction]
fn action_kind_ordinal(kind: &str) -> PyResult<u8> {
    ACTION_KIND_TABLE
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, ordinal)| *ordinal)
        .ok_or_else(|| PyValueError::new_err(format!("contracts unknown action kind {kind:?}")))
}

/// Every (kind, ordinal) pair in frozen ordinal order.
#[pyfunction]
fn action_kind_ordinals() -> Vec<(String, u8)> {
    ACTION_KIND_TABLE
        .iter()
        .map(|(name, ordinal)| ((*name).to_string(), *ordinal))
        .collect()
}
/// Frozen SPEC 7.1 event vocabulary (`event_vocab.py:65-87`, SPEC order).
const EVENT_KINDS: [&str; 21] = [
    "game_start",
    "round_start",
    "turn_advance",
    "draw_tile",
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "call_window",
    "call_resolved",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
    "draw_end",
    "abortive_draw",
    "round_end",
    "game_end",
];

/// Frozen visibility vocabulary (`event_vocab.py:39`).
const VISIBILITIES: [&str; 3] = ["public", "actor_private", "server_private"];

/// True iff the literal is a frozen SPEC 7.1 event kind.
#[pyfunction]
fn is_event_kind(kind: &str) -> bool {
    EVENT_KINDS.contains(&kind)
}

/// True iff the literal is a frozen visibility.
#[pyfunction]
fn is_visibility(visibility: &str) -> bool {
    VISIBILITIES.contains(&visibility)
}

// ---------------------------------------------------------------------------
// Ranks 4-5 - census records + table-id leaves.
// ---------------------------------------------------------------------------

/// One canonical action slot (SPEC 6.3) as a frozen record: the
/// `PacketSuccessor` precedent for records crossing the boundary (live
/// Python objects never cross; callers pass scalars, records come back).
#[pyclass(name = "ActionTemplate", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyActionTemplate {
    /// Frozen kind literal.
    #[pyo3(get)]
    pub kind: String,
    /// Tile id, or `None` where the kind forbids one.
    #[pyo3(get)]
    pub tile: Option<u8>,
    /// Called tile id, or `None` where the kind carries none.
    #[pyo3(get)]
    pub called_tile: Option<u8>,
    /// Consumed tiles, unique ascending (empty where the kind carries none).
    #[pyo3(get)]
    pub consumed_tiles: Vec<u8>,
    /// Relative source offset, or `None` where the kind carries none.
    #[pyo3(get)]
    pub source_offset: Option<i8>,
    /// True exactly for riichi-discard templates.
    #[pyo3(get)]
    pub declares_riichi: bool,
    /// True exactly for kakan templates.
    #[pyo3(get)]
    pub meld_ref_required: bool,
}

impl PyActionTemplate {
    fn from_template(template: &hydra_feed::census::ActionTemplate) -> Self {
        Self {
            kind: template.kind.name().to_string(),
            tile: template.tile,
            called_tile: template.called_tile,
            consumed_tiles: template.consumed_slice().to_vec(),
            source_offset: template.source_offset,
            declares_riichi: template.declares_riichi,
            meld_ref_required: template.meld_ref_required,
        }
    }
}

/// Census size (6792): counted from the frozen process-once census, never hardcoded.
#[pyfunction]
fn action_census_count(_py: Python<'_>) -> usize {
    hydra_feed::census::frozen_census().len()
}

/// Per-kind template counts in frozen ordinal order, counted from the
/// frozen census (analytic pins live in the feed tests).
#[pyfunction]
fn action_census_counts(_py: Python<'_>) -> Vec<(String, usize)> {
    hydra_feed::census::census_counts(hydra_feed::census::frozen_census())
        .iter()
        .map(|(kind, count)| (kind.name().to_string(), *count))
        .collect()
}

/// The full 6792-template census in generation order (frozen records).
#[pyfunction]
fn action_census(_py: Python<'_>) -> Vec<PyActionTemplate> {
    hydra_feed::census::frozen_census()
        .iter()
        .map(PyActionTemplate::from_template)
        .collect()
}

/// Generation-order census entry by index (table-id decode leaf).
#[pyfunction]
fn census_template_at(_py: Python<'_>, index: usize) -> PyResult<PyActionTemplate> {
    let census = hydra_feed::census::frozen_census();
    census
        .get(index)
        .map(PyActionTemplate::from_template)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "contracts census index {index} outside [0, {})",
                census.len()
            ))
        })
}

/// Optional validated tile-or-`None` gate shared by the codec leaf.
fn opt_tile(value: Option<Bound<'_, PyAny>>, name: &str) -> PyResult<Option<u8>> {
    match value {
        None => Ok(None),
        Some(obj) => {
            let v = plain_int_or_err(&obj, name)?;
            if !(0..=135).contains(&v) {
                return Err(PyValueError::new_err(format!(
                    "contracts {name}={v} outside [0, 135]"
                )));
            }
            // proof: `v` in 0..=135 (range-checked above), fits `u8`.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let b: u8 = v as u8;
            Ok(Some(b))
        }
    }
}

/// Generation-order index of a template slot, or `None` when absent (mirrors
/// `ActionTable.index_of`: bisect on sort keys, never a partial output).
/// Context-gated encode (actor/meld resolution) stays Python-side.
#[pyfunction]
#[pyo3(signature = (kind, consumed, tile=None, called_tile=None, source_offset=None))]
fn census_index_of(
    _py: Python<'_>,
    kind: &str,
    consumed: Vec<Bound<'_, PyAny>>,
    tile: Option<Bound<'_, PyAny>>,
    called_tile: Option<Bound<'_, PyAny>>,
    source_offset: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<usize>> {
    let action_kind = hydra_feed::census::ActionKind::from_name(kind)
        .ok_or_else(|| PyValueError::new_err(format!("contracts unknown action kind {kind:?}")))?;
    let tile = opt_tile(tile, "tile")?;
    let called_tile = opt_tile(called_tile, "called_tile")?;
    if consumed.len() > 4 {
        return Err(PyValueError::new_err(
            "contracts consumed_tiles holds at most 4 entries",
        ));
    }
    let mut arr = [0u8; 4];
    for (i, entry) in consumed.iter().enumerate() {
        let v = plain_int_or_err(entry, "consumed_tiles")?;
        if !(0..=135).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "contracts consumed_tiles entry {v} outside [0, 135]"
            )));
        }
        // proof: `v` in 0..=135 (range-checked above), fits `u8`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let b: u8 = v as u8;
        arr[i] = b;
    }
    let offset = match source_offset {
        None => None,
        Some(obj) => {
            let v = plain_int_or_err(&obj, "source_offset")?;
            if !(-1..=2).contains(&v) {
                return Err(PyValueError::new_err(format!(
                    "contracts source_offset={v} must be one of (None, -1, 0, 1, 2)"
                )));
            }
            // proof: `v` in -1..=2 (range-checked above), fits `i8`.
            #[allow(clippy::cast_possible_truncation)]
            let b: i8 = v as i8;
            Some(b)
        }
    };
    // riichi/meld flags are kind-derived, exactly as the Python record
    // constructor enforces: only genuinely present templates can hit.
    // proof: `consumed.len()` <= 4 (checked `> 4` above), fits `u8`.
    #[allow(clippy::cast_possible_truncation)]
    let consumed_len: u8 = consumed.len() as u8;
    let probe = hydra_feed::census::ActionTemplate {
        kind: action_kind,
        tile,
        called_tile,
        consumed: arr,
        consumed_len,
        source_offset: offset,
        declares_riichi: action_kind == hydra_feed::census::ActionKind::RiichiDiscard,
        meld_ref_required: action_kind == hydra_feed::census::ActionKind::Kakan,
    };
    // Attached sub-microsecond binary search over the frozen process-once
    // census: cheaper than a GIL round-trip (see module docs). Fail-closed
    // arg-check above is unchanged.
    Ok(hydra_feed::census::census_index_of(
        hydra_feed::census::frozen_census(),
        &probe,
    ))
}

// ---------------------------------------------------------------------------
// Rank 6 - packet fold (SPEC 7.2).
// ---------------------------------------------------------------------------

/// Fold canonical envelope documents into a chained public-state hash, in
/// input order (mirrors `public_state_chain_hash`). Only `visibility ==
/// "public"` documents advance the chain; the empty input yields the
/// empty-chain digest. Single-doc digests stay on `canon_rng.of_canonical_json`.
#[pyfunction]
fn public_chain_hash(py: Python<'_>, event_docs: Vec<Vec<u8>>) -> PyResult<String> {
    py.detach(|| hydra_feed::chain::public_chain_hash(&event_docs))
        .map_err(|e| PyValueError::new_err(format!("contracts public_chain_hash rejected: {e}")))
}

// ---------------------------------------------------------------------------
// Rank 7 - legacy sha256-CTR leaves (SPEC 13).
// ---------------------------------------------------------------------------

/// One legacy CTR block: `sha256(DOMAIN || len(seed)BE32 || seed ||
/// indexBE64)` (mirrors `RandomStream._block_at`). Pure function of
/// `(seed, index)`; empty seeds fail closed. Philox NEW streams are untouched
/// (`canon_rng`); ledgers and checkpoints stay Python-side.
#[pyfunction]
fn ctr_block(py: Python<'_>, seed: Vec<u8>, index: u64) -> PyResult<Py<PyBytes>> {
    if seed.is_empty() {
        return Err(PyValueError::new_err(
            "contracts ctr_block: seed must be nonempty bytes",
        ));
    }
    let out = py.detach(|| hydra_feed::digest::ctr_block(&seed, index));
    Ok(PyBytes::new(py, &out).unbind())
}

/// Counter-based byte window `[cursor, cursor + count)` over the CTR stream
/// (mirrors `RandomStream.get_bytes` slicing without the live cursor).
/// Checked arithmetic throughout: overflow fails closed, never wraps.
#[pyfunction]
fn ctr_stream_bytes(
    py: Python<'_>,
    seed: Vec<u8>,
    cursor: u64,
    count: usize,
) -> PyResult<Py<PyBytes>> {
    if seed.is_empty() {
        return Err(PyValueError::new_err(
            "contracts ctr_stream_bytes: seed must be nonempty bytes",
        ));
    }
    let first_block = cursor / CTR_BLOCK;
    // proof: remainder < 32 (`CTR_BLOCK` = 32), fits `usize` even on 32-bit.
    #[allow(clippy::cast_possible_truncation)]
    let offset = (cursor % CTR_BLOCK) as usize;
    let end = offset
        .checked_add(count)
        .ok_or_else(|| PyValueError::new_err("contracts ctr_stream_bytes: window overflows"))?;
    // proof: `CTR_BLOCK` = 32 const, fits `usize`.
    #[allow(clippy::cast_possible_truncation)]
    let needed = end.div_ceil(CTR_BLOCK as usize);
    let out = py
        .detach(|| -> Result<Vec<u8>, String> {
            let mut acc: Vec<u8> = Vec::new();
            let mut block = first_block;
            for _ in 0..needed {
                acc.extend_from_slice(&hydra_feed::digest::ctr_block(&seed, block));
                block = block.checked_add(1).ok_or_else(|| {
                    "contracts ctr_stream_bytes: block counter overflows u64".to_string()
                })?;
            }
            acc.get(offset..end).map(<[u8]>::to_vec).ok_or_else(|| {
                "contracts ctr_stream_bytes: window outside accumulated blocks".to_string()
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

// ---------------------------------------------------------------------------
// Rank 8 - final-rank resolution (SPEC 5.1).
// ---------------------------------------------------------------------------

/// Placement ranks 1..=4 from raw final scores with the east1 seat-wind
/// tie-break (mirrors `resolve_final_ranks`): higher score first, lower seat
/// on ties. Inputs are validated bridge-side (+/-1e12 domain, bool excluded);
/// the feed sort is total over `[i64; 4]`.
#[pyfunction]
fn resolve_final_ranks(py: Python<'_>, scores: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<u8>> {
    if scores.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts resolve_final_ranks: scores must be exactly 4 entries",
        ));
    }
    let mut arr = [0i64; 4];
    for (i, entry) in scores.iter().enumerate() {
        let v = plain_int_or_err(entry, "final_scores")?;
        if !(SCORE_MIN..=SCORE_MAX).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "contracts final_scores entry {v} outside [{SCORE_MIN}, {SCORE_MAX}]"
            )));
        }
        arr[i] = v;
    }
    Ok(py
        .detach(|| hydra_feed::fixed::resolve_final_ranks(arr))
        .to_vec())
}

// ---------------------------------------------------------------------------
// Wave 4 round-2 - R5 codec pure leaves (SPEC 6.3; context gating stays Python).
// ---------------------------------------------------------------------------

/// Relative source offset of `source` seen from `actor` (mirrors
/// `action_table._offset_from_source`): `None` in, `None` out; otherwise
/// `(source - actor) % 4` mapped `3 -> -1`. `source == actor` fails closed
/// (defensive; validated actions never self-source). Bool excluded, seats
/// `0..=3`. Attached (sub-microsecond arithmetic, cheaper than a GIL
/// round-trip; see census leaf docs). Counter-free deterministic, never
/// wall-clock.
#[pyfunction]
#[pyo3(signature = (source, actor))]
fn offset_from_source(
    source: Option<Bound<'_, PyAny>>,
    actor: Bound<'_, PyAny>,
) -> PyResult<Option<i8>> {
    let actor_v = plain_int_or_err(&actor, "actor")?;
    if !(0..=3).contains(&actor_v) {
        return Err(PyValueError::new_err(format!(
            "contracts actor={actor_v} outside [0, 3]"
        )));
    }
    match source {
        None => Ok(None),
        Some(obj) => {
            let source_v = plain_int_or_err(&obj, "source")?;
            if !(0..=3).contains(&source_v) {
                return Err(PyValueError::new_err(format!(
                    "contracts source={source_v} outside [0, 3]"
                )));
            }
            if source_v == actor_v {
                return Err(PyValueError::new_err("contracts source seat equals actor"));
            }
            let delta = (source_v - actor_v).rem_euclid(4);
            if delta == 3 {
                Ok(Some(-1))
            } else {
                i8::try_from(delta).map(Some).map_err(|_| {
                    PyValueError::new_err(format!("contracts offset delta {delta} outside i8"))
                })
            }
        }
    }
}

/// Absolute source seat of `offset` seen from `actor` (mirrors
/// `action_table._resolve_source`): `None` in, `None` out; otherwise
/// `(actor + delta) % 4` with `{-1: 3, 0: 0, 1: 1, 2: 2}`. Offsets outside
/// `(-1, 0, 1, 2)` fail closed. Attached (see `offset_from_source`).
/// Counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (offset, actor))]
fn resolve_source(
    offset: Option<Bound<'_, PyAny>>,
    actor: Bound<'_, PyAny>,
) -> PyResult<Option<u8>> {
    let actor_v = plain_int_or_err(&actor, "actor")?;
    if !(0..=3).contains(&actor_v) {
        return Err(PyValueError::new_err(format!(
            "contracts actor={actor_v} outside [0, 3]"
        )));
    }
    match offset {
        None => Ok(None),
        Some(obj) => {
            let offset_v = plain_int_or_err(&obj, "source_offset")?;
            let delta = match offset_v {
                -1 => 3i64,
                0 => 0i64,
                1 => 1i64,
                2 => 2i64,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "contracts source_offset={offset_v} must be one of (None, -1, 0, 1, 2)"
                    )));
                }
            };
            let seat = (actor_v + delta).rem_euclid(4);
            u8::try_from(seat).map(Some).map_err(|_| {
                PyValueError::new_err(format!("contracts source seat {seat} outside u8"))
            })
        }
    }
}

/// Frozen meld kinds (`observation_types.MELD_KINDS`, SPEC 8 order).
const MELD_KINDS: [&str; 5] = ["chi", "pon", "daiminkan", "ankan", "kakan"];

/// Canonical prior-meld reference `kind:t.t.t` (mirrors
/// `observation_types.visible_meld_id`): takes scalars (`kind` + tile list),
/// never a live `VisibleMeld` object, so callers pass ints and get text back
/// (`PacketSuccessor` precedent). Kind must be a meld kind, tiles must be
/// valid tile ids `0..=135` (bool excluded). Attached (string join only).
/// Counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (kind, tiles))]
fn visible_meld_id(kind: String, tiles: Vec<Bound<'_, PyAny>>) -> PyResult<String> {
    if !MELD_KINDS.contains(&kind.as_str()) {
        return Err(PyValueError::new_err(format!(
            "contracts meld kind must be one of {MELD_KINDS:?}, got {kind:?}"
        )));
    }
    let mut ids: Vec<i64> = Vec::with_capacity(tiles.len());
    for entry in tiles.iter() {
        let v = plain_int_or_err(entry, "tiles")?;
        if !(0..=135).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "contracts meld tiles entry {v} outside [0, 135]"
            )));
        }
        ids.push(v);
    }
    let joined = ids
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(".");
    Ok(format!("{kind}:{joined}"))
}

/// One exposed meld visible to every actor (SPEC 8 exact field order) as a
/// frozen record: the `PacketSuccessor` precedent for records crossing the
/// boundary (callers pass scalars, records come back; live
/// `ObservationBuilder` caches stay Python-side). Validation mirrors
/// `observation_types.VisibleMeld.__post_init__` bit-for-bit (kind vocab,
/// seat/tile gates with bool excluded, per-kind length, chi run vs same-type,
/// ankan consecutive-4, called/source presence with `source != owner` and
/// `called in tiles`, non-empty `meld_id` derived via `visible_meld_id`
/// when `None`). Attached (at most 4 tiles; cheaper than a GIL round-trip).
/// Counter-free deterministic, never wall-clock. Fail-closed: every shape
/// violation is `PyValueError`, never a default.
#[pyclass(name = "VisibleMeld", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyVisibleMeld {
    /// Resolved meld id (derived `kind:tiles` when `None` in).
    #[pyo3(get)]
    pub meld_id: String,
    /// Frozen meld kind literal.
    #[pyo3(get)]
    pub kind: String,
    /// Owning seat `0..=3`.
    #[pyo3(get)]
    pub owner: u8,
    /// Offering seat for chi/pon/daiminkan, `None` for ankan/kakan.
    #[pyo3(get)]
    pub source_seat: Option<u8>,
    /// Claimed tile for chi/pon/daiminkan, `None` for ankan/kakan.
    #[pyo3(get)]
    pub called_tile: Option<u8>,
    /// Meld tiles, unique ascending.
    #[pyo3(get)]
    pub tiles: Vec<u8>,
}

#[pymethods]
impl PyVisibleMeld {
    #[new]
    #[pyo3(signature = (kind, owner, tiles, meld_id=None, source_seat=None, called_tile=None))]
    fn new(
        kind: String,
        owner: Bound<'_, PyAny>,
        tiles: Vec<Bound<'_, PyAny>>,
        meld_id: Option<String>,
        source_seat: Option<Bound<'_, PyAny>>,
        called_tile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !MELD_KINDS.contains(&kind.as_str()) {
            return Err(PyValueError::new_err(format!(
                "contracts meld kind must be one of {MELD_KINDS:?}, got {kind:?}"
            )));
        }
        let owner_v = plain_int_or_err(&owner, "owner")?;
        if !(0..=3).contains(&owner_v) {
            return Err(PyValueError::new_err(format!(
                "contracts owner={owner_v} outside [0, 3]"
            )));
        }
        let mut tile_ids: Vec<u8> = Vec::with_capacity(tiles.len());
        for entry in tiles.iter() {
            let v = plain_int_or_err(entry, "tiles")?;
            if !(0..=135).contains(&v) {
                return Err(PyValueError::new_err(format!(
                    "contracts meld tiles entry {v} outside [0, 135]"
                )));
            }
            let byte = u8::try_from(v).map_err(|_| {
                PyValueError::new_err(format!("contracts meld tile {v} outside u8"))
            })?;
            tile_ids.push(byte);
        }
        if tile_ids.is_empty() {
            return Err(PyValueError::new_err(
                "contracts meld tiles must be non-empty",
            ));
        }
        let mut sorted = tile_ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != tile_ids.len() || tile_ids != sorted {
            return Err(PyValueError::new_err(format!(
                "contracts {kind} meld tiles must be unique ascending: {tile_ids:?}"
            )));
        }
        let expected_len = match kind.as_str() {
            "chi" | "pon" => 3,
            "daiminkan" | "ankan" | "kakan" => 4,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "contracts meld kind must be one of {MELD_KINDS:?}, got {kind:?}"
                )));
            }
        };
        if tile_ids.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "contracts {kind} meld must hold {expected_len} tiles, got {}",
                tile_ids.len()
            )));
        }
        let types: Vec<u8> = tile_ids.iter().map(|t| t / 4).collect();
        if kind == "chi" {
            let has_honor = types.iter().any(|t| *t >= 27);
            let mut suits: Vec<u8> = types.iter().map(|t| t / 9).collect();
            suits.sort_unstable();
            suits.dedup();
            let min_type = types.iter().min().copied().unwrap_or(0);
            let max_type = types.iter().max().copied().unwrap_or(0);
            let mut uniq_types = types.clone();
            uniq_types.sort_unstable();
            uniq_types.dedup();
            if has_honor || suits.len() != 1 || max_type - min_type != 2 || uniq_types.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "contracts chi meld is not a same-suit run: {tile_ids:?}"
                )));
            }
        } else {
            let mut uniq_types = types.clone();
            uniq_types.sort_unstable();
            uniq_types.dedup();
            if uniq_types.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "contracts {kind} meld tiles must share one logical type: {tile_ids:?}"
                )));
            }
        }
        let source_opt: Option<u8> = match source_seat {
            None => None,
            Some(obj) => {
                let v = plain_int_or_err(&obj, "source_seat")?;
                if !(0..=3).contains(&v) {
                    return Err(PyValueError::new_err(format!(
                        "contracts source_seat={v} outside [0, 3]"
                    )));
                }
                // proof: `v` in 0..=3 (range-checked above), fits `u8`.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b: u8 = v as u8;
                Some(b)
            }
        };
        let called_opt: Option<u8> = match called_tile {
            None => None,
            Some(obj) => {
                let v = plain_int_or_err(&obj, "called_tile")?;
                if !(0..=135).contains(&v) {
                    return Err(PyValueError::new_err(format!(
                        "contracts called_tile={v} outside [0, 135]"
                    )));
                }
                // proof: `v` in 0..=135 (range-checked above), fits `u8`.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let b: u8 = v as u8;
                Some(b)
            }
        };
        if kind == "ankan" || kind == "kakan" {
            if called_opt.is_some() || source_opt.is_some() {
                return Err(PyValueError::new_err(format!(
                    "contracts {kind} meld has no called tile or source seat"
                )));
            }
            if kind == "ankan" {
                let base = u16::from(types[0]) * 4;
                let mut want: Vec<u8> = Vec::with_capacity(4);
                for i in 0..4u16 {
                    let v = base + i;
                    let byte = u8::try_from(v).map_err(|_| {
                        PyValueError::new_err(format!("contracts ankan base {base} outside u8"))
                    })?;
                    want.push(byte);
                }
                if tile_ids != want {
                    return Err(PyValueError::new_err(format!(
                        "contracts ankan meld tiles {tile_ids:?} must be consecutive 4 of type {}",
                        types[0]
                    )));
                }
            }
        } else {
            let source = source_opt.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "contracts {kind} meld requires called_tile and source_seat"
                ))
            })?;
            let called = called_opt.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "contracts {kind} meld requires called_tile and source_seat"
                ))
            })?;
            // proof: `owner_v` in 0..=3 (range-checked above), fits `u8`.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let owner_b: u8 = owner_v as u8;
            if source == owner_b {
                return Err(PyValueError::new_err(format!(
                    "contracts {kind} meld source seat equals owner"
                )));
            }
            if !tile_ids.contains(&called) {
                return Err(PyValueError::new_err(format!(
                    "contracts {kind} meld called tile {called} not among tiles"
                )));
            }
        }
        let resolved = match meld_id {
            Some(text) => {
                if text.is_empty() {
                    return Err(PyValueError::new_err(
                        "contracts meld_id must resolve to a non-empty string",
                    ));
                }
                text
            }
            None => {
                let joined = tile_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(".");
                format!("{kind}:{joined}")
            }
        };
        // proof: `owner_v` in 0..=3 (range-checked above), fits `u8`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let owner_b2: u8 = owner_v as u8;
        Ok(Self {
            meld_id: resolved,
            kind,
            owner: owner_b2,
            source_seat: source_opt,
            called_tile: called_opt,
            tiles: tile_ids,
        })
    }
}

// ---------------------------------------------------------------------------
// Wave 4 round-2 - R6 packet folds (SPEC 7.2; live packets stay Python).
// ---------------------------------------------------------------------------

/// One public-state fold step `sha256(canonical({"prefix": prefix, "event":
/// event}))` (mirrors `event_packet._fold_public_hash`). Takes the running
/// prefix digest text plus ONE canonical envelope document (prepared
/// Python-side via `envelope_identity_document` + `canonical_bytes`, so the
/// canon authority stays Python and Rust only hashes). Complements the
/// multi-step `public_chain_hash`. Compute runs detached with zero Python API
/// inside; prefix shape and doc parse failures are `PyValueError`, never a
/// default. Counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (prefix, event_doc))]
fn fold_public_hash(py: Python<'_>, prefix: String, event_doc: Vec<u8>) -> PyResult<String> {
    if !is_digest_shape(&prefix) {
        return Err(PyValueError::new_err(format!(
            "contracts fold_public_hash prefix must be 'sha256:' + 64 lowercase hex, got {prefix:?}"
        )));
    }
    py.detach(|| hydra_feed::chain::fold_public_hash(&prefix, &event_doc))
        .map_err(|e| PyValueError::new_err(format!("contracts fold_public_hash rejected: {e}")))
}

/// Packet identity hex over canonical packet-identity bytes (mirrors
/// `event_packet.compute_packet_id` without the live `ActorVisiblePacket`):
/// the caller prepares `canonical_bytes(packet_identity_document(packet))`
/// (packet WITHOUT `packet_id`, canon authority stays Python); Rust
/// re-canonicalizes through the feed `canon` owner and hashes through the
/// feed `digest` owner, returning lowercase hex WITHOUT the `sha256:` prefix
/// (matching `PacketId(hexdigest)`). Distinct from the `sha256:`-prefixed
/// single-doc digests on `canon_rng.of_canonical_json`. Compute runs detached
/// with zero Python API inside; parse/canon failures are `PyValueError`.
/// Counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (packet_doc,))]
fn packet_id_from_doc(py: Python<'_>, packet_doc: Vec<u8>) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let value =
            hydra_feed::canon::parse_canonical_bytes(&packet_doc, "contracts:packet_id_from_doc")
                .map_err(|e| e.to_string())?;
        let bytes =
            hydra_feed::canon::canonical_bytes_value(&value, "contracts:packet_id_from_doc")
                .map_err(|e| e.to_string())?;
        let digest = hydra_feed::digest::sha256_hex(&bytes);
        digest
            .strip_prefix("sha256:")
            .map(|hex| hex.to_string())
            .ok_or_else(|| "contracts packet_id_from_doc: digest missing prefix".to_string())
    })
    .map_err(PyValueError::new_err)
}

/// Packet-partition span check over `(actor_view, start, end)` triples
/// (mirrors `event_packet.validate_packet_partition` span half without live
/// `ActorVisiblePacket`/`EventEnvelope` objects): every span has a valid
/// actor `0..=3` (bool excluded), nonnegative sequence bounds with
/// `start <= end`, and per-view spans are ordered and mutually exclusive
/// (`previous_end < start`, never overlap). Event-level exclusivity (sequence
/// sets across packets) stays Python as it needs envelope objects. Empty
/// input is valid (mirrors the Python early return). Attached
/// (sub-microsecond integer scan, cheaper than a GIL round-trip).
/// Counter-free deterministic, never wall-clock. Fail-closed: every shape
/// violation is `PyValueError`, never a silent skip.
#[pyfunction]
#[pyo3(signature = (spans,))]
fn validate_packet_spans(
    spans: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>)>,
) -> PyResult<()> {
    let mut per_view: BTreeMap<u8, Vec<(i64, i64)>> = BTreeMap::new();
    for (view_obj, start_obj, end_obj) in spans.iter() {
        let view = plain_int_or_err(view_obj, "actor_view")?;
        if !(0..=3).contains(&view) {
            return Err(PyValueError::new_err(format!(
                "contracts actor_view={view} outside [0, 3]"
            )));
        }
        let start = plain_int_or_err(start_obj, "source_sequence_start")?;
        let end = plain_int_or_err(end_obj, "source_sequence_end")?;
        if start < 0 || end < 0 {
            return Err(PyValueError::new_err(format!(
                "contracts packet span ({start}, {end}) must be nonnegative"
            )));
        }
        if start > end {
            return Err(PyValueError::new_err(format!(
                "contracts packet span start {start} exceeds end {end}"
            )));
        }
        // proof: `view` in 0..=3 (range-checked above), fits `u8`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let view_b: u8 = view as u8;
        per_view.entry(view_b).or_default().push((start, end));
    }
    for (view, mut list) in per_view {
        list.sort();
        let mut previous_end: Option<i64> = None;
        for (start, end) in list {
            if let Some(prev) = previous_end
                && start <= prev
            {
                return Err(PyValueError::new_err(format!(
                    "contracts packets overlap for actor {view} (mutual exclusivity violated)"
                )));
            }
            previous_end = Some(end);
        }
    }
    Ok(())
}
// ---------------------------------------------------------------------------
// Engine-state scalar leaves (prize 3, portable remainder).
//
// Pure integer/literal mirrors of `engines/riichienv/state.py`: the seat-wind
// permutation, the canonical score reorder, the dead-wall-adjusted wall
// count, and the furiten literal. The live `riichienv` package object never
// crosses — callers read attributes/methods Python-side (framework-locked
// lines) and hand detached scalars across. Settlement facts, envelope
// constructors, and `state_digest` stay Python: they build live contract
// dataclasses or walk the package object (attached-construction barrier;
// the digest itself already rides the canon bridge via `of_canonical`).
// Fail-closed: every shape violation is `ValueError`, never a default.
// ---------------------------------------------------------------------------

/// Seat winds for a dealer (`state.py:63-68`): `27 + floormod(dealer+off)`.
/// The oracle validates nothing (any int maps, negatives included) —
/// mirrored exactly via floor-mod, attached (sub-microsecond leaf).
#[pyfunction]
fn seat_winds_for_dealer(dealer: i64) -> (u8, u8, u8, u8) {
    let aligned = dealer.rem_euclid(4);
    // proof: `aligned` in 0..=3 (`rem_euclid(4)`), so `27+…` in 27..=30, fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let w0: u8 = (27 + aligned) as u8;
    // proof: `(aligned+1)%4` in 0..=3, so `27+…` in 27..=30, fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let w1: u8 = (27 + ((aligned + 1) % 4)) as u8;
    // proof: `(aligned+2)%4` in 0..=3, so `27+…` in 27..=30, fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let w2: u8 = (27 + ((aligned + 2) % 4)) as u8;
    // proof: `(aligned+3)%4` in 0..=3, so `27+…` in 27..=30, fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let w3: u8 = (27 + ((aligned + 3) % 4)) as u8;
    (w0, w1, w2, w3)
}

/// Canonical-order scores (`state.py:71-74`): `raw[permutation[seat]]` for
/// seats `0..3`. The oracle indexes blindly (`IndexError` on bad shapes);
/// this leaf fails closed (`ValueError`) instead — valid shapes are
/// identical, attached (four-element gather).
#[pyfunction]
fn engine_scores_canonical_order(scores: Vec<i64>, permutation: Vec<u8>) -> PyResult<Vec<i64>> {
    if permutation.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts engine_scores permutation must hold exactly 4 seats",
        ));
    }
    let mut out: Vec<i64> = Vec::with_capacity(4);
    for seat in 0..4 {
        let src = permutation.get(seat).ok_or_else(|| {
            PyValueError::new_err("contracts engine_scores permutation out of range")
        })?;
        let value = scores.get(*src as usize).ok_or_else(|| {
            PyValueError::new_err("contracts engine_scores seat maps outside scores")
        })?;
        out.push(*value);
    }
    Ok(out)
}

/// Drawable live-wall tiles left (`state.py:77-80`): `max(0, len - 14)`.
/// Attached trivial leaf (`saturating_sub` is exactly the oracle clamp).
#[pyfunction]
fn live_wall_remaining(wall_len: usize) -> usize {
    wall_len.saturating_sub(14)
}

/// Furiten literal (`state.py:83-91`): riichi beats temporary beats none.
/// The caller reads the two package-object flags Python-side and crosses
/// the bools; attached trivial leaf.
#[pyfunction]
fn furiten_of(riichi_declared: bool, missed_agari_doujun: bool) -> &'static str {
    if riichi_declared {
        "riichi"
    } else if missed_agari_doujun {
        "temporary"
    } else {
        "none"
    }
}

/// One delta-path vocabulary row: path elements (names or seat indices),
/// allowed operations, and the value type (`event_vocab.DELTA_PATH_VOCABULARY`).
#[derive(Clone, Copy)]
enum PathElem {
    Name(&'static str),
    Seat(u8),
}

/// Frozen delta-path vocabulary in SPEC order (22 rows; seats expand 0..3).
const DELTA_PATH_VOCABULARY: [(&[PathElem], &[&str], &str); 22] = [
    (&[PathElem::Name("scores")], &["set"], "int4"),
    (
        &[PathElem::Name("scores"), PathElem::Seat(0)],
        &["set", "increment"],
        "int",
    ),
    (
        &[PathElem::Name("scores"), PathElem::Seat(1)],
        &["set", "increment"],
        "int",
    ),
    (
        &[PathElem::Name("scores"), PathElem::Seat(2)],
        &["set", "increment"],
        "int",
    ),
    (
        &[PathElem::Name("scores"), PathElem::Seat(3)],
        &["set", "increment"],
        "int",
    ),
    (&[PathElem::Name("honba")], &["set", "increment"], "int"),
    (
        &[PathElem::Name("riichi_sticks")],
        &["set", "increment"],
        "int",
    ),
    (&[PathElem::Name("round_index")], &["set"], "int"),
    (&[PathElem::Name("dora_indicators")], &["append"], "tile_id"),
    (
        &[PathElem::Name("melds"), PathElem::Name("actor")],
        &["append"],
        "meld",
    ),
    (
        &[PathElem::Name("riichi_states"), PathElem::Seat(0)],
        &["set"],
        "riichi_state",
    ),
    (
        &[PathElem::Name("riichi_states"), PathElem::Seat(1)],
        &["set"],
        "riichi_state",
    ),
    (
        &[PathElem::Name("riichi_states"), PathElem::Seat(2)],
        &["set"],
        "riichi_state",
    ),
    (
        &[PathElem::Name("riichi_states"), PathElem::Seat(3)],
        &["set"],
        "riichi_state",
    ),
    (
        &[PathElem::Name("riichi_states"), PathElem::Name("actor")],
        &["set"],
        "riichi_state",
    ),
    (
        &[PathElem::Name("ippatsu"), PathElem::Seat(0)],
        &["set"],
        "bool",
    ),
    (
        &[PathElem::Name("ippatsu"), PathElem::Seat(1)],
        &["set"],
        "bool",
    ),
    (
        &[PathElem::Name("ippatsu"), PathElem::Seat(2)],
        &["set"],
        "bool",
    ),
    (
        &[PathElem::Name("ippatsu"), PathElem::Seat(3)],
        &["set"],
        "bool",
    ),
    (
        &[PathElem::Name("ippatsu"), PathElem::Name("actor")],
        &["set"],
        "bool",
    ),
    (&[PathElem::Name("kan_count")], &["set", "increment"], "int"),
    (
        &[PathElem::Name("live_wall_tiles_remaining")],
        &["set", "increment"],
        "int",
    ),
];

/// Build one `{"path", "operations", "value_type"}` dict for the vocabulary.
fn vocab_entry<'py>(
    py: Python<'py>,
    path: &[PathElem],
    ops: &[&str],
    ty: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let mut items = Vec::with_capacity(path.len());
    for elem in path {
        items.push(match elem {
            PathElem::Name(s) => s.into_pyobject(py)?.into_any(),
            PathElem::Seat(i) => i.into_pyobject(py)?.into_any(),
        });
    }
    let entry = PyDict::new(py);
    entry.set_item("path", PyTuple::new(py, items)?)?;
    entry.set_item("operations", PyTuple::new(py, ops)?)?;
    entry.set_item("value_type", ty)?;
    Ok(entry)
}

/// Register the `contracts` submodule (mirrors `canon_rng::register`):
/// compute detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "contracts")?;
    sub.add_class::<PyActionTemplate>()?;
    sub.add_class::<PyVisibleMeld>()?;
    sub.add_function(wrap_pyfunction!(is_seat, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_tile_id, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_digest_text, &sub)?)?;
    sub.add_function(wrap_pyfunction!(make_seat, &sub)?)?;
    sub.add_function(wrap_pyfunction!(make_tile_id, &sub)?)?;
    sub.add_function(wrap_pyfunction!(make_action_id, &sub)?)?;
    sub.add_function(wrap_pyfunction!(make_digest_text, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_action_kind, &sub)?)?;
    sub.add_function(wrap_pyfunction!(action_kind_ordinal, &sub)?)?;
    sub.add_function(wrap_pyfunction!(action_kind_ordinals, &sub)?)?;
    sub.add_function(wrap_pyfunction!(tile_type_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_event_kind, &sub)?)?;
    sub.add_function(wrap_pyfunction!(is_visibility, &sub)?)?;
    sub.add_function(wrap_pyfunction!(action_census_count, &sub)?)?;
    sub.add_function(wrap_pyfunction!(action_census_counts, &sub)?)?;
    sub.add_function(wrap_pyfunction!(action_census, &sub)?)?;
    sub.add_function(wrap_pyfunction!(census_template_at, &sub)?)?;
    sub.add_function(wrap_pyfunction!(census_index_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(offset_from_source, &sub)?)?;
    sub.add_function(wrap_pyfunction!(resolve_source, &sub)?)?;
    sub.add_function(wrap_pyfunction!(visible_meld_id, &sub)?)?;
    sub.add_function(wrap_pyfunction!(public_chain_hash, &sub)?)?;
    sub.add_function(wrap_pyfunction!(fold_public_hash, &sub)?)?;
    sub.add_function(wrap_pyfunction!(packet_id_from_doc, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_packet_spans, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seat_winds_for_dealer, &sub)?)?;
    sub.add_function(wrap_pyfunction!(engine_scores_canonical_order, &sub)?)?;
    sub.add_function(wrap_pyfunction!(live_wall_remaining, &sub)?)?;
    sub.add_function(wrap_pyfunction!(furiten_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ctr_block, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ctr_stream_bytes, &sub)?)?;
    sub.add_function(wrap_pyfunction!(resolve_final_ranks, &sub)?)?;
    // SPEC 5.1 frozen vocab (single source; python/hydra2/contracts/rules_canonical.py
    // keeps thin re-exports of these values plus its ContractError validators).
    sub.add("RULES_ID", "tenhou_4p_hanchan_v1")?;
    sub.add("STARTING_POINTS", 25000)?;
    sub.add("RETURN_POINTS", 30000)?;
    sub.add("RED_TILE_IDS", PyTuple::new(py, [16, 52, 88])?)?;
    sub.add("STANDARD_CLOCK_SECONDS", PyTuple::new(py, [5, 10])?)?;
    sub.add("FAST_CLOCK_SECONDS", PyTuple::new(py, [3, 5])?)?;
    sub.add("OKA_POLICIES", PyTuple::new(py, ["none", "half_return"])?)?;
    sub.add("KUITAN_VALUES", PyTuple::new(py, ["allowed", "forbidden"])?)?;
    sub.add(
        "KUIKAE_POLICIES",
        PyTuple::new(py, ["forbidden", "permitted"])?,
    )?;
    sub.add(
        "FURITEN_POLICIES",
        PyTuple::new(
            py,
            ["river_only_permanent_after_riichi_miss_same_goaround_temporary"],
        )?,
    )?;
    sub.add("CHANKAN_POLICIES", PyTuple::new(py, ["permitted"])?)?;
    sub.add("RINSHAN_POLICIES", PyTuple::new(py, ["dead_wall_14"])?)?;
    sub.add(
        "KAN_DORA_REVEAL_POLICIES",
        PyTuple::new(py, ["ankan_immediate_open_delayed"])?,
    )?;
    sub.add("KAN_URA_POLICIES", PyTuple::new(py, ["present"])?)?;
    sub.add(
        "PAO_POLICIES",
        PyTuple::new(py, ["daisangen_daisuishi_tsumo_full_ron_half"])?,
    )?;
    sub.add(
        "YAKUMAN_POLICIES",
        PyTuple::new(py, ["compound_multiple_upgraded_forms_single"])?,
    )?;
    sub.add(
        "KAZOE_POLICIES",
        PyTuple::new(py, ["counted_yakuman_at_13_han"])?,
    )?;
    sub.add(
        "MULTIPLE_RON_POLICIES",
        PyTuple::new(py, ["all_winners_paid_sticks_to_dealer_left"])?,
    )?;
    sub.add(
        "RIICHI_STICK_ALLOCATIONS",
        PyTuple::new(py, ["end_top_take_abort_carry_dealin_exempt"])?,
    )?;
    sub.add(
        "ABORTIVE_DRAW_KINDS",
        PyFrozenSet::new(
            py,
            [
                "kyuushu_kyuuhai",
                "suucha_riichi",
                "sanchahou",
                "suukaikan",
                "suufon_renda",
            ],
        )?,
    )?;
    sub.add(
        "TENHOU_ABORTIVE_DRAWS",
        PyTuple::new(
            py,
            [
                "kyuushu_kyuuhai",
                "suucha_riichi",
                "sanchahou",
                "suukaikan",
                "suufon_renda",
            ],
        )?,
    )?;
    sub.add(
        "ALL_LAST_POLICIES",
        PyTuple::new(py, ["south_west_entry_renchan_extension"])?,
    )?;
    sub.add(
        "AGARI_YAME_POLICIES",
        PyTuple::new(py, ["dealer_top_auto_win_and_tenpai_stop"])?,
    )?;
    sub.add(
        "TOBI_POLICIES",
        PyTuple::new(py, ["negative_points_immediate_end"])?,
    )?;
    sub.add(
        "SUDDEN_DEATH_POLICIES",
        PyTuple::new(py, ["ge_return_points_excluding_sticks_dealer_priority"])?,
    )?;
    sub.add(
        "RANK_TIE_BREAKS",
        PyTuple::new(py, ["east1_seat_wind_order"])?,
    )?;
    sub.add(
        "PLACEMENT_CONVERSION_IDS",
        PyTuple::new(py, ["tenhou_rank_sticks_top_uma_v1"])?,
    )?;
    sub.add(
        "ADAPTER_COMPATIBILITY_STATUSES",
        PyTuple::new(py, ["supported", "unsupported", "qualified"])?,
    )?;
    sub.add(
        "RULES_MANIFEST_PAYLOAD_FIELDS",
        PyTuple::new(
            py,
            [
                "rules_id",
                "source",
                "players",
                "match_length",
                "starting_points",
                "return_points",
                "uma_by_rank",
                "oka_policy",
                "kuitan",
                "red_tile_ids",
                "clocks",
                "kuikae_policy",
                "furiten_policy",
                "chankan_policy",
                "rinshan_policy",
                "kan_dora_reveal_policy",
                "kan_ura_policy",
                "pao_policy",
                "yakuman_policy",
                "kazoe_policy",
                "multiple_ron_policy",
                "riichi_stick_allocation",
                "abortive_draws",
                "nagashi_mangan",
                "bankruptcy_threshold",
                "all_last_policy",
                "agari_yame_policy",
                "tobi_policy",
                "sudden_death_policy",
                "rank_tie_break",
                "placement_conversion_id",
                "adapter_compatibility",
            ],
        )?,
    )?;
    sub.add("SOURCE_EVIDENCE_KEY", "source_evidence")?;
    // SPEC 6.1 kind tables (single source; python/hydra2/contracts/action_kinds.py
    // keeps thin re-exports plus its Literal types, validators, and metadata logic).
    // Ordinals reuse the frozen ACTION_KIND_TABLE above (insertion order == ordinal order).
    {
        let ordinals = PyDict::new(py);
        for (name, ordinal) in ACTION_KIND_TABLE {
            ordinals.set_item(name, ordinal)?;
        }
        sub.add("ACTION_KIND_ORDINALS", ordinals)?;
    }
    sub.add(
        "ACTION_KINDS",
        PyTuple::new(py, ACTION_KIND_TABLE.map(|(name, _)| name))?,
    )?;
    // D-WP02C-1 engine-phase gating per action kind.
    const ACTION_PHASES_TABLE: [(&str, &[&str]); 13] = [
        ("pass", &["discard_response", "kan_response"]),
        ("discard", &["draw_decision"]),
        ("tsumogiri", &["draw_decision"]),
        ("riichi_discard", &["draw_decision"]),
        ("chi", &["discard_response"]),
        ("pon", &["discard_response"]),
        ("daiminkan", &["discard_response"]),
        ("ankan", &["draw_decision"]),
        ("kakan", &["draw_decision"]),
        ("ron", &["discard_response", "kan_response"]),
        ("tsumo", &["draw_decision"]),
        ("abort_nine_terminals", &["draw_decision"]),
        (
            "accept_abortive_draw",
            &["discard_response", "kan_response"],
        ),
    ];
    {
        let phases = PyDict::new(py);
        for (kind, members) in ACTION_PHASES_TABLE {
            phases.set_item(kind, PyFrozenSet::new(py, members)?)?;
        }
        sub.add("ACTION_PHASES", phases)?;
    }
    sub.add(
        "CLAIM_KINDS",
        PyFrozenSet::new(py, ["chi", "pon", "daiminkan"])?,
    )?;
    // SPEC 7.1 + SPEC 8 vocab tables (single source; event_vocab.py /
    // observation_types.py keep thin re-exports plus Literals/validators).
    sub.add("MELD_KINDS", PyTuple::new(py, MELD_KINDS)?)?;
    sub.add("EVENT_KINDS", PyTuple::new(py, EVENT_KINDS)?)?;
    sub.add("VISIBILITIES", PyTuple::new(py, VISIBILITIES)?)?;
    sub.add("DORA_SENTINEL", -1)?;
    sub.add("DORA_SHAPE", PyTuple::new(py, [5])?)?;
    sub.add(
        "PHASES",
        PyTuple::new(
            py,
            [
                "round_start",
                "draw_decision",
                "discard_response",
                "kan_response",
                "round_end",
                "game_end",
            ],
        )?,
    )?;
    sub.add(
        "DELTA_OPERATIONS",
        PyTuple::new(py, ["set", "append", "increment"])?,
    )?;
    sub.add(
        "PAYLOAD_SCALAR_FIELDS",
        PyTuple::new(
            py,
            [
                "tile",
                "action_id",
                "source_seat",
                "round_index",
                "scores",
                "reason",
            ],
        )?,
    )?;
    sub.add(
        "PAYLOAD_TUPLE_FIELDS",
        PyTuple::new(
            py,
            [
                "consumed_tiles",
                "offered_action_ids",
                "accepted_action_ids",
            ],
        )?,
    )?;
    {
        let vocab: Vec<Bound<'_, PyDict>> = DELTA_PATH_VOCABULARY
            .iter()
            .map(|(path, ops, ty)| vocab_entry(py, path, ops, ty))
            .collect::<PyResult<_>>()?;
        sub.add("DELTA_PATH_VOCABULARY", PyTuple::new(py, vocab)?)?;
    }
    crate::action_artifact::register(&sub)?;
    crate::event_schema::register(&sub)?;
    crate::observation_schema::register(&sub)?;
    crate::validate::register(&sub)?;
    crate::walls::register(&sub)?;
    crate::rc_digest::register(&sub)?;
    crate::rc_require::register(&sub)?;
    crate::belief_leaves::register(&sub)?;
    crate::eval_leaves::register(&sub)?;
    crate::rc_parse::register(&sub)?;
    crate::rc_parse_aux::register(&sub)?;
    crate::rc_resume::register(&sub)?;
    crate::rc_sections::register(&sub)?;
    crate::action_model::register(&sub)?;
    crate::randomness::register(&sub)?;
    crate::utility::register(&sub)?;
    crate::obs_actor::register(&sub)?;
    crate::obs_assembly::register(&sub)?;
    crate::belief_kernel::register(&sub)?;
    crate::belief_natural::register(&sub)?;
    crate::eval_duplicate::register(&sub)?;
    crate::eval_schedule::register(&sub)?;
    crate::event_packet::register(&sub)?;
    crate::loop_state::register(&sub)?;
    crate::training_leaves::register(&sub)?;
    crate::oracle_guard::register(&sub)?;
    crate::oracle_targets::register(&sub)?;
    crate::promotion::register(&sub)?;
    crate::sampled_kernel::register(&sub)?;
    crate::common_errors::register(&sub)?;
    crate::oracle_join::register(&sub)?;
    crate::rc_cross_validate::register(&sub)?;
    crate::rc_resume_gates::register(&sub)?;
    crate::common_roots::register(&sub)?;
    crate::distill_leaves::register(&sub)?;
    crate::record_leaves::register(&sub)?;
    crate::script_leaves::register(&sub)?;
    crate::tracking_leaves::register(&sub)?;
    m.add_submodule(&sub)?;
    Ok(())
}
