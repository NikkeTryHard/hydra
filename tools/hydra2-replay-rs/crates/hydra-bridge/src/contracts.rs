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
//! Future single-cdylib tree: registers as the `contracts` submodule
//! (`hydra2_replay_rs.contracts` today, `hydra_bridge._native.contracts` once
//! the Phase-6 maturin `module-name` cutover lands), mirroring
//! `canon_rng::register`. The legacy `hydra2_replay_rs` entry is untouched.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyModule};

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
    Ok(v as u8)
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
    Ok(v as u8)
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
    Ok(v as u32)
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
    Ok((v / 4) as u8)
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
    EVENT_KINDS.iter().any(|name| *name == kind)
}

/// True iff the literal is a frozen visibility.
#[pyfunction]
fn is_visibility(visibility: &str) -> bool {
    VISIBILITIES.iter().any(|name| *name == visibility)
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
            Ok(Some(v as u8))
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
    let action_kind = hydra_feed::census::ActionKind::from_name(kind).ok_or_else(|| {
        PyValueError::new_err(format!("contracts unknown action kind {kind:?}"))
    })?;
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
        arr[i] = v as u8;
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
            Some(v as i8)
        }
    };
    // riichi/meld flags are kind-derived, exactly as the Python record
    // constructor enforces: only genuinely present templates can hit.
    let probe = hydra_feed::census::ActionTemplate {
        kind: action_kind,
        tile,
        called_tile,
        consumed: arr,
        consumed_len: consumed.len() as u8,
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
    let offset = (cursor % CTR_BLOCK) as usize;
    let end = offset
        .checked_add(count)
        .ok_or_else(|| PyValueError::new_err("contracts ctr_stream_bytes: window overflows"))?;
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
            acc.get(offset..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| {
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
fn resolve_final_ranks(
    py: Python<'_>,
    scores: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<u8>> {
    if scores.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts resolve_final_ranks: scores must be exactly 4 entries",
        ));
    }
    let mut arr = [0i64; 4];
    for (i, entry) in scores.iter().enumerate() {
        let v = plain_int_or_err(entry, "final_scores")?;
        if v < SCORE_MIN || v > SCORE_MAX {
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

/// Register the `contracts` submodule (mirrors `canon_rng::register`):
/// compute detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "contracts")?;
    sub.add_class::<PyActionTemplate>()?;
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
    sub.add_function(wrap_pyfunction!(public_chain_hash, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ctr_block, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ctr_stream_bytes, &sub)?)?;
    sub.add_function(wrap_pyfunction!(resolve_final_ranks, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}
