//! obs_actor: actor-observation pure field leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-feed` ONLY (the `validate` owner for the dora sentinel
//! — same edge as `contracts.rs`, no new dependency). FORBIDS: live-object
//! orchestration (the `ActorObservation` dataclass, `ObservationBuilder`,
//! `visible_history` envelope filtering, `compute_observation_hash`, and
//! `make_actor_observation` stay Python — attached construction over validated
//! fields, hash-stamping stays Python), canon/sha re-implementation (the feed
//! `canon`/`digest` owners are untouched; the projection below only drops a
//! key), and wall-clock/RNG.
//!
//! TABLE ported here (frozen values + pure compute over plain sequences and
//! plain mappings, never live objects):
//! - `check_dora_indicators` <- the G3 dora block
//!   (`python/hydra2/contracts/observation_actor.py:272-301`): shape, per-slot
//!   int/tile checks, revealed-prefix contiguity. Gate-only return (`()`);
//!   Python stores the validated original tuple, so storage (including the
//!   oracle's int-subclass values) is bit-identical.
//! - `observation_identity_projection` <- the hash-doc projection half of
//!   `observation_identity_document` (`observation_actor.py:396-400`): copy a
//!   plain `to_json` mapping minus `observation_hash` (absent key is fine —
//!   the oracle pops with a default). The live `ActorObservation` never
//!   crosses; the caller serializes Python-side and hands the plain dict
//!   across (the `PacketSuccessor` precedent).
//!
//! LOGIC staying Python (`observation_actor.py`): the `ActorObservation`
//! dataclass + `__post_init__` firewall root (the dora leaf is called through
//! a thin `_check_dora_indicators` translator that maps `ValueError`/
//! `TypeError` to `ContractError` with byte-identical oracle text),
//! `to_json`, `compute_observation_hash` / `make_actor_observation`
//! (hash-stamping stays Python), the `legal_mask` gate (same-class pure
//! validator, deferred to a later wave — sketch: a `check_legal_mask` gate
//! shaped exactly like the dora gate over staged `Option<bool>` slots,
//! mirroring the `break`/`else` so junk past the first `True` keeps exact
//! oracle storage), and every `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `WIND_TILE|FURIETEN|RIICHI_STATES` over
//! `crates/feed/src` + `crates/search/src` + `crates/bridge/src` hits nothing
//! (those vocabularies stay owned by `observation_types.py`, explicitly out of
//! scope); `DORA_SENTINEL`/`DORA_SHAPE` already live on the shared submodule
//! (`contracts.rs:1263-1264`), so no const is restated here except the
//! five-slot length, which no crate owns as a named const (feed owns only the
//! `-1` sentinel at `feed/src/validate.rs:68-69`, referenced below, never
//! duplicated).
//!
//! Shape per fn: attached staging (str/bytes/sequence discrimination plus
//! `repr`/`int` views over the live Python API, so `{value!r}` renders
//! byte-exact) -> ONE `py.detach(|| ...)` over owned plain data with zero
//! Python API inside for the dora compute (per `contracts.rs:448-456`,
//! `rc_require.rs:827-843`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-125`). The projection runs attached end-to-end: a shallow
//! dict copy costs less than a GIL round-trip for a detach (the rank-4/5
//! census-leaf precedent at `contracts.rs:10-14`). Bool is excluded before
//! every int check (per `contracts.rs:69-74`); huge ints past `i64` keep the
//! staged repr with `as_i64` unset (per `rc_require.rs:86-104`); sequence
//! items cross via `cast::<PySequence>()` (the `cast` half of the
//! `rc_parse_aux.rs:490-496` list/tuple split, generalized to every
//! `collections.abc.Sequence` the oracle accepts); dict building uses
//! `PyDict::new` plus `copy`/`contains`/`del_item` (the `new` half per
//! `contracts.rs:1227-1231`, same `PyDict` surface). Fail-closed: every shape
//! violation is `ValueError`, never a default.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod obs_actor;` in `lib.rs` plus
//! `crate::obs_actor::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1300`).

// Precedent `use pyo3::exceptions::PyValueError;` -> contracts.rs:50 (via
// `pyo3::exceptions`); `prelude::*` -> contracts.rs:52; `PyModule` et al
// imports -> contracts.rs:53.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt, PyModule, PySequence, PyString};

// Sentinel for an unrevealed dora slot: reference the feed owner
// (`feed/src/validate.rs:68-69`, itself mirroring
// `observation_types.py:40`), never a second literal.
use hydra_feed::validate::DORA_SENTINEL;

/// Frozen indicator width (`observation_types.py:42` `DORA_SHAPE`, bridged at
/// `contracts.rs:1264` as `(5,)`): no crate owns the length as a named const,
/// so it is pinned here once with both oracle sites cited above.
const DORA_LEN: usize = 5;

/// Tile-id domain top (`contracts.rs:129-137` `make_tile_id` gate, mirrored
/// below so the `invalid tile` composition stays byte-identical without a
/// Python round-trip inside the detach).
const TILE_ID_MAX: i64 = 135;

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Owned plain-data view of one dora slot. Bool is excluded first (it
/// subclasses `int`), exactly like `contracts.rs:69-74`; a huge int past
/// `i64` keeps `is_int` true with `as_i64` unset (mirrors
/// `rc_require.rs:86-104`), so the detach still renders the oracle's
/// `invalid tile <repr>: contracts tile_id must be a plain int` composition.
struct DoraSlot {
    repr: String,
    is_int: bool,
    as_i64: Option<i64>,
}

/// Attached staging for one dora slot (GIL-held by construction; the pure
/// core below only reads the owned view).
fn stage_dora_slot(obj: &Bound<'_, PyAny>) -> PyResult<DoraSlot> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(DoraSlot {
            repr,
            is_int: false,
            as_i64: None,
        });
    }
    Ok(DoraSlot {
        repr,
        is_int: true,
        as_i64: obj.extract::<i64>().ok(),
    })
}

/// Pure shape-error composition shared by staging and the core (TOTAL at every
/// layer per the Wave-5 lesson: the core re-checks length instead of trusting
/// staging to pre-filter).
fn dora_shape_error(got: &str) -> String {
    format!(
        "dora_indicators must hold exactly {DORA_LEN} entries, got {got}; \
         the shape is fixed and NEVER padded"
    )
}

/// Pure dora gate over staged slots (mirrors
/// `observation_actor.py:272-301` in order: shape, per-slot int/tile checks
/// left to right, then revealed-prefix contiguity). No Python API inside
/// (detach-safe); failures are oracle-text `String`s.
fn check_dora_slots(slots: &[DoraSlot]) -> Result<(), String> {
    if slots.len() != DORA_LEN {
        return Err(dora_shape_error(&slots.len().to_string()));
    }
    for (index, slot) in slots.iter().enumerate() {
        if !slot.is_int {
            return Err(format!("dora_indicators[{index}] must be an int"));
        }
        match slot.as_i64 {
            // Huge int: `make_tile_id` would fail its plain-int extraction
            // (`contracts.rs:77-84`), so the inner text is the generic gate.
            None => {
                return Err(format!(
                    "dora_indicators[{index}] invalid tile {r}: \
                     contracts tile_id must be a plain int (bool excluded)",
                    r = slot.repr
                ));
            }
            Some(v) => {
                if v != DORA_SENTINEL && !(0i64..=TILE_ID_MAX).contains(&v) {
                    return Err(format!(
                        "dora_indicators[{index}] invalid tile {v}: \
                         contracts tile_id={v} outside [0, 135]"
                    ));
                }
            }
        }
    }
    // Contiguity: revealed indicators must occupy the `0..k` prefix and
    // sentinels the `k..5` tail (a sentinel before a revealed value fails).
    // The oracle's second disjunct (`DORA_SENTINEL in revealed`) is dead by
    // construction; only the prefix comparison is mirrored.
    let mut seen_sentinel = false;
    for slot in slots {
        if let Some(v) = slot.as_i64 {
            if v == DORA_SENTINEL {
                seen_sentinel = true;
            } else if seen_sentinel {
                return Err("revealed dora indicators must be contiguous from index 0; \
                     sentinels fill the unrevealed tail"
                    .to_string());
            }
        }
    }
    Ok(())
}

/// Dora contiguity/shape gate over plain sequences (mirrors
/// `ActorObservation.__post_init__` without live tile objects — callers pass
/// the raw indicator sequence, never an `ActorObservation`). Gate-only return
/// (`()`); Python stores the validated original tuple. Compute runs detached
/// with zero Python API inside; every rejection is `PyValueError` with
/// byte-identical oracle text, never a default. Counter-free deterministic,
/// never wall-clock.
#[pyfunction]
#[pyo3(signature = (indicators,))]
fn check_dora_indicators(indicators: Bound<'_, PyAny>) -> PyResult<()> {
    // Attached staging, in oracle order: `str`/`bytes` are rejected with
    // their length echoed (they ARE `Sequence`s, so `got` is numeric —
    // `chars().count()` keeps Python `len(str)` semantics for non-ASCII).
    if let Ok(text) = indicators.cast::<PyString>() {
        let n = text.to_str()?.chars().count();
        return Err(PyValueError::new_err(dora_shape_error(&n.to_string())));
    }
    if let Ok(blob) = indicators.cast::<PyBytes>() {
        let n = blob.as_bytes().len();
        return Err(PyValueError::new_err(dora_shape_error(&n.to_string())));
    }
    let seq = match indicators.cast::<PySequence>() {
        Ok(seq) => seq,
        Err(_) => {
            return Err(PyValueError::new_err(dora_shape_error("non-sequence")));
        }
    };
    let n = seq.len().map_err(|e| {
        PyValueError::new_err(format!("contracts dora_indicators length unreadable: {e}"))
    })?;
    if n != DORA_LEN {
        return Err(PyValueError::new_err(dora_shape_error(&n.to_string())));
    }
    let mut staged = Vec::with_capacity(n);
    for index in 0..n {
        let item = seq.get_item(index).map_err(|e| {
            PyValueError::new_err(format!(
                "contracts dora_indicators[{index}] unreadable: {e}"
            ))
        })?;
        staged.push(stage_dora_slot(&item)?);
    }
    let py = indicators.py();
    py.detach(|| check_dora_slots(&staged))
        .map_err(PyValueError::new_err)?;
    Ok(())
}

/// Identity-doc projection over a plain JSON mapping (mirrors
/// `observation_identity_document` without the live `ActorObservation`): the
/// caller serializes Python-side via `to_json` and hands the plain dict
/// across; Rust returns a shallow copy minus `observation_hash` (an absent
/// key is fine — the oracle pops with a default). Key order and value
/// identities are preserved, so downstream canonical bytes are bit-identical.
/// Attached end-to-end (dict copy); non-mapping input fails closed at the
/// extractor (`TypeError`). Counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (doc,))]
fn observation_identity_projection<'py>(doc: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let out = doc.copy()?;
    if out.contains("observation_hash")? {
        out.del_item("observation_hash")?;
    }
    Ok(out)
}

/// Register the actor-observation leaves on the shared `contracts` submodule
/// (mirrors `rc_require.rs:849-862`): compute detached, wrap attached; single
/// cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(check_dora_indicators, sub)?)?;
    sub.add_function(wrap_pyfunction!(observation_identity_projection, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_slot(v: i64) -> DoraSlot {
        DoraSlot {
            repr: v.to_string(),
            is_int: true,
            as_i64: Some(v),
        }
    }

    fn non_int_slot(repr: &str) -> DoraSlot {
        DoraSlot {
            repr: repr.to_owned(),
            is_int: false,
            as_i64: None,
        }
    }

    fn big_slot(repr: &str) -> DoraSlot {
        DoraSlot {
            repr: repr.to_owned(),
            is_int: true,
            as_i64: None,
        }
    }

    #[test]
    fn dora_accepts_full_partial_and_empty_reveals() {
        let full = [0, 4, 8, 12, 16].map(int_slot);
        assert_eq!(check_dora_slots(&full), Ok(()));
        let partial = [4, -1, -1, -1, -1].map(int_slot);
        assert_eq!(check_dora_slots(&partial), Ok(()));
        let empty = [-1, -1, -1, -1, -1].map(int_slot);
        assert_eq!(check_dora_slots(&empty), Ok(()));
    }

    #[test]
    fn dora_shape_gates_match_head_oracle_text() {
        // HEAD `observation_actor.py:278-282`: `(4,)` and `(5+1,)` reject with
        // the NEVER-padded text (mirrors `test_events_obs_wp02d.py:711-714`).
        let four = [0, 4, 8, 12].map(int_slot);
        assert_eq!(
            check_dora_slots(&four),
            Err("dora_indicators must hold exactly 5 entries, got 4; \
                 the shape is fixed and NEVER padded"
                .to_string())
        );
        let six = [0, 4, 8, 12, 16, 20].map(int_slot);
        assert_eq!(
            check_dora_slots(&six),
            Err("dora_indicators must hold exactly 5 entries, got 6; \
                 the shape is fixed and NEVER padded"
                .to_string())
        );
    }

    #[test]
    fn dora_gap_rejects_with_contiguity_text() {
        // HEAD `observation_actor.py:296-300` (mirrors
        // `test_non_contiguous_sentinel_gap_is_rejected`: `(-1, 4, -1, -1, -1)`).
        let gap = [-1, 4, -1, -1, -1].map(int_slot);
        assert_eq!(
            check_dora_slots(&gap),
            Err("revealed dora indicators must be contiguous from index 0; \
                 sentinels fill the unrevealed tail"
                .to_string())
        );
        let hole = [0, -1, 8, -1, -1].map(int_slot);
        assert_eq!(
            check_dora_slots(&hole),
            Err("revealed dora indicators must be contiguous from index 0; \
                 sentinels fill the unrevealed tail"
                .to_string())
        );
    }

    #[test]
    fn dora_slot_gates_match_head_oracle_text() {
        // HEAD `observation_actor.py:285-293`: non-int and out-of-range tiles
        // fail per index with the composed text.
        let non_int = [
            int_slot(0),
            int_slot(4),
            non_int_slot("'x'"),
            int_slot(-1),
            int_slot(-1),
        ];
        assert_eq!(
            check_dora_slots(&non_int),
            Err("dora_indicators[2] must be an int".to_string())
        );
        let high = [
            int_slot(136),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
        ];
        assert_eq!(
            check_dora_slots(&high),
            Err("dora_indicators[0] invalid tile 136: \
                 contracts tile_id=136 outside [0, 135]"
                .to_string())
        );
        let neg = [
            int_slot(-2),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
        ];
        assert_eq!(
            check_dora_slots(&neg),
            Err("dora_indicators[0] invalid tile -2: \
                 contracts tile_id=-2 outside [0, 135]"
                .to_string())
        );
        // Huge int past `i64` (2**100): passes the int check, then fails the
        // plain-int extraction inside `make_tile_id` (`contracts.rs:77-84`).
        let huge = "1267650600228229401496703205376";
        let big = [
            big_slot(huge),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
            int_slot(-1),
        ];
        assert_eq!(
            check_dora_slots(&big),
            Err(format!(
                "dora_indicators[0] invalid tile {huge}: \
                 contracts tile_id must be a plain int (bool excluded)"
            ))
        );
    }

    #[test]
    fn identity_projection_strips_only_the_hash_key() {
        Python::initialize();
        Python::attach(|py| {
            let doc = PyDict::new(py);
            assert!(doc.set_item("game_id", "g").is_ok());
            assert!(doc.set_item("kan_count", 1).is_ok());
            assert!(doc.set_item("observation_hash", "sha256:abc").is_ok());
            let projected = observation_identity_projection(doc);
            assert!(projected.is_ok());
            if let Ok(out) = projected {
                assert!(
                    out.get_item("game_id")
                        .map(|v| v.is_some())
                        .unwrap_or(false)
                );
                assert!(
                    out.get_item("kan_count")
                        .map(|v| v.is_some())
                        .unwrap_or(false)
                );
                assert!(
                    out.get_item("observation_hash")
                        .map(|v| v.is_none())
                        .unwrap_or(false)
                );
            }
            // Absent key is fine (the oracle pops with a default).
            let bare = PyDict::new(py);
            assert!(bare.set_item("game_id", "g").is_ok());
            let bare_out = observation_identity_projection(bare);
            assert!(bare_out.is_ok());
            if let Ok(out) = bare_out {
                assert!(
                    out.get_item("game_id")
                        .map(|v| v.is_some())
                        .unwrap_or(false)
                );
            }
        });
    }
}
