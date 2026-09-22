//! obs_assembly: per-seat assembly frozen tables + stateless pure leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (pure argument validation + frozen tables — same edge as
//! `rc_require.rs`, no new dependency). FORBIDS: canon re-printing (the feed
//! `canon` owner is the single printer), wall-clock/urandom seeding, live
//! `EventEnvelope`/`ActorObservation` construction, and `ContractError`
//! shaping (the bridge raises `ValueError`/`TypeError`; thin Python
//! translators map to `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/contracts/observation_assembly.py`
//! (worktree tag at port time: `observation_assembly.py#C716`):
//! - `HISTORY_EVENT_CAP` <- `:106`: 256, mirrors `HISTORY_BUCKET_LENGTHS[-1]`
//!   (models/schema, the model cap; pinned equal by test).
//! - `_PUBLIC_SNAPSHOT_FIELDS` <- `:114-128`: the 13-field closed
//!   public-snapshot vocabulary in oracle order.
//! - `obs_dora_append_slot` <- `:293-301` len gate: `len >= DORA_SHAPE[0]`
//!   fails with the oracle overflow text; the caller `int()`-coerces the tile
//!   Python-side first (oracle coercion preserved), the bridge gates the
//!   coerced pair.
//! - `obs_public_unknown` <- `:361-363`: `sorted(set(supplied) - set(FIELDS))`
//!   fails with the oracle unknown-fields text.
//! - `obs_public_missing` <- `:440-442`: `[f for f in FIELDS if f not in
//!   present]` (field order, never sorted) fails with the oracle incomplete
//!   text.
//!
//! LOGIC staying Python: `VisibilityValidator` + `VISIBILITY_VALIDATOR`
//! (live `EventEnvelope`/`ActorObservation` boundary checks), the whole
//! `ObservationBuilder` (mutable per-seat caches `_histories`/`_dora`/rivers/
//! melds/concealment maps, `append_visible` routing, `_apply_public_effects`
//! meld/riichi/kan orchestration, `_upgrade_kakan` pon search,
//! `set_concealed_hand`/`set_actor_state`/`update_public_state` field
//! validators, `build` assembly over live observations), `_CALL_MELD_KINDS`
//! (same `{"chi", "pon", "daiminkan"}` vocabulary already owned as
//! `CLAIM_KINDS` at `contracts.rs:1257` — membership checks are identical,
//! the tuple stays Python to avoid churn), the dora-indicator padding fold
//! (`tuple(dora) + (SENTINEL,) * (5 - len)`, one-line projection), the
//! legal-mask gates, sequence-monotonicity, and every `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `HISTORY_EVENT_CAP|PUBLIC_SNAPSHOT_FIELDS|
//! obs_dora_append|obs_public_unknown|obs_public_missing` over
//! `crates/feed/src` + `crates/search/src` + `crates/shard/src` hits only the
//! unrelated geometry guard `SHARD_MAX_HISTORY = 256` (`shard/src/writer.rs`)
//! and the `dora: [i8; 5]` ledger plane (`feed/src/ledger.rs`) — array
//! lengths, never the named contracts tables gated here.
//!
//! Shape per fn: attached staging (bool-excluded ints via the live Python API,
//! string vecs via `Vec<String>` extraction) -> ONE `py.detach(|| ...)` over
//! owned plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `rc_require.rs:827-844`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Consts via `sub.add` (per `contracts.rs:1153-1156`);
//! frozen tuples via `PyTuple::new` (per `contracts.rs:1281-1287`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_require.rs:846-862`); no new entry
//! point. MAIN wiring: `pub mod obs_assembly;` in `lib.rs` plus
//! `crate::obs_assembly::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1300`).

// Precedent `use pyo3::prelude::*;` -> contracts.rs:52.
use pyo3::prelude::*;
// Precedent `PyAny` + `PyBool` + `PyModule` + `PyTuple` imports -> contracts.rs:53.
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyBool, PyModule, PyTuple};

/// Max visible-history events a row may carry (`observation_assembly.py:106`).
const HISTORY_EVENT_CAP: usize = 256;

/// Dora-indicator slot count (`DORA_SHAPE[0]`, `contracts.rs:1264`; the
/// overflow text renders this value — restated here per the oracle line
/// cited above since no Rust crate owns a named const for it).
const OBS_DORA_SLOTS: usize = 5;

/// Closed public-snapshot vocabulary (`observation_assembly.py:114-128`) in
/// oracle order (missing reports preserve this order; never sorted).
const PUBLIC_SNAPSHOT_FIELDS: [&str; 13] = [
    "decision_id",
    "round_index",
    "round_wind",
    "hand_number",
    "seat_winds",
    "honba",
    "riichi_sticks",
    "dealer",
    "scores",
    "turn_actor",
    "phase",
    "live_wall_tiles_remaining",
    "ippatsu_active",
];

/// Plain-int view of a Python value: `bool` is excluded first (it subclasses
/// `int`), non-ints and out-of-`i64` values map to `None` (makers raise
/// instead). Restates `contracts.rs:69-74` per use.
fn plain_int_value(obj: &Bound<'_, PyAny>) -> Option<i64> {
    if obj.is_instance_of::<PyBool>() {
        return None;
    }
    obj.extract::<i64>().ok()
}

/// Pure dora-append gate (`observation_assembly.py:293-301`): the caller
/// `int()`-coerces the tile Python-side (oracle coercion preserved, including
/// `bool -> 0/1`); the bridge gates the coerced pair. No Python API inside
/// (detach-safe); failures are oracle-text `String`s.
fn check_dora_slot(dora_len: Option<i64>, revealed: Option<i64>) -> Result<usize, String> {
    let current_i: i64 = match dora_len {
        Some(v) if v >= 0 => v,
        Some(v) => {
            return Err(format!("contracts dora_len={v} must be a non-negative int"));
        }
        None => {
            return Err("contracts dora_len must be a plain int (bool excluded)".to_owned());
        }
    };
    // proof: `current_i >= 0` (guarded arm), fits `usize` on 64-bit.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let current: usize = current_i as usize;
    let tile = match revealed {
        Some(v) if (0..=135).contains(&v) => v,
        Some(v) => return Err(format!("contracts tile_id={v} outside [0, 135]")),
        None => {
            return Err("contracts tile_id must be a plain int (bool excluded)".to_owned());
        }
    };
    if current >= OBS_DORA_SLOTS {
        return Err(format!(
            "dora indicator {tile} exceeds the fixed {OBS_DORA_SLOTS}-slot shape; never padded or truncated"
        ));
    }
    Ok(current + 1)
}

/// Python-`repr`-exact rendering of a string list (`['a', 'b']` with single
/// quotes — the oracle f-strings render `list[str]` this way). Field names are
/// identifiers so no escaping domain exists. Precedent single-quote rendering
/// -> `rc_require.rs:385-392`.
fn py_str_list_repr(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

/// Pure unknown-snapshot gate (`observation_assembly.py:361-363`):
/// `sorted(set(supplied) - set(FIELDS))`; empty accepts. No Python API inside.
fn check_public_unknown(supplied: &[String]) -> Result<(), String> {
    let mut unknown_set = std::collections::BTreeSet::new();
    for key in supplied {
        if !PUBLIC_SNAPSHOT_FIELDS.contains(&key.as_str()) {
            unknown_set.insert(key.clone());
        }
    }
    if unknown_set.is_empty() {
        Ok(())
    } else {
        let unknown: Vec<String> = unknown_set.into_iter().collect();
        Err(format!(
            "unknown public snapshot fields: {}",
            py_str_list_repr(&unknown)
        ))
    }
}

/// Pure missing-snapshot gate (`observation_assembly.py:440-442`): fields in
/// oracle order (never sorted); empty accepts. No Python API inside.
fn check_public_missing(present: &[String]) -> Result<(), String> {
    let mut missing: Vec<String> = Vec::new();
    for field in PUBLIC_SNAPSHOT_FIELDS {
        if !present.iter().any(|p| p == field) {
            missing.push(field.to_owned());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "public snapshot incomplete; missing {}",
            py_str_list_repr(&missing)
        ))
    }
}

/// Dora-append len gate over the caller-coerced `(dora_len, revealed_tile)`
/// pair (mirrors `observation_assembly.py:296-301` without the live builder —
/// callers pass scalars). Compute runs detached; pure function of its inputs.
/// Out-of-slot appends fail closed with the oracle text, never padded or
/// truncated.
#[pyfunction]
#[pyo3(signature = (dora_len, revealed))]
fn obs_dora_append_slot(
    py: Python<'_>,
    dora_len: Bound<'_, PyAny>,
    revealed: Bound<'_, PyAny>,
) -> PyResult<usize> {
    let staged_len = plain_int_value(&dora_len);
    let staged_tile = plain_int_value(&revealed);
    py.detach(|| check_dora_slot(staged_len, staged_tile))
        .map_err(PyValueError::new_err)
}

/// Unknown public-snapshot gate over the supplied key list (mirrors
/// `observation_assembly.py:361-363` without the live `**snapshot` dict —
/// callers pass `list(snapshot)`). Compute runs detached; unknown keys fail
/// closed with the oracle text.
#[pyfunction]
#[pyo3(signature = (supplied,))]
fn obs_public_unknown(py: Python<'_>, supplied: Vec<String>) -> PyResult<()> {
    py.detach(|| check_public_unknown(&supplied))
        .map_err(PyValueError::new_err)
}

/// Missing public-snapshot gate over the present key list (mirrors
/// `observation_assembly.py:440-442` without the live `self._public` dict —
/// callers pass `list(self._public)`). Compute runs detached; gaps fail
/// closed with the oracle text in field order.
#[pyfunction]
#[pyo3(signature = (present,))]
fn obs_public_missing(py: Python<'_>, present: Vec<String>) -> PyResult<()> {
    py.detach(|| check_public_missing(&present))
        .map_err(PyValueError::new_err)
}

/// Register the assembly leaves on the shared `contracts` submodule (mirrors
/// `rc_require.rs:846-862`): compute detached, wrap attached; single cdylib,
/// no new entry point.
///
/// Precedents: `register` signature shape -> contracts.rs:1114;
/// `let py = sub.py()` -> contracts.rs:1115;
/// `wrap_pyfunction!(f, sub)` (no `&`) -> belief_leaves.rs:194-196;
/// `sub.add(<int>)` -> contracts.rs:1154;
/// `PyTuple::new(py, [...])` for frozen tuples -> contracts.rs:1156.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(obs_dora_append_slot, sub)?)?;
    sub.add_function(wrap_pyfunction!(obs_public_unknown, sub)?)?;
    sub.add_function(wrap_pyfunction!(obs_public_missing, sub)?)?;
    sub.add("HISTORY_EVENT_CAP", HISTORY_EVENT_CAP)?;
    sub.add(
        "_PUBLIC_SNAPSHOT_FIELDS",
        PyTuple::new(py, PUBLIC_SNAPSHOT_FIELDS)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_tables_match_oracle_literals() {
        // Byte-level pins of the oracle tables (`observation_assembly.py:106`,
        // `:114-128`); any drift fails here first.
        assert_eq!(HISTORY_EVENT_CAP, 256);
        assert_eq!(OBS_DORA_SLOTS, 5);
        assert_eq!(
            PUBLIC_SNAPSHOT_FIELDS,
            [
                "decision_id",
                "round_index",
                "round_wind",
                "hand_number",
                "seat_winds",
                "honba",
                "riichi_sticks",
                "dealer",
                "scores",
                "turn_actor",
                "phase",
                "live_wall_tiles_remaining",
                "ippatsu_active",
            ]
        );
    }

    #[test]
    fn dora_slot_matches_oracle_gate() {
        // Hand-derived from `observation_assembly.py:296-301` with
        // `DORA_SHAPE[0] == 5`: slots 0..=4 accept and return the next length,
        // slot 5 rejects with the oracle overflow text.
        assert_eq!(check_dora_slot(Some(0), Some(10)), Ok(1));
        assert_eq!(check_dora_slot(Some(4), Some(10)), Ok(5));
        assert_eq!(
            check_dora_slot(Some(5), Some(10)),
            Err(
                "dora indicator 10 exceeds the fixed 5-slot shape; never padded or truncated"
                    .to_owned()
            )
        );
        // TOTAL gates (no oracle text — internal-state misuse fails closed):
        // non-int len, negative len, non-int tile, out-of-range tile.
        assert_eq!(
            check_dora_slot(None, Some(10)),
            Err("contracts dora_len must be a plain int (bool excluded)".to_owned())
        );
        assert_eq!(
            check_dora_slot(Some(-1), Some(10)),
            Err("contracts dora_len=-1 must be a non-negative int".to_owned())
        );
        assert_eq!(
            check_dora_slot(Some(0), None),
            Err("contracts tile_id must be a plain int (bool excluded)".to_owned())
        );
        assert_eq!(
            check_dora_slot(Some(0), Some(136)),
            Err("contracts tile_id=136 outside [0, 135]".to_owned())
        );
    }

    #[test]
    fn snapshot_gates_match_oracle_projections() {
        // Hand-derived from `observation_assembly.py:361-363` (unknown sorted)
        // and `:440-442` (missing in field order).
        assert_eq!(check_public_unknown(&[]), Ok(()));
        assert_eq!(check_public_unknown(&["decision_id".to_owned()]), Ok(()));
        assert_eq!(
            check_public_unknown(&["zzz".to_owned(), "mmm".to_owned(), "decision_id".to_owned()]),
            Err("unknown public snapshot fields: ['mmm', 'zzz']".to_owned())
        );
        // Duplicates collapse (oracle `set()` semantics), still sorted.
        assert_eq!(
            check_public_unknown(&["zzz".to_owned(), "zzz".to_owned()]),
            Err("unknown public snapshot fields: ['zzz']".to_owned())
        );
        let full: Vec<String> = PUBLIC_SNAPSHOT_FIELDS
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(check_public_missing(&full), Ok(()));
        assert_eq!(
            check_public_missing(&[]),
            Err(format!(
                "public snapshot incomplete; missing {}",
                py_str_list_repr(&full)
            ))
        );
        // Missing preserves oracle field order even when present is shuffled.
        let partial = vec!["phase".to_owned(), "dealer".to_owned()];
        let err = check_public_missing(&partial).expect_err("must reject");
        assert_eq!(
            err,
            "public snapshot incomplete; missing ['decision_id', 'round_index', \
             'round_wind', 'hand_number', 'seat_winds', 'honba', 'riichi_sticks', \
             'scores', 'turn_actor', 'live_wall_tiles_remaining', 'ippatsu_active']"
        );
    }
}
