//! eval_schedule: SPEC 18.1 match-schedule frozen tables + pure leaves.
//!
//! Thin boundary over the search schedule owner
//! (`hydra_search::eval::schedule`, which owns the allocation math, the
//! `%3` latency draw, and every canon+hash seal through `feed::canon` +
//! `feed::digest`). This file only stages closed-domain values attached,
//! runs the checks and the owner pure functions detached, and returns
//! scalars/text. No wall-clock, no IO, no dataclass construction.
//!
//! DAG: pyo3 + `hydra-feed` (`canon`, `digest`) + `hydra-search`
//! (`eval::schedule`, `eval::semantic_seed` via the schedule owner) ONLY
//! (all already bridge dependencies, see `crates/bridge/Cargo.toml:19-23`;
//! no new dependency). FORBIDS: JCS re-printing (the feed `canon` owner is
//! the single printer), RNG reinvention (`%3` stays `%3` verbatim per M3;
//! Lemire-`below` is barred here), live-object orchestration (the
//! `MatchSchedule` dataclass, `build_match_schedule` wiring, and the
//! `_rust_canonical_digest` import guard stay Python-side in
//! `python/hydra2/eval/schedule.py`), and wall-clock/scheduling IO.
//!
//! TABLE ported here (frozen values + pure compute, oracle
//! `python/hydra2/eval/schedule.py`):
//! - `LATENCY_CLASSES` <- `schedule.py:41` (via the search owner const).
//! - `SYMMETRIC_ALLOCATIONS_PER_WALL` <- `:42` (via the owner const).
//! - `ROTATION_ALLOCATIONS_PER_WALL` <- `:43` (via the owner const).
//! - `TOTAL_GAMES_PER_WALL` <- `:44` (via the owner const).
//! - `eval_schedule_check_labels` <- `_validate_labels` (`:107-112`).
//! - `eval_schedule_symmetric_allocations` <- `symmetric_pair_allocations`
//!   (`:115-126`) via the owner (seat table `_SEAT_PAIRS` referenced, never
//!   recopied).
//! - `eval_schedule_rotation_allocations` <- `focal_rotation_allocations`
//!   (`:129-138`) via the owner.
//! - `eval_schedule_latency_draw` <- `_latency_draw` (`:187-189`):
//!   `u64BE(semantic_seed(master,key)[:8]) % 3` verbatim (M3); the key is
//!   built Rust-side via the owner's `schedule_stream_key` (same 17-field
//!   null-preserving projection `make_random_stream_key` produces for this
//!   purpose), then drawn via the owner's `latency_draw` (itself the
//!   `hydra-search` semantic-seed formula through feed canon+digest).
//! - `eval_schedule_seed_protocol_hash` <- `of_canonical(`
//!   `_SEED_PROTOCOL_PAYLOAD)` (`:55-64,183`): seals the owner's verbatim
//!   `seed_protocol_payload()` through the feed digest owner.
//! - `eval_schedule_commitment_hash_from_doc` <- `schedule_commitment_hash`
//!   (`:231-243`): the caller projects `schedule.to_json()` Python-side
//!   (live `MatchSchedule` objects never cross, mirroring
//!   `eval_case_manifest_hash_from_docs`); this pyfn stages the doc and
//!   seals it through the feed digest owner.
//! - `eval_schedule_placements_check` <- `seat_pair_placements_exact`
//!   (`:246-288`): the caller passes `wall_ids` + `seat_allocations` scalars
//!   (mirroring `telemetry_invalid_reason_for`); the gate renders the
//!   oracle-exact `wall '<id>': ...` texts.
//!
//! LOGIC staying Python (`eval/schedule.py`): the `MatchSchedule` dataclass
//! (`__post_init__`, `to_json`), `build_match_schedule` orchestration (wall
//! validation, allocation fan-out, latency rows, the four seals), the
//! `_rust_canonical_digest` import guard, and every `ContractError`
//! translator (the bridge raises `ValueError` with byte-identical text;
//! `schedule.py` maps to `ContractError`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `let py = sub.py()` mirrors
//! `contracts::register` (`crates/bridge/src/contracts.rs:1115`);
//! `sub.add(<str>, <&str>)` mirrors `contracts.rs:1153`;
//! `PyTuple::new(py, [...])` for frozen tuples mirrors `contracts.rs:1156`;
//! attached-check + `py.detach(|| ...)` with zero Python API inside mirrors
//! `contracts::ctr_block` (`crates/bridge/src/contracts.rs:447-456`);
//! `SearchError` mapped onto `ValueError` mirrors `eval::search_err`
//! (`crates/bridge/src/eval.rs:60-63`), except `InvalidArg` details are
//! returned bare (no `invalid search argument: ` prefix) so validator texts
//! stay byte-identical to the oracle; staged-`Value` JCS entry
//! `hydra_feed::canon::canonical_bytes_value` + fused
//! `hydra_feed::digest::of_canonical` mirror `validate::validation_hash`
//! (`crates/bridge/src/validate.rs:83-95`); the staging arms mirror
//! `canon_rng::py_to_value` (`crates/bridge/src/canon_rng.rs:275-363`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Feed owners (never reimplemented here): frozen seat/const tables via
//! `hydra_search::eval::schedule::{LATENCY_CLASSES,
//! SYMMETRIC_ALLOCATIONS_PER_WALL, ROTATION_ALLOCATIONS_PER_WALL,
//! TOTAL_GAMES_PER_WALL, SEAT_PAIRS}` (`crates/search/src/eval/schedule.rs:
//! 33-47`); allocation math via `symmetric_pair_allocations` /
//! `focal_rotation_allocations` (`:51-87`); key projection via
//! `schedule_stream_key` (`:193-217`); draws via `latency_draw` (`:223-230`);
//! protocol payload via `seed_protocol_payload` (`:179-187`); JCS via
//! `hydra_feed::canon::canonical_bytes_value` (`crates/feed/src/canon.rs:133`)
//! + the double-safe window `hydra_feed::canon::MAX_SAFE_INTEGER`
//!   (`crates/feed/src/canon.rs:146`); seals via
//!   `hydra_feed::digest::of_canonical` (`crates/feed/src/digest.rs:46`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Non-`str` labels fail `String` extraction and are rejected by the
//!   allocation leaves (`TypeError` for typed args, `ValueError` shape text
//!   for staged args), where the oracle would place them verbatim (it only
//!   unpacks positions). Both sides fail closed on well-typed callers only
//!   the bridge is stricter; valid `str` placements are unaffected.
//! - Non-`bytes` master seeds and non-`str` experiment/split ids fail typed
//!   extraction (`TypeError`), where the oracle raises `ContractError` with
//!   the same sentence. Both fail closed; the detached re-check keeps the
//!   byte-identical sentence for empty values.
//! - Out-of-`u64` `replicate_id` values (negative, huge) fail closed here
//!   (`ValueError` with the oracle sentence via staged `Bound` rejection),
//!   where the oracle would accept huge positives (it only checks
//!   nonnegativity). Schedule replicates are small wall-slot products;
//!   realistic callers are unaffected.
//! - Wall ids containing quotes/backslashes render with simple
//!   single-quote wrapping here, where the oracle renders `{wall_id!r}`
//!   (double quotes / escapes for exotic ids). Schedule wall ids are closed
//!   identifiers (`w-01`); well-formed callers are unaffected.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod eval_schedule;` in `lib.rs` plus
//! `crate::eval_schedule::register(&sub)?;` in `contracts::register` next to
//! `crate::eval_leaves::register(&sub)?;` (`contracts.rs:1296`).

use hydra_search::SearchError;
use hydra_search::eval::schedule as schedule_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Label gates (`_validate_labels`, `schedule.py:107-112`): four nonempty
/// distinct labels. `None` stages every non-sequence, wrong-length, or
/// empty/non-`str` shape so the helper is total on its own.
fn check_labels(staged: &Option<Vec<String>>) -> Result<(String, String, String, String), String> {
    let labels = match staged {
        Some(values) if values.len() == 4 && values.iter().all(|s| !s.is_empty()) => values,
        _ => return Err(String::from("labels must be four nonempty strings")),
    };
    let mut seen = std::collections::HashSet::new();
    for label in labels {
        seen.insert(label.as_str());
    }
    if seen.len() != 4 {
        return Err(String::from("labels must be distinct"));
    }
    Ok((
        labels[0].clone(),
        labels[1].clone(),
        labels[2].clone(),
        labels[3].clone(),
    ))
}

/// Allocation fan-out (`symmetric_pair_allocations`, `:115-126`) over owned
/// labels: length gates fire attached AND detached (total); the placement
/// itself runs through the search owner (seat table referenced, never
/// recopied).
fn symmetric_core(
    pair: &[String],
    field: &[String],
) -> Result<Vec<(String, String, String, String)>, String> {
    if pair.len() != 2 {
        return Err(String::from("pair must be two labels"));
    }
    if field.len() != 2 {
        return Err(String::from("field must be two labels"));
    }
    let pair_ref = match (pair.first(), pair.get(1)) {
        (Some(first), Some(second)) => (first.as_str(), second.as_str()),
        _ => return Err(String::from("pair must be two labels")),
    };
    let field_ref = match (field.first(), field.get(1)) {
        (Some(first), Some(second)) => (first.as_str(), second.as_str()),
        _ => return Err(String::from("field must be two labels")),
    };
    let rows = schedule_owner::symmetric_pair_allocations(pair_ref, field_ref);
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let quad = match (row.first(), row.get(1), row.get(2), row.get(3)) {
            (Some(a), Some(b), Some(c), Some(d)) => (a.clone(), b.clone(), c.clone(), d.clone()),
            _ => return Err(String::from("pair must be two labels")),
        };
        out.push(quad);
    }
    Ok(out)
}

/// Rotation fan-out (`focal_rotation_allocations`, `:129-138`): `others`
/// length gates fire attached AND detached (total); the focal label is
/// opaque (empty accepted verbatim, mirroring the oracle's unchecked
/// insert).
fn rotation_core(
    focal: &str,
    others: &[String],
) -> Result<Vec<(String, String, String, String)>, String> {
    if others.len() != 3 {
        return Err(String::from("others must be three labels"));
    }
    let triple = match (others.first(), others.get(1), others.get(2)) {
        (Some(first), Some(second), Some(third)) => [first.clone(), second.clone(), third.clone()],
        _ => return Err(String::from("others must be three labels")),
    };
    let rows = schedule_owner::focal_rotation_allocations(focal, &triple);
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let quad = match (row.first(), row.get(1), row.get(2), row.get(3)) {
            (Some(a), Some(b), Some(c), Some(d)) => (a.clone(), b.clone(), c.clone(), d.clone()),
            _ => return Err(String::from("others must be three labels")),
        };
        out.push(quad);
    }
    Ok(out)
}

/// Strip the `invalid search argument: ` prefix so validator texts stay
/// byte-identical to the oracle (`InvalidArg` details are already the
/// oracle sentences; every other variant is unreachable for these closed
/// inputs and keeps its full rendering).
fn search_detail(err: SearchError) -> String {
    match err {
        SearchError::InvalidArg { detail } => detail.to_string(),
        other => other.to_string(),
    }
}

/// Python-`list` rendering of plain integers (`[3, 3, 3, 3]`, `[]` when
/// empty). Domain: seat indices + hit counts (never quotes), so joining is
/// exact (same note as `hydra_search::eval::py_list_repr`).
fn py_int_list(items: &[usize]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&item.to_string());
    }
    out.push(']');
    out
}

/// Single-quote wall rendering (`'w-01'`). Domain: closed schedule wall
/// identifiers (no quotes/backslashes), so wrapping is exact there (same
/// note as `eval_leaves::py_str_list`).
fn wall_repr(wall_id: &str) -> String {
    let mut out = String::with_capacity(wall_id.len() + 2);
    out.push('\'');
    out.push_str(wall_id);
    out.push('\'');
    out
}

/// Exactness gates (`seat_pair_placements_exact`, `:246-288`) over caller
/// scalars: `wall_ids` + `seat_allocations` rows. Renders the oracle-exact
/// `wall '<id>': ...` texts (placements, seat-hit values, rotation values);
/// length/shape gates mirror the owner + `MatchSchedule.check` sentences so
/// the helper is total on its own.
#[allow(clippy::too_many_lines)]
fn check_placements(wall_ids: &[String], allocations: &[Vec<String>]) -> Result<(), String> {
    if allocations.len() < wall_ids.len() * schedule_owner::TOTAL_GAMES_PER_WALL
        || allocations.len() <= schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL
    {
        return Err(String::from("seat_allocations must carry 10 rows per wall"));
    }
    for row in allocations {
        if row.len() != 4 {
            return Err(String::from("allocation row is not a 4-label permutation"));
        }
    }
    if wall_ids.is_empty() {
        return Err(String::from("schedule needs at least one wall"));
    }
    if wall_ids.iter().any(String::is_empty) {
        return Err(String::from("wall_ids must be nonempty strings"));
    }
    {
        let mut seen = std::collections::HashSet::new();
        for wall_id in wall_ids {
            if !seen.insert(wall_id.as_str()) {
                return Err(String::from("wall_ids must be unique"));
            }
        }
    }
    if allocations.len() != wall_ids.len() * schedule_owner::TOTAL_GAMES_PER_WALL {
        return Err(String::from("seat_allocations must carry 10 rows per wall"));
    }
    let first_row = match allocations.first() {
        Some(row) => row,
        None => {
            return Err(String::from("seat_allocations must carry 10 rows per wall"));
        }
    };
    let pair_members: std::collections::HashSet<&str> = match (first_row.first(), first_row.get(1))
    {
        (Some(first), Some(second)) => [first.as_str(), second.as_str()].into_iter().collect(),
        _ => {
            return Err(String::from("allocation row is not a 4-label permutation"));
        }
    };
    let focal = match allocations
        .get(schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL)
        .and_then(|row| row.first())
    {
        Some(label) => label.clone(),
        None => {
            return Err(String::from("seat_allocations must carry 10 rows per wall"));
        }
    };
    for (wall_index, wall_id) in wall_ids.iter().enumerate() {
        let base = wall_index * schedule_owner::TOTAL_GAMES_PER_WALL;
        let symmetric =
            match allocations.get(base..base + schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL) {
                Some(rows) => rows,
                None => {
                    return Err(String::from("seat_allocations must carry 10 rows per wall"));
                }
            };
        let rotations = match allocations.get(
            base + schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL
                ..base + schedule_owner::TOTAL_GAMES_PER_WALL,
        ) {
            Some(rows) => rows,
            None => {
                return Err(String::from("seat_allocations must carry 10 rows per wall"));
            }
        };
        let mut placements: Vec<(usize, usize)> = Vec::with_capacity(symmetric.len());
        for row in symmetric {
            let mut seats: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, label)| pair_members.contains(label.as_str()))
                .map(|(index, _)| index)
                .collect();
            seats.sort_unstable();
            if seats.len() != 2 {
                return Err(format!(
                    "wall {}: A-pair placements not exact",
                    wall_repr(wall_id)
                ));
            }
            placements.push((seats[0], seats[1]));
        }
        placements.sort_unstable();
        let mut want = schedule_owner::SEAT_PAIRS.to_vec();
        want.sort_unstable();
        if placements != want {
            return Err(format!(
                "wall {}: A-pair placements not exact",
                wall_repr(wall_id)
            ));
        }
        let mut hits = [0usize; 4];
        for row in symmetric {
            for (index, label) in row.iter().enumerate() {
                if pair_members.contains(label.as_str()) && index < 4 {
                    hits[index] += 1;
                }
            }
        }
        if hits != [3, 3, 3, 3] {
            return Err(format!(
                "wall {}: seat coverage {} != [3,3,3,3]",
                wall_repr(wall_id),
                py_int_list(&hits)
            ));
        }
        let mut rotation_seats: Vec<usize> =
            rotations
                .iter()
                .flat_map(|row| {
                    row.iter().enumerate().filter_map(|(index, label)| {
                        if *label == focal { Some(index) } else { None }
                    })
                })
                .collect();
        rotation_seats.sort_unstable();
        if rotation_seats != [0, 1, 2, 3] {
            return Err(format!(
                "wall {}: focal rotation {} != all seats",
                wall_repr(wall_id),
                py_int_list(&rotation_seats)
            ));
        }
    }
    Ok(())
}

/// Stage one closed-domain Python object to a serde `Value` (attached only).
/// Mirrors `eval_leaves::py_to_value` arm-for-arm (`None` -> null; `bool`
/// BEFORE `int`; double-safe `int` only; finite `float` only; `str`; `list`
/// element-wise where `tuple` is rejected; `dict` with non-`str` keys
/// rejected). `record` names the digest call.
fn py_to_value(obj: &Bound<'_, PyAny>, record: &str) -> Result<serde_json::Value, String> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        let flag = obj
            .extract::<bool>()
            .map_err(|e| format!("{record}: bool unreadable: {e}"))?;
        return Ok(serde_json::Value::Bool(flag));
    }
    if obj.is_instance_of::<PyInt>() {
        match obj.extract::<i64>() {
            Ok(n) => {
                if !(-hydra_feed::canon::MAX_SAFE_INTEGER..=hydra_feed::canon::MAX_SAFE_INTEGER)
                    .contains(&n)
                {
                    return Err(format!(
                        "{record}: integer {n} exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                    ));
                }
                return Ok(serde_json::Value::Number(n.into()));
            }
            Err(_) => {
                return Err(format!(
                    "{record}: integer exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        let value = obj
            .extract::<f64>()
            .map_err(|e| format!("{record}: float unreadable: {e}"))?;
        if !value.is_finite() {
            return Err(format!(
                "{record}: non-finite number {value:?} has no canonical serialization"
            ));
        }
        match serde_json::Number::from_f64(value) {
            Some(number) => return Ok(serde_json::Value::Number(number)),
            None => {
                return Err(format!(
                    "{record}: non-finite number has no canonical serialization"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyString>() {
        match obj.extract::<String>() {
            Ok(text) => return Ok(serde_json::Value::String(text)),
            Err(e) => {
                return Err(format!(
                    "{record}: string contains an unpaired surrogate (invalid Unicode): {e}"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyList>() {
        let elements: Vec<Bound<'_, PyAny>> = obj
            .extract()
            .map_err(|e| format!("{record}: list unreadable: {e}"))?;
        let mut array = Vec::with_capacity(elements.len());
        for item in &elements {
            array.push(py_to_value(item, record)?);
        }
        return Ok(serde_json::Value::Array(array));
    }
    if obj.is_instance_of::<PyDict>() {
        let dict: Bound<'_, PyDict> = obj
            .extract()
            .map_err(|e| format!("{record}: dict unreadable: {e}"))?;
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(format!(
                    "{record}: object key is not a string; JSON objects are string-keyed only"
                ));
            }
            let name: String = key.extract().map_err(|e| {
                format!(
                    "{record}: object key contains an unpaired surrogate (invalid Unicode): {e}"
                )
            })?;
            map.insert(name, py_to_value(&value, record)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(format!(
        "{record}: value is outside the canonical JSON domain (null, bool, string, finite number, array, string-keyed object only)"
    ))
}

/// Pure seal over a staged value through the feed digest owner
/// (byte-identical to `artifacts/digest.py::of_canonical` by construction:
/// same fused `of_canonical` entry `validate::validation_hash` uses).
fn commitment_digest_value(value: &serde_json::Value) -> Result<String, String> {
    hydra_feed::digest::of_canonical(value)
        .map_err(|e| format!("contracts eval_schedule_commitment_hash_from_doc rejected: {e:?}"))
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Label gates (`_validate_labels`, `schedule.py:107-112`): four nonempty
/// distinct opaque labels. Accepts `list` and `tuple` (the oracle takes any
/// `Sequence`); non-sequences, wrong lengths, and empty/non-`str` elements
/// stage `None` and share the first oracle sentence. Decision runs detached.
#[pyfunction]
fn eval_schedule_check_labels(
    py: Python<'_>,
    labels: Bound<'_, PyAny>,
) -> PyResult<(String, String, String, String)> {
    let staged: Option<Vec<String>> =
        if labels.is_instance_of::<PyList>() || labels.is_instance_of::<PyTuple>() {
            labels.extract::<Vec<String>>().ok()
        } else {
            None
        };
    py.detach(|| check_labels(&staged))
        .map_err(PyValueError::new_err)
}

/// Symmetric placements (`symmetric_pair_allocations`, `:115-126`) via the
/// search owner. Length gates fire attached AND detached (total); the six
/// rows cross as a list of 4-tuples exactly like the oracle.
#[pyfunction]
fn eval_schedule_symmetric_allocations(
    py: Python<'_>,
    pair: Vec<String>,
    field: Vec<String>,
) -> PyResult<Vec<(String, String, String, String)>> {
    if pair.len() != 2 {
        return Err(PyValueError::new_err("pair must be two labels"));
    }
    if field.len() != 2 {
        return Err(PyValueError::new_err("field must be two labels"));
    }
    py.detach(|| symmetric_core(&pair, &field))
        .map_err(PyValueError::new_err)
}

/// Rotation placements (`focal_rotation_allocations`, `:129-138`) via the
/// search owner. The `others` length gate fires attached AND detached
/// (total); the focal label stays opaque.
#[pyfunction]
fn eval_schedule_rotation_allocations(
    py: Python<'_>,
    focal: String,
    others: Vec<String>,
) -> PyResult<Vec<(String, String, String, String)>> {
    if others.len() != 3 {
        return Err(PyValueError::new_err("others must be three labels"));
    }
    py.detach(|| rotation_core(&focal, &others))
        .map_err(PyValueError::new_err)
}

/// Latency draw (`_latency_draw`, `:187-189`): `u64BE(semantic_seed(master,
/// evaluation_schedule key)[:8]) % 3` verbatim (M3). The key is built
/// Rust-side via the owner's `schedule_stream_key` (same null-preserving
/// projection); the draw runs through the owner's `latency_draw` (feed
/// canon+digest, never a second hasher). Empty master/experiment/split
/// reject with the oracle sentences attached AND detached (total);
/// `replicate_id` stages through a `Bound` so `bool`/negative inputs fail
/// with the oracle sentence instead of a `TypeError`.
#[pyfunction]
#[pyo3(signature = (master_seed, experiment_id, split_id, replicate_id))]
fn eval_schedule_latency_draw(
    py: Python<'_>,
    master_seed: Vec<u8>,
    experiment_id: String,
    split_id: String,
    replicate_id: Bound<'_, PyAny>,
) -> PyResult<usize> {
    if master_seed.is_empty() {
        return Err(PyValueError::new_err("master_seed must be nonempty bytes"));
    }
    if experiment_id.is_empty() {
        return Err(PyValueError::new_err(
            "experiment_id must be a nonempty str",
        ));
    }
    if split_id.is_empty() {
        return Err(PyValueError::new_err("split_id must be a nonempty str"));
    }
    let staged_replicate: Option<u64> = if replicate_id.is_instance_of::<PyBool>() {
        None
    } else {
        replicate_id.extract::<u64>().ok()
    };
    let replicate = match staged_replicate {
        Some(value) => value,
        None => {
            return Err(PyValueError::new_err(
                "replicate_id must be a nonnegative int",
            ));
        }
    };
    py.detach(|| -> Result<usize, String> {
        if master_seed.is_empty() {
            return Err(String::from("master_seed must be nonempty bytes"));
        }
        if experiment_id.is_empty() {
            return Err(String::from("experiment_id must be a nonempty str"));
        }
        if split_id.is_empty() {
            return Err(String::from("split_id must be a nonempty str"));
        }
        let key = schedule_owner::schedule_stream_key(&experiment_id, &split_id, replicate);
        schedule_owner::latency_draw(&master_seed, &key).map_err(search_detail)
    })
    .map_err(PyValueError::new_err)
}

/// Seed-protocol seal (`of_canonical(_SEED_PROTOCOL_PAYLOAD)`, `:55-64,183`):
/// digests the owner's verbatim payload through the feed digest owner. No
/// arguments; the frozen payload cannot fork.
#[pyfunction]
fn eval_schedule_seed_protocol_hash(py: Python<'_>) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let payload = schedule_owner::seed_protocol_payload();
        hydra_feed::digest::of_canonical(&payload)
            .map_err(|e| format!("contracts eval_schedule_seed_protocol_hash rejected: {e:?}"))
    })
    .map_err(PyValueError::new_err)
}

/// Commitment seal (`schedule_commitment_hash`, `:231-243`) over the
/// caller-projected `schedule.to_json()` doc: the caller maps the live
/// `MatchSchedule` Python-side (live objects never cross); this pyfn stages
/// the doc attached, then runs ONE detach around the feed canon+digest seal
/// and returns the `sha256:<hex>` text.
#[pyfunction]
fn eval_schedule_commitment_hash_from_doc(
    py: Python<'_>,
    doc: Bound<'_, PyAny>,
) -> PyResult<String> {
    let staged = py_to_value(&doc, "bridge:eval_schedule_commitment_hash_from_doc")
        .map_err(PyValueError::new_err)?;
    py.detach(|| commitment_digest_value(&staged))
        .map_err(PyValueError::new_err)
}

/// Exactness gates (`seat_pair_placements_exact`, `:246-288`) over
/// caller-passed scalars: `wall_ids` + `seat_allocations` rows (each a
/// 4-label list). Shape gates fire attached AND detached (total); the
/// placement/coverage/rotation decisions render the oracle-exact
/// `wall '<id>': ...` texts. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (wall_ids, seat_allocations))]
fn eval_schedule_placements_check(
    py: Python<'_>,
    wall_ids: Vec<String>,
    seat_allocations: Vec<Vec<String>>,
) -> PyResult<()> {
    check_placements(&wall_ids, &seat_allocations).map_err(PyValueError::new_err)?;
    py.detach(|| check_placements(&wall_ids, &seat_allocations))
        .map_err(PyValueError::new_err)
}

/// Register the schedule tables + pure leaves on the shared `contracts`
/// submodule (mirrors `validate.rs:111-127`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "LATENCY_CLASSES",
        PyTuple::new(py, schedule_owner::LATENCY_CLASSES)?,
    )?;
    sub.add(
        "SYMMETRIC_ALLOCATIONS_PER_WALL",
        schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL,
    )?;
    sub.add(
        "ROTATION_ALLOCATIONS_PER_WALL",
        schedule_owner::ROTATION_ALLOCATIONS_PER_WALL,
    )?;
    sub.add("TOTAL_GAMES_PER_WALL", schedule_owner::TOTAL_GAMES_PER_WALL)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_check_labels, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_symmetric_allocations, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_rotation_allocations, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_latency_draw, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_seed_protocol_hash, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        eval_schedule_commitment_hash_from_doc,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(eval_schedule_placements_check, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle() {
        Python::initialize();
        // Oracle values hand-traced from `schedule.py:41-44`.
        assert_eq!(schedule_owner::LATENCY_CLASSES, ["low", "moderate", "high"]);
        assert_eq!(schedule_owner::SYMMETRIC_ALLOCATIONS_PER_WALL, 6);
        assert_eq!(schedule_owner::ROTATION_ALLOCATIONS_PER_WALL, 4);
        assert_eq!(schedule_owner::TOTAL_GAMES_PER_WALL, 10);
        assert_eq!(
            schedule_owner::SEAT_PAIRS,
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
    }

    #[test]
    fn label_gates_match_oracle() {
        // `schedule.py:107-112`: four nonempty distinct labels.
        let good = Some(
            ["a", "b", "c", "d"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
        );
        assert_eq!(
            check_labels(&good),
            Ok((
                String::from("a"),
                String::from("b"),
                String::from("c"),
                String::from("d")
            ))
        );
        assert_eq!(
            check_labels(&None),
            Err(String::from("labels must be four nonempty strings"))
        );
        let short = Some(vec![String::from("a"), String::from("b")]);
        assert_eq!(
            check_labels(&short),
            Err(String::from("labels must be four nonempty strings"))
        );
        let empty = Some(vec![
            String::from("a"),
            String::from(""),
            String::from("c"),
            String::from("d"),
        ]);
        assert_eq!(
            check_labels(&empty),
            Err(String::from("labels must be four nonempty strings"))
        );
        let dup = Some(vec![
            String::from("a"),
            String::from("a"),
            String::from("c"),
            String::from("d"),
        ]);
        assert_eq!(
            check_labels(&dup),
            Err(String::from("labels must be distinct"))
        );
    }

    #[test]
    fn allocation_geometry_matches_oracle() {
        Python::initialize();
        // Hand-derived from `schedule.py:115-138` over `_SEAT_PAIRS` order
        // `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)` with pair `(A,B)` + field
        // `(C,D)`, and focal `A` over others `[B,C,D]`.
        let pair = [String::from("A"), String::from("B")];
        let field = [String::from("C"), String::from("D")];
        assert_eq!(
            symmetric_core(&pair, &field),
            Ok(vec![
                (
                    String::from("A"),
                    String::from("B"),
                    String::from("C"),
                    String::from("D")
                ),
                (
                    String::from("A"),
                    String::from("C"),
                    String::from("B"),
                    String::from("D")
                ),
                (
                    String::from("A"),
                    String::from("C"),
                    String::from("D"),
                    String::from("B")
                ),
                (
                    String::from("C"),
                    String::from("A"),
                    String::from("B"),
                    String::from("D")
                ),
                (
                    String::from("C"),
                    String::from("A"),
                    String::from("D"),
                    String::from("B")
                ),
                (
                    String::from("C"),
                    String::from("D"),
                    String::from("A"),
                    String::from("B")
                ),
            ])
        );
        let others = [String::from("B"), String::from("C"), String::from("D")];
        assert_eq!(
            rotation_core("A", &others),
            Ok(vec![
                (
                    String::from("A"),
                    String::from("B"),
                    String::from("C"),
                    String::from("D")
                ),
                (
                    String::from("B"),
                    String::from("A"),
                    String::from("C"),
                    String::from("D")
                ),
                (
                    String::from("B"),
                    String::from("C"),
                    String::from("A"),
                    String::from("D")
                ),
                (
                    String::from("B"),
                    String::from("C"),
                    String::from("D"),
                    String::from("A")
                ),
            ])
        );
        assert_eq!(
            symmetric_core(&[String::from("A")], &field),
            Err(String::from("pair must be two labels"))
        );
        assert_eq!(
            rotation_core("A", &[String::from("B")]),
            Err(String::from("others must be three labels"))
        );
    }

    #[test]
    fn latency_percent3_kat_matches_owner() {
        Python::initialize();
        // `%3` KAT (`schedule.rs:427-433`): 20 verbatim draws on
        // `bytes(range(32))` with `exp-kat` / `split-kat` — a Lemire port
        // would fork slot 0.
        let master: Vec<u8> = (0u8..32).collect();
        let want = [
            1usize, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 2, 2, 0, 0, 2,
        ];
        for (slot, class_index) in want.iter().enumerate() {
            let key = schedule_owner::schedule_stream_key("exp-kat", "split-kat", slot as u64);
            assert_eq!(
                schedule_owner::latency_draw(&master, &key).expect("kat draw"),
                *class_index,
                "slot {slot}"
            );
        }
    }

    #[test]
    fn seal_goldens_match_owner() {
        // T2 goldens (`schedule.rs:393-421`): 2-wall KAT commitment +
        // walls/latency/proto hashes.
        let walls = vec![String::from("w-001"), String::from("w-002")];
        let labels = ["A", "B", "C", "D"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let master: Vec<u8> = (0u8..32).collect();
        let rules = format!("sha256:{}", "a".repeat(64));
        let schedule = schedule_owner::build_match_schedule(
            &walls,
            &labels,
            &rules,
            &master,
            "exp-kat",
            "split-kat",
        )
        .expect("kat schedule");
        assert_eq!(
            schedule_owner::schedule_commitment_hash(&schedule).expect("commitment"),
            "sha256:7e4bd2c5a1e5b8d054e1471e6df6a07945ffd08cbcc6b551848958243250783d"
        );
        assert_eq!(
            schedule.walls_hash,
            "sha256:a64106a6d9ee1186b6ead2341571119dba0ac8a5c49ac2e3c9eaf813cdef1f78"
        );
        assert_eq!(
            schedule.latency_schedule_hash,
            "sha256:18f4b186c7745ba2320b91bb8e53d9eeacd6b5cd79568bb1906769f272e5b2a2"
        );
        assert_eq!(
            schedule.seed_protocol_hash,
            "sha256:30538109867443d98bfc50220c1011b4fee656a702d348f2690632e6f9fa63b4"
        );
        // The fused feed seal over the same JSON value agrees (same owner
        // bytes by construction).
        assert_eq!(
            commitment_digest_value(&schedule.to_json_value()).expect("fused seal"),
            "sha256:7e4bd2c5a1e5b8d054e1471e6df6a07945ffd08cbcc6b551848958243250783d"
        );
        assert_eq!(
            hydra_feed::digest::of_canonical(&schedule_owner::seed_protocol_payload())
                .expect("proto seal"),
            "sha256:30538109867443d98bfc50220c1011b4fee656a702d348f2690632e6f9fa63b4"
        );
    }

    #[test]
    fn placements_gate_matches_oracle() {
        // Valid KAT schedule passes; the oracle-exact wall prefix renders
        // with single quotes over the closed wall-id domain.
        let walls = vec![String::from("w-001")];
        let labels = ["A", "B", "C", "D"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let master: Vec<u8> = (0u8..32).collect();
        let rules = format!("sha256:{}", "a".repeat(64));
        let schedule = schedule_owner::build_match_schedule(
            &walls,
            &labels,
            &rules,
            &master,
            "exp-kat",
            "split-kat",
        )
        .expect("kat schedule");
        let wall_ids = schedule.wall_ids.clone();
        let allocations: Vec<Vec<String>> = schedule
            .seat_allocations
            .iter()
            .map(|row| row.to_vec())
            .collect();
        assert!(check_placements(&wall_ids, &allocations).is_ok());
        // Tamper the first symmetric row so the A-pair no longer covers all
        // six unordered seat pairs: gate fails with the oracle sentence.
        let mut drifted = allocations.clone();
        drifted[1] = drifted[0].clone();
        assert_eq!(
            check_placements(&wall_ids, &drifted),
            Err(String::from("wall 'w-001': A-pair placements not exact"))
        );
        // Short allocations fail with the owner length sentence.
        assert_eq!(
            check_placements(&wall_ids, &allocations[..4]),
            Err(String::from("seat_allocations must carry 10 rows per wall"))
        );
    }

    #[test]
    fn pyfn_wrappers_agree_with_detached_checks() {
        Python::initialize();
        Python::attach(|py| {
            let labels = PyTuple::new(
                py,
                [
                    PyString::new(py, "a"),
                    PyString::new(py, "b"),
                    PyString::new(py, "c"),
                    PyString::new(py, "d"),
                ],
            )
            .expect("labels tuple")
            .into_any();
            let checked = eval_schedule_check_labels(py, labels).expect("valid labels");
            assert_eq!(
                checked,
                (
                    String::from("a"),
                    String::from("b"),
                    String::from("c"),
                    String::from("d")
                )
            );
            let bad = PyTuple::new(py, [PyString::new(py, "a")])
                .expect("short tuple")
                .into_any();
            assert!(eval_schedule_check_labels(py, bad).is_err());
            let rows = eval_schedule_symmetric_allocations(
                py,
                vec![String::from("A"), String::from("B")],
                vec![String::from("C"), String::from("D")],
            )
            .expect("symmetric rows");
            assert_eq!(rows.len(), 6);
            assert_eq!(
                rows[0],
                (
                    String::from("A"),
                    String::from("B"),
                    String::from("C"),
                    String::from("D")
                )
            );
            let rotations = eval_schedule_rotation_allocations(
                py,
                String::from("A"),
                vec![String::from("B"), String::from("C"), String::from("D")],
            )
            .expect("rotation rows");
            assert_eq!(rotations.len(), 4);
            let replicate = PyInt::new(py, 0).into_any();
            let draw = eval_schedule_latency_draw(
                py,
                (0u8..32).collect(),
                String::from("exp-kat"),
                String::from("split-kat"),
                replicate,
            )
            .expect("latency draw");
            assert_eq!(draw, 1);
            let proto = eval_schedule_seed_protocol_hash(py).expect("proto hash");
            assert_eq!(
                proto,
                "sha256:30538109867443d98bfc50220c1011b4fee656a702d348f2690632e6f9fa63b4"
            );
            let walls = vec![String::from("w-001")];
            let labels = ["A", "B", "C", "D"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            let schedule = schedule_owner::build_match_schedule(
                &walls,
                &labels,
                &format!("sha256:{}", "a".repeat(64)),
                &(0u8..32).collect::<Vec<u8>>(),
                "exp-kat",
                "split-kat",
            )
            .expect("kat schedule");
            let wall_ids = schedule.wall_ids.clone();
            let allocations: Vec<Vec<String>> = schedule
                .seat_allocations
                .iter()
                .map(|row| row.to_vec())
                .collect();
            assert!(eval_schedule_placements_check(py, wall_ids, allocations).is_ok());
        });
    }
}
