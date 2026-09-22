//! dataset_parse: frozen actor-dataset gate tables + pure column/filename checks
//! for `training/dataset_parse.py` over the `hydra-shard` owners.
//!
//! DAG: this module depends on pyo3 + `hydra-shard` ONLY (the same edge as
//! `oracle_guard.rs:3-5`, which gates key vocabularies over the shard
//! `FORBIDDEN_IN_ACTOR` owner; no feed/shard-IO/arrow edge, no new
//! dependency, so no manifest change). FORBIDS: dataclass construction (the
//! `VisibleMeld` / `PublicStateDelta` / `EventPayload` / `EventEnvelope` /
//! `ActorObservation` rebuilders stay Python-side attached construction over
//! live validated objects), pyarrow/parquet IO (`_verify_shards` keeps its
//! `memory_map` read + `verify_no_privileged_leakage` call in Python),
//! filesystem globbing (`_require_actor_parquet_dir` keeps its `Path` IO in
//! Python and crosses only plain name strings), and any JCS/sha/digest math
//! (none on this path — the envelope digest texts ride the existing
//! `contracts.make_digest_text` bridge entries, never reimplemented here).
//!
//! Frozen surface (free functions + tables only; no pyclass, no shared
//! state — shard verification is cold path, per-shard granularity):
//! - const `DATASET_PARSE_ACTOR_FIELDS` (13) <- `hydra_shard::parity::
//!   ACTOR_FIELDS` (referenced, never restated — mirrors
//!   `oracle_guard.rs:69` referencing `hydra_shard::replay::expand::
//!   FORBIDDEN_IN_ACTOR`).
//! - const `DATASET_PARSE_FORBIDDEN_IN_ACTOR` (6) <- `hydra_shard::replay::
//!   expand::FORBIDDEN_IN_ACTOR` (same reference, no literal duplication).
//! - const `DATASET_PARSE_REQUIRED_ACTOR_COLUMNS` (3) <- the
//!   `dataset_parse.py:291` required triple, frozen here in HEAD iteration
//!   order (fresh: no Rust owner — `rg` for `missing required column` over
//!   `crates/` is empty outside this file's own reject strings).
//! - fns `dataset_parse_check_actor_columns` /
//!   `dataset_parse_require_actor_columns` /
//!   `dataset_parse_check_actor_dir_names`: the `_verify_shards` column gate
//!   and the `_require_actor_parquet_dir` privileged-filename gate as total
//!   pure cores over plain strings, with byte-identical HEAD oracle texts
//!   (verified against `git show HEAD:src/hydra2/training/dataset_parse.py`;
//!   error values render Python-repr-style single-quoted, case-preserved).
//!
//! Concurrency pattern (matches `canon_rng.rs` + `rows_seal.rs`): validate
//! attached, ONE `py.detach(|| ...)` around the whole check (never per-item
//! attach), unit return wrapped attached. Error mapping: every reject is a
//! caller- or data-shape reject, so all map to `PyValueError` (never
//! `PyOSError`, never silent skip — mirroring `packet.rs:43-45`); the Python
//! translator maps the require-fn rejects to `CorruptArtifactError` by call
//! site (same `ValueError` type, distinct function).
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its consts/pyfunctions to the caller-provided `columnar` submodule,
//! so the Python translator keeps one bridge surface
//! (`hydra2._native.columnar`). MAIN wiring (shared files, MAIN-only):
//! `pub mod dataset_parse;` in `lib.rs` (alphabetical, after `contracts`)
//! plus `crate::dataset_parse::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:403-404`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rows_seal.rs:484-489`;
//! `let py = sub.py()` mirrors `contracts.rs:1115`;
//! `PyTuple::new(py, [...])` for frozen tuples mirrors `contracts.rs:1156`;
//! `PyFrozenSet::new(py, [...])` mirrors `contracts.rs:1186`;
//! `py.detach(|| ...)` for Rust-only compute mirrors
//! `canon_rng::batch_canonical_bytes`.

use hydra_shard::parity::ACTOR_FIELDS as SHARD_ACTOR_FIELDS;
use hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR as SHARD_FORBIDDEN_IN_ACTOR;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyModule, PyTuple};

/// Required actor-shard columns in HEAD iteration order
/// (`dataset_parse.py:291`: `("decision_id", "chosen_action_id",
/// `"actor_observation")`); first-missing wins the require gate.
const REQUIRED_ACTOR_COLUMNS: [&str; 3] = ["decision_id", "chosen_action_id", "actor_observation"];

/// Map a dataset-parse gate reject onto the Python boundary: column/dirname
/// shape rejects are data-shape rejects (`PyValueError`), never silent skips
/// (mirrors `packet.rs:55-57` `map_framer_err`).
fn parse_err(detail: String) -> PyErr {
    PyValueError::new_err(detail)
}

/// Pure `_verify_shards` column gate (detach-safe: borrowed strings only, no
/// interpreter interaction). Iterates the caller-supplied columns in order;
/// the first offender wins, privileged before unexpected per column — the
/// exact HEAD loop order (`dataset_parse.py:286-290`).
fn check_actor_columns(shard_name: &str, column_names: &[String]) -> Result<(), String> {
    for col in column_names {
        if SHARD_FORBIDDEN_IN_ACTOR.contains(&col.as_str()) {
            return Err(format!(
                "actor shard {shard_name} contains privileged column '{col}'"
            ));
        }
        if !SHARD_ACTOR_FIELDS.contains(&col.as_str()) {
            return Err(format!(
                "actor shard {shard_name} has unexpected column '{col}'"
            ));
        }
    }
    Ok(())
}

/// Pure `_verify_shards` required-column gate (detach-safe, same shape).
/// Iterates the frozen required triple in HEAD order (`dataset_parse.py:
/// 291-295`); the first missing column wins.
fn require_actor_columns(shard_name: &str, column_names: &[String]) -> Result<(), String> {
    for required in REQUIRED_ACTOR_COLUMNS {
        if !column_names.iter().any(|col| col == required) {
            return Err(format!(
                "actor shard {shard_name} missing required column '{required}'"
            ));
        }
    }
    Ok(())
}

/// Pure `_require_actor_parquet_dir` privileged-filename gate (detach-safe).
/// `privileged_glob` rides in glob encounter order and `parquet_names` in
/// `*.parquet` glob order (the translator collects both without sorting, so
/// the named offender matches HEAD); the first offender wins, glob hits
/// before substring hits — the exact HEAD loop order
/// (`dataset_parse.py:256-263`). Substring matching is Unicode-lowercase,
/// mirroring `str.lower()` for ASCII-or-lowercase names.
fn check_actor_dir_names(
    dir_display: &str,
    privileged_glob: &[String],
    parquet_names: &[String],
) -> Result<(), String> {
    if let Some(name) = privileged_glob.first() {
        return Err(format!(
            "privileged shard present in actor dataset directory {dir_display}: {name} — \
             actor loader must never touch privileged parquet (WP-05B no privileged fields)"
        ));
    }
    for name in parquet_names {
        if name.to_lowercase().contains("privileged") {
            return Err(format!("privileged parquet detected in actor dir: {name}"));
        }
    }
    Ok(())
}

/// `_verify_shards` column gate over plain values: privileged/unexpected
/// column rejects as `ValueError` (the translator maps to `ContractError`).
/// Compute runs detached; zero Python API inside the closure.
#[pyfunction]
fn dataset_parse_check_actor_columns(
    py: Python<'_>,
    shard_name: String,
    column_names: Vec<String>,
) -> PyResult<()> {
    py.detach(|| check_actor_columns(&shard_name, &column_names))
        .map_err(parse_err)
}

/// `_verify_shards` required-column gate over plain values: missing-column
/// rejects as `ValueError` (the translator maps to `CorruptArtifactError` by
/// call site). Compute runs detached; zero Python API inside the closure.
#[pyfunction]
fn dataset_parse_require_actor_columns(
    py: Python<'_>,
    shard_name: String,
    column_names: Vec<String>,
) -> PyResult<()> {
    py.detach(|| require_actor_columns(&shard_name, &column_names))
        .map_err(parse_err)
}

/// `_require_actor_parquet_dir` privileged-filename gate over plain values:
/// glob-hit/substring rejects as `ValueError` (the translator maps to
/// `ContractError`). Compute runs detached; zero Python API inside the
/// closure.
#[pyfunction]
fn dataset_parse_check_actor_dir_names(
    py: Python<'_>,
    dir_display: String,
    privileged_glob: Vec<String>,
    parquet_names: Vec<String>,
) -> PyResult<()> {
    py.detach(|| check_actor_dir_names(&dir_display, &privileged_glob, &parquet_names))
        .map_err(parse_err)
}

/// Attach the dataset-parse consts + pyfunctions to the caller-provided
/// `columnar` submodule (mirrors the `sub.add_function(wrap_pyfunction!(...,
/// sub)?)?` shape at `rows_seal.rs:484-489`; no new submodule, no new entry
/// point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "DATASET_PARSE_ACTOR_FIELDS",
        PyTuple::new(py, SHARD_ACTOR_FIELDS)?,
    )?;
    sub.add(
        "DATASET_PARSE_FORBIDDEN_IN_ACTOR",
        PyFrozenSet::new(py, SHARD_FORBIDDEN_IN_ACTOR)?,
    )?;
    sub.add(
        "DATASET_PARSE_REQUIRED_ACTOR_COLUMNS",
        PyTuple::new(py, REQUIRED_ACTOR_COLUMNS)?,
    )?;
    sub.add_function(wrap_pyfunction!(dataset_parse_check_actor_columns, sub)?)?;
    sub.add_function(wrap_pyfunction!(dataset_parse_require_actor_columns, sub)?)?;
    sub.add_function(wrap_pyfunction!(dataset_parse_check_actor_dir_names, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod dataset_parse_tests {
    use super::*;

    fn cols(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn full_actor_columns() -> Vec<String> {
        SHARD_ACTOR_FIELDS.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn frozen_tables_match_shard_owners() {
        assert_eq!(
            SHARD_ACTOR_FIELDS,
            hydra_shard::parity::ACTOR_FIELDS,
            "actor fields must track the shard owner"
        );
        assert_eq!(SHARD_ACTOR_FIELDS.len(), 13);
        assert_eq!(
            SHARD_FORBIDDEN_IN_ACTOR,
            hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR,
            "forbidden keys must track the shard owner"
        );
        assert_eq!(SHARD_FORBIDDEN_IN_ACTOR.len(), 6);
        // Disjoint by construction: the privileged message always wins the
        // per-column gate, never the unexpected one.
        for key in SHARD_FORBIDDEN_IN_ACTOR {
            assert!(
                !SHARD_ACTOR_FIELDS.contains(&key),
                "forbidden key {key} must not be an actor field"
            );
        }
    }

    #[test]
    fn required_triple_matches_head_order() {
        // HEAD dataset_parse.py:291 iteration order; first-missing wins.
        assert_eq!(
            REQUIRED_ACTOR_COLUMNS,
            ["decision_id", "chosen_action_id", "actor_observation"]
        );
        for col in REQUIRED_ACTOR_COLUMNS {
            assert!(
                SHARD_ACTOR_FIELDS.contains(&col),
                "required column {col} must be an actor field"
            );
        }
    }

    #[test]
    fn column_gate_accepts_canonical_layout() {
        assert_eq!(
            check_actor_columns("actor-000.parquet", &full_actor_columns()),
            Ok(())
        );
        assert_eq!(check_actor_columns("actor-000.parquet", &[]), Ok(()));
    }

    #[test]
    fn column_gate_rejects_privileged_before_unexpected() {
        // HEAD dataset_parse.py:286-290: per-column order, forbidden first.
        assert_eq!(
            check_actor_columns("actor-007.parquet", &cols(&["decision_id", "wall"])),
            Err("actor shard actor-007.parquet contains privileged column 'wall'".to_string())
        );
        assert_eq!(
            check_actor_columns("actor-007.parquet", &cols(&["decision_id", "bogus"])),
            Err("actor shard actor-007.parquet has unexpected column 'bogus'".to_string())
        );
        // First offender in column order wins, regardless of class.
        assert_eq!(
            check_actor_columns("actor-007.parquet", &cols(&["bogus", "wall"])),
            Err("actor shard actor-007.parquet has unexpected column 'bogus'".to_string())
        );
        assert_eq!(
            check_actor_columns("actor-007.parquet", &cols(&["wall", "bogus"])),
            Err("actor shard actor-007.parquet contains privileged column 'wall'".to_string())
        );
    }

    #[test]
    fn require_gate_first_missing_wins_in_triple_order() {
        assert_eq!(
            require_actor_columns("actor-007.parquet", &full_actor_columns()),
            Ok(())
        );
        assert_eq!(
            require_actor_columns("actor-007.parquet", &cols(&["game_id"])),
            Err("actor shard actor-007.parquet missing required column 'decision_id'".to_string())
        );
        assert_eq!(
            require_actor_columns(
                "actor-007.parquet",
                &cols(&["decision_id", "actor_observation"])
            ),
            Err(
                "actor shard actor-007.parquet missing required column 'chosen_action_id'"
                    .to_string()
            )
        );
    }

    #[test]
    fn dir_gate_names_glob_hit_with_full_oracle_text() {
        assert_eq!(
            check_actor_dir_names("ds/actor", &[], &cols(&["actor-000.parquet"])),
            Ok(())
        );
        // Byte-identical HEAD text incl. the em-dash separator.
        assert_eq!(
            check_actor_dir_names(
                "ds/actor",
                &cols(&["privileged-000.parquet"]),
                &cols(&["actor-000.parquet"])
            ),
            Err(
                "privileged shard present in actor dataset directory ds/actor: \
                 privileged-000.parquet — actor loader must never touch privileged parquet \
                 (WP-05B no privileged fields)"
                    .to_string()
            )
        );
        // Glob hits precede substring hits; substring match is
        // case-insensitive but the reported name keeps its case.
        assert_eq!(
            check_actor_dir_names("ds/actor", &[], &cols(&["actor-a.Privileged.parquet"])),
            Err("privileged parquet detected in actor dir: actor-a.Privileged.parquet".to_string())
        );
    }
}
