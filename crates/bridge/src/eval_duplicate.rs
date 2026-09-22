//! eval_duplicate: exact-duplicate + disjoint-wall pure leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (pure argument validation + grouping, so no feed/shard/search
//! owner exists — see the NONE note below; `crates/bridge/Cargo.toml:25`
//! carries the only dependency, no new dependency). FORBIDS: wall hashing
//! (`packet.wall_hash` decides; `wall_hash_from_tiles` / `wall_fingerprint`
//! stay Python callers of it), tile validation (136-int gate stays Python),
//! live-object orchestration (`WallBlock` / `MatchSchedule` / telemetry /
//! `BlockTolerance` construction, `build_wall_blocks`, `make_block_manifest`,
//! `split_blocks_held_out`, `balance_audit`, `report_telemetry_completeness`,
//! `confirmation_sidecar` stay Python), and any new JCS printer.
//!
//! TABLE ported here (frozen pure checks over plain values, oracle
//! `python/hydra2/eval/duplicate.py:147-217`):
//! - `eval_find_exact_duplicates` ← `find_exact_duplicates` (:147-164):
//!   discovery-order duplicate pairs over a wall-id → digest map. The
//!   `Mapping` gate stays Python (translator normalises any `Mapping` via
//!   `dict()` and rejects non-mappings with the oracle text); the bridge
//!   takes the normalised `dict`, stages every entry attached, and groups
//!   detached.
//! - `eval_validate_walls_disjoint` ← `validate_walls_disjoint` (:200-217):
//!   cross-partition wall-id uniqueness. The public shape stays variadic
//!   Python-side; the translator forwards `list(wall_collections)` so the
//!   bridge takes one list. Non-iterable collections propagate `TypeError`
//!   (same as the oracle's `for` loop); every other reject is `ValueError`
//!   with the oracle text for the `ContractError` translator.
//!
//! LOGIC staying Python: `_require_wall_id` / `_require_digest_value`
//! shaping (bridge-checked scalars assembled Python-side for the mapping
//! gate), `_bridge_wall_digest` + `wall_hash_from_tiles` /
//! `wall_fingerprint` (bridge-hash callers), `find_near_duplicates`
//! (tile-map orchestration over the fingerprint hash), `DuplicateReport` /
//! `BlockManifest` / `BlockSplit` dataclasses, `validate_blocks_disjoint`
//! (live `WallBlock` objects), and every `ContractError` translator (the
//! bridge raises `ValueError` with byte-identical text; `duplicate.py` maps
//! to `ContractError` — except the digest-shape `ValueError`, which the
//! translator re-raises bare exactly like the oracle's `validate_digest`
//! passthrough, see `test_exact_duplicate_rejection_contract`).
//!
//! NONE-owner note: `rg` for `find_exact_duplicates|validate_walls_disjoint`
//! over `crates/feed/src` + `crates/search/src` hits nothing (no owner
//! exists); the lowercase digest-shape test restates `contracts.rs:99-107`
//! (private there) with the oracle line cited per use.
//!
//! Shape per fn: attached staging (repr via the live Python API, so
//! `{wall_id!r}` renders byte-exact) → ONE `py.detach(|| …)` over owned
//! plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `validate.rs:83-95`) → attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). `bool` never reaches here as a wall id (wall
//! ids are `str`-only; a `bool` stages `None` and shares the nonempty-str
//! message, exactly like the oracle's `isinstance(str)` gate).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); `PyDict::iter` staging
//! mirrors `validate::validation_hash` (`crates/bridge/src/validate.rs:74`);
//! attached `repr` mirrors `rc_require.rs:63-65`; `Python::initialize()` +
//! `Python::attach` in tests mirror (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Lone-surrogate `str` wall ids fail `String` extraction and are rejected
//!   with the nonempty-str message, where the oracle would accept the label
//!   (it only checks `isinstance` + emptiness). Both sides fail closed for
//!   well-formed UTF-8 callers; no realistic wall id is affected.
//! - Lone-surrogate `str` digests stage as missing (nonempty message) where
//!   the oracle's `validate_digest` would raise `UnicodeEncodeError`
//!   (`ValueError` subclass). Both sides raise `ValueError`; well-formed
//!   callers are unaffected.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod eval_duplicate;` in `lib.rs` plus
//! `crate::eval_duplicate::register(&sub)?;` in `contracts::register` next to
//! `crate::eval_leaves::register(&sub)?;` (`contracts.rs:1296`).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

/// Digest-shape test (oracle `artifacts/digest.py:76-78` via
/// `contracts::make_digest_text`, `contracts.rs:153-160`): `sha256:` +
/// 64 lowercase hex, restating `contracts.rs:99-107` (private there).
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// One staged wall-hash entry: `wall_id` / `digest` are `Some` iff the value
/// is a nonempty `str`; `wall_repr` is the exact Python `repr` for `{!r}`
/// slots and `field` is the pre-rendered `wall_hashes[{wall_id!r}]` label.
struct ExactEntry {
    wall_id: Option<String>,
    wall_repr: String,
    digest: Option<String>,
    field: String,
}

/// Exact-duplicate grouping (`duplicate.py:147-164`): validate in discovery
/// order (fail-fast on the first bad entry), then anchor each digest's first
/// wall id and emit `(first, second)` pairs in discovery order.
fn check_exact_duplicates(entries: &[ExactEntry]) -> Result<Vec<(String, String)>, String> {
    let mut validated: Vec<(String, String)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(wall_id) = entry.wall_id.clone() else {
            return Err(format!(
                "wall_id must be nonempty str, got {}",
                entry.wall_repr
            ));
        };
        let Some(digest) = entry.digest.clone() else {
            return Err(format!("{} must be nonempty digest str", entry.field));
        };
        if !is_digest_shape(&digest) {
            return Err("contracts digest_text must match sha256:<64 lowercase hex>".to_owned());
        }
        validated.push((wall_id, digest));
    }
    let mut first_by_digest: HashMap<&str, &str> = HashMap::with_capacity(validated.len());
    let mut dups: Vec<(String, String)> = Vec::new();
    for (wall_id, digest) in &validated {
        if let Some(first) = first_by_digest.get(digest.as_str()) {
            dups.push(((*first).to_owned(), wall_id.clone()));
        } else {
            first_by_digest.insert(digest.as_str(), wall_id.as_str());
        }
    }
    Ok(dups)
}

/// One staged wall id: `value` is `Some` iff the element is a nonempty `str`;
/// `repr` is the exact Python `repr` for `{!r}` slots.
struct DisjointEntry {
    value: Option<String>,
    repr: String,
}

/// Disjoint-wall check (`duplicate.py:200-217`): `None` collections fail with
/// their index; every wall id must be a nonempty `str`; the first repeat
/// across partitions fails with both partition indices.
fn check_walls_disjoint(collections: &[Option<Vec<DisjointEntry>>]) -> Result<(), String> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (index, collection) in collections.iter().enumerate() {
        let Some(entries) = collection else {
            return Err(format!("wall_collections[{index}] is None"));
        };
        for entry in entries {
            let Some(wall_id) = entry.value.as_deref() else {
                return Err(format!("wall_id must be nonempty str, got {}", entry.repr));
            };
            if let Some(first) = seen.get(wall_id) {
                return Err(format!(
                    "wall {} appears in both partition {first} and {index}: wall sets must be disjoint",
                    entry.repr
                ));
            }
            seen.insert(wall_id, index);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Exact-duplicate detection over a normalised wall-id → digest dict
/// (`duplicate.py:147-164`). The caller normalises any `Mapping` via
/// `dict()` Python-side (insertion order preserved); non-mappings never
/// reach here. Every entry is staged attached (non-`str`/empty stages
/// `None`; the digest-shape decision runs detached); grouping runs under
/// ONE detach. Digest-shape rejects carry the oracle `validate_digest`
/// text so the translator can re-raise them as `ValueError`.
#[pyfunction]
fn eval_find_exact_duplicates(
    py: Python<'_>,
    wall_hashes: Bound<'_, PyDict>,
) -> PyResult<Vec<(String, String)>> {
    let mut entries: Vec<ExactEntry> = Vec::with_capacity(wall_hashes.len());
    for (key, value) in wall_hashes.iter() {
        let wall_repr = py_repr(&key)?;
        let wall_id: Option<String> = key.extract().ok().filter(|s: &String| !s.is_empty());
        let field = format!("wall_hashes[{wall_repr}]");
        let digest: Option<String> = value.extract().ok().filter(|s: &String| !s.is_empty());
        entries.push(ExactEntry {
            wall_id,
            wall_repr,
            digest,
            field,
        });
    }
    py.detach(|| check_exact_duplicates(&entries))
        .map_err(PyValueError::new_err)
}

/// Cross-partition wall-id disjointness (`duplicate.py:200-217`) over the
/// translator-forwarded `list(wall_collections)`. Each collection is staged
/// attached (`None` stages outer-`None`; non-iterables propagate the native
/// `TypeError` exactly like the oracle's `for` loop); the uniqueness
/// decision runs under ONE detach.
#[pyfunction]
fn eval_validate_walls_disjoint(
    py: Python<'_>,
    collections: Vec<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut staged: Vec<Option<Vec<DisjointEntry>>> = Vec::with_capacity(collections.len());
    for collection in &collections {
        if collection.is_none() {
            staged.push(None);
            continue;
        }
        let mut entries: Vec<DisjointEntry> = Vec::new();
        for item in collection.try_iter()? {
            let item = item?;
            let repr = py_repr(&item)?;
            let value: Option<String> = item.extract().ok().filter(|s: &String| !s.is_empty());
            entries.push(DisjointEntry { value, repr });
        }
        staged.push(Some(entries));
    }
    py.detach(|| check_walls_disjoint(&staged))
        .map_err(PyValueError::new_err)
}

/// Register the duplicate-wall leaves on the shared `contracts` submodule
/// (mirrors `validate.rs:111-127`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(eval_find_exact_duplicates, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_validate_walls_disjoint, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    /// Build a wall-hash dict attached (insertion order = argument order).
    fn hash_dict<'py>(py: Python<'py>, pairs: &[(&str, &str)]) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (key, value) in pairs {
            dict.set_item(*key, *value).unwrap();
        }
        dict
    }

    #[test]
    fn exact_groups_in_discovery_order() {
        Python::initialize();
        // Oracle goldens hand-derived from `duplicate.py:147-164` + the
        // wave-3 parity pin (`test_eval_parity_wave3.py:289`).
        Python::attach(|py| {
            let same = format!("sha256:{}", "ab".repeat(32));
            let other = format!("sha256:{}", "cd".repeat(32));
            let dict = hash_dict(py, &[("a", &same), ("b", &same)]);
            assert_eq!(
                eval_find_exact_duplicates(py, dict).unwrap(),
                vec![("a".to_owned(), "b".to_owned())]
            );
            let clean = hash_dict(py, &[("w1", &same), ("w2", &other)]);
            assert!(eval_find_exact_duplicates(py, clean).unwrap().is_empty());
            // Three walls sharing one digest anchor two pairs on the first.
            let trio = hash_dict(py, &[("w1", &same), ("w2", &same), ("w3", &same)]);
            assert_eq!(
                eval_find_exact_duplicates(py, trio).unwrap(),
                vec![
                    ("w1".to_owned(), "w2".to_owned()),
                    ("w1".to_owned(), "w3".to_owned())
                ]
            );
        });
    }

    #[test]
    fn exact_rejects_fail_closed_with_oracle_text() {
        Python::initialize();
        Python::attach(|py| {
            let good = format!("sha256:{}", "aa".repeat(32));
            // Empty wall id → nonempty-str gate.
            let bad_id = hash_dict(py, &[("", &good)]);
            assert_eq!(
                eval_find_exact_duplicates(py, bad_id)
                    .unwrap_err()
                    .to_string(),
                "ValueError: wall_id must be nonempty str, got ''"
            );
            // Empty digest → per-field nonempty gate.
            let empty_digest = hash_dict(py, &[("w1", "")]);
            assert_eq!(
                eval_find_exact_duplicates(py, empty_digest)
                    .unwrap_err()
                    .to_string(),
                "ValueError: wall_hashes['w1'] must be nonempty digest str"
            );
            // Malformed digest → validate_digest shape text (translator
            // re-raises this one as ValueError, never ContractError).
            let bad_shape = hash_dict(py, &[("w1", "not-a-digest")]);
            assert_eq!(
                eval_find_exact_duplicates(py, bad_shape)
                    .unwrap_err()
                    .to_string(),
                "ValueError: contracts digest_text must match sha256:<64 lowercase hex>"
            );
        });
    }

    #[test]
    fn disjoint_accepts_and_rejects_with_oracle_text() {
        Python::initialize();
        Python::attach(|py| {
            let first = PyList::new(py, ["w1", "w2"]).unwrap().into_any();
            let second = PyList::new(py, ["w3", "w4"]).unwrap().into_any();
            assert!(eval_validate_walls_disjoint(py, vec![first.clone(), second.clone()]).is_ok());
            // Overlap names both partitions.
            let overlap = PyList::new(py, ["w2", "w3"]).unwrap().into_any();
            assert_eq!(
                eval_validate_walls_disjoint(py, vec![first.clone(), overlap])
                    .unwrap_err()
                    .to_string(),
                "ValueError: wall 'w2' appears in both partition 0 and 1: wall sets must be disjoint"
            );
            // Empty wall id fails the nonempty-str gate.
            let empty = PyList::new(py, [""]).unwrap().into_any();
            assert!(
                eval_validate_walls_disjoint(py, vec![empty, second])
                    .unwrap_err()
                    .to_string()
                    .contains("wall_id must be nonempty str, got ''")
            );
            // None collection names its index.
            let none: Bound<'_, PyAny> = py.eval(c"None", None, None).unwrap();
            assert_eq!(
                eval_validate_walls_disjoint(py, vec![none])
                    .unwrap_err()
                    .to_string(),
                "ValueError: wall_collections[0] is None"
            );
        });
    }
}
