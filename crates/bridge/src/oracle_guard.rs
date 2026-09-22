//! oracle_guard: WP-07B privileged-firewall vocabulary + fail-closed gates on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-shard` ONLY (the 6-key `FORBIDDEN_IN_ACTOR` owner for the
//! actor-firewall base — same edge as `columnar.rs`/`tiles.rs`, no new
//! dependency). FORBIDS: JCS/sha reimplementation (digests ride the feed
//! `digest` owner; this module only gates key vocabularies), IO/fsync/temp
//! discipline (the privileged/actor shard-path probes stay Python as OS
//! authority), live-object orchestration (the `PrivilegedOracleLoader`
//! dataclass, row assembly, and the actor-row JSON parse stay Python), and
//! `ContractError` shaping (the bridge raises `ValueError`; thin Python
//! translators map to `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/oracle_guard.py`:
//! - const `GUARD_AUTHORIZED_TRAIN_SPLIT` <- `AUTHORIZED_TRAIN_SPLIT`
//!   (`oracle_guard.py:50`: `"train"`).
//! - const `GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE` <- `AUTHORIZED_SPLITS_FOR_INFERENCE`
//!   (`oracle_guard.py:51`: `{"train", "held_out", "test", "eval"}`).
//! - const `GUARD_FORBIDDEN_IN_ACTOR_KEYS` (9) <- `FORBIDDEN_IN_ACTOR_KEYS`
//!   (`oracle_guard.py:45-47`): the shard-owner base
//!   (`hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR`, 6 — referenced, never
//!   restated) plus `GUARD_FORBIDDEN_EXTRA` (3).
//! - const `GUARD_PRIVILEGED_KEYS` (12) <- `PRIVILEGED_KEYS`
//!   (`oracle_guard.py:28-43`): the 9 forbidden plus `GUARD_PRIVILEGED_ONLY`
//!   (3 oracle-only keys).
//! - `guard_require_train_split` <- `_require_train_split`
//!   (`oracle_guard.py:55-59`): TOTAL over any value (a non-`str` rejects with
//!   its staged `repr`, never `TypeError`).
//! - `guard_validate_actor_batch` <- `validate_actor_batch_no_privileged`
//!   (`oracle_guard.py:62-82`): owned key lists cross (top-level + nested-obs
//!   keys staged attached, never live dicts); non-`str` keys skip exactly like
//!   the oracle membership test.
//! - `guard_check_wall_leakage` <- `check_wall_leakage`
//!   (`oracle_guard.py:114-128`): `BTreeSet` overlap (UTF-8 byte order equals
//!   code-point order, so the order agrees with `sorted()`) with the
//!   oracle-identical `N wall(s) overlap ... ['a', ...]` text.
//! - `guard_check_split_disjoint` <- `check_split_disjoint`
//!   (`oracle_guard.py:131-141`): same shape, `decision(s)` text.
//!
//! LOGIC staying Python: `assert_no_privileged_leakage_in_actor_row` (row JSON
//! parse + `CorruptArtifactError` + `decision_id` interpolation),
//! `_privileged_shard_paths` / `_actor_shard_paths` (filesystem probes — OS
//! authority), the `actor_observation`/`observation`/`obs` fallback selection,
//! every `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `AUTHORIZED_TRAIN_SPLIT|
//! AUTHORIZED_SPLITS_FOR_INFERENCE|PRIVILEGED_KEYS|guard_require|
//! guard_validate_actor|guard_check_wall|guard_check_split` over
//! `crates/bridge/src` + `crates/feed/src` + `crates/search/src` +
//! `crates/shard/src` hits only the 6-key shard base
//! (`shard/src/expand.rs:113`, `shard/src/writer.rs:225`) and an unrelated
//! ISMCTS redaction list (`search/src/ismcts_driver.rs:50-59` — redaction
//! substrings, never the guard tables); the combined 9/12 guard sets and the
//! four gates are fresh here.
//!
//! Shape per fn: attached staging (exact `repr(value)` text via the live
//! Python API, so `{value!r}` renders byte-exact per `rc_require.rs:60-65` +
//! the `loop_state.rs:40-46` firewall precedent) -> ONE `py.detach(|| ...)`
//! over owned plain data with zero Python API inside (per
//! `contracts.rs:434-437`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Consts via `sub.add` (`PyFrozenSet` per
//! `contracts.rs:1186`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `belief_leaves.rs:192-229`); no new entry
//! point. MAIN wiring: `pub mod oracle_guard;` in `lib.rs` plus
//! `crate::oracle_guard::register(&sub)?;` appended after the last
//! `crate::*::register(&sub)?;` line in `contracts.rs::register`.

use hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR as SHARD_FORBIDDEN_IN_ACTOR;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyFrozenSet, PyModule};

/// Only split authorized for oracle distillation (`oracle_guard.py:50`).
const GUARD_AUTHORIZED_TRAIN_SPLIT: &str = "train";

/// Splits authorized for inference (`oracle_guard.py:51`).
const GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE: [&str; 4] = ["train", "held_out", "test", "eval"];

/// Guard-only extras beyond the shard base set (`oracle_guard.py:45-47`).
const GUARD_FORBIDDEN_EXTRA: [&str; 3] =
    ["privileged_label", "wall_remaining", "hidden_tile_counts"];

/// Oracle-only privileged keys beyond the actor-forbidden set (`oracle_guard.py:38-41`).
const GUARD_PRIVILEGED_ONLY: [&str; 3] = ["opponent_concealed", "unrevealed_dora", "wait_tiles"];

/// Pure membership core: shard base + guard extras (mirrors the oracle's
/// `FORBIDDEN_IN_ACTOR_KEYS` union without restating the 6 owned literals).
fn is_forbidden_in_actor(key: &str) -> bool {
    SHARD_FORBIDDEN_IN_ACTOR.contains(&key) || GUARD_FORBIDDEN_EXTRA.contains(&key)
}

/// Pure membership core: every forbidden key is privileged (the oracle checks
/// both sets with the same message, so one predicate covers both branches).
fn is_privileged(key: &str) -> bool {
    is_forbidden_in_actor(key) || GUARD_PRIVILEGED_ONLY.contains(&key)
}

/// Pure split-gate core: only `"train"` passes (mirrors
/// `_require_train_split`; the oracle-identical message is built by the pyfn
/// from the staged `repr`, never rendered here).
fn train_split_ok(split: Option<&str>) -> bool {
    split == Some(GUARD_AUTHORIZED_TRAIN_SPLIT)
}

/// Pure batch-firewall core over staged `(text, repr)` key pairs (mirrors
/// `validate_actor_batch_no_privileged` order: top-level keys first with the
/// `actor batch` text, then nested-obs keys with the `actor_observation`
/// text). A `None` text is a non-`str` key, which the oracle membership test
/// skips — so this skips it too. No Python API inside (detach-safe).
fn validate_actor_batch_core(
    top_keys: &[(Option<String>, String)],
    obs_keys: &[(Option<String>, String)],
) -> Result<(), String> {
    for (text, repr_text) in top_keys {
        if let Some(key) = text
            && (is_forbidden_in_actor(key) || is_privileged(key))
        {
            return Err(format!("actor batch contains privileged field {repr_text}"));
        }
    }
    for (text, repr_text) in obs_keys {
        if let Some(key) = text
            && (is_forbidden_in_actor(key) || is_privileged(key))
        {
            return Err(format!(
                "actor_observation contains privileged field {repr_text}"
            ));
        }
    }
    Ok(())
}

/// Pure overlap core: deduped sorted intersection (mirrors `set(a) & set(b)`
/// plus `sorted()`; `BTreeSet` byte order equals code-point order for UTF-8,
/// so the order agrees with the oracle). No Python API inside (detach-safe).
fn overlap_sorted(first: &[String], second: &[String]) -> Vec<String> {
    use std::collections::BTreeSet;
    let set_first: BTreeSet<&str> = first.iter().map(String::as_str).collect();
    let set_second: BTreeSet<&str> = second.iter().map(String::as_str).collect();
    set_first
        .intersection(&set_second)
        .map(ToString::to_string)
        .collect()
}

/// Python-`repr`-style rendering of a string list (`['a', 'b']`, single-quoted
/// per the bridge convention; ids are plain ASCII so no escaping diverges).
fn py_str_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(']');
    out
}

/// Pure wall-leakage core (mirrors `check_wall_leakage` text bit-for-bit).
fn wall_leakage_error(train_ids: &[String], held_ids: &[String]) -> Result<(), String> {
    let overlap = overlap_sorted(train_ids, held_ids);
    if overlap.is_empty() {
        Ok(())
    } else {
        let count = overlap.len();
        let head = py_str_list(&overlap[..overlap.len().min(3)]);
        Err(format!(
            "wall leakage: {count} wall(s) overlap between train and held_out: {head}"
        ))
    }
}

/// Pure split-disjointness core (mirrors `check_split_disjoint` text bit-for-bit).
fn split_disjoint_error(train_ids: &[String], held_ids: &[String]) -> Result<(), String> {
    let overlap = overlap_sorted(train_ids, held_ids);
    if overlap.is_empty() {
        Ok(())
    } else {
        let count = overlap.len();
        let head = py_str_list(&overlap[..overlap.len().min(3)]);
        Err(format!(
            "split leakage: {count} decision(s) overlap: {head}"
        ))
    }
}

/// Train-split gate (mirrors `_require_train_split` bit-for-bit: any non-`str`
/// value rejects with its `{value!r}` rendering). Translators call
/// positionally; the bridge raises `ValueError`, Python maps to
/// `ContractError` with byte-identical text. Compute runs detached with zero
/// Python API inside; counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (split,))]
fn guard_require_train_split(py: Python<'_>, split: Bound<'_, PyAny>) -> PyResult<()> {
    let repr_text = split.repr()?.to_str()?.to_owned();
    let staged: Option<String> = split.extract().ok();
    let ok = py.detach(|| train_split_ok(staged.as_deref()));
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "PrivilegedOracleLoader may only load split='{}', got {repr_text}: ",
            GUARD_AUTHORIZED_TRAIN_SPLIT
        )))
    }
}

/// Actor-batch firewall over owned top-level + nested-obs key lists (mirrors
/// `validate_actor_batch_no_privileged` without live dicts — the caller
/// selects `actor_observation`/`observation`/`obs` and passes
/// `list(batch.keys())` + `list(obs.keys())` positionally). Translators call
/// positionally; the bridge raises `ValueError`, Python maps to
/// `ContractError` with byte-identical text. Compute runs detached with zero
/// Python API inside; counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (top_keys, obs_keys))]
fn guard_validate_actor_batch(
    py: Python<'_>,
    top_keys: Vec<Bound<'_, PyAny>>,
    obs_keys: Vec<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut top: Vec<(Option<String>, String)> = Vec::with_capacity(top_keys.len());
    for key in &top_keys {
        let repr_text = key.repr()?.to_str()?.to_owned();
        let text: Option<String> = key.extract().ok();
        top.push((text, repr_text));
    }
    let mut obs: Vec<(Option<String>, String)> = Vec::with_capacity(obs_keys.len());
    for key in &obs_keys {
        let repr_text = key.repr()?.to_str()?.to_owned();
        let text: Option<String> = key.extract().ok();
        obs.push((text, repr_text));
    }
    py.detach(|| validate_actor_batch_core(&top, &obs))
        .map_err(PyValueError::new_err)
}

/// Wall-disjointness gate over owned id lists (mirrors `check_wall_leakage`
/// without live sets — the caller passes `list(train)` + `list(held)`
/// positionally). Translators call positionally; the bridge raises
/// `ValueError`, Python maps to `ContractError` with byte-identical text.
/// Compute runs detached with zero Python API inside; counter-free
/// deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (train_ids, held_ids))]
fn guard_check_wall_leakage(
    py: Python<'_>,
    train_ids: Vec<String>,
    held_ids: Vec<String>,
) -> PyResult<()> {
    py.detach(|| wall_leakage_error(&train_ids, &held_ids))
        .map_err(PyValueError::new_err)
}

/// Split-disjointness gate over owned id lists (mirrors `check_split_disjoint`
/// without live sets — the caller passes `list(train)` + `list(held)`
/// positionally). Translators call positionally; the bridge raises
/// `ValueError`, Python maps to `ContractError` with byte-identical text.
/// Compute runs detached with zero Python API inside; counter-free
/// deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (train_ids, held_ids))]
fn guard_check_split_disjoint(
    py: Python<'_>,
    train_ids: Vec<String>,
    held_ids: Vec<String>,
) -> PyResult<()> {
    py.detach(|| split_disjoint_error(&train_ids, &held_ids))
        .map_err(PyValueError::new_err)
}

/// Register the guard tables + gates on the shared `contracts` submodule
/// (mirrors `belief_leaves.rs:192-229`): `let py = sub.py();` (per
/// `contracts.rs:1115`), fns via `wrap_pyfunction!(f, sub)` (per
/// `belief_leaves.rs:194`), consts via `sub.add` (`PyFrozenSet` per
/// `contracts.rs:1186`); single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(guard_require_train_split, sub)?)?;
    sub.add_function(wrap_pyfunction!(guard_validate_actor_batch, sub)?)?;
    sub.add_function(wrap_pyfunction!(guard_check_wall_leakage, sub)?)?;
    sub.add_function(wrap_pyfunction!(guard_check_split_disjoint, sub)?)?;
    sub.add("GUARD_AUTHORIZED_TRAIN_SPLIT", GUARD_AUTHORIZED_TRAIN_SPLIT)?;
    sub.add(
        "GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE",
        PyFrozenSet::new(py, GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE)?,
    )?;
    let forbidden: Vec<&str> = SHARD_FORBIDDEN_IN_ACTOR
        .iter()
        .copied()
        .chain(GUARD_FORBIDDEN_EXTRA.iter().copied())
        .collect();
    let privileged: Vec<&str> = forbidden
        .iter()
        .copied()
        .chain(GUARD_PRIVILEGED_ONLY.iter().copied())
        .collect();
    sub.add(
        "GUARD_FORBIDDEN_IN_ACTOR_KEYS",
        PyFrozenSet::new(py, forbidden)?,
    )?;
    sub.add("GUARD_PRIVILEGED_KEYS", PyFrozenSet::new(py, privileged)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn staged(keys: &[&str]) -> Vec<(Option<String>, String)> {
        keys.iter()
            .map(|k| (Some((*k).to_string()), format!("'{k}'")))
            .collect()
    }

    #[test]
    fn frozen_tables_match_oracle_literals_and_owner() {
        // Hand-derived from the oracle tables (`oracle_guard.py:28-51`); any
        // drift fails here first. The 6-key base is owner-pinned (never
        // restated); the 3+3 guard extras are literal-pinned below.
        assert_eq!(GUARD_AUTHORIZED_TRAIN_SPLIT, "train");
        assert_eq!(
            GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE,
            ["train", "held_out", "test", "eval"]
        );
        assert_eq!(
            SHARD_FORBIDDEN_IN_ACTOR,
            [
                "hidden_tiles",
                "wall",
                "dead_wall",
                "opponent_hand",
                "privileged",
                "full_world",
            ]
        );
        assert_eq!(
            GUARD_FORBIDDEN_EXTRA,
            ["privileged_label", "wall_remaining", "hidden_tile_counts"]
        );
        assert_eq!(
            GUARD_PRIVILEGED_ONLY,
            ["opponent_concealed", "unrevealed_dora", "wait_tiles"]
        );
        // Combined cardinalities: 6 + 3 forbidden, 9 + 3 privileged.
        assert_eq!(
            SHARD_FORBIDDEN_IN_ACTOR.len() + GUARD_FORBIDDEN_EXTRA.len(),
            9
        );
        assert_eq!(
            SHARD_FORBIDDEN_IN_ACTOR.len()
                + GUARD_FORBIDDEN_EXTRA.len()
                + GUARD_PRIVILEGED_ONLY.len(),
            12
        );
        // Forbidden is a strict subset of privileged (oracle checks both sets
        // with the same message, so the union check preserves behavior).
        for key in SHARD_FORBIDDEN_IN_ACTOR
            .iter()
            .chain(GUARD_FORBIDDEN_EXTRA.iter())
        {
            assert!(
                is_privileged(key),
                "forbidden key missing from privileged: {key}"
            );
            assert!(
                is_forbidden_in_actor(key),
                "forbidden key rejected by its own gate: {key}"
            );
        }
        for key in GUARD_PRIVILEGED_ONLY {
            assert!(is_privileged(key), "oracle-only key missing: {key}");
            assert!(
                !is_forbidden_in_actor(key),
                "oracle-only key must not be actor-forbidden: {key}"
            );
        }
        assert!(!is_privileged("decision_id"));
        assert!(!is_forbidden_in_actor("actor_observation"));
    }

    #[test]
    fn train_split_core_matches_oracle_gate() {
        // Hand-derived from `_require_train_split` (`oracle_guard.py:55-59`):
        // only "train" passes; the message is covered by the pyfn (needs the
        // live `repr` stage), the core pins the predicate.
        assert!(train_split_ok(Some("train")));
        assert!(!train_split_ok(Some("held_out")));
        assert!(!train_split_ok(Some("test")));
        assert!(!train_split_ok(Some("Train")));
        assert!(!train_split_ok(Some("")));
        assert!(!train_split_ok(None));
    }

    #[test]
    fn batch_core_matches_oracle_messages_and_order() {
        // Hand-derived from `validate_actor_batch_no_privileged`
        // (`oracle_guard.py:62-82`): clean passes; top-level forbidden and
        // privileged-only keys share the `actor batch` text (forbidden wins
        // the order); nested keys use the `actor_observation` text; non-`str`
        // keys skip like the oracle membership test.
        let clean = staged(&["decision_id", "seat", "split"]);
        assert_eq!(validate_actor_batch_core(&clean, &[]), Ok(()));
        assert_eq!(validate_actor_batch_core(&[], &clean), Ok(()));
        let top_forbidden = staged(&["decision_id", "wall"]);
        assert_eq!(
            validate_actor_batch_core(&top_forbidden, &[]),
            Err("actor batch contains privileged field 'wall'".to_string())
        );
        let top_privileged_only = staged(&["wait_tiles"]);
        assert_eq!(
            validate_actor_batch_core(&top_privileged_only, &[]),
            Err("actor batch contains privileged field 'wait_tiles'".to_string())
        );
        let nested = staged(&["opponent_concealed"]);
        assert_eq!(
            validate_actor_batch_core(&clean, &nested),
            Err("actor_observation contains privileged field 'opponent_concealed'".to_string())
        );
        // Top-level wins over nested (oracle checks top first).
        assert_eq!(
            validate_actor_batch_core(&top_forbidden, &nested),
            Err("actor batch contains privileged field 'wall'".to_string())
        );
        // Non-str keys skip (oracle `in` test is False for them).
        let mixed = vec![
            (None, "42".to_string()),
            (Some("seat".to_string()), "'seat'".to_string()),
        ];
        assert_eq!(validate_actor_batch_core(&mixed, &mixed), Ok(()));
    }

    #[test]
    fn leakage_cores_match_oracle_texts() {
        // Hand-derived from `check_wall_leakage` (`oracle_guard.py:114-128`)
        // and `check_split_disjoint` (`oracle_guard.py:131-141`): disjoint
        // (including empty) passes; overlap reports the deduped count plus
        // the first 3 sorted ids in Python-repr style.
        let train = vec!["wall-001".to_string(), "wall-002".to_string()];
        let held = vec!["wall-003".to_string()];
        assert_eq!(wall_leakage_error(&train, &held), Ok(()));
        let empty: Vec<String> = vec![];
        assert_eq!(wall_leakage_error(&empty, &held), Ok(()));
        assert_eq!(split_disjoint_error(&train, &held), Ok(()));
        let overlap_train = vec!["wall-002".to_string(), "wall-001".to_string()];
        let overlap_held = vec!["wall-002".to_string(), "wall-004".to_string()];
        assert_eq!(
            wall_leakage_error(&overlap_train, &overlap_held),
            Err(
                "wall leakage: 1 wall(s) overlap between train and held_out: ['wall-002']"
                    .to_string()
            )
        );
        // Dedup + sort + first-3 truncation (4 overlap, 3 shown).
        let big_train: Vec<String> = ["d", "c", "b", "a", "z"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let big_held: Vec<String> = ["a", "b", "c", "d", "y"]
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            split_disjoint_error(&big_train, &big_held),
            Err("split leakage: 4 decision(s) overlap: ['a', 'b', 'c']".to_string())
        );
        assert_eq!(
            wall_leakage_error(&big_train, &big_held),
            Err(
                "wall leakage: 4 wall(s) overlap between train and held_out: ['a', 'b', 'c']"
                    .to_string()
            )
        );
        assert_eq!(py_str_list(&[]), "[]");
    }
}
