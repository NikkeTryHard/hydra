//! loop_state: supervised-loop firewall vocabulary + sha gate on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (frozen literals + one detached shape check; no feed/search
//! edge, no new dependency). FORBIDS: JCS/sha reimplementation (digests ride
//! the feed `digest` owner through `canon_rng.sha256_hex`; this module only
//! gates the `sha256:<64>` shape), IO/fsync/temp discipline (the atomic
//! best-ckpt publish stays Python as OS authority), live-object orchestration
//! (the `TrainingState`/`TrainingLoopConfig` dataclasses, ckpt
//! publish/verify-closed roots, and every torch caller stay Python), and
//! `ContractError` shaping (the bridge raises `ValueError`; thin Python
//! translators map to `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/training/loop_state.py`:
//! - const `LOOP_FORBIDDEN_BATCH_KEYS` <- `FORBIDDEN_BATCH_KEYS`
//!   (`loop_state.py:22-34`: 9 privileged field names, WP-05B firewall).
//! - const `LOOP_REQUIRED_MANIFEST_KEYS` <- `_REQUIRED_MANIFEST_KEYS`
//!   (`loop_state.py:37-48`: 10 `*_hash` manifest keys, SPEC order).
//! - `loop_require_sha256` <- `_require_sha256` (`loop_state.py:51-54`):
//!   owned scalars cross (`name` + `value`, never live state — the
//!   `PacketSuccessor` precedent at `contracts.rs:259-261`); the shape test
//!   is the oracle-loose `startswith("sha256:") + len == 71` (NO hex check,
//!   unlike the strict `contracts.rs:99-107` `is_digest_shape` —
//!   `"sha256:" + "Z" * 64` passes here by oracle parity, pinned below).
//!
//! LOGIC staying Python: `TrainingState` (mutable slots dataclass),
//! `TrainingLoopConfig` (frozen dataclass + `validate()`/`objective_weights`
//! roots), `_best_ckpt_digest_for` (one-line feed-owner delegate),
//! `_atomic_publish_best` (mkdir/read/write/fsync/replace — OS authority),
//! `_verify_best_ckpt` (filesystem probe + digest compare), every `__all__`.
//!
//! NONE-owner note: `rg` for `LOOP_FORBIDDEN|LOOP_REQUIRED|
//! loop_require_sha256|loop_state` over `crates/bridge/src` is empty before
//! this file (the `search_shared.rs:22` `FORBIDDEN_IN_STRATEGY_KEY` and the
//! `contracts.rs:99-160` digest gates are unrelated vocabularies —
//! strategy-key substrings and strict lowercase-hex shapes, never the
//! loop firewall or its loose gate); the replay twin
//! (`replay_state.py:23-56`) keeps its own `FORBIDDEN_REPLAY_KEYS` (11
//! entries) and stays Python until its own port claims names.
//!
//! Shape per fn: attached staging (exact `repr(value)` text + `str(name)`
//! via the live Python API, so `{value!r}` renders byte-exact per
//! `rc_require.rs:60-65` + the `canon_rng.rs:322-323` round-trip) -> ONE
//! `py.detach(|| ...)` over owned plain data with zero Python API inside
//! (per `contracts.rs:434-437`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Consts via `sub.add` (`PyFrozenSet` per
//! `contracts.rs:1186`, `PyTuple` per `contracts.rs:1156`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `belief_leaves.rs:192-229`); no new entry
//! point. MAIN wiring: `pub mod loop_state;` in `lib.rs` plus
//! `crate::loop_state::register(&sub)?;` in `contracts.rs` next to the
//! `crate::belief_kernel::register(&sub)?;` line.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyFrozenSet, PyModule, PyTuple};

/// Privileged batch field names (`loop_state.py:22-34`, WP-05B firewall).
const LOOP_FORBIDDEN_BATCH_KEYS: [&str; 9] = [
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
    "privileged_label",
    "wall_remaining",
    "hidden",
];

/// Checkpoint manifest keys (`loop_state.py:37-48`, SPEC order, never reordered).
const LOOP_REQUIRED_MANIFEST_KEYS: [&str; 10] = [
    "run_spec_hash",
    "model_spec_hash",
    "optimizer_spec_hash",
    "scheduler_spec_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
];

/// Pure shape core: the oracle-loose `sha256:` prefix + 71-char gate
/// (mirrors `loop_state.py:52` exactly — no hex validity check). Char count
/// (not byte length) so non-ASCII inputs agree with the oracle's `len()`.
/// No Python API inside (detach-safe); the caller maps `false` to the
/// oracle-identical message.
fn sha_shape_ok(text: &str) -> bool {
    text.starts_with("sha256:") && text.chars().count() == 71
}

/// Oracle-identical error text (`loop_state.py:53`): `repr` is the
/// attached-staged Python `repr(value)`, never a Rust rendering.
fn require_message(name: &str, repr: &str) -> String {
    format!("{name} must be sha256:<64 hex>, got {repr}")
}

/// Validated digest text (mirrors `_require_sha256` bit-for-bit: any
/// non-`str` value rejects with its `{value!r}` rendering; `name` stages via
/// `str()` so non-`str` names render exactly like the oracle f-string).
/// Translators call positionally; the bridge raises `ValueError`, Python maps
/// to `ContractError` with byte-identical text. Compute runs detached with
/// zero Python API inside; counter-free deterministic, never wall-clock.
#[pyfunction]
#[pyo3(signature = (name, value))]
fn loop_require_sha256(
    py: Python<'_>,
    name: Bound<'_, PyAny>,
    value: Bound<'_, PyAny>,
) -> PyResult<String> {
    let name_text = name.str()?.to_str()?.to_owned();
    let repr_text = value.repr()?.to_str()?.to_owned();
    let staged: Option<String> = value.extract().ok();
    let ok = py.detach(|| staged.as_deref().is_some_and(sha_shape_ok));
    if ok {
        match staged {
            Some(text) => Ok(text),
            None => Err(PyValueError::new_err(require_message(
                &name_text, &repr_text,
            ))),
        }
    } else {
        Err(PyValueError::new_err(require_message(
            &name_text, &repr_text,
        )))
    }
}

/// Register the loop-state tables + sha gate on the shared `contracts`
/// submodule (mirrors `belief_leaves.rs:192-229`): `let py = sub.py();`
/// (per `contracts.rs:1115`), fns via `wrap_pyfunction!(f, sub)` (per
/// `belief_leaves.rs:194`), consts via `sub.add` (per
/// `contracts.rs:1153-1156`); single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(loop_require_sha256, sub)?)?;
    sub.add(
        "LOOP_FORBIDDEN_BATCH_KEYS",
        PyFrozenSet::new(py, LOOP_FORBIDDEN_BATCH_KEYS)?,
    )?;
    sub.add(
        "LOOP_REQUIRED_MANIFEST_KEYS",
        PyTuple::new(py, LOOP_REQUIRED_MANIFEST_KEYS)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Hand-derived from the oracle literals (`loop_state.py:22-48`); any
        // drift fails here first. Sorted-order pin mirrors the
        // `test_firewall_vocabulary_pinned` oracle expectation.
        assert_eq!(
            LOOP_FORBIDDEN_BATCH_KEYS,
            [
                "hidden_tiles",
                "wall",
                "dead_wall",
                "opponent_hand",
                "privileged",
                "full_world",
                "privileged_label",
                "wall_remaining",
                "hidden",
            ]
        );
        let mut sorted = LOOP_FORBIDDEN_BATCH_KEYS.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            [
                "dead_wall",
                "full_world",
                "hidden",
                "hidden_tiles",
                "opponent_hand",
                "privileged",
                "privileged_label",
                "wall",
                "wall_remaining",
            ]
        );
        assert_eq!(
            LOOP_REQUIRED_MANIFEST_KEYS,
            [
                "run_spec_hash",
                "model_spec_hash",
                "optimizer_spec_hash",
                "scheduler_spec_hash",
                "environment_hash",
                "rules_hash",
                "utility_manifest_hash",
                "action_schema_hash",
                "observation_schema_hash",
                "dataset_manifest_hash",
            ]
        );
    }

    #[test]
    fn sha_shape_ok_matches_loose_oracle_gate() {
        // Hand-derived from `loop_state.py:52` (`startswith("sha256:")` and
        // `len == 71` ONLY). The uppercase-hex row is the deliberate
        // divergence from the strict `contracts.rs:99-107` gate: the oracle
        // accepts it, so this leaf accepts it too.
        assert!(sha_shape_ok(&format!("sha256:{}", "a".repeat(64))));
        assert!(sha_shape_ok(&format!("sha256:{}", "Z".repeat(64))));
        assert!(sha_shape_ok(&format!("sha256:{}", "é".repeat(64))));
        assert!(!sha_shape_ok(&format!("SHA256:{}", "a".repeat(64))));
        assert!(!sha_shape_ok("sha256:abc"));
        assert!(!sha_shape_ok(""));
        assert!(!sha_shape_ok(&format!("sha256:{}", "a".repeat(63))));
        assert!(!sha_shape_ok(&format!("sha256:{}", "a".repeat(65))));
        assert!(!sha_shape_ok(&format!("xsha256:{}", "a".repeat(64))));
    }

    #[test]
    fn require_message_is_byte_identical_to_oracle() {
        // Hand-derived from `loop_state.py:53`
        // (`f"{name} must be sha256:<64 hex>, got {value!r}"`); the Python
        // translator re-raises `str(exc)` verbatim.
        assert_eq!(
            require_message("run_spec_hash", "'abc'"),
            "run_spec_hash must be sha256:<64 hex>, got 'abc'"
        );
        assert_eq!(
            require_message("digest", "None"),
            "digest must be sha256:<64 hex>, got None"
        );
    }
}
