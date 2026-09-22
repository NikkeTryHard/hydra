//! gumbel_core: frozen Candidate 6 root-scalar projection on the shared `search` submodule.
//!
//! DAG: pyo3 ONLY (pure 4-vector projection, so no feed/search-crate owner
//! exists — same edge as `common_roots`, see `crates/bridge/Cargo.toml:25`;
//! no new dependency). FORBIDS: float sums (stay Python-side per the wave-8
//! compensated-`sum()` vs naive-fold 1-ulp lesson — this file copies one
//! lane, never adds), selection math (lives `hydra-search::gumbel` /
//! `hydra-search::ismcts::IsNode::scalar_mean`), digest/hash/canonical bytes
//! (live `feed::canon`/`feed::digest` — never reimplemented here), RNG
//! streams, belief/world construction, and live-object orchestration (all
//! stay Python in `gumbel_core.py`).
//!
//! TABLED (stays Python in `gumbel_core.py`, per-name evidence):
//! - `deterministic_gumbel` / `deterministic_root_gumbels` <- already bridged
//!   as `gumbel_for_action` / `gumbel_roots` (`search.rs:254-299`); reuse,
//!   never second.
//! - `model_vector_for_world` / `terminal_vector_for_world` <- digest shapes
//!   already ride `ismcts_model_vector` / `ismcts_terminal_vector`
//!   (`search.rs:646-672`); the non-digest hash fallback stays Python.
//! - `FORBIDDEN_IN_TREE_KEY` (17 names) <- owned by
//!   `hydra_search::ismcts_driver::FORBIDDEN_IN_TREE_KEY`
//!   (`ismcts_driver.rs:42-60`); PortIsmctsCore exposes the `ISMCTS_*` const,
//!   this file never seconds it.
//! - `info_key_for_observation` <- live `ActorObservation` extraction +
//!   contracts `observation_identity_document` stay Python; the canon+hash
//!   half already rides `ismcts_info_key` (`search.rs:794-800`) with a
//!   different drop set (`legal_mask`+`observation_hash` vs gumbel's
//!   `legal_mask`-only), so a second bridge fn would fork the oracle hash.
//! - `exact_transition` / `learned_rules_transition_rejected` /
//!   `validate_hidden_permutation_invariance` <- belief `make_full_world` /
//!   `world_actor_observation` construction stays Python; the pure math half
//!   already rides `ismcts_transition` with `domain="gumbel"`
//!   (`search.rs:716-785`).
//! - `_actor_to_move` <- already owned as `TinyWorld::actor_to_move`
//!   (`ismcts_driver.rs:477-483`); `_is_terminal` /
//!   `_legal_ids_for_observation` are planner-loop guards with silent
//!   `(0, 1)` fallbacks that must stay Python-visible.
//! - `cached_full_history_agreement` / `validate_packet_partition` <-
//!   trivially-true stubs (`full == cached`, `return True`) with no frozen
//!   compute to own.
//!
//! PORTED here — one pure leaf, oracle
//! `python/hydra2/search/gumbel_core.py:240-251` (HEAD; identical logic in
//! `ismcts_core.py:268-284`, which stays Python per the PortIsmctsCore split
//! — this single `gumbel_scalarize` is the one bridge fn, never seconded):
//! - `gumbel_scalarize` <- `scalarize_vector` (root-seat projection
//!   `vector[root_seat]`, length-4 + seat-range + per-lane finite gates).
//!   Exact copy, no arithmetic: float tolerance is zero ulps (bit-identical).
//!
//! Shape per fn: attached staging (owned `Vec<f64>` + `u32` cross the detach,
//! memcpy-attached semantics like the columnar descriptors) -> ONE
//! `py.detach(|| ...)` over owned plain data with zero Python API inside
//! (per `search.rs:254-274` `gumbel_for_action`) -> attached wrap as
//! `PyValueError` (per `search.rs:180-182` `search_err`). Python keeps its
//! verbatim `ContractError` validation ahead of the call and maps bridge
//! failures to `ContractError(f"gumbel bridge scalarize failed: {exc}")`,
//! so the oracle sentences stay byte-identical.
//!
//! Single-cdylib tree: registers on the EXISTING `hydra2._native.search`
//! submodule via `register` (mirrors `search_shared::register`,
//! `search_shared.rs:59-89` and `gumbel_spec::register`,
//! `gumbel_spec.rs:118-138`); no new entry point. MAIN wiring:
//! `pub mod gumbel_core;` in `lib.rs` (after `gumbel_spec`) plus
//! `crate::gumbel_core::register(&sub)?;` in `search::register` next to
//! `crate::gumbel_spec::register(&sub)?;` (`search.rs:1736`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirrors `search_shared::register`
//! (`search_shared.rs:59-89`); attached-extract + ONE `py.detach` with zero
//! Python API inside mirrors `search.rs::gumbel_for_action`
//! (`search.rs:254-274`); `PyValueError::new_err` mapping mirrors
//! `search.rs::search_err` (`search.rs:180-182`); pure gate returning
//! `Result<_, String>` mirrors `search.rs::check_budget_args`
//! (`search.rs:125-138`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Pure root-scalar gate (`gumbel_core.py:240-251`): copy lane `root_seat`
/// out of a 4-vector. Length, seat-range, and per-lane finite gates fail
/// closed; the copy itself is exact (zero-ulp, no arithmetic). Pure and
/// detach-safe.
fn check_scalarize(vector: &[f64], root_seat: u32) -> Result<f64, String> {
    if vector.len() != 4 {
        return Err(format!(
            "gumbel scalarize vector must hold 4 entries, got {}",
            vector.len()
        ));
    }
    if root_seat >= 4 {
        return Err(format!("seat must be 0..3, got {root_seat}"));
    }
    let mut i = 0;
    while i < vector.len() {
        if !vector[i].is_finite() {
            return Err(format!("gumbel scalarize vector[{i}] must be finite"));
        }
        i += 1;
    }
    Ok(vector[root_seat as usize])
}

/// Root scalar `s_i` — projection onto the root seat
/// (`gumbel_core.py:240-251` via `check_scalarize`).
///
/// Pure copy, exact (no arithmetic, zero-ulp vs the oracle's
/// `return vector[root_seat]`). Compute runs detached; only owned scalars
/// cross.
#[pyfunction]
#[pyo3(signature = (vector, root_seat))]
fn gumbel_scalarize(py: Python<'_>, vector: Vec<f64>, root_seat: u32) -> PyResult<f64> {
    let out = py.detach(|| check_scalarize(&vector, root_seat));
    out.map_err(PyValueError::new_err)
}

/// Register the Gumbel-core leaf on the EXISTING `search` submodule
/// (mirrors `search_shared::register`, `search_shared.rs:59-89`; MAIN calls
/// this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(gumbel_scalarize, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod gumbel_core_tests {
    use super::*;

    #[test]
    fn scalarize_projects_root_seat_exact() {
        // Oracle vectors hand-derived from HEAD (`gumbel_core.py:240-251`):
        // pure copy, so every seat reads back bit-identical.
        let vector = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(check_scalarize(&vector, 0).unwrap(), 1.0);
        assert_eq!(check_scalarize(&vector, 1).unwrap(), 2.0);
        assert_eq!(check_scalarize(&vector, 2).unwrap(), 3.0);
        assert_eq!(check_scalarize(&vector, 3).unwrap(), 4.0);
        let mixed = [0.5, -1.5, 3.25, 0.0];
        assert_eq!(check_scalarize(&mixed, 2).unwrap(), 3.25);
    }

    #[test]
    fn scalarize_rejects_shapes_fail_closed() {
        // Length gate (`gumbel_core.py:244-245` oracle:
        // "vector must be length 4, got ...").
        assert!(check_scalarize(&[1.0, 2.0, 3.0], 0).is_err());
        assert!(check_scalarize(&[1.0, 2.0, 3.0, 4.0, 5.0], 0).is_err());
        assert!(check_scalarize(&[], 0).is_err());
        // Seat gate (`gumbel_core.py:246-247` oracle: seat `0..3`).
        assert!(check_scalarize(&[1.0, 2.0, 3.0, 4.0], 4).is_err());
        assert!(check_scalarize(&[1.0, 2.0, 3.0, 4.0], u32::MAX).is_err());
        // Finite gate (`gumbel_core.py:248-250` oracle:
        // "vector[{idx}] must be finite, got ...").
        assert!(check_scalarize(&[f64::NAN, 0.0, 0.0, 0.0], 0).is_err());
        assert!(check_scalarize(&[0.0, f64::INFINITY, 0.0, 0.0], 1).is_err());
        assert!(check_scalarize(&[0.0, 0.0, f64::NEG_INFINITY, 0.0], 2).is_err());
    }
}
