//! ismcts_core: frozen Candidate 1 ISMCTS core leaves — firewall set, master
//! seed, and tree-key scan.
//!
//! DAG: pyo3 + `hydra-search` `ismcts_driver` ONLY (the 17-name tree-key
//! firewall owner — same edge as `search.rs`, no new dependency;
//! `crates/bridge/Cargo.toml:21` carries it). FORBIDS: canon/digest bytes
//! (the info-key hash rides `ismcts_info_key` via `feed::canon`, never a
//! local hasher), UCT
//! selection math (`uct_select`), vector stubs (`ismcts_model_vector` /
//! `ismcts_terminal_vector`), step hashes (`ismcts_step_hashes`), RNG streams,
//! belief sampling, dataclass construction, and error shaping (the scan
//! returns `bool` and never raises; translators map bridge staging failures
//! to `ContractError` with oracle text).
//!
//! TABLE ported here — frozen values + one pure scan, oracle
//! `python/hydra2/search/ismcts_core.py` (HEAD `src/hydra2/search/
//! ismcts_core.py`, equals worktree at port time):
//! - `ISMCTS_MASTER_SEED` <- `_MASTER_SEED` (`:157`,
//!   `b"wp08b_ismcts_natural_v1"`; fresh — `rg` for `wp08b_ismcts` over
//!   `crates/` hits only this file, so this const is the first Rust surface).
//! - `ISMCTS_FORBIDDEN_IN_TREE_KEY` <- `FORBIDDEN_IN_TREE_KEY` (`:135-155`,
//!   17 names) via `hydra_search::ismcts_driver::FORBIDDEN_IN_TREE_KEY`
//!   (same reference, no literal duplication — canon-wins B3, see
//!   `search.rs` docs).
//! - `ismcts_validate_tree_keys` <- `validate_tree_keys_contain_no_world_id`
//!   (`:333-347`): per-key `world`-substring gate + forbidden-substring scan,
//!   check order verbatim. The crate checks single keys (`ismcts_driver`
//!   info-key/doc gates); the batch scan over caller stringified keys is the
//!   unbridged leaf and lands here.
//!
//! LOGIC staying Python: `info_key_for_observation` (ActorObservation -> doc
//! orchestration over the bridged `ismcts_info_key`), `scalarize_vector`
//! (gumbel-owned surface — PortGumbelCore ports it as `gumbel_scalarize`),
//! `model_vector_for_world` / `terminal_vector_for_world` (digest lanes ride
//! the bridged `ismcts_model_vector` / `ismcts_terminal_vector`; non-digest
//! hash lanes keep their sole Python implementation), `_uct_select` (rides
//! the bridged `uct_select`), `NaturalISMCTSConfig` (frozen dataclass +
//! `__post_init__` validator), `InformationSetNode.mean_vector` /
//! `scalar_mean` (float division beside mutable node state), the
//! `UniformContinuationPolicy` tilt/uniform rows (`_distribution_for` pure
//! oracle — the tilt law is crate-owned by `continuation_sample`) with the
//! `distribution` / `sample` guards (belief-lazy `ActorObservation` checks +
//! live RNG draws are OUT), and
//! `attempt_redeterminize` (always-raises negative control — documented, never
//! ported as a fn that only raises).
//!
//! NONE-owner note: `rg` for `wp08b_ismcts|ISMCTS_MASTER|ismcts_validate_tree`
//! over `crates/bridge/src` + `crates/search/src` + `crates/feed/src` is
//! empty before this file (the `ismcts_driver.rs:42` firewall is the
//! referenced owner, never restated — canon-wins B3, see `search.rs` docs);
//! `joint_types.rs:39-42` TABLEs the 19-name joint firewall which
//! keeps its own Python literal (2 theta-private names, no Rust owner).
//!
//! Shape per fn: caller-stringified rows cross attached, ONE `py.detach(|| ...)`
//! runs the scan over owned strings with zero Python API inside (per
//! `search.rs:360-367`); the `bool` crosses back attached and never raises.
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_shared::register`, `search_shared.rs:59-89`); no new entry point.
//! MAIN wiring: `pub mod ismcts_core;` in `lib.rs` plus
//! `crate::ismcts_core::register(&sub)?;` in `search.rs` next to
//! `crate::joint_types::register(&sub)?;` (`search.rs:1741`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirrors `gumbel_spec::register`
//! (`gumbel_spec.rs:118-138`); ONE `py.detach` with zero Python API inside
//! mirrors `search::uct_select` (`search.rs:360-367`); `PyBytes::new` +
//! `PyFrozenSet::new` mirror `search_shared::register`
//! (`search_shared.rs:59-68`); `PyResult<bool>` mirrors
//! `search::persistence_commit_equals_rebuild` (`search.rs:1370-1391`).

use hydra_search::ismcts_driver::FORBIDDEN_IN_TREE_KEY as DRIVER_FORBIDDEN_IN_TREE_KEY;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFrozenSet, PyModule};

/// Frozen master seed (`ismcts_core.py:157`).
const ISMCTS_MASTER_SEED: &[u8] = b"wp08b_ismcts_natural_v1";

/// Pure tree-key scan over caller-stringified keys (detach-safe): mirrors
/// `validate_tree_keys_contain_no_world_id` (`ismcts_core.py:333-347`) — the
/// `world`-substring gate (`len > 32` counts chars like Python `len(str)`,
/// compare is Unicode-lowercased like `str.lower`), then the
/// driver-owned forbidden-substring scan. Returns `false` on the first leak,
/// never raises.
fn tree_keys_clean_owned(keys: &[String]) -> bool {
    for ks in keys {
        if ks.to_lowercase().contains("world")
            && ks.chars().count() > 32
            && (ks.starts_with("world") || ks.contains("FullWorld"))
        {
            return false;
        }
        let mut i = 0;
        while i < DRIVER_FORBIDDEN_IN_TREE_KEY.len() {
            if ks.contains(DRIVER_FORBIDDEN_IN_TREE_KEY[i]) {
                return false;
            }
            i += 1;
        }
    }
    true
}

/// Batch tree-key firewall (`ismcts_core.py:333-347`): caller-stringified key
/// rows in, `false` on the first world-id / privileged leak. Compute runs
/// detached; only owned strings cross.
#[pyfunction]
#[pyo3(signature = (tree_keys,))]
fn ismcts_validate_tree_keys(py: Python<'_>, tree_keys: Vec<String>) -> PyResult<bool> {
    Ok(py.detach(|| tree_keys_clean_owned(&tree_keys)))
}

/// Register the ISMCTS core leaves on the EXISTING `search` submodule
/// (mirrors `search_shared::register`, `search_shared.rs:59-89`; MAIN calls
/// this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("ISMCTS_MASTER_SEED", PyBytes::new(py, ISMCTS_MASTER_SEED))?;
    sub.add(
        "ISMCTS_FORBIDDEN_IN_TREE_KEY",
        PyFrozenSet::new(py, DRIVER_FORBIDDEN_IN_TREE_KEY.iter().copied())?,
    )?;
    sub.add_function(wrap_pyfunction!(ismcts_validate_tree_keys, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod ismcts_core_tests {
    use super::*;

    #[test]
    fn frozen_values_match_head() {
        assert_eq!(ISMCTS_MASTER_SEED, b"wp08b_ismcts_natural_v1".as_slice());
        assert_eq!(DRIVER_FORBIDDEN_IN_TREE_KEY.len(), 17);
        assert!(DRIVER_FORBIDDEN_IN_TREE_KEY.contains(&"world_id"));
        assert!(DRIVER_FORBIDDEN_IN_TREE_KEY.contains(&"hidden_tiles"));
        assert!(DRIVER_FORBIDDEN_IN_TREE_KEY.contains(&"unrevealed_dora"));
        assert!(!DRIVER_FORBIDDEN_IN_TREE_KEY.contains(&"theta_private"));
    }

    #[test]
    fn tree_key_scan_matches_oracle() {
        // Oracle vectors hand-derived from HEAD
        // (`python/hydra2/search/ismcts_core.py:333-347`).
        let owned =
            |rows: &[&str]| -> Vec<String> { rows.iter().map(|s| (*s).to_owned()).collect() };
        assert!(tree_keys_clean_owned(&owned(&[])));
        assert!(tree_keys_clean_owned(&owned(&["sha256:abc"])));
        assert!(tree_keys_clean_owned(&owned(&[
            "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
        ])));
        // Forbidden substring leaks, even in short keys.
        assert!(!tree_keys_clean_owned(&owned(&["world_id"])));
        assert!(!tree_keys_clean_owned(&owned(&["xxprivilegedyy"])));
        assert!(!tree_keys_clean_owned(&owned(&[
            "sha256:abc",
            "opponent_hand"
        ])));
        // Long raw-world shapes trip the world gate.
        assert!(!tree_keys_clean_owned(&[format!(
            "world{}",
            "x".repeat(40)
        )]));
        assert!(!tree_keys_clean_owned(&[format!(
            "FullWorld{}",
            "y".repeat(40)
        )]));
        // Short world mentions are benign (oracle length gate).
        assert!(tree_keys_clean_owned(&owned(&["world"])));
        // Case-sensitive forbidden scan: "World" (capital W) is not "world_id".
        assert!(tree_keys_clean_owned(&owned(&["World"])));
    }
}
