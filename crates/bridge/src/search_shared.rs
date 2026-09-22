//! search_shared: frozen Batch-3 search leaf consts (Gumbel configs, local-shared).
//!
//! DAG: pyo3 ONLY (frozen literals; no feed/search-crate dependency — the
//! Gumbel selection math lives `hydra-search::gumbel`, the canon bytes live
//! `feed::canon`, and the fail-closed import guards stay Python-side in
//! `local_shared.py`, see `search.rs` canon-wins B3 docs). FORBIDS: selection
//! math, digest computation, RNG streams, belief sampling, file/JSON IO, and
//! builder/validator logic (validating roots stay Python in the dataclass
//! `__post_init__` methods).
//!
//! Single-cdylib tree: registers its consts on the shared
//! `hydra2._native.search` submodule via `register` (mirrors
//! `action_artifact::register` on `contracts`); no new entry point.
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFrozenSet, PyTuple};

/// Deterministic master seed for Candidate 5 local resolving
/// (`local_shared.py:119`).
const LOCAL_RESOLVING_MASTER_SEED: &[u8] = b"wp09d_local_resolving_v1";

/// Substrings rejected inside strategy keys (`local_shared.py:120`), sorted.
const FORBIDDEN_IN_STRATEGY_KEY: [&str; 3] = ["full_hidden", "privileged", "world_id"];

/// Frozen Candidate 6 halving rounds (`gumbel_config.py:36`): single source is
/// `hydra_search::gumbel::DEFAULT_HALVING_ROUNDS` (same value, crate-owned).
/// Frozen Candidate 6 visits per round (`gumbel_config.py:37`): single source is
/// `hydra_search::gumbel::DEFAULT_VISITS_PER_ROUND`.
/// Frozen Candidate 6 descent depth (`gumbel_config.py:38`): single source is
/// `hydra_search::gumbel::DEFAULT_MAX_DEPTH`.
/// Frozen Candidate 6 model-call cap (`gumbel_config.py:39`).
const GUMBEL_MAX_MODEL_CALLS: u32 = 32;
/// Frozen Candidate 6 transition cap (`gumbel_config.py:40`).
const GUMBEL_MAX_TRANSITIONS: u32 = 64;
/// Frozen Candidate 6 tie-break arm (`gumbel_config.py:41`).
const GUMBEL_TIE_BREAK: &str = "lowest_action_id";
/// Frozen Candidate 6 candidate id (`gumbel_config.py:42`).
const GUMBEL_CANDIDATE_ID: &str = "candidate6";
/// Frozen Candidate 6 resource view (`gumbel_config.py:43`).
const GUMBEL_RESOURCE_VIEW: &str = "calls";

/// Frozen PUCT comparator constant (`gumbel_config.py:92`): single source is
/// `hydra_search::gumbel::PUCT_C` (same value, crate-owned).
/// Frozen PUCT comparator descent depth (`gumbel_config.py:93`).
const PUCT_MAX_DEPTH: u32 = 6;
/// Frozen PUCT comparator model-call cap (`gumbel_config.py:94`).
const PUCT_MAX_MODEL_CALLS: u32 = 32;
/// Frozen PUCT comparator transition cap (`gumbel_config.py:95`).
const PUCT_MAX_TRANSITIONS: u32 = 64;
/// Frozen PUCT comparator simulation count (`gumbel_config.py:96`): single source is
/// `hydra_search::gumbel::PUCT_NUM_SIMULATIONS` (same value, crate-owned).
/// Frozen PUCT comparator tie-break arm (`gumbel_config.py:97`).
const PUCT_TIE_BREAK: &str = "lowest_action_id";
/// Frozen PUCT comparator candidate id (`gumbel_config.py:98`).
const PUCT_CANDIDATE_ID: &str = "puct_baseline";
/// Frozen PUCT comparator resource view (`gumbel_config.py:99`).
const PUCT_RESOURCE_VIEW: &str = "calls";

/// Register the Batch-3 search leaf consts on the shared `search` submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "LOCAL_RESOLVING_MASTER_SEED",
        PyBytes::new(py, LOCAL_RESOLVING_MASTER_SEED),
    )?;
    sub.add(
        "FORBIDDEN_IN_STRATEGY_KEY",
        PyFrozenSet::new(py, FORBIDDEN_IN_STRATEGY_KEY)?,
    )?;
    sub.add(
        "GUMBEL_HALVING_ROUNDS",
        hydra_search::gumbel::DEFAULT_HALVING_ROUNDS,
    )?;
    sub.add(
        "GUMBEL_VISITS_PER_ROUND",
        PyTuple::new(py, hydra_search::gumbel::DEFAULT_VISITS_PER_ROUND)?,
    )?;
    sub.add("GUMBEL_MAX_DEPTH", hydra_search::gumbel::DEFAULT_MAX_DEPTH)?;
    sub.add("GUMBEL_MAX_MODEL_CALLS", GUMBEL_MAX_MODEL_CALLS)?;
    sub.add("GUMBEL_MAX_TRANSITIONS", GUMBEL_MAX_TRANSITIONS)?;
    sub.add("GUMBEL_TIE_BREAK", GUMBEL_TIE_BREAK)?;
    sub.add("GUMBEL_CANDIDATE_ID", GUMBEL_CANDIDATE_ID)?;
    sub.add("GUMBEL_RESOURCE_VIEW", GUMBEL_RESOURCE_VIEW)?;
    sub.add("PUCT_C", hydra_search::gumbel::PUCT_C)?;
    sub.add("PUCT_MAX_DEPTH", PUCT_MAX_DEPTH)?;
    sub.add("PUCT_MAX_MODEL_CALLS", PUCT_MAX_MODEL_CALLS)?;
    sub.add("PUCT_MAX_TRANSITIONS", PUCT_MAX_TRANSITIONS)?;
    sub.add(
        "PUCT_NUM_SIMULATIONS",
        hydra_search::gumbel::PUCT_NUM_SIMULATIONS,
    )?;
    sub.add("PUCT_TIE_BREAK", PUCT_TIE_BREAK)?;
    sub.add("PUCT_CANDIDATE_ID", PUCT_CANDIDATE_ID)?;
    sub.add("PUCT_RESOURCE_VIEW", PUCT_RESOURCE_VIEW)?;
    Ok(())
}

#[cfg(test)]
mod search_shared_tests {
    use super::*;

    #[test]
    fn frozen_values_match_python_leaves() {
        assert_eq!(
            LOCAL_RESOLVING_MASTER_SEED,
            b"wp09d_local_resolving_v1".as_slice()
        );
        assert_eq!(
            FORBIDDEN_IN_STRATEGY_KEY,
            ["full_hidden", "privileged", "world_id"]
        );
        assert_eq!(
            (
                hydra_search::gumbel::DEFAULT_HALVING_ROUNDS,
                hydra_search::gumbel::DEFAULT_VISITS_PER_ROUND,
                hydra_search::gumbel::DEFAULT_MAX_DEPTH,
                GUMBEL_MAX_MODEL_CALLS,
                GUMBEL_MAX_TRANSITIONS,
            ),
            (2, [8, 8], 6, 32, 64)
        );
        assert_eq!(
            (GUMBEL_TIE_BREAK, GUMBEL_CANDIDATE_ID, GUMBEL_RESOURCE_VIEW),
            ("lowest_action_id", "candidate6", "calls")
        );
        assert_eq!(
            (
                hydra_search::gumbel::PUCT_C,
                PUCT_MAX_DEPTH,
                PUCT_MAX_MODEL_CALLS,
                PUCT_MAX_TRANSITIONS,
                hydra_search::gumbel::PUCT_NUM_SIMULATIONS,
            ),
            (1.5, 6, 32, 64, 16)
        );
        assert_eq!(
            (PUCT_TIE_BREAK, PUCT_CANDIDATE_ID, PUCT_RESOURCE_VIEW),
            ("lowest_action_id", "puct_baseline", "calls")
        );
    }
}
