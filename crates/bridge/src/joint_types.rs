//! joint_types: frozen Candidate 8 joint type/world vocabulary + joint-Gumbel draw.
//!
//! DAG: pyo3 + `hydra-search` (`joint` owner) ONLY — the same edge as
//! `search::joint_posterior_weights` (`search.rs:1418-1479`), no new
//! dependency (see `crates/bridge/Cargo.toml:19-25`). FORBIDS: canon/JCS
//! re-printing (info keys seal through `feed::canon` via the existing joint
//! fused path, never here), digest reimplementation (no `sha2`/`hashlib`
//! here — the owner hashes), live dataclass orchestration (the
//! `OpponentTypePolicy` / `JointParticle` / `JointPosterior` records stay
//! Python in `joint_types.py`), world/observation IO (info-key assembly and
//! the hidden-permutation validators stay Python), and second weight
//! implementations (`joint_posterior_weights` / `joint_normalize_weights` /
//! `joint_posterior_fused` already bridge the posterior lanes — reused, never
//! reimplemented here).
//!
//! TABLE ported here — frozen values + pure compute, oracle
//! `python/hydra2/search/joint_types.py` (HEAD `src/hydra2/search/
//! joint_types.py`, equals worktree at port time):
//! - `JOINT_THETA_IDS` <- `THETA_IDS` (`:148` `("tight", "loose")`) via the
//!   owners `joint::THETA_TIGHT` / `joint::THETA_LOOSE` (`joint.rs:22-24`,
//!   referenced, never retyped — mirrors `search_shared.rs:69-74`).
//! - `JOINT_DIVERGENCE_DIRECTIONS` <- `DIVERGENCE_DIRECTIONS` (`:149`),
//!   `JOINT_SUPPORT_CLASSES` <- `SUPPORT_CLASSES` (`:150`),
//!   `JOINT_RATIONALITY_RULES` <- `RATIONALITY_RULES` (`:151`): frozen here
//!   in HEAD iteration order (fresh: no Rust owner — see NONE note).
//! - `JOINT_MASTER_SEED` <- `_MASTER_SEED` (`:153`
//!   `b"wp13_joint_type_world_v1"`, fresh: no Rust owner — see NONE note).
//! - `JOINT_GUMBEL_DOMAIN` <- `_JOINT_GUMBEL_DOMAIN` (`:154`) via the owner
//!   `joint::JOINT_GUMBEL_DOMAIN` (`joint.rs:26`, referenced, never retyped).
//! - `joint_types_gumbel` <- `deterministic_joint_gumbel` (`:162-182`) via
//!   the owner `joint::deterministic_joint_gumbel` (`joint.rs:147-183`):
//!   same `DOMAIN + f"{case}:{seat}:{cand}:{theta}:{action}"` payload, same
//!   `u/G/clamp` pipeline as the root Gumbels. Positional
//!   `(case_id, root_seat, candidate_id, action_id, theta)` matches the
//!   Python keyword order so the translator calls positionally; the owner
//!   takes `(case_id, root_seat, candidate_id, theta, action_id)` and this
//!   file reorders across the detach.
//!
//! TABLED (stays Python in `joint_types.py`): `FORBIDDEN_IN_TREE_KEY` (the
//! 19-name joint firewall — the shared 17 ride the `ismcts_driver` owner,
//! the 2 theta-private names have no Rust owner, and the per-module firewall
//! literal stays the single Python source until its own port claims a
//! `JOINT_`-prefixed name); `OpponentTypePolicy` + `distribution_for` /
//! `log_prob` (frozen dataclass + hash-seeded Dirichlet pseudo-counts — the
//! likelihood lane already rides `joint_posterior_weights` via the private
//! `joint_row_likelihood` core, `search.rs:1187-1265`, reused here by
//! reference, never a second impl); `info_key_for_observation` (live
//! `ActorObservation` + `feed::canon` owner reached through the fused joint
//! path, never a second printer); `validate_hidden_permutation_invariance`
//! + `validate_same_information_equality` (live `FullWorld` orchestration
//!   over caller-side worlds, bool verdicts, never plain values); the joint
//!   hash-draw sampler has no pyfn cover and stays Python per recon B4.
//!
//! NONE-duplication note: `rg` for `wp13|DIVERGENCE|SUPPORT_CLASSES|
//! RATIONALITY_RULES|quantal_softmax|kl_q_nom` over `crates/` hits nothing
//! before this file (no owner exists — frozen here with oracle line cites,
//! never recopied elsewhere); `rg` for `JOINT_` over `crates/bridge/src`
//! hits only `joint.rs` owner consts (`JOINT_GUMBEL_DOMAIN`, `JOINT_SUM_TOL`,
//! `JOINT_TIE_EPS`, `JOINT_CLIP`, `JOINT_CLAMP`, `JOINT_U64_DENOM`) plus this
//! file (no second weights/gumbel site); `THETA_TIGHT|THETA_LOOSE` over
//! `crates/bridge/src` hits only `search.rs:1247-1253` (the likelihood core
//! match arms) plus this file's reference (no retyped `"tight"`/`"loose"`
//! literal beyond the test pins).
//!
//! Shape per fn: Python validates with the oracle `ContractError` text first
//! (byte-identical HEAD text, `joint_types.py:166-173`), then calls
//! positionally; attached staging here is vacuous (typed `String`/`u32`
//! extraction) plus empty/seat guards, ONE `py.detach` over owned plain data
//! with zero Python API inside, `ValueError` on any reject (per
//! `contracts.rs:117-123`). Exotic `action_id` domains the oracle accepts
//! without validation (`bool` -> `"True"` payload, negative, non-`int`)
//! stay Python-side: the translator only rides the bridge for plain
//! non-negative `int` action ids and keeps the `hashlib` fallback for the
//! rest, so no new `ContractError` is introduced.
//!
//! Single-cdylib tree: registers its consts + pyfn on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_shared::register`, `search_shared.rs:59-89`); no new entry point.
//! MAIN wiring: `pub mod joint_types;` in `lib.rs` (alphabetical, between
//! the `ingest`-family and `local_spec` entries) plus
//! `crate::joint_types::register(&sub)?;` in `search::register` next to the
//! `search_shared` line (`crates/bridge/src/search.rs:1729-1730`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `search_shared::register`
//! (`search_shared.rs:59-89`); `let py = sub.py()` mirrors
//! `action_artifact::register` (`action_artifact.rs:42-43`); `sub.add`
//! consts mirror `search_shared::register`; `PyTuple::new` mirrors
//! `search_shared.rs:69-74` (`GUMBEL_VISITS_PER_ROUND`); `PyFrozenSet::new`
//! mirrors `local_spec::register` (`local_spec.rs:273-298`);
//! `PyBytes::new` mirrors `search_shared.rs:60-64`
//! (`LOCAL_RESOLVING_MASTER_SEED`); `#[pyfunction]` + `py.detach` +
//! `search_err` mirror `search::gumbel_for_action`
//! (`search.rs:254-274`).

use hydra_search::SearchError;
use hydra_search::joint as joint_mod;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFrozenSet, PyModule, PyTuple};

/// Frozen theta ids (`joint_types.py:148`): single source is the owner pair
/// `joint::THETA_TIGHT` / `joint::THETA_LOOSE` (`joint.rs:22-24`).
const JOINT_THETA_IDS: [&str; 2] = [joint_mod::THETA_TIGHT, joint_mod::THETA_LOOSE];
/// Frozen divergence vocabulary (`joint_types.py:149`), HEAD iteration order.
const JOINT_DIVERGENCE_DIRECTIONS: [&str; 3] = ["kl_q_nom", "kl_nom_q", "tv"];
/// Frozen support-class vocabulary (`joint_types.py:150`), HEAD iteration order.
const JOINT_SUPPORT_CLASSES: [&str; 2] = ["finite_categorical", "quantal"];
/// Frozen rationality-rule vocabulary (`joint_types.py:151`), HEAD iteration order.
const JOINT_RATIONALITY_RULES: [&str; 2] = ["quantal_softmax", "epsilon_greedy"];
/// Frozen joint master seed (`joint_types.py:153`).
const JOINT_MASTER_SEED: &[u8] = b"wp13_joint_type_world_v1";

/// Map an owner error onto the bridge failure (`ValueError`, never default).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Deterministic joint Gumbel (`joint_types.py:162-182` via
/// `joint::deterministic_joint_gumbel`, `joint.rs:147-183`): same `u/G/clamp`
/// pipeline as the root Gumbels, payload extended with `theta`. Positional
/// `(case_id, root_seat, candidate_id, action_id, theta)` matches the Python
/// keyword order; the owner takes theta fourth and this file reorders across
/// the detach. Python validates with the oracle text first; empty/seat guards
/// here are defense-in-depth with generic text, `ValueError` never a default.
#[pyfunction]
#[pyo3(signature = (case_id, root_seat, candidate_id, action_id, theta))]
fn joint_types_gumbel(
    py: Python<'_>,
    case_id: String,
    root_seat: u32,
    candidate_id: String,
    action_id: u32,
    theta: String,
) -> PyResult<f64> {
    if case_id.is_empty() {
        return Err(PyValueError::new_err(
            "search joint_types_gumbel case_id must be non-empty",
        ));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err(
            "search joint_types_gumbel candidate_id must be non-empty",
        ));
    }
    if root_seat >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    let out = py.detach(|| {
        joint_mod::deterministic_joint_gumbel(&case_id, root_seat, &candidate_id, &theta, action_id)
    });
    out.map_err(search_err)
}

/// Register the joint-types consts + draw on the EXISTING `search` submodule
/// (mirrors `search_shared::register`, `search_shared.rs:59-89`; MAIN calls
/// this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("JOINT_THETA_IDS", PyTuple::new(py, JOINT_THETA_IDS)?)?;
    sub.add(
        "JOINT_DIVERGENCE_DIRECTIONS",
        PyFrozenSet::new(py, JOINT_DIVERGENCE_DIRECTIONS)?,
    )?;
    sub.add(
        "JOINT_SUPPORT_CLASSES",
        PyFrozenSet::new(py, JOINT_SUPPORT_CLASSES)?,
    )?;
    sub.add(
        "JOINT_RATIONALITY_RULES",
        PyFrozenSet::new(py, JOINT_RATIONALITY_RULES)?,
    )?;
    sub.add("JOINT_MASTER_SEED", PyBytes::new(py, JOINT_MASTER_SEED))?;
    sub.add(
        "JOINT_GUMBEL_DOMAIN",
        PyBytes::new(py, joint_mod::JOINT_GUMBEL_DOMAIN),
    )?;
    sub.add_function(wrap_pyfunction!(joint_types_gumbel, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_values_match_head() {
        assert_eq!(JOINT_THETA_IDS, ["tight", "loose"]);
        assert_eq!(JOINT_DIVERGENCE_DIRECTIONS, ["kl_q_nom", "kl_nom_q", "tv"]);
        assert_eq!(JOINT_SUPPORT_CLASSES, ["finite_categorical", "quantal"]);
        assert_eq!(
            JOINT_RATIONALITY_RULES,
            ["quantal_softmax", "epsilon_greedy"]
        );
        assert_eq!(JOINT_MASTER_SEED, b"wp13_joint_type_world_v1".as_slice());
        assert_eq!(
            joint_mod::JOINT_GUMBEL_DOMAIN,
            b"joint_type_world_gumbel_v1".as_slice()
        );
        assert_eq!(joint_mod::THETA_TIGHT, "tight");
        assert_eq!(joint_mod::THETA_LOOSE, "loose");
    }

    #[test]
    fn joint_gumbel_matches_head_oracle() {
        // Hand-derived from HEAD (`pixi run python` oracle,
        // `python/hydra2/search/joint_types.py:162-182`).
        let tight_a0 = joint_mod::deterministic_joint_gumbel(
            "case_determinism_0",
            0,
            "candidate8",
            "tight",
            0,
        )
        .expect("gumbel");
        assert!((tight_a0 - 1.034810067605787).abs() < 1e-15);
        let loose_a0 = joint_mod::deterministic_joint_gumbel(
            "case_determinism_0",
            0,
            "candidate8",
            "loose",
            0,
        )
        .expect("gumbel");
        assert!((loose_a0 - (-0.5005849169687927)).abs() < 1e-15);
        let tight_a1 = joint_mod::deterministic_joint_gumbel(
            "case_determinism_0",
            0,
            "candidate8",
            "tight",
            1,
        )
        .expect("gumbel");
        assert!((tight_a1 - (-0.4293057507184001)).abs() < 1e-15);
        let tight_a2 = joint_mod::deterministic_joint_gumbel(
            "case_determinism_0",
            0,
            "candidate8",
            "tight",
            2,
        )
        .expect("gumbel");
        assert!((tight_a2 - 2.3718583128796475).abs() < 1e-15);
        // Theta perturbs the same arm (matches the `jc` oracle pair).
        let jc_tight =
            joint_mod::deterministic_joint_gumbel("jc", 0, "jcand", "tight", 5).expect("gumbel");
        assert!((jc_tight - 1.1860356661096125).abs() < 1e-15);
        let jc_loose =
            joint_mod::deterministic_joint_gumbel("jc", 0, "jcand", "loose", 5).expect("gumbel");
        assert!((jc_loose - 2.2149196079598723).abs() < 1e-15);
        assert_ne!(jc_tight, jc_loose);
    }

    #[test]
    fn joint_gumbel_rejects_match_owner() {
        // Owner gates (mirrored by the Python translator's oracle text):
        // empty case/candidate, seat 4, unknown theta.
        assert!(joint_mod::deterministic_joint_gumbel("", 0, "c", "tight", 0).is_err());
        assert!(joint_mod::deterministic_joint_gumbel("x", 0, "", "tight", 0).is_err());
        assert!(joint_mod::deterministic_joint_gumbel("x", 4, "c", "tight", 0).is_err());
        assert!(joint_mod::deterministic_joint_gumbel("x", 0, "c", "wide", 0).is_err());
    }
}
