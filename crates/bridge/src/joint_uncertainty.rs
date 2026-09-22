//! joint_uncertainty: frozen Candidate 8 uncertainty-set math + config defaults.
//!
//! DAG: pyo3 ONLY (frozen literals + pure divergence/feasibility math; no
//! feed/shard/search-crate edge — the joint likelihood + normalize lanes
//! already ride `search::joint_posterior_weights` / `joint_normalize_weights`
//! / `joint_posterior_fused` (`search.rs:1418-1596`, reused, never
//! reimplemented here), the canon bytes live `feed::canon` via that fused
//! path, and the live dataclass orchestration stays Python in
//! `joint_uncertainty.py`). FORBIDS: canon/JCS re-printing, digest/hash math
//! (`sha2`/`hashlib` — the owner hashes), live `JointPosterior` /
//! `OpponentTypePolicy` / `FullWorld` IO (info-key assembly, hidden-permutation
//! validators, and the hash-draw sampler stay Python), second weight
//! implementations (the posterior lanes above are reused by reference), float
//! `sum()` compensation (KL accumulates with `+=` like the oracle; TV notes
//! its 1-ulp set-order below), and `ContractError` shaping (the bridge raises
//! `ValueError`; thin Python translators map to `ContractError` with
//! byte-identical oracle text).
//!
//! TABLE ported here — frozen values + pure compute, oracle
//! `python/hydra2/search/joint_uncertainty.py` (HEAD
//! `src/hydra2/search/joint_uncertainty.py`, equals worktree at port time
//! except the `hydra2_replay_rs` -> `hydra2._native` import heads):
//! - `JOINT_UNCERTAINTY_DEFAULT_*` <- `JointTypeWorldConfig` field defaults
//!   (`:417-424`: `rho` `0.15`, `epsilon` `0.05`, `divergence_direction`
//!   `"kl_q_nom"`, `support_class` `"finite_categorical"`, `rationality_rule`
//!   `"quantal_softmax"`, `max_particles` `16`, `calibration_threshold`
//!   `0.05`).
//! - `JOINT_UNCERTAINTY_FEAS_TOL` <- `is_feasible` `div <= rho + 1e-9` window
//!   (`:405`; same `1e-9` audits `contains_nominal` `:354/:361`).
//! - `JOINT_UNCERTAINTY_CORR_REL_TOL` / `JOINT_UNCERTAINTY_CORR_ABS_TOL` <-
//!   `preserve_correlation_check` `isclose(rel_tol=0.05, abs_tol=0.01)`
//!   (`:210`; TABLED fn below, tolerances frozen here for the future claim).
//! - `joint_uncertainty_divergence` <- `UncertaintySet.divergence`
//!   (`:371-398`): `kl_q_nom` iterates `q` order (`pq * ln(pq/pn)`, `inf` when
//!   `pq > 0` and `pn == 0`, skip when `pq <= 0`), `kl_nom_q` iterates
//!   `nominal` order symmetrically, `tv` is `0.5 * sum |q-p|` over the key
//!   union. Positional `(direction, q, nominal)`; `q`/`nominal` ride as
//!   `Vec<(u32, f64)>` preserving the translator's dict insertion order for
//!   the KL arms (bit-parity with the oracle's `+=` loop).
//! - `joint_uncertainty_is_feasible` <- `UncertaintySet.is_feasible`
//!   (`:400-405`): detached `divergence_core` + `is_finite` + `div <= rho +
//!   1e-9` (the `nan`/`inf` arms read exactly like the oracle: non-finite
//!   div is `false`, `nan` rho compares `false`, `inf` rho admits any finite
//!   div). Positional `(direction, q, nominal, rho)`.
//!
//! TABLED (stays Python in `joint_uncertainty.py`, per-name evidence):
//! - `exact_joint_posterior_oracle` (`:52-175`): already bridged —
//!   `search::joint_posterior_weights` runs likelihood + normalize detached;
//!   the remainder is live-world assembly (`world_actor_observation` +
//!   `info_key_for_observation` memoized per `world_ref`), `sum()` mass audits
//!   (compensated `sum()` vs naive fold diverges 1 ulp — wave-8 rule, sums
//!   stay Python), and `JointParticle` provenance rebuild. Reused, never a
//!   second impl.
//! - `hidden_marginalization` (`:178-180`): alias delegating to
//!   `JointPosterior.marginal_theta()` (live dataclass, no math of its own).
//! - `preserve_correlation_check` (`:183-214`): live-`JointPosterior`
//!   orchestration over `(theta, world_ref, weight)` with `marginal_theta()`
//!   renormalization (`tot = sum(out.values()); out[k] /= tot`,
//!   `joint_types.py:524-527`, compensated `sum()` + divide) plus `+=`
//!   world/joint accumulation and the `0.05`/`0.01` `isclose` scan. Repackaging
//!   the renormalization bit-exactly across the compensated/nave divide
//!   exceeds simple pure fns; tolerances above are frozen here, logic stays.
//! - `sequential_joint_update` (`:217-237`): driver looping the oracle (live
//!   posterior threading, not frozen consts + plain values).
//! - `coherent_trajectory` (`:240-292`): hash-draw sampler explicitly OUT —
//!   `L254` wave-2 audit (`sha256(rng_seed + canonical_bytes(weights))` +
//!   `sha256(rng_seed + f"act:{theta}:{key}")` over live worlds/policy tables,
//!   generic hash not CTR `natural_indices`/`sampled_draws`; no pyfn covers
//!   it per recon B4).
//! - `UncertaintySet` / `JointTypeWorldConfig` dataclasses (`:300-349`,
//!   `:413-452`) + `contains_nominal` (`:351-365`, live
//!   `distribution_for` per theta + `1e-9` sum audit) + `is_nonempty`
//!   (`:367-369`, one-line `rho >= 0` gate) + `make_joint_type_world_candidate_spec`
//!   (`:455-537`, digest binding via `contracts::make_digest_text`, budget
//!   assembly, `CandidateSpec` construction): frozen records + live
//!   orchestration stay Python; only their math (`divergence`/`is_feasible`)
//!   and defaults move.
//!
//! NONE-duplication note: `rg` for `JOINT_UNCERTAINTY_|joint_uncertainty_divergence|
//! joint_uncertainty_is_feasible` over `crates/` hits nothing before this file
//! (no owner exists — frozen here with oracle line cites, never recopied
//! elsewhere); `rg` for `candidate8|0\.15|calibration_threshold` over
//! `crates/bridge/src` hits only `joint.rs` test string literals
//! (`"candidate8"`) plus this file (no second defaults site); `rg` for
//! `kl_q_nom|finite_categorical|quantal_softmax` over `crates/search/src`
//! hits nothing (the vocabulary owners are the `JOINT_*` consts in
//! `joint_types.rs:106-110`, referenced, never retyped — this file stores only
//! the single default choice per family); the posterior lanes
//! (`joint_posterior_weights`, `joint_normalize_weights`,
//! `joint_posterior_fused`) are reused by reference (`search.rs:1418-1596`),
//! never reimplemented.
//!
//! Shape per fn: Python validates with the oracle `ContractError` text first
//! (byte-identical HEAD text), then calls positionally for plain
//! non-negative `u32` keys + numeric values only (exotic `bool`/negative/
//! non-`int` domains keep the `hashlib`/`math` fallback below so no new
//! `ContractError` is introduced, mirroring `joint_types_gumbel`
//! `joint_types.rs:71-74`); attached staging here is vacuous (typed
//! `String`/`Vec<(u32, f64)>` extraction) plus the direction match, ONE
//! `py.detach` over owned plain data with zero Python API inside, `ValueError`
//! on any reject (per `contracts.rs:117-123`). TV sums its sorted-union
//! `abs` diffs with a naive fold while the oracle sums its `set`-order union
//! with compensated `sum()`: at most 1 ulp apart (wave-8), tests pin `1e-15`.
//! `ln` vs `math.log` is likewise pinned at `1e-15`.
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `despot_seeds::register`, `despot_seeds.rs:281-286`); no new entry point.
//! MAIN wiring: `pub mod joint_uncertainty;` in `lib.rs` (alphabetical,
//! between the `joint_types` and `local_spec` entries) plus
//! `crate::joint_uncertainty::register(&sub)?;` in `search::register` next to
//! the `joint_types` line (`crates/bridge/src/search.rs:1741`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `despot_seeds::register`
//! (`despot_seeds.rs:281-286`); `sub.add` plain consts mirror
//! `search_shared::register` (`search_shared.rs:69-87`); `#[pyfunction]` +
//! `py.detach` + `ValueError` map mirror `search::gumbel_for_action`
//! (`search.rs:254-274`); `Python::initialize()` in tests mirrors
//! `canon_rng.rs:688-689` (only where Python APIs are touched — pure-core
//! tests below need none).

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Frozen `JointTypeWorldConfig.rho` default (`joint_uncertainty.py:418`).
const JOINT_UNCERTAINTY_DEFAULT_RHO: f64 = 0.15;
/// Frozen `JointTypeWorldConfig.epsilon` default (`joint_uncertainty.py:419`).
const JOINT_UNCERTAINTY_DEFAULT_EPSILON: f64 = 0.05;
/// Frozen `JointTypeWorldConfig.divergence_direction` default (`:420`).
const JOINT_UNCERTAINTY_DEFAULT_DIVERGENCE: &str = "kl_q_nom";
/// Frozen `JointTypeWorldConfig.support_class` default (`:421`).
const JOINT_UNCERTAINTY_DEFAULT_SUPPORT: &str = "finite_categorical";
/// Frozen `JointTypeWorldConfig.rationality_rule` default (`:422`).
const JOINT_UNCERTAINTY_DEFAULT_RATIONALITY: &str = "quantal_softmax";
/// Frozen `JointTypeWorldConfig.max_particles` default (`:423`).
const JOINT_UNCERTAINTY_DEFAULT_MAX_PARTICLES: u64 = 16;
/// Frozen `JointTypeWorldConfig.calibration_threshold` default (`:424`).
const JOINT_UNCERTAINTY_DEFAULT_CALIBRATION: f64 = 0.05;
/// Frozen feasibility window (`is_feasible`, `joint_uncertainty.py:405`).
const JOINT_UNCERTAINTY_FEAS_TOL: f64 = 1e-9;
/// Frozen correlation `rel_tol` (`preserve_correlation_check`, `:210`).
const JOINT_UNCERTAINTY_CORR_REL_TOL: f64 = 0.05;
/// Frozen correlation `abs_tol` (`preserve_correlation_check`, `:210`).
const JOINT_UNCERTAINTY_CORR_ABS_TOL: f64 = 0.01;

/// Collect pairs into a sorted map (last-wins, matching dict uniqueness).
fn to_map(pairs: &[(u32, f64)]) -> BTreeMap<u32, f64> {
    let mut out = BTreeMap::new();
    for (key, val) in pairs {
        out.insert(*key, *val);
    }
    out
}

/// Pure divergence core (`UncertaintySet.divergence`, `:371-398`).
///
/// `q`/`nominal` ride in the translator's dict insertion order for the KL
/// arms (the oracle accumulates with `+=` in that order); the TV arm scans
/// the sorted key union (the oracle scans its `set` union with compensated
/// `sum()` — at most 1 ulp apart). `pq <= 0` contributes nothing in either KL
/// arm; `pq > 0` with `pn == 0` reads as `inf`; `nan` masses fall through
/// exactly like the oracle's chained comparisons. Owned plain data only —
/// detach-safe.
fn divergence_core(
    direction: &str,
    q: &[(u32, f64)],
    nominal: &[(u32, f64)],
) -> Result<f64, String> {
    match direction {
        "kl_q_nom" => {
            let nom = to_map(nominal);
            let mut total = 0.0;
            for (aid, pq) in q {
                if *pq > 0.0 {
                    let pn = nom.get(aid).copied().unwrap_or(0.0);
                    if pn > 0.0 {
                        total += *pq * (*pq / pn).ln();
                    } else if pn == 0.0 {
                        return Ok(f64::INFINITY);
                    }
                }
            }
            Ok(total)
        }
        "kl_nom_q" => {
            let dist = to_map(q);
            let mut total = 0.0;
            for (aid, pn) in nominal {
                if *pn > 0.0 {
                    let pq = dist.get(aid).copied().unwrap_or(0.0);
                    if pq > 0.0 {
                        total += *pn * (*pn / pq).ln();
                    } else if pq == 0.0 {
                        return Ok(f64::INFINITY);
                    }
                }
            }
            Ok(total)
        }
        "tv" => {
            let qmap = to_map(q);
            let nommap = to_map(nominal);
            // Sorted-union scan (the oracle scans its `set` union with
            // compensated `sum()` — at most 1 ulp apart, pinned at `1e-15`).
            let mut keys: Vec<u32> = qmap.keys().copied().chain(nommap.keys().copied()).collect();
            keys.sort_unstable();
            keys.dedup();
            let mut total = 0.0;
            for key in keys {
                let pq = qmap.get(&key).copied().unwrap_or(0.0);
                let pn = nommap.get(&key).copied().unwrap_or(0.0);
                total += (pq - pn).abs();
            }
            Ok(0.5 * total)
        }
        _ => Err(format!("unknown divergence '{direction}'")),
    }
}

/// Pure feasibility core (`UncertaintySet.is_feasible`, `:400-405`).
///
/// Detached `divergence_core` + finite audit + `div <= rho + 1e-9`. Owned
/// plain data only — detach-safe.
fn feasibility_core(
    direction: &str,
    q: &[(u32, f64)],
    nominal: &[(u32, f64)],
    rho: f64,
) -> Result<bool, String> {
    let div = divergence_core(direction, q, nominal)?;
    if !div.is_finite() {
        return Ok(false);
    }
    Ok(div <= rho + JOINT_UNCERTAINTY_FEAS_TOL)
}

/// Divergence over owned vectors (`UncertaintySet.divergence`,
/// `joint_uncertainty.py:371-398`).
///
/// Positional `(direction, q, nominal)` matches the translator order; the
/// translator validates with the oracle text first and only rides the bridge
/// for plain `u32` keys + numeric values. Compute runs detached; `ValueError`
/// on unknown direction, never a default.
#[pyfunction]
#[pyo3(signature = (direction, q, nominal))]
fn joint_uncertainty_divergence(
    py: Python<'_>,
    direction: String,
    q: Vec<(u32, f64)>,
    nominal: Vec<(u32, f64)>,
) -> PyResult<f64> {
    py.detach(|| divergence_core(&direction, &q, &nominal))
        .map_err(PyValueError::new_err)
}

/// Feasibility over owned vectors (`UncertaintySet.is_feasible`,
/// `joint_uncertainty.py:400-405`).
///
/// Positional `(direction, q, nominal, rho)`; single FFI incl. the `1e-9`
/// window. Compute runs detached; `ValueError` on unknown direction, never
/// a default.
#[pyfunction]
#[pyo3(signature = (direction, q, nominal, rho))]
fn joint_uncertainty_is_feasible(
    py: Python<'_>,
    direction: String,
    q: Vec<(u32, f64)>,
    nominal: Vec<(u32, f64)>,
    rho: f64,
) -> PyResult<bool> {
    py.detach(|| feasibility_core(&direction, &q, &nominal, rho))
        .map_err(PyValueError::new_err)
}

/// Register the joint-uncertainty consts + pyfns on the EXISTING `search`
/// submodule (mirrors `despot_seeds::register` on the shared submodule;
/// MAIN calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_RHO",
        JOINT_UNCERTAINTY_DEFAULT_RHO,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_EPSILON",
        JOINT_UNCERTAINTY_DEFAULT_EPSILON,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_DIVERGENCE",
        JOINT_UNCERTAINTY_DEFAULT_DIVERGENCE,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_SUPPORT",
        JOINT_UNCERTAINTY_DEFAULT_SUPPORT,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_RATIONALITY",
        JOINT_UNCERTAINTY_DEFAULT_RATIONALITY,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_MAX_PARTICLES",
        JOINT_UNCERTAINTY_DEFAULT_MAX_PARTICLES,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_DEFAULT_CALIBRATION",
        JOINT_UNCERTAINTY_DEFAULT_CALIBRATION,
    )?;
    sub.add("JOINT_UNCERTAINTY_FEAS_TOL", JOINT_UNCERTAINTY_FEAS_TOL)?;
    sub.add(
        "JOINT_UNCERTAINTY_CORR_REL_TOL",
        JOINT_UNCERTAINTY_CORR_REL_TOL,
    )?;
    sub.add(
        "JOINT_UNCERTAINTY_CORR_ABS_TOL",
        JOINT_UNCERTAINTY_CORR_ABS_TOL,
    )?;
    sub.add_function(wrap_pyfunction!(joint_uncertainty_divergence, sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_uncertainty_is_feasible, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_values_match_head() {
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_RHO, 0.15);
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_EPSILON, 0.05);
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_DIVERGENCE, "kl_q_nom");
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_SUPPORT, "finite_categorical");
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_RATIONALITY, "quantal_softmax");
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_MAX_PARTICLES, 16);
        assert_eq!(JOINT_UNCERTAINTY_DEFAULT_CALIBRATION, 0.05);
        assert_eq!(JOINT_UNCERTAINTY_FEAS_TOL, 1e-9);
        assert_eq!(JOINT_UNCERTAINTY_CORR_REL_TOL, 0.05);
        assert_eq!(JOINT_UNCERTAINTY_CORR_ABS_TOL, 0.01);
    }

    #[test]
    fn divergence_matches_head_oracle() {
        // Hand-derived from HEAD (`pixi run python` oracle,
        // `python/hydra2/search/joint_uncertainty.py:371-398`):
        // q={0:0.7,1:0.3} vs nominal={0:0.5,1:0.5}.
        let q = [(0u32, 0.7f64), (1u32, 0.3f64)];
        let nom = [(0u32, 0.5f64), (1u32, 0.5f64)];
        let kl = divergence_core("kl_q_nom", &q, &nom).expect("kl");
        assert!((kl - 0.08228287850505178).abs() < 1e-15);
        let kln = divergence_core("kl_nom_q", &q, &nom).expect("kln");
        assert!((kln - 0.08717669357238891).abs() < 1e-15);
        let tv = divergence_core("tv", &q, &nom).expect("tv");
        assert!((tv - 0.19999999999999998).abs() < 1e-15);
        // Self-divergence is exactly zero on every arm.
        for direction in ["kl_q_nom", "kl_nom_q", "tv"] {
            let zero = divergence_core(direction, &nom, &nom).expect("self");
            assert!(zero.abs() < 1e-15);
        }
        // Empty inputs read as zero; disjoint KL reads as inf, TV as 1.0.
        let empty: Vec<(u32, f64)> = Vec::new();
        assert_eq!(
            divergence_core("kl_q_nom", &empty, &empty).expect("empty"),
            0.0
        );
        assert_eq!(
            divergence_core("kl_nom_q", &empty, &empty).expect("empty"),
            0.0
        );
        assert_eq!(divergence_core("tv", &empty, &empty).expect("empty"), 0.0);
        let one_a = [(0u32, 1.0f64)];
        let one_b = [(1u32, 1.0f64)];
        assert_eq!(
            divergence_core("kl_q_nom", &one_a, &one_b).expect("disjoint"),
            f64::INFINITY
        );
        assert_eq!(
            divergence_core("kl_nom_q", &one_a, &one_b).expect("disjoint"),
            f64::INFINITY
        );
        assert_eq!(
            divergence_core("tv", &one_a, &one_b).expect("disjoint"),
            1.0
        );
        // Head oracle q_nom pair (tight policy at dummy key):
        // q_far={0:0.99,1:0.01} vs q_nom={0:0.7499..,1:0.2500..}.
        let q_nom = [
            (0u32, 0.7499462065985668f64),
            (1u32, 0.25005379340143313f64),
        ];
        let q_far = [(0u32, 0.99f64), (1u32, 0.01f64)];
        let far = divergence_core("kl_q_nom", &q_far, &q_nom).expect("far");
        assert!((far - 0.24273551931551848).abs() < 1e-15);
        let far_rev = divergence_core("kl_nom_q", &q_far, &q_nom).expect("far rev");
        assert!((far_rev - 0.596683250519453).abs() < 1e-15);
    }

    #[test]
    fn divergence_rejects_match_oracle_text() {
        let q = [(0u32, 0.5f64)];
        let nom = [(0u32, 0.5f64)];
        assert_eq!(
            divergence_core("bad", &q, &nom),
            Err(String::from("unknown divergence 'bad'"))
        );
    }

    #[test]
    fn feasibility_matches_head_oracle() {
        // Hand-derived from HEAD (`:400-405`): far pair exceeds rho=0.15,
        // self pair admits, inf div rejects.
        let q_nom = [
            (0u32, 0.7499462065985668f64),
            (1u32, 0.25005379340143313f64),
        ];
        let q_far = [(0u32, 0.99f64), (1u32, 0.01f64)];
        assert!(!feasibility_core("kl_q_nom", &q_far, &q_nom, 0.15).expect("feas"));
        assert!(feasibility_core("kl_q_nom", &q_nom, &q_nom, 0.15).expect("feas"));
        let one_a = [(0u32, 1.0f64)];
        let one_b = [(1u32, 1.0f64)];
        assert!(!feasibility_core("kl_q_nom", &one_a, &one_b, 0.5).expect("inf"));
        // rho + 1e-9 window: div == rho admits, div just beyond rejects.
        let q = [(0u32, 0.7f64), (1u32, 0.3f64)];
        let nom = [(0u32, 0.5f64), (1u32, 0.5f64)];
        let div = divergence_core("kl_q_nom", &q, &nom).expect("div");
        assert!(feasibility_core("kl_q_nom", &q, &nom, div).expect("edge"));
        assert!(!feasibility_core("kl_q_nom", &q, &nom, div - 2e-9).expect("edge"));
    }
}
