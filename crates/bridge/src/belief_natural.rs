//! belief_natural: natural-epoch frozen weights + pure epoch-math leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (pure `std` f64 math; the target-identity hash already routes
//! through the artifact digest owner `of_canonical`, and the epoch corpus tables
//! are owned by `belief_leaves` — same edge as `belief_leaves.rs`, no new
//! dependency). FORBIDS: JCS/sha reimplementation (hashing stays with the feed
//! `canon`/`digest` owners via `of_canonical`), wall-clock/RNG (every
//! `RandomStream` draw stays caller-side; the proposal scan takes the draw as a
//! scalar), live-object orchestration (`BeliefEpoch`/`Particle` construction,
//! corpus filtering/sorting, epoch stores, and the CTR cursor replay stay
//! Python), and `ContractError` shaping (the bridge raises `ValueError`; thin
//! Python translators map to `ContractError`/`ProposalSupportError` with
//! byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/natural.py` (worktree tag at
//! port time `natural.py#D91A`; HEAD `src/hydra2/belief/natural.py` is identical
//! modulo the retired `hydra2_replay_rs` import heads):
//! - `NATURAL_PROPOSAL_FIRST_WEIGHT` <- skewed head weight (`natural.py:472-474`):
//!   first world `0.5`, the rest share `0.5 / (K - 1)`.
//! - `NATURAL_POLICY_LOG_PROB` <- `PolicySet.log_prob` unity factor
//!   (`natural.py:145-148`): `log(1.0) == 0.0`, the kernel applies the exact
//!   0.5/0.5 split once per successor.
//! - `NATURAL_EPOCH_CORPUS_SIZE` <- epoch synthesis width (`natural.py:249-254`):
//!   4 worlds. NOT a new table: the 4 hands are rows `0..4` of the shared
//!   `TINY_CORPUS_OPTIONS` and the wall is the shared `TINY_CORPUS_WALL`
//!   (both owned by `belief_leaves.rs:75-85`); this const pins the prefix
//!   length only, and Python slices the shared tables.
//! - `natural_log_density_for_k` <- uniform `-ln(K)` (`natural.py:400,493,547,644`):
//!   natural `log_target == log_proposal`, proposal `log_target`, conditioned
//!   `log_prob`, and `log_density` share this one fold.
//! - `natural_proposal_probs_for_k` <- skewed construction (`natural.py:469-474`):
//!   `K == 1` yields `[1.0]`, else `[0.5] + [0.5 / (K - 1)] * (K - 1)`.
//! - `natural_proposal_index_for_draw` <- cumulative scan (`natural.py:484-491`):
//!   `cum += p` in index order, first `r < cum` wins, default `K - 1`; the
//!   caller's `rng.random_float()` draw crosses as a scalar.
//! - `natural_log_densities_for` <- per-particle pair (`natural.py:493-494`):
//!   `(-ln(K), ln(p[idx]))` in one call (Rust tuple over `PyTuple` for
//!   `vectorcall`, per the plan's PyO3 performance notes).
//!
//! LOGIC staying Python: the four dataclasses (`BeliefEpoch`/`Particle`/
//! `ProposalSpec`/`PolicySet` validating roots per the `contracts.rs:21-26`
//! rank-3 note) and every `__all__`/Literal, `_validate_finite` (Python-repr
//! error text has no Rust spelling), `_target_doc_for`/`_target_id_for`/
//! `_target_ids_for` (hash ownership already bridged at the artifact layer),
//! `_build_tiny_corpus_for_epoch` assembly (live `FullWorld` construction +
//! `world_id` sort + registry write; sources hands/wall from the shared bridge
//! tables), the `count` gates (positive-int/bool-excluded `ContractError`
//! shaping), the support gates (`ProposalSupportError` region/point texts),
//! `_require_search_bridge`/`_ctr_seed_cursor` (patch-points imported by
//! `kernel.py`/`sampled_kernel.py`), and every sampling loop around the leaves.
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - A `bool` `k` extracts as `0`/`1` through `u64` (the oracle's `len()` never
//!   yields `bool`, and the Python-side `count` gates exclude it); well-typed
//!   callers are unaffected.
//! - `k as f64` rounds exactly like CPython's int-to-double conversion feeding
//!   `math.log`, and `f64::ln` calls the same platform libm, so small-K leaves
//!   are bit-identical; the tests pin the `K = 1..4` bits.
//! - A NaN/+-inf draw flows through the same `<` chain as the oracle (`NaN`
//!   and `+inf` land on `K - 1`, `-inf` on `0`); draws are never validated.
//!
//! Shape per fn: arg-check attached (the `ctr_block` seed-gate precedent at
//! `contracts.rs:447-456`), compute under ONE `py.detach(|| ...)` with zero
//! Python API inside (the `world_id_from_doc` precedent at
//! `belief_leaves.rs:155-160`), errors wrapped attached as `PyValueError`.
//! Consts via `sub.add` (per `contracts.rs:1153-1156`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `belief_leaves.rs:192-229`); no new entry
//! point. MAIN wiring: `pub mod belief_natural;` in `lib.rs` plus
//! `crate::belief_natural::register(&sub)?;` in `contracts.rs` next to
//! `crate::belief_leaves::register(&sub)?;` (`contracts.rs:1295`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Skewed-proposal head weight (`natural.py:472-474`): the first corpus world
/// draws with probability `0.5`, the remaining `K - 1` share the other half.
const NATURAL_PROPOSAL_FIRST_WEIGHT: f64 = 0.5;

/// Policy unity factor (`natural.py:145-148`): `PolicySet.log_prob` returns
/// `log(1.0)`; the packet kernel applies the exact 0.5/0.5 split once per
/// successor, so the policy factor here is unity.
const NATURAL_POLICY_LOG_PROB: f64 = 0.0;

/// Epoch synthesis width (`natural.py:249-254`): 4 worlds, i.e. rows `0..4` of
/// the shared `TINY_CORPUS_OPTIONS` plus the shared `TINY_CORPUS_WALL` (both
/// owned by `belief_leaves`); pins the prefix length, never restates the rows.
const NATURAL_EPOCH_CORPUS_SIZE: usize = 4;

/// Pure uniform-log core: `-ln(K)` for `K >= 1` (mirrors `natural.py:400`).
/// No Python API inside (detach-safe); `K == 0` fails closed, never `-inf`.
fn uniform_log_density(k: u64) -> Result<f64, String> {
    if k == 0 {
        return Err("K must be positive int, got 0".to_string());
    }
    // proof: `K` is a small corpus size (< 2^53), exact; log density tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let k_f: f64 = k as f64;
    Ok(-k_f.ln())
}

/// Pure proposal-distribution core: `[1.0]` for `K == 1`, else `[0.5] +
/// [0.5 / (K - 1)] * (K - 1)` (mirrors `natural.py:469-474`). Valid by
/// construction (every entry finite and positive); the Python-side support
/// gate re-checks before use.
fn proposal_probs_for(k: u64) -> Result<Vec<f64>, String> {
    if k == 0 {
        return Err("K must be positive int, got 0".to_string());
    }
    if k == 1 {
        return Ok(vec![1.0]);
    }
    // proof: `K` is a small corpus size (< 2^53), exact; proposal math tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let k_m1_f: f64 = (k - 1) as f64;
    let rest = (1.0 - NATURAL_PROPOSAL_FIRST_WEIGHT) / k_m1_f;
    // proof: `K` is a small corpus size, fits `usize` on 64-bit.
    #[allow(clippy::cast_possible_truncation)]
    let k_u: usize = k as usize;
    let mut probs = Vec::with_capacity(k_u);
    probs.push(NATURAL_PROPOSAL_FIRST_WEIGHT);
    for _ in 1..k {
        probs.push(rest);
    }
    Ok(probs)
}

/// Pure proposal-scan core: cumulative `cum += p` in index order, first
/// `draw < cum` wins, default `K - 1` (mirrors `natural.py:484-491`, including
/// the NaN/+inf-to-last and -inf-to-first edges of the `<` chain).
fn proposal_index_for(k: u64, draw: f64) -> Result<u64, String> {
    let probs = proposal_probs_for(k)?;
    let mut cum = 0.0_f64;
    let mut idx = k - 1;
    for (i, p) in probs.iter().enumerate() {
        cum += *p;
        if draw < cum {
            idx = i as u64;
            break;
        }
    }
    Ok(idx)
}

/// Pure per-particle log pair core: `(-ln(K), ln(p[idx]))` (mirrors
/// `natural.py:493-494`). Rejects `K == 0` and `idx >= K`, never a default.
fn log_pair_for(k: u64, idx: u64) -> Result<(f64, f64), String> {
    let probs = proposal_probs_for(k)?;
    if idx >= k {
        return Err(format!("idx {idx} out of range for K={k}"));
    }
    // proof: `K` is a small corpus size (< 2^53), exact; log density tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let k_f2: f64 = k as f64;
    // proof: `idx < K` (checked above), fits `usize` on 64-bit.
    #[allow(clippy::cast_possible_truncation)]
    let idx_u: usize = idx as usize;
    Ok((-k_f2.ln(), probs[idx_u].ln()))
}

/// Uniform natural log density over `K` corpus worlds (mirrors the shared
/// `-ln(K)` fold without live `FullWorld`s — callers pass the corpus size,
/// the `PacketSuccessor` precedent). Compute runs detached; `K == 0` is
/// `PyValueError`, never `-inf`.
#[pyfunction]
#[pyo3(signature = (k,))]
fn natural_log_density_for_k(py: Python<'_>, k: u64) -> PyResult<f64> {
    py.detach(|| uniform_log_density(k)).map_err(|e| {
        PyValueError::new_err(format!(
            "belief_natural natural_log_density_for_k rejected: {e}"
        ))
    })
}

/// Skewed proposal distribution over `K` corpus worlds (mirrors the
/// `sample_proposal` construction without the live `RandomStream` — draws stay
/// caller-side). Compute runs detached; `K == 0` is `PyValueError`.
#[pyfunction]
#[pyo3(signature = (k,))]
fn natural_proposal_probs_for_k(py: Python<'_>, k: u64) -> PyResult<Vec<f64>> {
    py.detach(|| proposal_probs_for(k)).map_err(|e| {
        PyValueError::new_err(format!(
            "belief_natural natural_proposal_probs_for_k rejected: {e}"
        ))
    })
}

/// Proposal corpus index for one caller draw (mirrors the `sample_proposal`
/// cumulative scan without the live `RandomStream`). Compute runs detached;
/// `K == 0` is `PyValueError`, the draw itself is never rejected.
#[pyfunction]
#[pyo3(signature = (k, draw))]
fn natural_proposal_index_for_draw(py: Python<'_>, k: u64, draw: f64) -> PyResult<u64> {
    py.detach(|| proposal_index_for(k, draw)).map_err(|e| {
        PyValueError::new_err(format!(
            "belief_natural natural_proposal_index_for_draw rejected: {e}"
        ))
    })
}

/// Per-particle `(log_target_density, log_proposal_density)` pair for corpus
/// index `idx` over `K` worlds (mirrors the `sample_proposal` log fold in one
/// call). Compute runs detached; `K == 0` and `idx >= K` are `PyValueError`.
#[pyfunction]
#[pyo3(signature = (k, idx))]
fn natural_log_densities_for(py: Python<'_>, k: u64, idx: u64) -> PyResult<(f64, f64)> {
    py.detach(|| log_pair_for(k, idx)).map_err(|e| {
        PyValueError::new_err(format!(
            "belief_natural natural_log_densities_for rejected: {e}"
        ))
    })
}

/// Register the natural-epoch leaves on the shared `contracts` submodule
/// (mirrors `belief_leaves.rs:192-229`): arg-check attached, compute detached,
/// wrap attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(natural_log_density_for_k, sub)?)?;
    sub.add_function(wrap_pyfunction!(natural_proposal_probs_for_k, sub)?)?;
    sub.add_function(wrap_pyfunction!(natural_proposal_index_for_draw, sub)?)?;
    sub.add_function(wrap_pyfunction!(natural_log_densities_for, sub)?)?;
    sub.add(
        "NATURAL_PROPOSAL_FIRST_WEIGHT",
        NATURAL_PROPOSAL_FIRST_WEIGHT,
    )?;
    sub.add("NATURAL_POLICY_LOG_PROB", NATURAL_POLICY_LOG_PROB)?;
    sub.add("NATURAL_EPOCH_CORPUS_SIZE", NATURAL_EPOCH_CORPUS_SIZE)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_log_matches_oracle_neglog() {
        // Hand-derived from the oracle fold (`natural.py:400,493,547,644`):
        // -ln(K) via the platform libm, evaluated here on the same box.
        assert_eq!(uniform_log_density(1).unwrap(), -0.0);
        assert_eq!(uniform_log_density(2).unwrap(), -std::f64::consts::LN_2);
        assert_eq!(uniform_log_density(3).unwrap(), -1.0986122886681098_f64);
        assert_eq!(uniform_log_density(4).unwrap(), -1.3862943611198906_f64);
        assert!(uniform_log_density(0).is_err());
    }

    #[test]
    fn proposal_probs_match_oracle_skew() {
        // Hand-derived from the oracle construction (`natural.py:469-474`):
        // [1.0] for K == 1, else 0.5 head with 0.5 / (K - 1) tail sharing.
        assert_eq!(proposal_probs_for(1).unwrap(), vec![1.0]);
        assert_eq!(
            proposal_probs_for(4).unwrap(),
            vec![
                0.5,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666
            ]
        );
        let probs = proposal_probs_for(2).unwrap();
        assert_eq!(probs, vec![0.5, 0.5]);
        assert_eq!(probs.iter().sum::<f64>(), 1.0);
        assert!(proposal_probs_for(0).is_err());
    }

    #[test]
    fn proposal_scan_matches_oracle_cumsum() {
        // Hand-derived from the oracle scan (`natural.py:484-491`): first
        // r < cum wins with K - 1 default, boundary r == 0.5 falls through
        // to index 1, and NaN/+inf land last while -inf lands first.
        assert_eq!(proposal_index_for(4, 0.0).unwrap(), 0);
        assert_eq!(proposal_index_for(4, 0.499999).unwrap(), 0);
        assert_eq!(proposal_index_for(4, 0.5).unwrap(), 1);
        assert_eq!(proposal_index_for(4, 0.6).unwrap(), 1);
        assert_eq!(proposal_index_for(4, 0.999999).unwrap(), 3);
        assert_eq!(proposal_index_for(4, 1.0).unwrap(), 3);
        assert_eq!(proposal_index_for(4, f64::NAN).unwrap(), 3);
        assert_eq!(proposal_index_for(4, f64::INFINITY).unwrap(), 3);
        assert_eq!(proposal_index_for(4, f64::NEG_INFINITY).unwrap(), 0);
        assert_eq!(proposal_index_for(1, 0.999).unwrap(), 0);
        assert!(proposal_index_for(0, 0.5).is_err());
    }

    #[test]
    fn log_pair_matches_oracle_fold() {
        // Hand-derived from the oracle pair (`natural.py:493-494`):
        // (-ln(K), ln(p[idx])) with the K == 4 tail log shared.
        assert_eq!(
            log_pair_for(4, 0).unwrap(),
            (-2.0 * std::f64::consts::LN_2, -std::f64::consts::LN_2)
        );
        assert_eq!(
            log_pair_for(4, 2).unwrap(),
            (-1.3862943611198906_f64, -1.791759469228055_f64)
        );
        assert_eq!(log_pair_for(1, 0).unwrap(), (-0.0, 0.0));
        assert!(log_pair_for(4, 4).is_err());
        assert!(log_pair_for(0, 0).is_err());
    }

    #[test]
    fn frozen_weights_match_oracle_literals() {
        // Byte-level pins of the oracle literals (`natural.py:145-148`,
        // `natural.py:249-255,472-474`); any drift fails here first.
        assert_eq!(NATURAL_PROPOSAL_FIRST_WEIGHT, 0.5);
        assert_eq!(NATURAL_POLICY_LOG_PROB, 0.0);
        assert_eq!(NATURAL_EPOCH_CORPUS_SIZE, 4);
    }
}
