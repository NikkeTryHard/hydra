//! pbrf_forest: frozen PBRF forest-build/allocation math over `hydra-search` owners.
//!
//! DAG: pyo3 + `hydra-feed` (canon owner) + `hydra-search` (`pbrf` carry/verify
//! owners) ONLY (the feed canon edge mirrors `pbrf::canonical_hatch`,
//! `pbrf.rs:151-162`; the search edge mirrors `search_shared`'s
//! `hydra_search::gumbel` edge). FORBIDS: dataclass/forest ops/sampling
//! (`ImmutableForest`, `build_pbrf`, commit-compare, miss-rebuild sampling stay
//! Python in `pbrf_forest.py` / `pbrf_commit.py`), float-sum diagnostics
//! (`_z_hat_for_key` / `_normalized_weights` / `_ess_for_key` stay Python —
//! 3.12 compensated `sum()` vs a naive fold diverge 1 ulp), live-object gates
//! (`_is_target_compatible` packet binding, `_tile_for_successor` stay Python),
//! and digest validation (`make_digest_text` in contracts owns `sha256:` shape).
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `pbrf_spec::register`, `pbrf_spec.rs:121-124`); MAIN calls this from
//! `search::register`. No new entry point, no new consts (`PBRF_PARENTS` /
//! `PBRF_MAX_BATCHES` already ride `search` via `pbrf_spec::register`,
//! `pbrf_spec.rs:125-132`).

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// Map a `hydra-search` failure onto the boundary (mirrors `search::search_err`,
/// `search.rs:180-182`; owned here — `search.rs` is MAIN-owned).
fn forest_err(err: hydra_search::SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Conditional carry log-densities over owned raw weights
/// (`pbrf_forest.py:63-88` via `hydra_search::pbrf::conditional_carry_logps`,
/// `pbrf.rs:61-84`): `ln(raw_i / Z)` with `Z = sum(raw)`; zero/nonfinite `Z`
/// or entry mass is `ZeroMass` (the caller takes the MISS path; Python reshapes
/// it to the byte-identical `ContractError`, including the interpolated `Z`).
/// Entries cross as floats only (provenance stays caller-side); the owner reads
/// `raw_weight` alone, so the dummy shells below are exact.
/// Tolerance: bit-exact with the oracle on the frozen parent counts (`N <= 16`
/// positive weights sum exactly in both folds); general bound 2 ulp (naive
/// owner fold vs 3.12 compensated `sum()`).
fn carry_logps(raw_weights: &[f64]) -> Result<Vec<f64>, hydra_search::SearchError> {
    let entries: Vec<hydra_search::pbrf::ChildEntry> = raw_weights
        .iter()
        .map(|w: &f64| hydra_search::pbrf::ChildEntry {
            parent_id: String::new(),
            successor_world_ref: String::new(),
            successor_delta: String::new(),
            raw_weight: *w,
            target_id: String::new(),
            epoch: 0,
            ancestors: Vec::new(),
            tile: None,
        })
        .collect();
    hydra_search::pbrf::conditional_carry_logps(&entries)
}

/// Conditional carry log-densities (`pbrf_conditional_carry_logps`): attached
/// extraction, ONE `py.detach` with zero Python API inside (`search.rs:1061`
/// shape); `ValueError` on zero/nonfinite mass, never a default. Translators
/// call positionally with the entry `raw_weight` lane.
#[pyfunction]
#[pyo3(signature = (raw_weights,))]
fn pbrf_conditional_carry_logps(py: Python<'_>, raw_weights: Vec<f64>) -> PyResult<Vec<f64>> {
    py.detach(|| carry_logps(&raw_weights)).map_err(forest_err)
}

/// Delta reconstruction check (`pbrf_forest.py:114-180` via
/// `hydra_search::pbrf::verify_delta`, `pbrf.rs:114-146`): `succ =
/// "world_succ:"+sha(parent:tile:aid)[:16]`, `delta =
/// "delta:"+sha("delta:"+parent:tile)[:16]`; the stored tile is authoritative,
/// `None` probes the legacy `0..19` window, then the canonical parent+delta
/// hatch; empty refs read `false`. Bit-exact (sha hex + bool). Compute runs
/// detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (parent_world_ref, successor_world_ref, successor_delta, action_id, tile))]
fn pbrf_verify_delta(
    py: Python<'_>,
    parent_world_ref: String,
    successor_world_ref: String,
    successor_delta: String,
    action_id: u32,
    tile: Option<u32>,
) -> PyResult<bool> {
    Ok(py.detach(|| {
        hydra_search::pbrf::verify_delta(
            &parent_world_ref,
            &successor_world_ref,
            &successor_delta,
            action_id,
            tile,
        )
    }))
}

/// Lowercase hex of raw bytes (mirrors `search::hex_lower`,
/// `search.rs:1501-1512`; the fixed-allocation hash order must render the same
/// hex Python `hashlib.sha256(payload).hexdigest()` produces).
fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0F) as usize] as char);
        i += 1;
    }
    out
}

/// Deterministic fixed-batch allocation over parallel `(aid, pid)` lanes
/// (`pbrf_partition.py:348-379` parity): sorted keys take base batches, the
/// remainder goes in `sha256(canonical_bytes({"aid", "pid"}))` hex order
/// through the feed canon owner (never reimplemented here). `aids`/`pids` ride
/// parallel in caller order; the returned counts align to that order.
/// Bit-exact (integer division + sha hex ordering only — no floats).
/// `total_batches == 0` is `ValueError("total_batches must be positive int")`
/// (byte-identical to the oracle); row-length mismatch and duplicate keys fail
/// closed (Python dicts cannot carry them).
fn allocate_counts(aids: &[u32], pids: &[String], total_batches: u64) -> Result<Vec<u64>, String> {
    if aids.len() != pids.len() {
        return Err(String::from("pbrf fixed_allocate row length mismatch"));
    }
    if total_batches == 0 {
        return Err(String::from("total_batches must be positive int"));
    }
    let n = aids.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a: &usize, b: &usize| {
        (aids[*a], pids[*a].as_str()).cmp(&(aids[*b], pids[*b].as_str()))
    });
    let mut prev: Option<(u32, &str)> = None;
    for idx in order.iter() {
        let key = (aids[*idx], pids[*idx].as_str());
        if prev == Some(key) {
            return Err(String::from("pbrf fixed_allocate duplicate (aid, pid) key"));
        }
        prev = Some(key);
    }
    // proof: `n` is a small batch count, widens exactly into `u64`.
    let width: u64 = u64::try_from(n).map_err(|_| "pbrf batch count outside u64".to_string())?;
    let base = total_batches / width;
    // proof: `rem < width <= n` (small batch count), fits `usize`.
    #[allow(clippy::cast_possible_truncation)]
    let rem: usize = (total_batches % width) as usize;
    let mut rank: Vec<(String, usize)> = Vec::with_capacity(n);
    let mut pos = 0;
    while pos < n {
        let idx = order[pos];
        let mut map = BTreeMap::new();
        map.insert(String::from("aid"), serde_json::json!(aids[idx]));
        map.insert(String::from("pid"), serde_json::json!(pids[idx]));
        let bytes = hydra_feed::canon::canonical_bytes(&map, "pbrf:fixed_allocate")
            .map_err(|err| format!("pbrf fixed_allocate canonical seal rejected: {err:?}"))?;
        let digest = Sha256::digest(bytes.as_slice());
        rank.push((hex_lower(digest.as_slice()), pos));
        pos += 1;
    }
    rank.sort_by(|a: &(String, usize), b: &(String, usize)| a.0.cmp(&b.0));
    let mut extra = vec![false; n];
    let mut k = 0;
    while k < rem {
        extra[rank[k].1] = true;
        k += 1;
    }
    let mut sorted_counts = vec![base; n];
    let mut s = 0;
    while s < n {
        if extra[s] {
            sorted_counts[s] = sorted_counts[s].saturating_add(1);
        }
        s += 1;
    }
    let mut out = vec![0u64; n];
    let mut s2 = 0;
    while s2 < n {
        out[order[s2]] = sorted_counts[s2];
        s2 += 1;
    }
    Ok(out)
}

/// Fixed-batch allocation (`pbrf_fixed_allocate`): attached extraction, ONE
/// `py.detach` with zero Python API inside (`search.rs:1061` shape);
/// `ValueError` on bad batches/rows, never a default. Translators call
/// positionally as `(aids, pids, total_batches)` with parallel lanes in one
/// order; counts return aligned to that order.
#[pyfunction]
#[pyo3(signature = (aids, pids, total_batches))]
fn pbrf_fixed_allocate(
    py: Python<'_>,
    aids: Vec<u32>,
    pids: Vec<String>,
    total_batches: u64,
) -> PyResult<Vec<u64>> {
    py.detach(|| allocate_counts(&aids, &pids, total_batches))
        .map_err(PyValueError::new_err)
}

/// Register the PBRF forest pyfns on the EXISTING `search` submodule (mirrors
/// `pbrf_spec::register`, `pbrf_spec.rs:124-196` — no new consts:
/// `PBRF_PARENTS` / `PBRF_MAX_BATCHES` already ride `search` from there; MAIN
/// calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(pbrf_conditional_carry_logps, sub)?)?;
    sub.add_function(wrap_pyfunction!(pbrf_verify_delta, sub)?)?;
    sub.add_function(wrap_pyfunction!(pbrf_fixed_allocate, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod pbrf_forest_tests {
    use super::*;

    #[test]
    fn carry_matches_head_oracle() {
        // Hand-derived from HEAD (`pbrf_forest.py:63-88`, probe E):
        // `ln(0.25)`, `ln(0.75)` print `-1.3862943611198906`,
        // `-0.2876820724517809`.
        let out = carry_logps(&[1.0, 3.0]).unwrap_or_default();
        assert_eq!(out.len(), 2);
        assert!((out[0] - -1.3862943611198906).abs() <= 1e-15);
        assert!((out[1] - -0.2876820724517809).abs() <= 1e-15);
    }

    #[test]
    fn carry_zero_mass_takes_miss_path() {
        // Zero/nonfinite conditioning supports no population (probes F/G/M).
        assert!(!carry_logps(&[]).is_ok());
        assert!(!carry_logps(&[0.0]).is_ok());
        assert!(!carry_logps(&[1.0, 0.0]).is_ok());
        assert!(!carry_logps(&[-1.0]).is_ok());
        assert!(!carry_logps(&[f64::INFINITY]).is_ok());
        assert!(!carry_logps(&[f64::NAN]).is_ok());
    }

    #[test]
    fn verify_matches_head_oracle() {
        // Hand-derived from HEAD (`pbrf_forest.py:114-180`, probes I-N):
        // parent `p0rld`, tile 8, aid 4 build `world_succ:e6b44408c294ffbd`
        // + `delta:cdb06468dcb4c46b`.
        let parent = "p0rld";
        let succ = "world_succ:e6b44408c294ffbd";
        let delta = "delta:cdb06468dcb4c46b";
        assert!(hydra_search::pbrf::verify_delta(
            parent,
            succ,
            delta,
            4,
            Some(8)
        ));
        assert!(hydra_search::pbrf::verify_delta(
            parent, succ, delta, 4, None
        ));
        assert!(!hydra_search::pbrf::verify_delta(
            parent,
            succ,
            "delta:deadbeefdeadbeef",
            4,
            Some(8)
        ));
        assert!(!hydra_search::pbrf::verify_delta(
            "",
            succ,
            delta,
            4,
            Some(8)
        ));
        assert!(!hydra_search::pbrf::verify_delta(
            parent,
            succ,
            delta,
            4,
            Some(9)
        ));
        assert!(!hydra_search::pbrf::verify_delta(
            parent,
            succ,
            delta,
            5,
            Some(8)
        ));
    }

    #[test]
    fn hex_matches_hashlib() {
        // `sha256(b"abc")[:2]` renders `ba78` in both hex laws.
        let digest = Sha256::digest(b"abc");
        assert_eq!(hex_lower(&digest.as_slice()[..2]), String::from("ba78"));
    }

    #[test]
    fn allocate_matches_head_oracle() {
        // Hand-derived from HEAD (`pbrf_partition.py:348-379`, probes A-D):
        // hash order `53d5… (4,b) < 6a7f… (7,a) < df4a… (4,a)` gives the
        // remainder to `(4,b)` first.
        let aids = vec![4u32, 4, 7];
        let pids = vec![String::from("a"), String::from("b"), String::from("a")];
        assert_eq!(allocate_counts(&aids, &pids, 64), Ok(vec![21u64, 22, 21]));
        assert_eq!(allocate_counts(&aids, &pids, 65), Ok(vec![21u64, 22, 22]));
        // Input order only aligns the output; the math is order-free.
        let aids_rev = vec![7u32, 4, 4];
        let pids_rev = vec![String::from("a"), String::from("a"), String::from("b")];
        assert_eq!(
            allocate_counts(&aids_rev, &pids_rev, 64),
            Ok(vec![21u64, 21, 22])
        );
        assert_eq!(
            allocate_counts(&[1u32, 2], &[String::from("x"), String::from("y")], 64),
            Ok(vec![32u64, 32])
        );
        assert_eq!(allocate_counts(&[], &[], 10), Ok(Vec::new()));
    }

    #[test]
    fn allocate_rejects_byte_identical() {
        // Oracle text (`pbrf_partition.py:360-361`, probe H).
        assert_eq!(
            allocate_counts(&[1u32], &[String::from("x")], 0),
            Err(String::from("total_batches must be positive int"))
        );
        let mismatch = allocate_counts(&[1u32], &[String::from("x"), String::from("y")], 4);
        assert!(!mismatch.is_ok());
        let duplicate = allocate_counts(&[1u32, 1], &[String::from("x"), String::from("x")], 4);
        assert!(!duplicate.is_ok());
    }

    #[test]
    fn allocate_sums_to_total() {
        // Structural invariant the oracle asserts (`pbrf_partition.py:378`).
        let aids = vec![4u32, 4, 7, 9];
        let pids = vec![
            String::from("a"),
            String::from("b"),
            String::from("a"),
            String::from("z"),
        ];
        let out = allocate_counts(&aids, &pids, 64).unwrap_or_default();
        assert_eq!(out.len(), 4);
        assert_eq!(out.iter().sum::<u64>(), 64);
    }
}
