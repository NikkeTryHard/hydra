//! despot_seeds: frozen DESPOT seeding + tie-break + reversal-fixture leaves.
//!
//! DAG: pyo3 + `hydra-search` (`despot::scenario_seed` owner) + `sha2` ONLY
//! (all already bridge dependencies, see `crates/bridge/Cargo.toml:19-25`;
//! no new dependency). FORBIDS: JCS re-printing (the feed `canon` owner,
//! reached through `despot::scenario_seed`, is the single bytes site),
//! wall-clock/RNG (seeds are derived, never drawn), live-object
//! orchestration (successor sampling and node tables stay Python in
//! `despot_result.py` / `despot_act.py` / `_DespotNode`), and packet
//! validation (stays Python, see TABLE below).
//!
//! TABLE ported here (frozen values + pure compute, oracle
//! `python/hydra2/search/despot_core.py` via `hydra_search::despot`,
//! `crates/search/src/despot.rs`):
//! - `despot_scenario_seed_bytes` <- `_scenario_seed_bytes` (`:350-368`)
//!   via the owner `despot::scenario_seed` (feed canon + sha256, never a
//!   second printer); non-empty `str` + non-`bool` `u32` gates run attached,
//!   the derivation runs under ONE detach.
//! - `despot_hash_tie_break_index` <- `_hash_tie_break` (`:371-392`): the
//!   `aid` projection (including the salted-`hash(str(a))` fallback, which
//!   no Rust port could reproduce — same process, same seed) stays
//!   Python-side in the translator; only the deterministic
//!   `sha256(f"{candidate_id}:{aid}")` hex argmin runs here, mirroring the
//!   `StableHash` arm of `despot::best_feasible` (`despot.rs:199-219`)
//!   through `sha2` directly (plain format-string payload, never canon —
//!   the `search.rs:200-208` hex-render precedent).
//! - `despot_proposal_reversal_fixture` <- `proposal_reversal_fixture`
//!   (`:276-340`): frozen `(0.5, 0.5)` / `(0.1, 0.9)` probs and values
//!   (`{0: 0.9, 1: 0.0}`, `{0: 0.0, 1: 0.6}`) with the oracle's exact op
//!   order (IEEE round-trip is load-bearing for `0.09000000000000001`);
//!   first-wins `max`, same keys, same `note`.
//!
//! TABLED (stays Python in `despot_core.py`): `validate_packet_partition`
//! — live-object validator over one-shot iterables (the oracle's mass gate
//! re-iterates `successors`, so an exhausted generator skips the mass
//! check — observed: 0.5+0.4 mass over `iter([...])` does NOT raise —
//! while a staged-`Vec` bridge would raise; `str(s)` fallback on arbitrary
//! objects; `bool` passing as `probability` via `isinstance(_, int)`; and
//! Python-`repr` float interpolation (`1e-09`, `0.9`) in the mass/alias
//! texts, which no bridge owner reproduces).
//!
//! NONE-duplication note: `rg` for `scenario_seed|despot_master_hex`
//! over `crates/bridge/src` hits only the owner call below plus
//! `search::despot_lower_values` (no second seed site); `rg` for
//! `reversal_fixture|tie_break_index` over `crates/feed/src` +
//! `crates/search/src` hits nothing (no owner exists — frozen here with
//! oracle line cites, never recopied elsewhere).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `validate::register`
//! (`crates/bridge/src/validate.rs:111-127`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `despot_lower_values` (`crates/bridge/src/search.rs:1024-1041`);
//! `is_instance_of::<PyBool>` bool-rejection mirrors
//! `contracts::plain_int_value` (`crates/bridge/src/contracts.rs:69-74`);
//! `is_instance_of::<PyString>` + `extract::<String>()` text arm mirrors
//! `rc_parse::stage_opt_str` (`crates/bridge/src/rc_parse.rs:510-517`);
//! `search_err` mirrors `eval::search_err`
//! (`crates/bridge/src/eval.rs:60-63`); `PyBytes::new(py, &out).unbind()`
//! mirrors `contracts::ctr_block` (`crates/bridge/src/contracts.rs:454-455`);
//! `Python::initialize()` in tests mirrors `canon_rng.rs:688-689`.
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `eval_stats::register` on the shared submodule); no new entry point.
//! MAIN wiring: `pub mod despot_seeds;` in `lib.rs` (alphabetical, between
//! `dataset_parse` and `encoder`) plus `crate::despot_seeds::register(&sub)?;`
//! in `search::register` next to the `search_shared` line
//! (`crates/bridge/src/search.rs:1729-1730`).

use hydra_search::SearchError;
use hydra_search::despot as despot_mod;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyModule, PyString};
use sha2::{Digest, Sha256};

/// Map an owner error onto the bridge failure (`ValueError`, never default).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Attached staging helper: non-empty `str` (the oracle hashes `str`
/// payloads; the owner additionally requires non-empty ids).
fn stage_nonempty_str(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<String> {
    if !obj.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(format!(
            "search despot {what} must be str"
        )));
    }
    let text: String = obj
        .extract()
        .map_err(|_| PyValueError::new_err(format!("search despot {what} must be str")))?;
    if text.is_empty() {
        return Err(PyValueError::new_err(format!(
            "search despot {what} must be non-empty"
        )));
    }
    Ok(text)
}

/// Attached staging helper: plain `u32`, `bool` rejected first (it
/// subclasses `int` but canonicalizes as `true`/`false`, never `1`/`0`).
fn stage_u32(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<u32> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "search despot {what} must be u32, not bool"
        )));
    }
    obj.extract::<u32>()
        .map_err(|_| PyValueError::new_err(format!("search despot {what} must be u32")))
}

/// Deterministic 32-byte scenario seed (`despot_core.py:350-368` via the
/// `despot::scenario_seed` feed-canon owner): sha256 over the canonical
/// payload `{candidate_id, case_id, scenario_idx, attempt_id, master}`.
/// Staged attached (str/u32 gates); derivation runs under ONE detach with
/// zero Python API inside; `ValueError` on any reject, never a default.
#[pyfunction]
fn despot_scenario_seed_bytes(
    py: Python<'_>,
    candidate_id: Bound<'_, PyAny>,
    case_id: Bound<'_, PyAny>,
    scenario_idx: Bound<'_, PyAny>,
    attempt_id: Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let cid = stage_nonempty_str(&candidate_id, "candidate_id")?;
    let case = stage_nonempty_str(&case_id, "case_id")?;
    let idx = stage_u32(&scenario_idx, "scenario_idx")?;
    let att = stage_u32(&attempt_id, "attempt_id")?;
    let seed = py
        .detach(|| despot_mod::scenario_seed(&cid, &case, idx, att))
        .map_err(search_err)?;
    Ok(PyBytes::new(py, &seed).unbind())
}

/// Deterministic hex argmin over pre-staged aid texts
/// (`despot_core.py:371-392` core): `sha256(f"{candidate_id}:{aid}")`
/// lowercase hex order, first index wins ties (the oracle's strict-`<`
/// scan). The `aid` projection itself stays translator-side (salted
/// `hash(str(a))` fallback); only owned strings cross the detach.
/// Empty aids fail closed (`ValueError`, never a default) — the
/// translator returns `None` before calling, mirroring the oracle.
#[pyfunction]
fn despot_hash_tie_break_index(
    py: Python<'_>,
    candidate_id: String,
    aids: Vec<String>,
) -> PyResult<usize> {
    if aids.is_empty() {
        return Err(PyValueError::new_err(
            "search despot tie-break aids must be non-empty",
        ));
    }
    let best = py.detach(|| tie_break_core(&candidate_id, &aids));
    Ok(best)
}

/// Pure hex-argmin core (owned plain data only, zero Python API).
fn tie_break_core(candidate_id: &str, aids: &[String]) -> usize {
    let mut best_idx = 0usize;
    let mut best_hex: Option<String> = None;
    for (index, aid) in aids.iter().enumerate() {
        let digest = Sha256::digest(format!("{candidate_id}:{aid}").as_bytes());
        let mut hex = String::with_capacity(64);
        for byte in digest {
            hex.push_str(&format!("{byte:02x}"));
        }
        let replace = match &best_hex {
            None => true,
            Some(current) => hex < *current,
        };
        if replace {
            best_hex = Some(hex);
            best_idx = index;
        }
    }
    best_idx
}

/// Frozen fixture probs/values (`despot_core.py:299-301`).
const FIXTURE_NATURAL: (f64, f64) = (0.5, 0.5);
/// Frozen proposal law, heavily favoring world 1 (`despot_core.py:300`).
const FIXTURE_PROPOSAL: (f64, f64) = (0.1, 0.9);
/// Frozen world values (`despot_core.py:301`): `{0: 0.9, 1: 0.0}`,
/// `{0: 0.0, 1: 0.6}`.
const FIXTURE_VALUES: ((f64, f64), (f64, f64)) = ((0.9, 0.0), (0.0, 0.6));
/// Frozen fixture note (`despot_core.py:339`, em-dash verbatim).
const FIXTURE_NOTE: &str = "unweighted non-natural reverses; weighted restores \u{2014} proves proposal bias without correction is unsafe";

/// Owned fixture outcome (plain data only; dict assembly stays attached).
#[derive(Debug, Clone, Copy)]
struct FixtureOut {
    natural: (f64, f64),
    proposal: (f64, f64),
    weighted: (f64, f64),
    natural_choice: u8,
    proposal_choice: u8,
    weighted_choice: u8,
}

/// Pure fixture core (`despot_core.py:303-329`): oracle `expected` op order
/// per world (`acc[a] + p * v[a]`, actions 0 then 1), weighted correction
/// with `w = pb / qb` left-assoc, first-wins `max`.
fn fixture_core() -> FixtureOut {
    let (b0, b1) = FIXTURE_NATURAL;
    let (q0, q1) = FIXTURE_PROPOSAL;
    let ((v00, v01), (v10, v11)) = FIXTURE_VALUES;
    let mut n0 = 0.0_f64;
    let mut n1 = 0.0_f64;
    n0 += b0 * v00;
    n1 += b0 * v01;
    n0 += b1 * v10;
    n1 += b1 * v11;
    let mut u0 = 0.0_f64;
    let mut u1 = 0.0_f64;
    u0 += q0 * v00;
    u1 += q0 * v01;
    u0 += q1 * v10;
    u1 += q1 * v11;
    let mut w0 = 0.0_f64;
    let mut w1 = 0.0_f64;
    let w_i0: f64 = if q0 > 0.0 { b0 / q0 } else { 0.0 };
    w0 += q0 * w_i0 * v00;
    w1 += q0 * w_i0 * v01;
    let w_i1: f64 = if q1 > 0.0 { b1 / q1 } else { 0.0 };
    w0 += q1 * w_i1 * v10;
    w1 += q1 * w_i1 * v11;
    let pick = |a0: f64, a1: f64| -> u8 { if a1 > a0 { 1 } else { 0 } };
    FixtureOut {
        natural: (n0, n1),
        proposal: (u0, u1),
        weighted: (w0, w1),
        natural_choice: pick(n0, n1),
        proposal_choice: pick(u0, u1),
        weighted_choice: pick(w0, w1),
    }
}

/// Tiny proposal-reversal fixture (`despot_core.py:276-340`): unweighted
/// non-natural mean chooses the wrong action while natural and
/// proposal-weighted means agree. Compute runs detached; only owned
/// scalars cross, dict assembly runs attached with oracle-identical keys.
#[pyfunction]
fn despot_proposal_reversal_fixture(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = py.detach(fixture_core);
    let natural = PyDict::new(py);
    natural.set_item(0, out.natural.0)?;
    natural.set_item(1, out.natural.1)?;
    let proposal = PyDict::new(py);
    proposal.set_item(0, out.proposal.0)?;
    proposal.set_item(1, out.proposal.1)?;
    let weighted = PyDict::new(py);
    weighted.set_item(0, out.weighted.0)?;
    weighted.set_item(1, out.weighted.1)?;
    let doc = PyDict::new(py);
    doc.set_item("natural_mean", &natural)?;
    doc.set_item("proposal_unweighted_mean", &proposal)?;
    doc.set_item("proposal_weighted_mean", &weighted)?;
    doc.set_item("natural_choice", out.natural_choice)?;
    doc.set_item("proposal_unweighted_choice", out.proposal_choice)?;
    doc.set_item("proposal_weighted_choice", out.weighted_choice)?;
    doc.set_item("reversal", out.natural_choice != out.proposal_choice)?;
    doc.set_item(
        "correction_restores",
        out.natural_choice == out.weighted_choice,
    )?;
    doc.set_item("note", FIXTURE_NOTE)?;
    Ok(doc)
}

/// Register the despot-seed leaves on the EXISTING `search` submodule
/// (mirrors `eval_stats::register` on the shared submodule): compute
/// detached, wrap attached; single cdylib, no new entry point. MAIN calls
/// this from `search::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(despot_scenario_seed_bytes, sub)?)?;
    sub.add_function(wrap_pyfunction!(despot_hash_tie_break_index, sub)?)?;
    sub.add_function(wrap_pyfunction!(despot_proposal_reversal_fixture, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lowercase hex renderer for recorded-digest comparison (mirrors the
    /// `StableHash` hex loop in `despot::best_feasible`).
    fn hex_of(seed: &[u8; 32]) -> String {
        let mut out = String::with_capacity(64);
        for byte in seed {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    #[test]
    fn scenario_seed_matches_head_oracle() {
        // Hand-derived from HEAD (`pixi run python` oracle):
        // seed(candidate2_despot_natural, case_nat, idx 0) and (idx 3, attempt 2).
        let first =
            despot_mod::scenario_seed("candidate2_despot_natural", "case_nat", 0, 0).expect("seed");
        assert_eq!(
            hex_of(&first),
            "6b46121caebcebf9b16b1520f57a733139542fd58d4e9e825b22a509353cd10f"
        );
        let bumped =
            despot_mod::scenario_seed("candidate2_despot_natural", "case_nat", 3, 2).expect("seed");
        assert_eq!(
            hex_of(&bumped),
            "a97081cf5315f9d5a57d81d46d8c9a6ae365ad1d2a74387c3fb2fdf42766b839"
        );
        assert_ne!(first, bumped);
        assert_eq!(
            despot_mod::scenario_seed("candidate2_despot_natural", "case_nat", 0, 0).expect("seed"),
            first
        );
    }

    #[test]
    fn tie_break_core_matches_head_oracle() {
        // Hand-derived from HEAD (`_hash_tie_break((5, 12, 7), "tie-c1")`
        // chooses aid 5: hex 91107a.. < ac51de.. < ac8d5b..).
        let aids = ["5".to_string(), "12".to_string(), "7".to_string()];
        assert_eq!(tie_break_core("tie-c1", &aids), 0);
        // Order-independent: same winning aid from the reversed order.
        let flipped = ["7".to_string(), "12".to_string(), "5".to_string()];
        let winner = &flipped[tie_break_core("tie-c1", &flipped)];
        assert_eq!(winner, "5");
    }

    #[test]
    fn fixture_core_matches_head_oracle() {
        // Hand-derived from HEAD (`proposal_reversal_fixture` oracle reprs):
        // natural {0: 0.45, 1: 0.3}, proposal {0: 0.09000000000000001,
        // 1: 0.54}, weighted restores natural; choices 0 / 1 / 0.
        let out = fixture_core();
        assert_eq!(out.natural, (0.45, 0.3));
        assert_eq!(out.proposal, (0.09000000000000001, 0.54));
        assert_eq!(out.weighted, (0.45, 0.3));
        assert_eq!(
            (out.natural_choice, out.proposal_choice, out.weighted_choice),
            (0, 1, 0)
        );
        assert_ne!(out.natural_choice, out.proposal_choice);
        assert_eq!(out.natural_choice, out.weighted_choice);
    }
}
