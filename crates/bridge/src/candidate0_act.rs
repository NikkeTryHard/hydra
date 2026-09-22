//! candidate0_act: frozen candidate0 act-path values + pure compute leaves.
//!
//! DAG: pyo3 + std ONLY (no new dependency; `crates/bridge/Cargo.toml:18-25`
//! lists the bridge deps and none of feed/shard/search is needed here).
//! FORBIDS: torch math (the `candidate0` encode/evaluate/masked-policy path
//! is a HARD torch island and stays Python), live-object orchestration (the
//! `ActionContext` assembly and `FrozenCandidate0` wrapper stay Python in
//! `candidate0_act.py`), hash/digest work (nothing hashed here), and any
//! second act impl (`act_batch` + `ActJudge` already cover the act path).
//!
//! TABLE ported here (frozen values + pure compute, oracle
//! `python/hydra2/search/candidate0_act.py`):
//! - `candidate0_sorted_concealed` <- `:43-47` (`tuple(sorted({*concealed,
//!   own_drawn}))`, drawn-present arm only; the `None` arm keeps observation
//!   order in Python and never crosses).
//! - `candidate0_has_offer` <- `:61-62` (`phase in ("discard_response",
//!   "kan_response")`).
//! - `candidate0_offered_by_fallback` <- `:69-70` (`(int(actor) + 3) % 4`;
//!   `rem_euclid` reproduces Python floored `%` for negative seats).
//! - `CANDIDATE0_HISTORY_LIMIT` <- `:395-396` (`> 16` / `[-16:]` trim);
//!   `CANDIDATE0_NUM_SEATS` pins the `:70` modulus.
//!
//! TABLED (stays Python in `candidate0_act.py`): `candidate0` (torch island:
//! encoder/model/masked-policy/frozen-choice/decode/telemetry), the
//! `ActionContext` assembly over live observation objects (arbitrary tile
//! payloads, history scan, `contextlib.suppress` fallback),
//! `FrozenCandidate0` (spec-identity gate + history ring), and the
//! `frozen_choice` call itself (torch-bound, see
//! `candidate0_frozen.py:19,45,101` TORCH_KEEP).
//!
//! NONE-duplication note: `rg` for
//! `offered_by|sorted_concealed|has_offer|HISTORY_LIMIT|NUM_SEATS` over
//! `crates/bridge/src` hits nothing outside this file (no owner exists —
//! frozen here with oracle line cites, never recopied elsewhere).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `despot_seeds::register`
//! (`crates/bridge/src/despot_seeds.rs:281-286`); `#[pyfunction]` +
//! owned-extract / `py.detach` / plain-return shape mirrors
//! `despot_hash_tie_break_index` (`crates/bridge/src/despot_seeds.rs:145-159`);
//! frozen `const` leaves mirror `FIXTURE_NATURAL`
//! (`crates/bridge/src/despot_seeds.rs:183-191`).
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `despot_seeds::register` on the shared submodule); no new entry point.
//! MAIN wiring: `pub mod candidate0_act;` in `lib.rs` (alphabetical, between
//! `belief_natural` and `canon_rng`) plus
//! `crate::candidate0_act::register(&sub)?;` in `search::register` after the
//! `persistence_spec` line (`crates/bridge/src/search.rs:1739`).

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Frozen seat count (`candidate0_act.py:70` `% 4` modulus).
pub const CANDIDATE0_NUM_SEATS: i64 = 4;
/// Frozen planner history trim bound (`candidate0_act.py:395-396`).
pub const CANDIDATE0_HISTORY_LIMIT: usize = 16;

/// Pure core: sorted-unique union of concealed tiles with the drawn tile
/// (oracle `tuple(sorted({*concealed, own_drawn}))`; owned ints only, zero
/// Python API).
fn sorted_concealed_core(tiles: &[i64], drawn: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(tiles.len() + 1);
    out.extend_from_slice(tiles);
    out.push(drawn);
    out.sort_unstable();
    out.dedup();
    out
}

/// Pure core: offer-bearing phase gate (oracle `:61-62`).
fn has_offer_core(phase: &str) -> bool {
    matches!(phase, "discard_response" | "kan_response")
}

/// Pure core: fallback offer source when no pending discard names one
/// (oracle `:69-70`; `rem_euclid` matches Python floored `%`).
fn offered_by_core(actor_seat: i64) -> i64 {
    (actor_seat + 3).rem_euclid(CANDIDATE0_NUM_SEATS)
}

/// Sorted-unique concealed tiles plus the drawn tile
/// (`candidate0_act.py:43-47` drawn-present arm). Staged attached (owned
/// `Vec`/`i64` extracts, so non-int tiles reject here); the sort/dedup runs
/// under ONE detach; the translator wraps the returned list in `tuple` and
/// falls back to the oracle line on any staging reject.
#[pyfunction]
fn candidate0_sorted_concealed(py: Python<'_>, tiles: Vec<i64>, drawn: i64) -> PyResult<Vec<i64>> {
    Ok(py.detach(|| sorted_concealed_core(&tiles, drawn)))
}

/// Offer-bearing phase predicate (`candidate0_act.py:61-62`). Staging rejects
/// (`TypeError`/`OverflowError` on non-`str`) never surface — the translator
/// falls back to the oracle `in`-tuple on any extraction failure.
#[pyfunction]
fn candidate0_has_offer(py: Python<'_>, phase: String) -> PyResult<bool> {
    Ok(py.detach(|| has_offer_core(&phase)))
}

/// Fallback offer source (`candidate0_act.py:69-70`). The `int()` conversion
/// (and its errors) stays translator-side; only owned `i64` crosses, and the
/// translator falls back to the oracle line on any staging reject.
#[pyfunction]
fn candidate0_offered_by_fallback(py: Python<'_>, actor_seat: i64) -> PyResult<i64> {
    Ok(py.detach(|| offered_by_core(actor_seat)))
}

/// Register the candidate0-act leaves on the EXISTING `search` submodule
/// (mirrors `despot_seeds::register` on the shared submodule): consts via
/// `sub.add`, compute detached, wrap attached; single cdylib, no new entry
/// point. MAIN calls this from `search::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("CANDIDATE0_NUM_SEATS", CANDIDATE0_NUM_SEATS)?;
    sub.add("CANDIDATE0_HISTORY_LIMIT", CANDIDATE0_HISTORY_LIMIT)?;
    sub.add_function(wrap_pyfunction!(candidate0_sorted_concealed, sub)?)?;
    sub.add_function(wrap_pyfunction!(candidate0_has_offer, sub)?)?;
    sub.add_function(wrap_pyfunction!(candidate0_offered_by_fallback, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_concealed_matches_head_oracle() {
        // Hand-derived from HEAD (`tuple(sorted({*concealed, own_drawn}))`):
        // dupes collapse, drawn merges, output ascending.
        assert_eq!(sorted_concealed_core(&[5, 3, 3, 1], 3), vec![1, 3, 5]);
        assert_eq!(sorted_concealed_core(&[5], 5), vec![5]);
        assert_eq!(sorted_concealed_core(&[], 7), vec![7]);
        assert_eq!(sorted_concealed_core(&[1, 2], 9), vec![1, 2, 9]);
    }

    #[test]
    fn has_offer_matches_head_oracle() {
        // Hand-derived from HEAD (`phase in ("discard_response",
        // "kan_response")`).
        assert!(has_offer_core("discard_response"));
        assert!(has_offer_core("kan_response"));
        assert!(!has_offer_core("draw_decision"));
        assert!(!has_offer_core(""));
    }

    #[test]
    fn offered_by_matches_head_oracle() {
        // Hand-derived from HEAD (`(int(actor) + 3) % 4`), including the
        // floored-modulo negative arm (`(-1 + 3) % 4 == 2`).
        assert_eq!(offered_by_core(0), 3);
        assert_eq!(offered_by_core(1), 0);
        assert_eq!(offered_by_core(2), 1);
        assert_eq!(offered_by_core(3), 2);
        assert_eq!(offered_by_core(-1), 2);
    }

    #[test]
    fn frozen_values_match_head_oracle() {
        // Hand-derived from HEAD (`% 4` at `:70`, `> 16` / `[-16:]` at
        // `:395-396`).
        assert_eq!(CANDIDATE0_NUM_SEATS, 4);
        assert_eq!(CANDIDATE0_HISTORY_LIMIT, 16);
    }
}
