//! oracle_join: WP-07B oracle-join rank-mapping leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `sha2` ONLY (the `qual_replay.rs:50` + `search.rs:86` import
//! precedent; no feed/shard/search edge — the strict float-domain sort below
//! cannot ride the feed `fixed::resolve_final_ranks` owner, which stages
//! `[i64; 4]`, while this leaf widens through `float(s)` exactly like the
//! oracle). FORBIDS: tie-break duplication (ties raise with the
//! `resolve_final_ranks first` text, so the east1 seat-wind order never fires
//! here — `resolve_final_ranks` in `contracts.rs` stays the single tie-break
//! owner), utility-scoring duplication (values ride the wave-6
//! `utility_values_for_ranks` bridge fn through the Python translators — the
//! `ORACLE_RANK_VALUES` golden stays owned by `oracle_targets.rs`), and
//! `ContractError` shaping (the bridge raises `ValueError`; thin Python
//! translators map to `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/oracle_join.py`:
//! - `oracle_join_ranks_from_scores` <- `ranks_from_final_scores`
//!   (`oracle_join.py:42-81`): strict — exactly 4 finite distinct numbers
//!   (`bool` rejected; huge ints keep the oracle's `OverflowError` at the
//!   `float(s)` widening, never a `ContractError`); ties raise (Tenhou-resolve
//!   first). The `(-score, seat)` key matches the feed `precedes`
//!   (`feed/src/fixed.rs:427-432`) on distinct inputs, where the seat arm is
//!   unreachable — so the integer subdomain agrees with `resolve_final_ranks`
//!   (pinned below), never a second ranks implementation.
//! - `oracle_join_synthetic_ranks` <- `_synthetic_ranks`
//!   (`oracle_join.py:138-156`): `sha256(decision_id)` first-four-bytes seat
//!   order through `sha2` (the `qual_replay.rs:116-117` fold precedent),
//!   always a strict 1..4 permutation.
//!
//! LOGIC staying Python: the `candidate` dict union (`ranks` / 4-list
//! `final_placement`) plus the whole 1..4-or-0..3 placement gate with the -1
//! bridge (repr-interpolated texts stay where `repr` is free), the value row
//! (the wave-6 `utility_values_for_ranks` call with 0-based +1 -> 1..4), every
//! loader/firewall/spawn boundary (`join_oracle_targets`, `_child_load_worker`,
//! `load_oracle_batch_in_subprocess`,
//! `assert_privileged_loader_isolated_from_encoder`), and every
//! `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `oracle_join|synthetic_ranks` over
//! `crates/bridge/src` + `crates/feed/src` + `crates/search/src` +
//! `crates/shard/src` hits only the int-domain `ranks_from_final_scores`
//! (`shard/src/parity.rs:855` — `serde_json::Value` staging for baked rows,
//! never the live-Python float exporter) and the `resolve_final_ranks`
//! tie-break owner (`feed/src/fixed.rs:400`, `contracts.rs:508`); the strict
//! float exporter and the decision_id-hash permutation are fresh here. The
//! 1..4/0..3 placement gate stays Python by review scope (no owner needed).
//!
//! Shape per fn: attached staging (exact `repr(value)` text via the live
//! Python API, so `{value!r}` renders byte-exact per `rc_require.rs:60-65`;
//! `cast::<PyList>`/`cast::<PyTuple>` per `rc_parse_aux.rs:491-493`;
//! bool-rejection per `contracts.rs:69-74`) -> ONE `py.detach(|| ...)` over
//! owned plain data with zero Python API inside (per `contracts.rs:434-437`)
//! -> attached wrap as `PyValueError` (per `contracts.rs:117-123`).
//! `OverflowError` from the `float(s)` widening of huge ints propagates
//! untouched (per `rc_require.rs:176-183` `stage_map_number`), exactly like
//! the oracle.
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `utility.rs:196-201`); no new submodule, no new entry point. Wiring is
//! MAIN-ONLY (`pub mod oracle_join;` in `lib.rs` plus
//! `crate::oracle_join::register(&sub)?;` in `contracts.rs` next to the
//! `crate::oracle_targets::register(&sub)?;` line).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyList, PyModule, PyTuple};
use sha2::{Digest, Sha256};

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (the `rc_require.rs:63-65` `py_repr` precedent).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Owned item handles for a `list`/`tuple` value (`None` otherwise — the
/// `rc_parse_aux.rs:491-497` `cast` precedent, never `downcast`; items cross
/// by index via owned `get_item` handles per `eval_leaves.rs:446`, so nothing
/// borrows the staged sequence).
fn seq_items<'a>(obj: &Bound<'a, PyAny>) -> Option<Vec<Bound<'a, PyAny>>> {
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for index in 0..list.len() {
            // `index < len`, so this never fails — still fail closed, never unwrap.
            items.push(list.get_item(index).ok()?);
        }
        Some(items)
    } else if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tup.len());
        for index in 0..tup.len() {
            items.push(tup.get_item(index).ok()?);
        }
        Some(items)
    } else {
        None
    }
}

/// Owned plain-data view of one final score: `bool` rejected, non-`int`/
/// non-`float` unmapped; huge ints keep the oracle's `OverflowError` at the
/// `extract::<f64>()` widening (the `rc_require.rs:176-183` precedent).
struct ScoreView {
    repr: String,
    value: Option<f64>,
}

fn stage_score(obj: &Bound<'_, PyAny>) -> PyResult<ScoreView> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() {
        return Ok(ScoreView { repr, value: None });
    }
    if !(obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>()) {
        return Ok(ScoreView { repr, value: None });
    }
    let value: f64 = obj.extract()?;
    Ok(ScoreView {
        repr,
        value: Some(value),
    })
}

/// Pure strict-rank core over staged finite values: pairwise-distinct gate
/// (equivalent to the oracle's `len(set(vals)) != 4` — NaN is unreachable
/// post-finiteness, and `-0.0 == 0.0` collapses under both), then seats by
/// `(-score, seat)` (the feed `precedes` key at `feed/src/fixed.rs:427-432`;
/// distinct inputs never reach the seat arm). Zero Python API: runs detached.
fn ranks_for_values(vals: &[f64; 4]) -> Option<[u8; 4]> {
    let mut k = 0usize;
    while k < 4 {
        let mut j = k + 1;
        while j < 4 {
            if vals[k] == vals[j] {
                return None;
            }
            j += 1;
        }
        k += 1;
    }
    let mut order = [0usize, 1, 2, 3];
    let mut i = 1usize;
    while i < 4 {
        let mut j = i;
        while j > 0 && score_precedes(order[j], order[j - 1], vals) {
            order.swap(j, j - 1);
            j -= 1;
        }
        i += 1;
    }
    let mut ranks = [0u8; 4];
    let mut pos = 0usize;
    while pos < 4 {
        // proof: `pos` in 0..4, so `pos+1` in 1..4, fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let rank: u8 = (pos + 1) as u8;
        ranks[order[pos]] = rank;
        pos += 1;
    }
    Some(ranks)
}

/// Total seat order for the strict core: higher score first, lower seat on
/// exact ties (unreachable post-distinctness; kept so the key matches the
/// feed owner bit-for-bit on the shared integer subdomain).
fn score_precedes(a: usize, b: usize, vals: &[f64; 4]) -> bool {
    if vals[a] != vals[b] {
        return vals[a] > vals[b];
    }
    a < b
}

/// Pure synthetic-rank core over a staged sha256 digest: seats ordered by
/// `(digest[seat], seat)` over the first four bytes, rank = position + 1 —
/// always a strict 1..4 permutation. Zero Python API: runs detached.
fn synthetic_ranks_for_digest(digest: &[u8; 32]) -> [u8; 4] {
    let mut order = [0usize, 1, 2, 3];
    let mut i = 1usize;
    while i < 4 {
        let mut j = i;
        while j > 0 && (digest[order[j]], order[j]) < (digest[order[j - 1]], order[j - 1]) {
            order.swap(j, j - 1);
            j -= 1;
        }
        i += 1;
    }
    let mut ranks = [0u8; 4];
    let mut pos = 0usize;
    while pos < 4 {
        // proof: `pos` in 0..4, so `pos+1` in 1..4, fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let rank: u8 = (pos + 1) as u8;
        ranks[order[pos]] = rank;
        pos += 1;
    }
    ranks
}

/// Strict per-seat ranks 1..4 from terminal final scores (mirrors
/// `ranks_from_final_scores` bit-for-bit incl. every error text; ties defer
/// to the already-bridged `resolve_final_ranks` via the message, never a
/// second tie-break here). Translators call positionally; the bridge raises
/// `ValueError`, Python maps to `ContractError` with the identical text.
/// Staging attached; the distinct gate + sort run under ONE `py.detach` with
/// zero Python API inside; the result wraps attached.
#[pyfunction]
#[pyo3(signature = (scores,))]
fn oracle_join_ranks_from_scores(py: Python<'_>, scores: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let scores_repr = py_repr(&scores)?;
    let Some(items) = seq_items(&scores) else {
        return Err(PyValueError::new_err(format!(
            "ranks_from_final_scores: expected 4 final scores, got {scores_repr}"
        )));
    };
    if items.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "ranks_from_final_scores: expected 4 final scores, got {scores_repr}"
        )));
    }
    let mut reprs: Vec<String> = Vec::with_capacity(4);
    let mut vals = [0.0f64; 4];
    for (index, item) in items.iter().enumerate() {
        let view = stage_score(item)?;
        let Some(value) = view.value else {
            return Err(PyValueError::new_err(format!(
                "ranks_from_final_scores: score[{index}] must be a number, got {}",
                view.repr
            )));
        };
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "ranks_from_final_scores: score[{index}] must be finite, got {}",
                view.repr
            )));
        }
        vals[index] = value;
        reprs.push(view.repr);
    }
    // Rebuilt exactly like the oracle's `list(scores)!r`: CPython list repr
    // is `[` + `, `.join(element reprs) + `]` for any flat list, and every
    // element repr above is staged from the live object.
    let list_repr = format!("[{}]", reprs.join(", "));
    let ranks = py.detach(|| ranks_for_values(&vals));
    let Some(ranks) = ranks else {
        return Err(PyValueError::new_err(format!(
            "ranks_from_final_scores: scores must be distinct (ties need Tenhou resolve_final_ranks first), got {list_repr}"
        )));
    };
    Ok(ranks.to_vec())
}
/// Deterministic synthetic 1..4 ranks from a decision_id hash (mirrors
/// `_synthetic_ranks`: `sha256(decision_id)` first-four-bytes seat order via
/// `sha2` — the `qual_replay.rs:116-117` fold precedent; the feed digest
/// owner stays out because raw bytes, not `sha256:` text, hash here).
/// Total: every `str` maps to a strict permutation under ONE `py.detach`.
#[pyfunction]
#[pyo3(signature = (decision_id,))]
fn oracle_join_synthetic_ranks(py: Python<'_>, decision_id: String) -> PyResult<Vec<u8>> {
    Ok(py
        .detach(|| {
            let digest = Sha256::digest(decision_id.as_bytes());
            let mut raw = [0u8; 32];
            raw.copy_from_slice(digest.as_slice());
            synthetic_ranks_for_digest(&raw)
        })
        .to_vec())
}

/// Attach the oracle-join rank leaves to the caller-provided `contracts`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `utility.rs:196-201`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(oracle_join_ranks_from_scores, sub)?)?;
    sub.add_function(wrap_pyfunction!(oracle_join_synthetic_ranks, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_goldens_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from HEAD (`test_privileged_ranks_oracle_wp07b.py:154-156`
            // plus the `ranks-asc` / `ranks-mixed` parity rows).
            let scores = PyList::new(py, [35000i64, 25000, 15000, 30000])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, scores).unwrap(),
                vec![1u8, 3, 4, 2]
            );
            let mixed = PyList::new(py, [10000i64, 30000, 20000, 0])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, mixed).unwrap(),
                vec![3u8, 1, 2, 4]
            );
            // Tuple input takes the same path (the oracle accepts both).
            let tup = PyTuple::new(py, [31000i64, 27000, 23000, 19000])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, tup).unwrap(),
                vec![1u8, 2, 3, 4]
            );
            // Fractional floats widen exactly like the oracle's `float(s)`.
            let frac = PyList::new(py, [35000.5f64, 25000.25, 15000.0, 30000.75])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, frac).unwrap(),
                vec![1u8, 3, 4, 2]
            );
        });
    }

    #[test]
    fn strict_agrees_with_feed_owner_on_int_subdomain() {
        Python::initialize();
        Python::attach(|py| {
            // The integer subdomain is shared with the tie-break owner; on
            // distinct inputs the seat arm never fires, so both agree.
            let arr = [35000i64, 25000, 15000, 30000];
            let vals = [35000.0f64, 25000.0, 15000.0, 30000.0];
            let mine = py.detach(|| ranks_for_values(&vals).unwrap());
            assert_eq!(mine, hydra_feed::fixed::resolve_final_ranks(arr));
        });
    }

    #[test]
    fn strict_rejections_are_byte_identical() {
        Python::initialize();
        Python::attach(|py| {
            let short = PyList::new(py, [1i64, 2, 3]).unwrap().into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, short)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: expected 4 final scores, got [1, 2, 3]"
            );
            let long = PyTuple::new(py, [1i64, 2, 3, 4, 5]).unwrap().into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, long)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: expected 4 final scores, got (1, 2, 3, 4, 5)"
            );
            let text = "abcd".into_pyobject(py).unwrap().into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, text)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: expected 4 final scores, got 'abcd'"
            );
            let tie = PyList::new(py, [1i64, 2, 3, 3]).unwrap().into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, tie)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: scores must be distinct (ties need Tenhou resolve_final_ranks first), got [1, 2, 3, 3]"
            );
            // Tuple ties rebuild the `list(scores)` repr, not the tuple repr.
            let tie_tup = PyTuple::new(py, [1i64, 2, 2, 3]).unwrap().into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, tie_tup)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: scores must be distinct (ties need Tenhou resolve_final_ranks first), got [1, 2, 2, 3]"
            );
            // `-0.0 == 0.0` collapses exactly like the oracle's set gate.
            let negzero = PyList::new(py, [0.0f64, -0.0, 1.0, 2.0])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, negzero)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: scores must be distinct (ties need Tenhou resolve_final_ranks first), got [0.0, -0.0, 1.0, 2.0]"
            );
            let with_bool = PyList::new(py, [1i64, 2, 3, 4]).unwrap();
            with_bool.set_item(0, true).unwrap();
            assert_eq!(
                oracle_join_ranks_from_scores(py, with_bool.into_any())
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: score[0] must be a number, got True"
            );
            let with_inf = PyList::new(py, [f64::INFINITY, 1.0, 2.0, 3.0])
                .unwrap()
                .into_any();
            assert_eq!(
                oracle_join_ranks_from_scores(py, with_inf)
                    .unwrap_err()
                    .to_string(),
                "ValueError: ranks_from_final_scores: score[0] must be finite, got inf"
            );
        });
    }

    #[test]
    fn synthetic_goldens_match_hashlib() {
        Python::initialize();
        // Hand-derived via CPython `hashlib.sha256(decision_id.encode()).digest()`
        // this session: seats ordered by `(digest[seat], seat)`, rank = position + 1.
        assert_eq!(
            synthetic_ranks_for_digest(&sha256_of("dec-join-0000")),
            [4u8, 3, 2, 1]
        );
        assert_eq!(synthetic_ranks_for_digest(&sha256_of("")), [4u8, 2, 3, 1]);
        assert_eq!(
            synthetic_ranks_for_digest(&sha256_of("décision-日本語-🎲")),
            [3u8, 2, 4, 1]
        );
        Python::attach(|py| {
            assert_eq!(
                oracle_join_synthetic_ranks(py, "dec-join-0000".to_owned()).unwrap(),
                vec![4u8, 3, 2, 1]
            );
            assert_eq!(
                oracle_join_synthetic_ranks(py, "test-123".to_owned()).unwrap(),
                vec![3u8, 4, 2, 1]
            );
            // Deterministic + always a strict permutation.
            let again = oracle_join_synthetic_ranks(py, "dec-join-0000".to_owned()).unwrap();
            assert_eq!(again, vec![4u8, 3, 2, 1]);
            let mut sorted = again.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![1u8, 2, 3, 4]);
        });
    }

    fn sha256_of(text: &str) -> [u8; 32] {
        let digest = Sha256::digest(text.as_bytes());
        let mut raw = [0u8; 32];
        raw.copy_from_slice(digest.as_slice());
        raw
    }
}
