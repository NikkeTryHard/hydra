//! utility: SPEC 5.2 rank-to-score scoring leaves on the `contracts` submodule.
//!
//! DAG: this module depends on pyo3 ONLY (same edge as the rest of the bridge;
//! no new dependency, no feed/shard/search edge — the scoring math is plain
//! `f64` indexing with no canon/digest/census owner). FORBIDS: rank resolution
//! (`resolve_final_ranks` in `contracts.rs` already owns the SPEC 5.1 east1
//! tie-break — never a second ranks implementation here; this file only gates
//! the already-resolved 1..4 permutation), digest folds/HMAC/file IO (the
//! manifest identity stays Python: canon bytes + sha256 +
//! `hmac.compare_digest` in `contracts/utility.py`), dataclass validation
//! roots (`RawOutcome` / `UtilityManifest` / `UtilityVector` `__post_init__`
//! bodies stay Python as the validating roots with byte-identical
//! `ContractError` text), and wall-clock/RNG.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-485`); the stage-attached /
//! `py.detach` / wrap-attached shape mirrors
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); `is_instance_of::<PyBool>`
//! bool-rejection mirrors `contracts::plain_int_value`
//! (`crates/bridge/src/contracts.rs:69-74`); int-or-float number staging
//! mirrors `rc_require::stage_map_number`
//! (`crates/bridge/src/rc_require.rs:176-183`); `into_pyobject(py)` in tests
//! mirrors (`crates/bridge/src/contracts.rs:1101-1102`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Feed owners: NONE (grep evidence: no `UTILITY_OBJECTIVE` /
//! `UTILITY_TIE_POLICY` / rank-score indexing anywhere under `crates/`; the
//! score domain `±1e12` stays owned by `contracts.rs:55-57`
//! `SCORE_MIN`/`SCORE_MAX`, and the manifest/ranks authorities stay the
//! `canon_rng` judges + `resolve_final_ranks`).
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `action_artifact`/`event_schema`/`validate` on `contracts`); no new
//! submodule, no new entry point. Wiring is MAIN-ONLY (`contracts::register`
//! calls this `register` with its `sub`, alongside
//! `crates/bridge/src/contracts.rs:1288-1300`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyModule};

/// Primary objective: expected final placement
/// (`contracts/utility.py:62` `UTILITY_OBJECTIVE`).
const UTILITY_OBJECTIVE: &str = "expected_final_placement";

/// Ties never reach utility: rules-resolved ranks are required up front
/// (`contracts/utility.py:64` `UTILITY_TIE_POLICY`).
const UTILITY_TIE_POLICY: &str = "use_rules_resolved_rank";

/// Stage one rank value: `bool` rejected (it subclasses `int`), `int`/`float`
/// accepted and widened to `f64`, non-finite rejected. Mirrors
/// `utility.py::_require_finite_float` bit-for-bit on the accepted domain;
/// error text is bridge-local (the Python translator maps every bridge
/// rejection to the byte-identical oracle `ContractError`).
fn stage_rank_value(obj: &Bound<'_, PyAny>, index: usize) -> Result<f64, String> {
    let record = "contracts utility rank_values";
    if obj.is_instance_of::<PyBool>() {
        return Err(format!(
            "{record}[{index}] must be a finite number, not bool"
        ));
    }
    if !(obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>()) {
        return Err(format!("{record}[{index}] must be a finite number"));
    }
    let value: f64 = obj
        .extract()
        .map_err(|_| format!("{record}[{index}] must be a finite number"))?;
    if !value.is_finite() {
        return Err(format!("{record}[{index}] must be finite"));
    }
    Ok(value)
}

/// Stage one resolved rank: `bool` rejected, plain `int` in `1..=4` only
/// (never rank resolution — that stays `resolve_final_ranks`).
fn stage_rank(obj: &Bound<'_, PyAny>, index: usize) -> Result<u8, String> {
    let record = "contracts utility ranks";
    if obj.is_instance_of::<PyBool>() {
        return Err(format!("{record}[{index}] must be an int 1..=4, not bool"));
    }
    if !obj.is_instance_of::<PyInt>() {
        return Err(format!("{record}[{index}] must be an int 1..=4"));
    }
    let rank: i64 = obj
        .extract()
        .map_err(|_| format!("{record}[{index}] must be an int 1..=4"))?;
    if !(1..=4).contains(&rank) {
        return Err(format!("{record}[{index}]={rank} outside [1, 4]"));
    }
    // proof: `rank` in 1..=4 (range-checked above), fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = rank as u8;
    Ok(b)
}

/// Stage one seat index: `bool` rejected, plain `int` in `0..=3` only.
fn stage_seat(seat: &Bound<'_, PyAny>) -> Result<u8, String> {
    let record = "contracts utility seat";
    if seat.is_instance_of::<PyBool>() {
        return Err(format!("{record} must be an int 0..=3, not bool"));
    }
    if !seat.is_instance_of::<PyInt>() {
        return Err(format!("{record} must be an int 0..=3"));
    }
    let index: i64 = seat
        .extract()
        .map_err(|_| format!("{record} must be an int 0..=3"))?;
    if !(0..=3).contains(&index) {
        return Err(format!("{record}={index} outside [0, 3]"));
    }
    // proof: `index` in 0..=3 (range-checked above), fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = index as u8;
    Ok(b)
}

/// Pure rank-to-score indexing: `rank_values[rank - 1]` per seat (mirrors
/// `utility.py:568` `tuple(manifest.rank_values[rank - 1] for rank in ranks)`).
/// Zero Python API: runs detached.
fn values_for_ranks(rank_values: &[f64; 4], ranks: &[u8; 4]) -> [f64; 4] {
    [
        rank_values[usize::from(ranks[0] - 1)],
        rank_values[usize::from(ranks[1] - 1)],
        rank_values[usize::from(ranks[2] - 1)],
        rank_values[usize::from(ranks[3] - 1)],
    ]
}

/// Per-seat placement utilities from rank values through resolved ranks
/// (mirrors `utility.py` `utility()` values line, SPEC 5.2).
///
/// Staging is attached (touches Python memory); the indexing runs under ONE
/// `py.detach` with zero Python API inside; the result wraps attached.
/// Fail-closed: every shape violation is `ValueError`, never a default.
#[pyfunction]
fn utility_values_for_ranks(
    py: Python<'_>,
    rank_values: Vec<Bound<'_, PyAny>>,
    ranks: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<f64>> {
    if rank_values.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts utility rank_values must hold exactly 4 numbers indexed by rank 1..4",
        ));
    }
    if ranks.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts utility ranks must hold exactly 4 ints",
        ));
    }
    let mut staged_values = [0.0f64; 4];
    for (index, obj) in rank_values.iter().enumerate() {
        staged_values[index] = stage_rank_value(obj, index).map_err(PyValueError::new_err)?;
    }
    let mut staged_ranks = [0u8; 4];
    for (index, obj) in ranks.iter().enumerate() {
        staged_ranks[index] = stage_rank(obj, index).map_err(PyValueError::new_err)?;
    }
    let mut sorted = staged_ranks;
    sorted.sort_unstable();
    if sorted != [1, 2, 3, 4] {
        return Err(PyValueError::new_err(
            "contracts utility ranks must be a strict permutation of 1..4 with no ties or gaps",
        ));
    }
    Ok(py
        .detach(|| values_for_ranks(&staged_values, &staged_ranks))
        .to_vec())
}

/// Acting-seat root scalar: vector index selection (mirrors
/// `utility.py:602` `value.values[seat_index]`, SPEC 5.2).
///
/// Same stage-attached / detach / wrap-attached shape as
/// [`utility_values_for_ranks`]. Fail-closed: every shape violation is
/// `ValueError`, never a default.
#[pyfunction]
fn utility_root_scalar(
    py: Python<'_>,
    values: Vec<Bound<'_, PyAny>>,
    seat: Bound<'_, PyAny>,
) -> PyResult<f64> {
    if values.len() != 4 {
        return Err(PyValueError::new_err(
            "contracts utility values must hold exactly 4 floats, one per seat",
        ));
    }
    let mut staged = [0.0f64; 4];
    for (index, obj) in values.iter().enumerate() {
        staged[index] = stage_rank_value(obj, index).map_err(PyValueError::new_err)?;
    }
    let seat_index = stage_seat(&seat).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| staged[usize::from(seat_index)]))
}

/// Attach the utility consts + pyfns to the caller-provided `contracts`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `rows_seal.rs:485`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("UTILITY_OBJECTIVE", UTILITY_OBJECTIVE)?;
    sub.add("UTILITY_TIE_POLICY", UTILITY_TIE_POLICY)?;
    sub.add_function(wrap_pyfunction!(utility_values_for_ranks, sub)?)?;
    sub.add_function(wrap_pyfunction!(utility_root_scalar, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod utility_tests {
    use super::*;

    /// Frozen consts are byte-identical to the Python oracle
    /// (`contracts/utility.py:62-64`).
    #[test]
    fn consts_match_oracle() {
        assert_eq!(UTILITY_OBJECTIVE, "expected_final_placement");
        assert_eq!(UTILITY_TIE_POLICY, "use_rules_resolved_rank");
    }

    /// Pure indexing over the WP-02B golden rank values, no Python needed.
    #[test]
    fn pure_values_identity_and_permutation() {
        let rank_values = [20.0, 10.0, -10.0, -20.0];
        assert_eq!(
            values_for_ranks(&rank_values, &[1, 2, 3, 4]),
            [20.0, 10.0, -10.0, -20.0]
        );
        // WP-02B seat-permutation base outcome ranks=(2, 3, 1, 4).
        assert_eq!(
            values_for_ranks(&rank_values, &[2, 3, 1, 4]),
            [10.0, -10.0, 20.0, -20.0]
        );
        // Every top-rank holder gets the best value.
        assert_eq!(
            values_for_ranks(&rank_values, &[4, 1, 3, 2]),
            [-20.0, 20.0, -10.0, 10.0]
        );
    }

    /// WP-02B identity golden through the pyfn: ranks=(1,2,3,4) over
    /// RANK_VALUES=(20.0, 10.0, -10.0, -20.0).
    #[test]
    fn pyfn_values_identity_golden() {
        Python::initialize();
        Python::attach(|py| {
            let rank_values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, -20.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            let ranks: Vec<Bound<'_, PyAny>> = [1i64, 2, 3, 4]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert_eq!(
                utility_values_for_ranks(py, rank_values, ranks).unwrap(),
                vec![20.0, 10.0, -10.0, -20.0]
            );
        });
    }

    /// WP-02B permutation golden through the pyfn: ranks=(2,3,1,4).
    #[test]
    fn pyfn_values_permutation_golden() {
        Python::initialize();
        Python::attach(|py| {
            let rank_values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, -20.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            let ranks: Vec<Bound<'_, PyAny>> = [2i64, 3, 1, 4]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert_eq!(
                utility_values_for_ranks(py, rank_values, ranks).unwrap(),
                vec![10.0, -10.0, 20.0, -20.0]
            );
        });
    }

    /// Int rank_values widen exactly like the oracle (`float(value)`).
    #[test]
    fn pyfn_values_int_widening() {
        Python::initialize();
        Python::attach(|py| {
            let rank_values: Vec<Bound<'_, PyAny>> = [20i64, 10, -10, -20]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            let ranks: Vec<Bound<'_, PyAny>> = [4i64, 3, 2, 1]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert_eq!(
                utility_values_for_ranks(py, rank_values, ranks).unwrap(),
                vec![-20.0, -10.0, 10.0, 20.0]
            );
        });
    }

    /// Boundary: tied/gapped/out-of-range ranks fail closed (oracle
    /// `TestMalformedSettlementRejection` shapes: (1,2,2,4), (0,2,3,4),
    /// (1,2,3,5)); `bool` is never an int; non-finite values rejected.
    #[test]
    fn pyfn_values_rejects() {
        Python::initialize();
        Python::attach(|py| {
            let good_values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, -20.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            let good_ranks: Vec<Bound<'_, PyAny>> = [1i64, 2, 3, 4]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            // Tied ranks (WP-02B bad shape (1,2,2,4)).
            let tied: Vec<Bound<'_, PyAny>> = [1i64, 2, 2, 4]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert!(utility_values_for_ranks(py, good_values.clone(), tied).is_err());
            // Out-of-range ranks.
            for bad in [[0i64, 2, 3, 4], [1, 2, 3, 5]] {
                let ranks: Vec<Bound<'_, PyAny>> = bad
                    .into_iter()
                    .map(|v| v.into_pyobject(py).unwrap().into_any())
                    .collect();
                assert!(utility_values_for_ranks(py, good_values.clone(), ranks).is_err());
            }
            // Bool rank is never an int.
            let bool_rank: Vec<Bound<'_, PyAny>> = [1i64, 2, 3, 4]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect::<Vec<_>>();
            let tru = true.into_pyobject(py).unwrap().to_owned().into_any();
            let mut with_bool = bool_rank;
            with_bool[0] = tru;
            assert!(utility_values_for_ranks(py, good_values.clone(), with_bool).is_err());
            // Wrong lengths.
            let short_values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert!(utility_values_for_ranks(py, short_values, good_ranks.clone()).is_err());
            let short_ranks: Vec<Bound<'_, PyAny>> = [1i64, 2, 3]
                .into_iter()
                .map(|v| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert!(utility_values_for_ranks(py, good_values.clone(), short_ranks).is_err());
            // Non-finite rank value.
            let inf_values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, f64::INFINITY]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert!(utility_values_for_ranks(py, inf_values, good_ranks.clone()).is_err());
            // Bool rank value is never a number.
            let mut bool_values = good_values.clone();
            bool_values[1] = true.into_pyobject(py).unwrap().to_owned().into_any();
            assert!(utility_values_for_ranks(py, bool_values, good_ranks).is_err());
        });
    }

    /// Root-scalar golden: seat index selection over the WP-02B vector.
    #[test]
    fn pyfn_root_scalar_golden() {
        Python::initialize();
        Python::attach(|py| {
            let values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, -20.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            for (seat, expected) in [(0i64, 20.0), (1, 10.0), (2, -10.0), (3, -20.0)] {
                let seat_obj = seat.into_pyobject(py).unwrap().into_any();
                assert_eq!(
                    utility_root_scalar(py, values.clone(), seat_obj).unwrap(),
                    expected
                );
            }
        });
    }

    /// Boundary: bad seats (4, -1, bool) and bad vectors fail closed.
    #[test]
    fn pyfn_root_scalar_rejects() {
        Python::initialize();
        Python::attach(|py| {
            let values: Vec<Bound<'_, PyAny>> = [20.0, 10.0, -10.0, -20.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            for bad in [4i64, -1] {
                let seat = bad.into_pyobject(py).unwrap().into_any();
                assert!(utility_root_scalar(py, values.clone(), seat).is_err());
            }
            let bool_seat = true.into_pyobject(py).unwrap().to_owned().into_any();
            assert!(utility_root_scalar(py, values.clone(), bool_seat).is_err());
            let good_seat = 0i64.into_pyobject(py).unwrap().into_any();
            let short: Vec<Bound<'_, PyAny>> = [20.0, 10.0]
                .into_iter()
                .map(|v: f64| v.into_pyobject(py).unwrap().into_any())
                .collect();
            assert!(utility_root_scalar(py, short, good_seat).is_err());
        });
    }
}
