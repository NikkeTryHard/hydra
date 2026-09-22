//! eval: thin wall-block aggregation boundary over `hydra-search::eval`.
//!
//! DAG: this module depends on the search eval crate + pyo3 ONLY —
//! arg-check + detach + scalar return. The aggregation math lives
//! search-side in `hydra_search::eval::blocks::{WallBlock,
//! aggregate_wall_block}` (read-only reference:
//! `src/hydra2/eval/blocks.py::WallBlock/aggregate_wall_block`); this file
//! only reads the caller's block fields attached, runs the collapse
//! detached, and maps [`SearchError`] onto `ValueError` attached.
//!
//! Output-identity notes (Wave 3 bridge round):
//! - The pyfn takes the caller's `WallBlock` OBJECT (same signature as the
//!   oracle: `aggregate_wall_block(block)`), reading `wall_id`/`game_ids`/
//!   `contrasts` attached and validating through the real `WallBlock::new`
//!   (nonempty id, parallel lengths, nonempty game ids, finite contrasts) —
//!   so the cutover facade passes its existing blocks straight through with
//!   no repackaging, next round.
//! - Accumulation divisor order (byte-identity trap) is preserved
//!   EXPLICITLY: the oracle computes `math.fsum(contrasts) / len`, the owner
//!   computes `neumaier_sum(values) / len` (`eval::block_mean`) — sum first,
//!   divide second, divisor is the float block length on both sides. The
//!   input slice order crosses untouched (no reordering, no windowing).
//! - `math.fsum` is exactly rounded (order-independent); Neumaier matches it
//!   on the owner's recorded adversarial golden set (five cancellation
//!   vectors pinned in `eval::blocks` tests, e.g. `[1e16, 1.0, -1e16]` sums
//!   to exactly `1.0` where the naive left fold gives `0.0`). A full
//!   Shewchuk port stays out until a randomized differential proves
//!   bit-exactness (owner-documented bar) — this pyfn adds no new fork risk
//!   beyond the already-accepted Rust owner, it only exposes it.
//! - Fail-closed: empty blocks and non-finite contrasts raise (`ValueError`
//!   here vs `ContractError` there — both fail closed; messages are
//!   Rust-owner-worded per bridge precedent, the cutover facade translates).
//!   `bool`/`None` contrasts have no `f64` analogue and are unreachable via
//!   constructor-validated blocks.
//!
//! Deliberately NOT here (NO-NEW, named reasons): `aggregate_blocks` /
//! `_first_disqualification` need live `ResourceTelemetry` dataclasses +
//! `BlockTolerance` (dict-arg pyfns would be a new shape needing a Python
//! facade rewrite — next round); telemetry/promotion constructors and
//! `record_to_json`/`promotion_digest` bind contract digests (Wave 4 LAST);
//! schedule/partition/statistics owners stay caller-side until their facades
//! move (torch/numpy oracles: `randperm`, PCG64).
//!
//! Concurrency pattern (matches `packet_decode.rs` + `search.rs`): validate
//! attached, `py.detach(|| ...)` around the collapse, frozen surface only
//! (free functions; no pyclass, no shared state). Determinism: pure function
//! of the block, no seeds, never wall-clock. GIL attestation is spelled once
//! wave-wide in `canon_rng.rs` and inherited here.
//!
//! Single-cdylib tree: this module registers as the `hydra2._native.eval`
//! submodule via `register`, mirroring `search::register`.

use hydra_search::SearchError;
use hydra_search::eval::blocks::{WallBlock, aggregate_wall_block as aggregate_owner};
use hydra_search::eval::statistics::hedged_cs_path as hedged_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

/// Map an owner error onto the bridge failure (`ValueError`, never default).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Read one block field attached, failing closed with the field name.
fn block_field<T>(block: &Bound<'_, PyAny>, field: &str) -> PyResult<T>
where
    T: for<'a, 'py> pyo3::FromPyObject<'a, 'py>,
{
    block
        .getattr(field)
        .map_err(|e| {
            PyValueError::new_err(format!(
                "eval aggregate_wall_block: block.{field} unreadable: {e}"
            ))
        })?
        .extract()
        .map_err(|_| {
            PyValueError::new_err(format!(
                "eval aggregate_wall_block: block.{field} has the wrong type"
            ))
        })
}

/// Collapse ONE wall block to ONE number (`blocks.py:61-65`).
///
/// Takes the caller's `WallBlock` (fields `wall_id`/`game_ids`/`contrasts`
/// read attached and re-validated through `WallBlock::new`); the collapse
/// runs detached. Games inside the wall are not independent — callers must
/// feed whole blocks, never per-game contrasts (wall-block atomicity).
#[pyfunction]
fn aggregate_wall_block(py: Python<'_>, block: Bound<'_, PyAny>) -> PyResult<f64> {
    let wall_id: String = block_field(&block, "wall_id")?;
    let game_ids: Vec<String> = block_field(&block, "game_ids")?;
    let contrasts: Vec<f64> = block_field(&block, "contrasts")?;
    let owned = WallBlock::new(wall_id, game_ids, contrasts).map_err(search_err)?;
    py.detach(|| aggregate_owner(&owned)).map_err(search_err)
}
/// Time-uniform hedged-CS intervals (`statistics.py:280-329` via
/// `hydra_search::eval::statistics::hedged_cs_path`): draw-free deterministic
/// capital math. `values` are the raw block contrasts (scaling to `[0,1]`
/// under `(low, high)` runs owner-side); `peek_times=None` means the single
/// final peek. Caller-side type gates (bool rejection, int checks) stay in
/// the Python facade — `bool` has no `f64` analogue — the owner re-validates
/// finiteness, grid range, and peek windows and fails closed. Compute runs
/// detached; empty survivors yield `(inf, -inf)` exactly like the oracle.
#[pyfunction]
#[pyo3(signature = (values, alpha=0.05, low=0.0, high=1.0, grid_size=48, peek_times=None))]
fn hedged_cs_path(
    py: Python<'_>,
    values: Vec<f64>,
    alpha: f64,
    low: f64,
    high: f64,
    grid_size: usize,
    peek_times: Option<Vec<usize>>,
) -> PyResult<Vec<(f64, f64)>> {
    let out = py.detach(|| {
        hedged_owner(
            &values,
            alpha,
            (low, high),
            grid_size,
            peek_times.as_deref(),
        )
    });
    out.map_err(search_err)
}

/// Register the `eval` submodule (mirrors `search::register`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "eval")?;
    sub.add_function(wrap_pyfunction!(aggregate_wall_block, &sub)?)?;
    sub.add_function(wrap_pyfunction!(hedged_cs_path, &sub)?)?;
    crate::eval_blocks::register(&sub)?;
    crate::eval_leaves2::register(&sub)?;
    crate::record_leaves::register_eval(&sub)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    /// Build a `WallBlock`-shaped stub (`types.SimpleNamespace`) attached.
    fn stub_block<'py>(
        py: Python<'py>,
        wall_id: &str,
        game_ids: Vec<&str>,
        contrasts: Vec<f64>,
    ) -> Bound<'py, PyAny> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("wall_id", wall_id).unwrap();
        kwargs.set_item("game_ids", game_ids).unwrap();
        kwargs.set_item("contrasts", contrasts).unwrap();
        let types = PyModule::import(py, "types").unwrap();
        types
            .getattr("SimpleNamespace")
            .unwrap()
            .call((), Some(&kwargs))
            .unwrap()
    }

    #[test]
    fn block_mean_matches_fsum_then_divide() {
        Python::initialize();
        Python::attach(|py| {
            let block = stub_block(py, "w1", vec!["g1", "g2", "g3"], vec![1.0, 2.0, 3.0]);
            assert_eq!(aggregate_wall_block(py, block).unwrap(), 2.0);
            // Cancellation vector: fsum is exactly 1.0 (naive fold gives
            // 0.0); the mean rides the compensated sum, divisor last.
            let cancel = stub_block(py, "w2", vec!["g1", "g2", "g3"], vec![1e16, 1.0, -1e16]);
            assert_eq!(aggregate_wall_block(py, cancel).unwrap(), 1.0 / 3.0);
            // Single-game block collapses to its contrast.
            let solo = stub_block(py, "w3", vec!["g1"], vec![-0.5]);
            assert_eq!(aggregate_wall_block(py, solo).unwrap(), -0.5);
        });
    }

    #[test]
    fn block_shape_violations_fail_closed() {
        Python::initialize();
        Python::attach(|py| {
            // Empty block: construction allows it, collapse rejects it.
            let empty = stub_block(py, "w0", vec![], vec![]);
            assert!(aggregate_wall_block(py, empty).is_err());
            // Parallel-length violation rejected by the real constructor.
            let ragged = stub_block(py, "w1", vec!["g1"], vec![1.0, 2.0]);
            assert!(aggregate_wall_block(py, ragged).is_err());
            // Missing fields never default.
            let types = PyModule::import(py, "types").unwrap();
            let bare_kwargs = PyDict::new(py);
            bare_kwargs.set_item("wall_id", "w9").unwrap();
            let bare = types
                .getattr("SimpleNamespace")
                .unwrap()
                .call((), Some(&bare_kwargs))
                .unwrap();
            assert!(aggregate_wall_block(py, bare).is_err());
        });
    }
}
