//! search_select: node-selection pyfunctions (UCT/PUCT over info-set nodes).
//!
//! DAG: `hydra-search` ismcts/gumbel + pyo3 ONLY (thin wrappers like
//! `search.rs`: validate attached, run ONE arena call detached, wrap the
//! outcome attached). Split from `search.rs` under the module-size ceiling;
//! registration, names, and error text are unchanged (single-cdylib tree:
//! registers on the shared `hydra2._native.search` submodule via
//! `register`, mirroring the sibling modules).
use crate::search::{parse_tie, search_err};
use hydra_search::SearchError;
use hydra_search::gumbel as gumbel_mod;
use hydra_search::ismcts::{ActionStats, IsNode, IsmctsParams};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build one information-set node from parallel arrays (pure, detach-safe).
///
/// `actions`/`visits` pair up 1:1; `sums_flat` holds 4 back-up sums per
/// action in `actions` order; `total` becomes `node.visits` (the UCT
/// exploration term reads it verbatim, mirroring `InformationSetNode.visits`).
/// Uniqueness rides a sorted-window scan (never `hash()`); visited arms with
/// non-finite sums fail closed here, unvisited arms skip the check exactly
/// like `scalar_mean` (visits `0` reads as unvisited).
fn build_node(
    actions: Vec<u32>,
    visits: Vec<u64>,
    sums_flat: Vec<f64>,
    total: u64,
) -> Result<IsNode, SearchError> {
    if actions.len() != visits.len() {
        return Err(SearchError::InvalidArg {
            detail: "node actions/visits length mismatch",
        });
    }
    let want = actions
        .len()
        .checked_mul(4)
        .ok_or(SearchError::InvalidArg {
            detail: "node actions too many",
        })?;
    if sums_flat.len() != want {
        return Err(SearchError::InvalidArg {
            detail: "node sums must hold 4 entries per action",
        });
    }
    let mut order = actions.clone();
    order.sort_unstable();
    if let Some(dup) = order.windows(2).find(|w| w[0] == w[1]).map(|w| w[0]) {
        return Err(SearchError::DuplicateAction { action: dup });
    }
    let mut arms = Vec::with_capacity(actions.len());
    let mut i = 0usize;
    while i < actions.len() {
        let action = match actions.get(i) {
            Some(action) => *action,
            None => {
                return Err(SearchError::InvalidArg {
                    detail: "node action out of range",
                });
            }
        };
        let n = match visits.get(i) {
            Some(n) => *n,
            None => {
                return Err(SearchError::InvalidArg {
                    detail: "node visits out of range",
                });
            }
        };
        let mut sum = [0.0f64; 4];
        let mut j = 0usize;
        while j < 4 {
            let value = match sums_flat.get(i * 4 + j) {
                Some(value) => *value,
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "node sums out of range",
                    });
                }
            };
            if n > 0 && !value.is_finite() {
                return Err(SearchError::NonFinite {
                    context: "search node value sum",
                });
            }
            sum[j] = value;
            j += 1;
        }
        arms.push((
            action,
            ActionStats {
                visits: n,
                value_sum: sum,
            },
        ));
        i += 1;
    }
    Ok(IsNode {
        visits: total,
        arms,
    })
}

/// UCT select over one information-set node (`ismcts_core.py:520-555` via
/// `uct_select`): unvisited arms win in sorted order first, else `q+u`.
///
/// Node shape crosses as parallel arrays (`actions`, per-action `visits`,
/// `sums_flat` with 4 back-up sums per action) plus the caller-visible
/// `total_visits` (`InformationSetNode.visits`); only scalars cross, never
/// hidden worlds (info-keys only).
#[pyfunction]
#[pyo3(signature = (actions, visits, sums_flat, legal_ids, root_seat, total_visits, uct_c, tie_break))]
fn uct_select(
    py: Python<'_>,
    actions: Vec<u32>,
    visits: Vec<u64>,
    sums_flat: Vec<f64>,
    legal_ids: Vec<u32>,
    root_seat: u32,
    total_visits: u64,
    uct_c: f64,
    tie_break: String,
) -> PyResult<u32> {
    if root_seat >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    if !uct_c.is_finite() || uct_c <= 0.0 {
        return Err(PyValueError::new_err("uct_c must be finite >0"));
    }
    let tie = parse_tie(&tie_break)?;
    let out = py.detach(|| {
        let node = build_node(actions, visits, sums_flat, total_visits)?;
        let mut params = IsmctsParams::defaults();
        params.uct_c = uct_c;
        params.tie = tie;
        params.validate()?;
        hydra_search::ismcts::uct_select(&node, &legal_ids, root_seat, &params)
    });
    out.map_err(search_err)
}

/// PUCT comparator select (`gumbel_puct.py:240-258` via `puct_select`):
/// `q + c*prior*sqrt(N)/(1+n)` with uniform priors, `1e-12` eps + tie arm.
#[pyfunction]
#[pyo3(signature = (actions, visits, sums_flat, legal_ids, root_seat, puct_c, tie_break))]
fn puct_select(
    py: Python<'_>,
    actions: Vec<u32>,
    visits: Vec<u64>,
    sums_flat: Vec<f64>,
    legal_ids: Vec<u32>,
    root_seat: u32,
    puct_c: f64,
    tie_break: String,
) -> PyResult<u32> {
    if root_seat >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    if !puct_c.is_finite() || puct_c <= 0.0 {
        return Err(PyValueError::new_err("puct_c must be finite >0"));
    }
    let tie = parse_tie(&tie_break)?;
    let out = py.detach(|| {
        let total = total_visits_for(&actions, &visits, &legal_ids);
        let node = build_node(actions, visits, sums_flat, total)?;
        gumbel_mod::puct_select(&node, &legal_ids, root_seat, puct_c, tie)
    });
    out.map_err(search_err)
}
/// PUCT visit total: `sum(visits)` over the legal set only (mirrors the
/// crate, which totals `node.stats` over legal internally — kept explicit
/// here so `build_node` keeps one shape and `node.visits` stays truthful).
fn total_visits_for(actions: &[u32], visits: &[u64], legal: &[u32]) -> u64 {
    let mut total = 0u64;
    let mut i = 0usize;
    while i < actions.len() && i < visits.len() {
        if let Some(action) = actions.get(i)
            && legal.contains(action)
            && let Some(n) = visits.get(i)
        {
            total = total.saturating_add(*n);
        }
        i += 1;
    }
    total
}

/// Register the node-selection pyfunctions on the shared `search` submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(uct_select, sub)?)?;
    sub.add_function(wrap_pyfunction!(puct_select, sub)?)?;
    Ok(())
}
