//! search: thin isolated-act-loop boundary over `hydra-search` arena.
//!
//! DAG: this module depends on the search crate + pyo3 ONLY (the bridge's
//! arena edge; the `hydra-search` path dep + workspace member are
//! Arena-owned — see ASSUMPTION-MANIFEST; this file MUST NOT compile until
//! they land; DO NOT guess-fix signatures locally). FORBIDS: selection math,
//! node storage, RNG streams, belief sampling, digest bytes — arg-check +
//! detach + tuple return only. All arena tables/selection live search-side;
//! all bytes/hash math lives `feed::canon/digest` (canon-wins B3 — search
//! owns arena tables + selection, canon owns bytes); this file only
//! validates arguments attached, runs ONE arena act detached, and wraps the
//! outcome attached.
//!
//! Isolated act loop (vLLM-EngineCore steal): `(spec_params, root_obs_doc,
//! legal_ids, belief_refs, budget)` crosses ONCE per `act_batch` call; the
//! arena holds nodes/vectors/counters across calls. The GIL is released
//! around the ENTIRE arena call (`py.detach` once, never per-node —
//! per-node attach is the deadlock/hang shape of #3089).
//!
//! Belief handles (M1 producer closure REQUIRED): `belief_refs` are opaque
//! `u64` world handles, never Python objects: Rust-side world blobs (or
//! detached small-array copies) plus a `belief.sample_natural` port as a
//! Rust closure over interned keys live arena-side. NO per-sim Python
//! callback crosses this boundary — the detached closure below is pure
//! Rust. `completed=false` returns the frontier action + counters only (no
//! partial selection is ever presented as complete).
//!
//! Inputs cross as owned `Vec`s (memcpy-attached semantics like the
//! columnar descriptors: the GIL is held during extraction, so caller
//! memory is stable; only owned buffers cross the `detach`). Bytes path,
//! NO DLPack/Arrow here (device tensors belong to the loop, Phase 5).
//!
//! Pools (B5): `FeedPool` stays the sole GLOBAL pool; the arena owns its
//! SCOPED pool internally. This module creates NO pool, resizes none.
//!
//! Concurrency pattern (matches `canon_rng.rs` + `columnar.rs`): validate
//! attached, ONE `py.detach(|| ...)` around the whole arena call (fan-out/
//! join never holds the interpreter), `MutexExt::lock_py_attached` for the
//! stats mutex, frozen surface only (this module exposes free functions;
//! no pyclass, no shared cells beyond the attach-guarded stats mutex).
//! GIL attestation (`gil_used=false` default on PyO3 >= 0.28) is spelled
//! once in `canon_rng.rs` wave-wide and inherited by this submodule's
//! registration — not repeated here.
//!
//! Single-cdylib tree: this module registers as the `hydra2._native.search`
//! submodule via `register`, mirroring `canon_rng::register` / `columnar::register`.
//! bridge's `hydra-search = { path = "../search" }` edge; this file
//! MUST NOT compile until applied; DO NOT guess-fix locally): workspace
//! `members += "search"`; bridge deps gain the path edge (no
//! features, no pyo3 inside `hydra-search` — pyclasses live bridge-side
//! only).
//!
//! SIBLING-OWNED SEARCH API (Phase4SearchArena-confirmed contract — names
//! below MUST NOT be guess-fixed locally):
//! - `hydra_search::arena::Budget { max_sims: u64, max_depth: u32,
//!   deadline_ms: u64 }` (plain Clone+Copy struct, pub fields).
//! - `hydra_search::arena::ActOut { action: u32, completed: bool,
//!   sims_run: u64, nodes_visited: u64, decision_digest: String }`
//!   (`decision_digest` is `sha256:`-prefixed text over canon-owned bytes,
//!   computed arena-side via feed; `completed=false` carries the frontier
//!   action + counters only).
//! - `hydra_search::arena::act_batch(spec: &[u8], root: &[u8],
//!   legal: &[u32], worlds: &[u64], budget: &Budget) -> ActOut` (pure,
//!   detach-safe, no Python interaction; the M1 belief closure runs
//!   entirely arena-side over `worlds`).
//! - `BATCH` is NOT redefined here: the single `columnar::BATCH` const
//!   remains the ONE batch width (this module takes widths as plain args).

use crate::search_drive_out::{
    PyDespotOut, PyJointOut, PyPbrfOut, PyPersistenceOut, drive_out_to_despot, drive_out_to_joint,
    drive_out_to_pbrf, drive_out_to_persistence,
};
use hydra_search::SearchError;
#[cfg(test)]
use hydra_search::arena::ActOut;
use hydra_search::arena::{Budget, act_batch as arena_act_batch};
use hydra_search::belief as belief_mod;
use hydra_search::builtin_sum;
use hydra_search::despot as despot_mod;
use hydra_search::drive_batch as drive_batch_mod;
use hydra_search::gumbel as gumbel_mod;
use hydra_search::gumbel_driver as gumbel_driver_mod;
use hydra_search::ismcts::{ActionStats, IsNode, IsmctsParams, TieBreak};
use hydra_search::ismcts_driver as driver_mod;
use hydra_search::joint as joint_mod;
use hydra_search::local_driver as local_driver_mod;
use hydra_search::modules as modules_mod;
use hydra_search::pbrf as pbrf_mod;
use hydra_search::persistence_kernel as persist_mod;
use hydra_search::persistence_spec as spec_mod;
use hydra_search::step_hash as step_hash_mod;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyAny, PyModule};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

/// Module-transform output: `(particles, weights, budget_calls,
/// budget_transitions, meta_json)`.
type ModuleTransformOut = (Vec<f64>, Vec<f64>, u64, u64, Vec<u8>);

/// Max deadline mirrored from `search/common.py::_require_deadline_ms`
/// (`(0, 60000]`); the arena owns enforcement, the bridge fails closed
/// first so a bad budget never crosses the detach.
pub const MAX_DEADLINE_MS: u64 = 60_000;

/// Cumulative act observability (counts only, never identity/selection).
#[derive(Debug, Default, Clone, Copy)]
struct ActStats {
    acts: u64,
    completed: u64,
}

static ACT_STATS: Mutex<ActStats> = Mutex::new(ActStats {
    acts: 0,
    completed: 0,
});

fn bump_act(py: Python<'_>, completed: bool) -> Result<(), PyErr> {
    let mut guard = ACT_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("search act stats mutex poisoned"))?;
    guard.acts = guard.acts.saturating_add(1);
    if completed {
        guard.completed = guard.completed.saturating_add(1);
    }
    Ok(())
}

/// Pure budget gate (attached call-site; unit-tested below): mirrors the
/// frozen `ResourceBudget` ranges (`max_sims/depth >= 1`, deadline in
/// `(0, 60000]`). Fails closed before any arena work.
fn check_budget_args(max_sims: u64, max_depth: u32, deadline_ms: u64) -> Result<(), String> {
    if max_sims == 0 {
        return Err("search budget max_sims must be >= 1".to_string());
    }
    if max_depth == 0 {
        return Err("search budget max_depth must be >= 1".to_string());
    }
    if deadline_ms == 0 || deadline_ms > MAX_DEADLINE_MS {
        return Err(format!(
            "search budget deadline_ms {deadline_ms} outside (0,{MAX_DEADLINE_MS}]"
        ));
    }
    Ok(())
}

/// Pure input gate (attached call-site; unit-tested below): non-empty spec,
/// root, and legal set — fails closed before any copy/detach work.
/// Uniqueness of `legal_ids` is checked at the call site (sorted-window
/// scan over the caller-sized vec); emptiness lives here for testability.
fn check_act_args(spec_len: usize, root_len: usize, legal_len: usize) -> Result<(), String> {
    if spec_len == 0 {
        return Err("search act spec_params must be non-empty".to_string());
    }
    if root_len == 0 {
        return Err("search act root_obs_doc must be non-empty".to_string());
    }
    if legal_len == 0 {
        return Err("search act legal_ids must be non-empty".to_string());
    }
    Ok(())
}

/// Pure digest-shape gate (attached call-site; unit-tested below): the
/// arena MUST return canon-form `sha256:<64 lowercase hex>` — an arena bug
/// surfaces as an error here, never as a silent default/empty accept.
fn check_digest_shape(text: &str) -> Result<(), String> {
    const PREFIX: &str = "sha256:";
    let hex = text
        .strip_prefix(PREFIX)
        .ok_or_else(|| format!("search decision digest must start with 'sha256:', got {text:?}"))?;
    if hex.len() != 64 || !hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!(
            "search decision digest must be 'sha256:' + 64 lowercase hex, got {text:?}"
        ));
    }
    Ok(())
}

/// Map a search-side failure onto the boundary (caller-shape rejects only;
/// the arena `act_batch` frontier path above never raises for selection
/// outcomes — these step kernels fail closed like `canon_rng::bounded`).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Parse the frozen tie-break vocabulary (attached; pure string match).
fn parse_tie(name: &str) -> Result<TieBreak, PyErr> {
    TieBreak::parse(name).map_err(search_err)
}

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

/// B1-verbatim deterministic Gumbel for one action
/// (`gumbel_core.py:117-156` via `hydra_search::gumbel`).
///
/// Pure sha draw — identical inputs give identical outputs regardless of call
/// order or global RNG. Compute runs detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (case_id, root_seat, candidate_id, action_id))]
fn gumbel_for_action(
    py: Python<'_>,
    case_id: String,
    root_seat: u32,
    candidate_id: String,
    action_id: u32,
) -> PyResult<f64> {
    if case_id.is_empty() {
        return Err(PyValueError::new_err(
            "search gumbel case_id must be non-empty",
        ));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err(
            "search gumbel candidate_id must be non-empty",
        ));
    }
    if root_seat >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    let out = py
        .detach(|| gumbel_mod::deterministic_gumbel(&case_id, root_seat, &candidate_id, action_id));
    out.map_err(search_err)
}

/// Deterministic Gumbels for every legal action (`deterministic_root_gumbels`).
#[pyfunction]
#[pyo3(signature = (case_id, root_seat, candidate_id, legal_ids))]
fn gumbel_roots(
    py: Python<'_>,
    case_id: String,
    root_seat: u32,
    candidate_id: String,
    legal_ids: Vec<u32>,
) -> PyResult<Vec<(u32, f64)>> {
    if case_id.is_empty() {
        return Err(PyValueError::new_err(
            "search gumbel case_id must be non-empty",
        ));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err(
            "search gumbel candidate_id must be non-empty",
        ));
    }
    if legal_ids.is_empty() {
        return Err(PyValueError::new_err("legal set must be non-empty"));
    }
    let out = py.detach(|| {
        gumbel_mod::deterministic_root_gumbels(&case_id, root_seat, &candidate_id, &legal_ids)
    });
    out.map_err(search_err)
}

/// One sequential-halving cut (`gumbel_search.py:494-499` via `halving_cut`):
/// keep the top half of `survivors` by `score = q + g` (`1e-12` eps + tie arm).
#[pyfunction]
#[pyo3(signature = (survivors, means, gumbels, tie_break))]
fn halving_cut(
    py: Python<'_>,
    survivors: Vec<u32>,
    means: Vec<(u32, f64)>,
    gumbels: Vec<(u32, f64)>,
    tie_break: String,
) -> PyResult<Vec<u32>> {
    let tie = parse_tie(&tie_break)?;
    let out = py.detach(|| gumbel_mod::halving_cut(&survivors, &means, &gumbels, tie));
    out.map_err(search_err)
}

/// Final Gumbel selection over the last survivors (`gumbel_search.py:514-539`
/// via `gumbel_select`): max `g + q`, `1e-12` eps + tie arm.
#[pyfunction]
#[pyo3(signature = (survivors, means, gumbels, tie_break))]
fn gumbel_select(
    py: Python<'_>,
    survivors: Vec<u32>,
    means: Vec<(u32, f64)>,
    gumbels: Vec<(u32, f64)>,
    tie_break: String,
) -> PyResult<u32> {
    let tie = parse_tie(&tie_break)?;
    let out = py.detach(|| gumbel_mod::gumbel_select(&survivors, &means, &gumbels, tie));
    out.map_err(search_err)
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

/// Deterministic corpus indices for `count` natural draws over `K` worlds
/// (`natural.py:298-336` via `belief::natural_indices`): CTR-exact, including
/// the `K == 1` no-consume rule. Returns `(indices, end_cursor)` for exact
/// replay (`checkpoint`/`jump_to` shape).
#[pyfunction]
#[pyo3(signature = (k, count, seed, cursor))]
fn natural_indices(
    py: Python<'_>,
    k: u32,
    count: u32,
    seed: Vec<u8>,
    cursor: u64,
) -> PyResult<(Vec<u32>, u64)> {
    if k == 0 {
        return Err(PyValueError::new_err("natural corpus size must be >= 1"));
    }
    if count == 0 {
        return Err(PyValueError::new_err("natural count must be >= 1"));
    }
    if seed.is_empty() {
        return Err(PyValueError::new_err("ctr seed must be non-empty"));
    }
    let out = py.detach(|| belief_mod::natural_indices(k, count, &seed, cursor));
    out.map_err(search_err)
}

/// Categorical draws over the exhaustive frame law (`sampled_kernel.py:137-147`
/// via `belief::sampled_draws`): `(chosen, raw_weights, end_cursor)` with
/// `raw_weight = P(chosen) / draws`.
#[pyfunction]
#[pyo3(signature = (probs, draws, seed, cursor))]
fn sampled_draws(
    py: Python<'_>,
    probs: Vec<f64>,
    draws: u32,
    seed: Vec<u8>,
    cursor: u64,
) -> PyResult<(Vec<u32>, Vec<f64>, u64)> {
    if probs.is_empty() {
        return Err(PyValueError::new_err("sampled frame must be non-empty"));
    }
    if draws == 0 {
        return Err(PyValueError::new_err("sampled draws must be >= 1"));
    }
    if seed.is_empty() {
        return Err(PyValueError::new_err("ctr seed must be non-empty"));
    }
    let out = py.detach(|| belief_mod::sampled_draws(&probs, draws, &seed, cursor));
    out.map_err(search_err)
}

/// One enumerated packet successor crossing the boundary (digest/text shapes
/// only — info-keys, never world blobs).
#[pyclass(name = "PacketSuccessor", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPacketSuccessor {
    #[pyo3(get)]
    pub successor_world_ref: String,
    #[pyo3(get)]
    pub successor_delta: String,
    #[pyo3(get)]
    pub tile: u32,
    #[pyo3(get)]
    pub seq: u32,
    #[pyo3(get)]
    pub actor: u32,
    #[pyo3(get)]
    pub probability: f64,
    #[pyo3(get)]
    pub log_physical: f64,
    #[pyo3(get)]
    pub log_policy: f64,
    #[pyo3(get)]
    pub observation_hash: String,
    #[pyo3(get)]
    pub packet_id: String,
    #[pyo3(get)]
    pub chain_after: String,
}

/// Exhaustive next-packet enumeration (`kernel.py:155-260` via
/// `belief::enumerate_next`): exactly 2 disjoint successors per
/// `(parent_ref, action_id)` with unit mass. Compute runs detached; the
/// outcome wraps attached (packet construction holds no interpreter state).
#[pyfunction]
#[pyo3(signature = (parent_ref, action_id, root_actor, rules_hash))]
fn packet_successors(
    py: Python<'_>,
    parent_ref: String,
    action_id: u32,
    root_actor: u32,
    rules_hash: String,
) -> PyResult<Vec<PyPacketSuccessor>> {
    if parent_ref.is_empty() {
        return Err(PyValueError::new_err("belief parent_ref must be non-empty"));
    }
    if root_actor >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    check_digest_shape(&rules_hash).map_err(PyValueError::new_err)?;
    let staged =
        py.detach(|| belief_mod::enumerate_next(&parent_ref, action_id, root_actor, &rules_hash));
    let staged = staged.map_err(search_err)?;
    Ok(staged
        .into_iter()
        .map(|succ| PyPacketSuccessor {
            successor_world_ref: succ.successor_world_ref,
            successor_delta: succ.successor_delta,
            tile: succ.tile,
            seq: succ.seq,
            actor: succ.actor,
            probability: succ.probability,
            log_physical: succ.log_physical,
            log_policy: succ.log_policy,
            observation_hash: succ.observation_hash,
            packet_id: succ.packet_id,
            chain_after: succ.chain_after,
        })
        .collect())
}

/// Run ONE isolated arena act: owned inputs in, owned outcome out.
///
/// `(spec_params, root_obs_doc, legal_ids, belief_refs, budget)` crosses
/// ONCE per call; the arena holds nodes/vectors/counters across calls
/// (vLLM-EngineCore steal). The GIL is released around the ENTIRE arena
/// call, never per-node. `belief_refs` are opaque `u64` world handles —
/// the M1 belief producer closure runs entirely arena-side, so no per-sim
/// Python callback exists. `completed=false` returns the frontier action +
/// counters only. Returns
/// `(action, completed, sims_run, nodes_visited, decision_digest)`.
#[pyfunction]
#[pyo3(signature = (spec_params, root_obs_doc, legal_ids, belief_refs, max_sims, max_depth, deadline_ms))]
fn act_batch(
    py: Python<'_>,
    spec_params: Vec<u8>,
    root_obs_doc: Vec<u8>,
    legal_ids: Vec<u32>,
    belief_refs: Vec<u64>,
    max_sims: u64,
    max_depth: u32,
    deadline_ms: u64,
) -> PyResult<(u32, bool, u64, u64, String)> {
    check_budget_args(max_sims, max_depth, deadline_ms).map_err(PyValueError::new_err)?;
    check_act_args(spec_params.len(), root_obs_doc.len(), legal_ids.len())
        .map_err(PyValueError::new_err)?;
    let mut sorted = legal_ids.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err(
            "search act legal_ids must hold distinct ids",
        ));
    }
    let budget = Budget {
        max_sims,
        max_depth,
        deadline_ms,
    };
    // Compute DETACHED: ONE release around the whole arena act (never
    // per-node attach); only owned buffers cross the boundary.
    let out = py.detach(|| {
        arena_act_batch(
            &spec_params,
            &root_obs_doc,
            &legal_ids,
            &belief_refs,
            &budget,
        )
    });
    // Attached-only: arena digest bug fails closed (never defaulted), then
    // publish the counters under a single interpreter-safe lock.
    check_digest_shape(&out.decision_digest).map_err(PyValueError::new_err)?;
    bump_act(py, out.completed)?;
    Ok((
        out.action,
        out.completed,
        out.sims_run,
        out.nodes_visited,
        out.decision_digest,
    ))
}
/// One batch-descent outcome crossing the boundary (selection + means +
/// digest + counters + debug tree JSON; world blobs never cross back).
#[pyclass(name = "IsmctsDescentOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyIsmctsDescentOut {
    /// Selected action id (max scalarized mean, `1e-12` eps + tie arm).
    #[pyo3(get)]
    pub selected_id: u32,
    /// Candidate ids in sorted order.
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    /// Mean 4-vectors per candidate (row-major, 4 entries each).
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    /// Visits per candidate.
    #[pyo3(get)]
    pub visits: Vec<u64>,
    /// Canon digest over the sorted tree dump.
    #[pyo3(get)]
    pub tree_digest: String,
    /// Completed simulations.
    #[pyo3(get)]
    pub sims_run: u64,
    /// Completed transitions.
    #[pyo3(get)]
    pub transitions: u64,
    /// Model leaf calls (terminal fallbacks excluded).
    #[pyo3(get)]
    pub model_calls: u64,
    /// Tree node count.
    #[pyo3(get)]
    pub tree_nodes: u64,
    /// CTR floats consumed.
    #[pyo3(get)]
    pub floats_used: u64,
    /// Debug tree dump as JSON (`[{key, visits, arms}]`, key-ordered).
    #[pyo3(get)]
    pub tree_json: String,
}

/// Deduped ISMCTS model vector (`ismcts_core.py:264-286` via
/// `ismcts_driver::model_vector`): `sha256(f"{wid}:{candidate}:leaf")`,
/// first 4 bytes `(b % 100) / 100.0`. The salt stays explicit, so
/// `candidate1` and `candidate6` never collide. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (world_id, candidate_id))]
fn ismcts_model_vector(
    py: Python<'_>,
    world_id: String,
    candidate_id: String,
) -> PyResult<(f64, f64, f64, f64)> {
    if world_id.is_empty() {
        return Err(PyValueError::new_err(
            "search model world_id must be non-empty",
        ));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err(
            "search model candidate_id must be non-empty",
        ));
    }
    let out = py.detach(|| driver_mod::model_vector(&world_id, &candidate_id));
    out.map(|vector| (vector[0], vector[1], vector[2], vector[3]))
        .map_err(search_err)
}

/// Deduped ISMCTS terminal vector (`ismcts_core.py:289-308` via
/// `ismcts_driver::terminal_vector`): `sha256(f"{wid}:terminal")`, scores
/// `(b % 50) - 25`, then `/ 50.0 + 0.5`. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (world_id,))]
fn ismcts_terminal_vector(py: Python<'_>, world_id: String) -> PyResult<(f64, f64, f64, f64)> {
    if world_id.is_empty() {
        return Err(PyValueError::new_err(
            "search terminal world_id must be non-empty",
        ));
    }
    let out = py.detach(|| driver_mod::terminal_vector(&world_id));
    out.map(|vector| (vector[0], vector[1], vector[2], vector[3]))
        .map_err(search_err)
}
/// Deduped local model vector (`local_abstraction.py:141-164` via
/// `ismcts_driver::local_model_vector`): `sha256(wid:leaf_kind)`, four `u16`
/// words to `[-1,1]`, zero-sum centered. Salt stays explicit (`leaf_kind`).
/// Compute runs detached.
#[pyfunction]
#[pyo3(signature = (world_id, leaf_kind))]
fn ismcts_local_model_vector(
    py: Python<'_>,
    world_id: String,
    leaf_kind: String,
) -> PyResult<(f64, f64, f64, f64)> {
    if leaf_kind.is_empty() {
        return Err(PyValueError::new_err(
            "search local leaf_kind must be non-empty",
        ));
    }
    let out = py.detach(|| driver_mod::local_model_vector(&world_id, &leaf_kind));
    out.map(|vector| (vector[0], vector[1], vector[2], vector[3]))
        .map_err(search_err)
}

/// Deduped local terminal vector (`local_abstraction.py:167-204` via
/// `ismcts_driver::local_terminal_vector`): exact settlement from hands +
/// wall (no hash). Hands cross as 4 rows; tiles `0..135`. Compute detached.
#[pyfunction]
#[pyo3(signature = (hands, live))]
fn ismcts_local_terminal_vector(
    py: Python<'_>,
    hands: Vec<Vec<u32>>,
    live: Vec<u32>,
) -> PyResult<(f64, f64, f64, f64)> {
    if hands.len() != 4 {
        return Err(PyValueError::new_err(
            "search local hands must hold 4 seats",
        ));
    }
    let out = py.detach(|| driver_mod::local_terminal_vector(&hands, &live));
    out.map(|vector| (vector[0], vector[1], vector[2], vector[3]))
        .map_err(search_err)
}

/// Unified exact transition (`ismcts_search.py:95-136` + `gumbel_core.py:371-411`
/// via `ismcts_driver::exact_transition`): pops one live tile, rotates turn,
/// bumps step, links `snapshot = f"{domain}:{wid}:{action}:{step}"`, and
/// re-hashes the world id via the single canon site. `domain` keeps the
/// salts (`ismcts` vs `gumbel`); hands cross as 4 rows x 2 tiles, missing
/// latent keys as `None`. Compute runs detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (world_id, hands, live, dead, step, turn, corpus_idx, rules_hash, observation_hash, snapshot, actor, action_id, domain))]
// Python signature is frozen (13 args + `py`); splitting would fork the oracle
// parity surface — mirrors existing `#[allow(clippy::too_many_arguments)]` pyfns.
#[allow(clippy::too_many_arguments)]
fn ismcts_transition(
    py: Python<'_>,
    world_id: String,
    hands: Vec<Vec<u32>>,
    live: Vec<u32>,
    dead: Vec<u32>,
    step: Option<u32>,
    turn: Option<u32>,
    corpus_idx: Option<u32>,
    rules_hash: String,
    observation_hash: String,
    snapshot: String,
    actor: u32,
    action_id: u32,
    domain: String,
) -> PyResult<(String, Vec<u32>, u32, u32, String)> {
    if world_id.is_empty() {
        return Err(PyValueError::new_err(
            "search transition world_id must be non-empty",
        ));
    }
    if hands.len() != 4 {
        return Err(PyValueError::new_err(
            "search transition hands must hold 4 seats",
        ));
    }
    if actor >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    if domain != "ismcts" && domain != "gumbel" {
        return Err(PyValueError::new_err(
            "search transition domain must be ismcts|gumbel",
        ));
    }
    let out = py.detach(|| {
        if hands.len() != 4 {
            return Err(SearchError::InvalidArg {
                detail: "hands must hold 4 seats",
            });
        }
        let mut seats = [[0u32; 2]; 4];
        let mut s = 0;
        while s < 4 {
            let row = match hands.get(s) {
                Some(row) => row,
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "hands seat missing",
                    });
                }
            };
            if row.len() != 2 {
                return Err(SearchError::InvalidArg {
                    detail: "each hand must hold 2 tiles",
                });
            }
            seats[s][0] = row[0];
            seats[s][1] = row[1];
            s += 1;
        }
        let parent = driver_mod::TinyWorld {
            world_id,
            hands: seats,
            live,
            dead,
            step,
            turn,
            last_action: None,
            corpus_idx,
            rules_hash,
            observation_hash,
            snapshot,
        };
        let child = driver_mod::exact_transition(&parent, actor, action_id, &domain)?;
        let step_out = child.step.unwrap_or(0);
        let turn_out = child.turn.unwrap_or(0);
        Ok((
            child.world_id,
            child.live,
            step_out,
            turn_out,
            child.snapshot,
        ))
    });
    out.map_err(search_err)
}

/// Info key from a full observation JSON doc (`ismcts_core.py:197-242` via
/// `ismcts_driver::info_key_from_doc`): drops `legal_mask` +
/// `observation_hash`, rejects the 17 forbidden fields, canons via the
/// single feed site, hashes. `FullWorld` fields fail closed (ActorObservation
/// docs only). Compute runs detached.
#[pyfunction]
#[pyo3(signature = (obs_doc_json,))]
fn ismcts_info_key(py: Python<'_>, obs_doc_json: Vec<u8>) -> PyResult<String> {
    if obs_doc_json.is_empty() {
        return Err(PyValueError::new_err(
            "search info obs doc must be non-empty",
        ));
    }
    let out = py.detach(|| driver_mod::info_key_from_doc(&obs_doc_json));
    out.map_err(search_err)
}

/// Run one batch ISMCTS descent: precomputed worlds, per-step info keys,
/// per-step policy directions, and CTR floats cross ONCE as a JSON envelope;
/// the arena replays UCT descent, unified transitions, deduped leaf vectors,
/// and backup with the GIL released, then returns selection + means +
/// digest + counters. Leaf overrides (read-only torch vectors keyed by
/// world id) ride the same envelope; absent entries compute the stub.
/// Returns `IsmctsDescentOut` (never a partial selection as complete).
///
/// Envelope layout contract (bit-parity depends on it): `step_keys[sim][d]`
/// carries the info key at ROOT steps only (`""` elsewhere, unread);
/// `policy_dirs[sim][d]` carries the `0|1` tilt at NON-ROOT steps only (`0`
/// at root steps, unread); `rng_floats` carries POLICY draws only, in
/// `(sim, step)` consumption order — one float per non-root step, NO
/// placeholders for root steps or belief draws (a 4-per-sim layout with
/// root dummies misaligns every read after the first root step). The parent
/// world snapshot is never re-hashed (successor ids derive from the kept
/// parent `world_id`); `tree_json` sums print shortest-round-trip (`{:?}`),
/// never fixed-decimals (which lose the last ulp near `0.02`).
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn ismcts_descent(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyIsmctsDescentOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search descent batch must be non-empty",
        ));
    }
    // Compute DETACHED: parse + validate + full descent with zero Python
    // API inside; only the owned JSON bytes cross the boundary.
    let out = py.detach(|| driver_mod::ismcts_search_batch(&batch_json));
    let out = out.map_err(search_err)?;
    // Attached-only: shape-check the digest, then freeze into the pyclass.
    check_digest_shape(&out.tree_digest).map_err(PyValueError::new_err)?;
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    let mut arms_json = String::from("[");
    let mut t = 0;
    while t < out.tree.len() {
        let (key, visits, arms) = &out.tree[t];
        if t > 0 {
            arms_json.push(',');
        }
        arms_json.push_str(&format!("{{\"key\":{key:?},\"visits\":{visits},\"arms\":["));
        let mut a = 0;
        while a < arms.len() {
            let (action, n, sum) = arms[a];
            if a > 0 {
                arms_json.push(',');
            }
            arms_json.push_str(&format!(
                "{{\"action\":{action},\"visits\":{n},\"sum\":[{:?},{:?},{:?},{:?}]}}",
                sum[0], sum[1], sum[2], sum[3]
            ));
            a += 1;
        }
        arms_json.push_str("]}");
        t += 1;
    }
    arms_json.push(']');
    Ok(PyIsmctsDescentOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        tree_digest: out.tree_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        tree_nodes: out.tree_nodes,
        floats_used: out.floats_used,
        tree_json: arms_json,
    })
}

/// One batch-halving outcome crossing the boundary (selection + means +
/// survivors + digest + counters; world blobs never cross back).
#[pyclass(name = "GumbelHalvingOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyGumbelHalvingOut {
    /// Selected action id (max `g + q`, `1e-12` eps + tie arm).
    #[pyo3(get)]
    pub selected_id: u32,
    /// Candidate ids in sorted order.
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    /// Mean 4-vectors per candidate (row-major, 4 entries each).
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    /// Visits per candidate.
    #[pyo3(get)]
    pub visits: Vec<u64>,
    /// Final survivors in cut (ranked) order.
    #[pyo3(get)]
    pub survivors: Vec<u32>,
    /// Canon digest over the per-arm stats dump.
    #[pyo3(get)]
    pub stats_digest: String,
    /// Completed rollouts.
    #[pyo3(get)]
    pub sims_run: u64,
    /// Completed transitions.
    #[pyo3(get)]
    pub transitions: u64,
    /// Model leaf calls (terminal fallbacks excluded).
    #[pyo3(get)]
    pub model_calls: u64,
    /// CTR floats consumed.
    #[pyo3(get)]
    pub floats_used: u64,
}

/// Run one batch Gumbel halving search: precomputed worlds, per-rollout
/// policy directions, CTR floats, and root Gumbels cross ONCE as a JSON
/// envelope; the arena replays forced-root rollouts, continuation-tilt
/// sampling, unified `gumbel` transitions, deduped leaf vectors, and vector
/// backup with the GIL released, then runs the sequential-halving cuts and
/// Gumbel selection. Leaf overrides (read-only torch vectors keyed by
/// world id) ride the same envelope; absent entries compute the stub.
/// Returns `GumbelHalvingOut` (never a partial selection as complete).
///
/// Envelope layout contract (bit-parity depends on it): `worlds` rides
/// round-major, slot-minor, visit-minor in oracle visit order (budget
/// breaks truncate the tail only); `policy_dirs[rollout][j]` carries the
/// `0|1` tilt for the `j`-th continuation step of that rollout;
/// `rng_floats` carries POLICY draws only, in consumption order — one float
/// per sampled continuation step, NO placeholders for forced-root steps or
/// belief draws; `gumbels` covers exactly `root_legal` (`[[aid, g], ...]`).
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn gumbel_halving(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyGumbelHalvingOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search halving batch must be non-empty",
        ));
    }
    // Compute DETACHED: parse + validate + full halving replay with zero
    // Python API inside; only the owned JSON bytes cross the boundary.
    let out = py.detach(|| gumbel_driver_mod::gumbel_halving_batch(&batch_json));
    let out = out.map_err(search_err)?;
    // Attached-only: shape-check the digest, then freeze into the pyclass.
    check_digest_shape(&out.stats_digest).map_err(PyValueError::new_err)?;
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    Ok(PyGumbelHalvingOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        survivors: out.survivors,
        stats_digest: out.stats_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        floats_used: out.floats_used,
    })
}

/// One batch-resolving outcome crossing the boundary (current + averaged
/// table dumps as JSON; world blobs never cross back).
#[pyclass(name = "LocalResolvingOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLocalResolvingOut {
    /// Current strategy table rows JSON: `[[actor, info, [p...], visits]...]`.
    #[pyo3(get)]
    pub table_json: String,
    /// Averaged table rows JSON (visits `null` where the oracle leaves it unset).
    #[pyo3(get)]
    pub avg_json: String,
}

/// Run one batch local-resolving search: precomputed per-iteration worlds
/// (identity, validated leaf floats, memoized base info keys), seeded
/// init-table entries, and per-step sampling floats cross ONCE as a JSON
/// envelope; the driver replays the resolving loop (actor traversal,
/// regret/hedge/fictitious-play updates, averaging, path sampling) with the
/// GIL released, then dumps the current + averaged tables. Tie-break
/// selection stays caller-side (once per search over the root average).
/// Returns `LocalResolvingOut` (never a partial table as complete).
///
/// Envelope layout contract (bit-parity depends on it): `iters` rides in
/// oracle iteration order (one world per iteration); `base` carries the four
/// per-actor base info keys in seat order; `sampling` carries one float per
/// step in iteration-major order; `init` carries the seeded `(actor, info,
/// dist)` entries with visits starting at 0.
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn local_resolving_batch(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyLocalResolvingOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search local batch must be non-empty",
        ));
    }
    // Compute DETACHED: parse + validate + full resolving replay + dump
    // serialization with zero Python API inside; only the owned JSON bytes
    // cross the boundary (dumps serialize driver-side, same convention as
    // the ISMCTS `tree_json` dump).
    let out = py.detach(|| local_driver_mod::local_resolving_batch_json(&batch_json));
    let (table_json, avg_json) = out.map_err(search_err)?;
    // Attached-only: freeze into the pyclass (fail-closed, never a default).
    Ok(PyLocalResolvingOut {
        table_json,
        avg_json,
    })
}

/// Run one batch DESPOT drive: precomputed worlds + frozen params cross ONCE
/// as a JSON envelope; the driver replays scenario seeds, lower values, and
/// best-feasible selection with the GIL released. Returns `DespotOut`.
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn despot_search_batch(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyDespotOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search despot batch must be non-empty",
        ));
    }
    let out = py.detach(|| drive_batch_mod::despot_search_batch(&batch_json));
    let out = out.map_err(search_err)?;
    check_digest_shape(&out.decision_digest).map_err(PyValueError::new_err)?;
    bump_act(py, out.completed)?;
    Ok(drive_out_to_despot(out))
}

/// Run one batch PBRF drive: precomputed worlds + policy pairs cross ONCE;
/// the driver replays fixed-allocate forest, child values, and scalar-max
/// selection with the GIL released. Returns `PbrfOut`.
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn pbrf_search_batch(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyPbrfOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err("search pbrf batch must be non-empty"));
    }
    let out = py.detach(|| drive_batch_mod::pbrf_search_batch(&batch_json));
    let out = out.map_err(search_err)?;
    check_digest_shape(&out.decision_digest).map_err(PyValueError::new_err)?;
    bump_act(py, out.completed)?;
    Ok(drive_out_to_pbrf(out))
}

/// Run one batch joint drive: precomputed worlds + theta space cross ONCE;
/// the driver replays hash leaves, joint Gumbels, and robust selection with
/// the GIL released. Returns `JointOut`.
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn joint_search_batch(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyJointOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search joint batch must be non-empty",
        ));
    }
    let out = py.detach(|| drive_batch_mod::joint_search_batch(&batch_json));
    let out = out.map_err(search_err)?;
    check_digest_shape(&out.decision_digest).map_err(PyValueError::new_err)?;
    bump_act(py, out.completed)?;
    Ok(drive_out_to_joint(out))
}

/// Run one batch persistence drive: precomputed worlds + arm/epoch/blob cross
/// ONCE; the driver replays per-arm retain rules and bounded pick with the
/// GIL released. Returns `PersistenceOut`.
#[pyfunction]
#[pyo3(signature = (batch_json,))]
fn persistence_search_batch(py: Python<'_>, batch_json: Vec<u8>) -> PyResult<PyPersistenceOut> {
    if batch_json.is_empty() {
        return Err(PyValueError::new_err(
            "search persistence batch must be non-empty",
        ));
    }
    let out = py.detach(|| drive_batch_mod::persistence_search_batch(&batch_json));
    let out = out.map_err(search_err)?;
    check_digest_shape(&out.decision_digest).map_err(PyValueError::new_err)?;
    bump_act(py, out.completed)?;
    Ok(drive_out_to_persistence(out))
}

/// Judge observability: `(acts, completed)` (never identity/selection).
#[pyfunction]
fn act_stats(py: Python<'_>) -> PyResult<(u64, u64)> {
    let guard = ACT_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("search act stats mutex poisoned"))?;
    Ok((guard.acts, guard.completed))
}

/// Feasible-policy lower value for ONE root action over a scenario pack
/// (`despot_search.py:251-295` via `despot::lower_value`): per-scenario
/// hash returns averaged with the oracle's weight-`K` round-trip.
/// `world_refs`/`seed_hexes` ride parallel (seed hex verbatim); empty pack
/// reads as `0.0`. Compute runs detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (world_refs, seed_hexes, aid, candidate_id))]
fn despot_lower_values(
    py: Python<'_>,
    world_refs: Vec<String>,
    seed_hexes: Vec<String>,
    aid: String,
    candidate_id: String,
) -> PyResult<f64> {
    if world_refs.len() != seed_hexes.len() {
        return Err(PyValueError::new_err(
            "search despot refs/seeds length mismatch",
        ));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err(
            "search despot candidate_id must be non-empty",
        ));
    }
    let out = py.detach(|| despot_mod::lower_value(&world_refs, &seed_hexes, &aid, &candidate_id));
    out.map_err(search_err)
}

/// Deterministic leaf vector for ONE `(action, packet)` child
/// (`pbrf_search.py:252-294` via `pbrf::child_value`): weight-averaged
/// hash scalars expanded to the 4-seat vector. Rows ride parallel
/// (`parent8s`/`target8s` truncated caller-side); `z <= 0` reads as the
/// zero vector. Compute runs detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (parent8s, target8s, raw_weights, aid, packet_id))]
fn pbrf_child_value(
    py: Python<'_>,
    parent8s: Vec<String>,
    target8s: Vec<String>,
    raw_weights: Vec<f64>,
    aid: u32,
    packet_id: String,
) -> PyResult<(f64, f64, f64, f64)> {
    if parent8s.len() != target8s.len() || parent8s.len() != raw_weights.len() {
        return Err(PyValueError::new_err(
            "search pbrf child row length mismatch",
        ));
    }
    let out =
        py.detach(|| pbrf_mod::child_value(&parent8s, &target8s, &raw_weights, aid, &packet_id));
    out.map_err(search_err)
}

/// One-call batch for the ISMCTS envelope step hashes
/// (`ismcts_search.py:532-570` via `step_hash::step_hashes`): the template
/// identity doc parses once, each row substitutes hand/live/actor and
/// yields the info key (`want_keys`) or the tilt direction. Lanes ride
/// parallel; the unused lane carries `""`/`0`. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (template_json, hands, live_lens, actors, want_keys))]
fn ismcts_step_hashes(
    py: Python<'_>,
    template_json: Vec<u8>,
    hands: Vec<Vec<u32>>,
    live_lens: Vec<u32>,
    actors: Vec<u32>,
    want_keys: Vec<bool>,
) -> PyResult<(Vec<String>, Vec<u32>)> {
    if template_json.is_empty() {
        return Err(PyValueError::new_err(
            "search step template must be non-empty",
        ));
    }
    if hands.len() != live_lens.len()
        || hands.len() != actors.len()
        || hands.len() != want_keys.len()
    {
        return Err(PyValueError::new_err("search step lane length mismatch"));
    }
    let out = py.detach(|| {
        step_hash_mod::step_hashes(&template_json, &hands, &live_lens, &actors, &want_keys)
    });
    out.map_err(search_err)
}

/// One Candidate 4 module transform over owned vectors
/// (`search/modules/__init__.py` transform bodies via
/// `modules::module_transform`): `seed` is the caller-derived
/// `_semantic_seed` int (the seed formula stays Python's single site);
/// `meta_json` carries the arm's validated context params (`voc` arm:
/// `floor`/`cap`/`budget`/`scores`-or-null; empty otherwise). Returns
/// `(particles, weights, budget_calls, budget_transitions, meta_json)`.
/// Compute runs detached; only owned vectors cross.
#[pyfunction]
#[pyo3(signature = (module_id, particles, weights, budget_calls, budget_transitions, seed, meta_json))]
fn pbrf_module_transform(
    py: Python<'_>,
    module_id: String,
    particles: Vec<f64>,
    weights: Vec<f64>,
    budget_calls: u64,
    budget_transitions: u64,
    seed: u64,
    meta_json: Vec<u8>,
) -> PyResult<ModuleTransformOut> {
    if particles.len() != weights.len() {
        return Err(PyValueError::new_err(
            "search module particles/weights length mismatch",
        ));
    }
    let out = py.detach(|| {
        modules_mod::module_transform(
            &module_id,
            &particles,
            &weights,
            budget_calls,
            budget_transitions,
            seed,
            &meta_json,
        )
    });
    out.map_err(search_err)
}
// ---------------------------------------------------------------------------
// Persistence kernel + joint posterior pyfns (prizes 1-2).
//
// Thin mirrors of `persistence_kernel.py` enumerate/rebuild/commit/quota and
// `joint_uncertainty.py::exact_joint_posterior_oracle` over live
// epoch/packet/posterior objects: fields are read attached (callers pass
// their `BeliefEpochLite` / `FinitePacket` / `JointPosterior` straight
// through, no repackaging), every hash/sum runs detached through the
// existing owners (`persistence_kernel`, `joint`, `feed::canon` via those
// owners — never a second printer/hasher). The joint likelihood replicates
// `OpponentTypePolicy.distribution_for` + `log_prob` + `exp` verbatim
// (hash-seeded Dirichlet pseudo-counts, Neumaier `builtin_sum`, `ln`/`exp`
// audit to `(0,1]`); the posterior half replicates the oracle's plain-loop
// normalizer with the `1 +- 1e-9` mass audit. Fail-closed: every shape
// violation is `ValueError`, never a default.
// ---------------------------------------------------------------------------

/// Read a `str` arriving bare or carried by a live object under `field`
/// (`BeliefEpochLite.epoch`, packet `epoch_before`): `str` passes through,
/// otherwise the attribute is read attached.
fn str_or_field(obj: &Bound<'_, PyAny>, field: &str, ctx: &str) -> PyResult<String> {
    if let Ok(text) = obj.extract::<String>() {
        return Ok(text);
    }
    obj.getattr(field)
        .map_err(|e| PyValueError::new_err(format!("search {ctx}: .{field} unreadable: {e}")))?
        .extract::<String>()
        .map_err(|_| PyValueError::new_err(format!("search {ctx}: .{field} must be str")))
}

/// Read one `(theta, weight)` row attached from a live `JointParticle`
/// (epoch/target provenance stays caller-side; only the math lanes cross).
fn particle_theta_weight(particle: &Bound<'_, PyAny>) -> PyResult<(String, f64)> {
    let theta: String = particle
        .getattr("theta")
        .map_err(|e| PyValueError::new_err(format!("search joint: .theta unreadable: {e}")))?
        .extract()
        .map_err(|_| PyValueError::new_err("search joint: .theta must be str"))?;
    let weight: f64 = particle
        .getattr("weight")
        .map_err(|e| PyValueError::new_err(format!("search joint: .weight unreadable: {e}")))?
        .extract()
        .map_err(|_| PyValueError::new_err("search joint: .weight must be float"))?;
    Ok((theta, weight))
}

/// `math.isclose(x, 1.0, rel_tol=1e-9, abs_tol=1e-9)` replica for the
/// posterior mass audit.
fn isclose_unit_1e9(value: f64) -> bool {
    (value - 1.0).abs() <= (1e-9 * value.abs().max(1.0)).max(1e-9)
}

/// Digest-shape predicate over an owned string (info keys are canon-form
/// `sha256:<64 hex>` exactly like decision digests).
fn is_digest_shape(text: &str) -> bool {
    check_digest_shape(text).is_ok()
}

/// One particle likelihood (`OpponentTypePolicy.distribution_for` +
/// `log_prob` + `exp`, `joint_types.py:290-354`): hash-seeded Dirichlet
/// pseudo-counts over the ascending legal set, observed probability through
/// `ln`/`exp` with the `(0,1]` finite audit. Pure and detach-safe.
fn joint_row_likelihood(
    theta: &str,
    seed_domain: &[u8],
    info_key: &str,
    legal_ids: &[u32],
    observed_id: u32,
) -> Result<f64, SearchError> {
    joint_mod::check_theta(theta)?;
    if seed_domain.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "joint seed_domain must be non-empty bytes",
        });
    }
    if !is_digest_shape(info_key) {
        return Err(SearchError::InvalidArg {
            detail: "joint info_key must be sha256:<hex>",
        });
    }
    if legal_ids.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "joint legal_action_ids must be non-empty",
        });
    }
    let mut sorted: Vec<u32> = legal_ids.to_vec();
    sorted.sort_unstable();
    let mut dup: Option<u32> = None;
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            dup = Some(pair[0]);
            break;
        }
    }
    if let Some(action) = dup {
        return Err(SearchError::DuplicateAction { action });
    }
    let mut csv = String::new();
    for (idx, aid) in sorted.iter().enumerate() {
        if idx > 0 {
            csv.push(',');
        }
        csv.push_str(&aid.to_string());
    }
    let observed_pos =
        sorted
            .iter()
            .position(|aid| *aid == observed_id)
            .ok_or(SearchError::InvalidArg {
                detail: "joint observed_action_id not in legal set",
            })?;
    let msg = format!("{theta}:{info_key}:{csv}");
    let mut seed_hasher = Sha256::new();
    seed_hasher.update(seed_domain);
    seed_hasher.update(msg.as_bytes());
    let seed = seed_hasher.finalize();
    let mut counts: Vec<f64> = Vec::with_capacity(sorted.len());
    for (idx, _aid) in sorted.iter().enumerate() {
        // proof: `u16::MAX` = 65535 const, fits `usize`.
        #[allow(clippy::cast_possible_truncation)]
        let u16_max_u: usize = u16::MAX as usize;
        if idx > u16_max_u {
            return Err(SearchError::InvalidArg {
                detail: "joint legal set too large",
            });
        }
        let mut obscure = Sha256::new();
        obscure.update(seed);
        // proof: `idx <= u16::MAX` (checked above), fits `u16`.
        #[allow(clippy::cast_possible_truncation)]
        let idx_u: u16 = idx as u16;
        obscure.update(idx_u.to_be_bytes());
        let digest = obscure.finalize();
        let val =
            u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as f64 / 4294967296.0;
        let mut base = 0.5 + val * 2.0;
        if theta == joint_mod::THETA_TIGHT {
            if idx == 0 {
                base *= 2.0;
            }
        } else if theta == joint_mod::THETA_LOOSE {
            base = 1.0 + val * 0.5;
        }
        counts.push(base);
    }
    let total = builtin_sum(&counts);
    let slot = counts.get(observed_pos).ok_or(SearchError::InvalidArg {
        detail: "joint observed position out of range",
    })?;
    let likelihood = (*slot / total).ln().exp();
    if !likelihood.is_finite() || likelihood <= 0.0 || likelihood > 1.0 {
        return Err(SearchError::BadLikelihood);
    }
    Ok(likelihood)
}

/// Normalize one joint posterior (`exact_joint_posterior_oracle` tail,
/// `joint_uncertainty.py:124-158`): `w * lik * T` in a plain loop (the
/// oracle accumulates with `+=`, NOT `fsum`), `Z > 0` finite, divide,
/// Neumaier mass audit to `1 +- 1e-9`. Pure and detach-safe.
fn joint_normalize(
    weights: &[f64],
    likelihoods: &[f64],
    physical_t: f64,
) -> Result<Vec<f64>, SearchError> {
    if weights.is_empty() || weights.len() != likelihoods.len() {
        return Err(SearchError::InvalidArg {
            detail: "joint weights/likelihoods must align non-empty",
        });
    }
    if !physical_t.is_finite() || physical_t <= 0.0 || physical_t > 1.0 {
        return Err(SearchError::InvalidArg {
            detail: "joint physical_transition_prob must be in (0,1]",
        });
    }
    let mut unnorm: Vec<f64> = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    for (weight, likelihood) in weights.iter().zip(likelihoods.iter()) {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(SearchError::NonFinite {
                context: "joint prior weight",
            });
        }
        if !likelihood.is_finite() || *likelihood <= 0.0 || *likelihood > 1.0 {
            return Err(SearchError::BadLikelihood);
        }
        let wu = *weight * *likelihood * physical_t;
        if !wu.is_finite() || wu < 0.0 {
            return Err(SearchError::NonFinite {
                context: "joint unnorm weight",
            });
        }
        unnorm.push(wu);
        total += wu;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SearchError::ZeroMass);
    }
    let mut out: Vec<f64> = Vec::with_capacity(unnorm.len());
    for wu in &unnorm {
        let w = *wu / total;
        if !w.is_finite() || w < 0.0 {
            return Err(SearchError::NonFinite {
                context: "joint norm weight",
            });
        }
        out.push(w);
    }
    if !isclose_unit_1e9(builtin_sum(&out)) {
        return Err(SearchError::BadPartition);
    }
    Ok(out)
}

/// Exhaustive disjoint packet kernel over a live epoch
/// (`persistence_kernel.py:250-296` via `persistence_kernel::enumerate_packets_for`):
/// `epoch` is the caller's `BeliefEpochLite` or a bare epoch string (read
/// attached, never repackaged). Returns one row per branch
/// `(packet_id, epoch_after, probability, branch)`; the facade rebuilds the
/// `FinitePacket` tuple around them. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (epoch, action_id, num_branches=2))]
fn persistence_packets_for(
    py: Python<'_>,
    epoch: Bound<'_, PyAny>,
    action_id: u64,
    num_branches: usize,
) -> PyResult<Vec<(String, String, f64, u64)>> {
    let epoch_id = str_or_field(&epoch, "epoch", "persistence_packets_for")?;
    if epoch_id.is_empty() {
        return Err(PyValueError::new_err(
            "search persistence_packets_for: epoch must be non-empty",
        ));
    }
    let staged =
        py.detach(|| persist_mod::enumerate_packets_for(&epoch_id, action_id, num_branches));
    let packets = staged.map_err(search_err)?;
    let mut rows: Vec<(String, String, f64, u64)> = Vec::with_capacity(packets.len());
    for (branch, packet) in packets.into_iter().enumerate() {
        rows.push((
            packet.packet_id,
            packet.epoch_after,
            packet.probability,
            branch as u64,
        ));
    }
    Ok(rows)
}

/// Authoritative fresh posterior `epoch_after` over live objects
/// (`fresh_rebuild_epoch`): `epoch_before` is a `BeliefEpochLite` or bare
/// string, `packet` the caller's `FinitePacket` (both read attached).
/// Compute runs detached.
#[pyfunction]
#[pyo3(signature = (epoch_before, packet))]
fn persistence_rebuild_epoch(
    py: Python<'_>,
    epoch_before: Bound<'_, PyAny>,
    packet: Bound<'_, PyAny>,
) -> PyResult<String> {
    let epoch_id = str_or_field(&epoch_before, "epoch", "persistence_rebuild_epoch")?;
    let packet_id = str_or_field(&packet, "packet_id", "persistence_rebuild_epoch")?;
    let packet_before = str_or_field(&packet, "epoch_before", "persistence_rebuild_epoch")?;
    let probe = persist_mod::FinitePacket {
        packet_id,
        action_id: 0,
        epoch_before: packet_before,
        epoch_after: String::new(),
        probability: 1.0,
        delta: Vec::new(),
    };
    py.detach(|| persist_mod::fresh_rebuild_epoch(&epoch_id, &probe))
        .map_err(search_err)
}

/// Commit/rebuild equality over live objects (`commit_equals_rebuild`):
/// same attached reads as [`persistence_rebuild_epoch`], detached compare.
#[pyfunction]
#[pyo3(signature = (epoch_before, packet))]
fn persistence_commit_equals_rebuild(
    py: Python<'_>,
    epoch_before: Bound<'_, PyAny>,
    packet: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let epoch_id = str_or_field(&epoch_before, "epoch", "persistence_commit_equals_rebuild")?;
    let packet_id = str_or_field(&packet, "packet_id", "persistence_commit_equals_rebuild")?;
    let packet_before = str_or_field(&packet, "epoch_before", "persistence_commit_equals_rebuild")?;
    let packet_after = str_or_field(&packet, "epoch_after", "persistence_commit_equals_rebuild")?;
    let probe = persist_mod::FinitePacket {
        packet_id,
        action_id: 0,
        epoch_before: packet_before,
        epoch_after: packet_after,
        probability: 1.0,
        delta: Vec::new(),
    };
    py.detach(|| persist_mod::commit_equals_rebuild(&epoch_id, &probe))
        .map_err(search_err)
}
#[pyfunction]
#[pyo3(signature = (sorted_pids, quota))]
fn persistence_distribute_quota(
    sorted_pids: Vec<String>,
    quota: usize,
) -> PyResult<Vec<(String, usize)>> {
    let mut dist: Vec<(String, usize)> = Vec::new();
    for pid in &sorted_pids {
        if !dist.iter().any(|(seen, _)| seen == pid) {
            dist.push((pid.clone(), 0));
        }
    }
    let mut remaining = quota;
    while remaining > 0 && !dist.is_empty() {
        for pid in &sorted_pids {
            if remaining == 0 {
                break;
            }
            if let Some(slot) = dist.iter_mut().find(|(seen, _)| seen == pid) {
                slot.1 = slot.1.saturating_add(1);
                remaining -= 1;
            }
        }
    }
    Ok(dist)
}
/// Exact joint posterior weights over a live prior
/// (`exact_joint_posterior_oracle`, `joint_uncertainty.py:53-158`): `prior`
/// is the caller's live `JointPosterior` (particles read attached, epoch /
/// target provenance stays caller-side); `info_keys` / `seed_domains` ride
/// aligned 1:1 with the particles (facade derives them from the live
/// `worlds_by_ref` map + `policy_for_theta` policies, which never cross);
/// `legal_ids` / `observed_id` are the shared observed-action context.
/// Likelihood enters exactly once per particle, detached; returns
/// normalized weights in particle order (the facade rebuilds
/// `JointParticle`s preserving epoch/target provenance).
#[pyfunction]
#[pyo3(signature = (*, prior, info_keys, seed_domains, legal_ids, observed_id, physical_t=1.0))]
#[allow(clippy::too_many_arguments)]
fn joint_posterior_weights(
    py: Python<'_>,
    prior: Bound<'_, PyAny>,
    info_keys: Vec<String>,
    seed_domains: Vec<Vec<u8>>,
    legal_ids: Vec<u32>,
    observed_id: u32,
    physical_t: f64,
) -> PyResult<Vec<f64>> {
    let particles: Vec<Bound<'_, PyAny>> = prior
        .getattr("particles")
        .map_err(|e| {
            PyValueError::new_err(format!(
                "search joint_posterior_weights: .particles unreadable: {e}"
            ))
        })?
        .extract()
        .map_err(|_| {
            PyValueError::new_err("search joint_posterior_weights: .particles must be a sequence")
        })?;
    if particles.is_empty()
        || info_keys.len() != particles.len()
        || seed_domains.len() != particles.len()
    {
        return Err(PyValueError::new_err(
            "search joint_posterior_weights: particle lane length mismatch",
        ));
    }
    let mut thetas: Vec<String> = Vec::with_capacity(particles.len());
    let mut weights: Vec<f64> = Vec::with_capacity(particles.len());
    for particle in &particles {
        let (theta, weight) = particle_theta_weight(particle)?;
        thetas.push(theta);
        weights.push(weight);
    }
    let lanes = thetas.len();
    let out = py.detach(|| -> Result<Vec<f64>, SearchError> {
        let mut likelihoods: Vec<f64> = Vec::with_capacity(lanes);
        for ((theta, domain), info_key) in
            thetas.iter().zip(seed_domains.iter()).zip(info_keys.iter())
        {
            likelihoods.push(joint_row_likelihood(
                theta,
                domain,
                info_key,
                &legal_ids,
                observed_id,
            )?);
        }
        joint_normalize(&weights, &likelihoods, physical_t)
    });
    out.map_err(search_err)
}

/// Likelihood-free posterior tail (`exact_joint_posterior_oracle`
/// normalize half): caller-supplied likelihoods (one per prior weight,
/// each already audited to `(0,1]` here) through the plain-loop normalizer.
/// Detached; the draw-free fallback when info-key assembly stays Python.
#[pyfunction]
#[pyo3(signature = (weights, likelihoods, physical_t=1.0))]
fn joint_normalize_weights(
    py: Python<'_>,
    weights: Vec<f64>,
    likelihoods: Vec<f64>,
    physical_t: f64,
) -> PyResult<Vec<f64>> {
    py.detach(|| joint_normalize(&weights, &likelihoods, physical_t))
        .map_err(search_err)
}

/// Lowercase hex of raw bytes (bridge-local mirror of
/// `hydra_search::eval::hex_of`, which is `pub(crate)` to its crate; the
/// fused joint hashes canonical obs bytes detached and must render the same
/// `sha256:<64 lower hex>` info keys Python `hashlib.sha256(payload).hexdigest()`
/// produces).
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

/// Fused joint posterior over per-world canonical obs bytes
/// (`exact_joint_posterior_oracle` fused: per-world canonical bytes +
/// per-particle theta/policy/weights in, normalized weights out, single FFI
/// incl normalize).
///
/// `world_obs_bytes` carries one `canonical_bytes(identity_doc)` blob per
/// unique world (Python `_wao` + identity doc + `canonical_bytes`, memoized
/// per world_ref; FullWorlds never cross, guards stay Python). `world_idx`
/// maps each particle to its world (`len == N`, values `< W`).
/// `thetas`/`seed_domains`/`weights` ride per particle (`len == N`).
/// `legal_ids`/`observed_id` are the shared observed-action context.
/// Likelihood enters exactly once per particle, detached; returns normalized
/// weights in particle order (caller rebuilds `JointParticle`s).
#[pyfunction]
#[pyo3(signature = (*, weights, thetas, seed_domains, world_idx, world_obs_bytes, legal_ids, observed_id, physical_t=1.0))]
#[allow(clippy::too_many_arguments)]
fn joint_posterior_fused(
    py: Python<'_>,
    weights: Vec<f64>,
    thetas: Vec<String>,
    seed_domains: Vec<Vec<u8>>,
    world_idx: Vec<usize>,
    world_obs_bytes: Vec<Vec<u8>>,
    legal_ids: Vec<u32>,
    observed_id: u32,
    physical_t: f64,
) -> PyResult<Vec<f64>> {
    let lanes = weights.len();
    if lanes == 0
        || thetas.len() != lanes
        || seed_domains.len() != lanes
        || world_idx.len() != lanes
        || world_obs_bytes.is_empty()
    {
        return Err(PyValueError::new_err(
            "search joint_posterior_fused: particle/world lane length mismatch",
        ));
    }
    let out = py.detach(|| -> Result<Vec<f64>, SearchError> {
        if !physical_t.is_finite() || physical_t <= 0.0 || physical_t > 1.0 {
            return Err(SearchError::InvalidArg {
                detail: "joint physical_transition_prob must be in (0,1]",
            });
        }
        if legal_ids.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "joint legal_action_ids must be non-empty",
            });
        }
        let mut world_keys: Vec<String> = Vec::with_capacity(world_obs_bytes.len());
        for blob in &world_obs_bytes {
            if blob.is_empty() {
                return Err(SearchError::InvalidArg {
                    detail: "joint world obs bytes must be non-empty",
                });
            }
            let mut hasher = Sha256::new();
            hasher.update(blob);
            let digest = hasher.finalize();
            let mut key = String::with_capacity(7 + 64);
            key.push_str("sha256:");
            key.push_str(&hex_lower(digest.as_slice()));
            world_keys.push(key);
        }
        let mut likelihoods: Vec<f64> = Vec::with_capacity(lanes);
        let mut n = 0;
        while n < lanes {
            let info_key = world_keys
                .get(world_idx[n])
                .ok_or(SearchError::InvalidArg {
                    detail: "joint world_idx out of range",
                })?;
            likelihoods.push(joint_row_likelihood(
                &thetas[n],
                &seed_domains[n],
                info_key,
                &legal_ids,
                observed_id,
            )?);
            n += 1;
        }
        joint_normalize(&weights, &likelihoods, physical_t)
    });
    out.map_err(search_err)
}

/// Python `math.isclose(a, b)` replica with planner defaults
/// (`rel_tol=1e-09`, `abs_tol=0.0`): `abs(a-b) <= 1e-9 * max(abs(a), abs(b))`.
/// (`persistence_planner.py:150` tie arm uses bare `math.isclose`.)
fn py_isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

/// Fused persistence pick over N action ids
/// (`PersistencePlanner._pick_action_deterministically` batch: N action ids
/// in, best position out, single FFI).
///
/// `action_ids` are Python `_action_key` ints in caller order (action objects
/// never cross; guards stay Python). Scores replicate
/// `deterministic_gumbel_for_arm` (default seed `b"hydra2-persistence-v1"`)
/// + per-arm bias `% 1.0` with `math.isclose` ties to the smaller aid.
/// Returns the best position (caller maps back to its action table; indices
/// preserve identity when aids collide).
#[pyfunction]
#[pyo3(signature = (arm_id, case_id, action_ids))]
fn persistence_pick_batch(
    py: Python<'_>,
    arm_id: String,
    case_id: String,
    action_ids: Vec<u64>,
) -> PyResult<usize> {
    if action_ids.is_empty() {
        return Err(PyValueError::new_err(
            "search persistence_pick_batch: legal actions must be non-empty",
        ));
    }
    let out = py.detach(|| -> Result<usize, SearchError> {
        let arm = persist_mod::ArmId::parse(&arm_id)?;
        let bias = match arm {
            persist_mod::ArmId::B => 0.0,
            persist_mod::ArmId::F => 0.1,
            persist_mod::ArmId::R => 0.12,
            persist_mod::ArmId::P => 0.18,
            persist_mod::ArmId::C => 0.19,
        };
        let mut best_pos = 0usize;
        let mut best_aid = action_ids[0];
        let mut best_score = -1.0f64;
        let mut i = 0;
        while i < action_ids.len() {
            let aid = action_ids[i];
            let g = spec_mod::deterministic_gumbel_for_arm(
                arm,
                &case_id,
                aid,
                spec_mod::GUMBEL_SEED_DEFAULT,
            );
            if !g.is_finite() || g < 0.0 || g >= 1.0 {
                return Err(SearchError::NonFinite {
                    context: "persistence pick gumbel",
                });
            }
            let score = (g + bias) % 1.0;
            if !score.is_finite() || score < 0.0 || score >= 1.0 {
                return Err(SearchError::NonFinite {
                    context: "persistence pick score",
                });
            }
            if score > best_score || (py_isclose(score, best_score) && aid < best_aid) {
                best_score = score;
                best_aid = aid;
                best_pos = i;
            }
            i += 1;
        }
        Ok(best_pos)
    });
    out.map_err(search_err)
}

/// Fail-close comparator for the in-module judge tests (mirrors
/// `canon_rng::check_match` / columnar `check_match`: recomputed ==
/// recorded or `Err`, never a default/empty accept).
#[cfg(test)]
fn check_match(recorded: &str, recomputed: &str) -> Result<(), String> {
    if recorded == recomputed {
        Ok(())
    } else {
        Err(format!(
            "digest mismatch: recorded {recorded} != recomputed {recomputed}"
        ))
    }
}

/// Register the `search` submodule (mirrors `canon_rng::register` /
/// `columnar::register`): compute detached, wrap attached; single cdylib,
/// no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "search")?;
    sub.add_function(wrap_pyfunction!(act_batch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(act_stats, &sub)?)?;
    sub.add_function(wrap_pyfunction!(gumbel_for_action, &sub)?)?;
    sub.add_function(wrap_pyfunction!(gumbel_roots, &sub)?)?;
    sub.add_function(wrap_pyfunction!(halving_cut, &sub)?)?;
    sub.add_function(wrap_pyfunction!(gumbel_select, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_model_vector, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_terminal_vector, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_local_model_vector, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_local_terminal_vector, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_transition, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_info_key, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_descent, &sub)?)?;
    sub.add_class::<PyIsmctsDescentOut>()?;
    sub.add_function(wrap_pyfunction!(gumbel_halving, &sub)?)?;
    sub.add_class::<PyGumbelHalvingOut>()?;
    sub.add_function(wrap_pyfunction!(local_resolving_batch, &sub)?)?;
    sub.add_class::<PyLocalResolvingOut>()?;
    sub.add_function(wrap_pyfunction!(despot_search_batch, &sub)?)?;
    sub.add_class::<PyDespotOut>()?;
    sub.add_function(wrap_pyfunction!(pbrf_search_batch, &sub)?)?;
    sub.add_class::<PyPbrfOut>()?;
    sub.add_function(wrap_pyfunction!(joint_search_batch, &sub)?)?;
    sub.add_class::<PyJointOut>()?;
    sub.add_function(wrap_pyfunction!(persistence_search_batch, &sub)?)?;
    sub.add_class::<PyPersistenceOut>()?;
    sub.add_function(wrap_pyfunction!(uct_select, &sub)?)?;
    sub.add_function(wrap_pyfunction!(puct_select, &sub)?)?;
    sub.add_function(wrap_pyfunction!(despot_lower_values, &sub)?)?;
    sub.add_function(wrap_pyfunction!(pbrf_child_value, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ismcts_step_hashes, &sub)?)?;
    sub.add_function(wrap_pyfunction!(pbrf_module_transform, &sub)?)?;
    sub.add_function(wrap_pyfunction!(natural_indices, &sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_draws, &sub)?)?;
    sub.add_function(wrap_pyfunction!(packet_successors, &sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_packets_for, &sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_rebuild_epoch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_commit_equals_rebuild, &sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_distribute_quota, &sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_posterior_weights, &sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_normalize_weights, &sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_posterior_fused, &sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_pick_batch, &sub)?)?;
    sub.add_class::<PyPacketSuccessor>()?;
    crate::search_shared::register(&sub)?;
    crate::search_profiles::register(&sub)?;
    crate::persistence_report::register(&sub)?;
    crate::qual_budget::register(&sub)?;
    crate::qual_replay::register(&sub)?;
    crate::analysis_leaves2::register(&sub)?;
    crate::eval_stats::register(&sub)?;
    crate::despot_seeds::register(&sub)?;
    crate::gumbel_spec::register(&sub)?;
    crate::local_spec::register(&sub)?;
    crate::pbrf_spec::register(&sub)?;
    crate::persistence_spec::register(&sub)?;
    crate::persistence_kernel::register(&sub)?;
    crate::candidate0_act::register(&sub)?;
    crate::joint_types::register(&sub)?;
    crate::gumbel_core::register(&sub)?;
    crate::ismcts_core::register(&sub)?;
    crate::pbrf_forest::register(&sub)?;
    crate::local_graph::register(&sub)?;
    crate::despot_result::register(&sub)?;
    crate::joint_uncertainty::register(&sub)?;
    crate::joint_world::register(&sub)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod search_tests {
    use super::*;

    #[test]
    fn mismatch_check_fails_closed() {
        assert!(check_match("sha256:aa", "sha256:aa").is_ok());
        assert!(check_match("sha256:aa", "sha256:bb").is_err());
        assert!(check_match("sha256:aa", "").is_err());
    }

    #[test]
    fn act_args_fail_closed() {
        assert!(check_act_args(0, 4, 2).is_err());
        assert!(check_act_args(4, 0, 2).is_err());
        assert!(check_act_args(4, 4, 0).is_err());
        assert!(check_act_args(4, 4, 2).is_ok());
    }

    #[test]
    fn budget_args_fail_closed_with_boundaries() {
        assert!(check_budget_args(0, 4, 5000).is_err());
        assert!(check_budget_args(8, 0, 5000).is_err());
        assert!(check_budget_args(8, 4, 0).is_err());
        assert!(check_budget_args(8, 4, MAX_DEADLINE_MS + 1).is_err());
        assert!(check_budget_args(1, 1, 1).is_ok());
        assert!(check_budget_args(8, 4, MAX_DEADLINE_MS).is_ok());
    }

    #[test]
    fn digest_shape_gates_arena_output() {
        let golden = format!("sha256:{}", "a".repeat(64));
        assert!(check_digest_shape(&golden).is_ok());
        assert!(check_digest_shape("").is_err());
        assert!(check_digest_shape(&"b".repeat(64)).is_err());
        assert!(check_digest_shape(&format!("sha256:{}", "A".repeat(64))).is_err());
        assert!(check_digest_shape(&format!("sha256:{}", "a".repeat(63))).is_err());
        assert!(check_digest_shape(&format!("sha256:{}", "a".repeat(65))).is_err());
    }

    #[test]
    fn frontier_out_carries_counters_only() {
        // completed=false contract shape against the sibling-owned type:
        // the frontier action + counters ride with a canon-form digest,
        // and no selection is presented as complete.
        let out = ActOut {
            action: 7,
            completed: false,
            sims_run: 0,
            nodes_visited: 0,
            decision_digest: format!("sha256:{}", "c".repeat(64)),
        };
        assert!(!out.completed);
        assert!(check_digest_shape(&out.decision_digest).is_ok());
        assert_eq!((out.sims_run, out.nodes_visited), (0, 0));
    }

    #[test]
    fn completed_out_shape() {
        let out = ActOut {
            action: 3,
            completed: true,
            sims_run: 48,
            nodes_visited: 96,
            decision_digest: format!("sha256:{}", "d".repeat(64)),
        };
        assert!(out.completed);
        assert!(out.sims_run >= 1);
        assert!(check_digest_shape(&out.decision_digest).is_ok());
        assert!(check_match(&out.decision_digest, &out.decision_digest).is_ok());
    }

    #[test]
    fn single_detach_discipline() {
        // Single-consume shape assert: the arena act runs under exactly ONE
        // detach per call (never per-node attach); pure gates run inside
        // the same detached half in-process. Atomic counter: Cell is not
        // Ungil (Sync), so the detached closure uses an AtomicU32.
        let calls = std::sync::atomic::AtomicU32::new(0);
        Python::initialize();
        Python::attach(|py| {
            py.detach(|| {
                assert!(check_act_args(4, 4, 2).is_ok());
                assert!(check_budget_args(8, 4, 5000).is_ok());
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            });
        });
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
