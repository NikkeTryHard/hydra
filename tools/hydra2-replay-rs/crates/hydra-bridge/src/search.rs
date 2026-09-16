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
//! Single-cdylib tree: this module registers as the `search` submodule
//! (`hydra2_replay_rs.search` today, `hydra_bridge._native.search` once
//! the Phase-6 maturin `module-name` cutover lands) via `register`,
//! mirroring `canon_rng::register` / `columnar::register`. The legacy
//! `hydra2_replay_rs` entry point is untouched.
//!
//! ASSUMPTION-MANIFEST (pending Phase4SearchArena — integration owner:
//! Arena owns the `hydra-search` crate + workspace `members` line + this
//! bridge's `hydra-search = { path = "../hydra-search" }` edge; this file
//! MUST NOT compile until applied; DO NOT guess-fix locally): workspace
//! `members += "crates/hydra-search"`; bridge deps gain the path edge (no
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

use std::sync::Mutex;
use hydra_search::arena::{act_batch as arena_act_batch, Budget};
use hydra_search::belief as belief_mod;
use hydra_search::gumbel as gumbel_mod;
use hydra_search::ismcts::{ActionStats, IsNode, IsmctsParams, TieBreak};
use hydra_search::SearchError;
#[cfg(test)]
use hydra_search::arena::ActOut;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyModule;

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
    let hex = text.strip_prefix(PREFIX).ok_or_else(|| {
        format!("search decision digest must start with 'sha256:', got {text:?}")
    })?;
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
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
        return Err(SearchError::InvalidArg { detail: "node actions/visits length mismatch" });
    }
    let want = actions
        .len()
        .checked_mul(4)
        .ok_or(SearchError::InvalidArg { detail: "node actions too many" })?;
    if sums_flat.len() != want {
        return Err(SearchError::InvalidArg { detail: "node sums must hold 4 entries per action" });
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
            None => return Err(SearchError::InvalidArg { detail: "node action out of range" }),
        };
        let n = match visits.get(i) {
            Some(n) => *n,
            None => return Err(SearchError::InvalidArg { detail: "node visits out of range" }),
        };
        let mut sum = [0.0f64; 4];
        let mut j = 0usize;
        while j < 4 {
            let value = match sums_flat.get(i * 4 + j) {
                Some(value) => *value,
                None => return Err(SearchError::InvalidArg { detail: "node sums out of range" }),
            };
            if n > 0 && !value.is_finite() {
                return Err(SearchError::NonFinite { context: "search node value sum" });
            }
            sum[j] = value;
            j += 1;
        }
        arms.push((action, ActionStats { visits: n, value_sum: sum }));
        i += 1;
    }
    Ok(IsNode { visits: total, arms })
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
        return Err(PyValueError::new_err("search gumbel case_id must be non-empty"));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err("search gumbel candidate_id must be non-empty"));
    }
    if root_seat >= 4 {
        return Err(PyValueError::new_err("seat must be 0..3"));
    }
    let out =
        py.detach(|| gumbel_mod::deterministic_gumbel(&case_id, root_seat, &candidate_id, action_id));
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
        return Err(PyValueError::new_err("search gumbel case_id must be non-empty"));
    }
    if candidate_id.is_empty() {
        return Err(PyValueError::new_err("search gumbel candidate_id must be non-empty"));
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
        if let Some(action) = actions.get(i) {
            if legal.contains(action) {
                if let Some(n) = visits.get(i) {
                    total = total.saturating_add(*n);
                }
            }
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

/// Judge observability: `(acts, completed)` (never identity/selection).
#[pyfunction]
fn act_stats(py: Python<'_>) -> PyResult<(u64, u64)> {
    let guard = ACT_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("search act stats mutex poisoned"))?;
    Ok((guard.acts, guard.completed))
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
    sub.add_function(wrap_pyfunction!(uct_select, &sub)?)?;
    sub.add_function(wrap_pyfunction!(puct_select, &sub)?)?;
    sub.add_function(wrap_pyfunction!(natural_indices, &sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_draws, &sub)?)?;
    sub.add_function(wrap_pyfunction!(packet_successors, &sub)?)?;
    sub.add_class::<PyPacketSuccessor>()?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use std::cell::Cell;

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
