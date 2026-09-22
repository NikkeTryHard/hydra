//! local_graph: frozen Candidate 5 public-history graph + strategy leaves.
//!
//! DAG: pyo3 + `hydra-feed` (digest owner) + `hydra-search` (`local` owner)
//! ONLY. FORBIDS: selection math, RNG streams, belief sampling, torch/CUDA,
//! wall-clock budgets, file IO, and `ContractError` shaping (the bridge raises
//! `ValueError`; thin Python translators map to `ContractError` /
//! `AbstractMappingError` / `CycleDetectedError` with byte-identical text).
//!
//! TABLE ported here — pure graph/strategy leaves over plain values
//! (`python/hydra2/search/local_abstraction.py`, `local_strategy.py`):
//! - `LOCAL_GRAPH_MAX_HORIZON` ← `hydra_search::local::LOCAL_MAX_HORIZON`
//!   (same value, crate-owned; exposed like `search_shared::GUMBEL_HALVING_ROUNDS`
//!   over `hydra_search::gumbel::DEFAULT_HALVING_ROUNDS`, `search_shared.rs:69`).
//! - `LOCAL_GRAPH_MAX_NODES` ← the `> 64` level-cap in `build_public_subgame`
//!   (`local_abstraction.py:516-518`); first freeze, no Rust owner exists
//!   (`rg 'MAX_NODES' crates/` is empty at port time).
//! - `local_graph_detect_cycle` ← `detect_cycle` DFS (`:535-560`):
//!   order-verbatim (node-order starts, edge-order adjacency, recursive gray
//!   check) so the first `cycle detected: {u} -> {v} closes loop` text is
//!   byte-identical.
//! - `local_graph_expand_subgame` ← the `build_public_subgame` level expansion
//!   (`:493-518`): `child = sha256("{parent}:{d}:{aid}")` per level in
//!   `(parent, aid)` order with the `> 64` break. Hashes fold through the feed
//!   digest owner (`hydra_feed::digest::sha256_hex`, infallible `sha256:`-prefixed
//!   text — same entry `canon_rng::batch_sha256` folds, see `canon_rng.rs:228`).
//! - `local_graph_uniform_strategy` ← `make_uniform_strategy`
//!   (`local_strategy.py:111-113`) via the crate owner
//!   `hydra_search::local::uniform_strategy` (`search/src/local.rs:136`), never
//!   reimplemented (one `1.0 / n` division is bit-identical either side).
//! - `local_graph_tiny_term` ← the per-step tiny-game scalar
//!   (`local_strategy.py:160-165`): `(int(sha256(f"{wid}:{aid}:{d}")
//!   .hexdigest()[:4], 16) % 100) / 1000.0 - 0.05`. Feed hex `[:4]` equals the
//!   first-two-digest-bytes big-endian word (the equivalence
//!   `ismcts_driver.rs:318` states). The per-path `sum(...)` stays Python-side
//!   (`builtin sum` — wave-8 lesson, naive Rust folds diverge 1 ulp), as do the
//!   re-centering and the cross-world average.
//!
//! NOT duplicated here (single-owner rule, grep evidence at port time):
//! - the whole-path sum `hydra_search::ismcts_driver::exhaustive_path_offset`
//!   (`ismcts_driver.rs:319`, naive `+=` fold, driver-owned) and the private
//!   planner primitive `local_driver::utility_offset` (`local_driver.rs:160`)
//!   stay planner-internal; neither covers an explicit-depth single step, so the
//!   oracle scalar is exposed here instead of reusing them.
//! - `_VALID_UPDATE_RULES` / `_VALID_AVERAGING` live in `local_spec.rs:62-64`
//!   and `FORBIDDEN_IN_STRATEGY_KEY` in `search_shared.rs:22` — referenced by
//!   the Python side, never re-frozen.
//!
//! LOGIC staying Python: the `LocalResolvingAbstraction` / `PublicSubgame` /
//! `StrategyTable` dataclasses (frozen/slots/pickling, mixed
//! `ContractError`/`AbstractMappingError` shaping), `validate_abstraction_mapping`
//! (dict/tuple/object normalization + name inference), info keys + vectors
//! (feed-canonical bytes + the already-bridged `ismcts_local_*` fns),
//! `StrategyTable` math (mutable planner state + `builtin sum`, both OUT),
//! `leaf_vector_replay` dispatch, `is_equilibrium_claimed` (bridge-independent
//! `False` — bridging it adds a failure mode for zero compute win), and
//! `preserves_vector_returns` (`isinstance` arms incl. the bool-subclass pass
//! and the huge-int `OverflowError` have no exact pyo3 staging shape;
//! FFI-negative 4-elem check).
//!
//! Shape per fn: attached staging (translators pass plain values positionally;
//! extraction failures are `TypeError`, fail-closed) → ONE `py.detach(|| …)`
//! over owned plain data with zero Python API inside (per `contracts.rs:434-437`)
//! → attached wrap as `PyValueError` (per `contracts.rs:117-123`).
//!
//! Single-cdylib tree: registers consts + pyfns on the EXISTING `search`
//! submodule (mirrors `local_spec::register`, `local_spec.rs:273-317`); no new
//! entry point. MAIN wiring: `pub mod local_graph;` in `lib.rs` plus
//! `crate::local_graph::register(&sub)?;` in `search.rs` next to
//! `crate::local_spec::register(&sub)?;` (`search.rs:1737`).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Level-expansion output: `(nodes, edges)` with edges as
/// `(from, to, abstract_id)`.
type SubgameGraph = (Vec<String>, Vec<(String, String, u32)>);

/// Level-cap mirrored from `build_public_subgame` (`local_abstraction.py:516-518`):
/// expansion stops after the first level pushing the node count past 64.
const MAX_NODES: usize = 64;

/// Unvisited mark mirroring the oracle DFS (`local_abstraction.py:546`).
const WHITE: u8 = 0;
/// On-stack mark mirroring the oracle DFS (`local_abstraction.py:546`).
const GRAY: u8 = 1;
/// Finished mark mirroring the oracle DFS (`local_abstraction.py:546`).
const BLACK: u8 = 2;

/// Pure DFS cycle check over plain values (`detect_cycle`, `:535-560`):
/// adjacency in edge order, starts in node order, recursive gray check.
/// Operates on owned plain data only — no Python API inside, detach-safe.
fn cycle_check(nodes: &[String], edges: &[(String, String, u32)]) -> Result<(), String> {
    let mut index: HashMap<&str, usize> = HashMap::with_capacity(nodes.len());
    for (pos, name) in nodes.iter().enumerate() {
        index.entry(name.as_str()).or_insert(pos);
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (from, to, _) in edges {
        let (Some(&u), Some(&v)) = (index.get(from.as_str()), index.get(to.as_str())) else {
            return Err(format!("edge references unknown node ('{from}' -> '{to}')"));
        };
        adj[u].push(v);
    }
    let mut colors: Vec<u8> = vec![WHITE; nodes.len()];
    for u in 0..nodes.len() {
        if colors[u] == WHITE {
            dfs_visit(&adj, &mut colors, nodes, u)?;
        }
    }
    Ok(())
}

/// One recursive DFS visit (mirrors the oracle closure `:549-556`).
fn dfs_visit(
    adj: &[Vec<usize>],
    colors: &mut [u8],
    nodes: &[String],
    u: usize,
) -> Result<(), String> {
    colors[u] = GRAY;
    for &v in &adj[u] {
        if colors[v] == GRAY {
            return Err(format!(
                "cycle detected: {} -> {} closes loop",
                nodes[u], nodes[v]
            ));
        }
        if colors[v] == WHITE {
            dfs_visit(adj, colors, nodes, v)?;
        }
    }
    colors[u] = BLACK;
    Ok(())
}

/// One hex nibble to its value; non-hex input reads as 0 (feed hex is always
/// valid, so the arm never fires on real digests — it exists for totality).
fn hex_nibble(value: u8) -> u16 {
    match value {
        b'0'..=b'9' => u16::from(value - b'0'),
        b'a'..=b'f' => u16::from(value - b'a') + 10,
        b'A'..=b'F' => u16::from(value - b'A') + 10,
        _ => 0,
    }
}

/// Pure per-step tiny-game scalar (`local_strategy.py:160-165`): the
/// `(int(head, 16) % 100) / 1000.0 - 0.05` term over `f"{wid}:{aid}:{depth}"`,
/// hashed through the feed digest owner. Infallible and detach-safe.
fn tiny_term(world_id: &str, aid: u32, depth: u32) -> f64 {
    let payload = format!("{world_id}:{aid}:{depth}");
    let digest = hydra_feed::digest::sha256_hex(payload.as_bytes());
    let head: u16 = digest
        .as_bytes()
        .iter()
        .skip(7)
        .take(4)
        .fold(0u16, |acc: u16, byte: &u8| acc * 16 + hex_nibble(*byte));
    f64::from(head % 100) / 1000.0 - 0.05
}

/// Pure level expansion (`build_public_subgame`, `local_abstraction.py:493-518`):
/// `child = sha256("{parent}:{d}:{aid}")` per level in `(parent, aid)` order
/// with the `> 64` break. Operates on owned plain data only — no Python API
/// inside, detach-safe.
fn expand_subgame(root: &str, abstract_ids: &[u32], horizon: u32) -> Result<SubgameGraph, String> {
    if horizon == 0 || horizon > hydra_search::local::LOCAL_MAX_HORIZON {
        return Err(format!("horizon must be int in 1..16, got {horizon}"));
    }
    let mut nodes: Vec<String> = vec![root.to_owned()];
    let mut edges: Vec<(String, String, u32)> = Vec::new();
    let mut current: Vec<String> = vec![root.to_owned()];
    for depth in 0..horizon {
        let mut next: Vec<String> = Vec::new();
        for parent in &current {
            for &aid in abstract_ids {
                let child =
                    hydra_feed::digest::sha256_hex(format!("{parent}:{depth}:{aid}").as_bytes());
                if !nodes.contains(&child) {
                    nodes.push(child.clone());
                }
                next.push(child.clone());
                edges.push((parent.clone(), child, aid));
            }
        }
        current = next;
        if nodes.len() > MAX_NODES {
            break;
        }
    }
    Ok((nodes, edges))
}

/// DFS cycle detection over plain node/edge lanes (`detect_cycle`).
///
/// `nodes` ride in oracle order, `edges` as `(from, to, abstract_id)`; the
/// translator unpacks them from the live `PublicSubgame` (which never crosses).
/// Compute runs detached; `ValueError` on the first back edge, never a default.
#[pyfunction]
#[pyo3(signature = (nodes, edges))]
fn local_graph_detect_cycle(
    py: Python<'_>,
    nodes: Vec<String>,
    edges: Vec<(String, String, u32)>,
) -> PyResult<()> {
    py.detach(|| cycle_check(&nodes, &edges))
        .map_err(PyValueError::new_err)
}

/// Level expansion over plain values (`build_public_subgame` kernel).
///
/// `root` is the caller-derived public-history digest, `abstract_ids` ride in
/// the abstraction's stored order, `horizon` is pre-validated `1..16`
/// caller-side (the bridge double-gates fail-closed). Returns
/// `(nodes, edges)`; the translator builds the `PublicSubgame` around them.
/// Compute runs detached.
#[pyfunction]
#[pyo3(signature = (root, abstract_ids, horizon))]
fn local_graph_expand_subgame(
    py: Python<'_>,
    root: String,
    abstract_ids: Vec<u32>,
    horizon: u32,
) -> PyResult<SubgameGraph> {
    py.detach(|| expand_subgame(&root, &abstract_ids, horizon))
        .map_err(PyValueError::new_err)
}

/// Uniform strategy over `n` abstract ids (`make_uniform_strategy` via
/// `hydra_search::local::uniform_strategy`): `1.0 / n` replicated.
/// Compute runs detached; `ValueError` on `n == 0`, never a default.
#[pyfunction]
#[pyo3(signature = (n,))]
fn local_graph_uniform_strategy(py: Python<'_>, n: usize) -> PyResult<Vec<f64>> {
    py.detach(|| hydra_search::local::uniform_strategy(n))
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Per-step tiny-game scalar (`local_strategy.py:160-165`): one
/// `(int(head, 16) % 100) / 1000.0 - 0.05` term. The translator sums terms
/// with `builtin sum` (wave-8: float sums stay Python-side). Infallible;
/// compute runs detached.
#[pyfunction]
#[pyo3(signature = (world_id, aid, depth))]
fn local_graph_tiny_term(py: Python<'_>, world_id: String, aid: u32, depth: u32) -> PyResult<f64> {
    Ok(py.detach(|| tiny_term(&world_id, aid, depth)))
}

/// Register the graph vocabularies + leaves on the EXISTING `search`
/// submodule (mirrors `local_spec::register`, `local_spec.rs:273-317`;
/// MAIN calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add(
        "LOCAL_GRAPH_MAX_HORIZON",
        hydra_search::local::LOCAL_MAX_HORIZON,
    )?;
    sub.add("LOCAL_GRAPH_MAX_NODES", MAX_NODES)?;
    sub.add_function(wrap_pyfunction!(local_graph_detect_cycle, sub)?)?;
    sub.add_function(wrap_pyfunction!(local_graph_expand_subgame, sub)?)?;
    sub.add_function(wrap_pyfunction!(local_graph_uniform_strategy, sub)?)?;
    sub.add_function(wrap_pyfunction!(local_graph_tiny_term, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod local_graph_tests {
    use super::*;

    #[test]
    fn frozen_values_match_python_leaves() {
        assert_eq!(hydra_search::local::LOCAL_MAX_HORIZON, 16);
        assert_eq!(MAX_NODES, 64);
    }

    #[test]
    fn cycle_check_accepts_dag_and_rejects_back_edge() {
        let nodes = vec!["n0".to_owned(), "n1".to_owned(), "n2".to_owned()];
        let dag = vec![
            ("n0".to_owned(), "n1".to_owned(), 0u32),
            ("n1".to_owned(), "n2".to_owned(), 1u32),
        ];
        assert_eq!(cycle_check(&nodes, &dag), Ok(()));
        let mut cyc = dag.clone();
        cyc.push(("n2".to_owned(), "n0".to_owned(), 0u32));
        assert_eq!(
            cycle_check(&nodes, &cyc),
            Err("cycle detected: n2 -> n0 closes loop".to_owned())
        );
    }

    #[test]
    fn cycle_check_rejects_unknown_endpoint() {
        let nodes = vec!["n0".to_owned()];
        let edges = vec![("n0".to_owned(), "ghost".to_owned(), 0u32)];
        assert!(cycle_check(&nodes, &edges).is_err());
    }

    #[test]
    fn uniform_strategy_matches_scalar_division() {
        let row = hydra_search::local::uniform_strategy(4).expect("non-empty uniform");
        assert_eq!(row, vec![0.25, 0.25, 0.25, 0.25]);
        assert_eq!(
            hydra_search::local::uniform_strategy(1).expect("singleton"),
            vec![1.0]
        );
        assert!(hydra_search::local::uniform_strategy(0).is_err());
    }

    #[test]
    fn tiny_term_matches_oracle_scalar() {
        // Golden from the HEAD oracle:
        // `(int(hashlib.sha256(b"w0:1:2").hexdigest()[:4], 16) % 100) / 1000.0
        // - 0.05` == `-0.023000000000000003` (`pixi run python -c`, stdlib only).
        assert_eq!(tiny_term("w0", 1, 2), -0.023000000000000003);
        // Bounded face: every term lands in `[-0.05, 0.05)` and is stable.
        for (wid, aid, depth) in [("w0", 0u32, 0u32), ("w1", 3u32, 5u32), ("", 99u32, 15u32)] {
            let term = tiny_term(wid, aid, depth);
            assert!(term.is_finite() && (-0.05..0.05).contains(&term));
            assert_eq!(term, tiny_term(wid, aid, depth));
        }
    }

    #[test]
    fn expand_matches_level_order_and_cap() {
        let (nodes, edges) = expand_subgame("root", &[0, 1], 1).expect("valid expansion");
        // Goldens from the HEAD oracle: `"sha256:" + hashlib.sha256(b"root:0:0" /
        // b"root:0:1").hexdigest()` (`pixi run python -c`, stdlib only).
        assert_eq!(
            nodes,
            vec![
                "root".to_owned(),
                "sha256:4eccbc5d07005ab373c9a90e1a0dc0a9438da622b9716bf748396c106d40f6f1"
                    .to_owned(),
                "sha256:16c9c36d5afae255f94216bb218c2fde9fb03e6f21841c2ed9b84e52cfce9125"
                    .to_owned(),
            ]
        );
        assert_eq!(
            edges,
            vec![
                ("root".to_owned(), nodes[1].clone(), 0u32),
                ("root".to_owned(), nodes[2].clone(), 1u32),
            ]
        );
        assert!(expand_subgame("root", &[0, 1], 0).is_err());
        assert!(expand_subgame("root", &[0, 1], 17).is_err());
        // Branching 4, horizon 16: levels add 4 + 16 + 64, then the cap breaks.
        let (big_nodes, big_edges) =
            expand_subgame("root", &[0, 1, 2, 3], 16).expect("capped expansion");
        assert_eq!(big_nodes.len(), 85);
        assert_eq!(big_edges.len(), 84);
    }
}
