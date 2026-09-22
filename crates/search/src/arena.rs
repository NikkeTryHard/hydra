//! `arena`: SegmentArena SoA + per-thread segments + isolated `act_batch`.
//!
//! SoA principle (feed-lean): the Python/bridge side passes
//! `(spec_params, root_obs_doc, legal_ids, belief_refs, budget)` ONCE per
//! `act()`; this arena holds nodes/vectors/counters across the whole batch
//! with NO per-sim Python callback (the M1 belief producer closure runs
//! arena-side over the opaque `u64` world handles in `act_batch`).
//!
//! Bridge contract (verbatim shapes — `SearchBridge` hard-uses these):
//! `Budget { max_sims: u64, max_depth: u32, deadline_ms: u64 }`
//! (Clone+Copy, pub fields) and `ActOut { action: u32, completed: bool,
//! sims_run: u64, nodes_visited: u64, decision_digest: String }`
//! (`decision_digest` is `sha256:`-prefixed text over canon-owned bytes,
//! computed arena-side via `feed::canon` + `feed::digest`).
//!
//! Pools (B5): `FeedPool` stays the sole GLOBAL pool; segments here are
//! SCOPED (borrowed cores, sized once, recorded). Budget split:
//! `total / threads` per segment, remainder to seg0 (deterministic).
//! Merge: deterministic `0..T` order + single-lock publish
//! (`commit = block + publish`, opstamp = completed sims).
//!
//! `act_batch` is pure + detach-safe (no clock, no Python, no global
//! state): same inputs give identical outputs regardless of call order.
//! Seat convention: the bridge signature carries no seat, so arena-side
//! descent scalarizes on seat `0` (`ARENA_ROOT_SEAT`); the spec bytes bind
//! the true seat into the decision digest. `completed=false` returns the
//! frontier action (sorted-min legal, or `0` when legal is empty) +
//! counters only — the caller MUST invoke candidate0, never retries.
//!
//! NO width const lives here: widths arrive as plain args (the ONE `BATCH`
//! const stays bridge-side, bridge-owned).

use std::collections::BTreeMap;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::ismcts::{IsNode, IsmctsParams, check_legal, uct_select};
use crate::keys::Categories;
use crate::rng::SegmentStream;

/// Mirror of `search/common.py::_require_deadline_ms` (`(0, 60000]`);
/// the arena fails closed outside it (bridge gates first with its own
/// copy; this mirror keeps the arena total when called directly).
pub const MAX_DEADLINE_MS: u64 = 60_000;

/// Arena-side scalarization seat (bridge signature carries no seat;
/// spec bytes bind the true seat into the digest).
pub const ARENA_ROOT_SEAT: u32 = 0;

/// Seed domain for the act stream (`act_seed`).
const ACT_SEED_DOMAIN: &[u8] = b"hydra_search_act_v1";
/// Seed domain for synthetic sim vectors (`sim_vector`).
const VEC_SEED_DOMAIN: &[u8] = b"hydra_search_vec_v1";

// ---------------------------------------------------------------------------
// Bridge contract types (verbatim — DO NOT rename)
// ---------------------------------------------------------------------------

/// Frozen per-act budget (plain Clone+Copy struct, pub fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Max simulations (`>= 1`).
    pub max_sims: u64,
    /// Max descent depth (`1..=32`).
    pub max_depth: u32,
    /// Deadline millis (`1..=60000`, range-gated only; wall enforcement
    /// is the caller's layered-budget composition).
    pub deadline_ms: u64,
}

/// Arena act outcome (owned, detach-safe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActOut {
    /// Selected action (robust child) or frontier action when incomplete.
    pub action: u32,
    /// Whether the act completed (false -> caller MUST invoke candidate0).
    pub completed: bool,
    /// Simulations actually run.
    pub sims_run: u64,
    /// Node visits (`1` root visit per sim on the single-level arena).
    pub nodes_visited: u64,
    /// `sha256:<64 lowercase hex>` over the canon decision doc.
    pub decision_digest: String,
}

// ---------------------------------------------------------------------------
// SegmentArena SoA (nodes + vectors + counters per worker)
// ---------------------------------------------------------------------------

/// One worker's arena segment: interned info-keys + information-set nodes
/// + budget sub-quota + Philox stream + counters.
#[derive(Debug)]
pub struct SegmentArena {
    /// Digest -> `u32` interner (never hashes; canon owns bytes).
    pub keys: Categories,
    /// Information-set nodes (SoA with `node_keys`).
    pub nodes: Vec<IsNode>,
    /// Interned key id per node (parallel to `nodes`).
    pub node_keys: Vec<u32>,
    /// Completed simulations (opstamp source).
    pub sims_completed: u64,
    /// Completed transitions.
    pub trans_completed: u64,
    /// Completed model calls.
    pub calls_completed: u64,
    /// Simulation sub-quota for this segment.
    pub quota_sims: u64,
    /// Transition sub-quota for this segment.
    pub quota_trans: u64,
    /// This segment's NEW Philox stream (never Gumbels, never splits).
    pub stream: SegmentStream,
}

impl SegmentArena {
    /// Open one segment: NEW stream from `seed`, sub-quotas as given.
    pub fn open(seed: u64, quota_sims: u64, quota_trans: u64) -> SegmentArena {
        SegmentArena {
            keys: Categories::new(),
            nodes: Vec::new(),
            node_keys: Vec::new(),
            sims_completed: 0,
            trans_completed: 0,
            calls_completed: 0,
            quota_sims,
            quota_trans,
            stream: SegmentStream::open(seed),
        }
    }

    /// Position of the node for `digest`, if present.
    pub fn position(&self, digest: &[u8]) -> Option<usize> {
        let id = self.keys.lookup(digest)?;
        self.node_keys.iter().position(|key| *key == id)
    }

    /// Insert a fresh node for `digest` with `legal` (first-encounter
    /// legal wins; callers insert sorted-unique or fail closed before).
    pub fn insert(&mut self, digest: &[u8], legal: &[u32]) -> Result<usize, SearchError> {
        if self.position(digest).is_some() {
            return Err(SearchError::InvalidArg {
                detail: "arena node already exists",
            });
        }
        let id = self.keys.intern(digest)?;
        let mut node = IsNode::new();
        for action in legal {
            node.backup(*action, [0.0, 0.0, 0.0, 0.0]);
            // Zero-backup only registers the arm; reset the counters it
            // bumped so registration is visit-free.
            if let Some(slot) = node.arms.iter_mut().find(|(aid, _)| aid == action) {
                slot.1.visits = 0;
                slot.1.value_sum = [0.0, 0.0, 0.0, 0.0];
            }
        }
        // Registration must leave the node visit-free.
        node.visits = 0;
        self.nodes.push(node);
        self.node_keys.push(id);
        Ok(self.nodes.len() - 1)
    }

    /// Get-or-insert (first-encounter legal wins on races).
    pub fn get_or_insert(&mut self, digest: &[u8], legal: &[u32]) -> Result<usize, SearchError> {
        if let Some(pos) = self.position(digest) {
            return Ok(pos);
        }
        self.insert(digest, legal)
    }

    /// Back up one 4-vector, charging one sim + one transition + one call.
    /// Quota is advisory (callers gate BEFORE the transition for oracle
    /// parity); the counters always record what ran.
    pub fn backup(&mut self, pos: usize, action: u32, vector: [f64; 4]) -> Result<(), SearchError> {
        if pos >= self.nodes.len() {
            return Err(SearchError::InvalidArg {
                detail: "arena node out of range",
            });
        }
        for value in vector {
            if !value.is_finite() {
                return Err(SearchError::NonFinite {
                    context: "arena backup vector",
                });
            }
        }
        self.nodes[pos].backup(action, vector);
        self.sims_completed = add(self.sims_completed, 1)?;
        self.trans_completed = add(self.trans_completed, 1)?;
        self.calls_completed = add(self.calls_completed, 1)?;
        Ok(())
    }

    /// Remaining sim quota (`saturating`: overruns report `0`, never wrap).
    pub fn sims_remaining(&self) -> u64 {
        self.quota_sims.saturating_sub(self.sims_completed)
    }

    /// Merge `other` into `self` (deterministic; caller fixes the `0..T`
    /// order). Nodes join by digest bytes (per-segment id spaces are NOT
    /// comparable); counters add checked.
    pub fn merge_from(&mut self, other: SegmentArena) -> Result<(), SearchError> {
        self.sims_completed = add(self.sims_completed, other.sims_completed)?;
        self.trans_completed = add(self.trans_completed, other.trans_completed)?;
        self.calls_completed = add(self.calls_completed, other.calls_completed)?;
        self.quota_sims = add(self.quota_sims, other.quota_sims)?;
        self.quota_trans = add(self.quota_trans, other.quota_trans)?;
        for (idx, node) in other.nodes.into_iter().enumerate() {
            let kid = match other.node_keys.get(idx) {
                Some(kid) => *kid,
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "segment keys/nodes skew",
                    });
                }
            };
            let digest: Vec<u8> = match other.keys.resolve(kid) {
                Some(bytes) => bytes.to_vec(),
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "segment key missing",
                    });
                }
            };
            let pos = match self.position(&digest) {
                Some(pos) => pos,
                None => self.insert(&digest, &[])?,
            };
            merge_node(&mut self.nodes[pos], node)?;
        }
        Ok(())
    }
}

/// Merge one node's stats into another (visits + per-arm sums, checked).
fn merge_node(into: &mut IsNode, from: IsNode) -> Result<(), SearchError> {
    into.visits = add(into.visits, from.visits)?;
    for (aid, stats) in from.arms {
        match into.arms.iter_mut().find(|(id, _)| *id == aid) {
            Some((_, slot)) => {
                slot.visits = add(slot.visits, stats.visits)?;
                let mut i = 0;
                while i < 4 {
                    slot.value_sum[i] += stats.value_sum[i];
                    i += 1;
                }
            }
            None => into.arms.push((aid, stats)),
        }
    }
    Ok(())
}

/// Checked `u64` add (overflow fails closed, never wraps).
fn add(left: u64, right: u64) -> Result<u64, SearchError> {
    left.checked_add(right).ok_or(SearchError::NonFinite {
        context: "arena counter overflow",
    })
}

/// Budget split: `total / threads` per segment, remainder to seg0
/// (deterministic). `threads == 0` degrades to a single segment (the
/// scoped pool always sizes `>= 1`; this keeps the helper total).
pub fn split_quota(total: u64, threads: u32) -> Vec<u64> {
    if threads == 0 {
        return vec![total];
    }
    let base = total / threads as u64;
    let rem = total % threads as u64;
    let mut out = vec![base; threads as usize];
    out[0] = add(out[0], rem).unwrap_or(total);
    out
}

/// Deterministic `0..T` merge of segments into one arena (fresh stream
/// from `seed`; quotas summed; nodes joined by digest).
pub fn merge_segments(segments: Vec<SegmentArena>, seed: u64) -> Result<SegmentArena, SearchError> {
    let mut merged = SegmentArena::open(seed, 0, 0);
    for segment in segments {
        merged.merge_from(segment)?;
    }
    Ok(merged)
}

// ---------------------------------------------------------------------------
// Shared tables: single-lock publish (commit = block + publish)
// ---------------------------------------------------------------------------

/// Shared arena tables behind ONE lock: workers publish whole segments;
/// readers observe the opstamp (completed sims).
#[derive(Debug)]
pub struct SharedArena {
    /// Merged tables (single lock — no striped partial states).
    inner: Mutex<SegmentArena>,
    /// Opstamp: completed sims at last publish.
    opstamp: Mutex<u64>,
}

impl SharedArena {
    /// Empty shared tables.
    pub fn new(seed: u64) -> SharedArena {
        SharedArena {
            inner: Mutex::new(SegmentArena::open(seed, 0, 0)),
            opstamp: Mutex::new(0),
        }
    }

    /// Publish one block: merge the segment under the single lock, then
    /// ratchet the opstamp to completed sims. Returns the new opstamp.
    pub fn publish(&self, segment: SegmentArena) -> Result<u64, SearchError> {
        let mut inner = self.inner.lock().map_err(|_| SearchError::Lock)?;
        inner.merge_from(segment)?;
        let stamp = inner.sims_completed;
        drop(inner);
        let mut opstamp = self.opstamp.lock().map_err(|_| SearchError::Lock)?;
        *opstamp = stamp;
        Ok(stamp)
    }

    /// Current opstamp (completed sims at last publish).
    pub fn opstamp(&self) -> Result<u64, SearchError> {
        let opstamp = self.opstamp.lock().map_err(|_| SearchError::Lock)?;
        Ok(*opstamp)
    }
}

// ---------------------------------------------------------------------------
// Isolated act_batch (pure, detach-safe)
// ---------------------------------------------------------------------------

/// Validate a budget (range gates only; wall enforcement is caller-side).
fn check_budget(budget: &Budget) -> Result<(), SearchError> {
    if budget.max_sims == 0 {
        return Err(SearchError::InvalidArg {
            detail: "max_sims must be >= 1",
        });
    }
    if budget.max_depth == 0 || budget.max_depth > 32 {
        return Err(SearchError::InvalidArg {
            detail: "max_depth must be 1..32",
        });
    }
    if budget.deadline_ms == 0 || budget.deadline_ms > MAX_DEADLINE_MS {
        return Err(SearchError::InvalidArg {
            detail: "deadline_ms must be 1..60000",
        });
    }
    Ok(())
}

/// Deterministic act seed from the crossing bytes.
fn act_seed(spec: &[u8], root: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(ACT_SEED_DOMAIN);
    hasher.update(spec);
    hasher.update([0u8]);
    hasher.update(root);
    let digest = hasher.finalize();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

/// Synthetic sim vector from `(spec, root, sim, action, world)`: first 8
/// digest bytes big-endian -> `f in [0,1)`, vector `[f, 1-f, 0, 0]`
/// (finite by construction; the M1 closure's arena-side stand-in for the
/// model vector — torch candidate0 stays the prod evaluator).
fn sim_vector(spec: &[u8], root: &[u8], sim: u64, action: u32, world: u64) -> [f64; 4] {
    let mut hasher = Sha256::new();
    hasher.update(VEC_SEED_DOMAIN);
    hasher.update(spec);
    hasher.update([0u8]);
    hasher.update(root);
    hasher.update([0u8]);
    hasher.update(sim.to_le_bytes());
    hasher.update(action.to_le_bytes());
    hasher.update(world.to_le_bytes());
    let digest = hasher.finalize();
    let raw = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    // proof: u64 word / 2^64 is in [0,1); 1ulp on the unit interval is oracle-tolerable.
    #[allow(clippy::cast_precision_loss)]
    let f: f64 = raw as f64 / 18_446_744_073_709_551_616.0;
    [f, 1.0 - f, 0.0, 0.0]
}

/// Decision digest over the canon decision doc (canon-wins: bytes via
/// `feed::canon`, hash via `feed::digest`; infallible — canon failure
/// falls back to hashing the raw shape bytes, never an empty default).
fn decision_digest(
    spec: &[u8],
    root: &[u8],
    action: u32,
    completed: bool,
    sims_run: u64,
    nodes_visited: u64,
) -> String {
    let spec_digest = hydra_feed::digest::sha256_hex(spec);
    let root_digest = hydra_feed::digest::sha256_hex(root);
    let mut doc = BTreeMap::new();
    doc.insert("action".to_string(), serde_json::json!(action));
    doc.insert("completed".to_string(), serde_json::json!(completed));
    doc.insert(
        "nodes_visited".to_string(),
        serde_json::json!(nodes_visited),
    );
    doc.insert("root_digest".to_string(), serde_json::json!(root_digest));
    doc.insert("sims_run".to_string(), serde_json::json!(sims_run));
    doc.insert("spec_digest".to_string(), serde_json::json!(spec_digest));
    match hydra_feed::canon::canonical_bytes(&doc, "search:decision") {
        Ok(bytes) => hydra_feed::digest::sha256_hex(&bytes),
        Err(_) => {
            let fallback = format!("{action}:{completed}:{sims_run}:{nodes_visited}");
            hydra_feed::digest::sha256_hex(fallback.as_bytes())
        }
    }
}

/// Frontier outcome (incomplete paths carry counters only).
fn frontier(spec: &[u8], root: &[u8], legal: &[u32], sims_run: u64, nodes_visited: u64) -> ActOut {
    let mut sorted = legal.to_vec();
    sorted.sort_unstable();
    let action = sorted.first().copied().unwrap_or(0);
    ActOut {
        action,
        completed: false,
        sims_run,
        nodes_visited,
        decision_digest: decision_digest(spec, root, action, false, sims_run, nodes_visited),
    }
}

/// Run ONE isolated arena act: owned inputs in, owned outcome out.
///
/// GIL released around the ENTIRE call (never per-node). The M1 belief
/// closure runs arena-side: `worlds[sim % n]` rotates deterministically
/// (no global RNG, no Python callback). Returns the robust child (max
/// visits, min-id ties) with a canon `decision_digest`; any invalid input
/// returns the frontier (`completed=false`) with counters only.
pub fn act_batch(
    spec: &[u8],
    root: &[u8],
    legal: &[u32],
    worlds: &[u64],
    budget: &Budget,
) -> ActOut {
    let sorted = match check_legal(legal) {
        Ok(sorted) => sorted,
        Err(_) => return frontier(spec, root, legal, 0, 0),
    };
    if check_budget(budget).is_err() {
        return frontier(spec, root, &sorted, 0, 0);
    }
    let params = IsmctsParams {
        uct_c: crate::ismcts::DEFAULT_UCT_C,
        max_depth: budget.max_depth,
        max_sims: u32::try_from(budget.max_sims).unwrap_or(u32::MAX),
        tie: crate::ismcts::TieBreak::LowestActionId,
    };
    let mut arena = SegmentArena::open(act_seed(spec, root), budget.max_sims, budget.max_sims);
    // B3: root table key binds via the single feed digest site (never a local hash).
    let root_hex = hydra_feed::digest::sha256_hex(root);
    let root_pos = match arena.get_or_insert(root_hex.as_bytes(), &sorted) {
        Ok(pos) => pos,
        Err(_) => return frontier(spec, root, &sorted, 0, 0),
    };
    let mut sim = 0;
    while sim < budget.max_sims {
        let action = match uct_select(&arena.nodes[root_pos], &sorted, ARENA_ROOT_SEAT, &params) {
            Ok(action) => action,
            Err(_) => return frontier(spec, root, &sorted, sim, sim),
        };
        // proof: `sim` < `max_sims` (loop bound, u32-range ticket), fits `usize` on 64-bit (saturates on 32-bit, same indexing modulo len).
        #[allow(clippy::cast_possible_truncation)]
        let sim_u: usize = sim as usize;
        let world = if worlds.is_empty() {
            0
        } else {
            worlds[sim_u % worlds.len()]
        };
        let vector = sim_vector(spec, root, sim, action, world);
        if arena.backup(root_pos, action, vector).is_err() {
            return frontier(spec, root, &sorted, sim, sim);
        }
        sim = match sim.checked_add(1) {
            Some(next) => next,
            None => break,
        };
    }
    // Robust child: max visits, min-id ties (deterministic).
    let node = &arena.nodes[root_pos];
    let mut best: Option<u32> = None;
    let mut best_visits = 0;
    for action in &sorted {
        let visits = node.stats(*action).map(|stats| stats.visits).unwrap_or(0);
        match best {
            None => {
                best = Some(*action);
                best_visits = visits;
            }
            Some(current) => {
                if visits > best_visits || (visits == best_visits && *action < current) {
                    best = Some(*action);
                    best_visits = visits;
                }
            }
        }
    }
    let action = best.unwrap_or(sorted[0]);
    let sims_run = arena.sims_completed;
    let nodes_visited = sims_run;
    ActOut {
        action,
        completed: true,
        sims_run,
        nodes_visited,
        decision_digest: decision_digest(spec, root, action, true, sims_run, nodes_visited),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(sims: u64) -> Budget {
        Budget {
            max_sims: sims,
            max_depth: 6,
            deadline_ms: 5000,
        }
    }

    fn worlds4() -> Vec<u64> {
        vec![101, 202, 303, 404]
    }

    #[test]
    fn act_batch_runs_full_budget_with_counters() {
        let out = act_batch(b"spec-a", b"root-a", &[3, 1, 2], &worlds4(), &budget(48));
        assert!(out.completed);
        assert_eq!(out.sims_run, 48);
        assert_eq!(out.nodes_visited, 48);
        assert!([1, 2, 3].contains(&out.action));
        assert!(out.decision_digest.starts_with("sha256:"));
        assert_eq!(out.decision_digest.len(), 7 + 64);
    }

    #[test]
    fn act_batch_is_deterministic_across_orders() {
        let first = act_batch(b"spec-b", b"root-b", &[5, 2, 9], &worlds4(), &budget(16));
        let second = act_batch(b"spec-b", b"root-b", &[9, 5, 2], &worlds4(), &budget(16));
        assert_eq!(first, second);
    }

    #[test]
    fn frontier_carries_counters_only() {
        // Empty legal -> frontier action 0, zero counters, canon digest.
        let out = act_batch(b"s", b"r", &[], &worlds4(), &budget(48));
        assert!(!out.completed);
        assert_eq!((out.action, out.sims_run, out.nodes_visited), (0, 0, 0));
        assert!(out.decision_digest.starts_with("sha256:"));
        // Bad budget -> frontier with sorted-min action, zero counters.
        let bad = Budget {
            max_sims: 0,
            max_depth: 6,
            deadline_ms: 5000,
        };
        let out = act_batch(b"s", b"r", &[7, 4], &worlds4(), &bad);
        assert!(!out.completed);
        assert_eq!((out.action, out.sims_run), (4, 0));
        // Duplicate legal -> frontier (fail closed, never a guess).
        let out = act_batch(b"s", b"r", &[4, 4], &worlds4(), &budget(8));
        assert!(!out.completed);
    }

    #[test]
    fn split_budget_remainder_to_seg0() {
        assert_eq!(split_quota(10, 3), vec![4, 3, 3]);
        assert_eq!(split_quota(16, 16), vec![1; 16]);
        assert_eq!(split_quota(5, 1), vec![5]);
        assert_eq!(split_quota(48, 4), vec![12, 12, 12, 12]);
        assert_eq!(split_quota(49, 4), vec![13, 12, 12, 12]);
        // Quotas always re-sum to the total.
        for (total, threads) in [(7, 3), (100, 7), (1, 8)] {
            let parts = split_quota(total, threads);
            assert_eq!(parts.len(), threads as usize);
            assert_eq!(parts.iter().sum::<u64>(), total);
        }
    }

    #[test]
    fn merge_is_deterministic_zero_to_t() {
        // Two segments backing up the same arms in opposite order merge to
        // identical totals (order-independent sums, deterministic 0..T).
        let mut left = SegmentArena::open(1, 4, 4);
        let lpos = left.get_or_insert(b"sha256:root", &[1, 2]).expect("insert");
        left.backup(lpos, 1, [1.0, 0.0, 0.0, 0.0]).expect("backup");
        left.backup(lpos, 2, [0.0, 1.0, 0.0, 0.0]).expect("backup");
        let mut right = SegmentArena::open(2, 4, 4);
        let rpos = right
            .get_or_insert(b"sha256:root", &[1, 2])
            .expect("insert");
        right.backup(rpos, 2, [0.0, 2.0, 0.0, 0.0]).expect("backup");
        right.backup(rpos, 1, [2.0, 0.0, 0.0, 0.0]).expect("backup");
        let ab = merge_segments(vec![left, right], 9).expect("merge");
        // Rebuild in the opposite segment order: same totals.
        let mut left2 = SegmentArena::open(1, 4, 4);
        let lpos2 = left2
            .get_or_insert(b"sha256:root", &[1, 2])
            .expect("insert");
        left2
            .backup(lpos2, 1, [1.0, 0.0, 0.0, 0.0])
            .expect("backup");
        left2
            .backup(lpos2, 2, [0.0, 1.0, 0.0, 0.0])
            .expect("backup");
        let mut right2 = SegmentArena::open(2, 4, 4);
        let rpos2 = right2
            .get_or_insert(b"sha256:root", &[1, 2])
            .expect("insert");
        right2
            .backup(rpos2, 2, [0.0, 2.0, 0.0, 0.0])
            .expect("backup");
        right2
            .backup(rpos2, 1, [2.0, 0.0, 0.0, 0.0])
            .expect("backup");
        let ba = merge_segments(vec![right2, left2], 9).expect("merge");
        assert_eq!(ab.sims_completed, 4);
        assert_eq!(ba.sims_completed, 4);
        let apos = ab.position(b"sha256:root").expect("root");
        let bpos = ba.position(b"sha256:root").expect("root");
        assert_eq!(ab.nodes[apos].mean_vector(1), ba.nodes[bpos].mean_vector(1));
        assert_eq!(ab.nodes[apos].mean_vector(2), ba.nodes[bpos].mean_vector(2));
    }

    #[test]
    fn single_lock_publish_ratchets_opstamp() {
        let shared = SharedArena::new(42);
        assert_eq!(shared.opstamp().expect("stamp"), 0);
        let mut first = SegmentArena::open(1, 8, 8);
        let pos = first.get_or_insert(b"sha256:k", &[1]).expect("insert");
        first.backup(pos, 1, [1.0, 0.0, 0.0, 0.0]).expect("backup");
        first.backup(pos, 1, [1.0, 0.0, 0.0, 0.0]).expect("backup");
        // commit = block + publish: opstamp becomes completed sims.
        assert_eq!(shared.publish(first).expect("publish"), 2);
        assert_eq!(shared.opstamp().expect("stamp"), 2);
        let mut second = SegmentArena::open(2, 8, 8);
        let pos = second.get_or_insert(b"sha256:k", &[1]).expect("insert");
        second.backup(pos, 1, [0.0, 1.0, 0.0, 0.0]).expect("backup");
        assert_eq!(shared.publish(second).expect("publish"), 3);
    }

    #[test]
    fn golden_decision_digest_stable() {
        // Golden: same inputs -> byte-identical digest (persistence-family
        // anchor: per-arm selected+vectors+counters hash inputs pin here).
        let digest = |spec: &[u8]| {
            act_batch(spec, b"golden-root", &[1, 2, 4, 8], &worlds4(), &budget(32)).decision_digest
        };
        let first = digest(b"golden-spec");
        assert_eq!(first, digest(b"golden-spec"));
        assert_ne!(first, digest(b"golden-spec-2"));
    }
}
