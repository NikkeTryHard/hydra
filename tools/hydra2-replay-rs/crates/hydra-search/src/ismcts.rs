//! `ismcts`: UCT select/expand (48 sims x depth 6, `1e-12` eps).
//!
//! Parity: `ismcts_core.py:520-555` (`_uct_select`). Unvisited actions win in
//! sorted-legal order FIRST; visited arms compare `q+u` with
//! `u = uct_c * sqrt(ln(total+1) / visits)`, `1e-12` eps gate, and the frozen
//! tie-break arm (`lowest_action_id` = min id; `stable_hash`/`lexicographic`
//! = min `sha256(f"{a}")` hex — NEVER `hash()`). `scalarize_vector` is the
//! root-seat projection of the 4-vector mean. All entries finite-checked.

use sha2::{Digest, Sha256};

use crate::SearchError;

/// UCT epsilon gate (`ismcts_core.py:539,542`): improvements beyond `1e-12`
/// win outright, differences within are ties for the tie-break arm.
pub const UCT_EPS: f64 = 1e-12;

/// Default frozen hyper-parameters (`NaturalISMCTSConfig`: `uct_c ~ sqrt2`,
/// depth 6, 48 sims).
#[allow(clippy::approx_constant)]
pub const DEFAULT_UCT_C: f64 = 1.414_213_562_37;
/// Frozen default: 48 simulations per search.
pub const DEFAULT_MAX_SIMS: u32 = 48;
/// Frozen default: depth 6 descents.
pub const DEFAULT_MAX_DEPTH: u32 = 6;

/// Frozen tie-break arms (`ismcts_core.py:544-552`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreak {
    /// Min action id wins ties.
    LowestActionId,
    /// Min `sha256(f"{a}")` hexdigest wins ties (deterministic, hash-seeded).
    StableHash,
    /// Same as `StableHash` (lexicographic over the hex digest).
    Lexicographic,
}

impl TieBreak {
    /// Parse the frozen vocabulary (`tie_break` names only).
    pub fn parse(name: &str) -> Result<TieBreak, SearchError> {
        match name {
            "lowest_action_id" => Ok(TieBreak::LowestActionId),
            "stable_hash" => Ok(TieBreak::StableHash),
            "lexicographic" => Ok(TieBreak::Lexicographic),
            _ => Err(SearchError::InvalidArg { detail: "unknown tie_break" }),
        }
    }
}

/// Frozen ISMCTS hyper-parameters.
#[derive(Debug, Clone, Copy)]
pub struct IsmctsParams {
    /// UCT exploration constant (finite, `> 0`).
    pub uct_c: f64,
    /// Max descent depth (`1..=32`).
    pub max_depth: u32,
    /// Max simulations (budget; caller enforces).
    pub max_sims: u32,
    /// Tie-break arm.
    pub tie: TieBreak,
}

impl IsmctsParams {
    /// Frozen defaults (sqrt2, 48 sims, depth 6, lowest-action-id).
    pub fn defaults() -> IsmctsParams {
        IsmctsParams {
            uct_c: DEFAULT_UCT_C,
            max_depth: DEFAULT_MAX_DEPTH,
            max_sims: DEFAULT_MAX_SIMS,
            tie: TieBreak::LowestActionId,
        }
    }

    /// Validate (`ContractError` edges: finite positive `uct_c`, depth
    /// `1..=32`, sims `>= 1`).
    pub fn validate(&self) -> Result<(), SearchError> {
        if !self.uct_c.is_finite() || self.uct_c <= 0.0 {
            return Err(SearchError::InvalidArg { detail: "uct_c must be finite >0" });
        }
        if self.max_depth == 0 || self.max_depth > 32 {
            return Err(SearchError::InvalidArg { detail: "max_depth must be 1..32" });
        }
        if self.max_sims == 0 {
            return Err(SearchError::InvalidArg { detail: "max_sims must be >= 1" });
        }
        Ok(())
    }
}

/// Per-action running stats: visit count + 4-seat value sums.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionStats {
    /// Visit count.
    pub visits: u64,
    /// Per-seat value sums (same 4-vector the backup writes).
    pub value_sum: [f64; 4],
}

/// One information-set node: visit count + per-action stats.
#[derive(Debug, Clone, Default)]
pub struct IsNode {
    /// Total visits through this node.
    pub visits: u64,
    /// `(action_id, stats)` in insertion order (iteration is always
    /// sorted-legal at selection, never map order).
    pub arms: Vec<(u32, ActionStats)>,
}

impl IsNode {
    /// Empty node.
    pub fn new() -> IsNode {
        IsNode::default()
    }

    /// Look up stats for one action.
    pub fn stats(&self, action: u32) -> Option<&ActionStats> {
        self.arms.iter().find(|(aid, _)| *aid == action).map(|(_, stats)| stats)
    }

    /// Back up one 4-vector to one action (`visits += 1` both levels).
    pub fn backup(&mut self, action: u32, vector: [f64; 4]) {
        self.visits += 1;
        match self.arms.iter_mut().find(|(aid, _)| *aid == action) {
            Some((_, stats)) => {
                stats.visits += 1;
                let mut i = 0;
                while i < 4 {
                    stats.value_sum[i] += vector[i];
                    i += 1;
                }
            }
            None => {
                self.arms.push((action, ActionStats { visits: 1, value_sum: vector }));
            }
        }
    }

    /// Mean 4-vector for one action (`None` when unvisited).
    pub fn mean_vector(&self, action: u32) -> Option<[f64; 4]> {
        let stats = self.stats(action)?;
        if stats.visits == 0 {
            return None;
        }
        let n = stats.visits as f64;
        Some([
            stats.value_sum[0] / n,
            stats.value_sum[1] / n,
            stats.value_sum[2] / n,
            stats.value_sum[3] / n,
        ])
    }

    /// Root scalarization: project the mean vector onto `root_seat`
    /// (`scalarize_vector`: `vector[root_seat]`), finite-checked.
    pub fn scalar_mean(&self, action: u32, root_seat: u32) -> Result<Option<f64>, SearchError> {
        if root_seat >= 4 {
            return Err(SearchError::InvalidSeat { seat: root_seat });
        }
        let mean = match self.mean_vector(action) {
            Some(mean) => mean,
            None => return Ok(None),
        };
        let value = mean[root_seat as usize];
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts scalar_mean" });
        }
        Ok(Some(value))
    }
}

/// Compare two visited arms for the tie-break arm (both values finite).
fn tie_wins(candidate: u32, best: u32, tie: TieBreak) -> bool {
    match tie {
        TieBreak::LowestActionId => candidate < best,
        TieBreak::StableHash | TieBreak::Lexicographic => {
            sha_hex(candidate) < sha_hex(best)
        }
    }
}

/// `sha256(f"{a}")` lowercase hex (deterministic tie order, never `hash()`).
fn sha_hex(action: u32) -> String {
    let sum = Sha256::digest(action.to_string().as_bytes());
    let mut out = String::with_capacity(64);
    for byte in sum {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Validate a legal set: non-empty, unique (sorted-window scan).
pub fn check_legal(legal: &[u32]) -> Result<Vec<u32>, SearchError> {
    if legal.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mut sorted = legal.to_vec();
    sorted.sort_unstable();
    let mut i = 1;
    while i < sorted.len() {
        if sorted[i] == sorted[i - 1] {
            return Err(SearchError::DuplicateAction { action: sorted[i] });
        }
        i += 1;
    }
    Ok(sorted)
}

/// UCT select (`ismcts_core.py:520-555` parity): unvisited in sorted order
/// first; else `q+u` with `1e-12` eps gate + tie-break arm.
pub fn uct_select(
    node: &IsNode,
    legal: &[u32],
    root_seat: u32,
    params: &IsmctsParams,
) -> Result<u32, SearchError> {
    let sorted = check_legal(legal)?;
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    params.validate()?;
    // Unvisited in deterministic sorted order first.
    for action in &sorted {
        match node.stats(*action) {
            Some(stats) if stats.visits > 0 => {}
            _ => return Ok(*action),
        }
    }
    let total = node.visits;
    let mut best: Option<u32> = None;
    let mut best_val = f64::NEG_INFINITY;
    for action in &sorted {
        let mean = match node.mean_vector(*action) {
            Some(mean) => mean,
            None => continue,
        };
        let q = mean[root_seat as usize];
        if !q.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts uct q" });
        }
        let stats = match node.stats(*action) {
            Some(stats) => stats,
            None => continue,
        };
        let u = params.uct_c * ((total as f64 + 1.0).ln() / stats.visits as f64).sqrt();
        if !u.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts uct u" });
        }
        let val = q + u;
        if !val.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts uct val" });
        }
        match best {
            None => {
                best = Some(*action);
                best_val = val;
            }
            Some(current) => {
                if val > best_val + UCT_EPS {
                    best = Some(*action);
                    best_val = val;
                } else if (val - best_val).abs() <= UCT_EPS
                    && tie_wins(*action, current, params.tie)
                {
                    best = Some(*action);
                    best_val = val;
                }
            }
        }
    }
    match best {
        Some(action) => Ok(action),
        None => Ok(sorted[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visited_node() -> IsNode {
        let mut node = IsNode::new();
        // Three visited arms with distinct means for seat 0.
        node.backup(10, [1.0, 0.0, 0.0, 0.0]);
        node.backup(10, [1.0, 0.0, 0.0, 0.0]);
        node.backup(20, [0.0, 0.0, 0.0, 0.0]);
        node.backup(20, [0.0, 0.0, 0.0, 0.0]);
        node.backup(30, [0.5, 0.0, 0.0, 0.0]);
        node.backup(30, [0.5, 0.0, 0.0, 0.0]);
        node
    }

    #[test]
    fn unvisited_wins_in_sorted_order() {
        let mut node = IsNode::new();
        node.backup(20, [9.0, 0.0, 0.0, 0.0]);
        // 10 is unvisited and sorts first -> wins despite 20's value.
        let params = IsmctsParams::defaults();
        assert_eq!(uct_select(&node, &[20, 10], 0, &params).expect("select"), 10);
    }

    #[test]
    fn tie_break_order_lowest_action_id() {
        // Identical stats on both arms: q+u equal within eps -> min id wins.
        let mut node = IsNode::new();
        node.backup(7, [1.0, 0.0, 0.0, 0.0]);
        node.backup(3, [1.0, 0.0, 0.0, 0.0]);
        let params = IsmctsParams::defaults();
        assert_eq!(uct_select(&node, &[7, 3], 0, &params).expect("select"), 3);
    }

    #[test]
    fn eps_gate_separates_close_values() {
        // Gap of 1e-9 >> 1e-12: the better arm wins outright, not the tie arm.
        let mut node = IsNode::new();
        node.backup(3, [1.0, 0.0, 0.0, 0.0]);
        node.backup(7, [1.0 + 1e-9, 0.0, 0.0, 0.0]);
        let params = IsmctsParams::defaults();
        assert_eq!(uct_select(&node, &[3, 7], 0, &params).expect("select"), 7);
    }

    #[test]
    fn stable_hash_tie_is_deterministic() {
        let mut node = IsNode::new();
        node.backup(11, [2.0, 0.0, 0.0, 0.0]);
        node.backup(22, [2.0, 0.0, 0.0, 0.0]);
        let params =
            IsmctsParams { tie: TieBreak::StableHash, ..IsmctsParams::defaults() };
        let first = uct_select(&node, &[11, 22], 0, &params).expect("select");
        let second = uct_select(&node, &[22, 11], 0, &params).expect("select");
        assert_eq!(first, second);
        // ... and equals the min-sha arm.
        let winner =
            if sha_hex(11) < sha_hex(22) { 11 } else { 22 };
        assert_eq!(first, winner);
    }

    #[test]
    fn empty_legal_is_err() {
        let node = IsNode::new();
        let params = IsmctsParams::defaults();
        assert_eq!(uct_select(&node, &[], 0, &params), Err(SearchError::EmptyLegal));
    }

    #[test]
    fn duplicate_legal_is_err() {
        let node = IsNode::new();
        let params = IsmctsParams::defaults();
        assert_eq!(
            uct_select(&node, &[4, 4], 0, &params),
            Err(SearchError::DuplicateAction { action: 4 })
        );
    }

    #[test]
    fn bad_seat_and_params_rejected() {
        let node = visited_node();
        let params = IsmctsParams::defaults();
        assert_eq!(
            uct_select(&node, &[10, 20], 4, &params),
            Err(SearchError::InvalidSeat { seat: 4 })
        );
        let bad = IsmctsParams { uct_c: f64::NAN, ..params };
        assert!(uct_select(&node, &[10, 20], 0, &bad).is_err());
    }

    #[test]
    fn backup_counters_accumulate() {
        let mut node = IsNode::new();
        node.backup(5, [1.0, 2.0, 3.0, 4.0]);
        node.backup(5, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(node.visits, 2);
        assert_eq!(node.stats(5).expect("stats").visits, 2);
        assert_eq!(node.mean_vector(5).expect("mean"), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(node.scalar_mean(5, 2).expect("scalar"), Some(3.0));
    }
}
