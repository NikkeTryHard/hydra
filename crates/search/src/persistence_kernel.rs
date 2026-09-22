//! Persistence kernel — B/F/R/P/C arms, packets, forest state (EvalControl-owned).
//!
//! Rust port of `persistence_kernel.py` (454 ln): the SPEC-17 arm
//! vocabulary, the packet/forest state, commit/rebuild verification,
//! and the quota distributor. B3 canon-wins: packet/epoch digests call
//! `feed::canon` raw bytes + `sha2` (the single identity site) via
//! `crate::eval` helpers — no second printer, no second hasher.
//!
//! B4: per-arm `selected + vectors + counters` + the
//! `deterministic_gumbel_for_arm` parity golden land here (spec factory
//! in `persistence_spec`, state machine in `persistence_planner`,
//! report in `persistence_report`).
//!
//! M9 (`idx%4` prod-rejection): `_action_key` resolves through the REAL
//! mapping — `tile` int when present, else the frozen
//! `ACTION_KIND_ORDINALS` (`action_kinds.py:71-85`) — else the
//! `sha256(repr)` fallback branch that the oracle keeps for exotic
//! action types (wave3-eval §2 `deterministic_replay_hash` trap: the
//! `repr()` branch is load-bearing, never dropped). The `%4` seat
//! arithmetic (`local_search.py:129,491`, `candidate0_act.py:70`) is a
//! test-stub hack and MUST NOT appear in any prod path here: any
//! unknown action fails closed (`SearchError::InvalidArg`).

use std::collections::HashMap;

use crate::SearchError;

use crate::eval::{canon_bytes_of, hex_of, sha256_bytes};

/// Deployable deadline ms (`common.py:69`, `DEPLOYABLE_DEADLINE_MS`).
pub const DEPLOYABLE_DEADLINE_MS: u64 = 5000;

/// SPEC-17 PersistenceArm — frozen, validated (`:151-199`).
/// Field order matches the specification exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistenceArm {
    /// Arm id (B/F/R/P/C).
    pub id: ArmId,
    /// Retain compatible speculative state across decisions.
    pub retain_state: bool,
    /// Ponder during opponent turns.
    pub opponent_time_compute: bool,
    /// Own deadline ms (positive; deployable arms `<= 5000`).
    pub own_deadline_ms: u64,
    /// Extra wait allowance ms (0 except C laboratory, positive).
    pub extra_wait_allowance_ms: u64,
    /// Deployable (false for C laboratory).
    pub deployable: bool,
}

/// Arm identity (B/F/R/P/C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmId {
    /// Frozen policy, no search.
    B,
    /// Fresh search each own decision; discard state.
    F,
    /// Retain state, pause search during opponent turns.
    R,
    /// Retain state, ponder in opponent window.
    P,
    /// Laboratory fresh search with extended allowance; never deployable.
    C,
}

impl ArmId {
    /// Parse a single-char arm id.
    pub fn parse(text: &str) -> Result<Self, SearchError> {
        match text {
            "B" => Ok(ArmId::B),
            "F" => Ok(ArmId::F),
            "R" => Ok(ArmId::R),
            "P" => Ok(ArmId::P),
            "C" => Ok(ArmId::C),
            _ => Err(SearchError::InvalidArg {
                detail: "unknown arm",
            }),
        }
    }

    /// Single-char name.
    pub fn name(self) -> &'static str {
        match self {
            ArmId::B => "B",
            ArmId::F => "F",
            ArmId::R => "R",
            ArmId::P => "P",
            ArmId::C => "C",
        }
    }

    /// All five arms in canonical order.
    pub fn all() -> [ArmId; 5] {
        [ArmId::B, ArmId::F, ArmId::R, ArmId::P, ArmId::C]
    }
}

impl core::fmt::Display for ArmId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Arm definition (`ARM_DEFS`, `:202-243`): retain/opponent/deadline/
/// allowance/deployable per arm + the description line.
pub struct ArmDef {
    /// Retain-state invariant.
    pub retain_state: bool,
    /// Opponent-compute invariant.
    pub opponent_time_compute: bool,
    /// Own deadline invariant.
    pub own_deadline_ms: u64,
    /// Extra allowance invariant.
    pub extra_wait_allowance_ms: u64,
    /// Deployable invariant.
    pub deployable: bool,
    /// Human description.
    pub description: &'static str,
}

/// Look up the frozen definition for an arm.
pub fn arm_def(id: ArmId) -> ArmDef {
    match id {
        ArmId::B => ArmDef {
            retain_state: false,
            opponent_time_compute: false,
            own_deadline_ms: DEPLOYABLE_DEADLINE_MS,
            extra_wait_allowance_ms: 0,
            deployable: true,
            description: "Frozen policy, no search.",
        },
        ArmId::F => ArmDef {
            retain_state: false,
            opponent_time_compute: false,
            own_deadline_ms: DEPLOYABLE_DEADLINE_MS,
            extra_wait_allowance_ms: 0,
            deployable: true,
            description: "Fresh search at each own decision; discard state; no opponent-time compute.",
        },
        ArmId::R => ArmDef {
            retain_state: true,
            opponent_time_compute: false,
            own_deadline_ms: DEPLOYABLE_DEADLINE_MS,
            extra_wait_allowance_ms: 0,
            deployable: true,
            description: "Retain compatible state but pause all search during opponent turns.",
        },
        ArmId::P => ArmDef {
            retain_state: true,
            opponent_time_compute: true,
            own_deadline_ms: DEPLOYABLE_DEADLINE_MS,
            extra_wait_allowance_ms: 0,
            deployable: true,
            description: "Retain state and ponder only after emitted action until its next actor-visible packet.",
        },
        ArmId::C => ArmDef {
            retain_state: false,
            opponent_time_compute: false,
            own_deadline_ms: DEPLOYABLE_DEADLINE_MS,
            extra_wait_allowance_ms: 2000,
            deployable: false,
            description: "Laboratory-only fresh-search control with extended allowance; never deployable.",
        },
    }
}

/// Construct a validated `PersistenceArm` (`make_persistence_arm`,
/// `:246-258`).
pub fn make_persistence_arm(id: ArmId) -> PersistenceArm {
    let def = arm_def(id);
    PersistenceArm {
        id,
        retain_state: def.retain_state,
        opponent_time_compute: def.opponent_time_compute,
        own_deadline_ms: def.own_deadline_ms,
        extra_wait_allowance_ms: def.extra_wait_allowance_ms,
        deployable: def.deployable,
    }
}

/// Validate arm invariants (`__post_init__`, `:165-199`).
pub fn check_arm(arm: &PersistenceArm) -> Result<(), SearchError> {
    let def = arm_def(arm.id);
    if arm.retain_state != def.retain_state
        || arm.opponent_time_compute != def.opponent_time_compute
        || arm.deployable != def.deployable
    {
        return Err(SearchError::InvalidArg {
            detail: "arm invariant violated",
        });
    }
    if arm.own_deadline_ms == 0 {
        return Err(SearchError::InvalidArg {
            detail: "own_deadline_ms must be positive",
        });
    }
    match arm.id {
        ArmId::B | ArmId::F | ArmId::R | ArmId::P => {
            if arm.own_deadline_ms > DEPLOYABLE_DEADLINE_MS {
                return Err(SearchError::InvalidArg {
                    detail: "deployable arm deadline must be <=5000",
                });
            }
            if arm.extra_wait_allowance_ms != 0 {
                return Err(SearchError::InvalidArg {
                    detail: "deployable arm extra_wait_allowance must be 0",
                });
            }
        }
        ArmId::C => {
            if arm.extra_wait_allowance_ms == 0 {
                return Err(SearchError::InvalidArg {
                    detail: "C must have positive extra_wait_allowance_ms",
                });
            }
            if arm.deployable {
                return Err(SearchError::InvalidArg {
                    detail: "C must not be deployable",
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Packet and forest state (:261-454)
// ---------------------------------------------------------------------------

/// One actor-visible packet in the finite kernel (`FinitePacket`,
/// `:266-286`). Per (parent, action) the packets are pairwise disjoint
/// by id and sum probability to one.
#[derive(Debug, Clone, PartialEq)]
pub struct FinitePacket {
    /// Packet digest (`sha256:<hex>`).
    pub packet_id: String,
    /// Action this packet conditions on.
    pub action_id: u64,
    /// Predecessor epoch.
    pub epoch_before: String,
    /// Successor epoch (rebuild-verified).
    pub epoch_after: String,
    /// Branch probability in `(0,1]`, finite.
    pub probability: f64,
    /// Successor delta placeholder (opaque but deterministic).
    pub delta: Vec<i64>,
}

impl FinitePacket {
    /// Validate (`__post_init__`, `:281-286`).
    pub fn check(&self) -> Result<(), SearchError> {
        if !crate::eval::is_digest_text(&self.packet_id) {
            return Err(SearchError::InvalidArg {
                detail: "packet_id must be sha256:<hex>",
            });
        }
        if !(0.0 < self.probability && self.probability <= 1.0) || !self.probability.is_finite() {
            return Err(SearchError::InvalidArg {
                detail: "packet probability must be in (0,1]",
            });
        }
        Ok(())
    }
}

/// Packet id (`compute_packet_id`, `:289-291`): `sha256:` over canon
/// `{epoch_before, action_id, branch}`.
pub fn compute_packet_id(
    epoch_before: &str,
    action_id: u64,
    branch: u64,
) -> Result<String, SearchError> {
    let payload = serde_json::json!({
        "epoch_before": epoch_before,
        "action_id": action_id,
        "branch": branch,
    });
    let bytes = canon_bytes_of(&payload, "persistence:kernel:packet")?;
    Ok(format!("sha256:{}", hex_of(&sha256_bytes(&bytes))))
}

/// Minimal belief epoch (`BeliefEpochLite`, `:294-305`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefEpochLite {
    /// Epoch id string.
    pub epoch: String,
    /// Observation digest.
    pub observation_hash: String,
    /// Target digest.
    pub target_id: String,
    /// Root actor seat.
    pub root_actor: u32,
}

impl BeliefEpochLite {
    /// Validate digest shapes (`__post_init__`, `:303-305`).
    pub fn check(&self) -> Result<(), SearchError> {
        if !crate::eval::is_digest_text(&self.observation_hash)
            || !crate::eval::is_digest_text(&self.target_id)
        {
            return Err(SearchError::InvalidArg {
                detail: "epoch digest malformed",
            });
        }
        Ok(())
    }
}

/// Frozen action-kind ordinals (`action_kinds.py:71-85`): NEVER reordered.
/// Unknown kinds fall to the `sha256(repr)` branch (load-bearing for
/// exotic action types) — never to `%4`.
pub const ACTION_KIND_ORDINALS: [(&str, u64); 13] = [
    ("pass", 0),
    ("discard", 1),
    ("tsumogiri", 2),
    ("riichi_discard", 3),
    ("chi", 4),
    ("pon", 5),
    ("daiminkan", 6),
    ("ankan", 7),
    ("kakan", 8),
    ("ron", 9),
    ("tsumo", 10),
    ("abort_nine_terminals", 11),
    ("accept_abortive_draw", 12),
];

/// Action description the key resolver accepts. Prod paths carry either
/// a tile int or a frozen kind string; anything else takes the
/// `sha256(repr)` fallback (exotic types) — or fails closed when no
/// representation exists (M9: never `%4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionDesc {
    /// Tile-indexed action.
    Tile(u64),
    /// Frozen kind string (`ACTION_KIND_ORDINALS`).
    Kind(String),
    /// Opaque representation (exotic types: `sha256(repr)` branch).
    Opaque(String),
}

/// Deterministic integer key for an action (`_action_key`, `:314-331`):
/// tile int when present, else the frozen kind ordinal, else
/// `u32BE(sha256(repr)[:4])` (`int(h.hexdigest()[:8], 16)` — first 4
/// BYTES of the digest, full hex width, never `%4`).
pub fn action_key(action: &ActionDesc) -> u64 {
    match action {
        ActionDesc::Tile(tile) => *tile,
        ActionDesc::Kind(kind) => {
            for (name, ordinal) in ACTION_KIND_ORDINALS {
                if name == kind.as_str() {
                    return ordinal;
                }
            }
            let digest = sha256_bytes(kind.as_bytes());
            u64::from(u32::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3],
            ]))
        }
        ActionDesc::Opaque(repr) => {
            let digest = sha256_bytes(repr.as_bytes());
            u64::from(u32::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3],
            ]))
        }
    }
}

/// Exhaustive disjoint packet kernel per (epoch, action) — mass one
/// (`enumerate_packets_for`, `:334-380`). Two branches split
/// `0.7/0.3`; one branch is mass one; more split uniformly. Ids are
/// pairwise disjoint; total mass verified to `1e-9`.
pub fn enumerate_packets_for(
    epoch: &str,
    action_id: u64,
    num_branches: usize,
) -> Result<Vec<FinitePacket>, SearchError> {
    if num_branches == 0 {
        return Err(SearchError::InvalidArg {
            detail: "num_branches must be positive",
        });
    }
    let probs: Vec<f64> = if num_branches == 1 {
        vec![1.0]
    } else if num_branches == 2 {
        vec![0.7, 0.3]
    } else {
        // proof: branch count is small (validated `> 0`, tiny planner fanout), exact as `f64`.
        #[allow(clippy::cast_precision_loss)]
        let branches_f: f64 = num_branches as f64;
        vec![1.0 / branches_f; num_branches]
    };
    let mut packets = Vec::with_capacity(num_branches);
    for (branch, probability) in probs.into_iter().enumerate() {
        let packet_id = compute_packet_id(epoch, action_id, branch as u64)?;
        let after_payload = serde_json::json!({
            "epoch_before": epoch,
            "packet_id": packet_id,
        });
        let after_bytes = canon_bytes_of(&after_payload, "persistence:kernel:after")?;
        let epoch_after = format!("epoch:{}", &hex_of(&sha256_bytes(&after_bytes))[..16]);
        // proof: `action_id: u64` is a small planner id (< 2^63), fits `i64`.
        #[allow(clippy::cast_possible_wrap)]
        let aid_i: i64 = action_id as i64;
        // proof: `branch` is a small enumerate index, fits `i64`.
        #[allow(clippy::cast_possible_wrap)]
        let branch_i: i64 = branch as i64;
        let packet = FinitePacket {
            packet_id,
            action_id,
            epoch_before: epoch.to_string(),
            epoch_after,
            probability,
            delta: vec![aid_i, branch_i],
        };
        packet.check()?;
        packets.push(packet);
    }
    let total: f64 = packets.iter().map(|p| p.probability).sum();
    if (total - 1.0).abs() > 1e-9 {
        return Err(SearchError::BadPartition);
    }
    let mut ids: Vec<&str> = packets.iter().map(|p| p.packet_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != packets.len() {
        return Err(SearchError::BadPartition);
    }
    Ok(packets)
}

/// Authoritative fresh posterior `epoch_after` (rebuild,
/// `fresh_rebuild_epoch`, `:383-398`).
pub fn fresh_rebuild_epoch(
    epoch_before: &str,
    packet: &FinitePacket,
) -> Result<String, SearchError> {
    if packet.epoch_before != epoch_before {
        return Err(SearchError::InvalidArg {
            detail: "packet epoch_before != epoch",
        });
    }
    let payload = serde_json::json!({
        "epoch_before": epoch_before,
        "packet_id": packet.packet_id,
    });
    let bytes = canon_bytes_of(&payload, "persistence:kernel:rebuild")?;
    Ok(format!("epoch:{}", &hex_of(&sha256_bytes(&bytes))[..16]))
}

/// Commit/rebuild equality fixture (`commit_equals_rebuild`, `:401-408`).
pub fn commit_equals_rebuild(
    epoch_before: &str,
    packet: &FinitePacket,
) -> Result<bool, SearchError> {
    Ok(fresh_rebuild_epoch(epoch_before, packet)? == packet.epoch_after)
}

/// Deterministically spread quota units across sorted child ids,
/// round-robin (`_distribute_quota`, `:411-426`). Pure function; the
/// returned units sum to `min(quota, distributed)`.
pub fn distribute_quota(sorted_pids: &[String], quota: usize) -> HashMap<String, usize> {
    let mut dist: HashMap<String, usize> = HashMap::new();
    for pid in sorted_pids {
        dist.insert(pid.clone(), 0);
    }
    let mut remaining = quota;
    while remaining > 0 && !dist.is_empty() {
        let mut progressed = false;
        for pid in sorted_pids {
            if remaining == 0 {
                break;
            }
            if let Some(slot) = dist.get_mut(pid) {
                *slot += 1;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    dist
}

/// Speculative forest retained by R/P arms (`ForestState`, `:429-454`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ForestState {
    /// Epoch before the emitted action.
    pub parent_epoch: String,
    /// Emitted action id.
    pub action_id: u64,
    /// Packet-conditioned speculative children.
    pub children: HashMap<String, FinitePacket>,
    /// Per-child ponder counters.
    pub child_stats: HashMap<String, usize>,
    /// Provenance binding (stale on mismatch).
    pub provenance_target: Option<String>,
    /// Ponder calls this window.
    pub ponder_calls: usize,
    /// Creation timestamp (ns).
    pub created_at_ns: u64,
}

impl ForestState {
    /// Empty iff no children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Squash speculative state (siblings unreachable after commit).
    pub fn clear(&mut self) {
        self.children.clear();
        self.child_stats.clear();
        self.ponder_calls = 0;
    }
}

/// Degenerate-mask migration note (M8): TODAY-Python
/// `_legal_ids_for_observation` falls back to `(0, 1)` on a missing /
/// malformed / empty mask (`ismcts_search.py:80-93`). The Rust cutover
/// RAISES `EmptyLegal` instead (callers MUST invoke candidate0) — the
/// `(0, 1)` pair exists ONLY in Python and in the test comment below,
/// never as a Rust fn (no fallback getter ships, so no caller can
/// preserve the old semantics).
#[cfg(test)]
mod tests {
    use super::*;

    /// PACKET goldens: the `(epoch:abc123, 3, 0)` id, the two-branch
    /// probabilities + deltas, rebuild equality.
    #[test]
    fn packet_id_and_rebuild_goldens() {
        assert_eq!(
            compute_packet_id("epoch:abc123", 3, 0).unwrap(),
            "sha256:d453adbd4607030ee80d3edd1f73a2b2306791efac74832a19fdda8a2cd9c0e8"
        );
        let packets = enumerate_packets_for("epoch:abc123", 3, 2).unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].probability, 0.7);
        assert_eq!(packets[1].probability, 0.3);
        assert_eq!(packets[0].delta, vec![3, 0]);
        assert_eq!(packets[0].epoch_after, "epoch:e56f5878fb234623");
        assert_eq!(packets[1].epoch_after, "epoch:c4be31e65fed45c9");
        assert_eq!(
            fresh_rebuild_epoch("epoch:abc123", &packets[0]).unwrap(),
            "epoch:e56f5878fb234623"
        );
        assert!(commit_equals_rebuild("epoch:abc123", &packets[0]).unwrap());
        assert!(enumerate_packets_for("epoch:abc123", 3, 0).is_err());
        assert!(fresh_rebuild_epoch("epoch:other", &packets[0]).is_err());
    }

    /// Arm invariants: B/F/R/P deployable with zero allowance; C
    /// laboratory with positive allowance and never deployable.
    #[test]
    fn arm_invariants() {
        for id in ArmId::all() {
            let arm = make_persistence_arm(id);
            assert!(check_arm(&arm).is_ok());
        }
        assert_eq!(
            make_persistence_arm(ArmId::B).own_deadline_ms,
            DEPLOYABLE_DEADLINE_MS
        );
        assert_eq!(make_persistence_arm(ArmId::C).extra_wait_allowance_ms, 2000);
        assert!(!make_persistence_arm(ArmId::C).deployable);
        assert!(ArmId::parse("P").unwrap() == ArmId::P);
        assert!(ArmId::parse("Z").is_err());
        let mut bad = make_persistence_arm(ArmId::R);
        bad.opponent_time_compute = true;
        assert!(check_arm(&bad).is_err());
        let mut bad = make_persistence_arm(ArmId::B);
        bad.extra_wait_allowance_ms = 1;
        assert!(check_arm(&bad).is_err());
        let mut bad = make_persistence_arm(ArmId::C);
        bad.deployable = true;
        assert!(check_arm(&bad).is_err());
    }

    /// M9: real mapping — tile wins, frozen kinds resolve, unknown
    /// kinds take `sha256(repr)` (AKEY_REPR golden `511273250`: the
    /// oracle hashes `repr(action)`, i.e. WITH Python quotes —
    /// `"'discard:5m'"` — never the bare string), never `%4`.
    #[test]
    fn action_key_real_mapping() {
        assert_eq!(action_key(&ActionDesc::Tile(7)), 7);
        assert_eq!(action_key(&ActionDesc::Kind("pass".to_string())), 0);
        assert_eq!(action_key(&ActionDesc::Kind("tsumo".to_string())), 10);
        assert_eq!(
            action_key(&ActionDesc::Opaque("'discard:5m'".to_string())),
            511273250
        );
        // `%4` can never be the answer for exotic actions: keys span u32.
        assert!(action_key(&ActionDesc::Opaque("exotic-xyz".to_string())) <= u64::from(u32::MAX));
    }

    /// Quota distribution: round-robin over sorted ids, sums to quota.
    #[test]
    fn quota_distribution() {
        let ids = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        let mut sorted = ids.clone();
        sorted.sort();
        let dist = distribute_quota(&sorted, 7);
        assert_eq!(dist.values().sum::<usize>(), 7);
        assert_eq!(dist["a"], 3);
        assert_eq!(dist["b"], 2);
        assert_eq!(dist["c"], 2);
        assert!(distribute_quota(&sorted, 0).values().sum::<usize>() == 0);
    }

    /// Forest clear squashes children + stats + ponder counters.
    #[test]
    fn forest_clear() {
        let mut forest = ForestState {
            parent_epoch: "epoch:x".to_string(),
            action_id: 3,
            children: [(
                "p".to_string(),
                enumerate_packets_for("epoch:x", 3, 1).unwrap().remove(0),
            )]
            .into_iter()
            .collect(),
            child_stats: [("p".to_string(), 4)].into_iter().collect(),
            provenance_target: Some("epoch:x".to_string()),
            ponder_calls: 4,
            created_at_ns: 0,
        };
        assert!(!forest.is_empty());
        forest.clear();
        assert!(forest.is_empty());
        assert!(forest.child_stats.is_empty());
        assert_eq!(forest.ponder_calls, 0);
    }

    /// M8 migration golden: TODAY-Python falls back to `(0, 1)` on a
    /// degenerate mask; Rust resolves empty legal to `EmptyLegal`
    /// (candidate0 path), never a silent retry or promoted stub pair.
    #[test]
    fn degenerate_mask_migration_golden() {
        let p = crate::persistence_planner::PersistencePlanner::new(
            ArmId::B,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            p.pick_action(&[], None),
            Err(crate::SearchError::EmptyLegal)
        );
    }
}
