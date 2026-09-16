//! `despot`: K=16 packs + per-idx independence.
//!
//! Parity: `despot_core.py:378-396` (`_scenario_seed_bytes`: sha256 over the
//! CANONICAL payload `{candidate_id, case_id, scenario_idx, attempt_id,
//! master=wp08c…}` — canon-wins B3, so seeding calls `feed::canon`, never a
//! second printer) + `despot_act.py:57-325` (sorted-legal expand, feasible
//! lower values, `1e-12` best-feasible tie) + `despot_search.py:116-213`
//! (per-idx independence: each scenario draws from its OWN stream, no
//! ledger dup) + `despot_core.py:250` (`validate_packet_partition`,
//! `tol=1e-9`).

use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::ismcts::{TieBreak, check_legal};

/// DESPOT scenario count (`K=16` packs).
pub const DESPOT_K: u32 = 16;
/// Packet-partition tolerance (`validate_packet_partition tol=1e-9`).
pub const PARTITION_TOL: f64 = 1e-9;
/// Best-feasible tie gate (`despot_act.py:282`: `abs(v-max) < 1e-12`).
pub const DESPOT_EPS: f64 = 1e-12;
/// Scenario master seed tag (`_MASTER_SEED = b"wp08c_despot_natural_v1"`).
pub const DESPOT_MASTER_TAG: &str = "wp08c_despot_natural_v1";

/// Master seed bytes (hex of the tag, as the oracle embeds
/// `_MASTER_SEED.hex()` in the canonical payload).
pub fn despot_master_hex() -> String {
    let mut out = String::with_capacity(DESPOT_MASTER_TAG.len() * 2);
    for byte in DESPOT_MASTER_TAG.as_bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// One scenario: deterministic seed + world handle + importance weight.
///
/// Natural ONLY: `log_target == log_proposal` (else `ContractError` —
/// the proposal law is NEVER consulted); weight `w = 1/K`.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Scenario index (`0..K`).
    pub idx: u32,
    /// 32-byte deterministic seed (`_scenario_seed_bytes`).
    pub seed: [u8; 32],
    /// Opaque world handle (arena-side `u64`, never a Python object).
    pub world_ref: u64,
    /// Target log-density (`-ln(K)` natural).
    pub log_target: f64,
    /// Proposal log-density (MUST equal target for natural scenarios).
    pub log_proposal: f64,
    /// Importance weight (`1/K`).
    pub weight: f64,
}

/// Deterministic 32-byte scenario seed (`despot_core.py:378-396` parity):
/// sha256 over the CANONICAL payload `{candidate_id, case_id, scenario_idx,
/// attempt_id, master}`.
///
/// Canon-wins (B3): the payload serializes through `feed::canon` (single
/// canon site). Key order inside the payload is the JCS sort (caller-side
/// `BTreeMap` discipline matches the oracle's `canonical_bytes`).
pub fn scenario_seed(
    candidate_id: &str,
    case_id: &str,
    scenario_idx: u32,
    attempt_id: u32,
) -> Result<[u8; 32], SearchError> {
    if candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "candidate_id must be non-empty" });
    }
    if case_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "case_id must be non-empty" });
    }
    let mut map = std::collections::BTreeMap::new();
    map.insert("attempt_id".to_string(), serde_json::json!(attempt_id));
    map.insert("candidate_id".to_string(), serde_json::json!(candidate_id));
    map.insert("case_id".to_string(), serde_json::json!(case_id));
    map.insert("master".to_string(), serde_json::json!(despot_master_hex()));
    map.insert("scenario_idx".to_string(), serde_json::json!(scenario_idx));
    let bytes = hydra_feed::canon::canonical_bytes(&map, "despot:scenario_seed")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    let digest = Sha256::digest(bytes.as_slice());
    let mut seed = [0u8; 32];
    seed.copy_from_slice(digest.as_slice());
    Ok(seed)
}

/// Sample one K-pack: `K` scenarios with per-idx independent world handles.
///
/// `world_for_idx` is the arena-side M1 belief closure (Rust closure over
/// interned keys — no per-sim Python callback): it maps `idx -> world
/// handle`. Per-idx independence is structural (each idx resolves its own
/// handle; no shared ledger draw), matching `despot_search.py:116-213`.
pub fn sample_packs(
    candidate_id: &str,
    case_id: &str,
    count: u32,
    mut world_for_idx: impl FnMut(u32) -> u64,
) -> Result<Vec<Scenario>, SearchError> {
    if count == 0 {
        return Err(SearchError::InvalidArg { detail: "pack count must be >= 1" });
    }
    let weight = 1.0 / count as f64;
    let logp = -(count as f64).ln();
    if !weight.is_finite() || !logp.is_finite() {
        return Err(SearchError::NonFinite { context: "despot pack weight" });
    }
    let mut packs = Vec::with_capacity(count as usize);
    let mut idx = 0;
    while idx < count {
        let seed = scenario_seed(candidate_id, case_id, idx, 0)?;
        packs.push(Scenario {
            idx,
            seed,
            world_ref: world_for_idx(idx),
            log_target: logp,
            log_proposal: logp,
            weight,
        });
        idx += 1;
    }
    Ok(packs)
}

/// Assert natural-only (`log_target == log_proposal` exactly; the proposal
/// law is NEVER consulted).
pub fn check_natural(scenario: &Scenario) -> Result<(), SearchError> {
    if scenario.log_target != scenario.log_proposal {
        return Err(SearchError::InvalidArg { detail: "non-natural scenario: proposal consulted" });
    }
    if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
        return Err(SearchError::NonFinite { context: "despot scenario weight" });
    }
    Ok(())
}

/// Validate a packet partition (`despot_core.py:250`): successor ids
/// distinct + probabilities sum to `1 +- 1e-9`, else `BadPartition`.
///
/// `entries` carries `(packet_id, prob)`; `prob` entries finite-checked.
pub fn validate_packet_partition(entries: &[(u32, f64)]) -> Result<(), SearchError> {
    if entries.is_empty() {
        return Err(SearchError::InvalidArg { detail: "packet entries must be non-empty" });
    }
    let mut ids: Vec<u32> = entries.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    let mut i = 1;
    while i < ids.len() {
        if ids[i] == ids[i - 1] {
            return Err(SearchError::BadPartition);
        }
        i += 1;
    }
    let mut sum = 0.0;
    for (_, prob) in entries {
        if !prob.is_finite() || *prob < 0.0 {
            return Err(SearchError::NonFinite { context: "despot packet prob" });
        }
        sum += *prob;
    }
    if (sum - 1.0).abs() > PARTITION_TOL {
        return Err(SearchError::BadPartition);
    }
    Ok(())
}

/// Best feasible root action (`despot_act.py:279-303`): max lower value,
/// `abs(v-max) < 1e-12` candidates, `stable_hash` -> min sha arm else min
/// id. Empty lower map is the caller's `completed=false` (candidate0)
/// path — surfaced here as `EmptyLegal`, never a default.
pub fn best_feasible(
    lower: &[(u32, f64)],
    tie: TieBreak,
    candidate_id: &str,
) -> Result<u32, SearchError> {
    if lower.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let legal: Vec<u32> = lower.iter().map(|(aid, _)| *aid).collect();
    check_legal(&legal)?;
    let max_val = lower.iter().map(|(_, value)| *value).fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return Err(SearchError::NonFinite { context: "despot lower max" });
    }
    let mut candidates = Vec::new();
    for (aid, value) in lower {
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "despot lower value" });
        }
        if (*value - max_val).abs() < DESPOT_EPS {
            candidates.push(*aid);
        }
    }
    if candidates.len() == 1 {
        return Ok(candidates[0]);
    }
    match tie {
        TieBreak::StableHash => {
            // `sha256(f"{candidate_id}:{aid}")` hex order
            // (`_hash_tie_break`, `despot_core.py:399-418`).
            let hex = |aid: u32| -> String {
                let digest = Sha256::digest(format!("{candidate_id}:{aid}").as_bytes());
                let mut out = String::with_capacity(64);
                for byte in digest {
                    out.push_str(&format!("{byte:02x}"));
                }
                out
            };
            let mut best = candidates[0];
            let mut best_key = hex(best);
            for aid in candidates.into_iter().skip(1) {
                let key = hex(aid);
                if key < best_key {
                    best = aid;
                    best_key = key;
                }
            }
            Ok(best)
        }
        TieBreak::LowestActionId | TieBreak::Lexicographic => {
            Ok(candidates.into_iter().min().unwrap_or(lower[0].0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k16_pack_weights_and_natural() {
        let packs = sample_packs("cand_k", "case_k", DESPOT_K, |idx| 1_000 + idx as u64)
            .expect("packs");
        assert_eq!(packs.len(), 16);
        let mut sum = 0.0;
        for pack in &packs {
            check_natural(pack).expect("natural");
            sum += pack.weight;
            assert_eq!(pack.log_target, pack.log_proposal);
        }
        assert!((sum - 1.0).abs() <= 1e-9, "pack weights sum {sum}");
    }

    #[test]
    fn per_idx_independence_seeds_differ() {
        // Adjacent idx MUST derive distinct 32 B seeds (no ledger dup).
        let left = scenario_seed("c", "case", 0, 0).expect("seed");
        let right = scenario_seed("c", "case", 1, 0).expect("seed");
        assert_ne!(left, right);
        // Same (cand, case, idx, attempt) re-derives identically.
        assert_eq!(left, scenario_seed("c", "case", 0, 0).expect("seed"));
        // Attempt bumps the seed (collision-loop KAT shape).
        assert_ne!(left, scenario_seed("c", "case", 0, 1).expect("seed"));
    }

    #[test]
    fn partition_tol_gate() {
        assert!(validate_packet_partition(&[(1, 0.5), (2, 0.5)]).is_ok());
        // Dup packet id -> BadPartition.
        assert_eq!(
            validate_packet_partition(&[(1, 0.5), (1, 0.5)]),
            Err(SearchError::BadPartition)
        );
        // Mass 0.999999 (1e-6 off) exceeds 1e-9 tol -> BadPartition.
        assert_eq!(
            validate_packet_partition(&[(1, 0.5), (2, 0.499_999)]),
            Err(SearchError::BadPartition)
        );
    }

    #[test]
    fn best_feasible_eps_and_tie() {
        // Gap 1e-9 >> 1e-12: leader wins outright.
        let lower = [(3, 1.0), (7, 1.0 + 1e-9)];
        assert_eq!(
            best_feasible(&lower, TieBreak::LowestActionId, "c").expect("best"),
            7
        );
        // Gap 5e-13 < 1e-12: tie -> min id.
        let tied = [(9, 2.0), (4, 2.0 + 5e-13)];
        assert_eq!(
            best_feasible(&tied, TieBreak::LowestActionId, "c").expect("best"),
            4
        );
        // Stable-hash tie is deterministic across input orders.
        let pair = [(11, 3.0), (22, 3.0)];
        let flipped = [(22, 3.0), (11, 3.0)];
        assert_eq!(
            best_feasible(&pair, TieBreak::StableHash, "cand_s").expect("a"),
            best_feasible(&flipped, TieBreak::StableHash, "cand_s").expect("b")
        );
    }

    #[test]
    fn empty_lower_is_completed_false_shape() {
        assert_eq!(
            best_feasible(&[], TieBreak::LowestActionId, "c"),
            Err(SearchError::EmptyLegal)
        );
    }

    #[test]
    fn non_natural_rejected() {
        let mut packs =
            sample_packs("c", "case", 2, |idx| idx as u64).expect("packs");
        packs[0].log_proposal = 0.0;
        assert!(check_natural(&packs[0]).is_err());
    }
}
