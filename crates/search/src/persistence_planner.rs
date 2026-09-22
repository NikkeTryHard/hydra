//! Persistence planner — per-arm B/F/R/P/C state machine (EvalControl-owned).
//!
//! Rust port of `persistence_planner.py` (582 ln): the per-arm state
//! machine enforcing B/F/R/P/C semantics exactly (`:52-106` modes +
//! invariants). Timing-driven fallback (`time.monotonic_ns`,
//! `deadline_monotonic_ns` request field) is caller-side — this module
//! owns the deterministic core: selection, forest install/squash,
//! commit verification, ponder accounting, and the counters the report
//! stratifies. B3 canon-wins: choice scores use
//! `persistence_spec::deterministic_gumbel_for_arm` (sha preimage),
//! never a new hash.
//!
//! Modes (`:55-61`):
//! - B: one model call (frozen policy), no tree, no ponder.
//! - F: fresh bounded search each own decision; forest cleared after
//!   act; ponder is a no-op.
//! - R: retain forest after act; ponder does zero work; commit verified
//!   packet.
//! - P: retain forest; ponder work only in the opponent window; commit
//!   verified packet.
//! - C: laboratory fresh search with extended budget; no retained
//!   state; never deployable.
//!
//! Per-arm `selected + vectors + counters` (B4): [`ActOutcome`]
//! carries the selected action id, the four-seat value vector (zeros —
//! the planner is control, not evaluation; values bind upstream), and
//! the charged counters.

use std::collections::HashMap;

use crate::SearchError;

use crate::persistence_kernel::{
    ArmId, FinitePacket, ForestState, PersistenceArm, commit_equals_rebuild, distribute_quota,
    enumerate_packets_for, fresh_rebuild_epoch, make_persistence_arm,
};
use crate::persistence_spec::{
    GUMBEL_SEED_PLANNER, arm_budget, deterministic_gumbel_for_arm, validate_deadline_and_fallback,
};

/// Per-arm selection bias (`_pick_action_deterministically`, `:144`).
fn arm_bias(id: ArmId) -> f64 {
    match id {
        ArmId::B => 0.0,
        ArmId::F => 0.1,
        ArmId::R => 0.12,
        ArmId::P => 0.18,
        ArmId::C => 0.19,
    }
}

/// Act outcome: selected + vectors + counters (B4).
#[derive(Debug, Clone, PartialEq)]
pub struct ActOutcome {
    /// Selected action id (resolved via [`action`](Self::action_ids)
    /// ordering — the caller maps back to its action table).
    pub selected: u64,
    /// Candidate action ids in caller order.
    pub candidate_actions: Vec<u64>,
    /// Four-seat value vectors (zeros; control plane, not evaluation).
    pub value_vectors: Vec<[f64; 4]>,
    /// Model calls charged this act.
    pub model_calls: u64,
    /// Exact transitions charged this act.
    pub exact_transitions: u64,
    /// Fallback taken (candidate-0 path).
    pub fallback_used: bool,
    /// Deadline exceeded.
    pub timeout: bool,
    /// Completed (not fallback).
    pub completed: bool,
}

/// Commit outcome for stratification (`_commit_log`, `:340-361`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitEntry {
    /// Arm that committed.
    pub arm: ArmId,
    /// Committed packet id.
    pub packet_id: String,
    /// `fresh` (B/F/C) | `hit` | `miss_recovery` | `rebuild_no_forest`.
    pub outcome: String,
    /// Ponder calls charged before commit.
    pub ponder_calls: usize,
}

/// Per-arm planner state machine (`PersistencePlanner`, `:52-106`).
#[derive(Debug)]
pub struct PersistencePlanner {
    /// Arm under test.
    pub arm: PersistenceArm,
    /// Choice seed (default `persistence-factorial-v1`).
    pub seed: Vec<u8>,
    /// Effective deadline ms.
    pub deadline_ms: u64,
    /// Fallback margin ms.
    pub fallback_margin_ms: u64,
    /// Max model calls per act.
    pub max_model_calls: u64,
    /// Max transitions per act.
    pub max_transitions: u64,
    /// Retained speculative forest (R/P only).
    pub forest: Option<ForestState>,
    /// Current epoch.
    pub current_epoch: Option<String>,
    /// Last emitted action id.
    pub last_emitted_action: Option<u64>,
    /// Total model calls (incl. ponder).
    pub total_model_calls: u64,
    /// Total transitions (incl. ponder).
    pub total_transitions: u64,
    /// Accumulated joules (0.04/call + 0.005/transition).
    pub total_joules: f64,
    /// Ponder calls this window.
    pub ponder_budget_used: usize,
    /// Surprise strata counters.
    pub surprise_counts: HashMap<String, usize>,
    /// Commit log for stratification.
    pub commit_log: Vec<CommitEntry>,
}

impl PersistencePlanner {
    /// Increment a surprise counter; unknown names fail closed (the map
    /// is fixed to hit/miss/recovery at construction).
    fn bump(counts: &mut HashMap<String, usize>, name: &str) -> Result<(), SearchError> {
        match counts.get_mut(name) {
            Some(slot) => {
                *slot += 1;
                Ok(())
            }
            None => Err(SearchError::InvalidArg {
                detail: "unknown surprise outcome",
            }),
        }
    }

    /// Build a planner (`__init__`, `:70-106`): resolves the arm,
    /// applies spec/budget/deadline overrides, validates
    /// deadline/fallback, zeroes counters.
    pub fn new(
        arm: ArmId,
        deadline_ms: Option<u64>,
        fallback_margin_ms: Option<u64>,
        max_model_calls: Option<u64>,
        max_transitions: Option<u64>,
        seed: Option<Vec<u8>>,
    ) -> Result<Self, SearchError> {
        let arm_state = make_persistence_arm(arm);
        let budget = arm_budget(arm, max_model_calls, max_transitions);
        let spec_deadline = match arm {
            ArmId::C => arm_state.own_deadline_ms + arm_state.extra_wait_allowance_ms,
            _ => arm_state.own_deadline_ms,
        };
        let deadline = deadline_ms.unwrap_or(spec_deadline);
        let margin = fallback_margin_ms.unwrap_or(500);
        validate_deadline_and_fallback(&arm_state, deadline, margin)?;
        let mut surprise_counts = HashMap::new();
        surprise_counts.insert("hit".to_string(), 0);
        surprise_counts.insert("miss".to_string(), 0);
        surprise_counts.insert("recovery".to_string(), 0);
        Ok(PersistencePlanner {
            arm: arm_state,
            seed: seed.unwrap_or_else(|| GUMBEL_SEED_PLANNER.to_vec()),
            deadline_ms: deadline,
            fallback_margin_ms: margin,
            max_model_calls: budget.max_model_calls,
            max_transitions: budget.max_transitions,
            forest: None,
            current_epoch: None,
            last_emitted_action: None,
            total_model_calls: 0,
            total_transitions: 0,
            total_joules: 0.0,
            ponder_budget_used: 0,
            surprise_counts,
            commit_log: Vec::new(),
        })
    }

    /// Retained speculative state present and nonempty.
    pub fn has_retained_state(&self) -> bool {
        self.forest.as_ref().is_some_and(|f| !f.is_empty())
    }

    /// Deterministic epoch id from an observation digest + arm
    /// (`_new_epoch_for_obs`, `:118-132`).
    pub fn epoch_for_obs(&self, observation_hash: &str) -> String {
        use crate::eval::{hex_of, sha256_bytes};
        let digest = sha256_bytes(format!("{observation_hash}:{}", self.arm.id).as_bytes());
        format!("epoch:{}", &hex_of(&digest)[..16])
    }

    /// Deterministic pick (`_pick_action_deterministically`,
    /// `:134-153`): max `(gumbel + bias) % 1`, ties to the smaller
    /// action id (`isclose` equality class).
    pub fn pick_action(
        &self,
        legal_actions: &[u64],
        case_id: Option<&str>,
    ) -> Result<u64, SearchError> {
        if legal_actions.is_empty() {
            return Err(SearchError::EmptyLegal);
        }
        let resolved = case_id.unwrap_or("default");
        let mut best = legal_actions[0];
        let mut best_score = -1.0;
        for action in legal_actions {
            let score = (deterministic_gumbel_for_arm(self.arm.id, resolved, *action, &self.seed)
                + arm_bias(self.arm.id))
                % 1.0;
            if score > best_score || (score - best_score).abs() < 1e-9 && *action < best {
                best_score = score;
                best = *action;
            }
        }
        Ok(best)
    }

    /// Act at an own decision (`act`, `:198-309`): per-arm forest
    /// management BEFORE search (B/F/C discard; R/P squash on epoch
    /// mismatch and count miss+recovery), bounded search or frozen
    /// policy, forest install for retain arms, counter charging.
    /// `remaining_ms` carries the caller's deadline check (`None` =
    /// no deadline pressure); `remaining < margin` falls back
    /// immediately (`:244-264`).
    pub fn act(
        &mut self,
        epoch: &str,
        legal_actions: &[u64],
        case_id: Option<&str>,
        remaining_ms: Option<f64>,
    ) -> Result<ActOutcome, SearchError> {
        if legal_actions.is_empty() {
            return Err(SearchError::EmptyLegal);
        }
        match self.arm.id {
            ArmId::B | ArmId::F | ArmId::C => {
                self.forest = None;
            }
            ArmId::R | ArmId::P => {
                if let Some(forest) = &self.forest
                    && forest.parent_epoch != epoch
                {
                    Self::bump(&mut self.surprise_counts, "miss")?;
                    Self::bump(&mut self.surprise_counts, "recovery")?;
                    self.forest = None;
                }
            }
        }
        // proof: ms budgets are u64 (< 2^53), exact; deadline compare tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let margin_f: f64 = self.fallback_margin_ms as f64;
        if let Some(remaining) = remaining_ms
            && remaining < margin_f
        {
            let selected = legal_actions[0];
            self.current_epoch = Some(epoch.to_string());
            self.last_emitted_action = Some(selected);
            self.total_model_calls += self.max_model_calls;
            self.total_transitions += self.max_transitions;
            // proof: ms/call counters are u64 (< 2^53), exact; joules accounting tolerates 1ulp.
            #[allow(clippy::cast_precision_loss)]
            let calls_f: f64 = self.max_model_calls as f64;
            #[allow(clippy::cast_precision_loss)]
            let trans_f: f64 = self.max_transitions as f64;
            self.total_joules += calls_f * 0.04 + trans_f * 0.005;
            return Ok(ActOutcome {
                selected,
                candidate_actions: legal_actions.to_vec(),
                value_vectors: vec![[0.0; 4]],
                model_calls: self.max_model_calls,
                exact_transitions: self.max_transitions,
                fallback_used: true,
                timeout: true,
                completed: false,
            });
        }
        let (selected, calls, trans) = match self.arm.id {
            ArmId::B => (self.pick_action(legal_actions, case_id)?, 1, 0),
            _ => (
                self.pick_action(legal_actions, case_id)?,
                self.max_model_calls,
                self.max_transitions,
            ),
        };
        if matches!(self.arm.id, ArmId::R | ArmId::P) {
            let children = enumerate_packets_for(epoch, selected, 2)?;
            let mut stats = HashMap::new();
            for packet in &children {
                stats.insert(packet.packet_id.clone(), 0);
            }
            self.forest = Some(ForestState {
                parent_epoch: epoch.to_string(),
                action_id: selected,
                children: children
                    .into_iter()
                    .map(|p| (p.packet_id.clone(), p))
                    .collect(),
                child_stats: stats,
                provenance_target: Some(epoch.to_string()),
                ponder_calls: 0,
                created_at_ns: 0,
            });
        } else if let Some(forest) = &mut self.forest {
            forest.clear();
            self.forest = None;
        }
        self.current_epoch = Some(epoch.to_string());
        self.last_emitted_action = Some(selected);
        self.total_model_calls += calls;
        self.total_transitions += trans;
        // proof: call counters are u64 (< 2^53), exact; joules accounting tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let calls_f: f64 = calls as f64;
        #[allow(clippy::cast_precision_loss)]
        let trans_f: f64 = trans as f64;
        self.total_joules += calls_f * 0.04 + trans_f * 0.005;
        Ok(ActOutcome {
            selected,
            candidate_actions: legal_actions.to_vec(),
            value_vectors: vec![[0.0; 4]],
            model_calls: calls,
            exact_transitions: trans,
            fallback_used: false,
            timeout: false,
            completed: true,
        })
    }

    /// Observe the next actor-visible packet (`observe`, `:311-418`).
    /// B/F/C squash any forest and log `fresh`; R/P with no forest log
    /// `rebuild_no_forest` (+miss/recovery); R with ponder work raises;
    /// miss logs `miss_recovery` (+miss/recovery) and squashes; hit
    /// verifies commit/rebuild equality, promotes the realized child
    /// (siblings squashed), and counts `hit`.
    pub fn observe(&mut self, packet_id: &str, epoch_after: &str) -> Result<(), SearchError> {
        if packet_id.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "packet must carry packet_id",
            });
        }
        match self.arm.id {
            ArmId::B | ArmId::F | ArmId::C => {
                self.forest = None;
                self.current_epoch = Some(epoch_after.to_string());
                self.commit_log.push(CommitEntry {
                    arm: self.arm.id,
                    packet_id: packet_id.to_string(),
                    outcome: "fresh".to_string(),
                    ponder_calls: 0,
                });
                Ok(())
            }
            ArmId::R | ArmId::P => {
                if self.forest.as_ref().is_none_or(|f| f.is_empty()) {
                    Self::bump(&mut self.surprise_counts, "miss")?;
                    Self::bump(&mut self.surprise_counts, "recovery")?;
                    self.current_epoch = Some(epoch_after.to_string());
                    self.commit_log.push(CommitEntry {
                        arm: self.arm.id,
                        packet_id: packet_id.to_string(),
                        outcome: "rebuild_no_forest".to_string(),
                        ponder_calls: 0,
                    });
                    return Ok(());
                }
                let forest = self.forest.as_mut().ok_or(SearchError::Lock)?;
                if self.arm.id == ArmId::R && forest.ponder_calls != 0 {
                    return Err(SearchError::InvalidArg {
                        detail: "R must not have ponder work",
                    });
                }
                let hit_packet: Option<FinitePacket> = forest.children.get(packet_id).cloned();
                match hit_packet {
                    None => {
                        let ponder = forest.ponder_calls;
                        let arm = self.arm.id;
                        self.forest = None;
                        Self::bump(&mut self.surprise_counts, "miss")?;
                        Self::bump(&mut self.surprise_counts, "recovery")?;
                        self.current_epoch = Some(epoch_after.to_string());
                        self.commit_log.push(CommitEntry {
                            arm,
                            packet_id: packet_id.to_string(),
                            outcome: "miss_recovery".to_string(),
                            ponder_calls: ponder,
                        });
                        Ok(())
                    }
                    Some(packet) => {
                        let rebuilt = fresh_rebuild_epoch(&forest.parent_epoch, &packet)?;
                        if rebuilt != packet.epoch_after || epoch_after != rebuilt {
                            return Err(SearchError::BadPartition);
                        }
                        if !commit_equals_rebuild(&forest.parent_epoch, &packet)? {
                            return Err(SearchError::BadPartition);
                        }
                        let ponder = forest.ponder_calls;
                        let retained = packet.packet_id.clone();
                        let stat = forest.child_stats.get(&retained).copied().unwrap_or(0);
                        forest.children = [(retained.clone(), packet)].into_iter().collect();
                        forest.child_stats = [(retained, stat)].into_iter().collect();
                        self.current_epoch = Some(rebuilt);
                        Self::bump(&mut self.surprise_counts, "hit")?;
                        self.commit_log.push(CommitEntry {
                            arm: self.arm.id,
                            packet_id: packet_id.to_string(),
                            outcome: "hit".to_string(),
                            ponder_calls: ponder,
                        });
                        Ok(())
                    }
                }
            }
        }
    }

    /// Opponent-time compute (`ponder`, `:420-476`): B/F/R/C are no-ops
    /// (R asserts zero); P distributes 2 calls per child (or the
    /// `ponder_quota_total` round-robin) and charges model/transitions/
    /// joules. Past-deadline is a no-op.
    pub fn ponder(
        &mut self,
        remaining_ms: f64,
        ponder_quota_total: Option<usize>,
    ) -> Result<(), SearchError> {
        if !matches!(self.arm.id, ArmId::P) {
            if self.arm.id == ArmId::R
                && let Some(forest) = &self.forest
                && forest.ponder_calls != 0
            {
                return Err(SearchError::InvalidArg {
                    detail: "R ponder must remain zero",
                });
            }
            return Ok(());
        }
        let forest = match &mut self.forest {
            Some(forest) if !forest.is_empty() => forest,
            _ => return Ok(()),
        };
        if self.last_emitted_action.is_none() || remaining_ms <= 0.0 {
            return Ok(());
        }
        let total: usize = match ponder_quota_total {
            Some(quota) => {
                if quota == 0 {
                    return Err(SearchError::InvalidArg {
                        detail: "ponder_quota_total must be a positive int or None",
                    });
                }
                let mut sorted: Vec<String> = forest.children.keys().cloned().collect();
                sorted.sort();
                let dist = distribute_quota(&sorted, quota);
                let mut sum = 0;
                for (pid, units) in dist {
                    match forest.child_stats.get_mut(&pid) {
                        Some(slot) => *slot += units,
                        None => return Err(SearchError::Lock),
                    }
                    sum += units;
                }
                sum
            }
            None => {
                let mut total = 0;
                let per_child = if remaining_ms < 0.5 { 1 } else { 2 };
                for (applied, stat) in forest.child_stats.values_mut().enumerate() {
                    let units = if remaining_ms < 0.5 && applied > 0 {
                        1.min(per_child)
                    } else {
                        per_child
                    };
                    *stat += units;
                    total += units;
                    if remaining_ms < 0.5 && total >= 1 {
                        break;
                    }
                }
                total
            }
        };
        forest.ponder_calls += total;
        // proof: `total` is a small ponder-unit sum, fits `u64` on 64-bit.
        #[allow(clippy::cast_possible_truncation)]
        let total_u: u64 = total as u64;
        self.total_model_calls += total_u;
        self.total_transitions += total_u / 2;
        self.ponder_budget_used += total;
        // proof: ponder-unit sums are small (< 2^53), exact; joules accounting tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let total_f: f64 = total as f64;
        self.total_joules += total_f * 0.04;
        Ok(())
    }

    /// Telemetry snapshot (`telemetry_snapshot`, `:573-582`).
    pub fn surprise_count(&self, outcome: &str) -> usize {
        self.surprise_counts.get(outcome).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B/F fresh (no retention), R/P retain with installed forests;
    /// B charges exactly 1 call / 0 transitions; F charges its budget.
    #[test]
    fn act_forest_semantics_and_counters() {
        let mut b = PersistencePlanner::new(ArmId::B, None, None, None, None, None).unwrap();
        let out = b.act("epoch:a", &[3, 5], Some("case-001"), None).unwrap();
        assert_eq!((out.model_calls, out.exact_transitions), (1, 0));
        assert!(out.completed && !out.fallback_used);
        assert!(!b.has_retained_state());
        assert_eq!(out.value_vectors, vec![[0.0; 4]]);

        let mut f = PersistencePlanner::new(ArmId::F, None, None, None, None, None).unwrap();
        let out = f.act("epoch:a", &[3, 5], Some("case-001"), None).unwrap();
        assert_eq!((out.model_calls, out.exact_transitions), (32, 128));
        assert!(!f.has_retained_state());

        let mut r = PersistencePlanner::new(ArmId::R, None, None, None, None, None).unwrap();
        r.act("epoch:a", &[3, 5], Some("case-001"), None).unwrap();
        assert!(r.has_retained_state());
        let mut p = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        p.act("epoch:a", &[3, 5], Some("case-001"), None).unwrap();
        assert!(p.has_retained_state());
        let mut c = PersistencePlanner::new(ArmId::C, None, None, None, None, None).unwrap();
        assert_eq!(c.deadline_ms, 7000);
        c.act("epoch:a", &[3, 5], Some("case-001"), None).unwrap();
        assert!(!c.has_retained_state());

        // Empty legal fails closed (EmptyLegal, never a retry).
        assert!(matches!(
            b.act("epoch:a", &[], None, None),
            Err(SearchError::EmptyLegal)
        ));
        // Deadline pressure falls back immediately (candidate-0 path).
        let out = b.act("epoch:a", &[3, 5], None, Some(100.0)).unwrap();
        assert!(out.fallback_used && out.timeout && !out.completed);
    }

    /// Ponder: B/F/R/C are no-ops; P charges 2/child (4 total over 2
    /// children); R with nonzero ponder raises; quota distributes
    /// round-robin.
    #[test]
    fn ponder_semantics() {
        let mut r = PersistencePlanner::new(ArmId::R, None, None, None, None, None).unwrap();
        r.act("epoch:a", &[3, 5], None, None).unwrap();
        r.ponder(100.0, None).unwrap();
        assert_eq!(r.ponder_budget_used, 0);

        let mut p = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        p.act("epoch:a", &[3, 5], None, None).unwrap();
        p.ponder(100.0, None).unwrap();
        assert_eq!(p.ponder_budget_used, 4);
        p.ponder(100.0, Some(3)).unwrap();
        assert_eq!(p.ponder_budget_used, 7);
        assert!(p.ponder(-1.0, None).is_ok());

        let mut f = PersistencePlanner::new(ArmId::F, None, None, None, None, None).unwrap();
        f.act("epoch:a", &[3, 5], None, None).unwrap();
        f.ponder(100.0, None).unwrap();
        assert_eq!(f.ponder_budget_used, 0);
        assert!(p.ponder(100.0, Some(0)).is_err());
    }

    /// Observe: B/F/C log `fresh`; R/P hit verifies rebuild and counts
    /// hit; unknown packet logs `miss_recovery` (+miss/recovery);
    /// mismatched `epoch_after` raises; empty id raises.
    #[test]
    fn observe_commit_paths() {
        let mut b = PersistencePlanner::new(ArmId::B, None, None, None, None, None).unwrap();
        b.observe("pid-x", "epoch:y").unwrap();
        assert_eq!(b.commit_log[0].outcome, "fresh");
        assert!(b.observe("", "epoch:y").is_err());

        let mut p = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        p.act("epoch:a", &[3, 5], None, None).unwrap();
        let packet_id = p
            .forest
            .as_ref()
            .unwrap()
            .children
            .keys()
            .next()
            .unwrap()
            .clone();
        let packet = p.forest.as_ref().unwrap().children[&packet_id].clone();
        p.observe(&packet_id, &packet.epoch_after).unwrap();
        assert_eq!(p.commit_log[0].outcome, "hit");
        assert_eq!(p.surprise_count("hit"), 1);
        assert_eq!(p.forest.as_ref().unwrap().children.len(), 1);

        let mut p2 = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        p2.act("epoch:a", &[3, 5], None, None).unwrap();
        p2.observe("pid-unknown", "epoch:zzz").unwrap();
        assert_eq!(p2.commit_log[0].outcome, "miss_recovery");
        assert_eq!(p2.surprise_count("miss"), 1);
        assert!(!p2.has_retained_state());

        // No forest (fallback path) -> rebuild_no_forest.
        let mut r = PersistencePlanner::new(ArmId::R, None, None, None, None, None).unwrap();
        r.observe("pid-x", "epoch:y").unwrap();
        assert_eq!(r.commit_log[0].outcome, "rebuild_no_forest");

        // Tampered epoch_after raises (commit/rebuild mismatch).
        let mut p3 = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        p3.act("epoch:a", &[3, 5], None, None).unwrap();
        let packet_id = p3
            .forest
            .as_ref()
            .unwrap()
            .children
            .keys()
            .next()
            .unwrap()
            .clone();
        assert!(p3.observe(&packet_id, "epoch:tampered").is_err());
    }

    /// Epochs are deterministic per (observation, arm); picks are
    /// deterministic and arm-discriminated; empty legal raises.
    #[test]
    fn epoch_and_pick_determinism() {
        let p = PersistencePlanner::new(ArmId::P, None, None, None, None, None).unwrap();
        let f = PersistencePlanner::new(ArmId::F, None, None, None, None, None).unwrap();
        assert_eq!(p.epoch_for_obs("sha256:abc"), p.epoch_for_obs("sha256:abc"));
        assert_ne!(p.epoch_for_obs("sha256:abc"), f.epoch_for_obs("sha256:abc"));
        let legal = vec![1, 2, 3, 4, 5];
        assert_eq!(
            p.pick_action(&legal, Some("c")),
            p.pick_action(&legal, Some("c"))
        );
        assert!(p.pick_action(&[], None).is_err());
    }
}
