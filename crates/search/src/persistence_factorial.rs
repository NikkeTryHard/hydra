//! Persistence factorial umbrella — public surface aliases (EvalControl-owned).
//!
//! Rust port of `persistence_factorial.py` (71 ln): the re-export facade
//! over the split modules — `persistence_kernel` (arms, packet kernel,
//! forest state), `persistence_spec` (per-arm candidate-spec factory),
//! `persistence_planner` (per-arm state machine + [`PersistencePlanner`]),
//! and `persistence_report` (frozen whole-block report). Import from this
//! path; the split modules stay the owners (mirrors `:13-61` `__all__`).
//!
//! B4: per-arm `selected + vectors + counters` live in
//! `persistence_planner::ActOutcome`; `deterministic_gumbel_for_arm`
//! parity lives in `persistence_spec`.

pub use crate::persistence_kernel::{
    ACTION_KIND_ORDINALS, ArmId, BeliefEpochLite, DEPLOYABLE_DEADLINE_MS, FinitePacket,
    ForestState, PersistenceArm, action_key, arm_def, check_arm, commit_equals_rebuild,
    compute_packet_id, enumerate_packets_for, fresh_rebuild_epoch, make_persistence_arm,
};
pub use crate::persistence_planner::{ActOutcome, CommitEntry, PersistencePlanner};
pub use crate::persistence_report::{
    ArmStrata, CONTRAST_PAIRS, FactorialContrasts, FactorialReport, ResourceSample, block_diffs,
    factorial_contrasts, generate_factorial_report, persistence_contrast_seed,
    stratify_surprise_miss_recovery,
};
pub use crate::persistence_spec::{
    ArmBudget, GUMBEL_SEED_DEFAULT, GUMBEL_SEED_PLANNER, PersistenceCandidateSpec, arm_budget,
    deterministic_gumbel_for_arm, effective_deadline_ms, make_persistence_candidate_spec,
    spec_parameters_digest, validate_deadline_and_fallback,
};

/// One arm definition row: `(retain_state, opponent_time_compute,
/// own_deadline_ms, extra_wait_allowance_ms, deployable)`.
pub type ArmDefRow = (ArmId, (bool, bool, u64, u64, bool));

/// Arm definition table (`ARM_DEFS`, `persistence_kernel.py:202-243`):
/// `(retain_state, opponent_time_compute, own_deadline_ms,
/// extra_wait_allowance_ms, deployable)` per arm, in B/F/R/P/C order.
pub fn arm_defs() -> [ArmDefRow; 5] {
    [
        (ArmId::B, (false, false, DEPLOYABLE_DEADLINE_MS, 0, true)),
        (ArmId::F, (false, false, DEPLOYABLE_DEADLINE_MS, 0, true)),
        (ArmId::R, (true, false, DEPLOYABLE_DEADLINE_MS, 0, true)),
        (ArmId::P, (true, true, DEPLOYABLE_DEADLINE_MS, 0, true)),
        (
            ArmId::C,
            (false, false, DEPLOYABLE_DEADLINE_MS, 2000, false),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Facade surface: every `:41-61` `__all__` name resolves through
    /// this path (B4 family ships as one surface).
    #[test]
    fn facade_surface_resolves() {
        for (id, (retain, opp, deadline, allowance, deployable)) in arm_defs() {
            let arm = make_persistence_arm(id);
            assert_eq!(arm.retain_state, retain);
            assert_eq!(arm.opponent_time_compute, opp);
            assert_eq!(arm.own_deadline_ms, deadline);
            assert_eq!(arm.extra_wait_allowance_ms, allowance);
            assert_eq!(arm.deployable, deployable);
            assert!(check_arm(&arm).is_ok());
        }
        let planner = PersistencePlanner::new(ArmId::F, None, None, None, None, None).unwrap();
        assert_eq!(planner.max_model_calls, 32);
        let spec = make_persistence_candidate_spec(ArmId::B, None, None, None, None).unwrap();
        assert_eq!(spec.candidate_id, "persistence-B");
        assert!(
            spec_parameters_digest(&spec)
                .unwrap()
                .starts_with("sha256:")
        );
        let packets = enumerate_packets_for("epoch:facade", 1, 2).unwrap();
        assert!(commit_equals_rebuild("epoch:facade", &packets[0]).unwrap());
        assert_eq!(
            compute_packet_id("epoch:facade", 1, 0).unwrap().len(),
            7 + 64
        );
        assert_eq!(
            fresh_rebuild_epoch("epoch:facade", &packets[1])
                .unwrap()
                .len(),
            6 + 16
        );
        let _ = BeliefEpochLite {
            epoch: "e".to_string(),
            observation_hash: format!("sha256:{}", "a".repeat(64)),
            target_id: format!("sha256:{}", "b".repeat(64)),
            root_actor: 0,
        };
        let _ = FinitePacket {
            packet_id: packets[0].packet_id.clone(),
            action_id: 1,
            epoch_before: "epoch:facade".to_string(),
            epoch_after: packets[0].epoch_after.clone(),
            probability: 0.7,
            delta: vec![1, 0],
        };
        let _ = ForestState::default();
        let _ = ActOutcome {
            selected: 1,
            candidate_actions: vec![1],
            value_vectors: vec![[0.0; 4]],
            model_calls: 1,
            exact_transitions: 0,
            fallback_used: false,
            timeout: false,
            completed: true,
        };
        let _ = CommitEntry {
            arm: ArmId::P,
            packet_id: "p".to_string(),
            outcome: "hit".to_string(),
            ponder_calls: 0,
        };
        let _: ArmBudget = arm_budget(ArmId::R, None, None);
        let _: u64 = effective_deadline_ms(&make_persistence_arm(ArmId::C), 5000).unwrap();
        let _: f64 = deterministic_gumbel_for_arm(ArmId::F, "c", 0, GUMBEL_SEED_DEFAULT);
        let _: f64 = deterministic_gumbel_for_arm(ArmId::F, "c", 0, GUMBEL_SEED_PLANNER);
        assert_eq!(CONTRAST_PAIRS.len(), 4);
        assert_eq!(ACTION_KIND_ORDINALS.len(), 13);
    }
}
