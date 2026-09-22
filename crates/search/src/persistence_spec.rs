//! Persistence spec factory — per-arm candidate specs (EvalControl-owned).
//!
//! Rust port of `persistence_spec.py` (193 ln): the per-arm
//! `CandidateSpec` factory, the deadline/fallback validator, and the
//! deterministic choice helpers. B3 canon-wins: choice digests use raw
//! `sha2` over explicit preimages (the single identity hasher) — `_GUMBEL`
//! parity below documents the exact preimage per line so a fork is
//! review-visible.
//!
//! `deterministic_gumbel_for_arm` (`:185-193`): `u64BE(sha256(seed +
//! ":arm:case:action")[:8]) / 2^64` in `[0,1)` — no RNG call-order
//! dependence. The seed default is `b"hydra2-persistence-v1"`; the
//! planner passes `b"persistence-factorial-v1"` — both pinned as
//! distinct goldens below (they MUST NOT be unified).

use crate::SearchError;

use crate::eval::{hex_of, sha256_bytes};
use crate::persistence_kernel::{ArmId, PersistenceArm, make_persistence_arm};

/// Default choice seed (`:186`).
pub const GUMBEL_SEED_DEFAULT: &[u8] = b"hydra2-persistence-v1";

/// Planner choice seed (`persistence_planner.py:77`).
pub const GUMBEL_SEED_PLANNER: &[u8] = b"persistence-factorial-v1";

/// Deterministic scalar in `[0,1)` from `(arm, case, action)`
/// (`:185-193`): `u64BE(sha256(seed + ":arm:case:action")[:8]) / 2^64`.
pub fn deterministic_gumbel_for_arm(arm: ArmId, case_id: &str, action_id: u64, seed: &[u8]) -> f64 {
    let mut preimage =
        Vec::with_capacity(seed.len() + 1 + arm.name().len() + 1 + case_id.len() + 1 + 20);
    preimage.extend_from_slice(seed);
    preimage.extend_from_slice(format!(":{}:{case_id}:{action_id}", arm.name()).as_bytes());
    let digest = sha256_bytes(&preimage);
    let mut n = 0u64;
    for b in &digest[..8] {
        n = (n << 8) | u64::from(*b);
    }
    // proof: u64 word / 2^64 is in [0,1); 1ulp on the unit interval is oracle-tolerable.
    #[allow(clippy::cast_precision_loss)]
    let f: f64 = n as f64;
    f / 18_446_744_073_709_551_616.0
}

/// Per-arm call/transition budgets (`make_persistence_candidate_spec`,
/// `:109-117`): B = 1/0, others = 32/128, C laboratory = 64/256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmBudget {
    /// Max model calls.
    pub max_model_calls: u64,
    /// Max exact transitions.
    pub max_transitions: u64,
}

/// Budget for an arm (explicit overrides pass through; `None` = default).
pub fn arm_budget(
    id: ArmId,
    max_model_calls: Option<u64>,
    max_transitions: Option<u64>,
) -> ArmBudget {
    let (dflt_calls, dflt_trans) = match id {
        ArmId::B => (1, 0),
        ArmId::C => (64, 256),
        ArmId::F | ArmId::R | ArmId::P => (32, 128),
    };
    ArmBudget {
        max_model_calls: max_model_calls.unwrap_or(dflt_calls),
        max_transitions: max_transitions.unwrap_or(dflt_trans),
    }
}

/// Effective deadline (`:118-121, :170-177`): C adds its extra allowance;
/// deployable arms pin `<= 5000`; fallback margin must sit in
/// `[0, deadline)`.
pub fn effective_deadline_ms(arm: &PersistenceArm, deadline_ms: u64) -> Result<u64, SearchError> {
    if deadline_ms == 0 {
        return Err(SearchError::InvalidArg {
            detail: "deadline_ms must be positive",
        });
    }
    match arm.id {
        ArmId::B | ArmId::F | ArmId::R | ArmId::P => {
            if deadline_ms > crate::persistence_kernel::DEPLOYABLE_DEADLINE_MS {
                return Err(SearchError::InvalidArg {
                    detail: "deployable arm deadline over 5000",
                });
            }
            Ok(deadline_ms)
        }
        ArmId::C => Ok(deadline_ms + arm.extra_wait_allowance_ms),
    }
}

/// Validate deadline + fallback margin (`validate_deadline_and_fallback`,
/// `:164-177`).
pub fn validate_deadline_and_fallback(
    arm: &PersistenceArm,
    deadline_ms: u64,
    fallback_margin_ms: u64,
) -> Result<(), SearchError> {
    if fallback_margin_ms >= deadline_ms {
        return Err(SearchError::InvalidArg {
            detail: "fallback_margin must be in [0, deadline)",
        });
    }
    if matches!(arm.id, ArmId::B | ArmId::F | ArmId::R | ArmId::P)
        && deadline_ms > crate::persistence_kernel::DEPLOYABLE_DEADLINE_MS
    {
        return Err(SearchError::InvalidArg {
            detail: "deployable arm deadline over 5000",
        });
    }
    if arm.id == ArmId::C && !arm.deployable && arm.extra_wait_allowance_ms == 0 {
        return Err(SearchError::InvalidArg {
            detail: "C extra allowance must be positive",
        });
    }
    Ok(())
}

/// Candidate-spec identity material for a persistence arm
/// (`make_persistence_candidate_spec`, `:83-161` minus IO): the
/// deterministic parameter projection (`:149-159`) + budget + deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceCandidateSpec {
    /// `persistence-{arm}`.
    pub candidate_id: String,
    /// Always `persistence_factorial`.
    pub algorithm: String,
    /// Always `1.0.0`.
    pub algorithm_version: String,
    /// Arm under test.
    pub arm: PersistenceArm,
    /// Call/transition budget.
    pub budget: ArmBudget,
    /// Effective deadline (C includes its allowance).
    pub deadline_ms: u64,
    /// Fallback margin (default 500).
    pub fallback_margin_ms: u64,
}

/// Build the candidate-spec material for an arm (`:83-161`, IO-free:
/// file-derived `_default_hashes` stay Python-side; digests bind at
/// commit via `candidate_spec_hash`).
pub fn make_persistence_candidate_spec(
    id: ArmId,
    deadline_ms: Option<u64>,
    fallback_margin_ms: Option<u64>,
    max_model_calls: Option<u64>,
    max_transitions: Option<u64>,
) -> Result<PersistenceCandidateSpec, SearchError> {
    let arm = make_persistence_arm(id);
    let deadline = deadline_ms.unwrap_or(arm.own_deadline_ms);
    let margin = fallback_margin_ms.unwrap_or(500);
    validate_deadline_and_fallback(&arm, deadline, margin)?;
    Ok(PersistenceCandidateSpec {
        candidate_id: format!("persistence-{}", arm.id),
        algorithm: "persistence_factorial".to_string(),
        algorithm_version: "1.0.0".to_string(),
        arm,
        budget: arm_budget(id, max_model_calls, max_transitions),
        deadline_ms: effective_deadline_ms(&arm, deadline)?,
        fallback_margin_ms: margin,
    })
}

/// Digest of the deterministic parameter projection (`:149-159` shape):
/// `{persistence_arm, retain_state, opponent_time_compute, deployable,
/// own_deadline_ms, extra_wait_allowance_ms}` over canon bytes.
pub fn spec_parameters_digest(spec: &PersistenceCandidateSpec) -> Result<String, SearchError> {
    let payload = serde_json::json!({
        "persistence_arm": spec.arm.id.name(),
        "retain_state": spec.arm.retain_state,
        "opponent_time_compute": spec.arm.opponent_time_compute,
        "deployable": spec.arm.deployable,
        "own_deadline_ms": spec.arm.own_deadline_ms,
        "extra_wait_allowance_ms": spec.arm.extra_wait_allowance_ms,
    });
    let bytes = crate::eval::canon_bytes_of(&payload, "persistence:spec:params")?;
    Ok(format!("sha256:{}", hex_of(&sha256_bytes(&bytes))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GUMBEL goldens (`:186` default seed): exact `[0,1)` scalars per
    /// (arm, case, action). A preimage fork (separator, order, encoding)
    /// fails all four.
    #[test]
    fn deterministic_gumbel_parity_golden() {
        assert_eq!(
            deterministic_gumbel_for_arm(ArmId::F, "case-001", 3, GUMBEL_SEED_DEFAULT),
            0.6055335298833656
        );
        assert_eq!(
            deterministic_gumbel_for_arm(ArmId::P, "case-001", 3, GUMBEL_SEED_DEFAULT),
            0.9460496117262222
        );
        assert_eq!(
            deterministic_gumbel_for_arm(ArmId::B, "default", 0, GUMBEL_SEED_DEFAULT),
            0.7000227248981264
        );
        assert_eq!(
            deterministic_gumbel_for_arm(ArmId::C, "block-7", 11, GUMBEL_SEED_DEFAULT),
            0.577475245085178
        );
        // Two seeds MUST NOT unify: planner seed diverges on same inputs.
        assert_ne!(
            deterministic_gumbel_for_arm(ArmId::F, "case-001", 3, GUMBEL_SEED_PLANNER),
            deterministic_gumbel_for_arm(ArmId::F, "case-001", 3, GUMBEL_SEED_DEFAULT)
        );
        // Arm/case/action all discriminate.
        assert_ne!(
            deterministic_gumbel_for_arm(ArmId::F, "case-001", 3, GUMBEL_SEED_DEFAULT),
            deterministic_gumbel_for_arm(ArmId::F, "case-001", 4, GUMBEL_SEED_DEFAULT)
        );
    }

    /// Budgets: B = 1/0 (frozen), F/R/P = 32/128, C = 64/256
    /// laboratory; explicit overrides pass through.
    #[test]
    fn arm_budgets() {
        assert_eq!(
            arm_budget(ArmId::B, None, None),
            ArmBudget {
                max_model_calls: 1,
                max_transitions: 0
            }
        );
        assert_eq!(
            arm_budget(ArmId::F, None, None),
            ArmBudget {
                max_model_calls: 32,
                max_transitions: 128
            }
        );
        assert_eq!(
            arm_budget(ArmId::C, None, None),
            ArmBudget {
                max_model_calls: 64,
                max_transitions: 256
            }
        );
        assert_eq!(
            arm_budget(ArmId::P, Some(7), Some(9)),
            ArmBudget {
                max_model_calls: 7,
                max_transitions: 9
            }
        );
    }

    /// Deadlines: deployable pins at 5000, C extends by its allowance;
    /// margin must sit inside `[0, deadline)`; C needs its allowance.
    #[test]
    fn deadline_and_fallback_gates() {
        let b = make_persistence_arm(ArmId::B);
        assert!(validate_deadline_and_fallback(&b, 5000, 500).is_ok());
        assert!(validate_deadline_and_fallback(&b, 5001, 500).is_err());
        assert!(validate_deadline_and_fallback(&b, 5000, 5000).is_err());
        let c = make_persistence_arm(ArmId::C);
        assert_eq!(effective_deadline_ms(&c, 5000).unwrap(), 7000);
        assert_eq!(effective_deadline_ms(&b, 5000).unwrap(), 5000);
        let spec = make_persistence_candidate_spec(ArmId::P, None, None, None, None).unwrap();
        assert_eq!(spec.candidate_id, "persistence-P");
        assert_eq!(spec.fallback_margin_ms, 500);
        assert_eq!(spec.budget.max_model_calls, 32);
        assert!(
            spec_parameters_digest(&spec)
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(make_persistence_candidate_spec(ArmId::B, Some(6000), None, None, None).is_err());
    }
}
