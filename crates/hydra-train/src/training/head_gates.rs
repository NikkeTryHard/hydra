//! Head activation discipline: density, interference, and warmup gates.
//!
//! Prevents sparse or noisy advanced heads from dragging the shared SE-ResNet
//! trunk backward via negative transfer. Implements the archive's gate pack
//! from `answer_13_combined.md` sections 3.3, 5, and 6:
//!
//! - **Density gate**: Per-head label density `rho_h` for dense heads
//!   (threshold: `rho >= 0.8`) and samples-per-param `spp_h` for sparse
//!   search-derived heads (threshold: `spp >= 5.0`).
//!
//! - **Gradient conflict gate**: Shared-trunk gradient cosine between each
//!   auxiliary head loss and the policy+value loss. Heads are kept off if
//!   cosine is negative on >30% of checks after warmup.
//!
//! - **Warmup protocol**: When activating a head, train head-only (trunk
//!   frozen) for a configurable number of steps before unfreezing. Transition
//!   to full activation only if gradient conflict gate passes.
//!
//! # Gate sequence
//!
//! 1. Target correctness audit (manual prerequisite, not automated here).
//! 2. Density gate: `rho_h >= min_dense_rho` or `spp_h >= min_sparse_spp`.
//! 3. Head-only warmup with trunk frozen for `warmup_steps` updates.
//! 4. Gradient conflict gate: negative cosine fraction < `max_negative_frac`.
//! 5. Feature-ablation gate (requires evaluation infrastructure, documented
//!    but not automated here).
//!
//! The controller manages per-head state transitions:
//! `Off` -> (density passes) -> `Warmup` -> (conflict passes) -> `Active`.
//!
//! # Integration
//!
//! The caller (orchestrator) is responsible for:
//! - Calling [`extract_target_presence`] and
//!   [`HeadActivationController::record_batch`] each training step.
//! - Periodically computing shared-trunk gradient cosine (see
//!   [`grad_cosine_from_flat`]) and calling
//!   [`HeadActivationController::record_grad_cosine`].
//! - Using [`HeadActivationController::approved_loss_config`] to get effective
//!   loss weights (unapproved heads are zeroed out).
//! - Checking [`HeadActivationController::warmup_heads`] and detaching trunk
//!   outputs for heads in warmup state so they train head-only.
//!
//! # Important: do not use `grad_norm_approx` for the conflict gate
//!
//! The existing `grad_norm_approx` in `losses.rs` is a loss-magnitude proxy,
//! not a true parameter-gradient norm. Use [`grad_cosine_from_flat`] with
//! real flattened shared-trunk gradients instead.

use crate::training::losses::{HydraLossConfig, HydraTargets};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

use burn::prelude::*;
pub use hydra_train_types::head_gates::{
    AdvancedHead, DEFAULT_MAX_NEGATIVE_FRAC, DEFAULT_MIN_CONFLICT_CHECKS, DEFAULT_MIN_DENSE_RHO,
    DEFAULT_MIN_EVAL_SAMPLES, DEFAULT_MIN_SPARSE_SPP, DEFAULT_WARMUP_STEPS, GradConflictTracker,
    HeadActivationConfig, HeadCoverage, HeadGateReport, HeadKind, HeadState, NUM_ADVANCED_HEADS,
    TargetPresence, grad_cosine_from_flat,
};

/// Head activation controller with Hydra loss-config adapter methods.
#[derive(Clone, Debug)]
pub struct HeadActivationController(hydra_train_types::head_gates::HeadActivationController);

impl HeadActivationController {
    /// Creates a controller with all heads in [`HeadState::Off`].
    pub fn new(config: HeadActivationConfig) -> Self {
        Self(hydra_train_types::head_gates::HeadActivationController::new(config))
    }

    /// Returns a [`HydraLossConfig`] with unapproved heads zeroed out.
    pub fn approved_loss_config(&self, base: &HydraLossConfig) -> HydraLossConfig {
        approved_loss_config(self, base)
    }
}

impl Deref for HeadActivationController {
    type Target = hydra_train_types::head_gates::HeadActivationController;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HeadActivationController {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
/// Extracts per-head target presence from a batch of [`HydraTargets`].
///
/// For targets with per-sample masks (`belief_fields`, `mixture_weight`),
/// counts the number of samples where the mask is nonzero. For targets
/// without per-sample masks, counts `batch_size` when the target is present.
pub fn extract_target_presence<B: Backend>(targets: &HydraTargets<B>) -> TargetPresence {
    if let Some(presence) = targets.target_presence {
        return presence;
    }
    let batch_size = targets.policy_target.dims()[0];
    let mut counts = [0usize; NUM_ADVANCED_HEADS];

    // Oracle critic: uses oracle_guidance_mask for per-sample gating.
    if targets.oracle_target.is_some() {
        counts[AdvancedHead::OracleCritic.index()] = match &targets.oracle_guidance_mask {
            Some(mask) => count_nonzero_1d(mask),
            None => batch_size,
        };
    }

    // Belief fields: per-sample mask.
    if targets.belief_fields_target.is_some() {
        counts[AdvancedHead::BeliefFields.index()] = match &targets.belief_fields_mask {
            Some(mask) => {
                count_nonzero_1d_with_optional_gate(mask, targets.oracle_guidance_mask.as_ref())
            }
            None => batch_size,
        };
    }

    // Mixture weight: per-sample mask.
    if targets.mixture_weight_target.is_some() {
        counts[AdvancedHead::MixtureWeight.index()] = match &targets.mixture_weight_mask {
            Some(mask) => {
                count_nonzero_1d_with_optional_gate(mask, targets.oracle_guidance_mask.as_ref())
            }
            None => batch_size,
        };
    }

    // Opponent hand type: shares oracle_guidance_mask.
    if targets.opponent_hand_type_target.is_some() {
        counts[AdvancedHead::OpponentHandType.index()] = match &targets.oracle_guidance_mask {
            Some(mask) => count_nonzero_1d(mask),
            None => batch_size,
        };
    }

    counts[AdvancedHead::DeltaQ.index()] = match (&targets.delta_q_target, &targets.delta_q_mask) {
        (Some(_), Some(mask)) => count_nonzero_rows_2d(mask),
        _ => 0,
    };

    counts[AdvancedHead::SafetyResidual.index()] = match (
        &targets.safety_residual_target,
        &targets.safety_residual_mask,
    ) {
        (Some(_), Some(mask)) => count_nonzero_rows_2d(mask),
        _ => 0,
    };

    TargetPresence {
        counts,
        delta_q_actions_present: 0,
        batch_size,
    }
}

pub fn borrow_or_extract_target_presence<B: Backend>(
    targets: &HydraTargets<B>,
) -> Cow<'_, TargetPresence> {
    if let Some(presence) = targets.target_presence.as_ref() {
        Cow::Borrowed(presence)
    } else {
        Cow::Owned(extract_target_presence(targets))
    }
}

/// Counts nonzero entries in a 1-D tensor.
fn count_nonzero_1d<B: Backend>(tensor: &Tensor<B, 1>) -> usize {
    match tensor.to_data().convert::<f32>().as_slice::<f32>() {
        Ok(data) => data.iter().filter(|&&v| v > 0.0).count(),
        Err(_) => 0,
    }
}

fn count_nonzero_1d_with_optional_gate<B: Backend>(
    tensor: &Tensor<B, 1>,
    gate: Option<&Tensor<B, 1>>,
) -> usize {
    let tensor_data = tensor.to_data().convert::<f32>();
    let Ok(data) = tensor_data.as_slice::<f32>() else {
        return 0;
    };
    let gate_data = gate.map(|gate| gate.to_data().convert::<f32>());
    let gate_slice = gate_data
        .as_ref()
        .and_then(|data| data.as_slice::<f32>().ok());
    data.iter()
        .enumerate()
        .filter(|(idx, value)| {
            **value > 0.0
                && gate_slice.is_none_or(|gate| gate.get(*idx).copied().unwrap_or(0.0) > 0.0)
        })
        .count()
}

fn count_nonzero_rows_2d<B: Backend>(tensor: &Tensor<B, 2>) -> usize {
    let [_rows, cols] = tensor.dims();
    match tensor.to_data().convert::<f32>().as_slice::<f32>() {
        Ok(data) => data
            .chunks(cols)
            .filter(|row| row.iter().any(|&v| v > 0.0))
            .count(),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Loss config integration
// ---------------------------------------------------------------------------
/// Returns a [`HydraLossConfig`] with unapproved heads zeroed out.
///
/// - [`HeadState::Off`] heads get weight `0.0`.
/// - [`HeadState::Warmup`] and [`HeadState::Active`] heads keep
///   their weight from `base`.
///
/// Baseline heads (policy, value, grp, tenpai, danger, opp_next, score)
/// are always passed through unchanged.
pub fn approved_loss_config(
    controller: &HeadActivationController,
    base: &HydraLossConfig,
) -> HydraLossConfig {
    let gate = |head: AdvancedHead, w: f32| -> f32 {
        match controller.head_state(head) {
            HeadState::Off => 0.0,
            HeadState::Warmup | HeadState::Active => w,
        }
    };
    HydraLossConfig::new()
        .with_w_pi(base.w_pi)
        .with_w_v(base.w_v)
        .with_w_grp(base.w_grp)
        .with_w_tenpai(base.w_tenpai)
        .with_w_danger(base.w_danger)
        .with_w_opp(base.w_opp)
        .with_w_score(base.w_score)
        .with_w_oracle_critic(gate(AdvancedHead::OracleCritic, base.w_oracle_critic))
        .with_w_belief_fields(gate(AdvancedHead::BeliefFields, base.w_belief_fields))
        .with_w_mixture_weight(gate(AdvancedHead::MixtureWeight, base.w_mixture_weight))
        .with_w_opponent_hand_type(gate(
            AdvancedHead::OpponentHandType,
            base.w_opponent_hand_type,
        ))
        .with_w_delta_q(gate(AdvancedHead::DeltaQ, base.w_delta_q))
        .with_w_safety_residual(gate(AdvancedHead::SafetyResidual, base.w_safety_residual))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    // -- AdvancedHead ------------------------------------------------------

    #[test]
    fn head_indices_are_unique_and_complete() {
        let mut seen = [false; NUM_ADVANCED_HEADS];
        for &head in &AdvancedHead::ALL {
            let idx = head.index();
            assert!(idx < NUM_ADVANCED_HEADS, "index out of range: {idx}");
            assert!(!seen[idx], "duplicate index: {idx}");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&v| v), "not all indices covered");
    }

    #[test]
    fn head_kind_classification() {
        assert_eq!(AdvancedHead::DeltaQ.kind(), HeadKind::SparseSearch);
        assert_eq!(AdvancedHead::SafetyResidual.kind(), HeadKind::Dense);
        assert_eq!(AdvancedHead::OracleCritic.kind(), HeadKind::Dense);
        assert_eq!(AdvancedHead::BeliefFields.kind(), HeadKind::Dense);
        assert_eq!(AdvancedHead::MixtureWeight.kind(), HeadKind::Dense);
        assert_eq!(AdvancedHead::OpponentHandType.kind(), HeadKind::Dense);
    }

    // -- HeadCoverage ------------------------------------------------------

    #[test]
    fn coverage_starts_empty() {
        let cov = HeadCoverage::new();
        assert_eq!(cov.total_samples(), 0);
        for &head in &AdvancedHead::ALL {
            assert_eq!(cov.rho(head), 0.0);
            assert_eq!(cov.labeled_samples(head), 0);
        }
    }

    #[test]
    fn coverage_tracks_density() {
        let mut cov = HeadCoverage::new();

        // Batch 1: 10 samples, safety_residual present for all.
        let mut p1 = TargetPresence::with_batch_size(10);
        p1.counts[AdvancedHead::SafetyResidual.index()] = 10;
        cov.record_batch(&p1);
        assert!((cov.rho(AdvancedHead::SafetyResidual) - 1.0).abs() < 1e-6);

        // Batch 2: 10 samples, safety_residual absent.
        let p2 = TargetPresence::with_batch_size(10);
        cov.record_batch(&p2);
        assert!((cov.rho(AdvancedHead::SafetyResidual) - 0.5).abs() < 1e-6);
        assert_eq!(cov.total_samples(), 20);
        assert_eq!(cov.labeled_samples(AdvancedHead::SafetyResidual), 10);
    }

    #[test]
    fn coverage_partial_mask() {
        let mut cov = HeadCoverage::new();
        let mut p = TargetPresence::with_batch_size(100);
        p.counts[AdvancedHead::BeliefFields.index()] = 60; // 60% have mask
        cov.record_batch(&p);
        assert!((cov.rho(AdvancedHead::BeliefFields) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn coverage_spp_computation() {
        let mut cov = HeadCoverage::new();
        // 100 batches of 100, delta_q present for 5 each.
        let mut p = TargetPresence::with_batch_size(100);
        p.counts[AdvancedHead::DeltaQ.index()] = 5;
        for _ in 0..100 {
            cov.record_batch(&p);
        }
        // 500 labeled / 1M params = 0.0005
        let spp = cov.spp(AdvancedHead::DeltaQ, 1_000_000);
        assert!((spp - 0.0005).abs() < 1e-7);
    }

    #[test]
    fn coverage_spp_zero_params() {
        let cov = HeadCoverage::new();
        assert_eq!(cov.spp(AdvancedHead::DeltaQ, 0), 0.0);
    }

    // -- GradConflictTracker -----------------------------------------------

    #[test]
    fn conflict_starts_clean() {
        let tracker = GradConflictTracker::new();
        for &head in &AdvancedHead::ALL {
            assert_eq!(tracker.negative_fraction(head), 0.0);
            assert_eq!(tracker.total_checks(head), 0);
            assert!(!tracker.is_conflicting(head, 0.3));
        }
    }

    #[test]
    fn conflict_tracks_negative_fraction() {
        let mut tracker = GradConflictTracker::new();
        let head = AdvancedHead::SafetyResidual;
        tracker.record(head, 0.5); // positive
        tracker.record(head, -0.3); // negative
        tracker.record(head, 0.1); // positive
        tracker.record(head, -0.2); // negative
        // 2/4 = 0.5
        assert!((tracker.negative_fraction(head) - 0.5).abs() < 1e-6);
        assert_eq!(tracker.total_checks(head), 4);
        assert!(tracker.is_conflicting(head, 0.3)); // 0.5 > 0.3
        assert!(!tracker.is_conflicting(head, 0.6)); // 0.5 < 0.6
    }

    #[test]
    fn conflict_per_head_independence() {
        let mut tracker = GradConflictTracker::new();
        tracker.record(AdvancedHead::SafetyResidual, -0.5);
        tracker.record(AdvancedHead::OracleCritic, 0.9);
        assert!((tracker.negative_fraction(AdvancedHead::SafetyResidual) - 1.0).abs() < 1e-6);
        assert!(tracker.negative_fraction(AdvancedHead::OracleCritic).abs() < 1e-6);
    }

    // -- grad_cosine_from_flat ---------------------------------------------

    #[test]
    fn cosine_parallel_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        assert!((grad_cosine_from_flat(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!(grad_cosine_from_flat(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposing_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        assert!((grad_cosine_from_flat(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(grad_cosine_from_flat(&a, &b), 0.0);
        assert_eq!(grad_cosine_from_flat(&b, &a), 0.0);
    }

    #[test]
    fn cosine_empty_vectors() {
        assert_eq!(grad_cosine_from_flat(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "gradient vectors must have equal length")]
    fn cosine_mismatched_lengths_panics() {
        grad_cosine_from_flat(&[1.0, 2.0], &[1.0]);
    }

    // -- HeadActivationController ------------------------------------------

    fn test_config() -> HeadActivationConfig {
        HeadActivationConfig {
            min_dense_rho: 0.8,
            min_sparse_spp: 5.0,
            max_negative_frac: 0.3,
            warmup_steps: 3,
            learner_params: 1_000_000,
            min_eval_samples: 10,
            min_conflict_checks: 3,
        }
    }

    fn fill_density(ctrl: &mut HeadActivationController, head: AdvancedHead, rho: f32) {
        let batch_size = 10;
        let count = (rho * batch_size as f32).round() as usize;
        for _ in 0..100 {
            let mut p = TargetPresence::with_batch_size(batch_size);
            p.counts[head.index()] = count;
            ctrl.record_batch(&p);
        }
    }

    #[test]
    fn controller_all_off_by_default() {
        let ctrl = HeadActivationController::new(test_config());
        for &head in &AdvancedHead::ALL {
            assert_eq!(ctrl.head_state(head), HeadState::Off);
        }
        assert!(ctrl.warmup_heads().is_empty());
    }

    #[test]
    fn controller_blocks_activation_insufficient_samples() {
        let mut ctrl = HeadActivationController::new(test_config());
        // Only 5 samples, below min_eval_samples=10.
        let mut p = TargetPresence::with_batch_size(5);
        p.counts[AdvancedHead::SafetyResidual.index()] = 5;
        ctrl.record_batch(&p);

        let report = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert!(!report.approved);
        assert!(report.failures.contains(&"insufficient_samples"));
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Off
        );
    }

    #[test]
    fn controller_blocks_activation_low_density() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.5); // rho=0.5 < 0.8

        let report = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert!(!report.approved);
        assert!(report.failures.contains(&"density_rho_below_threshold"));
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Off
        );
    }

    #[test]
    fn controller_dense_head_enters_warmup() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);

        let report = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert!(report.approved);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );
        assert!(ctrl.warmup_heads().contains(&AdvancedHead::SafetyResidual));
    }

    #[test]
    fn controller_sparse_head_spp_gate() {
        let mut ctrl = HeadActivationController::new(test_config());
        // delta_q is SparseSearch. With 1M params, need 5M labeled samples.
        // Record very few labeled samples.
        for _ in 0..100 {
            let mut p = TargetPresence::with_batch_size(10);
            p.counts[AdvancedHead::DeltaQ.index()] = 1; // very sparse
            ctrl.record_batch(&p);
        }

        let report = ctrl.try_activate(AdvancedHead::DeltaQ);
        assert!(!report.approved);
        assert!(report.failures.contains(&"density_spp_below_threshold"));
    }

    #[test]
    fn controller_warmup_to_active() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );

        // Record positive gradient cosines.
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.5);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.3);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.1);

        // Tick through warmup (3 steps).
        ctrl.tick_warmup();
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );
        ctrl.tick_warmup();
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );
        ctrl.tick_warmup();
        // Warmup complete + conflict passes -> Active.
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Active
        );
        assert!(ctrl.warmup_heads().is_empty());
    }

    #[test]
    fn controller_warmup_conflict_reverts_to_off() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);

        // Record mostly negative gradient cosines.
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, -0.5);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, -0.3);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.1);
        // 2/3 = 0.67 > 0.3 threshold.

        // Complete warmup.
        for _ in 0..3 {
            ctrl.tick_warmup();
        }

        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Off
        );
    }

    #[test]
    fn controller_warmup_stays_if_insufficient_conflict_data() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);

        // Only 1 cosine check, need 3.
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.5);

        // Complete warmup countdown.
        for _ in 0..3 {
            ctrl.tick_warmup();
        }

        // Still in warmup: not enough conflict data to decide.
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );

        // Add more checks and tick again.
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.3);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.2);
        ctrl.tick_warmup();

        // Now conflict data is sufficient, and all positive -> Active.
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Active
        );
    }

    #[test]
    fn controller_try_activate_idempotent() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);

        // First activation -> Warmup.
        let r1 = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert!(r1.approved);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );

        // Second call -> no change, still Warmup.
        let r2 = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert_eq!(r2.state, HeadState::Warmup);
    }

    #[test]
    fn controller_force_off() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );

        ctrl.force_off(AdvancedHead::SafetyResidual);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Off
        );
    }

    // -- approved_loss_config ----------------------------------------------

    #[test]
    fn approved_config_zeros_off_heads() {
        let ctrl = HeadActivationController::new(test_config());
        let base = HydraLossConfig::new()
            .with_w_safety_residual(0.5)
            .with_w_oracle_critic(1.0)
            .with_w_delta_q(0.2);
        let gated = ctrl.approved_loss_config(&base);

        // All off -> all zero.
        assert_eq!(gated.w_safety_residual, 0.0);
        assert_eq!(gated.w_oracle_critic, 0.0);
        assert_eq!(gated.w_delta_q, 0.0);

        // Baseline unchanged.
        assert!((gated.w_pi - 1.0).abs() < 1e-6);
        assert!((gated.w_v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn approved_config_preserves_warmup_weights() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);

        let base = HydraLossConfig::new().with_w_safety_residual(0.5);
        let gated = ctrl.approved_loss_config(&base);
        assert!((gated.w_safety_residual - 0.5).abs() < 1e-6);
    }

    #[test]
    fn approved_config_preserves_active_weights() {
        let mut ctrl = HeadActivationController::new(test_config());
        fill_density(&mut ctrl, AdvancedHead::SafetyResidual, 0.9);
        ctrl.try_activate(AdvancedHead::SafetyResidual);

        // Complete warmup with positive cosines.
        for _ in 0..3 {
            ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.5);
        }
        for _ in 0..3 {
            ctrl.tick_warmup();
        }
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Active
        );

        let base = HydraLossConfig::new().with_w_safety_residual(0.5);
        let gated = ctrl.approved_loss_config(&base);
        assert!((gated.w_safety_residual - 0.5).abs() < 1e-6);
    }

    // -- extract_target_presence -------------------------------------------

    fn dummy_targets(batch: usize) -> HydraTargets<B> {
        let device = Default::default();
        HydraTargets {
            policy_target: Tensor::ones([batch, 46], &device) / 46.0,
            legal_mask: Tensor::ones([batch, 46], &device),
            value_target: Tensor::zeros([batch], &device),
            grp_target: Tensor::ones([batch, 24], &device) / 24.0,
            tenpai_target: Tensor::ones([batch, 3], &device) / 3.0,
            danger_target: Tensor::zeros([batch, 3, 34], &device),
            danger_mask: Tensor::ones([batch, 3, 34], &device),
            opp_next_target: Tensor::ones([batch, 3, 34], &device) / 34.0,
            score_pdf_target: Tensor::ones([batch, 64], &device) / 64.0,
            score_cdf_target: Tensor::zeros([batch, 64], &device),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: None,
            delta_q_mask: None,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
            target_presence: None,
        }
    }

    #[test]
    fn extract_presence_all_none() {
        let targets = dummy_targets(4);
        let presence = extract_target_presence(&targets);
        assert_eq!(presence.batch_size, 4);
        for &head in &AdvancedHead::ALL {
            assert_eq!(presence.count(head), 0);
        }
    }

    #[test]
    fn extract_presence_safety_residual() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.safety_residual_target = Some(Tensor::zeros([4, 46], &device));
        targets.safety_residual_mask = Some(Tensor::ones([4, 46], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::SafetyResidual), 4);
    }

    #[test]
    fn extract_presence_safety_residual_counts_only_nonzero_mask_rows() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.safety_residual_target = Some(Tensor::zeros([4, 46], &device));
        let mut mask = [[0.0f32; 46]; 4];
        mask[0][0] = 1.0;
        mask[2][7] = 1.0;
        targets.safety_residual_mask = Some(Tensor::from_floats(mask, &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::SafetyResidual), 2);
    }

    #[test]
    fn extract_presence_oracle_with_mask() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.oracle_target = Some(Tensor::ones([4, 4], &device));
        targets.oracle_guidance_mask = Some(Tensor::from_floats([1.0, 0.0, 1.0, 0.0], &device));

        let presence = extract_target_presence(&targets);
        // 2 of 4 samples have oracle mask = 1.
        assert_eq!(presence.count(AdvancedHead::OracleCritic), 2);
    }

    #[test]
    fn extract_presence_belief_with_mask() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.belief_fields_target = Some(Tensor::zeros([4, 4, 34], &device));
        targets.belief_fields_mask = Some(Tensor::from_floats([1.0, 1.0, 0.0, 1.0], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::BeliefFields), 3);
    }

    #[test]
    fn extract_presence_belief_respects_oracle_guidance_gate() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.belief_fields_target = Some(Tensor::zeros([4, 4, 34], &device));
        targets.belief_fields_mask = Some(Tensor::from_floats([1.0, 1.0, 0.0, 1.0], &device));
        targets.oracle_guidance_mask = Some(Tensor::from_floats([1.0, 0.0, 1.0, 0.0], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::BeliefFields), 1);
    }

    #[test]
    fn extract_presence_mixture_respects_oracle_guidance_gate() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.mixture_weight_target = Some(Tensor::zeros([4, 4], &device));
        targets.mixture_weight_mask = Some(Tensor::from_floats([1.0, 0.0, 1.0, 1.0], &device));
        targets.oracle_guidance_mask = Some(Tensor::from_floats([0.0, 1.0, 1.0, 0.0], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::MixtureWeight), 1);
    }

    #[test]
    fn extract_presence_delta_q_counts_only_nonzero_mask_rows() {
        let device = Default::default();
        let mut targets = dummy_targets(8);
        targets.delta_q_target = Some(Tensor::zeros([8, 46], &device));
        let mut mask = [[0.0f32; 46]; 8];
        mask[0][1] = 1.0;
        mask[2][3] = 1.0;
        mask[7][10] = 1.0;
        targets.delta_q_mask = Some(Tensor::from_floats(mask, &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::DeltaQ), 3);
    }

    #[test]
    fn extract_presence_delta_q_invalid_pair_counts_zero() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.delta_q_target = Some(Tensor::zeros([4, 46], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::DeltaQ), 0);
    }

    #[test]
    fn extract_presence_prefers_cached_metadata() {
        let device = Default::default();
        let mut targets = dummy_targets(4);
        targets.delta_q_target = Some(Tensor::zeros([4, 46], &device));
        targets.delta_q_mask = Some(Tensor::ones([4, 46], &device));
        targets.target_presence = Some(TargetPresence {
            counts: [0, 0, 0, 0, 1, 0],
            delta_q_actions_present: 9,
            batch_size: 4,
        });

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::DeltaQ), 1);
        assert_eq!(presence.delta_q_actions_present, 9);
        assert_eq!(presence.batch_size, 4);
    }

    // -- Full integration: controller with extract_target_presence ----------

    #[test]
    fn controller_full_lifecycle() {
        let mut ctrl = HeadActivationController::new(test_config());
        let device: <B as Backend>::Device = Default::default();

        // Simulate 200 batches with safety_residual present in all.
        for _ in 0..200 {
            let mut targets = dummy_targets(4);
            targets.safety_residual_target = Some(Tensor::zeros([4, 46], &device));
            targets.safety_residual_mask = Some(Tensor::ones([4, 46], &device));
            let presence = extract_target_presence(&targets);
            ctrl.record_batch(&presence);
        }

        // rho should be 1.0 for safety_residual.
        assert!((ctrl.coverage().rho(AdvancedHead::SafetyResidual) - 1.0).abs() < 1e-6);

        // Activate.
        let report = ctrl.try_activate(AdvancedHead::SafetyResidual);
        assert!(report.approved);
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Warmup
        );

        // Record gradient cosines during warmup.
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.4);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.2);
        ctrl.record_grad_cosine(AdvancedHead::SafetyResidual, 0.6);

        // Complete warmup.
        for _ in 0..3 {
            ctrl.tick_warmup();
        }
        assert_eq!(
            ctrl.head_state(AdvancedHead::SafetyResidual),
            HeadState::Active
        );

        // Loss config should now pass through safety_residual weight.
        let base = HydraLossConfig::new().with_w_safety_residual(0.5);
        let gated = ctrl.approved_loss_config(&base);
        assert!((gated.w_safety_residual - 0.5).abs() < 1e-6);

        // Summary should be readable.
        let summary = ctrl.summary();
        assert!(summary.contains("safety_residual"));
        assert!(summary.contains("Active"));
    }

    // -- HeadGateReport summary -------------------------------------------

    #[test]
    fn gate_report_summary_format() {
        let report = HeadGateReport {
            head: AdvancedHead::SafetyResidual,
            approved: false,
            state: HeadState::Off,
            rho: 0.45,
            spp: None,
            negative_frac: 0.0,
            failures: vec!["density_rho_below_threshold"],
        };
        let s = report.summary();
        assert!(s.contains("FAIL"));
        assert!(s.contains("safety_residual"));
        assert!(s.contains("0.450"));
    }
}
