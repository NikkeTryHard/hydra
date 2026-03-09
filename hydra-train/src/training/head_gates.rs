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
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Constants (archive-recommended defaults from answer_13_combined.md)
// ---------------------------------------------------------------------------

/// Number of gated advanced heads.
pub const NUM_ADVANCED_HEADS: usize = 6;

/// Dense heads require at least 80% of samples to carry the target.
pub const DEFAULT_MIN_DENSE_RHO: f32 = 0.8;

/// Sparse search-derived heads require at least 5 labeled samples per
/// learner parameter.
pub const DEFAULT_MIN_SPARSE_SPP: f32 = 5.0;

/// A head is considered conflicting if shared-trunk gradient cosine with
/// policy+value is negative on more than 30% of checks.
pub const DEFAULT_MAX_NEGATIVE_FRAC: f32 = 0.3;

/// Head-only warmup duration (trunk frozen) before unfreeze decision.
pub const DEFAULT_WARMUP_STEPS: usize = 10_000;

/// Minimum accumulated samples before density evaluation is meaningful.
pub const DEFAULT_MIN_EVAL_SAMPLES: u64 = 1000;

/// Minimum gradient cosine checks before conflict gate is evaluated.
pub const DEFAULT_MIN_CONFLICT_CHECKS: u64 = 10;

// ---------------------------------------------------------------------------
// AdvancedHead -- the six gated output heads
// ---------------------------------------------------------------------------

/// Advanced output heads subject to activation gating.
///
/// These are the heads whose loss weights default to zero and require
/// density/interference clearance before activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdvancedHead {
    OracleCritic,
    BeliefFields,
    MixtureWeight,
    OpponentHandType,
    DeltaQ,
    SafetyResidual,
}

impl AdvancedHead {
    /// All advanced heads in index order.
    pub const ALL: [AdvancedHead; NUM_ADVANCED_HEADS] = [
        Self::OracleCritic,
        Self::BeliefFields,
        Self::MixtureWeight,
        Self::OpponentHandType,
        Self::DeltaQ,
        Self::SafetyResidual,
    ];

    /// Returns the array index for this head.
    pub fn index(self) -> usize {
        match self {
            Self::OracleCritic => 0,
            Self::BeliefFields => 1,
            Self::MixtureWeight => 2,
            Self::OpponentHandType => 3,
            Self::DeltaQ => 4,
            Self::SafetyResidual => 5,
        }
    }

    /// Returns whether this head uses dense or sparse-search density rules.
    pub fn kind(self) -> HeadKind {
        match self {
            Self::DeltaQ => HeadKind::SparseSearch,
            _ => HeadKind::Dense,
        }
    }

    /// Returns the snake_case name matching `HydraLossConfig` field names.
    pub fn name(self) -> &'static str {
        match self {
            Self::OracleCritic => "oracle_critic",
            Self::BeliefFields => "belief_fields",
            Self::MixtureWeight => "mixture_weight",
            Self::OpponentHandType => "opponent_hand_type",
            Self::DeltaQ => "delta_q",
            Self::SafetyResidual => "safety_residual",
        }
    }
}

// ---------------------------------------------------------------------------
// HeadKind -- density threshold selection
// ---------------------------------------------------------------------------

/// Classification that determines which density threshold applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// Replay-derived head. Gate: `rho_h >= min_dense_rho`.
    Dense,
    /// Search-derived head with sparse labels. Gate: `spp_h >= min_sparse_spp`.
    SparseSearch,
}

// ---------------------------------------------------------------------------
// HeadState -- per-head activation state machine
// ---------------------------------------------------------------------------

/// Per-head activation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadState {
    /// Head is off: loss weight is forced to zero.
    Off,
    /// Head is warming up: loss weight is nonzero but the caller should
    /// freeze (detach) trunk outputs for this head's loss so only the head
    /// parameters train.
    Warmup,
    /// Head is fully active: loss weight is nonzero and trunk gradient flow
    /// is unrestricted.
    Active,
}

// ---------------------------------------------------------------------------
// TargetPresence -- per-batch target availability snapshot
// ---------------------------------------------------------------------------

/// Per-head count of samples with valid targets in a single batch.
#[derive(Clone, Debug)]
pub struct TargetPresence {
    /// Per-head count of samples carrying a valid target in this batch.
    pub counts: [usize; NUM_ADVANCED_HEADS],
    /// Total samples in this batch.
    pub batch_size: usize,
}

impl Default for TargetPresence {
    fn default() -> Self {
        Self {
            counts: [0; NUM_ADVANCED_HEADS],
            batch_size: 0,
        }
    }
}

impl TargetPresence {
    /// Creates a presence snapshot with the given batch size and all counts zero.
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            counts: [0; NUM_ADVANCED_HEADS],
            batch_size,
        }
    }

    /// Returns the number of samples with a valid target for `head`.
    pub fn count(&self, head: AdvancedHead) -> usize {
        self.counts[head.index()]
    }
}

/// Extracts per-head target presence from a batch of [`HydraTargets`].
///
/// For targets with per-sample masks (`belief_fields`, `mixture_weight`),
/// counts the number of samples where the mask is nonzero. For targets
/// without per-sample masks, counts `batch_size` when the target is present.
pub fn extract_target_presence<B: Backend>(targets: &HydraTargets<B>) -> TargetPresence {
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
            Some(mask) => count_nonzero_1d(mask),
            None => batch_size,
        };
    }

    // Mixture weight: per-sample mask.
    if targets.mixture_weight_target.is_some() {
        counts[AdvancedHead::MixtureWeight.index()] = match &targets.mixture_weight_mask {
            Some(mask) => count_nonzero_1d(mask),
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

    // Delta Q: if present, all samples have it.
    if targets.delta_q_target.is_some() {
        counts[AdvancedHead::DeltaQ.index()] = batch_size;
    }

    // Safety residual: if present, all samples have it. The per-action mask
    // determines which actions contribute to loss, not which samples.
    if targets.safety_residual_target.is_some() {
        counts[AdvancedHead::SafetyResidual.index()] = batch_size;
    }

    TargetPresence { counts, batch_size }
}

/// Counts nonzero entries in a 1-D tensor.
fn count_nonzero_1d<B: Backend>(tensor: &Tensor<B, 1>) -> usize {
    match tensor.to_data().as_slice::<f32>() {
        Ok(data) => data.iter().filter(|&&v| v > 0.0).count(),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// HeadCoverage -- cumulative per-head label density tracker
// ---------------------------------------------------------------------------

/// Tracks cumulative per-head label density across training batches.
///
/// Used to compute `rho_h` (fraction of samples with target `h`) and `spp_h`
/// (labeled samples per learner parameter) for the density gate.
#[derive(Clone, Debug)]
pub struct HeadCoverage {
    samples_with_target: [u64; NUM_ADVANCED_HEADS],
    total_samples: u64,
}

impl HeadCoverage {
    /// Creates an empty coverage tracker.
    pub fn new() -> Self {
        Self {
            samples_with_target: [0; NUM_ADVANCED_HEADS],
            total_samples: 0,
        }
    }

    /// Records one batch of target presence.
    pub fn record_batch(&mut self, presence: &TargetPresence) {
        self.total_samples += presence.batch_size as u64;
        for &head in &AdvancedHead::ALL {
            self.samples_with_target[head.index()] += presence.count(head) as u64;
        }
    }

    /// Returns `rho_h`: fraction of all samples that carried target `h`.
    ///
    /// Returns 0.0 if no samples have been recorded.
    pub fn rho(&self, head: AdvancedHead) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.samples_with_target[head.index()] as f32 / self.total_samples as f32
    }

    /// Returns `spp_h`: labeled samples for head `h` per learner parameter.
    ///
    /// Returns 0.0 if `learner_params` is zero.
    pub fn spp(&self, head: AdvancedHead, learner_params: usize) -> f32 {
        if learner_params == 0 {
            return 0.0;
        }
        self.samples_with_target[head.index()] as f32 / learner_params as f32
    }

    /// Returns total accumulated samples.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Returns accumulated labeled samples for `head`.
    pub fn labeled_samples(&self, head: AdvancedHead) -> u64 {
        self.samples_with_target[head.index()]
    }
}

impl Default for HeadCoverage {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GradConflictTracker -- shared-trunk gradient interference detection
// ---------------------------------------------------------------------------

/// Tracks per-head gradient cosine measurements for conflict detection.
///
/// Records whether each gradient cosine check was negative (conflicting)
/// and computes the fraction of negative checks per head.
#[derive(Clone, Debug)]
pub struct GradConflictTracker {
    negative_counts: [u64; NUM_ADVANCED_HEADS],
    total_checks: [u64; NUM_ADVANCED_HEADS],
}

impl GradConflictTracker {
    /// Creates an empty conflict tracker.
    pub fn new() -> Self {
        Self {
            negative_counts: [0; NUM_ADVANCED_HEADS],
            total_checks: [0; NUM_ADVANCED_HEADS],
        }
    }

    /// Records a gradient cosine measurement for `head`.
    pub fn record(&mut self, head: AdvancedHead, cosine: f32) {
        let idx = head.index();
        self.total_checks[idx] += 1;
        if cosine < 0.0 {
            self.negative_counts[idx] += 1;
        }
    }

    /// Returns the fraction of checks where cosine was negative for `head`.
    ///
    /// Returns 0.0 if no checks have been recorded.
    pub fn negative_fraction(&self, head: AdvancedHead) -> f32 {
        let idx = head.index();
        if self.total_checks[idx] == 0 {
            return 0.0;
        }
        self.negative_counts[idx] as f32 / self.total_checks[idx] as f32
    }

    /// Returns true if the head has persistent negative gradient conflict.
    pub fn is_conflicting(&self, head: AdvancedHead, max_negative_frac: f32) -> bool {
        self.negative_fraction(head) > max_negative_frac
    }

    /// Returns total gradient cosine checks recorded for `head`.
    pub fn total_checks(&self, head: AdvancedHead) -> u64 {
        self.total_checks[head.index()]
    }
}

impl Default for GradConflictTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// grad_cosine_from_flat -- cosine similarity on flattened gradient vectors
// ---------------------------------------------------------------------------

/// Computes cosine similarity between two flattened gradient vectors.
///
/// Returns 0.0 if either vector has near-zero norm. Panics if the vectors
/// have different lengths (mismatched gradient vectors are a bug).
///
/// # Usage
///
/// The caller extracts flattened shared-trunk gradients from two separate
/// backward passes (one for the aux head loss, one for policy+value loss)
/// and passes them here. See module docs for integration guidance.
pub fn grad_cosine_from_flat(grad_a: &[f32], grad_b: &[f32]) -> f32 {
    assert_eq!(
        grad_a.len(),
        grad_b.len(),
        "gradient vectors must have equal length"
    );
    let dot: f32 = grad_a.iter().zip(grad_b).map(|(a, b)| a * b).sum();
    let norm_a: f32 = grad_a.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = grad_b.iter().map(|b| b * b).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// HeadActivationConfig
// ---------------------------------------------------------------------------

/// Configuration for the head activation gate system.
#[derive(Clone, Debug)]
pub struct HeadActivationConfig {
    /// Minimum `rho_h` for dense heads (default: 0.8).
    pub min_dense_rho: f32,
    /// Minimum `spp_h` for sparse search-derived heads (default: 5.0).
    pub min_sparse_spp: f32,
    /// Maximum fraction of negative gradient cosine checks before a head
    /// is considered conflicting (default: 0.3).
    pub max_negative_frac: f32,
    /// Number of head-only warmup steps before trunk unfreeze (default: 10000).
    pub warmup_steps: usize,
    /// Estimated learner model parameter count (for `spp` computation).
    pub learner_params: usize,
    /// Minimum accumulated samples before density evaluation (default: 1000).
    pub min_eval_samples: u64,
    /// Minimum gradient cosine checks before conflict gate is evaluated
    /// (default: 10).
    pub min_conflict_checks: u64,
}

impl HeadActivationConfig {
    /// Creates a config with archive-recommended defaults and the given
    /// learner parameter count.
    pub fn default_with_params(learner_params: usize) -> Self {
        Self {
            min_dense_rho: DEFAULT_MIN_DENSE_RHO,
            min_sparse_spp: DEFAULT_MIN_SPARSE_SPP,
            max_negative_frac: DEFAULT_MAX_NEGATIVE_FRAC,
            warmup_steps: DEFAULT_WARMUP_STEPS,
            learner_params,
            min_eval_samples: DEFAULT_MIN_EVAL_SAMPLES,
            min_conflict_checks: DEFAULT_MIN_CONFLICT_CHECKS,
        }
    }
}

// ---------------------------------------------------------------------------
// HeadGateReport -- per-head gate evaluation result
// ---------------------------------------------------------------------------

/// Result of evaluating activation gates for a single head.
#[derive(Clone, Debug)]
pub struct HeadGateReport {
    /// Which head was evaluated.
    pub head: AdvancedHead,
    /// Whether the head passed all applicable gates.
    pub approved: bool,
    /// Current head state.
    pub state: HeadState,
    /// Label density `rho_h` (fraction of samples with target).
    pub rho: f32,
    /// Samples-per-param `spp_h` (only meaningful for sparse search heads).
    pub spp: Option<f32>,
    /// Fraction of gradient cosine checks that were negative.
    pub negative_frac: f32,
    /// Human-readable list of gate failures.
    pub failures: Vec<&'static str>,
}

impl HeadGateReport {
    /// Returns a one-line summary for logging.
    pub fn summary(&self) -> String {
        let spp_str = self.spp.map_or(String::new(), |s| format!(", spp={s:.2}"));
        let status = if self.approved { "PASS" } else { "FAIL" };
        format!(
            "[{}] {} rho={:.3}{}, neg_frac={:.3}, state={:?}{}",
            status,
            self.head.name(),
            self.rho,
            spp_str,
            self.negative_frac,
            self.state,
            if self.failures.is_empty() {
                String::new()
            } else {
                format!(" ({})", self.failures.join(", "))
            },
        )
    }
}

// ---------------------------------------------------------------------------
// HeadActivationController -- orchestrates density, conflict, and warmup
// ---------------------------------------------------------------------------

/// Manages per-head activation state, density tracking, gradient conflict
/// monitoring, and warmup-to-active transitions.
///
/// All advanced heads start in [`HeadState::Off`]. The caller requests
/// activation via [`try_activate`](Self::try_activate), which checks the
/// density gate and transitions to [`HeadState::Warmup`] if it passes.
/// During warmup, the caller should freeze trunk gradient flow for the
/// head's loss. After the warmup countdown completes,
/// [`tick_warmup`](Self::tick_warmup) checks the gradient conflict gate
/// and transitions to [`HeadState::Active`] or back to [`HeadState::Off`].
#[derive(Clone, Debug)]
pub struct HeadActivationController {
    coverage: HeadCoverage,
    conflict: GradConflictTracker,
    config: HeadActivationConfig,
    states: [HeadState; NUM_ADVANCED_HEADS],
    warmup_steps_remaining: [usize; NUM_ADVANCED_HEADS],
}

impl HeadActivationController {
    /// Creates a controller with all heads in [`HeadState::Off`].
    pub fn new(config: HeadActivationConfig) -> Self {
        Self {
            coverage: HeadCoverage::new(),
            conflict: GradConflictTracker::new(),
            config,
            states: [HeadState::Off; NUM_ADVANCED_HEADS],
            warmup_steps_remaining: [0; NUM_ADVANCED_HEADS],
        }
    }

    // -- Data collection ---------------------------------------------------

    /// Records per-head target presence from one training batch.
    pub fn record_batch(&mut self, presence: &TargetPresence) {
        self.coverage.record_batch(presence);
    }

    /// Records a shared-trunk gradient cosine measurement for `head`.
    pub fn record_grad_cosine(&mut self, head: AdvancedHead, cosine: f32) {
        self.conflict.record(head, cosine);
    }

    // -- State queries -----------------------------------------------------

    /// Returns the current activation state of `head`.
    pub fn head_state(&self, head: AdvancedHead) -> HeadState {
        self.states[head.index()]
    }

    /// Returns all heads currently in [`HeadState::Warmup`].
    ///
    /// The caller should detach trunk outputs for these heads so only head
    /// parameters receive gradients.
    pub fn warmup_heads(&self) -> Vec<AdvancedHead> {
        AdvancedHead::ALL
            .iter()
            .copied()
            .filter(|h| self.states[h.index()] == HeadState::Warmup)
            .collect()
    }

    /// Returns a reference to the underlying coverage tracker.
    pub fn coverage(&self) -> &HeadCoverage {
        &self.coverage
    }

    /// Returns a reference to the underlying conflict tracker.
    pub fn conflict(&self) -> &GradConflictTracker {
        &self.conflict
    }

    // -- Gate evaluation ---------------------------------------------------

    /// Evaluates all applicable gates for `head` without changing state.
    pub fn evaluate(&self, head: AdvancedHead) -> HeadGateReport {
        let mut failures = Vec::new();
        let rho = self.coverage.rho(head);
        let spp = match head.kind() {
            HeadKind::SparseSearch => Some(self.coverage.spp(head, self.config.learner_params)),
            HeadKind::Dense => None,
        };
        let negative_frac = self.conflict.negative_fraction(head);

        // Check minimum samples.
        if self.coverage.total_samples() < self.config.min_eval_samples {
            failures.push("insufficient_samples");
        }

        // Density gate.
        match head.kind() {
            HeadKind::Dense => {
                if rho < self.config.min_dense_rho {
                    failures.push("density_rho_below_threshold");
                }
            }
            HeadKind::SparseSearch => {
                if let Some(s) = spp
                    && s < self.config.min_sparse_spp
                {
                    failures.push("density_spp_below_threshold");
                }
            }
        }

        // Gradient conflict gate (only if enough checks).
        if self.conflict.total_checks(head) >= self.config.min_conflict_checks
            && self
                .conflict
                .is_conflicting(head, self.config.max_negative_frac)
        {
            failures.push("gradient_conflict");
        }

        HeadGateReport {
            head,
            approved: failures.is_empty(),
            state: self.states[head.index()],
            rho,
            spp,
            negative_frac,
            failures,
        }
    }

    /// Evaluates all advanced heads.
    pub fn evaluate_all(&self) -> Vec<HeadGateReport> {
        AdvancedHead::ALL
            .iter()
            .map(|&h| self.evaluate(h))
            .collect()
    }

    // -- State transitions -------------------------------------------------

    /// Attempts to activate `head`.
    ///
    /// - If the head is [`HeadState::Off`] and the density gate passes,
    ///   transitions to [`HeadState::Warmup`] with the configured warmup
    ///   countdown.
    /// - If the head is already in `Warmup` or `Active`, returns a report
    ///   reflecting the current state without changing it.
    pub fn try_activate(&mut self, head: AdvancedHead) -> HeadGateReport {
        let idx = head.index();
        match self.states[idx] {
            HeadState::Warmup | HeadState::Active => {
                // Already activated or activating.
                return self.evaluate(head);
            }
            HeadState::Off => {}
        }

        let report = self.evaluate(head);

        // Only check density gate for Off -> Warmup transition.
        // Gradient conflict is checked after warmup completes.
        let density_ok = !report.failures.contains(&"insufficient_samples")
            && !report.failures.contains(&"density_rho_below_threshold")
            && !report.failures.contains(&"density_spp_below_threshold");

        if density_ok {
            self.states[idx] = HeadState::Warmup;
            self.warmup_steps_remaining[idx] = self.config.warmup_steps;
            // Return updated report reflecting new state.
            let mut updated = report;
            updated.state = HeadState::Warmup;
            // For Off -> Warmup, density passed so we approve the transition.
            // The conflict gate is deferred until warmup completes.
            updated.approved = true;
            updated.failures.retain(|f| *f != "gradient_conflict");
            return updated;
        }

        report
    }

    /// Attempts to activate all heads that are currently [`HeadState::Off`].
    pub fn try_activate_all(&mut self) -> Vec<HeadGateReport> {
        AdvancedHead::ALL
            .iter()
            .map(|&h| self.try_activate(h))
            .collect()
    }

    /// Advances warmup countdowns by one step and handles transitions.
    ///
    /// For each head in [`HeadState::Warmup`]:
    /// - If warmup steps remain, decrements the counter.
    /// - If warmup is complete and sufficient gradient conflict data exists,
    ///   transitions to [`HeadState::Active`] (conflict passes) or
    ///   [`HeadState::Off`] (conflict fails).
    /// - If warmup is complete but insufficient conflict data, stays in
    ///   `Warmup` until enough data accumulates.
    pub fn tick_warmup(&mut self) {
        for &head in &AdvancedHead::ALL {
            let idx = head.index();
            if self.states[idx] != HeadState::Warmup {
                continue;
            }

            if self.warmup_steps_remaining[idx] > 0 {
                self.warmup_steps_remaining[idx] -= 1;
                if self.warmup_steps_remaining[idx] > 0 {
                    continue;
                }
            }

            // Warmup countdown complete. Check conflict gate if we have
            // enough gradient cosine data.
            if self.conflict.total_checks(head) < self.config.min_conflict_checks {
                // Not enough data yet; stay in warmup.
                continue;
            }

            if self
                .conflict
                .is_conflicting(head, self.config.max_negative_frac)
            {
                self.states[idx] = HeadState::Off;
            } else {
                self.states[idx] = HeadState::Active;
            }
        }
    }

    /// Forces a head back to [`HeadState::Off`].
    pub fn force_off(&mut self, head: AdvancedHead) {
        let idx = head.index();
        self.states[idx] = HeadState::Off;
        self.warmup_steps_remaining[idx] = 0;
    }

    // -- Loss config integration -------------------------------------------

    /// Returns a [`HydraLossConfig`] with unapproved heads zeroed out.
    ///
    /// - [`HeadState::Off`] heads get weight `0.0`.
    /// - [`HeadState::Warmup`] and [`HeadState::Active`] heads keep
    ///   their weight from `base`.
    ///
    /// Baseline heads (policy, value, grp, tenpai, danger, opp_next, score)
    /// are always passed through unchanged.
    pub fn approved_loss_config(&self, base: &HydraLossConfig) -> HydraLossConfig {
        let gate = |head: AdvancedHead, w: f32| -> f32 {
            match self.states[head.index()] {
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

    /// Returns a multi-line summary of all heads for logging.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "HeadActivationController (samples={})",
            self.coverage.total_samples()
        ));
        for &head in &AdvancedHead::ALL {
            let report = self.evaluate(head);
            lines.push(format!("  {}", report.summary()));
        }
        lines.join("\n")
    }
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
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
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
    fn extract_presence_delta_q_all_present() {
        let device = Default::default();
        let mut targets = dummy_targets(8);
        targets.delta_q_target = Some(Tensor::zeros([8, 46], &device));

        let presence = extract_target_presence(&targets);
        assert_eq!(presence.count(AdvancedHead::DeltaQ), 8);
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
