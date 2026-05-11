//! Scalar and tensor-aware head activation gate helpers.
//!
//! This module owns backend-independent state machines, counters, thresholds,
//! gradient-vector math, tensor target-presence extraction, and loss-weight
//! gating helpers shared across training crates.

use crate::losses::{HydraLossConfig, HydraTargets};
use std::borrow::Cow;

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
    /// Oracle critic auxiliary head.
    OracleCritic,
    /// Belief-field auxiliary head.
    BeliefFields,
    /// Opponent mixture-weight auxiliary head.
    MixtureWeight,
    /// Opponent hand-type auxiliary head.
    OpponentHandType,
    /// Search-derived delta-Q auxiliary head.
    DeltaQ,
    /// Safety residual auxiliary head.
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
#[derive(Clone, Copy, Debug)]
pub struct TargetPresence {
    /// Per-head count of samples carrying a valid target in this batch.
    pub counts: [usize; NUM_ADVANCED_HEADS],
    /// Number of delta-Q legal action labels present in this batch.
    pub delta_q_actions_present: usize,
    /// Total samples in this batch.
    pub batch_size: usize,
}

impl Default for TargetPresence {
    fn default() -> Self {
        Self {
            counts: [0; NUM_ADVANCED_HEADS],
            delta_q_actions_present: 0,
            batch_size: 0,
        }
    }
}

impl TargetPresence {
    /// Creates a presence snapshot with the given batch size and all counts zero.
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            counts: [0; NUM_ADVANCED_HEADS],
            delta_q_actions_present: 0,
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

    /// Records per-head target presence from one training batch.
    pub fn record_batch(&mut self, presence: &TargetPresence) {
        self.coverage.record_batch(presence);
    }

    /// Records a shared-trunk gradient cosine measurement for `head`.
    pub fn record_grad_cosine(&mut self, head: AdvancedHead, cosine: f32) {
        self.conflict.record(head, cosine);
    }

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

    /// Returns a [`HydraLossConfig`] with unapproved heads zeroed out.
    pub fn approved_loss_config(&self, base: &HydraLossConfig) -> HydraLossConfig {
        approved_loss_config(self, base)
    }

    /// Evaluates all applicable gates for `head` without changing state.
    pub fn evaluate(&self, head: AdvancedHead) -> HeadGateReport {
        let mut failures = Vec::new();
        let rho = self.coverage.rho(head);
        let spp = match head.kind() {
            HeadKind::SparseSearch => Some(self.coverage.spp(head, self.config.learner_params)),
            HeadKind::Dense => None,
        };
        let negative_frac = self.conflict.negative_fraction(head);

        if self.coverage.total_samples() < self.config.min_eval_samples {
            failures.push("insufficient_samples");
        }

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
                return self.evaluate(head);
            }
            HeadState::Off => {}
        }

        let report = self.evaluate(head);

        let density_ok = !report.failures.contains(&"insufficient_samples")
            && !report.failures.contains(&"density_rho_below_threshold")
            && !report.failures.contains(&"density_spp_below_threshold");

        if density_ok {
            self.states[idx] = HeadState::Warmup;
            self.warmup_steps_remaining[idx] = self.config.warmup_steps;
            let mut updated = report;
            updated.state = HeadState::Warmup;
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
    /// - If warmup steps remain, decrements the countdown.
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

            if self.conflict.total_checks(head) < self.config.min_conflict_checks {
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

#[cfg(test)]
mod tests;
