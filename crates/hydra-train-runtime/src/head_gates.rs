//! Head activation discipline runtime adapter.

use std::ops::{Deref, DerefMut};

use hydra_train_types::losses::HydraLossConfig;

pub use hydra_train_types::head_gates::{
    AdvancedHead, DEFAULT_MAX_NEGATIVE_FRAC, DEFAULT_MIN_CONFLICT_CHECKS, DEFAULT_MIN_DENSE_RHO,
    DEFAULT_MIN_EVAL_SAMPLES, DEFAULT_MIN_SPARSE_SPP, DEFAULT_WARMUP_STEPS, GradConflictTracker,
    HeadActivationConfig, HeadCoverage, HeadGateReport, HeadKind, HeadState, NUM_ADVANCED_HEADS,
    TargetPresence, borrow_or_extract_target_presence, extract_target_presence,
    grad_cosine_from_flat,
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
        hydra_train_types::head_gates::approved_loss_config(&self.0, base)
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
