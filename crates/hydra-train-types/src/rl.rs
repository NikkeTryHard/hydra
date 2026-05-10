//! RL batch tensor contracts shared below the train facade.
//!
//! This module owns the backend-shaped RL batch container used by self-play
//! generation and RL train-step wrappers without depending on the `hydra-train`
//! facade.

use burn::prelude::*;

use crate::losses::HydraTargets;

/// Complete tensor batch consumed by one RL optimizer step.
#[derive(Clone)]
pub struct RlBatch<B: Backend> {
    /// Encoded observations shaped `[batch, channels, tiles]`.
    pub obs: Tensor<B, 3>,
    /// Selected Hydra action ids shaped `[batch]`.
    pub actions: Tensor<B, 1, Int>,
    /// Old-policy probabilities for selected actions shaped `[batch]`.
    pub pi_old: Tensor<B, 1>,
    /// GAE advantages shaped `[batch]`.
    pub advantages: Tensor<B, 1>,
    /// Baseline logits used by DRDA shaped `[batch, action_space]`.
    pub base_logits: Tensor<B, 2>,
    /// Auxiliary and baseline training targets.
    pub targets: HydraTargets<B>,
    /// Optional ExIt policy target shaped `[batch, action_space]`.
    pub exit_target: Option<Tensor<B, 2>>,
    /// Optional ExIt mask shaped `[batch, action_space]`.
    pub exit_mask: Option<Tensor<B, 2>>,
}

impl<B: Backend> RlBatch<B> {
    /// Number of samples in the batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.obs.dims()[0]
    }

    /// Returns whether all primary batch tensors share the same batch dimension.
    #[must_use]
    pub fn shapes_consistent(&self) -> bool {
        let b = self.batch_size();
        self.actions.dims()[0] == b
            && self.pi_old.dims()[0] == b
            && self.advantages.dims()[0] == b
            && self.base_logits.dims()[0] == b
            && self.targets.legal_mask.dims()[0] == b
    }

    /// Slice all batch tensors along dim 0 to produce `[start..end)`.
    #[allow(
        clippy::single_range_in_vec_init,
        reason = "Burn slice API expects a one-element range slice"
    )]
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let r1 = [start..end];
        Self {
            obs: self.obs.clone().slice(r1.clone()),
            actions: self.actions.clone().slice(r1.clone()),
            pi_old: self.pi_old.clone().slice(r1.clone()),
            advantages: self.advantages.clone().slice(r1.clone()),
            base_logits: self.base_logits.clone().slice(r1.clone()),
            targets: self.targets.slice_batch(start, end),
            exit_target: self
                .exit_target
                .as_ref()
                .map(|t| t.clone().slice(r1.clone())),
            exit_mask: self.exit_mask.as_ref().map(|t| t.clone().slice(r1)),
        }
    }
}
