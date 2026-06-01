//! Live AFBS ExIt producer for self-play decision-time label generation.
//!
//! Implements the Agent 22 blueprint: a learner-only, root-only AFBS
//! producer that generates visit-based ExIt labels at decision time during
//! self-play.  The producer is **default-on** after clearing the infrastructure
//! validation matrix, and emits `None` on any failed gate.
//!
//! The surviving evaluator is the current public model value head, used as
//! a leaf scorer inside root-only AFBS over all legal discard children.
//! The teacher object is root child visits via [`crate::exit::build_exit_from_afbs_tree`],
//! not q-softmax via `root_exit_policy()`.

mod adapter;
mod config;
mod hash;
mod producer;
mod root;

pub use adapter::{ExitSearchAdapter, SelfPlayExitAdapter};
pub use config::{LiveExitConfig, make_live_exit_fn};
pub use hash::obs_hash;
pub use producer::{
    try_exit_label_from_context, try_exit_label_from_context_with_batched_child_values,
    try_live_exit_label, try_live_search_labels, try_live_search_labels_selfplay,
    try_search_labels_from_context, try_search_labels_from_context_with_batched_child_values,
};
pub use root::{
    base_pi_from_logits, budget_from_legal_count, legal_discard_actions,
    seed_root_children_all_legal,
};

pub use hydra_train_types::selfplay::{RootDecisionContext, StepRecord};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrajectorySearchLabels {
    pub exit: Option<hydra_core::arena::TrajectoryExitLabel>,
    pub delta_q: Option<hydra_core::arena::TrajectoryDeltaQLabel>,
}

#[cfg(test)]
mod tests;
