//! Game loop runner with proper phase handling and safety tracking.
//!
//! Provides `GameRunner` which orchestrates the full game loop:
//! WaitAct/WaitResponse handling, SafetyInfo updates, and
//! policy-driven action selection.

mod encode;
mod external_step;
mod lifecycle;
mod outcome;
mod pending;
mod record;
mod runner;
mod safety_track;
mod selector;
mod stepping;

pub use outcome::StepOutcome;
pub use pending::{CachedLegalActions, PendingDecision, PendingDecisionTiming};
pub use record::DecisionRecord;
pub use runner::GameRunner;
pub use selector::{ActionDecision, ActionSelector, FirstActionSelector};

const MAX_STEPS: u32 = 50_000;

#[cfg(test)]
mod tests;
