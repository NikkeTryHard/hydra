//! Phase progression state and budget helpers for Hydra training.
//!
//! This module owns the backend-independent pipeline phase enum and scalar
//! progress counters shared by training crates.

/// Ordered Hydra training phases and their fixed GPU-hour budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainingPhase {
    /// Benchmark and gate-validation phase before learning starts.
    BenchmarkGates,
    /// Supervised behavior-cloning warm start.
    BcWarmStart,
    /// Oracle-guided supervised refinement.
    OracleGuiding,
    /// DRDA/ACH self-play phase.
    DrdaAchSelfPlay,
    /// ExIt pondering phase.
    ExitPondering,
}

impl TrainingPhase {
    /// Returns this phase's standalone GPU-hour budget.
    pub fn gpu_hours_budget(self) -> u32 {
        match self {
            Self::BenchmarkGates => 150,
            Self::BcWarmStart => 50,
            Self::OracleGuiding => 200,
            Self::DrdaAchSelfPlay => 800,
            Self::ExitPondering => 800,
        }
    }

    /// Returns the cumulative GPU-hour budget consumed before this phase.
    pub fn cumulative_budget_before(self) -> u32 {
        match self {
            Self::BenchmarkGates => 0,
            Self::BcWarmStart => Self::BenchmarkGates.gpu_hours_budget(),
            Self::OracleGuiding => {
                Self::BenchmarkGates.gpu_hours_budget() + Self::BcWarmStart.gpu_hours_budget()
            }
            Self::DrdaAchSelfPlay => {
                Self::BenchmarkGates.gpu_hours_budget()
                    + Self::BcWarmStart.gpu_hours_budget()
                    + Self::OracleGuiding.gpu_hours_budget()
            }
            Self::ExitPondering => {
                Self::BenchmarkGates.gpu_hours_budget()
                    + Self::BcWarmStart.gpu_hours_budget()
                    + Self::OracleGuiding.gpu_hours_budget()
                    + Self::DrdaAchSelfPlay.gpu_hours_budget()
            }
        }
    }

    /// Returns the cumulative GPU-hour budget through the end of this phase.
    pub fn cumulative_budget_through(self) -> u32 {
        self.cumulative_budget_before() + self.gpu_hours_budget()
    }

    /// Returns the ExIt schedule phase associated with this training phase.
    pub fn exit_schedule_phase(self) -> u8 {
        match self {
            Self::BenchmarkGates | Self::BcWarmStart | Self::OracleGuiding => 1,
            Self::DrdaAchSelfPlay => 2,
            Self::ExitPondering => 3,
        }
    }

    /// Returns the next phase, or `None` when already at the final phase.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::BenchmarkGates => Some(Self::BcWarmStart),
            Self::BcWarmStart => Some(Self::OracleGuiding),
            Self::OracleGuiding => Some(Self::DrdaAchSelfPlay),
            Self::DrdaAchSelfPlay => Some(Self::ExitPondering),
            Self::ExitPondering => None,
        }
    }

    /// Returns whether this phase trains/uses ExIt outputs.
    pub fn uses_exit(self) -> bool {
        matches!(self, Self::DrdaAchSelfPlay | Self::ExitPondering)
    }

    /// Returns whether this phase consumes oracle guidance.
    pub fn uses_oracle(self) -> bool {
        matches!(
            self,
            Self::OracleGuiding | Self::DrdaAchSelfPlay | Self::ExitPondering
        )
    }

    /// Returns this phase's zero-based rollout index.
    pub fn phase_index(self) -> u8 {
        match self {
            Self::BenchmarkGates => 0,
            Self::BcWarmStart => 1,
            Self::OracleGuiding => 2,
            Self::DrdaAchSelfPlay => 3,
            Self::ExitPondering => 4,
        }
    }
}

/// Serializable scalar state for the phase-aware training pipeline.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PipelineState {
    /// Current training phase.
    pub phase: TrainingPhase,
    /// Total GPU-hours consumed across all phases.
    pub gpu_hours_used: f32,
    /// Total games observed by the pipeline.
    pub total_games: u64,
    /// Total training samples observed by the pipeline.
    pub total_samples: u64,
    /// Learner model version.
    pub learner_version: u32,
    /// Actor model version.
    pub actor_version: u32,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            phase: TrainingPhase::BenchmarkGates,
            gpu_hours_used: 0.0,
            total_games: 0,
            total_samples: 0,
            learner_version: 0,
            actor_version: 0,
        }
    }
}

impl PipelineState {
    /// Advances to the next phase, saturating at the final phase.
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            TrainingPhase::BenchmarkGates => TrainingPhase::BcWarmStart,
            TrainingPhase::BcWarmStart => TrainingPhase::OracleGuiding,
            TrainingPhase::OracleGuiding => TrainingPhase::DrdaAchSelfPlay,
            TrainingPhase::DrdaAchSelfPlay => TrainingPhase::ExitPondering,
            TrainingPhase::ExitPondering => TrainingPhase::ExitPondering,
        };
    }

    /// Returns remaining GPU-hour budget across the complete rollout.
    pub fn remaining_budget(&self) -> f32 {
        2000.0 - self.gpu_hours_used
    }

    /// Returns total GPU-hour budget across all phases.
    pub fn total_budget() -> f32 {
        2000.0
    }

    /// Returns total rollout progress clamped to `[0, 1]`.
    pub fn overall_progress(&self) -> f32 {
        (self.gpu_hours_used / Self::total_budget()).min(1.0)
    }

    /// Returns current phase-local progress.
    pub fn phase_progress(&self) -> f32 {
        let budget = self.phase.gpu_hours_budget() as f32;
        if budget == 0.0 {
            return 0.0;
        }
        self.phase_hours_used() / budget
    }

    /// Returns GPU-hours consumed within the current phase.
    pub fn phase_hours_used(&self) -> f32 {
        let phase_start = self.phase.cumulative_budget_before() as f32;
        let phase_budget = self.phase.gpu_hours_budget() as f32;
        (self.gpu_hours_used - phase_start).clamp(0.0, phase_budget)
    }

    /// Increments the learner version counter.
    pub fn increment_learner_version(&mut self) {
        self.learner_version += 1;
    }

    /// Increments the actor version counter.
    pub fn increment_actor_version(&mut self) {
        self.actor_version += 1;
    }

    /// Records one game and its sample count.
    pub fn record_game(&mut self, num_samples: usize) {
        self.total_games += 1;
        self.total_samples += num_samples as u64;
    }

    /// Adds elapsed GPU-hours.
    pub fn tick_gpu_hours(&mut self, hours: f32) {
        self.gpu_hours_used += hours;
    }

    /// Returns whether cumulative budget permits advancing out of this phase.
    pub fn should_advance_phase(&self) -> bool {
        self.gpu_hours_used >= self.phase.cumulative_budget_through() as f32
    }

    /// Returns a stable human-readable progress summary.
    pub fn progress_summary(&self) -> String {
        format!(
            "phase={:?} phase_hours={:.1}/{} total_hours={:.1} games={} v{}->v{}",
            self.phase,
            self.phase_hours_used(),
            self.phase.gpu_hours_budget(),
            self.gpu_hours_used,
            self.total_games,
            self.learner_version,
            self.actor_version
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_state_defaults() {
        let state = PipelineState::default();
        assert_eq!(state.phase, TrainingPhase::BenchmarkGates);
        assert!((state.remaining_budget() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn phase_advancement() {
        let mut state = PipelineState::default();
        state.advance_phase();
        assert_eq!(state.phase, TrainingPhase::BcWarmStart);
        state.advance_phase();
        assert_eq!(state.phase, TrainingPhase::OracleGuiding);
    }

    #[test]
    fn phase_budgets_are_cumulative() {
        assert_eq!(TrainingPhase::BenchmarkGates.cumulative_budget_before(), 0);
        assert_eq!(TrainingPhase::BcWarmStart.cumulative_budget_before(), 150);
        assert_eq!(TrainingPhase::OracleGuiding.cumulative_budget_before(), 200);
        assert_eq!(
            TrainingPhase::DrdaAchSelfPlay.cumulative_budget_before(),
            400
        );
        assert_eq!(
            TrainingPhase::ExitPondering.cumulative_budget_before(),
            1200
        );
        assert_eq!(
            TrainingPhase::ExitPondering.cumulative_budget_through(),
            2000
        );
    }

    #[test]
    fn phase_progress_uses_phase_local_hours() {
        let state = PipelineState {
            phase: TrainingPhase::DrdaAchSelfPlay,
            gpu_hours_used: 600.0,
            ..PipelineState::default()
        };
        assert!((state.phase_hours_used() - 200.0).abs() < 1e-6);
        assert!((state.phase_progress() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn phase_advance_requires_cumulative_budget() {
        let state = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: 250.0,
            ..PipelineState::default()
        };
        assert!(!state.should_advance_phase());

        let ready = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: 400.0,
            ..PipelineState::default()
        };
        assert!(ready.should_advance_phase());
    }

    #[test]
    fn exit_schedule_phase_matches_hydra_final_rollout_plan() {
        assert_eq!(TrainingPhase::BcWarmStart.exit_schedule_phase(), 1);
        assert_eq!(TrainingPhase::OracleGuiding.exit_schedule_phase(), 1);
        assert_eq!(TrainingPhase::DrdaAchSelfPlay.exit_schedule_phase(), 2);
        assert_eq!(TrainingPhase::ExitPondering.exit_schedule_phase(), 3);
    }

    #[test]
    fn phase_uses_exit_oracle() {
        assert!(!TrainingPhase::BcWarmStart.uses_exit());
        assert!(TrainingPhase::ExitPondering.uses_exit());
        assert!(!TrainingPhase::BcWarmStart.uses_oracle());
        assert!(TrainingPhase::OracleGuiding.uses_oracle());
    }
}
