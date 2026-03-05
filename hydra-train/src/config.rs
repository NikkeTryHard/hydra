//! Model configuration and constants for the Hydra neural architecture.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};

pub const INPUT_CHANNELS: usize = NUM_CHANNELS;
pub const TILE_DIM: usize = NUM_TILES;
pub const HIDDEN_CHANNELS: usize = 256;
pub const SE_REDUCTION: usize = 4;
pub const SE_BOTTLENECK: usize = HIDDEN_CHANNELS / SE_REDUCTION;
pub const NUM_GROUPS: usize = 32;
pub const ACTION_SPACE: usize = HYDRA_ACTION_SPACE;
pub const SCORE_BINS: usize = 64;
pub const NUM_OPPONENTS: usize = 3;
pub const GRP_CLASSES: usize = 24;

pub const RIICHI_ACTION: u8 = 37;

pub fn is_discard_action(action: u8) -> bool {
    action <= 36
}

pub const PASS_ACTION: u8 = 45;

pub const AGARI_ACTION: u8 = 43;
pub const KAN_ACTION: u8 = 44;
pub const FIRST_CALL_ACTION: u8 = 38;
pub const LAST_CALL_ACTION: u8 = 42;

pub fn is_agari_action(action: u8) -> bool {
    action == 43
}

pub fn action_type_name(action: u8) -> &'static str {
    match action {
        0..=36 => "discard",
        37 => "riichi",
        38..=42 => "call",
        43 => "agari",
        44 => "kan",
        45 => "pass",
        _ => "unknown",
    }
}

pub fn is_pass_action(action: u8) -> bool {
    action == PASS_ACTION
}

pub fn is_riichi_action(action: u8) -> bool {
    action == RIICHI_ACTION
}

pub fn is_call_action(action: u8) -> bool {
    (38..=45).contains(&action)
}

pub const GAE_GAMMA: f32 = 0.995;
pub const GAE_LAMBDA: f32 = 0.95;
pub const BC_LR: f64 = 2.5e-4;
pub const BC_LR_MIN: f64 = 1e-6;
pub const ACH_LR: f64 = 2.5e-4;
pub const GRAD_CLIP_NORM: f32 = 1.0;
pub const TAU_DRDA: f32 = 4.0;
pub const TAU_EXIT: f32 = 1.0;
pub const C_PUCT: f32 = 2.5;
pub const AFBS_TOP_K: usize = 5;
pub const CT_SMC_PARTICLES: usize = 128;

pub const ON_TURN_BEAM_W: usize = 64;
pub const ON_TURN_DEPTH: u8 = 4;
pub const ON_TURN_PARTICLES: usize = 128;
pub const PONDER_BEAM_W: usize = 256;
pub const PONDER_DEPTH: u8 = 10;
pub const PONDER_PARTICLES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    BenchmarkGates,
    BcWarmStart,
    OracleGuiding,
    DrdaAchSelfPlay,
    ExitPondering,
}

impl TrainingPhase {
    pub fn gpu_hours_budget(self) -> u32 {
        match self {
            Self::BenchmarkGates => 150,
            Self::BcWarmStart => 50,
            Self::OracleGuiding => 200,
            Self::DrdaAchSelfPlay => 800,
            Self::ExitPondering => 800,
        }
    }

    pub fn uses_exit(self) -> bool {
        matches!(self, Self::DrdaAchSelfPlay | Self::ExitPondering)
    }

    pub fn uses_oracle(self) -> bool {
        matches!(
            self,
            Self::OracleGuiding | Self::DrdaAchSelfPlay | Self::ExitPondering
        )
    }

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

pub struct OracleGuidingConfig {
    pub dropout_start: f32,
    pub dropout_end: f32,
    pub lr_decay_factor: f32,
}

impl OracleGuidingConfig {
    pub fn dropout_at_step(&self, step: usize, total_steps: usize) -> f32 {
        if total_steps == 0 {
            return self.dropout_start;
        }
        let t = (step as f32 / total_steps as f32).min(1.0);
        self.dropout_start + (self.dropout_end - self.dropout_start) * t
    }
}

impl Default for OracleGuidingConfig {
    fn default() -> Self {
        Self {
            dropout_start: 1.0,
            dropout_end: 0.0,
            lr_decay_factor: 0.1,
        }
    }
}

pub fn validate_training_config(
    model_cfg: &crate::model::HydraModelConfig,
    ach_cfg: &crate::training::ach::AchConfig,
    exit_cfg: &crate::training::exit::ExitConfig,
) -> Result<(), &'static str> {
    model_cfg.validate()?;
    ach_cfg.validate()?;
    exit_cfg.validate()?;
    Ok(())
}

pub struct PipelineState {
    pub phase: TrainingPhase,
    pub gpu_hours_used: f32,
    pub total_games: u64,
    pub total_samples: u64,
    pub learner_version: u32,
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
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            TrainingPhase::BenchmarkGates => TrainingPhase::BcWarmStart,
            TrainingPhase::BcWarmStart => TrainingPhase::OracleGuiding,
            TrainingPhase::OracleGuiding => TrainingPhase::DrdaAchSelfPlay,
            TrainingPhase::DrdaAchSelfPlay => TrainingPhase::ExitPondering,
            TrainingPhase::ExitPondering => TrainingPhase::ExitPondering,
        };
    }

    pub fn remaining_budget(&self) -> f32 {
        2000.0 - self.gpu_hours_used
    }

    pub fn total_budget() -> f32 {
        2000.0
    }

    pub fn phase_progress(&self) -> f32 {
        let budget = self.phase.gpu_hours_budget() as f32;
        if budget == 0.0 {
            return 0.0;
        }
        (self.gpu_hours_used / budget).min(1.0)
    }

    pub fn record_game(&mut self, num_samples: usize) {
        self.total_games += 1;
        self.total_samples += num_samples as u64;
    }

    pub fn tick_gpu_hours(&mut self, hours: f32) {
        self.gpu_hours_used += hours;
    }

    pub fn should_advance_phase(&self) -> bool {
        self.gpu_hours_used >= self.phase.gpu_hours_budget() as f32
    }

    pub fn progress_summary(&self) -> String {
        format!(
            "phase={:?} hours={:.1}/{} games={} v{}->v{}",
            self.phase,
            self.gpu_hours_used,
            self.phase.gpu_hours_budget(),
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
    fn constants_match_hydra_core() {
        assert_eq!(INPUT_CHANNELS, 85);
        assert_eq!(TILE_DIM, 34);
        assert_eq!(ACTION_SPACE, 46);
    }

    #[test]
    fn derived_constants_correct() {
        assert_eq!(SE_BOTTLENECK, 64);
        assert_eq!(HIDDEN_CHANNELS / NUM_GROUPS, 8);
        assert_eq!(INPUT_CHANNELS * TILE_DIM, 2890);
    }

    #[test]
    fn hyperparameters_match_spec() {
        assert!((GAE_GAMMA - 0.995).abs() < 1e-6);
        assert!((GAE_LAMBDA - 0.95).abs() < 1e-6);
        assert!((BC_LR - 2.5e-4).abs() < 1e-10);
        assert!((TAU_DRDA - 4.0).abs() < 1e-6);
        assert!((C_PUCT - 2.5).abs() < 1e-6);
        assert_eq!(AFBS_TOP_K, 5);
        assert_eq!(CT_SMC_PARTICLES, 128);
    }

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
    fn action_type_classification() {
        assert_eq!(action_type_name(0), "discard");
        assert_eq!(action_type_name(36), "discard");
        assert_eq!(action_type_name(RIICHI_ACTION), "riichi");
        assert_eq!(action_type_name(AGARI_ACTION), "agari");
        assert_eq!(action_type_name(PASS_ACTION), "pass");
        assert!(is_discard_action(5));
        assert!(!is_discard_action(RIICHI_ACTION));
        assert!(is_call_action(40));
        assert!(!is_call_action(0));
    }

    #[test]
    fn phase_uses_exit_oracle() {
        assert!(!TrainingPhase::BcWarmStart.uses_exit());
        assert!(TrainingPhase::ExitPondering.uses_exit());
        assert!(!TrainingPhase::BcWarmStart.uses_oracle());
        assert!(TrainingPhase::OracleGuiding.uses_oracle());
    }
}
