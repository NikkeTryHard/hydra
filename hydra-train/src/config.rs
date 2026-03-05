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
}
