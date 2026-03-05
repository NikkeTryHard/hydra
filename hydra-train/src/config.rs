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
