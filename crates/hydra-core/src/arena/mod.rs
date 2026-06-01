//! Self-play arena: batch game simulation with trajectory collection.

mod buffer;
mod config;
mod labels;
mod policy;
mod scores;
mod trajectory;

pub use buffer::Arena;
pub use config::{ArenaConfig, SelfPlayConfig};
pub use labels::{TrajectoryDeltaQLabel, TrajectoryExitLabel};
pub use policy::{greedy_action, sample_action_with_temperature, softmax_temperature};
pub use scores::{
    avg_score, compute_placements, fourth_place_rate, games_played, mean_placement_from_scores,
    score_std, top_two_rate, total_score_sum, win_rate_from_scores,
};
pub use trajectory::{Trajectory, TrajectoryStep};

#[cfg(test)]
use crate::action::{DISCARD_END, HYDRA_ACTION_SPACE};
#[cfg(test)]
use crate::encoder::OBS_SIZE;

#[cfg(test)]
mod tests;
