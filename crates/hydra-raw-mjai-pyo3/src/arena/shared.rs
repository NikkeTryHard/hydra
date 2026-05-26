use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::GameRunner;
use rand::rngs::StdRng;

use crate::PLAYER_COUNT;

pub(crate) struct ArenaGame {
    pub(crate) runner: GameRunner,
    pub(crate) rng: StdRng,
    pub(crate) candidate_seats: [bool; PLAYER_COUNT],
}

pub(crate) struct ArenaRequest {
    pub(crate) game_idx: usize,
    pub(crate) model_id: usize,
    pub(crate) seat_id: u8,
    pub(crate) obs: [f32; OBS_SIZE],
    pub(crate) legal_mask: [bool; HYDRA_ACTION_SPACE],
}

pub(crate) struct ShardRequest {
    pub(crate) shard_idx: usize,
    pub(crate) local_game_idx: usize,
    pub(crate) model_id: usize,
    pub(crate) obs: [f32; OBS_SIZE],
    pub(crate) legal_mask: [bool; HYDRA_ACTION_SPACE],
}
