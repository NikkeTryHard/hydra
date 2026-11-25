//! Feature geometry shared by more than one encoder submodule.
//!
//! Row schemas live with the writer that owns them: tile columns in `records`,
//! seat and match columns in `seats`, descriptor columns in `candidates`. What
//! remains here is the vocabulary more than one writer needs.

use crate::view::HISTORY_EVENT_CAP;
use hydra_rules::rl::actions::RED_TIDS;

use super::WIDTH;

pub(super) const TILE_ROWS: usize = 34;
pub(super) const ABSENT_SEAT: u8 = 4;

/// Relative seat in turn order: 0 self, 1 next, 2 opposite, 3 previous.
pub(super) fn relative_seat(seat: u8, actor: u8) -> u8 {
    (seat + 4 - actor) % 4
}

pub(super) fn tile_face(tile: u8) -> Result<usize, String> {
    if tile > 135 {
        return Err(format!("physical tile {tile} outside 0..=135"));
    }
    Ok(usize::from(tile / 4))
}

pub(super) fn is_red(tile: u8) -> bool {
    RED_TIDS.contains(&tile)
}

pub(super) fn norm_position(position: Option<u16>) -> f32 {
    position.map_or(0.0, |value| {
        (f32::from(value) + 1.0) / (HISTORY_EVENT_CAP as f32 + 1.0)
    })
}

pub(super) fn base(row: usize) -> usize {
    row * WIDTH
}
