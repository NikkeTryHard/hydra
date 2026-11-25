//! Named feature geometry shared by the encoder submodules.

use crate::view::{HISTORY_EVENT_CAP, Observation as NativeObservation};
use hydra_rules::rl::actions::RED_TIDS;

use super::WIDTH;

pub(super) const TILE_ROWS: usize = 34;
pub(super) const SEAT_ROW0: usize = TILE_ROWS;
pub(super) const MATCH_ROW: usize = SEAT_ROW0 + 4;
pub(super) const DORA_SLOTS: usize = 5;
pub(super) const RECORDS: usize = 4;
pub(super) const RECORD_WIDTH: usize = 9;
pub(super) const ABSENT_SEAT: u8 = 4;

pub(super) mod tile {
    pub(crate) const OWN_COUNT: usize = 0;
    pub(crate) const OWN_RED: usize = 1;
    pub(crate) const DRAWN_FACE: usize = 2;
    pub(crate) const DRAWN_RED: usize = 3;
    pub(crate) const PUBLIC_COUNT: usize = 4;
    pub(crate) const PUBLIC_RED: usize = 5;
    pub(crate) const DORA0: usize = 6;
    pub(crate) const RECORD0: usize = 11;
    pub(crate) const RESERVED: usize = 47;
}

pub(super) mod seat {
    pub(crate) const PRESENT: usize = 0;
    pub(crate) const RELATIVE: usize = 1;
    pub(crate) const WIND: usize = 2;
    pub(crate) const DEALER: usize = 3;
    pub(crate) const SCORE: usize = 4;
    pub(crate) const RANK: usize = 5;
    pub(crate) const RIICHI: usize = 6;
    pub(crate) const IPPATSU: usize = 7;
    pub(crate) const MELD_COUNT: usize = 8;
    pub(crate) const LAST_PUBLIC: usize = 9;
    pub(crate) const LAST_DISCARD: usize = 10;
    pub(crate) const LAST_CALL: usize = 11;
    pub(crate) const RIICHI_DECLARED: usize = 12;
    pub(crate) const RIICHI_ACCEPTED: usize = 13;
    pub(crate) const IS_TURN: usize = 14;
    pub(crate) const FURITEN: usize = 15;
    pub(crate) const CAN_TSUMO: usize = 16;
    pub(crate) const CAN_RIICHI: usize = 17;
    pub(crate) const DEALER_OFFSET: usize = 18;
    pub(crate) const WIND_OFFSET: usize = 19;
}

pub(super) mod match_row {
    pub(crate) const PRESENT: usize = 0;
    pub(crate) const ROUND_WIND: usize = 1;
    pub(crate) const HAND_NUMBER: usize = 2;
    pub(crate) const HONBA: usize = 3;
    pub(crate) const RIICHI_STICKS: usize = 4;
    pub(crate) const DEALER_OFFSET: usize = 5;
    pub(crate) const LIVE_WALL: usize = 6;
    pub(crate) const KAN_COUNT: usize = 7;
    pub(crate) const PHASE: usize = 8;
    pub(crate) const ACTOR_DEALER: usize = 9;
    pub(crate) const ACTOR_WIND: usize = 10;
    pub(crate) const ROUND_INDEX: usize = 11;
    pub(crate) const SCORE_SPREAD: usize = 12;
    pub(crate) const ACTOR_SCORE: usize = 13;
    pub(crate) const ACTOR_RANK: usize = 14;
    pub(crate) const TURN_ACTOR: usize = 15;
}

pub(super) mod desc {
    pub(crate) const ID: usize = 0;
    pub(crate) const KIND: usize = 1;
    pub(crate) const SOURCE: usize = 2;
    pub(crate) const FACE: usize = 3;
    pub(crate) const RED: usize = 4;
    pub(crate) const FROM_DRAW: usize = 5;
    pub(crate) const RIICHI: usize = 6;
    pub(crate) const REMOVALS: usize = 7;
    pub(crate) const REMOVAL_RED: usize = 11;
    pub(crate) const CONTESTED: usize = 15;
}

/// Relative seat in turn order: 0 self, 1 next, 2 opposite, 3 previous.
pub(super) fn relative_seat(seat: u8, actor: u8) -> u8 {
    (seat + 4 - actor) % 4
}

pub(super) fn relative_owner(owner: u8, actor: u8) -> u8 {
    if owner < 4 {
        relative_seat(owner, actor)
    } else {
        ABSENT_SEAT
    }
}

pub(super) fn relative_source(source: Option<u8>, actor: u8) -> Result<u8, String> {
    match source {
        None => Ok(ABSENT_SEAT),
        Some(seat) if seat < 4 && seat != actor => Ok(relative_seat(seat, actor)),
        Some(seat) => Err(format!("invalid source seat {seat} for actor {actor}")),
    }
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

pub(super) fn rank_of(scores: &[i32; 4], seat: u8) -> u8 {
    let own = scores[usize::from(seat)];
    let better = scores.iter().filter(|score| **score > own).count();
    u8::try_from(better + 1).unwrap_or(4)
}

pub(super) fn wind_ordinal(obs: &NativeObservation, seat: u8) -> Result<u8, String> {
    wind_ordinal_of(obs.seat_winds[usize::from(seat)])
        .map_err(|error| format!("seat wind: {error}"))
}

pub(super) fn round_wind_ordinal(obs: &NativeObservation) -> Result<u8, String> {
    wind_ordinal_of(obs.round_wind).map_err(|error| format!("round wind: {error}"))
}

fn wind_ordinal_of(tile: i64) -> Result<u8, String> {
    if (27..=30).contains(&tile) {
        Ok((tile - 27) as u8)
    } else {
        Err(format!("wind tile {tile} not in 27..30"))
    }
}

pub(super) fn riichi_ordinal(state: &str) -> Result<u8, String> {
    match state {
        "none" => Ok(0),
        "declared" => Ok(1),
        "accepted" => Ok(2),
        _ => Err(format!("unknown riichi state {state:?}")),
    }
}

pub(super) fn phase_ordinal(phase: &str) -> Result<u8, String> {
    match phase {
        "draw_decision" => Ok(1),
        "discard_response" => Ok(2),
        "kan_response" => Ok(3),
        _ => Err(format!("unknown decision phase {phase:?}")),
    }
}

pub(super) fn furiten_ordinal(state: &str) -> Result<u8, String> {
    match state {
        "none" => Ok(0),
        "temporary" => Ok(1),
        "riichi" => Ok(2),
        _ => Err(format!("unknown furiten state {state:?}")),
    }
}

pub(super) fn base(row: usize) -> usize {
    row * WIDTH
}
