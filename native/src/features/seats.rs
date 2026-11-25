//! Seat-relative and match token rows.

use crate::view::Observation as NativeObservation;

use super::WIDTH;
use super::layout::{TILE_ROWS, base, norm_position, relative_seat};
use super::ledger::Ledger;

pub(super) const SEAT_ROW0: usize = TILE_ROWS;
pub(super) const MATCH_ROW: usize = SEAT_ROW0 + 4;

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

pub(super) fn write_seat_rows(
    obs: &NativeObservation,
    ledger: &Ledger,
    tokens: &mut [f32],
) -> Result<(), String> {
    let round_wind = round_wind_ordinal(obs)?;
    for relative in 0..4u8 {
        let absolute = (u8::from(obs.actor) + relative) % 4;
        let slot = usize::from(absolute);
        let offset = base(SEAT_ROW0 + usize::from(relative));
        if tokens.len() < offset + WIDTH {
            return Err("seat token row outside buffer".to_owned());
        }
        let wind = wind_ordinal(obs, absolute)?;
        let riichi = riichi_ordinal(obs.riichi_states[slot])?;
        let melds = obs.meld_counts[slot];
        if melds > 4 {
            return Err(format!(
                "seat {absolute} exposes {melds} melds, above cap 4"
            ));
        }

        tokens[offset + seat::PRESENT] = 1.0;
        tokens[offset + seat::RELATIVE] = f32::from(relative) / 3.0;
        tokens[offset + seat::WIND] = f32::from(wind) / 3.0;
        tokens[offset + seat::DEALER] = f32::from(u8::from(obs.dealer == absolute));
        tokens[offset + seat::SCORE] = obs.scores[slot] as f32 / 100_000.0;
        tokens[offset + seat::RANK] = f32::from(rank_of(&obs.scores, absolute) - 1) / 3.0;
        tokens[offset + seat::RIICHI] = f32::from(riichi) / 2.0;
        tokens[offset + seat::IPPATSU] = f32::from(u8::from(obs.ippatsu_active[slot]));
        tokens[offset + seat::MELD_COUNT] = melds as f32 / 4.0;
        tokens[offset + seat::LAST_PUBLIC] = norm_position(ledger.seats[slot].last_public);
        tokens[offset + seat::LAST_DISCARD] = norm_position(ledger.seats[slot].last_discard);
        tokens[offset + seat::LAST_CALL] = norm_position(ledger.seats[slot].last_call);
        tokens[offset + seat::RIICHI_DECLARED] = norm_position(ledger.seats[slot].riichi_declared);
        tokens[offset + seat::RIICHI_ACCEPTED] = norm_position(ledger.seats[slot].riichi_accepted);
        tokens[offset + seat::IS_TURN] = f32::from(u8::from(obs.turn_actor == absolute));
        tokens[offset + seat::DEALER_OFFSET] = f32::from((obs.dealer + 4 - absolute) % 4) / 3.0;
        tokens[offset + seat::WIND_OFFSET] = f32::from((wind + 4 - round_wind) % 4) / 3.0;

        if relative == 0 {
            let furiten = furiten_ordinal(obs.actor_furiten)?;
            tokens[offset + seat::FURITEN] = f32::from(furiten) / 3.0;
            tokens[offset + seat::CAN_TSUMO] = f32::from(u8::from(obs.actor_can_tsumo));
            tokens[offset + seat::CAN_RIICHI] = f32::from(u8::from(obs.actor_can_riichi));
        }
        // Every other seat row keeps its actor-only columns zero; the
        // observation does not expose another seat's furiten or legal set.
    }
    Ok(())
}

pub(super) fn write_match_row(obs: &NativeObservation, tokens: &mut [f32]) -> Result<(), String> {
    let offset = base(MATCH_ROW);
    if tokens.len() < offset + WIDTH {
        return Err("match token row outside buffer".to_owned());
    }
    let round_wind = round_wind_ordinal(obs)?;
    let actor_wind = wind_ordinal(obs, obs.actor)?;
    let phase = phase_ordinal(obs.phase)?;
    let best = obs.scores.iter().copied().max().unwrap_or(0);
    let worst = obs.scores.iter().copied().min().unwrap_or(0);
    let actor_rank = rank_of(&obs.scores, obs.actor);

    tokens[offset + match_row::PRESENT] = 1.0;
    tokens[offset + match_row::ROUND_WIND] = f32::from(round_wind) / 3.0;
    tokens[offset + match_row::HAND_NUMBER] = obs.hand_number as f32 / 16.0;
    tokens[offset + match_row::HONBA] = obs.honba as f32 / 8.0;
    tokens[offset + match_row::RIICHI_STICKS] = obs.riichi_sticks as f32 / 4.0;
    tokens[offset + match_row::DEALER_OFFSET] =
        f32::from(relative_seat(obs.dealer, obs.actor)) / 3.0;
    tokens[offset + match_row::LIVE_WALL] = obs.live_wall_remaining as f32 / 70.0;
    tokens[offset + match_row::KAN_COUNT] = obs.kan_count as f32 / 4.0;
    tokens[offset + match_row::PHASE] = f32::from(phase) / 5.0;
    tokens[offset + match_row::ACTOR_DEALER] = f32::from(u8::from(obs.dealer == obs.actor));
    tokens[offset + match_row::ACTOR_WIND] = f32::from(actor_wind) / 3.0;
    tokens[offset + match_row::ROUND_INDEX] = obs.round_index as f32 / 16.0;
    tokens[offset + match_row::SCORE_SPREAD] = (best - worst) as f32 / 100_000.0;
    tokens[offset + match_row::ACTOR_SCORE] = obs.scores[usize::from(obs.actor)] as f32 / 100_000.0;
    tokens[offset + match_row::ACTOR_RANK] = f32::from(actor_rank - 1) / 3.0;
    tokens[offset + match_row::TURN_ACTOR] =
        f32::from(relative_seat(obs.turn_actor, obs.actor)) / 3.0;
    Ok(())
}

fn rank_of(scores: &[i32; 4], seat: u8) -> u8 {
    let own = scores[usize::from(seat)];
    let better = scores.iter().filter(|score| **score > own).count();
    u8::try_from(better + 1).unwrap_or(4)
}

fn wind_ordinal(obs: &NativeObservation, seat: u8) -> Result<u8, String> {
    wind_ordinal_of(obs.seat_winds[usize::from(seat)])
        .map_err(|error| format!("seat wind: {error}"))
}

fn round_wind_ordinal(obs: &NativeObservation) -> Result<u8, String> {
    wind_ordinal_of(obs.round_wind).map_err(|error| format!("round wind: {error}"))
}

fn wind_ordinal_of(tile: i64) -> Result<u8, String> {
    if (27..=30).contains(&tile) {
        Ok((tile - 27) as u8)
    } else {
        Err(format!("wind tile {tile} not in 27..30"))
    }
}

fn riichi_ordinal(state: &str) -> Result<u8, String> {
    match state {
        "none" => Ok(0),
        "declared" => Ok(1),
        "accepted" => Ok(2),
        _ => Err(format!("unknown riichi state {state:?}")),
    }
}

fn phase_ordinal(phase: &str) -> Result<u8, String> {
    match phase {
        "draw_decision" => Ok(1),
        "discard_response" => Ok(2),
        "kan_response" => Ok(3),
        _ => Err(format!("unknown decision phase {phase:?}")),
    }
}

fn furiten_ordinal(state: &str) -> Result<u8, String> {
    match state {
        "none" => Ok(0),
        "temporary" => Ok(1),
        "riichi" => Ok(2),
        _ => Err(format!("unknown furiten state {state:?}")),
    }
}
