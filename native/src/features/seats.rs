//! Seat-relative and match token rows.

use crate::view::Observation as NativeObservation;

use super::WIDTH;
use super::layout::{
    MATCH_ROW, SEAT_ROW0, base, furiten_ordinal, match_row, norm_position, phase_ordinal, rank_of,
    relative_seat, riichi_ordinal, round_wind_ordinal, seat, wind_ordinal,
};
use super::ledger::Ledger;

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
