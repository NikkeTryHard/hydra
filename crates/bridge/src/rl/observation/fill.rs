//! Per-row plane fill for actor observations.
//!
//! This module writes one validated observation into one row of the plane
//! views produced by the slab splitter. Callers pre-fill the planes with
//! zeroes, except the dora and own-drawn planes which start as all-`0xFF`
//! (the erased byte pattern decodes to the `-1` empty sentinel for those
//! int32 slots); this fill then overwrites every valid prefix entry.
//!
//! Tile-to-kind mapping divides the physical tile id by 4 (34 kinds).
//! Concealed counts are exact per-kind tallies. Visible counts merge rivers
//! and meld tiles and cap each kind at 4, since at most four copies of one
//! kind are observable. History writes the kind id and mask for the visible
//! prefix only; a history longer than the bucket is rejected, never cut.
//! The legal plane copies the full-width mask verbatim.

use super::layout::PlaneBuffers;
use super::{DORA_WIDTH, NativeObservation, ObservationError, furiten_id, phase_id, riichi_id, wind_id};
use crate::rl::actions::BASELINE_ACTIONS;
use crate::rl::events::{EventArena, history_kind_id};

/// Write one little-endian int64 at `offset`, failing closed on overrun.
fn write_i64_le(buf: &mut [u8], offset: usize, value: i64) -> Result<(), ObservationError> {
    let dst = buf.get_mut(offset..offset + 8).ok_or_else(|| {
        ObservationError(format!("plane overrun at offset {offset}"))
    })?;
    dst.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write one little-endian int32 at `offset`, failing closed on overrun.
fn write_i32_le(buf: &mut [u8], offset: usize, value: i32) -> Result<(), ObservationError> {
    let dst = buf.get_mut(offset..offset + 4).ok_or_else(|| {
        ObservationError(format!("plane overrun at offset {offset}"))
    })?;
    dst.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write one boolean byte at `offset`, failing closed on overrun.
fn write_bool(buf: &mut [u8], offset: usize, value: bool) -> Result<(), ObservationError> {
    let dst = buf.get_mut(offset).ok_or_else(|| {
        ObservationError(format!("plane overrun at offset {offset}"))
    })?;
    *dst = u8::from(value);
    Ok(())
}

/// Fill one active row at `row` into the plane buffers. Buffers carry the
/// caller pre-fill described above; this function overwrites the valid
/// entries for the row and writes the uniform sample only when
/// `with_uniform` is set. Unknown phase, furiten, riichi, or wind values,
/// out-of-range concealed kinds, a drifted legal-mask width, or a history
/// longer than the bucket all fail closed.
#[allow(clippy::too_many_arguments)]
pub fn fill_row(
    planes: &mut PlaneBuffers<'_>,
    row: usize,
    obs: &NativeObservation,
    arena: &EventArena,
    bucket_t: usize,
    uniform: f64,
    with_uniform: bool,
) -> Result<(), ObservationError> {
    let b = row;
    write_i64_le(planes.actor, b * 8, i64::from(obs.actor))?;
    write_bool(planes.can_riichi, b, obs.actor_can_riichi)?;
    write_bool(planes.can_tsumo, b, obs.actor_can_tsumo)?;
    write_i64_le(planes.furiten, b * 8, furiten_id(&obs.actor_furiten)?)?;
    write_i64_le(planes.seats, b * 8, i64::from(obs.actor))?;
    write_i64_le(planes.dealer, b * 8, i64::from(obs.dealer))?;
    write_i32_le(planes.hand_number, b * 4, obs.hand_number as i32)?;
    write_i32_le(planes.honba, b * 4, obs.honba as i32)?;
    write_i32_le(planes.kan_count, b * 4, obs.kan_count as i32)?;
    write_i32_le(planes.live_wall, b * 4, obs.live_wall_remaining as i32)?;
    write_i32_le(
        planes.own_drawn,
        b * 4,
        obs.own_drawn_tile.map_or(-1, |t| i32::from(t)),
    )?;
    write_i64_le(planes.phase, b * 8, phase_id(&obs.phase)?)?;
    write_i32_le(planes.riichi_sticks, b * 4, obs.riichi_sticks as i32)?;
    write_i32_le(planes.round_index, b * 4, obs.round_index as i32)?;
    write_i64_le(planes.round_wind, b * 8, wind_id(obs.round_wind)?)?;
    write_i64_le(planes.turn_actor, b * 8, i64::from(obs.turn_actor))?;
    for (j, w) in obs.seat_winds.iter().enumerate() {
        write_i64_le(planes.seat_winds, (b * 4 + j) * 8, wind_id(*w)?)?;
    }
    for (j, s) in obs.riichi_states.iter().enumerate() {
        write_i64_le(planes.riichi_states, (b * 4 + j) * 8, riichi_id(s)?)?;
    }
    for (j, s) in obs.scores.iter().enumerate() {
        write_i32_le(planes.scores, (b * 4 + j) * 4, *s)?;
    }
    for (j, v) in obs.ippatsu_active.iter().enumerate() {
        write_bool(planes.ippatsu, b * 4 + j, *v)?;
    }
    for (j, d) in obs.dora_indicators.iter().enumerate() {
        if j >= DORA_WIDTH {
            break;
        }
        write_i32_le(planes.dora, (b * 5 + j) * 4, *d)?;
    }
    let mut concealed_counts = [0i32; 34];
    for t in &obs.concealed_hand {
        let kind = (*t as usize) / 4;
        *concealed_counts.get_mut(kind).ok_or_else(|| {
            ObservationError(format!("concealed kind {kind} out of range"))
        })? += 1;
    }
    for (j, c) in concealed_counts.iter().enumerate() {
        write_i32_le(planes.concealed, (b * 34 + j) * 4, *c)?;
    }
    let mut visible_counts = [0i32; 34];
    for river in &obs.visible_discards {
        for t in river {
            let kind = (*t as usize) / 4;
            if let Some(slot) = visible_counts.get_mut(kind) {
                *slot += 1;
            }
        }
    }
    for row_melds in &obs.visible_melds {
        for m in row_melds {
            for t in &m.tiles {
                let kind = (*t as usize) / 4;
                if let Some(slot) = visible_counts.get_mut(kind) {
                    *slot += 1;
                }
            }
        }
    }
    for slot in visible_counts.iter_mut() {
        if *slot > 4 {
            *slot = 4;
        }
    }
    for (j, c) in visible_counts.iter().enumerate() {
        write_i32_le(planes.visible, (b * 34 + j) * 4, *c)?;
    }
    for (pos, index) in obs.history_indices.iter().enumerate() {
        if pos >= bucket_t {
            return Err(ObservationError(format!(
                "history {pos} exceeds bucket {bucket_t}"
            )));
        }
        let event = arena.get(*index).ok_or_else(|| {
            ObservationError(format!("history index {index} outside arena"))
        })?;
        write_i64_le(
            planes.history_kind,
            (b * bucket_t + pos) * 8,
            history_kind_id(&event.kind),
        )?;
        write_bool(planes.history_mask, b * bucket_t + pos, true)?;
    }
    if obs.legal_mask.len() != BASELINE_ACTIONS {
        return Err(ObservationError("legal_mask width drift".to_string()));
    }
    for (j, flag) in obs.legal_mask.iter().enumerate() {
        write_bool(planes.legal, b * BASELINE_ACTIONS + j, *flag)?;
    }
    if with_uniform {
        let dst = planes.uniform.get_mut(b * 8..b * 8 + 8).ok_or_else(|| {
            ObservationError("plane overrun at uniform".to_string())
        })?;
        dst.copy_from_slice(&uniform.to_le_bytes());
    }
    Ok(())
}
