//! Actor-visible copy ledger for the current hand.
//!
//! `build` scans only the current hand's visible history indices, capped at
//! `HISTORY_EVENT_CAP`. It never walks the whole arena. The public
//! `call_window` marker is skipped because its presence depends on whether
//! another seat has a legal response; it therefore consumes no provenance
//! position and cannot shift a feature value. Server-private events are
//! rejected by the visibility predicate if a malformed observation names one.

mod apply;
mod copies;

use crate::view::{HISTORY_EVENT_CAP, Snapshot};

use super::draw::{DrawState, current_drawn, kept_draw};
use super::records::{FaceCopies, ORIGIN_OWN_CONCEALED, ORIGIN_OWN_DRAWN};

#[derive(Clone, Copy, Default)]
pub(super) struct SeatTiming {
    pub last_public: Option<u16>,
    pub last_discard: Option<u16>,
    pub last_call: Option<u16>,
    pub riichi_declared: Option<u16>,
    pub riichi_accepted: Option<u16>,
}

pub(super) struct Ledger {
    pub faces: [FaceCopies; 34],
    pub seats: [SeatTiming; 4],
    pub drawn: Option<(u8, u16)>,
}

impl Ledger {
    fn new() -> Self {
        Self {
            faces: std::array::from_fn(|_| FaceCopies::default()),
            seats: [SeatTiming::default(); 4],
            drawn: None,
        }
    }
}

pub(super) fn build(snapshot: &Snapshot<'_>) -> Result<Ledger, String> {
    let obs = &snapshot.observation;
    if obs.actor > 3 {
        return Err(format!("actor {} outside 0..4", obs.actor));
    }
    for indicator in &obs.dora_indicators {
        if *indicator < -1 || *indicator > 135 {
            return Err(format!("dora indicator {indicator} outside -1..=135"));
        }
    }
    if obs.history_indices.len() > HISTORY_EVENT_CAP {
        return Err(format!(
            "visible history {} exceeds cap {HISTORY_EVENT_CAP}",
            obs.history_indices.len()
        ));
    }

    let mut ledger = Ledger::new();
    let mut draw = DrawState::default();
    let mut provenance_position = 0u16;
    for index in &obs.history_indices {
        let event = snapshot
            .events
            .get(*index)
            .ok_or_else(|| format!("history index {index} outside event arena"))?;
        if !event.visible_to(obs.actor) {
            return Err(format!(
                "history index {index} holds a {} event hidden from actor {}",
                event.kind, obs.actor
            ));
        }
        if event.kind == "call_window" {
            continue;
        }
        apply::apply_event(&mut ledger, obs, &mut draw, event, provenance_position)?;
        provenance_position = provenance_position
            .checked_add(1)
            .ok_or_else(|| "visible history exceeds u16 positions".to_owned())?;
    }

    let current = current_drawn(obs, &draw);
    let kept = kept_draw(obs, &draw);
    for tile in &obs.concealed_hand {
        ledger.add_own(obs.actor, *tile, ORIGIN_OWN_CONCEALED, None)?;
    }
    if let Some((tile, position)) = current {
        ledger.add_own(obs.actor, tile, ORIGIN_OWN_DRAWN, Some(position))?;
    } else if let Some((tile, _)) = kept {
        ledger.add_own(obs.actor, tile, ORIGIN_OWN_CONCEALED, None)?;
    }
    ledger.drawn = current;
    Ok(ledger)
}
