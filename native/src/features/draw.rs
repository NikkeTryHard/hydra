//! The actor's own drawn-tile cache, kept current from visible events.

use crate::view::Observation as NativeObservation;
use hydra_rules::census::ActionKind;

#[derive(Clone, Copy, Default)]
pub(super) struct DrawState {
    current: Option<(u8, u16)>,
    last_is_draw: bool,
}

impl DrawState {
    pub(super) fn observe_draw(&mut self, self_actor: u8, actor: u8, tile: u8, position: u16) {
        if actor == self_actor {
            self.current = Some((tile, position));
            self.last_is_draw = true;
        }
    }

    pub(super) fn observe_self_discard(&mut self, tile: u8) {
        if self.current.is_some_and(|(drawn, _)| drawn == tile) {
            self.current = None;
        }
        self.last_is_draw = false;
    }

    pub(super) fn consume_self(&mut self, consumed: &[u8]) {
        if self
            .current
            .is_some_and(|(drawn, _)| consumed.contains(&drawn))
        {
            self.current = None;
        }
        self.last_is_draw = false;
    }

    pub(super) fn consume_one(&mut self, tile: u8) {
        if self.current.is_some_and(|(drawn, _)| drawn == tile) {
            self.current = None;
        }
        self.last_is_draw = false;
    }

    pub(super) fn reset(&mut self) {
        self.current = None;
        self.last_is_draw = false;
    }
}

/// The current drawn copy, accepted only at a genuine draw decision.
pub(super) fn current_drawn(obs: &NativeObservation, draw: &DrawState) -> Option<(u8, u16)> {
    if obs.phase != "draw_decision" || !draw.last_is_draw {
        return None;
    }
    draw.current
        .filter(|(tile, _)| obs.own_drawn_tile == Some(*tile) && !obs.concealed_hand.contains(tile))
}

/// A cached draw still in hand after a tedashi discard or call.
///
/// A retained copy absent from the concealed list is still counted as
/// concealed after a call, never as a new draw.
pub(super) fn kept_draw(obs: &NativeObservation, draw: &DrawState) -> Option<(u8, u16)> {
    if obs.phase != "draw_decision" || draw.last_is_draw {
        return None;
    }
    let (tile, position) = draw.current?;
    if obs.own_drawn_tile != Some(tile) || obs.concealed_hand.contains(&tile) {
        return None;
    }
    Some((tile, position))
}

pub(super) fn own_current(obs: &NativeObservation, draw: &DrawState, actor: u8) -> Option<u8> {
    if actor == obs.actor {
        draw.current.map(|(tile, _)| tile)
    } else {
        None
    }
}

pub(super) fn discard_drawn(
    obs: &NativeObservation,
    draw: &DrawState,
    actor: u8,
    tile: u8,
    kind: ActionKind,
) -> bool {
    match kind {
        ActionKind::Tsumogiri => true,
        ActionKind::Discard => false,
        ActionKind::RiichiDiscard => {
            own_current(obs, draw, actor).is_some_and(|drawn| drawn == tile)
        }
        _ => false,
    }
}
