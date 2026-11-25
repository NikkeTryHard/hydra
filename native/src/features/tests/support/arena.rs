//! Synthetic typed-event arena fixtures.

use crate::view::{EventArena, TypedEvent, Visibility};

pub(crate) struct Log {
    arena: EventArena,
}

impl Log {
    pub(crate) fn new() -> Self {
        let mut log = Self {
            arena: EventArena::new(),
        };
        log.push(event("game_start"));
        log
    }

    pub(crate) fn push(&mut self, event: TypedEvent) -> usize {
        let index = self.arena.len();
        self.arena.push(event);
        index
    }

    pub(crate) fn arena(&self) -> &EventArena {
        &self.arena
    }
}

fn event(kind: &'static str) -> TypedEvent {
    TypedEvent::public(kind, None)
}

pub(crate) fn draw(actor: u8, tile: u8) -> TypedEvent {
    let mut event = event("draw_tile");
    event.actor = Some(actor);
    event.visibility = Visibility::ActorPrivate;
    event.tile = Some(tile);
    event
}

pub(crate) fn discard(actor: u8, tile: u8, action_id: u32) -> TypedEvent {
    let mut event = event("discard");
    event.actor = Some(actor);
    event.tile = Some(tile);
    event.action_id = Some(action_id);
    event
}

pub(crate) fn riichi_declared(actor: u8, tile: u8, action_id: u32) -> TypedEvent {
    let mut event = event("riichi_declared");
    event.actor = Some(actor);
    event.tile = Some(tile);
    event.action_id = Some(action_id);
    event
}

pub(crate) fn meld(
    kind: &'static str,
    actor: u8,
    called: u8,
    source: u8,
    consumed: &[u8],
    action_id: u32,
) -> TypedEvent {
    let mut event = event(kind);
    event.actor = Some(actor);
    event.tile = Some(called);
    event.action_id = Some(action_id);
    event.source_seat = Some(source);
    event.consumed_tiles = consumed.to_vec();
    event
}

pub(crate) fn ankan(actor: u8, consumed: &[u8], action_id: u32) -> TypedEvent {
    let mut event = event("ankan");
    event.actor = Some(actor);
    event.action_id = Some(action_id);
    event.consumed_tiles = consumed.to_vec();
    event
}

pub(crate) fn kakan(actor: u8, added: u8, action_id: u32) -> TypedEvent {
    let mut event = event("kakan");
    event.actor = Some(actor);
    event.tile = Some(added);
    event.action_id = Some(action_id);
    event
}

pub(crate) fn dora(tile: u8) -> TypedEvent {
    let mut event = event("dora_revealed");
    event.tile = Some(tile);
    event
}

pub(crate) fn call_window() -> TypedEvent {
    event("call_window")
}

pub(crate) fn hidden_call() -> TypedEvent {
    let mut event = event("call_resolved");
    event.visibility = Visibility::ServerPrivate;
    event
}
