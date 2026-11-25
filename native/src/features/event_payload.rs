//! Decode and validate the closed payload fields used by the ledger.

use crate::view::TypedEvent;
use hydra_rules::census::{ActionKind, frozen_census};

use super::layout::tile_face;

pub(super) fn actor_of(event: &TypedEvent) -> Result<u8, String> {
    let actor = event
        .actor
        .ok_or_else(|| format!("{} event without actor", event.kind))?;
    if actor > 3 {
        return Err(format!("{} actor {actor} outside 0..4", event.kind));
    }
    Ok(actor)
}

pub(super) fn tile_of(event: &TypedEvent) -> Result<u8, String> {
    let tile = event
        .tile
        .ok_or_else(|| format!("{} event without tile", event.kind))?;
    tile_face(tile)?;
    Ok(tile)
}

pub(super) fn source_of(event: &TypedEvent, actor: u8) -> Result<u8, String> {
    let source = event
        .source_seat
        .ok_or_else(|| format!("{} event without source seat", event.kind))?;
    if source > 3 || source == actor {
        return Err(format!(
            "{} event source {source} is invalid for actor {actor}",
            event.kind
        ));
    }
    Ok(source)
}

pub(super) fn action_kind(event: &TypedEvent) -> Result<ActionKind, String> {
    let id = event
        .action_id
        .ok_or_else(|| format!("{} event without canonical action id", event.kind))?;
    let census = frozen_census();
    census
        .get(id as usize)
        .map(|template| template.kind)
        .ok_or_else(|| format!("canonical action id {id} outside census {}", census.len()))
}

pub(super) fn validate_consumed(event: &TypedEvent, expected: usize) -> Result<&[u8], String> {
    if event.consumed_tiles.len() != expected {
        return Err(format!(
            "{} must consume exactly {expected} hand tiles",
            event.kind
        ));
    }
    for tile in &event.consumed_tiles {
        tile_face(*tile)?;
    }
    Ok(&event.consumed_tiles)
}
