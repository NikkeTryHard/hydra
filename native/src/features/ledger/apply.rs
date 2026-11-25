//! Decode visible events into ledger edits and seat timings.

use crate::view::{Observation as NativeObservation, TypedEvent};
use hydra_rules::census::ActionKind;

use super::super::copy_modes::{DiscardEnvelope, Drawn};
use super::super::draw::{DrawState, discard_drawn, own_current};
use super::super::event_payload::{action_kind, actor_of, source_of, tile_of, validate_consumed};
use super::Ledger;

pub(super) fn apply_event(
    ledger: &mut Ledger,
    obs: &NativeObservation,
    draw: &mut DrawState,
    event: &TypedEvent,
    position: u16,
) -> Result<(), String> {
    match event.kind {
        "draw_tile" => {
            let actor = actor_of(event)?;
            let tile = tile_of(event)?;
            draw.observe_draw(obs.actor, actor, tile, position);
        }
        "discard" => {
            let actor = actor_of(event)?;
            let tile = tile_of(event)?;
            let kind = action_kind(event)?;
            if !matches!(
                kind,
                ActionKind::Discard | ActionKind::Tsumogiri | ActionKind::RiichiDiscard
            ) {
                return Err(format!("discard envelope carries action kind {kind:?}"));
            }
            let drawn = Drawn::from(discard_drawn(obs, draw, actor, tile, kind));
            ledger.record_discard(actor, tile, position, kind, drawn, DiscardEnvelope::Discard)?;
            ledger.seats[usize::from(actor)].last_public = Some(position);
            ledger.seats[usize::from(actor)].last_discard = Some(position);
            if actor == obs.actor {
                draw.observe_self_discard(tile);
            }
        }
        "riichi_declared" => {
            let actor = actor_of(event)?;
            let tile = tile_of(event)?;
            if action_kind(event)? != ActionKind::RiichiDiscard {
                return Err("riichi declaration does not carry a riichi action".to_owned());
            }
            let drawn = Drawn::from(discard_drawn(
                obs,
                draw,
                actor,
                tile,
                ActionKind::RiichiDiscard,
            ));
            ledger.record_discard(
                actor,
                tile,
                position,
                ActionKind::RiichiDiscard,
                drawn,
                DiscardEnvelope::RiichiDeclared,
            )?;
            let slot = &mut ledger.seats[usize::from(actor)];
            slot.last_public = Some(position);
            slot.last_discard = Some(position);
            slot.riichi_declared = Some(position);
            if actor == obs.actor {
                draw.observe_self_discard(tile);
            }
        }
        "chi" => claim(ledger, obs, draw, event, position, ActionKind::Chi, 2)?,
        "pon" => claim(ledger, obs, draw, event, position, ActionKind::Pon, 2)?,
        "daiminkan" => {
            claim(ledger, obs, draw, event, position, ActionKind::Daiminkan, 3)?;
        }
        "ankan" => {
            let actor = actor_of(event)?;
            if action_kind(event)? != ActionKind::Ankan {
                return Err("ankan envelope does not carry an ankan action".to_owned());
            }
            let consumed = validate_consumed(event, 4)?;
            let drawn = own_current(obs, draw, actor);
            for tile in consumed {
                ledger.add_meld_copy(
                    actor,
                    *tile,
                    position,
                    ActionKind::Ankan,
                    Drawn::from(drawn == Some(*tile)),
                )?;
            }
            ledger.seats[usize::from(actor)].last_public = Some(position);
            ledger.seats[usize::from(actor)].last_call = Some(position);
            if actor == obs.actor {
                draw.consume_self(consumed);
            }
        }
        "kakan" => {
            let actor = actor_of(event)?;
            let added = tile_of(event)?;
            if action_kind(event)? != ActionKind::Kakan {
                return Err("kakan envelope does not carry a kakan action".to_owned());
            }
            ledger.upgrade_kakan(actor, added, position)?;
            let drawn = own_current(obs, draw, actor).is_some_and(|tile| tile == added);
            ledger.add_meld_copy(
                actor,
                added,
                position,
                ActionKind::Kakan,
                Drawn::from(drawn),
            )?;
            ledger.seats[usize::from(actor)].last_public = Some(position);
            ledger.seats[usize::from(actor)].last_call = Some(position);
            if actor == obs.actor {
                draw.consume_one(added);
            }
        }
        "dora_revealed" => {
            let tile = tile_of(event)?;
            ledger.add_indicator(tile, position)?;
        }
        "riichi_accepted" => {
            let actor = actor_of(event)?;
            ledger.seats[usize::from(actor)].last_public = Some(position);
            ledger.seats[usize::from(actor)].riichi_accepted = Some(position);
        }
        "turn_advance" => {
            let actor = actor_of(event)?;
            ledger.seats[usize::from(actor)].last_public = Some(position);
        }
        "tsumo" | "ron" => {
            let actor = actor_of(event)?;
            ledger.seats[usize::from(actor)].last_public = Some(position);
            if actor == obs.actor {
                draw.reset();
            }
        }
        "round_start" | "game_start" | "round_end" | "draw_end" | "abortive_draw" | "game_end" => {
            draw.reset();
        }
        _ => {}
    }
    Ok(())
}

fn claim(
    ledger: &mut Ledger,
    obs: &NativeObservation,
    draw: &mut DrawState,
    event: &TypedEvent,
    position: u16,
    expected: ActionKind,
    consumed_len: usize,
) -> Result<(), String> {
    let actor = actor_of(event)?;
    let called = tile_of(event)?;
    let source = source_of(event, actor)?;
    if action_kind(event)? != expected {
        return Err(format!(
            "{} envelope does not carry the matching canonical action",
            event.kind
        ));
    }
    let consumed = validate_consumed(event, consumed_len)?;
    ledger.claim_called(actor, called, source, position, expected)?;
    let drawn = own_current(obs, draw, actor);
    for tile in consumed {
        ledger.add_meld_copy(
            actor,
            *tile,
            position,
            expected,
            Drawn::from(drawn == Some(*tile)),
        )?;
    }
    ledger.seats[usize::from(actor)].last_public = Some(position);
    ledger.seats[usize::from(actor)].last_call = Some(position);
    if actor == obs.actor {
        draw.consume_self(consumed);
    }
    Ok(())
}
