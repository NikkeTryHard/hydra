//! Canonical legal-action descriptors: identity, source, primary tile, edit.

use crate::view::Observation as NativeObservation;
use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::{BASELINE_ACTIONS, CanonicalAction};

use super::layout::{desc, is_red, relative_source, tile_face};
use super::ledger::Ledger;

pub(super) fn write_candidates(
    obs: &NativeObservation,
    ledger: &Ledger,
    legal: &[(u32, CanonicalAction)],
    candidates: &mut [i32],
) -> Result<(), String> {
    for (index, (id, action)) in legal.iter().enumerate() {
        let offset = index * super::ACTION;
        let row = &mut candidates[offset..offset + super::ACTION];
        row.fill(-1);
        if *id as usize >= BASELINE_ACTIONS {
            return Err(format!("canonical action id {id} outside legal table"));
        }
        if action.actor != obs.actor {
            return Err(format!(
                "action {id} actor {} != observing actor {}",
                action.actor, obs.actor
            ));
        }
        row[desc::ID] = *id as i32;
        row[desc::KIND] = i32::from(action.kind.ordinal());
        row[desc::SOURCE] = i32::from(relative_source(action.source, obs.actor)?);
        row[desc::RIICHI] = i32::from(u8::from(
            action.declares_riichi || action.kind == ActionKind::RiichiDiscard,
        ));
        row[desc::RED] = 0;

        if let Some(tile) = primary_tile(action)? {
            let face = tile_face(tile)?;
            row[desc::FACE] = face as i32;
            row[desc::RED] = i32::from(u8::from(is_red(tile)));
        }
        row[desc::FROM_DRAW] = i32::from(u8::from(from_draw(ledger, action)?));
        row[desc::CONTESTED] = i32::from(u8::from(is_contested(action.kind)));

        let removals = proposed_removals(action)?;
        for (slot, removal) in removals.iter().enumerate() {
            if let Some(tile) = removal {
                let face = tile_face(*tile)?;
                row[desc::REMOVALS + slot] = face as i32;
                row[desc::REMOVAL_RED + slot] = i32::from(u8::from(is_red(*tile)));
            } else {
                row[desc::REMOVAL_RED + slot] = 0;
            }
        }
    }
    Ok(())
}

fn primary_tile(action: &CanonicalAction) -> Result<Option<u8>, String> {
    match action.kind {
        ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan => action
            .called_tile
            .ok_or_else(|| format!("{} action lacks called tile", action.kind.name()))
            .map(Some),
        ActionKind::Discard
        | ActionKind::Tsumogiri
        | ActionKind::RiichiDiscard
        | ActionKind::Kakan
        | ActionKind::Ron
        | ActionKind::Tsumo => action
            .tile
            .ok_or_else(|| format!("{} action lacks primary tile", action.kind.name()))
            .map(Some),
        ActionKind::Ankan
        | ActionKind::Pass
        | ActionKind::AbortNineTerminals
        | ActionKind::AcceptAbortiveDraw => Ok(None),
    }
}

fn proposed_removals(action: &CanonicalAction) -> Result<[Option<u8>; 4], String> {
    let mut removals = [None; 4];
    match action.kind {
        ActionKind::Discard
        | ActionKind::Tsumogiri
        | ActionKind::RiichiDiscard
        | ActionKind::Kakan => {
            let tile = action
                .tile
                .ok_or_else(|| format!("{} action lacks removed tile", action.kind.name()))?;
            removals[0] = Some(tile);
        }
        ActionKind::Chi | ActionKind::Pon => {
            fill_consumed(&mut removals, action, 2)?;
        }
        ActionKind::Daiminkan => {
            fill_consumed(&mut removals, action, 3)?;
        }
        ActionKind::Ankan => {
            fill_consumed(&mut removals, action, 4)?;
        }
        ActionKind::Ron
        | ActionKind::Tsumo
        | ActionKind::Pass
        | ActionKind::AbortNineTerminals
        | ActionKind::AcceptAbortiveDraw => {}
    }
    Ok(removals)
}

fn fill_consumed(
    removals: &mut [Option<u8>; 4],
    action: &CanonicalAction,
    expected: usize,
) -> Result<(), String> {
    if action.consumed.len() != expected {
        return Err(format!(
            "{} action consumes {} tiles, expected {expected}",
            action.kind.name(),
            action.consumed.len()
        ));
    }
    let mut tiles = [0u8; 4];
    tiles[..expected].copy_from_slice(&action.consumed);
    tiles[..expected].sort_unstable();
    for (slot, tile) in tiles[..expected].iter().enumerate() {
        tile_face(*tile)?;
        removals[slot] = Some(*tile);
    }
    Ok(())
}

fn from_draw(ledger: &Ledger, action: &CanonicalAction) -> Result<bool, String> {
    match action.kind {
        ActionKind::Tsumogiri => Ok(true),
        ActionKind::Discard => Ok(false),
        ActionKind::RiichiDiscard => Ok(ledger
            .drawn
            .is_some_and(|(tile, _)| Some(tile) == action.tile)),
        ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan | ActionKind::Ron => {
            let tile = primary_tile(action)?
                .ok_or_else(|| format!("{} action lacks public tile", action.kind.name()))?;
            let face = tile_face(tile)?;
            Ok(ledger.faces[face]
                .find(tile)
                .is_some_and(|record| record.drawn))
        }
        _ => Ok(false),
    }
}

fn is_contested(kind: ActionKind) -> bool {
    matches!(
        kind,
        ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan | ActionKind::Ron
    )
}
