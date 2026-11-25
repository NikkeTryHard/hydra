//! Per-hand copy reconciliation: own, discard, claim, kan, indicator.

use hydra_rules::census::ActionKind;

use super::super::copy_modes::{DiscardEnvelope, Drawn};
use super::super::layout::{ABSENT_SEAT, is_red, tile_face};
use super::super::records::{
    CopyRecord, KIND_DORA, KIND_PRIVATE, ORIGIN_DISCARD, ORIGIN_INDICATOR, ORIGIN_MELD,
    ORIGIN_OWN_DRAWN,
};
use super::Ledger;

impl Ledger {
    pub(super) fn add_own(
        &mut self,
        actor: u8,
        tile: u8,
        origin: u8,
        position: Option<u16>,
    ) -> Result<(), String> {
        let face = tile_face(tile)?;
        let record = CopyRecord {
            origin,
            owner: actor,
            source: ABSENT_SEAT,
            exposed: position,
            call: None,
            red: is_red(tile),
            drawn: origin == ORIGIN_OWN_DRAWN,
            kind: KIND_PRIVATE,
            copy: tile,
            discard_seen: false,
            riichi_seen: false,
        };
        self.faces[face].push(record)
    }

    pub(super) fn record_discard(
        &mut self,
        actor: u8,
        tile: u8,
        position: u16,
        kind: ActionKind,
        drawn: Drawn,
        envelope: DiscardEnvelope,
    ) -> Result<(), String> {
        let drawn_flag = drawn.is_drawn();
        let discard_event = matches!(envelope, DiscardEnvelope::Discard);
        let riichi_event = matches!(envelope, DiscardEnvelope::RiichiDeclared);
        let face = tile_face(tile)?;
        let copies = &mut self.faces[face];
        if let Some(record) = copies.find_mut(tile) {
            if record.origin != ORIGIN_DISCARD {
                return Err(format!(
                    "discard of tile {tile} collides with existing public provenance"
                ));
            }
            if discard_event {
                if record.discard_seen {
                    return Err(format!("duplicate discard envelope for tile {tile}"));
                }
                record.discard_seen = true;
            }
            if riichi_event {
                if record.riichi_seen {
                    return Err(format!(
                        "duplicate riichi declaration envelope for tile {tile}"
                    ));
                }
                record.riichi_seen = true;
            }
            record.drawn |= drawn_flag;
            if riichi_event || kind == ActionKind::RiichiDiscard {
                record.kind = ActionKind::RiichiDiscard.ordinal();
            } else if !record.riichi_seen {
                record.kind = kind.ordinal();
            }
            return Ok(());
        }
        let record = CopyRecord {
            origin: ORIGIN_DISCARD,
            owner: actor,
            source: ABSENT_SEAT,
            exposed: Some(position),
            call: None,
            red: is_red(tile),
            drawn: drawn_flag,
            kind: if riichi_event || kind == ActionKind::RiichiDiscard {
                ActionKind::RiichiDiscard.ordinal()
            } else {
                kind.ordinal()
            },
            copy: tile,
            discard_seen: discard_event,
            riichi_seen: riichi_event,
        };
        copies.push(record)
    }

    pub(super) fn claim_called(
        &mut self,
        actor: u8,
        tile: u8,
        source: u8,
        position: u16,
        kind: ActionKind,
    ) -> Result<(), String> {
        let face = tile_face(tile)?;
        let record = self.faces[face]
            .find_mut(tile)
            .ok_or_else(|| format!("claim of tile {tile} without a visible discard envelope"))?;
        if record.origin != ORIGIN_DISCARD {
            return Err(format!(
                "claim of tile {tile} does not reference an unclaimed discard"
            ));
        }
        record.origin = ORIGIN_MELD;
        record.owner = actor;
        record.source = source;
        record.call = Some(position);
        record.kind = kind.ordinal();
        Ok(())
    }

    pub(super) fn add_meld_copy(
        &mut self,
        actor: u8,
        tile: u8,
        position: u16,
        kind: ActionKind,
        drawn: Drawn,
    ) -> Result<(), String> {
        let face = tile_face(tile)?;
        let record = CopyRecord {
            origin: ORIGIN_MELD,
            owner: actor,
            source: ABSENT_SEAT,
            exposed: Some(position),
            call: Some(position),
            red: is_red(tile),
            drawn: drawn.is_drawn(),
            kind: kind.ordinal(),
            copy: tile,
            discard_seen: false,
            riichi_seen: false,
        };
        self.faces[face].push(record)
    }

    pub(super) fn upgrade_kakan(
        &mut self,
        actor: u8,
        added: u8,
        position: u16,
    ) -> Result<(), String> {
        let face = tile_face(added)?;
        let mut found = 0usize;
        for record in self.faces[face].iter_mut() {
            if record.origin == ORIGIN_MELD
                && record.owner == actor
                && record.kind == ActionKind::Pon.ordinal()
            {
                record.kind = ActionKind::Kakan.ordinal();
                record.call = Some(position);
                found += 1;
            }
        }
        if found != 3 {
            return Err(format!(
                "kakan of tile {added}: found {found} prior pon copies, expected 3"
            ));
        }
        Ok(())
    }

    pub(super) fn add_indicator(&mut self, tile: u8, position: u16) -> Result<(), String> {
        let face = tile_face(tile)?;
        let record = CopyRecord {
            origin: ORIGIN_INDICATOR,
            owner: ABSENT_SEAT,
            source: ABSENT_SEAT,
            exposed: Some(position),
            call: None,
            red: is_red(tile),
            drawn: false,
            kind: KIND_DORA,
            copy: tile,
            discard_seen: false,
            riichi_seen: false,
        };
        self.faces[face].push(record)
    }
}
