//! Fixed-capacity per-face copy records and the tile-row renderer.

use super::WIDTH;
use super::layout::{ABSENT_SEAT, TILE_ROWS, base, is_red, norm_position, relative_seat, tile_face};

pub(super) const DORA_SLOTS: usize = 5;
pub(super) const RECORDS: usize = 4;
pub(super) const RECORD_WIDTH: usize = 9;

pub(super) mod tile {
    pub(crate) const OWN_COUNT: usize = 0;
    pub(crate) const OWN_RED: usize = 1;
    pub(crate) const DRAWN_FACE: usize = 2;
    pub(crate) const DRAWN_RED: usize = 3;
    pub(crate) const PUBLIC_COUNT: usize = 4;
    pub(crate) const PUBLIC_RED: usize = 5;
    pub(crate) const DORA0: usize = 6;
    pub(crate) const RECORD0: usize = 11;
    pub(crate) const RESERVED: usize = 47;
}

pub(super) const ORIGIN_OWN_CONCEALED: u8 = 1;
pub(super) const ORIGIN_OWN_DRAWN: u8 = 2;
pub(super) const ORIGIN_DISCARD: u8 = 3;
pub(super) const ORIGIN_MELD: u8 = 4;
pub(super) const ORIGIN_INDICATOR: u8 = 5;

pub(super) const KIND_PRIVATE: u8 = 0;
pub(super) const KIND_DORA: u8 = 13;

#[derive(Clone, Copy)]
pub(super) struct CopyRecord {
    pub origin: u8,
    pub owner: u8,
    pub source: u8,
    pub exposed: Option<u16>,
    pub call: Option<u16>,
    pub red: bool,
    pub drawn: bool,
    pub kind: u8,
    pub copy: u8,
    pub discard_seen: bool,
    pub riichi_seen: bool,
}

pub(super) const EMPTY: CopyRecord = CopyRecord {
    origin: 0,
    owner: ABSENT_SEAT,
    source: ABSENT_SEAT,
    exposed: None,
    call: None,
    red: false,
    drawn: false,
    kind: KIND_PRIVATE,
    copy: 0,
    discard_seen: false,
    riichi_seen: false,
};

#[derive(Clone)]
pub(super) struct FaceCopies {
    records: [CopyRecord; RECORDS],
    len: u8,
}

impl Default for FaceCopies {
    fn default() -> Self {
        Self {
            records: [EMPTY; RECORDS],
            len: 0,
        }
    }
}

impl FaceCopies {
    pub(super) fn find(&self, copy: u8) -> Option<&CopyRecord> {
        self.iter().find(|record| record.copy == copy)
    }

    pub(super) fn find_mut(&mut self, copy: u8) -> Option<&mut CopyRecord> {
        self.iter_mut().find(|record| record.copy == copy)
    }

    pub(super) fn push(&mut self, record: CopyRecord) -> Result<(), String> {
        if self.len as usize >= RECORDS {
            return Err(format!(
                "face {} has more than {RECORDS} visible copies",
                record.copy / 4
            ));
        }
        if self.iter().any(|existing| existing.copy == record.copy) {
            return Err(format!(
                "physical copy {} appears twice on face {}",
                record.copy,
                record.copy / 4
            ));
        }
        self.records[self.len as usize] = record;
        self.len += 1;
        Ok(())
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &CopyRecord> {
        self.records[..self.len as usize].iter()
    }

    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut CopyRecord> {
        self.records[..self.len as usize].iter_mut()
    }

    fn own_count(&self) -> usize {
        self.iter()
            .filter(|record| matches!(record.origin, ORIGIN_OWN_CONCEALED | ORIGIN_OWN_DRAWN))
            .count()
    }

    fn own_red(&self) -> bool {
        self.iter().any(|record| {
            record.red && matches!(record.origin, ORIGIN_OWN_CONCEALED | ORIGIN_OWN_DRAWN)
        })
    }

    pub(super) fn public_count(&self) -> usize {
        self.iter()
            .filter(|record| record.origin >= ORIGIN_DISCARD)
            .count()
    }

    fn public_red(&self) -> usize {
        self.iter()
            .filter(|record| record.red && record.origin >= ORIGIN_DISCARD)
            .count()
    }
}

fn sort_key(record: &CopyRecord) -> (u32, u32, u8, u8, u8, u8, bool, bool) {
    (
        record.exposed.map_or(0, |position| u32::from(position) + 1),
        record.call.map_or(0, |position| u32::from(position) + 1),
        record.origin,
        record.owner,
        record.source,
        record.kind,
        record.drawn,
        record.red,
    )
}

fn relative_owner(owner: u8, actor: u8) -> u8 {
    if owner < 4 {
        relative_seat(owner, actor)
    } else {
        ABSENT_SEAT
    }
}

pub(super) fn write_tile_row(
    face: usize,
    copies: &FaceCopies,
    actor: u8,
    drawn: Option<(u8, u16)>,
    dora: &[i32; DORA_SLOTS],
    tokens: &mut [f32],
) -> Result<(), String> {
    if face >= TILE_ROWS {
        return Err(format!("tile row {face} outside 0..34"));
    }
    let row = base(face);
    if tokens.len() < row + WIDTH {
        return Err("token row outside buffer".to_owned());
    }
    let own_count = copies.own_count();
    let own_red = copies.own_red();
    let public_count = copies.public_count();
    let public_red = copies.public_red();
    let drawn_face = drawn.is_some_and(|(tile, _)| usize::from(tile / 4) == face);
    let drawn_red = drawn.is_some_and(|(tile, _)| is_red(tile));

    tokens[row + tile::OWN_COUNT] = own_count as f32;
    tokens[row + tile::OWN_RED] = f32::from(u8::from(own_red));
    tokens[row + tile::DRAWN_FACE] = f32::from(u8::from(drawn_face));
    tokens[row + tile::DRAWN_RED] = f32::from(u8::from(drawn_red));
    tokens[row + tile::PUBLIC_COUNT] = public_count as f32;
    tokens[row + tile::PUBLIC_RED] = public_red as f32;
    tokens[row + tile::RESERVED] = 0.0;

    for (slot, indicator) in dora.iter().enumerate() {
        if *indicator >= 0 {
            let indicator_face = tile_face(*indicator as u8)?;
            if indicator_face == face {
                tokens[row + tile::DORA0 + slot] = 1.0;
            }
        }
    }

    let mut order = [0usize, 1, 2, 3];
    let count = copies.len as usize;
    for index in 1..count {
        let mut cursor = index;
        while cursor > 0
            && sort_key(&copies.records[order[cursor - 1]])
                > sort_key(&copies.records[order[cursor]])
        {
            order.swap(cursor - 1, cursor);
            cursor -= 1;
        }
    }

    for slot in 0..RECORDS {
        let offset = row + tile::RECORD0 + slot * RECORD_WIDTH;
        let Some(record) = (slot < count).then(|| &copies.records[order[slot]]) else {
            for field in 0..RECORD_WIDTH {
                tokens[offset + field] = 0.0;
            }
            continue;
        };
        tokens[offset] = 1.0;
        tokens[offset + 1] = f32::from(record.origin);
        tokens[offset + 2] = f32::from(relative_owner(record.owner, actor));
        tokens[offset + 3] = f32::from(if record.source < 4 {
            relative_seat(record.source, actor)
        } else {
            ABSENT_SEAT
        });
        tokens[offset + 4] = norm_position(record.exposed);
        tokens[offset + 5] = norm_position(record.call);
        tokens[offset + 6] = f32::from(u8::from(record.red));
        tokens[offset + 7] = f32::from(u8::from(record.drawn));
        tokens[offset + 8] = f32::from(record.kind);
    }
    Ok(())
}
