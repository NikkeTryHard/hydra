//! Actor features for the tile-edit Transformer.
//!
//! `encode` writes one 39x48 token block and one 16-column descriptor per
//! legal action. It reads only the acting seat's [`NativeObservation`] and the
//! events named by `obs.history_indices`; server-private events, hidden hands,
//! wall order, and absolute event sequence numbers never influence a value.
//!
//! # Token rows
//!
//! - `0..34`: tile faces, where face = physical tile id / 4.
//! - `34..38`: seats relative to the actor in turn order: 0 self, 1 next
//!   seat, 2 opposite, 3 previous seat. Row `34 + relative_seat`.
//! - `38`: match token.
//!
//! # Tile row columns
//!
//! | columns | meaning |
//! | --- | --- |
//! | 0 | own current count including the current drawn tile (raw 0..4) |
//! | 1 | own red copy held (0/1) |
//! | 2 | current drawn tile has this face (0/1) |
//! | 3 | current drawn tile is this red copy (0/1) |
//! | 4 | public visible copies: unclaimed discards, meld tiles, revealed indicators; own concealed copies excluded |
//! | 5 | public visible red copies |
//! | 6..11 | revealed dora indicator slots 0..4, one-hot on the indicator face |
//! | 11..47 | four copy records of nine fields each |
//! | 47 | reserved, zero |
//!
//! Copy record fields, in order:
//!
//! | field | meaning |
//! | --- | --- |
//! | 0 | present (0/1) |
//! | 1 | origin: 1 own concealed, 2 own drawn, 3 public discard, 4 public meld, 5 dora indicator |
//! | 2 | owner seat relative to the actor (0..3), or 4 for the table |
//! | 3 | source seat relative to the actor (0..3), or 4 when absent |
//! | 4 | normalized position of first actor-visible exposure; 0 for a concealed copy |
//! | 5 | normalized position of the meld/call that exposed the copy; 0 when none |
//! | 6 | red copy (0/1) |
//! | 7 | shown as drawn (0/1); for a called discard this is the discard's public drawn/held flag |
//! | 8 | exposing kind: 0 own/private, 1..12 frozen `ActionKind` ordinal, 13 dora indicator |
//!
//! Records are sorted by actor-visible provenance: exposure position, call
//! position, origin, owner, source, kind, drawn flag, red flag. Physical copy
//! ids are used only to reconcile the same copy across events and are never
//! rendered. Unused record slots are entirely zero.
//!
//! # Seat row columns
//!
//! | columns | meaning |
//! | --- | --- |
//! | 0 | present (always 1) |
//! | 1 | relative seat / 3 |
//! | 2 | seat wind: east..north as 0..3, divided by 3 |
//! | 3 | dealer (0/1) |
//! | 4 | score / 100000 |
//! | 5 | competitive rank minus one, divided by 3 |
//! | 6 | riichi state: none/declared/accepted as 0..2, divided by 2 |
//! | 7 | ippatsu active (0/1) |
//! | 8 | visible meld count / 4 |
//! | 9 | last public action position |
//! | 10 | last discard position |
//! | 11 | last call position |
//! | 12 | riichi declaration position |
//! | 13 | riichi acceptance position |
//! | 14 | this seat is the turn actor (0/1) |
//! | 15 | actor-only furiten state / 3; zero for every other seat |
//! | 16 | actor-only can-tsumo flag; zero for every other seat |
//! | 17 | actor-only can-riichi flag; zero for every other seat |
//! | 18 | offset from this seat to the dealer / 3 |
//! | 19 | offset from this seat wind to the round wind / 3 |
//! | 20..48 | reserved, zero |
//!
//! # Match row columns
//!
//! | columns | meaning |
//! | --- | --- |
//! | 0 | present (always 1) |
//! | 1 | round wind / 3 |
//! | 2 | hand number / 16 |
//! | 3 | honba / 8 |
//! | 4 | riichi sticks / 4 |
//! | 5 | dealer offset from actor / 3 |
//! | 6 | live wall remaining / 70 |
//! | 7 | kan count / 4 |
//! | 8 | phase id 0..5, divided by 5 |
//! | 9 | actor is dealer (0/1) |
//! | 10 | actor seat wind / 3 |
//! | 11 | round index / 16 |
//! | 12 | score spread / 100000 |
//! | 13 | actor score / 100000 |
//! | 14 | actor competitive rank minus one, divided by 3 |
//! | 15 | turn actor offset from actor / 3 |
//! | 16..48 | reserved, zero |
//!
//! # Legal descriptor columns
//!
//! | columns | meaning |
//! | --- | --- |
//! | 0 | canonical action id (external id; mask/apply only) |
//! | 1 | frozen `ActionKind` ordinal 0..12 |
//! | 2 | source seat relative to the actor (0..3), or 4 absent |
//! | 3 | primary tile face, or -1 |
//! | 4 | primary tile is red (0/1) |
//! | 5 | public discard-from-draw flag when known (0/1) |
//! | 6 | declares riichi (0/1) |
//! | 7..11 | up to four proposed actor-hand removals by face, -1 unused |
//! | 11..15 | matching removal red flags; zero when unused |
//! | 15 | contested claim proposal (chi/pon/daiminkan/ron), not an award |
//!
//! Ordinary discards and riichi remove one tile; chi/pon remove the two
//! consumed hand tiles; daiminkan removes three; ankan removes four; kakan
//! removes the added tile only; ron, tsumo, pass, and abort actions remove
//! none. A claim descriptor names the source and proposed hand tiles but
//! never assumes the claim is awarded or that another seat passed.
//!
//! # Exceptional behavior
//!
//! A cached own draw is counted as DRAWN only during `draw_decision` when the
//! latest actor action is the matching `draw_tile` and the copy is absent from
//! the concealed hand. A copy retained after a hand discard or call stays
//! concealed; responder snapshots already hold the full hand. Discarded,
//! consumed, or reset copies are never resurrected. A
//! riichi discard emits both a `discard` and a `riichi_declared` envelope for
//! the same copy; the two are merged into one record. A claimed discard keeps
//! its original exposure position but becomes the meld's called tile, so the
//! river and meld do not count it twice. Kakan reclassifies the three pon
//! tiles in place and adds the fourth. Positions are ordinals among
//! actor-visible provenance events, never absolute arena indices or event
//! sequences. The public `call_window` marker is excluded from that ordinal
//! because its presence depends on whether another seat has a legal
//! response; it cannot shift a feature value. Another seat's riichi discard
//! has no public drawn/held evidence, so its drawn flag is zero. The scan is
//! bounded to the current-hand visible window (`HISTORY_EVENT_CAP` events);
//! no prior-hand scan occurs.

mod candidates;
mod copy_modes;
mod draw;
mod event_payload;
mod layout;
mod ledger;
mod records;
mod seats;
#[cfg(test)]
mod tests;

use crate::view::{EventArena, Observation as NativeObservation};
use hydra_rules::rl::actions::CanonicalAction;

pub const TOKENS: usize = 39;
pub const WIDTH: usize = 48;
pub const ACTION: usize = 16;

/// Encode one acting seat's observation and canonical legal set.
///
/// `tokens` must be exactly `TOKENS * WIDTH` floats and `candidates` must be
/// exactly `legal.len() * ACTION` integers. Every output element is written;
/// buffer lengths, tile ids, event shapes, and action identities are checked.
pub fn encode(
    obs: &NativeObservation,
    arena: &EventArena,
    legal: &[(u32, CanonicalAction)],
    tokens: &mut [f32],
    candidates: &mut [i32],
) -> Result<(), String> {
    if tokens.len() != TOKENS * WIDTH {
        return Err(format!(
            "tokens length {} != {}",
            tokens.len(),
            TOKENS * WIDTH
        ));
    }
    let wanted = legal
        .len()
        .checked_mul(ACTION)
        .ok_or_else(|| "candidate count overflows usize".to_owned())?;
    if candidates.len() != wanted {
        return Err(format!(
            "candidates length {} != {} for {} legal actions",
            candidates.len(),
            wanted,
            legal.len()
        ));
    }
    tokens.fill(0.0);
    let ledger = ledger::build(obs, arena)?;
    for face in 0..layout::TILE_ROWS {
        records::write_tile_row(
            face,
            &ledger.faces[face],
            obs.actor,
            ledger.drawn,
            &obs.dora_indicators,
            tokens,
        )?;
    }
    seats::write_seat_rows(obs, &ledger, tokens)?;
    seats::write_match_row(obs, tokens)?;
    candidates::write_candidates(obs, &ledger, legal, candidates)?;
    Ok(())
}
