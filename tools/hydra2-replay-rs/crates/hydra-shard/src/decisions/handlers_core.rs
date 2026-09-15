use super::handlers_close::{do_ankan, do_claim, do_hora, do_kakan, do_ryukyoku};
use super::table::ActionTable;
use super::walker::{WalkReject, Walker, in_skip, int_value, is_end, is_start, prescan};
use crate::engine::{EngineVersion, draw_offer};
use crate::mjai_event::{is_claim_kind, is_row_kind};
use crate::parity::{ChosenAction, ReplayRow};
use crate::stream::{KyokuTrack, ParsedGame};
use crate::tile::{mjai_string_of, physical_of};

/// Replay one framed game into decision rows (walled + wall-less, one gate).
///
/// S7 walled path: `sim.reset(wall)` is the wall-digest mint
/// (`replay-{game_id}` schedule via `wall_schedule_digest`); each logged
/// decision is one `sim.apply` in log order with the identical sequence
/// Python emits — pass placement in sorted pending order (no rows), claim
/// matching on exact strings + consumed + source, reach collapsed onto its
/// declaration dahai (`nxt + 1`), terminal checks (`end_game` break,
/// post-terminal rows fail closed). Wall-less games (`wall_tiles: None`)
/// keep the SIM-mark derivation; walled games bind the real digest.
pub fn walk_game(game: &ParsedGame, table: &ActionTable) -> Result<Vec<ReplayRow>, WalkReject> {
    walk_game_with_version(game, table, EngineVersion::V1)
}

/// Version-selected replay: [`EngineVersion::V1`] runs the pinned
/// drained-oracle backend; [`EngineVersion::V2`] runs the single-pass
/// live-engine semantics (kuikae-strict, complete-chi, kyushu, take
/// conservation). V1 is the frozen default for parity. Both versions share
/// the S7 mode machine and the walled/wall-less digest binding.
pub fn walk_game_with_version(
    game: &ParsedGame,
    table: &ActionTable,
    version: EngineVersion,
) -> Result<Vec<ReplayRow>, WalkReject> {
    prescan(game)?;
    // S7 `reset(wall)`: mint the real digest once per game (never a
    // placeholder). Wall-less stays `None` (SIM mark).
    let wall_digest: Option<String> = match &game.wall_tiles {
        None => None,
        Some(tiles) => {
            if tiles.len() != 136 {
                return Err(WalkReject::new(
                    &game.game_id,
                    -1,
                    "tile-conservation",
                    "wall",
                    "wall must carry 136 tiles",
                ));
            }
            let schedule = crate::parity::schedule_id_for(&game.game_id);
            Some(crate::parity::wall_schedule_digest(&schedule, tiles))
        }
    };
    let mut walker = Walker {
        game,
        table,
        seq: 0,
        round_idx: -1,
        hand_index: -1,
        kyoku_ordinal: -1,
        track: None,
        dealer: 0,
        histories: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        last_kind: None,
        last_discard: None,
        opened_by_discard: false,
        decided: false,
        terminal: false,
        rows: Vec::new(),
        scores: [0; 4],
        version,
        kuikae_gate: None,
        acted: [false; 4],
        claims: 0,
        wall_digest,
    };
    let total = game.events.len();
    let mut idx = 0usize;
    while idx < total {
        let kind = game.events[idx].type_.clone();
        if is_end(&kind) {
            if walker.track.is_none() || walker.kyoku_ordinal < 0 {
                return Err(walker.fail("turn-order", "end_game", "game ends before any kyoku"));
            }
            walker.terminal = true;
            break;
        }
        if kind == "start_kyoku" {
            do_start_kyoku(&mut walker, idx)?;
            idx += 1;
            continue;
        }
        if is_start(&kind) {
            idx += 1;
            continue;
        }
        if walker.track.is_none() || walker.kyoku_ordinal < 0 {
            return Err(walker.fail("turn-order", &kind, "decision before the first start_kyoku"));
        }
        match kind.as_str() {
            "ryukyoku" => {
                do_ryukyoku(&mut walker, idx)?;
                idx += 1;
            }
            "end_kyoku" => {
                walker.push_public("round_end");
                walker.track_mut("end_kyoku")?.drawer = None;
                idx += 1;
            }
            "tsumo" => {
                do_tsumo(&mut walker, idx)?;
                idx += 1;
            }
            "dora" => {
                let marker = game.events[idx].dora_marker.clone().unwrap_or_default();
                if marker.is_empty() {
                    return Err(walker.fail(
                        "bare-dora",
                        "dora",
                        "dora event without a dora_marker",
                    ));
                }
                walker.track_mut("dora")?.dora.push(marker);
                walker.push_public("dora_revealed");
                idx += 1;
            }
            "reach_accepted" => {
                game.events[idx].actor_seat("reach_accepted").map_err(|e| {
                    walker.fail(
                        "turn-order",
                        "reach_accepted",
                        &format!("event actor must be 0..3 ({e})"),
                    )
                })?;
                walker.push_public("riichi_accepted");
                idx += 1;
            }
            _ if in_skip(&kind) => {
                idx += 1;
            }
            "reach" => {
                // Collapse with the following declaration dahai (mirrors
                // `_skip_index`: transparent + skip events never reset the
                // collapse; the declaration index `nxt + 1` is the resume
                // point, exactly like `_step_draw`'s reach path).
                let mut nxt = idx + 1;
                while nxt < total && in_skip(game.events[nxt].type_.as_str()) {
                    nxt += 1;
                }
                if nxt >= total {
                    return Err(walker.fail("turn-order", "reach", "declaration missing"));
                }
                if game.events[nxt].type_ != "dahai" {
                    return Err(walker.fail("turn-order", "reach", "declaration is not a dahai"));
                }
                // S7 mode gate (`_step_draw` reach path): the declarer must
                // own the draw decision; post-terminal reach fails closed.
                if walker.wall_digest.is_some() {
                    if let Ok(actor) = game.events[idx].actor_seat("reach") {
                        walker.check_mode_for("reach", actor, idx)?;
                    }
                }
                do_reach(&mut walker, idx, nxt)?;
                idx = nxt + 1;
            }
            _ if is_row_kind(&kind) => {
                // S7 mode gate (`_step_decision`): expected-actor /
                // window-pending / draw-mode / terminal against engine
                // answers. Wall-less skips (frozen parity); walled fails
                // closed on post-terminal rows and claims outside any
                // window, with pass placement in sorted pending order.
                if walker.wall_digest.is_some() {
                    if let Ok(actor) = game.events[idx].actor_seat(&kind) {
                        walker.check_mode_for(&kind, actor, idx)?;
                    }
                }
                match kind.as_str() {
                    k if is_claim_kind(k) => do_claim(&mut walker, idx, k)?,
                    "ankan" => do_ankan(&mut walker, idx)?,
                    "kakan" => do_kakan(&mut walker, idx)?,
                    "hora" => do_hora(&mut walker, idx)?,
                    _ => do_dahai(&mut walker, idx)?,
                }
                idx += 1;
            }
            _ => {
                return Err(walker.fail(
                    "unknown-event",
                    &kind,
                    &format!("unmapped mjai event type {kind:?}"),
                ));
            }
        }
    }
    if !walker.terminal {
        return Err(walker.fail("turn-order", "walk", "game never reached end_game"));
    }
    Ok(walker.rows)
}

/// Engine-free row count: one per row kind with `reach` collapsed onto its
/// declaration dahai (mirrors `_count_row_decisions` + `_skip_index`).
/// The walk must emit exactly this many rows on mutually-ok games; on
/// truncated games it still counts the prefix the walk would have emitted.
pub fn count_row_decisions(game: &ParsedGame) -> Result<usize, WalkReject> {
    let total = game.events.len();
    let mut count = 0usize;
    let mut kyoku_idx: i64 = -1;
    let mut idx = 0usize;
    while idx < total {
        let kind = game.events[idx].type_.as_str();
        if is_end(kind) {
            break;
        }
        if kind == "start_kyoku" {
            kyoku_idx += 1;
        }
        if kind == "reach" {
            match skip_index(game, idx + 1, kyoku_idx)? {
                None => {
                    return Err(WalkReject::new(
                        &game.game_id,
                        kyoku_idx,
                        "turn-order",
                        "reach",
                        "reach event without a following dahai",
                    ));
                }
                Some(nxt) => {
                    if game.events[nxt].type_ != "dahai" {
                        return Err(WalkReject::new(
                            &game.game_id,
                            kyoku_idx,
                            "turn-order",
                            "reach",
                            "reach event must be followed by its declaration dahai",
                        ));
                    }
                    count += 1;
                    idx = nxt + 1;
                    continue;
                }
            }
        }
        if is_row_kind(kind) {
            count += 1;
        } else if !in_skip(kind) {
            return Err(WalkReject::new(
                &game.game_id,
                kyoku_idx,
                "unknown-event",
                kind,
                &format!("unmapped mjai event type {kind:?}"),
            ));
        }
        idx += 1;
    }
    Ok(count)
}

/// First index >= start holding a decision, reach, or end event (None when
/// the tail holds none). Mirrors `_skip_index`.
fn skip_index(game: &ParsedGame, start: usize, kyoku: i64) -> Result<Option<usize>, WalkReject> {
    let mut idx = start;
    while idx < game.events.len() {
        let kind = game.events[idx].type_.as_str();
        if is_end(kind) || is_row_kind(kind) || kind == "reach" {
            return Ok(Some(idx));
        }
        if !in_skip(kind) {
            return Err(WalkReject::new(
                &game.game_id,
                kyoku,
                "unknown-event",
                kind,
                &format!("unmapped mjai event type {kind:?}"),
            ));
        }
        idx += 1;
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

fn do_start_kyoku(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let event = &walker.game.events[idx];
    let ordinal = walker.kyoku_ordinal + 1;
    let game_id = walker.game.game_id.clone();
    let bad =
        |code: &str, why: String| WalkReject::new(&game_id, ordinal, code, "start_kyoku", &why);
    let oya = int_value(&event.oya)
        .filter(|v| (0..=3).contains(v))
        .ok_or_else(|| {
            bad(
                "turn-order",
                "malformed hand header: oya must be 0..3".to_string(),
            )
        })? as u8;
    for (name, value) in [("honba", &event.honba), ("kyotaku", &event.kyotaku)] {
        if int_value(value).filter(|v| *v >= 0).is_none() {
            return Err(bad(
                "turn-order",
                format!("malformed hand header: {name} must be a non-negative int"),
            ));
        }
    }
    let scores_raw = event
        .scores
        .clone()
        .ok_or_else(|| bad("turn-order", "scores must cover 4 seats".to_string()))?;
    if scores_raw.len() != 4 || scores_raw.iter().any(|v| v.as_i64().is_none()) {
        return Err(bad("turn-order", "scores must cover 4 seats".to_string()));
    }
    let scores = [
        scores_raw[0].as_i64().unwrap_or(0) as i32,
        scores_raw[1].as_i64().unwrap_or(0) as i32,
        scores_raw[2].as_i64().unwrap_or(0) as i32,
        scores_raw[3].as_i64().unwrap_or(0) as i32,
    ];
    let bakaze = event.bakaze.clone().unwrap_or_default();
    if !matches!(bakaze.as_str(), "E" | "S" | "W" | "N") {
        return Err(bad("turn-order", format!("unknown bakaze {bakaze:?}")));
    }
    let tehais = event
        .tehais
        .clone()
        .ok_or_else(|| bad("turn-order", "malformed hand header: tehais".to_string()))?;
    let track = KyokuTrack::install(
        &walker.game.events,
        ordinal as usize,
        idx,
        &tehais,
        &game_id,
    )
    .map_err(|err| WalkReject {
        game_id: game_id.clone(),
        code: err.code,
        detail: err.detail,
    })?;
    walker.kyoku_ordinal = ordinal;
    walker.round_idx += 1;
    walker.hand_index += 1;
    walker.dealer = oya;
    walker.track = Some(track);
    walker.scores = scores;
    walker.last_discard = None;
    walker.opened_by_discard = false;
    walker.decided = false;
    walker.kuikae_gate = None;
    walker.acted = [false; 4];
    walker.claims = 0;
    walker.histories = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if ordinal == 0 {
        walker.push_public("game_start");
    }
    walker.push_public("round_start");
    Ok(())
}

pub(crate) fn check_drawer(walker: &mut Walker, seat: u8, what: &str) -> Result<(), WalkReject> {
    let track = walker.track_mut(what)?;
    match track.drawer {
        None => track.drawer = Some(seat),
        Some(d) if d == seat => {}
        Some(_) => {
            return Err(walker.fail(
                "turn-order",
                what,
                &format!("{what}: actor {seat} != expected decision seat"),
            ));
        }
    }
    Ok(())
}

fn do_tsumo(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("tsumo")
            .map_err(|e| walker.fail("turn-order", "tsumo", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "tsumo", "draw without a pai string"));
        }
        (actor, pai)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "tsumo", "draw after the kyoku was decided"));
    }
    let expected = if walker.track("tsumo")?.kan_pending {
        walker.track("tsumo")?.exp_drawer
    } else if let Some((last, _)) = walker.last_discard {
        Some((last + 1) % 4)
    } else {
        Some(walker.track("tsumo")?.exp_drawer.unwrap_or(walker.dealer))
    };
    if Some(actor) != expected {
        return Err(walker.fail(
            "turn-order",
            "tsumo",
            &format!("seat {actor} tsumo breaks turn order (out-of-turn draw)"),
        ));
    }
    if walker.track("tsumo")?.draws >= 70 {
        return Err(walker.fail("draw-past-wall", "tsumo", "draw past the end of the wall"));
    }
    let game_id = walker.game.game_id.clone();
    let copy = walker.track_mut("tsumo").and_then(|track| {
        track
            .pop_draw(actor, &pai, &game_id)
            .map_err(|err| WalkReject {
                game_id: game_id.clone(),
                code: err.code,
                detail: err.detail,
            })
    })?;
    {
        let track = walker.track_mut("tsumo")?;
        track.hands[actor as usize].push(copy);
        track.drawn[actor as usize] = Some(copy);
        track.drawn_live[actor as usize] = Some(copy);
        track.draws += 1;
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = false;
        track.tsumo_counts[actor as usize] += 1;
        if track.first_draws[actor as usize].is_none() {
            track.first_draws[actor as usize] = Some(pai.clone());
        }
    }
    if walker.last_discard.map(|(s, _)| s) == Some(actor) {
        walker.last_discard = None;
    }
    walker.push_public("turn_advance");
    walker.histories[actor as usize].push("draw_tile".to_string());
    Ok(())
}

/// Remove the discard copy from the tracked hand (exact value preferred,
/// drawn-preferred string match otherwise) and record the river.
pub(crate) fn apply_discard(
    walker: &mut Walker,
    actor: u8,
    tile: u8,
    drawn: Option<u8>,
) -> Result<(), WalkReject> {
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    let in_hand = walker
        .track("dahai")
        .map(|track| track.hands[actor as usize].contains(&tile))
        .unwrap_or(false);
    let track = walker.track_mut("dahai")?;
    let hand = &mut track.hands[actor as usize];
    if in_hand {
        let pos = hand.iter().position(|t| *t == tile).unwrap_or(0);
        hand.remove(pos);
    } else {
        let pai = mjai_string_of(tile).unwrap_or_default();
        KyokuTrack::remove_one(hand, &pai, drawn, &game_id, ordinal as usize, "dahai").map_err(
            |err| WalkReject {
                game_id: game_id.clone(),
                code: err.code,
                detail: err.detail,
            },
        )?;
    }
    track.rivers[actor as usize].push(tile);
    track.exp_drawer = Some(actor);
    Ok(())
}

fn do_dahai(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai, tsumogiri_flag) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("dahai")
            .map_err(|e| walker.fail("turn-order", "dahai", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "dahai", "discard without a pai string"));
        }
        (actor, pai, event.tsumogiri)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "dahai", "discard after the kyoku was decided"));
    }
    check_drawer(walker, actor, "dahai")?;
    let drawn = walker.track("dahai")?.drawn[actor as usize];
    let collapsed =
        physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "dahai", &e.to_string()))?;
    // Post-reach discards report the true drawn take (never the collapsed
    // twin), mirroring the drained forced answers. Pre-reach discards
    // report the collapsed id, exactly like the yielded steps.
    let (kind, tile) = if walker.track("dahai")?.riichi_declared[actor as usize] {
        let drawn_true = walker.track("dahai")?.drawn_live[actor as usize];
        let drawn_pai = drawn_true
            .map(|d| {
                mjai_string_of(d)
                    .map_err(|e| walker.fail("tile-conservation", "dahai", &e.to_string()))
            })
            .transpose()?;
        if drawn_pai.as_deref() != Some(pai.as_str()) {
            return Err(walker.fail(
                "turn-order",
                "dahai",
                "forced discard is not the drawn tile",
            ));
        }
        ("tsumogiri", drawn_true.unwrap_or(collapsed))
    } else if tsumogiri_flag {
        ("tsumogiri", collapsed)
    } else {
        ("discard", collapsed)
    };
    let chosen = match kind {
        "tsumogiri" => ChosenAction::Tsumogiri { tile },
        _ => ChosenAction::Discard { tile },
    };
    // Engine offer: riichi-forced rows take the true-take forced pair
    // (nothing else is offered on forced discards: 2191/2191 pair-only
    // rows); every other row reads the pool-first offer.
    let mask = if walker.track("dahai")?.riichi_declared[actor as usize] {
        let forced_take = walker.track("dahai")?.drawn_live[actor as usize].ok_or_else(|| {
            walker.fail(
                "engine-desync",
                "dahai",
                "riichi draw decision without a live drawn tile",
            )
        })?;
        let offer = crate::engine::forced_pair(forced_take);
        walker.draw_mask_ids(actor, &offer, &chosen, false)
    } else {
        let view = walker.seat_view(actor, "dahai")?;
        let mut offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
        let mut kyushu = false;
        if walker.version == crate::engine::EngineVersion::V2 {
            // One-shot immediate filter after chi/pon claims. Withholds the
            // just-claimed CALLED take (live takes never re-offer it) plus
            // (chi only) non-melded takes near the chi. Consumed takes stay
            // offered even when duped in hand (pinned: v2 offers them).
            if let Some((_, called, meld, chi)) = walker.kuikae_gate.clone() {
                let mut cands = offer.discards.clone();
                if let Some(t) = offer.tsumogiri {
                    cands.push(t);
                }
                let meldset: std::collections::HashSet<u8> = meld.into_iter().collect();
                let mut withheld: Vec<u8> = vec![called];
                for w in crate::engine::kuikae_withheld(&chi, &cands) {
                    if !meldset.contains(&w) {
                        withheld.push(w);
                    }
                }
                offer.discards.retain(|d| !withheld.contains(d));
                if let Some(t) = offer.tsumogiri {
                    if withheld.contains(&t) {
                        offer.tsumogiri = None;
                    }
                }
            }
            // Standard kyushu: first draw, no calls yet, 9+ distinct
            // terminals/honors offers the nine-terminals abort.
            let live = walker.track("dahai")?.drawn_live[actor as usize].is_some();
            if live && !walker.acted[actor as usize] && walker.claims == 0 {
                let hand14 = walker.reported_hand(actor)?;
                if crate::engine::distinct_yaochu(&hand14) >= crate::engine::KYUSHU_DISTINCT_YAOCHU
                {
                    kyushu = true;
                }
            }
        }
        let mut mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
        if kyushu {
            if let Some(id) =
                walker
                    .table
                    .lookup("abort_nine_terminals", None, None, &[], None, false, false)
            {
                mask.push(id);
                mask.sort_unstable();
            }
        }
        mask
    };
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.kuikae_gate = None;
    walker.track_mut("dahai")?.drawn_live[actor as usize] = None;
    apply_discard(walker, actor, tile, drawn)?;
    walker.push_public("discard");
    walker.last_discard = Some((actor, tile));
    walker.opened_by_discard = true;
    walker.track_mut("dahai")?.drawer = Some(actor);
    walker.open_window(actor, tile, false, idx + 1);
    Ok(())
}

fn do_reach(walker: &mut Walker, reach_idx: usize, decl_idx: usize) -> Result<(), WalkReject> {
    let actor = walker.game.events[reach_idx]
        .actor_seat("reach")
        .map_err(|e| walker.fail("turn-order", "reach", &e))?;
    if walker.decided {
        return Err(walker.fail(
            "turn-order",
            "reach",
            "declaration after the kyoku was decided",
        ));
    }
    let (dactor, pai) = {
        let decl = &walker.game.events[decl_idx];
        let dactor = decl
            .actor_seat("dahai")
            .map_err(|e| walker.fail("turn-order", "reach-declaration", &e))?;
        let pai = decl.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail(
                "tile-conservation",
                "reach",
                "declaration without a pai string",
            ));
        }
        (dactor, pai)
    };
    if dactor != actor {
        return Err(walker.fail(
            "turn-order",
            "reach",
            "declaration actor differs from reach actor",
        ));
    }
    check_drawer(walker, actor, "reach")?;
    // Declaration discard: the yielded reach step resolves drawn-preferred
    // from collapsed ids, so the tile is always the string-canonical id
    // (ownership still enforced against the tracked hand, fail closed).
    let drawn = walker.track("reach")?.drawn[actor as usize];
    let declaration_tile =
        physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "reach", &e.to_string()))?;
    {
        let track = walker.track("reach")?;
        let owned = track.hands[actor as usize]
            .iter()
            .any(|t| mjai_string_of(*t).map(|p| p == pai).unwrap_or(false));
        if !owned {
            return Err(walker.fail(
                "tile-conservation",
                "reach",
                "declaration discard not owned",
            ));
        }
    }
    let chosen = ChosenAction::RiichiDiscard {
        tile: declaration_tile,
    };
    let view = walker.seat_view(actor, "reach")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    if !offer.riichi.contains(&declaration_tile) {
        return Err(walker.fail(
            "engine-desync",
            "reach",
            &format!("declaration discard {declaration_tile} is not a riichi candidate"),
        ));
    }
    let mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("reach")?.drawn_live[actor as usize] = None;
    apply_discard(walker, actor, declaration_tile, drawn)?;
    walker.push_public("discard");
    walker.last_discard = Some((actor, declaration_tile));
    walker.opened_by_discard = true;
    walker.track_mut("reach")?.drawer = Some(actor);
    walker.track_mut("reach")?.riichi_declared[actor as usize] = true;
    walker.scores[actor as usize] -= 1000;
    walker.open_window(actor, declaration_tile, false, decl_idx + 1);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod walk_skeleton_tests {
    //! Slice S2 gate: row COUNT + decision/round id sequences equal
    //! `_count_row_decisions` (+ the `d{seq:04}` / `h{idx:02}` scheme), and
    //! the walk skeleton (start_kyoku / END / terminal-required /
    //! reach-collapse / unknown-event) quarantines like the oracle.
    //!
    //! Fixture W mirrors /tmp/s2/oracle_dump.py (throwaway probe): its
    //! count (5), decision ids, and round ids below are the oracle outputs
    //! verbatim. Masks/observations are excluded from this gate.
    use super::*;
    use crate::stream::parse_game;

    const TILES: [&str; 34] = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p",
        "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        "P", "F", "C",
    ];

    fn tile_at(slot: usize) -> String {
        TILES[slot % TILES.len()].to_string()
    }

    fn tehais_json() -> String {
        let hands: Vec<String> = (0..4)
            .map(|seat| {
                let tiles: Vec<String> = match seat {
                    // Seat 1: 13-tile shanpon tenpai (567p/123s/456s + 7s/8s
                    // pairs), so the logged 2s declaration below is a
                    // genuine riichi candidate (engine match-offered rule).
                    1 => [
                        "5p", "6p", "7p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "7s", "8s",
                        "8s",
                    ]
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect(),
                    // Seat 2: EEE/SSS/WWW/PPP + FF pair, so the logged E
                    // tsumo below completes a genuine win (shape gate).
                    2 => [
                        "E", "E", "S", "S", "S", "W", "W", "W", "P", "P", "P", "F", "F",
                    ]
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect(),
                    _ => (0..13)
                        .map(|i| format!("{:?}", tile_at(seat * 13 + i)))
                        .collect(),
                };
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn start_kyoku(oya: u8, no: u8) -> String {
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":{no},\"honba\":0,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":{}}}",
            tehais_json()
        )
    }

    fn empty_table() -> ActionTable {
        ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap()
    }

    /// Two-kyoku walk: dahai, reach (transparent dora before its
    /// declaration), tsumo-hora, then dahai/dahai + exhaustive ryukyoku.
    fn two_kyoku_text() -> String {
        let w: Vec<String> = (52..57).map(tile_at).collect();
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", w[0]),
            format!(
                "{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}",
                w[0]
            ),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", w[1]),
            "{\"type\":\"reach\",\"actor\":1}".to_string(),
            "{\"type\":\"dora\",\"dora_marker\":\"5p\"}".to_string(),
            format!(
                "{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}",
                w[1]
            ),
            "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"E\"}".to_string(),
            format!(
                "{{\"type\":\"hora\",\"actor\":2,\"target\":2,\"tsumo\":true,\"deltas\":[8000,-2000,-2000,-4000]}}"
            ),
            "{\"type\":\"end_kyoku\"}".to_string(),
            start_kyoku(1, 2),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", w[3]),
            format!(
                "{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}",
                w[3]
            ),
            format!("{{\"type\":\"tsumo\",\"actor\":2,\"pai\":{:?}}}", w[4]),
            format!(
                "{{\"type\":\"dahai\",\"actor\":2,\"pai\":{:?},\"tsumogiri\":true}}",
                w[4]
            ),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}"
                .to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        lines.join("\n") + "\n"
    }

    #[test]
    fn count_and_ids_match_oracle_on_two_kyoku() {
        let game = parse_game(&two_kyoku_text(), "s2-walk").unwrap();
        let table = empty_table();
        // Oracle `_count_row_decisions` == 5 on this fixture.
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(rows.len(), 5);
        // Decision ids stay positional and gapless across the kyoku seam.
        let ids: Vec<&str> = rows.iter().map(|r| r.decision_id.as_str()).collect();
        for (seq, id) in ids.iter().enumerate() {
            assert_eq!(*id, format!("{}:d{seq:04}", game.game_id));
        }
        // Round ids flip at the kyoku boundary (3 rows in h00, 2 in h01).
        let rounds: Vec<&str> = rows.iter().map(|r| r.round_id.as_str()).collect();
        assert_eq!(
            rounds,
            vec![
                format!("{}:h00", game.game_id),
                format!("{}:h00", game.game_id),
                format!("{}:h00", game.game_id),
                format!("{}:h01", game.game_id),
                format!("{}:h01", game.game_id),
            ]
        );
        assert_eq!(
            rows.iter().map(|r| r.seat).collect::<Vec<_>>(),
            vec![0, 1, 2, 1, 2]
        );
        // Countdown wall: 70 minus per-kyoku draws at capture time (kyoku 1
        // rows land after 1, 2, and 3 draws; kyoku 2 resets the countdown).
        assert_eq!(
            rows.iter().map(|r| r.wall_remaining).collect::<Vec<_>>(),
            vec![69, 68, 67, 69, 68]
        );
    }

    #[test]
    fn kan_fixture_count_matches_oracle() {
        // Count-only (no row walk): the kan/rinshan kyoku carries 9 row
        // decisions per the oracle (6 dahai + ankan + kakan + daiminkan).
        let text = crate::stream::s2_kan_game_text();
        let game = parse_game(&text, "s2-kan").unwrap();
        assert_eq!(count_row_decisions(&game).unwrap(), 9);
    }

    #[test]
    fn unknown_event_quarantines_both_paths() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}",
                tile_at(52)
            ),
            "{\"type\":\"frobnicator\",\"actor\":0}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = parse_game(&(lines.join("\n") + "\n"), "s2-unknown").unwrap();
        // The oracle `_count_row_decisions` raises unmapped here too.
        let count_err = count_row_decisions(&game).unwrap_err();
        assert_eq!(count_err.code, "unknown-event");
        let walk_err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(walk_err.code, "unknown-event");
    }

    #[test]
    fn truncated_game_fails_terminal_required_but_counts_prefix() {
        let mut game = parse_game(&two_kyoku_text(), "s2-trunc").unwrap();
        game.events.pop();
        assert_ne!(
            game.events.last().map(|e| e.type_.as_str()),
            Some("end_game")
        );
        let err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(err.code, "turn-order");
        assert!(err.detail.contains("game never reached end_game"));
        // The engine-free count still reports the prefix honestly.
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
    }

    #[test]
    fn end_before_any_kyoku_is_turn_order() {
        let game = parse_game(
            "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
            "s2-empty",
        )
        .unwrap();
        let err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(err.code, "turn-order");
        assert_eq!(count_row_decisions(&game).unwrap(), 0);
    }

    #[test]
    fn reach_without_declaration_fails_both_paths() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}",
                tile_at(52)
            ),
            "{\"type\":\"reach\",\"actor\":0}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = parse_game(&(lines.join("\n") + "\n"), "s2-reach").unwrap();
        assert_eq!(
            walk_game(&game, &empty_table()).unwrap_err().code,
            "turn-order"
        );
        assert_eq!(count_row_decisions(&game).unwrap_err().code, "turn-order");
    }
}
