use super::handlers_core::check_drawer;
use super::walker::{WalkReject, Walker};
use crate::engine::{draw_offer, kyushu_first_draw};
use crate::parity::ChosenAction;
use crate::stream::KyokuTrack;
use crate::tile::{copies_of_string, mjai_string_of, physical_of};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Claim and terminal handlers.
// ---------------------------------------------------------------------------

pub(crate) fn do_claim(walker: &mut Walker, idx: usize, kind: &str) -> Result<(), WalkReject> {
    let (actor, pai, consumed_raw, target) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat(kind)
            .map_err(|e| walker.fail("turn-order", kind, &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail(
                "tile-conservation",
                kind,
                "logged claim without a pai string",
            ));
        }
        let consumed = event.consumed.clone().unwrap_or_default();
        let target = event
            .target_seat()
            .map_err(|e| walker.fail("turn-order", kind, &e))?;
        (actor, pai, consumed, target)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", kind, "claim after the kyoku was decided"));
    }
    let (discarder, called) = match walker.last_discard {
        Some(pair) => pair,
        None => {
            return Err(walker.fail("claim-no-offer", kind, "claim without a live discard offer"));
        }
    };
    if target != discarder {
        return Err(walker.fail(
            "claim-no-offer",
            kind,
            &format!("claim target {target} != discarder {discarder}"),
        ));
    }
    let called_pai = mjai_string_of(called)
        .map_err(|e| walker.fail("tile-conservation", kind, &e.to_string()))?;
    if called_pai != pai {
        return Err(walker.fail(
            "claim-no-offer",
            kind,
            "claim names a different tile than the offer",
        ));
    }
    let needed = if kind == "daiminkan" { 3 } else { 2 };
    // Pool-first consumed resolution with ownership enforced against the
    // tracked hand (unowned or missing copies fail closed).
    let consumed = resolve_claim_tiles(walker, actor, &consumed_raw, needed, Some(called))?;
    let chosen = match kind {
        "chi" => ChosenAction::Chi {
            called,
            consumed: consumed.clone(),
        },
        "pon" => ChosenAction::Pon {
            called,
            consumed: consumed.clone(),
            source: discarder,
        },
        _ => ChosenAction::Daiminkan {
            called,
            consumed: consumed.clone(),
            source: discarder,
        },
    };
    let v2 = walker.version == crate::engine::EngineVersion::V2;
    // Claim-response offer. V1 offers exactly `{pass, chosen}` (no chi
    // enumeration beyond the logged claim). V2 additionally enumerates
    // every legal chi pattern (complete-chi, kamicha-gated). Melded-take
    // withholding rides the one-shot immediate filter in `do_dahai`.
    let mut mask = walker.claim_mask_ids(actor, &chosen);
    if v2 && actor == (discarder + 1) % 4 {
        // Complete-chi: every sequence position containing the called
        // tile, with min-held takes per consumed type (live takes are
        // pool-first-held; pinned by d0282 [48,53]+[53,60]+[60,64] and
        // d0299 [8,17]). Patterns whose takes are absent resolve to
        // nothing and are skipped.
        let hand_true = walker.track(kind)?.hands[actor as usize].clone();
        for pair in crate::engine::complete_chi(&hand_true, called) {
            let probe = ChosenAction::Chi {
                called,
                consumed: pair,
            };
            if let Some(id) = probe.lookup(walker.table, actor)
                && !mask.contains(&id) {
                    mask.push(id);
                }
        }
        mask.sort_unstable();
    }
    walker.capture_row(actor, "discard_response", discarder, chosen, &mask)?;
    walker.track_mut(kind)?.drawn_live[actor as usize] = None;
    // Meld record (sorted tiles, like the oracle's VisibleMeld).
    let mut tiles = consumed.clone();
    tiles.push(called);
    tiles.sort_unstable();
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    let kind_owned = kind.to_string();
    {
        let track = walker.track_mut(&kind_owned)?;
        let hand = &mut track.hands[actor as usize];
        for tile in &consumed {
            if let Some(pos) = hand.iter().position(|t| t == tile) {
                hand.remove(pos);
            } else {
                let pai = mjai_string_of(*tile).unwrap_or_default();
                // proof: kyoku ordinals are small non-negative game indices.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    KyokuTrack::remove_one(
                        hand,
                        &pai,
                        None,
                        &game_id,
                        ordinal as usize,
                        &kind_owned,
                    )
                    .map_err(|err| WalkReject {
                        game_id: game_id.clone(),
                        code: err.code,
                        detail: err.detail,
                    })?;
                }
            }
        }
        track.melds[actor as usize].push(crate::stream::TrackedMeld {
            kind: kind_owned,
            owner: actor,
            tiles,
            source: Some(discarder),
            called: Some(called),
        });
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = kind == "daiminkan";
    }
    if walker.version == crate::engine::EngineVersion::V2 && (kind == "chi" || kind == "pon") {
        let gate: Vec<u8> = walker.track(kind)?.melds[actor as usize]
            .last()
            .map(|m| m.tiles.clone())
            .unwrap_or_default();
        // Chi gates both the meld takes and their same-suit neighborhood;
        // pon gates only the meld takes (no kuikae neighborhood, observed).
        let chi = if kind == "chi" {
            gate.clone()
        } else {
            Vec::new()
        };
        walker.kuikae_gate = Some((actor, called, gate, chi));
    }
    walker.push_public(kind);
    walker.last_discard = None;
    Ok(())
}
/// Pool-first consumed resolution: sorted strings take the smallest pool
/// copies (the engine reports string-canonical ids), after enforcing
/// ownership against the tracked hand. A picked copy colliding with the
/// called tile is swapped for an unused pool copy of the same string.
pub(crate) fn resolve_claim_tiles(
    walker: &Walker,
    seat: u8,
    consumed_strings: &[String],
    needed: usize,
    called: Option<u8>,
) -> Result<Vec<u8>, WalkReject> {
    let step = "claim";
    let track = walker.track(step)?;
    let hand = &track.hands[seat as usize];
    let mut sorted = consumed_strings.to_vec();
    sorted.sort();
    if sorted.len() != needed {
        return Err(walker.fail(
            "tile-conservation",
            step,
            &format!("claim needs {needed} consumed tiles"),
        ));
    }
    // Ownership: the tracked hand must render every consumed string.
    let mut remaining: HashMap<String, usize> = HashMap::new();
    for id in hand {
        if let Ok(pai) = mjai_string_of(*id) {
            *remaining.entry(pai).or_insert(0) += 1;
        }
    }
    let mut picked: Vec<u8> = Vec::with_capacity(sorted.len());
    let mut picked_counts: HashMap<String, usize> = HashMap::new();
    for pai in &sorted {
        let have = remaining.get(pai).copied().unwrap_or(0);
        let used = picked_counts.get(pai).copied().unwrap_or(0);
        if used >= have {
            return Err(walker.fail(
                "tile-conservation",
                step,
                &format!("no tracked copy left for meld tile {pai:?}"),
            ));
        }
        let pool = copies_of_string(pai)
            .map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?;
        let take = pool.get(used).copied().ok_or_else(|| {
            walker.fail(
                "tile-conservation",
                step,
                &format!("no tracked copy left for meld tile {pai:?}"),
            )
        })?;
        picked.push(take);
        picked_counts.insert(pai.clone(), used + 1);
    }
    if let Some(called_id) = called {
        for pos in 0..picked.len() {
            if picked[pos] == called_id {
                let pai = mjai_string_of(picked[pos])
                    .map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?;
                let used: HashSet<u8> = picked.iter().copied().chain([called_id]).collect();
                let mut swapped = false;
                for candidate in copies_of_string(&pai)
                    .map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?
                {
                    if !used.contains(&candidate) {
                        picked[pos] = candidate;
                        swapped = true;
                        break;
                    }
                }
                if !swapped {
                    return Err(walker.fail(
                        "tile-conservation",
                        step,
                        &format!("no distinct copy left for meld tile {pai:?}"),
                    ));
                }
            }
        }
    }
    picked.sort_unstable();
    Ok(picked)
}

pub(crate) fn do_ankan(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, consumed_raw) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("ankan")
            .map_err(|e| walker.fail("turn-order", "ankan", &e))?;
        (actor, event.consumed.clone().unwrap_or_default())
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "ankan", "kan after the kyoku was decided"));
    }
    check_drawer(walker, actor, "ankan")?;
    let block = resolve_claim_tiles(walker, actor, &consumed_raw, 4, None)?;
    let base = (block[0] / 4) * 4;
    if block != vec![base, base + 1, base + 2, base + 3] {
        return Err(walker.fail(
            "tile-conservation",
            "ankan",
            &format!("ankan tiles {block:?} are not one block"),
        ));
    }
    let chosen = ChosenAction::Ankan {
        consumed: block.clone(),
    };
    // Engine draw offer. Under riichi the row offers the pool-first forced
    // pair (never full discards), like tsumo wins; otherwise the full
    // pool-first offer.
    let view = walker.seat_view(actor, "ankan")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    let mask = if walker.track("ankan")?.riichi_declared[actor as usize] {
        // Kans need a live draw turn; without one the log is illegal
        // (post-claim kan) and fails closed.
        let live_drawn = view
            .drawn
            .ok_or_else(|| walker.fail("engine-desync", "ankan", "kan without a live draw"))?;
        let forced = crate::engine::forced_pair(live_drawn);
        // The logged quad is always offered (log authority, mirroring the
        // oracle's chosen forcing); other quads ride the same-wait offer.
        // Membership fails closed only when no quad is offerable at all
        // (exhausted wall or no full quad in hand).
        let mut quads = offer.ankan;
        if !quads.contains(&block) {
            if view.wall < crate::engine::KAN_MIN_WALL {
                return Err(walker.fail(
                    "engine-desync",
                    "ankan",
                    &format!("logged quad {block:?} is not an offered ankan"),
                ));
            }
            let mut have = block.clone();
            have.sort_unstable();
            quads.push(have);
        }
        let combined = crate::engine::DrawOffer {
            discards: forced.discards,
            tsumogiri: forced.tsumogiri,
            ankan: quads,
            ..Default::default()
        };
        walker.draw_mask_ids(actor, &combined, &chosen, false)
    } else {
        // Match-offered check (mirrors `_strict_row`): the logged quad
        // must sit in the engine's ankan answers, else desync.
        if !offer.ankan.contains(&block) {
            return Err(walker.fail(
                "engine-desync",
                "ankan",
                &format!("logged quad {block:?} is not an offered ankan"),
            ));
        }
        walker.draw_mask_ids(actor, &offer, &chosen, false)
    };
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("ankan")?.drawn_live[actor as usize] = None;
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    {
        let track = walker.track_mut("ankan")?;
        let hand = &mut track.hands[actor as usize];
        for tile in &block {
            if let Some(pos) = hand.iter().position(|t| t == tile) {
                hand.remove(pos);
            } else {
                let pai = mjai_string_of(*tile).unwrap_or_default();
                // proof: kyoku ordinals are small non-negative game indices.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    KyokuTrack::remove_one(hand, &pai, None, &game_id, ordinal as usize, "ankan")
                        .map_err(|err| WalkReject {
                        game_id: game_id.clone(),
                        code: err.code,
                        detail: err.detail,
                    })?;
                }
            }
        }
        track.melds[actor as usize].push(crate::stream::TrackedMeld {
            kind: "ankan".to_string(),
            owner: actor,
            tiles: block,
            source: None,
            called: None,
        });
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = true;
    }
    walker.push_public("ankan");
    walker.claims += 1;
    Ok(())
}

pub(crate) fn do_kakan(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("kakan")
            .map_err(|e| walker.fail("turn-order", "kakan", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail(
                "tile-conservation",
                "kakan",
                "logged kakan without a pai string",
            ));
        }
        (actor, pai)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "kakan", "kan after the kyoku was decided"));
    }
    check_drawer(walker, actor, "kakan")?;
    // The added fourth copy is the one pool copy of this type absent from
    // the prior pon triple (deterministic, contract-valid).
    let added = {
        let track = walker.track("kakan")?;
        let step_tile = physical_of(&pai)
            .map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))?;
        let added_type = step_tile / 4;
        let prior = track.melds[actor as usize]
            .iter()
            .find(|m| m.kind == "pon" && m.tiles.first().map(|t| t / 4) == Some(added_type))
            .ok_or_else(|| {
                walker.fail(
                    "tile-conservation",
                    "kakan",
                    &format!("no prior pon owned by seat {actor}"),
                )
            })?;
        let mut pool = copies_of_string(&pai)
            .map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))?;
        if pai.len() == 2 && pai.as_bytes()[0] == b'5' {
            let base = (physical_of(&pai)
                .map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))?
                / 4)
                * 4;
            pool = vec![base, base + 1, base + 2, base + 3];
        }
        let owned: HashSet<u8> = prior.tiles.iter().copied().collect();
        let missing: Vec<u8> = pool.into_iter().filter(|c| !owned.contains(c)).collect();
        if missing.len() != 1 {
            return Err(walker.fail(
                "tile-conservation",
                "kakan",
                "prior pon leaves no single added copy",
            ));
        }
        missing[0]
    };
    let chosen = ChosenAction::Kakan { tile: added };
    let view = walker.seat_view(actor, "kakan")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    let mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("kakan")?.drawn_live[actor as usize] = None;
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    {
        let track = walker.track_mut("kakan")?;
        // The meld record keeps the prior pon (mirrors the oracle, which
        // never rewrites state melds on kakan); the added copy leaves the hand.
        let hand = &mut track.hands[actor as usize];
        if let Some(pos) = hand.iter().position(|t| *t == added) {
            hand.remove(pos);
        } else {
            let owed = mjai_string_of(added).unwrap_or_default();
            // proof: kyoku ordinals are small non-negative game indices.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                KyokuTrack::remove_one(hand, &owed, None, &game_id, ordinal as usize, "kakan")
                    .map_err(|err| WalkReject {
                        game_id: game_id.clone(),
                        code: err.code,
                        detail: err.detail,
                    })?;
            }
        }
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = true;
    }
    walker.push_public("kakan");
    walker.claims += 1;
    walker.last_discard = Some((actor, added));
    walker.opened_by_discard = false;
    // Kan windows never emit envelopes (the adapter grammar routes kakan
    // straight to ron); the window still resolves responders upstream.
    walker.open_window(actor, added, true, idx + 1);
    Ok(())
}

pub(crate) fn do_hora(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (winner, tsumo_flag, target, deltas) = {
        let event = &walker.game.events[idx];
        let winner = event
            .actor_seat("hora")
            .map_err(|e| walker.fail("turn-order", "hora", &e))?;
        let target = event
            .target_seat()
            .map_err(|e| walker.fail("turn-order", "hora", &e))?;
        let deltas = event.deltas.clone().unwrap_or_default();
        if deltas.len() != 4 || deltas.iter().any(|d| d.as_i64().is_none()) {
            return Err(walker.fail("turn-order", "hora", "hora without a 4-seat deltas quad"));
        }
        (winner, event.tsumo, target, deltas)
    };
    // Tenhou-real tsumo wins carry no flag (and no pai); target==actor
    // marks them, while ron names the discarder.
    let self_draw = tsumo_flag || target == winner;
    if walker.decided {
        return Err(walker.fail("turn-order", "hora", "win after the kyoku was decided"));
    }
    // Tracked scores follow the log deltas (riichi sticks already moved at
    // declaration time), mirroring `tracked_scores`.
    for (seat, delta) in deltas.iter().enumerate() {
        if let Some(add) = delta.as_i64() {
            // proof: mahjong score deltas fit in i32 (<100k).
            #[allow(clippy::cast_possible_truncation)]
            {
                walker.scores[seat] += add as i32;
            }
        }
    }
    if self_draw {
        check_drawer(walker, winner, "hora")?;
        let drawn = walker.reported_drawn(winner)?.ok_or_else(|| {
            walker.fail(
                "tile-conservation",
                "hora",
                "tsumo win without a drawn tile",
            )
        })?;
        let chosen = ChosenAction::Tsumo { tile: drawn };
        // Engine draw offer (post-reach rows collapse to the forced pair
        // inside the engine) plus the taken-win tsumo answer. The shape
        // gate is the fail-closed net: no win holds without a winning
        // shape, so a shapeless logged win desyncs instead of emitting.
        let track = walker.track("hora")?;
        // Shape gate over the collapsed takes (mirrors the oracle's
        // collapsed hand the evaluator reads).
        let hand14 = walker.reported_hand(winner)?;
        if !crate::engine::win_shape_14(&hand14, track.melds[winner as usize].len()) {
            return Err(walker.fail("engine-desync", "hora", "tsumo win without a winning shape"));
        }
        let mask = if walker.track("hora")?.riichi_declared[winner as usize] {
            // Tsumo wins under riichi offer the pool-first forced pair
            // (never full discards), plus the taken-win tsumo answer and
            // same-wait ankan answers (none occur in corpus, kept for
            // engine fidelity).
            let view = walker.seat_view(winner, "hora")?;
            let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
            let forced = crate::engine::forced_pair(drawn);
            let combined = crate::engine::DrawOffer {
                discards: forced.discards,
                tsumogiri: forced.tsumogiri,
                ankan: offer.ankan,
                ..Default::default()
            };
            walker.draw_mask_ids(winner, &combined, &chosen, true)
        } else {
            let view = walker.seat_view(winner, "hora")?;
            let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
            walker.draw_mask_ids(winner, &offer, &chosen, true)
        };
        walker.capture_row(winner, "draw_decision", winner, chosen, &mask)?;
        walker.push_public("tsumo");
    } else {
        let (discarder, tile) = match walker.last_discard {
            Some(pair) => pair,
            None => {
                return Err(walker.fail(
                    "claim-no-offer",
                    "hora",
                    "ron without a live discard offer",
                ));
            }
        };
        if target != discarder {
            return Err(walker.fail(
                "claim-no-offer",
                "hora",
                &format!("ron target {target} != discarder {discarder}"),
            ));
        }
        let phase = if walker.opened_by_discard {
            "discard_response"
        } else {
            "kan_response"
        };
        // Shape gate, as above: the discard must complete a winning shape
        // for the winner's tracked hand.
        let track = walker.track("hora")?;
        let mut completed = track.hands[winner as usize].clone();
        completed.push(tile);
        if !crate::engine::win_shape_14(&completed, track.melds[winner as usize].len()) {
            return Err(walker.fail("engine-desync", "hora", "ron win without a winning shape"));
        }
        let chosen = ChosenAction::Ron {
            tile,
            source: discarder,
        };
        let mask = walker.ron_mask_ids(winner, tile, discarder);
        walker.capture_row(winner, phase, discarder, chosen, &mask)?;
        walker.push_public("ron");
    }
    walker.decided = true;
    walker.track_mut("hora")?.drawer = None;
    walker.track_mut("hora")?.exp_drawer = None;
    walker.track_mut("hora")?.kan_pending = false;
    Ok(())
}

pub(crate) fn do_ryukyoku(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (reason_raw, deltas) = {
        let event = &walker.game.events[idx];
        (
            event.reason.clone(),
            event.deltas.clone().unwrap_or_default(),
        )
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "ryukyoku", "draw after the kyoku was decided"));
    }
    if deltas.len() != 4 || deltas.iter().any(|d| d.as_i64().is_none()) {
        return Err(walker.fail(
            "turn-order",
            "ryukyoku",
            "draw without a 4-seat deltas quad",
        ));
    }
    for (seat, delta) in deltas.iter().enumerate() {
        if let Some(add) = delta.as_i64() {
            // proof: mahjong score deltas fit in i32 (<100k).
            #[allow(clippy::cast_possible_truncation)]
            {
                walker.scores[seat] += add as i32;
            }
        }
    }
    // Tenhou-real MJAI omits the reason for wall exhaustion and first-turn
    // kyushu aborts; every other abort carries one. The kyushu inference
    // calls the engine's first-draw rule (never a mask offer: the drained
    // engine answers no kyushu bit on any row).
    let reason = match reason_raw {
        Some(r) if !r.is_empty() => r,
        _ => {
            let track = walker.track("ryukyoku")?;
            let draws = track.draws;
            if draws >= 60 {
                "exhaustive_draw".to_string()
            } else if draws <= 4 {
                let kyushu = match track.drawer {
                    Some(seat) => track.first_draws[seat as usize].clone().map(|first| {
                        kyushu_first_draw(
                            &track.tehais[seat as usize],
                            &first,
                            track.tsumo_counts[seat as usize] as usize,
                        )
                    }),
                    None => None,
                };
                match kyushu {
                    Some(true) => "kyushu_kyuhai".to_string(),
                    _ => {
                        return Err(walker.fail(
                            "kyushu-ambiguous",
                            "ryukyoku",
                            "draw without a reason string outside kyushu/exhaustive shape",
                        ));
                    }
                }
            } else {
                return Err(walker.fail(
                    "kyushu-ambiguous",
                    "ryukyoku",
                    "draw without a reason string outside kyushu/exhaustive shape",
                ));
            }
        }
    };
    // History envelope kind follows the rules-manifest reason classes.
    let draw_end = matches!(reason.as_str(), "exhaustive_draw" | "nagashi_mangan");
    let abortive = matches!(
        reason.as_str(),
        "kyushu_kyuhai"
            | "kyuushu_kyuuhai"
            | "suucha_riichi"
            | "sanchaho"
            | "sanchahou"
            | "suukaikan"
            | "suukansansen"
            | "suufon_renda"
            | "sufuurenta"
    );
    if !draw_end && !abortive {
        return Err(walker.fail(
            "unmapped-ryukyoku-reason",
            "ryukyoku",
            &format!("unmapped ryukyoku reason {reason:?}"),
        ));
    }
    walker.push_public(if draw_end {
        "draw_end"
    } else {
        "abortive_draw"
    });
    walker.decided = true;
    let track = walker.track_mut("ryukyoku")?;
    track.drawer = None;
    track.exp_drawer = None;
    track.kan_pending = false;
    Ok(())
}

#[cfg(test)]
mod s3_chosen_resolution_tests {
    //! Slice S3 gate (PG-S3): chosen-action resolution on ledger copies.
    //!
    //! Pins `resolve_claim_tiles` (pool-first smallest-copy + called-collision
    //! swap, matching the `_tracked_consumed` outcome on the copy-collapsed
    //! oracle wall), `apply_discard` (exact-then-drawn-preferred, matching
    //! `_track_discard`), and the `do_*` dispatch order
    //! (tsumo/dahai/reach/claim/ankan/kakan/hora/ryukyoku). Matching stays
    //! ledger-copy based: no engine offer is consulted (that is S6).
    //!
    //! KNOWN-RESIDUAL (recorded, never fixed here — fixing any of these would
    //! reimplement engine rules in a second language; S6 deletes the
    //! approximations with the engine-answer bridge):
    //! - kuikae-discards: the live engine withholds kuikae-illegal discards
    //!   after melds; the wall-less driver offers every string-kind instead
    //!   (probe class `mask-extra/kuikae-discards`, post-claim rows).
    //! - complete-chi variants: string-distinct chi enumeration rejoins here
    //!   while the replay engine offers fewer (probe class
    //!   `mask-extra/chi-variants`, claim rows).
    //! - kyushu proxy over-fire + missing riichi-declaration candidates: win
    //!   evaluation is engine-owned (probe classes `mask-extra/kyushu-proxy`
    //!   and `mask-missing/riichi-candidates`).
    //!
    //! Copy-identity battery: red discard/tsumogiri, ankan strings, chi
    //! order-permutation, kakan-from-pon — smallest-physical-id choice,
    //! byte-identical to the Python oracle on the copy-collapsed wall
    //! (every occurrence of one MJAI string reports its base id, so the
    //! collapsed id IS the oracle id).
    use super::*;
    use crate::decisions::handlers_core::{apply_discard, count_row_decisions, walk_game};
    use crate::decisions::table::{
        ACTION_TABLE_SCHEMA_VERSION, ActionTable, Template, parse_template, table_content_digest,
    };
    use crate::parity::{ChosenAction, ReplayRow};
    use crate::stream::{KyokuTrack, ParsedGame, parse_game};

    const TILES: [&str; 34] = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p",
        "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        "P", "F", "C",
    ];

    fn tile_at(slot: usize) -> String {
        TILES[slot % TILES.len()].to_string()
    }

    /// Cycling tehais (52 distinct slot tiles over 34 kinds, at most two
    /// copies per kind, never red) — overuse-safe for simple games.
    fn cycling_tehais() -> String {
        let hands: Vec<String> = (0..4)
            .map(|seat| {
                let tiles: Vec<String> = (0..13)
                    .map(|i| format!("{:?}", tile_at(seat * 13 + i)))
                    .collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn tehais_of(seats: &[&[&str]]) -> String {
        let hands: Vec<String> = seats
            .iter()
            .map(|hand| {
                let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn start_kyoku_of(oya: u8, no: u8, tehais: &str) -> String {
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":{no},\"honba\":0,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":{tehais}}}"
        )
    }

    fn game_of(lines: &[String], id: &str) -> ParsedGame {
        parse_game(&(lines.join("\n") + "\n"), id).unwrap()
    }

    fn tpl(
        kind: &str,
        tile: Option<u8>,
        called: Option<u8>,
        consumed: &[u8],
        offset: Option<i8>,
        riichi: bool,
        meldref: bool,
    ) -> String {
        let consumed_json = consumed
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let tile_json = tile
            .map(|t| t.to_string())
            .unwrap_or_else(|| "null".to_string());
        let called_json = called
            .map(|t| t.to_string())
            .unwrap_or_else(|| "null".to_string());
        let offset_json = offset
            .map(|o| o.to_string())
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"kind\":{kind:?},\"tile\":{tile_json},\"called_tile\":{called_json},\"consumed_tiles\":[{consumed_json}],\"source_offset\":{offset_json},\"declares_riichi\":{riichi},\"meld_ref_required\":{meldref}}}"
        )
    }

    fn table_of(entries: &[String]) -> ActionTable {
        ActionTable::load_unverified_json(&format!(
            "{{\"payload\":{{\"actions\":[{}]}}}}",
            entries.join(",")
        ))
        .unwrap()
    }

    /// Assemble a fully-declared envelope over synthetic entries: strict
    /// parse, digest in file order, declared digest spliced in.
    fn verified_table_of(entries: &[String]) -> Result<ActionTable, String> {
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).expect("entry json"))
            .collect();
        let mut templates = Vec::new();
        for (id, entry) in actions.iter().enumerate() {
            templates.push(parse_template(entry, id)?);
        }
        let digest = table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates);
        let text = format!(
            "{{\"artifact_type\":\"hydra2.action_table\",\"schema_version\":\"1.0.0\",\"compatibility\":\"exact\",\"payload\":{{\"schema_version\":\"1.0.0\",\"actions\":[{}],\"digest\":{digest:?}}}}}",
            entries.join(",")
        );
        ActionTable::load_json(&text)
    }

    #[test]
    fn verified_loader_accepts_sorted_and_binds_content_digest() {
        let entries = vec![
            tpl("pass", None, None, &[], None, false, false),
            tpl("discard", Some(44), None, &[], None, false, false),
        ];
        let table = verified_table_of(&entries).expect("sorted synthetic table loads");
        assert_eq!(table.len, 2);
        // Stored digest is the content digest over file order (what binds
        // `action_table_hash`), not a file-bytes hash.
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let templates: Vec<Template> = actions
            .iter()
            .enumerate()
            .map(|(id, e)| parse_template(e, id).unwrap())
            .collect();
        assert_eq!(
            table.digest,
            table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates)
        );
        assert_eq!(
            table.lookup("discard", Some(44), None, &[], None, false, false),
            Some(1)
        );
    }

    #[test]
    fn verified_loader_rejects_unsorted_and_duplicates() {
        // Generation order is pass(0) before discard(1): reversed file
        // order with a MATCHING declared digest still fails closed via the
        // rebuild check.
        let reversed = vec![
            tpl("discard", Some(44), None, &[], None, false, false),
            tpl("pass", None, None, &[], None, false, false),
        ];
        let err = verified_table_of(&reversed).unwrap_err();
        assert!(err.contains("generation order"), "unexpected: {err}");
        // Duplicate templates collapse the index: rejected.
        let duplicated = vec![
            tpl("pass", None, None, &[], None, false, false),
            tpl("pass", None, None, &[], None, false, false),
        ];
        let err = verified_table_of(&duplicated).unwrap_err();
        assert!(err.contains("duplicate"), "unexpected: {err}");
    }

    #[test]
    fn verified_loader_rejects_tamper_and_bad_envelopes() {
        let entries = [tpl("pass", None, None, &[], None, false, false),
            tpl("discard", Some(44), None, &[], None, false, false)];
        // Rebuild the declared envelope text for mutation.
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let templates: Vec<Template> = actions
            .iter()
            .enumerate()
            .map(|(id, e)| parse_template(e, id).unwrap())
            .collect();
        let digest = table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates);
        let text = format!(
            "{{\"artifact_type\":\"hydra2.action_table\",\"schema_version\":\"1.0.0\",\"compatibility\":\"exact\",\"payload\":{{\"schema_version\":\"1.0.0\",\"actions\":[{}],\"digest\":{digest:?}}}}}",
            entries.join(",")
        );
        // One template byte flipped: declared-vs-recomputed mismatch.
        let tampered = text.replacen("\"tile\":44", "\"tile\":45", 1);
        assert_ne!(tampered, text);
        let err = ActionTable::load_json(&tampered).unwrap_err();
        assert!(err.contains("declared digest"), "unexpected: {err}");
        // Declared digest flipped instead: same mismatch, other direction.
        let zero_digest = format!("sha256:{}", "0".repeat(64));
        let tampered = text.replacen(&digest, &zero_digest, 1);
        assert!(
            ActionTable::load_json(&tampered)
                .unwrap_err()
                .contains("declared digest")
        );
    }
    fn kinds_of(rows: &[ReplayRow]) -> Vec<&str> {
        rows.iter().map(|r| r.chosen.kind_str()).collect()
    }

    fn assert_ids_exact(rows: &[ReplayRow], ids: &[u32]) {
        assert_eq!(rows.len(), ids.len(), "row count must match the id vector");
        for (row, want) in rows.iter().zip(ids) {
            assert_eq!(
                row.chosen_action_id,
                Some(*want),
                "chosen id for {}",
                row.decision_id
            );
            assert_eq!(
                row.chosen_unresolved, None,
                "no unresolved chosen for {}",
                row.decision_id
            );
        }
    }

    /// Minimal installed walker for direct `resolve_*` / `apply_*` pins: a
    fn walker_for<'g>(
        game: &'g ParsedGame,
        table: &'g ActionTable,
        tehais: &[Vec<String>],
    ) -> Walker<'g> {
        let track = KyokuTrack::install(&game.events, 0, 1, tehais, &game.game_id).unwrap();
        Walker {
            game,
            table,
            seq: 0,
            round_idx: 0,
            hand_index: 0,
            kyoku_ordinal: 0,
            track: Some(track),
            dealer: 0,
            histories: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            last_kind: None,
            last_discard: None,
            opened_by_discard: false,
            decided: false,
            terminal: false,
            rows: Vec::new(),
            scores: [25000; 4],
            version: crate::engine::EngineVersion::V1,
            kuikae_gate: None,
            acted: [false; 4],
            claims: 0,
            wall_digest: None,
        }
    }

    fn direct_tehais() -> Vec<Vec<String>> {
        [
            &[
                "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p",
            ][..],
            &[
                "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P",
            ][..],
            &[
                "F", "C", "E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "6s",
            ][..],
            &[
                "E", "S", "W", "N", "P", "F", "C", "5m", "5p", "7p", "8p", "9p", "2m",
            ][..],
        ]
        .iter()
        .map(|hand| hand.iter().map(|t| t.to_string()).collect())
        .collect()
    }

    fn direct_game() -> ParsedGame {
        let seats: Vec<&[&str]> = vec![
            &[
                "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p",
            ],
            &[
                "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P",
            ],
            &[
                "F", "C", "E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "6s",
            ],
            &[
                "E", "S", "W", "N", "P", "F", "C", "5m", "5p", "7p", "8p", "9p", "2m",
            ],
        ];
        let text = format!(
            "{{\"type\":\"start_game\"}}\n{}\n{{\"type\":\"end_game\"}}\n",
            start_kyoku_of(0, 1, &tehais_of(&seats))
        );
        parse_game(&text, "s3-direct").unwrap()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|t| t.to_string()).collect()
    }

    // -- resolve_claim_tiles: pool-first smallest-copy -----------------------

    #[test]
    fn resolve_takes_smallest_pool_copy_not_hand_position() {
        // Hand holds later copies of "2m" (pool [4,5,6,7]); the oracle's
        // copy-collapsed wall reports every "2m" as its base id, so the
        // resolved copy is pool-first (4), never the hand position (5).
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![5, 6, 40];
        let picked = resolve_claim_tiles(&walker, 1, &strings(&["2m"]), 1, None).unwrap();
        assert_eq!(picked, vec![4]);
    }

    #[test]
    fn resolve_called_collision_swaps_to_unused_pool_copy() {
        // Pool-first picks [44, 45] for two "3p", but 44 is the called tile:
        // the collision swaps to the first unused pool copy (46).
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![44, 45, 40];
        let picked = resolve_claim_tiles(&walker, 1, &strings(&["3p", "3p"]), 2, Some(44)).unwrap();
        assert_eq!(picked, vec![45, 46]);
    }

    #[test]
    fn resolve_chi_consumed_order_permutation_is_identity() {
        // The log may list consumed tiles in either order; resolution sorts
        // strings first, so both spellings resolve to the same copies.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![0, 8, 40];
        let forward =
            resolve_claim_tiles(&walker, 1, &strings(&["3m", "1m"]), 2, Some(48)).unwrap();
        let backward =
            resolve_claim_tiles(&walker, 1, &strings(&["1m", "3m"]), 2, Some(48)).unwrap();
        assert_eq!(forward, vec![0, 8]);
        assert_eq!(backward, vec![0, 8]);
    }

    #[test]
    fn resolve_red_pools_stay_disjoint() {
        // "5p" (pool [53,54,55]) and "5pr" (pool [52]) never share copies.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        let plain = resolve_claim_tiles(&walker, 0, &strings(&["5p"]), 1, None).unwrap();
        let aka = resolve_claim_tiles(&walker, 0, &strings(&["5pr"]), 1, None).unwrap();
        assert_eq!(plain, vec![53]);
        assert_eq!(aka, vec![52]);
    }

    #[test]
    fn resolve_ownership_and_arity_fail_closed() {
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![40, 41, 42];
        // No tracked "E" copy left.
        let missing = resolve_claim_tiles(&walker, 1, &strings(&["E"]), 1, None).unwrap_err();
        assert_eq!(missing.code, "tile-conservation");
        // One logged string cannot satisfy a pair claim.
        let arity = resolve_claim_tiles(&walker, 1, &strings(&["2p"]), 2, None).unwrap_err();
        assert_eq!(arity.code, "tile-conservation");
    }

    // -- apply_discard: exact-then-drawn-preferred ---------------------------

    #[test]
    fn apply_discard_prefers_exact_copy_over_drawn_twin() {
        // Hand holds both the collapsed id (53) and the drawn twin (54) of
        // "5p": the exact copy leaves, the drawn twin stays.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![53, 54, 40];
        apply_discard(&mut walker, 0, 53, Some(54)).unwrap();
        let track = walker.track("test").unwrap();
        assert_eq!(track.hands[0], vec![54, 40]);
        assert_eq!(track.rivers[0], vec![53]);
        assert_eq!(track.exp_drawer, Some(0));
    }

    #[test]
    fn apply_discard_falls_back_to_drawn_preferred_string_match() {
        // Collapsed id (53) absent: the drawn copy rendering "5p" leaves.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![54, 55, 40];
        apply_discard(&mut walker, 0, 53, Some(54)).unwrap();
        let track = walker.track("test").unwrap();
        assert_eq!(track.hands[0], vec![55, 40]);
        assert_eq!(track.rivers[0], vec![53]);
    }

    #[test]
    fn apply_discard_keeps_red_and_normal_copies_distinct() {
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        // Normal "5p" discard removes 53 and keeps the aka 52.
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        apply_discard(&mut walker, 0, 53, None).unwrap();
        assert_eq!(walker.track("test").unwrap().hands[0], vec![52, 40]);
        // Red "5pr" discard removes exactly the aka.
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        apply_discard(&mut walker, 0, 52, Some(52)).unwrap();
        assert_eq!(walker.track("test").unwrap().hands[0], vec![53, 40]);
    }

    // -- end-to-end copy-identity battery ------------------------------------

    #[test]
    fn red_tsumogiri_and_normal_discard_report_collapsed_ids() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &cycling_tehais()),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"5pr\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"5pr\",\"tsumogiri\":true}".to_string(),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}",
                tile_at(53)
            ),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"5p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}"
                .to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-red");
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[
            tpl("tsumogiri", Some(52), None, &[], None, false, false),
            tpl("discard", Some(53), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["tsumogiri", "discard"]);
        assert_eq!(rows[0].chosen, ChosenAction::Tsumogiri { tile: 52 });
        assert_eq!(rows[0].phase, "draw_decision");
        // Plain "5p" reports the collapsed second copy (53), never the aka.
        assert_eq!(rows[1].chosen, ChosenAction::Discard { tile: 53 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn pon_then_kakan_resolves_missing_pon_copy() {
        let tehais = tehais_of(&[
            &[
                "3p", "1m", "2m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "6p", "7p",
            ],
            &[
                "3p", "3p", "3p", "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S",
            ],
            &[
                "W", "N", "P", "F", "C", "1m", "2m", "4m", "6m", "7m", "8m", "9m", "1p",
            ],
            &[
                "E", "S", "W", "N", "P", "F", "C", "1s", "2s", "4s", "6s", "7s", "8s",
            ],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"3p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"3p\",\"3p\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"7m\"}".to_string(),
            "{\"type\":\"kakan\",\"actor\":1,\"pai\":\"3p\"}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"8m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"8m\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"9s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"9s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-pon-kakan");
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
        let table = table_of(&[
            tpl("discard", Some(44), None, &[], None, false, false),
            tpl("pon", None, Some(44), &[45, 46], Some(-1), false, false),
            tpl("kakan", Some(47), None, &[], None, false, true),
            tpl("tsumogiri", Some(28), None, &[], None, false, false),
            tpl("tsumogiri", Some(104), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(
            kinds_of(&rows),
            vec!["discard", "pon", "kakan", "tsumogiri", "tsumogiri"]
        );
        assert_eq!(rows[0].chosen, ChosenAction::Discard { tile: 44 });
        // Called 44 collides with pool-first 44: the consumed pair swaps to
        // the unused copies, and the kakan adds the one copy missing from
        // the prior pon triple.
        assert_eq!(
            rows[1].chosen,
            ChosenAction::Pon {
                called: 44,
                consumed: vec![45, 46],
                source: 0
            }
        );
        assert_eq!(rows[1].phase, "discard_response");
        assert_eq!(rows[2].chosen, ChosenAction::Kakan { tile: 47 });
        assert_eq!(rows[3].chosen, ChosenAction::Tsumogiri { tile: 28 });
        // Seat 2 never held collapsed 104: the drawn copy (105) leaves while
        // the row still reports the collapsed id.
        assert_eq!(rows[4].chosen, ChosenAction::Tsumogiri { tile: 104 });
        assert_ids_exact(&rows, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn chi_consumed_permutation_resolves_sorted_pair() {
        let tehais = tehais_of(&[
            &[
                "4p", "1m", "2m", "3m", "6m", "7m", "8m", "9m", "1p", "2p", "6p", "7p", "8p",
            ],
            &[
                "3p", "5p", "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W",
            ],
            &[
                "N", "P", "F", "C", "1m", "2m", "3m", "6m", "7m", "8m", "9m", "1p", "2p",
            ],
            &[
                "E", "S", "W", "N", "P", "F", "C", "1s", "2s", "4s", "6s", "7s", "9s",
            ],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"4p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"4p\",\"consumed\":[\"5p\",\"3p\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"1s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"1s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-chi");
        assert_eq!(count_row_decisions(&game).unwrap(), 3);
        let table = table_of(&[
            tpl("discard", Some(48), None, &[], None, false, false),
            tpl("chi", None, Some(48), &[44, 53], Some(-1), false, false),
            tpl("tsumogiri", Some(72), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["discard", "chi", "tsumogiri"]);
        // Log order ["5p","3p"] still resolves to the sorted smallest pair.
        assert_eq!(
            rows[1].chosen,
            ChosenAction::Chi {
                called: 48,
                consumed: vec![44, 53]
            }
        );
        assert_eq!(rows[1].phase, "discard_response");
        assert_ids_exact(&rows, &[0, 1, 2]);
    }

    #[test]
    fn ankan_strings_resolve_to_one_physical_block() {
        let tehais = tehais_of(&[
            &[
                "7s", "7s", "7s", "1m", "2m", "3m", "4m", "6m", "8m", "9m", "1p", "2p", "3p",
            ],
            &[
                "1s", "2s", "3s", "4s", "6s", "8s", "9s", "E", "S", "W", "N", "P", "F",
            ],
            &[
                "C", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "6m",
            ],
            &[
                "8m", "9m", "1p", "2p", "3p", "4p", "6p", "7p", "8p", "9p", "1s", "2s", "5m",
            ],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"7s\"}".to_string(),
            "{\"type\":\"ankan\",\"actor\":0,\"consumed\":[\"7s\",\"7s\",\"7s\",\"7s\"]}"
                .to_string(),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"5s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"5s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}"
                .to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-ankan");
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[
            tpl("ankan", None, None, &[96, 97, 98, 99], None, false, false),
            tpl("tsumogiri", Some(89), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["ankan", "tsumogiri"]);
        assert_eq!(
            rows[0].chosen,
            ChosenAction::Ankan {
                consumed: vec![96, 97, 98, 99]
            }
        );
        assert_eq!(rows[0].phase, "draw_decision");
        // Drawn 5s is the pool-first take (no 5s anywhere in tehais), so
        // the tsumogiri twin is the true take itself.
        assert_eq!(rows[1].chosen, ChosenAction::Tsumogiri { tile: 89 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn daiminkan_claim_resolves_called_collision_triple() {
        let tehais = tehais_of(&[
            &[
                "6s", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ],
            &[
                "6s", "6s", "6s", "1s", "2s", "3s", "4s", "7s", "8s", "9s", "E", "S", "W",
            ],
            &[
                "N", "P", "F", "C", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p",
            ],
            &[
                "E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "1s", "2s", "5s",
            ],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"6s\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"daiminkan\",\"actor\":1,\"target\":0,\"pai\":\"6s\",\"consumed\":[\"6s\",\"6s\",\"6s\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"1m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"1m\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-daiminkan");
        assert_eq!(count_row_decisions(&game).unwrap(), 3);
        let table = table_of(&[
            tpl("discard", Some(92), None, &[], None, false, false),
            tpl(
                "daiminkan",
                None,
                Some(92),
                &[93, 94, 95],
                Some(-1),
                false,
                false,
            ),
            tpl("tsumogiri", Some(0), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["discard", "daiminkan", "tsumogiri"]);
        assert_eq!(
            rows[1].chosen,
            ChosenAction::Daiminkan {
                called: 92,
                consumed: vec![93, 94, 95],
                source: 0
            }
        );
        assert_ids_exact(&rows, &[0, 1, 2]);
    }

    // -- do_* dispatch order --------------------------------------------------

    #[test]
    fn reach_collapses_declaration_into_one_riichi_row() {
        // Seat 0 holds a genuine tenpai (123m/456m/789m + 1p pair + 1s/6m
        // floaters): discarding either 6m keeps the shanpon wait, so the
        // logged 6m declaration is a true engine candidate. Seat 1 holds
        // no 2s, so its drawn 2s is the pool-first take (tsumogiri twin).
        let tehais = tehais_of(&[
            &[
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1s", "6m",
            ],
            &[
                "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P", "F",
            ],
            &[
                "9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m",
            ],
            &[
                "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
            ],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}",
                tile_at(52)
            ),
            "{\"type\":\"reach\",\"actor\":0}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"6m\",\"tsumogiri\":true}".to_string(),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}",
                tile_at(53)
            ),
            format!(
                "{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}",
                tile_at(53)
            ),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}"
                .to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-reach");
        // The reach plus its declaration dahai count (and walk) as exactly
        // one row: no orphan dahai row, no zero-row reach.
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[
            tpl("riichi_discard", Some(20), None, &[], None, true, false),
            tpl("tsumogiri", Some(76), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["riichi_discard", "tsumogiri"]);
        // The declaration reports the collapsed id, exactly like yielded steps.
        assert_eq!(rows[0].chosen, ChosenAction::RiichiDiscard { tile: 20 });
        assert_eq!(rows[1].chosen, ChosenAction::Tsumogiri { tile: 76 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn ron_and_tsumo_hora_close_dispatch() {
        // Ron answers the live discard offer from the discard_response phase.
        // Seat 1 holds 11p/234p/567p/88p/999p: the logged 1p discard
        // completes a genuine win (shape gate), and its 1p pair opens the
        // window by pon counts.
        let ron_tehais = tehais_of(&[
            &[
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ],
            &[
                "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "8p", "9p", "9p", "9p",
            ],
            &[
                "9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m",
            ],
            &[
                "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
            ],
        ]);
        let ron_lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &ron_tehais),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}",
                tile_at(52)
            ),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"hora\",\"actor\":1,\"target\":0,\"deltas\":[-8000,12000,-2000,-2000]}"
                .to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let ron_game = game_of(&ron_lines, "s3-ron");
        assert_eq!(count_row_decisions(&ron_game).unwrap(), 2);
        let ron_table = table_of(&[
            tpl("discard", Some(36), None, &[], None, false, false),
            tpl("ron", Some(36), None, &[], Some(-1), false, false),
        ]);
        let ron_rows = walk_game(&ron_game, &ron_table).unwrap();
        assert_eq!(kinds_of(&ron_rows), vec!["discard", "ron"]);
        assert_eq!(
            ron_rows[1].chosen,
            ChosenAction::Ron {
                tile: 36,
                source: 0
            }
        );
        assert_eq!(ron_rows[1].phase, "discard_response");
        assert_ids_exact(&ron_rows, &[0, 1]);
        // Tsumo answers the seat's own draw from the draw_decision phase and
        // reports the string-canonical first copy of the drawn string. Seat
        // 0 holds 111m/234m/567m/23s/99m: the logged first-take 1s draw
        // completes a genuine win (shape gate) under the drained pool
        // discipline (drawn take-index below the hand count, as on every
        // corpus draw).
        let tsumo_tehais = tehais_of(&[
            &[
                "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "2s", "3s", "9m", "9m",
            ],
            &[
                "3p", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P",
            ],
            &[
                "9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m",
            ],
            &[
                "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
            ],
        ]);
        let tsumo_lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tsumo_tehais),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"hora\",\"actor\":0,\"target\":0,\"tsumo\":true,\"deltas\":[12000,-2000,-4000,-6000]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let tsumo_game = game_of(&tsumo_lines, "s3-tsumo-hora");
        assert_eq!(count_row_decisions(&tsumo_game).unwrap(), 1);
        let tsumo_table = table_of(&[tpl("tsumo", Some(72), None, &[], None, false, false)]);
        let tsumo_rows = walk_game(&tsumo_game, &tsumo_table).unwrap();
        assert_eq!(kinds_of(&tsumo_rows), vec!["tsumo"]);
        assert_eq!(tsumo_rows[0].chosen, ChosenAction::Tsumo { tile: 72 });
        assert_eq!(tsumo_rows[0].phase, "draw_decision");
        assert_ids_exact(&tsumo_rows, &[0]);
    }

    #[test]
    fn decided_kyoku_rejects_late_rows() {
        // do_ryukyoku decides the kyoku (emitting no row itself); any later
        // row event in the same kyoku fails closed instead of appending.
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &cycling_tehais()),
            format!(
                "{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}",
                tile_at(52)
            ),
            format!(
                "{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}",
                tile_at(52)
            ),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}"
                .to_string(),
            format!(
                "{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}",
                tile_at(53)
            ),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-decided");
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let err = walk_game(&game, &table).unwrap_err();
        assert_eq!(err.code, "turn-order");
    }
}
