use super::handlers_core::{WalkResult, check_drawer, rej};
use super::headers::{RyukyokuReason, parse_ryukyoku};
use super::ids::{
    BASE_ANKAN, EXHAUSTIVE_MIN_DRAWS, KYUSHU_DISTINCT_YAOCHU, KYUSHU_MAX_DRAWS, PASS_ID,
    UNRESOLVED_ID, YAOCHU_TYPES, chosen_id, source_offset,
};
use super::offers::{canonical_take, draw_offer_ids, insert_chosen, seat_takes};
use super::shape::{isort_u32, win_shape_14};
use super::spans::{WalkState, cached_window, deltas_quad, hora_tsumo_flag, mint_row};
use crate::ingest::{KIND_CHI, KIND_DAIMINKAN, KIND_HORA, KIND_PON, StackedEvent};
use crate::ledger::{
    ACT_DISCARD, ACT_RON, ACT_TSUMO, ACT_TSUMOGIRI, GameCtx, H_ABORTIVE, H_DRAW_END, H_RON,
    H_TSUMO, Ledger, PHASE_DRAW, PHASE_KAN_RESPONSE, PHASE_RESPONSE, RowSink,
    WALK_ACTION_UNRESOLVED, WALK_CLAIM_NO_OFFER, WALK_ENGINE_DESYNC, WALK_KYUSHU_AMBIGUOUS,
    WALK_TILE_CONSERVATION, WALK_TURN_ORDER, WALK_UNMAPPED_RYUKYOKU,
};

pub(crate) fn do_hora(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    out: &mut RowSink<'_>,
    ev_all: &[StackedEvent<'_>],
    idx: usize,
) -> WalkResult<()> {
    let ev = &ev_all[idx];
    if ev.actor > 3 || ev.target > 3 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let winner = ev.actor;
    let s = winner as usize;
    let deltas = deltas_quad(ev.span).ok_or(rej(ctx.game_idx, idx, WALK_TURN_ORDER))?;
    // Settlement applies AFTER the row is minted (see tail): the hora row is
    // the winning declaration, so its observation must read pre-payoff
    // scores. Quad shape is still checked here, never interpreted.
    // Tenhou-real tsumo wins carry no flag (and no pai); target==actor
    // marks them, while ron names the discarder.
    let self_draw = hora_tsumo_flag(ev.span) || ev.target == winner;
    if self_draw {
        check_drawer(l, ctx.game_idx, idx, winner)?;
        let drawn = l.drawn_col[s].ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        let mut takes = [0u8; 24];
        let tn = seat_takes(l, winner, &mut takes);
        if !win_shape_14(&takes[..tn], l.meld_lens[s] as usize) {
            return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
        }
        // Report tile: folded canonical in fold regime (SIM oracle renders
        // the pool-first copy), true takes unfolded.
        let rep = if l.fold_reports {
            canonical_take(drawn)
        } else {
            drawn
        };
        let chosen = chosen_id(ACT_TSUMO, rep, 0);
        if chosen == UNRESOLVED_ID {
            return Err(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED));
        }
        // Offer (+ forced pair under riichi); the taken-win tsumo answer
        // always rides (log-win forcing).
        let mut buf = [0u32; 32];
        let mlen: usize;
        if l.riichi[s] != 0 {
            let a = chosen_id(ACT_DISCARD, rep, 0);
            let b = chosen_id(ACT_TSUMOGIRI, rep, 0);
            let mut combo = [0u32; 32];
            combo[0] = a;
            combo[1] = b;
            let mut cn = if a == b { 1 } else { 2 };
            let mut offer = [0u32; 32];
            let olen = draw_offer_ids(&mut *l, st, winner, &mut offer);
            let mut i = 0usize;
            while i < olen {
                if offer[i] >= BASE_ANKAN && offer[i] < BASE_ANKAN + 34
                    && cn < 32 {
                        combo[cn] = offer[i];
                        cn += 1;
                    }
                i += 1;
            }
            isort_u32(&mut combo, cn);
            buf = combo;
            mlen = cn;
        } else {
            mlen = draw_offer_ids(&mut *l, st, winner, &mut buf);
        }
        let mlen8 = insert_chosen(&mut buf, mlen, chosen);
        mint_row(
            l,
            st,
            out,
            winner,
            PHASE_DRAW,
            winner,
            chosen,
            &buf[..mlen8 as usize],
        );
        st.push_public(H_TSUMO);
    } else {
        let (discarder, tile) = match st.last_discard {
            Some(pair) => pair,
            None => return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER)),
        };
        if ev.target != discarder {
            return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
        }
        let phase = if st.opened_by_discard {
            PHASE_RESPONSE
        } else {
            PHASE_KAN_RESPONSE
        };
        // Shape gate: the discard must complete a winning shape.
        let mut takes = [0u8; 24];
        let tn = seat_takes(l, winner, &mut takes);
        let mut completed = [0u8; 24];
        let mut cn = 0usize;
        let mut i = 0usize;
        while i < tn && cn < 23 {
            completed[cn] = takes[i];
            cn += 1;
            i += 1;
        }
        if cn < 24 {
            completed[cn] = tile;
            cn += 1;
        }
        if !win_shape_14(&completed[..cn], l.meld_lens[s] as usize) {
            return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
        }
        let offcode = match source_offset(discarder, winner) {
            -1 => 0u8,
            1 => 1u8,
            2 => 2u8,
            _ => return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC)),
        };
        let chosen = chosen_id(ACT_RON, tile, offcode);
        if chosen == UNRESOLVED_ID {
            return Err(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED));
        }
        // Ron-response mask: exactly {pass, ron}.
        let mask = [PASS_ID, chosen];
        let mlen = if chosen == PASS_ID { 1 } else { 2 };
        // Ron rides no draw turn: abilities read false (clear BEFORE mint).
        l.riichi_offered = false;
        l.tsumo_offered = false;
        mint_row(l, st, out, winner, phase, discarder, chosen, &mask[..mlen]);
        st.push_public(H_RON);
    }
    // Settlement AFTER the row (both hora paths): the observation above
    // (scores plane and any score-gated offers) reads pre-payoff state. The
    // ledger still settles for subsequent use. Pre-payoff inputs are
    // load-bearing: post-payoff scores would leak the win payout (the
    // target) into the decision input.
    let mut i = 0usize;
    while i < 4 {
        l.scores[i] += deltas[i];
        i += 1;
    }
    st.decided = true;
    st.hora_terminal = true;
    st.last_claim = None;
    l.drawer = None;
    l.exp_drawer = None;
    l.kan_pending = false;
    Ok(())
}

pub(crate) fn kyushu_infer(l: &Ledger) -> bool {
    let seat = match l.drawer {
        Some(s) => s,
        None => return false,
    };
    if l.tsumo_cnt[seat as usize] != 1 {
        return false;
    }
    let first = match l.first_draw[seat as usize] {
        Some(t) => t,
        None => return false,
    };
    let mut kinds = [false; 34];
    let mut t = 0usize;
    while t < 34 {
        if l.tehai_cnt[t] > 0 {
            kinds[t] = true;
        }
        t += 1;
    }
    let ft = (first / 4) as usize;
    if ft < 34 {
        kinds[ft] = true;
    }
    let mut n = 0usize;
    let mut i = 0usize;
    while i < YAOCHU_TYPES.len() {
        if kinds[YAOCHU_TYPES[i] as usize] {
            n += 1;
        }
        i += 1;
    }
    n >= KYUSHU_DISTINCT_YAOCHU
}

pub(crate) fn do_ryukyoku(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    idx: usize,
    span: &[u8],
) -> WalkResult<()> {
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let body = parse_ryukyoku(span).map_err(|reason| rej(ctx.game_idx, idx, reason))?;
    let mut i = 0usize;
    while i < 4 {
        l.scores[i] += body.deltas[i];
        i += 1;
    }
    let hid = match body.reason {
        Some(RyukyokuReason::DrawEnd) => H_DRAW_END,
        Some(RyukyokuReason::Abortive) => H_ABORTIVE,
        Some(RyukyokuReason::Unmapped) => {
            return Err(rej(ctx.game_idx, idx, WALK_UNMAPPED_RYUKYOKU));
        }
        None => {
            if l.draws >= EXHAUSTIVE_MIN_DRAWS {
                H_DRAW_END
            } else if l.draws <= KYUSHU_MAX_DRAWS && kyushu_infer(l) {
                H_ABORTIVE
            } else {
                return Err(rej(ctx.game_idx, idx, WALK_KYUSHU_AMBIGUOUS));
            }
        }
    };
    st.push_public(hid);
    st.decided = true;
    st.last_claim = None;
    l.drawer = None;
    l.exp_drawer = None;
    l.kan_pending = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Walled mode gate (frozen turn-order checks; wall-less skips entirely).
// ---------------------------------------------------------------------------

/// Sorted pending responders for a live discard (cached per discard).
pub(crate) fn mode_pending(
    l: &Ledger,
    st: &mut WalkState,
    discarder: u8,
    tile: u8,
    disc_idx: usize,
) -> ([u8; 3], u8) {
    cached_window(st, l, discarder, tile, disc_idx, false).1
}

pub(crate) fn check_mode_for(
    ctx: &GameCtx<'_>,
    l: &Ledger,
    st: &mut WalkState,
    ev_all: &[StackedEvent<'_>],
    idx: usize,
    kind: u8,
    actor: u8,
) -> WalkResult<()> {
    if !ctx.is_walled() {
        return Ok(());
    }
    if actor > 3 {
        return Ok(());
    }
    // Terminal: post-terminal rows fail closed.
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    // Window vs draw.
    if let Some((discarder, tile)) = st.last_discard
        && st.opened_by_discard {
            let (pending, plen) = mode_pending(l, st, discarder, tile, st.last_discard_idx);
            if plen > 0 {
                let is_claim = matches!(kind, KIND_CHI | KIND_PON | KIND_DAIMINKAN);
                let ev = &ev_all[idx];
                let is_ron = kind == KIND_HORA && !hora_tsumo_flag(ev.span);
                if is_claim {
                    let mut inside = false;
                    let mut i = 0u8;
                    while i < plen {
                        if pending[i as usize] == actor {
                            inside = true;
                            break;
                        }
                        i += 1;
                    }
                    if !inside {
                        return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
                    }
                } else if is_ron {
                    let mut inside = false;
                    let mut i = 0u8;
                    while i < plen {
                        if pending[i as usize] == actor {
                            inside = true;
                            break;
                        }
                        i += 1;
                    }
                    if !inside {
                        // Non-riichi ron passes on shape alone (log fact).
                        let mut takes = [0u8; 24];
                        let tn = seat_takes(l, actor, &mut takes);
                        let mut completed = [0u8; 24];
                        let mut cn = 0usize;
                        let mut i = 0usize;
                        while i < tn && cn < 23 {
                            completed[cn] = takes[i];
                            cn += 1;
                            i += 1;
                        }
                        if cn < 24 {
                            completed[cn] = tile;
                            cn += 1;
                        }
                        if !win_shape_14(&completed[..cn], l.meld_lens[actor as usize] as usize) {
                            return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
                        }
                    }
                }
                return Ok(());
            }
        }
    // Draw mode: claims outside any window fail closed.
    if matches!(kind, KIND_CHI | KIND_PON | KIND_DAIMINKAN) {
        return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
    }
    if kind == KIND_HORA {
        let ev = &ev_all[idx];
        if !hora_tsumo_flag(ev.span) {
            // Chankan: kakan set last_discard with opened_by_discard false.
            if let Some((discarder, tile)) = st.last_discard
                && ev.target == discarder {
                    let mut takes = [0u8; 24];
                    let tn = seat_takes(l, actor, &mut takes);
                    let mut completed = [0u8; 24];
                    let mut cn = 0usize;
                    let mut i = 0usize;
                    while i < tn && cn < 23 {
                        completed[cn] = takes[i];
                        cn += 1;
                        i += 1;
                    }
                    if cn < 24 {
                        completed[cn] = tile;
                        cn += 1;
                    }
                    if win_shape_14(&completed[..cn], l.meld_lens[actor as usize] as usize) {
                        return Ok(());
                    }
                }
            return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
        }
    }
    Ok(())
}
