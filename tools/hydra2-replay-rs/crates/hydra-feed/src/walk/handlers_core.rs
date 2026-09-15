use super::headers::parse_header;
use super::ids::chosen_id;
use super::ids::{MAX_DRAWS, UNRESOLVED_ID};
use super::offers::{
    alloc_take, canonical_take, draw_offer_ids, insert_chosen, kyushu_offer_append, min_held_take,
    remove_take, seat_takes,
};
use super::shape::{isort_u32, win_shape_14};
use super::spans::{
    WalkState, cached_window, eval_window, mint_row, peek_window_claim, peek_window_ron,
};
use crate::ingest::{NO_PAI, StackedEvent};
use crate::ledger::{
    ACT_DISCARD, ACT_RIICHI_DISCARD, ACT_TSUMOGIRI, GameCtx, H_CALL_WINDOW, H_DISCARD, H_DRAW_TILE,
    H_GAME_START, H_RIICHI_ACCEPTED, H_ROUND_START, H_TURN_ADVANCE, Ledger, Meld, PHASE_DRAW,
    RowSink, WALK_ACTION_UNRESOLVED, WALK_DRAW_PAST_WALL, WALK_ENGINE_DESYNC,
    WALK_TILE_CONSERVATION, WALK_TURN_ORDER, WalkReject,
};
use crate::tiles::TileCodec;

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

pub(crate) type WalkResult<T> = Result<T, WalkReject>;

pub(crate) fn rej(game_idx: u32, idx: usize, reason: u8) -> WalkReject {
    WalkReject::new(game_idx, idx, reason)
}

pub(crate) fn check_drawer(l: &mut Ledger, game_idx: u32, idx: usize, seat: u8) -> WalkResult<()> {
    match l.drawer {
        None => {
            l.drawer = Some(seat);
            Ok(())
        }
        Some(d) if d == seat => Ok(()),
        Some(_) => Err(rej(game_idx, idx, WALK_TURN_ORDER)),
    }
}

pub(crate) fn do_start_kyoku(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    idx: usize,
    span: &[u8],
) -> WalkResult<()> {
    let h = parse_header(span).map_err(|reason| rej(ctx.game_idx, idx, reason))?;
    // Reset kyoku-scoped table state (hands/melds/rivers/takes/turn machine).
    l.hands = [[[0u8; 4]; 34]; 4];
    l.melds = [[Meld::EMPTY; 4]; 4];
    l.meld_lens = [0u8; 4];
    l.rivers = [[0u8; 32]; 4];
    l.rivers_len = [0u8; 4];
    l.draws = 0;
    l.dora = [-1i8; 5];
    l.dora_len = 0;
    l.riichi = [0u8; 4];
    l.drawer = None;
    l.exp_drawer = None;
    l.kan_pending = false;
    l.drawn_col = [None; 4];
    l.drawn_live = [None; 4];
    l.used_n = [0u8; 34];
    l.used_aka = [0u8; 3];
    l.tehai_cnt = [0u8; 34];
    l.first_draw = [None; 4];
    l.tsumo_cnt = [0u32; 4];
    l.accepted = [0u8; 4];
    l.ippatsu = [0u8; 4];
    l.missed = [0u8; 4];
    l.kan_count = 0;
    l.riichi_offered = false;
    l.tsumo_offered = false;
    // Retain header scalars; round_index reads the ordinal pre-increment.
    l.honba = h.honba;
    l.sticks = h.kyotaku;
    l.hand_number = h.kyoku_no;
    l.round_index = l.kyoku_ordinal;
    l.kyoku_ordinal = l.kyoku_ordinal.saturating_add(1);
    // Install tehais seats 0..3 in listed order (take order, fail closed).
    let mut s = 0usize;
    while s < 4 {
        let mut j = 0usize;
        while j < 13 {
            let take = alloc_take(l, s as u8, h.tehais[s][j]).ok_or(rej(
                ctx.game_idx,
                idx,
                WALK_TILE_CONSERVATION,
            ))?;
            let t = (take / 4) as usize;
            if t < 34 {
                l.tehai_cnt[t] += 1;
            }
            j += 1;
        }
        s += 1;
    }
    l.dealer = h.oya;
    l.round_wind = h.bakaze;
    let mut s = 0u8;
    while s < 4 {
        // Oracle-authoritative rotation (`seat_winds_for_dealer`): winds run
        // East/South/West/North ALIGNED BY SEAT FROM DEALER as
        // 27+((oya+s)%4). Both training oracles share that function, so the
        // walk matches it bit-for-bit (verified on oya=1 multi-kyoku rows).
        l.seat_wind[s as usize] = 27 + (s + h.oya) % 4;
        s += 1;
    }
    l.scores = h.scores;
    l.snap_scores = h.scores;
    l.snap_sticks = h.kyotaku;
    l.kyoku_active = true;
    st.reset_discard();
    st.histories = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if !st.started {
        st.push_public(H_GAME_START);
        st.started = true;
    }
    st.push_public(H_ROUND_START);
    Ok(())
}

pub(crate) fn do_tsumo(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    idx: usize,
    ev: &StackedEvent<'_>,
) -> WalkResult<()> {
    if ev.actor > 3 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    if ev.pai == NO_PAI {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let actor = ev.actor;
    let expected = if l.kan_pending {
        l.exp_drawer
    } else if let Some((last, _)) = st.last_discard {
        Some((last + 1) % 4)
    } else {
        Some(l.exp_drawer.unwrap_or(l.dealer))
    };
    if Some(actor) != expected {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    if l.draws >= MAX_DRAWS {
        return Err(rej(ctx.game_idx, idx, WALK_DRAW_PAST_WALL));
    }
    // Post-claim conservation (fail-closed favors oracle): a non-kan claim
    // (pon/chi) leaves the claimer holding a full ledger (pair consumed,
    // called tile melded, no discard yet); a same-seat tsumo with no
    // intervening discard and no rinshan pending would deal a 15th take
    // (12 concealed + 3 meld) the wall-less oracle desyncs on. Reject as
    // tile-conservation (G5 closed set, no taxonomy churn). Kans disarm
    // below (rinshan replacement is real).
    if st.last_claim.map(|(s, _)| s) == Some(actor) && !l.kan_pending {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    let take =
        alloc_take(l, actor, ev.pai).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
    let s = actor as usize;
    l.drawn_col[s] = Some(take);
    l.drawn_live[s] = Some(take);
    l.draws += 1;
    l.drawer = Some(actor);
    l.exp_drawer = Some(actor);
    l.kan_pending = false;
    l.tsumo_cnt[s] += 1;
    if l.first_draw[s].is_none() {
        l.first_draw[s] = Some(take);
    }
    if st.last_discard.map(|(x, _)| x) == Some(actor) {
        st.last_discard = None;
    }
    st.push_public(H_TURN_ADVANCE);
    st.histories[s].push(H_DRAW_TILE);
    Ok(())
}

/// Remove the exact `tile` take from the hand; rivers record true takes.
pub(crate) fn apply_discard(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    idx: usize,
    actor: u8,
    tile: u8,
    _drawn: Option<u8>,
) -> WalkResult<()> {
    remove_take(l, actor, tile).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
    push_river(ctx, l, idx, actor, tile)?;
    l.exp_drawer = Some(actor);
    Ok(())
}

pub(crate) fn push_river(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    idx: usize,
    actor: u8,
    tile: u8,
) -> WalkResult<()> {
    let s = actor as usize;
    if l.rivers_len[s] >= 32 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    l.rivers[s][l.rivers_len[s] as usize] = tile;
    l.rivers_len[s] += 1;
    Ok(())
}

/// Emit `call_window` when the engine opens one (or the log proves one via
/// a taken ron — peeked FIRST so the responder eval is skipped then).
pub(crate) fn open_window(
    l: &Ledger,
    st: &mut WalkState,
    ev: &[StackedEvent<'_>],
    discarder: u8,
    tile: u8,
    chankan: bool,
    from_idx: usize,
    disc_idx: usize,
) {
    let log_ron = !chankan && peek_window_ron(ev, from_idx, discarder);
    let engine_open = if log_ron {
        true
    } else {
        cached_window(st, l, discarder, tile, disc_idx, chankan).0
    };
    if (engine_open || log_ron) && st.opened_by_discard && st.last_kind == Some(H_DISCARD) {
        st.push_public(H_CALL_WINDOW);
    }
}

pub(crate) fn do_dahai(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    out: &mut RowSink<'_>,
    ev_all: &[StackedEvent<'_>],
    idx: usize,
) -> WalkResult<()> {
    let ev = &ev_all[idx];
    if ev.actor > 3 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    if ev.pai == NO_PAI {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let actor = ev.actor;
    let s = actor as usize;
    check_drawer(l, ctx.game_idx, idx, actor)?;
    // Report tile (chosen/offers/last-discard): folded canonical in fold
    // regime, true takes unfolded. State tile (removal/river): true takes
    // always — holdings evolve identically either way.
    let (kind, tile, state) = if l.riichi[s] != 0 {
        let live_take = l.drawn_live[s].ok_or(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC))?;
        if TileCodec::bytes_of(live_take) != TileCodec::bytes_of(ev.pai) {
            return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
        }
        let rep = if l.fold_reports {
            canonical_take(live_take)
        } else {
            live_take
        };
        (ACT_TSUMOGIRI, rep, live_take)
    } else if ev.tsumogiri {
        if l.fold_reports {
            // Fold regime keeps the old lenient path: report the logged
            // canonical id; removal takes the lowest held copy rendering it
            // (exactly the old rank-slot removal).
            let state = min_held_take(l, actor, ev.pai).ok_or(rej(
                ctx.game_idx,
                idx,
                WALK_TILE_CONSERVATION,
            ))?;
            (ACT_TSUMOGIRI, ev.pai, state)
        } else {
            let live_take = l.drawn_live[s].ok_or(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC))?;
            if TileCodec::bytes_of(live_take) != TileCodec::bytes_of(ev.pai) {
                return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
            }
            (ACT_TSUMOGIRI, live_take, live_take)
        }
    } else {
        let take = min_held_take(l, actor, ev.pai).ok_or(rej(
            ctx.game_idx,
            idx,
            WALK_TILE_CONSERVATION,
        ))?;
        let rep = if l.fold_reports { ev.pai } else { take };
        (ACT_DISCARD, rep, take)
    };
    let chosen = chosen_id(kind, tile, 0);
    if chosen == UNRESOLVED_ID {
        return Err(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED));
    }
    // Legal: riichi-forced rows take the forced pair only (folded in fold
    // regime); every other row reads the regime offer.
    let mut buf = [0u32; 32];
    let mut mlen: usize;
    if l.riichi[s] != 0 {
        let a = chosen_id(ACT_DISCARD, tile, 0);
        let b = chosen_id(ACT_TSUMOGIRI, tile, 0);
        buf[0] = a;
        buf[1] = b;
        mlen = if a == b { 1 } else { 2 };
        isort_u32(&mut buf, mlen);
        // Forced pair rows skip the offer build: can_riichi is false
        // (declared), can_tsumo mirrors the winning shape (ability, not
        // the un-offered answer).
        l.riichi_offered = false;
        {
            let mut takes = [0u8; 24];
            let tn = seat_takes(l, actor, &mut takes);
            l.tsumo_offered = tn == 14 && win_shape_14(&takes[..tn], l.meld_lens[s] as usize);
        }
    } else {
        mlen = draw_offer_ids(&mut *l, st, actor, &mut buf);
        mlen = kyushu_offer_append(l, actor, &mut buf, mlen);
    }
    let mlen8 = insert_chosen(&mut buf, mlen, chosen);
    mint_row(
        l,
        st,
        out,
        actor,
        PHASE_DRAW,
        actor,
        chosen,
        &buf[..mlen8 as usize],
    );
    // Post-mint state: the discarder's doujun clears (missed mark + own
    // ippatsu window end); responders offered ron who let the window pass
    // are marked (temporary furiten) — except the seat that claims next.
    l.missed[s] = 0;
    l.ippatsu[s] = 0;
    {
        let (_, _, ron) = eval_window(l, actor, tile, false);
        let claim_seat = peek_window_claim(ev_all, idx + 1, actor).map(|ci| ev_all[ci].actor);
        let mut m = 0u8;
        while m < 4 {
            if m != actor && ron[m as usize] && Some(m) != claim_seat {
                l.missed[m as usize] = 1;
            }
            m += 1;
        }
    }
    let drawn = l.drawn_col[s];
    l.drawn_live[s] = None;
    apply_discard(ctx, l, idx, actor, state, drawn)?;
    st.push_public(H_DISCARD);
    st.last_discard = Some((actor, tile));
    // Intervening discard disarms the post-claim tripwire (ledger flows on).
    st.last_claim = None;
    st.last_discard_idx = idx;
    st.opened_by_discard = true;
    l.drawer = Some(actor);
    open_window(l, st, ev_all, actor, tile, false, idx + 1, idx);
    Ok(())
}

pub(crate) fn do_reach(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    out: &mut RowSink<'_>,
    ev_all: &[StackedEvent<'_>],
    reach_idx: usize,
    decl_idx: usize,
) -> WalkResult<()> {
    let ractor = ev_all[reach_idx].actor;
    if ractor > 3 {
        return Err(rej(ctx.game_idx, reach_idx, WALK_TURN_ORDER));
    }
    if st.decided {
        return Err(rej(ctx.game_idx, reach_idx, WALK_TURN_ORDER));
    }
    let decl = &ev_all[decl_idx];
    if decl.actor != ractor {
        return Err(rej(ctx.game_idx, decl_idx, WALK_TURN_ORDER));
    }
    if decl.pai == NO_PAI {
        return Err(rej(ctx.game_idx, decl_idx, WALK_TILE_CONSERVATION));
    }
    check_drawer(l, ctx.game_idx, reach_idx, ractor)?;
    // Declaration tiles: report folded canonical in fold regime, true takes
    // unfolded; state (removal/river) is the true take always.
    let state = min_held_take(l, ractor, decl.pai).ok_or(rej(
        ctx.game_idx,
        decl_idx,
        WALK_TILE_CONSERVATION,
    ))?;
    let tile = if l.fold_reports { decl.pai } else { state };
    let mut buf = [0u32; 32];
    let mlen = draw_offer_ids(&mut *l, st, ractor, &mut buf);
    let chosen = chosen_id(ACT_RIICHI_DISCARD, tile, 0);
    let mut offered = false;
    let mut i = 0usize;
    while i < mlen {
        if buf[i] == chosen {
            offered = true;
            break;
        }
        i += 1;
    }
    if !offered {
        return Err(rej(ctx.game_idx, decl_idx, WALK_ENGINE_DESYNC));
    }
    let mlen8 = insert_chosen(&mut buf, mlen, chosen);
    mint_row(
        l,
        st,
        out,
        ractor,
        PHASE_DRAW,
        ractor,
        chosen,
        &buf[..mlen8 as usize],
    );
    // Post-mint state (same doujun/window rules as a plain discard first).
    let s = ractor as usize;
    l.missed[s] = 0;
    l.ippatsu[s] = 0;
    {
        let (_, _, ron) = eval_window(l, ractor, tile, false);
        let claim_seat = peek_window_claim(ev_all, decl_idx + 1, ractor).map(|ci| ev_all[ci].actor);
        let mut m = 0u8;
        while m < 4 {
            if m != ractor && ron[m as usize] && Some(m) != claim_seat {
                l.missed[m as usize] = 1;
            }
            m += 1;
        }
    }
    let drawn = l.drawn_col[s];
    l.drawn_live[s] = None;
    apply_discard(ctx, l, decl_idx, ractor, state, drawn)?;
    // Unfold auto-accept (wall oracle declares natively): accepted state,
    // table stick, and ippatsu window open post-declaration. History carries
    // the accept right after the declaration discard (native order).
    if !l.fold_reports {
        l.accepted[s] = 1;
        l.ippatsu[s] = 1;
    }
    st.push_public(H_DISCARD);
    if !l.fold_reports {
        st.push_public(H_RIICHI_ACCEPTED);
    }
    st.last_discard = Some((ractor, tile));
    // Riichi declaration rides its discard: same disarm as a plain dahai.
    st.last_claim = None;
    st.last_discard_idx = decl_idx;
    st.opened_by_discard = true;
    l.drawer = Some(ractor);
    l.riichi[s] = 1;
    l.scores[s] -= 1000;
    l.sticks = l.sticks.saturating_add(1);
    open_window(l, st, ev_all, ractor, tile, false, decl_idx + 1, decl_idx);
    Ok(())
}
