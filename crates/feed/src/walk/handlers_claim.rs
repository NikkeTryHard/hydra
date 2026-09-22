use super::handlers_core::{WalkResult, check_drawer, open_window, rej};
use super::ids::{
    BASE_ANKAN, KAN_MIN_WALL, PASS_ID, UNRESOLVED_ID, chosen_claim_id, chosen_id, source_offset,
};
use super::offers::{
    canonical_take, draw_offer_ids, insert_chosen, min_held_take, presence_count, remove_take,
};
use super::shape::isort_u32;
use super::spans::{WalkState, eval_window, mint_row};
use crate::ingest::{KIND_CHI, KIND_DAIMINKAN, KIND_PON, NO_PAI, StackedEvent};
use crate::ledger::{
    ACT_ANKAN, ACT_CHI, ACT_DAIMINKAN, ACT_DISCARD, ACT_KAKAN, ACT_PON, ACT_TSUMOGIRI, GameCtx,
    H_ANKAN, H_CHI, H_DAIMINKAN, H_KAKAN, H_PON, Ledger, Meld, PHASE_DRAW, PHASE_RESPONSE, RowSink,
    TileClass, WALK_ACTION_UNRESOLVED, WALK_CLAIM_NO_OFFER, WALK_ENGINE_DESYNC,
    WALK_TILE_CONSERVATION, WALK_TURN_ORDER, aka_suit, class_of_canonical, held_takes,
    is_five_type,
};
use crate::tiles::TileCodec;

/// Resolve logged consumed ids to sorted takes for reports and state.
///
/// Fold regime: pool-first takes per string occurrence with the
/// called-collision swap (SIM oracle renders). Unfold regime: lowest held
/// takes per string with a called-collision tripwire (wall oracle encodes
/// held copies). Ownership validation is identical either way.
pub(crate) fn resolve_consumed(
    ctx: &GameCtx<'_>,
    l: &Ledger,
    idx: usize,
    seat: u8,
    consumed: &[u8],
    needed: usize,
    called: Option<u8>,
    fold: bool,
) -> WalkResult<[u8; 4]> {
    if seat > 3 || consumed.len() != needed {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    // Sort ids (string-sort equivalent: order only matters within equal
    // strings, where id order == pool order).
    let mut sorted = [0u8; 4];
    let mut i = 0usize;
    while i < consumed.len() {
        if consumed[i] >= 136 || TileCodec::bytes_of(consumed[i]).is_none() {
            return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
        }
        sorted[i] = consumed[i];
        i += 1;
    }
    let mut a = 0usize;
    while a < consumed.len() {
        let mut b = a + 1;
        while b < consumed.len() {
            if sorted[b] < sorted[a] {
                sorted.swap(a, b);
            }
            b += 1;
        }
        a += 1;
    }
    // Ownership: the hand must render every consumed string.
    let mut need_aka = [0u8; 3];
    let mut need_plain = [0u8; 34];
    let mut need_norm = [0u8; 34];
    let mut i = 0usize;
    while i < consumed.len() {
        let (t, class) = class_of_canonical(sorted[i]);
        match class {
            TileClass::Aka => need_aka[aka_suit(t)] += 1,
            TileClass::PlainFive => need_plain[t as usize] += 1,
            TileClass::Norm => need_norm[t as usize] += 1,
        }
        i += 1;
    }
    let mut t = 0u8;
    while t < 34 {
        let row = &l.hands[seat as usize][t as usize];
        if is_five_type(t) {
            if need_aka[aka_suit(t)] > (row[0] != 0) as u8
                || need_plain[t as usize]
                    > (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8
                || need_norm[t as usize] > 0
            {
                return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
            }
        } else if need_norm[t as usize] > presence_count(row) {
            return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
        }
        t += 1;
    }
    if fold {
        // counters), then the called-collision swap for an unused pool copy
        // of the same string.
        let mut picked = [0u8; 4];
        let mut pc_norm = [0u8; 34];
        let mut pc_plain = [0u8; 34];
        let mut i = 0usize;
        while i < consumed.len() {
            let (t, class) = class_of_canonical(sorted[i]);
            let take = match class {
                TileClass::Aka => t * 4,
                TileClass::PlainFive => {
                    let k = pc_plain[t as usize];
                    pc_plain[t as usize] += 1;
                    t * 4 + 1 + k
                }
                TileClass::Norm => {
                    let k = pc_norm[t as usize];
                    pc_norm[t as usize] += 1;
                    t * 4 + k
                }
            };
            picked[i] = take;
            i += 1;
        }
        if let Some(called_id) = called {
            let mut p = 0usize;
            while p < consumed.len() {
                if picked[p] == called_id {
                    let (t, class) = class_of_canonical(sorted[p]);
                    let base = t * 4;
                    let mut cands = [0u8; 4];
                    let cn: usize = match class {
                        TileClass::Aka => {
                            cands[0] = base;
                            1
                        }
                        TileClass::PlainFive => {
                            cands[0] = base + 1;
                            cands[1] = base + 2;
                            cands[2] = base + 3;
                            3
                        }
                        TileClass::Norm => {
                            if is_five_type(t) {
                                return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
                            }
                            cands[0] = base;
                            cands[1] = base + 1;
                            cands[2] = base + 2;
                            cands[3] = base + 3;
                            4
                        }
                    };
                    let mut swapped = false;
                    let mut ci = 0usize;
                    while ci < cn {
                        let cand = cands[ci];
                        if cand != called_id {
                            let mut used = false;
                            let mut q = 0usize;
                            while q < consumed.len() {
                                if picked[q] == cand {
                                    used = true;
                                    break;
                                }
                                q += 1;
                            }
                            if !used {
                                picked[p] = cand;
                                swapped = true;
                                break;
                            }
                        }
                        ci += 1;
                    }
                    if !swapped {
                        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
                    }
                }
                p += 1;
            }
        }
        let mut a = 0usize;
        while a < consumed.len() {
            let mut b = a + 1;
            while b < consumed.len() {
                if picked[b] < picked[a] {
                    picked.swap(a, b);
                }
                b += 1;
            }
            a += 1;
        }
        return Ok(picked);
    }
    // Pick the lowest HELD take rendering each logged string (oracle
    // min-match: the engine consumes its lowest copies and `_match_claim`
    // keeps the min consumed set). Claim-local slot use never picks one
    // take twice.
    let mut picked = [0u8; 4];
    let mut used = [[false; 4]; 34];
    let mut i = 0usize;
    while i < consumed.len() {
        let want =
            TileCodec::bytes_of(sorted[i]).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        let t = (sorted[i] / 4) as usize;
        if t >= 34 {
            return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
        }
        let mut best: Option<(u8, usize)> = None;
        let mut c = 0usize;
        while c < 4 {
            if !used[t][c] {
                let v = l.hands[seat as usize][t][c];
                if v != 0 && TileCodec::bytes_of(v - 1) == Some(want) {
                    let take = v - 1;
                    if best.map(|b| take < b.0).unwrap_or(true) {
                        best = Some((take, c));
                    }
                }
            }
            c += 1;
        }
        let (take, slot) = best.ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        used[t][slot] = true;
        picked[i] = take;
        i += 1;
    }
    // The called take sits in the river, never in the hand: a pick equal to
    // it is a conservation violation (double-held take), never resolved.
    // (Removal re-resolves pass `called=None`: the canonical report take may
    // coincide with a held take by design; exact-take removal guards state.)
    if let Some(called_id) = called {
        let mut p = 0usize;
        while p < consumed.len() {
            if picked[p] == called_id {
                return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
            }
            p += 1;
        }
    }
    // Sort picked ascending.
    let mut a = 0usize;
    while a < consumed.len() {
        let mut b = a + 1;
        while b < consumed.len() {
            if picked[b] < picked[a] {
                picked.swap(a, b);
            }
            b += 1;
        }
        a += 1;
    }
    Ok(picked)
}

pub(crate) fn claim_history_kind(kind: u8) -> u8 {
    match kind {
        KIND_CHI => H_CHI,
        KIND_PON => H_PON,
        _ => H_DAIMINKAN,
    }
}

pub(crate) fn claim_meld_kind(kind: u8) -> u8 {
    match kind {
        KIND_CHI => ACT_CHI,
        KIND_PON => ACT_PON,
        _ => ACT_DAIMINKAN,
    }
}

pub(crate) fn do_claim(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    st: &mut WalkState,
    out: &mut RowSink<'_>,
    ev_all: &[StackedEvent<'_>],
    idx: usize,
    kind: u8,
) -> WalkResult<()> {
    let ev = &ev_all[idx];
    if ev.actor > 3 || ev.target > 3 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    if ev.pai == NO_PAI {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let actor = ev.actor;
    let (discarder, called) = match st.last_discard {
        Some(pair) => pair,
        None => return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER)),
    };
    if ev.target != discarder {
        return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
    }
    // The claim must name the offered tile (render-equal, red-sensitive).
    if TileCodec::bytes_of(called) != TileCodec::bytes_of(ev.pai) {
        return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
    }
    let needed = if kind == KIND_DAIMINKAN { 3 } else { 2 };
    let mut consumed_ids = [0u8; 4];
    let mut i = 0u8;
    while i < ev.consumed.1 {
        consumed_ids[i as usize] = ev.consumed.0[i as usize];
        i += 1;
    }
    let picked = resolve_consumed(
        ctx,
        l,
        idx,
        actor,
        &consumed_ids[..ev.consumed.1 as usize],
        needed,
        Some(called),
        l.fold_reports,
    )?;
    // Removal always resolves true takes: folded reports name canonical
    // copies the seat may not hold, so re-resolve unfolded for state. The
    // canonical report take may coincide with a held take by design, so the
    // removal pass skips the called-collision tripwire (`None`); exact-take
    // removal still fails closed on unheld takes.
    let removal = if l.fold_reports {
        resolve_consumed(
            ctx,
            l,
            idx,
            actor,
            &consumed_ids[..ev.consumed.1 as usize],
            needed,
            None,
            false,
        )?
    } else {
        picked
    };
    let offset = if kind == KIND_CHI {
        -1
    } else {
        source_offset(discarder, actor)
    };
    let act = claim_meld_kind(kind);
    let chosen = chosen_claim_id(act, called, &picked[..needed], offset).ok_or(rej(
        ctx.game_idx,
        idx,
        WALK_ACTION_UNRESOLVED,
    ))?;
    // Responder mask: fold regime offers exactly {pass, chosen} (the SIM
    // oracle min-picks a singleton); unfold regime offers every engine
    // variant (all take-subset / pattern claim actions) plus pass.
    let mut mask = [0u32; 32];
    mask[0] = PASS_ID;
    let mut mlen = 1usize;
    if !l.fold_reports {
        let ct = (called / 4) as usize;
        if act == ACT_PON || act == ACT_DAIMINKAN {
            // All 2-subsets of held takes of the called type.
            let mut takes = [0u8; 8];
            let tn = held_takes(
                &l.hands[actor as usize][ct.min(33)],
                u8::try_from(ct.min(33)).unwrap_or(0),
                &mut takes,
            );
            let mut a = 0usize;
            while a < tn {
                let mut b = a + 1;
                while b < tn {
                    if mlen < 32
                        && let Some(id) =
                            chosen_claim_id(ACT_PON, called, &[takes[a], takes[b]], offset)
                        {
                            mask[mlen] = id;
                            mlen += 1;
                        }
                    b += 1;
                }
                a += 1;
            }
            // Daiminkan alternative whenever three copies are held.
            if tn >= 3 {
                let mut quad = [takes[0], takes[1], takes[2]];
                quad.sort_unstable();
                if mlen < 32
                    && let Some(id) = chosen_claim_id(ACT_DAIMINKAN, called, &quad, offset) {
                        mask[mlen] = id;
                        mlen += 1;
                    }
            }
        } else if act == ACT_CHI {
            // All (pattern × take-combo) variants over held takes.
            let tc = called / 4;
            if tc < 27 {
                let suit = tc / 9;
                let rank = tc % 9 + 1;
                let mut p = 0usize;
                while p < 3 {
                    let (lo, hi) = [
                        (rank.saturating_sub(2), rank.saturating_sub(1)),
                        (rank - 1, rank + 1),
                        (rank + 1, rank + 2),
                    ][p];
                    p += 1;
                    if lo < 1 || hi > 9 || lo == rank || hi == rank {
                        continue;
                    }
                    let t0 = suit * 9 + lo - 1;
                    let t1 = suit * 9 + hi - 1;
                    let mut legs0 = [0u8; 8];
                    let mut legs1 = [0u8; 8];
                    let n0 = held_takes(&l.hands[actor as usize][t0 as usize], t0, &mut legs0);
                    let n1 = held_takes(&l.hands[actor as usize][t1 as usize], t1, &mut legs1);
                    let mut i = 0usize;
                    while i < n0 {
                        let mut j = 0usize;
                        while j < n1 {
                            let (c0, c1) = if legs0[i] < legs1[j] {
                                (legs0[i], legs1[j])
                            } else {
                                (legs1[j], legs0[i])
                            };
                            if mlen < 32
                                && let Some(id) = chosen_claim_id(ACT_CHI, called, &[c0, c1], -1) {
                                    mask[mlen] = id;
                                    mlen += 1;
                                }
                            j += 1;
                        }
                        i += 1;
                    }
                }
            }
        }
        // Duplicate ids collapse. PASS_ID is 0, below every action id, so
        // sorting the whole prefix keeps pass first. Singleton stays as-is.
        if mlen > 1 {
            isort_u32(&mut mask, mlen);
            let mut w = 0usize;
            let mut r = 1usize;
            while r < mlen {
                if mask[r] != mask[w] {
                    w += 1;
                    mask[w] = mask[r];
                }
                r += 1;
            }
            mlen = w + 1;
        }
    }
    let mlen8 = insert_chosen(&mut mask, mlen, chosen);
    // Claims ride no draw turn: abilities read false (clear BEFORE mint).
    l.riichi_offered = false;
    l.tsumo_offered = false;
    mint_row(
        l,
        st,
        out,
        actor,
        PHASE_RESPONSE,
        discarder,
        chosen,
        &mask[..mlen8 as usize],
    );
    l.drawn_live[actor as usize] = None;
    // The call breaks every open ippatsu window (post-mint observation).
    l.ippatsu = [0u8; 4];
    // Meld record (sorted takes, like the oracle's VisibleMeld).
    let mut tiles = [0u8; 4];
    let mut i = 0usize;
    while i < needed {
        tiles[i] = picked[i];
        i += 1;
    }
    tiles[needed] = called;
    let total = needed + 1;
    let mut a = 0usize;
    while a < total {
        let mut b = a + 1;
        while b < total {
            if tiles[b] < tiles[a] {
                tiles.swap(a, b);
            }
            b += 1;
        }
        a += 1;
    }
    // Remove consumed takes from the hand (class ranks, seat-local).
    let mut i = 0usize;
    while i < needed {
        let take = removal[i];
        remove_take(l, actor, take).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        i += 1;
    }
    let s = actor as usize;
    if l.meld_lens[s] >= 4 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    l.melds[s][l.meld_lens[s] as usize] = Meld {
        kind: act,
        owner: actor,
        tiles: (tiles, u8::try_from(total).unwrap_or(0)),
    };
    l.meld_lens[s] += 1;
    l.drawer = Some(actor);
    l.exp_drawer = Some(actor);
    l.kan_pending = kind == KIND_DAIMINKAN;
    if kind == KIND_DAIMINKAN {
        l.kan_count = l.kan_count.saturating_add(1);
    }
    // Arm (pon/chi) or disarm (daiminkan: rinshan pending) the post-claim
    // conservation tripwire.
    st.last_claim = if kind == KIND_DAIMINKAN {
        None
    } else {
        Some((actor, called))
    };
    st.push_public(claim_history_kind(kind));
    st.last_discard = None;
    Ok(())
}

pub(crate) fn do_ankan(
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
    if st.decided {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    let actor = ev.actor;
    let s = actor as usize;
    check_drawer(l, ctx.game_idx, idx, actor)?;
    let mut consumed_ids = [0u8; 4];
    let mut i = 0u8;
    while i < ev.consumed.1 {
        consumed_ids[i as usize] = ev.consumed.0[i as usize];
        i += 1;
    }
    let picked = resolve_consumed(
        ctx,
        l,
        idx,
        actor,
        &consumed_ids[..ev.consumed.1 as usize],
        4,
        None,
        l.fold_reports,
    )?;
    let base = (picked[0] / 4) * 4;
    if picked != [base, base + 1, base + 2, base + 3] {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    let chosen = chosen_claim_id(ACT_ANKAN, 0, &picked, 0).ok_or(rej(
        ctx.game_idx,
        idx,
        WALK_ACTION_UNRESOLVED,
    ))?;
    // Offer + match check (logged quad must be offered, else desync).
    let mut offer = [0u32; 32];
    let olen = draw_offer_ids(&mut *l, st, actor, &mut offer);
    if l.riichi[s] != 0 {
        // Kans need a live draw turn; without one the log post-claim kan
        // fails closed.
        let live_drawn = l.drawn_col[s].ok_or(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC))?;
        // Forced pair + same-wait offer quads + logged quad (log authority).
        let mut combo = [0u32; 32];
        let rep_drawn = if l.fold_reports {
            canonical_take(live_drawn)
        } else {
            live_drawn
        };
        let a = chosen_id(ACT_DISCARD, rep_drawn, 0);
        let b = chosen_id(ACT_TSUMOGIRI, rep_drawn, 0);
        combo[0] = a;
        combo[1] = b;
        let mut cn = if a == b { 1 } else { 2 };
        let mut i = 0usize;
        while i < olen {
            if offer[i] >= BASE_ANKAN && offer[i] < BASE_ANKAN + 34
                && cn < 32 {
                    combo[cn] = offer[i];
                    cn += 1;
                }
            i += 1;
        }
        let mut has = false;
        let mut i = 0usize;
        while i < cn {
            if combo[i] == chosen {
                has = true;
                break;
            }
            i += 1;
        }
        if !has {
            if l.live_wall() < KAN_MIN_WALL {
                return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
            }
            if cn < 32 {
                combo[cn] = chosen;
                cn += 1;
            }
        }
        isort_u32(&mut combo, cn);
        let mlen8 = insert_chosen(&mut combo, cn, chosen);
        mint_row(
            l,
            st,
            out,
            actor,
            PHASE_DRAW,
            actor,
            chosen,
            &combo[..mlen8 as usize],
        );
    } else {
        let mut has = false;
        let mut i = 0usize;
        while i < olen {
            if offer[i] == chosen {
                has = true;
                break;
            }
            i += 1;
        }
        if !has {
            return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
        }
        let mlen8 = insert_chosen(&mut offer, olen, chosen);
        mint_row(
            l,
            st,
            out,
            actor,
            PHASE_DRAW,
            actor,
            chosen,
            &offer[..mlen8 as usize],
        );
    }
    l.drawn_live[s] = None;
    let mut i = 0usize;
    while i < 4 {
        let take = picked[i];
        remove_take(l, actor, take).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        i += 1;
    }
    if l.meld_lens[s] >= 4 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    l.melds[s][l.meld_lens[s] as usize] = Meld {
        kind: ACT_ANKAN,
        owner: actor,
        tiles: (picked, 4),
    };
    l.meld_lens[s] += 1;
    l.drawer = Some(actor);
    l.exp_drawer = Some(actor);
    l.kan_pending = true;
    l.kan_count = l.kan_count.saturating_add(1);
    l.ippatsu = [0u8; 4];
    st.last_claim = None;
    st.push_public(H_ANKAN);
    Ok(())
}

pub(crate) fn do_kakan(
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
    // The added copy is the one pool copy of this type absent from the
    // prior pon triple (deterministic, contract-valid).
    let added = {
        let atype = ev.pai / 4;
        let mut triple: Option<([u8; 4], u8)> = None;
        let mut m = 0u8;
        while m < l.meld_lens[s] {
            let meld = &l.melds[s][m as usize];
            if meld.kind == ACT_PON && meld.tiles.0[0] / 4 == atype {
                triple = Some(meld.tiles);
                break;
            }
            m += 1;
        }
        let (tiles, len) = triple.ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        // Pool: full block for plain/normal strings, aka singleton for red.
        let (_, ev_class) = class_of_canonical(ev.pai);
        let base = atype * 4;
        let mut missing: Option<u8> = None;
        if is_five_type(atype) && ev_class == TileClass::Aka {
            let mut has = false;
            let mut k = 0u8;
            while k < len {
                if tiles[k as usize] == base {
                    has = true;
                }
                k += 1;
            }
            if !has {
                missing = Some(base);
            }
        } else {
            let mut nmiss = 0u8;
            let mut c = 0u8;
            while c < 4 {
                let id = base + c;
                let mut has = false;
                let mut k = 0u8;
                while k < len {
                    if tiles[k as usize] == id {
                        has = true;
                    }
                    k += 1;
                }
                if !has {
                    missing = Some(id);
                    nmiss += 1;
                }
                c += 1;
            }
            if nmiss != 1 {
                missing = None;
            }
        }
        missing.ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?
    };
    let chosen = chosen_id(ACT_KAKAN, added, 0);
    if chosen == UNRESOLVED_ID {
        return Err(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED));
    }
    let mut buf = [0u32; 32];
    let mlen = draw_offer_ids(&mut *l, st, actor, &mut buf);
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
    l.drawn_live[s] = None;
    // The added copy leaves the hand (the meld record keeps the pon).
    // Reports may name a canonical copy (fold regime); removal resolves
    // the true held take rendering the same string (exactly one: the pon
    // holds the other three).
    {
        let state =
            min_held_take(l, actor, added).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        remove_take(l, actor, state).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
    }
    l.drawer = Some(actor);
    l.exp_drawer = Some(actor);
    l.kan_pending = true;
    l.kan_count = l.kan_count.saturating_add(1);
    l.ippatsu = [0u8; 4];
    // Chankan ron offers passed (only ron answers a kakan; a taken ron
    // ends the hand, so no claim exclusion applies).
    {
        let (_, _, ron) = eval_window(l, actor, added, true);
        let mut m = 0u8;
        while m < 4 {
            if m != actor && ron[m as usize] {
                l.missed[m as usize] = 1;
            }
            m += 1;
        }
    }
    st.last_claim = None;
    st.push_public(H_KAKAN);
    st.last_discard = Some((actor, added));
    st.last_discard_idx = idx;
    st.opened_by_discard = false;
    // Kan windows never emit envelopes; responders still resolve upstream.
    open_window(l, st, ev_all, actor, added, true, idx + 1, idx);
    Ok(())
}
