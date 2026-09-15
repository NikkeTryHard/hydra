use super::handlers_claim::{do_ankan, do_claim, do_kakan};
use super::handlers_close::{check_mode_for, do_hora, do_ryukyoku};
use super::handlers_core::rej;
use super::handlers_core::{do_dahai, do_reach, do_start_kyoku, do_tsumo};
use super::spans::WalkState;
use crate::ingest::{
    KIND_ANKAN, KIND_CHI, KIND_DAHAI, KIND_DAIMINKAN, KIND_DORA, KIND_END, KIND_END_KYOKU,
    KIND_HORA, KIND_KAKAN, KIND_OTHER, KIND_PON, KIND_REACH, KIND_REACH_ACCEPTED, KIND_RYUKYOKU,
    KIND_START, KIND_START_KYOKU, KIND_TRANSPARENT_OTHER, KIND_TSUMO, StackedEvent,
};
use crate::ledger::{
    GameCtx, H_DORA, H_RIICHI_ACCEPTED, H_ROUND_END, Ledger, RowSink, WALK_BARE_DORA,
    WALK_PAST_TERMINAL, WALK_TILE_CONSERVATION, WALK_TURN_ORDER, WALK_UNKNOWN_EVENT, WalkReject,
};
use crate::tiles::TileCodec;

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Skip kinds for reach-collapse lookahead (mirrors `SKIP_TYPES`).
fn is_skip_kind(kind: u8) -> bool {
    matches!(
        kind,
        KIND_START_KYOKU
            | KIND_TSUMO
            | KIND_DORA
            | KIND_REACH_ACCEPTED
            | KIND_RYUKYOKU
            | KIND_END_KYOKU
            | KIND_START
    )
}

/// Row kinds (mirrors `ROW_TYPES`, == `should_sample` set).
fn is_row_kind(kind: u8) -> bool {
    matches!(
        kind,
        KIND_DAHAI | KIND_CHI | KIND_PON | KIND_DAIMINKAN | KIND_ANKAN | KIND_KAKAN | KIND_HORA
    )
}

/// Walk one framed game DIRECT into the sink's minimal planes.
///
/// `ev` must be gate-accepted (`gate_game` verdict `Sample`; gate owns
/// bare-dora / double-ron / hora-shape skips). Returns the row count
/// (`seq`, `u16`) — numeric keys only, no `format!` ids. Rows are staged
/// in emission order; on `Err` the sink may hold a PREFIX (the driver
/// must reset-or-discard it: stage-then-commit lives in `feed::fill`).
pub fn walk_game(
    ctx: &GameCtx<'_>,
    ev: &[StackedEvent<'_>],
    ledger: &mut Ledger,
    out: &mut RowSink<'_>,
) -> Result<u16, WalkReject> {
    let mut st = WalkState::new();
    // Report regime: wall-less rows fold to pool-first canonical takes (the
    // SIM oracle reports canonical copies); walled rows report true takes
    // (the wall oracle encodes held copies). State stays true takes either
    // way; only offers/chosen fold.
    ledger.fold_reports = !ctx.is_walled();
    let total = ev.len();
    let mut idx = 0usize;
    while idx < total {
        let kind = ev[idx].kind;
        if kind == KIND_END {
            if !ledger.kyoku_active {
                return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
            }
            st.terminal = true;
            break;
        }
        if kind == KIND_START_KYOKU {
            do_start_kyoku(ctx, ledger, &mut st, idx, ev[idx].span)?;
            idx += 1;
            continue;
        }
        if kind == KIND_START {
            idx += 1;
            continue;
        }
        if !ledger.kyoku_active {
            return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
        }
        match kind {
            KIND_RYUKYOKU => {
                do_ryukyoku(ctx, ledger, &mut st, idx, ev[idx].span)?;
                idx += 1;
            }
            KIND_END_KYOKU => {
                st.push_public(H_ROUND_END);
                ledger.drawer = None;
                st.hora_terminal = false;
                idx += 1;
            }
            KIND_TSUMO => {
                do_tsumo(ctx, ledger, &mut st, idx, &ev[idx])?;
                idx += 1;
            }
            KIND_DORA => match ev[idx].dora_span {
                None => return Err(rej(ctx.game_idx, idx, WALK_BARE_DORA)),
                Some(marker) if marker.is_empty() => {
                    return Err(rej(ctx.game_idx, idx, WALK_BARE_DORA));
                }
                Some(marker) => {
                    let pid = TileCodec::physical(marker)
                        .map_err(|_| rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
                    if ledger.dora_len < 5 {
                        ledger.dora[ledger.dora_len as usize] = pid as i8;
                        ledger.dora_len += 1;
                    }
                    st.push_public(H_DORA);
                    idx += 1;
                }
            },
            KIND_REACH_ACCEPTED => {
                if ev[idx].actor > 3 {
                    return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
                }
                // First accept per seat per kyoku: accepted state, table
                // stick, ippatsu window (oracle deltas). Repeat accepts are
                // absorbed (unfold auto-accepts at declaration, so a later
                // accept must not double-count the stick).
                let s = ev[idx].actor as usize;
                let first = ledger.accepted[s] == 0;
                if first {
                    ledger.accepted[s] = 1;
                    ledger.ippatsu[s] = 1;
                }
                // Unfold auto-accepts at declaration (history already carries
                // it): a later accept event must not duplicate it. Fold keeps
                // every push (SIM parity).
                if first || !ctx.is_walled() {
                    st.push_public(H_RIICHI_ACCEPTED);
                }
                idx += 1;
            }
            KIND_REACH => {
                // Collapse with the following declaration dahai.
                let mut nxt = idx + 1;
                let decl = loop {
                    if nxt >= total {
                        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
                    }
                    let k = ev[nxt].kind;
                    if k == KIND_END || is_row_kind(k) || k == KIND_REACH {
                        break nxt;
                    }
                    if !is_skip_kind(k) {
                        return Err(rej(ctx.game_idx, nxt, WALK_UNKNOWN_EVENT));
                    }
                    nxt += 1;
                };
                if ev[decl].kind != KIND_DAHAI {
                    return Err(rej(ctx.game_idx, decl, WALK_TURN_ORDER));
                }
                if ctx.is_walled() && ev[idx].actor <= 3 {
                    check_mode_for(ctx, ledger, &mut st, ev, idx, KIND_REACH, ev[idx].actor)?;
                }
                do_reach(ctx, ledger, &mut st, out, ev, idx, decl)?;
                idx = decl + 1;
            }
            k if is_row_kind(k) => {
                if ctx.is_walled() && ev[idx].actor <= 3 {
                    check_mode_for(ctx, ledger, &mut st, ev, idx, k, ev[idx].actor)?;
                }
                match k {
                    KIND_CHI | KIND_PON | KIND_DAIMINKAN => {
                        do_claim(ctx, ledger, &mut st, out, ev, idx, k)?;
                    }
                    KIND_ANKAN => {
                        do_ankan(ctx, ledger, &mut st, out, ev, idx)?;
                    }
                    KIND_KAKAN => {
                        do_kakan(ctx, ledger, &mut st, out, ev, idx)?;
                    }
                    KIND_HORA => {
                        do_hora(ctx, ledger, &mut st, out, ev, idx)?;
                    }
                    _ => {
                        do_dahai(ctx, ledger, &mut st, out, ev, idx)?;
                    }
                }
                idx += 1;
            }
            KIND_TRANSPARENT_OTHER => {
                idx += 1;
            }
            KIND_OTHER => {
                return Err(rej(ctx.game_idx, idx, WALK_UNKNOWN_EVENT));
            }
            _ => {
                return Err(rej(ctx.game_idx, idx, WALK_UNKNOWN_EVENT));
            }
        }
    }
    if !st.terminal {
        return Err(rej(ctx.game_idx, total, WALK_TURN_ORDER));
    }
    // Hora with no closing end_kyoku: the wall oracle is terminal and
    // rejects the trailing events (ron_win parity: quarantine agreement).
    // Unfold-only: the SIM oracle has no terminal rule (it walks the same
    // shape), so fold keeps the old accept.
    if st.hora_terminal && ctx.is_walled() {
        return Err(rej(ctx.game_idx, idx, WALK_PAST_TERMINAL));
    }
    Ok(if st.seq > u16::MAX as u32 {
        u16::MAX
    } else {
        st.seq as u16
    })
}
