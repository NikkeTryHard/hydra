use super::ids::{
    ABORT_NINE_TERMINALS_ID, BASE_ANKAN, BASE_DISCARD, BASE_KAKAN, BASE_RIICHI, BASE_TSUMOGIRI,
    KAN_MIN_WALL, KYUSHU_DISTINCT_YAOCHU, RIICHI_MIN_SCORE, RIICHI_MIN_WALL, UNRESOLVED_ID,
    YAOCHU_TYPES, chosen_id,
};
use super::shape::{isort_u32, tenpai_discards, win_shape_14};
use super::spans::WalkState;
use crate::ledger::{
    ACT_ANKAN, ACT_PON, ACT_TSUMO, Ledger, TileClass, aka_suit, class_of_canonical, held_takes,
    is_aka_id, is_five_type,
};
use crate::tiles::TileCodec;

// ---------------------------------------------------------------------------
// Ledger reads (presence → takes / counts / offers).
// ---------------------------------------------------------------------------

/// Collapsed takes of one seat into `out` (ascending by type). Returns len.
pub(crate) fn seat_takes(l: &Ledger, seat: u8, out: &mut [u8; 24]) -> usize {
    let s = (seat & 3) as usize;
    let mut n = 0usize;
    let mut t = 0u8;
    while t < 34 {
        let mut cell = [0u8; 8];
        let k = held_takes(&l.hands[s][t as usize], t, &mut cell);
        let mut i = 0usize;
        while i < k {
            if n < out.len() {
                out[n] = cell[i];
                n += 1;
            }
            i += 1;
        }
        t += 1;
    }
    n
}

/// Concealed type counts for the row plane (live draw excluded).
pub(crate) fn concealed_counts(l: &Ledger, seat: u8, out: &mut [u8; 34]) {
    let s = (seat & 3) as usize;
    let mut t = 0usize;
    while t < 34 {
        let row = &l.hands[s][t];
        out[t] =
            (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8;
        t += 1;
    }
    if l.drawn_live[s].is_some()
        && let Some(dc) = l.drawn_col[s] {
            let dt = (dc / 4) as usize;
            if dt < 34 {
                out[dt] = out[dt].saturating_sub(1);
            }
        }
}

/// Visible type counts for the row plane (rivers + meld tiles, clamp 4).
pub(crate) fn visible_counts(l: &Ledger, out: &mut [u8; 34]) {
    let mut i = 0usize;
    while i < 34 {
        out[i] = 0;
        i += 1;
    }
    let mut s = 0usize;
    while s < 4 {
        let mut j = 0u8;
        while j < l.rivers_len[s] {
            let t = (l.rivers[s][j as usize] / 4) as usize;
            if t < 34 && out[t] < 9 {
                out[t] += 1;
            }
            j += 1;
        }
        let mut m = 0u8;
        while m < l.meld_lens[s] {
            let meld = &l.melds[s][m as usize];
            let (tiles, len) = meld.tiles;
            let mut k = 0u8;
            while k < len {
                let t = (tiles[k as usize] / 4) as usize;
                if t < 34 && out[t] < 9 {
                    out[t] += 1;
                }
                k += 1;
            }
            m += 1;
        }
        s += 1;
    }
    let mut t = 0usize;
    while t < 34 {
        if out[t] > 4 {
            out[t] = 4;
        }
        t += 1;
    }
}

/// Type counts over one seat's present takes (offer/window reads).
pub(crate) fn seat_type_counts(l: &Ledger, seat: u8, out: &mut [u8; 34]) {
    let s = (seat & 3) as usize;
    let mut t = 0usize;
    while t < 34 {
        let row = &l.hands[s][t];
        out[t] =
            (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8;
        t += 1;
    }
}

/// River type set (furiten proxy reads).
pub(crate) fn river_type_set(l: &Ledger, seat: u8, out: &mut [bool; 34]) {
    let mut t = 0usize;
    while t < 34 {
        out[t] = false;
        t += 1;
    }
    let s = seat as usize;
    let mut j = 0u8;
    while j < l.rivers_len[s] {
        let t = (l.rivers[s][j as usize] / 4) as usize;
        if t < 34 {
            out[t] = true;
        }
        j += 1;
    }
}

// ---------------------------------------------------------------------------
// Take allocation + removal (string-ledger semantics, zero heap).
// ---------------------------------------------------------------------------

/// Allocate the next take copy for a canonical ingest id into `seat`.
///
/// Mirrors `take_next` pool order: aka strings take the aka copy (cap 1),
/// plain-five strings take base+1.. (cap 3), everything else takes the
/// block in order (cap 4). The returned take id is the global (true) copy;
/// seat presence uses rank slots (lowest free of the class), so collapse
/// stays a pure rank read. `None` = pool overused (conservation).
pub(crate) fn alloc_take(l: &mut Ledger, seat: u8, pai: u8) -> Option<u8> {
    if pai >= 136 || seat > 3 {
        return None;
    }
    let s = seat as usize;
    let (t, class) = class_of_canonical(pai);
    let ti = t as usize;
    // Lowest free slot of the class (keeps ranks compacted low).
    let slot = |lo: u8, hi: u8| -> Option<usize> {
        let mut c = lo;
        while c < hi {
            if l.hands[s][ti][c as usize] == 0 {
                return Some(c as usize);
            }
            c += 1;
        }
        None
    };
    match class {
        TileClass::Aka => {
            let i = aka_suit(t);
            if l.used_aka[i] >= 1 {
                return None;
            }
            let c = slot(0, 1)?;
            l.used_aka[i] += 1;
            l.hands[s][ti][c] = t * 4 + 1;
            Some(t * 4)
        }
        TileClass::PlainFive => {
            let k = l.used_n[ti];
            if k >= 3 {
                return None;
            }
            let c = slot(1, 4)?;
            l.used_n[ti] += 1;
            l.hands[s][ti][c] = t * 4 + 1 + k + 1;
            Some(t * 4 + 1 + k)
        }
        TileClass::Norm => {
            let k = l.used_n[ti];
            if k >= 4 {
                return None;
            }
            let c = slot(0, 4)?;
            l.used_n[ti] += 1;
            l.hands[s][ti][c] = t * 4 + k + 1;
            Some(t * 4 + k)
        }
    }
}
// ---------------------------------------------------------------------------
// Draw offers → compact legal.
// ---------------------------------------------------------------------------

/// Remove the exact held `take` from `seat` (true-take removal).
///
/// Cells store take id + 1, so the slot holding this take clears; any other
/// copy of the string stays. Fails when the seat does not hold the take
/// (conservation tripwire — the oracle only discards/consumes held copies).
pub(crate) fn remove_take(l: &mut Ledger, seat: u8, take: u8) -> Option<()> {
    if seat > 3 || take >= 136 {
        return None;
    }
    let s = seat as usize;
    let ti = (take / 4) as usize;
    if ti >= 34 {
        return None;
    }
    let want = take + 1;
    let mut c = 0usize;
    while c < 4 {
        if l.hands[s][ti][c] == want {
            l.hands[s][ti][c] = 0;
            return Some(());
        }
        c += 1;
    }
    None
}

/// Present-take count of one hand row (cells store take id + 1, 0 == empty).
#[inline]
pub(crate) fn presence_count(row: &[u8; 4]) -> u8 {
    (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8
}

/// Lowest held take of `seat` rendering the same string as canonical `pai`.
///
/// Oracle min-match (`_match_dahai`/`_match_kakan` keep the lowest take id
/// per string; aka-sensitive via byte render). `None` when the seat holds
/// no such copy (conservation tripwire).
pub(crate) fn min_held_take(l: &Ledger, seat: u8, pai: u8) -> Option<u8> {
    if pai >= 136 || seat > 3 {
        return None;
    }
    let want = TileCodec::bytes_of(pai)?;
    let t = (pai / 4) as usize;
    if t >= 34 {
        return None;
    }
    let row = &l.hands[(seat & 3) as usize][t];
    let mut best: Option<u8> = None;
    let mut c = 0usize;
    while c < 4 {
        let v = row[c];
        if v != 0 {
            let take = v - 1;
            if TileCodec::bytes_of(take) == Some(want) && best.map(|b| take < b).unwrap_or(true) {
                best = Some(take);
            }
        }
        c += 1;
    }
    best
}
/// Nine-terminals abort offer for qualifying first draws (live parity).
///
/// First draw of the seat (`tsumo_cnt == 1`) with no melds anywhere and 9+
/// distinct terminal/honor types in the seat-local 14 takes (concealed takes
/// including the live draw): appends [`ABORT_NINE_TERMINALS_ID`]. The count
/// MUST be seat-local: `tehai_cnt` is global install state and would credit
/// other seats' terminals. Mirrors the live engine and the V2 reference gate
/// (first draw, no claims, 9+ yaochu); plain dahai rows only (riichi/kan/hora
/// rows keep V1 behavior — no observed live counterpart). Returns the
/// extended count (unchanged when capped or unqualified).
pub(crate) fn kyushu_offer_append(l: &Ledger, seat: u8, buf: &mut [u32; 32], n: usize) -> usize {
    let s = (seat & 3) as usize;
    if l.tsumo_cnt[s] != 1 {
        return n;
    }
    let mut m = 0usize;
    while m < 4 {
        if l.meld_lens[m] != 0 {
            return n;
        }
        m += 1;
    }
    let mut takes = [0u8; 24];
    let tn = seat_takes(l, seat, &mut takes);
    let mut kinds = [false; 34];
    let mut i = 0usize;
    while i < tn {
        let t = (takes[i] / 4) as usize;
        if t < 34 {
            kinds[t] = true;
        }
        i += 1;
    }
    let mut distinct = 0usize;
    let mut k = 0usize;
    while k < YAOCHU_TYPES.len() {
        if kinds[YAOCHU_TYPES[k] as usize] {
            distinct += 1;
        }
        k += 1;
    }
    if distinct < KYUSHU_DISTINCT_YAOCHU {
        return n;
    }
    if n < buf.len() {
        buf[n] = ABORT_NINE_TERMINALS_ID;
        return n + 1;
    }
    n
}
/// Canonical twin preserving held rank (fold regime): the k-th held take of
/// a string reports the k-th pool copy (aka → base always). `hand` is the
/// seat's take list; `take` must be a member. Mirrors the old collapse,
/// which assigned first-k pool copies in slot (allocation) order.
pub(crate) fn canonical_rank_take(hand: &[u8], take: u8) -> u8 {
    let t = take / 4;
    if is_aka_id(take) {
        return t * 4;
    }
    let mut rank = 0u8;
    for h in hand {
        if *h == take {
            break;
        }
        if h / 4 == t && !is_aka_id(*h) {
            rank += 1;
        }
    }
    if is_five_type(t) {
        t * 4 + 1 + rank
    } else {
        t * 4 + rank
    }
}

/// Collapse a true take to its pool-first report twin (fold regime).
/// Aka stays, plain five → base+1, else base. Mirrors the SIM oracle's
/// canonical renders.
#[inline]
pub(crate) fn canonical_take(take: u8) -> u8 {
    let t = take / 4;
    if is_aka_id(take) {
        take
    } else if is_five_type(t) {
        t * 4 + 1
    } else {
        t * 4
    }
}

/// Discard offers: fold regime renders one pool-first id per held string
/// (SIM oracle); unfold regime offers every held take (wall oracle).
/// Plus the tsumogiri twin + riichi candidates (gated) + ankan quads
/// (gated, same-wait under riichi) + kakan adds (gated). Sorted ascending,
/// stack-only. Returns the count (capped at 32).
pub(crate) fn draw_offer_ids(
    l: &mut Ledger,
    st: &WalkState,
    seat: u8,
    out: &mut [u32; 32],
) -> usize {
    if seat > 3 {
        return 0;
    }
    // Ability stash (can_riichi/can_tsumo mirrors the gated offers below):
    // reset here so every offer build leaves fresh flags for mint.
    l.riichi_offered = false;
    l.tsumo_offered = false;
    let s = seat as usize;
    // Wall oracle withholds the just-claimed string on the immediate
    // post-claim turn (a pon remainder reappears once a discard disarms the
    // tripwire; chi remainders of other strings stay offered). Fold regime
    // keeps per-string offers (SIM min-picks).
    let claim_take = match st.last_claim {
        Some((cseat, ctake)) if cseat == seat => Some(ctake),
        _ => None,
    };
    let post_claim = !l.fold_reports && claim_take.is_some();
    let claimed_type = claim_take.unwrap_or(0) / 4;
    let mut n = 0usize;
    let mut t = 0u8;
    while t < 34 {
        let row = &l.hands[s][t as usize];
        if l.fold_reports {
            // SIM oracle: one pool-first id per held string.
            if is_five_type(t) {
                if row[0] != 0
                    && n < 32 {
                        out[n] = BASE_DISCARD + (t * 4) as u32;
                        n += 1;
                    }
                if (row[1] != 0 || row[2] != 0 || row[3] != 0)
                    && n < 32 {
                        out[n] = BASE_DISCARD + (t * 4 + 1) as u32;
                        n += 1;
                    }
            } else if (row[0] != 0 || row[1] != 0 || row[2] != 0 || row[3] != 0)
                && n < 32 {
                    out[n] = BASE_DISCARD + (t * 4) as u32;
                    n += 1;
                }
        } else {
            // Wall oracle: one id per HELD take (true copies, ascending),
            // minus the just-claimed string on the post-claim turn.
            let mut takes = [0u8; 8];
            let tn = held_takes(row, t, &mut takes);
            let mut i = 0usize;
            while i < tn {
                let take = takes[i];
                let hide = post_claim && take / 4 == claimed_type;
                if !hide && n < 32 {
                    out[n] = BASE_DISCARD + take as u32;
                    n += 1;
                }
                i += 1;
            }
        }
        t += 1;
    }
    let live = l.drawn_live[s].is_some();
    if live
        && let Some(dc) = l.drawn_col[s] {
            let rep = if l.fold_reports {
                canonical_take(dc)
            } else {
                dc
            };
            if n < 32 {
                out[n] = BASE_TSUMOGIRI + rep as u32;
                n += 1;
            }
        }
    let wall = l.live_wall();
    let riichi = l.riichi[s] != 0;
    let open = l.meld_lens[s] as usize;
    // Riichi candidates behind the closed/score/wall/live gate.
    if live && !riichi && wall >= RIICHI_MIN_WALL && l.scores[s] >= RIICHI_MIN_SCORE {
        let mut closed = true;
        let mut m = 0u8;
        while m < l.meld_lens[s] {
            if l.melds[s][m as usize].kind != ACT_ANKAN {
                closed = false;
                break;
            }
            m += 1;
        }
        if closed {
            let mut takes = [0u8; 24];
            let tn = seat_takes(l, seat, &mut takes);
            let mut cands = [0u8; 16];
            let cn = tenpai_discards(&takes[..tn], open, &mut cands);
            // Ability flag mirrors the gated offer: candidates exist.
            l.riichi_offered = cn > 0;
            let mut i = 0usize;
            while i < cn {
                // Fold regime reports the rank-preserving canonical twin
                let rep = if l.fold_reports {
                    canonical_rank_take(&takes[..tn], cands[i])
                } else {
                    cands[i]
                };
                if n < 32 {
                    out[n] = BASE_RIICHI + rep as u32;
                    n += 1;
                }
                i += 1;
            }
        }
    }
    // Kan answers need a live draw turn + a live wall.
    if live && wall >= KAN_MIN_WALL {
        // Ankan quads among present takes (popcount 4 → canonical block).
        // P2: under riichi only the drawn type can pass the same-wait gate
        // (`has_drawn` is true for exactly one type), so check that type
        // alone; non-riichi keeps the full 34-type scan (same outputs: all
        // other types fail `has_drawn` under riichi).
        let drawn_true = l.drawn_live[s];
        if riichi {
            // Same-wait: no prior melds and the quad holds the
            // just-drawn take (faithful `melds.is_empty()` rule).
            let empty = l.meld_lens[s] == 0;
            if empty
                && let Some(d) = drawn_true {
                    let t = d / 4;
                    if t < 34 {
                        let row = &l.hands[s][t as usize];
                        let has_drawn = drawn_true.map(|d| d / 4 == t).unwrap_or(false);
                        if presence_count(row) == 4 && has_drawn && n < 32 {
                            out[n] = BASE_ANKAN + t as u32;
                            n += 1;
                        }
                    }
                }
        } else {
            let mut t = 0u8;
            while t < 34 {
                let row = &l.hands[s][t as usize];
                if presence_count(row) == 4 && n < 32 {
                    out[n] = BASE_ANKAN + t as u32;
                    n += 1;
                }
                t += 1;
            }
        }
        // Kakan: prior pon + fourth copy present, never under riichi.
        // P3: no melds means no pon to extend, so the scan is vacuous —
        // early-out keeps outputs identical (the loop would emit nothing).
        if !riichi && l.meld_lens[s] != 0 {
            let mut m = 0u8;
            while m < l.meld_lens[s] {
                let meld = &l.melds[s][m as usize];
                if meld.kind == ACT_PON {
                    let mt = meld.tiles.0[0] / 4;
                    let base = mt * 4;
                    let plain_held = if is_five_type(mt) {
                        l.hands[s][mt as usize][1] != 0
                            || l.hands[s][mt as usize][2] != 0
                            || l.hands[s][mt as usize][3] != 0
                    } else {
                        true
                    };
                    let mut missing: Option<u8> = None;
                    if is_five_type(mt) && !plain_held {
                        // Aka singleton pool: missing iff meld lacks it.
                        let mut has = false;
                        let mut k = 0u8;
                        while k < meld.tiles.1 {
                            if meld.tiles.0[k as usize] == base {
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
                            while k < meld.tiles.1 {
                                if meld.tiles.0[k as usize] == id {
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
                    if let Some(add) = missing {
                        // Take-level presence: the missing copy itself must be held.
                        let want = add + 1;
                        let hrow = &l.hands[s][mt as usize];
                        let held = hrow[0] == want
                            || hrow[1] == want
                            || hrow[2] == want
                            || hrow[3] == want;
                        if held && n < 32 {
                            out[n] = BASE_KAKAN + add as u32;
                            n += 1;
                        }
                    }
                }
                m += 1;
            }
        }
    }
    // Tsumo-win answer on winning draws (taken or not): the engine offers
    // tsumo on every winning draw turn. Tile folds in fold regime.
    if live {
        let mut takes = [0u8; 24];
        let tn = seat_takes(l, seat, &mut takes);
        if tn == 14 && win_shape_14(&takes[..tn], l.meld_lens[s] as usize) {
            // Ability flag mirrors the gated offer: winning draw turn.
            l.tsumo_offered = true;
            if let Some(dc) = l.drawn_col[s] {
                let rep = if l.fold_reports {
                    canonical_take(dc)
                } else {
                    dc
                };
                let id = chosen_id(ACT_TSUMO, rep, 0);
                if id != UNRESOLVED_ID && n < 32 {
                    out[n] = id;
                    n += 1;
                }
            }
        }
    }
    isort_u32(out, n.min(32));
    n.min(32)
}

/// Compact legal set for `seat`: sorted action ids + length.
///
/// K=32 mandatory (16 truncates): offers past 32 keep the smallest 32.
/// A 14-take hand bounds discards+twin to 15, so overflow needs
/// riichi+kan coincidences and stays deterministic when hit.
pub fn legal_ids_sorted(seat: u8, l: &mut Ledger, post_claim: Option<(u8, u8)>) -> ([u32; 32], u8) {
    let mut buf = [0u32; 32];
    if seat > 3 {
        return (buf, 0);
    }
    let st = WalkState {
        last_claim: post_claim,
        ..WalkState::new()
    };
    let n = draw_offer_ids(&mut *l, &st, seat, &mut buf);
    (buf, u8::try_from(n.min(32)).unwrap_or(u8::MAX))
}

/// Insert `chosen` into a sorted id buffer (draw-mask chosen forcing),
/// keeping ascending order; on overflow keeps chosen + smallest 31.
pub(crate) fn insert_chosen(buf: &mut [u32; 32], len: usize, chosen: u32) -> u8 {
    let mut i = 0usize;
    while i < len && i < 32 {
        if buf[i] == chosen {
            return u8::try_from(len.min(32)).unwrap_or(u8::MAX);
        }
        i += 1;
    }
    if len < 32 {
        // P4: `buf[..len]` arrives sorted (every caller sorts first), so an
        // insertion-shift keeps ascending order — identical to append + full
        // re-sort of the len+1 prefix, without re-scanning it.
        let mut pos = 0usize;
        while pos < len && buf[pos] < chosen {
            pos += 1;
        }
        let mut j = len;
        while j > pos {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[pos] = chosen;
        u8::try_from(len + 1).unwrap_or(u8::MAX)
    } else {
        // Defense-in-depth (unreachable on legal logs): chosen + smallest.
        buf[31] = chosen;
        isort_u32(buf, 32);
        32
    }
}
