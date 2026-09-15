use super::offers::{
    canonical_take, concealed_counts, river_type_set, seat_takes, seat_type_counts, visible_counts,
};
use super::shape::{chi_offered_types, shape_win_with, win_shape_counts};
use crate::ingest::{
    KIND_CHI, KIND_DAIMINKAN, KIND_DORA, KIND_HORA, KIND_PON, KIND_REACH_ACCEPTED, StackedEvent,
};
use crate::ledger::{
    Ledger, N_WALK_PLANES, PHASE_DRAW, PLANE_ACTOR, PLANE_CAN_RIICHI, PLANE_CAN_TSUMO,
    PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DEALER, PLANE_DORA, PLANE_FURITEN, PLANE_HAND_NUMBER,
    PLANE_HIST_KIND, PLANE_HIST_MASK, PLANE_HONBA, PLANE_IPPATSU, PLANE_KAN_COUNT, PLANE_LEGAL,
    PLANE_LIVE_WALL, PLANE_OWN_DRAWN, PLANE_PHASE, PLANE_RIICHI_STATES, PLANE_ROUND_INDEX,
    PLANE_ROUND_WIND, PLANE_SCORES, PLANE_SEAT_WINDS, PLANE_STICKS, PLANE_TURN_ACTOR,
    PLANE_VISIBLE, RowSink,
};

// ---------------------------------------------------------------------------
// Walk state (row-assembly scratch; table state stays in `Ledger`).
// ---------------------------------------------------------------------------

/// Per-discard window cache: peek-first log-ron fact short-circuits the
/// responder eval; otherwise one eval per discard is cached and shared by
/// emission (`open_window`) and the walled mode gate (`walker_mode`).
pub(crate) struct WalkState {
    pub(crate) histories: [Vec<u8>; 4],
    pub(crate) last_kind: Option<u8>,
    pub(crate) last_discard: Option<(u8, u8)>,
    pub(crate) last_discard_idx: usize,
    /// Last non-kan claim (pon/chi seat + called take) with no intervening
    /// discard or kan: a same-seat tsumo while armed would deal a 15th
    /// ledger take (12 concealed + 3 meld) the wall-less oracle desyncs on.
    /// Conservation tripwire (fail-closed): reject as tile-conservation.
    /// Draw offers also withhold the just-claimed string while armed.
    pub(crate) last_claim: Option<(u8, u8)>,
    pub(crate) opened_by_discard: bool,
    pub(crate) decided: bool,
    pub(crate) terminal: bool,
    /// A hora applied with no closing end_kyoku since: the wall oracle goes
    /// terminal and rejects trailing row/end events (ron_win shape). Cleared
    /// by end_kyoku; checked at walk end.
    pub(crate) hora_terminal: bool,
    pub(crate) started: bool,
    pub(crate) seq: u32,
    pub(crate) cache_key: Option<(usize, bool)>,
    pub(crate) cache_open: bool,
    pub(crate) cache_pending: ([u8; 3], u8),
}

impl WalkState {
    pub(crate) fn new() -> Self {
        Self {
            histories: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            last_kind: None,
            last_discard: None,
            last_discard_idx: 0,
            last_claim: None,
            opened_by_discard: false,
            decided: false,
            terminal: false,
            hora_terminal: false,
            started: false,
            seq: 0,
            cache_key: None,
            cache_open: false,
            cache_pending: ([0u8; 3], 0),
        }
    }

    pub(crate) fn push_public(&mut self, hid: u8) {
        let mut s = 0usize;
        while s < 4 {
            self.histories[s].push(hid);
            s += 1;
        }
        self.last_kind = Some(hid);
    }

    pub(crate) fn reset_discard(&mut self) {
        self.last_discard = None;
        self.last_claim = None;
        self.opened_by_discard = false;
        self.decided = false;
        self.cache_key = None;
    }
}

// ---------------------------------------------------------------------------
// Span scanners (zero-alloc byte loops; serde only at kyoku boundaries).
// ---------------------------------------------------------------------------

/// Skip ASCII whitespace.
#[inline]
pub(crate) fn skip_ws(s: &[u8], mut j: usize) -> usize {
    while j < s.len() && (s[j] == b' ' || s[j] == b'\t' || s[j] == b'\n' || s[j] == b'\r') {
        j += 1;
    }
    j
}

/// Find a top-level `"key"` occurrence; returns the value start.
pub(crate) fn top_value(span: &[u8], key: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + key.len() + 2 <= span.len() {
        if span[i] == b'"'
            && &span[i + 1..i + 1 + key.len()] == key
            && span[i + 1 + key.len()] == b'"'
        {
            let j = skip_ws(span, i + 1 + key.len() + 1);
            if j < span.len() && span[j] == b':' {
                return Some(skip_ws(span, j + 1));
            }
        }
        i += 1;
    }
    None
}

/// Hora `"tsumo":true` flag (exact `"tsumo"` key match — `"tsumogiri"`
/// cannot match: the closing quote is part of the pattern).
pub(crate) fn hora_tsumo_flag(span: &[u8]) -> bool {
    match top_value(span, b"tsumo") {
        Some(j) => span[j..].starts_with(b"true"),
        None => false,
    }
}

/// Parse `-?(0|[1-9][0-9]{0,18})` at `s[j..]`; returns `(value, end)`.
pub(crate) fn parse_int(s: &[u8], mut j: usize) -> Option<(i64, usize)> {
    let neg = j < s.len() && s[j] == b'-';
    if neg {
        j += 1;
    }
    let d0 = j;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
    }
    let run = j - d0;
    if run == 0 || run > 19 {
        return None;
    }
    if run > 1 && s[d0] == b'0' {
        return None;
    }
    let mut v: i64 = 0;
    let mut k = d0;
    while k < j {
        v = v.checked_mul(10)?.checked_add((s[k] - b'0') as i64)?;
        k += 1;
    }
    Some((if neg { -v } else { v }, j))
}

/// `"deltas":[i,i,i,i]` values from a hora/ryukyoku span (`None` malformed).
pub(crate) fn deltas_quad(span: &[u8]) -> Option<[i32; 4]> {
    let mut j = top_value(span, b"deltas")?;
    if j >= span.len() || span[j] != b'[' {
        return None;
    }
    j = skip_ws(span, j + 1);
    let mut out = [0i32; 4];
    let mut k = 0usize;
    while k < 4 {
        let (v, nj) = parse_int(span, j)?;
        out[k] = v as i32;
        j = skip_ws(span, nj);
        if k < 3 {
            if j >= span.len() || span[j] != b',' {
                return None;
            }
            j = skip_ws(span, j + 1);
        }
        k += 1;
    }
    if j >= span.len() || span[j] != b']' {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Window evaluation (peek-first + per-discard cache).
// ---------------------------------------------------------------------------

/// Responder eval for one discard: `(window_open, pending seats, ron seats)`.
/// `tile` is the discard take id (type read only — aka folds).
pub(crate) fn eval_window(
    l: &Ledger,
    discarder: u8,
    tile: u8,
    chankan: bool,
) -> (bool, ([u8; 3], u8), [bool; 4]) {
    let ttype = tile / 4;
    let mut pending = [0u8; 3];
    let mut plen = 0u8;
    let mut ron = [false; 4];
    let mut open = false;
    let mut seat = 0u8;
    while seat < 4 {
        if seat == discarder {
            seat += 1;
            continue;
        }
        let s = seat as usize;
        let mut counts = [0u8; 34];
        seat_type_counts(l, seat, &mut counts);
        let meld_open = l.meld_lens[s] as usize;
        let riichi = l.riichi[s] != 0;
        // Ron-shape gate over hand + discard (aka-insensitive types).
        // Count-space trial: `counts` (from `seat_type_counts` just above)
        // equals `type_counts_of(hand)` — one id per present take, each of
        // the row's type — so `counts` plus one `ttype` is multiset-equal
        // at the type level to `hand` plus the discard copy (concealed
        // takes are ≤14, so the old 23/24 caps never bound); the win check
        // is type-level (aka-insensitive) — identical checks, no recount.
        let mut shape = false;
        if ttype < 34 {
            let mut trial = counts;
            trial[ttype as usize] += 1;
            shape = win_shape_counts(&trial, meld_open);
        }
        // Takes + river build only on shape (the proxy below needs them;
        // shape-false rows offer nothing either way).
        // Ron offers go to every shape-winning responder (riichi or not —
        // the engine offers non-riichi winners too); the river proxy
        // withholds offers the engine withholds (furiten).
        let ron_offered = if shape && ttype < 34 {
            let mut takes = [0u8; 24];
            let tn = seat_takes(l, seat, &mut takes);
            let hand = &takes[..tn];
            let mut river = [false; 34];
            river_type_set(l, seat, &mut river);
            if !river[ttype as usize] {
                // River furiten proxy: winning tile unseen and no river
                // kind completes the shape.
                let mut furiten = false;
                let mut rt = 0usize;
                while rt < 34 {
                    if river[rt] && shape_win_with(hand, rt as u8, meld_open) {
                        furiten = true;
                        break;
                    }
                    rt += 1;
                }
                if !furiten {
                    open = true;
                }
            }
            !river[ttype as usize]
        } else {
            false
        };
        if !riichi && !chankan && ttype < 34 {
            if counts[ttype as usize] >= 2 {
                open = true;
            }
            if seat == (discarder + 1) % 4 && chi_offered_types(&counts, ttype) {
                open = true;
            }
        }
        // Pending set (engine-derived; the log-ron fact ORs at the gate).
        if ttype < 34 && !chankan {
            let pon = counts[ttype as usize] >= 2;
            let chi = seat == (discarder + 1) % 4 && chi_offered_types(&counts, ttype);
            if (pon || chi || ron_offered) && plen < 3 {
                pending[plen as usize] = seat;
                plen += 1;
            }
        } else if ron_offered && plen < 3 {
            pending[plen as usize] = seat;
            plen += 1;
        }
        ron[s] = ron_offered;
        seat += 1;
    }
    (open, (pending, plen), ron)
}

/// Peek the next claim on `discarder`'s tile past transparent events
/// (mirrors `_peek_window_claim`): claims/ron naming this discarder, else
/// the window passes with no claim.
pub(crate) fn peek_window_claim(
    ev: &[StackedEvent<'_>],
    from_idx: usize,
    discarder: u8,
) -> Option<usize> {
    let mut idx = from_idx;
    while idx < ev.len() {
        let e = &ev[idx];
        match e.kind {
            KIND_DORA | KIND_REACH_ACCEPTED => {
                idx += 1;
                continue;
            }
            KIND_CHI | KIND_PON | KIND_DAIMINKAN => {
                if e.target == discarder {
                    return Some(idx);
                }
                return None;
            }
            KIND_HORA => {
                if hora_tsumo_flag(e.span) || e.actor == e.target {
                    return None;
                }
                if e.target == discarder {
                    return Some(idx);
                }
                return None;
            }
            _ => return None,
        }
    }
    None
}

/// Whether the log takes a ron on this discard (sound window-open fact):
/// peek-first short-circuit for the responder eval.
pub(crate) fn peek_window_ron(ev: &[StackedEvent<'_>], from_idx: usize, discarder: u8) -> bool {
    match peek_window_claim(ev, from_idx, discarder) {
        Some(i) => ev[i].kind == KIND_HORA,
        None => false,
    }
}

/// Cached window eval keyed by (discard event idx, chankan).
pub(crate) fn cached_window(
    st: &mut WalkState,
    l: &Ledger,
    discarder: u8,
    tile: u8,
    disc_idx: usize,
    chankan: bool,
) -> (bool, ([u8; 3], u8)) {
    if st.cache_key == Some((disc_idx, chankan)) {
        return (st.cache_open, st.cache_pending);
    }
    let (open, pending, _) = eval_window(l, discarder, tile, chankan);
    st.cache_key = Some((disc_idx, chankan));
    st.cache_open = open;
    st.cache_pending = pending;
    (open, pending)
}

// ---------------------------------------------------------------------------
// Row minting (DIRECT to minimal hot planes, canonical plane order).
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub(crate) fn put_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Mint one row into the sink: fixed planes + T-variable history + lens.
pub(crate) fn mint_row(
    l: &mut Ledger,
    st: &mut WalkState,
    out: &mut RowSink<'_>,
    seat: u8,
    phase: u8,
    turn_actor: u8,
    chosen: u32,
    legal: &[u32],
) {
    debug_assert_eq!(out.planes.len(), N_WALK_PLANES);
    let s = seat as usize;
    // #0 concealed counts (live draw excluded).
    {
        let mut counts = [0u8; 34];
        concealed_counts(l, seat, &mut counts);
        out.planes[PLANE_CONCEALED].extend_from_slice(&counts);
    }
    // #1 visible counts (rivers + melds, clamp 4).
    {
        let mut counts = [0u8; 34];
        visible_counts(l, &mut counts);
        out.planes[PLANE_VISIBLE].extend_from_slice(&counts);
    }
    // #2 dora indicators (i32, −1 tail).
    {
        let mut i = 0usize;
        while i < 5 {
            put_i32(&mut out.planes[PLANE_DORA], l.dora[i] as i32);
            i += 1;
        }
    }
    // #3 scores. Draw rows read live (and refresh the window snapshot);
    // responder rows (claims/ron) reuse the discarder's snapshot: the
    // oracle captures responder observations at window time, so riichi
    // payments and accept sticks landing between window and claim stay
    // out of the claim row (visible from the next draw row on).
    let is_draw = phase == PHASE_DRAW;
    {
        let mut i = 0usize;
        while i < 4 {
            let v = if is_draw {
                l.scores[i]
            } else {
                l.snap_scores[i]
            };
            put_i32(&mut out.planes[PLANE_SCORES], v);
            i += 1;
        }
        if is_draw {
            l.snap_scores = l.scores;
            l.snap_sticks = l.sticks;
        }
    }
    // #4/#5 history kinds + mask (T-variable; lens recorded).
    {
        let hist = &st.histories[s];
        for h in hist.iter() {
            put_i64(&mut out.planes[PLANE_HIST_KIND], *h as i64);
            out.planes[PLANE_HIST_MASK].push(1u8);
        }
        out.hist_lens.push(hist.len() as u32);
    }
    // #6..#11 scalars.
    put_i64(&mut out.planes[PLANE_CHOSEN], chosen as i64);
    put_i64(&mut out.planes[PLANE_ACTOR], seat as i64);
    put_i64(&mut out.planes[PLANE_DEALER], l.dealer as i64);
    put_i64(
        &mut out.planes[PLANE_ROUND_WIND],
        (l.round_wind - 27) as i64,
    );
    put_i64(&mut out.planes[PLANE_PHASE], phase as i64);
    {
        let mut i = 0usize;
        while i < 4 {
            put_i64(
                &mut out.planes[PLANE_SEAT_WINDS],
                (l.seat_wind[i] - 27) as i64,
            );
            i += 1;
        }
    }
    // #12 legal_ids[32] + legal_len.
    {
        let mut k = 0usize;
        while k < 32 {
            let id = if k < legal.len() { legal[k] } else { 0 };
            put_i32(&mut out.planes[PLANE_LEGAL], id as i32);
            k += 1;
        }
        put_i64(&mut out.planes[PLANE_LEGAL], legal.len().min(32) as i64);
    }
    l.phase = phase;
    // #13..#25 row scalars (model inputs; read pre-mutation state — callers
    // mint before applying clears/settlements, so observations stay
    // pre-decision like the oracle's capture-then-emit).
    put_i64(&mut out.planes[PLANE_TURN_ACTOR], turn_actor as i64);
    let furiten = if l.riichi[s] != 0 {
        2
    } else if l.missed[s] != 0 {
        1
    } else {
        0
    };
    put_i64(&mut out.planes[PLANE_FURITEN], furiten);
    put_i32(&mut out.planes[PLANE_HONBA], l.honba as i32);
    let stick_v = if is_draw { l.sticks } else { l.snap_sticks };
    put_i32(&mut out.planes[PLANE_STICKS], stick_v as i32);
    put_i32(&mut out.planes[PLANE_LIVE_WALL], l.live_wall() as i32);
    put_i32(&mut out.planes[PLANE_KAN_COUNT], l.kan_count as i32);
    put_i32(&mut out.planes[PLANE_ROUND_INDEX], l.round_index as i32);
    put_i32(&mut out.planes[PLANE_HAND_NUMBER], l.hand_number as i32);
    let drawn_rep = match l.drawn_col[s] {
        Some(dc) => {
            if l.fold_reports {
                canonical_take(dc) as i32
            } else {
                dc as i32
            }
        }
        None => -1,
    };
    put_i32(&mut out.planes[PLANE_OWN_DRAWN], drawn_rep);
    {
        let mut i = 0usize;
        while i < 4 {
            out.planes[PLANE_IPPATSU].push(l.ippatsu[i]);
            i += 1;
        }
    }
    {
        let mut i = 0usize;
        while i < 4 {
            // accepted → 2; declared (1) never surfaces on log rows.
            let v = if l.accepted[i] != 0 { 2i64 } else { 0i64 };
            put_i64(&mut out.planes[PLANE_RIICHI_STATES], v);
            i += 1;
        }
    }
    out.planes[PLANE_CAN_RIICHI].push(if l.riichi_offered { 1u8 } else { 0u8 });
    out.planes[PLANE_CAN_TSUMO].push(if l.tsumo_offered { 1u8 } else { 0u8 });
    out.rows += 1;
    st.seq += 1;
}
