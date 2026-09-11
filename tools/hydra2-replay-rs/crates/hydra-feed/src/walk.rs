//! S3 walk — u8 ledger DIRECT to minimal hot planes (`feed::walk`).
//!
//! OWNER: LedgerWalker (P2-A). Plan §6.3 + FusedHotPathDesigner S3.
//!
//! Engine move: the `engine.rs → feed::walk` rules live here as pure `u8`
//! functions (win shape, tenpai, chi/pon/window responder predicates, draw
//! offers). `hydra-shard/src/engine.rs` keeps serving the cold/oracle path;
//! this module ports its V1 (drained-oracle) semantics rule-by-rule onto the
//! copy-slot ledger — no `String`, no `HashMap`, no heap on the row path.
//! V2 (live single-pass deltas) is out of scope, with one evidence-backed
//! exception: the nine-terminals abort offer on qualifying first draws
//! matches live riichienv (differential-proven; user-approved V2 item).
//! Chi-completeness, kuikae-strictness, and take conservation stay V1.
//!
//! Oracle pin (Main ruling 2026-09-09): the wall-less replay oracle reports
//! COLLAPSED takes on pre-reach discards (F1 golden
//! `[192,193,193,193,196]`), while the walled expander reports true takes
//! (`[192,…,196]`). This walk has no wall takes by construction, so F1
//! bitwise matches the wall-less collapse (== shard walk, == F7 frozen
//! rows). Walled true-takes assert ONLY under F3 via the borrowed context
//! digest — never as an F1 target.
//!
//! Allocation audit (row hot path — tsumo/dahai/claim/kan/hora dispatch,
//! offers, legal, window eval + peek, plane appends):
//! - NO `String` / `format!` / `HashMap` / `HashSet` / `Vec` / `sort` with
//!   allocation. Sorting is insertion sort over stack arrays.
//! - `serde_json::Value` parses run ONLY at kyoku boundaries (`start_kyoku`
//!   headers, `ryukyoku` reason: ≤2 parses per kyoku, amortized, never per
//!   row). Hora `deltas`/`tsumo` read spans byte-wise. Everything else is
//!   borrowed ingest fields.
//! - `Vec` pushes are output buffering only (per-seat history kinds,
//!   Scratch plane bytes): amortized, never per-row realloc in steady state.
//! - Non-sampled games allocate NOTHING: the gate skips them before the
//!   walk is ever entered.
//!
//! Kill list (vs cold reference `hydra-shard/src/decisions.rs`):
//! - window re-render (:982-1026, 3-4 wasted responder evals per discard)
//!   → [`WalkState`] peek-first log-ron fact + per-discard eval cache.
//! - `HashSet` mask builds + `sort_unstable` + `mask.to_vec` → stack
//!   `[u32; 32]` + insertion sort, chosen forcing by id arithmetic.
//! - `chosen.kind_str()` + table `HashMap` lookup → [`chosen_id`] /
//!   [`chosen_claim_id`] closed-form arithmetic pinned to action-table
//!   digest [`crate::ledger::ACTION_TABLE_DIGEST`] (verified entry-wise in
//!   the module tests).

use crate::ingest::{
    StackedEvent, KIND_ANKAN, KIND_CHI, KIND_DAHAI, KIND_DAIMINKAN, KIND_DORA, KIND_END,
    KIND_END_KYOKU, KIND_HORA, KIND_KAKAN, KIND_OTHER, KIND_PON, KIND_REACH, KIND_REACH_ACCEPTED,
    KIND_RYUKYOKU, KIND_START, KIND_START_KYOKU, KIND_TRANSPARENT_OTHER, KIND_TSUMO, NO_PAI,
};
use crate::ledger::{
    GameCtx, Ledger, Meld, RowSink, TileClass, WalkReject, N_WALK_PLANES, PHASE_DRAW,
    PHASE_KAN_RESPONSE, PHASE_RESPONSE, WALK_ACTION_UNRESOLVED, WALK_BARE_DORA,
    WALK_CLAIM_NO_OFFER, WALK_DRAW_PAST_WALL, WALK_ENGINE_DESYNC, WALK_KYUSHU_AMBIGUOUS,
    WALK_PAST_TERMINAL, WALK_TILE_CONSERVATION, WALK_TURN_ORDER, WALK_UNKNOWN_EVENT,
    WALK_UNMAPPED_RYUKYOKU, ACT_ANKAN,
    ACT_CHI, ACT_DAIMINKAN, ACT_DISCARD, ACT_KAKAN, ACT_PASS, ACT_PON, ACT_RIICHI_DISCARD, ACT_RON,
    ACT_TSUMOGIRI, ACT_TSUMO, H_ABORTIVE, H_ANKAN, H_CALL_WINDOW, H_CHI, H_DAIMINKAN, H_DORA,
    H_DRAW_END, H_DRAW_TILE, H_DISCARD, H_GAME_START, H_KAKAN, H_PON, H_RIICHI_ACCEPTED, H_RON,
    H_ROUND_END, H_ROUND_START, H_TSUMO, H_TURN_ADVANCE, PLANE_ACTOR, PLANE_CAN_RIICHI,
    PLANE_CAN_TSUMO, PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DEALER, PLANE_DORA, PLANE_FURITEN,
    PLANE_HAND_NUMBER, PLANE_HIST_KIND, PLANE_HIST_MASK, PLANE_HONBA, PLANE_IPPATSU,
    PLANE_KAN_COUNT, PLANE_LEGAL, PLANE_LIVE_WALL, PLANE_OWN_DRAWN, PLANE_PHASE,
    PLANE_RIICHI_STATES, PLANE_ROUND_INDEX, PLANE_ROUND_WIND, PLANE_SCORES, PLANE_SEAT_WINDS,
    PLANE_STICKS, PLANE_TURN_ACTOR, PLANE_VISIBLE, aka_suit, class_of_canonical, held_takes,
    is_aka_id, is_five_type,
};
use crate::tiles::TileCodec;

// ---------------------------------------------------------------------------
// Tunables (mirror the cold oracles; tests pin behavior, not values).
// ---------------------------------------------------------------------------

    #[test]
    fn quad_heavy_shanten1_yields_no_riichi_candidates() {
        // Three concealed quads + two singles (4m/8m/3p quads, 4p, drawn 5p)
        // is shanten 1 (engine-verified): no discard completes tenpai, so
        // tenpai_discards must report zero candidates (no riichi offers).
        let hand14: [u8; 14] = [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 48, 53];
        let mut out = [0u8; 16];
        assert_eq!(tenpai_discards(&hand14, 0, &mut out), 0);
    }


/// Live-wall countdown base: 136 − 4×13 dealt − 14 dead wall.
pub const LIVE_WALL_BASE: u32 = 70;
/// Riichi needs a full round left (wall ≥ 4 at declaration).
pub const RIICHI_MIN_WALL: u32 = 4;
/// Riichi costs 1000 points.
pub const RIICHI_MIN_SCORE: i32 = 1000;
/// Kan needs a replacement tile (wall ≥ 1).
pub const KAN_MIN_WALL: u32 = 1;
/// Draws at/after this count fail `draw-past-wall`.
pub const MAX_DRAWS: u8 = 70;
/// Kyushu/exhaustive inference bounds (draw counts, mirror the oracle).
pub const KYUSHU_MAX_DRAWS: u8 = 4;
/// Exhaustive-draw inference bound.
pub const EXHAUSTIVE_MIN_DRAWS: u8 = 60;
/// Distinct yaochu kinds for the nine-terminals abort inference.
pub const KYUSHU_DISTINCT_YAOCHU: usize = 9;
/// Yaochu types (terminals + honors), norm-folded.
const YAOCHU_TYPES: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
/// Nine-terminals abort offer id (`abort_nine_terminals`, no tile).
/// Table-tail singleton, pinned like the BASE_* ids to
/// [`crate::ledger::ACTION_TABLE_DIGEST`]; verified against the live engine
/// (offers 6790 on qualifying first draws) and the Python action table
/// (same digest, same id). Additions here stay singletons: abort offers
/// carry no tile, so no take arithmetic applies.
pub const ABORT_NINE_TERMINALS_ID: u32 = 6790;

/// Claim-response pass id (`pass`, offset `None`).
pub const PASS_ID: u32 = 0;
/// Unresolved-action sentinel (never a real id; walk rejects on it).
pub const UNRESOLVED_ID: u32 = u32::MAX;

// Table bases (pinned to [`crate::ledger::ACTION_TABLE_DIGEST`]).
const BASE_DISCARD: u32 = 4;
const BASE_TSUMOGIRI: u32 = 140;
const BASE_RIICHI: u32 = 276;
const BASE_CHI: u32 = 412;
const BASE_PON: u32 = 4444;
const BASE_DAIMINKAN: u32 = 5668;
const BASE_ANKAN: u32 = 6076;
const BASE_KAKAN: u32 = 6110;
const BASE_RON: u32 = 6246;
const BASE_TSUMO: u32 = 6654;

/// Per-type chi entry base: `64 × Σ patterns(earlier types)`, where
/// patterns by rank are 1/2/3/3/3/3/3/2/1 (one pattern × 16 copy combos).
const fn chi_type_base() -> [u16; 27] {
    let pats: [u16; 9] = [1, 2, 3, 3, 3, 3, 3, 2, 1];
    let mut out = [0u16; 27];
    let mut acc: u16 = 0;
    let mut t = 0usize;
    while t < 27 {
        out[t] = acc;
        acc += 64 * pats[t % 9];
        t += 1;
    }
    out
}
const CHI_TYPE_BASE: [u16; 27] = chi_type_base();

/// Chi pattern count for a rank 1..=9.
#[inline]
const fn chi_patterns(rank: u8) -> u8 {
    match rank {
        1 | 9 => 1,
        2 | 8 => 2,
        _ => 3,
    }
}

// ---------------------------------------------------------------------------
// Action ids without strings.
// ---------------------------------------------------------------------------

/// Relative seat offset, kamicha folded to −1 (mirrors `source_offset`).
#[inline]
pub const fn source_offset(source: u8, actor: u8) -> i8 {
    match (source + 4 - actor) % 4 {
        3 => -1,
        d => d as i8,
    }
}

/// Offset-table index for ron/pon/daiminkan (−1→0, 1→1, 2→2).
#[inline]
pub const fn offset_idx(offset: i8) -> Option<usize> {
    match offset {
        -1 => Some(0),
        1 => Some(1),
        2 => Some(2),
        _ => None,
    }
}

/// Chosen id for tile-only kinds + pass + ankan-by-type + ron-by-offset.
///
/// `called` semantics per kind: ignored for tile-only kinds and pass;
/// ankan takes the tile TYPE (0..34); ron takes the offset code
/// (0→−1 kamicha, 1→1, 2→2). Chi/pon/daiminkan need consumed tiles — use
/// [`chosen_claim_id`]. Returns [`UNRESOLVED_ID`] when out of range
/// (cold-detectable; the walk rejects instead of emitting it).
pub const fn chosen_id(kind: u8, tile: u8, called: u8) -> u32 {
    match kind {
        ACT_PASS => PASS_ID,
        ACT_DISCARD => {
            if tile < 136 {
                BASE_DISCARD + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_TSUMOGIRI => {
            if tile < 136 {
                BASE_TSUMOGIRI + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_RIICHI_DISCARD => {
            if tile < 136 {
                BASE_RIICHI + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_KAKAN => {
            if tile < 136 {
                BASE_KAKAN + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_TSUMO => {
            if tile < 136 {
                BASE_TSUMO + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_ANKAN => {
            if tile < 34 {
                BASE_ANKAN + tile as u32
            } else {
                UNRESOLVED_ID
            }
        }
        ACT_RON => {
            if tile < 136 {
                let k = match called {
                    0 => 0u32,
                    1 => 1u32,
                    2 => 2u32,
                    _ => return UNRESOLVED_ID,
                };
                BASE_RON + tile as u32 * 3 + k
            } else {
                UNRESOLVED_ID
            }
        }
        _ => UNRESOLVED_ID,
    }
}

/// Chosen id for claim/kan kinds with consumed tiles.
///
/// - chi: `consumed` = 2 sorted takes forming a sequence window with
///   `called` (suited only); offset is always −1.
/// - pon: `consumed` = 2 sorted takes, same type as `called`, both
///   distinct from it (a 2-subset of the block minus the called copy).
/// - daiminkan: `consumed` = the other three block takes.
/// - ankan: `consumed` = the full sorted block.
/// Returns `None` on any mismatch (walk rejects `action-id-unresolved`).
pub fn chosen_claim_id(kind: u8, called: u8, consumed: &[u8], offset: i8) -> Option<u32> {
    match kind {
        ACT_CHI => {
            if offset != -1 || consumed.len() != 2 || called >= 108 {
                return None;
            }
            let (c0, c1) = (consumed[0], consumed[1]);
            if c0 >= c1 || c1 >= 108 {
                return None;
            }
            let tc = called / 4;
            let suit = tc / 9;
            let rank = tc % 9 + 1;
            let (t0, t1) = (c0 / 4, c1 / 4);
            if t0 / 9 != suit || t1 / 9 != suit {
                return None;
            }
            let (r0, r1) = (t0 % 9 + 1, t1 % 9 + 1);
            // Ascending sequence windows containing `rank`, by position.
            let mut pat: Option<u8> = None;
            let mut seen = 0u8;
            let lo_hi = [
                (rank.saturating_sub(2), rank.saturating_sub(1)),
                (rank - 1, rank + 1),
                (rank + 1, rank + 2),
            ];
            let mut k = 0usize;
            while k < 3 {
                let (lo, hi) = lo_hi[k];
                if lo >= 1 && hi <= 9 {
                    if lo == r0 && hi == r1 {
                        pat = Some(seen);
                    }
                    seen += 1;
                }
                k += 1;
            }
            let p = pat?;
            let (ci0, ci1) = (c0 - t0 * 4, c1 - t1 * 4);
            if ci0 >= 4 || ci1 >= 4 {
                return None;
            }
            let base = BASE_CHI
                + CHI_TYPE_BASE[tc as usize] as u32
                + (called % 4) as u32 * 16 * chi_patterns(rank) as u32;
            Some(base + p as u32 * 16 + ci0 as u32 * 4 + ci1 as u32)
        }
        ACT_PON => {
            if consumed.len() != 2 || called >= 136 {
                return None;
            }
            let tc = called / 4;
            let (c0, c1) = (consumed[0], consumed[1]);
            if c0 >= c1 || c0 / 4 != tc || c1 / 4 != tc || c0 == called || c1 == called {
                return None;
            }
            let k = called % 4;
            // 2-subsets of the block minus the called copy, lex order.
            let pairs: [(u8, u8); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
            let (q0, q1) = (c0 % 4, c1 % 4);
            let mut pair_idx: Option<u32> = None;
            let mut n = 0u32;
            let mut i = 0usize;
            while i < 6 {
                let (a, b) = pairs[i];
                if a != k && b != k {
                    if a == q0 && b == q1 {
                        pair_idx = Some(n);
                    }
                    n += 1;
                }
                i += 1;
            }
            let oi = offset_idx(offset)?;
            Some(BASE_PON + called as u32 * 9 + pair_idx? * 3 + oi as u32)
        }
        ACT_DAIMINKAN => {
            if consumed.len() != 3 || called >= 136 {
                return None;
            }
            let base = (called / 4) * 4;
            let mut m = 0usize;
            let mut j = 0u8;
            while j < 4 {
                let id = base + j;
                if id != called {
                    if m >= 3 || consumed[m] != id {
                        return None;
                    }
                    m += 1;
                }
                j += 1;
            }
            if m != 3 {
                return None;
            }
            let oi = offset_idx(offset)?;
            Some(BASE_DAIMINKAN + called as u32 * 3 + oi as u32)
        }
        ACT_ANKAN => {
            if consumed.len() != 4 {
                return None;
            }
            let base = (consumed[0] / 4) * 4;
            if consumed != [base, base + 1, base + 2, base + 3] {
                return None;
            }
            Some(BASE_ANKAN + (base / 4) as u32)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Engine rules on u8 takes (V1 port; yaku-blind shape core verbatim).
// ---------------------------------------------------------------------------

/// Type counts over take ids (aka shares its type).
fn type_counts_of(ids: &[u8], counts: &mut [u8; 34]) {
    for c in counts.iter_mut() {
        *c = 0;
    }
    for id in ids {
        let t = (*id / 4) as usize;
        if t < 34 {
            counts[t] += 1;
        }
    }
}

/// Recursive meld decomposition (port of `melds_out`; stack only).
/// Bounded variant: positions below `start` are zero by construction, so the
/// first-nonzero scan starts at `start` instead of 0.
fn melds_out_from(counts: &mut [u8; 34], need: u8, start: usize) -> bool {
    if need == 0 {
        // All positions below `start` are already zero; check the rest.
        let mut i = start;
        while i < 34 {
            if counts[i] != 0 {
                return false;
            }
            i += 1;
        }
        return true;
    }
    let mut i = start;
    while i < 34 {
        if counts[i] > 0 {
            break;
        }
        i += 1;
    }
    if i == 34 {
        return false;
    }
    if counts[i] >= 3 {
        counts[i] -= 3;
        // Positions below `i` stay zero, so resume the scan at `i`.
        if melds_out_from(counts, need - 1, i) {
            return true;
        }
        counts[i] += 3;
    }
    let suit = i as u8 / 9;
    let pos = i as u8 % 9;
    if suit < 3 && pos <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0 {
        counts[i] -= 1;
        counts[i + 1] -= 1;
        counts[i + 2] -= 1;
        // `i` is now zero and everything below is still zero; resume at `i`.
        if melds_out_from(counts, need - 1, i) {
            return true;
        }
        counts[i] += 1;
        counts[i + 1] += 1;
        counts[i + 2] += 1;
    }
    false
}

fn melds_out(counts: &mut [u8; 34], need: u8) -> bool {
    melds_out_from(counts, need, 0)
}

/// Yaku-blind win over type counts + open meld count.
///
/// Seven-pairs (no quad-as-two-pairs) / thirteen-orphans when fully
/// concealed, else (4 − open) melds + a pair. Aka-insensitive (types).
fn win_shape_counts(counts: &[u8; 34], open_melds: usize) -> bool {
    if open_melds > 4 {
        return false;
    }
    if open_melds == 0 {
        // Count-space length gate: id-level `concealed.len() == 14` is
        // `sum == 14` via 34 adds (trial ids == counts at type level).
        let mut sum = 0u8;
        let mut s = 0usize;
        while s < 34 {
            sum += counts[s];
            s += 1;
        }
        if sum == 14 {
            {
                let mut pairs = 0u8;
                let mut ok = true;
                let mut i = 0usize;
                while i < 34 {
                    if counts[i] != 0 && counts[i] != 2 {
                        ok = false;
                        break;
                    }
                    if counts[i] == 2 {
                        pairs += 1;
                    }
                    i += 1;
                }
                if ok && pairs == 7 {
                    return true;
                }
            }
            {
                let terms: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
                let mut ok = true;
                let mut has_pair = false;
                let mut i = 0usize;
                while i < 34 {
                    if counts[i] == 2 {
                        has_pair = true;
                    }
                    i += 1;
                }
                let mut k = 0usize;
                while k < 13 {
                    if counts[terms[k]] < 1 {
                        ok = false;
                        break;
                    }
                    k += 1;
                }
                if ok && has_pair {
                    return true;
                }
            }
        }
    }
    let need = 4 - open_melds as u8;
    let mut work = *counts;
    let mut pair = 0usize;
    while pair < 34 {
        if work[pair] >= 2 {
            // In-place pair removal; `work` is function-local, restore before continuing.
            work[pair] -= 2;
            let hit = melds_out(&mut work, need);
            work[pair] += 2;
            if hit {
                return true;
            }
        }
        pair += 1;
    }
    false
}

/// Yaku-blind standard-shape win over take ids + open meld count.
///
/// Seven-pairs (no quad-as-two-pairs) / thirteen-orphans when fully
/// concealed, else (4 − open) melds + a pair. Aka-insensitive (types).
pub fn win_shape_14(concealed: &[u8], open_melds: usize) -> bool {
    let mut counts = [0u8; 34];
    type_counts_of(concealed, &mut counts);
    win_shape_counts(&counts, open_melds)
}

/// Insertion sort over a `u32` prefix (deterministic, no alloc).
fn isort_u32(buf: &mut [u32; 32], len: usize) {
    let mut i = 1usize;
    while i < len {
        let x = buf[i];
        let mut j = i;
        while j > 0 && buf[j - 1] > x {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = x;
        i += 1;
    }
}

/// Insertion sort over a `u8` slice prefix (deterministic, no alloc).
fn isort_u8(buf: &mut [u8], len: usize) {
    let mut i = 1usize;
    while i < len {
        let x = buf[i];
        let mut j = i;
        while j > 0 && buf[j - 1] > x {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = x;
        i += 1;
    }
}

/// Tenpai discards of a 14-take hand: one entry per take whose removal
/// leaves tenpai (some completion wins by shape). Port of
/// `tenpai_discards` onto stack arrays: distinct types ascending, then
/// every take of a tenpai type, sorted.
fn tenpai_discards(hand14: &[u8], open_melds: usize, out: &mut [u8; 16]) -> usize {
    // Count-space trials: `counts` is built once from `hand14` via `id / 4`
    // (aka shares its type, so the id/4 mapping agrees). Each trial
    // `counts - ty + kind` is multiset-equal at the type level to the old
    // id-level trial (`rest` + `kind * 4`), and the win checks are
    // type-level per docs (aka-insensitive), so predicates are identical.
    let mut counts = [0u8; 34];
    type_counts_of(hand14, &mut counts);
    // C0 hoist: full-hand win; kind==ty restores this multiset (rest+ty == hand14).
    let c0_win = win_shape_counts(&counts, open_melds);
    let mut n = 0usize;
    let mut ty = 0usize;
    while ty < 34 {
        if counts[ty] > 0 {
            counts[ty] -= 1;
            let mut tenpai = false;
            let mut kind = 0usize;
            while kind < 34 {
                // kind==ty trial multiset == C0 input multiset: reuse c0_win, skip +1/check/-1.
                if kind == ty {
                    tenpai = c0_win;
                } else if counts[kind] < 4 {
                    // No fifth copy exists: all four copies of `kind` are
                    // already held, so nothing can complete this wait (dead).
                    // Without this gate, quad-heavy hands report phantom
                    // tenpai (e.g. waiting on a fifth copy of a held quad).
                    counts[kind] += 1;
                    if win_shape_counts(&counts, open_melds) {
                        tenpai = true;
                    }
                    counts[kind] -= 1;
                }
                if tenpai {
                    break;
                }
                kind += 1;
            }
            counts[ty] += 1;
            if tenpai {
                for t in hand14 {
                    if (*t / 4) as usize == ty && n < out.len() {
                        out[n] = *t;
                        n += 1;
                    }
                }
            }
        }
        ty += 1;
    }
    isort_u8(out, n);
    n
}

/// Kamicha chi availability over type counts (aka folds; honors never).
/// Mirrors `chi_offered` exactly (patterns per called rank).
fn chi_offered_types(counts: &[u8; 34], tile_type: u8) -> bool {
    if tile_type >= 27 {
        return false;
    }
    let suit = tile_type / 9;
    let value = tile_type % 9 + 1;
    let mut has = [false; 10];
    let mut r = 1u8;
    while r <= 9 {
        has[r as usize] = counts[(suit * 9 + r - 1) as usize] > 0;
        r += 1;
    }
    if value >= 3 && has[(value - 2) as usize] && has[(value - 1) as usize] {
        return true;
    }
    if value >= 2 && value <= 8 && has[(value - 1) as usize] && has[(value + 1) as usize] {
        return true;
    }
    if value <= 7 && has[(value + 1) as usize] && has[(value + 2) as usize] {
        return true;
    }
    false
}

/// Shape win with one extra take of `win_type` (norm pool-first copy).
fn shape_win_with(hand: &[u8], win_type: u8, open: usize) -> bool {
    if win_type >= 34 {
        return false;
    }
    // Count-space trial: `type_counts_of(hand)` plus one `win_type` is
    // multiset-equal at the type level to the old id trial (`hand` capped
    // at 23 plus `norm_pool_first(win_type)`, whose type IS `win_type`);
    // callers pass ≤14-take concealed hands so the cap never binds, and
    // the win check is type-level (aka-insensitive) — identical checks.
    let mut trial = [0u8; 34];
    type_counts_of(hand, &mut trial);
    trial[win_type as usize] += 1;
    win_shape_counts(&trial, open)
}

// ---------------------------------------------------------------------------
// Ledger reads (presence → takes / counts / offers).
// ---------------------------------------------------------------------------

/// Collapsed takes of one seat into `out` (ascending by type). Returns len.
fn seat_takes(l: &Ledger, seat: u8, out: &mut [u8; 24]) -> usize {
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
fn concealed_counts(l: &Ledger, seat: u8, out: &mut [u8; 34]) {
    let s = (seat & 3) as usize;
    let mut t = 0usize;
    while t < 34 {
        let row = &l.hands[s][t];
        out[t] = (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8;
        t += 1;
    }
    if l.drawn_live[s].is_some() {
        if let Some(dc) = l.drawn_col[s] {
            let dt = (dc / 4) as usize;
            if dt < 34 {
                out[dt] = out[dt].saturating_sub(1);
            }
        }
    }
}

/// Visible type counts for the row plane (rivers + meld tiles, clamp 4).
fn visible_counts(l: &Ledger, out: &mut [u8; 34]) {
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
fn seat_type_counts(l: &Ledger, seat: u8, out: &mut [u8; 34]) {
    let s = (seat & 3) as usize;
    let mut t = 0usize;
    while t < 34 {
        let row = &l.hands[s][t];
        out[t] = (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8;
        t += 1;
    }
}

/// River type set (furiten proxy reads).
fn river_type_set(l: &Ledger, seat: u8, out: &mut [bool; 34]) {
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
fn alloc_take(l: &mut Ledger, seat: u8, pai: u8) -> Option<u8> {
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
fn remove_take(l: &mut Ledger, seat: u8, take: u8) -> Option<()> {
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
fn presence_count(row: &[u8; 4]) -> u8 {
    (row[0] != 0) as u8 + (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8
}

/// Lowest held take of `seat` rendering the same string as canonical `pai`.
///
/// Oracle min-match (`_match_dahai`/`_match_kakan` keep the lowest take id
/// per string; aka-sensitive via byte render). `None` when the seat holds
/// no such copy (conservation tripwire).
fn min_held_take(l: &Ledger, seat: u8, pai: u8) -> Option<u8> {
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
fn kyushu_offer_append(l: &Ledger, seat: u8, buf: &mut [u32; 32], n: usize) -> usize {
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
fn canonical_rank_take(hand: &[u8], take: u8) -> u8 {
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
fn canonical_take(take: u8) -> u8 {
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
fn draw_offer_ids(l: &mut Ledger, st: &WalkState, seat: u8, out: &mut [u32; 32]) -> usize {
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
                if row[0] != 0 {
                    if n < 32 {
                        out[n] = BASE_DISCARD + (t * 4) as u32;
                        n += 1;
                    }
                }
                if row[1] != 0 || row[2] != 0 || row[3] != 0 {
                    if n < 32 {
                        out[n] = BASE_DISCARD + (t * 4 + 1) as u32;
                        n += 1;
                    }
                }
            } else if row[0] != 0 || row[1] != 0 || row[2] != 0 || row[3] != 0 {
                if n < 32 {
                    out[n] = BASE_DISCARD + (t * 4) as u32;
                    n += 1;
                }
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
    if live {
        if let Some(dc) = l.drawn_col[s] {
            let rep = if l.fold_reports { canonical_take(dc) } else { dc };
            if n < 32 {
                out[n] = BASE_TSUMOGIRI + rep as u32;
                n += 1;
            }
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
            if empty {
                if let Some(d) = drawn_true {
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
                let rep = if l.fold_reports { canonical_take(dc) } else { dc };
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
    let st = WalkState { last_claim: post_claim, ..WalkState::new() };
    let n = draw_offer_ids(&mut *l, &st, seat, &mut buf);
    (buf, n.min(32) as u8)
}

/// Insert `chosen` into a sorted id buffer (draw-mask chosen forcing),
/// keeping ascending order; on overflow keeps chosen + smallest 31.
fn insert_chosen(buf: &mut [u32; 32], len: usize, chosen: u32) -> u8 {
    let mut i = 0usize;
    while i < len && i < 32 {
        if buf[i] == chosen {
            return len.min(32) as u8;
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
        (len + 1) as u8
    } else {
        // Defense-in-depth (unreachable on legal logs): chosen + smallest.
        buf[31] = chosen;
        isort_u32(buf, 32);
        32
    }
}

// ---------------------------------------------------------------------------
// Walk state (row-assembly scratch; table state stays in `Ledger`).
// ---------------------------------------------------------------------------

/// Per-discard window cache: peek-first log-ron fact short-circuits the
/// responder eval; otherwise one eval per discard is cached and shared by
/// emission (`open_window`) and the walled mode gate (`walker_mode`).
struct WalkState {
    histories: [Vec<u8>; 4],
    last_kind: Option<u8>,
    last_discard: Option<(u8, u8)>,
    last_discard_idx: usize,
    /// Last non-kan claim (pon/chi seat + called take) with no intervening
    /// discard or kan: a same-seat tsumo while armed would deal a 15th
    /// ledger take the oracle never deals (Wave-C conservation tripwire).
    /// Draw offers also withhold the just-claimed string while armed.
    last_claim: Option<(u8, u8)>,
    opened_by_discard: bool,
    decided: bool,
    terminal: bool,
    /// A hora applied with no closing end_kyoku since: the wall oracle goes
    /// terminal and rejects trailing row/end events (ron_win shape). Cleared
    /// by end_kyoku; checked at walk end.
    hora_terminal: bool,
    started: bool,
    seq: u32,
    cache_key: Option<(usize, bool)>,
    cache_open: bool,
    cache_pending: ([u8; 3], u8),
}

impl WalkState {
    fn new() -> Self {
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

    fn push_public(&mut self, hid: u8) {
        let mut s = 0usize;
        while s < 4 {
            self.histories[s].push(hid);
            s += 1;
        }
        self.last_kind = Some(hid);
    }

    fn reset_discard(&mut self) {
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
fn skip_ws(s: &[u8], mut j: usize) -> usize {
    while j < s.len() && (s[j] == b' ' || s[j] == b'\t' || s[j] == b'\n' || s[j] == b'\r') {
        j += 1;
    }
    j
}

/// Find a top-level `"key"` occurrence; returns the value start.
fn top_value(span: &[u8], key: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + key.len() + 2 <= span.len() {
        if span[i] == b'"' && &span[i + 1..i + 1 + key.len()] == key && span[i + 1 + key.len()] == b'"'
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
fn hora_tsumo_flag(span: &[u8]) -> bool {
    match top_value(span, b"tsumo") {
        Some(j) => span[j..].starts_with(b"true"),
        None => false,
    }
}

/// Parse `-?(0|[1-9][0-9]{0,18})` at `s[j..]`; returns `(value, end)`.
fn parse_int(s: &[u8], mut j: usize) -> Option<(i64, usize)> {
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
fn deltas_quad(span: &[u8]) -> Option<[i32; 4]> {
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
fn eval_window(l: &Ledger, discarder: u8, tile: u8, chankan: bool) -> (bool, ([u8; 3], u8), [bool; 4]) {
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
fn peek_window_claim(ev: &[StackedEvent<'_>], from_idx: usize, discarder: u8) -> Option<usize> {
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
fn peek_window_ron(ev: &[StackedEvent<'_>], from_idx: usize, discarder: u8) -> bool {
    match peek_window_claim(ev, from_idx, discarder) {
        Some(i) => ev[i].kind == KIND_HORA,
        None => false,
    }
}

/// Cached window eval keyed by (discard event idx, chankan).
fn cached_window(
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
// Row minting (DIRECT to minimal hot planes, §7 order).
// ---------------------------------------------------------------------------

#[inline]
fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Mint one row into the sink: fixed planes + T-variable history + lens.
fn mint_row(
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
            let v = if is_draw { l.scores[i] } else { l.snap_scores[i] };
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
    put_i64(&mut out.planes[PLANE_ROUND_WIND], (l.round_wind - 27) as i64);
    put_i64(&mut out.planes[PLANE_PHASE], phase as i64);
    {
        let mut i = 0usize;
        while i < 4 {
            put_i64(&mut out.planes[PLANE_SEAT_WINDS], (l.seat_wind[i] - 27) as i64);
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

// ---------------------------------------------------------------------------
// Kyoku boundary parses (serde; amortized, never per row).
// ---------------------------------------------------------------------------

struct KyokuHeader {
    oya: u8,
    bakaze: u8,
    scores: [i32; 4],
    tehais: [[u8; 13]; 4],
    honba: u8,
    kyotaku: u8,
    kyoku_no: u8,
}

/// Parse a `start_kyoku` span into validated header takes.
fn parse_header(span: &[u8]) -> Result<KyokuHeader, u8> {
    // Object gate (mirrors serde `as_object` + trailing-garbage reject):
    // a header span is exactly one `{...}` object.
    let p0 = skip_ws(span, 0);
    if p0 >= span.len() || span[p0] != b'{' {
        return Err(WALK_TURN_ORDER);
    }
    let mut pe = span.len();
    while pe > 0
        && (span[pe - 1] == b' ' || span[pe - 1] == b'\t' || span[pe - 1] == b'\n' || span[pe - 1] == b'\r')
    {
        pe -= 1;
    }
    if pe == 0 || span[pe - 1] != b'}' {
        return Err(WALK_TURN_ORDER);
    }
    // Single-pass header field collection (last-wins preserved): one byte
    // scan records the LAST `"key":` value start per known key plus a
    // presence mask. Each hit overwrites the prior one == serde duplicate
    // last-wins == old per-key `hdr_last` rescans, minus the ~6x full-span
    // rewals; unknown/extra keys ignored.
    const HDR_OYA: u8 = 1 << 0;
    const HDR_HONBA: u8 = 1 << 1;
    const HDR_KYOTAKU: u8 = 1 << 2;
    const HDR_SCORES: u8 = 1 << 3;
    const HDR_BAKAZE: u8 = 1 << 4;
    const HDR_TEHAIS: u8 = 1 << 5;
    const HDR_KYOKU: u8 = 1 << 6;
    struct HdrSpans {
        oya: Option<usize>,
        honba: Option<usize>,
        kyotaku: Option<usize>,
        scores: Option<usize>,
        bakaze: Option<usize>,
        tehais: Option<usize>,
        kyoku: Option<usize>,
        mask: u8,
    }
    fn hdr_collect(span: &[u8]) -> HdrSpans {
        let mut out = HdrSpans {
            oya: None,
            honba: None,
            kyotaku: None,
            scores: None,
            bakaze: None,
            tehais: None,
            kyoku: None,
            mask: 0,
        };
        let mut i = 0usize;
        while i < span.len() {
            if span[i] != b'"' {
                i += 1;
                continue;
            }
            // Last-wins preserved: each `"key":` hit overwrites the prior
            // slot, exactly as the old `found = Some(..)` rescan did. Checks
            // mirror the old per-key guard (`"` + key + `"` + ws + `:`) at the
            // same `i`, so the collected value starts are byte-identical.
            if i + 5 <= span.len() && &span[i + 1..i + 4] == b"oya" && span[i + 4] == b'"' {
                let j = skip_ws(span, i + 5);
                if j < span.len() && span[j] == b':' {
                    out.oya = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_OYA;
                }
            }
            if i + 7 <= span.len() && &span[i + 1..i + 6] == b"honba" && span[i + 6] == b'"' {
                let j = skip_ws(span, i + 7);
                if j < span.len() && span[j] == b':' {
                    out.honba = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_HONBA;
                }
            }
            if i + 9 <= span.len() && &span[i + 1..i + 8] == b"kyotaku" && span[i + 8] == b'"' {
                let j = skip_ws(span, i + 9);
                if j < span.len() && span[j] == b':' {
                    out.kyotaku = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_KYOTAKU;
                }
            }
            if i + 8 <= span.len() && &span[i + 1..i + 7] == b"scores" && span[i + 7] == b'"' {
                let j = skip_ws(span, i + 8);
                if j < span.len() && span[j] == b':' {
                    out.scores = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_SCORES;
                }
            }
            if i + 8 <= span.len() && &span[i + 1..i + 7] == b"bakaze" && span[i + 7] == b'"' {
                let j = skip_ws(span, i + 8);
                if j < span.len() && span[j] == b':' {
                    out.bakaze = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_BAKAZE;
                }
            }
            if i + 8 <= span.len() && &span[i + 1..i + 7] == b"tehais" && span[i + 7] == b'"' {
                let j = skip_ws(span, i + 8);
                if j < span.len() && span[j] == b':' {
                    out.tehais = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_TEHAIS;
                }
            }
            if i + 7 <= span.len() && &span[i + 1..i + 6] == b"kyoku" && span[i + 6] == b'"' {
                let j = skip_ws(span, i + 7);
                if j < span.len() && span[j] == b':' {
                    out.kyoku = Some(skip_ws(span, j + 1));
                    out.mask |= HDR_KYOKU;
                }
            }
            i += 1;
        }
        out
    }
    /// String interior bounds at opening quote `p`: `(inner, end, after)`.
    /// Escape-aware; `None` on unterminated/bad-escape/raw-control.
    fn hdr_str(s: &[u8], p: usize) -> Option<(usize, usize, usize)> {
        let mut i = p + 1;
        while i < s.len() {
            let c = s[i];
            if c == b'"' {
                return Some((p + 1, i, i + 1));
            }
            if c == b'\\' {
                i += 1;
                if i >= s.len() {
                    return None;
                }
                match s[i] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                    b'u' => {
                        if i + 4 >= s.len() {
                            return None;
                        }
                        let mut k = 1usize;
                        while k <= 4 {
                            if !s[i + k].is_ascii_hexdigit() {
                                return None;
                            }
                            k += 1;
                        }
                        i += 4;
                    }
                    _ => return None,
                }
            } else if c < 0x20 {
                return None;
            }
            i += 1;
        }
        None
    }
    fn hdr_hex(h: u8) -> u32 {
        match h {
            b'0'..=b'9' => (h - b'0') as u32,
            b'a'..=b'f' => (h - b'a' + 10) as u32,
            b'A'..=b'F' => (h - b'A' + 10) as u32,
            _ => 0,
        }
    }
    /// Decode a validated interior into `out`; `None` on malformed/overflow.
    /// (After `hdr_str` success the only failure left is overflow.)
    fn hdr_dec(raw: &[u8], out: &mut [u8]) -> Option<usize> {
        let mut w = 0usize;
        let mut i = 0usize;
        while i < raw.len() {
            let mut b = raw[i];
            if b == b'\\' {
                i += 1;
                if i >= raw.len() {
                    return None;
                }
                match raw[i] {
                    b'"' => b = b'"',
                    b'\\' => b = b'\\',
                    b'/' => b = b'/',
                    b'b' => b = 0x08,
                    b'f' => b = 0x0C,
                    b'n' => b = b'\n',
                    b'r' => b = b'\r',
                    b't' => b = b'\t',
                    b'u' => {
                        let mut cp: u32 = 0;
                        let mut k = 1usize;
                        while k <= 4 {
                            cp = cp * 16 + hdr_hex(raw[i + k]);
                            k += 1;
                        }
                        i += 4;
                        if (0xD800..0xDC00).contains(&cp) {
                            if raw.len() - i >= 7 && raw[i + 1] == b'\\' && raw[i + 2] == b'u' {
                                let mut lo: u32 = 0;
                                let mut k2 = 3usize;
                                while k2 <= 6 {
                                    if !raw[i + k2].is_ascii_hexdigit() {
                                        break;
                                    }
                                    lo = lo * 16 + hdr_hex(raw[i + k2]);
                                    k2 += 1;
                                }
                                if k2 == 7 && (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    i += 6;
                                } else {
                                    cp = 0xFFFD;
                                }
                            } else {
                                cp = 0xFFFD;
                            }
                        } else if (0xDC00..0xE000).contains(&cp) {
                            cp = 0xFFFD;
                        }
                        if cp < 0x80 {
                            if w >= out.len() {
                                return None;
                            }
                            out[w] = cp as u8;
                            w += 1;
                        } else if cp < 0x800 {
                            if w + 2 > out.len() {
                                return None;
                            }
                            out[w] = (0xC0 | (cp >> 6)) as u8;
                            out[w + 1] = (0x80 | (cp & 0x3F)) as u8;
                            w += 2;
                        } else if cp < 0x10000 {
                            if w + 3 > out.len() {
                                return None;
                            }
                            out[w] = (0xE0 | (cp >> 12)) as u8;
                            out[w + 1] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
                            out[w + 2] = (0x80 | (cp & 0x3F)) as u8;
                            w += 3;
                        } else {
                            if w + 4 > out.len() {
                                return None;
                            }
                            out[w] = (0xF0 | (cp >> 18)) as u8;
                            out[w + 1] = (0x80 | ((cp >> 12) & 0x3F)) as u8;
                            out[w + 2] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
                            out[w + 3] = (0x80 | (cp & 0x3F)) as u8;
                            w += 4;
                        }
                        i += 1;
                        continue;
                    }
                    _ => return None,
                }
            } else if b < 0x20 {
                return None;
            }
            if w >= out.len() {
                return None;
            }
            out[w] = b;
            w += 1;
            i += 1;
        }
        Some(w)
    }
    /// Plain-integer token classes mirroring serde (`-?(0|[1-9][0-9]*)`, no
    /// frac/exp, magnitude capped at u64): `(is_int, is_i64, i64val, end)`.
    /// Floats/exponents/plus/leading-zero and out-of-u64 magnitudes report
    /// `is_int == false` (serde `f64`/invalid → the field rule); `u64` above
    /// `i64::MAX` reports `(true, false, _)` (serde `is_u64` w/o `as_i64`).
    fn hdr_int(s: &[u8], mut j: usize) -> (bool, bool, i64, usize) {
        let neg = j < s.len() && s[j] == b'-';
        if neg {
            j += 1;
        }
        let d0 = j;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        let run = j - d0;
        if run == 0 || (run > 1 && s[d0] == b'0') {
            return (false, false, 0, j);
        }
        if j < s.len() && (s[j] == b'.' || s[j] == b'e' || s[j] == b'E') {
            return (false, false, 0, j);
        }
        let mut mag: u64 = 0;
        let mut k = d0;
        while k < j {
            let d = (s[k] - b'0') as u64;
            match mag.checked_mul(10).and_then(|x| x.checked_add(d)) {
                Some(x) => mag = x,
                None => return (false, false, 0, j),
            }
            k += 1;
        }
        if neg {
            if mag <= i64::MAX as u64 {
                (true, true, -(mag as i64), j)
            } else if mag == i64::MAX as u64 + 1 {
                (true, true, i64::MIN, j)
            } else {
                (false, false, 0, j)
            }
        } else if mag <= i64::MAX as u64 {
            (true, true, mag as i64, j)
        } else {
            (true, false, 0, j)
        }
    }
    /// Required `i64` scalar field (serde `as_i64`): plain integer fitting
    /// `i64` followed by `,`/`}`; `None` covers missing/non-integer/huge-u64.
    fn hdr_i64(s: &[u8], pos: usize) -> Option<i64> {
        let (is_int, is_i64, v, e) = hdr_int(s, pos);
        if !is_int || !is_i64 {
            return None;
        }
        let f = skip_ws(s, e);
        if f < s.len() && s[f] != b',' && s[f] != b'}' {
            return None;
        }
        Some(v)
    }
    /// Required 4-integer `scores` array (serde `is_i64 || is_u64` per
    /// element, then `as_i64().unwrap_or(0) as i32` — huge-u64 maps to `0`,
    /// negatives and i32 overflow wrap via `as`).
    fn hdr_scores(s: &[u8], mut j: usize, out: &mut [i32; 4]) -> bool {
        if j >= s.len() || s[j] != b'[' {
            return false;
        }
        j = skip_ws(s, j + 1);
        let mut k = 0usize;
        while k < 4 {
            let (is_int, is_i64, v, e) = hdr_int(s, j);
            if !is_int {
                return false;
            }
            out[k] = if is_i64 { v as i32 } else { 0 };
            j = skip_ws(s, e);
            if k < 3 {
                if j >= s.len() || s[j] != b',' {
                    return false;
                }
                j = skip_ws(s, j + 1);
            }
            k += 1;
        }
        if j >= s.len() || s[j] != b']' {
            return false;
        }
        let f = skip_ws(s, j + 1);
        if f < s.len() && s[f] != b',' && s[f] != b'}' {
            return false;
        }
        true
    }
    /// Required `bakaze` wind id (`E/S/W/N` → 27/28/29/30, else reject).
    fn hdr_bakaze(s: &[u8], pos: usize) -> Option<u8> {
        if pos >= s.len() || s[pos] != b'"' {
            return None;
        }
        let (vs, ve, q) = hdr_str(s, pos)?;
        let mut buf = [0u8; 16];
        let n = hdr_dec(&s[vs..ve], &mut buf)?;
        let f = skip_ws(s, q);
        if f < s.len() && s[f] != b',' && s[f] != b'}' {
            return None;
        }
        if n == 1 {
            match buf[0] {
                b'E' => Some(27),
                b'S' => Some(28),
                b'W' => Some(29),
                b'N' => Some(30),
                _ => None,
            }
        } else {
            None
        }
    }
    /// Required 4×13 `tehais` takes. Outer shape failures reject
    /// `TURN_ORDER` (serde array/len gate); hand/tile failures reject
    /// `TILE_CONSERVATION` (serde `as_array`/`as_str`/`physical` gate).
    fn hdr_tehais(s: &[u8], pos: usize, out: &mut [[u8; 13]; 4]) -> Result<(), u8> {
        if pos >= s.len() || s[pos] != b'[' {
            return Err(WALK_TURN_ORDER);
        }
        let mut j = skip_ws(s, pos + 1);
        let mut h = 0usize;
        while h < 4 {
            // `]` here ends the outer array early (wrong count → TURN_ORDER);
            // any other non-`[` element is a non-array hand → TILE.
            if j >= s.len() {
                return Err(WALK_TURN_ORDER);
            }
            if s[j] == b']' {
                return Err(WALK_TURN_ORDER);
            }
            if s[j] != b'[' {
                return Err(WALK_TILE_CONSERVATION);
            }
            j = skip_ws(s, j + 1);
            let mut t = 0usize;
            while t < 13 {
                if j >= s.len() || s[j] != b'"' {
                    return Err(WALK_TILE_CONSERVATION);
                }
                let (vs, ve, q) = hdr_str(s, j).ok_or(WALK_TILE_CONSERVATION)?;
                let mut buf = [0u8; 16];
                let n = hdr_dec(&s[vs..ve], &mut buf).ok_or(WALK_TILE_CONSERVATION)?;
                out[h][t] = TileCodec::physical(&buf[..n]).map_err(|_| WALK_TILE_CONSERVATION)?;
                j = skip_ws(s, q);
                if t < 12 {
                    if j >= s.len() || s[j] != b',' {
                        return Err(WALK_TILE_CONSERVATION);
                    }
                    j = skip_ws(s, j + 1);
                }
                t += 1;
            }
            if j >= s.len() || s[j] != b']' {
                return Err(WALK_TILE_CONSERVATION);
            }
            j = skip_ws(s, j + 1);
            if h < 3 {
                if j >= s.len() || s[j] != b',' {
                    return Err(WALK_TURN_ORDER);
                }
                j = skip_ws(s, j + 1);
            }
            h += 1;
        }
        if j >= s.len() || s[j] != b']' {
            return Err(WALK_TURN_ORDER);
        }
        let f = skip_ws(s, j + 1);
        if f < s.len() && s[f] != b',' && s[f] != b'}' {
            return Err(WALK_TURN_ORDER);
        }
        Ok(())
    }
    // Single-pass field uses (last-wins preserved): same collected values via
    // last-occurrence overwrite, so the existing validation order below gives
    // the same verdicts as the old per-key rescans.
    let hdr = hdr_collect(span);
    // Presence-mask invariant (last-wins preserved): bit set iff the slot is
    // `Some`; executable equivalence hook for the collector above.
    debug_assert_eq!(
        hdr.mask,
        (hdr.oya.is_some() as u8) * HDR_OYA
            | (hdr.honba.is_some() as u8) * HDR_HONBA
            | (hdr.kyotaku.is_some() as u8) * HDR_KYOTAKU
            | (hdr.scores.is_some() as u8) * HDR_SCORES
            | (hdr.bakaze.is_some() as u8) * HDR_BAKAZE
            | (hdr.tehais.is_some() as u8) * HDR_TEHAIS
            | (hdr.kyoku.is_some() as u8) * HDR_KYOKU
    );
    let oya = hdr_i64(span, hdr.oya.ok_or(WALK_TURN_ORDER)?)
        .filter(|x| (0..=3).contains(x))
        .ok_or(WALK_TURN_ORDER)? as u8;
    // Last-wins preserved: honba then kyotaku, same order/rules as before.
    // Values retained (u8 range; fail closed beyond it).
    let mut hvals = [0u8; 3];
    let mut hv = 0usize;
    for pos in [
        hdr.honba.ok_or(WALK_TURN_ORDER)?,
        hdr.kyotaku.ok_or(WALK_TURN_ORDER)?,
        hdr.kyoku.ok_or(WALK_TURN_ORDER)?,
    ] {
        let v = hdr_i64(span, pos).filter(|x| (0..=255).contains(x)).ok_or(WALK_TURN_ORDER)?;
        hvals[hv] = v as u8;
        hv += 1;
    }
    let mut scores = [0i32; 4];
    // Last-wins preserved: scores slot is the old `hdr_last(scores)` value.
    let spos = hdr.scores.ok_or(WALK_TURN_ORDER)?;
    if !hdr_scores(span, spos, &mut scores) {
        return Err(WALK_TURN_ORDER);
    }
    // Last-wins preserved: bakaze slot is the old `hdr_last(bakaze)` value.
    let bakaze = hdr_bakaze(span, hdr.bakaze.ok_or(WALK_TURN_ORDER)?).ok_or(WALK_TURN_ORDER)?;
    let mut tehais = [[0u8; 13]; 4];
    // Last-wins preserved: tehais slot is the old `hdr_last(tehais)` value.
    hdr_tehais(span, hdr.tehais.ok_or(WALK_TURN_ORDER)?, &mut tehais)?;
    Ok(KyokuHeader {
        oya,
        bakaze,
        scores,
        tehais,
        honba: hvals[0],
        kyotaku: hvals[1],
        kyoku_no: hvals[2],
    })
}

struct RyukyokuBody {
    deltas: [i32; 4],
    reason: Option<RyukyokuReason>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum RyukyokuReason {
    DrawEnd,
    Abortive,
    Unmapped,
}

/// Parse a `ryukyoku` span: deltas quad (required) + reason class.
fn parse_ryukyoku(span: &[u8]) -> Result<RyukyokuBody, u8> {
    let deltas = deltas_quad(span).ok_or(WALK_TURN_ORDER)?;
    // Object gate (mirrors serde `as_object`): a ryukyoku span is one object.
    let p0 = skip_ws(span, 0);
    if p0 >= span.len() || span[p0] != b'{' {
        return Err(WALK_TURN_ORDER);
    }
    /// LAST `"key":` value start (serde duplicate last-wins); `None` if absent.
    fn ryu_last(span: &[u8], key: &[u8]) -> Option<usize> {
        let mut found = None;
        let mut i = 0usize;
        while i + key.len() + 2 <= span.len() {
            if span[i] == b'"'
                && &span[i + 1..i + 1 + key.len()] == key
                && span[i + 1 + key.len()] == b'"'
            {
                let j = skip_ws(span, i + 1 + key.len() + 1);
                if j < span.len() && span[j] == b':' {
                    found = Some(skip_ws(span, j + 1));
                }
            }
            i += 1;
        }
        found
    }
    /// String interior bounds at opening quote `p`: `(inner, end, after)`.
    fn ryu_str(s: &[u8], p: usize) -> Option<(usize, usize, usize)> {
        let mut i = p + 1;
        while i < s.len() {
            let c = s[i];
            if c == b'"' {
                return Some((p + 1, i, i + 1));
            }
            if c == b'\\' {
                i += 1;
                if i >= s.len() {
                    return None;
                }
                match s[i] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                    b'u' => {
                        if i + 4 >= s.len() {
                            return None;
                        }
                        let mut k = 1usize;
                        while k <= 4 {
                            if !s[i + k].is_ascii_hexdigit() {
                                return None;
                            }
                            k += 1;
                        }
                        i += 4;
                    }
                    _ => return None,
                }
            } else if c < 0x20 {
                return None;
            }
            i += 1;
        }
        None
    }
    fn ryu_hex(h: u8) -> u32 {
        match h {
            b'0'..=b'9' => (h - b'0') as u32,
            b'a'..=b'f' => (h - b'a' + 10) as u32,
            b'A'..=b'F' => (h - b'A' + 10) as u32,
            _ => 0,
        }
    }
    /// Decode a validated interior into `out`; `None` iff it overflows
    /// (post-validation escapes cannot otherwise fail; overlong reasons
    /// exceed every table entry so the caller maps this to Unmapped).
    fn ryu_dec(raw: &[u8], out: &mut [u8]) -> Option<usize> {
        let mut w = 0usize;
        let mut i = 0usize;
        while i < raw.len() {
            let mut b = raw[i];
            if b == b'\\' {
                i += 1;
                if i >= raw.len() {
                    return None;
                }
                match raw[i] {
                    b'"' => b = b'"',
                    b'\\' => b = b'\\',
                    b'/' => b = b'/',
                    b'b' => b = 0x08,
                    b'f' => b = 0x0C,
                    b'n' => b = b'\n',
                    b'r' => b = b'\r',
                    b't' => b = b'\t',
                    b'u' => {
                        let mut cp: u32 = 0;
                        let mut k = 1usize;
                        while k <= 4 {
                            cp = cp * 16 + ryu_hex(raw[i + k]);
                            k += 1;
                        }
                        i += 4;
                        if (0xD800..0xDC00).contains(&cp) {
                            if raw.len() - i >= 7 && raw[i + 1] == b'\\' && raw[i + 2] == b'u' {
                                let mut lo: u32 = 0;
                                let mut k2 = 3usize;
                                while k2 <= 6 {
                                    if !raw[i + k2].is_ascii_hexdigit() {
                                        break;
                                    }
                                    lo = lo * 16 + ryu_hex(raw[i + k2]);
                                    k2 += 1;
                                }
                                if k2 == 7 && (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    i += 6;
                                } else {
                                    cp = 0xFFFD;
                                }
                            } else {
                                cp = 0xFFFD;
                            }
                        } else if (0xDC00..0xE000).contains(&cp) {
                            cp = 0xFFFD;
                        }
                        if cp < 0x80 {
                            if w >= out.len() {
                                return None;
                            }
                            out[w] = cp as u8;
                            w += 1;
                        } else if cp < 0x800 {
                            if w + 2 > out.len() {
                                return None;
                            }
                            out[w] = (0xC0 | (cp >> 6)) as u8;
                            out[w + 1] = (0x80 | (cp & 0x3F)) as u8;
                            w += 2;
                        } else if cp < 0x10000 {
                            if w + 3 > out.len() {
                                return None;
                            }
                            out[w] = (0xE0 | (cp >> 12)) as u8;
                            out[w + 1] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
                            out[w + 2] = (0x80 | (cp & 0x3F)) as u8;
                            w += 3;
                        } else {
                            if w + 4 > out.len() {
                                return None;
                            }
                            out[w] = (0xF0 | (cp >> 18)) as u8;
                            out[w + 1] = (0x80 | ((cp >> 12) & 0x3F)) as u8;
                            out[w + 2] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
                            out[w + 3] = (0x80 | (cp & 0x3F)) as u8;
                            w += 4;
                        }
                        i += 1;
                        continue;
                    }
                    _ => return None,
                }
            } else if b < 0x20 {
                return None;
            }
            if w >= out.len() {
                return None;
            }
            out[w] = b;
            w += 1;
            i += 1;
        }
        Some(w)
    }
    let reason = match ryu_last(span, b"reason") {
        // Missing or non-string (`as_str` is `None`) decodes as `""` → None.
        None => None,
        Some(j) => {
            if j >= span.len() || span[j] != b'"' {
                None
            } else {
                let (vs, ve, _) = ryu_str(span, j).ok_or(WALK_TURN_ORDER)?;
                let mut buf = [0u8; 64];
                let n = match ryu_dec(&span[vs..ve], &mut buf) {
                    Some(n) => n,
                    None => return Ok(RyukyokuBody { deltas, reason: Some(RyukyokuReason::Unmapped) }),
                };
                match &buf[..n] {
                    b"" => None,
                    b"exhaustive_draw" | b"nagashi_mangan" => Some(RyukyokuReason::DrawEnd),
                    b"kyushu_kyuhai" | b"kyuushu_kyuuhai" | b"suucha_riichi" | b"sanchaho"
                    | b"sanchahou" | b"suukaikan" | b"suukansansen" | b"suufon_renda"
                    | b"sufuurenta" => Some(RyukyokuReason::Abortive),
                    _ => Some(RyukyokuReason::Unmapped),
                }
            }
        }
    };
    Ok(RyukyokuBody { deltas, reason })
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

type WalkResult<T> = Result<T, WalkReject>;

fn rej(game_idx: u32, idx: usize, reason: u8) -> WalkReject {
    WalkReject::new(game_idx, idx, reason)
}

fn check_drawer(l: &mut Ledger, game_idx: u32, idx: usize, seat: u8) -> WalkResult<()> {
    match l.drawer {
        None => {
            l.drawer = Some(seat);
            Ok(())
        }
        Some(d) if d == seat => Ok(()),
        Some(_) => Err(rej(game_idx, idx, WALK_TURN_ORDER)),
    }
}

fn do_start_kyoku(
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
            let take = alloc_take(l, s as u8, h.tehais[s][j])
                .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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

fn do_tsumo(
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
    // Wave-C conservation (fail-closed favors oracle): a non-kan claim
    // (pon/chi) leaves the claimer holding a full ledger (pair consumed,
    // called tile melded, no discard yet); a same-seat tsumo with no
    // intervening discard and no rinshan pending would deal a 15th take
    // (12 concealed + 3 meld on the pon_ext vector) the strict wall-less
    // oracle desyncs on. Reject as tile-conservation (G5 closed set, no
    // taxonomy churn). Kans disarm below (rinshan replacement is real).
    if st.last_claim.map(|(s, _)| s) == Some(actor) && !l.kan_pending {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    let take = alloc_take(l, actor, ev.pai).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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
fn apply_discard(
    ctx: &GameCtx<'_>,
    l: &mut Ledger,
    idx: usize,
    actor: u8,
    tile: u8,
    _drawn: Option<u8>,
) -> WalkResult<()> {
    remove_take(l, actor, tile)
        .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
    push_river(ctx, l, idx, actor, tile)?;
    l.exp_drawer = Some(actor);
    Ok(())
}

fn push_river(ctx: &GameCtx<'_>, l: &mut Ledger, idx: usize, actor: u8, tile: u8) -> WalkResult<()> {
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
fn open_window(
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

fn do_dahai(
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
        let rep = if l.fold_reports { canonical_take(live_take) } else { live_take };
        (ACT_TSUMOGIRI, rep, live_take)
    } else if ev.tsumogiri {
        if l.fold_reports {
            // Fold regime keeps the old lenient path: report the logged
            // canonical id; removal takes the lowest held copy rendering it
            // (exactly the old rank-slot removal).
            let state = min_held_take(l, actor, ev.pai)
                .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
            (ACT_TSUMOGIRI, ev.pai, state)
        } else {
            let live_take = l.drawn_live[s].ok_or(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC))?;
            if TileCodec::bytes_of(live_take) != TileCodec::bytes_of(ev.pai) {
                return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
            }
            (ACT_TSUMOGIRI, live_take, live_take)
        }
    } else {
        let take =
            min_held_take(l, actor, ev.pai).ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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
            l.tsumo_offered =
                tn == 14 && win_shape_14(&takes[..tn], l.meld_lens[s] as usize);
        }
    } else {
        mlen = draw_offer_ids(&mut *l, st, actor, &mut buf);
        mlen = kyushu_offer_append(l, actor, &mut buf, mlen);
    }
    let mlen8 = insert_chosen(&mut buf, mlen, chosen);
    mint_row(l, st, out, actor, PHASE_DRAW, actor, chosen, &buf[..mlen8 as usize]);
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

fn do_reach(
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
    let state = min_held_take(l, ractor, decl.pai)
        .ok_or(rej(ctx.game_idx, decl_idx, WALK_TILE_CONSERVATION))?;
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
    mint_row(l, st, out, ractor, PHASE_DRAW, ractor, chosen, &buf[..mlen8 as usize]);
    // Post-mint state (same doujun/window rules as a plain discard first).
    let s = ractor as usize;
    l.missed[s] = 0;
    l.ippatsu[s] = 0;
    {
        let (_, _, ron) = eval_window(l, ractor, tile, false);
        let claim_seat =
            peek_window_claim(ev_all, decl_idx + 1, ractor).map(|ci| ev_all[ci].actor);
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
/// Resolve logged consumed ids to sorted takes for reports and state.
///
/// Fold regime: pool-first takes per string occurrence with the
/// called-collision swap (SIM oracle renders). Unfold regime: lowest held
/// takes per string with a called-collision tripwire (wall oracle encodes
/// held copies). Ownership validation is identical either way.
fn resolve_consumed(
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
                let tmp = sorted[a];
                sorted[a] = sorted[b];
                sorted[b] = tmp;
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
                || need_plain[t as usize] > (row[1] != 0) as u8 + (row[2] != 0) as u8 + (row[3] != 0) as u8
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
                    let tmp = picked[a];
                    picked[a] = picked[b];
                    picked[b] = tmp;
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
        let want = TileCodec::bytes_of(sorted[i])
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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
                let tmp = picked[a];
                picked[a] = picked[b];
                picked[b] = tmp;
            }
            b += 1;
        }
        a += 1;
    }
    Ok(picked)
}

fn claim_history_kind(kind: u8) -> u8 {
    match kind {
        KIND_CHI => H_CHI,
        KIND_PON => H_PON,
        _ => H_DAIMINKAN,
    }
}

fn claim_meld_kind(kind: u8) -> u8 {
    match kind {
        KIND_CHI => ACT_CHI,
        KIND_PON => ACT_PON,
        _ => ACT_DAIMINKAN,
    }
}

fn do_claim(
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
    let chosen = chosen_claim_id(act, called, &picked[..needed], offset)
        .ok_or(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED))?;
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
            let tn = held_takes(&l.hands[actor as usize][ct.min(33)], ct.min(33) as u8, &mut takes);
            let mut a = 0usize;
            while a < tn {
                let mut b = a + 1;
                while b < tn {
                    if mlen < 32 {
                        if let Some(id) =
                            chosen_claim_id(ACT_PON, called, &[takes[a], takes[b]], offset)
                        {
                            mask[mlen] = id;
                            mlen += 1;
                        }
                    }
                    b += 1;
                }
                a += 1;
            }
            // Daiminkan alternative whenever three copies are held.
            if tn >= 3 {
                let mut quad = [takes[0], takes[1], takes[2]];
                quad.sort_unstable();
                if mlen < 32 {
                    if let Some(id) = chosen_claim_id(ACT_DAIMINKAN, called, &quad, offset) {
                        mask[mlen] = id;
                        mlen += 1;
                    }
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
                            if mlen < 32 {
                                if let Some(id) =
                                    chosen_claim_id(ACT_CHI, called, &[c0, c1], -1)
                                {
                                    mask[mlen] = id;
                                    mlen += 1;
                                }
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
                let tmp = tiles[a];
                tiles[a] = tiles[b];
                tiles[b] = tmp;
            }
            b += 1;
        }
        a += 1;
    }
    // Remove consumed takes from the hand (class ranks, seat-local).
    let mut i = 0usize;
    while i < needed {
        let take = removal[i];
        remove_take(l, actor, take)
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        i += 1;
    }
    let s = actor as usize;
    if l.meld_lens[s] >= 4 {
        return Err(rej(ctx.game_idx, idx, WALK_TURN_ORDER));
    }
    l.melds[s][l.meld_lens[s] as usize] = Meld {
        kind: act,
        owner: actor,
        tiles: (tiles, total as u8),
    };
    l.meld_lens[s] += 1;
    l.drawer = Some(actor);
    l.exp_drawer = Some(actor);
    l.kan_pending = kind == KIND_DAIMINKAN;
    if kind == KIND_DAIMINKAN {
        l.kan_count = l.kan_count.saturating_add(1);
    }
    // Arm (pon/chi) or disarm (daiminkan: rinshan pending) the Wave-C
    // post-claim conservation tripwire.
    st.last_claim = if kind == KIND_DAIMINKAN { None } else { Some((actor, called)) };
    st.push_public(claim_history_kind(kind));
    st.last_discard = None;
    Ok(())
}

fn do_ankan(
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
    let picked = resolve_consumed(ctx, l, idx, actor, &consumed_ids[..ev.consumed.1 as usize], 4, None, l.fold_reports)?;
    let base = (picked[0] / 4) * 4;
    if picked != [base, base + 1, base + 2, base + 3] {
        return Err(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION));
    }
    let chosen = chosen_claim_id(ACT_ANKAN, 0, &picked, 0)
        .ok_or(rej(ctx.game_idx, idx, WALK_ACTION_UNRESOLVED))?;
    // Offer + match check (logged quad must be offered, else desync).
    let mut offer = [0u32; 32];
    let olen = draw_offer_ids(&mut *l, st, actor, &mut offer);
    if l.riichi[s] != 0 {
        // Kans need a live draw turn; without one the log post-claim kan
        // fails closed.
        let live_drawn = l.drawn_col[s].ok_or(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC))?;
        // Forced pair + same-wait offer quads + logged quad (log authority).
        let mut combo = [0u32; 32];
        let rep_drawn = if l.fold_reports { canonical_take(live_drawn) } else { live_drawn };
        let a = chosen_id(ACT_DISCARD, rep_drawn, 0);
        let b = chosen_id(ACT_TSUMOGIRI, rep_drawn, 0);
        combo[0] = a;
        combo[1] = b;
        let mut cn = if a == b { 1 } else { 2 };
        let mut i = 0usize;
        while i < olen {
            if offer[i] >= BASE_ANKAN && offer[i] < BASE_ANKAN + 34 {
                if cn < 32 {
                    combo[cn] = offer[i];
                    cn += 1;
                }
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
        mint_row(l, st, out, actor, PHASE_DRAW, actor, chosen, &combo[..mlen8 as usize]);
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
        mint_row(l, st, out, actor, PHASE_DRAW, actor, chosen, &offer[..mlen8 as usize]);
    }
    l.drawn_live[s] = None;
    let mut i = 0usize;
    while i < 4 {
        let take = picked[i];
        remove_take(l, actor, take)
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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

fn do_kakan(
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
    mint_row(l, st, out, actor, PHASE_DRAW, actor, chosen, &buf[..mlen8 as usize]);
    l.drawn_live[s] = None;
    // The added copy leaves the hand (the meld record keeps the pon).
    // Reports may name a canonical copy (fold regime); removal resolves
    // the true held take rendering the same string (exactly one: the pon
    // holds the other three).
    {
        let state = min_held_take(l, actor, added)
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        remove_take(l, actor, state)
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
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

fn do_hora(
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
        let drawn = l
            .drawn_col[s]
            .ok_or(rej(ctx.game_idx, idx, WALK_TILE_CONSERVATION))?;
        let mut takes = [0u8; 24];
        let tn = seat_takes(l, winner, &mut takes);
        if !win_shape_14(&takes[..tn], l.meld_lens[s] as usize) {
            return Err(rej(ctx.game_idx, idx, WALK_ENGINE_DESYNC));
        }
        // Report tile: folded canonical in fold regime (SIM oracle renders
        // the pool-first copy), true takes unfolded.
        let rep = if l.fold_reports { canonical_take(drawn) } else { drawn };
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
                if offer[i] >= BASE_ANKAN && offer[i] < BASE_ANKAN + 34 {
                    if cn < 32 {
                        combo[cn] = offer[i];
                        cn += 1;
                    }
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
        mint_row(l, st, out, winner, PHASE_DRAW, winner, chosen, &buf[..mlen8 as usize]);
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

fn kyushu_infer(l: &Ledger) -> bool {
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

fn do_ryukyoku(
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
        Some(RyukyokuReason::Unmapped) => return Err(rej(ctx.game_idx, idx, WALK_UNMAPPED_RYUKYOKU)),
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
fn mode_pending(
    l: &Ledger,
    st: &mut WalkState,
    discarder: u8,
    tile: u8,
    disc_idx: usize,
) -> ([u8; 3], u8) {
    cached_window(st, l, discarder, tile, disc_idx, false).1
}

fn check_mode_for(
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
    if let Some((discarder, tile)) = st.last_discard {
        if st.opened_by_discard {
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
    }
    // Draw mode: claims outside any window fail closed.
    if matches!(kind, KIND_CHI | KIND_PON | KIND_DAIMINKAN) {
        return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
    }
    if kind == KIND_HORA {
        let ev = &ev_all[idx];
        if !hora_tsumo_flag(ev.span) {
            // Chankan: kakan set last_discard with opened_by_discard false.
            if let Some((discarder, tile)) = st.last_discard {
                if ev.target == discarder {
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
            }
            return Err(rej(ctx.game_idx, idx, WALK_CLAIM_NO_OFFER));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Skip kinds for reach-collapse lookahead (mirrors `SKIP_TYPES`).
fn is_skip_kind(kind: u8) -> bool {
    matches!(
        kind,
        KIND_START_KYOKU | KIND_TSUMO | KIND_DORA | KIND_REACH_ACCEPTED | KIND_RYUKYOKU
            | KIND_END_KYOKU | KIND_START
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

// ---------------------------------------------------------------------------
// Tests (P2-A walk gates: F1 bitwise, ids, quarantine agreement).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod walk_tests {
    use super::*;
    use crate::gate::gate_game;
    use crate::ingest::{frame_spans, KIND_OTHER};
    use crate::ledger::{N_WALK_PLANES, WALK_BARE_DORA, WALK_TILE_CONSERVATION, WALK_TURN_ORDER};

    const TEHAIS_F1: [&[&str]; 4] = [
        &["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
        &["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
        &["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
        &["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
    ];

    fn kyoku_line(tehais: &[&[&str]], oya: u8) -> String {
        let mut seats = Vec::new();
        for hand in tehais {
            let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
            seats.push(format!("[{}]", tiles.join(",")));
        }
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"F\",\"honba\":0,\"kyoku\":1,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
            seats.join(",")
        )
    }

    fn tsumo_line(actor: u8, pai: &str) -> String {
        format!("{{\"type\":\"tsumo\",\"actor\":{actor},\"pai\":{pai:?}}}")
    }

    fn dahai_line(actor: u8, pai: &str, tsumogiri: bool) -> String {
        format!("{{\"type\":\"dahai\",\"actor\":{actor},\"pai\":{pai:?},\"tsumogiri\":{tsumogiri}}}")
    }

    /// F1 golden content (wall-less): 5 tsumogiri discards, no hora.
    fn f1_text() -> Vec<u8> {
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&TEHAIS_F1, 0)];
        for (actor, pai) in [(0, "5pr"), (1, "5p"), (2, "5p"), (3, "5p"), (0, "6p")] {
            lines.push(tsumo_line(actor, pai));
            lines.push(dahai_line(actor, pai, true));
        }
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    struct Walked {
        rows: u16,
        planes: [Vec<u8>; N_WALK_PLANES],
        hist_lens: Vec<u32>,
        ledger: Ledger,
    }

    fn walk_bytes(text: &[u8], game_idx: u32, digest: Option<(&str, std::sync::Arc<str>)>) -> Walked {
        let mut arena = Vec::new();
        let game = frame_spans(&mut arena, text, 7, game_idx).expect("frame");
        let verdict = gate_game(game_idx, &game.events).expect("gate ok");
        assert!(verdict.is_sample());
        let (wall_digest, digest_arc) = match digest {
            Some((d, a)) => (Some(d), Some(a)),
            None => (None, None),
        };
        let ctx = GameCtx {
            game_idx,
            wall_digest,
            digest_arc,
        };
        let mut ledger = Ledger::new();
        let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist_lens = Vec::new();
        let rows = {
            let mut sink = RowSink::new(&mut planes, &mut hist_lens);
            walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect("walk")
        };
        Walked {
            rows,
            planes,
            hist_lens,
            ledger,
        }
    }

    fn rd_i64(plane: &[u8], row: usize) -> i64 {
        let o = row * 8;
        i64::from_le_bytes(plane[o..o + 8].try_into().unwrap())
    }

    fn rd_dora(plane: &[u8], row: usize, slot: usize) -> i32 {
        let o = row * 20 + slot * 4;
        i32::from_le_bytes(plane[o..o + 4].try_into().unwrap())
    }

    fn rd_score(plane: &[u8], row: usize, slot: usize) -> i32 {
        let o = row * 16 + slot * 4;
        i32::from_le_bytes(plane[o..o + 4].try_into().unwrap())
    }

    fn rd_u8(plane: &[u8], row: usize, slot: usize) -> u8 {
        plane[row * 34 + slot]
    }

    fn legal_of(w: &Walked, row: usize) -> Vec<u32> {
        let base = row * 136;
        let p = &w.planes[PLANE_LEGAL];
        let len = i64::from_le_bytes(p[base + 128..base + 136].try_into().unwrap()) as usize;
        let mut out = Vec::new();
        for k in 0..len {
            out.push(i32::from_le_bytes(p[base + k * 4..base + k * 4 + 4].try_into().unwrap()) as u32);
        }
        out
    }

    fn hist_of(w: &Walked, row: usize) -> Vec<i64> {
        let mut off = 0usize;
        for r in 0..row {
            off += w.hist_lens[r] as usize;
        }
        let p = &w.planes[PLANE_HIST_KIND];
        let mut out = Vec::new();
        for k in 0..w.hist_lens[row] as usize {
            out.push(i64::from_le_bytes(p[(off + k) * 8..(off + k) * 8 + 8].try_into().unwrap()));
        }
        out
    }

    #[test]
    fn f1_five_decisions_collapsed_bitwise() {
        let w = walk_bytes(&f1_text(), 3, None);
        assert_eq!(w.rows, 5);
        // Wall-less collapse oracle pin: [192,193,193,193,196].
        let mut chosen = Vec::new();
        let mut actors = Vec::new();
        for r in 0..5 {
            chosen.push(rd_i64(&w.planes[PLANE_CHOSEN], r));
            actors.push(rd_i64(&w.planes[PLANE_ACTOR], r));
        }
        assert_eq!(chosen, [192, 193, 193, 193, 196]);
        assert_eq!(actors, [0, 1, 2, 3, 0]);
        // Scalars: draw phase, dealer 0, round wind E=0, seat winds 0..3.
        for r in 0..5 {
            assert_eq!(rd_i64(&w.planes[PLANE_PHASE], r), 1);
            assert_eq!(rd_i64(&w.planes[PLANE_DEALER], r), 0);
            assert_eq!(rd_i64(&w.planes[PLANE_ROUND_WIND], r), 0);
        }
        for r in 0..5 {
            for s in 0..4 {
                assert_eq!(rd_score(&w.planes[PLANE_SCORES], r, s), 25000);
            }
        }
        // Dora never revealed: all −1 (G3).
        for r in 0..5 {
            for k in 0..5 {
                assert_eq!(rd_dora(&w.planes[PLANE_DORA], r, k), -1);
            }
        }
        // Concealed row0 exact: 1m×4, 5m×4, 9m×4, 4p (draw excluded).
        let mut c0 = [0u8; 34];
        for t in 0..34 {
            c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
        }
        assert_eq!(c0[0], 4);
        assert_eq!(c0[4], 4);
        assert_eq!(c0[8], 4);
        assert_eq!(c0[12], 1);
        // Histories: [game,round,advance,draw] then +2 per same-private
        // prefix; seat3's 5p discard opens a kamicha-chi window (3p×4+4p
        // held), so row3/row4 carry the call_window envelope.
        assert_eq!(w.hist_lens, [4, 6, 8, 11, 14]);
        assert_eq!(hist_of(&w, 0), [0, 1, 2, 3]);
        assert!(hist_of(&w, 3).contains(&(H_CALL_WINDOW as i64)));
        // Legal: sorted, non-empty, chosen ∈ set (G2 legality at fill).
        for r in 0..5 {
            let legal = legal_of(&w, r);
            assert!(!legal.is_empty());
            let mut sorted = legal.clone();
            sorted.sort_unstable();
            assert_eq!(legal, sorted);
            assert!(legal.contains(&(chosen[r] as u32)));
        }
        // Row0 offers the pool-first strings + twin.
        let l0 = legal_of(&w, 0);
        for id in [4, 20, 21, 36, 52, 56, 192] {
            assert!(l0.contains(&id), "row0 legal missing {id}: {l0:?}");
        }
    }

    #[test]
    fn f1_walled_gate_agrees_on_claim_free_game() {
        // Walled regime reports true takes (wall oracle): seats 1-3 draw
        // pool takes 53, 54, 55, reported unfolded.
        let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
        let w = {
            let digest: &str = &arc;
            walk_bytes(&f1_text(), 3, Some((digest, std::sync::Arc::clone(&arc))))
        };
        assert_eq!(w.rows, 5);
        let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen, [192, 193, 194, 195, 196]);
    }

    #[test]
    fn chosen_id_arithmetic_pinned() {
        assert_eq!(chosen_id(ACT_PASS, 0, 0), 0);
        assert_eq!(chosen_id(ACT_DISCARD, 0, 0), 4);
        assert_eq!(chosen_id(ACT_DISCARD, 135, 0), 139);
        assert_eq!(chosen_id(ACT_TSUMOGIRI, 52, 0), 192);
        assert_eq!(chosen_id(ACT_TSUMOGIRI, 53, 0), 193);
        assert_eq!(chosen_id(ACT_RIICHI_DISCARD, 56, 0), 332);
        assert_eq!(chosen_id(ACT_KAKAN, 0, 0), 6110);
        assert_eq!(chosen_id(ACT_TSUMO, 53, 0), 6707);
        assert_eq!(chosen_id(ACT_ANKAN, 0, 0), 6076);
        assert_eq!(chosen_id(ACT_ANKAN, 33, 0), 6109);
        assert_eq!(chosen_id(ACT_RON, 0, 0), 6246);
        assert_eq!(chosen_id(ACT_RON, 0, 1), 6247);
        assert_eq!(chosen_id(ACT_RON, 0, 2), 6248);
        assert_eq!(chosen_id(ACT_RON, 135, 2), 6653);
        assert_eq!(chosen_id(99, 0, 0), UNRESOLVED_ID);
        assert_eq!(chosen_id(ACT_DISCARD, 136, 0), UNRESOLVED_ID);
        assert_eq!(chosen_id(ACT_RON, 0, 3), UNRESOLVED_ID);
        assert_eq!(source_offset(0, 1), -1);
        assert_eq!(source_offset(1, 0), 1);
        assert_eq!(source_offset(2, 0), 2);
    }

    #[test]
    fn claim_ids_match_published_table() {
        // chi: called 4m-copy0 (12), consumed [2m-copy0, 5m-aka] (8,16).
        assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 16], -1), Some(812));
        // chi: first entry (called 1m, [2m,3m] copies 0,0).
        assert_eq!(chosen_claim_id(ACT_CHI, 0, &[4, 8], -1), Some(412));
        // pon: called 0 over [1,2] kamicha.
        assert_eq!(chosen_claim_id(ACT_PON, 0, &[1, 2], -1), Some(4444));
        assert_eq!(chosen_claim_id(ACT_PON, 1, &[0, 2], 1), Some(4454));
        // daiminkan: called 0 over the other three.
        assert_eq!(chosen_claim_id(ACT_DAIMINKAN, 0, &[1, 2, 3], 2), Some(5670));
        assert_eq!(chosen_claim_id(ACT_DAIMINKAN, 17, &[16, 18, 19], -1), Some(5719));
        // ankan: full blocks.
        assert_eq!(chosen_claim_id(ACT_ANKAN, 0, &[0, 1, 2, 3], 0), Some(6076));
        // Rejects: bad offset, bad pattern, wrong counts, honors chi.
        assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 16], 1), None);
        assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 9], -1), None);
        assert_eq!(chosen_claim_id(ACT_CHI, 108, &[109, 110], -1), None);
        assert_eq!(chosen_claim_id(ACT_PON, 0, &[1, 2], 0), None);
        assert_eq!(chosen_claim_id(ACT_PON, 0, &[0, 1], -1), None);
        assert_eq!(chosen_claim_id(ACT_DAIMINKAN, 0, &[1, 2, 4], -1), None);
        assert_eq!(chosen_claim_id(ACT_ANKAN, 0, &[0, 1, 2, 4], 0), None);
    }

    fn custom_tehais() -> Vec<Vec<&'static str>> {
        vec![
            vec!["1m", "1m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p", "3p", "4p"],
            vec!["1m", "1m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "7s", "7s", "7s"],
            vec!["2m", "2m", "2m", "2m", "3m", "3m", "3m", "3m", "4p", "4p", "4p", "5s", "5s"],
            vec!["7m", "7m", "7m", "7m", "8m", "8m", "8m", "8m", "E", "E", "E", "E", "F"],
        ]
    }

    fn pon_v1_text() -> Vec<u8> {
        // Oracle-legal prefix: the pon row itself, kyoku closed without a
        // post-claim draw (V1 in the parity notes).
        let held = custom_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "1m", false));
        lines.push(
            "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
                .to_string(),
        );
        lines.push("{\"type\":\"end_kyoku\"}".to_string());
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    fn pon_text() -> Vec<u8> {
        let held = custom_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "1m", false));
        lines.push(
            "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
                .to_string(),
        );
        lines.push(tsumo_line(1, "6p"));
        lines.push(dahai_line(1, "6p", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn pon_claim_rows_match_oracle_exact() {
        // V1 vectors from the wall-less oracle, bit-for-bit: chosen, full
        // legal sets (discards + twin + riichi + ankan on row0), concealed
        // takes, histories, phases.
        let w = walk_bytes(&pon_v1_text(), 9, None);
        assert_eq!(w.rows, 2);
        let chosen: Vec<i64> = (0..2).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen, [4, 4444]);
        assert_eq!(legal_of(&w, 0), [4, 36, 44, 48, 52, 57, 193, 308, 309, 310, 311, 6084]);
        assert_eq!(legal_of(&w, 1), [0, 4444]);
        assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 0), 1);
        assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 1), 2);
        assert_eq!(hist_of(&w, 0), [0, 1, 2, 3]);
        assert_eq!(hist_of(&w, 1), [0, 1, 2, 4, 7]);
        let mut c0 = [0u8; 34];
        let mut c1 = [0u8; 34];
        for t in 0..34 {
            c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
            c1[t] = rd_u8(&w.planes[PLANE_CONCEALED], 1, t);
        }
        // Row0: 1m×2, 9m×4, 2p×3, 3p×3, 4p (drawn 5p excluded).
        assert_eq!(&c0[0..14], &[2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 3, 1, 0]);
        // Row1: capture-before-mutate — the consumed 1m pair is still
        // present (oracle row shows the same 13 takes).
        assert_eq!(c1[0], 2);
        assert_eq!(c1[5], 4);
        assert_eq!(c1[9], 4);
        assert_eq!(c1[24], 3);
        assert_eq!(c1.iter().map(|x| *x as u32).sum::<u32>(), 13);
        // Meld installed on seat 1: pon [1,2] + called 0.
        assert_eq!(w.ledger.meld_lens[1], 1);
        let meld = &w.ledger.melds[1][0];
        assert_eq!(meld.kind, ACT_PON);
        assert_eq!(meld.owner, 1);
        assert_eq!(meld.tiles, ([0, 1, 2, 0], 3));
    }

    #[test]
    fn pon_ext_post_claim_draw_is_conservation_reject() {
        // Wave-C closure (was: open parity item, feed accepted 3 rows
        // [4,4444,196]): a claimer tsumo after a non-kan claim with no
        // intervening discard and no rinshan pending deals a 15th ledger
        // take (12 concealed + 3 meld) the strict wall-less oracle desyncs
        // on (whole-game quarantine, zero rows). Fail-closed favors the
        // oracle: conservation reject, no taxonomy churn. The V1 prefix
        // (pon_claim_rows_match_oracle_exact) still walks 2 rows.
        let text = pon_text();
        let mut arena = Vec::new();
        let game = frame_spans(&mut arena, &text, 7, 9).expect("frame");
        let verdict = gate_game(9, &game.events).expect("gate ok");
        assert!(verdict.is_sample());
        let ctx = GameCtx::wall_less(9);
        let mut ledger = Ledger::new();
        let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist = Vec::new();
        let mut sink = RowSink::new(&mut planes, &mut hist);
        let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("post-claim draw must reject");
        // Events: 0 start, 1 kyoku, 2 tsumo(0), 3 dahai(0), 4 pon(1), 5 tsumo(1).
        assert_eq!(err.event_idx, 5);
        assert_eq!(err.reason, WALK_TILE_CONSERVATION);
        assert_eq!(err.name(), "tile-conservation");
    }

    fn chi_tehais() -> Vec<Vec<&'static str>> {
        // Pool-exact tehais (global <=4 per type): seat0 discards 3p, seat1
        // chis 2p-3p-4p kamicha.
        vec![
            vec!["3p", "3p", "1m", "1m", "1m", "2m", "2m", "2m", "7s", "7s", "7s", "E", "E"],
            vec!["2p", "2p", "4p", "4p", "6m", "6m", "6m", "1p", "1p", "1p", "S", "S", "S"],
            vec!["8m", "8m", "8m", "9m", "9m", "9m", "2s", "2s", "2s", "6s", "6s", "6s", "W"],
            vec!["4s", "4s", "4s", "8s", "8s", "8s", "9s", "9s", "9s", "N", "N", "N", "F"],
        ]
    }

    fn chi_text() -> Vec<u8> {
        let held = chi_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "3p", false));
        lines.push(
            "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"2p\",\"4p\"]}"
                .to_string(),
        );
        lines.push(tsumo_line(1, "6p"));
        lines.push(dahai_line(1, "6p", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn chi_ext_post_claim_draw_is_conservation_reject() {
        // Same tripwire via chi (second non-kan claim kind): claimer tsumo
        // with no intervening discard rejects as tile-conservation.
        let text = chi_text();
        let mut arena = Vec::new();
        let game = frame_spans(&mut arena, &text, 7, 21).expect("frame");
        let verdict = gate_game(21, &game.events).expect("gate ok");
        assert!(verdict.is_sample());
        let ctx = GameCtx::wall_less(21);
        let mut ledger = Ledger::new();
        let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist = Vec::new();
        let mut sink = RowSink::new(&mut planes, &mut hist);
        let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("chi claimer draw must reject");
        assert_eq!(err.event_idx, 5);
        assert_eq!(err.reason, WALK_TILE_CONSERVATION);
    }

    fn chi_then_discard_text() -> Vec<u8> {
        // Claimant discards after chi (chi-twin of pon_then_discard): the
        // post-chi dahai row carries the remainder takes.
        let held = chi_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "3p", false));
        lines.push(
            "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"2p\",\"4p\"]}"
                .to_string(),
        );
        lines.push(dahai_line(1, "6m", false));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn chi_remainder_offers_skip_only_called_string() {
        // Walled: seat1 chis 2p-3p-4p holding a second 2p and 4p. The
        // leftovers render non-called strings, so the post-chi dahai still
        // offers them (the wall oracle withholds only the just-claimed
        // string: a scripted-wall probe of the live engine deals chi
        // consuming {44,48} and offers discard-45 on the post-chi turn).
        let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
        let w = {
            let digest: &str = &arc;
            walk_bytes(&chi_then_discard_text(), 23, Some((digest, std::sync::Arc::clone(&arc))))
        };
        assert_eq!(w.rows, 3);
        assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 2), 24);
        let legal = legal_of(&w, 2);
        assert!(legal.contains(&45), "2p remainder missing: {legal:?}");
        assert!(legal.contains(&53), "4p remainder missing: {legal:?}");
    }

    fn pon_then_discard_text() -> Vec<u8> {
        // Claimant discards before the next draw: tripwire disarmed, the
        // rotation walks (4 rows: dahai, pon, dahai, dahai).
        let held = custom_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "1m", false));
        lines.push(
            "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
                .to_string(),
        );
        lines.push(dahai_line(1, "6m", false));
        lines.push(tsumo_line(2, "7p"));
        lines.push(dahai_line(2, "2m", false));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn post_claim_intervening_discard_disarms_tripwire() {
        // Guard against over-rejection: pon -> claimer discard -> rotation
        // draw walks normally.
        let w = walk_bytes(&pon_then_discard_text(), 22, None);
        assert_eq!(w.rows, 4);
        let chosen: Vec<i64> = (0..4).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen[1], 4444);
    }

    #[test]
    fn kyushu_offer_on_qualifying_first_draw() {
        // 13 distinct terminals/honors + first draw: the abort offer rides
        // (live parity; id pinned against the live engine + action table).
        let te = vec![
            &["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"][..],
            &["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "6p"][..],
            &["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "7p"][..],
            &["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "8p"][..],
        ];
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "2m"));
        lines.push(dahai_line(0, "2m", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        let w = walk_bytes(out.as_bytes(), 75, None);
        assert_eq!(w.rows, 1);
        let legal = legal_of(&w, 0);
        assert!(legal.contains(&ABORT_NINE_TERMINALS_ID), "kyushu offer missing");
        assert_eq!(ABORT_NINE_TERMINALS_ID, 6790);
    }

    #[test]
    fn kyushu_absent_without_nine_yaochu() {
        // F1 golden game (simple hands, no yaochu mass): no row offers abort.
        let w = walk_bytes(&f1_text(), 76, None);
        assert!(w.rows > 0);
        let mut r = 0usize;
        while r < w.rows as usize {
            assert!(!legal_of(&w, r).contains(&ABORT_NINE_TERMINALS_ID));
            r += 1;
        }
    }

    fn ankan_rinshan_tehais() -> Vec<Vec<&'static str>> {
        // Pool-exact tehais: seat0 holds a 9m quad for the ankan; every
        // other type is collision-free globally.
        vec![
            vec!["9m", "9m", "9m", "9m", "1m", "1m", "1m", "2m", "2m", "2m", "3p", "3p", "5p"],
            vec!["4m", "4m", "4m", "6m", "6m", "6m", "7m", "7m", "7m", "1p", "1p", "1p", "E"],
            vec!["3m", "3m", "3m", "8m", "8m", "8m", "2p", "2p", "2p", "4p", "4p", "4p", "S"],
            vec!["6p", "6p", "6p", "7p", "7p", "7p", "8p", "8p", "8p", "9p", "9p", "9p", "N"],
        ]
    }

    fn ankan_rinshan_text() -> Vec<u8> {
        let held = ankan_rinshan_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "6s"));
        lines.push(
            "{\"type\":\"ankan\",\"actor\":0,\"consumed\":[\"9m\",\"9m\",\"9m\",\"9m\"]}".to_string(),
        );
        lines.push(tsumo_line(0, "7s"));
        lines.push(dahai_line(0, "7s", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn post_claim_kan_rinshan_still_accepted() {
        // Guard against over-rejection: ankan arms rinshan pending (never
        // the non-kan tripwire), so the replacement draw + discard walk.
        let w = walk_bytes(&ankan_rinshan_text(), 23, None);
        assert_eq!(w.rows, 2);
        let chosen: Vec<i64> = (0..2).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen[0], 6084);
        assert_eq!(w.ledger.meld_lens[0], 1);
    }
    fn tenpai_tehais() -> Vec<Vec<&'static str>> {
        vec![
            vec!["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5p"],
            vec!["6m", "6m", "6m", "6m", "7m", "7m", "7m", "7m", "1p", "1p", "1p", "2p", "5p"],
            vec!["8m", "8m", "8m", "8m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p"],
            vec!["1s", "1s", "1s", "1s", "2s", "2s", "2s", "2s", "E", "E", "E", "F", "F"],
        ]
    }

    fn tsumo_win_text() -> Vec<u8> {
        let held = tenpai_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        // Tenhou-real tsumo: no flag, no pai, target == actor.
        lines.push("{\"type\":\"hora\",\"actor\":0,\"target\":0,\"deltas\":[8000,-2000,-3000,-3000]}".to_string());
        lines.push("{\"type\":\"end_kyoku\"}".to_string());
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn declined_tsumo_win_still_offered() {
        // Winning draw declined (dahai instead of hora): the tsumo answer
        // still rides the draw row (engine offers wins taken or not).
        let held = tenpai_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "5p"));
        lines.push(dahai_line(0, "1m", false));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        let w = walk_bytes(out.as_bytes(), 14, None);
        assert_eq!(w.rows, 1);
        // Chosen is the 1m discard (take 0); tsumo on the drawn 5p (take 53).
        assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 0), 4);
        let legal = legal_of(&w, 0);
        assert!(legal.contains(&6707), "declined tsumo missing: {legal:?}");
    }

    #[test]
    fn tsumo_win_row_scores_and_shape() {
        let w = walk_bytes(&tsumo_win_text(), 11, None);
        assert_eq!(w.rows, 1);
        assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 0), 6707);
        assert_eq!(rd_i64(&w.planes[PLANE_ACTOR], 0), 0);
        assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 0), 1);
        // Pre-payoff observation: the hora row is the winning declaration,
        // so its scores plane reads the ledger BEFORE the win deltas apply
        // (post-payoff scores would leak the payout target into the input).
        let scores: Vec<i32> = (0..4).map(|s| rd_score(&w.planes[PLANE_SCORES], 0, s)).collect();
        assert_eq!(scores, [25000, 25000, 25000, 25000]);
        let legal = legal_of(&w, 0);
        assert_eq!(
            legal,
            [4, 8, 12, 16, 57, 193, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290, 329, 330, 6707]
        );
    }

    fn ron_win_text() -> Vec<u8> {
        let held = tenpai_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "6p"));
        lines.push(dahai_line(0, "6p", true));
        lines.push(tsumo_line(1, "7p"));
        lines.push(dahai_line(1, "5p", false));
        lines.push("{\"type\":\"hora\",\"actor\":0,\"target\":1,\"deltas\":[8000,-8000,0,0]}".to_string());
        lines.push("{\"type\":\"end_kyoku\"}".to_string());
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn ron_win_row_window_and_mask() {
        let w = walk_bytes(&ron_win_text(), 12, None);
        assert_eq!(w.rows, 3);
        // Tedashi 5p discards map to Discard ids (collapse 53 → 57).
        let chosen: Vec<i64> = (0..3).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen, [196, 57, 6406]);
        // Exact oracle legal sets (log_replay wall-less): discards + twin +
        // riichi candidates + ankan quads, chosen forced.
        assert_eq!(
            legal_of(&w, 0),
            [4, 8, 12, 16, 57, 60, 196, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290, 329, 332]
        );
        assert_eq!(legal_of(&w, 1), [24, 28, 40, 44, 57, 64, 200, 6081, 6082]);
        // Ron row: discard-response phase, turn actor = discarder, {pass,ron}.
        assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 2), 2);
        assert_eq!(legal_of(&w, 2), [0, 6406]);
        // The taken ron proves the window: call_window in the ron history.
        assert_eq!(hist_of(&w, 2), [0, 1, 2, 3, 4, 2, 4, 7]);
        // Concealed row0: triples + single 5p (draw excluded); drawn 56.
        let mut c0 = [0u8; 34];
        for t in 0..34 {
            c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
        }
        assert_eq!(&c0[0..14], &[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn hora_without_end_kyoku_quarantines() {
        // ron_win shape (hora straight into end_game): the wall oracle goes
        // terminal and rejects the trailing end (quarantine parity with
        // replay_expand's post-loop terminal check). Walled-only: the SIM
        // oracle walks the same shape.
        let held = tenpai_tehais();
        let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "6p"));
        lines.push(dahai_line(0, "6p", true));
        lines.push(tsumo_line(1, "7p"));
        lines.push(dahai_line(1, "5p", false));
        lines.push("{\"type\":\"hora\",\"actor\":0,\"target\":1,\"deltas\":[8000,-8000,0,0]}".to_string());
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        let text = out.into_bytes();
        let mut arena = Vec::new();
        let game = frame_spans(&mut arena, &text, 7, 99).expect("frame");
        let verdict = gate_game(99, &game.events).expect("gate ok");
        assert!(verdict.is_sample());
        let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
        let ctx = GameCtx::walled(99, &arc, Some(std::sync::Arc::clone(&arc)));
        let mut ledger = Ledger::new();
        let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist_lens = Vec::new();
        let mut sink = RowSink::new(&mut planes, &mut hist_lens);
        let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("must quarantine");
        assert_eq!(err.reason, WALK_PAST_TERMINAL);
    }

    fn riichi_text() -> Vec<u8> {
        let teanais: Vec<Vec<&str>> = vec![
            vec!["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5p"],
            vec!["6m", "6m", "6m", "6m", "7m", "7m", "7m", "7m", "1p", "1p", "1p", "1p", "2p"],
            vec!["8m", "8m", "8m", "8m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p"],
            vec!["1s", "1s", "1s", "1s", "2s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s"],
        ];
        let te: Vec<&[&str]> = teanais.iter().map(|v| &v[..]).collect();
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
        lines.push(tsumo_line(0, "6p"));
        lines.push("{\"type\":\"reach\",\"actor\":0}".to_string());
        lines.push(dahai_line(0, "6p", false));
        lines.push(tsumo_line(1, "S"));
        lines.push(dahai_line(1, "S", true));
        lines.push(tsumo_line(2, "W"));
        lines.push(dahai_line(2, "W", true));
        lines.push(tsumo_line(3, "N"));
        lines.push(dahai_line(3, "N", true));
        lines.push(tsumo_line(0, "7p"));
        lines.push(dahai_line(0, "7p", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn riichi_declaration_and_forced_pair() {
        // Full oracle vectors (wall-less log_replay): riichi declaration,
        // three rotation discards (ankan quads + a riichi candidate ride),
        // then the true-take forced pair.
        let w = walk_bytes(&riichi_text(), 13, None);
        assert_eq!(w.rows, 5);
        let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen, [332, 252, 256, 260, 200]);
        assert_eq!(
            legal_of(&w, 0),
            [4, 8, 12, 16, 57, 60, 196, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290, 329, 332]
        );
        assert_eq!(legal_of(&w, 1), [24, 28, 40, 44, 116, 252, 6081, 6082, 6085]);
        assert_eq!(legal_of(&w, 2), [32, 36, 44, 48, 120, 256, 392, 6083, 6084]);
        assert_eq!(
            legal_of(&w, 3),
            [76, 80, 84, 88, 124, 260, 352, 353, 354, 355, 396, 6094, 6095]
        );
        // Forced row offers exactly the true-take pair.
        assert_eq!(legal_of(&w, 4), [64, 200]);
        // Riichi stick moved at declaration.
        let s0: Vec<i32> = (0..4).map(|s| rd_score(&w.planes[PLANE_SCORES], 4, s)).collect();
        assert_eq!(s0, [24000, 25000, 25000, 25000]);
        assert_eq!(w.ledger.riichi[0], 1);
        // Forced-row history: no windows on the honor discards.
        assert_eq!(hist_of(&w, 4), [0, 1, 2, 3, 4, 2, 4, 2, 4, 2, 4, 2, 3]);
    }

    fn dora_text() -> Vec<u8> {
        let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&TEHAIS_F1, 0)];
        lines.push(tsumo_line(0, "5pr"));
        lines.push(dahai_line(0, "5pr", true));
        lines.push("{\"type\":\"dora\",\"dora_marker\":\"5p\"}".to_string());
        lines.push(tsumo_line(1, "5p"));
        lines.push(dahai_line(1, "5p", true));
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }

    #[test]
    fn dora_indicator_plane_reveals_in_order() {
        let w = walk_bytes(&dora_text(), 14, None);
        assert_eq!(w.rows, 2);
        for k in 0..5 {
            assert_eq!(rd_dora(&w.planes[PLANE_DORA], 0, k), -1);
        }
        assert_eq!(rd_dora(&w.planes[PLANE_DORA], 1, 0), 53);
        for k in 1..5 {
            assert_eq!(rd_dora(&w.planes[PLANE_DORA], 1, k), -1);
        }
    }

    #[test]
    fn rejects_are_closed_triples() {
        // Dahai before any kyoku: turn-order.
        let text = "{\"type\":\"start_game\"}\n{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n";
        let mut arena = Vec::new();
        let game = frame_spans(&mut arena, text.as_bytes(), 0, 77).expect("frame");
        let ctx = GameCtx::wall_less(77);
        let mut ledger = Ledger::new();
        let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist = Vec::new();
        let mut sink = RowSink::new(&mut planes, &mut hist);
        let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("reject");
        assert_eq!(err.game_idx, 77);
        assert_eq!(err.reason, WALK_TURN_ORDER);
        // Truncated game (no end_game): turn-order.
        let text2 = "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        let mut arena2 = Vec::new();
        let game2 = frame_spans(&mut arena2, text2.as_bytes(), 0, 78).expect("frame");
        let ctx2 = GameCtx::wall_less(78);
        let mut ledger2 = Ledger::new();
        let mut planes2: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist2 = Vec::new();
        let mut sink2 = RowSink::new(&mut planes2, &mut hist2);
        let err2 = walk_game(&ctx2, &game2.events, &mut ledger2, &mut sink2).expect_err("reject");
        assert_eq!(err2.reason, WALK_TURN_ORDER);
        // Unknown stacked kind (gate bypassed): unknown-event.
        let mut arena3 = Vec::new();
        let text3 = f1_text();
        let mut game3 = frame_spans(&mut arena3, &text3, 0, 79).expect("frame");
        game3.events[3].kind = KIND_OTHER;
        let ctx3 = GameCtx::wall_less(79);
        let mut ledger3 = Ledger::new();
        let mut planes3: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist3 = Vec::new();
        let mut sink3 = RowSink::new(&mut planes3, &mut hist3);
        let err3 = walk_game(&ctx3, &game3.events, &mut ledger3, &mut sink3).expect_err("reject");
        assert_eq!(err3.reason, WALK_UNKNOWN_EVENT);
        assert_eq!(err3.event_idx, 3);
        // Bare dora at walk level (gate bypassed): bare-dora.
        let mut arena4 = Vec::new();
        let mut game4 = frame_spans(&mut arena4, &text3, 0, 80).expect("frame");
        game4.events[2].kind = crate::ingest::KIND_DORA;
        game4.events[2].dora_span = None;
        let ctx4 = GameCtx::wall_less(80);
        let mut ledger4 = Ledger::new();
        let mut planes4: [Vec<u8>; N_WALK_PLANES] = Default::default();
        let mut hist4 = Vec::new();
        let mut sink4 = RowSink::new(&mut planes4, &mut hist4);
        let err4 = walk_game(&ctx4, &game4.events, &mut ledger4, &mut sink4).expect_err("reject");
        assert_eq!(err4.reason, WALK_BARE_DORA);
        // Reason names join the closed sink vocabulary (G5).
        assert_eq!(err3.name(), "unknown-event");
        assert_eq!(err4.name(), "bare-dora");
        let _ = WALK_TILE_CONSERVATION;
    }


    #[test]
    fn walk_is_bitwise_deterministic() {
        // G1 facet: two walks of one game produce identical planes.
        let text = f1_text();
        let a = walk_bytes(&text, 3, None);
        let b = walk_bytes(&text, 3, None);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.hist_lens, b.hist_lens);
        for p in 0..N_WALK_PLANES {
            assert_eq!(a.planes[p], b.planes[p]);
        }
    }
    #[test]
    fn win_shape_units() {
        // Four triples + pair wins; junk does not.
        assert!(win_shape_14(&[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 52, 52], 0));
        // Seven pairs wins; honor quad + six pairs does not (E cannot
        // sequence and the pairs cannot triple: no pair+4-meld split).
        assert!(win_shape_14(&[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24], 0));
        assert!(!win_shape_14(&[108, 108, 108, 108, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20], 0));
        // Open hand needs melds + pair from concealed.
        assert!(win_shape_14(&[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12], 1));
        assert!(!win_shape_14(&[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12], 2));
    }

    #[test]
    fn legal_ids_sorted_contract() {
        // Bad seat: empty set, never a panic.
        let mut l = Ledger::new();
        let (ids, len) = legal_ids_sorted(9, &mut l, None);
        assert_eq!(len, 0);
        assert!(ids.iter().all(|x| *x == 0));
    }
}
