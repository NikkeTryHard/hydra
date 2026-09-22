use crate::ledger::{
    ACT_ANKAN, ACT_CHI, ACT_DAIMINKAN, ACT_DISCARD, ACT_KAKAN, ACT_PASS, ACT_PON,
    ACT_RIICHI_DISCARD, ACT_RON, ACT_TSUMO, ACT_TSUMOGIRI,
};

// ---------------------------------------------------------------------------
// Tunables (mirror the cold oracles; tests pin behavior, not values).
// ---------------------------------------------------------------------------

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
pub(crate) const YAOCHU_TYPES: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
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
pub(crate) const BASE_DISCARD: u32 = 4;
pub(crate) const BASE_TSUMOGIRI: u32 = 140;
pub(crate) const BASE_RIICHI: u32 = 276;
pub(crate) const BASE_CHI: u32 = 412;
pub(crate) const BASE_PON: u32 = 4444;
pub(crate) const BASE_DAIMINKAN: u32 = 5668;
pub(crate) const BASE_ANKAN: u32 = 6076;
pub(crate) const BASE_KAKAN: u32 = 6110;
pub(crate) const BASE_RON: u32 = 6246;
pub(crate) const BASE_TSUMO: u32 = 6654;

/// Per-type chi entry base: `64 × Σ patterns(earlier types)`, where
/// patterns by rank are 1/2/3/3/3/3/3/2/1 (one pattern × 16 copy combos).
pub(crate) const fn chi_type_base() -> [u16; 27] {
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
pub(crate) const CHI_TYPE_BASE: [u16; 27] = chi_type_base();

/// Chi pattern count for a rank 1..=9.
#[inline]
pub(crate) const fn chi_patterns(rank: u8) -> u8 {
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
        d => d.cast_signed(),
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
///
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
            Some(
                BASE_PON
                    + called as u32 * 9
                    + pair_idx? * 3
                    + u32::try_from(oi).unwrap_or(u32::MAX),
            )
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
            Some(BASE_DAIMINKAN + called as u32 * 3 + u32::try_from(oi).unwrap_or(u32::MAX))
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
