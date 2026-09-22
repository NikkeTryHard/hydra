// ---------------------------------------------------------------------------
// Engine rules on u8 takes (V1 port; yaku-blind shape core verbatim).
// ---------------------------------------------------------------------------

/// Type counts over take ids (aka shares its type).
pub(crate) fn type_counts_of(ids: &[u8], counts: &mut [u8; 34]) {
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
pub(crate) fn melds_out_from(counts: &mut [u8; 34], need: u8, start: usize) -> bool {
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
    let suit = u8::try_from(i).unwrap_or(0) / 9;
    let pos = u8::try_from(i).unwrap_or(0) % 9;
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

pub(crate) fn melds_out(counts: &mut [u8; 34], need: u8) -> bool {
    melds_out_from(counts, need, 0)
}

/// Yaku-blind win over type counts + open meld count.
///
/// Seven-pairs (no quad-as-two-pairs) / thirteen-orphans when fully
/// concealed, else (4 − open) melds + a pair. Aka-insensitive (types).
pub(crate) fn win_shape_counts(counts: &[u8; 34], open_melds: usize) -> bool {
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
    let need = 4 - u8::try_from(open_melds).unwrap_or(0);
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
pub(crate) fn isort_u32(buf: &mut [u32; 32], len: usize) {
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
pub(crate) fn isort_u8(buf: &mut [u8], len: usize) {
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
pub(crate) fn tenpai_discards(hand14: &[u8], open_melds: usize, out: &mut [u8; 16]) -> usize {
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
pub(crate) fn chi_offered_types(counts: &[u8; 34], tile_type: u8) -> bool {
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
    if (2..=8).contains(&value) && has[(value - 1) as usize] && has[(value + 1) as usize] {
        return true;
    }
    if value <= 7 && has[(value + 1) as usize] && has[(value + 2) as usize] {
        return true;
    }
    false
}

/// Shape win with one extra take of `win_type` (norm pool-first copy).
pub(crate) fn shape_win_with(hand: &[u8], win_type: u8, open: usize) -> bool {
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

#[test]
fn quad_heavy_shanten1_yields_no_riichi_candidates() {
    // Three concealed quads + two singles (4m/8m/3p quads, 4p, drawn 5p)
    // is shanten 1 (engine-verified): no discard completes tenpai, so
    // tenpai_discards must report zero candidates (no riichi offers).
    let hand14: [u8; 14] = [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 48, 53];
    let mut out = [0u8; 16];
    assert_eq!(tenpai_discards(&hand14, 0, &mut out), 0);
}
