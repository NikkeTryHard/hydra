use super::spans::{deltas_quad, skip_ws};
use crate::ledger::{WALK_TILE_CONSERVATION, WALK_TURN_ORDER};
use crate::tiles::TileCodec;

/// Low byte of a UTF-8 construction word (all call sites mask to <256).
#[inline]
fn utf8_byte(v: u32) -> u8 {
    // proof: UTF-8 byte words are masked (`& 0x3F`) or range-checked (`< 0x80`) to <256.
    #[allow(clippy::cast_possible_truncation)]
    let b = v as u8;
    b
}

// ---------------------------------------------------------------------------
// Kyoku boundary parses (serde; amortized, never per row).
// ---------------------------------------------------------------------------
pub(crate) struct KyokuHeader {
    pub(crate) oya: u8,
    pub(crate) bakaze: u8,
    pub(crate) scores: [i32; 4],
    pub(crate) tehais: [[u8; 13]; 4],
    pub(crate) honba: u8,
    pub(crate) kyotaku: u8,
    pub(crate) kyoku_no: u8,
}

/// Parse a `start_kyoku` span into validated header takes.
pub(crate) fn parse_header(span: &[u8]) -> Result<KyokuHeader, u8> {
    // Object gate (mirrors serde `as_object` + trailing-garbage reject):
    // a header span is exactly one `{...}` object.
    let p0 = skip_ws(span, 0);
    if p0 >= span.len() || span[p0] != b'{' {
        return Err(WALK_TURN_ORDER);
    }
    let mut pe = span.len();
    while pe > 0
        && (span[pe - 1] == b' '
            || span[pe - 1] == b'\t'
            || span[pe - 1] == b'\n'
            || span[pe - 1] == b'\r')
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
                            out[w] = utf8_byte(cp);
                            w += 1;
                        } else if cp < 0x800 {
                            if w + 2 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xC0 | (cp >> 6));
                            out[w + 1] = utf8_byte(0x80 | (cp & 0x3F));
                            w += 2;
                        } else if cp < 0x10000 {
                            if w + 3 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xE0 | (cp >> 12));
                            out[w + 1] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
                            out[w + 2] = utf8_byte(0x80 | (cp & 0x3F));
                            w += 3;
                        } else {
                            if w + 4 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xF0 | (cp >> 18));
                            out[w + 1] = utf8_byte(0x80 | ((cp >> 12) & 0x3F));
                            out[w + 2] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
                            out[w + 3] = utf8_byte(0x80 | (cp & 0x3F));
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
                (true, true, -(mag.cast_signed()), j)
            } else if mag == i64::MAX as u64 + 1 {
                (true, true, i64::MIN, j)
            } else {
                (false, false, 0, j)
            }
        } else if mag <= i64::MAX as u64 {
            (true, true, mag.cast_signed(), j)
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
            // proof: serde-parity wrapping cast (`as_i64().unwrap_or(0) as i32`); overflow wraps identically.
            #[allow(clippy::cast_possible_truncation)]
            {
                out[k] = if is_i64 { v as i32 } else { 0 };
            }
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
        ((hdr.oya.is_some() as u8) * HDR_OYA)
            | ((hdr.honba.is_some() as u8) * HDR_HONBA)
            | ((hdr.kyotaku.is_some() as u8) * HDR_KYOTAKU)
            | ((hdr.scores.is_some() as u8) * HDR_SCORES)
            | ((hdr.bakaze.is_some() as u8) * HDR_BAKAZE)
            | ((hdr.tehais.is_some() as u8) * HDR_TEHAIS)
            | ((hdr.kyoku.is_some() as u8) * HDR_KYOKU)
    );
    // proof: filtered to 0..=3, fits in u8 without sign loss.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let oya = hdr_i64(span, hdr.oya.ok_or(WALK_TURN_ORDER)?)
        .filter(|x| (0..=3).contains(x))
        .ok_or(WALK_TURN_ORDER)? as u8;
    // Last-wins preserved: honba then kyotaku, same order/rules as before.
    // Values retained (u8 range; fail closed beyond it).
    let mut hvals = [0u8; 3];
    for (hv, pos) in [
        hdr.honba.ok_or(WALK_TURN_ORDER)?,
        hdr.kyotaku.ok_or(WALK_TURN_ORDER)?,
        hdr.kyoku.ok_or(WALK_TURN_ORDER)?,
    ]
    .into_iter()
    .enumerate()
    {
        let v = hdr_i64(span, pos)
            .filter(|x| (0..=255).contains(x))
            .ok_or(WALK_TURN_ORDER)?;
        // proof: filtered to 0..=255, fits in u8 without sign loss.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            hvals[hv] = v as u8;
        }
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

pub(crate) struct RyukyokuBody {
    pub(crate) deltas: [i32; 4],
    pub(crate) reason: Option<RyukyokuReason>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum RyukyokuReason {
    DrawEnd,
    Abortive,
    Unmapped,
}

/// Parse a `ryukyoku` span: deltas quad (required) + reason class.
pub(crate) fn parse_ryukyoku(span: &[u8]) -> Result<RyukyokuBody, u8> {
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
                            out[w] = utf8_byte(cp);
                            w += 1;
                        } else if cp < 0x800 {
                            if w + 2 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xC0 | (cp >> 6));
                            out[w + 1] = utf8_byte(0x80 | (cp & 0x3F));
                            w += 2;
                        } else if cp < 0x10000 {
                            if w + 3 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xE0 | (cp >> 12));
                            out[w + 1] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
                            out[w + 2] = utf8_byte(0x80 | (cp & 0x3F));
                            w += 3;
                        } else {
                            if w + 4 > out.len() {
                                return None;
                            }
                            out[w] = utf8_byte(0xF0 | (cp >> 18));
                            out[w + 1] = utf8_byte(0x80 | ((cp >> 12) & 0x3F));
                            out[w + 2] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
                            out[w + 3] = utf8_byte(0x80 | (cp & 0x3F));
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
                    None => {
                        return Ok(RyukyokuBody {
                            deltas,
                            reason: Some(RyukyokuReason::Unmapped),
                        });
                    }
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
