use super::kinds::{KIND_OTHER, NO_PAI, NO_SEAT, StackedEvent, boundary_class, kind_from_bytes};
use crate::gate::{self, GateReject};
use crate::tiles::TileCodec;

/// Raw value bounds `(start, end)` of the LAST top-level `"key"`
/// occurrence, if the line is a well-formed object. Span-location only
/// (values are interpreted from the serde parse); `None` on any structural
/// surprise — never panics.
pub(crate) fn raw_value_bounds(line: &[u8], key: &[u8]) -> Option<(usize, usize)> {
    let mut p = skip_ws(line, 0);
    if p >= line.len() || line[p] != b'{' {
        return None;
    }
    p += 1;
    let mut found = None;
    loop {
        p = skip_ws(line, p);
        if p < line.len() && line[p] == b'}' {
            return found;
        }
        if p >= line.len() || line[p] != b'"' {
            return None;
        }
        let (ks, ke, q) = string_bounds(line, p)?;
        p = skip_ws(line, q);
        if p >= line.len() || line[p] != b':' {
            return None;
        }
        let v = skip_ws(line, p + 1);
        let e = skip_value_at(line, v, 0)?;
        if &line[ks..ke] == key {
            found = Some((v, e));
        }
        p = skip_ws(line, e);
        if p >= line.len() {
            return None;
        }
        if line[p] == b',' {
            p += 1;
            continue;
        }
        if line[p] == b'}' {
            return found;
        }
        return None;
    }
}

// ---------------------------------------------------------------------------
// Fast per-line parser (hand-rolled; zero Value/String)
// ---------------------------------------------------------------------------

/// Fast line parse: integrated single-scan field extraction. Same verbs as
/// the baseline; `NaN`/`Infinity` rejected exactly like serde (pair parity).
pub(crate) fn parse_line_fast<'a>(
    line: &'a [u8],
    idx: u32,
    game_idx: u32,
    events: &mut Vec<StackedEvent<'a>>,
    wall: &mut Option<[u8; 136]>,
    starts: &mut u32,
    ends: &mut u32,
) -> Result<u8, GateReject> {
    if is_blank_line(line) {
        return Err(GateReject::blank_line(game_idx, idx));
    }
    let inv = || GateReject::invalid_json(game_idx, idx);
    let mut p = skip_ws(line, 0);
    if p >= line.len() || line[p] != b'{' {
        return Err(inv());
    }
    p += 1;
    let mut tbuf = [0u8; 64];
    let mut tlen: Option<usize> = None;
    let mut tover = false;
    let mut actor = NO_SEAT;
    let mut target = NO_SEAT;
    let mut pai = NO_PAI;
    let mut consumed = ([0u8; 4], 0u8);
    let mut tsumogiri = false;
    let mut dora_span: Option<&'a [u8]> = None;
    let mut w_wall: Option<(usize, usize)> = None;
    let mut w_tiles: Option<(usize, usize)> = None;
    let mut w_list: Option<(usize, usize)> = None;
    p = skip_ws(line, p);
    if p < line.len() && line[p] == b'}' {
        p += 1;
    } else {
        loop {
            p = skip_ws(line, p);
            if p >= line.len() || line[p] != b'"' {
                return Err(inv());
            }
            let (ks, ke, q) = string_bounds(line, p).ok_or_else(inv)?;
            let key = &line[ks..ke];
            p = skip_ws(line, q);
            if p >= line.len() || line[p] != b':' {
                return Err(inv());
            }
            p = skip_ws(line, p + 1);
            if key == b"type" {
                if p >= line.len() || line[p] != b'"' {
                    return Err(GateReject::bad_type(game_idx, idx));
                }
                let (vs, ve, q2) = string_bounds(line, p).ok_or_else(inv)?;
                // Last wins on duplicate keys.
                tover = false;
                tlen = None;
                match decode_into(&line[vs..ve], &mut tbuf) {
                    StrDec::Ok(n) => tlen = Some(n),
                    StrDec::Overflow => tover = true,
                    StrDec::Malformed => return Err(inv()),
                }
                p = q2;
            } else if key == b"actor" {
                let (s, q2) = parse_seat(line, p).ok_or_else(inv)?;
                actor = s;
                p = q2;
            } else if key == b"target" {
                let (s, q2) = parse_seat(line, p).ok_or_else(inv)?;
                target = s;
                p = q2;
            } else if key == b"pai" {
                let s = skip_ws(line, p);
                if s < line.len() && line[s] == b'n' {
                    p = match_lit(line, s, b"null").ok_or_else(inv)?;
                    pai = NO_PAI;
                } else if s < line.len() && line[s] == b'"' {
                    let (vs, ve, q2) = string_bounds(line, s).ok_or_else(inv)?;
                    let mut pbuf = [0u8; 64];
                    match decode_into(&line[vs..ve], &mut pbuf) {
                        StrDec::Ok(n) => {
                            pai = TileCodec::physical(&pbuf[..n])
                                .map_err(|_| GateReject::bad_tile(game_idx, idx))?;
                        }
                        _ => return Err(GateReject::bad_tile(game_idx, idx)),
                    }
                    p = q2;
                } else {
                    // Present non-string: mirror the legacy `from_value`
                    // failure → framing reject.
                    skip_value_at(line, p, 0).ok_or_else(inv)?;
                    return Err(GateReject::bad_tile(game_idx, idx));
                }
            } else if key == b"consumed" {
                let s = skip_ws(line, p);
                if s < line.len() && line[s] == b'n' {
                    p = match_lit(line, s, b"null").ok_or_else(inv)?;
                    consumed = ([0u8; 4], 0);
                } else if s < line.len() && line[s] == b'[' {
                    p = parse_consumed(line, s, idx, game_idx, &mut consumed)?;
                } else {
                    skip_value_at(line, p, 0).ok_or_else(inv)?;
                    return Err(GateReject::bad_tile(game_idx, idx));
                }
            } else if key == b"tsumogiri" {
                let s = skip_ws(line, p);
                if let Some(q2) = match_lit(line, s, b"true") {
                    tsumogiri = true;
                    p = q2;
                } else {
                    p = skip_value_at(line, p, 0).ok_or_else(inv)?;
                }
            } else if key == b"dora_marker" {
                let s = skip_ws(line, p);
                if s < line.len() && line[s] == b'"' {
                    let (vs, ve, q2) = string_bounds(line, s).ok_or_else(inv)?;
                    // `Some` only when present AND non-empty (hub contract).
                    dora_span = if ve - vs > 0 {
                        Some(&line[vs..ve])
                    } else {
                        None
                    };
                    p = q2;
                } else {
                    // Auxiliary cursor only: non-strings ignored (walk
                    // re-derives from `span` when it must fail closed).
                    p = skip_value_at(line, p, 0).ok_or_else(inv)?;
                }
            } else if key == b"wall" {
                p = capture_array(line, p, &mut w_wall).ok_or_else(inv)?;
            } else if key == b"wall_tiles" {
                p = capture_array(line, p, &mut w_tiles).ok_or_else(inv)?;
            } else if key == b"tiles" {
                p = capture_array(line, p, &mut w_list).ok_or_else(inv)?;
            } else {
                p = skip_value_at(line, p, 0).ok_or_else(inv)?;
            }
            p = skip_ws(line, p);
            if p >= line.len() {
                return Err(inv());
            }
            if line[p] == b',' {
                p += 1;
                continue;
            }
            if line[p] == b'}' {
                p += 1;
                break;
            }
            return Err(inv());
        }
    }
    p = skip_ws(line, p);
    if p != line.len() {
        return Err(inv());
    }
    let (kind, boundary) = if tover {
        (KIND_OTHER, 0)
    } else if let Some(n) = tlen {
        if n == 0 {
            return Err(GateReject::bad_type(game_idx, idx));
        }
        let ty = &tbuf[..n];
        (kind_from_bytes(ty), boundary_class(ty))
    } else {
        return Err(GateReject::missing_type(game_idx, idx));
    };
    match boundary {
        1 => *starts += 1,
        3 => *ends += 1,
        _ => {}
    }
    let cands = [w_wall, w_tiles, w_list];
    let line_wall_span = cands.iter().flatten().next().map(|c| &line[c.0..c.1]);
    if wall.is_none() {
        for c in cands.iter().flatten() {
            match try_wall(&line[c.0..c.1]) {
                WallTry::Found(ids) => {
                    if !gate::is_wall_perm136(&ids) {
                        return Err(GateReject::bad_wall(game_idx));
                    }
                    *wall = Some(ids);
                    break;
                }
                WallTry::Bad => return Err(GateReject::bad_wall(game_idx)),
                WallTry::Ignore => continue,
            }
        }
    }
    events.push(StackedEvent {
        kind,
        actor,
        target,
        pai,
        consumed,
        tsumogiri,
        span: line,
        dora_span,
        wall_span: line_wall_span,
    });
    Ok(boundary)
}

// ---------------------------------------------------------------------------
// JSON micro-parser (fast path only; strict, serde-agreeing)
// ---------------------------------------------------------------------------

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | 0x0B | 0x0C)
}

fn skip_ws(line: &[u8], mut p: usize) -> usize {
    while p < line.len() && is_ws(line[p]) {
        p += 1;
    }
    p
}

pub(crate) fn is_blank_line(line: &[u8]) -> bool {
    for c in line {
        if !is_ws(*c) {
            return false;
        }
    }
    true
}

/// Bounds of a JSON string value: `(inner_start, inner_end, after_close)`
/// for the opening quote at `p`. Escape-aware; `None` on any malformation
/// (unterminated, bad escape, raw control byte). `NaN`/`Infinity` are NOT
/// strings and never reach here.
fn string_bounds(line: &[u8], p: usize) -> Option<(usize, usize, usize)> {
    let mut i = p + 1;
    while i < line.len() {
        let c = line[i];
        if c == b'"' {
            return Some((p + 1, i, i + 1));
        }
        if c == b'\\' {
            i += 1;
            if i >= line.len() {
                return None;
            }
            let e = line[i];
            if e == b'u' {
                if i + 4 >= line.len() {
                    return None;
                }
                for k in 1..=4 {
                    if !line[i + k].is_ascii_hexdigit() {
                        return None;
                    }
                }
                i += 4;
            } else if !matches!(e, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                return None;
            }
        } else if c < 0x20 {
            return None;
        }
        i += 1;
    }
    None
}

pub(crate) enum StrDec {
    Ok(usize),
    Overflow,
    Malformed,
}

/// Decode escape sequences of a validated string interior into `out`.
pub(crate) fn decode_into(raw: &[u8], out: &mut [u8]) -> StrDec {
    let mut w = 0usize;
    let mut i = 0usize;
    while i < raw.len() {
        let mut b = raw[i];
        if b == b'\\' {
            i += 1;
            if i >= raw.len() {
                return StrDec::Malformed;
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
                    // `string_bounds` validated the 4 hexits.
                    let mut cp: u32 = 0;
                    for k in 1..=4 {
                        cp = cp * 16 + hex_val(raw[i + k]) as u32;
                    }
                    i += 4;
                    if (0xD800..0xDC00).contains(&cp) {
                        if raw.len() - i >= 7 && raw[i + 1] == b'\\' && raw[i + 2] == b'u' {
                            let mut lo: u32 = 0;
                            for k in 3..=6 {
                                lo = lo * 16 + hex_val(raw[i + k]) as u32;
                            }
                            if (0xDC00..0xE000).contains(&lo) {
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
                    let mut tmp = [0u8; 4];
                    let n = utf8_encode(cp, &mut tmp);
                    // HOT-PATH: dual-slice copy (tmp read, out[w] write with manual w);
                    // index form retains bound-check elision, no rows/s proof yet.
                    #[allow(
                        clippy::needless_range_loop,
                        reason = "index form retains bound-check elision on the hot copy"
                    )]
                    for k in 0..n {
                        if w >= out.len() {
                            return StrDec::Overflow;
                        }
                        out[w] = tmp[k];
                        w += 1;
                    }
                    i += 1;
                    continue;
                }
                _ => return StrDec::Malformed,
            }
        } else if b < 0x20 {
            return StrDec::Malformed;
        }
        if w >= out.len() {
            return StrDec::Overflow;
        }
        out[w] = b;
        w += 1;
        i += 1;
    }
    StrDec::Ok(w)
}

fn hex_val(h: u8) -> u8 {
    match h {
        b'0'..=b'9' => h - b'0',
        b'a'..=b'f' => h - b'a' + 10,
        b'A'..=b'F' => h - b'A' + 10,
        _ => 0,
    }
}

/// Low byte of a UTF-8 construction word (all call sites mask to <256).
#[inline]
fn utf8_byte(v: u32) -> u8 {
    // proof: UTF-8 byte words are masked (`& 0x3F`) or range-checked (`< 0x80`) to <256.
    #[allow(clippy::cast_possible_truncation)]
    let b = v as u8;
    b
}

/// Encode one scalar value as UTF-8; returns the byte count.
fn utf8_encode(cp: u32, out: &mut [u8; 4]) -> usize {
    if cp < 0x80 {
        out[0] = utf8_byte(cp);
        1
    } else if cp < 0x800 {
        out[0] = utf8_byte(0xC0 | (cp >> 6));
        out[1] = utf8_byte(0x80 | (cp & 0x3F));
        2
    } else if cp < 0x10000 {
        out[0] = utf8_byte(0xE0 | (cp >> 12));
        out[1] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
        out[2] = utf8_byte(0x80 | (cp & 0x3F));
        3
    } else {
        out[0] = utf8_byte(0xF0 | (cp >> 18));
        out[1] = utf8_byte(0x80 | ((cp >> 12) & 0x3F));
        out[2] = utf8_byte(0x80 | ((cp >> 6) & 0x3F));
        out[3] = utf8_byte(0x80 | (cp & 0x3F));
        4
    }
}

/// Skip one strict JSON value at `p` (nesting capped at 64; `NaN` and
/// `Infinity` rejected exactly like serde). Returns the end position.
fn skip_value_at(line: &[u8], p: usize, depth: usize) -> Option<usize> {
    if depth > 64 {
        return None;
    }
    let p = skip_ws(line, p);
    if p >= line.len() {
        return None;
    }
    match line[p] {
        b'"' => string_bounds(line, p).map(|(_, _, q)| q),
        b'{' => {
            let mut i = skip_ws(line, p + 1);
            if i < line.len() && line[i] == b'}' {
                return Some(i + 1);
            }
            loop {
                let k = skip_ws(line, i);
                if k >= line.len() || line[k] != b'"' {
                    return None;
                }
                let (_, _, q) = string_bounds(line, k)?;
                let c = skip_ws(line, q);
                if c >= line.len() || line[c] != b':' {
                    return None;
                }
                let v = skip_value_at(line, c + 1, depth + 1)?;
                let w = skip_ws(line, v);
                if w >= line.len() {
                    return None;
                }
                if line[w] == b',' {
                    i = w + 1;
                    continue;
                }
                if line[w] == b'}' {
                    return Some(w + 1);
                }
                return None;
            }
        }
        b'[' => {
            let mut i = skip_ws(line, p + 1);
            if i < line.len() && line[i] == b']' {
                return Some(i + 1);
            }
            loop {
                let v = skip_value_at(line, i, depth + 1)?;
                let w = skip_ws(line, v);
                if w >= line.len() {
                    return None;
                }
                if line[w] == b',' {
                    i = w + 1;
                    continue;
                }
                if line[w] == b']' {
                    return Some(w + 1);
                }
                return None;
            }
        }
        b't' => match_lit(line, p, b"true"),
        b'f' => match_lit(line, p, b"false"),
        b'n' => match_lit(line, p, b"null"),
        c if c == b'-' || c.is_ascii_digit() => skip_number(line, p),
        _ => None,
    }
}

/// Match a literal word with a delimiter guard (`truex` is not `true`).
fn match_lit(line: &[u8], p: usize, word: &[u8]) -> Option<usize> {
    let q = p + word.len();
    if line.len() >= q
        && &line[p..q] == word
        && (q >= line.len() || (!line[q].is_ascii_alphanumeric() && line[q] != b'_'))
    {
        return Some(q);
    }
    None
}

/// Skip a strict JSON number (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`).
fn skip_number(line: &[u8], mut p: usize) -> Option<usize> {
    if p < line.len() && line[p] == b'-' {
        p += 1;
    }
    if p >= line.len() {
        return None;
    }
    if line[p] == b'0' {
        p += 1;
    } else if line[p] >= b'1' && line[p] <= b'9' {
        while p < line.len() && line[p].is_ascii_digit() {
            p += 1;
        }
    } else {
        return None;
    }
    if p < line.len() && line[p] == b'.' {
        p += 1;
        if p >= line.len() || !line[p].is_ascii_digit() {
            return None;
        }
        while p < line.len() && line[p].is_ascii_digit() {
            p += 1;
        }
    }
    if p < line.len() && (line[p] == b'e' || line[p] == b'E') {
        p += 1;
        if p < line.len() && (line[p] == b'+' || line[p] == b'-') {
            p += 1;
        }
        if p >= line.len() || !line[p].is_ascii_digit() {
            return None;
        }
        while p < line.len() && line[p].is_ascii_digit() {
            p += 1;
        }
    }
    Some(p)
}

/// Parse a seat value: JSON int `0..=3`, else [`NO_SEAT`] (liberal — walk
/// fails closed, like the legacy `seat_of` use sites). `None` only when
/// the value itself is malformed JSON.
fn parse_seat(line: &[u8], p: usize) -> Option<(u8, usize)> {
    let s = skip_ws(line, p);
    let mut i = s;
    let neg = if i < line.len() && line[i] == b'-' {
        i += 1;
        true
    } else {
        false
    };
    if i >= line.len() || !line[i].is_ascii_digit() {
        let q = skip_value_at(line, p, 0)?;
        return Some((NO_SEAT, q));
    }
    let ds = i;
    while i < line.len() && line[i].is_ascii_digit() {
        i += 1;
    }
    if i < line.len() && (line[i] == b'.' || line[i] == b'e' || line[i] == b'E') {
        // Well-formed non-int (`1.0`, `1e2`): valid JSON, not a seat.
        let q = skip_value_at(line, p, 0)?;
        return Some((NO_SEAT, q));
    }
    if i - ds > 1 && line[ds] == b'0' {
        return None; // `007`: invalid JSON, serde/decode agree.
    }
    let mut v: u32 = 0;
    for b in &line[ds..i] {
        v = v.saturating_mul(10).saturating_add((*b - b'0') as u32);
    }
    // proof: checked v <= 3 above, fits in u8.
    #[allow(clippy::cast_possible_truncation)]
    let seat = if !neg && v <= 3 { v as u8 } else { NO_SEAT };
    Some((seat, i))
}

/// Parse a `consumed` array of tile strings into physical ids (≤4).
/// Anything malformed-as-JSON is `invalid_json`; anything well-formed but
/// unusable (non-string item, bad tile, >4 items, non-array non-null) is
/// `bad_tile`, mirroring the legacy `from_value` failure.
fn parse_consumed(
    line: &[u8],
    p: usize,
    idx: u32,
    game_idx: u32,
    out: &mut ([u8; 4], u8),
) -> Result<usize, GateReject> {
    let inv = || GateReject::invalid_json(game_idx, idx);
    let bad = || GateReject::bad_tile(game_idx, idx);
    let mut i = skip_ws(line, p + 1);
    if i < line.len() && line[i] == b']' {
        *out = ([0u8; 4], 0);
        return Ok(i + 1);
    }
    let mut arr = [0u8; 4];
    let mut n = 0u8;
    loop {
        let s = skip_ws(line, i);
        if s >= line.len() || line[s] != b'"' {
            skip_value_at(line, i, 0).ok_or_else(inv)?;
            return Err(bad());
        }
        let (vs, ve, q) = string_bounds(line, s).ok_or_else(inv)?;
        let mut pbuf = [0u8; 64];
        let id = match decode_into(&line[vs..ve], &mut pbuf) {
            StrDec::Ok(k) => TileCodec::physical(&pbuf[..k]).map_err(|_| bad())?,
            _ => return Err(bad()),
        };
        if n >= 4 {
            return Err(bad());
        }
        arr[n as usize] = id;
        n += 1;
        let w = skip_ws(line, q);
        if w >= line.len() {
            return Err(inv());
        }
        if line[w] == b',' {
            i = w + 1;
            continue;
        }
        if line[w] == b']' {
            *out = (arr, n);
            return Ok(w + 1);
        }
        return Err(inv());
    }
}

/// Capture an array value's inner bounds into `slot` (last wins; non-array
/// values clear the slot). The value must be well-formed (`None` else).
fn capture_array(line: &[u8], p: usize, slot: &mut Option<(usize, usize)>) -> Option<usize> {
    let q = skip_value_at(line, p, 0)?;
    let s = skip_ws(line, p);
    if s < line.len() && line[s] == b'[' {
        *slot = Some((s + 1, q - 1));
    } else {
        *slot = None;
    }
    Some(q)
}

enum WallTry {
    Ignore,
    Found([u8; 136]),
    Bad,
}

/// Classify a raw wall-array interior: `Ignore` unless it is exactly 136
/// ints (any float/bool/string/nested element, or any other length,
/// ignores the candidate — mirrors the legacy scan's `continue`);
/// `Bad` on any out-of-range id (negatives included — mirrors the legacy
/// range-fail `return Err`).
fn try_wall(inner: &[u8]) -> WallTry {
    let mut out = [0u8; 136];
    let mut n = 0usize;
    let mut oob = false;
    let mut p = skip_ws(inner, 0);
    if p >= inner.len() {
        return WallTry::Ignore;
    }
    loop {
        p = skip_ws(inner, p);
        if p >= inner.len() {
            return WallTry::Ignore; // defensive: validated upstream
        }
        let c = inner[p];
        if c == b'-' {
            oob = true;
            p += 1;
            let ds = p;
            while p < inner.len() && inner[p].is_ascii_digit() {
                p += 1;
            }
            if p == ds {
                return WallTry::Ignore; // defensive
            }
            if p < inner.len() && (inner[p] == b'.' || inner[p] == b'e' || inner[p] == b'E') {
                return WallTry::Ignore;
            }
        } else if c.is_ascii_digit() {
            let mut v: u32 = 0;
            let mut digits = 0u32;
            while p < inner.len() && inner[p].is_ascii_digit() {
                if digits < 3 {
                    v = v * 10 + (inner[p] - b'0') as u32;
                }
                digits += 1;
                p += 1;
            }
            if p < inner.len() && (inner[p] == b'.' || inner[p] == b'e' || inner[p] == b'E') {
                return WallTry::Ignore;
            }
            if digits > 3 || v > 135 {
                oob = true;
            } else if n < 136 {
                // proof: checked v <= 135 above, fits in u8.
                #[allow(clippy::cast_possible_truncation)]
                {
                    out[n] = v as u8;
                }
            }
        } else {
            return WallTry::Ignore;
        }
        n += 1;
        p = skip_ws(inner, p);
        if p >= inner.len() {
            break;
        }
        if inner[p] != b',' {
            return WallTry::Ignore; // defensive
        }
        p += 1;
    }
    if n == 136 {
        if oob {
            WallTry::Bad
        } else {
            WallTry::Found(out)
        }
    } else {
        WallTry::Ignore
    }
}
