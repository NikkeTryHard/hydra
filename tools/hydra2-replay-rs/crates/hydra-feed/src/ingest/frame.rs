use super::kinds::{
    FramedGame, KIND_START, NO_PAI, NO_SEAT, StackedEvent, boundary_class, kind_from_bytes,
};
use super::parse::{StrDec, decode_into, is_blank_line, parse_line_fast, raw_value_bounds};
use crate::gate::{self, GateReject};
use crate::tiles::TileCodec;
use serde_json::Value;
use std::io::Read as _;

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Frame one payload into stacked events — hand-rolled DEFAULT scanner.
///
/// DOUBLE-BUFFERED arena (flip: fill-one/clear-other) — the returned game
/// borrows `arena`; NEVER clear-while-borrowed. Framing failures surface as
/// [`GateReject`] via its `empty`/`missing_newline`/`blank_line`
/// /`utf8`/`decompress`/`invalid_json`/`missing_type`/`bad_type`/
/// `bad_boundary`/`start_end_count`/`bad_tile`/`bad_wall` constructors.
pub fn frame_spans<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    game_idx: u32,
) -> Result<FramedGame<'a>, GateReject> {
    fill_arena(arena, bytes, game_idx)?;
    let buf: &'a [u8] = arena;
    frame_lines(buf, (0, buf.len()), object_id, game_idx, parse_line_fast)
}

/// Frame one payload into stacked events — serde_json BASELINE reference.
///
/// Same contract and verdicts as [`frame_spans`], `memchr` line framing +
/// one `serde_json::Value` per line; serde owns ALL number parsing (seats,
/// wall ints). Kept for parity testing, never the hot path.
pub fn frame_spans_serde<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    game_idx: u32,
) -> Result<FramedGame<'a>, GateReject> {
    fill_arena(arena, bytes, game_idx)?;
    let buf: &'a [u8] = arena;
    frame_lines(
        buf,
        (0, buf.len()),
        object_id,
        game_idx,
        parse_line_baseline,
    )
}

/// Frame one payload into stacked events — alias of the DEFAULT scanner.
///
/// Same body as [`frame_spans`] (zero `Value`/`String`); retained so
/// existing callers keep compiling unchanged.
pub fn frame_spans_fast<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    game_idx: u32,
) -> Result<FramedGame<'a>, GateReject> {
    fill_arena(arena, bytes, game_idx)?;
    let buf: &'a [u8] = arena;
    frame_lines(buf, (0, buf.len()), object_id, game_idx, parse_line_fast)
}

/// Per-line parser plug for [`frame_lines`]: the baseline or the fast scan.
/// Both parsers share this exact shape, so one generic core serves every
/// entry point with zero-cost dispatch.
type LineParser<'a> = fn(
    &'a [u8],
    u32,
    u32,
    &mut Vec<StackedEvent<'a>>,
    &mut Option<[u8; 136]>,
    &mut u32,
    &mut u32,
) -> Result<u8, GateReject>;

/// Frame one game's whole lines over `buf[range]` (range covers complete
/// lines; the last byte must be `b'\n'`). Shared core behind [`frame_spans`],
/// [`frame_spans_serde`], [`frame_spans_fast`], [`frame_file`],
/// [`frame_file_serde`] and [`frame_file_fast`]: verdicts
/// identical to the original single-game body on every input.
fn frame_lines<'a>(
    buf: &'a [u8],
    range: (usize, usize),
    object_id: u32,
    game_idx: u32,
    parse: LineParser<'a>,
) -> Result<FramedGame<'a>, GateReject> {
    // No-panic-by-construction: out-of-range can only come from a caller
    // bug (both entry points build ranges from memchr splits); map it to
    // the closed empty verdict rather than indexing.
    let chunk = buf.get(range.0..range.1).unwrap_or(&[]);
    if chunk.is_empty() {
        return Err(GateReject::empty(game_idx));
    }
    if chunk[chunk.len() - 1] != b'\n' {
        return Err(GateReject::missing_newline(game_idx));
    }
    let body = &chunk[..chunk.len() - 1];
    let mut events: Vec<StackedEvent<'a>> = Vec::new();
    let mut starts = 0u32;
    let mut ends = 0u32;
    let mut wall: Option<[u8; 136]> = None;
    let mut first_boundary = 0u8;
    let mut idx = 0u32;
    let mut line_start = 0usize;
    for nl in memchr::memchr_iter(b'\n', body) {
        let b = parse(
            &body[line_start..nl],
            idx,
            game_idx,
            &mut events,
            &mut wall,
            &mut starts,
            &mut ends,
        )?;
        if idx == 0 {
            first_boundary = b;
        }
        idx += 1;
        line_start = nl + 1;
    }
    let b = parse(
        &body[line_start..],
        idx,
        game_idx,
        &mut events,
        &mut wall,
        &mut starts,
        &mut ends,
    )?;
    if idx == 0 {
        first_boundary = b;
    }
    let last_boundary = b;
    finish_game(
        game_idx,
        object_id,
        events,
        first_boundary,
        last_boundary,
        starts,
        ends,
        wall,
    )
}

/// True iff the line carries a start-alias `type`. Splitter predicate
/// shared by all file paths; structural only (values still parse per-path
/// downstream, so serde-owns-numbers on the baseline path is unaffected).
/// Malformed lines are never starts: they fail inside their own chunk,
/// keeping quarantine per-game.
fn is_start_line(line: &[u8]) -> bool {
    let (v, e) = match raw_value_bounds(line, b"type") {
        Some(b) => b,
        None => return false,
    };
    if e - v < 2 || line[v] != b'"' {
        return false;
    }
    let mut tbuf = [0u8; 64];
    match decode_into(&line[v + 1..e - 1], &mut tbuf) {
        StrDec::Ok(n) => kind_from_bytes(&tbuf[..n]) == KIND_START,
        _ => false,
    }
}

/// Split a filled buffer into per-game whole-line ranges at start-marker
/// lines. Every line lands in exactly one range: leading pre-start lines
/// form their own (failing) chunk; trailing lines attach to the last game.
/// Ranges never overlap and never slice a line (split points are always at
/// line starts), so chunk verdicts equal whole-file-independent framing.
fn split_game_ranges(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    if buf.is_empty() {
        return ranges;
    }
    let mut game_start = 0usize;
    let mut line_start = 0usize;
    // A start line opens a new chunk only when the current chunk already
    // holds a line (`line_start > game_start`); the start line itself
    // always belongs to the chunk it opens.
    for nl in memchr::memchr_iter(b'\n', buf) {
        if line_start > game_start && is_start_line(&buf[line_start..nl]) {
            ranges.push((game_start, line_start));
            game_start = line_start;
        }
        line_start = nl + 1;
    }
    if line_start < buf.len() && line_start > game_start && is_start_line(&buf[line_start..]) {
        ranges.push((game_start, line_start));
        game_start = line_start;
    }
    ranges.push((game_start, buf.len()));
    ranges
}

/// Frame a whole file (single `.jsonl` game, F8-style multi-game packs,
/// tar-bundled concatenations) — hand-rolled DEFAULT scanner. Returns the
/// per-game results in file order plus the staged-game count.
///
/// Keying contract (bridge `game_seq`): sub-game `k` (0-based file position)
/// emits `game_idx = base_game_idx.saturating_add(k)` — globally unique
/// across files when the caller advances its base by the previous file's
/// staged count. Rejects occupy keyspace too (every chunk, ok or err,
/// consumes exactly one index), so the staged count always equals the
/// results length; whole-file failures (decompress/utf8/empty) report a
/// single entry keyed at `base_game_idx`. Each chunk is framed by the
/// unchanged single-game core, so single-game files take the identical path
/// (one chunk, no special case). Games fail independently: a bad game
/// quarantines via its own `Err` while siblings still sample — feed each
/// `Ok` game to `gate_game` unchanged downstream.
pub fn frame_file<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    base_game_idx: u32,
) -> (Vec<Result<FramedGame<'a>, GateReject>>, u32) {
    frame_file_with(arena, bytes, object_id, base_game_idx, parse_line_fast)
}

/// Frame a whole file — serde_json BASELINE reference. Same contract and
/// verdicts as [`frame_file`]. Kept for parity testing, never the hot path.
pub fn frame_file_serde<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    base_game_idx: u32,
) -> (Vec<Result<FramedGame<'a>, GateReject>>, u32) {
    frame_file_with(arena, bytes, object_id, base_game_idx, parse_line_baseline)
}

/// Frame a whole file — alias of the DEFAULT scanner. Same contract and
/// verdicts as [`frame_file`], zero `Value`/`String`; retained so existing
/// callers keep compiling unchanged.
pub fn frame_file_fast<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    base_game_idx: u32,
) -> (Vec<Result<FramedGame<'a>, GateReject>>, u32) {
    frame_file_with(arena, bytes, object_id, base_game_idx, parse_line_fast)
}

/// File-level driver shared by both paths: fill once (the whole file
/// borrows one arena side — all returned games borrow it, so the caller
/// must not touch the arena until every game is consumed), split at start
/// markers, frame each chunk independently.
fn frame_file_with<'a>(
    arena: &'a mut Vec<u8>,
    bytes: &[u8],
    object_id: u32,
    base_game_idx: u32,
    parse: LineParser<'a>,
) -> (Vec<Result<FramedGame<'a>, GateReject>>, u32) {
    if let Err(e) = fill_arena(arena, bytes, base_game_idx) {
        return (vec![Err(e)], 1);
    }
    let buf: &'a [u8] = arena;
    if buf.is_empty() {
        return (vec![Err(GateReject::empty(base_game_idx))], 1);
    }
    let mut out: Vec<Result<FramedGame<'a>, GateReject>> = Vec::new();
    for (k, range) in split_game_ranges(buf).into_iter().enumerate() {
        let gi = base_game_idx.saturating_add(k as u32);
        out.push(frame_lines(buf, range, object_id, gi, parse));
    }
    let staged = out.len() as u32;
    (out, staged)
}

/// Split a whole file into per-game byte ranges over the filled arena.
///
/// Staging seam for file-level commit (FillBuilder owns the `stage_file`
/// step in `fill.rs`; bridge owns base/count wiring): fills `arena` once
/// (zstd/gzip transparent, same as [`frame_file`]) and returns one
/// `(chunk_start, chunk_end)` offset pair per game in file order. Offsets
/// are plain copies — no borrow is retained — so the caller slices
/// `arena[s..e]` per chunk afterwards (usually copying into an owned
/// per-game buffer for [`stage_one`]-style frame+gate+walk).
/// File-level failures (decompress/utf8/empty) return a single `Err` keyed
/// at `base_game_idx`, mirroring [`frame_file`]'s whole-file entries.
pub fn file_chunk_ranges(
    arena: &mut Vec<u8>,
    bytes: &[u8],
    base_game_idx: u32,
) -> Result<Vec<(usize, usize)>, GateReject> {
    fill_arena(arena, bytes, base_game_idx)?;
    if arena.is_empty() {
        return Err(GateReject::empty(base_game_idx));
    }
    Ok(split_game_ranges(arena))
}
// ---------------------------------------------------------------------------
// Shared: arena fill, game assembly, keys
// ---------------------------------------------------------------------------

/// Fill the inactive arena side: transparent zstd/gzip decode (streamed
/// into the reused buffer — never a fresh whole-file alloc) or a plain
/// copy; then fail closed on non-UTF-8 (mirrors `decode.py` + the legacy
/// `read_input_text`, minus its fresh `Vec` per file).
fn fill_arena(arena: &mut Vec<u8>, bytes: &[u8], game_idx: u32) -> Result<(), GateReject> {
    arena.clear();
    if bytes.len() >= 4
        && bytes[0] == 0x28
        && bytes[1] == 0xB5
        && bytes[2] == 0x2F
        && bytes[3] == 0xFD
    {
        let mut dec = zstd::Decoder::new(bytes).map_err(|_| GateReject::decompress(game_idx))?;
        dec.read_to_end(arena)
            .map_err(|_| GateReject::decompress(game_idx))?;
    } else if bytes.len() >= 2 && bytes[0] == 0x1F && bytes[1] == 0x8B {
        let mut dec = flate2::read::GzDecoder::new(bytes);
        dec.read_to_end(arena)
            .map_err(|_| GateReject::decompress(game_idx))?;
    } else {
        arena.extend_from_slice(bytes);
    }
    if std::str::from_utf8(arena).is_err() {
        return Err(GateReject::utf8(game_idx));
    }
    Ok(())
}

/// Boundary + exactly-one assembly shared by both paths.
fn finish_game<'a>(
    game_idx: u32,
    object_id: u32,
    events: Vec<StackedEvent<'a>>,
    first_boundary: u8,
    last_boundary: u8,
    starts: u32,
    ends: u32,
    wall: Option<[u8; 136]>,
) -> Result<FramedGame<'a>, GateReject> {
    if first_boundary != 1 && first_boundary != 2 {
        return Err(GateReject::bad_boundary(game_idx, 0));
    }
    let last_idx = events.len() as u32 - 1;
    if last_boundary != 3 && last_boundary != 4 {
        return Err(GateReject::bad_boundary(game_idx, last_idx));
    }
    if starts != 1 || ends != 1 {
        return Err(GateReject::start_end_count(game_idx));
    }
    Ok(FramedGame {
        game_idx,
        object_id,
        events,
        wall,
        game_key: mix_key(game_idx, object_id),
    })
}

/// Opaque game key: splitmix64 over `(object_id, game_idx)`. Deterministic,
/// no alloc, never a display id (string ids live cold-side only).
fn mix_key(game_idx: u32, object_id: u32) -> u64 {
    let mut z = ((object_id as u64) << 32 | game_idx as u64).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Baseline per-line parser (serde_json owns all number parsing)
// ---------------------------------------------------------------------------

/// Baseline line parse: one `serde_json::Value`, fields copied out as u8
/// codes/ids, spans borrowed from the line. Returns the boundary class.
fn parse_line_baseline<'a>(
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
    let value: Value =
        serde_json::from_slice(line).map_err(|_| GateReject::invalid_json(game_idx, idx))?;
    let obj = value
        .as_object()
        .ok_or_else(|| GateReject::invalid_json(game_idx, idx))?;
    let ty = match obj.get("type") {
        None => return Err(GateReject::missing_type(game_idx, idx)),
        Some(Value::String(s)) => {
            if s.is_empty() {
                return Err(GateReject::bad_type(game_idx, idx));
            }
            s
        }
        Some(_) => return Err(GateReject::bad_type(game_idx, idx)),
    };
    let ty_bytes = ty.as_bytes();
    let kind = kind_from_bytes(ty_bytes);
    let boundary = boundary_class(ty_bytes);
    match boundary {
        1 => *starts += 1,
        3 => *ends += 1,
        _ => {}
    }
    let actor = seat_of(obj.get("actor"));
    let target = seat_of(obj.get("target"));
    let pai = match obj.get("pai") {
        None | Some(Value::Null) => NO_PAI,
        Some(Value::String(s)) => {
            TileCodec::physical(s.as_bytes()).map_err(|_| GateReject::bad_tile(game_idx, idx))?
        }
        Some(_) => return Err(GateReject::bad_tile(game_idx, idx)),
    };
    let consumed = match obj.get("consumed") {
        None | Some(Value::Null) => ([0u8; 4], 0u8),
        Some(Value::Array(items)) => {
            if items.len() > 4 {
                return Err(GateReject::bad_tile(game_idx, idx));
            }
            let mut arr = [0u8; 4];
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => {
                        arr[i] = TileCodec::physical(s.as_bytes())
                            .map_err(|_| GateReject::bad_tile(game_idx, idx))?;
                    }
                    _ => return Err(GateReject::bad_tile(game_idx, idx)),
                }
            }
            (arr, items.len() as u8)
        }
        Some(_) => return Err(GateReject::bad_tile(game_idx, idx)),
    };
    let tsumogiri = obj
        .get("tsumogiri")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let dora_span = match raw_value_bounds(line, b"dora_marker") {
        Some((s, e)) if line[s] == b'"' && e - s > 2 => Some(&line[s + 1..e - 1]),
        _ => None,
    };
    // Per-line wall span: first array-valued key in wall priority order.
    let mut line_wall_span: Option<&'a [u8]> = None;
    for key in [
        b"wall".as_slice(),
        b"wall_tiles".as_slice(),
        b"tiles".as_slice(),
    ] {
        if let Some((s, e)) = raw_value_bounds(line, key)
            && line[s] == b'['
        {
            line_wall_span = Some(&line[s + 1..e - 1]);
            break;
        }
    }
    // Global wall bind: first event carrying a 136-int list under
    // wall > wall_tiles > tiles (mirrors `decode_game_object`); 136-int
    // lists that are not a 0..135 permutation fail closed (WallSchedule).
    if wall.is_none() {
        for key in ["wall", "wall_tiles", "tiles"] {
            let candidate = obj.get(key).and_then(|v| v.as_array());
            let arr = match candidate {
                Some(a) => a,
                None => continue,
            };
            if arr.len() != 136 {
                continue;
            }
            if !arr.iter().all(|x| x.is_i64() || x.is_u64()) {
                continue;
            }
            let mut ids = [0u8; 136];
            let mut oob = false;
            for (i, x) in arr.iter().enumerate() {
                // `is_*` checked above; negatives fail `as_u64` → sentinel → oob.
                let n = x.as_u64().unwrap_or(u64::MAX);
                if n > 135 {
                    oob = true;
                    break;
                }
                ids[i] = n as u8;
            }
            if oob || !gate::is_wall_perm136(&ids) {
                return Err(GateReject::bad_wall(game_idx));
            }
            *wall = Some(ids);
            break;
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

/// Baseline seat: JSON u64 `0..=3`, else [`NO_SEAT`] (liberal — walk fails
/// closed on unexpected seats, exactly like the legacy `seat_of` use sites).
fn seat_of(value: Option<&Value>) -> u8 {
    match value.and_then(|v| v.as_u64()) {
        Some(s) if s <= 3 => s as u8,
        _ => NO_SEAT,
    }
}
