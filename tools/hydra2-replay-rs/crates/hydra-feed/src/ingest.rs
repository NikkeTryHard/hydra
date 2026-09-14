//! S1 ingest — one-pass span framing (`feed::ingest`).
//!
//! §6.1 shapes + kind-LUT constants: the cross-crate contract `feed::gate`
//! compiles and tests against verbatim (see contract block below).
//!
//! Cross-crate contract:
//! - kind-LUT (u8): 0=Start, 1=End, 2=Dahai, 3=Chi, 4=Pon, 5=Daiminkan,
//!   6=Ankan, 7=Kakan, 8=Hora, 9=Dora, 10=Reach, 11=ReachAccepted, 12=Tsumo,
//!   13=StartKyoku, 14=EndKyoku, 15=Ryukyoku, 16=TransparentOther (reserved,
//!   nothing maps here today), 17=Other.
//! - `should_sample` set = `{2..=8}` (ROW_TYPES; owned by `crate::gate`).
//! - Unknown type strings → 17=Other (gate rejects Other→InvalidData).
//! - Bare `start`/`end` aliases map to KIND_START/KIND_END (gate span-scans
//!   the canonical 3-spellings for exact oracle count agreement).
//! - `span` = raw line bytes; `dora_span` = None when the marker is absent
//!   or empty, else the marker bytes; actor/target = `0xFF` unless the JSON
//!   value is a u64 `0..=3`.
//!
//! Two paths:
//! - [`frame_spans`] (DEFAULT): hand-rolled span parser, zero
//!   `Value`/`String`: one `memchr` pass, kind-LUT, tile-LUT, inline wall
//!   capture. Verdict-identical to the serde baseline on every input
//!   (71 feed tests + 22 parity + F1 vectors + counts).
//! - [`frame_spans_serde`] (BASELINE): serde_json reference. `memchr` line
//!   framing + one `serde_json::Value` per line; serde owns ALL number
//!   parsing (seats, wall ints). Replaces `mjai_event::frame_events`
//!   (double decode) and `stream::parse_game`'s wall re-scan (third parse)
//!   verdict-identical. Kept for parity testing, never the hot path.
//! - [`frame_spans_fast`]: retained alias of the DEFAULT fast scanner
//!   (same body as [`frame_spans`]); callers may use either name.
//!
//! Both paths enforce the full `decode.py::decode_game_object` line rules
//! (boundary + canonical exactly-one counts) plus the `0..135` wall
//! permutation (`stream::WallSchedule` contract), and agree byte-identical
//! accept/reject on the pinned fixtures (see `baseline_and_fast_agree`).
//!
//! File level ([`frame_file`] (DEFAULT, fast) / [`frame_file_serde`]
//! (BASELINE, serde) / [`frame_file_fast`] (alias of the default)):
//! whole files (single `.jsonl` games, F8-style 22-game packs, tar-bundled
//! concatenations) split at start markers into per-game chunks, each framed
//! by the unchanged single-game core. One `Vec` entry per game in file
//! order plus the staged-game count; sub-game `k` keys
//! `base_game_idx.saturating_add(k)` (rejects occupy keyspace too), so the
//! bridge advances `game_seq` by the staged count with no collisions.
//! Games fail independently (mixed valid/invalid quarantines only
//! the bad chunk). Per-game gate downstream is unchanged.
//!
//! Double-buffer discipline (caller-owned): keep two arenas, frame game N
//! into the inactive side while game N-1's `FramedGame` still borrows the
//! other. NEVER `clear` (or re-fill) an arena while its `FramedGame` is
//! alive — both entry points clear the side they are given. File level:
//! the whole file borrows ONE side, so the caller must not touch the arena
//! until every returned game is consumed.
//!
//! Allocation audit: no `String`/`Vec`/`format!` past spans on either path
//! (`Vec<StackedEvent>` event storage is the only heap use, sized by the
//! line count). `serde_json::Value` per-line temporaries exist on the
//! BASELINE path only (dropped before return; never retained).

use crate::gate::{self, GateReject};
use crate::tiles::TileCodec;
use serde_json::Value;
use std::io::Read as _;

/// Event kind: start aliases (`start_game`/`startGame`/`game_start`/`start`).
pub const KIND_START: u8 = 0;
/// Event kind: end aliases (`end_game`/`endGame`/`game_end`/`end`).
pub const KIND_END: u8 = 1;
/// Event kind: `dahai` (sampled).
pub const KIND_DAHAI: u8 = 2;
/// Event kind: `chi` (sampled).
pub const KIND_CHI: u8 = 3;
/// Event kind: `pon` (sampled).
pub const KIND_PON: u8 = 4;
/// Event kind: `daiminkan` (sampled).
pub const KIND_DAIMINKAN: u8 = 5;
/// Event kind: `ankan` (sampled).
pub const KIND_ANKAN: u8 = 6;
/// Event kind: `kakan` (sampled).
pub const KIND_KAKAN: u8 = 7;
/// Event kind: `hora` (sampled).
pub const KIND_HORA: u8 = 8;
/// Event kind: `dora`.
pub const KIND_DORA: u8 = 9;
/// Event kind: `reach` (declaration; collapses with its dahai, never a row).
pub const KIND_REACH: u8 = 10;
/// Event kind: `reach_accepted` (transparent for ron-pending).
pub const KIND_REACH_ACCEPTED: u8 = 11;
/// Event kind: `tsumo` (draw).
pub const KIND_TSUMO: u8 = 12;
/// Event kind: `start_kyoku` (kyoku boundary; resets ron-pending).
pub const KIND_START_KYOKU: u8 = 13;
/// Event kind: `end_kyoku`.
pub const KIND_END_KYOKU: u8 = 14;
/// Event kind: `ryukyoku`.
pub const KIND_RYUKYOKU: u8 = 15;
/// Event kind: reserved transparent extension (nothing maps here today).
pub const KIND_TRANSPARENT_OTHER: u8 = 16;
/// Event kind: unknown / unusable type (gate rejects → InvalidData).
pub const KIND_OTHER: u8 = 17;

/// Seat sentinel: field absent or out of range (`0xFF none`, plan §6.1).
pub const NO_SEAT: u8 = 0xFF;
/// Tile sentinel: no tile (`0xFF none`, plan §6.1).
pub const NO_PAI: u8 = 0xFF;

/// Framing vocabulary: game-boundary aliases (mirrors `decode.py`).
pub const START_TYPES: [&str; 4] = ["start_game", "startGame", "game_start", "start"];
/// Framing vocabulary: game-end aliases (mirrors `decode.py`).
pub const END_TYPES: [&str; 4] = ["end_game", "endGame", "game_end", "end"];
/// Non-decision events: ordering/round tracking only, never rows.
pub const SKIP_TYPES: [&str; 10] = [
    "start_kyoku",
    "tsumo",
    "dora",
    "reach_accepted",
    "ryukyoku",
    "end_kyoku",
    "start_game",
    "startGame",
    "game_start",
    "start",
];
/// One row per occurrence (`reach` collapses with its following `dahai`).
pub const ROW_TYPES: [&str; 7] = [
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "hora",
];
/// Row kinds that claim the live discard offer.
pub const CLAIM_TYPES: [&str; 3] = ["chi", "pon", "daiminkan"];
/// Log event kinds skipped when looking behind/ahead through a kyoku.
pub const TRANSPARENT_KINDS: [&str; 2] = ["dora", "reach_accepted"];

/// Interned kind code for a decoded `type` string (unknown → [`KIND_OTHER`]).
pub fn kind_from_bytes(ty: &[u8]) -> u8 {
    if ty == b"start_game" || ty == b"startGame" || ty == b"game_start" || ty == b"start" {
        KIND_START
    } else if ty == b"end_game" || ty == b"endGame" || ty == b"game_end" || ty == b"end" {
        KIND_END
    } else if ty == b"dahai" {
        KIND_DAHAI
    } else if ty == b"chi" {
        KIND_CHI
    } else if ty == b"pon" {
        KIND_PON
    } else if ty == b"daiminkan" {
        KIND_DAIMINKAN
    } else if ty == b"ankan" {
        KIND_ANKAN
    } else if ty == b"kakan" {
        KIND_KAKAN
    } else if ty == b"hora" {
        KIND_HORA
    } else if ty == b"dora" {
        KIND_DORA
    } else if ty == b"reach" {
        KIND_REACH
    } else if ty == b"reach_accepted" {
        KIND_REACH_ACCEPTED
    } else if ty == b"tsumo" {
        KIND_TSUMO
    } else if ty == b"start_kyoku" {
        KIND_START_KYOKU
    } else if ty == b"end_kyoku" {
        KIND_END_KYOKU
    } else if ty == b"ryukyoku" {
        KIND_RYUKYOKU
    } else {
        KIND_OTHER
    }
}

/// Boundary class: 1 = canonical start, 2 = bare `start` alias,
/// 3 = canonical end, 4 = bare `end` alias, 0 = not a boundary.
/// The exactly-one rule counts canonical spellings only (1/3), mirroring
/// both `decode.py` and the legacy `frame_events`.
fn boundary_class(ty: &[u8]) -> u8 {
    if ty == b"start_game" || ty == b"startGame" || ty == b"game_start" {
        1
    } else if ty == b"start" {
        2
    } else if ty == b"end_game" || ty == b"endGame" || ty == b"game_end" {
        3
    } else if ty == b"end" {
        4
    } else {
        0
    }
}

/// One stacked MJAI event over arena spans (plan §6.1, verbatim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackedEvent<'a> {
    /// Kind code (`KIND_*`).
    pub kind: u8,
    /// Actor seat `0..=3`, else [`NO_SEAT`].
    pub actor: u8,
    /// Target seat `0..=3`, else [`NO_SEAT`].
    pub target: u8,
    /// Physical tile id, else [`NO_PAI`].
    pub pai: u8,
    /// Claimed tiles + length.
    pub consumed: ([u8; 4], u8),
    /// Dahai tsumogiri flag.
    pub tsumogiri: bool,
    /// Raw line bytes (gate scans `deltas` / bare-alias type here, zero-alloc).
    pub span: &'a [u8],
    /// `dora_marker` bytes when present and non-empty.
    pub dora_span: Option<&'a [u8]>,
    /// Wall-list bytes when the line carries one.
    pub wall_span: Option<&'a [u8]>,
}

/// One framed game borrowing the ingest arena (plan §6.1, verbatim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramedGame<'a> {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// Hashed object id (numeric; string ids live cold-side only).
    pub object_id: u32,
    /// Stacked events in log order.
    pub events: Vec<StackedEvent<'a>>,
    /// Validated 136-permutation wall, when the payload carries one.
    pub wall: Option<[u8; 136]>,
    /// Opaque game key (seed material, never a display id).
    pub game_key: u64,
}

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
    frame_lines(buf, (0, buf.len()), object_id, game_idx, parse_line_baseline)
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
    if line_start < buf.len()
        && line_start > game_start
        && is_start_line(&buf[line_start..])
    {
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
    if bytes.len() >= 4 && bytes[0] == 0x28 && bytes[1] == 0xB5 && bytes[2] == 0x2F && bytes[3] == 0xFD
    {
        let mut dec =
            zstd::Decoder::new(bytes).map_err(|_| GateReject::decompress(game_idx))?;
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
        Some(Value::String(s)) => TileCodec::physical(s.as_bytes())
            .map_err(|_| GateReject::bad_tile(game_idx, idx))?,
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
    for key in [b"wall".as_slice(), b"wall_tiles".as_slice(), b"tiles".as_slice()] {
        if let Some((s, e)) = raw_value_bounds(line, key) && line[s] == b'[' {
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

/// Raw value bounds `(start, end)` of the LAST top-level `"key"`
/// occurrence, if the line is a well-formed object. Span-location only
/// (values are interpreted from the serde parse); `None` on any structural
/// surprise — never panics.
fn raw_value_bounds(line: &[u8], key: &[u8]) -> Option<(usize, usize)> {
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
fn parse_line_fast<'a>(
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
                    dora_span = if ve - vs > 0 { Some(&line[vs..ve]) } else { None };
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
    let line_wall_span = cands
        .iter()
        .flatten()
        .next()
        .map(|c| &line[c.0..c.1]);
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

fn is_blank_line(line: &[u8]) -> bool {
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
            } else if !matches!(
                e,
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
            ) {
                return None;
            }
        } else if c < 0x20 {
            return None;
        }
        i += 1;
    }
    None
}

enum StrDec {
    Ok(usize),
    Overflow,
    Malformed,
}

/// Decode escape sequences of a validated string interior into `out`.
fn decode_into(raw: &[u8], out: &mut [u8]) -> StrDec {
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

/// Encode one scalar value as UTF-8; returns the byte count.
fn utf8_encode(cp: u32, out: &mut [u8; 4]) -> usize {
    if cp < 0x80 {
        out[0] = cp as u8;
        1
    } else if cp < 0x800 {
        out[0] = (0xC0 | (cp >> 6)) as u8;
        out[1] = (0x80 | (cp & 0x3F)) as u8;
        2
    } else if cp < 0x10000 {
        out[0] = (0xE0 | (cp >> 12)) as u8;
        out[1] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
        out[2] = (0x80 | (cp & 0x3F)) as u8;
        3
    } else {
        out[0] = (0xF0 | (cp >> 18)) as u8;
        out[1] = (0x80 | ((cp >> 12) & 0x3F)) as u8;
        out[2] = (0x80 | ((cp >> 6) & 0x3F)) as u8;
        out[3] = (0x80 | (cp & 0x3F)) as u8;
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
        v = v
            .saturating_mul(10)
            .saturating_add((*b - b'0') as u32);
    }
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
                out[n] = v as u8;
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

// ---------------------------------------------------------------------------
// Tests (unit scope; oracle histogram agreement runs at P1-DONE)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{
        REASON_BLANK_LINE, REASON_BOUNDARY, REASON_FRAMING, REASON_UNKNOWN_KIND, REASON_WALL_PERM,
    };

    const GOOD_B: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/good-b.jsonl");
    const WALLED_SYNTH: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-synth.jsonl");
    const WALLED_REAL: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-real.jsonl");
    const WALLED_ANKAN: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-ankan.jsonl");
    const WALLED_KAKAN: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-kakan.jsonl");
    const WALLED_POST: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-post-terminal.jsonl");
    const WALLED_TRUNC: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s7/walled-truncated.jsonl");
    const Q_UNKNOWN: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/q-unknown-event.jsonl");
    const Q_NO_PAI: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/q-tile-conservation.jsonl");
    const Q_FRAMING: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/q-framing.jsonl");
    const Q_TRUNCATED: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/q-truncated.jsonl");
    const Q_WALL: &str =
        include_str!("../../../../../tools/hydra2-replay-rs/tests/fixtures/s4/q-wall-bearing.jsonl");

    fn frame_both(
        bytes: &[u8],
        object_id: u32,
        game_idx: u32,
    ) -> (Result<ProjGame, GateReject>, Result<ProjGame, GateReject>) {
        // Separate arenas: each game borrows its own side (flip discipline).
        let mut a = Vec::new();
        let mut b = Vec::new();
        let r0 = frame_spans_serde(&mut a, bytes, object_id, game_idx);
        let r1 = frame_spans_fast(&mut b, bytes, object_id, game_idx);
        // Owned projections (lifetimes differ by arena); borrow ends on move.
        (proj(r0), proj(r1))
    }

    fn proj(r: Result<FramedGame<'_>, GateReject>) -> Result<ProjGame, GateReject> {
        r.map(|g| ProjGame {
            game_idx: g.game_idx,
            object_id: g.object_id,
            game_key: g.game_key,
            wall: g.wall,
            events: g.events.iter().map(ProjEvent::of).collect(),
        })
    }

    /// Concatenate fixture texts into one multi-game file image (test-only).
    fn cat(parts: &[&str]) -> Vec<u8> {
        let mut v = Vec::new();
        for p in parts {
            v.extend_from_slice(p.as_bytes());
        }
        v
    }

    /// Paired per-path verdicts from [`frame_file_both`]
    /// (named alias for clippy::type_complexity).
    type BothVerdicts = (
        Vec<Result<ProjGame, GateReject>>,
        Vec<Result<ProjGame, GateReject>>,
    );

    /// Frame one file on both paths; project every per-game verdict so the
    /// two paths compare owned (arenas differ). Pins the staged-count
    /// invariant (`count == results.len()`) on every input.
    fn frame_file_both(bytes: &[u8], object_id: u32, base: u32) -> BothVerdicts {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let (r0, n0) = frame_file_serde(&mut a, bytes, object_id, base);
        let (r1, n1) = frame_file_fast(&mut b, bytes, object_id, base);
        assert_eq!(n0, r0.len() as u32);
        assert_eq!(n1, r1.len() as u32);
        (
            r0.into_iter().map(proj).collect(),
            r1.into_iter().map(proj).collect(),
        )
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ProjEvent {
        kind: u8,
        actor: u8,
        target: u8,
        pai: u8,
        consumed: ([u8; 4], u8),
        tsumogiri: bool,
        dora_span: Option<Vec<u8>>,
        wall_span: Option<Vec<u8>>,
    }

    impl ProjEvent {
        fn of(e: &StackedEvent<'_>) -> Self {
            Self {
                kind: e.kind,
                actor: e.actor,
                target: e.target,
                pai: e.pai,
                consumed: e.consumed,
                tsumogiri: e.tsumogiri,
                dora_span: e.dora_span.map(|s| s.to_vec()),
                wall_span: e.wall_span.map(|s| s.to_vec()),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ProjGame {
        game_idx: u32,
        object_id: u32,
        game_key: u64,
        wall: Option<[u8; 136]>,
        events: Vec<ProjEvent>,
    }

    #[test]
    fn f1_smoke_baseline_shape() {
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, GOOD_B.as_bytes(), 7, 3).unwrap();
        assert_eq!(g.events.len(), 14);
        assert_eq!(g.wall, None);
        assert_eq!(g.game_idx, 3);
        assert_eq!(g.object_id, 7);
        let kinds: Vec<u8> = g.events.iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            [
                KIND_START,
                KIND_START_KYOKU,
                KIND_TSUMO,
                KIND_DAHAI,
                KIND_TSUMO,
                KIND_DAHAI,
                KIND_TSUMO,
                KIND_DAHAI,
                KIND_TSUMO,
                KIND_DAHAI,
                KIND_TSUMO,
                KIND_HORA,
                KIND_END_KYOKU,
                KIND_END
            ]
        );
        // tsumogiri + hora seats.
        assert!(g.events[3].tsumogiri);
        assert_eq!(g.events[11].actor, 0);
        assert_eq!(g.events[11].target, 0);
        // Deterministic key.
        let mut arena2 = Vec::new();
        let g2 = frame_spans(&mut arena2, GOOD_B.as_bytes(), 7, 3).unwrap();
        assert_eq!(g.game_key, g2.game_key);
    }

    #[test]
    fn f3_walled_shapes() {
        for text in [WALLED_SYNTH, WALLED_REAL, WALLED_ANKAN, WALLED_KAKAN] {
            let mut arena = Vec::new();
            let g = frame_spans(&mut arena, text.as_bytes(), 1, 0).unwrap();
            let wall = g.wall.unwrap();
            let mut sorted = wall;
            sorted.sort_unstable();
            for (i, v) in sorted.iter().enumerate() {
                assert_eq!(*v as usize, i);
            }
            assert!(g.events[0].wall_span.is_some());
            assert!(g.events[1].wall_span.is_none());
        }
        // Consumed capture on the ankan/pon lines.
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, WALLED_ANKAN.as_bytes(), 1, 0).unwrap();
        let ankan = g.events.iter().find(|e| e.kind == KIND_ANKAN).unwrap();
        assert_eq!(ankan.consumed, ([0, 0, 0, 0], 4));
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, WALLED_KAKAN.as_bytes(), 1, 0).unwrap();
        let pon = g.events.iter().find(|e| e.kind == KIND_PON).unwrap();
        assert_eq!(pon.actor, 1);
        assert_eq!(pon.target, 0);
        assert_eq!(pon.consumed, ([108, 108, 0, 0], 2));
    }

    #[test]
    fn baseline_and_fast_agree_on_pinned_fixtures() {
        // Accept verdicts must project identically (fields + spans + wall).
        for text in [
            GOOD_B,
            WALLED_SYNTH,
            WALLED_REAL,
            WALLED_ANKAN,
            WALLED_KAKAN,
            WALLED_POST,
            WALLED_TRUNC,
            Q_UNKNOWN,
            Q_NO_PAI,
            Q_WALL,
        ] {
            let (r0, r1) = frame_both(text.as_bytes(), 11, 5);
            assert_eq!(r0, r1);
            assert!(r0.is_ok());
        }
        // Reject verdicts must agree exactly (reason + index).
        for text in [Q_FRAMING, Q_TRUNCATED] {
            let (r0, r1) = frame_both(text.as_bytes(), 11, 5);
            assert!(r0.is_err());
            assert_eq!(r0, r1);
        }
    }

    #[test]
    fn framing_rejects_with_oracle_reasons() {
        let mut arena = Vec::new();
        // Blank line (q-framing).
        let e = frame_spans(&mut arena, Q_FRAMING.as_bytes(), 0, 0).unwrap_err();
        assert_eq!(e.reason, REASON_BLANK_LINE);
        assert_eq!(e.event_idx, 1);
        // Truncated (q-truncated: last record is start_kyoku, not end_game).
        let mut arena = Vec::new();
        let e = frame_spans(&mut arena, Q_TRUNCATED.as_bytes(), 0, 0).unwrap_err();
        assert_eq!(e.reason, REASON_BOUNDARY);
        assert_eq!(e.event_idx, 1);
        // Missing type.
        let mut arena = Vec::new();
        let e = frame_spans(
            &mut arena,
            b"{\"type\":\"start_game\"}\n{\"x\":1}\n{\"type\":\"end_game\"}\n",
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(e.reason, REASON_UNKNOWN_KIND);
        assert_eq!(e.event_idx, 1);
        // Bad boundary.
        let mut arena = Vec::new();
        let e = frame_spans(
            &mut arena,
            b"{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n",
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(e.reason, REASON_BOUNDARY);
        // Double canonical start.
        let mut arena = Vec::new();
        let e = frame_spans(
            &mut arena,
            b"{\"type\":\"start_game\"}\n{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(e.reason, REASON_BOUNDARY);
        // Bad tile.
        let mut arena = Vec::new();
        let e = frame_spans(
            &mut arena,
            b"{\"type\":\"start_game\"}\n{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9x\"}\n{\"type\":\"end_game\"}\n",
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(e.reason, REASON_FRAMING);
        assert_eq!(e.event_idx, 1);
        // Bad wall (duplicate id 0, missing 135).
        let mut wall = [0u8; 136];
        for (i, w) in wall.iter_mut().enumerate() {
            *w = (i % 135) as u8;
        }
        assert!(!crate::gate::is_wall_perm136(&wall));
    }

    #[test]
    fn unknown_kind_stacks_as_other_and_missing_pai_is_none() {
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, Q_UNKNOWN.as_bytes(), 0, 0).unwrap();
        assert_eq!(g.events[2].kind, KIND_OTHER);
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, Q_NO_PAI.as_bytes(), 0, 0).unwrap();
        assert_eq!(g.events[2].pai, NO_PAI);
    }

    #[test]
    fn compressed_inputs_frame_identically() {
        let raw = GOOD_B.as_bytes();
        let zst = zstd::encode_all(raw, 3).unwrap();
        assert_eq!(&zst[..4], &[0x28, 0xB5, 0x2F, 0xFD]);
        let mut a = Vec::new();
        let mut b = Vec::new();
        let g0 = frame_spans_serde(&mut a, &zst, 9, 9).unwrap();
        let g1 = frame_spans_fast(&mut b, &zst, 9, 9).unwrap();
        assert_eq!(g0.events.len(), 14);
        assert_eq!(g0.events.len(), g1.events.len());
        assert_eq!(g0.wall, g1.wall);
        // gzip magic path.
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut enc, raw).unwrap();
        let gz = enc.finish().unwrap();
        let mut c = Vec::new();
        let g2 = frame_spans(&mut c, &gz, 9, 9).unwrap();
        assert_eq!(g2.events.len(), 14);
        // Corrupt magic-present payload fails closed.
        let mut d = Vec::new();
        let e = frame_spans(&mut d, &[0x28, 0xB5, 0x2F, 0xFD, 0x00], 0, 0).unwrap_err();
        assert_eq!(e.reason, REASON_FRAMING);
    }

    #[test]
    fn f6_packager_zst_rejects_like_decode_py() {
        // F6 (manifest 8a2f) decompresses to a single typeless envelope
        // line: decode.py rejects (missing `type`); both paths agree.
        const ZST: &[u8] = include_bytes!(
            "../../../../../tools/mjai-dataset-packager/tests/fixtures/c.mjai.json.zst"
        );
        let mut a = Vec::new();
        let mut b = Vec::new();
        let e0 = frame_spans_serde(&mut a, ZST, 0, 0).unwrap_err();
        let e1 = frame_spans_fast(&mut b, ZST, 0, 0).unwrap_err();
        assert_eq!(e0.reason, REASON_UNKNOWN_KIND);
        assert_eq!(e0, e1);
    }

    #[test]
    fn file_single_game_matches_spans_path() {
        // No special-casing: one game frames exactly like frame_spans(idx 0).
        let bytes = GOOD_B.as_bytes();
        let mut a = Vec::new();
        let mut c = Vec::new();
        let (mut f, n) = frame_file(&mut a, bytes, 7, 0);
        assert_eq!((f.len(), n), (1, 1));
        let g = frame_spans(&mut c, bytes, 7, 0).unwrap();
        let fg = f.remove(0).unwrap();
        assert_eq!(fg.events.len(), 14);
        assert_eq!(fg.events.len(), g.events.len());
        assert_eq!(fg.game_idx, 0);
        assert_eq!(fg.object_id, 7);
        assert_eq!(fg.wall, None);
        assert_eq!(fg.game_key, g.game_key);
        let kinds: Vec<u8> = fg.events.iter().map(|e| e.kind).collect();
        let kinds0: Vec<u8> = g.events.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, kinds0);
    }

    #[test]
    fn file_two_games_accept_in_order() {
        let bytes = cat(&[GOOD_B, WALLED_SYNTH]);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, 0);
        assert_eq!((f.len(), n), (2, 2));
        let g0 = f[0].as_ref().unwrap();
        let g1 = f[1].as_ref().unwrap();
        assert_eq!(g0.events.len(), 14);
        assert_eq!(g1.events.len(), 13);
        assert_eq!((g0.game_idx, g1.game_idx), (0, 1));
        assert_eq!(g0.wall, None);
        assert!(g1.wall.is_some());
        assert_ne!(g0.game_key, g1.game_key);
    }

    #[test]
    fn file_mixed_valid_invalid_quarantine_independent() {
        // The blank-line game fails alone; siblings still sample.
        let bytes = cat(&[GOOD_B, Q_FRAMING, GOOD_B]);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, 0);
        assert_eq!((f.len(), n), (3, 3));
        assert_eq!(f[0].as_ref().unwrap().events.len(), 14);
        assert_eq!(f[2].as_ref().unwrap().events.len(), 14);
        assert_eq!(f[2].as_ref().unwrap().game_idx, 2);
        match &f[1] {
            Err(e) => {
                assert_eq!(e.reason, REASON_BLANK_LINE);
                assert_eq!(e.event_idx, 1);
            }
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn file_f8_shape_22_games() {
        // F8 packs 22 games/file: every game samples independently.
        let parts = [GOOD_B; 22];
        let bytes = cat(&parts);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, 0);
        assert_eq!((f.len(), n), (22, 22));
        for (i, r) in f.iter().enumerate() {
            let g = r.as_ref().unwrap();
            assert_eq!(g.events.len(), 14);
            assert_eq!(g.game_idx, i as u32);
        }
    }

    #[test]
    fn file_leading_junk_chunk_fails_alone() {
        // F6-style typeless envelope line, then a good game.
        let bytes = cat(&[
            "{\"origin\":\"precompressed\",\"note\":\"byte-copy contract\"}\n",
            GOOD_B,
        ]);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, 0);
        assert_eq!((f.len(), n), (2, 2));
        match &f[0] {
            Err(e) => {
                assert_eq!(e.reason, REASON_UNKNOWN_KIND);
                assert_eq!(e.event_idx, 0);
            }
            Ok(_) => panic!(),
        }
        assert_eq!(f[1].as_ref().unwrap().events.len(), 14);
        assert_eq!(f[1].as_ref().unwrap().game_idx, 1);
    }

    #[test]
    fn file_unterminated_tail_fails_last_only() {
        let stripped = GOOD_B.strip_suffix('\n').unwrap();
        let bytes = cat(&[GOOD_B, stripped]);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, 0);
        assert_eq!((f.len(), n), (2, 2));
        assert_eq!(f[0].as_ref().unwrap().events.len(), 14);
        match &f[1] {
            Err(e) => assert_eq!(e.reason, REASON_FRAMING),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn file_baseline_fast_agree() {
        let stripped = GOOD_B.strip_suffix('\n').unwrap();
        let inputs = [
            cat(&[GOOD_B]),
            cat(&[GOOD_B, WALLED_SYNTH]),
            cat(&[GOOD_B, Q_FRAMING, GOOD_B]),
            cat(&[GOOD_B; 22]),
            cat(&[
                "{\"origin\":\"precompressed\",\"note\":\"byte-copy contract\"}\n",
                GOOD_B,
            ]),
            cat(&[GOOD_B, stripped]),
        ];
        for bytes in &inputs {
            let (r0, r1) = frame_file_both(bytes, 11, 0);
            assert_eq!(r0, r1);
        }
        // Nonzero base also agrees (exercises the offset path on both).
        let (r0, r1) = frame_file_both(&inputs[2], 11, 41);
        assert_eq!(r0, r1);
        assert_eq!(r0.len(), 3);
        // Plus every pinned single-game fixture through the file path.
        for text in [
            GOOD_B,
            WALLED_SYNTH,
            WALLED_REAL,
            WALLED_ANKAN,
            WALLED_KAKAN,
            WALLED_POST,
            WALLED_TRUNC,
            Q_UNKNOWN,
            Q_NO_PAI,
            Q_WALL,
        ] {
            let (r0, r1) = frame_file_both(text.as_bytes(), 11, 0);
            assert_eq!(r0.len(), 1);
            assert_eq!(r0, r1);
            assert!(r0[0].is_ok());
        }
    }

    #[test]
    fn file_base_offsets_key_globally_unique() {
        // Bridge advances game_seq by the staged count: consecutive files
        // must not collide. Rejects occupy keyspace too.
        fn idx_of(r: &Result<FramedGame<'_>, GateReject>) -> u32 {
            match r {
                Ok(g) => g.game_idx,
                Err(e) => e.game_idx,
            }
        }
        let bytes = cat(&[GOOD_B, Q_FRAMING, GOOD_B]);
        let mut a = Vec::new();
        let mut b = Vec::new();
        let (f0, n0) = frame_file(&mut a, &bytes, 7, 100);
        let (f1, n1) = frame_file(&mut b, &bytes, 7, 103);
        assert_eq!((n0, n1), (3, 3));
        let i0: Vec<u32> = f0.iter().map(idx_of).collect();
        let i1: Vec<u32> = f1.iter().map(idx_of).collect();
        assert_eq!(i0, [100, 101, 102]);
        assert_eq!(i1, [103, 104, 105]);
        // Same content, distinct keys: no cross-file collision.
        let k0 = f0[0].as_ref().unwrap().game_key;
        let k1 = f1[0].as_ref().unwrap().game_key;
        assert_ne!(k0, k1);
        // The middle reject sits at its keyed slot on both files.
        match &f0[1] {
            Err(e) => {
                assert_eq!(e.game_idx, 101);
                assert_eq!(e.reason, REASON_BLANK_LINE);
            }
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn file_base_saturates_without_wrap() {
        let bytes = cat(&[GOOD_B, GOOD_B]);
        let mut a = Vec::new();
        let (f, n) = frame_file(&mut a, &bytes, 7, u32::MAX);
        assert_eq!((f.len(), n), (2, 2));
        assert_eq!(f[0].as_ref().unwrap().game_idx, u32::MAX);
        assert_eq!(f[1].as_ref().unwrap().game_idx, u32::MAX);
        assert!(f.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn chunk_ranges_cover_bytes_exactly_once() {
        let bytes = cat(&[GOOD_B, Q_FRAMING, WALLED_SYNTH]);
        let mut a = Vec::new();
        let ranges = file_chunk_ranges(&mut a, &bytes, 0).unwrap();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[2].1, a.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        for (s, e) in &ranges {
            assert!(s < e);
        }
    }

    #[test]
    fn chunk_ranges_match_frame_file_chunks() {
        // Slicing the filled arena per range and framing each slice with
        // the single-game entry point equals the file path chunk-for-chunk
        // (verdicts, counts, walls, keys) — the staging seam is exact.
        let bytes = cat(&[GOOD_B, Q_FRAMING, WALLED_SYNTH]);
        let mut a = Vec::new();
        let ranges = file_chunk_ranges(&mut a, &bytes, 0).unwrap();
        let mut f = Vec::new();
        let (file_res, n) = frame_file(&mut f, &bytes, 7, 0);
        assert_eq!((ranges.len(), file_res.len(), n), (3, 3, 3));
        for (i, (s, e)) in ranges.iter().enumerate() {
            let mut c = Vec::new();
            let single = frame_spans(&mut c, &a[*s..*e], 7, i as u32);
            match (&single, &file_res[i]) {
                (Ok(g1), Ok(g2)) => {
                    assert_eq!(g1.events.len(), g2.events.len());
                    assert_eq!(g1.wall, g2.wall);
                    assert_eq!(g1.game_key, g2.game_key);
                }
                (Err(e1), Err(e2)) => assert_eq!(e1, e2),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn chunk_ranges_fail_closed_on_file_errors() {
        let mut a = Vec::new();
        let e = file_chunk_ranges(&mut a, b"", 5).unwrap_err();
        assert_eq!((e.game_idx, e.reason), (5, REASON_FRAMING));
        let mut b = Vec::new();
        let e = file_chunk_ranges(&mut b, &[0x28, 0xB5, 0x2F, 0xFD, 0x00], 5)
            .unwrap_err();
        assert_eq!((e.game_idx, e.reason), (5, REASON_FRAMING));
    }

    #[test]
    fn double_buffer_flip_never_clears_borrowed() {
        // Flip discipline: fill-one/clear-other across two arenas.
        let mut a = Vec::new();
        let mut b = Vec::new();
        let g0 = frame_spans(&mut a, GOOD_B.as_bytes(), 1, 0).unwrap();
        assert_eq!(g0.events.len(), 14);
        let g1 = frame_spans(&mut b, WALLED_SYNTH.as_bytes(), 2, 1).unwrap();
        assert!(g1.wall.is_some());
        // First game still fully readable while the second is framed.
        assert_eq!(g0.events[11].kind, KIND_HORA);
        assert_eq!(g0.events.len(), 14);
    }

    #[test]
    fn kind_lut_matches_vocab_consts() {
        for t in START_TYPES {
            assert_eq!(kind_from_bytes(t.as_bytes()), KIND_START);
        }
        for t in END_TYPES {
            assert_eq!(kind_from_bytes(t.as_bytes()), KIND_END);
        }
        for (t, k) in [
            ("dahai", KIND_DAHAI),
            ("chi", KIND_CHI),
            ("pon", KIND_PON),
            ("daiminkan", KIND_DAIMINKAN),
            ("ankan", KIND_ANKAN),
            ("kakan", KIND_KAKAN),
            ("hora", KIND_HORA),
        ] {
            assert_eq!(kind_from_bytes(t.as_bytes()), k);
        }
        assert_eq!(kind_from_bytes(b"frobnicate"), KIND_OTHER);
        assert_eq!(boundary_class(b"start_game"), 1);
        assert_eq!(boundary_class(b"start"), 2);
        assert_eq!(boundary_class(b"end_game"), 3);
        assert_eq!(boundary_class(b"end"), 4);
        assert_eq!(boundary_class(b"dahai"), 0);
    }


    #[test]
    fn wall_perm_helper_agrees_with_inline_check() {
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, WALLED_SYNTH.as_bytes(), 0, 0).unwrap();
        let wall = g.wall.unwrap();
        assert!(crate::gate::is_wall_perm136(&wall));
        assert!(!crate::gate::is_wall_perm136(&[0u8; 136]));
        let _ = REASON_WALL_PERM;
    }
}
