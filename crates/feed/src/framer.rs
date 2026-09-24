//! Packet framer: `ZstdLineStream` port + verified zstd decode + stem/group keys.
//!
//! Rust owner of the framing half of `src/hydra2/data/stream_read.py`
//! (`ZstdLineStream.iter_games`, wave3 m1 window `:272-359`; `fetch_game_at`
//! `:50-101`; `stem_of` `:104-110`; `group_key_for[_path]` `:113-128`) and
//! the decode half of `src/hydra2/data/ingest.py` (`decode_zstd_verified`
//! `:36-84`: 64 KiB stream reads, incremental sha, 512 MiB guard). Python
//! oracles are read-only this phase: this module ports their behavior.
//!
//! Input is a byte window (`&[u8]`): the bridge memmaps (`memmap2::Mmap`
//! derefs) or passes a `PyBuffer`, Rust never opens files here. Compression
//! is transparent (zstd / gzip / plain, mirroring `fill_arena`); magic bytes
//! NEVER authorize — verified decode hashes every output byte and checks
//! `sha`+`len` when the caller supplies them.
//!
//! Identity discipline: SHA-256 ONLY via `feed::digest`
//! (same primitive; streaming `update` per 64 KiB chunk, chunk-size
//! invariant), canon bytes ONLY via `feed::canon`. No second printer, no
//! second hasher, no BLAKE3.
//!
//! Split math is NOT duplicated here: `wall_hash` canonicalizes through
//! `feed::canon` and digests through `feed::partition`'s helper, and
//! assignment calls `feed::partition::assign_one` directly (wave3 1.10 ONE-fn
//! rule). zstd dict parity note (wave2 8.2): this phase ships WITHOUT a
//! dictionary; a dict path MUST use `Decoder::with_dictionary` with the SAME
//! dict both ends (wrong-dict decode errors, never silent bytes).
//!
//! Import path (Wave5 M13): `zstd::Decoder` (repo-established, same as
//! `ingest::frame::fill_arena`), never a second decoder path.

use sha2::Digest as _;
use std::io::Read;

use crate::canon::{CanonError, canonical_bytes};

/// 64 KiB stream chunk — `stream_read.py:46` `_CHUNK_SIZE`, `ingest.py:60`.
pub const CHUNK: usize = 65536;

/// Zip-bomb guard — `ingest.py:49` (`512 * 1024 * 1024`).
pub const ZIP_BOMB_LIMIT: u64 = 512 * 1024 * 1024;

/// One framed game: owned verbatim bytes + stable decompressed offsets.
///
/// `offset` is the decompressed-byte offset of the first byte (`buf_start`
/// accounting, `stream_read.py:333-351`); `end` is one-past-the-last.
/// `bytes` is the raw payload INCLUDING the trailing `\n`, except a final
/// no-newline remainder which is preserved as-is so decode rejects it
/// (`:354-359`). Filenames/positions crossing the ABI go through
/// (`file_idx`, `offset`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// File index in the manifest walk (caller-assigned).
    pub file_idx: u32,
    /// Decompressed-byte offset of the first byte (+ caller `base`).
    pub offset: u64,
    /// One-past-the-last decompressed byte (+ caller `base`).
    pub end: u64,
    /// Raw verbatim game bytes.
    pub bytes: Vec<u8>,
}

/// Framer failure taxonomy (fail-closed, never silent coercion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FramerError {
    /// Compressed payload fails to decode / violates shape limits.
    Decode { detail: String },
    /// Requested offset is not on a game boundary.
    Offset { detail: String },
}

impl core::fmt::Display for FramerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode { detail } => write!(f, "framer decode: {detail}"),
            Self::Offset { detail } => write!(f, "framer offset: {detail}"),
        }
    }
}

impl std::error::Error for FramerError {}

/// Fully decode one object with streaming hash/length checks.
///
/// Mirrors `ingest.decode_zstd_verified`: 64 KiB reads, `Sha256::update` per
/// chunk (same primitive `feed::digest` wraps — chunk-size invariant, never
/// re-hashed hex), `total > 512 MiB` fail-closed BEFORE buffering more.
/// `expected_sha256` (`sha256:<hex>`) and `expected_len` (decompressed bytes)
/// mismatch → `Err` (never silent skip). `None` expectations still fully
/// decode (magic bytes alone never authorize).
pub fn decode_zstd_verified(
    compressed: &[u8],
    expected_sha256: Option<&str>,
    expected_len: Option<u64>,
) -> Result<Vec<u8>, FramerError> {
    let mut decoder = open_decoder(compressed)?;
    let mut hasher = sha2::Sha256::new();
    let mut out: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; CHUNK];
    loop {
        let n = decoder.read(&mut chunk).map_err(|e| FramerError::Decode {
            detail: format!("zstd decode failed: {e}"),
        })?;
        if n == 0 {
            break;
        }
        let total = out.len() as u64 + n as u64;
        if total > ZIP_BOMB_LIMIT {
            return Err(FramerError::Decode {
                detail: format!("decoded size exceeds 512MiB guard: {total}"),
            });
        }
        hasher.update(&chunk[..n]);
        out.extend_from_slice(&chunk[..n]);
    }
    if let Some(want) = expected_sha256 {
        let sum = hasher.finalize();
        let mut got = String::with_capacity(7 + 64);
        got.push_str("sha256:");
        for b in sum {
            got.push_str(&format!("{b:02x}"));
        }
        if got != want {
            return Err(FramerError::Decode {
                detail: format!("decoded hash mismatch: expected {want} got {got}"),
            });
        }
    }
    if let Some(want) = expected_len {
        let got = out.len() as u64;
        if got != want {
            return Err(FramerError::Decode {
                detail: format!("decoded length mismatch: expected {want} got {got}"),
            });
        }
    }
    Ok(out)
}

/// Plain decompress (no expectations): shared core behind
/// [`decode_zstd_verified`] and [`iter_games`].
fn decompress_all(compressed: &[u8]) -> Result<Vec<u8>, FramerError> {
    decode_zstd_verified(compressed, None, None)
}

/// Open a streaming byte reader: zstd magic → `zstd::Decoder` (M13 path),
/// gzip magic → `flate2`, else a plain slice reader. Magic selects the
/// CODEC only; authorization always comes from full decode (+ expectations).
fn open_decoder(compressed: &[u8]) -> Result<Box<dyn Read + '_>, FramerError> {
    if compressed.len() >= 4
        && compressed[0] == 0x28
        && compressed[1] == 0xB5
        && compressed[2] == 0x2F
        && compressed[3] == 0xFD
    {
        let dec = zstd::Decoder::new(compressed).map_err(|e| FramerError::Decode {
            detail: format!("zstd decoder open failed: {e}"),
        })?;
        Ok(Box::new(dec))
    } else if compressed.len() >= 2 && compressed[0] == 0x1F && compressed[1] == 0x8B {
        Ok(Box::new(flate2::read::GzDecoder::new(compressed)))
    } else {
        Ok(Box::new(compressed))
    }
}
// ---------------------------------------------------------------------------
// Line scan: _peek_type + _feed port (stream_read.py:272-359)
// ---------------------------------------------------------------------------
/// Boundary vocabulary: re-exported from the ingest kinds owner
/// (`crate::ingest::kinds::START_TYPES` / `END_TYPES`, mirroring
/// `stream_read.py:44-45` `_START_TYPES`/`_END_TYPES`). ONE table —
/// the framer never forks boundary strings from the span path.
pub use crate::ingest::kinds::{END_TYPES, START_TYPES};

/// Boundary classes the framer acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Peeked {
    Start,
    End,
    Other,
}

/// Superset prefilter for one line: the exact `_peek_type` fast path
/// (`stream_read.py:278-285`). Lines without `start`/`end` (and without a
/// `\u` escape that could hide them) cannot frame, so they skip the parse.
/// Anything else falls through to the exact scan below; the line rules
/// (decode owner) stay authoritative in all cases.
fn needs_exact_scan(line: &[u8]) -> bool {
    contains_needle(line, b"\\u")
        || contains_needle(line, b"start")
        || contains_needle(line, b"end")
}

/// Byte-substring search without extra features: `memchr` first-byte
/// filter, then a manual `windows` compare. `memchr::memmem` needs the
/// `std` feature — this keeps the `memchr = "2"` default-features pin.
fn contains_needle(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    for start in memchr::memchr_iter(needle[0], haystack) {
        if haystack.len() - start >= needle.len()
            && haystack[start..start + needle.len()] == *needle
        {
            return true;
        }
    }
    false
}

/// Exact `type` extraction: locate the LAST top-level `"type"` key (same
/// helper the ingest baseline uses) and return its boundary class.
/// Structurally surprising lines (not an object, no string `type`) yield
/// `None` — the state machine treats them as content lines, exactly like
/// `_peek_type` returning a non-boundary value. A `\u`-escaped boundary
/// name that decodes to a member of [`START_TYPES`]/[`END_TYPES`] still
/// frames (the `:281-283` fallthrough).
fn peek_type(line: &[u8]) -> Option<Peeked> {
    if !needs_exact_scan(line) {
        return None;
    }
    let (vs, ve) = crate::ingest::parse::raw_value_bounds(line, b"type")?;
    if vs >= ve || line[vs] != b'"' {
        return None;
    }
    let decoded = decode_json_string(&line[vs..ve])?;
    if START_TYPES.contains(&decoded.as_str()) {
        Some(Peeked::Start)
    } else if END_TYPES.contains(&decoded.as_str()) {
        Some(Peeked::End)
    } else {
        Some(Peeked::Other)
    }
}

/// Decode one JSON string literal (INCLUDING quotes) to its value.
/// Handles `\" \\ \/ \b \f \n \r \t \uXXXX` (with surrogate pairs);
/// returns `None` on any malformed literal (then the line is content, and
/// the decode owner rejects it with the line rules if it must).
fn decode_json_string(lit: &[u8]) -> Option<String> {
    if lit.len() < 2 || lit[0] != b'"' || lit[lit.len() - 1] != b'"' {
        return None;
    }
    let body = &lit[1..lit.len() - 1];
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        let b = body[i];
        if b != b'\\' {
            if b < 0x20 {
                return None;
            }
            if b < 0x80 {
                out.push(b as char);
                i += 1;
            } else {
                let len = utf8_len(b)?;
                if i + len > body.len() {
                    return None;
                }
                let s = core::str::from_utf8(&body[i..i + len]).ok()?;
                out.push_str(s);
                i += len;
            }
            continue;
        }
        i += 1;
        if i >= body.len() {
            return None;
        }
        match body[i] {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                if i + 4 >= body.len() {
                    return None;
                }
                let hi = hex4(&body[i + 1..i + 5])?;
                i += 4;
                let c = if (0xD800..0xDC00).contains(&hi) {
                    if i + 2 >= body.len() || body[i + 1] != b'\\' || body[i + 2] != b'u' {
                        return None;
                    }
                    if i + 6 >= body.len() {
                        return None;
                    }
                    let lo = hex4(&body[i + 3..i + 7])?;
                    if !(0xDC00..0xE000).contains(&lo) {
                        return None;
                    }
                    i += 6;
                    let cp = 0x10000 + ((u32::from(hi - 0xD800)) << 10) + u32::from(lo - 0xDC00);
                    char::from_u32(cp)?
                } else if (0xDC00..0xE000).contains(&hi) {
                    return None;
                } else {
                    char::from_u32(u32::from(hi))?
                };
                out.push(c);
            }
            _ => return None,
        }
        i += 1;
    }
    Some(out)
}

/// UTF-8 sequence length from a lead byte; `None` for continuation bytes.
fn utf8_len(lead: u8) -> Option<usize> {
    if lead < 0x80 {
        Some(1)
    } else if lead >> 5 == 0b110 {
        Some(2)
    } else if lead >> 4 == 0b1110 {
        Some(3)
    } else if lead >> 3 == 0b11110 {
        Some(4)
    } else {
        None
    }
}

/// Four hex digits to u16; `None` on non-hex.
fn hex4(d: &[u8]) -> Option<u16> {
    if d.len() != 4 {
        return None;
    }
    let mut v: u16 = 0;
    for b in d {
        v = v.checked_mul(16)?.checked_add(u16::from(hex_val(*b)?))?;
    }
    Some(v)
}

/// One hex digit value; `None` on non-hex.
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Framing: iter_games + fetch_game_at (stream_read.py:272-359, :50-101)
// ---------------------------------------------------------------------------

/// Frame the decompressed stream into games, in file order.
///
/// Exact port of `ZstdLineStream.iter_games` + `_feed` (`:272-326`): the
/// input is decompressed (transparent zstd/gzip/plain via [`decompress_all`],
/// the `copy_decode` 64 KiB equivalent — chunking is an I/O detail, framing
/// sees identical bytes), split on `\n` with `buf_start` offset accounting
/// (`:341-351`), blank lines handled by position (`:299-304`), START/END via
/// [`peek_type`] (`:306-319`), stray lines opening an implicit game
/// (`:320-323`), trailing no-`\n` remainder framed with `final=true`
/// (`:354-357`), and an unterminated open game flushed at EOF (`:358-359`).
/// `base` shifts every offset (cursor resume); `file_idx` tags each frame.
pub fn iter_games(compressed: &[u8], file_idx: u32, base: u64) -> Result<Vec<Frame>, FramerError> {
    let text = decompress_all(compressed)?;
    Ok(frame_decompressed(&text, file_idx, base))
}

/// Frame already-decompressed bytes (pure scan; shared by [`iter_games`] and
/// [`fetch_game_at`] so offsets agree by construction).
fn frame_decompressed(text: &[u8], file_idx: u32, base: u64) -> Vec<Frame> {
    let mut out: Vec<Frame> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pending_start: u64 = 0;
    let mut in_game = false;
    // Feed one line (WITHOUT its `\n`) at decompressed offset `line_start`;
    // `final` marks the trailing no-newline remainder (`:295-326`).
    let mut feed = |line: &[u8], line_start: u64, is_final: bool| {
        let mut raw = Vec::with_capacity(line.len() + 1);
        raw.extend_from_slice(line);
        if !is_final {
            raw.push(b'\n');
        }
        if line.iter().all(|b| b.is_ascii_whitespace()) && !is_final {
            if in_game {
                pending.extend_from_slice(&raw);
            } else {
                out.push(Frame {
                    file_idx,
                    offset: base + line_start,
                    end: base + line_start + raw.len() as u64,
                    bytes: raw,
                });
            }
            return;
        }
        match peek_type(line) {
            Some(Peeked::Start) => {
                if in_game {
                    let end = pending_start + pending.len() as u64;
                    let bytes = core::mem::take(&mut pending);
                    out.push(Frame {
                        file_idx,
                        offset: base + pending_start,
                        end: base + end,
                        bytes,
                    });
                }
                pending = raw;
                pending_start = line_start;
                in_game = true;
            }
            Some(Peeked::End) => {
                if in_game {
                    pending.extend_from_slice(&raw);
                    let end = pending_start + pending.len() as u64;
                    let bytes = core::mem::take(&mut pending);
                    out.push(Frame {
                        file_idx,
                        offset: base + pending_start,
                        end: base + end,
                        bytes,
                    });
                    in_game = false;
                } else {
                    out.push(Frame {
                        file_idx,
                        offset: base + line_start,
                        end: base + line_start + raw.len() as u64,
                        bytes: raw,
                    });
                }
            }
            _ => {
                if !in_game {
                    pending = raw;
                    pending_start = line_start;
                    in_game = true;
                } else {
                    pending.extend_from_slice(&raw);
                }
            }
        }
    };
    let mut line_start: u64 = 0;
    for nl in memchr::memchr_iter(b'\n', text) {
        // proof: file offsets <4B fit in usize on all targets.
        #[allow(clippy::cast_possible_truncation)]
        {
            feed(&text[line_start as usize..nl], line_start, false);
        }
        line_start = nl as u64 + 1;
    }
    // proof: file offsets <4B fit in usize on all targets.
    #[allow(clippy::cast_possible_truncation)]
    if (line_start as usize) < text.len() {
        // proof: file offsets <4B fit in usize on all targets.
        #[allow(clippy::cast_possible_truncation)]
        {
            feed(&text[line_start as usize..], line_start, true);
        }
    }
    if in_game && !pending.is_empty() {
        let end = pending_start + pending.len() as u64;
        out.push(Frame {
            file_idx,
            offset: base + pending_start,
            end: base + end,
            bytes: pending,
        });
    }
    out
}

/// Fetch one game by decompressed-byte offset, fail-closed.
///
/// Mirrors `fetch_game_at` (`:66-76`): offsets come from [`iter_games`]; an
/// exact hit returns its bytes, a past-the-target scan or a miss raises
/// (never the nearest game, never `None`). The decode/validate/split steps
/// (`:77-101`) belong to the decode/validate/partition siblings — this
/// returns the framed bytes so those stages see the same input the stream
/// would have emitted.
pub fn fetch_game_at(
    compressed: &[u8],
    file_idx: u32,
    base: u64,
    offset: u64,
) -> Result<Frame, FramerError> {
    if offset < base {
        return Err(FramerError::Offset {
            detail: format!("game offset {offset} below base {base}"),
        });
    }
    let text = decompress_all(compressed)?;
    // Fail-closed walk on the SAME state machine: frame once (offsets agree
    // by construction), hit returns the bytes, past-target or miss raises.
    let frames = frame_decompressed(&text, file_idx, base);
    for f in frames {
        if f.offset == offset {
            return Ok(f);
        }
        if f.offset > offset {
            break;
        }
    }
    Err(FramerError::Offset {
        detail: format!("game offset {offset} not on a game boundary"),
    })
}

// ---------------------------------------------------------------------------
// Stem / group keys (stream_read.py:104-128) + wall_hash / assign ONE-fn call
// ---------------------------------------------------------------------------

/// Game identity stem: `<stem>.mjai.json.zst` → `<stem>`
/// (`stream_read.py:104-110`). Suffixes strip longest-first; anything else
/// keeps `Path::stem` behavior on the file name.
pub fn stem_of(file_name: &str) -> &str {
    for suffix in [".mjai.json.zst", ".mjai.json", ".zst"] {
        if let Some(stem) = file_name.strip_suffix(suffix) {
            return stem;
        }
    }
    match file_name.rfind('.') {
        Some(dot) if dot > 0 => &file_name[..dot],
        _ => file_name,
    }
}

/// Group key identical to the partition `(source, time)` grouping join
/// (`stream_read.py:113-115`). Thin alias over the partition owner's
/// function — ONE definition, never a second join string.
pub fn group_key_for(source: &str, time: &str) -> String {
    crate::partition::group_key_for_source_time(source, time)
}

/// Split a corpus path into `(source, time)`: source is the parent directory
/// name (mount-independent; empty/`.` → `"unknown"`), time is the leading
/// digit run of the stem (`_DIGIT_RUN.match`, Tenhou `YYYYMMDDHH...`), else
/// `"unknown"`. ONE derivation shared by both group-key paths below.
fn split_source_time<'a>(parent_dir: &'a str, file_name: &str) -> (&'a str, String) {
    let source = if parent_dir.is_empty() || parent_dir == "." {
        "unknown"
    } else {
        parent_dir
    };
    let stem = stem_of(file_name);
    let digits: String = stem.chars().take_while(|c| c.is_ascii_digit()).collect();
    let time = if digits.is_empty() {
        "unknown".to_string()
    } else {
        digits
    };
    (source, time)
}

/// Derive the `(root-id, source, time)` group from a corpus path plus its
/// manifest root id (`stream_read.group_key_for_entry`): the multi-root
/// stream/scan path. Root leads so per-root draws stay independent.
pub fn group_key_for_path_with_root(root_id: &str, parent_dir: &str, file_name: &str) -> String {
    let (source, time) = split_source_time(parent_dir, file_name);
    crate::partition::group_key_for_root(root_id, source, &time)
}

/// Derive the `(source, time)` group from a corpus path
/// (`stream_read.py:118-128`): legacy single-context path (ad-hoc fetch,
/// tests, and callers without a manifest root id).
pub fn group_key_for_path(parent_dir: &str, file_name: &str) -> String {
    let (source, time) = split_source_time(parent_dir, file_name);
    group_key_for(source, &time)
}

/// Wall hash identical to the partition identity (`stream_read.py:131-138`):
/// `None` when no wall (null corpus-wide for real MJAI), else `sha256:`
/// over the canon bytes of the wall list. The list → canon-bytes step is
/// `feed::canon` (Data M1); the digest call is the partition owner's helper
/// (ONE hasher, `feed::digest` inside).
pub fn wall_hash(wall: Option<&[u32; 136]>) -> Result<Option<String>, CanonError> {
    let Some(w) = wall else { return Ok(None) };
    let list: Vec<u32> = w.to_vec();
    let bytes = canonical_bytes(&list, "framer:wall_hash")?;
    Ok(Some(crate::partition::wall_hash_of_canon_bytes(&bytes)))
}

/// Split assignment for one group: thin alias over the partition owner's
/// [`crate::partition::assign_one`] (wave3 §1.10 ONE-fn rule — stream and
/// corpus paths share one draw, one walk, one order). The `u64BE/2**64`
/// draw is preserved inside; it is NOT Lemire `bounded`.
pub fn assign_one(
    group_key: &str,
    seed: u64,
    ratios: &std::collections::BTreeMap<String, f64>,
) -> Result<String, crate::partition::PartitionError> {
    crate::partition::assign_one(group_key, seed, ratios)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write as _;

    fn game(start_ty: &str, end_ty: &str, tag: &str) -> Vec<u8> {
        format!(
            "{{\"type\":\"{start_ty}\",\"tag\":\"{tag}\"}}\n{{\"type\":\"dahai\",\"pai\":5}}\n{{\"type\":\"{end_ty}\"}}\n"
        )
        .into_bytes()
    }

    fn zstd_of(raw: &[u8]) -> Vec<u8> {
        let mut enc = zstd::Encoder::new(Vec::new(), 3).unwrap();
        enc.write_all(raw).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn framer_offsets_are_stable_decompressed_bytes() {
        // Contract: offsets are decompressed-byte offsets of each game's
        // first byte (buf_start accounting); concatenation re-slices exact.
        let g0 = game("start_game", "end_game", "a");
        let g1 = game("start_game", "end_game", "b");
        let mut raw = Vec::new();
        raw.extend_from_slice(&g0);
        raw.extend_from_slice(&g1);
        let frames = iter_games(&zstd_of(&raw), 2, 100).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].offset, 100);
        assert_eq!(frames[0].end, 100 + g0.len() as u64);
        assert_eq!(frames[1].offset, 100 + g0.len() as u64);
        assert_eq!(&raw[..g0.len()], frames[0].bytes.as_slice());
        assert_eq!(&raw[g0.len()..], frames[1].bytes.as_slice());
    }

    #[test]
    fn framer_blank_line_inside_game_is_content_not_skip() {
        // Contract: a blank line mid-game is framed INSIDE the game bytes
        // (quarantine-range, not skip — `:299-304`).
        let raw = b"{\"type\":\"start_game\"}\n\n{\"type\":\"end_game\"}\n".to_vec();
        let frames = iter_games(&zstd_of(&raw), 0, 0).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, raw);
    }

    #[test]
    fn framer_trailing_no_newline_remainder_is_framed() {
        // Contract: a final remainder without `\n` is framed as-is (final=true)
        // so decode rejects it — never silently dropped (`:354-357`).
        let raw = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}".to_vec();
        let frames = iter_games(&zstd_of(&raw), 0, 0).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, raw);
    }

    #[test]
    fn framer_bare_aliases_frame() {
        // Contract: bare `start`/`end` aliases are boundary vocabulary
        // (`:44-45`); canonical spellings are not required to frame.
        let raw = b"{\"type\":\"start\"}\n{\"type\":\"dahai\"}\n{\"type\":\"end\"}\n".to_vec();
        let frames = iter_games(&zstd_of(&raw), 0, 0).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, raw);
    }

    #[test]
    fn fetch_game_at_hits_and_fails_closed() {
        // Contract: exact offset hits; mid-game and past-end offsets raise
        // (never the nearest game — `fetch_game_at :66-76`).
        let g0 = game("start_game", "end_game", "a");
        let g1 = game("start_game", "end_game", "b");
        let mut raw = Vec::new();
        raw.extend_from_slice(&g0);
        raw.extend_from_slice(&g1);
        let enc = zstd_of(&raw);
        let hit = fetch_game_at(&enc, 0, 0, g0.len() as u64).unwrap();
        assert_eq!(hit.bytes, g1);
        assert!(fetch_game_at(&enc, 0, 0, 1).is_err());
        assert!(fetch_game_at(&enc, 0, 0, raw.len() as u64 + 7).is_err());
    }

    #[test]
    fn wall_hash_none_and_perm_agree_with_partition() {
        // Contract: None → None; Some(0..136) digests the SAME canon bytes
        // the partition helper hashes (ONE hasher, no second printer).
        assert_eq!(wall_hash(None).unwrap(), None);
        let wall: [u32; 136] = core::array::from_fn(|i| u32::try_from(i).unwrap());
        let got = wall_hash(Some(&wall)).unwrap().unwrap();
        let list: Vec<u32> = wall.to_vec();
        let bytes = canonical_bytes(&list, "kat").unwrap();
        assert_eq!(got, crate::partition::wall_hash_of_canon_bytes(&bytes));
        assert!(got.starts_with("sha256:"));
    }

    #[test]
    fn assign_one_alias_agrees_with_partition() {
        // Contract: this ONE-fn alias returns exactly what the partition
        // owner computes for the same (group, seed, ratios).
        let ratios = BTreeMap::from([("train".to_string(), 0.8), ("validation".to_string(), 0.2)]);
        for group in ["s|20240101", "t|unknown"] {
            assert_eq!(
                assign_one(group, 7, &ratios).unwrap(),
                crate::partition::assign_one(group, 7, &ratios).unwrap()
            );
        }
    }

    #[test]
    fn iter_games_window_is_offset_ordered() {
        // Contract: emission order is file order with strictly increasing
        // offsets over a multi-game window (ordered-source join key).
        let mut raw = Vec::new();
        for tag in ["a", "b", "c", "d"] {
            raw.extend_from_slice(&game("start_game", "end_game", tag));
        }
        let frames = iter_games(&zstd_of(&raw), 0, 0).unwrap();
        assert_eq!(frames.len(), 4);
        let mut prev = 0u64;
        for (i, f) in frames.iter().enumerate() {
            if i > 0 {
                assert!(f.offset > prev);
            }
            prev = f.offset;
            assert!(f.end > f.offset);
        }
    }

    #[test]
    fn stem_and_group_keys_match_oracle() {
        // Contract: suffix strip longest-first + parent-dir source + leading
        // digit-run time (`stream_read.py:104-128`).
        assert_eq!(stem_of("2024010112.mjai.json.zst"), "2024010112");
        assert_eq!(stem_of("g.mjai.json"), "g");
        assert_eq!(stem_of("g.zst"), "g");
        assert_eq!(
            group_key_for_path("lobby", "2024010112.mjai.json.zst"),
            "lobby|2024010112"
        );
        assert_eq!(
            group_key_for_path("", "anon.mjai.json.zst"),
            "unknown|unknown"
        );
        assert_eq!(
            group_key_for_path_with_root("ryu", "lobby", "2024010112.mjai.json.zst"),
            "ryu|lobby|2024010112"
        );
        assert_eq!(
            group_key_for_path_with_root("ryu", "", "anon.mjai.json.zst"),
            "ryu|unknown|unknown"
        );
    }

    #[test]
    fn zstd_verified_round_trip_and_guards() {
        // Contract: encode(3) → verified decode round-trips; wrong sha, wrong
        // len, and corrupt bytes all fail closed (magic never authorizes).
        let raw = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n".to_vec();
        let enc = zstd_of(&raw);
        let sha = crate::digest::sha256_hex(&raw);
        assert_eq!(
            decode_zstd_verified(&enc, Some(&sha), Some(raw.len() as u64)).unwrap(),
            raw
        );
        assert!(decode_zstd_verified(&enc, Some("sha256:00"), Some(raw.len() as u64)).is_err());
        assert!(decode_zstd_verified(&enc, Some(&sha), Some(1)).is_err());
        assert!(decode_zstd_verified(b"\x28\xB5\x2F\xFD corrupt", None, None).is_err());
    }
}
