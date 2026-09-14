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

pub mod frame;
pub mod kinds;
pub mod parse;
#[cfg(test)]
mod tests;

pub use frame::{
    file_chunk_ranges, frame_file, frame_file_fast, frame_file_serde, frame_spans,
    frame_spans_fast, frame_spans_serde,
};
pub use kinds::{
    CLAIM_TYPES, END_TYPES, FramedGame, KIND_ANKAN, KIND_CHI, KIND_DAHAI, KIND_DAIMINKAN,
    KIND_DORA, KIND_END, KIND_END_KYOKU, KIND_HORA, KIND_KAKAN, KIND_OTHER, KIND_PON, KIND_REACH,
    KIND_REACH_ACCEPTED, KIND_RYUKYOKU, KIND_START, KIND_START_KYOKU, KIND_TRANSPARENT_OTHER,
    KIND_TSUMO, NO_PAI, NO_SEAT, ROW_TYPES, SKIP_TYPES, START_TYPES, StackedEvent,
    TRANSPARENT_KINDS, kind_from_bytes,
};
