//! stream_scan: bytes-in per-file scan worker for `training/stream_scan.py`.
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as
//! `rows_seal` / `packet_decode`; no shard/search/arrow edge, no new
//! dependency, no file IO here — the caller reads the file, this worker
//! only sees the compressed bytes).
//! FORBIDS: file IO, wall-clock, the JSON materialization round-trips
//! (`decode.py` / `validate.py` re-parse loops), the adapter probe (the
//! riichienv simulator verdict stays Python), and the merge/cache/pool
//! orchestration (wall sets, dedup, split assignment stay in
//! `stream_scan.py`). All framing/decode/hash math lives in `hydra-feed` —
//! this file never reimplements zstd framing, line rules, JCS/sha, or RNG.
//!
//! Concurrency pattern (matches `rows_seal.rs` + `packet_decode.rs`):
//! no attached validation needed (every input shape is total — any byte
//! string frames-or-fails, any stems decode-or-quarantine), ONE
//! `py.detach(|| ...)` around gate + frame + decode + wall fold (never
//! per-game attach), slots wrapped attached by pyo3 conversion. Frozen
//! surface only (one free function; no pyclass, no shared state, no stats
//! cell). GIL attestation (`gil_used=false` default on PyO3 >= 0.28) is
//! spelled once wave-wide in `canon_rng.rs` and inherited here.
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its pyfunction to the caller-provided `columnar` submodule, so the
//! Python translator keeps one bridge surface (`hydra2._native.columnar`).
//! MAIN wiring (shared files, MAIN-only): `pub mod stream_scan;` in `lib.rs`
//! (alphabetical, after `stream_driver`) plus
//! `crate::stream_scan::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:397-404` +
//! `rows_seal.rs:481-490`).
//!
//! Parity contract (mirrors `_scan_file_games`, `stream_scan.py:66-93`):
//! - Empty input scans zero games (HEAD: an empty file yields nothing —
//!   `stream_read.py:398-422` breaks at EOF with an empty buffer and
//!   `in_game == false`, so no frame is ever emitted).
//! - Non-empty input without the zstd frame magic fails (`ValueError`;
//!   the translator maps it to `CorruptArtifactError("zstd decode failed
//!   for {path}: ...")`, the same class HEAD raises from the zstd
//!   library). The gate exists because `framer::open_decoder`
//!   (`framer.rs:147-166`) transparently falls back to gzip/plain readers
//!   while HEAD's `ZstdLineStream` is strict zstd. Suffix detail text is
//!   feed-owned (the zstd-library suffix is input-dependent, never frozen).
//! - Framing is the shared owner by construction: `framer::iter_games`
//!   (`framer.rs:365-368`) is the exact port of `ZstdLineStream.iter_games`
//!   (`stream_read.py:335-422`; the class docstring at `:313-318` records
//!   both as byte-exact comparators pinned against each other), so blank
//!   lines, stray lines, boundary aliases, and the trailing no-newline
//!   remainder frame identically here.
//! - Per frame, any decode reject maps to a `None` slot (HEAD catches
//!   `ContractError`/`CorruptArtifactError`/`ValueError` around
//!   `decode_game_object` — `stream_scan.py:78-86` — and every
//!   `DecodeErr` class surfaces through one of those; the class itself is
//!   unobserved, only the quarantine is).
//! - Validation is skipped by design, output-identically: HEAD discards
//!   the outcome (`stream_scan.py:87-91` — content rejects arrive as
//!   invalid *outcomes*, never raises, so only a bridge-call failure
//!   could quarantine) and `hydra_feed::validate::validate_batch` is
//!   infallible (`validate.rs:550-551`, `&[GameRecord] -> Vec<Outcome>`),
//!   so in a pure-Rust composition there is no raise to observe and no
//!   side effect to preserve; only the adapter probe (Python-only) is
//!   consulted by the outcome, which is discarded either way.
//! - `wall_tiles is None` emits `(raw_sha, None, true)` (HEAD:
//!   `compute_wall_hash` returns `None` without calling the bridge and
//!   `is_sim` is `wall_tiles is None`).
//! - `wall_tiles is Some` converts each tile with `u32::try_from`: the
//!   first out-of-range tile aborts the whole call with `OverflowError`
//!   carrying the HEAD-observed pyo3 extraction text (`RANGE_MSG` —
//!   decode stores walls as `[i64; 136]` precisely so out-of-range ints
//!   stay representable, `decode.rs:29-31`, and HEAD's
//!   `packet.wall_hash(Vec<u32>)` extraction raises uncaught, propagating
//!   out of the file worker). A `wall_hash` canon reject aborts the call
//!   with `ValueError` carrying the feed-identical text: HEAD's wall-hash
//!   call sits *outside* the validate-only try, so every wall-hash reject
//!   propagates fail-closed out of the file call and never quarantines.
//!   `None` slots are reserved for decode rejects. The translator routes
//!   only `framer ...`-prefixed `ValueError`s (framing) to the corrupt-
//!   artifact mapping and lets every other `ValueError` (`canon ...`
//!   wall rejects) propagate with identical text — the two owners'
//!   Display prefixes are pinned disjoint by `error_prefixes_stay_disjoint`.
//! - Accepted residual: HEAD's decode materialization drift tripwire
//!   (`decode.py:98-102`, parsed-count vs bridge `event_count`) has no
//!   second implementation to drift against here, so it cannot fire; on
//!   decode-OK inputs both counts derive from the same line parse.
//! - Accepted residual: inputs past the 512 MiB `ZIP_BOMB_LIMIT`
//!   (`framer.rs:39-40`) fail closed here while HEAD streams them;
//!   corpus files are orders of magnitude smaller (fail-closed direction).

use hydra_feed::decode::WALL_LEN;
use hydra_feed::framer::{Frame, FramerError, iter_games, wall_hash};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

/// zstd standard-frame magic (`framer.rs:151-156` — the first branch of
/// `open_decoder`; skippable-frame-first inputs are out of scope, see the
/// module docs).
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// pyo3 `u32` extraction reject text, observed on the HEAD oracle
/// (`packet.wall_hash([-1] * 136)` and `([2**40] * 136)` both raise
/// `OverflowError` with exactly this message; the scan worker never
/// catches it, so it propagates out of the file call).
const RANGE_MSG: &str = "out of range integral type conversion attempted";

/// One scan slot: `None` quarantines the game, `Some` carries
/// `(raw_bytes_sha256, wall_hash_or_None, wall_tiles_is_None)` in file
/// order (the `_scan_file_games` tuple shape, `stream_scan.py:66-70`).
type Slot = Option<(String, Option<String>, bool)>;

/// Detach-local failure: framing rejects, the wall-range abort, and
/// wall-hash canon rejects (mapped to `PyValueError` / `PyOverflowError` /
/// `PyValueError` attached, after the detach — the two `ValueError` paths
/// carry disjoint feed-owned prefixes, see `error_prefixes_stay_disjoint`).
#[derive(Debug, PartialEq)]
enum ScanErr {
    Frame(FramerError),
    Range,
    Wall(String),
}

/// Strict-zstd framing: empty input scans zero games (HEAD empty-file
/// behavior); non-empty input must open with the zstd frame magic (HEAD
/// is strict zstd while `open_decoder` would fall back to gzip/plain).
/// Pure over bytes (detach-safe).
fn strict_frames(compressed: &[u8]) -> Result<Vec<Frame>, FramerError> {
    if compressed.is_empty() {
        return Ok(Vec::new());
    }
    if compressed.get(..ZSTD_MAGIC.len()) != Some(ZSTD_MAGIC.as_slice()) {
        return Err(FramerError::Decode {
            detail: "input lacks the zstd frame magic".to_string(),
        });
    }
    iter_games(compressed, 0, 0)
}

/// Per-frame decode + wall fold in file order (pure over owned frames,
/// detach-safe): decode rejects quarantine, wall-less games emit directly,
/// walled games convert tiles (first out-of-range tile aborts the call)
/// then fold through the wall-hash owner.
fn summarize_frames(
    frames: &[Frame],
    object_id: &str,
    packaged_object_id: &str,
) -> Result<Vec<Slot>, ScanErr> {
    let mut out: Vec<Slot> = Vec::with_capacity(frames.len());
    for frame in frames {
        let rec = match hydra_feed::decode::decode_game_object(
            object_id,
            packaged_object_id,
            &frame.bytes,
        ) {
            Err(_) => {
                out.push(None);
                continue;
            }
            Ok(rec) => rec,
        };
        let Some(wall) = rec.wall_tiles else {
            out.push(Some((rec.raw_bytes_sha256, None, true)));
            continue;
        };
        let mut fixed = [0u32; WALL_LEN];
        for (i, tile) in wall.iter().enumerate() {
            fixed[i] = u32::try_from(*tile).map_err(|_| ScanErr::Range)?;
        }
        match wall_hash(Some(&fixed)) {
            Ok(hash) => out.push(Some((rec.raw_bytes_sha256, hash, false))),
            Err(err) => return Err(ScanErr::Wall(err.to_string())),
        }
    }
    Ok(out)
}

/// Scan one corpus file from its compressed bytes (mirrors
/// `_scan_file_games`: file IO stays with the caller).
///
/// Returns per framed game, in file order: `None` when undecodable,
/// else `(raw_bytes_sha256, wall_hash_or_None, wall_tiles_is_None)`.
/// Non-zstd input raises `ValueError` (fail closed); a wall-hash canon
/// reject raises `ValueError` with the feed-identical text (fail closed,
/// like HEAD); a wall tile outside `u32` range raises `OverflowError` with
/// the HEAD-observed extraction text (fail closed, like HEAD).
#[pyfunction]
#[pyo3(signature = (compressed, object_id, packaged_object_id))]
fn scan_file_games(
    py: Python<'_>,
    compressed: Vec<u8>,
    object_id: String,
    packaged_object_id: String,
) -> PyResult<Vec<Slot>> {
    // Detached: borrow the attached-owned inputs across the detach (no
    // Python objects cross, no clone — mirrors `packet_decode.rs:235-246`).
    let slots: Result<Vec<Slot>, ScanErr> = py.detach(|| {
        let frames = strict_frames(&compressed).map_err(ScanErr::Frame)?;
        summarize_frames(&frames, &object_id, &packaged_object_id)
    });
    match slots {
        Ok(slots) => Ok(slots),
        Err(ScanErr::Frame(err)) => Err(PyValueError::new_err(err.to_string())),
        Err(ScanErr::Range) => Err(PyOverflowError::new_err(RANGE_MSG)),
        Err(ScanErr::Wall(detail)) => Err(PyValueError::new_err(detail)),
    }
}

/// Attach the scan pyfunction to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `columnar.rs:400-402` / `rows_seal.rs:484-489`; no new
/// submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(scan_file_games, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    fn frame_for(bytes: &[u8]) -> Frame {
        Frame {
            file_idx: 0,
            offset: 0,
            end: bytes.len() as u64,
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn frozen_consts_match_owners() {
        // zstd standard-frame magic (`framer.rs:151-156`); wall length
        // (`decode.rs:57-58` — the `[u32; WALL_LEN]` conversion target).
        assert_eq!(ZSTD_MAGIC, [0x28, 0xB5, 0x2F, 0xFD]);
        assert_eq!(WALL_LEN, 136);
        // OverflowError text observed on the HEAD oracle
        // (`packet.wall_hash` with out-of-`u32`-range tiles).
        assert_eq!(RANGE_MSG, "out of range integral type conversion attempted");
    }

    #[test]
    fn empty_input_scans_no_games() {
        // HEAD: an empty file yields zero framed games
        // (`stream_read.py:398-422` — EOF with an empty buffer and no
        // open game emits nothing), so the scan returns `[]`.
        assert_eq!(strict_frames(b""), Ok(Vec::new()));
    }

    #[test]
    fn non_zstd_input_fails_closed() {
        // HEAD: `ZstdLineStream` raises on non-zstd bytes (strict zstd
        // decode); the gate preserves the fail-closed class while the
        // translator keeps the oracle error prefix.
        assert!(strict_frames(b"{\"type\":\"start_game\"}\n").is_err());
        assert!(strict_frames(b"\x1f\x8b gibberish gzip magic").is_err());
        assert!(strict_frames(&[0x28, 0xB5, 0x2F, 0xFD, 0x00]).is_err());
    }

    #[test]
    fn decode_rejects_quarantine() {
        // HEAD: undecodable/invalid games append `None`
        // (`stream_scan.py:77-93`).
        assert_eq!(
            summarize_frames(&[frame_for(b"not json\n")], "o", "p"),
            Ok(vec![None])
        );
        assert_eq!(
            summarize_frames(&[frame_for(b"\n")], "o", "p"),
            Ok(vec![None])
        );
        assert_eq!(
            summarize_frames(&[frame_for(b"{\"type\":\"start_game\"}\n")], "o", "p"),
            Ok(vec![None])
        );
    }

    #[test]
    fn minimal_game_summarizes_wall_less() {
        // HEAD: a decodable wall-less game emits
        // `(raw_sha, None, True)`; the sha is the feed digest over the
        // verbatim bytes (compared programmatically, never by eye).
        let bytes = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        let want_sha = hydra_feed::digest::sha256_hex(bytes);
        assert_eq!(
            summarize_frames(&[frame_for(bytes)], "o", "p"),
            Ok(vec![Some((want_sha, None, true))])
        );
    }

    #[test]
    fn walled_game_carries_wall_hash() {
        // HEAD: a decodable walled game emits `(raw_sha, Some(hash),
        // False)` with the hash from the wall-hash owner (compared
        // programmatically, never by eye).
        let mut tiles = String::new();
        for (i, tile) in (0..WALL_LEN).enumerate() {
            if i > 0 {
                tiles.push(',');
            }
            tiles.push_str(&tile.to_string());
        }
        let raw = format!(
            "{{\"type\":\"start_game\",\"wall_tiles\":[{tiles}]}}\n{{\"type\":\"end_game\"}}\n"
        );
        let bytes = raw.as_bytes();
        let want_sha = hydra_feed::digest::sha256_hex(bytes);
        let mut arr = [0u32; WALL_LEN];
        for (i, tile) in (0..WALL_LEN).enumerate() {
            // proof: test `tile` in 0..136 (`WALL_LEN`), fits `u32`.
            #[allow(clippy::cast_possible_truncation)]
            let tile_u: u32 = tile as u32;
            arr[i] = tile_u;
        }
        let want_hash = match hydra_feed::framer::wall_hash(Some(&arr)) {
            Ok(hash) => hash,
            Err(_) => {
                panic!("wall_hash must succeed for test wall");
            }
        };
        assert_eq!(
            summarize_frames(&[frame_for(bytes)], "o", "p"),
            Ok(vec![Some((want_sha, want_hash, false))])
        );
    }

    #[test]
    fn negative_wall_aborts_range() {
        // HEAD: an out-of-`u32`-range wall tile raises `OverflowError`
        // out of the file worker (uncaught pyo3 extraction reject); the
        // translator lets it propagate unchanged.
        let mut tiles = String::from("-1");
        for tile in 1..WALL_LEN {
            tiles.push(',');
            tiles.push_str(&tile.to_string());
        }
        let raw = format!(
            "{{\"type\":\"start_game\",\"wall_tiles\":[{tiles}]}}\n{{\"type\":\"end_game\"}}\n"
        );
        assert_eq!(
            summarize_frames(&[frame_for(raw.as_bytes())], "o", "p"),
            Err(ScanErr::Range)
        );
    }

    #[test]
    fn error_prefixes_stay_disjoint() {
        // The translator routes only `framer ...`-prefixed ValueErrors to
        // the corrupt-artifact mapping and lets every other ValueError
        // (wall-hash canon rejects) propagate with identical text — HEAD's
        // wall-hash call sits outside the validate-only try, so the two
        // paths must stay distinguishable. Both prefixes are feed-owned;
        // pin every arm here so a new error shape cannot silently flip
        // the routing (`framer.rs:71-77`, `canon.rs:48-76`).
        use hydra_feed::canon::CanonError;
        let frame_err = FramerError::Decode {
            detail: "x".to_string(),
        };
        assert!(frame_err.to_string().starts_with("framer "));
        let canon_errs = [
            CanonError::DuplicateKey {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::NonFinite {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::LoneSurrogate {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::UnsafeNumber {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::NonStringKey {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::Bom {
                record: "r".to_string(),
            },
            CanonError::InvalidJson {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
            CanonError::Jcs {
                record: "r".to_string(),
                detail: "d".to_string(),
            },
        ];
        for err in canon_errs {
            let text = err.to_string();
            assert!(text.starts_with("canon "));
            assert!(!text.starts_with("framer "));
        }
    }
}
