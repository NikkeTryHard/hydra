//! ingest_leaves: compressed-digest + JSON-record-count leaves for `data/ingest.py`
//! on the shared `columnar` submodule.
//!
//! DAG: this module depends on pyo3 + `hydra-feed` digest ONLY (the same edge
//! as `canon_rng`; `serde_json` is already a bridge dependency per
//! `crates/bridge/Cargo.toml:22-24`, so no manifest change — no shard/search/
//! arrow edge, no new dependency, no file IO here; the caller reads files and
//! crosses only plain bytes).
//! FORBIDS: zstd decode (already bridged via `packet.decode_zstd_verified`,
//! Wave 1 R4 — never re-entered here), manifest/attestation orchestration
//! (`load_packaged_manifest`, `require_attestation`, `make_raw_object_row`
//! stay Python), and seal math (the `rows_seal` / feed-`rows` owners).
//!
//! Frozen surface (two free functions; no pyclass, no shared state — ingest
//! is cold path, per-object granularity):
//! - fn `ingest_compressed_digest` <- `ingest.py:141,147`: (`sha256:<hex>`,
//!   length) over the compressed bytes in ONE detach (one copy across the
//!   boundary; the hash runs through the feed `digest` owner, never a local
//!   hasher).
//! - fn `ingest_count_json_records` <- `ingest.py:152`: the lenient
//!   non-blank JSON-parsable line count over decoded bytes. The core restates
//!   `replay_ingest_bytes` (`crates/feed/src/rows.rs:1014-1022`), which is
//!   inline in the full-replay entry (seal + zstd + decode + validate +
//!   quarantine) and not callable as a leaf — NONE-owner note: `rg` for a
//!   standalone count fn over `crates/feed/src` hits only that inline loop
//!   (the `decode.rs:186,255-269` parser is the strict one-game gate that
//!   rejects blanks, a different gate, never the lenient count).
//!
//! Oracle-divergence notes (deliberate, garbage-input only — inherited from
//! the feed restatement, `rows.rs:1015-1021`):
//! - Lone-`\r` / `\x0b` / `\x0c` line splits: Python `splitlines` splits on
//!   them, the `b'\n'` split does not. Canonical JSONL ends lines with `\n`
//!   only (the packager writes `\n`), so real payloads never hit this.
//! - `NaN` / `Infinity` / `-Infinity` bare tokens: Python `json.loads`
//!   accepts them (the oracle counts `b'NaN\n'` as 1), `serde_json` rejects
//!   (counted 0 here). Writers emit `allow_nan=False` (`rows.py:102`), so
//!   real payloads never hit this.
//! - Non-UTF-8 bytes (incl. UTF-16/32): Python `json.loads(bytes)` sniffs the
//!   encoding, `serde_json::from_slice` requires UTF-8. Decoded payloads are
//!   UTF-8 JSONL by construction.
//!   Blank-line handling is exactly equivalent: a trailing `\n` leaves a final
//!   empty segment under BOTH `splitlines` and the `b'\n'` split, and the empty
//!   segment is vacuous-`all` whitespace (skipped) in both.
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its pyfunctions to the caller-provided `columnar` submodule, so the
//! Python translator keeps one bridge surface (`hydra2._native.columnar`).
//! MAIN wiring (shared files, MAIN-only): `pub mod ingest_leaves;` in
//! `lib.rs` (alphabetical, after `encoder`) plus
//! `crate::ingest_leaves::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:403-407`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-489`); `py.detach(|| ...)` around
//! Rust-only compute mirrors `canon_rng::sha256_hex`
//! (`crates/bridge/src/canon_rng.rs:184-189`); hashing through the feed
//! `digest` owner mirrors `belief_leaves::world_id_for_doc`
//! (`crates/bridge/src/belief_leaves.rs:114-121`); the per-line
//! `serde_json::from_slice::<Value>` validity probe mirrors the feed decode
//! boundary (`crates/feed/src/decode.rs:186`).

use pyo3::prelude::*;

/// Pure compressed-identity core (detach-safe: borrowed bytes only, no
/// interpreter interaction). The digest runs through the feed `digest` owner
/// (`sha256:`-prefixed lowercase hex, infallible — called bare); the length
/// is the plain byte length, exactly like the replaced `len(data)` (`ingest.py:147`).
fn compressed_digest_core(data: &[u8]) -> (String, u64) {
    (hydra_feed::digest::sha256_hex(data), data.len() as u64)
}

/// Pure lenient record-count core (detach-safe; restates the
/// `replay_ingest_bytes` loop at `crates/feed/src/rows.rs:1015-1021`).
/// Splits on `b'\n'`, skips all-ASCII-whitespace lines (empty included —
/// `all` over an empty slice is true, matching the oracle's `strip` gate),
/// and counts lines that parse as exactly one JSON value. Unparsable lines
/// are skipped, never an error (the pre-port oracle's `except JSONDecodeError:
/// continue`, formerly `ingest.py:123-124`).
fn count_json_records_core(decoded: &[u8]) -> u64 {
    let mut count: u64 = 0;
    for line in decoded.split(|b| *b == b'\n') {
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        if serde_json::from_slice::<serde_json::Value>(line).is_ok() {
            count = count.saturating_add(1);
        }
    }
    count
}

/// Compressed identity: `sha256:<hex>` + byte length over the compressed
/// object bytes (`ingest.py:141,147`). Thin wrapper over
/// [`compressed_digest_core`]; the whole hash runs detached, the tuple is
/// wrapped attached.
#[pyfunction]
#[pyo3(signature = (data,))]
fn ingest_compressed_digest(py: Python<'_>, data: Vec<u8>) -> PyResult<(String, u64)> {
    Ok(py.detach(|| compressed_digest_core(&data)))
}

/// Lenient JSON-record count over decoded bytes (`ingest.py:152`). Thin
/// wrapper over [`count_json_records_core`]; runs detached, wraps attached.
/// Infallible by construction (unparsable lines skip, never reject).
#[pyfunction]
#[pyo3(signature = (decoded,))]
fn ingest_count_json_records(py: Python<'_>, decoded: Vec<u8>) -> PyResult<u64> {
    Ok(py.detach(|| count_json_records_core(&decoded)))
}

/// Attach the ingest-leaf pyfunctions to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `rows_seal.rs:484-489`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(ingest_compressed_digest, sub)?)?;
    sub.add_function(wrap_pyfunction!(ingest_count_json_records, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod ingest_leaves_tests {
    use super::*;

    #[test]
    fn compressed_digest_matches_hashlib_oracle() {
        Python::initialize();
        // Hand-derived from the pre-port HEAD oracle
        // (`hashlib.sha256(data).hexdigest()` with the `sha256:` prefix,
        // formerly `ingest.py:104`; captured via CPython this session — never eyeballed hex).
        Python::attach(|py| {
            assert_eq!(
                ingest_compressed_digest(py, Vec::new()).unwrap(),
                (
                    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        .to_string(),
                    0u64
                )
            );
            assert_eq!(
                ingest_compressed_digest(py, b"compressed".to_vec()).unwrap(),
                (
                    "sha256:9da308c2e4bc33afa72df5c088b5fc5673c477f3ef21d6bdaa358393834f9804"
                        .to_string(),
                    10u64
                )
            );
            assert_eq!(
                ingest_compressed_digest(py, b"{\"a\":1}\n".to_vec()).unwrap(),
                (
                    "sha256:e346432021b04179518d9614f3560ccd71354a4ee101ddcb893d6959a9d6301c"
                        .to_string(),
                    8u64
                )
            );
        });
    }

    #[test]
    fn compressed_digest_agrees_with_feed_owner() {
        Python::initialize();
        // Programmatic cross-check (not eyeballed hex): the bridge digest is
        // the feed-owner digest over the same bytes, and the length is the
        // byte length — guards against hashing/wiring the wrong buffer.
        Python::attach(|py| {
            let data = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n".to_vec();
            let (digest, length) = ingest_compressed_digest(py, data.clone()).unwrap();
            assert_eq!(digest, hydra_feed::digest::sha256_hex(&data));
            assert_eq!(length, data.len() as u64);
            assert!(hydra_feed::partition::is_digest_text(&digest));
        });
    }

    #[test]
    fn count_matches_oracle_line_semantics() {
        Python::initialize();
        // Hand-derived from the pre-port HEAD oracle loop (formerly
        // `ingest.py:115-124`; oracle counts captured via CPython this session).
        Python::attach(|py| {
            // Empty payload: one empty segment, skipped as blank.
            assert_eq!(ingest_count_json_records(py, Vec::new()).unwrap(), 0u64);
            assert_eq!(ingest_count_json_records(py, b"\n".to_vec()).unwrap(), 0u64);
            assert_eq!(
                ingest_count_json_records(py, b"{\"a\":1}\n".to_vec()).unwrap(),
                1u64
            );
            // Missing trailing newline still counts (no `ends_with` gate here,
            // unlike the strict decode gate).
            assert_eq!(
                ingest_count_json_records(py, b"{\"a\":1}".to_vec()).unwrap(),
                1u64
            );
            // Invalid lines skip, whitespace-only lines skip, arrays count
            // (any JSON value, like `json.loads`).
            assert_eq!(
                ingest_count_json_records(py, b"{\"a\":1}\nnot json\n   \n[1,2]\n".to_vec())
                    .unwrap(),
                2u64
            );
            // CRLF pairs count (the `\r` rides as JSON surrounding
            // whitespace, accepted by both `json.loads` and `serde_json`).
            assert_eq!(
                ingest_count_json_records(py, b"{\"a\":1}\r\n{\"b\":2}\r\n".to_vec()).unwrap(),
                2u64
            );
        });
    }
}
