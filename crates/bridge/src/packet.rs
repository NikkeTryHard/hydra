//! packet: thin framer/decode/key boundary over `hydra-feed::framer`.
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as the
//! rest of the bridge; no new dependency). FORBIDS: parse/encode/hash/pool
//! logic — arg-check + detach + struct return only. All framing, verified
//! decode, stem/group math, wall digest, and split assignment live in
//! `hydra_feed::framer` (which itself calls the partition/canon owners);
//! this file only validates arguments attached, runs the feed pure functions
//! detached, and wraps the results attached.
//!
//! Bytes path, NO DLPack/Arrow here: inputs are `Vec<u8>` (bytes/bytearray)
//! in, `(offset, end, bytes)` triples + `bytes` + text out. No PyCapsule is
//! ever constructed in this module, so the capsule-outside-detach rule is
//! N/A here (compute detached, wrap attached still holds: batch work runs
//! under `detach`, Python objects are built only while attached).
//!
//! Concurrency pattern (matches `canon_rng.rs` + replay `expand_games`):
//! validate attached, `py.detach(|| ...)` around every blocking section
//! (fan-out/join never holds the interpreter), frozen pyclasses only where
//! stateful — this module holds no state, so there are no pyclasses here.
//! GIL attestation is spelled once wave-wide in `canon_rng.rs` (`gil_used =
//! false` default on PyO3 >= 0.28 covers this submodule too).
//! Single-cdylib tree: this module registers as the `hydra2._native.packet`
//! submodule via `register`, mirroring `canon_rng::register`.
//!
//! VERIFIED feed API (read in-tree, never guessed):
//! - `hydra_feed::framer::iter_games(&[u8], u32, u64)
//!   -> Result<Vec<Frame>, FramerError>` (offsets are decompressed-byte
//!   offsets + caller `base`; `Frame { file_idx, offset, end, bytes }`).
//! - `hydra_feed::framer::fetch_game_at(&[u8], u32, u64, u64)
//!   -> Result<Frame, FramerError>` (exact-offset hit, fail-closed miss).
//! - `hydra_feed::framer::decode_zstd_verified(&[u8], Option<&str>,
//!   Option<u64>) -> Result<Vec<u8>, FramerError>` (64 KiB reads, 512 MiB
//!   guard, `sha256:<hex>` + len checks).
//! - `hydra_feed::framer::stem_of(&str) -> &str` (longest-suffix-first).
//! - `hydra_feed::framer::group_key_for(&str, &str) -> String` (ONE join).
//! - `hydra_feed::framer::group_key_for_path(&str, &str) -> String`
//!   (parent-dir source + leading digit-run time).
//! - `hydra_feed::framer::wall_hash(Option<&[u32; 136]>)
//!   -> Result<Option<String>, CanonError>` (None -> None).
//! - `hydra_feed::framer::assign_one(&str, u64, &BTreeMap<String, f64>)
//!   -> Result<String, PartitionError>` (ONE-fn split alias).
//!   Error mapping: every feed reject (framer/canon/partition) is a caller- or
//!   data-shape reject, so all map to `PyValueError` (never `PyOSError`,
//!   never silent skip).

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

/// Map a framer reject onto the Python boundary: decode/offset failures are
/// data-shape rejects (`PyValueError`), never silent skips.
fn map_framer_err(e: hydra_feed::framer::FramerError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Frame the (possibly compressed) stream into games, in file order.
///
/// Thin wrapper over `framer::iter_games`: transparent zstd/gzip/plain
/// decode, then the exact `ZstdLineStream.iter_games` port. Returns one
/// `(offset, end, bytes)` triple per game — offsets are decompressed-byte
/// offsets shifted by `base`, `bytes` is the verbatim game payload. The
/// whole frame runs detached; triples are wrapped attached.
#[pyfunction]
fn frame_games(
    py: Python<'_>,
    compressed: Vec<u8>,
    file_idx: u32,
    base: u64,
) -> PyResult<Vec<(u64, u64, Py<PyBytes>)>> {
    let frames = py
        .detach(|| hydra_feed::framer::iter_games(&compressed, file_idx, base))
        .map_err(map_framer_err)?;
    Ok(frames
        .into_iter()
        .map(|f| (f.offset, f.end, PyBytes::new(py, &f.bytes).unbind()))
        .collect())
}

/// Fetch one game by decompressed-byte offset, fail-closed.
///
/// Thin wrapper over `framer::fetch_game_at`: offsets come from
/// [`frame_games`]; an exact hit returns its `(offset, end, bytes)` triple,
/// a past-the-target scan or a miss raises `ValueError` (never the nearest
/// game, never `None`). Runs detached, wraps attached.
#[pyfunction]
fn fetch_game_at(
    py: Python<'_>,
    compressed: Vec<u8>,
    file_idx: u32,
    base: u64,
    offset: u64,
) -> PyResult<(u64, u64, Py<PyBytes>)> {
    let frame = py
        .detach(|| hydra_feed::framer::fetch_game_at(&compressed, file_idx, base, offset))
        .map_err(map_framer_err)?;
    Ok((
        frame.offset,
        frame.end,
        PyBytes::new(py, &frame.bytes).unbind(),
    ))
}

/// Fully decode one object with streaming hash/length checks.
///
/// Thin wrapper over `framer::decode_zstd_verified`: 64 KiB reads, 512 MiB
/// fail-closed guard, `expected_sha` (`sha256:<hex>`) and `expected_len`
/// (decompressed bytes) mismatch → `ValueError`. `None` expectations still
/// fully decode (magic bytes alone never authorize). Runs detached, wraps
/// attached.
#[pyfunction]
#[pyo3(signature = (compressed, expected_sha=None, expected_len=None))]
fn decode_zstd_verified(
    py: Python<'_>,
    compressed: Vec<u8>,
    expected_sha: Option<String>,
    expected_len: Option<u64>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            hydra_feed::framer::decode_zstd_verified(
                &compressed,
                expected_sha.as_deref(),
                expected_len,
            )
        })
        .map_err(map_framer_err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Game identity stem: `<stem>.mjai.json.zst` → `<stem>` (suffixes strip
/// longest-first). Trivial pure work, still detached for GIL hygiene.
#[pyfunction]
fn stem_of(py: Python<'_>, file_name: String) -> String {
    py.detach(|| hydra_feed::framer::stem_of(&file_name).to_owned())
}

/// Group key identical to the partition `(source, time)` grouping join
/// (`source|time`). Thin alias over the framer owner (ONE join string).
#[pyfunction]
fn group_key_for(py: Python<'_>, source: String, time: String) -> String {
    py.detach(|| hydra_feed::framer::group_key_for(&source, &time))
}

/// Derive the `(source, time)` group from a corpus path: source is the
/// parent directory name (empty/`.` → `"unknown"`), time is the leading
/// digit run of the stem, else `"unknown"`.
#[pyfunction]
fn group_key_for_path(py: Python<'_>, parent_dir: String, file_name: String) -> String {
    py.detach(|| hydra_feed::framer::group_key_for_path(&parent_dir, &file_name))
}

/// Derive the `(root-id, source, time)` group from a corpus path plus its
/// manifest root id: the multi-root stream/scan path. Thin alias over the
/// framer owner (the namespaced join lives in `hydra-feed` only).
#[pyfunction]
fn group_key_for_path_with_root(
    py: Python<'_>,
    root_id: String,
    parent_dir: String,
    file_name: String,
) -> String {
    py.detach(|| {
        hydra_feed::framer::group_key_for_path_with_root(&root_id, &parent_dir, &file_name)
    })
}

/// Wall hash identical to the partition identity: `None` when no wall, else
/// `sha256:` over the canon bytes of the 136-entry wall list. Shape checked
/// attached (`Some` must hold exactly 136 entries, fail-closed); the digest
/// runs detached. Canon rejects surface as `ValueError`.
#[pyfunction]
#[pyo3(signature = (wall=None))]
fn wall_hash(py: Python<'_>, wall: Option<Vec<u32>>) -> PyResult<Option<String>> {
    let arr: Option<[u32; 136]> = match wall {
        None => None,
        Some(v) => {
            if v.len() != 136 {
                return Err(PyValueError::new_err(format!(
                    "packet wall_hash needs 136 entries, got {}",
                    v.len()
                )));
            }
            let mut fixed = [0u32; 136];
            fixed.copy_from_slice(&v);
            Some(fixed)
        }
    };
    let out = py
        .detach(|| hydra_feed::framer::wall_hash(arr.as_ref()))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out)
}

/// Split assignment for one group: thin alias over the framer owner's
/// `assign_one` (wave3 §1.10 ONE-fn rule). `ratios` is a caller dict
/// (`{partition: weight}`); feed rejects (no active partitions, bad ratio
/// values) surface as `ValueError`. Runs detached.
#[pyfunction]
fn assign_one(
    py: Python<'_>,
    group_key: String,
    seed: u64,
    ratios: BTreeMap<String, f64>,
) -> PyResult<String> {
    let out = py
        .detach(|| hydra_feed::framer::assign_one(&group_key, seed, &ratios))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out)
}

/// Register the `packet` submodule (mirrors `canon_rng::register`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "packet")?;
    sub.add_function(wrap_pyfunction!(frame_games, &sub)?)?;
    sub.add_function(wrap_pyfunction!(fetch_game_at, &sub)?)?;
    sub.add_function(wrap_pyfunction!(decode_zstd_verified, &sub)?)?;
    sub.add_function(wrap_pyfunction!(stem_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(group_key_for, &sub)?)?;
    sub.add_function(wrap_pyfunction!(group_key_for_path, &sub)?)?;
    sub.add_function(wrap_pyfunction!(group_key_for_path_with_root, &sub)?)?;
    sub.add_function(wrap_pyfunction!(wall_hash, &sub)?)?;
    sub.add_function(wrap_pyfunction!(assign_one, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod packet_tests {
    use super::*;

    fn game(tag: &str) -> Vec<u8> {
        format!(
            "{{\"type\":\"start_game\",\"tag\":\"{tag}\"}}\n{{\"type\":\"dahai\",\"pai\":5}}\n{{\"type\":\"end_game\"}}\n"
        )
        .into_bytes()
    }

    #[test]
    fn frame_offsets_are_stable_decompressed_bytes() {
        // Contract: offsets are decompressed-byte offsets of each game's
        // first byte (+ caller base); concatenation re-slices exact.
        // Plain bytes exercise the same scan `iter_games` runs post-decode.
        let g0 = game("a");
        let g1 = game("b");
        let mut raw = Vec::new();
        raw.extend_from_slice(&g0);
        raw.extend_from_slice(&g1);
        Python::initialize();
        Python::attach(|py| {
            let frames = frame_games(py, raw.clone(), 2, 100).unwrap();
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0].0, 100);
            assert_eq!(frames[0].1, 100 + g0.len() as u64);
            assert_eq!(frames[1].0, 100 + g0.len() as u64);
            assert_eq!(frames[1].1, 100 + raw.len() as u64);
            assert_eq!(frames[0].2.bind(py).as_bytes(), &g0[..]);
            assert_eq!(frames[1].2.bind(py).as_bytes(), &g1[..]);
        });
    }

    #[test]
    fn fetch_game_at_hits_and_fails_closed() {
        // Contract: exact offset hits; mid-game and past-end offsets raise
        // (never the nearest game).
        let g0 = game("a");
        let g1 = game("b");
        let mut raw = Vec::new();
        raw.extend_from_slice(&g0);
        raw.extend_from_slice(&g1);
        Python::initialize();
        Python::attach(|py| {
            let hit = fetch_game_at(py, raw.clone(), 0, 0, g0.len() as u64).unwrap();
            assert_eq!(hit.0, g0.len() as u64);
            assert_eq!(hit.1, raw.len() as u64);
            assert_eq!(hit.2.bind(py).as_bytes(), &g1[..]);
            assert!(fetch_game_at(py, raw.clone(), 0, 0, 1).is_err());
            assert!(fetch_game_at(py, raw.clone(), 0, 0, raw.len() as u64 + 7).is_err());
        });
    }

    #[test]
    fn verified_decode_round_trip_and_guards() {
        // Contract: verified decode round-trips with matching sha+len;
        // wrong sha, wrong len, and corrupt bytes all fail closed. Plain
        // input covers the shared hash/guard core (magic only selects the
        // codec); the zstd-codec round-trip is pinned feed-side in
        // `framer.rs::zstd_verified_round_trip_and_guards` through the same
        // feed fn this wrapper calls.
        let raw = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n".to_vec();
        let sha = hydra_feed::digest::sha256_hex(&raw);
        Python::initialize();
        Python::attach(|py| {
            let ok =
                decode_zstd_verified(py, raw.clone(), Some(sha.clone()), Some(raw.len() as u64))
                    .unwrap();
            assert_eq!(ok.bind(py).as_bytes(), &raw[..]);
            assert!(
                decode_zstd_verified(
                    py,
                    raw.clone(),
                    Some("sha256:00".to_string()),
                    Some(raw.len() as u64)
                )
                .is_err()
            );
            assert!(decode_zstd_verified(py, raw.clone(), Some(sha.clone()), Some(1)).is_err());
            assert!(
                decode_zstd_verified(py, b"\x28\xB5\x2F\xFD corrupt".to_vec(), None, None).is_err()
            );
        });
    }

    #[test]
    fn stem_group_wall_assign_parity() {
        // Contract: thin wrappers agree with the feed owners exactly.
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                stem_of(py, "2024010112.mjai.json.zst".to_string()),
                "2024010112"
            );
            assert_eq!(
                group_key_for(py, "lobby".to_string(), "2024010112".to_string()),
                hydra_feed::framer::group_key_for("lobby", "2024010112")
            );
            assert_eq!(
                group_key_for_path(
                    py,
                    "lobby".to_string(),
                    "2024010112.mjai.json.zst".to_string()
                ),
                hydra_feed::framer::group_key_for_path("lobby", "2024010112.mjai.json.zst")
            );
            assert_eq!(
                group_key_for_path_with_root(
                    py,
                    "ryu".to_string(),
                    "lobby".to_string(),
                    "2024010112.mjai.json.zst".to_string()
                ),
                hydra_feed::framer::group_key_for_path_with_root(
                    "ryu",
                    "lobby",
                    "2024010112.mjai.json.zst"
                )
            );
            assert_eq!(wall_hash(py, None).unwrap(), None);
            let wall: Vec<u32> = (0..136).collect();
            let got = wall_hash(py, Some(wall.clone())).unwrap().unwrap();
            let list: Vec<u32> = wall;
            let bytes = hydra_feed::canon::canonical_bytes(&list, "kat").unwrap();
            assert_eq!(got, hydra_feed::partition::wall_hash_of_canon_bytes(&bytes));
            assert!(got.starts_with("sha256:"));
            assert!(wall_hash(py, Some(vec![0u32; 5])).is_err());
            let ratios =
                BTreeMap::from([("train".to_string(), 0.8), ("validation".to_string(), 0.2)]);
            assert_eq!(
                assign_one(py, "s|20240101".to_string(), 7, ratios.clone()).unwrap(),
                hydra_feed::partition::assign_one("s|20240101", 7, &ratios).unwrap()
            );
        });
    }
}
