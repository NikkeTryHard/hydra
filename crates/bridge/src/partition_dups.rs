//! partition_dups: exact + near duplicate scans for `data/partition.py` on the shared `columnar` submodule.
//!
//! DAG: pyo3 + feed ONLY (`hydra-feed = { path = "../feed" }` already in
//! `crates/bridge/Cargo.toml:19`; no new dependency, no shard/search/arrow
//! edge, no file IO). FORBIDS: grouping/draw/digest math (packet_decode owns
//! `assign_partitions`), wall hashing (`packet.wall_hash` decides), tile
//! validation, manifest IO, and any new JCS printer. All grouping lives in
//! `hydra_feed::partition` — this file only stages plain pairs attached,
//! builds feed `GameIdentity` values, and calls the owner detached.
//!
//! Ported (frozen pure scans over plain values, oracle
//! `python/hydra2/data/partition.py:153-177`):
//! - `partition_detect_exact_duplicates` <- `detect_exact_duplicates`
//!   (:153-163): first-seen-wins scan on decoded hashes. The translator
//!   stages `[(game_id, decoded_hash)]` Python-side; the bridge builds feed
//!   identities (neutral dummy fields for the unused slots) and calls
//!   `hydra_feed::partition::detect_exact_duplicates` detached.
//! - `partition_detect_near_duplicates` <- `detect_near_duplicates`
//!   (:166-177): wall-hash scan with wall-less skip. The translator stages
//!   `[(game_id, wall_hash_or_None)]`; the bridge calls
//!   `hydra_feed::partition::detect_near_duplicates` detached (`None` skips,
//!   exactly like the oracle's `continue`).
//!
//! LOGIC staying Python: `GameIdentity` / `SplitSpec` / `SplitManifest`
//! dataclasses, `_game_identity_for` framing, `assign_partitions`
//! orchestration (packet_decode-owned), `write_split_manifest` IO.
//!
//! Owner note: grouping lives in `hydra_feed::partition::detect_*`
//! (`crates/feed/src/partition.rs:373-404`, BTreeMap first-seen-wins in input
//! order — output-identical to the oracle `dict` scan since emission follows
//! input discovery order, never map order). This file never reimplements the
//! grouping (staging + detach-call shape per `packet_decode.rs:456-521` and
//! `packet_decode.rs:611-613`). Zero consts defined here (no duplication;
//! `PARTITION_ORDER` / `DEFAULT_GROUPING_KEYS` stay feed-owned).
//!
//! Shape per fn: attached staging (pyo3 extraction to owned pairs, so
//! malformed shapes propagate native `TypeError` exactly like the oracle's
//! attribute loop) -> build owned feed structs attached -> ONE
//! `py.detach(|| ...)` over the feed owner with zero Python API inside (per
//! `canon_rng.rs:383-394`, `packet_decode.rs:611-613`) -> attached wrap as
//! pyo3 `Vec<(String, String)>` (per `eval_duplicate.rs:204-223`). The feed
//! scans are infallible, so no `ValueError` mapping exists here (only the
//! `ImportError` fail-closed gate lives Python-side).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `stream_scan::register`
//! (`crates/bridge/src/stream_scan.rs:209-212`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); feed-struct staging mirrors
//! `packet_decode::assign_partitions`
//! (`crates/bridge/src/packet_decode.rs:456-521`); detached feed call mirrors
//! `packet_decode.rs:611-613`; `Python::initialize()` + `Python::attach` in
//! tests mirror (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.columnar`
//! submodule via `register` (mirrors `stream_scan.rs:209-212`); no new entry
//! point. MAIN wiring: `pub mod partition_dups;` in `lib.rs` plus
//! `crate::partition_dups::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:403-407` +
//! `stream_scan.rs:26-30`).

use pyo3::prelude::*;

/// Build a feed identity carrying only the exact-scan signal.
///
/// Unused slots take neutral dummies (`object_id` empty, `source_id`
/// `"unknown"`, empty player list -> `["unknown"]` via `GameIdentity::new`,
/// no timestamp, no wall); the feed exact scan reads `game_id` +
/// `decoded_hash` only (`crates/feed/src/partition.rs:373-384`).
fn exact_identity(game_id: &str, decoded_hash: &str) -> hydra_feed::partition::GameIdentity {
    hydra_feed::partition::GameIdentity::new(
        game_id,
        "",
        "unknown",
        Vec::new(),
        None,
        None,
        decoded_hash,
    )
}

/// Build a feed identity carrying only the near-scan signal.
///
/// `decoded_hash` is the neutral dummy here; the feed near scan reads
/// `game_id` + `wall_hash` only (`crates/feed/src/partition.rs:390-404`).
fn near_identity(game_id: &str, wall_hash: Option<&str>) -> hydra_feed::partition::GameIdentity {
    hydra_feed::partition::GameIdentity::new(
        game_id,
        "",
        "unknown",
        Vec::new(),
        None,
        wall_hash.map(str::to_owned),
        "",
    )
}

/// First-seen-wins exact scan over plain `(game_id, decoded_hash)` pairs
/// (`partition.py:153-163`) through the feed owner, detached.
#[pyfunction]
fn partition_detect_exact_duplicates(
    py: Python<'_>,
    pairs: Vec<(String, String)>,
) -> PyResult<Vec<(String, String)>> {
    let mut owned: Vec<hydra_feed::partition::GameIdentity> = Vec::with_capacity(pairs.len());
    for (game_id, decoded_hash) in &pairs {
        owned.push(exact_identity(game_id, decoded_hash));
    }
    let out = py.detach(|| hydra_feed::partition::detect_exact_duplicates(&owned));
    Ok(out)
}

/// Wall-hash scan over plain `(game_id, wall_hash_or_None)` pairs
/// (`partition.py:166-177`) through the feed owner, detached. `None`
/// entries are skipped, exactly like the oracle's wall-less `continue`.
#[pyfunction]
fn partition_detect_near_duplicates(
    py: Python<'_>,
    pairs: Vec<(String, Option<String>)>,
) -> PyResult<Vec<(String, String)>> {
    let mut owned: Vec<hydra_feed::partition::GameIdentity> = Vec::with_capacity(pairs.len());
    for (game_id, wall_hash) in &pairs {
        owned.push(near_identity(game_id, wall_hash.as_deref()));
    }
    let out = py.detach(|| hydra_feed::partition::detect_near_duplicates(&owned));
    Ok(out)
}

/// Attach the partition duplicate scans to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `stream_scan.rs:209-212`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(partition_detect_exact_duplicates, sub)?)?;
    sub.add_function(wrap_pyfunction!(partition_detect_near_duplicates, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod dup_tests {
    use super::*;

    #[test]
    fn exact_empty_and_clean() {
        Python::initialize();
        Python::attach(|py| {
            // HEAD `partition.py:153-163`: empty input yields no pairs.
            assert!(
                partition_detect_exact_duplicates(py, Vec::new())
                    .unwrap()
                    .is_empty()
            );
            // Distinct hashes yield no pairs.
            let clean = vec![
                ("g1".to_owned(), "h1".to_owned()),
                ("g2".to_owned(), "h2".to_owned()),
            ];
            assert!(
                partition_detect_exact_duplicates(py, clean)
                    .unwrap()
                    .is_empty()
            );
        });
    }

    #[test]
    fn exact_first_seen_wins_and_anchors() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from `partition.py:157-163`: second game with a
            // repeated hash pairs against the first keeper.
            let one_dup = vec![
                ("g1".to_owned(), "h1".to_owned()),
                ("g2".to_owned(), "h1".to_owned()),
            ];
            assert_eq!(
                partition_detect_exact_duplicates(py, one_dup).unwrap(),
                vec![("g1".to_owned(), "g2".to_owned())]
            );
            // Three games sharing one hash anchor two pairs on the first.
            let trio = vec![
                ("g1".to_owned(), "h".to_owned()),
                ("g2".to_owned(), "h".to_owned()),
                ("g3".to_owned(), "h".to_owned()),
            ];
            assert_eq!(
                partition_detect_exact_duplicates(py, trio).unwrap(),
                vec![
                    ("g1".to_owned(), "g2".to_owned()),
                    ("g1".to_owned(), "g3".to_owned()),
                ]
            );
        });
    }

    #[test]
    fn near_skips_wall_less_and_anchors() {
        Python::initialize();
        Python::attach(|py| {
            // HEAD `partition.py:170-171`: wall-less games are skipped.
            let wall_less: Vec<(String, Option<String>)> =
                vec![("g1".to_owned(), None), ("g2".to_owned(), None)];
            assert!(
                partition_detect_near_duplicates(py, wall_less)
                    .unwrap()
                    .is_empty()
            );
            // Wall-less head skipped; walled repeat pairs in discovery order.
            let mixed: Vec<(String, Option<String>)> = vec![
                ("g1".to_owned(), None),
                ("g2".to_owned(), Some("w".to_owned())),
                ("g3".to_owned(), Some("w".to_owned())),
            ];
            assert_eq!(
                partition_detect_near_duplicates(py, mixed).unwrap(),
                vec![("g2".to_owned(), "g3".to_owned())]
            );
            // Distinct walls yield no pairs.
            let clean: Vec<(String, Option<String>)> = vec![
                ("g1".to_owned(), Some("w1".to_owned())),
                ("g2".to_owned(), Some("w2".to_owned())),
            ];
            assert!(
                partition_detect_near_duplicates(py, clean)
                    .unwrap()
                    .is_empty()
            );
        });
    }
}
