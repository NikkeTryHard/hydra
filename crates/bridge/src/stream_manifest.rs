//! stream_manifest: verbatim file-list digest on the shared `columnar` submodule.
//!
//! DAG: this module depends on pyo3 + `hydra-feed` ONLY (the same edge as
//! `dataset_parse.rs:4-8`, which gates vocabularies over the shard owners with
//! no new dependency; no shard/search/arrow edge, no new dependency, no file
//! IO here — the caller collects paths/sizes, this worker only sees plain
//! `(path, bytes)` pairs).
//! FORBIDS: file IO and directory walks (`build_manifest` keeps its `rglob` +
//! `stat` in Python and crosses only plain pairs), zstd framing/IO
//! (`write/read_reservoir_blob` keep their compressor + file handles in
//! Python; reservoir bytes already ride `resume.encode/decode_reservoir`),
//! scan-cache JSON/file orchestration (`load/save_scan_cache` keep their
//! `read_text`/`json.loads` + tmp+replace publish in Python; cache keying
//! stays Python per the port non-goals), shuffle-RNG live objects
//! (`serialize_shuffle_rng` keeps its `Random.getstate` in Python), the
//! sha-sort build (`build_manifest_from_paths` stays behind
//! `packet_decode.manifest_digest`, never called here), and any canon/sha
//! reimplementation (all bytes/hash math lives in `hydra-feed::manifest`,
//! referenced below, never restated).
//!
//! Frozen surface (one free function; no pyclass, no shared state, no stats
//! cell — the digest is cold path, once per run):
//! - fn `stream_manifest_digest` <- `manifest_digest`
//!   (`stream_manifest.py:149-194`): the verbatim-order file-list digest over
//!   plain `(relative-or-absolute POSIX path, compressed bytes)` pairs.
//!   Thin wrapper over `hydra_feed::manifest::manifest_digest` on an
//!   order-preserving `StreamManifest` (root is provenance display only and
//!   does not enter the digest). Compute runs detached; zero Python API
//!   inside the closure.
//!
//! Verbatim contract (mirrors the oracle docstring at `:164-168`): this
//! digest binds the stored `entry.path.as_posix()` strings VERBATIM in
//! manifest order — no sha-sort, no empty/absolute rejection. It deliberately
//! stays off `packet_decode.manifest_digest`: the packet-decode bridge digest
//! sha-sorts via `build_manifest_from_paths` and rejects empty/absolute
//! paths (`packet_decode.rs:646-658`), while this digest hashes the file list
//! exactly as stored — rerouting the file list through the packet-decode gate
//! would change every absolute-root digest. The two wrappers share the feed
//! owner (`hydra_feed::manifest::manifest_digest`) with disjoint gates, never
//! a second hasher or printer.
//!
//! Error mapping: the feed canon step is the only fallible step (ASCII
//! fast-path divergence, practically unreachable — production cross-checks
//! the fast bytes against the canon path fail-closed); it maps to
//! `PyValueError` (never `PyOSError`, never silent skip — mirroring
//! `packet.rs:43-45`). The Python translator maps it to `ContractError`.
//! The oracle has no equivalent reject (it would return a digest), so this
//! text is new hardening, never a re-pin: any divergence breaks scan-cache
//! keys loudly (raise here, miss downstream), never silently.
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its pyfunction to the caller-provided `columnar` submodule, so the
//! Python translator keeps one bridge surface (`hydra2._native.columnar`).
//! MAIN wiring (shared files, MAIN-only): `pub mod stream_manifest;` in
//! `lib.rs` (alphabetical, after `stream_driver`) plus
//! `crate::stream_manifest::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:403-406`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `stream_scan::register`
//! (`crates/bridge/src/stream_scan.rs:209-211`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `stream_scan::scan_file_games` (`crates/bridge/src/stream_scan.rs:183-203`);
//! `PyValueError` mapping of every feed reject mirrors `packet.rs:43-45`;
//! the attached `sha256:<hex>` shape gate mirrors
//! `packet_decode::manifest_digest` (`crates/bridge/src/packet_decode.rs:668-672`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bind a corpus file list for RunSpec provenance, verbatim order.
///
/// Thin wrapper over `hydra_feed::manifest::manifest_digest` (ASCII fast-path
/// with fail-closed canon cross-check, feed-owned). `files` are
/// `(root-id, POSIX path, compressed bytes)` triples in MANIFEST order (the
/// feed hashes them exactly as given — no sha-sort here; the hash binds,
/// never splits or orders). Paths ride verbatim: empty and absolute paths
/// hash (unlike `packet_decode.manifest_digest`, which rejects both
/// attached). `root` is provenance display only and does not enter the
/// digest (pinned empty).
///
/// Returns the `sha256:<hex>` digest (shape-gated attached, never defaulted).
#[pyfunction]
#[pyo3(signature = (files,))]
fn stream_manifest_digest(py: Python<'_>, files: Vec<(String, String, u64)>) -> PyResult<String> {
    let digest = py
        .detach(move || {
            let manifest = hydra_feed::manifest::StreamManifest {
                files: files
                    .into_iter()
                    .map(|(root_id, path, bytes)| hydra_feed::manifest::FileEntry {
                        path,
                        root_id,
                        bytes,
                    })
                    .collect(),
                root: String::new(),
            };
            hydra_feed::manifest::manifest_digest(&manifest)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !hydra_feed::partition::is_digest_text(&digest) {
        return Err(PyValueError::new_err(
            "stream_manifest digest failed the sha256:<hex> shape gate",
        ));
    }
    Ok(digest)
}

/// Attach the stream-manifest pyfunction to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `stream_scan.rs:209-211`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(stream_manifest_digest, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod stream_manifest_tests {
    use super::*;

    #[test]
    fn digest_byte_exact_vectors_from_head_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived for the `(root-id, path, bytes)` triple element
            // `{"bytes":N,"path":"P","root_id":"R"}`: empty binds `b"[]"`;
            // the ASCII singletons bind the raw-interpolated fast bytes;
            // escaped paths (quote, backslash, non-ASCII) take the
            // element-wise canon path to the same digest the oracle
            // returns (independently verified in
            // `test_manifest_digest_matches_canonical_oracle`).
            let empty: Vec<(String, String, u64)> = Vec::new();
            assert_eq!(
                stream_manifest_digest(py, empty).unwrap(),
                "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
            );
            assert_eq!(
                stream_manifest_digest(
                    py,
                    vec![("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64)]
                )
                .unwrap(),
                "sha256:7eef540635dc37dd0bff95df9e391e57d808b447a6d6b159ccefa737aa940bc3",
            );
            assert_eq!(
                stream_manifest_digest(
                    py,
                    vec![("ryu".to_string(), "z.mjai.json.zst".to_string(), 0u64)]
                )
                .unwrap(),
                "sha256:c551bff0c13e468abd92887395f9bd67729a98e2aa01bc8482c4eb736e57737d",
            );
            // Absolute paths hash verbatim (no gate here — the deliberate
            // fork from `packet_decode.manifest_digest`, which rejects).
            assert_eq!(
                stream_manifest_digest(py, vec![("ryu".to_string(), "/abs/x".to_string(), 1u64)])
                    .unwrap(),
                "sha256:852930b7532cd176bbfc1bf1732ff010409d9d1b702b4234352a05cac6f7cfaf",
            );
            // Escaped paths (canon path) are byte-exact against the oracle.
            assert_eq!(
                stream_manifest_digest(
                    py,
                    vec![("ryu".to_string(), "a\"b.mjai.json.zst".to_string(), 3u64)]
                )
                .unwrap(),
                "sha256:278c61e19de75e0aa89a72c8a9900b32addc3a1adc58d644203089413cd6d825",
            );
            assert_eq!(
                stream_manifest_digest(
                    py,
                    vec![("ryu".to_string(), "a\\b.mjai.json.zst".to_string(), 5u64)]
                )
                .unwrap(),
                "sha256:9c02860b1e6ca6af16f3c9a6e4fe3240bd5aee1f5f72e31ac5cb537e4e2f5831",
            );
            assert_eq!(
                stream_manifest_digest(
                    py,
                    vec![(
                        "ryu".to_string(),
                        "t\u{e9}nou/x.mjai.json.zst".to_string(),
                        7u64
                    )]
                )
                .unwrap(),
                "sha256:ee3e41138f517972f21aca114a2d1b3f5dc7f3171c137dae0a387033d3836de0",
            );
        });
    }

    #[test]
    fn digest_preserves_manifest_order_verbatim() {
        Python::initialize();
        Python::attach(|py| {
            // No sha-sort here (unlike `packet_decode.manifest_digest`):
            // input order is the digest order, exactly like the oracle
            // (`manifest_digest` hashes `manifest.files` as stored).
            let forward = vec![
                ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
            ];
            let backward = vec![
                ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
            ];
            assert_eq!(
                stream_manifest_digest(py, forward).unwrap(),
                "sha256:142d30ce7713578e4ab126d87e4a78f24fe52d9111a94711e274a72a039e2db8",
            );
            assert_eq!(
                stream_manifest_digest(py, backward).unwrap(),
                "sha256:b3d97971446694039d12b3b4d479099514f3a5ee217f977a21bf7761353841b1",
            );
            assert_ne!(
                stream_manifest_digest(
                    py,
                    vec![
                        ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                        ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                    ]
                )
                .unwrap(),
                stream_manifest_digest(
                    py,
                    vec![
                        ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                        ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                    ]
                )
                .unwrap(),
            );
        });
    }

    #[test]
    fn digest_output_is_shape_gated() {
        Python::initialize();
        Python::attach(|py| {
            let digest = stream_manifest_digest(
                py,
                vec![(
                    "ryu".to_string(),
                    "big.mjai.json.zst".to_string(),
                    9_007_199_254_740_991u64,
                )],
            )
            .unwrap();
            assert_eq!(
                digest,
                "sha256:d1a44dc7434d7fc431cd5a1a29d6abb9ac557375dfd32e152399817bb254d087",
            );
            assert!(hydra_feed::partition::is_digest_text(&digest));
        });
    }
}
