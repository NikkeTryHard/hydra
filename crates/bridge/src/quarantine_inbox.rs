//! quarantine_inbox: thin invalid-projection + manifest-frame boundary for `data/quarantine.py`.
//!
//! DAG: this module depends on the feed crate + pyo3 + serde_json ONLY (the
//! same edge as `rows_seal`; no shard/search/arrow edge, no new dependency,
//! no file IO here — the caller keeps `game_records`, dataclass framing, and
//! the atomic publish). FORBIDS: dataclass framing + contract validation (the
//! `QuarantinedRecord` / `QuarantineManifest` shapes stay Python-side),
//! `GameRecord` event access (the caller folds `len(rec.events)` to a plain
//! `game_events` count pre-cross), and any canon/sha reimplementation (all
//! bytes/hash math lives in `hydra_feed::{canon, digest}`, referenced below,
//! never restated).
//!
//! Scope split (no dual ownership):
//! - The invalid-record projection mirrors `quarantine.py:44-87`: iterate
//!   rows in input order (output order = input order; the manifest sorts
//!   separately), skip valid outcomes allocation-free, project every invalid
//!   outcome to a slot. A row with no outcome is a hard `ValueError` whose
//!   text is owned by `hydra_feed::quarantine::QuarantineErr` (byte-identical
//!   to the oracle `RuntimeError`, which the Python translator re-raises by
//!   prefix dispatch).
//! - `error_class` crosses VERBATIM (never feed-normalized): the oracle
//!   stores `outcome.error.error_class` as-is (`:72`), while
//!   `feed::quarantine::quarantine_invalid` folds unknown classes to
//!   `"other"` — delegating to it would fork parity on ad-hoc outcomes, so
//!   this module projects plain strings and only borrows the feed error text,
//!   schema-version const, and canon/digest owners.
//! - The manifest frame mirrors `quarantine.py:90-109` MINUS the publish:
//!   records sorted by `object_id` (byte order — identical to the oracle
//!   `sorted(key=object_id)` on ASCII ids, which covers every real
//!   `sha256:<hex>` id), digest `sha256(canonical(payload))`, canonical
//!   bytes of the digest-bearing envelope out. The caller keeps
//!   `atomic_replace_bytes` + the `str` return.
//! - Lineage crosses as one JSON document blob per record (any key order;
//!   re-parsed through the feed I-JSON boundary then embedded verbatim, so
//!   ad-hoc lineage keys survive exactly like the oracle, which serializes
//!   `r.lineage` as-is). Non-string keys cannot survive JSON transport —
//!   the Python translator gates them pre-cross (mirroring
//!   `rows_seal::_metadata_json`); dup keys / NaN / Inf / lone surrogates
//!   fail closed here via the feed reject.
//!
//! Concurrency pattern (matches `packet_decode.rs` + `rows_seal.rs`):
//! validate attached, ONE `py.detach(|| ...)` around the whole project /
//! sort + frame + JCS-emit + hash fold (never per-item attach), dicts/bytes
//! wrapped attached. Frozen surface only (free functions; no pyclass, no
//! shared state, no stats cell — quarantine is cold path, batch
//! granularity). GIL attestation (`gil_used=false` default on PyO3 >= 0.28)
//! is spelled once wave-wide in `canon_rng.rs` and inherited here.
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its pyfunctions to the caller-provided `columnar` submodule, so the
//! Python translators keep one bridge surface (`hydra2._native.columnar`).
//! MAIN wiring (shared files, MAIN-only): `pub mod quarantine_inbox;` in
//! `lib.rs` (alphabetical, after `promotion`) plus
//! `crate::quarantine_inbox::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:403-407`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-489`); borrowing the
//! attached-owned vecs across `py.detach` mirrors
//! `packet_decode::decode_frames_batch` (`crates/bridge/src/packet_decode.rs:235-246`);
//! attached `PyDict` wrap + `unbind` mirrors `packet_decode.rs:250-281`;
//! `PyValueError` mapping of every feed reject mirrors `packet.rs:43-45`;
//! the attached `sha256:<hex>` shape gate mirrors
//! `packet_decode::manifest_digest` via `hydra_feed::partition::is_digest_text`
//! (`crates/bridge/src/stream_manifest.rs:98`); `PyBytes` return mirrors
//! `rows_seal::raw_join_seal_bytes` (`crates/bridge/src/rows_seal.rs:402-431`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// One projected invalid slot (detach-safe: owned scalars only, no
/// interpreter interaction). Field order follows the oracle record shape
/// (`quarantine.py:68-85`) for readability; the attached wrap names each
/// key explicitly.
struct Projected {
    object_id: String,
    packaged_object_id: String,
    game_id: String,
    error_class: String,
    error_event_index: Option<i64>,
    confidential_source_id: String,
    authorization_attestation_id: String,
    permitted_purpose: Vec<String>,
    parent_ids: Vec<String>,
    game_events: u64,
    validation_checks: Vec<(String, String)>,
    validation_hash: Option<String>,
}

/// Project quarantine slots for invalid games (mirrors
/// `quarantine.py:44-87` over plain values).
///
/// Parallel row/outcome vectors in `RawObjectRow` order (never re-sorted;
/// output order = input order). `game_events` carries the caller-folded
/// `len(rec.events)` (`0` when the game never decoded — `:81`). Rows with
/// no outcome ride as `has_outcomes[i] == false` with dummy siblings and
/// fail the whole call with the feed-owned missing-outcome text (never a
/// silent skip). Valid rows need only `has_outcomes[i] == true` +
/// `valids[i] == true` (error/checks/hash siblings ignored, never read).
///
/// Returns one dict per invalid row, in input order: `{"object_id",
/// "packaged_object_id", "game_id", "error_class", "error_event_index",
/// "confidential_source_id", "authorization_attestation_id",
/// "permitted_purpose", "parent_ids", "game_events", "validation_checks",
/// "validation_hash"}` (`validation_checks` is a dict; the translator
/// rebuilds the 8-key lineage literal from these pieces).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quarantine_project_invalid(
    py: Python<'_>,
    object_ids: Vec<String>,
    packaged_object_ids: Vec<String>,
    confidential_source_ids: Vec<String>,
    authorization_attestation_ids: Vec<String>,
    permitted_purposes: Vec<Vec<String>>,
    parent_ids: Vec<Vec<String>>,
    game_ids: Vec<String>,
    valids: Vec<bool>,
    has_outcomes: Vec<bool>,
    error_classes: Vec<Option<String>>,
    error_event_indices: Vec<Option<i64>>,
    game_events: Vec<u64>,
    validation_checks: Vec<Vec<(String, String)>>,
    validation_hashes: Vec<Option<String>>,
) -> PyResult<Vec<Py<PyDict>>> {
    let n = object_ids.len();
    if packaged_object_ids.len() != n
        || confidential_source_ids.len() != n
        || authorization_attestation_ids.len() != n
        || permitted_purposes.len() != n
        || parent_ids.len() != n
        || game_ids.len() != n
        || valids.len() != n
        || has_outcomes.len() != n
        || error_classes.len() != n
        || error_event_indices.len() != n
        || game_events.len() != n
        || validation_checks.len() != n
        || validation_hashes.len() != n
    {
        return Err(PyValueError::new_err(
            "quarantine_project_invalid needs parallel row/outcome vectors \
             (equal lengths; rows are RawObjectRow order, never re-sorted)",
        ));
    }
    // Detached: borrow the attached-owned vecs across the detach (no Python
    // objects cross, no clone — mirrors `packet_decode.rs:235-246`).
    let slots: Result<Vec<Projected>, String> = py.detach(|| {
        let mut out = Vec::new();
        for i in 0..n {
            if !has_outcomes[i] {
                return Err(hydra_feed::quarantine::QuarantineErr::MissingOutcome(
                    object_ids[i].clone(),
                )
                .to_string());
            }
            if valids[i] {
                continue;
            }
            let Some(error_class) = error_classes[i].clone() else {
                return Err(
                    "quarantine_project_invalid invalid outcome missing error class".to_string(),
                );
            };
            out.push(Projected {
                object_id: object_ids[i].clone(),
                packaged_object_id: packaged_object_ids[i].clone(),
                game_id: game_ids[i].clone(),
                error_class,
                error_event_index: error_event_indices[i],
                confidential_source_id: confidential_source_ids[i].clone(),
                authorization_attestation_id: authorization_attestation_ids[i].clone(),
                permitted_purpose: permitted_purposes[i].clone(),
                parent_ids: parent_ids[i].clone(),
                game_events: game_events[i],
                validation_checks: validation_checks[i].clone(),
                validation_hash: validation_hashes[i].clone(),
            });
        }
        Ok(out)
    });
    let slots = slots.map_err(PyValueError::new_err)?;
    // Attached: wrap each slot (input order preserved by construction —
    // mirrors `packet_decode.rs:247-281`).
    let mut wrapped: Vec<Py<PyDict>> = Vec::with_capacity(slots.len());
    for slot in &slots {
        let dict = PyDict::new(py);
        dict.set_item("object_id", &slot.object_id)?;
        dict.set_item("packaged_object_id", &slot.packaged_object_id)?;
        dict.set_item("game_id", &slot.game_id)?;
        dict.set_item("error_class", &slot.error_class)?;
        dict.set_item("error_event_index", slot.error_event_index)?;
        dict.set_item("confidential_source_id", &slot.confidential_source_id)?;
        dict.set_item(
            "authorization_attestation_id",
            &slot.authorization_attestation_id,
        )?;
        dict.set_item("permitted_purpose", &slot.permitted_purpose)?;
        dict.set_item("parent_ids", &slot.parent_ids)?;
        dict.set_item("game_events", slot.game_events)?;
        let checks = PyDict::new(py);
        for (name, value) in &slot.validation_checks {
            checks.set_item(name, value)?;
        }
        dict.set_item("validation_checks", checks)?;
        dict.set_item("validation_hash", &slot.validation_hash)?;
        wrapped.push(dict.unbind());
    }
    Ok(wrapped)
}

/// Frame a quarantine manifest over plain values (mirrors
/// `quarantine.py:90-109` minus the publish).
///
/// Records ride unsorted; the frame sorts indices by `object_id` (byte
/// order — identical to the oracle `sorted(key=object_id)` on ASCII ids).
/// `lineage_json` carries one JSON document blob per record (any key order;
/// re-parsed through the feed I-JSON boundary then embedded verbatim).
/// `schema_version` is the feed-owned
/// `hydra_feed::quarantine::MANIFEST_SCHEMA_VERSION` (never restated here).
///
/// Returns `(digest, canonical_bytes)`: the digest seals the digest-free
/// payload (`of_canonical`), the bytes are the canonical envelope WITH the
/// digest (the caller publishes them). The digest is shape-gated attached
/// (never defaulted).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn quarantine_manifest_frame(
    py: Python<'_>,
    object_ids: Vec<String>,
    packaged_object_ids: Vec<String>,
    game_ids: Vec<Option<String>>,
    error_classes: Vec<String>,
    error_event_indices: Vec<Option<i64>>,
    lineage_json: Vec<Vec<u8>>,
    validation_hashes: Vec<Option<String>>,
) -> PyResult<(String, Py<PyBytes>)> {
    let n = object_ids.len();
    if packaged_object_ids.len() != n
        || game_ids.len() != n
        || error_classes.len() != n
        || error_event_indices.len() != n
        || lineage_json.len() != n
        || validation_hashes.len() != n
    {
        return Err(PyValueError::new_err(
            "quarantine_manifest_frame needs parallel record vectors \
             (equal lengths; records sort by object_id inside the frame)",
        ));
    }
    let (digest, bytes): (String, Vec<u8>) = py
        .detach(|| -> Result<(String, Vec<u8>), String> {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|a, b| object_ids[*a].cmp(&object_ids[*b]));
            let mut arr = Vec::with_capacity(n);
            for i in idx {
                let lineage = hydra_feed::canon::parse_canonical_bytes(
                    &lineage_json[i],
                    "bridge:quarantine_manifest_frame:lineage",
                )
                .map_err(|e| format!("lineage parse rejected: {e:?}"))?;
                let mut o = serde_json::Map::with_capacity(7);
                o.insert(
                    "object_id".to_owned(),
                    serde_json::Value::String(object_ids[i].clone()),
                );
                o.insert(
                    "packaged_object_id".to_owned(),
                    serde_json::Value::String(packaged_object_ids[i].clone()),
                );
                o.insert(
                    "game_id".to_owned(),
                    game_ids[i]
                        .clone()
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
                o.insert(
                    "error_class".to_owned(),
                    serde_json::Value::String(error_classes[i].clone()),
                );
                o.insert(
                    "error_event_index".to_owned(),
                    error_event_indices[i]
                        .map(serde_json::Value::from)
                        .unwrap_or(serde_json::Value::Null),
                );
                o.insert("lineage".to_owned(), lineage);
                o.insert(
                    "validation_hash".to_owned(),
                    validation_hashes[i]
                        .clone()
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
                arr.push(serde_json::Value::Object(o));
            }
            let mut payload = serde_json::Map::with_capacity(2);
            payload.insert(
                "schema_version".to_owned(),
                serde_json::Value::String(
                    hydra_feed::quarantine::MANIFEST_SCHEMA_VERSION.to_owned(),
                ),
            );
            payload.insert("quarantined".to_owned(), serde_json::Value::Array(arr));
            let digest =
                hydra_feed::digest::of_canonical(&serde_json::Value::Object(payload.clone()))
                    .map_err(|e| format!("manifest digest rejected: {e:?}"))?;
            payload.insert(
                "digest".to_owned(),
                serde_json::Value::String(digest.clone()),
            );
            let bytes = hydra_feed::canon::canonical_bytes_value(
                &serde_json::Value::Object(payload),
                "bridge:quarantine_manifest_frame",
            )
            .map_err(|e| format!("manifest canon emit rejected: {e:?}"))?;
            Ok((digest, bytes))
        })
        .map_err(PyValueError::new_err)?;
    if !hydra_feed::partition::is_digest_text(&digest) {
        return Err(PyValueError::new_err(
            "quarantine_manifest_frame digest failed the sha256:<hex> shape gate",
        ));
    }
    Ok((digest, PyBytes::new(py, &bytes).unbind()))
}

/// Attach the quarantine pyfunctions to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `rows_seal.rs:484-489`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(quarantine_project_invalid, sub)?)?;
    sub.add_function(wrap_pyfunction!(quarantine_manifest_frame, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod quarantine_inbox_tests {
    use super::*;

    const ID_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ID_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const ID_C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ID_D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const ID_E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn lineage_blob() -> Vec<u8> {
        let lineage = serde_json::json!({
            "object_id": ID_D,
            "packaged_object_id": ID_E,
            "confidential_source_id": "src",
            "authorization_attestation_id": "att",
            "permitted_purpose": ["train", "eval"],
            "parent_ids": [ID_C],
            "game_events": 3,
            "validation_checks": {"structure": "ok", "tile_conservation": "fail"},
        });
        serde_json::to_vec(&lineage).unwrap()
    }

    #[test]
    fn schema_version_matches_oracle_literal() {
        // Frozen const pin: the oracle payload literal (`quarantine.py:92`)
        // rides the feed owner, never restated here.
        assert_eq!(hydra_feed::quarantine::MANIFEST_SCHEMA_VERSION, "1.0.0");
    }

    #[test]
    fn project_filters_valid_and_preserves_lineage_verbatim() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from HEAD (`python/hydra2/data/quarantine.py:44-87`
            // via the pre-port oracle probe): two rows, first valid (skipped),
            // second invalid (projected verbatim, `game_events: 3`).
            let slots = quarantine_project_invalid(
                py,
                vec![ID_A.to_owned(), ID_D.to_owned()],
                vec![ID_B.to_owned(), ID_E.to_owned()],
                vec!["src".to_owned(), "src".to_owned()],
                vec!["att".to_owned(), "att".to_owned()],
                vec![
                    vec!["train".to_owned(), "eval".to_owned()],
                    vec!["train".to_owned(), "eval".to_owned()],
                ],
                vec![vec![ID_C.to_owned()], vec![ID_C.to_owned()]],
                vec!["g0".to_owned(), "g1".to_owned()],
                vec![true, false],
                vec![true, true],
                vec![None, Some("tile_conservation".to_owned())],
                vec![None, Some(3)],
                vec![0, 3],
                vec![
                    vec![("structure".to_owned(), "ok".to_owned())],
                    vec![
                        ("structure".to_owned(), "ok".to_owned()),
                        ("tile_conservation".to_owned(), "fail".to_owned()),
                    ],
                ],
                vec![None, None],
            )
            .unwrap();
            assert_eq!(slots.len(), 1);
            let slot = slots[0].bind(py);
            assert_eq!(
                slot.get_item("object_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                ID_D
            );
            assert_eq!(
                slot.get_item("error_class")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "tile_conservation"
            );
            assert_eq!(
                slot.get_item("error_event_index")
                    .unwrap()
                    .unwrap()
                    .extract::<Option<i64>>()
                    .unwrap(),
                Some(3)
            );
            assert_eq!(
                slot.get_item("game_events")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                3
            );
            // Verbatim passthrough (never feed-normalized): an ad-hoc class
            // survives as-is, exactly like the oracle `:72`.
            let slots = quarantine_project_invalid(
                py,
                vec![ID_D.to_owned()],
                vec![ID_E.to_owned()],
                vec!["src".to_owned()],
                vec!["att".to_owned()],
                vec![vec!["train".to_owned()]],
                vec![vec![ID_C.to_owned()]],
                vec!["g1".to_owned()],
                vec![false],
                vec![true],
                vec![Some("bogus_class".to_owned())],
                vec![None],
                vec![0],
                vec![Vec::new()],
                vec![None],
            )
            .unwrap();
            assert_eq!(
                slots[0]
                    .bind(py)
                    .get_item("error_class")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "bogus_class"
            );
        });
    }

    #[test]
    fn project_missing_outcome_fails_with_oracle_text() {
        Python::initialize();
        Python::attach(|py| {
            // Oracle text (`quarantine.py:61-63`, feed-owned render): the
            // translator routes this to `RuntimeError`, never silent skip.
            let err = quarantine_project_invalid(
                py,
                vec![ID_A.to_owned()],
                vec![ID_B.to_owned()],
                vec!["src".to_owned()],
                vec!["att".to_owned()],
                vec![vec!["train".to_owned()]],
                vec![vec![ID_C.to_owned()]],
                vec![String::new()],
                vec![true],
                vec![false],
                vec![None],
                vec![None],
                vec![0],
                vec![Vec::new()],
                vec![None],
            )
            .unwrap_err();
            assert!(err.to_string().contains(
                "missing validation outcome for sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (would be silent skip)"
            ));
        });
    }

    #[test]
    fn frame_digests_are_byte_exact_against_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from HEAD (`quarantine.py:90-109` via the pre-port
            // oracle probe): empty frame + the singleton projection above.
            let (digest, _bytes) = quarantine_manifest_frame(
                py,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            assert_eq!(
                digest,
                "sha256:c0586a5dbb8a62ccb5e801f40c8f61cb2df25670a20085e6c83364b71d9a9410"
            );
            assert!(hydra_feed::partition::is_digest_text(&digest));
            let (digest, bytes) = quarantine_manifest_frame(
                py,
                vec![ID_D.to_owned()],
                vec![ID_E.to_owned()],
                vec![Some("g1".to_owned())],
                vec!["tile_conservation".to_owned()],
                vec![Some(3)],
                vec![lineage_blob()],
                vec![None],
            )
            .unwrap();
            assert_eq!(
                digest,
                "sha256:3e9e42cc417287fd02267f88e3952d565c11d96492ff44ef3c2072e1a3cc39d1"
            );
            let bound = bytes.bind(py);
            let raw: &[u8] = bound.as_bytes();
            let doc: serde_json::Value = serde_json::from_slice(raw).unwrap();
            assert_eq!(
                doc.get("schema_version")
                    .and_then(serde_json::Value::as_str),
                Some(hydra_feed::quarantine::MANIFEST_SCHEMA_VERSION)
            );
            assert_eq!(
                doc.get("digest").and_then(serde_json::Value::as_str),
                Some(digest.as_str())
            );
            assert_eq!(
                doc.get("quarantined")
                    .and_then(serde_json::Value::as_array)
                    .map(Vec::len),
                Some(1)
            );
        });
    }
}
