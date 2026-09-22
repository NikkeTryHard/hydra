//! rows_seal: thin seal-bytes boundary for `data/rows.py` over `hydra-feed`.
//!
//! DAG: this module depends on the feed crate + pyo3 + serde_json ONLY (same
//! edge as `canon_rng`; no shard/search/arrow edge, no new dependency).
//! FORBIDS: dataclass framing + contract validation (the `PackagedObjectRow` /
//! `RawObjectRow` shapes, `__post_init__` gates, manifest JSONL framing, and
//! dedup stay Python-side), file IO, and wall-clock timestamps. All JCS bytes
//! math lives in `hydra_feed::canon` (the single printer — this file never
//! reimplements JCS, mirroring `canon_rng::batch_canonical_bytes`); all hash
//! math lives in `hydra_feed::digest`.
//!
//! Concurrency pattern (matches `canon_rng.rs` + `contracts.rs`): validate
//! attached, ONE `py.detach(|| ...)` around the whole build + JCS-emit +
//! hash fold (never per-field attach), blobs wrapped attached. Frozen surface
//! only (free functions; no pyclass, no shared state, no stats cell — seal
//! calls are cold path, manifest-row granularity). GIL attestation
//! (`gil_used=false` default on PyO3 >= 0.28) is spelled once wave-wide in
//! `canon_rng.rs` and inherited here — not repeated.
//!
//! Attachment: this module does NOT create its own submodule. `register`
//! adds its pyfunctions to the caller-provided `columnar` submodule, so the
//! Python translators keep one bridge surface (`hydra2._native.columnar`).
//! MAIN wiring (shared files, MAIN-only): `pub mod rows_seal;` in `lib.rs`
//! plus `crate::rows_seal::register(&sub)?;` at the end of
//! `columnar::register` (mirroring `columnar.rs:397-404`).
//!
//! Seal semantics (mirrors `python/hydra2/data/rows.py`, pinned by goldens):
//! - Packaged seal (15 keys + optional `packaged_object_id`): digest fields
//!   cross WITH their `sha256:` prefixes (stored dataclass form) and are
//!   stripped to bare hex here (strip-once, `None` stays null). `include_id`
//!   stays for call-site compat; the self-hash seal excludes the id.
//! - Raw join seal (12 keys, `canonical_bytes_without_id`): digest texts ride
//!   VERBATIM (prefixes kept) — never strip on this side. `object_id` is the
//!   `sha256:` digest of these bytes (fold provided as `*_digest`).
//! - `acquisition_metadata` crosses as one JSON document blob (any key order;
//!   re-parsed through the feed I-JSON boundary then re-canonicalized, so
//!   wire bytes are canonical regardless). Non-string keys cannot survive
//!   JSON transport as non-strings — the Python translator gates them before
//!   crossing; dup keys / NaN / Inf / lone surrogates fail closed here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Strip exactly one `sha256:` prefix (packaged-seal bare-hex norm, mirrors
/// `PackagedObjectRow.canonical_bytes.strip`). No prefix passes through
/// untouched; empty-after-strip is preserved verbatim (framing validation
/// stays Python-side).
fn strip_digest_prefix(text: &str) -> &str {
    text.strip_prefix("sha256:").unwrap_or(text)
}

/// Fail-closed seal error mapping: every feed reject (I-JSON parse, JCS
/// emit) is a caller- or data-shape reject, so all map to `PyValueError`
/// (never `PyOSError`, never silent skip) — mirroring `packet.rs:43-45`.
fn seal_err(context: &str, detail: String) -> PyErr {
    PyValueError::new_err(format!("rows_seal {context} rejected: {detail}"))
}

/// Pure packaged-seal document builder (detach-safe: owned scalars only, no
/// interpreter interaction). Key insertion follows the Python field order for
/// readability; the feed JCS printer sorts on the wire regardless.
#[allow(clippy::too_many_arguments)]
fn packaged_seal_value(
    include_id: bool,
    packaged_object_id: &str,
    source_kind: &str,
    source_container_sha256: Option<&str>,
    source_member_path: Option<&str>,
    source_bytes_sha256: &str,
    source_bytes_length: u64,
    compressed_path: &str,
    compressed_bytes_sha256: &str,
    compressed_bytes_length: u64,
    decoded_bytes_sha256: &str,
    decoded_bytes_length: u64,
    record_count: u64,
    canonical_jsonl: bool,
    packager_identity: &str,
    packager_config_hash: &str,
    created_at_utc: &str,
) -> serde_json::Value {
    let mut doc = serde_json::Map::with_capacity(16);
    if include_id {
        doc.insert(
            "packaged_object_id".to_string(),
            serde_json::Value::String(strip_digest_prefix(packaged_object_id).to_string()),
        );
    }
    doc.insert(
        "source_kind".to_string(),
        serde_json::Value::String(source_kind.to_string()),
    );
    doc.insert(
        "source_container_sha256".to_string(),
        source_container_sha256.map_or(serde_json::Value::Null, |v| {
            serde_json::Value::String(strip_digest_prefix(v).to_string())
        }),
    );
    doc.insert(
        "source_member_path".to_string(),
        source_member_path.map_or(serde_json::Value::Null, |v| {
            serde_json::Value::String(v.to_string())
        }),
    );
    doc.insert(
        "source_bytes_sha256".to_string(),
        serde_json::Value::String(strip_digest_prefix(source_bytes_sha256).to_string()),
    );
    doc.insert(
        "source_bytes_length".to_string(),
        serde_json::Value::from(source_bytes_length),
    );
    doc.insert(
        "compressed_path".to_string(),
        serde_json::Value::String(compressed_path.to_string()),
    );
    doc.insert(
        "compressed_bytes_sha256".to_string(),
        serde_json::Value::String(strip_digest_prefix(compressed_bytes_sha256).to_string()),
    );
    doc.insert(
        "compressed_bytes_length".to_string(),
        serde_json::Value::from(compressed_bytes_length),
    );
    doc.insert(
        "decoded_bytes_sha256".to_string(),
        serde_json::Value::String(strip_digest_prefix(decoded_bytes_sha256).to_string()),
    );
    doc.insert(
        "decoded_bytes_length".to_string(),
        serde_json::Value::from(decoded_bytes_length),
    );
    doc.insert(
        "record_count".to_string(),
        serde_json::Value::from(record_count),
    );
    doc.insert(
        "canonical_jsonl".to_string(),
        serde_json::Value::Bool(canonical_jsonl),
    );
    doc.insert(
        "packager_identity".to_string(),
        serde_json::Value::String(strip_digest_prefix(packager_identity).to_string()),
    );
    doc.insert(
        "packager_config_hash".to_string(),
        serde_json::Value::String(strip_digest_prefix(packager_config_hash).to_string()),
    );
    doc.insert(
        "created_at_utc".to_string(),
        serde_json::Value::String(created_at_utc.to_string()),
    );
    serde_json::Value::Object(doc)
}

/// Pure raw-join-seal document builder (detach-safe). Digest texts are
/// VERBATIM here — the packaged-side strip MUST NOT leak across (golden
/// ex4/ex5 pin the `sha256:`-prefixed join bytes).
#[allow(clippy::too_many_arguments)]
fn raw_join_seal_value(
    packaged_object_id: &str,
    confidential_source_id: &str,
    authorization_attestation_id: &str,
    permitted_purpose: &[String],
    disclosure_class: &str,
    acquisition_metadata: serde_json::Value,
    semantic_state: &str,
    semantic_validation_hash: Option<&str>,
    first_error_class: Option<&str>,
    first_error_event_index: Option<i64>,
    parent_ids: &[String],
    created_at_utc: &str,
) -> serde_json::Value {
    let mut doc = serde_json::Map::with_capacity(12);
    doc.insert(
        "packaged_object_id".to_string(),
        serde_json::Value::String(packaged_object_id.to_string()),
    );
    doc.insert(
        "confidential_source_id".to_string(),
        serde_json::Value::String(confidential_source_id.to_string()),
    );
    doc.insert(
        "authorization_attestation_id".to_string(),
        serde_json::Value::String(authorization_attestation_id.to_string()),
    );
    doc.insert(
        "permitted_purpose".to_string(),
        serde_json::Value::Array(
            permitted_purpose
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    );
    doc.insert(
        "disclosure_class".to_string(),
        serde_json::Value::String(disclosure_class.to_string()),
    );
    doc.insert("acquisition_metadata".to_string(), acquisition_metadata);
    doc.insert(
        "semantic_state".to_string(),
        serde_json::Value::String(semantic_state.to_string()),
    );
    doc.insert(
        "semantic_validation_hash".to_string(),
        semantic_validation_hash.map_or(serde_json::Value::Null, |v| {
            serde_json::Value::String(v.to_string())
        }),
    );
    doc.insert(
        "first_error_class".to_string(),
        first_error_class.map_or(serde_json::Value::Null, |v| {
            serde_json::Value::String(v.to_string())
        }),
    );
    doc.insert(
        "first_error_event_index".to_string(),
        first_error_event_index.map_or(serde_json::Value::Null, serde_json::Value::from),
    );
    doc.insert(
        "parent_ids".to_string(),
        serde_json::Value::Array(
            parent_ids
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    );
    doc.insert(
        "created_at_utc".to_string(),
        serde_json::Value::String(created_at_utc.to_string()),
    );
    serde_json::Value::Object(doc)
}

/// Attached kind gate (mirrors the dataclass `__post_init__` invariant so a
/// direct bridge caller cannot mint a seal for an invalid kind).
fn check_source_kind(source_kind: &str) -> PyResult<()> {
    if source_kind == "raw" || source_kind == "archive_member" || source_kind == "precompressed" {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "rows_seal packaged_seal: source_kind invalid {source_kind:?}"
        )))
    }
}

/// Attached state gate (mirrors the `RawObjectRow.__post_init__` invariant).
fn check_semantic_state(semantic_state: &str) -> PyResult<()> {
    if semantic_state == "unvalidated"
        || semantic_state == "valid"
        || semantic_state == "quarantined"
    {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "rows_seal raw_join_seal: semantic_state invalid {semantic_state:?}"
        )))
    }
}

/// Packaged seal bytes (mirrors `PackagedObjectRow.canonical_bytes`): owned
/// scalars in, JCS seal bytes out. JCS emit runs through the single feed
/// printer (`hydra_feed::canon::canonical_bytes_value` — the exact entry
/// `canon_rng::batch_canonical_bytes` uses, see `canon_rng.rs:388`), so
/// parity holds by construction. One detach around build + emit; bytes
/// wrapped attached (mirrors `contracts.rs:434-436` detach + `448-456`
/// `PyBytes` return).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn packaged_seal_bytes(
    py: Python<'_>,
    include_id: bool,
    packaged_object_id: String,
    source_kind: String,
    source_container_sha256: Option<String>,
    source_member_path: Option<String>,
    source_bytes_sha256: String,
    source_bytes_length: u64,
    compressed_path: String,
    compressed_bytes_sha256: String,
    compressed_bytes_length: u64,
    decoded_bytes_sha256: String,
    decoded_bytes_length: u64,
    record_count: u64,
    canonical_jsonl: bool,
    packager_identity: String,
    packager_config_hash: String,
    created_at_utc: String,
) -> PyResult<Py<PyBytes>> {
    check_source_kind(&source_kind)?;
    let out = py
        .detach(|| -> Result<Vec<u8>, String> {
            let doc = packaged_seal_value(
                include_id,
                &packaged_object_id,
                &source_kind,
                source_container_sha256.as_deref(),
                source_member_path.as_deref(),
                &source_bytes_sha256,
                source_bytes_length,
                &compressed_path,
                &compressed_bytes_sha256,
                compressed_bytes_length,
                &decoded_bytes_sha256,
                decoded_bytes_length,
                record_count,
                canonical_jsonl,
                &packager_identity,
                &packager_config_hash,
                &created_at_utc,
            );
            hydra_feed::canon::canonical_bytes_value(&doc, "bridge:packaged_seal_bytes")
                .map_err(|e| format!("canon JCS emit rejected: {e:?}"))
        })
        .map_err(|detail| seal_err("packaged_seal_bytes", detail))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Packaged seal digest (mirrors `verify_seal`): the `sha256:` digest of the
/// id-excluded (caller omits it) packaged seal bytes in ONE detach — JCS
/// emit through the single printer, hash through the feed digest owner
/// (`hydra_feed::digest::sha256_hex`, infallible `sha256:`-prefixed text —
/// same entry `canon_rng::batch_sha256` folds, see `canon_rng.rs:228`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn packaged_seal_digest(
    py: Python<'_>,
    include_id: bool,
    packaged_object_id: String,
    source_kind: String,
    source_container_sha256: Option<String>,
    source_member_path: Option<String>,
    source_bytes_sha256: String,
    source_bytes_length: u64,
    compressed_path: String,
    compressed_bytes_sha256: String,
    compressed_bytes_length: u64,
    decoded_bytes_sha256: String,
    decoded_bytes_length: u64,
    record_count: u64,
    canonical_jsonl: bool,
    packager_identity: String,
    packager_config_hash: String,
    created_at_utc: String,
) -> PyResult<String> {
    check_source_kind(&source_kind)?;
    py.detach(|| -> Result<String, String> {
        let doc = packaged_seal_value(
            include_id,
            &packaged_object_id,
            &source_kind,
            source_container_sha256.as_deref(),
            source_member_path.as_deref(),
            &source_bytes_sha256,
            source_bytes_length,
            &compressed_path,
            &compressed_bytes_sha256,
            compressed_bytes_length,
            &decoded_bytes_sha256,
            decoded_bytes_length,
            record_count,
            canonical_jsonl,
            &packager_identity,
            &packager_config_hash,
            &created_at_utc,
        );
        let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "bridge:packaged_seal_digest")
            .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
        Ok(hydra_feed::digest::sha256_hex(&bytes))
    })
    .map_err(|detail| seal_err("packaged_seal_digest", detail))
}

/// Raw join seal bytes (mirrors `RawObjectRow.canonical_bytes_without_id`):
/// owned scalars + one metadata JSON blob in, JCS join-seal bytes out. The
/// blob is re-parsed through the feed I-JSON boundary
/// (`hydra_feed::canon::parse_canonical_bytes` — the exact entry
/// `canon_rng::of_canonical_json` parses with, see `canon_rng.rs:212`) then
/// embedded; key order on the wire is canonical regardless of blob order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn raw_join_seal_bytes(
    py: Python<'_>,
    packaged_object_id: String,
    confidential_source_id: String,
    authorization_attestation_id: String,
    permitted_purpose: Vec<String>,
    disclosure_class: String,
    acquisition_metadata_json: Vec<u8>,
    semantic_state: String,
    semantic_validation_hash: Option<String>,
    first_error_class: Option<String>,
    first_error_event_index: Option<i64>,
    parent_ids: Vec<String>,
    created_at_utc: String,
) -> PyResult<Py<PyBytes>> {
    check_semantic_state(&semantic_state)?;
    let out = py
        .detach(|| -> Result<Vec<u8>, String> {
            let metadata = hydra_feed::canon::parse_canonical_bytes(
                &acquisition_metadata_json,
                "bridge:raw_join_seal_bytes:acquisition_metadata",
            )
            .map_err(|e| format!("metadata parse rejected: {e:?}"))?;
            let doc = raw_join_seal_value(
                &packaged_object_id,
                &confidential_source_id,
                &authorization_attestation_id,
                &permitted_purpose,
                &disclosure_class,
                metadata,
                &semantic_state,
                semantic_validation_hash.as_deref(),
                first_error_class.as_deref(),
                first_error_event_index,
                &parent_ids,
                &created_at_utc,
            );
            hydra_feed::canon::canonical_bytes_value(&doc, "bridge:raw_join_seal_bytes")
                .map_err(|e| format!("canon JCS emit rejected: {e:?}"))
        })
        .map_err(|detail| seal_err("raw_join_seal_bytes", detail))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Raw join object id (mirrors `_raw_object_id_for`): the `sha256:` digest
/// of the join seal bytes in ONE detach (parse + JCS + hash, zero Python API
/// inside the closure).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn raw_join_seal_digest(
    py: Python<'_>,
    packaged_object_id: String,
    confidential_source_id: String,
    authorization_attestation_id: String,
    permitted_purpose: Vec<String>,
    disclosure_class: String,
    acquisition_metadata_json: Vec<u8>,
    semantic_state: String,
    semantic_validation_hash: Option<String>,
    first_error_class: Option<String>,
    first_error_event_index: Option<i64>,
    parent_ids: Vec<String>,
    created_at_utc: String,
) -> PyResult<String> {
    check_semantic_state(&semantic_state)?;
    py.detach(|| -> Result<String, String> {
        let metadata = hydra_feed::canon::parse_canonical_bytes(
            &acquisition_metadata_json,
            "bridge:raw_join_seal_digest:acquisition_metadata",
        )
        .map_err(|e| format!("metadata parse rejected: {e:?}"))?;
        let doc = raw_join_seal_value(
            &packaged_object_id,
            &confidential_source_id,
            &authorization_attestation_id,
            &permitted_purpose,
            &disclosure_class,
            metadata,
            &semantic_state,
            semantic_validation_hash.as_deref(),
            first_error_class.as_deref(),
            first_error_event_index,
            &parent_ids,
            &created_at_utc,
        );
        let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "bridge:raw_join_seal_digest")
            .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
        Ok(hydra_feed::digest::sha256_hex(&bytes))
    })
    .map_err(|detail| seal_err("raw_join_seal_digest", detail))
}

/// Attach the seal pyfunctions to the caller-provided `columnar` submodule
/// (mirrors the `sub.add_function(wrap_pyfunction!(..., &sub)?)?` shape at
/// `columnar.rs:400-402`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(packaged_seal_bytes, sub)?)?;
    sub.add_function(wrap_pyfunction!(packaged_seal_digest, sub)?)?;
    sub.add_function(wrap_pyfunction!(raw_join_seal_bytes, sub)?)?;
    sub.add_function(wrap_pyfunction!(raw_join_seal_digest, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod rows_seal_tests {
    use super::*;

    const HEX64: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn packaged_value() -> serde_json::Value {
        packaged_seal_value(
            false,
            &format!("sha256:{HEX64}"),
            "raw",
            None,
            None,
            &format!("sha256:{HEX64}"),
            1234,
            "obj/aa/bb.zst",
            &format!("sha256:{HEX64}"),
            567,
            &format!("sha256:{HEX64}"),
            890,
            2,
            true,
            &format!("sha256:{HEX64}"),
            &format!("sha256:{HEX64}"),
            "2026-01-02T03:04:05Z",
        )
    }

    #[test]
    fn strip_applies_once_and_leaves_bare() {
        assert_eq!(strip_digest_prefix(&format!("sha256:{HEX64}")), HEX64);
        assert_eq!(strip_digest_prefix(HEX64), HEX64);
        assert_eq!(strip_digest_prefix("sha256:sha256:abc"), "sha256:abc");
    }

    #[test]
    fn packaged_doc_strips_digests_and_nulls_missing() {
        let doc = packaged_value();
        // 15 keys without the id (id excluded by flag).
        assert_eq!(doc.as_object().map_or(0, serde_json::Map::len), 15);
        assert!(doc.get("packaged_object_id").is_none());
        assert_eq!(
            doc.get("source_bytes_sha256")
                .and_then(serde_json::Value::as_str),
            Some(HEX64)
        );
        assert!(
            doc.get("source_container_sha256")
                .is_some_and(|v| v.is_null())
        );
        assert!(doc.get("source_member_path").is_some_and(|v| v.is_null()));
        assert_eq!(
            doc.get("canonical_jsonl"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn packaged_doc_includes_stripped_id_when_asked() {
        let mut doc = packaged_value();
        assert!(doc.get("packaged_object_id").is_none());
        doc = packaged_seal_value(
            true,
            &format!("sha256:{HEX64}"),
            "raw",
            None,
            None,
            &format!("sha256:{HEX64}"),
            0,
            "p",
            &format!("sha256:{HEX64}"),
            0,
            &format!("sha256:{HEX64}"),
            0,
            0,
            false,
            &format!("sha256:{HEX64}"),
            &format!("sha256:{HEX64}"),
            "2026-01-02T03:04:05Z",
        );
        assert_eq!(
            doc.get("packaged_object_id")
                .and_then(serde_json::Value::as_str),
            Some(HEX64)
        );
        assert_eq!(doc.as_object().map_or(0, serde_json::Map::len), 16);
    }

    #[test]
    fn raw_join_doc_keeps_digests_verbatim() {
        let meta = serde_json::json!({"origin": "packager-x"});
        let doc = raw_join_seal_value(
            &format!("sha256:{HEX64}"),
            "src",
            "att",
            &["research".to_string()],
            "internal",
            meta,
            "valid",
            Some(&format!("sha256:{HEX64}")),
            None,
            None,
            &[],
            "2026-01-03T00:00:00Z",
        );
        // Verbatim: the join side MUST keep the prefix (packaged side strips).
        let want = format!("sha256:{HEX64}");
        assert_eq!(
            doc.get("packaged_object_id")
                .and_then(serde_json::Value::as_str),
            Some(want.as_str())
        );
        assert_eq!(doc.as_object().map_or(0, serde_json::Map::len), 12);
        assert!(doc.get("first_error_class").is_some_and(|v| v.is_null()));
    }

    #[test]
    fn seal_bytes_are_deterministic_and_id_sensitive() {
        let a = hydra_feed::canon::canonical_bytes_value(&packaged_value(), "t:a").unwrap();
        let b = hydra_feed::canon::canonical_bytes_value(&packaged_value(), "t:b").unwrap();
        assert_eq!(a, b);
        let with_id = packaged_seal_value(
            true,
            &format!("sha256:{HEX64}"),
            "raw",
            None,
            None,
            &format!("sha256:{HEX64}"),
            1234,
            "obj/aa/bb.zst",
            &format!("sha256:{HEX64}"),
            567,
            &format!("sha256:{HEX64}"),
            890,
            2,
            true,
            &format!("sha256:{HEX64}"),
            &format!("sha256:{HEX64}"),
            "2026-01-02T03:04:05Z",
        );
        let c = hydra_feed::canon::canonical_bytes_value(&with_id, "t:c").unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn kind_and_state_gates_fail_closed() {
        assert!(check_source_kind("raw").is_ok());
        assert!(check_source_kind("bogus").is_err());
        assert!(check_semantic_state("valid").is_ok());
        assert!(check_semantic_state("bogus").is_err());
    }
}
