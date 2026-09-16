//! Transport + join rows: parse/framing authority, seal math via canon.
//!
//! Rust owner of `src/hydra2/data/rows.py` (PARTIAL — math to canon).
//! Python oracle (`rows.py:43-57` caller-ordered bytes + `rows.py:119-140`
//! normative field order) is read-only this phase and stays the frozen
//! oracle until the hash-migration re-freeze.
//!
//! Ownership (Data M1): `feed::canon` owns canonical-bytes + seal math;
//! this module owns framing/parse/transport and CALLS canon. It holds NO
//! second printer (no `json.dumps`, no `serde_json::to_string` on identity)
//! and NO second hasher (no `sha2` import — digests go through
//! `feed::digest`). There is NO `feed::shuffle` module anywhere (Data M2:
//! `feed::rng` owns ALL Philox draws; this file never shuffles).
//!
//! Hash migration (Wave5 B3 — load-bearing, read before touching seals):
//! `_canonical_row_bytes` (`rows.py:43-57`) joins `json.dumps(separators,
//! ensure_ascii=False)` fragments in CALLER-passed order, which is NOT JCS
//! (no UTF-16 sort, no ES6 number grammar, no `\b\t\n\f\r`-vs-`\uhhhh`
//! discipline). `serde` serializes structs as maps and JCS sorts ALL object
//! keys including struct fields, so Rust `serde_jcs` bytes CANNOT equal the
//! Python bytes on astral keys / `1.0`-vs-`1` / `é`-vs-`\u00e9`. The gate is
//! therefore a hash-MIGRATION phase: freeze TODAY-Python bytes as oracle →
//! Rust JCS bytes as new → re-freeze goldens with KAT-2 + seal-replay BEFORE
//! cutover; NEVER silently dual-print. This module emits the NEW (JCS) bytes
//! only; the `jcs_is_new_not_caller_order` KAT below pins that direction so
//! a future reader cannot mistake it for byte-parity.
//!
//! Preserved across the migration: digest-strip semantics (packaged seals
//! hash the bare-hex form — `rows.py:114-117` `strip()`; raw joins hash
//! verbatim — `rows.py:313-328` no strip), the `required − missing / extra`
//! key discipline (`:159-182`), bare-hex normalization at parse
//! (`:184-194`), the archive-member container/member rule (`:106-111`),
//! self-hash seals (`verify_seal` over id-excluded bytes; `object_id` over
//! id-excluded bytes), and `compressed_path` dedup (`:256-261`).

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

/// Row failure taxonomy. Mirrors `ContractError` edges in `rows.py`
/// (fail-closed, never silent coercion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowsError {
    /// JSON row is not an object.
    NotAnObject { detail: String },
    /// Required keys absent (`rows.py:177-179`).
    MissingKeys { detail: String },
    /// Unknown keys present (`rows.py:180-182`).
    ExtraKeys { detail: String },
    /// Digest field not `sha256:<64 hex>` after normalization.
    BadDigest { detail: String },
    /// Non-digest field has the wrong shape.
    BadField { detail: String },
    /// `source_kind` outside `raw | archive_member | precompressed`.
    BadSourceKind { detail: String },
    /// Container/member presence violates the `source_kind` rule.
    BadContainer { detail: String },
    /// Seal recomputation differs from the stored id.
    SealMismatch { detail: String },
    /// Duplicate `compressed_path` in one manifest (`rows.py:256-261`).
    DuplicatePath { detail: String },
    /// Canon step failed (unsafe number, non-string key, …).
    Canon { detail: String },
}

impl core::fmt::Display for RowsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RowsError::NotAnObject { detail } => write!(f, "rows: packaged row must be object ({detail})"),
            RowsError::MissingKeys { detail } => write!(f, "rows: packaged row missing keys {detail}"),
            RowsError::ExtraKeys { detail } => write!(f, "rows: packaged row extra keys {detail}"),
            RowsError::BadDigest { detail } => write!(f, "rows: bad digest ({detail})"),
            RowsError::BadField { detail } => write!(f, "rows: bad field ({detail})"),
            RowsError::BadSourceKind { detail } => write!(f, "rows: source_kind invalid ({detail})"),
            RowsError::BadContainer { detail } => write!(f, "rows: container/member rule violated ({detail})"),
            RowsError::SealMismatch { detail } => write!(f, "rows: seal mismatch ({detail})"),
            RowsError::DuplicatePath { detail } => write!(f, "rows: duplicate compressed_path ({detail})"),
            RowsError::Canon { detail } => write!(f, "rows: canonicalization failed ({detail})"),
        }
    }
}

impl std::error::Error for RowsError {}

impl From<crate::canon::CanonError> for RowsError {
    fn from(e: crate::canon::CanonError) -> Self {
        RowsError::Canon {
            detail: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Digest + field helpers (mirror rows.py:31-40 + :184-194)
// ---------------------------------------------------------------------------

/// `sha256:<64 lowercase hex>` shape check (mirrors `_require_digest`).
pub fn require_digest(text: &str, name: &str) -> Result<(), RowsError> {
    let Some(hex) = text.strip_prefix("sha256:") else {
        return Err(RowsError::BadDigest {
            detail: format!("{name} must be sha256:<64 hex>, got {text:?}"),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(RowsError::BadDigest {
            detail: format!("{name} must be sha256:<64 hex>, got {text:?}"),
        });
    }
    Ok(())
}

/// Bare-hex → `sha256:` normalization at parse (mirrors `norm`,
/// `rows.py:185-194`; pre-WP-01 compat). `None` passes through for nullable
/// fields; non-strings and malformed strings are kept as-is here and
/// rejected by the constructor validation (same two-step as Python).
fn norm_digest(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None => None,
        Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => {
            if s.starts_with("sha256:") {
                Some(s.clone())
            } else if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
                Some(format!("sha256:{s}"))
            } else {
                Some(s.clone())
            }
        }
        Some(other) => Some(other.to_string()),
    }
}

/// Strip `sha256:` for seal bytes (mirrors `PackagedObjectRow.strip`,
/// `rows.py:114-117`). Non-prefixed input passes through (validation
/// rejects it before any seal is minted).
fn strip_digest(text: &str) -> &str {
    text.strip_prefix("sha256:").unwrap_or(text)
}

// ---------------------------------------------------------------------------
// Packaged transport row (WP-00B authority)
// ---------------------------------------------------------------------------

/// WP-00B transport authority. Field order follows `rows.py:119-140` for
/// readability; JCS sorts keys on the wire (see module-doc B3 note).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackagedObjectRow {
    /// Self-hash id (`sha256:` form). Excluded from seal bytes.
    pub packaged_object_id: String,
    /// `raw | archive_member | precompressed`.
    pub source_kind: String,
    /// Container digest (only for `archive_member`).
    pub source_container_sha256: Option<String>,
    /// Member path (only for `archive_member`).
    pub source_member_path: Option<String>,
    /// Source bytes digest.
    pub source_bytes_sha256: String,
    /// Source byte length.
    pub source_bytes_length: u64,
    /// Compressed artifact path (manifest-dedup key).
    pub compressed_path: String,
    /// Compressed bytes digest.
    pub compressed_bytes_sha256: String,
    /// Compressed byte length.
    pub compressed_bytes_length: u64,
    /// Decoded bytes digest.
    pub decoded_bytes_sha256: String,
    /// Decoded byte length.
    pub decoded_bytes_length: u64,
    /// Record count.
    pub record_count: u64,
    /// Canonical-JSONL flag.
    pub canonical_jsonl: bool,
    /// Packager identity digest.
    pub packager_identity: String,
    /// Packager config digest.
    pub packager_config_hash: String,
    /// Creation timestamp (UTC, validated non-empty here).
    pub created_at_utc: String,
}

/// Seal document: id-excluded, digest-stripped form hashed by `verify_seal`.
///
/// Shape = `rows.py:119-140` minus `packaged_object_id`, digests in bare
/// hex (the `strip()` step). Serialized ONLY through `feed::canon`.
#[derive(Serialize)]
struct PackagedSealDoc<'a> {
    source_kind: &'a str,
    source_container_sha256: Option<&'a str>,
    source_member_path: Option<&'a str>,
    source_bytes_sha256: &'a str,
    source_bytes_length: u64,
    compressed_path: &'a str,
    compressed_bytes_sha256: &'a str,
    compressed_bytes_length: u64,
    decoded_bytes_sha256: &'a str,
    decoded_bytes_length: u64,
    record_count: u64,
    canonical_jsonl: bool,
    packager_identity: &'a str,
    packager_config_hash: &'a str,
    created_at_utc: &'a str,
}

impl PackagedObjectRow {
    /// Constructor validation (mirrors `__post_init__`, `rows.py:81-111`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Result<Self, RowsError> {
        require_digest(&packaged_object_id, "packaged_object_id")?;
        if !["raw", "archive_member", "precompressed"].contains(&source_kind.as_str()) {
            return Err(RowsError::BadSourceKind {
                detail: format!("source_kind invalid {source_kind:?}"),
            });
        }
        if let Some(d) = &source_container_sha256 {
            require_digest(d, "source_container_sha256")?;
        }
        require_digest(&source_bytes_sha256, "source_bytes_sha256")?;
        require_digest(&compressed_bytes_sha256, "compressed_bytes_sha256")?;
        require_digest(&decoded_bytes_sha256, "decoded_bytes_sha256")?;
        require_digest(&packager_identity, "packager_identity")?;
        require_digest(&packager_config_hash, "packager_config_hash")?;
        if compressed_path.is_empty() {
            return Err(RowsError::BadField {
                detail: "compressed_path must be non-empty string".to_string(),
            });
        }
        if created_at_utc.is_empty() {
            return Err(RowsError::BadField {
                detail: "created_at_utc must be non-empty string".to_string(),
            });
        }
        if source_kind == "archive_member" {
            if source_container_sha256.is_none() || source_member_path.is_none() {
                return Err(RowsError::BadContainer {
                    detail: "archive_member requires container and member".to_string(),
                });
            }
        } else if source_container_sha256.is_some() || source_member_path.is_some() {
            return Err(RowsError::BadContainer {
                detail: format!("{source_kind} must have no container/member"),
            });
        }
        Ok(PackagedObjectRow {
            packaged_object_id,
            source_kind,
            source_container_sha256,
            source_member_path,
            source_bytes_sha256,
            source_bytes_length,
            compressed_path,
            compressed_bytes_sha256,
            compressed_bytes_length,
            decoded_bytes_sha256,
            decoded_bytes_length,
            record_count,
            canonical_jsonl,
            packager_identity,
            packager_config_hash,
            created_at_utc,
        })
    }

    fn seal_doc(&self) -> PackagedSealDoc<'_> {
        PackagedSealDoc {
            source_kind: &self.source_kind,
            source_container_sha256: self
                .source_container_sha256
                .as_deref()
                .map(strip_digest),
            source_member_path: self.source_member_path.as_deref(),
            source_bytes_sha256: strip_digest(&self.source_bytes_sha256),
            source_bytes_length: self.source_bytes_length,
            compressed_path: &self.compressed_path,
            compressed_bytes_sha256: strip_digest(&self.compressed_bytes_sha256),
            compressed_bytes_length: self.compressed_bytes_length,
            decoded_bytes_sha256: strip_digest(&self.decoded_bytes_sha256),
            decoded_bytes_length: self.decoded_bytes_length,
            record_count: self.record_count,
            canonical_jsonl: self.canonical_jsonl,
            packager_identity: strip_digest(&self.packager_identity),
            packager_config_hash: strip_digest(&self.packager_config_hash),
            created_at_utc: &self.created_at_utc,
        }
    }

    /// Seal-input bytes: id-excluded, stripped, via `feed::canon` (M1).
    pub fn canonical_bytes_without_id(&self) -> Result<Vec<u8>, RowsError> {
        Ok(crate::canon::canonical_bytes(
            &self.seal_doc(),
            "rows:packaged-seal",
        )?)
    }

    /// Full bytes (id included, stripped form) via `feed::canon`.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RowsError> {
        #[derive(Serialize)]
        struct WithId<'a> {
            packaged_object_id: &'a str,
            #[serde(flatten)]
            rest: PackagedSealDoc<'a>,
        }
        let doc = WithId {
            packaged_object_id: strip_digest(&self.packaged_object_id),
            rest: self.seal_doc(),
        };
        Ok(crate::canon::canonical_bytes(&doc, "rows:packaged")?)
    }

    /// Self-hash check (mirrors `verify_seal`, `rows.py:143-152`):
    /// `sha256(seal bytes)` (through `feed::digest`) must equal the stored
    /// id. Bare-hex inputs normalize at parse; both forms compare equal
    /// after norm.
    pub fn verify_seal(&self) -> Result<(), RowsError> {
        let bytes = self.canonical_bytes_without_id()?;
        let expected = crate::digest::sha256_hex(&bytes);
        if strip_digest(&self.packaged_object_id) != strip_digest(&expected) {
            return Err(RowsError::SealMismatch {
                detail: format!(
                    "transport row failed self-hash for {}",
                    self.compressed_path
                ),
            });
        }
        Ok(())
    }
}

/// Required manifest keys (`rows.py:159-176`).
const PACKAGED_REQUIRED: [&str; 16] = [
    "packaged_object_id",
    "source_kind",
    "source_container_sha256",
    "source_member_path",
    "source_bytes_sha256",
    "source_bytes_length",
    "compressed_path",
    "compressed_bytes_sha256",
    "compressed_bytes_length",
    "decoded_bytes_sha256",
    "decoded_bytes_length",
    "record_count",
    "canonical_jsonl",
    "packager_identity",
    "packager_config_hash",
    "created_at_utc",
];

fn get_u64(obj: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<u64, RowsError> {
    let v = obj.get(key).ok_or_else(|| RowsError::MissingKeys {
        detail: format!("{{{key}}}"),
    })?;
    match v {
        serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
        serde_json::Value::String(s) => s.parse::<u64>().map_err(|_| RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
        _ => Err(RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
    }
}

fn get_str(obj: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<String, RowsError> {
    match obj.get(key) {
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(v) => Err(RowsError::BadField {
            detail: format!("{key} must be str, got {v}"),
        }),
        None => Err(RowsError::MissingKeys {
            detail: format!("{{{key}}}"),
        }),
    }
}

fn get_opt_str(
    obj: &BTreeMap<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>, RowsError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(RowsError::BadField {
            detail: format!("{key} must be str or null, got {v}"),
        }),
    }
}

/// Parse + normalize + validate + seal-check one packaged row.
///
/// Mirrors `_parse_packaged_row` (`rows.py:155-239`): object shape, exact
/// key set, bare-hex normalization, typed construction. Seal-checked by the
/// caller via [`PackagedObjectRow::verify_seal`] (manifest load checks).
pub fn parse_packaged_row(raw: &serde_json::Value) -> Result<PackagedObjectRow, RowsError> {
    let serde_json::Value::Object(map) = raw else {
        return Err(RowsError::NotAnObject {
            detail: "packaged row must be object".to_string(),
        });
    };
    let obj: BTreeMap<String, serde_json::Value> =
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let required: BTreeSet<&str> = PACKAGED_REQUIRED.into_iter().collect();
    let present: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    if !required.is_subset(&present) {
        let missing: Vec<&str> =
            required.difference(&present).copied().collect();
        return Err(RowsError::MissingKeys {
            detail: format!("{missing:?}"),
        });
    }
    if !present.is_subset(&required) {
        let extra: Vec<&str> =
            present.difference(&required).copied().collect();
        return Err(RowsError::ExtraKeys {
            detail: format!("{extra:?}"),
        });
    }
    let req = |key: &str| -> Result<String, RowsError> {
        let norm = norm_digest(obj.get(key));
        norm.ok_or_else(|| RowsError::MissingKeys {
            detail: format!("{{{key}}}"),
        })
    };
    let packaged_object_id = req("packaged_object_id")?;
    let source_kind = get_str(&obj, "source_kind")?;
    let container_raw = obj.get("source_container_sha256");
    let source_container_sha256 = norm_digest(container_raw);
    let source_member_path = get_opt_str(&obj, "source_member_path")?;
    let source_bytes_sha256 = req("source_bytes_sha256")?;
    let compressed_bytes_sha256 = req("compressed_bytes_sha256")?;
    let decoded_bytes_sha256 = req("decoded_bytes_sha256")?;
    let packager_identity = req("packager_identity")?;
    let packager_config_hash = req("packager_config_hash")?;
    let canonical_jsonl = match obj.get("canonical_jsonl") {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(v) => {
            return Err(RowsError::BadField {
                detail: format!("canonical_jsonl must be bool, got {v}"),
            });
        }
        None => {
            return Err(RowsError::MissingKeys {
                detail: "{canonical_jsonl}".to_string(),
            });
        }
    };
    PackagedObjectRow::new(
        packaged_object_id,
        source_kind,
        source_container_sha256,
        source_member_path,
        source_bytes_sha256,
        get_u64(&obj, "source_bytes_length")?,
        get_str(&obj, "compressed_path")?,
        compressed_bytes_sha256,
        get_u64(&obj, "compressed_bytes_length")?,
        decoded_bytes_sha256,
        get_u64(&obj, "decoded_bytes_length")?,
        get_u64(&obj, "record_count")?,
        canonical_jsonl,
        packager_identity,
        packager_config_hash,
        get_str(&obj, "created_at_utc")?,
    )
}

/// Reject duplicate `compressed_path` in one manifest load
/// (mirrors `load_packaged_manifest`, `rows.py:256-261`).
pub fn check_no_duplicate_paths(rows: &[PackagedObjectRow]) -> Result<(), RowsError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for row in rows {
        if !seen.insert(row.compressed_path.as_str()) {
            return Err(RowsError::DuplicatePath {
                detail: format!("duplicate compressed_path {:?}", row.compressed_path),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw join row (WP-04B authority)
// ---------------------------------------------------------------------------

/// WP-04B authority: join of one packaged row + one attestation
/// (`rows.py:265-282`). Identity is the join id (`object_id`); the packaged
/// row is never mutated by the join.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawObjectRow {
    /// Join id = `sha256:` over the id-excluded join bytes.
    pub object_id: String,
    /// Transport id (foreign key into the packaged manifest).
    pub packaged_object_id: String,
    /// Confidential source id (opaque, non-empty).
    pub confidential_source_id: String,
    /// Authorization attestation id (mandatory, non-empty).
    pub authorization_attestation_id: String,
    /// Permitted purposes (non-empty strings).
    pub permitted_purpose: Vec<String>,
    /// Disclosure class (non-empty).
    pub disclosure_class: String,
    /// Acquisition metadata (arbitrary JSON object, JCS-sorted on seal).
    pub acquisition_metadata: BTreeMap<String, serde_json::Value>,
    /// `unvalidated | valid | quarantined`.
    pub semantic_state: String,
    /// Semantic validation hash (nullable digest).
    pub semantic_validation_hash: Option<String>,
    /// First error class (nullable).
    pub first_error_class: Option<String>,
    /// First error event index (nullable).
    pub first_error_event_index: Option<u64>,
    /// Parent ids (digests).
    pub parent_ids: Vec<String>,
    /// Creation timestamp (UTC).
    pub created_at_utc: String,
}

/// Seal document for the join (mirrors `canonical_bytes_without_id`,
/// `rows.py:313-328`): verbatim digests (NO strip — unlike packaged),
/// `permitted_purpose`/`parent_ids` as lists. Serialized ONLY through
/// `feed::canon`.
#[derive(Serialize)]
struct RawSealDoc<'a> {
    packaged_object_id: &'a str,
    confidential_source_id: &'a str,
    authorization_attestation_id: &'a str,
    permitted_purpose: &'a [String],
    disclosure_class: &'a str,
    acquisition_metadata: &'a BTreeMap<String, serde_json::Value>,
    semantic_state: &'a str,
    semantic_validation_hash: Option<&'a str>,
    first_error_class: Option<&'a str>,
    first_error_event_index: Option<u64>,
    parent_ids: &'a [String],
    created_at_utc: &'a str,
}

/// Join inputs (everything except the derived id).
#[derive(Debug, Clone)]
pub struct RawJoinInput {
    /// Transport id.
    pub packaged_object_id: String,
    /// Confidential source id.
    pub confidential_source_id: String,
    /// Authorization attestation id (mandatory — empty is `Err`).
    pub authorization_attestation_id: String,
    /// Permitted purposes.
    pub permitted_purpose: Vec<String>,
    /// Disclosure class.
    pub disclosure_class: String,
    /// Acquisition metadata.
    pub acquisition_metadata: BTreeMap<String, serde_json::Value>,
    /// Semantic state.
    pub semantic_state: String,
    /// Semantic validation hash.
    pub semantic_validation_hash: Option<String>,
    /// First error class.
    pub first_error_class: Option<String>,
    /// First error event index.
    pub first_error_event_index: Option<u64>,
    /// Parent ids.
    pub parent_ids: Vec<String>,
    /// Creation timestamp.
    pub created_at_utc: String,
}

impl RawJoinInput {
    fn seal_doc(&self) -> RawSealDoc<'_> {
        RawSealDoc {
            packaged_object_id: &self.packaged_object_id,
            confidential_source_id: &self.confidential_source_id,
            authorization_attestation_id: &self.authorization_attestation_id,
            permitted_purpose: &self.permitted_purpose,
            disclosure_class: &self.disclosure_class,
            acquisition_metadata: &self.acquisition_metadata,
            semantic_state: &self.semantic_state,
            semantic_validation_hash: self.semantic_validation_hash.as_deref(),
            first_error_class: self.first_error_class.as_deref(),
            first_error_event_index: self.first_error_event_index,
            parent_ids: &self.parent_ids,
            created_at_utc: &self.created_at_utc,
        }
    }

    fn validate(&self) -> Result<(), RowsError> {
        require_digest(&self.packaged_object_id, "packaged_object_id")?;
        if self.confidential_source_id.is_empty() {
            return Err(RowsError::BadField {
                detail: "confidential_source_id must be non-empty str".to_string(),
            });
        }
        // Attestation presence is mandatory; missing cannot be represented
        // (mirrors `make_raw_object_row`, `rows.py:383-387`).
        if self.authorization_attestation_id.is_empty() {
            return Err(RowsError::BadField {
                detail: "authorization_attestation_id is required".to_string(),
            });
        }
        if self.permitted_purpose.is_empty()
            || self.permitted_purpose.iter().any(String::is_empty)
        {
            return Err(RowsError::BadField {
                detail: "permitted_purpose must be tuple of non-empty str".to_string(),
            });
        }
        if self.disclosure_class.is_empty() {
            return Err(RowsError::BadField {
                detail: "disclosure_class must be non-empty str".to_string(),
            });
        }
        if !["unvalidated", "valid", "quarantined"].contains(&self.semantic_state.as_str()) {
            return Err(RowsError::BadField {
                detail: format!("semantic_state invalid {:?}", self.semantic_state),
            });
        }
        if let Some(h) = &self.semantic_validation_hash {
            require_digest(h, "semantic_validation_hash")?;
        }
        for pid in &self.parent_ids {
            require_digest(pid, "parent_id")?;
        }
        if self.created_at_utc.is_empty() {
            return Err(RowsError::BadField {
                detail: "created_at_utc must be non-empty string".to_string(),
            });
        }
        Ok(())
    }
}

/// Derive the join id: `sha256:` over the id-excluded join canon bytes
/// (mirrors `_raw_object_id_for`, `rows.py:331-362`), via `feed::canon` +
/// `feed::digest` — never a local printer or hasher.
pub fn raw_object_id_for(input: &RawJoinInput) -> Result<String, RowsError> {
    input.validate()?;
    let bytes = crate::canon::canonical_bytes(&input.seal_doc(), "rows:raw-seal")?;
    Ok(crate::digest::sha256_hex(&bytes))
}

/// Join one immutable packaged row with one attestation (mirrors
/// `make_raw_object_row`, `rows.py:365-422`; never mutates packaged).
pub fn make_raw_object_row(input: RawJoinInput) -> Result<RawObjectRow, RowsError> {
    let object_id = raw_object_id_for(&input)?;
    Ok(RawObjectRow {
        object_id,
        packaged_object_id: input.packaged_object_id,
        confidential_source_id: input.confidential_source_id,
        authorization_attestation_id: input.authorization_attestation_id,
        permitted_purpose: input.permitted_purpose,
        disclosure_class: input.disclosure_class,
        acquisition_metadata: input.acquisition_metadata,
        semantic_state: input.semantic_state,
        semantic_validation_hash: input.semantic_validation_hash,
        first_error_class: input.first_error_class,
        first_error_event_index: input.first_error_event_index,
        parent_ids: input.parent_ids,
        created_at_utc: input.created_at_utc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packaged_fixture() -> serde_json::Value {
        serde_json::json!({
            "packaged_object_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "source_kind": "raw",
            "source_container_sha256": null,
            "source_member_path": null,
            "source_bytes_sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            "source_bytes_length": 100,
            "compressed_path": "a/20240101.mjai.json.zst",
            "compressed_bytes_sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
            "compressed_bytes_length": 80,
            "decoded_bytes_sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
            "decoded_bytes_length": 1000,
            "record_count": 4,
            "canonical_jsonl": true,
            "packager_identity": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
            "packager_config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
            "created_at_utc": "2024-01-01T00:00:00Z"
        })
    }

    fn sealable_row() -> PackagedObjectRow {
        // Fixed id placeholder is overwritten by the mint below; every
        // other field is the seal input.
        let base = PackagedObjectRow::new(
            "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            "raw".to_string(),
            None,
            None,
            "sha256:1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            100,
            "a/20240101.mjai.json.zst".to_string(),
            "sha256:2222222222222222222222222222222222222222222222222222222222222222".to_string(),
            80,
            "sha256:3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            1000,
            4,
            true,
            "sha256:4444444444444444444444444444444444444444444444444444444444444444".to_string(),
            "sha256:5555555555555555555555555555555555555555555555555555555555555555".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        )
        .unwrap();
        let bytes = base.canonical_bytes_without_id().unwrap();
        let id = crate::digest::sha256_hex(&bytes);
        PackagedObjectRow::new(
            id,
            base.source_kind,
            base.source_container_sha256,
            base.source_member_path,
            base.source_bytes_sha256,
            base.source_bytes_length,
            base.compressed_path,
            base.compressed_bytes_sha256,
            base.compressed_bytes_length,
            base.decoded_bytes_sha256,
            base.decoded_bytes_length,
            base.record_count,
            base.canonical_jsonl,
            base.packager_identity,
            base.packager_config_hash,
            base.created_at_utc,
        )
        .unwrap()
    }

    #[test]
    fn seal_helpers_call_canon_and_verify() {
        // M1: seal bytes come from feed::canon (recomputing through canon
        // directly agrees), digests through feed::digest; verify passes on
        // the minted row and fails on any tamper.
        let row = sealable_row();
        row.verify_seal().unwrap();
        let direct = crate::canon::canonical_bytes(&row.seal_doc(), "rows:test").unwrap();
        assert_eq!(direct, row.canonical_bytes_without_id().unwrap());
        let mut tampered = row.clone();
        tampered.record_count += 1;
        assert!(matches!(
            tampered.verify_seal(),
            Err(RowsError::SealMismatch { .. })
        ));
    }

    #[test]
    fn jcs_is_new_not_caller_order_b3() {
        // B3 migration pin: JCS sorts ALL object keys, so the seal bytes
        // open with the UTF-16-smallest key ("canonical_jsonl"), NOT the
        // Python normative head ("packaged_object_id", rows.py:119-140).
        // If this ever opens with packaged_object_id, someone reintroduced
        // the caller-ordered printer — reject that direction.
        let row = sealable_row();
        let bytes = row.canonical_bytes_without_id().unwrap();
        assert!(
            bytes.starts_with(b"{\"canonical_jsonl\":"),
            "JCS-new bytes must sort keys, got head {:?}",
            &bytes[..bytes.len().min(32)]
        );
        let full = row.canonical_bytes().unwrap();
        assert!(full.windows(b"\"packaged_object_id\"".len()).any(|w| w == b"\"packaged_object_id\""));
    }

    #[test]
    fn parse_rejects_missing_extra_and_bad_digests() {
        let good = packaged_fixture();
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove("record_count");
        assert!(matches!(
            parse_packaged_row(&missing),
            Err(RowsError::MissingKeys { .. })
        ));
        let mut extra = good.clone();
        extra.as_object_mut().unwrap().insert(
            "surprise".to_string(),
            serde_json::json!(1),
        );
        assert!(matches!(
            parse_packaged_row(&extra),
            Err(RowsError::ExtraKeys { .. })
        ));
        let mut bad_digest = good.clone();
        bad_digest.as_object_mut().unwrap().insert(
            "source_bytes_sha256".to_string(),
            serde_json::json!("not-a-digest"),
        );
        assert!(matches!(
            parse_packaged_row(&bad_digest),
            Err(RowsError::BadDigest { .. })
        ));
        // Bare-hex normalizes at parse (pre-WP-01 compat, rows.py:184-194).
        let mut bare = good.clone();
        bare.as_object_mut().unwrap().insert(
            "source_bytes_sha256".to_string(),
            serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        );
        let parsed = parse_packaged_row(&bare).unwrap();
        assert_eq!(
            parsed.source_bytes_sha256,
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        // Bad source_kind + container rule fail closed.
        let mut bad_kind = good.clone();
        bad_kind.as_object_mut().unwrap().insert(
            "source_kind".to_string(),
            serde_json::json!("tarball"),
        );
        assert!(matches!(
            parse_packaged_row(&bad_kind),
            Err(RowsError::BadSourceKind { .. })
        ));
        let mut bad_container = good.clone();
        bad_container.as_object_mut().unwrap().insert(
            "source_container_sha256".to_string(),
            serde_json::json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        );
        assert!(matches!(
            parse_packaged_row(&bad_container),
            Err(RowsError::BadContainer { .. })
        ));
    }

    #[test]
    fn duplicate_paths_reject() {
        let a = sealable_row();
        let b = sealable_row();
        assert!(matches!(
            check_no_duplicate_paths(&[a, b]),
            Err(RowsError::DuplicatePath { .. })
        ));
    }

    #[test]
    fn raw_join_derives_stable_id_and_requires_attestation() {
        let input = RawJoinInput {
            packaged_object_id: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            confidential_source_id: "src-1".to_string(),
            authorization_attestation_id: "att-1".to_string(),
            permitted_purpose: vec!["train".to_string()],
            disclosure_class: "open".to_string(),
            acquisition_metadata: BTreeMap::from([(
                "source".to_string(),
                serde_json::json!("tenhou"),
            )]),
            semantic_state: "unvalidated".to_string(),
            semantic_validation_hash: None,
            first_error_class: None,
            first_error_event_index: None,
            parent_ids: vec![],
            created_at_utc: "2024-01-01T00:00:00Z".to_string(),
        };
        let id = raw_object_id_for(&input).unwrap();
        assert!(id.starts_with("sha256:"));
        let row = make_raw_object_row(input.clone()).unwrap();
        assert_eq!(row.object_id, id);
        // Deterministic: same join ⇒ same id.
        assert_eq!(raw_object_id_for(&input).unwrap(), id);
        // Missing attestation cannot be represented.
        let mut no_att = input.clone();
        no_att.authorization_attestation_id = String::new();
        assert!(make_raw_object_row(no_att).is_err());
    }
}
