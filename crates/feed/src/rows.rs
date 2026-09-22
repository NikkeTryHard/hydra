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

use crate::rows_fields::{get_opt_str, get_str, get_u64, norm_digest, require_digest, strip_digest};

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
// ---------------------------------------------------------------------------
// Seal-replay (Wave5 B3 migration step 3 — JCS-only re-mint + replay)
// ---------------------------------------------------------------------------
//
// Migration-step-3 entry points: re-mint packaged manifests through the NEW
// (JCS) printer only, re-derive raw joins, and replay the full ingest →
// decode → validate → quarantine chain over a corpus dir. Python stays the
// primary printer until cutover; these helpers never read caller-ordered
// bytes (there is no old-printer path here — inputs arrive as field values
// and ids are recomputed, never translated by comparison).
//
// Single-printer rule: every seal id minted below goes through `feed::canon`
// (serde_jcs 0.2.0 exact) + `feed::digest` (SHA-256 ONLY) — the same two
// owners the verify path calls. Manifest *framing* (JSONL lines) is
// transport, not identity: callers serialize reminted rows with
// `feed::canon` (JCS) and parse with `serde_json` (order-insensitive, same
// as the Python loader). `serde_json` below is parse + line-count only; it
// never emits seal bytes.
//
// Fail-closed: single-row entries return `Err` (never a silent skip);
// [`replay_corpus`] never aborts mid-corpus — every divergence lands in
// [`SealReplayReport::mismatches`] / [`seal_failures`](SealReplayReport::seal_failures)
// and the driver (`scripts/seal_replay.py`) exits non-zero unless the
// report [`is_green`](SealReplayReport::is_green). Quarantined or invalid
// games are *successful* replays (the quarantine entry ran), not failures —
// only structural and digest/count divergences fail.

/// Corpus replay report: counts plus fail-closed detail lists.
///
/// Field names mirror the `scripts/seal_replay.py` report JSON exactly.
/// Green = [`is_green`](SealReplayReport::is_green) (both lists empty).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SealReplayReport {
    /// Rows successfully re-minted JCS-only.
    pub reminted: usize,
    /// Raw joins successfully re-derived from the reminted rows.
    pub rejoined: usize,
    /// Non-seal divergences: bad JSON/shape, `DuplicatePath`, container
    /// rule (via remint), compressed/decoded hash or length divergence,
    /// `record_count` divergence, missing artifacts (full mode), decode hard
    /// errors, IO. Each entry is `context: detail`.
    pub mismatches: Vec<String>,
    /// Seal verification failures observed during replay.
    pub seal_failures: Vec<String>,
}

impl SealReplayReport {
    /// Empty (untested) report.
    pub fn empty() -> Self {
        SealReplayReport {
            reminted: 0,
            rejoined: 0,
            mismatches: Vec::new(),
            seal_failures: Vec::new(),
        }
    }

    /// Green gate: no divergence of any kind recorded.
    pub fn is_green(&self) -> bool {
        self.mismatches.is_empty() && self.seal_failures.is_empty()
    }

    /// Route one replay error into its bucket: post-mint seal mismatches
    /// go to `seal_failures`, everything else to `mismatches`.
    fn record_replay_error(&mut self, context: &str, err: &ReplayError) {
        match err {
            ReplayError::Row(RowsError::SealMismatch { .. }) => {
                self.seal_failures.push(format!("{context}: {err}"));
            }
            _ => {
                self.mismatches.push(format!("{context}: {err}"));
            }
        }
    }
}

/// Fail-closed replay error for the single-row entry points.
///
/// `Row` reuses the transport taxonomy (seal/duplicate/container/shape);
/// the rest mirror the ingest byte gates (`ingest.py:116-142`
/// `CorruptArtifactError` family) as data, never panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// Transport-row failure (parse/verify/dedup).
    Row(RowsError),
    /// Compressed payload fails to decompress or misses decoded
    /// hash/length expectations.
    Framer { detail: String },
    /// `sha256(compressed)` or byte length differs from the manifest row.
    ArtifactMismatch { detail: String },
    /// Parsed JSON-line count differs from `record_count`.
    CountMismatch { detail: String },
    /// Quarantine construction failed (missing outcome — never a skip).
    Quarantine { detail: String },
    /// Corpus file unreadable.
    Io { detail: String },
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayError::Row(e) => write!(f, "{e}"),
            ReplayError::Framer { detail } => write!(f, "replay: framer decode ({detail})"),
            ReplayError::ArtifactMismatch { detail } => {
                write!(f, "replay: artifact mismatch ({detail})")
            }
            ReplayError::CountMismatch { detail } => {
                write!(f, "replay: record_count divergence ({detail})")
            }
            ReplayError::Quarantine { detail } => write!(f, "replay: quarantine ({detail})"),
            ReplayError::Io { detail } => write!(f, "replay: io ({detail})"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<RowsError> for ReplayError {
    fn from(e: RowsError) -> Self {
        ReplayError::Row(e)
    }
}

impl From<crate::framer::FramerError> for ReplayError {
    fn from(e: crate::framer::FramerError) -> Self {
        ReplayError::Framer {
            detail: e.to_string(),
        }
    }
}

impl From<crate::quarantine::QuarantineErr> for ReplayError {
    fn from(e: crate::quarantine::QuarantineErr) -> Self {
        ReplayError::Quarantine {
            detail: e.to_string(),
        }
    }
}

impl From<std::io::Error> for ReplayError {
    fn from(e: std::io::Error) -> Self {
        ReplayError::Io {
            detail: e.to_string(),
        }
    }
}

/// Replay attestation for re-derived joins (synthetic, deterministic).
///
/// Corpus packaged rows carry no attestations, so replay joins share one
/// fixed spec. Same packaged rows ⇒ same join ids. The nested
/// `acquisition_metadata` deliberately pins the nested-JCS fork surface
/// (BTreeMap + serializer sort, `rows.rs:541-557` vs `rows.py:313-328`
/// insertion order). Production joins use real attestations, never this.
#[derive(Debug, Clone)]
pub struct ReplayJoinSpec {
    /// Confidential source id (opaque, non-empty).
    pub confidential_source_id: String,
    /// Authorization attestation id (mandatory).
    pub authorization_attestation_id: String,
    /// Permitted purposes (non-empty).
    pub permitted_purpose: Vec<String>,
    /// Disclosure class (non-empty).
    pub disclosure_class: String,
    /// Acquisition metadata (nested keys exercise JCS nested sort).
    pub acquisition_metadata: BTreeMap<String, serde_json::Value>,
}

impl Default for ReplayJoinSpec {
    fn default() -> Self {
        ReplayJoinSpec {
            confidential_source_id: "seal-replay-src".to_string(),
            authorization_attestation_id: "seal-replay-att".to_string(),
            permitted_purpose: vec!["replay".to_string()],
            disclosure_class: "replay".to_string(),
            acquisition_metadata: BTreeMap::from([(
                "replay".to_string(),
                serde_json::json!({"b": 2, "a": 1}),
            )]),
        }
    }
}

/// Single-object replay outcome (ingest → decode → validate → quarantine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayIngestOutcome {
    /// Validation verdict (`false` when undecodable or invalid — both still
    /// build exactly one quarantine record).
    pub valid: bool,
    /// Decoded games (0 when undecodable, 1 otherwise — one game per
    /// object, `decode.py:49-135`).
    pub games: usize,
    /// Quarantine records built (0 when valid, 1 otherwise).
    pub quarantined: usize,
    /// Validation seal when valid.
    pub validation_hash: Option<String>,
}

/// Mint the JCS seal id for one validated row (NEW form only).
fn mint_jcs_id(row: &PackagedObjectRow) -> Result<String, RowsError> {
    let bytes = row.canonical_bytes_without_id()?;
    Ok(crate::digest::sha256_hex(&bytes))
}

/// Re-mint packaged manifests JCS-only (migration step 3).
///
/// Reads each manifest path (blank lines skipped, exactly like
/// `load_packaged_manifest`, `rows.py:242-261`), parses every row with
/// [`parse_packaged_row`] (exact 16-key set, bare-hex norm, container
/// rule), DISCARDS the stored `packaged_object_id` (old caller-ordered
/// mint — Rust holds no old printer, so stored seals are never compared),
/// recomputes it over JCS bytes, re-validates, verifies the new seal, and
/// rejects duplicate `compressed_path` per manifest. Fail-closed:
/// `DuplicatePath` / container / shape errors are `Err` (never a silent
/// skip); the mint-then-verify pair makes a `SealMismatch` escape a hard
/// internal error, not a quiet fork.
pub fn remint_manifest_jcs(
    paths: &[std::path::PathBuf],
) -> Result<Vec<PackagedObjectRow>, ReplayError> {
    let mut out = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|e| ReplayError::Io {
            detail: format!("cannot read {}: {e}", path.display()),
        })?;
        let mut file_rows = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value =
                serde_json::from_str(line).map_err(|e| ReplayError::Io {
                    detail: format!("{} line {} invalid JSON: {e}", path.display(), idx.saturating_add(1)),
                })?;
            let parsed = parse_packaged_row(&value)?;
            let id = mint_jcs_id(&parsed)?;
            let reminted = PackagedObjectRow::new(
                id,
                parsed.source_kind,
                parsed.source_container_sha256,
                parsed.source_member_path,
                parsed.source_bytes_sha256,
                parsed.source_bytes_length,
                parsed.compressed_path,
                parsed.compressed_bytes_sha256,
                parsed.compressed_bytes_length,
                parsed.decoded_bytes_sha256,
                parsed.decoded_bytes_length,
                parsed.record_count,
                parsed.canonical_jsonl,
                parsed.packager_identity,
                parsed.packager_config_hash,
                parsed.created_at_utc,
            )?;
            reminted.verify_seal()?;
            file_rows.push(reminted);
        }
        check_no_duplicate_paths(&file_rows)?;
        out.extend(file_rows);
    }
    Ok(out)
}

/// Build one replay join input from a reminted row (timestamp reuses the
/// packaged row so replays are deterministic — never wall-clock).
fn replay_join_input(row: &PackagedObjectRow, spec: &ReplayJoinSpec) -> RawJoinInput {
    RawJoinInput {
        packaged_object_id: row.packaged_object_id.clone(),
        confidential_source_id: spec.confidential_source_id.clone(),
        authorization_attestation_id: spec.authorization_attestation_id.clone(),
        permitted_purpose: spec.permitted_purpose.clone(),
        disclosure_class: spec.disclosure_class.clone(),
        acquisition_metadata: spec.acquisition_metadata.clone(),
        semantic_state: "unvalidated".to_string(),
        semantic_validation_hash: None,
        first_error_class: None,
        first_error_event_index: None,
        parent_ids: Vec::new(),
        created_at_utc: row.created_at_utc.clone(),
    }
}

/// Re-derive raw joins JCS-only (migration step 3).
///
/// One [`make_raw_object_row`] per join input (attestation presence +
/// digest shapes validated; id derived via `feed::canon` + `feed::digest`
/// — never a local printer or hasher). The packaged row is never mutated
/// by the join.
pub fn rederive_joins(rows: &[RawJoinInput]) -> Result<Vec<RawObjectRow>, ReplayError> {
    let mut out = Vec::with_capacity(rows.len());
    for input in rows {
        out.push(make_raw_object_row(input.clone())?);
    }
    Ok(out)
}

/// Replay one packaged object end to end (migration step 3 entry point).
///
/// Mirrors `ingest_packaged_objects` (`ingest.py:103-150`): seal verify →
/// compressed hash/length → verified zstd decode (binds decoded hash and
/// length) → `record_count` check (non-blank JSON-parsable lines, same
/// lenient count as `:128-142`) → decode (`decode.py:49-135`,
/// one-game-per-object) → validate (engine probe stands in as
/// [`AdapterProbe::ok`](crate::validate::AdapterProbe::ok), exactly like
/// the `valid_game_checks_order_and_seal` KAT) → quarantine. Invalid games
/// route to quarantine records (success); only structural and
/// digest/count divergences are `Err`.
pub fn replay_ingest_bytes(
    packaged: &PackagedObjectRow,
    joined: &RawObjectRow,
    compressed: &[u8],
) -> Result<ReplayIngestOutcome, ReplayError> {
    packaged.verify_seal()?;
    if crate::digest::sha256_hex(compressed) != packaged.compressed_bytes_sha256 {
        return Err(ReplayError::ArtifactMismatch {
            detail: format!(
                "compressed hash mismatch for {}",
                packaged.compressed_path
            ),
        });
    }
    if compressed.len() as u64 != packaged.compressed_bytes_length {
        return Err(ReplayError::ArtifactMismatch {
            detail: format!(
                "compressed length mismatch for {}: got {} want {}",
                packaged.compressed_path,
                compressed.len() as u64,
                packaged.compressed_bytes_length
            ),
        });
    }
    let decoded = crate::framer::decode_zstd_verified(
        compressed,
        Some(packaged.decoded_bytes_sha256.as_str()),
        Some(packaged.decoded_bytes_length),
    )?;
    let mut parsed_count: u64 = 0;
    for line in decoded.split(|b| *b == b'\n') {
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        if serde_json::from_slice::<serde_json::Value>(line).is_ok() {
            parsed_count = parsed_count.saturating_add(1);
        }
    }
    if parsed_count != packaged.record_count {
        return Err(ReplayError::CountMismatch {
            detail: format!(
                "record_count divergence for {}: manifest {} vs decoded {parsed_count}",
                packaged.compressed_path, packaged.record_count
            ),
        });
    }
    let raw = crate::quarantine::RawRow {
        object_id: joined.object_id.clone(),
        packaged_object_id: joined.packaged_object_id.clone(),
        confidential_source_id: joined.confidential_source_id.clone(),
        authorization_attestation_id: joined.authorization_attestation_id.clone(),
        permitted_purpose: joined.permitted_purpose.clone(),
        parent_ids: joined.parent_ids.clone(),
    };
    match crate::decode::decode_game_object(
        &joined.object_id,
        &packaged.packaged_object_id,
        &decoded,
    ) {
        Err(e) => {
            let _record = crate::quarantine::QuarantinedRecord::decode_failure(
                &raw,
                e.error_class(),
                e.event_index(),
            );
            Ok(ReplayIngestOutcome {
                valid: false,
                games: 0,
                quarantined: 1,
                validation_hash: None,
            })
        }
        Ok(rec) => {
            let outcome =
                crate::validate::validate_game(&rec, &crate::validate::AdapterProbe::ok());
            if outcome.valid {
                Ok(ReplayIngestOutcome {
                    valid: true,
                    games: 1,
                    quarantined: 0,
                    validation_hash: outcome.validation_hash.clone(),
                })
            } else {
                let morsel = crate::quarantine::MorselOutcome {
                    object_id: raw.object_id.as_str(),
                    outcome: &outcome,
                    game_events: Some(rec.events.len()),
                };
                let records = crate::quarantine::quarantine_invalid(
                    std::slice::from_ref(&raw),
                    std::slice::from_ref(&morsel),
                )?;
                Ok(ReplayIngestOutcome {
                    valid: false,
                    games: 1,
                    quarantined: records.len(),
                    validation_hash: outcome.validation_hash.clone(),
                })
            }
        }
    }
}

/// Seal-only vs full replay switch.
///
/// Full mode (default) resolves every row's `compressed_path` and replays
/// bytes; a missing artifact is a `mismatches` entry (fail-closed, never a
/// silent skip). Seal-only mode replays seals + joins for manifest-only
/// corpora (real manifests passed as paths whose artifacts live
/// elsewhere) — artifact absence is out of scope there, not a divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayOptions {
    /// Resolve + replay compressed artifacts (default `true`).
    pub require_artifacts: bool,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        ReplayOptions {
            require_artifacts: true,
        }
    }
}

/// Collect `*.manifest.jsonl` files under `dir` (sorted, deterministic).
fn collect_manifest_files(dir: &std::path::Path) -> (Vec<std::path::PathBuf>, Vec<String>) {
    let mut files = Vec::new();
    let mut problems = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(top) = stack.pop() {
        match std::fs::read_dir(&top) {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(e) => {
                            let path = e.path();
                            if path.is_dir() {
                                stack.push(path);
                            } else if path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|n| n.ends_with(".manifest.jsonl"))
                            {
                                files.push(path);
                            }
                        }
                        Err(e) => problems.push(format!(
                            "io: bad dir entry under {}: {e}",
                            top.display()
                        )),
                    }
                }
            }
            Err(e) => problems.push(format!("io: cannot list {}: {e}", top.display())),
        }
    }
    files.sort();
    (files, problems)
}

/// Replay one manifest file: parse → remint → rejoin → ingest per row.
///
/// Returns the reminted rows (for JCS-manifest writers); every divergence
/// lands in `report`. Artifact bytes resolve against `artifact_root`
/// (the manifest's canonical parent, so symlinked real manifests replay
/// without vendoring).
fn replay_manifest_file(
    path: &std::path::Path,
    artifact_root: &std::path::Path,
    spec: &ReplayJoinSpec,
    opts: &ReplayOptions,
    report: &mut SealReplayReport,
) -> Vec<PackagedObjectRow> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            report
                .mismatches
                .push(format!("io: cannot read {}: {e}", path.display()));
            return Vec::new();
        }
    };
    let mut values = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => values.push(v),
            Err(e) => report.mismatches.push(format!(
                "json: {} line {}: {e}",
                path.display(),
                idx.saturating_add(1)
            )),
        }
    }
    let parsed: Vec<PackagedObjectRow> = match values
        .iter()
        .map(parse_packaged_row)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(rows) => rows,
        Err(e) => {
            report
                .mismatches
                .push(format!("remint: {}: {e}", path.display()));
            return Vec::new();
        }
    };
    if let Err(e) = check_no_duplicate_paths(&parsed) {
        report
            .mismatches
            .push(format!("remint: {}: {e}", path.display()));
        return Vec::new();
    }
    let mut reminted = Vec::with_capacity(parsed.len());
    for row in &parsed {
        match mint_jcs_id(row).and_then(|id| {
            PackagedObjectRow::new(
                id,
                row.source_kind.clone(),
                row.source_container_sha256.clone(),
                row.source_member_path.clone(),
                row.source_bytes_sha256.clone(),
                row.source_bytes_length,
                row.compressed_path.clone(),
                row.compressed_bytes_sha256.clone(),
                row.compressed_bytes_length,
                row.decoded_bytes_sha256.clone(),
                row.decoded_bytes_length,
                row.record_count,
                row.canonical_jsonl,
                row.packager_identity.clone(),
                row.packager_config_hash.clone(),
                row.created_at_utc.clone(),
            )
        }) {
            Ok(fresh) => {
                if let Err(e) = fresh.verify_seal() {
                    report.record_replay_error(
                        &format!("remint: {}", path.display()),
                        &ReplayError::Row(e),
                    );
                } else {
                    reminted.push(fresh);
                }
            }
            Err(e) => report.record_replay_error(
                &format!("remint: {}", path.display()),
                &ReplayError::Row(e),
            ),
        }
    }
    report.reminted += reminted.len();
    let inputs: Vec<RawJoinInput> = reminted
        .iter()
        .map(|row| replay_join_input(row, spec))
        .collect();
    let joins = match rederive_joins(&inputs) {
        Ok(j) => j,
        Err(e) => {
            report.record_replay_error(&format!("join: {}", path.display()), &e);
            return reminted;
        }
    };
    report.rejoined += joins.len();
    for (row, joined) in reminted.iter().zip(joins.iter()) {
        let candidate = std::path::Path::new(&row.compressed_path);
        let resolved: std::path::PathBuf = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            artifact_root.join(candidate)
        };
        if !resolved.is_file() {
            if opts.require_artifacts {
                report.mismatches.push(format!(
                    "artifact-missing: {} resolves to {} (no silent skip)",
                    row.compressed_path,
                    resolved.display()
                ));
            }
            continue;
        }
        match std::fs::read(&resolved) {
            Ok(bytes) => {
                if let Err(e) = replay_ingest_bytes(row, joined, &bytes) {
                    report.record_replay_error(
                        &format!("ingest: {}", row.compressed_path),
                        &e,
                    );
                }
            }
            Err(e) => report.mismatches.push(format!(
                "io: cannot read {}: {e}",
                resolved.display()
            )),
        }
    }
    reminted
}

/// Replay a corpus dir: every `*.manifest.jsonl` → parse/remint/rejoin +
/// byte gates where artifacts resolve (migration step 3).
///
/// Files visit in sorted order (deterministic reports). Joins re-derive
/// under [`ReplayJoinSpec::default`] (synthetic replay attestation with
/// nested metadata). An empty corpus (no manifests, or no rows) is a
/// `mismatches` entry — silent empty replay is refused like the
/// empty-ingest refusal (`ingest.py:105-106`). Returns the aggregated
/// [`SealReplayReport`]; green iff
/// [`is_green`](SealReplayReport::is_green).
pub fn replay_corpus(dir: &std::path::Path) -> SealReplayReport {
    replay_corpus_with_options(dir, &ReplayOptions::default())
}

/// [`replay_corpus`] with an explicit [`ReplayOptions`] (seal-only mode
/// for manifest-only corpora).
pub fn replay_corpus_with_options(
    dir: &std::path::Path,
    opts: &ReplayOptions,
) -> SealReplayReport {
    let mut report = SealReplayReport::empty();
    let (files, problems) = collect_manifest_files(dir);
    report.mismatches.extend(problems);
    if files.is_empty() {
        report.mismatches.push(format!(
            "corpus {} holds no *.manifest.jsonl; refusing silent empty replay",
            dir.display()
        ));
        return report;
    }
    let spec = ReplayJoinSpec::default();
    for manifest in &files {
        let root: std::path::PathBuf = match std::fs::canonicalize(manifest) {
            Ok(abs) => match abs.parent() {
                Some(p) => p.to_path_buf(),
                None => dir.to_path_buf(),
            },
            Err(_) => dir.to_path_buf(),
        };
        replay_manifest_file(manifest, &root, &spec, opts, &mut report);
    }
    if report.reminted == 0 && report.is_green() {
        report.mismatches.push(format!(
            "corpus {} holds no rows; refusing silent empty replay",
            dir.display()
        ));
    }
    report
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

    // -- Seal-replay KATs (B3 migration step 3) --
    //
    // Synthetic e2e corpus: 3 objects (raw + archive_member +
    // precompressed) with real zstd bytes, placeholder stored ids (the old
    // caller-ordered mint is never compared — Rust holds no old printer),
    // staged under a unique temp dir per test (isolated, suite-safe;
    // best-effort cleanup).

    static REPLAY_TMP_COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    fn replay_tmpdir(tag: &str) -> std::path::PathBuf {
        let n = REPLAY_TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("hydra-seal-replay-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn replay_game_payload(tag: &str) -> Vec<u8> {
        format!(
            "{{\"type\":\"start_game\",\"game_id\":\"{tag}\"}}\n{{\"type\":\"dahai\",\"pai\":5}}\n{{\"type\":\"end_game\"}}\n"
        )
        .into_bytes()
    }

    fn zstd_of(raw: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        let mut enc = zstd::Encoder::new(Vec::new(), 3).unwrap();
        enc.write_all(raw).unwrap();
        enc.finish().unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_manifest_value(
        source_kind: &str,
        container_sha: Option<String>,
        member_path: Option<String>,
        source_sha: String,
        source_len: u64,
        compressed_path: String,
        compressed_sha: String,
        compressed_len: u64,
        decoded_sha: String,
        decoded_len: u64,
        record_count: u64,
    ) -> serde_json::Value {
        let opt = |o: Option<String>| o.map_or(serde_json::Value::Null, serde_json::Value::String);
        serde_json::json!({
            "packaged_object_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "source_kind": source_kind,
            "source_container_sha256": opt(container_sha),
            "source_member_path": opt(member_path),
            "source_bytes_sha256": source_sha,
            "source_bytes_length": source_len,
            "compressed_path": compressed_path,
            "compressed_bytes_sha256": compressed_sha,
            "compressed_bytes_length": compressed_len,
            "decoded_bytes_sha256": decoded_sha,
            "decoded_bytes_length": decoded_len,
            "record_count": record_count,
            "canonical_jsonl": true,
            "packager_identity": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
            "packager_config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
            "created_at_utc": "2024-01-01T00:00:00Z",
        })
    }

    fn write_manifest_lines(
        dir: &std::path::Path,
        name: &str,
        values: &[serde_json::Value],
    ) -> std::path::PathBuf {
        let mut text = String::new();
        for v in values {
            text.push_str(&serde_json::to_string(v).unwrap());
            text.push('\n');
        }
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    /// Three synthetic objects (raw + archive_member + precompressed) with
    /// real zstd artifacts; returns staged manifest values (artifacts
    /// written into `dir`).
    fn synthetic_values(tag: &str, dir: &std::path::Path) -> Vec<serde_json::Value> {
        let kinds: [(&str, Option<(&str, &str)>); 3] = [
            ("raw", None),
            (
                "archive_member",
                Some((
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "inner/m1.jsonl",
                )),
            ),
            ("precompressed", None),
        ];
        let mut values = Vec::new();
        for (i, (kind, container)) in kinds.iter().enumerate() {
            let payload = replay_game_payload(&format!("{tag}-{i}"));
            let hex = crate::digest::sha256_hex(&payload);
            let len = payload.len() as u64;
            let compressed = zstd_of(&payload);
            let compressed_sha = crate::digest::sha256_hex(&compressed);
            let compressed_len = compressed.len() as u64;
            let rel = format!("obj-{i}.mjai.json.zst");
            std::fs::write(dir.join(&rel), &compressed).unwrap();
            let lines = payload
                .split(|b| *b == b'\n')
                .filter(|l| !l.iter().all(|b| b.is_ascii_whitespace()))
                .count() as u64;
            let (copt, mopt) = match container {
                Some((c, m)) => (Some((*c).to_string()), Some((*m).to_string())),
                None => (None, None),
            };
            values.push(replay_manifest_value(
                kind,
                copt,
                mopt,
                hex.clone(),
                len,
                rel,
                compressed_sha,
                compressed_len,
                hex,
                len,
                lines,
            ));
        }
        values
    }

    fn stage_synthetic_corpus(tag: &str) -> std::path::PathBuf {
        let dir = replay_tmpdir(tag);
        let values = synthetic_values(tag, &dir);
        write_manifest_lines(&dir, "corpus.manifest.jsonl", &values);
        dir
    }

    #[test]
    fn seal_replay_green_report_on_synthetic_three_object_corpus() {
        let dir = stage_synthetic_corpus("green");
        let report = replay_corpus(&dir);
        assert!(
            report.mismatches.is_empty(),
            "mismatches: {:?}",
            report.mismatches
        );
        assert!(
            report.seal_failures.is_empty(),
            "seal failures: {:?}",
            report.seal_failures
        );
        assert_eq!(report.reminted, 3);
        assert_eq!(report.rejoined, 3);
        assert!(report.is_green());
        // Re-mint is deterministic: same content ⇒ same JCS ids twice, and
        // every minted row verifies (no dual print: the stored placeholder
        // never survives).
        let paths = vec![dir.join("corpus.manifest.jsonl")];
        let first = remint_manifest_jcs(&paths).unwrap();
        let second = remint_manifest_jcs(&paths).unwrap();
        assert_eq!(first.len(), 3);
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.packaged_object_id, b.packaged_object_id);
            a.verify_seal().unwrap();
        }
        // JCS-new direction still pinned through the remint path.
        assert!(first[0]
            .canonical_bytes_without_id()
            .unwrap()
            .starts_with(b"{\"canonical_jsonl\":"));
        // File-head pin: JCS-only out lines open canonical_jsonl-first,
        // parse order-insensitively, and verify green (locks the canon
        // render against a serde_json struct-order regression).
        let out_str = dir.to_str().unwrap().to_owned();
        let (_, out_path, n) = hook_write_manifest(&out_str, &paths[0], &first);
        assert_eq!(n, 3);
        let rendered = std::fs::read_to_string(&out_path).unwrap();
        let head = rendered.lines().next().unwrap();
        assert!(
            head.starts_with("{\"canonical_jsonl\":"),
            "JCS manifest head must be canonical_jsonl-first, got {:?}",
            &head[..head.len().min(48)]
        );
        for line in rendered.lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            parse_packaged_row(&value).unwrap().verify_seal().unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seal_replay_count_divergence_fail_closed() {
        let dir = stage_synthetic_corpus("count");
        // Corrupt the manifest count (payload still carries 3 lines).
        let manifest = dir.join("corpus.manifest.jsonl");
        let text = std::fs::read_to_string(&manifest).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        let mut first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        first
            .as_object_mut()
            .unwrap()
            .insert("record_count".to_string(), serde_json::json!(999u64));
        lines[0] = serde_json::to_string(&first).unwrap();
        std::fs::write(&manifest, lines.join("\n") + "\n").unwrap();
        let report = replay_corpus(&dir);
        assert!(!report.is_green());
        assert_eq!(report.reminted, 3);
        assert_eq!(report.rejoined, 3);
        assert!(report.seal_failures.is_empty());
        assert_eq!(report.mismatches.len(), 1);
        assert!(
            report.mismatches[0].contains("count"),
            "unexpected: {:?}",
            report.mismatches
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seal_replay_structural_errors_fail_closed() {
        // Duplicate compressed_path fails at remint (DuplicatePath).
        let dir = replay_tmpdir("dup");
        let payload = replay_game_payload("dup");
        let sha = crate::digest::sha256_hex(&payload);
        let enc = zstd_of(&payload);
        let csha = crate::digest::sha256_hex(&enc);
        let v = replay_manifest_value(
            "raw",
            None,
            None,
            sha.clone(),
            payload.len() as u64,
            "dup.mjai.json.zst".to_string(),
            csha,
            enc.len() as u64,
            sha,
            payload.len() as u64,
            3,
        );
        let manifest = write_manifest_lines(&dir, "dup.manifest.jsonl", &[v.clone(), v]);
        assert!(matches!(
            remint_manifest_jcs(std::slice::from_ref(&manifest)),
            Err(ReplayError::Row(RowsError::DuplicatePath { .. }))
        ));
        let report = replay_corpus(&dir);
        assert!(!report.is_green());
        assert!(
            report.mismatches.iter().any(|m| m.contains("duplicate")),
            "unexpected: {:?}",
            report.mismatches
        );
        let _ = std::fs::remove_dir_all(&dir);
        // archive_member without container/member fails (BadContainer).
        let bad = replay_manifest_value(
            "archive_member",
            None,
            None,
            "sha256:".to_string() + &"b".repeat(64),
            200,
            "arc.mjai.json.zst".to_string(),
            "sha256:".to_string() + &"c".repeat(64),
            160,
            "sha256:".to_string() + &"d".repeat(64),
            2000,
            4,
        );
        let dir2 = replay_tmpdir("container");
        let manifest2 =
            write_manifest_lines(&dir2, "c.manifest.jsonl", std::slice::from_ref(&bad));
        assert!(matches!(
            remint_manifest_jcs(std::slice::from_ref(&manifest2)),
            Err(ReplayError::Row(RowsError::BadContainer { .. }))
        ));
        let report2 = replay_corpus(&dir2);
        assert!(!report2.is_green());
        assert!(
            report2.mismatches.iter().any(|m| m.contains("container")),
            "unexpected: {:?}",
            report2.mismatches
        );
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn seal_replay_tampered_seal_routes_to_seal_failures() {
        // A minted row tampered post-mint fails closed as SealMismatch
        // through the ingest entry, and the report routes it to
        // seal_failures (not mismatches).
        let row = sealable_row();
        let mut tampered = row.clone();
        tampered.record_count += 1;
        let spec = ReplayJoinSpec::default();
        let input = RawJoinInput {
            packaged_object_id: row.packaged_object_id.clone(),
            confidential_source_id: spec.confidential_source_id.clone(),
            authorization_attestation_id: spec.authorization_attestation_id.clone(),
            permitted_purpose: spec.permitted_purpose.clone(),
            disclosure_class: spec.disclosure_class.clone(),
            acquisition_metadata: spec.acquisition_metadata.clone(),
            semantic_state: "unvalidated".to_string(),
            semantic_validation_hash: None,
            first_error_class: None,
            first_error_event_index: None,
            parent_ids: vec![],
            created_at_utc: row.created_at_utc.clone(),
        };
        let joins = rederive_joins(std::slice::from_ref(&input)).unwrap();
        let err = replay_ingest_bytes(&tampered, &joins[0], b"").unwrap_err();
        assert!(matches!(
            err,
            ReplayError::Row(RowsError::SealMismatch { .. })
        ));
        let mut report = SealReplayReport::empty();
        report.record_replay_error("ingest: tampered", &err);
        assert_eq!(report.seal_failures.len(), 1);
        assert!(report.mismatches.is_empty());
        assert!(!report.is_green());
        // A count divergence routes to mismatches instead.
        let mut routed = SealReplayReport::empty();
        routed.record_replay_error(
            "ingest: x",
            &ReplayError::CountMismatch {
                detail: "d".to_string(),
            },
        );
        assert_eq!(routed.mismatches.len(), 1);
        assert!(routed.seal_failures.is_empty());
    }

    #[test]
    fn seal_replay_decode_failure_quarantines_with_lineage() {
        // Two starts ⇒ StartEnd decode failure ⇒ decode-failure quarantine
        // record carrying the re-derived lineage (never a silent skip).
        let decoded =
            b"{\"type\":\"start_game\"}\n{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n".to_vec();
        let dsha = crate::digest::sha256_hex(&decoded);
        let dlen = decoded.len() as u64;
        let enc = zstd_of(&decoded);
        let csha = crate::digest::sha256_hex(&enc);
        let clen = enc.len() as u64;
        let v = replay_manifest_value(
            "raw",
            None,
            None,
            dsha.clone(),
            dlen,
            "bad.mjai.json.zst".to_string(),
            csha,
            clen,
            dsha,
            dlen,
            3,
        );
        let dir = replay_tmpdir("decode-fail");
        let manifest =
            write_manifest_lines(&dir, "bad.manifest.jsonl", std::slice::from_ref(&v));
        let rows = remint_manifest_jcs(std::slice::from_ref(&manifest)).unwrap();
        assert_eq!(rows.len(), 1);
        let spec = ReplayJoinSpec::default();
        let input = RawJoinInput {
            packaged_object_id: rows[0].packaged_object_id.clone(),
            confidential_source_id: spec.confidential_source_id.clone(),
            authorization_attestation_id: spec.authorization_attestation_id.clone(),
            permitted_purpose: spec.permitted_purpose.clone(),
            disclosure_class: spec.disclosure_class.clone(),
            acquisition_metadata: spec.acquisition_metadata.clone(),
            semantic_state: "unvalidated".to_string(),
            semantic_validation_hash: None,
            first_error_class: None,
            first_error_event_index: None,
            parent_ids: vec![],
            created_at_utc: rows[0].created_at_utc.clone(),
        };
        let joins = rederive_joins(std::slice::from_ref(&input)).unwrap();
        let outcome = replay_ingest_bytes(&rows[0], &joins[0], &enc).unwrap();
        assert!(!outcome.valid);
        assert_eq!(outcome.games, 0);
        assert_eq!(outcome.quarantined, 1);
        // Lineage binding: the quarantine record carries the join ids.
        let raw = crate::quarantine::RawRow {
            object_id: joins[0].object_id.clone(),
            packaged_object_id: joins[0].packaged_object_id.clone(),
            confidential_source_id: joins[0].confidential_source_id.clone(),
            authorization_attestation_id: joins[0].authorization_attestation_id.clone(),
            permitted_purpose: joins[0].permitted_purpose.clone(),
            parent_ids: joins[0].parent_ids.clone(),
        };
        let rec = crate::quarantine::QuarantinedRecord::decode_failure(&raw, "start_end", None);
        assert_eq!(rec.object_id, joins[0].object_id);
        assert_eq!(rec.packaged_object_id, joins[0].packaged_object_id);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Driver hook for `scripts/seal_replay.py` (env-gated; plain pass un-set).
    ///
    /// The feed crate ships no binary, so the thin Python driver stages
    /// inputs/outputs through this hook: `SEAL_REPLAY_INPUTS`
    /// (pathsep-joined manifest files or dirs), `SEAL_REPLAY_OUT`
    /// (report.json + `*.jcs.manifest.jsonl` outputs),
    /// `SEAL_REPLAY_SEAL_ONLY=1` (skip artifact resolution). Un-set env ⇒
    /// plain pass (never fails a normal `nextest` run).
    #[test]
    fn seal_replay_driver_hook() {
        let inputs = std::env::var("SEAL_REPLAY_INPUTS").unwrap_or_default();
        if inputs.is_empty() {
            return;
        }
        let out =
            std::env::var("SEAL_REPLAY_OUT").expect("SEAL_REPLAY_OUT with SEAL_REPLAY_INPUTS");
        let seal_only =
            std::env::var("SEAL_REPLAY_SEAL_ONLY").map(|v| v == "1").unwrap_or(false);
        let opts = ReplayOptions {
            require_artifacts: !seal_only,
        };
        let spec = ReplayJoinSpec::default();
        let root_override: Option<std::path::PathBuf> = std::env::var("SEAL_REPLAY_ARTIFACT_ROOT")
            .ok()
            .map(std::path::PathBuf::from);
        let mut report = SealReplayReport::empty();
        let mut written: Vec<(String, String, usize)> = Vec::new();
        std::fs::create_dir_all(&out).unwrap();
        for input in std::env::split_paths(&inputs) {
            if input.is_file() {
                // File inputs resolve artifacts against the override when
                // set (mirrors `ingest_packaged_objects(manifest_path,
                // output_root)` — real packager manifests store bare
                // relative paths while artifacts live under the output
                // root), else against the manifest's own parent.
                let root = root_override.clone().unwrap_or_else(|| {
                    input
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                });
                let rows = replay_manifest_file(&input, &root, &spec, &opts, &mut report);
                written.push(hook_write_manifest(&out, &input, &rows));
            } else if input.is_dir() {
                let (files, problems) = collect_manifest_files(&input);
                report.mismatches.extend(problems);
                for manifest in &files {
                    let root = manifest
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| input.clone());
                    let rows = replay_manifest_file(manifest, &root, &spec, &opts, &mut report);
                    written.push(hook_write_manifest(&out, manifest, &rows));
                }
            } else {
                report
                    .mismatches
                    .push(format!("input missing: {}", input.display()));
            }
        }
        let envelope = serde_json::json!({
            "mode": if seal_only { "seal-only" } else { "full" },
            "reminted": report.reminted,
            "rejoined": report.rejoined,
            "mismatches": report.mismatches,
            "seal_failures": report.seal_failures,
            "green": report.is_green(),
            "manifests": written.iter().map(|(inp, outp, n)| serde_json::json!({
                "in": inp, "out": outp, "rows": n,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            std::path::Path::new(&out).join("report.json"),
            serde_json::to_string_pretty(&envelope).unwrap(),
        )
        .unwrap();
        assert!(report.is_green(), "seal replay red: {report:?}");
    }

    /// JCS-only manifest writer for the replay out path (migration step 3).
    ///
    /// Serializes reminted rows ONLY through `feed::canon` (serde_jcs exact,
    /// `canonical_jsonl`-first key order per the `jcs_is_new` pin) — never
    /// `serde_json::to_string` on seal rows (struct-declaration order is the
    /// old caller-ordered head and would reintroduce it). Output lines stay
    /// parseable by [`parse_packaged_row`] (order-insensitive, bare-hex
    /// tolerant) and verify under [`PackagedObjectRow::verify_seal`].
    fn hook_write_manifest(
        out_dir: &str,
        manifest: &std::path::Path,
        rows: &[PackagedObjectRow],
    ) -> (String, String, usize) {
        let file_name = manifest
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("manifest.jsonl");
        let out_name = if let Some(stem) = file_name.strip_suffix(".manifest.jsonl") {
            format!("{stem}.jcs.manifest.jsonl")
        } else {
            format!("{file_name}.jcs.manifest.jsonl")
        };
        let mut text = String::new();
        for row in rows {
            let bytes = crate::canon::canonical_bytes(row, "rows:jcs-manifest")
                .expect("JCS manifest render");
            text.push_str(core::str::from_utf8(&bytes).expect("JCS manifest UTF-8"));
            text.push('\n');
        }
        let out_path = std::path::Path::new(out_dir).join(&out_name);
        std::fs::write(&out_path, text).unwrap();
        (
            manifest.display().to_string(),
            out_path.display().to_string(),
            rows.len(),
        )
    }
}
