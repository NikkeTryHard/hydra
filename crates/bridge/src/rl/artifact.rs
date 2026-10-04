//! Immutable canonical rollout artifact: exact SPEC12.3 row/header validation,
//! streaming single-document zstd seal, and streaming revalidation.
//!
//! Transport law (plan section 6, explicit): the compressed artifact hashes
//! the canonical DECOMPRESSED full-root bytes — one single canonical root
//! document (`header keys + rows array`, JCS) carried in ONE zstd stream.
//! The stored blob NEVER contains `digest` (hash-input form); the digest
//! lives in the seal receipt, the row metas, and the control snapshot.
//! Digest = sha256 over the streamed root bytes. Hash chains
//! (`sha256(prev || row)`) are forbidden: identical logical rows in any
//! completion order yield identical bytes and digest.
//!
//! Memory law: the 14GB expanded generation never materializes. The seal
//! runs two passes over the native tape, validating all rows before sorting
//! small metadata and independently reproducing row IDs in pass two. Native
//! workers stage at most eight rows each per bounded window; one writer emits
//! canonical order into SHA-256/zstd. Expanded generations never materialize.
//! Revalidation streams a bounded brace scanner. The 8GiB guard aborts the
//! seal loudly; partial blobs are orphans (reads always verify the digest,
//! and only the last fsynced tip is authoritative).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Write};

use serde_json::{Map, Value};

use crate::rl::trajectory::{
    ACTION_WIDTH, DIGEST_PREFIX, HISTORY_BUCKETS, TapeError, TapeHandle, bucket_history_len,
    canonical_row_doc_no_id, hex_of, parse_digest_text, row_id_of, sha256_raw, with_tape,
};

/// SPEC12.3 header digest set: the eight digest keys (schema_version is
/// added from the frozen const at seal time, never supplied by callers).
pub const HEADER_KEYS: [&str; 8] = [
    "rules_hash",
    "utility_manifest_hash",
    "observation_schema_hash",
    "action_table_hash",
    "behavior_policy_checkpoint_hash",
    "behavior_model_spec_hash",
    "source_wall_ledger_hash",
    "row_derivation_hash",
];

/// SPEC12.3 row field set (exactly these keys; `row_id` attached at seal).
pub const ROW_KEYS_NO_ID: [&str; 18] = [
    "wall_id",
    "trajectory_step",
    "game_id",
    "round_id",
    "decision_id",
    "actor",
    "observation_hash",
    "actor_observation",
    "action_table_hash",
    "legal_mask",
    "selected_action_id",
    "behavior_policy_checkpoint_hash",
    "old_selected_log_probability",
    "raw_advantage",
    "return_vector",
    "bc_reference_checkpoint_hash",
    "bc_reference_logits",
    "transition_provenance_hash",
];

/// Closed row-derivation document top-level fields (lifecycle intake
/// `row_derivation_doc`: four recipe digests plus objective/numeric/code
/// model groups). The seal binds the header hash to exactly this document
/// (sha256 over its JCS bytes) — never to an arbitrary digest.
pub const ROW_DERIVATION_KEYS: [&str; 8] = [
    "wall_recipe",
    "sampling_recipe",
    "return_recipe",
    "shuffle_recipe",
    "objective",
    "numeric",
    "code",
    "model",
];

/// Exact 19 CheckpointManifest fields (runtime `checkpoint.py:50-69`);
/// the ordinary receipt hashes the FULL JCS manifest with no exclusion.
pub const CHECKPOINT_MANIFEST_KEYS: [&str; 19] = [
    "checkpoint_version",
    "run_spec_hash",
    "model_spec_hash",
    "model_state_hash",
    "optimizer_spec_hash",
    "optimizer_state_hash",
    "scheduler_spec_hash",
    "scheduler_state_hash",
    "training_state_hash",
    "sampler_state_hash",
    "rng_state_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
    "rollout_artifact_hash",
    "parent_checkpoint_hash",
];

/// Frozen artifact schema version.
pub const ARTIFACT_SCHEMA_VERSION: &str = "1.0.0";
/// zstd stream level (matches the feed reservoir test default).
pub const ARTIFACT_ZSTD_LEVEL: i32 = 3;
/// Compressed artifact byte cap per generation (plan section 3).
pub const ARTIFACT_COMPRESSED_CAP: u64 = 8 * 1024 * 1024 * 1024;
/// Read chunk for streaming decode/scan.
pub const SCAN_CHUNK: usize = 1 << 20;

/// Closed validation context: header digests plus the two wall ledgers and
/// the objective's BC expectation (from the frozen RunSpec recipe). The
/// row-derivation document binds upstream (prepare path verifies the header
/// against `verify_row_derivation` before assembly); the seal binds every
/// row to the verified header digest.
pub struct SealContext {
    /// Header digest texts keyed by [`HEADER_KEYS`] name (all prefixed).
    pub header: BTreeMap<String, String>,
    /// Training source wall ids (complete cohort ledger).
    pub source_walls: HashSet<String>,
    /// Final-evaluation wall ids (MUST be disjoint from `source_walls`).
    pub eval_walls: HashSet<String>,
    /// True iff the frozen objective has `w_bc > 0` (BC fields present).
    pub expect_bc: bool,
}

/// One validated row ready for canonical ordering (small: no row bytes).
pub struct AdmittedRow {
    /// SPEC content id.
    pub row_id: [u8; 32],
    /// Sort key: (wall_id, game_id, trajectory_step, actor, decision_id).
    pub sort_key: (String, String, u64, u8, String),
    /// History bucket (native tape fact, bound into the replay index).
    pub t_bucket: u32,
    /// Address back into the registered tape (game index, row index).
    pub address: (usize, usize),
    /// Frozen GPU-boundary facts for the index (exact f32 values).
    pub old_lp: f32,
    pub raw_a: f32,
    pub g4: [f64; 4],
    pub v_behavior: f32,
}

/// Seal receipt: identities plus streaming sizes (no row bytes).
pub struct SealReceipt {
    pub artifact_digest: String,
    pub row_count: usize,
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
}

fn seal_err(detail: String) -> TapeError {
    TapeError(detail)
}

fn require_digest_text(doc: &Value, name: &str) -> Result<[u8; 32], TapeError> {
    let text = doc
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err(format!("row field {name} must be a digest-text string")))?;
    parse_digest_text(text)
}

fn require_bounded_str(doc: &Value, name: &str) -> Result<String, TapeError> {
    let text = doc
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err(format!("row field {name} must be a string")))?;
    if text.is_empty() || text.len() > 1024 {
        return Err(seal_err(format!("row field {name} has bad length")));
    }
    Ok(text.to_string())
}

fn require_u64_safe(doc: &Value, name: &str) -> Result<u64, TapeError> {
    let num = doc
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| seal_err(format!("row field {name} must be a nonnegative integer")))?;
    if num > hydra_feed::canon::MAX_SAFE_INTEGER as u64 {
        return Err(seal_err(format!("row field {name} exceeds JCS safe range")));
    }
    Ok(num)
}

/// Finite f64 that is exactly FP32-representable (the cross-language JCS
/// lane for frozen f32 facts: `(v as f32) as f64 == v`).
fn require_f32_lane(value: &Value, name: &str) -> Result<f32, TapeError> {
    let v = value
        .as_f64()
        .ok_or_else(|| seal_err(format!("row field {name} must be a JSON number")))?;
    if !v.is_finite() {
        return Err(seal_err(format!("row field {name} must be finite")));
    }
    let widened = f64::from(v as f32);
    if widened != v {
        return Err(seal_err(format!(
            "row field {name} is not exactly FP32-representable"
        )));
    }
    Ok(v as f32)
}

fn require_g4(value: &Value) -> Result<[f64; 4], TapeError> {
    let arr = value
        .as_array()
        .ok_or_else(|| seal_err("return_vector must be a four-element array".to_string()))?;
    if arr.len() != 4 {
        return Err(seal_err(
            "return_vector must hold all four seats".to_string(),
        ));
    }
    let mut g4 = [0.0f64; 4];
    for (i, item) in arr.iter().enumerate() {
        let v = item
            .as_f64()
            .ok_or_else(|| seal_err("return_vector entries must be JSON numbers".to_string()))?;
        if !v.is_finite() {
            return Err(seal_err("return_vector must be finite".to_string()));
        }
        g4[i] = v;
    }
    Ok(g4)
}

/// Verify the row-derivation binding for one header digest against one
/// closed derivation document (lifecycle intake `row_derivation_doc`):
/// the digest MUST equal sha256 over the document's JCS bytes — never an
/// arbitrary digest. Boundary check for prepare paths (called before the
/// header is assembled); the seal binds rows to the verified header.
/// The document carries four recipe digests plus objective/numeric/code/
/// model groups; shapes and ranges fail closed here, and the four numeric
/// behavior literals must name exactly the implemented lanes (FP32
/// log-probabilities, frozen FP32 advantages, f64 canonical returns,
/// legal-only entropy). Value semantics inside recipes stay the
/// pipeline's composition responsibility; this gate binds header to doc.
pub fn verify_row_derivation(
    header_digest: &str,
    derivation_doc: &[u8],
) -> Result<(), TapeError> {
    parse_digest_text(header_digest)?;
    if derivation_doc.is_empty() {
        return Err(seal_err("row-derivation document must be nonempty".to_string()));
    }
    let doc: Value = serde_json::from_slice(derivation_doc)
        .map_err(|e| seal_err(format!("row-derivation document is not JSON: {e}")))?;
    let map = doc
        .as_object()
        .ok_or_else(|| seal_err("row-derivation document must be an object".to_string()))?;
    if map.len() != ROW_DERIVATION_KEYS.len()
        || !ROW_DERIVATION_KEYS.iter().all(|k| map.contains_key(*k))
    {
        return Err(seal_err(
            "row-derivation document has a non-closed key set".to_string(),
        ));
    }
    for key in [
        "wall_recipe",
        "sampling_recipe",
        "return_recipe",
        "shuffle_recipe",
    ] {
        let text = map
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| seal_err(format!("row-derivation lacks {key}")))?;
        parse_digest_text(text)?;
    }
    let objective = map
        .get("objective")
        .and_then(Value::as_object)
        .ok_or_else(|| seal_err("row-derivation lacks objective".to_string()))?;
    let number = |name: &str| {
        objective
            .get(name)
            .and_then(Value::as_f64)
            .ok_or_else(|| seal_err(format!("row-derivation objective lacks {name}")))
    };
    let clip_eps = number("clip_eps")?;
    let eps_std = number("eps_std")?;
    let w_value = number("w_value")?;
    let w_bc = number("w_bc")?;
    let alpha = number("alpha")?;
    if !(0.0 < clip_eps && clip_eps < 1.0) {
        return Err(seal_err("row-derivation clip_eps must lie in (0,1)".to_string()));
    }
    if !(eps_std > 0.0) || !eps_std.is_finite() {
        return Err(seal_err("row-derivation eps_std must be positive finite".to_string()));
    }
    for (name, weight) in [("w_value", w_value), ("w_bc", w_bc), ("alpha", alpha)] {
        if !(weight >= 0.0) || !weight.is_finite() {
            return Err(seal_err(format!("row-derivation {name} must be finite nonnegative")));
        }
    }
    let numeric = map
        .get("numeric")
        .and_then(Value::as_object)
        .ok_or_else(|| seal_err("row-derivation lacks numeric".to_string()))?;
    for (name, want) in [
        ("logp", "fp32"),
        ("advantage", "fp32-frozen"),
        ("return_wire", "f64"),
        ("entropy", "legal-only"),
    ] {
        let got = numeric
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| seal_err(format!("row-derivation numeric lacks {name}")))?;
        if got != want {
            return Err(seal_err(format!(
                "row-derivation numeric.{name} must name {want} (implemented lane)"
            )));
        }
    }
    let code = map
        .get("code")
        .and_then(Value::as_object)
        .ok_or_else(|| seal_err("row-derivation lacks code".to_string()))?;
    for name in ["core_revision", "adapter_version"] {
        let text = code
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| seal_err(format!("row-derivation code lacks {name}")))?;
        if text.is_empty() || text.len() > 1024 {
            return Err(seal_err(format!("row-derivation code.{name} has bad length")));
        }
    }
    let model = map
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| seal_err("row-derivation lacks model".to_string()))?;
    for name in ["model_spec", "action_table"] {
        let text = model
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| seal_err(format!("row-derivation model lacks {name}")))?;
        parse_digest_text(text)?;
    }
    let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:row-derivation")
        .map_err(|e| seal_err(format!("row-derivation JCS emit failed: {e:?}")))?;
    let recomputed = format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(&bytes)));
    if recomputed != *header_digest {
        return Err(seal_err(
            "header row_derivation_hash diverges from the derivation document".to_string(),
        ));
    }
    Ok(())
}
/// Validate ONE canonical row document (without `row_id`) against the closed
/// context (full gate list lives on the seal contract; every row is checked
/// before admission, no spot checks).
pub fn validate_row(
    doc: &Value,
    ctx: &SealContext,
    t_bucket: u32,
    v_actor_f32: f32,
) -> Result<AdmittedRow, TapeError> {
    let map = doc
        .as_object()
        .ok_or_else(|| seal_err("row must be a JSON object".to_string()))?;
    if map.contains_key("row_id") {
        return Err(seal_err(
            "row must exclude row_id before sealing".to_string(),
        ));
    }
    if map.len() != ROW_KEYS_NO_ID.len() || !ROW_KEYS_NO_ID.iter().all(|k| map.contains_key(*k)) {
        return Err(seal_err("row has a non-SPEC12.3 key set".to_string()));
    }
    if HISTORY_BUCKETS.iter().all(|b| *b != t_bucket) {
        return Err(seal_err(format!(
            "row T bucket {t_bucket} not in 32/64/128/256"
        )));
    }

    let wall_id = require_bounded_str(doc, "wall_id")?;
    let game_id = require_bounded_str(doc, "game_id")?;
    let round_id = require_bounded_str(doc, "round_id")?;
    let decision_id = require_bounded_str(doc, "decision_id")?;
    if round_id.is_empty() || decision_id.is_empty() {
        return Err(seal_err(
            "round_id/decision_id must be nonempty".to_string(),
        ));
    }
    // Source/evaluation ledger isolation: training rows come from the source
    // ledger and NEVER from final-evaluation walls.
    if !ctx.source_walls.contains(&wall_id) {
        return Err(seal_err(format!(
            "wall {wall_id} not in the training source ledger"
        )));
    }
    if ctx.eval_walls.contains(&wall_id) {
        return Err(seal_err(format!(
            "wall {wall_id} is a final-evaluation wall and cannot enter training"
        )));
    }
    let step = require_u64_safe(doc, "trajectory_step")?;
    let actor_num = require_u64_safe(doc, "actor")?;
    if actor_num > 3 {
        return Err(seal_err("actor must be a seat 0..3".to_string()));
    }
    let actor = actor_num as u8;

    // Header equalities.
    let row_action_table = require_digest_text(doc, "action_table_hash")?;
    let header_action_table = parse_digest_text(ctx.header.get("action_table_hash").ok_or_else(
        || seal_err("header lacks action_table_hash".to_string()),
    )?)?;
    if row_action_table != header_action_table {
        return Err(seal_err(
            "row action_table_hash mismatches header".to_string(),
        ));
    }
    let row_behavior = require_digest_text(doc, "behavior_policy_checkpoint_hash")?;
    let header_behavior = parse_digest_text(ctx.header.get("behavior_policy_checkpoint_hash").ok_or_else(
        || seal_err("header lacks behavior_policy_checkpoint_hash".to_string()),
    )?)?;
    if row_behavior != header_behavior {
        return Err(seal_err(
            "row behavior checkpoint mismatches the artifact header".to_string(),
        ));
    }

    // Embedded observation identity: the row's `observation_hash` MUST equal
    // the embedded doc's `observation_hash`, which MUST equal the recomputed
    // sha256 over the embedded doc minus its `observation_hash` key (the
    // oracle's identity projection, `observation_actor.py:375-384`).
    let obs = doc
        .get("actor_observation")
        .ok_or_else(|| seal_err("row lacks actor_observation".to_string()))?;
    let obs_map = obs
        .as_object()
        .ok_or_else(|| seal_err("actor_observation must be an object".to_string()))?;
    let embedded_hash_text = obs_map
        .get("observation_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err("embedded observation lacks observation_hash".to_string()))?;
    let row_hash_text = doc
        .get("observation_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err("row lacks observation_hash".to_string()))?;
    if embedded_hash_text != row_hash_text {
        return Err(seal_err(
            "row observation_hash mismatches the embedded observation".to_string(),
        ));
    }
    let identity_bytes = hydra_feed::canon::canonical_object_without_key(
        obs_map, "observation_hash", "rl:obs-identity",
    ).map_err(|e| seal_err(format!("observation JCS emit failed: {e:?}")))?;
    let recomputed = format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(&identity_bytes)));
    if recomputed != embedded_hash_text {
        return Err(seal_err(format!(
            "embedded observation_hash fails independent re-derivation for decision {}: captured {embedded_hash_text}, rebuilt {recomputed}",
            obs_map.get("decision_id").and_then(Value::as_str).unwrap_or("<missing>"),
        )));
    }
    let embedded_table = obs_map
        .get("action_table_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err("embedded observation lacks action_table_hash".to_string()))?;
    if embedded_table != doc["action_table_hash"].as_str().unwrap_or_default() {
        return Err(seal_err(
            "embedded action_table_hash mismatches the row".to_string(),
        ));
    }
    let embedded_mask = obs_map
        .get("legal_mask")
        .and_then(Value::as_array)
        .ok_or_else(|| seal_err("embedded observation lacks legal_mask".to_string()))?;
    let row_mask = doc
        .get("legal_mask")
        .and_then(Value::as_array)
        .ok_or_else(|| seal_err("row lacks legal_mask".to_string()))?;
    if row_mask.len() != ACTION_WIDTH || embedded_mask.len() != ACTION_WIDTH {
        return Err(seal_err(format!(
            "legal_mask width must be exactly {ACTION_WIDTH}"
        )));
    }
    if row_mask != embedded_mask {
        return Err(seal_err(
            "row legal_mask mismatches the embedded observation".to_string(),
        ));
    }
    for (field, want) in [
        ("decision_id", decision_id.as_str()),
        ("game_id", game_id.as_str()),
    ] {
        let got = obs_map.get(field).and_then(Value::as_str).unwrap_or_default();
        if got != want {
            return Err(seal_err(format!("embedded {field} mismatches the row")));
        }
    }
    if obs_map.get("actor").and_then(Value::as_u64) != Some(u64::from(actor)) {
        return Err(seal_err("embedded actor mismatches the row".to_string()));
    }
    let legal_count = row_mask.iter().filter(|v| *v == &Value::Bool(true)).count();
    if legal_count == 0 {
        return Err(seal_err(
            "row with no legal action is inadmissible".to_string(),
        ));
    }
    let selected = require_u64_safe(doc, "selected_action_id")?;
    if selected >= ACTION_WIDTH as u64 {
        return Err(seal_err(
            "selected action outside the vocabulary".to_string(),
        ));
    }
    if row_mask[selected as usize] != Value::Bool(true) {
        return Err(seal_err("selected action is illegal".to_string()));
    }

    // Objective gates: stored old log-probability must describe a
    // representable categorical probability; G4 four finite doubles; frozen
    // advantage equals the terminal recipe (verified once at seal; the
    // learner consumes the stored value, never recomputes).
    let old_logp = require_f32_lane(
        doc.get("old_selected_log_probability").unwrap_or(&Value::Null),
        "old_selected_log_probability",
    )?;
    let old_prob = old_logp.exp();
    if !(old_prob > 0.0 && old_prob <= 1.0) {
        return Err(seal_err(
            "exp(old_selected_log_probability) must lie in (0,1]".to_string(),
        ));
    }
    let g4 = require_g4(doc.get("return_vector").unwrap_or(&Value::Null))?;
    let stored_a = require_f32_lane(doc.get("raw_advantage").unwrap_or(&Value::Null), "raw_advantage")?;
    if !v_actor_f32.is_finite() {
        return Err(seal_err("behavior value fact must be finite".to_string()));
    }
    let frozen_a = (g4[actor as usize] as f32) - v_actor_f32;
    if stored_a != frozen_a {
        return Err(seal_err(
            "raw_advantage mismatches the frozen G32-V32 recipe".to_string(),
        ));
    }

    // BC pair: present exactly when the frozen objective carries w_bc > 0.
    let bc_hash = doc.get("bc_reference_checkpoint_hash").unwrap_or(&Value::Null);
    let bc_logits = doc.get("bc_reference_logits").unwrap_or(&Value::Null);
    if ctx.expect_bc {
        if bc_hash == &Value::Null || bc_logits == &Value::Null {
            return Err(seal_err(
                "w_bc>0 requires BC checkpoint and logits".to_string(),
            ));
        }
        require_digest_text(doc, "bc_reference_checkpoint_hash")?;
        let logits = bc_logits
            .as_array()
            .ok_or_else(|| seal_err("bc_reference_logits must be an array".to_string()))?;
        if logits.len() != ACTION_WIDTH {
            return Err(seal_err(
                "BC logits must share the action geometry".to_string(),
            ));
        }
        for item in logits {
            require_f32_lane(item, "bc_reference_logits[]")?;
        }
    } else if bc_hash != &Value::Null || bc_logits != &Value::Null {
        return Err(seal_err(
            "w_bc==0 stores no BC fields (both must be null)".to_string(),
        ));
    }
    require_digest_text(doc, "transition_provenance_hash")?;

    let row_id = row_id_of(doc)?;
    Ok(AdmittedRow {
        row_id,
        sort_key: (wall_id, game_id, step, actor, decision_id),
        t_bucket,
        address: (usize::MAX, usize::MAX),
        old_lp: old_logp,
        raw_a: stored_a,
        g4,
        v_behavior: v_actor_f32,
    })
}


struct CountingWriter<W> {
    inner: W,
    count: u64,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count = self.count.saturating_add(n as u64);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}


/// JCS string encoding for header keys and digest texts (ASCII-safe).
fn json_string(text: &str) -> Result<Vec<u8>, TapeError> {
    serde_json::to_vec(&Value::String(text.to_string()))
        .map_err(|e| seal_err(format!("string JSON encode failed: {e}")))
}

/// Sorted header pairs (digest shapes validated fail-closed before hashing).
fn sorted_header_pairs(ctx: &SealContext) -> Result<Vec<(&str, String)>, TapeError> {
    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(HEADER_KEYS.len() + 1);
    for key in HEADER_KEYS {
        let value = ctx
            .header
            .get(key)
            .ok_or_else(|| seal_err(format!("header lacks {key}")))?;
        parse_digest_text(value)?;
        pairs.push((key, value.clone()));
    }
    pairs.push(("schema_version", ARTIFACT_SCHEMA_VERSION.to_string()));
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    Ok(pairs)
}


/// Attach the identity to the owned row; cloning its dense masks and history
/// would duplicate every staged value before emitting identical canonical bytes.
#[cfg(test)]
fn full_row_bytes(mut doc_no_id: Value, row_id: &[u8; 32]) -> Result<Vec<u8>, TapeError> {
    let full = doc_no_id
        .as_object_mut()
        .ok_or_else(|| seal_err("admitted row must stay an object".to_string()))?;
    full.insert(
        "row_id".to_string(),
        Value::String(format!("{DIGEST_PREFIX}{}", hex_of(row_id))),
    );
    hydra_feed::canon::canonical_bytes_value(&doc_no_id, "rl:row-full")
        .map_err(|e| seal_err(format!("row JCS emit failed: {e:?}")))
}

// Reuse the independently reproduced row bytes; emitting both no-id and full
// documents walks dense masks/history twice. Derive the insertion boundary
// from a canonical root suffix, never by searching nested JSON key text.
// The admitted root keys are closed ASCII and the suffix after row_id contains
// only scalars; UTF-8/UTF-16 ordering agrees here. A schema change must preserve
// that boundary or fail the suffix check rather than corrupt artifact identity.
fn attach_row_id_bytes(
    doc: &Value,
    bytes: &mut Vec<u8>,
    row_id: &[u8; 32],
) -> Result<(), TapeError> {
    let root = doc
        .as_object()
        .ok_or_else(|| seal_err("admitted row must stay an object".to_string()))?;
    let suffix: serde_json::Map<String, Value> = root
        .iter()
        .filter(|(key, _)| key.as_str() > "row_id")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if root.contains_key("row_id") || suffix.is_empty() {
        return Err(seal_err("row identity insertion requires a nonempty root suffix".to_string()));
    }
    let suffix = hydra_feed::canon::canonical_bytes_value(&Value::Object(suffix), "rl:row-suffix")
        .map_err(|e| seal_err(format!("row suffix JCS emit failed: {e:?}")))?;
    let tail = &suffix[1..];
    if !bytes.ends_with(tail) {
        return Err(seal_err("canonical row suffix boundary mismatch".to_string()));
    }
    let offset = bytes.len() - tail.len();
    let fragment = format!("\"row_id\":\"{DIGEST_PREFIX}{}\",", hex_of(row_id));
    let old_len = bytes.len();
    bytes.resize(old_len + fragment.len(), 0);
    bytes.copy_within(offset..old_len, offset + fragment.len());
    bytes[offset..offset + fragment.len()].copy_from_slice(fragment.as_bytes());
    Ok(())
}

/// Pass 1: materialize ONE bounded observation per row, validate every row,
/// and return the small sorted meta table (no row bytes retained).
// Scoped workers own at most eight results each; joining in chunk order keeps
// canonical output deterministic and a panic rejects instead of losing rows.
fn seal_workers() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from).min(20)
}

fn parallel_rows<T: Sync, R: Send>(
    input: &[T],
    work: impl Fn(&T) -> Result<R, TapeError> + Sync,
) -> Result<Vec<R>, TapeError> {
    std::thread::scope(|scope| {
        let jobs: Vec<_> = input.chunks(8).map(|chunk| {
            let work = &work;
            scope.spawn(move || chunk.iter().map(work).collect::<Result<Vec<_>, _>>())
        }).collect();
        let mut output = Vec::with_capacity(input.len());
        let mut failure = None;
        for job in jobs {
            match job.join() {
                Ok(Ok(rows)) => output.extend(rows),
                Ok(Err(error)) => { failure.get_or_insert(error); }
                Err(_) => { failure.get_or_insert_with(|| seal_err("seal row worker panicked".to_string())); }
            }
        }
        failure.map_or(Ok(output), Err)
    })
}

fn pass1_admit(handle: TapeHandle, ctx: &SealContext) -> Result<Vec<AdmittedRow>, TapeError> {
    // Row addresses first under one short lock; materialization and
    // validation then proceed per row so pool registration is never blocked
    // across the whole pass.
    let addresses: Vec<(usize, usize)> = with_tape(handle, |tape| {
        let mut out = Vec::new();
        for (game_idx, game) in tape.games.iter().enumerate() {
            for (row_idx, _) in game.rows.iter().enumerate() {
                out.push((game_idx, row_idx));
            }
        }
        Ok(out)
    })?;
    if addresses.is_empty() {
        return Err(seal_err("cannot seal an empty generation".to_string()));
    }
    let mut admitted = Vec::with_capacity(addresses.len());
    for window in addresses.chunks(seal_workers() * 8) {
        let rows = parallel_rows(window, |&(game_idx, row_idx)| {
            with_tape(handle, |tape| {
            let game = tape.games.get(game_idx).ok_or_else(|| {
                seal_err(format!("tape address game {game_idx} vanished mid-seal"))
            })?;
            let row = game.rows.get(row_idx).ok_or_else(|| {
                seal_err(format!("tape address row {row_idx} vanished mid-seal"))
            })?;
            if row.event_prefix_len as usize > game.arena.len() {
                return Err(seal_err(format!(
                    "row {} prefix {} exceeds arena {}",
                    row.decision_id,
                    row.event_prefix_len,
                    game.arena.len()
                )));
            }
            if row.actor as usize >= 4 {
                return Err(seal_err(format!(
                    "row {} actor {} outside four seats",
                    row.decision_id, row.actor
                )));
            }
            let obs = (tape.materialize)(row, &game.arena)?;
            let doc = canonical_row_doc_no_id(row, obs)?;
            // Model history is actor-visible and hand-scoped; the arena prefix
            // spans the hanchan, so bucketing it rejects valid later-hand rows.
            let history_len = doc
                .get("actor_observation")
                .and_then(|obs| obs.get("visible_history"))
                .and_then(Value::as_array)
                .ok_or_else(|| seal_err("row observation visible_history must be an array".to_string()))?
                .len();
            let t_bucket = bucket_history_len(
                u32::try_from(history_len)
                    .map_err(|_| seal_err("visible history length exceeds u32".to_string()))?,
            )?;
            let v_actor = row.value4[row.actor as usize];
            let mut row_admitted = validate_row(&doc, ctx, t_bucket, v_actor)?;
            row_admitted.address = (game_idx, row_idx);
            Ok(row_admitted)
            })
        })?;
        admitted.extend(rows);
    }
    let mut admitted = admitted;
    admitted.sort_by(|a, b| {
        a.sort_key
            .cmp(&b.sort_key)
            .then_with(|| a.row_id.cmp(&b.row_id))
    });
    let mut seen_ids = HashSet::with_capacity(admitted.len());
    let mut last_step: HashMap<(&str, &str), u64> = HashMap::with_capacity(admitted.len());
    for row in &admitted {
        if !seen_ids.insert(row.row_id) {
            return Err(seal_err(format!(
                "duplicate row id sha256:{}",
                hex_of(&row.row_id)
            )));
        }
        let key = (row.sort_key.0.as_str(), row.sort_key.1.as_str());
        if let Some(prev) = last_step.insert(key, row.sort_key.2) {
            if row.sort_key.2 <= prev {
                return Err(seal_err(format!(
                    "trajectory_step not strictly increasing in {}/{}",
                    key.0, key.1
                )));
            }
        }
    }
    Ok(admitted)
}

/// Production seal: two streaming passes over the native tape into a single
/// zstd stream plus the small row-metas companion file.
///
/// `artifact_path` receives the raw zstd stream (single canonical root
/// document, hash-input form); `metas_path` receives the structural
/// companion JSON (header, per-row facts in canonical order, digest,
/// counts). fsync/atomic publication is the snapshot layer's job; a crash
/// mid-seal leaves an unverifiable partial blob that tip authority ignores.
pub fn seal_from_tape(
    handle: TapeHandle,
    ctx: &SealContext,
    artifact_path: &std::path::Path,
    metas_path: &std::path::Path,
) -> Result<SealReceipt, TapeError> {
    if let Some(parent) = artifact_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| seal_err(format!("artifact dir create failed: {e}")))?;
        }
    }
    if let Some(parent) = metas_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| seal_err(format!("metas dir create failed: {e}")))?;
        }
    }
    let admitted = pass1_admit(handle, ctx)?;
    let header_pairs = sorted_header_pairs(ctx)?;

    let file = File::create(artifact_path)
        .map_err(|e| seal_err(format!("artifact create failed: {e}")))?;
    let counting = CountingWriter { inner: file, count: 0 };
    let encoder = zstd::stream::Encoder::new(counting, ARTIFACT_ZSTD_LEVEL)
        .map_err(|e| seal_err(format!("zstd encoder open failed: {e}")))?;
    // The zstd stream owns the counting file writer; pass 2 feeds identical
    // fragments to the running digest and the stream (checked writes).
    let mut pass2 = Pass2Writer::new(encoder);
    let mut uncompressed: u64 = 0;

    // Header framing first (identical bytes feed the hasher and the file).
    {
        let head = render_header_head(&header_pairs)?;
        pass2.hasher_update(&head);
        pass2.write_all_checked(&head)?;
        uncompressed = uncompressed.saturating_add(head.len() as u64);
    }
    pass2.hasher_update(b",\"rows\":[");
    pass2.write_all_checked(b",\"rows\":[")?;
    uncompressed = uncompressed.saturating_add(b",\"rows\":[".len() as u64);

    let window_size = seal_workers() * 8;
    for (window_index, window) in admitted.chunks(window_size).enumerate() {
        let rows = parallel_rows(window, |meta| {
            // Independent second-pass reproduction rejects changed row facts.
            with_tape(handle, |tape| {
            let (game_idx, row_idx) = meta.address;
            let game = tape.games.get(game_idx).ok_or_else(|| {
                seal_err(format!("tape address game {game_idx} vanished mid-seal"))
            })?;
            let row = game.rows.get(row_idx).ok_or_else(|| {
                seal_err(format!("tape address row {row_idx} vanished mid-seal"))
            })?;
            let obs = (tape.materialize)(row, &game.arena)?;
            let doc = canonical_row_doc_no_id(row, obs)?;
            let mut bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:row")
                .map_err(|e| seal_err(format!("row JCS emit failed: {e:?}")))?;
            let id = sha256_raw(&bytes);
            if id != meta.row_id {
                return Err(seal_err(format!(
                    "materializer nondeterminism at {}/{}",
                    meta.sort_key.0, meta.sort_key.1
                )));
            }
            attach_row_id_bytes(&doc, &mut bytes, &id)?;
            Ok(bytes)
            })
        })?;
        for (offset, full) in rows.into_iter().enumerate() {
            let i = window_index * window_size + offset;
        if i > 0 {
            pass2.hasher_update(b",");
            pass2.write_all_checked(b",")?;
            uncompressed = uncompressed.saturating_add(1);
        }
        pass2.hasher_update(&full);
        pass2.write_all_checked(&full)?;
        uncompressed = uncompressed.saturating_add(full.len() as u64);
        if pass2.compressed_count() > ARTIFACT_COMPRESSED_CAP {
            return Err(seal_err(format!(
                "compressed artifact exceeds 8GiB cap at row {i}"
            )));
        }
        }
    }
    pass2.hasher_update(b"]}");
    pass2.write_all_checked(b"]}")?;
    uncompressed = uncompressed.saturating_add(2);
    let (compressed_bytes, digest) = pass2.finish()?;
    if compressed_bytes > ARTIFACT_COMPRESSED_CAP {
        return Err(seal_err("compressed artifact exceeds 8GiB cap".to_string()));
    }

    let artifact_digest = format!("{DIGEST_PREFIX}{}", hex_of(&digest));
    write_metas_file(
        metas_path,
        ctx,
        &admitted,
        &artifact_digest,
        compressed_bytes,
        uncompressed,
    )?;
    crate::rl::trajectory::stamp_tape_digest(handle, &artifact_digest)?;
    Ok(SealReceipt {
        artifact_digest,
        row_count: admitted.len(),
        compressed_bytes,
        uncompressed_bytes: uncompressed,
    })
}

/// Render `{"k":"v",...}` header head (sorted pairs, no trailing comma).
fn render_header_head(header_pairs: &[(&str, String)]) -> Result<Vec<u8>, TapeError> {
    let mut head = vec![b'{'];
    for (i, (key, value)) in header_pairs.iter().enumerate() {
        if i > 0 {
            head.push(b',');
        }
        head.extend_from_slice(&json_string(key)?);
        head.push(b':');
        head.extend_from_slice(&json_string(value)?);
    }
    Ok(head)
}

/// Pass-2 dual sink: uncompressed fragments feed the running digest and the
/// zstd stream together (no raw temp file, no 14GB workspace).
struct Pass2Writer<W: Write> {
    encoder: zstd::stream::Encoder<'static, CountingWriter<W>>,
    hasher: sha2::Sha256,
}

impl<W: Write> Pass2Writer<W> {
    fn new(encoder: zstd::stream::Encoder<'static, CountingWriter<W>>) -> Self {
        use sha2::Digest as _;
        Self {
            encoder,
            hasher: sha2::Sha256::new(),
        }
    }
    fn hasher_update(&mut self, bytes: &[u8]) {
        use sha2::Digest as _;
        self.hasher.update(bytes);
    }
    fn write_all_checked(&mut self, bytes: &[u8]) -> Result<(), TapeError> {
        self.encoder
            .write_all(bytes)
            .map_err(|e| seal_err(format!("zstd stream write failed: {e}")))?;
        Ok(())
    }
    fn compressed_count(&self) -> u64 {
        self.encoder.get_ref().count
    }
    fn finish(self) -> Result<(u64, [u8; 32]), TapeError> {
        use sha2::Digest as _;
        let Self { encoder, hasher } = self;
        let counting = encoder
            .finish()
            .map_err(|e| seal_err(format!("zstd stream finish failed: {e}")))?;
        let digest: [u8; 32] = hasher.finalize().into();
        Ok((counting.count, digest))
    }
}


/// Structural companion file (small cold JSON): header, per-row frozen facts
/// in canonical order, digest, counts. NOT authority — the artifact bytes
/// plus full revalidation are.
pub fn write_metas_file(
    metas_path: &std::path::Path,
    ctx: &SealContext,
    admitted: &[AdmittedRow],
    artifact_digest: &str,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
) -> Result<(), TapeError> {
    let mut header_obj = Map::with_capacity(ctx.header.len());
    for (key, value) in ctx.header.iter() {
        header_obj.insert(key.clone(), Value::String(value.clone()));
    }
    let rows: Vec<Value> = admitted
        .iter()
        .map(|row| {
            let mut m = Map::with_capacity(12);
            m.insert(
                "row_id".to_string(),
                Value::String(format!("{DIGEST_PREFIX}{}", hex_of(&row.row_id))),
            );
            m.insert("t".to_string(), Value::from(u64::from(row.t_bucket)));
            m.insert("old_lp".to_string(), Value::from(f64::from(row.old_lp)));
            m.insert("raw_a".to_string(), Value::from(f64::from(row.raw_a)));
            m.insert(
                "g4".to_string(),
                Value::Array(row.g4.iter().map(|v| Value::from(*v)).collect()),
            );
            m.insert("v".to_string(), Value::from(f64::from(row.v_behavior)));
            m.insert("wall_id".to_string(), Value::String(row.sort_key.0.clone()));
            m.insert("game_id".to_string(), Value::String(row.sort_key.1.clone()));
            m.insert("step".to_string(), Value::from(row.sort_key.2));
            // Tape address back into the registered handle (export resolves
            // rows without touching the artifact file).
            m.insert("gi".to_string(), Value::from(row.address.0 as u64));
            m.insert("ri".to_string(), Value::from(row.address.1 as u64));
            Value::Object(m)
        })
        .collect();
    let mut doc = Map::with_capacity(8);
    doc.insert(
        "schema".to_string(),
        Value::String("hydra2.rl_metas_v1".to_string()),
    );
    doc.insert("artifact_digest".to_string(), Value::String(artifact_digest.to_string()));
    doc.insert("header".to_string(), Value::Object(header_obj));
    doc.insert("row_count".to_string(), Value::from(admitted.len() as u64));
    doc.insert("rows".to_string(), Value::Array(rows));
    doc.insert("compressed_bytes".to_string(), Value::from(compressed_bytes));
    doc.insert("uncompressed_bytes".to_string(), Value::from(uncompressed_bytes));
    let bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(doc), "rl:metas")
        .map_err(|e| seal_err(format!("metas JCS emit failed: {e:?}")))?;
    std::fs::write(metas_path, &bytes)
        .map_err(|e| seal_err(format!("metas write failed: {e}")))?;
    Ok(())
}

/// Parse a structural metas companion (cold JSON bytes are fine here: the
/// metas are small by construction, never the 14GB data path).
pub fn parse_metas(metas_bytes: &[u8]) -> Result<Value, TapeError> {
    let value: Value = serde_json::from_slice(metas_bytes)
        .map_err(|e| seal_err(format!("metas is not JSON: {e}")))?;
    let map = value
        .as_object()
        .ok_or_else(|| seal_err("metas must be an object".to_string()))?;
    if map.get("schema").and_then(Value::as_str) != Some("hydra2.rl_metas_v1") {
        return Err(seal_err("metas schema mismatch".to_string()));
    }
    Ok(value)
}

/// Recompute the artifact digest of decompressed canonical root bytes
/// (consumer check: sha256 over the exact bytes, no chain).
pub fn digest_of_root_bytes(root_bytes: &[u8]) -> String {
    format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(root_bytes)))
}

/// Streaming revalidation of a production artifact file against its metas:
/// decodes the single zstd stream, verifies the exact header prefix,
/// revalidates EVERY row (full [`validate_row`]), enforces metas canonical
/// order, and compares the recomputed digest. Bounded memory (one chunk +
/// one row). Returns the seal receipt on success.
pub fn validate_artifact_file(
    artifact_path: &std::path::Path,
    metas_bytes: &[u8],
    ctx: &SealContext,
) -> Result<SealReceipt, TapeError> {
    let metas = parse_metas(metas_bytes)?;
    let expect_digest = metas
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| seal_err("metas lacks artifact_digest".to_string()))?;
    parse_digest_text(expect_digest)?;
    let meta_rows = metas
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| seal_err("metas lacks rows".to_string()))?;
    for (key, want) in ctx.header.iter() {
        let got = metas
            .get("header")
            .and_then(|h| h.get(key))
            .and_then(Value::as_str)
            .ok_or_else(|| seal_err(format!("metas header lacks {key}")))?;
        if got != want {
            return Err(seal_err(format!("metas header {key} mismatches context")));
        }
    }

    let header_pairs = sorted_header_pairs(ctx)?;
    let mut expected_prefix = render_header_head(&header_pairs)?;
    expected_prefix.extend_from_slice(b",\"rows\":[");

    let file =
        File::open(artifact_path).map_err(|e| seal_err(format!("artifact open failed: {e}")))?;
    let mut decoder = zstd::stream::Decoder::new(BufReader::new(file))
        .map_err(|e| seal_err(format!("zstd decoder open failed: {e}")))?;

    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    let mut scanner = RootScanner::new(&expected_prefix);
    let mut chunk = vec![0u8; SCAN_CHUNK];
    let mut row_index = 0usize;
    let mut uncompressed: u64 = 0;
    loop {
        let n = decoder
            .read(&mut chunk)
            .map_err(|e| seal_err(format!("zstd stream decode failed: {e}")))?;
        if n == 0 {
            break;
        }
        uncompressed = uncompressed.saturating_add(n as u64);
        let mut consumed = 0usize;
        while consumed < n {
            let start = consumed;
            let (used, completed) = scanner.feed(&chunk[consumed..n])?;
            // Every consumed byte feeds the digest (prefix, rows,
            // separators, close) — exactly the sealed framing.
            hasher.update(&chunk[start..start + used]);
            consumed += used;
            if let Some(ScanRecord::Row(bytes)) = completed {
                        let value: Value = serde_json::from_slice(&bytes)
                            .map_err(|e| seal_err(format!("stored row is not JSON: {e}")))?;
                        let row_map = value.as_object().ok_or_else(|| {
                            seal_err("stored row must be an object".to_string())
                        })?;
                        let id_text = row_map
                            .get("row_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| seal_err("stored row lacks row_id".to_string()))?;
                        let id = parse_digest_text(id_text)?;
                        let meta = meta_rows.get(row_index).ok_or_else(|| {
                            seal_err("stored artifact holds more rows than metas".to_string())
                        })?;
                        let meta_id = meta
                            .get("row_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| seal_err("metas row lacks row_id".to_string()))?;
                        if meta_id != id_text {
                            return Err(seal_err(format!(
                                "stored row order diverges from metas at index {row_index}"
                            )));
                        }
                        let t = meta
                            .get("t")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| seal_err("metas row lacks t".to_string()))?
                            as u32;
                        let v = meta
                            .get("v")
                            .and_then(Value::as_f64)
                            .ok_or_else(|| seal_err("metas row lacks v".to_string()))?
                            as f32;
                        let mut no_id = row_map.clone();
                        no_id.remove("row_id");
                        let admitted = validate_row(&Value::Object(no_id), ctx, t, v)?;
                        if admitted.row_id != id {
                            return Err(seal_err(format!(
                                "stored row_id mismatch at index {row_index}"
                            )));
                        }
                        // Metas frozen facts must equal the revalidated row.
                        for (field, meta_key) in [("old_lp", admitted.old_lp), ("raw_a", admitted.raw_a)] {
                            let mv = meta.get(field).and_then(Value::as_f64).ok_or_else(|| {
                                seal_err(format!("metas row lacks {field}"))
                            })?;
                            if mv != f64::from(meta_key) {
                                return Err(seal_err(format!(
                                    "metas {field} diverges at index {row_index}"
                                )));
                            }
                        }
                        row_index += 1;
            }
        }
    }
    scanner.finish(row_index, meta_rows.len())?;
    let digest = format!("{DIGEST_PREFIX}{}", hex_of(&hasher.finalize()));
    if digest != expect_digest {
        return Err(seal_err("artifact digest mismatch after streaming revalidation".to_string()));
    }
    if row_index != meta_rows.len() {
        return Err(seal_err("stored artifact holds fewer rows than metas".to_string()));
    }
    let compressed = std::fs::metadata(artifact_path)
        .map_err(|e| seal_err(format!("artifact stat failed: {e}")))?
        .len();
    Ok(SealReceipt {
        artifact_digest: digest,
        row_count: row_index,
        compressed_bytes: compressed,
        uncompressed_bytes: uncompressed,
    })
}

enum ScanRecord {
    HeaderDone,
    Row(Vec<u8>),
}

/// Incremental scanner for the single-doc root framing: exact header-prefix
/// match, then balanced-brace row records (escape-aware), then `]` `}` EOF.
/// JCS emits no insignificant whitespace, so structural bytes are exact.
struct RootScanner<'a> {
    prefix: &'a [u8],
    prefix_pos: usize,
    header_done: bool,
    in_row: bool,
    depth: u32,
    in_string: bool,
    escape: bool,
    current: Vec<u8>,
    rows_done: bool,
    closed: bool,
}

impl<'a> RootScanner<'a> {
    fn new(prefix: &'a [u8]) -> Self {
        Self {
            prefix,
            prefix_pos: 0,
            header_done: false,
            in_row: false,
            depth: 0,
            in_string: false,
            escape: false,
            current: Vec::new(),
            rows_done: false,
            closed: false,
        }
    }

    /// Consume input; returns `(bytes_used, completed_record)`. At most one
    /// record completes per call (callers loop).
    fn feed(&mut self, input: &[u8]) -> Result<(usize, Option<ScanRecord>), TapeError> {
        if !self.header_done {
            let want = self.prefix.len() - self.prefix_pos;
            let take = want.min(input.len());
            if input[..take] != self.prefix[self.prefix_pos..self.prefix_pos + take] {
                return Err(seal_err("artifact header prefix mismatch".to_string()));
            }
            self.prefix_pos += take;
            if self.prefix_pos == self.prefix.len() {
                self.header_done = true;
                return Ok((take, Some(ScanRecord::HeaderDone)));
            }
            return Ok((take, None));
        }
        if self.rows_done {
            // Expect `]` already consumed at row end; now `}` then EOF.
            let b = input[0];
            if b != b'}' {
                return Err(seal_err("artifact trailing framing mismatch".to_string()));
            }
            self.closed = true;
            self.rows_done = false;
            return Ok((1, None));
        }
        if self.closed {
            return Err(seal_err("trailing bytes after artifact root".to_string()));
        }
        // Inside the rows array: either a separator/start or a row body.
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            i += 1;
            if !self.in_row {
                if b == b']' {
                    self.rows_done = true;
                    return Ok((i, None));
                }
                if b == b',' {
                    // Inter-row separator: consumed and hashed, no record.
                    continue;
                }
                if b != b'{' {
                    return Err(seal_err("artifact row must open with {".to_string()));
                }
                self.in_row = true;
                self.depth = 1;
                self.in_string = false;
                self.escape = false;
                self.current.clear();
                self.current.push(b);
                continue;
            }
            self.current.push(b);
            if self.escape {
                self.escape = false;
                continue;
            }
            if self.in_string {
                if b == b'\\' {
                    self.escape = true;
                } else if b == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match b {
                b'"' => self.in_string = true,
                b'{' => self.depth += 1,
                b'}' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.in_row = false;
                        let record = std::mem::take(&mut self.current);
                        return Ok((i, Some(ScanRecord::Row(record))));
                    }
                }
                _ => {}
            }
        }
        Ok((i, None))
    }

    fn finish(&self, rows_seen: usize, rows_want: usize) -> Result<(), TapeError> {
        if !self.header_done {
            return Err(seal_err("artifact truncated before rows".to_string()));
        }
        if self.in_row {
            return Err(seal_err("artifact truncated mid-row".to_string()));
        }
        if !self.closed {
            return Err(seal_err("artifact truncated before root close".to_string()));
        }
        if rows_seen != rows_want {
            return Err(seal_err("artifact row count diverges from metas".to_string()));
        }
        Ok(())
    }
}

/// Opaque handle for one admitted asynchronous seal job.
pub type SealJobId = u64;

/// Asynchronous sealer: the production seal runs on ONE native worker thread
/// so the CUDA owner stays useful. The owner admits a seal, alternates
/// eligible GPU work, polls for completion, and only blocks in `wait` when
/// no eligible work remains. Admission while a job is live fails closed
/// (backpressure, never a second thread). Quiesce drains the live job
/// before durable control; the synchronous [`seal_from_tape`] stays the
/// single implementation underneath (no second seal path).
struct SealJob {
    worker: std::thread::JoinHandle<Result<SealReceipt, TapeError>>,
    tape_handle: TapeHandle,
}

struct SealerState {
    jobs: std::collections::HashMap<SealJobId, SealJob>,
    next: SealJobId,
}

static SEALER: std::sync::LazyLock<std::sync::Mutex<SealerState>> =
    std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(SealerState {
            jobs: std::collections::HashMap::new(),
            next: 1,
        })
    });

/// Seal inputs owned across the thread boundary (the worker never touches
/// the interpreter or borrowed tape state).
pub struct SealRequest {
    pub tape_handle: TapeHandle,
    pub context: SealContext,
    pub artifact_path: std::path::PathBuf,
    pub metas_path: std::path::PathBuf,
}

/// Admit one seal job (fails closed while a job is live). The tape and its
/// compact owners MUST stay registered through worker completion — removal
/// mid-seal fails the job loudly at its next row address, never silently.
pub fn begin_seal(request: SealRequest) -> Result<SealJobId, TapeError> {
    let mut state = SEALER
        .lock()
        .map_err(|e| seal_err(format!("sealer state poisoned: {e}")))?;
    if !state.jobs.is_empty() {
        return Err(seal_err(
            "sealer busy: one native seal job admitted at a time".to_string(),
        ));
    }
    let id = state.next;
    state.next = state.next.saturating_add(1);
    if id == 0 || state.next == 0 {
        return Err(seal_err("seal job id space exhausted".to_string()));
    }
    let worker = std::thread::Builder::new()
        .name(format!("rl-seal-{id}"))
        .spawn(move || {
            seal_from_tape(
                request.tape_handle,
                &request.context,
                &request.artifact_path,
                &request.metas_path,
            )
        })
        .map_err(|e| seal_err(format!("sealer thread spawn failed: {e}")))?;
    state.jobs.insert(
        id,
        SealJob {
            worker,
            tape_handle: request.tape_handle,
        },
    );
    Ok(id)
}

/// Nonblocking poll: `Ok(None)` while the worker runs; on completion the
/// job is removed and the receipt (or the worker error, poison-loud) is
/// returned. A panicked worker reports as a seal failure naming the job.
pub fn poll_seal(job: SealJobId) -> Result<Option<SealReceipt>, TapeError> {
    let finished = {
        let state = SEALER
            .lock()
            .map_err(|e| seal_err(format!("sealer state poisoned: {e}")))?;
        match state.jobs.get(&job) {
            Some(slot) => slot.worker.is_finished(),
            None => {
                return Err(seal_err(format!("unknown seal job {job}")));
            }
        }
    };
    if !finished {
        return Ok(None);
    }
    wait_seal(job).map(Some)
}

/// Blocking drain for quiesce and tests: removes the job and joins the
/// worker. Call only when no eligible GPU work remains (production) or to
/// finish a submitted seal before durable control.
pub fn wait_seal(job: SealJobId) -> Result<SealReceipt, TapeError> {
    let slot = {
        let mut state = SEALER
            .lock()
            .map_err(|e| seal_err(format!("sealer state poisoned: {e}")))?;
        state.jobs.remove(&job).ok_or_else(|| {
            seal_err(format!("unknown seal job {job}"))
        })?
    };
    match slot.worker.join() {
        Ok(outcome) => outcome.map_err(|e| {
            seal_err(format!(
                "seal job {job} (tape {}) failed: {e}",
                slot.tape_handle
            ))
        }),
        Err(_) => Err(seal_err(format!(
            "sealer worker panicked on job {job} (tape {})",
            slot.tape_handle
        ))),
    }
}

/// True while a seal job is live (backpressure signal for the owner).
pub fn sealer_busy() -> bool {
    match SEALER.lock() {
        Ok(state) => !state.jobs.is_empty(),
        Err(_) => true,
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;
    use crate::rl::trajectory::{bucket_history_len, digest_text};

    #[test]
    fn parallel_windows_preserve_order_tails_and_reject_errors() {
        let input: Vec<usize> = (0..19).collect();
        assert_eq!(parallel_rows(&input, |value| Ok(value * 3)).unwrap(),
            input.iter().map(|value| value * 3).collect::<Vec<_>>());
        assert!(parallel_rows(&input, |value| if *value == 9 {
            Err(seal_err("invalid row".to_string()))
        } else { Ok(*value) }).is_err());
        assert!(parallel_rows(&input, |value| {
            if *value == 9 { panic!("materializer failed"); }
            Ok(*value)
        }).is_err());
    }

    #[test]
    fn row_identity_insertion_preserves_nested_keys_and_escaped_values() {
        for history in [serde_json::json!([]), serde_json::json!([
            {"selected_action_id": 3, "round_id": "escaped\\\"value"},
            {"row_id": "nested identity must remain unchanged"}
        ])] {
            let doc = serde_json::json!({
                "actor_observation": {"visible_history": history},
                "round_id": "round\\\"with\\\\escapes",
                "selected_action_id": 6791,
                "trajectory_step": 256,
                "wall_id": "wall-1"
            });
            let mut bytes = hydra_feed::canon::canonical_bytes_value(&doc, "test").unwrap();
            let id = sha256_raw(&bytes);
            attach_row_id_bytes(&doc, &mut bytes, &id).unwrap();
            assert_eq!(bytes, full_row_bytes(doc, &id).unwrap());
        }
    }

    fn header(behavior: &str) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for key in HEADER_KEYS {
            map.insert(key.to_string(), format!("{DIGEST_PREFIX}{}", "c".repeat(64)));
        }
        map.insert(
            "behavior_policy_checkpoint_hash".to_string(),
            behavior.to_string(),
        );
        map
    }

    fn ctx() -> SealContext {
        let mut source = HashSet::new();
        source.insert("wall-1".to_string());
        source.insert("wall-2".to_string());
        let mut eval = HashSet::new();
        eval.insert("eval-9".to_string());
        SealContext {
            header: header(&format!("{DIGEST_PREFIX}{}", "b".repeat(64))),
            source_walls: source,
            eval_walls: eval,
            expect_bc: false,
        }
    }

    /// Bounded embedded observation carrying the identity fields the
    /// validator binds; the row hash is recomputed exactly like production.
    fn obs_doc(game: &str, dec: &str, actor: u8, table: &str) -> Value {
        let mut mask = vec![Value::Bool(false); ACTION_WIDTH];
        mask[3] = Value::Bool(true);
        mask[9] = Value::Bool(true);
        let mut obs = Map::new();
        obs.insert("game_id".to_string(), Value::String(game.to_string()));
        obs.insert("decision_id".to_string(), Value::String(dec.to_string()));
        obs.insert("actor".to_string(), Value::from(u64::from(actor)));
        obs.insert("action_table_hash".to_string(), Value::String(table.to_string()));
        obs.insert("legal_mask".to_string(), Value::Array(mask));
        obs.insert("round_index".to_string(), Value::from(0u64));
        let identity = {
            let mut id = obs.clone();
            id.remove("observation_hash");
            Value::Object(id)
        };
        let bytes = hydra_feed::canon::canonical_bytes_value(&identity, "test:obs")
            .expect("test obs JCS");
        obs.insert(
            "observation_hash".to_string(),
            Value::String(digest_of_root_bytes(&bytes).replace("sha256:", DIGEST_PREFIX)),
        );
        Value::Object(obs)
    }

    fn row_doc(wall: &str, game: &str, step: u64, actor: u8, dec: &str, advantage: f64) -> Value {
        let table = format!("{DIGEST_PREFIX}{}", "c".repeat(64));
        let obs = obs_doc(game, dec, actor, &table);
        let obs_hash = obs
            .get("observation_hash")
            .and_then(Value::as_str)
            .expect("test obs hash")
            .to_string();
        let mask = obs
            .get("legal_mask")
            .and_then(Value::as_array)
            .expect("test mask")
            .clone();
        let mut doc = Map::new();
        doc.insert("wall_id".to_string(), Value::String(wall.to_string()));
        doc.insert("trajectory_step".to_string(), Value::from(step));
        doc.insert("game_id".to_string(), Value::String(game.to_string()));
        doc.insert("round_id".to_string(), Value::String("round-0".to_string()));
        doc.insert("decision_id".to_string(), Value::String(dec.to_string()));
        doc.insert("actor".to_string(), Value::from(u64::from(actor)));
        doc.insert("observation_hash".to_string(), Value::String(obs_hash));
        doc.insert("actor_observation".to_string(), obs);
        doc.insert("action_table_hash".to_string(), Value::String(table));
        doc.insert("legal_mask".to_string(), Value::Array(mask));
        doc.insert("selected_action_id".to_string(), Value::from(3u64));
        doc.insert(
            "behavior_policy_checkpoint_hash".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "b".repeat(64))),
        );
        doc.insert(
            "old_selected_log_probability".to_string(),
            Value::from(f64::from(-0.5f32)),
        );
        doc.insert("raw_advantage".to_string(), Value::from(advantage));
        doc.insert(
            "return_vector".to_string(),
            Value::Array(vec![
                Value::from(10.0),
                Value::from(20.0),
                Value::from(-15.0),
                Value::from(-15.0),
            ]),
        );
        doc.insert("bc_reference_checkpoint_hash".to_string(), Value::Null);
        doc.insert("bc_reference_logits".to_string(), Value::Null);
        doc.insert(
            "transition_provenance_hash".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "d".repeat(64))),
        );
        Value::Object(doc)
    }

    #[test]
    fn canonical_order_is_completion_independent() {
        let context = ctx();
        // Actor 1: G[1]=20, V=8 -> A=12. Actor 0: G[0]=10, V=8 -> A=2.
        let docs = vec![
            (row_doc("wall-1", "game-a", 1, 1, "dec-1", 12.0), 32, 8.0f32),
            (row_doc("wall-1", "game-a", 0, 0, "dec-0", 2.0), 32, 8.0f32),
        ];
        let order_a = seal_rows_in_memory(&context, &docs).expect("seal a");
        let mut rev = docs.clone();
        rev.reverse();
        let order_b = seal_rows_in_memory(&context, &rev).expect("seal b");
        assert_eq!(order_a.0, order_b.0);
        assert_eq!(order_a.1, order_b.1);
        assert_eq!(order_a.2[0].sort_key.2, 0);
    }

    #[test]
    fn validator_rejects_source_and_advantage_forks() {
        let context = ctx();
        // Eval wall.
        let bad = row_doc("eval-9", "game-e", 0, 1, "dec-e", 12.0);
        assert!(validate_row(&bad, &context, 32, 8.0).is_err());
        // Wrong frozen advantage.
        let bad_a = row_doc("wall-1", "game-a", 0, 1, "dec-x", 13.0);
        assert!(validate_row(&bad_a, &context, 32, 8.0).is_err());
        // Duplicate rows.
        let dup = row_doc("wall-1", "game-a", 0, 1, "dec-same", 12.0);
        assert!(seal_rows_in_memory(&context, &[(dup.clone(), 32, 8.0f32), (dup, 32, 8.0f32)]).is_err());
        // Non-monotonic steps within a game.
        let s0 = row_doc("wall-1", "game-a", 1, 1, "dec-p", 12.0);
        let s1 = row_doc("wall-1", "game-a", 1, 0, "dec-q", 2.0);
        assert!(seal_rows_in_memory(&context, &[(s0, 32, 8.0f32), (s1, 32, 8.0f32)]).is_err());
        let _ = bucket_history_len(32).expect("bucket");
    }

    #[test]
    fn scanner_round_trips_framing() {
        let context = ctx();
        let docs = vec![(row_doc("wall-1", "game-a", 0, 1, "dec-0", 12.0), 32, 8.0f32)];
        let (digest, root, _) = seal_rows_in_memory(&context, &docs).expect("seal");
        assert_eq!(digest, digest_of_root_bytes(&root));
        // Feed the scanner byte-by-byte odd splits.
        let pairs = sorted_header_pairs(&context).expect("pairs");
        let mut prefix = render_header_head(&pairs).expect("head");
        prefix.extend_from_slice(b",\"rows\":[");
        let mut scanner = RootScanner::new(&prefix);
        let mut rows = 0;
        let mut pos = 0;
        while pos < root.len() {
            let end = (pos + 7).min(root.len());
            let (used, record) = scanner.feed(&root[pos..end]).expect("feed");
            pos += used;
            if let Some(ScanRecord::Row(_)) = record {
                rows += 1;
            }
        }
        // Feed the closing (already consumed inside root) — finish checks.
        assert_eq!(rows, 1);
        scanner.finish(rows, 1).expect("finish");
    }

    /// Test-only in-memory seal (bounded fixtures): validates, sorts, and
    /// renders the single canonical root bytes plus digest. Production MUST
    /// use [`seal_from_tape`] (streaming, bounded memory).
    fn seal_rows_in_memory(
        ctx: &SealContext,
        rows: &[(Value, u32, f32)],
    ) -> Result<(String, Vec<u8>, Vec<AdmittedRow>), TapeError> {
        let mut admitted = Vec::with_capacity(rows.len());
        for (doc, t, v) in rows {
            admitted.push(validate_row(doc, ctx, *t, *v)?);
        }
        admitted.sort_by(|a, b| {
            a.sort_key
                .cmp(&b.sort_key)
                .then_with(|| a.row_id.cmp(&b.row_id))
        });
        let mut seen = HashSet::new();
        let mut last: HashMap<(&str, &str), u64> = HashMap::new();
        for row in &admitted {
            if !seen.insert(row.row_id) {
                return Err(seal_err("duplicate row id".to_string()));
            }
            let key = (row.sort_key.0.as_str(), row.sort_key.1.as_str());
            if let Some(prev) = last.insert(key, row.sort_key.2) {
                if row.sort_key.2 <= prev {
                    return Err(seal_err("steps not strictly increasing".to_string()));
                }
            }
        }
        let pairs = sorted_header_pairs(ctx)?;
        let mut by_id: HashMap<[u8; 32], &Value> = HashMap::new();
        for (doc, _, _) in rows {
            let id = row_id_of(doc)?;
            by_id.insert(id, doc);
        }
        let ordered: Vec<Vec<u8>> = admitted
            .iter()
            .map(|row| {
                by_id
                    .get(&row.row_id)
                    .ok_or_else(|| seal_err("row vanished".to_string()))
                    .and_then(|doc| full_row_bytes((*doc).clone(), &row.row_id))
            })
            .collect::<Result<_, _>>()?;
        let mut out = render_header_head(&pairs)?;
        out.extend_from_slice(b",\"rows\":[");
        for (i, bytes) in ordered.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(bytes);
        }
        out.extend_from_slice(b"]}");
        let digest = digest_of_root_bytes(&out);
        Ok((digest, out, admitted))
    }

    /// Canonical test observation: mask expanded from `legal_ids`, identity
    /// hash recomputed exactly like production (doc minus `observation_hash`).
    fn test_obs(game: &str, dec: &str, actor: u8, table: &str, legal: &[u32]) -> Value {
        let mut mask = vec![Value::Bool(false); ACTION_WIDTH];
        for id in legal {
            mask[*id as usize] = Value::Bool(true);
        }
        let mut obs = Map::new();
        obs.insert("game_id".to_string(), Value::String(game.to_string()));
        obs.insert("decision_id".to_string(), Value::String(dec.to_string()));
        obs.insert("actor".to_string(), Value::from(u64::from(actor)));
        obs.insert(
            "action_table_hash".to_string(),
            Value::String(table.to_string()),
        );
        obs.insert("legal_mask".to_string(), Value::Array(mask));
        obs.insert("round_index".to_string(), Value::from(0u64));
        obs.insert("visible_history".to_string(), Value::Array(Vec::new()));
        let identity = {
            let mut id = obs.clone();
            id.remove("observation_hash");
            Value::Object(id)
        };
        let bytes = hydra_feed::canon::canonical_bytes_value(&identity, "test:obs")
            .expect("test obs JCS");
        obs.insert(
            "observation_hash".to_string(),
            Value::String(digest_of_root_bytes(&bytes)),
        );
        Value::Object(obs)
    }

    fn core_row(
        wall: &str,
        game: &str,
        step: u64,
        actor: u8,
        dec: &str,
        table: &str,
        legal: Vec<u32>,
    ) -> crate::rl::core::GameRow {
        use crate::rl::core::GameRow;
        let obs = test_obs(game, dec, actor, table, &legal);
        let obs_hash = parse_digest_text(
            obs.get("observation_hash")
                .and_then(Value::as_str)
                .expect("test obs hash"),
        )
        .expect("test obs hash shape");
        GameRow {
            wall_id: wall.to_string(),
            game_id: game.to_string(),
            round_id: "round-0".to_string(),
            trajectory_step: step,
            actor,
            decision_id: dec.to_string(),
            sequence: step,
            row_id: [0u8; 32],
            round_index: 0,
            round_wind: 0,
            honba: 0,
            kyotaku: 0,
            dealer: 0,
            scores: [25000, 25000, 25000, 25000],
            live_remaining: 70,
            hand_number: 0,
            turn_actor: actor,
            phase_id: 1,
            concealed: vec![],
            drawn: None,
            dora5: [0, 1, 2, 3, 4],
            ippatsu_bits: 0,
            furiten_id: 0,
            can_tsumo: true,
            can_riichi: false,
            legal_ids: legal,
            legal_mask_bits: vec![],
            selected: 3,
            selected_logp: -0.5,
            value4: [0.0; 4],
            cursor_before: step * 8,
            cursor_after: step * 8 + 8,
            obs_hash,
            // Empty fixture arena: the seal fail-closes prefix > len, and the
            // custom materializer ignores the arena, so the prefix is 0.
            event_prefix_len: 0,
            derivation: [6u8; 32],
            action_table_hash: parse_digest_text(table).expect("test table shape"),
            action_table_hash_hex: table.to_string(),
            behavior_ckpt: parse_digest_text(&format!("{DIGEST_PREFIX}{}", "b".repeat(64)))
                .expect("test behavior shape"),
            transition_provenance: [0xDDu8; 32],
            rules_id: "tenhou".to_string(),
            rules_hash: format!("{DIGEST_PREFIX}{}", "01".repeat(32)),
            event_schema_hash: format!("{DIGEST_PREFIX}{}", "02".repeat(32)),
            observation_schema_hash: format!("{DIGEST_PREFIX}{}", "03".repeat(32)),
            packet_boundary_hash: format!("{DIGEST_PREFIX}{}", "04".repeat(32)),
            return_g4: None,
            raw_advantage_f32: None,
        }
    }

    #[test]
    fn streaming_seal_matches_in_memory_root() {
        use crate::rl::core::GameRow;
        use crate::rl::events::EventArena;
        use crate::rl::trajectory::{
            TapeGame, bind_terminal, register_tape, remove_tape,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_TMP: AtomicU64 = AtomicU64::new(0);

        let context = ctx();
        let table = format!("{DIGEST_PREFIX}{}", "c".repeat(64));
        // Actor 0 step 0: G[0]=10, V=8 -> A=2. Actor 1 step 1: G[1]=20 -> A=12.
        let mut rows = vec![
            core_row("wall-1", "game-a", 0, 0, "dec-0", &table, vec![3, 9]),
            core_row("wall-1", "game-a", 1, 1, "dec-1", &table, vec![3, 9]),
        ];
        rows[0].value4 = [8.0, 0.0, 0.0, 0.0];
        rows[1].value4 = [0.0, 8.0, 0.0, 0.0];
        bind_terminal(&mut rows, [10.0, 20.0, -15.0, -15.0]).expect("bind");
        assert_eq!(rows[0].raw_advantage_f32, Some(2.0));
        assert_eq!(rows[1].raw_advantage_f32, Some(12.0));

        // Deterministic materializer rebuilt from row fields alone (same
        // builder the fixture used, so identities hold by construction).
        let materialize = |row: &GameRow, _arena: &EventArena| {
            Ok(test_obs(
                &row.game_id,
                &row.decision_id,
                row.actor,
                &digest_text(&row.action_table_hash),
                &row.legal_ids,
            ))
        };
        let encode =
            |_row: &GameRow, _arena: &EventArena, _t: u32| Ok(vec![0u8; 7542]);
        let game = TapeGame {
            wall_id: "wall-1".to_string(),
            game_id: "game-a".to_string(),
            arena: EventArena::new(),
            rows,
            utilities: Some([10.0, 20.0, -15.0, -15.0]),
        };
        let handle =
            register_tape(vec![game], Box::new(materialize), Box::new(encode))
                .expect("register");

        let tag = TEST_TMP.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "rl-artifact-streaming-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("test tmp");
        let art = dir.join("a.zst");
        let metas_path = dir.join("m.json");

        // Synchronous production seal, then streaming revalidation.
        let receipt = seal_from_tape(handle, &context, &art, &metas_path).expect("seal");
        assert_eq!(receipt.row_count, 2);
        let metas_bytes = std::fs::read(&metas_path).expect("metas read");
        let again = validate_artifact_file(&art, &metas_bytes, &context).expect("revalidate");
        assert_eq!(again.artifact_digest, receipt.artifact_digest);
        assert_eq!(again.row_count, 2);

        // Gate: decompressed production bytes equal the bounded established
        // canonical root, and the digest matches over those bytes.
        let compressed = std::fs::read(&art).expect("artifact read");
        let mut decoder = zstd::stream::Decoder::new(&compressed[..]).expect("zstd open");
        let mut decompressed = Vec::new();
        use std::io::Read as _;
        decoder.read_to_end(&mut decompressed).expect("zstd decode");
        let docs: Vec<(Value, u32, f32)> = with_tape(handle, |tape| {
            let mut out = Vec::new();
            for game in &tape.games {
                for row in &game.rows {
                    let obs = (tape.materialize)(row, &game.arena).expect("test mat");
                    let doc = canonical_row_doc_no_id(row, obs).expect("test doc");
                    out.push((doc, 32u32, row.value4[row.actor as usize]));
                }
            }
            Ok(out)
        })
        .expect("docs");
        let (expect_digest, expect_root, _) =
            seal_rows_in_memory(&context, &docs).expect("in-memory seal");
        assert_eq!(decompressed, expect_root);
        assert_eq!(receipt.artifact_digest, expect_digest);

        // Asynchronous admission serves the identical receipt.
        let art2 = dir.join("b.zst");
        let metas2 = dir.join("m2.json");
        let job = begin_seal(SealRequest {
            tape_handle: handle,
            context: ctx(),
            artifact_path: art2.clone(),
            metas_path: metas2.clone(),
        })
        .expect("begin");
        let async_receipt = wait_seal(job).expect("wait");
        assert_eq!(async_receipt.artifact_digest, receipt.artifact_digest);
        assert!(!sealer_busy());
        assert!(remove_tape(handle));
        std::fs::remove_dir_all(&dir).ok();
    }

    fn derivation_fixture() -> Value {
        let mut objective = Map::new();
        objective.insert("clip_eps".to_string(), Value::from(0.2));
        objective.insert("eps_std".to_string(), Value::from(1e-8));
        objective.insert("w_value".to_string(), Value::from(0.005));
        objective.insert("w_bc".to_string(), Value::from(0.0));
        objective.insert("alpha".to_string(), Value::from(0.01));
        let mut numeric = Map::new();
        numeric.insert("logp".to_string(), Value::String("fp32".to_string()));
        numeric.insert("advantage".to_string(), Value::String("fp32-frozen".to_string()));
        numeric.insert("return_wire".to_string(), Value::String("f64".to_string()));
        numeric.insert("entropy".to_string(), Value::String("legal-only".to_string()));
        let mut code = Map::new();
        code.insert(
            "core_revision".to_string(),
            Value::String("479c1faeb33d082965eef8198f63261a79c0fce3".to_string()),
        );
        code.insert("adapter_version".to_string(), Value::String("2.0.0".to_string()));
        let mut model = Map::new();
        model.insert(
            "model_spec".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "e".repeat(64))),
        );
        model.insert(
            "action_table".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "c".repeat(64))),
        );
        let mut doc = Map::with_capacity(8);
        for (key, byte) in [
            ("wall_recipe", "aa"),
            ("sampling_recipe", "bb"),
            ("return_recipe", "cc"),
            ("shuffle_recipe", "dd"),
        ] {
            doc.insert(
                key.to_string(),
                Value::String(format!("{DIGEST_PREFIX}{}", byte.repeat(32))),
            );
        }
        doc.insert("objective".to_string(), Value::Object(objective));
        doc.insert("numeric".to_string(), Value::Object(numeric));
        doc.insert("code".to_string(), Value::Object(code));
        doc.insert("model".to_string(), Value::Object(model));
        Value::Object(doc)
    }

    #[test]
    fn derivation_binding_verifies_and_rejects() {
        let doc = derivation_fixture();
        let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "test:deriv").expect("JCS");
        let digest = digest_of_root_bytes(&bytes);
        assert!(verify_row_derivation(&digest, &bytes).is_ok());
        // Arbitrary digest never binds.
        let other = format!("{DIGEST_PREFIX}{}", "f".repeat(64));
        assert!(verify_row_derivation(&other, &bytes).is_err());
        // Each structural gate fires against the mutant's OWN digest, so
        // the rejection proves the gate (not the digest mismatch).
        let check_own = |mutant: &Value| {
            let mutant_bytes =
                hydra_feed::canon::canonical_bytes_value(mutant, "test:deriv").expect("JCS");
            let mutant_digest = digest_of_root_bytes(&mutant_bytes);
            (mutant_bytes, mutant_digest)
        };
        // Wrong behavior literal fails closed.
        let mut fork = doc.clone();
        fork["numeric"]["entropy"] = Value::String("all".to_string());
        let (fork_bytes, fork_digest) = check_own(&fork);
        assert!(verify_row_derivation(&fork_digest, &fork_bytes).is_err());
        // Non-closed key set fails closed.
        let mut slim = doc.clone();
        slim.as_object_mut().expect("obj").remove("code");
        let (slim_bytes, slim_digest) = check_own(&slim);
        assert!(verify_row_derivation(&slim_digest, &slim_bytes).is_err());
        // Out-of-range objective fails closed.
        let mut wild = doc.clone();
        wild["objective"]["clip_eps"] = Value::from(1.5);
        let (wild_bytes, wild_digest) = check_own(&wild);
        assert!(verify_row_derivation(&wild_digest, &wild_bytes).is_err());
    }
}
