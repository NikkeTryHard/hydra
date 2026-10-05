//! rl/config.rs: closed RL schema registry + schema-bound identities.
//!
//! DAG: pyo3 + serde_json + the feed crate (`canon`, `digest`) ONLY — same
//! edge shape as `rc_digest.rs:3-9`. FORBIDS: YAML/file IO, `${VAR}`
//! interpolation, run-config section parsing (all stay in `rc_parse.rs` /
//! Python `_rc_*.py`), wall-clock/RNG, `ContractError` shaping (the FFI
//! layer raises `ValueError`; thin Python translators map to
//! `ContractError` with byte-identical messages), and any new JCS printer
//! (all bytes/hash math lives in `hydra_feed::{canon, digest}`).
//!
//! Owns the four closed RL schema envelopes: `rl_pipeline` pinned at
//! schema `3.0.0` plus `rl_control`, `rl_native_snapshot`,
//! `rl_replay_index` pinned at `1.0.0` — each with `exact` compatibility,
//! RFC 8785 bytes, and SHA-256 payload identities excluding only their own
//! `digest` (the payload `digest` covers every payload field except
//! itself). File bytes live in `configs/contracts/rl_pipeline_v3.json` plus
//! `configs/contracts/rl_{control,native_snapshot,replay_index}_v1.json`;
//! this module registers the frozen artifact-type strings, the exact schema
//! version, and the payload field vocabulary (nested keys included) so
//! unknown-field rejection and own-digest-exclusion stay byte-identical
//! between the native registry and the JSON files. Struct shapes for the
//! runtime consumers (`PipelineSpec`, `ControlSnapshot`, `NativeSnapshot`,
//! `ReplayIndex`) live with their owners (ArtifactStorage owns snapshot IO);
//! this module is the identity authority they validate against, never a
//! parallel struct convention.
//!
//! Single-cdylib tree: pure logic here, PyO3 registration in `config_ffi.rs`
//! (mirrors the `rc_digest` pure/pyfn split). NativeWall wires
//! `pub mod config; pub mod config_ffi;` in `rl/mod.rs` plus
//! `config_ffi::register(&rl_sub)` from the `rl::register` registrar.

use std::collections::BTreeSet;

/// Frozen schema version of the three untouched RL envelopes
/// (`rl_control`, `rl_native_snapshot`, `rl_replay_index`: `1.0.0`, exact).
pub const RL_SCHEMA_VERSION: &str = "1.0.0";

/// Frozen schema version of the pipeline envelope (`3.0.0`, exact): the
/// resource profile binds the four-field runtime identity, so any execution
/// mode change moves the pipeline digest (and every shuffle/binding derived
/// from it). Old `2.0.0`/`1.0.0` pipeline envelopes reject, never restore.
/// Live `training_source='native_live_v1'` and `recovery='generation_restart_v1'`
/// are fixed const fields included before the digest.
pub const RL_PIPELINE_SCHEMA_VERSION: &str = "3.0.0";

/// Frozen artifact-type strings (envelope `artifact_type` per schema file).
pub const RL_PIPELINE_ARTIFACT: &str = "hydra2.rl_pipeline_schema";
pub const RL_CONTROL_ARTIFACT: &str = "hydra2.rl_control_schema";
pub const RL_NATIVE_SNAPSHOT_ARTIFACT: &str = "hydra2.rl_native_snapshot_schema";
pub const RL_REPLAY_INDEX_ARTIFACT: &str = "hydra2.rl_replay_index_schema";

/// Relative JSON paths of the four registered schema files.
pub const RL_PIPELINE_PATH: &str = "configs/contracts/rl_pipeline_v3.json";
pub const RL_CONTROL_PATH: &str = "configs/contracts/rl_control_v1.json";
pub const RL_NATIVE_SNAPSHOT_PATH: &str = "configs/contracts/rl_native_snapshot_v1.json";
pub const RL_REPLAY_INDEX_PATH: &str = "configs/contracts/rl_replay_index_v1.json";
/// Canonical `field_order` of each envelope payload, in document order (the
/// JCS `document_order` each schema file pins under `serialization`).
/// `closed` is schema metadata carried beside the payload, never an
/// artifact-data key: it is allowed by the gate but excluded here so the
/// lists stay byte-identical to the JSON `field_order` arrays.
pub const RL_PIPELINE_FIELDS: [&str; 33] = [
    "schema_version",
    "initial_checkpoint_id",
    "initial_model_spec_hash",
    "rules_hash",
    "utility_hash",
    "action_table_hash",
    "event_schema_hash",
    "observation_schema_hash",
    "packet_boundary_hash",
    "input_schema_hash",
    "master_seed_hex",
    "experiment_id",
    "split_scope",
    "replicate_id",
    "attempt_id",
    "core_revision",
    "wall_recipe_hash",
    "sampling_recipe_hash",
    "return_recipe_hash",
    "shuffle_recipe_hash",
    "wall_ledger_hash",
    "opponent_catalog_hash",
    "generations",
    "games_per_generation",
    "resource_profile",
    "learner_recipe",
    "objective_recipe",
    "return_recipe",
    "optimizer_recipe",
    "guards",
    "training_source",
    "recovery",
    "digest",
];

/// Nested object vocabularies of the pipeline envelope: `parent.child`
/// paths of every nested field (resource/learner/objective/return/
/// optimizer recipe groups plus guards), so validators reject unknown nested
/// keys without a second registry.
pub const RL_PIPELINE_NESTED_FIELDS: [&str; 27] = [
    "resource_profile.native_workers",
    "resource_profile.env_slots",
    "resource_profile.max_lookahead",
    "resource_profile.inference_cap",
    "resource_profile.execution_profile_hash",
    "learner_recipe.microbatch_cap",
    "learner_recipe.accumulation_cap",
    "learner_recipe.optimizer_minibatch_cap",
    "learner_recipe.epochs",
    "learner_recipe.max_updates_per_generation",
    "objective_recipe.clip_eps",
    "objective_recipe.eps_std",
    "objective_recipe.w_value",
    "objective_recipe.w_bc",
    "objective_recipe.ppo_alpha",
    "return_recipe.gamma",
    "return_recipe.lam",
    "return_recipe.bootstrap",
    "return_recipe.shaping",
    "optimizer_recipe.id",
    "optimizer_recipe.lr",
    "optimizer_recipe.beta1",
    "optimizer_recipe.beta2",
    "optimizer_recipe.weight_decay",
    "optimizer_recipe.head_lr_mult",
    "optimizer_recipe.scheduler",
    "optimizer_recipe.warmup_updates",
];

/// Guards nested vocabulary of the pipeline envelope (`guards.*`).
pub const RL_PIPELINE_GUARD_FIELDS: [&str; 6] = [
    "guards.row_cap",
    "guards.decision_cap_per_game",
    "guards.compressed_byte_cap",
    "guards.min_free_disk_bytes",
    "guards.host_byte_budget",
    "guards.gpu_cache_byte_budget",
];

/// Guard nested vocabulary: gradient clip norm lives in the optimizer
/// recipe, not guards.
pub const RL_PIPELINE_GRADIENT_CLIP_FIELD: &str = "optimizer_recipe.gradient_clip_norm";

/// Canonical `field_order` of the control envelope payload.
pub const RL_CONTROL_FIELDS: [&str; 26] = [
    "schema_version",
    "pipeline_spec_id",
    "learner_manifest_id",
    "learner_transport_sha",
    "learner_locator",
    "actor_manifest_id",
    "actor_transport_sha",
    "actor_locator",
    "generation",
    "generation_statuses",
    "source_refs",
    "order_ref",
    "ledger_refs",
    "native_snapshot_digest",
    "tape_blob_digest",
    "event_blob_digest",
    "cohort_membership",
    "frontier_current",
    "frontier_next",
    "terminal_set",
    "tickets",
    "committed_row_offset",
    "committed_update_offset",
    "publication_phase",
    "parent_checkpoint_hash",
    "digest",
];

/// Ticket-entry vocabulary of the control envelope (`tickets[].*`).
pub const RL_CONTROL_TICKET_FIELDS: [&str; 4] = [
    "ticket_descriptor",
    "phase",
    "slot_epoch",
    "stream_cursor",
];

/// Cohort/frontier nested vocabulary of the control envelope. Free maps
/// (`generation_statuses`, `source_refs`, `ledger_refs`: generation/source
/// names to digests/statuses) carry dynamic keys and are listed separately
/// in [`RL_CONTROL_FREE_MAPS`]; they are never enumerated here.
pub const RL_CONTROL_NESTED_FIELDS: [&str; 8] = [
    "cohort_membership.generations",
    "cohort_membership.games_per_generation",
    "frontier_current.generation",
    "frontier_current.round",
    "frontier_current.epoch",
    "frontier_next.generation",
    "frontier_next.round",
    "frontier_next.epoch",
];

/// Control-envelope parents whose keys are dynamic content references
/// (generation/source names), not closed vocabulary: nested checks skip
/// them (value shapes stay consumer-owned in `snapshot.rs` IO).
pub const RL_CONTROL_FREE_MAPS: [&str; 3] = [
    "generation_statuses",
    "source_refs",
    "ledger_refs",
];

/// Frozen publication phases (plan §14 owner transition 3).
pub const RL_PUBLICATION_PHASES: [&str; 4] = [
    "not_captured",
    "captured_at_update1",
    "checkpoint_durable",
    "actor_activated",
];

/// Frozen lease/ticket phases (plan §14 owner transition 1-2).
pub const RL_TICKET_PHASES: [&str; 8] = [
    "free",
    "filling",
    "ready",
    "h2d_issued",
    "infer_queued",
    "d2h_issued",
    "host_ready",
    "owned_capture",
];

/// Canonical `field_order` of the native-snapshot envelope payload.
pub const RL_NATIVE_SNAPSHOT_FIELDS: [&str; 12] = [
    "schema_version",
    "wall_id",
    "game_id",
    "seat_policy_ids",
    "ordinal",
    "core_carry_hash",
    "action_prefix_digest",
    "state_digest",
    "decision_count",
    "pending_responses",
    "stream_cursor",
    "digest",
];

/// Canonical `field_order` of the replay-index envelope payload.
pub const RL_REPLAY_INDEX_FIELDS: [&str; 9] = [
    "schema_version",
    "rlo_digest",
    "feature_layout",
    "history_bucket_T",
    "offset",
    "count",
    "checksum",
    "facts_digest",
    "digest",
];

/// Frozen history buckets the replay index admits (plan §2: 32/64/128/256).
pub const RL_HISTORY_BUCKETS: [u32; 4] = [32, 64, 128, 256];

/// The pinned frozen recipe the native `rl` profile parser pins, restated
/// here as typed literals so schema consumers share one source (plan §3,
/// SPEC §20.1 with `eps_std` INSIDE the sqrt — `normalize =
/// (A - mean) / sqrt(pop_var + eps)`, never `sqrt(var) + eps`).
pub struct RlFrozenRecipe;

impl RlFrozenRecipe {
    /// Frozen PPO epsilon: 0.2.
    pub const CLIP_EPS: f64 = 0.2;
    /// Frozen normalization epsilon, INSIDE the sqrt: 1e-8.
    pub const EPS_STD: f64 = 1e-8;
    /// Frozen value weight: 0.005.
    pub const W_VALUE: f64 = 0.005;
    /// Frozen BC weight: 0.0 (no BC fields stored at w_bc 0).
    pub const W_BC: f64 = 0.0;
    /// Frozen entropy weight: 0.01.
    pub const PPO_ALPHA: f64 = 0.01;
    /// Frozen terminal-MC return law: gamma 1, lambda 1, no bootstrap.
    pub const GAMMA: f64 = 1.0;
    /// Frozen terminal-MC return law: lambda 1.
    pub const LAM: f64 = 1.0;
    /// Frozen RL optimizer: AdamW lr 1e-5.
    pub const OPTIMIZER_LR: f64 = 1e-5;
    /// Frozen RL optimizer betas.
    pub const OPTIMIZER_BETAS: (f64, f64) = (0.9, 0.999);
    /// Frozen RL optimizer weight decay.
    pub const OPTIMIZER_WEIGHT_DECAY: f64 = 0.01;
    /// Frozen RL policy-head LR multiplier.
    pub const HEAD_LR_MULT: f64 = 1.0;
    /// Frozen RL scheduler: constant, zero warmup.
    pub const SCHEDULER_ID: &'static str = "constant";
    /// Frozen RL warmup updates: 0.
    pub const WARMUP_UPDATES: i64 = 0;
    /// Frozen RL gradient clip norm: 1.0.
    pub const GRADIENT_CLIP_NORM: f64 = 1.0;
    /// Frozen RL optimizer name: adamw.
    pub const OPTIMIZER_ID: &'static str = "adamw";
    /// Frozen cohort: 8 generations of 128 games, one epoch, tails trained.
    pub const GENERATIONS: i64 = 8;
    /// Frozen cohort games per generation: 128.
    pub const GAMES_PER_GENERATION: i64 = 128;
    /// Frozen lookahead: max 1 future generation.
    pub const MAX_LOOKAHEAD: i64 = 1;
    /// Frozen epochs: 1 (every row exactly once, tails included).
    pub const EPOCHS: i64 = 1;
    /// Frozen optimizer batch cap: 4096.
    pub const OPTIMIZER_MINIBATCH_CAP: i64 = 4096;
    /// Frozen guard: row cap 262144 per generation.
    pub const ROW_CAP: i64 = 262144;
    /// Frozen guard: decision cap 9000 per game.
    pub const DECISION_CAP_PER_GAME: i64 = 9000;
    /// Frozen guard: at most 67 updates per generation (`sum_T ceil(rows_T/4096)` under the cap).
    pub const MAX_UPDATES_PER_GENERATION: i64 = 67;
}

/// One registered schema identity: artifact type, schema version, JSON path,
/// and the digest of the payload-without-digest recorded in the file.
pub struct RlSchemaRecord {
    /// Envelope `artifact_type` string.
    pub artifact_type: &'static str,
    /// Envelope `schema_version` string (pipeline `3.0.0`, others `1.0.0`).
    pub schema_version: &'static str,
    /// Repo-relative JSON path of the registered bytes.
    pub path: &'static str,
    /// Recorded payload digest (`sha256:<hex>`, JCS payload without digest).
    pub digest: &'static str,
}

/// The four registered schema identities, in pipeline/control/snapshot/index
/// order. Digests are the payload digests recorded in the JSON files; the
/// FFI registry verifies recomputation against these bytes.
pub const RL_SCHEMA_RECORDS: [RlSchemaRecord; 4] = [
    RlSchemaRecord {
        artifact_type: RL_PIPELINE_ARTIFACT,
        schema_version: RL_PIPELINE_SCHEMA_VERSION,
        path: RL_PIPELINE_PATH,
        digest: "sha256:4d83759c2d8e03672b3d28586ab40adbebe7567d77c3144d7fb20cbda2e1bf66",
    },
    RlSchemaRecord {
        artifact_type: RL_CONTROL_ARTIFACT,
        schema_version: RL_SCHEMA_VERSION,
        path: RL_CONTROL_PATH,
        digest: "sha256:fd684dd04d13d1ac4c8b420082638be6d1ad9b7f67f6035a9325a6ed9f2d849d",
    },
    RlSchemaRecord {
        artifact_type: RL_NATIVE_SNAPSHOT_ARTIFACT,
        schema_version: RL_SCHEMA_VERSION,
        path: RL_NATIVE_SNAPSHOT_PATH,
        digest: "sha256:639e83d5f2fe1f3bab2aea24b9979a8e5f8cfc346db8af9d0b91926608eac3e5",
    },
    RlSchemaRecord {
        artifact_type: RL_REPLAY_INDEX_ARTIFACT,
        schema_version: RL_SCHEMA_VERSION,
        path: RL_REPLAY_INDEX_PATH,
        digest: "sha256:025e9c11831053b7a172b88b2a187ed40095c6323c308d59ce6cccb7eb7e5ee3",
    },
];

/// Closed-field gate: every key of one envelope payload level must belong to
/// the registered field set (top-level `field_order` minus `digest`, plus
/// the `closed` schema-metadata key and the nested `parent.child`
/// vocabulary for that envelope). Dynamic content-reference maps
/// ([`RL_CONTROL_FREE_MAPS`]) skip nested enumeration: their keys are
/// generation/source names, validated as value shapes by the IO consumer.
/// Unknown keys reject with the same sorted `has unknown keys [...]` shape
/// as `rc_require::rc_reject_unknown`. Pure, zero Python API.
pub fn check_envelope_keys(
    artifact: &str,
    keys: &[String],
    nested: &[(String, Vec<String>)],
) -> Result<(), String> {
    let free_maps: &[&str] = match artifact {
        RL_CONTROL_ARTIFACT => &RL_CONTROL_FREE_MAPS,
        _ => &[],
    };
    let allowed: BTreeSet<String> = match artifact {
        RL_PIPELINE_ARTIFACT => RL_PIPELINE_FIELDS
            .iter()
            .filter(|k| **k != "digest")
            .map(|s| (*s).to_owned())
            .chain(std::iter::once("closed".to_owned()))
            .chain(RL_PIPELINE_NESTED_FIELDS.iter().map(|s| (*s).to_owned()))
            .chain(RL_PIPELINE_GUARD_FIELDS.iter().map(|s| (*s).to_owned()))
            .chain(std::iter::once(
                RL_PIPELINE_GRADIENT_CLIP_FIELD.to_owned(),
            ))
            .collect(),
        RL_CONTROL_ARTIFACT => RL_CONTROL_FIELDS
            .iter()
            .filter(|k| **k != "digest")
            .map(|s| (*s).to_owned())
            .chain(std::iter::once("closed".to_owned()))
            .chain(RL_CONTROL_TICKET_FIELDS.iter().map(|s| {
                format!("tickets.{s}")
            }))
            .chain(RL_CONTROL_NESTED_FIELDS.iter().map(|s| (*s).to_owned()))
            .collect(),
        RL_NATIVE_SNAPSHOT_ARTIFACT => RL_NATIVE_SNAPSHOT_FIELDS
            .iter()
            .filter(|k| **k != "digest")
            .map(|s| (*s).to_owned())
            .chain(std::iter::once("closed".to_owned()))
            .collect(),
        RL_REPLAY_INDEX_ARTIFACT => RL_REPLAY_INDEX_FIELDS
            .iter()
            .filter(|k| **k != "digest")
            .map(|s| (*s).to_owned())
            .chain(std::iter::once("closed".to_owned()))
            .collect(),
        _ => {
            return Err(format!("unknown RL schema artifact {artifact:?}"));
        }
    };
    let mut unknown: Vec<&String> = Vec::new();
    for (parent, children) in nested {
        if free_maps.contains(&parent.as_str()) {
            continue;
        }
        for child in children {
            let dotted = format!("{parent}.{child}");
            if !allowed.contains(&dotted) && !allowed.contains(child) {
                unknown.push(child);
            }
        }
    }
    for key in keys {
        if !allowed.contains(key) {
            unknown.push(key);
        }
    }
    if unknown.is_empty() {
        return Ok(());
    }
    let mut rendered: Vec<String> = unknown.iter().map(|k| format!("'{k}'")).collect();
    rendered.sort();
    let mut allowed_list: Vec<String> = allowed.iter().map(|k| format!("'{k}'")).collect();
    allowed_list.sort();
    Err(format!(
        "{artifact} has unknown keys [{}]; allowed=[{}]",
        rendered.join(", "),
        allowed_list.join(", "),
    ))
}

/// Pipeline-only version/required gate: the staged payload must pin schema
/// `3.0.0`, carry a digest-shaped
/// `resource_profile.execution_profile_hash` (the four-field runtime
/// identity the producer stamps from the live runtime spec), and pin the
/// fixed live fields `training_source='native_live_v1'` and
/// `recovery='generation_restart_v1'`. Old `2.0.0`/`1.0.0`
/// envelopes and hash-less payloads reject here even when their own digest
/// is self-consistent, so they can never validate as current pipelines.
/// Pure over the staged `serde_json::Value`; the envelope validators call
/// it for the pipeline artifact only.
pub fn check_pipeline_envelope(payload: &serde_json::Value) -> Result<(), String> {
    let map = payload.as_object().ok_or_else(|| {
        format!("{RL_PIPELINE_ARTIFACT} payload must be a JSON object")
    })?;
    match map.get("schema_version").and_then(|v| v.as_str()) {
        Some(RL_PIPELINE_SCHEMA_VERSION) => {}
        other => {
            return Err(format!(
                "{RL_PIPELINE_ARTIFACT} schema_version must be {RL_PIPELINE_SCHEMA_VERSION:?}, got {other:?}"
            ));
        }
    }
    let hash = map
        .get("resource_profile")
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("execution_profile_hash"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            format!("{RL_PIPELINE_ARTIFACT} lacks resource_profile.execution_profile_hash")
        })?;
    if !is_digest_shape(hash) {
        return Err(format!(
            "{RL_PIPELINE_ARTIFACT} execution_profile_hash must be sha256:<64 hex>, got {hash:?}"
        ));
    }
    match map.get("training_source").and_then(|v| v.as_str()) {
        Some("native_live_v1") => {}
        other => {
            return Err(format!(
                "{RL_PIPELINE_ARTIFACT} training_source must be \"native_live_v1\", got {other:?}"
            ));
        }
    }
    match map.get("recovery").and_then(|v| v.as_str()) {
        Some("generation_restart_v1") => {}
        other => {
            return Err(format!(
                "{RL_PIPELINE_ARTIFACT} recovery must be \"generation_restart_v1\", got {other:?}"
            ));
        }
    }
    Ok(())
}

/// Own-digest-exclusion identity: SHA-256 over the RFC 8785 canonical bytes
/// of the payload object WITHOUT `digest` (matching the
/// `observation_schema_v1.json` convention). Pure over an already-staged
/// `serde_json::Value`; the feed crate owns bytes + hash, never reprinted
/// here. Rejects non-object payloads and a missing/non-string `digest`.
pub fn identity_of_payload_without_digest(
    payload: &serde_json::Value,
) -> Result<(String, serde_json::Value), String> {
    let map = payload.as_object().ok_or_else(|| {
        "RL payload must be a JSON object".to_owned()
    })?;
    let recorded = map.get("digest").and_then(|v| v.as_str()).ok_or_else(|| {
        "RL payload digest must be a sha256:<64 hex> string".to_owned()
    })?;
    if !is_digest_shape(recorded) {
        return Err(format!(
            "RL payload digest must be null-free sha256:<64 hex>, got {recorded:?}"
        ));
    }
    let mut without = map.clone();
    without.remove("digest");
    let value = serde_json::Value::Object(without);
    let bytes = hydra_feed::canon::canonical_bytes_value(&value, "rl_schema_payload")
        .map_err(|e| format!("RL payload not canonical-safe: {e}"))?;
    let recomputed = hydra_feed::digest::sha256_hex(&bytes);
    if recomputed != recorded {
        return Err(format!(
            "RL payload digest mismatch: recorded {recorded} recomputed {recomputed}"
        ));
    }
    Ok((recomputed, value))
}

/// Digest-shape test (restates `rc_parse.rs:172-180`, oracle `_DIGEST_RE`).
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_records_pin_exact_identities() {
        // Pipeline at 2.0.0, the other three envelopes at 1.0.0/exact;
        // distinct artifact types, paths, and digests.
        assert_eq!(RL_SCHEMA_RECORDS.len(), 4);
        let versions: Vec<&str> = RL_SCHEMA_RECORDS.iter().map(|r| r.schema_version).collect();
        assert_eq!(
            versions,
            [
                RL_PIPELINE_SCHEMA_VERSION,
                RL_SCHEMA_VERSION,
                RL_SCHEMA_VERSION,
                RL_SCHEMA_VERSION,
            ]
        );
        for record in &RL_SCHEMA_RECORDS {
            assert!(is_digest_shape(record.digest), "bad digest {}", record.digest);
        }
        let types: Vec<&str> = RL_SCHEMA_RECORDS.iter().map(|r| r.artifact_type).collect();
        assert_eq!(
            types,
            [
                RL_PIPELINE_ARTIFACT,
                RL_CONTROL_ARTIFACT,
                RL_NATIVE_SNAPSHOT_ARTIFACT,
                RL_REPLAY_INDEX_ARTIFACT,
            ]
        );
        // Frozen recipe restates the plan §3 pilot values with epsilon
        // INSIDE the sqrt (SPEC §20.1), never sqrt(var)+eps.
        assert_eq!(RlFrozenRecipe::CLIP_EPS, 0.2);
        assert_eq!(RlFrozenRecipe::EPS_STD, 1e-8);
        assert_eq!(RlFrozenRecipe::W_VALUE, 0.005);
        assert_eq!(RlFrozenRecipe::W_BC, 0.0);
        assert_eq!(RlFrozenRecipe::PPO_ALPHA, 0.01);
        assert_eq!(RlFrozenRecipe::GAMMA, 1.0);
        assert_eq!(RlFrozenRecipe::LAM, 1.0);
        assert_eq!(RlFrozenRecipe::GENERATIONS, 8);
        assert_eq!(RlFrozenRecipe::GAMES_PER_GENERATION, 128);
        assert_eq!(RlFrozenRecipe::ROW_CAP, 262144);
        assert_eq!(RL_HISTORY_BUCKETS, [32, 64, 128, 256]);
    }

    #[test]
    fn closed_gates_reject_unknown_and_wrong_artifact() {
        // Known pipeline keys (top + nested) pass.
        assert!(
            check_envelope_keys(
                RL_PIPELINE_ARTIFACT,
                &["schema_version".to_owned(), "resource_profile".to_owned()],
                &[(
                    "resource_profile".to_owned(),
                    vec!["native_workers".to_owned(), "env_slots".to_owned()]
                )]
            )
            .is_ok()
        );
        // Unknown nested child rejects.
        let err = check_envelope_keys(
            RL_PIPELINE_ARTIFACT,
            &["schema_version".to_owned()],
            &[(
                "resource_profile".to_owned(),
                vec!["native_workers".to_owned(), "bogus_knob".to_owned()],
            )],
        )
        .unwrap_err();
        assert!(err.contains("unknown keys ['bogus_knob']"), "got: {err}");
        assert!(err.contains(RL_PIPELINE_ARTIFACT), "got: {err}");
        // Unknown top-level key rejects with the sorted allowlist shape.
        let err = check_envelope_keys(
            RL_CONTROL_ARTIFACT,
            &["schema_version".to_owned(), "zzz".to_owned()],
            &[],
        )
        .unwrap_err();
        assert!(err.contains("unknown keys ['zzz']"), "got: {err}");
        // Unknown artifact rejects outright.
        assert!(
            check_envelope_keys("hydra2.nope", &[], &[])
                .unwrap_err()
                .contains("unknown RL schema artifact")
        );
    }

    #[test]
    fn pipeline_gate_rejects_old_version_and_missing_hash() {
        // The closed vocabulary admits the new nested execution key.
        assert!(
            check_envelope_keys(
                RL_PIPELINE_ARTIFACT,
                &["schema_version".to_owned(), "resource_profile".to_owned()],
                &[(
                    "resource_profile".to_owned(),
                    vec!["native_workers".to_owned(), "execution_profile_hash".to_owned()]
                )]
            )
            .is_ok()
        );
        // Old 1.0.0 pipeline envelope rejects even with a well-formed hash.
        let old = serde_json::json!({
            "schema_version": "1.0.0",
            "resource_profile": {"execution_profile_hash": format!("sha256:{}", "0".repeat(64))},
            "training_source": "native_live_v1",
            "recovery": "generation_restart_v1",
        });
        assert!(
            check_pipeline_envelope(&old).unwrap_err().contains("must be \"3.0.0\"")
        );
        // Old 2.0.0 pipeline envelope rejects after the live cutover.
        let old_v2 = serde_json::json!({
            "schema_version": "2.0.0",
            "resource_profile": {"execution_profile_hash": format!("sha256:{}", "0".repeat(64))},
            "training_source": "native_live_v1",
            "recovery": "generation_restart_v1",
        });
        assert!(
            check_pipeline_envelope(&old_v2).unwrap_err().contains("must be \"3.0.0\"")
        );
        // Missing execution_profile_hash rejects.
        let bare = serde_json::json!({"schema_version": "3.0.0", "training_source": "native_live_v1", "recovery": "generation_restart_v1"});
        assert!(
            check_pipeline_envelope(&bare)
                .unwrap_err()
                .contains("lacks resource_profile.execution_profile_hash")
        );
        // Malformed hash rejects.
        let malformed = serde_json::json!({
            "schema_version": "3.0.0",
            "resource_profile": {"execution_profile_hash": "nope"},
            "training_source": "native_live_v1",
            "recovery": "generation_restart_v1",
        });
        assert!(
            check_pipeline_envelope(&malformed)
                .unwrap_err()
                .contains("must be sha256:<64 hex>")
        );
        // Wrong live source rejects.
        let bad_source = serde_json::json!({
            "schema_version": "3.0.0",
            "resource_profile": {"execution_profile_hash": format!("sha256:{}", "0".repeat(64))},
            "training_source": "sealed_artifact_v1",
            "recovery": "generation_restart_v1",
        });
        assert!(
            check_pipeline_envelope(&bad_source)
                .unwrap_err()
                .contains("training_source must be")
        );
        // Wrong recovery rejects.
        let bad_recovery = serde_json::json!({
            "schema_version": "3.0.0",
            "resource_profile": {"execution_profile_hash": format!("sha256:{}", "0".repeat(64))},
            "training_source": "native_live_v1",
            "recovery": "exact_resume_v1",
        });
        assert!(
            check_pipeline_envelope(&bad_recovery)
                .unwrap_err()
                .contains("recovery must be")
        );
        // Current shape passes.
        let good = serde_json::json!({
            "schema_version": "3.0.0",
            "resource_profile": {"execution_profile_hash": format!("sha256:{}", "0".repeat(64))},
            "training_source": "native_live_v1",
            "recovery": "generation_restart_v1",
        });
        assert!(check_pipeline_envelope(&good).is_ok());
    }

    #[test]
    fn identity_excludes_own_digest() {
        // A payload whose recorded digest covers the without-digest bytes
        // verifies; tampering the recorded digest fails the comparison.
        let mut map = serde_json::Map::new();
        map.insert(
            "schema_version".to_owned(),
            serde_json::Value::String("1.0.0".to_owned()),
        );
        let bytes = hydra_feed::canon::canonical_bytes_value(
            &serde_json::Value::Object(map.clone()),
            "t",
        )
        .unwrap();
        let digest = hydra_feed::digest::sha256_hex(&bytes);
        map.insert("digest".to_owned(), serde_json::Value::String(digest.clone()));
        let payload = serde_json::Value::Object(map);
        let (recomputed, _) = identity_of_payload_without_digest(&payload).unwrap();
        assert_eq!(recomputed, digest);
        // Corrupt the recorded digest: mismatch, not silent accept.
        let mut bad = payload.as_object().unwrap().clone();
        bad.insert(
            "digest".to_owned(),
            serde_json::Value::String(format!("sha256:{}", "0".repeat(64))),
        );
        assert!(
            identity_of_payload_without_digest(&serde_json::Value::Object(bad))
                .unwrap_err()
                .contains("digest mismatch")
        );
    }
}
