//! rl/lifecycle_generation: canonical per-generation RunSpec + header.
//!
//! DAG: `super::{lifecycle_common, lifecycle_intake}` + `serde_json` +
//! `hydra_feed::canon` + `super::config` (control vocabulary, frozen
//! recipe). FORBIDS: PyO3, wall-clock, second preparation calls.
//!
//! `prepare_generation` binds one generation's sealed artifact + frozen
//! order to the canonical RunSpec document + digest, fixed parent, the
//! 9-field artifact header, deterministic payload skeletons, and a
//! schema-closed control skeleton. The actor id is ALWAYS the source
//! behavior C_g of this generation (g0 enforces C0 == initial id): the
//! first-update C_{g+1} snapshot reuses these same bindings (same RunSpec
//! digest, same parent, same A_g source) with no second preparation call.
//! The ordinary checkpoint's run_spec_hash for the generation IS the
//! RunSpec digest (it binds the exact rollout/order/behavior consumed).

use serde_json::{Map, Value};

use super::lifecycle_common::{
    canonical_digest, check_closed_object, descriptor_doc, descriptor_str, descriptor_u64,
    is_digest_shape, lc_err, req_digest, req_str, req_u64, LifecycleError,
};
use crate::rl::config::{
    RL_CONTROL_ARTIFACT, RL_PIPELINE_SCHEMA_VERSION, RL_PUBLICATION_PHASES, RlFrozenRecipe,
    check_envelope_keys,
};

/// SPEC RunSpec field order (IMPLEMENTATION_SPEC RunSpec dataclass); the
/// digest covers the document minus `digest` (project-wide convention).
/// Law owned HERE (no prior implementation exists): JCS sha256 over the
/// SPEC-ordered document. Every value is real validated input — frozen
/// recipe numerics, sealed source bindings, the fixed parent, and the
/// resolved-config sub-identities carried by the descriptor.
pub const RUNSPEC_FIELDS: [&str; 28] = [
    "schema_version",
    "run_kind",
    "model_spec_hash",
    "initial_checkpoint_hash",
    "dataset_manifest_hash",
    "rollout_artifact_hash",
    "objective_id",
    "objective_parameters",
    "behavior_policy_checkpoint_hash",
    "bc_reference_checkpoint_hash",
    "optimizer_id",
    "optimizer_parameters",
    "scheduler_id",
    "scheduler_parameters",
    "gradient_clip_norm",
    "optimizer_minibatch_size",
    "microbatch_size",
    "accumulation_steps",
    "minibatch_order_hash",
    "max_updates",
    "runtime_spec_hash",
    "resource_ledger_reservation_hash",
    "seed_manifest_hash",
    "selection_metric_id",
    "stopping_rule",
    "checkpoint_frequency_updates",
    "scaler_id",
    "digest",
];


pub fn prepare_generation(
    run_descriptor_json: &[u8],
    generation: u64,
    actor_manifest_id: &str,
    learner_initial_manifest_id: &str,
    artifact_receipt_json: &[u8],
    order_descriptor_json: &[u8],
) -> Result<Vec<u8>, LifecycleError> {
    if !is_digest_shape(actor_manifest_id) {
        return Err(lc_err(format!(
            "rl lifecycle actor_manifest_id must be sha256:<64 hex>, got {actor_manifest_id:?}"
        )));
    }
    if !is_digest_shape(learner_initial_manifest_id) {
        return Err(lc_err(format!(
            "rl lifecycle learner_initial_manifest_id must be sha256:<64 hex>, got {learner_initial_manifest_id:?}"
        )));
    }
    let descriptor: Value = serde_json::from_slice(run_descriptor_json)
        .map_err(|e| lc_err(format!("rl lifecycle run descriptor not JSON: {e}")))?;
    let doc = descriptor
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle run descriptor must be an object"))?;
    match doc.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle descriptor version must be '1.0.0', got {other:?}"
            )))
        }
    }
    // Old 1.0.0 pipeline envelopes never bind a generation: the envelope
    // must pin the schema whose resource profile carries the runtime
    // identity, or shuffle/control bindings would silently reuse stale ids.
    let envelope = descriptor_doc(doc, "pipeline_envelope")?;
    let envelope_version =
        req_str(envelope.get("schema_version"), "pipeline_envelope.schema_version")?;
    if envelope_version != RL_PIPELINE_SCHEMA_VERSION {
        return Err(lc_err(format!(
            "rl lifecycle pipeline_envelope.schema_version must be {RL_PIPELINE_SCHEMA_VERSION:?}, got {envelope_version:?}"
        )));
    }
    let cohort = descriptor_doc(doc, "cohort")?;
    let generations_total = req_u64(cohort.get("generations"), "cohort.generations")?;
    let games_per_generation =
        req_u64(cohort.get("games_per_generation"), "cohort.games_per_generation")?;
    if generation >= generations_total {
        return Err(lc_err(format!(
            "rl lifecycle generation {generation} outside 0..{generations_total}"
        )));
    }
    let pipeline_spec_id = descriptor_str(doc, "pipeline_spec_id")?;
    if !is_digest_shape(&pipeline_spec_id) {
        return Err(lc_err("rl lifecycle descriptor pipeline_spec_id has a bad shape"));
    }
    let initial_id = descriptor_str(doc, "initial_checkpoint_id")?;
    let identities = descriptor_doc(doc, "identities")?;
    let model_spec_hash = req_str(identities.get("model_spec_hash"), "identities.model_spec_hash")?;
    let master_seed_hex = descriptor_str(doc, "master_seed_hex")?;
    let experiment_id = descriptor_str(doc, "experiment_id")?;
    let replicate_id = descriptor_u64(doc, "replicate_id")?;
    let attempt_id = descriptor_u64(doc, "attempt_id")?;
    let environment_hash = descriptor_str(doc, "environment_hash")?;
    let precision = descriptor_str(doc, "precision")?;
    let objective = doc
        .get("objective")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle descriptor lacks objective"))?;
    let optimizer = doc
        .get("optimizer")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle descriptor lacks optimizer"))?;
    let row_cap = req_u64(cohort.get("row_cap"), "cohort.row_cap")?;
    let spec_hashes = doc
        .get("spec_hashes")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle descriptor lacks spec_hashes"))?;
    let runtime_spec_hash =
        req_digest(spec_hashes.get("runtime_spec_hash"), "spec_hashes.runtime_spec_hash")?;
    let seed_manifest_hash =
        req_digest(spec_hashes.get("seed_manifest_hash"), "spec_hashes.seed_manifest_hash")?;
    let selection_doc = doc
        .get("selection")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle descriptor lacks selection"))?;
    let _checkpoint_frequency_updates = descriptor_u64(doc, "checkpoint_frequency_updates")?;
    let wall_ledger_hash = descriptor_str(doc, "wall_ledger_hash")?;
    let row_derivation_hash = descriptor_str(doc, "row_derivation_hash")?;
    let intake_check = RlIntakeCheck {
        microbatch_cap: req_u64(cohort.get("microbatch_cap"), "cohort.microbatch_cap")?,
        accumulation_cap: req_u64(cohort.get("accumulation_cap"), "cohort.accumulation_cap")?,
        optimizer_minibatch_cap: req_u64(
            cohort.get("optimizer_minibatch_cap"),
            "cohort.optimizer_minibatch_cap",
        )?,
    };

    // Cold lineage: g0 collects under C0 from C0. Later generations name
    // their published behavior actor and fixed learner parent explicitly
    // (Main owns that lineage: g1's parent is the final learner of g0, not
    // the behavior C1); native records both without re-deriving.
    if generation == 0 {
        if actor_manifest_id != initial_id {
            return Err(lc_err(
                "rl lifecycle g0 actor must be the initial checkpoint (all-seat C0 selfplay)",
            ));
        }
        if learner_initial_manifest_id != initial_id {
            return Err(lc_err(
                "rl lifecycle g0 learner parent must be the initial checkpoint",
            ));
        }
    }

    // Sealed artifact receipt (ArtifactStorage seal keys).
    let receipt: Value = serde_json::from_slice(artifact_receipt_json)
        .map_err(|e| lc_err(format!("rl lifecycle artifact receipt not JSON: {e}")))?;
    let receipt_map = receipt
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle artifact receipt must be an object"))?;
    check_closed_object(
        receipt_map,
        &["artifact_digest", "row_count", "compressed_bytes", "uncompressed_bytes"],
        "artifact receipt",
    )?;
    let artifact_digest =
        req_digest(receipt_map.get("artifact_digest"), "receipt.artifact_digest")?;
    let row_count = req_u64(receipt_map.get("row_count"), "receipt.row_count")?;
    if row_count == 0 || row_count > row_cap {
        return Err(lc_err(format!(
            "rl lifecycle receipt row_count {row_count} outside 1..={row_cap}"
        )));
    }

    // Frozen minibatch order (replay-index descriptor keys).
    let order: Value = serde_json::from_slice(order_descriptor_json)
        .map_err(|e| lc_err(format!("rl lifecycle order descriptor not JSON: {e}")))?;
    let order_map = order
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle order descriptor must be an object"))?;
    check_closed_object(
        order_map,
        &["artifact_digest", "order_digest", "row_count", "batches"],
        "order descriptor",
    )?;
    let order_artifact = req_digest(order_map.get("artifact_digest"), "order.artifact_digest")?;
    if order_artifact != artifact_digest {
        return Err(lc_err(format!(
            "rl lifecycle order binds {order_artifact} != sealed {artifact_digest}"
        )));
    }
    let order_digest = req_digest(order_map.get("order_digest"), "order.order_digest")?;
    let order_rows = req_u64(order_map.get("row_count"), "order.row_count")?;
    if order_rows != row_count {
        return Err(lc_err(format!(
            "rl lifecycle order rows {order_rows} != sealed {row_count}"
        )));
    }
    let batches = order_map
        .get("batches")
        .and_then(Value::as_array)
        .ok_or_else(|| lc_err("rl lifecycle order lacks batches array"))?;
    if batches.is_empty() {
        return Err(lc_err("rl lifecycle order batches must be nonempty"));
    }
    let mut counted: u64 = 0;
    let mut batch_views: Vec<Value> = Vec::with_capacity(batches.len());
    for (i, batch) in batches.iter().enumerate() {
        let obj = batch
            .as_object()
            .ok_or_else(|| lc_err(format!("rl lifecycle order batches[{i}] must be an object")))?;
        check_closed_object(obj, &["t", "count"], &format!("order batches[{i}]"))?;
        let t = req_u64(obj.get("t"), &format!("order batches[{i}].t"))?;
        if ![32, 64, 128, 256].contains(&t) {
            return Err(lc_err(format!(
                "rl lifecycle order batches[{i}].t={t} outside 32/64/128/256"
            )));
        }
        let count = req_u64(obj.get("count"), &format!("order batches[{i}].count"))?;
        if count == 0 || count > 4096 {
            return Err(lc_err(format!(
                "rl lifecycle order batches[{i}].count={count} outside 1..=4096"
            )));
        }
        counted = counted
            .checked_add(count)
            .ok_or_else(|| lc_err("rl lifecycle order counts overflow"))?;
        batch_views.push(serde_json::json!({ "t": t, "count": count }));
    }
    if counted != row_count {
        return Err(lc_err(format!(
            "rl lifecycle order batch counts sum {counted} != sealed {row_count}"
        )));
    }
    let max_updates = batches.len() as u64;
    if max_updates > RlFrozenRecipe::MAX_UPDATES_PER_GENERATION as u64 {
        return Err(lc_err(format!("rl lifecycle max_updates {max_updates} exceeds 67")));
    }

    // Canonical RunSpec document (SPEC field order; digest covers the
    // document minus `digest`). The ordinary checkpoint's run_spec_hash for
    // this generation IS this digest: it binds the exact rollout, order,
    // behavior, optimizer, runtime, seeds, selection, and resource facts
    // the learner consumes.
    let resource_reservation = serde_json::json!({
        "resource_profile": {
            "native_workers": cohort.get("native_workers"),
            "env_slots": cohort.get("env_slots"),
            "max_lookahead": 1,
            "inference_cap": cohort.get("inference_cap"),
        },
        "host_byte_budget": 12u64 * 1024 * 1024 * 1024,
        "gpu_cache_byte_budget": 1024u64 * 1024 * 1024,
    });
    let resource_ledger_reservation_hash =
        canonical_digest(&resource_reservation, "rl:resource-reservation")?;
    let stopping_rule = serde_json::json!({
        "max_updates": max_updates,
        "selection": Value::Object(selection_doc.clone()),
    });
    // RL carries no scaler (plan section 3: BF16 no scaler, fp32 none);
    // any other precision failed closed at prepare_run.
    let mut runspec = Map::with_capacity(RUNSPEC_FIELDS.len());
    runspec.insert("schema_version".to_string(), Value::String("1.0.0".to_string()));
    runspec.insert("run_kind".to_string(), Value::String("rl".to_string()));
    runspec.insert("model_spec_hash".to_string(), Value::String(model_spec_hash.clone()));
    runspec.insert(
        "initial_checkpoint_hash".to_string(),
        Value::String(learner_initial_manifest_id.to_string()),
    );
    runspec.insert("dataset_manifest_hash".to_string(), Value::Null);
    runspec.insert(
        "rollout_artifact_hash".to_string(),
        Value::String(artifact_digest.clone()),
    );
    runspec.insert(
        "objective_id".to_string(),
        Value::String(super::lifecycle::RL_OBJECTIVE_ID.to_string()),
    );
    runspec.insert(
        "objective_parameters".to_string(),
        serde_json::json!({
            "clip_eps": objective.get("clip_eps"),
            "eps_std": objective.get("eps_std"),
            "w_value": objective.get("w_value"),
            "w_bc": objective.get("w_bc"),
            "alpha": objective.get("alpha"),
        }),
    );
    runspec.insert(
        "behavior_policy_checkpoint_hash".to_string(),
        Value::String(actor_manifest_id.to_string()),
    );
    runspec.insert("bc_reference_checkpoint_hash".to_string(), Value::Null);
    runspec.insert(
        "optimizer_id".to_string(),
        Value::String(super::lifecycle::RL_OPTIMIZER_ID.to_string()),
    );
    runspec.insert(
        "optimizer_parameters".to_string(),
        serde_json::json!({
            "lr": optimizer.get("lr"),
            "betas": [optimizer.get("beta1"), optimizer.get("beta2")],
            "weight_decay": optimizer.get("weight_decay"),
            "head_lr_mult": optimizer.get("head_lr_mult"),
        }),
    );
    runspec.insert(
        "scheduler_id".to_string(),
        Value::String(super::lifecycle::RL_SCHEDULER_ID.to_string()),
    );
    runspec.insert(
        "scheduler_parameters".to_string(),
        serde_json::json!({ "warmup_updates": optimizer.get("warmup_updates") }),
    );
    let gradient_clip_norm = optimizer
        .get("gradient_clip_norm")
        .cloned()
        .ok_or_else(|| lc_err("rl lifecycle descriptor optimizer lacks gradient_clip_norm"))?;
    runspec.insert("gradient_clip_norm".to_string(), gradient_clip_norm);
    runspec.insert(
        "optimizer_minibatch_size".to_string(),
        Value::from(intake_check.optimizer_minibatch_cap),
    );
    runspec.insert("microbatch_size".to_string(), Value::from(intake_check.microbatch_cap));
    runspec.insert("accumulation_steps".to_string(), Value::from(intake_check.accumulation_cap));
    runspec.insert("minibatch_order_hash".to_string(), Value::String(order_digest.clone()));
    runspec.insert("max_updates".to_string(), Value::from(max_updates));
    runspec.insert("runtime_spec_hash".to_string(), Value::String(runtime_spec_hash));
    runspec.insert(
        "resource_ledger_reservation_hash".to_string(),
        Value::String(resource_ledger_reservation_hash),
    );
    runspec.insert("seed_manifest_hash".to_string(), Value::String(seed_manifest_hash));
    runspec.insert(
        "selection_metric_id".to_string(),
        Value::String(super::lifecycle::RL_SELECTION_METRIC_ID.to_string()),
    );
    runspec.insert("stopping_rule".to_string(), stopping_rule);
    runspec.insert(
        "checkpoint_frequency_updates".to_string(),
        Value::from(descriptor_u64(doc, "checkpoint_frequency_updates")?),
    );
    runspec.insert("scaler_id".to_string(), Value::Null);
    {
        let mut keys: Vec<String> = runspec.keys().cloned().collect();
        let mut expected: Vec<String> =
            RUNSPEC_FIELDS.iter().filter(|k| **k != "digest").map(|s| s.to_string()).collect();
        // JSON object identity uses canonical key order, not insertion order;
        // comparing declaration order rejects valid maps before PPO admission.
        expected.sort_unstable();
        keys.sort_unstable();
        if keys != expected {
            return Err(lc_err(format!("rl lifecycle runspec field set drift: got {keys:?}")));
        }
    }
    let runspec_digest = canonical_digest(&Value::Object(runspec.clone()), "rl:runspec")?;
    runspec.insert("digest".to_string(), Value::String(runspec_digest.clone()));

    // Artifact header (exact SPEC12.3 keys) for the seal call.
    let artifact_header = serde_json::json!({
        "schema_version": "1.0.0",
        "rules_hash": identities.get("rules_hash"),
        "utility_manifest_hash": identities.get("utility_hash"),
        "observation_schema_hash": identities.get("observation_schema_hash"),
        "action_table_hash": identities.get("action_table_hash"),
        "behavior_policy_checkpoint_hash": actor_manifest_id,
        "behavior_model_spec_hash": model_spec_hash,
        "source_wall_ledger_hash": wall_ledger_hash,
        "row_derivation_hash": row_derivation_hash,
    });

    // Deterministic payload skeletons: counters and frozen order only.
    // No collector progress, queue telemetry, or timers ever enter the
    // ordinary payload (plan section 7).
    let sampler_skeleton = serde_json::json!({
        "pipeline_spec_id": pipeline_spec_id,
        "generation": generation,
        "epoch": 0,
        "source_hash": artifact_digest,
        "ordered_batches": batch_views,
        "row_count": row_count,
        "optimizer_cursor": 0,
        "micro_offset": 0,
    });
    let training_skeleton = serde_json::json!({
        "completed_updates": 0,
        "generation": generation,
        "epoch": 0,
    });
    let shuffle_key = serde_json::json!({
        "master_seed_hex": master_seed_hex,
        "pipeline_spec_id": pipeline_spec_id,
        "experiment_id": experiment_id,
        "generation": generation,
        "epoch": 0,
        "replicate_id": replicate_id,
        "attempt_id": attempt_id,
    });

    // Control skeleton: schema-closed null slate (progress digests and
    // transport SHAs fill at publish; the digest itself is computed by
    // `snapshot::publish_control`, never here).
    let mut control = Map::with_capacity(25);
    control.insert("schema_version".to_string(), Value::String("1.0.0".to_string()));
    control.insert("pipeline_spec_id".to_string(), Value::String(pipeline_spec_id.clone()));
    control.insert(
        "learner_manifest_id".to_string(),
        Value::String(learner_initial_manifest_id.to_string()),
    );
    control.insert("learner_transport_sha".to_string(), Value::Null);
    control.insert(
        "learner_locator".to_string(),
        Value::String(format!("checkpoints/g{generation:02}/learner.pt")),
    );
    control.insert(
        "actor_manifest_id".to_string(),
        Value::String(actor_manifest_id.to_string()),
    );
    control.insert("actor_transport_sha".to_string(), Value::Null);
    control.insert(
        "actor_locator".to_string(),
        Value::String(format!("checkpoints/g{generation:02}/actor.pt")),
    );
    control.insert("generation".to_string(), Value::from(generation));
    control.insert("generation_statuses".to_string(), Value::Object(Map::new()));
    control.insert("source_refs".to_string(), Value::Object(Map::new()));
    control.insert("order_ref".to_string(), Value::String(order_digest.clone()));
    control.insert("ledger_refs".to_string(), Value::Object(Map::new()));
    control.insert("native_snapshot_digest".to_string(), Value::Null);
    control.insert("tape_blob_digest".to_string(), Value::Null);
    control.insert("event_blob_digest".to_string(), Value::Null);
    control.insert(
        "cohort_membership".to_string(),
        serde_json::json!({
            "generations": generations_total,
            "games_per_generation": games_per_generation,
        }),
    );
    control.insert(
        "frontier_current".to_string(),
        serde_json::json!({"generation": generation, "round": 0, "epoch": 0}),
    );
    control.insert(
        "frontier_next".to_string(),
        serde_json::json!({"generation": generation, "round": 0, "epoch": 0}),
    );
    control.insert("terminal_set".to_string(), Value::Array(vec![]));
    control.insert("tickets".to_string(), Value::Array(vec![]));
    control.insert("committed_row_offset".to_string(), Value::from(0u64));
    control.insert("committed_update_offset".to_string(), Value::from(0u64));
    control.insert(
        "publication_phase".to_string(),
        Value::String(RL_PUBLICATION_PHASES[0].to_string()),
    );
    control.insert(
        "parent_checkpoint_hash".to_string(),
        Value::String(learner_initial_manifest_id.to_string()),
    );
    {
        let keys: Vec<String> = control.keys().cloned().collect();
        let nested: Vec<(String, Vec<String>)> = vec![
            ("cohort_membership".to_string(), vec![
                "generations".to_string(),
                "games_per_generation".to_string(),
            ]),
            ("frontier_current".to_string(), vec![
                "generation".to_string(),
                "round".to_string(),
                "epoch".to_string(),
            ]),
            ("frontier_next".to_string(), vec![
                "generation".to_string(),
                "round".to_string(),
                "epoch".to_string(),
            ]),
            ("generation_statuses".to_string(), vec![]),
            ("source_refs".to_string(), vec![]),
            ("ledger_refs".to_string(), vec![]),
        ];
        check_envelope_keys(RL_CONTROL_ARTIFACT, &keys, &nested)
            .map_err(|e| lc_err(format!("rl lifecycle control skeleton rejected: {e}")))?;
    }

    let published_actor = if generation + 1 < generations_total {
        Value::String(format!("checkpoints/g{:02}/actor.pt", generation + 1))
    } else {
        Value::Null
    };
    let output = serde_json::json!({
        "lifecycle_version": super::lifecycle::LIFECYCLE_VERSION,
        "generation": generation,
        "cohort_generations": generations_total,
        "cohort_games": games_per_generation,
        "pipeline_spec_id": pipeline_spec_id,
        "actor_manifest_id": actor_manifest_id,
        "learner_initial_manifest_id": learner_initial_manifest_id,
        "parent_checkpoint_hash": learner_initial_manifest_id,
        "artifact_digest": artifact_digest,
        "order_digest": order_digest,
        "row_count": row_count,
        "max_updates": max_updates,
        "runspec": runspec,
        "runspec_digest": runspec_digest,
        "environment_hash": environment_hash,
        "precision": precision,
        "artifact_header": artifact_header,
        "source_identities": {
            "model_spec_hash": model_spec_hash,
            "rules_hash": identities.get("rules_hash"),
            "utility_hash": identities.get("utility_hash"),
            "action_table_hash": identities.get("action_table_hash"),
            "observation_schema_hash": identities.get("observation_schema_hash"),
        },
        "sampler_skeleton": sampler_skeleton,
        "training_skeleton": training_skeleton,
        "shuffle_key": shuffle_key,
        "control_skeleton": control,
        "ordinary_locators": {
            "collection_learner": format!("checkpoints/g{generation:02}/learner.pt"),
            "collection_actor": format!("checkpoints/g{generation:02}/actor.pt"),
            "published_actor": published_actor,
        },
        "publication_note": "the first-update C_{g+1} snapshot reuses this generation's runspec digest, parent, and A_g source with no second preparation call; the published actor lands at published_actor",
    });
    hydra_feed::canon::canonical_bytes_value(&output, "rl:generation-bindings")
        .map_err(|e| lc_err(format!("rl lifecycle generation output not canonical-safe: {e}")))
}

struct RlIntakeCheck {
    microbatch_cap: u64,
    accumulation_cap: u64,
    optimizer_minibatch_cap: u64,
}
