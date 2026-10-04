//! rl/lifecycle_control: control assembly + ordinary-manifest binding.
//!
//! DAG: `super::{lifecycle_common, lifecycle_intake}` + `serde_json` +
//! `hydra_feed::canon` + `super::config` (control vocabulary). FORBIDS:
//! PyO3, wall-clock, digest computation (the digest + commit authority is
//! `snapshot::publish_control`, which re-verifies receipts against disk and
//! enforces the fixed parent + phase monotonicity).
//!
//! `build_control` assembles the control document from landed receipts and
//! runtime facts. `ordinary_bindings` verifies a Python-built ordinary
//! manifest against the generation bindings and returns its FULL-JCS id.
//! No replay logic lives here or in Python: replay is the native
//! pool/snapshot restore path over stored action prefixes.

use serde_json::{Map, Value};

use super::lifecycle_common::{
    check_closed_object, is_digest_shape, lc_err, req_digest, req_str, req_u64, LifecycleError,
};
use super::lifecycle_intake::{manifest_id_of, manifest_shape_gate};
use crate::rl::config::{
    RL_CONTROL_ARTIFACT, RL_PUBLICATION_PHASES, RL_TICKET_PHASES, check_envelope_keys,
};

/// Closed ticket-entry vocabulary (`RL_CONTROL_TICKET_FIELDS`).
pub const CONTROL_TICKET_FIELDS: [&str; 4] =
    ["ticket_descriptor", "phase", "slot_epoch", "stream_cursor"];

fn req_frontier(value: Option<&Value>, name: &str) -> Result<(u64, u64, u64), LifecycleError> {
    let obj = value
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err(format!("rl lifecycle {name} must be an object")))?;
    check_closed_object(obj, &["generation", "round", "epoch"], name)?;
    Ok((
        req_u64(obj.get("generation"), &format!("{name}.generation"))?,
        req_u64(obj.get("round"), &format!("{name}.round"))?,
        req_u64(obj.get("epoch"), &format!("{name}.epoch"))?,
    ))
}

/// Assemble the control document from landed receipts and runtime facts.
/// The digest is NOT computed here: `snapshot::publish_control` digests,
/// verifies, and commits. Receipt locators must match the generation
/// bindings (learner: collection slot; actor: collection or published
/// slot); receipt manifest ids must equal the bound learner/actor ids;
/// the fixed parent rides from the bindings unchanged.
#[allow(clippy::too_many_arguments)]
pub fn build_control(
    generation_bindings_json: &[u8],
    learner_receipt_json: &[u8],
    actor_receipt_json: &[u8],
    pool_snapshot_digest: &str,
    tape_blob_digest: &str,
    tape_blob_locator: &str,
    event_blob_digest: &str,
    event_blob_locator: &str,
    frontier_current_json: &str,
    frontier_next_json: &str,
    terminal_set: &[String],
    tickets_json: &[u8],
    committed_row_offset: u64,
    committed_update_offset: u64,
    publication_phase: &str,
    generation_statuses_json: &[u8],
    source_refs_json: &[u8],
    ledger_refs_json: &[u8],
) -> Result<Vec<u8>, LifecycleError> {
    let bindings: Value = serde_json::from_slice(generation_bindings_json)
        .map_err(|e| lc_err(format!("rl lifecycle generation bindings not JSON: {e}")))?;
    let bind = bindings
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle generation bindings must be an object"))?;
    match bind.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle bindings version must be '1.0.0', got {other:?}"
            )))
        }
    }
    let generation = req_u64(bind.get("generation"), "bindings.generation")?;
    let cohort_generations = req_u64(bind.get("cohort_generations"), "bindings.cohort_generations")?;
    let cohort_games = req_u64(bind.get("cohort_games"), "bindings.cohort_games")?;
    let pipeline_spec_id = req_digest(bind.get("pipeline_spec_id"), "bindings.pipeline_spec_id")?;
    let learner_init =
        req_digest(bind.get("learner_initial_manifest_id"), "bindings.learner_init")?;
    let actor_id = req_digest(bind.get("actor_manifest_id"), "bindings.actor")?;
    let order_ref = req_digest(bind.get("order_digest"), "bindings.order")?;
    let locators = bind
        .get("ordinary_locators")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle bindings lack ordinary_locators"))?;
    let learner_locator =
        req_str(locators.get("collection_learner"), "locators.collection_learner")?;
    let actor_collection =
        req_str(locators.get("collection_actor"), "locators.collection_actor")?;
    let actor_published = match locators.get("published_actor") {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Null) | None => None,
        other => {
            return Err(lc_err(format!(
                "rl lifecycle published_actor must be a string or null, got {other:?}"
            )))
        }
    };

    // Receipts: closed 4-key shape (ArtifactStorage `checkpoint_receipt`
    // law), manifest-id binding, locator binding. On-disk re-verification
    // and phase monotonicity belong to `publish_control` (the committing
    // authority), not to this assembler.
    let learner_receipt: Value = serde_json::from_slice(learner_receipt_json)
        .map_err(|e| lc_err(format!("rl lifecycle learner receipt not JSON: {e}")))?;
    let actor_receipt: Value = serde_json::from_slice(actor_receipt_json)
        .map_err(|e| lc_err(format!("rl lifecycle actor receipt not JSON: {e}")))?;
    for (name, receipt) in [("learner", &learner_receipt), ("actor", &actor_receipt)] {
        let map = receipt
            .as_object()
            .ok_or_else(|| lc_err(format!("rl lifecycle {name} receipt must be an object")))?;
        check_closed_object(
            map,
            &["manifest_id", "transport_sha256", "byte_len", "locator"],
            &format!("{name} receipt"),
        )?;
        req_digest(map.get("manifest_id"), &format!("{name} receipt.manifest_id"))?;
        req_digest(map.get("transport_sha256"), &format!("{name} receipt.transport_sha256"))?;
        req_u64(map.get("byte_len"), &format!("{name} receipt.byte_len"))?;
        req_str(map.get("locator"), &format!("{name} receipt.locator"))?;
    }
    let learner_manifest =
        learner_receipt.get("manifest_id").and_then(Value::as_str).unwrap_or("");
    if learner_manifest != learner_init {
        return Err(lc_err(format!(
            "rl lifecycle learner receipt binds {learner_manifest} != generation parent {learner_init}"
        )));
    }
    if learner_receipt.get("locator").and_then(Value::as_str).unwrap_or("") != learner_locator {
        return Err(lc_err(
            "rl lifecycle learner receipt locator != collection learner slot",
        ));
    }
    let actor_manifest =
        actor_receipt.get("manifest_id").and_then(Value::as_str).unwrap_or("");
    if actor_manifest != actor_id {
        return Err(lc_err(format!(
            "rl lifecycle actor receipt binds {actor_manifest} != generation actor {actor_id}"
        )));
    }
    let actor_locator = actor_receipt.get("locator").and_then(Value::as_str).unwrap_or("");
    if actor_locator != actor_collection && Some(actor_locator) != actor_published.as_deref() {
        return Err(lc_err(
            "rl lifecycle actor receipt locator is neither the collection nor the published actor slot",
        ));
    }

    if !is_digest_shape(pool_snapshot_digest) {
        return Err(lc_err("rl lifecycle pool_snapshot_digest must be sha256:<64 hex>"));
    }
    for (name, digest) in [("tape", tape_blob_digest), ("event", event_blob_digest)] {
        if !is_digest_shape(digest) {
            return Err(lc_err(format!(
                "rl lifecycle {name}_blob_digest must be sha256:<64 hex>"
            )));
        }
    }
    if tape_blob_locator.is_empty() || event_blob_locator.is_empty() {
        return Err(lc_err("rl lifecycle blob locators must be nonempty"));
    }
    let frontier_current: Value = serde_json::from_str(frontier_current_json)
        .map_err(|e| lc_err(format!("rl lifecycle frontier_current not JSON: {e}")))?;
    let frontier_next: Value = serde_json::from_str(frontier_next_json)
        .map_err(|e| lc_err(format!("rl lifecycle frontier_next not JSON: {e}")))?;
    let (cur_g, cur_r, cur_e) = req_frontier(Some(&frontier_current), "frontier_current")?;
    let (nxt_g, nxt_r, nxt_e) = req_frontier(Some(&frontier_next), "frontier_next")?;
    if cur_g != generation || nxt_g != generation {
        return Err(lc_err(format!(
            "rl lifecycle frontiers bind generation {cur_g}/{nxt_g} != bindings {generation}"
        )));
    }
    if !RL_PUBLICATION_PHASES.contains(&publication_phase) {
        return Err(lc_err(format!(
            "rl lifecycle publication_phase must be one of {RL_PUBLICATION_PHASES:?}, got {publication_phase:?}"
        )));
    }
    let tickets: Value = serde_json::from_slice(tickets_json)
        .map_err(|e| lc_err(format!("rl lifecycle tickets not JSON: {e}")))?;
    let tickets_array = tickets
        .as_array()
        .ok_or_else(|| lc_err("rl lifecycle tickets must be an array"))?;
    for (i, ticket) in tickets_array.iter().enumerate() {
        let obj = ticket
            .as_object()
            .ok_or_else(|| lc_err(format!("rl lifecycle tickets[{i}] must be an object")))?;
        check_closed_object(obj, &CONTROL_TICKET_FIELDS, &format!("tickets[{i}]"))?;
        req_str(obj.get("ticket_descriptor"), &format!("tickets[{i}].ticket_descriptor"))?;
        match obj.get("phase").and_then(Value::as_str) {
            Some(phase) if RL_TICKET_PHASES.contains(&phase) => {}
            other => {
                return Err(lc_err(format!(
                    "rl lifecycle tickets[{i}].phase must be one of {RL_TICKET_PHASES:?}, got {other:?}"
                )))
            }
        }
        req_u64(obj.get("slot_epoch"), &format!("tickets[{i}].slot_epoch"))?;
        req_u64(obj.get("stream_cursor"), &format!("tickets[{i}].stream_cursor"))?;
    }
    let generation_statuses: Value = serde_json::from_slice(generation_statuses_json)
        .map_err(|e| lc_err(format!("rl lifecycle generation_statuses not JSON: {e}")))?;
    let statuses_map = generation_statuses
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle generation_statuses must be an object"))?;
    for (key, value) in statuses_map {
        match value.as_str() {
            Some("collected") | Some("sealed") | Some("trained") | Some("published") => {}
            other => {
                return Err(lc_err(format!(
                    "rl lifecycle generation_statuses[{key:?}] must be collected/sealed/trained/published, got {other:?}"
                )))
            }
        }
    }
    for (name, doc) in [("source_refs", source_refs_json), ("ledger_refs", ledger_refs_json)] {
        let refs: Value = serde_json::from_slice(doc)
            .map_err(|e| lc_err(format!("rl lifecycle {name} not JSON: {e}")))?;
        let refs_map = refs
            .as_object()
            .ok_or_else(|| lc_err(format!("rl lifecycle {name} must be an object")))?;
        for (key, value) in refs_map {
            match value.as_str() {
                Some(text) if is_digest_shape(text) => {}
                _ => {
                    return Err(lc_err(format!(
                        "rl lifecycle {name}[{key:?}] must be sha256:<64 hex>"
                    )))
                }
            }
        }
    }

    let mut control = Map::with_capacity(25);
    control.insert("schema_version".to_string(), Value::String("1.0.0".to_string()));
    control.insert("pipeline_spec_id".to_string(), Value::String(pipeline_spec_id));
    control.insert("learner_manifest_id".to_string(), Value::String(learner_init.clone()));
    control.insert(
        "learner_transport_sha".to_string(),
        learner_receipt.get("transport_sha256").cloned().unwrap_or(Value::Null),
    );
    control.insert("learner_locator".to_string(), Value::String(learner_locator));
    control.insert("actor_manifest_id".to_string(), Value::String(actor_id));
    control.insert(
        "actor_transport_sha".to_string(),
        actor_receipt.get("transport_sha256").cloned().unwrap_or(Value::Null),
    );
    control.insert("actor_locator".to_string(), Value::String(actor_locator.to_string()));
    control.insert("generation".to_string(), Value::from(generation));
    control.insert("generation_statuses".to_string(), generation_statuses.clone());
    control.insert(
        "source_refs".to_string(),
        serde_json::from_slice(source_refs_json)
            .map_err(|e| lc_err(format!("rl lifecycle source_refs not JSON: {e}")))?,
    );
    control.insert("order_ref".to_string(), Value::String(order_ref));
    control.insert(
        "ledger_refs".to_string(),
        serde_json::from_slice(ledger_refs_json)
            .map_err(|e| lc_err(format!("rl lifecycle ledger_refs not JSON: {e}")))?,
    );
    control.insert(
        "native_snapshot_digest".to_string(),
        Value::String(pool_snapshot_digest.to_string()),
    );
    control.insert("tape_blob_digest".to_string(), Value::String(tape_blob_digest.to_string()));
    control.insert(
        "event_blob_digest".to_string(),
        Value::String(event_blob_digest.to_string()),
    );
    control.insert(
        "cohort_membership".to_string(),
        serde_json::json!({
            "generations": cohort_generations,
            "games_per_generation": cohort_games,
        }),
    );
    control.insert(
        "frontier_current".to_string(),
        serde_json::json!({"generation": cur_g, "round": cur_r, "epoch": cur_e}),
    );
    control.insert(
        "frontier_next".to_string(),
        serde_json::json!({"generation": nxt_g, "round": nxt_r, "epoch": nxt_e}),
    );
    control.insert(
        "terminal_set".to_string(),
        Value::Array(terminal_set.iter().map(|s| Value::String(s.clone())).collect()),
    );
    control.insert("tickets".to_string(), Value::Array(tickets_array.clone()));
    control.insert("committed_row_offset".to_string(), Value::from(committed_row_offset));
    control.insert("committed_update_offset".to_string(), Value::from(committed_update_offset));
    control.insert(
        "publication_phase".to_string(),
        Value::String(publication_phase.to_string()),
    );
    control.insert("parent_checkpoint_hash".to_string(), Value::String(learner_init));
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
            ("generation_statuses".to_string(), statuses_map.keys().cloned().collect()),
            ("source_refs".to_string(), vec![]),
            ("ledger_refs".to_string(), vec![]),
        ];
        check_envelope_keys(RL_CONTROL_ARTIFACT, &keys, &nested)
            .map_err(|e| lc_err(format!("rl lifecycle control document rejected: {e}")))?;
    }
    hydra_feed::canon::canonical_bytes_value(&Value::Object(control), "rl:control-doc")
        .map_err(|e| lc_err(format!("rl lifecycle control document not canonical-safe: {e}")))
}

/// Verify a Python-built ordinary manifest against the generation bindings
/// (source, parent, run-spec, and contract identities) and return its
/// FULL-JCS id. Section hashing stays in the existing Python writer; this
/// gate binds the manifest's identity claims to the native bindings before
/// any receipt or control publication references them.
pub fn ordinary_bindings(
    generation_bindings_json: &[u8],
    manifest_json: &[u8],
) -> Result<Vec<u8>, LifecycleError> {
    let bindings: Value = serde_json::from_slice(generation_bindings_json)
        .map_err(|e| lc_err(format!("rl lifecycle generation bindings not JSON: {e}")))?;
    let bind = bindings
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle generation bindings must be an object"))?;
    let runspec_digest = req_digest(bind.get("runspec_digest"), "bindings.runspec_digest")?;
    let learner_init =
        req_digest(bind.get("learner_initial_manifest_id"), "bindings.learner_init")?;
    let artifact_digest = req_digest(bind.get("artifact_digest"), "bindings.artifact")?;
    let sources = bind
        .get("source_identities")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle bindings lack source_identities"))?;

    let map = manifest_shape_gate(manifest_json, "ordinary")?;
    let get_digest = |key: &str| -> Result<Option<String>, LifecycleError> {
        match map.get(key) {
            Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            _ => Err(lc_err(format!("rl lifecycle ordinary manifest field {key} has a bad shape"))),
        }
    };
    let expect = |key: &str, want: &str| -> Result<(), LifecycleError> {
        match get_digest(key)? {
            Some(got) if got == want => Ok(()),
            other => Err(lc_err(format!(
                "rl lifecycle ordinary manifest {key} binds {other:?} != generation {want}"
            ))),
        }
    };
    expect("run_spec_hash", &runspec_digest)?;
    expect("rollout_artifact_hash", &artifact_digest)?;
    expect("parent_checkpoint_hash", &learner_init)?;
    if get_digest("dataset_manifest_hash")?.is_some() {
        return Err(lc_err(
            "rl lifecycle ordinary manifest must carry null dataset (immutable rollout source)",
        ));
    }
    for (manifest_key, source_key) in [
        ("model_spec_hash", "model_spec_hash"),
        ("rules_hash", "rules_hash"),
        ("utility_manifest_hash", "utility_hash"),
        ("action_schema_hash", "action_table_hash"),
        ("observation_schema_hash", "observation_schema_hash"),
    ] {
        let want = sources
            .get(source_key)
            .and_then(Value::as_str)
            .ok_or_else(|| lc_err(format!("rl lifecycle bindings lack source identity {source_key}")))?;
        expect(manifest_key, want)?;
    }
    let manifest_id = manifest_id_of(&map, "rl:ordinary-manifest")?;
    let output = serde_json::json!({
        "lifecycle_version": super::lifecycle::LIFECYCLE_VERSION,
        "manifest_id": manifest_id,
        "run_spec_hash": runspec_digest,
        "rollout_artifact_hash": artifact_digest,
        "parent_checkpoint_hash": learner_init,
    });
    hydra_feed::canon::canonical_bytes_value(&output, "rl:ordinary-bindings")
        .map_err(|e| lc_err(format!("rl lifecycle ordinary bindings not canonical-safe: {e}")))
}
