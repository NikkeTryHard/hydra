//! rl/lifecycle_collector: per-generation runtime collector arguments.
//!
//! DAG: `super::lifecycle_common` (gates + descriptor accessors, single
//! home) + `serde_json` + `hydra_feed::canon`. FORBIDS: PyO3, wall-clock,
//! placeholder policies.
//!
//! `collector_spec` emits every runtime collector argument for one
//! generation once its behavior policy id is real — profile, context, the
//! exact 9-key artifact header, disjoint source/eval ledgers, seal
//! locators, and the frozen shuffle key. Needed BEFORE the artifact
//! exists; `prepare_generation` runs only after the validated seal + order.
//! Python never mutates policy metadata: Main passes the published id in,
//! native binds it.

use serde_json::Value;

use super::lifecycle_common::{
    descriptor_doc, descriptor_str, descriptor_u64, is_digest_shape, lc_err, req_i64, req_str,
    req_u64, LifecycleError,
};

/// Shared collector construction for one generation: validates membership,
/// g0 identity, cohort bindings, and training/eval wall separation once,
/// then builds the pool profile, context, frozen shuffle key, source
/// ledger, and archive header inputs. Both the archive `collector_spec`
/// and the live `live_collector_spec` consume this single home, so the
/// live entry can never drift from the archive entry on valid inputs.
pub(crate) struct CollectorCore {
    pub pool_id: String,
    pub profile: Value,
    pub context: Value,
    pub shuffle_key: Value,
    pub source_ledger: Vec<Value>,
    pub header: Value,
    pub eval_walls_dir: Option<Value>,
}

/// Build the shared collector core (see [`CollectorCore`]). The policy id
/// must be real: g0 binds C0 (enforced); later generations bind their
/// published actor supplied at launch. Python never mutates policy
/// metadata — Main passes the published id in, native binds it.
pub(crate) fn collector_core(
    run_descriptor_json: &[u8],
    generation: u64,
    policy_checkpoint_id: &str,
) -> Result<CollectorCore, LifecycleError> {
    if !is_digest_shape(policy_checkpoint_id) {
        return Err(lc_err(format!(
            "rl lifecycle policy_checkpoint_id must be sha256:<64 hex>, got {policy_checkpoint_id:?}"
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
    let cohort = descriptor_doc(doc, "cohort")?;
    let generations_total = req_u64(cohort.get("generations"), "cohort.generations")?;
    if generation >= generations_total {
        return Err(lc_err(format!(
            "rl lifecycle generation {generation} outside 0..{generations_total}"
        )));
    }
    let initial_id = descriptor_str(doc, "initial_checkpoint_id")?;
    if generation == 0 && policy_checkpoint_id != initial_id {
        return Err(lc_err(
            "rl lifecycle g0 policy must be the initial checkpoint (all-seat C0 selfplay)",
        ));
    }
    let generations = doc
        .get("generations")
        .and_then(Value::as_array)
        .ok_or_else(|| lc_err("rl lifecycle descriptor lacks generations"))?;
    let entry = generations.get(generation as usize).ok_or_else(|| {
        lc_err(format!("rl lifecycle generation {generation} outside descriptor cohort"))
    })?;
    let entry_map = entry
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle generation entry must be an object"))?;
    let pool_id = req_str(entry_map.get("pool_id"), "generation.pool_id")?;
    let games = entry_map
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| lc_err("rl lifecycle generation entry lacks games"))?;
    if games.is_empty() {
        return Err(lc_err("rl lifecycle generation cohort must be nonempty"));
    }
    let master_seed_hex = descriptor_str(doc, "master_seed_hex")?;
    let experiment_id = descriptor_str(doc, "experiment_id")?;
    let replicate_id = descriptor_u64(doc, "replicate_id")?;
    let attempt_id = descriptor_u64(doc, "attempt_id")?;
    let pipeline_spec_id = descriptor_str(doc, "pipeline_spec_id")?;
    let adapter_version = descriptor_str(doc, "adapter_version")?;
    let worker_count = req_u64(cohort.get("native_workers"), "cohort.native_workers")?;
    let rules = descriptor_doc(doc, "rules")?;
    let starting_points = req_i64(rules.get("starting_points"), "rules.starting_points")?;
    let identities = descriptor_doc(doc, "identities")?;
    let identity_str = |name: &str| -> Result<String, LifecycleError> {
        req_str(identities.get(name), name)
    };
    let profile = serde_json::json!({
        "pool_id": pool_id,
        "generation": generation,
        "policy_id": policy_checkpoint_id,
        "master_seed_hex": master_seed_hex,
        "starting_points": starting_points,
        "worker_count": worker_count,
        "experiment_id": experiment_id,
        "split_scope": "selection_training",
        "replicate_id": replicate_id,
        "attempt_id": attempt_id,
        "games": games,
    });
    let context = serde_json::json!({
        "rules_id": req_str(rules.get("rules_id"), "rules.rules_id")?,
        "rules_hash": identity_str("rules_hash")?,
        "action_table_hash": identity_str("action_table_hash")?,
        "event_schema_hash": identity_str("event_schema_hash")?,
        "observation_schema_hash": identity_str("observation_schema_hash")?,
        "packet_boundary_hash": identity_str("packet_boundary_hash")?,
        "adapter_version": adapter_version,
        "rank_values": rules.get("rank_values"),
        "behavior_ckpt": policy_checkpoint_id,
    });
    let header = serde_json::json!({
        "rules_hash": identity_str("rules_hash")?,
        "utility_manifest_hash": identity_str("utility_hash")?,
        "observation_schema_hash": identity_str("observation_schema_hash")?,
        "action_table_hash": identity_str("action_table_hash")?,
        "behavior_policy_checkpoint_hash": policy_checkpoint_id,
        "behavior_model_spec_hash": identity_str("model_spec_hash")?,
        "source_wall_ledger_hash": descriptor_str(doc, "wall_ledger_hash")?,
        "row_derivation_hash": descriptor_str(doc, "row_derivation_hash")?,
    });
    let mut source_ledger: Vec<Value> = Vec::with_capacity(games.len());
    for game in games {
        match game.get("wall_id") {
            Some(Value::String(text)) if !text.is_empty() => {
                source_ledger.push(Value::String(text.clone()));
            }
            _ => return Err(lc_err("rl lifecycle ledger game lacks wall_id")),
        }
    }
    let shuffle_key = serde_json::json!({
        "master_seed_hex": master_seed_hex,
        "pipeline_spec_id": pipeline_spec_id,
        "experiment_id": experiment_id,
        "generation": generation,
        "epoch": 0,
        "replicate_id": replicate_id,
        "attempt_id": attempt_id,
    });
    Ok(CollectorCore {
        pool_id,
        profile,
        context,
        shuffle_key,
        source_ledger,
        header,
        eval_walls_dir: doc.get("eval_walls_dir").cloned(),
    })
}

/// Runtime collector arguments for one generation: the pool profile +
/// context bound to the real behavior policy id, the exact 9-key artifact
/// header, the disjoint source/eval ledgers, the seal locators, and the
/// frozen shuffle key. This is needed BEFORE the artifact exists;
/// `prepare_generation` runs only after the validated seal + order.
/// The policy id must be real: g0 binds C0 (enforced); later generations
/// bind their published actor supplied at launch. Python never mutates
/// policy metadata — Main passes the published id in, native binds it.
pub fn collector_spec(
    run_descriptor_json: &[u8],
    generation: u64,
    policy_checkpoint_id: &str,
) -> Result<Vec<u8>, LifecycleError> {
    let core = collector_core(run_descriptor_json, generation, policy_checkpoint_id)?;
    let output = serde_json::json!({
        "lifecycle_version": super::lifecycle::LIFECYCLE_VERSION,
        "generation": generation,
        "pool_id": core.pool_id,
        "policy_checkpoint_id": policy_checkpoint_id,
        "profile": core.profile,
        "context": core.context,
        "header": core.header,
        "source_ledger": core.source_ledger,
        "eval_ledger": [],
        "eval_walls_dir": core.eval_walls_dir,
        "artifact_locator": format!("artifacts/g{generation:02}/artifact.zst"),
        "metas_locator": format!("artifacts/g{generation:02}/metas.json"),
        "shuffle_key": core.shuffle_key,
    });
    hydra_feed::canon::canonical_bytes_value(&output, "rl:collector-spec")
        .map_err(|e| lc_err(format!("rl lifecycle collector spec not canonical-safe: {e}")))
}
