//! rl/lifecycle_run: cold `prepare_run` descriptor.
//!
//! DAG: `super::{lifecycle_common, lifecycle_intake, lifecycle_ledger}` +
//! `serde_json` + `hydra_feed::canon` + `super::config` (envelope gates,
//! frozen recipe). FORBIDS: PyO3, wall-clock, placeholder policies.
//!
//! `prepare_run` validates the resolved run config + IL manifest against
//! real repo assets and forms the run descriptor (no profiles: only C0
//! exists at cold time). Per-generation collector arguments live in
//! `lifecycle_collector::collector_spec`.

use serde_json::{Map, Value};

use super::lifecycle_common::{
    LifecycleError, canonical_digest, check_closed_object, is_digest_shape, lc_err, req_digest,
    req_i64, req_str, req_u64,
};
use super::lifecycle_intake::{
    parse_rl_intake, recipe_digests, row_derivation_doc, verify_assets, verify_initial_manifest,
};
use super::lifecycle_ledger::{generate_ledger, ledger_digest, ledger_split_for};
use crate::rl::config::{RL_PIPELINE_ARTIFACT, RL_PIPELINE_SCHEMA_VERSION, RlFrozenRecipe, check_envelope_keys};

/// Closed top-level input vocabulary of `prepare_run`: the resolved config
/// document (framed by the existing `run_config_to_dict` authority), the
/// live adapter version, the verified utility values, the immutable
/// opponent catalog, and the live engine environment hash. The initial
/// checkpoint locator is NOT part of any config: Main passes the manifest
/// content via `initial_manifest_json` after copying the verified file to
/// `checkpoints/c0.pt` with no sidecar.
pub const PREPARE_RUN_FIELDS: [&str; 5] = [
    "resolved",
    "adapter_version",
    "utility",
    "opponent_catalog",
    "environment_hash",
];

/// Resolved-document sections required for an RL run (`data` must be null:
/// RL forbids dataset roots; `rl` carries the closed profile).
pub const RESOLVED_FIELDS: [&str; 14] = [
    "run",
    "data",
    "model",
    "weights",
    "optimizer",
    "scheduler",
    "runtime",
    "loop",
    "seeds",
    "selection",
    "mirror",
    "eval",
    "output",
    "rl",
];

/// Closed `utility` intake vocabulary (values verified against the live
/// utility manifest Python-side; rank_values cross-checked against the
/// rules-file uma in the asset gate).
pub const PREPARE_RUN_UTILITY_FIELDS: [&str; 2] = ["utility_id", "rank_values"];

/// Cold run descriptor: validate config + manifest against real assets,
/// then form every structural identity natively. Returns canonical JSON
/// bytes of the descriptor (field contract in the facade docs).
pub fn prepare_run(
    run_config_json: &str,
    initial_manifest_json: &str,
    root_dir: &str,
    repo_dir: &str,
) -> Result<Vec<u8>, LifecycleError> {
    if root_dir.is_empty() {
        return Err(lc_err("rl lifecycle root_dir must be nonempty"));
    }
    if repo_dir.is_empty() {
        return Err(lc_err("rl lifecycle repo_dir must be nonempty"));
    }
    let config: Value = serde_json::from_str(run_config_json)
        .map_err(|e| lc_err(format!("rl lifecycle run config not JSON: {e}")))?;
    let config_map = config
        .as_object()
        .ok_or_else(|| lc_err("rl lifecycle run config must be an object"))?;
    check_closed_object(config_map, &PREPARE_RUN_FIELDS, "run config")?;
    let adapter_version = req_str(config_map.get("adapter_version"), "adapter_version")?;
    let environment_hash = req_digest(config_map.get("environment_hash"), "environment_hash")?;

    let utility_map = config_map
        .get("utility")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle run config lacks utility object"))?;
    check_closed_object(utility_map, &PREPARE_RUN_UTILITY_FIELDS, "utility")?;
    let utility_id = req_str(utility_map.get("utility_id"), "utility.utility_id")?;
    let ranks_value = utility_map
        .get("rank_values")
        .and_then(Value::as_array)
        .ok_or_else(|| lc_err("rl lifecycle utility.rank_values must hold four numbers"))?;
    if ranks_value.len() != 4 {
        return Err(lc_err("rl lifecycle utility.rank_values must hold four numbers"));
    }
    let mut utility_ranks = [0.0f64; 4];
    for (i, entry) in ranks_value.iter().enumerate() {
        let v = entry
            .as_f64()
            .ok_or_else(|| lc_err(format!("rl lifecycle utility.rank_values[{i}] must be a number")))?;
        if !v.is_finite() {
            return Err(lc_err("rl lifecycle utility.rank_values must be finite"));
        }
        utility_ranks[i] = v;
    }
    let catalog_value = config_map
        .get("opponent_catalog")
        .ok_or_else(|| lc_err("rl lifecycle run config lacks opponent_catalog"))?;
    let catalog_array = catalog_value
        .as_array()
        .ok_or_else(|| lc_err("rl lifecycle opponent_catalog must be an array"))?;
    for (i, entry) in catalog_array.iter().enumerate() {
        let obj = entry
            .as_object()
            .ok_or_else(|| lc_err(format!("rl lifecycle opponent_catalog[{i}] must be an object")))?;
        check_closed_object(obj, &["checkpoint_id", "role"], &format!("opponent_catalog[{i}]"))?;
        req_digest(obj.get("checkpoint_id"), &format!("opponent_catalog[{i}].checkpoint_id"))?;
        req_str(obj.get("role"), &format!("opponent_catalog[{i}].role"))?;
    }

    // Resolved document (existing framing authority): every section present,
    // dataset source absent. The digest over these exact bytes IS the
    // run_spec_hash lineage anchor (same stage+JCS+hash law as
    // `run_config_digest_from_dict`).
    let resolved = config_map
        .get("resolved")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle run config lacks resolved object"))?;
    check_closed_object(resolved, &RESOLVED_FIELDS, "resolved")?;
    match resolved.get("data") {
        Some(Value::Null) => {}
        _ => {
            return Err(lc_err(
                "rl lifecycle resolved.data must be null (RL forbids dataset roots)",
            ))
        }
    }
    let run_doc = resolved
        .get("run")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks run object"))?;
    let run_id = req_str(run_doc.get("id"), "resolved.run.id")?;
    let run_kind = req_str(run_doc.get("kind"), "resolved.run.kind")?;
    if run_kind != "rl" {
        return Err(lc_err(format!(
            "rl lifecycle resolved.run.kind must be 'rl', got {run_kind:?}"
        )));
    }
    let model_doc = resolved
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks model object"))?;
    let architecture_id =
        req_str(model_doc.get("architecture_id"), "resolved.model.architecture_id")?;
    let action_count = req_u64(model_doc.get("action_count"), "resolved.model.action_count")?;
    if action_count != 6792 {
        return Err(lc_err(format!(
            "rl lifecycle resolved.model.action_count {action_count} != baseline 6792"
        )));
    }
    let rl_map = resolved
        .get("rl")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks rl object"))?;
    let intake = parse_rl_intake(rl_map)?;

    // RunSpec sub-identity inputs: all from the real resolved sections,
    // every field required, no defaults.
    let runtime_doc = resolved
        .get("runtime")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks runtime object"))?;
    check_closed_object(
        runtime_doc,
        &["adapter_id", "device", "precision", "compile_mode"],
        "resolved.runtime",
    )?;
    let precision = req_str(runtime_doc.get("precision"), "resolved.runtime.precision")?;
    if precision != "fp32" && precision != "bf16_mixed" {
        return Err(lc_err(format!(
            "rl lifecycle resolved.runtime.precision must be fp32 or bf16_mixed (RL carries no scaler), got {precision:?}"
        )));
    }
    let seeds_doc = resolved
        .get("seeds")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks seeds object"))?;
    check_closed_object(
        seeds_doc,
        &["data_seed", "train_seed", "selection_seed"],
        "resolved.seeds",
    )?;
    for name in ["data_seed", "train_seed", "selection_seed"] {
        req_i64(seeds_doc.get(name), &format!("resolved.seeds.{name}"))?;
    }
    let selection_doc = resolved
        .get("selection")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks selection object"))?;
    check_closed_object(
        selection_doc,
        &[
            "N",
            "pilot_s",
            "delta",
            "alpha",
            "beta",
            "design",
            "declared_peeks",
            "margin",
            "resamples",
            "seed",
        ],
        "resolved.selection",
    )?;
    let loop_doc = resolved
        .get("loop")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle resolved lacks loop object"))?;
    let checkpoint_frequency_updates = req_u64(
        loop_doc.get("checkpoint_frequency_updates"),
        "resolved.loop.checkpoint_frequency_updates",
    )?;
    if checkpoint_frequency_updates == 0 {
        return Err(lc_err(
            "rl lifecycle resolved.loop.checkpoint_frequency_updates must be positive",
        ));
    }
    let eval_walls_dir = match resolved.get("eval").and_then(|e| e.get("walls_dir")) {
        Some(Value::Null) | None => Value::Null,
        Some(Value::String(dir)) if !dir.is_empty() => Value::String(dir.clone()),
        other => {
            return Err(lc_err(format!(
                "rl lifecycle resolved.eval.walls_dir must be a path string or null, got {other:?}"
            )))
        }
    };
    let runtime_spec_hash =
        canonical_digest(&Value::Object(runtime_doc.clone()), "rl:runtime-spec")?;
    let seed_manifest_hash =
        canonical_digest(&Value::Object(seeds_doc.clone()), "rl:seed-manifest")?;
    // run_spec_hash lineage anchor: JCS sha over the resolved document
    // bytes (existing `run_config_digest_from_dict` law).
    let run_spec_hash = canonical_digest(&Value::Object(resolved.clone()), "rl:run-spec")?;

    let assets = verify_assets(repo_dir, &intake, &utility_ranks)?;
    let initial_id = verify_initial_manifest(initial_manifest_json.as_bytes(), &intake)?;

    // Composition hashes: required pinned, enforced exactly. A mismatch
    // names the natively computed value so Main can pin it; the run never
    // proceeds unpinned.
    let recipes = recipe_digests()?;
    let recipe_str = |name: &str| -> Result<String, LifecycleError> {
        recipes
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| lc_err(format!("rl lifecycle recipe_digests lacks {name}")))
    };
    let wall_recipe = recipe_str("wall_recipe_hash")?;
    let sampling_recipe = recipe_str("sampling_recipe_hash")?;
    let return_recipe = recipe_str("return_recipe_hash")?;
    let shuffle_recipe = recipe_str("shuffle_recipe_hash")?;
    let check_pinned =
        |name: &str, configured: &str, computed: &str| -> Result<(), LifecycleError> {
            if configured != computed {
                return Err(lc_err(format!(
                    "rl lifecycle {name} mismatch: config {configured} != implementation {computed} (pin via recipe_digests)"
                )));
            }
            Ok(())
        };
    check_pinned("wall_recipe_hash", &intake.wall_recipe_hash, &wall_recipe)?;
    check_pinned("sampling_recipe_hash", &intake.sampling_recipe_hash, &sampling_recipe)?;
    check_pinned("return_recipe_hash", &intake.return_recipe_hash, &return_recipe)?;
    check_pinned("shuffle_recipe_hash", &intake.shuffle_recipe_hash, &shuffle_recipe)?;

    // Source ledger (deterministic; binds tiles, seats, and digests).
    let ledger = generate_ledger(
        &intake.master_seed,
        &intake.experiment_id,
        intake.replicate_id,
        intake.attempt_id,
        intake.generations,
        intake.games_per_generation,
    )?;
    let computed_ledger = ledger_digest(&ledger)?;
    if intake.wall_ledger_hash != computed_ledger {
        return Err(lc_err(format!(
            "rl lifecycle wall_ledger_hash mismatch: config {} != generated {computed_ledger}",
            intake.wall_ledger_hash
        )));
    }
    let catalog_digest = canonical_digest(catalog_value, "rl:opponent-catalog")?;
    if intake.opponent_catalog_hash != catalog_digest {
        return Err(lc_err(format!(
            "rl lifecycle opponent_catalog_hash mismatch: config {} != generated {catalog_digest}",
            intake.opponent_catalog_hash
        )));
    }

    // Frozen optimizer/scheduler spec hashes: the exact owner doc shapes
    // from `stream_build._manifest_hashes_for_loop` through the shared JCS
    // printer, so Python `of_canonical` over the same docs agrees bytewise.
    let optimizer_spec_doc = serde_json::json!({
        "head_lr_mult": RlFrozenRecipe::HEAD_LR_MULT,
        "id": super::lifecycle::RL_OPTIMIZER_ID,
        "lr": RlFrozenRecipe::OPTIMIZER_LR,
        "betas": [RlFrozenRecipe::OPTIMIZER_BETAS.0, RlFrozenRecipe::OPTIMIZER_BETAS.1],
        "weight_decay": RlFrozenRecipe::OPTIMIZER_WEIGHT_DECAY,
    });
    let optimizer_spec_hash = canonical_digest(&optimizer_spec_doc, "rl:optimizer-spec")?;
    let scheduler_spec_doc = serde_json::json!({
        "id": super::lifecycle::RL_SCHEDULER_ID,
        "warmup_updates": RlFrozenRecipe::WARMUP_UPDATES,
        "parameters": {},
    });
    let scheduler_spec_hash = canonical_digest(&scheduler_spec_doc, "rl:scheduler-spec")?;

    // Row-derivation identity (law owned in lifecycle_intake; sealed into
    // every artifact header by ArtifactStorage from this descriptor value).
    // The derivation binds the ACTUAL source bytes of the twelve law owners
    // (Torch sampling/objective/learner + native wall/observation/event/
    // identity/core/pool files), read from the repo and hashed here — not
    // labels or revisions. Any edit to those files changes the derivation
    // and invalidates this descriptor (re-run prepare_run); C0 weight
    // identity is unaffected (weights bind content, never these hashes).
    let mut code_hashes = Map::with_capacity(super::lifecycle::LIFECYCLE_CODE_SOURCES.len());
    for (name, relpath) in super::lifecycle::LIFECYCLE_CODE_SOURCES {
        let bytes = super::lifecycle_common::read_repo_file(repo_dir, relpath)?;
        let digest = hydra_feed::digest::sha256_hex(&bytes);
        code_hashes.insert(name.to_string(), Value::String(digest));
    }
    let derivation_doc = row_derivation_doc(
        &intake.wall_recipe_hash,
        &intake.sampling_recipe_hash,
        &intake.return_recipe_hash,
        &intake.shuffle_recipe_hash,
        intake.clip_eps,
        intake.eps_std,
        intake.w_value,
        intake.w_bc,
        intake.ppo_alpha,
        &intake.core_revision,
        &adapter_version,
        &intake.initial_model_spec_hash,
        &intake.action_table_hash,
        &code_hashes,
    );
    let row_derivation_hash = canonical_digest(&derivation_doc, "rl:row-derivation")?;
    // Bilateral composition gate: ArtifactStorage's verifier cross-checks
    // the doc (closed 8 top keys, recipe shapes, objective ranges, numeric
    // literals, code/model shapes, digest match) before header assembly.
    // The verifier input reuses the EXACT canonical bytes the digest came
    // from (never transport-order JSON), so hash and gate agree bytewise.
    let derivation_bytes =
        hydra_feed::canon::canonical_bytes_value(&derivation_doc, "rl:row-derivation")
            .map_err(|e| lc_err(format!("rl lifecycle derivation doc not canonical-safe: {e}")))?;
    crate::rl::artifact::verify_row_derivation(&row_derivation_hash, &derivation_bytes)
        .map_err(|e| lc_err(format!("rl lifecycle row derivation rejected: {e}")))?;

    // Pipeline envelope (registered 32 fields + digest, schema 3.0.0: the
    // resource profile carries the four-field runtime identity, so execution
    // mode changes move the pipeline digest and every binding derived from it.
    // Fixed live fields training_source/recovery pin the native-live path.)
    let mut envelope = Map::with_capacity(33);
    envelope.insert("schema_version".to_string(), Value::String(RL_PIPELINE_SCHEMA_VERSION.to_string()));
    envelope.insert("initial_checkpoint_id".to_string(), Value::String(initial_id.clone()));
    envelope.insert(
        "initial_model_spec_hash".to_string(),
        Value::String(intake.initial_model_spec_hash.clone()),
    );
    envelope.insert("rules_hash".to_string(), Value::String(intake.rules_hash.clone()));
    envelope.insert("utility_hash".to_string(), Value::String(intake.utility_hash.clone()));
    envelope.insert(
        "action_table_hash".to_string(),
        Value::String(intake.action_table_hash.clone()),
    );
    envelope.insert(
        "event_schema_hash".to_string(),
        Value::String(intake.event_schema_hash.clone()),
    );
    envelope.insert(
        "observation_schema_hash".to_string(),
        Value::String(intake.observation_schema_hash.clone()),
    );
    envelope.insert(
        "packet_boundary_hash".to_string(),
        Value::String(intake.packet_boundary_hash.clone()),
    );
    envelope.insert(
        "input_schema_hash".to_string(),
        Value::String(intake.input_schema_hash.clone()),
    );
    envelope.insert(
        "master_seed_hex".to_string(),
        Value::String(intake.master_seed_hex.clone()),
    );
    envelope.insert(
        "experiment_id".to_string(),
        Value::String(intake.experiment_id.clone()),
    );
    envelope.insert("split_scope".to_string(), Value::String("selection_training".to_string()));
    envelope.insert("replicate_id".to_string(), Value::from(intake.replicate_id));
    envelope.insert("attempt_id".to_string(), Value::from(intake.attempt_id));
    envelope.insert("core_revision".to_string(), Value::String(intake.core_revision.clone()));
    envelope.insert(
        "wall_recipe_hash".to_string(),
        Value::String(intake.wall_recipe_hash.clone()),
    );
    envelope.insert(
        "sampling_recipe_hash".to_string(),
        Value::String(intake.sampling_recipe_hash.clone()),
    );
    envelope.insert(
        "return_recipe_hash".to_string(),
        Value::String(intake.return_recipe_hash.clone()),
    );
    envelope.insert(
        "shuffle_recipe_hash".to_string(),
        Value::String(intake.shuffle_recipe_hash.clone()),
    );
    envelope.insert(
        "wall_ledger_hash".to_string(),
        Value::String(intake.wall_ledger_hash.clone()),
    );
    envelope.insert(
        "opponent_catalog_hash".to_string(),
        Value::String(intake.opponent_catalog_hash.clone()),
    );
    envelope.insert("generations".to_string(), Value::from(intake.generations));
    envelope.insert(
        "games_per_generation".to_string(),
        Value::from(intake.games_per_generation),
    );
    let mut resource = Map::new();
    resource.insert("native_workers".to_string(), Value::from(intake.native_workers));
    resource.insert("env_slots".to_string(), Value::from(intake.env_slots));
    resource.insert("max_lookahead".to_string(), Value::from(1u64));
    resource.insert("inference_cap".to_string(), Value::from(intake.inference_cap));
    resource.insert("execution_profile_hash".to_string(), Value::String(runtime_spec_hash.clone()));
    envelope.insert("resource_profile".to_string(), Value::Object(resource));
    let mut learner = Map::new();
    learner.insert("microbatch_cap".to_string(), Value::from(intake.microbatch_cap));
    learner.insert("accumulation_cap".to_string(), Value::from(intake.accumulation_cap));
    learner.insert(
        "optimizer_minibatch_cap".to_string(),
        Value::from(intake.optimizer_minibatch_cap),
    );
    learner.insert("epochs".to_string(), Value::from(1u64));
    learner.insert(
        "max_updates_per_generation".to_string(),
        Value::from(intake.max_updates_per_generation),
    );
    envelope.insert("learner_recipe".to_string(), Value::Object(learner));
    let mut objective = Map::new();
    objective.insert("clip_eps".to_string(), Value::from(intake.clip_eps));
    objective.insert("eps_std".to_string(), Value::from(intake.eps_std));
    objective.insert("w_value".to_string(), Value::from(intake.w_value));
    objective.insert("w_bc".to_string(), Value::from(intake.w_bc));
    objective.insert("ppo_alpha".to_string(), Value::from(intake.ppo_alpha));
    envelope.insert("objective_recipe".to_string(), Value::Object(objective));
    let mut ret = Map::new();
    ret.insert("gamma".to_string(), Value::from(RlFrozenRecipe::GAMMA));
    ret.insert("lam".to_string(), Value::from(RlFrozenRecipe::LAM));
    ret.insert("bootstrap".to_string(), Value::String("none".to_string()));
    ret.insert("shaping".to_string(), Value::String("none".to_string()));
    envelope.insert("return_recipe".to_string(), Value::Object(ret));
    let mut optim = Map::new();
    optim.insert(
        "id".to_string(),
        Value::String(super::lifecycle::RL_OPTIMIZER_ID.to_string()),
    );
    optim.insert("lr".to_string(), Value::from(RlFrozenRecipe::OPTIMIZER_LR));
    optim.insert("beta1".to_string(), Value::from(RlFrozenRecipe::OPTIMIZER_BETAS.0));
    optim.insert("beta2".to_string(), Value::from(RlFrozenRecipe::OPTIMIZER_BETAS.1));
    optim.insert(
        "weight_decay".to_string(),
        Value::from(RlFrozenRecipe::OPTIMIZER_WEIGHT_DECAY),
    );
    optim.insert("head_lr_mult".to_string(), Value::from(RlFrozenRecipe::HEAD_LR_MULT));
    optim.insert(
        "scheduler".to_string(),
        Value::String(super::lifecycle::RL_SCHEDULER_ID.to_string()),
    );
    optim.insert("warmup_updates".to_string(), Value::from(RlFrozenRecipe::WARMUP_UPDATES));
    optim.insert(
        "gradient_clip_norm".to_string(),
        Value::from(RlFrozenRecipe::GRADIENT_CLIP_NORM),
    );
    envelope.insert("optimizer_recipe".to_string(), Value::Object(optim));
    let mut guards = Map::new();
    guards.insert("row_cap".to_string(), Value::from(RlFrozenRecipe::ROW_CAP));
    guards.insert(
        "decision_cap_per_game".to_string(),
        Value::from(RlFrozenRecipe::DECISION_CAP_PER_GAME),
    );
    guards.insert("compressed_byte_cap".to_string(), Value::from(8u64 * 1024 * 1024 * 1024));
    guards.insert(
        "min_free_disk_bytes".to_string(),
        Value::from(20u64 * 1024 * 1024 * 1024),
    );
    guards.insert("host_byte_budget".to_string(), Value::from(12u64 * 1024 * 1024 * 1024));
    guards.insert("gpu_cache_byte_budget".to_string(), Value::from(1024u64 * 1024 * 1024));
    envelope.insert("guards".to_string(), Value::Object(guards));
    envelope.insert("training_source".to_string(), Value::String("native_live_v1".to_string()));
    envelope.insert("recovery".to_string(), Value::String("generation_restart_v1".to_string()));
    // Closed-shape gate over the envelope before digesting (same
    // vocabulary the control/snapshot consumers validate against).
    {
        let keys: Vec<String> = envelope.keys().cloned().collect();
        let mut nested: Vec<(String, Vec<String>)> = Vec::new();
        for name in [
            "resource_profile",
            "learner_recipe",
            "objective_recipe",
            "return_recipe",
            "optimizer_recipe",
            "guards",
        ] {
            let children: Vec<String> = envelope
                .get(name)
                .and_then(Value::as_object)
                .map_or(Vec::new(), |obj| obj.keys().cloned().collect());
            nested.push((name.to_string(), children));
        }
        check_envelope_keys(RL_PIPELINE_ARTIFACT, &keys, &nested)
            .map_err(|e| lc_err(format!("rl lifecycle pipeline envelope rejected: {e}")))?;
    }
    let pipeline_spec_id = canonical_digest(&Value::Object(envelope.clone()), "rl:pipeline")?;
    envelope.insert("digest".to_string(), Value::String(pipeline_spec_id.clone()));

    // Per-generation source bindings. Profiles are NOT emitted here: every
    // emitted profile must carry a real behavior policy id, and only C0
    // exists at cold time. Main calls `collector_spec` per generation once
    // the behavior id is fixed (g0 with C0 immediately, later generations
    // with their published actor at launch).
    let mut generations = Vec::with_capacity(ledger.len());
    for (generation, cohort) in ledger.iter().enumerate() {
        let generation_u64 = generation as u64;
        let games: Vec<Value> = cohort
            .iter()
            .map(|game| {
                serde_json::json!({
                    "wall_id": game.wall_id,
                    "schedule_id": game.schedule_id,
                    "schedule_digest": game.schedule_digest,
                    "schedule_tiles": game.schedule_tiles.iter().map(|t| Value::from(*t as u64)).collect::<Vec<_>>(),
                    "seat_perm": game.seat_perm.iter().map(|s| Value::from(*s as u64)).collect::<Vec<_>>(),
                })
            })
            .collect();
        generations.push(serde_json::json!({
            "generation": generation_u64,
            "pool_id": format!("{}-g{generation_u64:02}", intake.experiment_id),
            "games": games,
            "shuffle_split": ledger_split_for(generation_u64),
        }));
    }


    let descriptor = serde_json::json!({
        "lifecycle_version": super::lifecycle::LIFECYCLE_VERSION,
        "run_id": run_id,
        "run_spec_hash": run_spec_hash,
        "pipeline_spec_id": pipeline_spec_id,
        "pipeline_envelope": envelope,
        "code_hashes": code_hashes,
        "initial_checkpoint_id": initial_id,
        "initial_model_spec_hash": intake.initial_model_spec_hash,
        "row_derivation_hash": row_derivation_hash,
        "recipe_digests": {
            "wall_recipe_hash": intake.wall_recipe_hash,
            "sampling_recipe_hash": intake.sampling_recipe_hash,
            "return_recipe_hash": intake.return_recipe_hash,
            "shuffle_recipe_hash": intake.shuffle_recipe_hash,
        },
        "spec_hashes": {
            "optimizer_spec_hash": optimizer_spec_hash,
            "scheduler_spec_hash": scheduler_spec_hash,
            "runtime_spec_hash": runtime_spec_hash,
            "seed_manifest_hash": seed_manifest_hash,
        },
        "environment_hash": environment_hash,
        "master_seed_hex": intake.master_seed_hex,
        "experiment_id": intake.experiment_id,
        "split_scope": "selection_training",
        "replicate_id": intake.replicate_id,
        "attempt_id": intake.attempt_id,
        "model": {
            "architecture_id": architecture_id,
            "action_count": action_count,
            "utility_id": utility_id,
        },
        "adapter_version": adapter_version,
        "rules": {
            "rules_id": assets.rules_id,
            "starting_points": assets.starting_points,
            "rank_values": assets.rank_values.to_vec(),
        },
        "identities": {
            "rules_hash": intake.rules_hash,
            "utility_hash": intake.utility_hash,
            "action_table_hash": intake.action_table_hash,
            "event_schema_hash": intake.event_schema_hash,
            "observation_schema_hash": intake.observation_schema_hash,
            "packet_boundary_hash": intake.packet_boundary_hash,
            "input_schema_hash": intake.input_schema_hash,
            "model_spec_hash": intake.initial_model_spec_hash,
        },
        "objective": {
            "id": super::lifecycle::RL_OBJECTIVE_ID,
            "clip_eps": intake.clip_eps,
            "eps_std": intake.eps_std,
            "w_value": intake.w_value,
            "w_bc": intake.w_bc,
            "alpha": intake.ppo_alpha,
        },
        "optimizer": {
            "id": super::lifecycle::RL_OPTIMIZER_ID,
            "lr": RlFrozenRecipe::OPTIMIZER_LR,
            "beta1": RlFrozenRecipe::OPTIMIZER_BETAS.0,
            "beta2": RlFrozenRecipe::OPTIMIZER_BETAS.1,
            "weight_decay": RlFrozenRecipe::OPTIMIZER_WEIGHT_DECAY,
            "head_lr_mult": RlFrozenRecipe::HEAD_LR_MULT,
            "scheduler": super::lifecycle::RL_SCHEDULER_ID,
            "warmup_updates": RlFrozenRecipe::WARMUP_UPDATES,
            "gradient_clip_norm": RlFrozenRecipe::GRADIENT_CLIP_NORM,
        },
        "runtime": resolved.get("runtime"),
        "seeds": resolved.get("seeds"),
        "selection": resolved.get("selection"),
        "precision": precision,
        "checkpoint_frequency_updates": checkpoint_frequency_updates,
        "eval_walls_dir": eval_walls_dir,
        "cohort": {
            "generations": intake.generations,
            "games_per_generation": intake.games_per_generation,
            "epochs": 1,
            "row_cap": RlFrozenRecipe::ROW_CAP,
            "microbatch_cap": intake.microbatch_cap,
            "accumulation_cap": intake.accumulation_cap,
            "optimizer_minibatch_cap": intake.optimizer_minibatch_cap,
            "max_updates_per_generation": intake.max_updates_per_generation,
            "native_workers": intake.native_workers,
            "env_slots": intake.env_slots,
            "inference_cap": intake.inference_cap,
        },
        "generations": generations,
        "wall_ledger_hash": intake.wall_ledger_hash,
        "opponent_catalog_hash": intake.opponent_catalog_hash,
        "opponent_catalog": catalog_value,
        "locators": {
            "root": root_dir,
            "descriptor": format!("{root_dir}/run_descriptor.json"),
            "initial_checkpoint": format!("{root_dir}/checkpoints/c0.pt"),
            "checkpoints_dir": format!("{root_dir}/checkpoints"),
            "artifacts_dir": format!("{root_dir}/artifacts"),
            "control_dir": format!("{root_dir}/control"),
            "snapshots_dir": format!("{root_dir}/snapshots"),
        },
        "shuffle_key_template": {
            "purpose": "training_shuffle",
            "candidate_id": pipeline_spec_id,
            "experiment_id": intake.experiment_id,
            "split_format": "selection_training:g={g}:epoch={epoch}",
            "replicate_id": intake.replicate_id,
            "attempt_id": intake.attempt_id,
        },
        "sample_key_template": {
            "purpose": "actor_policy_sample",
            "scope": "selection_training",
            "split_format": "selection_training:g={g}",
            "candidate_role": "behavior-policy-id (pool policy_id)",
            "case_role": "derived game_id",
            "root_seat_role": "seat 0..3",
            "replicate_id": intake.replicate_id,
            "attempt_id": intake.attempt_id,
        },
        "asset_report": {
            "rules_bytes_sha": assets.rules_bytes_sha,
            "action_table_digest": assets.action_table_digest,
            "event_schema_digest": assets.event_schema_digest,
            "observation_schema_digest": assets.observation_schema_digest,
            "packet_boundary_digest": assets.packet_boundary_digest,
            "input_schema_digest": assets.input_schema_digest,
        },
    });
    hydra_feed::canon::canonical_bytes_value(&descriptor, "rl:run-descriptor")
        .map_err(|e| lc_err(format!("rl lifecycle descriptor not canonical-safe: {e}")))
}

/// Cold-init Torch seed: deterministic construction RNG derived BEFORE any
/// shell exists, so captured `rng_state` (hence ordinary manifest ids) is
/// identical across fresh processes given identical inputs.
///
/// Key law: registered `training_shuffle` purpose (no required fields, so
/// a content-bound init key validates; `candidate_id` is its allowed
/// optional), `selection_training:cold_init` split (never collides with
/// `selection_training:g={g}:epoch={epoch}` shuffle splits or the
/// `wall`/`actor_policy_sample` keys), `candidate_id` = initial C0 id
/// (binds the seed to the exact weights under construction), explicit
/// replicate/attempt, every other field null. Single seed owner is
/// `hydra_search::eval`. Torch applies the first 8 bytes BE as its u64
/// (`manual_seed`/`manual_seed_all`) via the thin Python wrapper.
pub fn cold_init_seed(
    master_seed_hex: &str,
    experiment_id: &str,
    replicate_id: u64,
    attempt_id: u64,
    initial_checkpoint_id: &str,
) -> Result<[u8; 32], LifecycleError> {
    if master_seed_hex.len() != 64
        || !master_seed_hex.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err(lc_err(
            "rl lifecycle cold seed master_seed_hex must be 64 bare hex chars",
        ));
    }
    if !super::lifecycle_intake::valid_experiment_id(experiment_id) {
        return Err(lc_err(
            "rl lifecycle cold seed experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]*",
        ));
    }
    if !is_digest_shape(initial_checkpoint_id) {
        return Err(lc_err(
            "rl lifecycle cold seed initial_checkpoint_id must be sha256:<64 hex>",
        ));
    }
    let mut master = [0u8; 32];
    for (i, byte) in master.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&master_seed_hex[2 * i..2 * i + 2], 16)
            .map_err(|_| lc_err("rl lifecycle cold seed master has non-hex digits"))?;
    }
    let mut key = Map::with_capacity(17);
    key.insert("purpose".to_string(), Value::String("training_shuffle".to_string()));
    key.insert("experiment_id".to_string(), Value::String(experiment_id.to_string()));
    key.insert(
        "split_id".to_string(),
        Value::String("selection_training:cold_init".to_string()),
    );
    key.insert(
        "candidate_id".to_string(),
        Value::String(initial_checkpoint_id.to_string()),
    );
    key.insert("case_id".to_string(), Value::Null);
    key.insert("wall_id".to_string(), Value::Null);
    for null_field in [
        "root_seat",
        "belief_epoch",
        "parent_id",
        "action_id",
        "packet_id",
        "fidelity_level",
        "population_id",
    ] {
        key.insert(null_field.to_string(), Value::Null);
    }
    key.insert("replicate_id".to_string(), Value::from(replicate_id));
    key.insert("scramble_id".to_string(), Value::Null);
    key.insert("visit_index".to_string(), Value::Null);
    key.insert("attempt_id".to_string(), Value::from(attempt_id));
    hydra_search::eval::semantic_seed(&master, &Value::Object(key))
        .map_err(|e| lc_err(format!("rl lifecycle cold seed rejected: {e}")))
}
