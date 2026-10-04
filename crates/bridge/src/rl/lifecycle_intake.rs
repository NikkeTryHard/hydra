//! rl/lifecycle_intake: closed RL config intake, recipe laws, asset gates.
//!
//! DAG: `super::lifecycle_common` + `serde_json` + `hydra_feed::{canon,
//! digest}` + `hydra_shard::parity` (rules/action pins) + `super::{config,
//! artifact}` (schema vocabulary, manifest keys). FORBIDS: ledger math
//! (lives in `lifecycle_ledger`), PyO3, wall-clock.
//!
//! Owns the exact `RlConfig` field vocabulary (53 fields), the frozen-value
//! intake with the capacity/count invariant, the four recipe documents plus
//! the row-derivation composition (see the composition pointer below), the
//! repo-asset verification against live authorities, and the IL manifest
//! gate.

use serde_json::{Map, Value};

use super::lifecycle_common::{
    canonical_digest, check_closed_object, is_digest_shape, lc_err, read_repo_file, req_digest,
    req_exact_f64, req_str, req_u64, LifecycleError,
};
use crate::rl::artifact::CHECKPOINT_MANIFEST_KEYS;
use crate::rl::config::{RL_SCHEMA_RECORDS, RlFrozenRecipe};
use crate::rl::wall::WALL_STREAM_NAME;
/// 53 fields — the count is load-bearing: the intake rejects anything else).
pub const RL_CONFIG_FIELDS: [&str; 53] = [
    "schema_version",
    "initial_checkpoint_id",
    "initial_model_spec_hash",
    "master_seed_hex",
    "experiment_id",
    "split_scope",
    "replicate_id",
    "attempt_id",
    "generations",
    "games_per_generation",
    "native_workers",
    "env_slots",
    "max_lookahead",
    "inference_cap",
    "microbatch_cap",
    "accumulation_cap",
    "optimizer_minibatch_cap",
    "epochs",
    "row_cap",
    "decision_cap_per_game",
    "max_updates_per_generation",
    "compressed_byte_cap",
    "min_free_disk_bytes",
    "host_byte_budget",
    "gpu_cache_byte_budget",
    "clip_eps",
    "eps_std",
    "w_value",
    "w_bc",
    "ppo_alpha",
    "gamma",
    "lam",
    "optimizer_lr",
    "optimizer_beta1",
    "optimizer_beta2",
    "optimizer_weight_decay",
    "head_lr_mult",
    "gradient_clip_norm",
    "warmup_updates",
    "rules_hash",
    "utility_hash",
    "action_table_hash",
    "event_schema_hash",
    "observation_schema_hash",
    "packet_boundary_hash",
    "input_schema_hash",
    "wall_ledger_hash",
    "opponent_catalog_hash",
    "wall_recipe_hash",
    "sampling_recipe_hash",
    "return_recipe_hash",
    "shuffle_recipe_hash",
    "core_revision",
];

/// Validated closed RL profile (all composition hashes required pinned).
pub(crate) struct RlIntake {
    pub initial_checkpoint_id: String,
    pub initial_model_spec_hash: String,
    pub master_seed_hex: String,
    pub master_seed: [u8; 32],
    pub experiment_id: String,
    pub replicate_id: u64,
    pub attempt_id: u64,
    pub generations: u64,
    pub games_per_generation: u64,
    pub native_workers: u64,
    pub env_slots: u64,
    pub inference_cap: u64,
    pub microbatch_cap: u64,
    pub accumulation_cap: u64,
    pub optimizer_minibatch_cap: u64,
    pub max_updates_per_generation: u64,
    pub clip_eps: f64,
    pub eps_std: f64,
    pub w_value: f64,
    pub w_bc: f64,
    pub ppo_alpha: f64,
    pub rules_hash: String,
    pub utility_hash: String,
    pub action_table_hash: String,
    pub event_schema_hash: String,
    pub observation_schema_hash: String,
    pub packet_boundary_hash: String,
    pub input_schema_hash: String,
    pub wall_ledger_hash: String,
    pub opponent_catalog_hash: String,
    pub wall_recipe_hash: String,
    pub sampling_recipe_hash: String,
    pub return_recipe_hash: String,
    pub shuffle_recipe_hash: String,
    pub core_revision: String,
}

pub(crate) fn valid_experiment_id(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

pub(crate) fn parse_rl_intake(rl: &Map<String, Value>) -> Result<RlIntake, LifecycleError> {
    check_closed_object(rl, &RL_CONFIG_FIELDS, "rl section")?;
    let get = |name: &str| -> Result<&Value, LifecycleError> {
        rl.get(name)
            .ok_or_else(|| lc_err(format!("rl lifecycle rl section lacks {name:?}")))
    };
    match get("schema_version")? {
        Value::String(v) if v == "1.0.0" => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle rl.schema_version must be '1.0.0', got {other:?}"
            )))
        }
    }
    match get("split_scope")? {
        Value::String(v) if v == "selection_training" => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle rl.split_scope must be 'selection_training', got {other:?}"
            )))
        }
    }
    let initial_checkpoint_id =
        req_digest(Some(get("initial_checkpoint_id")?), "rl.initial_checkpoint_id")?;
    let initial_model_spec_hash =
        req_digest(Some(get("initial_model_spec_hash")?), "rl.initial_model_spec_hash")?;
    let master_raw = req_str(Some(get("master_seed_hex")?), "rl.master_seed_hex")?;
    if master_raw.len() != 64 || !master_raw.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(lc_err(
            "rl lifecycle rl.master_seed_hex must be 64 bare hex chars",
        ));
    }
    let mut master_seed = [0u8; 32];
    for (i, byte) in master_seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&master_raw[2 * i..2 * i + 2], 16)
            .map_err(|_| lc_err("rl lifecycle rl.master_seed_hex has non-hex digits"))?;
    }
    let experiment_id = req_str(Some(get("experiment_id")?), "rl.experiment_id")?;
    if !valid_experiment_id(&experiment_id) {
        return Err(lc_err(format!(
            "rl lifecycle rl.experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {experiment_id:?}"
        )));
    }
    let replicate_id = req_u64(Some(get("replicate_id")?), "rl.replicate_id")?;
    let attempt_id = req_u64(Some(get("attempt_id")?), "rl.attempt_id")?;
    // Registered pipeline schema pins the full-pilot cohort (8x128): the
    // envelope must conform with no migration. Small fixtures live at the
    // pool level (pool cohorts accept any positive size); they never enter
    // a pipeline envelope.
    let generations = req_u64(Some(get("generations")?), "rl.generations")?;
    if generations != RlFrozenRecipe::GENERATIONS as u64 {
        return Err(lc_err(format!(
            "rl lifecycle rl.generations must be exactly {} (registered schema; no migration)",
            RlFrozenRecipe::GENERATIONS
        )));
    }
    let games_per_generation =
        req_u64(Some(get("games_per_generation")?), "rl.games_per_generation")?;
    if games_per_generation != RlFrozenRecipe::GAMES_PER_GENERATION as u64 {
        return Err(lc_err(format!(
            "rl lifecycle rl.games_per_generation must be exactly {} (registered schema; no migration)",
            RlFrozenRecipe::GAMES_PER_GENERATION
        )));
    }
    let native_workers = req_u64(Some(get("native_workers")?), "rl.native_workers")?;
    if native_workers == 0 || native_workers > 20 {
        return Err(lc_err(format!(
            "rl lifecycle rl.native_workers {native_workers} outside pipeline 1..=20"
        )));
    }
    let env_slots = req_u64(Some(get("env_slots")?), "rl.env_slots")?;
    if env_slots == 0 || env_slots > 256 {
        return Err(lc_err(format!(
            "rl lifecycle rl.env_slots {env_slots} outside pipeline 1..=256"
        )));
    }
    let inference_cap = req_u64(Some(get("inference_cap")?), "rl.inference_cap")?;
    if inference_cap == 0 || inference_cap > 128 {
        return Err(lc_err(format!(
            "rl lifecycle rl.inference_cap {inference_cap} outside 1..=128"
        )));
    }
    let microbatch_cap = req_u64(Some(get("microbatch_cap")?), "rl.microbatch_cap")?;
    if microbatch_cap == 0 || microbatch_cap > 128 {
        return Err(lc_err(format!(
            "rl lifecycle rl.microbatch_cap {microbatch_cap} outside 1..=128"
        )));
    }
    let accumulation_cap = req_u64(Some(get("accumulation_cap")?), "rl.accumulation_cap")?;
    if accumulation_cap == 0 || accumulation_cap > 32 {
        return Err(lc_err(format!(
            "rl lifecycle rl.accumulation_cap {accumulation_cap} outside 1..=32"
        )));
    }
    let optimizer_minibatch_cap =
        req_u64(Some(get("optimizer_minibatch_cap")?), "rl.optimizer_minibatch_cap")?;
    if optimizer_minibatch_cap == 0 || optimizer_minibatch_cap > 4096 {
        return Err(lc_err(format!(
            "rl lifecycle rl.optimizer_minibatch_cap {optimizer_minibatch_cap} outside 1..=4096"
        )));
    }
    // Capacity/count invariant (same equality as supervised
    // `optimizer_minibatch_size = microbatch_size * accumulation_steps`):
    // the cap product must equal the minibatch cap exactly, so short tails
    // are genuine remainder rows, never dropped or duplicated fill.
    let product = microbatch_cap
        .checked_mul(accumulation_cap)
        .ok_or_else(|| lc_err("rl lifecycle microbatch*accumulation overflows"))?;
    if product != optimizer_minibatch_cap {
        return Err(lc_err(format!(
            "rl lifecycle capacity invariant broken: microbatch_cap {microbatch_cap} * accumulation_cap {accumulation_cap} = {product} != optimizer_minibatch_cap {optimizer_minibatch_cap}"
        )));
    }
    let max_updates_per_generation = req_u64(
        Some(get("max_updates_per_generation")?),
        "rl.max_updates_per_generation",
    )?;
    if max_updates_per_generation == 0
        || max_updates_per_generation > RlFrozenRecipe::MAX_UPDATES_PER_GENERATION as u64
    {
        return Err(lc_err(format!(
            "rl lifecycle rl.max_updates_per_generation {max_updates_per_generation} outside 1..=67"
        )));
    }
    let req_guard = |name: &str, want: u64| -> Result<(), LifecycleError> {
        match get(name)? {
            Value::Number(n) if !n.is_f64() && n.as_u64() == Some(want) => Ok(()),
            other => Err(lc_err(format!(
                "rl lifecycle frozen rl.{name} must be exactly {want}, got {other:?}"
            ))),
        }
    };
    req_guard("epochs", 1)?;
    req_guard("row_cap", RlFrozenRecipe::ROW_CAP as u64)?;
    req_guard(
        "decision_cap_per_game",
        RlFrozenRecipe::DECISION_CAP_PER_GAME as u64,
    )?;
    req_guard("compressed_byte_cap", 8 * 1024 * 1024 * 1024)?;
    req_guard("min_free_disk_bytes", 20 * 1024 * 1024 * 1024)?;
    req_guard("host_byte_budget", 12 * 1024 * 1024 * 1024)?;
    req_guard("gpu_cache_byte_budget", 1024 * 1024 * 1024)?;
    req_guard("max_lookahead", 1)?;
    req_guard("warmup_updates", RlFrozenRecipe::WARMUP_UPDATES as u64)?;
    let clip_eps =
        req_exact_f64(Some(get("clip_eps")?), "rl.clip_eps", RlFrozenRecipe::CLIP_EPS)?;
    let eps_std = req_exact_f64(Some(get("eps_std")?), "rl.eps_std", RlFrozenRecipe::EPS_STD)?;
    let w_value = req_exact_f64(Some(get("w_value")?), "rl.w_value", RlFrozenRecipe::W_VALUE)?;
    let w_bc = req_exact_f64(Some(get("w_bc")?), "rl.w_bc", RlFrozenRecipe::W_BC)?;
    let ppo_alpha = req_exact_f64(
        Some(get("ppo_alpha")?),
        "rl.ppo_alpha",
        RlFrozenRecipe::PPO_ALPHA,
    )?;
    req_exact_f64(Some(get("gamma")?), "rl.gamma", RlFrozenRecipe::GAMMA)?;
    req_exact_f64(Some(get("lam")?), "rl.lam", RlFrozenRecipe::LAM)?;
    req_exact_f64(
        Some(get("optimizer_lr")?),
        "rl.optimizer_lr",
        RlFrozenRecipe::OPTIMIZER_LR,
    )?;
    req_exact_f64(
        Some(get("optimizer_beta1")?),
        "rl.optimizer_beta1",
        RlFrozenRecipe::OPTIMIZER_BETAS.0,
    )?;
    req_exact_f64(
        Some(get("optimizer_beta2")?),
        "rl.optimizer_beta2",
        RlFrozenRecipe::OPTIMIZER_BETAS.1,
    )?;
    req_exact_f64(
        Some(get("optimizer_weight_decay")?),
        "rl.optimizer_weight_decay",
        RlFrozenRecipe::OPTIMIZER_WEIGHT_DECAY,
    )?;
    req_exact_f64(
        Some(get("head_lr_mult")?),
        "rl.head_lr_mult",
        RlFrozenRecipe::HEAD_LR_MULT,
    )?;
    req_exact_f64(
        Some(get("gradient_clip_norm")?),
        "rl.gradient_clip_norm",
        RlFrozenRecipe::GRADIENT_CLIP_NORM,
    )?;
    let rules_hash = req_digest(Some(get("rules_hash")?), "rl.rules_hash")?;
    let utility_hash = req_digest(Some(get("utility_hash")?), "rl.utility_hash")?;
    let action_table_hash =
        req_digest(Some(get("action_table_hash")?), "rl.action_table_hash")?;
    let event_schema_hash =
        req_digest(Some(get("event_schema_hash")?), "rl.event_schema_hash")?;
    let observation_schema_hash =
        req_digest(Some(get("observation_schema_hash")?), "rl.observation_schema_hash")?;
    let packet_boundary_hash =
        req_digest(Some(get("packet_boundary_hash")?), "rl.packet_boundary_hash")?;
    let input_schema_hash =
        req_digest(Some(get("input_schema_hash")?), "rl.input_schema_hash")?;
    // All six composition hashes are required pinned (no empty bootstrap):
    // recipes come from `recipe_digests`, the ledger/catalog from the
    // deterministic builders. A mismatch names the computed value so Main
    // can pin it, but the run never proceeds unpinned.
    let wall_ledger_hash = req_digest(Some(get("wall_ledger_hash")?), "rl.wall_ledger_hash")?;
    let opponent_catalog_hash =
        req_digest(Some(get("opponent_catalog_hash")?), "rl.opponent_catalog_hash")?;
    let wall_recipe_hash = req_digest(Some(get("wall_recipe_hash")?), "rl.wall_recipe_hash")?;
    let sampling_recipe_hash =
        req_digest(Some(get("sampling_recipe_hash")?), "rl.sampling_recipe_hash")?;
    let return_recipe_hash =
        req_digest(Some(get("return_recipe_hash")?), "rl.return_recipe_hash")?;
    let shuffle_recipe_hash =
        req_digest(Some(get("shuffle_recipe_hash")?), "rl.shuffle_recipe_hash")?;
    let core_revision = match get("core_revision")? {
        Value::String(text) => text.clone(),
        _ => return Err(lc_err("rl lifecycle rl.core_revision must be a string")),
    };
    if core_revision != super::lifecycle::CORE_REVISION {
        return Err(lc_err(format!(
            "rl lifecycle rl.core_revision must be pinned {}, got {core_revision:?}",
            super::lifecycle::CORE_REVISION
        )));
    }
    Ok(RlIntake {
        initial_checkpoint_id,
        initial_model_spec_hash,
        master_seed_hex: master_raw,
        master_seed,
        experiment_id,
        replicate_id,
        attempt_id,
        generations,
        games_per_generation,
        native_workers,
        env_slots,
        inference_cap,
        microbatch_cap,
        accumulation_cap,
        optimizer_minibatch_cap,
        max_updates_per_generation,
        clip_eps,
        eps_std,
        w_value,
        w_bc,
        ppo_alpha,
        rules_hash,
        utility_hash,
        action_table_hash,
        event_schema_hash,
        observation_schema_hash,
        packet_boundary_hash,
        input_schema_hash,
        wall_ledger_hash,
        opponent_catalog_hash,
        wall_recipe_hash,
        sampling_recipe_hash,
        return_recipe_hash,
        shuffle_recipe_hash,
        core_revision,
    })
}

// ---------------------------------------------------------------------------
// Recipe documents (laws owned HERE).
//
// Composition pointer for ArtifactStorage: the seal binds
// `row_derivation_hash` by equality across every row; its preimage is
// exactly the 8-key `row_derivation_doc` below — wall/sampling/return/
// shuffle recipe digests (each the JCS sha of its documented recipe doc),
// the frozen objective numerics, the numeric dtype contract, code pins
// (core_revision + adapter_version) plus the live source-byte hashes of
// the four plan-6 law owners under code.code_hashes, and the
// model/action-table identities. `artifact::verify_row_derivation`
// cross-checks this doc before header assembly.
// ---------------------------------------------------------------------------

fn wall_recipe_doc() -> Value {
    serde_json::json!({
        "stream": WALL_STREAM_NAME,
        "numpy": "2.5.3",
        "seed": "sha256-first8-BE",
        "sequencer": "seedsequence-pool4",
        "prng": "pcg64-xsl-rr-128/64",
        "interval": "masked-inclusive-bound",
        "permutation": "descending-fisher-yates-136",
    })
}

fn sampling_recipe_doc() -> Value {
    serde_json::json!({
        "purpose": "actor_policy_sample",
        "scope": "selection_training",
        "uniform": "u53-8B-BE",
        "logp": "fp32-masked-log-softmax",
        "selection": "strict-upper-bound-fp64-normalized",
    })
}

fn return_recipe_doc() -> Value {
    serde_json::json!({
        "law": "terminal-mc",
        "gamma": 1.0,
        "lam": 1.0,
        "bootstrap": false,
        "shaping": "none",
        "advantage": "G32-minus-V32-fp32-frozen",
    })
}

fn shuffle_recipe_doc() -> Value {
    serde_json::json!({
        "purpose": "training_shuffle",
        "order": "descending-fisher-yates-random-below",
        "grouping": "stable-T",
        "cap": 4096,
        "epochs": 1,
    })
}

/// Frozen composition digests with no inputs (pinning tool: Main records
/// these into the RL config before the enforcing `prepare_run` call).
pub fn recipe_digests() -> Result<Value, LifecycleError> {
    Ok(serde_json::json!({
        "wall_recipe_hash": canonical_digest(&wall_recipe_doc(), "rl:wall-recipe")?,
        "sampling_recipe_hash": canonical_digest(&sampling_recipe_doc(), "rl:sampling-recipe")?,
        "return_recipe_hash": canonical_digest(&return_recipe_doc(), "rl:return-recipe")?,
        "shuffle_recipe_hash": canonical_digest(&shuffle_recipe_doc(), "rl:shuffle-recipe")?,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn row_derivation_doc(
    wall_recipe: &str,
    sampling_recipe: &str,
    return_recipe: &str,
    shuffle_recipe: &str,
    clip_eps: f64,
    eps_std: f64,
    w_value: f64,
    w_bc: f64,
    alpha: f64,
    core_revision: &str,
    adapter_version: &str,
    model_spec: &str,
    action_table: &str,
    code_hashes: &Map<String, Value>,
) -> Value {
    serde_json::json!({
        "wall_recipe": wall_recipe,
        "sampling_recipe": sampling_recipe,
        "return_recipe": return_recipe,
        "shuffle_recipe": shuffle_recipe,
        "objective": {
            "clip_eps": clip_eps,
            "eps_std": eps_std,
            "w_value": w_value,
            "w_bc": w_bc,
            "alpha": alpha,
        },
        "numeric": {
            "logp": "fp32",
            "advantage": "fp32-frozen",
            "return_wire": "f64",
            "entropy": "legal-only",
        },
        "code": {
            "core_revision": core_revision,
            "adapter_version": adapter_version,
            "code_hashes": code_hashes,
        },
        "model": {
            "model_spec": model_spec,
            "action_table": action_table,
        },
    })
}

// ---------------------------------------------------------------------------
// Asset verification (live authorities; content laws reused, never reprinted).
//
// Rules identity: the LIVE authority is `adapter_identity._rules_identity`
// (D-WP03A-4): sha256 over the PUBLISHED rules-file bytes wins when the
// file exists (`adapter_identity.py:145-157`), with the envelope-canonical
// recompute (`state.rules_identity_hash`) as fallback. The shard pin
// (`parity.PINNED_RULES_HASH`) records the same bytes. Native therefore
// hashes the repo file bytes directly — the identical law, not a guess —
// and cross-anchors against the shard pin so silent file drift fails here.
// ---------------------------------------------------------------------------

pub(crate) struct AssetReport {
    pub rules_bytes_sha: String,
    pub action_table_digest: String,
    pub event_schema_digest: String,
    pub observation_schema_digest: String,
    pub packet_boundary_digest: String,
    pub input_schema_digest: String,
    pub rules_id: String,
    pub starting_points: i64,
    pub rank_values: [f64; 4],
}

fn verify_schema_file(
    repo_dir: &str,
    relpath: &str,
    artifact_type: &str,
    expected: &str,
    name: &str,
) -> Result<String, LifecycleError> {
    let bytes = read_repo_file(repo_dir, relpath)?;
    let doc: Value = serde_json::from_slice(&bytes)
        .map_err(|e| lc_err(format!("rl lifecycle {name} asset not JSON: {e}")))?;
    let map = doc
        .as_object()
        .ok_or_else(|| lc_err(format!("rl lifecycle {name} asset must be an object")))?;
    let artifact = map.get("artifact_type").and_then(Value::as_str).unwrap_or("");
    if artifact != artifact_type {
        return Err(lc_err(format!(
            "rl lifecycle {name} artifact_type must be {artifact_type:?}, got {artifact:?}"
        )));
    }
    let schema = map.get("schema_version").and_then(Value::as_str).unwrap_or("");
    if schema != "1.0.0" {
        return Err(lc_err(format!(
            "rl lifecycle {name} schema_version must be '1.0.0', got {schema:?}"
        )));
    }
    let payload = map
        .get("payload")
        .cloned()
        .ok_or_else(|| lc_err(format!("rl lifecycle {name} asset lacks payload")))?;
    // Self-integrity: recompute the recorded payload digest with the shared
    // exclusion law (same convention as the schema owners).
    let (recomputed, _) = crate::rl::config::identity_of_payload_without_digest(&payload)
        .map_err(|e| lc_err(format!("rl lifecycle {name} self-digest rejected: {e}")))?;
    // Config<->asset match: the run config must name exactly this digest.
    if recomputed != expected {
        return Err(lc_err(format!(
            "rl lifecycle {name} mismatch: config {expected} != asset {recomputed}"
        )));
    }
    Ok(recomputed)
}

pub(crate) fn verify_assets(
    repo_dir: &str,
    intake: &RlIntake,
    utility_rank_values: &[f64; 4],
) -> Result<AssetReport, LifecycleError> {
    let rules_bytes = read_repo_file(repo_dir, super::lifecycle::RULES_RELPATH)?;
    let rules_bytes_sha = hydra_feed::digest::sha256_hex(&rules_bytes);
    if rules_bytes_sha != intake.rules_hash {
        return Err(lc_err(format!(
            "rl lifecycle rules mismatch: config {} != file bytes {}",
            intake.rules_hash, rules_bytes_sha
        )));
    }
    if rules_bytes_sha != hydra_shard::parity::PINNED_RULES_HASH {
        return Err(lc_err(format!(
            "rl lifecycle rules file drift: bytes {rules_bytes_sha} != shard pin {}",
            hydra_shard::parity::PINNED_RULES_HASH
        )));
    }
    let rules_doc: Value = serde_json::from_slice(&rules_bytes)
        .map_err(|e| lc_err(format!("rl lifecycle rules asset not JSON: {e}")))?;
    let payload = rules_doc
        .get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err("rl lifecycle rules asset lacks payload"))?;
    let rules_id = payload
        .get("rules_id")
        .and_then(Value::as_str)
        .ok_or_else(|| lc_err("rl lifecycle rules asset lacks rules_id"))?;
    if rules_id != hydra_shard::parity::RULES_ID {
        return Err(lc_err(format!(
            "rl lifecycle rules_id must be {:?}, got {rules_id:?}",
            hydra_shard::parity::RULES_ID
        )));
    }
    let starting_points = payload
        .get("starting_points")
        .and_then(Value::as_i64)
        .ok_or_else(|| lc_err("rl lifecycle rules asset lacks starting_points"))?;
    let uma = payload
        .get("uma_by_rank")
        .and_then(Value::as_array)
        .ok_or_else(|| lc_err("rl lifecycle rules asset lacks uma_by_rank"))?;
    if uma.len() != 4 {
        return Err(lc_err("rl lifecycle rules uma_by_rank must hold four ranks"));
    }
    let mut rank_values = [0.0f64; 4];
    for (i, entry) in uma.iter().enumerate() {
        let v = entry
            .as_f64()
            .ok_or_else(|| lc_err(format!("rl lifecycle rules uma_by_rank[{i}] must be a number")))?;
        if !v.is_finite() {
            return Err(lc_err("rl lifecycle rules uma must be finite"));
        }
        rank_values[i] = v;
    }
    // Config utility values must equal the rules file's uma (two sources,
    // one truth: the config cannot silently reprice placements).
    if rank_values != *utility_rank_values {
        return Err(lc_err(format!(
            "rl lifecycle utility rank_values {utility_rank_values:?} != rules uma {rank_values:?}"
        )));
    }
    let action_table_digest = verify_schema_file(
        repo_dir,
        super::lifecycle::ACTION_TABLE_RELPATH,
        "hydra2.action_table",
        &intake.action_table_hash,
        "action_table",
    )?;
    if action_table_digest != hydra_shard::parity::PINNED_ACTION_TABLE_HASH {
        return Err(lc_err(format!(
            "rl lifecycle action table drift: asset {action_table_digest} != shard pin {}",
            hydra_shard::parity::PINNED_ACTION_TABLE_HASH
        )));
    }
    let event_schema_digest = verify_schema_file(
        repo_dir,
        super::lifecycle::EVENT_SCHEMA_RELPATH,
        "hydra2.event_schema",
        &intake.event_schema_hash,
        "event_schema",
    )?;
    let observation_schema_digest = verify_schema_file(
        repo_dir,
        super::lifecycle::OBSERVATION_SCHEMA_RELPATH,
        "hydra2.observation_schema",
        &intake.observation_schema_hash,
        "observation_schema",
    )?;
    let packet_boundary_digest = verify_schema_file(
        repo_dir,
        super::lifecycle::PACKET_BOUNDARY_RELPATH,
        "hydra2.packet_boundary",
        &intake.packet_boundary_hash,
        "packet_boundary",
    )?;
    let input_schema_digest = verify_schema_file(
        repo_dir,
        super::lifecycle::MODEL_INPUT_RELPATH,
        "hydra2.model_input_schema",
        &intake.input_schema_hash,
        "model_input",
    )?;
    // Core pin: the repo dependency must carry the frozen revision.
    let cargo = read_repo_file(repo_dir, super::lifecycle::CARGO_RELPATH)?;
    let cargo_text =
        String::from_utf8(cargo).map_err(|_| lc_err("rl lifecycle Cargo.toml is not UTF-8"))?;
    if !cargo_text.contains(super::lifecycle::CORE_REVISION) {
        return Err(lc_err(format!(
            "rl lifecycle repo {} lacks pinned core rev {}",
            super::lifecycle::CARGO_RELPATH,
            super::lifecycle::CORE_REVISION
        )));
    }
    // Schema registry self-consistency: the registered payload digests must
    // match the files they were recorded from.
    for record in RL_SCHEMA_RECORDS {
        let bytes = read_repo_file(repo_dir, record.path)?;
        let doc: Value = serde_json::from_slice(&bytes)
            .map_err(|e| lc_err(format!("rl lifecycle schema asset {} not JSON: {e}", record.path)))?;
        let payload = doc
            .get("payload")
            .cloned()
            .ok_or_else(|| lc_err(format!("rl lifecycle schema asset {} lacks payload", record.path)))?;
        let (recomputed, _) =
            crate::rl::config::identity_of_payload_without_digest(&payload).map_err(|e| {
                lc_err(format!(
                    "rl lifecycle schema asset {} self-digest rejected: {e}",
                    record.path
                ))
            })?;
        if recomputed != record.digest {
            return Err(lc_err(format!(
                "rl lifecycle schema registry drift at {}: registered {} != file {}",
                record.path, record.digest, recomputed
            )));
        }
    }
    Ok(AssetReport {
        rules_bytes_sha,
        action_table_digest,
        event_schema_digest,
        observation_schema_digest,
        packet_boundary_digest,
        input_schema_digest,
        rules_id: rules_id.to_string(),
        starting_points,
        rank_values,
    })
}

// ---------------------------------------------------------------------------
// Manifest gates (19 fields; IL lineage for the cold start).
// ---------------------------------------------------------------------------

pub(crate) fn manifest_shape_gate(
    manifest_doc: &[u8],
    what: &str,
) -> Result<Map<String, Value>, LifecycleError> {
    let manifest: Value = serde_json::from_slice(manifest_doc)
        .map_err(|e| lc_err(format!("rl lifecycle {what} manifest not JSON: {e}")))?;
    let map = manifest
        .as_object()
        .ok_or_else(|| lc_err(format!("rl lifecycle {what} manifest must be an object")))?;
    if map.len() != CHECKPOINT_MANIFEST_KEYS.len()
        || !CHECKPOINT_MANIFEST_KEYS.iter().all(|k| map.contains_key(*k))
    {
        return Err(lc_err(format!(
            "rl lifecycle {what} manifest must carry exactly the 19 CheckpointManifest fields"
        )));
    }
    match map.get("checkpoint_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle {what} checkpoint_version must be '1.0.0', got {other:?}"
            )))
        }
    }
    for key in CHECKPOINT_MANIFEST_KEYS {
        // The version was validated above and is not a content digest;
        // hashing its shape here rejects every valid checkpoint manifest.
        if key == "checkpoint_version" {
            continue;
        }
        let value = map.get(key).ok_or_else(|| {
            lc_err(format!("rl lifecycle {what} manifest lacks {key}"))
        })?;
        let optional = matches!(
            key,
            "dataset_manifest_hash" | "rollout_artifact_hash" | "parent_checkpoint_hash"
        );
        match value {
            Value::Null if optional => {}
            Value::String(text) if is_digest_shape(text) => {}
            _ => {
                return Err(lc_err(format!(
                    "rl lifecycle {what} manifest field {key} has a bad shape"
                )))
            }
        }
    }
    Ok(map.clone())
}

/// FULL-JCS manifest id (the manifest carries no digest field to exclude).
pub(crate) fn manifest_id_of(map: &Map<String, Value>, what: &str) -> Result<String, LifecycleError> {
    canonical_digest(&Value::Object(map.clone()), what)
}

pub(crate) fn verify_initial_manifest(
    manifest_doc: &[u8],
    intake: &RlIntake,
) -> Result<String, LifecycleError> {
    let map = manifest_shape_gate(manifest_doc, "initial")?;
    // Cold lineage: warm start comes from a supervised IL checkpoint
    // (dataset source set, rollout null) — never from an RL checkpoint or
    // an unsourced file. Architecture-identical weights are migration-safe;
    // old reference trajectories are never reused (no wall/state crosses).
    let dataset = map.get("dataset_manifest_hash").unwrap_or(&Value::Null);
    let rollout = map.get("rollout_artifact_hash").unwrap_or(&Value::Null);
    match (dataset, rollout) {
        (Value::String(_), Value::Null) => {}
        _ => {
            return Err(lc_err(
                "rl lifecycle initial checkpoint must carry a dataset source and null rollout (validated IL weights)",
            ))
        }
    }
    let manifest_id = manifest_id_of(&map, "rl:checkpoint-manifest")?;
    if manifest_id != intake.initial_checkpoint_id {
        return Err(lc_err(format!(
            "rl lifecycle initial manifest id {manifest_id} != config {}",
            intake.initial_checkpoint_id
        )));
    }
    Ok(manifest_id)
}
