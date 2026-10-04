//! rl/pool/profile: closed cohort profile + prescribed sampler streams.
//!
//! Invariant: `PoolProfile::parse` rejects unknown fields and any split scope
//! other than `selection_training`, so every (game, seat) `actor_policy_sample`
//! key derives from validated pipeline material only (master seed, experiment,
//! generation split, candidate, case, seat, replicate/attempt). `sampler_draw`
//! reserves exactly 8 CTR bytes per real decision and fails closed on counter
//! overflow, so restores replay counters without redrawing. Failure mode: any
//! malformed profile/context, empty identity, or seat outside 0..=3 returns
//! `PoolError` (fail closed, never a default stream).

use super::super::core::GameContext;
use super::super::wall;
use super::{PoolError, check_closed, hex32, json_str, json_u64};

/// One cohort game: fixed wall + seating + identities.
#[derive(Debug, Clone)]
pub struct CohortGame {
    pub wall_id: String,
    pub schedule_id: String,
    pub schedule_digest: String,
    pub schedule_tiles: [u8; 136],
    pub seat_perm: [u8; 4],
}

/// Validated pool profile (closed fields; unknown rejected).
///
/// Key provenance for the prescribed `actor_policy_sample` sampler:
/// `master_seed_hex` is the selection-training scope material (hex);
/// `experiment_id` is the frozen pipeline experiment; `replicate_id` and
/// `attempt_id` are the explicit pipeline scope ( retries bump attempt via
/// `retry_key`); `generation` doubles as the collection split scope
/// (`selection_training:g={g}`). The pool derives one semantic seed per
/// (game, seat) from these fields; the lifecycle descriptor MUST carry them
/// all so the key is byte-exact across fork/restore.
#[derive(Debug, Clone)]
pub struct PoolProfile {
    pub pool_id: String,
    pub generation: u64,
    pub policy_id: String,
    pub master_seed_hex: String,
    pub experiment_id: String,
    pub split_scope: String,
    pub replicate_id: u64,
    pub attempt_id: u64,
    pub starting_points: i32,
    pub worker_count: usize,
    pub games: Vec<CohortGame>,
    pub shared: SharedIdentities,
}

/// Identities shared by every cohort game (from context JSON).
#[derive(Debug, Clone)]
pub struct SharedIdentities {
    pub rules_id: String,
    pub rules_hash: String,
    pub action_table_hash: String,
    pub event_schema_hash: String,
    pub observation_schema_hash: String,
    pub packet_boundary_hash: String,
    pub adapter_version: String,
    pub rank_values: [f64; 4],
    pub behavior_ckpt: [u8; 32],
}

impl PoolProfile {
    /// Parse + validate the closed profile/context pair.
    pub fn parse(profile_json: &str, context_json: &str) -> Result<Self, PoolError> {
        let profile: serde_json::Value = serde_json::from_str(profile_json)
            .map_err(|e| PoolError(format!("profile JSON rejected: {e}")))?;
        let context: serde_json::Value = serde_json::from_str(context_json)
            .map_err(|e| PoolError(format!("context JSON rejected: {e}")))?;
        check_closed(
            &profile,
            &[
                "pool_id",
                "generation",
                "policy_id",
                "master_seed_hex",
                "experiment_id",
                "split_scope",
                "replicate_id",
                "attempt_id",
                "starting_points",
                "worker_count",
                "games",
            ],
            "pool profile",
        )?;
        check_closed(
            &context,
            &[
                "rules_id",
                "rules_hash",
                "action_table_hash",
                "event_schema_hash",
                "observation_schema_hash",
                "packet_boundary_hash",
                "adapter_version",
                "rank_values",
                "behavior_ckpt",
            ],
            "pool context",
        )?;
        let games_value =
            profile.get("games").and_then(|g| g.as_array()).ok_or_else(|| {
                PoolError("profile missing games array".to_string())
            })?;
        if games_value.is_empty() {
            return Err(PoolError("cohort must hold at least one game".to_string()));
        }
        let mut games = Vec::with_capacity(games_value.len());
        for (i, game) in games_value.iter().enumerate() {
            check_closed(
                game,
                &["wall_id", "schedule_id", "schedule_digest", "schedule_tiles", "seat_perm"],
                "cohort game",
            )?;
            let tiles =
                game.get("schedule_tiles").and_then(|t| t.as_array()).ok_or_else(|| {
                    PoolError(format!("game {i} missing schedule_tiles"))
                })?;
            if tiles.len() != 136 {
                return Err(PoolError(format!("game {i} schedule_tiles must hold 136 tiles")));
            }
            let mut schedule_tiles = [0u8; 136];
            for (j, tile) in tiles.iter().enumerate() {
                let tid = tile
                    .as_u64()
                    .ok_or_else(|| PoolError(format!("game {i} tile {j} must be an int")))?;
                if tid > 135 {
                    return Err(PoolError(format!("game {i} tile {j} out of range")));
                }
                schedule_tiles[j] = tid as u8;
            }
            let perm_value =
                game.get("seat_perm").and_then(|p| p.as_array()).ok_or_else(|| {
                    PoolError(format!("game {i} missing seat_perm"))
                })?;
            if perm_value.len() != 4 {
                return Err(PoolError(format!("game {i} seat_perm must hold four seats")));
            }
            let mut seat_perm = [0u8; 4];
            for (j, seat) in perm_value.iter().enumerate() {
                seat_perm[j] = seat
                    .as_u64()
                    .ok_or_else(|| PoolError(format!("game {i} seat {j} must be an int")))?
                    as u8;
            }
            games.push(CohortGame {
                wall_id: json_str(game, "wall_id")?,
                schedule_id: json_str(game, "schedule_id")?,
                schedule_digest: json_str(game, "schedule_digest")?,
                schedule_tiles,
                seat_perm,
            });
        }
        let rank_values =
            context.get("rank_values").and_then(|v| v.as_array()).ok_or_else(|| {
                PoolError("context missing rank_values".to_string())
            })?;
        if rank_values.len() != 4 {
            return Err(PoolError("rank_values must hold exactly 4 numbers".to_string()));
        }
        let mut ranks = [0.0f64; 4];
        for (i, value) in rank_values.iter().enumerate() {
            ranks[i] = value
                .as_f64()
                .ok_or_else(|| PoolError(format!("rank_values[{i}] must be a number")))?;
            if !ranks[i].is_finite() {
                return Err(PoolError("rank_values must be finite".to_string()));
            }
        }
        let worker_count =
            profile.get("worker_count").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        if worker_count == 0 || worker_count > 64 {
            return Err(PoolError("worker_count outside 1..=64".to_string()));
        }
        let master_seed_hex = json_str(&profile, "master_seed_hex")?;
        hex32(&master_seed_hex, "master_seed_hex")?;
        let experiment_id = json_str(&profile, "experiment_id")?;
        if experiment_id.is_empty() {
            return Err(PoolError("profile experiment_id must be nonempty".to_string()));
        }
        // Authority gate: actor sampling runs on selection-training
        // material only; any other scope (in particular final_evaluation)
        // is rejected loudly rather than derived under.
        let split_scope = json_str(&profile, "split_scope")?;
        if split_scope != "selection_training" {
            return Err(PoolError(format!(
                "profile split_scope {split_scope:?} cannot feed actor_policy_sample"
            )));
        }
        Ok(Self {
            pool_id: json_str(&profile, "pool_id")?,
            generation: json_u64(&profile, "generation")?,
            policy_id: json_str(&profile, "policy_id")?,
            master_seed_hex,
            experiment_id,
            split_scope,
            replicate_id: json_u64(&profile, "replicate_id")?,
            attempt_id: json_u64(&profile, "attempt_id")?,
            starting_points: profile
                .get("starting_points")
                .and_then(|v| v.as_i64())
                .unwrap_or(25000) as i32,
            worker_count,
            games,
            shared: SharedIdentities {
                rules_id: json_str(&context, "rules_id")?,
                rules_hash: json_str(&context, "rules_hash")?,
                action_table_hash: json_str(&context, "action_table_hash")?,
                event_schema_hash: json_str(&context, "event_schema_hash")?,
                observation_schema_hash: json_str(&context, "observation_schema_hash")?,
                packet_boundary_hash: json_str(&context, "packet_boundary_hash")?,
                adapter_version: json_str(&context, "adapter_version")?,
                rank_values: ranks,
                behavior_ckpt: hex32(&json_str(&context, "behavior_ckpt")?, "behavior_ckpt")?,
            },
        })
    }

    /// Build the engine context for one cohort game (restore replays against
    /// the identical walls, so this mapping must stay byte-exact).
    pub(super) fn game_context(&self, game: &CohortGame) -> GameContext {
        GameContext {
            rules_id: self.shared.rules_id.clone(),
            rules_hash: self.shared.rules_hash.clone(),
            action_table_hash: self.shared.action_table_hash.clone(),
            event_schema_hash: self.shared.event_schema_hash.clone(),
            observation_schema_hash: self.shared.observation_schema_hash.clone(),
            packet_boundary_hash: self.shared.packet_boundary_hash.clone(),
            adapter_version: self.shared.adapter_version.clone(),
            schedule_id: game.schedule_id.clone(),
            schedule_digest: game.schedule_digest.clone(),
            schedule_tiles: game.schedule_tiles,
            seat_perm: game.seat_perm,
            rank_values: self.shared.rank_values,
            behavior_ckpt: self.shared.behavior_ckpt,
        }
    }
}

/// Collection split scope for actor sampling (RlLifecycle pin):
/// `{split_scope}:g={generation}` with `split_scope == "selection_training"`
/// (profile-validated): generation-scoped, epoch-independent, deterministic
/// given the same op sequence. The key schema forbids generation/epoch
/// anywhere else, so the split string is the only generation carrier.
pub fn sampler_split_id(split_scope: &str, generation: u64) -> String {
    format!("{split_scope}:g={generation}")
}

/// Semantic seed for one (game, seat) stream: sha256 over the JCS payload
/// `{"protocol": "hydra2_rng_v1", "master_seed": hex, "key": 17-field doc}`
/// with purpose `actor_policy_sample`, case = derived game id, candidate =
/// behavior/policy id, root_seat = seat, wall null, replicate/attempt
/// explicit, all other fields null (oracle: `randomness.py:162-185`
/// required/optional matrix + exactly-one-of case/wall rule).
/// Single printer: `hydra_search::eval::semantic_seed` (the same owner the
/// shuffle law uses); no second key printer lives in this module.
pub fn sampler_seed(
    master_seed_hex: &str,
    experiment_id: &str,
    split_id: &str,
    candidate_id: &str,
    game_id: &str,
    seat: u8,
    replicate_id: u64,
    attempt_id: u64,
) -> Result<[u8; 32], PoolError> {
    let master = hex32(master_seed_hex, "master_seed_hex")?;
    if experiment_id.is_empty() || split_id.is_empty() || candidate_id.is_empty() {
        return Err(PoolError("sampler key needs nonempty experiment/split/candidate".to_string()));
    }
    if game_id.is_empty() {
        return Err(PoolError("sampler key needs a nonempty case (game) id".to_string()));
    }
    if seat > 3 {
        return Err(PoolError(format!("sampler root_seat {seat} outside 0..=3")));
    }
    let key = serde_json::json!({
        "purpose": "actor_policy_sample",
        "experiment_id": experiment_id,
        "split_id": split_id,
        "candidate_id": candidate_id,
        "case_id": game_id,
        "wall_id": null,
        "root_seat": seat,
        "belief_epoch": null,
        "parent_id": null,
        "action_id": null,
        "packet_id": null,
        "fidelity_level": null,
        "population_id": null,
        "replicate_id": replicate_id,
        "scramble_id": null,
        "visit_index": null,
        "attempt_id": attempt_id,
    });
    hydra_search::eval::semantic_seed(&master, &key)
        .map_err(|e| PoolError(format!("sampler seed rejected: {e:?}")))
}

/// One uniform draw: 8 bytes at `[counter * 8, counter * 8 + 8)` of the
/// feed CTR window over the semantic seed, `(u64BE >> 11) * 2^-53` in
/// `[0, 1)` (forced singletons included: reservation is per real decision,
/// never per row). Returns `(cursor_before, uniform)`; the cursor binds
/// into the captured row so restores verify without redrawing.
pub fn sampler_draw(seed: &[u8; 32], counter: u64) -> Result<(u64, f64), PoolError> {
    let cursor = counter
        .checked_mul(8)
        .ok_or_else(|| PoolError("sampler counter overflows the byte window".to_string()))?;
    let mut words = wall::ctr_u53_window(seed, cursor, 1)
        .map_err(|e| PoolError(format!("sampler reserve rejected: {e:?}")))?;
    if words.len() != 1 {
        return Err(PoolError("sampler reserve returned no word".to_string()));
    }
    let uniform = words.pop().ok_or_else(|| PoolError("sampler reserve empty".to_string()))?;
    if !(0.0..1.0).contains(&uniform) {
        return Err(PoolError("sampler uniform escapes [0, 1)".to_string()));
    }
    Ok((cursor, uniform))
}
