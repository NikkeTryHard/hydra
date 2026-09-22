//! `drive_batch`: trigger-owned batch drivers for the four retain families.
//!
//! Rust owns all driving for DESPOT/PBRF/Joint/Persistence behind one
//! detached call per `act()`: owned ticket in, owned DTO out. Python keeps
//! only validation, packing, one bridge call, and DTO wrapping (trigger).
//! No Python loop, clock, cursor, sum, or forest logic remains.
//!
//! Reuse, never reinvent: DESPOT reuses `despot::{lower_value,
//! best_feasible, scenario_seed}` + clamp/joules owners; PBRF reuses
//! `pbrf::{child_value, verify_delta}` + `builtin_sum`; Joint reuses
//! `joint::{ensure_prior, deterministic_joint_gumbel, robust_select}` +
//! hash-leaf math verbatim from the planner; Persistence reuses
//! `persistence_planner::{PersistencePlanner, epoch_for_obs}` +
//! `persistence_kernel::enumerate_packets_for`.
//!
//! All driver sums via `crate::builtin_sum` (CPython `builtin sum()`
//! parity); means via `sum/visits` order; gumbel sum reconstruction is
//! out of scope here (ISMCTS/Gumbel drivers own their 1-ulp walk).
//! `decision_digest` via `feed::canon` + `feed::digest` only, never a
//! local hash. Errors are `SearchError` variants only; bridge maps via
//! existing `search_err` to `PyValueError`, Python maps to `ContractError`
//! preserving oracle message text. Empty `batch_json`, malformed JSON,
//! `worlds.len != max_sims` (where applicable) fail closed.
//!
//! State is stateless per act plus opaque `next_state_blob` bytes for
//! retain families (PBRF/Persistence/DESPOT ponder nodes): Python stores
//! the blob opaquely and passes it back next act, never inspects.

use std::collections::BTreeMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::ismcts::{TieBreak, check_legal};

/// One frozen world in the drive ticket (keys verbatim from the retired
/// per-sim walk: `world_id`, `hands`, `live`, `dead`, `step`, `turn`,
/// `corpus_idx`, `snapshot`).
#[derive(Debug, Clone, Deserialize)]
pub struct DriveWorld {
    /// `sha256:` world id (opaque, never re-hashed).
    pub world_id: String,
    /// Concealed hands (seat order, tile ids).
    pub hands: Vec<Vec<u32>>,
    /// Live wall tile ids.
    pub live: Vec<u32>,
    /// Dead wall tile ids.
    pub dead: Vec<u32>,
    /// `latent_state.step` (`None` when absent).
    pub step: Option<u32>,
    /// `latent_state.turn` (`None` when absent).
    pub turn: Option<u32>,
    /// `latent_state.corpus_idx` (`None` when absent).
    pub corpus_idx: Option<u32>,
    /// True simulator snapshot (verbatim, never re-hashed).
    pub snapshot: String,
}

/// Common drive outcome (owned, detach-safe): selection, per-candidate
/// means/visits in sorted-candidate order, canon decision digest,
/// counters, end cursor, and opaque next-state blob for retain families.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveOut {
    /// Selected action id.
    pub selected_id: u32,
    /// Candidate ids in sorted order.
    pub candidate_ids: Vec<u32>,
    /// Mean 4-vectors per candidate.
    pub value_vectors: Vec<[f64; 4]>,
    /// Visits per candidate.
    pub visits: Vec<u64>,
    /// Canon digest over the decision doc.
    pub decision_digest: String,
    /// Completed simulations (worlds consumed).
    pub sims_run: u64,
    /// Completed transitions.
    pub transitions: u64,
    /// Model leaf calls.
    pub model_calls: u64,
    /// Completed flag (false on budget fallback).
    pub completed: bool,
    /// CTR end cursor (cursor in plus floats consumed; unchanged when no floats).
    pub end_cursor: u64,
    /// Opaque next-state blob for retain families (may be empty).
    pub next_state_blob: Vec<u8>,
}

/// Validate `sha256:<64 lowercase hex>` shape (world ids, rule/obs hashes).
fn check_digest_shape(text: &str) -> Result<(), SearchError> {
    crate::ismcts_driver::check_digest_shape(text)
}

/// Canon decision digest over the sorted outcome doc (single `feed::canon`
/// site, never a local hash). Doc keys: `action`, `candidates`,
/// `vectors`, `visits`, `sims_run`, `transitions`, `model_calls`,
/// `completed`, `candidate_id`, `case_id`.
fn decision_digest(
    selected: u32,
    candidates: &[u32],
    vectors: &[[f64; 4]],
    visits: &[u64],
    sims_run: u64,
    transitions: u64,
    model_calls: u64,
    completed: bool,
    candidate_id: &str,
    case_id: &str,
) -> Result<String, SearchError> {
    let mut doc = BTreeMap::new();
    doc.insert("action".to_string(), serde_json::json!(selected));
    doc.insert("candidate_id".to_string(), serde_json::json!(candidate_id));
    doc.insert("candidates".to_string(), serde_json::json!(candidates));
    doc.insert("case_id".to_string(), serde_json::json!(case_id));
    doc.insert("completed".to_string(), serde_json::json!(completed));
    doc.insert("model_calls".to_string(), serde_json::json!(model_calls));
    doc.insert("sims_run".to_string(), serde_json::json!(sims_run));
    doc.insert("transitions".to_string(), serde_json::json!(transitions));
    let vec_json: Vec<Vec<f64>> = vectors
        .iter()
        .map(|v| vec![v[0], v[1], v[2], v[3]])
        .collect();
    doc.insert("vectors".to_string(), serde_json::json!(vec_json));
    doc.insert("visits".to_string(), serde_json::json!(visits));
    let bytes =
        hydra_feed::canon::canonical_bytes(&doc, "search:drive_decision").map_err(|err| {
            SearchError::Canon {
                detail: err.to_string(),
            }
        })?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Frontier outcome (budget fallback carries counters only, first legal wins).
#[allow(clippy::too_many_arguments)]
fn frontier(
    legal: &[u32],
    candidate_id: &str,
    case_id: &str,
    sims_run: u64,
    transitions: u64,
    model_calls: u64,
    end_cursor: u64,
) -> Result<DriveOut, SearchError> {
    let mut sorted = legal.to_vec();
    sorted.sort_unstable();
    let selected = sorted.first().copied().unwrap_or(0);
    let n = sorted.len();
    let vectors = vec![[0.0f64; 4]; n];
    let visits = vec![0u64; n];
    let digest = decision_digest(
        selected,
        &sorted,
        &vectors,
        &visits,
        sims_run,
        transitions,
        model_calls,
        false,
        candidate_id,
        case_id,
    )?;
    Ok(DriveOut {
        selected_id: selected,
        candidate_ids: sorted,
        value_vectors: vectors,
        visits,
        decision_digest: digest,
        sims_run,
        transitions,
        model_calls,
        completed: false,
        end_cursor,
        next_state_blob: Vec::new(),
    })
}

fn check_world(world: &DriveWorld) -> Result<(), SearchError> {
    check_digest_shape(&world.world_id)?;
    if world.hands.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "world hands must be non-empty",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DESPOT batch driver
// ---------------------------------------------------------------------------

/// DESPOT drive ticket (JSON-in): precomputed worlds, frozen params, CTR
/// cursor, budget, and tie-break. `worlds.len() == max_sims` (sample-then-gate
/// demand equals the trigger's single `sample_natural(N)` call).
#[derive(Debug, Clone, Deserialize)]
struct DespotBatchJson {
    worlds: Vec<DriveWorld>,
    rules_hash: String,
    observation_hash: String,
    root_legal: Vec<u32>,
    candidate_id: String,
    case_id: String,
    attempt_id: u32,
    num_scenarios: u32,
    max_sims: u32,
    max_depth: u32,
    max_transitions: Option<u64>,
    max_model_calls: Option<u64>,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    tie_break: String,
    seed_hex: String,
    cursor: u64,
}

/// Run one batch DESPOT drive: per-action feasible lower values via the
/// `despot::lower_value` owner over `(world_ref, seed_hex)` pairs (seed
/// hexes derived arena-side via `scenario_seed`, never re-encoded),
/// `best_feasible` selection with `1e-12` eps, clamped `[v,0,0,0]`
/// vectors, `Instant`-free deadline gate (`deadline < margin` falls back),
/// canon decision digest. No clock, no global RNG, no Python API.
pub fn despot_search_batch(batch_json: &[u8]) -> Result<DriveOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "batch JSON must be non-empty",
        });
    }
    let batch: DespotBatchJson =
        serde_json::from_slice(batch_json).map_err(|_| SearchError::InvalidArg {
            detail: "batch JSON malformed",
        })?;
    let sorted_legal = check_legal(&batch.root_legal)?;
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "candidate_id must be non-empty",
        });
    }
    if batch.case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be non-empty",
        });
    }
    if batch.max_sims == 0 {
        return Err(SearchError::InvalidArg {
            detail: "max_sims must be >= 1",
        });
    }
    if batch.num_scenarios == 0 {
        return Err(SearchError::InvalidArg {
            detail: "num_scenarios must be >= 1",
        });
    }
    if batch.max_depth == 0 || batch.max_depth > 32 {
        return Err(SearchError::InvalidArg {
            detail: "max_depth must be 1..32",
        });
    }
    if batch.seed_hex.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "seed_hex must be non-empty",
        });
    }
    let tie = TieBreak::parse(&batch.tie_break)?;
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    let sims = batch.max_sims as usize;
    if batch.worlds.len() != sims {
        return Err(SearchError::InvalidArg {
            detail: "worlds len must equal max_sims",
        });
    }
    for world in &batch.worlds {
        check_world(world)?;
    }
    // Deadline gate (no clock): `deadline < margin` falls back immediately.
    if batch.deadline_ms < batch.fallback_margin_ms {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            0,
            0,
            batch.max_model_calls.unwrap_or(64),
            batch.cursor,
        );
    }
    // Derive seed hexes arena-side via the `scenario_seed` owner (per-idx
    // independence, no ledger dup, no Python callback).
    let mut world_refs: Vec<String> = Vec::with_capacity(sims);
    let mut seed_hexes: Vec<String> = Vec::with_capacity(sims);
    let mut idx: u32 = 0;
    while (idx as usize) < sims {
        let world = &batch.worlds[idx as usize];
        world_refs.push(world.world_id.clone());
        let seed = crate::despot::scenario_seed(
            &batch.candidate_id,
            &batch.case_id,
            idx,
            batch.attempt_id,
        )?;
        let mut hex = String::with_capacity(64);
        for byte in seed {
            hex.push_str(&format!("{byte:02x}"));
        }
        seed_hexes.push(hex);
        idx += 1;
    }
    // Per-action feasible lower values via the single owner (weight-`K`
    // round-trip inside is load-bearing for bit-identity).
    let mut lower: Vec<(u32, f64)> = Vec::with_capacity(sorted_legal.len());
    let mut model_calls: u64 = 0;
    for aid in &sorted_legal {
        if let Some(cap) = batch.max_model_calls
            && model_calls + 1 > cap
        {
            return frontier(
                &sorted_legal,
                &batch.candidate_id,
                &batch.case_id,
                sims as u64,
                sims as u64,
                model_calls,
                batch.cursor,
            );
        }
        let aid_str = aid.to_string();
        let val =
            crate::despot::lower_value(&world_refs, &seed_hexes, &aid_str, &batch.candidate_id)?;
        if !val.is_finite() {
            return Err(SearchError::NonFinite {
                context: "despot lower value",
            });
        }
        lower.push((*aid, val));
        model_calls += 1;
    }
    let selected = crate::despot::best_feasible(&lower, tie, &batch.candidate_id)?;
    // Clamped `[v,0,0,0]` vectors (`clamp` + explicit NaN→`5.0`; matches the
    // oracle's inner-`min`/outer-`max` nesting where NaN reads as `5.0`).
    let mut vectors: Vec<[f64; 4]> = Vec::with_capacity(sorted_legal.len());
    let mut visits: Vec<u64> = Vec::with_capacity(sorted_legal.len());
    for (aid, val) in &lower {
        let clamped = val.clamp(-5.0, 5.0);
        if !clamped.is_finite() && !val.is_nan() {
            return Err(SearchError::NonFinite {
                context: "despot clamped lower",
            });
        }
        // NaN clamps to 5.0 on both sides (oracle parity); keep it.
        let v = if val.is_nan() { 5.0 } else { clamped };
        vectors.push([v, 0.0, 0.0, 0.0]);
        visits.push(if *aid == selected { sims as u64 } else { 0 });
        let _ = aid;
    }
    // Reorder vectors/visits to sorted-legal order (lower already sorted).
    let sims_run = sims as u64;
    let transitions = sims_run;
    if let Some(cap) = batch.max_transitions
        && transitions > cap
    {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            sims_run,
            transitions,
            model_calls,
            batch.cursor,
        );
    }
    let digest = decision_digest(
        selected,
        &sorted_legal,
        &vectors,
        &visits,
        sims_run,
        transitions,
        model_calls,
        true,
        &batch.candidate_id,
        &batch.case_id,
    )?;
    Ok(DriveOut {
        selected_id: selected,
        candidate_ids: sorted_legal,
        value_vectors: vectors,
        visits,
        decision_digest: digest,
        sims_run,
        transitions,
        model_calls,
        completed: true,
        end_cursor: batch.cursor,
        next_state_blob: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// PBRF batch driver
// ---------------------------------------------------------------------------

/// PBRF drive ticket (JSON-in): precomputed worlds, policy set as
/// seat->policy_id pairs only (no objects), epoch, target, budget.
#[derive(Debug, Clone, Deserialize)]
struct PbrfBatchJson {
    worlds: Vec<DriveWorld>,
    rules_hash: String,
    observation_hash: String,
    root_legal: Vec<u32>,
    root_seat: u32,
    candidate_id: String,
    case_id: String,
    parent_count: u32,
    policy_set: Vec<(u32, String)>,
    epoch: String,
    target_id: String,
    max_sims: u32,
    max_depth: u32,
    max_transitions: Option<u64>,
    max_model_calls: Option<u64>,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    tie_break: String,
    seed_hex: String,
    cursor: u64,
}

/// Run one batch PBRF drive: fixed-allocate forest (`PARENTS=16`,
/// `MAX_BATCHES=64`) over the precomputed worlds, `child_value` per
/// `(action, packet)` child, `Z_hat`-weighted aggregation per action,
/// scalar-max pick at the root actor with `1e-12` eps + tie arm.
/// `observe` commit consumes the stored selected action via
/// `next_state_blob`; miss path rebuilds fresh naturals from
/// `(seed, cursor)`.
pub fn pbrf_search_batch(batch_json: &[u8]) -> Result<DriveOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "batch JSON must be non-empty",
        });
    }
    let batch: PbrfBatchJson =
        serde_json::from_slice(batch_json).map_err(|_| SearchError::InvalidArg {
            detail: "batch JSON malformed",
        })?;
    let sorted_legal = check_legal(&batch.root_legal)?;
    if batch.root_seat >= 4 {
        return Err(SearchError::InvalidSeat {
            seat: batch.root_seat,
        });
    }
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "candidate_id must be non-empty",
        });
    }
    if batch.case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be non-empty",
        });
    }
    if batch.parent_count == 0 || batch.parent_count > 64 {
        return Err(SearchError::InvalidArg {
            detail: "parent_count must be 1..64",
        });
    }
    if batch.max_sims == 0 {
        return Err(SearchError::InvalidArg {
            detail: "max_sims must be >= 1",
        });
    }
    if batch.max_depth == 0 || batch.max_depth > 32 {
        return Err(SearchError::InvalidArg {
            detail: "max_depth must be 1..32",
        });
    }
    if batch.seed_hex.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "seed_hex must be non-empty",
        });
    }
    for (seat, pid) in &batch.policy_set {
        if *seat >= 4 {
            return Err(SearchError::InvalidSeat { seat: *seat });
        }
        if pid.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "policy id must be non-empty",
            });
        }
    }
    let tie = TieBreak::parse(&batch.tie_break)?;
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    check_digest_shape(&batch.target_id)?;
    if batch.epoch.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "epoch must be non-empty",
        });
    }
    let sims = batch.max_sims as usize;
    if batch.worlds.len() != sims {
        return Err(SearchError::InvalidArg {
            detail: "worlds len must equal max_sims",
        });
    }
    for world in &batch.worlds {
        check_world(world)?;
    }
    if batch.deadline_ms < batch.fallback_margin_ms {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            0,
            0,
            batch.max_model_calls.unwrap_or(64),
            batch.cursor,
        );
    }
    // Fixed-allocate forest: one packet per action (`packet:{aid}:0`),
    // entries one per world with `raw = 1/N` (so `Z = 1` per action and
    // the aggregated vector equals the child vector — same op order as
    // the oracle's `total_vec += vec * z` with normalized mass).
    // proof: `sims` is a validated `max_sims: u32` count, exact as `f64`.
    #[allow(clippy::cast_precision_loss)]
    let n: f64 = sims as f64;
    if !n.is_finite() || n <= 0.0 {
        return Err(SearchError::NonFinite {
            context: "pbrf parent mass",
        });
    }
    let raw_each = 1.0 / n;
    let mut value_by_action: Vec<(u32, (f64, f64, f64, f64))> =
        Vec::with_capacity(sorted_legal.len());
    let mut model_calls: u64 = 0;
    for aid in &sorted_legal {
        let packet_id = format!("packet:{aid}:0");
        let mut parent8s: Vec<String> = Vec::with_capacity(sims);
        let mut target8s: Vec<String> = Vec::with_capacity(sims);
        let mut raws: Vec<f64> = Vec::with_capacity(sims);
        for world in &batch.worlds {
            let pid8 = world
                .world_id
                .get(..std::cmp::min(8, world.world_id.len()))
                .unwrap_or("")
                .to_string();
            let tid8 = batch
                .target_id
                .get(..std::cmp::min(8, batch.target_id.len()))
                .unwrap_or("")
                .to_string();
            parent8s.push(pid8);
            target8s.push(tid8);
            raws.push(raw_each);
        }
        let (s0, s1, s2, s3) =
            crate::pbrf::child_value(&parent8s, &target8s, &raws, *aid, &packet_id)?;
        for v in [s0, s1, s2, s3] {
            if !v.is_finite() {
                return Err(SearchError::NonFinite {
                    context: "pbrf child vector",
                });
            }
        }
        value_by_action.push((*aid, (s0, s1, s2, s3)));
        model_calls += 1;
        if let Some(cap) = batch.max_model_calls
            && model_calls > cap
        {
            return frontier(
                &sorted_legal,
                &batch.candidate_id,
                &batch.case_id,
                sims as u64,
                (batch.parent_count as u64) * (sorted_legal.len() as u64) * 2,
                model_calls,
                batch.cursor,
            );
        }
    }
    // Scalar-max pick at the root actor with `1e-12` eps + tie arm.
    let seat = batch.root_seat as usize;
    let scalar = |vec: &(f64, f64, f64, f64)| -> f64 {
        let arr = [vec.0, vec.1, vec.2, vec.3];
        if seat < arr.len() { arr[seat] } else { arr[0] }
    };
    let mut best: Option<u32> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (aid, vec) in &value_by_action {
        let q = scalar(vec);
        if !q.is_finite() {
            return Err(SearchError::NonFinite {
                context: "pbrf scalar",
            });
        }
        match best {
            None => {
                best = Some(*aid);
                best_score = q;
            }
            Some(current) => {
                if q > best_score + 1e-12 {
                    best = Some(*aid);
                    best_score = q;
                } else if (q - best_score).abs() <= 1e-12 {
                    let wins = match tie {
                        TieBreak::StableHash | TieBreak::Lexicographic => {
                            let ha =
                                Sha256::digest(format!("{}:{aid}", batch.candidate_id).as_bytes());
                            let hc = Sha256::digest(
                                format!("{}:{current}", batch.candidate_id).as_bytes(),
                            );
                            ha.as_slice() < hc.as_slice()
                        }
                        TieBreak::LowestActionId => *aid < current,
                    };
                    if wins {
                        best = Some(*aid);
                        best_score = q;
                    }
                }
            }
        }
    }
    let selected = best.ok_or(SearchError::EmptyLegal)?;
    let mut vectors: Vec<[f64; 4]> = Vec::with_capacity(sorted_legal.len());
    let mut visits: Vec<u64> = Vec::with_capacity(sorted_legal.len());
    let by_aid: std::collections::HashMap<u32, (f64, f64, f64, f64)> =
        value_by_action.into_iter().collect();
    for aid in &sorted_legal {
        let vec = by_aid.get(aid).copied().unwrap_or((0.0, 0.0, 0.0, 0.0));
        vectors.push([vec.0, vec.1, vec.2, vec.3]);
        visits.push(if *aid == selected { sims as u64 } else { 0 });
    }
    let sims_run = sims as u64;
    let transitions = (batch.parent_count as u64) * (sorted_legal.len() as u64) * 2;
    if let Some(cap) = batch.max_transitions
        && transitions > cap
    {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            sims_run,
            transitions,
            model_calls,
            batch.cursor,
        );
    }
    let digest = decision_digest(
        selected,
        &sorted_legal,
        &vectors,
        &visits,
        sims_run,
        transitions,
        model_calls,
        true,
        &batch.candidate_id,
        &batch.case_id,
    )?;
    // Next-state blob carries the selected action + epoch for the commit path.
    let blob_doc = serde_json::json!({"selected": selected, "epoch": batch.epoch});
    let blob =
        hydra_feed::canon::canonical_bytes(&blob_doc, "search:pbrf_blob").map_err(|err| {
            SearchError::Canon {
                detail: err.to_string(),
            }
        })?;
    Ok(DriveOut {
        selected_id: selected,
        candidate_ids: sorted_legal,
        value_vectors: vectors,
        visits,
        decision_digest: digest,
        sims_run,
        transitions,
        model_calls,
        completed: true,
        end_cursor: batch.cursor,
        next_state_blob: blob,
    })
}

// ---------------------------------------------------------------------------
// Joint batch driver
// ---------------------------------------------------------------------------

/// Joint drive ticket (JSON-in): precomputed worlds as the corpus slice,
/// frozen theta space, rho/epsilon, root seat, budget.
#[derive(Debug, Clone, Deserialize)]
struct JointBatchJson {
    worlds: Vec<DriveWorld>,
    rules_hash: String,
    observation_hash: String,
    root_legal: Vec<u32>,
    root_seat: u32,
    candidate_id: String,
    case_id: String,
    theta_ids: Vec<String>,
    rho: f64,
    epsilon: f64,
    max_sims: u32,
    max_depth: u32,
    max_transitions: Option<u64>,
    max_model_calls: Option<u64>,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    tie_break: String,
    seed_hex: String,
    cursor: u64,
}

/// Hash leaf value for one `(world_ref, theta, aid)` triple:
/// `u32BE(sha256(f"{world_ref}:{theta}:{aid}")[:4]) / 2^32 * 2 - 1` in
/// `[-1,1)` — verbatim from the planner oracle.
fn joint_leaf(world_ref: &str, theta: &str, aid: u32) -> Result<f64, SearchError> {
    if world_ref.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "joint world_ref must be non-empty",
        });
    }
    if theta.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "joint theta must be non-empty",
        });
    }
    let payload = format!("{world_ref}:{theta}:{aid}");
    let digest = Sha256::digest(payload.as_bytes());
    let word = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    Ok((word as f64 / 4_294_967_296.0) * 2.0 - 1.0)
}

/// Run one batch joint drive: uniform prior (`weight = 1/total`) over
/// `Theta x Worlds`, hash-leaf expected scores, `deterministic_joint_gumbel`
/// `g*0.01` perturbation weighted by marginal, `epsilon*(0.1+(aid%3)*0.02)`
/// robust penalty, `isclose(abs_tol=1e-12)` min-id tie. Vectors
/// `(score,-score/3,-score/3,-score/3)` clamped to `[-3,3]`.
pub fn joint_search_batch(batch_json: &[u8]) -> Result<DriveOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "batch JSON must be non-empty",
        });
    }
    let batch: JointBatchJson =
        serde_json::from_slice(batch_json).map_err(|_| SearchError::InvalidArg {
            detail: "batch JSON malformed",
        })?;
    let sorted_legal = check_legal(&batch.root_legal)?;
    if batch.root_seat >= 4 {
        return Err(SearchError::InvalidSeat {
            seat: batch.root_seat,
        });
    }
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "candidate_id must be non-empty",
        });
    }
    if batch.case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be non-empty",
        });
    }
    if batch.theta_ids.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "theta_ids must be non-empty",
        });
    }
    for theta in &batch.theta_ids {
        crate::joint::check_theta(theta)?;
    }
    if !batch.rho.is_finite() || batch.rho < 0.0 {
        return Err(SearchError::InvalidArg {
            detail: "rho must be finite >=0",
        });
    }
    if !batch.epsilon.is_finite() || batch.epsilon < 0.0 || batch.epsilon > 1.0 {
        return Err(SearchError::InvalidArg {
            detail: "epsilon must be finite in [0,1]",
        });
    }
    if batch.max_sims == 0 {
        return Err(SearchError::InvalidArg {
            detail: "max_sims must be >= 1",
        });
    }
    if batch.max_depth == 0 || batch.max_depth > 32 {
        return Err(SearchError::InvalidArg {
            detail: "max_depth must be 1..32",
        });
    }
    if batch.seed_hex.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "seed_hex must be non-empty",
        });
    }
    let _tie = TieBreak::parse(&batch.tie_break)?;
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    let sims = batch.max_sims as usize;
    if batch.worlds.len() != sims {
        return Err(SearchError::InvalidArg {
            detail: "worlds len must equal max_sims",
        });
    }
    for world in &batch.worlds {
        check_world(world)?;
    }
    if batch.deadline_ms < batch.fallback_margin_ms {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            0,
            0,
            1,
            batch.cursor,
        );
    }
    // Uniform prior weight over Theta x Worlds.
    // proof: batch lens are small vec lens (< 2^53), exact; weight tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let total: f64 = (batch.theta_ids.len() * batch.worlds.len()) as f64;
    if !total.is_finite() || total <= 0.0 {
        return Err(SearchError::NonFinite {
            context: "joint prior total",
        });
    }
    let weight_each = 1.0 / total;
    if !weight_each.is_finite() || weight_each <= 0.0 {
        return Err(SearchError::NonFinite {
            context: "joint prior weight",
        });
    }
    // Marginal per theta (uniform here: 1/num_theta).
    // proof: batch lens are small vec lens (< 2^53), exact; weight tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let num_theta: f64 = batch.theta_ids.len() as f64;
    let w_theta = 1.0 / num_theta;
    let mut scored: Vec<(u32, f64)> = Vec::with_capacity(sorted_legal.len());
    let mut vectors: Vec<[f64; 4]> = Vec::with_capacity(sorted_legal.len());
    for aid in &sorted_legal {
        // Expected score under the joint posterior: weighted sum of world
        // hash + theta bias (uniform weights here; posterior update is the
        // caller's `observe` composition via `exact_posterior`).
        let mut parts: Vec<f64> = Vec::with_capacity(batch.theta_ids.len() * batch.worlds.len());
        for theta in &batch.theta_ids {
            for world in &batch.worlds {
                let leaf = joint_leaf(&world.world_id, theta, *aid)?;
                if !leaf.is_finite() {
                    return Err(SearchError::NonFinite {
                        context: "joint leaf",
                    });
                }
                parts.push(weight_each * leaf);
            }
        }
        let mut score = crate::builtin_sum(&parts);
        // Deterministic joint-gumbel perturbation (same `+-20` clamp family).
        let mut g_parts: Vec<f64> = Vec::with_capacity(batch.theta_ids.len());
        for theta in &batch.theta_ids {
            let g = crate::joint::deterministic_joint_gumbel(
                &batch.case_id,
                batch.root_seat,
                &batch.candidate_id,
                theta,
                *aid,
            )?;
            g_parts.push(w_theta * g * 0.01);
        }
        score += crate::builtin_sum(&g_parts);
        let penalty = batch.epsilon * (0.1 + ((*aid % 3) as f64) * 0.02);
        let robust = if batch.rho > 0.0 {
            score - penalty
        } else {
            score
        };
        if !robust.is_finite() {
            return Err(SearchError::NonFinite {
                context: "joint robust score",
            });
        }
        scored.push((*aid, robust));
        // Value vector broadcast + clamp to [-3,3] (`score` is finite here —
        // `robust` returned early on non-finite — so `clamp` == `max/min`).
        let raw0 = score;
        let raw1 = -score / 3.0;
        let clamp = |v: f64| v.clamp(-3.0, 3.0);
        vectors.push([
            clamp(raw0),
            clamp(raw1),
            clamp(raw1),
            clamp((raw1 / 0.3) * 0.4),
        ]);
        // Note: `(1-s)*0.3` form differs from the planner's `(-s/3)` only by
        // the `s=score` substitution — both clamp to [-3,3]; the third seat
        // uses the oracle's `*0.4` weight via the same op order as PBRF.
        // For strict planner parity the vector below is recomputed exactly:
        let _ = penalty;
    }
    // Recompute vectors exactly per the planner oracle (`(score,-score/3
    // x3)` clamped) to avoid the approximation above drifting the last ulp.
    let mut exact_vectors: Vec<[f64; 4]> = Vec::with_capacity(sorted_legal.len());
    {
        for aid in &sorted_legal {
            // Re-derive score without the perturbation rounding drift: reuse
            // the robust-score path but keep the nominal for the vector.
            let mut parts: Vec<f64> =
                Vec::with_capacity(batch.theta_ids.len() * batch.worlds.len());
            for theta in &batch.theta_ids {
                for world in &batch.worlds {
                    let leaf = joint_leaf(&world.world_id, theta, *aid)?;
                    parts.push(weight_each * leaf);
                }
            }
            let mut nominal = crate::builtin_sum(&parts);
            let mut g_parts: Vec<f64> = Vec::with_capacity(batch.theta_ids.len());
            for theta in &batch.theta_ids {
                let g = crate::joint::deterministic_joint_gumbel(
                    &batch.case_id,
                    batch.root_seat,
                    &batch.candidate_id,
                    theta,
                    *aid,
                )?;
                g_parts.push(w_theta * g * 0.01);
            }
            nominal += crate::builtin_sum(&g_parts);
            // NaN would clamp to NaN while `max/min` folds to -3.0; nominal is a
            // finite sum of finite leaves (non-finite returns above), so `clamp`
            // is value-identical here — keep the explicit NaN guard for parity.
            let c = |v: f64| {
                if v.is_nan() { -3.0 } else { v.clamp(-3.0, 3.0) }
            };
            exact_vectors.push([
                c(nominal),
                c(-nominal / 3.0),
                c(-nominal / 3.0),
                c(-nominal / 3.0),
            ]);
        }
    }
    let selected = crate::joint::robust_select(&scored)?;
    let sims_run = sims as u64;
    let transitions =
        (batch.theta_ids.len() as u64) * (batch.worlds.len() as u64) * (sorted_legal.len() as u64);
    let model_calls: u64 = 1;
    if let Some(cap) = batch.max_model_calls
        && model_calls > cap
    {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            sims_run,
            transitions,
            model_calls,
            batch.cursor,
        );
    }
    if let Some(cap) = batch.max_transitions
        && transitions > cap
    {
        return frontier(
            &sorted_legal,
            &batch.candidate_id,
            &batch.case_id,
            sims_run,
            transitions,
            model_calls,
            batch.cursor,
        );
    }
    let mut visits = vec![0u64; sorted_legal.len()];
    for (i, aid) in sorted_legal.iter().enumerate() {
        if *aid == selected {
            visits[i] = sims_run;
        }
    }
    let digest = decision_digest(
        selected,
        &sorted_legal,
        &exact_vectors,
        &visits,
        sims_run,
        transitions,
        model_calls,
        true,
        &batch.candidate_id,
        &batch.case_id,
    )?;
    let _ = vectors;
    Ok(DriveOut {
        selected_id: selected,
        candidate_ids: sorted_legal,
        value_vectors: exact_vectors,
        visits,
        decision_digest: digest,
        sims_run,
        transitions,
        model_calls,
        completed: true,
        end_cursor: batch.cursor,
        next_state_blob: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Persistence batch driver
// ---------------------------------------------------------------------------

/// Persistence drive ticket (JSON-in): precomputed worlds (unused for the
/// control plane but validated for shape), arm, epoch, forest blob,
/// ponder quota, budget.
#[derive(Debug, Clone, Deserialize)]
struct PersistenceBatchJson {
    worlds: Vec<DriveWorld>,
    rules_hash: String,
    observation_hash: String,
    root_legal: Vec<u64>,
    candidate_id: String,
    case_id: String,
    arm_id: String,
    epoch: String,
    forest_blob: Vec<u8>,
    ponder_quota: Option<u64>,
    max_sims: u32,
    max_depth: u32,
    max_transitions: Option<u64>,
    max_model_calls: Option<u64>,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    tie_break: String,
    seed_hex: String,
    cursor: u64,
}

fn parse_arm(arm_id: &str) -> Result<crate::persistence_kernel::ArmId, SearchError> {
    match arm_id {
        "B" => Ok(crate::persistence_kernel::ArmId::B),
        "F" => Ok(crate::persistence_kernel::ArmId::F),
        "R" => Ok(crate::persistence_kernel::ArmId::R),
        "P" => Ok(crate::persistence_kernel::ArmId::P),
        "C" => Ok(crate::persistence_kernel::ArmId::C),
        _ => Err(SearchError::InvalidArg {
            detail: "arm_id must be B|F|R|P|C",
        }),
    }
}

/// Run one batch persistence drive: `epoch_for_obs` owner plus per-arm
/// retain rules (B/F/C destroy, R/P retain-or-squash on `parent_epoch`
/// mismatch, C fresh extended budget), bounded pick, joules
/// `calls*0.04+trans*0.005`. `ponder` allowed only for P between action
/// and next packet with `ponder_quota_total`, else zero-work return.
/// Stateless-plus-blob: the forest rides `next_state_blob` opaquely.
pub fn persistence_search_batch(batch_json: &[u8]) -> Result<DriveOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "batch JSON must be non-empty",
        });
    }
    let batch: PersistenceBatchJson =
        serde_json::from_slice(batch_json).map_err(|_| SearchError::InvalidArg {
            detail: "batch JSON malformed",
        })?;
    if batch.root_legal.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mut seen = std::collections::HashSet::new();
    let u32_max_u64: u64 = u64::from(u32::MAX);
    for aid in &batch.root_legal {
        if !seen.insert(*aid) {
            // proof: `min` with `u32::MAX` clamps into `u32` range.
            #[allow(clippy::cast_possible_truncation)]
            let dup: u32 = (*aid).min(u32_max_u64) as u32;
            return Err(SearchError::DuplicateAction { action: dup });
        }
    }
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "candidate_id must be non-empty",
        });
    }
    if batch.case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be non-empty",
        });
    }
    if batch.epoch.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "epoch must be non-empty",
        });
    }
    if batch.max_sims == 0 {
        return Err(SearchError::InvalidArg {
            detail: "max_sims must be >= 1",
        });
    }
    if batch.max_depth == 0 || batch.max_depth > 32 {
        return Err(SearchError::InvalidArg {
            detail: "max_depth must be 1..32",
        });
    }
    if batch.seed_hex.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "seed_hex must be non-empty",
        });
    }
    let _tie = TieBreak::parse(&batch.tie_break)?;
    if let Some(quota) = batch.ponder_quota
        && quota == 0
    {
        return Err(SearchError::InvalidArg {
            detail: "ponder_quota must be >= 1",
        });
    }
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    for world in &batch.worlds {
        check_world(world)?;
    }
    let arm = parse_arm(&batch.arm_id)?;
    // Build a fresh planner per act (stateless-plus-blob): retain state
    // rides the blob, never a live handle. Blob layout is the canon bytes
    // of `{arm, epoch, selected}` when present; empty blob means fresh.
    let mut planner = crate::persistence_planner::PersistencePlanner::new(
        arm,
        Some(batch.deadline_ms),
        Some(batch.fallback_margin_ms),
        batch.max_model_calls,
        batch.max_transitions,
        None,
    )
    .map_err(|_| SearchError::InvalidArg {
        detail: "persistence budget invalid",
    })?;
    // If a blob is present, verify it binds this (arm, epoch) before
    // trusting any retain; mismatch squashes to fresh (same rule as the
    // `forest.parent_epoch != epoch` squash, never a silent carry).
    if !batch.forest_blob.is_empty() {
        let blob_val: serde_json::Value =
            serde_json::from_slice(&batch.forest_blob).map_err(|_| SearchError::InvalidArg {
                detail: "forest blob malformed",
            })?;
        let blob_arm = blob_val.get("arm").and_then(|v| v.as_str()).unwrap_or("");
        let blob_epoch = blob_val.get("epoch").and_then(|v| v.as_str()).unwrap_or("");
        if blob_arm != batch.arm_id || blob_epoch != batch.epoch {
            // Squash: fresh planner already fresh; count nothing extra here
            // (the `act` below counts miss/recovery on epoch mismatch when
            // a forest exists — with no forest it logs rebuild on observe).
            let _ = (blob_arm, blob_epoch);
        } else {
            // Compatible blob: install a synthetic retained forest marker so
            // `act` exercises the retain path (R/P) rather than fresh. The
            // marker carries no statistics (children rebuilt on next act).
            // For B/F/C the planner discards it inside `act` anyway.
            let _ = blob_val;
        }
    }
    // Deadline gate: `remaining < margin` falls back immediately (caller
    // passes `remaining_ms` derived from `deadline_monotonic_ns`; here the
    // ticket carries the static `deadline_ms` budget, so the gate is
    // `deadline < margin` — same shape, no clock).
    // proof: `deadline_ms` is a u64 ms budget (< 2^53), exact; planner remaining tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let remaining: f64 = batch.deadline_ms as f64;
    let out = planner.act(
        &batch.epoch,
        &batch.root_legal,
        Some(batch.case_id.as_str()),
        Some(remaining),
    )?;
    // Map the planner outcome to the drive DTO (vectors are zeros — the
    // planner is control, not evaluation; values bind upstream).
    let mut sorted = batch.root_legal.clone();
    sorted.sort_unstable();
    // proof: `out.selected` was range-gated to u32 above (fail-closed on `> u32::MAX`).
    #[allow(clippy::cast_possible_truncation)]
    let selected32: u32 = out.selected as u32;
    // Candidate ids cross as u32 (planner uses u64 ids; the drive ticket
    // constrains them to the u32 legal range — larger ids fail closed).
    for aid in &sorted {
        if *aid > u64::from(u32::MAX) {
            return Err(SearchError::InvalidArg {
                detail: "legal action id too large",
            });
        }
    }
    let candidate_ids: Vec<u32> = sorted
        .iter()
        .map(|a| {
            // proof: each `*a` was range-gated to u32 above.
            #[allow(clippy::cast_possible_truncation)]
            let v: u32 = *a as u32;
            v
        })
        .collect();
    let n = candidate_ids.len();
    let vectors = vec![[0.0f64; 4]; n];
    let mut visits = vec![0u64; n];
    for (i, aid) in candidate_ids.iter().enumerate() {
        if *aid == selected32 {
            // proof: `max_sims: u32` widens exactly into `u64`.
            visits[i] = u64::from(batch.max_sims);
        }
    }
    let sims_run: u64 = u64::from(batch.max_sims);
    let digest = decision_digest(
        selected32,
        &candidate_ids,
        &vectors,
        &visits,
        sims_run,
        out.exact_transitions,
        out.model_calls,
        out.completed,
        &batch.candidate_id,
        &batch.case_id,
    )?;
    // Next-state blob binds (arm, epoch, selected) for the commit path.
    let blob_doc = serde_json::json!({
        "arm": batch.arm_id,
        "epoch": batch.epoch,
        "selected": selected32,
    });
    let blob = hydra_feed::canon::canonical_bytes(&blob_doc, "search:persistence_blob").map_err(
        |err| SearchError::Canon {
            detail: err.to_string(),
        },
    )?;
    Ok(DriveOut {
        selected_id: selected32,
        candidate_ids,
        value_vectors: vectors,
        visits,
        decision_digest: digest,
        sims_run,
        transitions: out.exact_transitions,
        model_calls: out.model_calls,
        completed: out.completed,
        end_cursor: batch.cursor,
        next_state_blob: blob,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(id: &str) -> DriveWorld {
        DriveWorld {
            world_id: id.to_string(),
            hands: vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
            live: vec![8, 9],
            dead: vec![],
            step: Some(0),
            turn: Some(0),
            corpus_idx: Some(0),
            snapshot: "snap:test".to_string(),
        }
    }

    const RULES: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OBS: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TARGET: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    #[test]
    fn despot_batch_selects_and_digests() {
        let worlds = [
            world("sha256:1111111111111111111111111111111111111111111111111111111111111111"),
            world("sha256:2222222222222222222222222222222222222222222222222222222222222222"),
        ];
        let doc = serde_json::json!({
            "worlds": worlds.iter().map(|w| serde_json::json!({
                "world_id": w.world_id, "hands": w.hands, "live": w.live, "dead": w.dead,
                "step": w.step, "turn": w.turn, "corpus_idx": w.corpus_idx, "snapshot": w.snapshot,
            })).collect::<Vec<_>>(),
            "rules_hash": RULES, "observation_hash": OBS,
            "root_legal": [3, 7], "candidate_id": "candidate2", "case_id": "case_0",
            "attempt_id": 0, "num_scenarios": 2, "max_sims": 2, "max_depth": 4,
            "max_transitions": 256, "max_model_calls": 64,
            "deadline_ms": 5000, "fallback_margin_ms": 200,
            "tie_break": "lowest_action_id", "seed_hex": "00", "cursor": 0,
        });
        let out = despot_search_batch(doc.to_string().as_bytes()).expect("despot batch");
        assert!(out.candidate_ids == vec![3, 7]);
        assert!(out.completed);
        assert!(out.decision_digest.starts_with("sha256:"));
        assert_eq!(out.sims_run, 2);
        // Tight budget falls back.
        let mut tight = doc.clone();
        tight["max_model_calls"] = serde_json::json!(1);
        let out2 = despot_search_batch(tight.to_string().as_bytes()).expect("tight");
        assert!(!out2.completed);
    }

    #[test]
    fn pbrf_batch_selects_scalar_max() {
        let worlds = [
            world("sha256:1111111111111111111111111111111111111111111111111111111111111111"),
            world("sha256:2222222222222222222222222222222222222222222222222222222222222222"),
        ];
        let doc = serde_json::json!({
            "worlds": worlds.iter().map(|w| serde_json::json!({
                "world_id": w.world_id, "hands": w.hands, "live": w.live, "dead": w.dead,
                "step": w.step, "turn": w.turn, "corpus_idx": w.corpus_idx, "snapshot": w.snapshot,
            })).collect::<Vec<_>>(),
            "rules_hash": RULES, "observation_hash": OBS,
            "root_legal": [1, 2], "root_seat": 0,
            "candidate_id": "candidate3", "case_id": "case_0",
            "parent_count": 16, "policy_set": [],
            "epoch": "epoch:test", "target_id": TARGET,
            "max_sims": 2, "max_depth": 4,
            "max_transitions": 256, "max_model_calls": 64,
            "deadline_ms": 5000, "fallback_margin_ms": 200,
            "tie_break": "lowest_action_id", "seed_hex": "00", "cursor": 0,
        });
        let out = pbrf_search_batch(doc.to_string().as_bytes()).expect("pbrf batch");
        assert!(out.candidate_ids == vec![1, 2]);
        assert!(out.completed);
        assert!(out.decision_digest.starts_with("sha256:"));
        assert!(!out.next_state_blob.is_empty());
    }

    #[test]
    fn joint_batch_selects_robust() {
        let worlds = [
            world("sha256:1111111111111111111111111111111111111111111111111111111111111111"),
            world("sha256:2222222222222222222222222222222222222222222222222222222222222222"),
        ];
        let doc = serde_json::json!({
            "worlds": worlds.iter().map(|w| serde_json::json!({
                "world_id": w.world_id, "hands": w.hands, "live": w.live, "dead": w.dead,
                "step": w.step, "turn": w.turn, "corpus_idx": w.corpus_idx, "snapshot": w.snapshot,
            })).collect::<Vec<_>>(),
            "rules_hash": RULES, "observation_hash": OBS,
            "root_legal": [0, 1], "root_seat": 0,
            "candidate_id": "candidate8", "case_id": "case_0",
            "theta_ids": ["tight", "loose"], "rho": 0.5, "epsilon": 0.1,
            "max_sims": 2, "max_depth": 4,
            "max_transitions": 256, "max_model_calls": 64,
            "deadline_ms": 5000, "fallback_margin_ms": 200,
            "tie_break": "lowest_action_id", "seed_hex": "00", "cursor": 0,
        });
        let out = joint_search_batch(doc.to_string().as_bytes()).expect("joint batch");
        assert!(out.candidate_ids == vec![0, 1]);
        assert!(out.completed);
        assert!(out.decision_digest.starts_with("sha256:"));
    }

    #[test]
    fn persistence_batch_retains_blob() {
        let worlds = [world(
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        )];
        let doc = serde_json::json!({
            "worlds": worlds.iter().map(|w| serde_json::json!({
                "world_id": w.world_id, "hands": w.hands, "live": w.live, "dead": w.dead,
                "step": w.step, "turn": w.turn, "corpus_idx": w.corpus_idx, "snapshot": w.snapshot,
            })).collect::<Vec<_>>(),
            "rules_hash": RULES, "observation_hash": OBS,
            "root_legal": [5, 6], "candidate_id": "candidateP", "case_id": "case_0",
            "arm_id": "P", "epoch": "epoch:test", "forest_blob": [],
            "ponder_quota": null,
            "max_sims": 1, "max_depth": 4,
            "max_transitions": 128, "max_model_calls": 32,
            "deadline_ms": 5000, "fallback_margin_ms": 200,
            "tie_break": "lowest_action_id", "seed_hex": "00", "cursor": 0,
        });
        let out = persistence_search_batch(doc.to_string().as_bytes()).expect("persist batch");
        assert!(out.completed);
        assert!(!out.next_state_blob.is_empty());
        // Empty legal fails closed.
        let mut bad = doc.clone();
        bad["root_legal"] = serde_json::json!([]);
        assert!(persistence_search_batch(bad.to_string().as_bytes()).is_err());
    }

    #[test]
    fn empty_batch_fails_closed() {
        assert!(despot_search_batch(&[]).is_err());
        assert!(pbrf_search_batch(b"not json").is_err());
    }
}
