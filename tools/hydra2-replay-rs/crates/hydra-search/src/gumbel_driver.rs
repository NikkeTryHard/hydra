//! `gumbel_driver`: sequential-halving batch driver (rounds/aids/visits loops).
//!
//! Owns the `gumbel_search.py:419-547` halving loop plus the `_rollout`
//! (`273-326`) descent/continuation replay and the `gumbel_search.py:549-599`
//! final selection, in the same batch shape as `ismcts_driver`
//! (batch-worlds-in / leaf-overrides-in-batch / selection-out, one bridge
//! call per search, GIL released, no per-step Python crossing).
//!
//! Python precomputes the batch ONCE per search (sampled worlds in oracle
//! visit order, per-rollout policy directions, CTR policy floats, root
//! Gumbels) and Rust replays the oracle visit order bit-identically:
//! rounds `0..halving_rounds`, survivor slots in survivor-tuple order,
//! `visits_per_round[r]` visits each. Worlds ride round-major, slot-minor,
//! visit-minor; budget breaks only truncate the consumption tail (counters
//! never reset), so a sequential `world_ptr` replays the oracle sample
//! order exactly.
//!
//! Oracle-consumption fidelity notes (`gumbel_search.py` line refs):
//! - Pre-rollout gate (`429-451`): only the transitions cap breaks the
//!   visit loop. The model-call pre-check (`437-451`) breaks solely when the
//!   transitions cap is ALSO hit, which the first check already caught, so
//!   it is a no-op and is not replayed (documented, never forked).
//! - Forced root transition (`290-291`) runs unconditionally once a visit
//!   is entered (no gate before it); only the pre-rollout check guards entry.
//! - Every continuation step samples the policy (`293-309`): one CTR float
//!   per step even when the transitions gate breaks right after
//!   (sample-then-gate, `302-306`), and even for single-legal steps (the
//!   oracle draws `random_float` unconditionally, `gumbel_search.py:115`).
//!   Replay consumes one envelope float per continuation step the same way.
//! - Root-seat steps past the root are policy steps too (`300-301`: the
//!   continuation dict covers all four seats); there is no tree, no UCT,
//!   and no info key anywhere in this driver.
//! - The local `step` counter starts at `1` after the forced root (`292`)
//!   regardless of latent content, and terminal reads
//!   `step >= max_depth || live empty` (`344-353`); replay tracks the same
//!   explicit counter rather than the world latent.
//! - Leaf rule (`315-325`): terminal goes terminal vector; else an exhausted
//!   model budget falls back to terminal (no model call counted); else the
//!   model vector (+1 model call). Read-only leaf overrides win first
//!   (torch path, same precedence as the ISMCTS driver).
//! - Cut rule (`496-534`): when every survivor was visited the cut rides
//!   `halving_cut` (`score = q + g`, `1e-12` eps + tie arm); otherwise the
//!   exact Python fallback runs here (descending exact score, tie arm, keep
//!   `ceil(n/2)`, `-inf` allowed) — the bridge cut rejects non-finite, so
//!   the fallback MUST live in this driver for budget-starved parity.
//! - Final rule (`561-599`): same split (`gumbel_select` vs the eps-gated
//!   scan with the max-Gumbel default).
//! - Survivor ORDER rides the cut output (ranked, truncated): the next
//!   round iterates survivors in cut order, so float alignment depends on
//!   it. Reusing `halving_cut` for the complete path keeps the order bit
//!   identical.
//! - Unvisited value vectors (`601-615`): the oracle hashes the
//!   `unvisited:{aid}` tag into a synth world (`_world_for_particle`
//!   `232-271`) and reads the model stub over its world id — NOT the bare
//!   tag hash. `synth_placeholder` replicates that path through the shared
//!   `world_id_for` (canon-verbatim identity doc), so starved arms match.
//!
//! Firewall: world blobs cross in (never out); directions/floats/Gumbels are
//! scalars; tree keys never appear. Every shape violation fails closed
//! (`SearchError`, never a default). No clock, no global RNG, no Python API.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::Deserialize;

use crate::ismcts_driver::{
    BatchWorldJson, DOMAIN_GUMBEL, LeafOverrideJson, TinyWorld, check_digest_shape,
    continuation_sample, exact_transition, model_vector, terminal_vector, world_id_for,
};
use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::gumbel::{GumbelParams, gumbel_select, halving_cut};
use crate::ismcts::{TieBreak, check_legal};

/// Final-selection eps gate (`gumbel_search.py:582,585`): score gaps within
/// `1e-12` are ties for the tie-break arm.
pub const GUMBEL_FINAL_EPS: f64 = 1e-12;

/// Full halving batch envelope: precomputed worlds, per-rollout policy
/// directions, CTR floats, root Gumbels, frozen params, leaf overrides.
#[derive(Debug, Clone, Deserialize)]
struct HalvingBatchJson {
    /// One world per executed rollout in oracle visit order (round-major,
    /// survivor-slot-minor, visit-minor); budget truncation stops sampling
    /// mid-schedule, so the envelope holds exactly the sampled prefix —
    /// never more than the deterministic schedule total. Trailing surplus
    /// fails closed at the end of replay (`world_ptr` must match).
    worlds: Vec<BatchWorldJson>,
    /// Passthrough rules hash for successors (+ synth placeholders).
    rules_hash: String,
    /// Passthrough observation hash for successors (+ synth placeholders).
    observation_hash: String,
    /// Root legal action ids (sorted-unique checked; forced root arms).
    root_legal: Vec<u32>,
    /// Root seat `0..3` (scalarization seat + synth-placeholder turn).
    root_seat: u32,
    /// Root Gumbels `[[action, g], ...]` (must cover exactly `root_legal`).
    gumbels: Vec<(u32, f64)>,
    /// Halving rounds `1..=5`.
    halving_rounds: u32,
    /// Visits per round (`len == halving_rounds`, each `1..=64`).
    visits_per_round: Vec<u32>,
    /// Continuation legal set (sorted-unique checked; every policy step).
    continuation_legal: Vec<u32>,
    /// Per-rollout policy tilt dirs (one row per executed rollout,
    /// `max_depth` cols, values `0|1`; column `j` feeds the `j`-th
    /// continuation step, unread tail carries `0`).
    policy_dirs: Vec<Vec<u32>>,
    /// CTR policy floats in consumption order (one per continuation step
    /// actually sampled, sample-then-gate included).
    rng_floats: Vec<f64>,
    /// Descent depth `1..=32`.
    max_depth: u32,
    /// Transition cap (`None` = unbounded).
    max_transitions: Option<u64>,
    /// Model-call cap (`None` = unbounded; exhaustion falls back to
    /// terminal, never stops the search).
    max_model_calls: Option<u64>,
    /// Leaf salt (`candidate6` in Gumbel; kept, never defaulted).
    candidate_id: String,
    /// Snapshot salt (must be `gumbel`; kept explicit, never defaulted).
    domain: String,
    /// Tie-break arm name.
    tie_break: String,
    /// Optional read-only leaf overrides (possibly empty).
    leaf_overrides: Vec<LeafOverrideJson>,
}

/// Batch halving outcome (owned, detach-safe): selection, per-candidate
/// means/visits in sorted-candidate order, final survivors in cut order,
/// canon stats digest, and counters.
#[derive(Debug, Clone, PartialEq)]
pub struct HalvingOut {
    /// Selected action id (max `g + q`, `1e-12` eps + tie arm).
    pub selected_id: u32,
    /// Candidate ids in sorted order.
    pub candidate_ids: Vec<u32>,
    /// Mean 4-vectors per candidate (unvisited carry the synth model
    /// placeholder over the `unvisited:{aid}` world).
    pub value_vectors: Vec<[f64; 4]>,
    /// Visits per candidate.
    pub visits: Vec<u64>,
    /// Final survivors in cut (ranked) order.
    pub survivors: Vec<u32>,
    /// Canon digest over the per-arm stats dump.
    pub stats_digest: String,
    /// Completed rollouts.
    pub sims_run: u64,
    /// Completed transitions.
    pub transitions: u64,
    /// Model leaf calls (terminal fallbacks excluded).
    pub model_calls: u64,
    /// CTR floats consumed.
    pub floats_used: u64,
}

/// Transitions-cap gate (oracle parity: `Some` caps break, `None` never).
fn capped(transitions: u64, cap: Option<u64>) -> bool {
    match cap {
        Some(limit) => transitions >= limit,
        None => false,
    }
}

/// Look up the Gumbel for one action (validated to cover every candidate).
fn gumbel_for(gumbels: &[(u32, f64)], action: u32) -> Result<f64, SearchError> {
    let mut i = 0;
    while i < gumbels.len() {
        if gumbels[i].0 == action {
            return Ok(gumbels[i].1);
        }
        i += 1;
    }
    Err(SearchError::InvalidArg { detail: "gumbel missing action" })
}

/// Compare two actions under the frozen tie arm (byte-ordered digests for
/// the hash arms — same order as the hexdigest string compare, never
/// `hash()`).
fn tie_wins(candidate: u32, best: u32, tie: TieBreak) -> bool {
    match tie {
        TieBreak::LowestActionId => candidate < best,
        TieBreak::StableHash | TieBreak::Lexicographic => {
            let left = Sha256::digest(candidate.to_string().as_bytes());
            let right = Sha256::digest(best.to_string().as_bytes());
            left.as_slice() < right.as_slice()
        }
    }
}

/// Sort comparator for the budget-starved fallback cut
/// (`gumbel_search.py:521-528`): descending EXACT score (`==` ties,
/// so `+0.0`/`-0.0` tie exactly like the Python sort key), ties broken by
/// the tie arm. Scores are never NaN (finite Gumbels, finite-or-`-inf` q).
fn fallback_cmp(left: (f64, u32), right: (f64, u32), tie: TieBreak) -> Ordering {
    if left.0 == right.0 {
        if left.1 == right.1 {
            return Ordering::Equal;
        }
        if tie_wins(left.1, right.1, tie) {
            return Ordering::Less;
        }
        return Ordering::Greater;
    }
    if left.0 > right.0 {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// Budget-starved cut (`gumbel_search.py:510-534` else-branch): score every
/// survivor by `g + q` (`-inf` when unvisited), stable-sort by
/// (`-score`, tie arm), keep `ceil(n/2)`.
fn fallback_cut(
    survivors: &[u32],
    visits: &[u64],
    sums: &[[f64; 4]],
    candidates: &[u32],
    gumbels: &[(u32, f64)],
    root_seat: u32,
    tie: TieBreak,
) -> Result<Vec<u32>, SearchError> {
    if survivors.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    if survivors.len() == 1 {
        return Ok(survivors.to_vec());
    }
    let mean_of = |action: u32| -> Result<f64, SearchError> {
        let mut i = 0;
        while i < candidates.len() {
            if candidates[i] == action {
                if visits[i] == 0 {
                    return Ok(f64::NEG_INFINITY);
                }
                let q = sums[i][root_seat as usize] / visits[i] as f64;
                if !q.is_finite() {
                    return Err(SearchError::NonFinite { context: "gumbel fallback q" });
                }
                return Ok(q);
            }
            i += 1;
        }
        Err(SearchError::InvalidArg { detail: "gumbel survivor not a candidate" })
    };
    let mut scored: Vec<(f64, u32)> = Vec::with_capacity(survivors.len());
    let mut s = 0;
    while s < survivors.len() {
        let aid = survivors[s];
        let score = gumbel_for(gumbels, aid)? + mean_of(aid)?;
        scored.push((score, aid));
        s += 1;
    }
    scored.sort_by(|left, right| fallback_cmp(*left, *right, tie));
    let keep = survivors.len().div_ceil(2);
    scored.truncate(keep);
    let mut out = Vec::with_capacity(scored.len());
    let mut k = 0;
    while k < scored.len() {
        out.push(scored[k].1);
        k += 1;
    }
    Ok(out)
}

/// Budget-starved final pick (`gumbel_search.py:575-599` else-branch):
/// eps-gated scan in survivor order (the stored best score is KEPT on
/// tie-wins, exactly like the oracle), defaulting to the max-Gumbel
/// survivor (first maximal wins, like Python `max`).
fn fallback_select(
    survivors: &[u32],
    visits: &[u64],
    sums: &[[f64; 4]],
    candidates: &[u32],
    gumbels: &[(u32, f64)],
    root_seat: u32,
    tie: TieBreak,
) -> Result<u32, SearchError> {
    if survivors.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mean_of = |action: u32| -> Result<f64, SearchError> {
        let mut i = 0;
        while i < candidates.len() {
            if candidates[i] == action {
                if visits[i] == 0 {
                    return Ok(f64::NEG_INFINITY);
                }
                let q = sums[i][root_seat as usize] / visits[i] as f64;
                if !q.is_finite() {
                    return Err(SearchError::NonFinite { context: "gumbel final q" });
                }
                return Ok(q);
            }
            i += 1;
        }
        Err(SearchError::InvalidArg { detail: "gumbel survivor not a candidate" })
    };
    let mut best: Option<u32> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut s = 0;
    while s < survivors.len() {
        let aid = survivors[s];
        let score = gumbel_for(gumbels, aid)? + mean_of(aid)?;
        match best {
            None => {
                if score > best_score + GUMBEL_FINAL_EPS {
                    best = Some(aid);
                    best_score = score;
                }
            }
            Some(current) => {
                if score > best_score + GUMBEL_FINAL_EPS {
                    best = Some(aid);
                    best_score = score;
                } else if (score - best_score).abs() <= GUMBEL_FINAL_EPS
                    && tie_wins(aid, current, tie)
                {
                    best = Some(aid);
                }
            }
        }
        s += 1;
    }
    match best {
        Some(aid) => Ok(aid),
        None => {
            // No finite score (nothing visited): max Gumbel, first wins.
            let mut top = survivors[0];
            let mut top_g = gumbel_for(gumbels, top)?;
            let mut i = 1;
            while i < survivors.len() {
                let g = gumbel_for(gumbels, survivors[i])?;
                if g > top_g {
                    top = survivors[i];
                    top_g = g;
                }
                i += 1;
            }
            Ok(top)
        }
    }
}

/// Synth unvisited placeholder (`gumbel_search.py:218-271` synth path +
/// `608-615`): hash the `unvisited:{aid}` tag into hands/live exactly like
/// `_world_for_particle`, bind the world id through the shared canon site,
/// and read the model stub over it. `turn` rides `root_seat` (the oracle
/// reads `epoch.root_actor`, which is the root seat whenever the epoch is
/// a real `BeliefEpoch`).
fn synth_placeholder(
    action: u32,
    root_seat: u32,
    rules_hash: &str,
    observation_hash: &str,
    candidate_id: &str,
) -> Result<[f64; 4], SearchError> {
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    let tag = format!("unvisited:{action}");
    let digest = Sha256::digest(tag.as_bytes());
    let byte = |i: usize| -> u32 { u32::from(digest[i]) % 136 };
    let mut hands = [[0u32; 2]; 4];
    let mut seat = 0;
    while seat < 4 {
        let mut t0 = byte(seat * 2);
        let mut t1 = byte(seat * 2 + 1);
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
        }
        if t0 == t1 {
            t1 = (t1 + 1) % 136;
            if t0 > t1 {
                core::mem::swap(&mut t0, &mut t1);
            }
        }
        hands[seat][0] = t0;
        hands[seat][1] = t1;
        seat += 1;
    }
    let live = vec![byte(8), byte(9), byte(10), byte(11)];
    let snapshot = format!("gumbel_synth:{tag}");
    let world_id = world_id_for(
        &hands,
        &live,
        &[],
        Some(0),
        Some(root_seat),
        None,
        None,
        rules_hash,
        observation_hash,
        &snapshot,
    )?;
    model_vector(&world_id, candidate_id)
}

/// Canon stats digest over the per-arm dump (key-ordered aids, never
/// `hash()`), for golden debug.
fn stats_digest(stats: &[(u32, u64, [f64; 4])]) -> Result<String, SearchError> {
    let mut doc = BTreeMap::new();
    let mut i = 0;
    while i < stats.len() {
        let (action, visits, sum) = stats[i];
        let mut entry = BTreeMap::new();
        entry.insert("visits".to_string(), serde_json::json!(visits));
        entry.insert(
            "value_sum".to_string(),
            serde_json::json!([sum[0], sum[1], sum[2], sum[3]]),
        );
        doc.insert(action.to_string(), serde_json::Value::Object(entry.into_iter().collect()));
        i += 1;
    }
    let bytes = hydra_feed::canon::canonical_bytes(&doc, "search:gumbel_stats")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Run one batch Gumbel halving search: pure replay of precomputed worlds,
/// per-rollout policy directions, and CTR floats through forced-root
/// rollouts, continuation tilt sampling, unified `gumbel` transitions, and
/// deduped leaf vectors, with vector backup per arm (`gumbel_search.py`
/// `_rollout` + visit loops), then sequential-halving cuts and Gumbel
/// selection.
pub fn gumbel_halving_batch(batch_json: &[u8]) -> Result<HalvingOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg { detail: "batch JSON must be non-empty" });
    }
    let batch: HalvingBatchJson = serde_json::from_slice(batch_json)
        .map_err(|_| SearchError::InvalidArg { detail: "batch JSON malformed" })?;
    let sorted = check_legal(&batch.root_legal)?;
    if batch.root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: batch.root_seat });
    }
    let cont_legal = check_legal(&batch.continuation_legal)?;
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "candidate_id must be non-empty" });
    }
    if batch.domain != DOMAIN_GUMBEL {
        return Err(SearchError::InvalidArg { detail: "domain must be gumbel" });
    }
    let tie = TieBreak::parse(&batch.tie_break)?;
    let params = GumbelParams {
        halving_rounds: batch.halving_rounds,
        visits_per_round: batch.visits_per_round.clone(),
        max_depth: batch.max_depth,
        max_model_calls: batch.max_model_calls,
        max_transitions: batch.max_transitions,
        tie,
    };
    params.validate()?;
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    // Gumbels must cover exactly the sorted candidates, all finite.
    if batch.gumbels.len() != sorted.len() {
        return Err(SearchError::InvalidArg { detail: "gumbels must cover root_legal" });
    }
    let mut g = 0;
    while g < sorted.len() {
        let aid = sorted[g];
        let mut found = false;
        let mut k = 0;
        while k < batch.gumbels.len() {
            if batch.gumbels[k].0 == aid {
                if found {
                    return Err(SearchError::InvalidArg {
                        detail: "duplicate gumbel action",
                    });
                }
                if !batch.gumbels[k].1.is_finite() {
                    return Err(SearchError::NonFinite { context: "gumbel root draw" });
                }
                found = true;
            }
            k += 1;
        }
        if !found {
            return Err(SearchError::InvalidArg { detail: "gumbels must cover root_legal" });
        }
        g += 1;
    }
    // Deterministic schedule: slot counts halve per round (`M/2^r`
    // ceiling); a round executes iff it opens with more than one survivor
    // (the oracle `len(survivors) <= 1` break, `gumbel_search.py:421`), so
    // single-slot rounds consume zero worlds.
    let rounds = batch.halving_rounds as usize;
    let mut slots: Vec<usize> = Vec::with_capacity(rounds);
    let mut width = sorted.len();
    let mut r = 0;
    while r < rounds {
        slots.push(width);
        width = width.div_ceil(2);
        r += 1;
    }
    let mut total: usize = 0;
    r = 0;
    while r < rounds {
        if slots[r] > 1 {
            let visits = batch.visits_per_round[r] as usize;
            total = total
                .checked_add(slots[r].checked_mul(visits).ok_or(SearchError::InvalidArg {
                    detail: "batch dims overflow",
                })?)
                .ok_or(SearchError::InvalidArg { detail: "batch dims overflow" })?;
        }
        r += 1;
    }
    // Worlds ride the EXECUTED prefix of the schedule in oracle visit order
    // (round-major, slot-minor, visit-minor): budget truncation stops the
    // oracle sampling mid-schedule, so the envelope holds exactly the
    // sampled worlds — never more than the schedule total. Trailing surplus
    // fails closed at the end of replay (`world_ptr` must match).
    if batch.worlds.len() > total {
        return Err(SearchError::InvalidArg { detail: "worlds len exceeds schedule total" });
    }
    let executed = batch.worlds.len();
    // Policy directions: one row per executed rollout, `max_depth` cols.
    if batch.policy_dirs.len() != executed {
        return Err(SearchError::InvalidArg { detail: "policy_dirs rows must match worlds" });
    }
    let depth = batch.max_depth as usize;
    let mut d = 0;
    while d < executed {
        if batch.policy_dirs[d].len() != depth {
            return Err(SearchError::InvalidArg {
                detail: "policy_dirs cols must equal max_depth",
            });
        }
        let mut k = 0;
        while k < depth {
            if batch.policy_dirs[d][k] > 1 {
                return Err(SearchError::InvalidArg { detail: "direction must be 0|1" });
            }
            k += 1;
        }
        d += 1;
    }
    for value in &batch.rng_floats {
        if !value.is_finite() || *value < 0.0 || *value >= 1.0 {
            return Err(SearchError::InvalidArg { detail: "rng float must be in [0,1)" });
        }
    }
    // Leaf overrides: validated once, looked up by world id (read-only).
    let mut overrides: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let mut o = 0;
    while o < batch.leaf_overrides.len() {
        let entry = &batch.leaf_overrides[o];
        check_digest_shape(&entry.world_id)?;
        if entry.vector.len() != 4 {
            return Err(SearchError::InvalidArg { detail: "override vector must hold 4" });
        }
        let mut vector = [0.0f64; 4];
        let mut k = 0;
        while k < 4 {
            let value = entry.vector[k];
            if !value.is_finite() {
                return Err(SearchError::NonFinite { context: "gumbel override vector" });
            }
            vector[k] = value;
            k += 1;
        }
        if overrides.contains_key(&entry.world_id) {
            return Err(SearchError::InvalidArg { detail: "duplicate leaf override" });
        }
        overrides.insert(entry.world_id.clone(), vector);
        o += 1;
    }
    // Materialize worlds with shared passthrough hashes.
    let mut worlds: Vec<TinyWorld> = Vec::with_capacity(executed);
    let mut w = 0;
    while w < executed {
        let mut world = batch.worlds[w].materialize()?;
        world.rules_hash = batch.rules_hash.clone();
        world.observation_hash = batch.observation_hash.clone();
        world.validate()?;
        worlds.push(world);
        w += 1;
    }
    // Exact float demand: dry-run the consumption walk over the materialized
    // worlds, mirroring the replay loop below step-for-step — same explicit
    // `step = 1` start, same `step < max_depth && live` stop, same
    // sample-then-gate order (a breaking read counts in `need`), same
    // live-deplete advance, same post-cap stop. The per-read `exhausted`
    // check inside replay stays the ultimate backstop.
    let mut need: usize = 0;
    let mut walked: u64 = 0;
    let mut ord = 0usize;
    let mut dr = 0;
    let mut dry_done = false;
    while dr < rounds && !dry_done {
        if slots[dr] <= 1 {
            break;
        }
        let visits = batch.visits_per_round[dr] as usize;
        let mut ds = 0;
        while ds < slots[dr] && !dry_done {
            let mut dv = 0;
            while dv < visits {
                if capped(walked, batch.max_transitions) {
                    dry_done = true;
                    break;
                }
                let start = worlds.get(ord).ok_or(SearchError::InvalidArg {
                    detail: "world schedule exhausted",
                })?;
                ord += 1;
                walked = walked.checked_add(1).ok_or(SearchError::InvalidArg {
                    detail: "transition counter overflow",
                })?;
                let mut step: u32 = 1;
                let mut live = start.live.len();
                while (step as usize) < depth && live > 0 {
                    need = need.checked_add(1).ok_or(SearchError::InvalidArg {
                        detail: "batch dims overflow",
                    })?;
                    if capped(walked, batch.max_transitions) {
                        break;
                    }
                    live -= 1;
                    step = step.checked_add(1).ok_or(SearchError::InvalidArg {
                        detail: "step overflow",
                    })?;
                    walked = walked.checked_add(1).ok_or(SearchError::InvalidArg {
                        detail: "transition counter overflow",
                    })?;
                    if capped(walked, batch.max_transitions) {
                        break;
                    }
                }
                dv += 1;
            }
            ds += 1;
        }
        dr += 1;
    }
    if batch.rng_floats.len() < need {
        return Err(SearchError::InvalidArg { detail: "rng_floats too short" });
    }
    // Per-arm stats in sorted-candidate order: `(aid, visits, value_sum)`.
    let mut arm_visits: Vec<u64> = vec![0; sorted.len()];
    let mut arm_sums: Vec<[f64; 4]> = vec![[0.0; 4]; sorted.len()];
    let arm_pos = |aid: u32, candidates: &[u32]| -> Result<usize, SearchError> {
        let mut i = 0;
        while i < candidates.len() {
            if candidates[i] == aid {
                return Ok(i);
            }
            i += 1;
        }
        Err(SearchError::InvalidArg { detail: "gumbel survivor not a candidate" })
    };
    let mut survivors = sorted.clone();
    let mut transitions: u64 = 0;
    let mut model_calls: u64 = 0;
    let mut sims_run: u64 = 0;
    let mut float_ptr: usize = 0;
    let mut world_ptr: usize = 0;
    let mut rr = 0;
    while rr < rounds {
        if survivors.len() <= 1 {
            break;
        }
        let visits = batch.visits_per_round[rr] as usize;
        let mut s = 0;
        while s < survivors.len() {
            let aid = survivors[s];
            let pos = arm_pos(aid, &sorted)?;
            let mut t = 0;
            while t < visits {
                // Budget gate before the rollout (oracle parity: the
                // model-call pre-check is a no-op — see module docs).
                if capped(transitions, batch.max_transitions) {
                    break;
                }
                let start = worlds.get(world_ptr).ok_or(SearchError::InvalidArg {
                    detail: "world schedule exhausted",
                })?;
                world_ptr += 1;
                // Forced root action: unconditional transition, then the
                // explicit `step = 1` counter (never the world latent).
                let actor0 = start.actor_to_move();
                let mut current = exact_transition(start, actor0, aid, DOMAIN_GUMBEL)?;
                transitions = transitions.checked_add(1).ok_or(SearchError::InvalidArg {
                    detail: "transition counter overflow",
                })?;
                let mut step: u32 = 1;
                let row = world_ptr - 1;
                let mut cstep: usize = 0;
                while (step as usize) < depth && !current.live.is_empty() {
                    let actor = current.actor_to_move();
                    let dir_row = batch.policy_dirs.get(row).ok_or(SearchError::InvalidArg {
                        detail: "policy_dirs exhausted",
                    })?;
                    let direction = *dir_row.get(cstep).ok_or(SearchError::InvalidArg {
                        detail: "policy_dirs exhausted",
                    })?;
                    let float = *batch.rng_floats.get(float_ptr).ok_or(
                        SearchError::InvalidArg { detail: "rng_floats exhausted" },
                    )?;
                    float_ptr += 1;
                    let action = continuation_sample(&cont_legal, direction, float)?;
                    // Sample-then-gate: the float above counts even when the
                    // cap breaks here with no transition.
                    if capped(transitions, batch.max_transitions) {
                        break;
                    }
                    current = exact_transition(&current, actor, action, DOMAIN_GUMBEL)?;
                    transitions =
                        transitions.checked_add(1).ok_or(SearchError::InvalidArg {
                            detail: "transition counter overflow",
                        })?;
                    step = step.checked_add(1).ok_or(SearchError::InvalidArg {
                        detail: "step overflow",
                    })?;
                    cstep += 1;
                    if capped(transitions, batch.max_transitions) {
                        break;
                    }
                }
                // Leaf: terminal when stopped terminal; else model unless the
                // call budget is exhausted (then terminal fallback).
                // Overrides win when the leaf world id is pinned.
                let leaf_terminal = (step as usize) >= depth || current.live.is_empty();
                let allow_model = match batch.max_model_calls {
                    Some(cap) => model_calls < cap,
                    None => true,
                };
                let vector = match overrides.get(&current.world_id) {
                    Some(vector) => *vector,
                    None => {
                        if leaf_terminal || !allow_model {
                            terminal_vector(&current.world_id)?
                        } else {
                            let vector = model_vector(&current.world_id, &batch.candidate_id)?;
                            model_calls =
                                model_calls.checked_add(1).ok_or(SearchError::InvalidArg {
                                    detail: "model counter overflow",
                                })?;
                            vector
                        }
                    }
                };
                let mut f = 0;
                while f < 4 {
                    if !vector[f].is_finite() {
                        return Err(SearchError::NonFinite { context: "gumbel backup vector" });
                    }
                    f += 1;
                }
                arm_visits[pos] = arm_visits[pos].checked_add(1).ok_or(
                    SearchError::InvalidArg { detail: "visit counter overflow" },
                )?;
                let mut k = 0;
                while k < 4 {
                    arm_sums[pos][k] += vector[k];
                    k += 1;
                }
                sims_run =
                    sims_run.checked_add(1).ok_or(SearchError::InvalidArg {
                        detail: "sim counter overflow",
                    })?;
                // Post-visit budget: transitions stop the search (oracle
                // parity); model exhaustion never stops it.
                if capped(transitions, batch.max_transitions) {
                    break;
                }
                t += 1;
            }
            if capped(transitions, batch.max_transitions) {
                break;
            }
            s += 1;
        }
        // Sequential-halving cut over this round's survivors.
        let mut complete = true;
        let mut c = 0;
        while c < survivors.len() {
            let pos = arm_pos(survivors[c], &sorted)?;
            if arm_visits[pos] == 0 {
                complete = false;
                break;
            }
            c += 1;
        }
        if complete {
            let mut means: Vec<(u32, f64)> = Vec::with_capacity(survivors.len());
            let mut m = 0;
            while m < survivors.len() {
                let pos = arm_pos(survivors[m], &sorted)?;
                let q = arm_sums[pos][batch.root_seat as usize] / arm_visits[pos] as f64;
                if !q.is_finite() {
                    return Err(SearchError::NonFinite { context: "gumbel cut q" });
                }
                means.push((survivors[m], q));
                m += 1;
            }
            survivors = halving_cut(&survivors, &means, &batch.gumbels, tie)?;
        } else {
            survivors = fallback_cut(
                &survivors,
                &arm_visits,
                &arm_sums,
                &sorted,
                &batch.gumbels,
                batch.root_seat,
                tie,
            )?;
        }
        rr += 1;
    }
    // No trailing surplus: every supplied world must be consumed in schedule
    // order (an over-supplied tail means the precompute walk forked).
    if world_ptr != executed {
        return Err(SearchError::InvalidArg { detail: "world schedule miscount" });
    }
    // Final selection over the last survivors.
    let mut final_complete = true;
    let mut c = 0;
    while c < survivors.len() {
        let pos = arm_pos(survivors[c], &sorted)?;
        if arm_visits[pos] == 0 {
            final_complete = false;
            break;
        }
        c += 1;
    }
    let selected_id = if final_complete {
        let mut means: Vec<(u32, f64)> = Vec::with_capacity(survivors.len());
        let mut m = 0;
        while m < survivors.len() {
            let pos = arm_pos(survivors[m], &sorted)?;
            let q = arm_sums[pos][batch.root_seat as usize] / arm_visits[pos] as f64;
            if !q.is_finite() {
                return Err(SearchError::NonFinite { context: "gumbel final q" });
            }
            means.push((survivors[m], q));
            m += 1;
        }
        gumbel_select(&survivors, &means, &batch.gumbels, tie)?
    } else {
        fallback_select(
            &survivors,
            &arm_visits,
            &arm_sums,
            &sorted,
            &batch.gumbels,
            batch.root_seat,
            tie,
        )?
    };
    // Value vectors in sorted-candidate order (unvisited arms carry the
    // synth `unvisited:{aid}` model placeholder).
    let mut value_vectors: Vec<[f64; 4]> = Vec::with_capacity(sorted.len());
    let mut v = 0;
    while v < sorted.len() {
        if arm_visits[v] > 0 {
            let n = arm_visits[v] as f64;
            let mean = [
                arm_sums[v][0] / n,
                arm_sums[v][1] / n,
                arm_sums[v][2] / n,
                arm_sums[v][3] / n,
            ];
            let mut k = 0;
            while k < 4 {
                if !mean[k].is_finite() {
                    return Err(SearchError::NonFinite { context: "gumbel final mean" });
                }
                k += 1;
            }
            value_vectors.push(mean);
        } else {
            value_vectors.push(synth_placeholder(
                sorted[v],
                batch.root_seat,
                &batch.rules_hash,
                &batch.observation_hash,
                &batch.candidate_id,
            )?);
        }
        v += 1;
    }
    let mut dump: Vec<(u32, u64, [f64; 4])> = Vec::with_capacity(sorted.len());
    let mut q = 0;
    while q < sorted.len() {
        dump.push((sorted[q], arm_visits[q], arm_sums[q]));
        q += 1;
    }
    let digest = stats_digest(&dump)?;
    let floats_used = float_ptr as u64;
    Ok(HalvingOut {
        selected_id,
        candidate_ids: sorted,
        value_vectors,
        visits: arm_visits,
        survivors,
        stats_digest: digest,
        sims_run,
        transitions,
        model_calls,
        floats_used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OBS: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn world_json(wid: &str, live: &[u32]) -> serde_json::Value {
        serde_json::json!({
            "world_id": wid,
            "hands": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "live": live,
            "dead": [],
            "step": null,
            "turn": null,
            "corpus_idx": 0,
            "snapshot": "tiny:test:0",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn envelope(
        worlds: Vec<serde_json::Value>,
        dirs: Vec<Vec<u32>>,
        floats: Vec<f64>,
        root_legal: Vec<u32>,
        gumbels: Vec<(u32, f64)>,
        rounds: u32,
        visits: Vec<u32>,
        max_depth: u32,
        max_transitions: Option<u64>,
        max_model_calls: Option<u64>,
    ) -> Vec<u8> {
        let gumbel_pairs: Vec<serde_json::Value> =
            gumbels.iter().map(|(a, g)| serde_json::json!([a, g])).collect();
        serde_json::json!({
            "worlds": worlds,
            "rules_hash": RULES,
            "observation_hash": OBS,
            "root_legal": root_legal,
            "root_seat": 0,
            "gumbels": gumbel_pairs,
            "halving_rounds": rounds,
            "visits_per_round": visits,
            "continuation_legal": [0, 2],
            "policy_dirs": dirs,
            "rng_floats": floats,
            "max_depth": max_depth,
            "max_transitions": max_transitions,
            "max_model_calls": max_model_calls,
            "candidate_id": "candidate6",
            "domain": "gumbel",
            "tie_break": "lowest_action_id",
            "leaf_overrides": [],
        })
        .to_string()
        .into_bytes()
    }

    fn digest(id: &str) -> String {
        format!("sha256:{id}")
    }

    #[test]
    fn rejects_empty_batch() {
        let err = gumbel_halving_batch(&[]).unwrap_err();
        assert_eq!(err, SearchError::InvalidArg { detail: "batch JSON must be non-empty" });
    }

    #[test]
    fn rejects_malformed_json() {
        let err = gumbel_halving_batch(b"{nope").unwrap_err();
        assert_eq!(err, SearchError::InvalidArg { detail: "batch JSON malformed" });
    }

    #[test]
    fn rejects_bad_rounds_and_visits() {
        let wid = digest(&"a".repeat(64));
        let worlds = vec![world_json(&wid, &[8, 9])];
        // halving_rounds = 0.
        let batch = envelope(worlds.clone(), vec![vec![0]], vec![], vec![7], vec![(7, 0.5)], 0, vec![], 2, None, None);
        assert!(gumbel_halving_batch(&batch).is_err());
        // visits len mismatch.
        let batch = envelope(worlds.clone(), vec![vec![0]], vec![], vec![7], vec![(7, 0.5)], 1, vec![1, 1], 2, None, None);
        assert!(gumbel_halving_batch(&batch).is_err());
        // wrong domain.
        let mut raw: serde_json::Value = serde_json::from_slice(&envelope(
            worlds.clone(),
            vec![vec![0]],
            vec![],
            vec![7],
            vec![(7, 0.5)],
            1,
            vec![1],
            2,
            None,
            None,
        ))
        .unwrap();
        raw["domain"] = serde_json::json!("ismcts");
        let err = gumbel_halving_batch(&raw.to_string().into_bytes()).unwrap_err();
        assert_eq!(err, SearchError::InvalidArg { detail: "domain must be gumbel" });
        // duplicate legal.
        let batch = envelope(worlds, vec![vec![0]], vec![], vec![7, 7], vec![(7, 0.5)], 1, vec![1], 2, None, None);
        assert!(gumbel_halving_batch(&batch).is_err());
    }

    #[test]
    fn rejects_gumbel_coverage_gaps() {
        let wid = digest(&"a".repeat(64));
        let worlds = vec![world_json(&wid, &[8, 9]), world_json(&wid, &[8, 9])];
        // Missing arm 9.
        let batch = envelope(worlds.clone(), vec![vec![0], vec![0]], vec![], vec![7, 9], vec![(7, 0.5)], 1, vec![1], 2, None, None);
        assert!(gumbel_halving_batch(&batch).is_err());
        // Duplicate draw for arm 7 (JSON cannot carry inf, so the
        // non-finite gate stays defense-in-depth behind the parser).
        let batch = envelope(worlds, vec![vec![0], vec![0]], vec![], vec![7, 9], vec![(7, 0.5), (7, 0.6)], 1, vec![1], 2, None, None);
        let err = gumbel_halving_batch(&batch).unwrap_err();
        assert_eq!(err, SearchError::InvalidArg { detail: "duplicate gumbel action" });
    }

    #[test]
    fn single_candidate_needs_no_rollouts() {
        let batch = envelope(vec![], vec![], vec![], vec![7], vec![(7, 0.5)], 2, vec![8, 8], 6, Some(64), Some(32));
        let out = gumbel_halving_batch(&batch).unwrap();
        assert_eq!(out.selected_id, 7);
        assert_eq!(out.candidate_ids, vec![7]);
        assert_eq!(out.visits, vec![0]);
        assert_eq!(out.survivors, vec![7]);
        assert_eq!(out.sims_run, 0);
        assert_eq!(out.transitions, 0);
        assert_eq!(out.model_calls, 0);
        assert_eq!(out.floats_used, 0);
        // Unvisited arm carries the finite synth placeholder.
        assert_eq!(out.value_vectors.len(), 1);
        for value in out.value_vectors[0] {
            assert!(value.is_finite());
        }
        assert!(out.stats_digest.starts_with("sha256:"));
    }

    #[test]
    fn tiny_two_arm_shape_runs_end_to_end() {
        // M=2, rounds=1, visits=1, depth=1: two forced-root rollouts, no
        // continuation steps (step=1 !< 1), no floats needed.
        let w0 = digest(&"a".repeat(64));
        let w1 = digest(&"c".repeat(64));
        let worlds = vec![world_json(&w0, &[8, 9, 10, 11]), world_json(&w1, &[8, 9, 10, 11])];
        let batch = envelope(
            worlds,
            vec![vec![0], vec![0]],
            vec![],
            vec![7, 9],
            vec![(7, 0.1), (9, 5.0)],
            1,
            vec![1],
            1,
            None,
            None,
        );
        let out = gumbel_halving_batch(&batch).unwrap();
        assert_eq!(out.sims_run, 2);
        assert_eq!(out.transitions, 2);
        // Depth 1 stops terminal after the forced root: terminal leaves,
        // zero model calls (oracle `315-316` leaf rule).
        assert_eq!(out.model_calls, 0);
        assert_eq!(out.visits, vec![1, 1]);
        assert_eq!(out.survivors.len(), 1);
        // Arm 9 dominates (g=5.0): survives the cut and wins final.
        assert_eq!(out.survivors, vec![9]);
        assert_eq!(out.selected_id, 9);
    }

    #[test]
    fn transitions_cap_starves_late_arms_into_fallback() {
        // Cap 1: the first rollout runs, the rest break pre-rollout. The
        // envelope holds exactly the executed prefix (one world); the cut
        // and final ride the fallback paths (arm 9 unvisited).
        let w0 = digest(&"a".repeat(64));
        let worlds = vec![world_json(&w0, &[8, 9, 10, 11])];
        let batch = envelope(
            worlds,
            vec![vec![0]],
            vec![],
            vec![7, 9],
            vec![(7, 0.1), (9, 5.0)],
            1,
            vec![1],
            1,
            Some(1),
            None,
        );
        let out = gumbel_halving_batch(&batch).unwrap();
        assert_eq!(out.sims_run, 1);
        assert_eq!(out.transitions, 1);
        assert_eq!(out.visits, vec![1, 0]);
        // Fallback cut: visited arm 7 (q+g) vs unvisited arm 9 (-inf) keeps
        // arm 7; the single survivor then wins the complete final outright.
        assert_eq!(out.survivors, vec![7]);
        assert_eq!(out.selected_id, 7);
        // Unvisited arm 9 carries the synth placeholder, not zeros.
        assert_ne!(out.value_vectors[1], [0.0, 0.0, 0.0, 0.0]);
        for value in out.value_vectors[1] {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn synth_placeholder_matches_oracle_pins() {
        // Oracle pins via `GumbelSearchPlannerSearchMixin._world_for_particle`
        // (belief/epoch `None`) + `model_vector_for_world` (`candidate6`):
        // aid 7 -> (0.87, 0.76, 0.92, 0.72), aid 9 -> (0.5, 0.23, 0.12, 0.55).
        // Catches derivation forks (e.g. a dropped hand tile reads a
        // different world id and a different vector).
        assert_eq!(
            synth_placeholder(7, 0, RULES, OBS, "candidate6").unwrap(),
            [0.87, 0.76, 0.92, 0.72]
        );
        assert_eq!(
            synth_placeholder(9, 0, RULES, OBS, "candidate6").unwrap(),
            [0.5, 0.23, 0.12, 0.55]
        );
    }

    #[test]
    fn fallback_cut_orders_exact_scores_with_tie_arm() {
        // scores: arm 3 -> 1.0, arm 5 -> 1.0 (tie -> lower id first),
        // arm 9 -> -inf (last). Keep ceil(3/2) = 2.
        let candidates = vec![3, 5, 9];
        let survivors = vec![9, 5, 3];
        let visits = vec![2, 2, 0];
        let sums = vec![[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0; 4]];
        let gumbels = vec![(3, 0.0), (5, 0.0), (9, 100.0)];
        let cut =
            fallback_cut(&survivors, &visits, &sums, &candidates, &gumbels, 0, TieBreak::LowestActionId)
                .unwrap();
        assert_eq!(cut, vec![3, 5]);
    }

    #[test]
    fn fallback_select_prefers_max_gumbel_when_nothing_visited() {
        let survivors = vec![7, 9];
        let visits = vec![0, 0];
        let sums = vec![[0.0; 4], [0.0; 4]];
        let candidates = vec![7, 9];
        let gumbels = vec![(7, 0.1), (9, 5.0)];
        let pick = fallback_select(
            &survivors,
            &visits,
            &sums,
            &candidates,
            &gumbels,
            0,
            TieBreak::LowestActionId,
        )
        .unwrap();
        assert_eq!(pick, 9);
    }

    #[test]
    fn synth_placeholder_is_deterministic_and_tag_bound() {
        let a = synth_placeholder(7, 0, RULES, OBS, "candidate6").unwrap();
        let b = synth_placeholder(7, 0, RULES, OBS, "candidate6").unwrap();
        assert_eq!(a, b);
        let c = synth_placeholder(9, 0, RULES, OBS, "candidate6").unwrap();
        assert_ne!(a, c);
        for value in a {
            assert!(value.is_finite());
        }
    }
}
