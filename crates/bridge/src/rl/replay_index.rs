//! Minibatch order and frozen learner-slab replay: native semantic-counter
//! shuffle, one-epoch stable bucket grouping, exact tails/counts/order
//! digest, and the typed feature slab for one selected optimizer batch.
//!
//! Shuffle law (plan section 6, exact): `semantic_seed(actual master_seed,
//! training_shuffle key)` with `candidate_id = pipeline_spec_id`,
//! `experiment_id` frozen, `split_id = "selection_training:g={g}:epoch=
//! {epoch}"`, explicit replicate/attempt, every other field null
//! (`case_id`/`wall_id` are FORBIDDEN for training_shuffle by the key
//! schema); descending Fisher-Yates with the existing counter-stream
//! `random_below` rejection semantics; the Python `RandomStream` is the
//! oracle. Stable `T` grouping, 4096-cap splits, every row exactly once
//! including tails, one epoch. The order digest over ordered row-id batches
//! freezes before the RunSpec.
//!
//! Export law: after full artifact validation, the compact tape plus the
//! frozen replay handle are the acceleration source — no per-batch artifact
//! read or row re-parse. Features come from the registered pure encoder
//! over borrowed typed row/event-prefix facts; frozen LP/A/G4 ride from the
//! index facts. The learner consumes stored `A`; nothing is recomputed on
//! admission. The cold owned-buffer export stays the tooling oracle on the
//! identical law the production direct fill serves.
//!
//! Layout: the order/shuffle law and frozen plan types live here; the
//! process-once replay registry lives in `registry` and the shared
//! slab-staging law (cold oracle plus hot direct fill) in `fill`. External
//! crate paths stay flat through the re-exports below, so the artifact
//! surface keeps its contract without edits.

mod fill;
mod registry;

pub use fill::{
    LearnerSlab, SlabSpec, export_learner_slab, fill_live_slab_buffers, fill_slab_buffers,
    plane_table, slab_geometry, slab_spec_bytes, slab_spec_bytes_of, stage_batch,
};
pub use registry::{
    FrozenBatch, FrozenMetaRow, FrozenReplay, ReplayHandle, ReplaySource, drop_replay,
    register_replay_from_bytes, with_replay,
};

use serde_json::{Map, Value};

use crate::rl::artifact::parse_metas;
use crate::rl::trajectory::{ACTION_WIDTH, DIGEST_PREFIX, HISTORY_BUCKETS, TapeError, TapeHandle, bucket_history_len, hex_of, parse_digest_text, sha256_raw, tape_live_binding, with_tape};

/// Optimizer minibatch capacity: single source is the frozen recipe
/// (`config::RlFrozenRecipe::OPTIMIZER_MINIBATCH_CAP`); tails stay genuine.
pub const OPTIMIZER_CAP: usize =
    crate::rl::config::RlFrozenRecipe::OPTIMIZER_MINIBATCH_CAP as usize;
/// Feature bytes per row `S(T) = 7254 + 9*T`: the single byte law lives in
/// the encoder extraction (`observation::row_bytes_26`); this wrapper only
/// gates the bucket first so misuse fails closed with a tape error.
pub fn feature_row_bytes(t: u32) -> Result<usize, TapeError> {
    if HISTORY_BUCKETS.iter().all(|b| *b != t) {
        return Err(TapeError(format!("slab T bucket {t} not in 32/64/128/256")));
    }
    Ok(crate::rl::observation::row_bytes_26(t as usize))
}

/// Closed shuffle key (training_shuffle scope; generation/epoch ride the
/// named split scope, never the RunSpec or control tip).
pub struct ShuffleKey {
    pub master_seed_hex: String,
    pub pipeline_spec_id: String,
    pub experiment_id: String,
    pub generation: u64,
    pub epoch: u64,
    pub replicate_id: u64,
    pub attempt_id: u64,
}

/// One optimizer batch: T-homogeneous canonical positions plus true count.
pub struct BatchPlan {
    pub t: u32,
    pub positions: Vec<u32>,
}

/// Frozen order plan: batches plus identities.
pub struct OrderPlan {
    pub artifact_digest: String,
    pub order_digest: String,
    pub batches: Vec<BatchPlan>,
    pub row_count: usize,
}


/// Plane-major facts total per row-equivalent (8 + 4 + 4 + 32); the buffer
/// holds B selected, then B old-LP, then B raw-A, then B G4 vectors.
pub const FACTS48_STRIDE: usize = 48;

fn order_err(detail: String) -> TapeError {
    TapeError(detail)
}

fn decode_hex_even(text: &str) -> Result<Vec<u8>, TapeError> {
    if text.is_empty() || !text.len().is_multiple_of(2) {
        return Err(order_err("master seed hex must be nonempty even-length".to_string()));
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nybble(pair[0])
            .ok_or_else(|| order_err("master seed hex has non-hex digits".to_string()))?;
        let lo = hex_nybble(pair[1])
            .ok_or_else(|| order_err("master seed hex has non-hex digits".to_string()))?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nybble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Semantic stream seed: sha256 over the JCS payload `{"protocol":
/// "hydra2_rng_v1", "master_seed": <hex>, "key": <17-field key json>}`
/// (oracle: `randomness.py:137-165`). Key field order matches the SPEC
/// dataclass verbatim; JCS sorts on the wire regardless.
fn semantic_seed(key: &ShuffleKey) -> Result<Vec<u8>, TapeError> {
    if key.pipeline_spec_id.is_empty() || key.experiment_id.is_empty() {
        return Err(order_err(
            "shuffle key needs nonempty pipeline_spec_id and experiment_id".to_string(),
        ));
    }
    let master = decode_hex_even(&key.master_seed_hex)?;
    let split = format!(
        "selection_training:g={}:epoch={}",
        key.generation, key.epoch
    );
    let mut key_doc = Map::with_capacity(17);
    key_doc.insert("purpose".to_string(), Value::String("training_shuffle".to_string()));
    key_doc.insert(
        "experiment_id".to_string(),
        Value::String(key.experiment_id.clone()),
    );
    key_doc.insert("split_id".to_string(), Value::String(split));
    key_doc.insert(
        "candidate_id".to_string(),
        Value::String(key.pipeline_spec_id.clone()),
    );
    for null_field in [
        "case_id",
        "wall_id",
        "root_seat",
        "belief_epoch",
        "parent_id",
        "action_id",
        "packet_id",
        "fidelity_level",
        "population_id",
    ] {
        key_doc.insert(null_field.to_string(), Value::Null);
    }
    key_doc.insert(
        "replicate_id".to_string(),
        Value::from(key.replicate_id),
    );
    key_doc.insert("scramble_id".to_string(), Value::Null);
    key_doc.insert("visit_index".to_string(), Value::Null);
    key_doc.insert("attempt_id".to_string(), Value::from(key.attempt_id));
    let mut payload = Map::with_capacity(3);
    payload.insert("protocol".to_string(), Value::String("hydra2_rng_v1".to_string()));
    payload.insert(
        "master_seed".to_string(),
        Value::String(hex_of(&master)),
    );
    payload.insert("key".to_string(), Value::Object(key_doc));
    let bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(payload), "rl:shuffle-key")
        .map_err(|e| order_err(format!("shuffle key JCS emit failed: {e:?}")))?;
    Ok(sha256_raw(&bytes).to_vec())
}

/// Counter-stream window `[cursor, cursor+count)` over the legacy
/// `hydra2_ctr_v1` sha256-CTR primitive (oracle: `RandomStream.get_bytes`;
/// block law: `sha256(DOMAIN || len(seed)BE32 || seed || indexBE64)`, feed
/// `digest::ctr_block`). Byte position `p` depends only on `(seed, p/32)`.
fn ctr_window(seed: &[u8], cursor: u64, count: usize) -> Result<Vec<u8>, TapeError> {
    if seed.is_empty() {
        return Err(order_err("shuffle stream seed must be nonempty".to_string()));
    }
    const BLOCK: u64 = 32;
    cursor
        .checked_add(count as u64)
        .ok_or_else(|| order_err("shuffle stream window overflows".to_string()))?;
    let first_block = cursor / BLOCK;
    let offset = (cursor % BLOCK) as usize;
    // Only blocks intersecting this window are needed; using the absolute
    // cursor regenerates discarded suffix blocks and makes shuffling quadratic.
    let needed = offset
        .checked_add(count)
        .ok_or_else(|| order_err("shuffle stream window overflows".to_string()))?
        .div_ceil(BLOCK as usize);
    let mut acc = Vec::with_capacity(needed * BLOCK as usize);
    let mut block = first_block;
    for _ in 0..needed {
        acc.extend_from_slice(&hydra_feed::digest::ctr_block(seed, block));
        block = block
            .checked_add(1)
            .ok_or_else(|| order_err("shuffle block counter overflows".to_string()))?;
    }
    acc.get(offset..offset + count)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| order_err("shuffle window outside accumulated blocks".to_string()))
}

/// Seekable counter stream with the exact `random_below` rejection law
/// (oracle: `randomness.py:193-201`): minimal whole bytes covering the
/// bound, `limit = 256^nbytes/batch*bound` floor, BE draw, redraw while
/// `value >= limit`, result `value % bound`.
struct CtrStream {
    seed: Vec<u8>,
    cursor: u64,
}

impl CtrStream {
    fn get_bytes(&mut self, count: usize) -> Result<Vec<u8>, TapeError> {
        let out = ctr_window(&self.seed, self.cursor, count)?;
        self.cursor = self
            .cursor
            .checked_add(count as u64)
            .ok_or_else(|| order_err("shuffle cursor overflows".to_string()))?;
        Ok(out)
    }

    fn random_below(&mut self, bound: u64) -> Result<u64, TapeError> {
        if bound < 2 {
            return Err(order_err("random_below bound must be >= 2".to_string()));
        }
        let bits = 64 - (bound - 1).leading_zeros() as usize;
        let nbytes = bits / 8 + 1;
        if nbytes > 8 {
            return Err(order_err("random_below bound exceeds u64 draws".to_string()));
        }
        let base: u128 = 256;
        let limit = base.pow(nbytes as u32) / bound as u128 * bound as u128;
        loop {
            let raw = self.get_bytes(nbytes)?;
            let mut value: u128 = 0;
            for b in raw {
                value = (value << 8) | b as u128;
            }
            if value < limit {
                return Ok((value % bound as u128) as u64);
            }
        }
    }
}

/// Descending Fisher-Yates over `0..n` with counter-stream draws
/// (`j = random_below(i+1)`, swap `i <-> j` for `i = n-1 .. 1`).
fn fisher_yates(n: usize, stream: &mut CtrStream) -> Result<Vec<usize>, TapeError> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = stream.random_below(i as u64 + 1)? as usize;
        perm.swap(i, j);
    }
    Ok(perm)
}
/// Shared position planner: descending Fisher-Yates shuffle over canonical
/// positions with the closed shuffle key, stable ascending-`T` grouping,
/// and 4096-cap splitting with every row exactly once including tails, one
/// epoch. One home for the archive `build_order` and live registration; no
/// alternate shuffle or fastest-finisher ordering.
fn plan_positions(bucket_by_position: &[u32], key: &ShuffleKey) -> Result<Vec<BatchPlan>, TapeError> {
    let row_count = bucket_by_position.len();
    if row_count == 0 {
        return Err(order_err("cannot order an empty generation".to_string()));
    }
    for t in bucket_by_position {
        if HISTORY_BUCKETS.iter().all(|b| *b != *t) {
            return Err(order_err(format!("row T {t} not a frozen bucket")));
        }
    }
    let seed = semantic_seed(key)?;
    let mut stream = CtrStream { seed, cursor: 0 };
    let shuffled = fisher_yates(row_count, &mut stream)?;
    // Stable T grouping: ascending bucket, shuffled relative order within.
    let mut by_t: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for position in shuffled {
        let slot = HISTORY_BUCKETS
            .iter()
            .position(|b| *b == bucket_by_position[position])
            .ok_or_else(|| order_err("row T vanished after bucket gate".to_string()))?;
        by_t[slot].push(position);
    }
    let mut batches = Vec::new();
    for (slot, bucket) in HISTORY_BUCKETS.iter().enumerate() {
        for chunk in by_t[slot].chunks(OPTIMIZER_CAP) {
            batches.push(BatchPlan {
                t: *bucket,
                positions: chunk.iter().map(|p| *p as u32).collect(),
            });
        }
    }
    // Exact-once audit: every canonical position appears exactly once.
    let mut seen = vec![false; row_count];
    let mut covered = 0usize;
    for batch in &batches {
        if batch.positions.is_empty() || batch.positions.len() > OPTIMIZER_CAP {
            return Err(order_err("batch violates the 4096 cap".to_string()));
        }
        for position in &batch.positions {
            if *position as usize >= row_count || seen[*position as usize] {
                return Err(order_err("order drops or duplicates a row".to_string()));
            }
            seen[*position as usize] = true;
            covered += 1;
        }
    }
    if covered != row_count {
        return Err(order_err("order drops or duplicates a row".to_string()));
    }
    Ok(batches)
}

/// Build the frozen order plan from structural metas bytes (small cold JSON)
/// plus the closed shuffle key: shuffle canonical positions, stable-group
/// by ascending `T`, split into 4096-cap batches preserving every row
/// exactly once including tails, one epoch. Returns the plan plus the
/// canonical index bytes (opaque to Python, consumed by export).
pub fn build_order(metas_bytes: &[u8], key: &ShuffleKey) -> Result<(OrderPlan, Vec<u8>), TapeError> {
    let metas = parse_metas(metas_bytes)?;
    let artifact_digest = metas
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| order_err("metas lacks artifact_digest".to_string()))?
        .to_string();
    crate::rl::trajectory::parse_digest_text(&artifact_digest)?;
    let meta_rows = metas
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| order_err("metas lacks rows".to_string()))?;
    let row_count = meta_rows.len();
    if row_count == 0 {
        return Err(order_err("cannot order an empty generation".to_string()));
    }
    let mut buckets = Vec::with_capacity(row_count);
    for meta in meta_rows {
        let raw = meta
            .get("t")
            .and_then(Value::as_u64)
            .ok_or_else(|| order_err("metas row lacks t".to_string()))?;
        buckets.push(
            u32::try_from(raw).map_err(|_| order_err("metas row T outside u32".to_string()))?,
        );
    }
    let batches = plan_positions(&buckets, key)?;
    // Order digest over the ordered row-id batches (freezes before RunSpec).
    let mut batch_docs = Vec::with_capacity(batches.len());
    for batch in &batches {
        let ids: Vec<Value> = batch
            .positions
            .iter()
            .map(|position| {
                meta_rows[*position as usize]
                    .get("row_id")
                    .map_or(Value::Null, Clone::clone)
            })
            .collect();
        let mut doc = Map::with_capacity(3);
        doc.insert("t".to_string(), Value::from(u64::from(batch.t)));
        doc.insert("count".to_string(), Value::from(batch.positions.len() as u64));
        doc.insert("row_ids".to_string(), Value::Array(ids));
        batch_docs.push(Value::Object(doc));
    }
    let mut digest_doc = Map::with_capacity(2);
    digest_doc.insert(
        "artifact_digest".to_string(),
        Value::String(artifact_digest.clone()),
    );
    digest_doc.insert("batches".to_string(), Value::Array(batch_docs));
    let digest_bytes =
        hydra_feed::canon::canonical_bytes_value(&Value::Object(digest_doc), "rl:order-digest")
            .map_err(|e| order_err(format!("order digest JCS emit failed: {e:?}")))?;
    let order_digest = format!("{DIGEST_PREFIX}{}", hex_of(&sha256_raw(&digest_bytes)));

    // Canonical index bytes (structural; export parses them back).
    let mut index_batches = Vec::with_capacity(batches.len());
    for batch in &batches {
        let mut doc = Map::with_capacity(3);
        doc.insert("t".to_string(), Value::from(u64::from(batch.t)));
        doc.insert("count".to_string(), Value::from(batch.positions.len() as u64));
        doc.insert(
            "positions".to_string(),
            Value::Array(
                batch
                    .positions
                    .iter()
                    .map(|p| Value::from(u64::from(*p)))
                    .collect(),
            ),
        );
        index_batches.push(Value::Object(doc));
    }
    let mut index_doc = Map::with_capacity(6);
    index_doc.insert(
        "schema".to_string(),
        Value::String("hydra2.rl_order_plan_v1".to_string()),
    );
    index_doc.insert(
        "artifact_digest".to_string(),
        Value::String(artifact_digest.clone()),
    );
    index_doc.insert(
        "order_digest".to_string(),
        Value::String(order_digest.clone()),
    );
    index_doc.insert("row_count".to_string(), Value::from(row_count as u64));
    index_doc.insert("batches".to_string(), Value::Array(index_batches));
    let index_bytes =
        hydra_feed::canon::canonical_bytes_value(&Value::Object(index_doc), "rl:index")
            .map_err(|e| order_err(format!("index JCS emit failed: {e:?}")))?;
    Ok((
        OrderPlan {
            artifact_digest,
            order_digest,
            batches,
            row_count,
        },
        index_bytes,
    ))
}

/// Parse canonical index bytes back (structural cold JSON).
pub fn parse_index(index_bytes: &[u8]) -> Result<OrderPlan, TapeError> {
    let value: Value = serde_json::from_slice(index_bytes)
        .map_err(|e| order_err(format!("index is not JSON: {e}")))?;
    let map = value
        .as_object()
        .ok_or_else(|| order_err("index must be an object".to_string()))?;
    if map.get("schema").and_then(Value::as_str) != Some("hydra2.rl_order_plan_v1") {
        return Err(order_err("index schema mismatch".to_string()));
    }
    let artifact_digest = map
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| order_err("index lacks artifact_digest".to_string()))?
        .to_string();
    let order_digest = map
        .get("order_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| order_err("index lacks order_digest".to_string()))?
        .to_string();
    let row_count = map
        .get("row_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| order_err("index lacks row_count".to_string()))? as usize;
    let mut batches = Vec::new();
    for entry in map
        .get("batches")
        .and_then(Value::as_array)
        .ok_or_else(|| order_err("index lacks batches".to_string()))?
    {
        let t = entry
            .get("t")
            .and_then(Value::as_u64)
            .ok_or_else(|| order_err("index batch lacks t".to_string()))? as u32;
        let positions: Vec<u32> = entry
            .get("positions")
            .and_then(Value::as_array)
            .ok_or_else(|| order_err("index batch lacks positions".to_string()))?
            .iter()
            .map(|p| {
                p.as_u64().ok_or_else(|| {
                    order_err("index position must be an integer".to_string())
                })
            })
            .collect::<Result<Vec<u64>, _>>()?
            .into_iter()
            .map(|p| p as u32)
            .collect();
        batches.push(BatchPlan { t, positions });
    }
    Ok(OrderPlan {
        artifact_digest,
        order_digest,
        batches,
        row_count,
    })
}
/// Live replay descriptor schema (typed registration, no file round trip).
pub const LIVE_REPLAY_SCHEMA: &str = "hydra2.rl_live_replay/v1";

/// Register typed replay directly from a drained tape: no seal stamps, no
/// metas/index file round trip, no canonical observation expansion. The
/// collector/run descriptor must match the tape's immutable live binding;
/// every captured row is enumerated exactly once with small typed gates
/// (prefix bounds, four-seat actor, bound terminal facts, finite LP/V/G4/A,
/// behavior/rule/action identities, selected/legal relation), sorted by the
/// canonical `(wall_id, game_id, trajectory_step, actor, decision_id)` key
/// with duplicate keys and non-increasing per-game steps rejected, then
/// planned through the shared shuffle/group/cap law. Full positions remain
/// in Rust. Returns the replay handle plus the live descriptor
/// `{schema, pipeline_spec_id, generation, behavior_policy_id, row_count,
/// batches:[{t, count}]}`. Either one valid handle exists afterwards or an
/// error returns with nothing inserted.
pub fn register_live_replay(
    tape_handle: TapeHandle,
    collector_spec_json: &[u8],
    run_descriptor_json: &[u8],
) -> Result<(ReplayHandle, Vec<u8>), TapeError> {
    let collector: Value = serde_json::from_slice(collector_spec_json)
        .map_err(|e| order_err(format!("live collector spec not JSON: {e}")))?;
    let cspec = collector
        .as_object()
        .ok_or_else(|| order_err("live collector spec must be an object".to_string()))?;
    if cspec.get("schema").and_then(Value::as_str)
        != Some(crate::rl::live::LIVE_COLLECTOR_SCHEMA)
    {
        return Err(order_err("live collector schema mismatch".to_string()));
    }
    let generation = cspec
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| order_err("live collector lacks generation".to_string()))?;
    let policy_id = cspec
        .get("policy_id")
        .and_then(Value::as_str)
        .ok_or_else(|| order_err("live collector lacks policy_id".to_string()))?;
    parse_digest_text(policy_id)?;
    let context = cspec
        .get("context")
        .and_then(Value::as_object)
        .ok_or_else(|| order_err("live collector lacks context".to_string()))?;
    let ctx_str = |name: &str| {
        context
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| order_err(format!("live collector context lacks {name}")))
    };
    let rules_hash = ctx_str("rules_hash")?.to_string();
    let action_table_hash = ctx_str("action_table_hash")?.to_string();
    let behavior_ckpt = ctx_str("behavior_ckpt")?.to_string();
    if behavior_ckpt != policy_id {
        return Err(order_err(
            "live collector behavior_ckpt mismatches policy_id".to_string(),
        ));
    }
    let descriptor: Value = serde_json::from_slice(run_descriptor_json)
        .map_err(|e| order_err(format!("live run descriptor not JSON: {e}")))?;
    let doc = descriptor
        .as_object()
        .ok_or_else(|| order_err("live run descriptor must be an object".to_string()))?;
    let doc_str = |name: &str| {
        doc.get(name)
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .ok_or_else(|| order_err(format!("live run descriptor lacks {name}")))
    };
    let pipeline_spec_id = doc_str("pipeline_spec_id")?;
    parse_digest_text(&pipeline_spec_id)?;
    let master_seed_hex = doc_str("master_seed_hex")?;
    let experiment_id = doc_str("experiment_id")?;
    let replicate_id = doc
        .get("replicate_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| order_err("live run descriptor lacks replicate_id".to_string()))?;
    let attempt_id = doc
        .get("attempt_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| order_err("live run descriptor lacks attempt_id".to_string()))?;
    let generations_total = doc
        .get("cohort")
        .and_then(Value::as_object)
        .and_then(|c| c.get("generations"))
        .and_then(Value::as_u64)
        .ok_or_else(|| order_err("live run descriptor lacks cohort.generations".to_string()))?;
    if generation >= generations_total {
        return Err(order_err(format!(
            "live generation {generation} outside 0..{generations_total}"
        )));
    }
    // The collector shuffle key must equal the descriptor-derived closed key.
    let expected_key = serde_json::json!({
        "master_seed_hex": master_seed_hex,
        "pipeline_spec_id": pipeline_spec_id,
        "experiment_id": experiment_id,
        "generation": generation,
        "epoch": 0,
        "replicate_id": replicate_id,
        "attempt_id": attempt_id,
    });
    let shuffle_key_doc = cspec
        .get("shuffle_key")
        .ok_or_else(|| order_err("live collector lacks shuffle_key".to_string()))?;
    if shuffle_key_doc != &expected_key {
        return Err(order_err(
            "live collector shuffle key mismatches run descriptor".to_string(),
        ));
    }
    // Binding match before any row work: unknown/dropped tapes and
    // collector/descriptor mismatches fail here.
    let binding = tape_live_binding(tape_handle)?;
    if binding.generation != generation {
        return Err(order_err(format!(
            "live binding generation {} mismatches collector generation {generation}",
            binding.generation
        )));
    }
    if binding.behavior_policy_id != policy_id {
        return Err(order_err(
            "live binding behavior policy mismatches collector policy".to_string(),
        ));
    }
    if binding.pipeline_spec_id != pipeline_spec_id {
        return Err(order_err(
            "live binding pipeline mismatches run descriptor".to_string(),
        ));
    }
    let behavior_bytes = parse_digest_text(policy_id)?;
    // Enumerate immutable tape rows under one read hold with small typed
    // gates (never full observation rebuilds). Policy/visibility gates at
    // capture and encoder boundaries remain; these gates only admit rows
    // the frozen law can serve.
    struct LiveRow {
        sort: (String, String, u64, u8, String),
        gi: usize,
        ri: usize,
        t: u32,
        meta: FrozenMetaRow,
    }
    let mut rows: Vec<LiveRow> = with_tape(tape_handle, |tape| {
        let mut out = Vec::new();
        for (gi, game) in tape.games.iter().enumerate() {
            for (ri, row) in game.rows.iter().enumerate() {
                if row.actor as usize >= 4 {
                    return Err(order_err(format!(
                        "row {} actor {} outside four seats",
                        row.decision_id, row.actor
                    )));
                }
                let prefix_len = row.event_prefix_len as usize;
                if prefix_len > game.arena.len() {
                    return Err(order_err(format!(
                        "row {} prefix exceeds arena",
                        row.decision_id
                    )));
                }
                let (g4, raw_a) = match (row.return_g4, row.raw_advantage_f32) {
                    (Some(g), Some(a)) => (g, a),
                    (None, None) => {
                        return Err(order_err(format!(
                            "row {} missing terminal facts; freeze before registration",
                            row.decision_id
                        )))
                    }
                    _ => {
                        return Err(order_err(format!(
                            "row {} forked terminal facts",
                            row.decision_id
                        )))
                    }
                };
                if !row.selected_logp.is_finite() {
                    return Err(order_err(format!(
                        "row {} selected logp must be finite",
                        row.decision_id
                    )));
                }
                if row.value4.iter().any(|v| !v.is_finite()) {
                    return Err(order_err(format!(
                        "row {} values must be finite",
                        row.decision_id
                    )));
                }
                if g4.iter().any(|v| !v.is_finite()) || !raw_a.is_finite() {
                    return Err(order_err(format!(
                        "row {} frozen facts must be finite",
                        row.decision_id
                    )));
                }
                if row.behavior_ckpt != behavior_bytes {
                    return Err(order_err(format!(
                        "row {} behavior identity mismatches collector policy",
                        row.decision_id
                    )));
                }
                if row.rules_hash != rules_hash {
                    return Err(order_err(format!(
                        "row {} rules identity mismatches collector context",
                        row.decision_id
                    )));
                }
                if row.action_table_hash_hex != action_table_hash {
                    return Err(order_err(format!(
                        "row {} action identity mismatches collector context",
                        row.decision_id
                    )));
                }
                if row.legal_ids.is_empty() || !row.legal_ids.contains(&row.selected) {
                    return Err(order_err(format!(
                        "row {} selected action outside legal set",
                        row.decision_id
                    )));
                }
                if row.legal_ids.iter().any(|id| *id as usize >= ACTION_WIDTH) {
                    return Err(order_err(format!(
                        "row {} legal id outside the 6792 vocabulary",
                        row.decision_id
                    )));
                }
                let visible = game.arena.visible_count_for_hand(row.actor, prefix_len);
                let visible_u32 = u32::try_from(visible).map_err(|_| {
                    order_err(format!("row {} visible history outside u32", row.decision_id))
                })?;
                let t = bucket_history_len(visible_u32)?;
                out.push(LiveRow {
                    sort: (
                        game.wall_id.clone(),
                        game.game_id.clone(),
                        row.trajectory_step,
                        row.actor,
                        row.decision_id.clone(),
                    ),
                    gi,
                    ri,
                    t,
                    meta: FrozenMetaRow {
                        gi,
                        ri,
                        old_lp: row.selected_logp,
                        raw_a,
                        g4,
                    },
                });
            }
        }
        Ok(out)
    })?;
    if rows.is_empty() {
        return Err(order_err("live tape carries no rows".to_string()));
    }
    rows.sort_by(|a, b| a.sort.cmp(&b.sort));
    // Duplicate full sort keys and non-increasing per-game steps reject
    // before ordering/shuffling, so no row-id tie-break is needed.
    let mut prev_key: Option<&(String, String, u64, u8, String)> = None;
    let mut prev_game: Option<(&str, &str)> = None;
    let mut prev_step = 0u64;
    for row in &rows {
        if prev_key == Some(&row.sort) {
            return Err(order_err("live tape duplicates a full sort key".to_string()));
        }
        prev_key = Some(&row.sort);
        let game = (row.sort.0.as_str(), row.sort.1.as_str());
        if prev_game == Some(game) && row.sort.2 <= prev_step {
            return Err(order_err(
                "live tape trajectory steps must increase per game".to_string(),
            ));
        }
        prev_game = Some(game);
        prev_step = row.sort.2;
    }
    // Canonical positions: metas and buckets in the same sorted order.
    let row_count = rows.len();
    let mut metas = Vec::with_capacity(row_count);
    let mut buckets = Vec::with_capacity(row_count);
    for row in &rows {
        metas.push(FrozenMetaRow {
            gi: row.gi,
            ri: row.ri,
            old_lp: row.meta.old_lp,
            raw_a: row.meta.raw_a,
            g4: row.meta.g4,
        });
        buckets.push(row.t);
    }
    let key = ShuffleKey {
        master_seed_hex,
        pipeline_spec_id: pipeline_spec_id.clone(),
        experiment_id,
        generation,
        epoch: 0,
        replicate_id,
        attempt_id,
    };
    let plans = plan_positions(&buckets, &key)?;
    let source = registry::ReplaySource::Live {
        tape_handle,
        pipeline_spec_id: pipeline_spec_id.clone(),
        generation,
        behavior_policy_id: policy_id.to_string(),
    };
    let handle = registry::insert_frozen_replay(source, metas, &plans)?;
    let batches_doc: Vec<Value> = plans
        .iter()
        .map(|b| {
            serde_json::json!({"t": b.t, "count": b.positions.len()})
        })
        .collect();
    let descriptor = serde_json::json!({
        "schema": LIVE_REPLAY_SCHEMA,
        "pipeline_spec_id": pipeline_spec_id,
        "generation": generation,
        "behavior_policy_id": policy_id,
        "row_count": row_count,
        "batches": batches_doc,
    });
    let bytes = hydra_feed::canon::canonical_bytes_value(&descriptor, "rl:live-replay")
        .map_err(|e| order_err(format!("live replay descriptor JCS emit failed: {e:?}")))?;
    Ok((handle, bytes))
}


#[cfg(test)]
mod replay_index_tests {
    use super::*;

    fn key() -> ShuffleKey {
        ShuffleKey {
            master_seed_hex: "00".repeat(32),
            pipeline_spec_id: "pipe-1".to_string(),
            experiment_id: "exp-1".to_string(),
            generation: 3,
            epoch: 0,
            replicate_id: 0,
            attempt_id: 0,
        }
    }

    fn metas_fixture(count: usize) -> Vec<u8> {
        let mut rows = Vec::with_capacity(count);
        for i in 0..count {
            let mut m = Map::with_capacity(12);
            m.insert(
                "row_id".to_string(),
                Value::String(format!("{DIGEST_PREFIX}{:064x}", i + 1)),
            );
            // Alternate buckets to exercise grouping + tails.
            let t = [32u64, 64, 128, 256][i % 4];
            m.insert("t".to_string(), Value::from(t));
            m.insert("old_lp".to_string(), Value::from(f64::from(-0.5f32)));
            m.insert("raw_a".to_string(), Value::from(1.0));
            m.insert(
                "g4".to_string(),
                Value::Array(vec![
                    Value::from(1.0),
                    Value::from(2.0),
                    Value::from(3.0),
                    Value::from(4.0),
                ]),
            );
            m.insert("v".to_string(), Value::from(0.0));
            m.insert("wall_id".to_string(), Value::String("wall-1".to_string()));
            m.insert("game_id".to_string(), Value::String("game-a".to_string()));
            m.insert("step".to_string(), Value::from(i as u64));
            m.insert("gi".to_string(), Value::from(0u64));
            m.insert("ri".to_string(), Value::from(i as u64));
            rows.push(Value::Object(m));
        }
        let mut header = Map::new();
        for name in crate::rl::artifact::HEADER_KEYS {
            header.insert(
                name.to_string(),
                Value::String(format!("{DIGEST_PREFIX}{}", "c".repeat(64))),
            );
        }
        let mut doc = Map::new();
        doc.insert(
            "schema".to_string(),
            Value::String("hydra2.rl_metas_v1".to_string()),
        );
        doc.insert(
            "artifact_digest".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "a".repeat(64))),
        );
        doc.insert("header".to_string(), Value::Object(header));
        doc.insert("row_count".to_string(), Value::from(count as u64));
        doc.insert("rows".to_string(), Value::Array(rows));
        doc.insert("compressed_bytes".to_string(), Value::from(8u64));
        doc.insert("uncompressed_bytes".to_string(), Value::from(80u64));
        hydra_feed::canon::canonical_bytes_value(&Value::Object(doc), "test:metas")
            .expect("test metas JCS")
    }

    #[test]
    fn shuffle_is_deterministic_and_epoch_scoped() {
        let metas = metas_fixture(24);
        let (plan_a, _) = build_order(&metas, &key()).expect("order a");
        let (plan_b, _) = build_order(&metas, &key()).expect("order b");
        assert_eq!(plan_a.order_digest, plan_b.order_digest);
        assert_eq!(plan_a.batches.len(), plan_b.batches.len());
        // T-grouped ascending: 6 rows per bucket -> 4 homogeneous batches.
        assert_eq!(plan_a.batches.len(), 4);
        for (i, t) in [32u32, 64, 128, 256].iter().enumerate() {
            assert_eq!(plan_a.batches[i].t, *t);
            assert_eq!(plan_a.batches[i].positions.len(), 6);
        }
        // Every canonical position exactly once.
        let mut seen = vec![false; 24];
        for batch in &plan_a.batches {
            for p in &batch.positions {
                assert!(!seen[*p as usize]);
                seen[*p as usize] = true;
            }
        }
        // Different epoch -> different order (scope binds).
        let mut key2 = key();
        key2.epoch = 1;
        let (plan_c, _) = build_order(&metas, &key2).expect("order c");
        assert_ne!(plan_a.order_digest, plan_c.order_digest);
    }

    #[test]
    fn tails_survive_cap_splits() {
        // 4100 rows in one bucket -> 4096 + genuine 4-row tail.
        let full = metas_fixture(4100);
        let value: Value = serde_json::from_slice(&full).expect("metas json");
        let mut rows = value
            .get("rows")
            .and_then(Value::as_array)
            .expect("rows")
            .clone();
        for row in &mut rows {
            row["t"] = Value::from(64u64);
        }
        let mut doc = Map::new();
        doc.insert(
            "schema".to_string(),
            Value::String("hydra2.rl_metas_v1".to_string()),
        );
        doc.insert(
            "artifact_digest".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "a".repeat(64))),
        );
        doc.insert("compressed_bytes".to_string(), Value::from(8u64));
        doc.insert("uncompressed_bytes".to_string(), Value::from(80u64));
        doc.insert("rows".to_string(), Value::Array(rows));
        let bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(doc), "test:metas")
            .expect("metas");
        let (plan, index_bytes) = build_order(&bytes, &key()).expect("order");
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].positions.len(), 4096);
        assert_eq!(plan.batches[1].positions.len(), 4);
        // Index round-trips through its own parser.
        let parsed = parse_index(&index_bytes).expect("parse index");
        assert_eq!(parsed.order_digest, plan.order_digest);
        assert_eq!(parsed.batches.len(), 2);
    }

    #[test]
    fn random_below_is_deterministic_and_bounded() {
        // Two fresh streams over one seed agree draw-for-draw (counter
        // semantics: position, never call order, determines output); every
        // draw lies in range. Cross-language values are pinned by the
        // Python gate (RandomStream oracle), not by a hand constant here.
        let mut a = CtrStream {
            seed: vec![0u8; 32],
            cursor: 0,
        };
        let mut b = CtrStream {
            seed: vec![0u8; 32],
            cursor: 0,
        };
        for bound in [2u64, 3, 7, 136, 4097] {
            let x = a.random_below(bound).expect("draw a");
            let y = b.random_below(bound).expect("draw b");
            assert_eq!(x, y);
            assert!(x < bound);
        }
        assert!(a.random_below(1).is_err());
    }

    #[test]
    fn slab_layout_math() {
        assert_eq!(feature_row_bytes(32).expect("t32"), 7254 + 9 * 32);
        assert_eq!(feature_row_bytes(64).expect("t64"), 7254 + 9 * 64);
        assert_eq!(feature_row_bytes(128).expect("t128"), 7254 + 9 * 128);
        assert_eq!(feature_row_bytes(256).expect("t256"), 7254 + 9 * 256);
        assert!(feature_row_bytes(33).is_err());
    }
    // Live registration fixtures: real typed rows over a public arena.
    fn live_arena(events: usize) -> crate::rl::events::EventArena {
        use crate::rl::events::{EventArena, TypedEvent, Visibility};
        let mut arena = EventArena::new();
        let mut admit = |seq: u64,
                         kind: &str,
                         actor: Option<u8>,
                         tile: Option<u8>,
                         action_id: Option<u32>| {
            arena
                .admit(TypedEvent {
                    game_id: "g-live".to_string(),
                    sequence: seq,
                    kind: kind.to_string(),
                    actor,
                    visibility: Visibility::Public,
                    visible_to: vec![0, 1, 2, 3],
                    tile,
                    action_id,
                    source_seat: None,
                    consumed_tiles: vec![],
                    offered_action_ids: vec![],
                    accepted_action_ids: vec![],
                    round_index: if kind == "discard" { None } else { Some(0) },
                    scores: if kind == "discard" {
                        None
                    } else {
                        Some([25000, 25000, 25000, 25000])
                    },
                    reason: None,
                    public_delta: vec![],
                    rules_hash: "sha256:00".to_string(),
                    schema_hash: "sha256:02".to_string(),
                })
                .expect("admit public live event")
        };
        admit(0, "game_start", None, None, None);
        admit(1, "round_start", Some(0), None, None);
        for seq in 2..events as u64 {
            admit(
                seq,
                "discard",
                Some((seq % 4) as u8),
                Some((seq % 34) as u8),
                Some((60 + seq) as u32),
            );
        }
        arena
    }

    fn live_row(
        step: u64,
        actor: u8,
        dec: &str,
        prefix: u32,
        behavior: [u8; 32],
    ) -> crate::rl::core::GameRow {
        use crate::rl::core::GameRow;
        GameRow {
            wall_id: "wall-live".to_string(),
            game_id: "g-live".to_string(),
            round_id: "g-live:0".to_string(),
            trajectory_step: step,
            actor,
            decision_id: dec.to_string(),
            sequence: step,
            row_id: [0u8; 32],
            round_index: 0,
            round_wind: 27,
            honba: 0,
            kyotaku: 0,
            dealer: 0,
            scores: [25000, 25000, 25000, 25000],
            live_remaining: 65,
            hand_number: 0,
            turn_actor: actor,
            phase_id: 1,
            concealed: vec![0, 1, 2],
            drawn: Some(16),
            dora5: [4, -1, -1, -1, -1],
            ippatsu_bits: 0,
            furiten_id: 0,
            can_tsumo: true,
            can_riichi: false,
            legal_ids: vec![0, 61],
            legal_mask_bits: vec![],
            selected: 61,
            selected_logp: -0.5,
            value4: [0.1, 0.2, 0.3, 0.4],
            cursor_before: step * 8,
            cursor_after: step * 8 + 8,
            obs_hash: [0u8; 32],
            event_prefix_len: prefix,
            derivation: [0u8; 32],
            action_table_hash: [0u8; 32],
            action_table_hash_hex: "sha256:01".to_string(),
            behavior_ckpt: behavior,
            transition_provenance: [0u8; 32],
            rules_id: "test-rules".to_string(),
            rules_hash: "sha256:00".to_string(),
            event_schema_hash: "sha256:02".to_string(),
            observation_schema_hash: "sha256:03".to_string(),
            packet_boundary_hash: "sha256:04".to_string(),
            return_g4: Some([10.0, 20.0, -15.0, -15.0]),
            raw_advantage_f32: Some(2.0),
        }
    }

    fn live_binding(policy: &str, pipeline: &str) -> crate::rl::trajectory::LiveTapeBinding {
        crate::rl::trajectory::LiveTapeBinding {
            pipeline_spec_id: pipeline.to_string(),
            generation: 0,
            behavior_policy_id: policy.to_string(),
        }
    }

    fn live_docs(policy: &str, pipeline: &str) -> (Vec<u8>, Vec<u8>) {
        let collector = serde_json::json!({
            "schema": "hydra2.rl_live_collector/v1",
            "generation": 0,
            "pool_id": "live-test",
            "policy_id": policy,
            "profile": {"pool_id": "live-test"},
            "context": {
                "rules_hash": "sha256:00",
                "action_table_hash": "sha256:01",
                "behavior_ckpt": policy,
            },
            "shuffle_key": {
                "master_seed_hex": "ab".repeat(32),
                "pipeline_spec_id": pipeline,
                "experiment_id": "live-exp",
                "generation": 0,
                "epoch": 0,
                "replicate_id": 0,
                "attempt_id": 0,
            },
        });
        let descriptor = serde_json::json!({
            "pipeline_spec_id": pipeline,
            "master_seed_hex": "ab".repeat(32),
            "experiment_id": "live-exp",
            "replicate_id": 0,
            "attempt_id": 0,
            "cohort": {"generations": 8},
        });
        (
            serde_json::to_vec(&collector).expect("collector bytes"),
            serde_json::to_vec(&descriptor).expect("descriptor bytes"),
        )
    }

    fn live_tape_handle(
        rows: Vec<crate::rl::core::GameRow>,
        arena: crate::rl::events::EventArena,
        binding: crate::rl::trajectory::LiveTapeBinding,
    ) -> crate::rl::trajectory::TapeHandle {
        use crate::rl::trajectory::{TapeGame, register_live_tape};
        register_live_tape(
            vec![TapeGame {
                wall_id: "wall-live".to_string(),
                game_id: "g-live".to_string(),
                arena,
                rows,
                utilities: Some([10.0, 20.0, -15.0, -15.0]),
            }],
            Box::new(|_, _| Ok(serde_json::Value::Null)),
            Box::new(|_, _, _| Ok(vec![])),
            binding,
        )
        .expect("register live test tape")
    }

    #[test]
    fn live_registration_rejects_duplicate_sort_keys() {
        let policy = format!("sha256:{}", "07".repeat(32));
        let pipeline = format!("sha256:{}", "09".repeat(32));
        let arena = live_arena(40);
        let rows = vec![
            live_row(0, 0, "dec-0", 40, [7u8; 32]),
            live_row(0, 0, "dec-0", 40, [7u8; 32]),
        ];
        let tape = live_tape_handle(rows, arena, live_binding(&policy, &pipeline));
        let (cspec, desc) = live_docs(&policy, &pipeline);
        assert!(register_live_replay(tape, &cspec, &desc).is_err());
    }

    #[test]
    fn live_registration_rejects_nonincreasing_steps() {
        let policy = format!("sha256:{}", "07".repeat(32));
        let pipeline = format!("sha256:{}", "09".repeat(32));
        let arena = live_arena(40);
        // Same step, distinct decisions: steps must strictly increase.
        let rows = vec![
            live_row(0, 0, "dec-0", 40, [7u8; 32]),
            live_row(0, 1, "dec-1", 40, [7u8; 32]),
        ];
        let tape = live_tape_handle(rows, arena, live_binding(&policy, &pipeline));
        let (cspec, desc) = live_docs(&policy, &pipeline);
        assert!(register_live_replay(tape, &cspec, &desc).is_err());
        // Strictly increasing steps register with every row exactly once.
        let arena = live_arena(40);
        let rows = vec![
            live_row(0, 0, "dec-0", 40, [7u8; 32]),
            live_row(1, 1, "dec-1", 40, [7u8; 32]),
            live_row(2, 2, "dec-2", 40, [7u8; 32]),
        ];
        let tape = live_tape_handle(rows, arena, live_binding(&policy, &pipeline));
        let (handle, bytes) = register_live_replay(tape, &cspec, &desc).expect("live register");
        assert!(handle != 0);
        let doc: Value =
            serde_json::from_slice(&bytes).expect("live descriptor json");
        assert_eq!(doc["row_count"], 3);
        assert_eq!(doc["batches"].as_array().expect("batches").len(), 1);
    }

    #[test]
    fn live_registration_rejects_missing_facts_and_wrong_behavior() {
        let policy = format!("sha256:{}", "07".repeat(32));
        let pipeline = format!("sha256:{}", "09".repeat(32));
        let (cspec, desc) = live_docs(&policy, &pipeline);
        // Unfrozen row: terminal facts missing before registration.
        let mut unfrozen = live_row(0, 0, "dec-0", 40, [7u8; 32]);
        unfrozen.return_g4 = None;
        unfrozen.raw_advantage_f32 = None;
        let tape = live_tape_handle(vec![unfrozen], live_arena(40), live_binding(&policy, &pipeline));
        assert!(register_live_replay(tape, &cspec, &desc).is_err());
        // Wrong behavior identity: row binds another policy.
        let foreign = live_row(0, 0, "dec-0", 40, [8u8; 32]);
        let tape = live_tape_handle(vec![foreign], live_arena(40), live_binding(&policy, &pipeline));
        assert!(register_live_replay(tape, &cspec, &desc).is_err());
    }

    #[test]
    fn live_registration_splits_genuine_tails() {
        // 4100 rows in one bucket: 4096 plus a genuine 4-row tail.
        let policy = format!("sha256:{}", "07".repeat(32));
        let pipeline = format!("sha256:{}", "09".repeat(32));
        let arena = live_arena(40);
        let rows: Vec<_> = (0..4100u64)
            .map(|step| {
                live_row(
                    step,
                    (step % 4) as u8,
                    &format!("dec-{step}"),
                    40,
                    [7u8; 32],
                )
            })
            .collect();
        let tape = live_tape_handle(rows, arena, live_binding(&policy, &pipeline));
        let (cspec, desc) = live_docs(&policy, &pipeline);
        let (_, bytes) = register_live_replay(tape, &cspec, &desc).expect("live tails");
        let doc: Value = serde_json::from_slice(&bytes).expect("tails json");
        let counts: Vec<u64> = doc["batches"]
            .as_array()
            .expect("tails batches")
            .iter()
            .map(|b| b["count"].as_u64().expect("tail count"))
            .collect();
        assert_eq!(counts, vec![4096, 4]);
        assert_eq!(doc["row_count"], 4100);
    }
}
