//! rl/runtime_snapshot_control: runtime stop commit + verified resume.
//!
//! DAG: `super::{runtime_snapshot, runtime_snapshot_pool,
//! runtime_snapshot_tape, snapshot}` + `serde_json` + `crate::rl::config`
//! (closed vocabularies only).
//! FORBIDS: PyO3, wall-clock, RNG, new control vocabulary (the committed
//! document is always the exact registered 26-field control, assembled by
//! the projector below with lifecycle-mirrored gates and published by
//! `snapshot::publish_control`; the runtime state rides opaque CAS blobs).
//!
//! Commit (`runtime_control`): Main passes the opaque contract state plus
//! the actual latest ordinary receipts at a completed optimizer boundary
//! (CUDA owner quiesced, READY retained, ordinary snapshots + receipts
//! landed). Native validates the state schema (no defaults, every
//! id/count actual), verifies every referenced blob digest, re-derives
//! the pool collection facts and requires equality, projects the ACTUAL
//! latest receipt ids/locators into the control (ongoing learner Cx,
//! pending actor Cn) while the fixed generation parent and the RunSpec
//! stay read-only, publishes through the snapshot publisher (fixed
//! parent, phase monotonicity, receipt + blob re-verification against
//! disk), then advances the runtime pointer. A cold current (initial
//! collection, actual pool only) commits with null order/tape/event slots
//! rather than fabricated digests. Retried commits are idempotent.
//!
//! Resume (`restore_runtime`): last-committed runtime pointer authority,
//! cross-checked against the control tip; every receipt re-verified
//! against disk and every blob re-verified before any handle or locator
//! is returned. Tapes re-register with the existing pure observation
//! helpers and are re-stamped with the TIP-COMMITTED artifact digest only
//! (never a recomputed variant), so export's staleness gate keeps binding
//! the restored tape to the committed artifact. No inference, no
//! resampling, no new RNG, no version bump.

use std::path::Path;

use serde_json::{Map, Value};

use crate::rl::config::{RL_PUBLICATION_PHASES, RL_TICKET_PHASES};
use crate::rl::runtime_snapshot::{
    CasRef, ValidatedState, atomic_write, canon_bytes, is_digest_text, parse_cas_ref, parse_json,
    read_runtime_pointer, read_verified_blob, rt_err, validate_runtime_state, write_runtime_blob,
    write_runtime_pointer,
};
use crate::rl::runtime_snapshot_pool::{check_pool_ref, restore_pool_doc};
use crate::rl::runtime_snapshot_tape::restore_tape;
use crate::rl::snapshot::{read_tip, verify_receipt_against_disk};
use crate::rl::trajectory::TapeError;

/// Descriptor facts needed by the commit path (read-only projection of
/// the `prepare_run` descriptor; every field actual, never defaulted).
struct DescriptorView {
    pipeline_spec_id: String,
    initial_checkpoint_id: String,
    generations: u64,
    games_per_generation: u64,
    wall_ledger_hash: Option<String>,
    row_derivation_hash: Option<String>,
    /// Per-generation wall -> schedule digest for the commit generation.
    source_digests: Vec<(String, String)>,
}

fn parse_descriptor(doc: &[u8]) -> Result<DescriptorView, TapeError> {
    let value = parse_json(doc, "run descriptor")?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err("run descriptor must be an object"))?;
    match map.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(rt_err(format!(
                "descriptor version must be '1.0.0', got {other:?}"
            )));
        }
    }
    let str_of = |key: &str| {
        map.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| rt_err(format!("descriptor lacks {key}")))
    };
    let pipeline_spec_id = str_of("pipeline_spec_id")?;
    let initial_checkpoint_id = str_of("initial_checkpoint_id")?;
    if !is_digest_text(&pipeline_spec_id) || !is_digest_text(&initial_checkpoint_id) {
        return Err(rt_err("descriptor carries bad digest shapes"));
    }
    let cohort = map
        .get("cohort")
        .and_then(Value::as_object)
        .ok_or_else(|| rt_err("descriptor lacks cohort"))?;
    let generations = cohort.get("generations").and_then(Value::as_u64).ok_or_else(|| {
        rt_err("descriptor cohort lacks generations")
    })?;
    let games_per_generation = cohort
        .get("games_per_generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| rt_err("descriptor cohort lacks games_per_generation"))?;
    Ok(DescriptorView {
        pipeline_spec_id,
        initial_checkpoint_id,
        generations,
        games_per_generation,
        wall_ledger_hash: map
            .get("wall_ledger_hash")
            .and_then(Value::as_str)
            .map(str::to_string),
        row_derivation_hash: map
            .get("row_derivation_hash")
            .and_then(Value::as_str)
            .map(str::to_string),
        source_digests: Vec::new(),
    })
}

fn descriptor_sources(descriptor: &Value, generation: u64) -> Result<Vec<(String, String)>, TapeError> {
    let map = descriptor
        .as_object()
        .ok_or_else(|| rt_err("run descriptor must be an object"))?;
    let generations = map
        .get("generations")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("descriptor lacks generations array"))?;
    let entry = generations.get(generation as usize).ok_or_else(|| {
        rt_err(format!("descriptor generation {generation} outside cohort"))
    })?;
    let games = entry
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("descriptor generation lacks games"))?;
    let mut out = Vec::with_capacity(games.len());
    for (i, game) in games.iter().enumerate() {
        let wall = game
            .get("wall_id")
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("descriptor games[{i}] lacks wall_id")))?;
        let digest = game
            .get("schedule_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("descriptor games[{i}] lacks schedule_digest")))?;
        if !is_digest_text(digest) {
            return Err(rt_err(format!("descriptor games[{i}] schedule digest malformed")));
        }
        out.push((wall.to_string(), digest.to_string()));
    }
    Ok(out)
}

/// Bindings facts for read-only cross-checks (generation, pipeline, fixed
/// parent, behavior, order). Bindings are never mutated and never required.
struct BindingsView {
    generation: u64,
    pipeline_spec_id: String,
    learner_init: String,
    actor_id: String,
    order_digest: String,
}

fn parse_bindings(doc: &[u8]) -> Result<BindingsView, TapeError> {
    let value = parse_json(doc, "generation bindings")?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err("generation bindings must be an object"))?;
    match map.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(rt_err(format!(
                "bindings version must be '1.0.0', got {other:?}"
            )));
        }
    }
    let get_digest = |key: &str| {
        let text = map
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("bindings lack {key}")))?;
        if !is_digest_text(text) {
            return Err(rt_err(format!("bindings {key} has a bad shape")));
        }
        Ok::<String, TapeError>(text.to_string())
    };
    let generation = map.get("generation").and_then(Value::as_u64).ok_or_else(|| {
        rt_err("bindings lack generation")
    })?;
    Ok(BindingsView {
        generation,
        pipeline_spec_id: get_digest("pipeline_spec_id")?,
        learner_init: get_digest("learner_initial_manifest_id")?,
        actor_id: get_digest("actor_manifest_id")?,
        order_digest: get_digest("order_digest")?,
    })
}

fn receipt_manifest_id(receipt_doc: &[u8], role: &str) -> Result<(String, String), TapeError> {
    let value = parse_json(receipt_doc, &format!("{role} receipt"))?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{role} receipt must be an object")))?;
    let manifest = map
        .get("manifest_id")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{role} receipt lacks manifest_id")))?;
    let locator = map
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{role} receipt lacks locator")))?;
    if !is_digest_text(manifest) {
        return Err(rt_err(format!("{role} receipt manifest has a bad shape")));
    }
    Ok((manifest.to_string(), locator.to_string()))
}

/// Verify one side's blobs (index order linkage, tape counts + event
/// digest, pool re-derivation) and return the event ref + index plan ids +
/// pool cohort walls + tape game walls. Walls ride along from bytes already
/// verified here so callers stop re-reading 400MB blobs for small questions.
fn verify_side_blobs(
    root: &Path,
    side: &crate::rl::runtime_snapshot::StateSide,
    name: &str,
) -> Result<(CasRef, String, String, u64, Vec<String>, Vec<String>), TapeError> {
    let receipt = side
        .receipt
        .as_ref()
        .ok_or_else(|| rt_err(format!("{name} lacks a sealed receipt")))?;
    let receipt_map = receipt
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} receipt malformed")))?;
    let artifact = receipt_map
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} receipt lacks artifact_digest")))?
        .to_string();
    let rows = receipt_map.get("row_count").and_then(Value::as_u64).ok_or_else(|| {
        rt_err(format!("{name} receipt lacks row_count"))
    })?;
    let order = side
        .order
        .as_ref()
        .ok_or_else(|| rt_err(format!("{name} lacks a frozen order")))?;
    let order_map = order
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} order malformed")))?;
    let order_digest = order_map
        .get("order_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} order lacks order_digest")))?
        .to_string();
    // Index blob: canonical index bytes, order linkage actual.
    let index_ref = side
        .index_ref
        .as_ref()
        .ok_or_else(|| rt_err(format!("{name} lacks index_ref")))?;
    let index_bytes = read_verified_blob(root, &index_ref.digest, &index_ref.locator)?;
    let plan = crate::rl::replay_index::parse_index(&index_bytes)
        .map_err(|e| rt_err(format!("{name} index rejected: {e}")))?;
    if plan.artifact_digest != artifact || plan.order_digest != order_digest {
        return Err(rt_err(format!("{name} index diverges from the sealed order")));
    }
    if plan.row_count as u64 != rows {
        return Err(rt_err(format!("{name} index rows diverge from the receipt")));
    }
    // Tape blob: counts + embedded event ref (deep row verification runs
    // at save time and again at restore/registration).
    let tape_ref = side
        .tape_ref
        .as_ref()
        .ok_or_else(|| rt_err(format!("{name} lacks tape_ref")))?;
    let tape_bytes = read_verified_blob(root, &tape_ref.digest, &tape_ref.locator)?;
    let tape_doc = hydra_feed::json_projection::object_fields(&tape_bytes, &["row_count", "event_ref"])
        .map_err(|error| rt_err(format!("{name} tape blob is not JSON: {error}")))?;
    let tape_rows = tape_doc.get("row_count").and_then(Value::as_u64).ok_or_else(|| {
        rt_err(format!("{name} tape blob lacks row_count"))
    })?;
    if tape_rows != rows {
        return Err(rt_err(format!(
            "{name} tape rows {tape_rows} != sealed rows {rows}"
        )));
    }
    let tape_walls_doc =
        hydra_feed::json_projection::object_array_fields(&tape_bytes, "games", &["wall_id"])
            .map_err(|error| rt_err(format!("{name} tape blob is not JSON: {error}")))?;
    let mut tape_walls: Vec<String> = Vec::new();
    for game in tape_walls_doc
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err(format!("{name} tape blob lacks games")))?
    {
        let wall = game
            .get("wall_id")
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("{name} tape game lacks wall_id")))?;
        tape_walls.push(wall.to_owned());
    }
    let event_ref = parse_cas_ref(
        tape_doc.get("event_ref").unwrap_or(&Value::Null),
        &format!("{name} tape event_ref"),
    )?;
    read_verified_blob(root, &event_ref.digest, &event_ref.locator)?;
    // Pool blob: full re-derivation with equality (no invented progress).
    let verified_pool = check_pool_ref(
        root,
        &side.pool_ref.digest,
        &side.pool_ref.locator,
        &side.pool_ref.frontier,
        &side.pool_ref.terminal_set,
        &side.pool_ref.tickets,
        side.pool_ref.captured_rows,
    )?;
    Ok((
        event_ref,
        artifact,
        order_digest,
        rows,
        verified_pool.wall_ids,
        tape_walls,
    ))
}

fn status_for_phase(phase: &str) -> &'static str {
    match phase {
        "not_captured" => "collected",
        "captured_at_update1" => "sealed",
        "checkpoint_durable" => "trained",
        _ => "published",
    }
}
/// Read one string field from an ordinary receipt (`transport_sha256`,
/// `manifest_id`, `locator`).
fn receipt_field(receipt_doc: &[u8], role: &str, key: &str) -> Result<String, TapeError> {
    let value = parse_json(receipt_doc, role)?;
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| rt_err(format!("{role} lacks {key}")))
}

/// Attach the content digest the publisher verifies: the recorded
/// `digest` must equal sha256 over the JCS payload without it (the
/// snapshot publisher rejects a missing or mismatched digest, never
/// computes one silently).
fn attach_control_digest(control_nodigest: &[u8]) -> Result<Vec<u8>, TapeError> {
    let mut value: Value = parse_json(control_nodigest, "control doc")?;
    {
        let map = value
            .as_object_mut()
            .ok_or_else(|| rt_err("control doc must be an object"))?;
        if map.contains_key("digest") {
            return Err(rt_err("control doc must exclude digest (computed at attach)"));
        }
    }
    let canonical = canon_bytes(&value, "rl:control-doc")?;
    let digest = crate::rl::runtime_snapshot::blob_digest(&canonical);
    {
        let map = value
            .as_object_mut()
            .ok_or_else(|| rt_err("control doc must be an object"))?;
        map.insert("digest".to_string(), Value::String(digest));
    }
    canon_bytes(&value, "rl:control-full")
}

/// Null-or-digest control slot (order/tape/event): a digest string or
/// explicit null, nothing else.
fn digest_or_null(value: &Value, name: &str) -> Result<Option<String>, TapeError> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) if is_digest_text(text) => Ok(Some(text.clone())),
        _ => Err(rt_err(format!("control {name} must be a digest or null"))),
    }
}

/// Assemble the projected control: the exact registered 26 fields naming
/// the ACTUAL latest receipts (verified manifests + locators) while the
/// fixed generation parent, cohort, and order stay read-only bindings.
/// Order/tape/event are all-digest (sealed) or all-null (cold, never
/// fabricated). No digest attached (see `attach_control_digest`).
#[allow(clippy::too_many_arguments)]
fn assemble_projected_control(
    descriptor: &DescriptorView,
    state: &ValidatedState,
    learner_manifest: &str,
    learner_transport: &str,
    learner_locator: &str,
    actor_manifest: &str,
    actor_transport: &str,
    actor_locator: &str,
    order_ref: Value,
    tape_digest: Value,
    event_digest: Value,
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
) -> Result<Vec<u8>, TapeError> {
    for (role, id) in [("learner", learner_manifest), ("actor", actor_manifest)] {
        if !is_digest_text(id) {
            return Err(rt_err(format!("projected control {role} id has a bad shape")));
        }
    }
    for (role, sha) in [("learner", learner_transport), ("actor", actor_transport)] {
        if !is_digest_text(sha) {
            return Err(rt_err(format!("projected control {role} transport has a bad shape")));
        }
    }
    for (role, locator) in [("learner", learner_locator), ("actor", actor_locator)] {
        if locator.is_empty() {
            return Err(rt_err(format!("projected control {role} locator is empty")));
        }
    }
    let order = digest_or_null(&order_ref, "order_ref")?;
    let tape = digest_or_null(&tape_digest, "tape_blob_digest")?;
    let event = digest_or_null(&event_digest, "event_blob_digest")?;
    if order.is_none() != tape.is_none() || order.is_none() != event.is_none() {
        return Err(rt_err(
            "control order/tape/event slots must be all-digest or all-null",
        ));
    }
    if !is_digest_text(&descriptor.pipeline_spec_id) {
        return Err(rt_err("projected control pipeline has a bad shape"));
    }
    let frontier_value = |doc: &str, name: &str, generation: u64| -> Result<Value, TapeError> {
        let frontier: Value = parse_json(doc.as_bytes(), name)?;
        let fmap = frontier
            .as_object()
            .ok_or_else(|| rt_err(format!("control frontier {name} must be an object")))?;
        if fmap.get("generation").and_then(Value::as_u64) != Some(generation)
            || fmap.get("round").and_then(Value::as_u64).is_none()
            || fmap.get("epoch").and_then(Value::as_u64).is_none()
        {
            return Err(rt_err(format!("control frontier {name} diverges from state")));
        }
        Ok(frontier)
    };
    let frontier_current = frontier_value(frontier_current_json, "current", state.generation)?;
    // A published lookahead collector belongs to g+1; requiring g rejects
    // valid actor handoff and prevents its durable control transaction.
    let next_generation = if state.future.is_some() {
        state.generation.checked_add(1).ok_or_else(|| rt_err("next frontier generation overflow"))?
    } else {
        state.generation
    };
    let frontier_next = frontier_value(frontier_next_json, "next", next_generation)?;
    // Tickets mirror the lifecycle gates: closed 4-key entries, ticket
    // phase vocabulary, integer epoch + stream cursor.
    let tickets: Value = parse_json(tickets_json, "tickets")?;
    let tickets_array = tickets
        .as_array()
        .ok_or_else(|| rt_err("control tickets must be an array"))?;
    for (i, ticket) in tickets_array.iter().enumerate() {
        let obj = ticket
            .as_object()
            .ok_or_else(|| rt_err(format!("control tickets[{i}] must be an object")))?;
        if obj.len() != crate::rl::lifecycle_control::CONTROL_TICKET_FIELDS.len()
            || !crate::rl::lifecycle_control::CONTROL_TICKET_FIELDS
                .iter()
                .all(|k| obj.contains_key(*k))
        {
            return Err(rt_err(format!("control tickets[{i}] has a bad shape")));
        }
        match obj.get("phase").and_then(Value::as_str) {
            Some(phase) if RL_TICKET_PHASES.contains(&phase) => {}
            other => {
                return Err(rt_err(format!(
                    "control tickets[{i}].phase must be a ticket phase, got {other:?}"
                )));
            }
        }
        if obj.get("slot_epoch").and_then(Value::as_u64).is_none()
            || obj.get("stream_cursor").and_then(Value::as_u64).is_none()
        {
            return Err(rt_err(format!("control tickets[{i}] lacks integer epoch/cursor")));
        }
        if !obj
            .get("ticket_descriptor")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
        {
            return Err(rt_err(format!("control tickets[{i}] lacks a descriptor")));
        }
    }
    if !RL_PUBLICATION_PHASES.contains(&publication_phase) {
        return Err(rt_err(format!(
            "control publication_phase {publication_phase:?} unknown"
        )));
    }
    let statuses: Value = parse_json(generation_statuses_json, "statuses")?;
    let statuses_map = statuses
        .as_object()
        .ok_or_else(|| rt_err("control generation_statuses must be an object"))?;
    for (key, value) in statuses_map {
        match value.as_str() {
            Some("collected") | Some("sealed") | Some("trained") | Some("published") => {}
            other => {
                return Err(rt_err(format!(
                    "control generation_statuses[{key:?}] must name a status, got {other:?}"
                )));
            }
        }
    }
    for (name, doc) in [("source_refs", source_refs_json), ("ledger_refs", ledger_refs_json)] {
        let refs: Value = parse_json(doc, name)?;
        let refs_map = refs
            .as_object()
            .ok_or_else(|| rt_err(format!("control {name} must be an object")))?;
        for (key, value) in refs_map {
            match value.as_str() {
                Some(text) if is_digest_text(text) => {}
                _ => {
                    return Err(rt_err(format!("control {name}[{key:?}] must be a digest")));
                }
            }
        }
    }
    let mut control = Map::with_capacity(25);
    control.insert("schema_version".to_string(), Value::String("1.0.0".to_string()));
    control.insert(
        "pipeline_spec_id".to_string(),
        Value::String(descriptor.pipeline_spec_id.clone()),
    );
    control.insert(
        "learner_manifest_id".to_string(),
        Value::String(learner_manifest.to_string()),
    );
    control.insert(
        "learner_transport_sha".to_string(),
        Value::String(learner_transport.to_string()),
    );
    control.insert(
        "learner_locator".to_string(),
        Value::String(learner_locator.to_string()),
    );
    control.insert(
        "actor_manifest_id".to_string(),
        Value::String(actor_manifest.to_string()),
    );
    control.insert(
        "actor_transport_sha".to_string(),
        Value::String(actor_transport.to_string()),
    );
    control.insert(
        "actor_locator".to_string(),
        Value::String(actor_locator.to_string()),
    );
    control.insert("generation".to_string(), Value::from(state.generation));
    control.insert("generation_statuses".to_string(), statuses);
    control.insert(
        "source_refs".to_string(),
        parse_json(source_refs_json, "source_refs")?,
    );
    control.insert(
        "order_ref".to_string(),
        order.map(Value::String).unwrap_or(Value::Null),
    );
    control.insert(
        "ledger_refs".to_string(),
        parse_json(ledger_refs_json, "ledger_refs")?,
    );
    control.insert(
        "native_snapshot_digest".to_string(),
        Value::String(state.current.pool_ref.digest.clone()),
    );
    control.insert(
        "tape_blob_digest".to_string(),
        tape.map(Value::String).unwrap_or(Value::Null),
    );
    control.insert(
        "event_blob_digest".to_string(),
        event.map(Value::String).unwrap_or(Value::Null),
    );
    control.insert(
        "cohort_membership".to_string(),
        serde_json::json!({
            "generations": descriptor.generations,
            "games_per_generation": descriptor.games_per_generation,
        }),
    );
    control.insert("frontier_current".to_string(), frontier_current);
    control.insert("frontier_next".to_string(), frontier_next);
    control.insert(
        "terminal_set".to_string(),
        Value::Array(terminal_set.iter().map(|w| Value::String(w.clone())).collect()),
    );
    control.insert("tickets".to_string(), tickets);
    control.insert(
        "committed_row_offset".to_string(),
        Value::from(committed_row_offset),
    );
    control.insert(
        "committed_update_offset".to_string(),
        Value::from(committed_update_offset),
    );
    control.insert(
        "publication_phase".to_string(),
        Value::String(publication_phase.to_string()),
    );
    control.insert(
        "parent_checkpoint_hash".to_string(),
        Value::String(state.generation_initial_id.clone()),
    );
    canon_bytes(&Value::Object(control), "rl:control-doc")
}

/// Commit one runtime stop: validate, verify, assemble the exact
/// registered control, publish, advance the runtime pointer. Returns the
/// committed control tip bytes.
#[allow(clippy::too_many_arguments)]
pub fn runtime_control(
    root: &Path,
    descriptor_json: &[u8],
    bindings_json_or_null: Option<&[u8]>,
    learner_receipt_json: &[u8],
    actor_receipt_json: &[u8],
    runtime_state_json: &[u8],
    expected_parent: Option<&str>,
) -> Result<Vec<u8>, TapeError> {
    let descriptor_value = parse_json(descriptor_json, "run descriptor")?;
    let mut descriptor = parse_descriptor(descriptor_json)?;
    let state = validate_runtime_state(runtime_state_json)?;
    if state.generation >= descriptor.generations {
        return Err(rt_err(format!(
            "state generation {} outside 0..{}",
            state.generation, descriptor.generations
        )));
    }
    descriptor.source_digests = descriptor_sources(&descriptor_value, state.generation)?;
    // Fixed parent: every snapshot of the generation names the generation
    // init id; an explicit expectation must agree (never silently reset).
    if let Some(want) = expected_parent {
        if want != state.generation_initial_id {
            return Err(rt_err("expected parent is not the generation-init id"));
        }
    }
    // Receipts are verified against disk BEFORE any control assembly; the
    // returned manifest ids drive every downstream binding check.
    let learner_manifest = verify_receipt_against_disk(root, learner_receipt_json)
        .map_err(|e| rt_err(format!("learner receipt diverges: {e}")))?;
    let actor_manifest = verify_receipt_against_disk(root, actor_receipt_json)
        .map_err(|e| rt_err(format!("actor receipt diverges: {e}")))?;
    let (_, learner_locator) = receipt_manifest_id(learner_receipt_json, "learner")?;
    let (_, actor_locator) = receipt_manifest_id(actor_receipt_json, "actor")?;
    // Initial C0 stop: both roles are the original IL receipt (fresh Torch
    // optimizer derivation, no fabricated RL source). Later stops keep the
    // learner at the fixed generation parent while the actor is behavior.
    if state.generation == 0
        && state.batch_cursor == 0
        && state.completed_updates == 0
        && (learner_manifest != descriptor.initial_checkpoint_id
            || actor_manifest != descriptor.initial_checkpoint_id)
    {
        return Err(rt_err("initial cold stop must present the original C0 receipt"));
    }
    // Actual latest receipts: the control projects the verified manifests
    // + locators as committed (ongoing learner Cx, pending actor Cn), while
    // the fixed generation parent and the RunSpec stay untouched — state
    // generation_initial_id and bindings are read-only below, never mutated.
    // The cold invariant still holds: with zero progress both roles are C0.
    // Sealed vs cold: a sealed current carries the full frozen unit and
    // commits through the lifecycle publisher; a cold current carries the
    // actual pool only (initial collection, nothing sealed or fabricated)
    // and commits a pool-bound control with null order/tape/event slots.
    let sealed = state.current.receipt.is_some();
    // Bindings (when Main passes them) are cross-checked read-only against
    // the verified state and descriptor; they are never mutated and never
    // required — the control carries the actual receipts either way. The
    // cold path has no order to bind and always passes null.
    if let Some(doc) = bindings_json_or_null {
        let view = parse_bindings(doc)?;
        if view.generation != state.generation {
            return Err(rt_err("bindings generation diverges from state"));
        }
        if view.pipeline_spec_id != descriptor.pipeline_spec_id {
            return Err(rt_err("bindings pipeline diverges from the descriptor"));
        }
        if view.learner_init != state.generation_initial_id {
            return Err(rt_err("bindings learner parent diverges from state"));
        }
        if view.actor_id != state.behavior_id {
            return Err(rt_err("bindings actor diverges from state"));
        }
        if sealed {
            let order_digest = state
                .current
                .order
                .as_ref()
                .and_then(|o| o.get("order_digest"))
                .and_then(Value::as_str)
                .ok_or_else(|| rt_err("state current order lacks order_digest"))?;
            if view.order_digest != order_digest {
                return Err(rt_err("bindings order diverges from state"));
            }
        } else {
            return Err(rt_err("cold stop carries no bindings (nothing sealed yet)"));
        }
    }
    // Every referenced blob is digest-verified before control assembly.
    // The sealed path verifies the full frozen unit; the cold path
    // verifies the actual pool only (re-derivation with equality).
    let (event_pair, mut current_pool_walls, current_tape_walls): (
        Option<(String, String)>,
        Vec<String>,
        Vec<String>,
    ) = if sealed {
        let (event_ref, _, _, _, pool_walls, tape_walls) =
            verify_side_blobs(root, &state.current, "state.current")?;
        (
            Some((event_ref.digest, event_ref.locator)),
            pool_walls,
            tape_walls,
        )
    } else {
        let verified_pool = check_pool_ref(
            root,
            &state.current.pool_ref.digest,
            &state.current.pool_ref.locator,
            &state.current.pool_ref.frontier,
            &state.current.pool_ref.terminal_set,
            &state.current.pool_ref.tickets,
            state.current.pool_ref.captured_rows,
        )?;
        (None, verified_pool.wall_ids, Vec::new())
    };
    if let Some(future) = state.future.as_ref() {
        // Future pool progress is verified (READY retained, no resample);
        // future sealed bindings verify only when the future has sealed.
        check_pool_ref(
            root,
            &future.pool_ref.digest,
            &future.pool_ref.locator,
            &future.pool_ref.frontier,
            &future.pool_ref.terminal_set,
            &future.pool_ref.tickets,
            future.pool_ref.captured_rows,
        )?;
        if future.receipt.is_some() {
            verify_side_blobs(root, future, "state.future")?;
        } else if future.order.is_some() || future.index_ref.is_some() || future.tape_ref.is_some() {
            return Err(rt_err("state.future binds order/index/tape without a sealed receipt"));
        }
    }
    // Tape walls must come from the generation source ledger (actual
    // ledger, no ad-hoc identities). Sealed path only; the cold path has
    // no tape yet. Walls ride along from the verified tape bytes above, so
    // this gate never re-reads the 400MB blob.
    if sealed {
        let ledger: std::collections::HashSet<&str> = descriptor
            .source_digests
            .iter()
            .map(|(wall, _)| wall.as_str())
            .collect();
        for wall in &current_tape_walls {
            if !ledger.contains(wall.as_str()) {
                return Err(rt_err(format!("tape wall {wall:?} escapes the source ledger")));
            }
        }
    }
    // Pool cohort walls must equal the collector profile cohort (set
    // equality against the actual spec, not a re-guess). Walls ride along
    // from the verified pool above, so this gate never re-reads it.
    let spec_games = state
        .current
        .spec
        .get("profile")
        .and_then(|p| p.get("games"))
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("state spec profile lacks games"))?;
    let mut spec_walls: Vec<&str> = spec_games
        .iter()
        .filter_map(|g| g.get("wall_id").and_then(Value::as_str))
        .collect();
    spec_walls.sort_unstable();
    current_pool_walls.sort_unstable();
    let pool_walls: Vec<&str> = current_pool_walls.iter().map(String::as_str).collect();
    if pool_walls != spec_walls {
        return Err(rt_err("pool cohort walls diverge from the collector spec"));
    }
    // Control inputs: pool facts re-derived at save time (equality already
    // enforced above); learner progress from the true cursors.
    let frontier_current = serde_json::to_string(&state.current.pool_ref.frontier)
        .map_err(|e| rt_err(format!("frontier encode failed: {e}")))?;
    let frontier_next = match state.future.as_ref() {
        Some(future) => serde_json::to_string(&future.pool_ref.frontier)
            .map_err(|e| rt_err(format!("frontier encode failed: {e}")))?,
        None => frontier_current.clone(),
    };
    let mut tickets = state
        .current
        .pool_ref
        .tickets
        .as_array()
        .cloned()
        .unwrap_or_default();
    if let Some(future) = state.future.as_ref() {
        tickets.extend(
            future
                .pool_ref
                .tickets
                .as_array()
                .cloned()
                .unwrap_or_default(),
        );
    }
    tickets.sort_by(|a, b| {
        a.get("ticket_descriptor")
            .and_then(Value::as_str)
            .cmp(&b.get("ticket_descriptor").and_then(Value::as_str))
    });
    let tickets_json = canon_bytes(&Value::Array(tickets), "rl:tickets")?;
    let mut statuses = Map::with_capacity(2);
    statuses.insert(
        state.generation.to_string(),
        Value::String(status_for_phase(&state.publication_phase).to_string()),
    );
    if let Some(future) = state.future.as_ref() {
        if future.pool_ref.captured_rows > 0 {
            let fgen = future
                .spec
                .get("generation")
                .and_then(Value::as_u64)
                .unwrap_or(state.generation + 1);
            statuses.insert(fgen.to_string(), Value::String("collected".to_string()));
        }
    }
    let generation_statuses_json = canon_bytes(&Value::Object(statuses), "rl:statuses")?;
    let mut source_refs = Map::with_capacity(descriptor.source_digests.len());
    for (wall, digest) in &descriptor.source_digests {
        source_refs.insert(wall.clone(), Value::String(digest.clone()));
    }
    let source_refs_json = canon_bytes(&Value::Object(source_refs), "rl:source-refs")?;
    let mut ledger_refs = Map::new();
    if let Some(hash) = descriptor.wall_ledger_hash.as_ref() {
        ledger_refs.insert("wall_ledger".to_string(), Value::String(hash.clone()));
    }
    if let Some(hash) = descriptor.row_derivation_hash.as_ref() {
        ledger_refs.insert("row_derivation".to_string(), Value::String(hash.clone()));
    }
    let ledger_refs_json = canon_bytes(&Value::Object(ledger_refs), "rl:ledger-refs")?;
    // Control formation: one projector for both paths. The control names
    // the ACTUAL latest receipts (verified manifests + locators) while the
    // fixed generation parent, cohort, and order stay read-only bindings;
    // the cold path binds the actual pool with null order/tape/event slots
    // (never fabricated digests). The content digest the publisher
    // verifies is attached after assembly.
    let learner_transport =
        receipt_field(learner_receipt_json, "learner receipt", "transport_sha256")?;
    let actor_transport =
        receipt_field(actor_receipt_json, "actor receipt", "transport_sha256")?;
    let (order_ref, tape_value, event_value) = if sealed {
        let tape_ref = state.current.tape_ref.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks tape_ref")
        })?;
        let (event_digest, _) = event_pair.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks verified event blob")
        })?;
        let order_digest = state
            .current
            .order
            .as_ref()
            .and_then(|o| o.get("order_digest"))
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err("sealed current lacks order_digest"))?;
        (
            Value::String(order_digest.to_string()),
            Value::String(tape_ref.digest.clone()),
            Value::String(event_digest.clone()),
        )
    } else {
        (Value::Null, Value::Null, Value::Null)
    };
    let control_nodigest = assemble_projected_control(
        &descriptor,
        &state,
        &learner_manifest,
        &learner_transport,
        &learner_locator,
        &actor_manifest,
        &actor_transport,
        &actor_locator,
        order_ref,
        tape_value,
        event_value,
        &frontier_current,
        &frontier_next,
        &state.current.pool_ref.terminal_set,
        &tickets_json,
        state.current.pool_ref.captured_rows,
        state.completed_updates,
        &state.publication_phase,
        &generation_statuses_json,
        &source_refs_json,
        &ledger_refs_json,
    )?;
    let control_doc = attach_control_digest(&control_nodigest)?;
    // Persist the validated state + both ordinary receipts as CAS blobs,
    // then publish (the publisher re-verifies receipts and every listed
    // blob against disk; unreferenced files stay orphans, never commits).
    let state_receipt = write_runtime_blob(root, runtime_state_json)?;
    let state_ref: Value = parse_json(&state_receipt, "state receipt")?;
    let learner_blob = write_runtime_blob(root, learner_receipt_json)?;
    let learner_ref: Value = parse_json(&learner_blob, "learner receipt blob")?;
    let actor_blob = write_runtime_blob(root, actor_receipt_json)?;
    let actor_ref: Value = parse_json(&actor_blob, "actor receipt blob")?;
    let str_of = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| rt_err(format!("receipt blob lacks {key}")))
    };
    let mut blobs = vec![
        (
            state.current.pool_ref.digest.clone(),
            state.current.pool_ref.locator.clone(),
        ),
        (
            str_of(&state_ref, "digest")?,
            str_of(&state_ref, "locator")?,
        ),
        (
            str_of(&learner_ref, "digest")?,
            str_of(&learner_ref, "locator")?,
        ),
        (
            str_of(&actor_ref, "digest")?,
            str_of(&actor_ref, "locator")?,
        ),
    ];
    if sealed {
        let tape_ref = state.current.tape_ref.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks tape_ref")
        })?;
        let index_ref = state.current.index_ref.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks index_ref")
        })?;
        let (event_digest, event_locator) = event_pair.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks verified event blob")
        })?;
        blobs.push((tape_ref.digest.clone(), tape_ref.locator.clone()));
        blobs.push((event_digest.clone(), event_locator.clone()));
        blobs.push((index_ref.digest.clone(), index_ref.locator.clone()));
    }
    if let Some(future) = state.future.as_ref() {
        blobs.push((future.pool_ref.digest.clone(), future.pool_ref.locator.clone()));
        if let (Some(index), Some(tape)) = (future.index_ref.as_ref(), future.tape_ref.as_ref()) {
            blobs.push((index.digest.clone(), index.locator.clone()));
            blobs.push((tape.digest.clone(), tape.locator.clone()));
        }
    }
    let receipts = vec![learner_receipt_json.to_vec(), actor_receipt_json.to_vec()];
    let tip = crate::rl::snapshot::publish_control(
        root,
        &control_doc,
        &receipts,
        Some(&state.generation_initial_id),
        &blobs,
    )
    .map_err(|e| rt_err(format!("control publish rejected: {e}")))?;
    // The tip carries the control locator; persist the runtime pointer
    // binding state + control + receipt blobs (last-committed authority
    // for restore, cross-checked against the control tip).
    let tip_value: Value = parse_json(&tip, "control tip")?;
    let control_digest = tip_value
        .get("control_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("control tip lacks control_digest"))?
        .to_string();
    let control_locator = tip_value
        .get("control_locator")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("control tip lacks control_locator"))?
        .to_string();
    write_runtime_pointer(
        root,
        &str_of(&state_ref, "digest")?,
        &str_of(&state_ref, "locator")?,
        &control_digest,
        &control_locator,
        state.generation,
        &state.publication_phase,
        &str_of(&learner_ref, "locator")?,
        &str_of(&actor_ref, "locator")?,
    )?;
    Ok(tip)
}

/// Verified resume: last-committed runtime pointer cross-checked against
/// the control tip, then every receipt re-verified against disk and every
/// referenced blob re-verified before any handle or locator is returned.
/// Returns canonical bytes (`schema`, `generation`, `phase`,
/// `pipeline_spec_id`, `state`, `control`, `learner`, `actor`, `current`,
/// `future`) with pool/tape/index locators and live tape handles.
pub fn restore_runtime(root: &Path, descriptor_json: &[u8]) -> Result<Vec<u8>, TapeError> {
    let _descriptor_value = parse_json(descriptor_json, "run descriptor")?;
    let descriptor = parse_descriptor(descriptor_json)?;
    let control_bytes = read_tip(root).map_err(|e| rt_err(format!("control tip unreadable: {e}")))?;
    let control: Value = parse_json(&control_bytes, "committed control")?;
    let cmap = control
        .as_object()
        .ok_or_else(|| rt_err("committed control must be an object"))?;
    // Recompute the control digest (verified, never trusted).
    let (control_digest, _) =
        crate::rl::config::identity_of_payload_without_digest(&control)
            .map_err(|e| rt_err(format!("control digest recompute failed: {e}")))?;
    let control_generation = cmap.get("generation").and_then(Value::as_u64).ok_or_else(|| {
        rt_err("control lacks generation")
    })?;
    let control_phase = cmap
        .get("publication_phase")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("control lacks publication_phase"))?
        .to_string();
    let control_pipeline = cmap
        .get("pipeline_spec_id")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("control lacks pipeline_spec_id"))?;
    if control_pipeline != descriptor.pipeline_spec_id {
        return Err(rt_err("control pipeline diverges from the descriptor"));
    }
    let (state_digest, state_locator, pointer_control, pointer_generation, pointer_phase, learner_blob_locator, actor_blob_locator) =
        read_runtime_pointer(root)?;
    if pointer_control != control_digest
        || pointer_generation != control_generation
        || pointer_phase != control_phase
    {
        return Err(rt_err(
            "runtime pointer diverges from the committed control tip",
        ));
    }
    let state_bytes = read_verified_blob(root, &state_digest, &state_locator)?;
    let state = validate_runtime_state(&state_bytes)?;
    if state.generation != control_generation || state.publication_phase != control_phase {
        return Err(rt_err("runtime state diverges from the committed control"));
    }
    // Control linkage: every digest field must name the verified state.
    // Null order/tape/event slots are legal only for an unsealed current
    // (cold collection); a sealed current must name every frozen binding.
    let sealed = state.current.receipt.is_some();
    let ctl_str = |key: &str| -> Result<String, TapeError> {
        cmap.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| rt_err(format!("control lacks {key}")))
    };
    let ctl = |key: &str| cmap.get(key).cloned().unwrap_or(Value::Null);
    if ctl_str("native_snapshot_digest")? != state.current.pool_ref.digest {
        return Err(rt_err("control native snapshot diverges from state"));
    }
    let state_order = state
        .current
        .order
        .as_ref()
        .and_then(|o| o.get("order_digest"))
        .and_then(Value::as_str);
    let state_tape_digest = state.current.tape_ref.as_ref().map(|t| t.digest.as_str());
    if sealed {
        if ctl("order_ref").as_str() != state_order
            || ctl("tape_blob_digest").as_str() != state_tape_digest
        {
            return Err(rt_err("control order/tape diverges from state"));
        }
    } else if ctl("order_ref") != Value::Null
        || ctl("tape_blob_digest") != Value::Null
        || ctl("event_blob_digest") != Value::Null
    {
        return Err(rt_err("cold control must null its order/tape/event slots"));
    }
    // Fixed parent stays pinned; learner/actor name the actual latest
    // receipts (checked against the disk-verified receipts below, not
    // against the fixed ids — ongoing Cx and pending Cn differ by design).
    if ctl_str("parent_checkpoint_hash")? != state.generation_initial_id {
        return Err(rt_err("control parent diverges from state"));
    }
    if cmap.get("committed_row_offset").and_then(Value::as_u64)
        != Some(state.current.pool_ref.captured_rows)
        || cmap.get("committed_update_offset").and_then(Value::as_u64) != Some(state.completed_updates)
    {
        return Err(rt_err("control offsets diverge from state"));
    }
    // Receipts: CAS blobs named by the pointer, self-verified (filename
    // hash equals content hash) and re-verified against disk. All
    // verification completes before any handle or locator is returned,
    // so Main never mutates Torch state on unverified inputs.
    let learner_doc = read_cas_named_blob(root, &learner_blob_locator, "learner receipt")?;
    let actor_doc = read_cas_named_blob(root, &actor_blob_locator, "actor receipt")?;
    let learner_back = verify_receipt_against_disk(root, &learner_doc)
        .map_err(|e| rt_err(format!("learner receipt diverges: {e}")))?;
    let actor_back = verify_receipt_against_disk(root, &actor_doc)
        .map_err(|e| rt_err(format!("actor receipt diverges: {e}")))?;
    if learner_back != ctl_str("learner_manifest_id")?
        || actor_back != ctl_str("actor_manifest_id")?
    {
        return Err(rt_err("receipt manifests diverge from control"));
    }
    // Blob chain: index order linkage, tape counts + event digest, pool
    // re-derivation — sealed sides fully, cold sides pool-only.
    // Returns (tape_doc, index_doc, artifact, order, rows) with Nulls for
    // an unsealed current (no handle to register, nothing to stamp).
    let (current_tape, current_index, artifact_doc, order_doc, rows_doc) = if sealed {
        let (event_ref, artifact, order, rows, _, _) =
            verify_side_blobs(root, &state.current, "state.current")?;
        if ctl("event_blob_digest").as_str() != Some(event_ref.digest.as_str()) {
            return Err(rt_err("control event digest diverges from the tape blob"));
        }
        let tape_ref = state.current.tape_ref.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks tape_ref")
        })?;
        let tape_ref_doc = canon_bytes(
            &serde_json::json!({
                "digest": tape_ref.digest,
                "locator": tape_ref.locator,
                "row_count": tape_ref.row_count,
                "game_count": tape_ref.game_count,
            }),
            "rl:tape-ref",
        )?;
        let handle = restore_tape(root, &tape_ref_doc)?;
        crate::rl::trajectory::stamp_tape_digest(handle, &artifact)
            .map_err(|e| rt_err(format!("tape stamp rejected: {e}")))?;
        let index_ref = state.current.index_ref.as_ref().ok_or_else(|| {
            rt_err("sealed current lacks index_ref")
        })?;
        (
            serde_json::json!({
                "digest": tape_ref.digest,
                "locator": tape_ref.locator,
                "handle": handle,
                "row_count": rows,
            }),
            serde_json::json!({"digest": index_ref.digest, "locator": index_ref.locator}),
            Value::String(artifact),
            Value::String(order),
            Value::from(rows),
        )
    } else {
        (
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
        )
    };
    // Full-shape pool refs (the exact `save_pool` objects) verify on the
    // way out; Main feeds the returned bytes to `PyNativePool.restore`.
    let full_pool_ref = |digest: &str,
                         locator: &str,
                         frontier: &Value,
                         terminal_set: &[String],
                         tickets: &Value,
                         captured_rows: u64|
     -> Result<Vec<u8>, TapeError> {
        canon_bytes(
            &serde_json::json!({
                "digest": digest,
                "locator": locator,
                "frontier": frontier,
                "terminal_set": terminal_set,
                "tickets": tickets,
                "captured_rows": captured_rows,
            }),
            "rl:pool-ref",
        )
    };
    let mut future_doc = Value::Null;
    if let Some(future) = state.future.as_ref() {
        check_pool_ref(
            root,
            &future.pool_ref.digest,
            &future.pool_ref.locator,
            &future.pool_ref.frontier,
            &future.pool_ref.terminal_set,
            &future.pool_ref.tickets,
            future.pool_ref.captured_rows,
        )?;
        if future.receipt.is_some() {
            verify_side_blobs(root, future, "state.future")?;
        }
        let full = full_pool_ref(
            &future.pool_ref.digest,
            &future.pool_ref.locator,
            &future.pool_ref.frontier,
            &future.pool_ref.terminal_set,
            &future.pool_ref.tickets,
            future.pool_ref.captured_rows,
        )?;
        let fpool = restore_pool_doc(root, &full)?;
        let _ = fpool;
        future_doc = serde_json::json!({
            "pool": {"digest": future.pool_ref.digest, "locator": future.pool_ref.locator},
            "tape": future.tape_ref.as_ref().map(|t| serde_json::json!({"digest": t.digest, "locator": t.locator, "row_count": t.row_count, "game_count": t.game_count})).unwrap_or(Value::Null),
            "index": future.index_ref.as_ref().map(|t| serde_json::json!({"digest": t.digest, "locator": t.locator})).unwrap_or(Value::Null),
        });
    }
    let full_current = full_pool_ref(
        &state.current.pool_ref.digest,
        &state.current.pool_ref.locator,
        &state.current.pool_ref.frontier,
        &state.current.pool_ref.terminal_set,
        &state.current.pool_ref.tickets,
        state.current.pool_ref.captured_rows,
    )?;
    let current_pool_doc = restore_pool_doc(root, &full_current)?;
    let _ = current_pool_doc;
    let out = serde_json::json!({
        "schema": "rl.runtime_restore/v1",
        "schema_version": "1.0.0",
        "generation": state.generation,
        "phase": state.publication_phase,
        "pipeline_spec_id": descriptor.pipeline_spec_id,
        "batch_cursor": state.batch_cursor,
        "completed_updates": state.completed_updates,
        "behavior_id": state.behavior_id,
        "generation_initial_id": state.generation_initial_id,
        "state": {"digest": state_digest, "locator": state_locator},
        "control": {"digest": control_digest, "locator": control_locator_of(root, &control_digest)?},
        "learner": {"manifest_id": learner_back, "locator": ctl_str("learner_locator")?, "receipt_locator": learner_blob_locator},
        "actor": {"manifest_id": actor_back, "locator": ctl_str("actor_locator")?, "receipt_locator": actor_blob_locator},
        "current": {
            "pool": {"digest": state.current.pool_ref.digest, "locator": state.current.pool_ref.locator},
            "tape": current_tape,
            "index": current_index,
            "artifact_digest": artifact_doc,
            "order_digest": order_doc,
            "row_count": rows_doc,
        },
        "future": future_doc,
    });
    canon_bytes(&out, "rl:runtime-restore")
}


/// Read a CAS blob whose locator filename carries its own digest
/// (`runtime/blob_<hex>.json`): the content hash must equal the filename
/// hash (self-verifying even when no expected digest is on hand).
fn read_cas_named_blob(root: &Path, locator: &str, what: &str) -> Result<Vec<u8>, TapeError> {
    let path = crate::rl::snapshot::join_root(root, locator)?;
    let bytes =
        std::fs::read(&path).map_err(|e| rt_err(format!("{what} blob unreadable: {e}")))?;
    let file = locator.rsplit('/').next().unwrap_or(locator);
    let hex = file
        .strip_prefix("blob_")
        .and_then(|s| s.strip_suffix(".json"))
        .ok_or_else(|| rt_err(format!("{what} locator is not a CAS blob path")))?;
    let actual = crate::rl::runtime_snapshot::sha_hex(&bytes);
    if actual != hex {
        return Err(rt_err(format!("{what} blob diverges from its CAS name")));
    }
    Ok(bytes)
}

fn control_locator_of(root: &Path, control_digest: &str) -> Result<String, TapeError> {
    let tip = read_tip(root).map_err(|e| rt_err(format!("control tip unreadable: {e}")))?;
    let _: Value = parse_json(&tip, "committed control")?;
    let raw = crate::rl::trajectory::parse_digest_text(control_digest)
        .map_err(|e| rt_err(format!("control digest malformed: {e}")))?;
    Ok(format!(
        "control/control_{}.json",
        crate::rl::trajectory::hex_of(&raw)
    ))
}

/// Persist an arbitrary canonical runtime document (state, index, spec
/// projections) — thin alias Main uses for index bytes.
pub fn persist_runtime_doc(root: &Path, document: &[u8]) -> Result<Vec<u8>, TapeError> {
    write_runtime_blob(root, document)
}

/// Stage an atomic file write for caller-owned bytes (pointer-adjacent
/// tooling; CAS blobs always go through `write_runtime_blob`).
pub fn atomic_stage(root: &Path, locator: &str, bytes: &[u8]) -> Result<(), TapeError> {
    let path = crate::rl::snapshot::join_root(root, locator)?;
    atomic_write(&path, bytes)
}
