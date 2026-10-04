//! rl/live: native-live training source (pipeline v3, no archival coupling).
//!
//! DAG: `super::lifecycle_collector` (shared profile/context/shuffle-key
//! construction, single home) + `super::lifecycle_common` (gates, digest,
//! canonical bytes) + serde_json + `hydra_feed::canon`. FORBIDS: PyO3
//! (lives in `live_ffi.rs`), wall-clock, RNG seeding, observation rebuilds.
//!
//! `policy_identity` mints the deterministic snapshot-lineage identifier for
//! the frozen behavior used to collect generation g (NOT a weight-content
//! hash or ordinary checkpoint manifest id): g0 is the initial checkpoint
//! id; g1..7 hash the exact lineage object through the shared canonical
//! printer. `live_collector_spec` emits the archive-free collector binding
//! (profile/context/shuffle-key only, no header/locators/ledgers).

use serde_json::Value;

use super::lifecycle_common::{
    canonical_digest, descriptor_doc, descriptor_str, is_digest_shape, lc_err, req_u64,
    LifecycleError,
};

/// Lineage envelope schema bound into every g1..7 policy identity.
pub const POLICY_LINEAGE_SCHEMA: &str = "hydra2.rl_policy_lineage/v1";

/// Live collector envelope schema emitted by [`live_collector_spec`].
pub const LIVE_COLLECTOR_SCHEMA: &str = "hydra2.rl_live_collector/v1";

/// Fixed capture-update marker bound into the lineage object: the actor used
/// to collect g is frozen at the first completed optimizer update of g-1, so
/// every lineage identity in this profile carries update 1 by construction.
const CAPTURE_UPDATE: u64 = 1;

/// Deterministic snapshot-lineage identifier for the frozen behavior actor
/// that collects `generation`.
///
/// g0 returns the descriptor's initial checkpoint id and requires
/// `parent_policy_id` to equal it. g1..7 return the shared-printer digest of
/// the exact object `{schema, pipeline_spec_id, initial_checkpoint_id,
/// generation, parent_policy_id, capture_update: 1}`. Out-of-range
/// generations and malformed/missing inputs reject. The result binds the
/// pipeline/derivation recipe through `pipeline_spec_id`, so the v3 cutover
/// intentionally moves every g>0 identity (and its shuffle stream) versus
/// the old profile. This is a lineage identifier, never a weight hash.
pub fn policy_identity(
    run_descriptor_json: &[u8],
    generation: u64,
    parent_policy_id: &str,
) -> Result<String, LifecycleError> {
    if !is_digest_shape(parent_policy_id) {
        return Err(lc_err(format!(
            "rl live parent_policy_id must be sha256:<64 hex>, got {parent_policy_id:?}"
        )));
    }
    let descriptor: Value = serde_json::from_slice(run_descriptor_json)
        .map_err(|e| lc_err(format!("rl live run descriptor not JSON: {e}")))?;
    let doc = descriptor
        .as_object()
        .ok_or_else(|| lc_err("rl live run descriptor must be an object"))?;
    let cohort = descriptor_doc(doc, "cohort")?;
    let generations_total = req_u64(cohort.get("generations"), "cohort.generations")?;
    if generation >= generations_total {
        return Err(lc_err(format!(
            "rl live generation {generation} outside 0..{generations_total}"
        )));
    }
    if generation > 7 {
        return Err(lc_err(format!(
            "rl live generation {generation} outside the 0..8 live profile"
        )));
    }
    let initial_id = descriptor_str(doc, "initial_checkpoint_id")?;
    if !is_digest_shape(&initial_id) {
        return Err(lc_err(
            "rl live descriptor initial_checkpoint_id must be sha256:<64 hex>",
        ));
    }
    if generation == 0 {
        if parent_policy_id != initial_id {
            return Err(lc_err(
                "rl live g0 parent must be the initial checkpoint (all-seat C0 selfplay)",
            ));
        }
        return Ok(initial_id);
    }
    let pipeline_spec_id = descriptor_str(doc, "pipeline_spec_id")?;
    let lineage = serde_json::json!({
        "schema": POLICY_LINEAGE_SCHEMA,
        "pipeline_spec_id": pipeline_spec_id,
        "initial_checkpoint_id": initial_id,
        "generation": generation,
        "parent_policy_id": parent_policy_id,
        "capture_update": CAPTURE_UPDATE,
    });
    let digest = canonical_digest(&lineage, "rl:policy-lineage")?;
    if !is_digest_shape(&digest) {
        return Err(lc_err(
            "rl live policy lineage digest missed digest shape (printer drift)",
        ));
    }
    Ok(digest)
}

/// Archive-free collector binding for one generation: profile, context, and
/// frozen shuffle key bound to the real behavior policy id, with no
/// artifact header, locators, or ledgers. Profile/context/shuffle-key bytes
/// come from the single shared constructor in `lifecycle_collector`, so the
/// live entry can never drift from the archive entry on valid inputs.
/// Generation membership, g0 identity, cohort bindings, and training/eval
/// wall separation are validated once inside that shared construction.
pub fn live_collector_spec(
    run_descriptor_json: &[u8],
    generation: u64,
    policy_id: &str,
) -> Result<Vec<u8>, LifecycleError> {
    let core =
        super::lifecycle_collector::collector_core(run_descriptor_json, generation, policy_id)?;
    let output = serde_json::json!({
        "schema": LIVE_COLLECTOR_SCHEMA,
        "generation": generation,
        "pool_id": core.pool_id,
        "policy_id": policy_id,
        "profile": core.profile,
        "context": core.context,
        "shuffle_key": core.shuffle_key,
    });
    hydra_feed::canon::canonical_bytes_value(&output, "rl:live-collector-spec")
        .map_err(|e| lc_err(format!("rl live collector spec not canonical-safe: {e}")))
}
/// Generation restart descriptor schema (atomic learner/actor pairing).
pub const RESTART_DESCRIPTOR_SCHEMA: &str = "hydra2.rl_restart_descriptor/v1";

/// Generation restart pointer schema (mutable tip selecting one container).
pub const RESTART_POINTER_SCHEMA: &str = "hydra2.rl_restart_pointer/v1";

/// Validate a generation restart descriptor against the run descriptor:
/// closed 8-key shape, pipeline/C0/next-generation/completed lineage, and
/// the expected behavior policy derived from the previous id (Cg for
/// unfinished starts 1..7, C7 for the completed start 8, C0 for start 0).
/// Tensor state is never hashed here; fixed metadata is validated once on
/// save/load. Returns the canonical descriptor bytes.
pub fn validate_restart_descriptor(
    run_descriptor_json: &[u8],
    descriptor_json: &[u8],
) -> Result<Vec<u8>, LifecycleError> {
    let descriptor: Value = serde_json::from_slice(descriptor_json)
        .map_err(|e| lc_err(format!("rl restart descriptor not JSON: {e}")))?;
    let doc = descriptor
        .as_object()
        .ok_or_else(|| lc_err("rl restart descriptor must be an object"))?;
    const KEYS: [&str; 8] = [
        "schema",
        "pipeline_spec_id",
        "initial_checkpoint_id",
        "next_generation",
        "completed_updates",
        "behavior_policy_id",
        "previous_behavior_policy_id",
        "completed",
    ];
    if doc.len() != KEYS.len() || KEYS.iter().any(|k| !doc.contains_key(*k)) {
        return Err(lc_err(
            "rl restart descriptor must carry exactly the 8 restart fields",
        ));
    }
    if doc.get("schema").and_then(|v| v.as_str()) != Some(RESTART_DESCRIPTOR_SCHEMA) {
        return Err(lc_err("rl restart descriptor schema mismatch".to_string()));
    }
    let run: Value = serde_json::from_slice(run_descriptor_json)
        .map_err(|e| lc_err(format!("rl restart run descriptor not JSON: {e}")))?;
    let run_doc = run
        .as_object()
        .ok_or_else(|| lc_err("rl restart run descriptor must be an object"))?;
    let run_str = |name: &str| {
        run_doc
            .get(name)
            .and_then(|v| v.as_str())
            .ok_or_else(|| lc_err(format!("rl restart run descriptor lacks {name}")))
    };
    let get_str = |name: &str| {
        doc.get(name)
            .and_then(|v| v.as_str())
            .ok_or_else(|| lc_err(format!("rl restart descriptor lacks {name}")))
    };
    let pipeline_spec_id = get_str("pipeline_spec_id")?;
    if pipeline_spec_id != run_str("pipeline_spec_id")? {
        return Err(lc_err(
            "rl restart descriptor pipeline mismatches run descriptor".to_string(),
        ));
    }
    let initial_id = get_str("initial_checkpoint_id")?;
    if initial_id != run_str("initial_checkpoint_id")? {
        return Err(lc_err(
            "rl restart descriptor initial checkpoint mismatches run descriptor".to_string(),
        ));
    }
    let next_generation = doc
        .get("next_generation")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| lc_err("rl restart next_generation must be a non-bool integer"))?;
    if next_generation > 8 {
        return Err(lc_err(format!(
            "rl restart next_generation {next_generation} outside 0..8"
        )));
    }
    doc.get("completed_updates")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| lc_err("rl restart completed_updates must be a non-bool integer"))?;
    let completed = doc
        .get("completed")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| lc_err("rl restart completed must be a bool"))?;
    if completed != (next_generation == 8) {
        return Err(lc_err(
            "rl restart completed must hold exactly when next_generation is 8".to_string(),
        ));
    }
    let behavior = get_str("behavior_policy_id")?;
    if !is_digest_shape(&behavior) {
        return Err(lc_err(
            "rl restart behavior_policy_id must be sha256:<64 hex>".to_string(),
        ));
    }
    let previous = get_str("previous_behavior_policy_id")?;
    if !is_digest_shape(&previous) {
        return Err(lc_err(
            "rl restart previous_behavior_policy_id must be sha256:<64 hex>".to_string(),
        ));
    }
    // Expected behavior lineage: C0 for start 0, Cg from the previous id
    // for unfinished starts 1..7, C7 from C6 for the completed start 8.
    // The learner-final/actor pairing is atomic by construction; arbitrary
    // same-shape tensor corruption is deliberately not detected here.
    let expected = if next_generation == 0 {
        if behavior != initial_id || previous != initial_id {
            return Err(lc_err(
                "rl restart start-0 behavior and previous must both be C0".to_string(),
            ));
        }
        behavior.to_string()
    } else if next_generation < 8 {
        policy_identity(run_descriptor_json, next_generation, &previous)?
    } else {
        policy_identity(run_descriptor_json, 7, &previous)?
    };
    if behavior != expected {
        return Err(lc_err(
            "rl restart behavior policy mismatches the lineage from previous".to_string(),
        ));
    }
    hydra_feed::canon::canonical_bytes_value(&descriptor, "rl:restart-descriptor")
        .map_err(|e| lc_err(format!("rl restart descriptor not canonical-safe: {e}")))
}

/// Validate a restart pointer against the run descriptor: closed 4-key
/// shape, matching pipeline, generation in range, and the fixed locator
/// derived from that generation (callers never supply arbitrary locators).
/// Returns the canonical pointer bytes.
pub fn validate_restart_pointer(
    run_descriptor_json: &[u8],
    pointer_json: &[u8],
) -> Result<Vec<u8>, LifecycleError> {
    let pointer: Value = serde_json::from_slice(pointer_json)
        .map_err(|e| lc_err(format!("rl restart pointer not JSON: {e}")))?;
    let doc = pointer
        .as_object()
        .ok_or_else(|| lc_err("rl restart pointer must be an object"))?;
    const KEYS: [&str; 4] = ["schema", "pipeline_spec_id", "next_generation", "locator"];
    if doc.len() != KEYS.len() || KEYS.iter().any(|k| !doc.contains_key(*k)) {
        return Err(lc_err(
            "rl restart pointer must carry exactly schema/pipeline_spec_id/next_generation/locator",
        ));
    }
    if doc.get("schema").and_then(|v| v.as_str()) != Some(RESTART_POINTER_SCHEMA) {
        return Err(lc_err("rl restart pointer schema mismatch".to_string()));
    }
    let run: Value = serde_json::from_slice(run_descriptor_json)
        .map_err(|e| lc_err(format!("rl restart run descriptor not JSON: {e}")))?;
    let run_doc = run
        .as_object()
        .ok_or_else(|| lc_err("rl restart run descriptor must be an object"))?;
    let run_pipeline = run_doc
        .get("pipeline_spec_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| lc_err("rl restart run descriptor lacks pipeline_spec_id"))?;
    let pipeline = doc
        .get("pipeline_spec_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| lc_err("rl restart pointer lacks pipeline_spec_id"))?;
    if pipeline != run_pipeline {
        return Err(lc_err(
            "rl restart pointer pipeline mismatches run descriptor".to_string(),
        ));
    }
    let next_generation = doc
        .get("next_generation")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| lc_err("rl restart pointer next_generation must be a non-bool integer"))?;
    if next_generation > 8 {
        return Err(lc_err(format!(
            "rl restart pointer next_generation {next_generation} outside 0..8"
        )));
    }
    let locator = doc
        .get("locator")
        .and_then(|v| v.as_str())
        .ok_or_else(|| lc_err("rl restart pointer locator must be a string"))?;
    if locator != format!("checkpoints/restart-g{next_generation:02}.pt") {
        return Err(lc_err(
            "rl restart pointer locator must be the fixed generation path".to_string(),
        ));
    }
    hydra_feed::canon::canonical_bytes_value(&pointer, "rl:restart-pointer")
        .map_err(|e| lc_err(format!("rl restart pointer not canonical-safe: {e}")))
}

/// Publish the restart pointer after its container landed: validates the
/// descriptor, derives the fixed locator from its generation, and atomically
/// writes `checkpoints/restart.json` (temp + fsync + rename + dir fsync).
/// The pointer changes only through this path on container success; save
/// errors leave the prior pointer authoritative.
pub fn publish_restart_pointer(
    root_dir: &str,
    run_descriptor_json: &[u8],
    descriptor_json: &[u8],
) -> Result<(), LifecycleError> {
    let canonical = validate_restart_descriptor(run_descriptor_json, descriptor_json)?;
    let descriptor: Value = serde_json::from_slice(&canonical)
        .map_err(|e| lc_err(format!("rl restart validated descriptor not JSON: {e}")))?;
    let doc = descriptor
        .as_object()
        .ok_or_else(|| lc_err("rl restart validated descriptor must be an object"))?;
    let next_generation = doc
        .get("next_generation")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| lc_err("rl restart validated descriptor lacks next_generation"))?;
    let pipeline_spec_id = doc
        .get("pipeline_spec_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| lc_err("rl restart validated descriptor lacks pipeline_spec_id"))?;
    let pointer = serde_json::json!({
        "schema": RESTART_POINTER_SCHEMA,
        "pipeline_spec_id": pipeline_spec_id,
        "next_generation": next_generation,
        "locator": format!("checkpoints/restart-g{next_generation:02}.pt"),
    });
    let pointer_bytes = validate_restart_pointer(
        run_descriptor_json,
        &serde_json::to_vec(&pointer)
            .map_err(|e| lc_err(format!("rl restart pointer not JSON-encodable: {e}")))?,
    )?;
    let path = std::path::Path::new(root_dir).join("checkpoints/restart.json");
    crate::rl::runtime_snapshot::atomic_write(&path, &pointer_bytes)
        .map_err(|e| lc_err(format!("rl restart pointer publish rejected: {e}")))?;
    Ok(())
}

/// One Rust-owned live replay preparation slot: runs `register_live_replay`
/// on a CPU worker while the owner continues eligible learner/inference
/// work, and owns the tape lifetime until exactly one consumer takes the
/// result. At most one slot exists at a time; the owner joins (poll/wait)
/// before tape drop/shutdown. Failures release the tape claim and surface
/// the worker error (readiness poisoned, never a fabricated result). No
/// Python callbacks and no borrowed pointers cross into the worker.
///
/// Ownership law: the slot stays occupied from begin until the joining owner
/// clears it, so a second begin can never overlap an active join. The worker
/// handle is taken exactly once by the consuming poll/wait, which then owns
/// the join, the result, and the tape-claim release on every path including
/// worker panic. A taken handle with no result yet means another thread is
/// already joining: late poll/wait calls reject as a second consumption
/// instead of racing it, and stale ids keep rejecting after their job is gone.
struct PrepSlot {
    id: u64,
    tape: crate::rl::trajectory::TapeHandle,
    worker: Option<PrepWorker>,
}

/// Join handle for one preparation worker: the registered replay plus its
/// descriptor bytes, or the worker's registration failure.
type PrepWorker = std::thread::JoinHandle<
    Result<
        (
            crate::rl::replay_index::ReplayHandle,
            Vec<u8>,
        ),
        crate::rl::trajectory::TapeError,
    >,
>;

struct PrepInner {
    next_id: u64,
    slot: Option<PrepSlot>,
}

static PREP: std::sync::LazyLock<std::sync::Mutex<PrepInner>> =
    std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(PrepInner {
            next_id: 1,
            slot: None,
        })
    });

fn prep_lock() -> Result<std::sync::MutexGuard<'static, PrepInner>, LifecycleError> {
    PREP
        .lock()
        .map_err(|e| lc_err(format!("rl live preparation lock poisoned: {e}")))
}

fn unknown_prep_job(job_id: u64) -> LifecycleError {
    lc_err(format!("unknown live preparation job {job_id}"))
}

/// Take the worker for `job_id`, marking the slot join-in-progress. The
/// caller then owns the join, the result, and the tape-claim release on
/// every path. Unknown or already-consumed jobs fail without touching the
/// slot; a taken worker means another thread is already joining, which
/// rejects as a second consumption instead of racing it.
fn take_prep_worker(
    job_id: u64,
) -> Result<(crate::rl::trajectory::TapeHandle, PrepWorker), LifecycleError> {
    let mut prep = prep_lock()?;
    let slot = prep
        .slot
        .as_mut()
        .ok_or_else(|| unknown_prep_job(job_id))?;
    if slot.id != job_id {
        return Err(unknown_prep_job(job_id));
    }
    match slot.worker.take() {
        Some(worker) => Ok((slot.tape, worker)),
        None => Err(lc_err(format!(
            "live preparation job {job_id} already consumed"
        ))),
    }
}

/// Complete consumption after the joining owner produced its outcome: clear
/// the slot and release the tape claim. Only the join owner reaches here
/// and no other path clears the slot, so the taken slot is always this
/// job's; a poisoned preparation lock still releases the tape claim through
/// the independent tape registry instead of orphaning it.
fn finish_prep_consumption(tape: crate::rl::trajectory::TapeHandle) {
    if let Ok(mut prep) = PREP.lock() {
        prep.slot.take();
    }
    crate::rl::trajectory::prep_release_tape(tape);
}

/// Begin one live replay preparation (at most one outstanding): claims the
/// tape and spawns the registration worker. Returns the job id the owner
/// polls/waits.
pub fn begin_live_replay(
    tape_handle: crate::rl::trajectory::TapeHandle,
    collector_spec_json: Vec<u8>,
    run_descriptor_json: Vec<u8>,
) -> Result<u64, LifecycleError> {
    let mut prep = prep_lock()?;
    if prep.slot.is_some() {
        return Err(lc_err(
            "live replay preparation already running".to_string(),
        ));
    }
    // Reserve the consumption token before claiming the tape or spawning:
    // id exhaustion must fail before either side effect, never leak a claim
    // or orphan a worker. Tokens stay unique (never reused), so stale
    // poll/wait calls keep rejecting after their job is gone.
    let id = prep.next_id;
    if id == 0 {
        return Err(lc_err(
            "live replay preparation id space exhausted".to_string(),
        ));
    }
    prep.next_id = prep.next_id.checked_add(1).ok_or_else(|| {
        lc_err("live replay preparation id space exhausted".to_string())
    })?;
    crate::rl::trajectory::prep_claim_tape(tape_handle)
        .map_err(|e| lc_err(format!("rl live preparation rejected: {e}")))?;
    let worker = std::thread::spawn(move || {
        crate::rl::replay_index::register_live_replay(
            tape_handle,
            &collector_spec_json,
            &run_descriptor_json,
        )
    });
    prep.slot = Some(PrepSlot {
        id,
        tape: tape_handle,
        worker: Some(worker),
    });
    Ok(id)
}

/// Poll one preparation job without blocking: `running`, or `done` with
/// the replay handle plus descriptor. A done poll consumes the job once
/// (matching wait) and releases the tape claim; unknown/consumed jobs
/// error. Repeated polls while running return the same status.
pub fn poll_live_replay(job_id: u64) -> Result<Vec<u8>, LifecycleError> {
    // Peek under one hold: running stays, finished proceeds to the single
    // take below. A second consumer racing this take rejects there instead
    // of joining twice.
    {
        let prep = prep_lock()?;
        let slot = prep
            .slot
            .as_ref()
            .ok_or_else(|| unknown_prep_job(job_id))?;
        if slot.id != job_id {
            return Err(unknown_prep_job(job_id));
        }
        match &slot.worker {
            None => {
                return Err(lc_err(format!(
                    "live preparation job {job_id} already consumed"
                )));
            }
            Some(worker) if !worker.is_finished() => {
                return Ok(serde_json::json!({"status": "running"}).to_string().into_bytes());
            }
            Some(_) => {}
        }
    }
    let (tape, worker) = take_prep_worker(job_id)?;
    match worker.join() {
        Err(_) => {
            finish_prep_consumption(tape);
            Err(lc_err(format!(
                "live preparation job {job_id} panicked"
            )))
        }
        Ok(Err(detail)) => {
            finish_prep_consumption(tape);
            Err(lc_err(format!("live preparation failed: {}", detail.0)))
        }
        Ok(Ok((replay_handle, descriptor_bytes))) => {
            finish_prep_consumption(tape);
            let descriptor: Value = serde_json::from_slice(&descriptor_bytes)
                .map_err(|e| lc_err(format!("live preparation descriptor not JSON: {e}")))?;
            let done = serde_json::json!({
                "status": "done",
                "replay_handle": replay_handle,
                "descriptor": descriptor,
            });
            hydra_feed::canon::canonical_bytes_value(&done, "rl:live-prep")
                .map_err(|e| lc_err(format!("live preparation status not canonical-safe: {e}")))
        }
    }
}

/// Block for one preparation job and consume it (matching poll-done):
/// returns the replay handle plus descriptor bytes. Unknown/consumed jobs
/// error; worker failure surfaces instead of a fabricated result. The slot
/// stays occupied while this call joins, so no second begin overlaps it.
pub fn wait_live_replay(job_id: u64) -> Result<(u64, Vec<u8>), LifecycleError> {
    let (tape, worker) = take_prep_worker(job_id)?;
    match worker.join() {
        Err(_) => {
            finish_prep_consumption(tape);
            Err(lc_err(format!(
                "live preparation job {job_id} panicked"
            )))
        }
        Ok(outcome) => {
            finish_prep_consumption(tape);
            outcome.map_err(|e| lc_err(format!("live preparation failed: {}", e.0)))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_bytes() -> Vec<u8> {
        // Minimal descriptor carrying exactly the fields the live gates read.
        // Cohort ledger shape mirrors prepare_run: generations count plus a
        // per-generation entry with pool_id and games.
        serde_json::json!({
            "lifecycle_version": "1.0.0",
            "master_seed_hex": "ab".repeat(32),
            "experiment_id": "live-test",
            "replicate_id": 0,
            "attempt_id": 0,
            "pipeline_spec_id": format!("sha256:{}", "11".repeat(32)),
            "initial_checkpoint_id": format!("sha256:{}", "22".repeat(32)),
            "adapter_version": "test-adapter-0",
            "wall_ledger_hash": format!("sha256:{}", "33".repeat(32)),
            "row_derivation_hash": format!("sha256:{}", "44".repeat(32)),
            "cohort": {
                "generations": 8,
                "native_workers": 2,
            },
            "rules": {
                "rules_id": "test-rules",
                "starting_points": 25000,
                "rank_values": [30, 10, 0, -10],
            },
            "identities": {
                "rules_hash": format!("sha256:{}", "55".repeat(32)),
                "action_table_hash": format!("sha256:{}", "66".repeat(32)),
                "event_schema_hash": format!("sha256:{}", "77".repeat(32)),
                "observation_schema_hash": format!("sha256:{}", "88".repeat(32)),
                "packet_boundary_hash": format!("sha256:{}", "99".repeat(32)),
                "utility_hash": format!("sha256:{}", "aa".repeat(32)),
                "model_spec_hash": format!("sha256:{}", "bb".repeat(32)),
            },
            "generations": (0..8).map(|g| serde_json::json!({
                "pool_id": format!("live-test-g{g:02}"),
                "games": [{"wall_id": format!("wall-{g}-0")}],
            })).collect::<Vec<_>>(),
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn g0_identity_is_c0_and_requires_c0_parent() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        assert_eq!(policy_identity(&descriptor, 0, &c0).expect("g0 identity"), c0);
        let other = format!("sha256:{}", "00".repeat(32));
        assert!(policy_identity(&descriptor, 0, &other).is_err());
    }

    #[test]
    fn lineage_is_deterministic_and_scope_sensitive() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        let c1 = policy_identity(&descriptor, 1, &c0).expect("c1");
        assert_eq!(c1, policy_identity(&descriptor, 1, &c0).expect("c1 again"));
        // Different parent binds a different child.
        let other = format!("sha256:{}", "00".repeat(32));
        let c1b = policy_identity(&descriptor, 1, &other).expect("c1b");
        assert_ne!(c1, c1b);
        // Different generation binds a different identity under the same parent.
        let c2 = policy_identity(&descriptor, 2, &c1).expect("c2");
        assert_ne!(c1, c2);
        // Chained lineage: C2 derives from C1, not C0.
        let c2_direct = policy_identity(&descriptor, 2, &c0).expect("c2 direct");
        assert_ne!(c2, c2_direct);
    }

    #[test]
    fn invalid_generations_and_shapes_reject() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        assert!(policy_identity(&descriptor, 8, &c0).is_err());
        assert!(policy_identity(&descriptor, 99, &c0).is_err());
        assert!(policy_identity(&descriptor, 1, "not-a-digest").is_err());
        assert!(policy_identity(b"not json", 1, &c0).is_err());
    }

    #[test]
    fn live_spec_matches_archive_core_and_carries_no_locators() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        let live: Value = serde_json::from_slice(
            &live_collector_spec(&descriptor, 0, &c0).expect("live spec"),
        )
        .expect("live json");
        let archive: Value = serde_json::from_slice(
            &crate::rl::lifecycle_collector::collector_spec(&descriptor, 0, &c0).expect("archive spec"),
        )
        .expect("archive json");
        assert_eq!(live["schema"], LIVE_COLLECTOR_SCHEMA);
        assert_eq!(live["profile"], archive["profile"]);
        assert_eq!(live["context"], archive["context"]);
        assert_eq!(live["shuffle_key"], archive["shuffle_key"]);
        assert_eq!(live["pool_id"], archive["pool_id"]);
        // No artifact coupling in the live entry.
        for absent in ["header", "artifact_locator", "metas_locator", "source_ledger", "eval_ledger"] {
            assert!(live.get(absent).is_none(), "live leaks {absent}");
        }
    }

    fn restart_doc(descriptor: &[u8], next_generation: u64, behavior: &str, previous: &str) -> Vec<u8> {
        let doc: Value = serde_json::from_slice(descriptor).expect("descriptor json");
        let (pipeline, initial) = (
            doc["pipeline_spec_id"].as_str().expect("pipeline").to_string(),
            doc["initial_checkpoint_id"].as_str().expect("c0").to_string(),
        );
        serde_json::json!({
            "schema": RESTART_DESCRIPTOR_SCHEMA,
            "pipeline_spec_id": pipeline,
            "initial_checkpoint_id": initial,
            "next_generation": next_generation,
            "completed_updates": 29 * next_generation,
            "behavior_policy_id": behavior,
            "previous_behavior_policy_id": previous,
            "completed": next_generation == 8,
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn restart_lineage_round_trip_and_mismatch_rejects() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        // Start 1 pairs learner-final g0 with C1 derived from C0.
        let c1 = policy_identity(&descriptor, 1, &c0).expect("c1");
        let doc = restart_doc(&descriptor, 1, &c1, &c0);
        let canonical = validate_restart_descriptor(&descriptor, &doc).expect("start-1 valid");
        let back: Value = serde_json::from_slice(&canonical).expect("canonical json");
        assert_eq!(back["behavior_policy_id"], c1);
        // Wrong behavior for the previous id rejects.
        let other = format!("sha256:{}", "00".repeat(32));
        assert!(validate_restart_descriptor(&descriptor, &restart_doc(&descriptor, 1, &other, &c0)).is_err());
        // Completed start 8 binds C7 from C6.
        let c6 = format!("sha256:{}", "66".repeat(32));
        let c7 = policy_identity(&descriptor, 7, &c6).expect("c7");
        let done = restart_doc(&descriptor, 8, &c7, &c6);
        assert!(validate_restart_descriptor(&descriptor, &done).is_ok());
        let mut unfinished: Value = serde_json::from_slice(&done).expect("done json");
        unfinished["completed"] = serde_json::json!(false);
        assert!(validate_restart_descriptor(
            &descriptor,
            &unfinished.to_string().into_bytes()
        )
        .is_err());
    }

    #[test]
    fn restart_closed_schema_rejects() {
        let descriptor = descriptor_bytes();
        let c0 = format!("sha256:{}", "22".repeat(32));
        let c1 = policy_identity(&descriptor, 1, &c0).expect("c1");
        let mut doc: Value =
            serde_json::from_slice(&restart_doc(&descriptor, 1, &c1, &c0)).expect("doc json");
        // Extra key rejects.
        doc["extra"] = serde_json::json!(1);
        assert!(
            validate_restart_descriptor(&descriptor, &doc.to_string().into_bytes()).is_err()
        );
        // Missing key rejects.
        let mut short = doc.as_object().expect("object").clone();
        short.remove("completed");
        short.remove("extra");
        assert!(validate_restart_descriptor(
            &descriptor,
            &serde_json::to_vec(&short).expect("short bytes")
        )
        .is_err());
        // Wrong generation range and pipeline reject.
        let mut bad = serde_json::from_slice::<Value>(&restart_doc(&descriptor, 1, &c1, &c0))
            .expect("bad json");
        bad["next_generation"] = serde_json::json!(9);
        assert!(
            validate_restart_descriptor(&descriptor, &bad.to_string().into_bytes()).is_err()
        );
    }

    #[test]
    fn restart_pointer_pins_fixed_locator() {
        let descriptor = descriptor_bytes();
        let doc: Value = serde_json::from_slice(&descriptor).expect("descriptor json");
        let pointer = serde_json::json!({
            "schema": RESTART_POINTER_SCHEMA,
            "pipeline_spec_id": doc["pipeline_spec_id"],
            "next_generation": 3,
            "locator": "checkpoints/restart-g03.pt",
        });
        let canonical = validate_restart_pointer(&descriptor, &pointer.to_string().into_bytes())
            .expect("pointer valid");
        let back: Value = serde_json::from_slice(&canonical).expect("pointer json");
        assert_eq!(back["locator"], "checkpoints/restart-g03.pt");
        // Caller-supplied arbitrary locators reject.
        let mut evil = pointer.clone();
        evil["locator"] = serde_json::json!("checkpoints/evil.pt");
        assert!(
            validate_restart_pointer(&descriptor, &evil.to_string().into_bytes()).is_err()
        );
    }

    // -- live replay preparation ownership --------------------------------------
    //
    // The tests below drive observable begin/poll/wait/drop behavior: exactly
    // one consumer owns a finished result, the tape claim releases on every
    // terminal path including worker panic, and no second preparation
    // overlaps an active join. Setup reaches into the private slot only to
    // inject panicking/slow workers no public input can produce; every
    // assertion is on ownership and result behavior, never on map layout or
    // diagnostic wording.

    static PREP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn prep_test_guard() -> std::sync::MutexGuard<'static, ()> {
        // Recover the inner guard when a previous test panicked while holding
        // it: every test resets the slot first, so a poisoned lock must never
        // mask which test actually failed.
        PREP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    type PrepTestOut = Result<
        (crate::rl::replay_index::ReplayHandle, Vec<u8>),
        crate::rl::trajectory::TapeError,
    >;

    fn prep_test_tape() -> crate::rl::trajectory::TapeHandle {
        let game = crate::rl::trajectory::TapeGame {
            wall_id: "wall-prep-test".to_string(),
            game_id: "game-prep-test".to_string(),
            ..Default::default()
        };
        let materialize = Box::new(
            |row: &crate::rl::core::GameRow, _arena: &crate::rl::events::EventArena| {
                Ok(serde_json::json!({"decision_id": row.decision_id.clone()}))
            },
        );
        let encode = Box::new(
            |_row: &crate::rl::core::GameRow,
             _arena: &crate::rl::events::EventArena,
             _t: u32| { Ok(vec![0u8; 8]) },
        );
        crate::rl::trajectory::register_tape(vec![game], materialize, encode)
            .expect("prep test tape registers")
    }

    fn prep_test_begin(tape: crate::rl::trajectory::TapeHandle) -> u64 {
        begin_live_replay(tape, b"{}".to_vec(), b"{}".to_vec()).expect("prep test begin")
    }

    fn prep_test_reset() {
        if let Ok(mut prep) = PREP.lock() {
            if let Some(slot) = prep.slot.take() {
                crate::rl::trajectory::prep_release_tape(slot.tape);
                crate::rl::trajectory::remove_tape(slot.tape);
            }
        }
    }

    fn prep_test_insert(
        tape: crate::rl::trajectory::TapeHandle,
        worker: std::thread::JoinHandle<PrepTestOut>,
    ) -> u64 {
        crate::rl::trajectory::prep_claim_tape(tape).expect("prep test claim");
        let mut prep = PREP.lock().expect("prep test lock");
        assert!(prep.slot.is_none(), "prep slot must be free for the test");
        let id = prep.next_id;
        assert_ne!(id, 0, "prep test id space exhausted");
        prep.next_id = prep.next_id.wrapping_add(1);
        prep.slot = Some(PrepSlot {
            id,
            tape,
            worker: Some(worker),
        });
        id
    }

    fn poll_until_terminal(job: u64) -> Result<Vec<u8>, LifecycleError> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            match poll_live_replay(job) {
                Ok(bytes) => {
                    let status: Value =
                        serde_json::from_slice(&bytes).expect("prep status is JSON");
                    if status.get("status").and_then(|s| s.as_str()) == Some("running") {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "prep job never settled"
                        );
                        std::thread::yield_now();
                        continue;
                    }
                    return Ok(bytes);
                }
                Err(failed) => return Err(failed),
            }
        }
    }

    #[test]
    fn prep_failed_worker_releases_claim_and_consumes_once() {
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let job = prep_test_begin(tape);
        assert!(
            begin_live_replay(tape, b"{}".to_vec(), b"{}".to_vec()).is_err(),
            "second begin must not overlap a running preparation"
        );
        assert!(
            poll_until_terminal(job).is_err(),
            "garbage specs must fail the preparation"
        );
        assert!(
            crate::rl::trajectory::remove_tape(tape),
            "failed preparation must release the tape claim"
        );
        assert!(poll_live_replay(job).is_err(), "consumed job must reject poll");
        assert!(wait_live_replay(job).is_err(), "consumed job must reject wait");
        let tape2 = prep_test_tape();
        let job2 = prep_test_begin(tape2);
        assert!(poll_until_terminal(job2).is_err());
        assert!(crate::rl::trajectory::remove_tape(tape2));
    }

    #[test]
    fn prep_panicking_worker_releases_claim_and_clears_job() {
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let worker = std::thread::spawn(move || -> PrepTestOut {
            panic!("prep test panic");
        });
        let job = prep_test_insert(tape, worker);
        assert!(
            poll_until_terminal(job).is_err(),
            "panicking worker must surface an error"
        );
        assert!(
            crate::rl::trajectory::remove_tape(tape),
            "panicking preparation must release the tape claim"
        );
        assert!(poll_live_replay(job).is_err(), "panicked job must stay cleared");
        assert!(wait_live_replay(job).is_err(), "panicked job must stay cleared");
    }

    #[test]
    fn prep_running_poll_then_successful_consumption() {
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let worker = std::thread::spawn(move || -> PrepTestOut {
            std::thread::sleep(std::time::Duration::from_millis(300));
            Ok((7, b"{}".to_vec()))
        });
        let job = prep_test_insert(tape, worker);
        let status: Value =
            serde_json::from_slice(&poll_live_replay(job).expect("running poll"))
                .expect("status json");
        assert_eq!(
            status.get("status").and_then(|s| s.as_str()),
            Some("running")
        );
        let done = poll_until_terminal(job).expect("slow preparation succeeds");
        let done: Value = serde_json::from_slice(&done).expect("done json");
        assert_eq!(done.get("status").and_then(|s| s.as_str()), Some("done"));
        assert_eq!(
            done.get("replay_handle").and_then(|h| h.as_u64()),
            Some(7)
        );
        assert!(crate::rl::trajectory::remove_tape(tape));
        assert!(poll_live_replay(job).is_err());
    }

    #[test]
    fn prep_concurrent_poll_wait_has_single_winner_and_keeps_result() {
        let _guard = prep_test_guard();
        let rounds = 30;
        let mut winners = 0;
        let mut losses = 0;
        for _ in 0..rounds {
            prep_test_reset();
            let tape = prep_test_tape();
            let worker = std::thread::spawn(move || -> PrepTestOut {
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok((7, b"{}".to_vec()))
            });
            let job = prep_test_insert(tape, worker);
            std::thread::sleep(std::time::Duration::from_millis(100));
            let waiter = std::thread::spawn(move || wait_live_replay(job));
            let polled = poll_live_replay(job);
            let waited = waiter.join().expect("waiter thread joins");
            match (polled, waited) {
                (Ok(bytes), Err(_)) => {
                    let done: Value =
                        serde_json::from_slice(&bytes).expect("done json");
                    assert_eq!(
                        done.get("replay_handle").and_then(|h| h.as_u64()),
                        Some(7),
                        "poll winner must keep the replay result"
                    );
                    winners += 1;
                }
                (Err(_), Ok((handle, _))) => {
                    assert_eq!(handle, 7, "wait winner must keep the replay result");
                    winners += 1;
                }
                (Err(_), Err(_)) => losses += 1,
                (Ok(_), Ok(_)) => panic!("one preparation result must have one owner"),
            }
            assert!(
                crate::rl::trajectory::remove_tape(tape),
                "consumed preparation must release the tape claim"
            );
            assert!(poll_live_replay(job).is_err());
        }
        assert_eq!(losses, 0, "concurrent poll/wait lost successful results");
        assert_eq!(winners, rounds);
    }

    #[test]
    fn prep_begin_while_wait_is_joining_is_rejected() {
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let worker = std::thread::spawn(move || -> PrepTestOut {
            std::thread::sleep(std::time::Duration::from_millis(500));
            Ok((7, b"{}".to_vec()))
        });
        let job = prep_test_insert(tape, worker);
        let waiter = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            wait_live_replay(job)
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        let tape2 = prep_test_tape();
        let begun = begin_live_replay(tape2, b"{}".to_vec(), b"{}".to_vec());
        let waited = waiter.join().expect("waiter thread joins");
        if let Ok(job2) = begun {
            let _ = poll_until_terminal(job2);
        }
        assert!(crate::rl::trajectory::remove_tape(tape));
        assert!(crate::rl::trajectory::remove_tape(tape2));
        assert!(
            begun.is_err(),
            "begin must not overlap a preparation still being joined"
        );
        let (handle, _) = waited.expect("waiter owns the joined result");
        assert_eq!(handle, 7);
    }

    #[test]
    fn prep_drop_while_claimed_is_refused() {
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let job = prep_test_begin(tape);
        assert!(
            !crate::rl::trajectory::remove_tape(tape),
            "tape drop must refuse a claimed tape"
        );
        assert!(poll_until_terminal(job).is_err());
        assert!(
            crate::rl::trajectory::remove_tape(tape),
            "tape drop succeeds once the claim releases"
        );
    }

    #[test]
    fn prep_slot_reusable_after_failed_consumption() {
        // Same-process re-entry: a consumed failure must leave the slot and
        // the tape claim reusable, or every later begin in this process
        // rejects with "already running" after one failed preparation.
        let _guard = prep_test_guard();
        prep_test_reset();
        let tape = prep_test_tape();
        let job = prep_test_begin(tape);
        assert!(poll_until_terminal(job).is_err());
        assert!(
            crate::rl::trajectory::remove_tape(tape),
            "failed consumption must free its tape"
        );
        let tape2 = prep_test_tape();
        let worker = std::thread::spawn(move || -> PrepTestOut { Ok((7, b"{}".to_vec())) });
        let job2 = prep_test_insert(tape2, worker);
        let done = poll_until_terminal(job2).expect("slot must accept a new job");
        let done: Value = serde_json::from_slice(&done).expect("done json");
        assert_eq!(done.get("status").and_then(|s| s.as_str()), Some("done"));
        assert!(
            crate::rl::trajectory::remove_tape(tape2),
            "second consumption must free its tape"
        );
        assert!(poll_live_replay(job2).is_err(), "consumed job must reject");
        assert!(poll_live_replay(job).is_err(), "stale job must reject");
    }
}
