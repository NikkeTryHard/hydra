//! Compact native capture tape: registry of owned generation cohorts plus
//! the frozen terminal recipe and the canonical row-document builder.
//!
//! Zero-bridge terms with the pool: the tape holds the pool's canonical
//! types directly — [`crate::rl::core::GameRow`] rows plus the game
//! [`crate::rl::events::EventArena`] (moved in at registration, borrowed
//! in place thereafter). No conversion layer, no copied fact structs. The
//! pool's `materialize_canonical(&GameRow, &EventArena)` and
//! `encode_row_facts(&GameRow, &EventArena, bucket)` plug into the two
//! registry slots with a one-line error map each (pool side); the seal and
//! export paths resolve everything Rust-to-Rust, never through Python and
//! never via per-row interpreter callbacks.
//!
//! Float-lane rule (cross-language JCS identity): every FP32 capture fact
//! (`selected_logp`, `value4`, frozen advantage) rides the canonical wire
//! as f64 widened by exact cast (`f32 as f64`). Rust `serde_jcs` prints f32
//! with ryu-f32 shortest digits while the Python oracle prints the widened
//! double with `repr` — different bytes for the same value. The f64-widened
//! lane is byte-identical on both sides for every f32 value, and the
//! validator rejects any wire double that is not exactly
//! FP32-representable, so no Python-side float pollution can enter the
//! artifact. Canonical four-seat returns (`G4`) are manifest doubles and
//! stay f64 with no FP32 gate.
//!
//! Numeric recipe (frozen, bound by the artifact `row_derivation_hash`):
//! terminal utility mapping yields canonical f64 `G4` (pool-derived from
//! the rank manifest); per row `A = f32(G4[actor]) - V_behavior[actor]` in
//! f32 arithmetic, frozen here by [`bind_terminal`]. The learner consumes
//! stored `A`; it never recomputes it.
//!
//! History buckets cover the actor-visible events of the current hand, not
//! the hanchan-wide arena prefix. Over-capacity (257+) fails closed; using
//! the prefix would wrongly reject valid later-hand rows.

use serde_json::{Map, Value};

use crate::rl::core::GameRow;
use crate::rl::events::EventArena;

/// Process handle for one registered owned tape (pool generation cohort).
pub type TapeHandle = u64;
/// Bounded canonical observation materializer: pure Rust function of one
/// capture row plus its game arena (adapts the pool's
/// `observation::materialize_canonical` with an error map). The seal invokes
/// it twice per row across its two streaming passes and asserts the
/// identical row id both times, so it MUST be deterministic.
pub type MaterializeFn =
    Box<dyn Fn(&GameRow, &EventArena) -> Result<Value, TapeError> + Send + Sync>;

/// Pure feature encoder over typed facts: capture row, game arena (the row's
/// `event_prefix_len` selects its immutable prefix), and history
/// bucket, into exactly `S(T) = 7254 + 9*T` feature bytes (adapts the pool's
/// `observation::encode_row_facts` with an error map). The seal/export side
/// enforces the output length fail-closed and never substitutes values.
pub type EncodeFn =
    Box<dyn Fn(&GameRow, &EventArena, u32) -> Result<Vec<u8>, TapeError> + Send + Sync>;

/// One game's compact tape: immutable event arena plus row captures in
/// completion order. Terminal facts freeze per row via [`bind_terminal`]
/// (CPU oracle) or the GPU two-phase path
/// (`gather_terminal_inputs`/`freeze_terminal_advantages`).
#[derive(Debug, Default, Clone)]
pub struct TapeGame {
    pub wall_id: String,
    pub game_id: String,
    pub arena: EventArena,
    pub rows: Vec<GameRow>,
    /// Canonical per-game utility doubles (pool-derived from the rank
    /// manifest once the terminal outcome is known). Required on every
    /// game before terminal gather/freeze; the seal binds rows from it.
    pub utilities: Option<[f64; 4]>,
}

/// Live source binding installed atomically with a drained tape: the exact
/// three-key identity the pool verified against its immutable live
/// collector binding at construction. Archive tapes carry no binding;
/// live consumers require it and match the exact tape handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTapeBinding {
    pub pipeline_spec_id: String,
    pub generation: u64,
    pub behavior_policy_id: String,
}

/// One registered owned tape: the pool's games plus the two native
/// functions the seal/export path resolves Rust-to-Rust (never through
/// Python, never per-row interpreter callbacks).
pub struct OwnedTape {
    pub games: Vec<TapeGame>,
    pub materialize: MaterializeFn,
    pub encode: EncodeFn,
}
struct RegistryInner {
    /// Concurrent immutable reads use short shared holds; freezing, removal
    /// and stamping take exclusive holds. Serializing readers would prevent
    /// native seal workers from processing independent rows concurrently.
    /// Removal during live work fails subsequent steps closed rather than
    /// succeeding on orphaned data — removal runs only after seal/close,
    /// quiesce drains the seal first. Stamps gate export staleness.
    tapes: std::collections::HashMap<TapeHandle, OwnedTape>,
    /// Artifact digests stamped by successful seals (export staleness gate).
    stamps: std::collections::HashMap<TapeHandle, String>,
    /// Live source bindings installed by `register_live_tape` (one per
    /// live tape; archive tapes have none). Removed with the tape.
    bindings: std::collections::HashMap<TapeHandle, LiveTapeBinding>,
    /// Tapes owned by a live replay preparation job (at most one; removal
    /// refuses owned handles until the owner joins the job).
    prep_owned: std::collections::HashSet<TapeHandle>,
    next: u64,
}

/// Canonical action vocabulary width (mirrors `encoder::BASELINE_ACTIONS`).
pub const ACTION_WIDTH: usize = 6792;
/// Frozen history buckets (mirrors `encoder::HISTORY_BUCKETS`).
pub const HISTORY_BUCKETS: [u32; 4] = [32, 64, 128, 256];
/// `sha256:` digest-text prefix (SPEC `DigestText` form on the wire).
pub const DIGEST_PREFIX: &str = "sha256:";
/// Exact hex length of a bare digest.
pub const DIGEST_HEX_LEN: usize = 64;

/// Fail-closed tape error; every variant names the offending identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TapeError(pub String);

impl std::fmt::Display for TapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trajectory rejected: {}", self.0)
    }
}

impl std::error::Error for TapeError {}


static TAPE_REGISTRY: std::sync::LazyLock<std::sync::RwLock<RegistryInner>> =
    std::sync::LazyLock::new(|| {
        std::sync::RwLock::new(RegistryInner {
            tapes: std::collections::HashMap::new(),
            stamps: std::collections::HashMap::new(),
            bindings: std::collections::HashMap::new(),
            prep_owned: std::collections::HashSet::new(),
            next: 1,
        })
    });

fn registry() -> &'static std::sync::RwLock<RegistryInner> {
    &TAPE_REGISTRY
}

/// Register one owned tape; returns the process handle the seal/export FFI
/// passes back. Handles start at 1; 0 is never issued (0 = absent).
pub fn register_tape(
    games: Vec<TapeGame>,
    materialize: MaterializeFn,
    encode: EncodeFn,
) -> Result<TapeHandle, TapeError> {
    if games.is_empty() {
        return Err(TapeError("cannot register an empty tape".to_string()));
    }
    let mut guard = registry()
        .write()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    let handle = guard.next;
    guard.next = guard.next.saturating_add(1);
    if handle == 0 || guard.next == 0 {
        return Err(TapeError("tape handle space exhausted".to_string()));
    }
    guard.tapes.insert(
        handle,
        OwnedTape {
            games,
            materialize,
            encode,
        },
    );
    Ok(handle)
}

/// Register one drained live tape with its source binding installed
/// atomically in the same write hold: either the handle serves the exact
/// verified binding or no handle exists. Archive tapes (via
/// [`register_tape`]) carry no binding and are unaffected. Handle
/// allocation uses checked addition so exhaustion fails closed instead of
/// reusing a live handle.
pub fn register_live_tape(
    games: Vec<TapeGame>,
    materialize: MaterializeFn,
    encode: EncodeFn,
    binding: LiveTapeBinding,
) -> Result<TapeHandle, TapeError> {
    if games.is_empty() {
        return Err(TapeError("cannot register an empty live tape".to_string()));
    }
    let mut guard = registry()
        .write()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    let handle = guard.next;
    guard.next = guard
        .next
        .checked_add(1)
        .ok_or_else(|| TapeError("tape handle space exhausted".to_string()))?;
    if handle == 0 {
        return Err(TapeError("tape handle space exhausted".to_string()));
    }
    guard.tapes.insert(
        handle,
        OwnedTape {
            games,
            materialize,
            encode,
        },
    );
    guard.bindings.insert(handle, binding);
    Ok(handle)
}

/// Read the live source binding installed with a drained tape (unknown
/// handle or archive tape without a binding fails closed).
pub fn tape_live_binding(handle: TapeHandle) -> Result<LiveTapeBinding, TapeError> {
    let guard = registry()
        .read()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    guard
        .bindings
        .get(&handle)
        .cloned()
        .ok_or_else(|| TapeError(format!("tape handle {handle} has no live binding")))
}
/// Claim a tape for a live replay preparation job (at most one job owns a
/// tape; the owner joins before tape drop/shutdown). Unknown or
/// already-owned handles fail closed with no claim taken.
pub(crate) fn prep_claim_tape(handle: TapeHandle) -> Result<(), TapeError> {
    let mut guard = registry()
        .write()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    if !guard.tapes.contains_key(&handle) {
        return Err(TapeError(format!("unknown tape handle {handle}")));
    }
    if !guard.prep_owned.insert(handle) {
        return Err(TapeError(format!(
            "live replay preparation already owns tape {handle}"
        )));
    }
    Ok(())
}

/// Release a preparation claim (join consumption; always succeeds).
pub(crate) fn prep_release_tape(handle: TapeHandle) {
    if let Ok(mut guard) = registry().write() {
        guard.prep_owned.remove(&handle);
    }
}

/// Whether a live preparation job owns a tape (drop guard probe).
pub(crate) fn prep_owns_tape(handle: TapeHandle) -> bool {
    registry()
        .read()
        .map_or(false, |guard| guard.prep_owned.contains(&handle))
}

/// Run `f` against the owned tape behind `handle` (unknown handle fails
/// closed). Holds are short by construction (see registry docs); removal
/// during live work fails the next step closed instead of orphaning it.
pub fn with_tape<R>(
    handle: TapeHandle,
    f: impl FnOnce(&OwnedTape) -> Result<R, TapeError>,
) -> Result<R, TapeError> {
    let guard = registry()
        .read()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    let tape = guard
        .tapes
        .get(&handle)
        .ok_or_else(|| TapeError(format!("unknown tape handle {handle}")))?;
    f(tape)
}

/// Run `f` mutably against the owned tape behind `handle` (freeze paths
/// only; unknown handle fails closed, same short-hold discipline).
pub fn with_tape_mut<R>(
    handle: TapeHandle,
    f: impl FnOnce(&mut OwnedTape) -> Result<R, TapeError>,
) -> Result<R, TapeError> {
    let mut guard = registry()
        .write()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    let tape = guard
        .tapes
        .get_mut(&handle)
        .ok_or_else(|| TapeError(format!("unknown tape handle {handle}")))?;
    f(tape)
}

/// Gather GPU terminal inputs for one complete generation (cold, once):
/// per-row actor ids, per-row game indices, per-row behavior value quads,
/// and per-game canonical utility quads — no rows, no documents. Every
/// game must carry utilities (generation-complete invariant); rows arrive
/// in tape order (games, then rows), the exact order `freeze` consumes.
/// Returns `(descriptor_bytes, actor_i64_le, game_index_i64_le,
/// behavior_values_f32_le, canonical_utilities_f64_le)`.
pub fn gather_terminal_inputs(
    handle: TapeHandle,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), TapeError> {
    with_tape(handle, |tape| {
        let mut actors = Vec::new();
        let mut game_of_row = Vec::new();
        let mut values = Vec::new();
        let mut utilities = Vec::new();
        let mut games_doc = Vec::with_capacity(tape.games.len());
        let mut row_count = 0usize;
        for (game_idx, game) in tape.games.iter().enumerate() {
            let g4 = game.utilities.ok_or_else(|| {
                TapeError(format!("game {} lacks terminal utilities", game.game_id))
            })?;
            if g4.iter().any(|v| !v.is_finite()) {
                return Err(TapeError(format!(
                    "game {} utilities must be finite",
                    game.game_id
                )));
            }
            for row in &game.rows {
                if row.actor as usize >= 4 {
                    return Err(TapeError(format!(
                        "row {} actor {} outside four seats",
                        row.decision_id, row.actor
                    )));
                }
                actors.extend_from_slice(&(row.actor as i64).to_le_bytes());
                game_of_row.extend_from_slice(&(game_idx as i64).to_le_bytes());
                for v in row.value4 {
                    values.extend_from_slice(&v.to_le_bytes());
                }
                row_count += 1;
            }
            for v in g4 {
                utilities.extend_from_slice(&v.to_le_bytes());
            }
            games_doc.push(serde_json::json!({
                "game_id": game.game_id,
                "wall_id": game.wall_id,
                "row_start": row_count - game.rows.len(),
                "row_count": game.rows.len(),
            }));
        }
        let mut descriptor = Map::with_capacity(3);
        descriptor.insert("row_count".to_string(), Value::from(row_count as u64));
        descriptor.insert("game_count".to_string(), Value::from(tape.games.len() as u64));
        descriptor.insert("games".to_string(), Value::Array(games_doc));
        let descriptor_bytes =
            hydra_feed::canon::canonical_bytes_value(&Value::Object(descriptor), "rl:terminal-inputs")
                .map_err(|e| TapeError(format!("terminal descriptor JCS failed: {e:?}")))?;
        Ok((descriptor_bytes, actors, game_of_row, values, utilities))
    })
}

/// Freeze GPU-computed advantages: validates the entire `rawA` buffer
/// (length exactly `4*N`, every value finite and bit-identical to the
/// frozen FP32 recipe `f32(G[actor]) - V[actor]`), then attaches frozen A
/// plus the canonical G4 immutably, exactly once, before any seal. The
/// wire `return_vector` keeps the original f64 manifest values; the
/// existing CPU [`bind_terminal`] remains the validation oracle for this
/// formula (same expression, not production compute). Returns rows frozen.
pub fn freeze_terminal_advantages(
    handle: TapeHandle,
    rawa_f32_le: &[u8],
) -> Result<usize, TapeError> {
    with_tape_mut(handle, |tape| {
        let mut row_count = 0usize;
        for game in &tape.games {
            row_count += game.rows.len();
        }
        if rawa_f32_le.len() != row_count * 4 {
            return Err(TapeError(format!(
                "rawA buffer holds {} bytes (want {} for {row_count} rows)",
                rawa_f32_le.len(),
                row_count * 4
            )));
        }
        // Phase 1: validate everything (shape, finiteness, bit-identity,
        // exactly-once) before attaching anything.
        let mut cursor = 0usize;
        for game in &tape.games {
            let g4 = game.utilities.ok_or_else(|| {
                TapeError(format!("game {} lacks terminal utilities", game.game_id))
            })?;
            for row in &game.rows {
                if row.return_g4.is_some() || row.raw_advantage_f32.is_some() {
                    return Err(TapeError(format!(
                        "row {} terminal facts already frozen",
                        row.decision_id
                    )));
                }
                if row.actor as usize >= 4 {
                    return Err(TapeError(format!(
                        "row {} actor {} outside four seats",
                        row.decision_id, row.actor
                    )));
                }
                let bytes: [u8; 4] = rawa_f32_le[cursor..cursor + 4]
                    .try_into()
                    .map_err(|_| TapeError("rawA buffer misaligned".to_string()))?;
                let provided = f32::from_le_bytes(bytes);
                if !provided.is_finite() {
                    return Err(TapeError(format!(
                        "row {} rawA nonfinite",
                        row.decision_id
                    )));
                }
                let expected = (g4[row.actor as usize] as f32) - row.value4[row.actor as usize];
                if provided != expected {
                    return Err(TapeError(format!(
                        "row {} rawA diverges from the frozen recipe",
                        row.decision_id
                    )));
                }
                cursor += 4;
            }
        }
        // Phase 2: attach (reuses the oracle expression; infallible after
        // phase 1, errors still propagate).
        for game in &mut tape.games {
            if game.rows.is_empty() {
                continue;
            }
            let g4 = game.utilities.ok_or_else(|| {
                TapeError(format!("game {} lacks terminal utilities", game.game_id))
            })?;
            bind_terminal(&mut game.rows, g4)?;
        }
        Ok(row_count)
    })
}

/// Remove a tape (pool lifecycle: after seal/close). Returns false when the
/// handle was unknown — never an error, never a panic. Removal refuses a
/// tape owned by a live preparation job (false until the owner joins). The
/// live binding, if any, is removed with the tape, and registration never
/// reuses a handle.
pub fn remove_tape(handle: TapeHandle) -> bool {
    match registry().write() {
        Ok(mut guard) => {
            if guard.prep_owned.contains(&handle) {
                return false;
            }
            guard.stamps.remove(&handle);
            guard.bindings.remove(&handle);
            guard.tapes.remove(&handle).is_some()
        }
        Err(_) => false,
    }
}

/// Stamp the artifact digest of a successful seal (called once by the seal,
/// never by the pool). Export refuses handles whose stamp mismatches the
/// index's artifact digest — a replaced tape cannot serve stale rows.
pub fn stamp_tape_digest(handle: TapeHandle, artifact_digest: &str) -> Result<(), TapeError> {
    let mut guard = registry()
        .write()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    if !guard.tapes.contains_key(&handle) {
        return Err(TapeError(format!("unknown tape handle {handle}")));
    }
    guard.stamps.insert(handle, artifact_digest.to_string());
    Ok(())
}

/// Verify a handle's seal stamp equals the expected artifact digest.
pub fn check_tape_stamp(handle: TapeHandle, artifact_digest: &str) -> Result<(), TapeError> {
    let guard = registry()
        .read()
        .map_err(|e| TapeError(format!("tape registry poisoned: {e}")))?;
    match guard.stamps.get(&handle) {
        Some(stamped) if stamped == artifact_digest => Ok(()),
        Some(_) => Err(TapeError(format!(
            "tape handle {handle} serves a different generation than requested"
        ))),
        None => Err(TapeError(format!(
            "tape handle {handle} has no sealed artifact stamp"
        ))),
    }
}

/// Bucket a history length into the frozen `T` grid; over-capacity (257+)
/// fails closed — the caller never coerces it into the last bucket.
pub fn bucket_history_len(history_len: u32) -> Result<u32, TapeError> {
    for bucket in HISTORY_BUCKETS {
        if history_len <= bucket {
            return Ok(bucket);
        }
    }
    Err(TapeError(format!(
        "history length {history_len} exceeds the frozen 256 cap"
    )))
}

/// Freeze terminal facts for every row of `rows`.
///
/// `g4` MUST be the canonical four-seat utility doubles (finite,
/// pool-derived from the rank manifest); per row `A = f32(g4[actor]) -
/// value4[actor]` in f32 arithmetic. Rows already carrying terminal facts
/// fail closed — terminal facts are immutable once frozen.
pub fn bind_terminal(rows: &mut [GameRow], g4: [f64; 4]) -> Result<(), TapeError> {
    if g4.iter().any(|v| !v.is_finite()) {
        return Err(TapeError("terminal G4 must be finite".to_string()));
    }
    for row in rows.iter_mut() {
        if row.return_g4.is_some() || row.raw_advantage_f32.is_some() {
            return Err(TapeError(format!(
                "row {} already has terminal facts",
                row.decision_id
            )));
        }
        if row.actor as usize >= 4 {
            return Err(TapeError(format!(
                "row {} actor {} outside four seats",
                row.decision_id, row.actor
            )));
        }
        // Frozen recipe: cast the canonical double to f32 FIRST, then
        // subtract the stored behavior value — both steps in f32.
        let g_actor = g4[row.actor as usize] as f32;
        let advantage = g_actor - row.value4[row.actor as usize];
        if !advantage.is_finite() {
            return Err(TapeError(format!(
                "row {} frozen advantage nonfinite",
                row.decision_id
            )));
        }
        row.return_g4 = Some(g4);
        row.raw_advantage_f32 = Some(advantage);
    }
    Ok(())
}

/// Lowercase hex of raw bytes.
pub fn hex_of(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Prefixed digest text for raw bytes (`sha256:<hex>`).
pub fn digest_text(raw: &[u8; 32]) -> String {
    format!("{DIGEST_PREFIX}{}", hex_of(raw))
}

/// Parse one prefixed digest text into raw bytes; rejects missing prefix,
/// wrong length, and non-hex digits.
pub fn parse_digest_text(text: &str) -> Result<[u8; 32], TapeError> {
    let hex = text
        .strip_prefix(DIGEST_PREFIX)
        .ok_or_else(|| TapeError(format!("digest missing {DIGEST_PREFIX:?} prefix: {text:?}")))?;
    if hex.len() != DIGEST_HEX_LEN {
        return Err(TapeError(format!(
            "digest wrong length {} (want {DIGEST_HEX_LEN}): {text:?}",
            hex.len()
        )));
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi =
            hex_val(chunk[0]).ok_or_else(|| TapeError(format!("digest non-hex byte: {text:?}")))?;
        let lo =
            hex_val(chunk[1]).ok_or_else(|| TapeError(format!("digest non-hex byte: {text:?}")))?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

/// Canonical row document WITHOUT `row_id` (the hash input), as an ordered
/// `serde_json::Value`.
///
/// `actor_obs` is the ONE bounded canonical observation document the pool
/// materialized for this row; it is embedded verbatim. `legal_mask` is the
/// expanded 6792-bool array derived from `legal_ids` (the compact tape holds
/// ids; the canonical doc holds the SPEC mask). Float lanes are f64-widened
/// f32 facts (see module docs); `return_vector` keeps canonical f64 doubles.
pub fn canonical_row_doc_no_id(row: &GameRow, actor_obs: Value) -> Result<Value, TapeError> {
    if row.return_g4.is_none() || row.raw_advantage_f32.is_none() {
        return Err(TapeError(format!(
            "row {} sealed before terminal bind",
            row.decision_id
        )));
    }
    let g4 = row.return_g4.unwrap_or([0.0; 4]);
    let advantage = row.raw_advantage_f32.unwrap_or(0.0);
    let mut mask = vec![false; ACTION_WIDTH];
    for id in &row.legal_ids {
        if (*id as usize) >= ACTION_WIDTH {
            return Err(TapeError(format!(
                "row {} legal id {id} outside the 6792 vocabulary",
                row.decision_id
            )));
        }
        if mask[*id as usize] {
            return Err(TapeError(format!(
                "row {} duplicate legal id {id}",
                row.decision_id
            )));
        }
        mask[*id as usize] = true;
    }
    let mut doc = Map::with_capacity(20);
    doc.insert("wall_id".to_string(), Value::String(row.wall_id.clone()));
    doc.insert(
        "trajectory_step".to_string(),
        Value::from(row.trajectory_step),
    );
    doc.insert("game_id".to_string(), Value::String(row.game_id.clone()));
    doc.insert("round_id".to_string(), Value::String(row.round_id.clone()));
    doc.insert(
        "decision_id".to_string(),
        Value::String(row.decision_id.clone()),
    );
    doc.insert("actor".to_string(), Value::from(u64::from(row.actor)));
    doc.insert(
        "observation_hash".to_string(),
        Value::String(digest_text(&row.obs_hash)),
    );
    doc.insert("actor_observation".to_string(), actor_obs);
    doc.insert(
        "action_table_hash".to_string(),
        Value::String(digest_text(&row.action_table_hash)),
    );
    doc.insert(
        "legal_mask".to_string(),
        Value::Array(mask.into_iter().map(Value::Bool).collect()),
    );
    doc.insert(
        "selected_action_id".to_string(),
        Value::from(u64::from(row.selected)),
    );
    doc.insert(
        "behavior_policy_checkpoint_hash".to_string(),
        Value::String(digest_text(&row.behavior_ckpt)),
    );
    // f32 facts widened exactly to f64 for the cross-language JCS lane.
    doc.insert(
        "old_selected_log_probability".to_string(),
        Value::from(f64::from(row.selected_logp)),
    );
    doc.insert(
        "raw_advantage".to_string(),
        Value::from(f64::from(advantage)),
    );
    doc.insert(
        "return_vector".to_string(),
        Value::Array(g4.iter().map(|v| Value::from(*v)).collect()),
    );
    // Pilot freezes `w_bc == 0`: no BC fields are stored, both stay null.
    doc.insert("bc_reference_checkpoint_hash".to_string(), Value::Null);
    doc.insert("bc_reference_logits".to_string(), Value::Null);
    doc.insert(
        "transition_provenance_hash".to_string(),
        Value::String(digest_text(&row.transition_provenance)),
    );
    Ok(Value::Object(doc))
}

/// SPEC12.3 content id: sha256 over the canonical bytes of the complete row
/// excluding `row_id`.
pub fn row_id_of(doc_no_id: &Value) -> Result<[u8; 32], TapeError> {
    let bytes = hydra_feed::canon::canonical_bytes_value(doc_no_id, "rl:row")
        .map_err(|e| TapeError(format!("row JCS emit failed: {e:?}")))?;
    Ok(sha256_raw(&bytes))
}

/// Raw sha256 of bytes (feed helper returns prefixed text; the seal path
/// needs raw words for sort keys and index maps).
pub fn sha256_raw(bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest as _;
    let sum = sha2::Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&sum);
    out
}

#[cfg(test)]
mod trajectory_tests {
    use super::*;
    use crate::rl::core::GameRow;

    pub fn sample_row() -> GameRow {
        GameRow {
            wall_id: "wall-1".to_string(),
            game_id: "game-1".to_string(),
            round_id: "round-0".to_string(),
            trajectory_step: 7,
            actor: 2,
            decision_id: "dec-7".to_string(),
            sequence: 7,
            row_id: [0u8; 32],
            round_index: 0,
            round_wind: 0,
            honba: 0,
            kyotaku: 0,
            dealer: 0,
            scores: [25000, 25000, 25000, 25000],
            live_remaining: 70,
            hand_number: 0,
            turn_actor: 2,
            phase_id: 1,
            concealed: vec![3, 9, 21],
            drawn: Some(33),
            dora5: [10, 20, 30, 40, 50],
            ippatsu_bits: 0,
            furiten_id: 0,
            can_tsumo: true,
            can_riichi: false,
            legal_ids: vec![0, 5, 6791],
            legal_mask_bits: vec![],
            selected: 5,
            selected_logp: -1.25,
            value4: [100.0, -50.0, 25.0, -75.0],
            cursor_before: 64,
            cursor_after: 72,
            obs_hash: [9u8; 32],
            event_prefix_len: 11,
            derivation: [6u8; 32],
            action_table_hash: [3u8; 32],
            action_table_hash_hex: format!("{DIGEST_PREFIX}{}", "03".repeat(32)),
            behavior_ckpt: [4u8; 32],
            transition_provenance: [5u8; 32],
            rules_id: "tenhou".to_string(),
            rules_hash: format!("{DIGEST_PREFIX}{}", "01".repeat(32)),
            event_schema_hash: format!("{DIGEST_PREFIX}{}", "02".repeat(32)),
            observation_schema_hash: format!("{DIGEST_PREFIX}{}", "03".repeat(32)),
            packet_boundary_hash: format!("{DIGEST_PREFIX}{}", "04".repeat(32)),
            return_g4: None,
            raw_advantage_f32: None,
        }
    }

    #[test]
    fn terminal_bind_freezes_fp32_advantage_recipe() {
        let mut rows = vec![sample_row()];
        bind_terminal(&mut rows, [10.0, -20.0, 30.0, -20.0]).unwrap();
        // Actor 2: A = f32(30.0) - 25.0 = 5.0 exactly.
        assert_eq!(rows[0].raw_advantage_f32, Some(5.0));
        assert_eq!(rows[0].return_g4, Some([10.0, -20.0, 30.0, -20.0]));
        // Second bind is immutable-once: rejects.
        assert!(bind_terminal(&mut rows, [0.0; 4]).is_err());
    }

    #[test]
    fn terminal_bind_rejects_nan_and_bad_actor() {
        let mut rows = vec![sample_row()];
        assert!(bind_terminal(&mut rows, [f64::NAN, 0.0, 0.0, 0.0]).is_err());
        let mut bad = vec![sample_row()];
        bad[0].actor = 7;
        assert!(bind_terminal(&mut bad, [0.0; 4]).is_err());
    }

    #[test]
    fn history_bucket_gates_boundaries() {
        assert_eq!(bucket_history_len(0), Ok(32));
        assert_eq!(bucket_history_len(32), Ok(32));
        assert_eq!(bucket_history_len(33), Ok(64));
        assert_eq!(bucket_history_len(256), Ok(256));
        assert!(bucket_history_len(257).is_err());
    }

    #[test]
    fn digest_text_round_trips() {
        let raw = [0xABu8; 32];
        let text = digest_text(&raw);
        assert_eq!(text, format!("sha256:{}", "ab".repeat(32)));
        assert_eq!(parse_digest_text(&text).unwrap(), raw);
        assert!(parse_digest_text("ab").is_err());
        assert!(parse_digest_text(&format!("sha256:{}", "zz".repeat(32))).is_err());
    }

    #[test]
    fn row_doc_rejects_duplicate_and_oob_legal_ids() {
        let mut row = sample_row();
        row.return_g4 = Some([0.0; 4]);
        row.raw_advantage_f32 = Some(0.0);
        row.legal_ids = vec![1, 1];
        assert!(canonical_row_doc_no_id(&row, Value::Null).is_err());
        row.legal_ids = vec![6792];
        assert!(canonical_row_doc_no_id(&row, Value::Null).is_err());
    }

    #[test]
    fn registry_lifecycle() {
        let identity = |row: &GameRow, _arena: &EventArena| {
            Ok(Value::String(row.decision_id.clone()))
        };
        let stub_encode = |_row: &GameRow, _arena: &EventArena, _t: u32| {
            Ok(vec![0u8; 7542])
        };
        let game = TapeGame {
            wall_id: "wall-1".to_string(),
            game_id: "game-1".to_string(),
            arena: EventArena::new(),
            rows: vec![sample_row()],
            utilities: None,
        };
        let handle = register_tape(vec![game], Box::new(identity), Box::new(stub_encode))
            .expect("register");
        assert!(handle != 0);
        with_tape(handle, |tape| {
            assert_eq!(tape.games.len(), 1);
            Ok(())
        })
        .expect("with");
        assert!(with_tape(999_999, |_| Ok(())).is_err());
        assert!(remove_tape(handle));
        assert!(!remove_tape(handle));
    }

    #[test]
    fn gather_freeze_round_trip_matches_oracle() {
        let identity = |row: &GameRow, _arena: &EventArena| {
            Ok(Value::String(row.decision_id.clone()))
        };
        let stub_encode = |_row: &GameRow, _arena: &EventArena, _t: u32| {
            Ok(vec![0u8; 7542])
        };
        // Two unbound rows, actor 2 (V=25) and actor 0 (V=8).
        let mut rows = vec![sample_row(), sample_row()];
        rows[0].decision_id = "dec-a".to_string();
        rows[0].trajectory_step = 0;
        rows[1].decision_id = "dec-b".to_string();
        rows[1].trajectory_step = 1;
        rows[1].actor = 0;
        rows[1].value4 = [8.0, 0.0, 0.0, 0.0];
        let game = TapeGame {
            wall_id: "wall-1".to_string(),
            game_id: "game-1".to_string(),
            arena: EventArena::new(),
            rows,
            utilities: Some([10.0, -20.0, 30.0, -20.0]),
        };
        let handle = register_tape(vec![game], Box::new(identity), Box::new(stub_encode))
            .expect("register");
        let (descriptor, actors, game_of_row, values, utilities) =
            gather_terminal_inputs(handle).expect("gather");
        assert_eq!(actors.len(), 2 * 8);
        assert_eq!(game_of_row.len(), 2 * 8);
        // Behavior quads ride f32 (4 seats × 4B); the f64 quad is utilities.
        assert_eq!(values.len(), 2 * 16);
        assert_eq!(utilities.len(), 1 * 32);
        let first_actor = i64::from_le_bytes(actors[0..8].try_into().expect("actor bytes"));
        assert_eq!(first_actor, 2);
        // GPU-side math in f32 (mirrors the Torch boundary): A = f32(G)-V.
        let a0 = 30.0f32 - 25.0;
        let a1 = 10.0f32 - 8.0;
        let mut rawa = Vec::with_capacity(8);
        rawa.extend_from_slice(&a0.to_le_bytes());
        rawa.extend_from_slice(&a1.to_le_bytes());
        let frozen = freeze_terminal_advantages(handle, &rawa).expect("freeze");
        assert_eq!(frozen, 2);
        with_tape(handle, |tape| {
            assert_eq!(tape.games[0].rows[0].raw_advantage_f32, Some(5.0));
            assert_eq!(tape.games[0].rows[1].raw_advantage_f32, Some(2.0));
            Ok(())
        })
        .expect("verify");
        // Exactly-once: second freeze fails; tampered values fail.
        assert!(freeze_terminal_advantages(handle, &rawa).is_err());
        let _ = descriptor;
        assert!(remove_tape(handle));
    }

    #[test]
    fn freeze_rejects_shape_and_value_forks() {
        let identity = |row: &GameRow, _arena: &EventArena| {
            Ok(Value::String(row.decision_id.clone()))
        };
        let stub_encode = |_row: &GameRow, _arena: &EventArena, _t: u32| {
            Ok(vec![0u8; 7542])
        };
        let game = TapeGame {
            wall_id: "wall-1".to_string(),
            game_id: "game-1".to_string(),
            arena: EventArena::new(),
            rows: vec![sample_row()],
            utilities: Some([10.0, -20.0, 30.0, -20.0]),
        };
        let handle = register_tape(vec![game], Box::new(identity), Box::new(stub_encode))
            .expect("register");
        // Wrong length, nonfinite, and divergent values all fail closed.
        assert!(freeze_terminal_advantages(handle, &[0u8; 3]).is_err());
        assert!(freeze_terminal_advantages(handle, &f32::NAN.to_le_bytes()).is_err());
        assert!(freeze_terminal_advantages(handle, &0.0f32.to_le_bytes()).is_err());
        // Missing utilities fail the gather first.
        with_tape_mut(handle, |tape| {
            tape.games[0].utilities = None;
            Ok(())
        })
        .expect("clear");
        assert!(gather_terminal_inputs(handle).is_err());
        assert!(remove_tape(handle));
    }
}
