//! Sim-capture column builders: per-decision sim outputs -> one actor `RecordBatch`.
//!
//! Owner: this module owns the sim-capture builders ONLY (the expand side of
//! Phase 3). The shard writer/reader/cache (`writer.rs` / `reader.rs` /
//! `cache.rs` + the shard Cargo manifest) are owned by a sibling; the bridge
//! capsule shim (`hydra-bridge` `columnar.rs` + Python shim) by another. This
//! module lives at `hydra_shard::replay::expand` (declared via `#[path]` in
//! `replay.rs`) because the crate root `lib.rs` is sibling-owned — do NOT add
//! a second crate-root `pub mod expand`.
//!
//! Reference (read-only oracles, never edited):
//! `src/hydra2/data/replay_expand.py` (`ReplayExpander._emit_row` capture
//! shapes, `_count_row_decisions` capacity pass, `_expand_shard_game` pool
//! payload), `src/hydra2/data/shard_build.py` (expand pool + torch encode +
//! pad/cat/IPC per plane, `FIXED_HISTORY_T = 256`, `HISTORY_LEN_PLANE`),
//! `src/hydra2/data/parquet.py` (`_ACTOR_SCHEMA` 13-field layout,
//! `FORBIDDEN_IN_ACTOR`, `(5,)` dora firewall), `scripts/freeze_row_hashes.py`
//! (`row_hash` closed-allowlist loop), `src/hydra2/engines/protocol.py`
//! (`ExactSimulator` surface).
//!
//! # Engine-input boundary (sim stays Python this phase)
//!
//! Rust takes caller-supplied [`SimGameInput`] / [`SimDecisionInput`] structs
//! (game provenance + per-decision capture rows produced by the Python
//! `ReplayExpander` driving the `ExactSimulator`: identity seating
//! `(0,1,2,3)`, `min physical id` copy resolution, reach-collapse counting).
//! There is deliberately NO sim port here: no `reset` / `legal_actions` /
//! `apply` / `actor_observation` engine calls, no `Engine` trait, no wall
//! synthesis. A wall-less game is signalled by `wall_digest: None` (the Rust
//! side binds [`crate::parity::SIM_DERIVATION_MARK`], never an invented
//! digest); a walled game carries the real schedule digest. A later phase may
//! move the walk itself into Rust; these structs already carry everything the
//! walk would need except the live engine.
//!
//! # Capture shapes (mirror `replay_expand._emit_row` + `parquet` bucketing)
//!
//! [`actor_schema`] is byte-exact with `parquet._ACTOR_SCHEMA`: the 13
//! `ACTOR_FIELDS` names in order, `pa.string()` -> `Utf8`, `pa.int64()` ->
//! `Int64`, `chosen_action_id` nullable (null exactly when the action id is
//! unresolved — the S4 pin), all other columns non-nullable. `expand_batch`
//! buckets caller-ordered inputs into those columns the way
//! `write_actor_shards` buckets rows per split, except rows stay in caller
//! order (no per-split regrouping: ordering is the caller's `order` manifest
//! attestation, and `dataset_hash` is order-independent anyway).
//!
//! `actor_observation` text is serialized ONCE via the single canonical owner
//! (`hydra-feed::canon` through [`crate::parity::canonical_json_bytes`],
//! `serde_jcs` ONLY-exact — never `serde_json::to_string` for identity bytes).
//! Key order is therefore canonical (sorted), not Python dict-insertion
//! order: the freeze projection (`freeze_row_hashes.full_projection`) parses
//! the text back to a dict before hashing, so the projection hash is
//! order-insensitive while the stored text stays deterministic.
//!
//! # Determinism (counter-free, caller-order preserved)
//!
//! - `decision_id = "{game_id}:d{seq:04}"`, `round_id = "{game_id}:h{round:02}"`
//!   (oracle `:04d` / `:02d` formats, unchanged).
//! - `seq` is the decision's POSITION in the caller's slice, `round_idx` is
//!   caller-supplied per decision: no global/static/mutable counters anywhere
//!   (counter-free), so interleaved or multi-threaded callers cannot fork ids.
//! - Games emit in slice order, decisions in vec order (caller-order
//!   preserved); the batch is a pure function of the input slices, so the
//!   same inputs yield a byte-identical batch (see `deterministic_replay`).
//!
//! # Geometry notes (`[N,256]` + `history_len`, uint8 bools, sorted ids)
//!
//! This batch is the actor ROW envelope (N rows x 13 columns), not the tensor
//! planes. Tensor geometry is downstream and unchanged: torch encode (stays
//! torch — no Rust GPU math) pads history planes to `FIXED_HISTORY_T = 256`
//! with a `history_len` plane carrying true lengths; boolean planes ride
//! `uint8` (bit-packed Arrow booleans cannot `to_numpy`-view); `dataset_hash`
//! is `sha256:` over the newline-joined SORTED decision ids (use
//! [`ExpandedBatch::decision_ids`] to collect them — sorting is the
//! writer/manifest owner's job, not this module's). `BATCH = 8192` chunking
//! is owned by the bridge (`columnar.rs` single const; search/eval import
//! later): callers size the `games` slice so the row total lands near it.
//!
//! # Freeze closed-allowlist note (`freeze_row_hashes.py:210` loop)
//!
//! The per-row `row_hash` loop (RFC 8785 sha256 over closed-allowlist
//! projections) moves to the Rust writer (hash beside the write path) plus
//! the parallel bridge; it is NOT re-implemented here. This module's duty to
//! the freeze is narrower: emit exactly the row bytes the freeze reads
//! (stable ids, verified `observation_hash`, wall-bound `derivation_hash`,
//! canonical `actor_observation` text) so a frozen hash taken over
//! Python-oracle rows and over bridge-consumed batches agrees.
//!
//! # Firewall (capture leg of the actor-never-privileged triple)
//!
//! Capture here, write path in `writer.rs`, load/encode downstream: three
//! independent gates, any one fails closed. This leg mirrors
//! `shard_build._firewall_check_row` fail-closed (leakage is a caller bug,
//! never a quarantine): `FORBIDDEN_IN_ACTOR` keys rejected at the top level
//! AND one level nested, and `dora_indicators` must bind `(5,)` exactly — the
//! `(4,)` shim (or a missing/non-list binding) is rejected.

use std::fmt;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::array::{ArrayRef, StringArray};
use arrow_array::builder::{Int64Builder, StringBuilder};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::parity::{
    ACTOR_FIELDS, canonical_json_bytes, derivation_hash_for, derivation_hash_for_walled,
    observation_hash_for,
};

/// Privileged keys that MUST NEVER appear in an actor observation (mirror of
/// `parquet.FORBIDDEN_IN_ACTOR`: top level + one level nested).
pub const FORBIDDEN_IN_ACTOR: [&str; 6] = [
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
];

/// `RecordBatch` column index of `decision_id` (index 2 in `ACTOR_FIELDS`).
pub const DECISION_ID_COL: usize = 2;

/// One sim-captured decision supplied by the Python expander (borrowed, never
/// copied on ingest; the batch owns its copies on append).
#[derive(Debug, Clone, Copy)]
pub struct SimDecisionInput<'a> {
    /// Round index within the game (`round_id = "{game}:h{round:02}"`).
    pub round_idx: u32,
    /// Acting seat (must be 0..=3; identity seating, never permuted here).
    pub seat: u8,
    /// Encoded action id (`None` exactly when `chosen_unresolved` is set).
    pub chosen_action_id: Option<u32>,
    /// Unresolved-action reason (`Some` exactly when `chosen_action_id` is
    /// `None`; carried for the S4 pin check, never written to the batch).
    pub chosen_unresolved: Option<&'a str>,
    /// Actor observation JSON text as captured (re-parsed + re-serialized
    /// canonically; key order of this text does not affect output).
    pub actor_observation_json: &'a str,
    /// `sha256:` over the canonical observation bytes, as computed by the
    /// Python capture (`of_canonical`); re-verified here fail-closed.
    pub observation_hash: &'a str,
}

/// One game's provenance plus its caller-ordered sim-captured decisions.
///
/// `wall_digest`: `Some` real schedule digest (walled path) or `None`
/// (wall-less path — the SIM mark is bound, never an invented digest).
#[derive(Debug, Clone, Copy)]
pub struct SimGameInput<'a> {
    pub game_id: &'a str,
    pub source_object_id: &'a str,
    pub split: &'a str,
    pub rules_hash: &'a str,
    pub adapter_hash: &'a str,
    pub action_table_hash: &'a str,
    pub wall_digest: Option<&'a str>,
    pub decisions: &'a [SimDecisionInput<'a>],
}

/// Fail-closed capture errors (caller bugs, never quarantines: quarantine is
/// the walker's verdict over log content, while these inputs already passed
/// the sim — a failure here means the capture itself is corrupt).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpandError {
    EmptyGameId,
    EmptySplit,
    SeatOutOfRange { decision_id: String, seat: u8 },
    ObservationJson { decision_id: String, detail: String },
    ObservationNotObject { decision_id: String },
    ObservationHashMismatch { decision_id: String, expected: String, actual: String },
    ObservationUtf8 { decision_id: String, detail: String },
    PrivilegedLeak { decision_id: String, field: String },
    DoraShape { decision_id: String, detail: String },
    ChosenPin { decision_id: String },
    BadWallDigest { game_id: String, digest: String },
    BlobFrame { index: usize, detail: String },
    Arrow { detail: String },
}

impl fmt::Display for ExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpandError::EmptyGameId => write!(f, "expand: game_id must be non-empty"),
            ExpandError::EmptySplit => write!(f, "expand: split must be a non-empty string"),
            ExpandError::SeatOutOfRange { decision_id, seat } => {
                write!(f, "expand: seat {seat} out of range 0..=3 at {decision_id}")
            }
            ExpandError::ObservationJson { decision_id, detail } => {
                write!(f, "expand: actor_observation is not JSON at {decision_id}: {detail}")
            }
            ExpandError::ObservationNotObject { decision_id } => {
                write!(f, "expand: actor_observation must be an object at {decision_id}")
            }
            ExpandError::ObservationHashMismatch { decision_id, expected, actual } => {
                write!(
                    f,
                    "expand: observation_hash mismatch at {decision_id}: \
                     capture={expected} recomputed={actual}"
                )
            }
            ExpandError::ObservationUtf8 { decision_id, detail } => {
                write!(
                    f,
                    "expand: canonical observation bytes are not UTF-8 at {decision_id}: {detail}"
                )
            }
            ExpandError::PrivilegedLeak { decision_id, field } => {
                write!(
                    f,
                    "expand: privileged field leakage into actor observation: \
                     {field:?} at {decision_id}"
                )
            }
            ExpandError::DoraShape { decision_id, detail } => {
                write!(f, "expand: dora_indicators must bind (5,) at {decision_id}: {detail}")
            }
            ExpandError::ChosenPin { decision_id } => {
                write!(
                    f,
                    "expand: chosen_action_id must be null iff unresolved at {decision_id}"
                )
            }
            ExpandError::BadWallDigest { game_id, digest } => {
                write!(
                    f,
                    "expand: walled derivation needs a real wall digest for game \
                     {game_id:?}, got {digest:?}"
                )
            }
            ExpandError::BlobFrame { index, detail } => {
                write!(f, "expand: game blob [{index}] framing failed: {detail}")
            }
            ExpandError::Arrow { detail } => write!(f, "expand: arrow batch build failed: {detail}"),
        }
    }
}

impl std::error::Error for ExpandError {}

/// Canonical actor schema — MUST equal `parquet._ACTOR_SCHEMA` names, order,
/// and types (`pa.string()` -> `Utf8`, `pa.int64()` -> `Int64`,
/// `chosen_action_id` the sole nullable column). Any drift is a schema-identity
/// bug: the writer asserts before any byte is written.
pub fn actor_schema() -> SchemaRef {
    let fields: Vec<Field> = ACTOR_FIELDS
        .iter()
        .map(|name| {
            let (data_type, nullable) = match *name {
                "seat" | "chosen_action_id" => (DataType::Int64, *name == "chosen_action_id"),
                _ => (DataType::Utf8, false),
            };
            Field::new(*name, data_type, nullable)
        })
        .collect();
    debug_assert_eq!(fields.len(), 13, "actor schema must carry exactly 13 fields");
    Arc::new(Schema::new(fields))
}

/// One actor batch: the `RecordBatch` plus typed accessors the bridge and the
/// writer need without reaching into column indices.
#[derive(Debug, Clone)]
pub struct ExpandedBatch {
    batch: RecordBatch,
}

impl ExpandedBatch {
    /// The built batch (N rows in caller order over [`actor_schema`]).
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Consume into the batch (bridge wraps: `vec![expanded.into_batch()]`
    /// for the capsule stream).
    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }

    /// Committed row count.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// `decision_id` column (actor-schema index 2): collect + sort downstream
    /// for the order-independent `dataset_hash`. `None` when the batch was
    /// not built by [`expand_batch`] (wrong column type at index 2).
    pub fn decision_ids(&self) -> Option<&StringArray> {
        self.batch
            .column(DECISION_ID_COL)
            .as_any()
            .downcast_ref::<StringArray>()
    }
}

/// Expand caller-ordered sim-captured games into one actor `RecordBatch`.
///
/// Pure function of the input slices: games in slice order, decisions in vec
/// order, ids derived from positions (counter-free). The same inputs yield a
/// byte-identical batch.
pub fn expand_batch(games: &[SimGameInput<'_>]) -> Result<ExpandedBatch, ExpandError> {
    let total: usize = games.iter().map(|game| game.decisions.len()).sum();
    let mut builders = ActorBuilders::with_capacity(total);
    for game in games {
        expand_game_into(game, &mut builders)?;
    }
    Ok(ExpandedBatch {
        batch: builders.finish(actor_schema())?,
    })
}

/// Owned sim-capture inputs for FFI-adjacent callers (the bridge).
///
/// Field-for-field the owned twin of [`SimGameInput`] / [`SimDecisionInput`]:
/// the bridge copies descriptor bytes into these ATTACHED, then calls
/// [`expand_batch_owned`] under `detach`. Borrowed views are built inside
/// that call over the owned buffers (which the caller keeps alive for the
/// call), so the bridge NEVER juggles lifetimes across the detach boundary.
#[derive(Debug, Clone)]
pub struct OwnedSimDecisionInput {
    pub round_idx: u32,
    pub seat: u8,
    pub chosen_action_id: Option<u32>,
    pub chosen_unresolved: Option<String>,
    pub actor_observation_json: String,
    pub observation_hash: String,
}

/// Owned twin of [`SimGameInput`]. `wall_digest` is DigestText form
/// (`sha256:` + 64 lowercase hex) when walled, `None` when wall-less — NOT
/// `[u8; 32]` (digests ride the row envelope as text on both writers).
#[derive(Debug, Clone)]
pub struct OwnedSimGameInput {
    pub game_id: String,
    pub source_object_id: String,
    pub split: String,
    pub rules_hash: String,
    pub adapter_hash: String,
    pub action_table_hash: String,
    pub wall_digest: Option<String>,
    pub decisions: Vec<OwnedSimDecisionInput>,
}

/// Owned-bytes entry: builds the borrowed views internally and delegates to
/// the same capture path as [`expand_batch`] (identical output for identical
/// logical inputs — see `owned_entry_matches_borrowed_entry`).
pub fn expand_batch_owned(games: &[OwnedSimGameInput]) -> Result<ExpandedBatch, ExpandError> {
    let total: usize = games.iter().map(|game| game.decisions.len()).sum();
    let mut builders = ActorBuilders::with_capacity(total);
    for game in games {
        // Temporaries live to the end of the iteration; views never escape it.
        let decisions: Vec<SimDecisionInput<'_>> = game
            .decisions
            .iter()
            .map(|decision| SimDecisionInput {
                round_idx: decision.round_idx,
                seat: decision.seat,
                chosen_action_id: decision.chosen_action_id,
                chosen_unresolved: decision.chosen_unresolved.as_deref(),
                actor_observation_json: &decision.actor_observation_json,
                observation_hash: &decision.observation_hash,
            })
            .collect();
        let borrowed = SimGameInput {
            game_id: &game.game_id,
            source_object_id: &game.source_object_id,
            split: &game.split,
            rules_hash: &game.rules_hash,
            adapter_hash: &game.adapter_hash,
            action_table_hash: &game.action_table_hash,
            wall_digest: game.wall_digest.as_deref(),
            decisions: &decisions,
        };
        expand_game_into(&borrowed, &mut builders)?;
    }
    Ok(ExpandedBatch {
        batch: builders.finish(actor_schema())?,
    })
}


/// Assemble owned game inputs from per-game wire blobs (bridge entry).
///
/// Framing is owned HERE (the bridge FORBIDS parse and only memcpys):
/// each blob is one UTF-8 JSON object with string fields `game_id`,
/// `source_object_id`, `split`, `rules_hash`, `adapter_hash`,
/// `action_table_hash`, `wall_digest` (string or null), and `decisions`
/// (array of objects with `round_idx` / `seat` non-negative integers,
/// `chosen_action_id` integer or null, `chosen_unresolved` string or null,
/// `observation_hash` string, and `actor_observation` holding the
/// observation document — any JSON value; it is re-parsed and re-serialized
/// canonically at capture, so its wire key order does not affect output).
///
/// Shape failures report the blob index ([`ExpandError::BlobFrame`]); value
/// rules (empty ids, seat range, S4 pin, firewall, hash match, wall shape)
/// stay in the capture path so direct struct callers get the same gates.
pub fn assemble_owned_games(blobs: &[Vec<u8>]) -> Result<Vec<OwnedSimGameInput>, ExpandError> {
    blobs
        .iter()
        .enumerate()
        .map(|(index, blob)| assemble_one_blob(index, blob))
        .collect()
}

fn blob_err(index: usize, detail: impl Into<String>) -> ExpandError {
    ExpandError::BlobFrame {
        index,
        detail: detail.into(),
    }
}

fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn blob_field_str(
    doc: &serde_json::Map<String, serde_json::Value>,
    index: usize,
    key: &str,
) -> Result<String, ExpandError> {
    match doc.get(key) {
        Some(serde_json::Value::String(text)) => Ok(text.clone()),
        Some(other) => Err(blob_err(index, format!("{key} must be a string, got {}", json_kind(other)))),
        None => Err(blob_err(index, format!("missing {key}"))),
    }
}

fn blob_field_u32(
    doc: &serde_json::Map<String, serde_json::Value>,
    index: usize,
    key: &str,
) -> Result<u32, ExpandError> {
    match doc.get(key) {
        Some(serde_json::Value::Number(number)) => number
            .as_u64()
            .filter(|value| *value <= u32::MAX as u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| blob_err(index, format!("{key} must fit u32"))),
        Some(other) => Err(blob_err(
            index,
            format!("{key} must be an integer, got {}", json_kind(other)),
        )),
        None => Err(blob_err(index, format!("missing {key}"))),
    }
}

fn assemble_one_blob(index: usize, blob: &[u8]) -> Result<OwnedSimGameInput, ExpandError> {
    let value: serde_json::Value =
        serde_json::from_slice(blob).map_err(|err| blob_err(index, format!("not JSON: {err}")))?;
    let doc = value
        .as_object()
        .ok_or_else(|| blob_err(index, format!("top level must be an object, got {}", json_kind(&value))))?;
    let wall_digest = match doc.get("wall_digest") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(digest)) => Some(digest.clone()),
        Some(other) => {
            return Err(blob_err(
                index,
                format!("wall_digest must be a string or null, got {}", json_kind(other)),
            ));
        }
    };
    let decisions = match doc.get("decisions") {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(position, item)| assemble_one_decision(index, position, item))
            .collect::<Result<Vec<_>, _>>()?,
        Some(other) => {
            return Err(blob_err(
                index,
                format!("decisions must be an array, got {}", json_kind(other)),
            ));
        }
        None => return Err(blob_err(index, "missing decisions")),
    };
    Ok(OwnedSimGameInput {
        game_id: blob_field_str(doc, index, "game_id")?,
        source_object_id: blob_field_str(doc, index, "source_object_id")?,
        split: blob_field_str(doc, index, "split")?,
        rules_hash: blob_field_str(doc, index, "rules_hash")?,
        adapter_hash: blob_field_str(doc, index, "adapter_hash")?,
        action_table_hash: blob_field_str(doc, index, "action_table_hash")?,
        wall_digest,
        decisions,
    })
}

fn assemble_one_decision(
    index: usize,
    position: usize,
    item: &serde_json::Value,
) -> Result<OwnedSimDecisionInput, ExpandError> {
    let doc = item.as_object().ok_or_else(|| {
        blob_err(
            index,
            format!("decisions[{position}] must be an object, got {}", json_kind(item)),
        )
    })?;
    let seat = match doc.get("seat") {
        Some(serde_json::Value::Number(number)) => number
            .as_u64()
            .filter(|value| *value <= u8::MAX as u64)
            .and_then(|value| u8::try_from(value).ok())
            .ok_or_else(|| blob_err(index, format!("decisions[{position}].seat must fit u8")))?,
        Some(other) => {
            return Err(blob_err(
                index,
                format!(
                    "decisions[{position}].seat must be an integer, got {}",
                    json_kind(other)
                ),
            ));
        }
        None => return Err(blob_err(index, format!("decisions[{position}] missing seat"))),
    };
    let chosen_action_id = match doc.get("chosen_action_id") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(number)) => Some(
            number
                .as_u64()
                .filter(|value| *value <= u32::MAX as u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    blob_err(
                        index,
                        format!("decisions[{position}].chosen_action_id must fit u32"),
                    )
                })?,
        ),
        Some(other) => {
            return Err(blob_err(
                index,
                format!(
                    "decisions[{position}].chosen_action_id must be an integer or null, got {}",
                    json_kind(other)
                ),
            ));
        }
    };
    let chosen_unresolved = match doc.get("chosen_unresolved") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(reason)) => Some(reason.clone()),
        Some(other) => {
            return Err(blob_err(
                index,
                format!(
                    "decisions[{position}].chosen_unresolved must be a string or null, got {}",
                    json_kind(other)
                ),
            ));
        }
    };
    let observation_hash = blob_field_str(doc, index, "observation_hash").map_err(|err| {
        match err {
            ExpandError::BlobFrame { detail, .. } => {
                blob_err(index, format!("decisions[{position}].observation_hash: {detail}"))
            }
            other => other,
        }
    })?;
    // Any JSON value rides the wire; capture re-parses + re-canonicalizes,
    // so the stored text form cannot fork output either way.
    let actor_observation_json = match doc.get("actor_observation") {
        Some(value) => serde_json::to_string(value)
            .map_err(|err| blob_err(index, format!("decisions[{position}] observation: {err}")))?,
        None => {
            return Err(blob_err(
                index,
                format!("decisions[{position}] missing actor_observation"),
            ));
        }
    };
    Ok(OwnedSimDecisionInput {
        round_idx: blob_field_u32(doc, index, "round_idx").map_err(|err| match err {
            ExpandError::BlobFrame { detail, .. } => {
                blob_err(index, format!("decisions[{position}].round_idx: {detail}"))
            }
            other => other,
        })?,
        seat,
        chosen_action_id,
        chosen_unresolved,
        actor_observation_json,
        observation_hash,
    })
}

fn expand_game_into(
    game: &SimGameInput<'_>,
    builders: &mut ActorBuilders,
) -> Result<(), ExpandError> {
    if game.game_id.is_empty() {
        return Err(ExpandError::EmptyGameId);
    }
    if game.split.is_empty() {
        return Err(ExpandError::EmptySplit);
    }
    if let Some(digest) = game.wall_digest
        && !is_wall_digest(digest) {
            return Err(ExpandError::BadWallDigest {
                game_id: game.game_id.to_string(),
                digest: digest.to_string(),
            });
        }
    for (seq, decision) in game.decisions.iter().enumerate() {
        capture_row(game, *decision, seq, builders)?;
    }
    Ok(())
}

fn capture_row(
    game: &SimGameInput<'_>,
    decision: SimDecisionInput<'_>,
    seq: usize,
    builders: &mut ActorBuilders,
) -> Result<(), ExpandError> {
    // Counter-free ids: seq is the caller's position, never a mutable counter.
    let decision_id = format!("{}:d{:04}", game.game_id, seq);
    let round_id = format!("{}:h{:02}", game.game_id, decision.round_idx);
    if decision.seat > 3 {
        return Err(ExpandError::SeatOutOfRange {
            decision_id,
            seat: decision.seat,
        });
    }
    // S4 pin: chosen null iff unresolved (mirrors the walker capture that
    // builds both together; the envelope carries only the id).
    if decision.chosen_action_id.is_none() == decision.chosen_unresolved.is_none() {
        return Err(ExpandError::ChosenPin {
            decision_id,
        });
    }
    // Parse-only here (`serde_json` NEVER emits identity bytes); canonical
    // bytes come from the single owner below.
    let obs: serde_json::Value =
        serde_json::from_str(decision.actor_observation_json).map_err(|err| {
            ExpandError::ObservationJson {
                decision_id: decision_id.clone(),
                detail: err.to_string(),
            }
        })?;
    let obs_map = obs.as_object().ok_or_else(|| ExpandError::ObservationNotObject {
        decision_id: decision_id.clone(),
    })?;
    firewall_check_obs(&decision_id, obs_map)?;
    // Single canonical serialization: kills the `json.dumps` /
    // `json.loads` round-trip on the consume path and fixes the stored text
    // regardless of the capture's dict order.
    let obs_canonical = canonical_json_bytes(&obs);
    let obs_text = String::from_utf8(obs_canonical.clone()).map_err(|err| {
        ExpandError::ObservationUtf8 {
            decision_id: decision_id.clone(),
            detail: err.to_string(),
        }
    })?;
    let recomputed = observation_hash_for(&obs);
    if recomputed != decision.observation_hash {
        return Err(ExpandError::ObservationHashMismatch {
            decision_id,
            expected: decision.observation_hash.to_string(),
            actual: recomputed,
        });
    }
    // Wall binding: real digest (no mark) vs SIM mark (no digest), mirroring
    // `ReplayRow::to_decision_json` exactly.
    let derivation_hash = match game.wall_digest {
        Some(digest) => derivation_hash_for_walled(
            game.game_id,
            &decision_id,
            &recomputed,
            decision.chosen_action_id,
            digest,
            game.adapter_hash,
        ),
        None => derivation_hash_for(
            game.game_id,
            &decision_id,
            &recomputed,
            decision.chosen_action_id,
            game.adapter_hash,
        ),
    };
    builders.append(
        game.game_id,
        &round_id,
        &decision_id,
        decision.seat,
        game.source_object_id,
        game.split,
        game.rules_hash,
        game.adapter_hash,
        &recomputed,
        game.action_table_hash,
        &derivation_hash,
        &obs_text,
        decision.chosen_action_id,
    );
    Ok(())
}

/// Capture-side firewall leg (mirrors `shard_build._firewall_check_row`):
/// privileged keys top + one-nested rejected, `dora_indicators` bound to
/// `(5,)` exactly.
fn firewall_check_obs(
    decision_id: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), ExpandError> {
    for (key, value) in map {
        if FORBIDDEN_IN_ACTOR.contains(&key.as_str()) {
            return Err(ExpandError::PrivilegedLeak {
                decision_id: decision_id.to_string(),
                field: key.clone(),
            });
        }
        if let Some(nested) = value.as_object() {
            for sub in nested.keys() {
                if FORBIDDEN_IN_ACTOR.contains(&sub.as_str()) {
                    return Err(ExpandError::PrivilegedLeak {
                        decision_id: decision_id.to_string(),
                        field: format!("{key}.{sub}"),
                    });
                }
            }
        }
    }
    match map.get("dora_indicators") {
        Some(serde_json::Value::Array(items)) if items.len() == 5 => Ok(()),
        Some(serde_json::Value::Array(items)) => Err(ExpandError::DoraShape {
            decision_id: decision_id.to_string(),
            detail: format!("got {} indicators, expected 5", items.len()),
        }),
        other => Err(ExpandError::DoraShape {
            decision_id: decision_id.to_string(),
            detail: format!(
                "missing or non-list binding ({})",
                other.map_or("absent", |value| match value {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Object(_) => "object",
                    serde_json::Value::Array(_) => "unreachable",
                })
            ),
        }),
    }
}

fn is_wall_digest(digest: &str) -> bool {
    digest.len() == "sha256:".len() + 64
        && digest.starts_with("sha256:")
        && digest["sha256:".len()..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

struct ActorBuilders {
    game_id: StringBuilder,
    round_id: StringBuilder,
    decision_id: StringBuilder,
    seat: Int64Builder,
    source_object_id: StringBuilder,
    split: StringBuilder,
    rules_hash: StringBuilder,
    adapter_hash: StringBuilder,
    observation_hash: StringBuilder,
    action_table_hash: StringBuilder,
    derivation_hash: StringBuilder,
    actor_observation: StringBuilder,
    chosen_action_id: Int64Builder,
}

impl ActorBuilders {
    /// Pre-size from the capacity pass (mirrors `_count_row_decisions`: one
    /// counting pass over borrowed inputs, no per-row realloc churn).
    fn with_capacity(rows: usize) -> Self {
        let bytes = rows.saturating_mul(64);
        Self {
            game_id: StringBuilder::with_capacity(rows, bytes),
            round_id: StringBuilder::with_capacity(rows, bytes),
            decision_id: StringBuilder::with_capacity(rows, bytes),
            seat: Int64Builder::with_capacity(rows),
            source_object_id: StringBuilder::with_capacity(rows, bytes),
            split: StringBuilder::with_capacity(rows, bytes),
            rules_hash: StringBuilder::with_capacity(rows, bytes),
            adapter_hash: StringBuilder::with_capacity(rows, bytes),
            observation_hash: StringBuilder::with_capacity(rows, bytes),
            action_table_hash: StringBuilder::with_capacity(rows, bytes),
            derivation_hash: StringBuilder::with_capacity(rows, bytes),
            actor_observation: StringBuilder::with_capacity(rows, bytes),
            chosen_action_id: Int64Builder::with_capacity(rows),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append(
        &mut self,
        game_id: &str,
        round_id: &str,
        decision_id: &str,
        seat: u8,
        source_object_id: &str,
        split: &str,
        rules_hash: &str,
        adapter_hash: &str,
        observation_hash: &str,
        action_table_hash: &str,
        derivation_hash: &str,
        actor_observation: &str,
        chosen_action_id: Option<u32>,
    ) {
        self.game_id.append_value(game_id);
        self.round_id.append_value(round_id);
        self.decision_id.append_value(decision_id);
        self.seat.append_value(i64::from(seat));
        self.source_object_id.append_value(source_object_id);
        self.split.append_value(split);
        self.rules_hash.append_value(rules_hash);
        self.adapter_hash.append_value(adapter_hash);
        self.observation_hash.append_value(observation_hash);
        self.action_table_hash.append_value(action_table_hash);
        self.derivation_hash.append_value(derivation_hash);
        self.actor_observation.append_value(actor_observation);
        match chosen_action_id {
            Some(id) => self.chosen_action_id.append_value(i64::from(id)),
            None => self.chosen_action_id.append_null(),
        }
    }

    fn finish(self, schema: SchemaRef) -> Result<RecordBatch, ExpandError> {
        let mut builders = self;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(builders.game_id.finish()),
            Arc::new(builders.round_id.finish()),
            Arc::new(builders.decision_id.finish()),
            Arc::new(builders.seat.finish()),
            Arc::new(builders.source_object_id.finish()),
            Arc::new(builders.split.finish()),
            Arc::new(builders.rules_hash.finish()),
            Arc::new(builders.adapter_hash.finish()),
            Arc::new(builders.observation_hash.finish()),
            Arc::new(builders.action_table_hash.finish()),
            Arc::new(builders.derivation_hash.finish()),
            Arc::new(builders.actor_observation.finish()),
            Arc::new(builders.chosen_action_id.finish()),
        ];
        RecordBatch::try_new(schema, columns)
            .map_err(|err| ExpandError::Arrow { detail: err.to_string() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parity::observation_hash_for;

    fn fixture_obs(dora_len: usize) -> serde_json::Value {
        let dora: Vec<serde_json::Value> = (0..dora_len)
            .map(|i| serde_json::Value::String(format!("d{i}")))
            .collect();
        serde_json::json!({
            "dora_indicators": dora,
            "seat": 0,
            "phase": "draw",
        })
    }

    fn obs_text_and_hash(obs: &serde_json::Value) -> (String, String) {
        let text = serde_json::to_string(obs).expect("fixture encodes");
        let hash = observation_hash_for(obs);
        (text, hash)
    }

    #[test]
    fn batch_preserves_caller_order_across_games() {
        let obs = fixture_obs(5);
        let (text, hash) = obs_text_and_hash(&obs);
        let mk = |round: u32, seat: u8| SimDecisionInput {
            round_idx: round,
            seat,
            chosen_action_id: Some(7),
            chosen_unresolved: None,
            actor_observation_json: &text,
            observation_hash: &hash,
        };
        let late = [mk(1, 2), mk(0, 0)];
        let early = [mk(0, 1)];
        let games = [
            SimGameInput {
                game_id: "gB",
                source_object_id: "obj-b",
                split: "train",
                rules_hash: "sha256:r",
                adapter_hash: "sha256:a",
                action_table_hash: "sha256:t",
                wall_digest: None,
                decisions: &late,
            },
            SimGameInput {
                game_id: "gA",
                source_object_id: "obj-a",
                split: "train",
                rules_hash: "sha256:r",
                adapter_hash: "sha256:a",
                action_table_hash: "sha256:t",
                wall_digest: None,
                decisions: &early,
            },
        ];
        let expanded = expand_batch(&games).expect("valid inputs expand");
        assert_eq!(expanded.num_rows(), 3);
        let ids = expanded.decision_ids().expect("decision ids");
        assert_eq!(ids.value(0), "gB:d0000");
        assert_eq!(ids.value(1), "gB:d0001");
        assert_eq!(ids.value(2), "gA:d0000");
    }

    #[test]
    fn decision_seq_follows_position_not_round() {
        // Caller order wins even when round_idx runs backwards: seq numbers
        // the capture position, exactly like the oracle's append-order seq.
        let obs = fixture_obs(5);
        let (text, hash) = obs_text_and_hash(&obs);
        let decisions = [
            SimDecisionInput {
                round_idx: 3,
                seat: 0,
                chosen_action_id: Some(1),
                chosen_unresolved: None,
                actor_observation_json: &text,
                observation_hash: &hash,
            },
            SimDecisionInput {
                round_idx: 0,
                seat: 1,
                chosen_action_id: Some(2),
                chosen_unresolved: None,
                actor_observation_json: &text,
                observation_hash: &hash,
            },
        ];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        let expanded = expand_batch(&games).expect("valid inputs expand");
        let ids = expanded.decision_ids().expect("decision ids");
        assert_eq!(ids.value(0), "g:d0000");
        assert_eq!(ids.value(1), "g:d0001");
        let rounds: &StringArray = expanded
            .batch()
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("round_id is utf8");
        assert_eq!(rounds.value(0), "g:h03");
        assert_eq!(rounds.value(1), "g:h00");
    }

    #[test]
    fn schema_matches_actor_fields_exact() {
        let schema = actor_schema();
        assert_eq!(schema.fields().len(), ACTOR_FIELDS.len());
        for (index, expected) in ACTOR_FIELDS.iter().enumerate() {
            let field = schema.field(index);
            assert_eq!(field.name(), expected, "column {index} name");
            match *expected {
                "seat" | "chosen_action_id" => {
                    assert_eq!(*field.data_type(), DataType::Int64, "column {expected}");
                    assert_eq!(
                        field.is_nullable(),
                        *expected == "chosen_action_id",
                        "column {expected} nullability"
                    );
                }
                _ => {
                    assert_eq!(*field.data_type(), DataType::Utf8, "column {expected}");
                    assert!(!field.is_nullable(), "column {expected} non-null");
                }
            }
        }
        // The built batch carries exactly this schema (no drift at finish).
        let expanded = expand_batch(&[]).expect("empty games give an empty batch");
        assert_eq!(expanded.num_rows(), 0);
        assert_eq!(expanded.batch().schema(), schema);
    }

    #[test]
    fn firewall_rejects_leaks_and_dora_shim() {
        // (4,) dora shim rejected.
        let shim = fixture_obs(4);
        let (shim_text, shim_hash) = obs_text_and_hash(&shim);
        let decisions = [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: Some(1),
            chosen_unresolved: None,
            actor_observation_json: &shim_text,
            observation_hash: &shim_hash,
        }];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        assert!(matches!(
            expand_batch(&games),
            Err(ExpandError::DoraShape { .. })
        ));
        // Top-level privileged leak rejected.
        let mut leaked = fixture_obs(5);
        leaked["wall"] = serde_json::json!([1, 2, 3]);
        let (leak_text, leak_hash) = obs_text_and_hash(&leaked);
        let decisions = [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: Some(1),
            chosen_unresolved: None,
            actor_observation_json: &leak_text,
            observation_hash: &leak_hash,
        }];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        assert!(matches!(
            expand_batch(&games),
            Err(ExpandError::PrivilegedLeak { .. })
        ));
        // One-level-nested privileged leak rejected.
        let mut nested = fixture_obs(5);
        nested["context"] = serde_json::json!({"dead_wall": [4, 5]});
        let (nested_text, nested_hash) = obs_text_and_hash(&nested);
        let decisions = [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: Some(1),
            chosen_unresolved: None,
            actor_observation_json: &nested_text,
            observation_hash: &nested_hash,
        }];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        assert!(matches!(
            expand_batch(&games),
            Err(ExpandError::PrivilegedLeak { .. })
        ));
    }

    #[test]
    fn rejects_hash_mismatch_and_chosen_pin_breaks() {
        let obs = fixture_obs(5);
        let (text, _) = obs_text_and_hash(&obs);
        let wrong_hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        // Wrong capture hash fails closed (never silently rebound).
        let decisions = [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: Some(1),
            chosen_unresolved: None,
            actor_observation_json: &text,
            observation_hash: wrong_hash,
        }];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        assert!(matches!(
            expand_batch(&games),
            Err(ExpandError::ObservationHashMismatch { .. })
        ));
        // chosen null WITHOUT an unresolved reason breaks the S4 pin.
        let (_, hash) = obs_text_and_hash(&obs);
        let decisions = [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: None,
            chosen_unresolved: None,
            actor_observation_json: &text,
            observation_hash: &hash,
        }];
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        assert!(matches!(
            expand_batch(&games),
            Err(ExpandError::ChosenPin { .. })
        ));
    }

    #[test]
    fn derivation_binds_wall_or_sim_mark() {
        let obs = fixture_obs(5);
        let (text, hash) = obs_text_and_hash(&obs);
        let walled_digest = format!("sha256:{}", "ab".repeat(32));
        let mk_decisions = || [SimDecisionInput {
            round_idx: 0,
            seat: 0,
            chosen_action_id: Some(9),
            chosen_unresolved: None,
            actor_observation_json: &text,
            observation_hash: &hash,
        }];
        // Wall-less binds the SIM mark (never an invented digest).
        let decisions = mk_decisions();
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &decisions,
        }];
        let expanded = expand_batch(&games).expect("wall-less expands");
        let deriv: &StringArray = expanded
            .batch()
            .column(10)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("derivation_hash is utf8");
        assert_eq!(
            deriv.value(0),
            derivation_hash_for("g", "g:d0000", &hash, Some(9), "sha256:a")
        );
        // Walled binds the real digest with no mark.
        let decisions = mk_decisions();
        let games = [SimGameInput {
            game_id: "g",
            source_object_id: "obj",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: Some(&walled_digest),
            decisions: &decisions,
        }];
        let expanded = expand_batch(&games).expect("walled expands");
        let deriv: &StringArray = expanded
            .batch()
            .column(10)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("derivation_hash is utf8");
        assert_eq!(
            deriv.value(0),
            derivation_hash_for_walled("g", "g:d0000", &hash, Some(9), &walled_digest, "sha256:a")
        );
    }

    #[test]
    fn deterministic_replay_same_inputs_identical_batch() {
        let obs = fixture_obs(5);
        let (text, hash) = obs_text_and_hash(&obs);
        let run = || {
            let decisions = [
                SimDecisionInput {
                    round_idx: 0,
                    seat: 0,
                    chosen_action_id: Some(3),
                    chosen_unresolved: None,
                    actor_observation_json: &text,
                    observation_hash: &hash,
                },
                SimDecisionInput {
                    round_idx: 0,
                    seat: 1,
                    chosen_action_id: None,
                    chosen_unresolved: Some("action-id-unresolved:ron"),
                    actor_observation_json: &text,
                    observation_hash: &hash,
                },
            ];
            let games = [SimGameInput {
                game_id: "g7",
                source_object_id: "obj-7",
                split: "eval",
                rules_hash: "sha256:r",
                adapter_hash: "sha256:a",
                action_table_hash: "sha256:t",
                wall_digest: None,
                decisions: &decisions,
            }];
            expand_batch(&games).expect("valid inputs expand")
        };
        // NOTE: each `run()` builds and drops its own borrowed slices; the
        // returned batch owns every byte, so cross-call equality is exact.
        let first = run();
        let second = run();
        assert_eq!(first.batch(), second.batch());
        assert_eq!(first.num_rows(), 2);
    }

    #[test]
    fn owned_entry_matches_borrowed_entry() {
        let obs = fixture_obs(5);
        let (text, hash) = obs_text_and_hash(&obs);
        let borrowed_decisions = [
            SimDecisionInput {
                round_idx: 0,
                seat: 2,
                chosen_action_id: Some(11),
                chosen_unresolved: None,
                actor_observation_json: &text,
                observation_hash: &hash,
            },
            SimDecisionInput {
                round_idx: 1,
                seat: 3,
                chosen_action_id: None,
                chosen_unresolved: Some("action-id-unresolved:pon"),
                actor_observation_json: &text,
                observation_hash: &hash,
            },
        ];
        let borrowed_games = [SimGameInput {
            game_id: "go",
            source_object_id: "obj-o",
            split: "train",
            rules_hash: "sha256:r",
            adapter_hash: "sha256:a",
            action_table_hash: "sha256:t",
            wall_digest: None,
            decisions: &borrowed_decisions,
        }];
        let owned_games = [OwnedSimGameInput {
            game_id: "go".to_string(),
            source_object_id: "obj-o".to_string(),
            split: "train".to_string(),
            rules_hash: "sha256:r".to_string(),
            adapter_hash: "sha256:a".to_string(),
            action_table_hash: "sha256:t".to_string(),
            wall_digest: None,
            decisions: vec![
                OwnedSimDecisionInput {
                    round_idx: 0,
                    seat: 2,
                    chosen_action_id: Some(11),
                    chosen_unresolved: None,
                    actor_observation_json: text.clone(),
                    observation_hash: hash.clone(),
                },
                OwnedSimDecisionInput {
                    round_idx: 1,
                    seat: 3,
                    chosen_action_id: None,
                    chosen_unresolved: Some("action-id-unresolved:pon".to_string()),
                    actor_observation_json: text.clone(),
                    observation_hash: hash.clone(),
                },
            ],
        }];
        let from_borrowed = expand_batch(&borrowed_games).expect("borrowed expands");
        let from_owned = expand_batch_owned(&owned_games).expect("owned expands");
        assert_eq!(from_borrowed.batch(), from_owned.batch());
    }

    #[test]
    fn assembler_round_trip_matches_struct_call() {
        let obs = fixture_obs(5);
        let (_, hash) = obs_text_and_hash(&obs);
        let blob = serde_json::json!({
            "game_id": "ga",
            "source_object_id": "obj-ga",
            "split": "train",
            "rules_hash": "sha256:r",
            "adapter_hash": "sha256:a",
            "action_table_hash": "sha256:t",
            "wall_digest": null,
            "decisions": [
                {
                    "round_idx": 2,
                    "seat": 1,
                    "chosen_action_id": 4,
                    "chosen_unresolved": null,
                    "observation_hash": hash,
                    "actor_observation": obs,
                }
            ],
        });
        let bytes = serde_json::to_vec(&blob).expect("fixture encodes");
        let owned = assemble_owned_games(&[bytes]).expect("blob assembles");
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].game_id, "ga");
        assert_eq!(owned[0].decisions.len(), 1);
        assert_eq!(owned[0].decisions[0].seat, 1);
        let expanded = expand_batch_owned(&owned).expect("assembled expands");
        assert_eq!(expanded.num_rows(), 1);
        assert_eq!(expanded.decision_ids().expect("decision ids").value(0), "ga:d0000");
        // Second blob shape failure reports its index, not the first.
        let bad = assemble_owned_games(&[
            serde_json::to_vec(&blob).expect("fixture encodes"),
            b"{not json".to_vec(),
        ]);
        assert!(matches!(
            bad,
            Err(ExpandError::BlobFrame { index: 1, .. })
        ));
    }
}
