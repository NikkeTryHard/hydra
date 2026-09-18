//! Frozen replay orchestration entry points (moved verbatim from the root).
//!
//! `(text, object_id, table)` stays pinned so the dump example, the
//! `parity_84` join, and the PyO3 handoff share one call shape. Walled
//! inputs bind the real wall digest (no SIM mark); wall-less inputs bind
//! the SIM mark (no digest, never invented).

use crate::decisions::{ActionTable, walk_game, walk_game_with_version};
use crate::engine::EngineVersion;
use crate::parity::{Quarantine, ReplayRow};
use crate::stream::{GameReject, parse_game};

/// Replay one framed game text into rows or a whole-game quarantine.
pub fn replay_game_text(
    text: &str,
    object_id: &str,
    table: &ActionTable,
) -> Result<Vec<ReplayRow>, Quarantine> {
    let game = parse_game(text, object_id).map_err(|reject| match reject {
        GameReject::Framing(detail) => Quarantine {
            game_id: object_id.to_string(),
            reason_code: "framing".to_string(),
            detail,
        },
    })?;
    walk_game(&game, table).map_err(|reject| Quarantine {
        game_id: reject.game_id,
        reason_code: reject.code,
        detail: reject.detail,
    })
}

/// Version-selected replay entry (V1 default; V2 runs the single-pass
/// live-engine semantics). Shares the frozen quarantine mapping plus the S7
/// walled digest binding.
pub fn replay_game_text_with_version(
    text: &str,
    object_id: &str,
    table: &ActionTable,
    version: EngineVersion,
) -> Result<Vec<ReplayRow>, Quarantine> {
    let game = parse_game(text, object_id).map_err(|reject| match reject {
        GameReject::Framing(detail) => Quarantine {
            game_id: object_id.to_string(),
            reason_code: "framing".to_string(),
            detail,
        },
    })?;
    walk_game_with_version(&game, table, version).map_err(|reject| Quarantine {
        game_id: reject.game_id,
        reason_code: reject.code,
        detail: reject.detail,
    })
}

/// Sim-capture column builders (Phase 3 expand side): per-decision sim
/// outputs -> one actor `RecordBatch`.
///
/// Declared here (not at the crate root) because `lib.rs` is owned by the
/// shard-manifest sibling: `hydra_shard::replay::expand` is the canonical
/// path, and no crate-root re-export exists or is wanted.
#[path = "expand.rs"]
pub mod expand;

/// Batch sim-capture entry: caller-ordered sim inputs -> one actor batch.
///
/// Thin entry over [`expand::expand_batch`]; the sim itself stays Python this
/// phase (the engine-input boundary is documented on [`expand`]).
pub fn expand_batch_games(
    games: &[expand::SimGameInput<'_>],
) -> Result<expand::ExpandedBatch, expand::ExpandError> {
    expand::expand_batch(games)
}
