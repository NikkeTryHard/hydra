//! Walled + wall-less MJAI replay driver (slices 1-7): framed Tenhou games
//! -> actor decision rows with parity numbers vs the Python oracles.
//!
//! Pipeline (mirrors `log_replay.replay_game` row-construction order and
//! `replay_expand.ReplayExpander` mode machine):
//! 1. [`mjai_event::frame_events`] — strict JSONL framing (one shared gate
//!    for walled + wall-less);
//! 2. [`stream::parse_game`] — game identity (explicit id else
//!    `game-<sha12(object_id)>`) plus the optional 136-tile wall
//!    (`wall`/`wall_tiles`/`tiles`, permutation 0..135);
//! 3. `decisions::prescan` — bare-dora / double-ron whole-game quarantine;
//! 4. [`decisions::walk_game`] — log-order walk with the take ledger
//!    (tehais seats 0..3, live draws, rinshan draws), capture-then-emit rows
//!    through the S7 mode machine (expected-actor / window-pending /
//!    draw-mode / terminal) with `sim.reset(wall)` digest mint + `apply`
//!    order identical to Python.
//!
//! Quarantine-style: unknowns fail closed with named reason codes; the
//! privileged fields (seat-filtered only in the oracle) are unrepresentable
//! in [`hydra2_row::ReplayRow`] by construction (the walled `wall_digest`
//! rides the derivation + `wall_id`, never the actor envelope). No torch,
//! no tensors, no PyO3 here — the bridge skeleton is out of scope.
pub mod decisions;
pub mod decision_ids;
pub mod engine;
pub mod hydra2_row;
pub mod mjai_event;
pub mod privileged;
pub mod provenance;
pub mod py_stream;
pub mod stream;
pub mod tile;

pub use decision_ids::{decision_id, parse_decision_id, parse_round_id, round_id};
pub use decisions::{ActionTable, WalkReject, WalkerMode, count_row_decisions, walk_game, walk_game_with_version};
pub use engine::{
    DrawOffer, EngineDesync, EngineVersion, KAN_MIN_WALL, LIVE_WALL_BASE, RIICHI_MIN_SCORE,
    RIICHI_MIN_WALL, ResponderView, SeatView, draw_offer, kyushu_first_draw, tenpai_discards,
    window_open, win_shape_14,
};
pub use hydra2_row::{
    ACTOR_FIELDS, RowProvenance, SIM_DERIVATION_MARK, WALLED_PROJECTION, WALL_LESS_PROJECTION,
    canonical_json_bytes, ChosenAction, Quarantine, ReplayRow, derivation_hash_for,
    derivation_hash_for_walled, observation_hash_for, schedule_id_for, wall_schedule_digest,
};
pub use privileged::{
    PrivilegedRow, check_privileged_label, expand_privileged_rows, final_scores,
    ranks_from_final_scores, validate_privileged_ranks,
};
pub use provenance::{
    PINNED_ACTION_TABLE_HASH, PINNED_ADAPTER_HASH, PINNED_RULES_HASH, RulesInfo,
    adapter_hash, load_baked_rules, load_rules_manifest,
};
pub use py_stream::{
    FORBIDDEN_IN_ACTOR, RawNext, ReplayStream, StreamError, StreamOpen, StreamStats,
};
pub use stream::{GameReject, ParsedGame, parse_game};

/// Replay one framed game text into rows or a whole-game quarantine.
///
/// Frozen entry point: `(text, object_id, table)` stays pinned so the dump
/// CLI, the `parity_84` join, and the PyO3 handoff share one call shape.
/// Walled inputs bind the real wall digest (no SIM mark); wall-less inputs
/// bind the SIM mark (no digest, never invented).
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
