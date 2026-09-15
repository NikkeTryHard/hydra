//! S3 walk — u8 ledger DIRECT to minimal hot planes (`feed::walk`).
//!
//! §6.3 shapes: u8 ledger DIRECT to minimal hot planes (this module).
//!
//! Engine move: the `engine.rs → feed::walk` rules live here as pure `u8`
//! functions (win shape, tenpai (one tile from a win), chi/pon/window
//! responder predicates, draw offers). `hydra-shard/src/engine.rs` keeps
//! serving the cold/oracle path;
//! this module ports its V1 (drained-oracle) semantics rule-by-rule onto the
//! copy-slot ledger — no `String`, no `HashMap`, no heap on the row path.
//! V2 (live single-pass deltas) is out of scope, with one evidence-backed
//! exception: the nine-terminals abort offer on qualifying first draws
//! matches live riichienv (differential-proven; user-approved V2 item).
//! Chi-completeness, kuikae-strictness (no turn-around swap after a chi),
//! and take conservation stay V1.
//!
//! Oracle pin: the wall-less replay oracle reports COLLAPSED takes on
//! pre-reach discards (F1 golden `[192,193,193,193,196]`), while the walled
//! expander reports true takes. This walk has no wall takes by
//! construction, so F1 bitwise matches the wall-less collapse (== shard
//! walk, == F7 frozen rows). Walled true-takes assert ONLY under F3 via
//! the borrowed context digest — never as an F1 target.
//!
//! Allocation audit (row hot path — tsumo/dahai/claim/kan/hora dispatch,
//! offers, legal, window eval + peek, plane appends):
//! - NO `String` / `format!` / `HashMap` / `HashSet` / `Vec` / `sort` with
//!   allocation. Sorting is insertion sort over stack arrays.
//! - `serde_json::Value` parses run ONLY at kyoku boundaries (`start_kyoku`
//!   headers, `ryukyoku` reason: ≤2 parses per kyoku, amortized, never per
//!   row). Hora `deltas`/`tsumo` read spans byte-wise. Everything else is
//!   borrowed ingest fields.
//! - `Vec` pushes are output buffering only (per-seat history kinds,
//!   Scratch plane bytes): amortized, never per-row realloc in steady state.
//! - Non-sampled games allocate NOTHING: the gate skips them before the
//!   walk is ever entered.
//!
//! Kill list (vs cold reference `hydra-shard/src/decisions.rs`):
//! - window re-render (:982-1026, 3-4 wasted responder evals per discard)
//!   → [`WalkState`] peek-first log-ron fact + per-discard eval cache.
//! - `HashSet` mask builds + `sort_unstable` + `mask.to_vec` → stack
//!   `[u32; 32]` + insertion sort, chosen forcing by id arithmetic.
//! - `chosen.kind_str()` + table `HashMap` lookup → [`chosen_id`] /
//!   [`chosen_claim_id`] closed-form arithmetic pinned to action-table
//!   digest [`crate::ledger::ACTION_TABLE_DIGEST`] (verified entry-wise in
//!   the module tests).
mod driver;
mod handlers_claim;
mod handlers_close;
mod handlers_core;
mod headers;
mod ids;
mod offers;
mod shape;
mod spans;
#[cfg(test)]
mod tests;

pub use driver::walk_game;
pub use ids::{
    ABORT_NINE_TERMINALS_ID, EXHAUSTIVE_MIN_DRAWS, KAN_MIN_WALL, KYUSHU_DISTINCT_YAOCHU,
    KYUSHU_MAX_DRAWS, LIVE_WALL_BASE, MAX_DRAWS, PASS_ID, RIICHI_MIN_SCORE, RIICHI_MIN_WALL,
    UNRESOLVED_ID, chosen_claim_id, chosen_id, offset_idx, source_offset,
};
pub use offers::legal_ids_sorted;
pub use shape::win_shape_14;
