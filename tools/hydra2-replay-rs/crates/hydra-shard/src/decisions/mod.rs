//! Log-order decision walker: framed MJAI events -> decision rows.
//!
//! Reference (read-only): `src/hydra2/engines/riichienv/log_replay.py`
//! row-construction order (`_walk_game` dispatch, `_do_*` handlers,
//! `_capture_row` capture-then-emit, `_peek_window_claim`,
//! `_resolve_dora` string rule, `_TRACK` take order). Order reused; code
//! written fresh.
//!
//! Engine-owned surfaces (offered legals, riichi candidates, ankan/kakan
//! availability, ron/window state) come from the frozen engine-out
//! protocol (`crate::engine`, native v1 backend pinning the drained-oracle
//! semantics rule-by-rule) — never guessed from the log. The take ledger
//! still owns ORDER (tehais seats 0..3, live draws, rinshan draws);
//! everything else (ids, seats, chosen tiles, concealed strings, dora
//! strings, history kinds, wall countdown, tracked scores) mirrors the
//! oracle rule.

pub(crate) mod handlers_close;
pub(crate) mod handlers_core;
pub(crate) mod table;
pub(crate) mod walker;

pub use handlers_core::{count_row_decisions, walk_game, walk_game_with_version};
pub use table::{ACTION_TABLE_ARTIFACT_TYPE, ACTION_TABLE_SCHEMA_VERSION, ActionTable, Template};
pub use walker::{WalkReject, WalkerMode, source_offset};
