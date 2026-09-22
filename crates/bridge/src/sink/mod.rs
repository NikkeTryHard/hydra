//! S6 sink — quarantine lineage + accounting + terminal-hora han/fu
//! (scoring units / minipoints; cold leg).
//!
//! Cold only: everything here runs AFTER the hot
//! verdict (post-close lineage IO, accounting asserts, rare terminal-hora
//! recompute). The hot path never calls in: gate/walk/fill take no dependency
//! on this module. Same-crate `stream.rs` calls only [`quarantine_reason_name`]
//! while capturing quarantines, never the file/score legs.
//!
//! Shape (reason taxonomy + post-claim quarantine rule):
//! - reason renderer: [`quarantine_reason_name`] (`reason < 10` → gate names,
//!   else walk names; unknown bytes stay `"other"` and MUST grow the
//!   vocabulary, never pass silently — the histogram test asserts `"other"`
//!   is absent on the corpus).
//! - closed sink taxonomy: [`sink_bucket`] over [`SINK_VOCABULARY`] (6
//!   buckets: framing/vocab/conservation/hora-mismatch/wall-perm/history-cap).
//! - lineage file per quarantine: [`QuarantineLineage`]
//!   `{identity, event_idx, reason, obs_hash u64}` JSON lines +
//!   [`join_lineage`] back to consumed games.
//! - accounting assert path: [`check_accounting`] (`ok + quarantined + staged
//!   == consumed`; G5 `games_quarantined + games_ok == games` at drain).
//! - han/fu recompute HERE only, terminal hora only (rare): [`score_hora`] +
//!   [`reconcile_hora`] (G4 scores-vs-delta). Report-only: `Mismatch` goes to
//!   cold review, never into hot accept/reject (the hot walk keeps its
//!   shape-only hora check by design).

pub mod reconcile;
pub mod scoring_eval;
pub mod scoring_tables;
pub mod taxonomy;

pub use reconcile::{reconcile_hora, score_hora};
pub use scoring_tables::{
    HAN_YAKUMAN, HanFu, HoraInput, MeldKind, OpenMeld, Payment, Reconcile, ScoreOutcome, TILE_CHUN,
    TILE_E, TILE_HAKU, TILE_HATSU, TILE_N, TILE_S, TILE_W, WaitKind, WinKind, hora_payment,
};
pub use taxonomy::{
    AccountingMismatch, LineageError, QuarantineLineage, SINK_CONSERVATION, SINK_FRAMING,
    SINK_HISTORY_CAP, SINK_HORA_MISMATCH, SINK_VOCAB, SINK_VOCABULARY, SINK_WALL_PERM,
    check_accounting, join_lineage, quarantine_reason_name, read_lineage_file, sink_bucket,
    write_lineage_file,
};
