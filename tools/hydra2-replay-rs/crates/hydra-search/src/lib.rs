#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
//! hydra-search: offline search arena + selection math (ONE new crate, B5).
//!
//! Offline batch throughput vs online act latency/deadline/fallback are
//! different threading/lifetime/failure modes (polars-bridge +
//! tantivy-segments + vLLM-EngineCore steals): this crate owns the arena SoA
//! (`arena` + `keys` + `rng`) plus the six planner families
//! (`ismcts`/`gumbel`/`despot`/`pbrf`/`local`/`joint`). There is NO second
//! `hydra-eval` crate (B5 DENIED): eval CPU control
//! (`eval::{wall,blocks,schedule,telemetry,partition,statistics}`) plus the
//! persistence family live as sibling-owned submodules added by EvalControl,
//! NOT declared here.
//!
//! Ownership (disjoint): SearchArena owns this manifest + `lib.rs` +
//! `arena/keys/rng/ismcts/ismcts_driver/gumbel/despot/pbrf/local/joint` ONLY. EvalControl
//! owns `eval::*` + `persistence_*`. SearchBridge owns `bridge search.rs` +
//! the Python act shim. Nobody else touches those files.
//!
//! Invariants:
//! - B1 SHA-256 identity ONLY. `gumbel::deterministic_gumbel` replicates
//!   `gumbel_core.py:117-156` bytes VERBATIM
//!   (`payload=f"{case}:{seat}:{cand}:{action}"`, single domain prefix
//!   `sha256(b"gumbel_root_v1"+payload)`, `u=(u64BE(h[:8])+0.5)/2^64`,
//!   clip `1e-12`, `G=-ln(-ln(u))`, clamp `+-20`). Philox words for Gumbels
//!   are DELETED; Philox serves ONLY NEW segment/scenario substreams (`rng`).
//! - B2 splits stay the `torch.randperm` oracle behind KAT. This crate
//!   provides NO `philox_split_perm`; sampling uses Lemire `below` ONLY.
//! - B3 canon-wins: single `feed::canon` site. Search owns arena
//!   tables + selection; canon owns bytes. Arena/despot/pbrf call
//!   `feed::canon`/`feed::digest` (shared crate), never a second printer.
//! - B4 persistence family is sibling-owned (per-arm
//!   `selected+vectors+counters` + `deterministic_gumbel_for_arm` golden
//!   lands with EvalControl, not here).
//! - B5 pools: `FeedPool` sole GLOBAL + arena SCOPED segments. Budget split
//!   `max_trans/threads` remainder to seg0 + deterministic `0..T` merge +
//!   single-lock publish (`commit=block+publish`, opstamp=completed sims).
//! - `BATCH`: canonical owner is bridge `columnar::BATCH` (bridge-owned,
//!   passed in as a plain width arg). This crate defines NO second width
//!   const; widths arrive as plain args.
//! - dora `(5,)` untouched. NO Rust GPU math (StudentModel/distill stay
//!   torch). Candidate0 single encode+evaluate STAYS torch, never ported.
//! - NEVER `hash()` (PYTHONHASHSEED): tie-breaks are byte-ordered or
//!   `sha256(f"{a}")` hex-ordered. Empty legal is `completed=false`
//!   (caller MUST invoke candidate0), never a retry.
//!
//! Layout: one module per planner family; shared `arena/keys/rng` core.
//! All floats are f64 with stated eps; tie-breaks byte-ordered.
//! Eps gates: `1e-12` UCT/PUCT/halving/regret, `1e-9` value-tie/partition,
//! `1e-6` distribution floor.


pub mod arena;
pub mod belief;
pub mod despot;
pub mod gumbel;
pub mod gumbel_driver;
pub mod ismcts;
pub mod ismcts_driver;
pub mod joint;
pub mod keys;
pub mod local;
pub mod local_driver;
pub mod pbrf;
pub mod rng;
pub mod eval;
pub mod persistence_factorial;
pub mod persistence_kernel;
pub mod persistence_planner;
pub mod persistence_report;
pub mod persistence_spec;

// ---------------------------------------------------------------------------
// Shared contract error (fail-closed, numeric-or-static, never a panic)
// ---------------------------------------------------------------------------

/// Search failure taxonomy: mirrors `ContractError` edges, never a panic.
///
/// Cold path only (allocation on error is fine; hot selection paths return
/// `Result` without allocating on `Ok`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    /// Legal set empty where a selection was required. Callers map this to
    /// `completed=false` (candidate0 fallback), never a retry.
    EmptyLegal,
    /// Duplicate action id in an allegedly unique legal set.
    DuplicateAction {
        /// Rejected duplicate action id.
        action: u32,
    },
    /// Root/opponent seat outside `0..4`.
    InvalidSeat {
        /// Rejected seat.
        seat: u32,
    },
    /// Malformed scalar argument (static detail, no allocation on hot path).
    InvalidArg {
        /// What was rejected (`'static` so `Err` carries no heap on hot paths
        /// that use this variant).
        detail: &'static str,
    },
    /// Non-finite float where only finite math is contract-legal.
    NonFinite {
        /// Which stage observed the non-finite value.
        context: &'static str,
    },
    /// Zero/non-finite conditioning mass (caller must take the miss path).
    ZeroMass,
    /// Packet successors do not form a disjoint exhaustive partition.
    BadPartition,
    /// Likelihood outside `(0,1]` or non-finite (double-count audit).
    BadLikelihood,
    /// Canonical-bytes step failed (delegated `feed::canon` error text).
    Canon {
        /// Delegated detail.
        detail: String,
    },
    /// Single-lock publish poisoned (caller retries the block, never spins).
    Lock,
}

impl core::fmt::Display for SearchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SearchError::EmptyLegal => write!(f, "legal set must be non-empty"),
            SearchError::DuplicateAction { action } => {
                write!(f, "duplicate action_id {action}")
            }
            SearchError::InvalidSeat { seat } => {
                write!(f, "seat must be 0..3, got {seat}")
            }
            SearchError::InvalidArg { detail } => {
                write!(f, "invalid search argument: {detail}")
            }
            SearchError::NonFinite { context } => {
                write!(f, "non-finite value at {context}")
            }
            SearchError::ZeroMass => write!(f, "conditioning has zero/nonfinite mass"),
            SearchError::BadPartition => write!(f, "packet successors are not a partition"),
            SearchError::BadLikelihood => write!(f, "likelihood must be finite in (0,1]"),
            SearchError::Canon { detail } => write!(f, "canon failed: {detail}"),
            SearchError::Lock => write!(f, "arena publish lock poisoned"),
        }
    }
}

impl std::error::Error for SearchError {}
