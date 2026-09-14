#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
//! Test-only escape: production code is no-panic-by-construction (workspace
//! deny lints); tests keep plain unwrap/expect.
//! hydra-feed: hot S1-S4 ingest → gate → walk → fill path (sole rayon pool).
//!
//! DAG: this crate imports NOTHING from the downstream crates
//! (CI triple-grep enforces; data flows feed → shard/bridge one-way only).
//! FORBIDS (see manifest — this crate must never gain): Python bindings,
//! memory maps, hashing, or columnar formats.
//!
//! Modules: `gate` (§6.2), `ingest` + `tiles` (§6.1), `ledger` + `walk`
//! (§6.3: u8 ledger DIRECT to minimal hot planes; engine.rs rules ported
//! to u8, never imported), `fill` (§6.4 stage-then-commit).
pub mod fill;
pub mod gate;
pub mod ingest;
pub mod ledger;
pub mod tiles;
pub mod walk;
