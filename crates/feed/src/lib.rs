#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
//! Test-only escape: production code is no-panic-by-construction (workspace
//! deny lints); tests keep plain unwrap/expect.
//! hydra-feed: hot S1-S4 ingest → gate → walk → fill path (sole rayon pool).
//!
//! DAG: this crate imports NOTHING from the downstream crates
//! (CI triple-grep enforces; data flows feed → shard/bridge one-way only).
//! FORBIDS (see manifest — this crate must never gain): Python bindings,
//! memory maps, or columnar formats. Hashing/canon/RNG ARE feed-owned:
//! `canon` (`serde_jcs 0.2.0` ONLY-exact bytes) + `digest` (`sha2 0.11` SHA-256 identity)
//! + `rng` (Philox NEW streams + Lemire) + `fixed` (utility fixed-point) are
//!   the single owners every later phase calls; packet/columnar/search/eval
//!   CALL canon (never a second printer). Attestation seals (M17) canonicalize
//!   through `canon` exactly like every other seal path.
//!
//! Modules: `decode` (typed decode over framed bytes) + `framer` (packet
//! framing + verified zstd decode), `gate` (frame/gate verdicts), `ingest`
//! + `tiles` (span framing + kind/tile LUTs), `ledger` + `walk` (u8 ledger
//!   DIRECT to minimal hot planes; engine.rs rules ported to u8, never
//!   imported), `fill` (stage-then-commit), `manifest` + `partition` + `rows`
//!   (split/manifest math calling `canon`/`digest`, never a second printer),
//!   `quarantine` + `validate` (taxonomy-closed verdicts over decoded games).
pub mod canon;
pub mod census;
pub mod chain;
pub mod decode;
pub mod digest;
pub mod fill;
pub mod fixed;
pub mod framer;
pub mod gate;
pub mod ingest;
pub mod ledger;
pub mod manifest;
pub mod partition;
pub mod quarantine;
pub mod rng;
pub mod rows;
pub mod rows_fields;
pub mod stream_driver;
pub mod tiles;
pub mod validate;
pub mod walk;
