//! belief_leaves: tiny-belief frozen tables + scalar pure leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-feed` ONLY (the `canon` + `digest` owners for the
//! world-identity hash; pure `std` math for the rest — same edge as
//! `contracts.rs`, no new dependency). FORBIDS: JCS/sha reimplementation
//! (hashing routes through `hydra_feed::canon` + `hydra_feed::digest`),
//! wall-clock/RNG (the `RandomStream` draws stay caller-side), live-object
//! orchestration (`FullWorld`/`TinyCorpus` builders, corpus sorting, and the
//! confirmation runner stay Python), and `ContractError` shaping (the bridge
//! raises `ValueError`; thin Python translators map to `ContractError` with
//! byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/*.py` (worktree tags at
//! port time: `world.py#00E0`, `corpus.py#080B`, `confirmation.py#2534`; HEAD
//! `src/hydra2/belief/` predates the `FullWorld` validators, so the worktree
//! is the parity source for the world rows):
//! - `world_id_from_doc` <- `compute_world_id` (`world.py:105-107`) hash half:
//!   the caller prepares `canonical_bytes(identity_doc)` Python-side (canon
//!   authority stays Python); Rust re-canonicalizes through the feed `canon`
//!   owner and hashes through the feed `digest` owner, returning the
//!   `sha256:`-prefixed text (the `packet_id_from_doc` precedent at
//!   `contracts.rs:912-927`, prefix kept here per the oracle).
//! - `corpus_log_prob_for` <- `TinyCorpus.log_prob` (`corpus.py:55-59`):
//!   scalars cross (`world_ids` + `probabilities` + target, never live
//!   `FullWorld`s — the `PacketSuccessor` precedent); `zip` truncation
//!   mirrors `strict=False`, `p > 0` gates `ln` exactly like the oracle
//!   (NaN and zero both yield `-inf`, never reaching the log).
//! - `confirmation_value_mix` <- runner value fold (`confirmation.py:94-96`):
//!   `(val_raw % 1000) / 1000.0` mixed with the caller's single
//!   `random_float()` draw; the draw itself stays caller-side (live RNG).
//! - consts `TINY_CORPUS_*` <- builder tables (`corpus.py:67-68,86-95`):
//!   the 6-row hand-option table, the `(8, 9, 10, 11)` wall, and the
//!   `(0, 1)` / `4` builder defaults.
//! - consts `WORLD_*` <- frozen observation literals
//!   (`world.py:154,194-202,217`): tiny `game_id`, `discard_response` phase,
//!   round wind `27`, seat winds, starting scores, and the 3-wide tiny legal
//!   mask. `RULES_ID` is NOT restated (`contracts` already owns `RULES_ID`);
//!   the `b/c/d/e` dummy-hash fallbacks stay Python (one-line builder
//!   fallbacks, not a table).
//!
//! LOGIC staying Python: `FullWorld`/`TinyCorpus`/`ConfirmationCase`
//! `__post_init__` bodies (record validating roots per the `contracts.rs:21-26`
//! rank-3 note), `make_full_world` assembly, `build_tiny_corpus` (live-world
//! construction + `world_id` sort + uniform weights), `world_actor_observation`
//! (live `ActorObservation` construction), the confirmation runner loop (live
//! `RandomStream` cursor), and every `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `tiny_corpus|TinyCorpus|confirmation_value|
//! world_id_from_doc|corpus_log_prob` over `crates/feed/src` +
//! `crates/search/src` hits only an unrelated `FullWorld` doc comment in
//! `search/src/ismcts_driver.rs:28-31` (scalar-continuation note, no table,
//! no hash); no crate owns these leaves.
//!
//! Shape per fn: attached staging (none needed — all args are
//! `String`/`Vec`/`u32`/`f64` scalars) -> ONE `py.detach(|| ...)` over owned
//! plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `rc_require.rs:827-844`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`, `ctr_block` at `contracts.rs:447-456`). Consts via
//! `sub.add` (per `contracts.rs:1153-1156`); nested tables via
//! `Vec<Bound<'_, PyTuple>>` + `PyTuple::new` (per `contracts.rs:1281-1287`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_require.rs:846-862`); no new entry
//! point. MAIN wiring: `pub mod belief_leaves;` in `lib.rs` plus
//! `crate::belief_leaves::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1295`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

/// Tiny-domain hand-option table (`corpus.py:86-93`): six rows of four
/// 2-tile seats; the root seat is always `(0, 1)` before observation
/// rebinding. Frozen generation order (never reordered, never extended).
const TINY_CORPUS_OPTIONS: [[[u8; 2]; 4]; 6] = [
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[0, 1], [2, 4], [3, 5], [6, 7]],
    [[0, 1], [2, 5], [3, 4], [6, 7]],
    [[0, 1], [2, 6], [3, 4], [5, 7]],
    [[0, 1], [2, 7], [3, 4], [5, 6]],
    [[0, 1], [3, 6], [2, 4], [5, 7]],
];

/// Tiny-domain live wall (`corpus.py:95`).
const TINY_CORPUS_WALL: [u8; 4] = [8, 9, 10, 11];

/// Builder default root hand (`corpus.py:67`).
const TINY_CORPUS_DEFAULT_ROOT_HAND: [u8; 2] = [0, 1];

/// Builder default corpus size (`corpus.py:68`).
const TINY_CORPUS_DEFAULT_SIZE: usize = 4;

/// Tiny observation default game id (`world.py:154`).
const WORLD_DEFAULT_GAME_ID: &str = "game_tiny_001";

/// Tiny observation phase (`world.py:202`).
const WORLD_DEFAULT_PHASE: &str = "discard_response";

/// Tiny observation round wind (`world.py:194`).
const WORLD_DEFAULT_ROUND_WIND: u8 = 27;

/// Tiny observation seat winds (`world.py:196`).
const WORLD_SEAT_WINDS: [u8; 4] = [27, 28, 29, 30];

/// Tiny observation starting scores (`world.py:200`).
const WORLD_DEFAULT_SCORES: [u32; 4] = [25000, 25000, 25000, 25000];

/// Tiny-domain 3-wide legal mask (`world.py:217`).
const WORLD_TINY_LEGAL_MASK: [bool; 3] = [true, false, true];

/// Pure world-identity hash core: re-canonicalize through the feed `canon`
/// owner, hash through the feed `digest` owner, keep the `sha256:` prefix.
/// No Python API inside (detach-safe); failures are owner-text `String`s.
fn world_id_for_doc(identity_doc: &[u8]) -> Result<String, String> {
    let value =
        hydra_feed::canon::parse_canonical_bytes(identity_doc, "belief_leaves:world_id_from_doc")
            .map_err(|e| e.to_string())?;
    let bytes = hydra_feed::canon::canonical_bytes_value(&value, "belief_leaves:world_id_from_doc")
        .map_err(|e| e.to_string())?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Pure corpus-lookup core: first `world_id` match wins over the zipped
/// prefix (mirrors `zip(..., strict=False)` truncation); `p > 0` gates the
/// log exactly like the oracle, so zero/negative/NaN all yield `-inf`.
fn log_prob_for(world_ids: &[String], probabilities: &[f64], world_id: &str) -> f64 {
    for (id, prob) in world_ids.iter().zip(probabilities.iter()) {
        if id == world_id {
            return if *prob > 0.0 {
                prob.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
    }
    f64::NEG_INFINITY
}

/// Pure confirmation-value core: `(val_raw % 1000) / 1000.0` mixed with the
/// caller's draw (mirrors `confirmation.py:94-96` for the reachable domain
/// `val_raw < 2^32`, where the oracle's hex-slice int always lands).
fn confirm_mix(val_raw: u32, random_draw: f64) -> f64 {
    (f64::from(val_raw % 1000) / 1000.0 + random_draw) / 2.0
}

/// World-identity digest over caller-prepared canonical identity bytes
/// (mirrors `compute_world_id` without the live `FullWorld`): the caller
/// prepares `canonical_bytes(identity_doc)` (canon authority stays Python);
/// Rust re-canonicalizes through the feed `canon` owner and hashes through
/// the feed `digest` owner, returning `sha256:`-prefixed text (the
/// `packet_id_from_doc` precedent at `contracts.rs:912-927`, prefix kept).
/// Compute runs detached with zero Python API inside; parse/canon failures
/// are `PyValueError`, never a default. Counter-free deterministic, never
/// wall-clock.
#[pyfunction]
#[pyo3(signature = (identity_doc,))]
fn world_id_from_doc(py: Python<'_>, identity_doc: Vec<u8>) -> PyResult<String> {
    py.detach(|| world_id_for_doc(&identity_doc))
        .map_err(|e| PyValueError::new_err(format!("belief world_id_from_doc rejected: {e}")))
}

/// Corpus log-probability over parallel `(world_ids, probabilities)` slices
/// plus the target `world_id` (mirrors `TinyCorpus.log_prob` without live
/// `FullWorld` objects — callers pass scalars, the `PacketSuccessor`
/// precedent). `zip` truncation mirrors `strict=False` (no length check);
/// a miss yields `-inf`, never an error. Compute runs detached; pure
/// function of the inputs, never wall-clock.
#[pyfunction]
#[pyo3(signature = (world_ids, probabilities, world_id))]
fn corpus_log_prob_for(
    py: Python<'_>,
    world_ids: Vec<String>,
    probabilities: Vec<f64>,
    world_id: String,
) -> f64 {
    py.detach(|| log_prob_for(&world_ids, &probabilities, &world_id))
}

/// Confirmation value mix over the oracle hex int plus the caller's single
/// RNG draw (mirrors `confirmation.py:94-96` without the live
/// `RandomStream` — the draw stays caller-side). Compute runs detached;
/// pure function of its inputs, never wall-clock.
#[pyfunction]
#[pyo3(signature = (val_raw, random_draw))]
fn confirmation_value_mix(py: Python<'_>, val_raw: u32, random_draw: f64) -> f64 {
    py.detach(|| confirm_mix(val_raw, random_draw))
}

/// Register the belief leaves on the shared `contracts` submodule (mirrors
/// `rc_require.rs:846-862`): compute detached, wrap attached; single cdylib,
/// no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(world_id_from_doc, sub)?)?;
    sub.add_function(wrap_pyfunction!(corpus_log_prob_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(confirmation_value_mix, sub)?)?;
    sub.add("TINY_CORPUS_WALL", PyTuple::new(py, TINY_CORPUS_WALL)?)?;
    {
        let options: Vec<Bound<'_, PyTuple>> = TINY_CORPUS_OPTIONS
            .iter()
            .map(|seats| {
                let seat_tuples: Vec<Bound<'_, PyTuple>> = seats
                    .iter()
                    .map(|pair| PyTuple::new(py, [pair[0], pair[1]]))
                    .collect::<PyResult<_>>()?;
                PyTuple::new(py, seat_tuples)
            })
            .collect::<PyResult<_>>()?;
        sub.add("TINY_CORPUS_OPTIONS", PyTuple::new(py, options)?)?;
    }
    sub.add(
        "TINY_CORPUS_DEFAULT_ROOT_HAND",
        PyTuple::new(py, TINY_CORPUS_DEFAULT_ROOT_HAND)?,
    )?;
    sub.add("TINY_CORPUS_DEFAULT_SIZE", TINY_CORPUS_DEFAULT_SIZE)?;
    sub.add("WORLD_DEFAULT_GAME_ID", WORLD_DEFAULT_GAME_ID)?;
    sub.add("WORLD_DEFAULT_PHASE", WORLD_DEFAULT_PHASE)?;
    sub.add("WORLD_DEFAULT_ROUND_WIND", WORLD_DEFAULT_ROUND_WIND)?;
    sub.add("WORLD_SEAT_WINDS", PyTuple::new(py, WORLD_SEAT_WINDS)?)?;
    sub.add(
        "WORLD_DEFAULT_SCORES",
        PyTuple::new(py, WORLD_DEFAULT_SCORES)?,
    )?;
    sub.add(
        "WORLD_TINY_LEGAL_MASK",
        PyTuple::new(py, WORLD_TINY_LEGAL_MASK)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirm_mix_matches_oracle_fold() {
        // Hand-derived from the oracle fold (`confirmation.py:94-96`):
        // value = (val_raw % 1000) / 1000.0, then (value + draw) / 2.0.
        assert_eq!(confirm_mix(0, 0.0), 0.0);
        assert_eq!(confirm_mix(999, 1.0), 0.9995);
        assert_eq!(confirm_mix(1000, 0.5), 0.25);
        assert_eq!(confirm_mix(789, 0.0), 0.3945);
        assert_eq!(confirm_mix(1_000_789, 0.0), confirm_mix(789, 0.0));
    }

    #[test]
    fn log_prob_matches_oracle_scan() {
        // Hand-derived from `TinyCorpus.log_prob` (`corpus.py:55-59`):
        // first id match wins, log iff p > 0 else -inf, miss is -inf.
        let ids = vec!["a".to_string(), "b".to_string()];
        let probs = vec![0.25, 0.75];
        assert_eq!(log_prob_for(&ids, &probs, "a"), -1.3862943611198906);
        assert_eq!(log_prob_for(&ids, &probs, "b"), 0.75_f64.ln());
        assert_eq!(log_prob_for(&ids, &probs, "zzz"), f64::NEG_INFINITY);
        assert_eq!(log_prob_for(&ids, &[0.0, 0.75], "a"), f64::NEG_INFINITY);
        // zip truncation mirrors strict=False: ragged tails never match.
        assert_eq!(
            log_prob_for(&["only".to_string()], &[], "only"),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn world_id_core_is_prefixed_deterministic() {
        // Empty-object doc is trivially canonical; the core must return a
        // prefixed digest deterministically and separate distinct docs.
        let first = world_id_for_doc(b"{}").unwrap();
        let again = world_id_for_doc(b"{}").unwrap();
        assert_eq!(first, again);
        assert!(first.starts_with("sha256:"), "missing prefix: {first}");
        assert_eq!(first.len(), 71, "not sha256:<64 hex>: {first}");
        let other = world_id_for_doc(br#"{"a":1}"#).unwrap();
        assert_ne!(first, other);
        assert!(world_id_for_doc(b"not json").is_err());
    }

    #[test]
    fn frozen_tables_match_oracle_literals() {
        // Byte-level pins of the oracle tables (`corpus.py:86-95`,
        // `world.py:154,194-202,217`); any drift fails here first.
        assert_eq!(TINY_CORPUS_WALL, [8, 9, 10, 11]);
        assert_eq!(TINY_CORPUS_OPTIONS.len(), 6);
        assert_eq!(TINY_CORPUS_OPTIONS[0], [[0, 1], [2, 3], [4, 5], [6, 7]]);
        assert_eq!(TINY_CORPUS_OPTIONS[5], [[0, 1], [3, 6], [2, 4], [5, 7]]);
        assert_eq!(TINY_CORPUS_DEFAULT_ROOT_HAND, [0, 1]);
        assert_eq!(TINY_CORPUS_DEFAULT_SIZE, 4);
        assert_eq!(WORLD_DEFAULT_GAME_ID, "game_tiny_001");
        assert_eq!(WORLD_DEFAULT_PHASE, "discard_response");
        assert_eq!(WORLD_DEFAULT_ROUND_WIND, 27);
        assert_eq!(WORLD_SEAT_WINDS, [27, 28, 29, 30]);
        assert_eq!(WORLD_DEFAULT_SCORES, [25000, 25000, 25000, 25000]);
        assert_eq!(WORLD_TINY_LEGAL_MASK, [true, false, true]);
    }
}
