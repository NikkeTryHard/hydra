//! stream_report: wall-disjoint gate over plain wall-id lists for `training/stream_scan.py`.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge, no new dependency — the gate is
//! pure set logic over caller-staged wall ids; wall *extraction* stays
//! feed-owned via `stream_scan::scan_file_games`, split *assignment* stays
//! Python via `assign_split`, and the merge/cache/pool orchestration stays in
//! `stream_scan.py`). FORBIDS: file IO, framing/decode/validate, wall-hash
//! computation, split assignment, dedup/merge/cache/pool bookkeeping, and the
//! `_ScanReport` record assembly (all stay Python); hash/digest/canonical work
//! (feed owners, never reimplemented here).
//!
//! Parity contract (mirrors `_scan_report`, `stream_scan.py:166-197`):
//! - Inputs are plain wall-id lists (the translator stages
//!   `sorted(train_walls)` / `sorted(val_walls)` positionally) plus the two
//!   split names (`config.data.train_split` / `config.data.val_split`).
//! - Overlap is the sorted intersection; the first element names the failure
//!   with its 16-char head (`chars().take(16)`, matching Python `[:16]` —
//!   char-based, mirrors `stream_decode.rs:155`). Wall hashes are hex so both
//!   agree; the char path keeps non-ASCII garbage total (same note as
//!   `stream_decode.rs:83-85`).
//! - `sort_unstable` byte order equals Python `sorted` code-point order for
//!   UTF-8 (same note as `oracle_guard.rs:137-148`); duplicates in the staged
//!   lists collapse via `dedup` after sorting, matching the oracle's set
//!   intersection.
//! - Empty intersection passes (`Ok(())`); non-empty raises `ValueError` with
//!   exactly `wall {head} in splits {train} and {val}` (the translator maps to
//!   `ContractError` with byte-identical text, same discipline as
//!   `stream_decode::check_wall_disjoint`).
//!
//! Shape per fn: attached staging (pyo3 `Vec<String>` / `String` extraction)
//! -> ONE `py.detach(|| ...)` over owned plain data with zero Python API
//! inside (per `stream_scan.rs:183-203`) -> attached wrap as `PyValueError`
//! (per `stream_decode.rs:224-225`). Translators call positionally; the bridge
//! raises `ValueError`, Python maps to `ContractError`.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed `sub`
//! in `wrap_pyfunction!(f, sub)` mirror `stream_decode::register`
//! (`crates/bridge/src/stream_decode.rs:335-346`) and `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-489`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `stream_scan::scan_file_games` (`crates/bridge/src/stream_scan.rs:183-203`);
//! `PyValueError::new_err` mapping mirrors `stream_decode.rs:224-225`;
//! `Python::initialize()` + `Python::attach` in tests mirror
//! `stream_decode.rs:551-553` (via `canon_rng.rs:688-689`).
//!
//! NONE-owner note: `rg 'in splits'` over `crates/feed/src` +
//! `crates/shard/src` + `crates/search/src` hits only an unrelated
//! `stream_driver.rs` doc/test string (driver quarantine classes, never the
//! scan-report gate) — no crate owns this leaf, so the text lives here.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.columnar`
//! submodule (scan-adjacent; `stream_scan` / `stream_decode` precedent); no new
//! entry point. MAIN wiring (shared files, MAIN-only): `pub mod stream_report;`
//! in `lib.rs` next to the `stream_*` mods plus
//! `crate::stream_report::register(&sub)?;` at the end of
//! `columnar::register` (`columnar.rs:403-410`, after the `quarantine_inbox` line).

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Detached core (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Wall-disjoint gate (`stream_scan.py:180-185`): the sorted intersection of
/// the two wall-id lists names the failure with the 16-char head plus both
/// splits (char-based, matching Python `[:16]`).
fn scan_report_overlap_core(
    train_walls: &[String],
    val_walls: &[String],
    train_split: &str,
    val_split: &str,
) -> Result<(), String> {
    let train_set: HashSet<&str> = train_walls.iter().map(String::as_str).collect();
    let mut overlap: Vec<&str> = val_walls
        .iter()
        .map(String::as_str)
        .filter(|wall| train_set.contains(*wall))
        .collect();
    overlap.sort_unstable();
    overlap.dedup();
    if let Some(first) = overlap.first().copied() {
        let head: String = first.chars().take(16).collect();
        return Err(format!(
            "wall {head} in splits {train_split} and {val_split}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfn (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Wall-disjointness gate over staged wall-id lists (mirrors
/// `stream_scan.py:180-185` without the `_ScanReport` assembly, which stays
/// Python). Translators call positionally; the bridge raises `ValueError`,
/// Python maps to `ContractError` with byte-identical text. Compute runs
/// detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (train_walls, val_walls, train_split, val_split))]
fn stream_report_check_wall_disjoint(
    py: Python<'_>,
    train_walls: Vec<String>,
    val_walls: Vec<String>,
    train_split: String,
    val_split: String,
) -> PyResult<()> {
    py.detach(|| scan_report_overlap_core(&train_walls, &val_walls, &train_split, &val_split))
        .map_err(PyValueError::new_err)
}

/// Attach the report gate to the caller-provided `columnar` submodule
/// (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `stream_decode.rs:335-346` / `rows_seal.rs:484-489`; no new submodule, no
/// new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(stream_report_check_wall_disjoint, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod stream_report_tests {
    use super::*;

    #[test]
    fn disjoint_passes_vacuous_and_disjoint() {
        // HEAD: empty intersection passes (`stream_scan.py:180-181` —
        // `len(overlap) > 0` is false, so the report is built).
        let empty: Vec<String> = Vec::new();
        assert!(scan_report_overlap_core(&empty, &empty, "train", "test").is_ok());
        let train = vec!["wall-a".to_string()];
        let val = vec!["wall-b".to_string()];
        assert!(scan_report_overlap_core(&train, &val, "train", "test").is_ok());
        // Repeats within one side only are not cross-split overlap.
        let train_dup = vec!["wall-a".to_string(), "wall-a".to_string()];
        assert!(scan_report_overlap_core(&train_dup, &val, "train", "test").is_ok());
    }

    #[test]
    fn overlap_names_sorted_first_head_and_splits() {
        // HEAD: `overlap = sorted(train & val)` then
        // `wall {overlap[0][:16]} in splits {train} and {val}`
        // (`stream_scan.py:180-185`); hand-derived: the lexicographically
        // smallest shared wall names the failure.
        let train = vec!["b-wall".to_string(), "a-wall".to_string()];
        let val = vec!["b-wall".to_string(), "a-wall".to_string()];
        assert_eq!(
            scan_report_overlap_core(&train, &val, "train", "test").expect_err("must reject"),
            "wall a-wall in splits train and test",
        );
    }

    #[test]
    fn overlap_head_truncates_to_16_chars() {
        // HEAD: `overlap[0][:16]` keeps the first 16 chars
        // (`stream_scan.py:183`); mirrors the `stream_decode` head test
        // (`stream_decode.rs:394-404`).
        let wall = "abcdef0123456789XYZ".to_string();
        let train = vec![wall.clone()];
        let val = vec![wall];
        assert_eq!(
            scan_report_overlap_core(&train, &val, "train", "test").expect_err("must reject"),
            "wall abcdef0123456789 in splits train and test",
        );
    }

    #[test]
    fn short_wall_kept_whole() {
        // HEAD: short ids slice to themselves (`"abc"[:16] == "abc"`).
        let train = vec!["abc".to_string()];
        let val = vec!["abc".to_string()];
        assert_eq!(
            scan_report_overlap_core(&train, &val, "train", "held_out").expect_err("must reject"),
            "wall abc in splits train and held_out",
        );
    }

    #[test]
    fn duplicate_staged_inputs_still_gate() {
        // The translator stages `sorted(set)` (deduped), but the core stays
        // total when the staged lists carry duplicates.
        let train = vec!["w".to_string(), "w".to_string()];
        let val = vec!["w".to_string(), "w".to_string(), "other".to_string()];
        assert_eq!(
            scan_report_overlap_core(&train, &val, "train", "test").expect_err("must reject"),
            "wall w in splits train and test",
        );
    }

    #[test]
    fn report_pyfn_names_conflict() {
        Python::initialize();
        Python::attach(|py| {
            assert!(
                stream_report_check_wall_disjoint(
                    py,
                    vec!["a".to_string()],
                    vec!["b".to_string()],
                    "train".to_string(),
                    "test".to_string(),
                )
                .is_ok(),
            );
            let wall = "w".repeat(20);
            let err = stream_report_check_wall_disjoint(
                py,
                vec![wall.clone()],
                vec![wall],
                "train".to_string(),
                "held_out".to_string(),
            )
            .expect_err("must reject");
            assert_eq!(
                err.to_string(),
                "ValueError: wall wwwwwwwwwwwwwwww in splits train and held_out",
            );
        });
    }
}
