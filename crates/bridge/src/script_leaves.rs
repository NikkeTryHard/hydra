//! script_leaves: bench + scripts pure cores on the shared `contracts` submodule.
//!
//! DAG: pyo3 + feed `digest` owner ONLY (no new dependency; the bare-hex
//! helper folds through `hydra_feed::digest::sha256_hex`, the same entry
//! `canon_rng::sha256_hex` uses at `canon_rng.rs:184-189`). FORBIDS:
//! torch/CUDA/triton/JAX/riichienv/arrow/parquet/zstd/jsonschema/requests
//! (framework/IO owners stay Python), file IO (every `_sha256_file` /
//! `_zst_line_count` / manifest walk stays Python as OS authority), RNG
//! (`random.Random(F2)` / `torch.Generator` synth builders stay Python),
//! clocks (`time.perf_counter` walls stay Python), mutable state (ledgers,
//! registries, `PinnedRing`, stream drivers stay Python), and float SUMS
//! (every `total_us += dur` / `statistics.median` accumulation stays Python
//! by contract — this module ports only single-op index/divide/compare math
//! plus closed-form integer products and vocab gates).
//!
//! TABLE ported here (oracle file:line frozen verbatim):
//! - consts `SCRIPT_F1_TOTAL_LINES` (50) <- `_F1_TOTAL_LINES`
//!   (`bench/feed_rate.py:118`); `SCRIPT_F1_DECISIONS` (5) <- `_F1_DECISIONS`
//!   (`:119`); `SCRIPT_F2_TARGET_LINES` (1500) <- `_F2_TARGET_LINES`
//!   (`:114`); `SCRIPT_F8_FILES` (30) <- `_F8_FILES` (`:115`);
//!   `SCRIPT_F8_GAMES_PER_FILE` (22) <- `_F8_GAMES_PER_FILE` (`:116`);
//!   `SCRIPT_F8_MID` (3) <- `_F8_MID` (`:117`); `SCRIPT_F10_FILES` (8) <-
//!   `_F10_FILES` (`:120`); `SCRIPT_F10_GAMES_PER_FILE` (12) <- same line;
//!   `SCRIPT_F10_BUILDER` ("f10-v1") <- `_F10_BUILDER` (`:122`);
//!   `SCRIPT_F11_FILES` (64) <- `_F11_FILES` (`:123`);
//!   `SCRIPT_F11_GAMES_PER_FILE` (12) <- same line;
//!   `SCRIPT_F11_BUILDER` ("f11-v1") <- `_F11_BUILDER` (`:125`);
//!   `SCRIPT_SEED` (7) <- `_SEED` (`:112`); `SCRIPT_F2_SEED` (20260909) <-
//!   `_F2_SEED` (`:113`); derived closed forms `SCRIPT_F8_EXPECTED_GAMES`
//!   (660 = 30*22), `SCRIPT_F10_EXPECTED_GAMES` (96 = 8*12),
//!   `SCRIPT_F11_EXPECTED_GAMES` (768 = 64*12) (the `want` products at
//!   `bench/feed_rate.py:930,944-945`).
//! - consts `SCRIPT_LOC_LIMIT_PYTHON` (600) / `SCRIPT_LOC_LIMIT_TESTS`
//!   (1000) / `SCRIPT_LOC_LIMIT_LEAN` (800) / `SCRIPT_LOC_LIMIT_CRATES`
//!   (2000) <- `LIMITS` (`scripts/check_loc.py:15`, keys `python/tests/
//!   lean/crates`, ceilings verbatim).
//! - const `SCRIPT_CLOSED_ALLOW` <- `CLOSED_ALLOW`
//!   (`scripts/freeze_row_hashes.py:91`: `wall_game_id/copy_fold/
//!   refit_number/wall_less_marker`, order verbatim) plus the `ALLOW_DROP`
//!   map (`:92-97`: `wall_game_id -> (source_object_id, game_id, round_id)`;
//!   `copy_fold -> (observation_hash, concealed, drawn, dora)`;
//!   `refit_number -> (wall, live_wall_tiles_remaining)`;
//!   `wall_less_marker -> (derivation_hash, derivation, wall_digest)`).
//! - `script_percentile_index` <- `_percentile` index step
//!   (`bench/feed_rate.py:184-189`: `sorted`, `idx=min(int(q*n), n-1)`;
//!   the sort + float accumulation stay Python, only the index ports).
//! - `script_rate_per_sec` <- the `n / elapsed if elapsed > 0 else 0.0`
//!   legs (`bench/feed_rate.py:775-777` expand, `:858-862` stream,
//!   `:1025-1027` F2 single-file; one fn for decisions/events/batches).
//! - `script_loc_exceeds` <- `n > lim` (`scripts/check_loc.py:61`).
//! - `script_grid_total` <- the `n_files * per_file` closed forms
//!   (`bench/feed_rate.py:930` F8 660, `:944` F10 96, F11 768 same shape).
//! - `script_allow_is_valid` / `script_allow_drop_keys` <- the closed
//!   allowlist gate (`scripts/freeze_row_hashes.py:217-228`: `--allow`
//!   accepts ONLY `CLOSED_ALLOW`, open-ended waivers fail closed) and the
//!   `ALLOW_DROP` projection (`:198-205` `_drop_paths` key lookup; the dict
//!   copy itself stays Python).
//! - `script_sha256_bare_hex` <- bare-hex hashing of PRE-canonicalized bytes
//!   (`hashlib.sha256(canonical_bytes(...)).hexdigest()` at
//!   `bench/feed_rate.py:1061-1065` `entries_digest`; the JCS emit stays
//!   Python via `canonical_bytes`, the feed `digest` owner hashes — never a
//!   second SHA implementation; file chunking `_sha256_file` stays Python).
//!
//! LOGIC staying Python (entrypoints never port; file walks, process
//! orchestration, clocks, RNG, torch, float sums stay Python):
//! - `bench/feed_rate.py`: `main`/`cmd_*` argparse dispatch, `_configure_
//!   runtime` thread pinning, `_fingerprint` env/GPU probes, `_compress_zst`
//!   / `_zst_line_count` subprocess zstd CLI, `_frame_jsonl_games` /
//!   `_expand_pass` / `_stream_pass` / `_ring_probe` timed legs
//!   (`time.perf_counter`, `expand_game`, `PinnedRing`), `_summarize`
//!   (`statistics.median` float sum stays Python), every `build_corpus`
//!   byte-write.
//! - `bench/kbench.py`: ENTIRE file stays Python — `kernel_sum_from_chrome_
//!   trace` is file IO (`open`/`gzip.open` + `json.load`) plus a FLOAT SUM
//!   (`total_us += dur`, contract-forbidden) plus torch/CUDA profiler
//!   orchestration; `_build_synth_batch` is torch RNG.
//! - `scripts/check_loc.py`: `main` root walk (`pathlib.Path.rglob`,
//!   `read_text`, `.lake`/`target` skips) stays Python; only `n > lim` ports.
//! - `scripts/freeze_row_hashes.py`: `main` freeze/check dispatch, `oracle_
//!   pins` / `_sha256_file` file IO, `freeze_rows` / `check_fixture` JSONL
//!   walks, `compact_projection` / `full_projection` MJAI-string normalization
//!   (needs the live `tiles.mjai_string_of` bridge + `json.loads`), verdict
//!   joins (`sum(1 for ...)` int-sum + `sorted` stays Python — float sums
//!   are not the only sums; verdict assembly stays Python by ownership).
//! - `scripts/seal_replay.py`: ENTIRE file stays Python — `_synthetic_corpus`
//!   (zstandard + file writes), `_run_hook` / `_run_rust_gate` (subprocess
//!   cargo-nextest orchestration), `main` out-dir/collision guards.
//!
//! NONE-owner note: `rg` for `script_percentile_index|script_rate_per_sec|
//! script_loc_exceeds|script_grid_total|script_allow_is_valid|
//! script_allow_drop_keys|script_sha256_bare_hex|SCRIPT_F1_TOTAL_LINES|
//! SCRIPT_LOC_LIMIT_PYTHON|SCRIPT_CLOSED_ALLOW` over `crates/bridge/src` is
//! empty before this file (the `eval_stats_percentile_indexes` at
//! `eval_stats.rs:237-245` is the bootstrap-CI `floor/ceil` pair — never the
//! bench `min(int(q*n), n-1)` pick; `canon_rng::sha256_hex` returns
//! `sha256:`-PREFIXED text while this module returns BARE hex — different
//! presentations over the same feed owner, never a second hash).
//!
//! Shape per fn: attached staging (`u64`/`usize`/`&str`/`Vec<u8>`
//! extraction) -> compute (trivial leaves attached end-to-end per
//! `contracts.rs:10-13`; ONLY `script_sha256_bare_hex` runs ONE
//! `py.detach(|| ...)` with zero Python API inside per
//! `canon_rng.rs:184-189`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Translators call positionally; the bridge raises
//! `ValueError`, Python falls back to the byte-identical oracle only when the
//! attribute is missing (stale `.so`), never on `ValueError`.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `loop_state.rs:132-144`); no new entry
//! point. MAIN wiring: `pub mod script_leaves;` in `lib.rs` next to the
//! other leaves plus `crate::script_leaves::register(&sub)?;` at the end of
//! `contracts::register` (`contracts.rs:1321`, after the `distill_leaves` line).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

/// Smoke-test line budget (`bench/feed_rate.py:118`).
const SCRIPT_F1_TOTAL_LINES: u64 = 50;
/// Smoke-test decision count (`bench/feed_rate.py:119`).
const SCRIPT_F1_DECISIONS: u64 = 5;
/// F2 primary line budget (`bench/feed_rate.py:114`).
const SCRIPT_F2_TARGET_LINES: u64 = 1500;
/// F8 stream-leg file count (`bench/feed_rate.py:115`).
const SCRIPT_F8_FILES: u64 = 30;
/// F8 games per file (`bench/feed_rate.py:116`).
const SCRIPT_F8_GAMES_PER_FILE: u64 = 22;
/// F8 mid-game turn count (`bench/feed_rate.py:117`).
const SCRIPT_F8_MID: u64 = 3;
/// F10 decision-primary file count (`bench/feed_rate.py:120`).
const SCRIPT_F10_FILES: u64 = 8;
/// F10/F11 games per file (`bench/feed_rate.py:121,124`).
const SCRIPT_F10_GAMES_PER_FILE: u64 = 12;
const SCRIPT_F11_GAMES_PER_FILE: u64 = 12;
/// F10 builder version (`bench/feed_rate.py:122`).
const SCRIPT_F10_BUILDER: &str = "f10-v1";
/// F11 saturated file count (`bench/feed_rate.py:123`).
const SCRIPT_F11_FILES: u64 = 64;
/// F11 builder version (`bench/feed_rate.py:125`).
const SCRIPT_F11_BUILDER: &str = "f11-v1";
/// Stream seed (`bench/feed_rate.py:112`).
const SCRIPT_SEED: u64 = 7;
/// F2 synth seed (`bench/feed_rate.py:113`).
const SCRIPT_F2_SEED: u64 = 20260909;

/// F8 closed form: 30 files x 22 games = 660 emissions (`bench/feed_rate.py:930`).
const SCRIPT_F8_EXPECTED_GAMES: u64 = 660;
/// F10 closed form: 8 files x 12 games = 96 games (`bench/feed_rate.py:944`).
const SCRIPT_F10_EXPECTED_GAMES: u64 = 96;
/// F11 closed form: 64 files x 12 games = 768 games (F10 shape, `:586-590`).
const SCRIPT_F11_EXPECTED_GAMES: u64 = 768;

/// LOC ceiling for `python/` (`scripts/check_loc.py:15`).
const SCRIPT_LOC_LIMIT_PYTHON: u64 = 600;
/// LOC ceiling for `tests/` (`scripts/check_loc.py:15`).
const SCRIPT_LOC_LIMIT_TESTS: u64 = 1000;
/// LOC ceiling for `lean/` (`scripts/check_loc.py:15`).
const SCRIPT_LOC_LIMIT_LEAN: u64 = 800;
/// LOC ceiling for `crates/` (`scripts/check_loc.py:15`).
const SCRIPT_LOC_LIMIT_CRATES: u64 = 2000;

/// Closed `--allow` vocabulary (`scripts/freeze_row_hashes.py:91`, order verbatim).
const SCRIPT_CLOSED_ALLOW: [&str; 4] = [
    "wall_game_id",
    "copy_fold",
    "refit_number",
    "wall_less_marker",
];

/// Projection keys dropped for one allow name (`scripts/freeze_row_hashes.py:92-97`).
fn allow_drop_keys(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "wall_game_id" => Some(&["source_object_id", "game_id", "round_id"]),
        "copy_fold" => Some(&["observation_hash", "concealed", "drawn", "dora"]),
        "refit_number" => Some(&["wall", "live_wall_tiles_remaining"]),
        "wall_less_marker" => Some(&["derivation_hash", "derivation", "wall_digest"]),
        _ => None,
    }
}

/// Pure index core: `min(int(q*n), n-1)` with `None` for the empty series.
///
/// Mirrors `bench/feed_rate.py:184-189` bit-for-bit on the covered domain
/// (`q` in `[0, 1]`, called only with `0.50`/`0.99` via `_summarize`): the
/// Python `sorted()` stays caller-side, so float ORDER never crosses the
/// boundary — only this `usize` pick ports. `int(q*n)` truncates toward zero
/// for non-negative products, exactly what `as usize` does; out-of-range or
/// non-finite `q` fails closed (`PyValueError`, never Python's negative-index
/// wrap). Detach-safe (plain scalars); the caller maps the error verbatim.
fn percentile_index_core(n: usize, q: f64) -> Result<Option<usize>, String> {
    if n == 0 {
        return Ok(None);
    }
    if !q.is_finite() || q < 0.0 || q > 1.0 {
        return Err("script percentile q must lie in [0, 1]".to_string());
    }
    // proof: `n` is a small series len (< 2^53), exact; `q*n` in [0,n], `as` saturates like the oracle.
    #[allow(clippy::cast_precision_loss)]
    let n_f: f64 = n as f64;
    // proof: `q*n` in [0,n] (`q` in [0,1], checked above), fits `usize`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx: usize = (q * n_f) as usize;
    Ok(Some(idx.min(n - 1)))
}

/// Bench percentile index (`bench/feed_rate.py:189`).
///
/// Attached trivial leaf (sub-microsecond pick, per `contracts.rs:10-13`);
/// `None` for the empty series, `PyValueError` for `q` outside `[0, 1]`.
#[pyfunction]
#[pyo3(signature = (n, q))]
fn script_percentile_index(n: usize, q: f64) -> PyResult<Option<usize>> {
    percentile_index_core(n, q).map_err(PyValueError::new_err)
}

/// Pure rate core: `n / elapsed if elapsed > 0 else 0.0`.
///
/// Mirrors `bench/feed_rate.py:775-777,858-862,1025-1027` exactly, including
/// the `NaN -> 0.0` edge (`NaN > 0.0` is false on both sides) and the
/// `+inf -> n/inf == 0.0` edge. Single IEEE754 division — never a float SUM.
/// Total function (no error); attached trivial leaf.
fn rate_per_sec_core(n: u64, elapsed_s: f64) -> f64 {
    // proof: event counts are u64 (< 2^53), exact; rate tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let n_f: f64 = n as f64;
    if elapsed_s > 0.0 {
        n_f / elapsed_s
    } else {
        0.0
    }
}

/// Feed-rate decisions/events/batches per second (one fn for all three legs).
#[pyfunction]
#[pyo3(signature = (n, elapsed_s))]
fn script_rate_per_sec(n: u64, elapsed_s: f64) -> f64 {
    rate_per_sec_core(n, elapsed_s)
}

/// Pure LOC-gate core: `n > lim` (`scripts/check_loc.py:61`).
///
/// Attached trivial leaf; limits are the frozen `SCRIPT_LOC_LIMIT_*` consts.
fn loc_exceeds_core(n_lines: u64, limit: u64) -> bool {
    n_lines > limit
}

/// LOC-ceiling gate.
#[pyfunction]
#[pyo3(signature = (n_lines, limit))]
fn script_loc_exceeds(n_lines: u64, limit: u64) -> bool {
    loc_exceeds_core(n_lines, limit)
}

/// Pure grid core: `n_files * per_file` closed form (`bench/feed_rate.py:930`).
///
/// Python big-ints never overflow; `u64` overflow fails closed (never wraps).
/// Covered inputs (30x22, 8x12, 64x12) never approach the boundary.
fn grid_total_core(n_files: u64, per_file: u64) -> Result<u64, String> {
    n_files
        .checked_mul(per_file)
        .ok_or_else(|| "script grid_total overflow".to_string())
}

/// Corpus grid total (F8/F10/F11 `want` products).
#[pyfunction]
#[pyo3(signature = (n_files, per_file))]
fn script_grid_total(n_files: u64, per_file: u64) -> PyResult<u64> {
    grid_total_core(n_files, per_file).map_err(PyValueError::new_err)
}

/// Closed-allowlist membership (`scripts/freeze_row_hashes.py:220`).
#[pyfunction]
#[pyo3(signature = (name,))]
fn script_allow_is_valid(name: &str) -> bool {
    SCRIPT_CLOSED_ALLOW.contains(&name)
}

/// `ALLOW_DROP` projection for one allow name (`:92-97`); `None` when absent.
///
/// Returns owned `String`s so the list crosses intact; the `_drop_paths`
/// dict copy itself stays Python.
#[pyfunction]
#[pyo3(signature = (name,))]
fn script_allow_drop_keys(name: &str) -> Option<Vec<String>> {
    allow_drop_keys(name).map(|keys| keys.iter().map(|k| k.to_string()).collect())
}

/// Bare-hex digest of PRE-canonicalized bytes.
///
/// The JCS emit stays Python (`canonical_bytes`); this fn hashes the staged
/// bytes through the feed owner (`hydra_feed::digest::sha256_hex`, the same
/// entry `canon_rng::sha256_hex` folds) and strips the `sha256:` prefix the
/// scripts compare without (`hashlib...hexdigest()` at
/// `bench/feed_rate.py:1061-1065`). File chunking stays Python. Compute runs
/// under ONE `py.detach` with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (data,))]
fn script_sha256_bare_hex(py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
    let out = py.detach(|| hydra_feed::digest::sha256_hex(&data));
    Ok(out.strip_prefix("sha256:").unwrap_or(&out).to_string())
}

/// Register the script leaves on the shared `contracts` submodule (mirrors
/// `loop_state.rs:132-144`): `let py = sub.py();` (per `contracts.rs:1115`),
/// fns via `wrap_pyfunction!(f, sub)` (per `belief_leaves.rs:194`), consts
/// via `sub.add` (per `contracts.rs:1153-1156`); single cdylib, no new entry.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(script_percentile_index, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_rate_per_sec, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_loc_exceeds, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_grid_total, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_allow_is_valid, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_allow_drop_keys, sub)?)?;
    sub.add_function(wrap_pyfunction!(script_sha256_bare_hex, sub)?)?;
    sub.add("SCRIPT_F1_TOTAL_LINES", SCRIPT_F1_TOTAL_LINES)?;
    sub.add("SCRIPT_F1_DECISIONS", SCRIPT_F1_DECISIONS)?;
    sub.add("SCRIPT_F2_TARGET_LINES", SCRIPT_F2_TARGET_LINES)?;
    sub.add("SCRIPT_F8_FILES", SCRIPT_F8_FILES)?;
    sub.add("SCRIPT_F8_GAMES_PER_FILE", SCRIPT_F8_GAMES_PER_FILE)?;
    sub.add("SCRIPT_F8_MID", SCRIPT_F8_MID)?;
    sub.add("SCRIPT_F10_FILES", SCRIPT_F10_FILES)?;
    sub.add("SCRIPT_F10_GAMES_PER_FILE", SCRIPT_F10_GAMES_PER_FILE)?;
    sub.add("SCRIPT_F10_BUILDER", SCRIPT_F10_BUILDER)?;
    sub.add("SCRIPT_F11_FILES", SCRIPT_F11_FILES)?;
    sub.add("SCRIPT_F11_GAMES_PER_FILE", SCRIPT_F11_GAMES_PER_FILE)?;
    sub.add("SCRIPT_F11_BUILDER", SCRIPT_F11_BUILDER)?;
    sub.add("SCRIPT_SEED", SCRIPT_SEED)?;
    sub.add("SCRIPT_F2_SEED", SCRIPT_F2_SEED)?;
    sub.add("SCRIPT_F8_EXPECTED_GAMES", SCRIPT_F8_EXPECTED_GAMES)?;
    sub.add("SCRIPT_F10_EXPECTED_GAMES", SCRIPT_F10_EXPECTED_GAMES)?;
    sub.add("SCRIPT_F11_EXPECTED_GAMES", SCRIPT_F11_EXPECTED_GAMES)?;
    sub.add("SCRIPT_LOC_LIMIT_PYTHON", SCRIPT_LOC_LIMIT_PYTHON)?;
    sub.add("SCRIPT_LOC_LIMIT_TESTS", SCRIPT_LOC_LIMIT_TESTS)?;
    sub.add("SCRIPT_LOC_LIMIT_LEAN", SCRIPT_LOC_LIMIT_LEAN)?;
    sub.add("SCRIPT_LOC_LIMIT_CRATES", SCRIPT_LOC_LIMIT_CRATES)?;
    sub.add(
        "SCRIPT_CLOSED_ALLOW",
        PyTuple::new(py, SCRIPT_CLOSED_ALLOW)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Hand-derived from the oracle literals; any drift fails here first.
        assert_eq!(SCRIPT_F1_TOTAL_LINES, 50);
        assert_eq!(SCRIPT_F1_DECISIONS, 5);
        assert_eq!(SCRIPT_F2_TARGET_LINES, 1500);
        assert_eq!(SCRIPT_F8_FILES, 30);
        assert_eq!(SCRIPT_F8_GAMES_PER_FILE, 22);
        assert_eq!(SCRIPT_F8_MID, 3);
        assert_eq!(SCRIPT_F10_FILES, 8);
        assert_eq!(SCRIPT_F10_GAMES_PER_FILE, 12);
        assert_eq!(SCRIPT_F10_BUILDER, "f10-v1");
        assert_eq!(SCRIPT_F11_FILES, 64);
        assert_eq!(SCRIPT_F11_GAMES_PER_FILE, 12);
        assert_eq!(SCRIPT_F11_BUILDER, "f11-v1");
        assert_eq!(SCRIPT_SEED, 7);
        assert_eq!(SCRIPT_F2_SEED, 20260909);
        assert_eq!(SCRIPT_F8_EXPECTED_GAMES, 30 * 22);
        assert_eq!(SCRIPT_F10_EXPECTED_GAMES, 8 * 12);
        assert_eq!(SCRIPT_F11_EXPECTED_GAMES, 64 * 12);
        assert_eq!(SCRIPT_LOC_LIMIT_PYTHON, 600);
        assert_eq!(SCRIPT_LOC_LIMIT_TESTS, 1000);
        assert_eq!(SCRIPT_LOC_LIMIT_LEAN, 800);
        assert_eq!(SCRIPT_LOC_LIMIT_CRATES, 2000);
        assert_eq!(
            SCRIPT_CLOSED_ALLOW,
            [
                "wall_game_id",
                "copy_fold",
                "refit_number",
                "wall_less_marker"
            ]
        );
    }

    #[test]
    fn percentile_index_matches_oracle_pick() {
        // Oracle: min(int(q*n), n-1); empty -> None (the caller returns None).
        assert_eq!(percentile_index_core(0, 0.5).unwrap(), None);
        assert_eq!(percentile_index_core(5, 0.50).unwrap(), Some(2));
        assert_eq!(percentile_index_core(5, 0.99).unwrap(), Some(4));
        assert_eq!(percentile_index_core(1, 0.50).unwrap(), Some(0));
        assert_eq!(percentile_index_core(1, 0.99).unwrap(), Some(0));
        // Fail-closed: NaN / out-of-range never wrap to a negative index.
        assert!(percentile_index_core(5, f64::NAN).is_err());
        assert!(percentile_index_core(5, -0.1).is_err());
        assert!(percentile_index_core(5, 1.1).is_err());
        assert!(percentile_index_core(5, f64::INFINITY).is_err());
    }

    #[test]
    fn rate_matches_oracle_guard() {
        assert_eq!(rate_per_sec_core(100, 2.0), 50.0);
        assert_eq!(rate_per_sec_core(100, 0.0), 0.0);
        assert_eq!(rate_per_sec_core(100, -1.0), 0.0);
        assert_eq!(rate_per_sec_core(0, 2.0), 0.0);
        // NaN guard agrees with Python (`NaN > 0` is False -> 0.0).
        assert_eq!(rate_per_sec_core(100, f64::NAN), 0.0);
        assert_eq!(rate_per_sec_core(100, f64::INFINITY), 0.0);
    }

    #[test]
    fn loc_exceeds_matches_oracle_compare() {
        assert!(loc_exceeds_core(601, 600));
        assert!(!loc_exceeds_core(600, 600));
        assert!(!loc_exceeds_core(599, 600));
    }

    #[test]
    fn grid_total_matches_closed_forms() {
        assert_eq!(grid_total_core(30, 22).unwrap(), 660);
        assert_eq!(grid_total_core(8, 12).unwrap(), 96);
        assert_eq!(grid_total_core(64, 12).unwrap(), 768);
        assert!(grid_total_core(u64::MAX, 2).is_err());
    }

    #[test]
    fn allow_vocab_matches_oracle() {
        assert!(SCRIPT_CLOSED_ALLOW.contains(&"wall_game_id"));
        assert!(!SCRIPT_CLOSED_ALLOW.contains(&"anything_else"));
        assert_eq!(
            allow_drop_keys("wall_game_id").unwrap(),
            ["source_object_id", "game_id", "round_id"]
        );
        assert_eq!(
            allow_drop_keys("copy_fold").unwrap(),
            ["observation_hash", "concealed", "drawn", "dora"]
        );
        assert_eq!(
            allow_drop_keys("refit_number").unwrap(),
            ["wall", "live_wall_tiles_remaining"]
        );
        assert_eq!(
            allow_drop_keys("wall_less_marker").unwrap(),
            ["derivation_hash", "derivation", "wall_digest"]
        );
        assert!(allow_drop_keys("open_waiver").is_none());
    }

    #[test]
    fn pyfn_wiring_smoke() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(script_percentile_index(5, 0.50).unwrap(), Some(2));
            assert_eq!(script_rate_per_sec(100, 2.0), 50.0);
            assert!(script_loc_exceeds(601, 600));
            assert_eq!(script_grid_total(30, 22).unwrap(), 660);
            assert!(script_allow_is_valid("copy_fold"));
            assert!(!script_allow_is_valid("nope"));
            assert_eq!(
                script_allow_drop_keys("refit_number").unwrap(),
                vec!["wall".to_string(), "live_wall_tiles_remaining".to_string()]
            );
            assert!(script_allow_drop_keys("nope").is_none());
            // Bare-hex helper folds the feed owner (hello-world KAT).
            let hex = script_sha256_bare_hex(py, b"hello world".to_vec()).unwrap();
            assert_eq!(
                hex,
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            );
            assert!(!hex.starts_with("sha256:"));
        });
    }
}
