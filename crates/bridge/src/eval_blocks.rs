//! eval_blocks: WallBlock validators + invalid-block policy leaves on the shared `eval` submodule.
//!
//! DAG: pyo3 + `hydra-search` (`eval::blocks::EXCLUSION_REASONS`, `eval::py_list_repr`)
//! ONLY (both already bridge dependencies, see `crates/bridge/Cargo.toml:19-25`;
//! no new dependency). FORBIDS: wall means (already bridged via
//! `eval::aggregate_wall_block`, `crates/bridge/src/eval.rs:91-98` — never a
//! second mean), telemetry row decisions (`telemetry_invalid_reason_for`
//! decides per row on the `contracts` submodule; the per-game call order stays
//! Python), hash/digest math (feed-owned), live-object assembly (`WallBlock` /
//! `BlockTolerance` / `ExcludedBlock` / `BlockAggregateResult` construction
//! stays Python in `python/hydra2/eval/blocks.py`), and any new JCS printer.
//!
//! TABLE ported here (frozen values + pure checks over plain projected values,
//! oracle `python/hydra2/eval/blocks.py` via `hydra_search::eval::blocks`,
//! `crates/search/src/eval/blocks.rs`):
//! - `EXCLUSION_REASONS` <- `blocks.py:82-89` via the owner const
//!   (`blocks.rs:78-85`, same order) — referenced, never recopied.
//! - `eval_blocks_check_wall_block` <- `WallBlock.__post_init__`
//!   (`blocks.py:53-66`): nonempty id, parallel lengths, nonempty game ids,
//!   finite contrasts with bool rejection and `{value!r}` slots. The owner
//!   (`WallBlock::new`, `blocks.rs:41-67`) carries the first three sentences
//!   verbatim but renders contrast failures as `NonFinite` owner-wording, so
//!   the checks run here directly with the oracle text (cross-checked against
//!   the owner lines cited per arm).
//! - `eval_blocks_empty_or_missing` <- `_first_disqualification` empty +
//!   missing arms (`blocks.py:135-145`): `len(contrasts) == 0`, then
//!   block-ordered missing rows with the `no telemetry rows for games [...]`
//!   detail rendered through the shared `py_list_repr` owner
//!   (`crates/search/src/eval/mod.rs:235-250`).
//! - `eval_blocks_flag_exclusion` <- `_first_disqualification` flag arms
//!   (`blocks.py:159-178`): per-game `fallback_used`/`timeout`/
//!   `illegal_action` gates in game order with fallback > timeout > illegal
//!   priority inside each game. Row flags arrive as plain projected values
//!   (construction-guaranteed bools); tolerance allows stage attached via
//!   truthiness so `not tolerance.allow_*` matches the oracle exactly.
//! - `eval_blocks_sort_wall_ids` <- `aggregate_blocks` visit order
//!   (`blocks.py:121` via `blocks.rs:117`): wall-id sorted collapse order.
//!   Byte order equals codepoint order over valid UTF-8, so the Rust sort
//!   matches `sorted(..., key=wall_id)`.
//!
//! LOGIC staying Python (`eval/blocks.py`): the four dataclasses,
//! `aggregate_blocks` loop assembly (sort leaf + exclusion leaves + the
//! existing `aggregate_wall_block` mean), the row-invalidity arm loop (it must
//! call `telemetry_invalid_reason` lazily per game in block order with
//! early exit — eager projection would evaluate later games the oracle never
//! touches, observably diverging when a later row raises on a tolerated
//! mode-required extra while an earlier row already excludes), the
//! `row marked` prefix mapping (one-line `str.startswith`, an FFI round-trip
//! buys nothing), `BlockTolerance` construction (the `allow_missing` half is
//! already bridged via `contracts.telemetry_tolerance_check`; the three bool
//! flags carry no oracle gate — truthiness is preserved at the flag leaf),
//! and every `ContractError` translator (the bridge raises `ValueError` with
//! byte-identical text; `blocks.py` maps to `ContractError`, mirroring
//! `statistics.py:118-126` including the stale-`.so` `AttributeError`
//! fallback).
//!
//! NONE-duplication note: `rg` for `EXCLUSION_REASONS` over `crates/bridge/src`
//! hits only this file plus the search owner (no other bridge exposes it);
//! `py_list_repr` is called, not recopied; the mean is called, not rederived.
//!
//! Shape per fn: attached staging (`repr` via the live Python API, so
//! `{value!r}` renders byte-exact; truthiness via `is_truthy`, so declared
//! `bool` flags match oracle semantics) → ONE `py.detach(|| …)` over owned
//! plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `validate.rs:83-95`) → attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `let py = sub.py()` mirrors
//! `contracts::register` (`crates/bridge/src/contracts.rs:1115`);
//! `PyTuple::new(py, [...])` for the frozen tuple mirrors `contracts.rs:1156`;
//! `#[pyfunction]` mirrors `eval::aggregate_wall_block`
//! (`crates/bridge/src/eval.rs:91-98`); attached-extract + one
//! `py.detach(|| ...)` with zero Python API inside mirrors the
//! `eval_schedule` leaves (`crates/bridge/src/eval_schedule.rs:520-538`);
//! `is_instance_of::<PyBool>` bool-rejection mirrors
//! `contracts::plain_int_value` (`crates/bridge/src/contracts.rs:69-74`);
//! `Vec<Bound<'_, PyAny>>` staging mirrors
//! `eval_validate_walls_disjoint` (`crates/bridge/src/eval_duplicate.rs:231-253`);
//! attached `repr` mirrors `rc_require.rs:63-65`; `Option<(String, String)>`
//! returns mirror `telemetry_invalid_reason_for`
//! (`crates/bridge/src/eval_leaves.rs:532-551`); `py_list_repr` is the shared
//! owner (`crates/search/src/eval/mod.rs:235-250`); `Python::initialize()` +
//! `Python::attach` in tests mirror (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Lone-surrogate `str` wall ids fail `String` extraction and are rejected
//!   with the nonempty-str message, where the oracle would accept the label
//!   (it only checks `isinstance` + emptiness). Same class as the
//!   `eval_leaves` case-id gate; well-formed UTF-8 callers are unaffected.
//! - Non-`str` game ids fail `Vec<String>` extraction (`TypeError`) where the
//!   oracle's `== ""` comparison would pass them through to a later
//!   missing-telemetry exclusion. Same class as the existing
//!   `eval::aggregate_wall_block` game-ids extraction
//!   (`crates/bridge/src/eval.rs:94-95`); declared `tuple[str, ...]` callers
//!   are unaffected.
//! - Game ids containing quotes render with plain single-quote wrapping
//!   (shared `py_list_repr` owner) where the oracle escapes them. Same
//!   closed-identifier domain as the owner's own doc note; realistic
//!   schedule-derived ids never contain quotes.
//! - Non-iterable `game_ids`/`contrasts` fail typed extraction (`TypeError`)
//!   where the oracle's `len()` raises `TypeError` with different text; both
//!   fail closed, and the declared tuple types are unaffected.
//! - Exotic truthiness (`__bool__` raising) on projected flags is evaluated
//!   eagerly attached where the oracle short-circuits; rows carry
//!   construction-guaranteed real bools, so reachable callers never observe
//!   this.
//!
//! Single-cdylib tree: registers its const + pyfns on the EXISTING
//! `hydra2._native.eval` submodule via `register` (mirrors
//! `persistence_report::register` on the shared submodule); no new entry
//! point. MAIN wiring: `pub mod eval_blocks;` in `lib.rs` plus
//! `crate::eval_blocks::register(&sub)?;` in `eval::register` before
//! `m.add_submodule(&sub)` (`crates/bridge/src/eval.rs:124-131`).

use hydra_search::eval::blocks::EXCLUSION_REASONS;
use hydra_search::eval::py_list_repr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyModule, PyTuple};

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// One staged contrast: `value` is `Some` iff the element is a non-`bool`
/// numeric with a finite `f64` value; `repr` is the exact Python `repr` for
/// the `{value!r}` slot on the failure arm.
struct StagedContrast {
    value: Option<f64>,
    repr: String,
}

/// Wall-block field gates (`WallBlock.__post_init__`, `blocks.py:53-66`):
/// nonempty id, parallel lengths, nonempty game ids, finite contrasts.
/// `game_empty` is precomputed attached with exact oracle semantics
/// (only a `str` element equal to `""` trips it — non-`str` elements pass
/// through, exactly like `game_id == ""`).
fn check_wall_block(
    wall_id: &Option<String>,
    game_len: usize,
    contrast_len: usize,
    game_empty: bool,
    contrasts: &[StagedContrast],
) -> Result<(), String> {
    if wall_id.as_ref().is_none_or(String::is_empty) {
        return Err("wall_id must be a nonempty str".to_string());
    }
    if game_len != contrast_len {
        return Err("game_ids and contrasts must have equal length".to_string());
    }
    if game_empty {
        return Err("game_ids must be nonempty strings".to_string());
    }
    for staged in contrasts {
        if staged.value.is_none() {
            return Err(format!("contrast must be finite, got {}", staged.repr));
        }
    }
    Ok(())
}

/// Empty + missing-telemetry arms (`blocks.py:135-145`) over plain projected
/// values. `game_ids` order is preserved for the missing list; the detail
/// renders through the shared `py_list_repr` owner. Wall identity stays
/// Python-side (the translator owns `ExcludedBlock` assembly).
fn empty_or_missing(
    game_ids: &[String],
    n_contrasts: usize,
    telemetry_keys: &[String],
) -> Option<(String, String)> {
    if n_contrasts == 0 {
        return Some((
            "empty_block".to_string(),
            "block carries no games".to_string(),
        ));
    }
    let missing: Vec<&str> = game_ids
        .iter()
        .filter(|game| !telemetry_keys.contains(*game))
        .map(String::as_str)
        .collect();
    if missing.is_empty() {
        None
    } else {
        Some((
            "missing_telemetry".to_string(),
            format!("no telemetry rows for games {}", py_list_repr(&missing)),
        ))
    }
}

/// One game's projected flag triple (plain bools staged attached via
/// truthiness, so oracle `and`/`not` semantics carry over exactly).
struct StagedFlags {
    fallback_used: bool,
    timeout: bool,
    illegal_action: bool,
}

/// Fallback/timeout/illegal arms (`blocks.py:159-178`): games outer in block
/// order, fallback > timeout > illegal priority inside each game. Wall
/// identity stays Python-side (the translator owns `ExcludedBlock` assembly).
fn flag_exclusion(
    game_ids: &[String],
    flags: &[StagedFlags],
    allow_fallback_used: bool,
    allow_timeout: bool,
    allow_illegal_action: bool,
) -> Option<(String, String)> {
    for (game_id, staged) in game_ids.iter().zip(flags.iter()) {
        if staged.fallback_used && !allow_fallback_used {
            return Some((
                "fallback_used".to_string(),
                format!("game {game_id} used fallback"),
            ));
        }
        if staged.timeout && !allow_timeout {
            return Some(("timeout".to_string(), format!("game {game_id} timed out")));
        }
        if staged.illegal_action && !allow_illegal_action {
            return Some((
                "illegal_action".to_string(),
                format!("game {game_id} produced an illegal action"),
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Wall-block field gates (`blocks.py:53-66`) over caller-projected plain
/// values. `wall_id` stages attached (`None` covers non-`str`, empty, and
/// lone-surrogate inputs sharing the first oracle sentence); game-id
/// emptiness stages attached with exact `== ""` semantics; contrasts stage
/// attached with bool rejection first (`bool` subclasses `int`) and exact
/// `repr` capture; the accept/reject decision runs detached.
#[pyfunction]
fn eval_blocks_check_wall_block(
    py: Python<'_>,
    wall_id: Bound<'_, PyAny>,
    game_ids: Bound<'_, PyAny>,
    contrasts: Bound<'_, PyAny>,
) -> PyResult<()> {
    let staged_id: Option<String> = wall_id.extract().ok().filter(|s: &String| !s.is_empty());
    let game_items: Vec<Bound<'_, PyAny>> = staged_games(&game_ids)?;
    let game_empty = game_items.iter().any(|item| {
        item.extract::<String>()
            .map(|s| s.is_empty())
            .unwrap_or(false)
    });
    let game_len = game_items.len();
    let mut staged_contrasts: Vec<StagedContrast> = Vec::with_capacity(game_len);
    for item in contrasts.extract::<Vec<Bound<'_, PyAny>>>().map_err(|_| {
        PyValueError::new_err("eval_blocks_check_wall_block: contrasts must be a sequence")
    })? {
        let repr = py_repr(&item)?;
        let value = if item.is_instance_of::<PyBool>() {
            None
        } else {
            item.extract::<f64>().ok().filter(|v| v.is_finite())
        };
        staged_contrasts.push(StagedContrast { value, repr });
    }
    let contrast_len = staged_contrasts.len();
    py.detach(|| {
        check_wall_block(
            &staged_id,
            game_len,
            contrast_len,
            game_empty,
            &staged_contrasts,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Stage `game_ids` attached: any iterable of opaque values (only `str`
/// emptiness is inspected here; element types are NOT narrowed — non-`str`
/// elements pass through exactly like the oracle's `== ""` comparison).
fn staged_games<'py>(game_ids: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    game_ids.extract::<Vec<Bound<'py, PyAny>>>().map_err(|_| {
        PyValueError::new_err("eval_blocks_check_wall_block: game_ids must be a sequence")
    })
}

/// Empty + missing-telemetry arms (`blocks.py:135-145`) over caller-projected
/// plain values: `n_contrasts` is `len(block.contrasts)`, `telemetry_keys` is
/// `list(telemetry_by_game)`. Returns `(reason, detail)` for the
/// `ExcludedBlock` translator, or `None` when the block survives these arms.
/// Compute runs detached.
#[pyfunction]
fn eval_blocks_empty_or_missing(
    py: Python<'_>,
    game_ids: Vec<String>,
    n_contrasts: usize,
    telemetry_keys: Vec<String>,
) -> PyResult<Option<(String, String)>> {
    Ok(py.detach(|| empty_or_missing(&game_ids, n_contrasts, &telemetry_keys)))
}

/// Fallback/timeout/illegal arms (`blocks.py:159-178`) over caller-projected
/// plain values: per-game flag triples in block order plus the tolerance
/// allows. Flags stage attached via truthiness (matching the oracle's
/// `and`/`not` exactly); the length gate fires attached and fail-closed
/// (translator-contract, unreachable on translator-built inputs); the
/// exclusion decision runs detached. Returns `(reason, detail)` or `None`.
#[pyfunction]
fn eval_blocks_flag_exclusion(
    py: Python<'_>,
    game_ids: Vec<String>,
    fallback_used: Vec<Bound<'_, PyAny>>,
    timeout: Vec<Bound<'_, PyAny>>,
    illegal_action: Vec<Bound<'_, PyAny>>,
    allow_fallback_used: Bound<'_, PyAny>,
    allow_timeout: Bound<'_, PyAny>,
    allow_illegal_action: Bound<'_, PyAny>,
) -> PyResult<Option<(String, String)>> {
    if fallback_used.len() != game_ids.len()
        || timeout.len() != game_ids.len()
        || illegal_action.len() != game_ids.len()
    {
        return Err(PyValueError::new_err(
            "eval_blocks_flag_exclusion: flag vectors must parallel game_ids",
        ));
    }
    let mut flags: Vec<StagedFlags> = Vec::with_capacity(game_ids.len());
    for ((fallback, timeout), illegal) in fallback_used
        .iter()
        .zip(timeout.iter())
        .zip(illegal_action.iter())
    {
        flags.push(StagedFlags {
            fallback_used: fallback.is_truthy()?,
            timeout: timeout.is_truthy()?,
            illegal_action: illegal.is_truthy()?,
        });
    }
    let allow_fallback_used = allow_fallback_used.is_truthy()?;
    let allow_timeout = allow_timeout.is_truthy()?;
    let allow_illegal_action = allow_illegal_action.is_truthy()?;
    Ok(py.detach(|| {
        flag_exclusion(
            &game_ids,
            &flags,
            allow_fallback_used,
            allow_timeout,
            allow_illegal_action,
        )
    }))
}

/// Collapse-visit order (`aggregate_blocks`, `blocks.py:121`): wall-id sorted
/// ids. Compute runs detached; the caller replays its block loop in the
/// returned order and assembles `BlockAggregateResult` Python-side.
#[pyfunction]
fn eval_blocks_sort_wall_ids(py: Python<'_>, wall_ids: Vec<String>) -> PyResult<Vec<String>> {
    Ok(py.detach(|| {
        let mut ordered = wall_ids;
        ordered.sort();
        ordered
    }))
}

/// Register the block leaves on the EXISTING `eval` submodule (mirrors
/// `persistence_report::register` on the shared submodule): frozen const
/// plus compute-detached leaves; single cdylib, no new entry point. MAIN
/// calls this from `eval::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("EXCLUSION_REASONS", PyTuple::new(py, EXCLUSION_REASONS)?)?;
    sub.add_function(wrap_pyfunction!(eval_blocks_check_wall_block, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_blocks_empty_or_missing, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_blocks_flag_exclusion, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_blocks_sort_wall_ids, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_search::eval::blocks::{WallBlock, first_disqualification};
    use hydra_search::eval::telemetry::{BlockTolerance, TelemetryRow};
    use std::collections::{HashMap, HashSet};

    /// Strict-clean telemetry row for owner cross-checks (unknown mode carries
    /// no extras, mirroring `test_mode_extras_become_required_only_for_that_mode`).
    fn clean_row() -> TelemetryRow {
        TelemetryRow {
            mode: "reference_eager_cpu".to_string(),
            wall_id: None,
            case_id: None,
            candidate_spec_hash: format!("sha256:{}", "aa".repeat(32)),
            hardware_hash: format!("sha256:{}", "bb".repeat(32)),
            environment_hash: format!("sha256:{}", "cc".repeat(32)),
            cold_start: false,
            synchronized_elapsed_ms: 12.5,
            model_calls: 3,
            exact_transitions: 40,
            particles: 0,
            fallback_used: false,
            timeout: false,
            illegal_action: false,
            cuda_peak_allocated_bytes: None,
            cuda_peak_reserved_bytes: None,
            host_peak_bytes: None,
            energy_joules: None,
            graph_breaks: None,
            recompiles: None,
            invalid_reason: None,
        }
    }

    #[test]
    fn exclusion_reasons_match_oracle_order() {
        // Hand-derived from HEAD `src/hydra2/eval/blocks.py:82-89`.
        assert_eq!(
            EXCLUSION_REASONS.to_vec(),
            vec![
                "missing_telemetry",
                "fallback_used",
                "timeout",
                "illegal_action",
                "row_invalid",
                "empty_block",
            ]
        );
    }

    #[test]
    fn wall_block_gates_match_oracle_text() {
        let good = vec![StagedContrast {
            value: Some(1.0),
            repr: "1.0".to_string(),
        }];
        assert!(check_wall_block(&Some("w-1".to_string()), 1, 1, false, &good).is_ok());
        assert_eq!(
            check_wall_block(&Some(String::new()), 0, 0, false, &[]).unwrap_err(),
            "wall_id must be a nonempty str"
        );
        assert_eq!(
            check_wall_block(&None, 0, 0, false, &[]).unwrap_err(),
            "wall_id must be a nonempty str"
        );
        assert_eq!(
            check_wall_block(&Some("w-1".to_string()), 2, 1, false, &good).unwrap_err(),
            "game_ids and contrasts must have equal length"
        );
        assert_eq!(
            check_wall_block(&Some("w-1".to_string()), 1, 1, true, &good).unwrap_err(),
            "game_ids must be nonempty strings"
        );
        for (repr, want) in [
            ("True", "contrast must be finite, got True"),
            ("inf", "contrast must be finite, got inf"),
            ("nan", "contrast must be finite, got nan"),
            ("'x'", "contrast must be finite, got 'x'"),
            ("None", "contrast must be finite, got None"),
        ] {
            let bad = vec![StagedContrast {
                value: None,
                repr: repr.to_string(),
            }];
            assert_eq!(
                check_wall_block(&Some("w-1".to_string()), 1, 1, false, &bad).unwrap_err(),
                want
            );
        }
    }

    #[test]
    fn empty_and_missing_arms_match_owner() {
        // Empty arm: owner cross-check (programmatic, not restated).
        let empty = WallBlock::new("w-empty".to_string(), Vec::new(), Vec::new()).unwrap();
        let rows: HashMap<String, TelemetryRow> = HashMap::new();
        let owner = first_disqualification(&empty, &rows, &BlockTolerance::strict())
            .unwrap()
            .unwrap();
        let leaf = empty_or_missing(&[], 0, &[]).unwrap();
        assert_eq!((owner.reason, owner.detail), leaf);
        assert_eq!(leaf.1, "block carries no games");

        // Missing arm, block order preserved over two gaps.
        let block = WallBlock::new(
            "w-x".to_string(),
            vec!["gb".to_string(), "ga".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let owner = first_disqualification(&block, &rows, &BlockTolerance::strict())
            .unwrap()
            .unwrap();
        let leaf = empty_or_missing(&block.game_ids, 2, &[]).unwrap();
        assert_eq!((owner.reason, owner.detail), leaf);
        assert_eq!(leaf.0, "missing_telemetry");
        assert_eq!(leaf.1, "no telemetry rows for games ['gb', 'ga']");

        // Survivors yield None on both sides.
        let mut present = HashMap::new();
        present.insert("ga".to_string(), clean_row());
        present.insert("gb".to_string(), clean_row());
        assert!(
            first_disqualification(&block, &present, &BlockTolerance::strict())
                .unwrap()
                .is_none()
        );
        assert!(
            empty_or_missing(&block.game_ids, 2, &["ga".to_string(), "gb".to_string()]).is_none()
        );
    }

    #[test]
    fn flag_arms_match_owner_priority_and_order() {
        // Game-outer, fallback > timeout > illegal inner: game0 timeout beats
        // game1 fallback; single-game illegal renders its sentence.
        let flags = vec![
            StagedFlags {
                fallback_used: false,
                timeout: true,
                illegal_action: true,
            },
            StagedFlags {
                fallback_used: true,
                timeout: false,
                illegal_action: false,
            },
        ];
        let games = vec!["g0".to_string(), "g1".to_string()];
        assert_eq!(
            flag_exclusion(&games, &flags, false, false, false).unwrap(),
            ("timeout".to_string(), "game g0 timed out".to_string())
        );
        let solo = vec![StagedFlags {
            fallback_used: false,
            timeout: false,
            illegal_action: true,
        }];
        assert_eq!(
            flag_exclusion(&["g9".to_string()], &solo, false, false, false).unwrap(),
            (
                "illegal_action".to_string(),
                "game g9 produced an illegal action".to_string()
            )
        );
        let fallback = vec![StagedFlags {
            fallback_used: true,
            timeout: false,
            illegal_action: false,
        }];
        assert_eq!(
            flag_exclusion(&["gb".to_string()], &fallback, false, false, false).unwrap(),
            (
                "fallback_used".to_string(),
                "game gb used fallback".to_string()
            )
        );
        // Tolerated flags admit; strict default excludes (owner cross-check).
        assert!(flag_exclusion(&["gb".to_string()], &fallback, true, true, true).is_none());
        let block = WallBlock::new("w-f".to_string(), vec!["gb".to_string()], vec![9.9]).unwrap();
        let mut rows = HashMap::new();
        let mut row = clean_row();
        row.fallback_used = true;
        rows.insert("gb".to_string(), row);
        let owner = first_disqualification(&block, &rows, &BlockTolerance::strict())
            .unwrap()
            .unwrap();
        assert_eq!(
            (owner.reason, owner.detail),
            flag_exclusion(&["gb".to_string()], &fallback, false, false, false).unwrap()
        );
        let allow = BlockTolerance {
            inner: hydra_search::eval::telemetry::TelemetryTolerance {
                allow_missing: HashSet::new(),
            },
            allow_fallback_used: true,
            allow_timeout: false,
            allow_illegal_action: false,
        };
        assert!(
            first_disqualification(&block, &rows, &allow)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sort_matches_oracle_visit_order() {
        Python::initialize();
        Python::attach(|py| {
            let out = eval_blocks_sort_wall_ids(
                py,
                vec![
                    "w-good".to_string(),
                    "w-fallback".to_string(),
                    "w-empty".to_string(),
                ],
            )
            .unwrap();
            assert_eq!(out, vec!["w-empty", "w-fallback", "w-good"]);
            let empty: Vec<String> = Vec::new();
            assert!(eval_blocks_sort_wall_ids(py, empty).unwrap().is_empty());
        });
    }

    #[test]
    fn staging_rejects_bool_and_renders_repr() {
        Python::initialize();
        Python::attach(|py| {
            use pyo3::types::PyList;
            // Non-finite contrast fails with the oracle sentence and repr.
            let contrasts = PyList::new(py, [1.0, f64::INFINITY]).unwrap();
            let wall = "w-1".into_pyobject(py).unwrap().into_any();
            let games = PyList::new(py, ["g1", "g2"]).unwrap();
            let err = eval_blocks_check_wall_block(
                py,
                wall.clone(),
                games.clone().into_any(),
                contrasts.into_any(),
            )
            .unwrap_err();
            assert_eq!(
                err.to_string(),
                "ValueError: contrast must be finite, got inf"
            );
            // `True` is not a contrast even though it subclasses `int`.
            let with_bool = PyList::new(py, [1.0, 2.0]).unwrap();
            with_bool.set_item(1, true).unwrap();
            let games2 = PyList::new(py, ["g1", "g2"]).unwrap();
            let err = eval_blocks_check_wall_block(
                py,
                wall.clone(),
                games2.into_any(),
                with_bool.into_any(),
            )
            .unwrap_err();
            assert_eq!(
                err.to_string(),
                "ValueError: contrast must be finite, got True"
            );
            // Golden triple passes.
            let ok_contrasts = PyList::new(py, [1.0, 2.0, 6.0]).unwrap();
            let ok_games = PyList::new(py, ["g1", "g2", "g3"]).unwrap();
            assert!(
                eval_blocks_check_wall_block(
                    py,
                    wall,
                    ok_games.into_any(),
                    ok_contrasts.into_any()
                )
                .is_ok()
            );
            // Flag length gate fires attached and fail-closed.
            let yes = true.into_pyobject(py).unwrap().to_owned().into_any();
            let no = false.into_pyobject(py).unwrap().to_owned().into_any();
            let err = eval_blocks_flag_exclusion(
                py,
                vec!["g1".to_string(), "g2".to_string()],
                vec![yes.clone()],
                vec![yes.clone(), no.clone()],
                vec![yes.clone(), no.clone()],
                yes.clone(),
                yes.clone(),
                no.clone(),
            )
            .unwrap_err();
            assert_eq!(
                err.to_string(),
                "ValueError: eval_blocks_flag_exclusion: flag vectors must parallel game_ids"
            );
            // Clean flags admit through the pyfn boundary.
            assert!(
                eval_blocks_flag_exclusion(
                    py,
                    vec!["g1".to_string()],
                    vec![no.clone()],
                    vec![no.clone()],
                    vec![no.clone()],
                    no.clone(),
                    no.clone(),
                    no.clone(),
                )
                .unwrap()
                .is_none()
            );
        });
    }
}
