//! stream_decode: spawn-decode tail pure leaves on the shared `columnar` submodule.
//!
//! DAG: pyo3 + `hydra-shard` ONLY (the 6-key `FORBIDDEN_IN_ACTOR` owner for the
//! actor-payload firewall — same edge as `oracle_guard.rs`, no new dependency;
//! `crates/bridge/Cargo.toml:20` carries it). FORBIDS: JCS/sha reimplementation
//! (no digest work here), JSON/event parsing (the verbatim-frame `splitlines` +
//! `json.loads` loop stays Python per the `decode.py` no-bridge-line-parse rule),
//! live-object construction (`GameRecord` / `StreamGame` assembly stays Python),
//! and transport/IO/orchestration (framing, the spawn pool, GC discipline,
//! `validate_game`, `compute_wall_hash`, `assign_split` stay Python).
//!
//! TABLE ported here — oracle `python/hydra2/data/stream_decode.py` (HEAD
//! `src/hydra2/data/stream_decode.py`, texts byte-identical):
//! - `stream_decode_check_drift` <- drift tripwire (`:103-107`): parsed-count
//!   vs bridge `event_count` with the stem interpolated.
//! - `stream_decode_source_type` <- source derivation (`:108-110`): the first
//!   event's `type` projects iff it is a `str`, else absent. The translator
//!   keeps the `{"type": t} / {}` dict assembly (attached construction).
//! - `stream_decode_coerce_wall_tiles` <- wall-tuple coercion (`:111-112`):
//!   `None` passthrough, else the bridge slot ints carried detached. The
//!   translator keeps the `tuple(...)` assembly.
//! - `stream_decode_check_wall_disjoint` <- wall-disjointness predicate
//!   (`:354-364`): first-seen split wins, `None` hashes skipped, conflict
//!   names the 16-char head plus both splits.
//! - `stream_decode_verify_no_privileged_leakage` <- actor firewall (`:367-371`):
//!   sorted offender list over the shard-owned set (referenced, never restated).
//! - `stream_decode_microbatch_bounds` <- microbatch gate + index math
//!   (`:339-346`): the `type(x) is not int or x <= 0` gate plus the
//!   `range(0, len, size)` bounds. The translator keeps the generator and the
//!   shallow slicing over live games.
//!
//! LOGIC staying Python: `_materialize_record` parse loop + `GameRecord`
//! assembly (transport + attached construction), `_decode_frames_worker` /
//! `PrefetchGameStream` / `_decode_worker_init` (spawn pool, GC, IO),
//! `slice_microbatches` slicing, `count_decisions` (one-line sum over live
//! objects, no frozen consts), `actor_payload` assembly (attached dict over a
//! live `StreamGame`; its firewall delegates here), the `PrefetchGameStream`
//! `prefetch` / `max_workers` gates (constructor one-liners), every `__all__`
//! name, and every `ContractError` translator (the bridge raises `ValueError`;
//! thin Python translators map to `ContractError` with byte-identical text,
//! except the wall-coercion translator which lets `TypeError`/`OverflowError`
//! propagate bare exactly like the oracle's bare `int(x)`).
//!
//! NONE-owner note: `rg` for `materialization drift|in splits|privileged keys
//! in actor payload|microbatch_size must be` over `crates/feed/src` +
//! `crates/search/src` hits only an unrelated `stream_driver.rs` doc comment
//! (driver take counters, never the `slice_microbatches` gate) — no crate owns
//! these leaves. The feed decode owns wall *extraction* from events
//! (`decode.rs:217-228`, `[i64; 136]` from `wall|wall_tiles|tiles`); the
//! slot-int coercion here is a different shape over bridge-provided ints.
//! The 6-key actor set is shard-owned
//! (`hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR`, referenced below).
//!
//! Shape per fn: attached staging (exact `repr(value)` text for every `{!r}`
//! slot, `str` extraction with bool exclusion) -> ONE `py.detach(|| ...)`
//! over owned plain data with zero Python API inside (per
//! `stream_scan.rs:183-203`) -> attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Translators call positionally; the bridge raises
//! `ValueError`/`TypeError`, Python maps to `ContractError`.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed `sub`
//! in `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-489`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `stream_scan::scan_file_games` (`crates/bridge/src/stream_scan.rs:183-203`);
//! attached `repr` mirrors `rc_require.rs:63-65`; `PyBool` exclusion +
//! `PyInt` gate mirror `obs_actor.rs:112`; shard-set membership mirrors
//! `oracle_guard.rs:91`; single-quoted list rendering mirrors
//! `oracle_guard.rs:150-164`; `sort()` byte-order == code-point order per
//! `oracle_guard.rs:137-148`; `Python::initialize()` + `Python::attach` in
//! tests mirror (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only — bridge slots and
//! typed dataclasses never produce these):
//! - Drift/coerce/bounds take fixed-width ints: a non-int stem/element/size
//!   fails closed with `TypeError` (the oracle f-strings/`int()` render or
//!   coerce some of these — e.g. `int(3.7)` truncates, `int("12")` parses).
//!   Bridge slots always carry ints, so the live path is unaffected.
//! - The microbatch gate uses `PyBool`-exclusion + `PyInt` (in-tree pattern)
//!   rather than exact `type(x) is int`: an `int` subclass instance (e.g. an
//!   `IntEnum`) passes here where the oracle rejects it. Plain ints and bools
//!   — the only realistic inputs — agree exactly.
//! - The wall-conflict head slices `chars().take(16)`: Python `[:16]` counts
//!   code points, Rust byte slicing would split them. Wall hashes are hex, so
//!   both agree; the char path keeps non-ASCII garbage total.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.columnar`
//! submodule (decode-adjacent; `rows_seal` / `stream_scan` precedent); no new
//! entry point. MAIN wiring (shared files, MAIN-only): `pub mod stream_decode;`
//! in `lib.rs` next to the `stream_*` mods plus
//! `crate::stream_decode::register(&sub)?;` at the end of
//! `columnar::register` (`columnar.rs:403-404`, after the `stream_scan` line).

use std::collections::HashMap;

use hydra_shard::replay::expand::FORBIDDEN_IN_ACTOR as SHARD_FORBIDDEN_IN_ACTOR;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyInt, PyModule};

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Python-`repr`-style rendering of a string list (`['a', 'b']`, single-quoted
/// per the bridge convention; offender keys are ASCII identifiers so no
/// escaping diverges — mirrors `oracle_guard.rs:150-164`).
fn py_str_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// Detached cores (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Drift tripwire (`stream_decode.py:103-107`): the verbatim-frame parse must
/// agree with the bridge `event_count`, with the file stem interpolated.
fn drift_core(parsed_len: i64, event_count: i64, stem: &str) -> Result<(), String> {
    if parsed_len == event_count {
        Ok(())
    } else {
        Err(format!(
            "decode materialization drift for {stem}: {parsed_len} parsed vs bridge event_count {event_count}"
        ))
    }
}

/// Wall-disjointness core (`stream_decode.py:354-364`): `None` hashes skipped,
/// first-seen split wins, the first cross-split repeat fails naming the
/// 16-char head plus both splits (char-based, matching Python `[:16]`).
fn check_wall_disjoint_core(pairs: &[(Option<String>, String)]) -> Result<(), String> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    for (wall_opt, split) in pairs {
        let wall = match wall_opt {
            Some(wall) => wall,
            None => continue,
        };
        match seen.get(wall.as_str()) {
            None => {
                seen.insert(wall.as_str(), split.as_str());
            }
            Some(previous) if **previous != *split => {
                let head: String = wall.chars().take(16).collect();
                return Err(format!("wall {head} in splits {previous} and {split}"));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Actor-firewall offender core (`stream_decode.py:367-371`): non-`str` keys
/// stage `None` and are skipped exactly like the oracle membership test;
/// offenders keep input order here and are sorted by the pyfn (mirroring the
/// oracle's `sorted(...)` — byte order equals code-point order for UTF-8).
fn privileged_bad_core(keys: &[Option<String>]) -> Vec<String> {
    keys.iter()
        .filter_map(|key| {
            key.as_ref().and_then(|text| {
                if SHARD_FORBIDDEN_IN_ACTOR.contains(&text.as_str()) {
                    Some(text.clone())
                } else {
                    None
                }
            })
        })
        .collect()
}

/// Microbatch gate + index math (`stream_decode.py:339-346`): `size` stages
/// `None` unless it is a plain (non-`bool`) int; non-positive or non-int
/// rejects carry the oracle text with the staged `repr`.
fn microbatch_bounds_core(
    batch_len: usize,
    size: Option<i64>,
    repr_text: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let step: usize = match size {
        Some(value) if value > 0 => usize::try_from(value)
            .map_err(|_| format!("microbatch_size must be a positive int, got {repr_text}"))?,
        _ => {
            return Err(format!(
                "microbatch_size must be a positive int, got {repr_text}"
            ));
        }
    };
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < batch_len {
        bounds.push((start, (start + step).min(batch_len)));
        start += step;
    }
    Ok(bounds)
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Materialization drift gate over a parsed count plus the bridge
/// `event_count` and file stem (mirrors `stream_decode.py:103-107` without
/// the JSON parse loop, which stays Python). Translators call positionally;
/// the bridge raises `ValueError`, Python maps to `ContractError` with
/// byte-identical text. Compute runs detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (parsed_len, event_count, stem))]
fn stream_decode_check_drift(
    py: Python<'_>,
    parsed_len: i64,
    event_count: i64,
    stem: String,
) -> PyResult<()> {
    py.detach(|| drift_core(parsed_len, event_count, &stem))
        .map_err(PyValueError::new_err)
}

/// Source-type projection over the first event's `type` value (mirrors
/// `stream_decode.py:108-110` without the dict assembly, which stays
/// Python): `str` (or `str` subclass) projects, every other value — including
/// `None` — stages absent exactly like the oracle `isinstance` gate. Total:
/// never raises. Compute runs detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (first_type,))]
fn stream_decode_source_type(
    py: Python<'_>,
    first_type: Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let staged: Option<String> = first_type.extract().ok();
    Ok(py.detach(|| staged))
}

/// Wall-tile coercion over bridge slot ints (mirrors `stream_decode.py:112`
/// without the `tuple(...)` assembly, which stays Python): `None` passes
/// through, int vecs move detached. Non-int elements fail closed with
/// `TypeError` (the oracle `int(x)` would coerce floats/numeric strings —
/// unreachable for bridge slots, which always carry ints). Compute runs
/// detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (wall,))]
fn stream_decode_coerce_wall_tiles(
    py: Python<'_>,
    wall: Option<Vec<i64>>,
) -> PyResult<Option<Vec<i64>>> {
    Ok(py.detach(|| wall))
}

/// Wall-disjointness gate over staged `(wall_hash_or_None, split)` pairs
/// (mirrors `stream_decode.py:354-364` without live `StreamGame` objects —
/// the translator stages the pairs positionally). Translators call
/// positionally; the bridge raises `ValueError`, Python maps to
/// `ContractError` with byte-identical text. Compute runs detached with zero
/// Python API inside.
#[pyfunction]
#[pyo3(signature = (pairs,))]
fn stream_decode_check_wall_disjoint(
    py: Python<'_>,
    pairs: Vec<(Option<String>, String)>,
) -> PyResult<()> {
    py.detach(|| check_wall_disjoint_core(&pairs))
        .map_err(PyValueError::new_err)
}

/// Actor-payload firewall over staged key values (mirrors
/// `stream_decode.py:367-371` without the live mapping — the translator
/// passes `list(payload)` positionally). Non-`str` keys stage absent and are
/// skipped exactly like the oracle membership test. Translators call
/// positionally; the bridge raises `ValueError`, Python maps to
/// `ContractError` with byte-identical text. Compute runs detached with zero
/// Python API inside.
#[pyfunction]
#[pyo3(signature = (keys,))]
fn stream_decode_verify_no_privileged_leakage(
    py: Python<'_>,
    keys: Vec<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut staged: Vec<Option<String>> = Vec::with_capacity(keys.len());
    for key in &keys {
        staged.push(key.extract().ok());
    }
    py.detach(|| {
        let mut bad = privileged_bad_core(&staged);
        bad.sort();
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "privileged keys in actor payload: {}",
                py_str_list(&bad)
            ))
        }
    })
    .map_err(PyValueError::new_err)
}

/// Microbatch gate plus slice bounds over the batch length (mirrors
/// `stream_decode.py:343-346` without the generator or the shallow slicing
/// over live games, which stay Python). The `PyBool`-exclusion + `PyInt`
/// gate mirrors `obs_actor.rs:112`; the `{!r}` slot carries the exact staged
/// `repr`. Translators call positionally; the bridge raises `ValueError`,
/// Python maps to `ContractError` with byte-identical text. Compute runs
/// detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (batch_len, microbatch_size))]
fn stream_decode_microbatch_bounds(
    py: Python<'_>,
    batch_len: usize,
    microbatch_size: Bound<'_, PyAny>,
) -> PyResult<Vec<(usize, usize)>> {
    let repr_text = py_repr(&microbatch_size)?;
    let staged: Option<i64> = if microbatch_size.is_instance_of::<PyBool>()
        || !microbatch_size.is_instance_of::<PyInt>()
    {
        None
    } else {
        microbatch_size.extract().ok()
    };
    py.detach(|| microbatch_bounds_core(batch_len, staged, &repr_text))
        .map_err(PyValueError::new_err)
}

/// Attach the stream-decode leaves to the caller-provided `columnar`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `rows_seal.rs:484-489`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(stream_decode_check_drift, sub)?)?;
    sub.add_function(wrap_pyfunction!(stream_decode_source_type, sub)?)?;
    sub.add_function(wrap_pyfunction!(stream_decode_coerce_wall_tiles, sub)?)?;
    sub.add_function(wrap_pyfunction!(stream_decode_check_wall_disjoint, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        stream_decode_verify_no_privileged_leakage,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(stream_decode_microbatch_bounds, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod stream_decode_tests {
    use super::*;
    use pyo3::types::PyString;

    #[test]
    fn drift_passes_on_equal_counts() {
        assert!(drift_core(3, 3, "stem").is_ok());
        assert!(drift_core(0, 0, "stem").is_ok());
    }

    #[test]
    fn drift_names_stem_and_counts() {
        assert_eq!(
            drift_core(3, 2, "part-00a9").expect_err("must reject"),
            "decode materialization drift for part-00a9: 3 parsed vs bridge event_count 2",
        );
    }

    #[test]
    fn wall_disjoint_vacuous_without_hashes() {
        let pairs = vec![(None, "train".to_string()), (None, "test".to_string())];
        assert!(check_wall_disjoint_core(&pairs).is_ok());
        let empty: Vec<(Option<String>, String)> = Vec::new();
        assert!(check_wall_disjoint_core(&empty).is_ok());
    }

    #[test]
    fn wall_disjoint_same_split_repeat_ok() {
        let pairs = vec![
            (Some("abcdef0123456789XYZ".to_string()), "train".to_string()),
            (Some("abcdef0123456789XYZ".to_string()), "train".to_string()),
        ];
        assert!(check_wall_disjoint_core(&pairs).is_ok());
    }

    #[test]
    fn wall_disjoint_conflict_names_head_and_splits() {
        let wall = "abcdef0123456789XYZ".to_string();
        let pairs = vec![
            (Some(wall.clone()), "train".to_string()),
            (Some("other".to_string()), "train".to_string()),
            (Some(wall), "test".to_string()),
        ];
        assert_eq!(
            check_wall_disjoint_core(&pairs).expect_err("must reject"),
            "wall abcdef0123456789 in splits train and test",
        );
    }

    #[test]
    fn firewall_passes_on_clean_keys() {
        let keys = vec![Some("game_id".to_string()), Some("split".to_string()), None];
        assert!(privileged_bad_core(&keys).is_empty());
        let empty: Vec<Option<String>> = Vec::new();
        assert!(privileged_bad_core(&empty).is_empty());
    }

    #[test]
    fn firewall_collects_shard_set_members() {
        assert!(SHARD_FORBIDDEN_IN_ACTOR.contains(&"wall"));
        let keys = vec![
            Some("wall".to_string()),
            Some("game_id".to_string()),
            Some("hidden_tiles".to_string()),
        ];
        assert_eq!(
            privileged_bad_core(&keys),
            vec!["wall".to_string(), "hidden_tiles".to_string()],
        );
    }

    #[test]
    fn microbatch_bounds_tile_cover() {
        assert_eq!(
            microbatch_bounds_core(10, Some(3), "3").expect("bounds"),
            vec![(0, 3), (3, 6), (6, 9), (9, 10)],
        );
        assert_eq!(
            microbatch_bounds_core(0, Some(3), "3").expect("bounds"),
            Vec::new(),
        );
        assert_eq!(
            microbatch_bounds_core(2, Some(8), "8").expect("bounds"),
            vec![(0, 2)],
        );
    }

    #[test]
    fn microbatch_gate_rejects_with_repr() {
        assert_eq!(
            microbatch_bounds_core(10, Some(0), "0").expect_err("must reject"),
            "microbatch_size must be a positive int, got 0",
        );
        assert_eq!(
            microbatch_bounds_core(10, Some(-2), "-2").expect_err("must reject"),
            "microbatch_size must be a positive int, got -2",
        );
        assert_eq!(
            microbatch_bounds_core(10, None, "'3'").expect_err("must reject"),
            "microbatch_size must be a positive int, got '3'",
        );
    }

    #[test]
    fn source_type_projects_only_strings() {
        Python::initialize();
        Python::attach(|py| {
            let text = PyString::new(py, "start_game").into_any();
            assert_eq!(
                stream_decode_source_type(py, text).expect("str projects"),
                Some("start_game".to_string()),
            );
            let number = 42i64.into_pyobject(py).expect("int stages").into_any();
            assert_eq!(
                stream_decode_source_type(py, number).expect("int absent"),
                None,
            );
            let none = py.None().into_bound(py);
            assert_eq!(
                stream_decode_source_type(py, none).expect("none absent"),
                None,
            );
        });
    }

    #[test]
    fn wall_coerce_passes_none_and_ints_through() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                stream_decode_coerce_wall_tiles(py, None).expect("none"),
                None,
            );
            assert_eq!(
                stream_decode_coerce_wall_tiles(py, Some(vec![1, 2, 3])).expect("ints"),
                Some(vec![1, 2, 3]),
            );
        });
    }

    #[test]
    fn firewall_pyfn_sorts_offenders() {
        Python::initialize();
        Python::attach(|py| {
            let keys = vec![
                PyString::new(py, "wall").into_any(),
                PyString::new(py, "game_id").into_any(),
                PyString::new(py, "dead_wall").into_any(),
            ];
            let err =
                stream_decode_verify_no_privileged_leakage(py, keys).expect_err("must reject");
            assert_eq!(
                err.to_string(),
                "ValueError: privileged keys in actor payload: ['dead_wall', 'wall']",
            );
            let clean = vec![PyString::new(py, "game_id").into_any()];
            assert!(stream_decode_verify_no_privileged_leakage(py, clean).is_ok(),);
        });
    }

    #[test]
    fn microbatch_pyfn_rejects_bool_and_text() {
        Python::initialize();
        Python::attach(|py| {
            let size = 4i64.into_pyobject(py).expect("int stages").into_any();
            assert_eq!(
                stream_decode_microbatch_bounds(py, 10, size).expect("bounds"),
                vec![(0, 4), (4, 8), (8, 10)],
            );
            let flag = true
                .into_pyobject(py)
                .expect("bool stages")
                .to_owned()
                .into_any();
            assert_eq!(
                stream_decode_microbatch_bounds(py, 10, flag)
                    .expect_err("bool must reject")
                    .to_string(),
                "ValueError: microbatch_size must be a positive int, got True",
            );
            let text = PyString::new(py, "3").into_any();
            assert_eq!(
                stream_decode_microbatch_bounds(py, 10, text)
                    .expect_err("str must reject")
                    .to_string(),
                "ValueError: microbatch_size must be a positive int, got '3'",
            );
        });
    }

    #[test]
    fn disjoint_pyfn_names_conflict() {
        Python::initialize();
        Python::attach(|py| {
            let pairs = vec![
                (Some("w".repeat(20)), "train".to_string()),
                (Some("w".repeat(20)), "held_out".to_string()),
            ];
            let err = stream_decode_check_wall_disjoint(py, pairs).expect_err("must reject");
            assert_eq!(
                err.to_string(),
                "ValueError: wall wwwwwwwwwwwwwwww in splits train and held_out",
            );
        });
    }

    #[test]
    fn drift_pyfn_checks_counts() {
        Python::initialize();
        Python::attach(|py| {
            assert!(stream_decode_check_drift(py, 2, 2, "s".to_string()).is_ok(),);
            assert_eq!(
                stream_decode_check_drift(py, 2, 5, "s".to_string())
                    .expect_err("must reject")
                    .to_string(),
                "ValueError: decode materialization drift for s: 2 parsed vs bridge event_count 5",
            );
        });
    }
}
