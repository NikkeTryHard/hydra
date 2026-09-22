//! eval_leaves2: held-out count + score-selection seed leaves on the EXISTING
//! `hydra2._native.eval` submodule.
//!
//! DAG: pyo3 + `hydra-search` (`eval::partition::held_count`,
//! `eval::score_selection_seed`, `eval::wall::WALL_TILES`) — already bridge
//! dependencies, see `crates/bridge/Cargo.toml:19-24`; no new dependency.
//! FORBIDS: torch/numpy draws (`torch.randperm` stays the split oracle in
//! `python/hydra2/eval/duplicate.py:425-428`, numpy PCG64 stays the
//! bootstrap/sign-flip oracle in `statistics.py:194-204,227-240` — draws never
//! cross); float sums (every mean stays Python); live-object assembly
//! (`BlockSplit` / `BlockAggregateResult` / `SelectionConfig` construction
//! stays Python); and any new JCS printer (all bytes/hash math lives in the
//! search owners below — this file only stages closed-domain values attached,
//! runs the checks detached, and returns scalars/bytes).
//!
//! TABLE ported here (frozen values + pure compute, oracles
//! `python/hydra2/eval/duplicate.py` + `statistics.py` via the search owners
//! `crates/search/src/eval/{partition,wall}.rs` + `eval/mod.rs`):
//! - `WALL_TILES` <- `wall::WALL_TILES` (`wall.rs:27`, physical wall length
//!   `136`) — referenced, never recopied. The 136-int gates stay Python in
//!   `duplicate.py:118-124,138-144`, now single-sourced on this const with
//!   the `hasattr` fallback (same shape as `schedule.py:43-65`).
//! - `eval_held_count` <- `split_blocks_held_out` held-N half
//!   (`duplicate.py:424` via `partition::held_count`, `partition.rs:49-59`):
//!   `max(1, min(n-1, round(n*ratio)))` with Python banker's `round`
//!   (`round_half_even`, `partition.rs:62-74`). The torch permutation
//!   (`duplicate.py:425-431`) stays the oracle — this leaf is the draw-free
//!   count half only (B2 held).
//! - `eval_score_selection_seed_bytes` <- `score_selection` seed derivation
//!   (`statistics.py:577` via `score_selection_seed`, `eval/mod.rs:231-233`):
//!   `sha256("score-selection-v1:{seed}")` raw 32 bytes. Takes the explicit
//!   `int` seed (never an RNG object); the `RandomStream` draw lane stays
//!   Python. The owner hashes — never recopied here.
//!
//! LOGIC staying Python (per-fn reasons):
//! - `find_near_duplicates` grouping: stays per `eval_duplicate.rs:31-32`
//!   (tile-map orchestration over the fingerprint hash — the per-wall
//!   `wall_fingerprint` calls and the grouping move only together).
//! - `validate_blocks_disjoint`: stays per `eval_duplicate.rs:33-34` (live
//!   `WallBlock` objects — a projected parallel-list shape would need a
//!   facade rewrite; the whole disjoint family moves in a later wave).
//! - `split_blocks_held_out` digest: stays Python `of_canonical` (no
//!   standalone owner fn exists — the seal is built inline at
//!   `partition.rs:164-170`, so a bridge leaf would restate owner law from a
//!   construction site and fork on the next payload change; feed stays the
//!   single hash site).
//! - `torch.randperm` split assembly + sorted sides
//!   (`duplicate.py:425-435`), `build_wall_blocks` / `make_block_manifest` /
//!   `balance_audit` / telemetry reports / `confirmation_sidecar`
//!   orchestration, the numpy draw lanes + `SelectionConfig` +
//!   `selection_gate_check` + `score_selection` assembly (TABLED in
//!   `eval_stats.rs:26-34` — untouched here), every `ContractError`
//!   translator (the bridge raises `ValueError` with byte-identical text; the
//!   translators map to `ContractError`), and the stale-`.so`
//!   `AttributeError` fallbacks.
//!
//! NONE-duplication note: `held_count` / `score_selection_seed` /
//! `WALL_TILES` are called, not recopied.
//!
//! Shape per fn: attached staging (`repr` via the live Python API, so
//! `{value!r}` renders byte-exact) → ONE `py.detach(|| …)` over owned plain
//! data with zero Python API inside (per `contracts.rs:434-437`,
//! `validate.rs:83-95`) → attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `sub.add(<str>, <usize>)` for the
//! frozen const mirrors `contracts.rs:1153`; `#[pyfunction]` mirrors
//! `eval::aggregate_wall_block` (`crates/bridge/src/eval.rs:91-98`);
//! attached-extract + one `py.detach(|| …)` with zero Python API inside
//! mirrors the `eval_schedule` leaves
//! (`crates/bridge/src/eval_schedule.rs:520-538`); `SearchError` mapped onto
//! `ValueError` mirrors `eval::search_err`
//! (`crates/bridge/src/eval.rs:60-63`); `is_instance_of::<PyBool>`
//! bool-rejection mirrors `contracts::plain_int_value`
//! (`crates/bridge/src/contracts.rs:69-74`); `Bound<'_, PyAny>` scalar
//! staging mirrors `qual_budget::as_u64`
//! (`crates/bridge/src/qual_budget.rs:230-235`); `Vec<u8>` → `bytes` return
//! crosses as the `bytes` object the translator compares with the `hashlib`
//! oracle; `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Feed/search owners (never reimplemented here): held-out count
//! `hydra_search::eval::partition::held_count`
//! (`crates/search/src/eval/partition.rs:49-59`, banker's `round_half_even`
//! at `:62-74`); score-seed bytes `hydra_search::eval::score_selection_seed`
//! (`crates/search/src/eval/mod.rs:231-233`, `sha256_bytes` over the
//! domain-separated ASCII); wall length `hydra_search::eval::wall::WALL_TILES`
//! (`crates/search/src/eval/wall.rs:27`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Non-`f64` ratios (`str`, `None`, `list`) fail staging here
//!   (`ValueError` with the oracle sentence) where the oracle's `0 < x < 1`
//!   raises `TypeError`. Both fail closed; validated `float` callers (the
//!   only translator path — `_validate_block_split_input` runs first) are
//!   unaffected.
//! - `bool` ratios/seeds stage as invalid with the oracle sentence where the
//!   oracle's comparison/`isinstance` would accept `True` as `1`. The split
//!   validator rejects non-`(0,1)` ratios and the selection config rejects
//!   `bool` seeds before any bridge call; well-typed callers are unaffected.
//! - Absurd-huge int seeds (`> i64::MAX`) fail closed here where the oracle
//!   would hash their decimal rendering (same class as the `eval_stats`
//!   absurd-huge-`resamples` arm); realistic seeds are unaffected.
//! - Negative seeds hash here (`-3` renders `-3`, exactly like the oracle
//!   f-string) even though `SelectionConfig` rejects them — the seed leaf
//!   accepts them exactly like `_validate_block_split_input` (which only
//!   checks `isinstance(int)`, `duplicate.py:406-407`).
//!
//! Single-cdylib tree: registers its const + pyfns on the EXISTING
//! `hydra2._native.eval` submodule via `register` (mirrors
//! `eval_blocks::register`); no new entry point. MAIN wiring:
//! `pub mod eval_leaves2;` in `lib.rs` (after `eval_leaves`, before
//! `eval_schedule`) plus `crate::eval_leaves2::register(&sub)?;` in
//! `eval::register` before `m.add_submodule(&sub)`
//! (`crates/bridge/src/eval.rs:124-131`).

use hydra_search::SearchError;
use hydra_search::eval::partition as partition_owner;
use hydra_search::eval::wall as wall_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyModule};

/// Physical wall length, tiles `0..136` (`duplicate.py` 136-gates,
/// `wall::WALL_TILES`, `wall.rs:27`) — referenced, never recopied.
const WALL_TILES: usize = wall_owner::WALL_TILES;

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Map the owner's short held-count keys onto the oracle sentences
/// (`_validate_block_split_input`, `duplicate.py:398-405`): the empty-blocks
/// arm gains its tuple sentence; the ratio arm interpolates the staged live
/// `repr` (byte-exact for every input).
fn held_owner_text(detail: &str, ratio_repr: &str) -> String {
    match detail {
        "blocks must be nonempty" => "blocks must be nonempty tuple of WallBlock".to_string(),
        "held_out_ratio must be in (0,1)" => {
            format!("held_out_ratio must be in (0,1), got {ratio_repr}")
        }
        _ => detail.to_string(),
    }
}

/// Held-out count (`split_blocks_held_out`, `duplicate.py:424` via
/// `partition::held_count`, `partition.rs:49-59`): `max(1,
/// min(n-1, round(n*ratio)))` with banker's rounding owned by
/// `round_half_even` (`partition.rs:62-74`). `ratio_repr` is the staged live
/// `repr` for the `{held_out_ratio!r}` slot; `ratio_value` is `Some` iff the
/// element is a non-`bool` numeric (the oracle comparison rejects `bool`
/// through the same sentence, staged here).
fn held_count_checked(
    n: usize,
    ratio_repr: &str,
    ratio_value: Option<f64>,
) -> Result<usize, String> {
    let ratio = match ratio_value {
        Some(v) if v.is_finite() && 0.0 < v && v < 1.0 => v,
        _ => {
            return Err(format!("held_out_ratio must be in (0,1), got {ratio_repr}"));
        }
    };
    if n == 0 {
        return Err("blocks must be nonempty tuple of WallBlock".to_string());
    }
    partition_owner::held_count(n, ratio).map_err(|err| match err {
        SearchError::InvalidArg { detail } => held_owner_text(detail, ratio_repr),
        other => other.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Held-out count (`duplicate.py:424`): `n` is `len(blocks)` (translator
/// passes the already-validated length); `ratio` stages attached (bool /
/// non-numeric / out-of-range share the oracle sentence with the live
/// `repr`); the banker's count runs detached via the partition owner.
#[pyfunction]
fn eval_held_count(py: Python<'_>, n: usize, ratio: Bound<'_, PyAny>) -> PyResult<usize> {
    let ratio_repr = py_repr(&ratio)?;
    let ratio_value: Option<f64> = if ratio.is_instance_of::<PyBool>() {
        None
    } else {
        ratio.extract().ok()
    };
    py.detach(|| held_count_checked(n, &ratio_repr, ratio_value))
        .map_err(PyValueError::new_err)
}

/// Score-selection seed bytes (`statistics.py:577` via
/// `score_selection_seed`, `eval/mod.rs:231-233`):
/// `sha256("score-selection-v1:{seed}")` raw 32 bytes. The explicit `int`
/// seed stages attached (`bool` shares the `SelectionConfig` sentence,
/// negatives pass through exactly like the oracle f-string, absurd-huge ints
/// fail closed); the owner hashes detached.
#[pyfunction]
fn eval_score_selection_seed_bytes(py: Python<'_>, seed: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let seed_repr = py_repr(&seed)?;
    let staged: Option<i64> = if seed.is_instance_of::<PyBool>() {
        None
    } else {
        seed.extract().ok()
    };
    let value = staged.ok_or_else(|| {
        PyValueError::new_err(format!("seed must be a non-negative int, got {seed_repr}"))
    })?;
    Ok(py
        .detach(|| hydra_search::eval::score_selection_seed(value))
        .to_vec())
}

/// Register the leaves2 const + pyfns on the EXISTING `eval` submodule
/// (mirrors `eval_blocks::register`): frozen const plus compute-detached
/// leaves; single cdylib, no new entry point. MAIN calls this from
/// `eval::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("WALL_TILES", WALL_TILES)?;
    sub.add_function(wrap_pyfunction!(eval_held_count, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_score_selection_seed_bytes, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_tiles_matches_physical_length() {
        // HEAD oracle pins: `duplicate.py:118,138` gate `136` tiles.
        assert_eq!(WALL_TILES, 136);
        assert_eq!(WALL_TILES, wall_owner::WALL_TILES);
    }

    #[test]
    fn held_count_matches_oracle_bankers_rounding() {
        // `max(1, min(n-1, round(n*ratio)))` with round-half-to-even:
        // 10*0.25 = 2.5 -> 2 (even), 10*0.35 = 3.5 -> 4 (even).
        assert_eq!(held_count_checked(10, "0.25", Some(0.25)).unwrap(), 2);
        assert_eq!(held_count_checked(10, "0.35", Some(0.35)).unwrap(), 4);
        // Clamp arms: n=4, ratio=0.2 -> round(0.8) = 1; n=1 clamps to 1.
        assert_eq!(held_count_checked(4, "0.2", Some(0.2)).unwrap(), 1);
        assert_eq!(held_count_checked(1, "0.2", Some(0.2)).unwrap(), 1);
        // Oracle sentences.
        assert_eq!(
            held_count_checked(0, "0.2", Some(0.2)).unwrap_err(),
            "blocks must be nonempty tuple of WallBlock"
        );
        assert_eq!(
            held_count_checked(4, "1.5", Some(1.5)).unwrap_err(),
            "held_out_ratio must be in (0,1), got 1.5"
        );
        assert_eq!(
            held_count_checked(4, "True", None).unwrap_err(),
            "held_out_ratio must be in (0,1), got True"
        );
    }

    #[test]
    fn score_seed_matches_hashlib_golden() {
        // Golden from `hashlib.sha256(b"score-selection-v1:0").hexdigest()`.
        let bytes = hydra_search::eval::score_selection_seed(0);
        assert_eq!(
            bytes,
            [
                0x54, 0xae, 0xbf, 0x5c, 0x2f, 0xb7, 0x62, 0xc6, 0x67, 0x3f, 0xb0, 0xa9, 0x38, 0x5a,
                0xfd, 0x9e, 0xfc, 0x77, 0xcc, 0x56, 0x27, 0xbb, 0x96, 0xd4, 0xdf, 0x76, 0x80, 0x22,
                0xb1, 0x04, 0x66, 0xe6,
            ]
        );
        // Distinct seeds diverge; negatives render (oracle f-string parity).
        assert_ne!(
            hydra_search::eval::score_selection_seed(0),
            hydra_search::eval::score_selection_seed(7)
        );
        assert_eq!(hydra_search::eval::score_selection_seed(-3).len(), 32);
    }
}
