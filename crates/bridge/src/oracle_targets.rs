//! oracle_targets: WP-07B oracle teacher-target frozen consts + pure target-math cores
//! on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (same edge as `utility.rs`; no feed/shard/search edge — digest
//! bytes cross staged from the existing `sha256_digest` canon bridge, and the
//! ranks→`utility()` mapping stays the Python oracle, so no hash/scoring owner is
//! touched here). FORBIDS: JCS/sha reimplementation (the hash owner stays
//! Python-side; Rust only folds already-staged digest bytes/words), rank
//! resolution and `utility()` duplication (`resolve_final_ranks` in
//! `contracts.rs` plus `utility_values_for_ranks` in `utility.rs` stay the
//! owners — this file only pins the manifest anchors they consume), live-object
//! orchestration (`OracleTarget` assembly, privileged-dict dispatch, the
//! `make_utility_manifest` builder, and every `ContractError` text stay Python),
//! and wall-clock/RNG.
//!
//! TABLE ported here — oracle `python/hydra2/belief/oracle_targets.py`:
//! - consts `ORACLE_BELIEF_DIM` / `ORACLE_VALUE_DIM` / `ORACLE_EVENT_KIND_COUNT`
//!   <- target dims (`oracle_targets.py:27-28,32,35`): 34-wide belief, 4-wide
//!   value, `0..20` event kinds.
//! - consts `ORACLE_RANK_VALUES` / `ORACLE_VALUE_MIN` / `ORACLE_VALUE_MAX` /
//!   `ORACLE_ZERO_SUM_ABS_TOL` <- utility anchors (`oracle_targets.py:111,113-114,233`):
//!   the day-one `(20, 10, -10, -20)` golden, `[-100, 100]` bounds, and the
//!   `abs_tol=1e-9` zero-sum gate.
//! - consts `ORACLE_SCORE_FOR_RANK` / `ORACLE_SCORE_BASE` <- score synthesis
//!   (`oracle_targets.py:149,151`): rank `r` → `{40000, 30000, 20000, 10000}[r-1]`
//!   with `point_deltas = scores - 25000`. `ORACLE_SCORE_BASE` is numerically
//!   equal to the already-bridged `STARTING_POINTS` (`contracts.rs:1154`) but is
//!   pinned here as the oracle's literal so teacher math never follows a future
//!   rules change by accident (the `WORLD_DEFAULT_SCORES` precedent at
//!   `belief_leaves.rs:105-106`).
//! - consts `ORACLE_UTILITY_ID` / `ORACLE_UTILITY_SCHEMA_VERSION` /
//!   `ORACLE_RULES_HASH` <- manifest identity (`oracle_targets.py:106-109`).
//!   `rules_id` is NOT restated (`contracts` already owns `RULES_ID`);
//!   `objective`/`tie_policy` are NOT restated either (`utility.rs` owns
//!   `UTILITY_OBJECTIVE`/`UTILITY_TIE_POLICY` on this same submodule).
//! - `oracle_normalize_34` <- hidden-count / wait-histogram tails
//!   (`oracle_targets.py:65-67,75-77`): the total-or-`1.0` rule + division over
//!   caller-staged floats and the caller-computed `float(sum(source))` total
//!   (shape dispatch + `sum()` + `float()` staging stay Python, so Python
//!   `sum()` type semantics and error types never drift — see [`normalize_34`]).
//! - `oracle_teacher_logits_for` <- teacher lane
//!   (`oracle_targets.py:278-280`): `ln(max(p, 1e-6))` with the oracle's NaN
//!   order (see [`teacher_logit`]).
//! - `oracle_value_from_rank` <- legacy single-rank shape
//!   (`oracle_targets.py:247-251`): `RANK_VALUES[rank]` at the rank index.
//! - `oracle_nibble_scores_from_word` <- hash-fallback value tail
//!   (`oracle_targets.py:266`): low-nibble `/15.0` lanes over the staged
//!   `int(digest_hex[:8], 16)` word (the sum + rule + division stay in the
//!   oracle's own expression — see [`nibble_scores_from_word`]).
//! - `oracle_belief_from_digest` <- hash-fallback belief tail
//!   (`oracle_targets.py:86-93`): the `(h * 3)[:34]` repeat fold (a 32-byte
//!   digest never reaches the `len(h) >= 34` branch) plus `1.0`, normalized.
//!
//! LOGIC staying Python: the `OracleTarget` dataclass, every privileged-dict
//! shape dispatch (`hidden_tiles` / `wait_tiles` / `ranks` / `value_vector` /
//! single-rank), the `_oracle_utility_manifest` builder (now sourced from these
//! consts), `_value_from_ranks_via_utility` (the `utility()` call stays the fixed
//! point — never duplicated here), the explicit 4-list number/bounds/zero-sum
//! validation (decision_id-interpolated `ContractError` text), and every
//! `__all__`/Literal.
//!
//! NONE-owner note: `rg` for `ORACLE_|TEACHER_|BELIEF_DIM|VALUE_DIM|teacher_logits|
//! belief_from_digest|synthetic_value` over `crates/bridge/src` hits only lowercase
//! prose (`oracle literals`, `oracle fold`) — no caps const or pyfn owns these
//! leaves; `rg` for `3042a493|RANK_VALUES|score_for_rank` over `crates/feed/src` +
//! `crates/search/src` hits only the caller-sided `rank_values` params of
//! `feed::fixed` (never the day-one golden) — the manifest anchors are fresh here.
//!
//! Shape per fn: attached staging (bool-rejection + range checks touch Python
//! memory, never detached — the `contracts.rs:69-74` `plain_int_value`
//! precedent) -> ONE `py.detach(|| ...)` over owned plain data with zero Python
//! API inside (per `utility.rs:128-130,163-165`) -> attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Consts via `sub.add`, frozen
//! tuples via `PyTuple::new` (per `belief_leaves.rs:197`). Total helpers: every
//! input maps to a value or a `ValueError` — never a default, never a panic.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `utility.rs:196-201`); no new entry point.
//! MAIN wiring: `pub mod oracle_targets;` in `lib.rs` plus
//! `crate::oracle_targets::register(&sub)?;` in `contracts.rs` next to the
//! `crate::belief_natural::register(&sub)?;` line.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyInt, PyModule, PyTuple};

/// Deterministic belief-target width (`oracle_targets.py:27,64,71`): hidden-count
/// lists, wait histograms, and the hash-synthetic slice are all 34-wide.
const ORACLE_BELIEF_DIM: usize = 34;

/// Deterministic value-target width (`oracle_targets.py:32,134,199-204,247`):
/// ranks quads, explicit value vectors, and single-rank placements are all 4-wide.
const ORACLE_VALUE_DIM: usize = 4;

/// Frozen event-kind bound (`oracle_targets.py:35`): the `OracleTarget.event_target`
/// next-event kind id lives in `0..20`. Materialized by the store loader, which
/// stays Python — this const only pins the dataclass invariant.
const ORACLE_EVENT_KIND_COUNT: usize = 20;

/// Teacher-logit epsilon (`oracle_targets.py:278`): `logits = log(max(p, eps))`.
const ORACLE_TEACHER_EPS: f64 = 1e-6;

/// Canonical day-one rank values (`oracle_targets.py:111`): the utility manifest's
/// `rank_values` golden `(20, 10, -10, -20)` — zero-sum on the utility scale.
const ORACLE_RANK_VALUES: [f64; 4] = [20.0, 10.0, -10.0, -20.0];

/// Manifest value floor (`oracle_targets.py:113`).
const ORACLE_VALUE_MIN: f64 = -100.0;

/// Manifest value ceiling (`oracle_targets.py:114`).
const ORACLE_VALUE_MAX: f64 = 100.0;

/// Explicit-vector zero-sum tolerance (`oracle_targets.py:233`):
/// `math.isclose(total, 0.0, rel_tol=0.0, abs_tol=1e-9)`.
const ORACLE_ZERO_SUM_ABS_TOL: f64 = 1e-9;

/// Rank-indexed final scores (`oracle_targets.py:149`): rank `r` in `1..=4` maps
/// to `ORACLE_SCORE_FOR_RANK[r - 1]` before the `utility()` ranks call.
const ORACLE_SCORE_FOR_RANK: [u32; 4] = [40_000, 30_000, 20_000, 10_000];

/// Point-delta base (`oracle_targets.py:151`): `point_deltas = scores - 25000`.
const ORACLE_SCORE_BASE: u32 = 25_000;

/// Canonical day-one utility id (`oracle_targets.py:106`).
const ORACLE_UTILITY_ID: &str = "expected_final_placement_tenhou_4p_hanchan_v1";

/// Canonical day-one utility schema version (`oracle_targets.py:107`).
const ORACLE_UTILITY_SCHEMA_VERSION: &str = "1.0.0";

/// Canonical day-one rules hash (`oracle_targets.py:109`).
const ORACLE_RULES_HASH: &str =
    "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b";

/// Pure 34-normalize core (`oracle_targets.py:65-67,75-77,92-93`): every lane
/// divided by the caller-computed `total`, except a `total` of exactly `0.0`
/// (including `-0.0`, which `== 0.0`) maps to `1.0`. The total itself is staged
/// caller-side as `float(sum(source))` over the ORIGINAL list because Python
/// `sum()` semantics are type-dependent (exact big-int accumulation for int
/// lanes, sequential float adds otherwise) — no float fold here reproduces both,
/// so the sum stays where the oracle wrote it and only the rule + division live
/// here. A NaN total propagates lane-wise exactly like the oracle
/// (`nan != 0.0` keeps the NaN total). Zero Python API: runs detached.
fn normalize_34(values: &[f64; 34], total: f64) -> [f64; 34] {
    let denom = if total != 0.0 { total } else { 1.0 };
    let mut out = [0.0f64; ORACLE_BELIEF_DIM];
    for i in 0..ORACLE_BELIEF_DIM {
        out[i] = values[i] / denom;
    }
    out
}

/// Pure teacher-logit lane (`oracle_targets.py:278-280`): `ln(max(p, eps))` with
/// the oracle's argument order preserved for NaN — Python `max(p, eps)` keeps a
/// NaN `p` (the `eps > p` comparison is False), so NaN maps to `ln(nan) = nan`
/// here too instead of `ln(eps)`. Zero Python API: runs detached.
fn teacher_logit(prob: f64) -> f64 {
    let bounded = if prob.is_nan() {
        prob
    } else {
        prob.max(ORACLE_TEACHER_EPS)
    };
    bounded.ln()
}

/// Pure single-rank value core (`oracle_targets.py:247-251`): an all-zero 4-vector
/// with `ORACLE_RANK_VALUES[rank]` stored at the rank index (utility scale, never
/// one-hot). The caller gates `rank` to `0..3`; the index below is proven in
/// range. Zero Python API: runs detached.
fn value_at_rank(rank: u8) -> [f64; 4] {
    let mut vec = [0.0f64; ORACLE_VALUE_DIM];
    vec[usize::from(rank)] = ORACLE_RANK_VALUES[usize::from(rank)];
    vec
}

/// Pure synthetic nibble-score core (`oracle_targets.py:266`): the low nibble of
/// each word lane over `15.0`. Per-lane exact widening plus one deterministic
/// division each — deliberately NO summation here: the caller sums the returned
/// lanes with the oracle's own `float(sum(scores))`, and CPython compensates
/// float sums (3.12+), so only the interpreter's expression reproduces its totals
/// bit for bit. The `u32` word itself is staged caller-side from
/// `int(digest_hex[:8], 16)` (the digest authority stays Python). Zero Python
/// API: runs detached.
fn nibble_scores_from_word(word: u32) -> [f64; 4] {
    let mut scores = [0.0f64; ORACLE_VALUE_DIM];
    for (i, slot) in scores.iter_mut().enumerate() {
        // proof: `i` in 0..4 (nibble lane), so `i*4` in 0..12, fits `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let shift: u32 = (i * 4) as u32;
        *slot = f64::from((word >> shift) & 0xF) / 15.0;
    }
    scores
}

/// Pure synthetic-belief core (`oracle_targets.py:86-93`): a 32-byte sha256 digest
/// never reaches the `len(h) >= 34` branch, so the oracle always folds
/// `(h * 3)[:34]` — lane `i` of the 34-lane raw vector is digest byte `i % 32`
/// plus `1.0`, normalized with the 34-wide total-or-`1.0` rule (the total is at
/// least `34.0`, so the fallback never fires here, exactly like the oracle).
/// Zero Python API: runs detached.
fn belief_from_digest(digest: &[u8; 32]) -> [f64; 34] {
    let mut raw = [0.0f64; ORACLE_BELIEF_DIM];
    for (i, slot) in raw.iter_mut().enumerate() {
        *slot = f64::from(digest[i % 32]) + 1.0;
    }
    // Byte lanes widen exactly (`u8` -> `f64`) and the partial sums stay far
    // below 2^53, so this sequential fold matches the oracle's `sum(raw)` bit
    // for bit (unlike caller-typed lanes, which sum Python-side).
    let mut total = 0.0f64;
    for v in &raw {
        total += *v;
    }
    normalize_34(&raw, total)
}

/// Stage one legacy single rank: `bool` rejected (it subclasses `int`), plain
/// `int` in `0..=3` only (mirrors the `_value_target_from_privileged` legacy gate
/// at `oracle_targets.py:247`; the shape dispatch itself stays Python).
fn stage_rank_index(obj: &Bound<'_, PyAny>) -> Result<u8, String> {
    if obj.is_instance_of::<PyBool>() {
        return Err("oracle_targets value rank must be a plain int 0..=3, not bool".to_string());
    }
    if !obj.is_instance_of::<PyInt>() {
        return Err("oracle_targets value rank must be a plain int 0..=3".to_string());
    }
    let rank: i64 = obj
        .extract()
        .map_err(|_| "oracle_targets value rank must be a plain int 0..=3".to_string())?;
    if !(0..=3).contains(&rank) {
        return Err(format!("oracle_targets value rank={rank} outside [0, 3]"));
    }
    // proof: `rank` in 0..=3 (range-checked above), fits `u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let b: u8 = rank as u8;
    Ok(b)
}

/// Stage one synthetic hash word: `bool` rejected, plain `int` in
/// `0..=u32::MAX` only (the translator always passes
/// `int(digest_hex[:8], 16)`, so this gate is defensive-only — fail-closed,
/// never a default).
fn stage_hash_word(obj: &Bound<'_, PyAny>) -> Result<u32, String> {
    if obj.is_instance_of::<PyBool>() {
        return Err(
            "oracle_targets synthetic word must be a plain int 0..=4294967295, not bool"
                .to_string(),
        );
    }
    if !obj.is_instance_of::<PyInt>() {
        return Err("oracle_targets synthetic word must be a plain int 0..=4294967295".to_string());
    }
    let word: i64 = obj.extract().map_err(|_| {
        "oracle_targets synthetic word must be a plain int 0..=4294967295".to_string()
    })?;
    if !(0..=i64::from(u32::MAX)).contains(&word) {
        return Err(format!(
            "oracle_targets synthetic word={word} outside [0, 4294967295]"
        ));
    }
    // proof: `word` in 0..=u32::MAX (range-checked above), fits `u32`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let w: u32 = word as u32;
    Ok(w)
}

/// 34-dim belief normalize over caller-staged floats and the caller-computed
/// `float(sum(source))` total (mirrors the hidden-count / wait-histogram tails of
/// `_belief_target_from_privileged` without the privileged-dict dispatch, which
/// stays Python).
///
/// Staging is attached (the `Vec<f64>` extraction touches Python memory); the
/// rule + division run under ONE `py.detach` with zero Python API inside; the
/// result wraps attached. Fail-closed: anything but exactly 34 lanes is
/// `ValueError`, never a default (unreachable via the shape-gated Python
/// translators — defensive only). Any finite-or-not `total` maps (see
/// [`normalize_34`] for the `0.0`/`NaN` rule).
#[pyfunction]
fn oracle_normalize_34(py: Python<'_>, values: Vec<f64>, total: f64) -> PyResult<Vec<f64>> {
    if values.len() != ORACLE_BELIEF_DIM {
        return Err(PyValueError::new_err(format!(
            "oracle_targets normalize_34 needs exactly {ORACLE_BELIEF_DIM} floats"
        )));
    }
    let mut staged = [0.0f64; ORACLE_BELIEF_DIM];
    staged.copy_from_slice(&values);
    Ok(py.detach(|| normalize_34(&staged, total)).to_vec())
}

/// Teacher-logit map over caller-staged probabilities (mirrors
/// `_teacher_logits_from_targets` lane-for-lane without the tuple packing, which
/// stays Python). Total function: every `f64` maps (see [`teacher_logit`] for the
/// NaN mirror); compute runs detached; never an error.
#[pyfunction]
fn oracle_teacher_logits_for(py: Python<'_>, probs: Vec<f64>) -> Vec<f64> {
    py.detach(|| {
        let mut out = Vec::with_capacity(probs.len());
        for prob in probs.iter() {
            out.push(teacher_logit(*prob));
        }
        out
    })
}

/// Legacy single-rank value vector over a staged rank (mirrors the
/// `_value_target_from_privileged` single-`rank` branch at
/// `oracle_targets.py:247-251` without the dict dispatch, which stays Python).
///
/// Same stage-attached / detach / wrap-attached shape as
/// [`oracle_normalize_34`]. Fail-closed: every shape violation is `ValueError`,
/// never a default.
#[pyfunction]
fn oracle_value_from_rank(py: Python<'_>, rank: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let staged = stage_rank_index(&rank).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| value_at_rank(staged)).to_vec())
}

/// Hash-fallback synthetic nibble lanes over a staged hash word (mirrors the
/// `_value_target_from_privileged` synthetic tail at `oracle_targets.py:261-266`
/// without the digest call or the sum, which stay Python — the caller sums the
/// returned lanes with the oracle's own `float(sum(scores))`).
///
/// Same stage-attached / detach / wrap-attached shape as
/// [`oracle_normalize_34`]. Fail-closed: every shape violation is `ValueError`,
/// never a default.
#[pyfunction]
fn oracle_nibble_scores_from_word(py: Python<'_>, word: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let staged = stage_hash_word(&word).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| nibble_scores_from_word(staged)).to_vec())
}

/// Hash-fallback synthetic belief vector over staged digest bytes (mirrors the
/// `_belief_target_from_privileged` synthetic tail at `oracle_targets.py:86-93`
/// without the digest call, which stays Python).
///
/// Same stage-attached / detach / wrap-attached shape as
/// [`oracle_normalize_34`]. Fail-closed: anything but exactly 32 sha256 bytes is
/// `ValueError`, never a default (unreachable via the translators — defensive only).
#[pyfunction]
fn oracle_belief_from_digest(py: Python<'_>, digest: Vec<u8>) -> PyResult<Vec<f64>> {
    if digest.len() != 32 {
        return Err(PyValueError::new_err(
            "oracle_targets belief digest needs exactly 32 sha256 bytes",
        ));
    }
    let mut staged = [0u8; 32];
    staged.copy_from_slice(&digest);
    Ok(py.detach(|| belief_from_digest(&staged)).to_vec())
}

/// Attach the oracle-target consts + pyfns to the caller-provided `contracts`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `utility.rs:196-201`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("ORACLE_BELIEF_DIM", ORACLE_BELIEF_DIM)?;
    sub.add("ORACLE_VALUE_DIM", ORACLE_VALUE_DIM)?;
    sub.add("ORACLE_EVENT_KIND_COUNT", ORACLE_EVENT_KIND_COUNT)?;
    sub.add("ORACLE_TEACHER_EPS", ORACLE_TEACHER_EPS)?;
    sub.add("ORACLE_RANK_VALUES", PyTuple::new(py, ORACLE_RANK_VALUES)?)?;
    sub.add("ORACLE_VALUE_MIN", ORACLE_VALUE_MIN)?;
    sub.add("ORACLE_VALUE_MAX", ORACLE_VALUE_MAX)?;
    sub.add("ORACLE_ZERO_SUM_ABS_TOL", ORACLE_ZERO_SUM_ABS_TOL)?;
    sub.add(
        "ORACLE_SCORE_FOR_RANK",
        PyTuple::new(py, ORACLE_SCORE_FOR_RANK)?,
    )?;
    sub.add("ORACLE_SCORE_BASE", ORACLE_SCORE_BASE)?;
    sub.add("ORACLE_UTILITY_ID", ORACLE_UTILITY_ID)?;
    sub.add(
        "ORACLE_UTILITY_SCHEMA_VERSION",
        ORACLE_UTILITY_SCHEMA_VERSION,
    )?;
    sub.add("ORACLE_RULES_HASH", ORACLE_RULES_HASH)?;
    sub.add_function(wrap_pyfunction!(oracle_normalize_34, sub)?)?;
    sub.add_function(wrap_pyfunction!(oracle_teacher_logits_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(oracle_value_from_rank, sub)?)?;
    sub.add_function(wrap_pyfunction!(oracle_nibble_scores_from_word, sub)?)?;
    sub.add_function(wrap_pyfunction!(oracle_belief_from_digest, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Byte-level pins of the oracle literals (`oracle_targets.py:27-28,32,35,
        // 106-115,149,151,233,278`); any drift fails here first.
        assert_eq!(ORACLE_BELIEF_DIM, 34);
        assert_eq!(ORACLE_VALUE_DIM, 4);
        assert_eq!(ORACLE_EVENT_KIND_COUNT, 20);
        assert_eq!(ORACLE_TEACHER_EPS, 1e-6);
        assert_eq!(ORACLE_RANK_VALUES, [20.0, 10.0, -10.0, -20.0]);
        assert_eq!(ORACLE_VALUE_MIN, -100.0);
        assert_eq!(ORACLE_VALUE_MAX, 100.0);
        assert_eq!(ORACLE_ZERO_SUM_ABS_TOL, 1e-9);
        assert_eq!(ORACLE_SCORE_FOR_RANK, [40_000, 30_000, 20_000, 10_000]);
        assert_eq!(ORACLE_SCORE_BASE, 25_000);
        assert_eq!(
            ORACLE_UTILITY_ID,
            "expected_final_placement_tenhou_4p_hanchan_v1"
        );
        assert_eq!(ORACLE_UTILITY_SCHEMA_VERSION, "1.0.0");
        assert_eq!(
            ORACLE_RULES_HASH,
            "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
        );
    }

    #[test]
    fn normalize_34_matches_oracle_fold() {
        // Hand-derived from the oracle fold (`oracle_targets.py:65-67`): the
        // caller passes `float(sum(source))`; the core owns the total-or-1.0
        // rule plus lane-wise division.
        let uniform = normalize_34(&[1.0f64; 34], 34.0);
        for &v in &uniform {
            assert_eq!(v, 1.0f64 / 34.0);
        }
        let total: f64 = uniform.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Zero total fires the 1.0 fallback (oracle `else 1.0` branch).
        assert_eq!(normalize_34(&[0.0f64; 34], 0.0), [0.0f64; 34]);
        assert_eq!(normalize_34(&[1.0f64; 34], 0.0), [1.0f64; 34]);
        assert_eq!(normalize_34(&[1.0f64; 34], -0.0), [1.0f64; 34]);
        let mut hot = [0.0f64; 34];
        hot[0] = 34.0;
        let normed = normalize_34(&hot, 34.0);
        assert_eq!(normed[0], 1.0);
        for &v in normed.iter().skip(1) {
            assert_eq!(v, 0.0);
        }
        // NaN totals propagate lane-wise (oracle `nan != 0.0` keeps NaN).
        let nan_lanes = normalize_34(&[1.0f64; 34], f64::NAN);
        for &v in &nan_lanes {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn teacher_logit_matches_oracle_max_log() {
        // Hand-derived (`oracle_targets.py:278-280`): ln(max(p, 1e-6)) with the
        // oracle's NaN order (NaN stays NaN); platform libm, same box.
        assert_eq!(teacher_logit(1.0), 0.0);
        assert_eq!(teacher_logit(0.0), 1e-6f64.ln());
        assert_eq!(teacher_logit(-5.0), 1e-6f64.ln());
        assert_eq!(teacher_logit(20.0), 20.0f64.ln());
        assert_eq!(teacher_logit(f64::INFINITY), f64::INFINITY);
        assert_eq!(teacher_logit(f64::NEG_INFINITY), 1e-6f64.ln());
        assert!(teacher_logit(f64::NAN).is_nan());
    }

    #[test]
    fn value_at_rank_matches_oracle_single_shape() {
        // Hand-derived (`oracle_targets.py:247-251`): rank_values[rank] at the
        // rank index, zeros elsewhere (utility scale, never one-hot).
        assert_eq!(value_at_rank(0), [20.0, 0.0, 0.0, 0.0]);
        assert_eq!(value_at_rank(1), [0.0, 10.0, 0.0, 0.0]);
        assert_eq!(value_at_rank(2), [0.0, 0.0, -10.0, 0.0]);
        assert_eq!(value_at_rank(3), [0.0, 0.0, 0.0, -20.0]);
    }

    #[test]
    fn nibble_scores_match_oracle_lanes() {
        // Hand-derived (`oracle_targets.py:266`): low-nibble `/15.0` lanes over
        // the staged word. Per-lane deterministic division only (no summation),
        // so every vector below is bit-exact.
        assert_eq!(nibble_scores_from_word(0), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(nibble_scores_from_word(0xFFFF_FFFF), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            nibble_scores_from_word(0x0000_0001),
            [1.0f64 / 15.0, 0.0, 0.0, 0.0]
        );
        // Word 0x4321 carries nibbles 1, 2, 3, 4 from lane 0 up.
        assert_eq!(
            nibble_scores_from_word(0x0000_4321),
            [1.0f64 / 15.0, 2.0f64 / 15.0, 3.0f64 / 15.0, 4.0f64 / 15.0]
        );
    }

    #[test]
    fn belief_from_digest_matches_oracle_repeat_fold() {
        // Hand-derived (`oracle_targets.py:86-93`): 32 digest bytes always take
        // the `(h * 3)[:34]` branch — lane i is byte (i % 32) plus 1.0.
        let zeros = belief_from_digest(&[0u8; 32]);
        for &v in &zeros {
            assert_eq!(v, 1.0f64 / 34.0);
        }
        let mut seq = [0u8; 32];
        for (i, slot) in seq.iter_mut().enumerate() {
            // proof: test index `i` in 0..32, fits `u8`.
            #[allow(clippy::cast_possible_truncation)]
            let b: u8 = i as u8;
            *slot = b;
        }
        // raw = 1..=32 then 1, 2; total = 528 + 3 = 531.
        let out = belief_from_digest(&seq);
        assert_eq!(out.len(), ORACLE_BELIEF_DIM);
        assert_eq!(out[0], 1.0f64 / 531.0);
        assert_eq!(out[31], 32.0f64 / 531.0);
        assert_eq!(out[32], 1.0f64 / 531.0);
        assert_eq!(out[33], 2.0f64 / 531.0);
        let total: f64 = out.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
