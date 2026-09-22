//! local_spec: frozen Candidate 5 resolving-config vocabularies + validator.
//!
//! DAG: pyo3 ONLY (frozen literals + pure argument validation; no feed/shard/
//! search-crate edge — the update/averaging math lives `hydra-search::local`
//! + `local_driver` as string match arms, the canon bytes live `feed::canon`,
//!   and the file-backed hashes / torch model probes stay Python-side in
//!   `local_spec.py`, see `search.rs` canon-wins B3 docs). FORBIDS: selection
//!   math, digest computation, RNG streams, belief sampling, file/JSON IO,
//!   torch/CUDA, wall-clock budgets, and `ContractError` shaping (the bridge
//!   raises `ValueError`; thin Python translators map to `ContractError` with
//!   byte-identical messages).
//!
//! TABLE ported here — the frozen vocabularies + `__post_init__` gate, oracle
//! `python/hydra2/search/local_spec.py:42-87` (HEAD `src/hydra2/search/
//! local_spec.py:42-87`, equals worktree at port time):
//! - `LOCAL_SPEC_VALID_*` ← the six frozen vocabularies (`_VALID_UPDATE_RULES`
//!   / `_VALID_AVERAGING` via `local_strategy.py:50-51`, plus the four inline
//!   tuples in `__post_init__`)
//! - `LOCAL_SPEC_DEFAULT_*` ← the dataclass field defaults (`:46-54`)
//! - `LOCAL_SPEC_{CANDIDATE_ID,ALGORITHM,ALGORITHM_VERSION,UTILITY_ID,
//!   FALLBACK_CANDIDATE_ID}` ← the frozen factory identity strings
//!   (`make_candidate5_spec`, `:436-454`)
//! - `local_spec_check_config` ← `LocalResolvingConfig.__post_init__`
//!   (`:56-87`), check order verbatim.
//!
//! LOGIC staying Python: the `LocalResolvingConfig` dataclass (slots/frozen,
//! `to_parameters`/`from_parameters`), `_build_abstraction_from_config`
//! (descriptor-to-mapping translation), `_file_sha256` / `_load_default_hashes`
//! (file-backed config hashes), `_model_hash_from_identity` /
//! `_derive_utility_manifest_hash` (lazy torch-backed model probes with
//! fail-closed fallback), and `make_candidate5_spec` orchestration (hash
//! binding, budget assembly, `CandidateSpec` construction).
//!
//! NONE-owner note: `rg` for `regret_matching|fictitious_play|pair_merge|
//! temperature_0\.5|value_break` over `crates/feed/src` + `crates/search/src`
//! + `crates/shard/src` hits only `search/src/local.rs` + `local_driver.rs`
//!   string match arms (`"regret_matching" => …`, `batch.update_rule != …`)
//!   with generic `"unknown update_rule"` errors — no exported frozen vocabulary
//!   const exists there; `rg` for `LOCAL_SPEC_|local_spec_check|VALID_UPDATE_`
//!   over `crates/bridge/src` hits nothing before this file. These consts are
//!   the first frozen vocabulary surface, not a duplication.
//!
//! Shape per fn: attached staging (exact `repr(value)` text via the live
//! Python API, so `{value!r}` renders byte-exact per `rc_require.rs:63-65`;
//! bool rejection per `contracts.rs:69-74` + `canon_rng.rs:279`) → ONE
//! `py.detach(|| …)` over owned plain data with zero Python API inside (per
//! `contracts.rs:434-437`, `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`).
//!
//! Single-cdylib tree: registers its consts + pyfn on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_profiles::register`, `search_profiles.rs:364-377`); no new entry
//! point. MAIN wiring: `pub mod local_spec;` in `lib.rs` plus
//! `crate::local_spec::register(&sub)?;` in `search.rs` next to
//! `crate::search_profiles::register(&sub)?;` (`search.rs:1730`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFrozenSet, PyModule};

/// Frozen update-rule vocabulary (`local_strategy.py:50`, via `local_spec.py:70`).
const VALID_UPDATE_RULES: [&str; 3] = ["regret_matching", "hedge", "fictitious_play"];
/// Frozen averaging vocabulary (`local_strategy.py:51`, via `local_spec.py:74`).
const VALID_AVERAGING: [&str; 2] = ["uniform", "linear"];
/// Frozen abstraction-descriptor vocabulary (`local_spec.py:78`).
const VALID_ABSTRACTIONS: [&str; 4] = ["identity", "pair_merge", "tile_type", "custom"];
/// Frozen leaf-model vocabulary (`local_spec.py:82`).
const VALID_LEAF_MODELS: [&str; 2] = ["model", "terminal"];
/// Frozen tie-break vocabulary (`local_spec.py:84`).
const VALID_TIE_BREAKS: [&str; 4] = [
    "greedy",
    "temperature_0.5",
    "temperature_1.0",
    "value_break",
];
/// Frozen resource-view vocabulary (`local_spec.py:86`).
const VALID_RESOURCE_VIEWS: [&str; 3] = ["calls", "transitions", "joules"];

/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_HORIZON: u64 = 2;
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_ITERATIONS: u64 = 16;
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_UPDATE_RULE: &str = "regret_matching";
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_AVERAGING: &str = "uniform";
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_ABSTRACTION: &str = "identity";
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_LEAF_MODEL: &str = "model";
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_TIE_BREAK: &str = "greedy";
/// Frozen dataclass defaults (`local_spec.py:46-54`).
const DEFAULT_RESOURCE_VIEW: &str = "calls";

/// Frozen factory identity (`make_candidate5_spec`, `local_spec.py:436-454`).
const CANDIDATE_ID: &str = "candidate5";
/// Frozen factory identity (`make_candidate5_spec`, `local_spec.py:436-454`).
const ALGORITHM: &str = "local_resolving";
/// Frozen factory identity (`make_candidate5_spec`, `local_spec.py:436-454`).
const ALGORITHM_VERSION: &str = "1.0.0";
/// Frozen factory identity (`make_candidate5_spec`, `local_spec.py:441`).
const UTILITY_ID: &str = "expected_final_placement_tenhou_4p_hanchan_v1";
/// Frozen factory identity (`make_candidate5_spec`, `local_spec.py:453`).
const FALLBACK_CANDIDATE_ID: &str = "candidate0";

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Owned plain-data view of an int-or-other Python value. Bool is excluded
/// first (it subclasses `int`), exactly like `contracts.rs:69-74`; negatives
/// and oversized ints keep their staged repr and read as absent, so the range
/// arms below reject with the oracle text.
struct StagedInt {
    repr: String,
    value: Option<u64>,
}

fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<StagedInt> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() {
        return Ok(StagedInt { repr, value: None });
    }
    let value = obj.extract::<u64>().ok();
    Ok(StagedInt { repr, value })
}

/// Owned plain-data view of a str-or-other Python value. Non-`str` inputs
/// keep their staged repr and read as absent, matching the oracle's
/// `x not in (…)` reject with `{x!r}`.
struct StagedStr {
    repr: String,
    value: Option<String>,
}

fn stage_str(obj: &Bound<'_, PyAny>) -> PyResult<StagedStr> {
    let repr = py_repr(obj)?;
    let value: Option<String> = obj.extract().ok();
    Ok(StagedStr { repr, value })
}

/// Pure `__post_init__` mirror (`local_spec.py:56-87`), check order verbatim.
/// Operates on staged owned data only — no Python API inside, detach-safe.
#[allow(clippy::too_many_arguments)]
fn check_staged(
    horizon: &StagedInt,
    iterations: &StagedInt,
    update_rule: &StagedStr,
    averaging: &StagedStr,
    abstraction: &StagedStr,
    leaf_model: &StagedStr,
    tie_break: &StagedStr,
    resource_view: &StagedStr,
) -> Result<(), String> {
    if horizon.value.is_none_or(|v| !(1..=16).contains(&v)) {
        return Err(format!(
            "horizon must be int in 1..16, got {}",
            horizon.repr
        ));
    }
    if iterations.value.is_none_or(|v| v == 0 || v > 1024) {
        return Err(format!(
            "iterations must be positive int <=1024, got {}",
            iterations.repr
        ));
    }
    if update_rule
        .value
        .as_deref()
        .is_none_or(|v| !VALID_UPDATE_RULES.contains(&v))
    {
        return Err(format!(
            "update_rule must be one of ['fictitious_play', 'hedge', 'regret_matching'], got {}",
            update_rule.repr
        ));
    }
    if averaging
        .value
        .as_deref()
        .is_none_or(|v| !VALID_AVERAGING.contains(&v))
    {
        return Err(format!(
            "averaging must be one of ['linear', 'uniform'], got {}",
            averaging.repr
        ));
    }
    if abstraction
        .value
        .as_deref()
        .is_none_or(|v| !VALID_ABSTRACTIONS.contains(&v))
    {
        return Err(format!(
            "abstraction must be identity/pair_merge/tile_type/custom, got {}",
            abstraction.repr
        ));
    }
    if leaf_model
        .value
        .as_deref()
        .is_none_or(|v| !VALID_LEAF_MODELS.contains(&v))
    {
        return Err(format!(
            "leaf_model must be model or terminal, got {}",
            leaf_model.repr
        ));
    }
    if tie_break
        .value
        .as_deref()
        .is_none_or(|v| !VALID_TIE_BREAKS.contains(&v))
    {
        return Err(format!("tie_break {} unknown", tie_break.repr));
    }
    if resource_view
        .value
        .as_deref()
        .is_none_or(|v| !VALID_RESOURCE_VIEWS.contains(&v))
    {
        return Err(format!("resource_view {} unknown", resource_view.repr));
    }
    Ok(())
}

/// Field validator mirroring `LocalResolvingConfig.__post_init__`
/// (`local_spec.py:56-87`): attached staging only (like
/// `profiles_check_profile`, `search_profiles.rs:168-183`); `ValueError` on
/// any reject, never a default. Translators call positionally; Python maps to
/// `ContractError` with byte-identical text.
#[pyfunction]
#[pyo3(signature = (horizon, iterations, update_rule, averaging, abstraction, leaf_model, tie_break, resource_view))]
#[allow(clippy::too_many_arguments)]
fn local_spec_check_config(
    py: Python<'_>,
    horizon: Bound<'_, PyAny>,
    iterations: Bound<'_, PyAny>,
    update_rule: Bound<'_, PyAny>,
    averaging: Bound<'_, PyAny>,
    abstraction: Bound<'_, PyAny>,
    leaf_model: Bound<'_, PyAny>,
    tie_break: Bound<'_, PyAny>,
    resource_view: Bound<'_, PyAny>,
) -> PyResult<()> {
    let staged_horizon = stage_int(&horizon)?;
    let staged_iterations = stage_int(&iterations)?;
    let staged_update_rule = stage_str(&update_rule)?;
    let staged_averaging = stage_str(&averaging)?;
    let staged_abstraction = stage_str(&abstraction)?;
    let staged_leaf_model = stage_str(&leaf_model)?;
    let staged_tie_break = stage_str(&tie_break)?;
    let staged_resource_view = stage_str(&resource_view)?;
    py.detach(|| {
        check_staged(
            &staged_horizon,
            &staged_iterations,
            &staged_update_rule,
            &staged_averaging,
            &staged_abstraction,
            &staged_leaf_model,
            &staged_tie_break,
            &staged_resource_view,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Register the spec vocabularies + validator on the EXISTING `search`
/// submodule (mirrors `search_profiles::register`,
/// `search_profiles.rs:364-377`, on the shared submodule; MAIN calls this
/// from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "LOCAL_SPEC_VALID_UPDATE_RULES",
        PyFrozenSet::new(py, VALID_UPDATE_RULES)?,
    )?;
    sub.add(
        "LOCAL_SPEC_VALID_AVERAGING",
        PyFrozenSet::new(py, VALID_AVERAGING)?,
    )?;
    sub.add(
        "LOCAL_SPEC_VALID_ABSTRACTIONS",
        PyFrozenSet::new(py, VALID_ABSTRACTIONS)?,
    )?;
    sub.add(
        "LOCAL_SPEC_VALID_LEAF_MODELS",
        PyFrozenSet::new(py, VALID_LEAF_MODELS)?,
    )?;
    sub.add(
        "LOCAL_SPEC_VALID_TIE_BREAKS",
        PyFrozenSet::new(py, VALID_TIE_BREAKS)?,
    )?;
    sub.add(
        "LOCAL_SPEC_VALID_RESOURCE_VIEWS",
        PyFrozenSet::new(py, VALID_RESOURCE_VIEWS)?,
    )?;
    sub.add("LOCAL_SPEC_DEFAULT_HORIZON", DEFAULT_HORIZON)?;
    sub.add("LOCAL_SPEC_DEFAULT_ITERATIONS", DEFAULT_ITERATIONS)?;
    sub.add("LOCAL_SPEC_DEFAULT_UPDATE_RULE", DEFAULT_UPDATE_RULE)?;
    sub.add("LOCAL_SPEC_DEFAULT_AVERAGING", DEFAULT_AVERAGING)?;
    sub.add("LOCAL_SPEC_DEFAULT_ABSTRACTION", DEFAULT_ABSTRACTION)?;
    sub.add("LOCAL_SPEC_DEFAULT_LEAF_MODEL", DEFAULT_LEAF_MODEL)?;
    sub.add("LOCAL_SPEC_DEFAULT_TIE_BREAK", DEFAULT_TIE_BREAK)?;
    sub.add("LOCAL_SPEC_DEFAULT_RESOURCE_VIEW", DEFAULT_RESOURCE_VIEW)?;
    sub.add("LOCAL_SPEC_CANDIDATE_ID", CANDIDATE_ID)?;
    sub.add("LOCAL_SPEC_ALGORITHM", ALGORITHM)?;
    sub.add("LOCAL_SPEC_ALGORITHM_VERSION", ALGORITHM_VERSION)?;
    sub.add("LOCAL_SPEC_UTILITY_ID", UTILITY_ID)?;
    sub.add("LOCAL_SPEC_FALLBACK_CANDIDATE_ID", FALLBACK_CANDIDATE_ID)?;
    sub.add_function(wrap_pyfunction!(local_spec_check_config, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod local_spec_tests {
    use super::*;

    fn staged_int(repr: &str, value: Option<u64>) -> StagedInt {
        StagedInt {
            repr: repr.to_owned(),
            value,
        }
    }

    fn staged_str(repr: &str, value: Option<&str>) -> StagedStr {
        StagedStr {
            repr: repr.to_owned(),
            value: value.map(str::to_owned),
        }
    }

    fn valid() -> (
        StagedInt,
        StagedInt,
        StagedStr,
        StagedStr,
        StagedStr,
        StagedStr,
        StagedStr,
        StagedStr,
    ) {
        (
            staged_int("2", Some(2)),
            staged_int("16", Some(16)),
            staged_str("'regret_matching'", Some("regret_matching")),
            staged_str("'uniform'", Some("uniform")),
            staged_str("'identity'", Some("identity")),
            staged_str("'model'", Some("model")),
            staged_str("'greedy'", Some("greedy")),
            staged_str("'calls'", Some("calls")),
        )
    }

    #[test]
    fn frozen_values_match_python_leaves() {
        assert_eq!(
            VALID_UPDATE_RULES,
            ["regret_matching", "hedge", "fictitious_play"]
        );
        assert_eq!(VALID_AVERAGING, ["uniform", "linear"]);
        assert_eq!(
            VALID_ABSTRACTIONS,
            ["identity", "pair_merge", "tile_type", "custom"]
        );
        assert_eq!(VALID_LEAF_MODELS, ["model", "terminal"]);
        assert_eq!(
            VALID_TIE_BREAKS,
            [
                "greedy",
                "temperature_0.5",
                "temperature_1.0",
                "value_break"
            ]
        );
        assert_eq!(VALID_RESOURCE_VIEWS, ["calls", "transitions", "joules"]);
        assert_eq!((DEFAULT_HORIZON, DEFAULT_ITERATIONS), (2, 16));
        assert_eq!(
            (
                DEFAULT_UPDATE_RULE,
                DEFAULT_AVERAGING,
                DEFAULT_ABSTRACTION,
                DEFAULT_LEAF_MODEL,
                DEFAULT_TIE_BREAK,
                DEFAULT_RESOURCE_VIEW,
            ),
            (
                "regret_matching",
                "uniform",
                "identity",
                "model",
                "greedy",
                "calls",
            )
        );
        assert_eq!(
            (
                CANDIDATE_ID,
                ALGORITHM,
                ALGORITHM_VERSION,
                UTILITY_ID,
                FALLBACK_CANDIDATE_ID,
            ),
            (
                "candidate5",
                "local_resolving",
                "1.0.0",
                "expected_final_placement_tenhou_4p_hanchan_v1",
                "candidate0",
            )
        );
    }

    #[test]
    fn defaults_accept() {
        let (h, it, ur, av, ab, lm, tb, rv) = valid();
        assert_eq!(check_staged(&h, &it, &ur, &av, &ab, &lm, &tb, &rv), Ok(()));
    }

    #[test]
    fn rejects_match_head_oracle_text() {
        // Hand-derived from `local_spec.py:56-87` (HEAD `src/...:56-87`).
        let (h, it, ur, av, ab, lm, tb, rv) = valid();
        assert_eq!(
            check_staged(&staged_int("0", Some(0)), &it, &ur, &av, &ab, &lm, &tb, &rv),
            Err("horizon must be int in 1..16, got 0".to_owned())
        );
        assert_eq!(
            check_staged(&h, &staged_int("0", Some(0)), &ur, &av, &ab, &lm, &tb, &rv),
            Err("iterations must be positive int <=1024, got 0".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &staged_str("'nope'", Some("nope")),
                &av,
                &ab,
                &lm,
                &tb,
                &rv
            ),
            Err("update_rule must be one of ['fictitious_play', 'hedge', 'regret_matching'], got 'nope'".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &ur,
                &staged_str("'exponential'", Some("exponential")),
                &ab,
                &lm,
                &tb,
                &rv
            ),
            Err("averaging must be one of ['linear', 'uniform'], got 'exponential'".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &ur,
                &av,
                &staged_str("'unknown_abstraction'", Some("unknown_abstraction")),
                &lm,
                &tb,
                &rv
            ),
            Err("abstraction must be identity/pair_merge/tile_type/custom, got 'unknown_abstraction'".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &ur,
                &av,
                &ab,
                &staged_str("'oracle'", Some("oracle")),
                &tb,
                &rv
            ),
            Err("leaf_model must be model or terminal, got 'oracle'".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &ur,
                &av,
                &ab,
                &lm,
                &staged_str("'random'", Some("random")),
                &rv
            ),
            Err("tie_break 'random' unknown".to_owned())
        );
        assert_eq!(
            check_staged(
                &h,
                &it,
                &ur,
                &av,
                &ab,
                &lm,
                &tb,
                &staged_str("'flops'", Some("flops"))
            ),
            Err("resource_view 'flops' unknown".to_owned())
        );
    }

    #[test]
    fn bool_and_non_int_render_repr() {
        // `True` stages as absent with repr `True` (bool subclasses int).
        let (h, it, ur, av, ab, lm, tb, rv) = valid();
        assert_eq!(
            check_staged(&staged_int("True", None), &it, &ur, &av, &ab, &lm, &tb, &rv),
            Err("horizon must be int in 1..16, got True".to_owned())
        );
        assert_eq!(
            check_staged(&h, &it, &ur, &av, &ab, &lm, &tb, &staged_str("None", None)),
            Err("resource_view None unknown".to_owned())
        );
    }

    #[test]
    fn check_order_is_horizon_first() {
        // Every lane invalid at once: the horizon arm wins (oracle order).
        assert_eq!(
            check_staged(
                &staged_int("99", Some(99)),
                &staged_int("0", Some(0)),
                &staged_str("'nope'", Some("nope")),
                &staged_str("'nope'", Some("nope")),
                &staged_str("'nope'", Some("nope")),
                &staged_str("'nope'", Some("nope")),
                &staged_str("'nope'", Some("nope")),
                &staged_str("'nope'", Some("nope")),
            ),
            Err("horizon must be int in 1..16, got 99".to_owned())
        );
    }
}
