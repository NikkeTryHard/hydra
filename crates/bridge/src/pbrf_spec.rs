//! pbrf_spec: frozen Candidate-3 PBRF spec-factory values + pure assembly helpers.
//!
//! DAG: pyo3 + `hydra-feed` (canon/digest owners) + `hydra-search` (`pbrf`
//! parent/batch owners) ONLY. The feed canon+digest edge mirrors
//! `eval_leaves::case_manifest_digest_value` (`eval_leaves.rs:416-421`); the
//! `hydra-search::pbrf` edge mirrors `search_shared`'s `hydra_search::gumbel`
//! edge (`search_shared.rs:69-74`). FORBIDS: file/JSON IO (the file-backed
//! config-hash loader stays Python in `pbrf_spec.py`), torch/model probes
//! (candidate0/model/utility digests stay Python behind the lazy-probe rule),
//! digest validation (`make_digest_text` in contracts owns `sha256:` shape),
//! and builder logic (`CandidateSpec`/`ResourceBudget`/`PbrfConfig`
//! construction and validation stay Python in `search/common.py` and
//! `pbrf_partition.py`).
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_shared::register`, `search_shared.rs:59-89`, and
//! `search_profiles::register`, `search_profiles.rs:364-377`); MAIN calls this
//! from `search::register`. No new entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Frozen kernel tolerance (`pbrf_spec.py:140` default, `pbrf_partition.py:177`
/// `PbrfConfig.kernel_tolerance` default): small positive float in (0, 0.01).
const PBRF_SPEC_DEFAULT_KERNEL_TOLERANCE: f64 = 1e-9;
/// Frozen resource view (`pbrf_spec.py:142`, `pbrf_partition.py:179`).
const PBRF_SPEC_DEFAULT_RESOURCE_VIEW: &str = "calls";
/// Frozen candidate id (`pbrf_spec.py:143`).
const PBRF_SPEC_DEFAULT_CANDIDATE_ID: &str = "candidate3_pbrf_core_v1";
/// Frozen utility id (`pbrf_spec.py:146`).
const PBRF_SPEC_DEFAULT_UTILITY_ID: &str = "expected_final_placement";
/// Frozen tie-break (`pbrf_spec.py:147`, `pbrf_partition.py:180`).
const PBRF_SPEC_DEFAULT_TIE_BREAK: &str = "lexicographic";
/// Frozen algorithm tag (`pbrf_spec.py:230`).
const PBRF_SPEC_ALGORITHM: &str = "pbrf_core";
/// Frozen algorithm version (`pbrf_spec.py:231`).
const PBRF_SPEC_ALGORITHM_VERSION: &str = "1.0.0";
/// Frozen fallback candidate (`pbrf_spec.py:245`).
const PBRF_SPEC_FALLBACK_CANDIDATE_ID: &str = "candidate0";
/// Frozen default budget mode (`pbrf_spec.py:208`).
const PBRF_SPEC_DEFAULT_BUDGET_MODE: &str = "gameplay_5s";
/// Frozen default budget deadline (`pbrf_spec.py:209`).
const PBRF_SPEC_DEFAULT_BUDGET_DEADLINE_MS: u32 = 5_000;
/// Frozen default budget fallback margin (`pbrf_spec.py:210`).
const PBRF_SPEC_DEFAULT_BUDGET_FALLBACK_MARGIN_MS: u32 = 200;
/// Frozen default budget model-call cap (`pbrf_spec.py:211`).
const PBRF_SPEC_DEFAULT_BUDGET_MAX_MODEL_CALLS: u32 = 64;
/// Frozen default budget transition cap (`pbrf_spec.py:212`).
const PBRF_SPEC_DEFAULT_BUDGET_MAX_TRANSITIONS: u32 = 256;
// NOTE: `max_particles` is not a const: it tracks the caller's `parent_count`
// (`pbrf_spec.py:213`); `max_memory_bytes` is always `None` there.
// NOTE: `parent_count`/`max_search_batches` add no consts here: the single
// source is already Rust-owned (`hydra_search::pbrf::PBRF_PARENTS` /
// `PBRF_MAX_BATCHES`, `pbrf.rs:21-24`), referenced in `register` below.

/// Frozen rules-config relpath (`pbrf_spec.py:67`).
const PBRF_SPEC_RULES_CONFIG_REL: &str = "configs/rules/tenhou_4p_hanchan_v1.json";
/// Frozen action-table relpath (`pbrf_spec.py:68`).
const PBRF_SPEC_ACTION_TABLE_CONFIG_REL: &str = "configs/contracts/action_table_v1.json";
/// Frozen observation-schema relpath (`pbrf_spec.py:69`).
const PBRF_SPEC_OBSERVATION_SCHEMA_CONFIG_REL: &str = "configs/models/model_input_v1.json";
/// Frozen packet-boundary relpath (`pbrf_spec.py:70`).
const PBRF_SPEC_PACKET_BOUNDARY_CONFIG_REL: &str = "configs/contracts/packet_boundary_v1.json";
/// Frozen observation-schema contract override relpath (`pbrf_spec.py:83`).
const PBRF_SPEC_OBSERVATION_SCHEMA_CONTRACT_REL: &str =
    "configs/contracts/observation_schema_v1.json";

/// Canonical rng/stream/case digests (`pbrf_spec.py:122-134` `_canonical_hashes`):
/// JCS-emit + `sha256:`-seal through the feed owners (never reimplemented here).
/// Returns `(rng_protocol_hash, random_stream_schema_hash, case_manifest_hash)`
/// in the oracle dict order so the translator rebuilds the dict positionally.
fn canonical_hashes() -> Result<(String, String, String), String> {
    let rng = hydra_feed::digest::of_canonical(&serde_json::json!({
        "protocol": "counter_based_v1",
        "version": "1.0.0",
    }))
    .map_err(|e| format!("pbrf_spec canonical seal rejected: {e:?}"))?;
    let stream = hydra_feed::digest::of_canonical(&serde_json::json!({
        "schema": "random_stream_v1",
        "purposes": ["candidate0_tie"],
    }))
    .map_err(|e| format!("pbrf_spec canonical seal rejected: {e:?}"))?;
    let case = hydra_feed::digest::of_canonical(&serde_json::json!([]))
        .map_err(|e| format!("pbrf_spec canonical seal rejected: {e:?}"))?;
    Ok((rng, stream, case))
}

/// Override-or-default pick (`pbrf_spec.py:167-197`): a non-empty override wins,
/// `None`/empty reads as the bound default. Digest-shape validation stays
/// caller-side in `make_digest_text` (contracts owns digests).
fn resolve_hash(override_hash: Option<&str>, default_hash: &str) -> String {
    match override_hash {
        Some(text) if !text.is_empty() => text.to_owned(),
        _ => default_hash.to_owned(),
    }
}

/// Canonical triple over the frozen descriptors: attached staging is vacuous
/// (inputs are frozen), detached compute runs ONE `py.detach` with zero Python
/// API inside (`search.rs:1061` shape); `ValueError` on seal reject, never a
/// default.
#[pyfunction]
fn pbrf_spec_canonical_hashes(py: Python<'_>) -> PyResult<(String, String, String)> {
    py.detach(canonical_hashes).map_err(PyValueError::new_err)
}

/// Override-or-default pick over caller-resolved hashes: translators call
/// positionally; a non-`str` override fails at extraction with `TypeError`
/// (bridge raises; Python maps to `ContractError`).
#[pyfunction]
#[pyo3(signature = (override_hash, default_hash))]
fn pbrf_spec_resolve_hash(
    py: Python<'_>,
    override_hash: Option<String>,
    default_hash: String,
) -> PyResult<String> {
    Ok(py.detach(|| resolve_hash(override_hash.as_deref(), &default_hash)))
}

/// Register the PBRF spec-factory consts + helpers on the EXISTING `search`
/// submodule (mirrors `search_shared::register` on the shared submodule; MAIN
/// calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add(
        "PBRF_SPEC_DEFAULT_PARENT_COUNT",
        hydra_search::pbrf::PBRF_PARENTS,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_MAX_SEARCH_BATCHES",
        hydra_search::pbrf::PBRF_MAX_BATCHES,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_KERNEL_TOLERANCE",
        PBRF_SPEC_DEFAULT_KERNEL_TOLERANCE,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_RESOURCE_VIEW",
        PBRF_SPEC_DEFAULT_RESOURCE_VIEW,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_CANDIDATE_ID",
        PBRF_SPEC_DEFAULT_CANDIDATE_ID,
    )?;
    sub.add("PBRF_SPEC_DEFAULT_UTILITY_ID", PBRF_SPEC_DEFAULT_UTILITY_ID)?;
    sub.add("PBRF_SPEC_DEFAULT_TIE_BREAK", PBRF_SPEC_DEFAULT_TIE_BREAK)?;
    sub.add("PBRF_SPEC_ALGORITHM", PBRF_SPEC_ALGORITHM)?;
    sub.add("PBRF_SPEC_ALGORITHM_VERSION", PBRF_SPEC_ALGORITHM_VERSION)?;
    sub.add(
        "PBRF_SPEC_FALLBACK_CANDIDATE_ID",
        PBRF_SPEC_FALLBACK_CANDIDATE_ID,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_BUDGET_MODE",
        PBRF_SPEC_DEFAULT_BUDGET_MODE,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_BUDGET_DEADLINE_MS",
        PBRF_SPEC_DEFAULT_BUDGET_DEADLINE_MS,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_BUDGET_FALLBACK_MARGIN_MS",
        PBRF_SPEC_DEFAULT_BUDGET_FALLBACK_MARGIN_MS,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_BUDGET_MAX_MODEL_CALLS",
        PBRF_SPEC_DEFAULT_BUDGET_MAX_MODEL_CALLS,
    )?;
    sub.add(
        "PBRF_SPEC_DEFAULT_BUDGET_MAX_TRANSITIONS",
        PBRF_SPEC_DEFAULT_BUDGET_MAX_TRANSITIONS,
    )?;
    sub.add("PBRF_SPEC_RULES_CONFIG_REL", PBRF_SPEC_RULES_CONFIG_REL)?;
    sub.add(
        "PBRF_SPEC_ACTION_TABLE_CONFIG_REL",
        PBRF_SPEC_ACTION_TABLE_CONFIG_REL,
    )?;
    sub.add(
        "PBRF_SPEC_OBSERVATION_SCHEMA_CONFIG_REL",
        PBRF_SPEC_OBSERVATION_SCHEMA_CONFIG_REL,
    )?;
    sub.add(
        "PBRF_SPEC_PACKET_BOUNDARY_CONFIG_REL",
        PBRF_SPEC_PACKET_BOUNDARY_CONFIG_REL,
    )?;
    sub.add(
        "PBRF_SPEC_OBSERVATION_SCHEMA_CONTRACT_REL",
        PBRF_SPEC_OBSERVATION_SCHEMA_CONTRACT_REL,
    )?;
    sub.add_function(wrap_pyfunction!(pbrf_spec_canonical_hashes, sub)?)?;
    sub.add_function(wrap_pyfunction!(pbrf_spec_resolve_hash, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod pbrf_spec_tests {
    use super::*;

    #[test]
    fn frozen_values_match_python_factory() {
        // Parent/batch single-source lives in `hydra-search` (referenced, not
        // copied); the rest pins the `pbrf_spec.py` / `PbrfConfig` literals.
        assert_eq!(hydra_search::pbrf::PBRF_PARENTS, 16);
        assert_eq!(hydra_search::pbrf::PBRF_MAX_BATCHES, 64);
        assert_eq!(PBRF_SPEC_DEFAULT_KERNEL_TOLERANCE, 1e-9);
        assert_eq!(
            (
                PBRF_SPEC_DEFAULT_RESOURCE_VIEW,
                PBRF_SPEC_DEFAULT_CANDIDATE_ID,
                PBRF_SPEC_DEFAULT_UTILITY_ID,
                PBRF_SPEC_DEFAULT_TIE_BREAK,
            ),
            (
                "calls",
                "candidate3_pbrf_core_v1",
                "expected_final_placement",
                "lexicographic",
            )
        );
        assert_eq!(
            (
                PBRF_SPEC_ALGORITHM,
                PBRF_SPEC_ALGORITHM_VERSION,
                PBRF_SPEC_FALLBACK_CANDIDATE_ID,
            ),
            ("pbrf_core", "1.0.0", "candidate0")
        );
        assert_eq!(
            (
                PBRF_SPEC_DEFAULT_BUDGET_MODE,
                PBRF_SPEC_DEFAULT_BUDGET_DEADLINE_MS,
                PBRF_SPEC_DEFAULT_BUDGET_FALLBACK_MARGIN_MS,
                PBRF_SPEC_DEFAULT_BUDGET_MAX_MODEL_CALLS,
                PBRF_SPEC_DEFAULT_BUDGET_MAX_TRANSITIONS,
            ),
            ("gameplay_5s", 5_000, 200, 64, 256)
        );
        assert_eq!(
            (
                PBRF_SPEC_RULES_CONFIG_REL,
                PBRF_SPEC_ACTION_TABLE_CONFIG_REL,
                PBRF_SPEC_OBSERVATION_SCHEMA_CONFIG_REL,
                PBRF_SPEC_PACKET_BOUNDARY_CONFIG_REL,
                PBRF_SPEC_OBSERVATION_SCHEMA_CONTRACT_REL,
            ),
            (
                "configs/rules/tenhou_4p_hanchan_v1.json",
                "configs/contracts/action_table_v1.json",
                "configs/models/model_input_v1.json",
                "configs/contracts/packet_boundary_v1.json",
                "configs/contracts/observation_schema_v1.json",
            )
        );
    }

    #[test]
    fn resolve_prefers_nonempty_override() {
        assert_eq!(resolve_hash(Some("sha256:aa"), "sha256:bb"), "sha256:aa");
        assert_eq!(resolve_hash(Some(""), "sha256:bb"), "sha256:bb");
        assert_eq!(resolve_hash(None, "sha256:bb"), "sha256:bb");
    }

    #[test]
    fn canonical_hashes_match_head_oracle() {
        // Hand-derived from HEAD: `python -c
        // "from hydra2.search.pbrf_spec import _canonical_hashes"` prints this
        // triple (JCS byte strings verified independently in the wave notes).
        let (rng, stream, case) = canonical_hashes().expect("fixed descriptors are canon-safe");
        assert_eq!(
            rng,
            "sha256:fae44479a890defcc3ae0255d5b2d107e1d895788ea73902e0aaafb3c02efada"
        );
        assert_eq!(
            stream,
            "sha256:31800c9df746595ddd775f57d7bc4f89269e47b250b75337d4de374c63eeed78"
        );
        assert_eq!(
            case,
            "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        );
    }
}
