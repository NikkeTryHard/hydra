//! gumbel_spec: frozen Candidate 6 spec-factory values + pure hash assembly.
//!
//! DAG: pyo3 + `hydra-feed` canon/digest owners ONLY (the three canonical
//! descriptor digests seal through `feed::canon` + `feed::digest`, never a
//! local hasher — canon-wins B3, see `search.rs` docs). File reads, the lazy
//! torch model probe, and the `CandidateSpec`/`ResourceBudget` dataclass
//! build stay Python in `gumbel_spec.py` (file/YAML IO and torch are OUT OF
//! SCOPE; live dataclasses never cross, mirroring `search_profiles.rs`).
//! FORBIDS: file/JSON IO, torch/CUDA, model probes, digest reimplementation
//! (no `sha2`/`hashlib` here — `hydra_feed::digest` owns hex), selection
//! math, RNG streams, and builder/validator logic (config `__post_init__`
//! stays Python in `gumbel_config.py`).
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_shared::register`, `search_shared.rs:59-89`); no new entry point.
//! Wiring is MAIN-ONLY (`search::register` calls this `register` with its
//! `sub`, alongside `search.rs:1729-1734`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `search_shared::register`
//! (`search_shared.rs:59-89`); attached-extract + ONE `py.detach` with zero
//! Python API inside mirrors `search_profiles::profiles_jobs_for`
//! (`search_profiles.rs:189-201`); `Bound<'_, PyAny>` +
//! `is_instance_of::<PyBool>` mirrors `search_profiles::as_u64`
//! (`search_profiles.rs:42-47`); feed seal `canonical_bytes_value` +
//! `sha256_hex` mirrors `eval_leaves::case_manifest_digest_value`
//! (`eval_leaves.rs:416-420`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

/// Frozen Gumbel algorithm name (`gumbel_spec.py:190`).
const GUMBEL_SPEC_ALGORITHM: &str = "gumbel_search";
/// Frozen PUCT comparator algorithm name (`gumbel_spec.py:274`).
const PUCT_SPEC_ALGORITHM: &str = "puct_search";
/// Frozen spec algorithm version (`gumbel_spec.py:191,275`).
const GUMBEL_SPEC_ALGORITHM_VERSION: &str = "1.0.0";
/// Frozen fallback candidate (`gumbel_spec.py:207,289`).
const GUMBEL_SPEC_FALLBACK_CANDIDATE_ID: &str = "candidate0";
/// Frozen budget mode (`gumbel_spec.py:180,264`).
const GUMBEL_SPEC_BUDGET_MODE: &str = "gameplay_5s";
/// Frozen fallback margin ms (`gumbel_spec.py:182,266`).
const GUMBEL_SPEC_FALLBACK_MARGIN_MS: u64 = 200;
/// Frozen budget particle cap (`gumbel_spec.py:185,269`).
const GUMBEL_SPEC_MAX_PARTICLES: u64 = 16;

/// Seal one frozen descriptor through the feed owners: `feed::canon`
/// JCS-emits the staged value, then `feed::digest` renders the
/// `sha256:<hex>` text — bit-identical to the retired
/// `"sha256:" + hashlib.sha256(canonical_bytes(...)).hexdigest()` lines.
/// Pure and detach-safe.
fn seal_descriptor(value: &serde_json::Value, record: &str) -> Result<String, String> {
    let bytes = hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Caller-override-wins hash pick (`gumbel_spec.py:156-165,192-194,245-248`):
/// `None`/empty reads as the bound default, a non-empty `str` wins. Pure
/// and detach-safe.
fn pick_override(caller: Option<&str>, default: &str) -> String {
    match caller {
        Some(text) if !text.is_empty() => text.to_owned(),
        _ => default.to_owned(),
    }
}

/// Frozen RNG/stream/case digests (`gumbel_spec.py:117-129`
/// `_canonical_hashes`): the three candidate0-authority descriptors sealed
/// detached through the feed owners. Returns
/// `(rng_protocol_hash, random_stream_schema_hash, case_manifest_hash)`.
#[pyfunction]
fn gumbel_spec_canonical_hashes(py: Python<'_>) -> PyResult<(String, String, String)> {
    py.detach(|| {
        let rng = seal_descriptor(
            &serde_json::json!({"protocol": "counter_based_v1", "version": "1.0.0"}),
            "gumbel_spec:rng_protocol",
        )?;
        let stream = seal_descriptor(
            &serde_json::json!({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]}),
            "gumbel_spec:random_stream",
        )?;
        let case = seal_descriptor(&serde_json::json!([]), "gumbel_spec:case_manifest")?;
        Ok((rng, stream, case))
    })
    .map_err(|err: String| PyValueError::new_err(err))
}

/// Caller-override-wins hash pick over plain strings (file-hash defaults
/// cross as `default_value`, never read here): `None`/empty reads as the
/// default, a non-empty `str` wins. Attached only (no compute — moves
/// only). `TypeError` on non-str overrides, never a silent default.
#[pyfunction]
#[pyo3(signature = (caller_value, default_value))]
fn gumbel_spec_pick_hash(
    caller_value: Bound<'_, PyAny>,
    default_value: String,
) -> PyResult<String> {
    if caller_value.is_none() {
        return Ok(default_value);
    }
    if caller_value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "gumbel spec hash override must be str or None",
        ));
    }
    let text: String = caller_value
        .extract()
        .map_err(|_| PyTypeError::new_err("gumbel spec hash override must be str or None"))?;
    Ok(pick_override(Some(text.as_str()), default_value.as_str()))
}

/// Register the spec-factory consts + kernels on the EXISTING `search`
/// submodule (mirrors `search_shared::register`, `search_shared.rs:59-89`;
/// MAIN calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("GUMBEL_SPEC_ALGORITHM", GUMBEL_SPEC_ALGORITHM)?;
    sub.add("PUCT_SPEC_ALGORITHM", PUCT_SPEC_ALGORITHM)?;
    sub.add(
        "GUMBEL_SPEC_ALGORITHM_VERSION",
        GUMBEL_SPEC_ALGORITHM_VERSION,
    )?;
    sub.add(
        "GUMBEL_SPEC_FALLBACK_CANDIDATE_ID",
        GUMBEL_SPEC_FALLBACK_CANDIDATE_ID,
    )?;
    sub.add("GUMBEL_SPEC_BUDGET_MODE", GUMBEL_SPEC_BUDGET_MODE)?;
    sub.add(
        "GUMBEL_SPEC_FALLBACK_MARGIN_MS",
        GUMBEL_SPEC_FALLBACK_MARGIN_MS,
    )?;
    sub.add("GUMBEL_SPEC_MAX_PARTICLES", GUMBEL_SPEC_MAX_PARTICLES)?;
    sub.add_function(wrap_pyfunction!(gumbel_spec_canonical_hashes, sub)?)?;
    sub.add_function(wrap_pyfunction!(gumbel_spec_pick_hash, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod gumbel_spec_tests {
    use super::*;

    #[test]
    fn frozen_values_match_head() {
        assert_eq!(GUMBEL_SPEC_ALGORITHM, "gumbel_search");
        assert_eq!(PUCT_SPEC_ALGORITHM, "puct_search");
        assert_eq!(GUMBEL_SPEC_ALGORITHM_VERSION, "1.0.0");
        assert_eq!(GUMBEL_SPEC_FALLBACK_CANDIDATE_ID, "candidate0");
        assert_eq!(GUMBEL_SPEC_BUDGET_MODE, "gameplay_5s");
        assert_eq!(
            (GUMBEL_SPEC_FALLBACK_MARGIN_MS, GUMBEL_SPEC_MAX_PARTICLES),
            (200, 16)
        );
    }

    #[test]
    fn canonical_hashes_match_oracle() {
        // Oracle vectors hand-derived from HEAD
        // (`python/hydra2/search/gumbel_spec.py:117-129` via
        // `_canonical_hashes()` on the current tree).
        let rng = seal_descriptor(
            &serde_json::json!({"protocol": "counter_based_v1", "version": "1.0.0"}),
            "gumbel_spec:rng_protocol",
        )
        .unwrap();
        let stream = seal_descriptor(
            &serde_json::json!({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]}),
            "gumbel_spec:random_stream",
        )
        .unwrap();
        let case = seal_descriptor(&serde_json::json!([]), "gumbel_spec:case_manifest").unwrap();
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

    #[test]
    fn override_pick_matches_oracle_binding() {
        // Oracle binding (`gumbel_spec.py:156-165`): `None`/empty reads as
        // the default, a non-empty caller value wins.
        assert_eq!(pick_override(None, "sha256:d"), "sha256:d".to_owned());
        assert_eq!(pick_override(Some(""), "sha256:d"), "sha256:d".to_owned());
        assert_eq!(
            pick_override(Some("sha256:x"), "sha256:d"),
            "sha256:x".to_owned()
        );
    }
}
