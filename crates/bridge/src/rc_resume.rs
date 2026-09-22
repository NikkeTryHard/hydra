//! rc_resume: resume-plan compatibility gates on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists — same NONE shape as `rc_require.rs:32-38`; `rg` for
//! `run_digest|worker_plan|micro_in_update|shuffle_buffer|manifest_hash` over
//! `crates/feed/src` + `crates/search/src` hits only unrelated span/manifest
//! comments, no resume authority). FORBIDS: checkpoint-dir globs, sidecar file
//! reads, `run.yaml` loading, `ResumePlan`/`RunConfig` construction, and plan
//! formatting (all stay Python in `python/hydra2/training/_rc_resume.py`),
//! wall-clock/RNG, and `ContractError` shaping (the bridge raises
//! `ValueError`; the thin Python translator maps to `ContractError` with
//! byte-identical messages).
//!
//! TABLE ported here — the 7 pure gates, oracle
//! `python/hydra2/training/_rc_resume.py:246-284` (worktree at port time;
//! message bytes captured live from the oracle pre-edit, gate order
//! precedence probed with multi-fault inputs):
//! - `rc_check_compat` ← `_check_compat` (:246-284; digest, num_workers,
//!   world_size, seed, accum window, shuffle buffer, manifest pin)
//!
//! LOGIC staying Python: `_CKPT_STEM_RE` (compiled-pattern literal),
//! `_Sidecar` + `_parse_shuffle`/`_parse_accum`/`_parse_worker_plan`/
//! `_parse_sidecar_manifests` (section dataclasses stay Python per wave-4; the
//! cursor envelope already rides the bridged `rc_unpack_cursor`),
//! `_read_sidecar` (file IO), `_iter_structural_candidates`/
//! `find_latest_checkpoint` (dir globs + newest-first sort over live paths),
//! `resolve_resume_plan` (IO orchestration; calls the bridged gate once per
//! candidate), `format_plan` (human-readable summary), and the
//! `ContractError` translator (`__all__` unchanged).
//!
//! Shape per fn: attached staging (owned `String`/`i64`/`Option<String>`
//! extraction — every message slot is plain str/int interpolation, so no
//! `{value!r}` repr staging is needed) → ONE `py.detach(|| …)` over owned
//! plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `validate.rs:83-95`) → attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Inputs arrive post-validation (dataclass fields,
//! never raw mappings), so bool-exclusion staging is unnecessary
//! (`rc_require.rs:74-84` documents the raw-mapping counterpart).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod rc_resume;` in `lib.rs` plus
//! `crate::rc_resume::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1295`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

// ---------------------------------------------------------------------------
// Detached check (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Seven resume-compatibility gates in oracle order (never reordered: the
/// resolver returns the first compatible candidate, and multi-fault inputs
/// surface the earliest gate — probed against the oracle pre-edit).
#[allow(clippy::too_many_arguments)]
fn check_compat(
    expected_digest: &str,
    parsed_run_digest: &str,
    plan_num_workers: i64,
    cfg_num_workers: i64,
    plan_world_size: i64,
    cfg_world_size: i64,
    cursor_seed: i64,
    cfg_data_seed: i64,
    micro_in_update: i64,
    accumulation_steps: i64,
    buffered_keys: i64,
    shuffle_buffer_size: i64,
    pinned: Option<&str>,
    recorded: Option<&str>,
    ckpt: &str,
) -> Result<(), String> {
    if parsed_run_digest != expected_digest {
        return Err(format!(
            "resume checkpoint {ckpt} records run_digest {parsed_run_digest} but run.yaml resolves to {expected_digest} (config drift)"
        ));
    }
    if plan_num_workers != cfg_num_workers {
        return Err(format!(
            "resume worker_plan.num_workers={plan_num_workers} != config data.num_workers={cfg_num_workers}: {ckpt}"
        ));
    }
    if plan_world_size != cfg_world_size {
        return Err(format!(
            "resume worker_plan.world_size={plan_world_size} != config data.world_size={cfg_world_size}: {ckpt}"
        ));
    }
    if cursor_seed != cfg_data_seed {
        return Err(format!(
            "resume cursor seed={cursor_seed} != config seeds.data_seed={cfg_data_seed}: {ckpt}"
        ));
    }
    if micro_in_update >= accumulation_steps {
        return Err(format!(
            "resume accum.micro_in_update={micro_in_update} outside [0, {accumulation_steps}): {ckpt}"
        ));
    }
    if buffered_keys > shuffle_buffer_size {
        return Err(format!(
            "resume shuffle buffer {buffered_keys} keys exceeds config data.shuffle_buffer_size={shuffle_buffer_size}: {ckpt}"
        ));
    }
    if let (Some(pinned), Some(recorded)) = (pinned, recorded)
        && pinned != recorded
    {
        return Err(format!(
            "resume dataset_manifest_hash {recorded} != config pin {pinned}: {ckpt}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfn (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `_check_compat` gate (oracle `_rc_resume.py:246-284`): the translator
/// passes dataclass fields positionally in this exact order — digests
/// (`expected_digest`, `parsed_run_digest`), worker plan
/// (`plan_num_workers`, `cfg_num_workers`, `plan_world_size`,
/// `cfg_world_size`), seed (`cursor_seed`, `cfg_data_seed`), accum window
/// (`micro_in_update`, `accumulation_steps`), shuffle buffer
/// (`buffered_keys`, `shuffle_buffer_size`), manifest pins (`pinned`,
/// `recorded`), `ckpt` display string last; the accept/reject decision runs
/// detached.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rc_check_compat(
    py: Python<'_>,
    expected_digest: String,
    parsed_run_digest: String,
    plan_num_workers: i64,
    cfg_num_workers: i64,
    plan_world_size: i64,
    cfg_world_size: i64,
    cursor_seed: i64,
    cfg_data_seed: i64,
    micro_in_update: i64,
    accumulation_steps: i64,
    buffered_keys: i64,
    shuffle_buffer_size: i64,
    pinned: Option<String>,
    recorded: Option<String>,
    ckpt: String,
) -> PyResult<()> {
    py.detach(|| {
        check_compat(
            &expected_digest,
            &parsed_run_digest,
            plan_num_workers,
            cfg_num_workers,
            plan_world_size,
            cfg_world_size,
            cursor_seed,
            cfg_data_seed,
            micro_in_update,
            accumulation_steps,
            buffered_keys,
            shuffle_buffer_size,
            pinned.as_deref(),
            recorded.as_deref(),
            &ckpt,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Register the resume-compat gate on the shared `contracts` submodule
/// (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `rows_seal.rs:485`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_check_compat, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIGEST_D: &str =
        "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const CKPT: &str = "/tmp/runs/x/checkpoints/ckpt-000100.pt";

    /// Staged argument tuple in translator order: digests, worker plan,
    /// seed, accum window, shuffle buffer, manifest pins, ckpt display.
    type CompatArgs = (
        String,
        String,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        Option<String>,
        Option<String>,
        String,
    );

    /// Spread a staged argument tuple into the detached check.
    fn run(v: CompatArgs) -> Result<(), String> {
        check_compat(
            &v.0,
            &v.1,
            v.2,
            v.3,
            v.4,
            v.5,
            v.6,
            v.7,
            v.8,
            v.9,
            v.10,
            v.11,
            v.12.as_deref(),
            v.13.as_deref(),
            &v.14,
        )
    }

    /// Base inputs: every gate passes (oracle `ok` + both manifest-`None`
    /// rules probed pre-edit).
    fn base() -> CompatArgs {
        (
            DIGEST_A.to_owned(),
            DIGEST_A.to_owned(),
            0,
            0,
            1,
            1,
            7,
            7,
            0,
            4,
            3,
            10_000,
            None,
            None,
            CKPT.to_owned(),
        )
    }

    #[test]
    fn compat_digests_are_64_hex() {
        for digest in [DIGEST_A, DIGEST_B, DIGEST_C, DIGEST_D] {
            assert_eq!(digest.len(), 7 + 64, "digest literal truncated: {digest}");
        }
    }

    #[test]
    fn compat_accepts_matching_plan() {
        assert!(run(base()).is_ok());
    }

    #[test]
    fn compat_manifest_none_rules_pass() {
        // Pinned but nothing recorded, or recorded but nothing pinned: OK.
        let mut pinned_only = base();
        pinned_only.12 = Some(DIGEST_C.to_owned());
        assert!(run(pinned_only).is_ok());
        let mut recorded_only = base();
        recorded_only.13 = Some(DIGEST_D.to_owned());
        assert!(run(recorded_only).is_ok());
    }

    /// Hand-derived oracle bytes (`_rc_resume.py:248-284`, captured live
    /// pre-edit): each single-fault input surfaces its own gate text.
    #[test]
    fn compat_gate_text_matches_oracle() {
        let mut faulty = base();
        faulty.1 = DIGEST_B.to_owned();
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume checkpoint {CKPT} records run_digest {DIGEST_B} but run.yaml resolves to {DIGEST_A} (config drift)"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.2 = 2;
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume worker_plan.num_workers=2 != config data.num_workers=0: {CKPT}"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.4 = 2;
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume worker_plan.world_size=2 != config data.world_size=1: {CKPT}"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.6 = 9;
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume cursor seed=9 != config seeds.data_seed=7: {CKPT}"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.8 = 4;
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume accum.micro_in_update=4 outside [0, 4): {CKPT}"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.10 = 10_001;
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume shuffle buffer 10001 keys exceeds config data.shuffle_buffer_size=10000: {CKPT}"
            )),
            "unexpected error text: {err}"
        );
        let mut faulty = base();
        faulty.12 = Some(DIGEST_C.to_owned());
        faulty.13 = Some(DIGEST_D.to_owned());
        let err = run(faulty).unwrap_err();
        assert!(
            err.contains(&format!(
                "resume dataset_manifest_hash {DIGEST_D} != config pin {DIGEST_C}: {CKPT}"
            )),
            "unexpected error text: {err}"
        );
    }

    /// Gate order is load-bearing: multi-fault inputs surface the earliest
    /// gate (probed against the oracle pre-edit).
    #[test]
    fn compat_first_gate_wins() {
        let mut v = base();
        v.1 = DIGEST_B.to_owned();
        v.2 = 2;
        v.6 = 9;
        v.8 = 9;
        v.10 = 20_000;
        let err = run(v).unwrap_err();
        assert!(
            err.contains("records run_digest"),
            "unexpected error text: {err}"
        );
        let mut w = base();
        w.2 = 2;
        w.6 = 9;
        let err = run(w).unwrap_err();
        assert!(
            err.contains("worker_plan.num_workers=2"),
            "unexpected error text: {err}"
        );
    }

    /// Pyfn boundary: positional wiring maps translator order onto the same
    /// gates (first, scalar-middle, and `Option` tail positions probed).
    #[test]
    fn pyfn_wiring_matches_translator_order() {
        Python::initialize();
        Python::attach(|py| {
            let b = base();
            assert!(
                rc_check_compat(
                    py,
                    b.0.clone(),
                    b.1.clone(),
                    b.2,
                    b.3,
                    b.4,
                    b.5,
                    b.6,
                    b.7,
                    b.8,
                    b.9,
                    b.10,
                    b.11,
                    b.12.clone(),
                    b.13.clone(),
                    b.14.clone(),
                )
                .is_ok()
            );
            let err = rc_check_compat(
                py,
                b.0.clone(),
                DIGEST_B.to_owned(),
                b.2,
                b.3,
                b.4,
                b.5,
                b.6,
                b.7,
                b.8,
                b.9,
                b.10,
                b.11,
                b.12.clone(),
                b.13.clone(),
                b.14.clone(),
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("(config drift)"),
                "unexpected error text: {err}"
            );
            let err = rc_check_compat(
                py,
                b.0.clone(),
                b.1.clone(),
                b.2,
                b.3,
                b.4,
                b.5,
                9,
                b.7,
                b.8,
                b.9,
                b.10,
                b.11,
                b.12.clone(),
                b.13.clone(),
                b.14.clone(),
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("resume cursor seed=9"),
                "unexpected error text: {err}"
            );
            let err = rc_check_compat(
                py,
                b.0.clone(),
                b.1.clone(),
                b.2,
                b.3,
                b.4,
                b.5,
                b.6,
                b.7,
                b.8,
                b.9,
                b.10,
                b.11,
                Some(DIGEST_C.to_owned()),
                Some(DIGEST_D.to_owned()),
                b.14.clone(),
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("dataset_manifest_hash"),
                "unexpected error text: {err}"
            );
        });
    }
}
