//! tracking_leaves: pure tracking record-shape leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search owner exists for the observer-mirror
//! record shapes — `ls crates` shows no `tracking-core` member, so every table
//! below is frozen from HEAD Python text with the oracle line cited per use;
//! same NONE-owner restatement precedent as `common_roots.rs:38-45`). FORBIDS:
//! transport (the `mirror.rs`-owned REST sender), file IO, env reads,
//! wall-clock, RNG, and mutable run registries — the bridge takes every input
//! as an argument and returns plain data.
//!
//! TABLE ported here — pure leaves:
//! - `tracking_filter_metrics` <- `_filter_metrics`
//!   (`python/hydra2/tracking/clearml_mirror.py:168-184`, shared with the
//!   MLflow mirror through its `clearml_mirror` import)
//! - `tracking_flatten_params` <- `_flatten_params`
//!   (`clearml_mirror.py:187-200`)
//! - `tracking_manifest_tags` <- `_manifest_tags`
//!   (`clearml_mirror.py:203-214`)
//! - `tracking_resolve_base_url` <- `_resolve_base_url`
//!   (`clearml_mirror.py:233-246`; the translator passes
//!   `os.environ.get(env_name)` as `env_value`, so Rust stays env-free)
//! - `tracking_promotion_scalars` <- `log_promotion` scalar loop
//!   (`clearml_mirror.py:488-497`, shared shape with
//!   `python/hydra2/tracking/mlflow_mirror.py:371-383`)
//! - `tracking_eval_metrics_for` <- `log_eval_report` prefix render
//!   (`clearml_mirror.py:464-466`, shared shape with
//!   `mlflow_mirror.py:348-350`)
//! - `tracking_duplicate_audit_tags` <- `log_duplicate_audit` tag render
//!   (`clearml_mirror.py:533-537`)
//! - `tracking_clamp_interval_ms` <- `resolve_interval_ms` explicit arm
//!   (`python/hydra2/tracking/verbose_sampler.py:75-89` without the warn; the
//!   warn text plus the `HYDRA2_VERBOSE_INTERVAL_MS` env arm stay Python in
//!   the translator)
//! - consts `TRACKING_*` <- frozen tables (`clearml_mirror.py:73-123`,
//!   `mlflow_mirror.py:80-87`, `verbose_sampler.py:56-66`)
//!
//! LOGIC staying Python (TABLE — IO/service/clock/mutable-state):
//! - `ClearmlMirror` / `MlflowMirror` / `NullMirror` / `NullMlflowMirror` /
//!   `make_mirror` (transport via `mirror.rs`, file fallback, mutable run ids)
//! - `_ensure_store_dir` / `_append_jsonl` (file IO), `_bridge_mirror` (lazy
//!   transport import), `default_offline_dir` / `default_tracking_dir` (env +
//!   `hydra2.config.artifact_root`), `is_enabled` / `_env_truthy` (env reads)
//! - `start_run` offline ids (`clearml_mirror.py:390`, `mlflow_mirror.py:260`:
//!   `abs(hash(...))` is salted per-process — never portable, stays Python)
//! - `VerboseSampler` / `NullVerboseSampler` / `make_verbose_sampler`,
//!   `_null_row` (clock reads), `_sample_gpu/cpu/torch` plus the thread
//!   lifecycle (NVML/psutil/torch SDKs, sink IO)
//! - `resolve_interval_ms` warn text + env arm (translator-owned); float sums
//!   anywhere (none ported — sums stay Python)
//!
//! Argument contract: dict-domain inputs take `Bound<PyDict>` (non-dict inputs
//! fail argument extraction with `TypeError` and the Python translator falls
//! back to the byte-identical oracle body — the `rc_sections.rs:159`
//! `Bound<PyDict>` precedent); every leaf is infallible over its domain
//! (per-pair skips mirror the oracle `continue`s, never a whole-call error).
//!
//! Shape per fn: attached staging (Python memory touched, so `str()` renders
//! byte-exact — per `rc_require.rs:60-72`) → ONE `py.detach(|| …)` over owned
//! plain data with zero Python API inside (per `validate.rs:83-95`) → attached
//! wrap (`PyDict::new` + `set_item` per `contracts.rs:1227-1231`).
//! Flat string walks (`flatten`, `manifest_tags`, `duplicate_tags`) run
//! attached end-to-end (the walk touches Python memory throughout — the
//! `canon_rng.rs:263` attached-only precedent).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed `sub`
//! in `wrap_pyfunction!(f, sub)` mirror `validate::register`
//! (`crates/bridge/src/validate.rs:111-127`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); `PyFrozenSet::new` mirrors
//! `contracts.rs:1186`; `PyTuple::new` mirrors `contracts.rs:1156`; scalar
//! `sub.add` mirrors `belief_kernel::register`
//! (`crates/bridge/src/belief_kernel.rs:254-263`); `cast::<PyDict>` mirrors
//! `rc_resume_gates.rs:233`; `Bound<PyDict>` params mirror
//! `rc_sections.rs:159`; plain-tuple returns mirror
//! `contracts::seat_winds_for_dealer` (`crates/bridge/src/contracts.rs:999`).
//!
//! No `hydra2._native.tracking` submodule exists yet (`lib.rs` registers
//! `mirror` for transport, never `tracking`): this file registers on the
//! EXISTING shared `hydra2._native.contracts` submodule via `register`
//! (mirrors `common_roots::register`); no new entry point. MAIN wiring:
//! `pub mod tracking_leaves;` in `lib.rs` (between `tiles` and `utility`)
//! plus `crate::tracking_leaves::register(&sub)?;` in `contracts.rs` next to
//! `crate::distill_leaves::register(&sub)?;` (`contracts.rs:1321`).

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFrozenSet, PyModule, PyTuple};

/// ClearML/MLflow experiment name (`clearml_mirror.py:74`, `mlflow_mirror.py:81`).
const TRACKING_EXPERIMENT_DEFAULT: &str = "hydra2-tenhou-4p";
/// ClearML task fallback (`clearml_mirror.py:77`).
const TRACKING_TASK_DEFAULT: &str = "hydra2-training";
/// MLflow run-name fallback (`mlflow_mirror.py:84`; same text, distinct owner).
const TRACKING_RUN_NAME_DEFAULT: &str = "hydra2-training";
/// Allowlisted scalar keys (`clearml_mirror.py:82-110`, frozen 25).
const TRACKING_METRIC_ALLOWLIST: [&str; 25] = [
    "total",
    "policy",
    "placement",
    "value",
    "event",
    "belief",
    "masked_nll",
    "top1",
    "top3",
    "top5",
    "calibration_ece",
    "legal_uniform_nll",
    "legal_uniform_gap",
    "legal_uniform_comparison",
    "support_min",
    "support_max",
    "confusion",
    "strata",
    "observed_estimate",
    "ci_lo",
    "ci_hi",
    "num_eval_batches",
    "temperature",
    "calibrated_nll",
    "calibrated_ece",
];
/// Allowlist prefix families (`clearml_mirror.py:117`, frozen 5).
const TRACKING_METRIC_ALLOWLIST_PREFIXES: [&str; 5] = [
    "event_",
    "belief_",
    "per_type/",
    "calibrated_",
    "temperature",
];
/// Promotion CI triple (`clearml_mirror.py:120`, `mlflow_mirror.py:87`).
const TRACKING_PROMOTION_METRIC_KEYS: [&str; 3] = ["observed_estimate", "ci_lo", "ci_hi"];
/// Closed-loopback default (`clearml_mirror.py:123`).
const TRACKING_DEFAULT_BASE_URL: &str = "http://127.0.0.1:9";
/// Telemetry row schema version (`verbose_sampler.py:57`).
const TRACKING_SAMPLER_SCHEMA_VERSION: u32 = 1;
/// Default sampler cadence in ms (`verbose_sampler.py:60`).
const TRACKING_DEFAULT_INTERVAL_MS: i64 = 50;
/// Allowed cadences (`verbose_sampler.py:63`).
const TRACKING_VALID_INTERVALS_MS: [i64; 2] = [20, 50];
/// NVML-error budget before the GPU group drops (`verbose_sampler.py:66`).
const TRACKING_GPU_DROP_AFTER_ERRORS: u32 = 100;

/// Owned numeric view of one metric value: `Skip` mirrors every oracle
/// `continue` (non-`str` key, unallowlisted key handled by the core,
/// non-numeric, `bool`, unrepresentable); `Num` carries the `float(value)`
/// result for the detached finite gate.
#[derive(Clone, Copy)]
enum StagedNumber {
    Skip,
    Num(f64),
}

/// `float(value)` staging (`clearml_mirror.py:178-180`): `bool` excluded first
/// (it subclasses `int`), then `i64`, then `f64`. Huge ints that miss `i64`
/// still land through `f64` (`__float__`, e.g. `10**30` → `1e30`); values with
/// no numeric view stage `Skip` (the oracle `isinstance` gate drops them).
fn stage_number(value: &Bound<'_, PyAny>) -> StagedNumber {
    if value.is_instance_of::<PyBool>() {
        return StagedNumber::Skip;
    }
    if let Ok(n) = value.extract::<i64>() {
        // proof: int lanes are small oracle ints (< 2^53), exact; tracking math tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let n_f: f64 = n as f64;
        return StagedNumber::Num(n_f);
    }
    if let Ok(f) = value.extract::<f64>() {
        return StagedNumber::Num(f);
    }
    StagedNumber::Skip
}

/// Allowlist test (`clearml_mirror.py:176`): exact set or any prefix family
/// (`str.startswith` with a tuple is an any-match — mirrored exactly).
fn is_allowlisted(key: &str) -> bool {
    if TRACKING_METRIC_ALLOWLIST.contains(&key) {
        return true;
    }
    TRACKING_METRIC_ALLOWLIST_PREFIXES
        .iter()
        .any(|&prefix| key.starts_with(prefix))
}

/// Detached filter core over staged `(key, number)` pairs: allowlist gate then
/// the `math.isfinite` gate (`clearml_mirror.py:176-183`). Insertion order
/// preserved (never a `HashMap` — the ClearML tag list observes order).
fn filter_core(pairs: Vec<(Option<String>, StagedNumber)>) -> Vec<(String, f64)> {
    let mut out = Vec::with_capacity(pairs.len());
    for (key, staged) in pairs {
        let Some(k) = key else { continue };
        if !is_allowlisted(&k) {
            continue;
        }
        let StagedNumber::Num(n) = staged else {
            continue;
        };
        if !n.is_finite() {
            continue;
        }
        out.push((k, n));
    }
    out
}

/// Detached promotion-triple core (`clearml_mirror.py:490-497`): `promotion_`
/// prefixed scalars over the shared triple with the same numeric + finite
/// gates as the filter.
fn promotion_core(staged: Vec<(&'static str, StagedNumber)>) -> Vec<(String, f64)> {
    let mut out = Vec::with_capacity(staged.len());
    for (key, number) in staged {
        let StagedNumber::Num(n) = number else {
            continue;
        };
        if !n.is_finite() {
            continue;
        }
        out.push((format!("promotion_{key}"), n));
    }
    out
}

/// Detached explicit/env/default fold (`clearml_mirror.py:235-246`): first
/// non-empty-after-`strip` wins, else the closed-loopback default. `strip`
/// here is Unicode-whitespace, matching the oracle `str.strip()`.
fn resolve_base_url_core(explicit: Option<&str>, env_value: Option<&str>) -> String {
    for candidate in [explicit, env_value].into_iter().flatten() {
        if !candidate.trim().is_empty() {
            return candidate.trim().to_owned();
        }
    }
    TRACKING_DEFAULT_BASE_URL.to_owned()
}

/// Detached interval-clamp core (`verbose_sampler.py:77-89` without the warn):
/// the narrowed `int(explicit)` wins when it names an allowed cadence,
/// otherwise the default with the warn flag set.
fn clamp_core(coerced: Option<i64>) -> (i64, bool) {
    match coerced {
        Some(n) if TRACKING_VALID_INTERVALS_MS.contains(&n) => (n, false),
        _ => (TRACKING_DEFAULT_INTERVAL_MS, true),
    }
}

/// Attached `str()` view of an optional-text slot: `None` stages absent and
/// `str()` failures stage absent (the oracle `try/except` falls through —
/// same outcome).
fn opt_str(obj: &Bound<'_, PyAny>) -> Option<String> {
    if obj.is_none() {
        return None;
    }
    obj.str().ok()?.to_str().ok().map(str::to_owned)
}

/// Metric allowlist filter (`clearml_mirror.py:168-184`, shared with the
/// MLflow mirror through its `clearml_mirror` import): keep allowlisted
/// numeric series, drop everything else. Infallible over dicts (per-pair skips
/// mirror the oracle `continue`s); insertion order preserved. Compute runs
/// detached; pure function of the inputs, never wall-clock.
#[pyfunction]
fn tracking_filter_metrics(py: Python<'_>, entry: Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let mut staged = Vec::with_capacity(entry.len());
    for (key, value) in entry.iter() {
        staged.push((key.extract::<String>().ok(), stage_number(&value)));
    }
    let filtered = py.detach(|| filter_core(staged));
    let out = PyDict::new(py);
    for (key, number) in filtered {
        out.set_item(key, number)?;
    }
    Ok(out.unbind())
}

/// Recursive `parent.child` flattener (`clearml_mirror.py:187-200`): nested
/// dicts recurse under the dotted prefix, every other leaf renders via
/// `str()`. Attached end-to-end (the walk touches Python memory throughout).
/// Deliberate garbage-input-only divergence: nested non-dict mappings render
/// as `str()` of the whole object (the oracle recurses into any `Mapping`);
/// realistic loop configs are JSON-plain dicts.
#[pyfunction]
fn tracking_flatten_params(params: Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let py = params.py();
    let mut pairs = Vec::new();
    flatten_into("", &params, &mut pairs)?;
    let out = PyDict::new(py);
    for (name, text) in pairs {
        out.set_item(name, text)?;
    }
    Ok(out.unbind())
}

/// Attached flatten worker: `str(key)` names each level (`f"{prefix}.{key}"`
/// below the top), dict values recurse, all other leaves stringify.
fn flatten_into(
    prefix: &str,
    params: &Bound<'_, PyDict>,
    out: &mut Vec<(String, String)>,
) -> PyResult<()> {
    for (key, value) in params.iter() {
        let key_text = key.str()?.to_str()?.to_owned();
        let name = if prefix.is_empty() {
            key_text
        } else {
            format!("{prefix}.{key_text}")
        };
        if let Ok(nested) = value.cast::<PyDict>() {
            flatten_into(&name, nested, out)?;
        } else {
            out.push((name, value.str()?.to_str()?.to_owned()));
        }
    }
    Ok(())
}

/// Manifest-digest tags plus the environment digest
/// (`clearml_mirror.py:203-214`): `manifest.<key>` for non-empty `str`
/// digests, then `environment.digest` when non-empty. Attached end-to-end
/// (flat walk over Python memory); insertion order preserved.
#[pyfunction]
fn tracking_manifest_tags(
    py: Python<'_>,
    manifest_hashes: Bound<'_, PyDict>,
    environment_digest: Option<String>,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in manifest_hashes.iter() {
        if let (Ok(name), Ok(digest)) = (key.extract::<String>(), value.extract::<String>())
            && !digest.is_empty()
        {
            out.set_item(format!("manifest.{name}"), digest)?;
        }
    }
    if let Some(digest) = environment_digest
        && !digest.is_empty()
    {
        out.set_item("environment.digest", digest)?;
    }
    Ok(out.unbind())
}

/// Explicit-then-env base-URL resolution (`clearml_mirror.py:233-246`): the
/// translator passes `os.environ.get(env_name)` as `env_value`, so Rust stays
/// env-free. Compute runs detached; pure function of the inputs.
#[pyfunction]
fn tracking_resolve_base_url(
    py: Python<'_>,
    explicit: Bound<'_, PyAny>,
    env_value: Bound<'_, PyAny>,
) -> PyResult<String> {
    let staged_explicit = opt_str(&explicit);
    let staged_env = opt_str(&env_value);
    Ok(py.detach(|| resolve_base_url_core(staged_explicit.as_deref(), staged_env.as_deref())))
}

/// Promotion CI triple as `promotion_` scalars (`clearml_mirror.py:483-498`,
/// shared shape with `mlflow_mirror.py:367-385`): the record stages attached,
/// the triple fold runs detached. Infallible over dicts; insertion order is
/// the frozen triple order.
#[pyfunction]
fn tracking_promotion_scalars(py: Python<'_>, record: Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let mut staged = Vec::with_capacity(TRACKING_PROMOTION_METRIC_KEYS.len());
    for key in TRACKING_PROMOTION_METRIC_KEYS {
        let number = match record.get_item(key) {
            Ok(Some(value)) => stage_number(&value),
            _ => StagedNumber::Skip,
        };
        staged.push((key, number));
    }
    let scalars = py.detach(|| promotion_core(staged));
    let out = PyDict::new(py);
    for (name, number) in scalars {
        out.set_item(name, number)?;
    }
    Ok(out.unbind())
}

/// `eval_<label>_<key>` render over the allowlisted subset
/// (`clearml_mirror.py:455-466`, shared shape with
/// `mlflow_mirror.py:339-350`): the label renders via `str()` exactly like
/// the oracle f-string; the report stages like the filter and the fold runs
/// detached. Infallible over dicts.
#[pyfunction]
fn tracking_eval_metrics_for(
    py: Python<'_>,
    label: Bound<'_, PyAny>,
    report: Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let label_text = label.str()?.to_str()?.to_owned();
    let mut staged = Vec::with_capacity(report.len());
    for (key, value) in report.iter() {
        staged.push((key.extract::<String>().ok(), stage_number(&value)));
    }
    let metrics = py.detach(move || {
        let mut out = Vec::with_capacity(staged.len());
        for (key, number) in filter_core(staged) {
            out.push((format!("eval_{label_text}_{key}"), number));
        }
        out
    });
    let out = PyDict::new(py);
    for (name, number) in metrics {
        out.set_item(name, number)?;
    }
    Ok(out.unbind())
}

/// Duplicate-wall tag render (`clearml_mirror.py:533-537`): empty and `None`
/// digests contribute nothing; manifest tag precedes telemetry (insertion
/// order, never a set). Pure function of the inputs, never wall-clock.
#[pyfunction]
fn tracking_duplicate_audit_tags(
    manifest_digest: Option<String>,
    telemetry_digest: Option<String>,
) -> Vec<String> {
    let mut tags = Vec::with_capacity(2);
    if let Some(digest) = manifest_digest.as_deref()
        && !digest.is_empty()
    {
        tags.push(format!("duplicate.manifest_digest={digest}"));
    }
    if let Some(digest) = telemetry_digest.as_deref()
        && !digest.is_empty()
    {
        tags.push(format!("duplicate.telemetry_digest={digest}"));
    }
    tags
}

/// Explicit-arm interval clamp (`verbose_sampler.py:75-89` without the warn):
/// the `builtins.int` coercion runs attached so `int(50.7) == 50`,
/// `int("50") == 50`, and `int(True) == 1` match exactly; the `(20, 50)`
/// membership runs detached. Returns `(value, warned)` — the translator owns
/// the warn text plus `stacklevel`, so the message stays byte-identical.
/// `None`/uncoercible inputs stage `(DEFAULT, true)`; the translator never
/// passes `None` (it owns the `HYDRA2_VERBOSE_INTERVAL_MS` env arm), so that
/// pair is unreachable except via direct bridge calls.
#[pyfunction]
fn tracking_clamp_interval_ms(py: Python<'_>, explicit: Bound<'_, PyAny>) -> (i64, bool) {
    let coerced: Option<i64> = (|| -> PyResult<i64> {
        let int_type = py.import("builtins")?.getattr("int")?;
        let narrowed = int_type.call1((explicit,))?;
        narrowed.extract::<i64>()
    })()
    .ok();
    py.detach(|| clamp_core(coerced))
}

/// Register the tracking leaves + frozen tables on the shared `contracts`
/// submodule (mirrors `common_roots::register`): consts via `sub.add`, fns via
/// `wrap_pyfunction!(f, sub)`; single cdylib, no new entry point. MAIN calls
/// this from `contracts::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("TRACKING_EXPERIMENT_DEFAULT", TRACKING_EXPERIMENT_DEFAULT)?;
    sub.add("TRACKING_TASK_DEFAULT", TRACKING_TASK_DEFAULT)?;
    sub.add("TRACKING_RUN_NAME_DEFAULT", TRACKING_RUN_NAME_DEFAULT)?;
    sub.add(
        "TRACKING_METRIC_ALLOWLIST",
        PyFrozenSet::new(py, TRACKING_METRIC_ALLOWLIST)?,
    )?;
    sub.add(
        "TRACKING_METRIC_ALLOWLIST_PREFIXES",
        PyTuple::new(py, TRACKING_METRIC_ALLOWLIST_PREFIXES)?,
    )?;
    sub.add(
        "TRACKING_PROMOTION_METRIC_KEYS",
        PyTuple::new(py, TRACKING_PROMOTION_METRIC_KEYS)?,
    )?;
    sub.add("TRACKING_DEFAULT_BASE_URL", TRACKING_DEFAULT_BASE_URL)?;
    sub.add(
        "TRACKING_SAMPLER_SCHEMA_VERSION",
        TRACKING_SAMPLER_SCHEMA_VERSION,
    )?;
    sub.add("TRACKING_DEFAULT_INTERVAL_MS", TRACKING_DEFAULT_INTERVAL_MS)?;
    sub.add(
        "TRACKING_VALID_INTERVALS_MS",
        PyTuple::new(py, TRACKING_VALID_INTERVALS_MS)?,
    )?;
    sub.add(
        "TRACKING_GPU_DROP_AFTER_ERRORS",
        TRACKING_GPU_DROP_AFTER_ERRORS,
    )?;
    sub.add_function(wrap_pyfunction!(tracking_filter_metrics, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_flatten_params, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_manifest_tags, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_resolve_base_url, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_promotion_scalars, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_eval_metrics_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_duplicate_audit_tags, sub)?)?;
    sub.add_function(wrap_pyfunction!(tracking_clamp_interval_ms, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_table_shapes_match_head() {
        assert_eq!(TRACKING_METRIC_ALLOWLIST.len(), 25);
        assert_eq!(TRACKING_METRIC_ALLOWLIST_PREFIXES.len(), 5);
        assert_eq!(TRACKING_PROMOTION_METRIC_KEYS.len(), 3);
        assert_eq!(TRACKING_EXPERIMENT_DEFAULT, "hydra2-tenhou-4p");
        assert_eq!(TRACKING_TASK_DEFAULT, "hydra2-training");
        assert_eq!(TRACKING_RUN_NAME_DEFAULT, "hydra2-training");
        assert_eq!(TRACKING_DEFAULT_BASE_URL, "http://127.0.0.1:9");
        assert_eq!(TRACKING_SAMPLER_SCHEMA_VERSION, 1);
        assert_eq!(TRACKING_DEFAULT_INTERVAL_MS, 50);
        assert_eq!(TRACKING_VALID_INTERVALS_MS, [20, 50]);
        assert_eq!(TRACKING_GPU_DROP_AFTER_ERRORS, 100);
    }

    #[test]
    fn allowlist_gates_exact_and_prefix() {
        assert!(is_allowlisted("value"));
        assert!(is_allowlisted("ci_hi"));
        assert!(is_allowlisted("num_eval_batches"));
        assert!(is_allowlisted("event_win_rate"));
        assert!(is_allowlisted("belief_head0"));
        assert!(is_allowlisted("per_type/pon/recall"));
        assert!(is_allowlisted("calibrated_new_metric"));
        assert!(is_allowlisted("temperature"));
        assert!(is_allowlisted("temperature_scaled"));
        assert!(!is_allowlisted("loss"));
        assert!(!is_allowlisted("events"));
        assert!(!is_allowlisted("temp"));
        assert!(!is_allowlisted(""));
    }

    #[test]
    fn filter_core_drops_non_numeric_and_unallowlisted() {
        let out = filter_core(vec![
            (Some("value".to_owned()), StagedNumber::Num(1.5)),
            (Some("loss".to_owned()), StagedNumber::Num(1.5)),
            (None, StagedNumber::Num(1.0)),
            (Some("top1".to_owned()), StagedNumber::Skip),
            (
                Some("event_acc".to_owned()),
                StagedNumber::Num(f64::INFINITY),
            ),
            (Some("belief_h".to_owned()), StagedNumber::Num(f64::NAN)),
            (Some("ci_lo".to_owned()), StagedNumber::Num(-2.0)),
            (Some("num_eval_batches".to_owned()), StagedNumber::Num(1e30)),
        ]);
        assert_eq!(
            out,
            vec![
                ("value".to_owned(), 1.5),
                ("ci_lo".to_owned(), -2.0),
                ("num_eval_batches".to_owned(), 1e30),
            ]
        );
    }

    #[test]
    fn promotion_core_prefixes_the_triple_in_order() {
        let out = promotion_core(vec![
            ("observed_estimate", StagedNumber::Num(0.5)),
            ("ci_lo", StagedNumber::Num(0.1)),
            ("ci_hi", StagedNumber::Skip),
        ]);
        assert_eq!(
            out,
            vec![
                ("promotion_observed_estimate".to_owned(), 0.5),
                ("promotion_ci_lo".to_owned(), 0.1),
            ]
        );
    }

    #[test]
    fn base_url_fold_prefers_first_nonempty_stripped() {
        assert_eq!(
            resolve_base_url_core(Some("  https://x  "), Some("http://env:1")),
            "https://x"
        );
        assert_eq!(
            resolve_base_url_core(Some(""), Some("http://env:1")),
            "http://env:1"
        );
        assert_eq!(
            resolve_base_url_core(None, Some("   ")),
            TRACKING_DEFAULT_BASE_URL
        );
        assert_eq!(resolve_base_url_core(None, None), TRACKING_DEFAULT_BASE_URL);
    }

    #[test]
    fn clamp_core_names_cadences_else_default_with_warn() {
        assert_eq!(clamp_core(Some(20)), (20, false));
        assert_eq!(clamp_core(Some(50)), (50, false));
        assert_eq!(clamp_core(Some(99)), (50, true));
        assert_eq!(clamp_core(Some(21)), (50, true));
        assert_eq!(clamp_core(None), (50, true));
    }
}
