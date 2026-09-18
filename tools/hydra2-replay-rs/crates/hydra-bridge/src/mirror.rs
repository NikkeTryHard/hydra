//! mirror: thin observer-REST transport boundary over `reqwest` (blocking).
//!
//! DAG: this module depends on `reqwest` (blocking + json + rustls-tls) +
//! pyo3 ONLY — no feed/shard/search edge. Transport carries already-filtered
//! JSON; the metric allowlist/filter lives Python-side in
//! `hydra2.tracking.clearml_mirror._filter_metrics` (frozen both sides, the
//! byte-identical filter gate). FORBIDS: allowlist decisions, metric math,
//! retries, persistent clients, payload shaping beyond the fixed transport
//! envelope — arg-check + detach + bool return only.
//!
//! Observer discipline (M2 declared+landed): blocking client, rustls,
//! timeout 5s, retries 0 — the observer must not stall train. Every
//! transport failure (client build, connect/refused, timeout, non-2xx)
//! returns `Ok(false)`; the Python mirrors treat `false` as warn-only +
//! file fallback (`mirror.jsonl` + `manifests/<stem>.json`, layout frozen).
//! Only malformed arguments fail closed (`ValueError`) before any I/O.
//!
//! Endpoints (transport contract, new — not the frozen filter shape):
//! - `POST {base}/runs/{run}/scalars` with
//!   `{"run_id": ..., "step": N, "metrics": {...allowlisted...}}`
//! - `POST {base}/runs/{run}/checkpoints` with
//!   `{"run_id": ..., "checkpoint": name, "manifest": {...}}`
//!
//! Concurrency pattern (matches `canon_rng.rs` + `search.rs`): validate
//! attached, ONE `py.detach(|| ...)` around the whole blocking POST
//! (fan-out/join never holds the interpreter), `MutexExt::lock_py_attached`
//! for the stats mutex, frozen surface only (free functions; no pyclass,
//! no shared cells beyond the attach-guarded stats mutex). The blocking
//! client is built per call (no persistent pool): the mirror posts at most
//! once per checkpoint, so a cached client would buy nothing and would add
//! shared-state lifetime the bridge does not need.
//!
//! Single-cdylib tree: registers as the `mirror` submodule
//! (`hydra2_replay_rs.mirror` today, `hydra_bridge._native.mirror` once the
//! Phase-6 maturin `module-name` cutover lands) via `register`, mirroring
//! `canon_rng::register`. The legacy `hydra2_replay_rs` entry point is
//! untouched.

use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyModule;

/// Observer POST timeout: the mirror must not stall train (M2 contract).
pub const MIRROR_TIMEOUT_SECS: u64 = 5;

/// Fail-closed bounds so the observer cannot be turned into a memory sink.
pub const MAX_BASE_URL_LEN: usize = 1024;
pub const MAX_RUN_ID_LEN: usize = 256;
pub const MAX_NAME_LEN: usize = 256;
pub const MAX_BODY_LEN: usize = 1_048_576;

/// Cumulative transport observability (counts only, never payloads).
#[derive(Debug, Default, Clone, Copy)]
struct MirrorStats {
    posts: u64,
    ok: u64,
    fallback: u64,
}

static MIRROR_STATS: Mutex<MirrorStats> = Mutex::new(MirrorStats {
    posts: 0,
    ok: 0,
    fallback: 0,
});

fn bump(py: Python<'_>, ok: bool) {
    if let Ok(mut guard) = MIRROR_STATS.lock_py_attached(py) {
        guard.posts = guard.posts.saturating_add(1);
        if ok {
            guard.ok = guard.ok.saturating_add(1);
        } else {
            guard.fallback = guard.fallback.saturating_add(1);
        }
    }
}

/// Strip one trailing-slash run and require an http(s) base (pure, tested).
fn check_base(base: &str) -> Result<String, String> {
    if base.is_empty() || base.len() > MAX_BASE_URL_LEN {
        return Err(format!(
            "mirror base_url must be 1..={MAX_BASE_URL_LEN} bytes"
        ));
    }
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("mirror base_url must start with http:// or https://".to_string());
    }
    Ok(base.trim_end_matches('/').to_string())
}

/// Run ids travel as a path segment: reject separators/query/whitespace.
fn check_segment(label: &str, value: &str, max_len: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > max_len {
        return Err(format!("mirror {label} must be 1..={max_len} bytes"));
    }
    let bad = value
        .chars()
        .any(|c| c == '/' || c == '?' || c == '#' || c.is_whitespace() || (c as u32) < 0x20);
    if bad {
        return Err(format!("mirror {label} must not contain / ? # or whitespace"));
    }
    Ok(())
}

/// Transport bodies are JSON objects built Python-side (already filtered).
fn check_json_body(label: &str, body: &str) -> Result<(), String> {
    if body.is_empty() || body.len() > MAX_BODY_LEN {
        return Err(format!("mirror {label} body must be 1..={MAX_BODY_LEN} bytes"));
    }
    let trimmed = body.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        Ok(())
    } else {
        Err(format!("mirror {label} body must be a JSON object"))
    }
}

fn scalars_url(base: &str, run_id: &str) -> Result<String, String> {
    let root = check_base(base)?;
    check_segment("run_id", run_id, MAX_RUN_ID_LEN)?;
    Ok(format!("{root}/runs/{run_id}/scalars"))
}

fn checkpoints_url(base: &str, run_id: &str) -> Result<String, String> {
    let root = check_base(base)?;
    check_segment("run_id", run_id, MAX_RUN_ID_LEN)?;
    Ok(format!("{root}/runs/{run_id}/checkpoints"))
}

fn push_quoted(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Fixed transport envelope (pure, byte-tested): `metrics_json` splices
/// verbatim — it is a complete JSON object validated by `check_json_body`.
fn scalars_envelope(run_id: &str, step: u64, metrics_json: &str) -> String {
    let body = metrics_json.trim();
    let mut out = String::with_capacity(run_id.len() + body.len() + 48);
    out.push_str("{\"run_id\":");
    push_quoted(&mut out, run_id);
    out.push_str(",\"step\":");
    out.push_str(&step.to_string());
    out.push_str(",\"metrics\":");
    out.push_str(body);
    out.push('}');
    out
}

/// Fixed transport envelope (pure, byte-tested): `manifest_json` splices
/// verbatim — it is a complete JSON object validated by `check_json_body`.
fn checkpoint_envelope(run_id: &str, name: &str, manifest_json: &str) -> String {
    let body = manifest_json.trim();
    let mut out = String::with_capacity(run_id.len() + name.len() + body.len() + 64);
    out.push_str("{\"run_id\":");
    push_quoted(&mut out, run_id);
    out.push_str(",\"checkpoint\":");
    push_quoted(&mut out, name);
    out.push_str(",\"manifest\":");
    out.push_str(body);
    out.push('}');
    out
}

/// ONE blocking POST, retries 0 (observer must not stall train). `false`
/// on every transport failure: client build, connect/refused/DNS, timeout,
/// or non-2xx. Never raises, never retries.
fn do_post(url: &str, body: &str) -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(MIRROR_TIMEOUT_SECS))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    match client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_owned())
        .send()
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// POST allowlisted scalars to `{base}/runs/{run}/scalars` (blocking,
/// timeout 5s, retries 0). Returns `true` on 2xx, `false` on every
/// transport failure (caller runs the warn-only file fallback). Fails
/// closed (`ValueError`) only on malformed arguments, before any I/O.
#[pyfunction]
#[pyo3(signature = (base_url, run_id, step, metrics_json))]
fn post_scalars(
    py: Python<'_>,
    base_url: String,
    run_id: String,
    step: u64,
    metrics_json: String,
) -> PyResult<bool> {
    let url = scalars_url(&base_url, &run_id).map_err(PyValueError::new_err)?;
    check_json_body("metrics", &metrics_json).map_err(PyValueError::new_err)?;
    let body = scalars_envelope(&run_id, step, &metrics_json);
    let ok = py.detach(|| do_post(&url, &body));
    bump(py, ok);
    Ok(ok)
}

/// POST a checkpoint manifest to `{base}/runs/{run}/checkpoints`
/// (blocking, timeout 5s, retries 0). Tensor bytes never cross: `name` is
/// the checkpoint file name and `manifest_json` its JSON snapshot. Returns
/// `true` on 2xx, `false` on every transport failure (caller runs the
/// warn-only file fallback). Fails closed (`ValueError`) only on malformed
/// arguments, before any I/O.
#[pyfunction]
#[pyo3(signature = (base_url, run_id, name, manifest_json))]
fn post_checkpoint(
    py: Python<'_>,
    base_url: String,
    run_id: String,
    name: String,
    manifest_json: String,
) -> PyResult<bool> {
    let url = checkpoints_url(&base_url, &run_id).map_err(PyValueError::new_err)?;
    check_segment("checkpoint name", &name, MAX_NAME_LEN).map_err(PyValueError::new_err)?;
    check_json_body("manifest", &manifest_json).map_err(PyValueError::new_err)?;
    let body = checkpoint_envelope(&run_id, &name, &manifest_json);
    let ok = py.detach(|| do_post(&url, &body));
    bump(py, ok);
    Ok(ok)
}

/// Observer POST timeout in seconds (M2 contract, asserted without I/O).
#[pyfunction]
fn timeout_secs() -> u64 {
    MIRROR_TIMEOUT_SECS
}

/// Transport observability: `(posts, ok, fallback)` (counts only).
#[pyfunction]
fn mirror_stats(py: Python<'_>) -> PyResult<(u64, u64, u64)> {
    let guard = MIRROR_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("mirror stats mutex poisoned"))?;
    Ok((guard.posts, guard.ok, guard.fallback))
}

/// Register the `mirror` submodule (mirrors `canon_rng::register`):
/// compute detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "mirror")?;
    sub.add_function(wrap_pyfunction!(post_scalars, &sub)?)?;
    sub.add_function(wrap_pyfunction!(post_checkpoint, &sub)?)?;
    sub.add_function(wrap_pyfunction!(timeout_secs, &sub)?)?;
    sub.add_function(wrap_pyfunction!(mirror_stats, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_is_five_seconds_no_retries() {
        assert_eq!(MIRROR_TIMEOUT_SECS, 5);
    }

    #[test]
    fn endpoint_urls_trim_trailing_slash() {
        assert_eq!(
            scalars_url("http://localhost:8080/", "run-1"),
            Ok("http://localhost:8080/runs/run-1/scalars".to_string())
        );
        assert_eq!(
            checkpoints_url("https://mlflow.local///", "run-1"),
            Ok("https://mlflow.local/runs/run-1/checkpoints".to_string())
        );
    }

    #[test]
    fn endpoint_urls_reject_non_http_and_bad_run() {
        assert!(scalars_url("ftp://host/x", "run-1").is_err());
        assert!(scalars_url("localhost:8080", "run-1").is_err());
        assert!(scalars_url("", "run-1").is_err());
        assert!(scalars_url("http://host", "").is_err());
        assert!(scalars_url("http://host", "run/1").is_err());
        assert!(scalars_url("http://host", "run 1").is_err());
        assert!(checkpoints_url("http://host", "run?id").is_err());
    }

    #[test]
    fn bodies_must_be_json_objects_within_bounds() {
        assert!(check_json_body("metrics", "{\"total\":0.5}").is_ok());
        assert!(check_json_body("metrics", "  {\"total\":0.5}\n").is_ok());
        assert!(check_json_body("metrics", "").is_err());
        assert!(check_json_body("metrics", "[1,2]").is_err());
        assert!(check_json_body("metrics", "null").is_err());
        assert!(check_json_body("metrics", "{\"unclosed\":").is_err());
    }

    #[test]
    fn scalars_envelope_is_byte_exact() {
        assert_eq!(
            scalars_envelope("run-1", 7, "{\"total\":0.5}"),
            "{\"run_id\":\"run-1\",\"step\":7,\"metrics\":{\"total\":0.5}}"
        );
    }

    #[test]
    fn checkpoint_envelope_is_byte_exact() {
        assert_eq!(
            checkpoint_envelope("run-1", "ckpt-000003.pt", "{\"global_update\":3}"),
            "{\"run_id\":\"run-1\",\"checkpoint\":\"ckpt-000003.pt\",\
             \"manifest\":{\"global_update\":3}}"
        );
    }

    #[test]
    fn envelopes_escape_quotes_and_backslashes() {
        assert_eq!(
            scalars_envelope("run-\"1\\", 0, "{}"),
            "{\"run_id\":\"run-\\\"1\\\\\",\"step\":0,\"metrics\":{}}"
        );
    }
}
