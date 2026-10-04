//! rl/lifecycle_common: shared error + closed JSON gates for cold lifecycle.
//!
//! DAG: `serde_json` + `hydra_feed::{canon, digest}` ONLY. Every lifecycle
//! submodule funnels fallible construction through here so gates read
//! identically and no second implementation exists. FORBIDS: domain logic,
//! asset IO beyond the guarded repo read, PyO3, wall-clock.

use serde_json::{Map, Value};

/// Native failure (fail-closed text; the FFI layer raises `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleError(pub String);

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl lifecycle: {}", self.0)
    }
}

impl std::error::Error for LifecycleError {}

pub(crate) fn lc_err(detail: impl Into<String>) -> LifecycleError {
    LifecycleError(detail.into())
}

/// Canonical `sha256:<64 lowercase hex>` shape (oracle `_DIGEST_RE`).
pub(crate) fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

pub(crate) fn req_digest(value: Option<&Value>, name: &str) -> Result<String, LifecycleError> {
    match value {
        Some(Value::String(text)) if is_digest_shape(text) => Ok(text.clone()),
        Some(Value::String(text)) => Err(lc_err(format!(
            "rl lifecycle {name} must be sha256:<64 hex>, got {text:?}"
        ))),
        _ => Err(lc_err(format!("rl lifecycle {name} must be a digest string"))),
    }
}

pub(crate) fn req_str(value: Option<&Value>, name: &str) -> Result<String, LifecycleError> {
    match value {
        Some(Value::String(text)) if !text.is_empty() => Ok(text.clone()),
        _ => Err(lc_err(format!("rl lifecycle {name} must be a nonempty string"))),
    }
}

/// Unsigned int slot; JSON floats and bools never pass (bool would stage
/// elsewhere — reject here like `contracts.rs:69-74`).
pub(crate) fn req_u64(value: Option<&Value>, name: &str) -> Result<u64, LifecycleError> {
    match value {
        Some(Value::Number(n)) if !n.is_f64() => n
            .as_u64()
            .ok_or_else(|| lc_err(format!("rl lifecycle {name} must be a nonnegative int"))),
        _ => Err(lc_err(format!("rl lifecycle {name} must be a nonnegative int"))),
    }
}

pub(crate) fn req_i64(value: Option<&Value>, name: &str) -> Result<i64, LifecycleError> {
    match value {
        Some(Value::Number(n)) if !n.is_f64() => n
            .as_i64()
            .ok_or_else(|| lc_err(format!("rl lifecycle {name} must be an int"))),
        _ => Err(lc_err(format!("rl lifecycle {name} must be an int"))),
    }
}

pub(crate) fn req_f64_finite(value: Option<&Value>, name: &str) -> Result<f64, LifecycleError> {
    match value {
        Some(Value::Number(n)) => {
            let v = n
                .as_f64()
                .ok_or_else(|| lc_err(format!("rl lifecycle {name} must be a finite number")))?;
            if !v.is_finite() {
                return Err(lc_err(format!("rl lifecycle {name} must be finite")));
            }
            Ok(v)
        }
        _ => Err(lc_err(format!("rl lifecycle {name} must be a finite number"))),
    }
}

pub(crate) fn req_exact_f64(
    value: Option<&Value>,
    name: &str,
    want: f64,
) -> Result<f64, LifecycleError> {
    let got = req_f64_finite(value, name)?;
    if got != want {
        return Err(lc_err(format!(
            "rl lifecycle frozen {name} must be exactly {want}, got {got}"
        )));
    }
    Ok(got)
}

/// Closed-object gate: unknown keys fail with the field name for adjudication.
pub(crate) fn check_closed_object(
    map: &Map<String, Value>,
    allowed: &[&str],
    what: &str,
) -> Result<(), LifecycleError> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(lc_err(format!("rl lifecycle {what} rejects unknown field {key:?}")));
        }
    }
    Ok(())
}


/// JCS digest through the shared feed printer (single owner, never reprinted).
pub(crate) fn canonical_digest(value: &Value, record: &str) -> Result<String, LifecycleError> {
    let bytes = hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| lc_err(format!("rl lifecycle {record} not canonical-safe: {e}")))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Root-relative repo read with escape rejection (mirrors the
/// `snapshot::join_root` locator guard; absolute paths and `..` fail).
pub(crate) fn read_repo_file(repo_dir: &str, relpath: &str) -> Result<Vec<u8>, LifecycleError> {
    if relpath.is_empty()
        || relpath.starts_with('/')
        || relpath.contains('\\')
        || relpath.split('/').any(|c| c == ".." || c == ".")
    {
        return Err(lc_err(format!("rl lifecycle asset path escapes: {relpath:?}")));
    }
    let mut path = std::path::PathBuf::from(repo_dir);
    path.push(relpath);
    std::fs::read(&path).map_err(|e| lc_err(format!("rl lifecycle asset unreadable {relpath}: {e}")))
}

/// Descriptor field accessors (single home: run/generation/control share
/// these; no duplicated impls across lifecycle shards).
pub(crate) fn descriptor_str(doc: &Map<String, Value>, name: &str) -> Result<String, LifecycleError> {
    req_str(doc.get(name), name)
}

pub(crate) fn descriptor_u64(doc: &Map<String, Value>, name: &str) -> Result<u64, LifecycleError> {
    req_u64(doc.get(name), name)
}

pub(crate) fn descriptor_doc<'a>(
    doc: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a Map<String, Value>, LifecycleError> {
    doc.get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| lc_err(format!("rl lifecycle descriptor lacks {name}")))
}
