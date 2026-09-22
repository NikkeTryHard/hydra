//! Digest and field helpers for transport rows (split from `rows` to hold the gate).
//!
//! Pure shape checks and normalizers; seal math stays in `crate::canon` and
//! hashing in `crate::digest`. Failure mode is `RowsError` fail-closed on bad
//! digests, never silent coercion; nullable non-strings pass through for the
//! constructor to reject in its two-step validation.
use std::collections::BTreeMap;

use super::rows::RowsError;
// ---------------------------------------------------------------------------
// Digest + field helpers (mirror rows.py:31-40 + :184-194)
// ---------------------------------------------------------------------------

/// `sha256:<64 lowercase hex>` shape check (mirrors `_require_digest`).
pub fn require_digest(text: &str, name: &str) -> Result<(), RowsError> {
    let Some(hex) = text.strip_prefix("sha256:") else {
        return Err(RowsError::BadDigest {
            detail: format!("{name} must be sha256:<64 hex>, got {text:?}"),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(RowsError::BadDigest {
            detail: format!("{name} must be sha256:<64 hex>, got {text:?}"),
        });
    }
    Ok(())
}

/// Bare-hex → `sha256:` normalization at parse (mirrors `norm`,
/// `rows.py:185-194`; compat with older caller-ordered seals). `None` passes through for nullable
/// fields; non-strings and malformed strings are kept as-is here and
/// rejected by the constructor validation (same two-step as Python).
pub(crate) fn norm_digest(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None => None,
        Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => {
            if s.starts_with("sha256:") {
                Some(s.clone())
            } else if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
                Some(format!("sha256:{s}"))
            } else {
                Some(s.clone())
            }
        }
        Some(other) => Some(other.to_string()),
    }
}

/// Strip `sha256:` for seal bytes (mirrors `PackagedObjectRow.strip`,
/// `rows.py:114-117`). Non-prefixed input passes through (validation
/// rejects it before any seal is minted).
pub(crate) fn strip_digest(text: &str) -> &str {
    text.strip_prefix("sha256:").unwrap_or(text)
}
pub(crate) fn get_u64(obj: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<u64, RowsError> {
    let v = obj.get(key).ok_or_else(|| RowsError::MissingKeys {
        detail: format!("{{{key}}}"),
    })?;
    match v {
        serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
        serde_json::Value::String(s) => s.parse::<u64>().map_err(|_| RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
        _ => Err(RowsError::BadField {
            detail: format!("{key} must be nonnegative int, got {v}"),
        }),
    }
}

pub(crate) fn get_str(obj: &BTreeMap<String, serde_json::Value>, key: &str) -> Result<String, RowsError> {
    match obj.get(key) {
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(v) => Err(RowsError::BadField {
            detail: format!("{key} must be str, got {v}"),
        }),
        None => Err(RowsError::MissingKeys {
            detail: format!("{{{key}}}"),
        }),
    }
}

pub(crate) fn get_opt_str(
    obj: &BTreeMap<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>, RowsError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(RowsError::BadField {
            detail: format!("{key} must be str or null, got {v}"),
        }),
    }
}
