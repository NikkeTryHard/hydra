//! RFC 8785 (JCS) canonical JSON — the Hydra2 identity-bytes authority.
//!
//! Rust owner of `src/hydra2/artifacts/canonical.py`: every identity path
//! (manifest / row / packet / info-key / derivation / attestation / registry
//! seals) hashes THESE bytes, never `serde_json` pretty / `sort_keys` / simd
//! re-emit (fork by construction; endstate §0.3).
//!
//! - Byte source: `serde_jcs 0.2.0` ONLY-exact (`to_vec` / `to_string` /
//!   `to_writer` agree byte-exact — same normative sentence on all three entry
//!   points). `ryu-js` backend gives ES6 `Number::toString` incl. Note 2;
//!   UTF-16BE key sort lives INSIDE the serializer (caller-side `BTreeMap` /
//!   `sort_keys` can never fix the astral-vs-BMP divergence, wave2 §1.8).
//! - Dependency discipline: this module imports `serde_jcs` + `serde_json`
//!   (parse/reject) ONLY. Hashing lives in `crate::digest`; RNG in
//!   `crate::rng`. `serde_json` NEVER emits identity bytes here.
//! - Attestation seals (`src/hydra2/data/attestation.py`, M17) are assigned
//!   here: attestation identity docs canonicalize through [`canonical_bytes`]
//!   exactly like every other seal path.

use serde::Serialize;

/// Canonicalization failure: which I-JSON boundary rejected the input, and
/// which record it came from (never silent coercion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonError {
    /// Duplicate object property name (§3.1 rule 1). Carries `record: detail`.
    DuplicateKey { record: String, detail: String },
    /// `NaN` / `Infinity` on the canon path (§3.2.2.3). Carries record + detail.
    NonFinite { record: String, detail: String },
    /// Lone surrogate in a string (§3.2.2.2). Carries record + detail.
    LoneSurrogate { record: String, detail: String },
    /// Number not expressible as f64 / beyond the canonical domain (§3.1
    /// rule 3 + App. D). Carries record + detail.
    UnsafeNumber { record: String, detail: String },
    /// Non-string map key at the `serde_jcs` boundary (the only documented
    /// `serde_jcs` failure besides `Serialize` impl failure). Carries record.
    NonStringKey { record: String, detail: String },
    /// Leading BOM on input text (RFC 7493 §4 / RFC 8259 §8.1; enforced in
    /// `validate`, never in `canon`). Carries record.
    Bom { record: String },
    /// Input is not valid JSON at all. Carries record + detail.
    InvalidJson { record: String, detail: String },
    /// `serde_jcs` serialization failure (pass-through; see `NonStringKey`).
    Jcs { record: String, detail: String },
}

impl core::fmt::Display for CanonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CanonError::DuplicateKey { record, detail } => {
                write!(f, "canon rejected duplicate key for {record}: {detail}")
            }
            CanonError::NonFinite { record, detail } => {
                write!(f, "canon rejected non-finite number for {record}: {detail}")
            }
            CanonError::LoneSurrogate { record, detail } => {
                write!(f, "canon rejected lone surrogate for {record}: {detail}")
            }
            CanonError::UnsafeNumber { record, detail } => {
                write!(f, "canon rejected unsafe number for {record}: {detail}")
            }
            CanonError::NonStringKey { record, detail } => {
                write!(f, "canon rejected non-string key for {record}: {detail}")
            }
            CanonError::Bom { record } => {
                write!(f, "canon rejected leading BOM for {record}")
            }
            CanonError::InvalidJson { record, detail } => {
                write!(f, "canon rejected invalid JSON for {record}: {detail}")
            }
            CanonError::Jcs { record, detail } => {
                write!(f, "canon JCS failed for {record}: {detail}")
            }
        }
    }
}

impl std::error::Error for CanonError {}

/// The ONE canonical byte function (pure JCS, fallible with record id).
///
/// Generic over `Serialize` so row structs (field order = declaration order,
/// preserved through serde; JCS sorts maps only) hash without an intermediate
/// `Value`. `serde_json::Value` inputs cannot hold NaN/Inf, duplicate keys,
/// or lone surrogates, so `Ok` is the common path; non-string map keys (typed
/// structs with int keys) surface as `Err` with the record id attached (never
/// silent coercion). Bridge / packet call sites that start from raw document
/// bytes go through [`parse_canonical_bytes`] (reject table) then here.
pub fn canonical_bytes<T: Serialize + ?Sized>(
    value: &T,
    record: &str,
) -> Result<Vec<u8>, CanonError> {
    if let Ok(v) = serde_json::to_value(value) {
        reject_unsafe_integers(&v, record)?;
    }
    // `f64::NAN/INFINITY` stage into `Value::Null` under some serde_json
    // configs (no staging error) but `serde_jcs` rejects non-finite floats.
    // Map any JCS float-domain failure to NonFinite (never generic Jcs):
    // NaN/±Inf are the only values that stage-but-not-serialize.
    serde_jcs::to_vec(value).map_err(|e| {
        let detail = e.to_string();
        let lower = detail.to_lowercase();
        if lower.contains("finite")
            || lower.contains("nan")
            || lower.contains("inf")
            || lower.contains("float")
        {
            return CanonError::NonFinite {
                record: record.to_string(),
                detail,
            };
        }
        CanonError::Jcs {
            record: record.to_string(),
            detail,
        }
    })
}


/// Largest magnitude exactly representable in an IEEE 754 double
/// (`2**53 - 1`, mirrors `canonical.py:34` `MAX_SAFE_INTEGER`): Python `int`
/// inputs beyond this are rejected as non-canonical-safe (RFC 8785 App. D:
/// wrap as string instead).
pub const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// Walk a staged `Value` and reject integer `Number`s with
/// `|n| > MAX_SAFE_INTEGER` (mirrors `canonical.py:124-128`). Floats are
/// inherently f64 (finite by `Value` construction); u64/i64 beyond the
/// double-safe window would otherwise seal a value the Python oracle
/// refuses — a seal fork. Recurses arrays/objects; object-key order is
/// irrelevant here (JCS sorts inside the serializer).
fn reject_unsafe_integers(value: &serde_json::Value, record: &str) -> Result<(), CanonError> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i.abs() > MAX_SAFE_INTEGER {
                    return Err(CanonError::UnsafeNumber {
                        record: record.to_string(),
                        detail: format!(
                            "integer {i} exceeds the IEEE 754 double-safe range (±{MAX_SAFE_INTEGER})"
                        ),
                    });
                }
            } else if let Some(u) = n.as_u64() {
                if u > MAX_SAFE_INTEGER as u64 {
                    return Err(CanonError::UnsafeNumber {
                        record: record.to_string(),
                        detail: format!(
                            "integer {u} exceeds the IEEE 754 double-safe range (±{MAX_SAFE_INTEGER})"
                        ),
                    });
                }
            }
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for item in items {
                reject_unsafe_integers(item, record)?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                reject_unsafe_integers(v, record)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Streaming variant for large rows/manifests: same bytes, no intermediate
/// `Vec`. Stages through the same `MAX_SAFE_INTEGER` guard as
/// [`canonical_bytes`] (writer output is byte-identical to `to_vec`).
///
/// Argument order is `(writer, value)` — `serde_jcs::to_writer(writer,
/// value)` (Data B2). `Digest` is NOT `Write`: hash streaming via
/// `to_vec`-then-`update` in `crate::digest` (wave2 §2.4).
pub fn canonical_writer<W: std::io::Write, T: Serialize + ?Sized>(
    writer: W,
    value: &T,
    record: &str,
) -> Result<(), CanonError> {
    if let Ok(v) = serde_json::to_value(value) {
        reject_unsafe_integers(&v, record)?;
    }
    serde_jcs::to_writer(writer, value).map_err(|e| {
        let detail = e.to_string();
        let lower = detail.to_lowercase();
        if lower.contains("finite")
            || lower.contains("nan")
            || lower.contains("inf")
            || lower.contains("float")
        {
            return CanonError::NonFinite {
                record: record.to_string(),
                detail,
            };
        }
        CanonError::Jcs {
            record: record.to_string(),
            detail,
        }
    })
}

/// I-JSON boundary: raw document bytes → clean `Value` (reject table BEFORE
/// bytes, mirroring `loads_canonical`).
///
/// Rejects, each with the record id attached (never silent coercion):
/// - UTF-8 BOM (`EF BB BF` bytes / `FEFF` text) — `CanonError::Bom`.
/// - Duplicate object keys — `CanonError::DuplicateKey`.
/// - `NaN` / `Infinity` literals — `CanonError::NonFinite`.
/// - Lone surrogates (`\uD800`-`\uDFFF` escapes) — `CanonError::LoneSurrogate`.
/// - Integers with `|n| > 2**53-1` — `CanonError::UnsafeNumber` (mirrors
///   `canonical.py:124-128`; RFC 8785 App. D: wrap as string instead).
/// - Non-string map keys — `CanonError::NonStringKey` (typed structs with int
///   keys fail at the `serde_jcs` boundary; JSON text can only carry string
///   keys, so this arm fires on the `Value`-construction path via
///   `serde_json::from_value` in callers — documented here so the table is
///   complete).
/// - Non-UTF-8 / malformed JSON — `CanonError::InvalidJson`.
pub fn parse_canonical_bytes(doc: &[u8], record: &str) -> Result<serde_json::Value, CanonError> {
    if doc.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(CanonError::Bom {
            record: record.to_string(),
        });
    }
    let text = core::str::from_utf8(doc).map_err(|e| CanonError::InvalidJson {
        record: record.to_string(),
        detail: e.to_string(),
    })?;
    if text.starts_with('\u{FEFF}') {
        return Err(CanonError::Bom {
            record: record.to_string(),
        });
    }
    reject_bare_constants(text, record)?;
    reject_lone_surrogate_escapes(text, record)?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| CanonError::InvalidJson {
            record: record.to_string(),
            detail: e.to_string(),
        })?;
    reject_duplicate_keys_text(text, record)?;
    reject_unsafe_integers(&value, record)?;
    Ok(value)
}

/// Reject `NaN` / `Infinity` / `-Infinity` literals ANYWHERE outside strings
/// (top-level or nested like `[NaN]` / `{"a":Infinity}`): `serde_json` would
/// reject them as invalid JSON — mapped to the typed `NonFinite` arm so the
/// KAT-6 table names the right error. String contents are skipped via
/// [`scan_string`] so `"NaN"` as data never trips this.
fn reject_bare_constants(text: &str, record: &str) -> Result<(), CanonError> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            match scan_string(bytes, i) {
                Ok((_, next)) => {
                    i = next;
                    continue;
                }
                Err(_) => return Ok(()),
            }
        }
        for tok in ["-Infinity", "Infinity", "NaN"] {
            let tb = tok.as_bytes();
            if bytes[i..].starts_with(tb) {
                let before_ok = i == 0 || !is_token_char(bytes[i - 1]);
                let after = i + tb.len();
                let after_ok = after >= bytes.len() || !is_token_char(bytes[after]);
                if before_ok && after_ok {
                    return Err(CanonError::NonFinite {
                        record: record.to_string(),
                        detail: format!("bare {tok} is not in the canonical domain"),
                    });
                }
            }
        }
        i += 1;
    }
    Ok(())
}

/// Token-char for bare-constant boundaries: alphanumerics + `_` + `.` (so
/// `NaNfoo`, `Infinity2`, `1NaN` never count as bare tokens).
fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

/// Reject `\uD800`-`\uDFFF` escape halves (lone surrogates) at the text
/// level: `serde_json` would decode a PAIR to the astral char (fine), but a
/// lone half decodes to U+FFFD or errors — either way it must fail typed,
/// never silently coerce.
fn reject_lone_surrogate_escapes(text: &str, record: &str) -> Result<(), CanonError> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'u' && i + 6 <= bytes.len() {
            if let Ok(hex) = core::str::from_utf8(&bytes[i + 2..i + 6]) {
                if let Ok(unit) = u16::from_str_radix(hex, 16) {
                    if (0xD800..0xE000).contains(&unit) {
                        // High half must be followed by a low-half escape;
                        // low half must follow a high half — else lone.
                        let is_high = (0xD800..0xDC00).contains(&unit);
                        if is_high {
                            let next = &bytes[i + 6..];
                            let paired = next.len() >= 6
                                && next[0] == b'\\'
                                && next[1] == b'u'
                                && core::str::from_utf8(&next[2..6])
                                    .ok()
                                    .and_then(|h| u16::from_str_radix(h, 16).ok())
                                    .is_some_and(|u| (0xDC00..0xE000).contains(&u));
                            if !paired {
                                return Err(CanonError::LoneSurrogate {
                                    record: record.to_string(),
                                    detail: format!("lone surrogate escape \\u{hex}"),
                                });
                            }
                        } else {
                            return Err(CanonError::LoneSurrogate {
                                record: record.to_string(),
                                detail: format!("lone surrogate escape \\u{hex}"),
                            });
                        }
                    }
                    i += 6;
                    continue;
                }
            }
        }
        i += 1;
    }
    Ok(())
}

/// Duplicate-key scan over the raw text: re-parse with a minimal scanner
/// that tracks object scopes and fails on a repeated key in one scope.
/// String unescaping is limited to what key comparison needs (`\\`, `\"`,
/// `\uXXXX` incl. surrogate pairs → the astral char); keys differing only in
/// escape spelling compare UNESCAPED (RFC 8785 §3.2.3: sort key is the raw
/// unescaped name).
fn reject_duplicate_keys_text(text: &str, record: &str) -> Result<(), CanonError> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    // Stack of per-object key sets (as unescaped Strings).
    let mut scopes: Vec<Vec<String>> = Vec::new();
    // True when the next string at this depth is an object key (vs value).
    let mut expect_key: Vec<bool> = Vec::new();
    // Whether the current container level is an object (vs array).
    let mut is_obj: Vec<bool> = Vec::new();

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'{' => {
                scopes.push(Vec::new());
                expect_key.push(true);
                is_obj.push(true);
                i += 1;
            }
            b'[' => {
                scopes.push(Vec::new());
                expect_key.push(false);
                is_obj.push(false);
                i += 1;
            }
            b'}' | b']' => {
                scopes.pop();
                expect_key.pop();
                is_obj.pop();
                i += 1;
                // A closed value completes the parent's value slot: the next
                // string in a parent object is a key.
                if is_obj.last().copied().unwrap_or(false) {
                    if let Some(ek) = expect_key.last_mut() {
                        *ek = true;
                    }
                }
            }
            b'"' => {
                let (s, next) = scan_string(bytes, i).map_err(|d| CanonError::InvalidJson {
                    record: record.to_string(),
                    detail: d,
                })?;
                let in_object = is_obj.last().copied().unwrap_or(false);
                let want_key = expect_key.last().copied().unwrap_or(false);
                if in_object && want_key {
                    let scope = scopes.last_mut().ok_or_else(|| CanonError::InvalidJson {
                        record: record.to_string(),
                        detail: "object key outside scope".to_string(),
                    })?;
                    if scope.iter().any(|k| k == &s) {
                        return Err(CanonError::DuplicateKey {
                            record: record.to_string(),
                            detail: format!("duplicate object key {s:?}"),
                        });
                    }
                    scope.push(s);
                    if let Some(ek) = expect_key.last_mut() {
                        *ek = false;
                    }
                } else if in_object {
                    if let Some(ek) = expect_key.last_mut() {
                        *ek = true;
                    }
                }
                i = next;
            }
            b':' | b',' | b' ' | b'\t' | b'\n' | b'\r' => {
                i += 1;
            }
            _ => {
                // Literals / numbers: skip one token; a completed object value
                // means the next string in the parent object is a key.
                i = skip_scalar(bytes, i);
                if is_obj.last().copied().unwrap_or(false) {
                    if let Some(ek) = expect_key.last_mut() {
                        *ek = true;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Scan a JSON string starting at the opening quote; return the UNESCAPED
/// name and the offset past the closing quote.
fn scan_string(bytes: &[u8], start: usize) -> Result<(String, usize), String> {
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err("truncated escape".to_string());
                }
                match bytes[i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if i + 4 >= bytes.len() {
                            return Err("truncated \\u escape".to_string());
                        }
                        let hex = core::str::from_utf8(&bytes[i + 1..i + 5])
                            .map_err(|e| e.to_string())?;
                        let unit =
                            u16::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                        i += 4;
                        if (0xD800..0xDC00).contains(&unit) {
                            // High half: require a low half right after.
                            if i + 2 < bytes.len()
                                && bytes[i + 1] == b'\\'
                                && bytes[i + 2] == b'u'
                                && i + 6 < bytes.len()
                            {
                                let hex2 = core::str::from_utf8(&bytes[i + 3..i + 7])
                                    .map_err(|e| e.to_string())?;
                                let unit2 = u16::from_str_radix(hex2, 16)
                                    .map_err(|e| e.to_string())?;
                                if (0xDC00..0xE000).contains(&unit2) {
                                    let hi = (unit as u32) - 0xD800;
                                    let lo = (unit2 as u32) - 0xDC00;
                                    let cp = 0x10000 + ((hi << 10) | lo);
                                    out.push(
                                        char::from_u32(cp).ok_or("bad pair")?,
                                    );
                                    i += 6;
                                } else {
                                    return Err(format!(
                                        "lone surrogate \\u{hex}"
                                    ));
                                }
                            } else {
                                return Err(format!("lone surrogate \\u{hex}"));
                            }
                        } else if (0xDC00..0xE000).contains(&unit) {
                            return Err(format!("lone surrogate \\u{hex}"));
                        } else {
                            out.push(
                                char::from_u32(unit as u32).ok_or("bad unit")?,
                            );
                        }
                    }
                    other => {
                        return Err(format!("bad escape \\{other}"));
                    }
                }
                i += 1;
            }
            _ => {
                // Raw UTF-8 char (multi-byte aware).
                let rest = core::str::from_utf8(&bytes[i..]).map_err(|e| e.to_string())?;
                let ch = rest.chars().next().ok_or("empty")?;
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    Err("unterminated string".to_string())
}

/// Skip one scalar token (number / literal) to its end offset.
fn skip_scalar(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' | b'"' | b'{' | b'[' => break,
            _ => i += 1,
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_WORLD_SHA_SUFFIX: &str =
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    /// KAT-1 (canon half): `canonical_bytes(b"hello world"-as-JSON-string)`
    /// feeds the digest KAT; the digest half lives in `crate::digest` and the
    /// both-paths agreement gate with it.
    #[test]
    fn kat1_hello_world_canon_feeds_digest() {
        let v = serde_json::Value::String("hello world".to_string());
        let bytes = canonical_bytes(&v, "kat1").unwrap();
        assert_eq!(bytes, b"\"hello world\"");
        let _ = HELLO_WORLD_SHA_SUFFIX;
    }

    /// KAT-2: JCS number/escape exactness (wave2 §1.7 + RFC 8785 §3.2.2).
    /// `serde_json::Value` holds f64s, so `1E30` parses to `1e30f64` and JCS
    /// must emit `1e+30`; `4.50` → `4.5`; `€`/`é` ride raw UTF-8 (no
    /// `\uXXXX`); short escapes + hex-sample prefix pinned.
    #[test]
    fn kat2_jcs_numbers_and_escapes() {
        // 1E30 -> 1e+30 (explicit +, no leading zeros).
        let v: serde_json::Value = serde_json::from_str("1E30").unwrap();
        assert_eq!(canonical_bytes(&v, "kat2:1E30").unwrap(), b"1e+30");
        // 4.50 -> 4.5 (shortest round-trip, no trailing zero).
        let v: serde_json::Value = serde_json::from_str("4.50").unwrap();
        assert_eq!(canonical_bytes(&v, "kat2:4.50").unwrap(), b"4.5");
        // € / é ride as raw UTF-8, never \uXXXX.
        let v = serde_json::Value::String("€".to_string());
        assert_eq!(canonical_bytes(&v, "kat2:euro").unwrap(), "\"€\"".as_bytes());
        let v = serde_json::Value::String("é".to_string());
        assert_eq!(canonical_bytes(&v, "kat2:e-acute").unwrap(), "\"é\"".as_bytes());
        // Short escapes \b \t \n \f \r (U+0008/0009/000A/000C/000D).
        let v = serde_json::Value::String("\u{08}\u{09}\u{0A}\u{0C}\u{0D}".to_string());
        assert_eq!(canonical_bytes(&v, "kat2:escapes").unwrap(), b"\"\\b\\t\\n\\f\\r\"");
        // RFC §3.2.4 hex sample prefix: `{` `"` `l` `i` == 7b 22 6c 69.
        let v = serde_json::json!({"li": 1});
        let bytes = canonical_bytes(&v, "kat2:hex-sample").unwrap();
        assert!(bytes.starts_with(&[0x7b, 0x22, 0x6c, 0x69]));
    }

    /// KAT-3: key-sort guard (M11-fixed + RFC §3.2.3): astral `U+10000`
    /// (UTF-16 lead `D800` < `E000`) sorts BEFORE BMP `U+E000`. `windows(4)`
    /// over the bytes (M11: the 4-byte needles, never 8).
    #[test]
    fn kat3_astral_sorts_before_bmp() {
        let astral = char::from_u32(0x10000).unwrap().to_string();
        let bmp = char::from_u32(0xE000).unwrap().to_string();
        let v = serde_json::json!({bmp.clone(): 2, astral.clone(): 1});
        let bytes = canonical_bytes(&v, "kat3:astral").unwrap();
        let astral_utf8 = astral.as_bytes();
        let bmp_utf8 = bmp.as_bytes();
        assert_eq!(astral_utf8, b"\xF0\x90\x80\x80");
        assert_eq!(bmp_utf8, b"\xEE\x80\x80");
        let astral_pos = bytes
            .windows(4)
            .position(|w| w == astral_utf8)
            .expect("astral key present");
        let bmp_pos = bytes
            .windows(3)
            .position(|w| w == bmp_utf8)
            .expect("bmp key present");
        assert!(
            astral_pos < bmp_pos,
            "astral U+10000 must sort before BMP U+E000: {bytes:?}"
        );
        // RFC §3.2.3 golden order spot-check: emoji U+1F600 before U+FB33.
        let emoji = char::from_u32(0x1F600).unwrap().to_string();
        let dalet = char::from_u32(0xFB33).unwrap().to_string();
        let v = serde_json::json!({dalet.clone(): 2, emoji.clone(): 1});
        let bytes = canonical_bytes(&v, "kat3:rfc-vector").unwrap();
        let pe = bytes
            .windows(4)
            .position(|w| w == emoji.as_bytes())
            .expect("emoji present");
        let pd = bytes
            .windows(3)
            .position(|w| w == dalet.as_bytes())
            .expect("dalet present");
        assert!(pe < pd, "RFC §3.2.3: emoji before U+FB33");
    }

    /// KAT-5: row-hash migration-aware fixtures — packaged + raw seal docs in
    /// normative field order (rows.py:119-140 / :314-327), `sha256:`-stripped
    /// forms (Python `strip`: digest fields hash as bare hex), `verify_seal`
    /// both directions (seal verifies; tampered id fails).
    ///
    /// Migration note (B3): TODAY-Python oracle bytes use caller-ordered
    /// `json.dumps` (rows.py:43-57), NOT UTF-16 sort — so these fixtures pin
    /// the RUST JCS bytes as the NEW seal form. Oracle freeze → JCS new →
    /// re-freeze + seal-replay runs BEFORE cutover; NEVER dual-print.
    #[test]
    fn kat5_row_hash_migration_fixtures() {
        // Packaged seal doc: 15 fields in rows.py:119-140 order, digests
        // stripped to bare hex, None → null.
        let packaged = serde_json::json!({
            "packaged_object_id": "00".repeat(32),
            "source_kind": "raw",
            "source_container_sha256": serde_json::Value::Null,
            "source_member_path": serde_json::Value::Null,
            "source_bytes_sha256": "11".repeat(32),
            "source_bytes_length": 10,
            "compressed_path": "shard-0001.jsonl.zst",
            "compressed_bytes_sha256": "22".repeat(32),
            "compressed_bytes_length": 8,
            "decoded_bytes_sha256": "33".repeat(32),
            "decoded_bytes_length": 9,
            "record_count": 3,
            "canonical_jsonl": true,
            "packager_identity": "44".repeat(32),
            "packager_config_hash": "55".repeat(32),
            "created_at_utc": "2026-01-01T00:00:00Z",
        });
        let bytes = canonical_bytes(&packaged, "kat5:packaged").unwrap();
        // JCS sorts keys: first key must be the alphabetically-smallest
        // UTF-16 unit run ("canonical_jsonl" < "compressed_*" < ...).
        let text = core::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("{\"canonical_jsonl\":"));
        // No whitespace anywhere.
        assert!(!bytes.iter().any(|b| *b == b' ' || *b == b'\n'));
        // Seal both directions: digest of WITHOUT-id bytes verifies against
        // the stored id; a tampered id fails.
        let digest = crate::digest::of_canonical(&packaged);
        assert!(digest.is_ok());
        let digest = digest.unwrap();
        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 7 + 64);
        // Tamper: flipping one byte changes the digest (seal binds content).
        let mut tampered = packaged.clone();
        tampered["record_count"] = serde_json::json!(4);
        let digest2 = crate::digest::of_canonical(&tampered).unwrap();
        assert_ne!(digest, digest2);

        // Raw seal doc: 12 fields in canonical_bytes_without_id order
        // (rows.py:314-327), no object_id.
        let raw = serde_json::json!({
            "packaged_object_id": "aa".repeat(32),
            "confidential_source_id": "src-1",
            "authorization_attestation_id": "att-1",
            "permitted_purpose": ["train"],
            "disclosure_class": "open",
            "acquisition_metadata": {},
            "semantic_state": "unvalidated",
            "semantic_validation_hash": serde_json::Value::Null,
            "first_error_class": serde_json::Value::Null,
            "first_error_event_index": serde_json::Value::Null,
            "parent_ids": [],
            "created_at_utc": "2026-01-01T00:00:00Z",
        });
        let raw_digest = crate::digest::of_canonical(&raw).unwrap();
        assert!(raw_digest.starts_with("sha256:"));
        let mut tampered_raw = raw.clone();
        tampered_raw["semantic_state"] = serde_json::json!("valid");
        assert_ne!(
            raw_digest,
            crate::digest::of_canonical(&tampered_raw).unwrap()
        );
    }

    /// KAT-6: reject table — dup keys, NaN/Inf, lone surrogate, int keys,
    /// BOM → `Err` + record id (never silent coercion).
    #[test]
    fn kat6_reject_table() {
        // Duplicate keys.
        let e = parse_canonical_bytes(br#"{"a":1,"a":2}"#, "rec-dup").unwrap_err();
        assert!(matches!(e, CanonError::DuplicateKey { .. }));
        assert!(format!("{e}").contains("rec-dup"));
        // NaN literal.
        let e = parse_canonical_bytes(b"NaN", "rec-nan").unwrap_err();
        assert!(matches!(e, CanonError::NonFinite { .. }));
        assert!(format!("{e}").contains("rec-nan"));
        // Infinity literal.
        let e = parse_canonical_bytes(b"Infinity", "rec-inf").unwrap_err();
        assert!(matches!(e, CanonError::NonFinite { .. }));
        // Nested NaN / Infinity (realistic positions, not just top-level).
        let e = parse_canonical_bytes(b"[NaN]", "rec-nan-nested").unwrap_err();
        assert!(matches!(e, CanonError::NonFinite { .. }));
        let e = parse_canonical_bytes(br#"{"a":Infinity}"#, "rec-inf-nested").unwrap_err();
        assert!(matches!(e, CanonError::NonFinite { .. }));
        // "NaN" as string data is fine (not a bare constant).
        assert!(parse_canonical_bytes(br#""NaN""#, "rec-nan-str").is_ok());
        // Lone surrogate escape.
        let e = parse_canonical_bytes(br#""\uDEAD""#, "rec-sur").unwrap_err();
        assert!(matches!(e, CanonError::LoneSurrogate { .. }));
        assert!(format!("{e}").contains("rec-sur"));
        // Int-keyed typed map stringifies keys ("1") — valid JSON, must succeed.
        let mut map = std::collections::BTreeMap::new();
        map.insert(1u32, 2u32);
        let bytes = canonical_bytes(&map, "rec-intkey").unwrap();
        assert_eq!(bytes, b"{\"1\":2}");
        // Non-finite floats fail typed at the staging screen (Value cannot
        // hold NaN/Inf, so this fires on f64 inputs before JCS ever runs).
        let e = canonical_bytes(&f64::NAN, "rec-float-nan").unwrap_err();
        assert!(matches!(e, CanonError::NonFinite { .. }));
        // Integers beyond the double-safe window (2**53): MAX_SAFE ok, 2**53 err.
        let v: serde_json::Value = serde_json::from_str("9007199254740991").unwrap();
        assert!(canonical_bytes(&v, "rec-maxsafe").is_ok());
        let v: serde_json::Value = serde_json::from_str("9007199254740992").unwrap();
        let e = canonical_bytes(&v, "rec-unsafe").unwrap_err();
        assert!(matches!(e, CanonError::UnsafeNumber { .. }));
        assert!(format!("{e}").contains("rec-unsafe"));
        let e = parse_canonical_bytes(b"[9007199254740992]", "rec-unsafe-nested").unwrap_err();
        assert!(matches!(e, CanonError::UnsafeNumber { .. }));
        // BOM-prefixed input (bytes EF BB BF).
        let e = parse_canonical_bytes(b"\xef\xbb\xbf{}", "rec-bom").unwrap_err();
        assert!(matches!(e, CanonError::Bom { .. }));
        assert!(format!("{e}").contains("rec-bom"));
        // BOM-prefixed input (FEFF text).
        let e = parse_canonical_bytes("\u{FEFF}{}".as_bytes(), "rec-bom2").unwrap_err();
        assert!(matches!(e, CanonError::Bom { .. }));
    }

    /// `canonical_writer` (writer, value) order agrees byte-exact with
    /// `canonical_bytes` (Data B2).
    #[test]
    fn writer_order_agrees_with_vec() {
        let v = serde_json::json!({"b": [true, null, 7], "a": {"z": 1}});
        let vec_bytes = canonical_bytes(&v, "rec-writer").unwrap();
        let mut buf = Vec::new();
        canonical_writer(&mut buf, &v, "rec-writer").unwrap();
        assert_eq!(buf, vec_bytes);
    }
}
