//! Provenance authorities: the digests Rust mints instead of echoing `spec_hash`.
//!
//! Reference (read-only): `src/hydra2/data/replay_expand.py` (`_load_rules`,
//! `_adapter_hash`), `src/hydra2/engines/riichienv/identity.py`
//! (`ENGINE_IDENTITY`, pinned `riichienv==0.4.8`, adapter `1.0.0`),
//! `src/hydra2/engines/riichienv/adapter.py` (`_rules_identity`: published
//! artifact bytes win) and `src/hydra2/engines/riichienv/state.py`
//! (`rules_identity_hash`: envelope-canonical recompute fallback). Rule
//! order reused; code written fresh.
//!
//! Digest rules (Slice 5):
//! - `rules_hash` is sha256 over the PUBLISHED rules-file bytes (the pinned
//!   `configs/rules/tenhou_4p_hanchan_v1.json` baked in at compile time).
//!   Published bytes win exactly like the adapter's `_rules_identity`; the
//!   envelope-canonical recompute is the fallback Python keeps for manifests
//!   without a published artifact, never the authority here.
//! - `adapter_hash` is `of_canonical` over the machine-stable identity
//!   document (engine name/version + adapter version, minus
//!   environment/source), byte-identical to `_adapter_hash()`.
//! - `action_table_hash` is the table CONTENT digest (`table.digest` on the
//!   Python `_table` path), owned by [`crate::decisions::ActionTable`].
//!
//! Rust never writes parquet; these digests bind rows (actor + privileged)
//! so the Python parquet owner can pin them per `decision_id`.

use sha2::Digest;

use crate::hydra2_row::{canonical_json_bytes, digest_text};

/// Pinned engine name (`ENGINE_NAME`).
pub const ENGINE_NAME: &str = "riichienv";
/// Pinned engine version (`RIICHENV_VERSION_PIN`, decision D-003).
pub const ENGINE_VERSION: &str = "0.4.8";
/// Pinned adapter version (`ADAPTER_VERSION`).
pub const ADAPTER_VERSION: &str = "1.0.0";
/// Pinned rules manifest id (`RULES_ID`).
pub const RULES_ID: &str = "tenhou_4p_hanchan_v1";
/// Rules artifact envelope type (`hydra2.rules_manifest`).
pub const RULES_ARTIFACT_TYPE: &str = "hydra2.rules_manifest";

/// `rules_hash` of the pinned rules artifact (sha256 of the published file
/// bytes; verified against the Python oracle before landing Slice 5).
pub const PINNED_RULES_HASH: &str =
    "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b";
/// `adapter_hash` of the pinned engine identity (verified against Python
/// `_adapter_hash()` before landing Slice 5).
pub const PINNED_ADAPTER_HASH: &str =
    "sha256:61f4d0328b5fbf46c0bdd9331e573b11e0166b357e93e218e96a5152c7c71c33";
/// `action_table_hash` of the pinned v1 table (the CONTENT digest
/// `table.digest`, not the artifact-file bytes hash; verified against
/// Python `load_action_table(...).digest` before landing Slice 5).
pub const PINNED_ACTION_TABLE_HASH: &str =
    "sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8";

/// Baked rules-manifest bytes: the published v1 artifact, hermetic default.
pub const BAKED_RULES_MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/rules/tenhou_4p_hanchan_v1.json"
));

/// Loaded rules manifest: validated id plus its published-bytes digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RulesInfo {
    pub rules_id: String,
    pub rules_hash: String,
}

/// Machine-stable adapter digest (identity minus environment/source).
///
/// Mirrors `_adapter_hash`: `of_canonical` over `{engine, engine_version,
/// adapter_version}`. Fails closed nowhere (pure function of pins); byte
/// equality with Python is pinned by [`PINNED_ADAPTER_HASH`].
pub fn adapter_hash() -> String {
    let doc = serde_json::json!({
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
    });
    digest_text(&canonical_json_bytes(&doc))
}

/// Load + validate a rules-manifest document, digesting the raw bytes.
///
/// Mirrors `_load_rules` + `_rules_identity`: the document must be the
/// `hydra2.rules_manifest` envelope carrying the pinned `rules_id`; the
/// digest is sha256 over the PUBLISHED BYTES (never a recompute), so any
/// tampered byte yields a digest that no longer matches
/// [`PINNED_RULES_HASH`] (structural tampering fails closed here with an
/// error instead).
pub fn load_rules_manifest(text: &str) -> Result<RulesInfo, String> {
    let doc: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("rules manifest parse: {e}"))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| "rules manifest must be a JSON object".to_string())?;
    if obj.get("artifact_type").and_then(|v| v.as_str()) != Some(RULES_ARTIFACT_TYPE) {
        return Err(format!(
            "rules manifest artifact_type must be {RULES_ARTIFACT_TYPE:?}, got {:?}",
            obj.get("artifact_type")
        ));
    }
    if obj.get("compatibility").and_then(|v| v.as_str()) != Some("exact") {
        return Err(format!(
            "rules manifest compatibility must be \"exact\", got {:?}",
            obj.get("compatibility")
        ));
    }
    let payload = obj
        .get("payload")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "rules manifest payload must be an object".to_string())?;
    if payload.get("rules_id").and_then(|v| v.as_str()) != Some(RULES_ID) {
        return Err(format!(
            "rules manifest payload rules_id must be {RULES_ID:?}, got {:?}",
            payload.get("rules_id")
        ));
    }
    let sum = sha2::Sha256::digest(text.as_bytes());
    let mut rules_hash = String::with_capacity(7 + 64);
    rules_hash.push_str("sha256:");
    for b in sum {
        rules_hash.push_str(&format!("{b:02x}"));
    }
    Ok(RulesInfo {
        rules_id: RULES_ID.to_string(),
        rules_hash,
    })
}

/// Load the baked pinned manifest (the hermetic Slice-5 default).
pub fn load_baked_rules() -> Result<RulesInfo, String> {
    load_rules_manifest(BAKED_RULES_MANIFEST)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_digest_matches_pinned_oracle_value() {
        assert_eq!(adapter_hash(), PINNED_ADAPTER_HASH);
    }

    #[test]
    fn baked_rules_digest_matches_pinned_oracle_value() {
        let info = load_baked_rules().expect("baked rules load");
        assert_eq!(info.rules_id, RULES_ID);
        assert_eq!(info.rules_hash, PINNED_RULES_HASH);
    }

    #[test]
    fn rules_loader_fails_closed() {
        assert!(load_rules_manifest("not json").is_err());
        assert!(load_rules_manifest("[1,2]").is_err());
        // Wrong envelope type.
        let wrong_type = BAKED_RULES_MANIFEST
            .replacen(RULES_ARTIFACT_TYPE, "hydra2.other", 1);
        assert!(load_rules_manifest(&wrong_type).is_err());
        // Wrong rules id.
        let wrong_id = BAKED_RULES_MANIFEST.replacen(RULES_ID, "other_rules_v9", 1);
        assert!(load_rules_manifest(&wrong_id).is_err());
        // Truncated bytes.
        assert!(load_rules_manifest(&BAKED_RULES_MANIFEST[..1024]).is_err());
    }

    #[test]
    fn single_byte_content_tamper_moves_digest_off_pin() {
        // Valid JSON, same envelope + id, one semantic byte changed: the
        // loader stays faithful (new digest) instead of silently keeping the
        // pin, so tampering is always evident at the pin comparison.
        let tampered = BAKED_RULES_MANIFEST.replacen("\"bankruptcy_threshold\":0", "\"bankruptcy_threshold\":1", 1);
        assert_ne!(tampered, BAKED_RULES_MANIFEST);
        let info = load_rules_manifest(&tampered).expect("tampered still parses");
        assert_eq!(info.rules_id, RULES_ID);
        assert_ne!(info.rules_hash, PINNED_RULES_HASH);
    }
}
