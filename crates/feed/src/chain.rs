//! Public-state chain folds over canonical envelope documents (SPEC 7.2).
//!
//! Mirrors `src/hydra2/contracts/event_packet.py` (`_fold_public_hash`,
//! `public_state_chain_hash`) without copying it: the fold input
//! `{"prefix": <digest>, "event": <envelope document>}` is rebuilt from the
//! caller-supplied canonical envelope bytes, re-canonicalized through the
//! feed [`canon`] owner, and hashed through the feed [`digest`] owner. Only
//! `visibility == "public"` documents advance the chain; the empty chain is
//! `sha256` over zero bytes, exactly like `_EMPTY_CHAIN_DIGEST`.
//!
//! Inputs are canonical envelope bytes as produced by
//! `artifacts.canonical.canonical_bytes(envelope.to_json())` (int/str/None
//! /list/dict domain, no floats), so the parse-then-canonicalize round trip
//! is byte-identical to the Python fold input. A document without a
//! `visibility` string fails closed (`CanonError::InvalidJson`); a non-public
//! visibility skips, mirroring the Python `if` guard.

use crate::canon::{CanonError, canonical_bytes_value, parse_canonical_bytes};
use crate::digest::sha256_hex;

/// One fold step over an already-parsed envelope document.
fn fold_parsed(prefix: &str, event: &serde_json::Value) -> Result<String, CanonError> {
    let mut map = serde_json::Map::new();
    map.insert("event".to_string(), event.clone());
    map.insert(
        "prefix".to_string(),
        serde_json::Value::String(prefix.to_string()),
    );
    let bytes = canonical_bytes_value(&serde_json::Value::Object(map), "chain:fold")?;
    Ok(sha256_hex(&bytes))
}

/// One fold step: `sha256` over the canonical bytes of
/// `{"prefix": prefix, "event": <envelope document>}`.
pub fn fold_public_hash(prefix: &str, event_doc: &[u8]) -> Result<String, CanonError> {
    let event = parse_canonical_bytes(event_doc, "chain:fold_public_hash")?;
    fold_parsed(prefix, &event)
}

/// True iff the parsed envelope document is public. A missing or non-string
/// `visibility` fails closed (real envelopes always carry the field; the
/// Python fold would raise `AttributeError` there, never skip silently).
fn is_public(event: &serde_json::Value, record: &str) -> Result<bool, CanonError> {
    match event.get("visibility").and_then(|v| v.as_str()) {
        Some("public") => Ok(true),
        Some(_) => Ok(false),
        None => Err(CanonError::InvalidJson {
            record: record.to_string(),
            detail: "envelope document has no visibility string".to_string(),
        }),
    }
}

/// Fold public envelope identities into a chained state hash, in input order
/// (mirrors `public_state_chain_hash`). The empty input yields the empty-chain
/// digest (`sha256` over zero bytes).
pub fn public_chain_hash(event_docs: &[Vec<u8>]) -> Result<String, CanonError> {
    let mut digest = sha256_hex(b"");
    for doc in event_docs {
        let event = parse_canonical_bytes(doc, "chain:public_chain_hash")?;
        if is_public(&event, "chain:public_chain_hash")? {
            digest = fold_parsed(&digest, &event)?;
        }
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_CHAIN: &str =
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    fn canon_doc(value: &serde_json::Value) -> Vec<u8> {
        canonical_bytes_value(value, "chain:test").unwrap()
    }

    #[test]
    fn empty_chain_is_sha256_of_nothing() {
        assert_eq!(public_chain_hash(&[]).unwrap(), EMPTY_CHAIN);
    }

    #[test]
    fn only_public_documents_advance_the_chain() {
        let public = canon_doc(&serde_json::json!({"visibility": "public", "sequence": 0}));
        let private =
            canon_doc(&serde_json::json!({"visibility": "actor_private", "sequence": 1}));
        let server =
            canon_doc(&serde_json::json!({"visibility": "server_private", "sequence": 2}));
        let one = public_chain_hash(std::slice::from_ref(&public)).unwrap();
        assert_ne!(one, EMPTY_CHAIN);
        assert_eq!(
            public_chain_hash(&[private.clone(), public.clone(), server.clone()]).unwrap(),
            one
        );
        assert_eq!(public_chain_hash(&[private, server]).unwrap(), EMPTY_CHAIN);
    }

    #[test]
    fn chain_is_order_sensitive_and_deterministic() {
        let a = canon_doc(&serde_json::json!({"visibility": "public", "sequence": 0}));
        let b = canon_doc(&serde_json::json!({"visibility": "public", "sequence": 1}));
        let ab = public_chain_hash(&[a.clone(), b.clone()]).unwrap();
        let ba = public_chain_hash(&[b.clone(), a.clone()]).unwrap();
        assert_ne!(ab, ba);
        assert_eq!(public_chain_hash(&[a, b]).unwrap(), ab);
        // Single-step fold agrees with the one-element chain.
        let first = fold_public_hash(
            EMPTY_CHAIN,
            &canon_doc(&serde_json::json!({"visibility": "public", "sequence": 0})),
        )
        .unwrap();
        assert_eq!(public_chain_hash(&[canon_doc(
            &serde_json::json!({"visibility": "public", "sequence": 0})
        )])
        .unwrap(), first);
    }

    #[test]
    fn missing_visibility_fails_closed() {
        let bad = canon_doc(&serde_json::json!({"sequence": 0}));
        assert!(public_chain_hash(&[bad]).is_err());
    }

    #[test]
    fn malformed_document_fails_closed() {
        assert!(public_chain_hash(&[b"{not json".to_vec()]).is_err());
    }
}
