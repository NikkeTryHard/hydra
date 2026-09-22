//! randomness: frozen SPEC 13 seed-derivation leaves on the shared contracts submodule.
//!
//! DAG: this module depends on `hydra-feed` (canon + digest owners) +
//! `hydra-search` (`eval::semantic_seed` formula owner) + pyo3 ONLY (same edge
//! as the rest of the bridge; no new dependency). FORBIDS: JCS re-printing
//! (the feed `canon` owner is the single printer), Philox/CTR reinvention
//! (feed `digest::ctr_block` + `rng` own the streams; `canon_rng` owns the NEW
//! stream surface), wall-clock/urandom seeding (semantic counter-based seeds
//! only), and live-object orchestration (dataclasses, `RandomStreamSchema`
//! validation matrix, `StreamLedger` issue sets, draw cursors stay
//! Python-side as validating roots + mutable state).
//!
//! What lives here (frozen consts + total pure leaves; everything else stays
//! in `python/hydra2/contracts/randomness.py`):
//! - domain strings + schema inventories (`RNG_PROTOCOL`, `RANDOM_PURPOSES`,
//!   `FINAL_EVALUATION_PURPOSES`, `RANDOM_SCOPES`, `RANDOM_KEY_FIELDS`);
//!   the legacy CTR tag stays owned by `hydra_feed::digest::CTR_DOMAIN_TAG`
//!   (never duplicated here) and the purpose required/optional matrix stays
//!   Python (`RandomStreamSchema` validating root, pinned by
//!   `test_schema_class_exposes_matrix`).
//! - `derive_scope_material` (SPEC 13 final bullet): raw-sha256 over
//!   `b"hydra2_master_scope_v1\x00" + scope + b"\x00" + root`, hashed through
//!   the feed `digest` owner, never a local hasher.
//! - `semantic_seed_from_key_doc`: the documented oracle payload
//!   `{"protocol": "hydra2_rng_v1", "master_seed": <hex>, "key": <key json>}`
//!   hashed via the `hydra-search` formula owner (which itself routes through
//!   feed canon + digest); the caller prepares the canonical key doc
//!   Python-side via `key_to_json` + `canonical_bytes`, Rust re-canonicalizes
//!   through the feed owner exactly like `packet_id_from_doc`.
//! - `check_authority_scope` + `is_random_purpose` +
//!   `is_final_evaluation_purpose`: the `authority_stream` scope gate as a
//!   total attached leaf (sub-microsecond string compares, cheaper than a GIL
//!   round-trip — see the frozen-census leaf note in `contracts.rs`).
//!
//! Fail-closed: every shape violation is `PyValueError` with Python-repr-style
//! single quotes (`got '{v}'`, input echoed verbatim so uppercase stays
//! uppercase); never a default or a fallback.
//!
//! Single-cdylib tree: registers its consts + fns on the shared
//! `hydra2._native.contracts` submodule via `register`; no new entry point.
//! MAIN wires this with two lines (`pub mod randomness;` in `lib.rs` plus
//! `crate::randomness::register(&sub)?` in `contracts::register`).
//!
//! Precedents (one per pyo3 API): `pub fn register(sub)` + borrowed `sub` in
//! `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:485`); `let py = sub.py()` mirrors
//! `contracts::register` (`crates/bridge/src/contracts.rs:1115`);
//! `sub.add(<str>, <&str>)` mirrors `contracts.rs:1153`;
//! `PyTuple::new(py, [...])` for frozen tuples mirrors `contracts.rs:1156`;
//! `PyFrozenSet::new(py, [...])` mirrors `contracts.rs:1186`;
//! attached-check + `py.detach(|| ...)` with zero Python API inside mirrors
//! `contracts::ctr_block` (`crates/bridge/src/contracts.rs:447-456`);
//! re-canonicalize-then-hash through the feed owners mirrors
//! `contracts::packet_id_from_doc` (`crates/bridge/src/contracts.rs:912-927`);
//! `SearchError` mapped onto `ValueError` mirrors `eval::search_err`
//! (`crates/bridge/src/eval.rs:60-63`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFrozenSet, PyModule, PyTuple};

/// Seed-derivation protocol tag (`randomness.py:7,32,352-354`); also embedded
/// in the `hydra-search` formula owner (`eval::semantic_seed` builds the same
/// `{"protocol": ...}` payload, `crates/search/src/eval/mod.rs:220-224`).
const RNG_PROTOCOL: &str = "hydra2_rng_v1";

/// Scope-derivation domain label without the trailing NUL
/// (`randomness.py:509-511` hashes `b"hydra2_master_scope_v1\x00" + scope +
/// `b"\x00" + root`; Rust re-adds both NULs so the preimage is byte-exact).
/// Fresh: no Rust crate exports this domain (`rg master_scope crates` is
/// empty); the legacy CTR tag by contrast stays owned by
/// `hydra_feed::digest::CTR_DOMAIN_TAG` and is never duplicated here.
const MASTER_SCOPE_TAG: &str = "hydra2_master_scope_v1";

/// Every stream purpose in SPEC order (`randomness.py:114-133`).
/// Fresh: no Rust crate exports this inventory as a const (the search crate
/// only embeds purpose strings inline in schedule keys, never the table).
const RANDOM_PURPOSES: [&str; 18] = [
    "wall",
    "belief_natural_sample",
    "belief_proposal_sample",
    "actor_policy_sample",
    "root_tree_selection",
    "rollout_transition",
    "rollout_advantage",
    "confirmation",
    "coupling_primitive",
    "mlmc_level",
    "rqmc_scramble",
    "smc_propagation",
    "smc_resampling",
    "training_shuffle",
    "training_dropout",
    "evaluation_schedule",
    "gumbel_root",
    "kernel_sample",
];

/// Purposes whose seeds are final-evaluation material only
/// (`randomness.py:137`; pinned by `test_final_evaluation_seed_isolation`).
const FINAL_EVALUATION_PURPOSES: [&str; 2] = ["confirmation", "evaluation_schedule"];

/// Scope vocabulary (`randomness.py:496,503-512`).
const RANDOM_SCOPES: [&str; 2] = ["selection_training", "final_evaluation"];

/// `RandomStreamKey` field order, SPEC-verbatim (`randomness.py:206-222`;
/// also the `key_to_json` projection order). The dataclass itself stays
/// Python; this inventory lets bridge-adjacent tooling name fields without
/// re-listing them.
const RANDOM_KEY_FIELDS: [&str; 17] = [
    "purpose",
    "experiment_id",
    "split_id",
    "candidate_id",
    "case_id",
    "wall_id",
    "root_seat",
    "belief_epoch",
    "parent_id",
    "action_id",
    "packet_id",
    "fidelity_level",
    "population_id",
    "replicate_id",
    "scramble_id",
    "visit_index",
    "attempt_id",
];

/// True iff the literal is a frozen SPEC 13 purpose (mirrors
/// `purpose in _REQUIRED_BY_PURPOSE`; precedent `contracts::is_action_kind`).
#[pyfunction]
fn is_random_purpose(purpose: &str) -> bool {
    RANDOM_PURPOSES.contains(&purpose)
}

/// True iff the purpose is final-evaluation material only.
#[pyfunction]
fn is_final_evaluation_purpose(purpose: &str) -> bool {
    FINAL_EVALUATION_PURPOSES.contains(&purpose)
}

/// Python-repr-style rendering of the frozen purpose inventory for reject
/// texts: `('wall', 'belief_natural_sample', ...)` with single quotes,
/// matching `f"... {RANDOM_PURPOSES}"` in `RandomStreamSchema.validate_key`
/// (`randomness.py:242`). Field names never contain quotes, so the simple
/// form is exact (same note as `hydra_search::eval::py_list_repr`).
fn purposes_repr() -> String {
    let mut out = String::from("(");
    for (i, purpose) in RANDOM_PURPOSES.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(purpose);
        out.push('\'');
    }
    out.push(')');
    out
}

/// Pure scope gate behind `check_authority_scope`: `Ok` iff the (scope,
/// purpose) pair may proceed under SPEC 13 isolation
/// (`randomness.py:515-527`). Total: unknown scopes and unknown purposes fail
/// closed with oracle-exact texts (single quotes, input echoed verbatim);
/// cross-scope pairs fail with the two `authority_stream` sentences.
fn authority_gate(scope: &str, purpose: &str) -> Result<(), String> {
    if !RANDOM_SCOPES.contains(&scope) {
        return Err(format!("unknown scope '{scope}'"));
    }
    if !RANDOM_PURPOSES.contains(&purpose) {
        return Err(format!(
            "purpose '{purpose}' is not one of {}",
            purposes_repr()
        ));
    }
    let key_is_final = FINAL_EVALUATION_PURPOSES.contains(&purpose);
    if scope == "final_evaluation" && !key_is_final {
        return Err(format!(
            "final-evaluation material cannot feed selection/training purpose '{purpose}'"
        ));
    }
    if scope == "selection_training" && key_is_final {
        return Err(format!(
            "selection/training material cannot feed final-evaluation purpose '{purpose}'"
        ));
    }
    Ok(())
}

/// Scope gate for `authority_stream` (`randomness.py:515-527`): fails closed
/// on unknown scopes/purposes and on cross-scope use. Attached
/// (sub-microsecond string compares over frozen tables).
#[pyfunction]
#[pyo3(signature = (scope, purpose))]
fn check_authority_scope(scope: &str, purpose: &str) -> PyResult<()> {
    authority_gate(scope, purpose).map_err(PyValueError::new_err)
}

/// Validate a scope label for the derivation path (shared by the attached
/// fast-fail and the detached re-check so the helper stays total).
fn check_scope_label(scope: &str) -> Result<(), String> {
    if RANDOM_SCOPES.contains(&scope) {
        Ok(())
    } else {
        Err(format!("unknown scope '{scope}'"))
    }
}

/// Byte-exact scope preimage (`randomness.py:509-511`):
/// `b"hydra2_master_scope_v1\x00" + scope + b"\x00" + root`.
fn scope_preimage(root_material: &[u8], scope: &str) -> Vec<u8> {
    let mut pre =
        Vec::with_capacity(MASTER_SCOPE_TAG.len() + 1 + scope.len() + 1 + root_material.len());
    pre.extend_from_slice(MASTER_SCOPE_TAG.as_bytes());
    pre.push(0x00);
    pre.extend_from_slice(scope.as_bytes());
    pre.push(0x00);
    pre.extend_from_slice(root_material);
    pre
}

/// One hex nibble value; `None` for non-hex bytes.
fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Decode the 64 lowercase hex chars behind a `sha256:<hex>` digest into raw
/// bytes (defensive-only: the input is produced one line above by the feed
/// owner, so failure is unreachable — still fails closed, never unwraps).
fn decode_digest_hex(digest: &str) -> Result<[u8; 32], String> {
    let hex = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| "randomness derive_scope_material: digest missing prefix".to_string())?;
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err("randomness derive_scope_material: digest has wrong length".to_string());
    }
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        let hi = hex_val(bytes[2 * i])
            .ok_or_else(|| "randomness derive_scope_material: digest has non-hex".to_string())?;
        let lo = hex_val(bytes[2 * i + 1])
            .ok_or_else(|| "randomness derive_scope_material: digest has non-hex".to_string())?;
        out[i] = (hi << 4) | lo;
        i += 1;
    }
    Ok(out)
}

/// Derive independent per-scope master material from one root secret
/// (mirrors `derive_scope_material`, `randomness.py:503-512`): raw sha256
/// over the domain-separated preimage, hashed through the feed `digest`
/// owner. Owned inputs so the whole hash runs detached with zero Python API
/// inside; scope/material rejects fire attached AND detached (total).
#[pyfunction]
#[pyo3(signature = (root_material, scope))]
fn derive_scope_material(
    py: Python<'_>,
    root_material: Vec<u8>,
    scope: String,
) -> PyResult<Py<PyBytes>> {
    if root_material.is_empty() {
        return Err(PyValueError::new_err(
            "root_material must be nonempty bytes",
        ));
    }
    check_scope_label(&scope).map_err(PyValueError::new_err)?;
    let out = py
        .detach(|| -> Result<[u8; 32], String> {
            check_scope_label(&scope)?;
            if root_material.is_empty() {
                return Err("root_material must be nonempty bytes".to_string());
            }
            let pre = scope_preimage(&root_material, &scope);
            let digest = hydra_feed::digest::sha256_hex(&pre);
            decode_digest_hex(&digest)
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Derive the 32-byte stream seed for a canonical key document (mirrors
/// `semantic_seed`, `randomness.py:328-377`): the caller prepares
/// `canonical_bytes(key_to_json(key))` Python-side (canon authority stays
/// Python for key staging); Rust parses through the feed `canon` owner and
/// hashes the documented `{"protocol", "master_seed", "key"}` payload through
/// the `hydra-search` formula owner (`eval::semantic_seed`, itself feed
/// canon + digest — never a second printer or hasher). Owned inputs so the
/// parse + hash runs detached with zero Python API inside. Empty master seeds
/// fail closed with the oracle text; key-doc parse/formula rejects surface as
/// `ValueError` for the Python translator to map onto `ContractError`.
#[pyfunction]
#[pyo3(signature = (master_seed, key_doc))]
fn semantic_seed_from_key_doc(
    py: Python<'_>,
    master_seed: Vec<u8>,
    key_doc: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    if master_seed.is_empty() {
        return Err(PyValueError::new_err("master_seed must be nonempty bytes"));
    }
    let out = py
        .detach(|| -> Result<[u8; 32], String> {
            if master_seed.is_empty() {
                return Err("master_seed must be nonempty bytes".to_string());
            }
            let key = hydra_feed::canon::parse_canonical_bytes(
                &key_doc,
                "randomness:semantic_seed_from_key_doc",
            )
            .map_err(|e| format!("randomness semantic_seed_from_key_doc rejected: {e}"))?;
            hydra_search::eval::semantic_seed(&master_seed, &key)
                .map_err(|e| format!("randomness semantic_seed_from_key_doc rejected: {e}"))
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Register the frozen randomness tables + seed-derivation leaves on the
/// shared `contracts` submodule (mirrors `contracts::register`; single
/// cdylib, no new entry).
///
/// Precedents: `register` signature shape -> `action_artifact.rs:42`;
/// `let py = sub.py()` -> `action_artifact.rs:43`;
/// `sub.add(<str>, <&str>)` -> `contracts.rs:1153`;
/// `PyTuple::new(py, [...])` for frozen tuples -> `contracts.rs:1156`;
/// `PyFrozenSet::new(py, [...])` -> `contracts.rs:1186`;
/// `wrap_pyfunction!(f, sub)` with a borrowed `sub` -> `rows_seal.rs:485`.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("RNG_PROTOCOL", RNG_PROTOCOL)?;
    sub.add("MASTER_SCOPE_TAG", MASTER_SCOPE_TAG)?;
    sub.add("RANDOM_PURPOSES", PyTuple::new(py, RANDOM_PURPOSES)?)?;
    sub.add(
        "FINAL_EVALUATION_PURPOSES",
        PyFrozenSet::new(py, FINAL_EVALUATION_PURPOSES)?,
    )?;
    sub.add("RANDOM_SCOPES", PyTuple::new(py, RANDOM_SCOPES)?)?;
    sub.add("RANDOM_KEY_FIELDS", PyTuple::new(py, RANDOM_KEY_FIELDS)?)?;
    sub.add_function(wrap_pyfunction!(is_random_purpose, sub)?)?;
    sub.add_function(wrap_pyfunction!(is_final_evaluation_purpose, sub)?)?;
    sub.add_function(wrap_pyfunction!(check_authority_scope, sub)?)?;
    sub.add_function(wrap_pyfunction!(derive_scope_material, sub)?)?;
    sub.add_function(wrap_pyfunction!(semantic_seed_from_key_doc, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod randomness_tests {
    use super::*;

    #[test]
    fn frozen_inventories_match_oracle_shape() {
        assert_eq!(RANDOM_PURPOSES.len(), 18);
        assert_eq!(RANDOM_PURPOSES[0], "wall");
        assert_eq!(RANDOM_PURPOSES[17], "kernel_sample");
        assert_eq!(RNG_PROTOCOL, "hydra2_rng_v1");
        assert_eq!(MASTER_SCOPE_TAG, "hydra2_master_scope_v1");
        assert_eq!(RANDOM_SCOPES, ["selection_training", "final_evaluation"]);
        assert_eq!(RANDOM_KEY_FIELDS.len(), 17);
        assert_eq!(RANDOM_KEY_FIELDS[0], "purpose");
        assert_eq!(RANDOM_KEY_FIELDS[16], "attempt_id");
        assert!(is_random_purpose("wall"));
        assert!(!is_random_purpose("WALL"));
        assert!(!is_random_purpose("everything"));
        assert!(is_final_evaluation_purpose("confirmation"));
        assert!(!is_final_evaluation_purpose("wall"));
    }

    #[test]
    fn purposes_repr_is_tuple_repr_exact() {
        assert_eq!(
            purposes_repr(),
            "('wall', 'belief_natural_sample', 'belief_proposal_sample', \
             'actor_policy_sample', 'root_tree_selection', 'rollout_transition', \
             'rollout_advantage', 'confirmation', 'coupling_primitive', 'mlmc_level', \
             'rqmc_scramble', 'smc_propagation', 'smc_resampling', 'training_shuffle', \
             'training_dropout', 'evaluation_schedule', 'gumbel_root', 'kernel_sample')"
        );
    }

    #[test]
    fn scope_preimage_is_oracle_byte_exact() {
        let root = [0x5au8; 32];
        let pre = scope_preimage(&root, "selection_training");
        let mut expect = Vec::new();
        expect.extend_from_slice(b"hydra2_master_scope_v1\x00");
        expect.extend_from_slice(b"selection_training");
        expect.push(0x00);
        expect.extend_from_slice(&root);
        assert_eq!(pre, expect);
    }

    #[test]
    fn authority_gate_matches_oracle_sentences() {
        assert!(authority_gate("selection_training", "training_shuffle").is_ok());
        assert!(authority_gate("final_evaluation", "confirmation").is_ok());
        assert!(authority_gate("final_evaluation", "evaluation_schedule").is_ok());
        let err = authority_gate("selection_training", "confirmation").unwrap_err();
        assert_eq!(
            err,
            "selection/training material cannot feed final-evaluation purpose 'confirmation'"
        );
        let err = authority_gate("final_evaluation", "training_shuffle").unwrap_err();
        assert_eq!(
            err,
            "final-evaluation material cannot feed selection/training purpose 'training_shuffle'"
        );
        let err = authority_gate("everything", "wall").unwrap_err();
        assert_eq!(err, "unknown scope 'everything'");
        // Uppercase echoes uppercase (never normalized).
        let err = authority_gate("SELECTION_TRAINING", "wall").unwrap_err();
        assert_eq!(err, "unknown scope 'SELECTION_TRAINING'");
        let err = authority_gate("selection_training", "WALL").unwrap_err();
        assert!(err.contains("'WALL'"));
        assert!(check_scope_label("selection_training").is_ok());
        assert!(check_scope_label("nope").is_err());
    }
}
