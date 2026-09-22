//! promotion: SPEC 18.4 promotion-record frozen tables + pure leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-feed` (`digest`, via `canon` fused) + `hydra-search`
//! (`eval::partition` vocabulary) ONLY (all already bridge dependencies, see
//! `crates/bridge/Cargo.toml:19-23`; no new dependency). FORBIDS: JCS
//! re-printing (the feed `canon`/`digest` owners are the single printer +
//! hasher), dataclass construction (every `PromotionRecord` / `ExcludedBlock`
//! record stays Python in `python/hydra2/eval/promotion.py` / `blocks.py`),
//! live-object orchestration (`make_promotion_record` assembly,
//! `record_to_json` projection over live `ExcludedBlock` objects, and every
//! `ContractError` translator stay Python), and wall-clock/IO.
//!
//! TABLE ported here (frozen values + pure checks over plain values, oracle
//! `python/hydra2/eval/promotion.py`):
//! - `PROMOTION_UNCERTAINTY_UNITS` <- `promotion.py:37` (re-export; vocab
//!   `UncertaintyUnit` L27-34) via the search owner const
//!   (`hydra_search::eval::partition::UNCERTAINTY_UNITS`, referenced, never
//!   recopied).
//! - `PROMOTION_GATE_VALUES` <- `promotion.py:38` (`_GATE_VALUES`).
//! - `PROMOTION_DISPOSITIONS` <- `promotion.py:39` (`_DISPOSITIONS`).
//! - `promotion_check_gates` <- gates gates (`promotion.py:137-143`): the
//!   `Mapping` gate stays Python (translator normalises via `dict()` and
//!   rejects non-mappings with the oracle text, mirroring
//!   `eval_duplicate.rs:13-20`); the bridge takes the normalised `dict`,
//!   stages every entry attached, and decides detached.
//! - `promotion_check_disposition` <- disposition vocab
//!   (`promotion.py:144-148`): the live `repr(disposition)` is staged
//!   attached so `{disposition!r}` renders byte-exact for every input; the
//!   decision runs detached.
//! - `promotion_check_promoted` <- fail-closed `promoted` gate
//!   (`promotion.py:149-157`): the caller passes the normalised gates `dict`
//!   plus two plain bools (`has_schedule_hash`, `is_promoted`); the failed
//!   set is computed Rust-side (sorted by name, rendered via the staged
//!   exact reprs) and the schedule binding is a bool (digest-shape judging
//!   stays Python via `_require_digest`). Gates are re-checked defensively
//!   inside so the helper is total on its own (oracle order: gates before
//!   promoted).
//! - `promotion_digest_from_doc` <- `promotion_digest`
//!   (`promotion.py:214-216`): the caller projects `record_to_json(record)`
//!   Python-side (live `PromotionRecord`/`ExcludedBlock` objects never
//!   cross, mirroring `eval_case_manifest_hash_from_docs`); this pyfn stages
//!   the doc attached, then runs ONE detach around the feed canon+digest
//!   seal and returns the `sha256:<hex>` text.
//!
//! LOGIC staying Python (`eval/promotion.py`): the `PromotionRecord`
//! dataclass (frozen, slots, SPEC 18.4 field order), `UncertaintyUnit` +
//! `UNCERTAINTY_UNITS` re-export (Literal typing stays), `_require_digest`
//! shaping (bridge-checked digest text assembled Python-side), the
//! `make_promotion_record` unknown/missing/comparator/resource/inequality/
//! estimate/bounds/environment/excluded orchestration (float finiteness needs
//! Python float semantics; `ExcludedBlock` isinstance needs the live class;
//! bridge-checked gates/disposition/promoted scalars assembled Python-side),
//! `record_to_json` projection (live objects + `sorted(gates)` ordering),
//! and every `ContractError` translator (the bridge raises `ValueError` with
//! byte-identical text; `promotion.py` maps to `ContractError` — except the
//! digest-shape `ValueError`, which propagates bare exactly like the
//! oracle's `validate_digest` passthrough).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `let py = sub.py()` mirrors
//! `contracts::register` (`crates/bridge/src/contracts.rs:1115`);
//! `PyTuple::new(py, [...])` for frozen tuples mirrors
//! (`crates/bridge/src/contracts.rs:1156`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); attached `repr` mirrors
//! `rc_require.rs:63-65`; staged-`Value` JCS entry via fused
//! `hydra_feed::digest::of_canonical` mirrors `validate::validation_hash`
//! (`crates/bridge/src/validate.rs:83-95`); the staging arms mirror
//! `canon_rng::py_to_value` (`crates/bridge/src/canon_rng.rs:275-363`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Feed owners (never reimplemented here): frozen uncertainty vocabulary via
//! `hydra_search::eval::partition::UNCERTAINTY_UNITS`
//! (`crates/search/src/eval/partition.rs:355-363`); seals via
//! `hydra_feed::digest::of_canonical` (`crates/feed/src/digest.rs:46`,
//! `sha256:`-prefixed lowercase hex, fallible only via the canon step).
//!
//! NONE-owner note: `rg` for `GATE_VALUES|_DISPOSITIONS|not_applicable`
//! over `crates/feed/src` + `crates/search/src` hits nothing (no owner
//! exists; only unrelated `gate bypassed` comments in `feed/src/walk`);
//! the two 3-tuples are stated here with the oracle line cited per use.
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Lone-surrogate `str` gate names/values/dispositions fail `String`
//!   extraction and are rejected with the shape message, where the oracle
//!   would accept the label (it only checks `isinstance(str)` + emptiness)
//!   and fail later at the canon seal. Both sides fail closed; well-formed
//!   UTF-8 callers are unaffected (same note as `eval_leaves.rs:65-69`).
//! - Non-`dict` `Mapping` gates (e.g. `MappingProxy`) never reach here: the
//!   Python translator normalises via `dict()` (insertion order preserved)
//!   after the exact `isinstance(Mapping)` gate, mirroring
//!   `eval_duplicate.rs:13-20`. Direct bridge callers with a non-`dict`
//!   mapping get a `TypeError`; oracle callers get the mapping gate.
//!   `dict` callers — every realistic caller — are unaffected.
//! - Exotic gate names containing quotes/backslashes render via the staged
//!   exact `repr` in gate-value errors and in the promoted-failures list
//!   (byte-exact for every input); the frozen vocabularies themselves are a
//!   closed identifier domain (no quotes), so the single-quote tuple join is
//!   exact there (same note as `eval_leaves::py_str_list`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod promotion;` in `lib.rs` plus
//! `crate::promotion::register(&sub)?;` in `contracts::register` next to
//! `crate::eval_schedule::register(&sub)?;` (`contracts.rs:1309`).

use hydra_search::eval::partition as partition_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

/// Frozen gate vocabulary (`promotion.py:44`, no owner — see NONE note).
const PROMOTION_GATE_VALUES: [&str; 3] = ["passed", "failed", "not_applicable"];

/// Frozen disposition vocabulary (`promotion.py:45`, no owner — see NONE note).
const PROMOTION_DISPOSITIONS: [&str; 3] = ["promoted", "rejected", "blocked"];

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Python-`tuple` rendering (`()`, `('a',)`, `('a', 'b')`). Same closed
/// identifier domain as `py_str_list` in `rc_sections.rs`, so the
/// trailing-comma single form is exact there.
fn py_str_tuple(items: &[String]) -> String {
    if items.is_empty() {
        return String::from("()");
    }
    if items.len() == 1 {
        return format!("('{}',)", items[0]);
    }
    let mut out = String::from("(");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(')');
    out
}

/// Python-`list` rendering over pre-staged exact reprs (`[repr, ...]`).
/// Byte-exact for every input (each element is already the live `repr`).
fn py_repr_list(reprs: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in reprs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(item);
    }
    out.push(']');
    out
}

fn gate_values_tuple() -> Vec<String> {
    PROMOTION_GATE_VALUES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn dispositions_tuple() -> Vec<String> {
    PROMOTION_DISPOSITIONS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// One staged gate entry: `name` is `Some` iff the key is a nonempty `str`;
/// `value` is `Some` iff the value is a `str` (membership decided detached
/// so the helper is total); reprs carry the exact Python `repr` for `{!r}`
/// slots.
struct StagedGate {
    name: Option<String>,
    name_repr: String,
    value: Option<String>,
    value_repr: String,
}

/// Gates gates (`promotion.py:137-143`): every name a nonempty `str`, every
/// value in the frozen gate vocabulary. `None` stages every non-`str`/empty
/// shape so the helper is total on its own.
fn check_gates(entries: &[StagedGate]) -> Result<(), String> {
    for entry in entries {
        match &entry.name {
            Some(text) if !text.is_empty() => {}
            _ => return Err(String::from("gate names must be nonempty strings")),
        }
        match &entry.value {
            Some(text) if PROMOTION_GATE_VALUES.contains(&text.as_str()) => {}
            _ => {
                return Err(format!(
                    "gate {} value {} not in {}",
                    entry.name_repr,
                    entry.value_repr,
                    py_str_tuple(&gate_values_tuple())
                ));
            }
        }
    }
    Ok(())
}

/// Disposition vocabulary (`promotion.py:144-148`): membership in the frozen
/// disposition vocabulary. The staged live `repr` renders byte-exact for
/// every input; `None` stages every non-`str` input so the helper is total.
fn check_disposition(staged: &Option<String>, disp_repr: &str) -> Result<String, String> {
    match staged {
        Some(text) if PROMOTION_DISPOSITIONS.contains(&text.as_str()) => Ok(text.clone()),
        _ => Err(format!(
            "disposition {} not in {}",
            disp_repr,
            py_str_tuple(&dispositions_tuple())
        )),
    }
}

/// Fail-closed `promoted` gate (`promotion.py:149-157`): only active when
/// `is_promoted`; otherwise `Ok(())`. Gates are re-checked first (oracle
/// order) so the helper is total on its own; the failed set renders via the
/// staged exact reprs sorted by gate name (byte-exact for every input).
fn check_promoted(
    entries: &[StagedGate],
    has_schedule_hash: bool,
    is_promoted: bool,
) -> Result<(), String> {
    if !is_promoted {
        return Ok(());
    }
    check_gates(entries)?;
    let mut failed: Vec<(&str, &str)> = Vec::new();
    for entry in entries {
        let is_passed = entry.value.as_deref() == Some("passed");
        if !is_passed && let (Some(name), Some(_)) = (entry.name.as_deref(), entry.value.as_deref())
        {
            failed.push((name, entry.name_repr.as_str()));
        }
    }
    failed.sort_by(|a, b| a.0.cmp(b.0));
    if entries.is_empty() || !failed.is_empty() {
        let reprs: Vec<String> = failed.iter().map(|(_, repr)| (*repr).to_string()).collect();
        return Err(format!(
            "promoted disposition requires all gates passed, got failures: {}",
            py_repr_list(&reprs)
        ));
    }
    if !has_schedule_hash {
        return Err(String::from(
            "promoted disposition requires a non-None schedule_hash binding (schedule commitment hash)",
        ));
    }
    Ok(())
}

/// Stage one closed-domain Python object to a serde `Value` (attached only:
/// touches Python memory, never detached). Mirrors
/// `eval_leaves::py_to_value` arm-for-arm (`None` -> null; `bool` BEFORE
/// `int` since Python `bool` subclasses `int`; `int` outside the
/// double-safe window rejected; finite `float` only; `str`; `list`
/// element-wise, where `tuple` is NOT a list and is rejected; `dict` with
/// non-`str` keys rejected; anything else rejected). `record` names the
/// digest call.
fn py_to_value(obj: &Bound<'_, PyAny>, record: &str) -> Result<serde_json::Value, String> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        let flag = obj
            .extract::<bool>()
            .map_err(|e| format!("{record}: bool unreadable: {e}"))?;
        return Ok(serde_json::Value::Bool(flag));
    }
    if obj.is_instance_of::<PyInt>() {
        match obj.extract::<i64>() {
            Ok(n) => {
                if !(-hydra_feed::canon::MAX_SAFE_INTEGER..=hydra_feed::canon::MAX_SAFE_INTEGER)
                    .contains(&n)
                {
                    return Err(format!(
                        "{record}: integer {n} exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                    ));
                }
                return Ok(serde_json::Value::Number(n.into()));
            }
            Err(_) => {
                return Err(format!(
                    "{record}: integer exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        let value = obj
            .extract::<f64>()
            .map_err(|e| format!("{record}: float unreadable: {e}"))?;
        if !value.is_finite() {
            return Err(format!(
                "{record}: non-finite number {value:?} has no canonical serialization"
            ));
        }
        match serde_json::Number::from_f64(value) {
            Some(number) => return Ok(serde_json::Value::Number(number)),
            None => {
                return Err(format!(
                    "{record}: non-finite number has no canonical serialization"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyString>() {
        match obj.extract::<String>() {
            Ok(text) => return Ok(serde_json::Value::String(text)),
            Err(e) => {
                return Err(format!(
                    "{record}: string contains an unpaired surrogate (invalid Unicode): {e}"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyList>() {
        let elements: Vec<Bound<'_, PyAny>> = obj
            .extract()
            .map_err(|e| format!("{record}: list unreadable: {e}"))?;
        let mut array = Vec::with_capacity(elements.len());
        for item in &elements {
            array.push(py_to_value(item, record)?);
        }
        return Ok(serde_json::Value::Array(array));
    }
    if obj.is_instance_of::<PyDict>() {
        let dict: Bound<'_, PyDict> = obj
            .extract()
            .map_err(|e| format!("{record}: dict unreadable: {e}"))?;
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(format!(
                    "{record}: object key is not a string; JSON objects are string-keyed only"
                ));
            }
            let name: String = key.extract().map_err(|e| {
                format!(
                    "{record}: object key contains an unpaired surrogate (invalid Unicode): {e}"
                )
            })?;
            map.insert(name, py_to_value(&value, record)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(format!(
        "{record}: value is outside the canonical JSON domain (null, bool, string, finite number, array, string-keyed object only)"
    ))
}

/// Pure seal over a staged value through the feed digest owner
/// (byte-identical to `artifacts/digest.py::of_canonical` by construction:
/// same fused `of_canonical` entry `validate::validation_hash` uses).
fn promotion_digest_value(value: &serde_json::Value) -> Result<String, String> {
    hydra_feed::digest::of_canonical(value)
        .map_err(|e| format!("bridge:promotion_digest_from_doc: canon JCS emit rejected: {e:?}"))
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Gates gates (`promotion.py:137-143`) over the translator-normalised
/// `dict` (the `Mapping` gate stays Python for exact `isinstance` semantics,
/// mirroring `eval_duplicate`). Every entry is staged attached (non-`str`
/// keys/values stage `None` with the exact `repr` kept); the accept/reject
/// decision runs under ONE detach.
#[pyfunction]
fn promotion_check_gates(py: Python<'_>, gates: Bound<'_, PyDict>) -> PyResult<()> {
    let mut entries: Vec<StagedGate> = Vec::with_capacity(gates.len());
    for (key, value) in gates.iter() {
        let name_repr = py_repr(&key)?;
        let value_repr = py_repr(&value)?;
        let name = key
            .extract::<String>()
            .ok()
            .filter(|s: &String| !s.is_empty());
        let value = value.extract::<String>().ok();
        entries.push(StagedGate {
            name,
            name_repr,
            value,
            value_repr,
        });
    }
    py.detach(|| check_gates(&entries))
        .map_err(PyValueError::new_err)
}

/// Disposition vocabulary (`promotion.py:144-148`): membership in the frozen
/// disposition vocabulary. The live `repr(disposition)` is staged attached
/// so `{disposition!r}` renders byte-exact for every input; the decision
/// runs detached.
#[pyfunction]
fn promotion_check_disposition(py: Python<'_>, disposition: Bound<'_, PyAny>) -> PyResult<String> {
    let disp_repr = py_repr(&disposition)?;
    let staged: Option<String> = disposition.extract().ok();
    py.detach(|| check_disposition(&staged, &disp_repr))
        .map_err(PyValueError::new_err)
}

/// Fail-closed `promoted` gate (`promotion.py:149-157`) over the
/// translator-normalised gates `dict` plus two plain bools. The failed set
/// is computed Rust-side (sorted by gate name, rendered via the staged
/// exact reprs); `has_schedule_hash` is the `schedule_hash is not None`
/// projection (digest-shape judging stays Python); `is_promoted` gates the
/// whole check (`disposition == "promoted"` Python-side). Gates are
/// re-checked first so the helper is total on its own. Compute runs
/// detached.
#[pyfunction]
#[pyo3(signature = (gates, has_schedule_hash, is_promoted))]
fn promotion_check_promoted(
    py: Python<'_>,
    gates: Bound<'_, PyDict>,
    has_schedule_hash: bool,
    is_promoted: bool,
) -> PyResult<()> {
    let mut entries: Vec<StagedGate> = Vec::with_capacity(gates.len());
    for (key, value) in gates.iter() {
        let name_repr = py_repr(&key)?;
        let value_repr = py_repr(&value)?;
        let name = key
            .extract::<String>()
            .ok()
            .filter(|s: &String| !s.is_empty());
        let value = value.extract::<String>().ok();
        entries.push(StagedGate {
            name,
            name_repr,
            value,
            value_repr,
        });
    }
    py.detach(|| check_promoted(&entries, has_schedule_hash, is_promoted))
        .map_err(PyValueError::new_err)
}

/// Promotion seal (`promotion_digest`, `promotion.py:214-216`) over the
/// caller-projected `record_to_json(record)` doc: the caller maps the live
/// `PromotionRecord`/`ExcludedBlock` objects Python-side (live objects never
/// cross); this pyfn stages the doc attached, then runs ONE detach around
/// the feed canon+digest seal and returns the `sha256:<hex>` text.
#[pyfunction]
fn promotion_digest_from_doc(py: Python<'_>, doc: Bound<'_, PyAny>) -> PyResult<String> {
    let staged =
        py_to_value(&doc, "bridge:promotion_digest_from_doc").map_err(PyValueError::new_err)?;
    py.detach(|| promotion_digest_value(&staged))
        .map_err(PyValueError::new_err)
}

/// Register the promotion tables + pure leaves on the shared `contracts`
/// submodule (mirrors `validate.rs:111-127`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "PROMOTION_UNCERTAINTY_UNITS",
        PyTuple::new(py, partition_owner::UNCERTAINTY_UNITS)?,
    )?;
    sub.add(
        "PROMOTION_GATE_VALUES",
        PyTuple::new(py, PROMOTION_GATE_VALUES)?,
    )?;
    sub.add(
        "PROMOTION_DISPOSITIONS",
        PyTuple::new(py, PROMOTION_DISPOSITIONS)?,
    )?;
    sub.add_function(wrap_pyfunction!(promotion_check_gates, sub)?)?;
    sub.add_function(wrap_pyfunction!(promotion_check_disposition, sub)?)?;
    sub.add_function(wrap_pyfunction!(promotion_check_promoted, sub)?)?;
    sub.add_function(wrap_pyfunction!(promotion_digest_from_doc, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn staged(name: Option<&str>, value: Option<&str>) -> StagedGate {
        StagedGate {
            name: name.map(str::to_string),
            name_repr: match name {
                Some(text) => format!("'{text}'"),
                None => String::from("None"),
            },
            value: value.map(str::to_string),
            value_repr: match value {
                Some(text) => format!("'{text}'"),
                None => String::from("None"),
            },
        }
    }

    #[test]
    fn frozen_consts_match_oracle() {
        Python::initialize();
        // Oracle values hand-traced from `promotion.py:35-45`.
        assert_eq!(
            partition_owner::UNCERTAINTY_UNITS,
            [
                "case",
                "iid_pair",
                "wall_block",
                "smc_population",
                "rqmc_scramble",
                "game_cluster",
            ]
        );
        assert_eq!(
            PROMOTION_GATE_VALUES,
            ["passed", "failed", "not_applicable"]
        );
        assert_eq!(PROMOTION_DISPOSITIONS, ["promoted", "rejected", "blocked"]);
        assert_eq!(
            py_str_tuple(&gate_values_tuple()),
            "('passed', 'failed', 'not_applicable')"
        );
        assert_eq!(
            py_str_tuple(&dispositions_tuple()),
            "('promoted', 'rejected', 'blocked')"
        );
    }

    #[test]
    fn gate_and_disposition_gates_match_oracle() {
        Python::initialize();
        // `promotion.py:137-143`: every name nonempty, every value in vocab.
        assert_eq!(check_gates(&[]), Ok(()));
        assert_eq!(
            check_gates(&[staged(Some("confirm"), Some("passed"))]),
            Ok(())
        );
        assert_eq!(
            check_gates(&[staged(Some(""), Some("passed"))]),
            Err(String::from("gate names must be nonempty strings"))
        );
        assert_eq!(
            check_gates(&[staged(None, Some("passed"))]),
            Err(String::from("gate names must be nonempty strings"))
        );
        assert_eq!(
            check_gates(&[staged(Some("confirm"), Some("maybe"))]),
            Err(String::from(
                "gate 'confirm' value 'maybe' not in ('passed', 'failed', 'not_applicable')"
            ))
        );
        assert_eq!(
            check_gates(&[staged(Some("confirm"), None)]),
            Err(String::from(
                "gate 'confirm' value None not in ('passed', 'failed', 'not_applicable')"
            ))
        );
        // `promotion.py:144-148`: disposition vocab with byte-exact repr.
        assert_eq!(
            check_disposition(&Some(String::from("promoted")), "'promoted'"),
            Ok(String::from("promoted"))
        );
        assert_eq!(
            check_disposition(&None, "None"),
            Err(String::from(
                "disposition None not in ('promoted', 'rejected', 'blocked')"
            ))
        );
        assert_eq!(
            check_disposition(&Some(String::from("maybe")), "'maybe'"),
            Err(String::from(
                "disposition 'maybe' not in ('promoted', 'rejected', 'blocked')"
            ))
        );
    }

    #[test]
    fn promoted_gate_matches_oracle() {
        Python::initialize();
        // `promotion.py:149-157`: fail-closed promoted; loose otherwise.
        assert_eq!(check_promoted(&[], false, false), Ok(()));
        assert_eq!(
            check_promoted(&[staged(Some("a"), Some("failed"))], false, false),
            Ok(())
        );
        assert_eq!(
            check_promoted(&[staged(Some("a"), Some("passed"))], true, true),
            Ok(())
        );
        assert_eq!(
            check_promoted(&[], true, true),
            Err(String::from(
                "promoted disposition requires all gates passed, got failures: []"
            ))
        );
        assert_eq!(
            check_promoted(
                &[
                    staged(Some("b"), Some("passed")),
                    staged(Some("a"), Some("failed")),
                ],
                true,
                true
            ),
            Err(String::from(
                "promoted disposition requires all gates passed, got failures: ['a']"
            ))
        );
        assert_eq!(
            check_promoted(&[staged(Some("a"), Some("passed"))], false, true),
            Err(String::from(
                "promoted disposition requires a non-None schedule_hash binding (schedule commitment hash)"
            ))
        );
        // TOTAL: gates shape fails before promoted (oracle order).
        assert_eq!(
            check_promoted(&[staged(Some(""), Some("passed"))], true, true),
            Err(String::from("gate names must be nonempty strings"))
        );
    }

    #[test]
    fn digest_seal_is_deterministic_and_prefixed() {
        Python::initialize();
        // Hand-derived shape (never eyeballed hex): `sha256:` + 64 lowercase
        // hex, deterministic per doc, sensitive to the gates order-freezing
        // input (JCS sorts keys, so input order is irrelevant).
        let first = serde_json::json!({"b": [1.0, 2.0], "a": "x"});
        let second = serde_json::json!({"a": "x", "b": [1.0, 2.0]});
        let other = serde_json::json!({"a": "x", "b": [1.0, 3.0]});
        let d1 = promotion_digest_value(&first).expect("seal first");
        let d2 = promotion_digest_value(&second).expect("seal second");
        let d3 = promotion_digest_value(&other).expect("seal other");
        assert_eq!(d1, d2);
        assert!(d1 != d3);
        assert!(d1.starts_with("sha256:"));
        assert_eq!(d1.len(), 7 + 64);
        let hex = &d1[7..];
        assert_eq!(hex.len(), 64);
        for ch in hex.chars() {
            assert!(ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase());
        }
        // Programmatic cross-check: fused `of_canonical` agrees with the
        // canon-then-hash split on the same value.
        let bytes = hydra_feed::canon::canonical_bytes_value(&first, "test").expect("canon bytes");
        assert_eq!(d1, hydra_feed::digest::sha256_hex(&bytes));
    }

    #[test]
    fn pyfn_wrappers_agree_with_detached_checks() {
        Python::initialize();
        Python::attach(|py| {
            let gates = PyDict::new(py);
            gates.set_item("confirm", "passed").expect("set gate");
            assert!(promotion_check_gates(py, gates.clone()).is_ok());
            let bad = PyDict::new(py);
            bad.set_item("confirm", "maybe").expect("set bad gate");
            assert!(promotion_check_gates(py, bad).is_err());
            let disp = promotion_check_disposition(py, PyString::new(py, "rejected").into_any())
                .expect("valid disposition");
            assert_eq!(disp, "rejected");
            assert!(
                promotion_check_disposition(py, PyString::new(py, "maybe").into_any()).is_err()
            );
            assert!(promotion_check_promoted(py, gates.clone(), true, true).is_ok());
            assert!(promotion_check_promoted(py, gates.clone(), false, true).is_err());
            assert!(promotion_check_promoted(py, gates.clone(), false, false).is_ok());
            let doc = PyDict::new(py);
            doc.set_item("a", "x").expect("set doc");
            let digest = promotion_digest_from_doc(py, doc.into_any()).expect("seal doc");
            assert!(digest.starts_with("sha256:"));
            assert_eq!(digest.len(), 7 + 64);
        });
    }
}
