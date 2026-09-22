//! distill_leaves: WP-10 teacher-gate registry + gate/record seals on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-feed` canon/digest owners ONLY — the gate/record seals
//! build a staged `serde_json::Value` and seal through `feed::canon` +
//! `feed::digest`, never a local hasher (canon-wins B3, see `search.rs`
//! docs); the pure shape/validation cores own no hash bytes and no tables.
//! File reads, the WP-12 gate loader, `CandidateSpec` factories/hashing, and
//! every dataclass stay Python-side in `distillation/_teacher_gate.py` +
//! `distillation/_teacher_records.py`. FORBIDS: file/JSON IO (the
//! action-table probe, `_load_action_table_num_actions`, and its mutable
//! `_ACTION_TABLE_CACHE` stay Python), factory orchestration
//! (`load_analysis_gate` / `select_teacher` / `_real_candidate_spec` /
//! `_spec_digest_of` bind live `CandidateSpec` objects that never cross),
//! trajectory orchestration (`generate_trajectories` walks live specs and
//! case observations), wall-clock (`_now_utc` stays Python), torch/CUDA, and
//! second sources for the registry literals (every const below cites its
//! oracle line; drift fails the golden tests first).
//!
//! TABLE ported here — `distillation/_teacher_gate.py` + `_teacher_records.py`
//! (HEAD; worktree `python/hydra2/distillation/*`, byte-identical at port time):
//! - `TEACHER_CANDIDATES` <- `TEACHER_CANDIDATES` (`_teacher_gate.py:33-41`)
//! - `REJECTED_CANDIDATES` <- `REJECTED_CANDIDATES` (`_teacher_gate.py:44`)
//! - `TEACHER_GATE_KINDS` <- `_GATE_KINDS` (`_teacher_gate.py:50`; renamed —
//!   the shared submodule carries no underscore consts, and the Python
//!   translator re-aliases `_GATE_KINDS` so `sorted()`/`len()` call sites are
//!   untouched)
//! - `ANALYSIS_ID_FOR_TEACHER` <- `_ANALYSIS_ID_FOR_TEACHER`
//!   (`_teacher_gate.py:55-62`; note `candidate4` is absent — unmapped ids
//!   stay blocked, never defaulted)
//! - `DISTILL_REAL_FEATURE_DIM` <- `_REAL_FEATURE_DIM` (`_teacher_gate.py:68`)
//! - `DISTILL_DEFAULT_NUM_ACTIONS` <- `_DEFAULT_NUM_ACTIONS`
//!   (`_teacher_gate.py:70`; the import-time *probe* `_NUM_ACTIONS` stays
//!   Python — it reads `configs/contracts/action_table_v1.json`)
//! - `DISTILL_TRAINING_NAMESPACE_TOKEN` <- `_TRAINING_NAMESPACE_TOKEN`
//!   (`_teacher_records.py:161`)
//! - `distill_require_sha256` <- `_require_sha256` (`_teacher_gate.py:226-233`,
//!   strict shape + lowercase-hex gates)
//! - `distill_gate_hash_for_kind` <- `_gate_hash_for_kind`
//!   (`_teacher_gate.py:385-402`)
//! - `distill_validate_trajectory_values` <- the mask/policy/vector decisions
//!   inside `TrajectoryRecord.__post_init__` (`_teacher_records.py:58-76`)
//! - `distill_trajectory_record_id` <- the record-id seal shared by
//!   `make_trajectory_record` (`_teacher_records.py:131-142`) and the
//!   `__post_init__` recompute (`_teacher_records.py:78-89`)
//!
//! LOGIC staying Python: `_budget_to_frozen` / `_provenance_to_frozen`
//! (one-line `sorted` freezes over `Any` values — staging `Any` dicts across
//! the boundary costs more than the line it replaces), the
//! `TeacherJustification` / `TrajectoryRecord` dataclasses (live-object
//! construction + `__post_init__` orchestration never cross; the bridge owns
//! only the pure decisions the translators delegate), `generate_trajectories`
//! / `make_trajectory_record` orchestration (spec binding, case observations,
//! budget provenance), `generate_privileged_labels` (its belief half needs
//! `_hash_to_uniform` from `_teacher_cases.py`, an explicit non-goal owner —
//! porting half the fn would split one digest domain across two files),
//! `_maybe_privileged_labels`, `_load_action_table_num_actions` /
//! `_action_table` / `_NUM_ACTIONS` / `_ACTION_TABLE_CACHE` (config IO +
//! mutable cache), `load_analysis_gate` / `select_teacher` /
//! `_real_candidate_spec` / `_spec_digest_of` / `_now_utc` (IO, wall-clock,
//! live specs), and every `ContractError` translator (identical messages, no
//! `__all__` change — neither distillation module carries one).
//!
//! Shape per fn: attached staging (`repr`/`str` via the live Python API, so
//! `{value!r}` slots render byte-exact — per `common_roots.rs:69-77`) → ONE
//! `py.detach(|| …)` over owned plain data with zero Python API inside (per
//! `validate.rs:83-95`) → attached wrap as `PyValueError` (per
//! `contracts.rs:117-123`). Translators call positionally and map
//! `ValueError`/`TypeError` onto `ContractError` with byte-identical text
//! (per `loop_state.py:70-78`); a missing/stale `.so` falls back to the
//! verbatim oracle (`hasattr`/`getattr`-default probe only, per
//! `loop_state.py:26-67`).
//!
//! Exactness notes (hand-verified against the oracle, pinned by the tests):
//! - `len(value) != 71` counts Unicode scalars, so the shape gate uses
//!   `chars().count()` (per `loop_state.rs:90-92`), never byte length: a
//!   71-char non-ASCII value must reach the hex gate and fail there with
//!   `invalid hex`, exactly like the oracle.
//! - The fence `math.isclose(s, 1.0, abs_tol=1e-6)` keeps its default
//!   `rel_tol=1e-09`, but the relative term can only exceed `1e-6` when
//!   `|s| >= 1000`, where `|s - 1| >= 999` already fails — so the detached
//!   `is_nan() || (s - 1.0).abs() > 1e-6` check is provably equivalent (no
//!   divergence input exists). The illegal-mass `math.isclose(p, 0.0,
//!   abs_tol=1e-9)` gate is likewise `|p| > 1e-9` except that `isclose`
//!   rejects `NaN` (all comparisons false) — the detached arm adds an
//!   explicit `is_nan()` disjunct for that one input.
//! - `policy_sum` and every `{p}`/`{v}`/`{s}` rendering are staged
//!   Python-side (`sum()` / `repr()`, exact by definition — the
//!   caller-prepares precedent at `contracts.rs:903-907`); the detach only
//!   compares, so no float formatter is restated here and CPython's
//!   compensated `sum` needs no Rust replica.
//! - Seal payloads embed JSON numbers type-faithfully (`bool` → `Bool`,
//!   `int` → integer `Number`, `float` → `from_f64`): an `int` policy entry
//!   seals as `1`, never `1.0`. Out-of-range integers and non-finite floats
//!   fail here as `ValueError`; the oracle's canonical owner rejects the
//!   same inputs (both sides refuse, never digest).
//! - Budget/provenance cross as canonical-doc bytes the translator prepares
//!   with the Python `canonical_bytes` (canon authority stays Python — the
//!   `packet_id_from_doc` precedent at `contracts.rs:903-907`); Rust
//!   re-parses through the feed I-JSON boundary and embeds, so parity holds
//!   by construction and exotic budget values fail with the oracle's own
//!   canonical text before the bridge is reached.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + `let py =
//! sub.py()` mirror `action_artifact::register`
//! (`crates/bridge/src/action_artifact.rs:42-43`); `#[pyfunction]` mirrors
//! `loop_require_sha256` (`crates/bridge/src/loop_state.rs:106-108`);
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirrors
//! `loop_state::register` (`crates/bridge/src/loop_state.rs:134`);
//! `py.detach` over owned data mirrors `loop_require_sha256`
//! (`crates/bridge/src/loop_state.rs:116`); `PyTuple::new` consts mirror
//! `action_artifact::register` (`crates/bridge/src/action_artifact.rs:47`);
//! `PyFrozenSet::new` mirrors `contracts.rs:1186`; `PyDict::new` +
//! `set_item` mirror `contracts.rs:1227-1231`; feed seal
//! `canonical_bytes_value` + `sha256_hex` mirrors `gumbel_spec`
//! (`crates/bridge/src/gumbel_spec.rs:54-58`); `is_instance_of::<PyBool>`
//! mirrors `search_profiles::as_u64`
//! (`crates/bridge/src/search_profiles.rs:42-47`);
//! `#[allow(clippy::too_many_arguments)]` on the 9-arg seal mirrors
//! `rc_check_compat` (`crates/bridge/src/rc_resume.rs:129`);
//! `Python::initialize()` + `Python::attach` in tests mirror the eval-duplicate
//! attach discipline (`crates/bridge/src/eval_duplicate.rs:56-60` cite).
//!
//! Single-cdylib tree: registers its consts/fns on the shared
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `action_artifact::register` on `contracts`); no new entry point. Wiring is
//! MAIN-ONLY (`pub mod distill_leaves;` in `lib.rs` plus
//! `crate::distill_leaves::register(&sub)?;` in `contracts.rs` next to the
//! `crate::common_roots::register(&sub)?;` line).
//!
//! Tolerances (float-bit parity CLAIMED — determinism + golden lane):
//! - `distill_gate_hash_for_kind` / `distill_trajectory_record_id`: bit-exact
//!   (same `canonical_bytes_value` entry `canon_rng::batch_canonical_bytes`
//!   uses, see `canon_rng.rs:388`, plus infallible `sha256_hex`; the goldens
//!   below are live-oracle digests, recomputed in the wave-14 parity smoke).
//! - `distill_validate_trajectory_values`: exact decisions + byte-identical
//!   messages (all renderings staged Python-side).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyModule, PyTuple};
use serde_json::Value;

/// Teacher registry, mirrors Candidates 0-6 (`_teacher_gate.py:33-41`).
pub const TEACHER_CANDIDATES: [&str; 7] = [
    "candidate0",
    "candidate1",
    "candidate2",
    "candidate3",
    "candidate4",
    "candidate5",
    "candidate6",
];

/// Rejected ids remain registry evidence, never teachers (`_teacher_gate.py:44`).
pub const REJECTED_CANDIDATES: [&str; 1] = ["candidate4"];

/// Gate hash domain (`_teacher_gate.py:50`).
pub const TEACHER_GATE_KINDS: [&str; 5] = ["contract", "exact", "search", "match", "analysis"];

/// Teacher candidate -> WP-12 analysis candidate id (`_teacher_gate.py:55-62`).
/// `candidate4` is absent: unmapped ids are blocked, never defaulted.
pub const ANALYSIS_ID_FOR_TEACHER: [(&str, &str); 6] = [
    ("candidate0", "candidate0"),
    ("candidate1", "candidate1"),
    ("candidate2", "candidate2"),
    ("candidate3", "candidate3_pbrf_core_v1"),
    ("candidate5", "candidate5"),
    ("candidate6", "candidate6"),
];

/// Actor-visible feature dim for the real model_input_v1 path (`_teacher_gate.py:68`).
pub const DISTILL_REAL_FEATURE_DIM: i64 = 48;

/// Import-time fallback only; real paths resolve via the action table (`_teacher_gate.py:70`).
pub const DISTILL_DEFAULT_NUM_ACTIONS: i64 = 32;

/// Isolated privileged-label namespace token (`_teacher_records.py:161`).
pub const DISTILL_TRAINING_NAMESPACE_TOKEN: &str = "training_namespace_v1";

/// `len(value) != 71` counts Unicode scalars: `chars().count()`, never byte
/// length (per `loop_state.rs:90-92`).
fn sha_shape_ok(text: &str) -> bool {
    text.starts_with("sha256:") && text.chars().count() == 71
}

/// Lowercase-hex tail gate (oracle `_teacher_gate.py:230-232`): exactly
/// `[0-9a-f]`; the `starts_with` prefix is 7 ASCII chars, so `[7..]` is always
/// a char boundary here.
fn hexpart_ok(text: &str) -> bool {
    text[7..]
        .chars()
        .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// Oracle-identical shape text (`_teacher_gate.py:229`): `repr` is the
/// attached-staged Python `repr(value)`, never a Rust rendering.
fn shape_message(name: &str, repr: &str) -> String {
    format!("{name} must be sha256:<64 hex>, got {repr}")
}

/// Oracle-identical hex text (`_teacher_gate.py:232`).
fn hex_message(name: &str, repr: &str) -> String {
    format!("{name} invalid hex: {repr}")
}

/// Strict sha gate core over staged plain data (no Python API): `text` is
/// `None` for non-`str` values, which reject with their staged `repr`.
fn check_require_sha256(name: &str, text: Option<&str>, repr: &str) -> Result<String, String> {
    let body = match text {
        Some(body) => body,
        None => return Err(shape_message(name, repr)),
    };
    if !sha_shape_ok(body) {
        return Err(shape_message(name, repr));
    }
    if !hexpart_ok(body) {
        return Err(hex_message(name, repr));
    }
    Ok(body.to_owned())
}

/// Validated digest text (mirrors `_require_sha256` bit-for-bit: any non-`str`
/// value rejects with its `{value!r}` rendering; `name` stages via `str()` so
/// non-`str` names render exactly like the oracle f-string — per
/// `loop_state.rs:100-102`). Translators call positionally; the bridge raises
/// `ValueError`, Python maps to `ContractError` with byte-identical text.
/// Compute runs detached with zero Python API inside.
#[pyfunction]
#[pyo3(signature = (name, value))]
fn distill_require_sha256(
    py: Python<'_>,
    name: Bound<'_, PyAny>,
    value: Bound<'_, PyAny>,
) -> PyResult<String> {
    let name_text = name.str()?.to_str()?.to_owned();
    let repr_text = value.repr()?.to_str()?.to_owned();
    let staged: Option<String> = value.extract().ok();
    let verdict = py.detach(|| check_require_sha256(&name_text, staged.as_deref(), &repr_text));
    verdict.map_err(PyValueError::new_err)
}

/// Gate-hash core over owned plain data (no Python API): the `analysis` kind
/// carries the WP-12 analysis spec hash verbatim after the strict shape gate
/// (oracle `_teacher_gate.py:394-395`); every other kind seals the content
/// digest over the real gate digests (oracle `_teacher_gate.py:396-402`).
fn gate_hash_value(
    kind: &str,
    candidate_spec_hash: &str,
    analysis_gate_digest: &str,
    report_hash: &str,
    analysis_spec_hash: Option<&str>,
    analysis_repr: &str,
) -> Result<String, String> {
    if kind == "analysis" {
        return check_require_sha256("gate_hash:analysis", analysis_spec_hash, analysis_repr);
    }
    let mut obj = serde_json::Map::with_capacity(4);
    obj.insert("kind".to_owned(), Value::String(kind.to_owned()));
    obj.insert(
        "candidate_spec_hash".to_owned(),
        Value::String(candidate_spec_hash.to_owned()),
    );
    obj.insert(
        "analysis_gate_digest".to_owned(),
        Value::String(analysis_gate_digest.to_owned()),
    );
    obj.insert(
        "report_hash".to_owned(),
        Value::String(report_hash.to_owned()),
    );
    let bytes = hydra_feed::canon::canonical_bytes_value(
        &Value::Object(obj),
        "bridge:distill_gate_hash_for_kind",
    )
    .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Bind a gate kind to real outcome-record digests (never synthesized).
/// Translators call positionally with `str` gate fields (the oracle `str()`s
/// them at `_teacher_gate.py:207-210`); the bridge raises `ValueError`,
/// Python maps to `ContractError`. Compute runs detached with zero Python API
/// inside.
#[pyfunction]
#[pyo3(signature = (kind, candidate_spec_hash, analysis_gate_digest, report_hash, analysis_spec_hash))]
fn distill_gate_hash_for_kind(
    py: Python<'_>,
    kind: String,
    candidate_spec_hash: String,
    analysis_gate_digest: String,
    report_hash: String,
    analysis_spec_hash: Bound<'_, PyAny>,
) -> PyResult<String> {
    let analysis_repr = analysis_spec_hash.repr()?.to_str()?.to_owned();
    let analysis_text: Option<String> = analysis_spec_hash.extract().ok();
    let verdict = py.detach(|| {
        gate_hash_value(
            &kind,
            &candidate_spec_hash,
            &analysis_gate_digest,
            &report_hash,
            analysis_text.as_deref(),
            &analysis_repr,
        )
    });
    verdict.map_err(PyValueError::new_err)
}

/// Trajectory mask/policy/vector decisions over owned plain data (no Python
/// API): the `__post_init__` gates at `_teacher_records.py:58-76` with
/// byte-identical texts. `policy_sum` + every `{p}`/`{v}`/`{s}` rendering is
/// staged Python-side (`sum()`/`repr()`), so the detach only compares —
/// including the explicit `NaN` disjunct the `isclose` illegal-mass gate
/// needs (all `NaN` comparisons are false).
#[allow(clippy::too_many_arguments)]
fn check_trajectory_values(
    legal_mask: &[bool],
    teacher_policy: &[f64],
    policy_sum: f64,
    policy_sum_repr: &str,
    policy_reprs: &[String],
    vector_return: &[f64],
    vector_reprs: &[String],
) -> Result<(), String> {
    if legal_mask.len() != teacher_policy.len() {
        return Err("legal_mask and teacher_policy length mismatch".to_owned());
    }
    if legal_mask.is_empty() {
        return Err("legal_mask empty".to_owned());
    }
    if !legal_mask.iter().any(|m| *m) {
        return Err("legal_mask all false — nonterminal must have legal".to_owned());
    }
    if policy_reprs.len() != teacher_policy.len() {
        return Err("policy_reprs length must match teacher_policy".to_owned());
    }
    if vector_reprs.len() != vector_return.len() {
        return Err("vector_reprs length must match vector_return".to_owned());
    }
    // `isclose` is false for a `NaN` sum while every float comparison is
    // also false, so `NaN` needs its own disjunct (same reason as below).
    if policy_sum.is_nan() || (policy_sum - 1.0).abs() > 1e-6 {
        return Err(format!("teacher_policy sum {policy_sum_repr} !=1"));
    }
    for ((policy, legal), rep) in teacher_policy
        .iter()
        .zip(legal_mask.iter())
        .zip(policy_reprs.iter())
    {
        if !legal {
            if policy.is_nan() || (policy - 0.0).abs() > 1e-9 {
                return Err(format!("illegal action has non-zero prob {rep}"));
            }
        } else if *policy < -1e-9 || !policy.is_finite() {
            return Err(format!("legal prob invalid {rep}"));
        }
    }
    for (value, rep) in vector_return.iter().zip(vector_reprs.iter()) {
        if !value.is_finite() {
            return Err(format!("vector_return non-finite {rep}"));
        }
    }
    Ok(())
}

/// Validate a trajectory record's mask/policy/vector Return `()` on success.
/// Translators call positionally with plain sequences plus the staged
/// `sum()`/`repr()` values; the bridge raises `ValueError`, Python maps to
/// `ContractError` with byte-identical text. Compute runs detached with zero
/// Python API inside.
#[pyfunction]
#[pyo3(signature = (legal_mask, teacher_policy, policy_sum, policy_sum_repr, policy_reprs, vector_return, vector_reprs))]
#[allow(clippy::too_many_arguments)]
fn distill_validate_trajectory_values(
    py: Python<'_>,
    legal_mask: Vec<bool>,
    teacher_policy: Vec<f64>,
    policy_sum: f64,
    policy_sum_repr: String,
    policy_reprs: Vec<String>,
    vector_return: Vec<f64>,
    vector_reprs: Vec<String>,
) -> PyResult<()> {
    let verdict = py.detach(|| {
        check_trajectory_values(
            &legal_mask,
            &teacher_policy,
            policy_sum,
            &policy_sum_repr,
            &policy_reprs,
            &vector_return,
            &vector_reprs,
        )
    });
    verdict.map_err(PyValueError::new_err)
}

/// Stage one JSON number type-faithfully (`bool` → `Bool`, `int` →
/// integer `Number`, `float` → `from_f64`): an `int` policy entry seals as
/// `1`, never `1.0`. Out-of-range integers and non-finite floats fail as
/// `ValueError`; anything else surfaces the pyo3 `TypeError`.
fn stage_json_number(obj: &Bound<'_, PyAny>, record: &str) -> PyResult<Value> {
    if obj.is_instance_of::<PyBool>() {
        let flag: bool = obj.extract()?;
        return Ok(Value::Bool(flag));
    }
    if obj.is_instance_of::<PyFloat>() {
        let num: f64 = obj.extract()?;
        return serde_json::Number::from_f64(num)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err(format!("{record}: non-finite float")));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(int) = obj.extract::<i64>() {
            return Ok(Value::from(int));
        }
        if let Ok(uint) = obj.extract::<u64>() {
            return Ok(Value::from(uint));
        }
        return Err(PyValueError::new_err(format!(
            "{record}: integer out of range"
        )));
    }
    let type_name = obj.get_type().name()?.to_str()?.to_owned();
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{record}: expected bool/int/float, got {type_name}"
    )))
}

/// Record-id seal core over staged plain data (no Python API): the
/// `make_trajectory_record` / `__post_init__` payload at
/// `_teacher_records.py:78-89,131-141`, sealed through the feed owners so the
/// digest is bit-identical to the retired
/// `"sha256:" + hashlib.sha256(canonical_bytes(...)).hexdigest()` lines.
#[allow(clippy::too_many_arguments)]
fn seal_trajectory_record(
    observation_hash: Option<&str>,
    observation_repr: &str,
    legal_mask: &[bool],
    teacher_policy: &[Value],
    vector_return: &[Value],
    event_label: Option<&str>,
    belief_label: Option<&[Value]>,
    teacher_spec_hash: &str,
    budget_doc: &[u8],
    provenance_doc: &[u8],
) -> Result<String, String> {
    let obs = check_require_sha256("observation_hash", observation_hash, observation_repr)?;
    let budget = hydra_feed::canon::parse_canonical_bytes(
        budget_doc,
        "bridge:distill_trajectory_record_id:budget",
    )
    .map_err(|e| format!("budget doc parse rejected: {e:?}"))?;
    let provenance = hydra_feed::canon::parse_canonical_bytes(
        provenance_doc,
        "bridge:distill_trajectory_record_id:provenance",
    )
    .map_err(|e| format!("provenance doc parse rejected: {e:?}"))?;
    let mut obj = serde_json::Map::with_capacity(9);
    obj.insert("observation_hash".to_owned(), Value::String(obs));
    obj.insert(
        "legal_mask".to_owned(),
        Value::Array(legal_mask.iter().map(|m| Value::Bool(*m)).collect()),
    );
    obj.insert(
        "teacher_policy".to_owned(),
        Value::Array(teacher_policy.to_vec()),
    );
    obj.insert(
        "vector_return".to_owned(),
        Value::Array(vector_return.to_vec()),
    );
    obj.insert(
        "event_label".to_owned(),
        event_label.map_or(Value::Null, |label| Value::String(label.to_owned())),
    );
    obj.insert(
        "belief_label".to_owned(),
        belief_label.map_or(Value::Null, |belief| Value::Array(belief.to_vec())),
    );
    obj.insert(
        "teacher_spec_hash".to_owned(),
        Value::String(teacher_spec_hash.to_owned()),
    );
    obj.insert("budget".to_owned(), budget);
    obj.insert("provenance".to_owned(), provenance);
    let bytes = hydra_feed::canon::canonical_bytes_value(
        &Value::Object(obj),
        "bridge:distill_trajectory_record_id",
    )
    .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Hash the trajectory payload to its `record_id`. Translators call
/// positionally: `observation_hash` (shape-gated here, error name
/// `observation_hash`), plain mask/policy/vector sequences (numbers staged
/// type-faithfully), optional labels, the pass-through `teacher_spec_hash`,
/// and the `budget`/`provenance` canonical-doc bytes the translator prepares
/// with the Python `canonical_bytes`. The bridge raises `ValueError`, Python
/// maps to `ContractError`. Compute runs detached with zero Python API
/// inside.
#[pyfunction]
#[pyo3(signature = (observation_hash, legal_mask, teacher_policy, vector_return, event_label, belief_label, teacher_spec_hash, budget_doc, provenance_doc))]
#[allow(clippy::too_many_arguments)]
fn distill_trajectory_record_id(
    py: Python<'_>,
    observation_hash: Bound<'_, PyAny>,
    legal_mask: Vec<bool>,
    teacher_policy: Vec<Bound<'_, PyAny>>,
    vector_return: Vec<Bound<'_, PyAny>>,
    event_label: Option<String>,
    belief_label: Option<Vec<Bound<'_, PyAny>>>,
    teacher_spec_hash: String,
    budget_doc: Vec<u8>,
    provenance_doc: Vec<u8>,
) -> PyResult<String> {
    let observation_repr = observation_hash.repr()?.to_str()?.to_owned();
    let observation_text: Option<String> = observation_hash.extract().ok();
    let mut policy_vals = Vec::with_capacity(teacher_policy.len());
    for (idx, item) in teacher_policy.iter().enumerate() {
        policy_vals.push(stage_json_number(
            item,
            &format!("bridge:distill_trajectory_record_id:teacher_policy[{idx}]"),
        )?);
    }
    let mut vector_vals = Vec::with_capacity(vector_return.len());
    for (idx, item) in vector_return.iter().enumerate() {
        vector_vals.push(stage_json_number(
            item,
            &format!("bridge:distill_trajectory_record_id:vector_return[{idx}]"),
        )?);
    }
    let belief_vals: Option<Vec<Value>> = belief_label
        .as_ref()
        .map(|belief| {
            belief
                .iter()
                .enumerate()
                .map(|(idx, item)| {
                    stage_json_number(
                        item,
                        &format!("bridge:distill_trajectory_record_id:belief_label[{idx}]"),
                    )
                })
                .collect::<PyResult<Vec<Value>>>()
        })
        .transpose()?;
    let verdict = py.detach(|| {
        seal_trajectory_record(
            observation_text.as_deref(),
            &observation_repr,
            &legal_mask,
            &policy_vals,
            &vector_vals,
            event_label.as_deref(),
            belief_vals.as_deref(),
            &teacher_spec_hash,
            &budget_doc,
            &provenance_doc,
        )
    });
    verdict.map_err(PyValueError::new_err)
}

/// Register the distill consts/fns on the shared `contracts` submodule
/// (mirrors `action_artifact::register` on `contracts`): compute detached,
/// wrap attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("TEACHER_CANDIDATES", PyTuple::new(py, TEACHER_CANDIDATES)?)?;
    sub.add(
        "REJECTED_CANDIDATES",
        PyFrozenSet::new(py, REJECTED_CANDIDATES)?,
    )?;
    sub.add("TEACHER_GATE_KINDS", PyTuple::new(py, TEACHER_GATE_KINDS)?)?;
    let analysis_ids = PyDict::new(py);
    for (teacher, analysis) in ANALYSIS_ID_FOR_TEACHER {
        analysis_ids.set_item(teacher, analysis)?;
    }
    sub.add("ANALYSIS_ID_FOR_TEACHER", analysis_ids)?;
    sub.add("DISTILL_REAL_FEATURE_DIM", DISTILL_REAL_FEATURE_DIM)?;
    sub.add("DISTILL_DEFAULT_NUM_ACTIONS", DISTILL_DEFAULT_NUM_ACTIONS)?;
    sub.add(
        "DISTILL_TRAINING_NAMESPACE_TOKEN",
        DISTILL_TRAINING_NAMESPACE_TOKEN,
    )?;
    sub.add_function(wrap_pyfunction!(distill_require_sha256, sub)?)?;
    sub.add_function(wrap_pyfunction!(distill_gate_hash_for_kind, sub)?)?;
    sub.add_function(wrap_pyfunction!(distill_validate_trajectory_values, sub)?)?;
    sub.add_function(wrap_pyfunction!(distill_trajectory_record_id, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str =
        "sha256:ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12";

    fn repr_of(text: &str) -> String {
        format!("'{text}'")
    }

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Hand-derived from the oracle literals (`_teacher_gate.py:33-70`,
        // `_teacher_records.py:161`); any drift fails here first.
        assert_eq!(
            TEACHER_CANDIDATES,
            [
                "candidate0",
                "candidate1",
                "candidate2",
                "candidate3",
                "candidate4",
                "candidate5",
                "candidate6",
            ]
        );
        assert_eq!(REJECTED_CANDIDATES, ["candidate4"]);
        assert_eq!(
            TEACHER_GATE_KINDS,
            ["contract", "exact", "search", "match", "analysis"]
        );
        assert_eq!(
            ANALYSIS_ID_FOR_TEACHER,
            [
                ("candidate0", "candidate0"),
                ("candidate1", "candidate1"),
                ("candidate2", "candidate2"),
                ("candidate3", "candidate3_pbrf_core_v1"),
                ("candidate5", "candidate5"),
                ("candidate6", "candidate6"),
            ]
        );
        assert_eq!(DISTILL_REAL_FEATURE_DIM, 48);
        assert_eq!(DISTILL_DEFAULT_NUM_ACTIONS, 32);
        assert_eq!(DISTILL_TRAINING_NAMESPACE_TOKEN, "training_namespace_v1");
    }

    #[test]
    fn require_sha256_accepts_wellformed() {
        assert_eq!(
            check_require_sha256("observation_hash", Some(DIGEST_A), &repr_of(DIGEST_A)),
            Ok(DIGEST_A.to_owned())
        );
    }

    #[test]
    fn require_sha256_rejects_shape_with_oracle_text() {
        // Oracle `_teacher_gate.py:229`.
        assert_eq!(
            check_require_sha256("observation_hash", Some("sha256:xyz"), "'sha256:xyz'"),
            Err("observation_hash must be sha256:<64 hex>, got 'sha256:xyz'".to_owned())
        );
        assert_eq!(
            check_require_sha256("digest", None, "None"),
            Err("digest must be sha256:<64 hex>, got None".to_owned())
        );
        assert_eq!(
            check_require_sha256("digest", None, "123"),
            Err("digest must be sha256:<64 hex>, got 123".to_owned())
        );
    }

    #[test]
    fn require_sha256_rejects_bad_hex_with_oracle_text() {
        // Oracle `_teacher_gate.py:232`; 71 chars with one uppercase nibble.
        let bad = format!("sha256:{}A", "ab12".repeat(15) + "ab1");
        assert_eq!(bad.chars().count(), 71);
        let expected = format!("gate_hash:analysis invalid hex: '{bad}'");
        assert_eq!(
            check_require_sha256("gate_hash:analysis", Some(&bad), &repr_of(&bad)),
            Err(expected)
        );
    }

    #[test]
    fn gate_hash_analysis_kind_passes_spec_hash_through() {
        assert_eq!(
            gate_hash_value(
                "analysis",
                DIGEST_A,
                DIGEST_A,
                DIGEST_A,
                Some(DIGEST_A),
                &repr_of(DIGEST_A)
            ),
            Ok(DIGEST_A.to_owned())
        );
    }

    #[test]
    fn gate_hash_analysis_kind_rejects_misshapen_spec_hash() {
        assert_eq!(
            gate_hash_value(
                "analysis",
                DIGEST_A,
                DIGEST_A,
                DIGEST_A,
                Some("nope"),
                "'nope'"
            ),
            Err("gate_hash:analysis must be sha256:<64 hex>, got 'nope'".to_owned())
        );
    }

    #[test]
    fn gate_hash_contract_kind_matches_live_oracle_golden() {
        // Live-oracle digest from the wave-14 parity smoke
        // (`_gate_hash_for_kind("contract", candidate_spec_hash=h,
        // gate={"digest": h, "report_hash": h, ...})` with
        // `h = "sha256:" + "ab12" * 16`).
        assert_eq!(
            gate_hash_value(
                "contract",
                DIGEST_A,
                DIGEST_A,
                DIGEST_A,
                Some(DIGEST_A),
                &repr_of(DIGEST_A)
            ),
            Ok(
                "sha256:f4e46ac00b9128e9197af771971174368b4b6815ec3ae400dbb45be1c642e23a"
                    .to_owned()
            )
        );
    }

    #[test]
    fn gate_hash_kinds_differ() {
        let contract = gate_hash_value(
            "contract",
            DIGEST_A,
            DIGEST_A,
            DIGEST_A,
            Some(DIGEST_A),
            &repr_of(DIGEST_A),
        );
        let exact = gate_hash_value(
            "exact",
            DIGEST_A,
            DIGEST_A,
            DIGEST_A,
            Some(DIGEST_A),
            &repr_of(DIGEST_A),
        );
        assert!(contract.is_ok());
        assert!(exact.is_ok());
        assert_ne!(contract.unwrap(), exact.unwrap());
    }

    #[test]
    fn trajectory_values_accept_valid_distribution() {
        assert_eq!(
            check_trajectory_values(
                &[true, false],
                &[1.0, 0.0],
                1.0,
                "1.0",
                &["1.0".to_owned(), "0.0".to_owned()],
                &[0.5, -0.25, 0.0, 1.5],
                &[
                    "0.5".to_owned(),
                    "-0.25".to_owned(),
                    "0.0".to_owned(),
                    "1.5".to_owned()
                ],
            ),
            Ok(())
        );
    }

    #[test]
    fn trajectory_values_reject_with_oracle_texts() {
        // Oracle `_teacher_records.py:58-76`, one arm per message.
        assert_eq!(
            check_trajectory_values(
                &[true],
                &[1.0, 0.0],
                1.0,
                "1.0",
                &["1.0".to_owned(), "0.0".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("legal_mask and teacher_policy length mismatch".to_owned())
        );
        assert_eq!(
            check_trajectory_values(&[], &[], 0.0, "0", &[], &[], &[]),
            Err("legal_mask empty".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[false, false],
                &[0.5, 0.5],
                1.0,
                "1.0",
                &["0.5".to_owned(), "0.5".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("legal_mask all false — nonterminal must have legal".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[true, true],
                &[0.25, 0.25],
                0.5,
                "0.5",
                &["0.25".to_owned(), "0.25".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("teacher_policy sum 0.5 !=1".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[true, false],
                &[0.5, 0.5],
                1.0,
                "1.0",
                &["0.5".to_owned(), "0.5".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("illegal action has non-zero prob 0.5".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[true],
                &[f64::NAN],
                f64::NAN,
                "nan",
                &["nan".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("teacher_policy sum nan !=1".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[true],
                &[f64::INFINITY],
                1.0,
                "1.0",
                &["inf".to_owned()],
                &[0.0],
                &["0.0".to_owned()],
            ),
            Err("legal prob invalid inf".to_owned())
        );
        assert_eq!(
            check_trajectory_values(
                &[true],
                &[1.0],
                1.0,
                "1.0",
                &["1.0".to_owned()],
                &[f64::NEG_INFINITY],
                &["-inf".to_owned()],
            ),
            Err("vector_return non-finite -inf".to_owned())
        );
    }

    #[test]
    fn trajectory_record_id_matches_live_oracle_golden() {
        // Live-oracle digest from the wave-14 parity smoke
        // (`make_trajectory_record(case_id="case_00000", actor=0,
        // observation_hash=h, legal_mask=(True, False),
        // teacher_policy=(1.0, 0.0), vector_return=(0.5, -0.25, 0.0, 1.5),
        // teacher_spec_hash=h, budget={"mode": "gameplay_5s",
        // "deadline_ms": 5000}, provenance={"case_id": "case_00000"})`
        // with `h = "sha256:" + "ab12" * 16`).
        let budget = serde_json::json!({"deadline_ms": 5000, "mode": "gameplay_5s"});
        let budget_doc = hydra_feed::canon::canonical_bytes_value(&budget, "test:budget").unwrap();
        let provenance = serde_json::json!({"case_id": "case_00000"});
        let provenance_doc =
            hydra_feed::canon::canonical_bytes_value(&provenance, "test:provenance").unwrap();
        let policy = [Value::from(1.0), Value::from(0.0)];
        let vector = [
            Value::from(0.5),
            Value::from(-0.25),
            Value::from(0.0),
            Value::from(1.5),
        ];
        assert_eq!(
            seal_trajectory_record(
                Some(DIGEST_A),
                &repr_of(DIGEST_A),
                &[true, false],
                &policy,
                &vector,
                None,
                None,
                DIGEST_A,
                &budget_doc,
                &provenance_doc,
            ),
            Ok(
                "sha256:fd609a93976640f8a90e9fa27c1d50c03a81a87670fdfb798edfa91cd45e2532"
                    .to_owned()
            )
        );
    }

    #[test]
    fn trajectory_record_id_is_field_sensitive() {
        let budget = serde_json::json!({"deadline_ms": 5000, "mode": "gameplay_5s"});
        let budget_doc = hydra_feed::canon::canonical_bytes_value(&budget, "test:budget").unwrap();
        let provenance = serde_json::json!({"case_id": "case_00000"});
        let provenance_doc =
            hydra_feed::canon::canonical_bytes_value(&provenance, "test:provenance").unwrap();
        let seal = |policy: &[Value]| {
            seal_trajectory_record(
                Some(DIGEST_A),
                &repr_of(DIGEST_A),
                &[true, false],
                policy,
                &[Value::from(0.0)],
                None,
                None,
                DIGEST_A,
                &budget_doc,
                &provenance_doc,
            )
            .unwrap()
        };
        let int_policy = [Value::from(1), Value::from(0)];
        let float_policy = [Value::from(1.0), Value::from(0.0)];
        // JCS number form identifies int `1` with float `1.0` (both seal as
        // `1`), so the digests MUST agree — verified against the Python
        // canon owner, not eyeballed.
        assert_eq!(seal(&int_policy), seal(&float_policy));
        // Field sensitivity proper: a different policy value seals differently.
        let other_policy = [Value::from(2), Value::from(0)];
        assert_ne!(seal(&int_policy), seal(&other_policy));
    }

    #[test]
    fn pyfn_wiring_smoke() {
        Python::initialize();
        Python::attach(|py| {
            let name = pyo3::types::PyString::new(py, "observation_hash");
            let value = pyo3::types::PyString::new(py, DIGEST_A);
            let out = distill_require_sha256(py, name.into_any(), value.into_any());
            assert_eq!(out.unwrap(), DIGEST_A.to_owned());
            let bad_name = pyo3::types::PyString::new(py, "digest");
            let bad_value = pyo3::types::PyString::new(py, "nope");
            let err = distill_require_sha256(py, bad_name.into_any(), bad_value.into_any());
            assert!(err.is_err());
        });
    }
}
