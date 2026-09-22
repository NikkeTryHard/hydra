//! record_leaves: census-verified record-shape leftovers on the EXISTING
//! `hydra2._native.eval` + `hydra2._native.contracts` submodules.
//!
//! DAG: pyo3 + `hydra-feed` + `hydra-shard` (all already bridge dependencies,
//! see `crates/bridge/Cargo.toml:19-21`; no new dependency). FORBIDS:
//! torch/CUDA/triton/JAX/riichienv/arrow/parquet/zstd/IO/RNG/clocks/mutable-state
//! (all stay Python), float sums over data (every mean stays Python per the
//! wave-16 contract — the only float here is the opaque `gauss_next`
//! passthrough plus the `observed_estimate`/`confidence_bounds` scalars the
//! translator stages, never a reduction), and any new JCS printer or hasher
//! (all bytes/hash math lives in `hydra-feed::{canon, digest}`, referenced
//! below, never restated; the JSON transport parse below uses the same
//! `serde_json` the `feed::decode` owner parses with, never a second canon).
//!
//! TABLE ported here (frozen values + pure compute, oracles cited per use):
//! - `eval_case_to_json` <- `eval/case.py:80-88` dict projection (eval submodule).
//! - `record_to_json` <- `eval/promotion.py:190-211` canonical projection
//!   (gates sorted, excluded blocks listed; eval submodule).
//! - `record_require_attestation` <- `data/attestation.py:213-239` flag gate
//!   (designated owner `feed::gate`; no direct attestation fn exists there —
//!   the verdict vocabulary lives there, the D-017 sentences are frozen from
//!   HEAD text with the oracle line cited per arm, same NONE-owner restatement
//!   precedent as `common_roots.rs:38-45`; closest shape owner is
//!   `feed::rows::make_raw_object_row` non-empty attestation presence).
//! - `record_decode_json_line` <- `data/decode.py:46-48` transport parse
//!   (owner `feed::decode`; same `serde_json` the owner parses with,
//!   `START/END_TYPES` + `WALL_LEN` stay owned there, never recopied).
//! - `record_microbatch_bounds` <- `data/stream_decode.py:398-421` int/slice
//!   bounds (owner `feed::stream_driver` contiguous-prefix takes; same gate
//!   text as the `columnar.stream_decode_microbatch_bounds` twin, which stays
//!   the columnar site — this contracts site is the record-leaves alias per
//!   the wave-18 assignment, same core, never a fork).
//! - `record_count_decisions` <- `data/stream_decode.py:424-426` int sum over
//!   staged per-game event counts (translator stages the lens Python-side so
//!   live `StreamGame` objects never cross; int sums only, float sums stay
//!   Python).
//! - `RECORD_PARTITION_ORDER` <- `data/stream_manifest.py:109` via
//!   `hydra_feed::manifest::PARTITION_ORDER` (value-equal to
//!   `hydra_feed::partition::PARTITION_ORDER`; referenced, never recopied).
//! - `record_serialize_shuffle_rng` / `record_parse_shuffle_rng` <-
//!   `data/stream_manifest.py:424-464` shuffle-RNG codec (owner
//!   `hydra_feed::manifest::{serialize_shuffle_rng, parse_shuffle_rng,
//!   ShuffleRngState}`; the live `Random.getstate`/`setstate` stays Python,
//!   only plain `version/state/gauss_next` cross).
//! - shard shape/filename consts <- `data/shard_build.py:92-100`
//!   (`FIXED_HISTORY_T` references `hydra_shard::writer::SHARD_MAX_HISTORY`;
//!   `HISTORY_LEN_PLANE` + the two sidecar filenames have no crate const —
//!   frozen from HEAD text with the oracle line cited per use, same NONE-owner
//!   precedent; designated owner `shard::expand` documents both at
//!   `expand.rs:68-70`, never defines them).
//! - `RECORD_WALL_TILE_COUNT` <- `engines/protocol.py:44` via
//!   `hydra_feed::tiles::TILE_REV.len()` (owner `feed::tiles`; cross-checked
//!   against `feed::decode::WALL_LEN` + `feed::validate::WALL_LEN`, all 136).
//! - `RECORD_PBRF_CODES` + `record_pbrf_error_class` <-
//!   `contracts/common.py:135-153` code table (owner bridge `common_errors.rs`
//!   exception classes; this file only freezes the 17 code strings in oracle
//!   order plus the code->class-name view, never the classes themselves).
//! - `record_require_action_width` <- `training/dataset_encode.py:57-68` int
//!   gate vs the frozen baseline (owner `hydra_feed::census::CENSUS_TOTAL` =
//!   6792; cross-checked against `feed::ledger::ACTION_TABLE_LEN`,
//!   `bridge encoder::BASELINE_ACTIONS`, `ring::BASELINE_ACTIONS`).
//! - `RECORD_BASELINE_ACTIONS` <- the same census owner (referenced, never
//!   recopied).
//! - `record_validate_blocks_disjoint` <- `eval/duplicate.py:229-244`
//!   wall/game-id uniqueness across blocks (designated owner
//!   `hydra_search::eval::partition::validate_blocks_disjoint`,
//!   `crates/search/src/eval/partition.rs:181-199`; the owner details carry
//!   no id, so this leaf renders the oracle-exact `{id!r}` texts here —
//!   eval submodule).
//! - `record_verify_held_out_disjoint` <- `eval/baseline_metrics.py:333-343`
//!   train/held-out set disjointness (designated owner
//!   `hydra_search::eval::partition` split membership; eval submodule).
//! - `record_to_jsonable` <- `eval/baseline_eval.py:241-262` tuple→list/dict
//!   recursion (adjacent to the `hydra_feed::canon` JCS owner, which consumes
//!   only the list/dict domain — RFC 8785 arrays; contracts submodule).
//!
//! LOGIC staying Python (per-fn reasons, weightless trivials untouched):
//! - `EvalCase` / `PromotionRecord` / `Attestation` / `GameRecord` /
//!   `StreamGame` / `StreamManifest` / `WallSchedule` construction + every
//!   `ContractError` translator (the bridge raises `ValueError` with
//!   byte-identical oracle text; translators map, except the transport parse
//!   which propagates `ValueError` exactly like the oracle `json.loads`).
//! - `make_eval_case` / `make_promotion_record` / `load_attestation` /
//!   `decode_game_object` / `PrefetchGameStream` / `build_manifest` /
//!   `manifest_digest` / `write/read_reservoir_blob` / `load/save_scan_cache`
//!   / `build_shards` / `validate_manifest` / `tensorize_actor_row` /
//!   `encode_observation_rows` (torch/IO/orchestration owned).
//! - `deep_merge` / sidecar / row-kind / wall-override consts (weightless
//!   trivials per the assignment — deliberately NOT ported).
//! - `__all__` in every touched Python file (unchanged by contract).
//!
//! Argument contract: dict-domain inputs take `Bound<PyDict>` (non-dict inputs
//! fail argument extraction with `TypeError` and the Python translator falls
//! back to the byte-identical oracle body — the `rc_sections.rs:159`
//! `Bound<PyDict>` precedent); live dataclasses never cross (the translator
//! unpacks to plain scalars first — the `eval_leaves.rs:21-24` caller-projects
//! precedent); every gate is infallible over its valid domain (per-arm
//! messages mirror the oracle f-strings byte-exact via staged `repr`/`str`).
//!
//! Shape per fn: attached staging (`repr`/`str` via the live Python API, so
//! `{value!r}` / `{value}` render byte-exact — per `rc_require.rs:60-72`) →
//! ONE `py.detach(|| …)` over owned plain data with zero Python API inside
//! (per `validate.rs:83-95`) → attached wrap (`PyDict::new` + `set_item` per
//! `contracts.rs:1227-1231`, `PyTuple::new` per `contracts.rs:1156`).
//! Attached-only assembly (`eval_case_to_json`, final dict wraps) runs
//! end-to-end attached (the walk touches Python memory throughout — the
//! `canon_rng.rs:263` attached-only precedent).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed `sub`
//! in `wrap_pyfunction!(f, sub)` mirror `validate::register`
//! (`crates/bridge/src/validate.rs:111-127`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); `PyTuple::new` mirrors
//! `contracts.rs:1156`; scalar `sub.add` mirrors `belief_kernel::register`
//! (`crates/bridge/src/belief_kernel.rs:254-263`); `Bound<PyDict>` params
//! mirror `rc_sections.rs:159`; `is_instance_of::<PyBool>` bool-rejection
//! mirrors `contracts::plain_int_value` (`crates/bridge/src/contracts.rs:69-74`);
//! `Bound<'_, PyAny>` scalar staging mirrors `qual_budget::as_u64`
//! (`crates/bridge/src/qual_budget.rs:230-235`); attached `repr`/`str`
//! mirrors `rc_require.rs:63-65`; single-quote `py_str_list` mirrors
//! `promotion.rs:131-143`; `Vec<u8>` → `bytes` never used here (the JSON
//! transport returns live objects via the manual `Value`→Python fold below,
//! same totality note as `stream_decode.rs:35-36` live-object staging).
//!
//! Feed/shard owners (never reimplemented here): partition order
//! `hydra_feed::manifest::PARTITION_ORDER` (`crates/feed/src/manifest.rs:65-66`,
//! value-equal to `hydra_feed::partition::PARTITION_ORDER`,
//! `crates/feed/src/partition.rs:65-66`); wall length via
//! `hydra_feed::tiles::TILE_REV` (`crates/feed/src/tiles.rs:96`, 136 entries)
//! cross-checked against `hydra_feed::decode::WALL_LEN`
//! (`crates/feed/src/decode.rs:58`) + `hydra_feed::validate::WALL_LEN`
//! (`crates/feed/src/validate.rs:73`); shuffle codec
//! `hydra_feed::manifest::{serialize_shuffle_rng, parse_shuffle_rng,
//! ShuffleRngState}` (`crates/feed/src/manifest.rs:597-682`); census width
//! `hydra_feed::census::CENSUS_TOTAL` (`crates/feed/src/census.rs:28`, 6792)
//! cross-checked against `hydra_feed::ledger::ACTION_TABLE_LEN`
//! (`crates/feed/src/ledger.rs:37`); shard history top
//! `hydra_shard::writer::SHARD_MAX_HISTORY` (`crates/shard/src/writer.rs:238`,
//! 256); PBRF classes owned by `crate::common_errors` (this file only names
//! them, never defines them).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - `record_decode_json_line` success values are JSON-value-identical;
//!   error texts diverge (serde vs CPython `json`) but both raise
//!   `ValueError` (`json.JSONDecodeError` subclasses `ValueError`); absurd-huge
//!   ints (`> u64::MAX`) land as `f64` here vs exact `int` there (realistic
//!   game lines never carry them).
//! - `record_parse_shuffle_rng` unknown-key list renders with single quotes
//!   (`['k']`) exactly like the oracle for identifier keys; exotic keys
//!   containing quotes/backslashes render single-quote-wrapped without the
//!   full `repr` escapes (realistic sidecars never carry them).
//! - `record_count_decisions` overflows fail closed (`ValueError`, new
//!   hardening) where the oracle arbitrary-int sum would succeed (realistic
//!   corpora never approach `u64::MAX` events).
//! - `record_pbrf_error_class` on unknown codes raises `ValueError` (new
//!   hardening) where the oracle dict lookup would raise `KeyError`
//!   (translators only call with the frozen 17, so unreachable).
//! - Non-`bool` `allow_synthetic` truthiness: the oracle admits any truthy,
//!   this leaf takes strict `bool` (translators always pass `bool`).
//! - `record_validate_blocks_disjoint` / `record_verify_held_out_disjoint`
//!   `{id!r}` / overlap-list slots render single-quote-wrapped without the
//!   full `repr` escapes (exact for ids without quotes/backslashes — the
//!   closed wall/game-id domain, same note as `eval_schedule::wall_repr`);
//!   non-`str` ids never reach here (translators fall back to the oracle
//!   body on extraction `TypeError`).
//! - `record_to_jsonable` depth guard (4096) raises `ValueError`; the
//!   translator falls back to the verbatim oracle body on `ValueError`, so
//!   deep docs raise the oracle `RecursionError` exactly like before — no
//!   divergence on any input.
//!   Single-cdylib tree: registers on the EXISTING shared submodules —
//!   contracts leaves via `register` (mirrors `common_roots::register`), eval
//!   leaves via `register_eval` (mirrors `eval_blocks::register`); no new entry
//!   point. MAIN wiring:
//!   `pub mod record_leaves;` in `lib.rs` (alphabetical, after `qual_replay`)
//!   plus `crate::record_leaves::register(&sub)?;` in `contracts::register`
//!   next to `crate::tracking_leaves::register(&sub)?;` (`contracts.rs:1324`)
//!   plus `crate::record_leaves::register_eval(&sub)?;` in `eval::register`
//!   next to `crate::eval_leaves2::register(&sub)?;` (`eval.rs:130`).

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyModule, PyString,
    PyTuple,
};

// ---------------------------------------------------------------------------
// Frozen consts (referenced owners, never recopied law).
// ---------------------------------------------------------------------------

/// Partition names in cumulative-threshold order (`stream_manifest.py:109`
/// via `hydra_feed::manifest::PARTITION_ORDER`; value-equal to
/// `hydra_feed::partition::PARTITION_ORDER`).
const RECORD_PARTITION_ORDER: [&str; 5] = hydra_feed::manifest::PARTITION_ORDER;

/// Complete wall length (`protocol.py:44` via `hydra_feed::tiles::TILE_REV.len()`;
/// cross-checked against `hydra_feed::decode::WALL_LEN` (136) in tests).
const RECORD_WALL_TILE_COUNT: usize = hydra_feed::tiles::TILE_REV.len();

/// Fixed history width (`shard_build.py:92` via
/// `hydra_shard::writer::SHARD_MAX_HISTORY`; the designated
/// `shard::expand` documents the same 256 at `expand.rs:68-70`).
const RECORD_FIXED_HISTORY_T: usize = hydra_shard::writer::SHARD_MAX_HISTORY;

/// Extra plane carrying per-row true history lengths (`shard_build.py:95`;
/// no crate const exists — frozen from HEAD text).
const RECORD_HISTORY_LEN_PLANE: &str = "history_len";

/// Row-identity sidecar filenames (`shard_build.py:99-100`; no crate const
/// exists — frozen from HEAD text).
const RECORD_DECISION_IDS_FILENAME: &str = "decision_ids.json";
/// Row-identity sidecar filenames (`shard_build.py:100`; frozen from HEAD text).
const RECORD_OBSERVATION_HASHES_FILENAME: &str = "observation_hashes.json";

/// Frozen baseline action width (`dataset_encode.py:61` via
/// `hydra_feed::census::CENSUS_TOTAL`; cross-checked against
/// `hydra_feed::ledger::ACTION_TABLE_LEN` in tests).
const RECORD_BASELINE_ACTIONS: usize = hydra_feed::census::CENSUS_TOTAL;

/// PBRF code vocabulary in oracle order (`common.py:135-153`, 17 entries).
const RECORD_PBRF_CODES: [&str; 17] = [
    "PBRF_PARTITION_EMPTY",
    "PBRF_PARTITION_ALIAS",
    "PBRF_PARTITION_MASS",
    "PBRF_PARTITION_CHILD_NORM",
    "PBRF_STALE_EPOCH",
    "PBRF_STALE_TARGET",
    "PBRF_STALE_PARENT",
    "PBRF_STALE_PROVENANCE",
    "PBRF_STALE_WORLDREF",
    "PBRF_DIGEST_DELTA",
    "PBRF_DIGEST_WORLD_ID",
    "PBRF_VIS_TREE_KEY",
    "PBRF_VIS_TREE_KEY_NESTED",
    "PBRF_VIS_POLICY_WORLD",
    "PBRF_VIS_POLICY_HANDS",
    "PBRF_SUPPORT_REGION",
    "PBRF_SUPPORT_POINT",
];

/// Code → owning exception-class name (`common_errors.rs`-owned classes;
/// this table only names them, never defines them).
const RECORD_PBRF_TABLE: [(&str, &str); 17] = [
    ("PBRF_PARTITION_EMPTY", "PacketPartitionError"),
    ("PBRF_PARTITION_ALIAS", "PacketPartitionError"),
    ("PBRF_PARTITION_MASS", "PacketPartitionError"),
    ("PBRF_PARTITION_CHILD_NORM", "PacketPartitionError"),
    ("PBRF_STALE_EPOCH", "StaleBeliefError"),
    ("PBRF_STALE_TARGET", "StaleBeliefError"),
    ("PBRF_STALE_PARENT", "StaleBeliefError"),
    ("PBRF_STALE_PROVENANCE", "StaleBeliefError"),
    ("PBRF_STALE_WORLDREF", "StaleBeliefError"),
    ("PBRF_DIGEST_DELTA", "DigestMismatchError"),
    ("PBRF_DIGEST_WORLD_ID", "DigestMismatchError"),
    ("PBRF_VIS_TREE_KEY", "VisibilityViolationError"),
    ("PBRF_VIS_TREE_KEY_NESTED", "VisibilityViolationError"),
    ("PBRF_VIS_POLICY_WORLD", "VisibilityViolationError"),
    ("PBRF_VIS_POLICY_HANDS", "VisibilityViolationError"),
    ("PBRF_SUPPORT_REGION", "ProposalSupportError"),
    ("PBRF_SUPPORT_POINT", "ProposalSupportError"),
];

// ---------------------------------------------------------------------------
// Attached staging helpers.
// ---------------------------------------------------------------------------

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `str(value)` text for every `{value}` slot
/// (mirrors `rc_require.rs:67-72` via `str()`; for ints equals `repr`).
fn py_str(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.str()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `type(value).__name__` text (per
/// `rc_require.rs:67-72`).
fn py_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_str()?.to_owned())
}

/// Python-`list` rendering of plain identifier keys (`['a', 'b']`, `[]` when
/// empty; mirrors `promotion.rs:131-143`). Domain: shuffle sidecar keys
/// (identifier spellings, never quotes) so single-quote joining is exact.
fn py_key_list(keys: &[String]) -> String {
    let mut out = String::from("[");
    for (index, key) in keys.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(key);
        out.push('\'');
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// Detached cores (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Attestation flag gate (`attestation.py:213-239`): `None` stages
/// `missing=true`; every other arm stages the owned field view. Returns
/// `Ok(())` when the value passes, else the byte-identical oracle sentence.
#[derive(Debug, Clone)]
struct AttestationView {
    missing: bool,
    kind: Option<String>,
    attestation_id: Option<String>,
    confidential_source_id: Option<String>,
    permitted_purpose_len: usize,
    permitted_purpose_present: bool,
    disclosure_class: Option<String>,
    acquisition_is_dict: bool,
    acquisition_len: usize,
    acquisition_dataset: String,
}

fn attestation_core(view: &AttestationView, allow_synthetic: bool) -> Result<(), String> {
    if view.missing {
        return Err(String::from(
            "attestation missing; cannot create authorized RawObjectRow (D-017 pending)",
        ));
    }
    let kind = view.kind.as_deref().unwrap_or("");
    let att_id = view.attestation_id.as_deref().unwrap_or("");
    if kind == "real" && att_id == "D-017" {
        if view
            .confidential_source_id
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            return Err(String::from(
                "real D-017 confidential_source_id must be non-empty",
            ));
        }
        if !view.permitted_purpose_present || view.permitted_purpose_len == 0 {
            return Err(String::from(
                "real D-017 permitted_purpose must be non-empty",
            ));
        }
        if view.disclosure_class.as_deref().unwrap_or("").is_empty() {
            return Err(String::from(
                "real D-017 disclosure_class must be non-empty",
            ));
        }
        if !view.acquisition_is_dict || view.acquisition_len == 0 {
            return Err(String::from(
                "real D-017 acquisition_metadata must be non-empty dict",
            ));
        }
        if !view.acquisition_dataset.is_empty()
            && !view.acquisition_dataset.contains("tenhou-houou")
        {
            return Err(format!(
                "real D-017 dataset must reference tenhou-houou, got {:?}",
                view.acquisition_dataset
            ));
        }
        return Ok(());
    }
    if kind == "synthetic" && !allow_synthetic {
        return Err(String::from(
            "synthetic attestation not permitted for this path",
        ));
    }
    if att_id.is_empty() {
        return Err(String::from("attestation_id must be non-empty"));
    }
    Ok(())
}

/// Microbatch gate plus slice bounds (`stream_decode.py:409-414` without the
/// generator or the shallow slicing over live games, which stay Python).
/// `size=None` stages every non-plain-int input (bool excluded, matching the
/// oracle `type(x) is not int` head); the `{!r}` slot carries the staged
/// live `repr`.
fn microbatch_bounds_core(
    batch_len: usize,
    size: Option<i64>,
    repr_text: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let step: usize = match size {
        Some(value) if value > 0 => usize::try_from(value)
            .map_err(|_| format!("microbatch_size must be a positive int, got {repr_text}"))?,
        _ => {
            return Err(format!(
                "microbatch_size must be a positive int, got {repr_text}"
            ));
        }
    };
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < batch_len {
        bounds.push((start, (start + step).min(batch_len)));
        start += step;
    }
    Ok(bounds)
}

/// Int-sum core (`stream_decode.py:426`): total events across staged counts.
/// Overflow fails closed (new hardening, unreachable for realistic corpora).
fn count_decisions_core(counts: &[u64]) -> Result<u64, String> {
    let mut total: u64 = 0;
    for count in counts {
        total = total
            .checked_add(*count)
            .ok_or_else(|| String::from("decision count overflow"))?;
    }
    Ok(total)
}

/// Shuffle-serialize gate (`stream_manifest.py:430-435`): version must be a
/// plain int, gauss must be `None` or a float. The state-tuple shape itself
/// is checked attached (`isinstance(tuple)`), so this core only gates the
/// two scalar slots.
fn serialize_shuffle_core(
    version_ok: bool,
    gauss_ok: bool,
    gauss_is_none: bool,
) -> Result<(), String> {
    if !version_ok {
        return Err(String::from("shuffle RNG state malformed"));
    }
    if !gauss_ok && !gauss_is_none {
        return Err(String::from("shuffle RNG gauss state malformed"));
    }
    Ok(())
}

/// Shuffle-parse gate (`stream_manifest.py:442-463`): unknown keys, version,
/// state, and gauss arms in oracle order with byte-identical sentences.
/// `unknown_sorted` is pre-sorted attached; `state_*` are staged plain-data
/// flags so this core stays total.
#[allow(clippy::too_many_arguments)]
fn parse_shuffle_core(
    is_mapping: bool,
    unknown_sorted: &[String],
    version_ok: bool,
    state_is_list: bool,
    state_len: usize,
    state_all_nonneg_int: bool,
    gauss_is_none: bool,
    gauss_is_bool: bool,
    gauss_is_number: bool,
) -> Result<(), String> {
    if !is_mapping {
        return Err(String::from("shuffle buffer_rng_state must be a mapping"));
    }
    if !unknown_sorted.is_empty() {
        return Err(format!(
            "shuffle buffer_rng_state unknown keys {}",
            py_key_list(unknown_sorted)
        ));
    }
    if !version_ok {
        return Err(String::from(
            "shuffle buffer_rng_state.version must be an int",
        ));
    }
    if !state_is_list || state_len == 0 || !state_all_nonneg_int {
        if !state_is_list || state_len == 0 {
            return Err(String::from(
                "shuffle buffer_rng_state.state must be a non-empty int list",
            ));
        }
        return Err(String::from(
            "shuffle buffer_rng_state.state must hold non-negative ints",
        ));
    }
    if !gauss_is_none && (gauss_is_bool || !gauss_is_number) {
        return Err(String::from(
            "shuffle buffer_rng_state.gauss_next must be a float or null",
        ));
    }
    Ok(())
}

/// Action-width gate (`dataset_encode.py:64-68`): the frozen baseline passes,
/// anything else needs the explicit test-only narrow flag. `num_is_baseline`
/// stages the `== 6792` comparison over the extracted int (non-int stages
/// false, so the oracle `!=` arm is preserved); `num_str` is the staged live
/// `str()` for the `{num_actions}` slot; `baseline` is the census owner.
fn action_width_core(
    num_is_baseline: bool,
    allow_narrow: bool,
    where_str: &str,
    num_str: &str,
    baseline: usize,
) -> Result<(), String> {
    if !num_is_baseline && !allow_narrow {
        return Err(format!(
            "{where_str}: num_actions {num_str} != baseline {baseline} requires allow_narrow=True (test-only narrow vocab)"
        ));
    }
    Ok(())
}

/// PBRF code → class-name view (`common.py:135-153` over the
/// `common_errors.rs`-owned classes; this table only names them).
fn pbrf_class_core(code: &str) -> Result<&'static str, String> {
    for (known, class) in RECORD_PBRF_TABLE {
        if known == code {
            return Ok(class);
        }
    }
    Err(format!("unknown PBRF code {code:?}"))
}

/// Set core for `validate_blocks_disjoint` (`duplicate.py:229-244` over the
/// `hydra_search::eval::partition::validate_blocks_disjoint` designated
/// owner, `crates/search/src/eval/partition.rs:181-199`): first duplicate
/// wall id, then first duplicate game id, in block order. The owner details
/// carry no id, so the oracle-exact `{id!r}` texts render here via
/// single-quote wrapping (exact for the closed-domain ids — same note as
/// `eval_schedule::wall_repr`, see the module divergence note). Pairing is
/// total by type (`Vec` of staged translator pairs).
fn blocks_disjoint_core(blocks: &[(String, Vec<String>)]) -> Result<(), String> {
    let mut walls = HashSet::new();
    let mut games = HashSet::new();
    for (wall_id, game_ids) in blocks {
        if !walls.insert(wall_id.as_str()) {
            return Err(format!("duplicate wall_id '{wall_id}' across blocks"));
        }
        for game_id in game_ids {
            if !games.insert(game_id.as_str()) {
                return Err(format!("game '{game_id}' appears in multiple blocks"));
            }
        }
    }
    Ok(())
}

/// Set core for `verify_held_out_disjoint` (`baseline_metrics.py:333-343`
/// over the `hydra_search::eval::partition` split-membership designated
/// owner): disjointness decides — the oracle's trailing size-consistency arm
/// compares the same two sets, so it is dead once disjoint and needs no
/// twin. Overlap renders sorted, first five, in the oracle `{sorted!r}`
/// shape via the single-quote `py_key_list` below.
fn held_out_disjoint_core(train_ids: &[String], held_ids: &[String]) -> Result<(), String> {
    let train: HashSet<&str> = train_ids.iter().map(String::as_str).collect();
    let held: HashSet<&str> = held_ids.iter().map(String::as_str).collect();
    if train.is_disjoint(&held) {
        return Ok(());
    }
    let mut overlap: Vec<&str> = train.intersection(&held).copied().collect();
    overlap.sort_unstable();
    overlap.truncate(5);
    Err(format!(
        "held-out leakage: overlap {}",
        py_key_list(&overlap.iter().map(ToString::to_string).collect::<Vec<_>>())
    ))
}

// ---------------------------------------------------------------------------
// Value → Python fold (transport parse wrap).
// ---------------------------------------------------------------------------

/// Convert a parsed JSON value into live Python objects (attached; the walk
/// touches Python memory throughout — the `canon_rng.rs:263` attached-only
/// precedent). Numbers: `i64`/`u64` ride as `int`, `f64` as `float`
/// (absurd-huge ints already landed as `f64` in the detached parse — see the
/// module divergence note).
fn value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(flag) => Ok(PyBool::new(py, *flag).to_owned().into_any().unbind()),
        serde_json::Value::Number(number) => {
            if let Some(signed) = number.as_i64() {
                Ok(signed.into_pyobject(py)?.into_any().unbind())
            } else if let Some(unsigned) = number.as_u64() {
                Ok(unsigned.into_pyobject(py)?.into_any().unbind())
            } else if let Some(float) = number.as_f64() {
                Ok(PyFloat::new(py, float).into_any().unbind())
            } else {
                Err(PyValueError::new_err(
                    "bridge:record_decode_json_line: number out of range",
                ))
            }
        }
        serde_json::Value::String(text) => Ok(PyString::new(py, text).into_any().unbind()),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// Pyfns — eval submodule (dict projections over translator-staged scalars).
// ---------------------------------------------------------------------------

/// Dict projection (`case.py:80-88`): the translator unpacks the live
/// `EvalCase` Python-side (live objects never cross); this leaf only
/// assembles the six keys attached. Infallible over staged strings.
#[pyfunction]
#[pyo3(signature = (case_id, arms, primary_metric, uncertainty_unit, rules_hash, diagnostic_only))]
fn eval_case_to_json(
    py: Python<'_>,
    case_id: String,
    arms: Vec<String>,
    primary_metric: String,
    uncertainty_unit: String,
    rules_hash: String,
    diagnostic_only: bool,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("case_id", case_id)?;
    dict.set_item("arms", arms)?;
    dict.set_item("primary_metric", primary_metric)?;
    dict.set_item("uncertainty_unit", uncertainty_unit)?;
    dict.set_item("rules_hash", rules_hash)?;
    dict.set_item("diagnostic_only", diagnostic_only)?;
    Ok(dict.unbind())
}

/// Canonical JSON projection (`promotion.py:190-211`): the translator unpacks
/// the live `PromotionRecord`/`ExcludedBlock` objects Python-side (gates as an
/// unstaged dict, blocks as `(wall_id, reason, detail)` triples); the gate
/// sort runs detached, the assembly runs attached. Infallible over staged
/// strings (records are constructor-validated before projection).
#[pyfunction]
#[pyo3(signature = (candidate_spec_hash, utility_manifest_hash, comparator_spec_hashes, case_manifest_hash, result_table_hash, resource_view, uncertainty_unit, pass_inequality, observed_estimate, confidence_bounds, gates, disposition, schedule_hash, environment_hash, excluded_blocks))]
#[allow(clippy::too_many_arguments)]
fn record_to_json(
    py: Python<'_>,
    candidate_spec_hash: String,
    utility_manifest_hash: String,
    comparator_spec_hashes: Vec<String>,
    case_manifest_hash: String,
    result_table_hash: String,
    resource_view: String,
    uncertainty_unit: String,
    pass_inequality: String,
    observed_estimate: f64,
    confidence_bounds: (f64, f64),
    gates: Bound<'_, PyDict>,
    disposition: String,
    schedule_hash: Option<String>,
    environment_hash: Option<String>,
    excluded_blocks: Vec<(String, String, String)>,
) -> PyResult<Py<PyDict>> {
    let mut staged: Vec<(String, String)> = Vec::with_capacity(gates.len());
    for (key, value) in gates.iter() {
        let name: String = key.extract().map_err(|_| {
            PyValueError::new_err("bridge:record_to_json: gate names must be strings")
        })?;
        let gate: String = value.extract().map_err(|_| {
            PyValueError::new_err("bridge:record_to_json: gate values must be strings")
        })?;
        staged.push((name, gate));
    }
    let sorted = py.detach(|| {
        staged.sort_by(|left, right| left.0.cmp(&right.0));
        staged
    });
    let dict = PyDict::new(py);
    dict.set_item("candidate_spec_hash", candidate_spec_hash)?;
    dict.set_item("utility_manifest_hash", utility_manifest_hash)?;
    dict.set_item("comparator_spec_hashes", comparator_spec_hashes)?;
    dict.set_item("case_manifest_hash", case_manifest_hash)?;
    dict.set_item("result_table_hash", result_table_hash)?;
    dict.set_item("resource_view", resource_view)?;
    dict.set_item("uncertainty_unit", uncertainty_unit)?;
    dict.set_item("pass_inequality", pass_inequality)?;
    dict.set_item("observed_estimate", observed_estimate)?;
    dict.set_item(
        "confidence_bounds",
        vec![confidence_bounds.0, confidence_bounds.1],
    )?;
    let gates_dict = PyDict::new(py);
    for (name, gate) in &sorted {
        gates_dict.set_item(name, gate)?;
    }
    dict.set_item("gates", gates_dict)?;
    dict.set_item("disposition", disposition)?;
    dict.set_item("schedule_hash", schedule_hash)?;
    dict.set_item("environment_hash", environment_hash)?;
    let blocks = PyList::empty(py);
    for (wall_id, reason, detail) in &excluded_blocks {
        let entry = PyDict::new(py);
        entry.set_item("wall_id", wall_id)?;
        entry.set_item("reason", reason)?;
        entry.set_item("detail", detail)?;
        blocks.append(entry)?;
    }
    dict.set_item("excluded_blocks", blocks)?;
    Ok(dict.unbind())
}

/// Block-disjointness gate (`duplicate.py:229-244`): the translator unpacks
/// the live `WallBlock` objects Python-side to `(wall_id, game_ids)` pairs
/// (live dataclasses never cross — the `eval_leaves.rs:21-24`
/// caller-projects precedent); the set decision runs detached. The bridge
/// raises `ValueError` with the byte-identical oracle sentence; the
/// translator maps to `ContractError` (extraction `TypeError` on exotic
/// non-`str` ids falls back to the oracle body).
#[pyfunction]
#[pyo3(signature = (blocks,))]
fn record_validate_blocks_disjoint(
    py: Python<'_>,
    blocks: Vec<(String, Vec<String>)>,
) -> PyResult<()> {
    py.detach(|| blocks_disjoint_core(&blocks))
        .map_err(PyValueError::new_err)
}

/// Held-out disjointness gate (`baseline_metrics.py:333-343`): the translator
/// unpacks the live `HeldOutSplit` Python-side to plain id lists; the set
/// decision runs detached. The bridge raises `ValueError` with the
/// byte-identical oracle sentence; the translator maps to `ContractError`
/// (extraction `TypeError` falls back to the oracle body).
#[pyfunction]
#[pyo3(signature = (train_ids, held_out_ids))]
fn record_verify_held_out_disjoint(
    py: Python<'_>,
    train_ids: Vec<String>,
    held_out_ids: Vec<String>,
) -> PyResult<()> {
    py.detach(|| held_out_disjoint_core(&train_ids, &held_out_ids))
        .map_err(PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// Pyfns — contracts submodule (gates + transport + codec + consts).
// ---------------------------------------------------------------------------

/// Flag gate (`attestation.py:213-239`): the translator passes the live
/// `Attestation` (or `None`) plus the explicit flag; the field view stages
/// attached, the verdict runs detached, the input object returns on success.
/// The bridge raises `ValueError` with the byte-identical oracle sentence;
/// the translator maps to `ContractError`.
#[pyfunction]
#[pyo3(signature = (value, allow_synthetic=true))]
fn record_require_attestation(
    py: Python<'_>,
    value: Bound<'_, PyAny>,
    allow_synthetic: bool,
) -> PyResult<Py<PyAny>> {
    let missing = value.is_none();
    let (
        kind,
        attestation_id,
        confidential_source_id,
        purpose_len,
        purpose_present,
        disclosure_class,
        acquisition_is_dict,
        acquisition_len,
        acquisition_dataset,
    ) = if missing {
        (None, None, None, 0, false, None, false, 0, String::new())
    } else {
        let kind: Option<String> = value.getattr("kind").ok().and_then(|v| v.extract().ok());
        let attestation_id: Option<String> = value
            .getattr("attestation_id")
            .ok()
            .and_then(|v| v.extract().ok());
        let confidential_source_id: Option<String> = value
            .getattr("confidential_source_id")
            .ok()
            .and_then(|v| v.extract().ok());
        let (purpose_len, purpose_present) = value
            .getattr("permitted_purpose")
            .ok()
            .map(|v| {
                v.extract::<Vec<String>>()
                    .map(|items| (items.len(), true))
                    .unwrap_or((0, false))
            })
            .unwrap_or((0, false));
        let disclosure_class: Option<String> = value
            .getattr("disclosure_class")
            .ok()
            .and_then(|v| v.extract().ok());
        let (is_dict, len, dataset) = value
            .getattr("acquisition_metadata")
            .ok()
            .map(|v| {
                v.cast::<PyDict>()
                    .map(|dict| {
                        let dataset = dict
                            .get_item("dataset")
                            .ok()
                            .flatten()
                            .and_then(|item| {
                                if item.is_none() {
                                    None
                                } else {
                                    item.extract::<String>().ok().filter(|s| !s.is_empty())
                                }
                            })
                            .unwrap_or_default();
                        (true, dict.len(), dataset)
                    })
                    .unwrap_or((false, 0, String::new()))
            })
            .unwrap_or((false, 0, String::new()));
        (
            kind,
            attestation_id,
            confidential_source_id,
            purpose_len,
            purpose_present,
            disclosure_class,
            is_dict,
            len,
            dataset,
        )
    };
    let view = AttestationView {
        missing,
        kind,
        attestation_id,
        confidential_source_id,
        permitted_purpose_len: purpose_len,
        permitted_purpose_present: purpose_present,
        disclosure_class,
        acquisition_is_dict,
        acquisition_len,
        acquisition_dataset,
    };
    py.detach(|| attestation_core(&view, allow_synthetic))
        .map_err(PyValueError::new_err)?;
    Ok(value.unbind())
}

/// Transport-only JSON parse (`decode.py:46-48` via the `feed::decode` owner
/// parser family — same `serde_json` the owner parses with). Success values
/// are JSON-value-identical; error texts diverge (serde vs CPython) but both
/// raise `ValueError`. Compute (the parse) runs detached; staging + wrap run
/// attached.
#[pyfunction]
#[pyo3(signature = (line,))]
fn record_decode_json_line(py: Python<'_>, line: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let text: String = if let Ok(s) = line.cast::<PyString>() {
        s.to_str()?.to_owned()
    } else if let Ok(b) = line.cast::<PyBytes>() {
        String::from_utf8(b.as_bytes().to_vec()).map_err(|err| {
            PyValueError::new_err(format!("bridge:record_decode_json_line: {err}"))
        })?
    } else if let Ok(b) = line.cast::<PyByteArray>() {
        String::from_utf8(b.to_vec()).map_err(|err| {
            PyValueError::new_err(format!("bridge:record_decode_json_line: {err}"))
        })?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "the JSON object must be str, bytes or bytearray, not {}",
            py_type_name(&line)?
        )));
    };
    let staged = text;
    let parsed: serde_json::Value = py
        .detach(|| {
            serde_json::from_str::<serde_json::Value>(&staged)
                .map_err(|err| format!("bridge:record_decode_json_line: {err}"))
        })
        .map_err(PyValueError::new_err)?;
    value_to_py(py, &parsed)
}

/// Microbatch gate plus slice bounds (`stream_decode.py:398-421` without the
/// generator or the shallow slicing over live games, which stay Python). The
/// `PyBool`-exclusion + `PyInt` gate mirrors `obs_actor.rs:112`; the `{!r}`
/// slot carries the exact staged `repr`. The bridge raises `ValueError`, the
/// translator maps to `ContractError` with byte-identical text.
#[pyfunction]
#[pyo3(signature = (batch_len, microbatch_size))]
fn record_microbatch_bounds(
    py: Python<'_>,
    batch_len: usize,
    microbatch_size: Bound<'_, PyAny>,
) -> PyResult<Vec<(usize, usize)>> {
    let repr_text = py_repr(&microbatch_size)?;
    let staged: Option<i64> = if microbatch_size.is_instance_of::<PyBool>()
        || !microbatch_size.is_instance_of::<PyInt>()
    {
        None
    } else {
        microbatch_size.extract().ok()
    };
    py.detach(|| microbatch_bounds_core(batch_len, staged, &repr_text))
        .map_err(PyValueError::new_err)
}

/// Decision proxy for throughput math (`stream_decode.py:424-426`): total
/// events across staged per-game counts. The translator stages
/// `[len(game.game.events) for game in games]` Python-side (live objects
/// never cross); the int sum runs detached (float sums stay Python).
#[pyfunction]
#[pyo3(signature = (counts,))]
fn record_count_decisions(py: Python<'_>, counts: Vec<u64>) -> PyResult<u64> {
    py.detach(|| count_decisions_core(&counts))
        .map_err(PyValueError::new_err)
}

/// JSON-safe snapshot of a shuffle `random.Random`
/// (`stream_manifest.py:424-437`): the translator passes the plain
/// `getstate()` triple (`version`, `internal` tuple, `gauss_next`); the live
/// `Random` never crosses (no RNG in Rust). Validation messages are
/// byte-identical to the oracle; the `{version, state, gauss_next}` assembly
/// runs attached.
#[pyfunction]
#[pyo3(signature = (version, internal, gauss_next))]
fn record_serialize_shuffle_rng(
    py: Python<'_>,
    version: Bound<'_, PyAny>,
    internal: Bound<'_, PyAny>,
    gauss_next: Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let version_ok = version.is_instance_of::<PyInt>() && !version.is_instance_of::<PyBool>();
    let version_value: i64 = version.extract().unwrap_or(0);
    let internal_is_tuple = internal.is_instance_of::<PyTuple>();
    let internal_values: Vec<u64> = internal
        .cast::<PyTuple>()
        .map(|tuple| {
            tuple
                .iter()
                .map(|item| {
                    if item.is_instance_of::<PyBool>() {
                        None
                    } else {
                        item.extract::<u64>().ok()
                    }
                })
                .collect::<Option<Vec<u64>>>()
        })
        .unwrap_or(None)
        .unwrap_or_default();
    let gauss_is_none = gauss_next.is_none();
    // Oracle `serialize` admits `None` or `float` only (int fails) — bool is
    // never a float, so a single `PyFloat` test plus the `None` arm is exact.
    let gauss_ok = gauss_next.is_instance_of::<PyFloat>();
    let gauss_value: Option<f64> = if gauss_is_none || gauss_next.is_instance_of::<PyBool>() {
        None
    } else {
        gauss_next.extract::<f64>().ok()
    };
    let version_ok_owned = version_ok;
    let gauss_ok_owned = gauss_ok || gauss_is_none;
    py.detach(|| serialize_shuffle_core(version_ok_owned, gauss_ok_owned, gauss_is_none))
        .map_err(PyValueError::new_err)?;
    if !internal_is_tuple {
        return Err(PyValueError::new_err("shuffle RNG state malformed"));
    }
    let dict = PyDict::new(py);
    dict.set_item("version", version_value)?;
    dict.set_item("state", internal_values)?;
    match gauss_value {
        Some(value) => dict.set_item("gauss_next", value)?,
        None => dict.set_item("gauss_next", py.None())?,
    }
    Ok(dict.unbind())
}

/// Inverse of `record_serialize_shuffle_rng`
/// (`stream_manifest.py:440-464`): non-mapping, unknown keys, non-int
/// version, empty/non-int state, negative words, and non-float `gauss_next`
/// all fail closed with byte-identical oracle sentences. Returns
/// `(version, state_list, gauss_next)` — the translator wraps the list as a
/// tuple (bridge `Vec` crosses as `list`, the oracle returns `tuple`).
#[pyfunction]
#[pyo3(signature = (raw,))]
fn record_parse_shuffle_rng(
    py: Python<'_>,
    raw: Bound<'_, PyAny>,
) -> PyResult<(i64, Vec<u64>, Option<f64>)> {
    let is_mapping = raw.is_instance_of::<PyDict>();
    let mut unknown_sorted: Vec<String> = Vec::new();
    let mut version_ok = false;
    let mut version_value: i64 = 0;
    let mut state_is_list = false;
    let mut state_values: Vec<u64> = Vec::new();
    let mut state_orig_len: usize = 0;
    let mut state_all_ok = false;
    let mut gauss_is_none = false;
    let mut gauss_is_bool = false;
    let mut gauss_is_number = false;
    let mut gauss_value: Option<f64> = None;
    if let Ok(dict) = raw.cast::<PyDict>() {
        for key in dict.keys() {
            let name: String = key.extract().unwrap_or_default();
            if name != "version" && name != "state" && name != "gauss_next" {
                unknown_sorted.push(name);
            }
        }
        unknown_sorted.sort();
        if let Some(item) = dict.get_item("version").ok().flatten() {
            version_ok = item.is_instance_of::<PyInt>() && !item.is_instance_of::<PyBool>();
            version_value = item.extract::<i64>().unwrap_or(0);
        }
        if let Some(item) = dict.get_item("state").ok().flatten()
            && let Ok(list) = item.cast::<PyList>()
        {
            state_is_list = true;
            state_orig_len = list.len();
            let mut values = Vec::with_capacity(state_orig_len);
            let mut all_ok = state_orig_len > 0;
            for entry in list.iter() {
                if entry.is_instance_of::<PyBool>() {
                    all_ok = false;
                    break;
                }
                match entry.extract::<u64>() {
                    Ok(number) => values.push(number),
                    Err(_) => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok {
                state_values = values;
            }
            state_all_ok = all_ok && state_orig_len > 0;
        }
        match dict.get_item("gauss_next").ok().flatten() {
            None => {
                gauss_is_none = true;
            }
            Some(item) if item.is_none() => {
                gauss_is_none = true;
            }
            Some(item) if item.is_instance_of::<PyBool>() => {
                gauss_is_bool = true;
            }
            Some(item) => {
                if let Ok(number) = item.extract::<f64>() {
                    gauss_is_number = true;
                    gauss_value = Some(number);
                }
            }
        }
    }
    let unknown_for_core = unknown_sorted.clone();
    py.detach(|| {
        parse_shuffle_core(
            is_mapping,
            &unknown_for_core,
            version_ok,
            state_is_list,
            state_orig_len,
            state_all_ok,
            gauss_is_none,
            gauss_is_bool,
            gauss_is_number,
        )
    })
    .map_err(PyValueError::new_err)?;
    Ok((version_value, state_values, gauss_value))
}

/// Vocab-width gate (`dataset_encode.py:57-68`): the frozen baseline passes,
/// anything else needs the explicit test-only narrow flag. The `{where}` and
/// `{num_actions}` slots carry the staged live `str()` (for ints equals
/// `repr`); the baseline is the census owner, never a literal.
#[pyfunction]
#[pyo3(signature = (num_actions, allow_narrow, where_str))]
fn record_require_action_width(
    py: Python<'_>,
    num_actions: Bound<'_, PyAny>,
    allow_narrow: bool,
    where_str: String,
) -> PyResult<()> {
    let num_str = py_str(&num_actions)?;
    let num_value: Option<i64> = if num_actions.is_instance_of::<PyBool>() {
        num_actions.extract().ok().map(|flag: bool| i64::from(flag))
    } else {
        num_actions.extract().ok()
    };
    let baseline = RECORD_BASELINE_ACTIONS;
    // proof: `baseline` = 6792 const and guard is equality-only; negative `value` never equals it, `as` vs `try_from` identical here.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let is_baseline = matches!(num_value, Some(value) if value as usize == baseline);
    py.detach(|| action_width_core(is_baseline, allow_narrow, &where_str, &num_str, baseline))
        .map_err(PyValueError::new_err)
}

/// PBRF code → owning exception-class name (`common.py:135-153` over the
/// `common_errors.rs`-owned classes). Translators resolve the name to the
/// bridge class (`getattr(contracts, name)`); unknown codes fail closed.
#[pyfunction]
#[pyo3(signature = (code,))]
fn record_pbrf_error_class(py: Python<'_>, code: String) -> PyResult<String> {
    py.detach(|| pbrf_class_core(&code))
        .map(|name| name.to_owned())
        .map_err(PyValueError::new_err)
}

/// Depth guard for the tuple→list walk below (realistic report docs nest
/// < 20; past this the oracle raises `RecursionError` — see the module
/// divergence note).
const JSONABLE_MAX_DEPTH: usize = 4096;

/// Tuple→list/dict recursion (`baseline_eval.py:241-262`), adjacent to the
/// `hydra_feed::canon` JCS owner which consumes only the list/dict domain
/// (RFC 8785 arrays). Attached throughout — the walk touches Python memory
/// at every level (the `canon_rng.rs:263` attached-only precedent).
/// `tuple`/`list` (incl. subclasses, same as `isinstance`) rebuild as
/// `list` with recursed elements; `dict` (incl. subclasses) rebuilds with
/// the same keys and recursed values (keys pass through unstaged exactly
/// like the oracle); every other object returns as-is.
fn jsonable_walk<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if depth > JSONABLE_MAX_DEPTH {
        return Err(PyValueError::new_err(
            "record_to_jsonable: nesting exceeds 4096 levels",
        ));
    }
    if value.is_instance_of::<PyTuple>() {
        let tuple = value.cast::<PyTuple>()?;
        let out = PyList::empty(py);
        for item in tuple.iter() {
            out.append(jsonable_walk(py, &item, depth + 1)?)?;
        }
        return Ok(out.into_any());
    }
    if value.is_instance_of::<PyList>() {
        let list = value.cast::<PyList>()?;
        let out = PyList::empty(py);
        for item in list.iter() {
            out.append(jsonable_walk(py, &item, depth + 1)?)?;
        }
        return Ok(out.into_any());
    }
    if value.is_instance_of::<PyDict>() {
        let dict = value.cast::<PyDict>()?;
        let out = PyDict::new(py);
        for (key, val) in dict.iter() {
            out.set_item(key, jsonable_walk(py, &val, depth + 1)?)?;
        }
        return Ok(out.into_any());
    }
    Ok(value.clone())
}

/// JSON-safe normalisation (`baseline_eval.py:241-262`): tuples become lists
/// recursively so the canonical report digest sees arrays only. Infallible
/// over live values (the translator keeps the verbatim oracle fallback for
/// a stale `.so`).
#[pyfunction]
#[pyo3(signature = (value,))]
fn record_to_jsonable(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(jsonable_walk(py, &value, 0)?.unbind())
}

// ---------------------------------------------------------------------------
// Registration.
// ---------------------------------------------------------------------------

/// Register the record leaves + frozen tables on the shared `contracts`
/// submodule (mirrors `common_roots::register`): consts via `sub.add`, fns via
/// `wrap_pyfunction!(f, sub)`; single cdylib, no new entry point. MAIN calls
/// this from `contracts::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "RECORD_PARTITION_ORDER",
        PyTuple::new(py, RECORD_PARTITION_ORDER)?,
    )?;
    sub.add("RECORD_WALL_TILE_COUNT", RECORD_WALL_TILE_COUNT)?;
    sub.add("RECORD_FIXED_HISTORY_T", RECORD_FIXED_HISTORY_T)?;
    sub.add("RECORD_HISTORY_LEN_PLANE", RECORD_HISTORY_LEN_PLANE)?;
    sub.add("RECORD_DECISION_IDS_FILENAME", RECORD_DECISION_IDS_FILENAME)?;
    sub.add(
        "RECORD_OBSERVATION_HASHES_FILENAME",
        RECORD_OBSERVATION_HASHES_FILENAME,
    )?;
    sub.add("RECORD_BASELINE_ACTIONS", RECORD_BASELINE_ACTIONS)?;
    sub.add("RECORD_PBRF_CODES", PyTuple::new(py, RECORD_PBRF_CODES)?)?;
    sub.add_function(wrap_pyfunction!(record_require_attestation, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_decode_json_line, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_microbatch_bounds, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_count_decisions, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_serialize_shuffle_rng, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_parse_shuffle_rng, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_require_action_width, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_pbrf_error_class, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_to_jsonable, sub)?)?;
    Ok(())
}

/// Register the eval projections on the EXISTING `eval` submodule (mirrors
/// `eval_blocks::register`): pure dict assembly over translator-staged
/// scalars; single cdylib, no new entry point. MAIN calls this from
/// `eval::register` — no new submodule.
pub fn register_eval(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(eval_case_to_json, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_to_json, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_validate_blocks_disjoint, sub)?)?;
    sub.add_function(wrap_pyfunction!(record_verify_held_out_disjoint, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_track_owners() {
        assert_eq!(
            RECORD_PARTITION_ORDER,
            hydra_feed::manifest::PARTITION_ORDER,
            "partition order must track the feed manifest owner"
        );
        assert_eq!(
            RECORD_PARTITION_ORDER,
            hydra_feed::partition::PARTITION_ORDER,
            "manifest and partition owners must agree"
        );
        assert_eq!(RECORD_WALL_TILE_COUNT, 136);
        assert_eq!(RECORD_WALL_TILE_COUNT, hydra_feed::decode::WALL_LEN);
        assert_eq!(RECORD_WALL_TILE_COUNT, hydra_feed::validate::WALL_LEN);
        assert_eq!(hydra_feed::tiles::TILE_REV.len(), 136);
        assert_eq!(RECORD_FIXED_HISTORY_T, 256);
        assert_eq!(
            RECORD_FIXED_HISTORY_T,
            hydra_shard::writer::SHARD_MAX_HISTORY
        );
        assert_eq!(RECORD_HISTORY_LEN_PLANE, "history_len");
        assert_eq!(RECORD_DECISION_IDS_FILENAME, "decision_ids.json");
        assert_eq!(
            RECORD_OBSERVATION_HASHES_FILENAME,
            "observation_hashes.json"
        );
        assert_eq!(RECORD_BASELINE_ACTIONS, 6792);
        assert_eq!(RECORD_BASELINE_ACTIONS, hydra_feed::census::CENSUS_TOTAL);
        // proof: `RECORD_BASELINE_ACTIONS` = 6792 const, fits `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let baseline_u: u32 = RECORD_BASELINE_ACTIONS as u32;
        assert_eq!(baseline_u, hydra_feed::ledger::ACTION_TABLE_LEN);
        assert_eq!(RECORD_PBRF_CODES.len(), 17);
        assert_eq!(RECORD_PBRF_TABLE.len(), 17);
        for (code, _class) in RECORD_PBRF_TABLE {
            assert!(RECORD_PBRF_CODES.contains(&code));
        }
        assert_eq!(
            pbrf_class_core("PBRF_PARTITION_EMPTY").unwrap(),
            "PacketPartitionError"
        );
        assert_eq!(
            pbrf_class_core("PBRF_STALE_EPOCH").unwrap(),
            "StaleBeliefError"
        );
        assert_eq!(
            pbrf_class_core("PBRF_DIGEST_DELTA").unwrap(),
            "DigestMismatchError"
        );
        assert_eq!(
            pbrf_class_core("PBRF_VIS_TREE_KEY").unwrap(),
            "VisibilityViolationError"
        );
        assert_eq!(
            pbrf_class_core("PBRF_SUPPORT_POINT").unwrap(),
            "ProposalSupportError"
        );
        assert!(pbrf_class_core("PBRF_UNKNOWN").is_err());
    }

    #[test]
    fn attestation_gate_matches_oracle_sentences() {
        let missing = AttestationView {
            missing: true,
            kind: None,
            attestation_id: None,
            confidential_source_id: None,
            permitted_purpose_len: 0,
            permitted_purpose_present: false,
            disclosure_class: None,
            acquisition_is_dict: false,
            acquisition_len: 0,
            acquisition_dataset: String::new(),
        };
        assert_eq!(
            attestation_core(&missing, true).expect_err("missing must reject"),
            "attestation missing; cannot create authorized RawObjectRow (D-017 pending)"
        );
        let synthetic = AttestationView {
            missing: false,
            kind: Some(String::from("synthetic")),
            attestation_id: Some(String::from("synthetic-attestation-v1")),
            confidential_source_id: Some(String::from("synthetic-source-v1")),
            permitted_purpose_len: 2,
            permitted_purpose_present: true,
            disclosure_class: Some(String::from("synthetic")),
            acquisition_is_dict: true,
            acquisition_len: 4,
            acquisition_dataset: String::new(),
        };
        assert!(attestation_core(&synthetic, true).is_ok());
        assert_eq!(
            attestation_core(&synthetic, false).expect_err("narrow must reject"),
            "synthetic attestation not permitted for this path"
        );
        let real = AttestationView {
            missing: false,
            kind: Some(String::from("real")),
            attestation_id: Some(String::from("D-017")),
            confidential_source_id: Some(String::from("src")),
            permitted_purpose_len: 1,
            permitted_purpose_present: true,
            disclosure_class: Some(String::from("confidential")),
            acquisition_is_dict: true,
            acquisition_len: 1,
            acquisition_dataset: String::from("tenhou-houou-corpus"),
        };
        assert!(attestation_core(&real, true).is_ok());
        let bad_dataset = AttestationView {
            acquisition_dataset: String::from("other-corpus"),
            ..real.clone()
        };
        assert_eq!(
            attestation_core(&bad_dataset, true).expect_err("dataset must reject"),
            "real D-017 dataset must reference tenhou-houou, got \"other-corpus\""
        );
    }

    #[test]
    fn microbatch_bounds_match_oracle() {
        assert_eq!(
            microbatch_bounds_core(5, Some(2), "2").unwrap(),
            vec![(0, 2), (2, 4), (4, 5)]
        );
        assert_eq!(
            microbatch_bounds_core(0, Some(2), "2").unwrap(),
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(
            microbatch_bounds_core(3, None, "None").expect_err("must reject"),
            "microbatch_size must be a positive int, got None"
        );
        assert_eq!(
            microbatch_bounds_core(3, Some(0), "0").expect_err("must reject"),
            "microbatch_size must be a positive int, got 0"
        );
        assert!(count_decisions_core(&[3, 0, 7]).unwrap() == 10);
        assert!(count_decisions_core(&[]).unwrap() == 0);
    }

    #[test]
    fn shuffle_codec_gates_match_oracle() {
        assert!(serialize_shuffle_core(true, true, true).is_ok());
        assert_eq!(
            serialize_shuffle_core(false, true, true).expect_err("version must reject"),
            "shuffle RNG state malformed"
        );
        assert!(parse_shuffle_core(true, &[], true, true, 3, true, true, false, false).is_ok());
        assert_eq!(
            parse_shuffle_core(false, &[], true, true, 3, true, true, false, false)
                .expect_err("mapping must reject"),
            "shuffle buffer_rng_state must be a mapping"
        );
        assert_eq!(
            parse_shuffle_core(
                true,
                &[String::from("extra")],
                true,
                true,
                3,
                true,
                true,
                false,
                false
            )
            .expect_err("unknown must reject"),
            "shuffle buffer_rng_state unknown keys ['extra']"
        );
    }

    #[test]
    fn action_width_gate_matches_oracle() {
        assert!(action_width_core(true, false, "tensorize_actor_row", "6792", 6792).is_ok());
        assert!(action_width_core(false, true, "tensorize_actor_row", "136", 6792).is_ok());
        assert_eq!(
            action_width_core(false, false, "tensorize_actor_row", "136", 6792)
                .expect_err("must reject"),
            "tensorize_actor_row: num_actions 136 != baseline 6792 requires allow_narrow=True (test-only narrow vocab)"
        );
    }

    #[test]
    fn blocks_disjoint_guards_match_oracle() {
        let ok = vec![
            ("w-a".to_string(), vec!["g1".to_string(), "g2".to_string()]),
            ("w-b".to_string(), vec!["g3".to_string()]),
        ];
        assert!(blocks_disjoint_core(&ok).is_ok());
        let mut dup_wall = ok.clone();
        dup_wall[1].0 = "w-a".to_string();
        assert_eq!(
            blocks_disjoint_core(&dup_wall).unwrap_err(),
            "duplicate wall_id 'w-a' across blocks"
        );
        let mut dup_game = ok.clone();
        dup_game[1].1.push("g2".to_string());
        assert_eq!(
            blocks_disjoint_core(&dup_game).unwrap_err(),
            "game 'g2' appears in multiple blocks"
        );
    }

    #[test]
    fn held_out_disjoint_guards_match_oracle() {
        assert!(held_out_disjoint_core(&["a".to_string()], &["b".to_string()]).is_ok());
        assert_eq!(
            held_out_disjoint_core(
                &["a".to_string(), "b".to_string()],
                &["b".to_string(), "c".to_string()]
            )
            .unwrap_err(),
            "held-out leakage: overlap ['b']"
        );
        let ids: Vec<String> = (0..6).map(|i| format!("g{i}")).collect();
        assert_eq!(
            held_out_disjoint_core(&ids, &ids).unwrap_err(),
            "held-out leakage: overlap ['g0', 'g1', 'g2', 'g3', 'g4']"
        );
    }
}
