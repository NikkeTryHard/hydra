//! rc_resume_gates: checkpoint-sidecar section gates on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists for the envelope — the nested RNG-state CONTENT codec
//! (`ShuffleRngState` in `crates/feed/src/manifest.rs:597-677`) owns the
//! `version`/`state`/`gauss_next` shape *inside* `buffer_rng_state`, and the
//! live window hash lives in `crates/shard/src/stream_driver.rs:8`; `rg` for
//! `buffer_keys|micro_in_update|worker_plan|sidecar` over `crates/feed/src` +
//! `crates/search/src` + `crates/shard/src` hits only those unrelated owners
//! plus doc comments, never the checkpoint-sidecar envelope gates). FORBIDS:
//! checkpoint-dir globs, sidecar file reads, `run.yaml` loading,
//! `ResumePlan`/`RunConfig` construction, and plan formatting (all stay Python
//! in `python/hydra2/training/_rc_resume.py`), wall-clock/RNG, and
//! `ContractError` shaping (the bridge raises `ValueError`; thin Python
//! translators map to `ContractError` with byte-identical messages).
//!
//! TABLE ported here — the four sidecar section gates, oracle
//! `python/hydra2/training/_rc_resume.py:84-189` (message bytes captured live
//! from the oracle pre-edit; gate order inside each check mirrors the oracle
//! statement order so the FIRST error on multi-fault inputs is identical):
//! - `rc_resume_shuffle` ← `_parse_shuffle` (:84-140; unknown keys,
//!   buffer_keys str-list, epoch/buffer non-negative ints, buffer overflow,
//!   buffer_rng mapping, prefix_hashes file/count/sha256 triple; the tolerated
//!   `buffer_entries` key (:87-94) stays allowed-but-ignored exactly like the
//!   oracle, which never reads it)
//! - `rc_resume_accum` ← `_parse_accum` (:143-154; yielded_batches,
//!   micro_in_update)
//! - `rc_resume_worker_plan` ← `_parse_worker_plan` (:157-172; num_workers,
//!   world_size, rank window against world, big-int exact via decimal reprs)
//! - `rc_resume_manifests` ← `_parse_sidecar_manifests` (:175-189; optional
//!   `sha256:<64 hex>` pins, null when absent)
//!
//! TABLED (stays Python, not ported): `format_plan` (`_rc_resume.py:353-422`)
//! is IO/format orchestration, not gates — it resolves `run_dir_for` (OS
//! layout authority, stays Python per `_rc_digest.py`), `run_config_digest`
//! (already bridged; hashes run through the owner), and `effective_run_id`,
//! then interpolates ~20 dataclass fields into f-strings. It never raises,
//! owns no frozen table, and its only pure fragment (the
//! `eval.microbatch_size or loop.microbatch_size` fallback) is a one-liner
//! Python already evaluates for free; a bridge round-trip would add cost for
//! zero win. `_read_sidecar` (file IO), `_iter_structural_candidates` /
//! `find_latest_checkpoint` (dir globs), and `resolve_resume_plan`
//! (orchestration over the bridged `rc_check_compat`) likewise stay Python,
//! as already recorded in `rc_resume.rs:27-29`.
//!
//! Shape per fn: attached staging (int views with bool exclusion per
//! `rc_parse_aux.rs:125`, `len()` emptiness without decoding per
//! `rc_parse_aux.rs:381-410`, live `startswith` for the prefix digest per the
//! `strip()` precedent at `rc_parse_aux.rs:402`, per-slot `get_item` reads per
//! `rc_parse_aux.rs:309-317`) → ONE `py.detach(|| …)` over owned plain data
//! with zero Python API inside (per `contracts.rs:434-437`) → attached wrap
//! as `PyValueError` (per `contracts.rs:117-123`) into a resolved `PyDict`
//! the translator converts into the frozen dataclass (`tuple(...)` /
//! `dict(...)`, exactly the oracle construction expressions). Bool is
//! excluded before every int check (per `contracts.rs:69-74`); missing slots
//! read as the oracle defaults exactly like the oracle's
//! `raw.get(key, default)` (per `packet_decode.rs:375-378`); result dicts are
//! built with `PyDict::new` plus `set_item` (per `contracts.rs:1227-1231`);
//! `PyTuple::new` for the key tuple (per `contracts.rs:1156`).
//! Unknown-key lists render from staged Python reprs sorted by value,
//! byte-exact for the JSON string-keyed domain (sidecar mappings always come
//! from `json.loads`, so keys are `str`; a non-str key would sort by its repr
//! deterministically instead of raising the oracle's `sorted()` TypeError —
//! unreachable through `_read_sidecar`); the digest-shape test restates
//! `rc_require.rs:189-197` (which restates `contracts.rs:99-107`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_require.rs:849-862`); no new entry
//! point. MAIN wiring: `pub mod rc_resume_gates;` in `lib.rs` plus
//! `crate::rc_resume_gates::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_resume::register(&sub)?;` (`contracts.rs:1299`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyModule, PyString, PyTuple};

/// Allowed sidecar section keys (oracle `_rc_resume.py:87-94,146,160,178`).
/// `buffer_entries` is allowed-but-ignored: the oracle tolerates the legacy
/// key yet never reads it, so the wrap never emits it either.
const SHUFFLE_ALLOWED: [&str; 6] = [
    "buffer_keys",
    "epoch_seed",
    "buffer_size",
    "buffer_rng_state",
    "buffer_entries",
    "prefix_hashes",
];
const ACCUM_ALLOWED: [&str; 2] = ["yielded_batches", "micro_in_update"];
const WORKER_ALLOWED: [&str; 3] = ["num_workers", "world_size", "rank"];
const MANIFEST_ALLOWED: [&str; 2] = ["stream_manifest_hash", "dataset_manifest_hash"];
const PREFIX_ALLOWED: [&str; 3] = ["file", "count", "sha256"];

// ---------------------------------------------------------------------------
// Attached staging helpers (all GIL-held; mirrors `rc_parse_aux.rs:102-112`
// through `rc_parse_aux.rs:322-334`).
// ---------------------------------------------------------------------------

/// Attached staging helper: exact `repr(value)` text (mirrors
/// `rc_require.rs:63-65`). Only feeds `IntView.repr` here — every sidecar
/// message is static except the unknown-key lists, which keep per-key reprs.
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Owned plain-data view of an int-or-other Python value (mirrors
/// `rc_require.rs:79-84`: bool excluded first, big ints past `i64` keep
/// `is_int` with `as_i64` unset and sign from the staged repr).
struct IntView {
    repr: String,
    is_int: bool,
    as_i64: Option<i64>,
    negative: bool,
}

/// Stage one int-or-other slot (mirrors `rc_parse_aux.rs:125-133`).
fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<IntView> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(IntView {
            repr,
            is_int: false,
            as_i64: None,
            negative: false,
        });
    }
    let as_i64 = obj.extract::<i64>().ok();
    let negative = as_i64.map_or_else(|| repr.starts_with('-'), |v| v < 0);
    Ok(IntView {
        repr,
        is_int: true,
        as_i64,
        negative,
    })
}

/// Staged unknown-key envelope (mirrors `rc_parse_aux.rs:293-305`, but the
/// sidecar messages carry `section` + `sidecar`, never an allowlist suffix).
struct SidecarUnknown {
    keys: Vec<(String, Option<String>)>,
    allowed: Vec<String>,
    section: String,
    sidecar: String,
}

impl SidecarUnknown {
    fn empty(section: &str, sidecar: &str, allowed: &[&str]) -> Self {
        SidecarUnknown {
            keys: Vec::new(),
            allowed: allowed.iter().map(|s| s.to_string()).collect(),
            section: section.to_owned(),
            sidecar: sidecar.to_owned(),
        }
    }
}

/// Extract the section mapping or fail with the envelope's `must be a
/// mapping` text (mirrors `rc_parse_aux.rs:278-291`; the sidecar messages
/// name the section and sidecar, never the offending type).
fn stage_sidecar_mapping<'py>(
    raw: &Bound<'py, PyAny>,
    section: &str,
    sidecar: &str,
) -> PyResult<Bound<'py, PyDict>> {
    match raw.extract() {
        Ok(dict) => Ok(dict),
        Err(_) => Err(PyValueError::new_err(format!(
            "checkpoint sidecar {section} must be a mapping: {sidecar}"
        ))),
    }
}

/// Stage the unknown-key envelope view (mirrors `rc_parse_aux.rs:293-305`).
fn stage_sidecar_unknown(
    dict: &Bound<'_, PyDict>,
    allowed: &[&str],
    section: &str,
    sidecar: &str,
) -> PyResult<SidecarUnknown> {
    let mut keys = Vec::with_capacity(dict.len());
    for (head, _value) in dict.iter() {
        let head_repr = py_repr(&head)?;
        keys.push((head_repr, head.extract::<String>().ok()));
    }
    Ok(SidecarUnknown {
        keys,
        allowed: allowed.iter().map(|s| s.to_string()).collect(),
        section: section.to_owned(),
        sidecar: sidecar.to_owned(),
    })
}

/// Read one optional slot: missing (`get_item` → `None`, mirroring the
/// oracle's `raw.get`) vs present (mirrors `rc_parse_aux.rs:309-317`).
fn slot_of<'a>(
    dict: &Bound<'a, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<Option<Bound<'a, PyAny>>> {
    dict.get_item(key).map_err(|e| {
        PyValueError::new_err(format!("contracts {owner} field {key:?} unreadable: {e}"))
    })
}

/// Stage one optional int slot: `None` when the oracle default applies,
/// otherwise the int view plus the original handle (big ints pass through
/// exactly like the oracle's plain assignment; mirrors
/// `rc_parse_aux.rs:322-334`).
fn stage_opt_int(
    dict: &Bound<'_, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<(Option<IntView>, Option<Py<PyAny>>)> {
    match slot_of(dict, key, owner)? {
        None => Ok((None, None)),
        Some(obj) => {
            let view = stage_int(&obj)?;
            Ok((Some(view), Some(obj.unbind())))
        }
    }
}

/// Shape bit plus copy handle for one optional opaque-dict slot (`None` when
/// the oracle `{}` default applies; `false` when present-but-not-a-dict).
/// The wrap branches only on presence, never on validity — validity gates
/// solely in the detached check, so no dead arm exists past a passed check.
fn stage_dict_slot(
    dict: &Bound<'_, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<(bool, Option<Py<PyDict>>)> {
    match slot_of(dict, key, owner)? {
        None => Ok((true, None)),
        Some(obj) => match obj.cast::<PyDict>() {
            Ok(inner) => Ok((true, Some(inner.clone().unbind()))),
            Err(_) => Ok((false, None)),
        },
    }
}

/// Plain-data state of the `shuffle.buffer_keys` slot: the oracle accepts
/// exactly a `list` of non-empty `str` (tuples fail — mirrors
/// `_rc_resume.py:98-100`). Element emptiness uses `len()`, so
/// lone-surrogate strings never decode (per `rc_parse_aux.rs:381-410`); the
/// original handles rebuild the tuple verbatim.
struct KeysCheck {
    ok: bool,
    len: usize,
}

/// Stage `shuffle.buffer_keys`, keeping the original element handles so the
/// wrap rebuilds the oracle `tuple(keys)` exactly (mirrors the
/// handle-keeping shape of `rc_parse_aux.rs:482-540`).
fn stage_keys(dict: &Bound<'_, PyDict>, owner: &str) -> PyResult<(KeysCheck, Vec<Py<PyAny>>)> {
    let unreadable = |e: pyo3::PyErr| {
        PyValueError::new_err(format!(
            "contracts {owner} field \"buffer_keys\" unreadable: {e}"
        ))
    };
    let slot = slot_of(dict, "buffer_keys", owner)?;
    let Some(obj) = slot else {
        return Ok((KeysCheck { ok: true, len: 0 }, Vec::new()));
    };
    let Ok(list) = obj.cast::<PyList>() else {
        return Ok((KeysCheck { ok: false, len: 0 }, Vec::new()));
    };
    let mut ok = true;
    let mut handles = Vec::with_capacity(list.len());
    for item in list.iter() {
        let item_ok = if item.is_instance_of::<PyString>() {
            item.len().map_err(unreadable)? != 0
        } else {
            false
        };
        ok = ok && item_ok;
        handles.push(item.unbind());
    }
    let len = handles.len();
    Ok((KeysCheck { ok, len }, handles))
}

/// Plain-data state of the `shuffle.prefix_hashes` slot. The triple is
/// required (no defaults) exactly when the mapping is present and non-empty
/// (oracle `_rc_resume.py:115`); `count: None` means missing-or-unstaged and
/// always fails the detached gate.
struct PrefixCheck {
    present: bool,
    dict_ok: bool,
    nonempty: bool,
    unknown: SidecarUnknown,
    file_ok: bool,
    count: Option<IntView>,
    sha_ok: bool,
}

/// Stage `shuffle.prefix_hashes`: mapping bit plus the triple views and the
/// copy handle. File emptiness uses `len()` and the digest uses a live
/// `startswith` (the `strip()` precedent at `rc_parse_aux.rs:402`), so
/// lone-surrogate strings decide exactly like the oracle without decoding.
fn stage_prefix(
    dict: &Bound<'_, PyDict>,
    owner: &str,
    sidecar: &str,
) -> PyResult<(PrefixCheck, Option<Py<PyDict>>)> {
    let unreadable = |e: pyo3::PyErr| {
        PyValueError::new_err(format!(
            "contracts {owner} field \"prefix_hashes\" unreadable: {e}"
        ))
    };
    let slot = slot_of(dict, "prefix_hashes", owner)?;
    let Some(obj) = slot else {
        return Ok((
            PrefixCheck {
                present: false,
                dict_ok: false,
                nonempty: false,
                unknown: SidecarUnknown::empty("shuffle.prefix_hashes", sidecar, &PREFIX_ALLOWED),
                file_ok: false,
                count: None,
                sha_ok: false,
            },
            None,
        ));
    };
    let Ok(pdict) = obj.cast::<PyDict>() else {
        return Ok((
            PrefixCheck {
                present: true,
                dict_ok: false,
                nonempty: false,
                unknown: SidecarUnknown::empty("shuffle.prefix_hashes", sidecar, &PREFIX_ALLOWED),
                file_ok: false,
                count: None,
                sha_ok: false,
            },
            None,
        ));
    };
    let handle = pdict.clone().unbind();
    let unknown = stage_sidecar_unknown(pdict, &PREFIX_ALLOWED, "shuffle.prefix_hashes", sidecar)?;
    let nonempty = pdict.len() != 0;
    let file_ok = match slot_of(pdict, "file", owner)? {
        Some(text) if text.is_instance_of::<PyString>() => text.len().map_err(unreadable)? != 0,
        _ => false,
    };
    let count = match slot_of(pdict, "count", owner)? {
        None => None,
        Some(value) => Some(stage_int(&value)?),
    };
    let sha_ok = match slot_of(pdict, "sha256", owner)? {
        Some(text) if text.is_instance_of::<PyString>() => text
            .call_method1("startswith", ("sha256:",))
            .map_err(unreadable)?
            .extract::<bool>()
            .map_err(unreadable)?,
        _ => false,
    };
    Ok((
        PrefixCheck {
            present: true,
            dict_ok: true,
            nonempty,
            unknown,
            file_ok,
            count,
            sha_ok,
        },
        Some(handle),
    ))
}

/// Stage one optional manifest pin: missing-or-null reads as `None` (valid);
/// otherwise the digest-shape bit plus the original handle. Extract failure
/// (non-str, or undecodable) stages `ok: false`, matching the oracle's
/// `not isinstance(str) or fullmatch is None` disjunction exactly — a valid
/// pin is pure ASCII, so no valid value can fail extraction.
fn stage_manifest_pin(
    dict: &Bound<'_, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<(bool, Option<Py<PyAny>>)> {
    match slot_of(dict, key, owner)? {
        None => Ok((true, None)),
        Some(obj) if obj.is_none() => Ok((true, None)),
        Some(obj) => {
            let ok = obj
                .extract::<String>()
                .ok()
                .is_some_and(|text| is_digest_shape(&text));
            if ok {
                Ok((true, Some(obj.unbind())))
            } else {
                Ok((false, None))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API). Gate order in
// each check mirrors the oracle statement order, so the FIRST error on
// multi-fault inputs is identical.
// ---------------------------------------------------------------------------

/// Render the unknown-key list from staged Python reprs sorted by value —
/// byte-exact for the JSON string-keyed domain (mirrors the
/// `[...]`-joining shape of `rc_require.rs:415-422`, without the config
/// `allowed=` suffix the sidecar messages never carry).
fn check_sidecar_unknown(view: &SidecarUnknown) -> Result<(), String> {
    let mut unknown: Vec<&(String, Option<String>)> = view
        .keys
        .iter()
        .filter(|(_, as_str)| as_str.as_ref().is_none_or(|s| !view.allowed.contains(s)))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_by(|a, b| {
        let key_a = a.1.as_deref().unwrap_or(a.0.as_str());
        let key_b = b.1.as_deref().unwrap_or(b.0.as_str());
        key_a.cmp(key_b)
    });
    let list = format!(
        "[{}]",
        unknown
            .iter()
            .map(|(repr, _)| repr.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Err(format!(
        "checkpoint sidecar {} unknown keys {list}: {}",
        view.section, view.sidecar
    ))
}

/// Detached non-negative-int gate with the sidecar static message (mirrors
/// `rc_parse_aux.rs:243-257`; big positive ints pass exactly like the
/// oracle's `< 0` comparison).
fn check_sidecar_nonneg(field: &str, view: &IntView, sidecar: &str) -> Result<(), String> {
    let ok = view.is_int
        && match view.as_i64 {
            Some(value) => value >= 0,
            None => !view.negative,
        };
    if ok {
        Ok(())
    } else {
        Err(format!("checkpoint sidecar {field} invalid: {sidecar}"))
    }
}

/// Digest-shape test (oracle `_rc_require.py:201`
/// `_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")` via `fullmatch`):
/// lowercase-only, restating `rc_require.rs:189-197` (which restates
/// `contracts.rs:99-107`, private there).
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

/// Compare two non-negative Python-int decimal reprs. Python int `repr` has
/// no leading zeros, so longer digit strings are bigger and equal-length
/// strings compare lexicographically — exact for arbitrarily large ints
/// with zero Python API.
fn decimal_lt(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return left.len() < right.len();
    }
    left < right
}

/// `0 <= rank < world` over staged views (oracle `_rc_resume.py:170`). The
/// caller validates `world` first, so a big `world` here is non-negative
/// (small ranks always fit; two big values compare by decimal repr).
fn rank_in_window(rank: &IntView, world: &Option<IntView>) -> bool {
    if !rank.is_int {
        return false;
    }
    let rank_negative = match rank.as_i64 {
        Some(value) => value < 0,
        None => rank.negative,
    };
    if rank_negative {
        return false;
    }
    match (rank.as_i64, world) {
        (Some(value), None) => value < 1,
        (Some(value), Some(plan)) => match plan.as_i64 {
            Some(size) => value < size,
            None => !plan.negative,
        },
        (None, None) => false,
        (None, Some(plan)) => match plan.as_i64 {
            Some(_) => false,
            None => !plan.negative && decimal_lt(&rank.repr, &plan.repr),
        },
    }
}

/// Owned inputs for the shuffle gate.
struct ShuffleView {
    unknown: SidecarUnknown,
    keys: KeysCheck,
    epoch: Option<IntView>,
    buffer: Option<IntView>,
    rng_ok: bool,
    prefix: PrefixCheck,
    sidecar: String,
}

/// `_parse_shuffle` decisions (oracle `_rc_resume.py:84-140`).
fn validate_shuffle(view: &ShuffleView) -> Result<(), String> {
    check_sidecar_unknown(&view.unknown)?;
    if !view.keys.ok {
        return Err(format!(
            "checkpoint sidecar shuffle.buffer_keys must be str list: {}",
            view.sidecar
        ));
    }
    if let Some(seed) = &view.epoch {
        check_sidecar_nonneg("shuffle.epoch_seed", seed, &view.sidecar)?;
    }
    if let Some(size) = &view.buffer {
        check_sidecar_nonneg("shuffle.buffer_size", size, &view.sidecar)?;
    }
    let buffer_is_big = view
        .buffer
        .as_ref()
        .is_some_and(|size| size.as_i64.is_none());
    let buffer_small: i64 = view
        .buffer
        .as_ref()
        .and_then(|size| size.as_i64)
        .unwrap_or(0);
    // proof: key counts are small vec lens (< 2^63), widen exactly into `i64`.
    let keys_i: i64 = i64::try_from(view.keys.len).map_err(|_| {
        format!(
            "checkpoint sidecar shuffle keys length outside i64: {}",
            view.sidecar
        )
    })?;
    if !buffer_is_big && keys_i > buffer_small {
        return Err(format!(
            "checkpoint sidecar shuffle overflows buffer_size: {}",
            view.sidecar
        ));
    }
    if !view.rng_ok {
        return Err(format!(
            "checkpoint sidecar shuffle.buffer_rng_state invalid: {}",
            view.sidecar
        ));
    }
    if view.prefix.present {
        if !view.prefix.dict_ok {
            return Err(format!(
                "checkpoint sidecar shuffle.prefix_hashes invalid: {}",
                view.sidecar
            ));
        }
        if view.prefix.nonempty {
            check_sidecar_unknown(&view.prefix.unknown)?;
            if !view.prefix.file_ok {
                return Err(format!(
                    "checkpoint sidecar shuffle.prefix_hashes.file invalid: {}",
                    view.sidecar
                ));
            }
            match &view.prefix.count {
                Some(count) => {
                    check_sidecar_nonneg("shuffle.prefix_hashes.count", count, &view.sidecar)?;
                }
                None => {
                    return Err(format!(
                        "checkpoint sidecar shuffle.prefix_hashes.count invalid: {}",
                        view.sidecar
                    ));
                }
            }
            if !view.prefix.sha_ok {
                return Err(format!(
                    "checkpoint sidecar shuffle.prefix_hashes.sha256 invalid: {}",
                    view.sidecar
                ));
            }
        }
    }
    Ok(())
}

/// Owned inputs for the accum gate.
struct AccumView {
    unknown: SidecarUnknown,
    yielded: Option<IntView>,
    micro: Option<IntView>,
    sidecar: String,
}

/// `_parse_accum` decisions (oracle `_rc_resume.py:143-154`; yielded before
/// micro, the oracle loop order).
fn validate_accum(view: &AccumView) -> Result<(), String> {
    check_sidecar_unknown(&view.unknown)?;
    if let Some(yielded) = &view.yielded {
        check_sidecar_nonneg("accum.yielded_batches", yielded, &view.sidecar)?;
    }
    if let Some(micro) = &view.micro {
        check_sidecar_nonneg("accum.micro_in_update", micro, &view.sidecar)?;
    }
    Ok(())
}

/// Owned inputs for the worker-plan gate.
struct WorkerView {
    unknown: SidecarUnknown,
    workers: Option<IntView>,
    world: Option<IntView>,
    rank: Option<IntView>,
    sidecar: String,
}

/// `_parse_worker_plan` decisions (oracle `_rc_resume.py:157-172`). A missing
/// rank defaults to 0, which always sits inside the validated `world >= 1`
/// window.
fn validate_worker(view: &WorkerView) -> Result<(), String> {
    check_sidecar_unknown(&view.unknown)?;
    if let Some(workers) = &view.workers {
        check_sidecar_nonneg("worker_plan.num_workers", workers, &view.sidecar)?;
    }
    if let Some(world) = &view.world {
        let ok = world.is_int
            && match world.as_i64 {
                Some(size) => size >= 1,
                None => !world.negative,
            };
        if !ok {
            return Err(format!(
                "checkpoint sidecar worker_plan.world_size invalid: {}",
                view.sidecar
            ));
        }
    }
    let default_rank = IntView {
        repr: "0".to_owned(),
        is_int: true,
        as_i64: Some(0),
        negative: false,
    };
    let rank: &IntView = view.rank.as_ref().unwrap_or(&default_rank);
    if !rank_in_window(rank, &view.world) {
        return Err(format!(
            "checkpoint sidecar worker_plan.rank invalid: {}",
            view.sidecar
        ));
    }
    Ok(())
}

/// Owned inputs for the manifests gate.
struct ManifestView {
    unknown: SidecarUnknown,
    stream_ok: bool,
    dataset_ok: bool,
    sidecar: String,
}

/// `_parse_sidecar_manifests` decisions (oracle `_rc_resume.py:175-189`;
/// stream pin before dataset pin, the oracle loop order).
fn validate_manifests(view: &ManifestView) -> Result<(), String> {
    check_sidecar_unknown(&view.unknown)?;
    if !view.stream_ok {
        return Err(format!(
            "checkpoint sidecar manifests.stream_manifest_hash invalid: {}",
            view.sidecar
        ));
    }
    if !view.dataset_ok {
        return Err(format!(
            "checkpoint sidecar manifests.dataset_manifest_hash invalid: {}",
            view.sidecar
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attached wrap helpers (dict assembly only; run after the detach succeeds).
// ---------------------------------------------------------------------------

/// Emit one optional int slot: the validated original, or the oracle default
/// (mirrors `rc_parse_aux.rs:778-789`).
fn emit_opt_int(
    out: &Bound<'_, PyDict>,
    key: &str,
    handle: Option<Py<PyAny>>,
    default: i64,
) -> PyResult<()> {
    match handle {
        Some(value) => out.set_item(key, value)?,
        None => out.set_item(key, default)?,
    }
    Ok(())
}

/// Emit one optional opaque-dict slot: a shallow copy of the validated
/// original (the oracle's `dict(...)`), or a fresh empty mapping for the
/// oracle `{}` default. Copies entry-wise (per `contracts.rs:1227-1231`)
/// rather than aliasing the caller's mapping.
fn emit_dict_slot(out: &Bound<'_, PyDict>, key: &str, handle: Option<Py<PyDict>>) -> PyResult<()> {
    let fresh = PyDict::new(out.py());
    if let Some(source) = handle {
        for (head, value) in source.bind(out.py()).iter() {
            fresh.set_item(head, value)?;
        }
    }
    out.set_item(key, fresh)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `_parse_shuffle` gate (oracle `_rc_resume.py:84-140`): the translator
/// passes the raw shuffle mapping plus the sidecar display string
/// positionally; the resolved dict carries the dataclass fields
/// (`buffer_keys` as a tuple, both ints, both mapping copies).
#[pyfunction]
fn rc_resume_shuffle(
    py: Python<'_>,
    raw: Bound<'_, PyAny>,
    sidecar: String,
) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_resume_shuffle";
    let dict = stage_sidecar_mapping(&raw, "shuffle", &sidecar)?;
    let unknown = stage_sidecar_unknown(&dict, &SHUFFLE_ALLOWED, "shuffle", &sidecar)?;
    let (keys, key_handles) = stage_keys(&dict, OWNER)?;
    let (epoch_view, epoch_handle) = stage_opt_int(&dict, "epoch_seed", OWNER)?;
    let (buffer_view, buffer_handle) = stage_opt_int(&dict, "buffer_size", OWNER)?;
    let (rng_ok, rng_handle) = stage_dict_slot(&dict, "buffer_rng_state", OWNER)?;
    let (prefix, prefix_handle) = stage_prefix(&dict, OWNER, &sidecar)?;
    let view = ShuffleView {
        unknown,
        keys,
        epoch: epoch_view,
        buffer: buffer_view,
        rng_ok,
        prefix,
        sidecar,
    };
    py.detach(|| validate_shuffle(&view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("buffer_keys", PyTuple::new(py, key_handles)?)?;
    emit_opt_int(&out, "epoch_seed", epoch_handle, 0)?;
    emit_opt_int(&out, "buffer_size", buffer_handle, 0)?;
    emit_dict_slot(&out, "buffer_rng_state", rng_handle)?;
    emit_dict_slot(&out, "prefix_hashes", prefix_handle)?;
    Ok(out.unbind())
}

/// `_parse_accum` gate (oracle `_rc_resume.py:143-154`).
#[pyfunction]
fn rc_resume_accum(py: Python<'_>, raw: Bound<'_, PyAny>, sidecar: String) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_resume_accum";
    let dict = stage_sidecar_mapping(&raw, "accum", &sidecar)?;
    let unknown = stage_sidecar_unknown(&dict, &ACCUM_ALLOWED, "accum", &sidecar)?;
    let (yielded_view, yielded_handle) = stage_opt_int(&dict, "yielded_batches", OWNER)?;
    let (micro_view, micro_handle) = stage_opt_int(&dict, "micro_in_update", OWNER)?;
    let view = AccumView {
        unknown,
        yielded: yielded_view,
        micro: micro_view,
        sidecar,
    };
    py.detach(|| validate_accum(&view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_int(&out, "yielded_batches", yielded_handle, 0)?;
    emit_opt_int(&out, "micro_in_update", micro_handle, 0)?;
    Ok(out.unbind())
}

/// `_parse_worker_plan` gate (oracle `_rc_resume.py:157-172`).
#[pyfunction]
fn rc_resume_worker_plan(
    py: Python<'_>,
    raw: Bound<'_, PyAny>,
    sidecar: String,
) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_resume_worker_plan";
    let dict = stage_sidecar_mapping(&raw, "worker_plan", &sidecar)?;
    let unknown = stage_sidecar_unknown(&dict, &WORKER_ALLOWED, "worker_plan", &sidecar)?;
    let (workers_view, workers_handle) = stage_opt_int(&dict, "num_workers", OWNER)?;
    let (world_view, world_handle) = stage_opt_int(&dict, "world_size", OWNER)?;
    let (rank_view, rank_handle) = stage_opt_int(&dict, "rank", OWNER)?;
    let view = WorkerView {
        unknown,
        workers: workers_view,
        world: world_view,
        rank: rank_view,
        sidecar,
    };
    py.detach(|| validate_worker(&view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_int(&out, "num_workers", workers_handle, 0)?;
    emit_opt_int(&out, "world_size", world_handle, 1)?;
    emit_opt_int(&out, "rank", rank_handle, 0)?;
    Ok(out.unbind())
}

/// `_parse_sidecar_manifests` gate (oracle `_rc_resume.py:175-189`): the
/// resolved dict carries both pins (`str` or `None`, exactly the oracle
/// `out` mapping).
#[pyfunction]
fn rc_resume_manifests(
    py: Python<'_>,
    raw: Bound<'_, PyAny>,
    sidecar: String,
) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_resume_manifests";
    let dict = stage_sidecar_mapping(&raw, "manifests", &sidecar)?;
    let unknown = stage_sidecar_unknown(&dict, &MANIFEST_ALLOWED, "manifests", &sidecar)?;
    let (stream_ok, stream_handle) = stage_manifest_pin(&dict, "stream_manifest_hash", OWNER)?;
    let (dataset_ok, dataset_handle) = stage_manifest_pin(&dict, "dataset_manifest_hash", OWNER)?;
    let view = ManifestView {
        unknown,
        stream_ok,
        dataset_ok,
        sidecar,
    };
    py.detach(|| validate_manifests(&view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    match stream_handle {
        Some(pin) => out.set_item("stream_manifest_hash", pin)?,
        None => out.set_item("stream_manifest_hash", py.None())?,
    }
    match dataset_handle {
        Some(pin) => out.set_item("dataset_manifest_hash", pin)?,
        None => out.set_item("dataset_manifest_hash", py.None())?,
    }
    Ok(out.unbind())
}

/// Register the sidecar section gates on the shared `contracts` submodule
/// (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `rows_seal.rs:485`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_resume_shuffle, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_resume_accum, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_resume_worker_plan, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_resume_manifests, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDECAR: &str = "ckpt-000100.json";

    fn unknown(section: &str, allowed: &[&str], keys: &[&str]) -> SidecarUnknown {
        SidecarUnknown {
            keys: keys
                .iter()
                .map(|key| (format!("'{key}'"), Some((*key).to_owned())))
                .collect(),
            allowed: allowed.iter().map(|s| (*s).to_owned()).collect(),
            section: section.to_owned(),
            sidecar: SIDECAR.to_owned(),
        }
    }

    fn empty_prefix() -> PrefixCheck {
        PrefixCheck {
            present: false,
            dict_ok: false,
            nonempty: false,
            unknown: SidecarUnknown::empty("shuffle.prefix_hashes", SIDECAR, &PREFIX_ALLOWED),
            file_ok: false,
            count: None,
            sha_ok: false,
        }
    }

    fn shuffle_view() -> ShuffleView {
        ShuffleView {
            unknown: unknown("shuffle", &SHUFFLE_ALLOWED, &[]),
            keys: KeysCheck { ok: true, len: 0 },
            epoch: None,
            buffer: None,
            rng_ok: true,
            prefix: empty_prefix(),
            sidecar: SIDECAR.to_owned(),
        }
    }

    fn accum_view() -> AccumView {
        AccumView {
            unknown: unknown("accum", &ACCUM_ALLOWED, &[]),
            yielded: None,
            micro: None,
            sidecar: SIDECAR.to_owned(),
        }
    }

    fn worker_view() -> WorkerView {
        WorkerView {
            unknown: unknown("worker_plan", &WORKER_ALLOWED, &[]),
            workers: None,
            world: None,
            rank: None,
            sidecar: SIDECAR.to_owned(),
        }
    }

    fn manifest_view() -> ManifestView {
        ManifestView {
            unknown: unknown("manifests", &MANIFEST_ALLOWED, &[]),
            stream_ok: true,
            dataset_ok: true,
            sidecar: SIDECAR.to_owned(),
        }
    }

    fn int_view(repr: &str, is_int: bool, as_i64: Option<i64>, negative: bool) -> IntView {
        IntView {
            repr: repr.to_owned(),
            is_int,
            as_i64,
            negative,
        }
    }

    fn small(value: i64) -> IntView {
        int_view(&value.to_string(), true, Some(value), value < 0)
    }

    /// Big non-negative int past `i64` (oracle accepts it wherever `< 0`
    /// rejects only negatives; probed live as `epoch_seed=10**30`).
    fn big(digits: &str) -> IntView {
        int_view(digits, true, None, false)
    }

    // ------------------------------------------------------------------
    // Shuffle gates (oracle `_rc_resume.py:84-140`, bytes captured live).
    // ------------------------------------------------------------------

    #[test]
    fn shuffle_unknown_keys_exact() {
        let mut view = shuffle_view();
        view.unknown = unknown("shuffle", &SHUFFLE_ALLOWED, &["zzz"]);
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle unknown keys ['zzz']: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_unknown_keys_sorted() {
        let mut view = shuffle_view();
        view.unknown = unknown("shuffle", &SHUFFLE_ALLOWED, &["z", "a"]);
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle unknown keys ['a', 'z']: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_keys_must_be_str_list() {
        let mut view = shuffle_view();
        view.keys = KeysCheck { ok: false, len: 0 };
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.buffer_keys must be str list: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_epoch_bool_rejected() {
        // Oracle excludes bool before the int check, same gate message.
        let mut view = shuffle_view();
        view.epoch = Some(int_view("True", false, None, false));
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.epoch_seed invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_epoch_negative_rejected() {
        let mut view = shuffle_view();
        view.epoch = Some(small(-1));
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.epoch_seed invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_epoch_big_accepted() {
        // Probed live: `epoch_seed=10**30` passes the oracle through.
        let mut view = shuffle_view();
        view.epoch = Some(big("1000000000000000000000000000000"));
        assert_eq!(validate_shuffle(&view), Ok(()));
    }

    #[test]
    fn shuffle_buffer_negative_rejected() {
        let mut view = shuffle_view();
        view.buffer = Some(small(-5));
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.buffer_size invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_overflow_rejected() {
        let mut view = shuffle_view();
        view.keys = KeysCheck { ok: true, len: 3 };
        view.buffer = Some(small(2));
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle overflows buffer_size: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_overflow_boundary_accepted() {
        let mut view = shuffle_view();
        view.keys = KeysCheck { ok: true, len: 2 };
        view.buffer = Some(small(2));
        assert_eq!(validate_shuffle(&view), Ok(()));
    }

    #[test]
    fn shuffle_overflow_big_buffer_accepted() {
        // A huge buffer dwarfs any staged key count.
        let mut view = shuffle_view();
        view.keys = KeysCheck { ok: true, len: 2 };
        view.buffer = Some(big("1000000000000000000000000000000"));
        assert_eq!(validate_shuffle(&view), Ok(()));
    }

    #[test]
    fn shuffle_rng_must_be_mapping() {
        let mut view = shuffle_view();
        view.rng_ok = false;
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.buffer_rng_state invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_not_mapping_rejected() {
        let mut view = shuffle_view();
        view.prefix.present = true;
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_empty_accepted() {
        // Oracle `:115` guards the triple behind `len(...) > 0`.
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        assert_eq!(validate_shuffle(&view), Ok(()));
    }

    #[test]
    fn shuffle_prefix_unknown_keys_exact() {
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        view.prefix.nonempty = true;
        view.prefix.unknown = unknown("shuffle.prefix_hashes", &PREFIX_ALLOWED, &["w"]);
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes unknown keys ['w']: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_file_rejected() {
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        view.prefix.nonempty = true;
        view.prefix.unknown =
            SidecarUnknown::empty("shuffle.prefix_hashes", SIDECAR, &PREFIX_ALLOWED);
        view.prefix.count = Some(small(1));
        view.prefix.sha_ok = true;
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes.file invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_count_missing_rejected() {
        // Missing and invalid counts share the oracle gate text.
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        view.prefix.nonempty = true;
        view.prefix.file_ok = true;
        view.prefix.sha_ok = true;
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes.count invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_count_negative_rejected() {
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        view.prefix.nonempty = true;
        view.prefix.file_ok = true;
        view.prefix.count = Some(small(-1));
        view.prefix.sha_ok = true;
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes.count invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_prefix_sha_rejected() {
        let mut view = shuffle_view();
        view.prefix.present = true;
        view.prefix.dict_ok = true;
        view.prefix.nonempty = true;
        view.prefix.file_ok = true;
        view.prefix.count = Some(small(1));
        let err = validate_shuffle(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar shuffle.prefix_hashes.sha256 invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn shuffle_defaults_accepted() {
        assert_eq!(validate_shuffle(&shuffle_view()), Ok(()));
    }

    // ------------------------------------------------------------------
    // Accum gates (oracle `_rc_resume.py:143-154`, bytes captured live).
    // ------------------------------------------------------------------

    #[test]
    fn accum_unknown_keys_exact() {
        let mut view = accum_view();
        view.unknown = unknown("accum", &ACCUM_ALLOWED, &["zzz"]);
        let err = validate_accum(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar accum unknown keys ['zzz']: ckpt-000100.json"
        );
    }

    #[test]
    fn accum_yielded_negative_rejected() {
        let mut view = accum_view();
        view.yielded = Some(small(-1));
        let err = validate_accum(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar accum.yielded_batches invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn accum_micro_bool_rejected() {
        let mut view = accum_view();
        view.micro = Some(int_view("True", false, None, false));
        let err = validate_accum(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar accum.micro_in_update invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn accum_yielded_first_on_multi_fault() {
        // Oracle loop order: yielded_batches gates before micro_in_update.
        let mut view = accum_view();
        view.yielded = Some(small(-1));
        view.micro = Some(int_view("'x'", false, None, false));
        let err = validate_accum(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar accum.yielded_batches invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn accum_defaults_accepted() {
        assert_eq!(validate_accum(&accum_view()), Ok(()));
    }

    // ------------------------------------------------------------------
    // Worker-plan gates (oracle `_rc_resume.py:157-172`, bytes captured live).
    // ------------------------------------------------------------------

    #[test]
    fn worker_unknown_keys_exact() {
        let mut view = worker_view();
        view.unknown = unknown("worker_plan", &WORKER_ALLOWED, &["zzz"]);
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan unknown keys ['zzz']: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_num_workers_negative_rejected() {
        let mut view = worker_view();
        view.workers = Some(small(-1));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.num_workers invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_world_size_zero_rejected() {
        let mut view = worker_view();
        view.world = Some(small(0));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.world_size invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_world_size_bool_rejected() {
        let mut view = worker_view();
        view.world = Some(int_view("True", false, None, false));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.world_size invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_rank_defaults_accepted() {
        // Missing rank (0) inside the missing world (1): `0 <= 0 < 1`.
        assert_eq!(validate_worker(&worker_view()), Ok(()));
    }

    #[test]
    fn worker_rank_at_world_rejected() {
        // `rank == world` fails the strict upper bound (probed live: 2 of 2).
        let mut view = worker_view();
        view.world = Some(small(2));
        view.rank = Some(small(2));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.rank invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_rank_window_accepted() {
        let mut view = worker_view();
        view.workers = Some(small(2));
        view.world = Some(small(4));
        view.rank = Some(small(3));
        assert_eq!(validate_worker(&view), Ok(()));
    }

    #[test]
    fn worker_rank_small_in_big_world_accepted() {
        let mut view = worker_view();
        view.world = Some(big("1000000000000000000000000000000"));
        view.rank = Some(small(2));
        assert_eq!(validate_worker(&view), Ok(()));
    }

    #[test]
    fn worker_rank_big_in_small_world_rejected() {
        let mut view = worker_view();
        view.world = Some(small(2));
        view.rank = Some(big("1000000000000000000000000000000"));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.rank invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_rank_big_world_decimal_order() {
        // Two big values compare by decimal repr: 10**30 < 10**30 + 1.
        let mut less = worker_view();
        less.world = Some(big("1000000000000000000000000000001"));
        less.rank = Some(big("1000000000000000000000000000000"));
        assert_eq!(validate_worker(&less), Ok(()));
        let mut greater = worker_view();
        greater.world = Some(big("1000000000000000000000000000000"));
        greater.rank = Some(big("1000000000000000000000000000001"));
        let err = validate_worker(&greater).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.rank invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn worker_rank_negative_big_rejected() {
        let mut view = worker_view();
        view.rank = Some(int_view(
            "-1000000000000000000000000000000",
            true,
            None,
            true,
        ));
        let err = validate_worker(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar worker_plan.rank invalid: ckpt-000100.json"
        );
    }

    // ------------------------------------------------------------------
    // Manifest gates (oracle `_rc_resume.py:175-189`, bytes captured live).
    // ------------------------------------------------------------------

    #[test]
    fn manifests_unknown_keys_exact() {
        let mut view = manifest_view();
        view.unknown = unknown("manifests", &MANIFEST_ALLOWED, &["zzz"]);
        let err = validate_manifests(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar manifests unknown keys ['zzz']: ckpt-000100.json"
        );
    }

    #[test]
    fn manifests_stream_pin_rejected() {
        let mut view = manifest_view();
        view.stream_ok = false;
        let err = validate_manifests(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar manifests.stream_manifest_hash invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn manifests_stream_first_on_multi_fault() {
        // Oracle loop order: stream pin gates before dataset pin.
        let mut view = manifest_view();
        view.stream_ok = false;
        view.dataset_ok = false;
        let err = validate_manifests(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar manifests.stream_manifest_hash invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn manifests_dataset_pin_rejected() {
        let mut view = manifest_view();
        view.dataset_ok = false;
        let err = validate_manifests(&view).expect_err("must reject");
        assert_eq!(
            err,
            "checkpoint sidecar manifests.dataset_manifest_hash invalid: ckpt-000100.json"
        );
    }

    #[test]
    fn manifests_absent_accepted() {
        assert_eq!(validate_manifests(&manifest_view()), Ok(()));
    }

    // ------------------------------------------------------------------
    // Digest shape (restates `rc_require.rs:189-197` for `_DIGEST_RE`).
    // ------------------------------------------------------------------

    #[test]
    fn digest_shape_gates() {
        let good = format!("sha256:{}", "ab".repeat(32));
        assert!(is_digest_shape(&good));
        assert!(!is_digest_shape(&format!("sha256:{}", "AB".repeat(32))));
        assert!(!is_digest_shape("sha256:ab"));
        assert!(!is_digest_shape("md5:ab"));
        assert!(!is_digest_shape(""));
        assert!(!is_digest_shape(&format!("sha256:{}", "ab".repeat(33))));
    }
}
