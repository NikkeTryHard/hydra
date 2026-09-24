//! packet_decode: thin decode/validate/partition/manifest boundary over `hydra-feed`.
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as the
//! sibling-owned `packet` framer; no new dependency). FORBIDS: parse/encode/
//! hash/pool/selection logic — arg-check + detach + struct return only. All
//! decode/validate/partition/manifest math lives in
//! `hydra_feed::{decode, validate, partition, manifest}` (which themselves
//! call the canon/digest owners); this file only validates arguments
//! attached, runs the feed pure functions detached, and wraps the results
//! attached.
//!
//! Scope split (no dual ownership):
//! - Framer fns (`frame_games`, `fetch_game_at`, `decode_zstd_verified`,
//!   `stem_of`, `group_key_for[_path]`, `wall_hash`, `assign_one`) live in
//!   sibling-owned `packet.rs`. This module NEVER frames bytes and NEVER
//!   synthesizes stems.
//! - This module owns the decode side of feed: `decode_frames_batch`
//!   (ordered merge), `validate_batch` + `validation_hash_of` (trap-8
//!   adapter flag IN, sim stays Python), `assign_partitions` (whole-corpus
//!   path), and `manifest_digest` (file-list digest).
//!
//! Stems are parallel params (NOT synthesized): the Python oracle passes
//! `stem_of(fpath)` as BOTH `object_id` and `packaged_object_id`
//! (`stream_decode.py:123-126`, `stream_read.py:78-81`). `object_id` feeds
//! the synthetic `game_id` fallback (`game-<12hex>` of the stem) when no
//! explicit id exists — synthesizing `frame-{idx}-{offset}` here would fork
//! game identity vs the oracle. Callers pass `stems.len() == frames.len()`;
//! a length mismatch is `ValueError`, never a silent truncate/pad.
//!
//! B1/B2 untouched (no RNG/Gumbel/randperm here): the split draw stays the
//! feed-owned `u64BE / 2**64` float on `sha256(f"{seed}|{group}")`, held-out
//! splits stay on the `torch.randperm` oracle behind KAT. There is NO
//! `feed::shuffle` edge here.
//!
//! Bytes path, NO DLPack/Arrow here: inputs are `Vec<u8>` (bytes/bytearray)
//! in, dicts + `sha256:<hex>` text out. The opaque `GameRecord` handle
//! carries the feed-owned `Vec<Value>` events across the validate boundary
//! so Python never re-parses (no second printer, no JSON round-trip fork).
//! No PyCapsule is ever constructed in this module, so the
//! capsule-outside-detach rule is N/A here (compute detached, wrap attached
//! still holds: batch work runs under `detach`, Python objects are built
//! only while attached).
//!
//! Concurrency pattern (matches `packet.rs` + `canon_rng.rs`): validate
//! attached, `py.detach(|| ...)` around every batch/blocking section
//! (fan-out/join never holds the interpreter), `MutexExt::lock_py_attached`
//! for the stats mutex, frozen pyclasses only (`&self` methods). GIL
//! attestation is spelled once wave-wide in `canon_rng.rs` (`gil_used =
//! false` default on PyO3 >= 0.28 covers this submodule too).
//! Single-cdylib tree: this module registers as the `hydra2._native.packet_decode`
//! submodule via `register`, mirroring `canon_rng::register` / `packet::register`.
//!
//! VERIFIED feed API (read in-tree, never guessed):
//! - `hydra_feed::decode::decode_game_object(&str, &str, &[u8])`
//!   `-> Result<GameRecord, DecodeErr>` (per-game owner; the batch below
//!   loops it in input order — the same ordered merge as
//!   `decode_frames_batch`, offsets travel out-of-band).
//! - `hydra_feed::decode::decode_frames_batch(&[Frame], &[(&str, &str)])`
//!   `-> Vec<Result<GameRecord, DecodeErr>>` (index-aligned; stems parallel).
//! - `hydra_feed::decode::GameRecord { game_id, object_id,`
//!   `packaged_object_id, events, raw_bytes_sha256, wall_tiles }`.
//! - `hydra_feed::decode::DecodeErr::{Utf8, TrailingNewline, BlankLine(u32),`
//!   `Json(u32), Shape(u32), StartEnd}` + `error_class()` (closed
//!   `utf8|trailing_newline|blank_line|json|shape|start_end`) +
//!   `event_index()`.
//! - `hydra_feed::validate::validate_batch(&[GameRecord], &AdapterProbe)`
//!   `-> Vec<Outcome>` (index-aligned; failures occupy their slot).
//! - `hydra_feed::validate::validation_hash_of(&GameRecord, &AdapterProbe)`
//!   `-> Option<String>` (thin).
//! - `hydra_feed::validate::AdapterProbe::{ok(), failed(&'static str)}`
//!   (bridge pins `"NotWired"` — no Python exception to name on this path).
//! - `hydra_feed::validate::Outcome { game_id, object_id, valid, error,`
//!   `validation_hash, checks }` + `ValidationError { class, event_index,`
//!   `message }` + `ErrorClass::as_str()` (closed `structure|event_order|`
//!   `tile_conservation|red_identity|legality|dora_shape`).
//! - `hydra_feed::partition::assign_partitions(&[GameIdentity], &SplitSpec)`
//!   `-> Result<SplitManifest, PartitionError>` (grouping + duplicates +
//!   wall-disjoint + canon-bound digest).
//! - `hydra_feed::partition::GameIdentity { game_id, object_id, source_id,`
//!   `player_ids, timestamp, wall_hash, decoded_hash }` +
//!   `GameIdentity::new` (empty player_ids -> `["unknown"]`).
//! - `hydra_feed::partition::SplitSpec { algorithm, version, seed, ratios,`
//!   `grouping_keys, wall_disjoint }` + `SplitManifest { spec, assignments,`
//!   `input_hashes, digest }` + `is_digest_text(&str) -> bool`
//!   (`sha256:<64 hex>` shape gate).
//! - `hydra_feed::manifest::build_manifest_from_paths(&str,`
//!   `Vec<(String, u64)>) -> StreamManifest` (sha-sort file order) +
//!   `manifest_digest(&StreamManifest) -> Result<String, ManifestError>`.
//!   Error mapping: every feed reject (decode/validate-shape/partition/
//!   manifest-canon) is a caller- or data-shape reject, so all map to
//!   `PyValueError` (never `PyOSError`, never silent skip). `PyOSError` is
//!   reserved for the stats-mutex poison path only.

use std::collections::BTreeMap;
use std::sync::Mutex;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBool, PyDict, PyModule};

/// Static adapter label pinned bridge-side when the sim probe is unavailable.
///
/// Mirrors the `validate.rs` contract: the `RiichiEnvExactSimulator.reset`
/// probe stays Python (trap-8); Rust takes the flag and labels the skipped
/// path `skipped_adapter_error:<name>`. There is no Python exception to name
/// on this path, so the bridge pins one label.
const ADAPTER_ERR_NOT_WIRED: &str = "NotWired";

/// Closed decode error taxonomy (mirrors `DecodeErr::error_class`).
#[cfg(test)]
const DECODE_TAXONOMY: [&str; 6] = [
    "utf8",
    "trailing_newline",
    "blank_line",
    "json",
    "shape",
    "start_end",
];

/// Closed validation error taxonomy (mirrors `ErrorClass::as_str`).
#[cfg(test)]
const VALIDATE_TAXONOMY: [&str; 6] = [
    "structure",
    "event_order",
    "tile_conservation",
    "red_identity",
    "legality",
    "dora_shape",
];

/// Cumulative judge counters (observability only, never identity).
#[derive(Debug, Default, Clone, Copy)]
struct DecodeStats {
    decodes: u64,
    validates: u64,
}

static DECODE_STATS: Mutex<DecodeStats> = Mutex::new(DecodeStats {
    decodes: 0,
    validates: 0,
});

fn bump_decodes(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = DECODE_STATS.lock_py_attached(py) {
        guard.decodes = guard.decodes.saturating_add(n);
    }
}

fn bump_validates(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = DECODE_STATS.lock_py_attached(py) {
        guard.validates = guard.validates.saturating_add(n);
    }
}

/// Opaque decoded-game handle: the feed-owned record crosses the
/// decode -> validate boundary without a JSON round-trip.
///
/// Events stay Rust-side (the `Vec<Value>` is never converted to Python
/// here — no second printer, no `serde_json` bridge edge). Identity fields
/// are exposed as getters for parity diffing (`game_id`, `raw_bytes_sha256`,
/// `wall_tiles`, `event_count`); the full event list is intentionally NOT
/// exposed (callers needing event contents stay on the Python oracle until
/// the scan caller migrates).
#[pyclass(name = "GameRecord", frozen, skip_from_py_object)]
pub struct PyGameRecord {
    inner: hydra_feed::decode::GameRecord,
}

#[pymethods]
impl PyGameRecord {
    #[getter]
    fn game_id(&self) -> String {
        self.inner.game_id.clone()
    }

    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.clone()
    }

    #[getter]
    fn packaged_object_id(&self) -> String {
        self.inner.packaged_object_id.clone()
    }

    #[getter]
    fn raw_bytes_sha256(&self) -> String {
        self.inner.raw_bytes_sha256.clone()
    }

    #[getter]
    fn wall_tiles(&self) -> Option<Vec<i64>> {
        self.inner.wall_tiles.map(|arr| arr.to_vec())
    }

    #[getter]
    fn event_count(&self) -> usize {
        self.inner.events.len()
    }
}

/// Decode a batch of framed game bytes with caller-supplied stems, output
/// index-aligned with the input (position IS the merge key).
///
/// Thin ordered merge over `decode::decode_game_object`: per-item decode in
/// input order, detached; failures occupy their slot as `{"ok": false, ...}`
/// dicts and never shift siblings (trap-13). `frames` are verbatim game
/// payloads (INCLUDING the trailing `\n`); `stems` are `(object_id,
/// packaged_object_id)` parallel params from `stem_of` per file — NEVER
/// synthesized here (the synthetic `game_id` fallback hashes the stem).
/// Length mismatch is `ValueError`.
///
/// Returns one dict per input position:
/// - ok: `{"ok": true, "index": i, "game_id": str, "object_id": str,`
///   `"packaged_object_id": str, "raw_bytes_sha256": "sha256:<hex>",`
///   `"wall_tiles": [136 ints] | None, "event_count": int,`
///   `"record": GameRecord}` (`record` chains into `validate_batch`).
/// - err: `{"ok": false, "index": i, "error_class": closed taxonomy,`
///   `"event_index": int | None, "detail": str}`.
#[pyfunction]
fn decode_frames_batch(
    py: Python<'_>,
    frames: Vec<Vec<u8>>,
    stems: Vec<(String, String)>,
) -> PyResult<Vec<Py<PyDict>>> {
    if frames.len() != stems.len() {
        return Err(PyValueError::new_err(format!(
            "packet_decode decode_frames_batch needs parallel frames/stems \
             (got {} frames, {} stems; stems are stem_of per file, never synthesized)",
            frames.len(),
            stems.len()
        )));
    }
    // Detached: per-item decode in input order (borrows the attached-owned
    // vecs across the detach — no Python objects cross, no clone).
    let outcomes: Vec<Result<hydra_feed::decode::GameRecord, hydra_feed::decode::DecodeErr>> = py
        .detach(|| {
            frames
                .iter()
                .zip(stems.iter())
                .map(|(bytes, (object_id, packaged_object_id))| {
                    hydra_feed::decode::decode_game_object(object_id, packaged_object_id, bytes)
                })
                .collect()
        });
    // Attached: wrap each slot (batch order preserved by construction).
    let mut wrapped: Vec<Py<PyDict>> = Vec::with_capacity(outcomes.len());
    for (idx, res) in outcomes.into_iter().enumerate() {
        let dict = PyDict::new(py);
        match res {
            Ok(rec) => {
                let game_id = rec.game_id.clone();
                let object_id = rec.object_id.clone();
                let packaged_object_id = rec.packaged_object_id.clone();
                let raw_bytes_sha256 = rec.raw_bytes_sha256.clone();
                let wall_tiles: Option<Vec<i64>> = rec.wall_tiles.map(|arr| arr.to_vec());
                let event_count = rec.events.len();
                let handle = Py::new(py, PyGameRecord { inner: rec })?;
                dict.set_item("ok", true)?;
                dict.set_item("index", idx)?;
                dict.set_item("game_id", game_id)?;
                dict.set_item("object_id", object_id)?;
                dict.set_item("packaged_object_id", packaged_object_id)?;
                dict.set_item("raw_bytes_sha256", raw_bytes_sha256)?;
                dict.set_item("wall_tiles", wall_tiles)?;
                dict.set_item("event_count", event_count)?;
                dict.set_item("record", handle)?;
            }
            Err(err) => {
                dict.set_item("ok", false)?;
                dict.set_item("index", idx)?;
                dict.set_item("error_class", err.error_class())?;
                dict.set_item("event_index", err.event_index())?;
                dict.set_item("detail", err.to_string())?;
            }
        }
        wrapped.push(dict.unbind());
    }
    bump_decodes(py, wrapped.len() as u64);
    Ok(wrapped)
}

/// Validate a batch of decoded games, output index-aligned with the input.
///
/// Thin wrapper over `validate::validate_batch`: per-record local checks
/// vec, failures occupy their slot as `{"valid": false, ...}` (never shift
/// siblings). `adapter_ok` carries the trap-8 sim verdict (`true` = full
/// `legality|calls|scores|termination = ok`; `false` pins
/// `legality = skipped_adapter_error:NotWired` with `calls|scores|`
/// `termination` ABSENT, oracle-exact). B1/B2 untouched: no RNG, no Gumbel,
/// no permutation here.
///
/// Returns one dict per input position: `{"game_id": str, "object_id": str,`
/// `"valid": bool, "error_class": str | None, "event_index": int | None,`
/// `"message": str | None, "validation_hash": "sha256:<hex>" | None,`
/// `"checks": {name: "ok" | "skipped_..."}}`.
#[pyfunction]
#[pyo3(signature = (records, adapter_ok=true))]
fn validate_batch(
    py: Python<'_>,
    records: Vec<Bound<'_, PyGameRecord>>,
    adapter_ok: bool,
) -> PyResult<Vec<Py<PyDict>>> {
    // Attached memcpy: clone the Rust records (events stay Rust-owned).
    let owned: Vec<hydra_feed::decode::GameRecord> =
        records.iter().map(|r| r.borrow().inner.clone()).collect();
    let probe = if adapter_ok {
        hydra_feed::validate::AdapterProbe::ok()
    } else {
        hydra_feed::validate::AdapterProbe::failed(ADAPTER_ERR_NOT_WIRED)
    };
    let outcomes = py.detach(|| hydra_feed::validate::validate_batch(&owned, &probe));
    let mut wrapped: Vec<Py<PyDict>> = Vec::with_capacity(outcomes.len());
    for outcome in &outcomes {
        let dict = PyDict::new(py);
        dict.set_item("game_id", &outcome.game_id)?;
        dict.set_item("object_id", &outcome.object_id)?;
        dict.set_item("valid", outcome.valid)?;
        dict.set_item(
            "error_class",
            outcome.error.as_ref().map(|e| e.class.as_str()),
        )?;
        dict.set_item(
            "event_index",
            outcome.error.as_ref().and_then(|e| e.event_index),
        )?;
        dict.set_item(
            "message",
            outcome.error.as_ref().map(|e| e.message.as_str()),
        )?;
        dict.set_item("validation_hash", outcome.validation_hash.clone())?;
        let checks = PyDict::new(py);
        for (name, value) in &outcome.checks {
            checks.set_item(*name, value)?;
        }
        dict.set_item("checks", checks)?;
        wrapped.push(dict.unbind());
    }
    bump_validates(py, wrapped.len() as u64);
    Ok(wrapped)
}

/// Thin `stream_decode.py:130` shape: one decoded game -> validation hash.
///
/// Returns the `sha256(canonical({game_id, checks}))` seal when valid, `None`
/// when invalid (mirrors `validation_hash_of`; never raises on game content).
/// `adapter_ok` pins the same trap-8 probe as [`validate_batch`].
#[pyfunction]
#[pyo3(signature = (record, adapter_ok=true))]
fn validation_hash_of(py: Python<'_>, record: &PyGameRecord, adapter_ok: bool) -> Option<String> {
    let owned = record.inner.clone();
    let probe = if adapter_ok {
        hydra_feed::validate::AdapterProbe::ok()
    } else {
        hydra_feed::validate::AdapterProbe::failed(ADAPTER_ERR_NOT_WIRED)
    };
    let hash = py.detach(|| hydra_feed::validate::validation_hash_of(&owned, &probe));
    bump_validates(py, 1);
    hash
}

fn dict_required_str(
    dict: &Bound<'_, PyDict>,
    field: &str,
    idx: usize,
    what: &str,
) -> PyResult<String> {
    let value = dict.get_item(field).map_err(|e| {
        PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} unreadable: {e}"
        ))
    })?;
    let Some(value) = value else {
        return Err(PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} missing"
        )));
    };
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} must be str, not bool"
        )));
    }
    value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} must be str"
        ))
    })
}

fn dict_optional_str(
    dict: &Bound<'_, PyDict>,
    field: &str,
    idx: usize,
    what: &str,
) -> PyResult<Option<String>> {
    let value = dict.get_item(field).map_err(|e| {
        PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} unreadable: {e}"
        ))
    })?;
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} must be str | None, not bool"
        )));
    }
    let text: String = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} must be str | None"
        ))
    })?;
    Ok(Some(text))
}

fn check_digest_shape(value: &str, field: &str, idx: usize, what: &str) -> PyResult<()> {
    if !hydra_feed::partition::is_digest_text(value) {
        return Err(PyValueError::new_err(format!(
            "packet_decode {what} {idx} field {field:?} must be sha256:<64 lowercase hex>"
        )));
    }
    Ok(())
}

/// Assign whole games to partitions; enforces grouping and duplicate checks.
///
/// Thin wrapper over `partition::assign_partitions` (the whole-corpus path;
/// the per-group stream path stays on `packet.assign_one` — the wave3 §1.10
/// ONE-fn rule means both share `assign_one` math, this entry is the corpus
/// loop only). `games` are pre-built identity dicts
/// (`{"game_id": str, "object_id": str, "source_id": str,`
/// `"player_ids": [str, ...], "timestamp": str | None,`
/// `"wall_hash": "sha256:<hex>" | None, "decoded_hash": "sha256:<hex>"}` —
/// missing `source_id` defaults to `"unknown"`, missing/empty `player_ids`
/// to `["unknown"]`, mirroring `_game_identity_for`); `spec` is
/// (`{"algorithm": str, "version": str, "seed": int >= 0,`
/// `"ratios": {partition: weight}, "grouping_keys": [str, ...],`
/// `"wall_disjoint": bool}`). Duplicate walls/games raise (never split);
/// wall-disjoint violations raise (train walls never meet eval walls).
///
/// Returns `{"assignments": {game_id: partition}, "input_hashes": {"spec":`
/// `sha, "games": sha}, "digest": "sha256:<hex>"}`.
#[pyfunction]
fn assign_partitions(
    py: Python<'_>,
    games: Vec<Bound<'_, PyDict>>,
    spec: Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    if games.is_empty() {
        return Err(PyValueError::new_err(
            "packet_decode assign_partitions needs at least one game identity",
        ));
    }
    // Attached memcpy: identity dicts -> owned feed structs.
    let mut owned_games: Vec<hydra_feed::partition::GameIdentity> = Vec::with_capacity(games.len());
    for (idx, game) in games.iter().enumerate() {
        let game_id = dict_required_str(game, "game_id", idx, "game")?;
        let object_id = dict_required_str(game, "object_id", idx, "game")?;
        if game_id.is_empty() {
            return Err(PyValueError::new_err(format!(
                "packet_decode game {idx} field \"game_id\" must be non-empty"
            )));
        }
        if object_id.is_empty() {
            return Err(PyValueError::new_err(format!(
                "packet_decode game {idx} field \"object_id\" must be non-empty"
            )));
        }
        let source_id = match game.get_item("source_id").map_err(|e| {
            PyValueError::new_err(format!(
                "packet_decode game {idx} field \"source_id\" unreadable: {e}"
            ))
        })? {
            None => "unknown".to_string(),
            Some(v) => {
                if v.is_none() {
                    "unknown".to_string()
                } else {
                    v.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "packet_decode game {idx} field \"source_id\" must be str"
                        ))
                    })?
                }
            }
        };
        let player_ids: Vec<String> = match game.get_item("player_ids").map_err(|e| {
            PyValueError::new_err(format!(
                "packet_decode game {idx} field \"player_ids\" unreadable: {e}"
            ))
        })? {
            None => vec!["unknown".to_string()],
            Some(v) => {
                if v.is_none() {
                    vec!["unknown".to_string()]
                } else {
                    v.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "packet_decode game {idx} field \"player_ids\" must be list[str]"
                        ))
                    })?
                }
            }
        };
        let timestamp = dict_optional_str(game, "timestamp", idx, "game")?;
        let wall_hash = dict_optional_str(game, "wall_hash", idx, "game")?;
        let decoded_hash = dict_required_str(game, "decoded_hash", idx, "game")?;
        check_digest_shape(&decoded_hash, "decoded_hash", idx, "game")?;
        if let Some(wall) = wall_hash.as_deref() {
            check_digest_shape(wall, "wall_hash", idx, "game")?;
        }
        owned_games.push(hydra_feed::partition::GameIdentity::new(
            &game_id,
            &object_id,
            &source_id,
            player_ids,
            timestamp,
            wall_hash,
            &decoded_hash,
        ));
    }
    // Attached spec memcpy.
    let algorithm = dict_required_str(&spec, "algorithm", 0, "spec")?;
    let version = dict_required_str(&spec, "version", 0, "spec")?;
    let seed_value = spec.get_item("seed").map_err(|e| {
        PyValueError::new_err(format!("packet_decode spec field \"seed\" unreadable: {e}"))
    })?;
    let Some(seed_value) = seed_value else {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"seed\" missing (must be a non-negative int)",
        ));
    };
    if seed_value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"seed\" must be a non-negative int, not bool",
        ));
    }
    let seed: u64 = seed_value.extract().map_err(|_| {
        PyValueError::new_err("packet_decode spec field \"seed\" must be a non-negative int")
    })?;
    let ratios_value = spec.get_item("ratios").map_err(|e| {
        PyValueError::new_err(format!(
            "packet_decode spec field \"ratios\" unreadable: {e}"
        ))
    })?;
    let Some(ratios_value) = ratios_value else {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"ratios\" missing (must be {partition: weight})",
        ));
    };
    let ratios_dict: Bound<'_, PyDict> = ratios_value.extract().map_err(|_| {
        PyValueError::new_err("packet_decode spec field \"ratios\" must be a mapping")
    })?;
    let mut ratios: BTreeMap<String, f64> = BTreeMap::new();
    for (key, value) in ratios_dict.iter() {
        let name: String = key
            .extract()
            .map_err(|_| PyValueError::new_err("packet_decode spec ratios keys must be str"))?;
        if value.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "packet_decode spec ratios[{name:?}] must be a finite number, not bool"
            )));
        }
        let weight: f64 = value.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "packet_decode spec ratios[{name:?}] must be a finite number"
            ))
        })?;
        ratios.insert(name, weight);
    }
    let grouping_keys_value = spec.get_item("grouping_keys").map_err(|e| {
        PyValueError::new_err(format!(
            "packet_decode spec field \"grouping_keys\" unreadable: {e}"
        ))
    })?;
    let Some(grouping_keys_value) = grouping_keys_value else {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"grouping_keys\" missing (must be list[str])",
        ));
    };
    let grouping_keys: Vec<String> = grouping_keys_value.extract().map_err(|_| {
        PyValueError::new_err("packet_decode spec field \"grouping_keys\" must be list[str]")
    })?;
    let wall_disjoint_value = spec.get_item("wall_disjoint").map_err(|e| {
        PyValueError::new_err(format!(
            "packet_decode spec field \"wall_disjoint\" unreadable: {e}"
        ))
    })?;
    let Some(wall_disjoint_value) = wall_disjoint_value else {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"wall_disjoint\" missing (must be bool)",
        ));
    };
    if !wall_disjoint_value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "packet_decode spec field \"wall_disjoint\" must be bool",
        ));
    }
    let wall_disjoint: bool = wall_disjoint_value.extract().map_err(|_| {
        PyValueError::new_err("packet_decode spec field \"wall_disjoint\" must be bool")
    })?;
    let owned_spec = hydra_feed::partition::SplitSpec {
        algorithm,
        version,
        seed,
        ratios,
        grouping_keys,
        wall_disjoint,
    };
    // Detached: the whole assignment fan-out (grouping + draws + disjoint +
    // canon-bound digest) runs off-interpreter.
    let manifest = py
        .detach(|| hydra_feed::partition::assign_partitions(&owned_games, &owned_spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Attached: shape-gate the digest (fail-closed, never defaulted), then wrap.
    if !hydra_feed::partition::is_digest_text(&manifest.digest) {
        return Err(PyValueError::new_err(
            "packet_decode assign_partitions digest failed the sha256:<hex> shape gate",
        ));
    }
    let out = PyDict::new(py);
    let assignments = PyDict::new(py);
    for (game_id, part) in &manifest.assignments {
        assignments.set_item(game_id, part)?;
    }
    let input_hashes = PyDict::new(py);
    for (key, sha) in &manifest.input_hashes {
        input_hashes.set_item(key, sha)?;
    }
    out.set_item("assignments", assignments)?;
    out.set_item("input_hashes", input_hashes)?;
    out.set_item("digest", &manifest.digest)?;
    Ok(out.unbind())
}

/// Bind a corpus file list for RunSpec provenance.
///
/// Thin wrapper over `manifest::build_manifest_from_paths` (sha-sort file
/// order) + `manifest::digest` (ASCII fast-path with fail-closed canon
/// cross-check, feed-owned). `files` are `(root-id, relative POSIX path,
/// compressed bytes)` triples in ANY order (the feed sorts by sha256-hex of
/// `root-id + NUL + relpath` — the hash orders, never splits); `root` is
/// provenance display only and does not enter the digest. Empty path,
/// absolute path, or empty root-id is `ValueError`.
///
/// Returns the `sha256:<hex>` digest (shape-gated attached, never defaulted).
#[pyfunction]
#[pyo3(signature = (files, root=""))]
fn manifest_digest(
    py: Python<'_>,
    files: Vec<(String, String, u64)>,
    root: &str,
) -> PyResult<String> {
    for (idx, (root_id, path, _)) in files.iter().enumerate() {
        if root_id.is_empty() {
            return Err(PyValueError::new_err(format!(
                "packet_decode manifest_digest file {idx} root_id must be non-empty"
            )));
        }
        if path.is_empty() {
            return Err(PyValueError::new_err(format!(
                "packet_decode manifest_digest file {idx} path must be non-empty relative POSIX"
            )));
        }
        if path.starts_with('/') {
            return Err(PyValueError::new_err(format!(
                "packet_decode manifest_digest file {idx} path must be relative POSIX, got absolute"
            )));
        }
    }
    let owned_root = root.to_string();
    let digest = py
        .detach(move || {
            let manifest = hydra_feed::manifest::build_manifest_from_paths(&owned_root, files);
            hydra_feed::manifest::manifest_digest(&manifest)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !hydra_feed::partition::is_digest_text(&digest) {
        return Err(PyValueError::new_err(
            "packet_decode manifest_digest failed the sha256:<hex> shape gate",
        ));
    }
    Ok(digest)
}

/// Judge observability: `(decodes, validates)` via an interpreter-safe lock.
#[pyfunction]
fn judge_stats(py: Python<'_>) -> PyResult<(u64, u64)> {
    let guard = DECODE_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("packet_decode judge stats mutex poisoned"))?;
    Ok((guard.decodes, guard.validates))
}

/// Fail-close comparator for the in-module judge tests (mirrors
/// `canon_rng::check_match`: recomputed == recorded or `Err`, never a
/// default/empty accept).
#[cfg(test)]
fn check_match(recorded: &str, recomputed: &str) -> Result<(), String> {
    if recorded == recomputed {
        Ok(())
    } else {
        Err(format!(
            "digest mismatch: recorded {recorded} != recomputed {recomputed}"
        ))
    }
}

/// Register the `packet_decode` submodule (mirrors `canon_rng::register` /
/// `packet::register`): compute detached, wrap attached; single cdylib, no
/// new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "packet_decode")?;
    sub.add_class::<PyGameRecord>()?;
    sub.add_function(wrap_pyfunction!(decode_frames_batch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_batch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validation_hash_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(assign_partitions, &sub)?)?;
    sub.add_function(wrap_pyfunction!(manifest_digest, &sub)?)?;
    sub.add_function(wrap_pyfunction!(judge_stats, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod packet_decode_tests {
    use super::*;

    fn game_bytes(tag: &str) -> Vec<u8> {
        format!(
            "{{\"type\":\"start_game\",\"game_id\":\"{tag}\"}}\n{{\"type\":\"dahai\",\"pai\":5}}\n{{\"type\":\"end_game\"}}\n"
        )
        .into_bytes()
    }

    fn wall_game_bytes(tag: &str, wall: &[i64]) -> Vec<u8> {
        let tiles = wall
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"type\":\"start_game\",\"game_id\":\"{tag}\",\"wall\":[{tiles}]}}\n{{\"type\":\"end_game\"}}\n"
        )
        .into_bytes()
    }

    fn valid_wall() -> Vec<i64> {
        (0..136).collect()
    }

    fn stems(n: usize) -> Vec<(String, String)> {
        (0..n)
            .map(|i| (format!("stem-{i}"), format!("stem-{i}")))
            .collect()
    }

    fn identity_dict<'py>(
        py: Python<'py>,
        game_id: &str,
        decoded_hash: &str,
        wall_hash: Option<&str>,
    ) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        d.set_item("game_id", game_id).unwrap();
        d.set_item("object_id", format!("obj-{game_id}")).unwrap();
        d.set_item("source_id", "lobby").unwrap();
        d.set_item("player_ids", vec![format!("p-{game_id}")])
            .unwrap();
        d.set_item("timestamp", format!("20240101{game_id}"))
            .unwrap();
        d.set_item("wall_hash", wall_hash).unwrap();
        d.set_item("decoded_hash", decoded_hash).unwrap();
        d
    }

    fn spec_dict<'py>(py: Python<'py>) -> Bound<'py, PyDict> {
        let spec = PyDict::new(py);
        spec.set_item("algorithm", "sha-draw").unwrap();
        spec.set_item("version", "1").unwrap();
        spec.set_item("seed", 7u64).unwrap();
        let ratios = PyDict::new(py);
        ratios.set_item("train", 1.0f64).unwrap();
        spec.set_item("ratios", ratios).unwrap();
        spec.set_item("grouping_keys", Vec::<String>::new())
            .unwrap();
        spec.set_item("wall_disjoint", true).unwrap();
        spec
    }

    #[test]
    fn batch_order_and_identity_parity() {
        // Contract: output is index-aligned with the input (position IS the
        // merge key); per-game identity (game_id + verbatim-bytes sha) agrees
        // with the single-game feed owner exactly.
        Python::initialize();
        Python::attach(|py| {
            let frames = vec![game_bytes("g0"), game_bytes("g1"), game_bytes("g2")];
            let expected_shas: Vec<String> = frames
                .iter()
                .map(|b| hydra_feed::digest::sha256_hex(b))
                .collect();
            let out = decode_frames_batch(py, frames.clone(), stems(3)).unwrap();
            assert_eq!(out.len(), 3);
            for (idx, dict) in out.iter().enumerate() {
                let bound = dict.bind(py);
                assert!(
                    bound
                        .get_item("ok")
                        .unwrap()
                        .unwrap()
                        .extract::<bool>()
                        .unwrap()
                );
                assert_eq!(
                    bound
                        .get_item("index")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    idx
                );
                let want_id = format!("g{idx}");
                assert_eq!(
                    bound
                        .get_item("game_id")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    want_id
                );
                assert_eq!(
                    bound
                        .get_item("raw_bytes_sha256")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    expected_shas[idx]
                );
                // Single-game owner agrees byte-for-byte on identity.
                let solo = hydra_feed::decode::decode_game_object(
                    &format!("stem-{idx}"),
                    &format!("stem-{idx}"),
                    &frames[idx],
                )
                .unwrap();
                assert_eq!(
                    bound
                        .get_item("game_id")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    solo.game_id
                );
                assert_eq!(
                    bound
                        .get_item("raw_bytes_sha256")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    solo.raw_bytes_sha256
                );
                assert!(
                    check_match(
                        &bound
                            .get_item("raw_bytes_sha256")
                            .unwrap()
                            .unwrap()
                            .extract::<String>()
                            .unwrap(),
                        &solo.raw_bytes_sha256
                    )
                    .is_ok()
                );
            }
        });
    }

    #[test]
    fn stems_are_parallel_and_never_synthesized() {
        // Contract: stems.len() != frames.len() fails closed; the synthetic
        // game_id fallback hashes the CALLER stem (no frame-{idx} invention).
        Python::initialize();
        Python::attach(|py| {
            let frames = vec![
                game_bytes("explicit"),
                b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n".to_vec(),
            ];
            // Mismatch fails closed.
            assert!(decode_frames_batch(py, frames.clone(), stems(1)).is_err());
            assert!(decode_frames_batch(py, frames.clone(), stems(3)).is_err());
            // Fallback game_id hashes the stem, not the position.
            let out = decode_frames_batch(
                py,
                vec![frames[1].clone()],
                vec![("my-stem".to_string(), "my-stem".to_string())],
            )
            .unwrap();
            let bound = out[0].bind(py);
            let solo =
                hydra_feed::decode::decode_game_object("my-stem", "my-stem", &frames[1]).unwrap();
            assert!(solo.game_id.starts_with("game-"));
            assert_eq!(
                bound
                    .get_item("game_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                solo.game_id
            );
            // A different stem forks the fallback id (stem feeds identity).
            let other =
                hydra_feed::decode::decode_game_object("other-stem", "other-stem", &frames[1])
                    .unwrap();
            assert_ne!(solo.game_id, other.game_id);
        });
    }

    #[test]
    fn taxonomy_is_closed() {
        // Contract: every decode reject renders the closed taxonomy string;
        // per-slot failures never shift siblings.
        Python::initialize();
        Python::attach(|py| {
            let good = game_bytes("ok");
            let cases: Vec<(Vec<u8>, &str)> = vec![
                (b"\xff\xfe\n".to_vec(), "utf8"),
                (
                    b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}".to_vec(),
                    "trailing_newline",
                ),
                (
                    b"{\"type\":\"start_game\"}\n\n{\"type\":\"end_game\"}\n".to_vec(),
                    "blank_line",
                ),
                (
                    b"{\"type\":\"start_game\"}\nnot json\n{\"type\":\"end_game\"}\n".to_vec(),
                    "json",
                ),
                (
                    b"{\"type\":\"start_game\"}\n{\"no_type\":1}\n{\"type\":\"end_game\"}\n"
                        .to_vec(),
                    "shape",
                ),
                (
                    b"{\"type\":\"dahai\"}\n{\"type\":\"end_game\"}\n".to_vec(),
                    "start_end",
                ),
            ];
            let mut frames: Vec<Vec<u8>> = vec![good];
            for (bytes, _) in &cases {
                frames.push(bytes.clone());
            }
            let stem_list = stems(frames.len());
            let out = decode_frames_batch(py, frames, stem_list).unwrap();
            assert_eq!(out.len(), 1 + cases.len());
            assert!(
                out[0]
                    .bind(py)
                    .get_item("ok")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            for (i, (_, want_class)) in cases.iter().enumerate() {
                let bound = out[i + 1].bind(py);
                assert!(
                    !bound
                        .get_item("ok")
                        .unwrap()
                        .unwrap()
                        .extract::<bool>()
                        .unwrap()
                );
                let class: String = bound
                    .get_item("error_class")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                assert_eq!(&class, want_class);
                assert!(DECODE_TAXONOMY.contains(&class.as_str()));
                assert_eq!(
                    bound
                        .get_item("index")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    i + 1
                );
            }
        });
    }

    #[test]
    fn validate_batch_marks_and_hashes() {
        // Contract: valid games validate with a sha256: seal over
        // canonical({game_id, checks}); invalid games occupy their slot with
        // the closed class (tile_conservation here via a dup-tile wall).
        Python::initialize();
        Python::attach(|py| {
            let good = game_bytes("v0");
            let mut dup_wall = valid_wall();
            dup_wall[135] = 0; // duplicate 0, missing 135: not a permutation.
            let bad = wall_game_bytes("v1", &dup_wall);
            let out = decode_frames_batch(py, vec![good, bad], stems(2)).unwrap();
            assert!(out.iter().all(|d| {
                d.bind(py)
                    .get_item("ok")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            }));
            let records: Vec<Bound<'_, PyGameRecord>> = out
                .iter()
                .map(|d| {
                    d.bind(py)
                        .get_item("record")
                        .unwrap()
                        .unwrap()
                        .extract::<Bound<'_, PyGameRecord>>()
                        .unwrap()
                })
                .collect();
            let outcomes = validate_batch(py, records, true).unwrap();
            assert_eq!(outcomes.len(), 2);
            let ok0 = outcomes[0].bind(py);
            assert!(
                ok0.get_item("valid")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let hash0: Option<String> = ok0
                .get_item("validation_hash")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let hash0 = hash0.unwrap();
            assert!(hydra_feed::partition::is_digest_text(&hash0));
            let bad1 = outcomes[1].bind(py);
            assert!(
                !bad1
                    .get_item("valid")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let class: Option<String> = bad1
                .get_item("error_class")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(class.as_deref(), Some("tile_conservation"));
            assert!(VALIDATE_TAXONOMY.contains(&class.unwrap().as_str()));
            assert!(
                bad1.get_item("validation_hash")
                    .unwrap()
                    .unwrap()
                    .extract::<Option<String>>()
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn dora_4_shim_fails_with_shape_gate() {
        // Contract: the (4,) dora shim fails closed as dora_shape; dora (5,)
        // exact is preserved (no 4-shim, sentinel contiguity feed-owned).
        Python::initialize();
        Python::attach(|py| {
            let shim = b"{\"type\":\"start_game\",\"game_id\":\"d0\"}\n{\"type\":\"dahai\",\"dora_indicators\":[1,2,3,4]}\n{\"type\":\"end_game\"}\n"
                .to_vec();
            let out = decode_frames_batch(py, vec![shim], stems(1)).unwrap();
            assert!(
                out[0]
                    .bind(py)
                    .get_item("ok")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let record: Bound<'_, PyGameRecord> = out[0]
                .bind(py)
                .get_item("record")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let outcomes = validate_batch(py, vec![record], true).unwrap();
            let first = outcomes[0].bind(py);
            assert!(
                !first
                    .get_item("valid")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let class: Option<String> = first
                .get_item("error_class")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(class.as_deref(), Some("dora_shape"));
        });
    }

    #[test]
    fn validation_hash_of_is_thin_and_skipped_probe_labelled() {
        // Contract: validation_hash_of agrees with validate_batch's seal;
        // adapter_ok=false pins the NotWired skipped label (no sim here).
        Python::initialize();
        Python::attach(|py| {
            let out = decode_frames_batch(py, vec![game_bytes("h0")], stems(1)).unwrap();
            let record: Bound<'_, PyGameRecord> = out[0]
                .bind(py)
                .get_item("record")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let probe_ok = hydra_feed::validate::AdapterProbe::ok();
            let solo_owned = record.borrow().inner.clone();
            let expect = hydra_feed::validate::validation_hash_of(&solo_owned, &probe_ok);
            let via_thin = validation_hash_of(py, &record.borrow(), true);
            assert_eq!(via_thin, expect);
            assert!(via_thin.is_some());
            // Skipped probe still seals (legality skipped, never raised).
            let via_skipped = validation_hash_of(py, &record.borrow(), false);
            assert!(via_skipped.is_some());
            let probe_bad = hydra_feed::validate::AdapterProbe::failed(ADAPTER_ERR_NOT_WIRED);
            let expect_bad = hydra_feed::validate::validation_hash_of(&solo_owned, &probe_bad);
            assert_eq!(via_skipped, expect_bad);
            // Batch agrees with thin on the same record.
            let batch = validate_batch(py, vec![record], true).unwrap();
            let batch_hash: Option<String> = batch[0]
                .bind(py)
                .get_item("validation_hash")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(batch_hash, via_thin);
            assert!(check_match(&via_thin.unwrap(), &batch_hash.unwrap()).is_ok());
        });
    }

    #[test]
    fn assign_partitions_happy_digest_and_gates() {
        // Contract: whole games assign (never split), digest + input hashes
        // are sha256: shaped; exact/near duplicates + bad ratios fail closed.
        Python::initialize();
        Python::attach(|py| {
            let sha_a = format!("sha256:{}", "a".repeat(64));
            let sha_b = format!("sha256:{}", "b".repeat(64));
            let g0 = identity_dict(py, "pa", &sha_a, None);
            let g1 = identity_dict(py, "pb", &sha_b, None);
            let spec = spec_dict(py);
            let out = assign_partitions(py, vec![g0.clone(), g1.clone()], spec.clone()).unwrap();
            let bound = out.bind(py);
            let digest: String = bound
                .get_item("digest")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(hydra_feed::partition::is_digest_text(&digest));
            let assignments: Bound<'_, PyDict> = bound
                .get_item("assignments")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(assignments.len(), 2);
            assert_eq!(
                assignments
                    .get_item("pa")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "train"
            );
            let hashes: Bound<'_, PyDict> = bound
                .get_item("input_hashes")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(hashes.get_item("spec").unwrap().is_some());
            assert!(hashes.get_item("games").unwrap().is_some());
            // Feed owner agrees on the digest for the same inputs.
            let feed_games = vec![
                hydra_feed::partition::GameIdentity::new(
                    "pa",
                    "obj-pa",
                    "lobby",
                    vec!["p-pa".to_string()],
                    Some("20240101pa".to_string()),
                    None,
                    &sha_a,
                ),
                hydra_feed::partition::GameIdentity::new(
                    "pb",
                    "obj-pb",
                    "lobby",
                    vec!["p-pb".to_string()],
                    Some("20240101pb".to_string()),
                    None,
                    &sha_b,
                ),
            ];
            let feed_spec = hydra_feed::partition::SplitSpec {
                algorithm: "sha-draw".to_string(),
                version: "1".to_string(),
                seed: 7,
                ratios: BTreeMap::from([("train".to_string(), 1.0)]),
                grouping_keys: vec![],
                wall_disjoint: true,
            };
            let feed_manifest =
                hydra_feed::partition::assign_partitions(&feed_games, &feed_spec).unwrap();
            assert!(check_match(&digest, &feed_manifest.digest).is_ok());
            // Exact duplicates (same decoded hash) fail closed.
            let dup = identity_dict(py, "pc", &sha_a, None);
            assert!(assign_partitions(py, vec![g0.clone(), dup], spec.clone()).is_err());
            // Non-sha decoded_hash fails the attached shape gate.
            let shapeless = identity_dict(py, "pd", "not-a-digest", None);
            assert!(assign_partitions(py, vec![shapeless], spec.clone()).is_err());
            // Empty games fail closed.
            assert!(assign_partitions(py, vec![], spec.clone()).is_err());
            // Bad seed (negative) fails closed.
            let bad_spec = PyDict::new(py);
            bad_spec.set_item("algorithm", "sha-draw").unwrap();
            bad_spec.set_item("version", "1").unwrap();
            bad_spec.set_item("seed", -1i64).unwrap();
            let ratios = PyDict::new(py);
            ratios.set_item("train", 1.0f64).unwrap();
            bad_spec.set_item("ratios", ratios).unwrap();
            bad_spec
                .set_item("grouping_keys", Vec::<String>::new())
                .unwrap();
            bad_spec.set_item("wall_disjoint", true).unwrap();
            assert!(assign_partitions(py, vec![g0.clone()], bad_spec).is_err());
        });
    }

    #[test]
    fn near_duplicates_fail_closed() {
        // Contract: same wall hash across two games is a near duplicate
        // (rejected whole-game, never split across partitions).
        Python::initialize();
        Python::attach(|py| {
            let wall_hex = format!("sha256:{}", "c".repeat(64));
            let g0 = identity_dict(
                py,
                "na",
                &format!("sha256:{}", "1".repeat(64)),
                Some(&wall_hex),
            );
            let g1 = identity_dict(
                py,
                "nb",
                &format!("sha256:{}", "2".repeat(64)),
                Some(&wall_hex),
            );
            let spec = spec_dict(py);
            assert!(assign_partitions(py, vec![g0, g1], spec).is_err());
        });
    }

    #[test]
    fn manifest_digest_shapes_and_order() {
        // Contract: digest is sha256: shaped, deterministic, and
        // input-order independent (feed sha-sorts by root-id + relpath);
        // bad root-ids/paths fail.
        Python::initialize();
        Python::attach(|py| {
            let files_a = vec![
                ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
            ];
            let files_b = vec![
                ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
            ];
            let da = manifest_digest(py, files_a, "").unwrap();
            let db = manifest_digest(py, files_b, "").unwrap();
            assert!(hydra_feed::partition::is_digest_text(&da));
            assert_eq!(da, db);
            let manifest = hydra_feed::manifest::build_manifest_from_paths(
                "",
                vec![
                    ("ryu".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                    ("ryu".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                ],
            );
            // Feed file order is sha-sorted (hash orders, never splits).
            let keys: Vec<String> = manifest
                .files
                .iter()
                .map(|f| hydra_feed::manifest::order_key(&f.root_id, &f.path))
                .collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted);
            let expect = hydra_feed::manifest::manifest_digest(&manifest).unwrap();
            assert!(check_match(&da, &expect).is_ok());
            // Root id namespaces: same paths under another root differ.
            let dc = manifest_digest(
                py,
                vec![
                    ("lobby".to_string(), "a.mjai.json.zst".to_string(), 10u64),
                    ("lobby".to_string(), "b.mjai.json.zst".to_string(), 20u64),
                ],
                "",
            )
            .unwrap();
            assert_ne!(da, dc);
            // Absolute + empty paths fail closed attached.
            assert!(
                manifest_digest(
                    py,
                    vec![("ryu".to_string(), "/abs/x".to_string(), 1u64)],
                    ""
                )
                .is_err()
            );
            assert!(
                manifest_digest(py, vec![("ryu".to_string(), "".to_string(), 1u64)], "").is_err()
            );
            assert!(
                manifest_digest(py, vec![("".to_string(), "a".to_string(), 1u64)], "").is_err()
            );
        });
    }

    #[test]
    fn detach_smoke_and_stats() {
        // Shape assert: batch work runs detached; stats publish attached.
        Python::initialize();
        Python::attach(|py| {
            let before: (u64, u64) = judge_stats(py).unwrap();
            let out = decode_frames_batch(py, vec![game_bytes("s0")], stems(1)).unwrap();
            assert_eq!(out.len(), 1);
            let record: Bound<'_, PyGameRecord> = out[0]
                .bind(py)
                .get_item("record")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let _ = validate_batch(py, vec![record], true).unwrap();
            let _ = manifest_digest(
                py,
                vec![("ryu".to_string(), "x.mjai.json.zst".to_string(), 3u64)],
                "",
            )
            .unwrap();
            let after: (u64, u64) = judge_stats(py).unwrap();
            assert!(after.0 > before.0);
            assert!(after.1 > before.1);
            let sum = py.detach(|| 2 + 2);
            assert_eq!(sum, 4);
        });
    }

    #[test]
    fn mismatch_check_fails_closed() {
        assert!(check_match("sha256:aa", "sha256:aa").is_ok());
        assert!(check_match("sha256:aa", "sha256:bb").is_err());
        assert!(check_match("sha256:aa", "").is_err());
    }
}
