//! columnar: thin Arrow-C stream capsule boundary over `hydra-shard` expand.
//!
//! DAG (Wave5 B6): this module is the bridge's shard edge (`feed + shard +
//! pyo3` once the manifest edge lands; the `hydra-shard` + `arrow-array` /
//! `arrow-schema` bridge dependencies are Main-owned — see ASSUMPTION-MANIFEST).
//! FORBIDS: parse/encode/hash/pool logic, writer properties, batch contents —
//! arg-check + detach + capsule return only. All expand/sim math, the batch
//! schema, and every parquet/IPC writer knob live shard-side; this file only
//! validates descriptors attached, runs expand detached, and exports the
//! resulting batches as ONE single-consume Arrow-C stream capsule attached.
//!
//! B4 capsule-OUTSIDE-detach (load-bearing): attached `memcpy` of `(ptr, len)`
//! descriptors to owned buffers, then `py.detach(|| assemble + expand)` builds
//! ONE owned `RecordBatch` with no Python interaction; the capsule
//! (`PyCapsule_New`, destructor, move semantics) is created ATTACHED after
//! the detach returns. Never construct a capsule inside `detach` (capsule
//! creation needs the interpreter; the FFI `Box` handoff across the detach
//! boundary is the whole shape).
//!
//! Capsule protocol (wave2-arrow-parquet §§1-2+4, Arrow PyCapsule page):
//! - Stream capsule name is exactly `"arrow_array_stream"`; schema capsules
//!   are `"arrow_schema"`. A name mismatch fails closed at import
//!   (`PyCapsule_GetPointer` NULL + raised exception, propagated, never
//!   defaulted).
//! - Single-consume movable: one capsule → one immediate import. Consuming
//!   twice, holding the struct after `release`, or reading children after
//!   base release is UB/leak/double-free. The destructor calls `release` when
//!   non-NULL (consumer already moved: NULL, struct freed only) then frees
//!   the struct box — the Arrow §8.1 producer pattern verbatim.
//! - `requested_schema` is best-effort negotiation: `None` exports the
//!   canonical shard schema; `Some("arrow_schema" capsule)` must compare
//!   equal to the canonical schema or the call raises (incompatible physical
//!   representation is a `ValueError`, never a silent re-encode).
//! - String physical is `Utf8` (`pa.string()`), NOT `StringView`: the canonical
//!   batch schema equals Python `parquet._ACTOR_SCHEMA` exactly (13
//!   `ACTOR_FIELDS` order, `pa.string()`/`pa.int64`, `chosen_action_id` sole
//!   nullable). Compat-level negotiation stays consumer-side (polars/pyarrow);
//!   parquet goldens compare logical values regardless of arrival physical.
//!
//! Shard-owned contract (referenced, never implemented here):
//! batch schema == `parquet._ACTOR_SCHEMA` (13 fields, `decision_id` col 2 —
//! the `decision_ids()` sorted-ids `dataset_hash` input), `uint8` bool views,
//! caller-sized inputs so rows land near [`BATCH`], and the
//! actor-never-privileged triple firewall. `created_by` UNSET (M3),
//! `ARROW:schema` KV on BOTH writers (M4), the privileged re-freeze gate
//! (M5), and `store_schema` on BOTH sites (M8) are Writer-owned (`writer.rs`).
//! The Rust parquet writer is GREENFIELD (no Rust writer exists today —
//! `parquet_join.rs:1-12` reads privileged parquet only, never writes; the
//! parity test is greenfield, NOT regression).
//!
//! Probes: A (capsule 10k no-leak loop) runs from the Python judge
//! (`_rust_columnar.probe_a_capsule_release`); B (writer/footer close +
//! `store_schema` + `created_by` + KV) and C (same-batches → both-writers
//! `dataset_hash` + file-sha parity) are Writer-owned.
//! Bytes path, NO DLPack here: inputs are opaque caller descriptors, output
//! is an Arrow-C stream capsule. DLPack (device tensors) belongs to the loop
//! (Phase 5), never this packet-bytes seam.
//!
//! Concurrency pattern (matches `stream.rs` + `canon_rng.rs` + `replay.rs`):
//! validate attached, `py.detach(|| ...)` around the whole expand fan-out
//! (never per-item attach), capsule built attached. Frozen surface only (this
//! module exposes free functions; no pyclass, no shared cells beyond the
//! attach-guarded stats mutex). GIL attestation (`gil_used=false` default on
//! PyO3 ≥0.28) is spelled once in `canon_rng.rs` wave-wide and inherited by
//! this submodule's registration — not repeated here.
//!
//! Single-cdylib tree: this module registers as the `hydra2._native.columnar`
//! submodule via `register`, mirroring `replay::register` / `canon_rng::register`.
//!
//! ASSUMPTION-MANIFEST (pending Main — scope freeze: Writer stays shard-only,
//! Main applies the bridge Cargo edge; this file MUST NOT compile until
//! applied; DO NOT guess-fix locally): `hydra-shard = { path = "../shard" }` +
//! `arrow-array = "=59.3.0"` + `arrow-schema = "=59.3.0"` (both already in
//! the workspace lockfile via `parquet =59.3.0`; no other new dependency —
//! no serde_jcs/rayon/arrow-ipc/safetensors/anyhow bridge-side).
//!
//! SIBLING-OWNED SHARD API (this file MUST NOT compile until the sibling
//! `replay::expand` module lands; confirmed names MUST NOT be guess-fixed
//! locally):
//! - `hydra_shard::replay::expand::OwnedSimGameInput` (`game_id`,
//!   `source_object_id`, `split`, `rules_hash`, `adapter_hash`,
//!   `action_table_hash: String`, `wall_digest: Option<String>` as DigestText
//!   text — `Some("sha256:"+hex)` walled / `None` wall-less, SIM mark bound
//!   never invented — plus `decisions: Vec<OwnedSimDecisionInput>` with
//!   `round_idx: u32`, `seat: u8`, `chosen_action_id: Option<u32>`,
//!   `chosen_unresolved: Option<String>`, `actor_observation_json: String`,
//!   `observation_hash: String`).
//! - `expand_batch_owned(games: &[OwnedSimGameInput])
//!   -> Result<ExpandedBatch, ExpandError>` (Result — `.map_err` below via
//!   `Display`, no anyhow edge). Borrowed twins (`SimGameInput`,
//!   `expand_batch`) and the thin `replay::expand_batch_games` entry remain
//!   for zero-copy callers — the bridge uses the owned entry ONLY.
//! - `actor_schema() -> arrow_schema::SchemaRef` (canonical batch schema ==
//!   Python `parquet._ACTOR_SCHEMA`; NOT `expand_schema`).
//! - `ExpandedBatch::{into_batch(self) -> RecordBatch, num_rows,
//!   decision_ids(&self) -> &StringArray}` (col `DECISION_ID_COL = 2`;
//!   single `RecordBatch` per call — the bridge wraps `vec![batch]` for the
//!   capsule stream; the caller sizes inputs so rows land near [`BATCH`]).
//! - `ExpandError: Display + std::error::Error` (mapped to `PyValueError`).
//! - Confirmed (`assemble_owned_games`, Expand-provided): each blob
//!   is one UTF-8 JSON object (`game_id`/`source_object_id`/`split`/
//!   `rules_hash`/`adapter_hash`/`action_table_hash: string`,
//!   `wall_digest: string|null`, `decisions: [{round_idx, seat,
//!   chosen_action_id: int|null, chosen_unresolved: string|null,
//!   observation_hash: string, actor_observation: ANY json value
//!   (re-canonicalized at capture)}]`); shape failures surface as
//!   `ExpandError::BlobFrame{index, detail}`, value gates stay capture-side.
//!   Bridge flow: memcpy attached → ONE detach { assemble + expand } →
//!   capsule attached. No `ptrs`/`caps` entry on the shard side.
//!
//! Soundness (Expand-confirmed): the bridge copies `(ptr, len)` descriptors
//! into Rust-owned `Vec<u8>` buffers ATTACHED (GIL held ⇒ caller memory
//! stable), then detaches and crosses ONLY owned buffers — never borrows
//! Python memory across `detach`. No Engine/sim port this phase (sim stays
//! Python); owned==borrowed parity is proven by Expand's in-module test.

use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use hydra_shard::replay::expand::{actor_schema, assemble_owned_games, expand_batch_owned};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;

/// Single-writer batch width (Wave5: ONE const, owned HERE).
///
/// Both parquet writers pin `write_batch_size` to this value (5-exact), and
/// search/eval import it from here later — no second literal anywhere.
/// Writer references it with a doc note, never a duplicate definition.
pub const BATCH: usize = 8192;

/// Exact Arrow PyCapsule name for a C stream export (wave2 §1, MUST match).
pub const STREAM_CAPSULE_NAME: &str = "arrow_array_stream";

/// Exact Arrow PyCapsule name for a C schema capsule (wave2 §1, MUST match).
pub const SCHEMA_CAPSULE_NAME: &str = "arrow_schema";

/// Cumulative capsule observability (counts only, never identity).
#[derive(Debug, Default, Clone, Copy)]
struct CapsuleStats {
    capsules: u64,
    batches: u64,
}

static CAPSULE_STATS: Mutex<CapsuleStats> = Mutex::new(CapsuleStats {
    capsules: 0,
    batches: 0,
});

fn bump_capsules(py: Python<'_>, batches: u64) -> Result<(), PyErr> {
    let mut guard = CAPSULE_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("hydra2 columnar stats mutex poisoned"))?;
    guard.capsules = guard.capsules.saturating_add(1);
    guard.batches = guard.batches.saturating_add(batches);
    Ok(())
}

/// Interpreter-visible batch width: the ONE `BATCH` const (search/eval import
/// this, never a second literal).
#[pyfunction]
fn batch_size() -> usize {
    BATCH
}

/// Judge observability: `(capsules_exported, batches_exported)` (never identity).
#[pyfunction]
fn capsule_stats(py: Python<'_>) -> PyResult<(u64, u64)> {
    let guard = CAPSULE_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("hydra2 columnar stats mutex poisoned"))?;
    Ok((guard.capsules, guard.batches))
}

/// Fail-close string comparator for the in-module judge tests (mirrors
/// `canon_rng::check_match`: recomputed == recorded or `Err`, never a
/// default/empty accept).
#[cfg(test)]
fn check_match(recorded: &str, recomputed: &str) -> Result<(), String> {
    if recorded == recomputed {
        Ok(())
    } else {
        Err(String::from("columnar digest mismatch for judge subject"))
    }
}

/// Best-effort schema negotiation (pure, detach-safe): `None` accepts the
/// canonical export; `Some` must compare exactly equal or the call fails
/// closed (wave2 §3 — incompatible physical is an error, never a silent
/// re-encode; parquet goldens compare logical values regardless).
fn schemas_compatible(exported: &Schema, requested: &Schema) -> bool {
    exported == requested
}

/// Owned batch vec → `RecordBatchReader` without guessing at arrow-rs
/// constructor names (local impl over the certain trait surface).
struct VecReader {
    schema: SchemaRef,
    batches: std::vec::IntoIter<RecordBatch>,
}

impl RecordBatchReader for VecReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for VecReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next().map(Ok)
    }
}

/// Pure build half (detach-safe: no interpreter interaction): pin the
/// canonical schema, fail closed on a mixed-schema batch vec (a silent
/// writer fork is worse than an error — Writer normalizes dictionary domains
/// pre-write per wave2 §10, the bridge never re-encodes).
fn build_reader(
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    for (idx, batch) in batches.iter().enumerate() {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(format!(
                "columnar batch {idx} schema != canonical actor_schema (mixed-schema export refused)"
            ));
        }
    }
    Ok((schema, batches))
}

/// Attached-only half: import a `requested_schema` capsule (`"arrow_schema"`,
/// best-effort) and compare against the canonical schema. `None` skips
/// negotiation. A NULL pointer (name mismatch / invalid capsule) propagates
/// the capsule exception; an incompatible schema raises `ValueError`.
fn negotiate_schema(
    py: Python<'_>,
    exported: &Schema,
    requested_schema: &Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(capsule) = requested_schema else {
        return Ok(());
    };
    let ptr =
        unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"arrow_schema".as_ptr()) };
    if ptr.is_null() {
        return Err(PyErr::fetch(py));
    }
    let ffi_schema = unsafe { &*(ptr as *const FFI_ArrowSchema) };
    let requested = Schema::try_from(ffi_schema).map_err(|err| {
        PyValueError::new_err(format!("hydra2 requested_schema import failed: {err}"))
    })?;
    if schemas_compatible(exported, &requested) {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "hydra2 requested_schema incompatible with canonical columnar schema \
             (best-effort negotiation refused; parquet goldens compare logical values)",
        ))
    }
}

/// Arrow §8.1 producer destructor: call `release` when non-NULL (a moved/consumed
/// stream has NULL `release` — free the struct box only), then free the struct
/// box. Never touches `private_data` directly (opaque to consumers AND here).
unsafe extern "C" fn stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: CPython passes the capsule being destroyed; the pointer was set
    // by `export_stream_capsule` to a `Box<FFI_ArrowArrayStream>` with this
    // destructor. Release-if-non-NULL then free the box; never touch private_data.
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"arrow_array_stream".as_ptr()) }
        as *mut FFI_ArrowArrayStream;
    if ptr.is_null() {
        return;
    }
    let stream = unsafe { &mut *ptr };
    if let Some(release) = stream.release {
        unsafe { release(ptr) };
    }
    drop(unsafe { Box::from_raw(ptr) });
}

/// Attached-only half: box the FFI stream and wrap it in a single-consume
/// `"arrow_array_stream"` capsule (B4 — this function REQUIRES the attached
/// `Python<'_>` token by signature; callers pass batches built detached).
/// On capsule-allocation failure the stream box is dropped (release runs,
/// no leak) and the interpreter error propagates.
fn export_stream_capsule(py: Python<'_>, stream: FFI_ArrowArrayStream) -> PyResult<Py<PyAny>> {
    let raw = Box::into_raw(Box::new(stream));
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            raw as *mut std::ffi::c_void,
            c"arrow_array_stream".as_ptr(),
            Some(stream_capsule_destructor),
        )
    };
    if capsule.is_null() {
        drop(unsafe { Box::from_raw(raw) });
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

/// Pure descriptor check (attached call-site; unit-tested below): same count,
/// no nulls, non-empty — fails closed before any copy work.
fn check_descriptors(ptrs: &[u64], byte_caps: &[usize]) -> Result<(), String> {
    if ptrs.is_empty() {
        return Err(String::from(
            "hydra2 columnar descriptors must not be empty",
        ));
    }
    if ptrs.len() != byte_caps.len() {
        return Err(format!(
            "hydra2 columnar descriptor length mismatch: {} ptrs vs {} caps",
            ptrs.len(),
            byte_caps.len()
        ));
    }
    if ptrs.contains(&0) {
        return Err(String::from(
            "hydra2 columnar null pointer descriptor fails closed",
        ));
    }
    Ok(())
}

/// Attached-only copy: `(ptr, len)` descriptors → Rust-owned byte buffers.
/// `memcpy` ONLY (no parse — FORBIDS); sound because the GIL is held, so the
/// caller keeps the source memory stable for the call duration. The owned
/// buffers (never borrowed Python memory) are what crosses the `detach`.
fn copy_descriptors_attached(ptrs: &[u64], byte_caps: &[usize]) -> Vec<Vec<u8>> {
    ptrs.iter()
        .zip(byte_caps.iter())
        .map(|(ptr, cap)| unsafe { std::slice::from_raw_parts(*ptr as *const u8, *cap).to_vec() })
        .collect()
}

/// Expand games to columnar batches and export ONE single-consume Arrow-C
/// stream capsule.
///
/// `ptrs`/`byte_caps` are opaque caller descriptors (same count, no nulls,
/// non-empty — validated attached, fails closed) memcpied to Rust-owned
/// buffers ATTACHED, then assembled + expanded under ONE `py.detach`
/// (compute detached); the capsule is created ATTACHED from the owned batch
/// (B4). The caller sizes inputs so rows land near [`BATCH`].
/// `requested_schema` (`None`, or an `"arrow_schema"` capsule) negotiates
/// best-effort. `ExpandedBatch` yields a SINGLE `RecordBatch` per call,
/// wrapped as `vec![batch]` for the capsule stream.
///
/// Returns the capsule object (consumed via
/// `pa.RecordBatchReader.from_pycapsule`); consume exactly once, immediately.
#[pyfunction]
#[pyo3(signature = (ptrs, byte_caps, requested_schema = None))]
fn next_into_capsule(
    py: Python<'_>,
    ptrs: Vec<u64>,
    byte_caps: Vec<usize>,
    requested_schema: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    check_descriptors(&ptrs, &byte_caps).map_err(PyValueError::new_err)?;
    let blobs = copy_descriptors_attached(&ptrs, &byte_caps);
    // Compute DETACHED: shard-owned assemble (parse) + expand over OWNED
    // buffers; errors map via Display (no anyhow edge on the bridge).
    let batch: RecordBatch = py
        .detach(|| {
            assemble_owned_games(&blobs)
                .and_then(|games| expand_batch_owned(&games))
                .map(|expanded| expanded.into_batch())
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    // Attached-only: canonical schema pin + best-effort negotiation + export.
    let schema = actor_schema();
    negotiate_schema(py, &schema, &requested_schema)?;
    let (schema, batches) = build_reader(schema, vec![batch]).map_err(PyValueError::new_err)?;
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(VecReader {
        schema: schema.clone(),
        batches: batches.into_iter(),
    });
    let stream = FFI_ArrowArrayStream::new(reader);
    let capsule = export_stream_capsule(py, stream)?;
    bump_capsules(py, 1)?;
    Ok(capsule)
}

/// Register the `columnar` submodule (mirrors `replay::register` /
/// `canon_rng::register`): compute detached, capsule attached; single cdylib,
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "columnar")?;
    sub.add_function(wrap_pyfunction!(next_into_capsule, &sub)?)?;
    sub.add_function(wrap_pyfunction!(batch_size, &sub)?)?;
    sub.add_function(wrap_pyfunction!(capsule_stats, &sub)?)?;
    crate::rows_seal::register(&sub)?;
    crate::stream_scan::register(&sub)?;
    crate::dataset_parse::register(&sub)?;
    crate::stream_decode::register(&sub)?;
    crate::stream_manifest::register(&sub)?;
    crate::ingest_leaves::register(&sub)?;
    crate::partition_dups::register(&sub)?;
    crate::quarantine_inbox::register(&sub)?;
    crate::stream_report::register(&sub)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod columnar_tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_array::array::StringArray;
    use arrow_schema::{DataType, Field};

    fn canonical_test_schema() -> SchemaRef {
        // Mirrors the canonical actor physicals: Utf8 strings
        // (== parquet._ACTOR_SCHEMA pa.string(), NOT StringView).
        Schema::new(vec![
            Field::new("decision_id", DataType::Utf8, false),
            Field::new("seat", DataType::Int32, false),
        ])
        .into()
    }

    #[test]
    fn batch_width_is_8192_single_const() {
        assert_eq!(BATCH, 8192);
        assert_eq!(batch_size(), 8192);
    }

    #[test]
    fn capsule_names_match_arrow_spec() {
        // Wave2 §1 MUST-match strings (validity = GetPointer name check).
        assert_eq!(STREAM_CAPSULE_NAME, "arrow_array_stream");
        assert_eq!(SCHEMA_CAPSULE_NAME, "arrow_schema");
    }

    #[test]
    fn mismatch_check_fails_closed() {
        assert!(check_match("sha256:aa", "sha256:aa").is_ok());
        assert!(check_match("sha256:aa", "sha256:bb").is_err());
        assert!(check_match("sha256:aa", "").is_err());
    }

    #[test]
    fn requested_schema_negotiation_is_exact() {
        let exported = canonical_test_schema();
        assert!(schemas_compatible(&exported, &exported.clone()));
        let view_physical = Schema::new(vec![
            Field::new("decision_id", DataType::Utf8View, false),
            Field::new("seat", DataType::Int32, false),
        ]);
        assert!(!schemas_compatible(&exported, &view_physical));
        let widened = Schema::new(vec![
            Field::new("decision_id", DataType::LargeUtf8, false),
            Field::new("seat", DataType::Int32, false),
        ]);
        assert!(!schemas_compatible(&exported, &widened));
    }

    #[test]
    fn build_reader_accepts_uniform_rejects_mixed() -> Result<(), ArrowError> {
        let schema = canonical_test_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(StringArray::from(vec!["g0r0:0:0"]))
                    as std::sync::Arc<dyn arrow_array::Array>,
                std::sync::Arc::new(Int32Array::from(vec![0])) as _,
            ],
        )?;
        assert!(build_reader(schema.clone(), vec![batch.clone()]).is_ok());
        assert!(build_reader(schema.clone(), vec![]).is_ok());
        let other = Schema::new(vec![Field::new("other", DataType::Int32, false)]).into();
        let foreign = RecordBatch::try_new(
            other,
            vec![std::sync::Arc::new(Int32Array::from(vec![1])) as _],
        )?;
        assert!(build_reader(schema, vec![batch, foreign]).is_err());
        Ok(())
    }

    #[test]
    fn vec_reader_serves_schema_and_rows() -> Result<(), ArrowError> {
        let schema = canonical_test_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(StringArray::from(vec!["g0r0:0:0"]))
                    as std::sync::Arc<dyn arrow_array::Array>,
                std::sync::Arc::new(Int32Array::from(vec![0])) as _,
            ],
        )?;
        let mut reader = VecReader {
            schema: schema.clone(),
            batches: vec![batch].into_iter(),
        };
        assert_eq!(reader.schema(), schema);
        assert_eq!(
            reader.next().map(|r| r.map(|b| b.num_rows()).unwrap()),
            Some(1)
        );
        assert!(reader.next().is_none());
        Ok(())
    }

    #[test]
    fn descriptor_check_fails_closed() {
        assert!(check_descriptors(&[], &[]).is_err());
        assert!(check_descriptors(&[8], &[64]).is_ok());
        assert!(check_descriptors(&[8], &[64, 64]).is_err());
        assert!(check_descriptors(&[0], &[64]).is_err());
    }

    #[test]
    fn detach_smoke_shape() {
        // B4 shape assert: batch work runs detached; capsule export stays
        // attached by signature (`export_stream_capsule` requires
        // `Python<'_>`). This pins the detached half in-process.
        Python::initialize();
        Python::attach(|py| {
            let sum = py.detach(|| 2 + 2);
            assert_eq!(sum, 4);
            let (schema, batches) = py.detach(|| {
                let schema = canonical_test_schema();
                (schema.clone(), Vec::<RecordBatch>::new())
            });
            assert!(build_reader(schema, batches).is_ok());
        });
    }
}
