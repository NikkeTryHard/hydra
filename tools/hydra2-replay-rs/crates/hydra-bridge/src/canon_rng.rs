//! canon_rng: thin digest + new-stream RNG boundary over `hydra-feed`.
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as the
//! rest of the bridge; no new dependency). FORBIDS: parse/encode/hash/pool
//! logic — arg-check + detach + struct return only. All bytes/hash/RNG math
//! lives in `hydra_feed::{canon, digest, rng}`; this file only validates
//! arguments attached, runs the feed pure functions detached, and wraps the
//! results attached.
//!
//! Bytes path, NO DLPack/Arrow here: inputs are `&[u8]` (bytes/bytearray) in,
//! `sha256:<hex>` text out. No PyCapsule is ever constructed in this module,
//! so the capsule-outside-detach rule is N/A here (compute detached, wrap
//! attached still holds: batch work runs under `detach`, Python objects are
//! built only while attached).
//!
//! Concurrency pattern (matches `stream.rs` + replay `expand_games`): validate
//! attached, `py.detach(|| ...)` around every batch/blocking section
//! (fan-out/join never holds the interpreter), `MutexExt::lock_py_attached`
//! for the stats mutex, `PyOnceLock` for one-time defaults, frozen pyclasses
//! only (`&self` methods, interior mutability via `Mutex` where needed).
//! Future single-cdylib tree: this module registers as the `canon_rng`
//! submodule (`hydra2_replay_rs.canon_rng` today,
//! `hydra_bridge._native.canon_rng` once the Phase-6 maturin `module-name`
//! cutover lands) via `register`, mirroring `replay::register`. The legacy
//! `hydra2_replay_rs` entry point is untouched.
//!
//! GIL attestation (`gil_used=false` default ships on PyO3 >=0.28 — spelled
//! once here wave-wide, no per-submodule annotation): the single
//! `#[pymodule]` entry plus every `PyModule::new` submodule inherits the
//! `Py_MOD_GIL_NOT_USED` default, so interpreter-level thread-safety is
//! attested without annotating each submodule.
//!
//! DAG note (Wave5 B6, Phase 3 columnar): bridge deps become
//! `feed + shard + pyo3` when the Arrow-C batch surface lands; this module
//! adds no shard edge (bytes digests only) and needs no Cargo.toml change.
//!
//! SIBLING-OWNED FEED API (ASSUMPTION-pending-sibling — Phase1CanonDigest /
//! Phase1RngFixed own `hydra-feed`; this file MUST NOT compile until they
//! land; DO NOT guess-fix signatures locally):
//! - `hydra_feed::canon::parse_canonical_bytes(&[u8], &str)
//!   -> Result<serde_json::Value, CanonError>` (Result — `.map_err` below).
//! - `hydra_feed::digest::sha256_hex(&[u8]) -> String` (`sha256:`-prefixed
//!   lowercase hex, infallible — called bare below).
//! - `hydra_feed::digest::sha256_file(&Path) -> io::Result<String>` (1MiB
//!   chunks, `sha256:`-prefixed — `.map_err` to `PyOSError` below).
//! - `hydra_feed::digest::of_canonical(&serde_json::Value)
//!   -> Result<String, CanonError>` (Result — `.map_err` below).
//! - `hydra_feed::rng::StreamKey { domain: [u8; 14], len: u32, seed: [u8; 32] }`
//!   (pub fields; `len` = significant domain-tag length, owner-confirmed by
//!   Phase1RngFixed: `philox_block` mixes only `domain[..len]`, padding
//!   ignored — KAT `kat4_domain_separation_and_padding_ignored` pins it).
//! - `hydra_feed::rng::philox_block(&StreamKey, u64) -> [u32; 4]`.
//! - `hydra_feed::rng::bounded(u32, u32) -> u32` (Lemire; `n == 0` guarded
//!   bridge-side before the call).
//! - `hydra_feed::rng::seed_map_fixture() -> Vec<(String, [u8; 32])>` (3+).
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::exceptions::{PyOSError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyBytes, PyModule};

/// 1 MiB chunk default for the file digest path (mirrors
/// `artifacts/digest.py` `_CHUNK_SIZE`; the feed pure function owns the
/// loop, this is only the config default).
const DEFAULT_CHUNK: usize = 1 << 20;

/// One-time default, initialized without deadlocking the interpreter.
static DEFAULT_CHUNK_BYTES: PyOnceLock<usize> = PyOnceLock::new();

fn default_chunk_bytes_impl(py: Python<'_>) -> usize {
    *DEFAULT_CHUNK_BYTES.get_or_init(py, || DEFAULT_CHUNK)
}

/// Cumulative judge counters (observability only, never identity).
#[derive(Debug, Default, Clone, Copy)]
struct JudgeStats {
    digests: u64,
    batches: u64,
}

static JUDGE_STATS: Mutex<JudgeStats> = Mutex::new(JudgeStats {
    digests: 0,
    batches: 0,
});

fn bump_digests(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = JUDGE_STATS.lock_py_attached(py) {
        guard.digests = guard.digests.saturating_add(n);
    }
}

fn bump_batches(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = JUDGE_STATS.lock_py_attached(py) {
        guard.batches = guard.batches.saturating_add(n);
    }
}

/// Frozen digest config: cheapest Sync + free-threaded-safe holder for the
/// chunk/pool knobs. No `&mut` methods; shared state (if ever added) goes
/// behind a `Mutex`, never a borrowed cell.
#[pyclass(name = "CanonConfig", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CanonConfig {
    /// Default chunk size for the file digest path (bytes).
    #[pyo3(get)]
    pub chunk_bytes: usize,
    /// Fixed pool width recorded in run metadata (0 = caller pool).
    #[pyo3(get)]
    pub pool_threads: usize,
}

#[pymethods]
impl CanonConfig {
    #[new]
    fn new(chunk_bytes: usize, pool_threads: usize) -> PyResult<Self> {
        if chunk_bytes == 0 {
            return Err(PyValueError::new_err(
                "canon config chunk_bytes must be >= 1",
            ));
        }
        Ok(Self {
            chunk_bytes,
            pool_threads,
        })
    }
}

/// Frozen new-stream spec: domain tag (1..=14 bytes, zero-padded to
/// `[u8; 14]`) + 32-byte seed. `len` is the significant tag length, NOT a
/// stream extent (owner-confirmed: `philox_block` mixes only `domain[..len]`).
///
/// NEW streams only: never use for held-out splits (`torch.randperm` stays
/// the oracle) or sha-Gumbels (replicated verbatim elsewhere).
#[pyclass(name = "StreamSpec", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct StreamSpec {
    domain: [u8; 14],
    domain_len: u32,
    seed: [u8; 32],
}
#[pymethods]
impl StreamSpec {
    #[new]
    fn new(domain: Vec<u8>, seed: Vec<u8>) -> PyResult<Self> {
        if domain.is_empty() || domain.len() > 14 {
            return Err(PyValueError::new_err(
                "canon stream domain must be 1..=14 bytes",
            ));
        }
        if seed.len() != 32 {
            return Err(PyValueError::new_err(
                "canon stream seed must be exactly 32 bytes",
            ));
        }
        let mut padded_domain = [0u8; 14];
        padded_domain[..domain.len()].copy_from_slice(&domain);
        let mut fixed_seed = [0u8; 32];
        fixed_seed.copy_from_slice(&seed);
        Ok(Self {
            domain: padded_domain,
            domain_len: domain.len() as u32,
            seed: fixed_seed,
        })
    }

    #[getter]
    fn domain_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.domain[..self.domain_len as usize]).unbind()
    }

    #[getter]
    fn domain_len(&self) -> u32 {
        self.domain_len
    }

    #[getter]
    fn seed_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.seed).unbind()
    }
}

// ASSUMPTION: `digest::sha256_hex(&[u8]) -> String` (infallible, bare call).
/// Raw-bytes digest: `&[u8]` in, `sha256:<hex>` out. The feed owns the hash;
/// the whole hash runs detached.
#[pyfunction]
fn sha256_hex(py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
    let out = py.detach(|| hydra_feed::digest::sha256_hex(&data));
    bump_digests(py, 1);
    Ok(out)
}

/// Chunked file digest (second identity path; MUST agree with the in-memory
/// path on identical input). Blocking reads run detached.
#[pyfunction]
fn sha256_file(py: Python<'_>, path: PathBuf) -> PyResult<String> {
    let out = py
        .detach(|| hydra_feed::digest::sha256_file(&path))
        .map_err(|e| PyOSError::new_err(format!("canon sha256_file failed: {e}")))?;
    bump_digests(py, 1);
    Ok(out)
}

// ASSUMPTION: `canon::parse_canonical_bytes(&[u8], &str) -> Result<Value, _>` +
// ASSUMPTION: `digest::of_canonical(&Value) -> Result<String, _>` (Result).
/// Canonical-JSON digest: JSON document bytes in, `sha256:<hex>` out. Parses
/// with the feed I-JSON boundary (`parse_canonical_bytes`), canonicalizes
/// via JCS, and hashes — all detached. Rejects (dup keys, NaN/Inf, lone
/// surrogates, non-string keys) surface as `ValueError`.
#[pyfunction]
fn of_canonical_json(py: Python<'_>, doc: Vec<u8>) -> PyResult<String> {
    let out = py
        .detach(|| -> Result<String, String> {
            let value = hydra_feed::canon::parse_canonical_bytes(&doc, "bridge:of_canonical_json")
                .map_err(|e| format!("canon parse rejected: {e:?}"))?;
            hydra_feed::digest::of_canonical(&value)
                .map_err(|e| format!("canon digest failed: {e:?}"))
        })
        .map_err(PyValueError::new_err)?;
    bump_digests(py, 1);
    Ok(out)
}

/// Detached batch digest over caller-owned bytes: one `detach` around the
/// whole fan-out (never per-item attach), results wrapped attached.
#[pyfunction]
fn batch_sha256(py: Python<'_>, items: Vec<Vec<u8>>) -> PyResult<Vec<String>> {
    let count = items.len() as u64;
    let out =
        py.detach(|| items.iter().map(|b| hydra_feed::digest::sha256_hex(b)).collect::<Vec<_>>());
    bump_digests(py, count);
    bump_batches(py, 1);
    Ok(out)
}

// ASSUMPTION: same canon/digest Result pair as `of_canonical_json`.
/// Detached batch canonical-JSON digest. Per-item rejects ride as a
/// call-level `ValueError` naming the bridge entry (row-level record ids
/// belong to the packet slice, not this thin boundary).
#[pyfunction]
fn batch_of_canonical(py: Python<'_>, items: Vec<Vec<u8>>) -> PyResult<Vec<String>> {
    let count = items.len() as u64;
    let out = py
        .detach(|| -> Result<Vec<String>, String> {
            let mut acc = Vec::with_capacity(items.len());
            for doc in &items {
                let value =
                    hydra_feed::canon::parse_canonical_bytes(doc, "bridge:batch_of_canonical")
                        .map_err(|e| format!("canon parse rejected: {e:?}"))?;
                let digest = hydra_feed::digest::of_canonical(&value)
                    .map_err(|e| format!("canon digest failed: {e:?}"))?;
                acc.push(digest);
            }
            Ok(acc)
        })
        .map_err(PyValueError::new_err)?;
    bump_digests(py, count);
    bump_batches(py, 1);
    Ok(out)
}

/// Open a NEW stream spec (validated frozen value, no RNG consumed yet).
#[pyfunction]
fn open_stream(domain: Vec<u8>, seed: Vec<u8>) -> PyResult<StreamSpec> {
    StreamSpec::new(domain, seed)
}

// Contract (owner-confirmed): `rng::StreamKey{domain,len,seed}` pub fields;
// `rng::philox_block(&StreamKey, u64) -> [u32; 4]`, mixes `domain[..len]` only.
// `&StreamSpec` borrow via the generated `&T: PyFunctionArgument` impl
// (`extract_pyclass_ref`, valid for frozen) — verified in pyo3 0.29.2
// `src/impl_/extract_argument.rs:178-189`.
/// One Philox block for an opened spec: pure function of key + counter
/// (bit-exact replay; block index, NOT word index). Tiny pure work runs
/// detached for GIL hygiene.
#[pyfunction]
fn philox_block(
    py: Python<'_>,
    spec: &StreamSpec,
    block: u64,
) -> PyResult<(u32, u32, u32, u32)> {
    let key = hydra_feed::rng::StreamKey {
        domain: spec.domain,
        len: spec.domain_len,
        seed: spec.seed,
    };
    let words = py.detach(|| hydra_feed::rng::philox_block(&key, block));
    Ok((words[0], words[1], words[2], words[3]))
}

// ASSUMPTION: `rng::bounded(u32, u32) -> u32` (Lemire).
/// ONLY sanctioned `[0, n)` path (Lemire, never `% n`). Fails closed on
/// `n == 0`; `bounded(_, 1) == 0` is defined.
#[pyfunction]
fn bounded(word: u32, n: u32) -> PyResult<u32> {
    if n == 0 {
        return Err(PyValueError::new_err("canon bounded: n must be >= 1"));
    }
    Ok(hydra_feed::rng::bounded(word, n))
}

/// Detached batch Lemire draws (one `detach` around the whole fan-out).
#[pyfunction]
fn bounded_batch(py: Python<'_>, words: Vec<u32>, n: u32) -> PyResult<Vec<u32>> {
    if n == 0 {
        return Err(PyValueError::new_err(
            "canon bounded_batch: n must be >= 1",
        ));
    }
    Ok(py.detach(|| {
        words
            .into_iter()
            .map(|w| hydra_feed::rng::bounded(w, n))
            .collect()
    }))
}
// Contract (owner-confirmed): `fixed::validate_ranks(&[u8; 4]) -> Result<(), FixedError>`;
// `fixed::utility_fixed([u8; 4]) -> Result<[i64; 4], FixedError>` (rank `r` ->
// `(5 - 2*r) * FIXED_SCALE`); `fixed::utility_for_ranks_fixed(&[i64; 4],
// &[u8; 4]) -> Result<[i64; 4], FixedError>` (mirrors `utility()` indexing);
// `fixed::exact_total_is_zero(&[f64; 4]) -> Result<bool, FixedError>` (m8:
// wide exponent spreads resolve to `Ok(bool)` via the in-feed exact stack
// big-int fallback, Fraction-equivalent — never wrap, never assume zero;
// `Err(Overflow)` is defensive-only, unreachable for 4 finite `f64`).
// Error mapping (F7 discriminates): `InvalidRank`/`DuplicateRank`/
// `NonFiniteValue` -> `PyValueError`, `Overflow` -> `PyOverflowError`.
// below()/bounded() parity UNCHANGED (intentional divergence documented —
// `bounded(_, 1) == 0` stays; this surface is add-only).
/// Map a feed `FixedError` onto the Python boundary: overflow is distinct
/// (`PyOverflowError`) so callers can discriminate the m8 checked-fallback;
/// everything else is a caller-shape reject (`PyValueError`).
fn map_fixed_err(e: hydra_feed::fixed::FixedError) -> PyErr {
    match e {
        hydra_feed::fixed::FixedError::Overflow => PyOverflowError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Validate ranks as a strict permutation of `1..=4` (no ties, no gaps).
/// Shape checked attached (`len != 4` fails closed); the feed check runs
/// detached.
#[pyfunction]
fn validate_ranks(py: Python<'_>, ranks: Vec<u8>) -> PyResult<()> {
    if ranks.len() != 4 {
        return Err(PyValueError::new_err(
            "canon validate_ranks: ranks must be exactly 4 entries",
        ));
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&ranks);
    py.detach(|| hydra_feed::fixed::validate_ranks(&arr))
        .map_err(map_fixed_err)
}

/// Canonical zero-sum fixed placement points for validated ranks: rank 1 ->
/// `+3·S`, 2 -> `+1·S`, 3 -> `-1·S`, 4 -> `-3·S` (`S = FIXED_SCALE`
/// micro-points). Shapes checked attached; the mapping runs detached.
#[pyfunction]
fn utility_fixed(py: Python<'_>, ranks: Vec<u8>) -> PyResult<Vec<i64>> {
    if ranks.len() != 4 {
        return Err(PyValueError::new_err(
            "canon utility_fixed: ranks must be exactly 4 entries",
        ));
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&ranks);
    let out = py
        .detach(|| hydra_feed::fixed::utility_fixed(arr))
        .map_err(map_fixed_err)?;
    Ok(out.to_vec())
}

/// Map validated ranks through a caller manifest's fixed `rank_values`
/// (mirrors `utility()` indexing `rank_values[rank-1]`). Both quads checked
/// attached; the mapping runs detached.
#[pyfunction]
fn utility_for_ranks_fixed(
    py: Python<'_>,
    rank_values: Vec<i64>,
    ranks: Vec<u8>,
) -> PyResult<Vec<i64>> {
    if rank_values.len() != 4 {
        return Err(PyValueError::new_err(
            "canon utility_for_ranks_fixed: rank_values must be exactly 4 entries",
        ));
    }
    if ranks.len() != 4 {
        return Err(PyValueError::new_err(
            "canon utility_for_ranks_fixed: ranks must be exactly 4 entries",
        ));
    }
    let mut values = [0i64; 4];
    values.copy_from_slice(&rank_values);
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&ranks);
    let out = py
        .detach(|| hydra_feed::fixed::utility_for_ranks_fixed(&values, &arr))
        .map_err(map_fixed_err)?;
    Ok(out.to_vec())
}

/// Exact zero-sum over four `f64` values (integer-only, no float
/// accumulation, no epsilon). `true` iff the exact sum is zero; m8-scale
/// exponent spreads resolve via the in-feed exact fallback
/// (Fraction-equivalent, never assumed zero; `Overflow` ->
/// `PyOverflowError` is defensive-only); non-finite inputs are
/// `PyValueError`. Shape checked attached; the alignment runs detached.
#[pyfunction]
fn exact_total_is_zero(py: Python<'_>, vals: Vec<f64>) -> PyResult<bool> {
    if vals.len() != 4 {
        return Err(PyValueError::new_err(
            "canon exact_total_is_zero: vals must be exactly 4 entries",
        ));
    }
    let mut arr = [0f64; 4];
    arr.copy_from_slice(&vals);
    py.detach(|| hydra_feed::fixed::exact_total_is_zero(&arr))
        .map_err(map_fixed_err)
}

// ASSUMPTION: `rng::seed_map_fixture() -> Vec<(String, [u8; 32])>` (3+).
/// Recorded seed map (3+ entries): computed detached, wrapped attached
/// (bytes cross as `bytes`, never as int lists).
#[pyfunction]
fn seed_map_fixture(py: Python<'_>) -> Vec<(String, Py<PyBytes>)> {
    let rows = py.detach(hydra_feed::rng::seed_map_fixture);
    rows.into_iter()
        .map(|(name, seed)| (name, PyBytes::new(py, &seed).unbind()))
        .collect()
}

/// Interpreter-safe default chunk size (PyOnceLock init pattern).
#[pyfunction]
fn default_chunk_bytes(py: Python<'_>) -> usize {
    default_chunk_bytes_impl(py)
}

/// Judge observability: `(digests, batches)` via an interpreter-safe lock.
#[pyfunction]
fn judge_stats(py: Python<'_>) -> PyResult<(u64, u64)> {
    let guard = JUDGE_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("canon judge stats mutex poisoned"))?;
    Ok((guard.digests, guard.batches))
}

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

/// Register the `canon_rng` submodule (mirrors `replay::register`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "canon_rng")?;
    sub.add_class::<CanonConfig>()?;
    sub.add_class::<StreamSpec>()?;
    sub.add_function(wrap_pyfunction!(sha256_hex, &sub)?)?;
    sub.add_function(wrap_pyfunction!(sha256_file, &sub)?)?;
    sub.add_function(wrap_pyfunction!(of_canonical_json, &sub)?)?;
    sub.add_function(wrap_pyfunction!(batch_sha256, &sub)?)?;
    sub.add_function(wrap_pyfunction!(batch_of_canonical, &sub)?)?;
    sub.add_function(wrap_pyfunction!(open_stream, &sub)?)?;
    sub.add_function(wrap_pyfunction!(philox_block, &sub)?)?;
    sub.add_function(wrap_pyfunction!(bounded, &sub)?)?;
    sub.add_function(wrap_pyfunction!(bounded_batch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_ranks, &sub)?)?;
    sub.add_function(wrap_pyfunction!(utility_fixed, &sub)?)?;
    sub.add_function(wrap_pyfunction!(utility_for_ranks_fixed, &sub)?)?;
    sub.add_function(wrap_pyfunction!(exact_total_is_zero, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seed_map_fixture, &sub)?)?;
    sub.add_function(wrap_pyfunction!(default_chunk_bytes, &sub)?)?;
    sub.add_function(wrap_pyfunction!(judge_stats, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod canon_rng_tests {
    use super::*;

    const HELLO_WORLD_SHA: &str =
        "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    #[test]
    fn hello_golden_single_and_batch_agree() {
        let single = hydra_feed::digest::sha256_hex(b"hello world");
        assert_eq!(single, HELLO_WORLD_SHA);
        let items = vec![b"hello world".to_vec(), b"hello world".to_vec()];
        let batch: Vec<String> = items
            .iter()
            .map(|b| hydra_feed::digest::sha256_hex(b))
            .collect();
        assert_eq!(batch, [single.clone(), single]);
    }

    #[test]
    fn mismatch_check_fails_closed() {
        assert!(check_match(HELLO_WORLD_SHA, HELLO_WORLD_SHA).is_ok());
        assert!(check_match(HELLO_WORLD_SHA, "sha256:00").is_err());
        assert!(check_match("sha256:aa", "sha256:bb").is_err());
    }

    #[test]
    fn frozen_configs_are_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<CanonConfig>();
        assert_sync::<StreamSpec>();
    }

    #[test]
    fn config_rejects_zero_chunk() {
        assert!(CanonConfig::new(0, 0).is_err());
        assert!(CanonConfig::new(1 << 20, 0).is_ok());
    }

    #[test]
    fn stream_spec_validates_shapes() {
        assert!(StreamSpec::new(vec![], vec![0u8; 32]).is_err());
        assert!(StreamSpec::new(vec![0u8; 15], vec![0u8; 32]).is_err());
        assert!(StreamSpec::new(vec![1u8; 14], vec![0u8; 31]).is_err());
        assert!(StreamSpec::new(b"gumbel_root_v1".to_vec(), vec![9u8; 32]).is_ok());
    }

    #[test]
    fn bounded_edges() {
        assert!(bounded(7, 0).is_err());
        assert_eq!(bounded(12345, 1).unwrap_or(99), 0);
        assert!(bounded(u32::MAX, 2).unwrap_or(99) < 2);
    }

    #[test]
    fn seed_map_has_three_or_more() {
        let rows = hydra_feed::rng::seed_map_fixture();
        assert!(rows.len() >= 3);
    }

    #[test]
    fn detach_smoke() {
        Python::attach(|py| {
            let sum = py.detach(|| 2 + 2);
            assert_eq!(sum, 4);
            let digest = py.detach(|| hydra_feed::digest::sha256_hex(b"hello world"));
            assert_eq!(digest, HELLO_WORLD_SHA);
        });
    }
    #[test]
    fn fixed_ranks_reject_tie_gap_oob() {
        use hydra_feed::fixed::{validate_ranks, FixedError};
        assert!(validate_ranks(&[1, 2, 3, 4]).is_ok());
        // Tie.
        assert!(matches!(
            validate_ranks(&[1, 1, 3, 4]),
            Err(FixedError::DuplicateRank { .. })
        ));
        // Gap (in-range but not a permutation).
        assert!(matches!(
            validate_ranks(&[1, 2, 2, 4]),
            Err(FixedError::DuplicateRank { .. })
        ));
        // OOB: 0 underflows, 5 overflows.
        assert!(matches!(
            validate_ranks(&[0, 2, 3, 4]),
            Err(FixedError::InvalidRank { index: 0, value: 0 })
        ));
        assert!(matches!(
            validate_ranks(&[1, 2, 3, 5]),
            Err(FixedError::InvalidRank { index: 3, value: 5 })
        ));
    }

    #[test]
    fn fixed_canonical_points_and_zero_sum() {
        use hydra_feed::fixed::{exact_total_fixed, utility_fixed, FIXED_SCALE};
        let out = utility_fixed([1, 2, 3, 4]).unwrap();
        assert_eq!(
            out,
            [3 * FIXED_SCALE, FIXED_SCALE, -FIXED_SCALE, -3 * FIXED_SCALE]
        );
        assert_eq!(exact_total_fixed(&out), 0);
        // Permuted ranks permute the points; the integer sum stays zero.
        let perm = utility_fixed([4, 1, 3, 2]).unwrap();
        assert_eq!(
            perm,
            [-3 * FIXED_SCALE, 3 * FIXED_SCALE, -FIXED_SCALE, FIXED_SCALE]
        );
        assert_eq!(exact_total_fixed(&perm), 0);
    }

    #[test]
    fn fixed_manifest_indexing_and_zero_sum_shapes() {
        use hydra_feed::fixed::{exact_total_is_zero, utility_for_ranks_fixed};
        let rank_values = [10i64, 20, 30, 40];
        assert_eq!(
            utility_for_ranks_fixed(&rank_values, &[4, 3, 2, 1]).unwrap(),
            [40, 30, 20, 10]
        );
        // Exact f64 zero-sum shapes (no float accumulation, no epsilon).
        assert_eq!(exact_total_is_zero(&[1.0, -1.0, 2.0, -2.0]).unwrap(), true);
        assert_eq!(exact_total_is_zero(&[1.5, 2.5, -1.0, -2.0]).unwrap(), false);
        assert_eq!(exact_total_is_zero(&[0.0, 0.0, 0.0, 0.0]).unwrap(), true);
        // m8 (F7/F8): 1e12-scale normal mixed with a subnormal — exact
        // fallback, Fraction-equivalent (`Ok(true)` post-fallback,
        // `Err(Overflow)` pre-fallback); never `Ok(false)`, never
        // `ValueError`, never assumed zero.
        match exact_total_is_zero(&[1e12, -1e12, 5e-324, -5e-324]) {
            Ok(is_zero) => assert!(is_zero, "m8 exact sum is zero"),
            Err(hydra_feed::fixed::FixedError::Overflow) => {},
            Err(other) => panic!("m8 must be Ok(true) or Overflow, got {other:?}"),
        }
    }

    #[test]
    fn fixed_err_mapping_discriminates_overflow() {
        use hydra_feed::fixed::FixedError;
        use pyo3::exceptions::{PyOverflowError, PyValueError};
        Python::attach(|py| {
            let bad_rank = map_fixed_err(FixedError::InvalidRank { index: 0, value: 0 });
            assert!(bad_rank.is_instance_of::<PyValueError>(py));
            let dup = map_fixed_err(FixedError::DuplicateRank { value: 1 });
            assert!(dup.is_instance_of::<PyValueError>(py));
            let nonfinite = map_fixed_err(FixedError::NonFiniteValue { index: 2 });
            assert!(nonfinite.is_instance_of::<PyValueError>(py));
            let overflow = map_fixed_err(FixedError::Overflow);
            assert!(overflow.is_instance_of::<PyOverflowError>(py));
        });
    }

    #[test]
    fn fixed_pyfns_shape_and_value_gates() {
        use hydra_feed::fixed::FIXED_SCALE;
        use pyo3::exceptions::PyOverflowError;
        Python::attach(|py| {
            // Rank/tie/gap/OOB through the attached shape check + detached feed.
            assert!(validate_ranks(py, vec![1, 2, 3, 4]).is_ok());
            assert!(validate_ranks(py, vec![1, 1, 3, 4]).is_err());
            assert!(validate_ranks(py, vec![0, 2, 3, 4]).is_err());
            assert!(validate_ranks(py, vec![1, 2, 3]).is_err());
            // Points [1,2,3,4] -> [3S,S,-S,-3S].
            assert_eq!(
                utility_fixed(py, vec![1, 2, 3, 4]).unwrap(),
                vec![3 * FIXED_SCALE, FIXED_SCALE, -FIXED_SCALE, -3 * FIXED_SCALE]
            );
            assert_eq!(
                utility_for_ranks_fixed(py, vec![10, 20, 30, 40], vec![4, 3, 2, 1]).unwrap(),
                vec![40, 30, 20, 10]
            );
            // Zero-sum shapes.
            assert!(exact_total_is_zero(py, vec![1.0, -1.0, 2.0, -2.0]).unwrap());
            assert!(!exact_total_is_zero(py, vec![1.5, 2.5, -1.0, -2.0]).unwrap());
            // m8 via the pyfn: Ok(true) post-fallback or Overflow pre-fallback.
            match exact_total_is_zero(py, vec![1e12, -1e12, 5e-324, -5e-324]) {
                Ok(is_zero) => assert!(is_zero, "m8 exact sum is zero"),
                Err(e) => assert!(
                    e.is_instance_of::<PyOverflowError>(py),
                    "m8 err must be Overflow, got {e}"
                ),
            }
            // below()/bounded() parity UNCHANGED: bounded(_, 1) == 0 stays.
            assert_eq!(bounded(12345, 1).unwrap_or(99), 0);
        });
    }
}
