//! rl/ffi: batch-only PyO3 boundary for the native RL subtree.
//!
//! DAG: this module depends on `rl::wall` + pyo3 ONLY (pure bytes/math in
//! `wall.rs`, no feed/shard/search owner — the continuation law lives here
//! alone; the semantic CTR bytes come from `hydra_feed::digest` through
//! `wall::ctr_u53_window`). FORBIDS: per-row Python callbacks, wall-clock,
//! RNG seeding, live-object orchestration.
//!
//! Concurrency pattern (matches `walls.rs`/`contracts::ctr_block`):
//! validate attached, compute under ONE `py.detach(|| ...)` with zero
//! Python API inside, wrap attached. Fail-closed: every shape violation is
//! `ValueError`, never a default wall or a fallback stream.
//!
//! Single-cdylib tree: registers onto the `hydra2._native.rl` submodule
//! created by `rl::register` — no new top-level submodule.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

/// Frozen wall diagnostic: every vector a differential oracle needs to pin
/// the port, built from INDEPENDENT fresh generators so diagnostic draws
/// never perturb the shuffle transcript.
#[pyclass(name = "WallDiagnostic", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct WallDiagnostic {
    /// Continuation seed (`SHA first-8 BE` of the named key).
    pub seed: u64,
    /// SeedSequence pool (`[u32; 4]`).
    pub pool: Vec<u32>,
    /// `generate_state(4, uint64)` words.
    pub gen_state: Vec<u64>,
    /// Post-`srandom` `(state_hi, state_lo, inc_hi, inc_lo)`.
    pub pcg_state: Vec<u64>,
    /// First 6 `next_u64` draws (independent generator).
    pub raw_u64: Vec<u64>,
    /// First 8 `next_u32` draws, low-half-first (independent generator).
    pub raw_u32: Vec<u32>,
    /// Final 136-tile permutation bytes.
    pub wall: Vec<u8>,
}

#[pymethods]
impl WallDiagnostic {
    /// Continuation seed (`SHA first-8 BE` of the named key).
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// SeedSequence pool (`[u32; 4]`).
    #[getter]
    fn pool(&self) -> Vec<u32> {
        self.pool.clone()
    }

    /// `generate_state(4, uint64)` words.
    #[getter]
    fn gen_state(&self) -> Vec<u64> {
        self.gen_state.clone()
    }

    /// Post-`srandom` `(state_hi, state_lo, inc_hi, inc_lo)`.
    #[getter]
    fn pcg_state(&self) -> Vec<u64> {
        self.pcg_state.clone()
    }

    /// First 6 `next_u64` draws (independent generator).
    #[getter]
    fn raw_u64(&self) -> Vec<u64> {
        self.raw_u64.clone()
    }

    /// First 8 `next_u32` draws, low-half-first (independent generator).
    #[getter]
    fn raw_u32(&self) -> Vec<u32> {
        self.raw_u32.clone()
    }

    /// Final 136-tile permutation bytes.
    #[getter]
    fn wall(&self) -> Vec<u8> {
        self.wall.clone()
    }
}

/// Batch continuation walls for `hands` (every entry `>= 1`; hand 0 is the
/// supplied schedule wall and fails closed naming its item index). `digest`
/// accepts the canonical `sha256:<hex>` text or the bare hex (mirrors
/// `removeprefix`). One `detach` around the whole batch; walls return as
/// `bytes` (136 physical ids each).
#[pyfunction]
#[pyo3(signature = (*, schedule_id, digest, hands))]
fn derive_hand_walls(
    py: Python<'_>,
    schedule_id: &str,
    digest: &str,
    hands: Vec<u64>,
) -> PyResult<Vec<Py<PyBytes>>> {
    let mut idx = 0usize;
    for hand in &hands {
        if *hand == 0 {
            return Err(PyValueError::new_err(format!(
                "rl derive_hand_walls: hand 0 is the supplied schedule wall (item {idx})"
            )));
        }
        idx += 1;
    }
    let out = py
        .detach(|| -> Result<Vec<Vec<u8>>, String> {
            let noprefix = digest.strip_prefix("sha256:").unwrap_or(digest);
            let mut acc: Vec<Vec<u8>> = Vec::with_capacity(hands.len());
            for hand in &hands {
                acc.push(
                    crate::rl::wall::derive_hand_wall(schedule_id, noprefix, *hand)
                        .map(|w| w.to_vec())
                        .map_err(|e| e.to_string())?,
                );
            }
            Ok(acc)
        })
        .map_err(PyValueError::new_err)?;
    let mut wrapped: Vec<Py<PyBytes>> = Vec::with_capacity(out.len());
    for wall in &out {
        wrapped.push(PyBytes::new(py, wall).unbind());
    }
    Ok(wrapped)
}

/// Batch engine seeds for `hands` (any ordinal incl. 0; separate `rng:`
/// scope, never a wall input). Same `digest` spelling as
/// [`derive_hand_walls`]. One `detach` around the whole batch.
#[pyfunction]
#[pyo3(signature = (*, schedule_id, digest, hands))]
fn hand_rng_seeds(
    py: Python<'_>,
    schedule_id: &str,
    digest: &str,
    hands: Vec<u64>,
) -> PyResult<Vec<u64>> {
    Ok(py.detach(|| {
        let noprefix = digest.strip_prefix("sha256:").unwrap_or(digest);
        let mut acc: Vec<u64> = Vec::with_capacity(hands.len());
        for hand in &hands {
            acc.push(crate::rl::wall::hand_rng_seed(schedule_id, noprefix, *hand));
        }
        acc
    }))
}

/// Full diagnostic for one derived hand (`>= 1`): seed, pool, generated
/// state, post-seed PCG words, raw transcripts (independent generators),
/// and the final wall. Hand 0 fails closed (no schedule tiles here).
#[pyfunction]
#[pyo3(signature = (*, schedule_id, digest, hand))]
fn wall_diagnostic(
    py: Python<'_>,
    schedule_id: &str,
    digest: &str,
    hand: u64,
) -> PyResult<WallDiagnostic> {
    if hand == 0 {
        return Err(PyValueError::new_err(
            "rl wall_diagnostic: hand 0 is the supplied schedule wall".to_string(),
        ));
    }
    Ok(py.detach(|| {
        let noprefix = digest.strip_prefix("sha256:").unwrap_or(digest);
        let seed = crate::rl::wall::continuation_seed(schedule_id, noprefix, hand);
        let pool = crate::rl::wall::seedsequence_pool(seed);
        let state64 = crate::rl::wall::generate_state_u64(pool);
        let words = [
            ((state64[0] as u128) << 64) | (state64[1] as u128),
            ((state64[2] as u128) << 64) | (state64[3] as u128),
        ];
        let seeded = crate::rl::wall::Pcg64::srandom(words[0], words[1]);
        let (shi, slo, ihi, ilo) = seeded.state_words();
        let mut raw_gen = crate::rl::wall::Pcg64::from_seed(seed);
        let mut raw_u64 = Vec::with_capacity(6);
        let mut n = 0usize;
        while n < 6 {
            raw_u64.push(raw_gen.next_u64());
            n += 1;
        }
        let mut raw32_gen = crate::rl::wall::Pcg64::from_seed(seed);
        let mut raw_u32 = Vec::with_capacity(8);
        n = 0;
        while n < 8 {
            raw_u32.push(raw32_gen.next_u32());
            n += 1;
        }
        let mut shuffle_gen = crate::rl::wall::Pcg64::from_seed(seed);
        let wall = crate::rl::wall::shuffle136(&mut shuffle_gen);
        WallDiagnostic {
            seed,
            pool: pool.to_vec(),
            gen_state: state64.to_vec(),
            pcg_state: [shi, slo, ihi, ilo].to_vec(),
            raw_u64,
            raw_u32,
            wall: wall.to_vec(),
        }
    }))
}

/// Batched u53 reserves over the native semantic counter stream: draw `k`
/// reads 8 bytes at `[cursor + 8k, ...)` through `(word >> 11) * 2^-53`
/// (mirrors the actor-sampling endpoint law; cursor advance stays
/// caller-owned). Empty seeds and overflowing windows fail closed. One
/// `detach` around the whole batch.
#[pyfunction]
#[pyo3(signature = (*, seed, cursor, draws))]
fn ctr_u53_batch(
    py: Python<'_>,
    seed: Vec<u8>,
    cursor: u64,
    draws: usize,
) -> PyResult<Vec<f64>> {
    if seed.is_empty() {
        return Err(PyValueError::new_err(
            "rl ctr_u53_batch: seed must be nonempty bytes".to_string(),
        ));
    }
    py.detach(|| crate::rl::wall::ctr_u53_window(&seed, cursor, draws))
        .map_err(|e| PyValueError::new_err(format!("rl ctr_u53_batch rejected: {e}")))
}

/// Register the `rl` batch surface onto the already-created `rl`
/// submodule (mirrors `walls::register` onto `contracts`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_class::<WallDiagnostic>()?;
    sub.add_function(wrap_pyfunction!(derive_hand_walls, sub)?)?;
    sub.add_function(wrap_pyfunction!(hand_rng_seeds, sub)?)?;
    sub.add_function(wrap_pyfunction!(wall_diagnostic, sub)?)?;
    sub.add_function(wrap_pyfunction!(ctr_u53_batch, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rejects_hand_zero_naming_index() {
        Python::initialize();
        Python::attach(|py| {
            let err = derive_hand_walls(py, "s", "ab", vec![1, 0, 2]).unwrap_err();
            assert!(err.to_string().contains("item 1"));
            // Prefixed and bare digests agree (removeprefix mirror).
            let digest =
                "75c313f2c3f348c33075c056e3f011a274a03579675cc08591a64d733e6e09a1";
            let prefixed = format!("sha256:{digest}");
            let a = derive_hand_walls(py, "sched-wave3-001", &prefixed, vec![1]).unwrap();
            let b = derive_hand_walls(py, "sched-wave3-001", digest, vec![1]).unwrap();
            assert_eq!(a.len(), 1);
            assert_eq!(b.len(), 1);
            assert_eq!(a[0].bind(py).as_bytes(), b[0].bind(py).as_bytes());
            assert_eq!(a[0].bind(py).as_bytes().len(), crate::rl::wall::WALL_TILES);
            // Engine seeds batch through with the rng: scope.
            let seeds = hand_rng_seeds(py, "sched-wave3-001", digest, vec![0, 1]).unwrap();
            assert_eq!(seeds[0], 13683845269432369280);
            assert_eq!(seeds[1], 9583203488292165949);
            // u53 batch rides the semantic CTR bytes (cursor-0 golden bits).
            let raw: [u8; 32] = [
                0x75, 0xc3, 0x13, 0xf2, 0xc3, 0xf3, 0x48, 0xc3, 0x30, 0x75,
                0xc0, 0x56, 0xe3, 0xf0, 0x11, 0xa2, 0x74, 0xa0, 0x35, 0x79,
                0x67, 0x5c, 0xc0, 0x85, 0x91, 0xa6, 0x4d, 0x73, 0x3e, 0x6e,
                0x09, 0xa1,
            ];
            let u = ctr_u53_batch(py, raw.to_vec(), 0, 1).unwrap();
            assert_eq!(u.len(), 1);
            assert_eq!(u[0].to_bits(), 0x3fa89ef25c942a80);
            assert!(ctr_u53_batch(py, Vec::new(), 0, 1).is_err());
            // Diagnostic pins the whole chain for hand 1 end to end.
            let d = wall_diagnostic(py, "sched-wave3-001", digest, 1).unwrap();
            assert_eq!(d.seed, 4707380933110928792);
            assert_eq!(d.pool, [0x5ed67fd1, 0x2d83bbc7, 0xba5e91d0, 0xf0de49f1]);
            assert_eq!(d.wall.len(), crate::rl::wall::WALL_TILES);
            assert!(wall_diagnostic(py, "sched-wave3-001", digest, 0).is_err());
        });
    }
}
