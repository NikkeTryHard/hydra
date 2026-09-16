//! tiles: thin MJAI physical-id codec boundary over `hydra-shard::tile`.
//!
//! DAG: this module depends on the shard tile codec + pyo3 ONLY — arg-check
//! + detach + scalar return. The codec math (Tenhou order, red-five FIRST
//! copy {16, 52, 88}, plain `5x` -> second copy) lives shard-side in
//! `hydra_shard::tile::{physical_of, mjai_string_of}` (read-only reference:
//! `src/hydra2/engines/riichienv/tiles.py::physical_of/mjai_string_of`);
//! this file only validates the crossing types attached, runs the LUT codec
//! detached, and maps [`TileError`] onto `ValueError` attached.
//!
//! Output-identity notes (Wave 3 bridge round):
//! - Value-identical on the ASCII MJAI vocabulary: red aliases (`0x`),
//!   red-marked fives (`5xr`), plain fives, suits, honors, including the
//!   second-copy rule for unsuffixed `5x`. Non-ASCII input fails closed
//!   bridge-side (the oracle's `int()` accepts stray Unicode digits; the
//!   vocabulary is ASCII by contract, Wave 4 owns string validation).
//! - Fail-closed only on rejects (bridge precedent, like `eval.rs`): error
//!   TYPE is `ValueError` (vs oracle `InvalidTileError`) and error TEXT is
//!   shard-worded, not oracle-worded — observed `tile codec: invalid mjai
//!   tile "5x"` vs oracle `invalid mjai tile '5x'`: a `tile codec: ` prefix
//!   plus Rust `{:?}` double quotes where the oracle's `{text!r}` renders
//!   single-quoted. The cutover facade translates both, next round. (The
//!   out-of-range `physical tile out of range: {tile}` message has no
//!   quoting involved and matches for ints.)
//! - No byte-identity trap touched: per-tile pure maps, no accumulation, no
//!   windows, no sorts, no manifest keys.
//!
//! Concurrency pattern (matches `packet.rs` + `canon_rng.rs`): validate
//! attached, `py.detach(|| ...)` around the codec call, frozen surface only
//! (free functions; no pyclass, no shared state). GIL attestation is spelled
//! once wave-wide in `canon_rng.rs` and inherited here.
//!
//! Single-cdylib tree: this module registers as the `tiles` submodule
//! (`hydra2_replay_rs.tiles` today, `hydra_bridge._native.tiles` once the
//! Phase-6 maturin `module-name` cutover lands) via `register`, mirroring
//! `packet::register`. The legacy `hydra2_replay_rs` entry point is untouched.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// The unsuffixed `"5x"` resolves to the SECOND copy (17/53/89); the red
/// five is `"5xr"` (or the `"0x"` alias) resolving to the FIRST copy
/// (16/52/88). Anything else fails closed (`ValueError`, shard-worded —
/// see module docs on the quoting divergence).
#[pyfunction]
fn physical_of(py: Python<'_>, mjai_tile: String) -> PyResult<u8> {
    py.detach(|| hydra_shard::tile::physical_of(&mjai_tile))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Render one physical id as its canonical MJAI string (red fives marked).
///
/// Out-of-range ids fail closed (`ValueError`, oracle-worded
/// `physical tile out of range: {tile}`); floats/strings never coerce
/// (typed `int` only — the oracle renders garbage for floats, the bridge
/// rejects instead of echoing it).
#[pyfunction]
fn mjai_string_of(py: Python<'_>, tile: i64) -> PyResult<String> {
    if !(0..=135).contains(&tile) {
        return Err(PyValueError::new_err(format!(
            "physical tile out of range: {tile}"
        )));
    }
    let id = tile as u8;
    py.detach(|| hydra_shard::tile::mjai_string_of(id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Register the `tiles` submodule (mirrors `packet::register`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "tiles")?;
    sub.add_function(wrap_pyfunction!(physical_of, &sub)?)?;
    sub.add_function(wrap_pyfunction!(mjai_string_of, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn red_five_vectors_match_oracle() {
        Python::attach(|py| {
            // Red aliases and red-marked fives hit the FIRST copy.
            assert_eq!(physical_of(py, "0m".to_string()).unwrap(), 16);
            assert_eq!(physical_of(py, "0p".to_string()).unwrap(), 52);
            assert_eq!(physical_of(py, "0s".to_string()).unwrap(), 88);
            assert_eq!(physical_of(py, "5mr".to_string()).unwrap(), 16);
            assert_eq!(physical_of(py, "5pr".to_string()).unwrap(), 52);
            assert_eq!(physical_of(py, "5sr".to_string()).unwrap(), 88);
            // Plain fives skip the aka copy (oracle `mjai_to_tid`: "5s" -> 89).
            assert_eq!(physical_of(py, "5m".to_string()).unwrap(), 17);
            assert_eq!(physical_of(py, "5p".to_string()).unwrap(), 53);
            assert_eq!(physical_of(py, "5s".to_string()).unwrap(), 89);
            // Suits are sequential four-copy blocks; honors follow from 108.
            assert_eq!(physical_of(py, "1m".to_string()).unwrap(), 0);
            assert_eq!(physical_of(py, "9s".to_string()).unwrap(), 72 + 4 * 8);
            assert_eq!(physical_of(py, "E".to_string()).unwrap(), 108);
            assert_eq!(physical_of(py, "C".to_string()).unwrap(), 132);
            // Canonical rendering marks red fives only.
            assert_eq!(mjai_string_of(py, 16).unwrap(), "5mr");
            assert_eq!(mjai_string_of(py, 52).unwrap(), "5pr");
            assert_eq!(mjai_string_of(py, 88).unwrap(), "5sr");
            assert_eq!(mjai_string_of(py, 17).unwrap(), "5m");
            assert_eq!(mjai_string_of(py, 0).unwrap(), "1m");
            assert_eq!(mjai_string_of(py, 135).unwrap(), "C");
            // Fail-closed: unknown suits, red-on-non-five, range.
            assert!(physical_of(py, "5x".to_string()).is_err());
            assert!(physical_of(py, "".to_string()).is_err());
            assert!(physical_of(py, "10m".to_string()).is_err());
            assert!(physical_of(py, "6mr".to_string()).is_err());
            assert!(mjai_string_of(py, 136).is_err());
            assert!(mjai_string_of(py, -1).is_err());
        });
    }

    #[test]
    fn canonical_representatives_roundtrip() {
        Python::attach(|py| {
            // Canonical ids: first copy of every non-five block, both five
            // representatives (aka + plain), first copy of every honor.
            let mut canonical: Vec<i64> = Vec::with_capacity(37);
            for base in [0i64, 36, 72] {
                for rank in 1..=9i64 {
                    if rank == 5 {
                        canonical.push(base + 16);
                        canonical.push(base + 17);
                    } else {
                        canonical.push(base + 4 * (rank - 1));
                    }
                }
            }
            for base in [108i64, 112, 116, 120, 124, 128, 132] {
                canonical.push(base);
            }
            assert_eq!(canonical.len(), 37);
            for tile in canonical {
                let rendered = mjai_string_of(py, tile).unwrap();
                let back = physical_of(py, rendered.clone()).unwrap();
                assert_eq!(back as i64, tile, "roundtrip failed via {rendered}");
            }
            // Many-to-one rendering collapses to the canonical copy.
            assert_eq!(mjai_string_of(py, 18).unwrap(), "5m");
            assert_eq!(physical_of(py, "5m".to_string()).unwrap(), 17);
            assert_eq!(mjai_string_of(py, 1).unwrap(), "1m");
            assert_eq!(physical_of(py, "1m".to_string()).unwrap(), 0);
        });
    }
}
