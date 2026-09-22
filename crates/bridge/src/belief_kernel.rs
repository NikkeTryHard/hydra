//! belief_kernel: WP-07A packet-kernel frozen tables + scalar validators on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-search` ONLY (the `belief` owner for the successor-ref
//! core and the packet-layout consts — same edge as `search.rs`, no new
//! dependency). FORBIDS: JCS/sha reimplementation (hashing routes through the
//! `hydra_search::belief` owner, which itself folds through the feed `canon`
//! owner), wall-clock/RNG (draws stay caller-side), live-object orchestration
//! (`PacketSuccessor`/`EventEnvelope`/`ActorVisiblePacket` construction,
//! bridge==oracle packet assembly, and the mass/disjointness post-conditions
//! stay Python), and `ContractError` shaping (the bridge raises `ValueError`;
//! thin Python translators map to `ContractError` with byte-identical oracle
//! text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/kernel.py`:
//! - consts `KERNEL_TILE_BASE` / `KERNEL_SEQ_BASE` / `KERNEL_SUCCESSORS` /
//!   `KERNEL_PROB` / `KERNEL_MASS_TOL` / `KERNEL_GAME_ID` /
//!   `KERNEL_SCHEMA_HASH` <- kernel literals (`kernel.py:64,165,197-212,225`)
//!   via the `hydra_search::belief` owner (`PACKET_TILE_BASE`,
//!   `PACKET_SEQ_BASE`, `PACKET_SUCCESSORS`, `PACKET_PROB`,
//!   `PACKET_MASS_TOL`, `PACKET_GAME_ID`, `PACKET_SCHEMA_HASH` — referenced,
//!   never restated).
//! - consts `KERNEL_LOG_PHYSICAL` / `KERNEL_LOG_POLICY` / `KERNEL_PROB_TOL` /
//!   `KERNEL_TOL_UPPER` / `KERNEL_U32_MAX` <- split/tolerance literals
//!   (`kernel.py:120-126,179-180,287-290`); the log is computed from
//!   `0.5f64.ln()`, never hardcoded.
//! - `kernel_successor_refs` <- `_cached_successor_refs` (`kernel.py:97-114`):
//!   owned scalars cross (`parent_ref` + `tile` + `action_id`, never live
//!   `Particle`s — the `PacketSuccessor` precedent at
//!   `contracts.rs:259-261`); the sha core is the feed-descended
//!   `hydra_search::belief::successor_refs` owner (no reimplementation).
//! - `kernel_successor_layout` <- bridge-layout expectation
//!   (`kernel.py:206-217`): `tile = 8 + idx`, `seq = 100 + idx`,
//!   `actor = (root + 1 + idx) % 4` over plain `u32`s.
//! - `kernel_check_tolerance` <- `NaturalPacketKernel.__init__`
//!   (`kernel.py:120-126`): strict-`float` gate in `(0, 0.01)` with the
//!   oracle-identical message.
//! - `kernel_check_action_id` / `kernel_check_root_actor` <- input gates
//!   (`kernel.py:171-180`): `u32` (bool excluded) / seat `0..=3`.
//!
//! LOGIC staying Python: the `PacketSuccessor` dataclass, `_valid_digest`,
//! `_make_public_discard_event` (live `EventPayload`/`EventEnvelope`
//! construction), the `enumerate_next` orchestration (stale checks, action
//! normalization, `search.packet_successors` fan-out, observation/chain
//! packet assembly with the `hashlib` oracle cross-checks, probability
//! recombination + mass/disjointness post-conditions), every `__all__`.
//!
//! NONE-owner note: `rg` for `PACKET_TILE_BASE|PACKET_SEQ_BASE|
//! PACKET_SUCCESSORS|PACKET_PROB|PACKET_MASS_TOL|PACKET_GAME_ID|
//! PACKET_SCHEMA_HASH` over `crates/bridge/src` hits nothing — the search
//! crate owns the values but no bridge fn exposes them on `contracts` yet;
//! `rg` for `kernel_successor|KERNEL_` over `crates/bridge/src` is likewise
//! empty before this file.
//!
//! Shape per fn: attached staging (bool-rejection + range checks touch Python
//! memory, never detached) -> ONE `py.detach(|| ...)` over owned plain data
//! with zero Python API inside for the sha leaf (per `contracts.rs:434-437`;
//! the sub-microsecond integer/float gates run attached end-to-end per the
//! `contracts.rs:10-13` frozen-census note) -> attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Consts via `sub.add` (per
//! `contracts.rs:1153-1156`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `belief_leaves.rs:192-229`); no new entry
//! point. MAIN wiring: `pub mod belief_kernel;` in `lib.rs` plus
//! `crate::belief_kernel::register(&sub)?;` in `contracts.rs` next to the
//! `crate::belief_leaves::register(&sub)?;` line.

use hydra_search::belief as belief_mod;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyModule};

/// First discard tile (`kernel.py:210`: `tile = 8 + idx`), via the owner.
const KERNEL_TILE_BASE: u32 = belief_mod::PACKET_TILE_BASE;
/// First packet sequence (`kernel.py:211`: `seq = 100 + idx`), via the owner.
const KERNEL_SEQ_BASE: u32 = belief_mod::PACKET_SEQ_BASE;
/// Successors per `(parent, action)` (`kernel.py:195-196`: exactly 2), via the owner.
const KERNEL_SUCCESSORS: u32 = belief_mod::PACKET_SUCCESSORS;
/// Per-successor mass (`kernel.py:199-201`: uniform `0.5` each), via the owner.
const KERNEL_PROB: f64 = belief_mod::PACKET_PROB;
/// Packet mass gate (`NaturalPacketKernel` default `kernel_tolerance = 1e-9`), via the owner.
const KERNEL_MASS_TOL: f64 = belief_mod::PACKET_MASS_TOL;
/// Synthetic game id (`_make_public_discard_event` default), via the owner.
const KERNEL_GAME_ID: &str = belief_mod::PACKET_GAME_ID;
/// Synthetic schema hash (`kernel.py:165`: `"sha256:" + "c" * 64`), via the owner.
const KERNEL_SCHEMA_HASH: &str = belief_mod::PACKET_SCHEMA_HASH;
/// Exclusive tolerance upper bound (`kernel.py:124`: `>= 0.01` rejects).
const KERNEL_TOL_UPPER: f64 = 0.01;
/// Bridge probability/log equality gate (`kernel.py:287-289`: `1e-12`).
const KERNEL_PROB_TOL: f64 = 1e-12;
/// Action-id ceiling (`kernel.py:179`: `aid > 0xFFFF_FFFF` rejects).
const KERNEL_U32_MAX: u32 = 0xFFFF_FFFF;
/// Deterministic-opponent log policy (`kernel.py:200`: `log_policy = 0.0`).
const KERNEL_LOG_POLICY: f64 = 0.0;

/// Physical log mass (`kernel.py:201`: `log_phys = log(0.5)`): computed from
/// `0.5f64.ln()` so the value tracks the owner (`belief::enumerate_next`
/// uses the same expression) instead of a hardcoded literal.
fn kernel_log_physical() -> f64 {
    0.5f64.ln()
}

/// Attached staging helper: plain `u32` view of a Python value with `bool`
/// excluded first (it subclasses `int`), mirroring
/// `contracts.rs:69-74` (`plain_int_value`) and `search_profiles.rs:42-46`
/// (`as_u64`). Non-`u32` values map to `PyValueError`, never a default.
fn checked_u32(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<u32> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "{name} must be u32, not bool"
        )));
    }
    obj.extract::<u32>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be u32")))
}

/// Pure layout core: `tile = 8 + idx`, `seq = 100 + idx`,
/// `actor = (root + 1 + idx) % 4` (mirrors `kernel.py:210-212` for the
/// reachable domain `idx < 2`, `root < 4`). No Python API inside
/// (detach-safe); failures are marker `&str`s the pyfn maps to `ValueError`.
fn layout_core(index: u32, root: u32) -> Result<(u32, u32, u32), &'static str> {
    if index >= KERNEL_SUCCESSORS {
        return Err("index");
    }
    if root >= 4 {
        return Err("root");
    }
    Ok((
        KERNEL_TILE_BASE + index,
        KERNEL_SEQ_BASE + index,
        (root + 1 + index) % 4,
    ))
}

/// Pure tolerance core: strict `(0, 0.01)` gate (mirrors
/// `kernel.py:121-126`). Non-finite values reject here (TOTAL: NaN slips the
/// oracle's `<` comparisons, so the bridge closes the gap with the same
/// oracle-identical message the pyfn raises).
fn tolerance_core(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() || value <= 0.0 || value >= KERNEL_TOL_UPPER {
        return Err("tolerance");
    }
    Ok(value)
}

/// Successor/delta refs over owned scalars (mirrors `_cached_successor_refs`
/// without live `Particle` objects — callers pass scalars, the
/// `PacketSuccessor` precedent at `contracts.rs:259-261`). The sha core is
/// the `hydra_search::belief::successor_refs` owner (the
/// `packet_successors` precedent at `search.rs:519-520` borrows owned
/// `String`s inside the detach the same way). Compute runs detached with
/// zero Python API inside; empty parents and non-`u32` tiles/actions are
/// `PyValueError`, never a default. Counter-free deterministic, never
/// wall-clock.
#[pyfunction]
#[pyo3(signature = (parent_ref, tile, action_id))]
fn kernel_successor_refs(
    py: Python<'_>,
    parent_ref: String,
    tile: Bound<'_, PyAny>,
    action_id: Bound<'_, PyAny>,
) -> PyResult<(String, String)> {
    if parent_ref.is_empty() {
        return Err(PyValueError::new_err(
            "kernel parent_ref must be non-empty str",
        ));
    }
    let tile_v = checked_u32(&tile, "kernel tile")?;
    let action_v = checked_u32(&action_id, "kernel action_id")?;
    let out = py.detach(|| belief_mod::successor_refs(&parent_ref, tile_v, action_v));
    Ok(out)
}

/// Expected bridge-layout triple for successor `index` under `root_actor`
/// (mirrors `kernel.py:210-212` without the live bridge rows — callers pass
/// plain ints, triples come back). `index` must be below `KERNEL_SUCCESSORS`,
/// `root_actor` a seat `0..=3` (bool excluded); violations are `PyValueError`.
/// Sub-microsecond integer math runs attached end-to-end (per the
/// `contracts.rs:10-13` frozen-census note — cheaper than a GIL round-trip).
#[pyfunction]
#[pyo3(signature = (index, root_actor))]
fn kernel_successor_layout(
    index: Bound<'_, PyAny>,
    root_actor: Bound<'_, PyAny>,
) -> PyResult<(u32, u32, u32)> {
    let idx = checked_u32(&index, "kernel index")?;
    let root = checked_u32(&root_actor, "kernel root_actor")?;
    layout_core(idx, root).map_err(|slot| {
        if slot == "index" {
            PyValueError::new_err(format!(
                "kernel index must be below {}, got {idx}",
                KERNEL_SUCCESSORS
            ))
        } else {
            PyValueError::new_err(format!("kernel root_actor must be 0..3, got {root}"))
        }
    })
}

/// Validated kernel tolerance (mirrors `NaturalPacketKernel.__init__`
/// bit-for-bit: only a `float` in `(0, 0.01)` passes — `int`/`bool` reject
/// via the strict-`float` gate, the `search_profiles.rs:288-299`
/// `is_instance_of::<PyFloat>` precedent). The message is the oracle text
/// (`kernel.py:126`), so the Python translator re-raises it byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn kernel_check_tolerance(value: Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.is_instance_of::<PyFloat>() {
        return Err(PyValueError::new_err(
            "kernel_tolerance must be small positive float",
        ));
    }
    let staged: f64 = value
        .extract()
        .map_err(|_| PyValueError::new_err("kernel_tolerance must be small positive float"))?;
    tolerance_core(staged)
        .map_err(|_| PyValueError::new_err("kernel_tolerance must be small positive float"))
}

/// Validated action id (mirrors the `kernel.py:179` `u32` gate: `bool`
/// excluded, `0..=0xFFFF_FFFF`). The Python translator formats the
/// oracle-identical `got {aid!r}` text (Rust never renders Python reprs).
#[pyfunction]
#[pyo3(signature = (value,))]
fn kernel_check_action_id(value: Bound<'_, PyAny>) -> PyResult<u32> {
    checked_u32(&value, "kernel action_id")
}

/// Validated root seat (mirrors the `kernel.py:177-178` `0..=3` gate after
/// the caller-side `int()` narrowing). The Python translator formats the
/// oracle-identical `got {root!r}` text.
#[pyfunction]
#[pyo3(signature = (value,))]
fn kernel_check_root_actor(value: Bound<'_, PyAny>) -> PyResult<u32> {
    let staged = checked_u32(&value, "kernel root_actor")?;
    if staged >= 4 {
        return Err(PyValueError::new_err("kernel root_actor must be 0..3"));
    }
    Ok(staged)
}

/// Register the kernel tables + validators on the shared `contracts`
/// submodule (mirrors `belief_leaves.rs:192-229`): `let py = sub.py();`
/// (per `contracts.rs:1115`), fns via `wrap_pyfunction!(f, sub)` (per
/// `belief_leaves.rs:194`), consts via `sub.add` (per
/// `contracts.rs:1153-1156`); single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(kernel_successor_refs, sub)?)?;
    sub.add_function(wrap_pyfunction!(kernel_successor_layout, sub)?)?;
    sub.add_function(wrap_pyfunction!(kernel_check_tolerance, sub)?)?;
    sub.add_function(wrap_pyfunction!(kernel_check_action_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(kernel_check_root_actor, sub)?)?;
    sub.add("KERNEL_TILE_BASE", KERNEL_TILE_BASE)?;
    sub.add("KERNEL_SEQ_BASE", KERNEL_SEQ_BASE)?;
    sub.add("KERNEL_SUCCESSORS", KERNEL_SUCCESSORS)?;
    sub.add("KERNEL_PROB", KERNEL_PROB)?;
    sub.add("KERNEL_LOG_PHYSICAL", kernel_log_physical())?;
    sub.add("KERNEL_LOG_POLICY", KERNEL_LOG_POLICY)?;
    sub.add("KERNEL_MASS_TOL", KERNEL_MASS_TOL)?;
    sub.add("KERNEL_PROB_TOL", KERNEL_PROB_TOL)?;
    sub.add("KERNEL_TOL_UPPER", KERNEL_TOL_UPPER)?;
    sub.add("KERNEL_U32_MAX", KERNEL_U32_MAX)?;
    sub.add("KERNEL_GAME_ID", KERNEL_GAME_ID)?;
    sub.add("KERNEL_SCHEMA_HASH", KERNEL_SCHEMA_HASH)?;
    // Touch `py` the way const-tuple leaves do (per `contracts.rs:1156`):
    // keeps the handle live for the tuple-free scalar adds above.
    let _ = py;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals_and_owner() {
        // Hand-derived from the oracle literals (`kernel.py:64,124,165,
        // 179,197-212,287-290`); any drift fails here first. Owner pins
        // prove the references (never restated literals).
        assert_eq!(KERNEL_TILE_BASE, 8);
        assert_eq!(KERNEL_SEQ_BASE, 100);
        assert_eq!(KERNEL_SUCCESSORS, 2);
        assert_eq!(KERNEL_PROB, 0.5);
        assert_eq!(KERNEL_MASS_TOL, 1e-9);
        assert_eq!(KERNEL_GAME_ID, "game_tiny_001");
        assert_eq!(
            KERNEL_SCHEMA_HASH,
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        );
        assert_eq!(KERNEL_TOL_UPPER, 0.01);
        assert_eq!(KERNEL_PROB_TOL, 1e-12);
        assert_eq!(KERNEL_U32_MAX, 0xFFFF_FFFF);
        assert_eq!(KERNEL_LOG_POLICY, 0.0);
        assert_eq!(KERNEL_TILE_BASE, belief_mod::PACKET_TILE_BASE);
        assert_eq!(KERNEL_SEQ_BASE, belief_mod::PACKET_SEQ_BASE);
        assert_eq!(KERNEL_SUCCESSORS, belief_mod::PACKET_SUCCESSORS);
        assert_eq!(KERNEL_PROB, belief_mod::PACKET_PROB);
        assert_eq!(KERNEL_MASS_TOL, belief_mod::PACKET_MASS_TOL);
        assert_eq!(KERNEL_GAME_ID, belief_mod::PACKET_GAME_ID);
        assert_eq!(KERNEL_SCHEMA_HASH, belief_mod::PACKET_SCHEMA_HASH);
    }

    #[test]
    fn log_physical_matches_oracle_split() {
        // `kernel.py:201,289`: `log_phys = log(0.5)`; the oracle prints
        // `-0.6931471805599453` (verified via `math.log(0.5)` this session).
        let got = kernel_log_physical();
        assert_eq!(got, 0.5f64.ln());
        assert!((got - (-std::f64::consts::LN_2)).abs() < 1e-15);
    }

    #[test]
    fn layout_core_matches_oracle_expectation() {
        // Hand-derived from `kernel.py:210-212` (`exp_tile = 8 + idx`,
        // `exp_seq = 100 + idx`, `exp_actor = (root + 1 + idx) % 4`).
        assert_eq!(layout_core(0, 0), Ok((8, 100, 1)));
        assert_eq!(layout_core(1, 0), Ok((9, 101, 2)));
        assert_eq!(layout_core(0, 3), Ok((8, 100, 0)));
        assert_eq!(layout_core(1, 3), Ok((9, 101, 1)));
        assert_eq!(layout_core(0, 2), Ok((8, 100, 3)));
        assert!(layout_core(2, 0).is_err());
        assert!(layout_core(0, 4).is_err());
    }

    #[test]
    fn tolerance_core_matches_init_gate() {
        // Hand-derived from `kernel.py:121-126` (`float` in `(0, 0.01)`),
        // plus the TOTAL gap closure (NaN/inf reject with the same text).
        assert_eq!(tolerance_core(1e-9), Ok(1e-9));
        assert_eq!(tolerance_core(0.009), Ok(0.009));
        assert!(tolerance_core(0.0).is_err());
        assert!(tolerance_core(-1e-9).is_err());
        assert!(tolerance_core(0.01).is_err());
        assert!(tolerance_core(1.0).is_err());
        assert!(tolerance_core(f64::NAN).is_err());
        assert!(tolerance_core(f64::INFINITY).is_err());
    }

    #[test]
    fn successor_refs_match_hashlib_oracle_vectors() {
        // Hand-derived via the oracle formula (`hashlib.sha256(
        // f"{parent}:{tile}:{aid}").hexdigest()[:16]` with `world_succ:` /
        // `delta:` prefixes, computed with CPython hashlib this session).
        // The delta input omits `aid`, so aid changes succ but not delta.
        let (succ, delta) = belief_mod::successor_refs("p", 8, 0);
        assert_eq!(succ, "world_succ:c311be28bcd7abab");
        assert_eq!(delta, "delta:0bdd08db5ece6fa6");
        let (succ9, delta9) = belief_mod::successor_refs("p", 9, 0);
        assert_eq!(succ9, "world_succ:e45b8078306f1952");
        assert_eq!(delta9, "delta:ed4d4ea648017679");
        let (succ_a1, delta_a1) = belief_mod::successor_refs("p", 8, 1);
        assert_eq!(succ_a1, "world_succ:1a8ecb3be5bbaba0");
        assert_eq!(delta_a1, delta, "delta must ignore aid");
        let (succ_w, delta_w) = belief_mod::successor_refs("world:test:001", 8, 0);
        assert_eq!(succ_w, "world_succ:9a6750aa46453a1e");
        assert_eq!(delta_w, "delta:048db32da6854c91");
        // Shape: prefixes + 16 hex chars; determinism across calls.
        for text in [&succ, &delta, &succ9, &delta9] {
            let body = text.split(':').nth(1).unwrap();
            assert_eq!(body.len(), 16);
            assert!(body.bytes().all(|c| c.is_ascii_hexdigit()));
        }
        assert_eq!(belief_mod::successor_refs("p", 8, 0), (succ, delta));
    }
}
