//! event_packet: frozen SPEC 7.2 packet-boundary values.
//!
//! DAG: pyo3 ONLY (frozen literals; no feed dependency — packet identity
//! digests ride `contracts::packet_id_from_doc`, chain folds ride
//! `contracts::public_chain_hash` / `fold_public_hash`, span checks ride
//! `contracts::validate_packet_spans`; canon bytes stay Python-side, see
//! `contracts` rank 6 docs). FORBIDS: digest computation (needs live
//! `ActorVisiblePacket`/`EventEnvelope` objects plus the Python canon
//! authority), file/JSON IO (OS authority stays Python), and
//! partition/chain orchestration (live-object roots stay Python).
//!
//! Single-cdylib tree: registers its consts on the shared
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `contracts::register`); no new entry point. MAIN wires this with one
//! line inside `contracts::register` (see port report).

// Precedent `use pyo3::prelude::*;` -> contracts.rs:52.
use pyo3::prelude::*;
// Precedent `PyModule` + `PyTuple` imports -> contracts.rs:53.
use pyo3::types::{PyModule, PyTuple};

/// Published artifact type (`event_packet.py:82`).
const PACKET_BOUNDARY_ARTIFACT_TYPE: &str = "hydra2.packet_boundary";

/// Published schema version (`event_packet.py:83`).
const PACKET_BOUNDARY_SCHEMA_VERSION: &str = "1.0.0";

/// Artifact relpath text (`event_packet.py:84`): `Path("configs") /
/// "contracts" / "packet_boundary_v1.json"` joined (POSIX); Python re-wraps
/// with `Path(...)` to keep the `Path` type.
const PACKET_BOUNDARY_RELPATH: &str = "configs/contracts/packet_boundary_v1.json";

/// Frozen update boundary kinds (`event_packet.py:86-98`).
const PACKET_UPDATE_BOUNDARY_KINDS: [&str; 11] = [
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
];

/// Frozen terminal boundary kinds (`event_packet.py:99-104`).
const PACKET_TERMINAL_BOUNDARY_KINDS: [&str; 4] =
    ["round_end", "abortive_draw", "draw_end", "game_end"];

/// Register the frozen packet-boundary tables on the shared `contracts`
/// submodule (mirrors `contracts::register`; single cdylib, no new entry).
///
/// Precedents: `register` signature shape -> action_artifact.rs:42;
/// `let py = sub.py()` -> action_artifact.rs:43;
/// `sub.add(<str>, <&str>)` -> contracts.rs:1153;
/// `PyTuple::new(py, [...])` for frozen tuples -> contracts.rs:1156.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    // `event_packet.py:82`.
    sub.add(
        "PACKET_BOUNDARY_ARTIFACT_TYPE",
        PACKET_BOUNDARY_ARTIFACT_TYPE,
    )?;
    // `event_packet.py:83`.
    sub.add(
        "PACKET_BOUNDARY_SCHEMA_VERSION",
        PACKET_BOUNDARY_SCHEMA_VERSION,
    )?;
    // `event_packet.py:84`.
    sub.add("PACKET_BOUNDARY_RELPATH", PACKET_BOUNDARY_RELPATH)?;
    // `event_packet.py:86-98`.
    sub.add(
        "PACKET_UPDATE_BOUNDARY_KINDS",
        PyTuple::new(py, PACKET_UPDATE_BOUNDARY_KINDS)?,
    )?;
    // `event_packet.py:99-104`.
    sub.add(
        "PACKET_TERMINAL_BOUNDARY_KINDS",
        PyTuple::new(py, PACKET_TERMINAL_BOUNDARY_KINDS)?,
    )?;
    Ok(())
}
