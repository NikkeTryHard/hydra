//! rl: CPU-native continuous-RL subtree (plan sections 4/13).
//!
//! Pure game/wall/sampling logic lives in the leaf modules (`wall`, `pool`,
//! `core`, `actions`, `events`, `observation`); the ONLY PyO3 surfaces are
//! the `*_ffi` registrars below, each exporting through one `register` call.
//! Single cdylib, no new entry points. No per-row Python callbacks.
//!
//! Registrar ownership (each owner edits only its own modules):
//! - wall/config surface: `ffi`, `config_ffi` (ConfigContracts/NativeWall).
//! - pool surface: `pool_ffi` over `pool`/`core`/`actions`/`events`/
//!   `observation` (NativeIntegrate).
//! - lifecycle surface: `lifecycle_ffi` over `lifecycle*` (RlLifecycle).
//! - runtime-snapshot surface: `runtime_snapshot_ffi` over
//!   `runtime_snapshot*` (RuntimePersistence).
//! - metrics surface: `metrics_ffi` over `metrics` (Main).
//! - artifact surface: `artifact_ffi` over `trajectory`/`artifact`/
//!   `snapshot`/`replay_index` (ArtifactStorage; separate `rl_artifact`
//!   submodule on the TOP module, not on `rl`).
//! - package surface: `package_verify_ffi` over `package_verify`
//!   (PackageVerifier).

pub mod actions;
pub mod artifact;
pub mod artifact_ffi;
pub mod config;
pub mod config_ffi;
pub mod core;
pub mod events;
pub mod ffi;
pub mod lifecycle;
pub mod lifecycle_collector;
pub mod lifecycle_common;
pub mod lifecycle_control;
pub mod live;
pub mod live_ffi;
pub mod lifecycle_ffi;
pub mod lifecycle_generation;
pub mod lifecycle_intake;
pub mod lifecycle_ledger;
pub mod lifecycle_report;
pub mod lifecycle_run;
pub mod metrics;
pub mod metrics_ffi;
pub mod observation;
pub mod package_verify;
pub mod package_verify_ffi;
pub mod pool;
pub mod pool_ffi;
pub mod replay_index;
pub mod runtime_snapshot;
pub mod runtime_snapshot_control;
pub mod runtime_snapshot_ffi;
pub mod runtime_snapshot_pool;
pub mod runtime_snapshot_tape;
pub mod snapshot;
pub mod trajectory;
pub mod wall;

use pyo3::prelude::*;

/// Register the `hydra2._native.rl` submodule (batch wall/seed/u53 surface
/// from `ffi`, closed config surface from `config_ffi`, batch-only pool
/// surface from `pool_ffi`, lifecycle surface from `lifecycle_ffi`, live
/// surface from `live_ffi`, runtime-snapshot surface from
/// `runtime_snapshot_ffi`, metrics surface from `metrics_ffi`, package
/// surface from `package_verify_ffi`) plus the separate
/// `hydra2._native.rl_artifact` submodule from `artifact_ffi`
/// (ArtifactStorage-owned, registered on the TOP module). Single cdylib,
/// no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    artifact_ffi::register(m)?;
    let sub = PyModule::new(py, "rl")?;
    ffi::register(&sub)?;
    config_ffi::register(&sub)?;
    pool_ffi::register(&sub)?;
    lifecycle_ffi::register(&sub)?;
    live_ffi::register(&sub)?;
    runtime_snapshot_ffi::register(&sub)?;
    metrics_ffi::register(&sub)?;
    package_verify_ffi::register(&sub)?;
    m.add_submodule(&sub)?;
    Ok(())
}
