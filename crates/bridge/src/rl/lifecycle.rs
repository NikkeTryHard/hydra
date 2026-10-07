//! rl/lifecycle: cold native metadata for the continuous-RL run boundary.
//!
//! Facade over the scoped lifecycle submodules (each holds one cold
//! concern behind the shared `lifecycle_common` gates; no second
//! implementation anywhere):
//! - `lifecycle_common`: error + closed JSON gates (crate-internal).
//! - `lifecycle_intake`: 53-field RL profile, recipe laws, asset +
//!   manifest gates.
//! - `lifecycle_ledger`: deterministic source-ledger generation.
//! - `lifecycle_run`: `prepare_run` descriptor.
//! - `lifecycle_collector`: per-generation `collector_spec`.
//! - `lifecycle_generation`: canonical per-generation RunSpec + header.
//! - `lifecycle_control`: control assembly + ordinary-manifest binding.
//! - `lifecycle_report`: scheduling choice + PPO report reduction.
//! - `lifecycle_ffi`: batch PyO3 surface (the ONLY PyO3 in this subtree).
//! Cold data flow (each step states its own contract below):
//! `prepare_run` validates the resolved run config + IL manifest against
//! real repo assets and forms every structural identity;
//! `collector_spec` binds one generation's runtime collector arguments once
//! its behavior policy id is real;
//! `prepare_generation` binds the sealed artifact + frozen order to the
//! canonical RunSpec after the validated seal; `build_control` assembles
//! the control document from landed receipts; `ordinary_bindings` verifies
//! a Python-built ordinary manifest; `choose_work`/`ppo_report` serve the
//! GPU owner with pure scheduling + metrics. Control/evidence carry
//! progress; ordinary payloads carry counters only; policy/source/shuffle
//! ids never enter metric reports.

pub use super::lifecycle_collector::collector_spec;
pub use super::lifecycle_common::LifecycleError;
pub use super::lifecycle_control::{build_control, ordinary_bindings, CONTROL_TICKET_FIELDS};
pub use super::lifecycle_generation::{prepare_generation, RUNSPEC_FIELDS};
pub use super::lifecycle_intake::{recipe_digests, RL_CONFIG_FIELDS};
pub use super::lifecycle_ledger::{ledger_split_for, wall_id_for};
pub use super::lifecycle_report::{choose_work, ppo_report, PPO_REPORT_LEN, WorkKind};
pub use super::lifecycle_run::{cold_init_seed, prepare_run, PREPARE_RUN_FIELDS, PREPARE_RUN_UTILITY_FIELDS, RESOLVED_FIELDS};

/// Cold-descriptor schema version (this module's output contract).
pub const LIFECYCLE_VERSION: &str = "1.0.0";
/// Pinned rules-core revision (plan section 13; the repo dependency pin in
/// `crates/bridge/Cargo.toml` must contain it).
pub const CORE_REVISION: &str = "479c1faeb33d082965eef8198f63261a79c0fce3";
/// Objective id bound into every generation's RunSpec bindings.
pub const RL_OBJECTIVE_ID: &str = "ppo";
/// Optimizer/scheduler ids bound into every generation's bindings.
pub const RL_OPTIMIZER_ID: &str = "adamw";
pub const RL_SCHEDULER_ID: &str = "constant";
/// Lifecycle-owned frozen selection-metric vocabulary (first use, frozen in
/// `LIFECYCLE_VERSION`): promotion is decided by the real duplicate-wall
/// evaluation path, never by a live training metric.
pub const RL_SELECTION_METRIC_ID: &str = "duplicate_wall_placement";

/// Repo-relative contract asset paths read under `repo_dir`.
pub const RULES_RELPATH: &str = "configs/rules/tenhou_4p_hanchan_v1.json";
pub const ACTION_TABLE_RELPATH: &str = "configs/contracts/action_table_v1.json";
pub const EVENT_SCHEMA_RELPATH: &str = "configs/contracts/event_schema_v1.json";
pub const OBSERVATION_SCHEMA_RELPATH: &str = "configs/contracts/observation_schema_v1.json";
pub const PACKET_BOUNDARY_RELPATH: &str = "configs/contracts/packet_boundary_v1.json";
pub const MODEL_INPUT_RELPATH: &str = "configs/models/model_input_v1.json";
pub const CARGO_RELPATH: &str = "crates/bridge/Cargo.toml";

/// Code owners frozen into `row_derivation_hash`: the derivation binds the
/// ACTUAL source bytes of these sixteen files (name, repo-relative path),
/// read and hashed at `prepare_run` time. Any edit invalidates the
/// descriptor (re-run); weight identity is unaffected.
pub const LIFECYCLE_CODE_SOURCES: [(&str, &str); 16] = [
    ("rl_gpu.py", "python/hydra2/training/rl_gpu.py"),
    ("rl_actor_packed.py", "python/hydra2/training/rl_actor_packed.py"),
    ("observation/packed.rs", "crates/bridge/src/rl/observation/packed.rs"),
    ("rl_objective.py", "python/hydra2/training/rl_objective.py"),
    ("wall.rs", "crates/bridge/src/rl/wall.rs"),
    ("observation.rs", "crates/bridge/src/rl/observation.rs"),
    ("rl_learner.py", "python/hydra2/training/rl_learner.py"),
    ("events.rs", "crates/bridge/src/rl/events.rs"),
    ("events/arena.rs", "crates/bridge/src/rl/events/arena.rs"),
    ("observation/identity.rs", "crates/bridge/src/rl/observation/identity.rs"),
    ("core.rs", "crates/bridge/src/rl/core.rs"),
    ("core/row.rs", "crates/bridge/src/rl/core/row.rs"),
    ("core/driver.rs", "crates/bridge/src/rl/core/driver.rs"),
    ("pool/fill.rs", "crates/bridge/src/rl/pool/fill.rs"),
    ("pool/round_plan.rs", "crates/bridge/src/rl/pool/round_plan.rs"),
    ("pool/round_plan_runtime.rs", "crates/bridge/src/rl/pool/round_plan_runtime.rs"),
];
