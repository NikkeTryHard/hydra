//! event_schema: frozen SPEC 7.1 per-kind event-schema matrix values.
//!
//! TABLE owner for `python/hydra2/contracts/event_schema.py` read-only
//! tables: the artifact identity strings plus the per-kind `_row(...)`
//! call-site arguments (required scalars = require list, tuple specs =
//! forbid/allow list, constraints, raw delta paths, predecessor/successor
//! grammar, visibility, actor flag). The `EventSchemaRow` dataclass, the
//! `_row` / `_expand_actor_paths` builders, the payload/digest builders +
//! caches, and the artifact parse/load entry points stay Python-side
//! (validating roots + canon authority; see `contracts` rank 6-7 docs).
//! No pure-compute fns: digests fold through the Python canon owner and
//! every row derives via the Python `_row` builder, so there is no
//! detached-compute leaf to bridge. Fail-closed: nothing to fail (frozen
//! literals only).
//!
//! Single-cdylib tree: registers its consts on the shared
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `contracts::register`); no new entry point. MAIN wires this with two
//! lines (see port report).

// Precedent `use pyo3::prelude::*;` -> contracts.rs:52.
use pyo3::prelude::*;
// Precedent `PyDict` + `PyModule` + `PyTuple` imports -> contracts.rs:53.
use pyo3::types::{PyDict, PyModule, PyTuple};

/// Named structural constraint (`event_schema.py:370`).
const ACCEPTED_CONSTRAINT: &str = "call_resolved_resolution_shape";
/// Published artifact type (`event_schema.py:372`).
const EVENT_SCHEMA_ARTIFACT_TYPE: &str = "hydra2.event_schema";
/// Published schema version (`event_schema.py:373`).
const EVENT_SCHEMA_SCHEMA_VERSION: &str = "1.0.0";
/// `EVENT_SCHEMA_RELPATH` (`event_schema.py:374`) as POSIX text; Python
/// re-wraps with `Path(...)` to keep the `Path` type (precedent
/// `ACTION_TABLE_RELPATH` -> action_artifact.rs:23-26).
const EVENT_SCHEMA_RELPATH: &str = "configs/contracts/event_schema_v1.json";

/// One raw delta-path element: a field name or a concrete seat index.
/// Mirrors `PathElem` (precedent -> contracts.rs:1061-1064); the `"actor"`
/// placeholder stays a `Name` (placeholder expansion stays Python-side in
/// `_expand_actor_paths`, `event_schema.py:100-115`).
#[derive(Clone, Copy)]
enum SchemaPathElem {
    Name(&'static str),
    Seat(u8),
}

/// One frozen per-kind row spec: the `_row(...)` call-site arguments in
/// matrix order (`event_schema.py:138-366`). Matrix order is SPEC order
/// (`EVENT_KINDS`, precedent -> contracts.rs:216-237), so dict insertion
/// below preserves the published row order.
#[derive(Clone, Copy)]
struct EventSchemaSpec {
    kind: &'static str,
    visibility: &'static str,
    actor_required: bool,
    required: &'static [&'static str],
    tuples: &'static [(&'static str, &'static str)],
    constraints: &'static [&'static str],
    delta_paths: &'static [&'static [SchemaPathElem]],
    predecessors: &'static [&'static str],
    successors: &'static [&'static str],
}

/// Default tuple specs: every tuple field `"empty"` in
/// `PAYLOAD_TUPLE_FIELDS` order (`event_schema.py:88-92`; field order
/// precedent -> contracts.rs:1279).
const DEFAULT_TUPLE_SPECS: [(&str, &str); 3] = [
    ("consumed_tiles", "empty"),
    ("offered_action_ids", "empty"),
    ("accepted_action_ids", "empty"),
];

/// Meld-claim tuple specs (`chi`/`pon`/`daiminkan`/`ankan`,
/// `event_schema.py:234-238`).
const MELD_TUPLE_SPECS: [(&str, &str); 3] = [
    ("consumed_tiles", "any"),
    ("offered_action_ids", "empty"),
    ("accepted_action_ids", "empty"),
];

/// `call_resolved` tuple specs (`event_schema.py:220-224`).
const CALL_RESOLVED_TUPLE_SPECS: [(&str, &str); 3] = [
    ("consumed_tiles", "empty"),
    ("offered_action_ids", "any"),
    ("accepted_action_ids", "any"),
];

/// Shared `_SETTLEMENT_PATHS` contents (`event_schema.py:128-132`).
const SETTLEMENT_DELTA_PATHS: [&[SchemaPathElem]; 3] = [
    &[SchemaPathElem::Name("scores")],
    &[SchemaPathElem::Name("honba")],
    &[SchemaPathElem::Name("riichi_sticks")],
];

/// Shared meld-append delta path (`chi`/`pon`, `event_schema.py:239`).
const MELD_APPEND_DELTA_PATH: [&[SchemaPathElem]; 1] =
    [&[SchemaPathElem::Name("melds"), SchemaPathElem::Name("actor")]];

/// `riichi_declared` delta path (`event_schema.py:196`).
const RIICHI_DECLARED_DELTA_PATH: [&[SchemaPathElem]; 1] = [&[
    SchemaPathElem::Name("riichi_states"),
    SchemaPathElem::Name("actor"),
]];

/// Shared meld + kan-count delta paths (`daiminkan`/`ankan`/`kakan`,
/// `event_schema.py:267`).
const MELD_KAN_DELTA_PATHS: [&[SchemaPathElem]; 2] = [
    &[SchemaPathElem::Name("melds"), SchemaPathElem::Name("actor")],
    &[SchemaPathElem::Name("kan_count")],
];

/// Empty raw delta paths (`turn_advance`/`draw_tile`/`discard`/
/// `call_window`/`call_resolved`: no `delta_paths=` argument).
const NO_DELTA_PATHS: [&[SchemaPathElem]; 0] = [];

/// Frozen per-kind matrix, SPEC order (`event_schema.py:138-366`).
const EVENT_SCHEMA_SPECS: [EventSchemaSpec; 21] = [
    // `event_schema.py:138-146`.
    EventSchemaSpec {
        kind: "game_start",
        visibility: "public",
        actor_required: false,
        required: &["round_index", "scores"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[&[SchemaPathElem::Name("scores")]],
        predecessors: &[],
        successors: &["round_start"],
    },
    // `event_schema.py:147-160`.
    EventSchemaSpec {
        kind: "round_start",
        visibility: "public",
        actor_required: true,
        required: &["round_index", "scores"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[
            &[SchemaPathElem::Name("round_index")],
            &[SchemaPathElem::Name("honba")],
            &[SchemaPathElem::Name("riichi_sticks")],
            &[SchemaPathElem::Name("scores")],
        ],
        predecessors: &["game_start", "round_end", "draw_end", "abortive_draw"],
        successors: &["turn_advance"],
    },
    // `event_schema.py:161-175`.
    EventSchemaSpec {
        kind: "turn_advance",
        visibility: "public",
        actor_required: true,
        required: &[],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &NO_DELTA_PATHS,
        predecessors: &[
            "round_start",
            "discard",
            "riichi_accepted",
            "call_resolved",
            "dora_revealed",
            "ankan",
            "kakan",
        ],
        successors: &["draw_tile"],
    },
    // `event_schema.py:176-183`.
    EventSchemaSpec {
        kind: "draw_tile",
        visibility: "actor_private",
        actor_required: true,
        required: &["tile"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &NO_DELTA_PATHS,
        predecessors: &["turn_advance", "ankan", "kakan", "daiminkan"],
        successors: &["discard", "riichi_declared", "ankan", "kakan", "tsumo"],
    },
    // `event_schema.py:184-191`.
    EventSchemaSpec {
        kind: "discard",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &NO_DELTA_PATHS,
        predecessors: &[
            "draw_tile",
            "riichi_declared",
            "chi",
            "pon",
            "dora_revealed",
        ],
        successors: &["call_window", "turn_advance", "abortive_draw", "draw_end"],
    },
    // `event_schema.py:192-200`.
    EventSchemaSpec {
        kind: "riichi_declared",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &RIICHI_DECLARED_DELTA_PATH,
        predecessors: &["draw_tile"],
        successors: &["discard"],
    },
    // `event_schema.py:201-208`.
    EventSchemaSpec {
        kind: "riichi_accepted",
        visibility: "public",
        actor_required: true,
        required: &[],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[
            &[
                SchemaPathElem::Name("riichi_states"),
                SchemaPathElem::Name("actor"),
            ],
            &[SchemaPathElem::Name("riichi_sticks")],
            &[
                SchemaPathElem::Name("ippatsu"),
                SchemaPathElem::Name("actor"),
            ],
        ],
        predecessors: &["discard", "call_resolved"],
        successors: &["turn_advance"],
    },
    // `event_schema.py:209-215`.
    EventSchemaSpec {
        kind: "call_window",
        visibility: "public",
        actor_required: false,
        required: &[],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &NO_DELTA_PATHS,
        predecessors: &["discard"],
        successors: &["call_resolved"],
    },
    // `event_schema.py:216-228`.
    EventSchemaSpec {
        kind: "call_resolved",
        visibility: "server_private",
        actor_required: false,
        required: &[],
        tuples: &CALL_RESOLVED_TUPLE_SPECS,
        constraints: &["call_resolved_resolution_shape"],
        delta_paths: &NO_DELTA_PATHS,
        predecessors: &["call_window"],
        successors: &[
            "chi",
            "pon",
            "daiminkan",
            "ron",
            "turn_advance",
            "riichi_accepted",
        ],
    },
    // `event_schema.py:229-242`.
    EventSchemaSpec {
        kind: "chi",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id", "source_seat"],
        tuples: &MELD_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &MELD_APPEND_DELTA_PATH,
        predecessors: &["call_resolved"],
        successors: &["discard"],
    },
    // `event_schema.py:243-256`.
    EventSchemaSpec {
        kind: "pon",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id", "source_seat"],
        tuples: &MELD_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &MELD_APPEND_DELTA_PATH,
        predecessors: &["call_resolved"],
        successors: &["discard"],
    },
    // `event_schema.py:257-270`.
    EventSchemaSpec {
        kind: "daiminkan",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id", "source_seat"],
        tuples: &MELD_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &MELD_KAN_DELTA_PATHS,
        predecessors: &["call_resolved"],
        successors: &["dora_revealed", "draw_tile"],
    },
    // `event_schema.py:271-284`.
    EventSchemaSpec {
        kind: "ankan",
        visibility: "public",
        actor_required: true,
        required: &["action_id"],
        tuples: &MELD_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &MELD_KAN_DELTA_PATHS,
        predecessors: &["draw_tile"],
        successors: &["dora_revealed"],
    },
    // `event_schema.py:285-293` (default all-`"empty"` tuple specs: no
    // `tuples=` argument).
    EventSchemaSpec {
        kind: "kakan",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &MELD_KAN_DELTA_PATHS,
        predecessors: &["draw_tile"],
        successors: &["dora_revealed", "ron", "draw_tile"],
    },
    // `event_schema.py:294-311`.
    EventSchemaSpec {
        kind: "dora_revealed",
        visibility: "public",
        actor_required: false,
        required: &["tile"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[&[SchemaPathElem::Name("dora_indicators")]],
        predecessors: &["ankan", "kakan", "daiminkan", "discard", "draw_tile"],
        successors: &[
            "draw_tile",
            "turn_advance",
            "discard",
            "ron",
            "tsumo",
            "round_end",
            "abortive_draw",
            "draw_end",
        ],
    },
    // `event_schema.py:312-320`.
    EventSchemaSpec {
        kind: "ron",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id", "source_seat"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &SETTLEMENT_DELTA_PATHS,
        predecessors: &["call_resolved", "kakan"],
        successors: &["round_end"],
    },
    // `event_schema.py:321-329`.
    EventSchemaSpec {
        kind: "tsumo",
        visibility: "public",
        actor_required: true,
        required: &["tile", "action_id"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &SETTLEMENT_DELTA_PATHS,
        predecessors: &["draw_tile"],
        successors: &["round_end"],
    },
    // `event_schema.py:330-338`.
    EventSchemaSpec {
        kind: "draw_end",
        visibility: "public",
        actor_required: false,
        required: &["scores", "reason"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &SETTLEMENT_DELTA_PATHS,
        predecessors: &["discard"],
        successors: &["round_end", "game_end"],
    },
    // `event_schema.py:339-347`.
    EventSchemaSpec {
        kind: "abortive_draw",
        visibility: "public",
        actor_required: false,
        required: &["round_index", "scores", "reason"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[
            &[SchemaPathElem::Name("scores")],
            &[SchemaPathElem::Name("riichi_sticks")],
        ],
        predecessors: &["discard", "draw_tile", "ankan"],
        successors: &["round_start", "game_end"],
    },
    // `event_schema.py:348-356`: raw paths flatten `*_SETTLEMENT_PATHS`,
    // `("round_index",)`, `*_ALL_SEAT_RESET_PATHS`
    // (`event_schema.py:118-132` + `event_schema.py:353`).
    EventSchemaSpec {
        kind: "round_end",
        visibility: "public",
        actor_required: false,
        required: &["round_index", "scores"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[
            &[SchemaPathElem::Name("scores")],
            &[SchemaPathElem::Name("honba")],
            &[SchemaPathElem::Name("riichi_sticks")],
            &[SchemaPathElem::Name("round_index")],
            &[
                SchemaPathElem::Name("riichi_states"),
                SchemaPathElem::Seat(0),
            ],
            &[
                SchemaPathElem::Name("riichi_states"),
                SchemaPathElem::Seat(1),
            ],
            &[
                SchemaPathElem::Name("riichi_states"),
                SchemaPathElem::Seat(2),
            ],
            &[
                SchemaPathElem::Name("riichi_states"),
                SchemaPathElem::Seat(3),
            ],
            &[SchemaPathElem::Name("ippatsu"), SchemaPathElem::Seat(0)],
            &[SchemaPathElem::Name("ippatsu"), SchemaPathElem::Seat(1)],
            &[SchemaPathElem::Name("ippatsu"), SchemaPathElem::Seat(2)],
            &[SchemaPathElem::Name("ippatsu"), SchemaPathElem::Seat(3)],
        ],
        predecessors: &["ron", "tsumo", "draw_end"],
        successors: &["round_start", "game_end"],
    },
    // `event_schema.py:357-365`.
    EventSchemaSpec {
        kind: "game_end",
        visibility: "public",
        actor_required: false,
        required: &["round_index", "scores", "reason"],
        tuples: &DEFAULT_TUPLE_SPECS,
        constraints: &[],
        delta_paths: &[&[SchemaPathElem::Name("scores")]],
        predecessors: &["round_end", "draw_end", "abortive_draw"],
        successors: &[],
    },
];

/// Register the frozen event-schema tables on the shared `contracts`
/// submodule (mirrors `contracts::register`; single cdylib, no new entry).
///
/// Precedents: `register` signature shape -> action_artifact.rs:42;
/// `let py = sub.py()` -> action_artifact.rs:43;
/// `sub.add(<str>, <&str>)` -> contracts.rs:1153;
/// `PyTuple::new(py, [...])` for frozen tuples -> contracts.rs:1156;
/// `PyDict::new(py)` -> contracts.rs:1227; `set_item` -> contracts.rs:1229;
/// `sub.add(<str>, <dict>)` -> contracts.rs:1231;
/// mixed str/int path elems via `into_pyobject(py)?.into_any()` ->
/// contracts.rs:1101-1102; `PyTuple::new(py, <Vec<Bound<..>>>)` ->
/// contracts.rs:1286 (tuple of dicts) + contracts.rs:1106 (tuple of paths).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    // `event_schema.py:370-374`.
    sub.add("ACCEPTED_CONSTRAINT", ACCEPTED_CONSTRAINT)?;
    sub.add("EVENT_SCHEMA_ARTIFACT_TYPE", EVENT_SCHEMA_ARTIFACT_TYPE)?;
    sub.add("EVENT_SCHEMA_SCHEMA_VERSION", EVENT_SCHEMA_SCHEMA_VERSION)?;
    sub.add("EVENT_SCHEMA_RELPATH", EVENT_SCHEMA_RELPATH)?;
    let required = PyDict::new(py);
    let tuple_specs = PyDict::new(py);
    let constraints = PyDict::new(py);
    let delta_paths = PyDict::new(py);
    let visibility = PyDict::new(py);
    let actor_required = PyDict::new(py);
    let predecessors = PyDict::new(py);
    let successors = PyDict::new(py);
    for spec in EVENT_SCHEMA_SPECS {
        required.set_item(spec.kind, PyTuple::new(py, spec.required)?)?;
        let mut specs = Vec::with_capacity(spec.tuples.len());
        for &(field, mode) in spec.tuples {
            specs.push(PyTuple::new(py, [field, mode])?);
        }
        tuple_specs.set_item(spec.kind, PyTuple::new(py, specs)?)?;
        constraints.set_item(spec.kind, PyTuple::new(py, spec.constraints)?)?;
        let mut paths = Vec::with_capacity(spec.delta_paths.len());
        for &path in spec.delta_paths {
            let mut elems = Vec::with_capacity(path.len());
            for elem in path {
                elems.push(match elem {
                    SchemaPathElem::Name(s) => s.into_pyobject(py)?.into_any(),
                    SchemaPathElem::Seat(i) => i.into_pyobject(py)?.into_any(),
                });
            }
            paths.push(PyTuple::new(py, elems)?);
        }
        delta_paths.set_item(spec.kind, PyTuple::new(py, paths)?)?;
        visibility.set_item(spec.kind, spec.visibility)?;
        actor_required.set_item(spec.kind, spec.actor_required)?;
        predecessors.set_item(spec.kind, PyTuple::new(py, spec.predecessors)?)?;
        successors.set_item(spec.kind, PyTuple::new(py, spec.successors)?)?;
    }
    sub.add("EVENT_SCHEMA_REQUIRED", required)?;
    sub.add("EVENT_SCHEMA_TUPLE_SPECS", tuple_specs)?;
    sub.add("EVENT_SCHEMA_CONSTRAINTS", constraints)?;
    sub.add("EVENT_SCHEMA_DELTA_PATHS", delta_paths)?;
    sub.add("EVENT_SCHEMA_VISIBILITY", visibility)?;
    sub.add("EVENT_SCHEMA_ACTOR_REQUIRED", actor_required)?;
    sub.add("EVENT_SCHEMA_PREDECESSORS", predecessors)?;
    sub.add("EVENT_SCHEMA_SUCCESSORS", successors)?;
    Ok(())
}
