//! `hydra2._native.rl_artifact`: batch-only PyO3 surface for the
//! trajectory/artifact/control pipeline (ArtifactStorage-owned).
//!
//! Boundary law (mirrors `canon_rng`/`contracts`): validate attached, run
//! detached, wrap attached. Every function takes opaque JSON bytes, path
//! strings, ledger string lists, or integer handles — the expanded row list
//! NEVER crosses into Python on the production path (handles + paths +
//! descriptors/counts/digests only). Bounded-fixture byte helpers do not
//! exist here: gate tests drive the same path-based functions against
//! `tmp_path` fixtures. All native failures surface as `ValueError`
//! (fail-closed rejects, never silent skips), mirroring `rows_seal`.
//!
//! Registration: `register` takes the TOP `_native` module and installs the
//! `rl_artifact` submodule (stable path `hydra2._native.rl_artifact`).
mod replay;

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::rl::artifact::{
    HEADER_KEYS, SealContext, begin_seal, poll_seal, sealer_busy, seal_from_tape,
    validate_artifact_file, wait_seal,
};
use crate::rl::snapshot::{join_root, ordinary_checkpoint_receipt};
use crate::rl::trajectory::{TapeError, parse_digest_text};

fn artifact_err(context: &str, detail: String) -> PyErr {
    PyValueError::new_err(format!("rl_artifact {context} rejected: {detail}"))
}

fn tape_err(context: &'static str) -> impl Fn(TapeError) -> PyErr {
    move |e: TapeError| artifact_err(context, e.0)
}

/// Closed seal context from JSON bytes plus ledgers (attached arg-check).
fn seal_context_from_json(
    header_doc: &[u8],
    source_ledger: Vec<String>,
    eval_ledger: Vec<String>,
    expect_bc: bool,
) -> Result<SealContext, String> {
    let header_value: Value =
        serde_json::from_slice(header_doc).map_err(|e| format!("header is not JSON: {e}"))?;
    let header_map = header_value
        .as_object()
        .ok_or_else(|| "header must be an object".to_string())?;
    if header_map.len() != HEADER_KEYS.len()
        || !HEADER_KEYS.iter().all(|k| header_map.contains_key(*k))
    {
        return Err("header must carry exactly the eight SPEC12.3 digest keys".to_string());
    }
    let mut header = BTreeMap::new();
    for key in HEADER_KEYS {
        let text = header_map
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("header lacks {key}"))?;
        parse_digest_text(text).map_err(|e| e.0)?;
        header.insert(key.to_string(), text.to_string());
    }
    let source_walls: HashSet<String> = source_ledger.into_iter().collect();
    let eval_walls: HashSet<String> = eval_ledger.into_iter().collect();
    if source_walls.is_empty() {
        return Err("source ledger must be nonempty".to_string());
    }
    for wall in &eval_walls {
        if source_walls.contains(wall) {
            return Err(format!("wall {wall} cannot serve both ledgers"));
        }
    }
    Ok(SealContext {
        header,
        source_walls,
        eval_walls,
        expect_bc,
    })
}

/// Canonical 4-field seal receipt via the feed JCS printer.
fn receipt_json(
    artifact_digest: &str,
    row_count: usize,
    compressed_bytes: u64,
    uncompressed_bytes: u64,
) -> Result<Vec<u8>, String> {
    let doc = serde_json::json!({
        "artifact_digest": artifact_digest,
        "row_count": row_count,
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": uncompressed_bytes,
    });
    hydra_feed::canon::canonical_bytes_value(&doc, "rl_artifact:receipt")
        .map_err(|e| format!("receipt JCS emit failed: {e:?}"))
}

#[pyfunction]
fn seal_begin(
    py: Python<'_>,
    tape_handle: u64,
    header_doc: Vec<u8>,
    source_ledger: Vec<String>,
    eval_ledger: Vec<String>,
    expect_bc: bool,
    artifact_path: String,
    metas_path: String,
) -> PyResult<u64> {
    let context =
        seal_context_from_json(&header_doc, source_ledger, eval_ledger, expect_bc)
            .map_err(|detail| artifact_err("seal_begin", detail))?;
    py.detach(|| {
        begin_seal(crate::rl::artifact::SealRequest {
            tape_handle,
            context,
            artifact_path: PathBuf::from(artifact_path),
            metas_path: PathBuf::from(metas_path),
        })
        .map_err(tape_err("seal_begin"))
    })
}

/// Bounded gate-fixture staging (TESTS AND TOOLING ONLY — never production):
/// builds a native tape from canonical row documents plus per-row behavior
/// values, registers it, and returns the handle. Production registers pool
/// tapes Rust-to-Rust; this helper exists so gate tests drive the identical
/// seal/validate/order/export path without a full pool simulation.
///
/// Input: JSON array of `{wall_id, game_id, utilities?, rows: [{doc, v,
/// bound?}]}` where `doc` is a SPEC12.3 row WITHOUT `row_id`, `v` is the
/// frozen behavior value (f32-exact JSON number), `bound` defaults true
/// (unbound rows omit return_vector/raw_advantage and freeze later from
/// per-game `utilities`), and `utilities` is the canonical G4 quad.
/// GameRows synthesize neutral pool scalars (empty concealed, unrevealed
/// dora, draw-phase); arenas stay empty so
/// every row is `T=32` (multi-`T` order coverage uses structural metas
/// directly). The materializer returns each row's embedded observation
/// (identity-bound by the seal); the encoder is the REAL
/// `observation::encode_row_facts` over the typed facts (zero-history
/// planes) — no stub values anywhere on this path.
#[pyfunction]
fn stage_test_tape(py: Python<'_>, games_doc: Vec<u8>) -> PyResult<u64> {
    let games_value: Value = serde_json::from_slice(&games_doc)
        .map_err(|e| artifact_err("stage_test_tape", format!("games not JSON: {e}")))?;
    let games_array = games_value
        .as_array()
        .ok_or_else(|| artifact_err("stage_test_tape", "games must be an array".to_string()))?;
    if games_array.is_empty() {
        return Err(artifact_err(
            "stage_test_tape",
            "games must be nonempty".to_string(),
        ));
    }
    let mut games = Vec::with_capacity(games_array.len());
    let mut obs_by_decision: std::collections::HashMap<String, Value> =
        std::collections::HashMap::new();
    for game_value in games_array {
        let game_map = game_value
            .as_object()
            .ok_or_else(|| artifact_err("stage_test_tape", "game must be an object".to_string()))?;
        let wall_id = game_map
            .get("wall_id")
            .and_then(Value::as_str)
            .ok_or_else(|| artifact_err("stage_test_tape", "game lacks wall_id".to_string()))?;
        let game_id = game_map
            .get("game_id")
            .and_then(Value::as_str)
            .ok_or_else(|| artifact_err("stage_test_tape", "game lacks game_id".to_string()))?;
        let rows_array = game_map
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| artifact_err("stage_test_tape", "game lacks rows".to_string()))?;
        // Optional per-game canonical utilities (required for the
        // terminal gather/freeze path; bound rows carry their own G4).
        let utilities = match game_map.get("utilities") {
            None => None,
            Some(Value::Null) => None,
            Some(value) => {
                let arr = value.as_array().ok_or_else(|| {
                    artifact_err("stage_test_tape", "utilities must hold four seats".to_string())
                })?;
                if arr.len() != 4 {
                    return Err(artifact_err(
                        "stage_test_tape",
                        "utilities must hold four seats".to_string(),
                    ));
                }
                let mut g4 = [0.0f64; 4];
                for (i, seat) in arr.iter().enumerate() {
                    g4[i] = seat.as_f64().ok_or_else(|| {
                        artifact_err("stage_test_tape", "utilities must be numbers".to_string())
                    })?;
                }
                if g4.iter().any(|v| !v.is_finite()) {
                    return Err(artifact_err(
                        "stage_test_tape",
                        "utilities must be finite".to_string(),
                    ));
                }
                Some(g4)
            }
        };
        let mut rows = Vec::with_capacity(rows_array.len());
        for entry in rows_array {
            rows.push(test_gamerow(wall_id, game_id, entry, &mut obs_by_decision)?);
        }
        if rows.is_empty() {
            return Err(artifact_err(
                "stage_test_tape",
                "game rows must be nonempty".to_string(),
            ));
        }
        games.push(crate::rl::trajectory::TapeGame {
            wall_id: wall_id.to_string(),
            game_id: game_id.to_string(),
            arena: crate::rl::events::EventArena::new(),
            rows,
            utilities,
        });
    }
    py.detach(|| {
        let materialize = move |row: &crate::rl::core::GameRow,
                                _arena: &crate::rl::events::EventArena| {
            obs_by_decision
                .get(&row.decision_id)
                .cloned()
                .ok_or_else(|| TapeError(format!("test tape lacks obs for {}", row.decision_id)))
        };
        let encode = |row: &crate::rl::core::GameRow,
                      arena: &crate::rl::events::EventArena,
                      t: u32| {
            crate::rl::observation::encode_row_facts(row, arena, t as usize)
                .map_err(|e| TapeError(e.0))
        };
        crate::rl::trajectory::register_tape(games, Box::new(materialize), Box::new(encode))
            .map_err(tape_err("stage_test_tape"))
    })
}

/// Synthesize one test GameRow from a canonical row doc plus behavior value.
fn test_gamerow(
    wall_id: &str,
    game_id: &str,
    entry: &Value,
    obs_by_decision: &mut std::collections::HashMap<String, Value>,
) -> Result<crate::rl::core::GameRow, PyErr> {
    let fail = |detail: String| artifact_err("stage_test_tape", detail);
    let entry_map = entry
        .as_object()
        .ok_or_else(|| fail("row entry must be an object".to_string()))?;
    let doc = entry_map
        .get("doc")
        .ok_or_else(|| fail("row entry lacks doc".to_string()))?;
    let v = entry_map
        .get("v")
        .and_then(Value::as_f64)
        .ok_or_else(|| fail("row entry lacks numeric v".to_string()))?;
    if !v.is_finite() || f64::from(v as f32) != v {
        return Err(fail("test behavior value must be f32-exact finite".to_string()));
    }
    let field_str = |name: &str| {
        doc.get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| fail(format!("doc lacks {name}")))
            .map(str::to_string)
    };
    let field_u64 = |name: &str| {
        doc.get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| fail(format!("doc lacks {name}")))
    };
    let row_wall = field_str("wall_id")?;
    let row_game = field_str("game_id")?;
    if row_wall != wall_id || row_game != game_id {
        return Err(fail("row identity mismatches its game".to_string()));
    }
    let actor_u64 = field_u64("actor")?;
    if actor_u64 > 3 {
        return Err(fail("test actor outside four seats".to_string()));
    }
    let actor = actor_u64 as u8;
    let mask = doc
        .get("legal_mask")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("doc lacks legal_mask".to_string()))?;
    let mut legal_ids = Vec::new();
    for (i, flag) in mask.iter().enumerate() {
        if flag == &Value::Bool(true) {
            legal_ids.push(i as u32);
        }
    }
    let selected = field_u64("selected_action_id")? as u32;
    let logp_f64 = doc
        .get("old_selected_log_probability")
        .and_then(Value::as_f64)
        .ok_or_else(|| fail("doc lacks old_selected_log_probability".to_string()))?;
    if !logp_f64.is_finite() || f64::from(logp_f64 as f32) != logp_f64 {
        return Err(fail("test old_logp must be f32-exact finite".to_string()));
    }
    // Bound rows carry frozen terminal facts in the doc; unbound rows (the
    // terminal gather/freeze path) omit them and freeze later from game
    // utilities. The seal rebuilds canonical docs from rows either way.
    let bound = entry_map.get("bound").and_then(Value::as_bool).unwrap_or(true);
    let (return_g4, raw_advantage_f32) = if bound {
        let g4_value = doc
            .get("return_vector")
            .and_then(Value::as_array)
            .ok_or_else(|| fail("doc lacks return_vector".to_string()))?;
        if g4_value.len() != 4 {
            return Err(fail("test return_vector must hold four seats".to_string()));
        }
        let mut g4 = [0.0f64; 4];
        for (i, seat) in g4_value.iter().enumerate() {
            g4[i] = seat
                .as_f64()
                .ok_or_else(|| fail("test return_vector entries must be numbers".to_string()))?;
        }
        let advantage_f64 = doc
            .get("raw_advantage")
            .and_then(Value::as_f64)
            .ok_or_else(|| fail("doc lacks raw_advantage".to_string()))?;
        if !advantage_f64.is_finite() || f64::from(advantage_f64 as f32) != advantage_f64 {
            return Err(fail("test raw_advantage must be f32-exact finite".to_string()));
        }
        (Some(g4), Some(advantage_f64 as f32))
    } else {
        (None, None)
    };
    let obs = doc
        .get("actor_observation")
        .cloned()
        .ok_or_else(|| fail("doc lacks actor_observation".to_string()))?;
    let decision_id = field_str("decision_id")?;
    if obs_by_decision.insert(decision_id.clone(), obs).is_some() {
        return Err(fail(format!("duplicate decision {decision_id}")));
    }
    let digest_field = |name: &str| {
        let text = field_str(name)?;
        parse_digest_text(&text).map_err(|e| fail(e.0))?;
        Ok::<String, PyErr>(text)
    };
    let table_text = digest_field("action_table_hash")?;
    let row = crate::rl::core::GameRow {
        wall_id: row_wall,
        game_id: row_game,
        round_id: field_str("round_id")?,
        trajectory_step: field_u64("trajectory_step")?,
        actor,
        decision_id,
        sequence: field_u64("trajectory_step")?,
        row_id: [0u8; 32],
        round_index: 0,
        // Round-wind tile id (East=27): `wind_id` rejects the 0..3 wind
        // index, so the synthetic row must carry the tile, not the index.
        round_wind: 27,
        honba: 0,
        kyotaku: 0,
        dealer: 0,
        scores: [25000, 25000, 25000, 25000],
        live_remaining: 70,
        hand_number: 0,
        turn_actor: actor,
        phase_id: 1,
        concealed: vec![],
        drawn: None,
        dora5: [-1, -1, -1, -1, -1],
        ippatsu_bits: 0,
        furiten_id: 0,
        can_tsumo: true,
        can_riichi: false,
        legal_ids,
        legal_mask_bits: vec![],
        selected,
        selected_logp: logp_f64 as f32,
        value4: {
            let mut values = [0.0f32; 4];
            values[actor as usize] = v as f32;
            values
        },
        cursor_before: 0,
        cursor_after: 8,
        obs_hash: parse_digest_text(&field_str("observation_hash")?).map_err(|e| fail(e.0))?,
        event_prefix_len: 0,
        derivation: [0u8; 32],
        action_table_hash: parse_digest_text(&table_text).map_err(|e| fail(e.0))?,
        action_table_hash_hex: table_text,
        behavior_ckpt: parse_digest_text(&digest_field("behavior_policy_checkpoint_hash")?)
            .map_err(|e| fail(e.0))?,
        transition_provenance: parse_digest_text(&digest_field("transition_provenance_hash")?)
            .map_err(|e| fail(e.0))?,
        rules_id: "tenhou".to_string(),
        rules_hash: format!("sha256:{}", "01".repeat(32)),
        event_schema_hash: format!("sha256:{}", "02".repeat(32)),
        observation_schema_hash: format!("sha256:{}", "03".repeat(32)),
        packet_boundary_hash: format!("sha256:{}", "04".repeat(32)),
        return_g4,
        raw_advantage_f32,
    };
    Ok(row)
}

#[pyfunction]
fn seal_poll(py: Python<'_>, job_id: u64) -> PyResult<Vec<u8>> {
    py.detach(|| {
        match poll_seal(job_id).map_err(tape_err("seal_poll"))? {
            None => hydra_feed::canon::canonical_bytes_value(
                &serde_json::json!({"status": "running"}),
                "rl_artifact:poll",
            )
            .map_err(|e| tape_err("seal_poll")(TapeError(format!("poll JCS failed: {e:?}")))),
            Some(receipt) => {
                let body = receipt_json(
                    &receipt.artifact_digest,
                    receipt.row_count,
                    receipt.compressed_bytes,
                    receipt.uncompressed_bytes,
                )
                .map_err(|detail| artifact_err("seal_poll", detail))?;
                let doc = serde_json::json!({
                    "status": "done",
                    "receipt": serde_json::from_slice::<Value>(&body)
                        .map_err(|e| artifact_err("seal_poll", e.to_string()))?,
                });
                hydra_feed::canon::canonical_bytes_value(&doc, "rl_artifact:poll-done").map_err(
                    |e| tape_err("seal_poll")(TapeError(format!("poll JCS failed: {e:?}"))),
                )
            }
        }
    })
}

#[pyfunction]
fn seal_wait(py: Python<'_>, job_id: u64) -> PyResult<Vec<u8>> {
    py.detach(|| {
        let receipt = wait_seal(job_id).map_err(tape_err("seal_wait"))?;
        receipt_json(
            &receipt.artifact_digest,
            receipt.row_count,
            receipt.compressed_bytes,
            receipt.uncompressed_bytes,
        )
        .map_err(|detail| artifact_err("seal_wait", detail))
    })
}

#[pyfunction]
fn seal_busy(py: Python<'_>) -> PyResult<bool> {
    Ok(py.detach(sealer_busy))
}

/// GPU terminal inputs for one complete generation (cold, once): per-row
/// actor ids, per-row game indices, per-row behavior value quads, and
/// per-game canonical utility quads — no rows, no documents. The Torch
/// boundary computes raw advantages in f32 and returns them to
/// `freeze_terminal_advantages`; the native CPU formula stays the
/// validation oracle, never production compute.
#[pyfunction]
fn terminal_inputs(
    py: Python<'_>,
    tape_handle: u64,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    py.detach(|| {
        crate::rl::trajectory::gather_terminal_inputs(tape_handle)
            .map_err(tape_err("terminal_inputs"))
    })
}

/// Freeze GPU-computed advantages: validates the entire buffer then
/// attaches frozen A immutably, exactly once, before any seal.
#[pyfunction]
fn freeze_terminal_advantages(
    py: Python<'_>,
    tape_handle: u64,
    rawa_f32_le: Vec<u8>,
) -> PyResult<Vec<u8>> {
    py.detach(|| {
        let rows = crate::rl::trajectory::freeze_terminal_advantages(tape_handle, &rawa_f32_le)
            .map_err(tape_err("freeze_terminal_advantages"))?;
        let doc = serde_json::json!({"rows_frozen": rows});
        hydra_feed::canon::canonical_bytes_value(&doc, "rl_artifact:freeze-receipt")
            .map_err(|e| tape_err("freeze_terminal_advantages")(TapeError(format!("{e:?}"))))
    })
}

/// Synchronous production seal (quiesce/drain and single-threaded tooling;
/// the continuous loop uses begin/poll/wait).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn seal_sync(
    py: Python<'_>,
    tape_handle: u64,
    header_doc: Vec<u8>,
    source_ledger: Vec<String>,
    eval_ledger: Vec<String>,
    expect_bc: bool,
    artifact_path: String,
    metas_path: String,
) -> PyResult<Vec<u8>> {
    let context =
        seal_context_from_json(&header_doc, source_ledger, eval_ledger, expect_bc)
            .map_err(|detail| artifact_err("seal_sync", detail))?;
    py.detach(|| {
        let receipt = seal_from_tape(
            tape_handle,
            &context,
            &PathBuf::from(artifact_path),
            &PathBuf::from(metas_path),
        )
        .map_err(tape_err("seal_sync"))?;
        receipt_json(
            &receipt.artifact_digest,
            receipt.row_count,
            receipt.compressed_bytes,
            receipt.uncompressed_bytes,
        )
        .map_err(|detail| artifact_err("seal_sync", detail))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn validate_artifact(
    py: Python<'_>,
    artifact_path: String,
    metas_path: String,
    header_doc: Vec<u8>,
    source_ledger: Vec<String>,
    eval_ledger: Vec<String>,
    expect_bc: bool,
) -> PyResult<Vec<u8>> {
    let context =
        seal_context_from_json(&header_doc, source_ledger, eval_ledger, expect_bc)
            .map_err(|detail| artifact_err("validate_artifact", detail))?;
    py.detach(|| {
        let metas_bytes = std::fs::read(&metas_path)
            .map_err(|e| artifact_err("validate_artifact", format!("metas unreadable: {e}")))?;
        let receipt = validate_artifact_file(&PathBuf::from(artifact_path), &metas_bytes, &context)
            .map_err(tape_err("validate_artifact"))?;
        receipt_json(
            &receipt.artifact_digest,
            receipt.row_count,
            receipt.compressed_bytes,
            receipt.uncompressed_bytes,
        )
        .map_err(|detail| artifact_err("validate_artifact", detail))
    })
}

/// Boundary check for prepare paths: verifies a header digest equals the
/// JCS digest of the closed row-derivation document (never an arbitrary
/// digest). Returns the canonical binding receipt. The seal itself binds
/// rows to the verified header; composition stays upstream.
#[pyfunction]
fn verify_row_derivation(
    py: Python<'_>,
    header_digest: String,
    derivation_doc: Vec<u8>,
) -> PyResult<Vec<u8>> {
    py.detach(|| {
        crate::rl::artifact::verify_row_derivation(&header_digest, &derivation_doc)
            .map_err(tape_err("verify_row_derivation"))?;
        let receipt = serde_json::json!({"row_derivation_hash": header_digest});
        hydra_feed::canon::canonical_bytes_value(&receipt, "rl_artifact:derivation-receipt")
            .map_err(|e| tape_err("verify_row_derivation")(TapeError(format!("{e:?}"))))
    })
}


#[pyfunction]
fn checkpoint_receipt(
    py: Python<'_>,
    root_dir: String,
    locator: String,
    manifest_doc: Vec<u8>,
) -> PyResult<Vec<u8>> {
    py.detach(|| {
        ordinary_checkpoint_receipt(&PathBuf::from(root_dir), &locator, &manifest_doc)
            .map_err(tape_err("checkpoint_receipt"))
    })
}

#[pyfunction]
fn publish_control(
    py: Python<'_>,
    root_dir: String,
    control_doc: Vec<u8>,
    receipts: Vec<Vec<u8>>,
    expected_parent: Option<String>,
    blobs_doc: Vec<u8>,
) -> PyResult<Vec<u8>> {
    py.detach(|| {
        let blobs_value: Value = serde_json::from_slice(&blobs_doc)
            .map_err(|e| tape_err("publish_control")(TapeError(format!("blobs not JSON: {e}"))))?;
        let blobs_array = blobs_value
            .as_array()
            .ok_or_else(|| tape_err("publish_control")(TapeError("blobs must be an array".to_string())))?;
        let mut blobs = Vec::with_capacity(blobs_array.len());
        for entry in blobs_array {
            let digest = entry
                .get("digest")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    tape_err("publish_control")(TapeError("blob lacks digest".to_string()))
                })?
                .to_string();
            let locator = entry
                .get("locator")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    tape_err("publish_control")(TapeError("blob lacks locator".to_string()))
                })?
                .to_string();
            blobs.push((digest, locator));
        }
        crate::rl::snapshot::publish_control(
            &PathBuf::from(root_dir),
            &control_doc,
            &receipts,
            expected_parent.as_deref(),
            &blobs,
        )
        .map_err(tape_err("publish_control"))
    })
}

#[pyfunction]
fn read_tip(py: Python<'_>, root_dir: String) -> PyResult<Vec<u8>> {
    py.detach(|| crate::rl::snapshot::read_tip(&PathBuf::from(root_dir)).map_err(tape_err("read_tip")))
}

#[pyfunction]
fn stage_native_snapshot(py: Python<'_>, root_dir: String, game_doc: Vec<u8>) -> PyResult<Vec<u8>> {
    py.detach(|| {
        crate::rl::snapshot::stage_native_snapshot(&PathBuf::from(root_dir), &game_doc)
            .map_err(tape_err("stage_native_snapshot"))
    })
}

#[pyfunction]
fn read_native_snapshot(
    py: Python<'_>,
    root_dir: String,
    game_id: String,
    digest: String,
) -> PyResult<Vec<u8>> {
    py.detach(|| {
        crate::rl::snapshot::read_native_snapshot(&PathBuf::from(root_dir), &game_id, &digest)
            .map_err(tape_err("read_native_snapshot"))
    })
}

#[pyfunction]
fn commit_blob(py: Python<'_>, path: String) -> PyResult<()> {
    py.detach(|| crate::rl::snapshot::commit_blob(&PathBuf::from(path)).map_err(tape_err("commit_blob")))
}

#[pyfunction]
fn locate(root_dir: String, locator: String) -> PyResult<String> {
    join_root(&PathBuf::from(root_dir), &locator)
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(tape_err("locate"))
}

/// Register the `rl_artifact` submodule on the TOP `_native` module
/// (stable path `hydra2._native.rl_artifact`; the `rl` submodule registrar
/// owned by NativeWall calls this with its received top module).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "rl_artifact")?;
    sub.add_function(wrap_pyfunction!(seal_begin, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seal_poll, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seal_wait, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seal_busy, &sub)?)?;
    sub.add_function(wrap_pyfunction!(seal_sync, &sub)?)?;
    sub.add_function(wrap_pyfunction!(stage_test_tape, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_artifact, &sub)?)?;
    replay::register(&sub)?;
    sub.add_function(wrap_pyfunction!(verify_row_derivation, &sub)?)?;
    sub.add_function(wrap_pyfunction!(terminal_inputs, &sub)?)?;
    sub.add_function(wrap_pyfunction!(freeze_terminal_advantages, &sub)?)?;
    sub.add_function(wrap_pyfunction!(checkpoint_receipt, &sub)?)?;
    sub.add_function(wrap_pyfunction!(publish_control, &sub)?)?;
    sub.add_function(wrap_pyfunction!(read_tip, &sub)?)?;
    sub.add_function(wrap_pyfunction!(stage_native_snapshot, &sub)?)?;
    sub.add_function(wrap_pyfunction!(read_native_snapshot, &sub)?)?;
    sub.add_function(wrap_pyfunction!(commit_blob, &sub)?)?;
    sub.add_function(wrap_pyfunction!(locate, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}
