//! 84-probe parity: Rust wall-less rows vs the Python oracle, decision-for-decision.
//!
//! Env-gated (the 3.7 GB oracle dump never enters the crate):
//! - `HYDRA2_REPLAY_PROBE`: dir with `inputs/`, `manifest.json`,
//!   `oracle-compact.jsonl`, `oracle-quarantine.json` (Step-0 materializer);
//! - `HYDRA2_REPLAY_TABLE`: `configs/contracts/action_table_v1.json`.
//! Absent env skips gracefully (nextest stays green); set env runs the full
//! join and prints the parity table.
//!
//! Iteration-1 assertions (strict): every Rust row id joins an oracle row,
//! every oracle row on mutually-ok games joins a Rust row, and the
//! quarantine game sets agree. Everything else is classified and counted.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;

use hydra_shard::decisions::ActionTable;
use hydra_shard::tile::mjai_string_of;
use hydra_shard::replay_game_text;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// Slice-1 scratch discipline (runs even when the probe env is unset, so
/// nextest stays green either way; each check is vacuous when its env
/// inputs are absent):
/// - probe inputs live on scratch: `HYDRA2_REPLAY_PROBE` (when set) must
///   sit under the platform temp dir (`/tmp` here; honors `TMPDIR` like
///   `hydra2.config.artifact_root`), never inside the repo or a corpus mount.
/// - `HYDRA2_ARTIFACT_ROOT` (when set) must sit outside the raw corpus
///   roots (`HYDRA2_DATA_ROOT`, `HYDRA2_TENHOU_MOUNT` when set): the corpus
///   is read-only (D-017) and artifacts never land inside raw mounts.
fn assert_scratch_discipline() {
    use std::path::Path;
    if let Some(probe) = env("HYDRA2_REPLAY_PROBE") {
        let tmp = std::env::temp_dir();
        assert!(
            Path::new(&probe).starts_with(&tmp),
            "HYDRA2_REPLAY_PROBE must sit under scratch temp dir {}: {probe}",
            tmp.display(),
        );
    }
    if let Some(artifact) = env("HYDRA2_ARTIFACT_ROOT") {
        let artifact_path = Path::new(&artifact);
        for key in ["HYDRA2_DATA_ROOT", "HYDRA2_TENHOU_MOUNT"] {
            if let Some(raw) = env(key) {
                let raw_path = Path::new(&raw);
                assert!(
                    !artifact_path.starts_with(&raw_path)
                        && !raw_path.starts_with(artifact_path),
                    "{key} raw root must sit outside HYDRA2_ARTIFACT_ROOT: artifact={artifact} raw={raw}",
                );
            }
        }
    }
}

#[derive(Debug, Default)]
struct ClassCounts {
    rows_joined: u64,
    seat_match: u64,
    phase_match: u64,
    chosen_id_match: u64,
    chosen_kind_match: u64,
    concealed_match: u64,
    drawn_match: u64,
    dora_match: u64,
    wall_match: u64,
    history_match: u64,
    mask_exact: u64,
    mask_missing_sum: u64,
    mask_extra_sum: u64,
    unresolved_chosen: u64,
    kind_mismatch: BTreeMap<(String, String), u64>,
    unconcealed_samples: Vec<String>,
    chosen_id_mismatch_kind: BTreeMap<String, u64>,
    chosen_id_samples: Vec<String>,
    history_samples: Vec<String>,
    mask_missing_kind: BTreeMap<String, u64>,
    mask_extra_kind: BTreeMap<String, u64>,
    mask_missing_phase: BTreeMap<String, u64>,
    mask_extra_phase: BTreeMap<String, u64>,
    extra_discard_postclaim: u64,
    extra_discard_clean: u64,
    extra_discard_samples: Vec<String>,
    missing_discard_samples: Vec<String>,
    hist_forensics: Vec<String>,
}

#[test]
fn parity_84_probe() {
    assert_scratch_discipline();
    let (Some(probe), Some(table_path)) = (env("HYDRA2_REPLAY_PROBE"), env("HYDRA2_REPLAY_TABLE"))
    else {
        eprintln!("parity_84: HYDRA2_REPLAY_PROBE/HYDRA2_REPLAY_TABLE unset, skipping");
        return;
    };
    let probe = std::path::PathBuf::from(probe);
    let table_text = fs::read_to_string(&table_path).expect("action table read");
    let table = ActionTable::load_json(&table_text).expect("action table load");
    let id2template: HashMap<u32, String> = {
        let doc: serde_json::Value = serde_json::from_str(&table_text).expect("table json");
        doc.pointer("/payload/actions")
            .and_then(|v| v.as_array())
            .expect("payload.actions")
            .iter()
            .enumerate()
            .map(|(id, entry)| (id as u32, entry.to_string()))
            .collect()
    };
    let id2kind: HashMap<u32, String> = id2template
        .iter()
        .map(|(id, entry)| {
            let kind = serde_json::from_str::<serde_json::Value>(entry)
                .ok()
                .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(|s| s.to_string()))
                .unwrap_or_else(|| "?".to_string());
            (*id, kind)
        })
        .collect();

    // Manifest: file -> (game_id, object_id).
    let manifest_text =
        fs::read_to_string(probe.join("manifest.json")).expect("manifest read");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("manifest json");
    let mut file_meta: BTreeMap<String, (String, String)> = BTreeMap::new();
    for entry in manifest.get("games").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
        let file = entry.get("file").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let game_id = entry.get("game_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let object_id = entry.get("object_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        file_meta.insert(file, (game_id, object_id));
    }

    // Replay every input game through the Rust driver.
    let mut rust_rows: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut rust_quarantine: BTreeMap<String, String> = BTreeMap::new();
    let mut input_files: Vec<String> = file_meta.keys().cloned().collect();
    input_files.sort();
    for file in &input_files {
        let text = fs::read_to_string(probe.join("inputs").join(file)).expect("input read");
        let (game_id, object_id) = file_meta[file].clone();
        match replay_game_text(&text, &object_id, &table) {
            Ok(rows) => {
                for row in rows {
                    let value = serde_json::to_value(&row).expect("row json");
                    rust_rows.insert(row.decision_id.clone(), value);
                }
                let _ = game_id;
            }
            Err(quarantine) => {
                rust_quarantine.insert(file.clone(), quarantine.reason_code.clone());
                eprintln!("rust quarantine {file}: {}", quarantine.reason_code);
            }
        }
    }

    // Stream the oracle compact rows.
    let compact =
        fs::read_to_string(probe.join("oracle-compact.jsonl")).expect("oracle compact read");
    let mut oracle_rows: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for line in compact.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line).expect("compact json");
        let id = row.get("d").and_then(|v| v.as_str()).unwrap_or("").to_string();
        oracle_rows.insert(id, row);
    }

    // Oracle quarantine game ids + per-game reason codes (S4: sets AND codes
    // must agree, not just sets). The Step-0 materializer may record a bare
    // code string or a richer object/detail; `quarantine_code` normalizes
    // either shape onto the closed Slice-3 taxonomy.
    let oracle_q_text =
        fs::read_to_string(probe.join("oracle-quarantine.json")).expect("oracle quarantine read");
    let oracle_q: serde_json::Value =
        serde_json::from_str(&oracle_q_text).expect("oracle quarantine json");
    let oracle_q_games: BTreeSet<String> = oracle_q
        .as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    let oracle_q_codes: BTreeMap<String, String> = oracle_q
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(game, value)| (game.clone(), quarantine_code(value)))
                .collect()
        })
        .unwrap_or_default();
    let rust_q_games: BTreeSet<String> = rust_quarantine
        .keys()
        .map(|file| file_meta[file].0.clone())
        .collect();
    let rust_q_codes: BTreeMap<String, String> = rust_quarantine
        .iter()
        .map(|(file, code)| (file_meta[file].0.clone(), code.clone()))
        .collect();

    let mut classes = ClassCounts::default();
    let mut rust_only: Vec<String> = Vec::new();
    for (id, rust) in &rust_rows {
        match oracle_rows.get(id) {
            None => rust_only.push(id.clone()),
            Some(oracle) => classify(id, rust, oracle, &id2kind, &id2template, &mut classes),
        }
    }
    let mut oracle_only: Vec<String> = Vec::new();
    for id in oracle_rows.keys() {
        if !rust_rows.contains_key(id) {
            // Oracle rows on Rust-quarantined games have no Rust counterpart
            // by design (whole-game quarantine, like the oracle).
            let game = id.split(":d").next().unwrap_or("").to_string();
            if !rust_q_games.contains(&game) {
                oracle_only.push(id.clone());
            }
        }
    }

    println!("=== parity_84 ===");
    println!("rust rows: {} oracle rows: {}", rust_rows.len(), oracle_rows.len());
    println!("rust quarantined files: {} oracle quarantined games: {}", rust_quarantine.len(), oracle_q_games.len());
    println!("oracle quarantine games: {oracle_q_games:?}");
    println!("rust quarantine games: {rust_q_games:?}");
    println!("oracle quarantine codes: {oracle_q_codes:?}");
    println!("rust quarantine codes: {rust_q_codes:?}");
    println!("joined rows: {}", classes.rows_joined);
    println!("seat match: {}", classes.seat_match);
    println!("phase match: {}", classes.phase_match);
    println!("chosen-id exact: {}", classes.chosen_id_match);
    println!("chosen-kind match: {}", classes.chosen_kind_match);
    println!("concealed-strings exact: {}", classes.concealed_match);
    println!("drawn exact: {}", classes.drawn_match);
    println!("dora exact: {}", classes.dora_match);
    println!("wall exact: {}", classes.wall_match);
    println!("history-kinds exact: {}", classes.history_match);
    println!("mask exact: {}", classes.mask_exact);
    println!("mask missing-bit total: {}", classes.mask_missing_sum);
    println!("mask extra-bit total: {}", classes.mask_extra_sum);
    println!("unresolved chosen: {}", classes.unresolved_chosen);
    println!("kind mismatches (oracle -> rust):");
    for ((oracle_kind, rust_kind), count) in &classes.kind_mismatch {
        println!("  {count:6} {oracle_kind} -> {rust_kind}");
    }
    println!("chosen-id mismatch by oracle kind:");
    for (kind, count) in &classes.chosen_id_mismatch_kind {
        println!("  {count:6} {kind}");
    }
    for sample in classes.chosen_id_samples.iter().take(15) {
        println!("chosen-id sample: {sample}");
    }
    println!("history mismatch samples:");
    for sample in &classes.history_samples {
        println!("  history sample: {sample}");
    }
    println!("mask missing-bit kinds:");
    for (kind, count) in &classes.mask_missing_kind {
        println!("  {count:6} {kind}");
    }
    println!("mask extra-bit kinds:");
    for (kind, count) in &classes.mask_extra_kind {
        println!("  {count:6} {kind}");
    }
    println!("mask missing by phase: {:?}", classes.mask_missing_phase);
    println!("mask extra by phase: {:?}", classes.mask_extra_phase);
    println!("extra discards post-claim: {} clean: {}", classes.extra_discard_postclaim, classes.extra_discard_clean);
    for sample in classes.extra_discard_samples.iter().take(8) {
        println!("extra-discard sample: {sample}");
    }
    for sample in classes.missing_discard_samples.iter().take(8) {
        println!("missing-discard sample: {sample}");
    }
    let mut mismatch_ids: Vec<String> = Vec::new();
    for (id, rust) in &rust_rows {
        if let Some(oracle) = oracle_rows.get(id) {
            let rust_hk = str_list(rust.get("history_kinds").unwrap_or(&serde_json::Value::Null));
            let oracle_hk = str_list(oracle.get("hk").unwrap_or(&serde_json::Value::Null));
            if rust_hk != oracle_hk {
                mismatch_ids.push(id.clone());
            }
        }
    }
    for line in history_forensics(&probe, &file_meta, &rust_rows, &mismatch_ids) {
        println!("{line}");
    }
    // Per-game identity. All non-mask/history fields matched 100% globally,
    // so a row is fully identical iff its mask and history are exact.
    let mut per_game: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut clean_extra_by_kind: BTreeMap<String, u64> = BTreeMap::new();
    for (id, rust) in &rust_rows {
        if let Some(oracle) = oracle_rows.get(id) {
            let game = id.split(":d").next().unwrap_or("").to_string();
            let entry = per_game.entry(game).or_insert((0, 0));
            entry.0 += 1;
            let (mask_exact, hist_exact) = mask_hist_exact(rust, oracle);
            if mask_exact && hist_exact {
                entry.1 += 1;
            } else if !mask_exact {
                let rust_hk = str_list(rust.get("history_kinds").unwrap_or(&serde_json::Value::Null));
                let postclaim = rust_hk
                    .iter()
                    .any(|k| matches!(k.as_str(), "chi" | "pon" | "daiminkan" | "ankan" | "kakan"));
                if !postclaim {
                    let rust_kind = rust
                        .get("chosen")
                        .and_then(|v| v.get("kind"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();
                    for bit in mask_extra_ids(rust, oracle) {
                        if id2kind.get(&(bit as u32)).map(|k| k == "discard").unwrap_or(false) {
                            *clean_extra_by_kind.entry(rust_kind.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }
    let identical_games = per_game.values().filter(|(t, i)| t == i).count();
    println!("games fully identical: {identical_games}/{} (quarantined: {})", per_game.len(), rust_q_games.len());
    for (game, (total, identical)) in &per_game {
        if total != identical {
            println!("  game {game}: {identical}/{total} identical");
        }
    }
    println!("clean extra discards by rust chosen kind: {clean_extra_by_kind:?}");
    println!("rust-only ids (first 10): {:?}", &rust_only[..rust_only.len().min(10)]);

    assert!(rust_only.is_empty(), "rust rows without oracle ids: {}", rust_only.len());
    assert!(oracle_only.is_empty(), "oracle rows without rust ids: {}", oracle_only.len());
    assert_eq!(rust_q_games, oracle_q_games, "quarantine game sets differ");
    assert_eq!(rust_q_codes, oracle_q_codes, "quarantine game codes differ");
    // S6 frozen-oracle gate (PG-S6 strictest): full identity on mutually-ok
    // games (every named field 100%, zero mask residual, zero unresolved),
    // all 78 ok games fully identical, and the frozen quarantine set (same
    // 6 double-ron). Any oracle move voids parity (re-pin and re-run).
    assert_eq!(classes.rows_joined, 42553, "frozen joined-row count");
    assert_eq!(classes.seat_match, classes.rows_joined, "seat identity");
    assert_eq!(classes.phase_match, classes.rows_joined, "phase identity");
    assert_eq!(classes.chosen_id_match, classes.rows_joined, "chosen-id identity");
    assert_eq!(classes.chosen_kind_match, classes.rows_joined, "chosen-kind identity");
    assert_eq!(classes.concealed_match, classes.rows_joined, "concealed identity");
    assert_eq!(classes.drawn_match, classes.rows_joined, "drawn identity");
    assert_eq!(classes.dora_match, classes.rows_joined, "dora identity");
    assert_eq!(classes.wall_match, classes.rows_joined, "wall identity");
    assert_eq!(classes.history_match, classes.rows_joined, "history identity");
    assert_eq!(classes.mask_exact, classes.rows_joined, "mask identity");
    assert_eq!(classes.mask_missing_sum, 0, "mask burn-down (missing)");
    assert_eq!(classes.mask_extra_sum, 0, "mask burn-down (extra)");
    assert_eq!(classes.unresolved_chosen, 0, "unresolved chosen");
    assert_eq!(identical_games, 78, "all ok games fully identical");
    assert_eq!(rust_q_games.len(), 6, "frozen quarantine count");
    assert!(
        rust_q_codes.values().all(|c| c == "double-ron"),
        "frozen quarantine codes: {rust_q_codes:?}"
    );
}

/// Normalize one `oracle-quarantine.json` value onto the closed Slice-3
/// reason-code taxonomy. Accepts a bare code string (`"double-ron"`) or a
/// richer object/detail (`{"reason_code": ...}` / ContractError text such
/// as `"... double ron on one discard ..."`); unknown shapes fall back to
/// the raw text so new codes fail loudly instead of joining silently.
fn quarantine_code(value: &serde_json::Value) -> String {
    const KNOWN: [&str; 12] = [
        "framing",
        "wall-bearing",
        "bare-dora",
        "double-ron",
        "unknown-event",
        "tile-conservation",
        "turn-order",
        "claim-no-offer",
        "draw-past-wall",
        "kyushu-ambiguous",
        "unmapped-ryukyoku-reason",
        "action-id-unresolved",
    ];
    let mut texts = Vec::new();
    match value {
        serde_json::Value::String(s) => texts.push(s.clone()),
        serde_json::Value::Object(map) => {
            for key in ["reason_code", "code", "reason", "detail", "message", "error"] {
                if let Some(s) = map.get(key).and_then(|v| v.as_str()) {
                    texts.push(s.to_string());
                }
            }
            if texts.is_empty() {
                texts.push(serde_json::to_string(value).unwrap_or_default());
            }
        }
        _ => texts.push(serde_json::to_string(value).unwrap_or_default()),
    }
    for text in &texts {
        let folded = text.to_lowercase().replace('_', "-");
        for code in KNOWN {
            let spaced = code.replace('-', " ");
            if folded.contains(code) || folded.contains(&spaced) {
                return code.to_string();
            }
        }
    }
    texts.into_iter().next().unwrap_or_default().chars().take(100).collect()
}

fn str_list(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect()
}

fn id_strings_in_order(ids: &[serde_json::Value]) -> Vec<String> {
    // Order-preserving: the oracle hand is already ascending by physical
    // id, and the Rust concealed strings are emitted in the same order.
    // Sorting rendered strings would break id order (lexicographic sort
    // groups suits, not ids), so compare sequences as emitted.
    ids.iter()
        .filter_map(|v| v.as_u64())
        .filter_map(|id| mjai_string_of(id as u8).ok())
        .collect()
}

fn classify(
    id: &str,
    rust: &serde_json::Value,
    oracle: &serde_json::Value,
    id2kind: &HashMap<u32, String>,
    id2template: &HashMap<u32, String>,
    classes: &mut ClassCounts,
) {
    classes.rows_joined += 1;
    let seat = rust.get("seat").and_then(|v| v.as_u64()).unwrap_or(99);
    if Some(seat) == oracle.get("s").and_then(|v| v.as_u64()) {
        classes.seat_match += 1;
    }
    let phase = rust.get("phase").and_then(|v| v.as_str()).unwrap_or("?");
    if Some(phase) == oracle.get("ph").and_then(|v| v.as_str()) {
        classes.phase_match += 1;
    } else {
        eprintln!("phase mismatch {id}: rust {phase} oracle {:?}", oracle.get("ph"));
    }
    let rust_chosen = rust.get("chosen_action_id").and_then(|v| v.as_u64());
    let oracle_chosen = oracle.get("a").and_then(|v| v.as_u64());
    let oracle_kind = oracle_chosen
        .and_then(|a| id2kind.get(&(a as u32)))
        .cloned()
        .unwrap_or_else(|| "?".to_string());
    if rust_chosen == oracle_chosen {
        classes.chosen_id_match += 1;
    } else {
        *classes.chosen_id_mismatch_kind.entry(oracle_kind.clone()).or_insert(0) += 1;
        if classes.chosen_id_samples.len() < 15 {
            let oracle_template = oracle_chosen
                .and_then(|a| id2template.get(&(a as u32)))
                .cloned()
                .unwrap_or_else(|| "null".to_string());
            classes.chosen_id_samples.push(format!(
                "{id} oracle {} rust {:?} rust-chosen {:?}",
                oracle_template,
                rust_chosen,
                rust.get("chosen")
            ));
        }
    }
    let rust_kind = rust
        .get("chosen")
        .and_then(|v| v.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    // Canonical kind names use snake_case on both sides.
    let oracle_kind_norm = oracle_kind.clone();
    if rust_kind == oracle_kind_norm {
        classes.chosen_kind_match += 1;
    } else {
        *classes.kind_mismatch.entry((oracle_kind, rust_kind)).or_insert(0) += 1;
    }
    if rust.get("chosen_action_id").map(|v| v.is_null()).unwrap_or(true) {
        classes.unresolved_chosen += 1;
    }
    let rust_concealed = str_list(rust.get("concealed_hand").unwrap_or(&serde_json::Value::Null));
    let oracle_hand = oracle.get("hand").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let oracle_concealed = id_strings_in_order(&oracle_hand);
    if rust_concealed == oracle_concealed {
        classes.concealed_match += 1;
    } else if classes.unconcealed_samples.len() < 20 {
        classes.unconcealed_samples.push(format!(
            "{id} rust {rust_concealed:?} oracle {oracle_concealed:?}"
        ));
    }
    let rust_drawn = rust.get("drawn_tile");
    let oracle_drawn = oracle.get("drawn");
    let drawn_equal = match (rust_drawn, oracle_drawn) {
        (Some(serde_json::Value::String(r)), Some(serde_json::Value::Number(o))) => {
            o.as_u64().and_then(|id| mjai_string_of(id as u8).ok()).as_deref() == Some(r.as_str())
        }
        (Some(serde_json::Value::Null) | None, Some(serde_json::Value::Null) | None) => true,
        _ => false,
    };
    if drawn_equal {
        classes.drawn_match += 1;
    } else if classes.unconcealed_samples.len() < 40 {
        classes.unconcealed_samples.push(format!("{id} drawn rust {rust_drawn:?} oracle {oracle_drawn:?}"));
    }
    let rust_dora = rust.get("dora_indicators").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let oracle_dora = oracle.get("dora").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let dora_equal = rust_dora.len() == oracle_dora.len()
        && rust_dora.iter().zip(oracle_dora.iter()).all(|(r, o)| match (r, o) {
            (serde_json::Value::Null, serde_json::Value::Number(n)) => n.as_i64() == Some(-1),
            (serde_json::Value::String(s), serde_json::Value::Number(n)) => {
                n.as_u64().and_then(|id| mjai_string_of(id as u8).ok()).as_deref() == Some(s.as_str())
            }
            _ => false,
        });
    if dora_equal {
        classes.dora_match += 1;
    }
    if rust.get("wall_remaining").and_then(|v| v.as_u64()) == oracle.get("wall").and_then(|v| v.as_u64()) {
        classes.wall_match += 1;
    }
    let rust_hk = str_list(rust.get("history_kinds").unwrap_or(&serde_json::Value::Null));
    let oracle_hk = str_list(oracle.get("hk").unwrap_or(&serde_json::Value::Null));
    if rust_hk == oracle_hk {
        classes.history_match += 1;
    } else if classes.history_samples.len() < 10 {
        let tail = |hk: &[String]| hk.iter().rev().take(6).rev().cloned().collect::<Vec<_>>();
        classes.history_samples.push(format!(
            "{id} len {} vs {} rust-tail {:?} oracle-tail {:?}",
            rust_hk.len(),
            oracle_hk.len(),
            tail(&rust_hk),
            tail(&oracle_hk)
        ));
    }
    let rust_mask: BTreeSet<u64> = rust
        .get("legal_mask")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    let oracle_mask: BTreeSet<u64> = oracle
        .get("mask")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    if rust_mask == oracle_mask {
        classes.mask_exact += 1;
    } else {
        let missing: Vec<u64> = oracle_mask.difference(&rust_mask).copied().collect();
        let extra: Vec<u64> = rust_mask.difference(&oracle_mask).copied().collect();
        classes.mask_missing_sum += missing.len() as u64;
        classes.mask_extra_sum += extra.len() as u64;
        // Post-claim rows (any claim kind already in this seat's history)
        // vs clean rows: separates kuikae-class extras from the rest.
        let postclaim = rust_hk
            .iter()
            .any(|k| matches!(k.as_str(), "chi" | "pon" | "daiminkan" | "ankan" | "kakan"));
        for bit in &missing {
            let kind = id2kind.get(&(*bit as u32)).cloned().unwrap_or_else(|| "?".to_string());
            *classes.mask_missing_kind.entry(kind.clone()).or_insert(0) += 1;
            if kind == "discard" && classes.missing_discard_samples.len() < 8 {
                let tile = id2template
                    .get(&(*bit as u32))
                    .cloned()
                    .unwrap_or_else(|| "?".to_string());
                classes.missing_discard_samples.push(format!("{id} {tile}"));
            }
        }
        for bit in &extra {
            let kind = id2kind.get(&(*bit as u32)).cloned().unwrap_or_else(|| "?".to_string());
            *classes.mask_extra_kind.entry(kind.clone()).or_insert(0) += 1;
            if kind == "discard" {
                if postclaim {
                    classes.extra_discard_postclaim += 1;
                } else {
                    classes.extra_discard_clean += 1;
                }
                if classes.extra_discard_samples.len() < 8 {
                    let tile = id2template
                        .get(&(*bit as u32))
                        .cloned()
                        .unwrap_or_else(|| "?".to_string());
                    classes.extra_discard_samples.push(format!(
                        "{id} postclaim={postclaim} {tile}"
                    ));
                }
            }
        }
        *classes.mask_missing_phase.entry(phase.to_string()).or_insert(0) += missing.len() as u64;
        *classes.mask_extra_phase.entry(phase.to_string()).or_insert(0) += extra.len() as u64;
    }
}

fn norm_mjai(pai: &str) -> String {
    match pai {
        "5mr" | "0m" => "5m".to_string(),
        "5pr" | "0p" => "5p".to_string(),
        "5sr" | "0s" => "5s".to_string(),
        other => other.to_string(),
    }
}

fn chi_patterns_offered(counts: &std::collections::HashMap<String, usize>, tile: &str) -> bool {
    if tile.len() != 2 {
        return false;
    }
    let bytes = tile.as_bytes();
    if !bytes[0].is_ascii_digit() || !"mps".contains(bytes[1] as char) {
        return false;
    }
    let value = (bytes[0] - b'0') as i64;
    let suit = bytes[1] as char;
    if !(1..=9).contains(&value) {
        return false;
    }
    let mut patterns: Vec<(i64, i64)> = Vec::new();
    if value >= 3 {
        patterns.push((value - 2, value - 1));
    }
    if (2..=8).contains(&value) {
        patterns.push((value - 1, value + 1));
    }
    if value <= 7 {
        patterns.push((value + 1, value + 2));
    }
    patterns.into_iter().any(|(low, high)| {
        counts.get(&format!("{low}{suit}")).copied().unwrap_or(0) > 0
            && counts.get(&format!("{high}{suit}")).copied().unwrap_or(0) > 0
    })
}

/// Window forensics for history-mismatch rows: re-walk each affected kyoku
/// with a string-level mini-tracker and classify every discard window as
/// pon/chi-possible (a thin miss would be a driver bug) or not (engine
/// win-eval residual). Reports aggregates only.
fn history_forensics(
    probe: &std::path::Path,
    file_meta: &BTreeMap<String, (String, String)>,
    rust_rows: &BTreeMap<String, serde_json::Value>,
    mismatch_ids: &[String],
) -> Vec<String> {
    use std::collections::{HashMap, HashSet};
    let mut game_of_file: HashMap<String, String> = HashMap::new();
    for (file, (game_id, _)) in file_meta {
        game_of_file.insert(game_id.clone(), file.clone());
    }
    // Affected (file, round_idx) kyokus.
    let mut kyokus: HashSet<(String, i64)> = HashSet::new();
    for id in mismatch_ids {
        let game = id.split(":d").next().unwrap_or("").to_string();
        let round_idx = rust_rows
            .get(id)
            .and_then(|r| r.get("round_id"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.rsplit('h').next())
            .and_then(|n| n.parse::<i64>().ok())
            .unwrap_or(-1);
        if let Some(file) = game_of_file.get(&game) {
            kyokus.insert((file.clone(), round_idx));
        }
    }
    let mut kyoku_count = 0u64;
    let mut window_count = 0u64;
    let mut ponchi_possible = 0u64;
    let mut riichi_present = 0u64;
    let mut claim_follows = 0u64;
    for (file, round_idx) in &kyokus {
        let text = match std::fs::read_to_string(probe.join("inputs").join(file)) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let events: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        // Slice this kyoku's events.
        let mut ordinal: i64 = -1;
        let mut slice: Vec<serde_json::Value> = Vec::new();
        for event in &events {
            let kind = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if kind == "start_kyoku" {
                ordinal += 1;
            }
            if ordinal == *round_idx
                && !matches!(kind, "start_kyoku" | "end_kyoku" | "end_game" | "endGame" | "game_end" | "end")
            {
                slice.push(event.clone());
            }
            if ordinal > *round_idx {
                break;
            }
        }
        if slice.is_empty() {
            continue;
        }
        kyoku_count += 1;
        let mut hands: [HashMap<String, usize>; 4] = Default::default();
        let mut riichi = [false; 4];
        let mut seen: i64 = -1;
        for event in &events {
            let kind = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if kind == "start_kyoku" {
                seen += 1;
                if seen == *round_idx {
                    if let Some(tehais) = event.get("tehais").and_then(|v| v.as_array()) {
                        for (seat, hand) in tehais.iter().enumerate().take(4) {
                            if let Some(tiles) = hand.as_array() {
                                for tile in tiles {
                                    if let Some(pai) = tile.as_str() {
                                        *hands[seat].entry(norm_mjai(pai)).or_insert(0) += 1;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
        let seat_of = |event: &serde_json::Value| -> Option<usize> {
            event.get("actor").and_then(|v| v.as_u64()).map(|s| s as usize).filter(|s| *s < 4)
        };
        for (pos, event) in slice.iter().enumerate() {
            let kind = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "tsumo" => {
                    if let (Some(seat), Some(pai)) =
                        (seat_of(event), event.get("pai").and_then(|v| v.as_str()))
                    {
                        *hands[seat].entry(norm_mjai(pai)).or_insert(0) += 1;
                    }
                }
                "dahai" => {
                    if let (Some(seat), Some(pai)) =
                        (seat_of(event), event.get("pai").and_then(|v| v.as_str()))
                    {
                        if let Some(count) = hands[seat].get_mut(&norm_mjai(pai)) {
                            *count = count.saturating_sub(1);
                        }
                        // Evaluate the window this discard opens.
                        let tile_norm = norm_mjai(pai);
                        let mut ponchi = false;
                        let mut riichi_any = false;
                        for responder in 0..4 {
                            if responder == seat {
                                continue;
                            }
                            if riichi[responder] {
                                riichi_any = true;
                                continue;
                            }
                            if hands[responder].get(&tile_norm).copied().unwrap_or(0) >= 2 {
                                ponchi = true;
                            }
                            if responder == (seat + 1) % 4
                                && chi_patterns_offered(&hands[responder], &tile_norm)
                            {
                                ponchi = true;
                            }
                        }
                        window_count += 1;
                        if ponchi {
                            ponchi_possible += 1;
                        }
                        if riichi_any {
                            riichi_present += 1;
                        }
                        // A logged claim on this discard proves the window.
                        let mut look = pos + 1;
                        while look < slice.len() {
                            let next_kind = slice[look]
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if matches!(next_kind, "dora" | "reach_accepted") {
                                look += 1;
                                continue;
                            }
                            if matches!(next_kind, "chi" | "pon" | "daiminkan") {
                                let target = slice[look]
                                    .get("target")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(9);
                                if target as usize == seat {
                                    claim_follows += 1;
                                }
                            }
                            break;
                        }
                    }
                }
                "reach" => {
                    if let Some(seat) = seat_of(event) {
                        riichi[seat] = true;
                    }
                }
                "chi" | "pon" | "daiminkan" => {
                    if let Some(seat) = seat_of(event) {
                        if let Some(consumed) =
                            event.get("consumed").and_then(|v| v.as_array())
                        {
                            for tile in consumed {
                                if let Some(pai) = tile.as_str() {
                                    if let Some(count) =
                                        hands[seat].get_mut(&norm_mjai(pai))
                                    {
                                        *count = count.saturating_sub(1);
                                    }
                                }
                            }
                        }
                    }
                }
                "ankan" => {
                    if let Some(seat) = seat_of(event) {
                        if let Some(consumed) =
                            event.get("consumed").and_then(|v| v.as_array())
                        {
                            for tile in consumed {
                                if let Some(pai) = tile.as_str() {
                                    if let Some(count) =
                                        hands[seat].get_mut(&norm_mjai(pai))
                                    {
                                        *count = count.saturating_sub(1);
                                    }
                                }
                            }
                        }
                    }
                }
                "kakan" => {
                    if let Some(seat) = seat_of(event) {
                        if let Some(pai) = event.get("pai").and_then(|v| v.as_str()) {
                            if let Some(count) = hands[seat].get_mut(&norm_mjai(pai)) {
                                *count = count.saturating_sub(1);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    vec![
        format!("forensics kyokus: {kyoku_count} windows: {window_count}"),
        format!("forensics ponchi-possible windows: {ponchi_possible}"),
        format!("forensics windows with riichi responder: {riichi_present}"),
        format!("forensics windows followed by logged claim: {claim_follows}"),
    ]
}

fn mask_sets(rust: &serde_json::Value, oracle: &serde_json::Value) -> (BTreeSet<u64>, BTreeSet<u64>) {
    let rust_mask: BTreeSet<u64> = rust
        .get("legal_mask")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    let oracle_mask: BTreeSet<u64> = oracle
        .get("mask")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    (rust_mask, oracle_mask)
}

fn mask_hist_exact(rust: &serde_json::Value, oracle: &serde_json::Value) -> (bool, bool) {
    let (rust_mask, oracle_mask) = mask_sets(rust, oracle);
    let rust_hk = str_list(rust.get("history_kinds").unwrap_or(&serde_json::Value::Null));
    let oracle_hk = str_list(oracle.get("hk").unwrap_or(&serde_json::Value::Null));
    (rust_mask == oracle_mask, rust_hk == oracle_hk)
}

fn mask_extra_ids(rust: &serde_json::Value, oracle: &serde_json::Value) -> Vec<u64> {
    let (rust_mask, oracle_mask) = mask_sets(rust, oracle);
    rust_mask.difference(&oracle_mask).copied().collect()
}
