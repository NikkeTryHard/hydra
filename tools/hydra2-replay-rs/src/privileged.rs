//! Privileged placement rows: terminal ranks joined to actor rows by `decision_id`.
//!
//! Reference (read-only): `src/hydra2/data/replay_expand.py`
//! (`expand_privileged_rows`, `_final_scores`, `_count_row_decisions`),
//! `src/hydra2/belief/oracle_loader.py` (`ranks_from_final_scores`) and
//! `src/hydra2/data/parquet.py` (`validate_privileged_ranks`,
//! `write_privileged_ranks` label shape, `FORBIDDEN_IN_ACTOR`). Rule order
//! reused; code written fresh.
//!
//! No engine touch: ranks are a pure function of the terminal score quad.
//! Rust never writes parquet — it emits privileged JSON (`decision_id` +
//! the opaque `privileged_label`) or paired decision ids, and the Python
//! parquet owner writes the privileged shard. The actor-privileged firewall
//! holds by construction: [`check_privileged_label`] pins the opaque label
//! to exactly `{ranks, split, wall_id?}`, and actor rows never carry those
//! keys (see the `leakage` battery in `tests/s5_privileged.rs`).
//!
//! `wall_id` rule (wall-less v1): the driver never binds walls, so
//! `wall_id: None` OMITS the key entirely (never a placeholder, never
//! `""`). An explicit `Some` digest passes through verbatim (the walled
//! path's real digest, minted once S6 wall support lands); `Some("")`
//! fails closed.

use crate::decision_ids::decision_id;
use crate::decisions::count_row_decisions;
use crate::mjai_event::END_TYPES;
use crate::stream::ParsedGame;

/// Opaque label keys: `ranks` + `split` always, `wall_id` only when walled
/// (mirrors `write_privileged_ranks`; kept out of any Arrow schema).
pub const PRIVILEGED_LABEL_FIELDS: [&str; 3] = ["ranks", "split", "wall_id"];

/// One privileged placement row: ranks joined to its actor row by opaque id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivilegedRow {
    pub decision_id: String,
    pub ranks: [u8; 4],
    pub wall_id: Option<String>,
    pub split: String,
}

impl PrivilegedRow {
    /// Opaque label document: `{ranks, split}` plus `wall_id` IFF walled.
    pub fn privileged_label(&self) -> serde_json::Value {
        let mut label = serde_json::Map::with_capacity(3);
        label.insert(
            "ranks".to_string(),
            serde_json::Value::Array(
                self.ranks
                    .iter()
                    .map(|r| serde_json::Value::Number(serde_json::Number::from(*r as u64)))
                    .collect(),
            ),
        );
        label.insert(
            "split".to_string(),
            serde_json::Value::String(self.split.clone()),
        );
        if let Some(wall_id) = &self.wall_id {
            label.insert(
                "wall_id".to_string(),
                serde_json::Value::String(wall_id.clone()),
            );
        }
        serde_json::Value::Object(label)
    }

    /// Privileged JSON: exactly `{decision_id, privileged_label}` (the row
    /// shape the Python privileged writer consumes; never parquet bytes).
    pub fn to_privileged_json(&self) -> serde_json::Value {
        serde_json::json!({
            "decision_id": self.decision_id,
            "privileged_label": self.privileged_label(),
        })
    }
}

/// Terminal 4-score quad from the closing end-game event (reverse scan).
///
/// Mirrors `_final_scores`: the first end-alias event from the tail wins;
/// `scores` is read before its `final_scores` alias; a present-but-short
/// quad falls through to the alias, then fails closed. Errors name the
/// oracle's messages so quarantine joins keep working.
pub fn final_scores(game: &ParsedGame) -> Result<Vec<serde_json::Value>, String> {
    for event in game.events.iter().rev() {
        if END_TYPES.contains(&event.type_.as_str()) {
            for scores in [&event.scores, &event.final_scores] {
                if let Some(values) = scores {
                    if values.len() == 4 {
                        return Ok(values.clone());
                    }
                }
            }
            return Err(format!(
                "privileged ranks game {:?}: end_game event carries no 4-score quad (scores/final_scores)",
                game.game_id,
            ));
        }
    }
    Err(format!(
        "privileged ranks game {:?}: game has no closing end_game event for privileged ranks",
        game.game_id,
    ))
}

/// Per-seat ranks 1..4 from a terminal final-score quad (rank 1 = top).
///
/// Mirrors `ranks_from_final_scores`: strict — exactly 4 finite distinct
/// numbers (`bool` rejected; JSON carries no non-finite floats). Ties raise:
/// equal scores need Tenhou east-1 seat-wind context (`resolve_final_ranks`)
/// that raw scores alone do not carry, so tied games must be
/// Tenhou-resolved before calling.
pub fn ranks_from_final_scores(scores: &[serde_json::Value]) -> Result<[u8; 4], String> {
    if scores.len() != 4 {
        return Err(format!(
            "ranks_from_final_scores: expected 4 final scores, got {scores:?}"
        ));
    }
    let mut vals = [0.0f64; 4];
    for (i, score) in scores.iter().enumerate() {
        let value = match score {
            serde_json::Value::Number(n) => n.as_f64().ok_or_else(|| {
                format!("ranks_from_final_scores: score[{i}] must be a number, got {score}")
            })?,
            _ => {
                return Err(format!(
                    "ranks_from_final_scores: score[{i}] must be a number, got {score}"
                ));
            }
        };
        if !value.is_finite() {
            return Err(format!(
                "ranks_from_final_scores: score[{i}] must be finite, got {score}"
            ));
        }
        vals[i] = value;
    }
    let mut ordered = vals;
    ordered.sort_by(|a, b| a.total_cmp(b));
    if ordered.windows(2).any(|w| w[0] == w[1]) {
        return Err(format!(
            "ranks_from_final_scores: scores must be distinct (ties need Tenhou resolve_final_ranks first), got {scores:?}"
        ));
    }
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&a, &b| {
        vals[b]
            .total_cmp(&vals[a])
            .then_with(|| a.cmp(&b))
    });
    let mut ranks = [0u8; 4];
    for (position, seat) in order.iter().enumerate() {
        ranks[*seat] = (position + 1) as u8;
    }
    Ok(ranks)
}

/// Pin the day-one ranks convention: strict 1..4 4-list permutation.
///
/// Mirrors `validate_privileged_ranks`: non-array/wrong length/non-int
/// (`bool` included — it is a distinct JSON variant, never an int) or a
/// non-permutation (including the 0..3 seat convention, which is a
/// read-only bridge in the oracle loader and must never be written) all
/// fail closed with the oracle's message shape.
pub fn validate_privileged_ranks(
    ranks: &[serde_json::Value],
    decision_id: &str,
) -> Result<[u8; 4], String> {
    let where_id = if decision_id.is_empty() {
        String::new()
    } else {
        format!(" for {decision_id:?}")
    };
    if ranks.len() != 4 {
        return Err(format!(
            "privileged_label['ranks'] must be a 4-list{where_id}"
        ));
    }
    let mut ordered = [0u8; 4];
    for (i, rank) in ranks.iter().enumerate() {
        let value = match rank {
            serde_json::Value::Number(n) => {
                if let Some(v) = n.as_u64() {
                    v
                } else if let Some(v) = n.as_i64() {
                    if v < 0 {
                        return Err(format!(
                            "privileged_label['ranks'] must be int 1..4{where_id}"
                        ));
                    }
                    v as u64
                } else {
                    return Err(format!(
                        "privileged_label['ranks'] must be int 1..4{where_id}"
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "privileged_label['ranks'] must be int 1..4{where_id}"
                ));
            }
        };
        if !(1..=4).contains(&value) {
            return Err(format!(
                "privileged_label['ranks'] must be int 1..4{where_id}"
            ));
        }
        ordered[i] = value as u8;
    }
    let mut sorted = ordered;
    sorted.sort_unstable();
    if sorted != [1, 2, 3, 4] {
        return Err(format!(
            "privileged_label['ranks'] must be a strict 1..4 permutation{where_id}, got {:?}",
            ordered.to_vec()
        ));
    }
    Ok(ordered)
}

/// Opaque-label gate: exactly `{ranks, split, wall_id?}` (mirrors
/// `write_privileged_ranks` per-row validation; the writer additionally
/// requires `split == "train"`, which stays the caller's contract).
pub fn check_privileged_label(label: &serde_json::Value) -> Result<(), String> {
    let obj = label
        .as_object()
        .ok_or_else(|| "privileged_label must be a JSON object".to_string())?;
    for key in obj.keys() {
        if !PRIVILEGED_LABEL_FIELDS.contains(&key.as_str()) {
            return Err(format!("privileged_label carries non-label key {key:?}"));
        }
    }
    let ranks = obj
        .get("ranks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "privileged_label must carry a ranks 4-list".to_string())?;
    validate_privileged_ranks(ranks, "")?;
    match obj.get("split").and_then(|v| v.as_str()) {
        Some(split) if !split.is_empty() => {}
        _ => return Err("privileged_label split must be a non-empty string".to_string()),
    }
    if let Some(wall) = obj.get("wall_id") {
        match wall.as_str() {
            Some(id) if !id.is_empty() => {}
            _ => return Err("privileged_label wall_id must be a non-empty string".to_string()),
        }
    }
    Ok(())
}

/// Build privileged placement rows joining the game's actor rows.
///
/// Mirrors `expand_privileged_rows`: the row COUNT comes from the
/// engine-free counter (so lengths join actor rows on mutually-ok games),
/// ranks come from the terminal quad, and per-seq `decision_id`s
/// (`{game}:d{seq:04}`) pair 1:1 with `expand_game` ids. `wall_id` defaults
/// to absent (wall-less: never invented); an explicit digest rides the
/// label verbatim, `Some("")` fails closed.
pub fn expand_privileged_rows(
    game: &ParsedGame,
    split: &str,
    wall_id: Option<&str>,
) -> Result<Vec<PrivilegedRow>, String> {
    if split.is_empty() {
        return Err("privileged rows: split must be a non-empty string".to_string());
    }
    let count = count_row_decisions(game)
        .map_err(|reject| format!("privileged rows count: {}: {}", reject.code, reject.detail))?;
    let scores = final_scores(game)?;
    let ranks = ranks_from_final_scores(&scores)?;
    let resolved_wall = match wall_id {
        None => None,
        Some(id) if id.is_empty() => {
            return Err("privileged rows: wall_id must be a non-empty string".to_string());
        }
        Some(id) => Some(id.to_string()),
    };
    Ok((0..count)
        .map(|seq| PrivilegedRow {
            decision_id: decision_id(&game.game_id, seq as u32),
            ranks,
            wall_id: resolved_wall.clone(),
            split: split.to_string(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_of(lines: &[&str], game_id: &str) -> ParsedGame {
        let mut text = lines.join("\n");
        text.push('\n');
        crate::stream::parse_game(&text, game_id).expect("fixture parses")
    }

    fn kyoku_header() -> &'static str {
        "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"3m\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"tehais\":[[\"1m\",\"2m\",\"3m\",\"4m\",\"5m\",\"6m\",\"7m\",\"8m\",\"9m\",\"1p\",\"2p\",\"3p\",\"4p\"],[\"5p\",\"6p\",\"7p\",\"8p\",\"9p\",\"1s\",\"2s\",\"3s\",\"4s\",\"5s\",\"6s\",\"7s\",\"8s\"],[\"9s\",\"E\",\"S\",\"W\",\"N\",\"P\",\"F\",\"C\",\"1m\",\"2m\",\"3m\",\"4m\",\"5m\"],[\"6m\",\"7m\",\"8m\",\"9m\",\"1p\",\"2p\",\"3p\",\"4p\",\"5p\",\"6p\",\"7p\",\"8p\",\"9p\"]]}"
    }

    fn two_dahai_lines() -> Vec<&'static str> {
        vec![
            "{\"type\":\"start_game\"}",
            kyoku_header(),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1s\"}",
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1s\",\"tsumogiri\":true}",
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"2s\"}",
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"2s\",\"tsumogiri\":true}",
            "{\"type\":\"end_kyoku\"}",
        ]
    }

    #[test]
    fn final_scores_prefers_scores_then_alias() {
        let mut lines = two_dahai_lines();
        lines.push("{\"type\":\"end_game\",\"scores\":[31000,27000,23000,19000],\"final_scores\":[19000,19001,19002,19003]}");
        let game = game_of(&lines, "s5-scores");
        let scores = final_scores(&game).expect("scores quad");
        assert_eq!(scores, vec![31000, 27000, 23000, 19000]);
        assert_eq!(ranks_from_final_scores(&scores).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn final_scores_falls_back_to_alias() {
        let mut lines = two_dahai_lines();
        lines.push("{\"type\":\"end_game\",\"final_scores\":[19000,23000,27000,31000]}");
        let game = game_of(&lines, "s5-alias");
        let scores = final_scores(&game).expect("alias quad");
        assert_eq!(ranks_from_final_scores(&scores).unwrap(), [4, 3, 2, 1]);
    }

    #[test]
    fn final_scores_fails_closed() {
        // No quad on the end event.
        let mut lines = two_dahai_lines();
        lines.push("{\"type\":\"end_game\"}");
        let game = game_of(&lines, "s5-missing");
        let err = final_scores(&game).unwrap_err();
        assert!(err.contains("no 4-score quad"), "unexpected: {err}");
        // Present but short quad falls through to the same failure.
        let mut lines = two_dahai_lines();
        lines.push("{\"type\":\"end_game\",\"scores\":[1,2,3]}");
        let game = game_of(&lines, "s5-short");
        assert!(final_scores(&game).unwrap_err().contains("no 4-score quad"));
    }

    #[test]
    fn ranks_rejects_ties_and_non_numbers() {
        let tie: Vec<serde_json::Value> =
            serde_json::from_str("[25000,25000,25000,25000]").unwrap();
        let err = ranks_from_final_scores(&tie).unwrap_err();
        assert!(err.contains("distinct"), "unexpected: {err}");
        let boolean: Vec<serde_json::Value> =
            serde_json::from_str("[true,27000,23000,19000]").unwrap();
        assert!(ranks_from_final_scores(&boolean).unwrap_err().contains("must be a number"));
        let text: Vec<serde_json::Value> =
            serde_json::from_str("[\"x\",27000,23000,19000]").unwrap();
        assert!(ranks_from_final_scores(&text).is_err());
        let short: Vec<serde_json::Value> = serde_json::from_str("[1,2,3]").unwrap();
        assert!(ranks_from_final_scores(&short).unwrap_err().contains("expected 4"));
    }

    #[test]
    fn ranks_order_is_score_descending_then_seat() {
        let scores: Vec<serde_json::Value> =
            serde_json::from_str("[10000,30000,20000,0]").unwrap();
        assert_eq!(ranks_from_final_scores(&scores).unwrap(), [3, 1, 2, 4]);
    }

    #[test]
    fn validate_ranks_pins_permutation() {
        let ok: Vec<serde_json::Value> = serde_json::from_str("[2,1,4,3]").unwrap();
        assert_eq!(validate_privileged_ranks(&ok, "g:d0000").unwrap(), [2, 1, 4, 3]);
        // 0..3 seat convention is a read bridge, never writable.
        let seat: Vec<serde_json::Value> = serde_json::from_str("[0,1,2,3]").unwrap();
        assert!(validate_privileged_ranks(&seat, "").is_err());
        let dup: Vec<serde_json::Value> = serde_json::from_str("[1,2,3,3]").unwrap();
        assert!(validate_privileged_ranks(&dup, "").unwrap_err().contains("permutation"));
        let short: Vec<serde_json::Value> = serde_json::from_str("[1,2,3]").unwrap();
        assert!(validate_privileged_ranks(&short, "").unwrap_err().contains("4-list"));
        let boolean: Vec<serde_json::Value> = serde_json::from_str("[true,2,3,4]").unwrap();
        assert!(validate_privileged_ranks(&boolean, "").unwrap_err().contains("int 1..4"));
        let float: Vec<serde_json::Value> = serde_json::from_str("[1.5,2,3,4]").unwrap();
        assert!(validate_privileged_ranks(&float, "").is_err());
        let out_of_range: Vec<serde_json::Value> = serde_json::from_str("[1,2,3,5]").unwrap();
        assert!(validate_privileged_ranks(&out_of_range, "").is_err());
    }

    #[test]
    fn label_gate_pins_shape() {
        let row = PrivilegedRow {
            decision_id: "g:d0000".to_string(),
            ranks: [1, 2, 3, 4],
            wall_id: None,
            split: "train".to_string(),
        };
        let label = row.privileged_label();
        assert!(!label.as_object().unwrap().contains_key("wall_id"));
        check_privileged_label(&label).expect("wall-less label");
        let doc = row.to_privileged_json();
        let keys: Vec<&str> = doc.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys.len(), 2);
        // Extra keys, empty split, empty wall_id all fail closed.
        let mut bad = label.clone();
        bad["wall_id"] = serde_json::json!("");
        assert!(check_privileged_label(&bad).is_err());
        let mut bad = label.clone();
        bad["split"] = serde_json::json!("");
        assert!(check_privileged_label(&bad).is_err());
        let mut bad = label;
        bad["ranks"] = serde_json::json!([0, 1, 2, 3]);
        assert!(check_privileged_label(&bad).is_err());
        let mut bad = row.to_privileged_json()["privileged_label"].clone();
        bad["hidden_tiles"] = serde_json::json!([]);
        assert!(check_privileged_label(&bad).is_err());
        assert!(check_privileged_label(&serde_json::json!({"split": "train"})).is_err());
    }

    #[test]
    fn expand_rejects_empty_split_and_wall() {
        let mut lines = two_dahai_lines();
        lines.push("{\"type\":\"end_game\",\"scores\":[31000,27000,23000,19000]}");
        let game = game_of(&lines, "s5-expand");
        assert!(expand_privileged_rows(&game, "", None).unwrap_err().contains("split"));
        assert!(expand_privileged_rows(&game, "train", Some("")).unwrap_err().contains("wall_id"));
        let rows = expand_privileged_rows(&game, "train", None).expect("expand");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].decision_id, format!("{}:d0000", game.game_id));
        assert_eq!(rows[1].decision_id, format!("{}:d0001", game.game_id));
        assert_eq!(rows[0].ranks, [1, 2, 3, 4]);
        assert!(rows.iter().all(|r| r.wall_id.is_none()));
    }
}
