//! `step_hash`: one-call batch for the ISMCTS envelope step hashes.
//!
//! Parity: `ismcts_search.py:532-570` (`_doc_for_step` + `_policy_dir_for_step`
//! + `_info_key_for_step` + `_observation_hash_for_doc` +
//!   `_policy_direction_for_hash`). Per descent step the envelope substitutes
//!   five fields into the search-constant identity-doc template
//!   (`concealed_hand` sorted, `live_wall_tiles_remaining`, `actor`,
//!   `turn_actor`, `decision_id = "dec_hand_<unsorted-hand>_\<actor>"`) and
//!   either (root step) hashes the canon doc MINUS `legal_mask` as the
//!   information-set key, or (continuation step) hashes the canon doc WITH
//!   the mask as `sha256:` text and tilts on `sha256(obs_hash_text)[0] & 1`.
//!
//! Canon-wins (B3): every doc serializes through `feed::canon`
//! (`canonical_bytes_value` over the parsed template `Value`; JCS sorts,
//! matching the oracle's `canonical_bytes`). The template parses ONCE per
//! batch; per-row substitution clones the template `Value` (rows are
//! small: ~40 template keys, sims × depth rows). Decision ids join the
//! UNSORTED hand exactly as the oracle (only `concealed_hand` is sorted).
//!
//! Inputs are parallel arrays (`hands`, `live_lens`, `actors`, `want_keys`)
//! with one entry per envelope step; outputs are `(keys, dirs)` where a
//! key row carries `""` and a dir row carries `0` in the unused lane.

use sha2::{Digest, Sha256};

use crate::SearchError;

/// Batch the envelope step hashes for one search.
///
/// `template_json` is the `observation_identity_document` template
/// serialized once caller-side. Each row substitutes its hand/live/actor
/// and yields either the info key (`want_key`) or the tilt direction.
/// Length mismatch across the four lanes is `InvalidArg`; a non-object
/// template or a canon failure is `Canon`/fail-closed, never a default.
pub fn step_hashes(
    template_json: &[u8],
    hands: &[Vec<u32>],
    live_lens: &[u32],
    actors: &[u32],
    want_keys: &[bool],
) -> Result<(Vec<String>, Vec<u32>), SearchError> {
    let rows = hands.len();
    if live_lens.len() != rows || actors.len() != rows || want_keys.len() != rows {
        return Err(SearchError::InvalidArg {
            detail: "ismcts step lane length mismatch",
        });
    }
    let template: serde_json::Value =
        serde_json::from_slice(template_json).map_err(|err| SearchError::Canon {
            detail: err.to_string(),
        })?;
    if !template.is_object() {
        return Err(SearchError::InvalidArg {
            detail: "ismcts template must be a JSON object",
        });
    }
    let mut keys = Vec::with_capacity(rows);
    let mut dirs = Vec::with_capacity(rows);
    let mut row = 0;
    while row < rows {
        let mut doc = template.clone();
        let hand = &hands[row];
        let live = live_lens[row];
        let actor = actors[row];
        let obj = doc.as_object_mut().ok_or(SearchError::InvalidArg {
            detail: "ismcts template must be a JSON object",
        })?;
        let mut sorted = hand.clone();
        sorted.sort_unstable();
        obj.insert(
            "concealed_hand".to_string(),
            serde_json::Value::Array(sorted.iter().map(|t| serde_json::json!(*t)).collect()),
        );
        obj.insert(
            "live_wall_tiles_remaining".to_string(),
            serde_json::json!(live),
        );
        obj.insert("actor".to_string(), serde_json::json!(actor));
        obj.insert("turn_actor".to_string(), serde_json::json!(actor));
        let dec = format!(
            "dec_hand_{}_{actor}",
            hand.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        obj.insert("decision_id".to_string(), serde_json::json!(dec));
        if want_keys[row] {
            obj.remove("legal_mask");
            let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "search:step_key").map_err(
                |err| SearchError::Canon {
                    detail: err.to_string(),
                },
            )?;
            keys.push(hydra_feed::digest::sha256_hex(&bytes));
            dirs.push(0);
        } else {
            let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "search:step_obs").map_err(
                |err| SearchError::Canon {
                    detail: err.to_string(),
                },
            )?;
            let obs_hash = hydra_feed::digest::sha256_hex(&bytes);
            let tilt = Sha256::digest(obs_hash.as_bytes());
            keys.push(String::new());
            dirs.push((tilt[0] & 1) as u32);
        }
        row += 1;
    }
    Ok((keys, dirs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template() -> Vec<u8> {
        serde_json::json!({
            "game_id": "g",
            "decision_id": "dec_root",
            "actor": 0,
            "turn_actor": 0,
            "live_wall_tiles_remaining": 32,
            "concealed_hand": [0, 1],
            "legal_mask": [true, false, true],
            "scores": [25000, 25000, 25000, 25000],
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn key_lane_drops_mask_and_dir_lane_keeps_it() {
        let tpl = template();
        let (keys, dirs) = step_hashes(
            &tpl,
            &[vec![1, 0], vec![3, 2]],
            &[31, 30],
            &[0, 1],
            &[true, false],
        )
        .expect("hashes");
        assert!(keys[0].starts_with("sha256:"));
        assert_eq!(keys[1], "");
        assert_eq!(dirs[0], 0);
        assert!(dirs[1] <= 1);
        // Deterministic across calls.
        let again = step_hashes(&tpl, &[vec![1, 0]], &[31], &[0], &[true]).expect("again");
        assert_eq!(again.0[0], keys[0]);
    }

    #[test]
    fn lane_mismatch_and_bad_template_fail_closed() {
        let tpl = template();
        assert!(step_hashes(&tpl, &[vec![0]], &[1, 2], &[0], &[true]).is_err());
        assert!(step_hashes(b"[1,2]", &[vec![0]], &[1], &[0], &[true]).is_err());
        assert!(step_hashes(b"\xff\xfe", &[vec![0]], &[1], &[0], &[true]).is_err());
    }
}
