//! rl/core/row: captured GameRow snapshot serialization and construction.
//!
//! A GameRow binds one committed decision's compact facts: lineage ids,
//! round state, concealed hand and draw, legal set, selected action with its
//! log-probability and value vector, sampler cursors, observation hash, event
//! prefix length, and content hashes. `to_json` emits the closed projection
//! the pool snapshot embeds so restore never re-infers or resamples; every
//! 32-bit float round-trips through 64-bit exactly. `from_json` is the
//! fail-closed inverse. `capture_row` builds a row from a bound observation
//! and GPU outputs, advancing the capture counter for `trajectory_step`.
//! No wall law, no RNG, no Python.

use sha2::{Digest, Sha256};

use crate::rl::observation::NativeObservation;

use super::{CoreError, GameRow, pack_bits_lsb};

fn row_hex(raw: &[u8; 32]) -> String {
    raw.iter().map(|b| format!("{b:02x}")).collect()
}

fn row_unhex(text: &str, key: &str) -> Result<[u8; 32], CoreError> {
    let noprefix = text.trim_start_matches("sha256:");
    if noprefix.len() != 64 {
        return Err(CoreError(format!("row field {key} must be 32 bytes hex")));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&noprefix[2 * i..2 * i + 2], 16)
            .map_err(|_| CoreError(format!("row field {key} must be 32 bytes hex")))?;
    }
    Ok(out)
}

impl super::GameRow {
    /// Closed JSON projection of the full captured compact facts (LP, value,
    /// selected, cursors, hashes, lineage): the pool snapshot embeds these
    /// so restore never re-infers or resamples. f32 facts round-trip
    /// through f64 exactly (every f32 is representable in f64).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "wall_id": self.wall_id,
            "game_id": self.game_id,
            "round_id": self.round_id,
            "trajectory_step": self.trajectory_step,
            "actor": self.actor,
            "decision_id": self.decision_id,
            "sequence": self.sequence,
            "row_id": row_hex(&self.row_id),
            "round_index": self.round_index,
            "round_wind": self.round_wind,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "dealer": self.dealer,
            "scores": self.scores,
            "live_remaining": self.live_remaining,
            "hand_number": self.hand_number,
            "turn_actor": self.turn_actor,
            "phase_id": self.phase_id,
            "concealed": self.concealed,
            "drawn": self.drawn,
            "dora5": self.dora5,
            "ippatsu_bits": self.ippatsu_bits,
            "furiten_id": self.furiten_id,
            "can_tsumo": self.can_tsumo,
            "can_riichi": self.can_riichi,
            "legal_ids": self.legal_ids,
            "legal_mask_bits": self.legal_mask_bits,
            "selected": self.selected,
            "selected_logp": self.selected_logp,
            "value4": self.value4,
            "cursor_before": self.cursor_before,
            "cursor_after": self.cursor_after,
            "obs_hash": row_hex(&self.obs_hash),
            "event_prefix_len": self.event_prefix_len,
            "derivation": row_hex(&self.derivation),
            "action_table_hash": row_hex(&self.action_table_hash),
            "action_table_hash_hex": self.action_table_hash_hex,
            "behavior_ckpt": row_hex(&self.behavior_ckpt),
            "transition_provenance": row_hex(&self.transition_provenance),
            "rules_id": self.rules_id,
            "rules_hash": self.rules_hash,
            "event_schema_hash": self.event_schema_hash,
            "observation_schema_hash": self.observation_schema_hash,
            "packet_boundary_hash": self.packet_boundary_hash,
            "return_g4": self.return_g4,
            "raw_advantage_f32": self.raw_advantage_f32,
        })
    }

    /// Inverse of `to_json` (fail-closed on any missing/m mistyped field;
    /// unknown fields rejected by the caller, not here).
    pub fn from_json(value: &serde_json::Value) -> Result<Self, CoreError> {
        let err = |key: &str| CoreError(format!("row field {key} missing or mistyped"));
        let str_of = |key: &str| {
            value.get(key).and_then(|v| v.as_str()).map(str::to_string).ok_or_else(|| err(key))
        };
        let u64_of = |key: &str| value.get(key).and_then(|v| v.as_u64()).ok_or_else(|| err(key));
        let u8_of = |key: &str| {
            u64_of(key).and_then(|n| {
                u8::try_from(n).map_err(|_| err(key))
            })
        };
        let u32_of = |key: &str| {
            u64_of(key).and_then(|n| {
                u32::try_from(n).map_err(|_| err(key))
            })
        };
        let i64_of = |key: &str| value.get(key).and_then(|v| v.as_i64()).ok_or_else(|| err(key));
        let bool_of = |key: &str| value.get(key).and_then(|v| v.as_bool()).ok_or_else(|| err(key));
        let bytes_of = |key: &str| -> Result<Vec<u8>, CoreError> {
            value
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| err(key))?
                .iter()
                .map(|t| t.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or_else(|| err(key)))
                .collect()
        };
        let u32s_of = |key: &str| -> Result<Vec<u32>, CoreError> {
            value
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| err(key))?
                .iter()
                .map(|t| t.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| err(key)))
                .collect()
        };
        let i32s_of = |key: &str| -> Result<Vec<i32>, CoreError> {
            value
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| err(key))?
                .iter()
                .map(|t| t.as_i64().and_then(|n| i32::try_from(n).ok()).ok_or_else(|| err(key)))
                .collect()
        };
        let f32_of = |key: &str| {
            value.get(key).and_then(|v| v.as_f64()).map(|f| f as f32).ok_or_else(|| err(key))
        };
        let arr4 = |key: &str| -> Result<[f32; 4], CoreError> {
            let items = value.get(key).and_then(|v| v.as_array()).ok_or_else(|| err(key))?;
            if items.len() != 4 {
                return Err(err(key));
            }
            let mut out = [0f32; 4];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = items[i].as_f64().map(|f| f as f32).ok_or_else(|| err(key))?;
            }
            Ok(out)
        };
        let scores_vec = i32s_of("scores")?;
        if scores_vec.len() != 4 {
            return Err(err("scores"));
        }
        let dora_vec = i32s_of("dora5")?;
        if dora_vec.len() != 5 {
            return Err(err("dora5"));
        }
        Ok(Self {
            wall_id: str_of("wall_id")?,
            game_id: str_of("game_id")?,
            round_id: str_of("round_id")?,
            trajectory_step: u64_of("trajectory_step")?,
            actor: u8_of("actor")?,
            decision_id: str_of("decision_id")?,
            sequence: u64_of("sequence")?,
            row_id: row_unhex(value.get("row_id").and_then(|v| v.as_str()).ok_or_else(|| err("row_id"))?, "row_id")?,
            round_index: u64_of("round_index")?,
            round_wind: i64_of("round_wind")?,
            honba: u32_of("honba")?,
            kyotaku: u32_of("kyotaku")?,
            dealer: u8_of("dealer")?,
            scores: [scores_vec[0], scores_vec[1], scores_vec[2], scores_vec[3]],
            live_remaining: u32_of("live_remaining")?,
            hand_number: u32_of("hand_number")?,
            turn_actor: u8_of("turn_actor")?,
            phase_id: u8_of("phase_id")?,
            concealed: bytes_of("concealed")?,
            drawn: match value.get("drawn") {
                None | Some(serde_json::Value::Null) => None,
                Some(v) => Some(v.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or_else(|| err("drawn"))?),
            },
            dora5: [dora_vec[0], dora_vec[1], dora_vec[2], dora_vec[3], dora_vec[4]],
            ippatsu_bits: u8_of("ippatsu_bits")?,
            furiten_id: u8_of("furiten_id")?,
            can_tsumo: bool_of("can_tsumo")?,
            can_riichi: bool_of("can_riichi")?,
            legal_ids: u32s_of("legal_ids")?,
            legal_mask_bits: bytes_of("legal_mask_bits")?,
            selected: u32_of("selected")?,
            selected_logp: f32_of("selected_logp")?,
            value4: arr4("value4")?,
            cursor_before: u64_of("cursor_before")?,
            cursor_after: u64_of("cursor_after")?,
            obs_hash: row_unhex(value.get("obs_hash").and_then(|v| v.as_str()).ok_or_else(|| err("obs_hash"))?, "obs_hash")?,
            event_prefix_len: u32_of("event_prefix_len")?,
            derivation: row_unhex(value.get("derivation").and_then(|v| v.as_str()).ok_or_else(|| err("derivation"))?, "derivation")?,
            action_table_hash: row_unhex(value.get("action_table_hash").and_then(|v| v.as_str()).ok_or_else(|| err("action_table_hash"))?, "action_table_hash")?,
            action_table_hash_hex: str_of("action_table_hash_hex")?,
            behavior_ckpt: row_unhex(value.get("behavior_ckpt").and_then(|v| v.as_str()).ok_or_else(|| err("behavior_ckpt"))?, "behavior_ckpt")?,
            transition_provenance: row_unhex(value.get("transition_provenance").and_then(|v| v.as_str()).ok_or_else(|| err("transition_provenance"))?, "transition_provenance")?,
            rules_id: str_of("rules_id")?,
            rules_hash: str_of("rules_hash")?,
            event_schema_hash: str_of("event_schema_hash")?,
            observation_schema_hash: str_of("observation_schema_hash")?,
            packet_boundary_hash: str_of("packet_boundary_hash")?,
            return_g4: match value.get("return_g4") {
                None | Some(serde_json::Value::Null) => None,
                Some(v) => {
                    let items = v.as_array().ok_or_else(|| err("return_g4"))?;
                    if items.len() != 4 {
                        return Err(err("return_g4"));
                    }
                    let mut out = [0f64; 4];
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = items[i].as_f64().ok_or_else(|| err("return_g4"))?;
                    }
                    Some(out)
                }
            },
            raw_advantage_f32: match value.get("raw_advantage_f32") {
                None | Some(serde_json::Value::Null) => None,
                Some(v) => Some(v.as_f64().map(|f| f as f32).ok_or_else(|| err("raw_advantage_f32"))?),
            },
        })
    }
}

impl super::NativeGame {
    // -- capture ----------------------------------------------------------------------

    /// Capture compact row facts for a committed decision (pool calls this
    /// after GPU outputs land). `cursor_before` is the pool-supplied sampler
    /// reservation for this exact decision (per-(game, seat) stream word at
    /// `counter * 8`, reserved once at fill): the row binds the reservation
    /// that produced its uniform, so forked restores can verify cursors
    /// without redrawing. `trajectory_step` still rides `self.captured`
    /// (capture order), independent of sampler counters.
    pub fn capture_row(
        &mut self,
        obs: &NativeObservation,
        selected: u32,
        selected_logp: f32,
        value4: [f32; 4],
        wall_id: &str,
        action_table_hash: [u8; 32],
        derivation: [u8; 32],
        cursor_before: u64,
    ) -> Result<GameRow, CoreError> {
        if !selected_logp.is_finite() || !value4.iter().all(|v| v.is_finite()) {
            return Err(CoreError("commit values must be finite".to_string()));
        }
        if (selected as usize) >= crate::rl::actions::BASELINE_ACTIONS
            || !obs.legal_mask[selected as usize]
        {
            return Err(CoreError(format!("selected action {selected} outside served legal set")));
        }
        let prefix_len = self.arena.len() as u32;
        // Capture ordinal BEFORE increment: the row's scheduling identity.
        // Captures precede applies within a commit, so reading the live
        // applied-action count here would give every same-fill seat one
        // shared value and erase their per-answer distinction.
        let capture_ordinal = self.captured;
        let provenance = self.transition_provenance(prefix_len, capture_ordinal)?;
        let mut ippatsu_bits = 0u8;
        for (i, active) in obs.ippatsu_active.iter().enumerate() {
            if *active {
                ippatsu_bits |= 1 << i;
            }
        }
        let row = GameRow {
            wall_id: wall_id.to_string(),
            game_id: self.game_id.clone(),
            round_id: format!("{}:{}", self.game_id, self.hand_index),
            trajectory_step: self.captured,
            actor: obs.actor,
            decision_id: obs.decision_id.clone(),
            sequence: obs.sequence,
            row_id: [0u8; 32],
            round_index: self.hand_index,
            round_wind: obs.round_wind,
            honba: obs.honba,
            kyotaku: obs.riichi_sticks,
            dealer: obs.dealer,
            scores: obs.scores,
            live_remaining: obs.live_wall_remaining,
            hand_number: obs.hand_number,
            turn_actor: obs.turn_actor,
            phase_id: crate::rl::observation::phase_id(&obs.phase).map_err(CoreError::from)? as u8,
            concealed: obs.concealed_hand.clone(),
            drawn: obs.own_drawn_tile,
            dora5: obs.dora_indicators.as_slice().try_into().map_err(|_| {
                CoreError("captured dora_indicators must have exactly five visible slots".to_string())
            })?,
            ippatsu_bits,
            furiten_id: crate::rl::observation::furiten_id(&obs.actor_furiten)
                .map_err(CoreError::from)? as u8,
            can_tsumo: obs.actor_can_tsumo,
            can_riichi: obs.actor_can_riichi,
            legal_ids: obs
                .legal_mask
                .iter()
                .enumerate()
                .filter(|(_, f)| **f)
                .map(|(i, _)| i as u32)
                .collect(),
            legal_mask_bits: pack_bits_lsb(&obs.legal_mask),
            selected,
            selected_logp,
            value4,
            cursor_before,
            cursor_after: cursor_before + 8,
            obs_hash: obs.observation_hash.ok_or_else(|| {
                CoreError("observation hash must be bound before capture".to_string())
            })?,
            event_prefix_len: prefix_len,
            derivation,
            action_table_hash,
            action_table_hash_hex: self.ctx.action_table_hash.clone(),
            behavior_ckpt: self.ctx.behavior_ckpt,
            transition_provenance: provenance,
            rules_id: self.ctx.rules_id.clone(),
            rules_hash: self.ctx.rules_hash.clone(),
            event_schema_hash: self.ctx.event_schema_hash.clone(),
            observation_schema_hash: self.ctx.observation_schema_hash.clone(),
            packet_boundary_hash: self.ctx.packet_boundary_hash.clone(),
            return_g4: None,
            raw_advantage_f32: None,
        };
        if std::env::var_os("HYDRA2_DIAGNOSE_SEAL").is_some() {
            let live = obs.identity_json(&self.arena).map_err(CoreError::from)?;
            let mut rebuilt = crate::rl::observation::materialize_canonical(&row, &self.arena)
                .map_err(CoreError::from)?;
            rebuilt.as_object_mut().ok_or_else(|| CoreError("rebuilt observation not object".to_string()))?
                .remove("observation_hash");
            if live != rebuilt {
                let differences: Vec<_> = live.as_object().ok_or_else(|| CoreError("live observation not object".to_string()))?
                    .iter().filter(|(key, value)| rebuilt.get(*key) != Some(*value))
                    .map(|(key, value)| serde_json::json!({"field": key, "live": value, "rebuilt": rebuilt.get(key)})).collect();
                return Err(CoreError(format!("seal reconstruction differs at {}: {}", row.decision_id, serde_json::Value::Array(differences))));
            }
        }
        self.captured += 1;
        Ok(row)
    }

    fn transition_provenance(&self, prefix_len: u32, capture_ordinal: u64) -> Result<[u8; 32], CoreError> {
        let doc = serde_json::json!({
            "game": self.game_id,
            "ordinal": self.hand_index,
            "prefix": prefix_len,
            // Capture ordinal, NOT the live applied count: on sequential paths
            // with no outstanding applies it equals the applied length (one
            // apply lands per captured row before the next stages), while
            // coalesced same-fill seats get consecutive distinct values.
            // Sharing one live-count value across a coalesced window would
            // collide provenances and fork row identities downstream.
            "applied": capture_ordinal,
        });
        let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:transition-provenance")
            .map_err(|e| CoreError(format!("provenance canon failed: {e:?}")))?;
        Ok(Sha256::digest(&bytes).into())
    }
}
