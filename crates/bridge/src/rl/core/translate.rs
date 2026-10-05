//! rl/core/translate: mjai record batch translation to typed envelopes.
//!
//! Each drained mjai batch is walked in order: game and round boundaries,
//! draws, discards, riichi declarations and acceptances, open calls, closed
//! quads, added quads, dora reveals, wins (merged across consecutive records),
//! draws and abortives, round end, and game end. Every envelope is built with
//! an explicit payload and public delta, then admitted to the arena or the
//! resolution staging area. A trailing lone declaration that needs its pair
//! stops the batch without consuming it. Round end either carries to the next
//! hand, steps the engine on an empty tail, or continues; game end freezes
//! the terminal outcome and marks the game terminal. Unknown record types
//! fail closed. No wall law, no RNG, no Python.

use std::collections::HashMap;

use hydra_feed::census::ActionKind;

use crate::rl::events::{
    DeltaPathSeg, StateDelta, Visibility, abortive_literal, meld_doc_value, reason_kind,
};

use super::{BatchOutcome, CoreError, HandCarry, SettlementFact, TerminalOutcome, TypedPayload, canonical_scores};

impl super::NativeGame {
    pub(super) fn translate_batch(
        &mut self,
        batch: &[serde_json::Value],
        consumed: &mut usize,
    ) -> Result<BatchOutcome, CoreError> {
        let mut index = 0usize;
        while index < batch.len() {
            let record = &batch[index];
            let kind = record.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
                CoreError("core mjai record without a type".to_string())
            })?;
            match kind {
                "start_game" => {
                    index += 1;
                }
                "start_kyoku" => {
                    self.on_start_kyoku(record)?;
                    index += 1;
                }
                "tsumo" => {
                    self.on_tsumo(record)?;
                    index += 1;
                }
                "dahai" => {
                    self.on_dahai(record)?;
                    index += 1;
                }
                "reach" => {
                    if index + 1 >= batch.len() {
                        break;
                    }
                    self.on_reach(record, &batch[index + 1])?;
                    index += 2;
                }
                "reach_accepted" => {
                    self.on_reach_accepted(record)?;
                    index += 1;
                }
                "chi" | "pon" | "daiminkan" => {
                    self.on_call(record, kind)?;
                    index += 1;
                }
                "ankan" => {
                    self.on_ankan(record)?;
                    index += 1;
                }
                "kakan" => {
                    self.on_kakan(record)?;
                    index += 1;
                }
                "dora" => {
                    self.on_dora()?;
                    index += 1;
                }
                "hora" => {
                    let mut merged = vec![record.clone()];
                    let mut position = index + 1;
                    while position < batch.len()
                        && batch[position].get("type").and_then(|t| t.as_str()) == Some("hora")
                    {
                        merged.push(batch[position].clone());
                        position += 1;
                    }
                    self.on_hora(&merged)?;
                    index = position;
                    continue;
                }
                "ryukyoku" => {
                    self.on_ryukyoku(record)?;
                    index += 1;
                }
                "end_kyoku" => {
                    let following = batch.get(index + 1);
                    self.on_end_kyoku(following)?;
                    if let Some(next) = following
                        && next.get("type").and_then(|t| t.as_str()) == Some("start_kyoku")
                    {
                        *consumed = index + 2;
                        return Ok(BatchOutcome::Carry(carry_of(next)?));
                    }
                    if following.is_none() {
                        *consumed = index + 1;
                        let pre = self.state.mjai_log.len();
                        self.state.step(&HashMap::new());
                        self.check_error()?;
                        self.note_step_draws(pre);
                        return Ok(BatchOutcome::Continue);
                    }
                    index += 1;
                }
                "end_game" => {
                    self.on_end_game()?;
                    *consumed = index + 1;
                    return Ok(BatchOutcome::Terminal);
                }
                other => {
                    return Err(CoreError(format!("unexpected mjai event type {other:?}")));
                }
            }
        }
        *consumed = index;
        Ok(BatchOutcome::Continue)
    }

    fn mjai_actor(&self, record: &serde_json::Value) -> Result<u8, CoreError> {
        let engine = record.get("actor").and_then(|a| a.as_u64()).ok_or_else(|| {
            CoreError("mjai record without an actor".to_string())
        })? as u8;
        Ok(self.inv[engine as usize] as u8)
    }

    fn mjai_u64(&self, record: &serde_json::Value, key: &str) -> Result<u64, CoreError> {
        record.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
            CoreError(format!("mjai record without {key:?}"))
        })
    }

    fn on_start_kyoku(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let engine_oya = self.mjai_u64(record, "oya")? as u8;
        let dealer = self.inv[engine_oya as usize] as u8;
        let scores = canonical_scores(&self.state, self.inv);
        let bakaze = record.get("bakaze").and_then(|b| b.as_str()).unwrap_or("E");
        if !matches!(bakaze, "E" | "S" | "W" | "N") {
            return Err(CoreError(format!("bad bakaze {bakaze:?}")));
        }
        let honba = record.get("honba").and_then(|h| h.as_u64()).unwrap_or(0) as i64;
        let kyotaku = record.get("kyotaku").and_then(|k| k.as_u64()).unwrap_or(0) as i64;
        let deltas = vec![
            StateDelta {
                path: vec![DeltaPathSeg::Key("round_index".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(self.hand_index),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("honba".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(honba),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("riichi_sticks".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(kyotaku),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("scores".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(scores.to_vec()),
            },
        ];
        let envelope = self.make_envelope(
            "round_start",
            Some(dealer),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                round_index: Some(self.hand_index),
                scores: Some(scores),
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        Ok(())
    }

    fn on_tsumo(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let actor = self.mjai_actor(record)?;
        if self.last_discard.0 == Some(actor) && self.window_offered_ids.is_empty() {
            self.last_discard = (None, None);
        }
        let tile = self.exact_draw()?;
        let advance = self.make_envelope(
            "turn_advance",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload::empty(),
            vec![],
        )?;
        self.emit(advance)?;
        let draw = self.make_envelope(
            "draw_tile",
            Some(actor),
            Visibility::ActorPrivate,
            vec![actor],
            TypedPayload { tile: Some(tile), ..TypedPayload::empty() },
            vec![],
        )?;
        self.emit(draw)?;
        // Retain for the hand (oracle `_drawn` cache): responders observe
        // this tile long after discarding; only a new hand clears it.
        self.drawn_cache[actor as usize] = Some(tile);
        Ok(())
    }

    fn on_dahai(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let engine_pid = self.mjai_u64(record, "actor")? as u8;
        let actor = self.inv[engine_pid as usize] as u8;
        let pai = record.get("pai").and_then(|p| p.as_str()).unwrap_or("");
        let tile = self.next_discard(engine_pid, pai)?;
        let mut action_id = self.stamp_if(actor, ActionKind::Discard);
        if action_id.is_none() {
            action_id = self.stamp_if(actor, ActionKind::Tsumogiri);
        }
        let action_id = match action_id {
            Some(id) => id,
            None => self.fallback_discard_id(actor, tile)?,
        };
        let envelope = self.make_envelope(
            "discard",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                tile: Some(tile),
                action_id: Some(action_id),
                ..TypedPayload::empty()
            },
            vec![],
        )?;
        self.emit(envelope)?;
        self.last_discard = (Some(actor), Some(tile));
        self.window_opened_by_discard = true;
        self.ippatsu[actor as usize] = false;
        self.stamp = None;
        Ok(())
    }

    fn on_reach(
        &mut self,
        record: &serde_json::Value,
        declaration: &serde_json::Value,
    ) -> Result<(), CoreError> {
        let actor = self.mjai_actor(record)?;
        let engine_pid = declaration.get("actor").and_then(|a| a.as_u64()).ok_or_else(|| {
            CoreError("reach declaration without an actor".to_string())
        })? as u8;
        let pai = declaration.get("pai").and_then(|p| p.as_str()).unwrap_or("");
        let tile = self.peek_discard(engine_pid, pai)?;
        let action_id = self.stamp_if(actor, ActionKind::RiichiDiscard).ok_or_else(|| {
            CoreError("reach event without an applied riichi_discard".to_string())
        })?;
        let deltas = vec![StateDelta {
            path: vec![
                DeltaPathSeg::Key("riichi_states".to_string()),
                DeltaPathSeg::Index(actor),
            ],
            operation: "set".to_string(),
            value: serde_json::Value::String("declared".to_string()),
        }];
        let envelope = self.make_envelope(
            "riichi_declared",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                tile: Some(tile),
                action_id: Some(action_id),
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        self.riichi_states[actor as usize] = "declared".to_string();
        Ok(())
    }

    fn on_reach_accepted(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let actor = self.mjai_actor(record)?;
        let deltas = vec![
            StateDelta {
                path: vec![
                    DeltaPathSeg::Key("riichi_states".to_string()),
                    DeltaPathSeg::Index(actor),
                ],
                operation: "set".to_string(),
                value: serde_json::Value::String("accepted".to_string()),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("riichi_sticks".to_string())],
                operation: "increment".to_string(),
                value: serde_json::Value::from(1),
            },
            StateDelta {
                path: vec![
                    DeltaPathSeg::Key("ippatsu".to_string()),
                    DeltaPathSeg::Index(actor),
                ],
                operation: "set".to_string(),
                value: serde_json::Value::Bool(true),
            },
        ];
        let envelope = self.make_envelope(
            "riichi_accepted",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload::empty(),
            deltas,
        )?;
        self.emit(envelope)?;
        self.ippatsu[actor as usize] = true;
        self.riichi_states[actor as usize] = "accepted".to_string();
        Ok(())
    }

    fn on_call(&mut self, record: &serde_json::Value, kind: &str) -> Result<(), CoreError> {
        let engine_pid = self.mjai_u64(record, "actor")? as u8;
        let actor = self.inv[engine_pid as usize] as u8;
        let engine_target = self.mjai_u64(record, "target")? as u8;
        let called = self.last_river_tile(engine_target)?;
        let meld_tiles = self.latest_meld_tiles(engine_pid)?;
        let consumed: Vec<u8> = {
            let mut v: Vec<u8> = meld_tiles.into_iter().filter(|t| *t != called).collect();
            v.sort_unstable();
            v
        };
        let action_id = self.accepted_claim_id(actor)?;
        // Open calls carry an explicit engine target even when a prior drain
        // cleared last_discard; dropping that source creates an invalid event.
        let source = Some(self.inv[engine_target as usize] as u8);
        let mut deltas = vec![StateDelta {
            path: vec![
                DeltaPathSeg::Key("melds".to_string()),
                DeltaPathSeg::Index(actor),
            ],
            operation: "append".to_string(),
            value: meld_doc_value(
                kind,
                actor,
                source,
                Some(called),
                &[consumed.clone(), vec![called]].concat(),
            ),
        }];
        if kind == "daiminkan" {
            deltas.push(StateDelta {
                path: vec![DeltaPathSeg::Key("kan_count".to_string())],
                operation: "increment".to_string(),
                value: serde_json::Value::from(1),
            });
            self.kan_count += 1;
        }
        let envelope = self.make_envelope(
            kind,
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                tile: Some(called),
                action_id: Some(action_id),
                source_seat: source,
                consumed_tiles: consumed,
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        self.last_discard = (None, None);
        self.ippatsu = [false; 4];
        self.stamp = None;
        Ok(())
    }

    fn on_ankan(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let engine_pid = self.mjai_u64(record, "actor")? as u8;
        let actor = self.inv[engine_pid as usize] as u8;
        let consumed = self.latest_meld_tiles(engine_pid)?;
        let action_id = self.stamp_if(actor, ActionKind::Ankan);
        let deltas = vec![
            StateDelta {
                path: vec![
                    DeltaPathSeg::Key("melds".to_string()),
                    DeltaPathSeg::Index(actor),
                ],
                operation: "append".to_string(),
                value: meld_doc_value("ankan", actor, None, None, &consumed),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("kan_count".to_string())],
                operation: "increment".to_string(),
                value: serde_json::Value::from(1),
            },
        ];
        let envelope = self.make_envelope(
            "ankan",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                action_id,
                consumed_tiles: consumed,
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        self.kan_count += 1;
        self.ippatsu = [false; 4];
        self.stamp = None;
        Ok(())
    }

    fn on_kakan(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let engine_pid = self.mjai_u64(record, "actor")? as u8;
        let actor = self.inv[engine_pid as usize] as u8;
        let meld_tiles = self.latest_meld_tiles(engine_pid)?;
        let added = self.kakan_added.remove(&engine_pid).ok_or_else(|| {
            CoreError(format!("kakan by engine seat {engine_pid}: no captured added tile"))
        })?;
        let consumed: Vec<u8> = {
            let mut v: Vec<u8> = meld_tiles.into_iter().filter(|t| *t != added).collect();
            v.sort_unstable();
            v
        };
        let action_id = self.stamp_if(actor, ActionKind::Kakan);
        self.window_opened_by_discard = false;
        let deltas = vec![
            StateDelta {
                path: vec![
                    DeltaPathSeg::Key("melds".to_string()),
                    DeltaPathSeg::Index(actor),
                ],
                operation: "append".to_string(),
                value: meld_doc_value(
                    "kakan",
                    actor,
                    None,
                    None,
                    &[consumed.clone(), vec![added]].concat(),
                ),
            },
            StateDelta {
                path: vec![DeltaPathSeg::Key("kan_count".to_string())],
                operation: "increment".to_string(),
                value: serde_json::Value::from(1),
            },
        ];
        let envelope = self.make_envelope(
            "kakan",
            Some(actor),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                tile: Some(added),
                action_id,
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        self.kan_count += 1;
        self.ippatsu = [false; 4];
        self.last_discard = (Some(actor), Some(added));
        self.stamp = None;
        Ok(())
    }

    fn on_dora(&mut self) -> Result<(), CoreError> {
        let tile = *self.state.wall.dora_indicators.last().ok_or_else(|| {
            CoreError("dora event without an indicator".to_string())
        })?;
        let deltas = vec![StateDelta {
            path: vec![DeltaPathSeg::Key("dora_indicators".to_string())],
            operation: "append".to_string(),
            value: serde_json::Value::from(u32::from(tile)),
        }];
        let envelope = self.make_envelope(
            "dora_revealed",
            None,
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload { tile: Some(tile), ..TypedPayload::empty() },
            deltas,
        )?;
        self.emit(envelope)?;
        Ok(())
    }

    fn on_hora(&mut self, horas: &[serde_json::Value]) -> Result<(), CoreError> {
        let first = &horas[0];
        let engine_winner = first.get("actor").and_then(|a| a.as_u64()).ok_or_else(|| {
            CoreError("hora without an actor".to_string())
        })? as u8;
        let winner = self.inv[engine_winner as usize] as u8;
        let self_draw = first.get("tsumo").and_then(|t| t.as_bool()).unwrap_or(false);
        let tile = if self_draw { self.state.drawn_tile } else { self.last_discard.1 };
        let tile = tile
            .ok_or_else(|| CoreError("hora without a winning tile".to_string()))?;
        let kind = if self_draw { ActionKind::Tsumo } else { ActionKind::Ron };
        let mut action_id = self.stamp_if(winner, kind);
        if action_id.is_none() && !self_draw && !self.buffered.is_empty() {
            action_id = self.buffered.get(&winner).map(|b| b.canonical_id);
        }
        let winners: Vec<u8> = {
            let mut w: Vec<u8> = horas
                .iter()
                .filter_map(|h| h.get("actor").and_then(|a| a.as_u64()))
                .map(|e| self.inv[e as usize] as u8)
                .collect();
            w.sort_unstable();
            w
        };
        let mut deltas = [0i32; 4];
        for hora in horas {
            if let Some(values) = hora.get("deltas").and_then(|d| d.as_array()) {
                for (seat, value) in values.iter().enumerate().take(4) {
                    deltas[seat] += value.as_i64().unwrap_or(0) as i32;
                }
            }
        }
        let kind_str = if self_draw { "tsumo" } else { "ron" };
        let mut public = vec![StateDelta {
            path: vec![DeltaPathSeg::Key("scores".to_string())],
            operation: "set".to_string(),
            value: serde_json::Value::from(deltas.to_vec()),
        }];
        if !self_draw && self.state.riichi_sticks != 0 {
            public.push(StateDelta {
                path: vec![DeltaPathSeg::Key("riichi_sticks".to_string())],
                operation: "increment".to_string(),
                value: serde_json::Value::from(-(self.state.riichi_sticks as i32)),
            });
        }
        let source = if self_draw {
            None
        } else {
            first
                .get("target")
                .and_then(|t| t.as_u64())
                .map(|e| self.inv[e as usize] as u8)
        };
        let envelope = self.make_envelope(
            kind_str,
            Some(winner),
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                tile: Some(tile),
                action_id,
                source_seat: source,
                ..TypedPayload::empty()
            },
            public,
        )?;
        self.emit(envelope)?;
        self.settlements.push(SettlementFact {
            kind: kind_str.to_string(),
            deltas,
            payer: source,
            winners,
        });
        self.stamp = None;
        Ok(())
    }

    fn on_ryukyoku(&mut self, record: &serde_json::Value) -> Result<(), CoreError> {
        let reason = record.get("reason").and_then(|r| r.as_str()).unwrap_or("");
        let classification = reason_kind(reason)?;
        let mut deltas = [0i32; 4];
        if let Some(values) = record.get("deltas").and_then(|d| d.as_array()) {
            for (seat, value) in values.iter().enumerate().take(4) {
                deltas[seat] = value.as_i64().unwrap_or(0) as i32;
            }
        }
        let pre = canonical_scores(&self.state, self.inv);
        let post: [i32; 4] = std::array::from_fn(|i| pre[i] + deltas[i]);
        let public = vec![StateDelta {
            path: vec![DeltaPathSeg::Key("scores".to_string())],
            operation: "set".to_string(),
            value: serde_json::Value::from(deltas.to_vec()),
        }];
        if classification == "abortive_draw" {
            let literal = abortive_literal(reason).unwrap_or(reason);
            let envelope = self.make_envelope(
                "abortive_draw",
                None,
                Visibility::Public,
                vec![0, 1, 2, 3],
                TypedPayload {
                    round_index: Some(self.hand_index),
                    scores: Some(post),
                    reason: Some(literal.to_string()),
                    ..TypedPayload::empty()
                },
                public,
            )?;
            self.emit(envelope)?;
        } else {
            let envelope = self.make_envelope(
                "draw_end",
                None,
                Visibility::Public,
                vec![0, 1, 2, 3],
                TypedPayload {
                    scores: Some(post),
                    reason: Some(reason.to_string()),
                    ..TypedPayload::empty()
                },
                public,
            )?;
            self.emit(envelope)?;
        }
        self.settlements.push(SettlementFact {
            kind: classification.to_string(),
            deltas,
            payer: None,
            winners: (0..4u8).filter(|s| deltas[*s as usize] > 0).collect(),
        });
        Ok(())
    }

    fn on_end_kyoku(&mut self, following: Option<&serde_json::Value>) -> Result<(), CoreError> {
        let final_scores = canonical_scores(&self.state, self.inv);
        let mut deltas = vec![StateDelta {
            path: vec![DeltaPathSeg::Key("scores".to_string())],
            operation: "set".to_string(),
            value: serde_json::Value::from(final_scores.to_vec()),
        }];
        for seat in 0..4u8 {
            deltas.push(StateDelta {
                path: vec![
                    DeltaPathSeg::Key("riichi_states".to_string()),
                    DeltaPathSeg::Index(seat),
                ],
                operation: "set".to_string(),
                value: serde_json::Value::String("none".to_string()),
            });
        }
        if let Some(next) = following
            && next.get("type").and_then(|t| t.as_str()) == Some("start_kyoku")
        {
            let honba = next.get("honba").and_then(|h| h.as_u64()).unwrap_or(0);
            let kyotaku = next.get("kyotaku").and_then(|k| k.as_u64()).unwrap_or(0);
            deltas.push(StateDelta {
                path: vec![DeltaPathSeg::Key("honba".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(honba),
            });
            deltas.push(StateDelta {
                path: vec![DeltaPathSeg::Key("riichi_sticks".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(kyotaku),
            });
        }
        for seat in 0..4u8 {
            deltas.push(StateDelta {
                path: vec![
                    DeltaPathSeg::Key("ippatsu".to_string()),
                    DeltaPathSeg::Index(seat),
                ],
                operation: "set".to_string(),
                value: serde_json::Value::Bool(false),
            });
        }
        let envelope = self.make_envelope(
            "round_end",
            None,
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                round_index: Some(self.hand_index),
                scores: Some(final_scores),
                ..TypedPayload::empty()
            },
            deltas,
        )?;
        self.emit(envelope)?;
        Ok(())
    }

    fn on_end_game(&mut self) -> Result<(), CoreError> {
        let end_scores = canonical_scores(&self.state, self.inv);
        let envelope = self.make_envelope(
            "game_end",
            None,
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                round_index: Some(self.hand_index),
                scores: Some(end_scores),
                reason: Some("hanchan_complete".to_string()),
                ..TypedPayload::empty()
            },
            vec![StateDelta {
                path: vec![DeltaPathSeg::Key("scores".to_string())],
                operation: "set".to_string(),
                value: serde_json::Value::from(end_scores.to_vec()),
            }],
        )?;
        self.emit(envelope)?;
        let ranks64 = hydra_feed::fixed::resolve_final_ranks([
            i64::from(end_scores[0]),
            i64::from(end_scores[1]),
            i64::from(end_scores[2]),
            i64::from(end_scores[3]),
        ]);
        let ranks = [ranks64[0], ranks64[1], ranks64[2], ranks64[3]];
        let point_deltas: [i32; 4] =
            std::array::from_fn(|i| end_scores[i] - self.starting_scores[i]);
        let utility_g4: [f64; 4] =
            std::array::from_fn(|i| self.ctx.rank_values[ranks[i] as usize - 1]);
        self.outcome = Some(TerminalOutcome {
            final_scores: end_scores,
            ranks,
            point_deltas,
            utility_g4,
        });
        self.terminal = true;
        Ok(())
    }
}

fn carry_of(record: &serde_json::Value) -> Result<HandCarry, CoreError> {
    let values = record.get("scores").and_then(|s| s.as_array()).ok_or_else(|| {
        CoreError("start_kyoku without scores".to_string())
    })?;
    if values.len() != 4 {
        return Err(CoreError("start_kyoku scores must hold four seats".to_string()));
    }
    Ok(HandCarry {
        oya: record.get("oya").and_then(|o| o.as_u64()).unwrap_or(0) as u8,
        honba: record.get("honba").and_then(|h| h.as_u64()).unwrap_or(0) as u8,
        kyotaku: record.get("kyotaku").and_then(|k| k.as_u64()).unwrap_or(0) as u32,
        scores: std::array::from_fn(|i| values[i].as_i64().unwrap_or(0) as i32),
        bakaze: record.get("bakaze").and_then(|b| b.as_str()).unwrap_or("E").to_string(),
    })
}
