//! Replay-indexed offline delta-q producer and sidecar join helpers.

use std::collections::HashMap;
use std::io;
use std::io::BufRead;

use burn::prelude::Backend;
use hydra_core::action::{
    build_legal_mask, riichienv_to_hydra, ActionPhase, AKA_5M, AKA_5P, AKA_5S, DISCARD_END,
    HYDRA_ACTION_SPACE,
};
use hydra_core::arena::TrajectoryDeltaQLabel;
use hydra_core::bridge::encode_observation;
use hydra_core::safety::SafetyInfo;
use riichienv_core::replay::{mjai_event_actor, mjai_event_to_action, MjaiEvent};
use riichienv_core::state::GameState;
use serde::{Deserialize, Serialize};

use crate::data::mjai_loader::{
    bool_mask_to_f32, invalid_data, should_sample_replay_event, tile136_to_type, update_safety,
};
use crate::model::HydraModel;
use crate::training::delta_q_validation::DeltaQValidationReport;
use crate::training::exit::ExitConfig;
use crate::training::live_exit::{
    budget_from_legal_count, obs_hash, try_search_labels_from_context, RootDecisionContext,
    SelfPlayExitAdapter,
};
use crate::training::replay_exit::{
    legal_mask_digest_from_f32, source_hash_from_identity, ReplayDecisionKey,
};

pub const REPLAY_DELTA_Q_SEMANTICS_V1: &str = "delta_q_child_minus_root_v1";
pub const REPLAY_DELTA_Q_PROVENANCE: &str = "search-derived";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDeltaQLookupKey {
    pub replay: ReplayDecisionKey,
    pub action: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDeltaQRecordV1 {
    pub version: u32,
    pub semantics: String,
    pub provenance: String,
    pub key: ReplayDecisionKey,
    pub action: u8,
    pub legal_mask_digest: u64,
    pub source_net_hash: u64,
    pub source_version: u32,
    pub target: Vec<f32>,
    pub mask: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct DeltaQSidecarIndex {
    records: HashMap<ReplayDeltaQLookupKey, ReplayDeltaQRecordV1>,
}

impl DeltaQSidecarIndex {
    pub fn from_records(records: Vec<ReplayDeltaQRecordV1>) -> Self {
        let records = records
            .into_iter()
            .map(|record| {
                (
                    ReplayDeltaQLookupKey {
                        replay: record.key,
                        action: record.action,
                    },
                    record,
                )
            })
            .collect();
        Self { records }
    }

    pub fn lookup_label(
        &self,
        key: &ReplayDecisionKey,
        action: u8,
        legal_mask: &[f32; HYDRA_ACTION_SPACE],
        source_net_hash: u64,
        source_version: u32,
    ) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])> {
        let record = self.records.get(&ReplayDeltaQLookupKey {
            replay: *key,
            action,
        })?;
        if record.version != 1
            || record.semantics != REPLAY_DELTA_Q_SEMANTICS_V1
            || record.provenance != REPLAY_DELTA_Q_PROVENANCE
            || record.legal_mask_digest != legal_mask_digest_from_f32(legal_mask)
            || record.source_net_hash != source_net_hash
            || record.source_version != source_version
            || record.target.len() != HYDRA_ACTION_SPACE
            || record.mask.len() != HYDRA_ACTION_SPACE
        {
            return None;
        }
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target.copy_from_slice(&record.target);
        mask.copy_from_slice(&record.mask);
        let validated = validate_delta_q_contract(&target, &mask, legal_mask)?;
        Some((validated.target, validated.mask))
    }

    pub fn from_jsonl_reader(reader: impl BufRead) -> io::Result<Self> {
        let mut records = Vec::new();
        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ReplayDeltaQRecordV1 = serde_json::from_str(&line).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid replay delta_q sidecar line {}: {err}",
                        line_idx + 1
                    ),
                )
            })?;
            records.push(record);
        }
        Ok(Self::from_records(records))
    }

    pub fn from_jsonl_path(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_jsonl_reader(std::io::BufReader::new(file))
    }
}

fn validate_delta_q_contract(
    target: &[f32; HYDRA_ACTION_SPACE],
    mask: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
) -> Option<TrajectoryDeltaQLabel> {
    let label = TrajectoryDeltaQLabel::from_slices(target, mask)?;
    let mut saw_masked = false;
    for action_idx in 0..HYDRA_ACTION_SPACE {
        let mask_value = label.mask[action_idx];
        if mask_value < -1e-6 || ((mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6) {
            return None;
        }
        let target_value = label.target[action_idx];
        if !target_value.is_finite() {
            return None;
        }
        if mask_value > 0.5 {
            saw_masked = true;
            if legal_mask[action_idx] <= 0.0 {
                return None;
            }
            if action_idx > DISCARD_END as usize {
                return None;
            }
            if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
                return None;
            }
        } else if target_value.abs() > 1e-5 {
            return None;
        }
    }
    saw_masked.then_some(label)
}

pub fn generate_replay_delta_q_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)> {
    let mut state = GameState::new(
        0,
        true,
        Some(0),
        0,
        riichienv_core::rule::GameRule::default_tenhou(),
    );
    let mut safety = std::array::from_fn(|_| SafetyInfo::default());
    let mut encoder = hydra_core::encoder::ObservationEncoder::new();
    let mut adapter = SelfPlayExitAdapter::new();
    let mut records = Vec::new();
    let mut report = DeltaQValidationReport::new();

    for (idx, event) in events.iter().enumerate() {
        if should_sample_replay_event(event) {
            let env_action = mjai_event_to_action(event)
                .map_err(|err| invalid_data(format!("replay action conversion failed: {err}")))?;
            if let (Some(actor), Some(env_action)) = (mjai_event_actor(event), env_action) {
                let obs = state
                    .get_observation_for_replay(actor as u8, &env_action, &env_action.to_mjai())
                    .map_err(|err| invalid_data(format!("replay observation failed: {err}")))?;
                let hydra_action = riichienv_to_hydra(&env_action)
                    .map_err(|err| invalid_data(format!("hydra action mapping failed: {err}")))?;
                let legal = obs.legal_actions_method();
                let phase = if matches!(event, MjaiEvent::Dahai { .. })
                    && state.players[actor].riichi_declared
                {
                    ActionPhase::RiichiSelect
                } else {
                    ActionPhase::Normal
                };
                let legal_mask = build_legal_mask(&legal, phase);
                if legal_mask[hydra_action.id() as usize] {
                    let obs_encoded = encode_observation(
                        &mut encoder,
                        &obs,
                        &safety[actor],
                        state.drawn_tile.map(tile136_to_type),
                    );
                    let ctx = RootDecisionContext {
                        obs_encoded,
                        legal_mask,
                        policy_logits: model.policy_value_cpu(&obs_encoded, device).0,
                        player_id: actor as u8,
                    };
                    let key = ReplayDecisionKey {
                        source_hash,
                        event_index: idx as u32,
                        actor: actor as u8,
                        obs_hash: obs_hash(&ctx.obs_encoded),
                    };

                    report.total_states += 1;

                    let labels = try_search_labels_from_context(
                        &state,
                        &obs,
                        &ctx,
                        &safety[actor],
                        exit_cfg,
                        &mut |obs_encoded| model.policy_value_cpu(obs_encoded, device),
                        &mut adapter,
                    );

                    if let Some(delta_q) = labels.and_then(|labels| labels.delta_q) {
                        let legal_discard_count = ctx.legal_mask[..=DISCARD_END as usize]
                            .iter()
                            .filter(|&&is_legal| is_legal)
                            .count();
                        let supported_actions = delta_q.mask.iter().filter(|&&m| m > 0.0).count();
                        let coverage = if legal_discard_count == 0 {
                            0.0
                        } else {
                            supported_actions as f64 / legal_discard_count as f64
                        };
                        report.labels_emitted += 1;
                        report.coverage_sum += coverage;
                        report.supported_actions_sum += supported_actions as u64;
                        report.root_visits_sum +=
                            u64::from(budget_from_legal_count(exit_cfg, legal_discard_count));
                        for action_idx in 0..HYDRA_ACTION_SPACE {
                            if delta_q.mask[action_idx] <= 0.0 {
                                continue;
                            }
                            let value = delta_q.target[action_idx] as f64;
                            report.masked_abs_sum += value.abs();
                            report.masked_entry_count += 1;
                            if value > 0.0 {
                                report.masked_positive_count += 1;
                            } else if value < 0.0 {
                                report.masked_negative_count += 1;
                            } else {
                                report.masked_zero_count += 1;
                            }
                        }
                        records.push(ReplayDeltaQRecordV1 {
                            version: 1,
                            semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                            provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
                            key,
                            action: hydra_action.id(),
                            legal_mask_digest: legal_mask_digest_from_f32(&bool_mask_to_f32(
                                ctx.legal_mask,
                            )),
                            source_net_hash,
                            source_version,
                            target: delta_q.target.to_vec(),
                            mask: delta_q.mask.to_vec(),
                        });
                    } else {
                        report.labels_rejected += 1;
                        report.rejected_other += 1;
                    }
                }
            }
        }

        update_safety(&mut safety, event)?;
        state.apply_mjai_event(event.clone());
    }

    Ok((records, report))
}

pub fn replay_delta_q_records_for_identity<B: Backend>(
    source_identity: &str,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)> {
    generate_replay_delta_q_records(
        source_hash_from_identity(source_identity),
        events,
        model,
        device,
        exit_cfg,
        source_net_hash,
        source_version,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::mjai_loader::load_game_from_events_with_sidecar;
    use burn::backend::NdArray;
    use riichienv_core::replay::read_mjai_events;
    use std::io::Cursor;

    type B = NdArray<f32>;

    fn sample_log() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    #[test]
    fn sidecar_lookup_requires_matching_contract() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        mask[2] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        target[2] = 0.25;
        let record = ReplayDeltaQRecordV1 {
            version: 1,
            semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
            provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
            key,
            action: 2,
            legal_mask_digest: legal_mask_digest_from_f32(&mask),
            source_net_hash: 9,
            source_version: 1,
            target: target.to_vec(),
            mask: mask.to_vec(),
        };
        let index = DeltaQSidecarIndex::from_records(vec![record]);
        assert!(index.lookup_label(&key, 2, &mask, 9, 1).is_some());
        assert!(index.lookup_label(&key, 3, &mask, 9, 1).is_none());
        assert!(index.lookup_label(&key, 2, &mask, 10, 1).is_none());
        assert!(index.lookup_label(&key, 2, &mask, 9, 2).is_none());
    }

    #[test]
    fn loader_with_sidecar_populates_delta_q_fields() {
        let events = read_mjai_events(Cursor::new(sample_log())).expect("parse events");
        let device = Default::default();
        let model = crate::model::HydraModelConfig::learner().init::<B>(&device);
        let (records, _report) = replay_delta_q_records_for_identity(
            "game-1",
            &events,
            &model,
            &device,
            &ExitConfig::default_phase3(),
            123,
            1,
        )
        .expect("generate sidecar records");
        let index = DeltaQSidecarIndex::from_records(records);

        let game = load_game_from_events_with_sidecar("game-1", 123, 1, events, None, Some(&index))
            .expect("load with sidecar");
        assert!(game
            .samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some()));
        assert!(game
            .samples
            .iter()
            .any(|sample| sample.delta_q_mask.is_some()));
    }

    #[test]
    fn replay_delta_q_records_are_tagged_search_derived() {
        let device = Default::default();
        let model = crate::model::HydraModelConfig::learner().init::<B>(&device);
        let events = read_mjai_events(Cursor::new(sample_log())).expect("parse events");
        let (records, report) = replay_delta_q_records_for_identity(
            "game-1",
            &events,
            &model,
            &device,
            &ExitConfig::default_phase3(),
            123,
            1,
        )
        .expect("generate records");
        assert!(report.total_states > 0);
        for record in records {
            assert_eq!(record.provenance, REPLAY_DELTA_Q_PROVENANCE);
            assert_eq!(record.semantics, REPLAY_DELTA_Q_SEMANTICS_V1);
            assert_eq!(record.version, 1);
            assert!(record.action <= DISCARD_END);
        }
    }

    #[test]
    fn sidecar_lookup_rejects_invalid_delta_q_structure() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[2] = 1.0;
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        mask[40] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        target[40] = 0.25;
        let record = ReplayDeltaQRecordV1 {
            version: 1,
            semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
            provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
            key,
            action: 2,
            legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
            source_net_hash: 9,
            source_version: 1,
            target: target.to_vec(),
            mask: mask.to_vec(),
        };
        let index = DeltaQSidecarIndex::from_records(vec![record]);
        assert!(index.lookup_label(&key, 2, &legal_mask, 9, 1).is_none());
    }
}
