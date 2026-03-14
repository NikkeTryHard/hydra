//! Replay-indexed offline ExIt producer and sidecar join helpers.

use std::collections::HashMap;
use std::io;
use std::io::BufRead;

use burn::prelude::Backend;
use hydra_core::action::{ActionPhase, HYDRA_ACTION_SPACE, build_legal_mask, riichienv_to_hydra};
use hydra_core::bridge::encode_observation;
use hydra_core::safety::SafetyInfo;
use riichienv_core::replay::{MjaiEvent, mjai_event_actor, mjai_event_to_action};
use riichienv_core::state::GameState;
use serde::{Deserialize, Serialize};

use crate::data::mjai_loader::{
    bool_mask_to_f32, final_scores, invalid_data, should_sample_replay_event, tile136_to_type,
    update_safety,
};
use crate::model::HydraModel;
use crate::training::exit::ExitConfig;
use crate::training::exit_validation::ExitValidationReport;
use crate::training::live_exit::{
    RootDecisionContext, SelfPlayExitAdapter, budget_from_legal_count, obs_hash,
    try_exit_label_from_context,
};

pub const REPLAY_EXIT_SEMANTICS_V1: &str = "exit_root_child_visits_v1";
pub const REPLAY_EXIT_PROVENANCE: &str = "search-derived";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDecisionKey {
    pub source_hash: u64,
    pub event_index: u32,
    pub actor: u8,
    pub obs_hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayExitLookupKey {
    pub replay: ReplayDecisionKey,
    pub action: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayExitRecordV1 {
    pub version: u32,
    pub semantics: String,
    pub provenance: String,
    pub key: ReplayDecisionKey,
    pub action: u8,
    pub legal_mask_digest: u64,
    pub source_net_hash: u64,
    pub source_version: u32,
    pub root_visit_count: u32,
    pub legal_discard_count: u8,
    pub supported_actions: u8,
    pub coverage: f32,
    pub kl_to_base: f32,
    pub target: Vec<f32>,
    pub mask: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct ExitSidecarIndex {
    records: HashMap<ReplayExitLookupKey, ReplayExitRecordV1>,
}

impl ExitSidecarIndex {
    pub fn from_records(records: Vec<ReplayExitRecordV1>) -> Self {
        let records = records
            .into_iter()
            .map(|record| {
                (
                    ReplayExitLookupKey {
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
        let record = self.records.get(&ReplayExitLookupKey {
            replay: *key,
            action,
        })?;
        if record.version != 1
            || record.semantics != REPLAY_EXIT_SEMANTICS_V1
            || record.provenance != REPLAY_EXIT_PROVENANCE
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
        Some((target, mask))
    }

    pub fn from_jsonl_reader(reader: impl BufRead) -> io::Result<Self> {
        let mut records = Vec::new();
        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ReplayExitRecordV1 = serde_json::from_str(&line).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid replay ExIt sidecar line {}: {err}", line_idx + 1),
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

pub fn source_hash_from_identity(identity: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in identity.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn source_net_hash_from_checkpoint_identity(identity: &str) -> u64 {
    source_hash_from_identity(identity)
}

pub fn legal_mask_digest_from_f32(mask: &[f32; HYDRA_ACTION_SPACE]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in mask {
        hash ^= u64::from(value > 0.0);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn legal_mask_digest_from_bool(mask: &[bool; HYDRA_ACTION_SPACE]) -> u64 {
    legal_mask_digest_from_f32(&bool_mask_to_f32(*mask))
}

pub type ReplayExitAdapter = SelfPlayExitAdapter;

pub fn generate_replay_exit_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayExitRecordV1>, ExitValidationReport)> {
    let _ = final_scores(events);
    let mut state = GameState::new(
        0,
        true,
        Some(0),
        0,
        riichienv_core::rule::GameRule::default_tenhou(),
    );
    let mut safety = std::array::from_fn(|_| SafetyInfo::default());
    let mut encoder = hydra_core::encoder::ObservationEncoder::new();
    let mut adapter = ReplayExitAdapter::new();
    let mut records = Vec::new();
    let mut report = ExitValidationReport::new();

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
                    let label = try_exit_label_from_context(
                        &state,
                        &obs,
                        &ctx,
                        &safety[actor],
                        exit_cfg,
                        &mut |obs_encoded| model.policy_value_cpu(obs_encoded, device),
                        &mut adapter,
                    );

                    if let Some(label) = label {
                        let supported_actions =
                            label.mask.iter().filter(|&&m| m > 0.0).count() as u8;
                        let legal_discard_count = ctx.legal_mask[..=36]
                            .iter()
                            .filter(|&&is_legal| is_legal)
                            .count() as u8;
                        let coverage = if legal_discard_count == 0 {
                            0.0
                        } else {
                            supported_actions as f32 / legal_discard_count as f32
                        };
                        let base_pi = hydra_core::arena::softmax_temperature(
                            &ctx.policy_logits,
                            &ctx.legal_mask,
                            1.0,
                        );
                        let mut kl_to_base = 0.0f32;
                        for action in 0..HYDRA_ACTION_SPACE {
                            let p = label.target[action];
                            let q = base_pi[action];
                            if label.mask[action] > 0.0 && p > 1e-8 && q > 1e-8 {
                                kl_to_base += p * (p / q).ln();
                            }
                        }

                        report.labels_emitted += 1;
                        report.coverage_sum += coverage as f64;
                        report.supported_actions_sum += u64::from(supported_actions);
                        report.root_visits_sum += u64::from(budget_from_legal_count(
                            exit_cfg,
                            legal_discard_count as usize,
                        ));
                        report.kl_sum += kl_to_base as f64;

                        records.push(ReplayExitRecordV1 {
                            version: 1,
                            semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
                            provenance: REPLAY_EXIT_PROVENANCE.to_string(),
                            key,
                            action: hydra_action.id(),
                            legal_mask_digest: legal_mask_digest_from_bool(&ctx.legal_mask),
                            source_net_hash,
                            source_version,
                            root_visit_count: budget_from_legal_count(
                                exit_cfg,
                                legal_discard_count as usize,
                            ),
                            legal_discard_count,
                            supported_actions,
                            coverage,
                            kl_to_base,
                            target: label.target.to_vec(),
                            mask: label.mask.to_vec(),
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

pub fn replay_exit_records_for_identity<B: Backend>(
    source_identity: &str,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayExitRecordV1>, ExitValidationReport)> {
    generate_replay_exit_records(
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
    use hydra_core::action::DISCARD_END;
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
    fn legal_mask_digest_changes_when_support_changes() {
        let mut a = [0.0f32; HYDRA_ACTION_SPACE];
        let mut b = [0.0f32; HYDRA_ACTION_SPACE];
        a[0] = 1.0;
        b[1] = 1.0;
        assert_ne!(
            legal_mask_digest_from_f32(&a),
            legal_mask_digest_from_f32(&b)
        );
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
        target[2] = 1.0;
        let record = ReplayExitRecordV1 {
            version: 1,
            semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
            provenance: REPLAY_EXIT_PROVENANCE.to_string(),
            key,
            action: 2,
            legal_mask_digest: legal_mask_digest_from_f32(&mask),
            source_net_hash: 9,
            source_version: 1,
            root_visit_count: 64,
            legal_discard_count: 1,
            supported_actions: 1,
            coverage: 1.0,
            kl_to_base: 0.0,
            target: target.to_vec(),
            mask: mask.to_vec(),
        };
        let index = ExitSidecarIndex::from_records(vec![record]);
        assert!(index.lookup_label(&key, 2, &mask, 9, 1).is_some());
        assert!(index.lookup_label(&key, 3, &mask, 9, 1).is_none());
        assert!(index.lookup_label(&key, 2, &mask, 10, 1).is_none());
        assert!(index.lookup_label(&key, 2, &mask, 9, 2).is_none());
    }

    #[test]
    fn sidecar_index_keeps_distinct_actions_for_same_replay_state() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        mask[2] = 1.0;
        mask[5] = 1.0;
        let mut target_a = [0.0f32; HYDRA_ACTION_SPACE];
        target_a[2] = 1.0;
        let mut target_b = [0.0f32; HYDRA_ACTION_SPACE];
        target_b[5] = 1.0;
        let records = vec![
            ReplayExitRecordV1 {
                version: 1,
                semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
                provenance: REPLAY_EXIT_PROVENANCE.to_string(),
                key,
                action: 2,
                legal_mask_digest: legal_mask_digest_from_f32(&mask),
                source_net_hash: 9,
                source_version: 1,
                root_visit_count: 64,
                legal_discard_count: 2,
                supported_actions: 2,
                coverage: 1.0,
                kl_to_base: 0.0,
                target: target_a.to_vec(),
                mask: mask.to_vec(),
            },
            ReplayExitRecordV1 {
                version: 1,
                semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
                provenance: REPLAY_EXIT_PROVENANCE.to_string(),
                key,
                action: 5,
                legal_mask_digest: legal_mask_digest_from_f32(&mask),
                source_net_hash: 9,
                source_version: 1,
                root_visit_count: 64,
                legal_discard_count: 2,
                supported_actions: 2,
                coverage: 1.0,
                kl_to_base: 0.0,
                target: target_b.to_vec(),
                mask: mask.to_vec(),
            },
        ];
        let index = ExitSidecarIndex::from_records(records);
        assert_eq!(index.lookup_label(&key, 2, &mask, 9, 1).unwrap().0[2], 1.0);
        assert_eq!(index.lookup_label(&key, 5, &mask, 9, 1).unwrap().0[5], 1.0);
    }

    #[test]
    fn replay_exit_records_are_tagged_search_derived() {
        let device = Default::default();
        let model = crate::model::HydraModelConfig::learner().init::<B>(&device);
        let events = read_mjai_events(Cursor::new(sample_log())).expect("parse events");
        let (records, _report) = replay_exit_records_for_identity(
            "game-1",
            &events,
            &model,
            &device,
            &ExitConfig::default_phase3(),
            123,
            1,
        )
        .expect("generate records");
        for record in records {
            assert_eq!(record.provenance, REPLAY_EXIT_PROVENANCE);
            assert_eq!(record.semantics, REPLAY_EXIT_SEMANTICS_V1);
            assert_eq!(record.version, 1);
            assert!(record.action <= DISCARD_END);
        }
    }

    #[test]
    fn loader_with_sidecar_populates_exit_fields() {
        let events = read_mjai_events(Cursor::new(sample_log())).expect("parse events");
        let device = Default::default();
        let model = crate::model::HydraModelConfig::learner().init::<B>(&device);
        let (records, _report) = replay_exit_records_for_identity(
            "game-1",
            &events,
            &model,
            &device,
            &ExitConfig::default_phase3(),
            123,
            1,
        )
        .expect("generate sidecar records");
        let index = ExitSidecarIndex::from_records(records);

        let game = load_game_from_events_with_sidecar("game-1", 123, 1, events, Some(&index))
            .expect("load with sidecar");
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.exit_target.is_some())
        );
        assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
    }
}
