//! Replay-indexed offline delta-q producer and sidecar join helpers.

use std::io;

use burn::prelude::Backend;
use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::safety::SafetyInfo;
use riichienv_core::replay::MjaiEvent;
use riichienv_core::state::GameState;

pub use hydra_replay_sidecar::{
    DeltaQSidecarIndex, REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1,
    ReplayDeltaQLookupKey, ReplayDeltaQRecordV1, validate_delta_q_contract,
};

use crate::delta_q_validation::DeltaQValidationReport;
use crate::exit::ExitConfig;
use crate::live_exit::{
    RootDecisionContext, SelfPlayExitAdapter, budget_from_legal_count, obs_hash,
    try_search_labels_from_context_with_batched_child_values,
};
use crate::replay_exit::{
    ReplayDecisionKey, legal_mask_digest_from_f32, source_hash_from_identity,
};
use hydra_model::model::HydraModel;
use hydra_replay_loader::mjai_loader::{prepare_replay_decision, update_safety};
use hydra_replay_loader::replay_targets::bool_mask_to_f32;

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
    let mut flat_buf = Vec::new();
    let mut values_buf = Vec::new();
    let mut records = Vec::new();
    let mut report = DeltaQValidationReport::new();

    for (idx, event) in events.iter().enumerate() {
        if let Some(decision) = prepare_replay_decision(event, &mut state, &safety, &mut encoder)? {
            let actor = decision.actor;
            let ctx = RootDecisionContext {
                obs_encoded: decision.obs_encoded,
                legal_mask: decision.legal_mask,
                policy_logits: model.policy_cpu(&decision.obs_encoded, device),
                player_id: actor as u8,
            };
            let key = ReplayDecisionKey {
                source_hash,
                event_index: idx as u32,
                actor: actor as u8,
                obs_hash: obs_hash(&ctx.obs_encoded),
            };

            report.total_states += 1;
            let labels = try_search_labels_from_context_with_batched_child_values(
                &state,
                &decision.obs,
                &ctx,
                &safety[actor],
                exit_cfg,
                &mut |child_obs| {
                    model.fill_batch_value_cpu(child_obs, device, &mut flat_buf, &mut values_buf);
                    values_buf.clone()
                },
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
                    action: decision.action_id,
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
    use hydra_core::action::AKA_5M;
    use hydra_core::encoder::ObservationEncoder;
    use hydra_replay_loader::mjai_loader::load_game_from_events_with_sidecar;
    use riichienv_core::replay::read_mjai_events;
    use riichienv_core::rule::GameRule;
    use std::array;
    use std::io::Cursor;
    use std::io::ErrorKind;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_jsonl_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra-{name}-{nanos}.jsonl"))
    }

    fn guardrail_log() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["6m","6m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p","7p","8p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"8p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn synthetic_delta_q_records(
        source_net_hash: u64,
        source_version: u32,
    ) -> Vec<ReplayDeltaQRecordV1> {
        let events = read_mjai_events(Cursor::new(guardrail_log())).expect("parse events");
        let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
        let mut safety = array::from_fn(|_| SafetyInfo::default());
        let mut encoder = ObservationEncoder::new();

        let mut records = Vec::new();

        for (idx, event) in events.iter().enumerate() {
            if let Some(decision) =
                prepare_replay_decision(event, &mut state, &safety, &mut encoder)
                    .expect("prepare replay decision")
            {
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                mask[decision.action_id as usize] = 1.0;
                target[decision.action_id as usize] = 0.25;
                records.push(ReplayDeltaQRecordV1 {
                    version: 1,
                    semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                    provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
                    key: ReplayDecisionKey {
                        source_hash: source_hash_from_identity("game-1"),
                        event_index: idx as u32,
                        actor: decision.actor as u8,
                        obs_hash: obs_hash(&decision.obs_encoded),
                    },
                    action: decision.action_id,
                    legal_mask_digest: legal_mask_digest_from_f32(&decision.legal_mask_f32),
                    source_net_hash,
                    source_version,
                    target: target.to_vec(),
                    mask: mask.to_vec(),
                });
                if records.len() == 2 {
                    update_safety(&mut safety, event).expect("update safety");
                    state.apply_mjai_event(event.clone());
                    break;
                }
            }
            update_safety(&mut safety, event).expect("update safety");
            state.apply_mjai_event(event.clone());
        }

        records
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
        let events = read_mjai_events(Cursor::new(guardrail_log())).expect("parse events");
        let records = synthetic_delta_q_records(123, 1);
        let index = DeltaQSidecarIndex::from_records(records);

        let game = load_game_from_events_with_sidecar(
            "game-1",
            hydra_replay_loader::mjai_loader::SidecarProvenance::default(),
            hydra_replay_loader::mjai_loader::SidecarProvenance::new(Some(123), Some(1)),
            hydra_replay_loader::mjai_loader::ReplayTargetProfile::with_optional_heads(
                false, false, false, false, false, true,
            ),
            events,
            None,
            Some(&index),
        )
        .expect("load with sidecar");
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_target.is_some())
        );
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_mask.is_some())
        );
    }

    #[test]
    fn duplicate_delta_q_sidecar_key_last_record_wins() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        mask[2] = 1.0;
        let mut first_target = [0.0f32; HYDRA_ACTION_SPACE];
        first_target[2] = 0.25;
        let mut second_target = [0.0f32; HYDRA_ACTION_SPACE];
        second_target[2] = -0.5;
        let records = vec![
            ReplayDeltaQRecordV1 {
                version: 1,
                semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
                key,
                action: 2,
                legal_mask_digest: legal_mask_digest_from_f32(&mask),
                source_net_hash: 9,
                source_version: 1,
                target: first_target.to_vec(),
                mask: mask.to_vec(),
            },
            ReplayDeltaQRecordV1 {
                version: 1,
                semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
                key,
                action: 2,
                legal_mask_digest: legal_mask_digest_from_f32(&mask),
                source_net_hash: 9,
                source_version: 1,
                target: second_target.to_vec(),
                mask: mask.to_vec(),
            },
        ];
        let index = DeltaQSidecarIndex::from_records(records);
        let (target, loaded_mask) = index.lookup_label(&key, 2, &mask, 9, 1).expect("lookup");
        assert!((target[2] + 0.5).abs() < 1e-6);
        assert_eq!(loaded_mask[2], 1.0);
    }

    #[test]
    fn replay_delta_q_records_are_tagged_search_derived() {
        let records = synthetic_delta_q_records(123, 1);
        assert!(!records.is_empty());
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

    #[test]
    fn delta_q_contract_rejects_missing_masked_actions_and_nonzero_unmasked_targets() {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[2] = 1.0;

        let target = [0.0f32; HYDRA_ACTION_SPACE];
        let mask = [0.0f32; HYDRA_ACTION_SPACE];
        assert!(validate_delta_q_contract(&target, &mask, &legal_mask).is_none());

        let mut bad_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut bad_mask = [0.0f32; HYDRA_ACTION_SPACE];
        bad_target[4] = 0.25;
        bad_mask[2] = 1.0;
        assert!(validate_delta_q_contract(&bad_target, &bad_mask, &legal_mask).is_none());
    }

    #[test]
    fn delta_q_contract_rejects_aka_actions_even_when_legal() {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[AKA_5M as usize] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[AKA_5M as usize] = 0.1;
        mask[AKA_5M as usize] = 1.0;
        assert!(validate_delta_q_contract(&target, &mask, &legal_mask).is_none());
    }

    #[test]
    fn delta_q_contract_rejects_illegal_masked_actions_and_non_finite_targets() {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[2] = 1.0;

        let mut illegal_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut illegal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        illegal_target[3] = 0.1;
        illegal_mask[3] = 1.0;
        assert!(validate_delta_q_contract(&illegal_target, &illegal_mask, &legal_mask).is_none());

        let mut nan_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut valid_mask = [0.0f32; HYDRA_ACTION_SPACE];
        nan_target[2] = f32::NAN;
        valid_mask[2] = 1.0;
        assert!(validate_delta_q_contract(&nan_target, &valid_mask, &legal_mask).is_none());
    }

    #[test]
    fn delta_q_contract_accepts_regular_masked_discard_actions() {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[2] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[2] = -0.25;
        mask[2] = 1.0;

        let label = validate_delta_q_contract(&target, &mask, &legal_mask)
            .expect("regular masked discard action should be accepted");

        assert_eq!(label.target[2], -0.25);
        assert_eq!(label.mask[2], 1.0);
    }

    #[test]
    fn delta_q_sidecar_reader_reports_invalid_line_numbers() {
        let err = DeltaQSidecarIndex::from_jsonl_reader(Cursor::new("\nnot-json\n"))
            .expect_err("invalid jsonl should fail");
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("invalid replay delta_q sidecar line 2")
        );
    }

    #[test]
    fn delta_q_sidecar_reader_accepts_blank_lines_and_contract_rejects_bad_mask_values() {
        let records = synthetic_delta_q_records(123, 1);
        let raw = format!(
            "\n{}\n\n",
            serde_json::to_string(&records[0]).expect("record should serialize")
        );
        let index = DeltaQSidecarIndex::from_jsonl_reader(Cursor::new(raw))
            .expect("valid jsonl with blanks should parse");

        assert_eq!(index.len(), 1);

        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[records[0].action as usize] = 0.1;
        mask[records[0].action as usize] = 0.25;
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[records[0].action as usize] = 1.0;
        assert!(validate_delta_q_contract(&target, &mask, &legal_mask).is_none());
    }

    #[test]
    fn delta_q_sidecar_lookup_rejects_version_semantics_and_provenance_mismatches() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[2] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[2] = 0.25;
        mask[2] = 1.0;

        let mut record = ReplayDeltaQRecordV1 {
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

        record.version = 2;
        assert!(
            DeltaQSidecarIndex::from_records(vec![record.clone()])
                .lookup_label(&key, 2, &legal_mask, 9, 1)
                .is_none()
        );

        record.version = 1;
        record.semantics = "wrong-semantics".to_string();
        assert!(
            DeltaQSidecarIndex::from_records(vec![record.clone()])
                .lookup_label(&key, 2, &legal_mask, 9, 1)
                .is_none()
        );

        record.semantics = REPLAY_DELTA_Q_SEMANTICS_V1.to_string();
        record.provenance = "manual".to_string();
        assert!(
            DeltaQSidecarIndex::from_records(vec![record])
                .lookup_label(&key, 2, &legal_mask, 9, 1)
                .is_none()
        );
    }

    #[test]
    fn delta_q_sidecar_lookup_rejects_digest_mismatch_and_missing_masked_action() {
        let key = ReplayDecisionKey {
            source_hash: 7,
            event_index: 3,
            actor: 1,
            obs_hash: 11,
        };
        let mut stored_legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        stored_legal_mask[2] = 1.0;
        let mut lookup_legal_mask = stored_legal_mask;
        lookup_legal_mask[3] = 1.0;
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[2] = 0.25;
        mask[2] = 1.0;
        let record = ReplayDeltaQRecordV1 {
            version: 1,
            semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
            provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
            key,
            action: 2,
            legal_mask_digest: legal_mask_digest_from_f32(&stored_legal_mask),
            source_net_hash: 9,
            source_version: 1,
            target: target.to_vec(),
            mask: mask.to_vec(),
        };

        assert!(
            DeltaQSidecarIndex::from_records(vec![record.clone()])
                .lookup_label(&key, 2, &lookup_legal_mask, 9, 1)
                .is_none()
        );

        let no_mask_record = ReplayDeltaQRecordV1 {
            mask: vec![0.0; HYDRA_ACTION_SPACE],
            ..record
        };
        assert!(
            DeltaQSidecarIndex::from_records(vec![no_mask_record])
                .lookup_label(&key, 2, &stored_legal_mask, 9, 1)
                .is_none()
        );
    }

    #[test]
    fn delta_q_sidecar_can_load_from_jsonl_path() {
        let records = synthetic_delta_q_records(123, 1);
        let path = unique_temp_jsonl_path("replay-delta-q-sidecar");
        std::fs::write(
            &path,
            format!(
                "{}\n",
                serde_json::to_string(&records[0]).expect("record should serialize")
            ),
        )
        .expect("jsonl fixture should write");

        let index = DeltaQSidecarIndex::from_jsonl_path(&path).expect("jsonl path should parse");
        std::fs::remove_file(&path).expect("temp jsonl should be removable");

        assert_eq!(index.len(), 1);
    }
}
