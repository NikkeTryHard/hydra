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
        if let Some(decision) = prepare_replay_decision(event, &mut state, &safety, &mut encoder)
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
