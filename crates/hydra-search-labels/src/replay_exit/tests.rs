use super::*;
use hydra_core::action::DISCARD_END;
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

fn synthetic_exit_records(source_net_hash: u64, source_version: u32) -> Vec<ReplayExitRecordV1> {
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
            target[decision.action_id as usize] = 1.0;
            records.push(ReplayExitRecordV1 {
                version: 1,
                semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
                provenance: REPLAY_EXIT_PROVENANCE.to_string(),
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
                root_visit_count: 64,
                legal_discard_count: decision.legal_mask[..=DISCARD_END as usize]
                    .iter()
                    .filter(|&&value| value)
                    .count() as u8,
                supported_actions: 1,
                coverage: 1.0,
                kl_to_base: 0.0,
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
fn duplicate_exit_sidecar_key_last_record_wins() {
    let key = ReplayDecisionKey {
        source_hash: 7,
        event_index: 3,
        actor: 1,
        obs_hash: 11,
    };
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    mask[2] = 1.0;
    let mut first_target = [0.0f32; HYDRA_ACTION_SPACE];
    first_target[2] = 1.0;
    let mut second_target = [0.0f32; HYDRA_ACTION_SPACE];
    second_target[2] = 0.75;
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
            legal_discard_count: 1,
            supported_actions: 1,
            coverage: 1.0,
            kl_to_base: 0.0,
            target: first_target.to_vec(),
            mask: mask.to_vec(),
        },
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
            legal_discard_count: 1,
            supported_actions: 1,
            coverage: 1.0,
            kl_to_base: 0.0,
            target: second_target.to_vec(),
            mask: mask.to_vec(),
        },
    ];
    let index = ExitSidecarIndex::from_records(records);
    let (target, loaded_mask) = index.lookup_label(&key, 2, &mask, 9, 1).expect("lookup");
    assert!((target[2] - 0.75).abs() < 1e-6);
    assert_eq!(loaded_mask[2], 1.0);
}

#[test]
fn replay_exit_records_are_tagged_search_derived() {
    let records = synthetic_exit_records(123, 1);
    for record in records {
        assert_eq!(record.provenance, REPLAY_EXIT_PROVENANCE);
        assert_eq!(record.semantics, REPLAY_EXIT_SEMANTICS_V1);
        assert_eq!(record.version, 1);
        assert!(record.action <= DISCARD_END);
    }
}

#[test]
fn loader_with_sidecar_populates_exit_fields() {
    let events = read_mjai_events(Cursor::new(guardrail_log())).expect("parse events");
    let records = synthetic_exit_records(123, 1);
    let index = ExitSidecarIndex::from_records(records);

    let game = load_game_from_events_with_sidecar(
        "game-1",
        hydra_replay_loader::mjai_loader::SidecarProvenance::new(Some(123), Some(1)),
        hydra_replay_loader::mjai_loader::SidecarProvenance::default(),
        hydra_replay_loader::mjai_loader::ReplayTargetProfile::with_optional_heads(
            false, false, false, false, true, false,
        ),
        hydra_replay_loader::mjai_loader::ReplayObservationProfile::BcMinimal,
        events,
        Some(&index),
        None,
    )
    .expect("load with sidecar");
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.exit_target.is_some())
    );
    assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
}

#[test]
fn copy_label_arrays_rejects_wrong_lengths() {
    assert!(
        copy_label_arrays(&[0.0; HYDRA_ACTION_SPACE - 1], &[0.0; HYDRA_ACTION_SPACE]).is_none()
    );
    assert!(
        copy_label_arrays(&[0.0; HYDRA_ACTION_SPACE], &[0.0; HYDRA_ACTION_SPACE - 1]).is_none()
    );
}

#[test]
fn copy_label_arrays_accepts_exact_action_space_lengths() {
    let mut target = vec![0.0; HYDRA_ACTION_SPACE];
    let mut mask = vec![0.0; HYDRA_ACTION_SPACE];
    target[3] = 0.75;
    mask[3] = 1.0;

    let (target_arr, mask_arr) =
        copy_label_arrays(&target, &mask).expect("exact-size vectors should copy");

    assert_eq!(target_arr[3], 0.75);
    assert_eq!(mask_arr[3], 1.0);
    assert_eq!(target_arr.iter().filter(|&&value| value > 0.0).count(), 1);
}

#[test]
fn read_jsonl_records_reports_invalid_line_numbers() {
    let err = read_jsonl_records::<ReplayExitRecordV1>(
        Cursor::new("\nnot-json\n"),
        "replay ExIt sidecar",
    )
    .expect_err("invalid jsonl should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(
        err.to_string()
            .contains("invalid replay ExIt sidecar line 2")
    );
}

#[test]
fn exit_sidecar_index_rejects_malformed_target_shapes() {
    let key = ReplayDecisionKey {
        source_hash: 7,
        event_index: 3,
        actor: 1,
        obs_hash: 11,
    };
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    mask[2] = 1.0;
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
        target: vec![1.0; HYDRA_ACTION_SPACE - 1],
        mask: vec![1.0; HYDRA_ACTION_SPACE],
    };
    let index = ExitSidecarIndex::from_records(vec![record]);
    assert!(index.lookup_label(&key, 2, &mask, 9, 1).is_none());
}

#[test]
fn exit_sidecar_index_rejects_malformed_mask_shapes() {
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
        mask: vec![1.0; HYDRA_ACTION_SPACE - 1],
    };
    let index = ExitSidecarIndex::from_records(vec![record]);
    assert!(index.lookup_label(&key, 2, &mask, 9, 1).is_none());
}

#[test]
fn exit_sidecar_reader_skips_blank_lines_and_hash_helpers_match() {
    let records = synthetic_exit_records(123, 1);
    let raw = format!(
        "\n{}\n\n",
        serde_json::to_string(&records[0]).expect("record should serialize")
    );
    let index = ExitSidecarIndex::from_jsonl_reader(Cursor::new(raw))
        .expect("valid jsonl with blanks should parse");

    assert_eq!(index.len(), 1);
    assert_eq!(
        source_net_hash_from_checkpoint_identity("checkpoint-a"),
        source_hash_from_identity("checkpoint-a")
    );
}

#[test]
fn exit_sidecar_index_rejects_version_semantics_and_provenance_mismatches() {
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

    let mut record = ReplayExitRecordV1 {
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

    record.version = 2;
    assert!(
        ExitSidecarIndex::from_records(vec![record.clone()])
            .lookup_label(&key, 2, &mask, 9, 1)
            .is_none()
    );

    record.version = 1;
    record.semantics = "wrong-semantics".to_string();
    assert!(
        ExitSidecarIndex::from_records(vec![record.clone()])
            .lookup_label(&key, 2, &mask, 9, 1)
            .is_none()
    );

    record.semantics = REPLAY_EXIT_SEMANTICS_V1.to_string();
    record.provenance = "manual".to_string();
    assert!(
        ExitSidecarIndex::from_records(vec![record])
            .lookup_label(&key, 2, &mask, 9, 1)
            .is_none()
    );
}

#[test]
fn exit_sidecar_index_rejects_legal_mask_digest_mismatch() {
    let key = ReplayDecisionKey {
        source_hash: 7,
        event_index: 3,
        actor: 1,
        obs_hash: 11,
    };
    let mut stored_mask = [0.0f32; HYDRA_ACTION_SPACE];
    stored_mask[2] = 1.0;
    let mut lookup_mask = stored_mask;
    lookup_mask[3] = 1.0;
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    target[2] = 1.0;
    let record = ReplayExitRecordV1 {
        version: 1,
        semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
        provenance: REPLAY_EXIT_PROVENANCE.to_string(),
        key,
        action: 2,
        legal_mask_digest: legal_mask_digest_from_f32(&stored_mask),
        source_net_hash: 9,
        source_version: 1,
        root_visit_count: 64,
        legal_discard_count: 1,
        supported_actions: 1,
        coverage: 1.0,
        kl_to_base: 0.0,
        target: target.to_vec(),
        mask: stored_mask.to_vec(),
    };

    assert!(
        ExitSidecarIndex::from_records(vec![record])
            .lookup_label(&key, 2, &lookup_mask, 9, 1)
            .is_none()
    );
}

#[test]
fn exit_sidecar_index_can_load_from_jsonl_path() {
    let records = synthetic_exit_records(123, 1);
    let path = unique_temp_jsonl_path("replay-exit-sidecar");
    std::fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string(&records[0]).expect("record should serialize")
        ),
    )
    .expect("jsonl fixture should write");

    let index = ExitSidecarIndex::from_jsonl_path(&path).expect("jsonl path should parse");
    std::fs::remove_file(&path).expect("temp jsonl should be removable");

    assert_eq!(index.len(), 1);
}
