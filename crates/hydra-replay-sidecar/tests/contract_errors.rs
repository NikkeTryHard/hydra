use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_replay_sidecar::{
    DeltaQSidecarIndex, ExitSidecarIndex, REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1,
    REPLAY_EXIT_PROVENANCE, REPLAY_EXIT_SEMANTICS_V1, ReplayDecisionKey, ReplayDeltaQRecordV1,
    ReplayExitRecordV1, SidecarContractError, SidecarKind, legal_mask_digest_from_f32,
};
use std::io::{Cursor, ErrorKind};

fn key() -> ReplayDecisionKey {
    ReplayDecisionKey {
        source_hash: 7,
        event_index: 3,
        actor: 1,
        obs_hash: 11,
    }
}

fn one_hot(index: usize, value: f32) -> [f32; HYDRA_ACTION_SPACE] {
    let mut output = [0.0f32; HYDRA_ACTION_SPACE];
    output[index] = value;
    output
}

fn exit_record() -> ReplayExitRecordV1 {
    let legal_mask = one_hot(2, 1.0);
    ReplayExitRecordV1 {
        version: 1,
        semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
        provenance: REPLAY_EXIT_PROVENANCE.to_string(),
        key: key(),
        action: 2,
        legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
        source_net_hash: 9,
        source_version: 1,
        root_visit_count: 64,
        legal_discard_count: 1,
        supported_actions: 1,
        coverage: 1.0,
        kl_to_base: 0.0,
        target: one_hot(2, 1.0).to_vec(),
        mask: legal_mask.to_vec(),
    }
}

fn delta_q_record() -> ReplayDeltaQRecordV1 {
    let legal_mask = one_hot(2, 1.0);
    ReplayDeltaQRecordV1 {
        version: 1,
        semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
        provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
        key: key(),
        action: 2,
        legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
        source_net_hash: 9,
        source_version: 1,
        target: one_hot(2, 0.25).to_vec(),
        mask: legal_mask.to_vec(),
    }
}

#[test]
fn exit_lookup_missing_key_is_absent_but_present_mismatch_hard_errors() {
    let legal_mask = one_hot(2, 1.0);
    let index = ExitSidecarIndex::from_records(vec![exit_record()]);

    assert!(
        index
            .lookup_label(&key(), 3, &legal_mask, 9, 1)
            .expect("missing action should not hard-error")
            .is_none()
    );
    assert!(matches!(
        index.lookup_label(&key(), 2, &legal_mask, 10, 1),
        Err(SidecarContractError::SourceNetHash {
            sidecar: SidecarKind::Exit,
            expected: 10,
            actual: 9,
        })
    ));
}

#[test]
fn delta_q_lookup_missing_key_is_absent_but_present_mismatch_hard_errors() {
    let legal_mask = one_hot(2, 1.0);
    let index = DeltaQSidecarIndex::from_records(vec![delta_q_record()]);

    assert!(
        index
            .lookup_label(&key(), 3, &legal_mask, 9, 1)
            .expect("missing action should not hard-error")
            .is_none()
    );
    assert!(matches!(
        index.lookup_label(&key(), 2, &legal_mask, 9, 2),
        Err(SidecarContractError::SourceVersion {
            sidecar: SidecarKind::DeltaQ,
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn exit_lookup_hard_errors_on_shape_version_and_provenance() {
    let legal_mask = one_hot(2, 1.0);
    let mut record = exit_record();
    record.target.pop();
    let index = ExitSidecarIndex::from_records(vec![record]);
    assert!(matches!(
        index.lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Shape {
            sidecar: SidecarKind::Exit,
            field: "target",
            expected: HYDRA_ACTION_SPACE,
            actual,
        }) if actual == HYDRA_ACTION_SPACE - 1
    ));

    let mut record = exit_record();
    record.version = 2;
    assert!(matches!(
        ExitSidecarIndex::from_records(vec![record]).lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Version {
            sidecar: SidecarKind::Exit,
            expected: 1,
            actual: 2,
        })
    ));

    let mut record = exit_record();
    record.provenance = "manual".to_string();
    assert!(matches!(
        ExitSidecarIndex::from_records(vec![record]).lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Provenance {
            sidecar: SidecarKind::Exit,
            expected: REPLAY_EXIT_PROVENANCE,
        })
    ));
}

#[test]
fn delta_q_lookup_hard_errors_on_shape_version_and_provenance() {
    let legal_mask = one_hot(2, 1.0);
    let mut record = delta_q_record();
    record.mask.pop();
    let index = DeltaQSidecarIndex::from_records(vec![record]);
    assert!(matches!(
        index.lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Shape {
            sidecar: SidecarKind::DeltaQ,
            field: "mask",
            expected: HYDRA_ACTION_SPACE,
            actual,
        }) if actual == HYDRA_ACTION_SPACE - 1
    ));

    let mut record = delta_q_record();
    record.version = 2;
    assert!(matches!(
        DeltaQSidecarIndex::from_records(vec![record]).lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Version {
            sidecar: SidecarKind::DeltaQ,
            expected: 1,
            actual: 2,
        })
    ));

    let mut record = delta_q_record();
    record.provenance = "manual".to_string();
    assert!(matches!(
        DeltaQSidecarIndex::from_records(vec![record]).lookup_label(&key(), 2, &legal_mask, 9, 1),
        Err(SidecarContractError::Provenance {
            sidecar: SidecarKind::DeltaQ,
            expected: REPLAY_DELTA_Q_PROVENANCE,
        })
    ));
}

#[test]
fn sidecar_jsonl_load_reports_invalid_data() {
    let exit_err = ExitSidecarIndex::from_jsonl_reader(Cursor::new("not-json\n"))
        .expect_err("invalid ExIt JSONL should fail");
    assert_eq!(exit_err.kind(), ErrorKind::InvalidData);
    assert!(
        exit_err
            .to_string()
            .contains("invalid replay ExIt sidecar line 1")
    );

    let delta_q_err = DeltaQSidecarIndex::from_jsonl_reader(Cursor::new("not-json\n"))
        .expect_err("invalid delta-q JSONL should fail");
    assert_eq!(delta_q_err.kind(), ErrorKind::InvalidData);
    assert!(
        delta_q_err
            .to_string()
            .contains("invalid replay delta_q sidecar line 1")
    );
}
