use super::*;
use crate::{BC_SHARD_HEADER_SIZE, FLAG_EXIT, FLAG_SAFETY_RESIDUAL, record_size_for_flags};

fn dummy_sample() -> MjaiSample {
    let mut legal_mask = [0.0; HYDRA_ACTION_SPACE];
    legal_mask[3] = 1.0;
    let mut safety = [0.0; HYDRA_ACTION_SPACE];
    safety[1] = 0.5;
    let mut safety_mask = [0.0; HYDRA_ACTION_SPACE];
    safety_mask[1] = 1.0;
    let mut exit_target = [0.0; HYDRA_ACTION_SPACE];
    exit_target[3] = 0.75;
    let mut exit_mask = [0.0; HYDRA_ACTION_SPACE];
    exit_mask[3] = 1.0;
    MjaiSample {
        obs: [0.25; OBS_SIZE],
        action: 3,
        legal_mask,
        placement: 1,
        score_delta: 1200,
        grp_label: 7,
        oracle_target: Some([0.1, 0.2, 0.3, 0.4]),
        tenpai: [1.0, 0.0, 1.0],
        opp_next: [3, 8, 255],
        danger: [0.0; SPATIAL_TARGET_SIZE],
        danger_mask: [1.0; SPATIAL_TARGET_SIZE],
        safety_residual: Some(safety),
        safety_residual_mask: Some(safety_mask),
        exit_target: Some(exit_target),
        exit_mask: Some(exit_mask),
        delta_q_target: None,
        delta_q_mask: None,
        belief_fields: Some([0.0; 16 * 34]),
        mixture_weights: Some([0.25; 4]),
        belief_fields_present: true,
        mixture_weights_present: true,
    }
}

#[test]
fn compact_header_size_constant_matches_written_bytes() {
    let mut bytes = Vec::new();
    write_shard_header(
        &mut bytes,
        BcShardSplit::Train,
        2,
        10,
        100,
        FLAG_SAFETY_RESIDUAL,
        record_size_for_flags(FLAG_SAFETY_RESIDUAL),
    )
    .expect("header write should succeed");
    assert_eq!(bytes.len(), BC_SHARD_HEADER_SIZE as usize);
}

#[test]
fn compact_record_size_constant_matches_written_bytes() {
    let sample = dummy_sample();
    let flags = FLAG_SAFETY_RESIDUAL | FLAG_EXIT;
    let mut bytes = Vec::new();
    write_sample_record(&mut bytes, &sample, flags).expect("sample write should succeed");
    assert_eq!(bytes.len(), record_size_for_flags(flags) as usize);
}
