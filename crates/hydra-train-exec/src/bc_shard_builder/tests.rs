use super::*;
use hydra_core::action::HYDRA_ACTION_SPACE;

#[test]
fn policy_target_vec_from_actions_ignores_negative_and_out_of_range() {
    let targets = policy_target_vec_from_actions(&[-1, 0, HYDRA_ACTION_SPACE as i64, 45], 4);
    assert_eq!(targets.len(), 4 * HYDRA_ACTION_SPACE);
    assert!(
        targets[0..HYDRA_ACTION_SPACE]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_eq!(targets[HYDRA_ACTION_SPACE], 1.0);
    assert!(
        targets[2 * HYDRA_ACTION_SPACE..3 * HYDRA_ACTION_SPACE]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_eq!(targets[3 * HYDRA_ACTION_SPACE + 45], 1.0);
}

#[test]
fn build_bc_shards_rejects_zero_shard_samples() {
    let config = BuildBcShardsConfig {
        shard_samples: 0,
        ..BuildBcShardsConfig::default()
    };
    let err = build_bc_shards(&config).expect_err("zero shard_samples should fail");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert_eq!(err.to_string(), "shard_samples must be > 0");
}
