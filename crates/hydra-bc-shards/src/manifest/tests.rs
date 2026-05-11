use super::*;

#[test]
fn bc_shard_manifest_geometry_uses_frozen_runtime_abi() {
    let manifest = BcShardManifest {
        manifest_version: BC_SHARD_MANIFEST_VERSION,
        shard_version: BC_SHARD_VERSION,
        shard_header_size: BC_SHARD_HEADER_SIZE,
        base_record_size: BC_BASE_RECORD_SIZE,
        max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        obs_size: OBS_SIZE,
        num_channels: NUM_CHANNELS,
        action_space: HYDRA_ACTION_SPACE,
        train_fraction: 1.0,
        shard_samples: 1,
        augment_runtime: true,
        input: String::new(),
        output_dir: String::new(),
        created_at: String::new(),
        source_count: 0,
        source_total_games_hint: 0,
        source_train_count_hint: 0,
        source_val_count_hint: 0,
        source_counts_exact: true,
        exit_sidecar: None,
        delta_q_sidecar: None,
        totals: BcShardBuildTotals::default(),
        splits: Vec::new(),
    };

    assert_eq!(OBS_SIZE, 6528);
    assert_eq!(NUM_CHANNELS, 192);
    assert_eq!(HYDRA_ACTION_SPACE, 46);
    assert_eq!(manifest.obs_size, OBS_SIZE);
    assert_eq!(manifest.num_channels, NUM_CHANNELS);
    assert_eq!(manifest.action_space, HYDRA_ACTION_SPACE);
    validate_bc_shard_manifest_contract(&manifest).expect("manifest geometry should be valid");
}
