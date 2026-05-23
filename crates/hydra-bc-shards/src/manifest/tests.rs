use super::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};

fn base_manifest() -> BcShardManifest {
    BcShardManifest {
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
        split_mode: "both".to_string(),
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
        storage_layout: STORAGE_LAYOUT_COMPACT.to_string(),
    }
}

fn split_manifest(split: BcShardSplit, sample_count: u64) -> BcShardSplitManifest {
    BcShardSplitManifest {
        split,
        shard_count: 1,
        sample_count,
        feature_flags: 0,
        record_size: BC_BASE_RECORD_SIZE,
        shards: vec![BcShardDescriptor {
            split,
            shard_index: 0,
            file_name: match split {
                BcShardSplit::Train => "train.hydra-bc".to_string(),
                BcShardSplit::Validation => "validation.hydra-bc".to_string(),
            },
            sample_count,
            first_sample_index: 0,
            byte_len: BC_SHARD_HEADER_SIZE as u64 + sample_count * u64::from(BC_BASE_RECORD_SIZE),
            feature_flags: 0,
            record_size: BC_BASE_RECORD_SIZE,
        }],
    }
}

#[test]
fn bc_shard_manifest_geometry_uses_frozen_runtime_abi() {
    let manifest = base_manifest();

    assert_eq!(OBS_SIZE, 6528);
    assert_eq!(NUM_CHANNELS, 192);
    assert_eq!(HYDRA_ACTION_SPACE, 46);
    assert_eq!(manifest.obs_size, OBS_SIZE);
    assert_eq!(manifest.num_channels, NUM_CHANNELS);
    assert_eq!(manifest.action_space, HYDRA_ACTION_SPACE);
    validate_bc_shard_manifest_contract(&manifest).expect("manifest geometry should be valid");
    assert_eq!(COMPACT_OBS_BASELINE_FACT_BYTES, 1_675);
    assert_eq!(OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_COUNT, 0);
    assert_eq!(OBS_ADVANCED_DENSE_CHANNEL_COUNT, 0);
    assert_eq!(COMPACT_OBS_SCALAR_REPEATED_BYTES, 0);
    assert_eq!(COMPACT_OBS_DENSE_BYTES, 0);
    assert_eq!(COMPACT_OBS_BYTES, 1_675);
}

#[test]
fn bc_shard_manifest_contract_snapshot_pins_abi_values() {
    assert_eq!(BC_SHARD_MAGIC, *b"HYBCS3\0\0");
    assert_eq!(BC_DENSE_SHARD_MAGIC, *b"HYBCS2\0\0");
    assert_eq!(
        DENSE_REBUILD_MESSAGE,
        "dense BC shards are obsolete; rebuild from replay"
    );
    assert_eq!(BC_SHARD_VERSION, 3);
    assert_eq!(BC_SHARD_MANIFEST_VERSION, 3);
    assert_eq!(BC_SHARD_LAYOUT_VERSION, 1);
    assert_eq!(BC_SHARD_HEADER_SIZE, 80);
    assert_eq!(STORAGE_LAYOUT_COMPACT, "compact");
    assert_eq!(TILE_COUNT, 34);
    assert_eq!(OPPONENT_COUNT, 3);
    assert_eq!(PLAYER_COUNT, 4);
    assert_eq!(SPATIAL_TARGET_SIZE, 102);
    assert_eq!(PACKED_ACTION_MASK_BYTES, 6);
    assert_eq!(FLAG_SAFETY_RESIDUAL, 1);
    assert_eq!(FLAG_EXIT, 2);
    assert_eq!(FLAG_DELTA_Q, 4);
    assert_eq!(FLAG_BELIEF_FIELDS, 8);
    assert_eq!(FLAG_MIXTURE_WEIGHTS, 16);
    assert_eq!(BC_BASE_RECORD_SIZE, 1_734);
    assert_eq!(BC_RECORD_SIZE_WITH_ALL_OPTIONALS, 4_496);
    assert_eq!(BcShardSplit::Train.split_id(), 0);
    assert_eq!(BcShardSplit::Validation.split_id(), 1);
}

#[test]
fn checked_record_bytes_rejects_overflow() {
    let err = checked_record_bytes(u64::MAX, BC_BASE_RECORD_SIZE)
        .expect_err("record bytes must overflow");
    assert!(
        err.contains("record byte count overflow"),
        "unexpected error: {err}"
    );
}

#[test]
fn bc_shard_manifest_rejects_num_channel_mismatch() {
    let mut manifest = base_manifest();
    manifest.num_channels = NUM_CHANNELS - 1;

    let err = validate_bc_shard_manifest_contract(&manifest).expect_err("num_channels mismatch");
    assert!(err.contains("num_channels"), "unexpected error: {err}");
}

#[test]
fn bc_shard_manifest_rejects_invalid_split_mode() {
    let mut manifest = base_manifest();
    manifest.split_mode = "holdout".to_string();

    let err = validate_bc_shard_manifest_contract(&manifest).expect_err("invalid split mode");
    assert!(err.contains("split_mode"), "unexpected error: {err}");
}

#[test]
fn bc_shard_manifest_rejects_duplicate_split_entries() {
    let mut manifest = base_manifest();
    manifest.split_mode = "train".to_string();
    manifest.splits = vec![
        split_manifest(BcShardSplit::Train, 2),
        split_manifest(BcShardSplit::Train, 3),
    ];
    manifest.totals.sample_count = 5;
    manifest.totals.shard_count = 2;

    let err = validate_bc_shard_manifest_contract(&manifest).expect_err("duplicate split");
    assert!(err.contains("duplicate train"), "unexpected error: {err}");
}

#[test]
fn bc_shard_manifest_requires_splits_for_non_empty_split_mode() {
    let mut manifest = base_manifest();
    manifest.splits = vec![split_manifest(BcShardSplit::Train, 2)];
    manifest.totals.sample_count = 2;
    manifest.totals.shard_count = 1;

    let err = validate_bc_shard_manifest_contract(&manifest).expect_err("missing validation split");
    assert!(
        err.contains("requires train and validation"),
        "unexpected error: {err}"
    );

    manifest.split_mode = "train".to_string();
    validate_bc_shard_manifest_contract(&manifest).expect("train-only manifest should be valid");

    manifest.splits = vec![split_manifest(BcShardSplit::Validation, 2)];
    manifest.split_mode = "validation".to_string();
    validate_bc_shard_manifest_contract(&manifest)
        .expect("validation-only manifest should be valid");
}

#[test]
fn bc_shard_manifest_rejects_splits_excluded_by_split_mode() {
    let mut manifest = base_manifest();
    manifest.split_mode = "train".to_string();
    manifest.splits = vec![
        split_manifest(BcShardSplit::Train, 2),
        split_manifest(BcShardSplit::Validation, 1),
    ];
    manifest.totals.sample_count = 3;
    manifest.totals.shard_count = 2;

    let err = validate_bc_shard_manifest_contract(&manifest)
        .expect_err("train mode should reject validation split");
    assert!(err.contains("excludes"), "unexpected error: {err}");

    manifest.split_mode = "validation".to_string();
    let err = validate_bc_shard_manifest_contract(&manifest)
        .expect_err("validation mode should reject train split");
    assert!(err.contains("excludes"), "unexpected error: {err}");
}

#[test]
fn bc_shard_manifest_deserializes_missing_split_mode_as_both() {
    let mut manifest = base_manifest();
    manifest.splits = vec![
        split_manifest(BcShardSplit::Train, 2),
        split_manifest(BcShardSplit::Validation, 1),
    ];
    manifest.totals.sample_count = 3;
    manifest.totals.shard_count = 2;
    let mut value = serde_json::to_value(&manifest).expect("manifest should serialize");
    value
        .as_object_mut()
        .expect("manifest JSON should be an object")
        .remove("split_mode");

    let decoded: BcShardManifest =
        serde_json::from_value(value).expect("missing split_mode should deserialize");
    assert_eq!(decoded.split_mode, "both");
    validate_bc_shard_manifest_contract(&decoded).expect("default both manifest should validate");
}

#[test]
fn bc_shard_manifest_rejects_split_sample_total_overflow() {
    let mut manifest = base_manifest();
    let mut train = split_manifest(BcShardSplit::Train, 0);
    train.sample_count = u64::MAX;
    let validation = split_manifest(BcShardSplit::Validation, 1);
    manifest.splits = vec![train, validation];

    let err = validate_bc_shard_manifest_contract(&manifest)
        .expect_err("split sample total should overflow");
    assert!(
        err.contains("sample_count total overflow"),
        "unexpected error: {err}"
    );
}

#[test]
fn bc_shard_manifest_rejects_split_shard_total_overflow() {
    let mut manifest = base_manifest();
    let mut train = split_manifest(BcShardSplit::Train, 0);
    train.shard_count = usize::MAX;
    let validation = split_manifest(BcShardSplit::Validation, 0);
    manifest.splits = vec![train, validation];

    let err = validate_bc_shard_manifest_contract(&manifest)
        .expect_err("split shard total should overflow");
    assert!(
        err.contains("shard_count total overflow"),
        "unexpected error: {err}"
    );
}
