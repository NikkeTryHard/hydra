use std::fs;
use std::path::{Path, PathBuf};

use hydra_bc_shards::{
    ActiveShardWriter, BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE, BC_SHARD_MANIFEST_VERSION, BC_SHARD_VERSION, BcShardBuildTotals,
    BcShardDescriptor, BcShardManifest, BcShardSidecarManifest, BcShardSplit, BcShardSplitManifest,
};
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_data_core::sample::{
    COMPACT_MISSING_SHANTEN, COMPACT_MISSING_TILE, CompactObservationFacts, CompactPlayerDiscards,
    CompactPlayerMelds, CompactSafetyFacts, MjaiSample,
};

use crate::epoch_runner::BcShardPrefetcher;
use crate::test_support::unique_test_path;

fn dummy_sample(action: usize) -> MjaiSample {
    let mut legal_mask = [0.0; HYDRA_ACTION_SPACE];
    legal_mask[action] = 1.0;
    let obs = [0.0; OBS_SIZE];
    MjaiSample {
        obs,
        compact_facts: Some(CompactObservationFacts {
            hand_counts: [0; 34],
            open_meld_counts: [0; 34],
            drawn_tile: COMPACT_MISSING_TILE,
            shanten_base: 0,
            shanten_discard: [COMPACT_MISSING_SHANTEN; 34],
            discards: [CompactPlayerDiscards::default(); 4],
            melds: [CompactPlayerMelds::default(); 4],
            dora_indicators: [0; 5],
            dora_indicator_count: 0,
            aka_flags: [false; 3],
            riichi: [false; 4],
            scores: [0; 4],
            kyoku_index: 0,
            honba: 0,
            kyotaku: 0,
            safety: CompactSafetyFacts {
                genbutsu_all: [0; 3],
                genbutsu_tedashi: [0; 3],
                genbutsu_riichi_era: [0; 3],
                suji: [[0.0; 34]; 3],
                half_suji: [0; 3],
                matagi: [[0.0; 34]; 3],
                kabe: 0,
                one_chance: 0,
                visible_counts: [0; 34],
                opponent_riichi: [false; 3],
                cached_tenpai_prob: [0.0; 3],
            },
            advanced_tail: None,
        }),
        action: action as u8,
        legal_mask,
        placement: 1,
        score_delta: 0,
        grp_label: 0,
        oracle_target: Some([0.0; 4]),
        tenpai: [0.0; 3],
        opp_next: [255; 3],
        danger: [0.0; hydra_bc_shards::SPATIAL_TARGET_SIZE],
        danger_mask: [0.0; hydra_bc_shards::SPATIAL_TARGET_SIZE],
        safety_residual: None,
        safety_residual_mask: None,
        exit_target: None,
        exit_mask: None,
        delta_q_target: None,
        delta_q_mask: None,
        belief_fields: None,
        mixture_weights: None,
        belief_fields_present: false,
        mixture_weights_present: false,
    }
}

fn manifest_for_descriptor(descriptor: BcShardDescriptor) -> BcShardManifest {
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
        shard_samples: descriptor.sample_count as usize,
        split_mode: "train".to_string(),
        augment_runtime: false,
        input: "test".to_string(),
        output_dir: "test".to_string(),
        created_at: "test".to_string(),
        source_count: 1,
        source_total_games_hint: 1,
        source_train_count_hint: 1,
        source_val_count_hint: 0,
        source_counts_exact: true,
        exit_sidecar: None::<BcShardSidecarManifest>,
        delta_q_sidecar: None::<BcShardSidecarManifest>,
        totals: BcShardBuildTotals {
            sample_count: descriptor.sample_count,
            skipped_games: 0,
            empty_games: 0,
            shard_count: 1,
        },
        splits: vec![BcShardSplitManifest {
            split: descriptor.split,
            shard_count: 1,
            sample_count: descriptor.sample_count,
            feature_flags: descriptor.feature_flags,
            record_size: descriptor.record_size,
            shards: vec![descriptor],
        }],
        storage_layout: hydra_bc_shards::STORAGE_LAYOUT_COMPACT.to_string(),
    }
}

fn test_manifest(label: &str, samples: usize) -> (PathBuf, PathBuf) {
    let dir = unique_test_path("hydra-bc-prefetch", label);
    fs::create_dir_all(&dir).expect("test dir should be creatable");
    let mut writer =
        ActiveShardWriter::new(&dir, BcShardSplit::Train, 0, 0, 0).expect("writer should open");
    let samples = (0..samples)
        .map(|idx| dummy_sample(idx % HYDRA_ACTION_SPACE))
        .collect::<Vec<_>>();
    writer
        .write_samples(&samples)
        .expect("test samples should write");
    let descriptor = writer.finish().expect("writer should finish");
    let manifest = manifest_for_descriptor(descriptor);
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    (dir, manifest_path)
}

fn drain_order(manifest_path: &Path, depth: usize) -> Vec<(usize, usize)> {
    let prefetcher = BcShardPrefetcher::spawn(manifest_path, 3, false, 0, 10, depth)
        .expect("prefetcher should spawn");
    let mut ranges = Vec::new();
    while let Some(batch) = prefetcher.recv().expect("recv should succeed") {
        ranges.push((batch.start_index, batch.sample_count));
        prefetcher.recycle(batch.slot_seq, batch.host_batch);
    }
    prefetcher.join().expect("producer should join");
    ranges
}

#[test]
fn bc_shard_prefetcher_depth_one_preserves_order() {
    let (_dir, manifest_path) = test_manifest("depth-one-order", 10);
    assert_eq!(
        drain_order(&manifest_path, 1),
        vec![(0, 3), (3, 3), (6, 3), (9, 1)]
    );
}

#[test]
fn bc_shard_prefetcher_depth_four_preserves_order() {
    let (_dir, manifest_path) = test_manifest("depth-four-order", 10);
    assert_eq!(
        drain_order(&manifest_path, 4),
        vec![(0, 3), (3, 3), (6, 3), (9, 1)]
    );
}

#[test]
fn bc_shard_prefetcher_empty_recycle_drains_past_depth() {
    let (_dir, manifest_path) = test_manifest("empty-recycle-depth-two", 18);
    let prefetcher = BcShardPrefetcher::spawn(&manifest_path, 3, false, 0, 18, 2)
        .expect("prefetcher should spawn");
    let mut ranges = Vec::new();
    while let Some(batch) = prefetcher.recv().expect("recv should succeed") {
        ranges.push((batch.start_index, batch.sample_count));
        prefetcher.recycle(batch.slot_seq, hydra_bc_shards::BcShardHostBatch::empty());
    }
    prefetcher.join().expect("producer should join");
    assert_eq!(
        ranges,
        vec![(0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 3)]
    );
}

#[test]
fn bc_shard_prefetcher_early_drop_joins_producer() {
    let (_dir, manifest_path) = test_manifest("early-drop", 12);
    let prefetcher = BcShardPrefetcher::spawn(&manifest_path, 2, false, 0, 12, 2)
        .expect("prefetcher should spawn");
    let first = prefetcher.recv().expect("recv should succeed");
    assert!(first.is_some());
    drop(prefetcher);
}

#[test]
fn bc_shard_prefetcher_reuses_slot_capacity_after_warmup() {
    let (_dir, manifest_path) = test_manifest("reuse-capacity", 12);
    let prefetcher = BcShardPrefetcher::spawn(&manifest_path, 3, false, 0, 12, 1)
        .expect("prefetcher should spawn");
    let first = prefetcher
        .recv()
        .expect("recv should succeed")
        .expect("first batch should exist");
    let first_capacity = first.host_batch.obs_flat.capacity();
    prefetcher.recycle(first.slot_seq, first.host_batch);
    let second = prefetcher
        .recv()
        .expect("recv should succeed")
        .expect("second batch should exist");
    assert_eq!(second.host_batch.obs_flat.capacity(), first_capacity);
    prefetcher.recycle(second.slot_seq, second.host_batch);
    prefetcher.join().expect("producer should join");
}

#[test]
fn bc_shard_prefetcher_reports_wait_and_occupancy_metrics() {
    let (_dir, manifest_path) = test_manifest("metrics", 8);
    let prefetcher = BcShardPrefetcher::spawn(&manifest_path, 2, false, 0, 8, 2)
        .expect("prefetcher should spawn");
    let batch = prefetcher
        .recv()
        .expect("recv should succeed")
        .expect("batch should exist");
    prefetcher.recycle(batch.slot_seq, batch.host_batch);
    let metrics = prefetcher.metrics();
    assert!(metrics.producer_wait_seconds >= 0.0);
    assert!(metrics.consumer_wait_seconds >= 0.0);
    assert!(metrics.ring_occupancy_avg >= 0.0);
    assert!(metrics.ring_occupancy_min <= 2);
    prefetcher.join().expect("producer should join");
}
