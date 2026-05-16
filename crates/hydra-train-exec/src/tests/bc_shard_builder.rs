use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bc_shard_builder::{
    BuildBcShardsConfig, build_bc_shards, policy_target_vec_from_actions,
    set_test_stop_after_built_chunks,
};
use hydra_bc_shards::{BcShardSplit, BcShardSplitMode, load_bc_shard_reader};
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_data_core::{DataManifest, DataSource};

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tiny_real_mjai_replay() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"test-game"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

fn test_dir(name: &str) -> PathBuf {
    let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "hydra-bc-shard-builder-{name}-{}-{id}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("test dir should be creatable");
    path
}

fn write_replay(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, tiny_real_mjai_replay()).expect("fixture replay should be writable");
    path
}

fn manifest_for(paths: &[PathBuf]) -> DataManifest {
    DataManifest {
        sources: paths.iter().cloned().map(DataSource::LooseFile).collect(),
        total_games: paths.len(),
        train_count: paths.len(),
        val_count: 0,
        counts_exact: true,
    }
}

fn base_config(input: &Path, output_dir: &Path, paths: &[PathBuf]) -> BuildBcShardsConfig {
    BuildBcShardsConfig {
        input: input.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        manifest_name: "manifest.json".to_string(),
        train_fraction: 1.0,
        shard_samples: 2,
        split_mode: BcShardSplitMode::Train,
        source_manifest: Some(manifest_for(paths)),
        report_name: Some("report.json".to_string()),
        ..BuildBcShardsConfig::default()
    }
}

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

#[test]
fn parallel_loose_build_matches_serial_manifest_counts() {
    let input_dir = test_dir("parallel-input");
    let paths = vec![
        write_replay(&input_dir, "a.mjai"),
        write_replay(&input_dir, "b.mjai"),
        write_replay(&input_dir, "c.mjai"),
    ];
    let serial_dir = test_dir("serial-output");
    let parallel_dir = test_dir("parallel-output");

    let serial = build_bc_shards(&BuildBcShardsConfig {
        num_threads: Some(1),
        ..base_config(&input_dir, &serial_dir, &paths)
    })
    .expect("serial build should pass");
    let parallel = build_bc_shards(&BuildBcShardsConfig {
        num_threads: Some(4),
        queue_bound: 1,
        ..base_config(&input_dir, &parallel_dir, &paths)
    })
    .expect("parallel build should pass");

    assert_eq!(
        parallel.manifest.totals.sample_count,
        serial.manifest.totals.sample_count
    );
    assert_eq!(
        parallel.manifest.totals.skipped_games,
        serial.manifest.totals.skipped_games
    );
    assert_eq!(
        parallel.manifest.totals.empty_games,
        serial.manifest.totals.empty_games
    );
    assert_eq!(
        parallel.manifest.totals.shard_count,
        serial.manifest.totals.shard_count
    );
    assert_eq!(parallel.manifest.splits.len(), serial.manifest.splits.len());
    assert_eq!(
        parallel.manifest.splits[0].sample_count,
        serial.manifest.splits[0].sample_count
    );
    assert_eq!(
        parallel.manifest.splits[0].shard_count,
        serial.manifest.splits[0].shard_count
    );
    assert_eq!(
        parallel.manifest.splits[0].record_size,
        serial.manifest.splits[0].record_size
    );
    assert_eq!(
        parallel.manifest.splits[0].feature_flags,
        serial.manifest.splits[0].feature_flags
    );

    let reader = load_bc_shard_reader(&parallel.manifest_path, BcShardSplit::Train)
        .expect("parallel shards should load");
    assert_eq!(
        reader.sample_count() as u64,
        parallel.manifest.totals.sample_count
    );
    let batch = reader
        .collate_host_batch_range(0, reader.sample_count().min(2), false)
        .expect("parallel shards should decode compact fact rows");
    assert_eq!(
        batch.obs_flat.len(),
        reader.sample_count().min(2) * hydra_core::encoder::OBS_SIZE
    );
    assert!(
        parallel
            .report_path
            .as_ref()
            .is_some_and(|path| path.exists())
    );
    assert_eq!(
        parallel
            .report
            .as_ref()
            .expect("report should be retained")
            .build
            .total_samples,
        parallel.manifest.totals.sample_count
    );
}

#[test]
fn bounded_loose_fixture_decodes_metadata_without_dense_tail() {
    let input_dir = test_dir("bounded-input");
    let paths = vec![write_replay(&input_dir, "bounded.mjai")];
    let output_dir = test_dir("bounded-output");

    let built = build_bc_shards(&BuildBcShardsConfig {
        num_threads: Some(1),
        queue_bound: 1,
        shard_samples: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("bounded loose build should pass");
    let reader = load_bc_shard_reader(&built.manifest_path, BcShardSplit::Train)
        .expect("bounded fixture shards should load");
    let batch = reader
        .collate_host_batch_range(0, 1, false)
        .expect("bounded fixture first sample should decode");

    assert_eq!(batch.obs_flat.len(), hydra_core::encoder::OBS_SIZE);
    assert_eq!(batch.obs_flat[47 * 34], 0.25);
    assert_eq!(batch.obs_flat[48 * 34], 0.25);
    assert_eq!(batch.obs_flat[59 * 34], 0.0);
    assert_eq!(batch.obs_flat[60 * 34], 0.0);
    assert_eq!(batch.obs_flat[61 * 34], 0.0);
}

#[test]
fn resume_reuses_committed_fragments_and_rejects_mismatch() {
    let input_dir = test_dir("resume-input");
    let paths = vec![
        write_replay(&input_dir, "a.mjai"),
        write_replay(&input_dir, "b.mjai"),
    ];
    let output_dir = test_dir("resume-output");
    let base = BuildBcShardsConfig {
        resume: true,
        chunk_games: 1,
        num_threads: Some(2),
        queue_bound: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    };

    set_test_stop_after_built_chunks(Some(1));
    let err = build_bc_shards(&base).expect_err("test hook should abort after first chunk");
    set_test_stop_after_built_chunks(None);
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(!output_dir.join("manifest.json").exists());

    let resumed = build_bc_shards(&base).expect("resume should finish from committed fragment");
    assert_eq!(
        resumed
            .report
            .as_ref()
            .expect("report should exist")
            .build
            .resume_chunks_reused,
        1
    );
    assert_eq!(
        resumed
            .report
            .as_ref()
            .expect("report should exist")
            .build
            .resume_chunks_built,
        1
    );
    let reader = load_bc_shard_reader(&resumed.manifest_path, BcShardSplit::Train)
        .expect("resumed shards should load");
    assert_eq!(
        reader.sample_count() as u64,
        resumed.manifest.totals.sample_count
    );

    let mismatch = BuildBcShardsConfig {
        train_fraction: 0.5,
        ..base
    };
    let err = build_bc_shards(&mismatch).expect_err("mismatched resume config should fail");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("resume state mismatch"));
}
