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

struct TestStopGuard;

impl TestStopGuard {
    fn clear() -> Self {
        set_test_stop_after_built_chunks(None);
        Self
    }
}

impl Drop for TestStopGuard {
    fn drop(&mut self) {
        set_test_stop_after_built_chunks(None);
    }
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
fn build_bc_shards_rejects_invalid_train_fraction() {
    for (train_fraction, message) in [
        (f32::NAN, "train_fraction must be finite"),
        (-0.1, "train_fraction must be in 0..=1"),
        (1.1, "train_fraction must be in 0..=1"),
    ] {
        let config = BuildBcShardsConfig {
            train_fraction,
            ..BuildBcShardsConfig::default()
        };
        let err = build_bc_shards(&config).expect_err("invalid train_fraction should fail");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(err.to_string(), message);
    }
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
    let report = parallel.report.as_ref().expect("report should be retained");
    assert_eq!(
        report.abi.storage_layout,
        hydra_bc_shards::STORAGE_LAYOUT_COMPACT
    );
    assert_eq!(
        report.abi.feature_flags,
        parallel.manifest.splits[0].feature_flags
    );
    assert_eq!(
        report.abi.record_size,
        parallel.manifest.splits[0].record_size
    );
    assert_eq!(report.disk.output_dir, parallel_dir.display().to_string());
    assert_eq!(report.disk.projected_output_bytes, None);
    assert_eq!(report.disk.projected_sample_count, None);
    assert_eq!(report.disk.projection_source, "unavailable");
    assert_eq!(report.plan_splits.len(), 1);
    assert_eq!(report.plan_splits[0].split, "train");
    assert_eq!(report.plan_splits[0].planned_games, paths.len());
    assert_eq!(
        report.output.splits[0].feature_flags,
        parallel.manifest.splits[0].feature_flags
    );
    assert_eq!(
        report.output.splits[0].record_size,
        parallel.manifest.splits[0].record_size
    );
    assert_eq!(
        report.output.splits[0].bytes_per_sample,
        Some(report.output.splits[0].byte_len as f64 / report.output.splits[0].sample_count as f64)
    );
    assert!(report.output.splits[0].min_shard_bytes.is_some());
    assert!(report.output.splits[0].max_shard_bytes.is_some());
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

#[test]
fn build_bc_shards_stops_after_max_games() {
    let _stop_guard = TestStopGuard::clear();
    let input_dir = test_dir("max-games-input");
    let paths = vec![
        write_replay(&input_dir, "a.mjai"),
        write_replay(&input_dir, "b.mjai"),
        write_replay(&input_dir, "c.mjai"),
    ];
    let output_dir = test_dir("max-games-output");

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(2),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("max-games build should pass");
    let report = built.report.as_ref().expect("report should exist");

    assert_eq!(
        report.build.loaded_games + report.build.skipped_games + report.build.empty_games,
        2
    );
    assert_eq!(report.command.limit_max_games, Some(2));
    assert!(report.build.limit_reached);
    assert_eq!(
        built.manifest.totals.sample_count,
        report.build.total_samples
    );
}

#[test]
fn build_bc_shards_stops_after_max_samples_after_current_game() {
    let _stop_guard = TestStopGuard::clear();
    let input_dir = test_dir("max-samples-input");
    let paths = vec![
        write_replay(&input_dir, "a.mjai"),
        write_replay(&input_dir, "b.mjai"),
        write_replay(&input_dir, "c.mjai"),
    ];
    let output_dir = test_dir("max-samples-output");

    let single = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(1),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &test_dir("max-samples-single"), &paths)
    })
    .expect("single game build should pass");
    let per_game_samples = single.manifest.totals.sample_count;
    assert!(per_game_samples > 0);

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_samples: Some(per_game_samples as usize + 1),
        num_threads: Some(1),
        queue_bound: 1,
        chunk_games: paths.len(),
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("max-samples build should pass");
    let report = built.report.as_ref().expect("report should exist");

    assert_eq!(report.build.loaded_games, 2);
    assert_eq!(report.build.total_samples, per_game_samples * 2);
    assert!(report.build.total_samples > per_game_samples + 1);
    assert_eq!(
        report.command.limit_max_samples,
        Some(per_game_samples as usize + 1)
    );
    assert!(report.build.limit_reached);
}

#[test]
fn build_bc_shards_stops_after_max_samples_before_next_source_group() {
    let _stop_guard = TestStopGuard::clear();
    let input_a = test_dir("max-samples-group-a");
    let input_b = test_dir("max-samples-group-b");
    let first = write_replay(&input_a, "a.mjai");
    let second = write_replay(&input_b, "b.mjai");
    let paths = vec![first.clone(), second.clone()];
    let output_dir = test_dir("max-samples-group-output");

    let single = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(1),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_a, &test_dir("max-samples-group-single"), &[first])
    })
    .expect("single game build should pass");
    let per_game_samples = single.manifest.totals.sample_count;
    assert!(per_game_samples > 0);

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_samples: Some(per_game_samples as usize),
        num_threads: Some(1),
        queue_bound: 1,
        chunk_games: paths.len(),
        ..base_config(&input_a, &output_dir, &paths)
    })
    .expect("max-samples cross-group build should pass");
    let report = built.report.as_ref().expect("report should exist");

    assert_eq!(report.build.loaded_games, 1);
    assert_eq!(report.build.total_samples, per_game_samples);
    assert!(report.build.limit_reached);
}

#[test]
fn build_bc_shards_max_samples_counts_prior_chunks() {
    let _stop_guard = TestStopGuard::clear();
    let input_dir = test_dir("max-samples-prior-chunks-input");
    let paths = vec![
        write_replay(&input_dir, "a.mjai"),
        write_replay(&input_dir, "b.mjai"),
        write_replay(&input_dir, "c.mjai"),
        write_replay(&input_dir, "d.mjai"),
    ];
    let single = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(1),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &test_dir("max-samples-prior-single"), &paths)
    })
    .expect("single game build should pass");
    let per_game_samples = single.manifest.totals.sample_count;
    assert!(per_game_samples > 0);

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_samples: Some(per_game_samples as usize + 1),
        chunk_games: 1,
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &test_dir("max-samples-prior-output"), &paths)
    })
    .expect("max-samples build should pass");

    assert_eq!(built.manifest.totals.sample_count, per_game_samples * 2);
    let report = built.report.as_ref().expect("report should exist");
    assert_eq!(report.build.loaded_games, 2);
    assert!(report.build.limit_reached);
}
#[test]
fn build_report_records_limit_fields() {
    let input_dir = test_dir("limit-report-input");
    let paths = vec![write_replay(&input_dir, "a.mjai")];
    let output_dir = test_dir("limit-report-output");

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(1),
        max_samples: Some(1),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("limit report build should pass");
    let report = built.report.as_ref().expect("report should exist");

    assert_eq!(report.command.limit_max_games, Some(1));
    assert_eq!(report.command.limit_max_samples, Some(1));
    assert!(report.build.limit_reached);
    assert_eq!(
        report.build.samples_per_loaded_non_empty_game,
        Some(report.build.total_samples as f64 / report.build.loaded_games as f64)
    );
}

#[test]
fn build_report_does_not_mark_natural_exact_max_games_as_limit_reached() {
    let input_dir = test_dir("natural-max-games-input");
    let paths = vec![write_replay(&input_dir, "a.mjai")];
    let output_dir = test_dir("natural-max-games-output");

    let built = build_bc_shards(&BuildBcShardsConfig {
        max_games: Some(1),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("natural max-games build should pass");
    let report = built.report.as_ref().expect("report should exist");

    assert_eq!(report.command.limit_max_games, Some(1));
    assert_eq!(
        report.build.loaded_games + report.build.skipped_games + report.build.empty_games,
        1
    );
    assert!(!report.build.limit_reached);
}

#[test]
fn progress_jsonl_truncates_and_starts_with_run_started() {
    let _stop_guard = TestStopGuard::clear();
    let input_dir = test_dir("progress-input");
    let paths = vec![write_replay(&input_dir, "a.mjai")];
    let output_dir = test_dir("progress-output");
    let progress_path = output_dir.join("progress.jsonl");
    fs::write(&progress_path, "stale\n").expect("stale progress should be writable");

    let built = build_bc_shards(&BuildBcShardsConfig {
        progress_jsonl_name: Some("progress.jsonl".to_string()),
        num_threads: Some(1),
        queue_bound: 1,
        ..base_config(&input_dir, &output_dir, &paths)
    })
    .expect("progress build should pass");

    let contents = fs::read_to_string(&progress_path).expect("progress should be readable");
    assert!(!contents.contains("stale"));
    let first = contents
        .lines()
        .next()
        .expect("progress should have first event");
    let event: serde_json::Value =
        serde_json::from_str(first).expect("progress event should be json");
    assert_eq!(event["event"], "run_started");
    assert_eq!(event["scope"], "cumulative");
    assert_eq!(event["output_dir"], output_dir.display().to_string());
    assert_eq!(event["manifest_name"], "manifest.json");
    assert_eq!(event["split_mode"], "train");
    assert_eq!(event["shard_samples"], 2);
    assert_eq!(event["train_fraction"], 1.0);
    assert_eq!(
        event["storage_layout"],
        hydra_bc_shards::STORAGE_LAYOUT_COMPACT
    );
    assert_eq!(
        event["feature_flags"],
        built.manifest.splits[0].feature_flags
    );
    assert_eq!(event["record_size"], built.manifest.splits[0].record_size);
}
