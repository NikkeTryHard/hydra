use std::fs;
use std::path::{Path, PathBuf};

use hydra_train_types::checkpoint::CheckpointMeta;

use crate::artifacts::{
    ManifestCacheRequest, PreflightPaths, checkpoint_meta_semantically_matches,
    load_or_scan_manifest_cache, read_discovery_manifest_cache, scan_and_write_discovery_cache,
    scan_discovery_manifest_with_progress, write_checkpoint_meta, write_discovery_manifest_cache,
    write_manifest_cache,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_checkpoint_base(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("hydra_exec_{label}_{unique}"))
}

fn temp_dir_path(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("hydra_exec_{label}_{unique}"))
}

fn touch(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(path, b"{}").expect("write test file");
}

#[test]
fn checkpoint_meta_semantic_match_ignores_timestamp_only() {
    let mut existing = CheckpointMeta::new(3, 1.25, Some(0.5), Some(1.0), Some(2.0));
    let mut candidate = existing.clone();
    candidate.timestamp = existing.timestamp.saturating_add(10);

    assert!(checkpoint_meta_semantically_matches(&existing, &candidate));

    existing.hidden_channels += 1;
    assert!(!checkpoint_meta_semantically_matches(&existing, &candidate));
}

#[test]
fn write_checkpoint_meta_preserves_existing_semantic_match() {
    let base = temp_checkpoint_base("checkpoint_meta_preserve");
    let meta_path = base.with_extension("meta.json");
    let meta = CheckpointMeta::new(4, 2.5, None, None, None);

    write_checkpoint_meta(&base, &meta).expect("write checkpoint metadata");
    let first_raw = fs::read_to_string(&meta_path).expect("read checkpoint metadata");
    let mut same_semantics = meta.clone();
    same_semantics.timestamp = meta.timestamp.saturating_add(60);
    write_checkpoint_meta(&base, &same_semantics).expect("rewrite matching checkpoint metadata");

    assert_eq!(
        fs::read_to_string(&meta_path).expect("read preserved checkpoint metadata"),
        first_raw
    );
    let _ = fs::remove_file(meta_path);
}

#[test]
fn discovery_loose_only_uses_loose_games_mode_and_exact_counts() {
    let root = temp_dir_path("discovery_loose_only");
    touch(&root.join("b.json"));
    touch(&root.join("nested").join("a.json.gz"));

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan loose-only discovery");

    assert_eq!(
        discovery.summary.mode,
        hydra_data_core::DiscoveryMode::LooseGames
    );
    assert_eq!(discovery.summary.source_count, 2);
    assert_eq!(discovery.summary.loose_file_count, 2);
    assert_eq!(discovery.summary.archive_count, 0);
    assert_eq!(discovery.summary.ignored_archive_count, 0);
    assert_eq!(discovery.summary.total_games, 2);
    assert_eq!(discovery.summary.train_count, 2);
    assert_eq!(discovery.summary.val_count, 0);
    assert!(discovery.summary.counts_exact);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_archive_only_folder_uses_archive_multi() {
    let root = temp_dir_path("discovery_archive_only");
    touch(&root.join("b.tar"));
    touch(&root.join("nested").join("a.tar.zst"));

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        0.9,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan archive-only discovery");

    assert_eq!(
        discovery.summary.mode,
        hydra_data_core::DiscoveryMode::ArchiveMulti
    );
    assert_eq!(discovery.summary.source_count, 2);
    assert_eq!(discovery.summary.loose_file_count, 0);
    assert_eq!(discovery.summary.archive_count, 2);
    assert_eq!(discovery.summary.ignored_archive_count, 0);
    assert!(!discovery.summary.counts_exact);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_mixed_folder_uses_loose_games_and_ignores_archives() {
    let root = temp_dir_path("discovery_mixed");
    touch(&root.join("game.json"));
    touch(&root.join("ignored.tar"));
    touch(&root.join("nested").join("ignored.tar.zst"));

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan mixed discovery");

    assert_eq!(
        discovery.summary.mode,
        hydra_data_core::DiscoveryMode::LooseGames
    );
    assert_eq!(discovery.summary.source_count, 1);
    assert_eq!(discovery.summary.loose_file_count, 1);
    assert_eq!(discovery.summary.archive_count, 0);
    assert_eq!(discovery.summary.ignored_archive_count, 2);
    assert_eq!(discovery.summary.total_games, 1);
    assert!(discovery.summary.counts_exact);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_empty_or_junk_folder_errors_with_counts_and_examples() {
    let root = temp_dir_path("discovery_junk_error");
    touch(&root.join("notes.txt"));
    touch(&root.join("nested").join("partial.tmp"));

    let err = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect_err("junk-only discovery should hard error")
    .to_string();

    assert!(err.contains("no usable training data sources"));
    assert!(err.contains("loose_files=0"));
    assert!(err.contains("archive_files=0"));
    assert!(err.contains("ignored_files=2"));
    assert!(err.contains("notes.txt"));
    assert!(err.contains("partial.tmp"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_binary_index_roundtrips_with_summary_counts() {
    let root = temp_dir_path("discovery_roundtrip");
    touch(&root.join("game-a.json"));
    touch(&root.join("game-b.json.zst"));
    let summary_path = root.join("summary.json");
    let index_path = root.join("index.bin");

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan discovery");
    write_discovery_manifest_cache(&summary_path, &index_path, &discovery)
        .expect("write discovery cache");
    let roundtrip = read_discovery_manifest_cache(&summary_path, &index_path)
        .expect("read discovery cache")
        .expect("discovery cache exists");

    assert_eq!(roundtrip, discovery);
    assert_eq!(roundtrip.summary.source_count, 2);
    assert_eq!(roundtrip.summary.total_games, 2);
    assert_eq!(roundtrip.summary.train_count, 2);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_binary_index_stores_root_relative_paths() {
    let root = temp_dir_path("discovery_relative_roundtrip");
    touch(&root.join("nested").join("game-a.json"));
    let summary_path = root.join("summary.json");
    let index_path = root.join("index.bin");

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan discovery");
    write_discovery_manifest_cache(&summary_path, &index_path, &discovery)
        .expect("write discovery cache");

    let bytes = fs::read(&index_path).expect("read index bytes");
    assert!(!String::from_utf8_lossy(&bytes).contains(&root.display().to_string()));
    let roundtrip = read_discovery_manifest_cache(&summary_path, &index_path)
        .expect("read discovery cache")
        .expect("discovery cache exists");
    assert_eq!(roundtrip.sources, discovery.sources);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn load_or_scan_prefers_valid_discovery_cache_over_legacy_manifest() {
    let root = temp_dir_path("discovery_authoritative_hit");
    let data_dir = root.join("data");
    touch(&data_dir.join("game-a.json"));
    let paths = PreflightPaths::new(&crate::artifacts::BcArtifactPaths::new(&root, 0));
    let expected = scan_and_write_discovery_cache(
        &paths.discovery_summary_path,
        &paths.discovery_index_path,
        &data_dir,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
        "test data",
    )
    .expect("write compact discovery");
    touch(&data_dir.join("game-b.json"));
    write_manifest_cache(
        &paths.manifest_cache_path,
        &hydra_train_runtime::preflight::ManifestCacheEntry {
            data_dir: data_dir.clone(),
            train_fraction_bits: 1.0f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: hydra_data_core::DataManifest {
                sources: vec![hydra_data_core::DataSource::LooseFile(
                    data_dir.join("legacy.json"),
                )],
                total_games: 1,
                train_count: 1,
                val_count: 0,
                counts_exact: true,
            },
        },
    )
    .expect("write legacy manifest");

    let mut hit = false;
    let manifest = load_or_scan_manifest_cache(
        ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &data_dir,
            train_fraction: 1.0,
            source_filters: &hydra_train_runtime::config::SourceFilterConfig::default(),
            progress: None,
            scan_error_context: "test data",
        },
        |_| hit = true,
    )
    .expect("load cache");

    assert!(hit);
    assert_eq!(manifest, expected);
    assert_eq!(manifest.sources.len(), 1);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stale_discovery_metadata_is_rejected_and_rescanned() {
    let root = temp_dir_path("discovery_stale_metadata");
    let data_dir = root.join("data");
    touch(&data_dir.join("game-a.json"));
    let paths = PreflightPaths::new(&crate::artifacts::BcArtifactPaths::new(&root, 0));
    scan_and_write_discovery_cache(
        &paths.discovery_summary_path,
        &paths.discovery_index_path,
        &data_dir,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
        "test data",
    )
    .expect("write compact discovery");
    let mut raw: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&paths.discovery_summary_path).expect("read summary"),
    )
    .expect("parse summary");
    raw["train_fraction_bits"] = serde_json::json!(0.5f32.to_bits());
    fs::write(
        &paths.discovery_summary_path,
        serde_json::to_string(&raw).expect("serialize stale summary"),
    )
    .expect("write stale summary");
    touch(&data_dir.join("game-b.json"));

    let mut hit = false;
    let manifest = load_or_scan_manifest_cache(
        ManifestCacheRequest {
            cache_path: &paths.manifest_cache_path,
            discovery_summary_path: &paths.discovery_summary_path,
            discovery_index_path: &paths.discovery_index_path,
            data_dir: &data_dir,
            train_fraction: 1.0,
            source_filters: &hydra_train_runtime::config::SourceFilterConfig::default(),
            progress: None,
            scan_error_context: "test data",
        },
        |_| hit = true,
    )
    .expect("rescan stale cache");

    assert!(!hit);
    assert_eq!(manifest.sources.len(), 2);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_archive_extensions_cover_tgz_and_tar_gz() {
    let root = temp_dir_path("discovery_archive_exts");
    touch(&root.join("a.tgz"));
    touch(&root.join("b.tar.gz"));

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan archives");

    assert_eq!(
        discovery.summary.mode,
        hydra_data_core::DiscoveryMode::ArchiveMulti
    );
    assert_eq!(discovery.summary.archive_count, 2);
    assert_eq!(discovery.summary.source_count, 2);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn discovery_summary_persists_ignored_counts_and_examples() {
    let root = temp_dir_path("discovery_ignored_summary");
    touch(&root.join("game-a.json"));
    touch(&root.join("notes.txt"));
    touch(&root.join("nested").join("partial.tmp"));
    let summary_path = root.join("summary.json");
    let index_path = root.join("index.bin");

    let discovery = scan_discovery_manifest_with_progress(
        &root,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
    )
    .expect("scan discovery");
    write_discovery_manifest_cache(&summary_path, &index_path, &discovery)
        .expect("write discovery cache");
    let raw: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&summary_path).expect("read summary"))
            .expect("parse summary");

    assert_eq!(raw["ignored_file_count"], serde_json::json!(2));
    assert_eq!(
        raw["ignored_file_examples"]
            .as_array()
            .expect("ignored examples array")
            .len(),
        2
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn preflight_paths_use_output_root_layout_and_compact_discovery_names() {
    let output_dir = temp_dir_path("preflight_output_root_layout");
    let artifacts = crate::artifacts::BcArtifactPaths::new(&output_dir, 0);
    let paths = PreflightPaths::new(&artifacts);

    assert_eq!(
        paths.cache_path,
        output_dir.join("preflight/cache/preflight_cache.json")
    );
    assert_eq!(
        paths.manifest_cache_path,
        output_dir.join("preflight/cache/preflight_manifest.json")
    );
    assert_eq!(
        paths.discovery_summary_path,
        output_dir.join("preflight/discovery/summary.json")
    );
    assert_eq!(
        paths.discovery_index_path,
        output_dir.join("preflight/discovery/index.bin")
    );
    assert_eq!(
        paths.events_log_path,
        output_dir.join("preflight/logs/events.jsonl")
    );
    assert_eq!(
        paths.metrics_log_path,
        output_dir.join("preflight/logs/metrics.jsonl")
    );
    assert_eq!(
        paths.candidates_log_path,
        output_dir.join("preflight/logs/candidates.jsonl")
    );
    assert_eq!(
        paths.state_path,
        output_dir.join("preflight/state/preflight_state.json")
    );
    assert_eq!(
        paths.report_path,
        output_dir.join("preflight/reports/preflight_report.json")
    );
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn scan_and_write_discovery_cache_does_not_write_full_legacy_manifest() {
    let root = temp_dir_path("compact_discovery_write");
    let data_dir = root.join("data");
    touch(&data_dir.join("game-a.json"));
    touch(&data_dir.join("game-b.json.zst"));
    let paths = PreflightPaths::new(&crate::artifacts::BcArtifactPaths::new(&root, 0));

    let manifest = scan_and_write_discovery_cache(
        &paths.discovery_summary_path,
        &paths.discovery_index_path,
        &data_dir,
        1.0,
        &hydra_train_runtime::config::SourceFilterConfig::default(),
        None,
        "test data",
    )
    .expect("scan compact discovery");

    assert_eq!(manifest.sources.len(), 2);
    assert!(paths.discovery_summary_path.exists());
    assert!(paths.discovery_index_path.exists());
    assert!(!paths.manifest_cache_path.exists());
    let _ = fs::remove_dir_all(root);
}
