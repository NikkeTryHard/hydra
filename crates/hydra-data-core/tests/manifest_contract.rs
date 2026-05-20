use std::io::{self, Cursor};
use std::path::PathBuf;

use hydra_data_core::{
    DataManifest, DataSource, DiscoveryManifest, DiscoveryMode, DiscoverySummary,
};

fn discovery_fixture() -> DiscoveryManifest {
    let manifest = DataManifest {
        sources: vec![
            DataSource::LooseFile(PathBuf::from("games/a.json")),
            DataSource::Archive(PathBuf::from("archives/b.tar.zst")),
            DataSource::ParsedSampleCache {
                path: PathBuf::from("cache/c.bin"),
                original_identity: "raw/c.json".to_owned(),
                original_source_path: PathBuf::from("raw/c.json"),
            },
        ],
        total_games: 2,
        train_count: 1,
        val_count: 1,
        counts_exact: false,
    };
    DiscoveryManifest::from_data_manifest(manifest, DiscoveryMode::LooseGames, 4, 0, Vec::new())
}

#[test]
fn discovery_binary_index_roundtrips_sources_and_summary() {
    let discovery = discovery_fixture();
    let mut bytes = Vec::new();
    discovery
        .write_binary_index(&mut bytes)
        .expect("write binary discovery index");
    let roundtrip =
        DiscoveryManifest::read_binary_index(&mut Cursor::new(bytes), discovery.summary.clone())
            .expect("read binary discovery index");

    assert_eq!(roundtrip, discovery);
    assert_eq!(roundtrip.summary.source_count, 3);
    assert_eq!(roundtrip.summary.ignored_archive_count, 4);
}

#[test]
fn discovery_binary_index_roundtrips_root_relative_paths() {
    let discovery = discovery_fixture();
    let root = PathBuf::from("/data/root");
    let rooted = DiscoveryManifest {
        sources: discovery
            .sources
            .into_iter()
            .map(|source| match source {
                DataSource::Archive(path) => DataSource::Archive(root.join(path)),
                DataSource::LooseFile(path) => DataSource::LooseFile(root.join(path)),
                DataSource::ParsedSampleCache {
                    path,
                    original_identity,
                    original_source_path,
                } => DataSource::ParsedSampleCache {
                    path: root.join(path),
                    original_identity,
                    original_source_path: root.join(original_source_path),
                },
            })
            .collect(),
        summary: discovery.summary,
    };

    let mut bytes = Vec::new();
    rooted
        .write_binary_index_with_root(&mut bytes, Some(&root))
        .expect("write rooted binary discovery index");
    let roundtrip = DiscoveryManifest::read_binary_index_with_root(
        &mut Cursor::new(bytes),
        rooted.summary.clone(),
        Some(&root),
    )
    .expect("read rooted binary discovery index");

    assert_eq!(roundtrip, rooted);
}

#[test]
fn discovery_binary_index_rejects_source_count_before_allocation() {
    let summary = DiscoverySummary {
        mode: DiscoveryMode::LooseGames,
        data_dir: PathBuf::new(),
        train_fraction_bits: 0,
        include_source_patterns: Vec::new(),
        exclude_source_patterns: Vec::new(),
        fingerprint: 7,
        source_count: 1,
        loose_file_count: 0,
        parsed_cache_count: 0,
        archive_count: 0,
        ignored_archive_count: 0,
        ignored_file_count: 0,
        ignored_file_examples: Vec::new(),
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: false,
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HDRIDX");
    bytes.push(1);
    bytes.push(DiscoveryMode::LooseGames as u8);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&7u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let err = DiscoveryManifest::read_binary_index(&mut Cursor::new(bytes), summary)
        .expect_err("oversized source count must fail");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("source count"));
}

#[test]
fn discovery_binary_index_rejects_matching_source_count_before_allocation() {
    let source_count = 1_000_001;
    let summary = DiscoverySummary {
        mode: DiscoveryMode::LooseGames,
        data_dir: PathBuf::new(),
        train_fraction_bits: 0,
        include_source_patterns: Vec::new(),
        exclude_source_patterns: Vec::new(),
        fingerprint: 7,
        source_count,
        loose_file_count: 0,
        parsed_cache_count: 0,
        archive_count: 0,
        ignored_archive_count: 0,
        ignored_file_count: 0,
        ignored_file_examples: Vec::new(),
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: false,
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HDRIDX");
    bytes.push(1);
    bytes.push(DiscoveryMode::LooseGames as u8);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&7u64.to_le_bytes());
    bytes.extend_from_slice(&(source_count as u64).to_le_bytes());

    let err = DiscoveryManifest::read_binary_index(&mut Cursor::new(bytes), summary)
        .expect_err("excessive matching source count must fail");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("source count"));
}

#[test]
fn discovery_binary_index_rejects_string_length_before_allocation() {
    let mut discovery = discovery_fixture();
    discovery.summary.source_count = 1;
    discovery.sources = vec![DataSource::LooseFile(PathBuf::from("a.json"))];

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HDRIDX");
    bytes.push(1);
    bytes.push(DiscoveryMode::LooseGames as u8);
    bytes.extend_from_slice(&discovery.summary.ignored_archive_count.to_le_bytes());
    bytes.extend_from_slice(&discovery.summary.ignored_file_count.to_le_bytes());
    bytes.extend_from_slice(&discovery.summary.fingerprint.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let err = DiscoveryManifest::read_binary_index(&mut Cursor::new(bytes), discovery.summary)
        .expect_err("oversized string length must fail");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("string length"));
}
