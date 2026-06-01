use std::path::{Path, PathBuf};

mod io;

/// Identifies a game's location for deterministic train/validation splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameLocator {
    /// For loose files: filename. For archive entries: "archive_name/entry_name".
    pub identity: String,
}

/// Shared include/exclude filters for replay source discovery.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub struct SourceFilterConfig {
    /// Source path substrings to include. Empty means include all sources.
    #[serde(default)]
    pub include_source_patterns: Vec<String>,
    /// Source path substrings to exclude after include filtering.
    #[serde(default)]
    pub exclude_source_patterns: Vec<String>,
}
/// Discovery policy selected for a dataset path.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMode {
    /// A single archive file was provided directly.
    ArchiveSingle,
    /// Loose MJAI files and parsed caches are used directly.
    #[default]
    LooseGames,
    /// A directory contained only archive files.
    ArchiveMulti,
}

/// Compact summary for discovered training data sources.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DiscoverySummary {
    /// Discovery policy selected for the input path.
    pub mode: DiscoveryMode,
    /// Root path used for source discovery.
    #[serde(default)]
    pub data_dir: PathBuf,
    /// Raw train fraction bits used for deterministic split assignment.
    #[serde(default)]
    pub train_fraction_bits: u32,
    /// Source path substrings included during discovery. Empty means include all sources.
    #[serde(default)]
    pub include_source_patterns: Vec<String>,
    /// Source path substrings excluded during discovery.
    #[serde(default)]
    pub exclude_source_patterns: Vec<String>,
    /// Cheap fingerprint of the discovery inputs and selected sources.
    #[serde(default)]
    pub fingerprint: u64,
    /// Sources kept after discovery and filtering.
    pub source_count: usize,
    /// Loose MJAI replay files kept after filtering.
    pub loose_file_count: usize,
    /// Parsed sample cache files kept after filtering.
    pub parsed_cache_count: usize,
    /// Archive files kept after filtering.
    pub archive_count: usize,
    /// Archive files ignored because a mixed directory selected loose-games mode.
    #[serde(default)]
    pub ignored_archive_count: usize,
    /// Unsupported/junk files ignored during discovery.
    #[serde(default)]
    pub ignored_file_count: usize,
    /// First few unsupported/junk files ignored during discovery.
    #[serde(default)]
    pub ignored_file_examples: Vec<PathBuf>,
    /// Total game count when exact counts are available.
    pub total_games: usize,
    /// Number of games assigned to training when exact counts are available.
    pub train_count: usize,
    /// Number of games assigned to validation when exact counts are available.
    pub val_count: usize,
    /// True when total/train/validation counts were computed exactly.
    pub counts_exact: bool,
}

/// Compact manifest plus human-readable discovery summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DiscoveryManifest {
    /// Ordered source list used by streaming loaders.
    pub sources: Vec<DataSource>,
    /// Summary counts and selected discovery policy.
    pub summary: DiscoverySummary,
}

impl SourceFilterConfig {
    /// Returns true when no include or exclude filters are configured.
    pub fn is_empty(&self) -> bool {
        self.include_source_patterns.is_empty() && self.exclude_source_patterns.is_empty()
    }
}

/// Serializable manifest for discovered training data sources.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DataManifest {
    /// Ordered source list used by streaming loaders.
    pub sources: Vec<DataSource>,
    /// Total game count when exact counts are available; archive-only sources may leave this at 0.
    pub total_games: usize,
    /// Number of games assigned to the training split when exact counts are available.
    pub train_count: usize,
    /// Number of games assigned to the validation split when exact counts are available.
    pub val_count: usize,
    /// True when total/train/validation counts were computed exactly.
    pub counts_exact: bool,
}

impl DataManifest {
    /// Builds the legacy training manifest view from a compact discovery manifest.
    #[must_use]
    pub fn from_discovery(discovery: &DiscoveryManifest) -> Self {
        Self {
            sources: discovery.sources.clone(),
            total_games: discovery.summary.total_games,
            train_count: discovery.summary.train_count,
            val_count: discovery.summary.val_count,
            counts_exact: discovery.summary.counts_exact,
        }
    }
}

impl DiscoveryManifest {
    /// Builds a compact discovery manifest from the legacy training manifest view.
    #[must_use]
    pub fn from_data_manifest(
        manifest: DataManifest,
        mode: DiscoveryMode,
        ignored_archive_count: usize,
        ignored_file_count: usize,
        ignored_file_examples: Vec<PathBuf>,
    ) -> Self {
        let mut loose_file_count = 0usize;
        let mut parsed_cache_count = 0usize;
        let mut archive_count = 0usize;
        for source in &manifest.sources {
            match source {
                DataSource::Archive(_) => archive_count += 1,
                DataSource::LooseFile(_) => loose_file_count += 1,
                DataSource::ParsedSampleCache { .. } => parsed_cache_count += 1,
            }
        }
        Self {
            summary: DiscoverySummary {
                mode,
                data_dir: PathBuf::new(),
                train_fraction_bits: 0,
                include_source_patterns: Vec::new(),
                exclude_source_patterns: Vec::new(),
                fingerprint: 0,
                source_count: manifest.sources.len(),
                loose_file_count,
                parsed_cache_count,
                archive_count,
                ignored_archive_count,
                ignored_file_count,
                ignored_file_examples,
                total_games: manifest.total_games,
                train_count: manifest.train_count,
                val_count: manifest.val_count,
                counts_exact: manifest.counts_exact,
            },
            sources: manifest.sources,
        }
    }
}

/// Binary discovery index format version.
pub const DISCOVERY_INDEX_VERSION: u8 = 1;

/// Serializable identity for a discovered data source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DataSource {
    /// Tar or tar.zst archive containing MJAI replay entries.
    Archive(PathBuf),
    /// Single loose MJAI replay file.
    LooseFile(PathBuf),
    /// Parsed sample cache plus original replay identity for deterministic splitting.
    ParsedSampleCache {
        /// Parsed sample cache path.
        path: PathBuf,
        /// Stable original replay identity used for train/validation splitting.
        original_identity: String,
        /// Original source replay path used for user-facing filtering and diagnostics.
        original_source_path: PathBuf,
    },
}

impl DataSource {
    /// Returns the path that identifies this source on disk.
    pub fn path(&self) -> &Path {
        match self {
            Self::Archive(path) | Self::LooseFile(path) => path.as_path(),
            Self::ParsedSampleCache { path, .. } => path.as_path(),
        }
    }
}
