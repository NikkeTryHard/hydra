use std::path::{Path, PathBuf};

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
