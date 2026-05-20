use std::io::{self, Read, Write};
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
const MAX_DISCOVERY_INDEX_STRING_LEN: usize = 16 * 1024 * 1024;
const MAX_DISCOVERY_INDEX_SOURCES: usize = 1_000_000;

impl DiscoveryManifest {
    /// Writes the discovery source index in a compact binary format.
    pub fn write_binary_index<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.write_binary_index_with_root(writer, None)
    }

    /// Writes the discovery source index, storing paths under `root` as root-relative.
    pub fn write_binary_index_with_root<W: Write>(
        &self,
        writer: &mut W,
        root: Option<&Path>,
    ) -> io::Result<()> {
        writer.write_all(b"HDRIDX")?;
        writer.write_all(&[DISCOVERY_INDEX_VERSION])?;
        writer.write_all(&[self.summary.mode as u8])?;
        write_u64(writer, self.summary.ignored_archive_count as u64)?;
        write_u64(writer, self.summary.ignored_file_count as u64)?;
        write_u64(writer, self.summary.fingerprint)?;
        write_u64(writer, self.sources.len() as u64)?;
        for source in &self.sources {
            match source {
                DataSource::Archive(path) => {
                    writer.write_all(&[0])?;
                    write_path(writer, root, path)?;
                }
                DataSource::LooseFile(path) => {
                    writer.write_all(&[1])?;
                    write_path(writer, root, path)?;
                }
                DataSource::ParsedSampleCache {
                    path,
                    original_identity,
                    original_source_path,
                } => {
                    writer.write_all(&[2])?;
                    write_path(writer, root, path)?;
                    write_str(writer, original_identity)?;
                    write_path(writer, root, original_source_path)?;
                }
            }
        }
        Ok(())
    }

    /// Reads a binary discovery source index.
    pub fn read_binary_index<R: Read>(
        reader: &mut R,
        summary: DiscoverySummary,
    ) -> io::Result<Self> {
        Self::read_binary_index_with_root(reader, summary, None)
    }

    /// Reads a binary discovery source index, reconstructing root-relative paths.
    pub fn read_binary_index_with_root<R: Read>(
        reader: &mut R,
        summary: DiscoverySummary,
        root: Option<&Path>,
    ) -> io::Result<Self> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != b"HDRIDX" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid discovery index magic",
            ));
        }
        let version = read_u8(reader)?;
        if version != DISCOVERY_INDEX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported discovery index version",
            ));
        }
        let mode = read_u8(reader)?;
        if mode != summary.mode as u8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "discovery summary mode does not match index",
            ));
        }
        let ignored_archive_count = read_count(reader, "ignored archive count")?;
        if ignored_archive_count != summary.ignored_archive_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "discovery summary ignored archive count does not match index",
            ));
        }
        let ignored_file_count = read_count(reader, "ignored file count")?;
        if ignored_file_count != summary.ignored_file_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "discovery summary ignored file count does not match index",
            ));
        }
        let fingerprint = read_u64(reader)?;
        if fingerprint != summary.fingerprint {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "discovery summary fingerprint does not match index",
            ));
        }
        let len = read_count(reader, "source count")?;
        if len != summary.source_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "discovery summary source count does not match index",
            ));
        }
        if len > MAX_DISCOVERY_INDEX_SOURCES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "discovery index source count {len} exceeds maximum {MAX_DISCOVERY_INDEX_SOURCES}"
                ),
            ));
        }
        let mut sources = Vec::with_capacity(len);
        for _ in 0..len {
            sources.push(match read_u8(reader)? {
                0 => DataSource::Archive(read_path(reader, root)?),
                1 => DataSource::LooseFile(read_path(reader, root)?),
                2 => DataSource::ParsedSampleCache {
                    path: read_path(reader, root)?,
                    original_identity: read_string(reader)?,
                    original_source_path: read_path(reader, root)?,
                },
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid discovery index source tag",
                    ));
                }
            });
        }

        Ok(Self { sources, summary })
    }
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_count<R: Read>(reader: &mut R, field: &'static str) -> io::Result<usize> {
    usize::try_from(read_u64(reader)?).map_err(|_| invalid_count(field))
}

fn invalid_count(field: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("discovery index {field} exceeds platform usize"),
    )
}

fn write_path<W: Write>(writer: &mut W, root: Option<&Path>, path: &Path) -> io::Result<()> {
    if let Some(root) = root
        && let Ok(relative) = path.strip_prefix(root)
    {
        writer.write_all(&[1])?;
        write_str(writer, &relative.to_string_lossy())
    } else {
        writer.write_all(&[0])?;
        write_str(writer, &path.to_string_lossy())
    }
}

fn read_path<R: Read>(reader: &mut R, root: Option<&Path>) -> io::Result<PathBuf> {
    let relative = read_u8(reader)? != 0;
    let path = PathBuf::from(read_string(reader)?);
    if relative {
        Ok(root.unwrap_or_else(|| Path::new("")).join(path))
    } else {
        Ok(path)
    }
}

fn write_str<W: Write>(writer: &mut W, value: &str) -> io::Result<()> {
    let bytes = value.as_bytes();
    write_u64(writer, bytes.len() as u64)?;
    writer.write_all(bytes)
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = read_count(reader, "string length")?;
    if len > MAX_DISCOVERY_INDEX_STRING_LEN {
        return Err(invalid_count("string length"));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
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
