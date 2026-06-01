//! Parsed-sample cache file format for BC raw-path reuse.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use hydra_data_core::sample::MjaiSample;

mod header;
mod limits;
mod path;
mod primitives;
mod sample;

pub use path::{is_parsed_sample_cache_file, parsed_sample_cache_file_name};

#[cfg(test)]
use hydra_core::action::HYDRA_ACTION_SPACE;
#[cfg(test)]
use hydra_core::encoder::OBS_SIZE;

#[cfg(test)]
use limits::{MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN, MAX_PARSED_SAMPLE_CACHE_SAMPLES};
#[cfg(test)]
use sample::test_support::{FLAG_BELIEF_FIELDS, FLAG_MIXTURE_WEIGHTS};

pub const PARSED_SAMPLE_CACHE_EXTENSION: &str = ".samples.cache";

#[cfg(test)]
const DANGER_TARGET_SIZE: usize = 3 * 34;
#[cfg(test)]
const BELIEF_FIELD_SIZE: usize = 16 * 34;

/// Parsed-sample cache metadata header.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParsedSampleCacheMetadata {
    /// Original source path recorded when the cache was written.
    pub original_source_path: PathBuf,
    /// Stable source identity used for data splits.
    pub original_identity: String,
    /// Number of parsed samples stored in the cache.
    pub sample_count: usize,
}

/// Parsed-sample cache metadata and game payload.
pub struct ParsedSampleCacheFile {
    pub metadata: ParsedSampleCacheMetadata,
    /// Parsed game payload.
    pub game: ParsedSampleCacheGame,
}

/// Parsed game payload stored in a parsed-sample cache file.
pub struct ParsedSampleCacheGame {
    /// Parsed decision samples.
    pub samples: Vec<MjaiSample>,
    /// Final game scores for all four seats.
    pub final_scores: [i32; 4],
}

pub fn write_parsed_sample_cache(
    path: &Path,
    original_source_path: &Path,
    original_identity: &str,
    game: &ParsedSampleCacheGame,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    header::write_header(
        &mut writer,
        original_source_path,
        original_identity,
        game.samples.len(),
    )?;
    header::write_final_scores(&mut writer, game.final_scores)?;

    for sample in &game.samples {
        sample::write_sample(&mut writer, sample)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_parsed_sample_cache_metadata(path: &Path) -> io::Result<ParsedSampleCacheMetadata> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = header::read_header_internal(&mut reader)?;
    let _ = header::read_final_scores(&mut reader)?;
    Ok(ParsedSampleCacheMetadata {
        original_source_path: PathBuf::from(header.original_source_path),
        original_identity: header.original_identity,
        sample_count: header.sample_count as usize,
    })
}

pub fn load_parsed_sample_cache(path: &Path) -> io::Result<ParsedSampleCacheFile> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = header::read_header_internal(&mut reader)?;
    let final_scores = header::read_final_scores(&mut reader)?;
    let sample_count = header.sample_count as usize;
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        samples.push(sample::read_sample(&mut reader)?);
    }
    Ok(ParsedSampleCacheFile {
        metadata: ParsedSampleCacheMetadata {
            original_source_path: PathBuf::from(header.original_source_path),
            original_identity: header.original_identity,
            sample_count,
        },
        game: ParsedSampleCacheGame {
            samples,
            final_scores,
        },
    })
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests;
