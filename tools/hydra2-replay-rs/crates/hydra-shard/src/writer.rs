//! Cold shard file writer (P4-B, cold only — NEVER hot).
//!
//! Reference pattern (read-only): hydra1 `bc-shards writer.rs` — `BufWriter`
//! with a placeholder header patched at `finish`, game-atomic appends (one
//! `write_all` per game, so a crash can only lose a whole game, never a torn
//! row), rotation at game boundaries via [`ShardWriter::fits`].
//!
//! File layout: [`crate::reader::HEADER_LEN`] header bytes (magic, version,
//! bucket, row count, row stride, payload SHA-256) followed by `row_count`
//! compact rows of [`crate::full26::compact_row_bytes`]`(t)` bytes.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::digest::{PayloadHasher, raw_hex_prefixed};
use crate::full26::compact_row_bytes;
use crate::reader::{HEADER_LEN, encode_header};

/// Shard writer failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum ShardWriteError {
    Io(std::io::Error),
    /// Appended bytes are not a whole number of rows (`got` bytes).
    TornGame { got: usize },
}

impl core::fmt::Display for ShardWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShardWriteError::Io(e) => write!(f, "shard write io: {e}"),
            ShardWriteError::TornGame { got } => {
                write!(f, "shard append is not game-atomic: {got} bytes")
            }
        }
    }
}

impl std::error::Error for ShardWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShardWriteError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ShardWriteError {
    fn from(e: std::io::Error) -> Self {
        ShardWriteError::Io(e)
    }
}

/// Finished-file receipt (the reader verifies every field on open).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedShard {
    pub path: PathBuf,
    pub rows: u64,
    pub t_bucket: usize,
    pub row_bytes: usize,
    pub payload_sha256: String,
}

/// Cold shard writer: placeholder header, game-atomic appends, digest patch.
#[derive(Debug)]
pub struct ShardWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    t_bucket: usize,
    row_bytes: usize,
    rows: u64,
    hasher: PayloadHasher,
    max_rows: Option<u64>,
}

impl ShardWriter {
    /// Create a shard file at `path` for bucket `t` (the header placeholder
    /// is zeros until `finish`).
    pub fn create(path: &Path, t: usize, max_rows: Option<u64>) -> Result<Self, ShardWriteError> {
        let row_bytes = compact_row_bytes(t);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&[0u8; HEADER_LEN])?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            t_bucket: t,
            row_bytes,
            rows: 0,
            hasher: PayloadHasher::new(),
            max_rows,
        })
    }

    /// File bucket `T`.
    pub fn t_bucket(&self) -> usize {
        self.t_bucket
    }

    /// Compact row stride.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Rows committed so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// True when a `game_rows`-row game still fits under `max_rows`
    /// (rotation happens at game boundaries only; `None` max never rotates).
    pub fn fits(&self, game_rows: u64) -> bool {
        match self.max_rows {
            None => true,
            Some(max) => self.rows + game_rows <= max,
        }
    }

    /// Append one whole game's compact rows (single `write_all`; `rows` must
    /// be a whole number of file rows or nothing is written).
    pub fn append_game(&mut self, rows: &[u8]) -> Result<(), ShardWriteError> {
        if rows.len() % self.row_bytes != 0 {
            return Err(ShardWriteError::TornGame { got: rows.len() });
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.writer.write_all(rows)?;
        self.hasher.update(rows);
        self.rows += (rows.len() / self.row_bytes) as u64;
        Ok(())
    }

    /// Flush, patch the header (counts + payload digest), and return the
    /// receipt (the exact digest bound into the header).
    pub fn finish(mut self) -> Result<FinishedShard, ShardWriteError> {
        self.writer.flush()?;
        let digest = self.hasher.finalize_raw();
        let header = encode_header(self.t_bucket, self.rows, self.row_bytes, &digest);
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.flush()?;
        Ok(FinishedShard {
            path: self.path.clone(),
            rows: self.rows,
            t_bucket: self.t_bucket,
            row_bytes: self.row_bytes,
            payload_sha256: raw_hex_prefixed(&digest),
        })
    }
}
