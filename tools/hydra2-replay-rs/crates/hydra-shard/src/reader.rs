//! Cold shard mmap reader (P4-B, cold only — NEVER hot).
//!
//! Reference pattern (read-only): hydra1 `bc-shards reader.rs` — mmap with
//! `Advice::Sequential`, header verify, exact-length assert, then
//! `collate_into` (see [`crate::collate`]).
//!
//! Header layout (64 bytes, LE):
//! ```text
//! magic[8] = b"HYSHARD1" | version u16 = 1 | t_bucket u16 | flags u16 | reserved u16
//! row_count u64 | row_bytes u32 | reserved2 u32 | payload_sha256[32]
//! ```
//! `open` verifies, in order: magic, version, bucket geometry (`row_bytes`
//! equals [`crate::full26::compact_row_bytes`]`(t_bucket)`), exact file
//! length (`HEADER_LEN + rows * row_bytes`), then the payload SHA-256.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Advice, Mmap};

use crate::digest::raw_hex_prefixed;
use crate::full26::{COMPACT_T_BUCKETS, compact_row_bytes};

/// Shard magic (`HYSHARD1`).
pub const SHARD_MAGIC: &[u8; 8] = b"HYSHARD1";
/// Shard format version.
pub const SHARD_VERSION: u16 = 1;
/// Header length in bytes.
pub const HEADER_LEN: usize = 64;

/// Parsed file header (verified on open).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardHeader {
    pub t_bucket: usize,
    pub rows: u64,
    pub row_bytes: usize,
    pub payload_sha256: [u8; 32],
}

/// Shard reader failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum ShardReadError {
    Io(std::io::Error),
    Mmap(std::io::Error),
    Advise(std::io::Error),
    BadMagic { got: [u8; 8] },
    VersionMismatch { got: u16 },
    UnknownBucket { got: u16 },
    GeometryMismatch { expected: usize, got: usize },
    LengthMismatch { expected: u64, got: u64 },
    ShaMismatch { expected: String, got: String },
    Range { index: u64, rows: u64 },
}

impl core::fmt::Display for ShardReadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShardReadError::Io(e) => write!(f, "shard read io: {e}"),
            ShardReadError::Mmap(e) => write!(f, "shard mmap: {e}"),
            ShardReadError::Advise(e) => write!(f, "shard advise: {e}"),
            ShardReadError::BadMagic { got } => {
                write!(f, "shard bad magic: {got:02X?}")
            }
            ShardReadError::VersionMismatch { got } => {
                write!(f, "shard version: expected {SHARD_VERSION}, got {got}")
            }
            ShardReadError::UnknownBucket { got } => {
                write!(f, "shard unknown T bucket: {got}")
            }
            ShardReadError::GeometryMismatch { expected, got } => {
                write!(f, "shard row stride: expected {expected}, got {got}")
            }
            ShardReadError::LengthMismatch { expected, got } => {
                write!(f, "shard length: expected {expected}, got {got}")
            }
            ShardReadError::ShaMismatch { expected, got } => {
                write!(f, "shard sha: expected {expected}, got {got}")
            }
            ShardReadError::Range { index, rows } => {
                write!(f, "shard row {index} out of {rows}")
            }
        }
    }
}

impl std::error::Error for ShardReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShardReadError::Io(e)
            | ShardReadError::Mmap(e)
            | ShardReadError::Advise(e) => Some(e),
            _ => None,
        }
    }
}

/// Encode a verified header (writer-side; the reader parses the same bytes).
pub fn encode_header(t_bucket: usize, rows: u64, row_bytes: usize, digest: &[u8; 32]) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[..8].copy_from_slice(SHARD_MAGIC);
    h[8..10].copy_from_slice(&SHARD_VERSION.to_le_bytes());
    h[10..12].copy_from_slice(&(t_bucket as u16).to_le_bytes());
    // h[12..16]: flags + reserved (zero).
    h[16..24].copy_from_slice(&rows.to_le_bytes());
    h[24..28].copy_from_slice(&(row_bytes as u32).to_le_bytes());
    // h[28..32]: reserved2 (zero).
    h[32..64].copy_from_slice(digest);
    h
}

fn parse_header(bytes: &[u8]) -> Result<ShardHeader, ShardReadError> {
    if bytes.len() < HEADER_LEN {
        return Err(ShardReadError::LengthMismatch {
            expected: HEADER_LEN as u64,
            got: bytes.len() as u64,
        });
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[..8]);
    if &magic != SHARD_MAGIC {
        return Err(ShardReadError::BadMagic { got: magic });
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != SHARD_VERSION {
        return Err(ShardReadError::VersionMismatch { got: version });
    }
    let bucket = u16::from_le_bytes([bytes[10], bytes[11]]);
    if !COMPACT_T_BUCKETS.contains(&(bucket as usize)) {
        return Err(ShardReadError::UnknownBucket { got: bucket });
    }
    let t = bucket as usize;
    let rows = u64::from_le_bytes(bytes[16..24].try_into().map_err(|_| {
        ShardReadError::LengthMismatch {
            expected: HEADER_LEN as u64,
            got: bytes.len() as u64,
        }
    })?);
    let row_bytes = u32::from_le_bytes(bytes[24..28].try_into().map_err(|_| {
        ShardReadError::LengthMismatch {
            expected: HEADER_LEN as u64,
            got: bytes.len() as u64,
        }
    })?) as usize;
    let expected_stride = compact_row_bytes(t);
    if row_bytes != expected_stride {
        return Err(ShardReadError::GeometryMismatch {
            expected: expected_stride,
            got: row_bytes,
        });
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes[32..64]);
    Ok(ShardHeader {
        t_bucket: t,
        rows,
        row_bytes,
        payload_sha256: digest,
    })
}

/// Cold shard reader: mmap + sequential advice + verified header.
#[derive(Debug)]
pub struct ShardReader {
    path: PathBuf,
    mmap: Mmap,
    header: ShardHeader,
}

impl ShardReader {
    /// Open + verify: magic, version, bucket geometry, exact length, payload
    /// SHA-256 (recomputed over the mapped payload — cold path, rare).
    pub fn open(path: &Path) -> Result<Self, ShardReadError> {
        let file = File::open(path).map_err(ShardReadError::Io)?;
        let file_len = file.metadata().map_err(ShardReadError::Io)?.len();
        // SAFETY: read-only mapping of a file we never write through this
        // handle; the writer is finished (header patched) before readers open.
        let mmap = unsafe { Mmap::map(&file).map_err(ShardReadError::Mmap)? };
        #[cfg(unix)]
        mmap.advise(Advice::Sequential)
            .map_err(ShardReadError::Advise)?;
        if mmap.len() != file_len as usize {
            return Err(ShardReadError::LengthMismatch {
                expected: file_len,
                got: mmap.len() as u64,
            });
        }
        let header = parse_header(&mmap)?;
        let expected = HEADER_LEN as u64 + header.rows * header.row_bytes as u64;
        if file_len != expected {
            return Err(ShardReadError::LengthMismatch {
                expected,
                got: file_len,
            });
        }
        // Payload SHA over the exact row region (exact-len already pins it).
        let payload = &mmap[HEADER_LEN..HEADER_LEN + header.rows as usize * header.row_bytes];
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest as _;
        hasher.update(payload);
        let sum = hasher.finalize();
        if sum.as_slice() != header.payload_sha256 {
            return Err(ShardReadError::ShaMismatch {
                expected: raw_hex_prefixed(&header.payload_sha256),
                got: raw_hex_prefixed(sum.as_slice()),
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            header,
        })
    }

    /// Source path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Verified header.
    pub fn header(&self) -> &ShardHeader {
        &self.header
    }

    /// File bucket `T`.
    pub fn t_bucket(&self) -> usize {
        self.header.t_bucket
    }

    /// Row count.
    pub fn rows(&self) -> u64 {
        self.header.rows
    }

    /// Compact row stride.
    pub fn row_bytes(&self) -> usize {
        self.header.row_bytes
    }

    /// Borrow compact row `index` (bounds-checked).
    pub fn row(&self, index: u64) -> Result<&[u8], ShardReadError> {
        if index >= self.header.rows {
            return Err(ShardReadError::Range {
                index,
                rows: self.header.rows,
            });
        }
        let base = HEADER_LEN + index as usize * self.header.row_bytes;
        Ok(&self.mmap[base..base + self.header.row_bytes])
    }

    /// Borrow the whole compact payload (verified bytes, row-major).
    pub fn payload(&self) -> &[u8] {
        let len = self.header.rows as usize * self.header.row_bytes;
        &self.mmap[HEADER_LEN..HEADER_LEN + len]
    }
}
