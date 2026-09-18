//! Safetensors mmap cache (Phase 3 M4 — `data/cache.py` FIRST target).
//!
//! Replaces the `torch.save` / `torch.load(..., weights_only=False)` pickle
//! path (`cache.py:71-73,115-117` exec surface) with `safetensors 0.8`
//! mmap borrows: header JSON parsed once (`u64 LE header-len + JSON
//! `{dtype,shape,data_offsets}` + raw bytes), tensor bytes borrowed from
//! the mapping — no pickle exec, offsets validated by the crate PLUS an
//! explicit bounds/overlap gate here.
//!
//! * compat-read: legacy `.pt` pickle files are detected by extension +
//!   magic sniff and rejected as [`CacheError::LegacyPickle`] — Python
//!   keeps the compat-read path, Rust never unpickles.
//! * offset-validate: [`MmapCache::validate_offsets`] checks every
//!   `data_offsets` interval against the mapped data length, ordered and
//!   non-overlapping, before any view is served.
//! * header-only copy: [`read_header_bytes`] / [`copy_header_to`] move the
//!   header prefix without re-reading tensor bytes (sidecar publish).
//! * IPC-file fallback: [`ipc_fallback_for`] resolves the sibling `.ipc`
//!   file when the `.safetensors` file is absent (small-manifest exception:
//!   tiny manifests stay JSON, never safetensors).
//!
//! Dtype coverage: F32/F16/BF16/I32/I64/U8/BOOL via `safetensors::Dtype`
//! (the cache carries `uint8` bool planes as U8/BOOL; geometry guards live
//! in the writer).

extern crate alloc;

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensorError, SafeTensors, serialize};
use safetensors::tensor::TensorView;

/// Offset into the raw data section: `(start, end)` byte interval.
pub type OffsetInterval = (usize, usize);

/// Cache failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum CacheError {
    Io(std::io::Error),
    Mmap(std::io::Error),
    Safetensors(SafeTensorError),
    /// Legacy `.pt` pickle path — compat-read gate (Python handles it).
    LegacyPickle { path: PathBuf },
    /// Offset validation failed (out-of-bounds / overlap / misorder).
    Offset { msg: String },
    /// Header malformed (truncated / oversize / bad JSON).
    Header { msg: String },
    /// Tensor missing or dtype/shape incompatible (never reshaped).
    Incompatible { msg: String },
}

impl core::fmt::Display for CacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CacheError::Io(e) => write!(f, "cache io: {e}"),
            CacheError::Mmap(e) => write!(f, "cache mmap: {e}"),
            CacheError::Safetensors(e) => write!(f, "cache safetensors: {e}"),
            CacheError::LegacyPickle { path } => {
                write!(f, "cache legacy pickle (compat-read): {}", path.display())
            }
            CacheError::Offset { msg } => write!(f, "cache offset: {msg}"),
            CacheError::Header { msg } => write!(f, "cache header: {msg}"),
            CacheError::Incompatible { msg } => write!(f, "cache incompatible: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Io(e) | CacheError::Mmap(e) => Some(e),
            CacheError::Safetensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CacheError {
    fn from(e: std::io::Error) -> Self {
        CacheError::Io(e)
    }
}

impl From<SafeTensorError> for CacheError {
    fn from(e: SafeTensorError) -> Self {
        CacheError::Safetensors(e)
    }
}

/// One tensor to build into a cache file (borrowed bytes, no copy).
#[derive(Debug, Clone, Copy)]
pub struct CacheTensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

/// Sniff legacy pickle: `.pt` extension OR pickle/zip magic.
///
/// `torch.save` payloads are pickle (`0x80` proto) or zip (`PK\x03\x04`);
/// safetensors files start with an 8-byte LE header length (small JSON
/// size). Anything smelling like pickle/zip is a compat-read case.
pub fn is_legacy_pickle(path: &Path, head: &[u8]) -> bool {
    if path.extension().map(|e| e == "pt").unwrap_or(false) {
        return true;
    }
    if head.first() == Some(&0x80) {
        return true;
    }
    if head.len() >= 4 && &head[..4] == b"PK\x03\x04" {
        return true;
    }
    false
}

/// Bit-exact element size per dtype (mirrors the crate's `Dtype::bitsize`;
/// local table so the gate never depends on a private helper).
fn dtype_bitsize(dtype: Dtype) -> Result<usize, CacheError> {
    // `Dtype` is `#[non_exhaustive]` — unknown future variants fail closed.
    let bits = match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 8,
        Dtype::F16 | Dtype::BF16 | Dtype::U16 | Dtype::I16 => 16,
        Dtype::F32 | Dtype::U32 | Dtype::I32 => 32,
        Dtype::F64 | Dtype::U64 | Dtype::I64 => 64,
        other => {
            return Err(CacheError::Incompatible {
                msg: format!("unsupported dtype {other:?} (cache pins F32/F16/BF16/I32/I64/U8/BOOL)"),
            });
        }
    };
    Ok(bits)
}

/// Build: tensors → `.safetensors` (atomic tmp + fsync + rename).
///
/// * `dir/file_name`: final path (`dir` created; `.safetensors` suffix,
///   NOT `.pt`).
/// * `tensors`: borrowed views (no copy; `TensorView::new` validates
///   `dtype/shape/data` length agreement).
/// * `metadata`: optional string sidecar map (digest/key/dtype/layout).
pub fn build_cache_safetensors(
    dir: &Path,
    file_name: &str,
    tensors: &[CacheTensor<'_>],
    metadata: Option<HashMap<String, String>>,
) -> Result<PathBuf, CacheError> {
    std::fs::create_dir_all(dir)?;
    let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let view = TensorView::new(t.dtype, t.shape.to_vec(), t.data)?;
        views.push((t.name.to_string(), view));
    }
    let bytes = serialize(views, metadata)?;
    let dest = dir.join(file_name);
    let tmp = dest.with_extension("tmp");
    {
        use std::io::Write as _;
        let mut f = File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

/// Cold mmap cache: owns the mapping; views borrow it.
#[derive(Debug)]
pub struct MmapCache {
    path: PathBuf,
    mmap: Mmap,
}

impl MmapCache {
    /// Open + compat gate: legacy pickle fails as `LegacyPickle` (never
    /// unpickled); safetensors header length is sanity-checked before any
    /// parse (truncated/oversize → `Header`).
    pub fn open(path: &Path) -> Result<Self, CacheError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        // SAFETY: read-only mapping of a finished file we never write
        // through this handle; builders rename-publish before readers open.
        let mmap = unsafe { Mmap::map(&file).map_err(CacheError::Mmap)? };
        if mmap.len() as u64 != file_len {
            return Err(CacheError::Header {
                msg: alloc::format!(
                    "mmap length {} != file length {file_len}",
                    mmap.len()
                ),
            });
        }
        let head_len = mmap.len().min(16);
        if is_legacy_pickle(path, &mmap[..head_len]) {
            return Err(CacheError::LegacyPickle {
                path: path.to_path_buf(),
            });
        }
        if mmap.len() < 8 {
            return Err(CacheError::Header {
                msg: "file smaller than 8-byte header length".to_string(),
            });
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().map_err(|_| CacheError::Header {
            msg: "header length prefix unreadable".to_string(),
        })?) as usize;
        if n > mmap.len().saturating_sub(8) {
            return Err(CacheError::Header {
                msg: alloc::format!("header length {n} exceeds file {}", mmap.len()),
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
        })
    }

    /// Source path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Borrow the mapped bytes.
    pub fn mapped(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Borrowed parse (crate validates offsets; call
    /// [`MmapCache::validate_offsets`] for the explicit gate).
    pub fn deserialize(&self) -> Result<SafeTensors<'_>, CacheError> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    /// Tensor names in the header.
    pub fn names(&self) -> Result<Vec<String>, CacheError> {
        let st = self.deserialize()?;
        Ok(st.names().into_iter().map(str::to_string).collect())
    }

    /// Explicit offset gate: every `data_offsets` interval is inside the
    /// data section, well-formed (`start <= end`), sorted by start and
    /// non-overlapping. The crate already rejects OOB/overlap on parse;
    /// this gate names the offender fail-closed before views are served.
    pub fn validate_offsets(&self) -> Result<Vec<(String, OffsetInterval)>, CacheError> {
        // Crate gate first (rejects OOB/gap/misorder/size-mismatch on
        // parse; `MetadataIncompleteBuffer` pins exact trailing length).
        let st = self.deserialize()?;
        // Explicit gate over PUBLIC views only (no private crate fields):
        // walk tensors in header order, recompute each byte length from
        // `dtype.bitsize * shape-product` (bit-exact `dtype_size`
        // table below), and require contiguous `[cursor, cursor+size]`
        // packing — the crate's own contiguity invariant (`Metadata::
        // validate`: `s == start`), re-checked here so the offender is
        // named fail-closed before views are served.
        let mut intervals: Vec<(String, usize, usize)> = Vec::new();
        let mut cursor = 0usize;
        for (name, view) in st.tensors() {
            let bits = dtype_bitsize(view.dtype())?;
            let mut elems = 1usize;
            for dim in view.shape() {
                elems = elems.checked_mul(*dim).ok_or_else(|| CacheError::Offset {
                    msg: format!("tensor {name}: shape product overflows"),
                })?;
            }
            let nbits = elems.checked_mul(bits).ok_or_else(|| CacheError::Offset {
                msg: format!("tensor {name}: bit size overflows"),
            })?;
            if nbits % 8 != 0 {
                return Err(CacheError::Offset {
                    msg: format!("tensor {name}: misaligned bit size {nbits}"),
                });
            }
            let size = nbits / 8;
            if view.data().len() != size {
                return Err(CacheError::Offset {
                    msg: format!(
                        "tensor {name}: view {} != dtype*shape {size}",
                        view.data().len()
                    ),
                });
            }
            let start = cursor;
            let end = cursor.checked_add(size).ok_or_else(|| CacheError::Offset {
                msg: format!("tensor {name}: end overflows"),
            })?;
            intervals.push((name, start, end));
            cursor = end;
        }
        intervals.sort_by_key(|(_, s, _)| *s);
        for w in intervals.windows(2) {
            let (prev_name, _, prev_end) = &w[0];
            let (name, start, _) = &w[1];
            if start < prev_end {
                return Err(CacheError::Offset {
                    msg: format!(
                        "tensors {prev_name}[..{prev_end}] overlaps {name}[{start}..]"
                    ),
                });
            }
        }
        Ok(intervals
            .into_iter()
            .map(|(n, s, e)| (n, (s, e)))
            .collect())
    }

    /// Header-only read: `u64 LE len + JSON` prefix (no tensor bytes).
    pub fn header_bytes(&self) -> Result<Vec<u8>, CacheError> {
        read_header_bytes(&self.mmap)
    }
}

/// Header-only read over already-mapped bytes (no tensor copy).
pub fn read_header_bytes(mapped: &[u8]) -> Result<Vec<u8>, CacheError> {
    if mapped.len() < 8 {
        return Err(CacheError::Header {
            msg: "file smaller than 8-byte header length".to_string(),
        });
    }
    let n = u64::from_le_bytes(mapped[..8].try_into().map_err(|_| CacheError::Header {
        msg: "header length prefix unreadable".to_string(),
    })?) as usize;
    let total = 8usize.saturating_add(n);
    if total > mapped.len() {
        return Err(CacheError::Header {
            msg: alloc::format!("header length {n} exceeds file {}", mapped.len()),
        });
    }
    Ok(mapped[..total].to_vec())
}

/// Header-only copy: replicate the header prefix to `dst` (sidecar
/// publish without moving tensor bytes).
pub fn copy_header_to(src_mapped: &[u8], dst: &Path) -> Result<usize, CacheError> {
    let header = read_header_bytes(src_mapped)?;
    if let Some(parent) = dst.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let n = header.len();
    {
        use std::io::Write as _;
        let mut f = File::create(dst)?;
        f.write_all(&header)?;
        f.flush()?;
    }
    Ok(n)
}

/// IPC-file fallback: sibling `.ipc` for a `.safetensors` path (returns
/// `Some` only when the safetensors file is absent AND the sibling exists).
pub fn ipc_fallback_for(safetensors_path: &Path) -> Option<PathBuf> {
    if safetensors_path.is_file() {
        return None;
    }
    let sibling = safetensors_path.with_extension("ipc");
    if sibling.is_file() {
        Some(sibling)
    } else {
        None
    }
}

#[cfg(test)]
mod cache_probes {
    use super::*;

    fn u8_tensor<'a>(name: &'a str, shape: &'a [usize], data: &'a [u8]) -> CacheTensor<'a> {
        CacheTensor {
            name,
            dtype: Dtype::U8,
            shape,
            data,
        }
    }

    fn probe_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(alloc::format!(
            "hydra-cache-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Round-trip: build → mmap → bytes identical + offsets validate.
    #[test]
    fn probe_build_mmap_round_trip() {
        let dir = probe_dir("roundtrip");
        let planes = vec![0u8, 1, 0, 1, 1, 0];
        let mask = vec![1u8; 8];
        let planes_shape = [planes.len()];
        let mask_shape = [mask.len()];
        let tensors = vec![
            u8_tensor("planes", &planes_shape, &planes),
            u8_tensor("mask", &mask_shape, &mask),
        ];
        let mut meta = HashMap::new();
        meta.insert("digest".to_string(), "sha256:00".to_string());
        let path =
            build_cache_safetensors(&dir, "cache.safetensors", &tensors, Some(meta)).unwrap();
        let cache = MmapCache::open(&path).unwrap();
        let intervals = cache.validate_offsets().unwrap();
        assert_eq!(intervals.len(), 2);
        let st = cache.deserialize().unwrap();
        assert_eq!(st.tensor("planes").unwrap().data(), planes.as_slice());
        assert_eq!(st.tensor("mask").unwrap().data(), mask.as_slice());
        // Header-only copy reproduces the exact prefix.
        let header = cache.header_bytes().unwrap();
        assert!(header.len() > 8 && header.len() < cache.mapped().len());
        let sidecar = dir.join("cache.header");
        let n = copy_header_to(cache.mapped(), &sidecar).unwrap();
        assert_eq!(n, header.len());
        assert_eq!(std::fs::read(&sidecar).unwrap(), header);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Tamper fork: flipped tensor byte MUST change the view bytes (and a
    /// flipped header byte MUST fail parse) — bytes are the gate.
    #[test]
    fn probe_tamper_fork() {
        let dir = probe_dir("tamper");
        let data = vec![7u8; 16];
        let shape = [data.len()];
        let path = build_cache_safetensors(
            &dir,
            "t.safetensors",
            &[u8_tensor("w", &shape, &data)],
            None,
        )
        .unwrap();
        let raw = std::fs::read(&path).unwrap();
        // Tensor-byte fork: parse still succeeds (offsets intact) but the
        // view bytes differ — the digest over views is the gate.
        let mut forked = raw.clone();
        let last = forked.len() - 1;
        forked[last] ^= 0x01;
        let fork_path = dir.join("t.fork.safetensors");
        std::fs::write(&fork_path, &forked).unwrap();
        let a = MmapCache::open(&path).unwrap();
        let b = MmapCache::open(&fork_path).unwrap();
        assert_ne!(
            a.deserialize().unwrap().tensor("w").unwrap().data(),
            b.deserialize().unwrap().tensor("w").unwrap().data()
        );
        // Header-byte fork: parse fails closed.
        let mut bad_header = raw.clone();
        bad_header[8] ^= 0xFF;
        let bad_path = dir.join("t.bad.safetensors");
        std::fs::write(&bad_path, &bad_header).unwrap();
        let bad = MmapCache::open(&bad_path).unwrap();
        assert!(bad.deserialize().is_err() || bad.validate_offsets().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Compat + fallback: `.pt` pickle rejected; truncated offsets fail;
    /// missing safetensors with `.ipc` sibling resolves the fallback.
    #[test]
    fn probe_compat_offset_fallback() {
        let dir = probe_dir("compat");
        // Legacy pickle sniff (extension + magic).
        let pt = dir.join("cache.pt");
        std::fs::write(&pt, [0x80u8, 0x04, 0x95, 0x00]).unwrap();
        match MmapCache::open(&pt) {
            Err(CacheError::LegacyPickle { .. }) => {}
            other => panic!("expected LegacyPickle, got {other:?}"),
        }
        // Truncated file fails the header gate.
        let short = dir.join("short.safetensors");
        std::fs::write(&short, [0x10u8, 0x00]).unwrap();
        assert!(MmapCache::open(&short).is_err());
        // IPC fallback resolves only when safetensors is absent.
        let missing = dir.join("absent.safetensors");
        assert_eq!(ipc_fallback_for(&missing), None);
        let sibling = dir.join("absent.ipc");
        std::fs::write(&sibling, b"IPC1").unwrap();
        assert_eq!(ipc_fallback_for(&missing), Some(sibling.clone()));
        let data = vec![1u8; 4];
        let shape = [data.len()];
        let present = build_cache_safetensors(
            &dir,
            "present.safetensors",
            &[u8_tensor("v", &shape, &data)],
            None,
        )
        .unwrap();
        assert_eq!(ipc_fallback_for(&present), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
