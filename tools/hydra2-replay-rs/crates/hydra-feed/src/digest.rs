//! SHA-256 identity digests — `sha256:<64 lowercase hex>` on ALL identity paths.
//!
//! Rust owner of `src/hydra2/artifacts/digest.py` (`sha256_digest` in-mem +
//! `sha256_file` 1 MiB chunked second path). The two paths MUST agree on
//! identical input (BUILD WP-02A exit; KAT-1 gates it). BLAKE3 is barred on
//! identity (endstate §0.3); transient non-identity keys with an explicit
//! allowlist are the ONLY exception, and they never live here.
//!
//! - `sha2 0.11` ONLY on identity (`Sha256::digest` / `new` + `update` +
//!   `finalize`). Lowercase hex. Never re-hash hex.
//! - `of_canonical` hashes `crate::canon::canonical_bytes` output — canon +
//!   hash fused at the boundary, exactly like Python `of_canonical`.
//! - `Digest` is NOT `Write` (wave2 §2.4): streaming goes `to_vec`-then-
//!   `update`, never a second printer.

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::canon::{CanonError, canonical_bytes};

/// 1 MiB chunk — matches `digest.py:32` `_CHUNK_SIZE` (identity file path).
/// Chunk-size invariance is the gate: any chunking yields the same digest;
/// 1 MiB is the pinned identity Path B (wave2 §5.4's 64 KiB is the zstd
/// streaming standard, NOT the digest file path).
pub const CHUNK: usize = 1 << 20;

/// Path A: in-memory one-shot over raw bytes → `sha256:<hex>`.
///
/// Infallible by construction (SHA-256 has no failure mode); lowercase hex
/// with the `sha256:` prefix attached at the boundary.
pub fn sha256_hex(data: &[u8]) -> String {
    let sum = Sha256::digest(data);
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for b in sum {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Canon+hash fused: digest over RFC 8785 canonical bytes of `value`
/// (mirrors Python `of_canonical = sha256_digest(canonical_bytes(value))`).
///
/// Fallible ONLY via the canon step (non-string map keys); the hash itself
/// cannot fail.
pub fn of_canonical<T: Serialize + ?Sized>(value: &T) -> Result<String, CanonError> {
    let bytes = canonical_bytes(value, "digest:of_canonical")?;
    Ok(sha256_hex(&bytes))
}

/// Path B: chunked streaming second path over a file (MUST agree with Path A
/// on identical input). 1 MiB chunks per [`CHUNK`].
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read as _;
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let sum = h.finalize();
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for b in sum {
        out.push_str(&format!("{b:02x}"));
    }
    Ok(out)
}

/// `Write`-for-hash adapter (wave2 §2.4, 10-line form): lets callers stream
/// canonical bytes straight into the hasher when they already hold a writer
/// path, without a second printer. `to_vec`-then-`update` stays the default.
#[derive(Debug, Clone, Default)]
pub struct HashWriter {
    inner: Sha256,
}

impl HashWriter {
    pub fn new() -> Self {
        HashWriter {
            inner: Sha256::new(),
        }
    }

    pub fn finish(self) -> String {
        let sum = self.inner.finalize();
        let mut out = String::with_capacity(7 + 64);
        out.push_str("sha256:");
        for b in sum {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

impl std::io::Write for HashWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_WORLD_SHA: &str =
        "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    /// KAT-1 sha-hello, BOTH paths: in-mem one-shot AND 1 MiB-chunked file
    /// agree on `b"hello world"`, matching the `sha2` doc golden
    /// (`b94d27…efcde9`).
    #[test]
    fn kat1_sha_hello_both_paths_agree() {
        // Path A: in-memory.
        assert_eq!(sha256_hex(b"hello world"), HELLO_WORLD_SHA);
        // Incremental API agrees with one-shot (sha2 doc golden pair).
        let mut h = Sha256::new();
        h.update(b"hello ");
        h.update(b"world");
        let sum = h.finalize();
        let mut text = String::from("sha256:");
        for b in sum {
            text.push_str(&format!("{b:02x}"));
        }
        assert_eq!(text, HELLO_WORLD_SHA);
        // Path B: file with 1 MiB chunks agrees with Path A.
        let dir = std::env::temp_dir().join("hydra-feed-kat1");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("hello.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let filed = sha256_file(&path).unwrap();
        assert_eq!(filed, HELLO_WORLD_SHA);
        assert_eq!(filed, sha256_hex(b"hello world"));
        let _ = std::fs::remove_file(&path);
        // Never re-hash hex: hashing the TEXT differs from the golden.
        assert_ne!(sha256_hex(HELLO_WORLD_SHA.as_bytes()), HELLO_WORLD_SHA);
    }

    /// Path-B chunk invariance + `of_canonical` seal: a payload larger than
    /// one chunk hashes identically via file and memory; `of_canonical`
    /// fuses canon+hash.
    #[test]
    fn digest_file_chunk_invariance_and_of_canonical() {
        let payload: Vec<u8> = (0..(CHUNK + 123))
            .map(|i| (i % 251) as u8)
            .collect();
        let mem = sha256_hex(&payload);
        let dir = std::env::temp_dir().join("hydra-feed-kat1b");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("big.bin");
        std::fs::write(&path, &payload).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), mem);
        let _ = std::fs::remove_file(&path);

        // of_canonical == sha256_hex(canonical_bytes).
        let v = serde_json::json!({"b": 1, "a": [true, null]});
        let fused = of_canonical(&v).unwrap();
        let bytes = crate::canon::canonical_bytes(&v, "kat1:fused").unwrap();
        assert_eq!(fused, sha256_hex(&bytes));

        // HashWriter adapter agrees with one-shot.
        let mut w = HashWriter::new();
        use std::io::Write as _;
        w.write_all(&bytes).unwrap();
        assert_eq!(w.finish(), fused);
    }
}
