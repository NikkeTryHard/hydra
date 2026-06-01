use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::{DISCOVERY_INDEX_VERSION, DataSource, DiscoveryManifest, DiscoverySummary};

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
