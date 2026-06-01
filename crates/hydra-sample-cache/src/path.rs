use std::io;
use std::path::Path;

use crate::PARSED_SAMPLE_CACHE_EXTENSION;

/// Returns whether a path names a parsed-sample cache file.
pub fn is_parsed_sample_cache_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(PARSED_SAMPLE_CACHE_EXTENSION)
    )
}

pub fn parsed_sample_cache_file_name(source_path: &Path) -> io::Result<String> {
    let file_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "source path does not have a valid UTF-8 filename: {}",
                    source_path.display()
                ),
            )
        })?;
    let stem = file_name
        .strip_suffix(".json.gz")
        .or_else(|| file_name.strip_suffix(".json"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected loose MJAI file ending in .json or .json.gz, got {}",
                    source_path.display()
                ),
            )
        })?;
    Ok(format!("{stem}{PARSED_SAMPLE_CACHE_EXTENSION}"))
}
