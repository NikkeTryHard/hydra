use std::io;
use std::path::Path;

pub fn compact_identity(identity: &str) -> &str {
    identity.rsplit('/').next().unwrap_or(identity)
}

pub fn compact_error_message(err: &dyn std::fmt::Display) -> &'static str {
    let raw = err.to_string();
    if raw.contains("Replay desync") {
        "replay desync"
    } else if raw.contains("replay observation failed") {
        "replay observation failed"
    } else if raw.contains("replay action conversion failed") {
        "replay action conversion failed"
    } else if raw.contains("hydra action mapping failed") {
        "hydra action mapping failed"
    } else if raw.contains("failed to parse MJAI events") {
        "invalid mjai events"
    } else if raw.contains("failed to load MJAI events") {
        "failed to load mjai events"
    } else if raw.contains("failed to inspect MJAI stream") {
        "failed to inspect mjai stream"
    } else {
        "load error"
    }
}

pub fn identity_for_archive_entry(archive_path: &Path, entry_path: &Path) -> io::Result<String> {
    let archive_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid archive name {}", archive_path.display()),
            )
        })?;
    Ok(format!("{archive_name}/{}", entry_path.display()))
}

pub fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
    )
}

pub fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".json")
                || name.ends_with(".json.gz")
                || name.ends_with(".json.zst")
                || name.ends_with(".mjai.json")
                || name.ends_with(".mjai.json.gz")
                || name.ends_with(".mjai.json.zst")
    )
}

#[cfg(test)]
mod tests;
