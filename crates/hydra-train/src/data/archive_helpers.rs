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
                || name.ends_with(".mjai.json")
                || name.ends_with(".mjai.json.gz")
    )
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::io;
    use std::os::unix::ffi::OsStringExt;
    use std::path::Path;

    use super::{
        compact_error_message, compact_identity, identity_for_archive_entry, is_mjai_archive_entry,
        is_tar_zst_file,
    };

    #[test]
    fn compact_error_message_reduces_to_short_reason() {
        let raw = format!(
            "replay observation failed:\n  Replay desync:\n    phase: WaitAct\n    drawn: Some(128)\n    {}",
            "extra ".repeat(64)
        );
        let compact = compact_error_message(&raw);
        assert_eq!(compact, "replay desync");
    }

    #[test]
    fn compact_identity_uses_file_name_only() {
        let identity = "majsoul-jade-mjai-2021.tar.zst/./210614_44a21457_86ce_4215_9ac2_aeb845f15521.mjai.json";
        assert_eq!(
            compact_identity(identity),
            "210614_44a21457_86ce_4215_9ac2_aeb845f15521.mjai.json"
        );
    }

    #[test]
    fn identity_for_archive_entry_uses_archive_file_name() {
        let archive = identity_for_archive_entry(
            Path::new("/tmp/archive.tar.zst"),
            Path::new("nested/game.json"),
        )
        .expect("archive identity should build");
        assert_eq!(archive, "archive.tar.zst/nested/game.json");
    }

    #[test]
    fn identity_for_archive_entry_rejects_invalid_archive_name() {
        let bad_archive = OsString::from_vec(vec![0xFF]);
        let err = identity_for_archive_entry(Path::new(&bad_archive), Path::new("game.json"))
            .expect_err("invalid utf-8 archive names should fail");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn path_classifiers_match_expected_suffix_rules() {
        assert!(is_tar_zst_file(Path::new("dataset.tar.zst")));
        assert!(is_tar_zst_file(Path::new("dataset.tar-0001.zst")));
        assert!(!is_tar_zst_file(Path::new("dataset.zip")));

        assert!(is_mjai_archive_entry(Path::new("round.mjai.json")));
        assert!(is_mjai_archive_entry(Path::new("round.mjai.json.gz")));
        assert!(is_mjai_archive_entry(Path::new("round.json")));
        assert!(!is_mjai_archive_entry(Path::new("round.txt")));
        assert!(!is_mjai_archive_entry(Path::new("round.mjai")));
        assert!(!is_mjai_archive_entry(Path::new("round.log")));
        assert!(!is_mjai_archive_entry(Path::new("round.json.zst")));
    }
}
