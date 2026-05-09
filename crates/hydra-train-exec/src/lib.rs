//! Training execution support modules migrated from the train binary.

#![deny(missing_docs)]

/// Runtime advisory formatting and selection helpers.
pub mod advisory;
/// Probe result summary helpers shared by execution support modules.
pub mod probe_summary;
/// Resume state contracts and helpers.
pub mod resume;

/// Artifact support intentionally remains in the train binary for now because it
/// still owns validation and promotion artifact types that have not moved yet.
pub mod artifacts {
    use std::fs;
    use std::path::Path;

    /// Atomically writes text by writing a timestamped temporary sibling then renaming it.
    pub fn atomic_write_text(path: &Path, contents: &str, label: &str) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create {label} dir {}: {err}", parent.display())
            })?;
        }
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("tmp");
        let tmp_path = path.with_extension(format!(
            "{extension}.tmp-{}-{}",
            std::process::id(),
            crate::resume::current_timestamp_s()
        ));
        fs::write(&tmp_path, contents).map_err(|err| {
            format!(
                "failed to write temporary {label} {}: {err}",
                tmp_path.display()
            )
        })?;
        fs::rename(&tmp_path, path).map_err(|err| {
            let _ = fs::remove_file(&tmp_path);
            format!(
                "failed to finalize {label} {} from {}: {err}",
                path.display(),
                tmp_path.display()
            )
        })
    }
}

/// Marker for the train execution boundary.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct TrainExec;

impl TrainExec {
    /// Creates a train execution boundary marker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
