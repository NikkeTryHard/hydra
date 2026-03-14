use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExportCli {
    pub(crate) config_path: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OptionalCarrierMode {
    BaselinePlusSafetyResidual,
    IncludeFutureCarriers,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportConfig {
    pub(crate) data_dir: PathBuf,
    pub(crate) output_dir: PathBuf,
    #[serde(default = "default_train_fraction")]
    pub(crate) train_fraction: f32,
    #[serde(default = "default_seed")]
    pub(crate) seed: u64,
    #[serde(default = "default_shard_sample_cap")]
    pub(crate) max_samples_per_shard: usize,
    #[serde(default = "default_shard_game_cap")]
    pub(crate) max_games_per_shard: usize,
    #[serde(default = "default_buffer_games")]
    pub(crate) buffer_games: usize,
    #[serde(default = "default_buffer_samples")]
    pub(crate) buffer_samples: usize,
    #[serde(default = "default_archive_queue_bound")]
    pub(crate) archive_queue_bound: usize,
    #[serde(default = "default_optional_carriers")]
    pub(crate) optional_carriers: OptionalCarrierMode,
}

pub(crate) fn parse_args(args: impl IntoIterator<Item = String>) -> Result<ExportCli, String> {
    let args: Vec<String> = args.into_iter().collect();
    if args.len() != 2 {
        return Err(format!(
            "Usage: {} <export_config.yaml>",
            args.first()
                .map(String::as_str)
                .unwrap_or("export_bc_phase0")
        ));
    }
    Ok(ExportCli {
        config_path: PathBuf::from(&args[1]),
    })
}

pub(crate) fn read_config(path: &Path) -> Result<ExportConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => serde_yaml::from_str(&raw)
            .map_err(|err| format!("failed to parse YAML config {}: {err}", path.display())),
        Some("json") => serde_json::from_str(&raw)
            .map_err(|err| format!("failed to parse JSON config {}: {err}", path.display())),
        _ => Err(format!(
            "unsupported config format for {} (expected .yaml, .yml, or .json)",
            path.display()
        )),
    }
}

pub(crate) fn default_train_fraction() -> f32 {
    0.9
}

pub(crate) fn default_seed() -> u64 {
    0
}

pub(crate) fn default_shard_sample_cap() -> usize {
    131_072
}

pub(crate) fn default_shard_game_cap() -> usize {
    4_096
}

pub(crate) fn default_buffer_games() -> usize {
    50_000
}

pub(crate) fn default_buffer_samples() -> usize {
    32_768
}

pub(crate) fn default_archive_queue_bound() -> usize {
    128
}

pub(crate) fn default_optional_carriers() -> OptionalCarrierMode {
    OptionalCarrierMode::BaselinePlusSafetyResidual
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_single_config_path() {
        let cli = parse_args(["export_bc_phase0".to_string(), "config.yaml".to_string()])
            .expect("cli should parse");
        assert_eq!(cli.config_path, PathBuf::from("config.yaml"));
    }

    #[test]
    fn parse_args_rejects_missing_config() {
        let err = parse_args(["export_bc_phase0".to_string()]).expect_err("should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn read_config_applies_defaults() {
        let path = std::env::temp_dir().join(format!(
            "hydra_phase0_export_config_{}.yaml",
            std::process::id()
        ));
        let yaml = r#"
data_dir: /tmp/data
output_dir: /tmp/out
"#;
        fs::write(&path, yaml).expect("write config");
        let config = read_config(&path).expect("read config");
        assert_eq!(config.data_dir, PathBuf::from("/tmp/data"));
        assert_eq!(config.output_dir, PathBuf::from("/tmp/out"));
        assert!((config.train_fraction - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.seed, 0);
        assert_eq!(config.max_samples_per_shard, 131_072);
        assert_eq!(config.max_games_per_shard, 4_096);
        assert_eq!(config.buffer_games, 50_000);
        assert_eq!(config.buffer_samples, 32_768);
        assert_eq!(config.archive_queue_bound, 128);
        assert_eq!(
            config.optional_carriers,
            OptionalCarrierMode::BaselinePlusSafetyResidual
        );
        fs::remove_file(path).ok();
    }
}
