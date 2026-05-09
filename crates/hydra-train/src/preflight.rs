pub use hydra_train_runtime::preflight::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ManifestCacheEntry {
    pub data_dir: std::path::PathBuf,
    pub train_fraction_bits: u32,
    #[serde(default)]
    pub include_source_patterns: Vec<String>,
    #[serde(default)]
    pub exclude_source_patterns: Vec<String>,
    pub manifest: crate::data::pipeline::DataManifest,
}
