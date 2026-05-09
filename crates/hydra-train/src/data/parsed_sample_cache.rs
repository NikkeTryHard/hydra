//! Compatibility re-exports for parsed-sample cache format.

pub use hydra_sample_cache::*;

use crate::data::mjai_loader::MjaiGame;

/// Writes a parsed-sample cache from the legacy hydra-train `MjaiGame` type.
pub fn write_parsed_sample_cache(
    path: &std::path::Path,
    original_source_path: &std::path::Path,
    original_identity: &str,
    game: &MjaiGame,
) -> std::io::Result<()> {
    let cache_game = hydra_sample_cache::ParsedSampleCacheGame {
        samples: game.samples.clone(),
        final_scores: game.final_scores,
    };
    hydra_sample_cache::write_parsed_sample_cache(
        path,
        original_source_path,
        original_identity,
        &cache_game,
    )
}

/// Loads a parsed-sample cache into the legacy hydra-train `MjaiGame` type.
pub fn load_parsed_sample_cache(path: &std::path::Path) -> std::io::Result<ParsedSampleCacheFile> {
    let cache = hydra_sample_cache::load_parsed_sample_cache(path)?;
    Ok(ParsedSampleCacheFile {
        metadata: cache.metadata,
        game: MjaiGame {
            samples: cache.game.samples,
            final_scores: cache.game.final_scores,
        },
    })
}

/// Legacy parsed-sample cache file shape using `hydra_train::data::mjai_loader::MjaiGame`.
pub struct ParsedSampleCacheFile {
    /// Cache metadata header.
    pub metadata: hydra_sample_cache::ParsedSampleCacheMetadata,
    /// Parsed game payload.
    pub game: MjaiGame,
}
