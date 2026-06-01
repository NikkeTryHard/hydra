pub(crate) const PARSED_SAMPLE_CACHE_MAGIC_LEN: usize = 8;
pub(crate) const PARSED_SAMPLE_CACHE_MAGIC: &[u8; PARSED_SAMPLE_CACHE_MAGIC_LEN] = b"HPSCACHE";
pub(crate) const PARSED_SAMPLE_CACHE_VERSION: u32 = 1;
pub(crate) const FINAL_SCORE_COUNT: usize = 4;
pub(crate) const OPPONENT_COUNT: usize = 3;
pub(crate) const MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN: usize = 64 * 1024;
pub(crate) const MAX_PARSED_SAMPLE_CACHE_SAMPLES: u32 = 10_000;
