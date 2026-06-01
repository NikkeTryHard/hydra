//! Frozen compact BC shard ABI constants.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;

/// Opponent count encoded in spatial auxiliary targets.
pub const OPPONENT_COUNT: usize = 3;
/// Player count encoded in oracle targets.
pub const PLAYER_COUNT: usize = 4;
/// Tile count encoded per observation channel.
pub const TILE_COUNT: usize = 34;
/// Opponent-by-tile spatial target size.
pub const SPATIAL_TARGET_SIZE: usize = OPPONENT_COUNT * TILE_COUNT;

/// Compact BC shard magic bytes.
pub const BC_SHARD_MAGIC: [u8; 8] = *b"HYBCS3\0\0";
/// Obsolete dense BC shard magic bytes, kept only for hard-error detection.
pub const BC_DENSE_SHARD_MAGIC: [u8; 8] = *b"HYBCS2\0\0";
/// Error emitted when a dense shard is presented to the compact reader.
pub const DENSE_REBUILD_MESSAGE: &str = "dense BC shards are obsolete; rebuild from replay";
/// Compact BC shard binary version.
pub const BC_SHARD_VERSION: u32 = 3;
/// Compact BC shard manifest version.
pub const BC_SHARD_MANIFEST_VERSION: u32 = 3;
/// Compact record layout version inside shard headers.
pub const BC_SHARD_LAYOUT_VERSION: u32 = 1;
/// Compact-only manifest storage layout tag.
pub const STORAGE_LAYOUT_COMPACT: &str = "compact";
/// Compact BC shard header byte length.
pub const BC_SHARD_HEADER_SIZE: u32 = 80;

/// Feature flag enabling safety residual action targets.
pub const FLAG_SAFETY_RESIDUAL: u32 = 1 << 0;
/// Feature flag enabling ExIt action targets.
pub const FLAG_EXIT: u32 = 1 << 1;
/// Feature flag enabling delta-Q action targets.
pub const FLAG_DELTA_Q: u32 = 1 << 2;
/// Feature flag enabling belief-field targets.
pub const FLAG_BELIEF_FIELDS: u32 = 1 << 3;
/// Feature flag enabling mixture-weight targets.
pub const FLAG_MIXTURE_WEIGHTS: u32 = 1 << 4;
/// All compact feature flags supported by this binary.
pub const VALID_FEATURE_FLAGS: u32 =
    FLAG_SAFETY_RESIDUAL | FLAG_EXIT | FLAG_DELTA_Q | FLAG_BELIEF_FIELDS | FLAG_MIXTURE_WEIGHTS;

/// Dense-equivalent observation byte count, for reporting only.
pub const DENSE_OBS_F32_BYTES: usize = OBS_SIZE * 4;
/// Packed legal/action-mask bytes for 46 actions.
pub const PACKED_ACTION_MASK_BYTES: usize = HYDRA_ACTION_SPACE.div_ceil(8);
/// Packed legal-mask byte count per record.
pub const PACKED_LEGAL_MASK_BYTES: usize = PACKED_ACTION_MASK_BYTES;
/// Packed 34 tile-count bytes, using 3 bits per count.
pub const TILE34_COUNT_BYTES: usize = (TILE_COUNT * 3).div_ceil(8);
/// Packed 34-tile bitset byte count.
pub const TILE34_BITSET_BYTES: usize = TILE_COUNT.div_ceil(8);
/// Packed 102-bit spatial mask byte count.
pub const PACKED_SPATIAL_MASK_BYTES: usize = SPATIAL_TARGET_SIZE.div_ceil(8);
/// Encoded oracle-float byte count per record.
pub const ORACLE_FLOAT32_BYTES: usize = PLAYER_COUNT * 4;
/// Encoded oracle-presence byte count per record.
pub const ORACLE_MASK_BYTES: usize = 1;
/// Encoded opponent-next byte count per record.
pub const OPP_NEXT_BYTES: usize = OPPONENT_COUNT;
/// Encoded optional action-float byte count per target.
pub const OPTIONAL_ACTION_FLOAT32_BYTES: usize = HYDRA_ACTION_SPACE * 4;
/// Encoded optional packed action-mask byte count per target.
pub const OPTIONAL_ACTION_MASK_BYTES: usize = PACKED_ACTION_MASK_BYTES;
/// Encoded belief-field byte count per record when enabled.
pub const BELIEF_FIELDS_BYTES: usize = 16 * TILE_COUNT * 4;
/// Encoded mixture-weight byte count per record when enabled.
pub const MIXTURE_WEIGHTS_BYTES: usize = PLAYER_COUNT * 4;
/// Half-open observation channel range in the compact shard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObsChannelRange {
    pub start: usize,
    pub end: usize,
}

const fn obs_channel_count<const N: usize>(ranges: [ObsChannelRange; N]) -> usize {
    let mut idx = 0usize;
    let mut total = 0usize;
    while idx < N {
        total += ranges[idx].end - ranges[idx].start;
        idx += 1;
    }
    total
}
/// Baseline observation fact bytes for exact dense reconstruction of channels 0..85.
///
/// This section stores replay/encoder facts rather than broad dense channel tails:
/// tile counts, tile bitsets, compact metadata scalars, discard temporal indices,
/// and the six exact f32 safety float planes that are not yet integer fact-shaped.
pub const COMPACT_OBS_BASELINE_FACT_BYTES: usize = 1_675;
/// Search/belief scalar channels are absent from compact replay BC shards.
pub const OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_RANGES: [ObsChannelRange; 0] = [];
/// Search/belief and Hand-EV channels are absent from compact replay BC shards.
pub const OBS_ADVANCED_DENSE_CHANNEL_RANGES: [ObsChannelRange; 0] = [];
/// Number of repeated-scalar advanced observation channels.
pub const OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_COUNT: usize =
    obs_channel_count(OBS_ADVANCED_SCALAR_REPEATED_CHANNEL_RANGES);
/// Number of dense advanced observation channels.
pub const OBS_ADVANCED_DENSE_CHANNEL_COUNT: usize =
    obs_channel_count(OBS_ADVANCED_DENSE_CHANNEL_RANGES);
/// Repeated scalar advanced observation bytes stored per record.
pub const COMPACT_OBS_SCALAR_REPEATED_BYTES: usize = 0;
/// Dense advanced observation bytes stored per record.
pub const COMPACT_OBS_DENSE_BYTES: usize = 0;
/// Compact observation bytes rebuilt losslessly by the reader.
pub const COMPACT_OBS_BYTES: usize = COMPACT_OBS_BASELINE_FACT_BYTES;

/// Compact base record byte size without optional action/search targets.
pub const BC_BASE_RECORD_SIZE: u32 = (COMPACT_OBS_BYTES
    + 1
    + PACKED_LEGAL_MASK_BYTES
    + 4
    + 1
    + ORACLE_FLOAT32_BYTES
    + ORACLE_MASK_BYTES
    + 1
    + OPP_NEXT_BYTES
    + PACKED_SPATIAL_MASK_BYTES
    + PACKED_SPATIAL_MASK_BYTES) as u32;
/// Compact maximum record byte size with every optional target.
pub const BC_RECORD_SIZE_WITH_ALL_OPTIONALS: u32 = BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32) * 3
    + BELIEF_FIELDS_BYTES as u32
    + MIXTURE_WEIGHTS_BYTES as u32;

/// Backward-compatible alias for dense-equivalent observation byte reporting.
pub const OBS_F32_BYTES: usize = DENSE_OBS_F32_BYTES;
/// Backward-compatible alias for packed legal-mask bytes.
pub const LEGAL_MASK_BYTES: usize = PACKED_LEGAL_MASK_BYTES;
/// Backward-compatible alias for packed tenpai bytes.
pub const TENPAI_BYTES: usize = 1;
/// Backward-compatible alias for packed danger bytes.
pub const DANGER_BYTES: usize = PACKED_SPATIAL_MASK_BYTES;
/// Backward-compatible alias for packed danger-mask bytes.
pub const DANGER_MASK_BYTES: usize = PACKED_SPATIAL_MASK_BYTES;

/// Returns an error when `flags` contains unsupported compact sections.
pub fn validate_feature_flags(flags: u32) -> Result<(), String> {
    let unknown = flags & !VALID_FEATURE_FLAGS;
    if unknown == 0 {
        Ok(())
    } else {
        Err(format!(
            "BC shard feature_flags contain unsupported bits {unknown:#x}"
        ))
    }
}

/// Returns compact record size for supported feature flags.
pub fn checked_compact_record_size(flags: u32) -> Result<u32, String> {
    validate_feature_flags(flags)?;
    let mut size = BC_BASE_RECORD_SIZE;
    for flag in [FLAG_SAFETY_RESIDUAL, FLAG_EXIT, FLAG_DELTA_Q] {
        if flags & flag != 0 {
            size = size
                .checked_add(OPTIONAL_ACTION_FLOAT32_BYTES as u32)
                .and_then(|value| value.checked_add(OPTIONAL_ACTION_MASK_BYTES as u32))
                .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
        }
    }
    if flags & FLAG_BELIEF_FIELDS != 0 {
        size = size
            .checked_add(BELIEF_FIELDS_BYTES as u32)
            .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        size = size
            .checked_add(MIXTURE_WEIGHTS_BYTES as u32)
            .ok_or_else(|| "BC shard compact record size overflow".to_string())?;
    }
    Ok(size)
}

/// Returns total record payload bytes, rejecting overflow before allocation or I/O.
pub fn checked_record_bytes(sample_count: u64, record_size: u32) -> Result<u64, String> {
    sample_count
        .checked_mul(u64::from(record_size))
        .ok_or_else(|| "BC shard record byte count overflow".to_string())
}

/// Returns encoded record buffer length for an in-memory sample slice.
pub fn checked_encoded_record_len(sample_count: usize, record_size: u32) -> Result<usize, String> {
    let byte_count = checked_record_bytes(sample_count as u64, record_size)?;
    usize::try_from(byte_count).map_err(|_| "BC shard record byte count overflow".to_string())
}
