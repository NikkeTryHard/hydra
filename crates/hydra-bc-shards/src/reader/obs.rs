//! Compact observation reconstruction.

use crate::compact::{decode_counts_threshold_planes, unpack_binary_mask_into, unpack_tile_counts};
use crate::manifest::{
    COMPACT_OBS_BASELINE_FACT_BYTES, TILE_COUNT, TILE34_BITSET_BYTES, TILE34_COUNT_BYTES,
};

use super::header::{read_f32_array, read_u32_le, take, take_array};

pub(super) fn decode_compact_obs(
    facts: &[u8],
    scalars: &[u8],
    dense: &[u8],
    dst: &mut [f32],
) -> Result<(), String> {
    if !scalars.is_empty() || !dense.is_empty() {
        return Err("BC shard compact observation advanced sections must be empty".to_string());
    }
    dst.fill(0.0);
    decode_baseline_obs_facts(facts, dst)
}
fn decode_baseline_obs_facts(bytes: &[u8], dst: &mut [f32]) -> Result<(), String> {
    if bytes.len() != COMPACT_OBS_BASELINE_FACT_BYTES {
        return Err("BC shard compact observation fact section has invalid length".to_string());
    }
    let mut cursor = 0usize;
    decode_tile_counts(
        take_array::<TILE34_COUNT_BYTES>(bytes, &mut cursor)?,
        dst,
        0,
    )?;
    decode_tile_counts(
        take_array::<TILE34_COUNT_BYTES>(bytes, &mut cursor)?,
        dst,
        4,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        8,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        9,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        10,
    )?;
    decode_discard_facts(bytes, &mut cursor, dst)?;
    decode_meld_facts(bytes, &mut cursor, dst)?;
    decode_dora_facts(take(bytes, &mut cursor, TILE_COUNT)?, dst)?;
    decode_aka_facts(take(bytes, &mut cursor, 1)?[0], dst);
    decode_metadata_facts(bytes, &mut cursor, dst)?;
    decode_safety_facts(bytes, &mut cursor, dst)?;
    debug_assert_eq!(cursor, COMPACT_OBS_BASELINE_FACT_BYTES);
    Ok(())
}

fn decode_tile_counts(
    bytes: &[u8; TILE34_COUNT_BYTES],
    dst: &mut [f32],
    channel_start: usize,
) -> Result<(), String> {
    let mut counts = [0u8; TILE_COUNT];
    unpack_tile_counts(bytes, &mut counts).map_err(|err| err.to_string())?;
    decode_counts_threshold_planes(&counts, dst, channel_start);
    Ok(())
}

fn decode_tile_bitset(
    bytes: &[u8; TILE34_BITSET_BYTES],
    dst: &mut [f32],
    channel: usize,
) -> Result<(), String> {
    let start = channel * TILE_COUNT;
    unpack_binary_mask_into(bytes, TILE_COUNT, &mut dst[start..start + TILE_COUNT])
        .map_err(|err| err.to_string())
}

fn decode_channel_bitsets(
    bytes: &[u8],
    cursor: &mut usize,
    dst: &mut [f32],
    channel_start: usize,
    channel_count: usize,
) -> Result<(), String> {
    for channel in channel_start..channel_start + channel_count {
        decode_tile_bitset(
            take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
            dst,
            channel,
        )?;
    }
    Ok(())
}

fn decode_discard_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    for player in 0..4usize {
        let base = 11 + player * 3;
        decode_tile_bitset(take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?, dst, base)?;
        decode_tile_bitset(
            take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
            dst,
            base + 1,
        )?;
        let row = (base + 2) * TILE_COUNT;
        for tile in 0..TILE_COUNT {
            let idx = read_u32_le(take(bytes, cursor, 4)?);
            dst[row + tile] = temporal_value(idx)?;
        }
    }
    Ok(())
}

fn temporal_value(index: u32) -> Result<f32, String> {
    #[allow(
        clippy::excessive_precision,
        reason = "table entries preserve encoder discard decay exactly"
    )]
    const DISCARD_EXP_TABLE: [f32; 31] = [
        1.0,
        0.818_730_8,
        0.670_320_0,
        0.548_811_6,
        0.449_329_0,
        0.367_879_5,
        0.301_194_2,
        0.246_597_0,
        0.201_896_5,
        0.165_298_9,
        0.135_335_3,
        0.110_803_2,
        0.090_717_96,
        0.074_273_58,
        0.060_810_06,
        0.049_787_07,
        0.040_762_20,
        0.033_373_27,
        0.027_323_72,
        0.022_370_77,
        0.018_315_64,
        0.014_995_58,
        0.012_277_34,
        0.010_051_84,
        0.008_229_747,
        0.006_737_947,
        0.005_516_564,
        0.004_516_581,
        0.003_697_864,
        0.003_027_555,
        0.002_478_752,
    ];
    if index == u32::MAX {
        return Ok(0.0);
    }
    DISCARD_EXP_TABLE
        .get(index as usize)
        .copied()
        .ok_or_else(|| format!("BC shard discard temporal index {index} out of range"))
}

fn decode_meld_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_channel_bitsets(bytes, cursor, dst, 23, 12)
}

fn decode_dora_facts(bytes: &[u8], dst: &mut [f32]) -> Result<(), String> {
    for (tile, &count) in bytes.iter().enumerate() {
        if count > 5 {
            return Err(format!("BC shard dora count {count} out of range"));
        }
        for threshold in 0..5usize {
            if count as usize > threshold {
                dst[(35 + threshold) * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    Ok(())
}

fn decode_aka_facts(flags: u8, dst: &mut [f32]) {
    for suit in 0..3usize {
        if (flags & (1u8 << suit)) != 0 {
            dst[(40 + suit) * TILE_COUNT..(41 + suit) * TILE_COUNT].fill(1.0);
        }
    }
}

fn decode_metadata_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_repeated_bool_channels(
        take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
        dst,
        43,
        4,
    )?;
    for channel in 47..55usize {
        let value = read_f32_array::<1>(take(bytes, cursor, 4)?)[0];
        dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(value);
    }
    decode_repeated_bool_channels(
        take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
        dst,
        55,
        4,
    )?;
    for channel in 59..62usize {
        let value = read_f32_array::<1>(take(bytes, cursor, 4)?)[0];
        dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(value);
    }
    Ok(())
}

fn decode_repeated_bool_channels(
    bytes: &[u8; TILE34_BITSET_BYTES],
    dst: &mut [f32],
    channel_start: usize,
    channel_count: usize,
) -> Result<(), String> {
    let mut values = [0.0f32; TILE_COUNT];
    unpack_binary_mask_into(bytes, TILE_COUNT, &mut values).map_err(|err| err.to_string())?;
    for (channel_offset, &value) in values.iter().enumerate().take(channel_count) {
        if value != 0.0 {
            let channel = channel_start + channel_offset;
            dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(1.0);
        }
    }
    Ok(())
}

fn decode_safety_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_channel_bitsets(bytes, cursor, dst, 62, 9)?;
    for channel in 71..74usize {
        decode_dense_channel(take(bytes, cursor, TILE_COUNT * 4)?, dst, channel)?;
    }
    decode_channel_bitsets(bytes, cursor, dst, 74, 3)?;
    for channel in 77..80usize {
        decode_dense_channel(take(bytes, cursor, TILE_COUNT * 4)?, dst, channel)?;
    }
    decode_channel_bitsets(bytes, cursor, dst, 80, 5)
}

fn decode_dense_channel(bytes: &[u8], dst: &mut [f32], channel: usize) -> Result<(), String> {
    let values = read_f32_array::<TILE_COUNT>(bytes);
    dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].copy_from_slice(&values);
    Ok(())
}
