//! Compact observation fact encoding.

use std::io::{self, Write};

use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_data_core::sample::{
    COMPACT_MISSING_SHANTEN, COMPACT_MISSING_TILE, CompactMeldType, CompactObservationFacts,
    MjaiSample,
};

use crate::compact::{pack_tile_counts, validate_u8_range};
use crate::manifest::{
    COMPACT_OBS_BASELINE_FACT_BYTES, TILE_COUNT, TILE34_BITSET_BYTES, TILE34_COUNT_BYTES,
};

use super::invalid_data;
use super::primitives::write_u8;

pub(super) fn write_compact_obs<W: Write>(writer: &mut W, sample: &MjaiSample) -> io::Result<()> {
    let facts = sample
        .compact_facts
        .as_ref()
        .ok_or_else(|| invalid_data("compact observation facts required for BC shard writer"))?;
    write_baseline_obs_facts(writer, facts)?;
    validate_absent_advanced_obs(&sample.obs)
}

pub(super) fn write_baseline_obs_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<()> {
    let mut written = 0usize;
    written += write_tile_counts(writer, &facts.hand_counts)?;
    written += write_tile_counts(writer, &facts.open_meld_counts)?;
    written += write_optional_tile_bitset(writer, facts.drawn_tile)?;
    written += write_shanten_fact_bitsets(writer, facts.shanten_base, &facts.shanten_discard)?;
    written += write_discard_facts(writer, facts)?;
    written += write_meld_facts(writer, facts)?;
    written += write_dora_facts(writer, facts)?;
    written += write_aka_facts(writer, facts)?;
    written += write_metadata_facts(writer, facts)?;
    written += write_safety_facts(writer, facts)?;
    debug_assert_eq!(written, COMPACT_OBS_BASELINE_FACT_BYTES);
    Ok(())
}

fn write_tile_counts<W: Write>(writer: &mut W, counts: &[u8; TILE_COUNT]) -> io::Result<usize> {
    let mut packed = [0u8; TILE34_COUNT_BYTES];
    pack_tile_counts(counts, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)?;
    Ok(TILE34_COUNT_BYTES)
}

fn write_optional_tile_bitset<W: Write>(writer: &mut W, tile: u8) -> io::Result<usize> {
    let mut packed = [0u8; TILE34_BITSET_BYTES];
    if tile != COMPACT_MISSING_TILE {
        validate_u8_range("drawn_tile", tile, TILE_COUNT as u8).map_err(invalid_data)?;
        packed[tile as usize / 8] |= 1u8 << (tile % 8);
    }
    writer.write_all(&packed)?;
    Ok(TILE34_BITSET_BYTES)
}

fn write_shanten_fact_bitsets<W: Write>(
    writer: &mut W,
    base: i8,
    discard: &[i8; TILE_COUNT],
) -> io::Result<usize> {
    let mut keep = [0u8; TILE34_BITSET_BYTES];
    let mut next = [0u8; TILE34_BITSET_BYTES];
    for (tile, &after) in discard.iter().enumerate() {
        if after == COMPACT_MISSING_SHANTEN {
            continue;
        }
        if after <= base {
            keep[tile / 8] |= 1u8 << (tile % 8);
        }
        if after < base {
            next[tile / 8] |= 1u8 << (tile % 8);
        }
    }
    writer.write_all(&keep)?;
    writer.write_all(&next)?;
    Ok(TILE34_BITSET_BYTES * 2)
}

fn write_u64_bitset<W: Write>(writer: &mut W, value: u64, name: &'static str) -> io::Result<usize> {
    if value >> TILE_COUNT != 0 {
        return Err(invalid_data(format!(
            "compact {name} bitset has out-of-range tiles"
        )));
    }
    let mut packed = [0u8; TILE34_BITSET_BYTES];
    for tile in 0..TILE_COUNT {
        if ((value >> tile) & 1) != 0 {
            packed[tile / 8] |= 1u8 << (tile % 8);
        }
    }
    writer.write_all(&packed)?;
    Ok(TILE34_BITSET_BYTES)
}

fn write_discard_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<usize> {
    let mut written = 0usize;
    for player in 0..4usize {
        let player_discards = &facts.discards[player];
        let mut presence = 0u64;
        let mut tedashi = 0u64;
        let mut latest = [u32::MAX; TILE_COUNT];
        let len = usize::from(player_discards.len);
        if len > player_discards.discards.len() {
            return Err(invalid_data("compact discard length out of range"));
        }
        let t_max = player_discards.discards[..len]
            .last()
            .map(|entry| entry.turn)
            .unwrap_or(0);
        for entry in &player_discards.discards[..len] {
            if entry.tile as usize >= TILE_COUNT {
                return Err(invalid_data("compact discard tile out of range"));
            }
            let tile = entry.tile as usize;
            presence |= 1u64 << tile;
            if entry.is_tedashi {
                tedashi |= 1u64 << tile;
            }
            latest[tile] = u32::from((t_max - entry.turn).min(30));
        }
        written += write_u64_bitset(writer, presence, "discard_presence")?;
        written += write_u64_bitset(writer, tedashi, "discard_tedashi")?;
        for index in latest {
            writer.write_all(&index.to_le_bytes())?;
        }
        written += TILE_COUNT * 4;
    }
    Ok(written)
}

fn write_meld_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<usize> {
    let mut written = 0usize;
    for player in 0..4usize {
        let player_melds = &facts.melds[player];
        let len = usize::from(player_melds.len);
        if len > player_melds.melds.len() {
            return Err(invalid_data("compact meld length out of range"));
        }
        let mut chi = 0u64;
        let mut pon = 0u64;
        let mut kan = 0u64;
        for meld in &player_melds.melds[..len] {
            let target = match meld.meld_type {
                CompactMeldType::Chi => &mut chi,
                CompactMeldType::Pon => &mut pon,
                CompactMeldType::Kan => &mut kan,
            };
            let tile_count = usize::from(meld.tile_count);
            if tile_count > meld.tiles.len() {
                return Err(invalid_data("compact meld tile_count out of range"));
            }
            match meld.meld_type {
                CompactMeldType::Chi => {
                    for &tile in &meld.tiles[..tile_count] {
                        validate_u8_range("meld_tile", tile, TILE_COUNT as u8)
                            .map_err(invalid_data)?;
                        *target |= 1u64 << tile;
                    }
                }
                CompactMeldType::Pon | CompactMeldType::Kan => {
                    let tile = meld.tiles[0];
                    validate_u8_range("meld_tile", tile, TILE_COUNT as u8).map_err(invalid_data)?;
                    *target |= 1u64 << tile;
                }
            }
        }
        written += write_u64_bitset(writer, chi, "meld_chi")?;
        written += write_u64_bitset(writer, pon, "meld_pon")?;
        written += write_u64_bitset(writer, kan, "meld_kan")?;
    }
    Ok(written)
}

fn write_dora_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<usize> {
    let mut counts = [0u8; TILE_COUNT];
    let count = usize::from(facts.dora_indicator_count);
    if count > facts.dora_indicators.len() {
        return Err(invalid_data("compact dora indicator count out of range"));
    }
    for &tile in &facts.dora_indicators[..count] {
        validate_u8_range("dora_indicator", tile, TILE_COUNT as u8).map_err(invalid_data)?;
        counts[tile as usize] = counts[tile as usize].saturating_add(1);
    }
    writer.write_all(&counts)?;
    Ok(TILE_COUNT)
}

fn write_aka_facts<W: Write>(writer: &mut W, facts: &CompactObservationFacts) -> io::Result<usize> {
    let mut flags = 0u8;
    for (suit, &has_aka) in facts.aka_flags.iter().enumerate() {
        if has_aka {
            flags |= 1u8 << suit;
        }
    }
    write_u8(writer, flags)?;
    Ok(1)
}

fn write_metadata_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<usize> {
    let mut riichi = 0u64;
    for (idx, &active) in facts.riichi.iter().enumerate() {
        if active {
            riichi |= 1u64 << idx;
        }
    }
    let mut written = write_u64_bitset(writer, riichi, "riichi")?;
    for score in facts.scores {
        writer.write_all(&(score as f32 / 100_000.0).to_le_bytes())?;
        written += 4;
    }
    let my_score = facts.scores[0];
    for score in facts.scores {
        writer.write_all(&((my_score - score) as f32 / 30_000.0).to_le_bytes())?;
        written += 4;
    }
    let shanten = facts.shanten_base.clamp(0, 3) as u64;
    written += write_u64_bitset(writer, 1u64 << shanten, "shanten")?;
    writer.write_all(&(facts.kyoku_index as f32 / 8.0).to_le_bytes())?;
    writer.write_all(&(facts.honba as f32 / 10.0).to_le_bytes())?;
    writer.write_all(&(facts.kyotaku as f32 / 10.0).to_le_bytes())?;
    written += 12;
    Ok(written)
}

fn write_safety_facts<W: Write>(
    writer: &mut W,
    facts: &CompactObservationFacts,
) -> io::Result<usize> {
    let mut written = 0usize;
    for &bits in &facts.safety.genbutsu_all {
        written += write_u64_bitset(writer, bits, "genbutsu_all")?;
    }
    for &bits in &facts.safety.genbutsu_tedashi {
        written += write_u64_bitset(writer, bits, "genbutsu_tedashi")?;
    }
    for &bits in &facts.safety.genbutsu_riichi_era {
        written += write_u64_bitset(writer, bits, "genbutsu_riichi_era")?;
    }
    for plane in &facts.safety.suji {
        for &value in plane {
            writer.write_all(&value.to_le_bytes())?;
        }
        written += TILE_COUNT * 4;
    }
    for &bits in &facts.safety.half_suji {
        written += write_u64_bitset(writer, bits, "half_suji")?;
    }
    for plane in &facts.safety.matagi {
        for &value in plane {
            writer.write_all(&value.to_le_bytes())?;
        }
        written += TILE_COUNT * 4;
    }
    written += write_u64_bitset(writer, facts.safety.kabe, "kabe")?;
    written += write_u64_bitset(writer, facts.safety.one_chance, "one_chance")?;
    for opp in 0..3usize {
        let active =
            facts.safety.opponent_riichi[opp] || facts.safety.cached_tenpai_prob[opp] > 0.5;
        written += write_u64_bitset(
            writer,
            if active { (1u64 << TILE_COUNT) - 1 } else { 0 },
            "tenpai_hint",
        )?;
    }
    Ok(written)
}

fn validate_absent_advanced_obs(values: &[f32; OBS_SIZE]) -> io::Result<()> {
    for channel in hydra_data_core::sample::COMPACT_BASELINE_CHANNELS..NUM_CHANNELS {
        let start = channel * TILE_COUNT;
        if values[start..start + TILE_COUNT]
            .iter()
            .any(|&value| value != 0.0)
        {
            return Err(invalid_data(format!(
                "compact replay BC shards require absent advanced observation channels; channel {channel} was nonzero"
            )));
        }
    }
    Ok(())
}
