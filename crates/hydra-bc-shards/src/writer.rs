//! BC shard writer helpers for the frozen host binary format.

use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_data_core::sample::{
    COMPACT_MISSING_SHANTEN, COMPACT_MISSING_TILE, CompactMeldType, CompactObservationFacts,
    MjaiSample,
};

use crate::compact::{
    pack_action_mask, pack_binary_f32_mask, pack_spatial_mask, pack_tile_counts, validate_u8_range,
};
use crate::host::record_size_for_flags;
use crate::manifest::{
    BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MAGIC, BC_SHARD_VERSION,
    BcShardDescriptor, BcShardSplit, BcShardSplitManifest, COMPACT_OBS_BASELINE_FACT_BYTES,
    FLAG_BELIEF_FIELDS, FLAG_DELTA_Q, FLAG_EXIT, FLAG_MIXTURE_WEIGHTS, FLAG_SAFETY_RESIDUAL,
    OPPONENT_COUNT, OPTIONAL_ACTION_FLOAT32_BYTES, ORACLE_FLOAT32_BYTES, PACKED_ACTION_MASK_BYTES,
    PACKED_SPATIAL_MASK_BYTES, PLAYER_COUNT, SPATIAL_TARGET_SIZE, TILE_COUNT, TILE34_BITSET_BYTES,
    TILE34_COUNT_BYTES, checked_compact_record_size, validate_feature_flags,
};

/// Active writer for one BC shard file.
pub struct ActiveShardWriter {
    split: BcShardSplit,
    shard_index: usize,
    file_name: String,
    first_sample_index: u64,
    sample_count: u64,
    feature_flags: u32,
    record_size: u32,
    writer: BufWriter<fs::File>,
}

/// Incremental state for building one split's BC shard files.
pub struct SplitBuildState {
    split: BcShardSplit,
    next_shard_index: usize,
    total_samples: u64,
    feature_flags: u32,
    record_size: u32,
    shards: Vec<BcShardDescriptor>,
    active: Option<ActiveShardWriter>,
}

impl SplitBuildState {
    /// Creates split build state for `split` and `feature_flags`.
    pub fn new(split: BcShardSplit, feature_flags: u32) -> Self {
        Self {
            split,
            next_shard_index: 0,
            total_samples: 0,
            feature_flags,
            record_size: record_size_for_flags(feature_flags),
            shards: Vec::new(),
            active: None,
        }
    }

    /// Pushes all samples from one game into this split, opening or rotating shards as needed.
    pub fn push_samples(
        &mut self,
        output_dir: &Path,
        shard_samples: usize,
        samples: &[MjaiSample],
    ) -> io::Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let game_samples = samples.len() as u64;
        if let Some(active) = self.active.as_ref()
            && active.sample_count > 0
            && active.sample_count + game_samples > shard_samples.max(1) as u64
        {
            self.finish_active()?;
        }
        if self.active.is_none() {
            let shard = ActiveShardWriter::new(
                output_dir,
                self.split,
                self.next_shard_index,
                self.total_samples,
                self.feature_flags,
            )?;
            self.next_shard_index += 1;
            self.active = Some(shard);
        }
        let active = self.active.as_mut().expect("active shard should exist");
        active.write_samples(samples)?;
        self.total_samples += game_samples;
        Ok(())
    }

    /// Finishes any active shard and records its descriptor.
    pub fn finish_active(&mut self) -> io::Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let descriptor = active.finish()?;
        self.shards.push(descriptor);
        Ok(())
    }

    /// Finalizes this split and returns its manifest.
    pub fn finalize(mut self) -> io::Result<BcShardSplitManifest> {
        self.finish_active()?;
        Ok(BcShardSplitManifest {
            split: self.split,
            shard_count: self.shards.len(),
            sample_count: self.total_samples,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
            shards: self.shards,
        })
    }
}

impl ActiveShardWriter {
    /// Opens a new active shard writer in `output_dir` using the default shard file name.
    pub fn new(
        output_dir: &Path,
        split: BcShardSplit,
        shard_index: usize,
        first_sample_index: u64,
        feature_flags: u32,
    ) -> io::Result<Self> {
        let file_name = format!("{}-{shard_index:05}.hydra-bc", split.shard_prefix());
        Self::new_named(
            output_dir,
            split,
            shard_index,
            first_sample_index,
            feature_flags,
            file_name,
        )
    }

    /// Opens a new active shard writer in `output_dir` using `file_name`.
    pub fn new_named(
        output_dir: &Path,
        split: BcShardSplit,
        shard_index: usize,
        first_sample_index: u64,
        feature_flags: u32,
        file_name: String,
    ) -> io::Result<Self> {
        let path = output_dir.join(&file_name);
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let record_size = checked_compact_record_size(feature_flags).map_err(invalid_data)?;
        write_shard_header(
            &mut writer,
            split,
            shard_index as u32,
            0,
            first_sample_index,
            feature_flags,
            record_size,
        )?;
        Ok(Self {
            split,
            shard_index,
            file_name,
            first_sample_index,
            sample_count: 0,
            feature_flags,
            record_size,
            writer,
        })
    }

    /// Writes samples into the active shard.
    pub fn write_samples(&mut self, samples: &[MjaiSample]) -> io::Result<()> {
        for sample in samples {
            write_sample_record(&mut self.writer, sample, self.feature_flags)?;
            self.sample_count += 1;
        }
        Ok(())
    }

    /// Writes already-validated compact records into the active shard.
    pub fn write_encoded_records(&mut self, records: &[u8], sample_count: usize) -> io::Result<()> {
        let expected_len = sample_count
            .checked_mul(self.record_size as usize)
            .ok_or_else(|| invalid_data("encoded BC shard record byte count overflow"))?;
        if records.len() != expected_len {
            return Err(invalid_data(
                "encoded BC shard records have invalid byte length",
            ));
        }
        self.writer.write_all(records)?;
        self.sample_count += sample_count as u64;
        Ok(())
    }

    /// Finishes the shard and returns its descriptor.
    pub fn finish(mut self) -> io::Result<BcShardDescriptor> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        write_shard_header(
            file,
            self.split,
            self.shard_index as u32,
            self.sample_count,
            self.first_sample_index,
            self.feature_flags,
            self.record_size,
        )?;
        file.flush()?;
        let byte_len = file.metadata()?.len();
        Ok(BcShardDescriptor {
            split: self.split,
            shard_index: self.shard_index,
            file_name: self.file_name,
            sample_count: self.sample_count,
            first_sample_index: self.first_sample_index,
            byte_len,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
        })
    }
}

/// Rewrites a shard file header from a finalized descriptor.
pub fn rewrite_shard_header_for_descriptor(
    path: &Path,
    descriptor: &BcShardDescriptor,
) -> io::Result<()> {
    let mut file = fs::OpenOptions::new().write(true).open(path)?;
    write_shard_header(
        &mut file,
        descriptor.split,
        descriptor.shard_index as u32,
        descriptor.sample_count,
        descriptor.first_sample_index,
        descriptor.feature_flags,
        descriptor.record_size,
    )?;
    file.flush()?;
    file.sync_all()
}

/// Writes one BC shard header.
pub fn write_shard_header<W: Write>(
    writer: &mut W,
    split: BcShardSplit,
    shard_index: u32,
    sample_count: u64,
    first_sample_index: u64,
    feature_flags: u32,
    record_size: u32,
) -> io::Result<()> {
    writer.write_all(&BC_SHARD_MAGIC)?;
    write_u32_le(writer, BC_SHARD_VERSION)?;
    write_u32_le(writer, BC_SHARD_HEADER_SIZE)?;
    write_u32_le(writer, record_size)?;
    write_u32_le(writer, split.split_id())?;
    write_u32_le(writer, shard_index)?;
    write_u64_le(writer, sample_count)?;
    write_u32_le(writer, NUM_CHANNELS as u32)?;
    write_u32_le(writer, TILE_COUNT as u32)?;
    write_u32_le(writer, HYDRA_ACTION_SPACE as u32)?;
    write_u64_le(writer, first_sample_index)?;
    write_u32_le(writer, feature_flags)?;
    write_u32_le(writer, BC_SHARD_LAYOUT_VERSION)?;
    write_u64_le(writer, 0)?;
    write_u64_le(writer, 0)?;
    Ok(())
}

fn invalid_data(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// Writes one compact BC shard sample record.
pub fn write_sample_record<W: Write>(
    writer: &mut W,
    sample: &MjaiSample,
    flags: u32,
) -> io::Result<()> {
    validate_feature_flags(flags).map_err(invalid_data)?;
    if record_size_for_flags(flags) != checked_compact_record_size(flags).map_err(invalid_data)? {
        return Err(invalid_data("BC shard compact record-size helper mismatch"));
    }
    write_compact_obs(writer, sample)?;
    validate_u8_range("action", sample.action, HYDRA_ACTION_SPACE as u8).map_err(invalid_data)?;
    write_u8(writer, sample.action)?;
    write_packed_action_mask(writer, &sample.legal_mask)?;
    write_i32_le(writer, sample.score_delta)?;
    validate_u8_range(
        "grp_label",
        sample.grp_label,
        crate::host::GRP_CLASS_COUNT as u8,
    )
    .map_err(invalid_data)?;
    write_u8(writer, sample.grp_label)?;
    write_optional_oracle_f32(writer, sample.oracle_target.as_ref())?;
    write_u8(writer, u8::from(sample.oracle_target.is_some()))?;
    write_packed_triplet(writer, &sample.tenpai)?;
    write_opp_next(writer, &sample.opp_next)?;
    write_packed_spatial_mask(writer, &sample.danger)?;
    write_packed_spatial_mask(writer, &sample.danger_mask)?;

    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        write_optional_action_f32(writer, sample.safety_residual.as_ref())?;
        write_optional_action_mask_packed(writer, sample.safety_residual_mask.as_ref())?;
    }
    if flags & FLAG_EXIT != 0 {
        write_optional_action_f32(writer, sample.exit_target.as_ref())?;
        write_optional_action_mask_packed(writer, sample.exit_mask.as_ref())?;
    }
    if flags & FLAG_DELTA_Q != 0 {
        write_optional_action_f32(writer, sample.delta_q_target.as_ref())?;
        write_optional_action_mask_packed(writer, sample.delta_q_mask.as_ref())?;
    }
    if flags & FLAG_BELIEF_FIELDS != 0 {
        write_required_f32_slice(
            writer,
            sample.belief_fields.as_ref().ok_or_else(|| {
                invalid_data("belief fields flag set but sample has no belief fields")
            })?,
        )?;
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        write_required_f32_slice(
            writer,
            sample.mixture_weights.as_ref().ok_or_else(|| {
                invalid_data("mixture weights flag set but sample has no mixture weights")
            })?,
        )?;
    }
    Ok(())
}

/// Encodes compact BC shard sample records into caller-owned bytes.
pub fn encode_sample_records(
    samples: &[MjaiSample],
    flags: u32,
    record_size: u32,
) -> io::Result<Vec<u8>> {
    validate_feature_flags(flags).map_err(invalid_data)?;
    let checked = checked_compact_record_size(flags).map_err(invalid_data)?;
    if record_size_for_flags(flags) != checked || record_size != checked {
        return Err(invalid_data("BC shard compact record-size helper mismatch"));
    }
    let record_size = record_size as usize;
    let record_len = crate::manifest::checked_encoded_record_len(samples.len(), checked)
        .map_err(invalid_data)?;
    let mut records = vec![0u8; record_len];
    for (sample, dst) in samples.iter().zip(records.chunks_exact_mut(record_size)) {
        write_sample_record(&mut &mut *dst, sample, flags)?;
    }
    Ok(records)
}

fn write_compact_obs<W: Write>(writer: &mut W, sample: &MjaiSample) -> io::Result<()> {
    let facts = sample
        .compact_facts
        .as_ref()
        .ok_or_else(|| invalid_data("compact observation facts required for BC shard writer"))?;
    write_baseline_obs_facts(writer, facts)?;
    validate_absent_advanced_obs(&sample.obs)
}

fn write_baseline_obs_facts<W: Write>(
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

fn write_packed_action_mask<W: Write>(
    writer: &mut W,
    values: &[f32; HYDRA_ACTION_SPACE],
) -> io::Result<()> {
    let mut packed = [0u8; PACKED_ACTION_MASK_BYTES];
    pack_action_mask(values, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

fn write_packed_triplet<W: Write>(
    writer: &mut W,
    values: &[f32; OPPONENT_COUNT],
) -> io::Result<()> {
    let mut packed = [0u8; 1];
    pack_binary_f32_mask(values, OPPONENT_COUNT, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

fn write_opp_next<W: Write>(writer: &mut W, values: &[u8; OPPONENT_COUNT]) -> io::Result<()> {
    for &value in values {
        if value != 255 {
            validate_u8_range("opp_next", value, TILE_COUNT as u8).map_err(invalid_data)?;
        }
    }
    writer.write_all(values)
}

fn write_optional_oracle_f32<W: Write>(
    writer: &mut W,
    values: Option<&[f32; PLAYER_COUNT]>,
) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, ORACLE_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_packed_spatial_mask<W: Write>(
    writer: &mut W,
    values: &[f32; SPATIAL_TARGET_SIZE],
) -> io::Result<()> {
    let mut packed = [0u8; PACKED_SPATIAL_MASK_BYTES];
    pack_spatial_mask(values, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

fn write_optional_action_f32<W: Write>(
    writer: &mut W,
    values: Option<&[f32; HYDRA_ACTION_SPACE]>,
) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, OPTIONAL_ACTION_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_optional_action_mask_packed<W: Write>(
    writer: &mut W,
    values: Option<&[f32; HYDRA_ACTION_SPACE]>,
) -> io::Result<()> {
    if let Some(values) = values {
        write_packed_action_mask(writer, values)
    } else {
        write_zero_bytes(writer, PACKED_ACTION_MASK_BYTES)
    }
}

fn write_required_f32_slice<W: Write, const N: usize>(
    writer: &mut W,
    values: &[f32; N],
) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_zero_bytes<W: Write>(writer: &mut W, total: usize) -> io::Result<()> {
    const ZERO_CHUNK: [u8; 4096] = [0u8; 4096];
    let mut remaining = total;
    while remaining > 0 {
        let chunk = remaining.min(ZERO_CHUNK.len());
        writer.write_all(&ZERO_CHUNK[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64_le<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_i32_le<W: Write>(writer: &mut W, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests;
