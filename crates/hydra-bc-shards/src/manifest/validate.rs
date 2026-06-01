//! BC shard manifest validators.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};

use super::constants::*;
use super::types::*;

/// Validates a BC shard manifest against the current frozen runtime ABI.
pub fn validate_bc_shard_manifest_contract(manifest: &BcShardManifest) -> Result<(), String> {
    if manifest.storage_layout != STORAGE_LAYOUT_COMPACT {
        return Err(format!(
            "BC shard manifest storage_layout {:?} is unsupported; expected compact. Shards must be rebuilt from replay.",
            manifest.storage_layout,
        ));
    }
    if manifest.manifest_version != BC_SHARD_MANIFEST_VERSION {
        return Err(format!(
            "BC shard manifest version {} is unsupported; expected {}. Shards must be rebuilt from replay.",
            manifest.manifest_version, BC_SHARD_MANIFEST_VERSION,
        ));
    }
    if manifest.shard_version != BC_SHARD_VERSION {
        return Err(format!(
            "BC shard version {} is unsupported; expected {}. Shards must be rebuilt from replay.",
            manifest.shard_version, BC_SHARD_VERSION,
        ));
    }
    if manifest.shard_header_size != BC_SHARD_HEADER_SIZE {
        return Err(format!(
            "BC shard header size {} does not match current {}. Shards must be rebuilt from replay.",
            manifest.shard_header_size, BC_SHARD_HEADER_SIZE,
        ));
    }
    if manifest.obs_size != OBS_SIZE {
        return Err(format!(
            "BC shard manifest obs_size {} does not match current OBS_SIZE {} \
             (num_channels: manifest={}, binary={}). \
             Shards must be rebuilt with the current encoder.",
            manifest.obs_size, OBS_SIZE, manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.num_channels != NUM_CHANNELS {
        return Err(format!(
            "BC shard manifest num_channels {} does not match current NUM_CHANNELS {}. \
             Shards must be rebuilt with the current encoder.",
            manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.base_record_size != BC_BASE_RECORD_SIZE {
        return Err(format!(
            "BC shard manifest base_record_size {} does not match current compact BC_BASE_RECORD_SIZE {}. Shards must be rebuilt from replay.",
            manifest.base_record_size, BC_BASE_RECORD_SIZE,
        ));
    }
    if manifest.max_record_size != BC_RECORD_SIZE_WITH_ALL_OPTIONALS {
        return Err(format!(
            "BC shard manifest max_record_size {} does not match current compact max {}. Shards must be rebuilt from replay.",
            manifest.max_record_size, BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        ));
    }
    if manifest.action_space != HYDRA_ACTION_SPACE {
        return Err(format!(
            "BC shard manifest action_space {} does not match current HYDRA_ACTION_SPACE {}. \
             Shards must be rebuilt with the current action contract.",
            manifest.action_space, HYDRA_ACTION_SPACE,
        ));
    }
    let split_mode = match manifest.split_mode.as_str() {
        "both" => BcShardSplitMode::Both,
        "train" => BcShardSplitMode::Train,
        "validation" => BcShardSplitMode::Validation,
        other => {
            return Err(format!(
                "BC shard manifest split_mode {other:?} is unsupported; expected one of both, train, validation"
            ));
        }
    };
    let required_split = match split_mode {
        BcShardSplitMode::Both => None,
        BcShardSplitMode::Train => Some(BcShardSplit::Train),
        BcShardSplitMode::Validation => Some(BcShardSplit::Validation),
    };
    let mut has_train_split = false;
    let mut has_validation_split = false;
    let mut total_samples = 0u64;
    let mut total_shards = 0usize;
    for split in &manifest.splits {
        if !split_mode.includes(split.split) {
            return Err(format!(
                "BC shard manifest split_mode {} excludes {:?} split entries",
                split_mode.as_str(),
                split.split,
            ));
        }
        match split.split {
            BcShardSplit::Train => {
                if has_train_split {
                    return Err(
                        "BC shard manifest contains duplicate train split entries".to_string()
                    );
                }
                has_train_split = true;
            }
            BcShardSplit::Validation => {
                if has_validation_split {
                    return Err(
                        "BC shard manifest contains duplicate validation split entries".to_string(),
                    );
                }
                has_validation_split = true;
            }
        }
        total_samples = total_samples
            .checked_add(split.sample_count)
            .ok_or_else(|| "BC shard manifest split sample_count total overflow".to_string())?;
        total_shards = total_shards
            .checked_add(split.shard_count)
            .ok_or_else(|| "BC shard manifest split shard_count total overflow".to_string())?;
    }
    for split in &manifest.splits {
        validate_bc_shard_split_manifest_contract(split)?;
    }
    if manifest.totals.sample_count != total_samples {
        return Err(format!(
            "BC shard manifest totals.sample_count {} does not match split total {}",
            manifest.totals.sample_count, total_samples
        ));
    }
    if manifest.totals.shard_count != total_shards {
        return Err(format!(
            "BC shard manifest totals.shard_count {} does not match split shard total {}",
            manifest.totals.shard_count, total_shards
        ));
    }
    if manifest.totals.sample_count > 0 {
        match required_split {
            Some(BcShardSplit::Train) if !has_train_split => {
                return Err(
                    "BC shard manifest split_mode train requires a train split entry".to_string(),
                );
            }
            Some(BcShardSplit::Validation) if !has_validation_split => {
                return Err(
                    "BC shard manifest split_mode validation requires a validation split entry"
                        .to_string(),
                );
            }
            None if !has_train_split || !has_validation_split => {
                return Err(
                    "BC shard manifest split_mode both requires train and validation split entries"
                        .to_string(),
                );
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validates split-level shard descriptor contiguity and consistency.
pub fn validate_bc_shard_split_manifest_contract(
    split: &BcShardSplitManifest,
) -> Result<(), String> {
    if split.shard_count != split.shards.len() {
        return Err(format!(
            "BC shard manifest {:?} shard_count {} does not match descriptor count {}",
            split.split,
            split.shard_count,
            split.shards.len()
        ));
    }
    let mut expected_start = 0u64;
    for (idx, shard) in split.shards.iter().enumerate() {
        if shard.split != split.split {
            return Err(format!(
                "BC shard descriptor {} has split {:?}, expected {:?}",
                idx, shard.split, split.split
            ));
        }
        if shard.shard_index != idx {
            return Err(format!(
                "BC shard descriptor for {:?} has shard_index {}, expected {}",
                split.split, shard.shard_index, idx
            ));
        }
        if shard.first_sample_index != expected_start {
            return Err(format!(
                "BC shard descriptor {} for {:?} starts at {}, expected contiguous start {}",
                idx, split.split, shard.first_sample_index, expected_start
            ));
        }
        if shard.feature_flags != split.feature_flags {
            return Err(format!(
                "BC shard descriptor {} for {:?} feature_flags {} does not match split feature_flags {}",
                idx, split.split, shard.feature_flags, split.feature_flags
            ));
        }
        let expected_record_size = checked_compact_record_size(shard.feature_flags)?;
        if shard.record_size != expected_record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match compact record size {} for flags {:#x}",
                idx, split.split, shard.record_size, expected_record_size, shard.feature_flags
            ));
        }
        if shard.record_size != split.record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match split record_size {}",
                idx, split.split, shard.record_size, split.record_size
            ));
        }
        if !is_safe_relative_shard_name(&shard.file_name) {
            return Err(format!(
                "BC shard descriptor {} for {:?} has unsafe file name {:?}",
                idx, split.split, shard.file_name
            ));
        }
        let expected_byte_len = (BC_SHARD_HEADER_SIZE as u64)
            .checked_add(checked_record_bytes(shard.sample_count, shard.record_size)?)
            .ok_or_else(|| "BC shard descriptor byte_len overflow".to_string())?;
        if shard.byte_len != expected_byte_len {
            return Err(format!(
                "BC shard descriptor {} for {:?} byte_len {} does not match header + records {}",
                idx, split.split, shard.byte_len, expected_byte_len
            ));
        }
        expected_start = expected_start
            .checked_add(shard.sample_count)
            .ok_or_else(|| "BC shard split sample_count overflow".to_string())?;
    }
    if split.sample_count != expected_start {
        return Err(format!(
            "BC shard split {:?} sample_count {} does not match descriptor total {}",
            split.split, split.sample_count, expected_start
        ));
    }
    Ok(())
}

fn is_safe_relative_shard_name(name: &str) -> bool {
    let path = std::path::Path::new(name);
    !name.is_empty()
        && path.is_relative()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}
