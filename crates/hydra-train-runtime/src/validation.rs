use crate::config::{AdvancedLossConfig, TrainConfig, ValidationGateConfig};
use crate::config_runtime::{
    shard_prefetch_depth, validation_microbatch_size, validation_sample_limit,
};

#[derive(Debug, Clone)]
pub struct ValidationRunConfig {
    pub limits: ValidationRunLimits,
    pub gates: ValidationGateConfig,
    pub advanced_loss: Option<AdvancedLossConfig>,
    pub shard_prefetch_depth: usize,
}

impl ValidationRunConfig {
    #[must_use]
    pub fn from_config(config: &TrainConfig) -> Self {
        Self {
            limits: ValidationRunLimits::from_config(config),
            gates: config.validation_gates.clone(),
            advanced_loss: config.advanced_loss.clone(),
            shard_prefetch_depth: shard_prefetch_depth(config),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationRunLimits {
    pub microbatch_size: usize,
    pub sample_limit: Option<usize>,
}

impl ValidationRunLimits {
    #[must_use]
    pub fn from_config(config: &TrainConfig) -> Self {
        Self {
            microbatch_size: validation_microbatch_size(config),
            sample_limit: validation_sample_limit(config),
        }
    }

    #[must_use]
    pub fn target_samples_label(self) -> String {
        match self.sample_limit {
            Some(limit) => format!("target_samples={limit}"),
            None => "target_samples=all".to_string(),
        }
    }

    #[must_use]
    pub fn capped_len(self, processed_samples: usize, chunk_len: usize) -> usize {
        self.sample_limit
            .map(|limit| limit.saturating_sub(processed_samples).min(chunk_len))
            .unwrap_or(chunk_len)
    }

    #[must_use]
    pub fn reached_sample_limit(self, processed_samples: usize) -> bool {
        self.sample_limit
            .is_some_and(|limit| processed_samples >= limit)
    }

    #[must_use]
    pub fn bounded_total_rows(self, total_rows: usize) -> usize {
        self.sample_limit.unwrap_or(total_rows).min(total_rows)
    }
}

#[cfg(test)]
mod tests;
