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
mod tests {
    use super::*;
    use crate::test_support::dummy_train_config;

    #[test]
    fn from_config_preserves_validation_microbatch_and_sample_limit_rules() {
        let mut config = dummy_train_config();
        config.microbatch_size = Some(64);
        config.validation_microbatch_size = Some(20);
        config.max_validation_batches = Some(3);
        config.max_validation_samples = None;

        let limits = ValidationRunLimits::from_config(&config);

        assert_eq!(limits.microbatch_size, 20);
        assert_eq!(limits.sample_limit, Some(60));
        assert_eq!(limits.target_samples_label(), "target_samples=60");
    }

    #[test]
    fn run_config_from_config_preserves_gate_loss_and_prefetch_settings() {
        let mut config = dummy_train_config();
        config.microbatch_size = Some(64);
        config.validation_microbatch_size = Some(16);
        config.max_validation_samples = Some(37);
        config.validation_gates.enabled = true;
        config.validation_gates.min_policy_agreement_delta = Some(0.05);
        config.advanced_loss = Some(crate::config::AdvancedLossConfig {
            delta_q: Some(0.25),
            ..Default::default()
        });
        config.shard_prefetch_depth = Some(4);

        let run_config = ValidationRunConfig::from_config(&config);

        assert_eq!(run_config.limits.microbatch_size, 16);
        assert_eq!(run_config.limits.sample_limit, Some(37));
        assert!(run_config.gates.enabled);
        assert_eq!(run_config.gates.min_policy_agreement_delta, Some(0.05));
        assert_eq!(
            run_config
                .advanced_loss
                .as_ref()
                .and_then(|loss| loss.delta_q),
            Some(0.25)
        );
        assert_eq!(run_config.shard_prefetch_depth, 4);
    }

    #[test]
    fn helpers_cap_chunks_and_rows_without_changing_unlimited_behavior() {
        let limited = ValidationRunLimits {
            microbatch_size: 8,
            sample_limit: Some(10),
        };
        assert_eq!(limited.capped_len(0, 8), 8);
        assert_eq!(limited.capped_len(8, 8), 2);
        assert_eq!(limited.capped_len(10, 8), 0);
        assert!(limited.reached_sample_limit(10));
        assert_eq!(limited.bounded_total_rows(99), 10);

        let unlimited = ValidationRunLimits {
            microbatch_size: 8,
            sample_limit: None,
        };
        assert_eq!(unlimited.capped_len(99, 8), 8);
        assert!(!unlimited.reached_sample_limit(usize::MAX));
        assert_eq!(unlimited.bounded_total_rows(99), 99);
        assert_eq!(unlimited.target_samples_label(), "target_samples=all");
    }
}
