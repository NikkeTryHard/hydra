use crate::config::TrainConfig;
use crate::config_runtime::{validation_microbatch_size, validation_sample_limit};

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
