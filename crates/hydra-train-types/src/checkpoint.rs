//! Checkpoint metadata DTOs shared by training and execution crates.

/// Metadata persisted next to model checkpoint payloads.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_agreement: Option<f64>,
    pub eval_policy_loss: Option<f64>,
    pub eval_total_loss: Option<f64>,
    pub timestamp: u64,
    pub num_blocks: usize,
    pub hidden_channels: usize,
}

impl CheckpointMeta {
    #[must_use]
    pub fn with_timestamp(
        epoch: u32,
        train_loss: f64,
        eval_agreement: Option<f64>,
        eval_policy_loss: Option<f64>,
        eval_total_loss: Option<f64>,
        timestamp: u64,
    ) -> Self {
        Self {
            epoch,
            train_loss,
            eval_agreement,
            eval_policy_loss,
            eval_total_loss,
            timestamp,
            num_blocks: 24,
            hidden_channels: 256,
        }
    }

    #[must_use]
    pub fn new(
        epoch: u32,
        train_loss: f64,
        eval_agreement: Option<f64>,
        eval_policy_loss: Option<f64>,
        eval_total_loss: Option<f64>,
    ) -> Self {
        Self::with_timestamp(
            epoch,
            train_loss,
            eval_agreement,
            eval_policy_loss,
            eval_total_loss,
            current_unix_timestamp_secs(),
        )
    }

    #[must_use]
    pub fn summary(&self) -> String {
        let eval_summary = match (
            self.eval_policy_loss,
            self.eval_total_loss,
            self.eval_agreement,
        ) {
            (Some(policy_loss), Some(total_loss), Some(agreement)) => format!(
                "policy_ce={policy_loss:.4} total={total_loss:.4} agree={:.2}%",
                agreement * 100.0
            ),
            _ => "eval=n/a".to_string(),
        };
        format!(
            "epoch={} loss={:.4} {}",
            self.epoch, self.train_loss, eval_summary
        )
    }
}

fn current_unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests;
