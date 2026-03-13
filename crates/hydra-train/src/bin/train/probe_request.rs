use std::path::PathBuf;

use hydra_train::preflight::ProbeKind;

use super::config::{ProbeChildRequest, ProbeCliRequest, TrainConfig};

#[derive(Debug, Clone, Copy)]
pub(super) struct ProbeRequest {
    pub(super) kind: ProbeKind,
    pub(super) candidate_microbatch: usize,
    pub(super) warmup_steps: usize,
    pub(super) measure_steps: usize,
}

pub(super) fn probe_request_from_cli(
    config: &TrainConfig,
    probe: Option<ProbeCliRequest>,
) -> Result<Option<ProbeRequest>, String> {
    let Some(probe) = probe else {
        return Ok(None);
    };
    let warmup_steps = probe.warmup_steps.unwrap_or(config.preflight.warmup_steps);
    let measure_steps = probe
        .measure_steps
        .unwrap_or(config.preflight.measure_steps);
    if probe.candidate_microbatch == 0 {
        return Err("--probe-candidate-microbatch must be greater than 0".to_string());
    }
    if warmup_steps == 0 {
        return Err("--probe-warmup-steps must be greater than 0".to_string());
    }
    if measure_steps == 0 {
        return Err("--probe-measure-steps must be greater than 0".to_string());
    }
    Ok(Some(ProbeRequest {
        kind: probe.kind,
        candidate_microbatch: probe.candidate_microbatch,
        warmup_steps,
        measure_steps,
    }))
}

pub(super) fn probe_child_request_from_cli(
    child: Option<ProbeChildRequest>,
) -> Result<Option<(ProbeRequest, PathBuf)>, String> {
    let Some(child) = child else {
        return Ok(None);
    };
    let request = ProbeRequest {
        kind: child.request.kind,
        candidate_microbatch: child.request.candidate_microbatch,
        warmup_steps: child
            .request
            .warmup_steps
            .ok_or_else(|| "internal probe child missing resolved warmup steps".to_string())?,
        measure_steps: child
            .request
            .measure_steps
            .ok_or_else(|| "internal probe child missing resolved measure steps".to_string())?,
    };
    Ok(Some((request, child.result_path)))
}

pub(super) fn probe_candidate_ceiling(request: ProbeRequest) -> usize {
    request.candidate_microbatch.max(1)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hydra_train::preflight::{PreflightConfig, ProbeKind};

    use super::*;

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            bc: Default::default(),
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
        }
    }

    #[test]
    fn probe_request_from_cli_uses_probe_overrides() {
        let config = dummy_config();

        let request = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 192,
                warmup_steps: Some(7),
                measure_steps: Some(9),
            }),
        )
        .expect("probe request should parse")
        .expect("probe request should be present");
        assert_eq!(request.kind, ProbeKind::Validation);
        assert_eq!(request.candidate_microbatch, 192);
        assert_eq!(request.warmup_steps, 7);
        assert_eq!(request.measure_steps, 9);
    }

    #[test]
    fn probe_request_from_cli_falls_back_to_preflight_defaults() {
        let mut config = dummy_config();
        config.preflight.warmup_steps = 11;
        config.preflight.measure_steps = 13;
        let request = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 256,
                warmup_steps: None,
                measure_steps: None,
            }),
        )
        .expect("probe request should parse")
        .expect("probe request should be present");
        assert_eq!(request.warmup_steps, 11);
        assert_eq!(request.measure_steps, 13);
    }

    #[test]
    fn probe_request_from_cli_rejects_zero_values() {
        let config = dummy_config();

        let err = probe_request_from_cli(
            &config,
            Some(ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 0,
                warmup_steps: Some(0),
                measure_steps: Some(0),
            }),
        )
        .expect_err("zero candidate should fail");
        assert!(err.contains("--probe-candidate-microbatch"));
    }

    #[test]
    fn probe_child_request_from_cli_parses_child_probe_inputs() {
        let (request, path) = probe_child_request_from_cli(Some(ProbeChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 192,
                warmup_steps: Some(4),
                measure_steps: Some(12),
            },
            result_path: PathBuf::from("/tmp/probe.json"),
        }))
        .expect("child request should parse")
        .expect("child request should be present");
        assert_eq!(request.kind, ProbeKind::Train);
        assert_eq!(request.candidate_microbatch, 192);
        assert_eq!(request.warmup_steps, 4);
        assert_eq!(request.measure_steps, 12);
        assert_eq!(path, PathBuf::from("/tmp/probe.json"));
    }
}
