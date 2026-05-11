use hydra_train_exec::modes::{
    handle_delta_q_promotion_mode as run_exec_delta_q_promotion_mode,
    handle_preflight_mode as run_exec_preflight_mode, handle_probe_mode as run_exec_probe_mode,
};
use hydra_train_runtime::config::TrainConfig;
use hydra_train_runtime::probe_request::ProbeRequest;
use std::path::{Path, PathBuf};

pub(super) fn handle_preflight_mode(
    config_path: &Path,
    config: &TrainConfig,
) -> Result<(), String> {
    run_exec_preflight_mode(config_path, config)
}

pub(super) fn handle_probe_mode(
    config_path: &Path,
    config: &TrainConfig,
    request: ProbeRequest,
) -> Result<(), String> {
    run_exec_probe_mode(config_path, config, request)
}

pub(super) fn handle_delta_q_promotion_mode(
    config_path: &Path,
    config: TrainConfig,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String> {
    run_exec_delta_q_promotion_mode(config_path, config, baseline_checkpoint)
}

#[cfg(test)]
mod tests {
    use hydra_train_exec::delta_q_promotion::{
        default_arena_confirmation_request, delta_q_arena_requirement_summary,
        delta_q_promotion_stage, format_delta_q_offline_gate_message,
        format_delta_q_policy_holdout_message, format_delta_q_policy_transfer_gate_message,
        pre_arena_recommendation,
    };
    use hydra_train_exec::modes::{
        format_bc_preflight_selection_message, format_probe_best_candidate_detail,
        format_probe_only_status_detail, format_probe_only_status_message,
        format_probe_table_message, format_rl_preflight_selection_message,
    };
    use hydra_train_runtime::preflight::{ProbeKind, ProbeResult, ProbeStatus};
    use hydra_train_types::delta_q_promotion::DeltaQArenaConfirmationRequest;
    use hydra_train_types::delta_q_promotion::DeltaQPromotionRecommendation;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::test_support::{dummy_train_config, unique_test_path as shared_unique_test_path};
    use hydra_train_runtime::config::{RlTrainConfig, TrainConfig};

    fn dummy_config() -> TrainConfig {
        let mut config = dummy_train_config();
        config.num_threads = Some(1);
        config
    }

    fn dummy_probe_request(kind: ProbeKind) -> ProbeRequest {
        ProbeRequest {
            kind,
            candidate_microbatch: 192,
            warmup_steps: 4,
            measure_steps: 8,
        }
    }

    fn dummy_probe_result(
        kind: ProbeKind,
        candidate_microbatch: usize,
        selected: bool,
    ) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(if selected { 512.0 } else { 384.0 }),
            elapsed_seconds: Some(if selected { 1.5 } else { 2.0 }),
            detail: String::new(),
        }
    }

    fn unique_test_path(label: &str) -> PathBuf {
        shared_unique_test_path("hydra-modes-test", label)
    }

    #[test]
    fn format_probe_only_status_detail_is_stable() {
        assert_eq!(
            format_probe_only_status_detail(dummy_probe_request(ProbeKind::RlMicrobatch)),
            "kind=rl_microbatch candidate_mb=192 warmup_steps=4 measure_steps=8"
        );
    }

    #[test]
    fn format_probe_best_candidate_detail_uses_kind_name() {
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::Validation, 96),
            "validation=96"
        );
    }

    #[test]
    fn format_probe_table_message_supports_rl_games_rows() {
        let message = format_probe_table_message(
            "RL games probe table",
            ProbeKind::RlGames,
            &[dummy_probe_result(ProbeKind::RlGames, 24, true)],
            24,
        );

        assert!(message.contains("RL games probe table"));
        assert!(message.contains("rl_games"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("yes       24"));
    }

    #[test]
    fn pre_arena_recommendation_requires_both_offline_and_transfer_gate() {
        assert_eq!(
            pre_arena_recommendation(true, Some(true)),
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        );
        assert_eq!(
            pre_arena_recommendation(true, None),
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        );
        assert_eq!(
            pre_arena_recommendation(true, Some(false)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            pre_arena_recommendation(false, Some(true)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
    }

    #[test]
    fn default_arena_confirmation_request_tracks_recommendation() {
        let request = default_arena_confirmation_request(
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        )
        .expect("arena confirmation request should exist");
        assert!(request.same_seeds);
        assert_eq!(request.min_games, 10_000);
        assert!(
            default_arena_confirmation_request(DeltaQPromotionRecommendation::RejectAtOfflineGate,)
                .is_none()
        );
    }

    #[test]
    fn delta_q_stage_and_requirement_summary_follow_arena_presence() {
        assert_eq!(
            delta_q_promotion_stage(true),
            "offline_transfer_and_arena_gate"
        );
        assert_eq!(
            delta_q_promotion_stage(false),
            "offline_and_policy_transfer_gate"
        );

        let request = DeltaQArenaConfirmationRequest::default();
        let summary = delta_q_arena_requirement_summary(Some(&request));
        assert!(summary.contains("same_seeds=true"));
        assert!(summary.contains("min_games=10000"));
        assert_eq!(delta_q_arena_requirement_summary(None), "n/a");
    }

    #[test]
    fn format_probe_table_message_includes_title_selection_and_rows() {
        let selected = 64;
        let message = format_probe_table_message(
            "Probe final table",
            ProbeKind::Train,
            &[
                dummy_probe_result(ProbeKind::Train, selected, true),
                dummy_probe_result(ProbeKind::Train, 48, false),
            ],
            selected,
        );

        assert!(message.contains("Probe final table"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("train        yes       64"));
        assert!(message.contains("train        no        48"));
    }

    #[test]
    fn handle_preflight_mode_returns_validation_errors_before_runtime_work() {
        let mut config = dummy_config();
        config.num_epochs = 0;

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid config should fail before preflight runtime");
        assert_eq!(err, "num_epochs must be greater than 0");
    }

    #[test]
    fn handle_preflight_mode_rl_branch_still_validates_before_device_or_runtime_work() {
        let mut config = dummy_config();
        config.num_epochs = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid config should fail before RL preflight setup");

        assert_eq!(err, "num_epochs must be greater than 0");
    }

    #[test]
    fn handle_preflight_mode_bc_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-bc-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 BC preflight data dir");
        let output_dir = unique_test_path("bf16-bc-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
        let config_path =
            unique_test_path("bf16-bc-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 BC preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 BC preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("BF16 BC preflight should fall through the mode gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_returns_validation_errors_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Train),
        )
        .expect_err("invalid config should fail before probe runtime");
        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn handle_probe_mode_validates_rl_probe_requests_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("invalid config should fail before RL probe wrapper work");

        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn handle_probe_mode_bc_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-bc-probe-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 BC probe data dir");
        let output_dir = unique_test_path("bf16-bc-probe-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
        let config_path = unique_test_path("bf16-bc-probe-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 BC probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 BC probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("BF16 BC probe should fall through the mode gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_rl_branch_allows_bf16_past_top_level_gate() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
        config.device = "definitely-not-a-device".to_string();
        config.data_dir = unique_test_path("missing-rl-bf16-probe-data");
        config.output_dir = unique_test_path("missing-rl-bf16-probe-out");

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("RL probe mode should fall through the top-level gate");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_preflight_mode_rl_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-rl-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 RL preflight data dir");
        let output_dir = unique_test_path("bf16-rl-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.rl = Some(RlTrainConfig::default());
        config.precision_mode = hydra_train_runtime::config::PrecisionMode::Bf16Autocast;
        let config_path =
            unique_test_path("bf16-rl-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 RL preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 RL preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("BF16 RL preflight should fall through the top-level gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_rejects_invalid_device_before_rl_runtime() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.device = "definitely-not-a-device".to_string();

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid device should fail before rl preflight runtime");
        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    }

    #[test]
    fn handle_delta_q_promotion_mode_requires_baseline_checkpoint_after_bootstrap() {
        let mut config = dummy_config();
        let data_dir = unique_test_path("promotion-data");
        let output_dir = unique_test_path("promotion-out");
        std::fs::create_dir_all(&data_dir).expect("create empty promotion data dir");
        std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();

        let err = handle_delta_q_promotion_mode(Path::new("config.yaml"), config, None)
            .expect_err("promotion mode should require a baseline checkpoint after bootstrap");

        assert_eq!(
            err,
            "delta_q promotion mode requires --delta-q-baseline-checkpoint for arena confirmation"
        );
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
    }

    #[test]
    fn handle_delta_q_promotion_mode_bubbles_baseline_checkpoint_load_errors() {
        let mut config = dummy_config();
        let data_dir = unique_test_path("promotion-load-error-data");
        let output_dir = unique_test_path("promotion-load-error-out");
        let baseline_checkpoint = unique_test_path("missing-baseline-checkpoint");
        std::fs::create_dir_all(&data_dir).expect("create empty promotion data dir");
        std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();

        let err = handle_delta_q_promotion_mode(
            Path::new("config.yaml"),
            config,
            Some(baseline_checkpoint.clone()),
        )
        .expect_err("missing baseline checkpoint should fail during load");

        assert!(err.contains("failed to load delta_q baseline checkpoint"));
        assert!(err.contains(baseline_checkpoint.to_string_lossy().as_ref()));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
    }

    #[test]
    fn format_probe_table_message_preserves_selected_candidate_even_without_rows() {
        let message = format_probe_table_message("Empty probe table", ProbeKind::RlGames, &[], 256);
        assert!(message.contains("Empty probe table"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("selected"));
    }

    #[test]
    fn preflight_and_probe_status_message_helpers_render_expected_labels() {
        let probe_message = format_probe_only_status_message(dummy_probe_request(ProbeKind::Train));
        assert!(probe_message.contains("Probe-only:"));
        assert!(probe_message.contains("kind=train"));

        let rl_message = format_rl_preflight_selection_message(64, 16);
        assert!(rl_message.contains("Preflight:"));
        assert!(rl_message.contains("selected rl.games_per_batch=64 rl.microbatch_size=16"));

        let runtime = hydra_train_runtime::preflight::EffectiveRuntimeConfig {
            selected: hydra_train_runtime::preflight::SelectedRuntimeConfig {
                train_microbatch_size: 64,
                validation_microbatch_size: 32,
                accum_steps: 4,
            },
            loader: hydra_train_runtime::preflight::LoaderRuntimeConfig {
                num_threads: Some(6),
                buffer_games: 16,
                buffer_samples: 128,
                archive_queue_bound: 8,
            },
        };
        let explicit = hydra_train_runtime::preflight::ExplicitSettings {
            train_microbatch_explicit: false,
            validation_microbatch_explicit: true,
        };
        let bc_message = format_bc_preflight_selection_message(runtime, explicit);
        assert!(bc_message.contains("Preflight:"));
        assert!(bc_message.contains("saved train_mb=64 val_mb=32"));
        assert!(bc_message.contains("accum_steps=4"));
        assert!(bc_message.contains("threads=6"));
        assert!(bc_message.contains("explicit(train=false, val=true)"));
    }

    #[test]
    fn probe_mode_helpers_cover_all_probe_kinds() {
        assert_eq!(
            format_probe_only_status_detail(dummy_probe_request(ProbeKind::Train)),
            "kind=train candidate_mb=192 warmup_steps=4 measure_steps=8"
        );
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::RlGames, 512),
            "rl_games=512"
        );
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::RlMicrobatch, 32),
            "rl_microbatch=32"
        );
    }

    #[test]
    fn print_probe_table_message_shape_stays_stable_for_validation_kind() {
        let message = format_probe_table_message(
            "Validation probe table",
            ProbeKind::Validation,
            &[dummy_probe_result(ProbeKind::Validation, 32, true)],
            32,
        );

        assert!(message.contains("Validation probe table"));
        assert!(message.contains("validation"));
        assert!(message.contains("candidate_mb"));
    }

    #[test]
    fn pre_arena_and_stage_helpers_cover_all_rejecting_paths() {
        assert_eq!(
            pre_arena_recommendation(false, None),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            pre_arena_recommendation(false, Some(false)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            delta_q_promotion_stage(true),
            "offline_transfer_and_arena_gate"
        );
        assert_eq!(
            delta_q_promotion_stage(false),
            "offline_and_policy_transfer_gate"
        );
    }

    #[test]
    fn delta_q_arena_requirement_summary_reports_custom_request_fields() {
        let request = DeltaQArenaConfirmationRequest {
            min_games: 256,
            same_seeds: false,
            same_seat_rotation_schedule: false,
            same_search_budget: false,
            same_temperature: false,
            same_frozen_opponent_pool: false,
        };
        let summary = delta_q_arena_requirement_summary(Some(&request));
        assert!(summary.contains("same_seeds=false"));
        assert!(summary.contains("min_games=256"));
    }

    #[test]
    fn delta_q_promotion_formatters_cover_offline_holdout_and_gate_messages() {
        let offline = format_delta_q_offline_gate_message(
            64,
            hydra_train_exec::validation::DeltaQPromotionSnapshot {
                compared_states: 12,
                candidate_top1_agreement: 0.75,
                candidate_mean_regret: 0.2,
                baseline_mean_regret: 0.3,
                mean_decision_lift: 0.1,
                negative_lift_fraction: 0.25,
                regret_beats_baseline_rate: 0.8,
                top1_beats_baseline_rate: 0.7,
                passed: true,
            },
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
            "same_seeds=true min_games=10000",
            Path::new("/tmp/delta_q.json"),
        );
        assert!(offline.contains("DeltaQ offline gate"));
        assert!(offline.contains("samples=64"));
        assert!(offline.contains("compared=12"));
        assert!(offline.contains("next=requires_arena_confirmation"));
        assert!(offline.contains("artifact=/tmp/delta_q.json"));

        let holdout = format_delta_q_policy_holdout_message(
            hydra_train_exec::validation::DeltaQPolicyTransferSnapshot {
                compared_states: 20,
                candidate_policy_top1_to_teacher: 0.6,
                baseline_policy_top1_to_teacher: 0.5,
                candidate_policy_mean_teacher_regret: 0.2,
                baseline_policy_mean_teacher_regret: 0.25,
                candidate_beats_baseline_rate: 0.7,
                negative_transfer_fraction: 0.1,
            },
        );
        assert!(holdout.contains("DeltaQ policy-vs-teacher holdout"));
        assert!(holdout.contains("compared=20"));
        assert!(holdout.contains("policy_top1=60.00%/50.00%"));

        let gate = format_delta_q_policy_transfer_gate_message(
            true,
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        );
        assert!(gate.contains("DeltaQ policy transfer gate"));
        assert!(gate.contains("pass=true"));
        assert!(gate.contains("next=requires_arena_confirmation"));
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_runtime_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-bc-data");
        config.output_dir = unique_test_path("bc-out");

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("missing dataset should fail during BC preflight runtime");

        assert!(err.contains("failed to read config config.yaml"));
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("bc-preflight-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("file-backed output path should fail BC artifact dir creation");

        assert!(err.contains("failed to create BC artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_no_stable_train_result() {
        let data_dir = unique_test_path("bc-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BC preflight data dir");
        let output_dir = unique_test_path("bc-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("bc-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BC preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BC preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("all-failing BC preflight should bubble the no-stable train error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_bubbles_config_read_errors() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.output_dir = unique_test_path("rl-out");

        let err = handle_preflight_mode(Path::new("config.txt"), &config)
            .expect_err("invalid config extension should fail during RL preflight runtime");

        assert!(err.contains("failed to read config config.txt"));
    }

    #[test]
    fn handle_preflight_mode_rl_branch_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("rl-preflight-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();
        config.rl = Some(RlTrainConfig::default());

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("file-backed output path should fail RL artifact dir creation");

        assert!(err.contains("failed to create RL artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_rejects_invalid_device_before_slow_rl_preflight_work() {
        let data_dir = unique_test_path("rl-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty RL preflight data dir");
        let output_dir = unique_test_path("rl-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.rl = Some(RlTrainConfig::default());
        let config_path = unique_test_path("rl-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid RL preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid RL preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("invalid RL device should fail before expensive RL preflight ladder work");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_probe_ladder_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-probe-data");
        config.output_dir = unique_test_path("probe-out");

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Validation),
        )
        .expect_err("missing dataset should fail during probe ladder setup");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_probe_mode_bubbles_rl_probe_ladder_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-rl-probe-data");
        config.output_dir = unique_test_path("rl-probe-out");
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("missing dataset should fail during RL probe ladder setup");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_probe_mode_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("probe-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Train),
        )
        .expect_err("file-backed output path should fail probe artifact dir creation");

        assert!(err.contains("failed to create BC artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_no_stable_result_when_ladder_returns_only_failures() {
        let data_dir = unique_test_path("probe-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty probe data dir");
        let output_dir = unique_test_path("probe-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("probe-no-stable-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("all-failing probe ladder should bubble the no-stable-result error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_no_stable_train_result_when_ladder_returns_only_failures() {
        let data_dir = unique_test_path("probe-train-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty train probe data dir");
        let output_dir = unique_test_path("probe-train-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("probe-train-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid train probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid train probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("all-failing train probe ladder should bubble the no-stable-result error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn format_probe_only_and_rl_selection_helpers_render_exact_details() {
        let probe_message =
            format_probe_only_status_message(dummy_probe_request(ProbeKind::Validation));
        assert!(probe_message.contains("Probe-only:"));
        assert!(
            probe_message
                .contains("kind=validation candidate_mb=192 warmup_steps=4 measure_steps=8")
        );

        let rl_message = format_rl_preflight_selection_message(32, 8);
        assert!(rl_message.contains("Preflight:"));
        assert!(rl_message.contains("selected rl.games_per_batch=32 rl.microbatch_size=8"));
    }

    #[test]
    fn format_probe_table_message_supports_rl_microbatch_rows() {
        let message = format_probe_table_message(
            "RL microbatch probe table",
            ProbeKind::RlMicrobatch,
            &[dummy_probe_result(ProbeKind::RlMicrobatch, 16, true)],
            16,
        );

        assert!(message.contains("RL microbatch probe table"));
        assert!(message.contains("rl_microbatch"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("yes       16"));
    }

    #[test]
    fn delta_q_policy_transfer_gate_and_offline_messages_cover_reject_paths() {
        let gate = format_delta_q_policy_transfer_gate_message(
            false,
            DeltaQPromotionRecommendation::RejectAtOfflineGate,
        );
        assert!(gate.contains("pass=false"));
        assert!(gate.contains("next=reject_at_offline_gate"));

        let offline = format_delta_q_offline_gate_message(
            8,
            hydra_train_exec::validation::DeltaQPromotionSnapshot {
                compared_states: 4,
                candidate_top1_agreement: 0.25,
                candidate_mean_regret: 0.5,
                baseline_mean_regret: 0.4,
                mean_decision_lift: -0.1,
                negative_lift_fraction: 0.75,
                regret_beats_baseline_rate: 0.25,
                top1_beats_baseline_rate: 0.1,
                passed: false,
            },
            DeltaQPromotionRecommendation::RejectAtOfflineGate,
            "n/a",
            Path::new("/tmp/reject.json"),
        );
        assert!(offline.contains("dq_offline_gate=false"));
        assert!(offline.contains("next=reject_at_offline_gate"));
        assert!(offline.contains("artifact=/tmp/reject.json"));
    }

    #[test]
    fn default_arena_confirmation_request_returns_default_request_for_requires_confirmation() {
        let request = default_arena_confirmation_request(
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        )
        .expect("requires-confirmation should create a default arena request");

        assert_eq!(
            request.min_games,
            DeltaQArenaConfirmationRequest::default().min_games
        );
        assert_eq!(
            request.same_seeds,
            DeltaQArenaConfirmationRequest::default().same_seeds
        );
    }

    #[test]
    fn format_rl_and_bc_preflight_selection_messages_cover_small_values() {
        let rl_message = format_rl_preflight_selection_message(1, 2);
        assert!(rl_message.contains("selected rl.games_per_batch=1 rl.microbatch_size=2"));

        let runtime = hydra_train_runtime::preflight::EffectiveRuntimeConfig {
            selected: hydra_train_runtime::preflight::SelectedRuntimeConfig {
                train_microbatch_size: 8,
                validation_microbatch_size: 4,
                accum_steps: 1,
            },
            loader: hydra_train_runtime::preflight::LoaderRuntimeConfig {
                num_threads: None,
                buffer_games: 2,
                buffer_samples: 16,
                archive_queue_bound: 1,
            },
        };
        let explicit = hydra_train_runtime::preflight::ExplicitSettings {
            train_microbatch_explicit: true,
            validation_microbatch_explicit: false,
        };

        let bc_message = format_bc_preflight_selection_message(runtime, explicit);
        assert!(bc_message.contains("saved train_mb=8 val_mb=4"));
        assert!(bc_message.contains("accum_steps=1"));
        assert!(bc_message.contains("explicit(train=true, val=false)"));
    }

    #[test]
    fn format_probe_best_candidate_detail_supports_rl_games_kind() {
        let detail = format_probe_best_candidate_detail(ProbeKind::RlGames, 8);

        assert_eq!(detail, "rl_games=8");
    }

    #[test]
    fn handle_probe_mode_rl_games_request_still_validates_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlGames),
        )
        .expect_err("invalid config should fail before RL games probe wrapper work");

        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn format_probe_only_status_message_supports_rl_games_kind() {
        let message = format_probe_only_status_message(dummy_probe_request(ProbeKind::RlGames));

        assert!(message.contains("Probe-only:"));
        assert!(message.contains("kind=rl_games candidate_mb=192 warmup_steps=4 measure_steps=8"));
    }

    #[test]
    fn format_probe_best_candidate_detail_supports_train_kind() {
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::Train, 48),
            "train=48"
        );
    }
}
