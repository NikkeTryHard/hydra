use hydra_train_runtime::config::BcHyperparamConfig;
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightBenchMode, PreflightBenchReport,
    PreflightBenchRow, PreflightBenchStatus, PreflightCodec, PreflightShuffleMode, ProbeKind,
    ProbeResult, ProbeStatus, SelectedRuntimeConfig,
};

use super::*;

fn strip_ansi(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && matches!(chars.peek(), Some('[')) {
            chars.next();
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        output.push(ch);
    }
    output
}

fn probe_result(
    kind: ProbeKind,
    candidate_microbatch: usize,
    status: ProbeStatus,
    measured_samples_per_second: Option<f64>,
    elapsed_seconds: Option<f64>,
    detail: &str,
) -> ProbeResult {
    ProbeResult {
        kind,
        candidate_microbatch,
        status,
        measured_samples_per_second,
        elapsed_seconds,
        detail: detail.to_string(),
    }
}

fn assert_timestamped_message(rendered: &str, expected_message: &str) {
    let stripped = strip_ansi(rendered);
    let (prefix, message) = stripped
        .split_once("] ")
        .expect("timestamped message should contain bracketed prefix");
    assert!(prefix.starts_with('['));
    assert!(prefix.contains('T'));
    assert!(prefix.ends_with('Z'));
    assert_eq!(message, expected_message);
}

#[test]
fn bc_hyperparam_summary_includes_resolved_values() {
    let input = BcHyperparamSummaryInput {
        lr: 2.5e-4,
        min_learning_rate: 1e-6,
        weight_decay: 1e-5,
        grad_clip_norm: 1.0,
        warmup_steps: 1000,
    };
    assert_eq!(
        bc_hyperparam_summary(input),
        "lr=2.50e-4 min_lr=1.00e-6 wd=1.0e-5 clip=1.00 warmup_steps=1000"
    );
}

#[test]
fn progress_bar_builders_report_success_and_template_errors() {
    assert!(make_bar(32, "{bar:40.cyan/blue} {pos}/{len}").is_ok());
    assert!(make_spinner("{spinner} probing").is_ok());

    let bar_err = make_bar(8, "{bar:bogus}").expect_err("invalid bar template should fail");
    assert!(bar_err.contains("failed to build progress style"));

    let spinner_err =
        make_spinner("{spinner:bogus}").expect_err("invalid spinner template should fail");
    assert!(spinner_err.contains("failed to build spinner style"));
}

#[test]
fn timestamp_helpers_preserve_message_after_stripping_ansi() {
    assert_timestamped_message(&with_utc_timestamp("hello".to_string()), "hello");
    assert_timestamped_message(&timestamped(42), "42");
}

#[test]
fn advisory_line_renders_key_and_message() {
    let advisory = crate::advisory::RuntimeAdvisory::warning(
        "steady_state_cuda_bc_uses_loose_replay",
        "use shards for steady-state CUDA BC",
    );

    assert_timestamped_message(
        &format_advisory_line(&advisory),
        "Warning steady_state_cuda_bc_uses_loose_replay: use shards for steady-state CUDA BC",
    );
}

#[test]
fn phase_and_progress_helpers_render_expected_text() {
    assert_eq!(preflight_phase_label("scan"), "preflight scan");
    assert_eq!(phase_label("epoch", 0, 1), "epoch");
    assert_eq!(phase_label("epoch", 1, 3), "epoch 2/3");
    assert_eq!(
        format_progress_message(0.12345, 0.875, "lr=2.5e-4", 48.678),
        "loss=0.1235 agree=87.50% steps/s=48.68 lr=2.5e-4"
    );
}

#[test]
fn formatted_status_lines_keep_core_text_when_stripped() {
    let runtime = strip_ansi(&format_runtime_tuning_message(
        "train",
        "64".to_string(),
        0,
        0,
    ));
    assert!(runtime.contains("[preflight:runtime]"));
    assert!(runtime.contains("phase=train"));
    assert!(runtime.contains("candidate=64 option=1/1"));

    let timed = strip_ansi(&format_timed_phase_message("scan", "done", 12.345));
    assert!(timed.contains("[preflight:timing]"));
    assert!(timed.contains("phase=scan"));
    assert!(timed.contains("done elapsed=12.35s"));

    let summary = strip_ansi(&format_preflight_summary_line("Status", "ok"));
    assert!(summary.contains("Status ok"));

    let selection = strip_ansi(&format_preflight_selection_line("train_mb=64"));
    assert!(selection.contains("Preflight selected: train_mb=64"));

    let status = strip_ansi(&format_status_line("Device", "cuda:0"));
    assert!(status.contains("Device cuda:0"));

    let warning = strip_ansi(&format_warning_line("watch memory"));
    assert!(warning.contains("Warning: watch memory"));
}

#[test]
fn probe_status_and_failure_reason_cover_all_branches() {
    assert_eq!(probe_status_label(&ProbeStatus::Success), "success");
    assert_eq!(probe_status_label(&ProbeStatus::Oom), "oom");
    assert_eq!(
        probe_status_label(&ProbeStatus::BackendError),
        "backend_error"
    );
    assert_eq!(probe_status_label(&ProbeStatus::DataError), "data_error");

    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Success,
            Some(100.0),
            Some(1.0),
            "ok"
        )),
        "success"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Oom,
            None,
            None,
            "CUDA out of memory"
        )),
        "oom(cuda)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Oom,
            None,
            None,
            "tripped HOST RAM GUARD"
        )),
        "oom(host_ram_guard)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::Oom,
            None,
            None,
            "plain oom"
        )),
        "oom(generic)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::BackendError,
            None,
            None,
            "host-ram guard tripped"
        )),
        "backend_error(host_ram_guard)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::BackendError,
            None,
            None,
            "probe process status child=9"
        )),
        "backend_error(child_exit)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::BackendError,
            None,
            None,
            "misc backend failure"
        )),
        "backend_error(generic)"
    );
    assert_eq!(
        probe_failure_reason(&probe_result(
            ProbeKind::Train,
            64,
            ProbeStatus::DataError,
            None,
            None,
            "bad data"
        )),
        "data_error"
    );
}

#[test]
fn parse_probe_progress_fields_rejects_missing_prefix_and_malformed_tokens() {
    let fields = parse_probe_progress_fields(
        "probe_progress kind=train candidate_mb=64 phase=measure throughput=123.45",
    )
    .expect("well-formed line should parse");
    assert_eq!(fields.get("kind"), Some(&"train"));
    assert_eq!(fields.get("candidate_mb"), Some(&"64"));
    assert_eq!(fields.get("throughput"), Some(&"123.45"));

    assert!(parse_probe_progress_fields("kind=train candidate_mb=64").is_none());
    assert!(parse_probe_progress_fields("probe_progress kind=train broken_token").is_none());
}

#[test]
fn format_probe_progress_line_covers_each_supported_phase_and_fallbacks() {
    let scan_start = strip_ansi(
        &format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=scan_start")
            .expect("scan_start should render"),
    );
    assert!(scan_start.contains("[preflight:train] candidate_mb=64 phase=scan dataset=streaming"));

    let scan_complete_exact = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_complete sources=8 total_games=320 counts_exact=true",
            )
            .expect("scan_complete exact should render"),
        );
    assert!(scan_complete_exact.contains("sources=8 games=320"));

    let scan_complete_streaming = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_complete sources=8 counts_exact=false",
            )
            .expect("scan_complete streaming should render"),
        );
    assert!(scan_complete_streaming.contains("sources=8 games=streaming"));

    let starting = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=10",
        )
        .expect("starting should render"),
    );
    assert!(starting.contains("phase=probe warmup=10 measure=?"));

    let warmup = strip_ansi(
        &format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=warmup")
            .expect("warmup should render"),
    );
    assert!(warmup.contains("phase=warmup step=?"));

    let measure = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=measure step=3",
        )
        .expect("measure should render"),
    );
    assert!(measure.contains("phase=measure step=3 throughput=0.00 samples/s"));

    let measure_start = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=measure_start",
        )
        .expect("measure_start should render"),
    );
    assert!(measure_start.contains("phase=measure_start total_steps=?"));

    let rl_selfplay = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=rl_games candidate_mb=16 phase=rl_selfplay",
        )
        .expect("rl_selfplay should render"),
    );
    assert!(rl_selfplay.contains("phase=rl_selfplay running cooperative self-play + learner step"));

    let done = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=done elapsed=1.25",
        )
        .expect("done should render"),
    );
    assert!(done.contains("phase=done throughput=0.00 samples/s elapsed=1.25s"));

    assert!(
        format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=unknown")
            .is_none()
    );
    assert!(format_probe_progress_line("probe_progress kind=train phase=measure").is_none());
}

#[test]
fn format_probe_progress_line_covers_init_sub_stages() {
    let init_model = strip_ansi(
        &format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=init_model")
            .expect("init_model should render"),
    );
    assert!(init_model.contains("[preflight:train] candidate_mb=64 phase=init_model"));
    assert!(init_model.contains("initializing backbone + heads"));

    let init_optimizer = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=init_optimizer",
        )
        .expect("init_optimizer should render"),
    );
    assert!(init_optimizer.contains("phase=init_optimizer creating optimizer"));

    let init_loss = strip_ansi(
        &format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=init_loss")
            .expect("init_loss should render"),
    );
    assert!(init_loss.contains("phase=init_loss building loss functions"));

    let init_cuda_staging = strip_ansi(
        &format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=init_cuda_staging",
        )
        .expect("init_cuda_staging should render"),
    );
    assert!(init_cuda_staging.contains("phase=init_cuda_staging"));
    assert!(init_cuda_staging.contains("allocating CUDA staging buffers"));

    let init_ready = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8",
            )
            .expect("init_ready should render"),
        );
    assert!(init_ready.contains("phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8"));

    let starting = format_probe_progress_line(
        "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=2 measure_steps=3",
    );
    assert!(starting.is_some());
}

#[test]
fn format_probe_spinner_message_covers_init_sub_stages() {
    let model =
        format_probe_spinner_message("probe_progress kind=train candidate_mb=64 phase=init_model")
            .expect("spinner init_model");
    assert!(model.contains("initializing model (backbone + heads)"));

    let optimizer = format_probe_spinner_message(
        "probe_progress kind=train candidate_mb=64 phase=init_optimizer",
    )
    .expect("spinner init_optimizer");
    assert!(optimizer.contains("creating optimizer"));

    let loss =
        format_probe_spinner_message("probe_progress kind=train candidate_mb=64 phase=init_loss")
            .expect("spinner init_loss");
    assert!(loss.contains("building loss functions"));

    let cuda = format_probe_spinner_message(
        "probe_progress kind=train candidate_mb=64 phase=init_cuda_staging",
    )
    .expect("spinner init_cuda_staging");
    assert!(cuda.contains("CUDA staging"));

    let ready = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8",
        )
        .expect("spinner init_ready");
    assert!(ready.contains("init complete"));
    assert!(ready.contains("model=142ms"));

    let starting =
        format_probe_spinner_message("probe_progress kind=train candidate_mb=64 phase=starting")
            .expect("spinner starting backward compat");
    assert!(starting.contains("building model"));
}

#[test]
fn format_probe_status_line_handles_success_oom_and_backend_error_cases() {
    let success = strip_ansi(&format_probe_status_line(&probe_result(
        ProbeKind::Validation,
        32,
        ProbeStatus::Success,
        Some(456.789),
        Some(2.345),
        "",
    )));
    assert!(success.contains(
        "[validation] candidate_mb=32 outcome=success throughput=456.79 samples/s elapsed=2.35s"
    ));

    let oom = strip_ansi(&format_probe_status_line(&probe_result(
        ProbeKind::RlGames,
        8,
        ProbeStatus::Oom,
        None,
        None,
        "",
    )));
    assert!(oom.contains(
        "[rl_games] candidate_mb=8 outcome=oom(generic) next=smaller_microbatch detail=n/a"
    ));

    let backend = strip_ansi(&format_probe_status_line(&probe_result(
        ProbeKind::RlMicrobatch,
        4,
        ProbeStatus::BackendError,
        None,
        None,
        "probe process status child=137",
    )));
    assert!(backend.contains(
            "[rl_microbatch] candidate_mb=4 outcome=backend_error(child_exit) detail=probe process status child=137"
        ));
}

#[test]
fn format_probe_results_table_renders_selection_averages_and_missing_metrics() {
    let table = format_probe_results_table(
        ProbeKind::Train,
        &[
            probe_result(
                ProbeKind::Train,
                64,
                ProbeStatus::Success,
                Some(400.0),
                Some(2.0),
                "",
            ),
            probe_result(
                ProbeKind::Train,
                64,
                ProbeStatus::Success,
                Some(500.0),
                Some(4.0),
                "",
            ),
            probe_result(
                ProbeKind::Train,
                32,
                ProbeStatus::DataError,
                None,
                None,
                "bad archive",
            ),
        ],
        Some(64),
    );

    let lines: Vec<_> = table.lines().collect();
    assert!(lines[0].contains("kind         selected  candidate_mb"));
    assert!(lines[2].contains("train        yes       64"));
    assert!(lines[2].contains("success"));
    assert!(lines[2].contains("450.00"));
    assert!(lines[2].contains("3.00"));
    assert!(lines[3].contains("train        no        32"));
    assert!(lines[3].contains("data_error"));
    assert!(lines[3].contains("-"));
}

#[test]
fn unsafe_preflight_math_summary_reports_fresh_start_policy() {
    let unsafe_runtime = EffectiveRuntimeConfig {
        selected: SelectedRuntimeConfig {
            train_microbatch_size: 128,
            validation_microbatch_size: 32,
            accum_steps: 2,
            unsafe_selected_batch_size: Some(512),
            unsafe_selected_learning_rate: Some(5.0e-4),
            unsafe_selected_min_learning_rate: Some(2.0e-6),
            unsafe_selected_warmup_steps: Some(2000),
        },
        loader: LoaderRuntimeConfig {
            num_threads: None,
            buffer_games: 128,
            buffer_samples: 4096,
            archive_queue_bound: 16,
        },
        requested_precision: hydra_train_runtime::config::PrecisionMode::Bf16Autocast,
        effective_precision:
            hydra_train_runtime::config::EffectivePrecision::Fp32NoopForBf16Request,
    };
    let bc = BcHyperparamConfig {
        learning_rate: 2.5e-4,
        min_learning_rate: 1.0e-6,
        warmup_steps: 1000,
        ..Default::default()
    };
    let math_summary = unsafe_preflight_math_summary(unsafe_runtime, &bc)
        .expect("unsafe runtime should report math summary");
    assert!(math_summary.contains("selected_batch=512"));
    assert!(math_summary.contains("selected_lr=5.00e-4"));
    assert!(math_summary.contains("selected_min_lr=2.00e-6"));
    assert!(math_summary.contains("selected_warmup_steps=2000"));
    assert!(math_summary.contains("apply=fresh_start_only"));
    assert!(math_summary.contains("resume=ignored_or_refused_if_checkpoint_contract_mismatch"));
}

#[test]
fn preflight_bench_markdown_table_matches_snapshot() {
    let report = PreflightBenchReport {
        schema_version: 1,
        rows: vec![PreflightBenchRow {
            index: 0,
            status: PreflightBenchStatus::Pass,
            device: "cpu".to_string(),
            mode: PreflightBenchMode::LoaderOnly,
            batch_size: 1024,
            ring_batches: 2,
            loader_threads: 1,
            prefetch_batches: 1,
            shuffle: PreflightShuffleMode::None,
            codec: PreflightCodec::None,
            samples_per_second: Some(1234.5),
            mib_per_second: Some(67.89),
            p50_batch_ms: Some(1.2),
            p95_batch_ms: Some(3.4),
            producer_wait_ratio: Some(0.01),
            consumer_wait_ratio: Some(0.02),
            disk_wait_ratio: Some(0.03),
            gpu_input_wait_ratio: Some(0.04),
            cpu_user_seconds: Some(5.6),
            cpu_system_seconds: Some(7.8),
            error: None,
        }],
        total_elapsed_seconds: 9.0,
    };
    assert_eq!(
        format_preflight_bench_markdown_table(&report),
        "| idx | status | device | mode | batch | ring | threads | prefetch | shuffle | codec | samples/s | MiB/s | p50 ms | p95 ms | producer wait % | consumer wait % | disk wait % | gpu input wait % | cpu user s | cpu sys s | error |\n|---:|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n| 0 | pass | cpu | loader_only | 1024 | 2 | 1 | 1 | none | none | 1234.50 | 67.89 | 1.20 | 3.40 | 1.00 | 2.00 | 3.00 | 4.00 | 5.60 | 7.80 |  |"
    );
}

#[test]
fn preflight_generated_output_contains_no_authority_words() {
    let report = PreflightBenchReport {
        schema_version: 1,
        rows: Vec::new(),
        total_elapsed_seconds: 0.0,
    };
    let output = format_preflight_bench_markdown_table(&report);
    for forbidden in [
        "selected",
        "cache_hit",
        "runtime",
        "cache_key",
        "saved",
        "best",
        "recommended",
    ] {
        assert!(
            !output.contains(forbidden),
            "forbidden word in output: {forbidden}"
        );
    }
}
