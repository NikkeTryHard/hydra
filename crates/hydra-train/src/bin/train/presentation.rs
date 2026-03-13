use colored::Colorize;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, ProbeKind, ProbeResult, ProbeStatus,
};

use super::artifacts::BcArtifactPaths;
use super::config::TrainConfig;
use super::config::display_num_threads;
use super::preflight_runtime::summarize_probe_results;
use super::progress::BannerStats;
use hydra_train::training::bc::BCTrainerConfig;

pub(super) fn make_bar(len: u64, template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new(len);
    pb.set_draw_target(ProgressDrawTarget::stdout());
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build progress style: {err}"))?
        .progress_chars("=> ");
    pb.set_style(style);
    Ok(pb)
}

pub(super) fn make_spinner(template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new_spinner();
    pb.set_draw_target(ProgressDrawTarget::stdout());
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build spinner style: {err}"))?
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    pb.set_style(style);
    pb.enable_steady_tick(Duration::from_millis(120));
    Ok(pb)
}

pub(super) fn preflight_phase_label(phase: &str) -> String {
    format!("preflight {phase}")
}

fn utc_log_prefix() -> String {
    let ts = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
    format!("[{ts}]")
}

pub(super) fn with_utc_timestamp(message: String) -> String {
    format!("{} {}", utc_log_prefix().dimmed(), message)
}

pub(super) fn timestamped(message: impl std::fmt::Display) -> String {
    with_utc_timestamp(message.to_string())
}

pub(super) fn format_runtime_tuning_message(
    knob: &str,
    candidate: String,
    index: usize,
    total: usize,
) -> String {
    with_utc_timestamp(format!(
        "{} {} {}",
        "[preflight:runtime]".bold().cyan(),
        format!("phase={knob}").yellow(),
        format!(
            "candidate={candidate} option={}/{}",
            index + 1,
            total.max(1)
        )
        .white(),
    ))
}

pub(super) fn format_runtime_tuning_result(
    knob: &str,
    candidate: String,
    throughput: f64,
    best_candidate: String,
    best_throughput: f64,
) -> String {
    with_utc_timestamp(format!(
        "{} {} {} {}",
        "[preflight:runtime]".bold().cyan(),
        format!("phase={knob}").yellow(),
        format!("candidate={candidate} throughput={throughput:.2} samples/s").green(),
        format!("best={} ({best_throughput:.2} samples/s)", best_candidate).magenta(),
    ))
}

pub(super) fn format_timed_phase_message(
    phase: &str,
    detail: &str,
    elapsed_seconds: f64,
) -> String {
    with_utc_timestamp(format!(
        "{} {} {}",
        "[preflight:timing]".bold().cyan(),
        format!("phase={phase}").yellow(),
        format!("{detail} elapsed={elapsed_seconds:.2}s").green(),
    ))
}

pub(super) fn format_preflight_summary_line(label: &str, detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        label.bold().cyan(),
        detail.to_string().yellow()
    ))
}

pub(super) fn format_preflight_selection_line(detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        "Preflight selected:".bold().cyan(),
        detail.to_string().green()
    ))
}

pub(super) fn format_status_line(label: &str, detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        label.bold().cyan(),
        detail.to_string().yellow()
    ))
}

pub(super) fn format_warning_line(detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        "Warning:".bold().yellow(),
        detail.to_string().yellow()
    ))
}

pub(super) fn phase_label(prefix: &str, epoch_index: usize, num_epochs: usize) -> String {
    if num_epochs <= 1 {
        prefix.to_string()
    } else {
        format!("{prefix} {}/{}", epoch_index + 1, num_epochs)
    }
}

pub(super) fn format_progress_message(
    loss: f64,
    agreement: f64,
    lr_message: &str,
    step_rate: f64,
) -> String {
    format!(
        "loss={loss:.4} agree={:.2}% steps/s={step_rate:.2} {lr_message}",
        agreement * 100.0
    )
}

pub(super) fn model_kind(config: &HydraModelConfig) -> &'static str {
    if config.is_learner() {
        "learner"
    } else {
        "actor"
    }
}

pub(super) fn bc_hyperparam_summary(train_cfg: &BCTrainerConfig) -> String {
    format!(
        "lr={:.2e} min_lr={:.2e} wd={:.1e} clip={:.2} warmup_steps={}",
        train_cfg.lr,
        train_cfg.min_learning_rate,
        train_cfg.weight_decay,
        train_cfg.grad_clip_norm,
        train_cfg.warmup_steps,
    )
}

fn print_header_block(title: &str) {
    println!();
    println!();
    println!("{}", title.bold().cyan());
}

fn print_banner_field(label: &str, value: impl std::fmt::Display) {
    println!("  {} {}", format!("{label}:").white(), value);
}

fn probe_status_label(status: &ProbeStatus) -> &'static str {
    match status {
        ProbeStatus::Success => "success",
        ProbeStatus::Oom => "oom",
        ProbeStatus::BackendError => "backend_error",
        ProbeStatus::DataError => "data_error",
    }
}

fn parse_probe_progress_fields(line: &str) -> Option<std::collections::BTreeMap<&str, &str>> {
    let trimmed = line.trim();
    let payload = trimmed.strip_prefix("probe_progress ")?;
    let mut fields = std::collections::BTreeMap::new();
    for token in payload.split_whitespace() {
        let (key, value) = token.split_once('=')?;
        fields.insert(key, value);
    }
    Some(fields)
}

pub(super) fn format_probe_progress_line(line: &str) -> Option<String> {
    let fields = parse_probe_progress_fields(line)?;
    let kind = *fields.get("kind")?;
    let candidate = *fields.get("candidate_mb")?;
    let phase = *fields.get("phase")?;
    let prefix = format!("[preflight:{kind}]").cyan().bold();
    let label = format!("candidate_mb={candidate}").yellow();

    let message = match phase {
        "scan_start" => format!(
            "{} {} {}",
            prefix,
            label,
            "phase=scan dataset=streaming".white()
        ),
        "scan_complete" => {
            let sources = fields.get("sources").copied().unwrap_or("?");
            let total_games = fields.get("total_games").copied().unwrap_or("?");
            let counts_exact = fields.get("counts_exact").copied().unwrap_or("false");
            let counts = if counts_exact == "true" {
                format!("sources={sources} games={total_games}")
            } else {
                format!("sources={sources} games=streaming")
            };
            format!("{} {} {}", prefix, label, counts.green())
        }
        "starting" => format!(
            "{} {} {}",
            prefix,
            label,
            format!(
                "phase=probe warmup={} measure={}",
                fields.get("warmup_steps").copied().unwrap_or("?"),
                fields.get("measure_steps").copied().unwrap_or("?")
            )
            .white()
        ),
        "warmup" => format!(
            "{} {} {}",
            prefix,
            label,
            format!(
                "phase=warmup step={}",
                fields.get("step").copied().unwrap_or("?"),
            )
            .dimmed()
        ),
        "measure" => format!(
            "{} {} {}",
            prefix,
            label,
            format!(
                "phase=measure step={} throughput={} samples/s",
                fields.get("step").copied().unwrap_or("?"),
                fields.get("throughput").copied().unwrap_or("0.00")
            )
            .green()
        ),
        "measure_start" => format!(
            "{} {} {}",
            prefix,
            label,
            format!(
                "phase=measure_start total_steps={}",
                fields.get("total_steps").copied().unwrap_or("?")
            )
            .dimmed()
        ),
        "done" => format!(
            "{} {} {}",
            prefix,
            label,
            format!(
                "phase=done throughput={} samples/s elapsed={}s",
                fields.get("throughput").copied().unwrap_or("0.00"),
                fields.get("elapsed").copied().unwrap_or("0.00")
            )
            .green()
        ),
        _ => return None,
    };

    Some(with_utc_timestamp(message))
}

pub(super) fn print_preflight_banner(title: &str, config: &TrainConfig, device_name: &str) {
    print_header_block(title);
    print_banner_field("Device", device_name.green());
    print_banner_field("Dataset", config.data_dir.display().to_string().green());
    print_banner_field(
        "Optimizer batch",
        format!("{} samples", config.batch_size).yellow(),
    );
    print_banner_field(
        "Runtime defaults",
        format!(
            "train_mb={} val_mb={} threads={} buffer_games={} buffer_samples={} archive_queue_bound={}",
            config.microbatch_size.unwrap_or(config.batch_size),
            config
                .validation_microbatch_size
                .unwrap_or(config.microbatch_size.unwrap_or(config.batch_size)),
            display_num_threads(config.num_threads),
            config.buffer_games,
            config.buffer_samples,
            config.archive_queue_bound,
        )
        .yellow(),
    );
    println!();
}

pub(super) fn format_probe_status_line(result: &ProbeResult) -> String {
    match result.status {
        ProbeStatus::Success => with_utc_timestamp(
            format!(
                "[{}] candidate_mb={} outcome=success throughput={:.2} samples/s elapsed={:.2}s",
                match result.kind {
                    ProbeKind::Train => "train",
                    ProbeKind::Validation => "validation",
                },
                result.candidate_microbatch,
                result.measured_samples_per_second.unwrap_or(0.0),
                result.elapsed_seconds.unwrap_or(0.0)
            )
            .green()
            .to_string(),
        ),
        ProbeStatus::Oom => with_utc_timestamp(
            format!(
                "[{}] candidate_mb={} outcome=oom next=smaller_microbatch",
                match result.kind {
                    ProbeKind::Train => "train",
                    ProbeKind::Validation => "validation",
                },
                result.candidate_microbatch,
            )
            .red()
            .to_string(),
        ),
        _ => with_utc_timestamp(
            format!(
                "[{}] candidate_mb={} outcome={} detail={}",
                match result.kind {
                    ProbeKind::Train => "train",
                    ProbeKind::Validation => "validation",
                },
                result.candidate_microbatch,
                probe_status_label(&result.status),
                result.detail
            )
            .red()
            .to_string(),
        ),
    }
}

pub(super) fn format_probe_results_table(
    kind: ProbeKind,
    results: &[ProbeResult],
    selected_candidate: Option<usize>,
) -> String {
    let kind_label = match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
    };
    let summaries = summarize_probe_results(results);
    let mut lines = vec![format!(
        "kind         selected  candidate_mb  attempts  status         avg_throughput(samples/s)  avg_elapsed(s)"
    )];
    lines.push(
        "------------ ---------  ------------  --------  -------------  -------------------------  --------------".to_string(),
    );
    for summary in summaries {
        let selected = if selected_candidate == Some(summary.candidate_microbatch) {
            "yes"
        } else {
            "no"
        };
        let throughput = summary
            .average_samples_per_second
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let elapsed = summary
            .average_elapsed_seconds
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "{kind_label:<12} {selected:<9} {candidate:<12} {attempts:<8} {status:<13} {throughput:>25} {elapsed:>15}",
            candidate = summary.candidate_microbatch,
            attempts = summary.attempts,
            status = probe_status_label(&summary.status),
        ));
    }
    lines.join("\n")
}

pub(super) fn explicit_preflight_summary(
    runtime: EffectiveRuntimeConfig,
    explicit: ExplicitSettings,
) -> String {
    format!(
        "saved train_mb={} val_mb={} accum_steps={} threads={} buffer_games={} buffer_samples={} archive_queue_bound={} explicit(train={}, val={})",
        runtime.selected.train_microbatch_size,
        runtime.selected.validation_microbatch_size,
        runtime.selected.accum_steps,
        display_num_threads(runtime.loader.num_threads),
        runtime.loader.buffer_games,
        runtime.loader.buffer_samples,
        runtime.loader.archive_queue_bound,
        explicit.train_microbatch_explicit,
        explicit.validation_microbatch_explicit,
    )
}

pub(super) fn explicit_preflight_recommendation() -> String {
    "using config runtime only; run train <config.yaml> --preflight to tune this machine before training"
        .to_string()
}

pub(super) fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &BannerStats,
    train_cfg: &BCTrainerConfig,
) {
    print_header_block("Hydra BC trainer");
    print_banner_field(
        "Model",
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green(),
    );
    print_banner_field("Device", device_name.green());
    print_banner_field(
        "Dataset",
        if stats.counts_exact {
            format!(
                "{} ({} sources, {} games)",
                config.data_dir.display(),
                stats.total_sources,
                stats.total_games
            )
        } else {
            format!(
                "{} ({} sources, archive counts deferred)",
                config.data_dir.display(),
                stats.total_sources,
            )
        }
        .green(),
    );
    print_banner_field(
        "Train",
        if stats.counts_exact {
            format!(
                "{} games | Val: {} games",
                stats.train_count, stats.val_count
            )
        } else {
            "streaming split, counts estimated while loading".to_string()
        }
        .green(),
    );
    print_banner_field(
        "Buffer",
        format!(
            "{} samples (max {} games, archive_queue_bound={}, threads={})",
            config.buffer_samples,
            config.buffer_games,
            config.archive_queue_bound,
            display_num_threads(config.num_threads)
        )
        .yellow(),
    );
    print_banner_field(
        "Optimizer batch",
        format!(
            "{} ({} x {} accum)",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps
        )
        .yellow(),
    );
    print_banner_field("BC hyperparams", bc_hyperparam_summary(train_cfg).yellow());
    print_banner_field("Epochs", config.num_epochs.to_string().yellow());
    print_banner_field(
        "Schedule",
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            train_cfg.warmup_steps,
            config
                .max_train_steps
                .map(|steps| steps.to_string())
                .unwrap_or_else(|| "epoch-derived".to_string())
        )
        .yellow(),
    );
    print_banner_field("Output", artifacts.root.display().to_string().green());
    print_banner_field(
        "TBoard",
        if config.tensorboard {
            artifacts.tb_session_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        },
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::bc_hyperparam_summary;
    use hydra_train::model::HydraModelConfig;
    use hydra_train::training::bc::BCTrainerConfig;

    #[test]
    fn bc_hyperparam_summary_includes_resolved_values() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::learner())
            .with_lr(2.5e-4)
            .with_min_learning_rate(1e-6)
            .with_weight_decay(1e-5)
            .with_grad_clip_norm(1.0)
            .with_warmup_steps(1000);
        assert_eq!(
            bc_hyperparam_summary(&cfg),
            "lr=2.50e-4 min_lr=1.00e-6 wd=1.0e-5 clip=1.00 warmup_steps=1000"
        );
    }
}
