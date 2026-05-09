use colored::Colorize;
use std::borrow::Cow;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, ProbeKind, ProbeResult, ProbeStatus,
};

use super::advisory::{AdvisorySeverity, RuntimeAdvisory};
use super::artifacts::BcArtifactPaths;
use super::config::TrainConfig;
use super::config::display_num_threads;
use super::probe_summary::probe_summary_iter;
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

pub(super) fn format_advisory_line(advisory: &RuntimeAdvisory) -> String {
    let severity = match advisory.severity {
        AdvisorySeverity::Info => "Info".bold().cyan(),
        AdvisorySeverity::Warning => "Warning".bold().yellow(),
    };
    let detail = format!("{}: {}", advisory.key, advisory.message);
    with_utc_timestamp(format!("{} {}", severity, detail.yellow()))
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

fn probe_kind_label(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
        ProbeKind::RlGames => "rl_games",
        ProbeKind::RlMicrobatch => "rl_microbatch",
    }
}

fn probe_status_label(status: &ProbeStatus) -> &'static str {
    match status {
        ProbeStatus::Success => "success",
        ProbeStatus::Oom => "oom",
        ProbeStatus::BackendError => "backend_error",
        ProbeStatus::DataError => "data_error",
    }
}

fn probe_failure_reason(result: &ProbeResult) -> &'static str {
    let detail = result.detail.to_ascii_lowercase();
    match result.status {
        ProbeStatus::Success => "success",
        ProbeStatus::Oom => {
            if detail.contains("cuda") || detail.contains("libtorch") || detail.contains("cudnn") {
                "oom(cuda)"
            } else if detail.contains("host-ram guard") || detail.contains("host ram guard") {
                "oom(host_ram_guard)"
            } else {
                "oom(generic)"
            }
        }
        ProbeStatus::BackendError => {
            if detail.contains("host-ram guard") || detail.contains("host ram guard") {
                "backend_error(host_ram_guard)"
            } else if detail.contains("probe process status") || detail.contains("child") {
                "backend_error(child_exit)"
            } else {
                "backend_error(generic)"
            }
        }
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

fn sanitize_probe_progress_line(line: &str) -> Cow<'_, str> {
    if line.contains(" samples/s") {
        Cow::Owned(line.replace(" samples/s", ""))
    } else {
        Cow::Borrowed(line)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeProgressPhase {
    ScanStart,
    ScanComplete,
    InitModel,
    InitOptimizer,
    InitLoss,
    InitCudaStaging,
    InitReady,
    Starting,
    Warmup,
    MeasureStart,
    Measure,
    Done,
    RlSelfplay,
}

impl ProbeProgressPhase {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "scan_start" => Some(Self::ScanStart),
            "scan_complete" => Some(Self::ScanComplete),
            "init_model" => Some(Self::InitModel),
            "init_optimizer" => Some(Self::InitOptimizer),
            "init_loss" => Some(Self::InitLoss),
            "init_cuda_staging" => Some(Self::InitCudaStaging),
            "init_ready" => Some(Self::InitReady),
            "starting" => Some(Self::Starting),
            "warmup" => Some(Self::Warmup),
            "measure_start" => Some(Self::MeasureStart),
            "measure" => Some(Self::Measure),
            "done" => Some(Self::Done),
            "rl_selfplay" => Some(Self::RlSelfplay),
            _ => None,
        }
    }
}

struct ProbeProgressEvent {
    kind: String,
    candidate_mb: String,
    phase: ProbeProgressPhase,
    fields: std::collections::BTreeMap<String, String>,
}

impl ProbeProgressEvent {
    fn parse(line: &str) -> Option<Self> {
        let sanitized = sanitize_probe_progress_line(line);
        let borrowed_fields = parse_probe_progress_fields(&sanitized)?;
        let kind = borrowed_fields.get("kind")?.to_string();
        let candidate_mb = borrowed_fields.get("candidate_mb")?.to_string();
        let phase = ProbeProgressPhase::parse(borrowed_fields.get("phase").copied()?)?;
        let fields = borrowed_fields
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        Some(Self {
            kind,
            candidate_mb,
            phase,
            fields,
        })
    }

    fn field(&self, key: &str) -> &str {
        self.fields.get(key).map(String::as_str).unwrap_or("?")
    }

    fn field_or<'b>(&'b self, key: &str, default: &'b str) -> &'b str {
        self.fields.get(key).map(String::as_str).unwrap_or(default)
    }

    fn spinner_prefix(&self) -> String {
        format!("[preflight:{}] mb={}", self.kind, self.candidate_mb)
    }

    fn spinner_message(&self) -> String {
        match self.phase {
            ProbeProgressPhase::ScanStart => "scanning dataset...".to_string(),
            ProbeProgressPhase::ScanComplete => {
                let sources = self.field("sources");
                let games = if self.fields.get("counts_exact").map(String::as_str) == Some("true") {
                    self.field("total_games")
                } else {
                    "streaming"
                };
                format!("dataset: {sources} sources, {games} games")
            }
            ProbeProgressPhase::InitModel => "initializing model (backbone + heads)...".to_string(),
            ProbeProgressPhase::InitOptimizer => "creating optimizer...".to_string(),
            ProbeProgressPhase::InitLoss => "building loss functions...".to_string(),
            ProbeProgressPhase::InitCudaStaging => "allocating CUDA staging buffers...".to_string(),
            ProbeProgressPhase::InitReady => format!(
                "init complete (model={}ms opt={}ms loss={}ms)",
                self.field("model_ms"),
                self.field("optimizer_ms"),
                self.field("loss_ms")
            ),
            ProbeProgressPhase::Starting => "building model...".to_string(),
            ProbeProgressPhase::Warmup => format!("warmup step {}", self.field("step")),
            ProbeProgressPhase::MeasureStart => {
                format!("measuring (0/{})", self.field("total_steps"))
            }
            ProbeProgressPhase::Measure => format!(
                "measure step {} @ {} samples/s",
                self.field("step"),
                self.field_or("throughput", "0.00")
            ),
            ProbeProgressPhase::Done => {
                format!(
                    "finalizing @ {} samples/s...",
                    self.field_or("throughput", "0.00")
                )
            }
            ProbeProgressPhase::RlSelfplay => "selfplay...".to_string(),
        }
    }

    fn progress_message(&self) -> String {
        let prefix = format!("[preflight:{}]", self.kind).cyan().bold();
        let label = format!("candidate_mb={}", self.candidate_mb).yellow();
        match self.phase {
            ProbeProgressPhase::ScanStart => {
                format!(
                    "{} {} {}",
                    prefix,
                    label,
                    "phase=scan dataset=streaming".white()
                )
            }
            ProbeProgressPhase::ScanComplete => {
                let sources = self.field("sources");
                let counts = if self.fields.get("counts_exact").map(String::as_str) == Some("true")
                {
                    format!("sources={sources} games={}", self.field("total_games"))
                } else {
                    format!("sources={sources} games=streaming")
                };
                format!("{} {} {}", prefix, label, counts.green())
            }
            ProbeProgressPhase::InitModel => format!(
                "{} {} {}",
                prefix,
                label,
                "phase=init_model initializing backbone + heads".white()
            ),
            ProbeProgressPhase::InitOptimizer => format!(
                "{} {} {}",
                prefix,
                label,
                "phase=init_optimizer creating optimizer".white()
            ),
            ProbeProgressPhase::InitLoss => format!(
                "{} {} {}",
                prefix,
                label,
                "phase=init_loss building loss functions".white()
            ),
            ProbeProgressPhase::InitCudaStaging => format!(
                "{} {} {}",
                prefix,
                label,
                "phase=init_cuda_staging allocating CUDA staging buffers".white()
            ),
            ProbeProgressPhase::InitReady => format!(
                "{} {} {}",
                prefix,
                label,
                format!(
                    "phase=init_ready model_ms={} optimizer_ms={} loss_ms={}",
                    self.field("model_ms"),
                    self.field("optimizer_ms"),
                    self.field("loss_ms"),
                )
                .green()
            ),
            ProbeProgressPhase::Starting => format!(
                "{} {} {}",
                prefix,
                label,
                format!(
                    "phase=probe warmup={} measure={}",
                    self.field("warmup_steps"),
                    self.field("measure_steps")
                )
                .white()
            ),
            ProbeProgressPhase::Warmup => format!(
                "{} {} {}",
                prefix,
                label,
                format!("phase=warmup step={}", self.field("step")).dimmed()
            ),
            ProbeProgressPhase::Measure => format!(
                "{} {} {}",
                prefix,
                label,
                format!(
                    "phase=measure step={} throughput={} samples/s",
                    self.field("step"),
                    self.field_or("throughput", "0.00")
                )
                .green()
            ),
            ProbeProgressPhase::MeasureStart => format!(
                "{} {} {}",
                prefix,
                label,
                format!(
                    "phase=measure_start total_steps={}",
                    self.field("total_steps")
                )
                .dimmed()
            ),
            ProbeProgressPhase::RlSelfplay => format!(
                "{} {} {}",
                prefix,
                label,
                "phase=rl_selfplay running cooperative self-play + learner step".bright_blue()
            ),
            ProbeProgressPhase::Done => format!(
                "{} {} {}",
                prefix,
                label,
                format!(
                    "phase=done throughput={} samples/s elapsed={}s",
                    self.field_or("throughput", "0.00"),
                    self.field_or("elapsed", "0.00")
                )
                .green()
            ),
        }
    }
}

pub(super) fn format_probe_spinner_message(line: &str) -> Option<String> {
    let event = ProbeProgressEvent::parse(line)?;
    Some(format!(
        "{} {}",
        event.spinner_prefix(),
        event.spinner_message()
    ))
}

#[cfg(not(test))]
fn format_probe_spinner_rate(samples_per_second: f64) -> String {
    if (samples_per_second.fract()).abs() < 0.005 {
        format!("{samples_per_second:.0}")
    } else {
        format!("{samples_per_second:.2}")
    }
}

#[cfg(not(test))]
fn format_probe_spinner_elapsed(elapsed_seconds: f64) -> String {
    if elapsed_seconds >= 10.0 {
        format!("{elapsed_seconds:.0}s")
    } else {
        format!("{elapsed_seconds:.1}s")
    }
}

#[cfg(not(test))]
pub(super) fn format_probe_spinner_finish_message(
    result: &ProbeResult,
    fallback_elapsed_seconds: f64,
) -> String {
    let kind = probe_kind_label(result.kind);
    let elapsed = format_probe_spinner_elapsed(
        result
            .elapsed_seconds
            .unwrap_or(fallback_elapsed_seconds.max(0.0)),
    );

    match result.status {
        ProbeStatus::Success => format!(
            "{} [{}] mb={} success @ {} samples/s ({elapsed})",
            "✔".green(),
            kind,
            result.candidate_microbatch,
            format_probe_spinner_rate(result.measured_samples_per_second.unwrap_or(0.0))
        ),
        ProbeStatus::Oom => format!(
            "{} [{}] mb={} oom ({elapsed})",
            "✘".red(),
            kind,
            result.candidate_microbatch,
        ),
        ProbeStatus::BackendError => format!(
            "{} [{}] mb={} backend error ({elapsed})",
            "✘".red(),
            kind,
            result.candidate_microbatch,
        ),
        ProbeStatus::DataError => format!(
            "{} [{}] mb={} data error ({elapsed})",
            "✘".red(),
            kind,
            result.candidate_microbatch,
        ),
    }
}

pub(super) fn format_probe_progress_line(line: &str) -> Option<String> {
    let event = ProbeProgressEvent::parse(line)?;
    Some(with_utc_timestamp(event.progress_message()))
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
                probe_kind_label(result.kind),
                result.candidate_microbatch,
                result.measured_samples_per_second.unwrap_or(0.0),
                result.elapsed_seconds.unwrap_or(0.0)
            )
            .green()
            .to_string(),
        ),
        ProbeStatus::Oom => with_utc_timestamp(
            format!(
                "[{}] candidate_mb={} outcome={} next=smaller_microbatch detail={}",
                probe_kind_label(result.kind),
                result.candidate_microbatch,
                probe_failure_reason(result),
                if result.detail.is_empty() {
                    "n/a"
                } else {
                    &result.detail
                },
            )
            .red()
            .to_string(),
        ),
        _ => with_utc_timestamp(
            format!(
                "[{}] candidate_mb={} outcome={} detail={}",
                probe_kind_label(result.kind),
                result.candidate_microbatch,
                probe_failure_reason(result),
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
        ProbeKind::RlGames => "rl_games",
        ProbeKind::RlMicrobatch => "rl_microbatch",
    };
    let mut lines = vec![format!(
        "kind         selected  candidate_mb  attempts  status                       avg_throughput(samples/s)  avg_elapsed(s)"
    )];
    lines.push(
        "------------ ---------  ------------  --------  ---------------------------  -------------------------  --------------".to_string(),
    );
    for summary in probe_summary_iter(results) {
        let selected = if selected_candidate == Some(summary.candidate_microbatch) {
            "yes"
        } else {
            "no"
        };
        let status = probe_status_label(&summary.status);
        let throughput = summary
            .average_samples_per_second
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let elapsed = summary
            .average_elapsed_seconds
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "{kind_label:<12} {selected:<9} {candidate:<12} {attempts:<8} {status:<27} {throughput:>25} {elapsed:>15}",
            candidate = summary.candidate_microbatch,
            attempts = summary.attempts,
            status = status,
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
pub(super) fn cuda_graph_replay_label() -> &'static str {
    "production_off_probe_only"
}

pub(super) fn optimized_path_summary(config: &TrainConfig) -> String {
    let shard_input = config.bc_shards_manifest_path.is_some();
    let pinned_staging = cfg!(feature = "cuda-graph") && shard_input;
    let preallocated_tensors = pinned_staging;
    let copy_compute_overlap = if pinned_staging {
        "unproven-single-buffer"
    } else {
        "off"
    };
    format!(
        "input={} pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay={} copy_compute_overlap={}",
        if shard_input {
            "bc_shards"
        } else {
            "raw_replay"
        },
        if pinned_staging { "on" } else { "off" },
        if preallocated_tensors { "on" } else { "off" },
        cuda_graph_replay_label(),
        copy_compute_overlap,
    )
}

pub(super) fn explicit_preflight_recommendation() -> String {
    "using config runtime except epoch-boundary selected-runtime reuse; run train <config.yaml> --preflight to tune this machine before training"
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
    print_banner_field("Optimized path", optimized_path_summary(config).yellow());
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
    use super::{
        bc_hyperparam_summary, explicit_preflight_recommendation, explicit_preflight_summary,
        format_advisory_line, format_preflight_selection_line, format_preflight_summary_line,
        format_probe_progress_line, format_probe_results_table, format_probe_spinner_message,
        format_probe_status_line, format_progress_message, format_runtime_tuning_message,
        format_status_line, format_timed_phase_message, format_warning_line, make_bar,
        make_spinner, model_kind, optimized_path_summary, parse_probe_progress_fields, phase_label,
        preflight_phase_label, probe_failure_reason, probe_status_label, timestamped,
        with_utc_timestamp,
    };
    use hydra_train::model::HydraModelConfig;
    use hydra_train::preflight::{
        EffectiveRuntimeConfig, ExplicitSettings, LoaderRuntimeConfig, ProbeKind, ProbeResult,
        ProbeStatus, SelectedRuntimeConfig,
    };
    use hydra_train::training::bc::BCTrainerConfig;

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
    fn optimized_path_summary_reports_raw_replay_defaults() {
        let mut config = crate::test_support::dummy_train_config();
        config.bc_shards_manifest_path = None;

        assert_eq!(
            optimized_path_summary(&config),
            "input=raw_replay pinned_h2d=off prealloc_gpu_tensors=off cuda_graph_replay=production_off_probe_only copy_compute_overlap=off"
        );
    }

    #[test]
    fn optimized_path_summary_reports_shard_path() {
        let mut config = crate::test_support::dummy_train_config();
        config.bc_shards_manifest_path = Some(std::path::PathBuf::from("/shards/manifest.json"));

        let summary = optimized_path_summary(&config);
        assert!(summary.contains("input=bc_shards"));
        assert!(summary.contains("cuda_graph_replay=production_off_probe_only"));
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
    fn model_kind_distinguishes_actor_and_learner_configs() {
        assert_eq!(model_kind(&HydraModelConfig::actor()), "actor");
        assert_eq!(model_kind(&HydraModelConfig::learner()), "learner");
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
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_start",
            )
            .expect("scan_start should render"),
        );
        assert!(
            scan_start.contains("[preflight:train] candidate_mb=64 phase=scan dataset=streaming")
        );

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
        assert!(
            rl_selfplay.contains("phase=rl_selfplay running cooperative self-play + learner step")
        );

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
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_model",
            )
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
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_loss",
            )
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
        let model = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_model",
        )
        .expect("spinner init_model");
        assert!(model.contains("initializing model (backbone + heads)"));

        let optimizer = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_optimizer",
        )
        .expect("spinner init_optimizer");
        assert!(optimizer.contains("creating optimizer"));

        let loss = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_loss",
        )
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

        let starting = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=starting",
        )
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
    fn explicit_preflight_helpers_render_saved_runtime_and_recommendation() {
        let summary = explicit_preflight_summary(
            EffectiveRuntimeConfig {
                selected: SelectedRuntimeConfig {
                    train_microbatch_size: 64,
                    validation_microbatch_size: 32,
                    accum_steps: 4,
                },
                loader: LoaderRuntimeConfig {
                    num_threads: Some(6),
                    buffer_games: 128,
                    buffer_samples: 4096,
                    archive_queue_bound: 16,
                },
            },
            ExplicitSettings {
                train_microbatch_explicit: true,
                validation_microbatch_explicit: false,
            },
        );
        assert_eq!(
            summary,
            "saved train_mb=64 val_mb=32 accum_steps=4 threads=6 buffer_games=128 buffer_samples=4096 archive_queue_bound=16 explicit(train=true, val=false)"
        );
        assert_eq!(
            explicit_preflight_recommendation(),
            "using config runtime except epoch-boundary selected-runtime reuse; run train <config.yaml> --preflight to tune this machine before training"
        );
    }
}
