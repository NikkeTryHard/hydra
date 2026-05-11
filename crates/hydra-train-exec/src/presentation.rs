//! Pure presentation formatting helpers shared by train execution seams.

#![allow(
    missing_docs,
    reason = "moved train execution support preserves existing public surface"
)]

use std::borrow::Cow;
use std::time::Duration;

use colored::Colorize;
use hydra_train_runtime::config::display_num_threads;
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, ProbeKind, ProbeResult, ProbeStatus,
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::advisory::{AdvisorySeverity, RuntimeAdvisory};
use crate::probe_summary::probe_summary_iter;

/// Scalar input for rendering BC optimizer hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BcHyperparamSummaryInput {
    /// Base learning rate.
    pub lr: f64,
    /// Minimum cosine-decay learning rate.
    pub min_learning_rate: f64,
    /// Optimizer weight decay.
    pub weight_decay: f64,
    /// Gradient clipping norm.
    pub grad_clip_norm: f64,
    /// Warmup steps before cosine schedule.
    pub warmup_steps: usize,
}

/// Builds an indicatif progress bar using the training CLI defaults.
pub fn make_bar(len: u64, template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new(len);
    pb.set_draw_target(ProgressDrawTarget::stdout());
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build progress style: {err}"))?
        .progress_chars("=> ");
    pb.set_style(style);
    Ok(pb)
}

/// Builds an indicatif spinner using the training CLI defaults.
pub fn make_spinner(template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new_spinner();
    pb.set_draw_target(ProgressDrawTarget::stdout());
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build spinner style: {err}"))?
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    pb.set_style(style);
    pb.enable_steady_tick(Duration::from_millis(120));
    Ok(pb)
}

/// Formats the preflight phase label used in progress bars.
pub fn preflight_phase_label(phase: &str) -> String {
    format!("preflight {phase}")
}

fn utc_log_prefix() -> String {
    let ts = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
    format!("[{ts}]")
}

/// Prefixes a message with a dimmed UTC timestamp.
pub fn with_utc_timestamp(message: String) -> String {
    format!("{} {}", utc_log_prefix().dimmed(), message)
}

/// Formats any displayable value with a UTC timestamp.
pub fn timestamped(message: impl std::fmt::Display) -> String {
    with_utc_timestamp(message.to_string())
}

/// Formats a runtime tuning progress line.
pub fn format_runtime_tuning_message(
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

/// Formats a timed preflight phase line.
pub fn format_timed_phase_message(phase: &str, detail: &str, elapsed_seconds: f64) -> String {
    with_utc_timestamp(format!(
        "{} {} {}",
        "[preflight:timing]".bold().cyan(),
        format!("phase={phase}").yellow(),
        format!("{detail} elapsed={elapsed_seconds:.2}s").green(),
    ))
}

/// Formats a preflight summary line.
pub fn format_preflight_summary_line(label: &str, detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        label.bold().cyan(),
        detail.to_string().yellow()
    ))
}

/// Formats a preflight selection line.
pub fn format_preflight_selection_line(detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        "Preflight selected:".bold().cyan(),
        detail.to_string().green()
    ))
}

/// Formats a generic status line.
pub fn format_status_line(label: &str, detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        label.bold().cyan(),
        detail.to_string().yellow()
    ))
}

/// Formats a warning line.
pub fn format_warning_line(detail: impl std::fmt::Display) -> String {
    with_utc_timestamp(format!(
        "{} {}",
        "Warning:".bold().yellow(),
        detail.to_string().yellow()
    ))
}

/// Formats a runtime advisory line.
pub fn format_advisory_line(advisory: &RuntimeAdvisory) -> String {
    let severity = match advisory.severity {
        AdvisorySeverity::Info => "Info".bold().cyan(),
        AdvisorySeverity::Warning => "Warning".bold().yellow(),
    };
    let detail = format!("{}: {}", advisory.key, advisory.message);
    with_utc_timestamp(format!("{} {}", severity, detail.yellow()))
}

/// Formats an epoch/phase label.
pub fn phase_label(prefix: &str, epoch_index: usize, num_epochs: usize) -> String {
    if num_epochs <= 1 {
        prefix.to_string()
    } else {
        format!("{prefix} {}/{}", epoch_index + 1, num_epochs)
    }
}

/// Formats a train progress message.
pub fn format_progress_message(
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

/// Formats scalar BC optimizer hyperparameters.
pub fn bc_hyperparam_summary(input: BcHyperparamSummaryInput) -> String {
    format!(
        "lr={:.2e} min_lr={:.2e} wd={:.1e} clip={:.2} warmup_steps={}",
        input.lr,
        input.min_learning_rate,
        input.weight_decay,
        input.grad_clip_norm,
        input.warmup_steps,
    )
}

/// Prints a blank-spaced section header.
pub fn print_header_block(title: &str) {
    println!();
    println!();
    println!("{}", title.bold().cyan());
}

/// Prints one banner field.
pub fn print_banner_field(label: &str, value: impl std::fmt::Display) {
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

/// Formats a probe progress line for a spinner message.
pub fn format_probe_spinner_message(line: &str) -> Option<String> {
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

/// Formats the finish message for a probe spinner.
#[cfg(not(test))]
pub fn format_probe_spinner_finish_message(
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

/// Formats a machine probe progress line for terminal output.
pub fn format_probe_progress_line(line: &str) -> Option<String> {
    let event = ProbeProgressEvent::parse(line)?;
    Some(with_utc_timestamp(event.progress_message()))
}

/// Formats a probe status line.
pub fn format_probe_status_line(result: &ProbeResult) -> String {
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

/// Formats an aggregate probe results table.
pub fn format_probe_results_table(
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

/// Formats an explicit preflight summary.
pub fn explicit_preflight_summary(
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

/// Returns the CUDA graph replay status label.
pub fn cuda_graph_replay_label() -> &'static str {
    "production_off_probe_only"
}

/// Formats the explicit-preflight recommendation.
pub fn explicit_preflight_recommendation() -> String {
    "using config runtime except epoch-boundary selected-runtime reuse; run train <config.yaml> --preflight to tune this machine before training"
        .to_string()
}

#[cfg(test)]
mod tests;
