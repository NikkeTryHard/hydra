use colored::Colorize;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{
    ExplicitSettings, ProbeKind, ProbeResult, ProbeStatus, SelectedRuntimeConfig,
};

use super::artifacts::BcArtifactPaths;
use super::config::TrainConfig;
use super::progress::BannerStats;
use hydra_train::training::bc::BCTrainerConfig;

pub(super) fn make_bar(len: u64, template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new(len);
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build progress style: {err}"))?
        .progress_chars("=> ");
    pb.set_style(style);
    Ok(pb)
}

pub(super) fn make_spinner(template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build spinner style: {err}"))?
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    pb.set_style(style);
    pb.enable_steady_tick(Duration::from_millis(120));
    Ok(pb)
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

fn probe_status_label(status: &ProbeStatus) -> &'static str {
    match status {
        ProbeStatus::Success => "success",
        ProbeStatus::Oom => "oom",
        ProbeStatus::BackendError => "backend_error",
        ProbeStatus::DataError => "data_error",
    }
}

pub(super) fn format_probe_status_line(result: &ProbeResult) -> String {
    match result.status {
        ProbeStatus::Success => format!(
            "[{}] candidate_mb={} status=success throughput={:.2} samples/s",
            match result.kind {
                ProbeKind::Train => "train",
                ProbeKind::Validation => "validation",
            },
            result.candidate_microbatch,
            result.measured_samples_per_second.unwrap_or(0.0)
        ),
        _ => format!(
            "[{}] candidate_mb={} status={}",
            match result.kind {
                ProbeKind::Train => "train",
                ProbeKind::Validation => "validation",
            },
            result.candidate_microbatch,
            probe_status_label(&result.status)
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
    let mut lines = vec![format!(
        "kind         selected  candidate_mb  status         throughput(samples/s)"
    )];
    lines.push(
        "------------ ---------  ------------  -------------  ---------------------".to_string(),
    );
    for result in results {
        let selected = if selected_candidate == Some(result.candidate_microbatch) {
            "yes"
        } else {
            "no"
        };
        let throughput = result
            .measured_samples_per_second
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "{kind_label:<12} {selected:<9} {candidate:<12} {status:<13} {throughput:>21}",
            candidate = result.candidate_microbatch,
            status = probe_status_label(&result.status),
        ));
    }
    lines.join("\n")
}

pub(super) fn explicit_preflight_summary(
    selected: SelectedRuntimeConfig,
    explicit: ExplicitSettings,
) -> String {
    format!(
        "saved train_mb={} val_mb={} accum_steps={} explicit(train={}, val={})",
        selected.train_microbatch_size,
        selected.validation_microbatch_size,
        selected.accum_steps,
        explicit.train_microbatch_explicit,
        explicit.validation_microbatch_explicit,
    )
}

pub(super) fn manual_runtime_warning(
    configured: SelectedRuntimeConfig,
    saved: SelectedRuntimeConfig,
) -> String {
    format!(
        "manual runtime values in use; configured train_mb={} val_mb={} accum_steps={} instead of saved preflight train_mb={} val_mb={} accum_steps={}",
        configured.train_microbatch_size,
        configured.validation_microbatch_size,
        configured.accum_steps,
        saved.train_microbatch_size,
        saved.validation_microbatch_size,
        saved.accum_steps,
    )
}

pub(super) fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &BannerStats,
    train_cfg: &BCTrainerConfig,
) {
    println!();
    println!();
    println!("{}", "Hydra BC trainer".bold().cyan());
    println!(
        "  {} {}",
        "Model:".white(),
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green()
    );
    println!("  {} {}", "Device:".white(), device_name.green());
    println!(
        "  {} {}",
        "Dataset:".white(),
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
        .green()
    );
    println!(
        "  {} {}",
        "Train:".white(),
        if stats.counts_exact {
            format!(
                "{} games | Val: {} games",
                stats.train_count, stats.val_count
            )
        } else {
            "streaming split, counts estimated while loading".to_string()
        }
        .green()
    );
    println!(
        "  {} {}",
        "Buffer:".white(),
        format!(
            "{} samples (max {} games)",
            config.buffer_samples, config.buffer_games
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Optimizer batch:".white(),
        format!(
            "{} ({} x {} accum)",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "BC hyperparams:".white(),
        bc_hyperparam_summary(train_cfg).yellow()
    );
    println!(
        "  {} {}",
        "Epochs:".white(),
        config.num_epochs.to_string().yellow()
    );
    println!(
        "  {} {}",
        "Schedule:".white(),
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            train_cfg.warmup_steps,
            config
                .max_train_steps
                .map(|steps| steps.to_string())
                .unwrap_or_else(|| "epoch-derived".to_string())
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Output:".white(),
        artifacts.root.display().to_string().green()
    );
    println!(
        "  {} {}",
        "TBoard:".white(),
        if config.tensorboard {
            artifacts.tb_session_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        }
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
