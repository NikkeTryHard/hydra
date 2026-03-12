use colored::Colorize;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use hydra_train::model::HydraModelConfig;

use super::artifacts::BcArtifactPaths;
use super::config::TrainConfig;
use super::progress::BannerStats;

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

pub(super) fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &BannerStats,
    scheduler_warmup_steps: usize,
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
        "Epochs:".white(),
        config.num_epochs.to_string().yellow()
    );
    println!(
        "  {} {}",
        "Schedule:".white(),
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            scheduler_warmup_steps,
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
