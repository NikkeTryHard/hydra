use hydra_train::data::pipeline::DataManifest;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EpochProgressEstimate {
    pub(super) completed_optimizer_steps: usize,
    pub(super) estimated_total_optimizer_steps: usize,
    pub(super) estimated_remaining_optimizer_steps: usize,
    pub(super) completion_fraction: f64,
}

pub(super) fn session_steps_completed(
    global_step: usize,
    session_start_global_step: usize,
) -> usize {
    global_step.saturating_sub(session_start_global_step)
}

pub(super) fn reached_session_step_budget(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> bool {
    max_train_steps
        .map(|budget| session_steps_completed(global_step, session_start_global_step) >= budget)
        .unwrap_or(false)
}

pub(super) fn display_step_label(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> String {
    let session_step = session_steps_completed(global_step, session_start_global_step);
    if let Some(total) = max_train_steps {
        format!("step {session_step}/{total} global={global_step}")
    } else {
        format!("step {global_step}")
    }
}

pub(super) fn display_validation_scope_label(
    global_step: usize,
    session_start_global_step: usize,
    max_train_steps: Option<usize>,
) -> String {
    let session_step = session_steps_completed(global_step, session_start_global_step);
    match max_train_steps {
        Some(total) => format!("validation @ step {session_step}/{total} global={global_step}"),
        None => format!("validation @ step {global_step}"),
    }
}

pub(super) fn estimate_epoch_progress(
    manifest: &DataManifest,
    seen_samples: usize,
    assumed_games_seen: usize,
    epoch_optimizer_steps: usize,
    batch_size: usize,
) -> Option<EpochProgressEstimate> {
    if !manifest.counts_exact || assumed_games_seen == 0 {
        return None;
    }
    let estimated_total_samples =
        seen_samples.saturating_mul(manifest.train_count) / assumed_games_seen.max(1);
    let estimated_total_optimizer_steps =
        optimizer_steps_for_samples(estimated_total_samples, batch_size)
            .max(epoch_optimizer_steps)
            .max(1);
    let estimated_remaining_optimizer_steps =
        estimated_total_optimizer_steps.saturating_sub(epoch_optimizer_steps);
    Some(EpochProgressEstimate {
        completed_optimizer_steps: epoch_optimizer_steps,
        estimated_total_optimizer_steps,
        estimated_remaining_optimizer_steps,
        completion_fraction: epoch_optimizer_steps as f64 / estimated_total_optimizer_steps as f64,
    })
}

pub(super) fn format_rough_duration(seconds: f64) -> String {
    let rounded = seconds.max(0.0).round() as u64;
    let hours = rounded / 3600;
    let minutes = (rounded % 3600) / 60;
    let secs = rounded % 60;
    if hours > 0 {
        format!("~{}h{}m", hours, minutes)
    } else if minutes > 0 {
        format!("~{}m{}s", minutes, secs)
    } else {
        format!("~{}s", secs)
    }
}

pub(super) fn epoch_progress_message_with_rate(
    progress: Option<EpochProgressEstimate>,
    step_rate: Option<f64>,
) -> String {
    match progress {
        Some(progress) => {
            let eta = step_rate
                .filter(|rate| *rate > 0.0)
                .map(|rate| {
                    format!(
                        " rough_eta={}",
                        format_rough_duration(
                            progress.estimated_remaining_optimizer_steps as f64 / rate
                        )
                    )
                })
                .unwrap_or_default();
            format!(
                "epoch={:.1}% epoch_left≈{} steps{}",
                progress.completion_fraction * 100.0,
                progress.estimated_remaining_optimizer_steps,
                eta,
            )
        }
        None => "epoch=pending".to_string(),
    }
}

fn optimizer_steps_for_samples(samples: usize, batch_size: usize) -> usize {
    if samples == 0 {
        0
    } else {
        samples.div_ceil(batch_size.max(1))
    }
}
