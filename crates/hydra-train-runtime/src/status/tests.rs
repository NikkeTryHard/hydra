use super::*;

struct Manifest {
    train_count: usize,
    counts_exact: bool,
}

impl EpochProgressManifest for Manifest {
    fn train_count(&self) -> usize {
        self.train_count
    }

    fn counts_exact(&self) -> bool {
        self.counts_exact
    }
}

fn manifest(train_count: usize, counts_exact: bool) -> Manifest {
    Manifest {
        train_count,
        counts_exact,
    }
}

#[test]
fn session_budget_and_labels_use_session_relative_steps() {
    assert_eq!(session_steps_completed(17, 10), 7);
    assert_eq!(session_steps_completed(4, 10), 0);
    assert!(reached_session_step_budget(17, 10, Some(7)));
    assert!(!reached_session_step_budget(16, 10, Some(7)));
    assert!(!reached_session_step_budget(16, 10, None));

    assert_eq!(display_step_label(17, 10, Some(20)), "step 7/20 global=17");
    assert_eq!(display_step_label(17, 10, None), "step 17");
    assert_eq!(
        display_validation_scope_label(17, 10, Some(20)),
        "validation @ step 7/20 global=17"
    );
    assert_eq!(
        display_validation_scope_label(17, 10, None),
        "validation @ step 17"
    );
}

#[test]
fn estimate_epoch_progress_requires_exact_counts_and_seen_games() {
    assert!(estimate_epoch_progress(&manifest(100, false), 40, 10, 3, 16).is_none());
    assert!(estimate_epoch_progress(&manifest(100, true), 40, 0, 3, 16).is_none());
}

#[test]
fn estimate_epoch_progress_projects_remaining_steps() {
    let progress = estimate_epoch_progress(&manifest(120, true), 30, 10, 3, 16)
        .expect("exact counts should produce an estimate");

    assert_eq!(progress.completed_optimizer_steps, 3);
    assert_eq!(progress.estimated_total_optimizer_steps, 23);
    assert_eq!(progress.estimated_remaining_optimizer_steps, 20);
    assert!((progress.completion_fraction - (3.0 / 23.0)).abs() < 1e-12);
}

#[test]
fn rough_duration_formats_seconds_minutes_and_hours() {
    assert_eq!(format_rough_duration(-3.2), "~0s");
    assert_eq!(format_rough_duration(59.4), "~59s");
    assert_eq!(format_rough_duration(61.0), "~1m1s");
    assert_eq!(format_rough_duration(3661.0), "~1h1m");
}

#[test]
fn epoch_progress_message_handles_missing_zero_and_positive_rates() {
    assert_eq!(
        epoch_progress_message_with_rate(None, Some(4.0)),
        "epoch=pending"
    );

    let progress = EpochProgressEstimate {
        completed_optimizer_steps: 2,
        estimated_total_optimizer_steps: 8,
        estimated_remaining_optimizer_steps: 6,
        completion_fraction: 0.25,
    };
    assert_eq!(
        epoch_progress_message_with_rate(Some(progress), Some(0.0)),
        "epoch=25.0% epoch_left≈6 steps"
    );
    assert_eq!(
        epoch_progress_message_with_rate(Some(progress), Some(2.0)),
        "epoch=25.0% epoch_left≈6 steps rough_eta=~3s"
    );
}

#[test]
fn optimizer_steps_rounds_up_and_guards_zero_batch() {
    assert_eq!(optimizer_steps_for_samples(0, 32), 0);
    assert_eq!(optimizer_steps_for_samples(1, 32), 1);
    assert_eq!(optimizer_steps_for_samples(33, 32), 2);
    assert_eq!(optimizer_steps_for_samples(7, 0), 7);
}
