use super::*;

#[test]
fn pipeline_state_defaults() {
    let state = PipelineState::default();
    assert_eq!(state.phase, TrainingPhase::BenchmarkGates);
    assert!((state.remaining_budget() - 2000.0).abs() < 0.01);
}

#[test]
fn budget_and_progress_are_clamped() {
    let mut state = PipelineState {
        gpu_hours_used: -10.0,
        ..PipelineState::default()
    };
    assert_eq!(state.remaining_budget(), PipelineState::total_budget());
    assert_eq!(state.overall_progress(), 0.0);

    state.tick_gpu_hours(f32::NAN);
    assert_eq!(state.gpu_hours_used, -10.0);
    state.tick_gpu_hours(3_000.0);
    assert_eq!(state.gpu_hours_used, PipelineState::total_budget());
    assert_eq!(state.remaining_budget(), 0.0);
    assert_eq!(state.overall_progress(), 1.0);

    state.tick_gpu_hours(-3_000.0);
    assert_eq!(state.gpu_hours_used, 0.0);
}

#[test]
fn stage_advancement() {
    let mut state = PipelineState::default();
    state.advance_phase();
    assert_eq!(state.phase, TrainingPhase::BcWarmStart);
    state.advance_phase();
    assert_eq!(state.phase, TrainingPhase::OracleGuiding);
}

#[test]
fn stage_budgets_are_cumulative() {
    assert_eq!(TrainingPhase::BenchmarkGates.cumulative_budget_before(), 0);
    assert_eq!(TrainingPhase::BcWarmStart.cumulative_budget_before(), 150);
    assert_eq!(TrainingPhase::OracleGuiding.cumulative_budget_before(), 200);
    assert_eq!(
        TrainingPhase::DrdaAchSelfPlay.cumulative_budget_before(),
        400
    );
    assert_eq!(
        TrainingPhase::ExitPondering.cumulative_budget_before(),
        1200
    );
    assert_eq!(
        TrainingPhase::ExitPondering.cumulative_budget_through(),
        2000
    );
}

#[test]
fn stage_progress_uses_stage_local_hours() {
    let state = PipelineState {
        phase: TrainingPhase::DrdaAchSelfPlay,
        gpu_hours_used: 600.0,
        ..PipelineState::default()
    };
    assert!((state.stage_hours_used() - 200.0).abs() < 1e-6);
    assert!((state.stage_progress() - 0.25).abs() < 1e-6);
}

#[test]
fn stage_advance_requires_cumulative_budget() {
    let state = PipelineState {
        phase: TrainingPhase::OracleGuiding,
        gpu_hours_used: 250.0,
        ..PipelineState::default()
    };
    assert!(!state.should_advance_phase());

    let ready = PipelineState {
        phase: TrainingPhase::OracleGuiding,
        gpu_hours_used: 400.0,
        ..PipelineState::default()
    };
    assert!(ready.should_advance_phase());
}

#[test]
fn exit_schedule_stage_matches_hydra_final_rollout_plan() {
    assert_eq!(TrainingPhase::BcWarmStart.exit_schedule_phase(), 1);
    assert_eq!(TrainingPhase::OracleGuiding.exit_schedule_phase(), 1);
    assert_eq!(TrainingPhase::DrdaAchSelfPlay.exit_schedule_phase(), 2);
    assert_eq!(TrainingPhase::ExitPondering.exit_schedule_phase(), 3);
}

#[test]
fn stage_uses_exit_oracle() {
    assert!(!TrainingPhase::BcWarmStart.uses_exit());
    assert!(TrainingPhase::ExitPondering.uses_exit());
    assert!(!TrainingPhase::BcWarmStart.uses_oracle());
    assert!(TrainingPhase::OracleGuiding.uses_oracle());
}
