use super::*;

#[test]
fn checkpoint_meta_summary_handles_missing_eval_metrics() {
    let meta = CheckpointMeta::new(10, 2.5, None, None, None);
    assert_eq!(meta.summary(), "epoch=10 loss=2.5000 eval=n/a");
}

#[test]
fn checkpoint_meta_summary_reports_eval_metrics() {
    let meta = CheckpointMeta::new(10, 2.5, Some(0.375), Some(1.75), Some(2.25));
    assert_eq!(
        meta.summary(),
        "epoch=10 loss=2.5000 policy_ce=1.7500 total=2.2500 agree=37.50%"
    );
}
