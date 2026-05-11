use super::run;

#[test]
fn run_reports_usage_without_required_args() {
    let err = run().expect_err("run without args should fail under test harness");
    assert!(err.contains("Usage:"));
}
