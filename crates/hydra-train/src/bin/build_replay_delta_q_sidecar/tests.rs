use super::validate_source_version;

#[test]
fn delta_q_source_version_one_is_accepted() {
    validate_source_version(1).expect("source-version 1 should be accepted");
}

#[test]
fn delta_q_source_version_other_values_are_rejected() {
    let err = validate_source_version(2).expect_err("non-1 source-version should fail");
    assert!(err.contains("source-version 1"));
}
