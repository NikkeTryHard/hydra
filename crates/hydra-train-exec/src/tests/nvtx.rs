use crate::nvtx::{scope, with_test_recorder};

#[test]
fn scope_without_test_recorder_is_noop() {
    let _guard = scope("train");
}

#[test]
fn scope_records_nested_push_pop_order() {
    let (_, events) = with_test_recorder(|| {
        let _outer = scope("bc_epoch");
        {
            let _inner = scope("validation");
        }
    });

    assert_eq!(
        events,
        vec![
            "push:bc_epoch".to_string(),
            "push:validation".to_string(),
            "pop:validation".to_string(),
            "pop:bc_epoch".to_string(),
        ]
    );
}
