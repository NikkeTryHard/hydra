use super::*;

#[test]
fn parse_consumed_tiles_rejects_too_many_and_invalid_tiles() {
    let too_many = vec![
        "1m".to_string(),
        "2m".to_string(),
        "3m".to_string(),
        "4m".to_string(),
        "5m".to_string(),
    ];
    let err = parse_consumed_tiles(&too_many).expect_err("too many consumed tiles should fail");
    assert!(matches!(err, RiichiError::InvalidAction { .. }));

    let invalid = vec!["1m".to_string(), "bad".to_string()];
    let err = parse_consumed_tiles(&invalid).expect_err("invalid consumed tile should fail");
    assert!(matches!(err, RiichiError::Parse { .. }));
}
