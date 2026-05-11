use super::*;

#[test]
fn display_messages_include_error_context() {
    let parse = RiichiError::Parse {
        input: "123x".to_string(),
        message: "bad suit".to_string(),
    };
    assert_eq!(parse.to_string(), "Parse error on '123x': bad suit");

    let invalid_action = RiichiError::InvalidAction {
        message: "missing tile".to_string(),
    };
    assert_eq!(invalid_action.to_string(), "Invalid action: missing tile");

    let invalid_state = RiichiError::InvalidState {
        message: "desynced replay".to_string(),
    };
    assert_eq!(invalid_state.to_string(), "Invalid state: desynced replay");

    let serialization = RiichiError::Serialization {
        message: "json failed".to_string(),
    };
    assert_eq!(
        serialization.to_string(),
        "Serialization error: json failed"
    );
}
