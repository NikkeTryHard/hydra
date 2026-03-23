use std::fmt;

#[derive(Debug)]
pub enum RiichiError {
    /// 牌文字列・手牌文字列のパースエラー
    Parse { input: String, message: String },
    /// アクション構成・エンコードのバリデーションエラー
    InvalidAction { message: String },
    /// ゲーム状態の不整合（リプレイ同期ずれ等）
    InvalidState { message: String },
    /// シリアライズ/デシリアライズの失敗
    Serialization { message: String },
}

impl fmt::Display for RiichiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiichiError::Parse { input, message } => {
                write!(f, "Parse error on '{}': {}", input, message)
            }
            RiichiError::InvalidAction { message } => {
                write!(f, "Invalid action: {}", message)
            }
            RiichiError::InvalidState { message } => {
                write!(f, "Invalid state: {}", message)
            }
            RiichiError::Serialization { message } => {
                write!(f, "Serialization error: {}", message)
            }
        }
    }
}

impl std::error::Error for RiichiError {}

pub type RiichiResult<T> = Result<T, RiichiError>;

#[cfg(test)]
mod tests {
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
}
