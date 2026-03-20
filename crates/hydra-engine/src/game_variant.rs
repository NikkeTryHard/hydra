use crate::rule::GameRule;
use crate::state::GameState;
use crate::state_3p::GameState3P;

#[derive(Debug, Clone)]
pub enum GameStateVariant {
    FourPlayer(Box<GameState>),
    ThreePlayer(Box<GameState3P>),
}

impl GameStateVariant {
    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        if game_mode >= 3 {
            GameStateVariant::ThreePlayer(Box::new(GameState3P::new(
                game_mode,
                skip_mjai_logging,
                seed,
                round_wind,
                rule,
            )))
        } else {
            GameStateVariant::FourPlayer(Box::new(GameState::new(
                game_mode,
                skip_mjai_logging,
                seed,
                round_wind,
                rule,
            )))
        }
    }

    pub fn num_players(&self) -> u8 {
        match self {
            GameStateVariant::FourPlayer(s) => s.mode.num_players(),
            GameStateVariant::ThreePlayer(_) => 3,
        }
    }

    pub fn is_three_player(&self) -> bool {
        matches!(self, GameStateVariant::ThreePlayer(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_uses_four_player_state_for_standard_modes() {
        let variant = GameStateVariant::new(0, true, Some(7), 0, GameRule::default_tenhou());
        assert!(matches!(variant, GameStateVariant::FourPlayer(_)));
        assert_eq!(variant.num_players(), 4);
        assert!(!variant.is_three_player());
    }

    #[test]
    fn new_uses_three_player_state_for_sanma_modes() {
        let variant = GameStateVariant::new(3, true, Some(11), 0, GameRule::default_tenhou_sanma());
        assert!(matches!(variant, GameStateVariant::ThreePlayer(_)));
        assert_eq!(variant.num_players(), 3);
        assert!(variant.is_three_player());
    }
}
